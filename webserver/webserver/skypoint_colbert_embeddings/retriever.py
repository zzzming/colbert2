import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

import torch
import re
from cassandra.cluster import Session
from langchain.docstore.document import Document
from pydantic import BaseModel
from torch import tensor
from urllib.parse import unquote

from .astra_db import AstraDB
from .colbert_token_embedding import (  # type: ignore
    ColbertTokenEmbeddings,
    get_colbert_embeddings,
)

# max similarity between a query vector and a list of embeddings
# The function returns the highest similarity score (i.e., the maximum dot product value) between the query vector and any of the embedding vectors in the list.

"""
# The function iterates over each embedding vector (e) in the embeddings.
# For each e, it performs a dot product operation (@) with the query vector (qv).
# The dot product of two vectors is a measure of their similarity. In the context of embeddings,
# a higher dot product value usually indicates greater similarity.
# The max function then takes the highest value from these dot product operations.
# Essentially, it's picking the embedding vector that has the highest similarity to the query vector qv.
def max_similary_operator_based(qv, embeddings, is_cuda: bool=False):
    if is_cuda:
        # Assuming qv and embeddings are PyTorch tensors
        qv = qv.to('cuda')  # Move qv to GPU
        embeddings = [e.to('cuda') for e in embeddings]  # Move all embeddings to GPU
    return max(qv @ e for e in embeddings)
def max_similarity_numpy_based(query_vector, embedding_list):
    # Convert the list of embeddings into a numpy matrix for vectorized operation
    embedding_matrix = np.vstack(embedding_list)

    # Calculate the dot products in a vectorized manner
    sims = np.dot(embedding_matrix, query_vector)

    # Find the maximum similarity (dot product) value
    max_sim = np.max(sims)

    return max_sim
"""


# this torch based max similarly has the best performance.
# it is at least 20 times faster than dot product operator and numpy based implementation CuDA and CPU
def max_similarity_torch(
    query_vector: Any, embedding_list: List[Any], is_cuda: bool = False
) -> Any:
    """
    Calculate the maximum similarity (dot product) between a query vector and a list of embedding vectors,
    optimized for performance using PyTorch for GPU acceleration.

    Parameters:
    - query_vector: A PyTorch tensor representing the query vector.
    - embedding_list: A list of PyTorch tensors, each representing an embedding vector.

    Returns:
    - max_sim: A float representing the highest similarity (dot product) score between the query vector and the embedding vectors in the list, computed on the GPU.
    """
    # stacks the list of embedding tensors into a single tensor
    if is_cuda:
        query_vector = query_vector.to("cuda")
        embedding_list = torch.stack(embedding_list).to("cuda")
    else:
        embedding_list = torch.stack(embedding_list)

    # Calculate the dot products in a vectorized manner on the GPU
    sims = torch.matmul(embedding_list, query_vector)

    # Find the maximum similarity (dot product) value
    max_sim = torch.max(sims)

    # returns a tensor; the item() is the score
    return max_sim


class ColbertAstraRetriever:
    """
    A retriever class that uses Colbert embeddings and AstraDB for document retrieval.
    It has a method to retrieve documents for a single query and another method to retrieve documents for multiple queries concurrently.

    Attributes:
    - astra: An instance of AstraDB for connecting to AstraDB.
    - colbertEmbeddings: An instance of ColbertTokenEmbeddings for encoding queries.
    - verbose: A boolean flag to enable verbose logging.
    - is_cuda: A boolean flag indicating whether GPU acceleration is available.
    """
    astra: AstraDB
    colbertEmbeddings: ColbertTokenEmbeddings
    verbose: bool
    is_cuda: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        astraDB: AstraDB,
        colbertEmbeddings: ColbertTokenEmbeddings,
        verbose: bool = False,
        **kwargs: dict,
    ):
        self.astra = astraDB
        self.colbertEmbeddings = colbertEmbeddings
        self.verbose = verbose
        self.is_cuda = torch.cuda.is_available()

    def retrieve(
        self, query: str, k: int = 5, query_maxlen: int = -1, **kwargs: dict[Any, Any]
    ) -> List[dict]:
        """
        Retrieve the top k documents for a single query.
        """
        #
        # if the query has fewer than a predefined number of of tokens Nq,
        # colbertEmbeddings will pad it with BERT special [mast] token up to length Nq.
        #
        start_time = time.time()
        index = kwargs.get("index", 0)
        query_encodings = self.colbertEmbeddings.encode_query(
            query, query_maxlen=query_maxlen
        )
        top_k = max(math.floor(len(query_encodings) / 2), 16)
        logging.info("query_maxlen = %s", query_maxlen)
        logging.info(f"query length {len(query)} embeddings top_k: {top_k}")

        # find the most relevant documents
        docparts: set = set()
        doc_futures = []
        for qv in query_encodings:
            # per token based retrieval
            doc_future = self.astra.session.execute_async(
                self.astra.query_colbert_ann_stmt, [list(qv), top_k]
            )
            doc_futures.append(doc_future)

        for future in doc_futures:
            rows = future.result()
            docparts.update((row.title, row.part) for row in rows)
        # score each document
        scores = {}
        futures = []
        for title, part in docparts:
            future = self.astra.session.execute_async(
                self.astra.query_colbert_parts_stmt, [title, part]
            )
            futures.append((future, title, part))
        for future, title, part in futures:
            rows = future.result()
            # find all the found parts so that we can do max similarity search
            embeddings_for_part = [tensor(row.bert_embedding) for row in rows]
            # score based on The function returns the highest similarity score
            # (i.e., the maximum dot product value) between the query vector and any of the embedding vectors in the list.
            scores[(title, part)] = sum(
                max_similarity_torch(qv, embeddings_for_part, self.is_cuda)
                for qv in query_encodings
            )
        # load the source chunk for the top k documents
        docs_by_score = sorted(scores, key=scores.get, reverse=True)[:k]  # type: ignore
        # query the doc body
        doc_futures1: Any = {}
        for title, part in docs_by_score:
            future = self.astra.session.execute_async(
                self.astra.query_part_by_pk_stmt, [title, part]
            )
            doc_futures1[(title, part)] = future

        answers = []
        rank = 1
        for title, part in docs_by_score:
            rs = doc_futures1[(title, part)].result()
            score = scores[(title, part)]
            answers.append(
                {
                    "title": title,
                    "score": score.item(),
                    "rank": rank,
                    "body": rs.one().body,
                }
            )
            rank = rank + 1
        # clean up on tensor memory on GPU
        del scores
        logging.info(
            f"{index} Index Time taken to execute all astra queries = %s",
            time.time() - start_time,
        )

        return answers

    def retrieve_concurrently(
        self, queries: list, k: int = 5, query_maxlen: int = -1, **kwargs: dict
    ) -> List[dict]:
        """
        Execute multiple retrieve calls concurrently using threads.

        :param queries: A list of query strings.
        :param k: The number of top documents to retrieve for each query.
        :param query_maxlen: Maximum length of the query for encoding.
        :return: A list of answers from all queries combined.
        """
        answers = []

        # Use ThreadPoolExecutor to execute retrieve calls in parallel
        with ThreadPoolExecutor() as executor:
            # Create a future for each query
            futures = [
                executor.submit(
                    self.retrieve, query, k, query_maxlen, index=index, **kwargs  # type: ignore
                )
                for index, query in enumerate(queries)
            ]
            # As each future completes, extend the answers list with its result
            for future in as_completed(futures):
                answers.extend(future.result())

        return answers


def get_colbert_answer(
    queries: List[str],
    astra_session: Session,
    astra_keyspace: str,
    colbert_text_table: str,
    colbert_embedding_table: str,
    k : int = 5,
    query_maxlen : int = -1,
) -> List[Document]:
    """
    Retrieve documents from AstraDB for a list of queries.
    """
    start_time = time.time()
    astra = AstraDB(
        session=astra_session,
        keyspace=astra_keyspace,
        text_table=colbert_text_table,
        embedding_table=colbert_embedding_table,
    )
    colbert = get_colbert_embeddings()
    retriever = ColbertAstraRetriever(
        astraDB=astra, colbertEmbeddings=colbert, verbose=True
    )
    answers = retriever.retrieve_concurrently(queries ,k, query_maxlen)
    documents = []
    for a in answers:
        logging.info(f"Title: {a.get('title')},\n body: {a.get('body')}")
        source = unquote(a.get("title"))
        source = re.sub(r'_[0-9]+$', '', source)
        documents.append(
            Document(page_content=a.get("body"), metadata={"source": source})
        )
    logging.info(
        f"Time taken to retrieve documents from AstraDB = {time.time() - start_time}"
    )
    return documents
