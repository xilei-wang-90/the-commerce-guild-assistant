"""Retrieve relevant documents from ChromaDB based on query similarity.

Embeds a user query and finds the closest matching documents in the
vector store, returning structured results with paths to the full-text
silver-tier files for downstream context building.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from langsmith import traceable

from guild_assistant.utils.model_adapter import EmbeddingAdapter

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result from the vector store."""

    doc_id: str
    summary: str
    silver_path: str
    distance: float


class Retriever:
    """Embed a query and find the top-k matching documents in ChromaDB.

    *db_path*             – path to the persisted ChromaDB database.
    *collection_name*     – name of the collection to query.
    *embedding_adapter*   – adapter used to embed the query text.
    *n_results*           – number of results to return per query.
    """

    def __init__(
        self,
        db_path: str | Path,
        collection_name: str,
        embedding_adapter: EmbeddingAdapter,
        n_results: int = 3,
    ) -> None:
        self._adapter = embedding_adapter
        self._n_results = n_results

        client = chromadb.PersistentClient(path=str(Path(db_path)))
        self._collection = client.get_collection(name=collection_name)

    @traceable(run_type="retriever", name="ChromaDB Retrieval")
    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Embed *query* and return the closest matching documents."""
        vector = self._adapter.embed(query)
        raw = self._collection.query(
            query_embeddings=[vector],
            n_results=self._n_results,
        )

        results: list[RetrievalResult] = []
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc_id, summary, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    summary=summary,
                    silver_path=metadata.get("silver_path", ""),
                    distance=distance,
                )
            )

        logger.info("Retrieved %d results for query: %.60s…", len(results), query)
        return results
