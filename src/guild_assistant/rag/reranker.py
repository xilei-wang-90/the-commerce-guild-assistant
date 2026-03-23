"""Cross-encoder reranking of retrieval results.

Scores each retrieved document against the user query using a
reranker adapter, then returns the top-scoring results.  This
provides a more semantically-aware ranking than the initial
bi-encoder similarity search.
"""

import logging
from pathlib import Path

from langsmith import traceable

from guild_assistant.rag.retriever import RetrievalResult
from guild_assistant.utils.model_adapter import RerankerAdapter

logger = logging.getLogger(__name__)


class Reranker:
    """Rerank retrieval results using a reranker adapter.

    *adapter* – a ``RerankerAdapter`` that scores (query, document) pairs.
    *top_n*   – number of top-scoring results to return.
    """

    def __init__(
        self,
        adapter: RerankerAdapter,
        top_n: int = 3,
    ) -> None:
        self._adapter = adapter
        self._top_n = top_n

    @traceable(run_type="chain", name="Cross-Encoder Reranking")
    def rerank(
        self, query: str, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Score each result's silver document against *query* and return the top matches."""
        pairs: list[tuple[str, str]] = []
        valid_results: list[RetrievalResult] = []

        for result in results:
            path = Path(result.silver_path)
            if not path.exists():
                logger.warning("Silver file not found, skipping rerank: %s", path)
                continue
            content = path.read_text(encoding="utf-8")
            pairs.append((query, content))
            valid_results.append(result)

        if not pairs:
            return []

        scores = self._adapter.score(pairs)

        scored = sorted(
            zip(scores, valid_results), key=lambda x: x[0], reverse=True
        )
        top = [result for _, result in scored[: self._top_n]]

        logger.info(
            "Reranked %d results down to %d for query: %.60s…",
            len(valid_results),
            len(top),
            query,
        )
        return top
