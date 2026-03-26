"""Evaluate retrieval quality against a labelled test set.

Loads test cases from a CSV, runs each question through a
:class:`~guild_assistant.rag.retriever.Retriever`, and computes
requested metrics using the :mod:`tests.benchmark.metrics` registry.

Optionally accepts a :class:`~guild_assistant.rag.reranker.Reranker` to
evaluate the effect of cross-encoder reranking.  When a reranker is
provided, the evaluator fetches a larger initial result set
(``retrieve_n``), computes metrics on the raw retrieval, reranks to the
top *k*, and computes metrics again — allowing before/after comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path

from guild_assistant.rag.reranker import Reranker
from guild_assistant.rag.retriever import Retriever
from tests.benchmark.metrics import METRIC_REGISTRY, QueryResult
from tests.benchmark.testset_loader import TestCase

logger = logging.getLogger(__name__)

# Maps collection names to the TestCase field used for relevance matching.
_MATCH_FIELDS: dict[str, str] = {
    "sandrock_wiki_summary": "page",
    "sandrock_wiki_section_reverse_hyde": "section",
    "sandrock_wiki_section_tagged_reverse_hyde": "section",
}


class RetrievalEvaluator:
    """Run retrieval evaluation on a test set.

    *retriever*        – configured :class:`Retriever` instance.
    *collection_name*  – name of the ChromaDB collection being evaluated,
                         used to decide whether to match on page or section.
    *k*                – number of top results to consider per query.
    *reranker*         – optional :class:`Reranker` for cross-encoder reranking.
    *retrieve_n*       – number of results to fetch before reranking (only
                         used when *reranker* is set; defaults to 10).
    """

    def __init__(
        self,
        retriever: Retriever,
        collection_name: str,
        k: int = 5,
        reranker: Reranker | None = None,
        retrieve_n: int = 10,
    ) -> None:
        self._retriever = retriever
        self._collection_name = collection_name
        self._k = k
        self._reranker = reranker
        self._retrieve_n = retrieve_n

    def _expected_id(self, case: TestCase) -> str:
        """Return the filename to match against based on the collection."""
        field = _MATCH_FIELDS.get(self._collection_name, "page")
        return getattr(case, field)

    def run(
        self,
        test_cases: list[TestCase],
        metrics: list[str],
    ) -> tuple[dict[str, float], list[QueryResult]]:
        """Evaluate *test_cases* and return scores and per-query results.

        Each question in *test_cases* is sent to the retriever with *k*
        results requested.  The retrieved ``silver_path`` basenames are
        compared to the expected filename (page or section, depending on
        the collection).

        *metrics* is a list of metric names from :data:`METRIC_REGISTRY`.
        Unknown metric names are logged as warnings and skipped.

        Returns ``(scores, query_results)`` where *scores* is a dict of
        ``{metric: score}`` and *query_results* is the list of
        :class:`QueryResult` instances (one per test case).
        """
        query_results: list[QueryResult] = []

        for i, case in enumerate(test_cases, 1):
            retrieval_results = self._retriever.retrieve(case.question)
            retrieved_ids = [
                Path(r.silver_path).name for r in retrieval_results
            ]
            expected = self._expected_id(case)
            query_results.append(
                QueryResult(
                    question=case.question,
                    expected_id=expected,
                    retrieved_ids=retrieved_ids,
                )
            )
            hit = expected in retrieved_ids
            logger.info(
                "[%d/%d] %s — %s",
                i,
                len(test_cases),
                "HIT" if hit else "MISS",
                case.question[:80],
            )

        scores: dict[str, float] = {}
        for name in metrics:
            fn = METRIC_REGISTRY.get(name)
            if fn is None:
                logger.warning("Unknown metric '%s', skipping", name)
                continue
            scores[name] = fn(query_results)

        return scores, query_results

    def run_with_rerank(
        self,
        test_cases: list[TestCase],
        metrics: list[str],
    ) -> tuple[dict[str, float], dict[str, float], list[QueryResult], list[QueryResult]]:
        """Evaluate with before/after reranking comparison.

        Fetches ``retrieve_n`` results per query, computes metrics on the
        top *k* of those (before rerank), then reranks all ``retrieve_n``
        results and computes metrics on the top *k* after reranking.

        Requires a reranker to have been provided at construction time.

        Returns ``(before_scores, after_scores, before_results, after_results)``.
        """
        if self._reranker is None:
            raise ValueError("run_with_rerank requires a Reranker")

        before_results: list[QueryResult] = []
        after_results: list[QueryResult] = []

        for i, case in enumerate(test_cases, 1):
            retrieval_results = self._retriever.retrieve(case.question)
            expected = self._expected_id(case)

            # Before rerank: take the top-k from the raw retrieval.
            before_ids = [
                Path(r.silver_path).name for r in retrieval_results[: self._k]
            ]
            before_results.append(
                QueryResult(
                    question=case.question,
                    expected_id=expected,
                    retrieved_ids=before_ids,
                )
            )

            # After rerank: rerank all retrieve_n results, take top-k.
            reranked = self._reranker.rerank(case.question, retrieval_results)
            after_ids = [
                Path(r.silver_path).name for r in reranked[: self._k]
            ]
            after_results.append(
                QueryResult(
                    question=case.question,
                    expected_id=expected,
                    retrieved_ids=after_ids,
                )
            )

            before_hit = expected in before_ids
            after_hit = expected in after_ids
            logger.info(
                "[%d/%d] before=%s after=%s — %s",
                i,
                len(test_cases),
                "HIT" if before_hit else "MISS",
                "HIT" if after_hit else "MISS",
                case.question[:80],
            )

        before_scores: dict[str, float] = {}
        after_scores: dict[str, float] = {}
        for name in metrics:
            fn = METRIC_REGISTRY.get(name)
            if fn is None:
                logger.warning("Unknown metric '%s', skipping", name)
                continue
            before_scores[name] = fn(before_results)
            after_scores[name] = fn(after_results)

        return before_scores, after_scores, before_results, after_results
