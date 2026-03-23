"""Tests for tests.benchmark.retrieval_evaluator."""

from unittest.mock import MagicMock

import pytest

from guild_assistant.rag.retriever import RetrievalResult
from tests.benchmark.retrieval_evaluator import RetrievalEvaluator
from tests.benchmark.testset_loader import TestCase

# Import for type hints only — Reranker is mocked in rerank tests.
from guild_assistant.rag.reranker import Reranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retriever(results_map: dict[str, list[RetrievalResult]]) -> MagicMock:
    """Return a mock Retriever that returns results based on the query."""
    retriever = MagicMock()

    def _retrieve(query: str) -> list[RetrievalResult]:
        return results_map.get(query, [])

    retriever.retrieve.side_effect = _retrieve
    return retriever


def _result(silver_path: str, distance: float = 0.1) -> RetrievalResult:
    return RetrievalResult(
        doc_id="id",
        summary="summary",
        silver_path=silver_path,
        distance=distance,
    )


# ---------------------------------------------------------------------------
# Summary collection (match on page)
# ---------------------------------------------------------------------------


class TestSummaryCollection:
    def test_hit_when_page_matches(self):
        retriever = _make_retriever({
            "What is a Yakmel?": [
                _result("/data/silver/yakmel.md"),
                _result("/data/silver/sword.md"),
            ],
        })
        cases = [TestCase("What is a Yakmel?", "A.", "yakmel-overview.md", "yakmel.md")]
        evaluator = RetrievalEvaluator(retriever, "sandrock_wiki_summary", k=5)
        scores, _ = evaluator.run(cases, metrics=["hit_rate"])
        assert scores["hit_rate"] == 1.0

    def test_miss_when_page_not_in_results(self):
        retriever = _make_retriever({
            "What is a Yakmel?": [
                _result("/data/silver/sword.md"),
                _result("/data/silver/shield.md"),
            ],
        })
        cases = [TestCase("What is a Yakmel?", "A.", "yakmel-overview.md", "yakmel.md")]
        evaluator = RetrievalEvaluator(retriever, "sandrock_wiki_summary", k=5)
        scores, _ = evaluator.run(cases, metrics=["hit_rate"])
        assert scores["hit_rate"] == 0.0

    def test_partial_hits(self):
        retriever = _make_retriever({
            "Q1?": [_result("/data/silver/yakmel.md")],
            "Q2?": [_result("/data/silver/sword.md")],
        })
        cases = [
            TestCase("Q1?", "A1.", "yakmel-overview.md", "yakmel.md"),
            TestCase("Q2?", "A2.", "shield-overview.md", "shield.md"),
        ]
        evaluator = RetrievalEvaluator(retriever, "sandrock_wiki_summary", k=5)
        scores, _ = evaluator.run(cases, metrics=["hit_rate"])
        assert scores["hit_rate"] == 0.5


# ---------------------------------------------------------------------------
# Section-reverse-hyde collection (match on section)
# ---------------------------------------------------------------------------


class TestSectionCollection:
    def test_hit_when_section_matches(self):
        retriever = _make_retriever({
            "Q?": [
                _result("/data/silver-sections/yakmel-overview.md"),
                _result("/data/silver-sections/sword-combat.md"),
            ],
        })
        cases = [TestCase("Q?", "A.", "yakmel-overview.md", "yakmel.md")]
        evaluator = RetrievalEvaluator(
            retriever, "sandrock_wiki_section_reverse_hyde", k=5,
        )
        scores, _ = evaluator.run(cases, metrics=["hit_rate"])
        assert scores["hit_rate"] == 1.0

    def test_miss_when_section_not_in_results(self):
        retriever = _make_retriever({
            "Q?": [_result("/data/silver-sections/sword-combat.md")],
        })
        cases = [TestCase("Q?", "A.", "yakmel-overview.md", "yakmel.md")]
        evaluator = RetrievalEvaluator(
            retriever, "sandrock_wiki_section_reverse_hyde", k=5,
        )
        scores, _ = evaluator.run(cases, metrics=["hit_rate"])
        assert scores["hit_rate"] == 0.0


# ---------------------------------------------------------------------------
# Multiple metrics / unknown metrics
# ---------------------------------------------------------------------------


class TestMetricSelection:
    def test_multiple_metrics(self):
        retriever = _make_retriever({
            "Q?": [_result("/data/silver/yakmel.md")],
        })
        cases = [TestCase("Q?", "A.", "yakmel-overview.md", "yakmel.md")]
        evaluator = RetrievalEvaluator(retriever, "sandrock_wiki_summary", k=5)
        scores, _ = evaluator.run(cases, metrics=["hit_rate"])
        assert "hit_rate" in scores

    def test_unknown_metric_skipped(self):
        retriever = _make_retriever({
            "Q?": [_result("/data/silver/yakmel.md")],
        })
        cases = [TestCase("Q?", "A.", "yakmel-overview.md", "yakmel.md")]
        evaluator = RetrievalEvaluator(retriever, "sandrock_wiki_summary", k=5)
        scores, _ = evaluator.run(cases, metrics=["hit_rate", "nonexistent_metric"])
        assert "hit_rate" in scores
        assert "nonexistent_metric" not in scores


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_test_cases(self):
        retriever = MagicMock()
        evaluator = RetrievalEvaluator(retriever, "sandrock_wiki_summary", k=5)
        scores, query_results = evaluator.run([], metrics=["hit_rate"])
        assert scores["hit_rate"] == 0.0
        assert query_results == []
        retriever.retrieve.assert_not_called()

    def test_empty_retrieval_results(self):
        retriever = _make_retriever({"Q?": []})
        cases = [TestCase("Q?", "A.", "s.md", "p.md")]
        evaluator = RetrievalEvaluator(retriever, "sandrock_wiki_summary", k=5)
        scores, _ = evaluator.run(cases, metrics=["hit_rate"])
        assert scores["hit_rate"] == 0.0

    def test_unknown_collection_defaults_to_page(self):
        """Unknown collection names fall back to matching on page."""
        retriever = _make_retriever({
            "Q?": [_result("/data/silver/yakmel.md")],
        })
        cases = [TestCase("Q?", "A.", "yakmel-overview.md", "yakmel.md")]
        evaluator = RetrievalEvaluator(retriever, "unknown_collection", k=5)
        scores, _ = evaluator.run(cases, metrics=["hit_rate"])
        assert scores["hit_rate"] == 1.0

    def test_query_results_returned(self):
        """run() returns per-query QueryResult objects."""
        retriever = _make_retriever({
            "Q1?": [_result("/data/silver/yakmel.md"), _result("/data/silver/sword.md")],
            "Q2?": [_result("/data/silver/shield.md")],
        })
        cases = [
            TestCase("Q1?", "A1.", "yakmel-overview.md", "yakmel.md"),
            TestCase("Q2?", "A2.", "shield-overview.md", "shield.md"),
        ]
        evaluator = RetrievalEvaluator(retriever, "sandrock_wiki_summary", k=5)
        _, query_results = evaluator.run(cases, metrics=["hit_rate"])

        assert len(query_results) == 2
        assert query_results[0].question == "Q1?"
        assert query_results[0].expected_id == "yakmel.md"
        assert query_results[0].retrieved_ids == ["yakmel.md", "sword.md"]
        assert query_results[1].expected_id == "shield.md"
        assert query_results[1].retrieved_ids == ["shield.md"]


# ---------------------------------------------------------------------------
# Rerank mode
# ---------------------------------------------------------------------------


def _make_reranker(reranked_map: dict[str, list[RetrievalResult]]) -> MagicMock:
    """Return a mock Reranker that returns pre-defined reranked results."""
    reranker = MagicMock(spec=Reranker)

    def _rerank(query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return reranked_map.get(query, [])

    reranker.rerank.side_effect = _rerank
    return reranker


class TestRerankMode:
    def test_run_with_rerank_returns_before_and_after(self):
        """run_with_rerank returns scores and results for both stages."""
        # Retriever returns 4 results; reranker reorders them.
        retriever = _make_retriever({
            "Q?": [
                _result("/data/silver/sword.md", 0.1),
                _result("/data/silver/shield.md", 0.2),
                _result("/data/silver/yakmel.md", 0.3),
                _result("/data/silver/axe.md", 0.4),
            ],
        })
        # Reranker puts yakmel first.
        reranker = _make_reranker({
            "Q?": [
                _result("/data/silver/yakmel.md", 0.3),
                _result("/data/silver/sword.md", 0.1),
            ],
        })
        cases = [TestCase("Q?", "A.", "yakmel-overview.md", "yakmel.md")]
        evaluator = RetrievalEvaluator(
            retriever, "sandrock_wiki_summary", k=2,
            reranker=reranker, retrieve_n=4,
        )
        before_scores, after_scores, before_results, after_results = evaluator.run_with_rerank(
            cases, metrics=["ndcg"],
        )

        # Before: yakmel is at position 3 (index 2) in top-2 → MISS.
        assert before_results[0].retrieved_ids == ["sword.md", "shield.md"]
        # After: reranker moved yakmel to position 1.
        assert after_results[0].retrieved_ids == ["yakmel.md", "sword.md"]

        assert before_scores["ndcg"] == 0.0  # yakmel not in top-2
        assert after_scores["ndcg"] > 0.0    # yakmel at rank 1

    def test_run_with_rerank_no_reranker_raises(self):
        """run_with_rerank raises ValueError if no reranker was provided."""
        retriever = _make_retriever({})
        evaluator = RetrievalEvaluator(retriever, "sandrock_wiki_summary", k=2)
        with pytest.raises(ValueError, match="requires a Reranker"):
            evaluator.run_with_rerank([], metrics=["ndcg"])

    def test_rerank_improves_ndcg(self):
        """Reranking should improve NDCG when it promotes the relevant doc."""
        # 5 results, expected doc is last.
        retriever = _make_retriever({
            "Q?": [
                _result("/data/silver/a.md", 0.1),
                _result("/data/silver/b.md", 0.2),
                _result("/data/silver/c.md", 0.3),
                _result("/data/silver/d.md", 0.4),
                _result("/data/silver/target.md", 0.5),
            ],
        })
        # Reranker promotes target to the top.
        reranker = _make_reranker({
            "Q?": [
                _result("/data/silver/target.md", 0.5),
                _result("/data/silver/a.md", 0.1),
                _result("/data/silver/b.md", 0.2),
            ],
        })
        cases = [TestCase("Q?", "A.", "target-overview.md", "target.md")]
        evaluator = RetrievalEvaluator(
            retriever, "sandrock_wiki_summary", k=3,
            reranker=reranker, retrieve_n=5,
        )
        before_scores, after_scores, _, _ = evaluator.run_with_rerank(
            cases, metrics=["ndcg"],
        )

        assert after_scores["ndcg"] > before_scores["ndcg"]

    def test_rerank_multiple_metrics(self):
        """run_with_rerank computes all requested metrics."""
        retriever = _make_retriever({
            "Q?": [
                _result("/data/silver/yakmel.md", 0.1),
                _result("/data/silver/sword.md", 0.2),
            ],
        })
        reranker = _make_reranker({
            "Q?": [
                _result("/data/silver/yakmel.md", 0.1),
            ],
        })
        cases = [TestCase("Q?", "A.", "yakmel-overview.md", "yakmel.md")]
        evaluator = RetrievalEvaluator(
            retriever, "sandrock_wiki_summary", k=2,
            reranker=reranker, retrieve_n=2,
        )
        before_scores, after_scores, _, _ = evaluator.run_with_rerank(
            cases, metrics=["hit_rate", "ndcg"],
        )

        assert "hit_rate" in before_scores
        assert "ndcg" in before_scores
        assert "hit_rate" in after_scores
        assert "ndcg" in after_scores

    def test_rerank_unknown_metric_skipped(self):
        """Unknown metrics are skipped in rerank mode too."""
        retriever = _make_retriever({"Q?": [_result("/data/silver/a.md")]})
        reranker = _make_reranker({"Q?": [_result("/data/silver/a.md")]})
        cases = [TestCase("Q?", "A.", "a-overview.md", "a.md")]
        evaluator = RetrievalEvaluator(
            retriever, "sandrock_wiki_summary", k=1,
            reranker=reranker, retrieve_n=1,
        )
        before_scores, after_scores, _, _ = evaluator.run_with_rerank(
            cases, metrics=["ndcg", "nonexistent"],
        )
        assert "ndcg" in before_scores
        assert "nonexistent" not in before_scores

    def test_rerank_empty_test_cases(self):
        """Empty test cases returns zero scores for both stages."""
        retriever = MagicMock()
        reranker = _make_reranker({})
        evaluator = RetrievalEvaluator(
            retriever, "sandrock_wiki_summary", k=3,
            reranker=reranker, retrieve_n=10,
        )
        before_scores, after_scores, before_results, after_results = evaluator.run_with_rerank(
            [], metrics=["ndcg"],
        )
        assert before_scores["ndcg"] == 0.0
        assert after_scores["ndcg"] == 0.0
        assert before_results == []
        assert after_results == []
