"""Tests for tests.benchmark.metrics."""

import math

import pytest

from tests.benchmark.metrics import (
    METRIC_REGISTRY,
    QueryResult,
    _page_slug,
    hit_rate,
    ndcg,
    page_hit_rate,
)


# ---------------------------------------------------------------------------
# hit_rate
# ---------------------------------------------------------------------------


class TestHitRate:
    def test_all_hits(self):
        results = [
            QueryResult("q1", "a.md", ["a.md", "b.md"]),
            QueryResult("q2", "c.md", ["c.md", "d.md"]),
        ]
        assert hit_rate(results) == 1.0

    def test_all_misses(self):
        results = [
            QueryResult("q1", "a.md", ["b.md", "c.md"]),
            QueryResult("q2", "d.md", ["e.md", "f.md"]),
        ]
        assert hit_rate(results) == 0.0

    def test_partial_hits(self):
        results = [
            QueryResult("q1", "a.md", ["a.md", "b.md"]),
            QueryResult("q2", "c.md", ["d.md", "e.md"]),
            QueryResult("q3", "f.md", ["f.md", "g.md"]),
            QueryResult("q4", "h.md", ["i.md", "j.md"]),
        ]
        assert hit_rate(results) == 0.5

    def test_empty_results(self):
        assert hit_rate([]) == 0.0

    def test_single_result_hit(self):
        results = [QueryResult("q1", "a.md", ["a.md"])]
        assert hit_rate(results) == 1.0

    def test_single_result_miss(self):
        results = [QueryResult("q1", "a.md", ["b.md"])]
        assert hit_rate(results) == 0.0

    def test_hit_not_first_result(self):
        """Hit counts even if expected doc is not the top result."""
        results = [
            QueryResult("q1", "a.md", ["x.md", "y.md", "a.md"]),
        ]
        assert hit_rate(results) == 1.0

    def test_empty_retrieved_ids(self):
        results = [QueryResult("q1", "a.md", [])]
        assert hit_rate(results) == 0.0


# ---------------------------------------------------------------------------
# _page_slug
# ---------------------------------------------------------------------------


class TestPageSlug:
    def test_page_filename(self):
        assert _page_slug("yakmel.md") == "yakmel"

    def test_section_filename(self):
        assert _page_slug("yakmel-overview.md") == "yakmel"

    def test_section_with_long_name(self):
        assert _page_slug("yakmel-battle-statistics.md") == "yakmel"

    def test_underscore_page(self):
        assert _page_slug("advanced_forging_machine.md") == "advanced_forging_machine"

    def test_underscore_section(self):
        assert _page_slug("advanced_forging_machine-obtaining.md") == "advanced_forging_machine"


# ---------------------------------------------------------------------------
# page_hit_rate
# ---------------------------------------------------------------------------


class TestPageHitRate:
    def test_section_miss_but_page_hit(self):
        """Different section of same page counts as a page-level hit."""
        results = [
            QueryResult(
                "What is a Yakmel?",
                "yakmel-overview.md",
                ["yakmel-obtaining.md", "sword-combat.md"],
            ),
        ]
        assert hit_rate(results) == 0.0
        assert page_hit_rate(results) == 1.0

    def test_page_miss(self):
        results = [
            QueryResult(
                "What is a Yakmel?",
                "yakmel-overview.md",
                ["sword-combat.md", "shield-overview.md"],
            ),
        ]
        assert page_hit_rate(results) == 0.0

    def test_partial_page_hits(self):
        results = [
            QueryResult("Q1?", "yakmel-overview.md", ["yakmel-obtaining.md"]),
            QueryResult("Q2?", "sword-combat.md", ["shield-overview.md"]),
        ]
        assert page_hit_rate(results) == 0.5

    def test_summary_collection_same_as_hit_rate(self):
        """For page-level IDs, page_hit_rate equals hit_rate."""
        results = [
            QueryResult("Q1?", "yakmel.md", ["yakmel.md", "sword.md"]),
            QueryResult("Q2?", "shield.md", ["sword.md", "axe.md"]),
        ]
        assert page_hit_rate(results) == hit_rate(results)

    def test_empty_results(self):
        assert page_hit_rate([]) == 0.0

    def test_empty_retrieved_ids(self):
        results = [QueryResult("Q?", "yakmel-overview.md", [])]
        assert page_hit_rate(results) == 0.0


# ---------------------------------------------------------------------------
# ndcg
# ---------------------------------------------------------------------------


class TestNdcg:
    def test_all_at_rank_1(self):
        """Expected doc at rank 1 gives NDCG = 1.0."""
        results = [
            QueryResult("q1", "a.md", ["a.md", "b.md"]),
            QueryResult("q2", "c.md", ["c.md", "d.md"]),
        ]
        assert ndcg(results) == pytest.approx(1.0)

    def test_all_misses(self):
        results = [
            QueryResult("q1", "a.md", ["b.md", "c.md"]),
            QueryResult("q2", "d.md", ["e.md", "f.md"]),
        ]
        assert ndcg(results) == 0.0

    def test_at_rank_2(self):
        """Expected doc at rank 2 gives 1/log2(3)."""
        results = [QueryResult("q1", "a.md", ["b.md", "a.md"])]
        assert ndcg(results) == pytest.approx(1.0 / math.log2(3))

    def test_at_rank_5(self):
        """Expected doc at rank 5 gives 1/log2(6)."""
        results = [
            QueryResult("q1", "a.md", ["b.md", "c.md", "d.md", "e.md", "a.md"]),
        ]
        assert ndcg(results) == pytest.approx(1.0 / math.log2(6))

    def test_mixed_ranks(self):
        """Average of individual NDCG scores."""
        results = [
            QueryResult("q1", "a.md", ["a.md", "b.md"]),  # rank 1 → 1.0
            QueryResult("q2", "c.md", ["d.md", "c.md"]),  # rank 2 → 1/log2(3)
            QueryResult("q3", "e.md", ["f.md", "g.md"]),  # miss → 0.0
        ]
        expected = (1.0 + 1.0 / math.log2(3) + 0.0) / 3
        assert ndcg(results) == pytest.approx(expected)

    def test_empty_results(self):
        assert ndcg([]) == 0.0

    def test_empty_retrieved_ids(self):
        results = [QueryResult("q1", "a.md", [])]
        assert ndcg(results) == 0.0


# ---------------------------------------------------------------------------
# METRIC_REGISTRY
# ---------------------------------------------------------------------------


class TestMetricRegistry:
    def test_hit_rate_registered(self):
        assert "hit_rate" in METRIC_REGISTRY
        assert METRIC_REGISTRY["hit_rate"] is hit_rate

    def test_page_hit_rate_registered(self):
        assert "page_hit_rate" in METRIC_REGISTRY
        assert METRIC_REGISTRY["page_hit_rate"] is page_hit_rate

    def test_ndcg_registered(self):
        assert "ndcg" in METRIC_REGISTRY
        assert METRIC_REGISTRY["ndcg"] is ndcg
