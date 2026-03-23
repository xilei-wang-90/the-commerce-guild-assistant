"""Tests for guild_assistant.rag.reranker."""

from unittest.mock import MagicMock

import pytest

from guild_assistant.rag.reranker import Reranker
from guild_assistant.rag.retriever import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(doc_id: str, silver_path: str) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id,
        summary=f"summary for {doc_id}",
        silver_path=silver_path,
        distance=0.1,
    )


@pytest.fixture()
def reranker():
    adapter = MagicMock()
    r = Reranker(adapter=adapter, top_n=3)
    return r


# ---------------------------------------------------------------------------
# rerank()
# ---------------------------------------------------------------------------

class TestRerank:
    def test_returns_top_n_sorted_by_score(self, reranker, tmp_path):
        files = []
        results = []
        for i in range(5):
            f = tmp_path / f"doc{i}.md"
            f.write_text(f"Content {i}")
            files.append(f)
            results.append(_make_result(f"doc{i}", str(f)))

        reranker._adapter.score.return_value = [0.1, 0.5, 0.9, 0.3, 0.7]

        reranked = reranker.rerank("test query", results)

        assert len(reranked) == 3
        assert reranked[0].doc_id == "doc2"  # score 0.9
        assert reranked[1].doc_id == "doc4"  # score 0.7
        assert reranked[2].doc_id == "doc1"  # score 0.5

    def test_skips_missing_silver_paths(self, reranker, tmp_path):
        existing = tmp_path / "exists.md"
        existing.write_text("content")

        results = [
            _make_result("missing", "/nonexistent/path.md"),
            _make_result("exists", str(existing)),
        ]

        reranker._adapter.score.return_value = [0.8]

        reranked = reranker.rerank("query", results)

        assert len(reranked) == 1
        assert reranked[0].doc_id == "exists"

    def test_empty_results_returns_empty(self, reranker):
        assert reranker.rerank("query", []) == []
        reranker._adapter.score.assert_not_called()

    def test_all_missing_returns_empty(self, reranker):
        results = [_make_result("a", "/no/such/file.md")]

        reranked = reranker.rerank("query", results)

        assert reranked == []
        reranker._adapter.score.assert_not_called()

    def test_top_n_greater_than_available(self, reranker, tmp_path):
        f = tmp_path / "only.md"
        f.write_text("content")
        results = [_make_result("only", str(f))]

        reranker._adapter.score.return_value = [0.5]

        reranked = reranker.rerank("query", results)

        assert len(reranked) == 1
        assert reranked[0].doc_id == "only"
