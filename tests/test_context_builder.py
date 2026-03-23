"""Tests for guild_assistant.rag.context_builder."""

from pathlib import Path

import pytest

from guild_assistant.rag.context_builder import ContextBuilder, _PROMPT_TEMPLATE
from guild_assistant.rag.retriever import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(silver_path: str, doc_id: str = "doc") -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id,
        summary="A summary.",
        silver_path=silver_path,
        distance=0.1,
    )


def _populate_silver(directory: Path, files: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

class TestContextBuilderBuild:
    def test_includes_all_documents(self, tmp_path):
        _populate_silver(tmp_path, {
            "alpha.md": "Alpha content.",
            "beta.md": "Beta content.",
        })
        results = [
            _make_result(str(tmp_path / "alpha.md"), "alpha"),
            _make_result(str(tmp_path / "beta.md"), "beta"),
        ]

        builder = ContextBuilder()
        prompt = builder.build("What is alpha?", results)

        assert "Alpha content." in prompt
        assert "Beta content." in prompt

    def test_includes_user_query(self, tmp_path):
        _populate_silver(tmp_path, {"a.md": "text"})
        results = [_make_result(str(tmp_path / "a.md"))]

        builder = ContextBuilder()
        prompt = builder.build("How do I craft iron?", results)

        assert "How do I craft iron?" in prompt

    def test_includes_dont_know_instruction(self, tmp_path):
        _populate_silver(tmp_path, {"a.md": "text"})
        results = [_make_result(str(tmp_path / "a.md"))]

        builder = ContextBuilder()
        prompt = builder.build("query", results)

        assert "I don't have enough information" in prompt

    def test_source_labels_from_filenames(self, tmp_path):
        _populate_silver(tmp_path, {
            "yakmel-obtaining.md": "A text",
            "grace.md": "B text",
        })
        results = [
            _make_result(str(tmp_path / "yakmel-obtaining.md"), "a"),
            _make_result(str(tmp_path / "grace.md"), "b"),
        ]

        builder = ContextBuilder()
        prompt = builder.build("query", results)

        assert "--- Source: {yakmel, Section - obtaining} ---" in prompt
        assert "--- Source: {grace} ---" in prompt

    def test_skips_missing_silver_file(self, tmp_path):
        results = [
            _make_result(str(tmp_path / "missing.md"), "missing"),
        ]

        builder = ContextBuilder()
        prompt = builder.build("query", results)

        assert "Source:" not in prompt
        assert "query" in prompt

    def test_mixed_existing_and_missing(self, tmp_path):
        _populate_silver(tmp_path, {"exists.md": "Real content."})
        results = [
            _make_result(str(tmp_path / "exists.md"), "exists"),
            _make_result(str(tmp_path / "gone.md"), "gone"),
        ]

        builder = ContextBuilder()
        prompt = builder.build("query", results)

        assert "Real content." in prompt
        assert "--- Source: {exists} ---" in prompt

    def test_empty_results(self, tmp_path):
        builder = ContextBuilder()
        prompt = builder.build("query", [])

        assert "query" in prompt
        assert "I don't have enough information" in prompt
