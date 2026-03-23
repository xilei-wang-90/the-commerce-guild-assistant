"""Tests for guild_assistant.rag.summarizer."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from guild_assistant.rag_setup.summarizer import Summarizer, _PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_router(response: str = "A mock summary.") -> MagicMock:
    """Return a mock ModelRouter whose generate() returns *response*."""
    router = MagicMock()
    router.generate.return_value = response
    return router


def _populate_input(input_dir: Path, files: dict[str, str]) -> None:
    """Write *files* mapping ``{name: content}`` into *input_dir*."""
    input_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (input_dir / name).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# summarize_file
# ---------------------------------------------------------------------------

class TestSummarizeFile:
    def test_writes_summary_and_returns_text(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"page.md": "# Page\nSome content."})

        router = _make_router("Summary of page.")
        s = Summarizer(router, input_dir, output_dir)
        result = s.summarize_file(input_dir / "page.md")

        assert result == "Summary of page."
        assert (output_dir / "page.md").read_text(encoding="utf-8") == "Summary of page."

    def test_prompt_contains_file_content(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        content = "# Yakmel\nA fluffy animal."
        _populate_input(input_dir, {"yakmel.md": content})

        router = _make_router()
        s = Summarizer(router, input_dir, output_dir)
        s.summarize_file(input_dir / "yakmel.md")

        prompt_sent = router.generate.call_args[0][0]
        assert content in prompt_sent
        assert "5-8 sentences" in prompt_sent

    def test_creates_output_dir_if_missing(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "deep" / "nested" / "output"
        _populate_input(input_dir, {"a.md": "text"})

        s = Summarizer(_make_router(), input_dir, output_dir)
        s.summarize_file(input_dir / "a.md")

        assert output_dir.is_dir()


# ---------------------------------------------------------------------------
# summarize_all
# ---------------------------------------------------------------------------

class TestSummarizeAll:
    def test_processes_all_md_files(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "alpha.md": "Alpha content",
            "beta.md": "Beta content",
        })

        router = _make_router("summary")
        s = Summarizer(router, input_dir, output_dir)
        results = s.summarize_all()

        assert set(results.keys()) == {"alpha.md", "beta.md"}
        assert router.generate.call_count == 2

    def test_skips_existing_summaries(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "done.md": "Already done",
            "new.md": "Needs summary",
        })
        # Pre-create the summary for done.md
        output_dir.mkdir(parents=True)
        (output_dir / "done.md").write_text("existing summary", encoding="utf-8")

        router = _make_router("new summary")
        s = Summarizer(router, input_dir, output_dir)
        results = s.summarize_all()

        assert list(results.keys()) == ["new.md"]
        router.generate.assert_called_once()

    def test_force_resumes_existing(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"done.md": "content"})
        output_dir.mkdir(parents=True)
        (output_dir / "done.md").write_text("old", encoding="utf-8")

        router = _make_router("refreshed")
        s = Summarizer(router, input_dir, output_dir)
        results = s.summarize_all(force=True)

        assert "done.md" in results
        router.generate.assert_called_once()

    def test_empty_directory_returns_empty(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        s = Summarizer(_make_router(), input_dir, output_dir)
        results = s.summarize_all()

        assert results == {}

    def test_continues_on_single_file_failure(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "aaa.md": "good",
            "bbb.md": "bad",
            "ccc.md": "good",
        })

        router = MagicMock()
        router.generate.side_effect = [
            "summary aaa",
            RuntimeError("model exploded"),
            "summary ccc",
        ]

        s = Summarizer(router, input_dir, output_dir)
        results = s.summarize_all(max_files=0)

        assert set(results.keys()) == {"aaa.md", "ccc.md"}

    def test_max_files_limits_summarized(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "a.md": "content a",
            "b.md": "content b",
            "c.md": "content c",
            "d.md": "content d",
            "e.md": "content e",
        })

        router = _make_router("summary")
        s = Summarizer(router, input_dir, output_dir)
        results = s.summarize_all(max_files=3)

        assert len(results) == 3
        assert router.generate.call_count == 3

    def test_max_files_zero_summarizes_all(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "a.md": "content a",
            "b.md": "content b",
            "c.md": "content c",
        })

        router = _make_router("summary")
        s = Summarizer(router, input_dir, output_dir)
        results = s.summarize_all(max_files=0)

        assert len(results) == 3
        assert router.generate.call_count == 3

    def test_max_files_skipped_dont_count(self, tmp_path):
        """Skipped (already summarized) files don't count toward max_files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "done.md": "already done",
            "a.md": "content a",
            "b.md": "content b",
            "c.md": "content c",
        })
        output_dir.mkdir(parents=True)
        (output_dir / "done.md").write_text("existing", encoding="utf-8")

        router = _make_router("summary")
        s = Summarizer(router, input_dir, output_dir)
        results = s.summarize_all(max_files=2)

        # done.md was skipped; 2 of the remaining 3 should be summarized
        assert len(results) == 2
        assert "done.md" not in results
        assert router.generate.call_count == 2
