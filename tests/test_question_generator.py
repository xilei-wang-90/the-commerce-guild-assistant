"""Tests for guild_assistant.rag_setup.question_generator."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from guild_assistant.rag_setup.question_generator import (
    QuestionGenerator,
    _OVERVIEW_PROMPT,
    _SECTION_PROMPT,
    _SECTION_NO_OVERVIEW_PROMPT,
    _is_overview,
    _overview_path_for,
    _parse_chunk_filename,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router(response: str = "What is this?\nWhy does it matter?") -> MagicMock:
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
# _parse_section_filename
# ---------------------------------------------------------------------------


class TestParseChunkFilename:
    def test_page_only(self):
        page, section = _parse_chunk_filename("yakmel.md")
        assert page == "yakmel"
        assert section == ""

    def test_section(self):
        page, section = _parse_chunk_filename("yakmel-obtaining.md")
        assert page == "yakmel"
        assert section == "obtaining"

    def test_overview(self):
        page, section = _parse_chunk_filename("yakmel-overview.md")
        assert page == "yakmel"
        assert section == "overview"

    def test_multi_word_page(self):
        page, section = _parse_chunk_filename("my_time_at_sandrock-gameplay.md")
        assert page == "my_time_at_sandrock"
        assert section == "gameplay"


# ---------------------------------------------------------------------------
# _is_overview
# ---------------------------------------------------------------------------


class TestIsOverview:
    def test_overview_file(self):
        assert _is_overview("yakmel-overview.md") is True

    def test_non_overview_file(self):
        assert _is_overview("yakmel-obtaining.md") is False

    def test_page_only_file(self):
        assert _is_overview("yakmel.md") is False

    def test_overview_in_middle_is_not_overview(self):
        assert _is_overview("yakmel-overview-extra.md") is False


# ---------------------------------------------------------------------------
# _overview_path_for
# ---------------------------------------------------------------------------


class TestOverviewPathFor:
    def test_returns_overview_sibling(self, tmp_path):
        section_file = tmp_path / "yakmel-obtaining.md"
        result = _overview_path_for(section_file)
        assert result == tmp_path / "yakmel-overview.md"

    def test_multi_word_page(self, tmp_path):
        section_file = tmp_path / "my_time_at_sandrock-gameplay.md"
        result = _overview_path_for(section_file)
        assert result == tmp_path / "my_time_at_sandrock-overview.md"


# ---------------------------------------------------------------------------
# generate_for_file — overview chunks
# ---------------------------------------------------------------------------


class TestGenerateForFileOverview:
    def test_overview_uses_overview_prompt(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"yakmel-overview.md": "Yakmel is a fluffy animal."})

        router = _make_router("Q1?\nQ2?\nQ3?\nQ4?\nQ5?")
        g = QuestionGenerator(router, input_dir, output_dir)
        g.generate_for_file(input_dir / "yakmel-overview.md")

        prompt_sent = router.generate.call_args[0][0]
        assert "Page: yakmel" in prompt_sent
        assert "5 questions" in prompt_sent
        assert "Overview content:" in prompt_sent
        assert "Yakmel is a fluffy animal." in prompt_sent

    def test_overview_does_not_include_section_title(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"yakmel-overview.md": "content"})

        router = _make_router("Q?")
        g = QuestionGenerator(router, input_dir, output_dir)
        g.generate_for_file(input_dir / "yakmel-overview.md")

        prompt_sent = router.generate.call_args[0][0]
        assert "Section:" not in prompt_sent

    def test_overview_writes_output(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"yakmel-overview.md": "content"})

        router = _make_router("Q1?\nQ2?\nQ3?\nQ4?\nQ5?")
        g = QuestionGenerator(router, input_dir, output_dir)
        result = g.generate_for_file(input_dir / "yakmel-overview.md")

        assert result == "Q1?\nQ2?\nQ3?\nQ4?\nQ5?"
        assert (output_dir / "yakmel-overview.md").read_text(encoding="utf-8") == result


# ---------------------------------------------------------------------------
# generate_for_file — non-overview chunks with overview context
# ---------------------------------------------------------------------------


class TestGenerateForFileSectionWithOverview:
    def test_includes_overview_context(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "yakmel-overview.md": "Yakmel is a fluffy animal in Sandrock.",
            "yakmel-obtaining.md": "You can find Yakmel in the wild.",
        })

        router = _make_router("Where can I find a Yakmel?")
        g = QuestionGenerator(router, input_dir, output_dir)
        g.generate_for_file(input_dir / "yakmel-obtaining.md")

        prompt_sent = router.generate.call_args[0][0]
        assert "Page: yakmel" in prompt_sent
        assert "Section: obtaining" in prompt_sent
        assert "1-3 questions" in prompt_sent
        assert "TARGET SECTION" in prompt_sent
        assert "OVERVIEW (for context only" in prompt_sent
        assert "Yakmel is a fluffy animal in Sandrock." in prompt_sent
        assert "You can find Yakmel in the wild." in prompt_sent

    def test_tells_llm_not_to_generate_overview_questions(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "yakmel-overview.md": "Overview text.",
            "yakmel-obtaining.md": "Section text.",
        })

        router = _make_router("Q?")
        g = QuestionGenerator(router, input_dir, output_dir)
        g.generate_for_file(input_dir / "yakmel-obtaining.md")

        prompt_sent = router.generate.call_args[0][0]
        assert "do NOT generate questions about this" in prompt_sent
        assert "Generate questions ONLY about the target section" in prompt_sent


# ---------------------------------------------------------------------------
# generate_for_file — non-overview chunks without overview context
# ---------------------------------------------------------------------------


class TestGenerateForFileSectionNoOverview:
    def test_falls_back_when_no_overview_exists(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"yakmel-obtaining.md": "You can find Yakmel."})

        router = _make_router("Where can I find a Yakmel?")
        g = QuestionGenerator(router, input_dir, output_dir)
        g.generate_for_file(input_dir / "yakmel-obtaining.md")

        prompt_sent = router.generate.call_args[0][0]
        assert "Page: yakmel" in prompt_sent
        assert "Section: obtaining" in prompt_sent
        assert "1-3 questions" in prompt_sent
        assert "Section content:" in prompt_sent
        assert "OVERVIEW" not in prompt_sent

    def test_writes_output(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"yakmel-obtaining.md": "content"})

        router = _make_router("Q?")
        g = QuestionGenerator(router, input_dir, output_dir)
        result = g.generate_for_file(input_dir / "yakmel-obtaining.md")

        assert result == "Q?"
        assert (output_dir / "yakmel-obtaining.md").read_text(encoding="utf-8") == "Q?"


# ---------------------------------------------------------------------------
# generate_for_file — general behavior
# ---------------------------------------------------------------------------


class TestGenerateForFile:
    def test_creates_output_dir_if_missing(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "deep" / "nested" / "output"
        _populate_input(input_dir, {"a-overview.md": "text"})

        g = QuestionGenerator(_make_router(), input_dir, output_dir)
        g.generate_for_file(input_dir / "a-overview.md")

        assert output_dir.is_dir()

    def test_output_filename_matches_input(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"sandrock_guide-combat.md": "Combat info."})

        g = QuestionGenerator(_make_router("Q1?\nQ2?"), input_dir, output_dir)
        g.generate_for_file(input_dir / "sandrock_guide-combat.md")

        assert (output_dir / "sandrock_guide-combat.md").exists()


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------


class TestGenerateAll:
    def test_processes_all_md_files(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "alpha-overview.md": "Content A",
            "alpha-details.md": "Content B",
        })

        router = _make_router("Q?")
        g = QuestionGenerator(router, input_dir, output_dir)
        results = g.generate_all()

        assert set(results.keys()) == {"alpha-overview.md", "alpha-details.md"}
        assert router.generate.call_count == 2

    def test_skips_existing_output(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "done-overview.md": "Already done",
            "new-overview.md": "Needs questions",
        })
        output_dir.mkdir(parents=True)
        (output_dir / "done-overview.md").write_text("existing questions", encoding="utf-8")

        router = _make_router("New question?")
        g = QuestionGenerator(router, input_dir, output_dir)
        results = g.generate_all()

        assert list(results.keys()) == ["new-overview.md"]
        router.generate.assert_called_once()

    def test_force_regenerates_existing(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"done-overview.md": "content"})
        output_dir.mkdir(parents=True)
        (output_dir / "done-overview.md").write_text("old questions", encoding="utf-8")

        router = _make_router("New question?")
        g = QuestionGenerator(router, input_dir, output_dir)
        results = g.generate_all(force=True)

        assert "done-overview.md" in results
        router.generate.assert_called_once()

    def test_empty_directory_returns_empty(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        g = QuestionGenerator(_make_router(), input_dir, output_dir)
        results = g.generate_all()

        assert results == {}

    def test_continues_on_single_file_failure(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "aaa-overview.md": "good",
            "bbb-overview.md": "bad",
            "ccc-overview.md": "good",
        })

        router = MagicMock()
        router.generate.side_effect = [
            "Q aaa?",
            RuntimeError("model exploded"),
            "Q ccc?",
        ]

        g = QuestionGenerator(router, input_dir, output_dir)
        results = g.generate_all(max_files=0)

        assert set(results.keys()) == {"aaa-overview.md", "ccc-overview.md"}

    def test_max_files_limits_processed(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "a-overview.md": "content",
            "b-overview.md": "content",
            "c-overview.md": "content",
            "d-overview.md": "content",
            "e-overview.md": "content",
        })

        router = _make_router("Q?")
        g = QuestionGenerator(router, input_dir, output_dir)
        results = g.generate_all(max_files=3)

        assert len(results) == 3
        assert router.generate.call_count == 3

    def test_max_files_zero_processes_all(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "a-overview.md": "content",
            "b-overview.md": "content",
            "c-overview.md": "content",
        })

        router = _make_router("Q?")
        g = QuestionGenerator(router, input_dir, output_dir)
        results = g.generate_all(max_files=0)

        assert len(results) == 3
        assert router.generate.call_count == 3

    def test_max_files_skipped_dont_count(self, tmp_path):
        """Skipped (already generated) files don't count toward max_files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "aaa-overview.md": "already done",
            "bbb-overview.md": "content",
            "ccc-overview.md": "content",
            "ddd-overview.md": "content",
        })
        output_dir.mkdir(parents=True)
        (output_dir / "aaa-overview.md").write_text("existing", encoding="utf-8")

        router = _make_router("Q?")
        g = QuestionGenerator(router, input_dir, output_dir)
        results = g.generate_all(max_files=2)

        # aaa-overview.md was skipped; 2 of the remaining 3 should be processed
        assert len(results) == 2
        assert "aaa-overview.md" not in results
        assert router.generate.call_count == 2
