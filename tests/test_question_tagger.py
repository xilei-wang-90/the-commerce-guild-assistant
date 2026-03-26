"""Tests for guild_assistant.rag_setup.question_tagger."""

from pathlib import Path

import pytest

from guild_assistant.rag_setup.question_tagger import (
    QuestionTagger,
    _build_tag_line,
    _extract_l2_l3_titles,
    _parse_page_slug,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate(directory: Path, files: dict[str, str]) -> None:
    """Write *files* mapping ``{name: content}`` into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# _extract_l2_l3_titles
# ---------------------------------------------------------------------------


class TestExtractL2L3Titles:
    def test_extracts_l2_and_l3(self):
        content = "## Obtaining\n### From NPCs\nSome text.\n### From Enemies\n"
        assert _extract_l2_l3_titles(content) == [
            "Obtaining",
            "From NPCs",
            "From Enemies",
        ]

    def test_ignores_l1_and_l4(self):
        content = "# Page Title\n## Section\n### Sub\n#### Deep\n"
        assert _extract_l2_l3_titles(content) == ["Section", "Sub"]

    def test_empty_content(self):
        assert _extract_l2_l3_titles("") == []

    def test_no_headings(self):
        assert _extract_l2_l3_titles("Just plain text.") == []

    def test_strips_whitespace(self):
        content = "##  Spaced Title  \n###  Sub Title  \n"
        assert _extract_l2_l3_titles(content) == ["Spaced Title", "Sub Title"]

    def test_setext_l2_heading(self):
        content = "Obtaining\n---------\nSome text.\n"
        assert _extract_l2_l3_titles(content) == ["Obtaining"]

    def test_setext_l1_ignored(self):
        content = "Page Title\n==========\nSome text.\n"
        assert _extract_l2_l3_titles(content) == []

    def test_mixed_atx_and_setext(self):
        content = "Obtaining\n---------\nSome text.\n### From NPCs\nMore text.\n"
        assert _extract_l2_l3_titles(content) == ["Obtaining", "From NPCs"]

    def test_setext_blank_title_ignored(self):
        content = "\n---------\nSome text.\n"
        assert _extract_l2_l3_titles(content) == []


# ---------------------------------------------------------------------------
# _build_tag_line
# ---------------------------------------------------------------------------


class TestBuildTagLine:
    def test_with_headings(self):
        result = _build_tag_line("yakmel", ["Obtaining", "From NPCs"])
        assert result == "[yakmel | Obtaining | From NPCs]"

    def test_no_headings(self):
        result = _build_tag_line("yakmel", [])
        assert result == "[yakmel]"

    def test_single_heading(self):
        result = _build_tag_line("page", ["Section"])
        assert result == "[page | Section]"


# ---------------------------------------------------------------------------
# _parse_page_slug
# ---------------------------------------------------------------------------


class TestParsePageSlug:
    def test_section_file(self):
        assert _parse_page_slug("yakmel-obtaining.md") == "yakmel"

    def test_overview_file(self):
        assert _parse_page_slug("yakmel-overview.md") == "yakmel"

    def test_no_section(self):
        assert _parse_page_slug("yakmel.md") == "yakmel"

    def test_multi_word_page(self):
        assert _parse_page_slug("my_time_at_sandrock-gameplay.md") == "my_time_at_sandrock"


# ---------------------------------------------------------------------------
# tag_file
# ---------------------------------------------------------------------------


class TestTagFile:
    def test_prepends_tag_line(self, tmp_path):
        questions_dir = tmp_path / "questions"
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "output"

        _populate(questions_dir, {"yakmel-obtaining.md": "Where can I find Yakmel?\nHow to tame Yakmel?"})
        _populate(sections_dir, {"yakmel-obtaining.md": "## Obtaining\n### From NPCs\nBuy from vendor.\n### From Wild\nCatch in desert.\n"})

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        result = tagger.tag_file(questions_dir / "yakmel-obtaining.md")

        assert result.startswith("[yakmel | Obtaining | From NPCs | From Wild]")
        assert "Where can I find Yakmel?" in result
        assert "How to tame Yakmel?" in result

    def test_writes_output_file(self, tmp_path):
        questions_dir = tmp_path / "questions"
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "output"

        _populate(questions_dir, {"a-overview.md": "Q1?\nQ2?"})
        _populate(sections_dir, {"a-overview.md": "## Overview\nText.\n"})

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        tagger.tag_file(questions_dir / "a-overview.md")

        out = (output_dir / "a-overview.md").read_text(encoding="utf-8")
        assert out.startswith("[a | Overview]")

    def test_missing_section_file_uses_slug_only(self, tmp_path):
        questions_dir = tmp_path / "questions"
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "output"

        _populate(questions_dir, {"orphan-details.md": "Q?"})
        sections_dir.mkdir(parents=True)  # empty, no matching file

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        result = tagger.tag_file(questions_dir / "orphan-details.md")

        assert result == "[orphan]\nQ?"

    def test_setext_headings_in_section(self, tmp_path):
        questions_dir = tmp_path / "questions"
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "output"

        _populate(questions_dir, {"yakmel-obtaining.md": "How to get yakmel?"})
        _populate(sections_dir, {"yakmel-obtaining.md": "Obtaining\n---------\nBuy from vendor.\n### From Wild\nCatch in desert.\n"})

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        result = tagger.tag_file(questions_dir / "yakmel-obtaining.md")

        assert result.startswith("[yakmel | Obtaining | From Wild]")
        assert "How to get yakmel?" in result

    def test_creates_output_dir(self, tmp_path):
        questions_dir = tmp_path / "questions"
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "deep" / "nested" / "output"

        _populate(questions_dir, {"a-overview.md": "Q?"})
        _populate(sections_dir, {"a-overview.md": "## Overview\nText.\n"})

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        tagger.tag_file(questions_dir / "a-overview.md")

        assert output_dir.is_dir()


# ---------------------------------------------------------------------------
# tag_all
# ---------------------------------------------------------------------------


class TestTagAll:
    def test_tags_all_files(self, tmp_path):
        questions_dir = tmp_path / "questions"
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "output"

        _populate(questions_dir, {
            "alpha-overview.md": "Q1?",
            "alpha-details.md": "Q2?",
        })
        _populate(sections_dir, {
            "alpha-overview.md": "## Overview\nText.\n",
            "alpha-details.md": "## Details\n### Sub\nText.\n",
        })

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        results = tagger.tag_all()

        assert set(results.keys()) == {"alpha-overview.md", "alpha-details.md"}

    def test_skips_existing_output(self, tmp_path):
        questions_dir = tmp_path / "questions"
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "output"

        _populate(questions_dir, {
            "done-overview.md": "Q old?",
            "new-overview.md": "Q new?",
        })
        _populate(sections_dir, {
            "done-overview.md": "## Overview\nText.\n",
            "new-overview.md": "## Overview\nText.\n",
        })
        output_dir.mkdir(parents=True)
        (output_dir / "done-overview.md").write_text("existing", encoding="utf-8")

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        results = tagger.tag_all()

        assert list(results.keys()) == ["new-overview.md"]

    def test_force_retags_existing(self, tmp_path):
        questions_dir = tmp_path / "questions"
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "output"

        _populate(questions_dir, {"done-overview.md": "Q?"})
        _populate(sections_dir, {"done-overview.md": "## Overview\nText.\n"})
        output_dir.mkdir(parents=True)
        (output_dir / "done-overview.md").write_text("old", encoding="utf-8")

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        results = tagger.tag_all(force=True)

        assert "done-overview.md" in results

    def test_empty_directory_returns_empty(self, tmp_path):
        questions_dir = tmp_path / "questions"
        questions_dir.mkdir()
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "output"

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        results = tagger.tag_all()

        assert results == {}

    def test_max_files_limits_processed(self, tmp_path):
        questions_dir = tmp_path / "questions"
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "output"

        _populate(questions_dir, {
            "a-overview.md": "Q?",
            "b-overview.md": "Q?",
            "c-overview.md": "Q?",
            "d-overview.md": "Q?",
        })
        _populate(sections_dir, {
            "a-overview.md": "## Overview\nT.\n",
            "b-overview.md": "## Overview\nT.\n",
            "c-overview.md": "## Overview\nT.\n",
            "d-overview.md": "## Overview\nT.\n",
        })

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        results = tagger.tag_all(max_files=2)

        assert len(results) == 2

    def test_max_files_zero_processes_all(self, tmp_path):
        questions_dir = tmp_path / "questions"
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "output"

        _populate(questions_dir, {
            "a-overview.md": "Q?",
            "b-overview.md": "Q?",
            "c-overview.md": "Q?",
        })
        _populate(sections_dir, {
            "a-overview.md": "## Overview\nT.\n",
            "b-overview.md": "## Overview\nT.\n",
            "c-overview.md": "## Overview\nT.\n",
        })

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        results = tagger.tag_all(max_files=0)

        assert len(results) == 3

    def test_skipped_dont_count_toward_max(self, tmp_path):
        questions_dir = tmp_path / "questions"
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "output"

        _populate(questions_dir, {
            "aaa-overview.md": "Q?",
            "bbb-overview.md": "Q?",
            "ccc-overview.md": "Q?",
            "ddd-overview.md": "Q?",
        })
        _populate(sections_dir, {
            "aaa-overview.md": "## Overview\nT.\n",
            "bbb-overview.md": "## Overview\nT.\n",
            "ccc-overview.md": "## Overview\nT.\n",
            "ddd-overview.md": "## Overview\nT.\n",
        })
        output_dir.mkdir(parents=True)
        (output_dir / "aaa-overview.md").write_text("existing", encoding="utf-8")

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        results = tagger.tag_all(max_files=2)

        assert len(results) == 2
        assert "aaa-overview.md" not in results

    def test_continues_on_single_file_failure(self, tmp_path):
        questions_dir = tmp_path / "questions"
        sections_dir = tmp_path / "sections"
        output_dir = tmp_path / "output"

        _populate(questions_dir, {
            "aaa-overview.md": "Q?",
            "ccc-overview.md": "Q?",
        })
        _populate(sections_dir, {
            "aaa-overview.md": "## Overview\nT.\n",
            "ccc-overview.md": "## Overview\nT.\n",
        })

        # Create a file that will cause a read error
        bad_file = questions_dir / "bbb-overview.md"
        bad_file.mkdir()  # directory instead of file → read will fail

        tagger = QuestionTagger(questions_dir, sections_dir, output_dir)
        results = tagger.tag_all(max_files=0)

        assert set(results.keys()) == {"aaa-overview.md", "ccc-overview.md"}
