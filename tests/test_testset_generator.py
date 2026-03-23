"""Tests for guild_assistant.rag_test.testset_generator."""

import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from guild_assistant.rag_test.testset_generator import (
    TestsetGenerator,
    _find_sections,
    _load_processed_pages,
    _page_slug_from_filename,
    _parse_response,
    _parse_section_filename,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RESPONSE = """\
FACTOID_Q: What is the price of a Yakmel?
FACTOID_A: 500 gols.
CONCEPTUAL_Q: How do you tame a Yakmel?
CONCEPTUAL_A: You approach it slowly and offer food.
MESSY_Q: yakmel taming tips sandrock
MESSY_A: Offer food and approach slowly to tame a Yakmel."""


def _make_router(response: str = _VALID_RESPONSE) -> MagicMock:
    """Return a mock ModelRouter whose generate() returns *response*."""
    router = MagicMock()
    router.generate.return_value = response
    return router


def _populate(directory: Path, files: dict[str, str]) -> None:
    """Write *files* mapping ``{name: content}`` into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")


def _write_golden_pages(path: Path, pages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pages) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_response(self):
        result = _parse_response(_VALID_RESPONSE)
        assert result is not None
        assert result["FACTOID_Q"] == "What is the price of a Yakmel?"
        assert result["FACTOID_A"] == "500 gols."
        assert result["CONCEPTUAL_Q"] == "How do you tame a Yakmel?"
        assert result["MESSY_Q"] == "yakmel taming tips sandrock"

    def test_missing_field_returns_none(self):
        response = "FACTOID_Q: question\nFACTOID_A: answer"
        assert _parse_response(response) is None

    def test_extra_lines_ignored(self):
        response = (
            "Here are the results:\n"
            + _VALID_RESPONSE
            + "\nThat's all!"
        )
        result = _parse_response(response)
        assert result is not None
        assert result["FACTOID_Q"] == "What is the price of a Yakmel?"

    def test_no_colon_in_values(self):
        response = (
            "FACTOID_Q: What costs 500: the sword or the shield?\n"
            "FACTOID_A: The sword costs 500.\n"
            "CONCEPTUAL_Q: Why?\n"
            "CONCEPTUAL_A: Because.\n"
            "MESSY_Q: sword price\n"
            "MESSY_A: 500 gols."
        )
        result = _parse_response(response)
        assert result is not None
        assert result["FACTOID_Q"] == "What costs 500: the sword or the shield?"


# ---------------------------------------------------------------------------
# _find_sections
# ---------------------------------------------------------------------------


class TestFindSections:
    def test_finds_matching_sections(self, tmp_path):
        _populate(tmp_path, {
            "yakmel-overview.md": "overview",
            "yakmel-obtaining.md": "obtaining",
            "sword-overview.md": "other page",
        })
        result = _find_sections(tmp_path, "yakmel")
        names = [p.name for p in result]
        assert "yakmel-overview.md" in names
        assert "yakmel-obtaining.md" in names
        assert "sword-overview.md" not in names

    def test_no_matches(self, tmp_path):
        _populate(tmp_path, {"sword-overview.md": "content"})
        assert _find_sections(tmp_path, "yakmel") == []


# ---------------------------------------------------------------------------
# _page_slug_from_filename
# ---------------------------------------------------------------------------


class TestPageSlugFromFilename:
    def test_strips_extension(self):
        assert _page_slug_from_filename("yakmel.md") == "yakmel"

    def test_multi_word(self):
        assert _page_slug_from_filename("my_time_at_sandrock.md") == "my_time_at_sandrock"


# ---------------------------------------------------------------------------
# _parse_section_filename
# ---------------------------------------------------------------------------


class TestParseSectionFilename:
    def test_with_section(self):
        page, section = _parse_section_filename("yakmel-obtaining.md")
        assert page == "yakmel"
        assert section == "obtaining"

    def test_overview(self):
        page, section = _parse_section_filename("yakmel-overview.md")
        assert page == "yakmel"
        assert section == "overview"


# ---------------------------------------------------------------------------
# _load_processed_pages
# ---------------------------------------------------------------------------


class TestLoadProcessedPages:
    def test_empty_dir(self, tmp_path):
        assert _load_processed_pages(tmp_path) == set()

    def test_reads_from_factoid_csv(self, tmp_path):
        csv_path = tmp_path / "factoid.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer", "section", "page"])
            writer.writerow(["Q?", "A.", "yakmel-overview.md", "yakmel.md"])
            writer.writerow(["Q2?", "A2.", "sword-combat.md", "sword.md"])
        result = _load_processed_pages(tmp_path)
        assert result == {"yakmel", "sword"}


# ---------------------------------------------------------------------------
# generate_for_page
# ---------------------------------------------------------------------------


class TestGenerateForPage:
    def test_generates_qa_pairs(self, tmp_path):
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        output = tmp_path / "output"
        _write_golden_pages(golden, ["yakmel.md"])
        _populate(sections, {"yakmel-overview.md": "Yakmel is fluffy."})

        g = TestsetGenerator(_make_router(), golden, sections, output, seed=42)
        result = g.generate_for_page("yakmel.md")

        assert result is not None
        assert result["FACTOID_Q"] == "What is the price of a Yakmel?"
        assert result["section"] == "yakmel-overview.md"
        assert result["page"] == "yakmel.md"

    def test_no_sections_returns_none(self, tmp_path):
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        sections.mkdir()
        output = tmp_path / "output"
        _write_golden_pages(golden, ["yakmel.md"])

        g = TestsetGenerator(_make_router(), golden, sections, output)
        assert g.generate_for_page("yakmel.md") is None

    def test_unparseable_response_returns_none(self, tmp_path):
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        output = tmp_path / "output"
        _write_golden_pages(golden, ["yakmel.md"])
        _populate(sections, {"yakmel-overview.md": "content"})

        router = _make_router("This is not a valid response.")
        g = TestsetGenerator(router, golden, sections, output)
        assert g.generate_for_page("yakmel.md") is None

    def test_prompt_includes_page_and_section(self, tmp_path):
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        output = tmp_path / "output"
        _write_golden_pages(golden, ["yakmel.md"])
        _populate(sections, {"yakmel-obtaining.md": "You can find Yakmel."})

        router = _make_router()
        g = TestsetGenerator(router, golden, sections, output, seed=42)
        g.generate_for_page("yakmel.md")

        prompt_sent = router.generate.call_args[0][0]
        assert "Page: yakmel" in prompt_sent
        assert "Section: obtaining" in prompt_sent
        assert "You can find Yakmel." in prompt_sent


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------


class TestGenerateAll:
    def test_processes_golden_pages(self, tmp_path):
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        output = tmp_path / "output"
        _write_golden_pages(golden, ["yakmel.md", "sword.md"])
        _populate(sections, {
            "yakmel-overview.md": "Yakmel content.",
            "sword-combat.md": "Sword content.",
        })

        router = _make_router()
        g = TestsetGenerator(router, golden, sections, output, seed=42)
        results = g.generate_all(max_questions=0)

        assert len(results) == 2
        assert router.generate.call_count == 2

    def test_writes_csv_files(self, tmp_path):
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        output = tmp_path / "output"
        _write_golden_pages(golden, ["yakmel.md"])
        _populate(sections, {"yakmel-overview.md": "content"})

        g = TestsetGenerator(_make_router(), golden, sections, output, seed=42)
        g.generate_all(max_questions=0)

        for name in ("factoid.csv", "conceptual.csv", "messy.csv"):
            csv_path = output / name
            assert csv_path.exists()
            with csv_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["page"] == "yakmel.md"
            assert rows[0]["question"]
            assert rows[0]["answer"]
            assert rows[0]["section"]

    def test_skips_already_processed(self, tmp_path):
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        output = tmp_path / "output"
        _write_golden_pages(golden, ["yakmel.md", "sword.md"])
        _populate(sections, {
            "yakmel-overview.md": "content",
            "sword-combat.md": "content",
        })

        # Pre-populate factoid.csv with yakmel
        output.mkdir(parents=True)
        with (output / "factoid.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer", "section", "page"])
            writer.writerow(["Q?", "A.", "yakmel-overview.md", "yakmel.md"])

        router = _make_router()
        g = TestsetGenerator(router, golden, sections, output, seed=42)
        results = g.generate_all(max_questions=0)

        assert len(results) == 1
        assert results[0]["page"] == "sword.md"
        router.generate.assert_called_once()

    def test_force_regenerates(self, tmp_path):
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        output = tmp_path / "output"
        _write_golden_pages(golden, ["yakmel.md"])
        _populate(sections, {"yakmel-overview.md": "content"})

        # Pre-populate
        output.mkdir(parents=True)
        with (output / "factoid.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer", "section", "page"])
            writer.writerow(["old Q?", "old A.", "yakmel-overview.md", "yakmel.md"])

        router = _make_router()
        g = TestsetGenerator(router, golden, sections, output, seed=42)
        results = g.generate_all(force=True, max_questions=0)

        assert len(results) == 1
        router.generate.assert_called_once()

    def test_max_questions_limits_processed(self, tmp_path):
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        output = tmp_path / "output"
        _write_golden_pages(golden, ["a.md", "b.md", "c.md", "d.md"])
        _populate(sections, {
            "a-overview.md": "content",
            "b-overview.md": "content",
            "c-overview.md": "content",
            "d-overview.md": "content",
        })

        router = _make_router()
        g = TestsetGenerator(router, golden, sections, output, seed=42)
        results = g.generate_all(max_questions=2)

        assert len(results) == 2
        assert router.generate.call_count == 2

    def test_max_questions_zero_processes_all(self, tmp_path):
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        output = tmp_path / "output"
        _write_golden_pages(golden, ["a.md", "b.md", "c.md"])
        _populate(sections, {
            "a-overview.md": "content",
            "b-overview.md": "content",
            "c-overview.md": "content",
        })

        router = _make_router()
        g = TestsetGenerator(router, golden, sections, output, seed=42)
        results = g.generate_all(max_questions=0)

        assert len(results) == 3

    def test_empty_golden_pages(self, tmp_path):
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        sections.mkdir()
        output = tmp_path / "output"
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text("", encoding="utf-8")

        g = TestsetGenerator(_make_router(), golden, sections, output)
        results = g.generate_all()
        assert results == []

    def test_continues_on_single_page_failure(self, tmp_path):
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        output = tmp_path / "output"
        _write_golden_pages(golden, ["aaa.md", "bbb.md", "ccc.md"])
        _populate(sections, {
            "aaa-overview.md": "good",
            "bbb-overview.md": "bad",
            "ccc-overview.md": "good",
        })

        router = MagicMock()
        router.generate.side_effect = [
            _VALID_RESPONSE,
            RuntimeError("model exploded"),
            _VALID_RESPONSE,
        ]

        g = TestsetGenerator(router, golden, sections, output, seed=42)
        results = g.generate_all(max_questions=0)

        assert len(results) == 2
        pages = [r["page"] for r in results]
        assert "aaa.md" in pages
        assert "ccc.md" in pages

    def test_skipped_dont_count_toward_max(self, tmp_path):
        """Skipped (already processed) pages don't count toward max_questions."""
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        output = tmp_path / "output"
        _write_golden_pages(golden, ["aaa.md", "bbb.md", "ccc.md", "ddd.md"])
        _populate(sections, {
            "aaa-overview.md": "done",
            "bbb-overview.md": "content",
            "ccc-overview.md": "content",
            "ddd-overview.md": "content",
        })

        # Pre-populate aaa
        output.mkdir(parents=True)
        with (output / "factoid.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer", "section", "page"])
            writer.writerow(["Q?", "A.", "aaa-overview.md", "aaa.md"])

        router = _make_router()
        g = TestsetGenerator(router, golden, sections, output, seed=42)
        results = g.generate_all(max_questions=2)

        assert len(results) == 2
        assert all(r["page"] != "aaa.md" for r in results)
        assert router.generate.call_count == 2

    def test_csv_headers(self, tmp_path):
        """CSV files have the correct header row."""
        golden = tmp_path / "golden.txt"
        sections = tmp_path / "sections"
        output = tmp_path / "output"
        _write_golden_pages(golden, ["yakmel.md"])
        _populate(sections, {"yakmel-overview.md": "content"})

        g = TestsetGenerator(_make_router(), golden, sections, output, seed=42)
        g.generate_all(max_questions=0)

        for name in ("factoid.csv", "conceptual.csv", "messy.csv"):
            with (output / name).open(encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
            assert header == ["question", "answer", "section", "page"]
