"""Tests for guild_assistant.rag_test.golden_dataset."""

from pathlib import Path

import pytest

from guild_assistant.rag_test.golden_dataset import (
    select_golden_pages,
    write_golden_dataset,
)
from guild_assistant.utils.page_classifier import PageType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate(directory: Path, files: dict[str, str]) -> None:
    """Write *files* mapping ``{name: content}`` into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")


def _item_page(name: str = "thing") -> str:
    return f"# {name}\nA useful item.\n## Obtaining\nCraft it."


def _character_page(name: str = "npc") -> str:
    return f"# {name}\nA friendly person.\n## Biographical Information\nAge: 20."


def _mission_page(name: str = "Mission: Go") -> str:
    return f"# {name}\nMain quest.\n## Walkthrough\nStep 1."


def _generic_page(name: str = "page") -> str:
    return f"# {name}\nSome content.\n## Trivia\nFun facts."


# ---------------------------------------------------------------------------
# select_golden_pages
# ---------------------------------------------------------------------------


class TestSelectGoldenPages:
    def test_empty_directory(self, tmp_path):
        silver = tmp_path / "silver"
        silver.mkdir()
        result = select_golden_pages(silver)
        assert result == []

    def test_single_type_under_limit(self, tmp_path):
        """All pages selected when fewer than per_type."""
        silver = tmp_path / "silver"
        _populate(silver, {
            "sword.md": _item_page("Sword"),
            "shield.md": _item_page("Shield"),
        })
        result = select_golden_pages(silver, per_type=10)
        assert sorted(result) == ["shield.md", "sword.md"]

    def test_single_type_over_limit(self, tmp_path):
        """Only per_type pages selected when more are available."""
        silver = tmp_path / "silver"
        files = {f"item_{i}.md": _item_page(f"Item {i}") for i in range(15)}
        _populate(silver, files)

        result = select_golden_pages(silver, per_type=5, seed=42)
        assert len(result) == 5
        # All selected files should be from the input set
        assert all(f in files for f in result)

    def test_multiple_types(self, tmp_path):
        """Pages from different types are all represented."""
        silver = tmp_path / "silver"
        _populate(silver, {
            "sword.md": _item_page("Sword"),
            "mi_an.md": _character_page("Mi-an"),
            "mission_go.md": _mission_page("Go"),
            "random.md": _generic_page("Random"),
        })
        result = select_golden_pages(silver, per_type=10)
        assert len(result) == 4
        assert "sword.md" in result
        assert "mi_an.md" in result
        assert "mission_go.md" in result
        assert "random.md" in result

    def test_seed_reproducibility(self, tmp_path):
        """Same seed produces the same selection."""
        silver = tmp_path / "silver"
        files = {f"item_{i}.md": _item_page(f"Item {i}") for i in range(20)}
        _populate(silver, files)

        result1 = select_golden_pages(silver, per_type=5, seed=123)
        result2 = select_golden_pages(silver, per_type=5, seed=123)
        assert result1 == result2

    def test_different_seeds_differ(self, tmp_path):
        """Different seeds produce different selections (with enough items)."""
        silver = tmp_path / "silver"
        files = {f"item_{i}.md": _item_page(f"Item {i}") for i in range(50)}
        _populate(silver, files)

        result1 = select_golden_pages(silver, per_type=5, seed=1)
        result2 = select_golden_pages(silver, per_type=5, seed=2)
        assert result1 != result2

    def test_result_is_sorted(self, tmp_path):
        silver = tmp_path / "silver"
        _populate(silver, {
            "zzz.md": _generic_page("Z"),
            "aaa.md": _generic_page("A"),
            "mmm.md": _generic_page("M"),
        })
        result = select_golden_pages(silver, per_type=10)
        assert result == sorted(result)

    def test_max_120_pages(self, tmp_path):
        """With 12 types × 10 per type, at most 120 pages are selected."""
        silver = tmp_path / "silver"
        files = {}
        # Create 15 items
        for i in range(15):
            files[f"item_{i}.md"] = _item_page(f"Item {i}")
        # Create 15 characters
        for i in range(15):
            files[f"char_{i}.md"] = _character_page(f"Char {i}")
        # Create 15 missions
        for i in range(15):
            files[f"mission_{i}.md"] = _mission_page(f"Mission {i}")
        # Create 15 generic
        for i in range(15):
            files[f"generic_{i}.md"] = _generic_page(f"Generic {i}")
        _populate(silver, files)

        result = select_golden_pages(silver, per_type=10, seed=42)
        # 4 types × 10 = 40 max
        assert len(result) <= 40
        # Each type should have at most 10
        from collections import Counter
        from guild_assistant.utils.page_classifier import classify_file

        type_counts: dict[PageType, int] = Counter()
        for f in result:
            ptypes = classify_file(silver / f)
            type_counts[ptypes[0]] += 1
        for ptype, count in type_counts.items():
            assert count <= 10, f"{ptype.value} has {count} pages, expected <= 10"

    def test_per_type_one(self, tmp_path):
        """per_type=1 selects exactly one per type."""
        silver = tmp_path / "silver"
        _populate(silver, {
            "sword.md": _item_page("Sword"),
            "axe.md": _item_page("Axe"),
            "mi_an.md": _character_page("Mi-an"),
            "owen.md": _character_page("Owen"),
        })
        result = select_golden_pages(silver, per_type=1, seed=42)
        assert len(result) == 2  # 1 item + 1 character


# ---------------------------------------------------------------------------
# write_golden_dataset
# ---------------------------------------------------------------------------


class TestWriteGoldenDataset:
    def test_creates_output_file(self, tmp_path):
        silver = tmp_path / "silver"
        output = tmp_path / "output"
        _populate(silver, {"sword.md": _item_page("Sword")})

        result_path = write_golden_dataset(silver, output, seed=42)

        assert result_path.exists()
        assert result_path.name == "golden_pages.txt"
        content = result_path.read_text(encoding="utf-8")
        assert "sword.md" in content

    def test_creates_output_dir(self, tmp_path):
        silver = tmp_path / "silver"
        output = tmp_path / "deep" / "nested" / "output"
        _populate(silver, {"sword.md": _item_page("Sword")})

        result_path = write_golden_dataset(silver, output, seed=42)
        assert output.is_dir()
        assert result_path.exists()

    def test_one_filename_per_line(self, tmp_path):
        silver = tmp_path / "silver"
        output = tmp_path / "output"
        _populate(silver, {
            "sword.md": _item_page("Sword"),
            "mi_an.md": _character_page("Mi-an"),
        })

        result_path = write_golden_dataset(silver, output, seed=42)
        lines = result_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_empty_input_writes_empty_file(self, tmp_path):
        silver = tmp_path / "silver"
        silver.mkdir()
        output = tmp_path / "output"

        result_path = write_golden_dataset(silver, output)
        content = result_path.read_text(encoding="utf-8")
        assert content == ""

    def test_custom_output_filename(self, tmp_path):
        silver = tmp_path / "silver"
        output = tmp_path / "output"
        _populate(silver, {"sword.md": _item_page("Sword")})

        result_path = write_golden_dataset(
            silver, output, output_filename="custom.txt", seed=42
        )
        assert result_path.name == "custom.txt"
        assert result_path.exists()
