"""Tests for tests.benchmark.testset_loader."""

import csv
from pathlib import Path

from tests.benchmark.testset_loader import TestCase, load_testset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "section", "page"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# load_testset
# ---------------------------------------------------------------------------


class TestLoadTestset:
    def test_loads_cases(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        _write_csv(csv_path, [
            {"question": "Q1?", "answer": "A1.", "section": "s1.md", "page": "p1.md"},
            {"question": "Q2?", "answer": "A2.", "section": "s2.md", "page": "p2.md"},
        ])
        cases = load_testset(csv_path)
        assert len(cases) == 2
        assert isinstance(cases[0], TestCase)
        assert cases[0].question == "Q1?"
        assert cases[0].page == "p1.md"
        assert cases[1].section == "s2.md"

    def test_missing_file_returns_empty(self, tmp_path):
        cases = load_testset(tmp_path / "nonexistent.csv")
        assert cases == []

    def test_empty_csv_returns_empty(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        _write_csv(csv_path, [])
        cases = load_testset(csv_path)
        assert cases == []

    def test_preserves_all_fields(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        _write_csv(csv_path, [{
            "question": "What is X?",
            "answer": "X is Y.",
            "section": "yakmel-overview.md",
            "page": "yakmel.md",
        }])
        case = load_testset(csv_path)[0]
        assert case.question == "What is X?"
        assert case.answer == "X is Y."
        assert case.section == "yakmel-overview.md"
        assert case.page == "yakmel.md"
