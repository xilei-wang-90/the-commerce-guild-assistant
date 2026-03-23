"""Load retrieval test cases from CSV files.

Provides a shared ``TestCase`` dataclass and ``load_testset()`` function
used by the retrieval evaluator and any future evaluation tools.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single retrieval test case."""

    question: str
    answer: str
    section: str
    page: str


def load_testset(csv_path: str | Path) -> list[TestCase]:
    """Read a testset CSV and return a list of :class:`TestCase` instances.

    The CSV must have columns: ``question``, ``answer``, ``section``, ``page``.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Testset file not found: %s", path)
        return []

    cases: list[TestCase] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(
                TestCase(
                    question=row["question"],
                    answer=row["answer"],
                    section=row["section"],
                    page=row["page"],
                )
            )
    logger.info("Loaded %d test cases from %s", len(cases), path.name)
    return cases
