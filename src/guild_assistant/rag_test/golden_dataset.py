"""Generate a golden dataset for retrieval testing.

Scans a directory of silver-tier Markdown files, classifies each page
by type, and randomly samples up to *per_type* pages per type.  The
result is a flat list of filenames suitable for building a retrieval
evaluation set.
"""

import logging
import random
from collections import defaultdict
from pathlib import Path

from guild_assistant.utils.page_classifier import PageType, classify_file

logger = logging.getLogger(__name__)


def select_golden_pages(
    silver_dir: str | Path,
    per_type: int = 10,
    seed: int | None = None,
) -> list[str]:
    """Select a stratified random sample of silver-tier pages.

    Classifies every ``.md`` file in *silver_dir* by page type, then
    randomly picks up to *per_type* pages from each type.  If a type
    has fewer than *per_type* pages, all of them are selected.

    Parameters
    ----------
    silver_dir:
        Directory containing silver-tier ``.md`` files.
    per_type:
        Maximum number of pages to select per type (default 10).
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    list[str]
        Sorted list of selected filenames (e.g. ``["yakmel.md", ...]``).
    """
    silver_path = Path(silver_dir)
    md_files = sorted(silver_path.glob("*.md"))

    if not md_files:
        logger.warning("No .md files found in %s", silver_path)
        return []

    # Group files by their page type.
    by_type: dict[PageType, list[str]] = defaultdict(list)
    for fp in md_files:
        page_types = classify_file(fp)
        # classify_file returns a single-element list; use the first type.
        ptype = page_types[0]
        by_type[ptype].append(fp.name)

    rng = random.Random(seed)
    selected: list[str] = []

    for ptype in PageType:
        pages = by_type.get(ptype, [])
        if not pages:
            continue
        if len(pages) <= per_type:
            chosen = pages
        else:
            chosen = rng.sample(pages, per_type)
        selected.extend(chosen)
        logger.info(
            "%-12s: %d available, %d selected",
            ptype.value,
            len(pages),
            len(chosen),
        )

    selected.sort()
    logger.info("Total selected: %d pages", len(selected))
    return selected


def write_golden_dataset(
    silver_dir: str | Path,
    output_dir: str | Path,
    output_filename: str = "golden_pages.txt",
    per_type: int = 10,
    seed: int | None = None,
) -> Path:
    """Select golden pages and write the list to a text file.

    Parameters
    ----------
    silver_dir:
        Directory containing silver-tier ``.md`` files.
    output_dir:
        Directory where the output file will be written.
    output_filename:
        Name of the output text file (default ``golden_pages.txt``).
    per_type:
        Maximum pages per type (default 10).
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    Path
        Path to the written output file.
    """
    pages = select_golden_pages(silver_dir, per_type=per_type, seed=seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / output_filename

    output_file.write_text("\n".join(pages) + "\n" if pages else "", encoding="utf-8")
    logger.info("Wrote %d pages to %s", len(pages), output_file)
    return output_file
