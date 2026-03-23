"""Break silver-tier Markdown files into type-aware chunks.

Reads ``.md`` files from ``data/silver/``, classifies each page by type
(item, character, location, etc.), and writes per-chunk files to
``data/silver-sections/``.

Each L2 section (with all L3+ subsections) forms one chunk.  Sections
designated as "Overview" for the page's type are merged into a single
Overview chunk (``<page_slug>-overview.md``); remaining L2 sections
become standalone chunks (``<page_slug>-<section_slug>.md``).

Run with::

    python3 scripts/run_section_breaker.py

Use ``--force`` to re-break files that already have output, and
``--max-files`` to cap the number of source files processed per run.
"""

import argparse
import logging
from pathlib import Path

from guild_assistant.rag_setup.section_breaker import SectionBreaker

INPUT_DIR = Path(__file__).parent.parent / "data" / "silver"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "silver-sections"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break silver-tier Markdown files into type-aware chunks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-break files even if output already exists.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of source files to process per run (0 = unlimited, default: 10).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    breaker = SectionBreaker(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    results = breaker.break_all(force=args.force, max_files=args.max_files)

    total_chunks = sum(len(v) for v in results.values())
    logging.getLogger(__name__).info(
        "Done — %d files processed, %d chunk files written",
        len(results),
        total_chunks,
    )


if __name__ == "__main__":
    main()
