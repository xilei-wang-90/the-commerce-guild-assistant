"""Prepend metadata tags to reverse-hyde question files.

Reads hypothetical question files from ``data/reverse-hyde/`` and the
corresponding silver-section Markdown files from ``data/silver-sections/``.
For each question file, extracts the page slug from the filename and all
L2/L3 heading titles from the section content, then writes a tagged copy
to ``data/tagged-reverse-hyde/`` with a bracketed tag line prepended.

The tagged files are intended for embedding into a separate ChromaDB
collection (via ``run_embedder.py --mode section-tagged-reverse-hyde``)
to test whether the extra metadata improves retrieval quality.

No LLM calls are made — this is a pure text-processing step.

Run with:

    python3 scripts/run_question_tagger.py
    python3 scripts/run_question_tagger.py --force --max-files 0
"""

import argparse
import logging
from pathlib import Path

from guild_assistant.rag_setup.question_tagger import QuestionTagger

_ROOT = Path(__file__).parent.parent

QUESTIONS_DIR = _ROOT / "data" / "reverse-hyde"
SECTIONS_DIR = _ROOT / "data" / "silver-sections"
OUTPUT_DIR = _ROOT / "data" / "tagged-reverse-hyde"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepend metadata tags to reverse-hyde question files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-tag files even if output already exists.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of files to tag per run (0 = unlimited, default: 10).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    tagger = QuestionTagger(
        questions_dir=QUESTIONS_DIR,
        sections_dir=SECTIONS_DIR,
        output_dir=OUTPUT_DIR,
    )

    results = tagger.tag_all(force=args.force, max_files=args.max_files)
    logging.getLogger(__name__).info("Done — %d files tagged", len(results))


if __name__ == "__main__":
    main()
