"""Generate a golden dataset for retrieval testing.

Scans silver-tier Markdown files, classifies each page by type, and
randomly selects up to 10 pages per type (12 types × 10 = 120 max).
Writes the selected filenames to ``data/test-data/golden_pages.txt``.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the package is importable when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from guild_assistant.rag_test.golden_dataset import write_golden_dataset

INPUT_DIR = "data/silver/"
OUTPUT_DIR = "data/test-data/"
OUTPUT_FILENAME = "golden_pages.txt"
PER_TYPE = 10


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a golden dataset for retrieval testing."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: non-deterministic)",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=PER_TYPE,
        help=f"Max pages per type (default: {PER_TYPE})",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Silver-tier input directory (default: {INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_file = write_golden_dataset(
        silver_dir=args.input_dir,
        output_dir=args.output_dir,
        output_filename=OUTPUT_FILENAME,
        per_type=args.per_type,
        seed=args.seed,
    )
    print(f"Golden dataset written to {output_file}")


if __name__ == "__main__":
    main()
