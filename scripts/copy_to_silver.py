#!/usr/bin/env python3
"""Copy filtered markdown files from data/raw/ to data/silver/.

Run from the project root:
    python3 scripts/copy_to_silver.py
"""

import logging
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
SILVER_DIR = Path(__file__).parent.parent / "data" / "silver"

EXCLUDED_FILES = {
    "changelist.md",
    "changelist_pre_release.md",
    "console_changelist.md",
    "kickstarter.md",
    "my_time_at_sandrock_wiki.md",
    "my_time_at_sandrock.md",
    "my_time_series.md",
    "panthea_games.md",
    "roadmap.md",
    "sandbox.md",
    "save_files.md",
    "sweetvetch.md",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    if not RAW_DIR.exists():
        logging.error("Source directory does not exist: %s", RAW_DIR)
        return

    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(RAW_DIR.glob("*.md"))
    copied = 0

    for src in md_files:
        if src.name in EXCLUDED_FILES:
            logging.info("Skipping excluded file: %s", src.name)
            continue

        dst = SILVER_DIR / src.name
        if "buyback" in src.stem:
            content = src.read_text(encoding="utf-8")
            dst.write_text(f"# {src.stem}\n\n{content}", encoding="utf-8")
        else:
            shutil.copy2(src, dst)
        copied += 1
        logging.info("Copied: %s", src.name)

    logging.info(
        "Done. %d file(s) copied to %s (%d excluded).",
        copied,
        SILVER_DIR,
        len(md_files) - copied,
    )


if __name__ == "__main__":
    main()
