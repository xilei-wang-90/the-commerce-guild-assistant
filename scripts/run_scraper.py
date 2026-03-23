#!/usr/bin/env python3
"""Entry point for the wiki scraper.

Run from the project root:
    python3 scripts/run_scraper.py
"""

import logging
import sys
from pathlib import Path
from queue import Queue

# Allow imports from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guild_assistant.scraper.discoverer import Discoverer
from guild_assistant.scraper.worker import Worker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
MAX_PAGES = 0
NUM_WORKERS = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    url_queue: Queue = Queue()

    discoverer = Discoverer(
        url_queue=url_queue,
        max_pages=MAX_PAGES,
        num_workers=NUM_WORKERS,
    )

    workers = [
        Worker(worker_id=i, url_queue=url_queue, output_dir=OUTPUT_DIR)
        for i in range(NUM_WORKERS)
    ]

    discoverer.start()
    for worker in workers:
        worker.start()

    discoverer.join()
    for worker in workers:
        worker.join()

    logging.info("All done. Pages saved to: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
