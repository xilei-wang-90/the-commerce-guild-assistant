"""Drop a ChromaDB collection from the local vector store.

Deletes the named collection and all its embeddings from the persisted
ChromaDB database.  Use with care — this is irreversible without re-running
the embedder.

Run with:

    python3 scripts/drop_collection.py
    python3 scripts/drop_collection.py --collection my_other_collection
"""

import argparse
import logging
from pathlib import Path

import chromadb

DB_PATH = Path(__file__).parent.parent / "sandrock_db"
DEFAULT_COLLECTION = "sandrock_wiki_summary"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop a ChromaDB collection from the local vector store.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Name of the collection to drop (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--db-path",
        default=str(DB_PATH),
        help=f"Path to the ChromaDB database directory (default: {DB_PATH}).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    client = chromadb.PersistentClient(path=args.db_path)
    existing = [c.name for c in client.list_collections()]

    if args.collection not in existing:
        log.error("Collection %r not found in %s", args.collection, args.db_path)
        raise SystemExit(1)

    client.delete_collection(args.collection)
    log.info("Dropped collection %r from %s", args.collection, args.db_path)


if __name__ == "__main__":
    main()
