"""Embed wiki content into a persistent ChromaDB vector store.

Supports two embedding modes selected via ``--mode``:

``summary``
    Reads ``.md`` files from ``data/summaries/``, embeds each one, and stores
    the vectors in the ``sandrock_wiki_summary`` ChromaDB collection.  Each
    entry links back to the corresponding full-text file in ``data/silver/``.
    This is the default production mode for summary-based retrieval.

``section-reverse-hyde``
    Reads hypothetical question files from ``data/reverse-hyde/``, embeds each
    one, and stores the vectors in the ``sandrock_wiki_section_reverse_hyde``
    ChromaDB collection.  Each entry links back to the corresponding
    full-text file in ``data/silver-sections/``.  This implements Reverse HyDE
    indexing: embedding questions a document answers improves semantic match
    quality for conversational queries.

If ``--mode`` is omitted or an unrecognised value is given the script
interactively prompts the user to choose.

Requires Ollama to be running locally with the ``all-minilm`` model pulled:

    ollama pull all-minilm

Run with:

    python3 scripts/run_embedder.py --mode summary
    python3 scripts/run_embedder.py --mode section-reverse-hyde

Use ``--force`` to re-embed files that are already in the database.
"""

import argparse
import logging
import sys
from pathlib import Path

from guild_assistant.rag_setup.embedder import Embedder
from guild_assistant.utils.model_adapter import OllamaEmbeddingAdapter

_ROOT = Path(__file__).parent.parent

VALID_MODES = ("summary", "section-reverse-hyde")

_MODE_CONFIG: dict[str, dict[str, object]] = {
    "summary": {
        "sources_dir": _ROOT / "data" / "summaries",
        "silver_dir": _ROOT / "data" / "silver",
        "collection_name": "sandrock_wiki_summary",
    },
    "section-reverse-hyde": {
        "sources_dir": _ROOT / "data" / "reverse-hyde",
        "silver_dir": _ROOT / "data" / "silver-sections",
        "collection_name": "sandrock_wiki_section_reverse_hyde",
    },
}

DB_PATH = _ROOT / "sandrock_db"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "all-minilm"


def _prompt_mode() -> str:
    """Interactively ask the user to choose a valid embedding mode."""
    choices = " / ".join(VALID_MODES)
    while True:
        try:
            value = input(f"Choose embedding mode [{choices}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)
        if value in VALID_MODES:
            return value
        print(f"Invalid mode '{value}'. Please enter one of: {choices}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed wiki content into ChromaDB via Ollama all-minilm.",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default=None,
        help=(
            "Embedding mode: 'summary' embeds LLM summaries (data/summaries → "
            "sandrock_wiki_summary); 'section-reverse-hyde' embeds hypothetical "
            "questions (data/reverse-hyde → sandrock_wiki_section_reverse_hyde). "
            "Prompted interactively if omitted."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed files even if they are already in the database.",
    )
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_URL,
        help=f"Ollama server base URL (default: {OLLAMA_URL}).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=10,
        help="Maximum number of files to embed in this run (0 = unlimited, default: 10).",
    )
    args = parser.parse_args()

    mode = args.mode if args.mode in VALID_MODES else _prompt_mode()
    config = _MODE_CONFIG[mode]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)
    log.info("Embedding mode: %s", mode)

    adapter = OllamaEmbeddingAdapter(model_name=EMBEDDING_MODEL, base_url=args.ollama_url)
    embedder = Embedder(
        sources_dir=config["sources_dir"],
        silver_dir=config["silver_dir"],
        db_path=DB_PATH,
        embedding_adapter=adapter,
        collection_name=config["collection_name"],
    )

    results = embedder.embed_all(force=args.force, max_records=args.max_records)
    log.info("Done — %d files embedded", len(results))


if __name__ == "__main__":
    main()
