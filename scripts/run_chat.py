"""Interactive chat with the Sandrock knowledge assistant.

Launches a REPL that accepts user questions, runs them through the full
RAG pipeline (retrieve → augment → generate), and prints the answer.

Requires ``--mode`` to select which ChromaDB collection to query:

``summary``
    Retrieves from the ``sandrock_wiki_summary`` collection.  Each match
    resolves to a full-text page in ``data/silver/``.

``section-reverse-hyde``
    Retrieves from the ``sandrock_wiki_section_reverse_hyde`` collection.
    Each match resolves to a per-section file in ``data/silver-sections/``.

If ``--mode`` is omitted the script interactively prompts the user to
choose.

Requires Ollama running locally with ``all-minilm`` (embedding) and a
text-generation model pulled.  For large prompts the router falls back
to Gemini (needs ``GEMINI_API_KEY`` in the environment or ``.env``).

Run with:

    python3 scripts/run_chat.py --mode summary
    python3 scripts/run_chat.py --mode section-reverse-hyde
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from guild_assistant.rag.context_builder import ContextBuilder
from guild_assistant.rag.pipeline import QueryPipeline
from guild_assistant.rag.reranker import Reranker
from guild_assistant.rag.retriever import Retriever
from guild_assistant.utils.model_adapter import (
    CrossEncoderRerankerAdapter,
    GeminiAdapter,
    OllamaEmbeddingAdapter,
)

VALID_MODES = ("summary", "section-reverse-hyde")

_MODE_CONFIG: dict[str, str] = {
    "summary": "sandrock_wiki_summary",
    "section-reverse-hyde": "sandrock_wiki_section_reverse_hyde",
}

DB_PATH = Path(__file__).parent.parent / "sandrock_db"
OLLAMA_URL = "http://localhost:11434"
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
EMBEDDING_MODEL = "all-minilm"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _prompt_mode() -> str:
    """Interactively ask the user to choose a valid retrieval mode."""
    choices = " / ".join(VALID_MODES)
    while True:
        try:
            value = input(f"Choose retrieval mode [{choices}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)
        if value in VALID_MODES:
            return value
        print(f"Invalid mode '{value}'. Please enter one of: {choices}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Sandrock knowledge assistant (RAG chat).",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default=None,
        help=(
            "Retrieval mode: 'summary' queries the sandrock_wiki_summary "
            "collection (full pages); 'section-reverse-hyde' queries "
            "sandrock_wiki_section_reverse_hyde (per-section chunks). "
            "Prompted interactively if omitted."
        ),
    )
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_URL,
        help=f"Ollama server base URL (default: {OLLAMA_URL}).",
    )
    args = parser.parse_args()

    mode = args.mode if args.mode in VALID_MODES else _prompt_mode()
    collection_name = _MODE_CONFIG[mode]

    load_dotenv()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    embedding_adapter = OllamaEmbeddingAdapter(
        model_name=EMBEDDING_MODEL, base_url=args.ollama_url,
    )
    retriever = Retriever(
        db_path=DB_PATH,
        collection_name=collection_name,
        embedding_adapter=embedding_adapter,
        n_results=10,
    )
    context_builder = ContextBuilder()
    reranker_adapter = CrossEncoderRerankerAdapter(model_name=RERANKER_MODEL)
    reranker = Reranker(adapter=reranker_adapter, top_n=3)

    model = GeminiAdapter(model_name=GEMINI_MODEL)

    pipeline = QueryPipeline(
        retriever=retriever,
        context_builder=context_builder,
        model=model,
        reranker=reranker,
    )

    print(f"Sandrock Knowledge Assistant (mode: {mode})")
    print("Type your question, or 'quit'/'exit' to stop.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit"}:
                break
            answer = pipeline.query(user_input)
            print(f"\nAssistant: {answer}\n")
    except (KeyboardInterrupt, EOFError):
        print()

    print("Goodbye!")


if __name__ == "__main__":
    main()
