"""Generate LLM summaries for all silver-tier Markdown files.

Reads ``.md`` files from ``data/silver/``, routes each prompt to the
appropriate model via ``ModelRouter``, and writes concise summaries to
``data/summaries/``.

Short prompts (< TOKEN_THRESHOLD tokens) are sent to the local Ollama model.
Longer prompts are sent to the Gemini cloud model (requires ``GEMINI_API_KEY``
in the environment or a ``.env`` file).

Requires Ollama to be running locally (default ``http://localhost:11434``).
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from guild_assistant.utils.model_adapter import GeminiAdapter, OllamaAdapter
from guild_assistant.utils.router import ModelRouter
from guild_assistant.rag_setup.summarizer import Summarizer

INPUT_DIR = Path(__file__).parent.parent / "data" / "silver"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "summaries"
MODEL_NAME = "sandrock-model"
OLLAMA_URL = "http://localhost:11434"
GEMINI_MODEL = "gemini-3-flash-preview"


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate LLM summaries for silver-tier Markdown files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-summarize files even if a summary already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    local = OllamaAdapter(model_name=MODEL_NAME, base_url=OLLAMA_URL)
    cloud = GeminiAdapter(model_name=GEMINI_MODEL)
    router = ModelRouter(local=local, cloud=cloud)
    summarizer = Summarizer(
        router=router,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
    )

    results = summarizer.summarize_all(force=args.force, max_files=58)
    logging.getLogger(__name__).info(
        "Done — %d files summarized", len(results)
    )


if __name__ == "__main__":
    main()
