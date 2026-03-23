"""Generate hypothetical questions for all silver-section Markdown files.

Reads ``.md`` files from ``data/silver-sections/``, routes each prompt to the
appropriate model via ``ModelRouter``, and writes per-file question lists
to ``data/reverse-hyde/``.

This implements the Reverse HyDE indexing strategy: questions that a
document answers are stored alongside it so that conversational queries
match semantically meaningful representations.

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
from guild_assistant.rag_setup.question_generator import QuestionGenerator

INPUT_DIR = Path(__file__).parent.parent / "data" / "silver-sections"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "reverse-hyde"
MODEL_NAME = "sandrock-model"
OLLAMA_URL = "http://localhost:11434"
GEMINI_MODEL = "gemini-3-flash-preview"


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate hypothetical questions for silver-tier Markdown files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate questions even if output already exists.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of files to process per run (0 = unlimited, default: 10).",
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
    generator = QuestionGenerator(
        router=router,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
    )

    results = generator.generate_all(force=args.force, max_files=args.max_files)
    logging.getLogger(__name__).info(
        "Done — %d files processed", len(results)
    )


if __name__ == "__main__":
    main()
