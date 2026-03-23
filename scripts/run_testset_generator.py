"""Generate retrieval test sets from golden pages.

Reads ``data/test-data/golden_pages.txt``, picks one random section per page
from ``data/silver-sections/``, and asks an LLM to generate factoid,
conceptual, and messy question/answer pairs.  Results are written to three
CSV files under ``data/test-data/``.

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
from guild_assistant.rag_test.testset_generator import TestsetGenerator

GOLDEN_PAGES_PATH = Path(__file__).parent.parent / "data" / "test-data" / "golden_pages.txt"
SECTIONS_DIR = Path(__file__).parent.parent / "data" / "silver-sections"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "test-data"
MODEL_NAME = "sandrock-model"
OLLAMA_URL = "http://localhost:11434"
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate retrieval test sets from golden pages.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate Q/A pairs even if output already exists.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of pages to process per run (0 = unlimited, default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible section selection (default: non-deterministic).",
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

    generator = TestsetGenerator(
        router=router,
        golden_pages_path=GOLDEN_PAGES_PATH,
        sections_dir=SECTIONS_DIR,
        output_dir=OUTPUT_DIR,
        seed=args.seed,
    )

    results = generator.generate_all(
        force=args.force,
        max_questions=args.max_questions,
    )
    logging.getLogger(__name__).info(
        "Done — %d pages processed", len(results)
    )


if __name__ == "__main__":
    main()
