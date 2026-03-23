"""Preview which model will be used for each silver-tier Markdown file.

Reads every ``.md`` file from ``data/silver/``, builds the same prompt that
the summarizer would send, counts its tokens, and prints the routing decision.
No model API calls are made.

Run with:
    python3 scripts/preview_model_routing.py
"""

import logging
from pathlib import Path

from guild_assistant.utils.model_adapter import GeminiAdapter, OllamaAdapter
from guild_assistant.utils.router import ModelRouter
from guild_assistant.rag_setup.summarizer import _PROMPT_TEMPLATE

INPUT_DIR = Path(__file__).parent.parent / "data" / "silver"
MODEL_NAME = "sandrock-model"
OLLAMA_URL = "http://localhost:11434"
GEMINI_MODEL = "gemini-3-flash-preview"

_COL_FILE = 55
_COL_TOKENS = 8


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    md_files = sorted(INPUT_DIR.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {INPUT_DIR}")
        return

    local = OllamaAdapter(model_name=MODEL_NAME, base_url=OLLAMA_URL)
    # GeminiAdapter can be constructed without a valid API key for routing only.
    cloud = GeminiAdapter(model_name=GEMINI_MODEL)
    router = ModelRouter(local=local, cloud=cloud)

    header = f"{'File':<{_COL_FILE}} {'Tokens':>{_COL_TOKENS}}  Model"
    print(header)
    print("-" * len(header))

    local_count = 0
    cloud_count = 0

    for file_path in md_files:
        content = file_path.read_text(encoding="utf-8")
        prompt = _PROMPT_TEMPLATE.format(content=content)
        selected = router.route(prompt)
        token_count = len(router._enc.encode(prompt))

        print(
            f"{file_path.name:<{_COL_FILE}} {token_count:>{_COL_TOKENS}}  {selected.model_name}"
        )

        if selected is local:
            local_count += 1
        else:
            cloud_count += 1

    print()
    print(
        f"Total: {len(md_files)} files — "
        f"{local_count} → {MODEL_NAME} (local), "
        f"{cloud_count} → {GEMINI_MODEL} (cloud)"
    )


if __name__ == "__main__":
    main()
