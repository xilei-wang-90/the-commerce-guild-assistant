"""Launch the Chainlit web UI for the Sandrock Knowledge Assistant.

Starts a local Chainlit server that serves the interactive RAG chatbot.
Requires Ollama running locally with ``all-minilm`` pulled and
``GEMINI_API_KEY`` in the environment (or ``.env``).

Run with:

    python3 scripts/run_web.py
    python3 scripts/run_web.py --host 0.0.0.0 --port 8080

Or directly via chainlit::

    chainlit run src/guild_assistant_web/app.py -w
"""

import argparse
import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent.parent / "src" / "guild_assistant_web" / "app.py"

DEFAULT_HOST = "localhost"
DEFAULT_PORT = "8000"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Chainlit web UI for the Sandrock Knowledge Assistant.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host to bind to (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Enable auto-reload on file changes.",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "chainlit", "run", str(APP_PATH),
        "--host", args.host,
        "--port", args.port,
    ]
    if args.watch:
        cmd.append("-w")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
