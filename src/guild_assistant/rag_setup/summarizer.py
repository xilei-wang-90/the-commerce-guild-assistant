"""Summarize wiki Markdown files using an LLM via a ModelAdapter."""

import logging
from pathlib import Path

from guild_assistant.utils.router import ModelRouter

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a knowledge-base assistant for the video game "My Time at Sandrock".
Below is a wiki article in Markdown format. Produce a concise summary that captures:
- What the subject is (item, character, mission, location, mechanic, etc.)
- Key facts, stats, or relationships
- Why it matters in the game
Constraint: Provide the summary only. Do not include any introductory phrases, conversational filler, or meta-talk (e.g., 'Sure, here is...' or 'In conclusion...'). Start the response immediately with the summary content.

Keep the summary to 5-8 sentences. Do not include markdown formatting.

Article:
{content}"""


class Summarizer:
    """Read ``.md`` files from *input_dir*, summarize them with an LLM,
    and write the summaries to *output_dir*.

    *router* is a :class:`~guild_assistant.utils.router.ModelRouter`
    that dispatches each prompt to the local or cloud model based on its
    token count.
    """

    def __init__(
        self,
        router: ModelRouter,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> None:
        self._router = router
        self._input_dir = Path(input_dir)
        self._output_dir = Path(output_dir)

    def summarize_file(self, file_path: Path) -> str:
        """Summarize a single Markdown file and write the result.

        Returns the summary text.
        """
        content = file_path.read_text(encoding="utf-8")
        prompt = _PROMPT_TEMPLATE.format(content=content)
        summary = self._router.generate(prompt)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / file_path.name
        out_path.write_text(summary, encoding="utf-8")

        logger.info("Summarized %s", file_path.name)
        return summary

    def summarize_all(self, force: bool = False, max_files: int = 10) -> dict[str, str]:
        """Summarize ``.md`` files in *input_dir*.

        Files that already have a corresponding summary in *output_dir* are
        skipped unless *force* is ``True``.  Skipped files do not count toward
        *max_files*.

        *max_files* caps the number of files actually summarized.  Set to
        ``0`` to summarize all eligible files.

        Returns a mapping of ``{filename: summary}``.
        """
        results: dict[str, str] = {}
        md_files = sorted(self._input_dir.glob("*.md"))

        if not md_files:
            logger.warning("No .md files found in %s", self._input_dir)
            return results

        logger.info("Found %d .md files to process", len(md_files))

        for file_path in md_files:
            if max_files and len(results) >= max_files:
                break
            if not force and (self._output_dir / file_path.name).exists():
                logger.info("Skipping %s (summary exists)", file_path.name)
                continue
            try:
                summary = self.summarize_file(file_path)
                results[file_path.name] = summary
            except Exception:
                logger.exception("Failed to summarize %s", file_path.name)

        logger.info(
            "Finished: %d summarized, %d total",
            len(results),
            len(md_files),
        )
        return results
