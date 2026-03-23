"""Generate hypothetical questions for wiki section Markdown files using an LLM.

Produces the question files used by the Reverse HyDE (Hypothetical Document
Embeddings) indexing strategy: for each section file, the LLM lists the
questions it answers. These question files can then be embedded separately
to improve retrieval recall for conversational queries.

Input files are expected to live in ``data/silver-sections/`` and follow the
naming convention ``<page_slug>-<section_name>.md`` produced by
:class:`~guild_assistant.rag_setup.section_breaker.SectionBreaker`.
"""

import logging
from pathlib import Path

from guild_assistant.utils.router import ModelRouter

logger = logging.getLogger(__name__)

_OVERVIEW_PROMPT = """\
You are a knowledge-base assistant for the video game "My Time at Sandrock".
Below is the overview section from a wiki article in Markdown format.
Page: {page_title}

Generate 5 questions that this overview section answers.
Constraints:
- Output questions only, one per line.
- Do not include section headers, opening words, closing remarks, numbering, bullet points, or any other text.
- Each line must be a plain question ending with a question mark.

Overview content:
{content}"""

_SECTION_PROMPT = """\
You are a knowledge-base assistant for the video game "My Time at Sandrock".
Below is a section from a wiki article in Markdown format along with the page overview for context.
Page: {page_title}
Section: {section_title}

Generate 1-3 questions that the TARGET SECTION answers.
Constraints:
- Output questions only, one per line.
- Do not include section headers, opening words, closing remarks, numbering, bullet points, or any other text.
- Each line must be a plain question ending with a question mark.
- Generate questions ONLY about the target section below, NOT about the overview.
- Use the overview only to understand context (e.g. what the page is about).

--- OVERVIEW (for context only, do NOT generate questions about this) ---
{overview_content}

--- TARGET SECTION (generate questions about this) ---
{content}"""

_SECTION_NO_OVERVIEW_PROMPT = """\
You are a knowledge-base assistant for the video game "My Time at Sandrock".
Below is a section from a wiki article in Markdown format.
Page: {page_title}
Section: {section_title}

Generate 1-3 questions that this section answers.
Constraints:
- Output questions only, one per line.
- Do not include section headers, opening words, closing remarks, numbering, bullet points, or any other text.
- Each line must be a plain question ending with a question mark.

Section content:
{content}"""


def _is_overview(filename: str) -> bool:
    """Return ``True`` if *filename* represents an overview chunk."""
    return Path(filename).stem.endswith("-overview")


def _parse_chunk_filename(filename: str) -> tuple[str, str]:
    """Derive page slug and section name from *filename*.

    The filename convention is ``<page_slug>-<section_name>.md`` where
    ``-`` separates the page slug from the section name and ``_``
    separates words within each part.  Each chunk corresponds to a
    single H2 section (or the merged overview).

    Returns ``(page_slug, section_name)``.  *section_name* is empty
    when the filename contains no ``-`` separator.

    Examples::

        >>> _parse_chunk_filename("yakmel-obtaining.md")
        ('yakmel', 'obtaining')
        >>> _parse_chunk_filename("yakmel-overview.md")
        ('yakmel', 'overview')
        >>> _parse_chunk_filename("yakmel.md")
        ('yakmel', '')
    """
    stem = Path(filename).stem
    page_slug, _, section_name = stem.partition("-")
    return page_slug, section_name


def _overview_path_for(file_path: Path) -> Path:
    """Return the expected overview file path for the same page as *file_path*."""
    page_slug, _ = _parse_chunk_filename(file_path.name)
    return file_path.parent / f"{page_slug}-overview.md"


class QuestionGenerator:
    """Read ``.md`` section files from *input_dir*, generate hypothetical
    questions with an LLM, and write the results to *output_dir*.

    Each file is treated as a single section.  The prompt includes the
    page title and section title (derived from the filename) and asks for
    1-3 questions.

    *router* is a :class:`~guild_assistant.utils.router.ModelRouter` that
    dispatches each prompt to the local or cloud model based on token count.
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

    def generate_for_file(self, file_path: Path) -> str:
        """Generate questions for a single section file and write the result.

        For overview chunks (``*-overview.md``), generates 5 questions using
        only the overview content.  For all other chunks, generates 1-3
        questions and includes the page's overview chunk as context (if it
        exists) so the LLM can better understand the target section.

        Returns the generated questions text.
        """
        content = file_path.read_text(encoding="utf-8")
        page_title, section_title = _parse_chunk_filename(file_path.name)

        if _is_overview(file_path.name):
            prompt = _OVERVIEW_PROMPT.format(
                page_title=page_title,
                content=content,
            )
        else:
            overview_file = _overview_path_for(file_path)
            if overview_file.exists():
                overview_content = overview_file.read_text(encoding="utf-8")
                prompt = _SECTION_PROMPT.format(
                    page_title=page_title,
                    section_title=section_title,
                    overview_content=overview_content,
                    content=content,
                )
            else:
                prompt = _SECTION_NO_OVERVIEW_PROMPT.format(
                    page_title=page_title,
                    section_title=section_title,
                    content=content,
                )

        questions = self._router.generate(prompt)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / file_path.name
        out_path.write_text(questions, encoding="utf-8")

        logger.info(
            "Generated questions for %s (page=%s, section=%s)",
            file_path.name,
            page_title,
            section_title or "(overview)",
        )
        return questions

    def generate_all(self, force: bool = False, max_files: int = 10) -> dict[str, str]:
        """Generate questions for ``.md`` files in *input_dir*.

        Files that already have a corresponding output in *output_dir* are
        skipped unless *force* is ``True``.  Skipped files do not count
        toward *max_files*.

        *max_files* caps the number of files actually processed.  Set to
        ``0`` to process all eligible files.

        Returns a mapping of ``{filename: questions}``.
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
                logger.info("Skipping %s (output exists)", file_path.name)
                continue
            try:
                questions = self.generate_for_file(file_path)
                results[file_path.name] = questions
            except Exception:
                logger.exception("Failed to generate questions for %s", file_path.name)

        logger.info(
            "Finished: %d processed, %d total",
            len(results),
            len(md_files),
        )
        return results
