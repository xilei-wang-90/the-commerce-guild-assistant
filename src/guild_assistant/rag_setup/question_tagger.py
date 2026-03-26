"""Prepend metadata tags to reverse-hyde question files.

Reads hypothetical question files from ``data/reverse-hyde/`` and the
corresponding silver-section Markdown files from ``data/silver-sections/``.
For each file, extracts the page slug (from the filename) and all L2/L3
heading titles (from the section content), then writes a tagged copy to
the output directory with a bracketed tag line prepended.

The tagged files can then be embedded into a separate ChromaDB collection
to test whether the extra metadata improves retrieval quality.

Tag format example::

    [yakmel | Obtaining | From NPCs | From Enemies]
    What items do yakmel drop?
    How do you obtain a yakmel?
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_SETEXT_RE = re.compile(r"^([^\n]+)\n(=+|-{2,})[ \t]*$", re.MULTILINE)


def _extract_l2_l3_titles(content: str) -> list[str]:
    """Return all L2 and L3 heading titles from Markdown *content*.

    Handles both ATX-style headings (``## Title``) and setext-style
    headings (title followed by ``---`` for L2 or ``===`` for L1).
    Setext headings can only express L1 and L2; only L2 (``---``)
    matches are included.
    """
    headings: list[tuple[int, int, str]] = []  # (position, level, title)

    for m in _ATX_HEADING_RE.finditer(content):
        level = len(m.group(1))
        if level in (2, 3):
            headings.append((m.start(), level, m.group(2).strip()))

    for m in _SETEXT_RE.finditer(content):
        title_text = m.group(1).strip()
        if not title_text:
            continue
        underline = m.group(2)
        level = 1 if underline[0] == "=" else 2
        if level == 2:
            headings.append((m.start(), level, title_text))

    headings.sort(key=lambda h: h[0])
    return [title for _, _, title in headings]


def _build_tag_line(page_slug: str, heading_titles: list[str]) -> str:
    """Build a bracketed tag line from *page_slug* and *heading_titles*.

    Example::

        >>> _build_tag_line("yakmel", ["Obtaining", "From NPCs"])
        '[yakmel | Obtaining | From NPCs]'
    """
    parts = [page_slug] + heading_titles
    return "[" + " | ".join(parts) + "]"


def _parse_page_slug(filename: str) -> str:
    """Extract the page slug from a section filename.

    The filename convention is ``<page_slug>-<section_name>.md``.
    """
    stem = Path(filename).stem
    page_slug, _, _ = stem.partition("-")
    return page_slug


class QuestionTagger:
    """Prepend metadata tags to reverse-hyde question files.

    *questions_dir*  – directory containing reverse-hyde question files
                       (e.g. ``data/reverse-hyde/``).
    *sections_dir*   – directory containing silver-section Markdown files
                       (e.g. ``data/silver-sections/``).
    *output_dir*     – directory to write tagged question files to
                       (e.g. ``data/tagged-reverse-hyde/``).
    """

    def __init__(
        self,
        questions_dir: str | Path,
        sections_dir: str | Path,
        output_dir: str | Path,
    ) -> None:
        self._questions_dir = Path(questions_dir)
        self._sections_dir = Path(sections_dir)
        self._output_dir = Path(output_dir)

    def tag_file(self, question_file: Path) -> str:
        """Tag a single question file and write the result.

        Reads the corresponding silver-section file to extract L2/L3
        headings, prepends a tag line to the questions, and writes to
        *output_dir*.

        Returns the tagged content.
        """
        questions = question_file.read_text(encoding="utf-8")
        page_slug = _parse_page_slug(question_file.name)

        section_file = self._sections_dir / question_file.name
        if section_file.exists():
            section_content = section_file.read_text(encoding="utf-8")
            heading_titles = _extract_l2_l3_titles(section_content)
        else:
            logger.warning(
                "No section file found for %s, using page slug only",
                question_file.name,
            )
            heading_titles = []

        tag_line = _build_tag_line(page_slug, heading_titles)
        tagged = tag_line + "\n" + questions

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / question_file.name
        out_path.write_text(tagged, encoding="utf-8")

        logger.info("Tagged %s (page=%s, headings=%d)", question_file.name, page_slug, len(heading_titles))
        return tagged

    def tag_all(self, force: bool = False, max_files: int = 10) -> dict[str, str]:
        """Tag all ``.md`` files in *questions_dir*.

        Files that already have a corresponding output in *output_dir*
        are skipped unless *force* is ``True``.  Skipped files do not
        count toward *max_files*.

        *max_files* caps the number of files actually processed.  Set to
        ``0`` to process all eligible files.

        Returns a mapping of ``{filename: tagged_content}``.
        """
        results: dict[str, str] = {}
        md_files = sorted(self._questions_dir.glob("*.md"))

        if not md_files:
            logger.warning("No .md files found in %s", self._questions_dir)
            return results

        logger.info("Found %d .md files to tag", len(md_files))

        for file_path in md_files:
            if max_files and len(results) >= max_files:
                break
            if not force and (self._output_dir / file_path.name).exists():
                logger.info("Skipping %s (output exists)", file_path.name)
                continue
            try:
                tagged = self.tag_file(file_path)
                results[file_path.name] = tagged
            except Exception:
                logger.exception("Failed to tag %s", file_path.name)

        logger.info(
            "Finished: %d tagged, %d total",
            len(results),
            len(md_files),
        )
        return results
