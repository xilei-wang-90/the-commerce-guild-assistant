"""Break silver-tier Markdown files into type-aware chunks.

Reads full-article ``.md`` files from an input directory (typically
``data/silver/``), classifies each page by type (item, character,
location, etc.) based on content and filename, and produces chunks
following type-specific rules.

Each L2 section (including all its L3+ subsections) forms one chunk.
Sections designated as "Overview" for the page's type are merged into a
single Overview chunk.  Chunks that contain only headings (no body text)
are skipped.

Output filenames follow ``<page_slug>-<chunk_name>.md`` where
*chunk_name* is ``overview`` or the slugified L2 heading title.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from guild_assistant.utils.page_classifier import PageType, classify_page

logger = logging.getLogger(__name__)

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_SETEXT_RE = re.compile(r"^([^\n]+)\n(=+|-{2,})[ \t]*$", re.MULTILINE)


# L2 sections that go to Overview for each page type (lowercase).
_TYPE_OVERVIEW_SECTIONS: dict[PageType, set[str]] = {
    PageType.ITEM: set(),  # uses "before Obtaining" rule instead
    PageType.LOCATION: {"establishment information", "locations"},
    PageType.CHARACTER: {
        "biographical information",
        "physical description",
        "personal information",
        "social interactions",
        "residence",
    },
    PageType.MONSTER: {"battle statistics", "drops"},
    PageType.MISSION: {"mission details", "rewards", "chronology"},
    PageType.EVENT: {"event information", "rewards"},
    PageType.STORE: {"establishment information"},
    PageType.FESTIVAL: {"time", "information", "unlock"},
}

# Sections that always go to Overview regardless of page type (lowercase).
_GENERAL_OVERVIEW_SECTIONS = {"overview", "information", "general information"}


# ---------------------------------------------------------------------------
# Heading parsing (shared with the old implementation)
# ---------------------------------------------------------------------------


@dataclass
class _HeadingMatch:
    """Unified representation of an ATX or setext heading."""

    start: int  # start position of the entire heading block
    end: int  # end position (content after the heading starts here)
    level: int  # 1–6
    title: str


def _find_headings(content: str) -> list[_HeadingMatch]:
    """Return all ATX and setext headings in *content*, sorted by position.

    Setext headings use ``=`` underlines for level 1 and ``-`` underlines
    (at least two characters) for level 2.  A ``---`` line is only treated
    as a setext heading when the preceding line contains non-whitespace text;
    otherwise it is a thematic break (horizontal rule) and is ignored.

    When an ATX and setext heading overlap in position the earlier one wins.
    """
    headings: list[_HeadingMatch] = []

    for m in _ATX_HEADING_RE.finditer(content):
        headings.append(
            _HeadingMatch(
                start=m.start(),
                end=m.end(),
                level=len(m.group(1)),
                title=m.group(2).strip(),
            )
        )

    for m in _SETEXT_RE.finditer(content):
        title_text = m.group(1).strip()
        if not title_text:
            # Blank line above the underline → thematic break, not a heading.
            continue
        underline = m.group(2)
        level = 1 if underline[0] == "=" else 2
        headings.append(
            _HeadingMatch(
                start=m.start(),
                end=m.end(),
                level=level,
                title=title_text,
            )
        )

    # Sort by position; drop overlaps (keep the one that starts first).
    headings.sort(key=lambda h: h.start)
    deduped: list[_HeadingMatch] = []
    for h in headings:
        if deduped and h.start < deduped[-1].end:
            continue  # overlaps with the previous heading
        deduped.append(h)
    return deduped


def _title_to_slug(title: str) -> str:
    """Convert a heading title to a filename-safe slug.

    Words are separated by underscores; only lowercase alphanumerics and
    underscores are kept.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower().strip())
    return slug.strip("_")


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------


@dataclass
class _L2Block:
    """An L2 section with all its content including subsections."""

    title: str
    content: str  # Full markdown from L2 heading to next L2/L1/EOF


def _has_non_heading_text(text: str) -> bool:
    """Return ``True`` if *text* contains content beyond just headings."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        # ATX heading
        if re.match(r"^#{1,6}\s+", stripped):
            i += 1
            continue
        # Setext underline (=== or ---)
        if re.match(r"^(=+|-{2,})\s*$", stripped):
            i += 1
            continue
        # Check if this line is a setext heading title (followed by underline)
        if i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            if re.match(r"^(=+|-{2,})\s*$", next_stripped):
                i += 2  # skip title + underline
                continue
        return True
    return False


def _extract_preamble_and_blocks(
    content: str, headings: list[_HeadingMatch]
) -> tuple[str, list[_L2Block]]:
    """Split *content* into preamble text and L2-level blocks.

    Preamble includes content before any heading, content directly under
    L1 headings, and content under orphan L3+ headings (those not nested
    under any L2).

    Each L2 block spans from its heading to the next L2 or L1 heading
    (or EOF) and includes all nested L3+ subsections.
    """
    preamble_parts: list[str] = []
    l2_blocks: list[_L2Block] = []

    if not headings:
        return content.strip(), []

    # Content before first heading
    pre = content[: headings[0].start].strip()
    if pre:
        preamble_parts.append(pre)

    current_l2_title: str | None = None
    current_l2_start: int = 0

    for i, h in enumerate(headings):
        next_start = headings[i + 1].start if i + 1 < len(headings) else len(content)

        if h.level == 1:
            # Close current L2 block
            if current_l2_title is not None:
                block = content[current_l2_start : h.start].rstrip()
                l2_blocks.append(_L2Block(title=current_l2_title, content=block))
                current_l2_title = None
            # L1 content → preamble
            text = content[h.end : next_start].strip()
            if text:
                preamble_parts.append(text)

        elif h.level == 2:
            # Close current L2 block
            if current_l2_title is not None:
                block = content[current_l2_start : h.start].rstrip()
                l2_blocks.append(_L2Block(title=current_l2_title, content=block))
            # Start new L2 block
            current_l2_title = h.title
            current_l2_start = h.start

        else:
            # L3+ heading not under any L2 → include in preamble
            if current_l2_title is None:
                text = content[h.start : next_start].strip()
                if text:
                    preamble_parts.append(text)

    # Close final L2 block
    if current_l2_title is not None:
        block = content[current_l2_start :].rstrip()
        l2_blocks.append(_L2Block(title=current_l2_title, content=block))

    preamble = "\n\n".join(preamble_parts)
    return preamble, l2_blocks



def _overview_block_indices(
    l2_blocks: list[_L2Block],
    page_types: list[PageType],
) -> set[int]:
    """Return indices of L2 blocks that belong to the Overview chunk."""
    overview: set[int] = set()
    titles_lower = [b.title.lower() for b in l2_blocks]

    # 1. General: "Overview", "Information", "General Information"
    for i, title in enumerate(titles_lower):
        if title in _GENERAL_OVERVIEW_SECTIONS:
            overview.add(i)

    # 2. If "Overview" section exists, everything before and including it
    for i, title in enumerate(titles_lower):
        if title == "overview":
            for j in range(i + 1):
                overview.add(j)
            break

    # 3. Type-specific rules
    for ptype in page_types:
        if ptype == PageType.ITEM:
            # All sections up to but not including "Obtaining"
            for i, title in enumerate(titles_lower):
                if title == "obtaining":
                    break
                overview.add(i)

        elif ptype in _TYPE_OVERVIEW_SECTIONS:
            type_sections = _TYPE_OVERVIEW_SECTIONS[ptype]
            seen: set[str] = set()
            for i, title in enumerate(titles_lower):
                if title in type_sections and title not in seen:
                    overview.add(i)
                    seen.add(title)

    # 4. Fill gaps: if any section is in Overview, all earlier sections
    #    must also be in Overview.
    if overview:
        max_idx = max(overview)
        for i in range(max_idx):
            overview.add(i)

    return overview


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SectionBreaker:
    """Split ``.md`` files from *input_dir* into type-aware chunks in
    *output_dir*.

    Pages are classified by type (item, character, location, etc.) and
    chunked accordingly.  Each L2 section (with all subsections) forms
    one chunk.  Type-designated overview sections are merged into a
    single Overview chunk.
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> None:
        self._input_dir = Path(input_dir)
        self._output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def break_file(self, file_path: Path) -> list[str]:
        """Break a single ``.md`` file into chunk files.

        Returns a list of output filenames that were written.
        """
        content = file_path.read_text(encoding="utf-8")
        page_slug = file_path.stem  # already snake_case

        headings = _find_headings(content)
        all_titles = [h.title for h in headings]
        page_types = classify_page(file_path.name, all_titles)
        preamble, l2_blocks = _extract_preamble_and_blocks(content, headings)
        overview_indices = _overview_block_indices(l2_blocks, page_types)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []

        # -- Overview chunk ------------------------------------------------
        overview_parts: list[str] = []
        if preamble:
            overview_parts.append(preamble)
        for i in sorted(overview_indices):
            overview_parts.append(l2_blocks[i].content)

        overview_text = "\n\n".join(overview_parts).strip()
        if overview_text and _has_non_heading_text(overview_text):
            filename = f"{page_slug}-overview.md"
            (self._output_dir / filename).write_text(overview_text, encoding="utf-8")
            written.append(filename)

        # -- Standalone chunks ---------------------------------------------
        # Count slugs among standalone blocks for duplicate handling.
        standalone_slugs: list[str] = []
        for i, block in enumerate(l2_blocks):
            if i not in overview_indices:
                standalone_slugs.append(_title_to_slug(block.title))
        slug_counts = Counter(standalone_slugs)

        seen: dict[str, int] = {}
        for i, block in enumerate(l2_blocks):
            if i in overview_indices:
                continue
            if not _has_non_heading_text(block.content):
                continue

            slug = _title_to_slug(block.title)
            if slug_counts[slug] > 1:
                seen[slug] = seen.get(slug, 0) + 1
                chunk_name = f"{slug}_{seen[slug]}"
            else:
                chunk_name = slug

            filename = f"{page_slug}-{chunk_name}.md"
            (self._output_dir / filename).write_text(
                block.content.strip(), encoding="utf-8"
            )
            written.append(filename)

        logger.info(
            "Broke %s into %d chunks (types: %s)",
            file_path.name,
            len(written),
            ", ".join(t.value for t in page_types),
        )
        return written

    def break_all(
        self, force: bool = False, max_files: int = 10
    ) -> dict[str, list[str]]:
        """Break all ``.md`` files in *input_dir* into chunk files.

        Files whose chunks already exist in *output_dir* are skipped
        unless *force* is ``True``.  Skipped files do not count toward
        *max_files*.

        *max_files* caps the number of source files processed per run.
        Set to ``0`` to process all eligible files.

        Returns ``{input_filename: [output_filenames]}``.
        """
        results: dict[str, list[str]] = {}
        md_files = sorted(self._input_dir.glob("*.md"))

        if not md_files:
            logger.warning("No .md files found in %s", self._input_dir)
            return results

        logger.info("Found %d .md files to process", len(md_files))

        for file_path in md_files:
            if max_files and len(results) >= max_files:
                break
            if not force and self._has_existing_output(file_path.stem):
                logger.info("Skipping %s (output exists)", file_path.name)
                continue
            try:
                written = self.break_file(file_path)
                results[file_path.name] = written
            except Exception:
                logger.exception("Failed to break %s", file_path.name)

        logger.info(
            "Finished: %d processed, %d total",
            len(results),
            len(md_files),
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_existing_output(self, stem: str) -> bool:
        """Return ``True`` if *output_dir* already contains files for *stem*."""
        if not self._output_dir.exists():
            return False
        prefix = f"{stem}-"
        return any(f.name.startswith(prefix) for f in self._output_dir.iterdir())
