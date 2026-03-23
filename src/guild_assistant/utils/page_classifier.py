"""Page type classification for wiki articles.

Determines the type of a wiki page (item, character, location, etc.)
based on filename patterns and heading content.  Used by both the
section breaker and the golden-dataset generator.
"""

import re
from enum import Enum
from pathlib import Path


class PageType(Enum):
    """Page type classification for wiki articles."""

    ITEM = "item"
    LOCATION = "location"
    CHARACTER = "character"
    MONSTER = "monster"
    MISSION = "mission"
    DIALOGUE = "dialogue"
    BUYBACK = "buyback"
    EVENT = "event"
    STORE = "store"
    REGION = "region"
    FESTIVAL = "festival"
    GENERIC = "generic"


_ATX_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_SETEXT_RE = re.compile(r"^([^\n]+)\n(=+|-{2,})[ \t]*$", re.MULTILINE)


def classify_page(filename: str, heading_titles: list[str]) -> list[PageType]:
    """Classify a page into exactly one type based on filename and headings.

    Filename-based rules are checked first and take priority over
    content-based rules.  Returns a single-element list containing the
    matched :class:`PageType`, or ``[PageType.GENERIC]`` if nothing matches.
    """
    stem = Path(filename).stem.lower()
    lower_titles = {t.lower() for t in heading_titles}

    # Filename-based rules (highest priority, checked first)
    if stem.startswith("mission"):
        return [PageType.MISSION]
    if stem.endswith("dialogue"):
        return [PageType.DIALOGUE]
    if stem.endswith("buyback"):
        return [PageType.BUYBACK]
    if stem.startswith("event"):
        return [PageType.EVENT]

    # Content-based rules (first match wins)
    if "obtaining" in lower_titles:
        return [PageType.ITEM]
    if "biographical information" in lower_titles or "physical description" in lower_titles:
        return [PageType.CHARACTER]
    if "battle statistics" in lower_titles:
        return [PageType.MONSTER]
    if "region" in lower_titles:
        return [PageType.LOCATION]
    if "stock" in lower_titles:
        return [PageType.STORE]
    if "population" in lower_titles:
        return [PageType.REGION]
    if "time" in lower_titles:
        return [PageType.FESTIVAL]

    return [PageType.GENERIC]


def extract_heading_titles(content: str) -> list[str]:
    """Extract all heading titles (ATX and setext) from Markdown *content*."""
    titles: list[str] = []

    for m in _ATX_HEADING_RE.finditer(content):
        titles.append(m.group(1).strip())

    for m in _SETEXT_RE.finditer(content):
        title_text = m.group(1).strip()
        if title_text:
            titles.append(title_text)

    return titles


def classify_file(file_path: Path) -> list[PageType]:
    """Classify a Markdown file by reading its content and headings."""
    content = file_path.read_text(encoding="utf-8")
    titles = extract_heading_titles(content)
    return classify_page(file_path.name, titles)
