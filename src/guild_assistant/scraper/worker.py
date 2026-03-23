import logging
import re
import threading
from pathlib import Path
from queue import Queue

import markdownify
import requests
from bs4 import BeautifulSoup, NavigableString

_BASE_URL = "https://mytimeatsandrock.fandom.com"
_API_URL = f"{_BASE_URL}/api.php"


class Worker(threading.Thread):
    """Fetches wiki pages by page ID using the MediaWiki Action API and saves
    the rendered HTML to *output_dir*.

    Workers stop when they receive a ``None`` sentinel from the queue,
    which the :class:`Discoverer` enqueues once it has finished.
    """

    def __init__(self, worker_id: int, url_queue: Queue, output_dir: str | Path) -> None:
        super().__init__(name=f"Worker-{worker_id}")
        self.worker_id = worker_id
        self.url_queue = url_queue
        self.output_dir = Path(output_dir)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "TheCommerceGuildBot/0.1"
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _JUNK_SELECTORS = [
        '.toc',              # Table of Contents
        '.mw-editsection',   # "[edit]" links
        '.navbox',           # Huge footer navigation tables
        '.vertical-navbox',  # Side navigation tables
        '.as-seen-on',       # Fandom social noise
        '.error',            # Any API or wiki error messages
    ]

    _REMOVE_TAGS = ['a', 'img', 'script', 'style']

    _NAV_TABLE_HEADERS = frozenset({
        "locations",
        "crafting materials",
        "characters",
        "mission items",
        "furniture",
        "consumables",
        "farming",
        "clothing",
        "shops and services",
        "currency",
        "cooking ingredients",
        "monsters",
        "tools",
    })

    _REMOVE_SECTION_IDS = ['Gallery', 'See_also', 'References']

    @staticmethod
    def _remove_section(soup: BeautifulSoup, span_id: str) -> None:
        """Remove a section identified by a ``<span id="...">`` inside a
        heading element, along with all content up to the next same-level
        or higher heading."""
        span = soup.find('span', id=span_id)
        if not span:
            return
        header = span.find_parent(['h2', 'h3', 'h4', 'h5'])
        if not header:
            return
        current_rank = int(header.name[1])
        stop_headers = [f'h{i}' for i in range(1, current_rank + 1)]
        to_delete = [header]
        for sibling in header.find_next_siblings():
            if sibling.name in stop_headers:
                break
            to_delete.append(sibling)
        for element in to_delete:
            element.decompose()

    @classmethod
    def _remove_unwanted_sections(cls, soup: BeautifulSoup) -> None:
        """Remove sections listed in ``_REMOVE_SECTION_IDS`` (Gallery,
        See_also, References) and convert remaining inline gallery divs
        into plain text lists so that caption names are preserved."""
        for span_id in cls._REMOVE_SECTION_IDS:
            cls._remove_section(soup, span_id)
        for gallery_div in soup.select('.wikia-gallery, .gallery'):
            cls._flatten_gallery(soup, gallery_div)

    @staticmethod
    def _flatten_gallery(soup: BeautifulSoup, gallery_div) -> None:
        """Replace a gallery div with a comma-separated list of its caption
        texts.  If no captions are found the gallery is simply removed."""
        captions = [
            cap.get_text(strip=True)
            for cap in gallery_div.select('.lightbox-caption, .gallerytext')
            if cap.get_text(strip=True)
        ]
        if captions:
            text_node = NavigableString(', '.join(captions))
            gallery_div.replace_with(text_node)
        else:
            gallery_div.decompose()

    @staticmethod
    def _unwrap_internal_paragraphs(soup: BeautifulSoup) -> None:
        """Replace <p> tags inside portable infoboxes with <br> line breaks.

        Without this, each <p> becomes a separate Markdown paragraph (double
        newline), which makes multi-value fields (e.g. Residents) balloon into
        many disconnected blocks. Using <br> keeps the values in a single
        block separated by soft line breaks instead."""
        for infobox in soup.select('aside.portable-infobox'):
            for p in infobox.find_all('p'):
                p.insert_before(soup.new_tag('br'))
                p.unwrap()

    @staticmethod
    def _remove_nav_tables(soup: BeautifulSoup) -> None:
        """Remove navigational tables whose header row contains exactly one
        cell matching a known navigation category name.

        Must be called *after* ``_promote_first_row_to_header`` so every table
        already has a ``<thead>`` with ``<th>`` cells to inspect."""
        for table in soup.find_all('table'):
            thead = table.find('thead')
            if not thead:
                continue
            header_row = thead.find('tr')
            if not header_row:
                continue
            cells = header_row.find_all(['th', 'td'])
            if len(cells) != 1:
                continue
            if cells[0].get_text(strip=True).lower() in Worker._NAV_TABLE_HEADERS:
                table.decompose()

    @staticmethod
    def _promote_first_row_to_header(soup: BeautifulSoup) -> None:
        """For tables that have no <thead>, wrap the first row in a <thead>
        and convert its <td> cells to <th> so markdownify emits a proper
        header row instead of an empty one."""
        for table in soup.find_all('table'):
            if table.find('thead'):
                continue
            first_row = table.find('tr')
            if not first_row:
                continue
            for td in first_row.find_all('td'):
                td.name = 'th'
            thead = soup.new_tag('thead')
            first_row.replace_with(thead)
            thead.append(first_row)

    def _clean(self, html: str) -> str:
        """Convert raw wiki HTML to clean Markdown.

        The pipeline runs in this order:
        1. Strip gallery sections and other junk elements.
        2. Fix structural HTML issues so markdownify renders correctly.
        3. Handle <img> tags: drop decorative images, inline meaningful ones as text.
        4. Escape # and | in text nodes so they are not treated as Markdown syntax.
        5. Convert the cleaned HTML tree to Markdown.
        """
        soup = BeautifulSoup(html, "html.parser")

        # --- Step 1: Remove unwanted sections and elements ---

        # Gallery, See also, and References sections add no textual value.
        self._remove_unwanted_sections(soup)

        # Clear table cells whose Lua template failed to render. The .error
        # element will be decomposed below, but its surrounding text nodes
        # (e.g. "[[File:" and ".png|...|...]]") would otherwise remain as
        # orphaned wikitext. Do this first, before .error is decomposed.
        for cell in soup.find_all(['td', 'th']):
            if cell.find(class_='scribunto-error'):
                cell.clear()

        # Strip noisy UI chrome (TOC, edit links, navboxes, etc.).
        for selector in self._JUNK_SELECTORS:
            for element in soup.select(selector):
                element.decompose()

        # --- Step 2: Fix HTML structure before conversion ---

        # Infobox <p> tags would become separate Markdown paragraphs (double
        # newlines); replace with <br> so values stay as soft line breaks
        # within a single block instead.
        self._unwrap_internal_paragraphs(soup)

        # markdownify needs a <thead> row to emit a proper Markdown table
        # header separator (the "---" line). Promote the first row when absent.
        self._promote_first_row_to_header(soup)

        # Drop inline navigation tables identified by a single header cell
        # matching a known category name (e.g. "Locations", "Characters").
        # Must run after _promote_first_row_to_header so all tables have <thead>.
        self._remove_nav_tables(soup)

        # markdownify strips <br> tags by default. Replace each BS4 Tag with
        # a raw string so the literal "<br>" survives into the Markdown output.
        for br in soup.find_all('br'):
            br.replace_with('<br>')

        # --- Step 3: Resolve <img> tags ---

        for img in soup.find_all('img'):
            alt = img.get('alt', '').strip()

            # Drop images with no alt text — they are purely decorative.
            if not alt:
                img.decompose()
                continue

            parent = img.find_parent(['td', 'th', 'div', 'p', 'dt', 'dd',
                                       'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if not parent:
                continue

            # \u2060 is the Unicode WORD JOINER character used invisibly in
            # some wiki templates; strip it before comparing text content.
            cell_text = parent.get_text().replace('\u2060', '').strip()

            if alt.lower() in cell_text.lower():
                # The alt text is already present as visible text in the same
                # cell/paragraph, so the image would be redundant — drop it.
                img.decompose()
            else:
                # Image carries unique information; inline it as text instead.
                has_other_content = bool(cell_text)
                if has_other_content:
                    # Bracket-wrap so it reads as an inline label alongside
                    # other text (e.g. "Spring [Flower] Season").
                    img.replace_with(f' [{alt}] ')
                else:
                    # Standalone image — just emit the alt text.
                    img.replace_with(f' {alt} ')

        # --- Step 4: Escape Markdown-sensitive characters in text content ---

        # The characters # and | are structural in Markdown (headings, tables).
        # When they appear in wiki text content they must be escaped so that
        # markdownify does not misinterpret them.  Walking NavigableString
        # nodes ensures we only touch actual text, leaving HTML tags and
        # attributes untouched.
        for text_node in list(soup.find_all(string=True)):
            if not isinstance(text_node, NavigableString):
                continue
            original = str(text_node)
            escaped = original.replace('#', r'\#').replace('|', r'\|')
            if escaped != original:
                text_node.replace_with(escaped)

        # --- Step 5: Convert to Markdown ---

        # Strip the remaining HTML tags listed in _REMOVE_TAGS (links, stray
        # scripts/styles) while preserving their inner text content.
        return markdownify.markdownify(
            str(soup),
            strip=self._REMOVE_TAGS,
        )

    @staticmethod
    def _title_to_snake(title: str) -> str:
        """Convert a page title to a snake_case filename stem.

        All characters that are not alphanumeric or underscore are replaced
        with ``_``, consecutive underscores are collapsed, and the result is
        lowercased and stripped of leading/trailing underscores."""
        snake = re.sub(r'[^a-zA-Z0-9]+', '_', title)
        return snake.strip('_').lower()

    def _fetch(self, pageid: int) -> None:
        try:
            params = {
                "action": "parse",
                "pageid": pageid,
                "format": "json",
                "prop": "text",
            }
            response = self._session.get(_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                self._logger.error(
                    "[%s] API error for pageid=%d: %s", self.name, pageid, data["error"]
                )
                return
            actual_pageid = data["parse"]["pageid"]
            title = data["parse"]["title"]
            raw_html = data["parse"]["text"]["*"]
            if 'class="redirectMsg"' in raw_html:
                self._logger.info("[%s] Skipping redirect pageid=%d", self.name, pageid)
                return
            stem = self._title_to_snake(title)
            html_path = self.output_dir / f"{stem}.html"
            html_path.write_text(raw_html, encoding="utf-8")
            md_path = self.output_dir / f"{stem}.md"
            md_path.write_text(self._clean(raw_html), encoding="utf-8")
            self._logger.info(
                "[%s] Saved pageid=%d (queued=%d) title=%r -> %s, %s",
                self.name, actual_pageid, pageid, title, html_path, md_path,
            )
        except Exception as exc:
            self._logger.error("[%s] Failed to fetch pageid=%d: %s", self.name, pageid, exc)

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        while True:
            pageid = self.url_queue.get()
            try:
                if pageid is None:      # sentinel — time to stop
                    break
                self._fetch(pageid)
            finally:
                self.url_queue.task_done()
        self._logger.info("[%s] Shutting down.", self.name)
