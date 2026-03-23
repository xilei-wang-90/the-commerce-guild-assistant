"""Unit tests for the Worker thread."""

from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from guild_assistant.scraper.worker import Worker


class TestTitleToSnake:
    """Worker._title_to_snake should convert titles to clean snake_case."""

    @pytest.mark.parametrize(
        "title, expected",
        [
            ("Simple Page", "simple_page"),
            ("My Time at Sandrock", "my_time_at_sandrock"),
            ("Builder's Workshop", "builder_s_workshop"),
            ("Owen (Mission)", "owen_mission"),
            ("   Leading/Trailing   ", "leading_trailing"),
            ("ALL-CAPS-TITLE", "all_caps_title"),
            ("Already_snake_case", "already_snake_case"),
            ("multiple   spaces", "multiple_spaces"),
            ("Special!@#$%^&*()chars", "special_chars"),
        ],
    )
    def test_title_conversion(self, title, expected):
        assert Worker._title_to_snake(title) == expected


class TestClean:
    """Worker._clean should produce clean Markdown from wiki HTML."""

    def _make_worker(self, tmp_path):
        queue = Queue()
        return Worker(0, queue, tmp_path)

    def test_strips_toc(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = '<div class="toc"><ul><li>Section 1</li></ul></div><p>Content</p>'
        result = w._clean(html)
        assert "Section 1" not in result
        assert "Content" in result

    def test_strips_editsection(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = '<h2>Title<span class="mw-editsection">[edit]</span></h2><p>Body</p>'
        result = w._clean(html)
        assert "[edit]" not in result
        assert "Body" in result

    def test_strips_navbox(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = '<div class="navbox">Nav stuff</div><p>Real content</p>'
        result = w._clean(html)
        assert "Nav stuff" not in result
        assert "Real content" in result

    def test_strips_gallery_section(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = (
            '<h2><span id="Gallery">Gallery</span></h2>'
            '<div class="wikia-gallery">images</div>'
            '<h2><span id="Notes">Notes</span></h2>'
            '<p>Some notes</p>'
        )
        result = w._clean(html)
        assert "Gallery" not in result
        assert "images" not in result
        assert "Notes" in result
        assert "Some notes" in result

    def test_strips_links_preserves_text(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = '<p><a href="/wiki/Owen">Owen</a> is a character.</p>'
        result = w._clean(html)
        assert "Owen" in result
        assert "href" not in result

    def test_drops_decorative_images(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = '<p><img src="icon.png" alt="">Some text</p>'
        result = w._clean(html)
        assert "icon.png" not in result
        assert "Some text" in result

    def test_inlines_image_alt_when_unique(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = '<td><img src="item.png" alt="Copper Ore"></td>'
        result = w._clean(html)
        assert "Copper Ore" in result

    def test_drops_redundant_image_alt(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = '<td>Copper Ore<img src="item.png" alt="Copper Ore"></td>'
        result = w._clean(html)
        # Should appear once (from the text), not duplicated
        assert result.count("Copper Ore") == 1

    def test_escapes_hash_in_text(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = "<p>Item #5 is rare</p>"
        result = w._clean(html)
        assert r"\#" in result

    def test_escapes_pipe_in_text(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = "<p>Choice A | Choice B</p>"
        result = w._clean(html)
        assert r"\|" in result

    def test_promotes_first_row_to_header(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = (
            "<table>"
            "<tr><td>Name</td><td>Value</td></tr>"
            "<tr><td>Iron</td><td>10</td></tr>"
            "</table>"
        )
        result = w._clean(html)
        # Should have a header separator line (---)
        assert "---" in result
        assert "Name" in result

    def test_table_with_existing_thead_unchanged(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = (
            "<table>"
            "<thead><tr><th>Name</th><th>Value</th></tr></thead>"
            "<tr><td>Iron</td><td>10</td></tr>"
            "</table>"
        )
        result = w._clean(html)
        assert "Name" in result
        assert "Iron" in result

    def test_scribunto_error_cells_cleared(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = (
            '<table><tr><td>'
            '[[File:<span class="scribunto-error">error</span>.png|...|...]]'
            '</td></tr></table>'
        )
        result = w._clean(html)
        assert "scribunto" not in result.lower()
        assert "[[File:" not in result

    def test_br_tags_preserved(self, tmp_path):
        w = self._make_worker(tmp_path)
        html = "<p>Line one<br>Line two</p>"
        result = w._clean(html)
        assert "<br>" in result


class TestFetch:
    """Worker._fetch should save HTML and Markdown files, and handle edge cases."""

    def test_saves_html_and_md(self, tmp_path):
        queue = Queue()
        w = Worker(0, queue, tmp_path)

        api_response = {
            "parse": {
                "pageid": 42,
                "title": "Test Page",
                "text": {"*": "<p>Hello world</p>"},
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response

        with patch.object(w._session, "get", return_value=mock_resp):
            w._fetch(42)

        html_file = tmp_path / "test_page.html"
        md_file = tmp_path / "test_page.md"
        assert html_file.exists()
        assert md_file.exists()
        assert "Hello world" in html_file.read_text()
        assert "Hello world" in md_file.read_text()

    def test_skips_redirects(self, tmp_path):
        queue = Queue()
        w = Worker(0, queue, tmp_path)

        api_response = {
            "parse": {
                "pageid": 99,
                "title": "Redirect Page",
                "text": {"*": '<div class="redirectMsg">Redirect to target</div>'},
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response

        with patch.object(w._session, "get", return_value=mock_resp):
            w._fetch(99)

        # No files should be created for redirects
        assert list(tmp_path.iterdir()) == []

    def test_handles_api_error(self, tmp_path):
        queue = Queue()
        w = Worker(0, queue, tmp_path)

        api_response = {"error": {"code": "nosuchpageid", "info": "no such page"}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response

        with patch.object(w._session, "get", return_value=mock_resp):
            w._fetch(999)

        assert list(tmp_path.iterdir()) == []

    def test_handles_request_exception(self, tmp_path):
        queue = Queue()
        w = Worker(0, queue, tmp_path)

        with patch.object(w._session, "get", side_effect=Exception("timeout")):
            # Should not raise
            w._fetch(1)

        assert list(tmp_path.iterdir()) == []


class TestWorkerRun:
    """Worker.run should process queue items and stop on sentinel."""

    def test_processes_items_and_stops_on_sentinel(self, tmp_path):
        queue = Queue()
        queue.put(42)
        queue.put(None)

        w = Worker(0, queue, tmp_path)

        api_response = {
            "parse": {
                "pageid": 42,
                "title": "Test",
                "text": {"*": "<p>content</p>"},
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response

        with patch.object(w._session, "get", return_value=mock_resp):
            w.run()

        assert (tmp_path / "test.md").exists()

    def test_creates_output_dir(self, tmp_path):
        output = tmp_path / "nested" / "output"
        queue = Queue()
        queue.put(None)

        w = Worker(0, queue, output)
        w.run()

        assert output.is_dir()

    def test_calls_task_done_for_each_item(self, tmp_path):
        queue = Queue()
        queue.put(42)
        queue.put(None)

        w = Worker(0, queue, tmp_path)

        api_response = {
            "parse": {
                "pageid": 42,
                "title": "Test",
                "text": {"*": "<p>ok</p>"},
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response

        with patch.object(w._session, "get", return_value=mock_resp):
            w.run()

        # Queue should be fully consumed with task_done called for each
        assert queue.unfinished_tasks == 0


class TestRemoveUnwantedSections:
    """Test the unwanted-section removal helper in isolation."""

    def test_removes_gallery_with_h2(self, tmp_path):
        from bs4 import BeautifulSoup

        html = (
            '<h2><span id="Gallery">Gallery</span></h2>'
            '<p>gallery images here</p>'
            '<h2><span id="Trivia">Trivia</span></h2>'
            '<p>Some trivia</p>'
        )
        soup = BeautifulSoup(html, "html.parser")
        Worker._remove_unwanted_sections(soup)
        text = soup.get_text()
        assert "gallery images here" not in text
        assert "Trivia" in text
        assert "Some trivia" in text

    def test_removes_standalone_gallery_divs(self, tmp_path):
        from bs4 import BeautifulSoup

        html = '<div class="wikia-gallery">gallery stuff</div><p>Content</p>'
        soup = BeautifulSoup(html, "html.parser")
        Worker._remove_unwanted_sections(soup)
        text = soup.get_text()
        assert "gallery stuff" not in text
        assert "Content" in text

    def test_no_gallery_is_noop(self, tmp_path):
        from bs4 import BeautifulSoup

        html = "<p>Just some content</p>"
        soup = BeautifulSoup(html, "html.parser")
        Worker._remove_unwanted_sections(soup)
        assert "Just some content" in soup.get_text()


class TestRemoveNavTables:
    """Test _remove_nav_tables in isolation."""

    def _soup(self, html: str):
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "html.parser")

    def _table_with_single_header(self, header_text: str) -> str:
        return (
            f"<table><thead><tr><th>{header_text}</th></tr></thead>"
            "<tr><td>Item A</td></tr></table>"
        )

    @pytest.mark.parametrize("header", [
        "Locations",
        "Crafting materials",
        "Characters",
        "Mission items",
        "Furniture",
        "Consumables",
        "Farming",
        "Clothing",
        "Shops and services",
        "Currency",
    ])
    def test_removes_nav_table_for_each_category(self, header):
        soup = self._soup(self._table_with_single_header(header))
        Worker._remove_nav_tables(soup)
        assert soup.find("table") is None

    def test_case_insensitive_match(self):
        soup = self._soup(self._table_with_single_header("LOCATIONS"))
        Worker._remove_nav_tables(soup)
        assert soup.find("table") is None

    def test_multi_cell_header_not_removed(self):
        html = (
            "<table><thead><tr><th>Locations</th><th>Extra</th></tr></thead>"
            "<tr><td>Item A</td></tr></table>"
        )
        soup = self._soup(html)
        Worker._remove_nav_tables(soup)
        assert soup.find("table") is not None

    def test_non_nav_single_header_not_removed(self):
        soup = self._soup(self._table_with_single_header("Stats"))
        Worker._remove_nav_tables(soup)
        assert soup.find("table") is not None

    def test_surrounding_content_preserved(self):
        html = (
            "<p>Before</p>"
            + self._table_with_single_header("Characters")
            + "<p>After</p>"
        )
        soup = self._soup(html)
        Worker._remove_nav_tables(soup)
        assert soup.find("table") is None
        assert "Before" in soup.get_text()
        assert "After" in soup.get_text()

    def test_only_nav_table_removed_when_mixed(self):
        html = (
            self._table_with_single_header("Currency")
            + "<table><thead><tr><th>Name</th><th>Value</th></tr></thead>"
            "<tr><td>Iron</td><td>10</td></tr></table>"
        )
        soup = self._soup(html)
        Worker._remove_nav_tables(soup)
        tables = soup.find_all("table")
        assert len(tables) == 1
        assert "Name" in tables[0].get_text()

    def test_table_without_thead_not_removed(self):
        html = (
            "<table><tr><th>Locations</th></tr>"
            "<tr><td>Item A</td></tr></table>"
        )
        soup = self._soup(html)
        Worker._remove_nav_tables(soup)
        # No <thead> means it won't be touched
        assert soup.find("table") is not None

    def test_clean_integration_removes_nav_table(self, tmp_path):
        from queue import Queue
        w = Worker(0, Queue(), tmp_path)
        html = (
            "<p>Page intro</p>"
            "<table><tr><td>Furniture</td></tr>"
            "<tr><td>Chair</td><td>10</td></tr></table>"
            "<p>More content</p>"
        )
        result = w._clean(html)
        assert "Furniture" not in result
        assert "Chair" not in result
        assert "Page intro" in result
        assert "More content" in result


class TestPromoteFirstRowToHeader:
    """Test thead promotion in isolation."""

    def test_promotes_first_row(self):
        from bs4 import BeautifulSoup

        html = (
            "<table><tr><td>A</td><td>B</td></tr>"
            "<tr><td>1</td><td>2</td></tr></table>"
        )
        soup = BeautifulSoup(html, "html.parser")
        Worker._promote_first_row_to_header(soup)
        assert soup.find("thead") is not None
        assert soup.find("th") is not None
        assert soup.find("th").string == "A"

    def test_skips_table_with_thead(self):
        from bs4 import BeautifulSoup

        html = (
            "<table><thead><tr><th>X</th></tr></thead>"
            "<tr><td>1</td></tr></table>"
        )
        soup = BeautifulSoup(html, "html.parser")
        Worker._promote_first_row_to_header(soup)
        # Should not add a second thead
        assert len(soup.find_all("thead")) == 1
