"""Unit tests for the Discoverer thread."""

from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from guild_assistant.scraper.discoverer import Discoverer, _API_URL


def _make_api_response(pages, continue_token=None):
    """Build a fake MediaWiki allpages API response."""
    data = {"query": {"allpages": pages}}
    if continue_token is not None:
        data["continue"] = {"apcontinue": continue_token, "continue": "-||"}
    return data


class TestDiscovererPagination:
    """Discoverer should page through API results and push IDs to the queue."""

    def test_single_page_of_results(self):
        queue = Queue()
        pages = [
            {"pageid": 1, "title": "Alpha"},
            {"pageid": 2, "title": "Beta"},
        ]
        response = MagicMock()
        response.json.return_value = _make_api_response(pages)

        with patch("guild_assistant.scraper.discoverer.requests.get", return_value=response):
            d = Discoverer(queue, max_pages=0, num_workers=1)
            d.run()

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert items == [1, 2, None]

    def test_multiple_pages_of_results(self):
        queue = Queue()
        resp1 = MagicMock()
        resp1.json.return_value = _make_api_response(
            [{"pageid": 1, "title": "A"}], continue_token="B"
        )
        resp2 = MagicMock()
        resp2.json.return_value = _make_api_response(
            [{"pageid": 2, "title": "B"}]
        )

        with patch(
            "guild_assistant.scraper.discoverer.requests.get",
            side_effect=[resp1, resp2],
        ):
            d = Discoverer(queue, max_pages=0, num_workers=1)
            d.run()

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert items == [1, 2, None]


class TestDiscovererMaxPages:
    """Discoverer should stop after max_pages IDs when max_pages > 0."""

    def test_stops_at_max_pages(self):
        queue = Queue()
        pages = [
            {"pageid": 1, "title": "A"},
            {"pageid": 2, "title": "B"},
            {"pageid": 3, "title": "C"},
        ]
        response = MagicMock()
        response.json.return_value = _make_api_response(pages)

        with patch("guild_assistant.scraper.discoverer.requests.get", return_value=response):
            d = Discoverer(queue, max_pages=2, num_workers=1)
            d.run()

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        # Only 2 page IDs + 1 sentinel
        assert items == [1, 2, None]

    def test_stops_across_pagination_boundaries(self):
        queue = Queue()
        resp1 = MagicMock()
        resp1.json.return_value = _make_api_response(
            [{"pageid": 1, "title": "A"}, {"pageid": 2, "title": "B"}],
            continue_token="C",
        )
        resp2 = MagicMock()
        resp2.json.return_value = _make_api_response(
            [{"pageid": 3, "title": "C"}, {"pageid": 4, "title": "D"}],
        )

        with patch(
            "guild_assistant.scraper.discoverer.requests.get",
            side_effect=[resp1, resp2],
        ):
            d = Discoverer(queue, max_pages=3, num_workers=1)
            d.run()

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert items == [1, 2, 3, None]


class TestDiscovererSentinels:
    """Discoverer should send one None sentinel per worker."""

    def test_sends_sentinels_per_worker(self):
        queue = Queue()
        response = MagicMock()
        response.json.return_value = _make_api_response([])

        with patch("guild_assistant.scraper.discoverer.requests.get", return_value=response):
            d = Discoverer(queue, max_pages=0, num_workers=3)
            d.run()

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert items == [None, None, None]


class TestDiscovererErrorHandling:
    """Discoverer should handle API errors gracefully."""

    def test_request_exception_sends_sentinels(self):
        queue = Queue()
        with patch(
            "guild_assistant.scraper.discoverer.requests.get",
            side_effect=Exception("connection failed"),
        ):
            d = Discoverer(queue, max_pages=0, num_workers=2)
            d.run()

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        # Should still send sentinels so workers don't hang
        assert items == [None, None]

    def test_http_error_sends_sentinels(self):
        queue = Queue()
        response = MagicMock()
        response.raise_for_status.side_effect = Exception("500 Server Error")

        with patch("guild_assistant.scraper.discoverer.requests.get", return_value=response):
            d = Discoverer(queue, max_pages=0, num_workers=1)
            d.run()

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert items == [None]
