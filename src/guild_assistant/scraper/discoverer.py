import logging
import threading
from queue import Queue

import requests

_BASE_URL = "https://mytimeatsandrock.fandom.com"
_API_URL = f"{_BASE_URL}/api.php"


class Discoverer(threading.Thread):
    """Discovers wiki pages via the MediaWiki API:Allpages endpoint and pushes
    page IDs into a shared queue.  If *max_pages* is 0 all pages are discovered
    (pagination continues until exhausted); otherwise stops after *max_pages*
    IDs have been enqueued.  Sends one ``None`` sentinel per worker so every
    worker knows when to stop.
    """

    def __init__(
        self,
        url_queue: Queue,
        max_pages: int,
        num_workers: int,
    ) -> None:
        super().__init__(name="Discoverer")
        self.url_queue = url_queue
        self.max_pages = max_pages
        self.num_workers = num_workers
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        unlimited = self.max_pages == 0
        limit_str = "unlimited" if unlimited else str(self.max_pages)

        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": "max",
            "format": "json",
        }
        last_continue: dict = {}
        discovered = 0

        while True:
            req = {**params, **last_continue}
            try:
                response = requests.get(_API_URL, params=req, timeout=10)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                self._logger.error("Error fetching page list: %s", exc)
                break

            for page in data.get("query", {}).get("allpages", []):
                # Guard before each put so discovered never exceeds max_pages
                if not unlimited and discovered >= self.max_pages:
                    break
                pageid = page["pageid"]
                self.url_queue.put(pageid)
                discovered += 1
                self._logger.info(
                    "Discovered (%d/%s): pageid=%d title=%r",
                    discovered, limit_str, pageid, page["title"],
                )

            # Stop if the cap has been reached or there are no more pages
            if not unlimited and discovered >= self.max_pages:
                break
            if "continue" not in data:
                break

            # Merge continuation tokens into the next request
            last_continue = data["continue"]

        # Signal each worker that there is no more work
        for _ in range(self.num_workers):
            self.url_queue.put(None)

        self._logger.info("Discovery complete. Enqueued %d pages.", discovered)
