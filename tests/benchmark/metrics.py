"""Retrieval evaluation metrics.

Each metric function takes a list of :class:`QueryResult` instances and
returns a float score.  New metrics can be added by defining a function
and registering it in :data:`METRIC_REGISTRY`.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Outcome of a single retrieval evaluation query.

    *retrieved_ids* contains the filenames (basenames) of the top-k
    retrieval results.  *expected_id* is the filename the query should
    have retrieved (either a page or section filename depending on the
    collection being evaluated).
    """

    question: str
    expected_id: str
    retrieved_ids: list[str]


MetricFn = Callable[[list[QueryResult]], float]


def hit_rate(results: list[QueryResult]) -> float:
    """Fraction of queries where the expected document appears in the top-k.

    Returns 0.0 when *results* is empty.
    """
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.expected_id in r.retrieved_ids)
    return hits / len(results)


def _page_slug(filename: str) -> str:
    """Extract the page slug from a page or section filename.

    Page filenames (``yakmel.md``) return the stem (``yakmel``).
    Section filenames (``yakmel-overview.md``) return the part before
    the first hyphen (``yakmel``).  Page slugs use underscores, so the
    first hyphen is always the page/section delimiter.
    """
    stem = Path(filename).stem
    return stem.partition("-")[0]


def page_hit_rate(results: list[QueryResult]) -> float:
    """Fraction of queries where *any* result from the expected page appears.

    Like :func:`hit_rate`, but matches at page level rather than exact
    filename.  For summary collections (where IDs are already page
    filenames) this equals ``hit_rate``.  For section collections it
    answers whether the retriever found the correct *page*, even if it
    returned a different section of that page.

    Returns 0.0 when *results* is empty.
    """
    if not results:
        return 0.0
    hits = sum(
        1
        for r in results
        if _page_slug(r.expected_id) in {_page_slug(rid) for rid in r.retrieved_ids}
    )
    return hits / len(results)


def ndcg(results: list[QueryResult]) -> float:
    """Mean Normalized Discounted Cumulative Gain across all queries.

    Each query has a single relevant document (``expected_id``).  The
    DCG for a query is ``1 / log2(rank + 1)`` where *rank* is the
    1-based position of the expected document in ``retrieved_ids``, or
    0 if the document is not found.  The ideal DCG is always 1.0
    (relevant document at rank 1), so NDCG equals DCG.

    Returns 0.0 when *results* is empty.
    """
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        try:
            rank = r.retrieved_ids.index(r.expected_id) + 1
            total += 1.0 / math.log2(rank + 1)
        except ValueError:
            pass
    return total / len(results)


METRIC_REGISTRY: dict[str, MetricFn] = {
    "hit_rate": hit_rate,
    "page_hit_rate": page_hit_rate,
    "ndcg": ndcg,
}
