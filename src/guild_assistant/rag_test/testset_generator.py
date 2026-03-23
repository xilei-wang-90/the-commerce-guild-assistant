"""Generate retrieval test sets from golden pages.

Reads a golden-pages file (one filename per line), finds related
silver-section chunks for each page, randomly selects one section per
page, and sends it to an LLM to generate three types of question/answer
pairs: factoid, conceptual, and messy.  Results are written to three
separate CSV files under the output directory.
"""

import csv
import logging
import random
from pathlib import Path

from guild_assistant.utils.router import ModelRouter

logger = logging.getLogger(__name__)

_PROMPT = """\
You are a wiki Q/A generator for "My Time at Sandrock".
Page: {page_title} | Section: {section_title}

From the section below, generate three Q/A pairs. Keep answers short (1-2 sentences max).

IMPORTANT: Every question MUST name the specific subject (item, character, \
place, or machine) so a reader can understand it without seeing the source page. \
For example, ask "How much fuel does the Advanced Forging Machine use per hour?" \
instead of "How much fuel does it consume?".

Reply in EXACTLY this format with all 6 lines, nothing else:

FACTOID_Q: <question asking for a specific name, place, or number>
FACTOID_A: <short answer>
CONCEPTUAL_Q: <question about an event, sequence, or how/why something happens>
CONCEPTUAL_A: <short answer>
MESSY_Q: <casual vague question like a quick Google search, minimal punctuation>
MESSY_A: <short answer>

Section:
{content}"""

_FIELD_KEYS = (
    "FACTOID_Q",
    "FACTOID_A",
    "CONCEPTUAL_Q",
    "CONCEPTUAL_A",
    "MESSY_Q",
    "MESSY_A",
)


def _parse_response(response: str) -> dict[str, str] | None:
    """Parse a structured LLM response into a dict of field values.

    Returns ``None`` if any expected field is missing.
    """
    fields: dict[str, str] = {}
    for line in response.splitlines():
        for key in _FIELD_KEYS:
            prefix = f"{key}:"
            if line.strip().startswith(prefix):
                fields[key] = line.strip()[len(prefix):].strip()
                break
    if all(k in fields for k in _FIELD_KEYS):
        return fields
    return None


def _find_sections(sections_dir: Path, page_slug: str) -> list[Path]:
    """Return all section files for *page_slug* in *sections_dir*."""
    return sorted(sections_dir.glob(f"{page_slug}-*.md"))


def _page_slug_from_filename(filename: str) -> str:
    """Strip ``.md`` to get the page slug (matches silver filenames)."""
    return Path(filename).stem


def _parse_section_filename(filename: str) -> tuple[str, str]:
    """Derive page slug and section name from a section filename.

    Convention: ``<page_slug>-<section_name>.md``.
    """
    stem = Path(filename).stem
    page_slug, _, section_name = stem.partition("-")
    return page_slug, section_name


def _load_processed_pages(output_dir: Path) -> set[str]:
    """Return the set of page slugs already present in output CSVs."""
    processed: set[str] = set()
    csv_path = output_dir / "factoid.csv"
    if not csv_path.exists():
        return processed
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = _page_slug_from_filename(row["page"])
            processed.add(slug)
    return processed


class TestsetGenerator:
    """Generate retrieval test question/answer pairs from golden pages.

    Reads a golden-pages file, finds per-page sections in *sections_dir*,
    randomly picks one section per page, and asks an LLM to generate
    factoid, conceptual, and messy Q/A pairs.

    *router* is a :class:`~guild_assistant.utils.router.ModelRouter` that
    dispatches each prompt to the local or cloud model based on token count.
    """

    def __init__(
        self,
        router: ModelRouter,
        golden_pages_path: str | Path,
        sections_dir: str | Path,
        output_dir: str | Path,
        seed: int | None = None,
    ) -> None:
        self._router = router
        self._golden_pages_path = Path(golden_pages_path)
        self._sections_dir = Path(sections_dir)
        self._output_dir = Path(output_dir)
        self._rng = random.Random(seed)

    def _read_golden_pages(self) -> list[str]:
        """Read the golden pages file and return the list of filenames."""
        text = self._golden_pages_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

    def generate_for_page(self, page_filename: str) -> dict[str, str] | None:
        """Generate Q/A pairs for one golden page.

        Returns a dict with keys ``FACTOID_Q``, ``FACTOID_A``,
        ``CONCEPTUAL_Q``, ``CONCEPTUAL_A``, ``MESSY_Q``, ``MESSY_A``,
        plus ``section`` and ``page``.  Returns ``None`` if no sections
        are found or the LLM response cannot be parsed.
        """
        page_slug = _page_slug_from_filename(page_filename)
        sections = _find_sections(self._sections_dir, page_slug)

        if not sections:
            logger.warning("No sections found for %s", page_filename)
            return None

        chosen = self._rng.choice(sections)
        content = chosen.read_text(encoding="utf-8")
        _, section_title = _parse_section_filename(chosen.name)

        prompt = _PROMPT.format(
            page_title=page_slug,
            section_title=section_title,
            content=content,
        )

        response = self._router.generate(prompt)
        fields = _parse_response(response)

        if fields is None:
            logger.warning(
                "Could not parse LLM response for %s (section=%s)",
                page_filename,
                chosen.name,
            )
            return None

        fields["section"] = chosen.name
        fields["page"] = page_filename
        logger.info(
            "Generated testset for %s (section=%s)",
            page_filename,
            chosen.name,
        )
        return fields

    def generate_all(
        self,
        force: bool = False,
        max_questions: int = 10,
    ) -> list[dict[str, str]]:
        """Generate Q/A pairs for golden pages.

        Pages that already have entries in the output CSVs are skipped
        unless *force* is ``True``.  Skipped pages do not count toward
        *max_questions*.

        *max_questions* caps the number of pages actually processed.  Set
        to ``0`` to process all eligible pages.

        Returns a list of result dicts (one per successfully processed page).
        """
        pages = self._read_golden_pages()
        if not pages:
            logger.warning("No golden pages found in %s", self._golden_pages_path)
            return []

        logger.info("Found %d golden pages", len(pages))

        processed = set[str]()
        if not force:
            processed = _load_processed_pages(self._output_dir)

        results: list[dict[str, str]] = []

        for page_filename in pages:
            if max_questions and len(results) >= max_questions:
                break
            page_slug = _page_slug_from_filename(page_filename)
            if not force and page_slug in processed:
                logger.info("Skipping %s (already processed)", page_filename)
                continue
            try:
                fields = self.generate_for_page(page_filename)
                if fields is not None:
                    results.append(fields)
            except Exception:
                logger.exception(
                    "Failed to generate testset for %s", page_filename
                )

        self._write_csvs(results)

        logger.info(
            "Finished: %d processed, %d total golden pages",
            len(results),
            len(pages),
        )
        return results

    def _write_csvs(self, results: list[dict[str, str]]) -> None:
        """Append results to the three output CSV files."""
        if not results:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)

        categories = {
            "factoid": ("FACTOID_Q", "FACTOID_A"),
            "conceptual": ("CONCEPTUAL_Q", "CONCEPTUAL_A"),
            "messy": ("MESSY_Q", "MESSY_A"),
        }

        for category, (q_key, a_key) in categories.items():
            csv_path = self._output_dir / f"{category}.csv"
            file_exists = csv_path.exists()

            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["question", "answer", "section", "page"])
                for row in results:
                    writer.writerow([
                        row[q_key],
                        row[a_key],
                        row["section"],
                        row["page"],
                    ])
