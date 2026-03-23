"""Build an augmented prompt from retrieved documents and a user query.

Reads the full-text silver-tier files referenced by retrieval results
and formats them into a structured prompt that instructs the LLM to
answer only from the provided documents.
"""

import logging
from pathlib import Path

from guild_assistant.rag.retriever import RetrievalResult
from guild_assistant.rag_setup.question_generator import _parse_chunk_filename

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a knowledgeable assistant for the video game "My Time at Sandrock".
Answer the user's question using ONLY the reference documents provided below.
Each document may be a full wiki page or a section of a page.
If the documents do not contain enough information to answer the question,
say "I don't have enough information to answer that question."
Do not make up facts or use knowledge outside of these documents.
When multiple documents cover the same topic, combine their information
into a single coherent answer.

{documents}

User question: {query}"""


def _source_label(silver_path: str) -> str:
    """Derive a human-readable source label from a silver-tier file path.

    Uses the filename convention ``<page_slug>-<section_name>.md`` to
    produce labels like ``Source: {yakmel, Section - obtaining}`` or
    ``Source: {yakmel}`` for full-page files.
    """
    filename = Path(silver_path).name
    page_slug, section_name = _parse_chunk_filename(filename)
    if section_name:
        return f"--- Source: {{{page_slug}, Section - {section_name}}} ---"
    return f"--- Source: {{{page_slug}}} ---"


class ContextBuilder:
    """Read full-text pages and format a RAG prompt.

    For each retrieval result, reads the corresponding silver-tier file
    and includes its full text in the prompt sent to the LLM.
    """

    def build(self, query: str, results: list[RetrievalResult]) -> str:
        """Build an augmented prompt from *query* and *results*.

        Missing silver files are skipped with a warning.
        """
        doc_sections: list[str] = []

        for result in results:
            path = Path(result.silver_path)
            if not path.exists():
                logger.warning("Silver file not found: %s", path)
                continue
            content = path.read_text(encoding="utf-8")
            header = _source_label(result.silver_path)
            doc_sections.append(f"{header}\n{content}")

        documents_block = "\n\n".join(doc_sections)
        return _PROMPT_TEMPLATE.format(documents=documents_block, query=query)
