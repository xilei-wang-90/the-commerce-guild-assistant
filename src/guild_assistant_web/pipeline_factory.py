"""Factory for building a fully-configured QueryPipeline."""

from pathlib import Path

from guild_assistant.rag.context_builder import ContextBuilder
from guild_assistant.rag.pipeline import QueryPipeline
from guild_assistant.rag.reranker import Reranker
from guild_assistant.rag.retriever import Retriever
from guild_assistant.utils.model_adapter import (
    CrossEncoderRerankerAdapter,
    GeminiAdapter,
    OllamaEmbeddingAdapter,
)

VALID_MODES = ("summary", "section-reverse-hyde")

_MODE_CONFIG: dict[str, str] = {
    "summary": "sandrock_wiki_summary",
    "section-reverse-hyde": "sandrock_wiki_section_reverse_hyde",
}

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "sandrock_db"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_EMBEDDING_MODEL = "all-minilm"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def create_pipeline(
    mode: str,
    *,
    db_path: Path | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
) -> QueryPipeline:
    """Build a QueryPipeline for the given mode.

    Parameters
    ----------
    mode:
        One of ``"summary"`` or ``"section-reverse-hyde"``.
    db_path:
        Path to the ChromaDB directory. Defaults to ``sandrock_db/``
        next to the project root.
    ollama_url:
        Base URL for the local Ollama instance.
    gemini_model:
        Gemini model name for generation.
    embedding_model:
        Ollama embedding model name.
    reranker_model:
        Cross-encoder model name for reranking.

    Raises
    ------
    ValueError
        If *mode* is not a recognised value.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode {mode!r}. Must be one of {VALID_MODES}."
        )

    resolved_db = db_path or DEFAULT_DB_PATH
    collection_name = _MODE_CONFIG[mode]

    embedding_adapter = OllamaEmbeddingAdapter(
        model_name=embedding_model, base_url=ollama_url,
    )
    retriever = Retriever(
        db_path=resolved_db,
        collection_name=collection_name,
        embedding_adapter=embedding_adapter,
        n_results=10,
    )
    context_builder = ContextBuilder()
    reranker_adapter = CrossEncoderRerankerAdapter(model_name=reranker_model)
    reranker = Reranker(adapter=reranker_adapter, top_n=3)

    model = GeminiAdapter(model_name=gemini_model)

    return QueryPipeline(
        retriever=retriever,
        context_builder=context_builder,
        model=model,
        reranker=reranker,
    )
