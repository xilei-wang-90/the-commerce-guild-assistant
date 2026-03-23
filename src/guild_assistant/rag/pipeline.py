"""Orchestrate the full RAG query pipeline.

Connects retrieval, context building, and generation into a single
``query()`` call.  Supports pluggable query transforms for pre-processing
(e.g. query expansion, spelling correction, synonym injection).
"""

import logging
from collections.abc import Callable

from langsmith import traceable

from guild_assistant.rag.context_builder import ContextBuilder
from guild_assistant.rag.reranker import Reranker
from guild_assistant.rag.retriever import Retriever
from guild_assistant.utils.model_adapter import ModelAdapter

logger = logging.getLogger(__name__)


class QueryPipeline:
    """End-to-end RAG pipeline: query -> retrieve -> augment -> generate.

    *retriever*        – embeds the query and finds matching documents.
    *context_builder*  – reads full-text docs and builds the LLM prompt.
    *model*            – generates the final answer (any ``ModelAdapter``).
    *reranker*         – optional cross-encoder reranker applied after retrieval.
    """

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        model: ModelAdapter,
        reranker: Reranker | None = None,
    ) -> None:
        self._retriever = retriever
        self._context_builder = context_builder
        self._model = model
        self._reranker = reranker
        self._transforms: list[Callable[[str], str]] = []

    def add_query_transform(self, fn: Callable[[str], str]) -> None:
        """Register a query transform applied before retrieval.

        Transforms are applied in the order they are added.
        """
        self._transforms.append(fn)

    @traceable(run_type="chain", name="RAG Pipeline")
    def query(self, user_query: str) -> str:
        """Run the full RAG pipeline and return the generated answer."""
        processed = user_query
        for transform in self._transforms:
            processed = transform(processed)

        results = self._retriever.retrieve(processed)
        if self._reranker:
            results = self._reranker.rerank(user_query, results)
        prompt = self._context_builder.build(user_query, results)
        answer = self._generate(prompt)

        logger.info("Pipeline answered query: %.60s…", user_query)
        return answer

    @traceable(run_type="llm", name="LLM Generation")
    def _generate(self, prompt: str) -> str:
        """Send the augmented prompt to the model (traced as an LLM step)."""
        return self._model.generate(prompt)
