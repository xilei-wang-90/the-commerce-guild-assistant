"""Tests for guild_assistant.rag.pipeline."""

from unittest.mock import MagicMock

import pytest

from guild_assistant.rag.pipeline import QueryPipeline
from guild_assistant.rag.retriever import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retrieval_result(doc_id: str = "doc1") -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id, summary="summary", silver_path="/silver/doc.md", distance=0.1,
    )


def _make_pipeline() -> tuple[QueryPipeline, MagicMock, MagicMock, MagicMock]:
    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_retrieval_result()]

    context_builder = MagicMock()
    context_builder.build.return_value = "augmented prompt"

    model = MagicMock()
    model.generate.return_value = "The answer is 42."

    pipeline = QueryPipeline(
        retriever=retriever,
        context_builder=context_builder,
        model=model,
    )
    return pipeline, retriever, context_builder, model


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------

class TestQueryPipeline:
    def test_full_flow(self):
        pipeline, retriever, context_builder, model = _make_pipeline()

        answer = pipeline.query("What is a Yakmel?")

        retriever.retrieve.assert_called_once_with("What is a Yakmel?")
        context_builder.build.assert_called_once()
        model.generate.assert_called_once_with("augmented prompt")
        assert answer == "The answer is 42."

    def test_passes_original_query_to_context_builder(self):
        pipeline, _, context_builder, _ = _make_pipeline()

        pipeline.query("original question")

        args = context_builder.build.call_args
        assert args[0][0] == "original question"

    def test_passes_retrieval_results_to_context_builder(self):
        pipeline, retriever, context_builder, _ = _make_pipeline()
        results = [_make_retrieval_result("a"), _make_retrieval_result("b")]
        retriever.retrieve.return_value = results

        pipeline.query("query")

        args = context_builder.build.call_args
        assert args[0][1] is results


# ---------------------------------------------------------------------------
# Query transforms
# ---------------------------------------------------------------------------

class TestQueryTransforms:
    def test_single_transform(self):
        pipeline, retriever, _, _ = _make_pipeline()
        pipeline.add_query_transform(str.lower)

        pipeline.query("UPPERCASE QUERY")

        retriever.retrieve.assert_called_once_with("uppercase query")

    def test_multiple_transforms_applied_in_order(self):
        pipeline, retriever, _, _ = _make_pipeline()
        pipeline.add_query_transform(str.strip)
        pipeline.add_query_transform(str.lower)

        pipeline.query("  HELLO  ")

        retriever.retrieve.assert_called_once_with("hello")

    def test_transform_does_not_affect_context_builder_query(self):
        pipeline, _, context_builder, _ = _make_pipeline()
        pipeline.add_query_transform(str.upper)

        pipeline.query("original")

        args = context_builder.build.call_args
        assert args[0][0] == "original"

    def test_no_transforms_passes_query_unchanged(self):
        pipeline, retriever, _, _ = _make_pipeline()

        pipeline.query("unchanged")

        retriever.retrieve.assert_called_once_with("unchanged")


# ---------------------------------------------------------------------------
# Reranker integration
# ---------------------------------------------------------------------------

class TestPipelineWithReranker:
    def test_reranker_called_between_retrieve_and_build(self):
        pipeline, retriever, context_builder, model = _make_pipeline()
        reranker = MagicMock()
        reranked_results = [_make_retrieval_result("reranked")]
        reranker.rerank.return_value = reranked_results
        pipeline._reranker = reranker

        pipeline.query("What is a Yakmel?")

        reranker.rerank.assert_called_once()
        # Context builder should receive the reranked results
        args = context_builder.build.call_args
        assert args[0][1] is reranked_results

    def test_reranker_receives_original_query(self):
        pipeline, _, _, _ = _make_pipeline()
        pipeline.add_query_transform(str.upper)
        reranker = MagicMock()
        reranker.rerank.return_value = [_make_retrieval_result()]
        pipeline._reranker = reranker

        pipeline.query("original query")

        args = reranker.rerank.call_args
        assert args[0][0] == "original query"

    def test_pipeline_works_without_reranker(self):
        pipeline, retriever, context_builder, model = _make_pipeline()

        answer = pipeline.query("question")

        retriever.retrieve.assert_called_once()
        context_builder.build.assert_called_once()
        model.generate.assert_called_once()
        assert answer == "The answer is 42."
