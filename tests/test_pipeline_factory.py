"""Tests for guild_assistant_web.pipeline_factory."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from guild_assistant_web.pipeline_factory import (
    VALID_MODES,
    _MODE_CONFIG,
    create_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCH_PREFIX = "guild_assistant_web.pipeline_factory"


def _patch_all():
    """Return a dict of started patches for all external dependencies."""
    targets = {
        "OllamaEmbeddingAdapter": f"{_PATCH_PREFIX}.OllamaEmbeddingAdapter",
        "Retriever": f"{_PATCH_PREFIX}.Retriever",
        "ContextBuilder": f"{_PATCH_PREFIX}.ContextBuilder",
        "CrossEncoderRerankerAdapter": f"{_PATCH_PREFIX}.CrossEncoderRerankerAdapter",
        "Reranker": f"{_PATCH_PREFIX}.Reranker",
        "GeminiAdapter": f"{_PATCH_PREFIX}.GeminiAdapter",
        "QueryPipeline": f"{_PATCH_PREFIX}.QueryPipeline",
    }
    patches = {}
    mocks = {}
    for name, target in targets.items():
        p = patch(target)
        mocks[name] = p.start()
        patches[name] = p
    return mocks, patches


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreatePipeline:
    def test_summary_mode_uses_correct_collection(self):
        mocks, patches = _patch_all()
        try:
            create_pipeline("summary")
            mocks["Retriever"].assert_called_once()
            call_kwargs = mocks["Retriever"].call_args
            assert call_kwargs.kwargs["collection_name"] == "sandrock_wiki_summary"
        finally:
            for p in patches.values():
                p.stop()

    def test_section_mode_uses_correct_collection(self):
        mocks, patches = _patch_all()
        try:
            create_pipeline("section-reverse-hyde")
            call_kwargs = mocks["Retriever"].call_args
            assert call_kwargs.kwargs["collection_name"] == "sandrock_wiki_section_reverse_hyde"
        finally:
            for p in patches.values():
                p.stop()

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            create_pipeline("unknown")

    def test_returns_query_pipeline(self):
        mocks, patches = _patch_all()
        try:
            result = create_pipeline("summary")
            assert result is mocks["QueryPipeline"].return_value
        finally:
            for p in patches.values():
                p.stop()

    def test_custom_db_path(self):
        mocks, patches = _patch_all()
        try:
            custom_path = Path("/tmp/test_db")
            create_pipeline("summary", db_path=custom_path)
            call_kwargs = mocks["Retriever"].call_args
            assert call_kwargs.kwargs["db_path"] == custom_path
        finally:
            for p in patches.values():
                p.stop()

    def test_custom_ollama_url(self):
        mocks, patches = _patch_all()
        try:
            create_pipeline("summary", ollama_url="http://custom:1234")
            call_kwargs = mocks["OllamaEmbeddingAdapter"].call_args
            assert call_kwargs.kwargs["base_url"] == "http://custom:1234"
        finally:
            for p in patches.values():
                p.stop()

    def test_custom_gemini_model(self):
        mocks, patches = _patch_all()
        try:
            create_pipeline("summary", gemini_model="gemini-custom")
            call_kwargs = mocks["GeminiAdapter"].call_args
            assert call_kwargs.kwargs["model_name"] == "gemini-custom"
        finally:
            for p in patches.values():
                p.stop()

    def test_custom_reranker_model(self):
        mocks, patches = _patch_all()
        try:
            create_pipeline("summary", reranker_model="custom-reranker")
            call_kwargs = mocks["CrossEncoderRerankerAdapter"].call_args
            assert call_kwargs.kwargs["model_name"] == "custom-reranker"
        finally:
            for p in patches.values():
                p.stop()

    def test_pipeline_receives_reranker(self):
        mocks, patches = _patch_all()
        try:
            create_pipeline("summary")
            call_kwargs = mocks["QueryPipeline"].call_args
            assert call_kwargs.kwargs["reranker"] is mocks["Reranker"].return_value
        finally:
            for p in patches.values():
                p.stop()


class TestConstants:
    def test_valid_modes(self):
        assert "summary" in VALID_MODES
        assert "section-reverse-hyde" in VALID_MODES

    def test_mode_config_covers_all_modes(self):
        for mode in VALID_MODES:
            assert mode in _MODE_CONFIG
