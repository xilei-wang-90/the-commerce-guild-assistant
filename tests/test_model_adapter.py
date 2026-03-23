"""Tests for guild_assistant.rag.model_adapter."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from guild_assistant.utils.model_adapter import (
    CrossEncoderRerankerAdapter,
    EmbeddingAdapter,
    GeminiAdapter,
    ModelAdapter,
    OllamaAdapter,
    OllamaEmbeddingAdapter,
    RerankerAdapter,
)


# ---------------------------------------------------------------------------
# ModelAdapter (abstract base)
# ---------------------------------------------------------------------------

class TestModelAdapterABC:
    """ModelAdapter cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ModelAdapter("some-model")


# ---------------------------------------------------------------------------
# OllamaAdapter
# ---------------------------------------------------------------------------

class TestOllamaAdapterInit:
    def test_defaults(self):
        adapter = OllamaAdapter()
        assert adapter.model_name == "llama3.1"
        assert adapter._base_url == "http://localhost:11434"

    def test_custom_values(self):
        adapter = OllamaAdapter(model_name="mistral", base_url="http://myhost:9999/")
        assert adapter.model_name == "mistral"
        assert adapter._base_url == "http://myhost:9999"  # trailing slash stripped


class TestOllamaAdapterGenerate:
    @patch("guild_assistant.utils.model_adapter.requests.post")
    def test_returns_response_text(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "A concise summary."}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        adapter = OllamaAdapter(model_name="llama3.1")
        result = adapter.generate("Summarize this.")

        assert result == "A concise summary."
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1",
                "prompt": "Summarize this.",
                "stream": False,
            },
            timeout=600,
        )

    @patch("guild_assistant.utils.model_adapter.requests.post")
    def test_raises_on_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_post.return_value = mock_resp

        adapter = OllamaAdapter()
        with pytest.raises(requests.HTTPError):
            adapter.generate("prompt")

    @patch("guild_assistant.utils.model_adapter.requests.post")
    def test_raises_on_connection_error(self, mock_post):
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        adapter = OllamaAdapter()
        with pytest.raises(requests.ConnectionError):
            adapter.generate("prompt")


# ---------------------------------------------------------------------------
# GeminiAdapter
# ---------------------------------------------------------------------------

class TestGeminiAdapterInit:
    def test_defaults(self):
        adapter = GeminiAdapter(api_key="test-key")
        assert adapter.model_name == "gemini-3-flash-preview"
        assert adapter._api_key == "test-key"

    def test_custom_model_name(self):
        adapter = GeminiAdapter(model_name="gemini-custom", api_key="k")
        assert adapter.model_name == "gemini-custom"

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        adapter = GeminiAdapter()
        assert adapter._api_key == "env-key"

    def test_api_key_none_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        adapter = GeminiAdapter()
        assert adapter._api_key is None


class TestGeminiAdapterGenerate:
    @patch("guild_assistant.utils.model_adapter.requests.post")
    def test_returns_response_text(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "Gemini summary."}]}}
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        adapter = GeminiAdapter(api_key="test-key")
        result = adapter.generate("Summarize this.")

        assert result == "Gemini summary."
        mock_post.assert_called_once_with(
            "https://generativelanguage.googleapis.com/v1beta/models"
            "/gemini-3-flash-preview:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": "test-key",
            },
            json={"contents": [{"parts": [{"text": "Summarize this."}]}]},
            timeout=600,
        )

    def test_raises_if_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        adapter = GeminiAdapter()
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            adapter.generate("prompt")

    @patch("guild_assistant.utils.model_adapter.requests.post")
    def test_raises_on_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("403")
        mock_post.return_value = mock_resp

        adapter = GeminiAdapter(api_key="key")
        with pytest.raises(requests.HTTPError):
            adapter.generate("prompt")


# ---------------------------------------------------------------------------
# EmbeddingAdapter (abstract base)
# ---------------------------------------------------------------------------

class TestEmbeddingAdapterABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            EmbeddingAdapter("some-model")


# ---------------------------------------------------------------------------
# OllamaEmbeddingAdapter
# ---------------------------------------------------------------------------

class TestOllamaEmbeddingAdapterInit:
    def test_defaults(self):
        adapter = OllamaEmbeddingAdapter()
        assert adapter.model_name == "all-minilm"
        assert adapter._base_url == "http://localhost:11434"

    def test_custom_values(self):
        adapter = OllamaEmbeddingAdapter(model_name="nomic-embed", base_url="http://myhost:9999/")
        assert adapter.model_name == "nomic-embed"
        assert adapter._base_url == "http://myhost:9999"  # trailing slash stripped


class TestOllamaEmbeddingAdapterEmbed:
    @patch("guild_assistant.utils.model_adapter.requests.post")
    def test_returns_vector(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        adapter = OllamaEmbeddingAdapter(model_name="all-minilm")
        result = adapter.embed("some text")

        assert result == [0.1, 0.2, 0.3]
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/embed",
            json={"model": "all-minilm", "input": "some text"},
            timeout=60,
        )

    @patch("guild_assistant.utils.model_adapter.requests.post")
    def test_raises_on_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        mock_post.return_value = mock_resp

        adapter = OllamaEmbeddingAdapter()
        with pytest.raises(requests.HTTPError):
            adapter.embed("text")

    @patch("guild_assistant.utils.model_adapter.requests.post")
    def test_raises_on_connection_error(self, mock_post):
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        adapter = OllamaEmbeddingAdapter()
        with pytest.raises(requests.ConnectionError):
            adapter.embed("text")


# ---------------------------------------------------------------------------
# RerankerAdapter (abstract base)
# ---------------------------------------------------------------------------

class TestRerankerAdapterABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            RerankerAdapter("some-model")


# ---------------------------------------------------------------------------
# CrossEncoderRerankerAdapter
# ---------------------------------------------------------------------------

class TestCrossEncoderRerankerAdapterInit:
    @patch("sentence_transformers.CrossEncoder")
    def test_defaults(self, mock_ce):
        adapter = CrossEncoderRerankerAdapter()
        assert adapter.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        mock_ce.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @patch("sentence_transformers.CrossEncoder")
    def test_custom_model(self, mock_ce):
        adapter = CrossEncoderRerankerAdapter(model_name="cross-encoder/custom")
        assert adapter.model_name == "cross-encoder/custom"
        mock_ce.assert_called_once_with("cross-encoder/custom")


class TestCrossEncoderRerankerAdapterScore:
    @patch("sentence_transformers.CrossEncoder")
    def test_returns_scores_as_list(self, mock_ce):
        import numpy as np

        mock_instance = MagicMock()
        mock_instance.predict.return_value = np.array([0.9, 0.1])
        mock_ce.return_value = mock_instance

        adapter = CrossEncoderRerankerAdapter()
        result = adapter.score([("q", "doc1"), ("q", "doc2")])

        assert result == [0.9, 0.1]
        mock_instance.predict.assert_called_once_with([("q", "doc1"), ("q", "doc2")])
