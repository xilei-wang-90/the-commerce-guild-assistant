"""Adapter pattern for LLM model access.

Provides base classes and concrete implementations for two kinds of model
interactions:

* ``ModelAdapter`` / ``generate`` — send a prompt and receive text (used by
  the summarisation pipeline).
* ``EmbeddingAdapter`` / ``embed`` — embed a piece of text and receive a
  float vector (used by the embedding pipeline).
"""

import os
from abc import ABC, abstractmethod

import requests


class ModelAdapter(ABC):
    """Base class for LLM adapters.

    Subclasses must implement ``generate`` to send a prompt to a specific
    model backend and return the generated text.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send *prompt* to the model and return the generated text."""


class EmbeddingAdapter(ABC):
    """Base class for embedding adapters.

    Subclasses must implement ``embed`` to send a piece of text to an
    embedding model and return its vector representation.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed *text* and return its float vector."""


class OllamaAdapter(ModelAdapter):
    """Adapter for models served by a local Ollama instance."""

    def __init__(
        self,
        model_name: str = "llama3.1",
        base_url: str = "http://localhost:11434",
    ) -> None:
        super().__init__(model_name)
        self._base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        """Send *prompt* to the Ollama ``/api/generate`` endpoint."""
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()["response"]


class GeminiAdapter(ModelAdapter):
    """Adapter for Google Gemini models via the Generative Language REST API.

    The API key is read from the ``GEMINI_API_KEY`` environment variable when
    not supplied directly.  The key is only required when ``generate`` is
    called, so instances can be created freely (e.g. for routing decisions)
    without a valid key.
    """

    _API_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models"
        "/{model}:generateContent"
    )

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        api_key: str | None = None,
    ) -> None:
        super().__init__(model_name)
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")

    def generate(self, prompt: str) -> str:
        """Send *prompt* to the Gemini ``generateContent`` endpoint."""
        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or pass api_key= explicitly."
            )
        url = self._API_URL.format(model=self.model_name)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


class RerankerAdapter(ABC):
    """Base class for reranking adapters.

    Subclasses must implement ``score`` to score a list of
    (query, document) pairs and return relevance scores.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score each (query, document) pair and return relevance scores."""


class CrossEncoderRerankerAdapter(RerankerAdapter):
    """Reranking adapter using a sentence-transformers CrossEncoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        super().__init__(model_name)
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score each (query, document) pair with the cross-encoder."""
        scores = self._model.predict(pairs)
        return scores.tolist()


class OllamaEmbeddingAdapter(EmbeddingAdapter):
    """Embedding adapter for models served by a local Ollama instance."""

    def __init__(
        self,
        model_name: str = "all-minilm",
        base_url: str = "http://localhost:11434",
    ) -> None:
        super().__init__(model_name)
        self._base_url = base_url.rstrip("/")

    def embed(self, text: str) -> list[float]:
        """Send *text* to the Ollama ``/api/embed`` endpoint and return its vector."""
        url = f"{self._base_url}/api/embed"
        resp = requests.post(
            url,
            json={"model": self.model_name, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
