"""Tests for guild_assistant.utils.router.ModelRouter."""

from unittest.mock import MagicMock, patch

import pytest

from guild_assistant.utils.router import ModelRouter, _DEFAULT_TOKEN_THRESHOLD

_THRESHOLD = _DEFAULT_TOKEN_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(name: str = "mock-model") -> MagicMock:
    adapter = MagicMock()
    adapter.model_name = name
    adapter.generate.return_value = f"Summary from {name}."
    return adapter


def _router_with_mocks() -> tuple[ModelRouter, MagicMock, MagicMock]:
    local = _make_adapter("llama-local")
    cloud = _make_adapter("gemini-cloud")
    router = ModelRouter(local=local, cloud=cloud)
    return router, local, cloud


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestModelRouterInit:
    def test_model_name_is_router(self):
        router, _, _ = _router_with_mocks()
        assert router.model_name == "router"

    def test_default_threshold(self):
        router, _, _ = _router_with_mocks()
        assert router._token_threshold == 11_000

    def test_custom_threshold(self):
        local = _make_adapter("local")
        cloud = _make_adapter("cloud")
        router = ModelRouter(local=local, cloud=cloud, token_threshold=500)
        assert router._token_threshold == 500


# ---------------------------------------------------------------------------
# route() — token-based dispatch
# ---------------------------------------------------------------------------

class TestModelRouterRoute:
    def test_short_prompt_routes_to_local(self):
        router, local, cloud = _router_with_mocks()
        short_prompt = "short " * 10  # well under 11 000 tokens
        assert router.route(short_prompt) is local

    def test_long_prompt_routes_to_cloud(self):
        router, local, cloud = _router_with_mocks()
        # Build a prompt guaranteed to exceed the threshold.
        long_prompt = "word " * (_THRESHOLD + 500)
        assert router.route(long_prompt) is cloud

    def test_prompt_at_exact_threshold_routes_to_cloud(self):
        """A prompt whose token count equals TOKEN_THRESHOLD goes to cloud."""
        router, local, cloud = _router_with_mocks()

        # Patch the encoder to report exactly TOKEN_THRESHOLD tokens.
        router._enc = MagicMock()
        router._enc.encode.return_value = list(range(_THRESHOLD))

        assert router.route("any prompt") is cloud

    def test_prompt_one_below_threshold_routes_to_local(self):
        router, local, cloud = _router_with_mocks()

        router._enc = MagicMock()
        router._enc.encode.return_value = list(range(_THRESHOLD - 1))

        assert router.route("any prompt") is local


# ---------------------------------------------------------------------------
# generate() — delegates to the selected adapter
# ---------------------------------------------------------------------------

class TestModelRouterGenerate:
    def test_generate_calls_local_for_short_prompt(self):
        router, local, cloud = _router_with_mocks()
        short_prompt = "tiny prompt"
        result = router.generate(short_prompt)

        local.generate.assert_called_once_with(short_prompt)
        cloud.generate.assert_not_called()
        assert result == "Summary from llama-local."

    def test_generate_calls_cloud_for_long_prompt(self):
        router, local, cloud = _router_with_mocks()
        long_prompt = "word " * (_THRESHOLD + 500)
        result = router.generate(long_prompt)

        cloud.generate.assert_called_once_with(long_prompt)
        local.generate.assert_not_called()
        assert result == "Summary from gemini-cloud."
