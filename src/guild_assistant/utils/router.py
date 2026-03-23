"""Token-based model router.

Routes prompts to a local LLM (Ollama/llama) when they are small enough to
fit comfortably within the local model's context window, and to a cloud model
(Gemini) for larger prompts that would exceed it.

Used by both the summarization pipeline and the RAG query pipeline.

Token counting uses tiktoken's ``cl100k_base`` encoding as a portable
approximation that works across model families.
"""

import tiktoken

from guild_assistant.utils.model_adapter import ModelAdapter

_DEFAULT_TOKEN_THRESHOLD = 11_000
_ENCODING_NAME = "cl100k_base"


class ModelRouter(ModelAdapter):
    """Route prompts to the best available model based on token count.

    Prompts with fewer than ``token_threshold`` tokens are sent to *local*
    (the Ollama/llama adapter).  Larger prompts are sent to *cloud* (the
    Gemini adapter).

    Because ``ModelRouter`` implements the ``ModelAdapter`` interface it can
    be used as a drop-in replacement anywhere a ``ModelAdapter`` is expected.
    """

    def __init__(
        self,
        local: ModelAdapter,
        cloud: ModelAdapter,
        token_threshold: int = _DEFAULT_TOKEN_THRESHOLD,
    ) -> None:
        super().__init__("router")
        self._local = local
        self._cloud = cloud
        self._token_threshold = token_threshold
        self._enc = tiktoken.get_encoding(_ENCODING_NAME)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def route(self, prompt: str) -> ModelAdapter:
        """Return the appropriate adapter for *prompt*.

        Counts the tokens in *prompt* using ``cl100k_base`` encoding.  If the
        count is below ``token_threshold``, the local adapter is returned;
        otherwise the cloud adapter is returned.
        """
        token_count = len(self._enc.encode(prompt))
        return self._local if token_count < self._token_threshold else self._cloud

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Route *prompt* to the selected adapter and return its output."""
        return self.route(prompt).generate(prompt)
