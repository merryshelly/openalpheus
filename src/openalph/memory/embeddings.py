"""Embedding provider abstraction using llama-cpp-python with GGUF models.

Loads a GGUF embedding model in-process via llama-cpp-python.
Graceful degradation: returns None on any failure (model not found,
library not installed, etc.) — never raises.
"""

import logging

logger = logging.getLogger(__name__)

# Default model path (nomic-embed-text-v1.5, Q8_0 quantization, 768 dims)
DEFAULT_MODEL_PATH = "/opt/openalph/models/nomic-embed-text-v1.5.Q8_0.gguf"
DEFAULT_BASE_URL = "http://localhost:11434"  # kept for interface compat


class EmbeddingProvider:
    """Embedding provider using a local GGUF model via llama-cpp-python.

    Falls back gracefully: if the model can't be loaded, embed() returns None.
    """

    def __init__(self, model: str = DEFAULT_MODEL_PATH,
                 base_url: str = DEFAULT_BASE_URL):
        self.model = model
        self.base_url = base_url  # kept for interface compatibility
        self._llm = None
        self._load_failed = False

    def _ensure_model(self):
        """Lazy-load the GGUF model. Returns the Llama instance or None."""
        if self._llm is not None:
            return self._llm
        if self._load_failed:
            return None
        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=self.model,
                embedding=True,
                verbose=False,
                n_ctx=2048,
            )
            logger.info("Loaded embedding model: %s", self.model)
            return self._llm
        except Exception as e:
            self._load_failed = True
            logger.warning("Failed to load embedding model %s: %s", self.model, e)
            return None

    async def embed(self, text: str) -> list[float] | None:
        """Embed a single text string.

        Returns a list of floats (the embedding vector), or None on any failure.
        """
        try:
            llm = self._ensure_model()
            if llm is None:
                return None
            result = llm.create_embedding(text)
            return result["data"][0]["embedding"]
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return None

    async def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed multiple texts. Returns a list with one embedding per input.

        On total failure, returns a list of Nones. On partial failure,
        individual entries may be None.
        """
        if not texts:
            return []
        try:
            llm = self._ensure_model()
            if llm is None:
                return [None] * len(texts)
            # Sequential embedding — batch decode has issues in llama-cpp-python 0.3.x
            results = []
            for text in texts:
                try:
                    result = llm.create_embedding(text)
                    results.append(result["data"][0]["embedding"])
                except Exception as e:
                    logger.warning("Embedding failed for chunk: %s", e)
                    results.append(None)
            return results
        except Exception as e:
            logger.warning("Batch embedding failed: %s", e)
            return [None] * len(texts)
