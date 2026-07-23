"""Embedding provider abstraction using llama-cpp-python with GGUF models.

Loads a GGUF embedding model in-process via llama-cpp-python.
Graceful degradation: returns None on any failure (model not found,
library not installed, etc.) — never raises.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# Default model path (nomic-embed-text-v1.5, Q8_0 quantization, 768 dims)
DEFAULT_MODEL_PATH = "/opt/openalph/models/nomic-embed-text-v1.5.Q8_0.gguf"
DEFAULT_BASE_URL = "http://localhost:11434"  # kept for interface compat

# Suppress C-level llama.cpp output (stderr warnings during embedding calls).
# Falls back to no-op if llama_cpp is not installed.
try:
    from llama_cpp import suppress_stdout_stderr
except ImportError:
    from contextlib import nullcontext as suppress_stdout_stderr


class EmbeddingProvider:
    """Embedding provider using a local GGUF model via llama-cpp-python.

    Falls back gracefully: if the model can't be loaded, embed() returns None.
    Embedding calls run in a thread pool to avoid blocking the async event loop.
    """

    def __init__(self, model: str = DEFAULT_MODEL_PATH,
                 base_url: str = DEFAULT_BASE_URL):
        self.model = model
        self.base_url = base_url  # kept for interface compatibility
        self._llm = None
        self._load_failed = False
        # llama.cpp Llama objects are NOT thread-safe. Concurrent
        # create_embedding() on a shared instance corrupts the context -> SIGSEGV.
        # Serialize all embedding work (and the lazy model load) per provider.
        self._lock = asyncio.Lock()

    def _ensure_model(self):
        """Lazy-load the GGUF model. Returns the Llama instance or None."""
        if self._llm is not None:
            return self._llm
        if self._load_failed:
            return None
        try:
            from llama_cpp import Llama
            with suppress_stdout_stderr():
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

    @staticmethod
    def _create_embedding(llm, text: str):
        """Synchronous embedding call with C-level output suppressed."""
        with suppress_stdout_stderr():
            return llm.create_embedding(text)

    def _ensure_and_create(self, text: str):
        """Load the model (if needed) and embed one text -- all synchronous, so
        it can be dispatched to a worker thread in one hop (BUG-9)."""
        llm = self._ensure_model()
        if llm is None:
            return None
        return self._create_embedding(llm, text)

    def _ensure_and_embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Load the model (if needed) and embed a batch, synchronously (BUG-9)."""
        llm = self._ensure_model()
        if llm is None:
            return [None] * len(texts)
        return self._embed_batch_sync(llm, texts)

    async def embed(self, text: str) -> list[float] | None:
        """Embed a single text string.

        Returns a list of floats (the embedding vector), or None on any failure.
        Runs in a thread pool to avoid blocking the async event loop.
        """
        try:
            async with self._lock:
                # BUG-9: run the model load inside the thread as well. Only the
                # create_embedding call used to be dispatched via to_thread; the
                # first `_ensure_model()` did a multi-hundred-MB GGUF `Llama(...)`
                # load INLINE in the coroutine, freezing the event loop (Matrix
                # sync stalls, other rooms' turns halt, heartbeat timers slip).
                result = await asyncio.to_thread(self._ensure_and_create, text)
            if result is None:
                return None
            return result["data"][0]["embedding"]
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return None

    def _embed_batch_sync(self, llm, texts: list[str]) -> list[list[float] | None]:
        """Synchronous batch embedding with C-level output suppressed."""
        results = []
        for text in texts:
            try:
                result = self._create_embedding(llm, text)
                results.append(result["data"][0]["embedding"])
            except Exception as e:
                logger.warning("Embedding failed for chunk: %s", e)
                results.append(None)
        return results

    async def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed multiple texts. Returns a list with one embedding per input.

        On total failure, returns a list of Nones. On partial failure,
        individual entries may be None.
        Runs in a thread pool to avoid blocking the async event loop.
        """
        if not texts:
            return []
        try:
            async with self._lock:
                # BUG-9: load the model inside the thread (see embed()).
                return await asyncio.to_thread(self._ensure_and_embed_batch, texts)
        except Exception as e:
            logger.warning("Batch embedding failed: %s", e)
            return [None] * len(texts)
