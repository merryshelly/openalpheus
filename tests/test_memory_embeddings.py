"""Tests for embedding provider abstraction using llama-cpp-python.

Interface contract:
    EmbeddingProvider(model: str, base_url: str)
    async embed(text: str) -> list[float] | None
    async embed_batch(texts: list[str]) -> list[list[float] | None]

Provider loads a GGUF model via llama-cpp-python.
Returns None (not raises) when model is unavailable (graceful degradation).
"""

import pytest
from unittest.mock import patch, MagicMock
from openalph.memory.embeddings import EmbeddingProvider
import asyncio
import contextlib
import threading
import time
from openalph.memory import embeddings


class TestEmbeddingProvider:

    def test_init_stores_config(self):
        p = EmbeddingProvider(model="/path/to/model.gguf", base_url="http://localhost:11434")
        assert p.model == "/path/to/model.gguf"
        assert p.base_url == "http://localhost:11434"

    def test_default_base_url(self):
        p = EmbeddingProvider(model="/path/to/model.gguf")
        assert p.base_url == "http://localhost:11434"


class TestEmbed:

    @pytest.mark.asyncio
    async def test_returns_float_list_on_success(self):
        """Successful embedding returns list of floats."""
        provider = EmbeddingProvider(model="/fake/model.gguf")
        fake_embedding = [0.1, 0.2, 0.3, 0.4]

        mock_llm = MagicMock()
        mock_llm.create_embedding.return_value = {
            "data": [{"embedding": fake_embedding}]
        }
        provider._llm = mock_llm

        result = await provider.embed("hello world")
        assert result == fake_embedding

    @pytest.mark.asyncio
    async def test_returns_none_on_model_load_failure(self):
        """Model load failure returns None (graceful degradation)."""
        provider = EmbeddingProvider(model="/nonexistent/model.gguf")
        provider._load_failed = True  # simulate failed load

        result = await provider.embed("hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_embedding_error(self):
        """Embedding error returns None."""
        provider = EmbeddingProvider(model="/fake/model.gguf")

        mock_llm = MagicMock()
        mock_llm.create_embedding.side_effect = RuntimeError("decode failed")
        provider._llm = mock_llm

        result = await provider.embed("hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_calls_create_embedding_with_text(self):
        """Verifies the correct text is passed to create_embedding."""
        provider = EmbeddingProvider(model="/fake/model.gguf")

        mock_llm = MagicMock()
        mock_llm.create_embedding.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        provider._llm = mock_llm

        await provider.embed("test input")
        mock_llm.create_embedding.assert_called_once_with("test input")


class TestEmbedBatch:

    @pytest.mark.asyncio
    async def test_returns_list_of_embeddings(self):
        """Batch embedding returns a list with one embedding per input."""
        provider = EmbeddingProvider(model="/fake/model.gguf")

        call_count = 0
        fake_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        mock_llm = MagicMock()
        def side_effect(text):
            nonlocal call_count
            result = {"data": [{"embedding": fake_embeddings[call_count]}]}
            call_count += 1
            return result
        mock_llm.create_embedding.side_effect = side_effect
        provider._llm = mock_llm

        result = await provider.embed_batch(["a", "b", "c"])
        assert len(result) == 3
        assert result[0] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_returns_nones_on_failure(self):
        """Total failure returns list of Nones."""
        provider = EmbeddingProvider(model="/fake/model.gguf")
        provider._load_failed = True

        result = await provider.embed_batch(["a", "b"])
        assert result == [None, None]

    @pytest.mark.asyncio
    async def test_empty_batch_returns_empty(self):
        provider = EmbeddingProvider(model="/fake/model.gguf")
        result = await provider.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_partial_failure_returns_some_nones(self):
        """If one embedding in a batch fails, that entry is None."""
        provider = EmbeddingProvider(model="/fake/model.gguf")

        call_count = 0
        mock_llm = MagicMock()
        def side_effect(text):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("decode failed for this one")
            return {"data": [{"embedding": [0.1, 0.2]}]}
        mock_llm.create_embedding.side_effect = side_effect
        provider._llm = mock_llm

        result = await provider.embed_batch(["a", "b", "c"])
        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        assert result[1] is None
        assert result[2] == [0.1, 0.2]


# ---------------------------------------------------------------------------
# Concurrency safety (regression: workspace-kdsn.165)
#
# llama.cpp Llama objects are NOT thread-safe. EmbeddingProvider runs
# create_embedding() via asyncio.to_thread() against a single shared Llama
# instance (cached per workspace in memory_search._index_cache). Two concurrent
# embeds therefore call into the same llama context from different pool threads
# -> memory corruption -> SIGSEGV (observed in production as status=11/SEGV).
# These tests pin the invariant that embedding work is serialized per provider.
# ---------------------------------------------------------------------------


class _ConcurrencyTrackingLlama:
    """Fake Llama that records the peak number of overlapping create_embedding
    calls. Thread-safe counter; a small sleep widens the overlap window so a
    serialization bug is reliably observable without invoking real llama.cpp."""

    def __init__(self, dims: int = 4, window_s: float = 0.05):
        self.dims = dims
        self.window_s = window_s
        self.calls = 0
        self.max_active = 0
        self._active = 0
        self._counter_lock = threading.Lock()

    def create_embedding(self, text):
        with self._counter_lock:
            self._active += 1
            self.calls += 1
            if self._active > self.max_active:
                self.max_active = self._active
        try:
            time.sleep(self.window_s)
        finally:
            with self._counter_lock:
                self._active -= 1
        return {"data": [{"embedding": [0.0] * self.dims}]}


class TestConcurrencySafety:

    @pytest.fixture(autouse=True)
    def _no_fd_suppression(self, monkeypatch):
        # Isolate from llama_cpp's real suppress_stdout_stderr, which does
        # os.dup2 on fds 1/2 — itself unsafe to run concurrently and disruptive
        # under pytest capture. The behavior under test is serialization, not I/O.
        monkeypatch.setattr(embeddings, "suppress_stdout_stderr", contextlib.nullcontext)

    @pytest.mark.asyncio
    async def test_concurrent_embeds_are_serialized(self):
        """N concurrent embed() calls must not overlap on the shared Llama."""
        provider = EmbeddingProvider(model="/fake/model.gguf")
        fake = _ConcurrencyTrackingLlama()
        provider._llm = fake

        texts = [f"probe {i}" for i in range(8)]
        results = await asyncio.gather(*[provider.embed(t) for t in texts])

        assert fake.max_active == 1, (
            f"create_embedding overlapped (peak {fake.max_active} concurrent); "
            "llama.cpp is not thread-safe and must be serialized"
        )
        assert fake.calls == 8
        assert all(r == [0.0] * 4 for r in results)

    @pytest.mark.asyncio
    async def test_embed_and_embed_batch_do_not_overlap(self):
        """A batch index running alongside live single-query embeds must not
        overlap either (covers the background-index vs. live-query trigger)."""
        provider = EmbeddingProvider(model="/fake/model.gguf")
        fake = _ConcurrencyTrackingLlama()
        provider._llm = fake

        results = await asyncio.gather(
            provider.embed_batch([f"batch {i}" for i in range(4)]),
            provider.embed("live query a"),
            provider.embed("live query b"),
        )

        assert fake.max_active == 1, (
            f"embed/embed_batch overlapped (peak {fake.max_active} concurrent)"
        )
        batch_result, a, b = results
        assert len(batch_result) == 4
        assert a == [0.0] * 4
        assert b == [0.0] * 4
