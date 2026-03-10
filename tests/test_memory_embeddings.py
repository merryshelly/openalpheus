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
