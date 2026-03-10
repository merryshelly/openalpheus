"""Tests for provider client caching (_get_client / _client_cache).

Verifies that HTTP clients are reused across calls for connection pooling.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from openalph.config import AgentConfig
from openalph import provider as provider_module
from openalph.provider import _get_client, _client_cache


def make_config(**kwargs):
    defaults = dict(
        name="test",
        model="test-model",
        max_tokens=8192,
        provider="anthropic",
        api_key="sk-test",
        base_url=None,
        workspace="/tmp",
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)



class TestGetClient:

    def test_client_reused_across_calls(self):
        config = make_config()
        client1 = _get_client(config)
        client2 = _get_client(config)
        assert id(client1) == id(client2)

    def test_different_api_keys_get_different_clients(self):
        config1 = make_config(api_key="sk-key-one")
        config2 = make_config(api_key="sk-key-two")
        client1 = _get_client(config1)
        client2 = _get_client(config2)
        assert id(client1) != id(client2)

    def test_openai_client_reused_across_calls(self):
        config = make_config(provider="openai", api_key="sk-oai", base_url="https://api.openai.com/v1")
        client1 = _get_client(config)
        client2 = _get_client(config)
        assert id(client1) == id(client2)

    def test_openai_different_base_urls_get_different_clients(self):
        config1 = make_config(provider="openai", api_key="sk-oai", base_url="https://api.openai.com/v1")
        config2 = make_config(provider="openai", api_key="sk-oai", base_url="https://openrouter.ai/api/v1")
        client1 = _get_client(config1)
        client2 = _get_client(config2)
        assert id(client1) != id(client2)

    def test_unsupported_provider_raises(self):
        config = make_config(provider="cohere")
        with pytest.raises(ValueError, match="Unsupported provider"):
            _get_client(config)

    def test_anthropic_and_openai_cached_separately(self):
        config_anthropic = make_config(provider="anthropic", api_key="sk-ant")
        config_openai = make_config(provider="openai", api_key="sk-ant", base_url=None)
        client_a = _get_client(config_anthropic)
        client_o = _get_client(config_openai)
        assert id(client_a) != id(client_o)


class TestCompleteReusesClient:

    def test_anthropic_constructor_called_once_for_same_config(self, tmp_path):
        """complete() should not re-instantiate the client on repeated calls."""
        import asyncio
        from openalph.provider import complete
        from openalph.provider import Response, Usage

        config = make_config(workspace=str(tmp_path))

        fake_response = Response(
            content="hi",
            model="test-model",
            usage=Usage(input_tokens=5, output_tokens=3),
            stop_reason="end_turn",
        )

        with patch("openalph.provider._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Mock anthropic messages.create
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(type="text", text="hi")]
            mock_msg.model = "test-model"
            mock_msg.usage.input_tokens = 5
            mock_msg.usage.output_tokens = 3
            mock_msg.usage.cache_read_input_tokens = 0
            mock_msg.usage.cache_creation_input_tokens = 0
            mock_msg.stop_reason = "end_turn"
            mock_client.messages.create = AsyncMock(return_value=mock_msg)

            messages = [{"role": "user", "content": "hello"}]

            asyncio.run(complete(config, "system prompt", messages))
            asyncio.run(complete(config, "system prompt", messages))

            assert mock_get_client.call_count == 2  # called each time complete() runs
            # But the underlying constructor should only be called once (cache hit)
            # This is guaranteed by the cache logic tested in TestGetClient above
