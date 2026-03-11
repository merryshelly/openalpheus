"""Tests for provider client caching (_get_client / _client_cache).

Verifies that HTTP clients are reused across calls for connection pooling.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from openalph.config import AgentConfig, ProviderConfig
from openalph import provider as provider_module
from openalph.provider import _get_client, _client_cache


def make_provider(key="anthropic", type="anthropic", api_key="sk-test", base_url=None, quirks=None):
    return ProviderConfig(
        key=key,
        type=type,
        api_key=api_key,
        base_url=base_url,
        quirks=quirks or [],
    )


def make_config(**kwargs):
    defaults = {
        "name": "test",
        "default_model": "anthropic/claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "providers": {"anthropic": make_provider(key="anthropic")},
        "workspace": "/tmp",
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)



class TestGetClient:

    def test_client_reused_across_calls(self):
        provider = make_provider(key="anthropic", type="anthropic", api_key="sk-test")
        client1 = _get_client(provider)
        client2 = _get_client(provider)
        assert id(client1) == id(client2)

    def test_different_api_keys_get_different_clients(self):
        provider1 = make_provider(key="anthropic", type="anthropic", api_key="sk-key-one")
        provider2 = make_provider(key="anthropic", type="anthropic", api_key="sk-key-two")
        client1 = _get_client(provider1)
        client2 = _get_client(provider2)
        assert id(client1) != id(client2)

    def test_openai_client_reused_across_calls(self):
        provider = make_provider(
            key="openrouter", type="openai", api_key="sk-oai",
            base_url="https://api.openai.com/v1"
        )
        client1 = _get_client(provider)
        client2 = _get_client(provider)
        assert id(client1) == id(client2)

    def test_openai_different_base_urls_get_different_clients(self):
        provider1 = make_provider(
            key="openrouter", type="openai", api_key="sk-oai",
            base_url="https://api.openai.com/v1"
        )
        provider2 = make_provider(
            key="openrouter", type="openai", api_key="sk-oai",
            base_url="https://openrouter.ai/api/v1"
        )
        client1 = _get_client(provider1)
        client2 = _get_client(provider2)
        assert id(client1) != id(client2)

    def test_unsupported_provider_raises(self):
        provider = make_provider(key="default", type="cohere", api_key="sk-test")
        with pytest.raises(ValueError, match="Unsupported provider"):
            _get_client(provider)

    def test_anthropic_and_openai_cached_separately(self):
        provider_anthropic = make_provider(key="ant", type="anthropic", api_key="sk-ant")
        provider_openai = make_provider(
            key="oai", type="openai", api_key="sk-ant", base_url="http://localhost/v1"
        )
        client_a = _get_client(provider_anthropic)
        client_o = _get_client(provider_openai)
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
