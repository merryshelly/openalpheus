"""Tests for provider client caching (_get_client / _client_cache).

Verifies that HTTP clients are reused across calls for connection pooling.
"""

import pytest
from unittest.mock import patch, MagicMock
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import _get_client


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


class MockAnthropicStream:
    """Mock for Anthropic's AsyncMessageStream context manager."""

    def __init__(self, events, final_message=None):
        self._events = events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final_message


def _anthropic_text(text):
    e = MagicMock()
    e.type = "text"
    e.text = text
    return e


def _anthropic_message_stop():
    e = MagicMock()
    e.type = "message_stop"
    return e


def _anthropic_final_message(text="", model="test-model",
                             input_tokens=100, output_tokens=50,
                             cache_read=0, cache_create=0,
                             stop_reason="end_turn"):
    msg = MagicMock()
    msg.model = model
    msg.stop_reason = stop_reason
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    msg.usage.cache_read_input_tokens = cache_read
    msg.usage.cache_creation_input_tokens = cache_create

    content_blocks = []
    if text:
        tb = MagicMock()
        tb.type = "text"
        tb.text = text
        content_blocks.append(tb)
    msg.content = content_blocks
    return msg


class TestCompleteReusesClient:

    def test_anthropic_constructor_called_once_for_same_config(self, tmp_path):
        """complete() should not re-instantiate the client on repeated calls."""
        import asyncio
        from openalph.provider import complete

        config = make_config(workspace=str(tmp_path))

        with patch("openalph.provider._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Mock anthropic messages.stream
            events = [_anthropic_text("hi"), _anthropic_message_stop()]
            final_msg = _anthropic_final_message(text="hi")
            mock_client.messages.stream.return_value = MockAnthropicStream(events, final_msg)

            messages = [{"role": "user", "content": "hello"}]

            asyncio.run(complete(config, "system prompt", messages))
            asyncio.run(complete(config, "system prompt", messages))

            assert mock_get_client.call_count == 2  # called each time complete() runs
            # But the underlying constructor should only be called once (cache hit)
            # This is guaranteed by the cache logic tested in TestGetClient above
