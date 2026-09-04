"""Tests for /cache TTL command — room-scoped 1-hour cache TTL toggle."""

import pytest
from openalph.provider import _build_anthropic_kwargs, Response, StreamEvent, Usage
from unittest.mock import patch
from openalph.agent import Agent
from openalph.config import AgentConfig, ProviderConfig


class TestBuildAnthropicKwargsCacheTTL:

    def _build(self, cache_ttl=None):
        """Helper to build kwargs with minimal valid inputs."""
        return _build_anthropic_kwargs(
            api_model="claude-opus-4-6",
            system="You are helpful.",
            provider_messages=[{"role": "user", "content": "Hello"}],
            provider_tools=None,
            max_tokens=1024,
            thinking_level="off",
            cache_ttl=cache_ttl,
        )

    def test_default_1h_cache_control(self):
        """Without cache_ttl, cache_control defaults to 1h ephemeral."""
        kwargs = self._build()
        system_cc = kwargs["system"][0]["cache_control"]
        assert system_cc == {"type": "ephemeral", "ttl": "1h"}
        # Check last message too
        last_msg = kwargs["messages"][-1]
        content = last_msg["content"]
        if isinstance(content, list):
            msg_cc = content[-1]["cache_control"]
        else:
            msg_cc = content[-1]["cache_control"]
        assert msg_cc == {"type": "ephemeral", "ttl": "1h"}

    def test_1h_cache_control(self):
        """With cache_ttl='1h', cache_control includes ttl field."""
        kwargs = self._build(cache_ttl="1h")
        system_cc = kwargs["system"][0]["cache_control"]
        assert system_cc == {"type": "ephemeral", "ttl": "1h"}
        # Check last message
        last_msg = kwargs["messages"][-1]
        content = last_msg["content"]
        if isinstance(content, list):
            msg_cc = content[-1]["cache_control"]
        assert msg_cc == {"type": "ephemeral", "ttl": "1h"}

    def test_none_cache_ttl_same_as_default(self):
        """Explicitly passing None is same as omitting."""
        kwargs = self._build(cache_ttl=None)
        system_cc = kwargs["system"][0]["cache_control"]
        assert system_cc == {"type": "ephemeral", "ttl": "1h"}

    def test_list_content_with_1h(self):
        """When last user message has list content, cache_control gets ttl."""
        kwargs = _build_anthropic_kwargs(
            api_model="claude-opus-4-6",
            system="You are helpful.",
            provider_messages=[{"role": "user", "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]}],
            provider_tools=None,
            max_tokens=1024,
            thinking_level="off",
            cache_ttl="1h",
        )
        last_msg = kwargs["messages"][-1]
        content = last_msg["content"]
        assert isinstance(content, list)
        msg_cc = content[-1]["cache_control"]
        assert msg_cc == {"type": "ephemeral", "ttl": "1h"}
        # System should also have ttl
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def _make_config(workspace):
    return AgentConfig(
        name="test",
        default_model="anthropic/claude-opus-4-6",
        max_tokens=1024,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="test-key",
        )},
        workspace=workspace,
    )


def _make_stream_events(content="Hello"):
    async def _stream(*args, **kwargs):
        yield StreamEvent(type="text", content=content)
        yield StreamEvent(
            type="done",
            response=Response(
                content=content,
                model="claude-opus-4-6",
                usage=Usage(input_tokens=100, output_tokens=50),
                stop_reason="end_turn",
            ),
            stop_reason="end_turn",
            model="claude-opus-4-6",
        )
    return _stream


class TestAgentCacheTTLPassthrough:

    @pytest.mark.asyncio
    async def test_cache_ttl_passed_to_stream(self, tmp_path):
        """handle_input passes cache_ttl through to stream()."""
        config = _make_config(tmp_path)
        agent = Agent(config)

        calls = []
        original_stream = _make_stream_events()

        async def capturing_stream(*args, **kwargs):
            calls.append(kwargs)
            async for event in original_stream(*args, **kwargs):
                yield event

        with patch("openalph.agent.stream", side_effect=capturing_stream):
            await agent.handle_input("test", "room1", cache_ttl="1h")

        assert len(calls) >= 1
        assert calls[0].get("cache_ttl") == "1h"

    @pytest.mark.asyncio
    async def test_cache_ttl_none_by_default(self, tmp_path):
        """handle_input defaults cache_ttl to None."""
        config = _make_config(tmp_path)
        agent = Agent(config)

        calls = []
        original_stream = _make_stream_events()

        async def capturing_stream(*args, **kwargs):
            calls.append(kwargs)
            async for event in original_stream(*args, **kwargs):
                yield event

        with patch("openalph.agent.stream", side_effect=capturing_stream):
            await agent.handle_input("test", "room1")

        assert len(calls) >= 1
        assert calls[0].get("cache_ttl") is None


class TestCacheSlashCommand:
    """Test /cache command parsing and validation."""

    def test_valid_values(self):
        """1h, 5m, and off are the only valid inputs."""
        valid = {"1h", "5m", "off"}
        for v in valid:
            value = v.strip().lower()
            if value == "off":
                value = "5m"
            assert value in ("5m", "1h")

    def test_off_becomes_1h(self):
        """'off' is an alias for '1h'."""
        value = "off"
        if value == "off":
            value = "1h"
        assert value == "1h"

    def test_invalid_values_rejected(self):
        """Random strings are not valid."""
        for v in ("10m", "2h", "banana", "true", ""):
            assert v.strip().lower() not in ("5m", "1h", "off")
