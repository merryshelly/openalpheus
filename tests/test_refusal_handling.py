"""Tests for stop_reason=refusal handling.

When the Anthropic API returns stop_reason="refusal", the agent must surface
that information to the caller so matrix.py can display a specific message
instead of the generic "Empty response" warning.

Covers:
    - last_stop_reason() correctly tracks stop_reason per room
    - Refusal produces empty text + stop_reason="refusal"
    - reset_room() clears stop_reason
    - Per-room isolation
"""

import pytest
from unittest.mock import patch

from openalph.agent import Agent
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import StreamEvent, Response, Usage


# ---------------------------------------------------------------------------
# Helpers (follow test_empty_response_retry.py patterns)
# ---------------------------------------------------------------------------

ROOM = "!test:matrix.local"


def make_provider(key="anthropic", type="anthropic", api_key="sk-test"):
    return ProviderConfig(key=key, type=type, api_key=api_key, base_url=None, quirks=[])


def make_config(workspace, **kwargs):
    defaults = {
        "name": "test-agent",
        "default_model": "anthropic/claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "providers": {"anthropic": make_provider()},
        "max_iterations": 25,
        "truncation_limit": 50000,
    }
    defaults.update(kwargs)
    defaults["workspace"] = workspace
    return AgentConfig(**defaults)


def make_response(content="", stop_reason="end_turn", output_tokens=50):
    return Response(
        content=content,
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=10, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


def _done(content="", stop_reason="end_turn"):
    return StreamEvent(
        type="done",
        response=make_response(content=content, stop_reason=stop_reason),
        stop_reason=stop_reason,
        model="claude-sonnet-4-20250514",
    )


def make_stream_fn(event_lists):
    """Patch for openalph.agent.stream: yields scripted events per invocation."""
    call_iter = iter(event_lists)

    async def _fake_stream(*args, **kwargs):
        events = next(call_iter)
        for event in events:
            yield event

    return _fake_stream


def text_events(text, stop_reason="end_turn"):
    events = []
    if text:
        events.append(StreamEvent(type="text", content=text))
    events.append(_done(content=text, stop_reason=stop_reason))
    return events


# ---------------------------------------------------------------------------
# stop_reason storage
# ---------------------------------------------------------------------------

class TestLastStopReason:

    @pytest.mark.asyncio
    async def test_stored_on_normal_response(self, tmp_path):
        """Normal text turn → last_stop_reason returns 'end_turn'."""
        agent = Agent(make_config(tmp_path))
        fn = make_stream_fn([text_events("Hello!", "end_turn")])
        with patch("openalph.agent.stream", new=fn):
            result = await agent.handle_input("hi", ROOM)

        assert result == "Hello!"
        assert agent.last_stop_reason(ROOM) == "end_turn"

    @pytest.mark.asyncio
    async def test_stored_on_refusal(self, tmp_path):
        """Refusal → empty text + last_stop_reason returns 'refusal'."""
        agent = Agent(make_config(tmp_path))
        fn = make_stream_fn([text_events("", "refusal")])
        with patch("openalph.agent.stream", new=fn):
            result = await agent.handle_input("bad request", ROOM)

        assert result == ""
        assert agent.last_stop_reason(ROOM) == "refusal"

    @pytest.mark.asyncio
    async def test_cleared_on_reset(self, tmp_path):
        """reset_room() clears last_stop_reason."""
        agent = Agent(make_config(tmp_path))
        fn = make_stream_fn([text_events("", "refusal")])
        with patch("openalph.agent.stream", new=fn):
            await agent.handle_input("x", ROOM)

        assert agent.last_stop_reason(ROOM) == "refusal"
        agent.reset_room(ROOM)
        assert agent.last_stop_reason(ROOM) is None

    def test_none_for_unseen_room(self, tmp_path):
        """Room with no prior turns → None."""
        agent = Agent(make_config(tmp_path))
        assert agent.last_stop_reason("!never:seen") is None

    @pytest.mark.asyncio
    async def test_per_room_isolation(self, tmp_path):
        """Different rooms track stop_reason independently."""
        agent = Agent(make_config(tmp_path))

        fn_refuse = make_stream_fn([text_events("", "refusal")])
        with patch("openalph.agent.stream", new=fn_refuse):
            await agent.handle_input("bad", "!roomA:s")

        fn_ok = make_stream_fn([text_events("Answer!", "end_turn")])
        with patch("openalph.agent.stream", new=fn_ok):
            await agent.handle_input("good", "!roomB:s")

        assert agent.last_stop_reason("!roomA:s") == "refusal"
        assert agent.last_stop_reason("!roomB:s") == "end_turn"

    @pytest.mark.asyncio
    async def test_updated_on_subsequent_turns(self, tmp_path):
        """A successful turn after a refusal overwrites the stop_reason."""
        agent = Agent(make_config(tmp_path))

        fn_refuse = make_stream_fn([text_events("", "refusal")])
        with patch("openalph.agent.stream", new=fn_refuse):
            await agent.handle_input("bad", ROOM)
        assert agent.last_stop_reason(ROOM) == "refusal"

        fn_ok = make_stream_fn([text_events("OK now", "end_turn")])
        with patch("openalph.agent.stream", new=fn_ok):
            await agent.handle_input("good", ROOM)
        assert agent.last_stop_reason(ROOM) == "end_turn"
