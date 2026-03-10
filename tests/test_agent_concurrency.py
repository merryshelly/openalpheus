"""Tests for per-room concurrency control in Agent.handle_input()."""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, patch

from openalph.agent import Agent
from openalph.config import AgentConfig
from openalph.provider import Response, Usage


def make_config(workspace, **kwargs):
    defaults = dict(
        name="test",
        model="test-model",
        max_tokens=8192,
        provider="anthropic",
        api_key="sk-test",
        base_url=None,
    )
    defaults.update(kwargs)
    defaults["workspace"] = workspace
    return AgentConfig(**defaults)


def make_response(content="Hello", input_tokens=10, output_tokens=5):
    return Response(
        content=content,
        model="test-model",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
    )


class TestConcurrentSameRoom:

    @pytest.mark.asyncio
    async def test_concurrent_handle_input_serializes(self, tmp_path):
        """Two concurrent calls on the same room are serialized — no interleaving."""
        config = make_config(tmp_path)
        agent = Agent(config)

        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            current = call_count
            # First call is slow to force overlap window
            if current == 1:
                await asyncio.sleep(0.1)
            return make_response(f"Response {current}")

        with patch("openalph.agent.complete", side_effect=mock_complete):
            r1, r2 = await asyncio.gather(
                agent.handle_input("Message 1", room_id="room1"),
                agent.handle_input("Message 2", room_id="room1"),
            )

        history = agent.history("room1")
        roles = [m["role"] for m in history]
        # Serialized: user, assistant, user, assistant (not user, user, ...)
        assert roles == ["user", "assistant", "user", "assistant"]
        # First message processed first (it acquired lock first)
        assert history[0]["content"] == "Message 1"
        assert history[1]["content"] == "Response 1"
        assert history[2]["content"] == "Message 2"
        assert history[3]["content"] == "Response 2"


class TestConcurrentDifferentRooms:

    @pytest.mark.asyncio
    async def test_concurrent_different_rooms_parallel(self, tmp_path):
        """Calls on different rooms run in parallel, not serialized."""
        config = make_config(tmp_path)
        agent = Agent(config)

        async def slow_complete(**kwargs):
            await asyncio.sleep(0.1)
            return make_response("Done")

        with patch("openalph.agent.complete", side_effect=slow_complete):
            start = time.monotonic()
            await asyncio.gather(
                agent.handle_input("A", room_id="room_a"),
                agent.handle_input("B", room_id="room_b"),
            )
            elapsed = time.monotonic() - start

        # If parallel, ~0.1s. If serialized, ~0.2s.
        assert elapsed < 0.18, f"Took {elapsed:.3f}s — rooms appear serialized"


class TestHeartbeatWaits:

    @pytest.mark.asyncio
    async def test_heartbeat_waits_for_active_processing(self, tmp_path):
        """A heartbeat arriving mid-processing waits for the lock."""
        config = make_config(tmp_path)
        agent = Agent(config)

        call_order = []

        async def mock_complete(**kwargs):
            msgs = kwargs.get("messages", [])
            last_user = [m for m in msgs if m["role"] == "user"][-1]["content"]
            if last_user == "user message":
                call_order.append("user_llm_start")
                await asyncio.sleep(0.1)
                call_order.append("user_llm_end")
            else:
                call_order.append("heartbeat_llm")
            return make_response(f"Re: {last_user}")

        with patch("openalph.agent.complete", side_effect=mock_complete):
            # Start user message, then fire heartbeat after a short delay
            async def delayed_heartbeat():
                await asyncio.sleep(0.02)  # fires while user msg is processing
                return await agent.handle_input("[heartbeat]", room_id="room1")

            user_result, hb_result = await asyncio.gather(
                agent.handle_input("user message", room_id="room1"),
                delayed_heartbeat(),
            )

        # User LLM must complete before heartbeat LLM starts
        assert call_order == ["user_llm_start", "user_llm_end", "heartbeat_llm"]

        # History is clean: user, assistant, user(heartbeat), assistant
        history = agent.history("room1")
        roles = [m["role"] for m in history]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert history[0]["content"] == "user message"
        assert history[2]["content"] == "[heartbeat]"
