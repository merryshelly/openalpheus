"""Integration tests for Matrix streaming wiring (Phase C of kdsn.65).

Tests the end-to-end flow: message in → agent streaming → Matrix delivery.

Contract:
    _process_message() wires on_text_delta → StreamingDelivery.push()
    Thinking blocks rendered as collapsible <details> messages
    Response already delivered via streaming — no duplicate send()
    Tool calls: finalize current stream, execute tools, new stream
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock, call
from pathlib import Path

from openalph.matrix import MatrixBot, StreamingDelivery
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.agent import Agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_matrix_config(**kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id="@test:matrix.local",
        device_id="TEST",
        password="test-pw",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def make_agent_config(workspace, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
        )},
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
    )
    defaults.update(kwargs)
    defaults["workspace"] = workspace
    return AgentConfig(**defaults)


def make_bot(tmp_path, **kwargs):
    """Create a MatrixBot with mocked Matrix client."""
    agent_config = make_agent_config(tmp_path, **kwargs)
    agent_config.matrix = make_matrix_config()
    agent = MagicMock(spec=Agent)
    agent.config = agent_config
    agent.system_prompt = "test prompt"
    agent.history = MagicMock(return_value=[])

    with patch("openalph.matrix.AsyncClient"):
        bot = MatrixBot(agent_config)
        bot.agent = agent
        bot.config = agent_config
        bot._synced = True
        bot._active_rooms = set()
        bot.session_log = None

        # Mock Matrix client methods
        bot.client = MagicMock()
        bot.client.room_typing = AsyncMock()
        send_resp = MagicMock()
        send_resp.event_id = "$evt_stream"
        bot.client.room_send = AsyncMock(return_value=send_resp)
        bot._room_send_with_retry = AsyncMock(return_value=send_resp)
        bot.send = AsyncMock()
        bot.send_notice = AsyncMock()

    return bot


def make_room(room_id="!test:matrix.local", members=2):
    room = MagicMock()
    room.room_id = room_id
    room.users = {f"@user{i}:matrix.local": MagicMock() for i in range(members)}
    room.name = "Test Room"
    room.display_name = "Test Room"
    return room


def make_event(sender="@user:matrix.local", body="Hello", event_id="$evt1"):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = 1000000
    event.source = {}
    return event


# ---------------------------------------------------------------------------
# Tests: Streaming wired in _process_message
# ---------------------------------------------------------------------------

class TestStreamingWiring:

    @pytest.mark.asyncio
    async def test_on_text_delta_passed_to_agent(self, tmp_path):
        """_process_message passes on_text_delta callback to handle_input."""
        bot = make_bot(tmp_path)
        bot.agent.handle_input = AsyncMock(return_value="Hello world!")

        room = make_room()
        event = make_event()

        await bot._process_message(room, event, "Hi")

        # handle_input should be called with on_text_delta kwarg
        kw = bot.agent.handle_input.call_args.kwargs
        assert "on_text_delta" in kw
        assert callable(kw["on_text_delta"])

    @pytest.mark.asyncio
    async def test_streaming_delivery_used(self, tmp_path):
        """A StreamingDelivery is created and wired via on_text_delta."""
        bot = make_bot(tmp_path)

        # Capture the on_text_delta callback
        captured_cb = None
        async def fake_handle_input(text, room_id, **kwargs):
            nonlocal captured_cb
            captured_cb = kwargs.get("on_text_delta")
            # Simulate streaming by calling the callback
            if captured_cb:
                await captured_cb("Hello ", done=False)
                await captured_cb("world!", done=False)
                await captured_cb("", done=True)
            return "Hello world!"

        bot.agent.handle_input = AsyncMock(side_effect=fake_handle_input)

        room = make_room()
        event = make_event()

        await bot._process_message(room, event, "Hi")

        assert captured_cb is not None

    @pytest.mark.asyncio
    async def test_no_duplicate_send_after_streaming(self, tmp_path):
        """Response is delivered via streaming edits — bot.send() NOT called for the response."""
        bot = make_bot(tmp_path)

        async def fake_handle_input(text, room_id, **kwargs):
            cb = kwargs.get("on_text_delta")
            if cb:
                await cb("x" * 50, done=False)  # Enough to trigger initial send
                await cb("", done=True)
            return "x" * 50

        bot.agent.handle_input = AsyncMock(side_effect=fake_handle_input)

        room = make_room()
        event = make_event()

        await bot._process_message(room, event, "Hi")

        # bot.send() should NOT be called — streaming delivery handles it
        # Check that send was not called with the response content
        for c in bot.send.call_args_list:
            if c[0][1] if len(c[0]) > 1 else c[1].get("text", ""):
                content = c[0][1] if len(c[0]) > 1 else c[1].get("text", "")
                assert content != "x" * 50, "Response should be delivered via streaming, not send()"


# ---------------------------------------------------------------------------
# Tests: Thinking block delivery
# ---------------------------------------------------------------------------

class TestThinkingBlockDelivery:

    @pytest.mark.asyncio
    async def test_thinking_sent_as_details_block(self, tmp_path):
        """Thinking blocks are sent as collapsible <details> HTML."""
        bot = make_bot(tmp_path)

        async def fake_handle_input(text, room_id, **kwargs):
            thinking_cb = kwargs.get("on_thinking_delta")
            text_cb = kwargs.get("on_text_delta")
            if thinking_cb:
                await thinking_cb("Deep reasoning here", done=False)
                await thinking_cb("", done=True)
            if text_cb:
                await text_cb("The answer is 42.", done=False)
                await text_cb("", done=True)
            return "The answer is 42."

        bot.agent.handle_input = AsyncMock(side_effect=fake_handle_input)

        room = make_room()
        event = make_event()

        await bot._process_message(room, event, "Think")

        # Should have sent a thinking message with <details> HTML
        all_sends = (bot._room_send_with_retry.call_args_list +
                     [call(room.room_id, c) for c in []])
        found_thinking = False
        for c in bot._room_send_with_retry.call_args_list:
            content = c[0][1] if len(c[0]) > 1 else {}
            formatted = content.get("formatted_body", "")
            new_formatted = content.get("m.new_content", {}).get("formatted_body", "")
            if "<details>" in formatted or "<details>" in new_formatted:
                found_thinking = True
                assert "💭" in (formatted or new_formatted)
                break

        # Also check bot.send calls
        for c in bot.send.call_args_list:
            if len(c[0]) > 1:
                text = c[0][1]
                if "<details>" in text or "💭" in text:
                    found_thinking = True
                    break

        assert found_thinking, "Thinking block should be sent as <details> HTML"

    @pytest.mark.asyncio
    async def test_no_thinking_message_when_no_thinking(self, tmp_path):
        """No thinking message sent when model doesn't produce thinking."""
        bot = make_bot(tmp_path)

        async def fake_handle_input(text, room_id, **kwargs):
            text_cb = kwargs.get("on_text_delta")
            if text_cb:
                await text_cb("Simple answer.", done=False)
                await text_cb("", done=True)
            return "Simple answer."

        bot.agent.handle_input = AsyncMock(side_effect=fake_handle_input)

        room = make_room()
        event = make_event()

        await bot._process_message(room, event, "Hi")

        # No <details> or thinking markers in any sent message
        for c in bot._room_send_with_retry.call_args_list:
            content = c[0][1] if len(c[0]) > 1 else {}
            formatted = content.get("formatted_body", "")
            assert "<details>" not in formatted


# ---------------------------------------------------------------------------
# Tests: Error handling during streaming
# ---------------------------------------------------------------------------

class TestStreamingErrors:

    @pytest.mark.asyncio
    async def test_provider_error_during_streaming(self, tmp_path):
        """ProviderError during streaming sends error to room."""
        from openalph.provider import ProviderError
        bot = make_bot(tmp_path)
        bot.agent.handle_input = AsyncMock(
            side_effect=ProviderError("model overloaded", status_code=529),
        )

        room = make_room()
        event = make_event()

        await bot._process_message(room, event, "Hi")

        # Error message sent to room
        bot.send.assert_called()
        error_text = bot.send.call_args[0][1]
        assert "error" in error_text.lower() or "⚠️" in error_text

    @pytest.mark.asyncio
    async def test_typing_cleared_after_streaming(self, tmp_path):
        """Typing indicator cleared even after streaming completes."""
        bot = make_bot(tmp_path)

        async def fake_handle_input(text, room_id, **kwargs):
            cb = kwargs.get("on_text_delta")
            if cb:
                await cb("Response", done=False)
                await cb("", done=True)
            return "Response"

        bot.agent.handle_input = AsyncMock(side_effect=fake_handle_input)

        room = make_room()
        event = make_event()

        await bot._process_message(room, event, "Hi")

        # Typing should be cleared (last typing call should be False)
        typing_calls = bot.client.room_typing.call_args_list
        if typing_calls:
            last_typing = typing_calls[-1]
            assert last_typing[0][1] == False or last_typing[1].get("typing_state") == False
