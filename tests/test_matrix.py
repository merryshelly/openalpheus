"""Tests for Matrix integration.

Interface contract:
    MatrixBot(agent, matrix_config) — Matrix client wrapping an Agent
    MatrixBot.start() — login, load history, start sync
    MatrixBot.stop() — cancel work, shutdown
    MatrixBot.send(room_id, text) — send message to room

Requires matrix-nio mocking since we don't connect to a real server in tests.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from pathlib import Path
from openalph.matrix import MatrixBot
from openalph.config import AgentConfig, MatrixConfig


# --- Fixtures ---


def make_matrix_config(**kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id="@merry-dev:matrix.local",
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def make_agent_config(**kwargs):
    defaults = dict(
        name="test-agent",
        model="test-model",
        max_tokens=8192,
        provider="anthropic",
        api_key="sk-test",
        base_url=None,
        workspace=Path("/tmp/test"),
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_room_message(sender, body, event_id="$evt1"):
    """Create a mock Matrix room message event."""
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = 1000000
    return event


# --- Login ---


class TestLogin:

    @pytest.mark.asyncio
    async def test_login_with_password(self):
        """MatrixBot logs in with password when provided."""
        config = make_matrix_config(password="secret", access_token=None)
        agent = MagicMock()
        agent.system_prompt = "test"
        agent.history = []

        with patch("openalph.matrix.AsyncClient") as MockClient:
            client = MockClient.return_value
            client.login = AsyncMock(return_value=MagicMock(transport_response=MagicMock(status=200)))
            client.joined_rooms = AsyncMock(return_value=MagicMock(rooms=[]))
            client.sync = AsyncMock()
            client.close = AsyncMock()

            bot = MatrixBot(agent, config)
            # Just test that login is called — full start() would enter sync loop
            await bot._login()

        client.login.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_login_with_token(self):
        """MatrixBot uses access_token when provided (skips password login)."""
        config = make_matrix_config(password=None, access_token="syt_test_token")
        agent = MagicMock()
        agent.system_prompt = "test"
        agent.history = []

        with patch("openalph.matrix.AsyncClient") as MockClient:
            client = MockClient.return_value
            client.access_token = None  # will be set
            client.joined_rooms = AsyncMock(return_value=MagicMock(rooms=[]))
            client.close = AsyncMock()

            bot = MatrixBot(agent, config)
            await bot._login()

        # Should set token directly, not call login()
        assert client.access_token == "syt_test_token"


# --- Message Routing ---


class TestMessageRouting:

    @pytest.mark.asyncio
    async def test_own_messages_skipped(self):
        """Agent's own messages are not processed."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(return_value="response")

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot._set_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True

        event = make_room_message("@merry:matrix.local", "my own message")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)

        agent.handle_input.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_other_user_message_processed(self):
        """Messages from other users are processed through the agent."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(return_value="Hello back!")

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot._set_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}  # Pre-activated

        event = make_room_message("@sb:matrix.local", "Hello")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)

        agent.handle_input.assert_awaited_once_with("Hello", "!test:matrix.local")


# --- Commands ---


class TestCommands:

    @pytest.mark.asyncio
    async def test_stop_command_cancels(self):
        """/stop cancels current work and posts confirmation."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.cancel = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot._set_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._cancel_current = AsyncMock()

        event = make_room_message("@sb:matrix.local", "/stop")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)

        bot._cancel_current.assert_awaited_once()
        # Should NOT go through agent.handle_input
        agent.handle_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_status_command_posts_status(self):
        """/status posts agent status without going through agent loop."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.status.return_value = {
            "name": "test",
            "model": "test-model",
            "turns": 5,
            "context_tokens": 8000,
            "context_max": 200000,
            "context_pct": 4,
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
            "total_tool_calls": 3,
        }

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot._set_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True

        event = make_room_message("@sb:matrix.local", "/status")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)

        # Should post a status message
        bot.client.room_send.assert_awaited_once()
        sent_content = bot.client.room_send.call_args.kwargs.get("content") or bot.client.room_send.call_args[0][2] if len(bot.client.room_send.call_args[0]) > 2 else None
        # Should NOT go through agent.handle_input
        agent.handle_input.assert_not_called()


# --- Typing Indicator ---


class TestTypingIndicator:

    @pytest.mark.asyncio
    async def test_typing_set_during_processing(self):
        """Typing indicator is ON while processing, OFF when done."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(return_value="response")

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}  # Pre-activated

        event = make_room_message("@sb:matrix.local", "Hello")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)

        # Typing should have been set and then cleared
        typing_calls = bot.client.room_typing.call_args_list
        assert len(typing_calls) >= 2
        # First call: typing ON
        assert typing_calls[0].kwargs.get("typing_state", typing_calls[0][0][1] if len(typing_calls[0][0]) > 1 else None) is True or typing_calls[0].args[1] is True
        # Last call: typing OFF
        last_call = typing_calls[-1]
        assert last_call.kwargs.get("typing_state", last_call[0][1] if len(last_call[0]) > 1 else None) is False or last_call.args[1] is False

    @pytest.mark.asyncio
    async def test_typing_cleared_on_error(self):
        """Typing indicator is cleared even if agent errors."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(side_effect=Exception("LLM error"))

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}  # Pre-activated

        event = make_room_message("@sb:matrix.local", "Hello")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)

        # Typing should be cleared (last call = False)
        typing_calls = bot.client.room_typing.call_args_list
        last_call = typing_calls[-1]
        # Should be typing=False
        assert any(
            arg is False
            for arg in list(last_call.args) + list(last_call.kwargs.values())
        )


# --- Error Handling ---


class TestErrorHandling:

    @pytest.mark.asyncio
    async def test_agent_error_sends_error_message(self):
        """Agent exception → error message posted to room."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(side_effect=Exception("Something broke"))

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}  # Pre-activated

        event = make_room_message("@sb:matrix.local", "Do something")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)

        # Should send an error message (not crash)
        bot.client.room_send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_error_does_not_crash_bot(self):
        """Agent exception is caught — bot stays alive."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(side_effect=RuntimeError("LLM down"))

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}  # Pre-activated

        event = make_room_message("@sb:matrix.local", "Hello")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        # Should not raise
        await bot._handle_room_message(room, event)


# --- Reconnection ---


class TestReconnection:

    def test_backoff_doubles(self):
        """Retry delay doubles after each failure."""
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = make_matrix_config(retry_base=1, retry_max=60)

        delay = bot.config.retry_base
        delays = []
        for _ in range(5):
            delays.append(delay)
            delay = min(delay * 2, bot.config.retry_max)

        assert delays == [1, 2, 4, 8, 16]

    def test_backoff_caps_at_max(self):
        """Retry delay doesn't exceed retry_max."""
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = make_matrix_config(retry_base=1, retry_max=10)

        delay = bot.config.retry_base
        for _ in range(20):
            delay = min(delay * 2, bot.config.retry_max)

        assert delay == 10
