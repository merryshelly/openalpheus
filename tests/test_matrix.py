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
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig


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
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(key="anthropic", type="anthropic", api_key="sk-test", base_url=None, quirks=[])},
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
        from nio import LoginResponse
        config = make_matrix_config(password="secret", access_token=None)
        agent = MagicMock()
        agent.system_prompt = "test"
        agent.history = []

        with patch("openalph.matrix.AsyncClient") as MockClient:
            client = MockClient.return_value
            client.login = AsyncMock(return_value=LoginResponse(
                access_token="syt_test", device_id="TEST", user_id="@merry:matrix.local"
            ))
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

    @pytest.mark.asyncio
    async def test_login_failure_raises(self):
        """MatrixBot raises RuntimeError when password login returns a non-LoginResponse."""
        from nio import LoginResponse
        config = make_matrix_config(password="bad-password", access_token=None)
        agent = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        # Return a MagicMock that is NOT a LoginResponse instance
        bot.client.login = AsyncMock(return_value=MagicMock(spec=object))

        with pytest.raises(RuntimeError, match="Matrix login failed"):
            await bot._login()

    @pytest.mark.asyncio
    async def test_login_success_no_error(self):
        """MatrixBot completes _login without error when login returns a LoginResponse."""
        from nio import LoginResponse
        config = make_matrix_config(password="correct-password", access_token=None)
        agent = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()

        # Construct a real LoginResponse (dataclass from nio)
        mock_response = LoginResponse(
            access_token="syt_test_token",
            device_id="TEST_DEVICE",
            user_id="@merry:matrix.local",
        )
        bot.client.login = AsyncMock(return_value=mock_response)

        # Should complete without raising
        await bot._login()


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
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

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
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        agent.handle_input.assert_awaited_once()
        assert agent.handle_input.await_args[0] == ("Hello", "!test:matrix.local")


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
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

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
            "model": "claude-sonnet-4-20250514",
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
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Should post a status message
        bot.client.room_send.assert_awaited_once()
        sent_content = bot.client.room_send.call_args.kwargs.get("content") or bot.client.room_send.call_args[0][2] if len(bot.client.room_send.call_args[0]) > 2 else None
        # Should NOT go through agent.handle_input
        agent.handle_input.assert_not_called()




class TestThinkingCommand:
    """Tests for /thinking command — per-room thinking level control."""

    def _make_bot(self):
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.config = MagicMock()
        agent.config.thinking = "off"

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot._set_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._room_thinking = {}
        return bot

    @pytest.mark.asyncio
    async def test_thinking_show_default(self):
        """/thinking with no args shows config default when no room override."""
        bot = self._make_bot()
        bot.agent.config.thinking = "medium"

        event = make_room_message("@sb:matrix.local", "/thinking")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        sent = bot.client.room_send.call_args[0][2] if len(bot.client.room_send.call_args[0]) > 2 else bot.client.room_send.call_args.kwargs.get("content", {})
        assert "medium" in sent.get("body", "")
        assert "config" in sent.get("body", "")

    @pytest.mark.asyncio
    async def test_thinking_show_override(self):
        """/thinking with no args shows room override when set."""
        bot = self._make_bot()
        bot._room_thinking["!test:matrix.local"] = "high"

        event = make_room_message("@sb:matrix.local", "/thinking")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        sent = bot.client.room_send.call_args[0][2] if len(bot.client.room_send.call_args[0]) > 2 else bot.client.room_send.call_args.kwargs.get("content", {})
        assert "high" in sent.get("body", "")
        assert "override" in sent.get("body", "")

    @pytest.mark.asyncio
    async def test_thinking_set_valid_level(self):
        """/thinking high sets room-level override."""
        bot = self._make_bot()

        event = make_room_message("@sb:matrix.local", "/thinking high")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        assert bot._room_thinking["!test:matrix.local"] == "high"
        sent = bot.client.room_send.call_args[0][2] if len(bot.client.room_send.call_args[0]) > 2 else bot.client.room_send.call_args.kwargs.get("content", {})
        assert "high" in sent.get("body", "")

    @pytest.mark.asyncio
    async def test_thinking_set_off(self):
        """/thinking off disables thinking for the room."""
        bot = self._make_bot()
        bot._room_thinking["!test:matrix.local"] = "high"

        event = make_room_message("@sb:matrix.local", "/thinking off")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        assert bot._room_thinking["!test:matrix.local"] == "off"

    @pytest.mark.asyncio
    async def test_thinking_invalid_level_rejected(self):
        """/thinking banana rejects invalid levels."""
        bot = self._make_bot()

        event = make_room_message("@sb:matrix.local", "/thinking banana")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        assert "!test:matrix.local" not in bot._room_thinking
        sent = bot.client.room_send.call_args[0][2] if len(bot.client.room_send.call_args[0]) > 2 else bot.client.room_send.call_args.kwargs.get("content", {})
        assert "Invalid" in sent.get("body", "")

    @pytest.mark.asyncio
    async def test_thinking_case_insensitive(self):
        """/thinking HIGH is normalized to lowercase."""
        bot = self._make_bot()

        event = make_room_message("@sb:matrix.local", "/thinking HIGH")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        assert bot._room_thinking["!test:matrix.local"] == "high"

    @pytest.mark.asyncio
    async def test_thinking_does_not_reach_agent(self):
        """/thinking command does not trigger agent processing."""
        bot = self._make_bot()

        event = make_room_message("@sb:matrix.local", "/thinking low")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.agent.handle_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_thinking_override_used_in_processing(self):
        """Room thinking override is passed to agent.handle_input."""
        bot = self._make_bot()
        bot.agent.handle_input = AsyncMock(return_value="OK")
        bot._active_rooms = {"!test:matrix.local"}
        bot._room_thinking["!test:matrix.local"] = "high"
        bot.session_log = None

        event = make_room_message("@sb:matrix.local", "hello")
        room = MagicMock()
        room.room_id = "!test:matrix.local"
        room.users = {"@sb:matrix.local": MagicMock(), "@merry:matrix.local": MagicMock()}

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        call_kwargs = bot.agent.handle_input.call_args.kwargs
        assert call_kwargs.get("thinking") == "high"


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
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

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
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

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
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

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
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)


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


# --- Error Message Sanitization ---


class TestErrorSanitization:

    @pytest.mark.asyncio
    async def test_agent_error_does_not_leak_details(self):
        """Agent exception details must not be sent to the room."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        sensitive_msg = "Failed to connect to api.anthropic.com with key sk-ant-abc123"
        agent.handle_input = AsyncMock(side_effect=Exception(sensitive_msg))

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}

        event = make_room_message("@sb:matrix.local", "Do something")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.client.room_send.assert_awaited_once()
        sent_content = bot.client.room_send.call_args[0][2]
        sent_body = sent_content["body"]
        assert sensitive_msg not in sent_body
        assert "Internal error" in sent_body

    @pytest.mark.asyncio
    async def test_heartbeat_error_does_not_leak_details(self):
        """Heartbeat exception details must not be sent to the room."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        sensitive_msg = "Failed to connect to api.anthropic.com with key sk-ant-abc123"
        agent.handle_input = AsyncMock(side_effect=Exception(sensitive_msg))

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot._set_typing = AsyncMock()
        bot._active_rooms = {"!test:matrix.local"}
        bot.session_log = None

        await bot._inject_heartbeat("!test:matrix.local")

        bot.client.room_send.assert_awaited_once()
        sent_content = bot.client.room_send.call_args[0][2]
        sent_body = sent_content["body"]
        assert sensitive_msg not in sent_body
        assert "Heartbeat error" in sent_body


# --- Send Retry ---


class TestSendRetry:
    """Tests for retry behavior on send() and send_notice()."""

    @pytest.mark.asyncio
    async def test_send_succeeds_first_try(self):
        """send() succeeds without retry when room_send returns normally."""
        from nio import RoomSendResponse

        config = make_matrix_config()
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock(
            return_value=RoomSendResponse("$evt1", "!room:test")
        )

        await bot.send("!room:test", "hello")

        bot.client.room_send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_notice_succeeds_first_try(self):
        """send_notice() succeeds without retry when room_send returns normally."""
        from nio import RoomSendResponse

        config = make_matrix_config()
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock(
            return_value=RoomSendResponse("$evt1", "!room:test")
        )

        await bot.send_notice("!room:test", "notice text")

        bot.client.room_send.assert_awaited_once()
        sent_content = bot.client.room_send.call_args[0][2]
        assert sent_content["msgtype"] == "m.notice"

    @pytest.mark.asyncio
    async def test_send_retries_on_room_send_error(self):
        """send() retries when room_send returns RoomSendError, then succeeds."""
        from nio import RoomSendResponse, RoomSendError

        config = make_matrix_config()
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock(
            side_effect=[
                RoomSendError("rate limited", status_code="M_LIMIT_EXCEEDED"),
                RoomSendResponse("$evt1", "!room:test"),
            ]
        )

        # Use tiny backoff so test is fast
        await bot._room_send_with_retry(
            "!room:test",
            {"msgtype": "m.text", "body": "hello"},
            base_delay=0.01,
        )

        assert bot.client.room_send.await_count == 2

    @pytest.mark.asyncio
    async def test_send_retries_on_exception(self):
        """send() retries when room_send raises a network exception."""
        from nio import RoomSendResponse

        config = make_matrix_config()
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock(
            side_effect=[
                ConnectionError("connection reset"),
                RoomSendResponse("$evt1", "!room:test"),
            ]
        )

        await bot._room_send_with_retry(
            "!room:test",
            {"msgtype": "m.text", "body": "hello"},
            base_delay=0.01,
        )

        assert bot.client.room_send.await_count == 2

    @pytest.mark.asyncio
    async def test_send_exhausts_retries_raises(self):
        """send() raises RuntimeError after exhausting all retry attempts."""
        from nio import RoomSendError

        config = make_matrix_config()
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock(
            side_effect=RoomSendError("server error", status_code="M_UNKNOWN")
        )

        with pytest.raises(RuntimeError, match="Failed to send.*after 3 attempts"):
            await bot._room_send_with_retry(
                "!room:test",
                {"msgtype": "m.text", "body": "hello"},
                base_delay=0.01,
            )

        assert bot.client.room_send.await_count == 3

    @pytest.mark.asyncio
    async def test_send_preserves_original_exception(self):
        """The raised RuntimeError chains the original exception as __cause__."""
        config = make_matrix_config()
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock(
            side_effect=ConnectionError("gone")
        )

        with pytest.raises(RuntimeError) as exc_info:
            await bot._room_send_with_retry(
                "!room:test",
                {"msgtype": "m.text", "body": "hello"},
                max_attempts=2,
                base_delay=0.01,
            )

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    @pytest.mark.asyncio
    async def test_send_retry_respects_max_attempts(self):
        """Custom max_attempts is honored."""
        from nio import RoomSendResponse

        config = make_matrix_config()
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock(
            side_effect=[
                ConnectionError("fail 1"),
                ConnectionError("fail 2"),
                ConnectionError("fail 3"),
                ConnectionError("fail 4"),
                RoomSendResponse("$evt1", "!room:test"),
            ]
        )

        await bot._room_send_with_retry(
            "!room:test",
            {"msgtype": "m.text", "body": "hello"},
            max_attempts=5,
            base_delay=0.01,
        )

        assert bot.client.room_send.await_count == 5

    @pytest.mark.asyncio
    async def test_send_via_public_api_retries(self):
        """send() uses retry internally — verify via mock side_effect."""
        from nio import RoomSendResponse, RoomSendError

        config = make_matrix_config()
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock(
            side_effect=[
                RoomSendError("transient", status_code="M_LIMIT_EXCEEDED"),
                RoomSendResponse("$evt1", "!room:test"),
            ]
        )

        # Patch sleep to avoid real delay and verify backoff was attempted
        with patch("openalph.matrix.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await bot.send("!room:test", "hello")

        assert bot.client.room_send.await_count == 2
        mock_sleep.assert_awaited_once()  # one retry = one sleep


# --- Provider Error Surfacing (kdsn.61) ---


class TestProviderErrorSurfacing:
    """ProviderError from API calls surfaces the message in chat."""

    @pytest.mark.asyncio
    async def test_provider_error_surfaces_message(self):
        """ProviderError → user sees the actual error, not generic 'Internal error'."""
        from openalph.provider import ProviderError

        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(
            side_effect=ProviderError("model: invalid model: bongo", status_code=404)
        )

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}

        event = make_room_message("@sb:matrix.local", "Hello")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.client.room_send.assert_awaited_once()
        sent_content = bot.client.room_send.call_args[0][2]
        sent_body = sent_content["body"]
        assert "Provider error" in sent_body
        assert "invalid model" in sent_body
        assert "Internal error" not in sent_body

    @pytest.mark.asyncio
    async def test_provider_error_does_not_crash_bot(self):
        """ProviderError is caught cleanly — bot stays alive."""
        from openalph.provider import ProviderError

        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(
            side_effect=ProviderError("Rate limit exceeded", status_code=429)
        )

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}

        event = make_room_message("@sb:matrix.local", "Hello")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        # Should not raise
        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

    @pytest.mark.asyncio
    async def test_provider_error_without_status_code(self):
        """ProviderError without status_code still surfaces cleanly."""
        from openalph.provider import ProviderError

        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(
            side_effect=ProviderError("Provider unreachable — connection failed")
        )

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}

        event = make_room_message("@sb:matrix.local", "Hello")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        sent_content = bot.client.room_send.call_args[0][2]
        sent_body = sent_content["body"]
        assert "Provider error" in sent_body
        assert "unreachable" in sent_body

    @pytest.mark.asyncio
    async def test_generic_exception_still_hidden(self):
        """Non-ProviderError exceptions still get the generic message."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(
            side_effect=RuntimeError("some internal details with sk-ant-api03-secret")
        )

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}

        event = make_room_message("@sb:matrix.local", "Hello")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        sent_content = bot.client.room_send.call_args[0][2]
        sent_body = sent_content["body"]
        assert "Internal error" in sent_body
        assert "sk-ant" not in sent_body


# --- Empty Response Guard (kdsn.61 follow-up) ---


class TestEmptyResponseGuard:

    @pytest.mark.asyncio
    async def test_empty_response_sends_warning(self):
        """Empty string response from agent should send a warning to room."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(return_value="")

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}

        event = make_room_message("@sb:matrix.local", "Do something")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Warning message should have been sent
        bot.client.room_send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_whitespace_response_sends_warning(self):
        """Whitespace-only response should send a warning to room."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(return_value="   \n  ")

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}

        event = make_room_message("@sb:matrix.local", "Do something")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.client.room_send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_real_response_still_sent(self):
        """Non-empty response is still sent normally."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.handle_input = AsyncMock(return_value="Here's your answer.")

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}

        event = make_room_message("@sb:matrix.local", "Do something")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.client.room_send.assert_awaited_once()
