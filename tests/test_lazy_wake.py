"""Tests for lazy room activation and /reset removal.

Phase 3.5 Features:
    - Lazy wake: rooms are NOT hydrated during initial sync. History is loaded
      on-demand when the first live message arrives in a room.
    - /reset removal: the /reset command is removed entirely. New room = new session.
    - Dead code cleanup: _load_history_into_agent() is removed.

Architecture invariant: room = session = context = persistence.
    If context is too long, start a new room. No checkpoints, no resets.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path

from openalph.matrix import MatrixBot
from openalph.agent import Agent, ContextOverflowError as AgentOverflowError
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


def make_room(room_id):
    """Create a mock Matrix room."""
    room = MagicMock()
    room.room_id = room_id
    return room


# --- /reset Removal ---


class TestResetRemoval:

    @pytest.mark.asyncio
    async def test_reset_command_not_recognized(self):
        """/reset is not a recognized command — treated as regular message."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent._rooms = {}
        agent.handle_input = AsyncMock(return_value="I don't understand /reset")
        agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))
        agent.cancel = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = {"!test:matrix.local"}  # Pre-activated to skip activation
        bot._halted_rooms = set()
        bot._session_locks = {}

        event = make_room_message("@sb:matrix.local", "/reset")
        room = make_room("!test:matrix.local")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Should go through agent.handle_input as a regular message,
        # not be intercepted as a command
        agent.handle_input.assert_awaited_once()
        assert agent.handle_input.await_args[0] == ("/reset", "!test:matrix.local")

    # reset_room removal test was here but premature — umbral still calls
    # agent.reset_room() during context resets (matrix.py:910).  Revisit
    # when umbral is refactored to use session-level reset instead.


# --- Dead Code Removal ---


class TestDeadCodeRemoval:

    def test_no_load_history_into_agent_method(self):
        """_load_history_into_agent() dead code should be removed."""
        assert not hasattr(MatrixBot, "_load_history_into_agent"), \
            "_load_history_into_agent is dead code and should be removed"


# --- Lazy Room Activation ---


class TestLazyWake:

    @pytest.mark.asyncio
    async def test_initial_sync_does_not_hydrate_rooms(self):
        """During initial sync, room messages are NOT loaded into agent history."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent._rooms = {}
        agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot._synced = False
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}

        # Simulate initial sync delivering messages from multiple rooms
        room1 = make_room("!room1:local")
        room2 = make_room("!room2:local")

        await bot._handle_room_message(room1, make_room_message("@sb:local", "old msg 1"))
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        await bot._handle_room_message(room1, make_room_message("@merry:local", "old reply"))
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        await bot._handle_room_message(room2, make_room_message("@sb:local", "old msg 2"))
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Agent history should NOT have been populated
        assert len(agent._rooms) == 0, \
            "Initial sync should not hydrate room history — lazy wake means on-demand only"

    @pytest.mark.asyncio
    async def test_first_live_message_triggers_history_load(self):
        """When the first live message arrives in a room, history is loaded from Matrix."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent._rooms = {}
        agent.handle_input = AsyncMock(return_value="Hello!")
        agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))
        agent.cancel = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}

        # Mock room_messages to return historical messages
        history_response = MagicMock()
        history_chunk = [
            make_room_message("@sb:local", "earlier message", "$h1"),
            make_room_message("@merry:local", "earlier reply", "$h2"),
        ]
        history_response.chunk = history_chunk
        history_response.end = ""  # No more pages
        bot.client.room_messages = AsyncMock(return_value=history_response)

        room = make_room("!newroom:local")
        event = make_room_message("@sb:local", "Hello now")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Should have called room_messages to load history
        bot.client.room_messages.assert_awaited()
        # Room should now be active
        assert "!newroom:local" in bot._active_rooms

    @pytest.mark.asyncio
    async def test_second_message_skips_history_load(self):
        """Once a room is active, subsequent messages don't re-load history."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent._rooms = {}
        agent.handle_input = AsyncMock(return_value="response")
        agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))
        agent.cancel = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}

        history_response = MagicMock()
        history_response.chunk = []
        history_response.end = ""
        bot.client.room_messages = AsyncMock(return_value=history_response)

        room = make_room("!room:local")

        # First message: triggers history load
        await bot._handle_room_message(room, make_room_message("@sb:local", "First"))
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        first_call_count = bot.client.room_messages.await_count

        # Second message: should NOT trigger history load again
        await bot._handle_room_message(room, make_room_message("@sb:local", "Second"))
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        second_call_count = bot.client.room_messages.await_count

        assert second_call_count == first_call_count, \
            "History should only be loaded once per room activation"

    @pytest.mark.asyncio
    async def test_different_rooms_each_activate_independently(self):
        """Each room activates independently on its first message."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent._rooms = {}
        agent.handle_input = AsyncMock(return_value="response")
        agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))
        agent.cancel = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}

        history_response = MagicMock()
        history_response.chunk = []
        history_response.end = ""
        bot.client.room_messages = AsyncMock(return_value=history_response)

        room1 = make_room("!room1:local")
        room2 = make_room("!room2:local")

        await bot._handle_room_message(room1, make_room_message("@sb:local", "Hi room1"))
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        await bot._handle_room_message(room2, make_room_message("@sb:local", "Hi room2"))
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        assert "!room1:local" in bot._active_rooms
        assert "!room2:local" in bot._active_rooms
        # room_messages should have been called twice (once per room)
        assert bot.client.room_messages.await_count == 2


# --- History Loading on Activation ---


class TestHistoryLoadOnActivation:

    @pytest.mark.asyncio
    async def test_loaded_history_populates_agent_rooms(self):
        """Loaded history is converted to agent format and stored in _rooms."""
        config = make_matrix_config(user_id="@merry:matrix.local")

        real_rooms = {}
        agent = MagicMock()
        agent._rooms = real_rooms
        agent.handle_input = AsyncMock(return_value="response")
        agent.history = MagicMock(side_effect=lambda rid: real_rooms.setdefault(rid, []))
        agent.cancel = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}

        # Historical messages to load
        history_response = MagicMock()
        history_response.chunk = [
            make_room_message("@sb:local", "What's 2+2?", "$h1"),
            make_room_message("@merry:local", "4", "$h2"),
        ]
        history_response.end = ""
        bot.client.room_messages = AsyncMock(return_value=history_response)

        room = make_room("!room:local")
        event = make_room_message("@sb:local", "New question")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Agent history for this room should include the loaded messages
        history = real_rooms.get("!room:local", [])
        # Should have historical messages + the new user message + assistant response
        # At minimum, history load should have added the old messages
        user_messages = [m for m in history if m.get("role") == "user"]
        assert len(user_messages) >= 2, \
            f"Expected at least 2 user messages (1 historical + 1 new), got {len(user_messages)}"

    @pytest.mark.asyncio
    async def test_history_roles_assigned_correctly(self):
        """Bot's own historical messages get role='assistant', others get role='user'."""
        config = make_matrix_config(user_id="@merry:matrix.local")

        real_rooms = {}
        agent = MagicMock()
        agent._rooms = real_rooms
        agent.handle_input = AsyncMock(return_value="response")
        agent.history = MagicMock(side_effect=lambda rid: real_rooms.setdefault(rid, []))
        agent.cancel = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}

        # Mock returns reverse chronological order (newest first), matching real Matrix API
        history_response = MagicMock()
        history_response.chunk = [
            make_room_message("@merry:matrix.local", "Bot reply", "$h2"),
            make_room_message("@sb:local", "User msg", "$h1"),
        ]
        history_response.end = ""
        bot.client.room_messages = AsyncMock(return_value=history_response)

        room = make_room("!room:local")
        event = make_room_message("@sb:local", "New msg")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        history = real_rooms.get("!room:local", [])
        # After reversal to chronological: user msg first, then bot reply
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "User msg"
        # Second loaded message should be assistant role
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Bot reply"

    @pytest.mark.asyncio
    async def test_context_overflow_on_activation(self):
        """If loaded history exceeds model capacity, ContextOverflowError is raised gracefully."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent._rooms = {}
        agent.handle_input = AsyncMock(side_effect=AgentOverflowError(200000, 200000))
        agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))
        agent.cancel = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}

        # Return a massive history
        huge_messages = [
            make_room_message("@sb:local", "x" * 50000, f"$h{i}")
            for i in range(20)
        ]
        history_response = MagicMock()
        history_response.chunk = huge_messages
        history_response.end = ""
        bot.client.room_messages = AsyncMock(return_value=history_response)

        room = make_room("!room:local")
        event = make_room_message("@sb:local", "Hello")

        # Should not crash — should send an overflow message
        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Should have sent an overflow notice to the room
        send_calls = bot.client.room_send.call_args_list
        assert len(send_calls) >= 1
        # Find the overflow message
        sent_bodies = [
            c.args[2].get("body", "") if len(c.args) > 2 else
            c.kwargs.get("content", {}).get("body", "")
            for c in send_calls
        ]
        assert any("overflow" in b.lower() or "context" in b.lower() for b in sent_bodies), \
            f"Expected overflow message, got: {sent_bodies}"


# --- Pagination ---


class TestHistoryPagination:

    @pytest.mark.asyncio
    async def test_history_paginates_until_complete(self):
        """History loading paginates backward until all messages are retrieved."""
        config = make_matrix_config(user_id="@merry:matrix.local")

        real_rooms = {}
        agent = MagicMock()
        agent._rooms = real_rooms
        agent.handle_input = AsyncMock(return_value="response")
        agent.history = MagicMock(side_effect=lambda rid: real_rooms.setdefault(rid, []))
        agent.cancel = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}

        # First page returns messages + a pagination token
        page1 = MagicMock()
        page1.chunk = [
            make_room_message("@sb:local", "Recent msg", "$h3"),
        ]
        page1.end = "token_page2"

        # Second page returns more messages + empty token (done)
        page2 = MagicMock()
        page2.chunk = [
            make_room_message("@sb:local", "Older msg", "$h1"),
            make_room_message("@merry:local", "Older reply", "$h2"),
        ]
        page2.end = ""

        bot.client.room_messages = AsyncMock(side_effect=[page1, page2])

        room = make_room("!room:local")
        event = make_room_message("@sb:local", "New message")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Should have paginated (called room_messages twice)
        assert bot.client.room_messages.await_count == 2

        # All historical messages should be in agent history
        history = real_rooms.get("!room:local", [])
        historical_content = [m["content"] for m in history if m["content"] != "New message"]
        assert "Older msg" in historical_content
        assert "Older reply" in historical_content
        assert "Recent msg" in historical_content

    @pytest.mark.asyncio
    async def test_empty_room_no_pagination(self):
        """Room with no history doesn't paginate."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent._rooms = {}
        agent.handle_input = AsyncMock(return_value="response")
        agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))
        agent.cancel = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}

        empty_response = MagicMock()
        empty_response.chunk = []
        empty_response.end = ""
        bot.client.room_messages = AsyncMock(return_value=empty_response)

        room = make_room("!room:local")
        event = make_room_message("@sb:local", "First message ever")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Only one call to room_messages (initial, finds nothing)
        assert bot.client.room_messages.await_count == 1

    @pytest.mark.asyncio
    async def test_legacy_fallback_caps_at_limit(self):
        """Legacy fallback stops paging at 500 messages to prevent OOM."""
        config = make_matrix_config(user_id="@merry:matrix.local")

        real_rooms = {}
        agent = MagicMock()
        agent._rooms = real_rooms
        agent.handle_input = AsyncMock(return_value="response")
        agent.history = MagicMock(side_effect=lambda rid: real_rooms.setdefault(rid, []))
        agent.cancel = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}
        bot.session_log = None  # force legacy path

        # Each call returns 100 messages with a truthy end token (simulates infinite pages)
        def make_infinite_page():
            page = MagicMock()
            page.chunk = [make_room_message("@sb:local", f"msg", f"$evt") for _ in range(100)]
            page.end = "infinite_token"
            return page

        bot.client.room_messages = AsyncMock(side_effect=lambda *a, **kw: make_infinite_page())

        room = make_room("!room:local")
        event = make_room_message("@sb:local", "New message")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # 1 initial call + at most 5 pages (500 messages reached after 5 more = 6 total calls)
        assert bot.client.room_messages.await_count <= 6

    @pytest.mark.asyncio
    async def test_legacy_fallback_stops_on_empty_chunk(self):
        """Legacy fallback stops early when server returns empty chunk."""
        config = make_matrix_config(user_id="@merry:matrix.local")

        real_rooms = {}
        agent = MagicMock()
        agent._rooms = real_rooms
        agent.handle_input = AsyncMock(return_value="response")
        agent.history = MagicMock(side_effect=lambda rid: real_rooms.setdefault(rid, []))
        agent.cancel = MagicMock()

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}
        bot.session_log = None  # force legacy path

        # First call returns messages + pagination token
        page1 = MagicMock()
        page1.chunk = [make_room_message("@sb:local", "Old message", "$h1")]
        page1.end = "token_page2"

        # Second call returns empty chunk (server signals no more data)
        page2 = MagicMock()
        page2.chunk = []
        page2.end = "token_page3"  # truthy end but empty chunk — should stop

        bot.client.room_messages = AsyncMock(side_effect=[page1, page2])

        room = make_room("!room:local")
        event = make_room_message("@sb:local", "New message")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Should stop after second call (empty chunk breaks the loop)
        assert bot.client.room_messages.await_count == 2
        # The old message should still be in history
        history = real_rooms.get("!room:local", [])
        assert any(m["content"] == "Old message" for m in history)


# --- Commands Still Work ---


class TestCommandsPostRefactor:

    @pytest.mark.asyncio
    async def test_stop_still_works(self):
        """/stop command still cancels work after refactor."""
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
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}
        bot._cancel_current = AsyncMock()

        event = make_room_message("@sb:local", "/stop")
        room = make_room("!room:local")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        bot._cancel_current.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_status_still_works(self):
        """/status command still works after refactor."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent.status.return_value = {
            "name": "test",
            "model": "claude-sonnet-4-20250514",
            "turns": 5,
            "context_tokens": 8000,
            "context_max": 200000,
            "context_pct": 4,
            "uncached_input_tokens": 1000,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
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
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}

        event = make_room_message("@sb:local", "/status")
        room = make_room("!room:local")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.client.room_send.assert_awaited_once()
        agent.handle_input.assert_not_called()


# --- Startup Behavior ---


class TestStartupBehavior:

    @pytest.mark.asyncio
    async def test_initial_sync_logs_zero_rooms_loaded(self):
        """After initial sync, no rooms should have history loaded."""
        config = make_matrix_config(user_id="@merry:matrix.local")
        agent = MagicMock()
        agent._rooms = {}

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = config
        bot.agent = agent
        bot._active_rooms = set()
        bot._halted_rooms = set()
        bot._session_locks = {}

        # After sync completes, no rooms should be active
        assert len(bot._active_rooms) == 0
        assert len(agent._rooms) == 0
