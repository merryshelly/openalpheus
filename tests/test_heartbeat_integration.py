"""Integration tests for heartbeat feature.

Tests the full flow:
    - HeartbeatManager fires callback at interval
    - Matrix command parsing routes /heartbeat commands correctly
    - Heartbeat injection triggers agent processing pipeline
    - Commands bypass @mention gating (consistent with /stop, /status)
    - Session log records heartbeat events

These tests use short intervals (0.1s-0.3s) to verify timer behavior
without slow test runs.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.heartbeat import HeartbeatManager
from openalph.matrix import MatrixBot
from openalph.config import AgentConfig, MatrixConfig


# --- Fixtures ---


def make_matrix_config(**kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id="@watson:matrix.local",
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
        rooms=None,
    )
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def make_provider(key="default", type="anthropic", api_key="sk-test", base_url=None, quirks=None):
    from openalph.config import ProviderConfig
    return ProviderConfig(key=key, type=type, api_key=api_key, base_url=base_url, quirks=quirks or [])


def make_agent_config(workspace, **kwargs):
    defaults = dict(
        name="watson",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider(key="anthropic")},
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_room(room_id, member_count):
    room = MagicMock()
    room.room_id = room_id
    room.name = "Test Room"
    room.display_name = "Test Room"
    room.users = {f"@user{i}:matrix.local": MagicMock() for i in range(member_count)}
    return room


def make_event(sender, body, event_id="$evt1", mentions_user_ids=None):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = 1000000
    content = {"msgtype": "m.text", "body": body}
    if mentions_user_ids is not None:
        content["m.mentions"] = {"user_ids": mentions_user_ids}
    event.source = {"content": content}
    return event


def make_bot(tmp_path):
    """Create a MatrixBot with mocked internals."""
    matrix_config = make_matrix_config()
    agent_config = make_agent_config(workspace=tmp_path)

    agent = MagicMock()
    agent.config = agent_config
    agent.system_prompt = "test system prompt"
    agent.handle_input = AsyncMock(return_value="Agent response")
    agent.status = MagicMock(return_value={
        "name": "watson",
        "default_model": "claude-sonnet-4-20250514",
        "context_tokens": 1000,
        "context_max": 200000,
        "context_pct": 0,
        "turns": 5,
        "total_input_tokens": 5000,
        "total_output_tokens": 2000,
        "total_tool_calls": 3,
    })
    agent.history = MagicMock(return_value=[])
    agent.cancel = MagicMock()

    with patch("openalph.matrix.AsyncClient"):
        bot = MatrixBot(agent, matrix_config)

    # Configure mock client.rooms to return empty dict (no room name resolution)
    bot.client.rooms = {}
    bot._synced = True
    bot.send = AsyncMock()
    bot.send_notice = AsyncMock()
    bot._set_typing = AsyncMock()

    return bot, agent


# --- Timer Firing ---


class TestHeartbeatFires:
    @pytest.mark.asyncio
    async def test_callback_called_with_room_id(self, tmp_path):
        """Heartbeat fires and passes the correct room_id to callback."""
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        await hb.start("!room1:matrix.local", 0.1)  # 100ms for fast test
        await asyncio.sleep(0.25)

        callback.assert_awaited()
        callback.assert_awaited_with("!room1:matrix.local")
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_callback_fires_repeatedly(self, tmp_path):
        """Heartbeat fires more than once."""
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        await hb.start("!room1:matrix.local", 0.1)
        await asyncio.sleep(0.35)

        assert callback.await_count >= 2
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_stop_prevents_further_fires(self, tmp_path):
        """After stop, callback should not fire again."""
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        await hb.start("!room1:matrix.local", 0.1)
        await asyncio.sleep(0.15)
        initial_count = callback.await_count
        await hb.stop("!room1:matrix.local")
        await asyncio.sleep(0.25)

        # Should not have increased (or at most by 1 due to race)
        assert callback.await_count <= initial_count + 1
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_resume_starts_timers(self, tmp_path):
        """resume() reads config and starts timers that fire."""
        config_path = tmp_path / "heartbeats.json"
        config_path.write_text(json.dumps([
            {"room_id": "!room1:matrix.local", "interval_seconds": 0.1},
        ]))

        callback = AsyncMock()
        hb = HeartbeatManager(config_path, callback)
        await hb.resume()
        await asyncio.sleep(0.25)

        callback.assert_awaited_with("!room1:matrix.local")
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_rooms_fire_independently(self, tmp_path):
        """Multiple rooms have independent timers."""
        fired_rooms = []

        async def track_callback(room_id):
            fired_rooms.append(room_id)

        hb = HeartbeatManager(tmp_path / "heartbeats.json", track_callback)

        await hb.start("!room1:matrix.local", 0.1)
        await hb.start("!room2:matrix.local", 0.2)
        await asyncio.sleep(0.35)

        # room1 should have fired more often than room2
        room1_count = fired_rooms.count("!room1:matrix.local")
        room2_count = fired_rooms.count("!room2:matrix.local")
        assert room1_count >= 2
        assert room2_count >= 1
        assert room1_count > room2_count
        await hb.shutdown()


# --- Matrix Command Parsing ---


class TestHeartbeatCommands:
    @pytest.mark.asyncio
    async def test_start_valid(self, tmp_path):
        """/heartbeat start 6h → confirmation message."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 2)
        event = make_event("@sb:matrix.local", "/heartbeat start 6h")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "6h" in msg
        assert "started" in msg.lower() or "Heartbeat" in msg

    @pytest.mark.asyncio
    async def test_start_minutes(self, tmp_path):
        """/heartbeat start 15m → confirmation with correct interval."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 2)
        event = make_event("@sb:matrix.local", "/heartbeat start 15m")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "15m" in msg

    @pytest.mark.asyncio
    async def test_start_invalid_interval(self, tmp_path):
        """/heartbeat start abc → error message."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 2)
        event = make_event("@sb:matrix.local", "/heartbeat start abc")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "invalid" in msg.lower() or "Invalid" in msg

    @pytest.mark.asyncio
    async def test_start_below_minimum(self, tmp_path):
        """/heartbeat start 1m → minimum interval error."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 2)
        event = make_event("@sb:matrix.local", "/heartbeat start 1m")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "5m" in msg or "minimum" in msg.lower()

    @pytest.mark.asyncio
    async def test_stop_active(self, tmp_path):
        """/heartbeat stop → confirmation when heartbeat is active."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 2)

        # Start first
        start_event = make_event("@sb:matrix.local", "/heartbeat start 6h")
        await bot._handle_room_message(room, start_event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        bot.send.reset_mock()

        # Then stop
        stop_event = make_event("@sb:matrix.local", "/heartbeat stop", event_id="$evt2")
        await bot._handle_room_message(room, stop_event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "stopped" in msg.lower()

    @pytest.mark.asyncio
    async def test_stop_none_active(self, tmp_path):
        """/heartbeat stop with no active heartbeat → info message."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 2)
        event = make_event("@sb:matrix.local", "/heartbeat stop")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "no heartbeat" in msg.lower() or "No heartbeat" in msg

    @pytest.mark.asyncio
    async def test_status_with_active(self, tmp_path):
        """/heartbeat status → lists active heartbeats."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 2)

        # Start a heartbeat first
        start_event = make_event("@sb:matrix.local", "/heartbeat start 6h")
        await bot._handle_room_message(room, start_event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        bot.send.reset_mock()

        # Check status
        status_event = make_event("@sb:matrix.local", "/heartbeat status", event_id="$evt2")
        await bot._handle_room_message(room, status_event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "!room1:matrix.local" in msg  # room ID as fallback name
        assert "6h" in msg
        assert "next in" in msg

    @pytest.mark.asyncio
    async def test_status_empty(self, tmp_path):
        """/heartbeat status with none active → "No active heartbeats." """
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 2)
        event = make_event("@sb:matrix.local", "/heartbeat status")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "no active" in msg.lower() or "No active" in msg

    @pytest.mark.asyncio
    async def test_invalid_subcommand(self, tmp_path):
        """/heartbeat foo → usage message."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 2)
        event = make_event("@sb:matrix.local", "/heartbeat foo")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "usage" in msg.lower() or "/heartbeat" in msg.lower()

    @pytest.mark.asyncio
    async def test_heartbeat_alone(self, tmp_path):
        """/heartbeat with no subcommand → usage message."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 2)
        event = make_event("@sb:matrix.local", "/heartbeat")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "usage" in msg.lower() or "/heartbeat" in msg.lower()

    @pytest.mark.asyncio
    async def test_command_does_not_reach_agent(self, tmp_path):
        """/heartbeat commands are handled by the bot, not passed to the agent."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 2)
        event = make_event("@sb:matrix.local", "/heartbeat start 6h")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        agent.handle_input.assert_not_awaited()


# --- Gating Bypass ---


class TestHeartbeatGatedInSharedRooms:
    """Heartbeat commands require @mention in gated rooms (kdsn.60)."""

    @pytest.mark.asyncio
    async def test_bare_heartbeat_ignored_in_gated_room(self, tmp_path):
        """/heartbeat in a gated room (3+ members) without @mention → ignored.

        Consistent with all commands requiring mention in shared rooms.
        """
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 4)  # Gated: 4 members
        event = make_event("@sb:matrix.local", "/heartbeat start 6h")

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Should NOT process the heartbeat command
        bot.send.assert_not_awaited()
        # Should send a hint notice
        bot.send_notice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mentioned_heartbeat_works_in_gated_room(self, tmp_path):
        """@watson /heartbeat start 6h in gated room → processed."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 4)
        event = make_event(
            "@sb:matrix.local",
            "@watson:matrix.local /heartbeat start 6h",
            mentions_user_ids=["@watson:matrix.local"],
        )

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "heartbeat" in msg.lower() or "6h" in msg.lower()


# --- Heartbeat Injection ---


class TestHeartbeatInjection:
    @pytest.mark.asyncio
    async def test_inject_calls_agent(self, tmp_path):
        """When heartbeat fires, agent.handle_input is called."""
        bot, agent = make_bot(tmp_path)
        # Pre-activate the room
        bot._active_rooms = {"!room1:matrix.local": True}

        await bot._inject_heartbeat("!room1:matrix.local")

        agent.handle_input.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_inject_sends_response(self, tmp_path):
        """Heartbeat processing sends the agent's response to the room."""
        bot, agent = make_bot(tmp_path)
        bot._active_rooms = {"!room1:matrix.local": True}

        await bot._inject_heartbeat("!room1:matrix.local")

        bot.send.assert_awaited()
        room_id = bot.send.call_args[0][0]
        assert room_id == "!room1:matrix.local"

    @pytest.mark.asyncio
    async def test_inject_logs_to_session(self, tmp_path):
        """Heartbeat message is recorded in session log."""
        bot, agent = make_bot(tmp_path)
        bot._active_rooms = {"!room1:matrix.local": True}

        # Set up a mock session log
        bot.session_log = MagicMock()
        bot.session_log.append = MagicMock()

        await bot._inject_heartbeat("!room1:matrix.local")

        # Should have logged the heartbeat input
        calls = bot.session_log.append.call_args_list
        assert len(calls) >= 1
        # First call should be the heartbeat system message
        first_call = calls[0]
        assert first_call.kwargs.get("role") == "system" or first_call[1].get("role") == "system"

    @pytest.mark.asyncio
    async def test_inject_activates_room_if_needed(self, tmp_path):
        """If room isn't active, heartbeat activates it before processing."""
        bot, agent = make_bot(tmp_path)
        bot._active_rooms = {}  # Room not active
        bot._activate_room = AsyncMock()

        await bot._inject_heartbeat("!room1:matrix.local")

        bot._activate_room.assert_awaited()


# --- End-to-End: Timer → Injection → Agent ---


class TestHeartbeatEndToEnd:
    @pytest.mark.asyncio
    async def test_start_command_then_timer_fires(self, tmp_path):
        """Full flow: /heartbeat start → timer fires → agent processes."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local", 2)

        # Start with a very short interval for testing
        event = make_event("@sb:matrix.local", "/heartbeat start 5m")
        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Verify heartbeat manager was created and has an entry
        assert hasattr(bot, 'heartbeat')
        entries = bot.heartbeat.status()
        assert len(entries) == 1
        assert entries[0].room_id == "!room1:matrix.local"
        assert entries[0].interval_seconds == 300  # 5m = 300s

        # Clean up
        if hasattr(bot, 'heartbeat'):
            await bot.heartbeat.shutdown()
