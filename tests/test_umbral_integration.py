"""Integration tests for umbral turns.

Tests the full flow through MatrixBot: slash commands, mutual exclusion
with heartbeat, injection → processing → archive → wipe → context reset,
and error paths (archive failure, context overflow).

Test fixtures follow the same patterns as test_heartbeat_integration.py
and test_heartbeat_overflow.py.
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.umbral import UmbralManager
from openalph.heartbeat import HeartbeatManager
from openalph.matrix import MatrixBot
from openalph.session import SessionLog
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.agent import ContextOverflowError
from openalph.provider import ProviderError


# --- Fixtures (same patterns as test_heartbeat_integration.py) ---


def make_matrix_config(**kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id="@saw:matrix.local",
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


def make_provider(key="default", type="anthropic", api_key="sk-test",
                  base_url=None, quirks=None):
    return ProviderConfig(
        key=key, type=type, api_key=api_key,
        base_url=base_url, quirks=quirks or [],
    )


def make_agent_config(workspace, **kwargs):
    defaults = dict(
        name="saw",
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


def make_room(room_id, member_count=2):
    room = MagicMock()
    room.room_id = room_id
    room.name = "Test Room"
    room.display_name = "Test Room"
    room.users = {
        f"@user{i}:matrix.local": MagicMock()
        for i in range(member_count)
    }
    return room


def make_event(sender, body, event_id="$evt1"):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = 1000000
    event.source = {"content": {"msgtype": "m.text", "body": body}}
    return event


def make_bot(tmp_path):
    """Create a MatrixBot with mocked internals, including umbral manager."""
    matrix_config = make_matrix_config()
    agent_config = make_agent_config(workspace=tmp_path)

    agent = MagicMock()
    agent.config = agent_config
    agent.handle_input = AsyncMock(return_value="Agent response")
    agent.status = MagicMock(return_value={
        "name": "saw", "model": "claude-sonnet-4-20250514",
        "context_tokens": 1000, "context_max": 200000, "context_pct": 0,
        "turns": 5, "uncached_input_tokens": 5000, "cache_read_tokens": 0, "cache_creation_tokens": 0, "total_output_tokens": 2000,
        "total_tool_calls": 3,
    })
    agent.history = MagicMock(return_value=[])
    agent.cancel = MagicMock()
    agent.reset_room = MagicMock()

    with patch("openalph.matrix.AsyncClient"):
        bot = MatrixBot(agent, matrix_config)

    bot.client.rooms = {}
    bot._synced = True
    bot.send = AsyncMock()
    bot.send_notice = AsyncMock()
    bot._set_typing = AsyncMock()

    return bot, agent


# --- Slash Command Tests ---


class TestUmbralCommands:
    """Test /umbral command parsing."""

    @pytest.mark.asyncio
    async def test_start_valid(self, tmp_path):
        """/umbral start 6h → confirmation."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        event = make_event("@sb:matrix.local", "/umbral start 6h")

        await bot._handle_room_message(room, event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "6h" in msg
        assert "Umbral" in msg or "umbral" in msg

    @pytest.mark.asyncio
    async def test_start_minutes(self, tmp_path):
        """/umbral start 30m → confirmation."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        event = make_event("@sb:matrix.local", "/umbral start 30m")

        await bot._handle_room_message(room, event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "30m" in msg

    @pytest.mark.asyncio
    async def test_start_below_minimum(self, tmp_path):
        """/umbral start 15m → minimum interval error."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        event = make_event("@sb:matrix.local", "/umbral start 15m")

        await bot._handle_room_message(room, event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "30m" in msg or "minimum" in msg.lower()

    @pytest.mark.asyncio
    async def test_start_invalid_interval(self, tmp_path):
        """/umbral start abc → error."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        event = make_event("@sb:matrix.local", "/umbral start abc")

        await bot._handle_room_message(room, event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "invalid" in msg.lower() or "Invalid" in msg

    @pytest.mark.asyncio
    async def test_stop_active(self, tmp_path):
        """/umbral stop → confirmation when active."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")

        # Start first
        start_event = make_event("@sb:matrix.local", "/umbral start 6h")
        await bot._handle_room_message(room, start_event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        bot.send.reset_mock()

        # Stop
        stop_event = make_event("@sb:matrix.local", "/umbral stop", event_id="$evt2")
        await bot._handle_room_message(room, stop_event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "stopped" in msg.lower()

    @pytest.mark.asyncio
    async def test_stop_none_active(self, tmp_path):
        """/umbral stop with none active → info message."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        event = make_event("@sb:matrix.local", "/umbral stop")

        await bot._handle_room_message(room, event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "no umbral" in msg.lower() or "No umbral" in msg

    @pytest.mark.asyncio
    async def test_status_with_active(self, tmp_path):
        """/umbral status → lists active timer."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")

        start_event = make_event("@sb:matrix.local", "/umbral start 6h")
        await bot._handle_room_message(room, start_event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        bot.send.reset_mock()

        status_event = make_event("@sb:matrix.local", "/umbral status", event_id="$evt2")
        await bot._handle_room_message(room, status_event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "6h" in msg
        assert "next in" in msg

    @pytest.mark.asyncio
    async def test_status_empty(self, tmp_path):
        """/umbral status with none → "No active umbral timers." """
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        event = make_event("@sb:matrix.local", "/umbral status")

        await bot._handle_room_message(room, event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "no active" in msg.lower() or "No active" in msg

    @pytest.mark.asyncio
    async def test_usage_on_bad_subcommand(self, tmp_path):
        """/umbral foo → usage message."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        event = make_event("@sb:matrix.local", "/umbral foo")

        await bot._handle_room_message(room, event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "/umbral" in msg.lower() or "usage" in msg.lower()

    @pytest.mark.asyncio
    async def test_command_not_passed_to_agent(self, tmp_path):
        """/umbral commands handled by bot, not agent."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        event = make_event("@sb:matrix.local", "/umbral start 6h")

        await bot._handle_room_message(room, event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        agent.handle_input.assert_not_awaited()


# --- Mutual Exclusion ---


class TestMutualExclusion:
    """Umbral and heartbeat cannot coexist in the same room."""

    @pytest.mark.asyncio
    async def test_umbral_blocked_when_heartbeat_active(self, tmp_path):
        """/umbral start rejected if heartbeat running."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")

        # Start heartbeat
        hb_event = make_event("@sb:matrix.local", "/heartbeat start 6h")
        await bot._handle_room_message(room, hb_event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        bot.send.reset_mock()

        # Try umbral
        um_event = make_event("@sb:matrix.local", "/umbral start 6h", event_id="$evt2")
        await bot._handle_room_message(room, um_event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "heartbeat" in msg.lower() and "stop" in msg.lower()

    @pytest.mark.asyncio
    async def test_heartbeat_blocked_when_umbral_active(self, tmp_path):
        """/heartbeat start rejected if umbral running."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")

        # Start umbral
        um_event = make_event("@sb:matrix.local", "/umbral start 6h")
        await bot._handle_room_message(room, um_event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        bot.send.reset_mock()

        # Try heartbeat
        hb_event = make_event("@sb:matrix.local", "/heartbeat start 6h", event_id="$evt2")
        await bot._handle_room_message(room, hb_event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        assert "umbral" in msg.lower() and "stop" in msg.lower()

    @pytest.mark.asyncio
    async def test_different_rooms_independent(self, tmp_path):
        """Heartbeat in room1 doesn't block umbral in room2."""
        bot, agent = make_bot(tmp_path)
        room1 = make_room("!room1:matrix.local")
        room2 = make_room("!room2:matrix.local")

        # Heartbeat in room1
        hb_event = make_event("@sb:matrix.local", "/heartbeat start 6h")
        await bot._handle_room_message(room1, hb_event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)
        bot.send.reset_mock()

        # Umbral in room2
        um_event = make_event("@sb:matrix.local", "/umbral start 6h")
        await bot._handle_room_message(room2, um_event)
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        bot.send.assert_awaited_once()
        msg = bot.send.call_args[0][1]
        # Should succeed, not be blocked
        assert "Umbral" in msg or "umbral" in msg
        assert "stop" not in msg.lower() or "started" in msg.lower()


# --- Session Archive & Wipe ---


class TestSessionArchiveWipe:
    """Test SessionLog.archive() and SessionLog.wipe() methods."""

    def test_archive_creates_timestamped_copy(self, tmp_path):
        """archive() copies JSONL to timestamped file and returns name."""
        sl = SessionLog(tmp_path, "@saw:matrix.local")

        # Create a session with some entries
        room_id = "!room1:matrix.local"
        sl.append(role="user", sender="@sb:matrix.local", room=room_id,
                  content="hello")
        sl.append(role="assistant", sender="@saw:matrix.local", room=room_id,
                  content="world")

        archive_name = sl.archive(room_id)

        # Archive exists and has same content as original
        archive_path = tmp_path / "sessions" / archive_name
        assert archive_path.exists()
        assert archive_name.startswith("room1_matrix.local-")
        assert archive_name.endswith(".jsonl")

        original = sl.read(room_id)
        assert len(original) == 2  # Original still intact

    def test_archive_nonexistent_raises(self, tmp_path):
        """archive() raises FileNotFoundError for nonexistent session."""
        sl = SessionLog(tmp_path, "@saw:matrix.local")

        with pytest.raises(FileNotFoundError):
            sl.archive("!nonexistent:matrix.local")

    def test_wipe_truncates_file(self, tmp_path):
        """wipe() truncates session file to empty."""
        sl = SessionLog(tmp_path, "@saw:matrix.local")

        room_id = "!room1:matrix.local"
        sl.append(role="user", sender="@sb:matrix.local", room=room_id,
                  content="hello")

        sl.wipe(room_id)

        entries = sl.read(room_id)
        assert entries == []

    def test_wipe_nonexistent_is_noop(self, tmp_path):
        """wipe() on nonexistent session doesn't raise."""
        sl = SessionLog(tmp_path, "@saw:matrix.local")
        sl.wipe("!nonexistent:matrix.local")  # should not raise

    def test_archive_then_wipe_then_seed(self, tmp_path):
        """Full cycle: archive, wipe, seed breadcrumb, read back."""
        sl = SessionLog(tmp_path, "@saw:matrix.local")

        room_id = "!room1:matrix.local"
        sl.append(role="user", sender="@sb:matrix.local", room=room_id,
                  content="turn 1")
        sl.append(role="assistant", sender="@saw:matrix.local", room=room_id,
                  content="response 1")

        # Archive
        archive_name = sl.archive(room_id)

        # Wipe
        sl.wipe(room_id)

        # Seed
        sl.append(
            role="system", sender="@saw:matrix.local", room=room_id,
            event="umbral_reset",
            detail=f"Context reset. Previous session archived to sessions/{archive_name}",
        )

        # Read back: should have only the seed entry
        entries = sl.read(room_id)
        assert len(entries) == 1
        assert entries[0]["role"] == "system"
        assert entries[0]["event"] == "umbral_reset"
        assert archive_name in entries[0]["detail"]


# --- Umbral Injection Flow ---


class TestUmbralInjection:
    """Test the _inject_umbral callback integration."""

    @pytest.mark.asyncio
    async def test_inject_calls_agent_and_archives(self, tmp_path):
        """Full umbral cycle: process → archive → wipe → reset."""
        bot, agent = make_bot(tmp_path)
        room_id = "!room1:matrix.local"
        bot._active_rooms.add(room_id)

        # Create session file so archive works
        sl = SessionLog(tmp_path, "@saw:matrix.local")
        sl.append(role="user", sender="@sb:matrix.local", room=room_id,
                  content="old context")
        bot.session_log = sl

        await bot._inject_umbral(room_id)

        # Agent was called
        agent.handle_input.assert_awaited()

        # Agent history was reset
        agent.reset_room.assert_called_once_with(room_id)

        # Room removed from active set (will reactivate on next message)
        assert room_id not in bot._active_rooms

        # Archive file exists in sessions dir
        sessions_dir = tmp_path / "sessions"
        archive_files = [f for f in sessions_dir.iterdir()
                         if "-" in f.stem and f.suffix == ".jsonl"]
        assert len(archive_files) == 1

        # Active session has only the seed entry
        entries = sl.read(room_id)
        assert len(entries) >= 1
        system_entries = [e for e in entries if e.get("event") == "umbral_reset"]
        assert len(system_entries) == 1

    @pytest.mark.asyncio
    async def test_inject_sends_matrix_notices(self, tmp_path):
        """Umbral sends beginning and concluded notices."""
        bot, agent = make_bot(tmp_path)
        room_id = "!room1:matrix.local"
        bot._active_rooms.add(room_id)

        sl = SessionLog(tmp_path, "@saw:matrix.local")
        sl.append(role="user", sender="@sb:matrix.local", room=room_id,
                  content="old")
        bot.session_log = sl

        await bot._inject_umbral(room_id)

        # Check notices
        notice_calls = bot.send_notice.call_args_list
        notice_bodies = [call[0][1] for call in notice_calls]
        assert any("beginning" in b.lower() for b in notice_bodies)
        assert any("concluded" in b.lower() for b in notice_bodies)


# --- Error Path Tests ---


class TestUmbralErrorPaths:
    """Error handling during umbral turns."""

    @pytest.mark.asyncio
    async def test_context_overflow_still_archives(self, tmp_path):
        """ContextOverflowError during turn → archive+wipe anyway (not stop)."""
        bot, agent = make_bot(tmp_path)
        room_id = "!room1:matrix.local"
        bot._active_rooms.add(room_id)

        agent.handle_input.side_effect = ContextOverflowError(150000, 128000)

        sl = SessionLog(tmp_path, "@saw:matrix.local")
        sl.append(role="user", sender="@sb:matrix.local", room=room_id,
                  content="bloated context")
        bot.session_log = sl

        await bot._inject_umbral(room_id)

        # Should still archive and wipe — NOT stop umbral
        agent.reset_room.assert_called_once_with(room_id)

        # Archive exists
        sessions_dir = tmp_path / "sessions"
        archive_files = [f for f in sessions_dir.iterdir()
                         if "-" in f.stem and f.suffix == ".jsonl"]
        assert len(archive_files) == 1

    @pytest.mark.asyncio
    async def test_provider_error_still_archives(self, tmp_path):
        """ProviderError during turn → archive+wipe, send error notice."""
        bot, agent = make_bot(tmp_path)
        room_id = "!room1:matrix.local"
        bot._active_rooms.add(room_id)

        agent.handle_input.side_effect = ProviderError("API down", status_code=503)

        sl = SessionLog(tmp_path, "@saw:matrix.local")
        sl.append(role="user", sender="@sb:matrix.local", room=room_id,
                  content="context")
        bot.session_log = sl

        await bot._inject_umbral(room_id)

        # Error message sent
        send_calls = bot.send.call_args_list
        error_msgs = [call[0][1] for call in send_calls if "Provider error" in call[0][1]]
        assert len(error_msgs) >= 1

        # Still archived and wiped
        agent.reset_room.assert_called_once_with(room_id)

    @pytest.mark.asyncio
    async def test_archive_failure_no_wipe_stops_umbral(self, tmp_path):
        """If archive raises OSError → do NOT wipe, stop umbral, alert."""
        bot, agent = make_bot(tmp_path)
        room_id = "!room1:matrix.local"
        bot._active_rooms.add(room_id)

        # Mock session_log with archive that raises
        sl = MagicMock()
        sl.archive = MagicMock(side_effect=OSError("disk full"))
        sl.wipe = MagicMock()
        sl.append = MagicMock()
        bot.session_log = sl

        # Mock umbral.stop
        bot.umbral.stop = AsyncMock()

        await bot._inject_umbral(room_id)

        # wipe MUST NOT be called
        sl.wipe.assert_not_called()

        # umbral stopped
        bot.umbral.stop.assert_awaited_once_with(room_id)

        # Alert sent
        send_calls = bot.send.call_args_list
        alert_msgs = [call[0][1] for call in send_calls
                      if "archive failed" in call[0][1].lower() or "🚨" in call[0][1]]
        assert len(alert_msgs) >= 1

    @pytest.mark.asyncio
    async def test_no_session_file_graceful(self, tmp_path):
        """If no session file exists, conclude gracefully without error."""
        bot, agent = make_bot(tmp_path)
        room_id = "!room1:matrix.local"
        bot._active_rooms.add(room_id)

        # Fresh session log with no prior entries
        sl = SessionLog(tmp_path, "@saw:matrix.local")
        bot.session_log = sl

        # This should not raise — FileNotFoundError on archive is handled
        await bot._inject_umbral(room_id)

        notice_calls = bot.send_notice.call_args_list
        notice_bodies = [call[0][1] for call in notice_calls]
        assert any("concluded" in b.lower() for b in notice_bodies)

    @pytest.mark.asyncio
    async def test_generic_turn_error_still_archives(self, tmp_path):
        """RuntimeError during turn → archive+wipe, keep umbral running."""
        bot, agent = make_bot(tmp_path)
        room_id = "!room1:matrix.local"
        bot._active_rooms.add(room_id)

        agent.handle_input.side_effect = RuntimeError("unexpected")

        sl = SessionLog(tmp_path, "@saw:matrix.local")
        sl.append(role="user", sender="@sb:matrix.local", room=room_id,
                  content="context")
        bot.session_log = sl

        await bot._inject_umbral(room_id)

        # Should still archive and reset — umbral continues
        agent.reset_room.assert_called_once_with(room_id)
