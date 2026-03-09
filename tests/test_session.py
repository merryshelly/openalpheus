"""Comprehensive tests for SessionLog (JSONL session persistence).

Tests:
    - SessionLog unit tests (append, read, last_event_id, build_context)
    - Overflow handling for large tool outputs
    - build_context format compatibility with agent.py
    - MatrixBot integration (with mocks)
"""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.session import SessionLog, OVERFLOW_THRESHOLD
from openalph.matrix import MatrixBot
from openalph.config import AgentConfig, MatrixConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOM_ID = "!abc123:matrix.local"
AGENT_USER = "@watson:matrix.local"
USER = "@sb:matrix.local"


def make_session_log(tmp_path: Path) -> SessionLog:
    return SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)


def make_matrix_config(**kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id=AGENT_USER,
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


def make_agent_config(workspace: Path, **kwargs):
    defaults = dict(
        name="test-agent",
        model="test-model",
        max_tokens=8192,
        provider="anthropic",
        api_key="sk-test",
        base_url=None,
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_room_message(sender, body, event_id="$evt1"):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = 1000000
    return event


# ---------------------------------------------------------------------------
# SessionLog unit tests
# ---------------------------------------------------------------------------


class TestSessionLogAppend:

    def test_append_creates_session_dir(self, tmp_path):
        sl = make_session_log(tmp_path)
        sessions_dir = tmp_path / "sessions"
        assert not sessions_dir.exists()

        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$e1", content="Hello")

        assert sessions_dir.exists()
        assert sessions_dir.is_dir()

    def test_append_creates_file(self, tmp_path):
        sl = make_session_log(tmp_path)
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$e1", content="Hello")

        expected = tmp_path / "sessions" / "abc123_matrix.local.jsonl"
        assert expected.exists()

    def test_room_id_sanitization(self, tmp_path):
        sl = make_session_log(tmp_path)
        sl.append(role="user", sender=USER, room="!abc:matrix.local", event_id="$e1", content="Hi")

        path = sl._session_path("!abc:matrix.local")
        assert path.name == "abc_matrix.local.jsonl"
        assert path.exists()

    def test_append_user_turn(self, tmp_path):
        sl = make_session_log(tmp_path)
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$e1", content="Check status")

        entries = sl.read(ROOM_ID)
        assert len(entries) == 1
        e = entries[0]
        assert e["role"] == "user"
        assert e["sender"] == USER
        assert e["room"] == ROOM_ID
        assert e["event_id"] == "$e1"
        assert e["content"] == "Check status"
        assert "ts" in e

    def test_append_assistant_turn(self, tmp_path):
        sl = make_session_log(tmp_path)
        tool_calls = [{"call_id": "c1", "name": "shell", "input": {"command": "uptime"}}]
        sl.append(
            role="assistant",
            sender=AGENT_USER,
            room=ROOM_ID,
            event_id="$e2",
            content="Checking...",
            tool_calls=tool_calls,
        )

        entries = sl.read(ROOM_ID)
        assert len(entries) == 1
        e = entries[0]
        assert e["role"] == "assistant"
        assert e["content"] == "Checking..."
        assert e["tool_calls"] == tool_calls

    def test_append_tool_turn(self, tmp_path):
        sl = make_session_log(tmp_path)
        sl.append(
            role="tool",
            sender=AGENT_USER,
            room=ROOM_ID,
            event_id=None,
            call_id="c1",
            name="shell",
            output="up 5 days",
        )

        entries = sl.read(ROOM_ID)
        assert len(entries) == 1
        e = entries[0]
        assert e["role"] == "tool"
        assert e["event_id"] is None
        assert e["call_id"] == "c1"
        assert e["name"] == "shell"
        assert e["output"] == "up 5 days"
        assert e["truncated"] is False

    def test_append_system_turn(self, tmp_path):
        sl = make_session_log(tmp_path)
        sl.append(
            role="system",
            sender=AGENT_USER,
            room=ROOM_ID,
            event_id=None,
            event="session_start",
            detail="Gap-filled 3 messages",
        )

        entries = sl.read(ROOM_ID)
        assert len(entries) == 1
        e = entries[0]
        assert e["role"] == "system"
        assert e["event"] == "session_start"
        assert e["detail"] == "Gap-filled 3 messages"


class TestSessionLogRead:

    def test_read_empty(self, tmp_path):
        sl = make_session_log(tmp_path)
        result = sl.read("!nonexistent:matrix.local")
        assert result == []

    def test_read_roundtrip(self, tmp_path):
        sl = make_session_log(tmp_path)

        entries_to_write = [
            dict(role="user", sender=USER, room=ROOM_ID, event_id="$e1", content="Hello"),
            dict(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id="$e2", content="Hi there"),
            dict(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                 call_id="c1", name="shell", output="ok"),
            dict(role="system", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                 event="session_start", detail="test"),
        ]

        for entry in entries_to_write:
            sl.append(**entry)

        result = sl.read(ROOM_ID)
        assert len(result) == 4

        # Verify order and content
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hi there"
        assert result[2]["role"] == "tool"
        assert result[2]["call_id"] == "c1"
        assert result[3]["role"] == "system"
        assert result[3]["event"] == "session_start"


class TestLastEventId:

    def test_last_event_id(self, tmp_path):
        sl = make_session_log(tmp_path)
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$e1", content="Hello")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id="$e2", content="Hi")
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  call_id="c1", name="shell", output="ok")
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$e3", content="Thanks")

        result = sl.last_event_id(ROOM_ID)
        assert result == "$e3"

    def test_last_event_id_empty(self, tmp_path):
        sl = make_session_log(tmp_path)
        assert sl.last_event_id("!nonexistent:matrix.local") is None

    def test_last_event_id_all_null(self, tmp_path):
        sl = make_session_log(tmp_path)
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  call_id="c1", name="shell", output="ok")
        sl.append(role="system", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  event="session_start")
        assert sl.last_event_id(ROOM_ID) is None


class TestOverflow:

    def test_large_output_overflow(self, tmp_path):
        sl = make_session_log(tmp_path)
        big_output = "x" * (OVERFLOW_THRESHOLD + 1000)

        sl.append(
            role="tool",
            sender=AGENT_USER,
            room=ROOM_ID,
            event_id=None,
            call_id="c1",
            name="shell",
            output=big_output,
        )

        entries = sl.read(ROOM_ID)
        assert len(entries) == 1
        e = entries[0]
        assert e["truncated"] is True
        assert len(e["output"]) == OVERFLOW_THRESHOLD
        assert "overflow_path" in e

    def test_overflow_file_contains_full_output(self, tmp_path):
        sl = make_session_log(tmp_path)
        big_output = "y" * (OVERFLOW_THRESHOLD + 500)

        sl.append(
            role="tool",
            sender=AGENT_USER,
            room=ROOM_ID,
            event_id=None,
            call_id="mycall",
            name="shell",
            output=big_output,
        )

        entries = sl.read(ROOM_ID)
        overflow_path = Path(entries[0]["overflow_path"])
        assert overflow_path.exists()
        full_content = overflow_path.read_text(encoding="utf-8")
        assert full_content == big_output


# ---------------------------------------------------------------------------
# build_context tests
# ---------------------------------------------------------------------------


class TestBuildContext:

    def test_build_context_user_assistant(self, tmp_path):
        sl = make_session_log(tmp_path)
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$e1", content="Hello")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id="$e2", content="Hi there")

        ctx = sl.build_context(ROOM_ID)
        assert len(ctx) == 2
        assert ctx[0] == {"role": "user", "content": "Hello"}
        assert ctx[1] == {"role": "assistant", "content": "Hi there"}

    def test_build_context_skips_system(self, tmp_path):
        sl = make_session_log(tmp_path)
        sl.append(role="system", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  event="session_start", detail="test")
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$e1", content="Hello")

        ctx = sl.build_context(ROOM_ID, skip_system=True)
        assert len(ctx) == 1
        assert ctx[0]["role"] == "user"

    def test_build_context_includes_system_when_requested(self, tmp_path):
        sl = make_session_log(tmp_path)
        sl.append(role="system", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  event="session_start", detail="test detail")
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$e1", content="Hello")

        ctx = sl.build_context(ROOM_ID, skip_system=False)
        assert len(ctx) == 2
        assert ctx[0]["role"] == "system"

    def test_build_context_includes_tool(self, tmp_path):
        sl = make_session_log(tmp_path)
        tool_calls = [{"call_id": "c1", "name": "shell", "input": {"command": "uptime"}}]
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id="$e1",
                  content="Running...", tool_calls=tool_calls)
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  call_id="c1", name="shell", output="up 3 days")

        ctx = sl.build_context(ROOM_ID)
        assert len(ctx) == 2

        # Tool result should use tool_call_id (not call_id) to match agent.py format
        tool_entry = ctx[1]
        assert tool_entry["role"] == "tool"
        assert tool_entry["tool_call_id"] == "c1"
        assert tool_entry["content"] == "up 3 days"

    def test_build_context_empty_room(self, tmp_path):
        sl = make_session_log(tmp_path)
        ctx = sl.build_context("!nonexistent:matrix.local")
        assert ctx == []

    def test_build_context_assistant_with_tool_calls(self, tmp_path):
        """tool_calls rehydrated as ToolCall objects (not raw dicts)."""
        sl = make_session_log(tmp_path)
        tool_calls = [{"call_id": "c1", "name": "shell", "input": {"command": "uptime"}}]
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id="$e1",
                  content="", tool_calls=tool_calls)

        ctx = sl.build_context(ROOM_ID)
        rehydrated = ctx[0]["tool_calls"]
        assert len(rehydrated) == 1
        tc = rehydrated[0]
        # Must be ToolCall objects so provider code can do tc.id, tc.name, tc.input
        assert tc.id == "c1"
        assert tc.name == "shell"
        assert tc.input == {"command": "uptime"}

    def test_build_context_assistant_no_tool_calls(self, tmp_path):
        sl = make_session_log(tmp_path)
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id="$e1",
                  content="Done.")

        ctx = sl.build_context(ROOM_ID)
        assert "tool_calls" not in ctx[0]


# ---------------------------------------------------------------------------
# MatrixBot integration tests (mocked)
# ---------------------------------------------------------------------------


def make_bot_with_session_log(tmp_path, user_id=AGENT_USER):
    """Create a MatrixBot instance (bypassing __init__) with session_log set."""
    config = make_matrix_config(user_id=user_id)
    agent = MagicMock()
    agent.handle_input = AsyncMock(return_value="I'm on it")
    agent.history = MagicMock(return_value=[])
    agent.config = make_agent_config(tmp_path)

    bot = MatrixBot.__new__(MatrixBot)
    bot.config = config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock()
    bot.client.room_typing = AsyncMock()
    bot.client.room_messages = AsyncMock(return_value=MagicMock(chunk=[], end=None))
    bot._set_typing = AsyncMock()
    bot.send = AsyncMock()
    bot.send_notice = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = set()
    bot.session_log = SessionLog(tmp_path, user_id)

    return bot, agent


class TestMatrixBotIntegration:

    @pytest.mark.asyncio
    async def test_activate_room_reads_session_log(self, tmp_path):
        """_activate_room reads from JSONL, not Matrix history."""
        bot, agent = make_bot_with_session_log(tmp_path)

        # Pre-populate session log
        bot.session_log.append(
            role="user", sender=USER, room=ROOM_ID,
            event_id="$e1", content="Old message"
        )

        # Mock room_messages to return nothing (to verify Matrix pagination NOT used)
        history_list = []
        agent.history = MagicMock(return_value=history_list)

        await bot._activate_room(ROOM_ID)

        # Room should be active
        assert ROOM_ID in bot._active_rooms
        # History should have been populated from JSONL (not empty)
        assert len(history_list) > 0

    @pytest.mark.asyncio
    async def test_user_message_appended_to_log(self, tmp_path):
        """Incoming user message is written to session log."""
        bot, agent = make_bot_with_session_log(tmp_path)
        bot._active_rooms.add(ROOM_ID)  # Pre-activate to skip _activate_room

        event = make_room_message(USER, "What's the status?", event_id="$evt99")
        room = MagicMock()
        room.room_id = ROOM_ID

        await bot._handle_room_message(room, event)

        entries = bot.session_log.read(ROOM_ID)
        user_entries = [e for e in entries if e["role"] == "user"]
        assert len(user_entries) >= 1
        assert user_entries[0]["content"] == "What's the status?"
        assert user_entries[0]["event_id"] == "$evt99"

    @pytest.mark.asyncio
    async def test_assistant_response_appended_to_log(self, tmp_path):
        """Agent response is written to session log."""
        bot, agent = make_bot_with_session_log(tmp_path)
        bot._active_rooms.add(ROOM_ID)
        agent.handle_input = AsyncMock(return_value="Here's the status!")

        event = make_room_message(USER, "Status please")
        room = MagicMock()
        room.room_id = ROOM_ID

        await bot._handle_room_message(room, event)

        entries = bot.session_log.read(ROOM_ID)
        assistant_entries = [e for e in entries if e["role"] == "assistant"]
        assert len(assistant_entries) >= 1
        assert assistant_entries[-1]["content"] == "Here's the status!"

    @pytest.mark.asyncio
    async def test_tool_call_appended_to_log(self, tmp_path):
        """Tool execution results are written to session log."""
        bot, agent = make_bot_with_session_log(tmp_path)
        bot._active_rooms.add(ROOM_ID)

        # Simulate agent calling _on_tool_call
        async def fake_handle_input(body, room_id):
            if bot.agent._on_tool_call:
                await bot.agent._on_tool_call("call_1", "shell", {"command": "uptime"}, "up 3 days", False)
            return "Done"

        agent.handle_input = fake_handle_input
        # Make _on_tool_call accessible on the bot's agent mock
        agent._on_tool_call = None

        event = make_room_message(USER, "Run uptime")
        room = MagicMock()
        room.room_id = ROOM_ID

        await bot._handle_room_message(room, event)

        entries = bot.session_log.read(ROOM_ID)
        tool_entries = [e for e in entries if e["role"] == "tool"]
        assert len(tool_entries) >= 1
        assert tool_entries[0]["name"] == "shell"
        assert tool_entries[0]["output"] == "up 3 days"

    @pytest.mark.asyncio
    async def test_tool_intent_appended_to_log(self, tmp_path):
        """Assistant tool-call intent is written to session log before execution."""
        bot, agent = make_bot_with_session_log(tmp_path)
        bot._active_rooms.add(ROOM_ID)

        # Simulate agent emitting tool intent then returning
        async def fake_handle_input(body, room_id):
            if bot.agent._on_tool_intent:
                # Simulate ToolCall objects
                tc = MagicMock()
                tc.id = "call_1"
                tc.name = "shell"
                tc.input = {"command": "uptime"}
                await bot.agent._on_tool_intent([tc], "Let me check...")
            if bot.agent._on_tool_call:
                await bot.agent._on_tool_call("call_1", "shell", {"command": "uptime"}, "up 3 days", False)
            return "System has been up 3 days."

        agent.handle_input = fake_handle_input
        agent._on_tool_call = None
        agent._on_tool_intent = None

        event = make_room_message(USER, "How long has it been up?")
        room = MagicMock()
        room.room_id = ROOM_ID

        await bot._handle_room_message(room, event)

        entries = bot.session_log.read(ROOM_ID)
        # Should see: user → assistant(intent) → tool → assistant(final)
        roles = [e["role"] for e in entries]
        assert roles == ["user", "assistant", "tool", "assistant"]

        # Intent entry has tool_calls
        intent = entries[1]
        assert intent["role"] == "assistant"
        assert "tool_calls" in intent
        assert intent["tool_calls"][0]["name"] == "shell"
        assert intent["tool_calls"][0]["call_id"] == "call_1"
        assert intent["content"] == "Let me check..."

        # Final entry is the text response
        final = entries[3]
        assert final["role"] == "assistant"
        assert final["content"] == "System has been up 3 days."
