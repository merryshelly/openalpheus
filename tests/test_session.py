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
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig


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
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(key="anthropic", type="anthropic", api_key="sk-test", base_url=None, quirks=[])},
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
        # Add matching tool result so crash recovery doesn't strip it
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  call_id="c1", name="shell", output="up 10 days")

        ctx = sl.build_context(ROOM_ID)
        rehydrated = ctx[0]["tool_calls"]
        assert len(rehydrated) == 1
        tc = rehydrated[0]
        # Must be ToolCall objects so provider code can do tc.id, tc.name, tc.input
        assert tc.id == "c1"
        assert tc.name == "shell"
        assert tc.input == {"command": "uptime"}
        # Tool result should also be present
        assert ctx[1]["role"] == "tool"
        assert ctx[1]["tool_call_id"] == "c1"

    def test_build_context_strips_orphaned_tool_calls(self, tmp_path):
        """Crash recovery: orphaned tool_calls at end of context are stripped."""
        sl = make_session_log(tmp_path)
        # User message followed by assistant with tool_calls but NO tool results
        # (simulates crash mid-tool-loop)
        sl.append(role="user", sender="@sb:matrix.local", room=ROOM_ID,
                  event_id="$u1", content="check uptime")
        tool_calls = [{"call_id": "c1", "name": "shell", "input": {"command": "uptime"}}]
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id="$e1",
                  content="", tool_calls=tool_calls)

        ctx = sl.build_context(ROOM_ID)
        # Orphaned assistant+tool_calls should be stripped, leaving only the user message
        assert len(ctx) == 1
        assert ctx[0]["role"] == "user"
        assert ctx[0]["content"] == "check uptime"

    def test_build_context_tool_result_includes_is_error(self, tmp_path):
        """Tool results preserve is_error flag through rehydration."""
        sl = make_session_log(tmp_path)
        tool_calls = [{"call_id": "c1", "name": "shell", "input": {"command": "fail"}}]
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id="$e1",
                  content="", tool_calls=tool_calls)
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  call_id="c1", name="shell", output="command not found", is_error=True)

        ctx = sl.build_context(ROOM_ID)
        assert ctx[1]["role"] == "tool"
        assert ctx[1]["is_error"] is True

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
    bot._halted_rooms = set()
    bot._room_thinking = {}
    bot._background_tasks = set()
    bot._session_locks = {}
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
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

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
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        entries = bot.session_log.read(ROOM_ID)
        assistant_entries = [e for e in entries if e["role"] == "assistant"]
        assert len(assistant_entries) >= 1
        assert assistant_entries[-1]["content"] == "Here's the status!"

    @pytest.mark.asyncio
    async def test_tool_call_appended_to_log(self, tmp_path):
        """Tool execution results are written to session log."""
        bot, agent = make_bot_with_session_log(tmp_path)
        bot._active_rooms.add(ROOM_ID)

        # Simulate agent calling on_tool_call
        async def fake_handle_input(body, room_id, *, on_tool_call=None, on_tool_intent=None, on_text_delta=None, on_thinking_delta=None, thinking=None, callbacks=None):
            if on_tool_call:
                await on_tool_call("call_1", "shell", {"command": "uptime"}, "up 3 days", False)
            return "Done"

        agent.handle_input = fake_handle_input

        event = make_room_message(USER, "Run uptime")
        room = MagicMock()
        room.room_id = ROOM_ID

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

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
        async def fake_handle_input(body, room_id, *, on_tool_call=None, on_tool_intent=None, on_text_delta=None, on_thinking_delta=None, thinking=None, callbacks=None):
            if on_tool_intent:
                # Simulate ToolCall objects
                tc = MagicMock()
                tc.id = "call_1"
                tc.name = "shell"
                tc.input = {"command": "uptime"}
                await on_tool_intent([tc], "Let me check...")
            if on_tool_call:
                await on_tool_call("call_1", "shell", {"command": "uptime"}, "up 3 days", False)
            return "System has been up 3 days."

        agent.handle_input = fake_handle_input

        event = make_room_message(USER, "How long has it been up?")
        room = MagicMock()
        room.room_id = ROOM_ID

        await bot._handle_room_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

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


class TestRehydrationRoundTrip:
    """End-to-end: write tool-use conversation to JSONL, rehydrate, verify
    the result passes through provider serialization without error.

    These tests catch serialization boundary bugs:
    - dict vs ToolCall objects (tc.id / tc.name / tc.input attribute access)
    - call_id mismatch between intent and tool result entries
    """

    def _write_tool_conversation(self, sl, room_id):
        """Write a realistic tool-use conversation to session log."""
        # User message
        sl.append(role="user", sender=USER, room=room_id,
                  event_id="$msg1", content="Check uptime")

        # Assistant intent (with tool_calls — what intent logging produces)
        sl.append(role="assistant", sender=AGENT_USER, room=room_id,
                  event_id=None, content="",
                  tool_calls=[{"call_id": "toolu_01ABC", "name": "shell",
                               "input": {"command": "uptime"}}])

        # Tool result (must use the same call_id as the intent)
        sl.append(role="tool", sender=AGENT_USER, room=room_id,
                  event_id=None, call_id="toolu_01ABC", name="shell",
                  output="up 5 days", truncated=False)

        # Final assistant response
        sl.append(role="assistant", sender=AGENT_USER, room=room_id,
                  event_id="$msg2", content="System has been up 5 days.")

    def test_rehydrated_tool_calls_are_toolcall_objects(self, tmp_path):
        """build_context returns ToolCall objects, not raw dicts."""
        from openalph.provider import ToolCall

        sl = make_session_log(tmp_path)
        self._write_tool_conversation(sl, ROOM_ID)
        ctx = sl.build_context(ROOM_ID)

        # Find the assistant turn with tool_calls
        intent_turns = [m for m in ctx if m["role"] == "assistant" and m.get("tool_calls")]
        assert len(intent_turns) == 1
        tc = intent_turns[0]["tool_calls"][0]
        assert isinstance(tc, ToolCall), f"Expected ToolCall, got {type(tc)}"
        assert tc.id == "toolu_01ABC"
        assert tc.name == "shell"
        assert tc.input == {"command": "uptime"}

    def test_tool_result_call_id_matches_intent(self, tmp_path):
        """tool_call_id on tool results must match the id in tool_calls."""
        sl = make_session_log(tmp_path)
        self._write_tool_conversation(sl, ROOM_ID)
        ctx = sl.build_context(ROOM_ID)

        intent = [m for m in ctx if m["role"] == "assistant" and m.get("tool_calls")][0]
        tool_result = [m for m in ctx if m["role"] == "tool"][0]

        assert tool_result["tool_call_id"] == intent["tool_calls"][0].id

    def test_rehydrated_context_survives_anthropic_serialization(self, tmp_path):
        """Full round-trip: JSONL → build_context → Anthropic message format."""
        from openalph.provider import _convert_messages_for_anthropic

        sl = make_session_log(tmp_path)
        self._write_tool_conversation(sl, ROOM_ID)
        ctx = sl.build_context(ROOM_ID)

        # This is what blows up if ToolCall rehydration is broken
        converted = _convert_messages_for_anthropic(ctx)

        # Verify structure: user → assistant(tool_use) → user(tool_result) → assistant
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"
        # Anthropic format puts tool_use in content blocks
        tool_use_blocks = [b for b in converted[1]["content"] if b["type"] == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["id"] == "toolu_01ABC"
        assert tool_use_blocks[0]["name"] == "shell"
        # Tool result becomes a user message with tool_result content block
        assert converted[2]["role"] == "user"
        tool_result_blocks = [b for b in converted[2]["content"] if b["type"] == "tool_result"]
        assert len(tool_result_blocks) == 1
        assert tool_result_blocks[0]["tool_use_id"] == "toolu_01ABC"

    def test_rehydrated_context_survives_openai_serialization(self, tmp_path):
        """Full round-trip: JSONL → build_context → OpenAI message format."""
        from openalph.provider import _convert_messages_for_openai

        sl = make_session_log(tmp_path)
        self._write_tool_conversation(sl, ROOM_ID)
        ctx = sl.build_context(ROOM_ID)

        converted = _convert_messages_for_openai(ctx)

        # Find assistant turn with tool_calls
        assistant_tc = [m for m in converted if m["role"] == "assistant" and m.get("tool_calls")]
        assert len(assistant_tc) == 1
        otc = assistant_tc[0]["tool_calls"][0]
        assert otc["id"] == "toolu_01ABC"
        assert otc["function"]["name"] == "shell"

        # Find tool result
        tool_msgs = [m for m in converted if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "toolu_01ABC"

    def test_multi_tool_call_round_trip(self, tmp_path):
        """Multiple parallel tool calls in one turn survive round-trip."""
        from openalph.provider import _convert_messages_for_anthropic

        sl = make_session_log(tmp_path)
        sl.append(role="user", sender=USER, room=ROOM_ID,
                  event_id="$m1", content="Check both")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID,
                  event_id=None, content="",
                  tool_calls=[
                      {"call_id": "toolu_A", "name": "shell", "input": {"command": "uptime"}},
                      {"call_id": "toolu_B", "name": "web_fetch", "input": {"url": "http://example.com"}},
                  ])
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID,
                  event_id=None, call_id="toolu_A", name="shell",
                  output="up 5 days", truncated=False)
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID,
                  event_id=None, call_id="toolu_B", name="web_fetch",
                  output="<html>...</html>", truncated=False)
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID,
                  event_id="$m2", content="Both checked.")

        ctx = sl.build_context(ROOM_ID)
        # Should not raise
        converted = _convert_messages_for_anthropic(ctx)

        tool_uses = [b for msg in converted for b in (msg.get("content") or [])
                     if isinstance(b, dict) and b.get("type") == "tool_use"]
        assert len(tool_uses) == 2
        assert {tu["id"] for tu in tool_uses} == {"toolu_A", "toolu_B"}


class TestBuildContextReorder:
    """Tests for build_context() reordering of interleaved messages.

    When a user message arrives during a long-running tool execution, the JSONL
    may record the sequence: assistant(tool_calls) → user → tool(result).
    The Anthropic API rejects this because tool_use must be immediately followed
    by tool_result. build_context() must reorder: assistant → tool → user.
    """

    def test_single_tool_call_user_interleaved(self, tmp_path):
        """User message wedged between assistant(tool_calls) and tool result is moved after."""
        sl = make_session_log(tmp_path)
        # assistant dispatches tool
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  content="", tool_calls=[{"call_id": "toolu_X", "name": "shell", "input": {"command": "sleep 60"}}])
        # user message arrives during tool execution
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$u1", content="new message")
        # tool finishes
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  call_id="toolu_X", name="shell", output="done")

        ctx = sl.build_context(ROOM_ID)
        roles = [m["role"] for m in ctx]
        # Must be: assistant → tool → user (not assistant → user → tool)
        assert roles == ["assistant", "tool", "user"]
        assert ctx[1]["tool_call_id"] == "toolu_X"
        assert ctx[2]["content"] == "new message"

    def test_parallel_tool_calls_user_interleaved(self, tmp_path):
        """User message between parallel tool calls is moved after all results."""
        sl = make_session_log(tmp_path)
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  content="", tool_calls=[
                      {"call_id": "toolu_A", "name": "shell", "input": {"command": "uptime"}},
                      {"call_id": "toolu_B", "name": "web_fetch", "input": {"url": "http://example.com"}},
                  ])
        # First tool result
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  call_id="toolu_A", name="shell", output="up 5 days")
        # User message arrives between results
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$u1", content="also check disk")
        # Second tool result
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  call_id="toolu_B", name="web_fetch", output="<html>ok</html>")

        ctx = sl.build_context(ROOM_ID)
        roles = [m["role"] for m in ctx]
        # Must be: assistant → tool → tool → user
        assert roles == ["assistant", "tool", "tool", "user"]
        # Tool results must be for the right calls
        assert {ctx[1]["tool_call_id"], ctx[2]["tool_call_id"]} == {"toolu_A", "toolu_B"}
        assert ctx[3]["content"] == "also check disk"

    def test_multiple_interleaved_user_messages(self, tmp_path):
        """Multiple user messages wedged in tool results are all moved."""
        sl = make_session_log(tmp_path)
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  content="", tool_calls=[{"call_id": "toolu_X", "name": "subagent", "input": {"task": "research"}}])
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$u1", content="first interruption")
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$u2", content="second interruption")
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  call_id="toolu_X", name="subagent", output="research done")

        ctx = sl.build_context(ROOM_ID)
        roles = [m["role"] for m in ctx]
        assert roles == ["assistant", "tool", "user", "user"]
        assert ctx[2]["content"] == "first interruption"
        assert ctx[3]["content"] == "second interruption"

    def test_no_reorder_when_user_after_all_tool_results(self, tmp_path):
        """User message legitimately after all tool results is not moved."""
        sl = make_session_log(tmp_path)
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  content="", tool_calls=[{"call_id": "toolu_X", "name": "shell", "input": {"command": "ls"}}])
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  call_id="toolu_X", name="shell", output="file.txt")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  content="Here are the files.")
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$u1", content="thanks")

        ctx = sl.build_context(ROOM_ID)
        roles = [m["role"] for m in ctx]
        # Already correct order — no reordering needed
        assert roles == ["assistant", "tool", "assistant", "user"]

    def test_reorder_survives_anthropic_serialization(self, tmp_path):
        """Reordered context passes Anthropic serialization without error."""
        from openalph.provider import _convert_messages_for_anthropic

        sl = make_session_log(tmp_path)
        # Simulate the interleaving bug
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$u0", content="run something")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  content="", tool_calls=[{"call_id": "toolu_X", "name": "shell", "input": {"command": "sleep 30"}}])
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$u1", content="still waiting?")
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  call_id="toolu_X", name="shell", output="done")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  content="All done.")

        ctx = sl.build_context(ROOM_ID)
        # This would blow up if the interleaving isn't fixed
        converted = _convert_messages_for_anthropic(ctx)

        # Verify: user → assistant(tool_use) → user(tool_result) → user(new msg) → assistant
        roles = [m["role"] for m in converted]
        assert roles == ["user", "assistant", "user", "user", "assistant"]
        # The tool_result should be in the first "user" after assistant
        tool_result_blocks = [b for b in converted[2]["content"] if b["type"] == "tool_result"]
        assert len(tool_result_blocks) == 1
        assert tool_result_blocks[0]["tool_use_id"] == "toolu_X"

    def test_reorder_with_completed_turn_before(self, tmp_path):
        """Reordering only affects the interleaved turn, not prior completed turns."""
        sl = make_session_log(tmp_path)
        # First turn: clean
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$u0", content="hello")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id=None, content="hi")
        # Second turn: interleaved
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$u1", content="run tool")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  content="", tool_calls=[{"call_id": "toolu_Y", "name": "shell", "input": {"command": "uptime"}}])
        sl.append(role="user", sender=USER, room=ROOM_ID, event_id="$u2", content="interruption")
        sl.append(role="tool", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  call_id="toolu_Y", name="shell", output="up 3 days")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_ID, event_id=None,
                  content="System up 3 days.")

        ctx = sl.build_context(ROOM_ID)
        roles = [m["role"] for m in ctx]
        assert roles == ["user", "assistant", "user", "assistant", "tool", "user", "assistant"]
        # The interleaved user message is after the tool result
        assert ctx[5]["content"] == "interruption"


class TestSafeCallId:
    """Tests for _safe_call_id() path traversal prevention."""

    def test_normal_call_id_passes_through(self, tmp_path):
        """Normal call_ids with safe chars are unchanged."""
        sl = SessionLog(tmp_path, "@agent:matrix.local")
        assert sl._safe_call_id("toolu_abc123") == "toolu_abc123"
        assert sl._safe_call_id("call-XYZ_99") == "call-XYZ_99"

    def test_path_traversal_is_sanitized(self, tmp_path):
        """Path traversal attempts are sanitized to safe filenames."""
        sl = SessionLog(tmp_path, "@agent:matrix.local")
        result = sl._safe_call_id("../../../etc/passwd")
        assert "/" not in result
        assert ".." not in result
        # Dots and slashes replaced with underscores; only safe chars remain
        import re
        assert re.fullmatch(r'[a-zA-Z0-9_-]+', result)

    def test_empty_string_falls_back_to_unknown(self, tmp_path):
        """Empty string call_id falls back to 'unknown'."""
        sl = SessionLog(tmp_path, "@agent:matrix.local")
        assert sl._safe_call_id("") == "unknown"

    def test_overflow_file_written_within_overflow_dir(self, tmp_path):
        """Overflow file for a path-traversal call_id stays inside overflow/."""
        sl = SessionLog(tmp_path, "@agent:matrix.local")
        big_output = "x" * (64 * 1024 + 1)
        sl.append(
            role="tool",
            sender="@agent:matrix.local",
            room="!room:matrix.local",
            event_id=None,
            call_id="../../../etc/cron.d/evil",
            name="shell",
            output=big_output,
        )
        overflow_dir = tmp_path / "sessions" / "overflow"
        # Only files within overflow/ should exist — no escape
        written = list(overflow_dir.iterdir())
        assert len(written) == 1
        # The written file must be directly inside overflow/, not a traversal
        assert written[0].parent == overflow_dir
        # Its name must not contain slashes or dots-dots
        assert "/" not in written[0].name
        assert ".." not in written[0].name
