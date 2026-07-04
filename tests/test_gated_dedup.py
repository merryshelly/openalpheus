"""Regression tests for the gated-room double-append bug.

Bug: In gated rooms (3+ members), the triggering user message is appended to
in-memory conversation history TWICE before being sent to the LLM.

Path of the bug:
  1. _handle_room_message → session_log.append(role="user", ...) → JSONL written
  2. _process_message → history.clear(); history.extend(build_context(...))
     → in-memory history rebuilt, now ends with the new user message
  3. agent.handle_input → history.append({"role": "user", ...})
     → user message appended AGAIN

Fix:
  - agent.handle_input gains append_user=True kwarg; when False skips the append
  - _process_message passes append_user=False on the gated/hydrated path
  - provider.py adds _dedup_trailing_user() guardrail called from both converters

These tests are designed to FAIL against the unfixed code and PASS after the fix.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

from openalph.agent import Agent
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.matrix import MatrixBot
from openalph.session import SessionLog


# ── Fixtures (mirrored from test_matrix_gating.py) ────────────────────────────

def make_matrix_config(rooms=None, **kwargs):
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
        rooms=rooms,
    )
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def make_agent_config(workspace, **kwargs):
    from openalph.config import ProviderConfig
    defaults = dict(
        name="watson",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={
            "anthropic": ProviderConfig(
                key="anthropic", type="anthropic",
                api_key="sk-test", base_url=None, quirks=[],
            )
        },
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
    event.server_timestamp = 1_000_000
    content = {"msgtype": "m.text", "body": body}
    if mentions_user_ids is not None:
        content["m.mentions"] = {"user_ids": mentions_user_ids}
    event.source = {"content": content}
    return event


def make_bot_with_real_agent(tmp_path, rooms=None, reminders=True):
    """
    Create a MatrixBot backed by a REAL Agent instance but with the
    provider's stream() function mocked so no network calls are made.

    Returns (bot, captured_messages_list).  Every call to the mocked
    stream() will append a copy of the messages list it received to
    captured_messages_list, letting us inspect exactly what was sent
    to the LLM.
    """
    matrix_config = make_matrix_config(rooms=rooms)
    agent_config = make_agent_config(workspace=tmp_path, reminders=reminders)

    # Write a minimal SOUL.md so Agent.__init__ can assemble a system prompt
    (tmp_path / "SOUL.md").write_text("Test soul.")

    agent = Agent(agent_config)

    captured_messages = []

    # We'll inject this mock via patch inside each test
    return agent, matrix_config, captured_messages


def make_stream_mock(captured_messages):
    """Return an async-generator factory that records messages and yields one text event."""
    from openalph.provider import StreamEvent, Response, Usage

    async def _stream(config, system, messages, **kwargs):
        captured_messages.append(list(messages))  # snapshot
        yield StreamEvent(type="text", content="ok")
        yield StreamEvent(
            type="done",
            response=Response(
                content="ok",
                model="claude-sonnet-4-20250514",
                usage=Usage(input_tokens=10, output_tokens=2),
                stop_reason="end_turn",
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-20250514",
        )

    return _stream


def _pre_populate_jsonl(session_log, room_id, n_pairs):
    """Write n user+assistant pairs to JSONL."""
    for i in range(n_pairs):
        session_log.append(
            role="user",
            sender="@alice:matrix.local",
            room=room_id,
            event_id=f"$prior_user_{i}",
            content=f"Prior user message {i}",
        )
        session_log.append(
            role="assistant",
            sender="@watson:matrix.local",
            room=room_id,
            event_id=None,
            content=f"Prior assistant reply {i}",
        )


# ── Integration Tests ──────────────────────────────────────────────────────────

class TestGatedDedup:
    """Core regression: gated-room messages must appear exactly once in the wire payload."""

    @pytest.mark.asyncio
    async def test_gated_single_user_entry_in_wire_payload(self, tmp_path):
        """
        REGRESSION TEST — Core bug.

        In a 3-member (gated) room, an @mentioned message must appear
        exactly ONCE in the messages list passed to stream().  Before the
        fix, it appeared twice (hydration + handle_input append).
        """
        # reminders disabled: this test pins gated-dedup payload shape absent guidance injection (kdsn.186)
        agent, matrix_config, captured = make_bot_with_real_agent(tmp_path, reminders=False)
        room_id = "!group:matrix.local"
        body = "Hey @watson please respond"

        # Pre-populate one prior exchange so we can tell real hydration apart
        session_log = SessionLog(tmp_path, matrix_config.user_id)
        session_log.append(
            role="user", sender="@alice:matrix.local", room=room_id,
            event_id="$prior1", content="Prior message",
        )
        session_log.append(
            role="assistant", sender="@watson:matrix.local", room=room_id,
            event_id=None, content="Prior reply",
        )

        with patch("openalph.matrix.AsyncClient"):
            bot = MatrixBot(agent, matrix_config)

        # Swap the session_log for the one we pre-populated
        bot.session_log = session_log
        bot._synced = True
        bot.send = AsyncMock()
        bot.send_notice = AsyncMock()
        bot._set_typing = AsyncMock()

        room = make_room(room_id, 3)
        event = make_event(
            "@alice:matrix.local",
            body,
            event_id="$trigger1",
            mentions_user_ids=["@watson:matrix.local"],
        )

        stream_mock = make_stream_mock(captured)

        with patch("openalph.agent.stream", stream_mock):
            await bot._handle_room_message(room, event)
            if bot._background_tasks:
                await asyncio.gather(*bot._background_tasks, return_exceptions=True)

        assert len(captured) >= 1, "stream() was never called"
        messages = captured[-1]

        # Collect all user-role messages at the tail
        trailing_users = []
        for msg in reversed(messages):
            if msg.get("role") == "user":
                trailing_users.append(msg)
            else:
                break  # stop at first non-user

        # Must have exactly ONE trailing user message
        assert len(trailing_users) == 1, (
            f"Expected 1 trailing user entry, got {len(trailing_users)}: "
            f"{[m.get('content') for m in trailing_users]}"
        )

        # That message must be the trigger body (after strip_mention, "@watson" prefix removed)
        last_user_content = messages[-1].get("content", "")
        assert "please respond" in last_user_content or body in last_user_content, (
            f"Trailing user entry doesn't look like the trigger: {last_user_content!r}"
        )

    @pytest.mark.asyncio
    async def test_gated_hydration_preserves_prior_history(self, tmp_path):
        """
        After fix, gated-room hydration must preserve all prior JSONL entries.

        5 prior user+assistant pairs + 1 new user message = 11 entries total
        in the messages list passed to stream().
        """
        N_PAIRS = 5
        # reminders disabled: this test pins gated-dedup payload shape absent guidance injection (kdsn.186)
        agent, matrix_config, captured = make_bot_with_real_agent(tmp_path, reminders=False)
        room_id = "!group2:matrix.local"

        session_log = SessionLog(tmp_path, matrix_config.user_id)
        _pre_populate_jsonl(session_log, room_id, N_PAIRS)

        with patch("openalph.matrix.AsyncClient"):
            bot = MatrixBot(agent, matrix_config)

        bot.session_log = session_log
        bot._synced = True
        bot.send = AsyncMock()
        bot.send_notice = AsyncMock()
        bot._set_typing = AsyncMock()

        room = make_room(room_id, 3)
        event = make_event(
            "@alice:matrix.local",
            "@watson hydration test",
            event_id="$hydr1",
            mentions_user_ids=["@watson:matrix.local"],
        )

        stream_mock = make_stream_mock(captured)

        with patch("openalph.agent.stream", stream_mock):
            await bot._handle_room_message(room, event)
            if bot._background_tasks:
                await asyncio.gather(*bot._background_tasks, return_exceptions=True)

        assert len(captured) >= 1, "stream() was never called"
        messages = captured[-1]

        # 5 pairs = 10 prior entries, + 1 new user message = 11
        expected = N_PAIRS * 2 + 1
        assert len(messages) == expected, (
            f"Expected {expected} messages (5 pairs + 1 new), got {len(messages)}:\n"
            + "\n".join(f"  [{m['role']}] {str(m.get('content',''))[:60]}" for m in messages)
        )

        # Last must be the new user message
        assert messages[-1]["role"] == "user"
        assert "hydration test" in messages[-1].get("content", "")

    @pytest.mark.asyncio
    async def test_ungated_appends_user_once(self, tmp_path):
        """
        Ungated path (2-member DM): user message must appear exactly once.
        This is a non-regression check — the fix must not break DM rooms.
        """
        agent, matrix_config, captured = make_bot_with_real_agent(tmp_path)
        room_id = "!dm:matrix.local"

        session_log = SessionLog(tmp_path, matrix_config.user_id)

        with patch("openalph.matrix.AsyncClient"):
            bot = MatrixBot(agent, matrix_config)

        bot.session_log = session_log
        bot._synced = True
        bot.send = AsyncMock()
        bot.send_notice = AsyncMock()
        bot._set_typing = AsyncMock()

        # 2-member room → ungated
        room = make_room(room_id, 2)
        event = make_event(
            "@alice:matrix.local",
            "Hello in DM",
            event_id="$dm1",
        )

        stream_mock = make_stream_mock(captured)

        with patch("openalph.agent.stream", stream_mock):
            await bot._handle_room_message(room, event)
            if bot._background_tasks:
                await asyncio.gather(*bot._background_tasks, return_exceptions=True)

        assert len(captured) >= 1, "stream() was never called"
        messages = captured[-1]

        # Count trailing user messages
        trailing_users = []
        for msg in reversed(messages):
            if msg.get("role") == "user":
                trailing_users.append(msg)
            else:
                break

        assert len(trailing_users) == 1, (
            f"DM: expected 1 trailing user entry, got {len(trailing_users)}"
        )
        assert messages[-1].get("content") == "Hello in DM"


# ── Unit Test: provider._dedup_trailing_user ──────────────────────────────────

class TestProviderDedupTrailingUser:
    """Unit tests for the _dedup_trailing_user() guardrail in provider.py."""

    def test_dedup_removes_identical_trailing_user(self):
        """Two identical consecutive user messages at tail → one removed."""
        from openalph.provider import _dedup_trailing_user
        messages = [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "hello"},
        ]
        result = _dedup_trailing_user(messages)
        assert len(result) == 2
        assert result[-1] == {"role": "user", "content": "hello"}

    def test_dedup_passes_through_non_duplicate(self):
        """Different content in consecutive user messages → both preserved."""
        from openalph.provider import _dedup_trailing_user
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = _dedup_trailing_user(messages)
        assert len(result) == 2

    def test_dedup_passes_through_single_user(self):
        """Single trailing user message → unchanged."""
        from openalph.provider import _dedup_trailing_user
        messages = [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "hello"},
        ]
        result = _dedup_trailing_user(messages)
        assert len(result) == 2

    def test_dedup_passes_through_user_assistant_user(self):
        """Non-consecutive user messages → unchanged."""
        from openalph.provider import _dedup_trailing_user
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "hello"},
        ]
        result = _dedup_trailing_user(messages)
        assert len(result) == 3

    def test_dedup_empty_messages(self):
        """Empty list → unchanged."""
        from openalph.provider import _dedup_trailing_user
        assert _dedup_trailing_user([]) == []

    def test_dedup_single_message(self):
        """Single message → unchanged."""
        from openalph.provider import _dedup_trailing_user
        msgs = [{"role": "user", "content": "x"}]
        assert _dedup_trailing_user(msgs) == msgs

    def test_dedup_called_by_convert_openai(self, caplog):
        """_convert_messages_for_openai must strip duplicate trailing user."""
        import logging
        from openalph.provider import _convert_messages_for_openai
        messages = [
            {"role": "user", "content": "dup"},
            {"role": "user", "content": "dup"},
        ]
        with caplog.at_level(logging.WARNING, logger="openalph.provider"):
            result = _convert_messages_for_openai(messages)
        assert len(result) == 1
        assert "dedup" in caplog.text.lower() or "trailing" in caplog.text.lower()

    def test_dedup_called_by_convert_anthropic(self, caplog):
        """_convert_messages_for_anthropic must strip duplicate trailing user."""
        import logging
        from openalph.provider import _convert_messages_for_anthropic
        messages = [
            {"role": "user", "content": "dup"},
            {"role": "user", "content": "dup"},
        ]
        with caplog.at_level(logging.WARNING, logger="openalph.provider"):
            result = _convert_messages_for_anthropic(messages)
        assert len(result) == 1
        assert "dedup" in caplog.text.lower() or "trailing" in caplog.text.lower()


# ── Unit Tests: agent.handle_input append_user kwarg ─────────────────────────

class TestHandleInputAppendUser:
    """Unit tests for the append_user kwarg on Agent.handle_input."""

    def _make_agent(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("Test soul.")
        config = make_agent_config(workspace=tmp_path)
        return Agent(config)

    def _make_stream_events(self, content="ok"):
        from openalph.provider import StreamEvent, Response, Usage
        async def _stream(*args, **kwargs):
            yield StreamEvent(type="text", content=content)
            yield StreamEvent(
                type="done",
                response=Response(
                    content=content,
                    model="claude-sonnet-4-20250514",
                    usage=Usage(input_tokens=5, output_tokens=2),
                    stop_reason="end_turn",
                ),
                stop_reason="end_turn",
                model="claude-sonnet-4-20250514",
            )
        return _stream

    @pytest.mark.asyncio
    async def test_handle_input_append_user_false(self, tmp_path):
        """
        With append_user=False, handle_input must NOT append another user entry.

        Pre-populate history with a user message that is already there
        (simulating what gated-room hydration does).  After the call,
        there must still be exactly one user entry, not two.
        """
        agent = self._make_agent(tmp_path)
        room_id = "!test:matrix.local"

        # Pre-populate: simulate what build_context+history.extend does
        history = agent.history(room_id)
        history.append({"role": "user", "content": "hi"})
        initial_len = len(history)
        assert initial_len == 1

        with patch("openalph.agent.stream", self._make_stream_events()):
            await agent.handle_input("hi", room_id, append_user=False)

        # After the call history should have grown by one (the assistant reply),
        # but NOT by a second user entry.
        final_history = agent.history(room_id)
        user_entries = [m for m in final_history if m.get("role") == "user"]
        assert len(user_entries) == 1, (
            f"append_user=False: expected 1 user entry, got {len(user_entries)}: "
            f"{user_entries}"
        )

    @pytest.mark.asyncio
    async def test_handle_input_append_user_default_true(self, tmp_path):
        """
        Default behaviour (append_user=True) is preserved.
        Starting from empty history, one call → one user entry in history.
        """
        agent = self._make_agent(tmp_path)
        room_id = "!test2:matrix.local"

        assert agent.history(room_id) == []

        with patch("openalph.agent.stream", self._make_stream_events()):
            await agent.handle_input("hello", room_id)

        final_history = agent.history(room_id)
        user_entries = [m for m in final_history if m.get("role") == "user"]
        assert len(user_entries) == 1
        assert user_entries[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_handle_input_append_user_explicit_true(self, tmp_path):
        """
        Explicit append_user=True behaves identically to the default.
        """
        agent = self._make_agent(tmp_path)
        room_id = "!test3:matrix.local"

        with patch("openalph.agent.stream", self._make_stream_events()):
            await agent.handle_input("hello", room_id, append_user=True)

        user_entries = [m for m in agent.history(room_id) if m.get("role") == "user"]
        assert len(user_entries) == 1
