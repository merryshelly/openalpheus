"""Tests for Phase 1 Track A CLI integration: session setup, resume, tool logging, round-trip.

Design spec: memory/projects/openalph/specs/cli-firstclass-and-matrix-decoupling.md
Bead: workspace-kdsn.237 (Phase 1) + workspace-kdsn.240 (persistence)

RED until cli.py is refactored to extract testable functions:
  - _sanitize_room_label, _generate_session_name
  - _setup_cli_session, _process_cli_line
And openalph/data/bip39-english.txt exists.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from openalph.session import SessionLog
from openalph.config import AgentConfig, ProviderConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(workspace):
    return AgentConfig(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic",
            api_key="sk-test", base_url=None, quirks=None,
        )},
        workspace=workspace,
        user_id="@test:server",
    )


def _make_agent(**overrides):
    agent = MagicMock()
    agent.system_prompt = "test prompt"
    agent.history = MagicMock(return_value=[])
    agent.status = MagicMock(return_value={
        "name": "test-agent", "model": "anthropic/claude-sonnet-4",
        "turns": 0, "context_tokens": 0, "context_max": 200000, "context_pct": 0,
    })
    agent.last_turn_usage = MagicMock(return_value={
        "input_tokens": 100, "output_tokens": 50,
        "cache_read_tokens": 0, "cache_creation_tokens": 0,
    })
    agent.restore_usage = MagicMock()
    agent.rehydrate_reminders = MagicMock()
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


# ---------------------------------------------------------------------------
# bip39 session name generation
# ---------------------------------------------------------------------------

class TestBip39SessionName:
    """_generate_session_name produces a 3-word bip39 mnemonic."""

    def test_importable_from_cli(self):
        from openalph.cli import _generate_session_name
        assert callable(_generate_session_name)

    def test_returns_three_words(self):
        from openalph.cli import _generate_session_name
        name = _generate_session_name()
        words = name.split()
        assert len(words) == 3

    def test_words_are_from_wordlist(self):
        """All words are in the BIP39 English wordlist."""
        from openalph.cli import _generate_session_name
        # Load the wordlist the same way the generator does
        data_path = Path(__file__).parent.parent / "src" / "openalph" / "data" / "bip39-english.txt"
        wordlist = set(w.strip() for w in data_path.read_text().splitlines() if w.strip())
        name = _generate_session_name()
        for word in name.split():
            assert word in wordlist, f"{word!r} not in BIP39 wordlist"

    def test_two_calls_differ(self):
        """Randomness: two calls produce different names (probabilistically certain)."""
        from openalph.cli import _generate_session_name
        names = {_generate_session_name() for _ in range(10)}
        assert len(names) > 1  # not all the same


# ---------------------------------------------------------------------------
# Room label sanitization (security — path traversal prevention)
# ---------------------------------------------------------------------------

class TestRoomLabelSanitization:
    """_sanitize_room_label prevents path traversal via --room.

    SessionLog._room_id_safe strips '!' and replaces ':' but does NOT handle
    '../', '/', or other path characters. The CLI must sanitize before the
    label hits _session_path.
    """

    def test_importable_from_cli(self):
        from openalph.cli import _sanitize_room_label
        assert callable(_sanitize_room_label)

    def test_normal_label_passes_through(self):
        from openalph.cli import _sanitize_room_label
        assert _sanitize_room_label("my_session") == "my_session"
        assert _sanitize_room_label("debug-01") == "debug-01"
        assert _sanitize_room_label("test.room") == "test.room"

    def test_rejects_path_traversal(self):
        """../evil must be rejected — not silently sanitized to a safe path."""
        from openalph.cli import _sanitize_room_label
        with pytest.raises((ValueError, SystemExit)):
            _sanitize_room_label("../../etc/passwd")

    def test_rejects_slashes(self):
        from openalph.cli import _sanitize_room_label
        with pytest.raises((ValueError, SystemExit)):
            _sanitize_room_label("foo/bar")

    def test_rejects_empty(self):
        from openalph.cli import _sanitize_room_label
        with pytest.raises((ValueError, SystemExit)):
            _sanitize_room_label("")

    def test_rejects_absolute_path(self):
        from openalph.cli import _sanitize_room_label
        with pytest.raises((ValueError, SystemExit)):
            _sanitize_room_label("/etc/passwd")


# ---------------------------------------------------------------------------
# CLI session setup (new + resume)
# ---------------------------------------------------------------------------

class TestCLISessionSetup:
    """_setup_cli_session creates or resumes a CLI session with SessionLog."""

    def test_importable_from_cli(self):
        from openalph.cli import _setup_cli_session
        assert callable(_setup_cli_session)

    def test_new_session_creates_session_log(self, tmp_path):
        """A new session creates a SessionLog and logs session_start."""
        from openalph.cli import _setup_cli_session
        config = _make_config(tmp_path)
        agent = _make_agent()
        sl, room_name, is_new = _setup_cli_session(config, agent, "_cli", explicit_room=False)
        assert isinstance(sl, SessionLog)
        assert is_new is True
        entries = sl.read("_cli")
        # session_start system entry
        starts = [e for e in entries if e.get("event") == "session_start"]
        assert len(starts) == 1
        assert starts[0]["room_name"] == room_name

    def test_new_session_generates_bip39_name(self, tmp_path):
        """Default (_cli) session gets a bip39 room_name, stored in session_start."""
        from openalph.cli import _setup_cli_session
        config = _make_config(tmp_path)
        agent = _make_agent()
        sl, room_name, is_new = _setup_cli_session(config, agent, "_cli", explicit_room=False)
        assert is_new is True
        assert len(room_name.split()) == 3  # bip39 3-word name

    def test_explicit_room_uses_label_as_name(self, tmp_path):
        """When --room is explicitly set, the label IS the room_name (no bip39)."""
        from openalph.cli import _setup_cli_session
        config = _make_config(tmp_path)
        agent = _make_agent()
        sl, room_name, is_new = _setup_cli_session(config, agent, "debug-session", explicit_room=True)
        assert room_name == "debug-session"

    def test_resume_hydrates_history_from_jsonl(self, tmp_path):
        """On resume, agent.history is rebuilt from build_context."""
        from openalph.cli import _setup_cli_session
        config = _make_config(tmp_path)
        agent = _make_agent()
        # Pre-populate a session JSONL
        sl = SessionLog(tmp_path, "@test:server")
        sl.append(role="user", sender="op", room="_cli", content="hello")
        sl.append(role="assistant", sender="@test:server", room="_cli", content="hi there")
        history_mock = MagicMock()
        agent.history = MagicMock(return_value=history_mock)
        sl2, room_name, is_new = _setup_cli_session(config, agent, "_cli", explicit_room=False)
        assert is_new is False
        # history.clear + history.extend were called (rebuild from JSONL)
        history_mock.clear.assert_called_once()
        history_mock.extend.assert_called_once()
        extended_with = history_mock.extend.call_args[0][0]
        assert len(extended_with) >= 2  # user + assistant entries

    def test_resume_restores_usage(self, tmp_path):
        """On resume, agent.restore_usage is called with usage_totals."""
        from openalph.cli import _setup_cli_session
        config = _make_config(tmp_path)
        agent = _make_agent()
        sl = SessionLog(tmp_path, "@test:server")
        sl.append(role="user", sender="op", room="_cli", content="hello")
        sl.append(role="assistant", sender="@test:server", room="_cli",
                  content="hi", usage={"input_tokens": 100, "output_tokens": 50,
                                       "cache_read_tokens": 0, "cache_creation_tokens": 0,
                                       "tool_calls": 0})
        _setup_cli_session(config, agent, "_cli", explicit_room=False)
        agent.restore_usage.assert_called_once()

    def test_resume_reads_stored_bip39_name(self, tmp_path):
        """On resume, the bip39 room_name is read from the session_start entry."""
        from openalph.cli import _setup_cli_session
        config = _make_config(tmp_path)
        agent = _make_agent()
        sl = SessionLog(tmp_path, "@test:server")
        sl.append(role="system", sender="@test:server", room="_cli",
                  event="session_start", room_name="ocean fabric thunder")
        sl.append(role="user", sender="op", room="_cli", content="hello")
        sl2, room_name, is_new = _setup_cli_session(config, agent, "_cli", explicit_room=False)
        assert is_new is False
        assert room_name == "ocean fabric thunder"

    def test_resume_honors_handoff_boundary_strip(self, tmp_path):
        """audit-fix (kdsn.322.9): CLI resume must render with handoff.

        A resumed session containing a handoff_boundary marker must have its
        pre-boundary entries DROPPED from the rebuilt history (full strip) —
        the old code constructed SessionLog without handoff_default, so
        resume rendered FULL pre-boundary history.
        """
        from openalph.cli import _setup_cli_session
        config = _make_config(tmp_path)
        agent = _make_agent()
        # Pre-populate: pre-boundary content, marker, snapshot, post-boundary.
        sl = SessionLog(tmp_path, "@test:server")
        sl.append(role="system", sender="@test:server", room="_cli",
                  event="session_start", room_name="ocean fabric thunder")
        sl.append(role="user", sender="op", room="_cli",
                  content="PRE-BOUNDARY-MARKER-XYZ")
        sl.append(role="assistant", sender="@test:server", room="_cli",
                  content="pre-boundary reply")
        sl.append(role="system", sender="@test:server", room="_cli",
                  event="handoff_boundary", entry_index=3,
                  detail="{}")
        sl.append(role="user", sender="@test:server", room="_cli",
                  content="SNAPSHOT-MARKER-XYZ",
                  source="handoff_snapshot")
        sl.append(role="user", sender="op", room="_cli",
                  content="POST-BOUNDARY-MARKER-XYZ")
        history_mock = MagicMock()
        agent.history = MagicMock(return_value=history_mock)
        _setup_cli_session(config, agent, "_cli", explicit_room=False)
        extended_with = history_mock.extend.call_args[0][0]
        contents = [m.get("content", "") for m in extended_with]
        blob = "\n".join(c for c in contents if isinstance(c, str))
        assert "PRE-BOUNDARY-MARKER-XYZ" not in blob  # dropped by full strip
        assert "SNAPSHOT-MARKER-XYZ" in blob  # snapshot renders verbatim
        assert "POST-BOUNDARY-MARKER-XYZ" in blob  # post-boundary survives

    def test_resume_handoff_disabled_renders_full(self, tmp_path):
        """audit-fix (kdsn.322.9): handoff_enabled=False → full verbatim resume."""
        from openalph.cli import _setup_cli_session
        config = _make_config(tmp_path)
        config.context.handoff_enabled = False
        agent = _make_agent()
        sl = SessionLog(tmp_path, "@test:server")
        sl.append(role="system", sender="@test:server", room="_cli",
                  event="session_start", room_name="ocean fabric thunder")
        sl.append(role="user", sender="op", room="_cli",
                  content="PRE-BOUNDARY-MARKER-XYZ")
        sl.append(role="system", sender="@test:server", room="_cli",
                  event="handoff_boundary", entry_index=2,
                  detail="{}")
        history_mock = MagicMock()
        agent.history = MagicMock(return_value=history_mock)
        _setup_cli_session(config, agent, "_cli", explicit_room=False)
        extended_with = history_mock.extend.call_args[0][0]
        blob = "\n".join(m.get("content", "") for m in extended_with
                         if isinstance(m.get("content"), str))
        assert "PRE-BOUNDARY-MARKER-XYZ" in blob  # flag off → no strip

    def test_resume_pre_phase1_session_no_room_name(self, tmp_path):
        """A pre-Phase-1 session (no room_name in session_start) doesn't crash."""
        from openalph.cli import _setup_cli_session
        config = _make_config(tmp_path)
        agent = _make_agent()
        sl = SessionLog(tmp_path, "@test:server")
        sl.append(role="system", sender="@test:server", room="_cli", event="session_start")
        sl.append(role="user", sender="op", room="_cli", content="hello")
        sl2, room_name, is_new = _setup_cli_session(config, agent, "_cli", explicit_room=False)
        assert is_new is False
        # room_name is None or a fallback — just must not crash
        assert room_name is None or isinstance(room_name, str)


# ---------------------------------------------------------------------------
# CLI message processing (slash commands + regular messages)
# ---------------------------------------------------------------------------

class TestCLIMessageProcessing:
    """_process_cli_line handles slash commands and regular messages."""

    def test_importable_from_cli(self):
        from openalph.cli import _process_cli_line
        assert callable(_process_cli_line)

    @pytest.mark.asyncio
    async def test_quit_returns_exit(self, tmp_path):
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        sl = SessionLog(tmp_path, "@test:server")
        response, should_exit = await _process_cli_line(
            agent, sl, {}, "_cli", config, "/quit")
        assert should_exit is True

    @pytest.mark.asyncio
    async def test_help_does_not_exit(self, tmp_path):
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        sl = SessionLog(tmp_path, "@test:server")
        response, should_exit = await _process_cli_line(
            agent, sl, {}, "_cli", config, "/help")
        assert should_exit is False

    @pytest.mark.asyncio
    async def test_regular_message_logs_user_entry(self, tmp_path):
        """A regular message logs a user entry to SessionLog before processing."""
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        agent.handle_input = AsyncMock(return_value="test response")
        sl = SessionLog(tmp_path, "@test:server")
        await _process_cli_line(
            agent, sl, {}, "_cli", config, "hello agent")
        entries = sl.read("_cli")
        user_entries = [e for e in entries if e["role"] == "user"]
        assert len(user_entries) >= 1
        assert user_entries[0]["content"] == "hello agent"

    @pytest.mark.asyncio
    async def test_regular_message_logs_assistant_entry(self, tmp_path):
        """After handle_input, the assistant response is logged to SessionLog."""
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        agent.handle_input = AsyncMock(return_value="test response")
        sl = SessionLog(tmp_path, "@test:server")
        await _process_cli_line(
            agent, sl, {}, "_cli", config, "hello agent")
        entries = sl.read("_cli")
        asst_entries = [e for e in entries if e["role"] == "assistant"]
        assert len(asst_entries) == 1
        assert asst_entries[0]["content"] == "test response"

    @pytest.mark.asyncio
    async def test_regular_message_passes_callbacks_to_handle_input(self, tmp_path):
        """handle_input receives the callbacks dict (not None) — the keystone payoff."""
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        agent.handle_input = AsyncMock(return_value="ok")
        sl = SessionLog(tmp_path, "@test:server")
        callbacks = {"room_id": "_cli", "context_status": AsyncMock(return_value={})}
        await _process_cli_line(
            agent, sl, callbacks, "_cli", config, "hello")
        # handle_input was called with callbacks kwarg
        _, kwargs = agent.handle_input.call_args
        assert kwargs.get("callbacks") is not None
        assert kwargs["callbacks"] is callbacks or "room_id" in (kwargs.get("callbacks") or {})

    @pytest.mark.asyncio
    async def test_status_command_shows_full_stack(self, tmp_path, capsys):
        """/status shows the full status stack (model, context, tokens)."""
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        agent.status = MagicMock(return_value={
            "name": "test-agent", "model": "anthropic/claude-sonnet-4",
            "turns": 3, "context_tokens": 5000, "context_max": 200000,
            "context_pct": 2, "context_remaining": 195000,
            "total_tool_calls": 5,
        })
        sl = SessionLog(tmp_path, "@test:server")
        response, should_exit = await _process_cli_line(
            agent, sl, {}, "_cli", config, "/status")
        assert should_exit is False
        captured = capsys.readouterr()
        # Full stack: model + context + tokens (not just the old basic 3 fields)
        assert "claude-sonnet" in captured.err or "claude-sonnet" in captured.out
        assert "context" in captured.err.lower() or "context" in captured.out.lower()


# ---------------------------------------------------------------------------
# CLI tool logging + round-trip (JSONL shapes match Matrix)
# ---------------------------------------------------------------------------

class TestCLIToolLoggingRoundTrip:
    """Tool results + assistant turns logged in the same JSONL shape as Matrix,
    so build_context reconstructs them identically.
    """

    @pytest.mark.asyncio
    async def test_tool_result_logged_correctly(self, tmp_path):
        """A tool result is logged with role=tool, call_id, name, output, is_error."""
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        # Simulate handle_input calling on_tool_call
        async def fake_handle_input(text, room_id, *, on_tool_call=None, on_tool_intent=None,
                                     callbacks=None, **kw):
            if on_tool_call:
                await on_tool_call("call_1", "shell", {"command": "echo hi"},
                                   "hi\n", is_error=False)
            return "done"
        agent.handle_input = fake_handle_input
        sl = SessionLog(tmp_path, "@test:server")
        await _process_cli_line(agent, sl, {}, "_cli", config, "run echo")
        entries = sl.read("_cli")
        tool_entries = [e for e in entries if e["role"] == "tool"]
        assert len(tool_entries) == 1
        assert tool_entries[0]["call_id"] == "call_1"
        assert tool_entries[0]["name"] == "shell"
        assert tool_entries[0]["output"] == "hi\n"
        assert tool_entries[0]["is_error"] is False

    @pytest.mark.asyncio
    async def test_round_trip_build_context_reconstructs(self, tmp_path):
        """A CLI-produced JSONL reconstructs via build_context identically to
        the in-memory history the agent would have."""
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        # Simulate a tool-using turn
        async def fake_handle_input(text, room_id, *, on_tool_call=None, on_tool_intent=None,
                                     callbacks=None, **kw):
            if on_tool_intent:
                from openalph.provider import ToolCall
                await on_tool_intent([ToolCall(id="c1", name="shell",
                                                input={"command": "ls"})], "")
            if on_tool_call:
                await on_tool_call("c1", "shell", {"command": "ls"}, "file.txt\n", False)
            return "I found file.txt"
        agent.handle_input = fake_handle_input
        sl = SessionLog(tmp_path, "@test:server")
        # Process a message
        await _process_cli_line(agent, sl, {}, "_cli", config, "list files")

        # Now build_context from the JSONL and verify it reconstructs
        context = sl.build_context("_cli")
        roles = [e["role"] for e in context]
        assert "user" in roles
        assert "assistant" in roles
        # The assistant entry should have tool_calls
        asst = next(e for e in context if e["role"] == "assistant" and e.get("tool_calls"))
        # build_context returns ToolCall dataclass objects (attribute access, not dict)
        assert asst["tool_calls"][0].name == "shell"
        # The tool result is mapped to wire format: {role: tool, tool_call_id, content}
        tool = next(e for e in context if e["role"] == "tool")
        assert tool["tool_call_id"] == "c1"
        assert "file.txt" in tool["content"]
