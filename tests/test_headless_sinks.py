"""Tests for Phase 1 Track A pure helpers: HeadlessSinks + persist_assistant_turn hoist.

Design spec: memory/projects/openalph/specs/cli-firstclass-and-matrix-decoupling.md
Bead: workspace-kdsn.237 (Phase 1) + workspace-kdsn.240 (persistence)

RED until:
  - openalph.session.persist_assistant_turn exists (hoisted from MatrixBot)
  - openalph.callbacks.HeadlessSinks exists
  - MatrixBot._persist_assistant_turn delegates to the shared function
"""

import pytest
from unittest.mock import MagicMock, patch

from openalph.session import SessionLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_config(workspace):
    from openalph.config import AgentConfig, ProviderConfig
    return AgentConfig(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic",
            api_key="sk-test", base_url=None, quirks=None,
        )},
        workspace=workspace,
    )


def _make_agent(**overrides):
    agent = MagicMock()
    agent.system_prompt = "test prompt"
    agent.history = MagicMock(return_value=[{"role": "user", "content": "hi"}])
    agent.last_turn_usage = MagicMock(return_value={
        "input_tokens": 100, "output_tokens": 50,
        "cache_read_tokens": 0, "cache_creation_tokens": 0,
    })
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


# ---------------------------------------------------------------------------
# persist_assistant_turn hoist (session.py)
# ---------------------------------------------------------------------------

class TestPersistAssistantTurnHoist:
    """persist_assistant_turn is hoisted from MatrixBot to session.py.

    It's comms-agnostic persistence logic, not callback assembly. MatrixBot
    delegates to it (same pattern as Phase 0).
    """

    def test_importable_from_session(self):
        from openalph.session import persist_assistant_turn
        assert callable(persist_assistant_turn)

    def test_writes_assistant_entry_with_content(self, tmp_path):
        """Writes a JSONL assistant entry with content + sender + room."""
        from openalph.session import persist_assistant_turn
        sl = SessionLog(tmp_path, "@bot:server")
        agent = _make_agent()
        persist_assistant_turn(agent, sl, "!room:server", content="Hello world")
        entries = sl.read("!room:server")
        assert len(entries) == 1
        assert entries[0]["role"] == "assistant"
        assert entries[0]["content"] == "Hello world"
        assert entries[0]["sender"] == "@bot:server"
        assert entries[0]["room"] == "!room:server"

    def test_captures_thinking_from_history(self, tmp_path):
        """Thinking is read from agent.history(room_id)[-1] (RC1 invariant)."""
        from openalph.session import persist_assistant_turn
        sl = SessionLog(tmp_path, "@bot:server")
        agent = _make_agent()
        agent.history = MagicMock(return_value=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "resp", "thinking": "deep thoughts"},
        ])
        persist_assistant_turn(agent, sl, "!room:server", content="resp")
        entries = sl.read("!room:server")
        assert entries[0]["thinking"] == "deep thoughts"

    def test_captures_usage_from_last_turn_usage(self, tmp_path):
        """Usage dict is read from agent.last_turn_usage(room_id)."""
        from openalph.session import persist_assistant_turn
        sl = SessionLog(tmp_path, "@bot:server")
        agent = _make_agent()
        persist_assistant_turn(agent, sl, "!room:server", content="resp")
        entries = sl.read("!room:server")
        assert entries[0]["usage"]["input_tokens"] == 100
        assert entries[0]["usage"]["output_tokens"] == 50
        assert entries[0]["usage"]["tool_calls"] == 0  # no tool_calls passed

    def test_tool_calls_logged(self, tmp_path):
        """tool_calls are logged as structured entries with call_id/name/input."""
        from openalph.session import persist_assistant_turn
        from openalph.provider import ToolCall
        sl = SessionLog(tmp_path, "@bot:server")
        agent = _make_agent()
        tc = ToolCall(id="call_1", name="shell", input={"command": "ls"})
        persist_assistant_turn(agent, sl, "!room:server", content="", tool_calls=[tc])
        entries = sl.read("!room:server")
        assert entries[0]["tool_calls"] == [
            {"call_id": "call_1", "name": "shell", "input": {"command": "ls"}}
        ]
        assert entries[0]["usage"]["tool_calls"] == 1

    def test_no_thinking_when_history_empty(self, tmp_path):
        """No thinking field when history[-1] is not assistant or has no thinking."""
        from openalph.session import persist_assistant_turn
        sl = SessionLog(tmp_path, "@bot:server")
        agent = _make_agent()
        agent.history = MagicMock(return_value=[])
        persist_assistant_turn(agent, sl, "!room:server", content="resp")
        entries = sl.read("!room:server")
        assert "thinking" not in entries[0] or entries[0]["thinking"] is None

    def test_matrixbot_delegates_to_shared(self):
        """MatrixBot._persist_assistant_turn delegates to session.persist_assistant_turn."""
        from openalph.session import persist_assistant_turn
        from openalph.matrix import MatrixBot
        bot = MatrixBot.__new__(MatrixBot)
        bot.agent = _make_agent()
        bot.config = MagicMock()
        bot.config.user_id = "@bot:server"
        bot.session_log = MagicMock()
        with patch("openalph.matrix.persist_assistant_turn", wraps=persist_assistant_turn) as mock:
            bot._persist_assistant_turn("!room:server", content="test")
            assert mock.called


# ---------------------------------------------------------------------------
# HeadlessSinks (callbacks.py)
# ---------------------------------------------------------------------------

class TestHeadlessSinks:
    """HeadlessSinks implements CommsSinks for CLI/headless mode.

    Notices → stderr. send_media → raises (clear error). log_reminder → SessionLog.
    """

    def test_importable_from_callbacks(self):
        from openalph.callbacks import HeadlessSinks
        assert HeadlessSinks is not None

    def test_satisfies_comms_sinks_protocol(self):
        """HeadlessSinks has all 7 CommsSinks methods (7th: log_vision_injection, kdsn.279)."""
        from openalph.callbacks import HeadlessSinks
        expected = {"send_notice", "log_reminder", "send_media",
                    "on_redaction", "on_keepalive_miss", "on_degenerate",
                    "log_vision_injection"}
        for method in expected:
            assert hasattr(HeadlessSinks, method), f"HeadlessSinks missing {method}"

    @pytest.mark.asyncio
    async def test_send_notice_prints_to_stderr(self, capsys):
        from openalph.callbacks import HeadlessSinks
        sinks = HeadlessSinks()
        await sinks.send_notice("!room:server", "test notice body")
        captured = capsys.readouterr()
        assert "test notice body" in captured.err
        assert captured.out == ""  # nothing on stdout

    @pytest.mark.asyncio
    async def test_send_media_raises_clear_error(self):
        """send_media raises — the tool catches it and returns is_error."""
        from openalph.callbacks import HeadlessSinks
        sinks = HeadlessSinks()
        with pytest.raises(Exception, match="(?i)no delivery|not available|cli"):
            await sinks.send_media("/tmp/file.txt", "text/plain", "file.txt")

    @pytest.mark.asyncio
    async def test_log_reminder_writes_to_session_log(self, tmp_path):
        """log_reminder appends to SessionLog with source='reminder'."""
        from openalph.callbacks import HeadlessSinks
        sl = SessionLog(tmp_path, "@bot:server")
        sinks = HeadlessSinks(session_log=sl, agent_user_id="@bot:server")
        reminder = MagicMock()
        reminder.content = "test reminder"
        reminder.trigger = "todo-nudge"
        await sinks.log_reminder("!room:server", reminder)
        entries = sl.read("!room:server")
        assert len(entries) == 1
        assert entries[0]["role"] == "user"
        assert entries[0]["source"] == "reminder"
        assert entries[0]["trigger"] == "todo-nudge"
        assert entries[0]["content"] == "test reminder"

    @pytest.mark.asyncio
    async def test_log_reminder_no_session_log_does_not_crash(self, capsys):
        """log_reminder without a session_log just prints to stderr (no crash)."""
        from openalph.callbacks import HeadlessSinks
        sinks = HeadlessSinks()
        reminder = MagicMock()
        reminder.content = "reminder text"
        reminder.trigger = "test"
        await sinks.log_reminder("!room:server", reminder)
        captured = capsys.readouterr()
        assert "reminder text" in captured.err

    @pytest.mark.asyncio
    async def test_log_vision_injection_writes_to_session_log(self, tmp_path, capsys):
        """kdsn.279: log_vision_injection appends source='view_image' with the
        framed tag text (pre-expansion) + prints a notice to stderr."""
        from openalph.callbacks import HeadlessSinks
        sl = SessionLog(tmp_path, "@bot:server")
        sinks = HeadlessSinks(session_log=sl, agent_user_id="@bot:server")
        framed = ("[view_image tool output — 1 image(s): a.jpg]\n"
                  "[media: a.jpg (image/jpeg, 104 B)]")
        await sinks.log_vision_injection("!room:server", framed)
        entries = sl.read("!room:server")
        assert len(entries) == 1
        assert entries[0]["role"] == "user"
        assert entries[0]["source"] == "view_image"
        assert entries[0]["content"] == framed
        captured = capsys.readouterr()
        assert "view_image" in captured.err

    @pytest.mark.asyncio
    async def test_log_vision_injection_no_session_log_does_not_crash(self, capsys):
        """log_vision_injection without a session_log just prints (no crash)."""
        from openalph.callbacks import HeadlessSinks
        sinks = HeadlessSinks()
        await sinks.log_vision_injection("_cli", "[view_image tool output — 1 image(s): a.jpg]")
        captured = capsys.readouterr()
        assert "view_image" in captured.err

    @pytest.mark.asyncio
    async def test_on_redaction_prints_to_stderr(self, capsys):
        from openalph.callbacks import HeadlessSinks
        sinks = HeadlessSinks()
        event = MagicMock()
        event.pattern_name = "api_key"
        event.char_count = 40
        await sinks.on_redaction("shell", [event])
        captured = capsys.readouterr()
        assert "redact" in captured.err.lower() or "api_key" in captured.err

    @pytest.mark.asyncio
    async def test_on_keepalive_miss_prints_to_stderr(self, capsys):
        from openalph.callbacks import HeadlessSinks
        sinks = HeadlessSinks()
        await sinks.on_keepalive_miss("!room:server")
        captured = capsys.readouterr()
        assert "keepalive" in captured.err.lower() or "cache" in captured.err.lower()

    @pytest.mark.asyncio
    async def test_on_degenerate_prints_to_stderr(self, capsys):
        from openalph.callbacks import HeadlessSinks
        sinks = HeadlessSinks()
        await sinks.on_degenerate(model="glm-5p2", generation_id="gen123")
        captured = capsys.readouterr()
        assert "degen" in captured.err.lower() or "glm" in captured.err.lower()

    @pytest.mark.asyncio
    async def test_all_sinks_use_stderr_not_stdout(self, capsys):
        """Critical: sinks must not corrupt stdout (agent responses go there)."""
        from openalph.callbacks import HeadlessSinks
        sinks = HeadlessSinks()
        await sinks.send_notice("!r:s", "notice")
        event = MagicMock()
        event.pattern_name = "x"
        event.char_count = 1
        await sinks.on_redaction("tool", [event])
        await sinks.on_keepalive_miss()
        await sinks.on_degenerate(model="m")
        captured = capsys.readouterr()
        assert captured.out == ""  # NOTHING on stdout from any sink
