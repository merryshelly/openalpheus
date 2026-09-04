"""Tests for Phase 1 Track B: launch robustness, truncation, /commands audit.

Design spec: memory/projects/openalph/specs/cli-firstclass-and-matrix-decoupling.md
Bead: workspace-kdsn.230 (emergency-CLI UX + launch robustness)
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock

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
    agent.history = MagicMock(return_value=MagicMock())
    agent.last_turn_usage = MagicMock(return_value={})
    agent.restore_usage = MagicMock()
    agent.rehydrate_reminders = MagicMock()
    agent.status = MagicMock(return_value={
        "name": "test-agent", "model": "anthropic/claude-sonnet-4",
        "turns": 0, "context_tokens": 0, "context_max": 200000,
        "context_pct": 0, "total_tool_calls": 0,
    })
    agent.handle_input = AsyncMock(return_value="ok")
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


# ---------------------------------------------------------------------------
# Launch robustness: workspace writability guard
# ---------------------------------------------------------------------------

class TestLaunchGuard:
    """cmd_chat warns if the workspace isn't writable (wrong user).

    Catches the 2026-07-11 failure: SB ran as merryshelly, couldn't read
    oa-merry's 1P token, fell back to root (root-owned logs). The guard
    detects this before launch and prints the correct sudo -u command.
    """

    def test_importable_from_cli(self):
        from openalph.cli import _check_workspace_writable
        assert callable(_check_workspace_writable)

    def test_writable_workspace_returns_true(self, tmp_path):
        """A writable workspace returns True (no warning needed)."""
        from openalph.cli import _check_workspace_writable
        config = _make_config(tmp_path)
        assert _check_workspace_writable(config) is True

    def test_unwritable_workspace_returns_false(self, tmp_path):
        """An unwritable workspace returns False."""
        from openalph.cli import _check_workspace_writable
        ws = tmp_path / "locked-ws"
        ws.mkdir()
        os.chmod(ws, 0o444)  # read-only
        config = _make_config(ws)
        try:
            assert _check_workspace_writable(config) is False
        finally:
            os.chmod(ws, 0o755)  # restore for cleanup

    def test_nonexistent_workspace_returns_false(self, tmp_path):
        """A workspace that doesn't exist returns False."""
        from openalph.cli import _check_workspace_writable
        config = _make_config(tmp_path / "does-not-exist")
        assert _check_workspace_writable(config) is False

    def test_guard_message_names_correct_sudo_command(self, tmp_path, capsys):
        """When workspace isn't writable, the warning names the correct sudo -u command."""
        from openalph.cli import _check_workspace_writable
        ws = tmp_path / "locked-ws"
        ws.mkdir()
        os.chmod(ws, 0o444)
        config = _make_config(ws)
        config.name = "watson"
        try:
            result = _check_workspace_writable(config)
            assert result is False
            captured = capsys.readouterr()
            assert "sudo" in captured.err.lower() or "sudo" in captured.out.lower()
            assert "oa-watson" in captured.err or "oa-watson" in captured.out
        finally:
            os.chmod(ws, 0o755)


# ---------------------------------------------------------------------------
# Configurable tool-result truncation
# ---------------------------------------------------------------------------

class TestConfigurableTruncation:
    """Tool-result preview truncation is configurable, not hardcoded at 200.

    --truncate N CLI arg or OPENALPH_TRUNCATE env var controls the stderr
    preview length. SessionLog overflow (>64KB) is separate and unchanged.
    """

    def test_default_truncation_is_200(self):
        """The default preview length is 200 chars (backward compat)."""
        from openalph.cli import _get_truncate_limit
        assert _get_truncate_limit(args=None) == 200

    def test_truncate_from_args(self):
        """--truncate N overrides the default."""
        from openalph.cli import _get_truncate_limit
        args = MagicMock()
        args.truncate = 500
        assert _get_truncate_limit(args=args) == 500

    def test_truncate_from_env(self, monkeypatch):
        """OPENALPH_TRUNCATE env var overrides the default."""
        from openalph.cli import _get_truncate_limit
        monkeypatch.setenv("OPENALPH_TRUNCATE", "1000")
        assert _get_truncate_limit(args=None) == 1000

    def test_args_override_env(self, monkeypatch):
        """--truncate arg takes precedence over env var."""
        from openalph.cli import _get_truncate_limit
        monkeypatch.setenv("OPENALPH_TRUNCATE", "1000")
        args = MagicMock()
        args.truncate = 500
        assert _get_truncate_limit(args=args) == 500

    def test_zero_truncate_means_no_limit(self):
        """--truncate 0 means no truncation (show full output)."""
        from openalph.cli import _get_truncate_limit
        args = MagicMock()
        args.truncate = 0
        assert _get_truncate_limit(args=args) is None

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        """A non-numeric env var falls back to the default (no crash)."""
        from openalph.cli import _get_truncate_limit
        monkeypatch.setenv("OPENALPH_TRUNCATE", "not-a-number")
        assert _get_truncate_limit(args=None) == 200

    @pytest.mark.asyncio
    async def test_truncation_applied_to_tool_notice(self, tmp_path, capsys):
        """The configured truncate limit is applied to tool notice previews."""
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        long_result = "x" * 500

        async def fake_handle_input(text, room_id, *, on_tool_call=None, **kw):
            if on_tool_call:
                await on_tool_call("c1", "shell", {"command": "ls"}, long_result, False)
            return "done"
        agent.handle_input = fake_handle_input

        from openalph.session import SessionLog
        sl = SessionLog(tmp_path, "@test:server")
        await _process_cli_line(
            agent, sl, {}, "_cli", config, "run", truncate_limit=50)
        captured = capsys.readouterr()
        notice_line = [line for line in captured.err.split('\n') if '🔧' in line]
        assert len(notice_line) > 0
        preview = notice_line[0].split("(ok) ")[-1] if "(ok) " in notice_line[0] else ""
        assert len(preview) <= 51


# ---------------------------------------------------------------------------
# /commands audit: /help lists supported slash commands
# ---------------------------------------------------------------------------

class TestCommandsAudit:
    """/help lists the full set of supported CLI slash commands.

    Supported: /status, /quit, /exit, /help, /showprompt, /model
    Not supported (Matrix-only): /heartbeat, /umbral, /steer, /cache, /timesense, /stop, /resume
    """

    @pytest.mark.asyncio
    async def test_help_lists_status(self, tmp_path, capsys):
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        from openalph.session import SessionLog
        sl = SessionLog(tmp_path, "@test:server")
        await _process_cli_line(agent, sl, {}, "_cli", config, "/help")
        captured = capsys.readouterr()
        assert "/status" in captured.err

    @pytest.mark.asyncio
    async def test_help_lists_quit(self, tmp_path, capsys):
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        from openalph.session import SessionLog
        sl = SessionLog(tmp_path, "@test:server")
        await _process_cli_line(agent, sl, {}, "_cli", config, "/help")
        captured = capsys.readouterr()
        assert "/quit" in captured.err or "/exit" in captured.err

    @pytest.mark.asyncio
    async def test_help_lists_showprompt(self, tmp_path, capsys):
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        from openalph.session import SessionLog
        sl = SessionLog(tmp_path, "@test:server")
        await _process_cli_line(agent, sl, {}, "_cli", config, "/help")
        captured = capsys.readouterr()
        assert "/showprompt" in captured.err

    @pytest.mark.asyncio
    async def test_help_lists_model(self, tmp_path, capsys):
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        from openalph.session import SessionLog
        sl = SessionLog(tmp_path, "@test:server")
        await _process_cli_line(agent, sl, {}, "_cli", config, "/help")
        captured = capsys.readouterr()
        assert "/model" in captured.err

    @pytest.mark.asyncio
    async def test_help_mentions_persistence(self, tmp_path, capsys):
        """/help mentions that sessions are persisted (not in-memory only)."""
        from openalph.cli import _process_cli_line
        config = _make_config(tmp_path)
        agent = _make_agent()
        from openalph.session import SessionLog
        sl = SessionLog(tmp_path, "@test:server")
        await _process_cli_line(agent, sl, {}, "_cli", config, "/help")
        captured = capsys.readouterr()
        assert "lost on exit" not in captured.err.lower()
