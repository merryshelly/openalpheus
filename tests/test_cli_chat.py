"""Tests for openalph chat CLI command (Phase 5.1).

TDD test suite — written before implementation. Tests verify the chat
subcommand, interactive loop, session management, and error handling.

cmd_chat() is a sync function that internally runs asyncio.run() over an
interactive loop. Tests mock builtins.input() to simulate user input and
redirect stdout/stderr to capture output routing.
"""

import asyncio
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_args(agent="test-agent", room=None):
    """Create argparse-like Namespace for chat command."""
    ns = MagicMock()
    ns.agent = agent
    ns.room = room
    ns.command = "chat"
    ns.verbose = False
    return ns


def run_chat(inputs, env, args=None):
    """Run cmd_chat with mocked inputs. Returns (stdout_str, stderr_str).

    ``inputs`` is a list of strings (returned by input()) and/or exception
    classes (raised by input()). EOFError is appended automatically if the
    sequence doesn't already end with an exception class.
    """
    from openalph.cli import cmd_chat

    if args is None:
        args = make_args()

    effects = list(inputs)
    if not effects or not (isinstance(effects[-1], type) and issubclass(effects[-1], BaseException)):
        effects.append(EOFError)

    out, err = StringIO(), StringIO()
    with (
        patch("builtins.input", side_effect=effects),
        redirect_stdout(out),
        redirect_stderr(err),
    ):
        cmd_chat(args)

    return out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_config():
    """Minimal AgentConfig mock — no [matrix] section."""
    config = MagicMock()
    config.name = "test-agent"
    config.model = "claude-haiku-4-5"
    config.workspace = Path("/tmp/test-workspace")
    config.matrix = None
    config.model_max_tokens = 200000
    return config


@pytest.fixture
def mock_agent():
    """Mock Agent instance with async handle_input."""
    agent = MagicMock()
    agent.handle_input = AsyncMock(return_value="Hello from the agent!")
    agent.history.return_value = MagicMock()  # must be MagicMock so .clear() and .extend() are trackable
    agent.status.return_value = {
        "name": "test-agent",
        "model": "claude-haiku-4-5",
        "turns": 2,
        "context_tokens": 1200,
        "context_max": 200000,
        "context_pct": 1,
        "uncached_input_tokens": 500,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "total_output_tokens": 700,
        "total_tool_calls": 3,
    }
    return agent


@pytest.fixture
def chat_env(fake_config, mock_agent):
    """Patch all imports used by cmd_chat. Yields env dict."""
    with (
        patch("openalph.cli.load_agent_config", return_value=fake_config),
        patch("openalph.agent.Agent", return_value=mock_agent) as agent_cls,
    ):
        yield {
            "config": fake_config,
            "agent": mock_agent,
            "agent_cls": agent_cls,
        }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestChatParseArgs:
    def test_basic(self):
        from openalph.cli import parse_args

        args = parse_args(["chat", "watson"])
        assert args.command == "chat"
        assert args.agent == "watson"

    def test_room_flag(self):
        from openalph.cli import parse_args

        args = parse_args(["chat", "watson", "--room", "debug-session"])
        assert args.room == "debug-session"

    def test_room_default_none(self):
        from openalph.cli import parse_args

        args = parse_args(["chat", "watson"])
        assert args.room is None


# ---------------------------------------------------------------------------
# Exit paths
# ---------------------------------------------------------------------------


class TestChatExits:
    def test_quit_command(self, chat_env):
        stdout, stderr = run_chat(["/quit"], chat_env)
        assert "Goodbye" in stderr
        chat_env["agent"].handle_input.assert_not_called()

    def test_exit_command(self, chat_env):
        stdout, stderr = run_chat(["/exit"], chat_env)
        assert "Goodbye" in stderr
        chat_env["agent"].handle_input.assert_not_called()

    def test_eof_exits(self, chat_env):
        stdout, stderr = run_chat([EOFError], chat_env)
        assert "Goodbye" in stderr

    def test_keyboard_interrupt_exits(self, chat_env):
        stdout, stderr = run_chat([KeyboardInterrupt], chat_env)
        assert "Goodbye" in stderr


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


class TestChatCommands:
    def test_help_on_stderr(self, chat_env):
        stdout, stderr = run_chat(["/help"], chat_env)
        assert "/status" in stderr
        assert "/quit" in stderr
        assert "--room" in stderr  # help mentions --room for session isolation
        # Help should NOT call handle_input
        chat_env["agent"].handle_input.assert_not_called()

    def test_status_on_stderr(self, chat_env):
        stdout, stderr = run_chat(["/status"], chat_env)
        assert "test-agent" in stderr
        assert "claude-haiku-4-5" in stderr
        assert "1,200" in stderr  # context_tokens formatted with commas
        chat_env["agent"].handle_input.assert_not_called()

    def test_empty_input_skipped(self, chat_env):
        stdout, stderr = run_chat(["", "   ", ""], chat_env)
        chat_env["agent"].handle_input.assert_not_called()


# ---------------------------------------------------------------------------
# Message handling
# ---------------------------------------------------------------------------


class TestChatMessages:
    def test_sends_to_agent(self, chat_env):
        run_chat(["hello world"], chat_env)
        chat_env["agent"].handle_input.assert_called_once(); assert chat_env["agent"].handle_input.call_args[0] == ("hello world", "_cli")

    def test_response_on_stdout(self, chat_env):
        stdout, stderr = run_chat(["hello"], chat_env)
        assert "Hello from the agent!" in stdout

    def test_response_not_on_stderr(self, chat_env):
        stdout, stderr = run_chat(["hello"], chat_env)
        assert "Hello from the agent!" not in stderr

    def test_tool_notice_on_stderr(self, chat_env):
        """Tool notices go to stderr, not stdout."""
        agent = chat_env["agent"]

        async def handle_with_tool(text, room_id, *, on_tool_call=None, on_tool_intent=None):
            if on_tool_call:
                await on_tool_call(
                    "call-1", "shell", {"command": "ls"}, "/tmp\n/var", False
                )
            return "Here are the files."

        agent.handle_input = AsyncMock(side_effect=handle_with_tool)

        stdout, stderr = run_chat(["list files"], chat_env)
        assert "🔧" in stderr
        assert "shell" in stderr
        assert "ok" in stderr
        # Response still on stdout
        assert "Here are the files." in stdout

    def test_tool_error_notice(self, chat_env):
        """Tool errors show 'error' status in notice."""
        agent = chat_env["agent"]

        async def handle_with_error(text, room_id, *, on_tool_call=None, on_tool_intent=None):
            if on_tool_call:
                await on_tool_call(
                    "call-2", "shell", {"command": "bad"}, "command not found", True
                )
            return "That command failed."

        agent.handle_input = AsyncMock(side_effect=handle_with_error)

        stdout, stderr = run_chat(["run bad"], chat_env)
        assert "error" in stderr
        assert "shell" in stderr

    def test_multiple_turns(self, chat_env):
        """Loop continues after handling a message — multiple turns work."""
        agent = chat_env["agent"]
        agent.handle_input = AsyncMock(side_effect=["First reply", "Second reply"])

        stdout, stderr = run_chat(["hello", "world"], chat_env)
        assert agent.handle_input.call_count == 2
        assert "First reply" in stdout
        assert "Second reply" in stdout


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class TestChatSession:
    def test_custom_room_id(self, chat_env):
        args = make_args(room="debug-room")
        run_chat(["hello"], chat_env, args=args)
        chat_env["agent"].handle_input.assert_called_once(); assert chat_env["agent"].handle_input.call_args[0] == ("hello", "debug-room")

    def test_default_room_id(self, chat_env):
        run_chat(["hello"], chat_env)
        chat_env["agent"].handle_input.assert_called_once(); assert chat_env["agent"].handle_input.call_args[0] == ("hello", "_cli")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestChatErrors:
    def test_context_overflow_handled(self, chat_env):
        """ContextOverflowError prints warning to stderr, doesn't crash."""
        from openalph.agent import ContextOverflowError

        chat_env["agent"].handle_input = AsyncMock(
            side_effect=ContextOverflowError(150000, 200000)
        )

        stdout, stderr = run_chat(["hello"], chat_env)
        assert "overflow" in stderr.lower()
        assert "150,000" in stderr
        assert "--room" in stderr

    def test_generic_exception_handled(self, chat_env):
        """Unexpected errors print to stderr, don't crash the loop."""
        chat_env["agent"].handle_input = AsyncMock(
            side_effect=RuntimeError("something broke")
        )

        stdout, stderr = run_chat(["hello"], chat_env)
        assert "something broke" in stderr

    def test_config_error_exits(self):
        """ConfigError prints message and exits with code 1."""
        from openalph.cli import cmd_chat
        from openalph.config import ConfigError

        err = StringIO()
        with (
            patch("openalph.cli.load_agent_config", side_effect=ConfigError("missing field")),
            redirect_stderr(err),
            pytest.raises(SystemExit) as exc_info,
        ):
            cmd_chat(make_args())

        assert exc_info.value.code == 1
        assert "Config error" in err.getvalue()
        assert "missing field" in err.getvalue()


# ---------------------------------------------------------------------------
# Config / construction
# ---------------------------------------------------------------------------


class TestChatConfig:
    def test_no_matrix_required(self, chat_env):
        """Chat works without [matrix] config section."""
        assert chat_env["config"].matrix is None
        stdout, stderr = run_chat(["/quit"], chat_env)
        assert "Goodbye" in stderr

    def test_agent_constructed_with_config(self, chat_env):
        """Agent class instantiated with the loaded config."""
        run_chat(["/quit"], chat_env)
        chat_env["agent_cls"].assert_called_once_with(chat_env["config"])

    def test_startup_banner_on_stderr(self, chat_env):
        """Startup banner with agent name and model goes to stderr."""
        stdout, stderr = run_chat(["/quit"], chat_env)
        assert "test-agent" in stderr
        assert "ready" in stderr.lower()
