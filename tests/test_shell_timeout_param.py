"""Tests for shell tool timeout parameter fix.

The shell tool dispatch in execute_tool must honour the agent-requested
`timeout` from `input` when provided, and fall back to the tool config
`default_timeout` when it is not.

Regression test for the bug where:
    timeout=tool_config.get("default_timeout", 30)
was used unconditionally, ignoring any `timeout` key in `input`.

Fix applied:
    timeout=input.get("timeout") or tool_config.get("default_timeout", 30)
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from openalph.tools import execute_tool, ToolResult
from openalph.config import AgentConfig, ProviderConfig


def make_provider():
    return ProviderConfig(
        key="anthropic", type="anthropic", api_key="sk-test",
        base_url=None, quirks=[],
    )


def make_config(workspace):
    return AgentConfig(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider()},
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
    )


class TestShellToolTimeout:
    """execute_tool passes the correct timeout to run_shell.

    Mocks run_shell to capture the timeout kwarg and asserts the
    right value was forwarded — no real subprocesses are launched.
    """

    @pytest.mark.asyncio
    async def test_input_timeout_overrides_config_default(self, tmp_path):
        """When input includes 'timeout', that value is passed to run_shell."""
        config = make_config(workspace=tmp_path)
        tool_config = {"default_timeout": 30, "max_output": 50000}

        captured = {}

        async def mock_run_shell(command, cwd=None, env=None, timeout=30, max_output=50000):
            captured["timeout"] = timeout
            return ToolResult(content="done")

        with patch("openalph.tools.shell.run_shell", mock_run_shell):
            await execute_tool(
                name="shell",
                input={"command": "sleep 120", "timeout": 120},
                tool_config=tool_config,
                agent_config=config,
            )

        assert captured["timeout"] == 120, (
            f"Expected timeout=120 from input, got {captured['timeout']}"
        )

    @pytest.mark.asyncio
    async def test_config_default_used_when_input_has_no_timeout(self, tmp_path):
        """When input does NOT include 'timeout', the config default is used."""
        config = make_config(workspace=tmp_path)
        tool_config = {"default_timeout": 60, "max_output": 50000}

        captured = {}

        async def mock_run_shell(command, cwd=None, env=None, timeout=30, max_output=50000):
            captured["timeout"] = timeout
            return ToolResult(content="done")

        with patch("openalph.tools.shell.run_shell", mock_run_shell):
            await execute_tool(
                name="shell",
                input={"command": "echo hello"},
                tool_config=tool_config,
                agent_config=config,
            )

        assert captured["timeout"] == 60, (
            f"Expected timeout=60 from config default, got {captured['timeout']}"
        )

    @pytest.mark.asyncio
    async def test_input_timeout_zero_falls_back_to_config(self, tmp_path):
        """A falsy timeout in input (0) falls back to the config default.

        The fix uses `or`, so timeout=0 in input is treated as falsy and
        the config default is used instead. This is consistent with the
        intent of the fix (0 is not a useful timeout).
        """
        config = make_config(workspace=tmp_path)
        tool_config = {"default_timeout": 45, "max_output": 50000}

        captured = {}

        async def mock_run_shell(command, cwd=None, env=None, timeout=30, max_output=50000):
            captured["timeout"] = timeout
            return ToolResult(content="done")

        with patch("openalph.tools.shell.run_shell", mock_run_shell):
            await execute_tool(
                name="shell",
                input={"command": "echo hi", "timeout": 0},
                tool_config=tool_config,
                agent_config=config,
            )

        # timeout=0 is falsy → falls back to config default
        assert captured["timeout"] == 45, (
            f"Expected timeout=45 (config default) when input timeout=0, "
            f"got {captured['timeout']}"
        )

    @pytest.mark.asyncio
    async def test_input_timeout_none_falls_back_to_config(self, tmp_path):
        """An explicit None timeout in input falls back to the config default."""
        config = make_config(workspace=tmp_path)
        tool_config = {"default_timeout": 90, "max_output": 50000}

        captured = {}

        async def mock_run_shell(command, cwd=None, env=None, timeout=30, max_output=50000):
            captured["timeout"] = timeout
            return ToolResult(content="done")

        with patch("openalph.tools.shell.run_shell", mock_run_shell):
            await execute_tool(
                name="shell",
                input={"command": "echo hi", "timeout": None},
                tool_config=tool_config,
                agent_config=config,
            )

        # timeout=None is falsy → falls back to config default
        assert captured["timeout"] == 90, (
            f"Expected timeout=90 (config default) when input timeout=None, "
            f"got {captured['timeout']}"
        )

    @pytest.mark.asyncio
    async def test_hardcoded_fallback_when_no_config_default(self, tmp_path):
        """Falls back to hardcoded 30s when config has no default_timeout."""
        config = make_config(workspace=tmp_path)
        tool_config = {}  # no default_timeout key

        captured = {}

        async def mock_run_shell(command, cwd=None, env=None, timeout=30, max_output=50000):
            captured["timeout"] = timeout
            return ToolResult(content="done")

        with patch("openalph.tools.shell.run_shell", mock_run_shell):
            await execute_tool(
                name="shell",
                input={"command": "echo hi"},
                tool_config=tool_config,
                agent_config=config,
            )

        assert captured["timeout"] == 30, (
            f"Expected hardcoded fallback timeout=30, got {captured['timeout']}"
        )
