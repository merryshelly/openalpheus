"""Tests for shell tool cwd workspace default (.126).

Shell tool should default cwd to config.workspace when no explicit cwd
is provided. This ensures relative paths resolve identically across
shell and file tools.
"""

import pytest

from openalph.tools import execute_tool
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


class TestShellCwdDefault:
    """Shell tool defaults cwd to workspace when not specified."""

    @pytest.mark.asyncio
    async def test_shell_uses_workspace_as_default_cwd(self, tmp_path):
        """When cwd is not provided, shell runs in config.workspace."""
        config = make_config(workspace=tmp_path)
        result = await execute_tool(
            name="shell",
            input={"command": "pwd"},
            tool_config={},
            agent_config=config,
        )
        assert result.is_error is False
        assert result.content.strip() == str(tmp_path)

    @pytest.mark.asyncio
    async def test_explicit_cwd_overrides_workspace(self, tmp_path):
        """When cwd is explicitly provided, it takes precedence."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        config = make_config(workspace=workspace)
        result = await execute_tool(
            name="shell",
            input={"command": "pwd", "cwd": str(other_dir)},
            tool_config={},
            agent_config=config,
        )
        assert result.is_error is False
        assert result.content.strip() == str(other_dir)

    @pytest.mark.asyncio
    async def test_relative_paths_match_file_tools(self, tmp_path):
        """Relative path in shell ls matches what file_read would resolve."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "test.txt").write_text("hello")

        config = make_config(workspace=workspace)

        # Shell should see the file via relative path
        result = await execute_tool(
            name="shell",
            input={"command": "cat test.txt"},
            tool_config={},
            agent_config=config,
        )
        assert result.is_error is False
        assert "hello" in result.content

    @pytest.mark.asyncio
    async def test_ls_relative_matches_file_tools(self, tmp_path):
        """Shell ls in workspace sees the same files as file_read would."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "a.txt").write_text("aaa")
        (workspace / "b.txt").write_text("bbb")

        config = make_config(workspace=workspace)

        result = await execute_tool(
            name="shell",
            input={"command": "ls *.txt | sort"},
            tool_config={},
            agent_config=config,
        )
        assert result.is_error is False
        assert "a.txt" in result.content
        assert "b.txt" in result.content

    @pytest.mark.asyncio
    async def test_no_workspace_attr_falls_through(self):
        """If agent_config has no workspace attr, cwd is None (process default)."""
        # Use a simple object without workspace attribute
        class MinimalConfig:
            pass

        config = MinimalConfig()
        result = await execute_tool(
            name="shell",
            input={"command": "echo ok"},
            tool_config={},
            agent_config=config,
        )
        assert result.is_error is False
        assert "ok" in result.content
