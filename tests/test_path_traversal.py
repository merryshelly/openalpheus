"""Tests for path traversal protection on file tools."""

import pytest
from pathlib import Path

from openalph.config import AgentConfig
from openalph.tools import execute_tool


def make_agent_config(workspace: Path):
    return AgentConfig(
        name="test-agent",
        model="test-model",
        max_tokens=8192,
        provider="anthropic",
        api_key="sk-test",
        base_url=None,
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
    )


@pytest.mark.asyncio
async def test_relative_traversal_blocked(tmp_path):
    cfg = make_agent_config(tmp_path)
    result = await execute_tool(
        "file_read", {"path": "../../etc/passwd"}, {}, cfg
    )
    assert result.is_error
    assert "escapes workspace boundary" in result.content


@pytest.mark.asyncio
async def test_absolute_path_outside_workspace_blocked(tmp_path):
    cfg = make_agent_config(tmp_path)
    result = await execute_tool(
        "file_read", {"path": "/etc/passwd"}, {}, cfg
    )
    assert result.is_error
    assert "escapes workspace boundary" in result.content


@pytest.mark.asyncio
async def test_relative_path_within_workspace_allowed(tmp_path):
    cfg = make_agent_config(tmp_path)
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "test.txt").write_text("hello")
    result = await execute_tool(
        "file_read", {"path": "subdir/test.txt"}, {}, cfg
    )
    assert not result.is_error
    assert "hello" in result.content


@pytest.mark.asyncio
async def test_absolute_path_within_workspace_allowed(tmp_path):
    cfg = make_agent_config(tmp_path)
    (tmp_path / "ok.txt").write_text("fine")
    result = await execute_tool(
        "file_read", {"path": str(tmp_path / "ok.txt")}, {}, cfg
    )
    assert not result.is_error
    assert "fine" in result.content


@pytest.mark.asyncio
async def test_write_traversal_blocked(tmp_path):
    cfg = make_agent_config(tmp_path)
    target = tmp_path.parent / "should_not_exist.txt"
    result = await execute_tool(
        "file_write",
        {"path": "../should_not_exist.txt", "content": "pwned"},
        {},
        cfg,
    )
    assert result.is_error
    assert "escapes workspace boundary" in result.content
    assert not target.exists()


@pytest.mark.asyncio
async def test_edit_traversal_blocked(tmp_path):
    cfg = make_agent_config(tmp_path)
    result = await execute_tool(
        "file_edit",
        {"path": "/etc/hosts", "old_text": "localhost", "new_text": "hacked"},
        {},
        cfg,
    )
    assert result.is_error
    assert "escapes workspace boundary" in result.content
