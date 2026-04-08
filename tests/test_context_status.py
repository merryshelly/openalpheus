"""Tests for context_status tool (.116).

Exposes agent self-monitoring data: context usage, session age, model info,
token stats, and heartbeat state. Agents use this to make context-aware
decisions about delegation, bead management, and turn planning.

Architecture:
    tools/__init__.py — BUILTIN_TOOLS entry + execute_tool dispatch
    tools/context_status.py — executor that assembles status from agent state
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from openalph.tools import ToolResult, execute_tool, BUILTIN_TOOLS


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:

    def test_context_status_in_builtin_tools(self):
        """context_status is registered as a built-in tool."""
        assert "context_status" in BUILTIN_TOOLS

    def test_no_required_parameters(self):
        """context_status takes no required parameters."""
        schema = BUILTIN_TOOLS["context_status"]["parameters"]
        assert schema.get("required", []) == []

    def test_has_room_id_parameter(self):
        """room_id is an optional parameter (framework injects it)."""
        props = BUILTIN_TOOLS["context_status"]["parameters"]["properties"]
        assert "room_id" in props


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _make_config(workspace=None):
    from openalph.config import AgentConfig, ProviderConfig
    return AgentConfig(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic",
            api_key="sk-test", base_url=None, quirks=None,
        )},
        workspace=workspace or Path("/tmp/test-workspace"),
    )


class TestContextStatusExecution:

    @pytest.mark.asyncio
    async def test_returns_tool_result(self, tmp_path):
        """Execution returns a ToolResult."""
        config = _make_config(workspace=tmp_path)
        status_data = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 5,
            "context_tokens": 10000,
            "context_max": 200000,
            "context_pct": 5,
            "total_input_tokens": 50000,
            "total_output_tokens": 15000,
            "total_tool_calls": 12,
        }
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks={"context_status": AsyncMock(return_value=status_data)},
        )
        assert isinstance(result, ToolResult)
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_returns_json(self, tmp_path):
        """Result content is valid JSON."""
        config = _make_config(workspace=tmp_path)
        status_data = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 3,
            "context_tokens": 5000,
            "context_max": 200000,
            "context_pct": 3,
            "total_input_tokens": 20000,
            "total_output_tokens": 8000,
            "total_tool_calls": 7,
        }
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks={"context_status": AsyncMock(return_value=status_data)},
        )
        data = json.loads(result.content)
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_contains_context_fields(self, tmp_path):
        """Result includes context usage fields."""
        config = _make_config(workspace=tmp_path)
        status_data = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 10,
            "context_tokens": 80000,
            "context_max": 200000,
            "context_pct": 40,
            "total_input_tokens": 100000,
            "total_output_tokens": 30000,
            "total_tool_calls": 25,
        }
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks={"context_status": AsyncMock(return_value=status_data)},
        )
        data = json.loads(result.content)
        assert "context_pct" in data
        assert "context_tokens" in data
        assert "context_max" in data
        assert data["context_pct"] == 40
        assert data["context_tokens"] == 80000

    @pytest.mark.asyncio
    async def test_contains_model_info(self, tmp_path):
        """Result includes model and agent name."""
        config = _make_config(workspace=tmp_path)
        status_data = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 1,
            "context_tokens": 1000,
            "context_max": 200000,
            "context_pct": 1,
            "total_input_tokens": 2000,
            "total_output_tokens": 500,
            "total_tool_calls": 0,
        }
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks={"context_status": AsyncMock(return_value=status_data)},
        )
        data = json.loads(result.content)
        assert data["model"] == "anthropic/claude-sonnet-4-20250514"
        assert data["name"] == "test-agent"

    @pytest.mark.asyncio
    async def test_contains_token_stats(self, tmp_path):
        """Result includes session token and tool call counts."""
        config = _make_config(workspace=tmp_path)
        status_data = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 5,
            "context_tokens": 10000,
            "context_max": 200000,
            "context_pct": 5,
            "total_input_tokens": 50000,
            "total_output_tokens": 15000,
            "total_tool_calls": 12,
        }
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks={"context_status": AsyncMock(return_value=status_data)},
        )
        data = json.loads(result.content)
        assert data["total_input_tokens"] == 50000
        assert data["total_output_tokens"] == 15000
        assert data["total_tool_calls"] == 12
        assert data["turns"] == 5

    @pytest.mark.asyncio
    async def test_contains_heartbeat_fields(self, tmp_path):
        """Result includes heartbeat state when callback provides it."""
        config = _make_config(workspace=tmp_path)
        status_data = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 5,
            "context_tokens": 10000,
            "context_max": 200000,
            "context_pct": 5,
            "total_input_tokens": 50000,
            "total_output_tokens": 15000,
            "total_tool_calls": 12,
            "heartbeat_active": True,
            "heartbeat_interval_minutes": 30,
            "heartbeat_next_minutes": 12,
        }
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks={"context_status": AsyncMock(return_value=status_data)},
        )
        data = json.loads(result.content)
        assert data["heartbeat_active"] is True
        assert data["heartbeat_interval_minutes"] == 30
        assert data["heartbeat_next_minutes"] == 12

    @pytest.mark.asyncio
    async def test_heartbeat_null_when_inactive(self, tmp_path):
        """Heartbeat fields are null when no heartbeat is active."""
        config = _make_config(workspace=tmp_path)
        status_data = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 2,
            "context_tokens": 3000,
            "context_max": 200000,
            "context_pct": 2,
            "total_input_tokens": 5000,
            "total_output_tokens": 2000,
            "total_tool_calls": 1,
            "heartbeat_active": False,
            "heartbeat_interval_minutes": None,
            "heartbeat_next_minutes": None,
        }
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks={"context_status": AsyncMock(return_value=status_data)},
        )
        data = json.loads(result.content)
        assert data["heartbeat_active"] is False
        assert data["heartbeat_interval_minutes"] is None
        assert data["heartbeat_next_minutes"] is None

    @pytest.mark.asyncio
    async def test_contains_session_age(self, tmp_path):
        """Result includes session_age_minutes."""
        config = _make_config(workspace=tmp_path)
        status_data = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 5,
            "context_tokens": 10000,
            "context_max": 200000,
            "context_pct": 5,
            "total_input_tokens": 50000,
            "total_output_tokens": 15000,
            "total_tool_calls": 12,
            "session_age_minutes": 45,
        }
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks={"context_status": AsyncMock(return_value=status_data)},
        )
        data = json.loads(result.content)
        assert data["session_age_minutes"] == 45

    @pytest.mark.asyncio
    async def test_error_when_no_callback(self, tmp_path):
        """Returns error if context_status callback is not provided."""
        config = _make_config(workspace=tmp_path)
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks={},
        )
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_error_when_no_callbacks(self, tmp_path):
        """Returns error if callbacks dict is None."""
        config = _make_config(workspace=tmp_path)
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks=None,
        )
        assert result.is_error is True
