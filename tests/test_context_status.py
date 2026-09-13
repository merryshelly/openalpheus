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
from unittest.mock import AsyncMock, MagicMock
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

    def test_no_room_id_parameter(self):
        """kdsn.342: room identity comes from the session, never the input.

        The old optional ``room_id`` param was documented "framework-
        injected, do not set manually" — but nothing injected it: the seam
        honoured ANY model-supplied id verbatim and would silently build a
        report for a NONEXISTENT room (zero counters + config-default model
        + the closure's REAL room_name — a composite Franken-report that
        read like boundary damage). The schema now carries no room selector
        at all — context_status reports the calling room, period."""
        entry = BUILTIN_TOOLS["context_status"]
        props = entry["parameters"]["properties"]
        assert "room_id" not in props, (
            "context_status must not accept a room selector — it always "
            "reports the calling room")
        assert "calling room" in entry["description"], (
            "the description must state the calling-room scope")


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


class TestCallingRoomOnly:
    """kdsn.342 (2026-09-12): context_status is session-scoped, period.

    Real-path pins (real Agent + real SessionLog + REAL build_callbacks
    seam): the seam must never let tool input select another room, and
    session accounting must survive a handoff boundary (the boundary
    strips RENDER, never state).
    """

    ROOM = "!cs342:test"

    def _scene(self, tmp_path, n_user=3):
        """Real Agent + real SessionLog + REAL callback-construction path."""
        from openalph.agent import Agent
        from openalph.session import SessionLog
        from openalph.callbacks import build_callbacks, HeadlessSinks

        config = _make_config(workspace=tmp_path)
        agent = Agent(config)
        log = SessionLog(tmp_path, "@cs342:test")
        for i in range(n_user):
            log.append(role="user", sender="@sb:test", room=self.ROOM,
                       content=f"q{i}")
            log.append(role="assistant", sender="@cs342:test",
                       room=self.ROOM, content=f"a{i} " + "z" * 300)
        agent._room_models[self.ROOM] = "macstudio-qwen/qwen38-coder"
        cbids = build_callbacks(
            agent, self.ROOM, HeadlessSinks(), turn_source=None,
            session_log=log, room_name="cs342 room")
        return agent, log, cbids

    @pytest.mark.asyncio
    async def test_bogus_input_room_id_cannot_select_another_room(self, tmp_path):
        """The seam must never build a report for a nonexistent room — a
        fabricated id previously produced zero-counters + config-default
        model + the closure's REAL room_name (the kdsn.342 Franken-report)."""
        agent, log, cbids = self._scene(tmp_path)
        res = await execute_tool(
            name="context_status",
            input={"room_id": "!totally-fabricated:room"},
            tool_config={}, agent_config=agent.config, callbacks=cbids)
        data = json.loads(res.content)
        assert data["room_id"] == self.ROOM, (
            "model-supplied room_id silently selected another room")
        assert data["room_name"] == "cs342 room", (
            "a bogus id produced a room_name belonging to neither room")
        assert data["model"] == "macstudio-qwen/qwen38-coder", (
            "fabricated room reported the config default, not the override")
        assert data["turns"] == 3, "fabricated room zeroed the session counters"
        assert "!totally-fabricated" not in res.content

    @pytest.mark.asyncio
    async def test_truthful_after_handoff_boundary(self, tmp_path):
        """A boundary strips RENDER, never accounting: turns/age/usage/
        model must remain truthful after apply + _note."""
        agent, log, cbids = self._scene(tmp_path)
        u = agent._usage_for(self.ROOM)
        u["uncached_input_tokens"] = 50000
        u["total_output_tokens"] = 1234
        from openalph.handoff import apply_boundary_and_rebuild
        outcome = apply_boundary_and_rebuild(
            agent, log, self.ROOM, trigger="tool", exclude_inflight=False)
        assert outcome.get("applied"), "fixture boundary must apply"
        agent._note_handoff_boundary_applied(self.ROOM, outcome)
        res = await execute_tool(
            name="context_status", input={}, tool_config={},
            agent_config=agent.config, callbacks=cbids)
        data = json.loads(res.content)
        assert data["turns"] == 3, (
            f"post-boundary turns collapsed to {data['turns']} — counted "
            f"from the stripped in-memory rebuild instead of the JSONL")
        assert data["session_age_minutes"] is not None
        assert data["uncached_input_tokens"] == 50000, (
            "boundary must not zero usage counters")
        assert data["model"] == "macstudio-qwen/qwen38-coder"
        assert data["room_id"] == self.ROOM


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
            "uncached_input_tokens": 50000,
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
            "uncached_input_tokens": 20000,
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
            "uncached_input_tokens": 100000,
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
            "uncached_input_tokens": 2000,
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
            "uncached_input_tokens": 50000,
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
        assert data["uncached_input_tokens"] == 50000
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
            "uncached_input_tokens": 50000,
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
            "uncached_input_tokens": 5000,
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
            "uncached_input_tokens": 50000,
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

    # ------------------------------------------------------------------
    # New tests: room identity fields
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_contains_room_identity(self, tmp_path):
        """status_data room_id and room_name pass through to JSON result."""
        config = _make_config(workspace=tmp_path)
        status_data = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 2,
            "context_tokens": 3000,
            "context_max": 200000,
            "context_pct": 2,
            "uncached_input_tokens": 5000,
            "total_output_tokens": 2000,
            "total_tool_calls": 1,
            "room_id": "!abc123:example.com",
            "room_name": "Dev Channel",
        }
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks={"context_status": AsyncMock(return_value=status_data)},
        )
        data = json.loads(result.content)
        assert data["room_id"] == "!abc123:example.com"
        assert data["room_name"] == "Dev Channel"

    @pytest.mark.asyncio
    async def test_contains_umbral_fields(self, tmp_path):
        """status_data umbral fields pass through to JSON result when active."""
        config = _make_config(workspace=tmp_path)
        status_data = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 3,
            "context_tokens": 5000,
            "context_max": 200000,
            "context_pct": 3,
            "uncached_input_tokens": 20000,
            "total_output_tokens": 8000,
            "total_tool_calls": 7,
            "umbral_active": True,
            "umbral_interval_minutes": 240,
            "umbral_next_minutes": 180,
        }
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks={"context_status": AsyncMock(return_value=status_data)},
        )
        data = json.loads(result.content)
        assert data["umbral_active"] is True
        assert data["umbral_interval_minutes"] == 240
        assert data["umbral_next_minutes"] == 180

    @pytest.mark.asyncio
    async def test_umbral_null_when_inactive(self, tmp_path):
        """Umbral fields are null when no umbral timer is active."""
        config = _make_config(workspace=tmp_path)
        status_data = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 1,
            "context_tokens": 1000,
            "context_max": 200000,
            "context_pct": 1,
            "uncached_input_tokens": 2000,
            "total_output_tokens": 500,
            "total_tool_calls": 0,
            "umbral_active": False,
            "umbral_interval_minutes": None,
            "umbral_next_minutes": None,
        }
        result = await execute_tool(
            name="context_status",
            input={},
            tool_config={},
            agent_config=config,
            callbacks={"context_status": AsyncMock(return_value=status_data)},
        )
        data = json.loads(result.content)
        assert data["umbral_active"] is False
        assert data["umbral_interval_minutes"] is None
        assert data["umbral_next_minutes"] is None


    @pytest.mark.asyncio
    async def test_contains_cache_breakdown(self, tmp_path):
        """Result includes uncached/cache_read/cache_creation breakdown."""
        config = _make_config(workspace=tmp_path)
        status_data = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 5,
            "context_tokens": 10000,
            "context_max": 200000,
            "context_pct": 5,
            "uncached_input_tokens": 1500,
            "cache_read_tokens": 80000,
            "cache_creation_tokens": 5000,
            "total_output_tokens": 3000,
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
        assert data["uncached_input_tokens"] == 1500
        assert data["cache_read_tokens"] == 80000
        assert data["cache_creation_tokens"] == 5000


class TestBuildContextStatus:
    """Tests for MatrixBot._build_context_status — the callback body factored out."""

    def _make_bot(self):
        """Create a minimal MatrixBot-like object for testing _build_context_status."""
        from openalph.matrix import MatrixBot

        agent = MagicMock()
        agent.status.return_value = {
            "name": "test-agent",
            "model": "anthropic/claude-sonnet-4-20250514",
            "turns": 5,
            "context_tokens": 10000,
            "context_max": 200000,
            "context_pct": 5,
            "uncached_input_tokens": 50000,
            "total_output_tokens": 15000,
            "total_tool_calls": 12,
        }

        # Build a minimal bot without triggering __init__ (avoids Matrix connection)
        bot = MatrixBot.__new__(MatrixBot)
        bot.agent = agent
        bot.config = MagicMock()
        bot.config.user_id = "@bot:example.com"
        bot.session_log = None
        bot.heartbeat = None
        bot.umbral = None
        bot.client = None
        return bot

    def test_returns_dict_with_basic_fields(self):
        """_build_context_status returns a dict with the base agent status fields."""
        bot = self._make_bot()
        result = bot._build_context_status("!room:example.com")
        assert isinstance(result, dict)
        assert result["name"] == "test-agent"
        assert result["turns"] == 5

    def test_room_id_always_populated(self):
        """room_id field is always set from the rid argument."""
        bot = self._make_bot()
        result = bot._build_context_status("!myroom:server.com")
        assert result["room_id"] == "!myroom:server.com"

    def test_room_name_none_when_no_client(self):
        """room_name is None when client is not set."""
        bot = self._make_bot()
        bot.client = None
        result = bot._build_context_status("!room:example.com")
        assert result["room_name"] is None

    def test_room_name_from_client(self):
        """room_name comes from client.rooms[rid].named_room_name()."""
        bot = self._make_bot()
        mock_room = MagicMock()
        mock_room.named_room_name.return_value = "My Test Room"
        mock_client = MagicMock()
        mock_client.rooms = {"!room:example.com": mock_room}
        bot.client = mock_client
        result = bot._build_context_status("!room:example.com")
        assert result["room_name"] == "My Test Room"
        mock_room.named_room_name.assert_called_once_with()

    def test_umbral_null_when_no_umbral(self):
        """umbral fields are all null/False when umbral manager is absent."""
        bot = self._make_bot()
        bot.umbral = None
        result = bot._build_context_status("!room:example.com")
        assert result["umbral_active"] is False
        assert result["umbral_interval_minutes"] is None
        assert result["umbral_next_minutes"] is None

    def test_umbral_fields_when_active(self):
        """umbral fields populated when umbral is active with an entry for this room."""
        bot = self._make_bot()
        mock_entry = MagicMock()
        mock_entry.room_id = "!room:example.com"
        mock_entry.interval_seconds = 14400  # 240 min
        mock_entry.seconds_until_next = 10800  # 180 min
        mock_umbral = MagicMock()
        mock_umbral.is_active.return_value = True
        mock_umbral.status.return_value = [mock_entry]
        bot.umbral = mock_umbral
        result = bot._build_context_status("!room:example.com")
        assert result["umbral_active"] is True
        assert result["umbral_interval_minutes"] == 240
        assert result["umbral_next_minutes"] == 180

    def test_heartbeat_null_when_no_heartbeat(self):
        """heartbeat fields are all null/False when heartbeat manager is absent (regression)."""
        bot = self._make_bot()
        bot.heartbeat = None
        result = bot._build_context_status("!room:example.com")
        assert result["heartbeat_active"] is False
        assert result["heartbeat_interval_minutes"] is None
        assert result["heartbeat_next_minutes"] is None

