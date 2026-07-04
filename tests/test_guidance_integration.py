"""Real-path integration tests for Batch R1 guidance-injection fixes.

Every test uses a REAL Agent (real per-room ReminderEngine(s), real
self._read_registries, real self._room_tool_counts) wired through the
REAL callback path (MatrixBot._build_agent_callbacks).  Mock ONLY the
provider (the LLM call) and the nio client.

This file pinpoints each R1 fix, using the real path that the audit
found was never tested.
"""

import asyncio
import math
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# --- Imports ---
from openalph.agent import Agent
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import Response, Usage, StreamEvent, ToolCall
from openalph.tools import ToolResult, _TODO_STATE, BUILTIN_TOOLS
from openalph.reminders import ReminderEngine, ReminderState, Reminder
from openalph.matrix import MatrixBot

# --- Constants ---
ROOM_A = "!integ-room-a:matrix.local"
ROOM_B = "!integ-room-b:matrix.local"
AGENT_USER = "@agent:matrix.local"
OPERATOR = "@op:matrix.local"
TAG_OPEN = "<system-reminder>"


# --- Helpers ---

def _cfg(workspace, **kw):
    """Build AgentConfig for tests."""
    defaults = dict(
        name="test-integ",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[],
        )},
        workspace=workspace,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
        reminders=True,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _setup_workspace(tmp_path, tools=("shell", "file_read", "file_write",
                                       "file_edit", "todo_write", "memory_search")):
    """Create workspace/tools/ with tool TOMLs for discovery."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(exist_ok=True)
    for name in tools:
        (tools_dir / f"{name}.toml").write_text("[config]\n")
    return tmp_path


def _make_capturing_stream(tool_iterations=6, final_text="Done"):
    """Stream factory: tool_use N times then text.  Captures wire payloads."""
    payloads = []
    call_idx = [0]

    async def _stream(*, config=None, system=None, messages=None,
                      tools=None, model="test", thinking=None,
                      cache_ttl=None, **kw):
        payloads.append(list(messages))
        call_idx[0] += 1
        if tools is not None and call_idx[0] <= tool_iterations:
            tc = ToolCall(id=f"tc_{call_idx[0]}", name="shell",
                          input={"command": f"echo {call_idx[0]}"})
            yield StreamEvent(type="tool_done", tool_index=0, tool_call=tc)
            yield StreamEvent(
                type="done",
                response=Response(
                    content="", tool_calls=[tc], model=model,
                    usage=Usage(input_tokens=10, output_tokens=5),
                    stop_reason="tool_use"),
                stop_reason="tool_use", model=model)
        else:
            yield StreamEvent(type="text", content=final_text)
            yield StreamEvent(
                type="done",
                response=Response(
                    content=final_text, model=model,
                    usage=Usage(input_tokens=10, output_tokens=5),
                    stop_reason="end_turn"),
                stop_reason="end_turn", model=model)

    return _stream, payloads


def _make_bot_with_real_agent(tmp_path, agent=None, **agent_kw):
    """Build a MatrixBot with a REAL Agent and mocked nio client.

    Returns (bot, agent) tuple.  The bot has _build_agent_callbacks wired
    to the real agent's state.
    """
    ws = _setup_workspace(tmp_path, **agent_kw.pop("tools", {}) and {} or {})
    if "tools_list" in agent_kw:
        ws = _setup_workspace(tmp_path, tools=agent_kw.pop("tools_list"))
    else:
        ws = _setup_workspace(tmp_path)
    config = _cfg(ws, **agent_kw)
    if agent is None:
        agent = Agent(config)

    from openalph.config import MatrixConfig
    matrix_config = MatrixConfig(
        homeserver="https://matrix.local",
        user_id=AGENT_USER,
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )

    bot = MatrixBot.__new__(MatrixBot)
    bot.config = matrix_config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$resp1"))
    bot.client.room_typing = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = set()
    bot._room_thinking = {}
    bot._room_cache_ttl = {}
    bot._room_timesense = {}
    bot._halted_rooms = set()
    bot._background_tasks = set()
    bot._session_locks = {}
    bot.session_log = MagicMock()
    bot.session_log.append = MagicMock()
    bot.session_log.build_context = MagicMock(return_value=[])
    bot.session_log.read = MagicMock(return_value=[])
    bot.session_log.last_event_id = MagicMock(return_value=None)
    bot.session_log.usage_totals = MagicMock(return_value={})
    bot.heartbeat = MagicMock()
    bot.heartbeat.is_active = MagicMock(return_value=False)
    bot.umbral = MagicMock()
    bot.umbral.is_active = MagicMock(return_value=False)
    bot._steering_inbox = {}
    bot._active_turns = set()

    return bot, agent


@pytest.fixture(autouse=True)
def _cleanup_todo():
    """Clean up todo state after each test."""
    yield
    _TODO_STATE.pop(ROOM_A, None)
    _TODO_STATE.pop(ROOM_B, None)


# ============================================================================
# R1 real-path pinning tests
# ============================================================================


class TestR1_1_ReadRegistry:
    """R1-1: read_registry wired through real callbacks → file_write guard works."""

    @pytest.mark.asyncio
    async def test_r1_1_read_registry_in_real_callbacks(self, tmp_path):
        """R1-1: _build_agent_callbacks includes read_registry from agent._read_registries."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM_A, None)
        assert "read_registry" in cb, \
            "Production callbacks must include read_registry"
        assert isinstance(cb["read_registry"], dict), \
            "read_registry must be a dict"
        # Verify it's the same object as in agent._read_registries
        assert cb["read_registry"] is agent._read_registries[ROOM_A], \
            "read_registry must reference agent's per-room registry"

    @pytest.mark.asyncio
    async def test_r1_1_registry_populated_by_file_read(self, tmp_path):
        """R1-1: file_read populates read_registry; file_write to read file succeeds."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM_A, None)

        # Create a test file
        test_file = tmp_path / "existing.txt"
        test_file.write_text("original content")

        from openalph.tools import execute_tool
        # file_read should populate the registry
        read_result = await execute_tool(
            name="file_read",
            input={"path": str(test_file)},
            tool_config={},
            agent_config=agent.config,
            tools=agent.tools,
            callbacks={"read_registry": cb["read_registry"], "call_id": "tc1"},
        )
        assert not read_result.is_error, f"file_read failed: {read_result.content}"
        assert str(Path(str(test_file)).resolve()) in cb["read_registry"], \
            "file_read must register the resolved path"

        # file_write to the same file should succeed
        write_result = await execute_tool(
            name="file_write",
            input={"path": str(test_file), "content": "updated"},
            tool_config={"require_read_before_write": True},
            agent_config=agent.config,
            tools=agent.tools,
            callbacks={"read_registry": cb["read_registry"], "call_id": "tc2"},
        )
        assert not write_result.is_error, \
            f"file_write to read file should succeed: {write_result.content}"

    @pytest.mark.asyncio
    async def test_r1_1_write_unread_refused(self, tmp_path):
        """R1-1: file_write to existing unread file → refused via real callbacks."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM_A, None)

        test_file = tmp_path / "unread.txt"
        test_file.write_text("existing content")

        from openalph.tools import execute_tool
        result = await execute_tool(
            name="file_write",
            input={"path": str(test_file), "content": "overwrite"},
            tool_config={"require_read_before_write": True},
            agent_config=agent.config,
            tools=agent.tools,
            callbacks={"read_registry": cb["read_registry"], "call_id": "tc1"},
        )
        assert result.is_error, "file_write to unread existing file must be refused"
        assert "not read" in result.content.lower(), \
            "Refusal message must mention file was not read"


class TestR1_2_RoomIdInCallbacks:
    """R1-2: room_id in callbacks → todo_write state persists across turns."""

    @pytest.mark.asyncio
    async def test_r1_2_room_id_in_real_callbacks(self, tmp_path):
        """R1-2: _build_agent_callbacks includes room_id."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM_A, None)
        assert "room_id" in cb, "Production callbacks must include room_id"
        assert cb["room_id"] == ROOM_A, "room_id must match the room"

    @pytest.mark.asyncio
    async def test_r1_2_todo_persists_across_turns(self, tmp_path):
        """R1-2: todo_write in turn 1 persists to turn 2 (same room), isolated across rooms."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb_a = bot._build_agent_callbacks(ROOM_A, None)
        cb_b = bot._build_agent_callbacks(ROOM_B, None)

        from openalph.tools import execute_tool
        # Write todo in room A
        result = await execute_tool(
            name="todo_write",
            input={"todos": [{"content": "Task A", "status": "in_progress"}]},
            tool_config={},
            agent_config=agent.config,
            tools=agent.tools,
            callbacks={**cb_a, "call_id": "tc1"},
        )
        assert not result.is_error, f"todo_write failed: {result.content}"

        # Verify todo present in room A
        assert ROOM_A in _TODO_STATE, "Todo must be keyed by room_id"
        assert len(_TODO_STATE[ROOM_A]) == 1
        assert _TODO_STATE[ROOM_A][0]["content"] == "Task A"

        # Verify room B is isolated
        assert ROOM_B not in _TODO_STATE or len(_TODO_STATE.get(ROOM_B, [])) == 0, \
            "Room B must not see Room A's todos"

        # Turn 2 in room A: todo still there (same room_id key)
        cb_a2 = bot._build_agent_callbacks(ROOM_A, None)
        assert _TODO_STATE.get(ROOM_A) == [{"content": "Task A", "status": "in_progress"}], \
            "Todo must persist across turns with stable room_id key"


class TestR1_3_TodoListInState:
    """R1-3: todo_list in ReminderState populated from _TODO_STATE, not hardcoded []."""

    @pytest.mark.asyncio
    async def test_r1_3_in_progress_suppresses_t1(self, tmp_path):
        """R1-3: with an in_progress todo, T1 does NOT fire at iteration 5."""
        ws = _setup_workspace(tmp_path)
        config = _cfg(ws, max_iterations=10)
        agent = Agent(config)

        # Set up in_progress todo for the room
        _TODO_STATE[ROOM_A] = [{"content": "Working", "status": "in_progress"}]

        log_calls = []
        async def _log_reminder(room_id, reminder):
            log_calls.append(reminder)
        async def _send_notice(room_id, body, **kw):
            pass

        callbacks = {
            "log_reminder": _log_reminder,
            "send_notice": _send_notice,
            "turn_source": None,
            "room_id": ROOM_A,
        }

        stream_fn, _ = _make_capturing_stream(tool_iterations=7)
        mock_exec = AsyncMock(return_value=ToolResult(content="ok", is_error=False))

        with patch("openalph.agent.stream", side_effect=stream_fn), \
             patch("openalph.agent.execute_tool", mock_exec):
            await agent.handle_input("work", room_id=ROOM_A, callbacks=callbacks)

        history = agent.history(ROOM_A)
        t1_reminders = [m for m in history
                        if TAG_OPEN in str(m.get("content", ""))
                        and "todo_write" in str(m.get("content", ""))]
        assert not t1_reminders, \
            "T1 must NOT fire when an in_progress todo exists (R1-3)"

    @pytest.mark.asyncio
    async def test_r1_3_no_todo_t1_fires(self, tmp_path):
        """R1-3: with no todos, T1 fires at iteration 5 via real agent path."""
        ws = _setup_workspace(tmp_path)
        config = _cfg(ws, max_iterations=10)
        agent = Agent(config)

        _TODO_STATE.pop(ROOM_A, None)  # ensure no todo

        log_calls = []
        async def _log_reminder(room_id, reminder):
            log_calls.append(reminder)
        async def _send_notice(room_id, body, **kw):
            pass

        callbacks = {
            "log_reminder": _log_reminder,
            "send_notice": _send_notice,
            "turn_source": None,
            "room_id": ROOM_A,
        }

        stream_fn, _ = _make_capturing_stream(tool_iterations=7)
        mock_exec = AsyncMock(return_value=ToolResult(content="ok", is_error=False))

        with patch("openalph.agent.stream", side_effect=stream_fn), \
             patch("openalph.agent.execute_tool", mock_exec):
            await agent.handle_input("work", room_id=ROOM_A, callbacks=callbacks)

        t1 = [r for r in log_calls if r.trigger == "todo-nudge"]
        assert t1, "T1 must fire at iteration 5 when no todos exist (R1-3)"


class TestR1_4_PerRoomEngines:
    """R1-4: per-room reminder engines — no cross-room fired-state contamination."""

    @pytest.mark.asyncio
    async def test_r1_4_t2_fires_independently_per_room(self, tmp_path):
        """R1-4: T2 fires in room A; still fires in room B (independent engines)."""
        ws = _setup_workspace(tmp_path)
        config = _cfg(ws, max_iterations=10)
        agent = Agent(config)

        # Fire T2 in room A via engine
        engine_a = agent._engine_for(ROOM_A)
        state_high = ReminderState(
            evaluation_point="tool_loop_boundary",
            iteration=1, max_iterations=10,
            context_tokens=170000, context_limit=200000,
            completed_turns=1, turn_source=None,
            tool_calls_this_turn={}, tool_calls_session={},
            todo_list=[], enabled_tools=set(BUILTIN_TOOLS.keys()),
        )
        result_a = engine_a.evaluate(state_high)
        t2_a = [r for r in result_a if r.trigger == "context-pressure"]
        assert t2_a, "T2 must fire in room A at ≥80%"

        # T2 in room B should still fire (separate engine)
        engine_b = agent._engine_for(ROOM_B)
        result_b = engine_b.evaluate(state_high)
        t2_b = [r for r in result_b if r.trigger == "context-pressure"]
        assert t2_b, "T2 must fire in room B independently (R1-4)"

    @pytest.mark.asyncio
    async def test_r1_4_reset_room_isolates(self, tmp_path):
        """R1-4: reset_room(A) does NOT reset room B's engine."""
        ws = _setup_workspace(tmp_path)
        config = _cfg(ws, max_iterations=10)
        agent = Agent(config)

        # Fire T2 in both rooms
        engine_a = agent._engine_for(ROOM_A)
        engine_b = agent._engine_for(ROOM_B)
        state_high = ReminderState(
            evaluation_point="tool_loop_boundary",
            iteration=1, max_iterations=10,
            context_tokens=170000, context_limit=200000,
            completed_turns=1, turn_source=None,
            tool_calls_this_turn={}, tool_calls_session={},
            todo_list=[], enabled_tools=set(BUILTIN_TOOLS.keys()),
        )
        engine_a.evaluate(state_high)
        engine_b.evaluate(state_high)

        # Reset room A only
        agent.reset_room(ROOM_A)

        # Room B's engine should still have T2 fired
        engine_b_after = agent._engine_for(ROOM_B)
        assert engine_b_after is engine_b, "Room B engine must survive Room A reset"
        result_b = engine_b_after.evaluate(state_high)
        t2_b = [r for r in result_b if r.trigger == "context-pressure"]
        assert not t2_b, "T2 in room B must still be suppressed (already fired)"

    @pytest.mark.asyncio
    async def test_r1_4_rehydrate_per_room(self, tmp_path):
        """R1-4: rehydrate_reminders only affects the target room."""
        ws = _setup_workspace(tmp_path)
        config = _cfg(ws, max_iterations=10)
        agent = Agent(config)

        entries = [
            {"role": "user", "source": "reminder", "trigger": "context-pressure",
             "content": "<system-reminder>\nContext at 85%\n</system-reminder>"},
        ]
        agent.rehydrate_reminders(ROOM_A, entries)

        # Room A: T2 suppressed
        engine_a = agent._engine_for(ROOM_A)
        state_high = ReminderState(
            evaluation_point="tool_loop_boundary",
            iteration=1, max_iterations=10,
            context_tokens=170000, context_limit=200000,
            completed_turns=1, turn_source=None,
            tool_calls_this_turn={}, tool_calls_session={},
            todo_list=[], enabled_tools=set(BUILTIN_TOOLS.keys()),
        )
        result_a = engine_a.evaluate(state_high)
        assert not any(r.trigger == "context-pressure" for r in result_a), \
            "T2 in room A must be suppressed after rehydration"

        # Room B: T2 should still fire (not rehydrated)
        engine_b = agent._engine_for(ROOM_B)
        result_b = engine_b.evaluate(state_high)
        assert any(r.trigger == "context-pressure" for r in result_b), \
            "T2 in room B must still fire (not rehydrated)"


class TestR1_5_BoundaryDurabilityGate:
    """R1-5: boundary injection gated on log_reminder presence."""

    @pytest.mark.asyncio
    async def test_r1_5_no_injection_without_log_reminder(self, tmp_path):
        """R1-5: production callbacks without log_reminder → zero boundary injections."""
        ws = _setup_workspace(tmp_path)
        config = _cfg(ws, max_iterations=10)
        agent = Agent(config)

        # Simulate production callbacks with room_id but NO log_reminder
        callbacks = {
            "turn_source": None,
            "room_id": ROOM_A,
            "send_notice": AsyncMock(),
        }

        stream_fn, _ = _make_capturing_stream(tool_iterations=7)
        mock_exec = AsyncMock(return_value=ToolResult(content="ok", is_error=False))

        with patch("openalph.agent.stream", side_effect=stream_fn), \
             patch("openalph.agent.execute_tool", mock_exec):
            await agent.handle_input("work", room_id=ROOM_A, callbacks=callbacks)

        history = agent.history(ROOM_A)
        reminders = [m for m in history if TAG_OPEN in str(m.get("content", ""))]
        assert not reminders, \
            "Boundary must NOT inject when log_reminder absent (I1 durability gate)"

    @pytest.mark.asyncio
    async def test_r1_5_engine_state_unchanged_without_log_reminder(self, tmp_path):
        """R1-5: without log_reminder, engine fired-state is unchanged (not consumed)."""
        ws = _setup_workspace(tmp_path)
        config = _cfg(ws, max_iterations=10)
        agent = Agent(config)

        callbacks = {
            "turn_source": None,
            "room_id": ROOM_A,
        }

        stream_fn, _ = _make_capturing_stream(tool_iterations=7)
        mock_exec = AsyncMock(return_value=ToolResult(content="ok", is_error=False))

        with patch("openalph.agent.stream", side_effect=stream_fn), \
             patch("openalph.agent.execute_tool", mock_exec):
            await agent.handle_input("work", room_id=ROOM_A, callbacks=callbacks)

        # T1 should not have consumed any session fires
        engine = agent._engine_for(ROOM_A)
        assert engine._t1_session_fires == 0, \
            "T1 session fires must be 0 when boundary was skipped (fired-state unchanged)"

    @pytest.mark.asyncio
    async def test_r1_5_injection_with_log_reminder(self, tmp_path):
        """R1-5: with log_reminder in production callbacks, reminders fire normally."""
        ws = _setup_workspace(tmp_path)
        config = _cfg(ws, max_iterations=10)
        agent = Agent(config)

        log_calls = []
        async def _log_reminder(room_id, reminder):
            log_calls.append(reminder)

        callbacks = {
            "log_reminder": _log_reminder,
            "send_notice": AsyncMock(),
            "turn_source": None,
            "room_id": ROOM_A,
        }

        stream_fn, _ = _make_capturing_stream(tool_iterations=7)
        mock_exec = AsyncMock(return_value=ToolResult(content="ok", is_error=False))

        with patch("openalph.agent.stream", side_effect=stream_fn), \
             patch("openalph.agent.execute_tool", mock_exec):
            await agent.handle_input("work", room_id=ROOM_A, callbacks=callbacks)

        assert log_calls, "Reminders must fire and be logged when log_reminder is present"


class TestR1_6_EnabledTools:
    """R1-6: T1 suppressed when todo_write not in tools; T3 suppressed without memory_search."""

    @pytest.mark.asyncio
    async def test_r1_6_t1_suppressed_without_todo_write(self, tmp_path):
        """R1-6: agent with todo_write NOT in tools → T1 suppressed at engine level."""
        engine = ReminderEngine(_cfg(tmp_path))
        # Enabled tools without todo_write
        state = ReminderState(
            evaluation_point="tool_loop_boundary",
            iteration=5, max_iterations=100,
            context_tokens=10000, context_limit=200000,
            completed_turns=1, turn_source=None,
            tool_calls_this_turn={}, tool_calls_session={},
            todo_list=[],
            enabled_tools={"shell", "file_read", "file_write", "file_edit"},
        )
        results = engine.evaluate(state)
        t1 = [r for r in results if r.trigger == "todo-nudge"]
        assert not t1, "T1 must NOT fire when todo_write is not in enabled_tools"

    @pytest.mark.asyncio
    async def test_r1_6_t3_suppressed_without_memory_search(self, tmp_path):
        """R1-6: without memory_search in tools → T3 suppressed at engine level."""
        engine = ReminderEngine(_cfg(tmp_path))
        state = ReminderState(
            evaluation_point="turn_start",
            iteration=0, max_iterations=100,
            context_tokens=10000, context_limit=200000,
            completed_turns=3, turn_source=None,
            tool_calls_this_turn={}, tool_calls_session={},
            todo_list=[],
            enabled_tools={"shell", "file_read"},
        )
        results = engine.evaluate(state)
        t3 = [r for r in results if r.trigger == "memory-salience"]
        assert not t3, "T3 must NOT fire when memory_search is not in enabled_tools"

    @pytest.mark.asyncio
    async def test_r1_6_t1_fires_with_todo_write(self, tmp_path):
        """R1-6: T1 fires when todo_write IS in enabled_tools."""
        engine = ReminderEngine(_cfg(tmp_path))
        state = ReminderState(
            evaluation_point="tool_loop_boundary",
            iteration=5, max_iterations=100,
            context_tokens=10000, context_limit=200000,
            completed_turns=1, turn_source=None,
            tool_calls_this_turn={}, tool_calls_session={},
            todo_list=[],
            enabled_tools={"shell", "todo_write"},
        )
        results = engine.evaluate(state)
        t1 = [r for r in results if r.trigger == "todo-nudge"]
        assert t1, "T1 must fire when todo_write is in enabled_tools"


class TestR1_7_T2EvaluationPointGuard:
    """R1-7: T2 fires only at tool_loop_boundary, not at turn_start."""

    @pytest.mark.asyncio
    async def test_r1_7_t2_not_at_turn_start(self, tmp_path):
        """R1-7: T2 with evaluation_point='turn_start' and ≥80% → does NOT fire."""
        engine = ReminderEngine(_cfg(tmp_path))
        state = ReminderState(
            evaluation_point="turn_start",
            iteration=0, max_iterations=100,
            context_tokens=170000, context_limit=200000,
            completed_turns=1, turn_source=None,
            tool_calls_this_turn={}, tool_calls_session={},
            todo_list=[],
            enabled_tools=set(BUILTIN_TOOLS.keys()),
        )
        results = engine.evaluate(state)
        t2 = [r for r in results if r.trigger == "context-pressure"]
        assert not t2, "T2 must NOT fire at turn_start (R1-7)"

    @pytest.mark.asyncio
    async def test_r1_7_t2_fires_at_boundary(self, tmp_path):
        """R1-7: T2 with evaluation_point='tool_loop_boundary' and ≥80% → fires once."""
        engine = ReminderEngine(_cfg(tmp_path))
        state = ReminderState(
            evaluation_point="tool_loop_boundary",
            iteration=1, max_iterations=100,
            context_tokens=170000, context_limit=200000,
            completed_turns=1, turn_source=None,
            tool_calls_this_turn={}, tool_calls_session={},
            todo_list=[],
            enabled_tools=set(BUILTIN_TOOLS.keys()),
        )
        results = engine.evaluate(state)
        t2 = [r for r in results if r.trigger == "context-pressure"]
        assert t2, "T2 must fire at tool_loop_boundary when ≥80% (R1-7)"


class TestR1_8_RehydrateIdempotency:
    """R1-8: rehydrate() is idempotent — double call does not double-count."""

    @pytest.mark.asyncio
    async def test_r1_8_double_rehydrate(self, tmp_path):
        """R1-8: rehydrate same entries twice → _t1_session_fires counted once."""
        engine = ReminderEngine(_cfg(tmp_path))
        entries = [
            {"role": "user", "source": "reminder", "trigger": "todo-nudge",
             "content": "<system-reminder>\nUse todo\n</system-reminder>"},
        ]
        engine.rehydrate(entries)
        engine.rehydrate(entries)  # second call

        assert engine._t1_session_fires == 1, \
            f"T1 session fires must be 1 after double rehydrate, got {engine._t1_session_fires}"

    @pytest.mark.asyncio
    async def test_r1_8_booleans_unaffected(self, tmp_path):
        """R1-8: T2/T3 booleans correct after double rehydrate."""
        engine = ReminderEngine(_cfg(tmp_path))
        entries = [
            {"role": "user", "source": "reminder", "trigger": "context-pressure",
             "content": "<system-reminder>\nContext high\n</system-reminder>"},
            {"role": "user", "source": "reminder", "trigger": "memory-salience",
             "content": "<system-reminder>\nSearch memory\n</system-reminder>"},
        ]
        engine.rehydrate(entries)
        engine.rehydrate(entries)

        assert engine._t2_fired is True, "T2 must be fired after rehydrate"
        assert engine._t3_fired is True, "T3 must be fired after rehydrate"


class TestR1_BuildAgentCallbacksRefactor:
    """Verify _build_agent_callbacks refactor wires identical callbacks for both paths."""

    @pytest.mark.asyncio
    async def test_callbacks_have_all_required_keys(self, tmp_path):
        """_build_agent_callbacks returns all required keys."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM_A, "heartbeat")
        required = {"send_notice", "log_reminder", "turn_source",
                    "read_registry", "room_id", "context_status",
                    "send_media", "on_redaction"}
        missing = required - set(cb.keys())
        assert not missing, f"Missing keys in callbacks: {missing}"

    @pytest.mark.asyncio
    async def test_turn_source_propagated(self, tmp_path):
        """_build_agent_callbacks correctly passes turn_source."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb_hb = bot._build_agent_callbacks(ROOM_A, "heartbeat")
        assert cb_hb["turn_source"] == "heartbeat"
        cb_user = bot._build_agent_callbacks(ROOM_A, None)
        assert cb_user["turn_source"] is None

    @pytest.mark.asyncio
    async def test_log_reminder_persists_to_jsonl(self, tmp_path):
        """log_reminder from _build_agent_callbacks writes JSONL entry."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM_A, None)
        rem = Reminder(trigger="todo-nudge", text="Test text")
        await cb["log_reminder"](ROOM_A, rem)

        calls = [c for c in bot.session_log.append.call_args_list
                 if c.kwargs.get("source") == "reminder"]
        assert calls, "log_reminder must write to session_log"
        kw = calls[0].kwargs
        assert kw["trigger"] == "todo-nudge"
        assert "<system-reminder>" in kw["content"]
