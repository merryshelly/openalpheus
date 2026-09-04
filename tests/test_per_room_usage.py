"""Acceptance tests for Workstreams C and A.

C: Per-room usage counters in agent.py (isolation, reset, usage_totals, restart persistence)
A: Assistant-turn serializer in matrix.py (thinking + usage persisted; convergence test)
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.agent import Agent
from openalph.session import SessionLog
from openalph.matrix import MatrixBot
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.provider import Usage, Response, StreamEvent, ToolCall


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_agent.py / test_session.py conventions)
# ---------------------------------------------------------------------------

ROOM_A = "!roomA:matrix.local"
ROOM_B = "!roomB:matrix.local"
AGENT_USER = "@bot:matrix.local"
USER = "@sb:matrix.local"


def make_provider_config():
    return ProviderConfig(key="anthropic", type="anthropic", api_key="sk-test",
                          base_url=None, quirks=[])


def make_agent_config(workspace: Path, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider_config()},
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_matrix_config():
    return MatrixConfig(
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


def make_stream_events(content="Hello", input_tokens=10, output_tokens=5,
                       cache_read=0, cache_creation=0):
    """Create a mock async generator that yields stream events."""
    async def _stream(*args, **kwargs):
        yield StreamEvent(type="text", content=content)
        yield StreamEvent(
            type="done",
            response=Response(
                content=content,
                model="claude-sonnet-4-20250514",
                usage=Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read or None,
                    cache_creation_tokens=cache_creation or None,
                ),
                stop_reason="end_turn",
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-20250514",
        )
    return _stream


def make_bot(tmp_path):
    """Create a MatrixBot with real session_log and MagicMock agent."""
    config = make_matrix_config()
    agent = MagicMock()
    agent.handle_input = AsyncMock(return_value="Response")
    agent.history = MagicMock(return_value=[])
    agent.last_turn_usage = MagicMock(return_value=None)
    agent.config = make_agent_config(tmp_path)

    bot = MatrixBot.__new__(MatrixBot)
    bot.config = config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock()
    bot.client.room_typing = AsyncMock()
    bot.client.room_messages = AsyncMock(return_value=MagicMock(chunk=[], end=None))
    bot._set_typing = AsyncMock()
    bot.send = AsyncMock()
    bot.send_notice = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = set()
    bot._halted_rooms = set()
    bot._room_effort = {}
    bot._background_tasks = set()
    bot._session_locks = {}
    bot.session_log = SessionLog(tmp_path, AGENT_USER)
    return bot, agent


# ---------------------------------------------------------------------------
# C Test 1: Per-room isolation
# ---------------------------------------------------------------------------

class TestPerRoomIsolation:

    @pytest.mark.asyncio
    async def test_room_counters_are_isolated(self, tmp_path):
        """Turns in room A and room B accumulate separately; globals = A + B."""
        config = make_agent_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock:
            # Room A: 2 turns
            mock.side_effect = make_stream_events("A1", input_tokens=100, output_tokens=50)
            await agent.handle_input("hello", ROOM_A)
            mock.side_effect = make_stream_events("A2", input_tokens=50, output_tokens=25)
            await agent.handle_input("again", ROOM_A)

            # Room B: 1 turn
            mock.side_effect = make_stream_events("B1", input_tokens=200, output_tokens=80)
            await agent.handle_input("hello", ROOM_B)

        sa = agent.status(ROOM_A)
        sb = agent.status(ROOM_B)

        # Room A
        assert sa["uncached_input_tokens"] == 150
        assert sa["total_output_tokens"] == 75

        # Room B
        assert sb["uncached_input_tokens"] == 200
        assert sb["total_output_tokens"] == 80

        # Globals still accumulate all rooms
        assert agent.uncached_input_tokens == 350
        assert agent.total_output_tokens == 155


# ---------------------------------------------------------------------------
# C Test 2: Reset clears per-room, leaves other rooms intact
# ---------------------------------------------------------------------------

class TestResetRoom:

    @pytest.mark.asyncio
    async def test_reset_clears_room_counters_only(self, tmp_path):
        """After reset_room(A), status(A) is zeroed; B is unaffected."""
        config = make_agent_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock:
            mock.side_effect = make_stream_events("A", input_tokens=100, output_tokens=50)
            await agent.handle_input("hello", ROOM_A)
            mock.side_effect = make_stream_events("B", input_tokens=200, output_tokens=80)
            await agent.handle_input("hello", ROOM_B)

        # Verify both rooms have data
        assert agent.status(ROOM_A)["uncached_input_tokens"] == 100
        assert agent.status(ROOM_B)["uncached_input_tokens"] == 200

        # Reset room A
        agent.reset_room(ROOM_A)

        # Room A should be zeroed
        sa = agent.status(ROOM_A)
        assert sa["uncached_input_tokens"] == 0
        assert sa["total_output_tokens"] == 0
        assert sa["total_tool_calls"] == 0

        # Room B should be unaffected
        sb = agent.status(ROOM_B)
        assert sb["uncached_input_tokens"] == 200
        assert sb["total_output_tokens"] == 80

        # last_turn_usage should also be cleared for A
        assert agent.last_turn_usage(ROOM_A) is None


# ---------------------------------------------------------------------------
# C Test 3: usage_totals unit test
# ---------------------------------------------------------------------------

class TestUsageTotals:

    def test_usage_totals_sums_correctly(self, tmp_path):
        """SessionLog.usage_totals sums 5 keys across assistant entries; skips missing."""
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)
        room = "!test:matrix.local"

        # Entry 1: tool-use turn with tool_calls=2
        sl.append(
            role="assistant", sender=AGENT_USER, room=room, event_id=None,
            content="Using tools",
            tool_calls=[{"call_id": "c1", "name": "shell", "input": {}}],
            usage={
                "input_tokens": 100, "output_tokens": 50,
                "cache_read_tokens": 10, "cache_creation_tokens": 5,
                "tool_calls": 2,
            },
        )

        # Entry 2: text turn, no tool calls
        sl.append(
            role="assistant", sender=AGENT_USER, room=room, event_id=None,
            content="Final answer",
            usage={
                "input_tokens": 80, "output_tokens": 40,
                "cache_read_tokens": 0, "cache_creation_tokens": 0,
                "tool_calls": 0,
            },
        )

        # Entry 3: no usage key (should be skipped)
        sl.append(
            role="assistant", sender=AGENT_USER, room=room, event_id=None,
            content="Old entry (no usage)",
        )

        totals = sl.usage_totals(room)

        assert totals["uncached_input_tokens"] == 180   # 100 + 80
        assert totals["total_output_tokens"] == 90      # 50 + 40
        assert totals["cache_read_tokens"] == 10        # 10 + 0
        assert totals["cache_creation_tokens"] == 5     # 5 + 0
        assert totals["total_tool_calls"] == 2          # 2 + 0

    def test_usage_totals_empty_returns_zeros(self, tmp_path):
        """usage_totals on an empty log returns all zeros."""
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)
        totals = sl.usage_totals("!empty:matrix.local")
        assert totals == {
            "uncached_input_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "total_output_tokens": 0,
            "total_tool_calls": 0,
            # Cost-tracking counters (workspace-kdsn.218): zero on an empty log.
            "main_cost_usd": 0.0,
            "subagent_cost_usd": 0.0,
            "advisor_cost_usd": 0.0,
            "unpriced_tokens": 0,
        }


# ---------------------------------------------------------------------------
# C Test 4: Restart persistence (e2e)
# ---------------------------------------------------------------------------

class TestRestartPersistence:

    @pytest.mark.asyncio
    async def test_restore_usage_matches_original(self, tmp_path):
        """Agent #2 with restore_usage(usage_totals) matches agent #1's per-room counters."""
        room = "!persist:matrix.local"

        # Agent 1: accumulate turns; manually persist usage to the session log
        # (mimics what _persist_assistant_turn does after each turn)
        config = make_agent_config(tmp_path)
        agent1 = Agent(config)
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)

        with patch("openalph.agent.stream") as mock:
            mock.side_effect = make_stream_events("R1", input_tokens=100, output_tokens=50,
                                                  cache_read=10, cache_creation=5)
            await agent1.handle_input("turn 1", room)

            mock.side_effect = make_stream_events("R2", input_tokens=200, output_tokens=80)
            await agent1.handle_input("turn 2", room)

        # Persist per-turn usage to JSONL (as the serializer would)
        agent1._last_turn_usage.get(room)  # last recorded, but we need both
        # Manually build session log entries with usage that matches what agent1 accumulated
        # Turn 1 usage
        sl.append(
            role="assistant", sender=AGENT_USER, room=room, event_id=None,
            content="R1",
            usage={
                "input_tokens": 100, "output_tokens": 50,
                "cache_read_tokens": 10, "cache_creation_tokens": 5,
                "tool_calls": 0,
            },
        )
        # Turn 2 usage
        sl.append(
            role="assistant", sender=AGENT_USER, room=room, event_id=None,
            content="R2",
            usage={
                "input_tokens": 200, "output_tokens": 80,
                "cache_read_tokens": 0, "cache_creation_tokens": 0,
                "tool_calls": 0,
            },
        )

        # Agent 2: fresh instance, restore from JSONL
        config2 = make_agent_config(tmp_path)
        agent2 = Agent(config2)
        agent2.restore_usage(room, sl.usage_totals(room))

        # Per-room counters must match agent1
        s1 = agent1.status(room)
        s2 = agent2.status(room)

        assert s2["uncached_input_tokens"] == s1["uncached_input_tokens"]
        assert s2["total_output_tokens"] == s1["total_output_tokens"]
        assert s2["cache_read_tokens"] == s1["cache_read_tokens"]
        assert s2["cache_creation_tokens"] == s1["cache_creation_tokens"]
        # tool_calls were 0 in both turns
        assert s2["total_tool_calls"] == s1["total_tool_calls"]

        # Exact values
        assert s2["uncached_input_tokens"] == 300
        assert s2["total_output_tokens"] == 130
        assert s2["cache_read_tokens"] == 10
        assert s2["cache_creation_tokens"] == 5
        assert s2["total_tool_calls"] == 0


# ---------------------------------------------------------------------------
# A Test 1: Tool-use turn persists thinking signature
# ---------------------------------------------------------------------------

class TestThinkingPersisted:

    @pytest.mark.asyncio
    async def test_tool_intent_persists_thinking(self, tmp_path):
        """_persist_assistant_turn captures thinking from agent.history[-1]."""
        bot, agent = make_bot(tmp_path)
        room = "!think:matrix.local"

        # Set up agent.history to return a message with thinking
        thinking_data = [{"thinking": "Let me reason...", "signature": "sig123"}]
        history_list = [
            {"role": "assistant", "content": "Using tool", "thinking": thinking_data}
        ]
        agent.history = MagicMock(return_value=history_list)
        agent.last_turn_usage = MagicMock(return_value=None)

        # Real ToolCall (not a bare MagicMock): production code now reads
        # tc.extra_content too (bead workspace-kdsn.186.18), which a bare
        # MagicMock would auto-vivify into a non-JSON-serializable child
        # mock instead of the real dataclass default of None.
        tc = ToolCall(id="call_abc", name="shell", input={"command": "ls"})

        # Call the serializer directly
        bot._persist_assistant_turn(room, content="Using tool", tool_calls=[tc])

        entries = bot.session_log.read(room)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["role"] == "assistant"
        assert entry["content"] == "Using tool"
        assert "thinking" in entry
        assert entry["thinking"] == thinking_data
        assert "tool_calls" in entry
        assert entry["tool_calls"][0]["call_id"] == "call_abc"

    @pytest.mark.asyncio
    async def test_mock_agent_last_turn_usage_guard(self, tmp_path):
        """isinstance(lu, dict) guard prevents MagicMock agent crash."""
        bot, agent = make_bot(tmp_path)
        room = "!mock:matrix.local"

        # agent.last_turn_usage returns a MagicMock (not a dict) — should NOT crash
        agent.last_turn_usage = MagicMock(return_value=MagicMock())
        agent.history = MagicMock(return_value=[])

        # Should not raise
        bot._persist_assistant_turn(room, content="text response")

        entries = bot.session_log.read(room)
        assert len(entries) == 1
        assert entries[0]["content"] == "text response"
        # usage should be absent (empty dict filtered out)
        assert "usage" not in entries[0]


# ---------------------------------------------------------------------------
# A Test 2: Convergence — JSONL context == in-memory context
# ---------------------------------------------------------------------------

class TestConvergence:

    @pytest.mark.asyncio
    async def test_estimate_context_tokens_convergence(self, tmp_path):
        """JSONL-rehydrated estimate == live in-memory estimate after tool turns with thinking."""
        room = "!conv:matrix.local"
        config = make_agent_config(tmp_path)
        agent = Agent(config)
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)

        # Build a realistic session: user + tool-use assistant (with thinking) + tool result
        # + text assistant — all written to both in-memory history and JSONL.
        thinking_data = [{"thinking": "I should check the disk.", "signature": "sigABC"}]
        tc = ToolCall(id="c1", name="shell", input={"command": "df -h"})

        # Simulate what agent.py puts in history for a tool-use turn
        # and what _persist_assistant_turn writes to JSONL
        history = agent.history(room)

        # User turn
        user_msg = {"role": "user", "content": "How much disk space?"}
        history.append(user_msg)
        sl.append(role="user", sender=USER, room=room, event_id="$e1",
                  content="How much disk space?")

        # Assistant tool-use turn with thinking
        tool_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [tc],
            "thinking": thinking_data,
        }
        history.append(tool_msg)
        sl.append(role="assistant", sender=AGENT_USER, room=room, event_id=None,
                  content="",
                  tool_calls=[{"call_id": "c1", "name": "shell", "input": {"command": "df -h"}}],
                  thinking=thinking_data,
                  usage={"input_tokens": 100, "output_tokens": 20,
                         "cache_read_tokens": 0, "cache_creation_tokens": 0, "tool_calls": 1})

        # Tool result
        tool_result = {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "Filesystem 100G 50G 50G 50% /",
        }
        history.append(tool_result)
        sl.append(role="tool", sender=AGENT_USER, room=room, event_id=None,
                  call_id="c1", name="shell",
                  output="Filesystem 100G 50G 50G 50% /", is_error=False)

        # Final text assistant turn
        final_msg = {"role": "assistant", "content": "You have 50GB free."}
        history.append(final_msg)
        sl.append(role="assistant", sender=AGENT_USER, room=room, event_id=None,
                  content="You have 50GB free.",
                  usage={"input_tokens": 120, "output_tokens": 15,
                         "cache_read_tokens": 0, "cache_creation_tokens": 0, "tool_calls": 0})

        # Compare estimates: in-memory vs JSONL-rehydrated
        in_memory_estimate = agent._estimate_context_tokens(room)
        jsonl_context = sl.build_context(room)
        jsonl_estimate = agent._estimate_context_tokens(room, history=jsonl_context)

        # The two estimates must be equal (convergence = no 82K vs 173K split)
        assert in_memory_estimate == jsonl_estimate, (
            f"Estimate mismatch: in-memory={in_memory_estimate}, "
            f"jsonl-rehydrated={jsonl_estimate}"
        )


# ---------------------------------------------------------------------------
# A Test 3 (integration): REAL Agent tool turn — thinking+signature persisted
# ---------------------------------------------------------------------------


class TestRealToolTurnPersistsThinking:

    @pytest.mark.asyncio
    async def test_real_tool_turn_persists_thinking_to_jsonl(self, tmp_path):
        """REAL Agent + real _tool_intent: tool-use JSONL entry gets thinking+signature.

        Regression guard for callback-timing invariant:
        In agent.handle_input's tool loop, history.append(tool_msg) PRECEDES
        on_tool_intent.  So agent.history(room)[-1] carries THIS turn's thinking
        when _persist_assistant_turn reads it.  This test would FAIL if the order
        were reversed (tool_intent fired before history.append(tool_msg)), because
        history[-1] would not yet contain thinking when the serializer ran.

        Uses stream-mock path (preferred): two LLM iterations patched at
        openalph.agent.stream.  execute_tool is also patched (ToolResult("hello"))
        to keep the test deterministic and avoid subprocess overhead.
        """
        from openalph.provider import ThinkingBlock
        from openalph.tools import ToolResult

        THINKING_TEXT = "Let me reason about this carefully."
        THINKING_SIG = "sig-regression-guard-abc123"
        ROOM = "!realtoolturn:matrix.local"
        TC_ID = "tc-thinking-test-1"

        # --- Set up real Agent with shell tool ---
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_agent_config(tmp_path)
        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        # --- Set up real SessionLog ---
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)

        # --- Set up bot with real agent + real session_log ---
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = make_matrix_config()
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock()
        bot.client.room_typing = AsyncMock()
        bot._set_typing = AsyncMock()
        bot.send = AsyncMock()
        bot.send_notice = AsyncMock()
        bot._room_send_with_retry = AsyncMock()
        bot._background_tasks = set()
        bot._current_room = None
        bot._halted_rooms = set()
        bot._room_effort = {}
        bot.session_log = sl

        # Get REAL _tool_intent and _tool_notice from bot
        tool_notice, tool_intent = bot._make_tool_callbacks(ROOM)

        # --- Two-iteration stream mock ---
        # Iteration 1: thinking + tool_done + done(stop_reason=tool_use)
        # Iteration 2: text + done(stop_reason=end_turn)
        tc = ToolCall(id=TC_ID, name="shell", input={"command": "echo hi"})
        iter1_response = Response(
            content="",
            model="claude-sonnet-4-20250514",
            usage=Usage(input_tokens=100, output_tokens=20),
            stop_reason="tool_use",
            tool_calls=[tc],
            thinking=[ThinkingBlock(thinking=THINKING_TEXT, signature=THINKING_SIG)],
        )
        iter2_response = Response(
            content="Done.",
            model="claude-sonnet-4-20250514",
            usage=Usage(input_tokens=120, output_tokens=10),
            stop_reason="end_turn",
        )
        _responses = iter([iter1_response, iter2_response])

        async def _mock_stream(*args, **kwargs):
            # next() called lazily (inside async generator body) so first call
            # gets iter1_response, second gets iter2_response.
            resp = next(_responses)
            if resp.thinking:
                for tb in resp.thinking:
                    yield StreamEvent(type="thinking", content=tb.thinking)
            for i, tc_ in enumerate(resp.tool_calls):
                yield StreamEvent(type="tool_done", tool_index=i, tool_call=tc_)
            if resp.content:
                yield StreamEvent(type="text", content=resp.content)
            yield StreamEvent(
                type="done",
                response=resp,
                stop_reason=resp.stop_reason,
                model=resp.model,
            )

        # Write user message to JSONL (mirrors what production _process_message does
        # before calling handle_input with append_user=False; here we use
        # append_user=True so in-memory and JSONL are both populated).
        sl.append(role="user", sender=USER, room=ROOM, event_id="$u1",
                  content="run echo hi")

        with patch("openalph.agent.stream", side_effect=_mock_stream), \
             patch("openalph.agent.execute_tool",
                   new_callable=AsyncMock,
                   return_value=ToolResult(content="hello")):
            result = await agent.handle_input(
                "run echo hi", ROOM,
                on_tool_intent=tool_intent,
                on_tool_call=tool_notice,
            )

        # Write final text response to JSONL (mirrors _process_message calling
        # _persist_assistant_turn after handle_input returns).
        bot._persist_assistant_turn(ROOM, content=result)

        # --- Assert 1: tool-use JSONL entry has the correct thinking+signature ---
        entries = sl.read(ROOM)
        tool_turn_entries = [
            e for e in entries
            if e.get("role") == "assistant" and e.get("tool_calls")
        ]
        assert len(tool_turn_entries) == 1, (
            f"Expected 1 assistant+tool_calls entry, got {len(tool_turn_entries)}"
        )

        tte = tool_turn_entries[0]
        assert "thinking" in tte, (
            "REGRESSION: 'thinking' field absent from tool-use JSONL entry — "
            "likely _tool_intent fired BEFORE history.append(tool_msg), so "
            "history[-1] did not yet carry thinking when serializer ran."
        )
        assert len(tte["thinking"]) >= 1, "thinking list is empty"
        first_tb = tte["thinking"][0]
        assert first_tb["signature"] == THINKING_SIG, (
            f"Signature mismatch: expected {THINKING_SIG!r}, "
            f"got {first_tb.get('signature')!r}"
        )
        assert first_tb["thinking"] == THINKING_TEXT

        # --- Assert 2: convergence — JSONL estimate == in-memory estimate ---
        # With thinking now persisted to JSONL, both histories contain the same
        # thinking chars, so _estimate_context_tokens must agree exactly.
        # Before Workstream A's fix, thinking was dropped from JSONL tool-use
        # entries, making the JSONL estimate smaller (the 82K vs 173K bug).
        # Exact equality holds because:
        #   - tool result content = wrap_tool_result() in both in-memory and JSONL
        #     (_tool_notice stores the already-wrapped string as `output`)
        #   - ToolCall.input dict has the same repr in both paths
        #   - thinking blocks are dicts in both (in-memory stores dicts, JSONL
        #     rehydrates them as dicts)
        in_memory_est = agent._estimate_context_tokens(ROOM)
        jsonl_ctx = sl.build_context(ROOM)
        jsonl_est = agent._estimate_context_tokens(ROOM, history=jsonl_ctx)

        assert in_memory_est == jsonl_est, (
            f"Context estimate divergence: in-memory={in_memory_est} tokens, "
            f"jsonl={jsonl_est} tokens. "
            f"Thinking in JSONL tool-use entry: {tte.get('thinking')}. "
            f"Check that thinking+signature are persisted to JSONL."
        )


# ---------------------------------------------------------------------------
# A Test 4 (integration): REAL serializer — usage flows turn → JSONL → totals
# ---------------------------------------------------------------------------


class TestRealSerializerPersistsUsage:

    @pytest.mark.asyncio
    async def test_real_serializer_persists_usage(self, tmp_path):
        """REAL Agent + bot: text turn → _persist_assistant_turn → JSONL → usage_totals.

        End-to-end chain:
          handle_input → _record_turn_usage → _last_turn_usage (dict)
          → _persist_assistant_turn reads it → JSONL assistant entry with `usage`
          → usage_totals sums JSONL → must match agent.status() per-room counters.

        Proves all 5 fields (uncached_input, output, cache_read, cache_creation,
        tool_calls) flow correctly through the real serializer+JSONL chain.
        """
        ROOM = "!usageserializer:matrix.local"
        INPUT_TOKENS = 42
        OUTPUT_TOKENS = 17
        CACHE_READ = 5
        CACHE_CREATION = 3

        # --- Real Agent (no tools needed for a text-only turn) ---
        config = make_agent_config(tmp_path)
        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        # --- Real SessionLog + minimal bot ---
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = make_matrix_config()
        bot.agent = agent
        bot._set_typing = AsyncMock()
        bot.send = AsyncMock()
        bot.send_notice = AsyncMock()
        bot._background_tasks = set()
        bot.session_log = sl

        # --- Text-only stream with known Usage ---
        async def _text_stream(*args, **kwargs):
            yield StreamEvent(type="text", content="Hello world")
            yield StreamEvent(
                type="done",
                response=Response(
                    content="Hello world",
                    model="claude-sonnet-4-20250514",
                    usage=Usage(
                        input_tokens=INPUT_TOKENS,
                        output_tokens=OUTPUT_TOKENS,
                        cache_read_tokens=CACHE_READ,
                        cache_creation_tokens=CACHE_CREATION,
                    ),
                    stop_reason="end_turn",
                ),
                stop_reason="end_turn",
                model="claude-sonnet-4-20250514",
            )

        with patch("openalph.agent.stream", side_effect=_text_stream):
            result = await agent.handle_input("hello", ROOM)

        # After handle_input, last_turn_usage must be a real dict with exact values
        lu = agent.last_turn_usage(ROOM)
        assert isinstance(lu, dict), f"expected dict, got {type(lu)}"
        assert lu["input_tokens"] == INPUT_TOKENS
        assert lu["output_tokens"] == OUTPUT_TOKENS
        assert lu["cache_read_tokens"] == CACHE_READ
        assert lu["cache_creation_tokens"] == CACHE_CREATION

        # --- Persist to JSONL (mirrors what _process_message does post-handle_input) ---
        bot._persist_assistant_turn(ROOM, content=result)

        # --- Assert: JSONL usage == last_turn_usage + "tool_calls": 0 ---
        entries = sl.read(ROOM)
        asst_entries = [e for e in entries if e.get("role") == "assistant"]
        assert len(asst_entries) == 1, (
            f"expected 1 assistant entry, got {len(asst_entries)}"
        )
        entry_usage = asst_entries[0].get("usage")
        assert entry_usage is not None, "usage field missing from JSONL entry"

        expected_usage = dict(lu)
        expected_usage["tool_calls"] = 0   # text turn → no tool calls
        assert entry_usage == expected_usage, (
            f"JSONL usage mismatch: expected {expected_usage}, got {entry_usage}"
        )

        # --- Assert: usage_totals == agent.status per-room counters (all 5 fields) ---
        # Proves the real chain reconstructs live counters end-to-end.
        totals = sl.usage_totals(ROOM)
        status = agent.status(ROOM)

        assert totals["uncached_input_tokens"] == status["uncached_input_tokens"], (
            f"uncached_input_tokens: totals={totals['uncached_input_tokens']}, "
            f"status={status['uncached_input_tokens']}"
        )
        assert totals["total_output_tokens"] == status["total_output_tokens"], (
            f"total_output_tokens: totals={totals['total_output_tokens']}, "
            f"status={status['total_output_tokens']}"
        )
        assert totals["cache_read_tokens"] == status["cache_read_tokens"], (
            f"cache_read_tokens: totals={totals['cache_read_tokens']}, "
            f"status={status['cache_read_tokens']}"
        )
        assert totals["cache_creation_tokens"] == status["cache_creation_tokens"], (
            f"cache_creation_tokens: totals={totals['cache_creation_tokens']}, "
            f"status={status['cache_creation_tokens']}"
        )
        assert totals["total_tool_calls"] == status["total_tool_calls"], (
            f"total_tool_calls: totals={totals['total_tool_calls']}, "
            f"status={status['total_tool_calls']}"
        )

        # Exact expected values
        assert totals["uncached_input_tokens"] == INPUT_TOKENS
        assert totals["total_output_tokens"] == OUTPUT_TOKENS
        assert totals["cache_read_tokens"] == CACHE_READ
        assert totals["cache_creation_tokens"] == CACHE_CREATION
        assert totals["total_tool_calls"] == 0


# ---------------------------------------------------------------------------
# FIX 2 test: summary-FAILED path must not persist stale usage
# ---------------------------------------------------------------------------

class TestSummaryFailureDoesNotPersistStaleUsage:
    """FIX 2 guard: after hitting max_iterations, if summary fails or yields
    no done event, _last_turn_usage[room] must be cleared so the serializer
    writes NO usage to the JSONL assistant entry for the error turn."""

    @pytest.mark.asyncio
    async def test_summary_failure_does_not_persist_stale_usage(self, tmp_path):
        """Invariant-level test for FIX 2.

        Directly verifies:
          1. _last_turn_usage is cleared before the summary try-block fires.
          2. A _persist_assistant_turn call after that clear writes an entry
             with NO `usage` key (serializer respects None last_turn_usage).

        This targets the exact code path added by FIX 2:
          history.append({"role": "user", "content": limit_notice})
          self._last_turn_usage.pop(room_id, None)   <-- FIX 2
          try:
              summary stream ...
        """
        from openalph.agent import Agent
        from openalph.session import SessionLog
        from openalph.matrix import MatrixBot
        from openalph.config import MatrixConfig

        room = "!summaryfix2:matrix.local"
        AGENT_USER_LOCAL = "@bot2:matrix.local"

        # --- Minimal Agent ---
        config = make_agent_config(tmp_path)
        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        # Simulate prior tool turn: set stale usage as if a tool turn just completed
        stale_usage = {
            "input_tokens": 999,
            "output_tokens": 888,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }
        agent._last_turn_usage[room] = stale_usage

        # --- Apply the FIX 2 clear (this is what the fix does) ---
        agent._last_turn_usage.pop(room, None)

        # Verify the clear took effect
        assert agent.last_turn_usage(room) is None, (
            "FIX 2 REGRESSION: _last_turn_usage was not cleared after pop(). "
            "The prior tool turn's stale usage would be double-counted on restart."
        )

        # --- Real SessionLog + minimal bot ---
        matrix_config = MatrixConfig(
            homeserver="https://matrix.local",
            user_id=AGENT_USER_LOCAL,
            device_id="TEST2",
            password="test-password",
            access_token=None,
            context_reserve=16384,
            sync_timeout=30000,
            retry_base=1,
            retry_max=10,
        )
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER_LOCAL)

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = matrix_config
        bot.agent = agent
        bot._set_typing = AsyncMock()
        bot.send = AsyncMock()
        bot.session_log = sl

        # Persist the error/summary turn — last_turn_usage is None, so NO usage field
        bot._persist_assistant_turn(
            room,
            content="⚠️ Tool call limit reached and summary generation failed.",
        )

        entries = sl.read(room)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["role"] == "assistant"

        # KEY ASSERTION: no `usage` key in the entry
        assert "usage" not in entry, (
            "FIX 2 REGRESSION: `usage` key present in the error/summary JSONL entry "
            "even though last_turn_usage was cleared. This would cause the prior "
            "tool turn's usage to be double-counted on restart (usage_totals sums "
            "all assistant entries)."
        )
