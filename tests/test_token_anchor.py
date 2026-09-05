"""Red suite for kdsn.329 — provider-usage token anchor for context-pressure tiers.

Spec: memory/projects/openalph/specs/kdsn.329-token-anchor-spec.md
Recon: tmp/kdsn329-recon-anchors.md

The defect (Florin PwgQ9SB3igw3OzGYjh, 2026-09-05): ALL context-pressure
thresholds (auto handoff tier 85%, hard guard 92%, reminder ladder, checkpoint)
consume chars//4 heuristics
(Agent._estimate_context_tokens in agent.py, module _estimate_context_tokens in
tools/subagent.py). Real sessions tokenize at up to ~3.27 chars/token, so the
estimate undercounted by ~22% and NO tier was reachable before the provider
returned 400 (268,017 > 262,144).

The fix anchors the estimate to the last provider-reported true prompt size
(input + cache_read + cache_creation) per room, plus marginal chars//4 growth
since that call, invalidating on history rebuild / reset / model switch, and
rehydrating a FLOOR-ONLY anchor from JSONL on restart.

Frozen internal contract under test:
  Agent._token_anchor                  dict[str, tuple[int, int | None]]
  Agent._context_char_total(room_id="_default", history=None) -> int
  Agent._set_anchor_floor(room_id, tokens)     # (tokens, None)
  Agent._record_turn_usage(..., pre_call_chars=None)
  SessionLog.last_prompt_tokens(room_id) -> int | None
  tools/subagent._estimate_context_tokens(msgs, anchor=None)

REAL-PATH discipline (tool-management "the one lesson"): C01/C02 drive a real
Agent inside a real MatrixBot-built callbacks path with the REAL boundary
writer — only the provider stream (and in C02, the tool executor) are fakes.
"""

import json as _json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.agent import Agent
from openalph.config import AgentConfig, ProviderConfig
from openalph.matrix import MatrixBot, MatrixConfig
from openalph.provider import Response, StreamEvent, ToolCall, Usage
from openalph.session import SessionLog
from openalph.tools import ToolResult
from openalph.tools.subagent import _estimate_context_tokens as _sub_estimate
from openalph.tools.subagent import run_subagent

ROOM = "!anchor:matrix.local"
AGENT_ID = "@anchor-agent:matrix.local"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_provider(key="anthropic", type="anthropic", api_key="sk-test",
                  base_url=None):
    return ProviderConfig(key=key, type=type, api_key=api_key,
                          base_url=base_url, quirks=[])


def make_config(workspace, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider()},
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_stream_fn(calls):
    """Fake stream(): each inner list is one invocation's events, in order."""
    call_iter = iter(calls)

    async def _fake(*args, **kwargs):
        events = next(call_iter)
        for event in events:
            yield event

    return _fake


def _done(input_tokens=100, output_tokens=50, cr=None, cc=None,
          stop_reason="end_turn", content="Done", tool_calls=None):
    return StreamEvent(
        type="done",
        response=Response(
            content=content,
            model="claude-sonnet-4-20250514",
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens,
                        cache_read_tokens=cr, cache_creation_tokens=cc),
            stop_reason=stop_reason,
            tool_calls=tool_calls or [],
        ),
        stop_reason=stop_reason, model="claude-sonnet-4-20250514",
    )


def _real_bot(tmp_path, *, model_max_tokens=150000, max_tokens=8192,
              truncation_limit=50000, tools=("shell",)):
    """Real MatrixBot + real Agent + real SessionLog; nio client mocked.
    Mirrors test_context_gc_integration._gc_real_bot."""
    ws = tmp_path
    (ws / "tools").mkdir(exist_ok=True)
    for name in tools:
        (ws / "tools" / f"{name}.toml").write_text("[config]\n")
    config = make_config(ws, model_max_tokens=model_max_tokens,
                         max_tokens=max_tokens,
                         truncation_limit=truncation_limit)
    agent = Agent(config)
    bot = MatrixBot.__new__(MatrixBot)
    bot.config = MatrixConfig(
        homeserver="https://matrix.local", user_id=AGENT_ID, device_id="TEST",
        password="p", access_token=None, context_reserve=16384,
        sync_timeout=30000, retry_base=1, retry_max=10,
    )
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$r1"))
    bot.client.room_typing = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = set()
    bot._room_effort = {}
    bot._room_cache_ttl = {}
    bot._room_timesense = {}
    bot._halted_rooms = set()
    bot._background_tasks = set()
    bot._session_locks = {}
    bot._degraded_provider_notice = {}
    bot.session_log = SessionLog(ws, AGENT_ID)
    bot.heartbeat = None
    bot.umbral = None
    return bot, agent


def _seed_room(bot, big_chars=60000, count=7, room=ROOM):
    """Seed user + assistant(tool_calls) + tool entries into the JSONL and
    refresh in-memory history (mirrors _seed_room in the gc suite)."""
    log = bot.session_log
    log.wipe(room)
    log.append(role="user", sender=AGENT_ID, room=room, content="go")
    for i in range(count):
        log.append(role="assistant", sender=AGENT_ID, room=room, content="",
                   tool_calls=[{"call_id": f"seed_{i}", "name": "shell",
                                "input": {"command": f"seed {i}"}}])
        log.append(role="tool", sender=AGENT_ID, room=room,
                   call_id=f"seed_{i}", name="shell", output="x" * big_chars)
    h = bot.agent.history(room)
    h.clear()
    h.extend(log.build_context(room))


def _hb_turn(bot, room=ROOM):
    """Drive a real heartbeat-shaped turn (directive persisted first — the
    production kdsn.322.15 shape)."""
    import asyncio
    bot.session_log.append(role="system", sender=AGENT_ID, room=room,
                           content="[Automated heartbeat]",
                           source="heartbeat")
    return asyncio.run(
        bot._run_heartbeat_turn(room, "[Automated heartbeat]",
                                turn_source="heartbeat"))


def _gc_stream(state, *, first_input_tokens=10):
    """Call 1 emits a shell tool call; call 2+ text. Usage shapes carried on
    the done event — this is how the live anchor learns true prompt size."""
    calls = [0]

    async def _stream(*, config=None, system=None, messages=None, tools=None,
                      model="test", thinking=None, cache_ttl=None, **kw):
        calls[0] += 1
        state["stream_calls"] = calls[0]
        state.setdefault("payloads", []).append(list(messages))
        if calls[0] == 1:
            tc = ToolCall(id="anchor_tc_1", name="shell",
                          input={"command": "bloat"})
            yield StreamEvent(type="tool_done", tool_index=0, tool_call=tc)
            yield _done(input_tokens=first_input_tokens, output_tokens=5,
                        stop_reason="tool_use", content="",
                        tool_calls=[tc])
        else:
            yield StreamEvent(type="text", content="DONE")
            yield _done(input_tokens=10, output_tokens=5,
                        stop_reason="end_turn", content="DONE")

    return _stream



def _boundary_markers(bot, room=ROOM):
    return [e for e in bot.session_log.read(room)
            if e.get("role") == "system" and e.get("event") == "handoff_boundary"]


# ---------------------------------------------------------------------------
# A: agent-level anchor mechanics (unit + record seam)
# ---------------------------------------------------------------------------

class TestAnchorMechanics:

    def test_A01_no_anchor_byte_parity(self, tmp_path):
        """Without an anchor, the estimate IS the legacy chars//4 heuristic.
        Pins the no-anchor path contract: merge must not alter it."""
        agent = Agent(make_config(tmp_path))
        agent.history(ROOM).append({"role": "user", "content": "x" * 1000})
        est = agent._estimate_context_tokens(ROOM)
        assert est == agent._context_char_total(ROOM) // 4

    @pytest.mark.asyncio
    async def test_A02_record_seam_sets_full_anchor(self, tmp_path):
        """A live call anchors the room: true prompt =
        input + cache_read + cache_creation; cursor is an int payload char
        count (the pre-call guard's walk). After the call the merge
        dominates a much smaller heuristic."""
        bot, agent = _real_bot(tmp_path)
        with patch("openalph.agent.stream", make_stream_fn([
            [StreamEvent(type="text", content="ok"),
             _done(input_tokens=120000, output_tokens=50, cr=60000, cc=20000)],
        ])):
            await agent.handle_input("hi", ROOM)
        anchor = agent._token_anchor.get(ROOM)
        assert anchor is not None, "record seam must set the anchor"
        tokens0, chars0 = anchor
        assert tokens0 == 120000 + 60000 + 20000 == 200000
        assert isinstance(chars0, int) and chars0 > 0
        # Merge: heuristic here is trivially small; anchored value wins.
        assert agent._estimate_context_tokens(ROOM) >= 200000

    def test_A03_growth_after_anchor_is_marginal_quarters(self, tmp_path):
        """chars//4 of ONLY the growth since the anchor, on top of tokens0.
        heuristic must not win when it contradicts ground truth upward."""
        agent = Agent(make_config(tmp_path))
        agent.history(ROOM).append({"role": "user", "content": "hi"})
        chars0 = agent._context_char_total(ROOM)
        agent._token_anchor[ROOM] = (400000, chars0)
        agent.history(ROOM).append({"role": "user", "content": "x" * 40000})
        # anchored = 400000 + 40000//4 = 410000 — heuristic is nowhere close.
        assert agent._estimate_context_tokens(ROOM) == 410000

    def test_A04_none_and_zero_cache_fields(self, tmp_path):
        """cache fields may be None (or 0) → coerced 0, anchor set from the
        remainder. UNTYPED usage (all zeros) sets NO anchor (would floor 0)."""
        agent = Agent(make_config(tmp_path))
        u = Usage(input_tokens=5, output_tokens=1,
                  cache_read_tokens=None, cache_creation_tokens=None)
        agent._record_turn_usage(ROOM, u, "anthropic/claude-sonnet-4-20250514",
                                 None, pre_call_chars=1000)
        assert agent._token_anchor.get(ROOM) == (5, 1000)
        z = Usage(input_tokens=0, output_tokens=0)
        agent._record_turn_usage(ROOM, z, "anthropic/claude-sonnet-4-20250514",
                                 None, pre_call_chars=2000)
        # Zero true-prompt must NOT trample a valid anchor.
        assert agent._token_anchor.get(ROOM) == (5, 1000)

    def test_A05_record_seam_requires_pre_call_chars(self, tmp_path):
        """No pre_call_chars (continuation/summary paths) → NO anchor update:
        we never anchor without the matching payload char cursor."""
        agent = Agent(make_config(tmp_path))
        u = Usage(input_tokens=300000, output_tokens=1)
        agent._record_turn_usage(ROOM, u, "anthropic/claude-sonnet-4-20250514",
                                 None)
        assert ROOM not in agent._token_anchor

    def test_A06_floor_anchor_merges_as_max(self, tmp_path):
        """Floor-only anchor (chars0=None): estimate = max(heuristic, floor)."""
        agent = Agent(make_config(tmp_path))
        agent.history(ROOM).append({"role": "user", "content": "x" * 4000})
        agent._set_anchor_floor(ROOM, 100000)  # below the auto clamp
        assert agent._token_anchor[ROOM] == (100000, None)
        assert agent._estimate_context_tokens(ROOM) == 100000
        agent._set_anchor_floor(ROOM, 4)  # heuristic (~1000+) beats a tiny floor
        assert agent._estimate_context_tokens(ROOM) >= 4  # floor clamps upward only


# ---------------------------------------------------------------------------
# B: invalidation + rehydration
# ---------------------------------------------------------------------------

class TestAnchorInvalidation:

    def test_B01_reset_room_invalidate(self, tmp_path):
        agent = Agent(make_config(tmp_path))
        agent._token_anchor[ROOM] = (400000, 1000)
        agent.reset_room(ROOM)
        assert ROOM not in agent._token_anchor

    def test_B02_committed_switch_invalidate_failed_retained(self, tmp_path):
        """Committed model switch: anchor out (tokenizer semantics change).
        FAILED switch: anchor retained. The overflow gate consumes the
        ANCHORED estimate (ground truth is what a window check wants) and
        the pop waits for the commit (audit F2/H3)."""
        cfg = make_config(tmp_path, providers={
            "anthropic": make_provider(),
            "openai": make_provider(key="openai", type="openai",
                                    base_url="http://x"),
        })
        agent = Agent(cfg)  # window 200000, max_tokens 8192 → gate ≤ 191808
        agent._token_anchor[ROOM] = (100000, 1000)
        res = agent.switch_model("bogus/model", ROOM)
        assert isinstance(res, str)  # error message; switch refused
        assert agent._token_anchor[ROOM] == (100000, 1000)
        res = agent.switch_model("openai/q", ROOM)
        assert res is None
        assert ROOM not in agent._token_anchor

    def test_B02b_switch_refused_by_overflow_gate_retains_anchor(self, tmp_path):
        """A switch refused by the anchored overflow gate must NOT destroy
        the old model's anchor while we stay on it (audit F2/H3)."""
        cfg = make_config(tmp_path, providers={
            "anthropic": make_provider(),
            "openai": make_provider(key="openai", type="openai",
                                    base_url="http://x"),
        })
        agent = Agent(cfg)  # gate ≤ 191808
        agent._token_anchor[ROOM] = (195000, 1000)  # true prompt > gate
        res = agent.switch_model("openai/q", ROOM)
        assert isinstance(res, str)  # refused — context exceeds window
        assert agent._token_anchor[ROOM] == (195000, 1000)

    def test_B03_boundary_bookkeeping_invalidate(self, tmp_path):
        """_note_handoff_boundary_applied pops the anchor BEFORE its churn
        re-arm estimate — a stale cursor must never survive a rebuild."""
        agent = Agent(make_config(tmp_path))
        agent._token_anchor[ROOM] = (400000, 5_000_000)
        agent._note_handoff_boundary_applied(ROOM, {
            "applied": True,
            "manifest": {"runway": {"tokens_after": 0, "available": 100000}},
        })
        assert ROOM not in agent._token_anchor

    def test_B04_rehydrate_floor_from_jsonl(self, tmp_path):
        """_activate_room restores a FLOOR from the LAST assistant usage entry
        (last-wins), as max(heuristic, floor) via a None cursor."""
        bot, agent = _real_bot(tmp_path)
        log = bot.session_log
        log.append(role="user", sender=AGENT_ID, room=ROOM, content="go")
        log.append(role="assistant", sender=AGENT_ID, room=ROOM, content="r1",
                   usage={"input_tokens": 100, "output_tokens": 5,
                          "cache_read_tokens": 50,
                          "cache_creation_tokens": 10})
        log.append(role="assistant", sender=AGENT_ID, room=ROOM, content="r2",
                   usage={"input_tokens": 200, "output_tokens": 5,
                          "cache_read_tokens": 0,
                          "cache_creation_tokens": 0})
        assert log.last_prompt_tokens(ROOM) == 200
        import asyncio
        asyncio.run(bot._activate_room(ROOM))
        assert agent._token_anchor.get(ROOM) == (200, None)

    def test_B06_floor_clamped_to_auto_threshold(self, tmp_path):
        """Audit H2: the phantom-floor fix. A JSONL floor above the usable
        runway (pre-boundary usage surviving in append-only JSONL) clamps to
        the AUTO threshold — enough to trip the protective boundary
        (handoff on) but never the pre-flight raise (handoff off), so the
        next live call re-anchors instead of wedging the room."""
        agent = Agent(make_config(tmp_path))  # 191808 avail → auto 163036
        agent._set_anchor_floor(ROOM, 999999)
        assert agent._token_anchor[ROOM] == (163036, None)

    def test_B07_phantom_floor_no_wedge_handoff_off(self, tmp_path):
        """The audit-H2 wedge scenario end to end: handoff DISABLED, a
        phantom 235,249 floor in the JSONL (the Florin magnitude), restart
        wake → floor clamps to available (229,376) → the turn runs and the
        live record seam replaces the phantom with real ground truth."""
        from openalph.config import ContextHandoffConfig
        bot, agent = _real_bot(tmp_path, model_max_tokens=262144,
                               max_tokens=32768)
        agent.config.context = ContextHandoffConfig(handoff_enabled=False)
        log = bot.session_log
        log.append(role="user", sender=AGENT_ID, room=ROOM, content="go")
        log.append(role="assistant", sender=AGENT_ID, room=ROOM, content="done",
                   usage={"input_tokens": 235249, "output_tokens": 5,
                          "cache_read_tokens": 0,
                          "cache_creation_tokens": 0})
        state = {}
        with patch("openalph.agent.stream", _gc_stream(state)):
            # MUST NOT raise ContextOverflowError — completion IS the wedge
            # check (heartbeat turns return None by contract).
            _hb_turn(bot)
        # Live re-anchor replaced the phantom floor entirely.
        anchor = agent._token_anchor.get(ROOM)
        assert anchor is not None and anchor[0] <= 1000, anchor

    def test_B05_last_prompt_tokens_fail_soft(self, tmp_path):
        """No usage entries → None; malformed (bool) counters coerced away."""
        bot, _agent = _real_bot(tmp_path)
        bot.session_log.append(role="assistant", sender=AGENT_ID, room=ROOM,
                               content="no-usage")
        assert bot.session_log.last_prompt_tokens(ROOM) is None
        bot.session_log.wipe(ROOM)
        bot.session_log.append(
            role="assistant", sender=AGENT_ID, room=ROOM, content="bad",
            usage={"input_tokens": True, "output_tokens": 1,
                   "cache_read_tokens": True, "cache_creation_tokens": True})
        assert bot.session_log.last_prompt_tokens(ROOM) is None


# ---------------------------------------------------------------------------
# C: real-path threshold proof (the Florin replay)
# ---------------------------------------------------------------------------

class TestAnchorRealPath:

    def test_C01_floor_anchor_fires_turn_start_auto_tier(self, tmp_path):
        """FLORIN REPLAY: heuristic sits BELOW the auto threshold (as in the
        failed session, where the estimate read ~80% usable while the true
        prompt was past it). A floor anchor just above the threshold must
        make the REAL turn-start auto tier fire a boundary."""
        bot, agent = _real_bot(tmp_path)
        _seed_room(bot, big_chars=60000, count=7)
        usable = agent._effective_available(150000)
        auto_th = int(usable * agent.config.context.auto_pct / 100)
        hard_th = int(usable * agent.config.context.hard_pct / 100)
        heuristic = agent._estimate_context_tokens(ROOM)
        assert heuristic < auto_th, (heuristic, auto_th)  # the defect shape
        # Floor anchored just above auto, below hard (true prompt ≠ est).
        agent._set_anchor_floor(ROOM, auto_th + 500)
        assert agent._estimate_context_tokens(ROOM) >= auto_th
        assert agent._estimate_context_tokens(ROOM) < hard_th

        state = {}
        with patch("openalph.agent.stream", _gc_stream(state)):
            _hb_turn(bot)

        marks = _boundary_markers(bot)
        assert len(marks) == 1
        assert _json.loads(marks[0]["detail"])["trigger"] == "auto"
        # Invalidation through _note_handoff_boundary_applied: the floor
        # must not survive the rebuild (the record seam may set a small
        # live anchor from the fake usage — but NEVER the old floor).
        after = agent._token_anchor.get(ROOM, (0, None))
        assert after[0] <= 1000, after

    def test_C01_control_no_anchor_no_boundary(self, tmp_path):
        """Same fixture WITHOUT the anchor: no boundary fires — the C01
        main test is discriminating (proves the anchor is what trips it)."""
        bot, agent = _real_bot(tmp_path)
        _seed_room(bot, big_chars=60000, count=7)
        state = {}
        with patch("openalph.agent.stream", _gc_stream(state)):
            _hb_turn(bot)
        assert _boundary_markers(bot) == []


    def test_C02_hard_tier(self, tmp_path):
        """The live record seam (usage from call 1) anchors mid-turn so the
        PRE-CALL guard trips at the hard threshold when a tool result grows
        the context — even though the chars//4 heuristic stays BELOW it."""
        bot, agent = _real_bot(tmp_path, truncation_limit=200000)
        _seed_room(bot, big_chars=60000, count=7)
        usable = agent._effective_available(150000)
        hard_th = int(usable * agent.config.context.hard_pct / 100)
        heuristic = agent._estimate_context_tokens(ROOM)
        assert heuristic < hard_th

        async def _big_tool(*args, name=None, input=None, **kw):
            return ToolResult(content="y" * 80000, is_error=False)

        state = {}
        with patch("openalph.agent.stream",
                   _gc_stream(state, first_input_tokens=112000)), \
             patch("openalph.agent.execute_tool", _big_tool):
            _hb_turn(bot)

        marks = _boundary_markers(bot)
        assert marks, "hard tier must apply a boundary mid-turn"
        assert _json.loads(marks[-1]["detail"])["trigger"] == "hard"
        assert state.get("stream_calls", 0) >= 2
        # Audit H1: the cursor captured AFTER the hard-tier rebuild pairs
        # the post-boundary payload (shredded, small) — never the stale
        # pre-boundary char total (~500K in this fixture).
        final = agent._token_anchor.get(ROOM)
        assert final is not None
        assert final[1] < 200000, final

    def test_C02_control_heuristic_below_hard_no_boundary(self, tmp_path):
        """Without anchor contributions the in-loop heuristic stays under
        the hard threshold: no boundary. Discriminates C02. (Saboteur for
        this suite: deleting the merge leaves ONLY these controls green.)"""
        bot, agent = _real_bot(tmp_path, truncation_limit=200000)
        _seed_room(bot, big_chars=1000, count=1)  # tiny heuristic ~250

        async def _big_tool(*args, name=None, input=None, **kw):
            return ToolResult(content="y" * 80000, is_error=False)

        usable = agent._effective_available(150000)
        hard_th = int(usable * agent.config.context.hard_pct / 100)
        # heuristic (seed + 80K result) ≈ (81K)//4 ≈ 20K — far below hard.
        assert (agent._estimate_context_tokens(ROOM) + 80000 // 4) < hard_th
        state = {}
        with patch("openalph.agent.stream",
                   _gc_stream(state, first_input_tokens=10)), \
             patch("openalph.agent.execute_tool", _big_tool):
            _hb_turn(bot)
        assert _boundary_markers(bot) == []


# ---------------------------------------------------------------------------
# D: sub-agent loop anchoring
# ---------------------------------------------------------------------------

class TestSubagentAnchor:

    def test_D01_module_function_merge_rules(self):
        """Identical merge semantics in the sub loop's module function."""
        msgs = [{"role": "user", "content": "x" * 200}]  # heuristic = 50
        assert _sub_estimate(msgs) == 50
        assert _sub_estimate(msgs, anchor=None) == 50
        assert _sub_estimate(msgs, anchor=(200000, None)) == 200000
        assert _sub_estimate(msgs, anchor=(100, 100)) == 100 + (200 - 100) // 4
        # shrink without invalidation clamps to the floor (fail-high)
        assert _sub_estimate(msgs, anchor=(50, 300)) == 50

    @pytest.mark.asyncio
    async def test_D02_loop_anchor_fires_boundary_and_resets(self, tmp_path):
        """call 1 usage anchors above the auto threshold while the chars//4
        heuristic stays far below → iteration 2's turn-start tier applies a
        REAL message-list boundary (payload 2 shrinks to the task-only
        carryover), then the anchor resets so no thrash follows."""
        config = make_config(tmp_path, model_max_tokens=10000,
                             max_tokens=8192)  # usable 1808, auto ≈ 1536
        payloads = []

        async def _complete(*, config=None, system=None, messages=None,
                            tools=None, max_tokens=None, thinking=None):
            payloads.append(list(messages))
            if len(payloads) == 1:
                return tool_response(usage_in=2000)
            return text_response()

        with patch("openalph.tools.subagent.complete", _complete):
            result = await run_subagent(
                "anchor-discrimination-task-marker", config)

        assert result.is_error is False
        assert len(payloads) == 2
        # Boundary applied between calls: the second payload carries ONLY
        # the boundary carryover (task text + manifest), not the grown list.
        assert len(payloads[1]) == 1
        assert "anchor-discrimination-task-marker" in payloads[1][0]["content"]

    @pytest.mark.asyncio
    async def test_D02_control_no_anchor_no_boundary(self, tmp_path):
        """Same loop, calls carry tiny usage: heuristic never reaches the
        threshold, the second payload is the GROWN list (no boundary)."""
        config = make_config(tmp_path, model_max_tokens=10000,
                             max_tokens=8192)
        payloads = []

        async def _complete(*, config=None, system=None, messages=None,
                            tools=None, max_tokens=None, thinking=None):
            payloads.append(list(messages))
            if len(payloads) == 1:
                return tool_response(usage_in=10)
            return text_response()

        with patch("openalph.tools.subagent.complete", _complete):
            result = await run_subagent(
                "anchor-discrimination-task-marker", config)

        assert result.is_error is False
        assert len(payloads) == 2
        assert len(payloads[1]) > 1  # assistant + tool result accumulated


def tool_response(usage_in=10):
    return Response(
        content="",
        tool_calls=[ToolCall(id="tc_1", name="file_read",
                             input={"path": "/nonexistent.file"})],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=usage_in, output_tokens=50),
        stop_reason="tool_use",
    )


def text_response():
    return Response(
        content="Sub-agent done", tool_calls=[],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=10, output_tokens=50),
        stop_reason="end_turn",
    )
