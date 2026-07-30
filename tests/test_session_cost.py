"""RED suite — Session USD cost tracking (workspace-kdsn.218).

The tests are the spec. Design: specs/session-cost-tracking-design.md
(incl. Implementation Addendum 2026-07-09). Recon anchors:
specs/cost-tracking-anchors.md.

Scope pinned here:
  - provider.py  : Usage TTL-split fields + _MODEL_PRICING + compute_cost()  (.1)
  - agent.py     : per-room USD counters, freeze in _record_turn_usage,
                   float-safe restore, reset, status()                       (.2)
  - session.py   : usage_totals sums main/subagent/advisor from JSONL        (.3)
  - matrix.py    : serializer persists frozen cost                           (.3)
  - subagent/advisor cost capture via callbacks bridge                       (.4)

Real-path philosophy (tool-management "one lesson"): construct a REAL Agent
and drive the REAL callback path (MatrixBot._build_agent_callbacks); mock ONLY
the provider (LLM). Helpers adapted from test_advisor_integration.py.

Symbols not yet implemented are import-guarded so the FILE ALWAYS COLLECTS;
tests then report as clean FAILED lines (not collection ERRORs).
"""

import asyncio
import pytest
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.provider import Usage, Response, StreamEvent, ToolCall, _parse_anthropic_response
from openalph.config import AgentConfig, ProviderConfig, MatrixConfig
from openalph.agent import Agent
from openalph.session import SessionLog
from openalph.matrix import MatrixBot

# --- Import-guarded (implemented by .1) ----------------------------------
try:
    from openalph.provider import compute_cost, CostResult, _MODEL_PRICING
except ImportError:
    compute_cost = None
    CostResult = None
    _MODEL_PRICING = None

try:
    from openalph.tools.advisor import run_advisor
except ImportError:
    run_advisor = None

try:
    from openalph.tools.subagent import run_subagent
except ImportError:
    run_subagent = None

NOT_IMPL_CALC = "compute_cost not implemented (.1)"
ROOM = "!cost-integ:matrix.local"
AGENT_USER = "@agent:matrix.local"

APPROX = 1e-9  # dollar tolerance for hand-computed fixtures


# =========================================================================
# Helpers
# =========================================================================

def _cfg(workspace, **kw):
    defaults = dict(
        name="cost-test",
        default_model="anthropic/claude-opus-4-8",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[],
        )},
        workspace=workspace,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=1048576,
        matrix=None,
        reminders=True,
        model_aliases={},
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _setup_workspace(tmp_path, tools=("shell", "advisor", "subagent", "file_read")):
    tools_dir = tmp_path / "tools"
    # parents=True: some callers pass a not-yet-created subdir (e.g. tmp_path / "ws")
    tools_dir.mkdir(parents=True, exist_ok=True)
    for name in tools:
        (tools_dir / f"{name}.toml").write_text("[config]\n")
    return tmp_path


def _make_bot(tmp_path, *, real_session_log=False, **agent_kw):
    """MatrixBot with a REAL Agent, mocked nio client. session_log is a real
    SessionLog when real_session_log=True (for persistence/rehydration tests),
    else a MagicMock."""
    ws = _setup_workspace(tmp_path)
    config = _cfg(ws, **agent_kw)
    agent = Agent(config)

    matrix_config = MatrixConfig(
        homeserver="https://matrix.local", user_id=AGENT_USER, device_id="TEST",
        password="test-password", access_token=None, context_reserve=16384,
        sync_timeout=30000, retry_base=1, retry_max=10,
    )
    bot = MatrixBot.__new__(MatrixBot)
    bot.config = matrix_config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$r1"))
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
    bot.heartbeat = MagicMock()
    bot.heartbeat.is_active = MagicMock(return_value=False)
    bot.umbral = MagicMock()
    bot.umbral.is_active = MagicMock(return_value=False)
    bot._steering_inbox = {}
    bot._active_turns = set()
    bot.agent_user_id = AGENT_USER

    if real_session_log:
        bot.session_log = SessionLog(str(tmp_path), AGENT_USER)
    else:
        bot.session_log = MagicMock()
        bot.session_log.append = MagicMock()
        bot.session_log.build_context = MagicMock(return_value=[])
        bot.session_log.read = MagicMock(return_value=[])
        bot.session_log.usage_totals = MagicMock(return_value={})
    return bot, agent


def _single_turn_stream(usage, model="claude-opus-4-8", text="done"):
    """Executor stream: one API call, no tools, carrying `usage`.

    NOTE: `resp_model` (the closure's bare model string) is always used to
    build the Response/StreamEvent, regardless of what `model=` kwarg the
    real call site passes in (e.g. agent.py passes self.get_model(room_id),
    which may be a fully-qualified "provider/model" routing string). This
    mirrors real SDK behavior: the Anthropic SDK's final_message.model is
    always the bare API model string, never our internal routing prefix
    (see provider.py's _parse_anthropic_response, which uses response.model
    verbatim from the SDK object) -- so the mock must not echo the routing
    kwarg back as if it were the SDK's returned model.
    """
    resp_model = model
    async def _stream(*, config=None, system=None, messages=None, tools=None,
                      model=None, thinking=None, cache_ttl=None, **kw):
        yield StreamEvent(type="text", content=text)
        yield StreamEvent(
            type="done",
            response=Response(content=text, model=resp_model, usage=usage,
                              stop_reason="end_turn"),
            stop_reason="end_turn", model=resp_model)
    return _stream


def _u(**kw):
    """Usage builder tolerating the new split fields (fails cleanly pre-.1)."""
    return Usage(**kw)


# =========================================================================
# .1 — Pricing table
# =========================================================================

class TestPricingTable:
    def test_table_has_fleet_models_with_verified_rates(self):
        assert _MODEL_PRICING is not None, NOT_IMPL_CALC
        exp = {
            "claude-opus-4-8": (5.0, 25.0),
            "claude-opus-4-7": (5.0, 25.0),
            "claude-sonnet-4-6": (3.0, 15.0),
            "claude-haiku-4-5": (1.0, 5.0),
            "claude-fable-5": (10.0, 50.0),
        }
        anthropic_table = _MODEL_PRICING["anthropic"]
        for model, (inp, out) in exp.items():
            assert model in anthropic_table, f"{model} missing from pricing table"
            entry = anthropic_table[model]
            # entry may carry effective-date structure; a plain-rate accessor
            # is exercised via compute_cost — here just assert the base rates
            # are discoverable as numbers somewhere in the entry.
            flat = repr(entry)
            assert str(inp) in flat and str(out) in flat


# =========================================================================
# .1 — compute_cost calculator (hand-computed fixtures)
# =========================================================================

class TestComputeCost:
    def test_opus_mixed_cache(self):
        assert compute_cost is not None, NOT_IMPL_CALC
        # opus in=5/out=25; read=0.5; 1h write=10 (per MTok)
        usage = _u(input_tokens=10_000, output_tokens=5_000, cache_read_tokens=40_000,
                   cache_creation_tokens=20_000, cache_creation_5m_tokens=0,
                   cache_creation_1h_tokens=20_000)
        r = compute_cost("claude-opus-4-8", usage)
        # 0.05 + 0.125 + 0.02 + 0.20 = 0.395
        assert r.priced is True
        assert r.unpriced_tokens == 0
        assert r.cost_usd == pytest.approx(0.395, abs=APPROX)

    def test_sonnet46_5m_write(self):
        assert compute_cost is not None, NOT_IMPL_CALC
        usage = _u(input_tokens=1_000, output_tokens=2_000, cache_read_tokens=0,
                   cache_creation_tokens=5_000, cache_creation_5m_tokens=5_000,
                   cache_creation_1h_tokens=0)
        r = compute_cost("claude-sonnet-4-6", usage)
        # 0.003 + 0.03 + 5000*3.75/1e6(0.01875) = 0.05175
        assert r.cost_usd == pytest.approx(0.05175, abs=APPROX)

    def test_fable_all_classes(self):
        assert compute_cost is not None, NOT_IMPL_CALC
        usage = _u(input_tokens=1_000, output_tokens=1_000, cache_read_tokens=10_000,
                   cache_creation_tokens=5_000, cache_creation_5m_tokens=2_000,
                   cache_creation_1h_tokens=3_000)
        r = compute_cost("claude-fable-5", usage)
        # 0.01 + 0.05 + 0.01 + 2000*12.5/1e6(0.025) + 3000*20/1e6(0.06) = 0.155
        assert r.cost_usd == pytest.approx(0.155, abs=APPROX)

    def test_provider_prefix_normalized(self):
        assert compute_cost is not None, NOT_IMPL_CALC
        usage = _u(input_tokens=100_000, output_tokens=50_000)
        r = compute_cost("anthropic/claude-opus-4-8", usage)
        # 0.5 + 1.25 = 1.75
        assert r.priced is True
        assert r.cost_usd == pytest.approx(1.75, abs=APPROX)

    def test_haiku_date_suffix_normalized(self):
        assert compute_cost is not None, NOT_IMPL_CALC
        usage = _u(input_tokens=100_000, output_tokens=50_000)
        r = compute_cost("claude-haiku-4-5-20251001", usage)
        # 0.1 + 0.25 = 0.35
        assert r.priced is True
        assert r.cost_usd == pytest.approx(0.35, abs=APPROX)

    def test_sonnet5_effective_date_intro_vs_standard(self):
        assert compute_cost is not None, NOT_IMPL_CALC
        usage = _u(input_tokens=1_000_000, output_tokens=1_000_000)
        intro = compute_cost("claude-sonnet-5", usage, now=date(2026, 8, 31))
        std = compute_cost("claude-sonnet-5", usage, now=date(2026, 9, 1))
        assert intro.cost_usd == pytest.approx(12.0, abs=APPROX)   # 2 + 10
        assert std.cost_usd == pytest.approx(18.0, abs=APPROX)     # 3 + 15

    def test_unpriced_non_anthropic(self):
        assert compute_cost is not None, NOT_IMPL_CALC
        usage = _u(input_tokens=1_000, output_tokens=2_000, cache_read_tokens=500,
                   cache_creation_tokens=100)
        r = compute_cost("codex/gpt-5.5", usage)
        assert r.priced is False
        assert r.cost_usd == 0.0
        assert r.unpriced_tokens == 3_600  # 1000+2000+500+100

    def test_unpriced_unlisted_anthropic(self):
        assert compute_cost is not None, NOT_IMPL_CALC
        usage = _u(input_tokens=1_000, output_tokens=1_000)
        r = compute_cost("claude-unobtainium-9", usage)
        assert r.priced is False
        assert r.unpriced_tokens == 2_000

    def test_aggregate_fallback_no_split(self):
        assert compute_cost is not None, NOT_IMPL_CALC
        # split both None but aggregate present -> cost at fallback multiplier
        usage = _u(input_tokens=0, output_tokens=0, cache_read_tokens=0,
                   cache_creation_tokens=8_000, cache_creation_5m_tokens=None,
                   cache_creation_1h_tokens=None)
        r1h = compute_cost("claude-opus-4-8", usage, cache_ttl_fallback="1h")
        r5m = compute_cost("claude-opus-4-8", usage, cache_ttl_fallback="5m")
        # 1h: 8000 * (5*2)/1e6 = 0.08 ; 5m: 8000 * (5*1.25)/1e6 = 0.05
        assert r1h.cost_usd == pytest.approx(0.08, abs=APPROX)
        assert r5m.cost_usd == pytest.approx(0.05, abs=APPROX)

    def test_zero_usage_priced_model_is_zero(self):
        assert compute_cost is not None, NOT_IMPL_CALC
        r = compute_cost("claude-opus-4-8", _u(input_tokens=0, output_tokens=0))
        assert r.priced is True
        assert r.cost_usd == 0.0
        assert r.unpriced_tokens == 0


# =========================================================================
# .1 — Usage split fields populated by _parse_anthropic_response
# =========================================================================

class TestUsageSplitFields:
    def _fake_resp(self, *, cc_5m, cc_1h, cc_agg=None, model="claude-opus-4-8"):
        if cc_5m is None and cc_1h is None:
            cache_creation = None
        else:
            cache_creation = SimpleNamespace(
                ephemeral_5m_input_tokens=cc_5m or 0,
                ephemeral_1h_input_tokens=cc_1h or 0)
        usage = SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=3,
            cache_creation_input_tokens=(cc_agg if cc_agg is not None
                                         else (cc_5m or 0) + (cc_1h or 0)),
            cache_creation=cache_creation)
        return SimpleNamespace(content=[], usage=usage, model=model,
                               stop_reason="end_turn", id="msg_1")

    def test_split_populated_from_cache_creation(self):
        resp = _parse_anthropic_response(self._fake_resp(cc_5m=100, cc_1h=250))
        assert resp.usage.cache_creation_5m_tokens == 100
        assert resp.usage.cache_creation_1h_tokens == 250
        # aggregate preserved
        assert resp.usage.cache_creation_tokens == 350

    def test_split_none_when_no_cache_creation(self):
        resp = _parse_anthropic_response(self._fake_resp(cc_5m=None, cc_1h=None, cc_agg=0))
        assert resp.usage.cache_creation_5m_tokens in (None, 0)
        assert resp.usage.cache_creation_1h_tokens in (None, 0)


# =========================================================================
# .2 — agent per-room cost counters
# =========================================================================

class TestAgentCostCounters:
    def _agent(self, tmp_path):
        return Agent(_cfg(_setup_workspace(tmp_path)))

    def test_record_turn_usage_freezes_and_accumulates(self, tmp_path):
        assert compute_cost is not None, NOT_IMPL_CALC
        agent = self._agent(tmp_path)
        usage = _u(input_tokens=10_000, output_tokens=5_000, cache_read_tokens=40_000,
                   cache_creation_tokens=20_000, cache_creation_5m_tokens=0,
                   cache_creation_1h_tokens=20_000)
        agent._record_turn_usage(ROOM, usage, "claude-opus-4-8", "1h")
        room_u = agent._usage_for(ROOM)
        assert room_u["main_cost_usd"] == pytest.approx(0.395, abs=APPROX)
        assert room_u["unpriced_tokens"] == 0
        # frozen per-turn delta available to serializer
        lt = agent.last_turn_usage(ROOM)
        assert lt["cost_usd"] == pytest.approx(0.395, abs=APPROX)
        assert lt["model"] == "claude-opus-4-8"
        assert lt["cache_creation_1h"] == 20_000

    def test_record_turn_usage_accumulates_across_calls(self, tmp_path):
        assert compute_cost is not None, NOT_IMPL_CALC
        agent = self._agent(tmp_path)
        u1 = _u(input_tokens=100_000, output_tokens=50_000)  # 1.75 opus
        agent._record_turn_usage(ROOM, u1, "claude-opus-4-8", "1h")
        agent._record_turn_usage(ROOM, u1, "claude-opus-4-8", "1h")
        assert agent._usage_for(ROOM)["main_cost_usd"] == pytest.approx(3.5, abs=APPROX)

    def test_record_turn_usage_unpriced_model(self, tmp_path):
        assert compute_cost is not None, NOT_IMPL_CALC
        agent = self._agent(tmp_path)
        usage = _u(input_tokens=1_000, output_tokens=2_000)
        agent._record_turn_usage(ROOM, usage, "macstudio/gemma-4-31b", "1h")
        room_u = agent._usage_for(ROOM)
        assert room_u["main_cost_usd"] == 0.0
        assert room_u["unpriced_tokens"] == 3_000

    def test_usage_for_initializes_cost_keys(self, tmp_path):
        agent = self._agent(tmp_path)
        u = agent._usage_for(ROOM)
        for k in ("main_cost_usd", "subagent_cost_usd", "advisor_cost_usd"):
            assert u[k] == 0.0
        assert u["unpriced_tokens"] == 0

    def test_restore_usage_is_float_safe(self, tmp_path):
        agent = self._agent(tmp_path)
        totals = {
            "uncached_input_tokens": 10, "cache_read_tokens": 20,
            "cache_creation_tokens": 30, "total_output_tokens": 40,
            "total_tool_calls": 2,
            "main_cost_usd": 1.2345, "subagent_cost_usd": 0.5,
            "advisor_cost_usd": 0.0678, "unpriced_tokens": 700,
        }
        agent.restore_usage(ROOM, totals)
        u = agent._usage_for(ROOM)
        assert u["main_cost_usd"] == pytest.approx(1.2345, abs=APPROX)  # NOT int-truncated
        assert u["subagent_cost_usd"] == pytest.approx(0.5, abs=APPROX)
        assert u["advisor_cost_usd"] == pytest.approx(0.0678, abs=APPROX)
        assert u["unpriced_tokens"] == 700

    def test_reset_room_clears_cost(self, tmp_path):
        assert compute_cost is not None, NOT_IMPL_CALC
        agent = self._agent(tmp_path)
        agent._record_turn_usage(ROOM, _u(input_tokens=100_000, output_tokens=50_000),
                                 "claude-opus-4-8", "1h")
        assert agent._usage_for(ROOM)["main_cost_usd"] > 0
        agent.reset_room(ROOM)
        assert agent._usage_for(ROOM)["main_cost_usd"] == 0.0

    def test_status_exposes_cost_and_derived_total(self, tmp_path):
        assert compute_cost is not None, NOT_IMPL_CALC
        agent = self._agent(tmp_path)
        u = agent._usage_for(ROOM)
        u["main_cost_usd"] = 1.0
        u["subagent_cost_usd"] = 2.0
        u["advisor_cost_usd"] = 0.25
        u["unpriced_tokens"] = 123
        st = agent.status(ROOM)
        assert st["main_cost_usd"] == pytest.approx(1.0, abs=APPROX)
        assert st["subagent_cost_usd"] == pytest.approx(2.0, abs=APPROX)
        assert st["advisor_cost_usd"] == pytest.approx(0.25, abs=APPROX)
        assert st["total_cost_usd"] == pytest.approx(3.25, abs=APPROX)
        assert st["unpriced_tokens"] == 123


# =========================================================================
# .3 — session.usage_totals sums main / subagent / advisor from JSONL
# =========================================================================

class TestUsageTotalsRehydration:
    def test_usage_totals_sums_three_cost_sources(self, tmp_path):
        sl = SessionLog(str(tmp_path), AGENT_USER)
        # main: two assistant turns w/ frozen cost in the usage dict
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM, content="a",
                  usage={"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 0,
                         "cache_creation_tokens": 0, "tool_calls": 0,
                         "cost_usd": 0.10, "model": "claude-opus-4-8",
                         "cache_creation_5m": 0, "cache_creation_1h": 0,
                         "unpriced_tokens": 0})
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM, content="b",
                  usage={"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 0,
                         "cache_creation_tokens": 0, "tool_calls": 0,
                         "cost_usd": 0.05, "model": "claude-opus-4-8",
                         "cache_creation_5m": 0, "cache_creation_1h": 0,
                         "unpriced_tokens": 0})
        # subagent: tool-result entry carrying frozen sub cost
        sl.append(role="tool", sender=AGENT_USER, room=ROOM, call_id="tc_s",
                  name="subagent", output="sub done", is_error=False,
                  cost_usd=0.20, unpriced_tokens=0)
        # advisor: system entry
        sl.append(role="system", sender=AGENT_USER, room=ROOM,
                  event="advisor_consult", model="claude-opus-4-8",
                  input_tokens=100, output_tokens=50, cache_read_tokens=0,
                  elapsed_s=1.0, cost_usd=0.03, unpriced_tokens=0)

        totals = sl.usage_totals(ROOM)
        assert totals["main_cost_usd"] == pytest.approx(0.15, abs=APPROX)
        assert totals["subagent_cost_usd"] == pytest.approx(0.20, abs=APPROX)
        assert totals["advisor_cost_usd"] == pytest.approx(0.03, abs=APPROX)

    def test_usage_totals_accumulates_unpriced(self, tmp_path):
        sl = SessionLog(str(tmp_path), AGENT_USER)
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM, content="a",
                  usage={"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0,
                         "unpriced_tokens": 500})
        sl.append(role="tool", sender=AGENT_USER, room=ROOM, call_id="tc_s",
                  name="subagent", output="x", is_error=False,
                  cost_usd=0.0, unpriced_tokens=200)
        totals = sl.usage_totals(ROOM)
        assert totals["unpriced_tokens"] == 700

    def test_restore_from_usage_totals_roundtrip(self, tmp_path):
        assert compute_cost is not None, NOT_IMPL_CALC
        sl = SessionLog(str(tmp_path), AGENT_USER)
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM, content="a",
                  usage={"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.42,
                         "unpriced_tokens": 0})
        agent = Agent(_cfg(_setup_workspace(tmp_path / "ws")))
        agent.restore_usage(ROOM, sl.usage_totals(ROOM))
        assert agent.status(ROOM)["main_cost_usd"] == pytest.approx(0.42, abs=APPROX)


# =========================================================================
# .2 + .3 — real-path: turn -> freeze -> persist -> rehydrate -> status
# =========================================================================

class TestRealPathMainCost:
    @pytest.mark.asyncio
    async def test_real_turn_freezes_cost_in_room_usage(self, tmp_path):
        assert compute_cost is not None, NOT_IMPL_CALC
        bot, agent = _make_bot(tmp_path)
        cb = bot._build_agent_callbacks(ROOM, None)
        usage = _u(input_tokens=100_000, output_tokens=50_000)  # opus -> 1.75
        with patch("openalph.agent.stream", side_effect=_single_turn_stream(usage)):
            await agent.handle_input("hi", ROOM, callbacks=cb, cache_ttl="1h")
        assert agent._usage_for(ROOM)["main_cost_usd"] == pytest.approx(1.75, abs=APPROX)
        assert agent.last_turn_usage(ROOM)["cost_usd"] == pytest.approx(1.75, abs=APPROX)

    @pytest.mark.asyncio
    async def test_real_turn_cost_persisted_to_jsonl(self, tmp_path):
        assert compute_cost is not None, NOT_IMPL_CALC
        bot, agent = _make_bot(tmp_path, real_session_log=True)
        cb = bot._build_agent_callbacks(ROOM, None)
        usage = _u(input_tokens=100_000, output_tokens=50_000)
        with patch("openalph.agent.stream", side_effect=_single_turn_stream(usage)):
            await agent.handle_input("hi", ROOM, callbacks=cb, cache_ttl="1h")
        bot._persist_assistant_turn(ROOM, content="done")
        entries = [e for e in bot.session_log.read(ROOM) if e.get("role") == "assistant"]
        assert entries, "no assistant entry persisted"
        u = entries[-1].get("usage")
        assert isinstance(u, dict)
        assert u.get("cost_usd") == pytest.approx(1.75, abs=APPROX)
        assert u.get("model") == "claude-opus-4-8"

    @pytest.mark.asyncio
    async def test_context_status_carries_cost_fields(self, tmp_path):
        assert compute_cost is not None, NOT_IMPL_CALC
        bot, agent = _make_bot(tmp_path)
        cb = bot._build_agent_callbacks(ROOM, None)
        usage = _u(input_tokens=100_000, output_tokens=50_000)
        with patch("openalph.agent.stream", side_effect=_single_turn_stream(usage)):
            await agent.handle_input("hi", ROOM, callbacks=cb, cache_ttl="1h")
        cs = bot._build_context_status(ROOM)
        assert cs["main_cost_usd"] == pytest.approx(1.75, abs=APPROX)
        assert cs["total_cost_usd"] == pytest.approx(1.75, abs=APPROX)


# =========================================================================
# .4 — advisor cost capture (bridge dict carries frozen cost)
# =========================================================================

class TestAdvisorCost:
    @pytest.mark.asyncio
    async def test_advisor_bridge_carries_cost(self, tmp_path):
        assert run_advisor is not None, "run_advisor not importable"
        assert compute_cost is not None, NOT_IMPL_CALC
        config = _cfg(_setup_workspace(tmp_path))
        advice_resp = Response(
            content="advice", model="claude-opus-4-8",
            usage=Usage(input_tokens=1_000, output_tokens=120, cache_read_tokens=42),
            stop_reason="end_turn")
        callbacks = {
            "advisor_uses": {}, "room_id": ROOM, "call_id": "tc_adv",
            "advisor_results": {}, "get_transcript": lambda: ("sys", []),
        }
        tool_config = {"model": "anthropic/claude-opus-4-8", "max_uses": 10,
                       "max_tokens": 8192, "thinking": "medium", "cache_ttl": "5m",
                       "include_system_prompt": True, "transcript_max_chars": 0,
                       "timeout": 300}
        with patch("openalph.tools.advisor.complete",
                   new=AsyncMock(return_value=advice_resp)):
            await run_advisor(focus="q", model="anthropic/claude-opus-4-8",
                              config=config, tool_config=tool_config,
                              callbacks=callbacks)
        info = callbacks["advisor_results"].get((ROOM, "tc_adv"))
        assert info is not None, "advisor bridge entry missing"
        # 1000*5/1e6 + 120*25/1e6 + 42*0.5/1e6 = 0.008021
        assert info["cost_usd"] == pytest.approx(0.008021, abs=APPROX)


# =========================================================================
# .4 — subagent cost capture (callbacks bridge)
# =========================================================================

class TestSubagentCost:
    @pytest.mark.asyncio
    async def test_subagent_writes_cost_to_bridge(self, tmp_path):
        assert run_subagent is not None, "run_subagent not importable"
        assert compute_cost is not None, NOT_IMPL_CALC
        config = _cfg(_setup_workspace(tmp_path))
        sub_resp = Response(
            content="sub done", model="claude-opus-4-8",
            usage=Usage(input_tokens=200_000, output_tokens=100_000),
            stop_reason="end_turn", tool_calls=[])
        callbacks = {"subagent_results": {}, "room_id": ROOM}
        with patch("openalph.tools.subagent.complete",
                   new=AsyncMock(return_value=sub_resp)):
            await run_subagent(task="do a thing", config=config, tools=[],
                               call_id="tc_sub", parent_room_id=ROOM,
                               callbacks=callbacks)
        info = callbacks["subagent_results"].get((ROOM, "tc_sub"))
        assert info is not None, "subagent bridge entry missing"
        # 200000*5/1e6 + 100000*25/1e6 = 3.5
        assert info["cost_usd"] == pytest.approx(3.5, abs=APPROX)


# =========================================================================
# .4 wiring — REAL-PATH: bridge -> parent JSONL entry (the "one lesson":
# test the wired path through handle_input + _make_tool_callbacks, NOT the
# bridge seam directly). Mirrors test_advisor_integration.py's tool round-trip.
# =========================================================================

def _tool_roundtrip_stream(tool_name, tool_input):
    """Executor stream: emit `tool_name` tool_call on iteration 1, finish on 2."""
    idx = [0]

    async def _stream(*, config=None, system=None, messages=None, tools=None,
                      model="claude-opus-4-8", thinking=None, cache_ttl=None, **kw):
        idx[0] += 1
        if idx[0] == 1:
            tc = ToolCall(id=f"{tool_name}_1", name=tool_name, input=tool_input)
            yield StreamEvent(
                type="done", model="claude-opus-4-8", stop_reason="tool_use",
                response=Response(content="", tool_calls=[tc], model="claude-opus-4-8",
                                  usage=Usage(input_tokens=10, output_tokens=5),
                                  stop_reason="tool_use"))
        else:
            yield StreamEvent(
                type="done", model="claude-opus-4-8", stop_reason="end_turn",
                response=Response(content="fin", model="claude-opus-4-8",
                                  usage=Usage(input_tokens=10, output_tokens=5),
                                  stop_reason="end_turn"))
    return _stream


class TestJSONLWiring:
    @pytest.mark.asyncio
    async def test_advisor_consult_jsonl_carries_cost(self, tmp_path):
        assert run_advisor is not None, "run_advisor not importable"
        assert compute_cost is not None, NOT_IMPL_CALC
        bot, agent = _make_bot(tmp_path)
        _tool_notice, _tool_intent = bot._make_tool_callbacks(ROOM)
        cb = bot._build_agent_callbacks(ROOM, None)
        advice = Response(content="advice", model="claude-opus-4-8",
                          usage=Usage(input_tokens=1_000, output_tokens=120,
                                      cache_read_tokens=42),
                          stop_reason="end_turn")
        with patch("openalph.agent.stream",
                   side_effect=_tool_roundtrip_stream(
                       "advisor", {"focus": "x", "model": "anthropic/claude-opus-4-8"})), \
             patch("openalph.tools.advisor.complete",
                   new=AsyncMock(return_value=advice)):
            await agent.handle_input("work", ROOM, callbacks=cb,
                                     on_tool_call=_tool_notice, on_tool_intent=_tool_intent)
        sys_calls = [c for c in bot.session_log.append.call_args_list
                     if c.kwargs.get("event") == "advisor_consult"]
        assert sys_calls, "advisor_consult JSONL entry must be written"
        # 1000*5/1e6 + 120*25/1e6 + 42*0.5/1e6 = 0.008021
        assert sys_calls[0].kwargs.get("cost_usd") == pytest.approx(0.008021, abs=APPROX)

    @pytest.mark.asyncio
    async def test_subagent_jsonl_carries_cost(self, tmp_path):
        assert run_subagent is not None, "run_subagent not importable"
        assert compute_cost is not None, NOT_IMPL_CALC
        bot, agent = _make_bot(tmp_path)
        _tool_notice, _tool_intent = bot._make_tool_callbacks(ROOM)
        cb = bot._build_agent_callbacks(ROOM, None)
        sub_resp = Response(content="done", model="claude-opus-4-8",
                            usage=Usage(input_tokens=200_000, output_tokens=100_000),
                            stop_reason="end_turn", tool_calls=[])
        with patch("openalph.agent.stream",
                   side_effect=_tool_roundtrip_stream("subagent", {"task": "do it"})), \
             patch("openalph.tools.subagent.complete",
                   new=AsyncMock(return_value=sub_resp)):
            await agent.handle_input("work", ROOM, callbacks=cb,
                                     on_tool_call=_tool_notice, on_tool_intent=_tool_intent)
        tool_calls = [c for c in bot.session_log.append.call_args_list
                      if c.kwargs.get("role") == "tool"
                      and c.kwargs.get("name") == "subagent"]
        assert tool_calls, "subagent tool-result JSONL entry must be written"
        # 200000*5/1e6 + 100000*25/1e6 = 3.5
        assert tool_calls[0].kwargs.get("cost_usd") == pytest.approx(3.5, abs=APPROX)


# =========================================================================
# REMEDIATION SUITE (kdsn.218 audit reconciliation, 2026-07-09)
# F1 alias · F2 provider gate · F3 fail-soft · F4 live accrual · F6 nested
# advisor · F8 partial-split. See tmp/code-audit/cost/AUDIT-REPORT.md.
# =========================================================================

# --- F8: partial / present-but-zero cache-creation split -----------------
class TestPartialSplitResidual:
    def test_present_but_zero_split_uses_aggregate_fallback(self):
        """Both split fields present but 0 while aggregate>0 must fall back to
        the aggregate (pre-fix: else-branch billed $0 for the write)."""
        assert compute_cost is not None, NOT_IMPL_CALC
        usage = _u(input_tokens=0, output_tokens=0, cache_read_tokens=0,
                   cache_creation_tokens=8_000, cache_creation_5m_tokens=0,
                   cache_creation_1h_tokens=0)
        r = compute_cost("claude-opus-4-8", usage, cache_ttl_fallback="1h")
        # 8000 * 5 * 2.0 / 1e6 = 0.08
        assert r.cost_usd == pytest.approx(0.08, abs=APPROX)

    def test_split_undercounts_aggregate_residual_costed(self):
        """split_sum < aggregate (SDK inconsistency): residual billed at
        fallback multiplier, not silently dropped."""
        assert compute_cost is not None, NOT_IMPL_CALC
        usage = _u(input_tokens=0, output_tokens=0, cache_read_tokens=0,
                   cache_creation_tokens=10_000, cache_creation_5m_tokens=4_000,
                   cache_creation_1h_tokens=0)
        r = compute_cost("claude-opus-4-8", usage, cache_ttl_fallback="1h")
        # 4000 @ 5m(1.25): 4000*5*1.25/1e6 = 0.025
        # residual 6000 @ 1h fallback(2.0): 6000*5*2/1e6 = 0.06 -> 0.085
        assert r.cost_usd == pytest.approx(0.085, abs=APPROX)


# --- F2: provider-type gate ----------------------------------------------
class TestProviderTypeGate:
    def test_is_anthropic_false_forces_unpriced(self):
        """A claude-shaped model served by a non-Anthropic provider must be
        tallied unpriced, never Anthropic-priced (scope: Anthropic-only)."""
        assert compute_cost is not None, NOT_IMPL_CALC
        usage = _u(input_tokens=100_000, output_tokens=50_000)
        r = compute_cost("claude-opus-4-8", usage,
                         provider_key="fireworks", provider_type="openai")
        assert r.priced is False
        assert r.cost_usd == 0.0
        assert r.unpriced_tokens == 150_000

    def test_is_anthropic_true_prices_normally(self):
        assert compute_cost is not None, NOT_IMPL_CALC
        usage = _u(input_tokens=100_000, output_tokens=50_000)
        r = compute_cost("claude-opus-4-8", usage,
                         provider_key="anthropic", provider_type="anthropic")
        assert r.priced is True
        assert r.cost_usd == pytest.approx(1.75, abs=APPROX)

    def test_none_default_preserves_string_shape_classification(self):
        """Legacy/pure-fn callers (both gate kwargs absent) keep string-shape
        classification so existing unit fixtures are unaffected."""
        assert compute_cost is not None, NOT_IMPL_CALC
        r = compute_cost("claude-opus-4-8", _u(input_tokens=100_000, output_tokens=50_000))
        assert r.priced is True and r.cost_usd == pytest.approx(1.75, abs=APPROX)

    def test_agent_provider_is_anthropic_helper(self, tmp_path):
        agent = Agent(_cfg(_setup_workspace(tmp_path)))
        assert agent._provider_gate("anthropic/claude-opus-4-8") == ("anthropic", "anthropic")
        # unknown model / resolution failure -> (None, None) (fail-soft, never mispriced)
        assert agent._provider_gate("nonesuch/whatever") == (None, None)


# --- F1: alias priced from served model + F2 non-Anthropic sub -----------
class TestSubagentAliasAndGate:
    @pytest.mark.asyncio
    async def test_subagent_alias_model_priced_from_served_model(self, tmp_path):
        """A bare-alias sub model (headline use case) must be priced from the
        SDK-served model, not $0-unpriced (pre-fix: compute_cost('opus') miss)."""
        assert run_subagent is not None and compute_cost is not None
        config = _cfg(_setup_workspace(tmp_path),
                      default_model="opus",
                      model_aliases={"opus": "anthropic/claude-opus-4-8"})
        sub_resp = Response(content="done", model="claude-opus-4-8",
                            usage=Usage(input_tokens=200_000, output_tokens=100_000),
                            stop_reason="end_turn", tool_calls=[])
        callbacks = {"subagent_results": {}, "room_id": ROOM}
        with patch("openalph.tools.subagent.complete", new=AsyncMock(return_value=sub_resp)):
            await run_subagent(task="x", config=config, tools=[],
                               call_id="tc_sub", parent_room_id=ROOM, callbacks=callbacks)
        info = callbacks["subagent_results"][(ROOM, "tc_sub")]
        assert info["cost_usd"] == pytest.approx(3.5, abs=APPROX)
        assert info["unpriced_tokens"] == 0

    @pytest.mark.asyncio
    async def test_subagent_non_anthropic_claude_shaped_is_unpriced(self, tmp_path):
        """A non-Anthropic (openai-type) provider returning a claude-shaped id
        must be tallied unpriced, not Anthropic-priced."""
        assert run_subagent is not None and compute_cost is not None
        config = _cfg(_setup_workspace(tmp_path),
                      default_model="router/claude-opus-4-8",
                      providers={"router": ProviderConfig(
                          key="router", type="openai", api_key="sk-x",
                          base_url="https://router.example/v1", quirks=[])})
        sub_resp = Response(content="done", model="claude-opus-4-8",
                            usage=Usage(input_tokens=200_000, output_tokens=100_000),
                            stop_reason="end_turn", tool_calls=[])
        callbacks = {"subagent_results": {}, "room_id": ROOM}
        with patch("openalph.tools.subagent.complete", new=AsyncMock(return_value=sub_resp)):
            await run_subagent(task="x", config=config, tools=[],
                               call_id="tc_sub", parent_room_id=ROOM, callbacks=callbacks)
        info = callbacks["subagent_results"][(ROOM, "tc_sub")]
        assert info["cost_usd"] == 0.0
        assert info["unpriced_tokens"] == 300_000


# --- F4: subagent/advisor cost accrues LIVE in status() ------------------
class TestLiveAccumulation:
    @pytest.mark.asyncio
    async def test_subagent_cost_accrues_live_in_status(self, tmp_path):
        bot, agent = _make_bot(tmp_path)
        _tool_notice, _tool_intent = bot._make_tool_callbacks(ROOM)
        cb = bot._build_agent_callbacks(ROOM, None)
        sub_resp = Response(content="done", model="claude-opus-4-8",
                            usage=Usage(input_tokens=200_000, output_tokens=100_000),
                            stop_reason="end_turn", tool_calls=[])
        with patch("openalph.agent.stream",
                   side_effect=_tool_roundtrip_stream("subagent", {"task": "do it"})), \
             patch("openalph.tools.subagent.complete", new=AsyncMock(return_value=sub_resp)):
            await agent.handle_input("work", ROOM, callbacks=cb,
                                     on_tool_call=_tool_notice, on_tool_intent=_tool_intent)
        # LIVE (no restart / no rehydrate): status must already reflect sub spend
        assert agent.status(ROOM)["subagent_cost_usd"] == pytest.approx(3.5, abs=APPROX)
        assert agent.status(ROOM)["total_cost_usd"] >= 3.5

    @pytest.mark.asyncio
    async def test_advisor_cost_accrues_live_in_status(self, tmp_path):
        bot, agent = _make_bot(tmp_path)
        _tool_notice, _tool_intent = bot._make_tool_callbacks(ROOM)
        cb = bot._build_agent_callbacks(ROOM, None)
        advice = Response(content="advice", model="claude-opus-4-8",
                          usage=Usage(input_tokens=1_000, output_tokens=120,
                                      cache_read_tokens=42),
                          stop_reason="end_turn")
        with patch("openalph.agent.stream",
                   side_effect=_tool_roundtrip_stream(
                       "advisor", {"focus": "x", "model": "anthropic/claude-opus-4-8"})), \
             patch("openalph.tools.advisor.complete", new=AsyncMock(return_value=advice)):
            await agent.handle_input("work", ROOM, callbacks=cb,
                                     on_tool_call=_tool_notice, on_tool_intent=_tool_intent)
        # 1000*5/1e6 + 120*25/1e6 + 42*0.5/1e6 = 0.008021
        assert agent.status(ROOM)["advisor_cost_usd"] == pytest.approx(0.008021, abs=APPROX)


# --- F5: advisor unpriced_tokens persisted to JSONL ----------------------
class TestAdvisorUnpricedPersisted:
    @pytest.mark.asyncio
    async def test_advisor_unpriced_written_to_jsonl(self, tmp_path):
        bot, agent = _make_bot(tmp_path)
        _tool_notice, _tool_intent = bot._make_tool_callbacks(ROOM)
        cb = bot._build_agent_callbacks(ROOM, None)
        # advisor served by a non-Anthropic provider alias -> unpriced tally
        agent.config.model_aliases["router"] = "router/claude-opus-4-8"
        agent.config.providers["router"] = ProviderConfig(
            key="router", type="openai", api_key="sk-x",
            base_url="https://router.example/v1", quirks=[])
        advice = Response(content="advice", model="claude-opus-4-8",
                          usage=Usage(input_tokens=1_000, output_tokens=500),
                          stop_reason="end_turn")
        with patch("openalph.agent.stream",
                   side_effect=_tool_roundtrip_stream(
                       "advisor", {"focus": "x", "model": "router"})), \
             patch("openalph.tools.advisor.complete", new=AsyncMock(return_value=advice)):
            await agent.handle_input("work", ROOM, callbacks=cb,
                                     on_tool_call=_tool_notice, on_tool_intent=_tool_intent)
        sys_calls = [c for c in bot.session_log.append.call_args_list
                     if c.kwargs.get("event") == "advisor_consult"]
        assert sys_calls, "advisor_consult JSONL entry must be written"
        assert sys_calls[0].kwargs.get("cost_usd") == 0.0
        assert sys_calls[0].kwargs.get("unpriced_tokens") == 1_500


# --- F6: nested advisor-in-subagent cost folded into sub_cost ------------
class TestNestedAdvisorCost:
    @pytest.mark.asyncio
    async def test_sub_advisor_cost_folded_into_subagent(self, tmp_path):
        from openalph.tools import discover_tools
        ws = _setup_workspace(tmp_path, tools=("advisor",))
        config = _cfg(ws)
        sub_tools = discover_tools(ws)
        adv_tc = ToolCall(id="adv_1", name="advisor",
                          input={"focus": "help", "model": "anthropic/claude-opus-4-8"})
        _iter = [0]

        async def _sub_complete(*a, **kw):
            _iter[0] += 1
            if _iter[0] == 1:
                return Response(content="", model="claude-opus-4-8",
                                usage=Usage(input_tokens=1_000, output_tokens=500),
                                stop_reason="tool_use", tool_calls=[adv_tc])
            return Response(content="done", model="claude-opus-4-8",
                            usage=Usage(input_tokens=1_000, output_tokens=500),
                            stop_reason="end_turn", tool_calls=[])

        advice = Response(content="advice", model="claude-opus-4-8",
                          usage=Usage(input_tokens=200_000, output_tokens=100_000),
                          stop_reason="end_turn")
        callbacks = {"subagent_results": {}, "room_id": ROOM}
        with patch("openalph.tools.subagent.complete", new=AsyncMock(side_effect=_sub_complete)), \
             patch("openalph.tools.advisor.complete", new=AsyncMock(return_value=advice)):
            await run_subagent(task="x", config=config, tools=sub_tools,
                               call_id="tc_sub", parent_room_id=ROOM, callbacks=callbacks)
        info = callbacks["subagent_results"][(ROOM, "tc_sub")]
        # sub's own 2 turns: (1000*5+500*25)/1e6 = 0.0175 each = 0.035
        # + nested advisor 3.5 = 3.535
        assert info["cost_usd"] == pytest.approx(3.535, abs=APPROX)


# --- F3: fail-soft (never crash a turn / brick room wake) ----------------
class TestFailSoft:
    def test_usage_totals_tolerates_malformed_values(self, tmp_path):
        sl = SessionLog(str(tmp_path), AGENT_USER)
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM, content="a",
                  usage={"input_tokens": 10, "output_tokens": 5,
                         "cost_usd": "not-a-number", "unpriced_tokens": None})
        sl.append(role="tool", sender=AGENT_USER, room=ROOM, call_id="t",
                  name="subagent", output="x", is_error=False,
                  cost_usd=["bad"], unpriced_tokens="x")
        totals = sl.usage_totals(ROOM)  # must not raise
        assert totals["main_cost_usd"] == 0.0
        assert totals["subagent_cost_usd"] == 0.0
        assert totals["total_output_tokens"] == 5

    def test_restore_usage_tolerates_malformed(self, tmp_path):
        agent = Agent(_cfg(_setup_workspace(tmp_path)))
        agent.restore_usage(ROOM, {"main_cost_usd": "x", "unpriced_tokens": None,
                                   "total_tool_calls": "y"})  # must not raise
        u = agent._usage_for(ROOM)
        assert u["main_cost_usd"] == 0.0
        assert u["unpriced_tokens"] == 0

    def test_compute_cost_failure_does_not_crash_turn(self, tmp_path):
        agent = Agent(_cfg(_setup_workspace(tmp_path)))
        with patch("openalph.agent.compute_cost", side_effect=RuntimeError("boom")):
            agent._record_turn_usage(ROOM, _u(input_tokens=10, output_tokens=5),
                                     "claude-opus-4-8", "1h",
                                     provider_key="anthropic", provider_type="anthropic")
        # turn survived; cost recorded $0; token counters (which precede the
        # cost block) still updated
        assert agent._usage_for(ROOM)["main_cost_usd"] == 0.0
        assert agent._usage_for(ROOM)["total_output_tokens"] == 5


# --- keepalive: a cache-keepalive ping must add $0 (spec regression guard)
class TestKeepaliveUncosted:
    @pytest.mark.asyncio
    async def test_keepalive_ping_is_never_costed(self, tmp_path):
        agent = Agent(_cfg(_setup_workspace(tmp_path)))
        # baseline main cost from one real turn
        agent._record_turn_usage(ROOM, _u(input_tokens=100_000, output_tokens=50_000),
                                 "claude-opus-4-8", "1h",
                                 provider_key="anthropic", provider_type="anthropic")
        baseline = agent._usage_for(ROOM)["main_cost_usd"]
        recorded = []
        orig = agent._record_turn_usage
        agent._record_turn_usage = lambda *a, **kw: (recorded.append(a), orig(*a, **kw))[1]
        stop = asyncio.Event()
        calls = [0]
        ping_usage = _u(input_tokens=1, output_tokens=1,
                        cache_read_tokens=200_000, cache_creation_tokens=0)

        async def _fake_ping(*a, **kw):
            calls[0] += 1
            stop.set()  # let the loop exit cleanly on the next wait
            return ping_usage

        with patch("openalph.agent.ping_cache", new=_fake_ping), \
             patch("openalph.agent._keepalive_interval", return_value=0.02):
            await asyncio.wait_for(agent._cache_keepalive(
                system="sys", messages=[], tools=None, cache_ttl="1h",
                model="anthropic/claude-opus-4-8", thinking="off",
                room_id=ROOM, on_miss=None, stop=stop, stream_elapsed=0.0), timeout=5)
        assert calls[0] >= 1                # a ping actually fired
        assert recorded == []               # ...but was never costed
        assert agent._usage_for(ROOM)["main_cost_usd"] == pytest.approx(baseline, abs=APPROX)


# --- F1 (advisor half): advisor alias priced from served model -----------
class TestAdvisorAliasPricing:
    @pytest.mark.asyncio
    async def test_advisor_alias_model_priced_from_served_model(self, tmp_path):
        assert run_advisor is not None and compute_cost is not None
        config = _cfg(_setup_workspace(tmp_path),
                      model_aliases={"opus": "anthropic/claude-opus-4-8"})
        advice = Response(content="advice", model="claude-opus-4-8",
                          usage=Usage(input_tokens=1_000, output_tokens=120,
                                      cache_read_tokens=42),
                          stop_reason="end_turn")
        callbacks = {"advisor_uses": {}, "room_id": ROOM, "call_id": "tc_adv",
                     "advisor_results": {}, "get_transcript": lambda: ("sys", [])}
        tool_config = {"model": None, "max_uses": 10, "max_tokens": 8192,
                       "thinking": "medium", "cache_ttl": "5m",
                       "include_system_prompt": True, "transcript_max_chars": 0,
                       "timeout": 300}
        with patch("openalph.tools.advisor.complete", new=AsyncMock(return_value=advice)):
            await run_advisor(focus="q", model="opus", config=config,
                              tool_config=tool_config, callbacks=callbacks)
        info = callbacks["advisor_results"][(ROOM, "tc_adv")]
        # bare alias 'opus' -> anthropic/claude-opus-4-8, served claude-opus-4-8:
        # 1000*5/1e6 + 120*25/1e6 + 42*0.5/1e6 = 0.008021 (NOT $0-unpriced)
        assert info["cost_usd"] == pytest.approx(0.008021, abs=APPROX)
        assert info["unpriced_tokens"] == 0


# --- F4: live totals == post-restart totals (no double-count, no drop) ---
class TestLiveRestartSymmetry:
    @pytest.mark.asyncio
    async def test_subagent_live_equals_post_restart(self, tmp_path):
        bot, agent = _make_bot(tmp_path, real_session_log=True)
        _tool_notice, _tool_intent = bot._make_tool_callbacks(ROOM)
        cb = bot._build_agent_callbacks(ROOM, None)
        sub_resp = Response(content="done", model="claude-opus-4-8",
                            usage=Usage(input_tokens=200_000, output_tokens=100_000),
                            stop_reason="end_turn", tool_calls=[])
        with patch("openalph.agent.stream",
                   side_effect=_tool_roundtrip_stream("subagent", {"task": "do it"})), \
             patch("openalph.tools.subagent.complete", new=AsyncMock(return_value=sub_resp)):
            await agent.handle_input("work", ROOM, callbacks=cb,
                                     on_tool_call=_tool_notice, on_tool_intent=_tool_intent)
        live = agent.status(ROOM)["subagent_cost_usd"]
        assert live == pytest.approx(3.5, abs=APPROX)
        # simulate a restart: a fresh Agent rehydrating from the SAME JSONL must
        # land on the identical figure — live increment and restart re-sum are
        # disjoint (no double-count) and complete (no drop).
        agent2 = Agent(_cfg(_setup_workspace(tmp_path / "ws2")))
        agent2.restore_usage(ROOM, bot.session_log.usage_totals(ROOM))
        assert agent2.status(ROOM)["subagent_cost_usd"] == pytest.approx(live, abs=APPROX)


# --- F3 (subagent half): cost failure must not error a successful run ----
class TestSubagentCostFailSoft:
    @pytest.mark.asyncio
    async def test_compute_cost_failure_does_not_error_run(self, tmp_path):
        assert run_subagent is not None
        config = _cfg(_setup_workspace(tmp_path))
        sub_resp = Response(content="done", model="claude-opus-4-8",
                            usage=Usage(input_tokens=100, output_tokens=50),
                            stop_reason="end_turn", tool_calls=[])
        callbacks = {"subagent_results": {}, "room_id": ROOM}
        with patch("openalph.tools.subagent.complete", new=AsyncMock(return_value=sub_resp)), \
             patch("openalph.tools.subagent.compute_cost", side_effect=RuntimeError("boom")):
            result = await run_subagent(task="x", config=config, tools=[],
                                        call_id="tc_sub", parent_room_id=ROOM,
                                        callbacks=callbacks)
        assert result.is_error is False   # accounting failure != work failure
        info = callbacks["subagent_results"][(ROOM, "tc_sub")]
        assert info["cost_usd"] == 0.0


# =========================================================================
# AREA A (RED) — provider-keyed pricing + Fireworks cost.
# These tests specify a feature that does NOT exist yet:
#   - _MODEL_PRICING becomes provider-namespaced ({"anthropic": {...},
#     "fireworks": {...}}) instead of a flat bare-model dict.
#   - compute_cost gains keyword-only `provider_key` and `provider_type`,
#     and the `is_anthropic` bool is REPLACED by them.
# Expected pre-build state: the feature-keyed tests fail (TypeError: compute_cost()
# got an unexpected keyword argument 'provider_key', or KeyError on the
# namespaced table); the legacy fallback tests stay green. The build will
# implement the feature and flip these to green. Imports are inside each test
# function so a missing symbol fails ONE test, never the whole module.
# =========================================================================

class TestFireworksCost:
    """Fireworks cost (non-anthropic priced formula): cost = (uncached_input*in
    + output*out + cache_read*cached_rate) / 1_000_000, with a normalized Usage
    (input_tokens = uncached only, cache_creation_tokens = 0). Uses the literal
    pre-computed dollar values; mirror TestComputeCost for approx style and
    TestProviderTypeGate for Usage construction."""

    def test_glm_cached(self):
        from openalph.provider import compute_cost, Usage
        usage = Usage(input_tokens=2000, output_tokens=2000,
                      cache_read_tokens=8000, cache_creation_tokens=0)
        r = compute_cost("accounts/fireworks/models/glm-5p2", usage,
                         provider_key="fireworks", provider_type="openai")
        assert r.priced is True
        assert r.unpriced_tokens == 0
        assert r.cost_usd == pytest.approx(0.01272, abs=APPROX)

    def test_kimi_k3_cached(self):
        from openalph.provider import compute_cost, Usage
        usage = Usage(input_tokens=10000, output_tokens=5000,
                      cache_read_tokens=90000, cache_creation_tokens=0)
        r = compute_cost("accounts/fireworks/models/kimi-k3", usage,
                         provider_key="fireworks", provider_type="openai")
        assert r.priced is True
        assert r.unpriced_tokens == 0
        assert r.cost_usd == pytest.approx(0.132, abs=APPROX)

    def test_kimi_k2p6_uncached(self):
        from openalph.provider import compute_cost, Usage
        usage = Usage(input_tokens=20000, output_tokens=1000,
                      cache_read_tokens=0, cache_creation_tokens=0)
        r = compute_cost("accounts/fireworks/models/kimi-k2p6", usage,
                         provider_key="fireworks", provider_type="openai")
        assert r.priced is True
        assert r.unpriced_tokens == 0
        assert r.cost_usd == pytest.approx(0.023, abs=APPROX)


class TestFireworksMissingCachedRate:
    """A fireworks-namespaced model with NO cached_input rate: cache_read is
    costed at the FULL input rate (never Anthropic's 0.1x), and a warn-once is
    emitted via a NEW module-level set _warned_missing_cached_rate. The model
    row is injected with patch.dict so this is catalog-independent and safe
    pre-/post-build (patch.dict adds the 'fireworks' key if absent)."""

    def test_missing_cached_rate_full_rate(self):
        from openalph.provider import compute_cost, Usage, _MODEL_PRICING
        from openalph import provider
        with patch.dict(provider._MODEL_PRICING,
                        {"fireworks": {"sizetier-test":
                                       {"input": 0.90, "output": 0.90}}}):
            usage = Usage(input_tokens=6000, output_tokens=1000,
                          cache_read_tokens=4000, cache_creation_tokens=0)
            r = compute_cost("sizetier-test", usage,
                             provider_key="fireworks", provider_type="openai")
            assert r.priced is True
            assert r.cost_usd == pytest.approx(0.0099, abs=APPROX)

    def test_missing_cached_rate_warns_once(self, caplog):
        import logging
        from openalph.provider import compute_cost, Usage
        from openalph import provider
        # Reset guard (safe pre- and post-build).
        s = getattr(provider, "_warned_missing_cached_rate", None)
        if s is not None:
            s.discard("sizetier-test")
        with patch.dict(provider._MODEL_PRICING,
                        {"fireworks": {"sizetier-test":
                                       {"input": 0.90, "output": 0.90}}}):
            usage = Usage(input_tokens=6000, output_tokens=1000,
                          cache_read_tokens=4000, cache_creation_tokens=0)
            with caplog.at_level(logging.WARNING, logger="openalph.provider"):
                compute_cost("sizetier-test", usage,
                             provider_key="fireworks", provider_type="openai")
                compute_cost("sizetier-test", usage,
                             provider_key="fireworks", provider_type="openai")
            warns = [rec for rec in caplog.records
                     if "sizetier-test" in rec.getMessage()]
            assert len(warns) == 1


class TestFireworksProviderGate:
    """Provider-gate matrix mirroring TestProviderTypeGate's Usage/call style.
    Namespace selection: ns = 'anthropic' if provider_type == 'anthropic' else
    provider_key; table = _MODEL_PRICING.get(ns)."""

    def test_fireworks_priced(self):
        from openalph.provider import compute_cost, Usage
        usage = Usage(input_tokens=1000, output_tokens=1000)
        r = compute_cost("accounts/fireworks/models/glm-5p2", usage,
                         provider_key="fireworks", provider_type="openai")
        assert r.priced is True
        assert r.cost_usd == pytest.approx(0.0058, abs=APPROX)
        assert r.unpriced_tokens == 0

    def test_other_openai_key_unpriced(self):
        from openalph.provider import compute_cost, Usage
        usage = Usage(input_tokens=1000, output_tokens=1000)
        r = compute_cost("some-model", usage,
                         provider_key="openrouter", provider_type="openai")
        assert r.priced is False
        assert r.cost_usd == 0.0
        assert r.unpriced_tokens == 2000

    def test_nonanthropic_serving_claude_unpriced(self):
        from openalph.provider import compute_cost, Usage
        usage = Usage(input_tokens=1000, output_tokens=1000)
        r = compute_cost("claude-opus-5", usage,
                         provider_key="fireworks", provider_type="openai")
        assert r.priced is False
        assert r.cost_usd == 0.0
        assert r.unpriced_tokens == 2000

    def test_anthropic_unchanged(self):
        from openalph.provider import compute_cost, Usage
        usage = Usage(input_tokens=1000, output_tokens=1000)
        r = compute_cost("claude-opus-5", usage,
                         provider_key="anthropic", provider_type="anthropic")
        assert r.priced is True
        assert r.cost_usd == pytest.approx(0.03, abs=APPROX)


class TestLegacyGateFallback:
    """EXPECTED-PASS regression guards: they call the CURRENT signature with NO
    gate kwargs (provider_key/provider_type both absent -> legacy string-shape
    classification). They pass now and must keep passing post-build."""

    def test_legacy_claude_priced(self):
        from openalph.provider import compute_cost, Usage
        r = compute_cost("claude-opus-5", Usage(input_tokens=1000,
                                                output_tokens=1000))
        assert r.priced is True
        assert r.cost_usd == pytest.approx(0.03, abs=APPROX)

    def test_legacy_fireworks_unpriced(self):
        from openalph.provider import compute_cost, Usage
        r = compute_cost("accounts/fireworks/models/glm-5p2",
                         Usage(input_tokens=1000, output_tokens=1000))
        assert r.priced is False


class TestFireworksPricingTable:
    """The namespaced Fireworks pricing table (mirror TestPricingTable for
    table-shape assertions). Pre-build this raises KeyError because the table
    is still flat, not namespaced under 'fireworks'."""

    def test_fireworks_rates(self):
        from openalph import provider
        assert provider._MODEL_PRICING["fireworks"]["glm-5p2"] == {
            "input": 1.40, "output": 4.40, "cached_input": 0.14}
        assert provider._MODEL_PRICING["fireworks"]["kimi-k3"] == {
            "input": 3.00, "output": 15.00, "cached_input": 0.30}
        assert provider._MODEL_PRICING["fireworks"]["kimi-k2p6"] == {
            "input": 0.95, "output": 4.00, "cached_input": 0.16}


# =========================================================================
# AREA INT (RED) — Fireworks REAL-PATH end-to-end cost + cache-read.
# Mirrors TestRealPathMainCost::test_real_turn_freezes_cost_in_room_usage and
# ::test_context_status_carries_cost_fields EXACTLY for harness (_make_bot,
# _single_turn_stream, patch("openalph.agent.stream", ...), ROOM, APPROX).
# The ONE difference: a Fireworks-configured bot via _make_bot's agent_kw
# passthrough (-> _cfg). Forces the build to thread provider_key through
# agent.py's cost path (today it resolves is_anthropic=False for Fireworks ->
# turn left UNPRICED -> main_cost_usd == 0.0, not 0.374). That gap is the red.
# Additive only — reuses module-level helpers; no new imports or redefinitions.
# =========================================================================

class TestFireworksRealPathCost:
    @pytest.mark.asyncio
    async def test_fireworks_turn_priced_and_cached(self, tmp_path):
        fw_providers = {"fireworks": ProviderConfig(
            key="fireworks", type="openai", api_key="sk-test",
            base_url="https://api.fireworks.ai/inference/v1", quirks=[])}
        bot, agent = _make_bot(
            tmp_path,
            default_model="fireworks/accounts/fireworks/models/glm-5p2",
            providers=fw_providers)
        cb = bot._build_agent_callbacks(ROOM, None)
        usage = _u(input_tokens=100000, output_tokens=50000,
                   cache_read_tokens=100000, cache_creation_tokens=0)
        stream_fn = _single_turn_stream(
            usage, model="accounts/fireworks/models/glm-5p2", text="done")
        with patch("openalph.agent.stream", side_effect=stream_fn):
            await agent.handle_input("hi", ROOM, callbacks=cb, cache_ttl="1h")
        # glm-5p2 @ input 1.40 / output 4.40 / cached_input 0.14 per MTok:
        # 100000*1.40 + 50000*4.40 + 100000*0.14 = 374000, /1e6 = 0.374.
        assert agent._usage_for(ROOM)["main_cost_usd"] == pytest.approx(0.374, abs=APPROX)
        assert agent.last_turn_usage(ROOM)["cost_usd"] == pytest.approx(0.374, abs=APPROX)
        cs = bot._build_context_status(ROOM)
        if "cache_read_tokens" not in cs:
            print("context_status keys:", sorted(cs.keys()))
        assert cs["cache_read_tokens"] == 100000
