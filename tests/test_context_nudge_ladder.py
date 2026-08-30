"""Context nudge ladder — escalating context-pressure reminders.

TDD test suite materializing ALL 17 items of the spec's test matrix
(context-nudge-ladder-spec.md §6), named T-* as in the matrix:

  T-boundary-60  T-threshold-each  T-monotonic  T-jump  T-turn-start-gap
  T-same-turn-dedup  T-rehydrate-legacy  T-rehydrate-tiered  T-reset
  T-degenerate  T-denominator-reachability  T-heartbeat-source  T-kill-switch
  T-remaining-figure  T-detail  T-model-switch-rearm  T-state-plumbing

Pure-engine style mirrors tests/test_guidance_injection.py (same
_cfg/_state/_engine helper shape, same trigger-ID constant conventions);
T-state-plumbing is integration-style following test_guidance_integration.py
(real Agent, real per-room engines, mocked provider, callbacks capture).

Key spec decisions pinned here:
  D8  denominator = usable runway (limit − max_tokens), not the full window
  D9  engine reads ONLY state.available_tokens (default 0 → silent skip)
  D10 integer cross-multiplication, inclusive (no float thresholds)
  D11 model-switch re-arm keyed on non-empty state.model_resolved
  D12 turn-start context_tokens includes inbound content_tokens

The legacy single-shot T2 contract (≥80%-of-window, boundary-only,
once-per-session, "of the window" text) is superseded — old-contract pins
live in the migrated wave-1 tests, not here.
"""

import asyncio
import inspect
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Guard imports — the ladder does not exist yet; base modules should load.
# ---------------------------------------------------------------------------
try:
    from openalph.agent import Agent
    from openalph.config import AgentConfig, ProviderConfig
    from openalph.provider import Response, Usage, StreamEvent, ToolCall
    from openalph.tools import ToolResult, BUILTIN_TOOLS
    from openalph.reminders import ReminderEngine, ReminderState, Reminder
    _base_imported = True
except Exception:
    _base_imported = False
    Agent = AgentConfig = ProviderConfig = None
    Response = Usage = StreamEvent = ToolCall = None
    ToolResult = None
    BUILTIN_TOOLS = {}
    ReminderEngine = ReminderState = Reminder = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LADDER_ID = "context-pressure"          # trigger ID unchanged from wave-1 T2
TAG_OPEN = "<system-reminder>"
TAG_CLOSE = "</system-reminder>"

# D8: fleet-standard blackwell config for the reachability test (T11).
WINDOW_262K = 262144
MAX_TOKENS_64K = 65536
AVAIL_196K = WINDOW_262K - MAX_TOKENS_64K        # 196608 usable runway
T4_AT = 176948                                    # == ⌈0.90 × 196608⌉
T4_BELOW = T4_AT - 1                              # 176947 — must NOT fire tier 4

# D11: two window sizes for the model-switch tests (curated-table free).
_MODEL_A = "ladder/model-a-262k"                  # 262144 window
_MODEL_B = "ladder/model-b-131k"                  # 131072 window


# ---------------------------------------------------------------------------
# Helpers (shape mirrors test_guidance_injection.py)
# ---------------------------------------------------------------------------

def _cfg(workspace, **kw):
    """Shorthand AgentConfig builder (same defaults as the wave-1 suite)."""
    defaults = dict(
        name="test-agent",
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
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _state(**kw):
    """Create a ReminderState. Fails if the module is not implemented."""
    assert ReminderState is not None, \
        "ReminderState not importable — src/openalph/reminders.py must be implemented"
    defaults = dict(
        evaluation_point="tool_loop_boundary",
        iteration=0,
        max_iterations=100,
        context_tokens=10000,
        context_limit=200000,
        completed_turns=0,
        turn_source=None,
        tool_calls_this_turn={},
        tool_calls_session={},
        todo_list=[],
        enabled_tools={"shell", "file_read", "file_write", "file_edit",
                       "memory_search", "todo_write", "subagent"},
    )
    defaults.update(kw)
    return ReminderState(**defaults)


def _engine(tmp_path, **config_kw):
    """Create a ReminderEngine. Fails if the module is not implemented."""
    assert ReminderEngine is not None, \
        "ReminderEngine not importable — src/openalph/reminders.py must be implemented"
    return ReminderEngine(_cfg(tmp_path, **config_kw))


def _ladder_fired(eng):
    """Read the engine's monotonic ladder latch (spec §3 fired-state column)."""
    return getattr(eng, "_ladder_fired", None)


def _cp(results):
    """context-pressure reminders among evaluate() results."""
    return [r for r in results if r.trigger == LADDER_ID]


# ============================================================================
# 1. T-boundary-60 — below threshold no fire; at threshold tier 1 fires
#    (BOTH evaluation points)
# ============================================================================

def test_T_boundary_60(tmp_path):
    eng = _engine(tmp_path)
    for point in ("turn_start", "tool_loop_boundary"):
        eng2 = _engine(tmp_path)
        # Below the 60% cross-multiplication: 119999*100 < 60*200000 (12000000)
        st_below = _state(evaluation_point=point,
                          context_tokens=119999, context_limit=200000,
                          available_tokens=200000)
        assert _cp(eng2.evaluate(st_below)) == [], \
            f"{point}: must NOT fire below the tier-1 threshold"
        # Exactly at the integer boundary: 120000*100 == 60*200000 → tier 1
        st_at = _state(evaluation_point=point,
                       context_tokens=120000, context_limit=200000,
                       available_tokens=200000)
        fired = _cp(eng2.evaluate(st_at))
        assert len(fired) == 1, \
            f"{point}: tier 1 must fire exactly at its threshold; got {len(fired)}"
        assert fired[0].detail == "tier=1", \
            f"{point}: tier-1 fire must carry detail='tier=1'; got {fired[0].detail!r}"


# ============================================================================
# 2. T-threshold-each — each tier fires exactly at its integer
#    cross-multiplication boundary (D10, inclusive)
# ============================================================================

def test_T_threshold_each(tmp_path):
    avail = 200000
    # (tier, first-firing context_tokens) — smallest c with c*100 >= pct*avail
    cases = [(1, 120000), (2, 140000), (3, 160000), (4, 180000)]
    prev_boundary = {1: None, 2: 120000, 3: 140000, 4: 160000}
    for tier, at in cases:
        eng = _engine(tmp_path)
        # Pre-latch the previous tier (if any) so the "below" evaluation is
        # about tier `tier` specifically — lower tiers legitimately fire first
        # on a fresh engine.
        if prev_boundary[tier] is not None:
            _cp(eng.evaluate(_state(context_tokens=prev_boundary[tier],
                                    context_limit=200000, available_tokens=avail)))
        # Pure-boundary evaluation (model_resolved="") — no D11 re-arm, so
        # the predicate is exactly the D10 cross-multiplication.
        st_at = _state(context_tokens=at, context_limit=200000, available_tokens=avail)
        # One token below the boundary: no fire (latch is t-1; nothing crossed)
        below = _state(context_tokens=at - 1, context_limit=200000,
                       available_tokens=avail)
        assert _cp(eng.evaluate(below)) == [], \
            f"tier {tier}: must NOT fire at {at - 1} ({(at - 1) * 100} < {tier}0 * {avail})"
        # Exactly at the boundary: fires, and this tier is the one named
        fired = _cp(eng.evaluate(st_at))
        assert len(fired) == 1, \
            f"tier {tier}: must fire exactly once at {at}; got {len(fired)}"
        assert fired[0].detail == f"tier={tier}", \
            f"tier {tier}: detail must be 'tier={tier}'; got {fired[0].detail!r}"


# ============================================================================
# 3. T-monotonic — after tier 2, a dip then re-rise past 60% does NOT re-fire
#    tiers 1–2; tier 3 still fires at 80% (D1)
# ============================================================================

def test_T_monotonic(tmp_path):
    eng = _engine(tmp_path)
    avail = 200000
    # Rise to 75% → tier 2 fires (subsumes tier 1 — one reminder)
    fired = _cp(eng.evaluate(_state(context_tokens=150000, context_limit=200000,
                                    available_tokens=avail)))
    assert len(fired) == 1 and fired[0].detail == "tier=2", \
        f"75% must fire tier 2 only; got {[r.detail for r in fired]}"
    # Dip below 60% — nothing
    assert _cp(eng.evaluate(_state(context_tokens=100000, context_limit=200000,
                                   available_tokens=avail))) == []
    # Re-rise past 60% (65%): tiers 1–2 already latched → NO fire
    again = _cp(eng.evaluate(_state(context_tokens=130000, context_limit=200000,
                                    available_tokens=avail)))
    assert again == [], \
        "re-rise past 60% after a dip must NOT re-fire latched tiers 1–2"
    # Cross 85%: tier 3 fires
    fired3 = _cp(eng.evaluate(_state(context_tokens=170000, context_limit=200000,
                                     available_tokens=avail)))
    assert len(fired3) == 1 and fired3[0].detail == "tier=3", \
        f"85% must fire tier 3; got {[r.detail for r in fired3]}"


# ============================================================================
# 4. T-jump — multi-tier crossing fires ONLY the highest tier (one reminder)
# ============================================================================

def test_T_jump(tmp_path):
    eng = _engine(tmp_path)
    avail = 200000
    st_low = _state(context_tokens=115000, context_limit=200000, available_tokens=avail)
    assert _cp(eng.evaluate(st_low)) == []   # 57.5%: quiet
    st_high = _state(context_tokens=170000, context_limit=200000, available_tokens=avail)
    fired = _cp(eng.evaluate(st_high))        # 85%: crosses tiers 1–3
    assert len(fired) == 1, \
        f"a 57.5%→85% jump must produce exactly ONE reminder; got {len(fired)}"
    assert fired[0].detail == "tier=3", \
        f"the single fire must be the HIGHEST crossed tier; got {fired[0].detail!r}"
    # And nothing further until 90%
    assert _cp(eng.evaluate(st_high)) == []


# ============================================================================
# 5. T-turn-start-gap — zero tool calls, turn_start, 81% → tier 3 fires
#    (closes wave-1 gap #2: boundary-only evaluation)
# ============================================================================

def test_T_turn_start_gap(tmp_path):
    eng = _engine(tmp_path)
    avail = 200000
    st = _state(evaluation_point="turn_start",
                context_tokens=170000, context_limit=200000,  # 85%
                available_tokens=avail,
                tool_calls_this_turn={})
    fired = _cp(eng.evaluate(st))
    assert len(fired) == 1 and fired[0].detail == "tier=3", \
        f"turn_start at 85% with zero tool calls must fire tier 3; got {[r.detail for r in fired]}"


# ============================================================================
# 6. T-same-turn-dedup — tier fired at turn_start does not re-fire at the
#    same turn's boundary
# ============================================================================

def test_T_same_turn_dedup(tmp_path):
    eng = _engine(tmp_path)
    avail = 200000
    st_start = _state(evaluation_point="turn_start",
                      context_tokens=170000, context_limit=200000,
                      available_tokens=avail)
    fired = _cp(eng.evaluate(st_start))
    assert len(fired) == 1 and fired[0].detail == "tier=3"
    st_boundary = _state(evaluation_point="tool_loop_boundary", iteration=1,
                         context_tokens=170000, context_limit=200000,
                         available_tokens=avail)
    again = _cp(eng.evaluate(st_boundary))
    assert again == [], \
        "the same turn's boundary must not re-fire a tier already fired at turn_start"


# ============================================================================
# 7. T-rehydrate-legacy — detail-less context-pressure entry → latch >= 3;
#    tier 4 still reachable
# ============================================================================

def test_T_rehydrate_legacy(tmp_path):
    eng = _engine(tmp_path)
    eng.rehydrate([
        {"role": "user", "source": "reminder", "trigger": LADDER_ID,
         "content": f"{TAG_OPEN}\nContext is at 85% of the window. Converge.\n{TAG_CLOSE}"},
    ])
    assert _ladder_fired(eng) is not None, \
        "engine must track a _ladder_fired latch (rehydrate must set it)"
    assert _ladder_fired(eng) >= 3, \
        f"legacy (detail-less) T2 entry must latch at least tier 3; got {_ladder_fired(eng)}"
    # Tiers <= latch stay quiet...
    st_85 = _state(context_tokens=170000, context_limit=200000, available_tokens=200000)
    assert _cp(eng.evaluate(st_85)) == [], \
        "85% after a legacy latch must not re-ping tiers 1–3"
    # ...but tier 4 remains reachable
    st_95 = _state(context_tokens=180000, context_limit=200000, available_tokens=200000)
    fired = _cp(eng.evaluate(st_95))
    assert len(fired) == 1 and fired[0].detail == "tier=4", \
        f"tier 4 must still fire after a legacy latch; got {[r.detail for r in fired]}"


# ============================================================================
# 8. T-rehydrate-tiered — detail="tier=2" then "tier=4" → latch 4 (max-wins);
#    "tier=4" then "tier=2" → still 4
# ============================================================================

def test_T_rehydrate_tiered(tmp_path):
    def _tier_entry(n):
        return {"role": "user", "source": "reminder", "trigger": LADDER_ID,
                "detail": f"tier={n}",
                "content": f"{TAG_OPEN}\ntier {n}\n{TAG_CLOSE}"}

    eng = _engine(tmp_path)
    eng.rehydrate([_tier_entry(2), _tier_entry(4)])
    assert _ladder_fired(eng) == 4, \
        f"tier=2 then tier=4 must max-wins to 4; got {_ladder_fired(eng)}"

    eng2 = _engine(tmp_path)
    eng2.rehydrate([_tier_entry(4), _tier_entry(2)])
    assert _ladder_fired(eng2) == 4, \
        f"tier=4 then tier=2 must still be 4 (max-wins, never last-wins); got {_ladder_fired(eng2)}"


# ============================================================================
# 9. T-reset — reset() clears latch + _ladder_model; reset_turn() does not
# ============================================================================

def test_T_reset(tmp_path):
    eng = _engine(tmp_path)
    avail = 200000
    # 190000/200000 = 95% → tier 4 fires, and D11 records the model.
    _cp(eng.evaluate(_state(evaluation_point="turn_start", model_resolved="model-x",
                            context_tokens=190000, context_limit=200000,
                            available_tokens=avail)))
    assert _ladder_fired(eng) == 4
    assert getattr(eng, "_ladder_model", "MISSING") == "model-x"

    eng.reset_turn()
    assert _ladder_fired(eng) == 4, \
        "reset_turn() must NOT clear the session-scoped ladder latch"
    assert getattr(eng, "_ladder_model", None) == "model-x", \
        "reset_turn() must NOT clear _ladder_model"

    eng.reset()
    assert _ladder_fired(eng) == 0, \
        f"reset() must clear the ladder latch; got {_ladder_fired(eng)}"
    assert getattr(eng, "_ladder_model", "NOT_NONE") is None, \
        "reset() must set _ladder_model=None (D11)"


# ============================================================================
# 10. T-degenerate — available_tokens <= 0 → no fire, no exception
#     (default-0 state included; D9 fail-safe direction)
# ============================================================================

def test_T_degenerate(tmp_path):
    eng = _engine(tmp_path)
    # Default-0 state: the dataclass default is the production fail-safe
    defaults = dict(
        evaluation_point="tool_loop_boundary", iteration=1, max_iterations=100,
        context_tokens=190000, context_limit=200000, completed_turns=1,
        turn_source=None, tool_calls_this_turn={}, tool_calls_session={},
        todo_list=[], enabled_tools=set(),
    )
    results = eng.evaluate(ReminderState(**defaults))
    assert _cp(results) == [], \
        "default available_tokens=0 must silently skip the ladder (D9)"
    # Explicit zero
    assert _cp(eng.evaluate(_state(context_tokens=190000, context_limit=200000,
                                   available_tokens=0))) == []
    # Negative (limit < max_tokens — degenerate config)
    assert _cp(eng.evaluate(_state(context_tokens=1000, context_limit=100,
                                   available_tokens=-900))) == []
    # Predicate 1: available > 0 but context_limit == 0 → silent skip
    assert _cp(eng.evaluate(_state(context_tokens=190000, context_limit=0,
                                   available_tokens=200000))) == []


# ============================================================================
# 11. T-denominator-reachability — 262K window / 64K max_tokens →
#     available=196,608; tier 4 fires at 176,948 and NOT at 176,947
#     (D8 + D10; the old 80%-of-window T2 was unreachable on this config)
# ============================================================================

def test_T_denominator_reachability(tmp_path):
    cfg = _cfg(tmp_path, max_tokens=MAX_TOKENS_64K,
               model_limits={"ladder/blackwell-262k": WINDOW_262K})
    assert ReminderEngine is not None
    eng = ReminderEngine(cfg)
    st = _state(context_tokens=1000, context_limit=WINDOW_262K,
                available_tokens=AVAIL_196K)
    # The engine must derive its denominator from state alone (D9).
    assert _cp(eng.evaluate(st)) == []

    # Tier 4 at the exact integer boundary: 176948*100 == 196608*90
    st_at = _state(context_tokens=T4_AT, context_limit=WINDOW_262K,
                   available_tokens=AVAIL_196K)
    fired = _cp(eng.evaluate(st_at))
    assert len(fired) == 1 and fired[0].detail == "tier=4", \
        f"tier 4 must fire at context_tokens={T4_AT} (⌈0.90 × {AVAIL_196K}⌉); " \
        f"got {[r.detail for r in fired]}"

    # And NOT one token below — the old float threshold (0.9) would have
    # fired 176947.999…-style drift; integer math must not.
    eng2 = ReminderEngine(_cfg(tmp_path, max_tokens=MAX_TOKENS_64K,
                               model_limits={"ladder/blackwell-262k": WINDOW_262K}))
    # (D11: the first populated evaluation records the model; 176947 is past
    # tier 3's boundary — re-arms to 3, no fire.)
    _cp(eng2.evaluate(st_at))
    st_below = _state(context_tokens=T4_BELOW, context_limit=WINDOW_262K,
                      available_tokens=AVAIL_196K)
    # One token below tier 4's boundary: the engine must NOT fire tier 4 —
    # and must not re-fire anything (latch=3).  Float thresholds would have
    # drifted: 176947/196608 == 0.9000000… inexact binary → flaky fires.
    assert _cp(eng2.evaluate(st_below)) == [], \
        f"tier 4 must NOT fire at {T4_BELOW} (integer cross-multiplication, D10)"

    # Full-window-60% context (157286 = 0.60 × 262144) — old-T2-territory
    # in spirit (mid-runway): under the usable-runway denominator this is
    # 80.0% of 196608 → the ladder has reached tier 3 long before the
    # 196,608 throw point. (Spec item 11's literal "74% → tier 1" clause is
    # arithmetically impossible — 74% of window is 94.9% of runway, past
    # tier 4; see REPORT.md Deviations.)
    eng3 = ReminderEngine(_cfg(tmp_path, max_tokens=MAX_TOKENS_64K,
                               model_limits={"ladder/blackwell-262k": WINDOW_262K}))
    st_60win = _state(context_tokens=157287, context_limit=WINDOW_262K,
                      available_tokens=AVAIL_196K)   # 60.0003% of full window
    fired3 = _cp(eng3.evaluate(st_60win))
    assert len(fired3) == 1 and fired3[0].detail == "tier=3", \
        "full-window-60% context must fire tier 3 of the usable runway " \
        "(the ladder leads the 196,608 throw on blackwell)"


# ============================================================================
# 12. T-heartbeat-source — turn_source="heartbeat" fires (D6: no gating)
# ============================================================================

def test_T_heartbeat_source(tmp_path):
    eng = _engine(tmp_path)
    st = _state(evaluation_point="turn_start", turn_source="heartbeat",
                context_tokens=170000, context_limit=200000, available_tokens=200000)
    fired = _cp(eng.evaluate(st))
    assert len(fired) == 1 and fired[0].detail == "tier=3", \
        f"heartbeat-sourced turns must fire the ladder (D6); got {[r.detail for r in fired]}"


# ============================================================================
# 13. T-kill-switch — reminders=False → nothing
# ============================================================================

def test_T_kill_switch(tmp_path):
    eng = _engine(tmp_path, reminders=False)
    st = _state(context_tokens=190000, context_limit=200000, available_tokens=200000)
    assert eng.evaluate(st) == [], \
        "reminders=False (kill switch) must suppress the ladder entirely"


# ============================================================================
# 14. T-remaining-figure — text contains pct + remaining-tokens figure
#     for the fired tier (spec §5 interpolation)
# ============================================================================

def test_T_remaining_figure(tmp_path):
    eng = _engine(tmp_path)
    avail = 200000
    # Pure-boundary evaluation (model_resolved="") — direct fire, no D11
    # re-arm involvement. 82.5% of runway → tier 3; remaining = 200000−165000.
    st = _state(context_tokens=165000, context_limit=200000, available_tokens=avail)
    fired = _cp(eng.evaluate(st))
    assert len(fired) == 1 and fired[0].detail == "tier=3"
    text = fired[0].text
    # Spec §5: the tier-3 text names the tier pct and the remaining-tokens
    # figure (~35,000) — the only interpolation is {remaining:,}.
    assert "80%" in text, f"tier-3 text must contain the tier pct '80%'; got: {text}"
    assert "35,000" in text, \
        f"tier-3 text must contain the remaining-tokens figure '35,000'; got: {text}"


# ============================================================================
# 15. T-detail — fired reminder carries detail="tier=<n>"
# ============================================================================

def test_T_detail(tmp_path):
    eng = _engine(tmp_path)
    avail = 200000
    # Walk the whole ladder; every fire must be tier-keyed in .detail.
    fired_details = []
    # Pure-boundary evaluations (model_resolved="") — one clean fire per tier.
    for ctx in (120000, 140000, 160000, 180000):
        for r in _cp(eng.evaluate(_state(context_tokens=ctx, context_limit=200000,
                                         available_tokens=avail))):
            fired_details.append(r.detail)
    assert fired_details == ["tier=1", "tier=2", "tier=3", "tier=4"], \
        f"each tier fire must carry detail='tier=<n>' (drives rehydrate); got {fired_details}"


# ============================================================================
# 16. T-model-switch-rearm (D11)
# ============================================================================

def test_T_model_switch_rearm(tmp_path):
    limits = {"ladder/model-a-262k": WINDOW_262K,
              "ladder/model-b-131k": 131072}

    # (a) latch=4 under model A (262K window; runway hand-flowed 196608,
    #     blackwell-style) → switch to model B (131072 window; runway
    #     131072−65536=65536, consistent with this config's max_tokens) at
    #     40% of the new runway.
    eng = ReminderEngine(_cfg(tmp_path, max_tokens=MAX_TOKENS_64K,
                              model_limits=limits))
    # Step 1 — fresh engine: first populated evaluation records model A and
    # re-arms to currently-exceeded tier (85% → 3) without firing.
    assert _cp(eng.evaluate(_state(evaluation_point="turn_start", model_resolved=_MODEL_A,
                                   context_tokens=167000, context_limit=WINDOW_262K,
                                   available_tokens=196608))) == []
    assert _ladder_fired(eng) == 3
    # Step 2 — same model, rise to 95.1%: tier 4 fires; latch=4 under A.
    st_a = _state(evaluation_point="turn_start", model_resolved=_MODEL_A,
                  context_tokens=186978, context_limit=WINDOW_262K,
                  available_tokens=196608)   # 95.1% of A's runway
    fired_a = _cp(eng.evaluate(st_a))
    assert fired_a and fired_a[0].detail == "tier=4", "setup: tier 4 must fire under A"
    assert _ladder_fired(eng) == 4 and getattr(eng, "_ladder_model") == _MODEL_A

    # Switch to B at 40% of B's runway: latch re-arms to 0 (below tier 1),
    # no immediate fire…
    st_b = _state(evaluation_point="turn_start", model_resolved=_MODEL_B,
                  context_tokens=26215, context_limit=131072,
                  available_tokens=131072 - MAX_TOKENS_64K)   # 65536 → 40.0%
    assert _cp(eng.evaluate(st_b)) == [], \
        "re-arm into below-tier-1 must not fire immediately"
    assert _ladder_fired(eng) == 0, \
        f"re-arm must reset the latch to the highest tier CURRENTLY exceeded (0); got {_ladder_fired(eng)}"
    # …and tier 1 fires on the next crossing of 60% of B's runway
    st_b60 = _state(evaluation_point="turn_start", model_resolved=_MODEL_B,
                    context_tokens=39322, context_limit=131072,
                    available_tokens=65536)   # 60.0% of B's runway
    fired_b = _cp(eng.evaluate(st_b60))
    assert len(fired_b) == 1 and fired_b[0].detail == "tier=1", \
        f"after re-arm, crossing 60% must fire tier 1; got {[r.detail for r in fired_b]}"

    # (b) Re-arm to the currently-exceeded tier: switch INTO a window already
    #     at 85% → latch=3, NO immediate fire, tier 4 reachable.
    eng2 = ReminderEngine(_cfg(tmp_path, max_tokens=MAX_TOKENS_64K,
                               model_limits=limits))
    # Fresh engine: first populated evaluation re-arms to currently-exceeded
    # tier (170000/196608 = 86.5% → 3) WITHOUT firing.
    assert _cp(eng2.evaluate(_state(evaluation_point="turn_start", model_resolved=_MODEL_A,
                                    context_tokens=170000, context_limit=WINDOW_262K,
                                    available_tokens=196608))) == []
    assert _ladder_fired(eng2) == 3
    # Switch INTO a window already at 85%: re-arms to 3, no duplicate fire.
    st_c = _state(evaluation_point="turn_start", model_resolved=_MODEL_B,
                  context_tokens=55706, context_limit=131072,
                  available_tokens=65536)   # 85.0% of B's runway
    assert _cp(eng2.evaluate(st_c)) == [], \
        "switching into 85% must re-arm silently (no duplicate fire)"
    assert _ladder_fired(eng2) == 3, \
        f"re-arm must latch the highest tier currently exceeded (3); got {_ladder_fired(eng2)}"
    fired4 = _cp(eng2.evaluate(_state(evaluation_point="turn_start", model_resolved=_MODEL_B,
                                      context_tokens=58983,
                                      context_limit=131072, available_tokens=65536)))
    assert len(fired4) == 1 and fired4[0].detail == "tier=4", \
        f"tier 4 must remain reachable after re-arm; got {[r.detail for r in fired4]}"

    # (c) model_resolved="" (boundary site, kdsn.298) NEVER re-arms.
    eng3 = ReminderEngine(_cfg(tmp_path, max_tokens=MAX_TOKENS_64K,
                               model_limits=limits))
    assert _cp(eng3.evaluate(_state(evaluation_point="turn_start", model_resolved=_MODEL_A,
                                    context_tokens=170000, context_limit=WINDOW_262K,
                                    available_tokens=196608))) == []
    assert _ladder_fired(eng3) == 3
    # Same window content at the boundary site (empty model_resolved):
    # 170000/196608 = 86.5% — below tier 4, latch must stay 3 and no fire.
    st_bd = _state(evaluation_point="tool_loop_boundary", iteration=2,
                   model_resolved="", context_tokens=170000, context_limit=WINDOW_262K,
                   available_tokens=196608)
    assert _cp(eng3.evaluate(st_bd)) == []
    assert _ladder_fired(eng3) == 3, \
        "model_resolved='' must never re-arm the ladder latch"


# ============================================================================
# 17. T-state-plumbing — agent.py sites populate available_tokens (D9) and
#     turn-start context_tokens includes inbound content (D12).
#     Integration-style per test_guidance_integration.py /
#     test_guidance_wiring.py: real Agent, mocked provider; the captured
#     ReminderState objects are what each site hands to evaluate().
# ============================================================================

ROOM = "!ladder-plumb:matrix.local"
AGENT_USER = "@agent:matrix.local"


def _make_text_stream(text="ok"):
    """Provider stream: one text response, no tool calls (loop ends after 1 call)."""
    async def _stream(*, config=None, system=None, messages=None,
                      tools=None, model="test", thinking=None,
                      cache_ttl=None, **kw):
        yield StreamEvent(type="text", content=text)
        yield StreamEvent(
            type="done",
            response=Response(content=text, model=model,
                              usage=Usage(input_tokens=10, output_tokens=5),
                              stop_reason="end_turn"),
            stop_reason="end_turn", model=model)
    return _stream


def test_T_state_plumbing(tmp_path):
    """Real Agent, real per-room engines, mocked provider. Catches the actual
    ReminderState objects both sites hand to engine.evaluate():

      (a) BOTH sites populate available_tokens == limit − max_tokens (D9 —
          same expression the overflow guard uses);
      (b) turn-start context_tokens includes inbound content_tokens when
          append_user=True (D12 — the guard's inclusive estimate).
    """
    ws_dir = tmp_path / "ws"
    (ws_dir / "tools").mkdir(parents=True)
    for name in ("shell", "file_read", "file_write", "file_edit",
                 "todo_write", "memory_search"):
        (ws_dir / "tools" / f"{name}.toml").write_text("[config]\n")

    cfg = AgentConfig(
        name="plumb-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[],
        )},
        workspace=ws_dir,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    agent = Agent(cfg)
    states: list = []

    async def _log_reminder(room_id, reminder):
        pass

    async def _send_notice(room_id, body, **kw):
        pass

    orig_evaluate = ReminderEngine.evaluate

    def capturing_evaluate(self, state):
        states.append(state)
        return orig_evaluate(self, state)

    # "hey!" is 4 chars → content_tokens == exactly 1 (estimation is len//4),
    # making the D12 inclusive-vs-exclusive delta observable.
    async def _drive():
        cb = {"log_reminder": _log_reminder, "send_notice": _send_notice,
              "turn_source": None, "room_id": ROOM}
        with patch("openalph.agent.stream", side_effect=_make_text_stream()), \
             patch.object(ReminderEngine, "evaluate", capturing_evaluate):
            await agent.handle_input("hey!", ROOM, callbacks=cb)

    asyncio.run(_drive())

    start_states = [s for s in states if s.evaluation_point == "turn_start"]
    boundary_states = [s for s in states if s.evaluation_point == "tool_loop_boundary"]
    assert start_states, "turn-start site must evaluate reminders"
    assert boundary_states, "boundary site must evaluate reminders"

    # (a) D9: BOTH sites populate available_tokens = limit − max_tokens —
    # the same expression the overflow guard computes.
    for s in start_states + boundary_states:
        assert hasattr(s, "available_tokens"), \
            "ReminderState must gain an available_tokens field (D9)"
        expected = s.context_limit - cfg.max_tokens
        assert s.available_tokens == expected, (
            f"{s.evaluation_point} site: available_tokens must be "
            f"limit − max_tokens ({expected}); got {s.available_tokens}"
        )
    assert all(s.available_tokens == agent._resolve_model_limit(ROOM) - cfg.max_tokens
               for s in states), \
        "available_tokens must match the overflow guard's own expression"

    # (b) D12: turn-start context_tokens = guard's inclusive estimate
    # (history-before-inbound + content_tokens when append_user=True).
    st_start = start_states[0]
    hist = agent.history(ROOM)
    # Turn 1: pre-inbound history was EMPTY; "hey!" contributes 1 token.
    # (Context is tiny — no reminders fired, so no extra history entries.)
    last_hey = max(i for i, m in enumerate(hist) if m.get("content") == "hey!")
    pre_inbound = hist[:last_hey]
    expected_inclusive = (
        agent._estimate_context_tokens(ROOM, history=pre_inbound)
        + agent._estimate_content_tokens("hey!")
    )
    assert st_start.context_tokens == expected_inclusive, (
        f"D12: turn-start context_tokens must include inbound content_tokens "
        f"(append_user=True): expected {expected_inclusive}, "
        f"got {st_start.context_tokens}"
    )
