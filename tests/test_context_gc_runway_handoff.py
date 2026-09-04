"""Context GC — runway-aware forced handoff + demoted durable budget.

Orchestrator-authored red suite for workspace-kdsn.305.12 (spec:
memory/projects/openalph/context-gc/spec-kdsn.305.12-durable-budget-rework.md).
Sub-agents implement against these assertions and NEVER modify this file —
report contract discrepancies to the orchestrator.

Pinned contract (spec D1–D6):
  - Apply-path handoff gates on POST-SNAPSHOT runway
    (available − tokens_after_est < max(window·handoff_pct, handoff_min)),
    decoupled from durable-set size (over_budget).
  - Over-budget keeps full verbatim attachment; its only observable is the
    snapshot-header marker, reworded to informational text.
  - ContextGCConfig: defaults durable 25.0/96000; new handoff_runway_pct /
    handoff_runway_min_tokens (10.0 / 24000); fail-loud parse.
  - Reminder trigger gc-budget REMOVED; gc-runway added (≥90% of runway
    consumed post-boundary, once/session, turn_start only, reset re-arms).
"""

import json
import pytest

from openalph.config import ConfigError, load_config
from openalph.context_gc import (
    GC_FORCED_HANDOFF_TRIGGER,
    apply_boundary,
    apply_boundary_and_rebuild,
    frame_snapshot,
    resolve_durable_set,
)
from openalph.reminders import ReminderEngine, ReminderState
from openalph.session import SessionLog

ROOM = "!gc-runway:matrix.local"
AGENT_ID = "@gc-runway-agent:matrix.local"


# ---------------------------------------------------------------------------
# Local entry builders (self-contained; mirrors test_context_gc.py helpers)
# ---------------------------------------------------------------------------

def _user(content, source=None):
    e = {"role": "user", "content": content}
    if source:
        e["source"] = source
    return e


def _assistant(content="", tool_calls=None, thinking=None):
    e = {"role": "assistant"}
    if content:
        e["content"] = content
    if tool_calls:
        e["tool_calls"] = tool_calls
    if thinking:
        e["thinking"] = thinking
    return e


def _tc(call_id, name, input=None):
    return {"call_id": call_id, "name": name, "input": input or {}}


def _tool(call_id, name, output):
    return {"role": "tool", "call_id": call_id, "name": name, "output": output}


def _log(tmp_path):
    return SessionLog(tmp_path, AGENT_ID)


def _append_all(log, entries):
    for e in entries:
        kw = {"role": e["role"], "sender": e.get("sender", AGENT_ID), "room": ROOM}
        kw.update({k: v for k, v in e.items() if k not in ("role", "sender", "room")})
        log.append(**kw)


def _entries():
    return [
        _user("hello"),
        _assistant("let me look", tool_calls=[_tc("c1", "file_read", {"path": "/tmp/big.md"})],
                   thinking="deep"),
        _tool("c1", "file_read", "X" * 5000),
        _assistant("", thinking="more"),
        _user("next"),
    ]


def _apply(log, tmp_path, **kw):
    defaults = dict(
        workspace=tmp_path, trigger="manual", window=1_000_000,
        budget_pct=0.25, budget_min=96000,
        max_tokens=0, handoff_pct=0.10, handoff_min=24000,
        bd_path=None,  # never raise real beads from a test
    )
    defaults.update(kw)
    return apply_boundary(log, ROOM, **defaults)


def _directives(log):
    return [e for e in log.read(ROOM)
            if e.get("source") == "reminder"
            and e.get("trigger") == GC_FORCED_HANDOFF_TRIGGER]


BASE_TOML = '''
[agent]
name = "gc-runway-agent"
default_model = "anthropic/claude-sonnet-4-20250514"
max_tokens = 8192

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
'''


def _toml(tmp_path, body):
    p = tmp_path / "agent.toml"
    p.write_text(body)
    return load_config(p)


# ============================================================================
# Config surface (spec D3, D4)
# ============================================================================

class TestRunwayHandoffConfig:
    def test_new_defaults(self, tmp_path):
        c = _toml(tmp_path, BASE_TOML).context
        assert c.durable_budget_pct == 25.0
        assert c.durable_budget_min_tokens == 96000
        assert c.handoff_runway_pct == 10.0
        assert c.handoff_runway_min_tokens == 24000

    def test_overrides_accepted(self, tmp_path):
        body = BASE_TOML + '''
[context]
handoff_runway_pct = 5.0
handoff_runway_min_tokens = 8000
durable_budget_pct = 30.0
durable_budget_min_tokens = 128000
'''
        c = _toml(tmp_path, body).context
        assert c.handoff_runway_pct == 5.0
        assert c.handoff_runway_min_tokens == 8000
        assert c.durable_budget_pct == 30.0
        assert c.durable_budget_min_tokens == 128000

    @pytest.mark.parametrize("line", [
        "handoff_runway_pct = 0",
        "handoff_runway_pct = 100",
        "handoff_runway_pct = -3.0",
        'handoff_runway_pct = "lots"',
        "handoff_runway_min_tokens = -1",
        'handoff_runway_min_tokens = "many"',
    ])
    def test_invalid_handoff_values_fail_loud(self, tmp_path, line):
        with pytest.raises(ConfigError):
            _toml(tmp_path, BASE_TOML + f"[context]\n{line}\n")


# ============================================================================
# Handoff semantics (spec D1: runway-gated, decoupled from durable-set size)
# ============================================================================

class TestRunwayHandoffSemantics:
    def _fat_durable(self, tmp_path, chars=4000):
        f = tmp_path / "skills" / "big.md"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("Z" * chars)
        return ["skills/big.md"]

    def test_over_budget_ample_runway_is_marker_only(self, tmp_path):
        """Spec D2/D1: over budget + abundant runway -> NO directive, NO bead
        call, full verbatim attachment, informational marker."""
        log = _log(tmp_path)
        paths = self._fat_durable(tmp_path)
        _append_all(log, _entries())
        # window 1M with max_tokens 0 -> runway ~1M after a small residue;
        # budget tiny so the set is over budget.
        r = _apply(log, tmp_path, config_paths=paths,
                   budget_pct=0.000001, budget_min=10)
        assert r["applied"] is True
        assert r["over_budget"] is True
        assert r["handoff_advised"] is False
        assert r["manifest"]["runway"]["handoff_advised"] is False
        # Structural: marker + snapshot appended, no reminder directive.
        assert _directives(log) == []
        snap = log.read(ROOM)[6]["content"]
        assert "Z" * 100 in snap  # attached verbatim, never degraded
        assert "over reinjection budget" in snap.lower()
        assert "informational" in snap.lower()
        assert "OVER BUDGET" not in snap  # alarm wording retired with demotion
        # No session-handoff directive language leaks into the snapshot.
        assert "execute session-handoff now" not in snap

    def test_over_budget_low_runway_fires_handoff(self, tmp_path):
        """Spec D1: over budget + exhausted runway -> directive with runway
        text, once per epoch, bead raised."""
        log = _log(tmp_path)
        paths = self._fat_durable(tmp_path, chars=200000)
        _append_all(log, _entries())
        # window 60K: snapshot (~50K tokens of Zs) leaves runway_after far
        # below max(60K*0.10=6K, 24K) -> handoff_advised.
        r = _apply(log, tmp_path, config_paths=paths,
                   window=60000, budget_pct=0.25, budget_min=10)
        assert r["over_budget"] is True
        assert r["handoff_advised"] is True
        assert r["forced_handoff"] is True
        d = _directives(log)
        assert len(d) == 1
        text = d[0]["content"]
        assert "&lt;system-reminder&gt;" in text  # literal tags (non-negotiable)
        assert "runway" in text.lower()
        assert "session-handoff" in text
        assert "%" in text  # states the consumed percentage
        # Latch: a second boundary same epoch does not re-fire.
        _append_all(log, [_user("more")])
        r2 = _apply(log, tmp_path, config_paths=paths,
                    window=60000, budget_pct=0.25, budget_min=10)
        assert r2["forced_handoff"] is False
        assert len(_directives(log)) == 1

    def test_under_budget_low_runway_still_fires(self, tmp_path):
        """Spec D1/decoupling: handoff must NOT require durable-set pressure.
        A room whose post-boundary residue is nearly all runway — with NO
        durable set at all — must still trigger the handoff."""
        log = _log(tmp_path)
        entries = _entries()
        # Fat in-flight tail stays post-boundary (exclude_inflight):
        # un-reducible residue.
        entries.append(_assistant("Q" * 20000, thinking="hold",
                                  tool_calls=[_tc("c9", "shell", {"command": "ls"})]))
        _append_all(log, entries)
        # window 20000, max_tokens 2000 -> available 18000. Residue tail
        # ~5K tokens + dirs; threshold = max(2000, 24000) = 24000 ->
        # runway_after (~13K) < 24000 -> handoff with zero durable set.
        r = _apply(log, tmp_path, exclude_inflight=True,
                   window=20000, max_tokens=2000)
        assert r["over_budget"] is False
        assert r["handoff_advised"] is True
        assert r["forced_handoff"] is True
        assert len(_directives(log)) == 1

    def test_under_budget_ample_runway_silent(self, tmp_path):
        """Baseline: no pressure anywhere -> nothing fires."""
        log = _log(tmp_path)
        _append_all(log, _entries())
        r = _apply(log, tmp_path)
        assert r["over_budget"] is False
        assert r["handoff_advised"] is False
        assert r.get("forced_handoff") is not True
        assert _directives(log) == []
        assert not any(e.get("trigger") == GC_FORCED_HANDOFF_TRIGGER
                       for e in log.read(ROOM))

    def test_threshold_dominance_pct_vs_min(self, tmp_path):
        """threshold = max(window*pct, min): min binds on small windows,
        pct binds on large. Same residue flips handoff_advised between the
        two regimes."""
        log = _log(tmp_path)
        _append_all(log, _entries())
        # Large window -> pct dominates (0.10*1M = 100K > 24K min).
        r_big = _apply(log, tmp_path, window=1_000_000, max_tokens=700_000)
        # available 300K, residue tiny -> ample runway under 100K threshold?
        # threshold 100K, runway_after ~300K -> no fire.
        assert r_big["handoff_advised"] is False

        log2 = _log(tmp_path / "s2")
        (tmp_path / "s2").mkdir()
        _append_all(log2, _entries())
        # Same residue, small window -> min (24000) dominates and exceeds
        # available entirely -> fire.
        r_small = _apply(log2, tmp_path / "s2", window=8000, max_tokens=0,
                         handoff_pct=0.001, handoff_min=24000)
        assert r_small["handoff_advised"] is True

    def test_equality_is_no_fire(self, tmp_path):
        """runway_after == threshold -> no fire; runway_after == threshold-1
        -> fire (strict <). Two-phase: measure the deterministic composite
        tokens_after (residual + framed snapshot + tail) on a sibling log
        first — the fixture is byte-identical every time — then size the
        window so runway_after lands exactly at threshold."""
        threshold = 5000
        # Phase 1: measure the composite on an identical fixture.
        log0 = _log(tmp_path / "m")
        (tmp_path / "m").mkdir()
        _append_all(log0, _entries())
        m0 = _apply(log0, tmp_path / "m")
        measured = m0["manifest"]["runway"]["tokens_after"]
        assert measured > 0
        window_eq = measured + threshold
        # Phase 2a: window sized so runway_after == threshold -> NO fire.
        log_eq = _log(tmp_path / "eq")
        (tmp_path / "eq").mkdir()
        _append_all(log_eq, _entries())
        r_eq = apply_boundary(
            log_eq, ROOM, workspace=tmp_path / "eq", trigger="manual",
            window=window_eq, budget_pct=0.25, budget_min=96000,
            max_tokens=0, handoff_pct=0.0000001, handoff_min=threshold,
            bd_path=None)
        assert r_eq["manifest"]["runway"]["runway_after"] == threshold
        assert r_eq["handoff_advised"] is False
        # Phase 2b: one token less runway -> fire.
        log_lt = _log(tmp_path / "lt")
        (tmp_path / "lt").mkdir()
        _append_all(log_lt, _entries())
        r_lt = apply_boundary(
            log_lt, ROOM, workspace=tmp_path / "lt", trigger="manual",
            window=window_eq - 1, budget_pct=0.25, budget_min=96000,
            max_tokens=0, handoff_pct=0.0000001, handoff_min=threshold,
            bd_path=None)
        assert r_lt["manifest"]["runway"]["runway_after"] == threshold - 1
        assert r_lt["handoff_advised"] is True

    def test_manifest_runway_block(self, tmp_path):
        """Manifest gains the runway block with all five fields; durable
        block shape unchanged."""
        log = _log(tmp_path)
        _append_all(log, _entries())
        _apply(log, tmp_path, window=300_000, max_tokens=48_000)
        m = json.loads(log.read(ROOM)[5]["detail"])
        rw = m["runway"]
        assert rw["available"] == 300_000 - 48_000
        # tokens_after is the POST-BOUNDARY RENDER estimate: the expunged-span
        # residual (tokens_after_est) PLUS the framed snapshot bytes always
        # appended with the marker (tail is empty in this fixture).
        snap = log.read(ROOM)[6]["content"]
        assert rw["tokens_after"] == m["tokens_after_est"] + len(snap) // 4
        assert rw["runway_after"] == rw["available"] - rw["tokens_after"]
        assert rw["threshold_tokens"] == max(int(300_000 * 0.10), 24000)
        assert rw["handoff_advised"] is False
        d = m["durable"]
        for key in ("project", "files", "budget_tokens", "used_tokens",
                    "over_budget"):
            assert key in d

    def test_handoff_bead_raised_only_on_handoff(self, tmp_path, monkeypatch):
        """Bead raise still exists — but ONLY for a real runway handoff,
        never for over-budget alone (spec D2/D6: kill the spam)."""
        import openalph.context_gc as gcmod
        calls = []

        class _FakeCompleted:
            returncode = 0

        monkeypatch.setattr(gcmod.subprocess, "run",
                            lambda *a, **kw: calls.append(a) or _FakeCompleted())
        # Ample runway + over budget -> no bead.
        log = _log(tmp_path)
        f = tmp_path / "skills" / "big.md"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("Z" * 4000)
        _append_all(log, _entries())
        r = _apply(log, tmp_path, config_paths=["skills/big.md"],
                   budget_pct=0.000001, budget_min=10,
                   bd_path=gcmod.BD_PATH)
        assert r["over_budget"] is True
        assert r["handoff_advised"] is False
        assert calls == []
        # Real runway handoff -> one bead, retitled.
        log2 = _log(tmp_path / "s2")
        (tmp_path / "s2").mkdir()
        f2 = tmp_path / "s2" / "skills" / "big.md"
        f2.parent.mkdir(parents=True, exist_ok=True)
        f2.write_text("Z" * 200000)
        _append_all(log2, _entries())
        r2 = _apply(log2, tmp_path / "s2", config_paths=["skills/big.md"],
                    window=60000, budget_pct=0.25, budget_min=10,
                    bd_path=gcmod.BD_PATH)
        assert r2["handoff_advised"] is True
        assert len(calls) == 1
        argv = calls[0][0]
        assert "create" in argv and "handoff" in argv
        assert any("runway" in str(a).lower() for a in argv)


# ============================================================================
# Snapshot marker (spec D2: informational reword)
# ============================================================================

class TestSnapshotMarkerReword:
    def test_informational_marker(self, tmp_path):
        f = tmp_path / "skills" / "bar.md"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("rule one\n")
        r = resolve_durable_set(tmp_path, None, config_paths=["skills/*.md"])
        s = frame_snapshot(6, r, over_budget=True, budget_tokens=100)
        # New wording: informational, names the reworded concept, keeps the
        # token counts, carries prune guidance — and drops the retired alarm.
        assert "over reinjection budget" in s.lower()
        assert "informational" in s.lower()
        assert "10/100" in s or "/100" in s  # used/budget counts retained
        assert "prune" in s.lower()
        assert "OVER BUDGET" not in s

    def test_under_budget_line_unchanged_shape(self, tmp_path):
        f = tmp_path / "skills" / "bar.md"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("rule one\n")
        r = resolve_durable_set(tmp_path, None, config_paths=["skills/*.md"])
        s = frame_snapshot(6, r, over_budget=False, budget_tokens=48000)
        assert "durable budget" in s and "tokens]" in s
        assert "OVER BUDGET" not in s and "over reinjection" not in s


# ============================================================================
# gc-runway reminder (spec D5: replaces gc-budget)
# ============================================================================

def _state(**kw):
    defaults = dict(
        evaluation_point="turn_start",
        iteration=0,
        max_iterations=100,
        context_tokens=10000,
        context_limit=200000,
        completed_turns=0,
        turn_source=None,
        tool_calls_this_turn={},
        tool_calls_session={},
        todo_list=[],
        enabled_tools={"shell", "file_read"},
    )
    defaults.update(kw)
    return ReminderState(**defaults)


class TestGCRunwayReminder:
    def _eng(self, tmp_path):
        from openalph.config import AgentConfig, ProviderConfig
        cfg = AgentConfig(
            name="gc-runway-agent",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=8192,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic", api_key="sk-test",
                base_url=None, quirks=[])},
            workspace=tmp_path,
            max_iterations=100,
            truncation_limit=50000,
            model_max_tokens=200000,
            matrix=None,
        )
        return ReminderEngine(cfg)

    def test_fires_at_90pct_runway_consumed(self, tmp_path):
        eng = self._eng(tmp_path)
        out = [r for r in eng.evaluate(_state(gc_runway_fraction=0.91))
               if r.trigger == "gc-runway"]
        assert len(out) == 1
        text = out[0].text.lower()
        assert "handoff" in text  # plans FOR a handoff, does not force one
        assert "runway" in text or "residue" in text
        assert "91%" in out[0].text  # numeric interpolation, states the pct

    def test_silent_below(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(gc_runway_fraction=0.89)
        assert [r for r in eng.evaluate(st) if r.trigger == "gc-runway"] == []

    def test_silent_no_boundary_seen(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(gc_runway_fraction=0.0)  # restart / never applied
        assert [r for r in eng.evaluate(st) if r.trigger == "gc-runway"] == []

    def test_once_per_session(self, tmp_path):
        eng = self._eng(tmp_path)
        assert [r for r in eng.evaluate(_state(gc_runway_fraction=0.95))
                if r.trigger == "gc-runway"]
        assert [r for r in eng.evaluate(_state(gc_runway_fraction=0.99))
                if r.trigger == "gc-runway"] == []

    def test_turn_start_only(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(evaluation_point="tool_loop_boundary",
                    gc_runway_fraction=0.95)
        assert [r for r in eng.evaluate(st) if r.trigger == "gc-runway"] == []

    def test_reset_rearms(self, tmp_path):
        eng = self._eng(tmp_path)
        assert [r for r in eng.evaluate(_state(gc_runway_fraction=0.95))
                if r.trigger == "gc-runway"]
        eng.reset()
        assert [r for r in eng.evaluate(_state(gc_runway_fraction=0.95))
                if r.trigger == "gc-runway"]

    def test_rehydrate_latches(self, tmp_path):
        eng = self._eng(tmp_path)
        close = "&lt;/system-reminder&gt;"
        eng.rehydrate([
            {"role": "user", "source": "reminder", "trigger": "gc-runway",
             "content": f"&lt;system-reminder&gt;\nplan a handoff\n{close}"},
        ])
        assert [r for r in eng.evaluate(_state(gc_runway_fraction=0.95))
                if r.trigger == "gc-runway"] == []

    def test_gc_budget_trigger_gone(self, tmp_path):
        """Spec D5 removal: the old trigger id must NEVER be produced, under
        any state — including states that exercised the old threshold."""
        eng = self._eng(tmp_path)
        for frac in (0.0, 0.5, 0.9, 1.4):
            out = eng.evaluate(_state(gc_runway_fraction=frac))
            assert [r for r in out if r.trigger == "gc-budget"] == []


# ============================================================================
# Real-path: apply_boundary_and_rebuild + ReminderState field
# ============================================================================

class _StubAgent:
    """Minimal agent double for the shared application path — real
    SessionLog, real callback contract, no mocks of the GC seam itself."""

    def __init__(self, workspace, window, available, history_box):
        from types import SimpleNamespace
        ctx = SimpleNamespace(
            gc_enabled=True, durable_paths=[],
            durable_budget_pct=25.0, durable_budget_min_tokens=96000,
            handoff_runway_pct=10.0, handoff_runway_min_tokens=24000,
        )
        self.config = SimpleNamespace(context=ctx, workspace=workspace,
                                      model_max_tokens=window)
        self._window = window
        self._available = available
        self._history_box = history_box

    def _resolve_model_limit(self, room_id):
        return self._window

    def _effective_available(self, limit):
        # REAL signature (agent.py: `_effective_available(self, limit: int)`)
        # — the audit C1 finding came from this stub pinning a room_id
        # signature and thereby validating a swallowed-TypeError seam.
        assert limit == self._window
        return self._available

    def history(self, room_id):
        return self._history_box


class TestApplyBoundaryAndRebuildRunway:
    def test_runway_plumbed_from_agent_available(self, tmp_path):
        """The one adapter: available must come from
        agent._effective_available (im7t.46 D9 single-source), not re-derived
        from window-only. max_tokens handed to apply_boundary = window −
        available, so runway math matches the overflow guard exactly."""
        log = _log(tmp_path)
        _append_all(log, _entries())
        hist = [e.copy() for e in log.read(ROOM)]
        agent = _StubAgent(tmp_path, window=1_000_000, available=640_000,
                           history_box=hist)
        out = apply_boundary_and_rebuild(agent, log, ROOM,
                                         trigger="manual",
                                         exclude_inflight=False)
        assert out["applied"] is True
        rw = out["manifest"]["runway"]
        assert rw["available"] == 640_000
        assert rw["runway_after"] == rw["available"] - rw["tokens_after"]
        assert rw["handoff_advised"] is False
        # Real in-place rebuild contract (wave 2): the stub's history box was
        # replaced IN PLACE with the boundary-aware render — the frozen
        # snapshot framing and pointer placeholders are present, and the fat
        # pre-boundary tool output is gone.
        assert any("[GC boundary" in str(e.get("content", "")) for e in hist)
        assert any("expunged at GC boundary" in str(e) for e in hist)
        assert not any("X" * 5000 in str(e) for e in hist)

    def test_handoff_advised_surfaces_in_outcome(self, tmp_path):
        log = _log(tmp_path)
        f = tmp_path / "skills" / "big.md"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("Z" * 200000)
        _append_all(log, _entries())
        hist = [e.copy() for e in log.read(ROOM)]
        agent = _StubAgent(tmp_path, window=60000, available=55000,
                           history_box=hist)
        # stub declares a durable path class so the snapshot is fat.
        agent.config.context.durable_paths = ["skills/*.md"]
        out = apply_boundary_and_rebuild(agent, log, ROOM,
                                         trigger="manual",
                                         exclude_inflight=False)
        assert out["applied"] is True
        assert out["handoff_advised"] is True
        assert out["manifest"]["runway"]["handoff_advised"] is True


class TestEffectiveAvailableContract:
    def test_real_agent_signature_is_limit(self):
        """Pin the REAL seam shape so a room_id-style call site can never be
        silently swallowed again (audit C1, kimi+glm convergent)."""
        import inspect
        from openalph.agent import Agent
        params = list(inspect.signature(Agent._effective_available).parameters)
        assert params[:2] == ["self", "limit"]


class TestReminderStateField:
    def test_state_accepts_gc_runway_fraction(self):
        st = _state(gc_runway_fraction=0.5)
        assert st.gc_runway_fraction == 0.5

    def test_state_runway_fraction_defaults_zero(self):
        assert _state().gc_runway_fraction == 0.0
