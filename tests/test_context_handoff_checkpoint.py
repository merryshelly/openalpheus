"""Context handoff — T2 checkpoint turn (kdsn.322, spec §3.2 + §3.3 ladder).

RED SUITE — orchestrator-authored (tdd-orchestration route). The tests ARE
the specification for the reminder-engine half of the handoff rework:
the ``handoff-checkpoint`` trigger (replaces ``gc-warn``), the
``handoff-runway`` trigger rename (replaces ``gc-runway``), the re-arm
seam rename, and the hard epoch on legacy trigger ids.

Interface contract (what T2 must export):
  - ReminderEngine trigger "handoff-checkpoint": fires at BOTH turn_start
    AND tool_loop_boundary (spec §3.3 — corrects gc-warn's turn_start-only
    blind spot, mirroring the im7t.46 ladder fix), once per boundary cycle
    (latched; the applied-boundary seam's engine reset re-arms it),
    threshold = state.checkpoint_threshold (absolute tokens, built by
    agent.py from config.context.checkpoint_pct × available).
  - Directive text: the spec §3.2 fixed template — compels set_active_
    project declaration + progress.md/durable-set.toml update, embeds the
    condensed skeletons. NO tool-name note (the old gc_tool_note seam gated
    on "context_gc" in enabled_tools is DELETED — dead key after T4's
    registry rename, spec §4 feedback item 4).
  - ReminderEngine trigger "handoff-runway" (replaces "gc-runway"):
    predicate UNCHANGED (once per session, ≥90% post-boundary runway
    consumption, silent before any applied boundary); only the id + text
    wording rename.
  - ReminderState fields: gc_warn_threshold → checkpoint_threshold,
    gc_runway_fraction → handoff_runway_fraction (agent.py builds state at
    the turn-start AND boundary sites — both renamed).
  - Re-arm seam: Agent._note_gc_boundary_applied → _note_handoff_boundary_
    applied (the single applied-boundary bookkeeping seam; matrix /cache
    path's getattr renamed with it).
  - Hard epoch on trigger ids: rehydrate() latches ONLY the new ids —
    legacy gc-warn/gc-runway JSONL entries must NOT consume the new
    triggers' once-per-cycle/once-per-session budgets (a pre-migration
    session still gets its checkpoint warning under the new mechanism).

Slice ownership of the old-name absence criterion: T2 (this file) pins
gc-warn / gc-runway / gc_warn_threshold / gc_runway_fraction /
_gc_warn_fired / _gc_runway_fired / _note_gc_boundary_applied absent from
src/**/*.py.
"""

from openalph.reminders import ReminderEngine, ReminderState


class _Cfg:
    """Minimal kill-switch stub: evaluate() reads only config.reminders."""

    reminders = True


def _engine():
    return ReminderEngine(_Cfg())


def _state(**kw):
    base = dict(
        evaluation_point="turn_start",
        iteration=0,
        max_iterations=100,
        context_tokens=1000,
        context_limit=10000,
        completed_turns=3,
        turn_source=None,
        tool_calls_this_turn=0,
        tool_calls_session=0,
        todo_list=None,
        enabled_tools=set(),
        checkpoint_threshold=750,
        handoff_runway_fraction=0.0,
    )
    base.update(kw)
    return ReminderState(**base)


def _evaluate(engine, state):
    return engine.evaluate(state)


# ===========================================================================
# handoff-checkpoint trigger
# ===========================================================================

class TestCheckpointTrigger:
    def test_fires_at_threshold_turn_start(self):
        eng = _engine()
        out = _evaluate(eng, _state(context_tokens=800, checkpoint_threshold=750))
        triggers = [r.trigger for r in out]
        assert "handoff-checkpoint" in triggers

    def test_fires_at_threshold_tool_loop_boundary(self):
        eng = _engine()
        out = _evaluate(eng, _state(
            evaluation_point="tool_loop_boundary",
            context_tokens=800, checkpoint_threshold=750))
        assert any(r.trigger == "handoff-checkpoint" for r in out), (
            "spec §3.3: the checkpoint tier evaluates at BOTH turn-start and "
            "tool-loop-boundary (gc-warn's turn_start-only blind spot is a "
            "corrected defect, not preserved behavior)")

    def test_silent_below_threshold(self):
        eng = _engine()
        assert _evaluate(eng, _state(context_tokens=700, checkpoint_threshold=750)) == []
        assert _evaluate(eng, _state(
            evaluation_point="tool_loop_boundary",
            context_tokens=700, checkpoint_threshold=750)) == []

    def test_zero_threshold_disabled(self):
        eng = _engine()
        assert _evaluate(eng, _state(context_tokens=9999, checkpoint_threshold=0)) == []

    def test_once_per_cycle_both_points(self):
        eng = _engine()
        first = _evaluate(eng, _state(context_tokens=800, checkpoint_threshold=750))
        assert any(r.trigger == "handoff-checkpoint" for r in first)
        # same cycle, either evaluation point: latched
        assert _evaluate(eng, _state(context_tokens=900, checkpoint_threshold=750)) == []
        assert _evaluate(eng, _state(
            evaluation_point="tool_loop_boundary",
            context_tokens=900, checkpoint_threshold=750)) == []

    def test_reset_re_arms(self):
        eng = _engine()
        _evaluate(eng, _state(context_tokens=800, checkpoint_threshold=750))
        eng.reset()
        again = _evaluate(eng, _state(context_tokens=800, checkpoint_threshold=750))
        assert any(r.trigger == "handoff-checkpoint" for r in again)

    def test_directive_text_contract(self):
        eng = _engine()
        out = _evaluate(eng, _state(context_tokens=800, checkpoint_threshold=750))
        rem = [r for r in out if r.trigger == "handoff-checkpoint"][0]
        text = rem.text
        # the spec §3.2 fixed template: declare-or-scaffold + artifact updates
        for needle in ("set_active_project", "progress.md", "durable-set.toml"):
            assert needle in text, f"directive must steer to {needle!r}"
        assert "memory/projects/" in text, "directive must name the scaffold path"
        # spec §3.2 "Template availability" (SB-ratified): cages have no /srv
        # mount — "the inline skeleton is the guarantee". The directive must
        # EMBED condensed skeletons for both artifacts.
        for needle in ("## State", "## Decisions", "## Next",
                       "[[entries]]", "path =", "reason =", "auto-included"):
            assert needle in text, f"directive must embed the skeleton piece {needle!r}"
        # the old tool-note seam is DELETED (spec §4 feedback item 4)
        assert "context_gc" not in text
        assert "context_handoff" not in text, (
            "the directive is a fixed template — it must not name ANY tool "
            "(the registry renames at T4; a tool-name note here is the dead "
            "seam class the spec explicitly retires)")

    def test_old_trigger_id_gone(self):
        eng = _engine()
        out = _evaluate(eng, _state(context_tokens=9999, checkpoint_threshold=750))
        assert all(r.trigger != "gc-warn" for r in out)


# ===========================================================================
# handoff-runway trigger (rename, predicate unchanged)
# ===========================================================================

class TestRunwayTriggerRename:
    def test_fires_at_90pct_consumption(self):
        eng = _engine()
        out = _evaluate(eng, _state(handoff_runway_fraction=0.91))
        assert any(r.trigger == "handoff-runway" for r in out)

    def test_silent_below_90(self):
        eng = _engine()
        assert _evaluate(eng, _state(
            context_tokens=100, handoff_runway_fraction=0.89)) == []

    def test_silent_at_zero_fraction(self):
        eng = _engine()
        assert _evaluate(eng, _state(
            context_tokens=100, handoff_runway_fraction=0.0)) == []

    def test_once_per_session(self):
        eng = _engine()
        first = _evaluate(eng, _state(handoff_runway_fraction=0.95))
        assert any(r.trigger == "handoff-runway" for r in first)
        assert _evaluate(eng, _state(handoff_runway_fraction=0.96)) == []

    def test_reset_re_arms_runway(self):
        eng = _engine()
        _evaluate(eng, _state(handoff_runway_fraction=0.95))
        eng.reset()
        again = _evaluate(eng, _state(handoff_runway_fraction=0.95))
        assert any(r.trigger == "handoff-runway" for r in again)

    def test_old_runway_id_gone(self):
        eng = _engine()
        out = _evaluate(eng, _state(handoff_runway_fraction=0.95))
        assert all(r.trigger != "gc-runway" for r in out)


# ===========================================================================
# Hard epoch: rehydrate latches ONLY the new trigger ids
# ===========================================================================

class TestRehydrateHardEpoch:
    def _entries(self, *triggers):
        return [{"role": "user", "source": "reminder", "trigger": t,
                 "content": "x"} for t in triggers]

    def test_new_ids_latch(self):
        eng = _engine()
        eng.rehydrate(self._entries("handoff-checkpoint", "handoff-runway"))
        assert _evaluate(
            eng, _state(context_tokens=9999, checkpoint_threshold=750,
                        handoff_runway_fraction=0.95)) == []

    def test_legacy_ids_do_not_latch_new_triggers(self):
        eng = _engine()
        eng.rehydrate(self._entries("gc-warn", "gc-runway"))
        out = _evaluate(
            eng, _state(context_tokens=9999, checkpoint_threshold=750,
                        handoff_runway_fraction=0.95))
        triggers = [r.trigger for r in out]
        assert "handoff-checkpoint" in triggers, (
            "hard epoch: a pre-migration session's legacy gc-warn entry must "
            "not consume the new checkpoint trigger's budget")
        assert "handoff-runway" in triggers

    def test_rehydrate_is_idempotent(self):
        eng = _engine()
        eng.rehydrate(self._entries("handoff-checkpoint"))
        eng.rehydrate(self._entries("handoff-checkpoint"))
        assert _evaluate(eng, _state(
            context_tokens=9999, checkpoint_threshold=750)) == []


# ===========================================================================
# Re-arm seam rename + agent wiring
# ===========================================================================

class TestReArmSeam:
    def test_agent_seam_renamed(self):
        from openalph.agent import Agent
        assert hasattr(Agent, "_note_handoff_boundary_applied")
        assert not hasattr(Agent, "_note_gc_boundary_applied")

    def test_applied_boundary_re_arms_checkpoint(self, tmp_path):
        # real-path wiring pin: a boundary applied through the agent's
        # callback consumer re-arms the checkpoint trigger (engine reset at
        # the single bookkeeping seam) — the once-per-CYCLE contract.
        import asyncio

        from openalph.agent import Agent
        from openalph.config import (
            AgentConfig, ContextHandoffConfig, ProviderConfig)

        config = AgentConfig(
            name="ckpt",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=100,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic", api_key="sk-test")},
            workspace=tmp_path,
            context=ContextHandoffConfig(),
        )
        agent = Agent(config)
        ROOM_ID = "!ckpt:test"
        manifest = {"runway": {"available": 900, "tokens_after": 100}}
        applied = {"n": 0}

        async def fake_cb(room_id, *, trigger, exclude_inflight):
            applied["n"] += 1
            return {"applied": True, "noop_reason": None,
                    "manifest": manifest, "over_budget": False}

        engine = agent._engine_for(ROOM_ID)
        engine._checkpoint_fired = True  # cycle consumed

        async def run():
            ok = await agent._gc_apply_boundary(
                ROOM_ID, callbacks={"apply_handoff_boundary": fake_cb},
                trigger="auto", exclude_inflight=False)
            return ok

        # kdsn.322.15: _gc_apply_boundary returns the APPLIED outcome dict
        res = asyncio.run(run())
        assert isinstance(res, dict) and res.get("applied") is True
        assert applied["n"] == 1
        # re-armed: the engine fires again at threshold after the boundary
        out = engine.evaluate(_state(
            context_tokens=800, checkpoint_threshold=750))
        assert any(r.trigger == "handoff-checkpoint" for r in out), (
            "an applied boundary re-arms the checkpoint trigger (once per "
            "boundary CYCLE, not once per session)")

    def test_boundary_site_state_carries_checkpoint_threshold(self, tmp_path):
        """audit-fix (kdsn.322.9): spec 3.3 — checkpoint evaluates at BOTH
        turn_start AND tool_loop_boundary. The boundary site must pass the
        same checkpoint_threshold as the turn-start site (0 only when
        handoff is disabled), or the predicate (gated on
        ``checkpoint_threshold > 0``) stays silent mid-turn and the
        ratified dual-point evaluation is defeated."""
        from openalph.agent import Agent
        from openalph.config import (
            AgentConfig, ContextHandoffConfig, ProviderConfig)

        config = AgentConfig(
            name="bkpt",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=100,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic", api_key="sk-test")},
            workspace=tmp_path,
            context=ContextHandoffConfig(),
        )
        agent = Agent(config)
        ROOM_ID = "!bkpt:test"
        limit = agent._resolve_model_limit(ROOM_ID)
        state = agent._boundary_reminder_state(
            ROOM_ID,
            iteration=3,
            context_tokens=agent._estimate_context_tokens(ROOM_ID),
            limit=limit,
            completed_turns=0,
            turn_source=None,
            tool_calls_this_turn=1,
            tool_calls_session=1,
            todo_list=None,
            enabled_tools=set(),
        )
        expected = int(agent._effective_available(limit)
                       * config.context.checkpoint_pct / 100)
        assert state.checkpoint_threshold == expected
        assert state.checkpoint_threshold > 0
        assert state.evaluation_point == "tool_loop_boundary"
        assert state.available_tokens == agent._effective_available(limit)

    def test_boundary_site_threshold_zero_when_handoff_disabled(self, tmp_path):
        """audit-fix (kdsn.322.9): handoff off → threshold 0 (engine silent),
        mirroring the turn-start site's ``if _gc_cfg.handoff_enabled else 0``."""
        from openalph.agent import Agent
        from openalph.config import (
            AgentConfig, ContextHandoffConfig, ProviderConfig)

        config = AgentConfig(
            name="bkoff",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=100,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic", api_key="sk-test")},
            workspace=tmp_path,
            context=ContextHandoffConfig(handoff_enabled=False),
        )
        agent = Agent(config)
        ROOM_ID = "!bkoff:test"
        state = agent._boundary_reminder_state(
            ROOM_ID,
            iteration=1,
            context_tokens=0,
            limit=agent._resolve_model_limit(ROOM_ID),
            completed_turns=0,
            turn_source=None,
            tool_calls_this_turn=0,
            tool_calls_session=0,
            todo_list=None,
            enabled_tools=set(),
        )
        assert state.checkpoint_threshold == 0

    def test_agent_state_fields_renamed(self, tmp_path):
        # the state-builder sites consume the renamed fields (both eval points)
        import inspect

        from openalph.agent import Agent
        src = inspect.getsource(Agent)
        assert "gc_warn_threshold" not in src
        assert "checkpoint_threshold" in src
        assert "gc_runway_fraction=" not in src
        assert "handoff_runway_fraction=" in src

    def test_matrix_slash_path_uses_renamed_seam(self):
        import inspect

        import openalph.matrix as m
        src = inspect.getsource(m)
        assert "_note_gc_boundary_applied" not in src
        assert "_note_handoff_boundary_applied" in src


# ===========================================================================
# Hard epoch: old spellings absent from src (T2 extension)
# ===========================================================================

class TestOldSpellingsAbsentFromSrc:
    def test_no_legacy_trigger_tokens_in_src(self):
        from _handoff_helpers import scan_src_for_tokens
        hits = scan_src_for_tokens([
            "gc-warn",
            "gc-runway",
            "gc_warn_threshold",
            "gc_runway_fraction",
            "_gc_warn_fired",
            "_gc_runway_fired",
            "_note_gc_boundary_applied",
        ])
        assert not hits, (
            "hard epoch: legacy trigger spellings remain in src:\n"
            + "\n".join(f"  {f}:{i}: {line}" for f, i, line in hits[:20]))
