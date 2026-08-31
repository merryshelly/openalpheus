"""Reminder engine for OpenAlph agents.

Provides state-triggered <system-reminder> reminder injections into the
message stream.  Pure logic module — no Matrix, provider, or I/O knowledge.

API surface:
    ReminderEngine(config: AgentConfig)
        .evaluate(state: ReminderState) -> list[Reminder]
        .rehydrate(entries: list[dict]) -> None
        .reset() -> None
        .reset_turn() -> None
    ReminderState  — dataclass snapshot of agent state
    Reminder       — dataclass with .trigger, .text, .content
"""

import math
from dataclasses import dataclass


# Context nudge ladder (workspace-im7t.46 spec): pct of USABLE runway
# (available_tokens = limit − max_tokens, D8) at which each tier escalates.
# Module constants by spec — no per-tier config.
CONTEXT_LADDER_TIERS = (60, 70, 80, 90)

# Escalating guidance per tier (D2: each tier's text subsumes the previous
# tier's).  Spec §5, verbatim — interactive rooms only (D5: sub-agent variant
# descoped; sub-agent loops never touch this engine).
_LADDER_TEXT = {
    1: (
        "Context has passed 60% of usable runway (~{remaining:,} tokens remain). "
        "No action needed yet — keep convergence in mind as you plan the rest "
        "of this session."
    ),
    2: (
        "Context is at 70% of usable runway (~{remaining:,} tokens remain). "
        "Wind down: checkpoint working state to files and avoid opening large "
        "new threads of work."
    ),
    3: (
        "Context is at 80% of usable runway (~{remaining:,} tokens remain). "
        "Converge now: checkpoint working state to files, finish or hand off "
        "current work, promote unfinished todos to your durable tracker, and "
        "prepare a session handoff."
    ),
    4: (
        "Context is at 90% of usable runway (~{remaining:,} tokens remain) — "
        "final warning before context overflow. Stop starting new work or "
        "large context writes: summarize state, execute the handoff, and keep "
        "responses minimal."
    ),
}


@dataclass
class ReminderState:
    """Snapshot of agent state for trigger evaluation.

    All fields are provided by the agent loop at evaluation time.
    The engine never reaches outside this dataclass for state.
    """
    evaluation_point: str           # "turn_start" | "tool_loop_boundary"
    iteration: int                  # 0-indexed iteration within the tool loop
    max_iterations: int
    context_tokens: int
    context_limit: int              # from _resolve_model_limit
    completed_turns: int            # user-role msgs in history incl. current
    turn_source: str | None         # "heartbeat" / "umbral" / None
    tool_calls_this_turn: dict      # {tool_name: count}
    tool_calls_session: dict        # {tool_name: count}
    todo_list: list                 # list of todo item dicts
    enabled_tools: set              # set of tool name strings
    # kdsn.298 session-orient: turn-start site only populates these; the
    # tool-loop-boundary site (and the __sub__ sentinel room) flows defaults —
    # model_resolved == "" == _oriented_model keeps the predicate silent there.
    model_resolved: str = ""        # post-room-override, post-alias-expansion
    model_vision: bool = False
    orient_ts: str = ""             # pre-rendered host-local timestamp string
    # Context nudge ladder (D9): the overflow guard's `available` =
    # limit − max_tokens, injected by BOTH agent.py sites from the same helper
    # expression the guard uses. The engine NEVER reads config.max_tokens
    # itself (the guard's formula may change — im7t.47 / kdsn.258 — and the
    # ladder follows automatically). Default 0 = unknown → the ladder silently
    # skips (fail-safe direction: fewer nudges, never false urgency; also keeps
    # existing ReminderState constructions valid).
    available_tokens: int = 0
    # Context GC (workspace-kdsn.305) — turn-start site only, both fields
    # computed by the agent from [context] config + the room's usable runway.
    # gc_warn_threshold: absolute token count at which the gc-warn reminder
    # fires (int(available * warn_pct / 100)). Default 0 = unknown/disabled →
    # SILENT SKIP, the same fail-safe convention as available_tokens (fewer
    # nudges, never false urgency; also keeps existing constructions valid).
    gc_warn_threshold: int = 0
    # gc_budget_fraction: durable-set tokens used / durable budget (0.0–1.0+),
    # from the last applied boundary's manifest. Default 0.0 = no boundary
    # seen yet (in-memory only; restarts empty) → gc-budget silent.
    gc_budget_fraction: float = 0.0


@dataclass
class Reminder:
    """A single reminder to inject into the message stream."""
    trigger: str
    text: str
    # kdsn.298: generic JSONL plumbing (session-orient carries the resolved
    # model here so rehydrate can restore _oriented_model without parsing the
    # framed text). Sinks pass it through only when not None.
    detail: str | None = None

    @property
    def content(self) -> str:
        """Framed content ready for history append."""
        return f"<system-reminder>\n{self.text}\n</system-reminder>"


class ReminderEngine:
    """Trigger registry with fired-state tracking and cooldown bookkeeping.

    Pure logic: evaluate() is a function of ReminderState plus internal
    fired-state counters.  No I/O, no side effects beyond state mutation.
    """

    def __init__(self, config):
        self._config = config
        # Session-level state
        self._t1_session_fires: int = 0     # T1: ≤2 per session
        self._t3_fired: bool = False        # T3: once per session
        self._t5_fired: bool = False        # T5: once per session
        self._t6_fired: bool = False        # T6: once per session
        # Context GC (kdsn.305): session-scoped fired-state for the two GC
        # triggers (once-per-session latches, re-armed by reset() and
        # rehydrated from JSONL like every other trigger).
        self._gc_warn_fired: bool = False   # gc-warn: once per session
        self._gc_budget_fired: bool = False # gc-budget: once per session
        # Context nudge ladder (replaces the single-shot T2 _t2_fired bool):
        # _ladder_fired — monotonic highest tier fired this session+model
        # (D1: single latch, never fire a tier ≤ one already fired; D3: no
        # re-fire above 90%); D11: re-armed on model switch, cleared on
        # reset().  _ladder_model — the resolved model the latch was armed
        # under (None = un-armed/rehydrated, awaiting the first populated
        # turn-start evaluation).
        self._ladder_fired: int = 0
        self._ladder_model: str | None = None
        # Per-turn state
        self._t4_fired_this_turn: bool = False  # T4: once per turn
        # kdsn.298: model-keyed orientation flag (spec MED-1 — also cleared in
        # reset(); rehydrate() calls reset() first, so a stale value surviving
        # reset would wrongly suppress a deserved fire).
        self._oriented_model: str | None = None

    def evaluate(self, state: ReminderState) -> list[Reminder]:
        """Evaluate all trigger predicates against current state.

        Returns list of Reminders to inject (may be empty).
        Pure function of (state, internal fired-state).
        """
        if not getattr(self._config, "reminders", True):
            return []

        results: list[Reminder] = []

        # session-orient (kdsn.298) — turn_start only. Model-keyed keyed flag
        # IS the per-epoch cap: same-model re-evaluations are suppressed,
        # model switches re-fire (each switch keys once).
        # Suppression forms (test t19 pins both sides):
        #  - fresh engine, unpopulated state (__sub__ sentinel room, or any
        #    construction site flowing defaults): "" vs None → suppressed.
        #  - oriented engine, unpopulated state: "" re-arms (winds the keyed
        #    state back toward un-armed) so the NEXT populated evaluation
        #    still fires — covers restart/history-clear without rehydrate.
        #  - populated state equal to the oriented model: suppressed.
        if (state.evaluation_point == "turn_start"
                and state.model_resolved != self._oriented_model
                and (state.model_resolved or self._oriented_model is not None)):
            self._oriented_model = state.model_resolved
            results.append(Reminder(
                trigger="session-orient",
                detail=state.model_resolved,
                text=(
                    "Session orientation (injected automatically at "
                    "context-epoch start):\n"
                    f"- Session began: {state.orient_ts}\n"
                    f"- Active model: {state.model_resolved}\n"
                    f"- Context window: {state.context_limit:,} tokens\n"
                    f"- Vision: {'yes' if state.model_vision else 'no'}"
                ),
            ))

        # T1: todo-nudge — boundary, iteration==5, todo_write enabled,
        # no todo_write call this turn, no in_progress item; ≤2/session
        if (state.evaluation_point == "tool_loop_boundary"
                and state.iteration == 5
                and "todo_write" in state.enabled_tools
                and state.tool_calls_this_turn.get("todo_write", 0) == 0
                and not any(item.get("status") == "in_progress"
                            for item in state.todo_list)
                and self._t1_session_fires < 2):
            self._t1_session_fires += 1
            results.append(Reminder(
                trigger="todo-nudge",
                text=(
                    "You are 5 tool calls into this turn with no active todo "
                    "item. If this task is multi-step, record a plan with "
                    "todo_write before continuing."
                ),
            ))

        # Context-pressure ladder (workspace-im7t.46; supersedes the wave-1
        # single-shot T2).  Fires at BOTH evaluation points (turn_start and
        # tool_loop_boundary — D6: no turn-source gating).  Predicate (spec §3):
        #   1. available_tokens > 0 and context_limit > 0 (D9: the engine reads
        #      ONLY state.available_tokens — default 0 = unknown → silent
        #      skip; degenerate limit → silent skip);
        #   2. D10 integer cross-multiplication, inclusive:
        #      context_tokens * 100 >= tier_pct * available_tokens — no float
        #      thresholds (0.7 is inexact in binary; float comparisons flake
        #      off-by-one at every tier);
        #   3. tier > _ladder_fired (D1: monotonic highest-tier latch — a
        #      multi-tier jump fires ONLY the highest crossed tier; D3: no
        #      re-fire, no above-90% churn).
        if (state.available_tokens > 0
                and state.context_limit > 0):
            # D11: model-switch re-arm — /model changes what `available` IS,
            # but the latch is one-way; a latch=4 armed under a 262K window
            # would silently suppress ALL nudges after switching to a larger
            # model. Rule: when state.model_resolved is non-empty and differs
            # from the model the latch was armed under, re-arm to the highest
            # tier CURRENTLY exceeded (0 if none) and record the new model.
            # model_resolved == "" (the boundary site, kdsn.298) NEVER re-arms
            # — only turn-start re-arms, which is correct because /model
            # applies between turns.  _ladder_model is None after rehydrate()
            # → the first populated turn-start evaluation re-arms to current
            # reality (accepted gray zone: a room already past 60% at restart
            # may see one early-tier duplicate).
            if (state.model_resolved
                    and state.model_resolved != self._ladder_model):
                self._ladder_fired = 0
                for _tier, _pct in enumerate(CONTEXT_LADDER_TIERS, start=1):
                    if (state.context_tokens * 100
                            >= _pct * state.available_tokens):
                        self._ladder_fired = _tier
                self._ladder_model = state.model_resolved
            fired_tier = 0
            for tier, tier_pct in enumerate(CONTEXT_LADDER_TIERS, start=1):
                if (state.context_tokens * 100
                        >= tier_pct * state.available_tokens
                        and tier > self._ladder_fired):
                    fired_tier = tier
            if fired_tier:
                self._ladder_fired = fired_tier
                remaining = max(
                    0, state.available_tokens - state.context_tokens)
                results.append(Reminder(
                    trigger="context-pressure",
                    detail=f"tier={fired_tier}",
                    text=_LADDER_TEXT[fired_tier].format(remaining=remaining),
                ))

        # Context GC — gc-warn (workspace-kdsn.305).  turn_start ONLY (the
        # agent applies the auto boundary tier at turn start; mid-loop
        # warnings would dilute the nudge ladder and can't be acted on
        # between tool calls).  Predicate: gc_warn_threshold > 0 (0 =
        # unknown/disabled → silent skip, the available_tokens convention)
        # AND context has crossed it; once per session (rehydratable via
        # source='reminder' trigger='gc-warn' entries, reset() re-arms).
        # Text teaches the continuity artifacts and — per audit R1-6 — names
        # the context_gc tool ONLY when the tool is actually enabled for
        # this room (enabling is transport-wired, so a headless room must
        # never be steered to a tool it cannot call).
        if (state.evaluation_point == "turn_start"
                and not self._gc_warn_fired
                and state.gc_warn_threshold > 0
                and state.context_tokens >= state.gc_warn_threshold):
            self._gc_warn_fired = True
            if "context_gc" in state.enabled_tools:
                gc_tool_note = " Use the context_gc tool to apply a boundary."
            else:
                gc_tool_note = ""
            results.append(Reminder(
                trigger="gc-warn",
                text=(
                    "Context has crossed the GC warning threshold — finalize "
                    "your continuity artifacts now: update progress.md and "
                    "the durable-set.toml so the next garbage-collection "
                    f"boundary re-injects current state.{gc_tool_note}"
                ),
            ))

        # Context GC — gc-budget (kdsn.305).  turn_start ONLY, once per
        # session.  Predicate: durable-set usage >= 50% of its budget
        # (integer cross-multiplication, like the ladder — no float
        # comparison at the 0.5 boundary).  0.0 = no boundary seen yet
        # (in-memory cache, empty after restart) → silent skip.  States the
        # percentage so the operator can see how close the durable set is to
        # its re-injection budget.
        if (state.evaluation_point == "turn_start"
                and not self._gc_budget_fired
                and state.gc_budget_fraction >= 0
                and int(state.gc_budget_fraction * 200) >= 100):
            self._gc_budget_fired = True
            results.append(Reminder(
                trigger="gc-budget",
                text=(
                    f"Your durable set uses {int(state.gc_budget_fraction * 100)}% "
                    "of its re-injection budget. Prune it: drop durable-set "
                    "entries that are no longer load-bearing and keep "
                    "progress.md within ~1–2K tokens."
                ),
            ))

        # T3: memory-salience — turn_start, user-sourced, completed≥2,
        # memory_search enabled, zero memory_search calls; once/session
        if (state.evaluation_point == "turn_start"
                and state.turn_source is None
                and state.completed_turns >= 2
                and "memory_search" in state.enabled_tools
                and state.tool_calls_session.get("memory_search", 0) == 0
                and not self._t3_fired):
            self._t3_fired = True
            results.append(Reminder(
                trigger="memory-salience",
                text=(
                    "You have not consulted memory this session. Before "
                    "asserting anything about prior work, decisions, dates, "
                    "people, or preferences, run memory_search."
                ),
            ))

        # T5: memory-salience-deep — boundary, ≥50K absolute tokens,
        # memory_search enabled, zero memory_search calls this session,
        # user-sourced turn; once per session.  Absolute tokens, NOT % of
        # window: salience tracks information mass ingested (v1.1 spec,
        # kdsn.186.22).  t5_fired_now defers T6 by one boundary so the two
        # never fire at the same boundary (reminders dilute).
        t5_fired_now = False
        if (state.evaluation_point == "tool_loop_boundary"
                and state.context_tokens >= 50000
                and "memory_search" in state.enabled_tools
                and state.tool_calls_session.get("memory_search", 0) == 0
                and state.turn_source is None
                and not self._t5_fired):
            self._t5_fired = True
            t5_fired_now = True
            results.append(Reminder(
                trigger="memory-salience-deep",
                text=(
                    "You are deep into this session and have not consulted "
                    "memory. Before asserting anything about prior work, "
                    "decisions, dates, people, or preferences, run "
                    "memory_search."
                ),
            ))

        # T6: advisor-salience — boundary, ≥75K absolute tokens, advisor
        # enabled, zero advisor calls this session, user-sourced turn
        # (interactive-only is load-bearing: automated sessions are rote by
        # nature, SB 2026-08-05); once per session.  Deferred one boundary
        # when T5 fires at the same boundary (v1.1 spec, kdsn.186.24).
        if (state.evaluation_point == "tool_loop_boundary"
                and state.context_tokens >= 75000
                and "advisor" in state.enabled_tools
                and state.tool_calls_session.get("advisor", 0) == 0
                and state.turn_source is None
                and not t5_fired_now
                and not self._t6_fired):
            self._t6_fired = True
            results.append(Reminder(
                trigger="advisor-salience",
                text=(
                    "You are deep into a substantial task and have not "
                    "consulted the advisor. Before a non-obvious design "
                    "decision, a first substantive write, or declaring "
                    "complex work done, a second-model opinion is cheap "
                    "insurance — consider the advisor tool."
                ),
            ))

        # T4: iteration-budget — boundary, iteration==floor(0.8*max); once/turn
        threshold = math.floor(0.8 * state.max_iterations)
        if (state.evaluation_point == "tool_loop_boundary"
                and state.iteration == threshold
                and not self._t4_fired_this_turn):
            self._t4_fired_this_turn = True
            results.append(Reminder(
                trigger="iteration-budget",
                text=(
                    f"You have used {state.iteration} of "
                    f"{state.max_iterations} tool calls this turn. Converge "
                    "now: complete the task, checkpoint state to a file, or "
                    "delegate the remainder."
                ),
            ))

        return results

    def rehydrate(self, entries: list[dict]) -> None:
        """Restore fired-state from JSONL entries.

        Scans entries for source='reminder' and updates internal counters.
        Called on session resume (before any evaluate calls).
        R1-8: reset() at top makes rehydrate idempotent — safe to call twice.
        """
        self.reset()
        for entry in entries:
            if entry.get("source") != "reminder":
                continue
            trigger = entry.get("trigger")
            if trigger == "todo-nudge":
                self._t1_session_fires += 1
            elif trigger == "context-pressure":
                # Ladder (replaces the old _t2_fired = True):
                #   - new writes carry detail="tier=<n>" → latch that tier;
                #   - legacy wave-1 entries have no detail — the old T2 fired
                #     at ≥80% of the FULL window ≈ late in usable-runway
                #     terms, so latch tier 3: do not re-ping tiers 1–2, tier 4
                #     stays available (spec §3 state migration);
                #   - max-wins across entries, never last-wins (monotonic by
                #     construction, D1).
                detail = entry.get("detail")
                if isinstance(detail, str) and detail.startswith("tier="):
                    try:
                        tier = int(detail[5:])
                    except ValueError:
                        tier = 0
                    tier = min(max(tier, 0), len(CONTEXT_LADDER_TIERS))
                else:
                    tier = 3
                self._ladder_fired = max(self._ladder_fired, tier)
            elif trigger == "memory-salience":
                self._t3_fired = True
            elif trigger == "memory-salience-deep":
                self._t5_fired = True
            elif trigger == "advisor-salience":
                self._t6_fired = True
            elif trigger == "gc-warn":
                # Once-per-session latch (kdsn.305): a persisted gc-warn entry
                # means it already fired this session — do not re-ping.
                self._gc_warn_fired = True
            elif trigger == "gc-budget":
                # Same once-per-session latch.
                self._gc_budget_fired = True
            elif trigger == "session-orient" and entry.get("detail") is not None:
                # kdsn.298: restore the model-keyed orientation flag from the
                # persisted detail. Last-wins in JSONL order (the final entry
                # is the epoch's last orientation — spec §4.4). Guarded: an
                # entry lacking detail preserves any known-good earlier one.
                self._oriented_model = entry["detail"]
            # T4 is per-turn — not rehydrated across sessions
        # D11: the latch is armed under UNKNOWN model identity — the first
        # populated turn-start evaluation re-arms to current reality.
        self._ladder_model = None

    def reset(self) -> None:
        """Clear all fired-state (umbral wipe = new session)."""
        self._t1_session_fires = 0
        self._t3_fired = False
        self._t5_fired = False
        self._t6_fired = False
        self._gc_warn_fired = False   # kdsn.305: re-arm both GC triggers
        self._gc_budget_fired = False
        self._t4_fired_this_turn = False
        # Ladder re-arm (D7): umbral = new session, all tiers re-arm.
        self._ladder_fired = 0
        self._ladder_model = None
        self._oriented_model = None

    def reset_turn(self) -> None:
        """Clear per-turn fired state (new turn boundary).

        The ladder latch is session-scoped (D4: one-way for the session) —
        reset_turn() deliberately does NOT touch it.
        """
        self._t4_fired_this_turn = False
