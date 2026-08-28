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
        self._t2_fired: bool = False        # T2: once per session
        self._t3_fired: bool = False        # T3: once per session
        self._t5_fired: bool = False        # T5: once per session
        self._t6_fired: bool = False        # T6: once per session
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

        # T2: context-pressure — ≥80% of resolved limit; once/session
        # R1-7: T2 fires only at tool_loop_boundary (spec §3/§4)
        if (state.evaluation_point == "tool_loop_boundary"
                and state.context_limit > 0
                and state.context_tokens / state.context_limit >= 0.80
                and not self._t2_fired):
            self._t2_fired = True
            pct = round(state.context_tokens / state.context_limit * 100)
            results.append(Reminder(
                trigger="context-pressure",
                text=(
                    f"Context is at {pct}% of the window. Converge: finish or "
                    "hand off current work, promote unfinished todos to your "
                    "durable tracker, and prepare a session handoff."
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
                self._t2_fired = True
            elif trigger == "memory-salience":
                self._t3_fired = True
            elif trigger == "memory-salience-deep":
                self._t5_fired = True
            elif trigger == "advisor-salience":
                self._t6_fired = True
            elif trigger == "session-orient" and entry.get("detail") is not None:
                # kdsn.298: restore the model-keyed orientation flag from the
                # persisted detail. Last-wins in JSONL order (the final entry
                # is the epoch's last orientation — spec §4.4). Guarded: an
                # entry lacking detail preserves any known-good earlier one.
                self._oriented_model = entry["detail"]
            # T4 is per-turn — not rehydrated across sessions

    def reset(self) -> None:
        """Clear all fired-state (umbral wipe = new session)."""
        self._t1_session_fires = 0
        self._t2_fired = False
        self._t3_fired = False
        self._t5_fired = False
        self._t6_fired = False
        self._t4_fired_this_turn = False
        self._oriented_model = None

    def reset_turn(self) -> None:
        """Clear per-turn fired state (new turn boundary)."""
        self._t4_fired_this_turn = False
