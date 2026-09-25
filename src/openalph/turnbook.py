"""Declare-done turn ledger (workspace-kdsn.179) — shared bookend emission.

Design contract: memory/projects/openalph/declare-done/design-memo.md §4
(SB-ratified 2026-09-24). Every turn on every funnel — live message,
heartbeat, umbral, subagent-completion (the latter two delegate to the
heartbeat funnel, so `_process_message` + `_run_heartbeat_turn` cover all
four) — books exactly ONE `turn.started` and ONE `turn.finished` system
entry into the room's session JSONL, sharing a per-turn `turn_id`, plus
ONE plain every-turn m.notice line (no threshold; the channel is ambient,
the word carries the status — no red/green dependence).

Invariants honoured here (June's, adopted):
- FAIL-SOFT: every emitter in this module is guarded so a booking failure
  never crashes or blocks the turn.
- NEVER FABRICATE: this module only WRITES what a funnel tells it. A crash
  between bookends leaves a stale `turn.started` — absence of
  `turn.finished` IS the crash signal (a watchdog/scan marks it
  `abandoned` later; that scan is out of this bead's scope).
- CONCLUSION taxonomy (closed vocabulary): declared | undeclared |
  cap_exhausted | overflow | cancelled | error | abandoned.

The funnels supply the semantics (which exception class they observed);
this module supplies the shape (entry kwargs land verbatim in the JSONL,
same all-keyword `SessionLog.append` shape as `cache_warning` /
`subagent_dispatched`).
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from openalph.agent import ContextOverflowError
from openalph.provider import ProviderError

logger = logging.getLogger(__name__)


def new_turn_id() -> str:
    """Per-TURN id (not per-process): uuid4 hex, first 16 chars.

    Concurrent rooms therefore always book distinct ids (L9).
    """
    return uuid.uuid4().hex[:16]


#: Closed vocabulary of LANDING markers the agent's
#: `last_turn_declaration(room_id)` may report (design §4 taxonomy,
#: minus the funnel-booked classes overflow/cancelled/error/abandoned
#: that no marker can carry). Shared by this ledger seam and the cli
#: exec conclusion field so the two consumers cannot drift.
MARKER_VOCABULARY = frozenset({"declared", "undeclared", "cap_exhausted"})


def conclusion_from_marker(agent, room_id: str) -> str:
    """Map the agent's `last_turn_declaration(room_id)` marker to a
    closed-vocabulary conclusion.

    "declared" | "undeclared" | "cap_exhausted" pass through as-is;
    `None` (tool unregistered / no declaration — Camp-B ending) maps to
    "undeclared": absence of declaration is always an anomaly class. A
    missing method, a raising call, a non-str result (e.g. a MagicMock
    auto-attribute on a stub agent), or an OUT-OF-TAXONOMY string
    (R9b: the closed vocabulary is enforced at the seam, never
    trusted from the marker) all degrade to "undeclared" — a ledger
    booking must never die on a marker read, and a malformed marker
    must never fabricate a taxonomy class.
    """
    fn = getattr(agent, "last_turn_declaration", None)
    if not callable(fn):
        return "undeclared"
    try:
        marker = fn(room_id)
    except Exception:
        logger.warning(
            "last_turn_declaration failed for %s (non-fatal); booking undeclared",
            room_id, exc_info=True,
        )
        return "undeclared"
    if marker is None:
        return "undeclared"
    if isinstance(marker, str) and marker in MARKER_VOCABULARY:
        return marker
    logger.warning(
        "last_turn_declaration returned out-of-taxonomy marker %r for %s "
        "(non-fatal); booking 'undeclared'", marker, room_id,
    )
    return "undeclared"


def classify_error(exc: BaseException) -> tuple[str, str | None]:
    """Map a funnel-observed exception to (conclusion, error_type).

    CancelledError  -> ("cancelled", None)          (/stop, shutdown, umbral pre-wipe)
    ContextOverflow -> ("overflow", None)           (agent.py, matrix.py: AgentOverflowError)
    ProviderError   -> ("error", "provider")
    anything else   -> ("error", <exception class name>)
    """
    if isinstance(exc, asyncio.CancelledError):
        return ("cancelled", None)
    if isinstance(exc, ContextOverflowError):
        return ("overflow", None)
    if isinstance(exc, ProviderError):
        return ("error", "provider")
    return ("error", type(exc).__name__)


# Prepended per-conclusion shapes for the end notice (SB 2026-09-25):
# visual consistency with the house tool-call notices (each class reads
# as a tool-call line at a glance). Shape-distinct, no red/green
# dependence — the SHAPE (with the word) IS the key, not the color.
CONCLUSION_ICONS = {
    "declared": "🏁",       # finish flag — clean terminal act
    "undeclared": "⚠️",  # the anomaly class
    "cap_exhausted": "⏳",      # iteration budget exhausted
    "overflow": "⚡",           # context breach
    "cancelled": "⛔",          # operator / control-path stop
    "error": "❌",              # provider / unknown exception
    "abandoned": "👻",      # stale started, found later — ghost
}


def turn_end_notice(conclusion: str, error_type: str | None = None) -> str:
    """The ONE plain every-turn m.notice line (matrix house notice channel).

    Closed vocabulary: conclusion-shape icon + fixed prefix + the
    conclusion word (+ the error_type for the error class). No elapsed
    threshold, no red/green dependence — the word IS the status. An
    out-of-vocabulary conclusion would be a bug: it renders with the ❓
    anomaly icon so it is visibly wrong rather than silently un-decorated.
    """
    icon = CONCLUSION_ICONS.get(conclusion, "❓")
    line = f"{icon} Turn ended — {conclusion}"
    if error_type:
        line += f" ({error_type})"
    return line


def _resolve_user_id(session_log, fallback) -> str:
    """Sender for system entries: the agent's own user id (the
    `subagent_dispatched` house shape), falling back defensively."""
    for candidate in (
        getattr(session_log, "agent_user_id", None),
        fallback,
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return "agent"


def _book(session_log, event: str, **fields) -> bool:
    """Fail-soft system-entry append. Never raises; returns whether the
    write actually landed (a caller chaining idempotency marks — e.g. the
    matrix `_turn_end` — must mark ONLY on True: a silently-failed booking
    must not permanently suppress a later compensating one, remediation
    re-audit F3).

    All-keyword `SessionLog.append` (the house shape: cache_warning at
    matrix.py:3133, subagent_dispatched at agent.py:1172) — kwargs land
    verbatim in the JSONL line after ts/role/sender/room/event_id.
    """
    if session_log is None:
        return False
    try:
        session_log.append(
            role="system",
            sender=_resolve_user_id(session_log, fields.get("user_id")),
            room=fields["room_id"],
            event_id=None,
            event=event,
            **{k: v for k, v in fields.items() if k != "user_id" and v is not None},
        )
    except Exception:
        logger.warning(
            "turn ledger booking %s failed (non-fatal; the turn goes on)",
            event, exc_info=True,
        )
        return False
    return True


def book_started(
    session_log,
    *,
    user_id,
    room_id: str,
    turn_id: str,
    origin: str,
    agent_name: str | None = None,
    trigger_event_id: str | None = None,
) -> None:
    """Book the turn's `turn.started` entry (fail-soft)."""
    _book(
        session_log,
        "turn.started",
        user_id=user_id,
        room_id=room_id,
        turn_id=turn_id,
        origin=origin,
        agent_name=agent_name,
        trigger_event_id=trigger_event_id,
    )


def book_finished(
    session_log,
    *,
    user_id,
    room_id: str,
    turn_id: str,
    conclusion: str,
    origin: str,
    elapsed_s: float,
    error_type: str | None = None,
    stop_reason: str | None = None,
    usage: dict | None = None,
) -> bool:
    """Book the turn's `turn.finished` entry (fail-soft; returns whether
    the write landed).

    Carries the per-room `stop_reason` / `usage` from the agent's existing
    `_last_turn_usage`-seam accessors (no new counters — design §4); both
    are optional and dropped when unavailable/non-clean.
    """
    return _book(
        session_log,
        "turn.finished",
        user_id=user_id,
        room_id=room_id,
        turn_id=turn_id,
        conclusion=conclusion,
        origin=origin,
        elapsed_s=elapsed_s,
        error_type=error_type,
        stop_reason=stop_reason,
        usage=usage,
    )
