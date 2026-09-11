"""kdsn.330 — dispatch ledger core for async (background) subagents.

Pure module (no Agent / Matrix imports): the per-dispatch record, its
state machine with terminal-state protection, the JSONL system-entry
payload builders (events ``subagent_dispatched`` / ``subagent_terminal``),
the error sanitizer, and the parser that rebuilds records from session-log
system entries (restart rehydration — wired by phase 2e).

Design rulings (specs/kdsn.330-p1-test-plan.md):
  R3  dispatch id = the tool-call id (``callbacks["call_id"]``)
  R4  record fields + states + terminal-state protection
  R5  ledger lifecycle entries are role="system" (never user/assistant —
      zero cache/byte-identity exposure), written BEFORE any notify
  R15 usage rides the subagent_terminal entry

States: ``running`` -> one of the terminal states
``completed / failed / cancelled / orphaned_at_restart / pending_delivery``.
Once terminal, later transitions are refused (no-op + log) — a late
in-flight event cannot regress a terminal record (sabotage pin 3).
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("openalph.subledger")

# --- states ----------------------------------------------------------------

RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
ORPHANED_AT_RESTART = "orphaned_at_restart"
PENDING_DELIVERY = "pending_delivery"

TERMINAL_STATES = frozenset({
    COMPLETED, FAILED, CANCELLED, ORPHANED_AT_RESTART, PENDING_DELIVERY,
})

# Notice/status head bound (spec §5: "task head (truncated for notices)").
TASK_HEAD_LIMIT = 80


def make_task_head(task: str) -> str:
    """Truncate a task to the notice head (≤ TASK_HEAD_LIMIT chars)."""
    return (task or "")[:TASK_HEAD_LIMIT]


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class DispatchRecord:
    """One background subagent dispatch, owned per-room on Agent (D4).

    ``usage`` is a flat dict with the kdsn.218 bridge fields plus the
    sub's token counters:
    ``{"input_tokens", "output_tokens", "cost_usd", "unpriced_tokens"}``.
    """

    dispatch_id: str
    task: str
    task_head: str
    model: str | None
    effort: str | None
    dispatched_at: str
    state: str = RUNNING
    terminal_at: str | None = None
    result: str | None = None
    error: str | None = None
    usage: dict = field(default_factory=dict)

    def transition(self, new_state: str) -> bool:
        """Apply a state transition, protected by terminal-state rules.

        - Transitions are only legal FROM ``running``.
        - Once in a terminal state, later transitions are refused: no-op
          + log (a late in-flight event cannot regress or restate a
          terminal record — D5/R4).
        - Returns True when the state actually changed.
        """
        if new_state == self.state:
            return False
        if self.state in TERMINAL_STATES:
            logger.warning(
                "subledger: refusing transition %s -> %s for %s "
                "(terminal-state protection)",
                self.state, new_state, self.dispatch_id)
            return False
        if self.state != RUNNING:
            logger.warning(
                "subledger: refusing transition %s -> %s for %s "
                "(transitions are only legal from running)",
                self.state, new_state, self.dispatch_id)
            return False
        self.state = new_state
        if new_state in TERMINAL_STATES:
            self.terminal_at = _now_iso()
        return True


# --- JSONL system-entry payloads (R5) ---------------------------------------

def dispatched_detail(rec: DispatchRecord) -> dict:
    """Detail dict for the ``subagent_dispatched`` system entry.

    Carries the record fields (full task text included — system entries are
    never rendered, pure audit + rehydration substrate).
    """
    return {
        "dispatch_id": rec.dispatch_id,
        "state": rec.state,
        "task": rec.task,
        "task_head": rec.task_head,
        "model": rec.model,
        "effort": rec.effort,
        "dispatched_at": rec.dispatched_at,
    }


def terminal_detail(rec: DispatchRecord) -> dict:
    """Detail dict for the ``subagent_terminal`` system entry.

    Usage fields ride flat (R15): ``input_tokens`` / ``output_tokens`` /
    ``cost_usd`` / ``unpriced_tokens`` — the ``usage_totals`` re-sum loop
    reads them straight off the entry. ``error`` present iff failed.
    """
    usage = rec.usage or {}
    return {
        "dispatch_id": rec.dispatch_id,
        "state": rec.state,
        "task": rec.task,
        "task_head": rec.task_head,
        "model": rec.model,
        "effort": rec.effort,
        "dispatched_at": rec.dispatched_at,
        "terminal_at": rec.terminal_at,
        "result": rec.result,
        "error": rec.error,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cost_usd": usage.get("cost_usd", 0.0),
        "unpriced_tokens": usage.get("unpriced_tokens", 0),
    }


def terminal_event_dict(rec: DispatchRecord) -> dict:
    """The terminal event deposited into the per-room pending-events inbox
    (harness-owned; drained at the next tool-loop boundary or fired by the
    idle-room completion turn — phase 2c). Small pointer + the report
    bytes the delivery surface renders (never anything else from the sub).
    """
    event = {
        "dispatch_id": rec.dispatch_id,
        "state": rec.state,
        "task_head": rec.task_head,
        "terminal_at": rec.terminal_at,
    }
    if rec.result is not None:
        event["result"] = rec.result
    if rec.error is not None:
        event["error"] = rec.error
    return event


# --- delivery framing (R6): store-raw / frame-at-build ----------------------
#
# The drained batch / synthetic-turn content is stored as
# role="user", source="subagent_event" with the RAW (unframed) event
# lines; the frame is CONTEXT-ONLY (steer precedent) and applied with
# ONE shared expression — frame_subagent_event_content — at live-append
# (agent.py tool-loop-top drain) and at build_context rebuild
# (session.py), so live and rebuilt bytes match (A05/A06). The frame is
# a PURE FUNCTION of the stored content: no ledger or mutable
# interpolation (pinned: test_frame_pure_function_of_stored_content).

SUBAGENT_EVENT_FRAME = ("[Automated sub-agent events — harness notice, "
                        "not operator input]")


def _one_line(text) -> str:
    """Collapse a field to a single line (notice lines never span)."""
    return " ".join(str(text).split())


def _escape(text: str) -> str:
    """Store-time escape for sub-produced bytes.

    Sub-produced text (task head, error, result) passes the reminder-tag
    escaper BEFORE the line is stored, so a forged harness tag in sub
    output lands in the delivered bytes as inert data. Idempotent: the
    sanitized error line (already escaped at deposit time) is untouched.
    """
    from openalph.tools import escape_system_reminder_tags
    return escape_system_reminder_tags(text)


def format_event_line(event: dict) -> str:
    """ONE harness-authored line per terminal event (R6/D3).

    Content-free rule (D1 notify-then-pull): dispatch id + state + task
    head, plus the sanitized error line for failed dispatches. The
    RESULT is never rendered here — it is retrieve-only (the parent
    pulls it via subagent_status; the delivery message is a pointer,
    not the payload). Error text is escaped at line build (store) time,
    so the stored bytes ARE the delivered bytes.
    """
    line = f"- {event.get('dispatch_id', '?')}: {event.get('state', '?')}"
    task_head = _one_line(event.get("task_head") or "")
    if task_head:
        line += f" — task: {_escape(task_head)}"
    error = event.get("error")
    if error is not None:
        line += f" — error: {_escape(_one_line(error))}"
    return line


def subagent_event_lines(events: list[dict]) -> str:
    """The RAW (unframed) stored content for a terminal-event batch:
    one line per event, newline-joined. This is exactly what the
    source="subagent_event" user entry stores."""
    return "\n".join(format_event_line(e) for e in events)


def frame_subagent_event_content(raw_lines: str) -> str:
    """Frame stored raw event lines for model context.

    THE single framing expression: the agent-side live drain-append and
    session.py's build_context rebuild both call this, so the frame is
    a pure function of the stored bytes (A06) and cannot drift between
    the two paths.
    """
    return SUBAGENT_EVENT_FRAME + "\n" + raw_lines


def terminal_notice_line(events: list[dict]) -> str:
    """Collapsed ONE-line in-room notice for a terminal burst (D1/R6).

    Carries the exact event line bytes the model sees — never a
    restated summary (inserts-visible lint pin)."""
    return SUBAGENT_EVENT_FRAME + " — " + " | ".join(
        format_event_line(e) for e in events)


# --- error sanitization (D5: sanitized error, never raw exception repr) -----

# system-reminder tags in BOTH the entity-escaped form (&lt;…&gt; — what a
# sub's output looks like after one escape pass) and the raw angle-bracket
# form. A stored ledger error must never carry either: the entity form is
# what the store-time escape produces, and keeping it would leave a
# re-escapable marker in the durable audit trail.
_REMINDER_ENTITY_RE = re.compile(
    r"&lt;\s*(/?)\s*system-reminder\b[^&]*&gt;", re.IGNORECASE)
_REMINDER_RAW_RE = re.compile(
    r"<\s*(/?)\s*system-reminder\b[^>]*>", re.IGNORECASE)

# Secret-shaped tokens for the ledger-error sanitizer only (NOT added to the
# shared CREDENTIAL_PATTERNS — this pass is scoped to stored sub errors):
# short-prefix credential shapes the shared pattern list deliberately
# leaves alone (its sk- pattern requires ≥20 tail chars).
_SECRET_SHAPED_RE = re.compile(
    r"\b(?:sk-|ghp_|gho_|ghs_|ghr_|github_pat_|xox[baprs]-|ops_|AIza|AKIA)"
    r"[A-Za-z0-9_\-]{4,}")


def sanitize_dispatch_error(exc_or_text, known_secrets=None) -> str:
    """Sanitize a sub exception (or an error ToolResult's content) for the
    durable ledger record.

    Guarantees:
      - never a raw exception repr — exception TYPE NAME + message text only;
      - ``<system-reminder>`` tags stripped in BOTH entity-escaped and raw
        form, then escaped again as a safety net (escape is idempotent);
      - credential-shaped text redacted: shared pattern pass + the caller's
        known live secret values + a scoped secret-shaped pass.
    Fail-soft: on ANY internal error the function returns the minimal
    ``"<type>: (error text unavailable)"`` form — sanitization must never
    turn a failed sub into a crashed finalize.
    """
    try:
        if isinstance(exc_or_text, BaseException):
            text = str(exc_or_text)
            type_name = type(exc_or_text).__name__
        else:
            text = str(exc_or_text or "")
            type_name = None

        text = _REMINDER_ENTITY_RE.sub("", text)
        text = _REMINDER_RAW_RE.sub("", text)

        from openalph.tools import escape_system_reminder_tags
        text = escape_system_reminder_tags(text)

        from openalph.tools.security import redact_credentials, redact_known_secrets
        text, _ = redact_credentials(text)
        if known_secrets:
            text, _ = redact_known_secrets(text, set(known_secrets))
        text = _SECRET_SHAPED_RE.sub("[REDACTED:api_key]", text)

        text = " ".join(text.split())  # collapse whitespace/stray newlines
        if not text:
            text = "(empty error message)"
        if type_name is None:
            return text
        return f"{type_name}: {text}"
    except Exception:
        _type = type(exc_or_text).__name__ if isinstance(
            exc_or_text, BaseException) else "error"
        return f"{_type}: (error text unavailable — sanitizer failure)"


# --- parser: rebuild records from session-log system entries (R5/R14) -------

def _parse_detail(detail) -> dict | None:
    """Tolerate both stored detail shapes: a dict (SessionLog.append with a
    dict detail) or a JSON string (writers that pre-serialize)."""
    if isinstance(detail, dict):
        return detail
    if isinstance(detail, str):
        import json
        try:
            parsed = json.loads(detail)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def parse_ledger_entries(entries: list[dict]) -> dict[str, DispatchRecord]:
    """Rebuild in-memory DispatchRecords from session-log entries (phase 2e
    calls this from ``_activate_room``).

    Scans ``role="system"`` entries for ``subagent_dispatched`` (creates the
    record) and ``subagent_terminal`` (applies the terminal state, result,
    error and usage). Returns ``dispatch_id -> DispatchRecord``.

    Records come back exactly as the JSONL chain records them: a dispatch
    with no terminal entry stays ``running`` — the CALLER resolves it to
    ``orphaned_at_restart`` (D5: kill-on-restart, never re-arm) and decides
    ``pending_delivery`` for terminal-but-undelivered records (the delivery
    check is the caller's — it needs the room's drain entries, not just the
    ledger chain). Missing optional fields (model/effort/dispatched_at on
    hand-written entries) degrade to None/empty, never raise.
    """
    records: dict[str, DispatchRecord] = {}
    for entry in entries:
        if entry.get("role") != "system":
            continue
        event = entry.get("event")
        detail = _parse_detail(entry.get("detail"))
        if detail is None:
            continue
        dispatch_id = detail.get("dispatch_id")
        if not dispatch_id:
            continue
        dispatch_id = str(dispatch_id)

        if event == "subagent_dispatched":
            if dispatch_id in records:
                continue  # first dispatched entry wins (duplicates are damage)
            task = detail.get("task") or ""
            records[dispatch_id] = DispatchRecord(
                dispatch_id=dispatch_id,
                task=task,
                task_head=detail.get("task_head") or make_task_head(task),
                model=detail.get("model"),
                effort=detail.get("effort"),
                dispatched_at=detail.get("dispatched_at") or entry.get("ts") or "",
                state=detail.get("state") or RUNNING,
                usage={},
            )
        elif event == "subagent_terminal":
            rec = records.get(dispatch_id)
            if rec is None:
                # Terminal without a dispatched entry (partial log) —
                # synthesize a minimal record so the terminal state is not
                # lost; the causality gap is the audit's problem.
                rec = DispatchRecord(
                    dispatch_id=dispatch_id,
                    task=detail.get("task") or "",
                    task_head=detail.get("task_head") or "",
                    model=detail.get("model"),
                    effort=detail.get("effort"),
                    dispatched_at=entry.get("ts") or "",
                    state=RUNNING,
                    usage={},
                )
                records[dispatch_id] = rec
            state = detail.get("state")
            if state in (COMPLETED, FAILED, CANCELLED, ORPHANED_AT_RESTART,
                         PENDING_DELIVERY):
                # kdsn.330 re-audit F1: terminal-state protection at rebuild —
                # a later chained terminal entry (e.g. a stray 'running' or a
                # failed-after-completed) must not regress an already-terminal
                # record, mirroring in-memory transition(). First terminal
                # wins; later ones are logged and skipped.
                if rec.state in TERMINAL_STATES and state != rec.state:
                    logger.warning(
                        "subledger rebuild: refusing chained terminal %s -> %s "
                        "for %s (terminal-state protection)",
                        rec.state, state, dispatch_id)
                else:
                    rec.state = state
                    rec.terminal_at = (detail.get("terminal_at")
                                       or entry.get("ts") or rec.terminal_at)
                    rec.result = detail.get("result")
                    rec.error = detail.get("error")
            rec.usage = {
                "input_tokens": detail.get("input_tokens", 0) or 0,
                "output_tokens": detail.get("output_tokens", 0) or 0,
                "cost_usd": detail.get("cost_usd", 0.0) or 0.0,
                "unpriced_tokens": detail.get("unpriced_tokens", 0) or 0,
            }
    return records
