"""file_ticket — the ONE filing channel for stations + workers (bead
workspace-e2uh.162, D14 follow-up; SB ruling 2026-09-02).

Filing follow-up tickets used to ride the one grammar-constrained terminal
call: a one-shot generation with no try/error/retry loop, where a lost
filing is silent capability loss — the exact fragile-chat-completion class
the terminal-tool mechanism was built to escape. This tool moves filing onto
the validated tool-call loop: the model calls it in-flight, a rejected call
tells it what to fix (steering text, sinks UNTOUCHED), and it retries
in-session. The loop IS the fix.

Accepted filings are appended to up to two sinks (exactly one is configured
per context in practice):

- per-run sink: ``callbacks["filed_proposals_sink"]`` — a per-run list the
  exec driver wires on the Agent (precedent: ``agent._terminal_submit``);
  exec emits it as the additive ``result["filed_proposals"]`` field.
- file transport: ``FILE_TICKET_TRANSPORT`` env > ``tool_config``
  ``transport_path`` — the in-cage worker path; JSONL append-only
  (create-or-append, O_APPEND single write).

Fail LOUD when neither sink is configured: a filing with nowhere to land
would be silently lost, and silent loss is the failure mode this tool
exists to kill (never accept-and-drop).

Per-run count cap (call-time): call N+1 when N >= cap errors with
"earlier filings stand" semantics — this REPLACES the whole-batch
count-cap rejection (amendment D) semantics at the tool boundary;
incremental calls must not nuke good siblings.

The filed object carries ONLY the six contract fields (title, description,
evidence, blocks_ticket, suspected_out_of_scope_paths, reason), with None
values stripped; ``blocks_ticket``/``suspected_out_of_scope_paths``/
``reason`` are WORKER ESCAPE fields (declare a ticket impossible-as-scoped;
daemon routes them to SPEC_FAILURE) — reviewers/critics leave them unset.

OA stays stigmergy-agnostic: no DB, no stigmergy imports, and the agent
process never touches the DB. Every path returns a ToolResult — the
handler NEVER raises.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openalph.tools import ToolResult

# Module-default filing cap — modest by design; the exec driver / weaver
# thread the charter cap via FILE_TICKET_MAX_FILINGS / config max_filings.
_MAX_FILINGS = 8

# The six contract fields, in filing order. The filed object carries ONLY
# these, with None values stripped (explicit False survives).
_FIELDS = (
    "title",
    "description",
    "evidence",
    "blocks_ticket",
    "suspected_out_of_scope_paths",
    "reason",
)


def _error_result(msg: str) -> ToolResult:
    """An is_error with a machine-readable JSON verdict: filed:false plus
    steering text (the field named / the cap / the environment defect +
    "re-call" guidance) so the model can machine-read WHY and fix the
    arguments — the loop is the fix. Sinks are untouched by construction
    (these results are always returned before any sink is touched)."""
    return ToolResult(
        content=json.dumps({"filed": False, "error": msg}, ensure_ascii=False),
        is_error=True,
    )


def _reject(msg: str) -> ToolResult:
    """A call-time rejection: is_error with steering text (the field named
    + "re-call" guidance) so the model can fix the arguments and retry."""
    return _error_result(msg)


def _transport_error_result(e: Exception) -> ToolResult:
    """Wrap a transport write failure as an is_error — never raise."""
    return _error_result(
        f"file_ticket failed to write the filing transport: {e} — the "
        f"filing was NOT recorded; fix the transport and re-call with the "
        f"same arguments to retry."
    )


def _fail_loud_result() -> ToolResult:
    """Fail loud when enabled but unconfigured: no sink, no transport.
    The content is JSON with filed:false so it cannot read as a success —
    silent loss is the failure mode this tool exists to kill."""
    return _error_result(
        "file_ticket enabled but no filing sink configured — neither the "
        "filed_proposals_sink callback nor a FILE_TICKET_TRANSPORT path "
        "(environment defect). Refusing to accept-and-drop: a silently "
        "lost filing is worse than a loud failure. If a filing is "
        "genuinely required, surface the defect and re-call once the "
        "sink is configured."
    )


def _cap_result(cap: int) -> ToolResult:
    """Per-run count cap (call-time): earlier filings STAND — never a
    whole-batch rejection (amendment D was one-shot-array semantics;
    incremental calls must not nuke siblings)."""
    return _error_result(
        f"filing cap ({cap}) reached — earlier filings stand. Your cap is "
        f"spent for this run: finish the work and submit your verdict. Do "
        f"not re-call file_ticket for this run."
    )


def _validating_rejection(field: Any, value: Any) -> ToolResult | None:
    """Validate one field. Returns a rejection ToolResult, or None when the
    value is acceptable. title/description are REQUIRED (None rejected);
    the four optional fields pass as None."""
    if field in ("title", "description"):
        if value is not None and isinstance(value, str) and value.strip():
            return None
        got = "missing" if value is None else f"got {type(value).__name__}"
        return _reject(
            f"{field} is required and must be a non-empty string ({got}); "
            f"re-call with corrected arguments."
        )
    if value is None:
        return None
    if field == "evidence":
        if isinstance(value, str):
            return None
        return _reject(
            f"evidence must be a string of file/line/commit pointers "
            f"(got {type(value).__name__}); re-call with corrected "
            f"arguments (omit evidence if you have no pointers)."
        )
    if field == "blocks_ticket":
        if isinstance(value, bool):
            return None
        return _reject(
            f"blocks_ticket must be a boolean, the worker-escape flag "
            f"declaring the ticket impossible-as-scoped (got "
            f"{type(value).__name__}); reviewers/critics leave it unset — "
            f"re-call with corrected arguments."
        )
    if field == "suspected_out_of_scope_paths":
        if isinstance(value, list) and all(
            isinstance(p, str) for p in value
        ):
            return None
        return _reject(
            f"suspected_out_of_scope_paths must be a list of strings "
            f"(worker-escape field naming the out-of-scope path(s) that "
            f"block completion; got {type(value).__name__}); "
            f"reviewers/critics leave it unset — re-call with corrected "
            f"arguments."
        )
    if field == "reason":
        if isinstance(value, str):
            return None
        return _reject(
            f"reason must be a string (worker-escape field stating why the "
            f"ticket cannot be completed within its target scope; got "
            f"{type(value).__name__}); reviewers/critics leave it unset — "
            f"re-call with corrected arguments."
        )
    return None


def _transport_count(transport_path: str) -> int:
    """Count of existing filings in the transport file (non-blank JSONL
    lines). Unreadable/absent file -> 0.

    INVARIANT (bead .162 audit): one transport file per episode — the
    worker driver points FILE_TICKET_TRANSPORT at the dispatch's own
    /work mount and station episodes carry no transport at all. The
    count-then-append sequence here is TOCTOU across writers; that is
    safe ONLY under the one-writer-per-file invariant. A future change
    that shares a transport between concurrent episodes trips review
    here first."""
    if not os.path.isfile(transport_path):
        return 0
    try:
        with open(transport_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def _write_transport(transport_path: str, filing: dict) -> None:
    """Append ONE json.dumps line (no embedded newlines) via
    os.open(O_WRONLY|O_CREAT|O_APPEND) + a single os.write + os.close —
    the JSONL append-only transport (create-or-append, no partial-line
    overwrite). Parent dirs created as needed. Raises on failure — the
    caller wraps it as is_error (never raise out of the handler)."""
    os.makedirs(os.path.dirname(transport_path) or ".", exist_ok=True)
    line = json.dumps(filing, ensure_ascii=False, separators=(",", ":")) + "\n"
    data = line.encode("utf-8")
    # 0600: the transport carries untrusted model-authored text; owner-only
    # (the cage worker's own uid — the daemon-side harvest reads through the
    # same mapped owner). World-readable was a needless exposure.
    fd = os.open(transport_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        # Durability: the transport is the WORKER's durable channel (the
        # daemon harvests the file, not the in-memory sink) — flush before
        # close so a host crash between append and harvest cannot silently
        # drop a filing the tool already acknowledged.
        os.fsync(fd)
    finally:
        os.close(fd)


async def run_file_ticket(
    *,
    title: str | None = None,
    description: str | None = None,
    evidence: str | None = None,
    blocks_ticket: bool | None = None,
    suspected_out_of_scope_paths: list[str] | None = None,
    reason: str | None = None,
    tool_config: dict | None = None,
    callbacks: dict | None = None,
) -> ToolResult:
    """File a proposed follow-up ticket for human triage (the ONE filing
    channel for stations + workers).

    Validates at call time: a bad shape returns an is_error with steering
    text naming the field + re-call guidance, with the sink and transport
    UNTOUCHED so the model retries in-session. An accepted call appends the
    clean filed object (the six contract fields minus None values) to the
    configured sink(s) and returns the ordinal.

    NEVER raises: garbage callbacks/transport fall back to fail-loud
    errors, not exceptions.
    """
    cfg = tool_config if isinstance(tool_config, dict) else {}
    cbs = callbacks if isinstance(callbacks, dict) else {}

    # (a) Sink resolution: the per-run list the exec driver wires on the
    # Agent; anything that is not a list is treated as absent (defensive —
    # garbage callbacks never crash the loop).
    sink = cbs.get("filed_proposals_sink")
    if not isinstance(sink, list):
        sink = None

    # (b) Transport + cap: env wins over config; config over module default.
    transport = os.environ.get("FILE_TICKET_TRANSPORT")
    if not transport:
        transport = cfg.get("transport_path") or None
    # Cap defects fail LOUD (bead .162 audit): a cap of 0/negative would
    # cap-reject EVERY call with benign-looking text — a silent channel
    # kill, the exact failure mode this tool exists to prevent.
    cap_raw = os.environ.get("FILE_TICKET_MAX_FILINGS")
    if cap_raw:
        cap_source = f"env FILE_TICKET_MAX_FILINGS={cap_raw!r}"
        try:
            cap = int(cap_raw)
        except ValueError:
            return _error_result(
                f"file_ticket filing-cap config defect: {cap_source} is not "
                f"an integer. Refusing to accept-and-drop: fix or remove "
                f"the env var, then re-call."
            )
    else:
        cap = cfg.get("max_filings", _MAX_FILINGS)
        cap_source = f"tool config max_filings={cap!r}"
        if isinstance(cap, bool) or not isinstance(cap, int):
            return _error_result(
                f"file_ticket filing-cap config defect: {cap_source} is not "
                f"an integer. Refusing to accept-and-drop: fix the tool "
                f"config, then re-call."
            )
    if cap < 1:
        return _error_result(
            f"file_ticket filing-cap config defect: {cap_source} is < 1 — "
            f"that cap-rejects EVERY call (silent channel kill). Refusing: "
            f"set the cap to >= 1, then re-call."
        )

    # (c) Fail LOUD when unconfigured — never accept-and-drop (silent loss
    # is the failure mode this tool exists to kill).
    if sink is None and not transport:
        return _fail_loud_result()

    # (d) Validate at call time, field by field; the FIRST defect gets the
    # steering text. Sinks are untouched on every rejection below.
    args = {
        "title": title,
        "description": description,
        "evidence": evidence,
        "blocks_ticket": blocks_ticket,
        "suspected_out_of_scope_paths": suspected_out_of_scope_paths,
        "reason": reason,
    }
    for field in _FIELDS:
        rejection = _validating_rejection(field, args[field])
        if rejection is not None:
            return rejection

    # (e) Per-run count cap, call-time: the count is the sink length when
    # the sink is configured (1:1 mirror with the transport append), else
    # the transport's existing line count. >= cap -> earlier filings stand.
    count = len(sink) if sink is not None else _transport_count(transport)
    if count >= cap:
        return _cap_result(cap)

    # (f) The clean filed object: the six contract fields minus None values
    # (explicit False survives — stripping is "minus None", not falsy).
    filing = {f: args[f] for f in _FIELDS if args[f] is not None}

    # (f2) Per-filing size cap (bead .162 audit): the harvest enforces a
    # per-item byte cap — a filing that passes here and dies there would be
    # a FALSE success (the steering loop never engages). Same envelope as
    # the count cap: env > config; None/absent = unchecked.
    max_bytes_raw = os.environ.get("FILE_TICKET_MAX_BYTES")
    if max_bytes_raw:
        try:
            max_bytes = int(max_bytes_raw)
        except ValueError:
            return _error_result(
                f"file_ticket size-cap config defect: "
                f"env FILE_TICKET_MAX_BYTES={max_bytes_raw!r} is not an "
                f"integer. Fix or remove the env var, then re-call."
            )
    else:
        max_bytes = cfg.get("max_bytes")
    if max_bytes is not None:
        size = len(
            json.dumps(filing, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        if size > max_bytes:
            return _reject(
                f"filing exceeds the {max_bytes}-byte size cap (got {size}) — "
                f"trim the description/evidence and re-call; nothing was "
                f"filed."
            )

    # (g) DURABLE CHANNEL FIRST (bead .162 audit): write the transport
    # before touching the sink, so a transport failure returns filed:false
    # with NOTHING recorded anywhere — the retry is idempotent-safe and the
    # sink can never hold an entry the durable transport lacks. (Sink-only
    # contexts — stations — skip the transport branch entirely.)
    if transport:
        try:
            _write_transport(transport, filing)
        except Exception as e:  # (h) never raise — wrap as is_error
            return _transport_error_result(e)
    if sink is not None:
        sink.append(filing)

    return ToolResult(
        content=json.dumps(
            {"filed": True, "ordinal": count + 1, "queued_for": "operator triage"}
        ),
        is_error=False,
    )
