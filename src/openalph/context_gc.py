"""Context GC — unified context-boundary mechanism (workspace-kdsn.305).

Mental model (SB-approved, 2026-08-30): context is append-only EXCEPT at
explicit GC boundaries. At a boundary, everything pre-boundary is reduced by
one uniform rule:

  - unbounded class (tool outputs / media) -> pointer-bearing placeholders
  - reasoning/thinking -> dropped entirely (thought-only turns deleted
    atomically — never leave empty shells)
  - durable set -> re-attached as a frozen snapshot taken at boundary time
  - post-boundary content untouched

JSONL is NEVER modified by a boundary: application only APPENDS (a manifest
event + a snapshot entry); all reduction happens render-time in
``session.SessionLog.build_context``. Between boundaries the render must have
the append-only byte-prefix property (cache invariant I1 / guardrails A05+A06).

THIS MODULE IS PURE: stdlib + tomllib only. It must NEVER import openalph.*
(session.py, agent.py, tools/ import THIS module, never the reverse). The ONE
deliberate exception is the R2-A escaping helper, which is imported LAZILY
inside the functions that need it (``tool_pointer`` / ``frame_snapshot``) so
no import cycle is created and module import stays side-effect-free.
All formatting is deterministic — no timestamps in any render-visible string
(timestamps live only inside manifest dicts, which are system events and are
never rendered into model context).

Entry-shape conventions (JSONL dicts, pre-ToolCall-construction):
  user entry:      {role, sender, room, event_id, ts, content, source?}
  assistant entry: {role, content?, tool_calls?: [{call_id|id, name, input,
                    extra_content?}], thinking?: str|list, usage?}
  tool entry:      {role, call_id, name, output, truncated?, is_error?}
  system entry:    {role, event, detail?, entry_index?}
"""

import json
import logging
import re
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: system-event name appended at each boundary application (the manifest).
GC_EVENT = "gc_boundary"
#: user-entry ``source`` for the frozen durable-set snapshot block.
GC_SNAPSHOT_SOURCE = "gc_snapshot"

#: Media tag pattern: [media: <path> (<mime>, <size>)] — the JSONL-persisted
#: form of a media attachment (the in-memory expanded form is a content-block
#: list). Canonical definition lives HERE; agent.py and session.py mirror or
#: import it (agent.py cannot import context_gc without a cycle the other
#: way, so its copy is sync-checked by tests).
MEDIA_TAG_RE = re.compile(r"\[media:\s*(.+?)\s+\(([^,]+),\s*([^)]+)\)\]")
#: legacy marker event (pre-GC /cache toolstrip). Honored as a boundary under
#: uniform rules when gc_enabled; recognized by ``current_boundary_index``.
LEGACY_EVENT = "toolstrip"
#: system-event name for an operator/agent project declaration.
ACTIVE_PROJECT_EVENT = "active_project"

#: reminder trigger ids (fired-state lives in the per-room ReminderEngine).
TRIGGER_GC_WARN = "gc-warn"

__all__ = [
    "GCConfigError",
    "GC_EVENT",
    "GC_SNAPSHOT_SOURCE",
    "LEGACY_EVENT",
    "ACTIVE_PROJECT_EVENT",
    "TRIGGER_GC_WARN",
    "current_boundary_index",
    "_marker_indexes",
    "read_active_project",
    "tool_pointer",
    "legacy_tool_placeholder",
    "strip_thinking_entry",
    "thinking_tail_indices",
    "gc_thinking_tail_kwargs",
    "parse_durable_set",
    "resolve_durable_set",
    "durable_budget_tokens",
    "frame_snapshot",
    "apply_boundary",
]


class GCConfigError(Exception):
    """Raised by parse_durable_set on malformed durable-set TOML."""


#: reminder trigger for the forced-handoff directive at post-boundary
#: RUNWAY exhaustion (workspace-kdsn.305.3, reworked kdsn.305.12 D1). One
#: per epoch; the bead is the cross-session signal. The durable budget is
#: informational only and never raises this.
GC_FORCED_HANDOFF_TRIGGER = "gc-forced-handoff"

#: fleet bd binary for the handoff pointer bead (fail-soft; None disables).
BD_PATH = "/srv/openalph/shared/bin/bd"


# ---------------------------------------------------------------------------
# Boundary discovery (render + application both need this)
# ---------------------------------------------------------------------------

def _marker_indexes(entries: list[dict]) -> list[int]:
    """All boundary-marker entry_index values (both kinds), ascending order.

    Same filter as current_boundary_index; shared by the render's
    creating-boundary attribution (wave-2.1 A1).
    """
    out = []
    for entry in entries:
        if entry.get("role") == "system" and entry.get("event") in (
            GC_EVENT,
            LEGACY_EVENT,
        ):
            try:
                idx = int(entry.get("entry_index", -1))
            except (TypeError, ValueError):
                idx = -1
            if idx >= 0:
                out.append(idx)
    return sorted(out)


def current_boundary_index(entries: list[dict]) -> int:
    """Max entry_index over all boundary markers (both kinds); -1 if none.

    Monotonicity rule: a new boundary index must strictly EXCEED this value
    (apply_boundary refuses computed index <= current, so re-applying at the
    same position is a no-op).
    """
    markers = _marker_indexes(entries)
    return markers[-1] if markers else -1


def read_active_project(entries: list[dict]) -> str | None:
    """Active project name from the latest ACTIVE_PROJECT_EVENT, else None.

    Per-epoch: the JSONL is wiped at umbral, so declaration is per-epoch by
    construction (declare-once re-arms after rotation).
    """
    project = None
    for entry in entries:
        if entry.get("role") == "system" and entry.get("event") == ACTIVE_PROJECT_EVENT:
            detail = entry.get("detail")
            if isinstance(detail, str) and detail:
                project = detail
    return project


# ---------------------------------------------------------------------------
# Placeholder formatting (render-side, deterministic)
# ---------------------------------------------------------------------------

#: tools whose first identifying parameter is included in the pointer
#: (param name -> whether the value is a re-loadable target).
POINTER_PARAMS = {
    "file_read": "path",
    "file_write": "path",
    "file_edit": "path",
    "file_patch": "path",
    "glob": "pattern",
    "grep": "pattern",
    "web_fetch": "url",
    "web_fetch_js": "url",
    "shell": "command",
    "memory_search": "query",
}

_PARAM_TRUNC = 80


def _escape_reminder_tags(text: str) -> str:
    """R2-A escape of ``<system-reminder>`` markup (lazy import, idempotent).

    The one sanctioned crossing of the pure-module boundary: openalph.tools
    owns the canonical escaping regex, and the escape is byte-pinned by
    contract, so we reuse it instead of forking it. Imported here (never at
    module top level) so this module stays importable without openalph.*.
    """
    from openalph.tools import escape_system_reminder_tags

    return escape_system_reminder_tags(text)


def _frame_field(value: str) -> str:
    """Snapshot FRAME metadata (path/reason/project/error text) hardened for
    embedding: whitespace flattened (no forged new framing lines), reminder
    markup escaped. Frame fields are agent-authored TOML metadata — they get
    the same treatment as file bytes (audit: forged ``--- BEGIN`` /
    reminder-markup injection via the reason field)."""
    flat = re.sub(r"\s+", " ", str(value)).strip()
    return _escape_reminder_tags(flat)


def _freeze_file_text(text: str) -> str:
    """File bytes at freeze time: credential redaction first (the snapshot
    bypasses the tool-output redaction pipeline, so the same pass runs here —
    audit: secrets persisted verbatim into the JSONL audit record), then the
    R2-A reminder escape."""
    try:
        from openalph.tools.security import redact_credentials
        text, _events = redact_credentials(text)
    except Exception as e:  # redaction is defense-in-depth; never block a
        logger.warning("gc snapshot redaction pass failed: %s", e)  # boundary
    return _escape_reminder_tags(text)


def _pointer_ident(name: str, params: dict | None) -> str:
    """Identifying fragment for a pointer: known param > first string param.

    Returns "" when params is absent or carries no usable string value.
    """
    if not params:
        return ""
    key = POINTER_PARAMS.get(name)
    if key is not None and key in params:
        value = params[key]
        if isinstance(value, str) and value:
            return value
    for value in params.values():
        if isinstance(value, str) and value:
            return value
    return ""


def tool_pointer(boundary_index: int, name: str, params: dict | None, n_chars: int) -> str:
    """Pointer-bearing placeholder for an expunged tool result.

    Format: ``[expunged at GC boundary N: <name> <identifying-param>
    (<n_chars> chars) — re-run the tool if the result is needed]``.
    The identifying param is POINTER_PARAMS[name] when present (value
    truncated to _PARAM_TRUNC chars, newlines flattened); otherwise the first
    string param value; otherwise bare ``<name> result``. Deterministic.
    """
    ident = _pointer_ident(name, params)
    if ident:
        ident = re.sub(r"\s+", " ", ident).strip()
        if len(ident) > _PARAM_TRUNC:
            ident = ident[:_PARAM_TRUNC]
        # Model-origin param values may carry reminder markup (R2-A): the
        # pointer is rendered context, so escape it the way every other
        # user-visible surface does.
        ident = _escape_reminder_tags(ident)
    else:
        ident = "result"
    return (
        f"[expunged at GC boundary {boundary_index}: {name} {ident} "
        f"({n_chars} chars) — re-run the tool if the result is needed]"
    )


def legacy_tool_placeholder(name: str, n_chars: int) -> str:
    """Byte-identical legacy strip placeholder: ``[stripped: {name} result, {n} chars]``."""
    return f"[stripped: {name} result, {n_chars} chars]"


# ---------------------------------------------------------------------------
# Thinking strip (pure entry transform)
# ---------------------------------------------------------------------------

def strip_thinking_entry(entry: dict) -> dict | None:
    """Return the entry with ``thinking`` removed, or None if thought-only.

    None (atomic delete) iff the entry is an assistant entry with no content
    (or whitespace-only) and no tool_calls after the drop. Non-assistant
    entries pass through unchanged. The input dict is not mutated.
    """
    if entry.get("role") != "assistant" or "thinking" not in entry:
        return entry
    out = {k: v for k, v in entry.items() if k != "thinking"}
    content = out.get("content")
    content_ok = isinstance(content, str) and bool(content.strip())
    tool_calls = out.get("tool_calls")
    if not content_ok and not tool_calls:
        return None
    return out


# ---------------------------------------------------------------------------
# Thinking-tail preservation (workspace-kdsn.305.13)
# ---------------------------------------------------------------------------

def _thinking_block_chars(thinking) -> int:
    """Deterministic char size of one thinking block (single source).

    str thinking: len(str).  Structured (list of blocks): len(json.dumps(
    thinking, sort_keys=True)) — sorted keys, so the estimate is stable
    across renders and across the selection/estimator boundary.  Anything
    else: 0.  BOTH consumers MUST go through this helper:
      - ``thinking_tail_indices`` counts block est = _chars // 4 tokens
        against the ceiling (whole-block, contiguous);
      - ``_render_char_estimates`` / the message-list transform add the raw
        _chars to the retained-thinking char total (divided ONCE, with the
        rest of the span — a shared division carry may shift the composite
        by one token vs the selection's per-block division).
    A divergence between selection-fit math and manifest math is a defect.
    """
    if isinstance(thinking, str):
        return len(thinking)
    if thinking:
        return len(json.dumps(thinking, sort_keys=True))
    return 0


def _tail_eligible(entry: dict) -> bool:
    """Tail-eligibility (T1/T1a): pre-boundary assistant entries whose
    thinking is retained — role assistant, non-empty thinking, AND
    non-empty content (str, non-whitespace) or tool_calls.  Thought-only
    entries (thinking with neither) are NEVER eligible: a thinking-only
    assistant message is an untested render shape on the Anthropic path,
    and thought-only turns are always stripped/deleted exactly as legacy.
    Shape-agnostic: works on both JSONL entries and plain message dicts
    (the subagent path's "id"-keyed tool_calls included — only truthiness
    is consulted, never the key names)."""
    if entry.get("role") != "assistant" or not entry.get("thinking"):
        return False
    content = entry.get("content")
    content_ok = isinstance(content, str) and bool(content.strip())
    return content_ok or bool(entry.get("tool_calls"))


def thinking_tail_indices(entries: list, boundary_index: int, n: int,
                          ceiling: int) -> frozenset[int]:
    """Positions of the retained thinking tail (workspace-kdsn.305.13 T1).

    Walk the pre-boundary span (position < boundary_index) NEWEST to
    OLDEST.  An entry is TAIL-ELIGIBLE iff it is an assistant entry with
    non-empty thinking AND non-empty content or tool_calls (thought-only
    entries are never eligible and never consume slots — T1a).  Retain an
    eligible entry's thinking while (a) retained count < n AND (b)
    cumulative retained estimate + this block's estimate <= ceiling —
    CONTIGUOUS-STOP on the first overflow: never skip to an older smaller
    block (an older-without-newer tail is incoherent).  Whole blocks only —
    never a partial thinking field.

    Block estimate: ``_thinking_block_chars(thinking) // 4`` tokens
    (deterministic — see the shared helper).  n <= 0, ceiling <= 0, or
    boundary_index < 0 → empty set (full strip).  Returns a frozenset of
    list POSITIONS (the same positions enumerate() assigns over the
    pre-boundary span — session.py render and the message-list transform
    both key on enumerate positions, so the set works for both shapes).
    """
    if n <= 0 or ceiling <= 0 or boundary_index < 0:
        return frozenset()
    retained: set[int] = set()
    count = 0
    cumulative = 0
    for pos in range(min(boundary_index, len(entries)) - 1, -1, -1):
        entry = entries[pos]
        if not isinstance(entry, dict) or not _tail_eligible(entry):
            continue
        block_est = _thinking_block_chars(entry["thinking"]) // 4
        if count < n and cumulative + block_est <= ceiling:
            retained.add(pos)
            count += 1
            cumulative += block_est
        else:
            break  # contiguous stop on first overflow
    return frozenset(retained)


def gc_thinking_tail_kwargs(config) -> dict:
    """build_context tail kwargs from an agent config (T4 default-to-legacy).

    The single source for the PRODUCTION render call sites (matrix/cli/
    callbacks): returns a dict ready to splat into
    ``build_context(room_id, **kw)`` so a real config's tail knobs reach the
    render.  Both keys are ALWAYS present (even when 0/0 — the legacy
    full-strip defaults, byte-identical to pre-.13).

    Fail-closed to FULL-STRIP (0/0) on anything that is not a clean
    non-negative int — this is the "config absent / not yet loaded / mock"
    case. Two distinct absences both land at 0/0 (legacy render):
      - ``config.context`` missing (pre-.13 config, or a MagicMock agent
        whose auto-attribute is not a real ContextHandoffConfig) → 0/0;
      - either field missing or non-int (a partial mock / a bool / a
        negative) → that field 0.
    A real ContextHandoffConfig carries real ints, so this passes its values
    through unchanged. The isinstance gate (not just getattr-with-default)
    is load-bearing: ``getattr(MagicMock, "thinking_tail_turns", 0)``
    returns a MagicMock, which would raise on ``> 0`` and, worse, be truthy
    enough to flip the tail ON for mock agents — the guard forces 0/0.
    """
    ctx = getattr(config, "context", None)
    turns = getattr(ctx, "thinking_tail_turns", 0)
    max_tokens = getattr(ctx, "thinking_tail_max_tokens", 0)
    if not (isinstance(turns, int) and not isinstance(turns, bool)
            and turns >= 0):
        turns = 0
    if not (isinstance(max_tokens, int) and not isinstance(max_tokens, bool)
            and max_tokens >= 0):
        max_tokens = 0
    return {
        "thinking_tail_turns": turns,
        "thinking_tail_max_tokens": max_tokens,
    }


# ---------------------------------------------------------------------------
# Durable set
# ---------------------------------------------------------------------------

def parse_durable_set(text: str) -> list[dict]:
    """Parse durable-set.toml -> [{"path": str, "reason": str}, ...].

    Raises GCConfigError on malformed TOML or non-conforming entries
    (missing/non-str path, non-str reason). Empty file -> []. Unknown keys
    are ignored (forward compatibility).
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise GCConfigError(f"durable-set.toml: malformed TOML: {e}") from e
    entries = data.get("entries")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise GCConfigError("durable-set.toml: 'entries' must be an array of tables")
    out: list[dict] = []
    for i, item in enumerate(entries):
        if not isinstance(item, dict):
            raise GCConfigError(f"durable-set.toml: entries[{i}] must be a table")
        path = item.get("path")
        reason = item.get("reason")
        if not isinstance(path, str) or not path:
            raise GCConfigError(f"durable-set.toml: entries[{i}].path must be a non-empty string")
        if not isinstance(reason, str):
            raise GCConfigError(f"durable-set.toml: entries[{i}].reason must be a string")
        out.append({"path": path, "reason": reason})
    return out


def durable_budget_tokens(window: int, budget_pct: float, budget_min: int) -> int:
    """Durable budget in tokens: max(window * budget_pct, budget_min)."""
    return max(int(window * budget_pct), budget_min)


def _rel_path(workspace: Path, p: Path) -> str:
    """Workspace-relative POSIX path when possible, else absolute."""
    try:
        return p.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(p)


def _read_durable_text(workspace: Path, p: Path, errors: list[str]) -> tuple[bool, str]:
    """Read one durable file: (exists, text). Problems -> errors, never raise."""
    if not p.exists():
        return False, ""
    try:
        return True, p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        errors.append(f"unreadable durable file: {_rel_path(workspace, p)}: {e}")
        return False, ""


def resolve_durable_set(
    workspace: Path,
    project: str | None,
    config_paths: list[str] | None = None,
) -> dict:
    """Resolve the full durable set for a boundary application.

    Union of (in order, deduped by resolved absolute path, first origin wins):
      1. auto-injected: <project>/progress.md and <project>/durable-set.toml
         (only when project is not None) — origin "auto"
      2. durable-set.toml [[entries]] (parsed via parse_durable_set) —
         origin "durable-set.toml"
      3. config-declared path classes (glob patterns, workspace-relative) —
         origin "config"

    Returns dict:
      {
        "project": str | None,
        "files": [ {path, reason, text, exists, origin}, ... ],   # text="" when missing
        "errors": [str, ...],      # malformed TOML, unreadable files — never raise
        "used_tokens": int,        # char//4 over RAW (pre-escape) text of existing files
        "missing": [path, ...],
      }
    """
    workspace = Path(workspace)
    errors: list[str] = []
    seen: set[str] = set()
    files: list[dict] = []
    missing: list[str] = []

    def add(p: Path, reason: str, origin: str) -> None:
        try:
            key = p.resolve()
        except (OSError, RuntimeError):
            key = Path(str(p).strip() or "/")
        # WORKSPACE CONTAINMENT (audit: arbitrary-file-read via durable-set
        # entries — `workspace / "../../etc/passwd"` or an absolute/symlinked
        # path resolved straight out of the sandbox). Agent-authored TOML
        # entries must never escape the workspace; violations are recorded,
        # never read.
        try:
            ws_root = workspace.resolve()
        except (OSError, RuntimeError):
            ws_root = workspace
        if not key.is_relative_to(ws_root):
            errors.append(f"durable path escapes workspace: {p}")
            return
        if key in seen:
            return
        seen.add(key)
        rel = _rel_path(workspace, p)
        exists, text = _read_durable_text(workspace, key, errors)
        if not exists:
            missing.append(rel)
        files.append(
            {"path": rel, "reason": reason, "text": text, "exists": exists, "origin": origin}
        )

    # 1. auto-injected project files (progress.md, then durable-set.toml).
    project_durable: list[dict] = []
    if project is not None:
        proj_dir = workspace / "memory" / "projects" / project
        add(proj_dir / "progress.md", "project working state", "auto")
        toml_path = proj_dir / "durable-set.toml"
        toml_exists, toml_text = _read_durable_text(workspace, toml_path, errors)
        add(toml_path, "project durable-set declaration", "auto")
        if toml_exists:
            try:
                project_durable = parse_durable_set(toml_text)
            except GCConfigError as e:
                errors.append(str(e))
        else:
            errors.append(f"durable-set.toml not found: {project}")

    # 2. durable-set.toml [[entries]].
    for item in project_durable:
        add(workspace / item["path"], item["reason"], "durable-set.toml")

    # 3. config-declared path classes (workspace-relative globs). A literal
    #    path (no glob metachars) that matches nothing is listed as missing —
    #    named, not silently skipped.
    for pattern in config_paths or []:
        try:
            matches = sorted(workspace.glob(pattern))
        except (OSError, RuntimeError, ValueError) as e:
            errors.append(f"config glob failed: {pattern}: {e}")
            continue
        for match in matches:
            if match.is_file():
                add(match, "config-declared", "config")
        if not matches and not re.search(r"[*?\[]", pattern):
            add(workspace / pattern, "config-declared", "config")

    used_tokens = sum(len(f["text"]) for f in files if f["exists"]) // 4
    return {
        "project": project,
        "files": files,
        "errors": errors,
        "used_tokens": used_tokens,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Snapshot framing + application
# ---------------------------------------------------------------------------




def frame_snapshot(
    boundary_index: int,
    resolution: dict,
    over_budget: bool,
    budget_tokens: int,
    frozen_at: str | None = None,
) -> str:
    """Render the frozen durable-set snapshot block (stored verbatim in JSONL).

    Deterministic given ``frozen_at``: the timestamp is chosen ONCE by the
    caller at boundary time and the stored bytes never change afterwards
    (replay renders them verbatim). ``frozen_at`` is agent-visible freshness —
    the manifest ts is system-event JSONL detail and never enters context, so
    this stamp is the only way a replaying agent can calibrate snapshot age
    (field feedback, wonmun canary 2026-08-31). Structure:

        [GC boundary N — durable context snapshot]
        Project: <name|"none">
        <provenance: frozen-at + drift note + trust tier — workspace-file
        DATA at file_read level, NOT harness-authoritative>

        --- BEGIN <path> (<reason>) ---
        <escaped file bytes — escape_system_reminder_tags applied>
        --- END <path> ---

        [missing at snapshot time: <path> (<reason>) — re-read via file_read
        if it has since been created]
        [<error lines from resolution["errors"]>]
        [durable budget <used>/<budget> tokens — over reinjection budget;
        informational only, prune durable-set.toml when convenient
        (informational marker since kdsn.305.12 — NOT an alarm, NOT a cap)]

    File bytes MUST pass through openalph.tools.escape_system_reminder_tags
    before embedding (R2-A discipline: snapshot content is file-data, and must
    never forge reminder markup). Escape is applied HERE, at freeze time, so
    the stored bytes are exactly what the model will see on every replay.
    """
    frozen_clause = (
        f"Content frozen at boundary time {frozen_at} and may have changed "
        "since — verify the live file via file_read if freshness matters to "
        "your next action."
        if frozen_at
        else "Content frozen at boundary time and may have changed since — "
        "verify the live file via file_read if freshness matters to your "
        "next action."
    )
    lines = [
        f"[GC boundary {boundary_index} — durable context snapshot]",
        f"Project: {_frame_field(str(resolution.get('project') or 'none'))}",
        f"Trust level: workspace-file DATA at file_read level, NOT "
        f"harness-authoritative. {frozen_clause}",
    ]
    for f in resolution.get("files", []):
        if f.get("exists"):
            escaped = _freeze_file_text(f.get("text", ""))
            lines.append(f"--- BEGIN {_frame_field(f['path'])} ({_frame_field(f['reason'])}) ---")
            lines.append(escaped)
            lines.append(f"--- END {_frame_field(f['path'])} ---")
    missing = resolution.get("missing", [])
    if missing:
        reasons = {f["path"]: f["reason"] for f in resolution.get("files", [])}
        for path in missing:
            lines.append(
                f"[missing at snapshot time: {_frame_field(path)} "
                f"({_frame_field(reasons.get(path, 'n/a'))}) — "
                "re-read via file_read if it has since been created]"
            )
    for err in resolution.get("errors", []):
        lines.append(f"[error: {_frame_field(str(err))}]")
    if over_budget:
        lines.append(
            f"[durable budget {resolution.get('used_tokens', 0)}/{budget_tokens} "
            "tokens — over reinjection budget; informational only, prune "
            "durable-set.toml when convenient]"
        )
    else:
        lines.append(
            f"[durable budget {resolution.get('used_tokens', 0)}/{budget_tokens} tokens]"
        )
    return "\n".join(lines) + "\n"


def _entry_content_chars(content) -> int:
    """Deterministic char estimate of one entry's renderable text content."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, (list, tuple)):
        total = 0
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    total += len(text)
            elif isinstance(part, str):
                total += len(part)
        return total
    return 0


def _tool_output_chars(entry: dict) -> int:
    """Char length of a tool entry's renderable output ("" when not str)."""
    output = entry.get("output", "")
    return len(output) if isinstance(output, str) else 0


def _is_pre_boundary(entry: dict, boundary_index: int, position: int | None = None) -> bool:
    """True for entries strictly BEFORE the boundary position.

    Position-based: ``entry_index`` is written only on marker events, so
    classifying by that field would count every ordinary entry (default 0)
    as pre-boundary. Callers pass the enumerate() position; the entry-key
    fallback exists only for marker-style entries that carry one.
    """
    if position is not None:
        return position < boundary_index
    return entry.get("entry_index", 0) < boundary_index


def _manifest_classes(entries: list[dict], boundary_index: int,
                      retained: frozenset[int] | None = None) -> dict:
    """Class counts over pre-boundary entries (deterministic).

    tools: tool-result entries; thinking: pre-boundary assistant entries
    whose thinking is STRIPPED (deleted or thinking-removed — i.e. NOT at a
    retained position); thinking_retained (workspace-kdsn.305.13 T5): the
    subset at retained positions (``retained`` — positions from
    ``thinking_tail_indices``; None or empty = legacy full strip, count 0).
    The two are a partition of the pre-boundary assistant entries carrying
    thinking: ``thinking + thinking_retained == total-with-thinking`` (the
    retained set only ever contains entries that carry truthy thinking, so
    every retained position is counted exactly once, in thinking_retained).
    inputs: RETIRED (workspace-37ch — input compaction removed; inputs
    render verbatim) — always 0, key retained for schema stability;
    media: pre-boundary user entries whose content is a non-string (part)
    list.
    """
    # "inputs" is pinned 0 (retired, workspace-37ch): the key stays in the
    # manifest so consumers (matrix.py, tools/__init__.py, subagent.py,
    # frame_sub_snapshot) keep working unchanged.
    tools = thinking = thinking_retained = inputs = media = 0
    retained = retained or frozenset()
    for position, entry in enumerate(entries):
        if _is_pre_boundary(entry, boundary_index, position):
            role = entry.get("role")
            if role == "tool":
                tools += 1
            elif role == "assistant":
                if entry.get("thinking"):
                    if position in retained:
                        thinking_retained += 1
                    else:
                        thinking += 1
                # (inputs counting retired, workspace-37ch — verbatim render)
            elif role == "user" and isinstance(entry.get("content"), (list, tuple)):
                media += 1
            elif (role == "user" and isinstance(entry.get("content"), str)
                  and MEDIA_TAG_RE.search(entry["content"])):
                # [media: ...] tag-string form (wave-2.1, wonmun A7): expunged
                # at the boundary exactly like list-form media — count it.
                media += 1
    return {"tools": tools, "thinking": thinking, "thinking_retained": thinking_retained,
            "media": media, "inputs": inputs}


def _render_char_estimates(entries: list[dict], boundary_index: int,
                           retained: frozenset[int] | None = None) -> tuple[int, int]:
    """(tokens_before, tokens_after_est) char//4 estimates — deterministic.

    before: every pre-boundary entry's full renderable text.

    after (kdsn.305.12 R4): the post-boundary RENDER estimate of the
    expunged span — what survives reduction in place:
      - tool outputs -> their pointer strings (the expunged bytes are gone);
      - media attachments (list/tuple content, or the [media: ...] tag
        string) -> the expunged-media placeholder (they do NOT survive
        verbatim — counted at placeholder size, not content size);
      - user/assistant TEXT content -> rendered in full after the boundary
        (it survives reduction, so it contributes its full content chars —
        the pre-R4 estimate was blind to it, which understated
        tokens_after and delayed the handoff, the wrong direction);
      - assistant thinking -> dropped (counted in before only) — EXCEPT the
        retained thinking tail (workspace-kdsn.305.13 T3/T3a): when
        ``retained`` (positions from ``thinking_tail_indices``) is passed,
        the retained blocks' chars (via the shared ``_thinking_block_chars``
        — single source with the selection's block estimate) are ADDED to
        after_chars, because they remain renderable pre-boundary content.
        ``tokens_before`` keeps its content-only basis either way (mild
        asymmetry: before never counted thinking; after now counts the
        retained subset — documented, never a blocker);
      - assistant tool_call string inputs -> the full input length
        (exact since 37ch: inputs render verbatim, so the estimate counts
        precisely what the render emits).
    Post-boundary entries contribute zero here — the in-flight tail is
    measured separately by ``_post_boundary_tail_tokens``; the two combine
    into the composite ``manifest.runway.tokens_after``.

    The retained set's chars are summed and divided ONCE at the end (with
    the rest of the span) — the selection helper divides per block; the
    shared division carry may therefore shift tokens_after_est by exactly
    one token relative to the selection-fit math (the red suite pins the
    delta to {q, q+1}).
    """
    before_chars = 0
    after_chars = 0
    retained = retained or frozenset()
    for position, entry in enumerate(entries):
        if not _is_pre_boundary(entry, boundary_index, position):
            continue
        role = entry.get("role")
        before_chars += _entry_content_chars(entry.get("content"))
        if role == "tool":
            output_chars = _tool_output_chars(entry)
            before_chars += output_chars
            # Pointer stands in for the full output after the boundary.
            after_chars += len(tool_pointer(boundary_index, entry.get("name", "?"), None,
                                            output_chars))
        elif role == "assistant":
            # Surviving text renders in full after the boundary (R4).
            after_chars += _entry_content_chars(entry.get("content"))
            for tc in entry.get("tool_calls") or []:
                value = tc.get("input") if isinstance(tc, dict) else None
                if isinstance(value, str):
                    after_chars += len(value)
            if position in retained:
                # Retained thinking tail (workspace-kdsn.305.13 T3): the
                # block survives reduction and renders verbatim — count its
                # chars (shared _thinking_block_chars, single source with
                # the selection's block estimate).
                after_chars += _thinking_block_chars(entry.get("thinking"))
        elif (role == "user"
              and isinstance(entry.get("content"), (list, tuple, str))
              and not (isinstance(entry.get("content"), str)
                       and MEDIA_TAG_RE.search(entry["content"]))):
            # Surviving text renders in full after the boundary (R4).
            after_chars += _entry_content_chars(entry.get("content"))
        elif (role == "user" and isinstance(entry.get("content"), str)
              and MEDIA_TAG_RE.search(entry["content"])):
            # Media (list/tuple or tag-string form) is expunged to the
            # placeholder, which is what renders after the boundary.
            after_chars += len(_expunged_media_string(boundary_index))
    return before_chars // 4, after_chars // 4


def _post_boundary_tail_tokens(entries: list[dict], boundary_index: int) -> int:
    """chars//4 estimate of the post-boundary tail (entries at position >=
    boundary_index — the in-flight tail that stays after the boundary when
    exclude_inflight; empty for a settled turn).

    The runway composite (kdsn.305.12 D1) needs it because
    ``tokens_after_est`` covers only the expunged span. Entries the module
    never reads are counted by their content via the shared
    ``_entry_content_chars`` helper; anything garbled (non-dict entry,
    non-string/str-part-list content) contributes 0 — the estimator is
    best-effort and must never raise inside a boundary.
    """
    total = 0
    for position in range(boundary_index, len(entries)):
        entry = entries[position]
        if not isinstance(entry, dict):
            continue
        total += _entry_content_chars(entry.get("content"))
    return total // 4


def project_echo_text(workspace: Path, project: str) -> str:
    """Confirmation body for a project declaration: echoes the parsed
    durable-set entries (path + reason). Malformed/missing durable-set.toml
    is named plainly — never silent. Pure read; never raises. Shared by the
    set_active_project tool callback and the /project room command."""
    proj_dir = workspace / "memory" / "projects" / project
    toml_path = proj_dir / "durable-set.toml"
    lines = [f"Project set to **{project}**."]
    if not toml_path.exists():
        lines.append(
            f"Durable set: none — no durable-set.toml at "
            f"{proj_dir / 'durable-set.toml'}."
        )
        return "\n".join(lines)
    try:
        entries = parse_durable_set(toml_path.read_text(encoding="utf-8"))
    except GCConfigError as e:
        lines.append(f"Durable set: MALFORMED — {e}")
        return "\n".join(lines)
    if not entries:
        lines.append("Durable set: 0 entries (durable-set.toml has no entries).")
    else:
        lines.append(f"Durable entries: {len(entries)}")
        for item in entries:
            lines.append(f"  - {item['path']} — {item['reason']}")
    return "\n".join(lines)


def apply_boundary_and_rebuild(
    agent,
    session_log,
    room_id: str,
    *,
    trigger: str,
    exclude_inflight: bool,
    live_turn: bool = False,
) -> dict:
    """Apply a GC boundary and rebuild the room's in-memory history in place.

    THE shared application path — used by the callbacks-seam closure (agent
    loop auto/hard tiers + context_gc tool) and the /cache gc room command.
    Config comes from ``agent.config.context`` (ContextHandoffConfig;
    defaults via getattr for pre-305 mock configs). window for the durable
    budget is the room model's context window. The thinking-tail knobs
    (workspace-kdsn.305.13 T4a) are read from the same config
    (ContextHandoffConfig defaults
    8/32768 when the fields are absent; FULL-STRIP 0/0 when the whole
    config is absent — pre-.13 mock agents keep the legacy render) and
    passed to BOTH apply_boundary (manifest + estimator) and the
    build_context rebuild. Never raises into the caller: failures
    return {"applied": False, "noop_reason": ...}. On applied=True the room's
    in-memory history is rebuilt IN PLACE (object identity preserved — the
    running loop's next request carries the reduced context).
    """
    cfg = getattr(agent.config, "context", None)
    durable_paths = getattr(cfg, "durable_paths", []) or []
    # [context].durable_budget_pct is a PERCENT (25.0 = 25%); the seam's
    # formula expects a FRACTION (0.25). Normalize here — the one
    # config→seam adapter (audit: percent-vs-fraction unit mismatch made
    # every over-budget mechanism silently dead).
    budget_pct = getattr(cfg, "durable_budget_pct", 25.0) / 100.0
    budget_min = getattr(cfg, "durable_budget_min_tokens", 96000)
    # Runway-gated handoff thresholds (kdsn.305.12 D1): [context].
    # handoff_runway_pct is a PERCENT too — same percent→fraction adapter.
    # getattr with NEW defaults: pre-305.12 mock configs lack the fields.
    handoff_pct = getattr(cfg, "handoff_runway_pct", 10.0) / 100.0
    handoff_min = getattr(cfg, "handoff_runway_min_tokens", 24000)
    # Thinking-tail preservation (workspace-kdsn.305.13 T4a): the config
    # knobs flow into BOTH the apply_boundary call (manifest + estimator)
    # and the in-memory rebuild (build_context). THE seam is
    # gc_thinking_tail_kwargs(agent.config) — the same fail-closed helper
    # every production render site uses (audit M2: a raw getattr here let a
    # MagicMock ``cfg`` smuggle non-int knobs into thinking_tail_indices,
    # raising TypeError that the broad except swallowed into a silent GC
    # no-op for the room; the isinstance-gated helper forces 0/0 instead).
    _tt = gc_thinking_tail_kwargs(agent.config)
    thinking_tail_turns = _tt["thinking_tail_turns"]
    thinking_tail_max_tokens = _tt["thinking_tail_max_tokens"]
    try:
        window = agent._resolve_model_limit(room_id)
    except Exception as e:  # fail-soft: a window resolution failure must not
        logger.warning("gc window resolution failed for %s: %s", room_id, e)
        window = getattr(agent.config, "model_max_tokens", 200000)
    # Usable runway (im7t.46 D9 single-source): available must come from the
    # agent's own expression (window − output reserve), NOT be re-derived
    # here. Agent._effective_available(self, limit: int) takes the RESOLVED
    # WINDOW LIMIT (an int), not a room id — pass the window resolved above.
    # getattr-guarded for pre-305 mock agents that lack the method. A raise
    # inside it (a pre-305 mock, or a signature mismatch on the agent's own
    # method) degrades the runway math to window-only — but it is ALWAYS
    # logged at WARNING so a REAL Agent mismatch can never be swallowed
    # invisibly (audit R1: a swallowed TypeError silently widened `available`
    # by the output reserve and delayed the handoff — spec D1b's wrong
    # direction). The boundary still applies either way.
    available = window
    _eff = getattr(agent, "_effective_available", None)
    if callable(_eff):
        try:
            available = _eff(window)
        except Exception as e:  # fail-soft: runway math degrades to
            logger.warning(          # window-only, the boundary still applies
                "gc available resolution failed for %s: %s", room_id, e)
            available = window
    try:
        outcome = apply_boundary(
            session_log,
            room_id,
            workspace=Path(agent.config.workspace),
            trigger=trigger,
            exclude_inflight=exclude_inflight,
            config_paths=list(durable_paths),
            window=window,
            budget_pct=budget_pct,
            budget_min=budget_min,
            max_tokens=max(window - available, 0),
            handoff_pct=handoff_pct,
            handoff_min=handoff_min,
            thinking_tail_turns=thinking_tail_turns,
            thinking_tail_max_tokens=thinking_tail_max_tokens,
        )
    except Exception as e:
        logger.warning("gc boundary failed for %s: %s", room_id, e, exc_info=True)
        return {
            "applied": False,
            "noop_reason": f"boundary failed: {type(e).__name__}",
            "manifest": None,
            "over_budget": False,
        }
    if outcome.get("applied"):
        # Materialize the rebuilt context BEFORE touching the live history —
        # a failed build must leave the old history intact (audit: clear()
        # before a fallible build left rooms amnesiac on transient I/O errors).
        try:
            rebuilt = session_log.build_context(
                room_id, gc_enabled=True, gc_preserve_trailing=live_turn,
                thinking_tail_turns=thinking_tail_turns,
                thinking_tail_max_tokens=thinking_tail_max_tokens)
        except Exception as e:
            logger.warning(
                "gc history rebuild failed for %s: %s — keeping the "
                "pre-boundary in-memory history (the next render picks up "
                "the boundary from JSONL)", room_id, e, exc_info=True
            )
        else:
            history = agent.history(room_id)
            history.clear()
            history.extend(rebuilt)
    return outcome


def project_valid_name(project: str) -> bool:
    """Bare-directory-name guard: the active_project detail feeds
    workspace-relative path joins at boundary application — no separators,
    no traversal, no dots-prefix."""
    return bool(project) and not (
        "/" in project
        or "\\" in project
        or project.startswith(".")
        or project in ("", ".", "..")
    )


def frame_forced_handoff(runway_after: int, available: int) -> str:
    """The runway-exhausted forced-handoff directive (kdsn.305.12 D1),
    stored as a reminder entry.

    Gates on POST-BOUNDARY runway, never on durable-set size: this fires
    when the post-snapshot residue consumes nearly all usable runway and
    the room cannot make useful forward progress. Deterministic; no
    timestamps (append-only render invariant)."""
    pct = 0 if available <= 0 else min(100, int((available - runway_after)
                                                * 100 / available))
    return (
        "&lt;system-reminder&gt;\n"
        f"Post-GC context consumes {pct}% of the usable runway — "
        f"{runway_after} tokens remain after this boundary. The room cannot "
        "make useful forward progress in what is left: finalize the "
        "continuity artifact (progress.md State/Decisions/Next) and execute "
        "session-handoff before the next turn.\n"
        "&lt;/system-reminder&gt;"
    )


def _raise_handoff_bead(room_id: str, project: str | None, bd_path: str | None) -> None:
    """Raise the handoff pointer bead (cross-session signal, per the
    session-handoff skill). Deterministic harness action — the agent might
    ignore a directive; the bead cannot be ignored by the next session's
    triage. Fail-soft: any failure logs and moves on."""
    if not bd_path:
        return
    try:
        title = f"GC forced handoff: post-GC runway exhausted ({room_id})"
        desc = (
            "The context-GC post-boundary runway was nearly exhausted at a "
            "GC boundary (the durable budget is an informational marker and "
            "is NOT the trigger). The harness injected the forced-handoff "
            f"directive (active project: {project or 'none'}). Execute "
            "session-handoff triage."
        )
        subprocess.run(
            [bd_path, "create", title, "-t", "task", "-p", "P2",
             "-l", "handoff", "-d", desc],
            timeout=15,
            capture_output=True,
            check=False,
        )
    except Exception as e:  # never break a boundary over bead bookkeeping
        logger.warning("gc forced-handoff bead raise failed for %s: %s",
                       room_id, e)


def apply_boundary(
    session_log,
    room_id: str,
    *,
    workspace: Path,
    trigger: str,
    exclude_inflight: bool = False,
    config_paths: list[str] | None = None,
    window: int,
    budget_pct: float,
    budget_min: int,
    max_tokens: int | None = None,
    handoff_pct: float = 0.10,
    handoff_min: int = 24000,
    bd_path: str | None = BD_PATH,
    thinking_tail_turns: int = 0,
    # NOTE the asymmetry (workspace-kdsn.305.13 T4/T5, pinned by the red
    # suite's TestTailEstimator._apply): ``thinking_tail_turns`` defaults to
    # 0 (FAIL-CLOSED — no retention unless a caller turns it ON) while
    # ``thinking_tail_max_tokens`` defaults to the config default 32768.
    # The two default together = full strip (turns 0 -> empty set regardless
    # of ceiling). The ceiling default mirrors the config default (and the
    # .12 handoff_min=24000 precedent) so a caller that sets ``turns>0`` and
    # omits the ceiling gets the documented config ceiling, not a silent
    # 0-ceiling that would make retention a no-op. The PRODUCTION caller
    # (apply_boundary_and_rebuild) always passes BOTH explicitly, so this
    # default only affects direct apply_boundary(...) callers / tests.
    thinking_tail_max_tokens: int = 32768,
) -> dict:
    """Apply a GC boundary: append manifest event + snapshot entry to JSONL.

    The ONLY writer of boundary state. Never rewrites existing entries.

    boundary index = len(entries) + 1 - 2*(1 if exclude_inflight else 0).
    The boundary marks the FIRST entry NOT yet covered: it lands at the
    next-append position, and when called from inside a live tool loop
    (``exclude_inflight=True``) the trailing in-flight assistant entry is
    pulled back in — it stays POST-boundary so pairing, thinking, and inputs
    survive intact. (Note: this supersedes the pre-implementation docstring
    wording "len - (1 if exclude_inflight else 0)"; the byte-pinned tests
    fix the boundary at len+1 for a settled turn and len-1 in-flight.)

    Monotonicity (mechanical): if the computed index does not EXCEED
    current_boundary_index (computed <= current), nothing is appended and
    {"applied": False, "noop_reason": ...} returns.

    Appends, in order:
      1. system entry event=GC_EVENT, entry_index=<index>, detail=json manifest:
         {ts, boundary_index, trigger, classes: {tools, thinking,
          thinking_retained, media, inputs},
          tokens_before, tokens_after_est, durable: {project, files:
          [{path, reason, origin, chars}], budget_tokens, used_tokens,
          over_budget}, runway: {available, tokens_after, runway_after,
          threshold_tokens, handoff_advised}, errors}
      where runway.tokens_after is the POST-BOUNDARY RENDER estimate
      (tokens_after_est + framed-snapshot bytes//4 + post-boundary tail
      chars//4), runway_after = available - runway.tokens_after.
      2. user entry source=GC_SNAPSHOT_SOURCE with frame_snapshot() content.

    Over budget (kdsn.305.12 D2): manifest + snapshot-header marker record
    it (informational — the budget is NOT a cap); the snapshot still includes
    ALL durable files (NEVER degrade-to-pointers for durable-class content —
    SB ruling). Over-budget alone NEVER forces a handoff.

    Runway-gated forced handoff (kdsn.305.12 D1): available = window −
    max_tokens (None → 0); tokens_after = tokens_after_est (expunged-span
    residual) + framed-snapshot bytes//4 (ALWAYS appended) + the
    post-boundary tail chars//4 (entries at/after boundary_index — the
    in-flight tail under exclude_inflight, zero for a settled turn);
    runway_after = available − tokens_after;
    threshold = max(int(window * handoff_pct), handoff_min),
    handoff_advised = runway_after < threshold (strict < — equality does
    NOT fire). The forced-handoff directive + handoff bead fire ONLY when
    handoff_advised — decoupled from the durable-set budget (the durable
    set may be over budget with ample runway: marker only; it may be
    under budget with exhausted runway: handoff still fires). The composite
    lives ONLY in manifest.runway.tokens_after — tokens_after_est keeps its
    documented meaning (expunged-span residual estimate, operator-visible in
    confirm text and subagent manifests). Once per epoch: the existing
    latch scans the room's JSONL for a prior GC_FORCED_HANDOFF_TRIGGER
    reminder entry.

    Thinking-tail preservation (workspace-kdsn.305.13 T3/T3a): with
    thinking_tail_turns / thinking_tail_max_tokens non-zero, the retained
    pre-boundary thinking blocks' chars are added to tokens_after_est (via
    the shared selection set — see ``thinking_tail_indices``), so the .12
    runway composite (runway.tokens_after) grows with retention and the
    handoff gate sees the retained load.  The manifest classes gain
    "thinking_retained" (present even when 0); "thinking" keeps its
    STRIPPED count semantics — it counts the pre-boundary assistant entries
    carrying thinking that were NOT retained (thinking +
    thinking_retained == total with thinking).  The kwarg defaults are
    ASYMMETRIC by decision (workspace-kdsn.305.13): thinking_tail_turns
    defaults to 0 (FAIL-CLOSED — no retention unless a caller turns it ON)
    while thinking_tail_max_tokens defaults to 32768, mirroring the config
    default (the .12 kwarg-defaults-equal-production-defaults precedent:
    handoff_min=24000). With turns=0 the ceiling is moot — the retained set
    is empty regardless — so the defaults together still give a full strip,
    but a caller that sets turns>0 and omits the ceiling gets the documented
    32768 ceiling instead of a silent 0-ceiling no-op. The PRODUCTION caller
    (apply_boundary_and_rebuild) always passes both explicitly, so this
    default only affects direct apply_boundary(...) callers / tests.
    turns=0 → byte-identical legacy manifest (empty retained set).

    Durable problems (missing files, malformed TOML, unreadable) never raise —
    they land in errors/warnings and the boundary still applies.
    Returns {"applied": bool, "noop_reason": str | None, "manifest": dict | None,
             "over_budget": bool, "handoff_advised": bool,
             "forced_handoff": bool}.
    """
    workspace = Path(workspace)
    entries = session_log.read(room_id)
    # Boundary = position of the first entry NOT yet covered: the next-append
    # position (len+1) for a settled turn; minus one more when the trailing
    # in-flight assistant entry is excluded so it stays post-boundary.
    # Pinned by tests/test_context_gc.py::TestApplyBoundary (happy path:
    # 5 scene entries -> boundary 6; in-flight: 6 entries -> boundary 5).
    boundary_index = len(entries) + 1 - (2 if exclude_inflight else 0)
    current = current_boundary_index(entries)
    if boundary_index <= current:
        logger.info(
            "gc_boundary refused: computed index %d <= current boundary index %d",
            boundary_index,
            current,
        )
        return {
            "applied": False,
            "noop_reason": (
                f"computed boundary index {boundary_index} does not exceed "
                f"current boundary index {current}"
            ),
            "manifest": None,
            "over_budget": False,
            "handoff_advised": False,
        }

    project = read_active_project(entries)
    resolution = resolve_durable_set(workspace, project, config_paths)
    budget_tokens = durable_budget_tokens(window, budget_pct, budget_min)
    used_tokens = resolution["used_tokens"]
    over_budget = used_tokens > budget_tokens
    # Thinking-tail selection (workspace-kdsn.305.13 T3a): apply_boundary
    # owns entries + boundary_index + the tail config, so it computes the
    # retained set ONCE and passes it to both the class tally and the
    # char-estimate (the estimator does not re-select).  Default-zero
    # kwargs → empty set → byte-identical legacy manifest.
    retained_tail = thinking_tail_indices(
        entries, boundary_index, thinking_tail_turns, thinking_tail_max_tokens)
    classes = _manifest_classes(entries, boundary_index, retained_tail)
    tokens_before, tokens_after_est = _render_char_estimates(
        entries, boundary_index, retained_tail)

    # Runway-gated handoff (kdsn.305.12 D1): the handoff decision is about
    # POST-BOUNDARY runway, never about durable-set size.  The composite
    # tokens_after is the POST-BOUNDARY RENDER estimate: the expunged-span
    # residual (tokens_after_est) PLUS the framed snapshot bytes ALWAYS
    # appended at this boundary PLUS the in-flight tail (entries at/after
    # boundary_index — zero for a settled turn).  A small durable set plus a
    # fat in-flight tail still triggers the handoff; a big durable snapshot
    # correctly consumes runway even though tokens_after_est never sees it.
    # The chars//4 estimator undercounts, which OVERSTATES runway_after and
    # therefore DELAYS the handoff — the wrong direction, strictly (accepted
    # v1, spec D1b: the 75/85/92 ladder guards exhaustion between
    # boundaries).
    _mt = 0 if max_tokens is None else int(max_tokens)
    available = int(window) - _mt
    # Single ts for the whole boundary application: the manifest stamp and
    # the snapshot's frozen_at must be the SAME value (frozen snapshot bytes
    # are deterministic given frozen_at — two now() calls could disagree).
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshot = frame_snapshot(boundary_index, resolution, over_budget,
                              budget_tokens, frozen_at=ts)
    tokens_after = tokens_after_est + len(snapshot) // 4 \
        + _post_boundary_tail_tokens(entries, boundary_index)
    runway_after = available - tokens_after
    threshold_tokens = max(int(int(window) * handoff_pct), int(handoff_min))
    handoff_advised = runway_after < threshold_tokens

    manifest = {
        "ts": ts,
        "boundary_index": boundary_index,
        "trigger": trigger,
        "classes": classes,
        "tokens_before": tokens_before,
        "tokens_after_est": tokens_after_est,
        "durable": {
            "project": project,
            "files": [
                {
                    "path": f["path"],
                    "reason": f["reason"],
                    "origin": f["origin"],
                    "chars": len(f["text"]),
                }
                for f in resolution["files"]
            ],
            "budget_tokens": budget_tokens,
            "used_tokens": used_tokens,
            "over_budget": over_budget,
        },
        "runway": {
            "available": available,
            "tokens_after": tokens_after,
            "runway_after": runway_after,
            "threshold_tokens": threshold_tokens,
            "handoff_advised": handoff_advised,
        },
        "errors": resolution["errors"],
    }

    # Belt against non-serializable stragglers; values are JSON-native anyway.
    detail = json.dumps(manifest, ensure_ascii=False, default=str)

    session_log.append(
        role="system",
        sender=session_log.agent_user_id,
        room=room_id,
        event=GC_EVENT,
        entry_index=boundary_index,
        detail=detail,
    )
    session_log.append(
        role="user",
        sender=session_log.agent_user_id,
        room=room_id,
        content=snapshot,
        source=GC_SNAPSHOT_SOURCE,
    )

    if over_budget:
        # kdsn.305.12 D2: over-budget is demoted to the snapshot-header
        # marker (informational) + manifest record.  NO directive, NO bead,
        # NO ntfy — the full durable set is attached verbatim regardless.
        logger.warning(
            "gc_boundary %d: durable set over reinjection budget (%d > %d "
            "tokens) — informational marker only",
            boundary_index,
            used_tokens,
            budget_tokens,
        )

    # FORCED HANDOFF (kdsn.305.12 D1, reworked from kdsn.305.3): fires ONLY
    # when the POST-BOUNDARY runway is exhausted — the room genuinely cannot
    # make useful forward progress.  Decoupled from the durable-set budget
    # (over-budget with ample runway is marker-only; under-budget with
    # exhausted runway still hands off). Once per epoch: skip when this
    # room's JSONL already carries the directive (a second boundary in the
    # same epoch must not re-fire). Fail-soft end to end: a bd failure logs
    # and moves on — the reminder entry is the durable signal.
    forced_handoff = False
    if handoff_advised:
        logger.warning(
            "gc_boundary %d: post-boundary runway exhausted (%d < %d tokens "
            "threshold) — forced handoff",
            boundary_index,
            runway_after,
            threshold_tokens,
        )
        if not any(
            e.get("role") == "user" and e.get("source") == "reminder"
            and e.get("trigger") == GC_FORCED_HANDOFF_TRIGGER
            for e in entries
        ):
            forced_handoff = True
            session_log.append(
                role="user",
                sender=session_log.agent_user_id,
                room=room_id,
                content=frame_forced_handoff(runway_after, available),
                source="reminder",
                trigger=GC_FORCED_HANDOFF_TRIGGER,
            )
            _raise_handoff_bead(room_id, project, bd_path)

    return {
        "applied": True,
        "noop_reason": None,
        "manifest": manifest,
        "over_budget": over_budget,
        "forced_handoff": forced_handoff,
        "handoff_advised": handoff_advised,
    }

# --- Message-list boundary transform (workspace-kdsn.305.4) -----------------

#: Pre-frame prefix identifying a sub-snapshot user message in a plain
#: message list. Load-bearing: the transform's superseded-snapshot rule
#: detects prior snapshots by this prefix (message lists have no source
#: field the way JSONL entries do).
SUB_SNAPSHOT_PREFIX = "[GC sub-snapshot — boundary "


def frame_sub_snapshot(boundary_index: int, task_text: str, manifest: dict,
                       trigger: str = "auto") -> str:
    """Frozen sub-snapshot content for the message-list boundary path.

    Single string, first line EXACTLY:
        [GC sub-snapshot — boundary {boundary_index} ({trigger})]
    followed by:
        task: <task_text escaped via _escape_reminder_tags — model-origin text>
        manifest: tools={t} thinking={th} media={m} inputs={i}; tokens {before} -> {after} (est.); durable: task preserved in full

    The durable ruling (SB): the task text is preserved IN FULL, never
    truncated, never degraded to a pointer — over budget is flagged in the
    manifest by the caller, never by dropping task content here.
    Deterministic; no locale-dependent formatting; plain ints.
    """
    classes = manifest.get("classes", {})
    return (
        f"[GC sub-snapshot — boundary {boundary_index} ({trigger})]\n"
        f"task: {_escape_reminder_tags(task_text)}\n"
        f"manifest: tools={classes.get('tools', 0)} "
        f"thinking={classes.get('thinking', 0)} "
        f"media={classes.get('media', 0)} "
        f"inputs={classes.get('inputs', 0)}; "
        f"tokens {manifest.get('tokens_before', 0)} -> "
        f"{manifest.get('tokens_after_est', 0)} (est.); "
        "durable: task preserved in full"
    )


def _harvest_params(raw_input):
    """Pairing-map params for pointer formatting: the input dict minus
    already-stripped placeholder values — a second boundary over a
    once-reduced list would otherwise cite "[stripped: N chars]" as the
    pointer's identifying param (wave-2 audit). None when nothing usable
    remains."""
    if not isinstance(raw_input, dict):
        return None
    cleaned = {k: v for k, v in raw_input.items()
               if not (isinstance(v, str) and v.startswith("[stripped: "))}
    return cleaned or None


def apply_boundary_to_messages(messages: list[dict], *, boundary_index: int,
                               task_text: str, trigger: str = "auto",
                               thinking_tail_turns: int = 0,
                               thinking_tail_max_tokens: int = 0) -> dict:
    """Apply a GC boundary to a plain OpenAI-style message list. PURE.

    The message-list analog of the session.py render transform: every message
    at list position < boundary_index is reduced by the uniform rule;
    positions >= boundary_index are untouched. The input list and its dicts
    are NEVER mutated (transformed entries are copied; the input list keeps
    its original nested values); the result carries a new list.

    Uniform rule over pre-boundary messages (mirrors session.py render):
      - user msg whose str content starts with SUB_SNAPSHOT_PREFIX -> dropped
        entirely (superseded by this boundary's snapshot; never
        placeholder-ized).
      - user msg with list/tuple content (media parts) -> content replaced by
        the expunged-media string:
        "[expunged at GC boundary {boundary_index}: media attachment — re-share or re-generate the image if needed]"
        (byte-identical to session.py's render string; count "media").
      - assistant msg: "thinking" field dropped (count "thinking" per message
        that carried a non-empty thinking list). Thought-only message (string
        content empty/whitespace AND no tool_calls) is deleted atomically —
        never an empty shell. tool_calls: inputs render VERBATIM since 37ch
        (input compaction retired — no "[stripped: N chars]" placeholders;
        the "inputs" class stays in the manifest, pinned 0).
      - assistant thinking-tail retention (workspace-kdsn.305.13 T4 parity):
        with thinking_tail_turns / thinking_tail_max_tokens non-zero, the
        positions from ``thinking_tail_indices`` (same selection over the
        message list, position semantics as everywhere here) keep their
        "thinking" field VERBATIM (both dict and ToolCall-object input
        shapes — only the thinking key is consulted, tool_calls are
        transformed exactly as legacy).  count "thinking_retained" per such
        message; "thinking" counts the STRIPPED ones (pre-boundary assistant
        msgs carrying thinking that were NOT retained — the two partition
        the total).  Retained thinking chars join the after-estimate
        (shared ``_thinking_block_chars`` — single source with the
        selection).  Default-zero kwargs → byte-identical legacy transform.
      - tool msg: content replaced by tool_pointer(boundary_index, name,
        params, n_chars) where (name, params) come from the paired
        pre-boundary assistant tool_call harvested by tool_call_id (pairing
        map built in the same pass — no second scan); n_chars = original
        content length (str content only, else 0). No pair found -> name
        "tool", params None (generic pointer). Count "tools" per pointer.
      - user msg with plain str content (non-snapshot) -> passes through
        UNCHANGED. (Main-path parity: text inputs are not placeholder-ized.)

    tool_calls entries may be ToolCall objects OR plain dicts — accept both;
    the output mirrors the input type (objects via dataclasses.replace with
    the raw input; dicts as new dicts — inputs verbatim since 37ch).
    ToolCall is imported lazily inside
    the function body from openalph.provider to keep context_gc importable
    standalone.

    Returns dict:
      {
        "applied": True,
        "messages": <new list: transformed pre-boundary messages, untouched
                     post-boundary messages, then ONE appended user message
                     {"role": "user", "content": frame_sub_snapshot(...)}>,
        "manifest": {
            "boundary_index": int, "trigger": str,
            "classes": {"tools": int, "thinking": int, "thinking_retained": int,
                        "media": int, "inputs": int},
            "tokens_before": int, "tokens_after_est": int,
            "messages_before": int, "messages_after": int,
        },
        "snapshot_content": <the framed snapshot string>,
        "noop_reason": None,
      }
    Never returns applied=False (monotonicity is the SEAM's job, tracked in
    the caller's own counter — message lists carry no marker entries).
    """
    import dataclasses
    from openalph.provider import ToolCall as _ToolCall

    # tool_call_id -> (name, ORIGINAL input). Harvested from pre-boundary
    # assistant tool_calls IN THE TRANSFORM PASS BELOW (no second scan —
    # mirrors session.py: an assistant call always precedes its tool result,
    # so a pre-boundary tool message's pair is already collected by the time
    # that tool message is rendered).
    pairing: dict = {}
    out: list = []
    # "inputs" retired (workspace-37ch): pinned 0, key retained for schema
    # stability.
    classes = {"tools": 0, "thinking": 0, "thinking_retained": 0,
               "media": 0, "inputs": 0}
    before_chars = 0
    after_chars = 0
    # Thinking-tail retention (workspace-kdsn.305.13 T4 parity): the same
    # selection as the entry path, over the message list (position
    # semantics). Default-zero kwargs → empty set → legacy transform.
    retained_tail = thinking_tail_indices(
        messages, boundary_index, thinking_tail_turns, thinking_tail_max_tokens)

    for pos, m in enumerate(messages):
        if pos < boundary_index:
            role = m.get("role")
            content = m.get("content")
            before_chars += _message_content_chars(content)

            if role == "user":
                if (
                    isinstance(content, str)
                    and content.startswith(SUB_SNAPSHOT_PREFIX)
                ):
                    continue  # superseded snapshot: dropped, contributes 0
                if isinstance(content, (list, tuple)):
                    classes["media"] += 1
                    after_chars += len(_expunged_media_string(boundary_index))
                    out.append({"role": "user",
                                "content": _expunged_media_string(boundary_index)})
                elif isinstance(content, str) and MEDIA_TAG_RE.search(content):
                    # [media: ...] tag-string form (wave-2.1, wonmun A7):
                    # expunged exactly like list-form media.
                    classes["media"] += 1
                    after_chars += len(_expunged_media_string(boundary_index))
                    out.append({"role": "user",
                                "content": _expunged_media_string(boundary_index)})
                else:
                    # Plain text passes through unchanged — surviving text
                    # contributes its full length to the after-estimate.
                    after_chars += _message_content_chars(content)
                    out.append(dict(m))
            elif role == "assistant":
                tcs = m.get("tool_calls") or []
                new_tc = []
                for tc in tcs:
                    if isinstance(tc, _ToolCall):
                        raw_input = tc.input
                        cid = tc.id
                        name = tc.name
                    else:
                        raw_input = tc.get("input")
                        cid = tc.get("call_id") or tc.get("id", "")
                        name = tc.get("name", "")
                    # Harvest (name, RAW input) for pointer pairing in the
                    # same pass that renders the call. The pointer ident
                    # cites the original value, so an over-500 identifying
                    # param (e.g. a long path) still shows in the pointer;
                    # _harvest_params also filters legacy "[stripped: "
                    # markers still present in pre-37ch lists.
                    if cid:
                        pairing[cid] = (name, _harvest_params(raw_input))
                    # 37ch: the input passes through VERBATIM (dict or
                    # non-dict, e.g. null) — rewriting it would falsify the
                    # model-visible history (wave-2 audit, 3 lineages).
                    if isinstance(tc, _ToolCall):
                        new_tc.append(dataclasses.replace(tc, input=raw_input))
                    else:
                        new_tc.append({**tc, "input": raw_input})
                    # The after-estimate counts the input's FULL length —
                    # exact since 37ch (the render no longer shortens it).
                    if isinstance(raw_input, dict):
                        for v in raw_input.values():
                            if isinstance(v, str):
                                after_chars += len(v)
                if m.get("thinking"):
                    if pos in retained_tail:
                        # Retained thinking tail (workspace-kdsn.305.13 T4):
                        # the block survives VERBATIM (str or list-of-blocks
                        # — both provider shapes) and its chars join the
                        # after-estimate (shared _thinking_block_chars).
                        classes["thinking_retained"] += 1
                        after_chars += _thinking_block_chars(m["thinking"])
                    else:
                        classes["thinking"] += 1
                content_str = content if isinstance(content, str) else ""
                if not content_str.strip() and not tcs:
                    continue  # thought-only: deleted atomically, contributes 0
                # Surviving text contributes its length to the after-estimate
                # (dropped thinking was never part of "content"; retained
                # thinking is counted above).
                after_chars += _message_content_chars(content)
                nm = dict(m)
                if pos not in retained_tail:
                    nm.pop("thinking", None)
                if tcs:
                    nm["tool_calls"] = new_tc
                out.append(nm)
            elif role == "tool":
                classes["tools"] += 1
                content_len = len(content) if isinstance(content, str) else 0
                call_id = m.get("tool_call_id")
                _n, _p = pairing.get(call_id, ("tool", None))
                # A1 (wave-2.1): an existing pointer was created by an EARLIER
                # boundary — its id IS the provenance. Preserve it; only the
                # current boundary's fresh expunges get this boundary's id.
                _stamp = boundary_index
                if isinstance(content, str):
                    _m = re.match(r"\[expunged at GC boundary (\d+):", content)
                    if _m:
                        _stamp = int(_m.group(1))
                pointer = tool_pointer(_stamp, _n, _p, content_len)
                after_chars += len(pointer)
                nm = dict(m)
                nm["content"] = pointer
                out.append(nm)
            else:
                # system or other role: pass through untouched (surviving
                # text still counts toward the after-estimate).
                after_chars += _message_content_chars(content)
                out.append(dict(m))
        else:
            # Post-boundary: untouched. A shallow copy keeps the returned
            # list independent of the input list while sharing (unmutated)
            # nested values — byte-identical either way, cheaper than a
            # deep copy of the tail.
            out.append(dict(m))

    manifest = {
        "boundary_index": boundary_index,
        "trigger": trigger,
        "classes": classes,
        "tokens_before": before_chars // 4,
        "tokens_after_est": after_chars // 4,
        "messages_before": len(messages),
        "messages_after": len(out) + 1,
    }
    snapshot = frame_sub_snapshot(boundary_index, task_text, manifest, trigger)
    out.append({"role": "user", "content": snapshot})

    return {
        "applied": True,
        "messages": out,
        "manifest": manifest,
        # Top-level classes (workspace-kdsn.305.13 T5 parity): the red suite's
        # TestTailSubagent reads ``out["classes"]`` at the top level (the same
        # dict object as ``manifest["classes"]``). Additive — the pre-.13
        # nested ``manifest["classes"]`` is unchanged, and the subagent.py
        # consumer (which reads ["messages"]/["manifest"]) is unaffected.
        "classes": classes,
        "snapshot_content": snapshot,
        "noop_reason": None,
    }


def _message_content_chars(content) -> int:
    """Deterministic char estimate of one message's renderable text content.

    Identical semantics to _entry_content_chars: str -> len; list/tuple ->
    sum of the str parts plus each dict part's str "text" value; else 0.
    """
    if isinstance(content, str):
        return len(content)
    if isinstance(content, (list, tuple)):
        total = 0
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    total += len(text)
            elif isinstance(part, str):
                total += len(part)
        return total
    return 0


def _expunged_media_string(boundary_index: int) -> str:
    """Byte-identical expunged-media placeholder used by the session.py
    render transform."""
    return (
        f"[expunged at GC boundary {boundary_index}: "
        "media attachment — re-share or re-generate the "
        "image if needed]"
    )
