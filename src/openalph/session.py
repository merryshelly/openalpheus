"""Local JSONL session persistence for OpenAlph agents.

Replaces JournalManager (Matrix rooms as persistence) with local JSONL files.
Matrix rooms remain the communication channel; session state lives on disk.

File layout:
    <workspace>/sessions/<room-id-safe>.jsonl

Where room-id-safe strips '!' and replaces ':' with '_'.
"""

import json
import os
import re
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from openalph.context_gc import (
    GC_EVENT,
    MEDIA_TAG_RE,
    _marker_indexes,
    GC_SNAPSHOT_SOURCE,
    LEGACY_EVENT,
    current_boundary_index,
    strip_thinking_entry,
    tool_pointer,
    legacy_tool_placeholder,
)
from openalph.provider import ToolCall
from openalph.spotter import frame_spotter_flag
from openalph.tools import escape_system_reminder_tags

logger = logging.getLogger(__name__)

# Max tool output before overflow to separate file (64KB)
OVERFLOW_THRESHOLD = 64 * 1024  # 64KB


class SessionLog:
    """Append-only JSONL session log for a single agent.

    One file per Matrix room. All I/O is synchronous (JSONL appends are fast
    enough that async adds complexity for no benefit). Files are fsync'd after
    each append for crash safety.
    """

    def __init__(self, workspace: str | Path, agent_user_id: str,
                 *, gc_default: bool = False):
        """Initialize session log.

        Args:
            workspace: Agent workspace directory (sessions/ created inside it)
            agent_user_id: Matrix user ID of the agent (e.g. @watson:matrix.local)
            gc_default: Default GC render mode for build_context when called
                without an explicit ``gc_enabled`` kwarg (audit: restart
                hydration / status / CLI call sites omitted the flag,
                resurrecting expunged content and superseded snapshots after
                every restart). Wired from ``config.context.gc_enabled`` by
                the transports; tests construct with the default (legacy).
        """
        self.workspace = Path(workspace)
        self.agent_user_id = agent_user_id
        self.gc_default = gc_default
        self._sessions_dir = self.workspace / "sessions"

    def _room_id_safe(self, room_id: str) -> str:
        """Convert Matrix room ID to filesystem-safe name.

        '!abc123:matrix.local' → 'abc123_matrix.local'
        """
        return room_id.lstrip("!").replace(":", "_")

    def _session_path(self, room_id: str) -> Path:
        """Deterministic path for a room's session JSONL file."""
        return self._sessions_dir / f"{self._room_id_safe(room_id)}.jsonl"

    def _safe_call_id(self, call_id: str) -> str:
        """Sanitize call_id for use as filename.

        Strips anything except alphanumeric, underscore, and hyphen to prevent
        path traversal attacks from a malicious model returning e.g.
        '../../../etc/cron.d/evil' as a call_id.
        """
        return re.sub(r'[^a-zA-Z0-9_-]', '_', call_id) or "unknown"

    def _overflow_path(self, call_id: str) -> Path:
        """Path for overflow file for a tool call."""
        return self._sessions_dir / "overflow" / f"{self._safe_call_id(call_id)}.txt"

    def _now_ts(self) -> str:
        """ISO 8601 UTC timestamp."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def append(
        self,
        *,
        role: str,
        sender: str,
        room: str,
        event_id: str | None = None,
        **kwargs,
    ) -> None:
        """Append a single JSONL entry to the session log.

        Common fields (role, sender, room, event_id, ts) are set automatically.
        Role-specific fields are passed as kwargs:
            - user/assistant: content, [tool_calls]
            - tool: call_id, name, output, [truncated], [overflow_path]
            - system: event, [detail]

        For tool entries with output > 64KB: truncates output, sets
        truncated=True, writes full output to overflow file.
        """
        # Handle large tool outputs
        if role == "tool" and "output" in kwargs:
            output = kwargs["output"]
            if isinstance(output, str) and len(output) > OVERFLOW_THRESHOLD:
                call_id = kwargs.get("call_id", "unknown")
                overflow_dir = self._sessions_dir / "overflow"
                overflow_dir.mkdir(parents=True, exist_ok=True)
                overflow_file = overflow_dir / f"{self._safe_call_id(call_id)}.txt"
                overflow_file.write_text(output, encoding="utf-8")
                kwargs["output"] = output[:OVERFLOW_THRESHOLD]
                kwargs["truncated"] = True
                kwargs["overflow_path"] = str(overflow_file)
            elif "truncated" not in kwargs:
                kwargs["truncated"] = False

        entry = {
            "ts": self._now_ts(),
            "role": role,
            "sender": sender,
            "room": room,
            "event_id": event_id,
            **kwargs,
        }

        path = self._session_path(room)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self, room_id: str) -> list[dict]:
        """Read all entries for a room.

        Returns:
            List of entry dicts in chronological order.
            Empty list if file doesn't exist.
        """
        path = self._session_path(room_id)
        if not path.exists():
            return []

        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping corrupt JSONL entry: %s", e)
        return entries

    def last_event_id(self, room_id: str) -> str | None:
        """Find the most recent non-null event_id in the session.

        Used for gap-fill: gives us the last Matrix event we know about
        so we can fetch any messages we missed.

        Returns:
            Most recent non-null event_id string, or None.
        """
        path = self._session_path(room_id)
        if not path.exists():
            return None

        last_id = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("event_id"):
                        last_id = entry["event_id"]
                except json.JSONDecodeError:
                    continue
        return last_id

    def archive(self, room_id: str) -> str:
        """Copy current session JSONL to timestamped archive.

        Returns archive filename (relative to sessions dir).
        Raises FileNotFoundError if no session file exists.
        Raises OSError on copy failure.
        """
        source = self._session_path(room_id)
        if not source.exists():
            raise FileNotFoundError(f"No session file for {room_id}")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_name = f"{self._room_id_safe(room_id)}-{ts}.jsonl"
        archive_path = self._sessions_dir / archive_name
        shutil.copy2(source, archive_path)
        return archive_name

    def wipe(self, room_id: str) -> None:
        """Truncate session JSONL to zero bytes.

        Uses truncate rather than unlink so the path stays valid.
        No-op if file doesn't exist.
        """
        path = self._session_path(room_id)
        if path.exists():
            path.write_text("")

    def strippable_stats(self, room_id: str) -> tuple[int, int]:
        """Count tool results not yet covered by a toolstrip marker.

        Returns:
            (count, total_chars) of tool result entries that would be
            affected by a new /cache toolstrip command.
        """
        entries = self.read(room_id)

        # Find the current strip boundary (max entry_index from any
        # context-boundary marker — GC or legacy toolstrip), or -1 if none.
        strip_boundary = current_boundary_index(entries)

        count = 0
        total_chars = 0
        for entry_idx, entry in enumerate(entries):
            if entry.get("role") == "tool" and entry_idx > strip_boundary:
                count += 1
                total_chars += len(entry.get("output", ""))

        return count, total_chars

    def usage_totals(self, room_id: str) -> dict:
        """Sum per-turn `usage` fields across assistant entries -> per-room counters.
        Returns zeros if no usage present. Maps:
          usage.input_tokens        -> uncached_input_tokens
          usage.output_tokens       -> total_output_tokens
          usage.cache_read_tokens   -> cache_read_tokens
          usage.cache_creation_tokens -> cache_creation_tokens
          usage.tool_calls          -> total_tool_calls
          usage.cost_usd            -> main_cost_usd
          usage.unpriced_tokens     -> unpriced_tokens
        Also re-sums frozen cost from subagent tool-result entries
        (role="tool", entry.cost_usd -> subagent_cost_usd) and advisor
        consult system entries (role="system", event="advisor_consult",
        entry.cost_usd -> advisor_cost_usd). Robust to entries with no
        `usage`/`cost_usd` key (skip).

        F3 (kdsn.218 remediation): every summed field is numeric-coerced
        fail-soft — a malformed persisted value (string/None/list) coerces to
        0 rather than raising through rehydration and bricking room wake."""
        def _num(v):
            if isinstance(v, bool):
                return 0
            return v if isinstance(v, (int, float)) else 0
        totals = {
            "uncached_input_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "total_output_tokens": 0,
            "total_tool_calls": 0,
            "main_cost_usd": 0.0,
            "subagent_cost_usd": 0.0,
            "advisor_cost_usd": 0.0,
            "unpriced_tokens": 0,
        }
        entries = self.read(room_id)
        for entry in entries:
            if entry.get("role") != "assistant":
                continue
            u = entry.get("usage")
            if not isinstance(u, dict):
                continue
            totals["uncached_input_tokens"] += _num(u.get("input_tokens", 0))
            totals["cache_read_tokens"] += _num(u.get("cache_read_tokens", 0))
            totals["cache_creation_tokens"] += _num(u.get("cache_creation_tokens", 0))
            totals["total_output_tokens"] += _num(u.get("output_tokens", 0))
            totals["total_tool_calls"] += _num(u.get("tool_calls", 0))
            totals["main_cost_usd"] += _num(u.get("cost_usd", 0.0))
            totals["unpriced_tokens"] += _num(u.get("unpriced_tokens", 0))

        for entry in entries:
            # F7 (kdsn.218): only subagent tool entries carry cost_usd; guard on
            # the tool name so a future cost-bearing tool can't be mis-attributed
            # to subagent spend.
            if entry.get("role") != "tool" or entry.get("name") != "subagent":
                continue
            totals["subagent_cost_usd"] += _num(entry.get("cost_usd", 0.0))
            totals["unpriced_tokens"] += _num(entry.get("unpriced_tokens", 0))

        for entry in entries:
            if entry.get("role") != "system" or entry.get("event") != "advisor_consult":
                continue
            totals["advisor_cost_usd"] += _num(entry.get("cost_usd", 0.0))
            totals["unpriced_tokens"] += _num(entry.get("unpriced_tokens", 0))

        return totals

    def build_context(
        self, room_id: str, *, skip_system: bool = True,
        gc_enabled: bool | None = None,
        gc_preserve_trailing: bool = False,
    ) -> list[dict]:
        """Build LLM conversation context from JSONL entries.

        Maps session log entries to the format agent.py expects:
            - user -> {"role": "user", "content": ...}
            - assistant -> {"role": "assistant", "content": ..., ["tool_calls": ...]}
            - tool -> {"role": "tool", "tool_call_id": ..., "content": ...}
            - system -> skipped by default

        Args:
            room_id: Matrix room ID
            skip_system: If True (default), exclude system entries.
            gc_enabled: Unified GC boundary transform (workspace-kdsn.305).
                When False (default) rendering is byte-identical to the
                legacy toolstrip behavior.  When True, every entry at a
                JSONL position below the current boundary index (both GC
                and legacy markers, see context_gc.current_boundary_index)
                is reduced by the uniform rule: thinking dropped
                (thought-only assistant entries deleted atomically), tool
                outputs replaced by pointer-bearing placeholders, media
                attachments expunged to a placeholder string, and
                superseded gc_snapshot entries dropped wholesale.  Entries
                at or above the boundary render exactly as today; a
                gc_snapshot entry at or above the boundary renders verbatim
                (already framed and escaped at freeze time — never
                re-framed).  JSONL entries are NEVER modified; all
                reduction happens here at render time.

        Returns:
            List of message dicts ready for the LLM.
        """
        if gc_enabled is None:
            gc_enabled = self.gc_default
        entries = self.read(room_id)
        context = []
        # JSONL source position of each rendered message — parallel to
        # ``context``; scopes the post-pass orphan recovery to pre-boundary
        # entries when the GC transform is active.
        msg_entry_idx = []

        # Boundary discovery: max entry_index over ALL boundary markers
        # (GC + legacy toolstrip), or -1 when none.  Entries at JSONL
        # positions < boundary are reduced (uniform rule when gc_enabled,
        # legacy stripping otherwise); positions >= boundary are untouched.
        boundary_index = current_boundary_index(entries)
        # All marker indexes (ascending) — a pre-boundary tool result was
        # expunged by the FIRST boundary above it; its pointer must carry
        # THAT id, not the newest boundary (wave-2.1 A1, wonmun canary).
        _marker_list = _marker_indexes(entries) if gc_enabled else []
        if gc_enabled and boundary_index >= 0:
            # Crash-atomicity tripwire (audit): a marker without its
            # following snapshot entry means the process died mid-sequence —
            # the durable set is absent from every render until the next
            # boundary. LOUD log; render semantics unchanged.
            _marker_pos = max(
                (i for i, e in enumerate(entries)
                 if e.get("role") == "system"
                 and e.get("event") in (GC_EVENT, LEGACY_EVENT)
                 and e.get("entry_index") == boundary_index),
                default=None,
            )
            if _marker_pos is not None and not any(
                e.get("source") == GC_SNAPSHOT_SOURCE
                for e in entries[_marker_pos + 1:]
            ):
                logger.warning(
                    "gc_boundary %d has no following snapshot entry — a "
                    "previous boundary application died mid-sequence; the "
                    "durable set is missing from renders until the next "
                    "boundary", boundary_index,
                )

        # Pre-boundary call_id -> (name, input) map, harvested from the
        # pre-boundary assistant rehydration below (no second scan).  The
        # uniform transform needs it to format tool-result pointers.
        pre_boundary_calls: dict = {}

        for entry_idx, entry in enumerate(entries):
            role = entry.get("role")

            if role == "system":
                if not skip_system:
                    context.append({
                        "role": "system",
                        "content": entry.get("detail", ""),
                    })
                continue

            if role == "user":
                _source = entry.get("source")
                _user_content = entry.get("content", "")
                # Superseded snapshot (uniform transform): a gc_snapshot
                # entry strictly below the current boundary was superseded
                # by a later boundary's snapshot — dropped entirely (never
                # placeholder-ized).  Legacy mode renders these
                # verbatim-as-stored, as today.
                if gc_enabled and _source == GC_SNAPSHOT_SOURCE and entry_idx < boundary_index:
                    continue
                # Steering notes: prepend framing prefix in context (JSONL stores original).
                # Mirrors the timesense precedent: framing is context-only, not stored.
                if _source == "steer":
                    _user_content = f"[Operator steering — mid-turn guidance]: {_user_content}"
                elif _source == "spotter":
                    # Spotter advisories (design §9): JSONL stores the RAW FLAG
                    # block (audit fidelity); the advisory frame is context-only.
                    # frame_spotter_flag escapes the model-origin payload
                    # internally, so live (agent.py loop-top drain) and rebuilt
                    # (this branch) advisory bytes are identical.
                    _user_content = frame_spotter_flag(_user_content)
                elif _source not in ("reminder", "steer", GC_SNAPSHOT_SOURCE):
                    # R2-A: Escape user-origin &lt;system-reminder&gt; tags in context
                    # to prevent spoofing.  Reminder entries (source="reminder")
                    # are trusted harness content replayed verbatim.  JSONL stores
                    # raw user text; escaping is context-only (audit fidelity).
                    if isinstance(_user_content, str):
                        _user_content = escape_system_reminder_tags(_user_content)
                # GC snapshot at or above the boundary renders VERBATIM as
                # a user message: content was already framed and escaped at
                # freeze time — do not re-frame, re-escape, or apply
                # steer/spotter framing.
                _render_snapshot = (
                    gc_enabled
                    and _source == GC_SNAPSHOT_SOURCE
                    and entry_idx >= boundary_index
                )
                # Uniform transform: pre-boundary media attachments are
                # expunged to a pointer placeholder (image bytes are gone
                # from context; the model is told how to recover).
                _is_media_string = (
                    isinstance(_user_content, str)
                    and MEDIA_TAG_RE.search(_user_content) is not None
                )
                if (
                    not _render_snapshot
                    and gc_enabled
                    and entry_idx < boundary_index
                    and (
                        isinstance(_user_content, (list, tuple))
                        or _is_media_string
                    )
                ):
                    # Uniform transform: media attachments — whether the
                    # in-memory expanded list form or the JSONL-persisted
                    # [media:] tag-string form (wave-2.1, wonmun A7) — are
                    # expunged to a pointer placeholder.
                    _user_content = (
                        f"[expunged at GC boundary {boundary_index}: "
                        "media attachment — re-share or re-generate the "
                        "image if needed]"
                    )
                context.append({
                    "role": "user",
                    "content": _user_content,
                })
                msg_entry_idx.append(entry_idx)

            elif role == "assistant":
                # Uniform transform: drop pre-boundary thinking; a
                # thought-only turn (no content, no tool_calls) is deleted
                # atomically — never leave an empty shell.
                if gc_enabled and entry_idx < boundary_index:
                    _stripped_entry = strip_thinking_entry(entry)
                    if _stripped_entry is None:
                        continue
                    entry = _stripped_entry
                msg: dict = {
                    "role": "assistant",
                    "content": entry.get("content", ""),
                }
                if entry.get("tool_calls"):
                    # Rehydrate dicts back to ToolCall objects so the
                    # provider serialisation path (tc.id, tc.name, tc.input)
                    # works unchanged.
                    if boundary_index >= 0 and entry_idx < boundary_index:
                        # Before the strip boundary: replace any large string
                        # input values with compact placeholders.
                        tool_calls = []
                        for tc in entry["tool_calls"]:
                            raw_input = tc.get("input", {})
                            # Harvest (name, input) for pointer formatting —
                            # the same pass, no second scan.
                            _cid = tc.get("call_id") or tc.get("id", "")
                            if _cid:
                                pre_boundary_calls[_cid] = (
                                    tc.get("name", ""),
                                    raw_input if isinstance(raw_input, dict) else None,
                                )
                            stripped_input = {
                                k: (f"[stripped: {len(v)} chars]"
                                    if isinstance(v, str) and len(v) > 500
                                    else v)
                                for k, v in raw_input.items()
                            }
                            tool_calls.append(ToolCall(
                                id=tc.get("call_id") or tc.get("id", ""),
                                name=tc.get("name", ""),
                                input=stripped_input,
                                extra_content=tc.get("extra_content"),
                            ))
                    else:
                        tool_calls = [
                            ToolCall(
                                id=tc.get("call_id") or tc.get("id", ""),
                                name=tc.get("name", ""),
                                input=tc.get("input", {}),
                                extra_content=tc.get("extra_content"),
                            )
                            for tc in entry["tool_calls"]
                        ]
                    msg["tool_calls"] = tool_calls
                if entry.get("thinking"):
                    msg["thinking"] = entry["thinking"]
                context.append(msg)
                msg_entry_idx.append(entry_idx)

            elif role == "tool":
                # Map call_id → tool_call_id to match agent.py format.
                # Before the strip boundary, replace full output with a
                # lightweight placeholder (JSONL is never touched).
                # Legacy mode: legacy "[stripped: ...]" placeholder.  GC
                # mode: pointer-bearing placeholder carrying the original
                # call's identifying param (no matching pre-boundary call →
                # generic pointer from the tool entry's own name).
                if boundary_index >= 0 and entry_idx < boundary_index:
                    original_output = entry.get("output", "")
                    name = entry.get("name", "tool")
                    n = len(original_output) if isinstance(original_output, str) else 0
                    if gc_enabled:
                        call_id = entry.get("call_id")
                        _call_name, _call_params = (
                            pre_boundary_calls.get(call_id, (name, None))
                            if call_id
                            else (name, None)
                        )
                        _exp_b = min(
                            (m for m in _marker_list if m > entry_idx),
                            default=boundary_index,
                        )
                        output = tool_pointer(_exp_b, _call_name, _call_params, n)
                    else:
                        output = legacy_tool_placeholder(name, n)
                else:
                    output = entry.get("output", "")
                tool_entry = {
                    "role": "tool",
                    "tool_call_id": entry.get("call_id"),
                    "content": output,
                }
                if entry.get("is_error"):
                    tool_entry["is_error"] = True
                context.append(tool_entry)
                msg_entry_idx.append(entry_idx)

        # Strip orphaned tool_calls anywhere in context (crash recovery).
        # If the agent crashed mid-tool-loop, the JSONL will have an assistant
        # message with tool_calls but no corresponding tool results. Sending
        # this to the API causes errors (tool_use requires tool_result).
        # Full-scan approach to handle orphans anywhere, not just at the tail.
        #
        # GC transform scoping: with gc_enabled and an active boundary, the
        # pass covers only PRE-boundary assistant messages.  A post-boundary
        # assistant with tool_calls and no results is, by contract, a live
        # in-flight turn (apply_boundary exclude_inflight deliberately pulls
        # it back in) and must survive intact — the crash case and the
        # in-flight case are indistinguishable at render time, and the
        # in-flight ruling wins.
        # (Audit revision: the pass runs FULL-SCAN in gc mode too — a
        # trailing unresolved assistant at render time is a crash orphan,
        # and pre-boundary scoping bricked rooms on exactly the crash case
        # the pass exists to repair. EXCEPTION: the mid-turn rebuild
        # (context_gc tool path, gc_preserve_trailing=True) — the loop is
        # LIVE and its in-flight assistant must survive until its result
        # lands; scope then excludes the trailing unresolved pair.
        # call_id collisions across a boundary are handled by the pairing
        # scan below, same as legacy.)
        _orphan_scope = None
        if gc_enabled and gc_preserve_trailing:
            for _pi in range(len(msg_entry_idx) - 1, -1, -1):
                _pm = context[_pi]
                if _pm.get("role") == "assistant" and _pm.get("tool_calls"):
                    _needed = {tc.id for tc in _pm["tool_calls"]}
                    _found = any(
                        context[_j].get("role") == "tool"
                        and context[_j].get("tool_call_id") in _needed
                        for _j in range(_pi + 1, len(context))
                    )
                    if not _found:
                        _orphan_scope = set(msg_entry_idx[:_pi])
                    break
        
        # First pass: identify orphaned assistant messages
        orphaned_indices = []
        orphaned_tool_call_ids = set()
        
        for i, msg in enumerate(context):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                if _orphan_scope is not None and msg_entry_idx[i] not in _orphan_scope:
                    continue
                needed_ids = {tc.id for tc in msg["tool_calls"]}
                
                # Look for tool results AFTER this assistant message
                found_ids = set()
                for j in range(i + 1, len(context)):
                    entry = context[j]
                    if entry.get("role") == "tool" and entry.get("tool_call_id") in needed_ids:
                        found_ids.add(entry["tool_call_id"])
                
                missing_ids = needed_ids - found_ids
                if missing_ids:
                    # This assistant message is orphaned
                    logger.warning(
                        "Stripping orphaned assistant+tool_calls from context "
                        "(crash recovery): missing results for %s",
                        missing_ids,
                    )
                    orphaned_indices.append(i)
                    orphaned_tool_call_ids.update(needed_ids)
        
        # Second pass: remove orphaned messages and their partial tool results
        if orphaned_indices or orphaned_tool_call_ids:
            filtered_context = []
            for i, msg in enumerate(context):
                # Skip orphaned assistant messages
                if i in orphaned_indices:
                    continue
                
                # Skip tool results that belong to orphaned assistant
                # messages — but ONLY when the orphan pass covers this
                # source entry (pre-boundary).  A pre-boundary orphan id
                # (e.g. "c2") can collide with a legitimate post-boundary
                # result for the same live call; scoped coverage protects it.
                if (
                    _orphan_scope is None
                    or msg_entry_idx[i] in _orphan_scope
                ):
                    if (msg.get("role") == "tool" and
                        msg.get("tool_call_id") in orphaned_tool_call_ids):
                        continue
                
                # Keep all other messages
                filtered_context.append(msg)
            
            context = filtered_context

        # Third pass: reorder interleaved messages.
        # If a user message was logged between an assistant(tool_calls) and its
        # tool results (race during long tool execution), move it after the last
        # tool result for that assistant turn.  The Anthropic API requires every
        # tool_use to be immediately followed by its tool_result(s).
        changed = True
        while changed:
            changed = False
            for i, msg in enumerate(context):
                if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                    continue

                needed_ids = {tc.id for tc in msg["tool_calls"]}

                # Find the index of the last tool_result belonging to this
                # assistant turn (scanning forward from the assistant message).
                last_tool_idx = None
                for j in range(i + 1, len(context)):
                    if (context[j].get("role") == "tool" and
                            context[j].get("tool_call_id") in needed_ids):
                        last_tool_idx = j

                if last_tool_idx is None:
                    # No tool results at all — orphan stripping should have
                    # caught this, but be defensive.
                    continue

                # Collect indices of non-tool messages wedged between the
                # assistant message and the last tool result.
                interleaved = []
                for j in range(i + 1, last_tool_idx):
                    entry = context[j]
                    if entry.get("role") == "tool" and entry.get("tool_call_id") in needed_ids:
                        continue  # this is a valid tool result for this turn
                    interleaved.append(j)

                if not interleaved:
                    continue

                # Move interleaved messages to just after last_tool_idx.
                logger.warning(
                    "Reordering %d interleaved message(s) from between "
                    "assistant(tool_calls) at index %d and tool results "
                    "(session resume fix)",
                    len(interleaved),
                    i,
                )
                moved = [context[j] for j in interleaved]
                # Remove in reverse order to keep indices stable
                for j in reversed(interleaved):
                    context.pop(j)
                # Recalculate insertion point: last_tool_idx shifted by
                # the number of items removed before it.
                removed_before = sum(1 for j in interleaved if j < last_tool_idx)
                insert_at = last_tool_idx - removed_before + 1
                for k, m in enumerate(moved):
                    context.insert(insert_at + k, m)
                changed = True
                break  # restart scan since indices shifted

        return context


# ---------------------------------------------------------------------------
# persist_assistant_turn — hoisted from MatrixBot (kdsn.237 Phase 1)
# ---------------------------------------------------------------------------

def persist_assistant_turn(agent, session_log, room_id, *, content, tool_calls=None) -> None:
    """Single serializer for assistant turns. Captures content + tool_calls +
    thinking (from agent.history[-1]) + usage (from agent.last_turn_usage).
    Used by BOTH the tool-use path (_tool_intent) and the final-text paths.

    INVARIANT (RC1): This method reads thinking from agent.history(room_id)[-1].
    It is correct ONLY because the caller (agent.handle_input) always appends the
    assistant message to history immediately before this serializer runs. Any future
    change that inserts a history mutation between that append and this call will
    silently break thinking capture.
    """
    if not session_log:
        return
    # thinking: read from the current last assistant turn in history
    thinking = None
    try:
        hist = agent.history(room_id)
        if hist and hist[-1].get("role") == "assistant":
            thinking = hist[-1].get("thinking")
    except Exception:
        logger.debug("persist_assistant_turn: thinking capture failed", exc_info=True)
        thinking = None
    # usage: per-turn delta; isinstance guard so MagicMock agents (tests) -> {}
    usage = {}
    try:
        lu = agent.last_turn_usage(room_id)
        if isinstance(lu, dict):
            usage = dict(lu)
            usage["tool_calls"] = len(tool_calls or [])
    except Exception:
        logger.debug("persist_assistant_turn: usage capture failed", exc_info=True)
        usage = {}
    kwargs = dict(role="assistant", sender=session_log.agent_user_id,
                  room=room_id, event_id=None, content=content or "")
    if tool_calls is not None:
        logged = []
        for tc in tool_calls:
            entry = {"call_id": tc.id, "name": tc.name, "input": tc.input}
            # Persist opaque provider metadata (e.g. Google's
            # extra_content.google.thought_signature) so it survives
            # rehydration and can be echoed back on a later turn — see
            # ToolCall.extra_content docstring / bead workspace-kdsn.186.18.
            extra_content = getattr(tc, "extra_content", None)
            if extra_content:
                entry["extra_content"] = extra_content
            logged.append(entry)
        kwargs["tool_calls"] = logged
    if thinking:
        kwargs["thinking"] = thinking
    if usage:
        kwargs["usage"] = usage
    session_log.append(**kwargs)
