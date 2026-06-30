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

from openalph.provider import ToolCall

logger = logging.getLogger(__name__)

# Max tool output before overflow to separate file (64KB)
OVERFLOW_THRESHOLD = 64 * 1024  # 64KB


class SessionLog:
    """Append-only JSONL session log for a single agent.

    One file per Matrix room. All I/O is synchronous (JSONL appends are fast
    enough that async adds complexity for no benefit). Files are fsync'd after
    each append for crash safety.
    """

    def __init__(self, workspace: str | Path, agent_user_id: str):
        """Initialize session log.

        Args:
            workspace: Agent workspace directory (sessions/ created inside it)
            agent_user_id: Matrix user ID of the agent (e.g. @watson:matrix.local)
        """
        self.workspace = Path(workspace)
        self.agent_user_id = agent_user_id
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

        # Find the current strip boundary (max entry_index from any toolstrip
        # marker), or -1 if none.
        strip_boundary = -1
        for entry in entries:
            if entry.get("role") == "system" and entry.get("event") == "toolstrip":
                idx = entry.get("entry_index", -1)
                if idx > strip_boundary:
                    strip_boundary = idx

        count = 0
        total_chars = 0
        for entry_idx, entry in enumerate(entries):
            if entry.get("role") == "tool" and entry_idx > strip_boundary:
                count += 1
                total_chars += len(entry.get("output", ""))

        return count, total_chars

    def build_context(self, room_id: str, *, skip_system: bool = True) -> list[dict]:
        """Build LLM conversation context from JSONL entries.

        Maps session log entries to the format agent.py expects:
            - user → {"role": "user", "content": ...}
            - assistant → {"role": "assistant", "content": ..., ["tool_calls": ...]}
            - tool → {"role": "tool", "tool_call_id": ..., "content": ...}
            - system → skipped by default

        Args:
            room_id: Matrix room ID
            skip_system: If True (default), exclude system entries.

        Returns:
            List of message dicts ready for the LLM.
        """
        entries = self.read(room_id)
        context = []

        # Scan for toolstrip markers to determine the strip boundary.
        # The boundary is the maximum entry_index stored in any toolstrip
        # system entry.  Entries at JSONL positions < boundary get lightweight
        # placeholder content instead of full output.  -1 means no stripping.
        strip_boundary = -1
        for entry in entries:
            if entry.get("role") == "system" and entry.get("event") == "toolstrip":
                idx = entry.get("entry_index", -1)
                if idx > strip_boundary:
                    strip_boundary = idx

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
                _user_content = entry.get("content", "")
                # Steering notes: prepend framing prefix in context (JSONL stores original).
                # Mirrors the timesense precedent: framing is context-only, not stored.
                if entry.get("source") == "steer":
                    _user_content = f"[Operator steering — mid-turn guidance]: {_user_content}"
                context.append({
                    "role": "user",
                    "content": _user_content,
                })

            elif role == "assistant":
                msg: dict = {
                    "role": "assistant",
                    "content": entry.get("content", ""),
                }
                if entry.get("tool_calls"):
                    # Rehydrate dicts back to ToolCall objects so the
                    # provider serialisation path (tc.id, tc.name, tc.input)
                    # works unchanged.
                    if strip_boundary >= 0 and entry_idx < strip_boundary:
                        # Before the strip boundary: replace any large string
                        # input values with compact placeholders.
                        tool_calls = []
                        for tc in entry["tool_calls"]:
                            raw_input = tc.get("input", {})
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
                            ))
                    else:
                        tool_calls = [
                            ToolCall(
                                id=tc.get("call_id") or tc.get("id", ""),
                                name=tc.get("name", ""),
                                input=tc.get("input", {}),
                            )
                            for tc in entry["tool_calls"]
                        ]
                    msg["tool_calls"] = tool_calls
                if entry.get("thinking"):
                    msg["thinking"] = entry["thinking"]
                context.append(msg)

            elif role == "tool":
                # Map call_id → tool_call_id to match agent.py format.
                # Before the strip boundary, replace full output with a
                # lightweight placeholder (JSONL is never touched).
                if strip_boundary >= 0 and entry_idx < strip_boundary:
                    original_output = entry.get("output", "")
                    name = entry.get("name", "tool")
                    n = len(original_output)
                    output = f"[stripped: {name} result, {n} chars]"
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

        # Strip orphaned tool_calls anywhere in context (crash recovery).
        # If the agent crashed mid-tool-loop, the JSONL will have an assistant
        # message with tool_calls but no corresponding tool results. Sending
        # this to the API causes errors (tool_use requires tool_result).
        # Full-scan approach to handle orphans anywhere, not just at the tail.
        
        # First pass: identify orphaned assistant messages
        orphaned_indices = []
        orphaned_tool_call_ids = set()
        
        for i, msg in enumerate(context):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
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
                
                # Skip tool results that belong to orphaned assistant messages
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
