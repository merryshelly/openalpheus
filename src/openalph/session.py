"""Local JSONL session persistence for OpenAlph agents.

Replaces JournalManager (Matrix rooms as persistence) with local JSONL files.
Matrix rooms remain the communication channel; session state lives on disk.

File layout:
    <workspace>/sessions/<room-id-safe>.jsonl

Where room-id-safe strips '!' and replaces ':' with '_'.
"""

import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path

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

    def _overflow_path(self, call_id: str) -> Path:
        """Path for overflow file for a tool call."""
        return self._sessions_dir / "overflow" / f"{call_id}.txt"

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
                overflow_file = overflow_dir / f"{call_id}.txt"
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

        for entry in entries:
            role = entry.get("role")

            if role == "system":
                if not skip_system:
                    context.append({
                        "role": "system",
                        "content": entry.get("detail", ""),
                    })
                continue

            if role == "user":
                context.append({
                    "role": "user",
                    "content": entry.get("content", ""),
                })

            elif role == "assistant":
                msg: dict = {
                    "role": "assistant",
                    "content": entry.get("content", ""),
                }
                if entry.get("tool_calls"):
                    msg["tool_calls"] = entry["tool_calls"]
                context.append(msg)

            elif role == "tool":
                # Map call_id → tool_call_id to match agent.py format
                context.append({
                    "role": "tool",
                    "tool_call_id": entry.get("call_id"),
                    "content": entry.get("output", ""),
                })

        return context
