"""Heartbeat module for OpenAlph.

Manages per-room recurring timers with persistence.
Communicates via async callback — has no Matrix-specific knowledge.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class HeartbeatEntry:
    room_id: str
    interval_seconds: int  # or float for test compatibility
    seconds_until_next: int


class HeartbeatManager:
    """Manages per-room recurring heartbeats with persistence.

    Args:
        config_path: Path to heartbeats.json
        callback: async function called with room_id when a heartbeat fires
    """

    def __init__(self, config_path: Path, callback: Callable[[str], Awaitable[None]]):
        self.config_path = Path(config_path)
        self.callback = callback
        self._tasks: dict[str, asyncio.Task] = {}
        self._start_times: dict[str, float] = {}
        self._intervals: dict[str, float] = {}

    async def start(self, room_id: str, interval_seconds: int | float) -> None:
        """Start or replace a heartbeat for a room. Persists to disk."""
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")

        # Cancel existing task if present
        if room_id in self._tasks:
            self._tasks[room_id].cancel()
            try:
                await self._tasks[room_id]
            except asyncio.CancelledError:
                pass

        # Store interval and start time
        self._intervals[room_id] = float(interval_seconds)
        self._start_times[room_id] = asyncio.get_event_loop().time()

        # Create new task
        self._tasks[room_id] = asyncio.create_task(
            self._heartbeat_loop(room_id, float(interval_seconds))
        )

        # Persist to disk
        await self._persist()

    async def stop(self, room_id: str) -> bool:
        """Stop a heartbeat. Returns False if none was active. Persists to disk.

        Safe to call from within the heartbeat callback itself (self-stop).
        In that case we skip cancel/await (which would deadlock or silently
        consume the CancelledError) and just clean up bookkeeping.  The
        _heartbeat_loop detects removal from _tasks and exits on its own.
        """
        if room_id not in self._tasks:
            return False

        task = self._tasks[room_id]
        self_stop = (asyncio.current_task() is task)

        if not self_stop:
            # External stop — cancel and wait for clean exit
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Clean up bookkeeping (loop checks _tasks to know it should exit)
        del self._tasks[room_id]
        del self._start_times[room_id]
        del self._intervals[room_id]

        # Persist to disk
        await self._persist()
        return True

    def status(self) -> list[HeartbeatEntry]:
        """Return all active heartbeats with next-fire time."""
        entries = []
        now = asyncio.get_event_loop().time()

        for room_id, task in self._tasks.items():
            if task.done():
                continue

            interval = self._intervals[room_id]
            start_time = self._start_times[room_id]
            elapsed = now - start_time
            seconds_until_next = max(0, int(interval - (elapsed % interval)))

            entries.append(HeartbeatEntry(
                room_id=room_id,
                interval_seconds=int(interval),
                seconds_until_next=seconds_until_next,
            ))

        return entries

    def is_active(self, room_id: str) -> bool:
        """Return True if room has an active timer."""
        return room_id in self._tasks and not self._tasks[room_id].done()

    async def resume(self) -> None:
        """Read config from disk, start timers for all persisted heartbeats.

        First fire is one full interval from resume (no drift tracking).
        """
        if not self.config_path.exists():
            return

        try:
            data = json.loads(self.config_path.read_text())
        except (json.JSONDecodeError, IOError):
            # Corrupt or unreadable file — start fresh
            return

        if not isinstance(data, list):
            return

        for entry in data:
            if not isinstance(entry, dict):
                continue
            room_id = entry.get("room_id")
            interval = entry.get("interval_seconds")
            if room_id and interval is not None:
                await self.start(room_id, interval)

    async def shutdown(self) -> None:
        """Cancel all timers cleanly. Idempotent."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()

        # Wait for all tasks to complete cancellation
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._tasks.clear()
        self._start_times.clear()
        self._intervals.clear()

    async def _heartbeat_loop(self, room_id: str, interval_seconds: float) -> None:
        """Run heartbeat loop for a room."""
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                try:
                    await self.callback(room_id)
                except Exception:
                    logger.exception("Heartbeat callback error for %s", room_id)
                # If stop() was called from inside the callback (self-stop),
                # the bookkeeping is already cleaned up — just exit the loop.
                if room_id not in self._tasks:
                    break
        except asyncio.CancelledError:
            # Normal shutdown — don't propagate
            pass

    async def _persist(self) -> None:
        """Write current heartbeats to disk atomically."""
        data = [
            {"room_id": room_id, "interval_seconds": self._intervals[room_id]}
            for room_id in self._tasks
        ]

        # Write to temp file then rename for atomicity
        tmp_path = self.config_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data))
        os.replace(tmp_path, self.config_path)


def parse_interval(s: str) -> int | None:
    """Parse '15m', '1h', '6h' etc. to seconds.

    Returns None on invalid input.
    Case-insensitive. Strips whitespace.
    Rejects decimals and negatives.
    """
    if not s:
        return None

    s = s.strip().lower()

    if not s:
        return None

    # Extract numeric part and unit
    num_str = ""
    unit = ""

    for i, ch in enumerate(s):
        if ch.isdigit():
            num_str += ch
        else:
            unit = s[i:]
            break

    if not num_str or not unit:
        return None

    # Reject decimal numbers (must be whole digits only)
    if "." in s:
        return None

    try:
        num = int(num_str)
    except ValueError:
        return None

    # Reject negatives and zero
    if num <= 0:
        return None

    # Parse unit
    if unit == "m":
        return num * 60
    elif unit == "h":
        return num * 3600
    elif unit == "s":
        return num
    else:
        return None


def format_interval(seconds: int) -> str:
    """Format seconds as human-readable, using mixed units when needed.

    3600 → '1h', 900 → '15m', 45 → '45s'
    7393 → '2h 3m', 3661 → '1h 1m', 90 → '1m 30s'
    """
    if seconds < 0:
        return "0s"
    if seconds == 0:
        return "0s"

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    # Only show seconds if no larger unit, or if total < 5 minutes (precision matters)
    if s and (not parts or seconds < 300):
        parts.append(f"{s}s")

    return " ".join(parts) if parts else "0s"
