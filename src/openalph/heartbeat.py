"""Heartbeat module for OpenAlph.

Manages per-room recurring timers with persistence.
Communicates via async callback — has no Matrix-specific knowledge.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class HeartbeatEntry:
    room_id: str
    interval_seconds: int  # or float for test compatibility
    seconds_until_next: int
    directive: str | None = None


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
        self._intervals: dict[str, float] = {}
        self._last_fired: dict[str, float] = {}
        self._directives: dict[str, str | None] = {}

    async def start(self, room_id: str, interval_seconds: int | float, directive: str | None = None, *, _initial_delay: float | None = None) -> None:
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

        # Store interval
        self._intervals[room_id] = float(interval_seconds)
        # Robustness (audit M1): coerce a non-str directive (corrupt/hand-edited JSON or a
        # future self-control caller) to None so it can't crash escape_system_reminder_tags
        # on every fire.  Single-seam guard shared by command handler, resume(), and v2.
        if directive is not None and not isinstance(directive, str):
            logger.warning("Ignoring non-str directive for %s (%s)", room_id, type(directive).__name__)
            directive = None
        self._directives[room_id] = directive

        # Only set last_fired to now on fresh start (not resume)
        if _initial_delay is None:
            self._last_fired[room_id] = time.time()

        # Create new task
        self._tasks[room_id] = asyncio.create_task(
            self._heartbeat_loop(room_id, float(interval_seconds), initial_delay=_initial_delay)
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
        del self._intervals[room_id]
        self._last_fired.pop(room_id, None)
        self._directives.pop(room_id, None)

        # Persist to disk
        await self._persist()
        return True

    def status(self) -> list[HeartbeatEntry]:
        """Return all active heartbeats with next-fire time."""
        entries = []
        now = time.time()

        for room_id, task in self._tasks.items():
            if task.done():
                continue

            interval = self._intervals[room_id]
            last = self._last_fired.get(room_id, now)
            elapsed = now - last
            seconds_until_next = max(0, int(interval - (elapsed % interval)))

            entries.append(HeartbeatEntry(
                room_id=room_id,
                interval_seconds=int(interval),
                seconds_until_next=seconds_until_next,
                directive=self._directives.get(room_id),
            ))

        return entries

    def is_active(self, room_id: str) -> bool:
        """Return True if room has an active timer."""
        return room_id in self._tasks and not self._tasks[room_id].done()

    async def resume(self) -> None:
        """Read config from disk, start timers for all persisted heartbeats.

        Uses last_fired_at to calculate remaining time so cadence survives restarts.
        Falls back to full interval if last_fired_at is missing (old format).
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
            last_fired = entry.get("last_fired_at")
            directive = entry.get("directive")
            if room_id and interval is not None:
                initial_delay = None
                if last_fired is not None:
                    elapsed = time.time() - last_fired
                    remaining = interval - elapsed
                    if remaining <= 0:
                        # Missed fire — fire immediately (0.01s to yield event loop)
                        initial_delay = 0.01
                    else:
                        # Clamp: if clock jumped backward, remaining > interval
                        initial_delay = min(remaining, float(interval))
                    # Preserve the persisted last_fired value
                    self._last_fired[room_id] = last_fired
                await self.start(room_id, interval, directive, _initial_delay=initial_delay)

    async def shutdown(self) -> None:
        """Cancel all timers cleanly. Idempotent."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()

        # Wait for all tasks to complete cancellation
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._tasks.clear()
        self._intervals.clear()
        self._last_fired.clear()
        self._directives.clear()

    async def _heartbeat_loop(self, room_id: str, interval_seconds: float, initial_delay: float | None = None) -> None:
        """Run heartbeat loop for a room."""
        try:
            first = True
            while True:
                if first and initial_delay is not None:
                    await asyncio.sleep(initial_delay)
                else:
                    await asyncio.sleep(interval_seconds)
                first = False
                try:
                    await self.callback(room_id)
                except Exception:
                    logger.exception("Heartbeat callback error for %s", room_id)
                # Update last_fired after each successful invocation
                self._last_fired[room_id] = time.time()
                await self._persist()
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
            {
                "room_id": room_id,
                "interval_seconds": self._intervals[room_id],
                "last_fired_at": self._last_fired.get(room_id),
                "directive": self._directives.get(room_id),
            }
            for room_id in self._tasks
        ]

        # Write to temp file then rename for atomicity
        tmp_path = self.config_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data))
        os.replace(tmp_path, self.config_path)

    def directive_for(self, room_id: str) -> str | None:
        return self._directives.get(room_id)


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
