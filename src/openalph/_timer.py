"""Shared base for OpenAlph's per-room recurring timers.

`HeartbeatManager` and `UmbralManager` were ~140 lines of near-character-for-
character copy-paste of each other (ARCH-1): identical
`start/stop/status/is_active/resume/shutdown/_persist/directive_for`, differing
only in the loop body. They had already begun to diverge -- heartbeat's loop was
not drift-corrected while umbral's was -- and every bug found in one existed in
the other (BUG-5 startup crash on a bad interval, BUG-6 start() race), doubling
the fix surface.

This base holds the shared machinery once. Subclasses set two flags to select
their loop cadence:

* ``_drift_correct`` -- sleep only the time REMAINING until the next scheduled
  fire (so a slow callback doesn't push the schedule later), and record the fire
  time BEFORE the callback. Umbral did this; heartbeat did not.
* ``_overlap_guard`` -- skip a fire if the previous one is still running. Umbral
  had this; heartbeat did not. (Note: with BUG-6 fixed, two loops can no longer
  run for the same room, so this guards nothing in normal operation -- see
  ARCH-6 -- but it is preserved to keep umbral's observable behaviour and tests
  unchanged.)

Communicates via an async callback and has no Matrix-specific knowledge.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass
class TimerEntry:
    room_id: str
    interval_seconds: int
    seconds_until_next: int
    directive: str | None = None


class RecurringTimerManager:
    """Per-room recurring timers with JSON persistence.

    Subclasses set ``_entry_cls``, ``_drift_correct``, ``_overlap_guard`` and a
    ``_loop_name`` used only in log lines.
    """

    _entry_cls = TimerEntry
    _drift_correct = False
    _overlap_guard = False
    _loop_name = "Timer"

    def __init__(self, config_path: Path, callback: Callable[[str], Awaitable[None]]):
        self.config_path = Path(config_path)
        self.callback = callback
        self._tasks: dict[str, asyncio.Task] = {}
        self._intervals: dict[str, float] = {}
        self._last_fired: dict[str, float] = {}
        self._directives: dict[str, str | None] = {}
        self._processing: dict[str, bool] = {}
        # BUG-6: one lock per room. start() suspends at `await self._tasks[room]`
        # (cancelling the old task) before replacing the dict entry, so two
        # concurrent same-room start() calls both created replacement loops and
        # the last write to _tasks orphaned the other -- untracked, so shutdown()
        # could not cancel it, and it fired at double cadence. The lock makes the
        # cancel-and-replace atomic per room.
        self._start_locks: dict[str, asyncio.Lock] = {}

    async def start(
        self,
        room_id: str,
        interval_seconds: int | float,
        directive: str | None = None,
        *,
        _initial_delay: float | None = None,
    ) -> None:
        """Start or replace a timer for a room. Persists to disk."""
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")

        lock = self._start_locks.get(room_id)
        if lock is None:
            lock = asyncio.Lock()
            self._start_locks[room_id] = lock

        async with lock:
            # Cancel existing task if present
            if room_id in self._tasks:
                self._tasks[room_id].cancel()
                try:
                    await self._tasks[room_id]
                except asyncio.CancelledError:
                    pass

            self._intervals[room_id] = float(interval_seconds)
            # Robustness (audit M1): coerce a non-str directive (corrupt/hand-
            # edited JSON, or a future self-control caller) to None so it can't
            # crash escape_system_reminder_tags on every fire.
            if directive is not None and not isinstance(directive, str):
                logger.warning(
                    "Ignoring non-str directive for %s (%s)",
                    room_id, type(directive).__name__,
                )
                directive = None
            self._directives[room_id] = directive

            # Only set last_fired to now on fresh start (not resume)
            if _initial_delay is None:
                self._last_fired[room_id] = time.time()

            self._tasks[room_id] = asyncio.create_task(
                self._timer_loop(room_id, float(interval_seconds), initial_delay=_initial_delay)
            )

            await self._persist()

    async def stop(self, room_id: str) -> bool:
        """Stop a timer. Returns False if none was active. Persists to disk.

        Safe to call from within the callback itself (self-stop): we skip the
        cancel/await (which would deadlock or swallow the CancelledError) and
        just clean up bookkeeping; the loop detects removal from _tasks and
        exits on its own.
        """
        if room_id not in self._tasks:
            return False

        task = self._tasks[room_id]
        self_stop = asyncio.current_task() is task

        if not self_stop:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        del self._tasks[room_id]
        del self._intervals[room_id]
        self._last_fired.pop(room_id, None)
        self._directives.pop(room_id, None)
        self._processing.pop(room_id, None)

        await self._persist()
        return True

    def status(self) -> list:
        """Return all active timers with next-fire time."""
        entries = []
        now = time.time()
        for room_id, task in self._tasks.items():
            if task.done():
                continue
            interval = self._intervals[room_id]
            last = self._last_fired.get(room_id, now)
            elapsed = now - last
            seconds_until_next = max(0, int(interval - (elapsed % interval)))
            entries.append(self._entry_cls(
                room_id=room_id,
                interval_seconds=int(interval),
                seconds_until_next=seconds_until_next,
                directive=self._directives.get(room_id),
            ))
        return entries

    def is_active(self, room_id: str) -> bool:
        return room_id in self._tasks and not self._tasks[room_id].done()

    async def resume(self) -> None:
        """Read config from disk and start timers for all persisted entries.

        Uses last_fired_at to preserve cadence across restarts, falling back to
        a full interval if it's missing.
        """
        if not self.config_path.exists():
            return

        try:
            data = json.loads(self.config_path.read_text())
        except (json.JSONDecodeError, IOError):
            return  # corrupt or unreadable — start fresh

        if not isinstance(data, list):
            return

        for entry in data:
            if not isinstance(entry, dict):
                continue
            room_id = entry.get("room_id")
            interval = entry.get("interval_seconds")
            last_fired = entry.get("last_fired_at")
            directive = entry.get("directive")

            if not room_id or interval is None:
                continue

            # BUG-5: validate the interval before using it. resume() guarded
            # against corrupt JSON, non-list, non-dict and non-str directive,
            # but NOT against a bad interval_seconds -- a hand-edited
            # {"interval_seconds": "15m"} raised TypeError and {"...": 0} raised
            # ValueError, both propagating out of resume() (called before
            # sync_forever) and preventing startup entirely: a systemd restart
            # loop from one bad value in a file the project invites editing with
            # nano. Skip and log the bad entry instead.
            try:
                interval = float(interval)
            except (TypeError, ValueError):
                logger.warning(
                    "%s: skipping %s entry with non-numeric interval_seconds %r",
                    self._loop_name, room_id, entry.get("interval_seconds"),
                )
                continue
            if interval <= 0:
                logger.warning(
                    "%s: skipping %s entry with non-positive interval_seconds %r",
                    self._loop_name, room_id, interval,
                )
                continue

            initial_delay = None
            if last_fired is not None:
                try:
                    last_fired = float(last_fired)
                except (TypeError, ValueError):
                    last_fired = None
            if last_fired is not None:
                elapsed = time.time() - last_fired
                remaining = interval - elapsed
                if remaining <= 0:
                    initial_delay = 0.01  # missed fire — fire almost immediately
                else:
                    initial_delay = min(remaining, float(interval))  # clamp clock jumps
                self._last_fired[room_id] = last_fired

            await self.start(room_id, interval, directive, _initial_delay=initial_delay)

    async def shutdown(self) -> None:
        """Cancel all timers cleanly. Idempotent."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._intervals.clear()
        self._last_fired.clear()
        self._directives.clear()
        self._processing.clear()

    async def _timer_loop(
        self, room_id: str, interval_seconds: float, initial_delay: float | None = None
    ) -> None:
        """Shared loop. Cadence is selected by ``_drift_correct``/``_overlap_guard``."""
        try:
            first = True
            while True:
                if first and initial_delay is not None:
                    await asyncio.sleep(initial_delay)
                elif self._drift_correct:
                    # Sleep only the time remaining since the last fire, so
                    # callback duration doesn't push the next fire later.
                    last = self._last_fired.get(room_id, time.time() - interval_seconds)
                    remaining = interval_seconds - (time.time() - last)
                    await asyncio.sleep(max(remaining, 0.01))
                else:
                    await asyncio.sleep(interval_seconds)
                first = False

                if self._overlap_guard and self._processing.get(room_id, False):
                    logger.warning(
                        "%s: skipping fire for %s — previous turn still processing",
                        self._loop_name, room_id,
                    )
                    continue

                if self._overlap_guard:
                    self._processing[room_id] = True

                # Drift-correcting loops record the fire time BEFORE the callback
                # (so the next sleep measures from schedule, not from completion).
                if self._drift_correct:
                    self._last_fired[room_id] = time.time()
                    await self._persist()

                try:
                    await self.callback(room_id)
                except Exception:
                    logger.exception("%s callback error for %s", self._loop_name, room_id)
                finally:
                    if self._overlap_guard:
                        self._processing[room_id] = False

                # Non-drift loops record the fire time AFTER the callback.
                if not self._drift_correct:
                    self._last_fired[room_id] = time.time()
                    await self._persist()

                # If stop() was called from inside the callback (self-stop), the
                # bookkeeping is already gone — just exit the loop.
                if room_id not in self._tasks:
                    break
        except asyncio.CancelledError:
            pass  # normal shutdown — don't propagate

    async def _persist(self) -> None:
        """Write current timers to disk atomically."""
        data = [
            {
                "room_id": room_id,
                "interval_seconds": self._intervals[room_id],
                "last_fired_at": self._last_fired.get(room_id),
                "directive": self._directives.get(room_id),
            }
            for room_id in self._tasks
        ]
        tmp_path = self.config_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data))
        os.replace(tmp_path, self.config_path)

    def directive_for(self, room_id: str) -> str | None:
        return self._directives.get(room_id)
