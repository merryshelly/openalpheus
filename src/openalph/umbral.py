"""Umbral module for OpenAlph.

Manages per-room recurring timers with persistence and overlap guard.
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
class UmbralEntry:
    room_id: str
    interval_seconds: int
    seconds_until_next: int
    directive: str | None = None


class UmbralManager:
    def __init__(self, config_path: Path, callback: Callable[[str], Awaitable[None]]):
        self.config_path = Path(config_path)
        self.callback = callback
        self._tasks: dict[str, asyncio.Task] = {}
        self._intervals: dict[str, float] = {}
        self._last_fired: dict[str, float] = {}
        self._processing: dict[str, bool] = {}
        self._directives: dict[str, str | None] = {}

    async def start(self, room_id: str, interval_seconds: int | float, directive: str | None = None, *, _initial_delay: float | None = None) -> None:
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")
        if room_id in self._tasks:
            self._tasks[room_id].cancel()
            try:
                await self._tasks[room_id]
            except asyncio.CancelledError:
                pass
        self._intervals[room_id] = float(interval_seconds)
        # Robustness (audit M1): a non-str directive (corrupt/hand-edited JSON, or a
        # future self-control caller) would crash escape_system_reminder_tags at inject
        # time — and in umbral that call precedes archive+wipe, so context would never
        # rotate.  Coerce to None (→ WAKE fallback) at this single seam.
        if directive is not None and not isinstance(directive, str):
            logger.warning("Ignoring non-str directive for %s (%s)", room_id, type(directive).__name__)
            directive = None
        self._directives[room_id] = directive
        # Only set last_fired to now on fresh start (not resume)
        if _initial_delay is None:
            self._last_fired[room_id] = time.time()
        self._tasks[room_id] = asyncio.create_task(
            self._umbral_loop(room_id, float(interval_seconds), initial_delay=_initial_delay)
        )
        await self._persist()

    async def stop(self, room_id: str) -> bool:
        if room_id not in self._tasks:
            return False
        task = self._tasks[room_id]
        self_stop = (asyncio.current_task() is task)
        if not self_stop:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        del self._tasks[room_id]
        del self._intervals[room_id]
        self._last_fired.pop(room_id, None)
        self._processing.pop(room_id, None)
        self._directives.pop(room_id, None)
        await self._persist()
        return True

    def status(self) -> list[UmbralEntry]:
        entries = []
        now = time.time()
        for room_id, task in self._tasks.items():
            if task.done():
                continue
            interval = self._intervals[room_id]
            last = self._last_fired.get(room_id, now)
            elapsed = now - last
            seconds_until_next = max(0, int(interval - (elapsed % interval)))
            entries.append(UmbralEntry(
                room_id=room_id,
                interval_seconds=int(interval),
                seconds_until_next=seconds_until_next,
                directive=self._directives.get(room_id),
            ))
        return entries

    def is_active(self, room_id: str) -> bool:
        return room_id in self._tasks and not self._tasks[room_id].done()

    async def resume(self) -> None:
        if not self.config_path.exists():
            return
        try:
            data = json.loads(self.config_path.read_text())
        except (json.JSONDecodeError, IOError):
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
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._intervals.clear()
        self._last_fired.clear()
        self._processing.clear()
        self._directives.clear()

    async def _umbral_loop(self, room_id: str, interval_seconds: float, initial_delay: float | None = None) -> None:
        try:
            first = True
            while True:
                if first and initial_delay is not None:
                    await asyncio.sleep(initial_delay)
                else:
                    # Drift-free: sleep only remaining time since last fire,
                    # so callback duration doesn’t push the next fire later.
                    last = self._last_fired.get(room_id, time.time() - interval_seconds)
                    elapsed = time.time() - last
                    remaining = interval_seconds - elapsed
                    await asyncio.sleep(max(remaining, 0.01))
                first = False
                if self._processing.get(room_id, False):
                    logger.warning("Umbral: skipping fire for %s — previous turn still processing", room_id)
                    continue
                self._processing[room_id] = True
                # Record fire time BEFORE callback to prevent drift
                self._last_fired[room_id] = time.time()
                await self._persist()
                try:
                    await self.callback(room_id)
                except Exception:
                    logger.exception("Umbral callback error for %s", room_id)
                finally:
                    self._processing[room_id] = False
                if room_id not in self._tasks:
                    break
        except asyncio.CancelledError:
            pass

    async def _persist(self) -> None:
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