"""Umbral module for OpenAlph.

Manages per-room recurring timers with persistence and overlap guard.
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
class UmbralEntry:
    room_id: str
    interval_seconds: int
    seconds_until_next: int


class UmbralManager:
    def __init__(self, config_path: Path, callback: Callable[[str], Awaitable[None]]):
        self.config_path = Path(config_path)
        self.callback = callback
        self._tasks: dict[str, asyncio.Task] = {}
        self._start_times: dict[str, float] = {}
        self._intervals: dict[str, float] = {}
        self._processing: dict[str, bool] = {}

    async def start(self, room_id: str, interval_seconds: int | float) -> None:
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")
        if room_id in self._tasks:
            self._tasks[room_id].cancel()
            try:
                await self._tasks[room_id]
            except asyncio.CancelledError:
                pass
        self._intervals[room_id] = float(interval_seconds)
        self._start_times[room_id] = asyncio.get_event_loop().time()
        self._tasks[room_id] = asyncio.create_task(
            self._umbral_loop(room_id, float(interval_seconds))
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
        del self._start_times[room_id]
        del self._intervals[room_id]
        self._processing.pop(room_id, None)
        await self._persist()
        return True

    def status(self) -> list[UmbralEntry]:
        entries = []
        now = asyncio.get_event_loop().time()
        for room_id, task in self._tasks.items():
            if task.done():
                continue
            interval = self._intervals[room_id]
            start_time = self._start_times[room_id]
            elapsed = now - start_time
            seconds_until_next = max(0, int(interval - (elapsed % interval)))
            entries.append(UmbralEntry(
                room_id=room_id,
                interval_seconds=int(interval),
                seconds_until_next=seconds_until_next,
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
            if room_id and interval is not None:
                await self.start(room_id, interval)

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._start_times.clear()
        self._intervals.clear()
        self._processing.clear()

    async def _umbral_loop(self, room_id: str, interval_seconds: float) -> None:
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                if self._processing.get(room_id, False):
                    logger.warning("Umbral: skipping fire for %s — previous turn still processing", room_id)
                    continue
                self._processing[room_id] = True
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
            {"room_id": room_id, "interval_seconds": self._intervals[room_id]}
            for room_id in self._tasks
        ]
        tmp_path = self.config_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data))
        os.replace(tmp_path, self.config_path)