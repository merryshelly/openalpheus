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
import contextvars
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

import openalph.schedule as schedule_mod

logger = logging.getLogger(__name__)

# Grace window for schedule-mode missed-fire catch-up on restart (seconds).
# A scheduled instant that elapsed while down fires immediately if it was
# within grace of now; older misses are logged and skipped. grace=0 disables
# catch-up entirely. Read via module attribute for test monkeypatching.
MISSED_FIRE_GRACE_SECONDS = 1800

# R1 loop-context detection: marks code executing INSIDE a timer's fired
# turn with (manager, room_id) — the STRONG manager reference itself, not
# an id() (re-audit F1: id()-keying was theoretically fragile to GC reuse
# of a freed manager's address by a new manager; identity matching on the
# object eliminates that class entirely). The contextvar holding a strong
# ref is acceptable and intended: it keeps the manager alive across its own
# fired turn (the bound callback already does the same), and the strict-
# finally reset clears the ref when the turn completes. Contextvars
# propagate INTO child tasks created by asyncio.gather (which is how
# Agent.handle_input runs tool calls), but NOT into unrelated tasks like
# operator slash-command handling — so an external stop still cancels an
# in-flight turn while a tool-driven self-stop through a gather child is
# detected and skips the fatal cancel+await of its own ancestor (cyclic
# cancellation → cascade of Task.cancel recursion frames / RecursionError,
# kdsn.290 audit R1).
_TIMER_LOOP_CTX: contextvars.ContextVar = contextvars.ContextVar(
    "openalph_timer_loop", default=None
)


@dataclass
class TimerEntry:
    room_id: str
    interval_seconds: int | None
    seconds_until_next: int
    directive: str | None = None
    schedule: str | None = None
    tz: str | None = None


class RecurringTimerManager:
    """Per-room recurring timers with JSON persistence.

    Subclasses set ``_entry_cls``, ``_drift_correct``, ``_overlap_guard``,
    ``_floor`` and a ``_loop_name`` used only in log lines.
    """

    _entry_cls = TimerEntry
    _drift_correct = False
    _overlap_guard = False
    _floor = 60  # default floor in seconds; subclasses override
    _loop_name = "Timer"

    def __init__(self, config_path: Path, callback: Callable[[str], Awaitable[None]]):
        self.config_path = Path(config_path)
        self.callback = callback
        self._tasks: dict[str, asyncio.Task] = {}
        self._intervals: dict[str, float] = {}
        self._schedules: dict[str, str] = {}
        self._timezones: dict[str, str] = {}
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
        # kdsn.310: the loop task serving each room, captured at loop entry.
        # apply_in_own_turn() uses it to RE-REGISTER the still-running loop
        # after a same-batch sibling stop() deleted the entry — spawning a
        # fresh task there would leave the old loop alive (its post-callback
        # exit check only sees the room back in _tasks) → two loops, one room.
        # Popped when the loop task actually exits (never by stop(), which
        # must leave it readable for the resurrect path mid-turn).
        self._fired_loop_tasks: dict[str, asyncio.Task] = {}

    async def apply_in_own_turn(
        self,
        room_id: str,
        interval_seconds: float,
        directive: str | None = None,
    ) -> None:
        """Replace the cadence/directive for ``room_id`` WITHOUT cancelling
        or spawning a loop task (kdsn.310).

        This is the manager seam for the heartbeat tool's own-turn start:
        the room's fired turn runs INSIDE the loop task, so the tool's call
        executes in a gather child of it — cancelling that ancestor kills
        the running turn (cyclic cancel, kdsn.290 audit R1), and spawning a
        replacement would race the still-running old loop into a
        double-fire. Instead:

        - entry present → mutate bookkeeping in place. The loop re-reads
          ``_intervals``/``_schedules`` at the top of every iteration, so
          the next sleep already uses the new cadence.
        - entry absent (same-batch sibling ``stop`` already removed it) →
          re-register the STILL-RUNNING loop task from ``_fired_loop_tasks``
          so the post-callback exit check finds the room live again and the
          loop continues on the new cadence.
        - no live loop at all → fall back to ``_start_common`` (fresh task),
          so a late/external apply can never strand the change.

        Interval-only by contract (the tool owns that policy): applying to
        a schedule-mode entry converts it to interval mode. Directive
        semantics match ``_start_common``: ``None`` clears. The
        decide+mutate block is fully synchronous (no awaits) so a concurrent
        gather sibling — the same-batch ``[start, stop]`` race — cannot
        interleave mid-mutation; the resulting state is always ONE coherent
        terminal state (running-with-new-params, or stopped).

        NOT a generic start replacement: unlike ``_start_common`` it takes
        no per-room lock (no cancel-and-replace to serialize), so a
        concurrent external ``start()`` is last-writer-wins — coherent, but
        don't build on the interleaving order.
        """
        # Defense-in-depth for non-tool callers (the tool path floors at
        # 300s before reaching here): a non-positive interval would arm a
        # busy-fire loop.
        if (
            not isinstance(interval_seconds, (int, float))
            or isinstance(interval_seconds, bool)
            or interval_seconds <= 0
        ):
            raise ValueError(
                f"interval_seconds must be a positive number, got "
                f"{interval_seconds!r}"
            )
        if room_id not in self._tasks:
            task = self._fired_loop_tasks.get(room_id)
            # cancelling() > 0 means an external stop() has already requested
            # cancellation of this loop — re-registering it would persist a
            # "running" entry for a task that dies at its next suspension
            # (audit qwen LOW-1). Fall through to a fresh start instead,
            # which is the coherent last-action-wins state.
            if (
                task is not None
                and not task.done()
                and task.cancelling() == 0
            ):
                self._tasks[room_id] = task
            else:
                await self._start_common(
                    room_id, interval_seconds, None, None, directive
                )
                return

        # Interval-only by contract (the heartbeat tool is interval-only;
        # cron schedules stay operator slash-command territory). Applying
        # to a schedule-mode entry converts it to interval mode — the same
        # conversion a slash `start <interval>` would perform.
        self._intervals[room_id] = interval_seconds
        self._schedules.pop(room_id, None)
        self._timezones.pop(room_id, None)

        # Same non-str coercion as _start_common (corrupt-JSON / future-
        # caller safety — the directive escaper must never see a non-str).
        if directive is not None and not isinstance(directive, str):
            logger.warning(
                "Ignoring non-str directive for %s (%s)",
                room_id, type(directive).__name__,
            )
            directive = None
        self._directives[room_id] = directive
        # Fresh-cadence semantics (mirror _start_common's fresh start): the
        # new interval counts from the apply, not from the old fire time.
        self._last_fired[room_id] = time.time()
        await self._persist()

    async def start(
        self,
        room_id: str,
        interval_seconds: int | float,
        directive: str | None = None,
        *,
        _initial_delay: float | None = None,
    ) -> None:
        """Start or replace an interval-mode timer for a room. Persists to disk."""
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")
        await self._start_common(
            room_id=room_id,
            interval_seconds=float(interval_seconds),
            schedule=None,
            tz=None,
            directive=directive,
            _initial_delay=_initial_delay,
        )

    async def start_schedule(
        self,
        room_id: str,
        schedule: str,
        tz: str | None = None,
        directive: str | None = None,
        *,
        _initial_delay: float | None = None,
    ) -> None:
        """Start or replace a schedule-mode timer for a room. Persists to disk."""
        if tz is None:
            tz = schedule_mod.DEFAULT_TZ
        await self._start_common(
            room_id=room_id,
            interval_seconds=None,
            schedule=schedule,
            tz=tz,
            directive=directive,
            _initial_delay=_initial_delay,
        )

    async def _start_common(
        self,
        room_id: str,
        interval_seconds: float | None,
        schedule: str | None,
        tz: str | None,
        directive: str | None = None,
        *,
        _initial_delay: float | None = None,
    ) -> None:
        """Shared logic for start() and start_schedule()."""
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

            # Set either interval or schedule (exactly one)
            if schedule is not None:
                self._schedules[room_id] = schedule
                self._timezones[room_id] = tz
                # Clear interval for schedule mode
                self._intervals.pop(room_id, None)
            else:
                self._intervals[room_id] = interval_seconds
                # Clear schedule for interval mode
                self._schedules.pop(room_id, None)
                self._timezones.pop(room_id, None)

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
                self._timer_loop(room_id, initial_delay=_initial_delay)
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
        # Self-stop = direct (same task: pre-tool-era slash self-stop) OR
        # tool-driven: the tool runs in a gather CHILD task, which is never
        # the timer task, but contextvars DO propagate into it — so the
        # loop-context mark set around the callback matches here even though
        # the task identity check no longer does. Only this manager's own
        # fired turn for this room matches (it holds a strict superset of
        # that turn's context); an external stop has no mark.
        self_stop = (asyncio.current_task() is task) or (
            _TIMER_LOOP_CTX.get() == (self, room_id)
        )

        if not self_stop:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        del self._tasks[room_id]
        self._intervals.pop(room_id, None)
        self._schedules.pop(room_id, None)
        self._timezones.pop(room_id, None)
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

            # Schedule mode
            if room_id in self._schedules:
                sched = self._schedules[room_id]
                tz_str = self._timezones.get(room_id, schedule_mod.DEFAULT_TZ)
                try:
                    tz = ZoneInfo(tz_str)
                    now_dt = datetime.fromtimestamp(now, tz=tz)
                    nxt = schedule_mod.next_fire(sched, now_dt, tz)
                    seconds_until_next = max(0, int((nxt - now_dt).total_seconds()))
                except Exception:
                    seconds_until_next = 0

                entries.append(self._entry_cls(
                    room_id=room_id,
                    interval_seconds=None,
                    seconds_until_next=seconds_until_next,
                    directive=self._directives.get(room_id),
                    schedule=sched,
                    tz=tz_str,
                ))
            # Interval mode
            else:
                interval = self._intervals[room_id]
                last = self._last_fired.get(room_id, now)
                elapsed = now - last
                seconds_until_next = max(0, int(interval - (elapsed % interval)))
                entries.append(self._entry_cls(
                    room_id=room_id,
                    interval_seconds=int(interval),
                    seconds_until_next=seconds_until_next,
                    directive=self._directives.get(room_id),
                    schedule=None,
                    tz=None,
                ))
        return entries

    def is_active(self, room_id: str) -> bool:
        return room_id in self._tasks and not self._tasks[room_id].done()

    def in_own_loop(self, room_id: str) -> bool:
        """True iff the current context is THIS manager's fired turn for
        ``room_id`` — including any gather child tasks spawned inside it
        (contextvars propagate into them). Unrelated tasks (e.g. an
        operator slash-command handler) return False even while a fire for
        the room is in flight.

        Exposed publicly so the heartbeat tool can refuse start-from-own-
        turn instead of letting tools import the contextvar directly."""
        # Identity match on the manager OBJECT (m is self) plus room id —
        # not id() — so a freed-and-reused address can never alias a
        # different manager's mark (F1).
        _ctx = _TIMER_LOOP_CTX.get()
        return (
            _ctx is not None and _ctx[0] is self and _ctx[1] == room_id
        )

    async def resume(self) -> list[str]:
        """Read config from disk and start timers for all persisted entries.

        Uses last_fired_at to preserve cadence across restarts, falling back to
        a full interval if it's missing.

        Returns:
            List of room_ids for which a schedule-mode catch-up fire was armed.
        """
        catchup_rooms = []

        if not self.config_path.exists():
            return catchup_rooms

        try:
            data = json.loads(self.config_path.read_text())
        except (json.JSONDecodeError, IOError):
            return catchup_rooms  # corrupt or unreadable — start fresh

        if not isinstance(data, list):
            return catchup_rooms

        for entry in data:
            if not isinstance(entry, dict):
                continue
            room_id = entry.get("room_id")
            sched = entry.get("schedule")
            interval = entry.get("interval_seconds")
            last_fired = entry.get("last_fired_at")
            directive = entry.get("directive")
            tz = entry.get("tz")

            if not room_id:
                continue

            # Exactly one of schedule or interval should be set
            if sched is not None:
                # Schedule mode
                if not isinstance(sched, str):
                    logger.warning(
                        "%s: skipping %s entry with non-str schedule %r",
                        self._loop_name, room_id, sched,
                    )
                    continue

                # Coerce tz
                if tz is None:
                    tz = schedule_mod.DEFAULT_TZ
                elif not isinstance(tz, str):
                    logger.warning(
                        "%s: skipping %s entry with non-str tz %r, using default",
                        self._loop_name, room_id, tz,
                    )
                    tz = schedule_mod.DEFAULT_TZ

                # Validate schedule spec
                try:
                    schedule_mod.validate_spec(sched)
                except schedule_mod.ScheduleError:
                    logger.warning(
                        "%s: skipping %s entry with invalid schedule spec %r",
                        self._loop_name, room_id, sched,
                    )
                    continue

                # Validate tz (try to create ZoneInfo to check validity)
                try:
                    tz_obj = ZoneInfo(tz)
                except Exception:
                    logger.warning(
                        "%s: skipping %s entry with invalid tz %r, falling back to default",
                        self._loop_name, room_id, tz,
                    )
                    tz = schedule_mod.DEFAULT_TZ
                    tz_obj = ZoneInfo(tz)

                # Validate floor: check if min_gap is below the subclass floor
                try:
                    min_gap_seconds = schedule_mod.min_gap(sched, tz_obj)
                    if min_gap_seconds < self._floor:
                        logger.warning(
                            "%s: skipping %s entry with schedule gap %ds < floor %ds",
                            self._loop_name, room_id, min_gap_seconds, self._floor,
                        )
                        continue
                except schedule_mod.ScheduleError:
                    logger.warning(
                        "%s: skipping %s entry, min_gap validation failed for spec %r",
                        self._loop_name, room_id, sched,
                    )
                    continue

                initial_delay = None
                if last_fired is not None:
                    try:
                        last_fired = float(last_fired)
                    except (TypeError, ValueError):
                        last_fired = None

                # Check for missed-fire catch-up
                if last_fired is not None:
                    self._last_fired[room_id] = last_fired
                    # Compute what next_fire would have been from last_fired_at
                    try:
                        last_dt = datetime.fromtimestamp(last_fired, tz=tz_obj)
                        # Use module attribute lookup so tests can monkeypatch
                        next_instant = schedule_mod.next_fire(sched, last_dt, tz_obj)
                        now_dt = datetime.now(tz_obj)

                        # Check if that instant is in the past (a miss occurred)
                        if next_instant <= now_dt:
                            # Compute grace window
                            import openalph._timer as timer_mod_self
                            grace = timer_mod_self.MISSED_FIRE_GRACE_SECONDS
                            time_since_miss = (now_dt - next_instant).total_seconds()

                            # If within grace (and grace > 0), arm a catch-up fire
                            if grace > 0 and time_since_miss <= grace:
                                initial_delay = 0.01  # fire immediately
                                catchup_rooms.append(room_id)
                            else:
                                # Miss is beyond grace or grace=0; skip it
                                if time_since_miss > grace:
                                    logger.warning(
                                        "%s: %s missed scheduled instant by %.1fs, beyond grace %.1fs; skipping catch-up",
                                        self._loop_name, room_id, time_since_miss, grace,
                                    )
                    except Exception:
                        logger.exception(
                            "%s: %s error computing missed-fire for schedule %r",
                            self._loop_name, room_id, sched,
                        )

                await self.start_schedule(room_id, sched, tz=tz, directive=directive, _initial_delay=initial_delay)
            elif interval is not None:
                # Interval mode
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

        return catchup_rooms

    async def shutdown(self) -> None:
        """Cancel all timers cleanly. Idempotent."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._intervals.clear()
        self._schedules.clear()
        self._timezones.clear()
        self._last_fired.clear()
        self._directives.clear()
        self._processing.clear()

    async def _timer_loop(
        self, room_id: str, initial_delay: float | None = None
    ) -> None:
        """Shared loop. Cadence is selected by ``_drift_correct``/``_overlap_guard``."""
        try:
            # kdsn.310: capture THIS task as the room's loop task so
            # apply_in_own_turn can re-register it after a same-batch
            # sibling stop() removed the entry mid-turn (a fresh task there
            # would double-fire against this still-running loop). Cleared
            # only when the loop actually exits — stop() deliberately does
            # NOT clear it, because the resurrect path reads it mid-turn.
            self._fired_loop_tasks[room_id] = asyncio.current_task()
            first = True
            while True:
                if first and initial_delay is not None:
                    await asyncio.sleep(initial_delay)
                elif room_id in self._schedules:
                    # Schedule mode: bounded hops with per-hop recomputation
                    sched = self._schedules[room_id]
                    tz_str = self._timezones.get(room_id, schedule_mod.DEFAULT_TZ)
                    tz = ZoneInfo(tz_str)

                    nxt = None
                    while True:
                        now_dt = datetime.fromtimestamp(time.time(), tz=tz)

                        # Check if we've reached the computed next fire time
                        if nxt is not None and now_dt >= nxt:
                            break

                        # Compute the next fire time
                        nxt = schedule_mod.next_fire(sched, now_dt, tz)

                        # Compute how long to sleep (bounded to 60s hops)
                        sleep_time = min(60, max(0.001, (nxt - now_dt).total_seconds()))
                        await asyncio.sleep(sleep_time)
                elif self._drift_correct:
                    # Sleep only the time remaining since the last fire, so
                    # callback duration doesn't push the next fire later.
                    interval_seconds = self._intervals[room_id]
                    last = self._last_fired.get(room_id, time.time() - interval_seconds)
                    remaining = interval_seconds - (time.time() - last)
                    await asyncio.sleep(max(remaining, 0.01))
                else:
                    interval_seconds = self._intervals[room_id]
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

                # Mark the fired turn (and any tasks spawned inside it —
                # contextvars propagate into gather children) as executing
                # inside this manager's loop for this room. stop()/start()
                # consult the mark to detect a self-stop (R1); it is reset
                # when the turn completes, whether cleanly or via exception.
                _ctx_token = _TIMER_LOOP_CTX.set((self, room_id))
                try:
                    await self.callback(room_id)
                except Exception:
                    logger.exception("%s callback error for %s", self._loop_name, room_id)
                finally:
                    _TIMER_LOOP_CTX.reset(_ctx_token)
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
        finally:
            # Identity-guarded: if this loop was superseded mid-turn (own-turn
            # apply falling back to a fresh _start_common task), the dict
            # already holds the NEW task's capture — a blind pop here would
            # erase it and blind a later resurrect.
            if self._fired_loop_tasks.get(room_id) is asyncio.current_task():
                self._fired_loop_tasks.pop(room_id, None)

    async def _persist(self) -> None:
        """Write current timers to disk atomically."""
        data = []
        for room_id in self._tasks:
            entry = {
                "room_id": room_id,
                "last_fired_at": self._last_fired.get(room_id),
                "directive": self._directives.get(room_id),
            }
            # Exactly one of interval_seconds or schedule is set
            if room_id in self._schedules:
                entry["schedule"] = self._schedules[room_id]
                entry["tz"] = self._timezones.get(room_id)
                entry["interval_seconds"] = None
            else:
                entry["interval_seconds"] = self._intervals[room_id]
                entry["schedule"] = None
            data.append(entry)

        tmp_path = self.config_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data))
        os.replace(tmp_path, self.config_path)

    def directive_for(self, room_id: str) -> str | None:
        return self._directives.get(room_id)
