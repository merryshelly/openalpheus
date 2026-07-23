"""Umbral timer for OpenAlph.

Per-room recurring timer with persistence and drift correction. A thin subclass
of RecurringTimerManager (see _timer.py). Communicates via an async callback and
has no Matrix-specific knowledge.

Cadence: umbral IS drift-corrected -- it sleeps only the time remaining until
the next scheduled fire, so callback duration does not push the schedule later.
It also carries an overlap guard (skip a fire while the previous is still
running); with BUG-6 fixed, two loops can no longer run for one room, so that
guard is effectively dead (ARCH-6) but is retained for behavioural compatibility.
"""

from dataclasses import dataclass

from openalph._timer import RecurringTimerManager


@dataclass
class UmbralEntry:
    room_id: str
    interval_seconds: int
    seconds_until_next: int
    directive: str | None = None


class UmbralManager(RecurringTimerManager):
    _entry_cls = UmbralEntry
    _drift_correct = True
    _overlap_guard = True
    _loop_name = "Umbral"
