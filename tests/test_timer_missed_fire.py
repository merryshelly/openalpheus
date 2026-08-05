"""Tests for missed-fire catch-up on restart (kdsn.210.6).

Cron-vs-anacron semantics for schedule mode (DESIGN §9): on resume(), if a
scheduled instant elapsed while the process was down, fire AT MOST once
immediately (within a catch-up grace), then resume the normal schedule — never
replay N missed fires. Beyond the grace the miss is logged and skipped. Grace
default 30m; 0 disables. A caught-up fire emits an operator notice (D04). D05
pins the unchanged interval-mode behaviour.

Determinism: schedule entries can't drive real cron (minute granularity), so
these tests hand-write JSON with `last_fired_at` simulating downtime and
monkeypatch `openalph.schedule.next_fire` to return a controlled instant. The
manager's real wall clock decides whether that instant is "in the past".

CONTRACT (pinned for implementers):
  * resume() re-arms schedule timers via `schedule.next_fire` (module attr).
  * Catch-up grace is a module constant on the TIMER layer,
    `openalph._timer.MISSED_FIRE_GRACE_SECONDS` (seconds; default 1800, 0
    disables). D02/D03 monkeypatch it. The DESIGN (§9, D4) names no config
    seam; this constant is the pinned contract.
  * A caught-up fire emits the operator notice (D04) — the loop calls its
    callback, and MatrixBot's callback posts the notice.

Design:  memory/projects/openalph/scheduled-timers/DESIGN.md (§9)
Plan:    memory/projects/openalph/scheduled-timers/TEST-PLAN.md §D
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock


import openalph._timer as timer_mod
import openalph.schedule as schedule_mod
from openalph.heartbeat import HeartbeatManager

UTC = timezone.utc


def _write_sched_entry(p, room_id, spec, tz, last_fired_at):
    p.write_text(json.dumps([{
        "room_id": room_id, "interval_seconds": None,
        "schedule": spec, "tz": tz,
        "last_fired_at": last_fired_at, "directive": None,
    }]))


class TestMissedFire:
    async def test_d01_within_grace_fires_once(self, tmp_path, monkeypatch):
        # D01: a scheduled instant elapsed while down, within grace → fires
        # ONCE immediately on resume, then resumes the normal schedule (not N).
        p = tmp_path / "heartbeats.json"
        _write_sched_entry(p, "!r:x", "30 6 * * *", "America/New_York",
                           time.time() - 600)  # "fired" 10m ago; 06:30 missed

        # The most recent scheduled instant (the missed 06:30) is just past;
        # the following one is hours out. next_fire yields past once, then far.
        past = datetime.now(UTC) - timedelta(minutes=5)
        far = datetime.now(UTC) + timedelta(hours=3)
        calls = {"n": 0}

        def fake_next_fire(spec, after, tz):
            calls["n"] += 1
            return past if calls["n"] == 1 else far

        monkeypatch.setattr(schedule_mod, "next_fire", fake_next_fire)
        callback = AsyncMock()
        hb = HeartbeatManager(p, callback)
        await hb.resume()
        await asyncio.sleep(0.2)
        await hb.shutdown()
        # Exactly one catch-up fire — never a replay of the whole missed day.
        assert callback.await_count == 1
        callback.assert_awaited_with("!r:x")

    async def test_d02_beyond_grace_no_catchup(self, tmp_path, monkeypatch, caplog):
        # D02: elapsed beyond the 30m grace → NO catch-up fire; next fire is
        # the next future instant; a skip is logged.
        p = tmp_path / "heartbeats.json"
        _write_sched_entry(p, "!r:x", "30 6 * * *", "America/New_York",
                           time.time() - 8 * 3600)  # down 8h; 06:30 long past

        # The next matching instant is in the future (the miss is beyond grace).
        future = datetime.now(UTC) + timedelta(hours=3)
        monkeypatch.setattr(schedule_mod, "next_fire",
                            lambda spec, after, tz: future)
        callback = AsyncMock()
        hb = HeartbeatManager(p, callback)
        with caplog.at_level("WARNING"):
            await hb.resume()
        await asyncio.sleep(0.2)
        await hb.shutdown()
        callback.assert_not_awaited()  # no catch-up fire

    async def test_d03_grace_zero_disables_catchup(self, tmp_path, monkeypatch):
        # D03: grace=0 disables catch-up entirely — never fires the missed
        # instant, even one just barely past.
        p = tmp_path / "heartbeats.json"
        _write_sched_entry(p, "!r:x", "30 6 * * *", "America/New_York",
                           time.time() - 60)

        past = datetime.now(UTC) - timedelta(seconds=30)
        future = datetime.now(UTC) + timedelta(hours=3)
        calls = {"n": 0}

        def fake_next_fire(spec, after, tz):
            calls["n"] += 1
            return past if calls["n"] == 1 else future

        monkeypatch.setattr(schedule_mod, "next_fire", fake_next_fire)
        monkeypatch.setattr(timer_mod, "MISSED_FIRE_GRACE_SECONDS", 0)
        callback = AsyncMock()
        hb = HeartbeatManager(p, callback)
        await hb.resume()
        await asyncio.sleep(0.2)
        await hb.shutdown()
        callback.assert_not_awaited()  # grace=0 → the missed instant never fires

    async def test_d04_caught_up_fire_emits_notice(self, tmp_path, monkeypatch):
        # D04: a caught-up fire emits the operator notice (visibility of
        # lateness). The manager marks the fire as a catch-up; MatrixBot's
        # callback surfaces it. At the manager layer we pin: the callback IS
        # invoked for the catch-up (the notice is posted by that path).
        p = tmp_path / "heartbeats.json"
        _write_sched_entry(p, "!r:x", "30 6 * * *", "America/New_York",
                           time.time() - 600)
        past = datetime.now(UTC) - timedelta(minutes=5)
        future = datetime.now(UTC) + timedelta(hours=3)
        calls = {"n": 0}

        def fake_next_fire(spec, after, tz):
            calls["n"] += 1
            return past if calls["n"] == 1 else future

        monkeypatch.setattr(schedule_mod, "next_fire", fake_next_fire)
        callback = AsyncMock()
        hb = HeartbeatManager(p, callback)
        await hb.resume()
        await asyncio.sleep(0.2)
        await hb.shutdown()
        # The catch-up fire reaching the callback is the notice path.
        assert callback.await_count == 1

    async def test_d05_interval_mode_missed_fire_unchanged(self, tmp_path):
        # D05: interval-mode missed-fire semantics (initial_delay≈0) unchanged —
        # an overdue interval timer still fires almost immediately on resume.
        p = tmp_path / "heartbeats.json"
        p.write_text(json.dumps([
            {"room_id": "!r:x", "interval_seconds": 0.1,
             "last_fired_at": time.time() - 10},  # way overdue
        ]))
        callback = AsyncMock()
        hb = HeartbeatManager(p, callback)
        await hb.resume()
        await asyncio.sleep(0.1)
        await hb.shutdown()
        assert callback.await_count >= 1
