"""Tests for schedule-mode persistence & resume (kdsn.210.6).

Pins the §6 JSON schema for schedule entries, backward compatibility with
interval-only files (C03), mixed files (C02), and fail-closed startup on
hand-edited/corrupt schedule entries (C04–C07, BUG-5 discipline): a bad,
floor-violating, or mistyped schedule is skipped + logged, never a crash-loop.

Resume arming is real; only `openalph.schedule.next_fire` is patched (to keep
the re-armed loops parked at a far-future instant) — `validate_spec` /
`min_gap` run for real so the floor (C05) and malformed (C04) paths are
genuinely exercised.

CONTRACT (pinned for implementers):
  * Resume calls `schedule.next_fire` via module attribute lookup.
  * Floors: heartbeat min gap 300s, umbral 1800s (DESIGN §7) — enforced in
    resume() via `schedule.min_gap`.
  * Default tz when absent: host-local, resolved via a module constant
    (DESIGN §8, D3). C07 asserts fallback to that constant on a bad tz.

Design:  memory/projects/openalph/scheduled-timers/DESIGN.md (§6, §7, §8)
Plan:    memory/projects/openalph/scheduled-timers/TEST-PLAN.md §C
"""

import json
from datetime import timedelta
from unittest.mock import AsyncMock


import openalph.schedule as schedule_mod
from openalph.schedule import DEFAULT_TZ
from openalph.heartbeat import HeartbeatManager


async def _park_next_fire(monkeypatch):
    """Park every re-armed loop at a far-future instant so resume() arms but
    never fires during the assertion window."""
    monkeypatch.setattr(schedule_mod, "next_fire",
                        lambda spec, after, tz: after + timedelta(hours=3))


class TestSchedulePersistence:
    async def test_c01_round_trip(self, tmp_path, monkeypatch):
        # C01: start_schedule → JSON fields exactly as §6 → resume() re-arms
        # with the same schedule/tz/directive.
        await _park_next_fire(monkeypatch)
        p = tmp_path / "heartbeats.json"
        hb = HeartbeatManager(p, AsyncMock())
        await hb.start_schedule("!r:x", "30 6 * * 1-5",
                                tz="America/New_York", directive="watch the room")
        data = json.loads(p.read_text())
        assert data[0]["schedule"] == "30 6 * * 1-5"
        assert data[0]["tz"] == "America/New_York"
        assert data[0]["interval_seconds"] is None
        assert data[0]["directive"] == "watch the room"
        await hb.shutdown()

        # Reload via resume() → re-armed with the same schedule/tz/directive.
        hb2 = HeartbeatManager(p, AsyncMock())
        await hb2.resume()
        assert hb2.is_active("!r:x")
        entry = hb2.status()[0]
        assert entry.schedule == "30 6 * * 1-5"
        assert entry.tz == "America/New_York"
        assert hb2.directive_for("!r:x") == "watch the room"
        await hb2.shutdown()

    async def test_c02_mixed_interval_and_schedule(self, tmp_path, monkeypatch):
        # C02: one interval entry + one schedule entry in the same JSON both
        # resume correctly.
        await _park_next_fire(monkeypatch)
        p = tmp_path / "heartbeats.json"
        p.write_text(json.dumps([
            {"room_id": "!interval:x", "interval_seconds": 0.2,
             "last_fired_at": None, "directive": None},
            {"room_id": "!sched:x", "interval_seconds": None,
             "schedule": "30 6 * * 1-5", "tz": "America/New_York",
             "last_fired_at": None, "directive": "sched dir"},
        ]))
        hb = HeartbeatManager(p, AsyncMock())
        await hb.resume()
        assert hb.is_active("!interval:x")
        assert hb.is_active("!sched:x")
        by_room = {e.room_id: e for e in hb.status()}
        assert by_room["!interval:x"].schedule is None
        assert by_room["!interval:x"].interval_seconds == 0
        assert by_room["!sched:x"].schedule == "30 6 * * 1-5"
        assert hb.directive_for("!sched:x") == "sched dir"
        await hb.shutdown()

    async def test_c03_interval_only_backward_compat(self, tmp_path, monkeypatch):
        # C03: a pre-existing interval-only JSON (no schedule/tz keys) loads and
        # behaves identically — no KeyError, no schedule path taken.
        await _park_next_fire(monkeypatch)
        p = tmp_path / "heartbeats.json"
        p.write_text(json.dumps([
            {"room_id": "!old:x", "interval_seconds": 0.2},  # pre-schedule format
        ]))
        hb = HeartbeatManager(p, AsyncMock())
        await hb.resume()  # must not raise
        assert hb.is_active("!old:x")
        entry = hb.status()[0]
        assert entry.schedule is None
        assert entry.tz is None
        assert entry.interval_seconds == 0
        await hb.shutdown()

    async def test_c04_malformed_schedule_skipped_logged(self, tmp_path, monkeypatch, caplog):
        # C04 (BUG-5 discipline): a hand-edited entry with a malformed schedule
        # is skipped + logged; resume() does not raise; other entries still start.
        await _park_next_fire(monkeypatch)
        p = tmp_path / "heartbeats.json"
        p.write_text(json.dumps([
            {"room_id": "!good:x", "interval_seconds": 0.2,
             "last_fired_at": None, "directive": None},
            {"room_id": "!bad:x", "interval_seconds": None,
             "schedule": "garbage", "tz": "America/New_York"},
        ]))
        hb = HeartbeatManager(p, AsyncMock())
        with caplog.at_level("WARNING"):
            await hb.resume()  # must NOT raise
        assert hb.is_active("!good:x")
        assert not hb.is_active("!bad:x")
        assert any("bad:x" in r.getMessage() or "schedule" in r.getMessage().lower()
                   for r in caplog.records)
        await hb.shutdown()

    async def test_c05_floor_violation_skipped_on_resume(self, tmp_path, monkeypatch, caplog):
        # C05: a persisted schedule under the heartbeat floor ("* * * * *" = 60s
        # < 300s) is skipped + logged on resume, not armed.
        await _park_next_fire(monkeypatch)
        p = tmp_path / "heartbeats.json"
        p.write_text(json.dumps([
            {"room_id": "!dense:x", "interval_seconds": None,
             "schedule": "* * * * *", "tz": "America/New_York"},
        ]))
        hb = HeartbeatManager(p, AsyncMock())
        with caplog.at_level("WARNING"):
            await hb.resume()  # must NOT raise
        assert not hb.is_active("!dense:x")
        await hb.shutdown()

    async def test_c06_nonstr_schedule_or_tz_safe(self, tmp_path, monkeypatch):
        # C06: non-str schedule / non-str tz are coerced/skipped safely (mirrors
        # the directive-coercion M1 rule) — resume() must not raise.
        await _park_next_fire(monkeypatch)
        p = tmp_path / "heartbeats.json"
        p.write_text(json.dumps([
            {"room_id": "!int-sched:x", "interval_seconds": None,
             "schedule": 12345, "tz": "America/New_York"},
            {"room_id": "!int-tz:x", "interval_seconds": None,
             "schedule": "30 6 * * 1-5", "tz": 999},
        ]))
        hb = HeartbeatManager(p, AsyncMock())
        await hb.resume()  # must not raise
        # A non-str schedule cannot be honoured → skipped. A valid schedule with
        # a non-str tz must not crash startup (tz coerced/defaulted or entry skipped).
        assert not hb.is_active("!int-sched:x")
        await hb.shutdown()

    async def test_c07_invalid_tz_falls_back_to_default(self, tmp_path, monkeypatch, caplog):
        # C07: an unknown/invalid tz name → falls back to the default with a
        # log, does not raise, and the entry is still armed under the default tz.
        await _park_next_fire(monkeypatch)
        p = tmp_path / "heartbeats.json"
        p.write_text(json.dumps([
            {"room_id": "!badtz:x", "interval_seconds": None,
             "schedule": "30 6 * * 1-5", "tz": "Not/AZone"},
        ]))
        hb = HeartbeatManager(p, AsyncMock())
        with caplog.at_level("WARNING"):
            await hb.resume()  # must not raise
        assert hb.is_active("!badtz:x")
        entry = hb.status()[0]
        assert entry.tz == DEFAULT_TZ
        await hb.shutdown()
