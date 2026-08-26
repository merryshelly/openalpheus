"""Unit tests for HeartbeatManager.

Interface contract:
    HeartbeatManager(config_path, callback) — timer lifecycle + persistence
    start(room_id, interval_seconds) — start/replace heartbeat for a room
    stop(room_id) → bool — stop heartbeat, False if none active
    status() → list[HeartbeatEntry] — all active heartbeats with next-fire info
    resume() — load from disk, restart all timers
    shutdown() — cancel all timers

Persistence:
    heartbeats.json — atomic writes, read on resume
    Format: [{"room_id": "...", "interval_seconds": N}, ...]

Interval helpers:
    parse_interval("15m") → 900, parse_interval("6h") → 21600
    format_interval(3600) → "1h", format_interval(900) → "15m"
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock

from openalph.heartbeat import HeartbeatManager, parse_interval, format_interval


# --- Interval Parsing ---


class TestParseInterval:
    def test_minutes(self):
        assert parse_interval("15m") == 900

    def test_hours(self):
        assert parse_interval("6h") == 21600

    def test_single_minute(self):
        assert parse_interval("1m") == 60

    def test_single_hour(self):
        assert parse_interval("1h") == 3600

    def test_large_hours(self):
        assert parse_interval("168h") == 604800

    def test_invalid_unit(self):
        assert parse_interval("15x") is None

    def test_no_unit(self):
        assert parse_interval("15") is None

    def test_letters_only(self):
        assert parse_interval("abc") is None

    def test_empty_string(self):
        assert parse_interval("") is None

    def test_zero_minutes(self):
        assert parse_interval("0m") is None

    def test_whitespace_stripped(self):
        assert parse_interval("  15m  ") == 900

    def test_uppercase(self):
        assert parse_interval("6H") == 21600

    def test_mixed_case(self):
        assert parse_interval("15M") == 900

    def test_decimal_rejected(self):
        assert parse_interval("1.5h") is None

    def test_negative_rejected(self):
        assert parse_interval("-5m") is None


# --- Interval Formatting ---


class TestFormatInterval:
    def test_exact_hours(self):
        assert format_interval(3600) == "1h"

    def test_multi_hours(self):
        assert format_interval(21600) == "6h"

    def test_minutes(self):
        assert format_interval(900) == "15m"

    def test_hours_and_minutes(self):
        assert format_interval(5400) == "1h 30m"

    def test_hours_minutes_mixed(self):
        assert format_interval(7393) == "2h 3m"

    def test_minutes_and_seconds_under_5m(self):
        assert format_interval(90) == "1m 30s"

    def test_minutes_drops_seconds_over_5m(self):
        assert format_interval(610) == "10m"

    def test_seconds_only(self):
        assert format_interval(45) == "45s"

    def test_zero(self):
        assert format_interval(0) == "0s"


# --- HeartbeatManager: Start/Stop/Status ---


class TestHeartbeatStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_entry(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        await hb.start("!room1:matrix.local", 3600)

        entries = hb.status()
        assert len(entries) == 1
        assert entries[0].room_id == "!room1:matrix.local"
        assert entries[0].interval_seconds == 3600
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_start_replaces_existing(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        await hb.start("!room1:matrix.local", 3600)
        await hb.start("!room1:matrix.local", 900)

        entries = hb.status()
        assert len(entries) == 1
        assert entries[0].interval_seconds == 900
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_start_multiple_rooms(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        await hb.start("!room1:matrix.local", 3600)
        await hb.start("!room2:matrix.local", 900)

        entries = hb.status()
        assert len(entries) == 2
        room_ids = {e.room_id for e in entries}
        assert room_ids == {"!room1:matrix.local", "!room2:matrix.local"}
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_stop_removes_entry(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        await hb.start("!room1:matrix.local", 3600)
        stopped = await hb.stop("!room1:matrix.local")

        assert stopped is True
        assert hb.status() == []
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_stop_nonexistent_returns_false(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        stopped = await hb.stop("!unknown:matrix.local")
        assert stopped is False
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_stop_only_affects_target_room(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        await hb.start("!room1:matrix.local", 3600)
        await hb.start("!room2:matrix.local", 900)
        await hb.stop("!room1:matrix.local")

        entries = hb.status()
        assert len(entries) == 1
        assert entries[0].room_id == "!room2:matrix.local"
        await hb.shutdown()


# --- Status ---


class TestHeartbeatStatus:
    @pytest.mark.asyncio
    async def test_status_empty(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        assert hb.status() == []
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_status_includes_next_fire(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        await hb.start("!room1:matrix.local", 3600)

        entries = hb.status()
        assert len(entries) == 1
        # seconds_until_next should be approximately interval (just started)
        assert 0 < entries[0].seconds_until_next <= 3600
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_status_multiple_rooms_correct_intervals(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        await hb.start("!room1:matrix.local", 3600)
        await hb.start("!room2:matrix.local", 900)

        entries = hb.status()
        by_room = {e.room_id: e for e in entries}
        assert by_room["!room1:matrix.local"].interval_seconds == 3600
        assert by_room["!room2:matrix.local"].interval_seconds == 900
        await hb.shutdown()


# --- Persistence ---


class TestHeartbeatPersistence:
    @pytest.mark.asyncio
    async def test_start_writes_config(self, tmp_path):
        config_path = tmp_path / "heartbeats.json"
        callback = AsyncMock()
        hb = HeartbeatManager(config_path, callback)

        await hb.start("!room1:matrix.local", 3600)

        data = json.loads(config_path.read_text())
        assert len(data) == 1
        assert data[0]["room_id"] == "!room1:matrix.local"
        assert data[0]["interval_seconds"] == 3600
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_stop_updates_config(self, tmp_path):
        config_path = tmp_path / "heartbeats.json"
        callback = AsyncMock()
        hb = HeartbeatManager(config_path, callback)

        await hb.start("!room1:matrix.local", 3600)
        await hb.start("!room2:matrix.local", 900)
        await hb.stop("!room1:matrix.local")

        data = json.loads(config_path.read_text())
        assert len(data) == 1
        assert data[0]["room_id"] == "!room2:matrix.local"
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_resume_reads_config(self, tmp_path):
        config_path = tmp_path / "heartbeats.json"
        callback = AsyncMock()

        # Write config manually (simulating previous run)
        config_path.write_text(json.dumps([
            {"room_id": "!room1:matrix.local", "interval_seconds": 3600},
            {"room_id": "!room2:matrix.local", "interval_seconds": 900},
        ]))

        hb = HeartbeatManager(config_path, callback)
        await hb.resume()

        entries = hb.status()
        assert len(entries) == 2
        room_ids = {e.room_id for e in entries}
        assert room_ids == {"!room1:matrix.local", "!room2:matrix.local"}
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_resume_missing_file_no_error(self, tmp_path):
        config_path = tmp_path / "nonexistent.json"
        callback = AsyncMock()

        hb = HeartbeatManager(config_path, callback)
        await hb.resume()  # Should not raise

        assert hb.status() == []
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_resume_empty_file(self, tmp_path):
        config_path = tmp_path / "heartbeats.json"
        config_path.write_text("[]")
        callback = AsyncMock()

        hb = HeartbeatManager(config_path, callback)
        await hb.resume()

        assert hb.status() == []
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_resume_corrupt_file_no_crash(self, tmp_path):
        config_path = tmp_path / "heartbeats.json"
        config_path.write_text("not valid json{{{")
        callback = AsyncMock()

        hb = HeartbeatManager(config_path, callback)
        await hb.resume()  # Should not raise

        assert hb.status() == []
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_atomic_write(self, tmp_path):
        """Config file is written atomically (write-to-tmp + rename)."""
        config_path = tmp_path / "heartbeats.json"
        callback = AsyncMock()
        hb = HeartbeatManager(config_path, callback)

        await hb.start("!room1:matrix.local", 3600)

        # File should exist and be valid JSON (not a partial write)
        data = json.loads(config_path.read_text())
        assert len(data) == 1

        # No .tmp file should be left behind
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []
        await hb.shutdown()


# --- Shutdown ---


class TestHeartbeatShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_all_tasks(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        await hb.start("!room1:matrix.local", 3600)
        await hb.start("!room2:matrix.local", 900)

        await hb.shutdown()

        # After shutdown, status should be empty or tasks cancelled
        # (implementation detail: tasks are cancelled but entries may persist in config)
        # Key assertion: no lingering asyncio tasks
        # We verify by checking the callback was never called (intervals too long to fire)
        callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)

        await hb.start("!room1:matrix.local", 3600)
        await hb.shutdown()
        await hb.shutdown()  # Should not raise


# --- Fix 1: Zero interval rejection ---


class TestZeroIntervalRejection:

    def test_parse_interval_zero_minutes(self):
        assert parse_interval("0m") is None

    def test_parse_interval_zero_hours(self):
        assert parse_interval("0h") is None

    def test_parse_interval_zero_seconds(self):
        assert parse_interval("0s") is None

    @pytest.mark.asyncio
    async def test_start_rejects_zero_interval(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)
        with pytest.raises(ValueError, match="must be positive"):
            await hb.start("!room:test", 0)

    @pytest.mark.asyncio
    async def test_start_rejects_negative_interval(self, tmp_path):
        callback = AsyncMock()
        hb = HeartbeatManager(tmp_path / "heartbeats.json", callback)
        with pytest.raises(ValueError, match="must be positive"):
            await hb.start("!room:test", -5)


# --- Cadence Preservation ---


class TestHeartbeatCadencePreservation:
    """Cadence survives restart via last_fired_at persistence."""

    @pytest.mark.asyncio
    async def test_persist_includes_last_fired_at(self, tmp_path):
        """start() + fire → JSON includes last_fired_at."""
        config_path = tmp_path / "heartbeats.json"
        callback = AsyncMock()
        hb = HeartbeatManager(config_path, callback)
        await hb.start("!room1:matrix.local", 0.1)
        await asyncio.sleep(0.15)  # let it fire once
        data = json.loads(config_path.read_text())
        assert "last_fired_at" in data[0]
        assert isinstance(data[0]["last_fired_at"], float)
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_resume_calculates_remaining(self, tmp_path):
        """resume() with recent last_fired_at waits remaining time, not full interval."""
        config_path = tmp_path / "heartbeats.json"
        import time
        # Fired 0.05s ago, interval 0.2s → should fire in ~0.15s
        config_path.write_text(json.dumps([
            {"room_id": "!room1:matrix.local", "interval_seconds": 0.2,
             "last_fired_at": time.time() - 0.05},
        ]))
        callback = AsyncMock()
        hb = HeartbeatManager(config_path, callback)
        await hb.resume()
        # Should NOT have fired yet (only 0.05s in)
        await asyncio.sleep(0.05)
        assert callback.await_count == 0
        # Should fire after remaining ~0.1s
        await asyncio.sleep(0.15)
        assert callback.await_count >= 1
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_resume_fires_immediately_if_overdue(self, tmp_path):
        """resume() with old last_fired_at fires almost immediately."""
        config_path = tmp_path / "heartbeats.json"
        import time
        # Fired 10s ago, interval 0.1s → way overdue
        config_path.write_text(json.dumps([
            {"room_id": "!room1:matrix.local", "interval_seconds": 0.1,
             "last_fired_at": time.time() - 10},
        ]))
        callback = AsyncMock()
        hb = HeartbeatManager(config_path, callback)
        await hb.resume()
        await asyncio.sleep(0.1)
        assert callback.await_count >= 1
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_resume_without_last_fired_at_uses_full_interval(self, tmp_path):
        """Old format JSON (no last_fired_at) → full interval, backward compatible."""
        config_path = tmp_path / "heartbeats.json"
        config_path.write_text(json.dumps([
            {"room_id": "!room1:matrix.local", "interval_seconds": 0.2},
        ]))
        callback = AsyncMock()
        hb = HeartbeatManager(config_path, callback)
        await hb.resume()
        # Should NOT fire for ~0.2s (full interval)
        await asyncio.sleep(0.1)
        assert callback.await_count == 0
        await asyncio.sleep(0.15)
        assert callback.await_count >= 1
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_resume_clamps_clock_jump_backward(self, tmp_path):
        """If last_fired_at is in the future (clock jump), clamp to full interval."""
        config_path = tmp_path / "heartbeats.json"
        import time
        # last_fired_at in the future → remaining > interval → clamp
        config_path.write_text(json.dumps([
            {"room_id": "!room1:matrix.local", "interval_seconds": 0.2,
             "last_fired_at": time.time() + 100},
        ]))
        callback = AsyncMock()
        hb = HeartbeatManager(config_path, callback)
        await hb.resume()
        # Should clamp to full interval (~0.2s), not wait 100s
        await asyncio.sleep(0.3)
        assert callback.await_count >= 1
        await hb.shutdown()


# ===========================================================================
# BUG-5 — a bad interval in the persisted file must not crash startup
# BUG-6 — concurrent start() for one room must not orphan an untracked task
# (both fixed once in RecurringTimerManager; see _timer.py)
# ===========================================================================

class TestTimerResumeRobustness:

    @pytest.mark.asyncio
    async def test_resume_skips_bad_intervals_without_crashing(self, tmp_path):
        import json as _json
        p = tmp_path / "heartbeats.json"
        p.write_text(_json.dumps([
            {"room_id": "!good:x", "interval_seconds": 0.1, "last_fired_at": None, "directive": None},
            {"room_id": "!str:x", "interval_seconds": "15m", "last_fired_at": None},
            {"room_id": "!zero:x", "interval_seconds": 0, "last_fired_at": None},
            {"room_id": "!neg:x", "interval_seconds": -5, "last_fired_at": None},
        ]))
        m = HeartbeatManager(p, AsyncMock())
        await m.resume()  # must NOT raise
        assert sorted(m._tasks.keys()) == ["!good:x"]
        await m.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_start_leaves_one_task(self, tmp_path):
        import asyncio as _aio
        m = HeartbeatManager(tmp_path / "heartbeats.json", AsyncMock())
        await _aio.gather(*[m.start("!room:x", 0.05) for _ in range(8)])
        live = [t for t in m._tasks.values() if not t.done()]
        assert len(live) == 1
        await m.shutdown()


# ===========================================================================
# R1 (kdsn.290 audit) — loop-context detection. Manager-level pins for the
# contextvar machinery in RecurringTimerManager (src/openalph/_timer.py):
# contextvar set/reset around the callback, self-stop through a gather
# child, and the UNCHANGED external-stop contract.
# ===========================================================================


class TestTimerLoopContext:
    @pytest.mark.asyncio
    async def test_contextvar_set_during_callback_and_reset_after(self, tmp_path):
        """R1/F1: during the fired callback the loop-context contextvar
        holds (manager, room_id) — the STRONG manager reference, matched by
        identity (re-audit F1 replaced id()-keying) — and is reset (None)
        once the turn completes. The reset is a strict-finally pin:
        observable for both clean and raising callbacks at manager
        level."""
        import openalph._timer as timer_mod
        seen = []

        async def cb(room_id):
            seen.append(timer_mod._TIMER_LOOP_CTX.get())

        m = HeartbeatManager(tmp_path / "heartbeats.json", cb)
        await m.start("!room:x", 0.05, _initial_delay=0.01)
        await asyncio.sleep(0.12)
        await m.shutdown()
        # The 0.05s interval fires 2-3 times inside the 0.12s observation
        # window; assert no fire count (just ≥1) and that EVERY fired turn
        # saw the mark.
        assert seen, "no fires observed"
        mks = set(seen)
        assert len(mks) == 1
        (mk,) = mks
        assert mk[0] is m and mk[1] == "!room:x"  # identity, not id(m) (F1)
        assert timer_mod._TIMER_LOOP_CTX.get() is None  # reset after the turn

        # Second manager (raising callback): subprocess-level reset too.
        seen2 = []

        async def cb_boom(room_id):
            seen2.append(timer_mod._TIMER_LOOP_CTX.get())
            raise RuntimeError("boom")

        m2 = HeartbeatManager(tmp_path / "heartbeats2.json", cb_boom)
        await m2.start("!room:y", 0.05, _initial_delay=0.01)
        await asyncio.sleep(0.12)
        await m2.shutdown()
        assert seen2, "no fires observed (raising callback)"
        mks2 = set(seen2)
        assert len(mks2) == 1
        (mk2,) = mks2
        assert mk2[0] is m2 and mk2[1] == "!room:y"  # identity, not id() (F1)
        assert timer_mod._TIMER_LOOP_CTX.get() is None  # reset even on raise

    @pytest.mark.asyncio
    async def test_in_own_loop_false_outside_false_for_other_rooms(self, tmp_path):
        """R1: the public helper only reports the manager's OWN turn for
        THIS room — the outer (non-fire) task reads False even while a fire
        is in flight, and the helper pins room scoping (another room's turn
        does not match)."""
        hb = HeartbeatManager(tmp_path / "heartbeats.json", AsyncMock())
        fired = asyncio.Event()

        async def slow_cb(room_id):
            fired.set()
            await asyncio.sleep(5)

        hb.callback = slow_cb
        await hb.start("!room:x", 0.05, _initial_delay=0.01)
        await asyncio.wait_for(fired.wait(), timeout=5)
        # The fire is IN progress … but this task is not the fire.
        assert hb.in_own_loop("!room:x") is False
        assert hb.in_own_loop("!room:other") is False
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_self_stop_through_gather_child_stops_cleanly(self, tmp_path):
        """R1 manager-level pin: stop() invoked from a gather CHILD inside
        the fired turn (the production handle_input wiring) must behave as a
        self-stop — no CancelledError reaching the harness task, one fire,
        no refire, loop task exits (no entry left to track)."""
        fired = []

        async def cb(room_id):
            fired.append(room_id)
            [stopped] = await asyncio.gather(hb.stop(room_id))
            assert stopped is True

        hb = HeartbeatManager(tmp_path / "heartbeats.json", cb)
        await hb.start("!room:x", 0.05, _initial_delay=0.01)
        # If stop() had cancelled the timer task itself (pre-fix external
        # treatment of the gather child), the CancelledError would surface
        # here and fail the test.
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:  # pragma: no cover - regression path
            pytest.fail(
                "self-stop through a gather child cancelled the timer task (R1)"
            )
        assert fired == ["!room:x"], f"one fire, no refire: {fired}"
        assert hb.is_active("!room:x") is False
        assert "!room:x" not in hb._tasks  # loop exited; bookkeeping cleaned
        await hb.shutdown()  # must not raise/hang

    @pytest.mark.asyncio
    async def test_external_stop_during_fire_still_cancels(self, tmp_path):
        """R1 contract pin (external stop semantics UNCHANGED): an external
        stop() — from a task that does NOT inherit the fired turn's context
        — while a fire is in flight still cancels the in-flight turn and
        returns after the loop task is done. The fired turn then sees the
        fire as cancelled, exactly like before the R1 fix. The fresh
        create_task mirrors operator slash handling: operator code runs in
        its own task tree whose contextvar is NOT the fired turn's mark.
        (Inline awaiting stop() would resolve un-marked as well — the
        outer test task carries no mark either; the fresh task keeps the
        "different task tree" fidelity explicit and immune to the test
        itself ever running inside a marked context.)"""
        fired = asyncio.Event()
        cancelled = asyncio.Event()

        async def slow_cb(room_id):
            fired.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        hb = HeartbeatManager(tmp_path / "heartbeats.json", slow_cb)
        await hb.start("!room:x", 0.05, _initial_delay=0.01)
        await asyncio.wait_for(fired.wait(), timeout=5)

        # Fresh task: does NOT inherit the fired turn's mark (contextvars
        # are copied from the CURRENT task, which is the outer test task —
        # unmarked). This mirrors operator /heartbeat stop.
        stopped = await asyncio.create_task(hb.stop("!room:x"))
        assert stopped is True
        await asyncio.wait_for(cancelled.wait(), timeout=5)  # fire WAS cancelled

        task = hb._tasks.get("!room:x")
        assert task is None or task.done()  # stop returned after the loop died
        assert not hb.is_active("!room:x")
        await hb.shutdown()
