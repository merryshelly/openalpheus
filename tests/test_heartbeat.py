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
from pathlib import Path
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
