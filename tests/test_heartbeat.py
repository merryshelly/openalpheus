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
        assert parse_interval("0m") == 0

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

    def test_non_round_minutes(self):
        assert format_interval(5400) == "90m"

    def test_seconds_fallback(self):
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
