"""Unit tests for UmbralManager.

Tests timer lifecycle, persistence, overlap guard, and interval validation.
Uses short intervals (0.1s-0.3s) to verify timer behavior without slow runs.
Minimum interval enforcement is tested separately from timer mechanics.
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from openalph.umbral import UmbralManager


class TestUmbralTimerLifecycle:
    """Start, stop, status, resume, shutdown."""

    @pytest.mark.asyncio
    async def test_callback_called_with_room_id(self, tmp_path):
        """Timer fires and passes correct room_id to callback."""
        callback = AsyncMock()
        um = UmbralManager(tmp_path / "umbral.json", callback)

        await um.start("!room1:matrix.local", 0.1)
        await asyncio.sleep(0.25)

        callback.assert_awaited()
        callback.assert_awaited_with("!room1:matrix.local")
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_callback_fires_repeatedly(self, tmp_path):
        """Timer fires more than once at interval."""
        callback = AsyncMock()
        um = UmbralManager(tmp_path / "umbral.json", callback)

        await um.start("!room1:matrix.local", 0.1)
        await asyncio.sleep(0.35)

        assert callback.await_count >= 2
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_stop_prevents_further_fires(self, tmp_path):
        """After stop, callback should not fire again."""
        callback = AsyncMock()
        um = UmbralManager(tmp_path / "umbral.json", callback)

        await um.start("!room1:matrix.local", 0.1)
        await asyncio.sleep(0.15)
        initial_count = callback.await_count
        await um.stop("!room1:matrix.local")
        await asyncio.sleep(0.25)

        assert callback.await_count <= initial_count + 1
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_stop_returns_false_if_not_active(self, tmp_path):
        """Stopping a non-existent timer returns False."""
        callback = AsyncMock()
        um = UmbralManager(tmp_path / "umbral.json", callback)

        result = await um.stop("!nonexistent:matrix.local")
        assert result is False
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_stop_returns_true_if_active(self, tmp_path):
        """Stopping an active timer returns True."""
        callback = AsyncMock()
        um = UmbralManager(tmp_path / "umbral.json", callback)

        await um.start("!room1:matrix.local", 0.1)
        result = await um.stop("!room1:matrix.local")
        assert result is True
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_status_shows_active_timers(self, tmp_path):
        """status() returns entries for all active timers."""
        callback = AsyncMock()
        um = UmbralManager(tmp_path / "umbral.json", callback)

        await um.start("!room1:matrix.local", 600)
        await um.start("!room2:matrix.local", 3600)

        entries = um.status()
        assert len(entries) == 2
        room_ids = {e.room_id for e in entries}
        assert room_ids == {"!room1:matrix.local", "!room2:matrix.local"}

        intervals = {e.room_id: e.interval_seconds for e in entries}
        assert intervals["!room1:matrix.local"] == 600
        assert intervals["!room2:matrix.local"] == 3600
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_status_empty_when_none_active(self, tmp_path):
        """status() returns empty list when no timers running."""
        callback = AsyncMock()
        um = UmbralManager(tmp_path / "umbral.json", callback)

        entries = um.status()
        assert entries == []
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_is_active(self, tmp_path):
        """is_active returns True only for rooms with running timers."""
        callback = AsyncMock()
        um = UmbralManager(tmp_path / "umbral.json", callback)

        assert um.is_active("!room1:matrix.local") is False
        await um.start("!room1:matrix.local", 0.5)
        assert um.is_active("!room1:matrix.local") is True
        await um.stop("!room1:matrix.local")
        assert um.is_active("!room1:matrix.local") is False
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_start_replaces_existing(self, tmp_path):
        """Starting a timer in a room that already has one replaces it."""
        callback = AsyncMock()
        um = UmbralManager(tmp_path / "umbral.json", callback)

        await um.start("!room1:matrix.local", 600)
        await um.start("!room1:matrix.local", 1800)

        entries = um.status()
        assert len(entries) == 1
        assert entries[0].interval_seconds == 1800
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_all(self, tmp_path):
        """shutdown() cancels all timers cleanly."""
        callback = AsyncMock()
        um = UmbralManager(tmp_path / "umbral.json", callback)

        await um.start("!room1:matrix.local", 0.1)
        await um.start("!room2:matrix.local", 0.1)
        await um.shutdown()

        assert um.status() == []

    @pytest.mark.asyncio
    async def test_multiple_rooms_fire_independently(self, tmp_path):
        """Multiple rooms have independent timers."""
        fired_rooms = []

        async def track(room_id):
            fired_rooms.append(room_id)

        um = UmbralManager(tmp_path / "umbral.json", track)

        await um.start("!room1:matrix.local", 0.1)
        await um.start("!room2:matrix.local", 0.2)
        await asyncio.sleep(0.35)

        room1_count = fired_rooms.count("!room1:matrix.local")
        room2_count = fired_rooms.count("!room2:matrix.local")
        assert room1_count >= 2
        assert room2_count >= 1
        assert room1_count > room2_count
        await um.shutdown()


class TestUmbralPersistence:
    """JSON persistence and resume."""

    @pytest.mark.asyncio
    async def test_persist_on_start(self, tmp_path):
        """start() writes to umbral.json."""
        config_path = tmp_path / "umbral.json"
        callback = AsyncMock()
        um = UmbralManager(config_path, callback)

        await um.start("!room1:matrix.local", 600)

        data = json.loads(config_path.read_text())
        assert len(data) == 1
        assert data[0]["room_id"] == "!room1:matrix.local"
        assert data[0]["interval_seconds"] == 600
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_persist_on_stop(self, tmp_path):
        """stop() removes entry from umbral.json."""
        config_path = tmp_path / "umbral.json"
        callback = AsyncMock()
        um = UmbralManager(config_path, callback)

        await um.start("!room1:matrix.local", 600)
        await um.stop("!room1:matrix.local")

        data = json.loads(config_path.read_text())
        assert data == []
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_resume_starts_timers(self, tmp_path):
        """resume() reads config and starts timers that fire."""
        config_path = tmp_path / "umbral.json"
        config_path.write_text(json.dumps([
            {"room_id": "!room1:matrix.local", "interval_seconds": 0.1},
        ]))

        callback = AsyncMock()
        um = UmbralManager(config_path, callback)
        await um.resume()
        await asyncio.sleep(0.25)

        callback.assert_awaited_with("!room1:matrix.local")
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_resume_ignores_corrupt_file(self, tmp_path):
        """resume() handles corrupt JSON gracefully."""
        config_path = tmp_path / "umbral.json"
        config_path.write_text("not json {{{")

        callback = AsyncMock()
        um = UmbralManager(config_path, callback)
        await um.resume()  # should not raise

        assert um.status() == []
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_resume_ignores_missing_file(self, tmp_path):
        """resume() handles missing file gracefully."""
        config_path = tmp_path / "umbral.json"  # does not exist

        callback = AsyncMock()
        um = UmbralManager(config_path, callback)
        await um.resume()  # should not raise

        assert um.status() == []
        await um.shutdown()


class TestUmbralOverlapGuard:
    """Turn overlap prevention."""

    @pytest.mark.asyncio
    async def test_overlapping_turn_skipped(self, tmp_path):
        """If callback is still running when timer fires again, skip the fire."""
        call_count = 0
        overlap_event = asyncio.Event()

        async def slow_callback(room_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call blocks long enough for second timer to fire
                await asyncio.sleep(0.3)
            overlap_event.set()

        um = UmbralManager(tmp_path / "umbral.json", slow_callback)
        await um.start("!room1:matrix.local", 0.1)

        # Wait enough for first call to block + second fire to be skipped
        # + first to complete + third to fire normally
        await asyncio.sleep(0.55)

        # First call takes 0.3s. Timer fires at 0.1, 0.2, 0.3, 0.4, 0.5.
        # Fire at 0.1: starts, blocks.
        # Fire at 0.2: skipped (processing=True).
        # Fire at ~0.3: skipped (still processing).
        # First call finishes at ~0.4.
        # Fire at ~0.4: starts, completes quickly.
        # So we expect 2 completed calls, not 4-5.
        assert call_count <= 3
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_callback_error_clears_processing_flag(self, tmp_path):
        """After callback raises, processing flag is cleared for next fire."""
        call_count = 0

        async def failing_callback(room_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")

        um = UmbralManager(tmp_path / "umbral.json", failing_callback)
        await um.start("!room1:matrix.local", 0.1)
        await asyncio.sleep(0.25)

        # Should fire more than once — error on first shouldn't block second
        assert call_count >= 2
        await um.shutdown()


class TestUmbralSelfStop:
    """Callback can stop its own timer (same pattern as HeartbeatManager)."""

    @pytest.mark.asyncio
    async def test_self_stop_from_callback(self, tmp_path):
        """Callback calling um.stop(room_id) on itself doesn't deadlock."""
        um = None

        async def self_stopping_callback(room_id):
            await um.stop(room_id)

        um = UmbralManager(tmp_path / "umbral.json", self_stopping_callback)
        await um.start("!room1:matrix.local", 0.1)
        await asyncio.sleep(0.25)

        # Timer should have stopped itself after first fire
        assert not um.is_active("!room1:matrix.local")
        await um.shutdown()


class TestUmbralIntervalValidation:
    """Minimum interval enforcement."""

    @pytest.mark.asyncio
    async def test_zero_interval_raises(self, tmp_path):
        callback = AsyncMock()
        um = UmbralManager(tmp_path / "umbral.json", callback)

        with pytest.raises(ValueError):
            await um.start("!room1:matrix.local", 0)
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_negative_interval_raises(self, tmp_path):
        callback = AsyncMock()
        um = UmbralManager(tmp_path / "umbral.json", callback)

        with pytest.raises(ValueError):
            await um.start("!room1:matrix.local", -100)
        await um.shutdown()


class TestUmbralCadencePreservation:
    """Cadence survives restart via last_fired_at persistence."""

    @pytest.mark.asyncio
    async def test_persist_includes_last_fired_at(self, tmp_path):
        """start() + fire → JSON includes last_fired_at."""
        config_path = tmp_path / "umbral.json"
        callback = AsyncMock()
        um = UmbralManager(config_path, callback)
        await um.start("!room1:matrix.local", 0.1)
        await asyncio.sleep(0.15)  # let it fire once
        data = json.loads(config_path.read_text())
        assert "last_fired_at" in data[0]
        assert isinstance(data[0]["last_fired_at"], float)
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_resume_calculates_remaining(self, tmp_path):
        """resume() with recent last_fired_at waits remaining time, not full interval."""
        config_path = tmp_path / "umbral.json"
        import time
        # Fired 0.05s ago, interval 0.2s → should fire in ~0.15s
        config_path.write_text(json.dumps([
            {"room_id": "!room1:matrix.local", "interval_seconds": 0.2,
             "last_fired_at": time.time() - 0.05},
        ]))
        callback = AsyncMock()
        um = UmbralManager(config_path, callback)
        await um.resume()
        # Should NOT have fired yet (only 0.05s in)
        await asyncio.sleep(0.05)
        assert callback.await_count == 0
        # Should fire after remaining ~0.1s
        await asyncio.sleep(0.15)
        assert callback.await_count >= 1
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_resume_fires_immediately_if_overdue(self, tmp_path):
        """resume() with old last_fired_at fires almost immediately."""
        config_path = tmp_path / "umbral.json"
        import time
        # Fired 10s ago, interval 0.1s → way overdue
        config_path.write_text(json.dumps([
            {"room_id": "!room1:matrix.local", "interval_seconds": 0.1,
             "last_fired_at": time.time() - 10},
        ]))
        callback = AsyncMock()
        um = UmbralManager(config_path, callback)
        await um.resume()
        await asyncio.sleep(0.1)
        assert callback.await_count >= 1
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_resume_without_last_fired_at_uses_full_interval(self, tmp_path):
        """Old format JSON (no last_fired_at) → full interval, backward compatible."""
        config_path = tmp_path / "umbral.json"
        config_path.write_text(json.dumps([
            {"room_id": "!room1:matrix.local", "interval_seconds": 0.2},
        ]))
        callback = AsyncMock()
        um = UmbralManager(config_path, callback)
        await um.resume()
        # Should NOT fire for ~0.2s (full interval)
        await asyncio.sleep(0.1)
        assert callback.await_count == 0
        await asyncio.sleep(0.15)
        assert callback.await_count >= 1
        await um.shutdown()

    @pytest.mark.asyncio
    async def test_resume_clamps_clock_jump_backward(self, tmp_path):
        """If last_fired_at is in the future (clock jump), clamp to full interval."""
        config_path = tmp_path / "umbral.json"
        import time
        # last_fired_at in the future → remaining > interval → clamp
        config_path.write_text(json.dumps([
            {"room_id": "!room1:matrix.local", "interval_seconds": 0.2,
             "last_fired_at": time.time() + 100},
        ]))
        callback = AsyncMock()
        um = UmbralManager(config_path, callback)
        await um.resume()
        # Should clamp to full interval (~0.2s), not wait 100s
        await asyncio.sleep(0.3)
        assert callback.await_count >= 1
        await um.shutdown()


class TestUmbralDriftPrevention:
    """Verify callback duration does not cause timer drift."""

    @pytest.mark.asyncio
    async def test_no_drift_from_callback_duration(self, tmp_path):
        """Fire-to-fire interval stays close to requested interval despite slow callback."""
        import time as _time
        fire_times = []

        async def timed_callback(room_id):
            fire_times.append(_time.time())
            await asyncio.sleep(0.05)  # simulate 50ms processing

        um = UmbralManager(tmp_path / "umbral.json", timed_callback)
        await um.start("!room1:matrix.local", 0.15)
        await asyncio.sleep(0.55)

        assert len(fire_times) >= 3, f"Expected >=3 fires, got {len(fire_times)}"
        for i in range(1, len(fire_times)):
            delta = fire_times[i] - fire_times[i - 1]
            # Without fix: delta ~0.20 (0.15 interval + 0.05 callback)
            # With fix: delta ~0.15 (callback absorbed into interval)
            assert delta < 0.18, (
                f"Drift detected: fire interval {i} was {delta:.3f}s "
                f"(expected ~0.15s, threshold 0.18s)"
            )

        await um.shutdown()


# ===========================================================================
# BUG-5 — a bad interval in the persisted file must not crash startup
# BUG-6 — concurrent start() for one room must not orphan an untracked task
# (both fixed once in RecurringTimerManager; see _timer.py)
# ===========================================================================

class TestTimerResumeRobustness:

    @pytest.mark.asyncio
    async def test_resume_skips_bad_intervals_without_crashing(self, tmp_path):
        import json as _json
        p = tmp_path / "umbral.json"
        p.write_text(_json.dumps([
            {"room_id": "!good:x", "interval_seconds": 0.1, "last_fired_at": None, "directive": None},
            {"room_id": "!str:x", "interval_seconds": "15m", "last_fired_at": None},
            {"room_id": "!zero:x", "interval_seconds": 0, "last_fired_at": None},
            {"room_id": "!neg:x", "interval_seconds": -5, "last_fired_at": None},
        ]))
        m = UmbralManager(p, AsyncMock())
        await m.resume()  # must NOT raise
        assert sorted(m._tasks.keys()) == ["!good:x"]
        await m.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_start_leaves_one_task(self, tmp_path):
        import asyncio as _aio
        m = UmbralManager(tmp_path / "umbral.json", AsyncMock())
        await _aio.gather(*[m.start("!room:x", 0.05) for _ in range(8)])
        live = [t for t in m._tasks.values() if not t.done()]
        assert len(live) == 1
        await m.shutdown()
