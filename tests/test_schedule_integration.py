"""Real-path integration for calendar-scheduled timers (kdsn.210.6) — THE ONE LESSON.

Per `tool-management`: construct a REAL MatrixBot with REAL
HeartbeatManager/UmbralManager (real `_timer.py`, real persistence to a tmp
path, real `_build_context_status`), mocking ONLY the nio client and the
provider. Do NOT mock the timer seam.

  F01 — end-to-end wiring: /umbral schedule → persisted → context_status for
        that room reports umbral_active + schedule + tz + umbral_next_minutes
        computed from next_fire (not an interval). The test a mocked-seam
        suite would miss.
  F02 — context_status for an INTERVAL timer is unchanged: no schedule fields,
        or nulls (the additive fields don't corrupt the interval path).
  F03 — restart simulation: bot A arms a schedule, tears down, bot B resumes
        from the same JSON (real resume()); context_status stays consistent.

context_status is assembled by callbacks.build_context_status from the manager
entries. The schedule/tz keys are pinned as `umbral_schedule` / `umbral_tz`
(and `heartbeat_schedule` / `heartbeat_tz`) per the existing naming
convention; `*_next_minutes` is named verbatim by TEST-PLAN §F.

Design:  memory/projects/openalph/scheduled-timers/DESIGN.md (§6, §8, §9)
Plan:    memory/projects/openalph/scheduled-timers/TEST-PLAN.md §F
"""

import asyncio
import json
from datetime import timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import openalph.schedule as schedule_mod
from openalph.schedule import DEFAULT_TZ
from openalph.matrix import MatrixBot
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig

UTC = timezone.utc


# --- Matrix fixtures (verbatim pattern from test_umbral_heartbeat_directives.py) ---

def make_matrix_config(**kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id="@saw:matrix.local",
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
        rooms=None,
    )
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def make_provider(key="default", type="anthropic", api_key="sk-test",
                  base_url=None, quirks=None):
    return ProviderConfig(
        key=key, type=type, api_key=api_key,
        base_url=base_url, quirks=quirks or [],
    )


def make_agent_config(workspace, **kwargs):
    defaults = dict(
        name="saw",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider(key="anthropic")},
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_room(room_id, member_count=2):
    room = MagicMock()
    room.room_id = room_id
    room.name = "Test Room"
    room.display_name = "Test Room"
    room.users = {f"@user{i}:matrix.local": MagicMock() for i in range(member_count)}
    room.named_room_name = MagicMock(return_value="Test Room")
    return room


def make_event(sender, body, event_id="$evt1"):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = 1000000
    event.source = {"content": {"msgtype": "m.text", "body": body}}
    return event


def make_bot(tmp_path):
    """REAL MatrixBot with REAL heartbeat/umbral managers; only nio + agent mocked."""
    matrix_config = make_matrix_config()
    agent_config = make_agent_config(workspace=tmp_path)

    agent = MagicMock()
    agent.config = agent_config
    agent.handle_input = AsyncMock(return_value="Agent response")
    agent.status = MagicMock(return_value={
        "name": "saw", "model": "claude-sonnet-4-20250514",
        "context_tokens": 1000, "context_max": 200000, "context_pct": 0,
        "turns": 5, "uncached_input_tokens": 5000, "cache_read_tokens": 0,
        "cache_creation_tokens": 0, "total_output_tokens": 2000,
        "total_tool_calls": 3,
    })
    agent.history = MagicMock(return_value=[])
    agent.cancel = MagicMock()
    agent.reset_room = MagicMock()
    agent.last_stop_reason = MagicMock(return_value=None)

    with patch("openalph.matrix.AsyncClient"):
        bot = MatrixBot(agent, matrix_config)

    bot.client.rooms = {}
    bot._synced = True
    bot.send = AsyncMock()
    bot.send_notice = AsyncMock()
    bot._set_typing = AsyncMock()
    return bot, agent


async def _drain(bot):
    if hasattr(bot, "_background_tasks"):
        await asyncio.gather(*bot._background_tasks, return_exceptions=True)


def _park_next_fire(monkeypatch, hours=3):
    """Deterministic next_fire at a fixed offset so seconds_until_next is stable."""
    monkeypatch.setattr(schedule_mod, "next_fire",
                        lambda spec, after, tz: after + timedelta(hours=hours))


class TestScheduleIntegration:
    async def test_f01_umbral_schedule_end_to_end_context_status(self, tmp_path, monkeypatch):
        # F01: /umbral schedule → persisted → context_status reports the
        # schedule + tz + umbral_next_minutes from next_fire (not an interval).
        _park_next_fire(monkeypatch, hours=3)
        bot, agent = make_bot(tmp_path)
        room_id = "!room1:matrix.local"
        room = make_room(room_id)
        bot.client.rooms = {room_id: room}
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", '/umbral schedule "0 20 * * 0"'))
        await _drain(bot)

        # Persisted.
        data = json.loads((tmp_path / "umbral.json").read_text())
        assert data[0]["schedule"] == "0 20 * * 0"
        assert data[0]["interval_seconds"] is None

        # context_status for that room reports the schedule wiring.
        status = bot._build_context_status(room_id)
        assert status["umbral_active"] is True
        assert status["umbral_schedule"] == "0 20 * * 0"
        assert status["umbral_tz"] == "America/New_York" or status["umbral_tz"] == DEFAULT_TZ
        # next-in computed from next_fire (~3h → ~180 min), not an interval field.
        assert status["umbral_next_minutes"] == pytest.approx(180, abs=2)
        await bot.umbral.shutdown()

    async def test_f02_interval_context_status_unchanged(self, tmp_path, monkeypatch):
        # F02: an interval umbral's context_status carries no schedule fields
        # (or nulls) — the additive fields don't corrupt the interval path.
        _park_next_fire(monkeypatch, hours=3)
        bot, agent = make_bot(tmp_path)
        room_id = "!room1:matrix.local"
        room = make_room(room_id)
        bot.client.rooms = {room_id: room}
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", "/umbral start 6h"))
        await _drain(bot)
        status = bot._build_context_status(room_id)
        assert status["umbral_active"] is True
        assert status.get("umbral_schedule") is None
        assert status.get("umbral_tz") is None
        # Interval path still reports interval + next minutes.
        assert status["umbral_interval_minutes"] == 360
        assert status["umbral_next_minutes"] is not None
        await bot.umbral.shutdown()

    async def test_f03_restart_resumes_schedule(self, tmp_path, monkeypatch):
        # F03: bot A arms a schedule, tears down; bot B (same JSON) resumes the
        # schedule timer via real resume(); context_status stays consistent.
        _park_next_fire(monkeypatch, hours=3)
        room_id = "!room1:matrix.local"

        # Bot A: arm + tear down.
        bot_a, _ = make_bot(tmp_path)
        room = make_room(room_id)
        bot_a.client.rooms = {room_id: room}
        await bot_a._handle_room_message(
            room, make_event("@sb:matrix.local", '/umbral schedule "0 20 * * 0" night job'))
        await _drain(bot_a)
        assert bot_a.umbral.is_active(room_id)
        await bot_a.umbral.shutdown()

        # Bot B: same JSON path → real resume() re-arms the schedule timer.
        bot_b, _ = make_bot(tmp_path)
        bot_b.client.rooms = {room_id: make_room(room_id)}
        await bot_b.umbral.resume()
        assert bot_b.umbral.is_active(room_id)
        status = bot_b._build_context_status(room_id)
        assert status["umbral_active"] is True
        assert status["umbral_schedule"] == "0 20 * * 0"
        assert status["umbral_next_minutes"] == pytest.approx(180, abs=2)
        assert bot_b.umbral.directive_for(room_id) == "night job"
        await bot_b.umbral.shutdown()
