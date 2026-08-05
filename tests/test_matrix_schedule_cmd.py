"""Tests for the /… schedule slash-command handlers (kdsn.210.6).

Pins the §10 grammar: a distinct `schedule` sub-verb carrying a shlex-quoted
5-field cron spec, everything after the closing quote is the directive. The
interval form (`start 4h`, `stop`, `status`) is unchanged (E10 regression).
Floor enforcement via `min_gap` (E04/E05), steering errors on unparseable /
unquoted specs (E06/E07), and the existing heartbeat↔umbral mutual exclusion
(E08) all hold.

Real-path (per tool-management "one lesson"): a REAL MatrixBot with REAL
HeartbeatManager/UmbralManager; only the agent (LLM) and the nio client are
mocked. Commands drive the real `_handle_room_message` dispatcher.

The tz default is host-local, resolved via `openalph.schedule.DEFAULT_TZ`
(DESIGN §8, D3). Confirmations and `/… status` name the spec + tz (+ next-in,
+ directive when set).

Design:  memory/projects/openalph/scheduled-timers/DESIGN.md (§7, §8, §10)
Plan:    memory/projects/openalph/scheduled-timers/TEST-PLAN.md §E
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch


from openalph.schedule import DEFAULT_TZ
from openalph.matrix import MatrixBot
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig


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
    """MatrixBot with a mocked agent + REAL heartbeat/umbral managers."""
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


def _last_send(bot):
    return bot.send.call_args[0][1]


class TestHeartbeatScheduleCommand:
    async def test_e01_heartbeat_schedule_armed(self, tmp_path):
        # E01: /heartbeat schedule "30 6 * * 1-5" → armed; confirmation names
        # the spec + tz; heartbeats.json has the schedule entry.
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", '/heartbeat schedule "30 6 * * 1-5"'))
        await _drain(bot)
        assert bot.heartbeat.is_active("!room1:matrix.local")
        msg = _last_send(bot)
        assert "30 6 * * 1-5" in msg
        assert "America/New_York" in msg or DEFAULT_TZ in msg
        data = json.loads((tmp_path / "heartbeats.json").read_text())
        assert data[0]["schedule"] == "30 6 * * 1-5"
        assert data[0]["interval_seconds"] is None
        await bot.heartbeat.shutdown()

    async def test_e02_directive_captured_verbatim_after_quote(self, tmp_path):
        # E02: directive text after the closing quote is captured verbatim
        # (shlex parse of the quoted spec).
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local",
                             '/heartbeat schedule "30 6 * * 1-5" some directive text'))
        await _drain(bot)
        assert bot.heartbeat.directive_for("!room1:matrix.local") == "some directive text"
        await bot.heartbeat.shutdown()

    async def test_e04_heartbeat_floor_rejected(self, tmp_path):
        # E04: "/heartbeat schedule '* * * * *'" (60s < 300s floor) → rejected
        # with steering text; nothing armed.
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", '/heartbeat schedule "* * * * *"'))
        await _drain(bot)
        msg = _last_send(bot)
        assert "5m" in msg or "minimum" in msg.lower() or "300" in msg
        assert not bot.heartbeat.is_active("!room1:matrix.local")

    async def test_e07_unquoted_spec_usage_error(self, tmp_path):
        # E07: unquoted spec (`/heartbeat schedule 30 6 * * 1-5`) → clear usage
        # error (don't silently mangle the spaces).
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", "/heartbeat schedule 30 6 * * 1-5"))
        await _drain(bot)
        msg = _last_send(bot)
        assert "usage" in msg.lower() or "quote" in msg.lower() or "/heartbeat schedule" in msg
        assert not bot.heartbeat.is_active("!room1:matrix.local")


class TestUmbralScheduleCommand:
    async def test_e03_umbral_schedule_armed_with_directive(self, tmp_path):
        # E03: /umbral schedule "0 20 * * 0" Execute … → armed with directive.
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local",
                             '/umbral schedule "0 20 * * 0" Execute memory/umbral-roles/x.md'))
        await _drain(bot)
        assert bot.umbral.is_active("!room1:matrix.local")
        assert bot.umbral.directive_for("!room1:matrix.local") == "Execute memory/umbral-roles/x.md"
        data = json.loads((tmp_path / "umbral.json").read_text())
        assert data[0]["schedule"] == "0 20 * * 0"
        await bot.umbral.shutdown()

    async def test_e05_umbral_floor_rejected(self, tmp_path):
        # E05: "/umbral schedule '*/10 * * * *'" (600s < 1800s floor) → rejected.
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", '/umbral schedule "*/10 * * * *"'))
        await _drain(bot)
        msg = _last_send(bot)
        assert "30m" in msg or "minimum" in msg.lower() or "1800" in msg
        assert not bot.umbral.is_active("!room1:matrix.local")


class TestScheduleCommandErrors:
    async def test_e06_unparseable_spec_steering_error(self, tmp_path):
        # E06: unparseable spec → steering error naming the 5-field format;
        # nothing armed.
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", '/heartbeat schedule "not a cron"'))
        await _drain(bot)
        msg = _last_send(bot)
        assert "5" in msg or "cron" in msg.lower() or "field" in msg.lower() or "invalid" in msg.lower()
        assert not bot.heartbeat.is_active("!room1:matrix.local")

    async def test_e08_mutual_exclusion_with_other_timer(self, tmp_path):
        # E08: schedule on a room with the other timer active → the existing
        # "stop the other first" guard fires.
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", "/umbral start 6h umbral job"))
        await _drain(bot)
        bot.send.reset_mock()
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local",
                             '/heartbeat schedule "30 6 * * 1-5"', event_id="$evt2"))
        await _drain(bot)
        msg = _last_send(bot)
        assert "umbral" in msg.lower() and "stop" in msg.lower()
        assert not bot.heartbeat.is_active("!room1:matrix.local")
        await bot.umbral.shutdown()


class TestScheduleStatusRender:
    async def test_e09_heartbeat_status_renders_schedule(self, tmp_path):
        # E09: /heartbeat status renders spec + tz + next-in (+ directive if set).
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local",
                             '/heartbeat schedule "30 6 * * 1-5" watch the room'))
        await _drain(bot)
        bot.send.reset_mock()
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", "/heartbeat status", event_id="$evt2"))
        await _drain(bot)
        msg = _last_send(bot)
        assert "30 6 * * 1-5" in msg
        assert "America/New_York" in msg or DEFAULT_TZ in msg
        assert "next in" in msg
        assert "watch the room" in msg
        await bot.heartbeat.shutdown()

    async def test_e09_umbral_status_renders_schedule(self, tmp_path):
        # E09 (umbral): /umbral status renders spec + tz + next-in.
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", '/umbral schedule "0 20 * * 0"'))
        await _drain(bot)
        bot.send.reset_mock()
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", "/umbral status", event_id="$evt2"))
        await _drain(bot)
        msg = _last_send(bot)
        assert "0 20 * * 0" in msg
        assert "America/New_York" in msg or DEFAULT_TZ in msg
        assert "next in" in msg
        await bot.umbral.shutdown()


class TestIntervalFormRegression:
    async def test_e10_interval_start_still_works(self, tmp_path):
        # E10: interval form `/heartbeat start 4h` unchanged (regression).
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", "/heartbeat start 4h"))
        await _drain(bot)
        assert bot.heartbeat.is_active("!room1:matrix.local")
        entries = bot.heartbeat.status()
        assert entries[0].interval_seconds == 4 * 3600
        assert entries[0].schedule is None
        await bot.heartbeat.shutdown()

    async def test_e10_interval_stop_still_works(self, tmp_path):
        # E10: interval form `stop` unchanged.
        bot, agent = make_bot(tmp_path)
        room = make_room("!room1:matrix.local")
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", "/heartbeat start 4h"))
        await _drain(bot)
        bot.send.reset_mock()
        await bot._handle_room_message(
            room, make_event("@sb:matrix.local", "/heartbeat stop", event_id="$evt2"))
        await _drain(bot)
        assert "stopped" in _last_send(bot).lower()
        assert not bot.heartbeat.is_active("!room1:matrix.local")
