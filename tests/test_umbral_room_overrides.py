"""Regression contract for room overrides across umbral + process restart.

workspace-kdsn.370: umbral archive+wipe must carry effective explicit room
settings into the reseeded active JSONL. Tests use real Agent, SessionLog,
MatrixBot._inject_umbral, and MatrixBot._activate_room. Only Matrix/network
and the model turn are mocked; all persistence is real under tmp_path.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openalph.agent import Agent
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.matrix import MatrixBot
from openalph.session import SessionLog

ROOM = "!umbral-overrides:matrix.local"
AGENT_USER = "@saw:matrix.local"
DEFAULT = "blackwell/qwen38-27b-fp8"
ALT_A = "syn2dsv41f"
ALT_B = "syndsv41f"
MODEL_A = "synthetic2/hf:deepseek-ai/DeepSeek-V4.1-Flash-1017"
MODEL_B = "synthetic/hf:deepseek-ai/DeepSeek-V4.1-Flash-1017"
OVERRIDE_EVENTS = {
    "model_override", "effort_override", "thinking_override",
    "cache_ttl_override", "timesense_override", "quiet_override",
}


def _config(workspace: Path) -> AgentConfig:
    cfg = AgentConfig(
        name="saw-umbral-override-test",
        default_model=DEFAULT,
        max_tokens=8192,
        providers={
            "blackwell": ProviderConfig(
                key="blackwell", type="openai", api_key="none",
                base_url="http://blackwell.invalid/v1", quirks=[],
            ),
            "synthetic": ProviderConfig(
                key="synthetic", type="openai", api_key="test-only-a",
                base_url="https://synthetic.invalid/v1", quirks=[],
            ),
            "synthetic2": ProviderConfig(
                key="synthetic2", type="openai", api_key="test-only-b",
                base_url="https://synthetic2.invalid/v1", quirks=[],
            ),
        },
        workspace=workspace,
        max_iterations=5,
        truncation_limit=50000,
        model_max_tokens=262144,
        matrix=None,
        model_aliases={ALT_A: MODEL_A, ALT_B: MODEL_B},
    )
    cfg.turn_stall_timeout_seconds = 0
    return cfg


def _bot(workspace: Path) -> tuple[MatrixBot, Agent, SessionLog]:
    agent = Agent(_config(workspace))
    sl = SessionLog(workspace, AGENT_USER)
    bot = MatrixBot.__new__(MatrixBot)
    bot.config = MatrixConfig(
        homeserver="https://matrix.invalid", user_id=AGENT_USER,
        device_id="TEST", password="not-used", access_token=None,
        context_reserve=16384, sync_timeout=30000,
        retry_base=1, retry_max=10, rooms=None,
    )
    bot.agent = agent
    bot.session_log = sl
    bot.umbral = None
    bot.client = None
    bot._active_rooms = set()
    bot._active_turns = set()
    bot._known_event_ids = {}
    bot._room_effort = {}
    bot._room_cache_ttl = {}
    bot._room_timesense = {}
    bot.send = AsyncMock()
    bot.send_notice = AsyncMock()
    bot._set_typing = AsyncMock()
    bot._run_heartbeat_turn = AsyncMock(return_value=None)
    return bot, agent, sl


def _append(sl: SessionLog, event: str, detail: str) -> None:
    sl.append(
        role="system", sender="@serious_business:matrix.local", room=ROOM,
        event_id=None, event=event, detail=detail,
    )


def _switch_model(agent: Agent, sl: SessionLog, model: str) -> None:
    error = agent.switch_model(model, ROOM)
    assert error is None, error
    _append(sl, "model_override", model)


def _events(sl: SessionLog) -> list[tuple[str, str]]:
    return [
        (e.get("event"), e.get("detail"))
        for e in sl.read(ROOM)
        if e.get("event") in OVERRIDE_EVENTS | {"umbral_reset"}
    ]


def _event_details(sl: SessionLog, event: str) -> list[str]:
    return [detail for name, detail in _events(sl) if name == event]


@pytest.mark.asyncio
async def test_plain_restart_control_restores_model(tmp_path):
    bot1, agent1, sl = _bot(tmp_path)
    await bot1._activate_room(ROOM)
    _switch_model(agent1, sl, ALT_A)

    bot2, agent2, _ = _bot(tmp_path)
    await bot2._activate_room(ROOM)
    assert agent2.get_model(ROOM) == ALT_A


@pytest.mark.asyncio
async def test_repeated_same_process_umbrals_preserve_model(tmp_path):
    bot, agent, sl = _bot(tmp_path)
    await bot._activate_room(ROOM)
    _switch_model(agent, sl, ALT_A)

    for _ in range(2):
        await bot._inject_umbral(ROOM)
        assert agent.get_model(ROOM) == ALT_A
        await bot._activate_room(ROOM)
        assert agent.get_model(ROOM) == ALT_A


@pytest.mark.asyncio
async def test_umbral_then_restart_restores_raw_model_alias_once(tmp_path):
    bot1, agent1, sl = _bot(tmp_path)
    await bot1._activate_room(ROOM)
    _switch_model(agent1, sl, ALT_A)

    await bot1._inject_umbral(ROOM)
    assert _events(sl)[0][0] == "umbral_reset"
    assert _event_details(sl, "model_override") == [ALT_A]

    bot2, agent2, _ = _bot(tmp_path)
    await bot2._activate_room(ROOM)
    assert agent2.get_model(ROOM) == ALT_A


@pytest.mark.asyncio
async def test_repeated_umbral_restarts_are_last_switch_wins_without_duplicates(tmp_path):
    bot, agent, sl = _bot(tmp_path)
    await bot._activate_room(ROOM)
    _switch_model(agent, sl, ALT_A)

    await bot._inject_umbral(ROOM)
    bot, agent, sl = _bot(tmp_path)
    await bot._activate_room(ROOM)
    assert agent.get_model(ROOM) == ALT_A
    _switch_model(agent, sl, ALT_B)

    await bot._inject_umbral(ROOM)
    assert _event_details(sl, "model_override") == [ALT_B]
    bot, agent, sl = _bot(tmp_path)
    await bot._activate_room(ROOM)
    assert agent.get_model(ROOM) == ALT_B


@pytest.mark.asyncio
async def test_explicit_model_equal_to_config_default_is_preserved(tmp_path):
    bot1, agent1, sl = _bot(tmp_path)
    await bot1._activate_room(ROOM)
    _switch_model(agent1, sl, DEFAULT)

    await bot1._inject_umbral(ROOM)
    assert _event_details(sl, "model_override") == [DEFAULT]
    bot2, agent2, _ = _bot(tmp_path)
    await bot2._activate_room(ROOM)
    assert agent2._room_models[ROOM] == DEFAULT


@pytest.mark.asyncio
async def test_no_explicit_overrides_stays_absent_and_follows_default(tmp_path):
    bot1, agent1, sl = _bot(tmp_path)
    await bot1._activate_room(ROOM)

    await bot1._inject_umbral(ROOM)
    assert _events(sl) == [("umbral_reset", _events(sl)[0][1])]
    bot2, agent2, _ = _bot(tmp_path)
    await bot2._activate_room(ROOM)
    assert agent2.get_model(ROOM) == DEFAULT
    assert ROOM not in agent2._room_models
    assert _event_details(sl, "model_override") == []


@pytest.mark.asyncio
async def test_all_sibling_overrides_roundtrip_through_umbral_restart(tmp_path):
    bot1, agent1, sl = _bot(tmp_path)
    await bot1._activate_room(ROOM)
    _switch_model(agent1, sl, ALT_A)
    bot1._room_effort[ROOM] = "xhigh"
    bot1._room_cache_ttl[ROOM] = "5m"
    bot1._room_timesense[ROOM] = True
    agent1._room_quiet[ROOM] = True
    _append(sl, "effort_override", "xhigh")
    _append(sl, "cache_ttl_override", "5m")
    _append(sl, "timesense_override", "on")
    _append(sl, "quiet_override", "on")

    await bot1._inject_umbral(ROOM)
    ordered_events = _events(sl)
    assert ordered_events == [
        ("umbral_reset", ordered_events[0][1]),
        ("model_override", ALT_A),
        ("effort_override", "xhigh"),
        ("cache_ttl_override", "5m"),
        ("timesense_override", "on"),
        ("quiet_override", "on"),
    ]
    assert sl.read(ROOM)[0]["event"] == "umbral_reset"

    bot2, agent2, _ = _bot(tmp_path)
    await bot2._activate_room(ROOM)
    assert bot2._room_effort[ROOM] == "xhigh"
    assert bot2._room_cache_ttl[ROOM] == "5m"
    assert bot2._room_timesense[ROOM] is True
    assert agent2._room_quiet[ROOM] is True


@pytest.mark.asyncio
async def test_legacy_effort_is_modernized_but_default_sentinels_stay_absent(tmp_path):
    bot1, agent1, sl = _bot(tmp_path)
    _append(sl, "thinking_override", "high")
    _append(sl, "cache_ttl_override", "1h")
    _append(sl, "timesense_override", "off")
    await bot1._activate_room(ROOM)
    assert bot1._room_effort[ROOM] == "high"
    assert ROOM not in bot1._room_cache_ttl
    assert ROOM not in bot1._room_timesense

    await bot1._inject_umbral(ROOM)
    assert _event_details(sl, "effort_override") == ["high"]
    assert _event_details(sl, "thinking_override") == []
    assert _event_details(sl, "cache_ttl_override") == []
    assert _event_details(sl, "timesense_override") == []

    bot2, _, _ = _bot(tmp_path)
    await bot2._activate_room(ROOM)
    assert bot2._room_effort[ROOM] == "high"
    assert ROOM not in bot2._room_cache_ttl
    assert ROOM not in bot2._room_timesense


@pytest.mark.asyncio
async def test_explicit_quiet_off_retains_presence_semantics(tmp_path):
    bot1, agent1, sl = _bot(tmp_path)
    await bot1._activate_room(ROOM)
    agent1._room_quiet[ROOM] = False
    _append(sl, "quiet_override", "off")

    await bot1._inject_umbral(ROOM)
    assert _event_details(sl, "quiet_override") == ["off"]
    bot2, agent2, _ = _bot(tmp_path)
    await bot2._activate_room(ROOM)
    assert ROOM in agent2._room_quiet
    assert agent2._room_quiet[ROOM] is False


@pytest.mark.asyncio
async def test_archive_failure_preserves_active_file_and_never_reseeds(tmp_path):
    bot, agent, sl = _bot(tmp_path)
    await bot._activate_room(ROOM)
    _switch_model(agent, sl, ALT_A)
    bot.umbral = SimpleNamespace(
        directive_for=MagicMock(return_value=None), stop=AsyncMock())
    sl.archive = MagicMock(side_effect=OSError("archive failed"))
    sl.wipe = MagicMock()

    await bot._inject_umbral(ROOM)
    sl.wipe.assert_not_called()
    bot.umbral.stop.assert_awaited_once_with(ROOM)
    assert _event_details(sl, "model_override") == [ALT_A]
    assert _event_details(sl, "umbral_reset") == []
    assert ROOM in bot._active_rooms
    alerts = [call.args[1] for call in bot.send.await_args_list]
    assert any("session preserved, umbral stopped" in alert
               for alert in alerts)


@pytest.mark.parametrize("failing_event", ["umbral_reset", "model_override"])
@pytest.mark.parametrize("exception_type", [OSError, FileNotFoundError])
@pytest.mark.asyncio
async def test_override_append_failure_reports_truthfully_and_stops(
        tmp_path, failing_event, exception_type):
    bot, agent, sl = _bot(tmp_path)
    await bot._activate_room(ROOM)
    _switch_model(agent, sl, ALT_A)

    bot.umbral = SimpleNamespace(
        directive_for=MagicMock(return_value=None), stop=AsyncMock())
    original_append = sl.append
    append_attempts = []

    def fail_on_reseed(**kwargs):
        event = kwargs.get("event")
        if event in OVERRIDE_EVENTS | {"umbral_reset"}:
            append_attempts.append(event)
        if event == failing_event:
            raise exception_type("reseed append failed")
        return original_append(**kwargs)

    sl.append = fail_on_reseed
    original_reset = agent.reset_room
    agent.reset_room = MagicMock(wraps=original_reset)

    await bot._inject_umbral(ROOM)
    bot.umbral.stop.assert_awaited_once_with(ROOM)
    agent.reset_room.assert_not_called()
    assert agent.get_model(ROOM) == ALT_A
    assert ROOM in bot._active_rooms
    expected_attempts = (["umbral_reset"] if failing_event == "umbral_reset"
                         else ["umbral_reset", "model_override"])
    assert append_attempts == expected_attempts

    archives = list((tmp_path / "sessions").glob(
        "umbral-overrides_matrix.local-*.jsonl"))
    assert len(archives) == 1
    alerts = [call.args[1] for call in bot.send.await_args_list]
    assert any("rotation incomplete after archive+wipe" in alert.lower()
               and archives[0].name in alert for alert in alerts)
    assert all("not wiping" not in alert.lower()
               and "session preserved" not in alert.lower()
               for alert in alerts)


@pytest.mark.asyncio
async def test_wipe_failure_names_archive_without_preservation_claim(tmp_path):
    bot, agent, sl = _bot(tmp_path)
    await bot._activate_room(ROOM)
    _switch_model(agent, sl, ALT_A)
    bot.umbral = SimpleNamespace(
        directive_for=MagicMock(return_value=None), stop=AsyncMock())
    sl.wipe = MagicMock(side_effect=OSError("wipe failed"))
    original_reset = agent.reset_room
    agent.reset_room = MagicMock(wraps=original_reset)

    await bot._inject_umbral(ROOM)
    bot.umbral.stop.assert_awaited_once_with(ROOM)
    agent.reset_room.assert_not_called()
    archives = list((tmp_path / "sessions").glob(
        "umbral-overrides_matrix.local-*.jsonl"))
    assert len(archives) == 1
    alerts = [call.args[1] for call in bot.send.await_args_list]
    assert any("wipe failed after archive" in alert.lower()
               and archives[0].name in alert for alert in alerts)
    assert all("not wiping" not in alert.lower()
               and "session preserved" not in alert.lower()
               for alert in alerts)


@pytest.mark.asyncio
async def test_reset_failure_reports_durable_reseed_not_append_failure(tmp_path):
    bot, agent, sl = _bot(tmp_path)
    await bot._activate_room(ROOM)
    _switch_model(agent, sl, ALT_A)
    bot.umbral = SimpleNamespace(
        directive_for=MagicMock(return_value=None), stop=AsyncMock())
    agent.reset_room = MagicMock(side_effect=OSError("reset failed"))

    await bot._inject_umbral(ROOM)
    bot.umbral.stop.assert_awaited_once_with(ROOM)
    assert _event_details(sl, "model_override") == [ALT_A]
    assert ROOM in bot._active_rooms
    archives = list((tmp_path / "sessions").glob(
        "umbral-overrides_matrix.local-*.jsonl"))
    assert len(archives) == 1
    alerts = [call.args[1] for call in bot.send.await_args_list]
    assert any("in-memory reset failed after archive+wipe" in alert.lower()
               and archives[0].name in alert for alert in alerts)
    assert all("reseed append failed" not in alert.lower()
               for alert in alerts)


@pytest.mark.asyncio
async def test_final_notice_failure_preserves_successful_rotation(tmp_path, caplog):
    bot, agent, sl = _bot(tmp_path)
    await bot._activate_room(ROOM)
    _switch_model(agent, sl, ALT_A)
    bot.umbral = SimpleNamespace(
        directive_for=MagicMock(return_value=None), stop=AsyncMock())
    bot.send_notice = MatrixBot.send_notice.__get__(bot, MatrixBot)

    async def fail_only_conclusion(_room, content):
        if "concluded" in content["body"]:
            raise RuntimeError("notice failed")

    bot._room_send_with_retry = fail_only_conclusion

    with caplog.at_level(logging.WARNING):
        await bot._inject_umbral(ROOM)
    assert "notice failed" in caplog.text
    assert "rotation already complete" in caplog.text
    bot.umbral.stop.assert_not_awaited()
    assert ROOM not in bot._active_rooms
    assert _event_details(sl, "model_override") == [ALT_A]
    alerts = [call.args[1] for call in bot.send.await_args_list]
    assert all("rotation incomplete" not in alert.lower()
               and "reseed append failed" not in alert.lower()
               for alert in alerts)


@pytest.mark.asyncio
async def test_timer_stop_failure_still_alerts_with_deactivation_unconfirmed(tmp_path):
    bot, agent, sl = _bot(tmp_path)
    await bot._activate_room(ROOM)
    _switch_model(agent, sl, ALT_A)
    bot.umbral = SimpleNamespace(
        directive_for=MagicMock(return_value=None),
        stop=AsyncMock(side_effect=OSError("timer persistence failed")))
    original_append = sl.append

    def fail_on_model(**kwargs):
        if kwargs.get("event") == "model_override":
            raise OSError("reseed append failed")
        return original_append(**kwargs)

    sl.append = fail_on_model
    await bot._inject_umbral(ROOM)
    bot.umbral.stop.assert_awaited_once_with(ROOM)
    archives = list((tmp_path / "sessions").glob(
        "umbral-overrides_matrix.local-*.jsonl"))
    assert len(archives) == 1
    alerts = [call.args[1] for call in bot.send.await_args_list]
    assert any("rotation incomplete after archive+wipe" in alert.lower()
               and archives[0].name in alert
               and "deactivation could not be confirmed" in alert.lower()
               for alert in alerts)
    assert all("umbral stopped" not in alert.lower() for alert in alerts)


@pytest.mark.asyncio
async def test_later_override_failure_preserves_partial_seed_and_stops_loop(tmp_path):
    bot, agent, sl = _bot(tmp_path)
    await bot._activate_room(ROOM)
    _switch_model(agent, sl, ALT_A)
    bot._room_effort[ROOM] = "high"
    _append(sl, "effort_override", "high")
    bot.umbral = SimpleNamespace(
        directive_for=MagicMock(return_value=None), stop=AsyncMock())
    original_append = sl.append
    append_attempts = []

    def fail_on_effort(**kwargs):
        event = kwargs.get("event")
        if event in OVERRIDE_EVENTS | {"umbral_reset"}:
            append_attempts.append(event)
        if event == "effort_override":
            raise OSError("later append failed")
        return original_append(**kwargs)

    sl.append = fail_on_effort
    original_reset = agent.reset_room
    agent.reset_room = MagicMock(wraps=original_reset)

    await bot._inject_umbral(ROOM)
    bot.umbral.stop.assert_awaited_once_with(ROOM)
    agent.reset_room.assert_not_called()
    assert append_attempts == [
        "umbral_reset", "model_override", "effort_override"]
    assert _event_details(sl, "model_override") == [ALT_A]
    assert _event_details(sl, "effort_override") == []
    assert ROOM in bot._active_rooms


@pytest.mark.asyncio
async def test_stop_and_alert_failures_do_not_mask_original_error(tmp_path, caplog):
    bot, agent, sl = _bot(tmp_path)
    await bot._activate_room(ROOM)
    _switch_model(agent, sl, ALT_A)
    bot.umbral = SimpleNamespace(
        directive_for=MagicMock(return_value=None),
        stop=AsyncMock(side_effect=OSError("timer persistence failed")))
    bot.send = MatrixBot.send.__get__(bot, MatrixBot)
    bot._room_send_with_retry = AsyncMock(
        side_effect=RuntimeError("alert delivery failed"))
    original_append = sl.append

    def fail_on_model(**kwargs):
        if kwargs.get("event") == "model_override":
            raise OSError("reseed append failed")
        return original_append(**kwargs)

    sl.append = fail_on_model
    with caplog.at_level(logging.CRITICAL):
        await bot._inject_umbral(ROOM)
    assert "reseed append failed" in caplog.text
    assert "timer persistence failed" in caplog.text
    assert "alert delivery failed" in caplog.text
