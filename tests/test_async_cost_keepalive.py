"""kdsn.330 G8/G9 — cost attribution + keepalive re-gate. Spec §5; rulings R9, R15.

Keepalive pins observe the (task, stop) lifecycle via a recording wrapper
around _maybe_arm_cache_keepalive — ping internals deliberately untouched.
"""

import asyncio
import json
from unittest.mock import patch

import pytest



from subledger_fixtures import (
    ROOM, build_bot, real_callbacks, make_main_stream, sub_complete_factory,
    sub_tool_call, sub_tool_call_tc, await_terminal, settle, KeepaliveRecorder,
)

from subledger_fixtures import default_main_stream  # noqa: F401

pytestmark = pytest.mark.usefixtures("default_main_stream")


def _cb(bot, call_id=None, **extra):
    cb = real_callbacks(bot)
    if call_id:
        cb["call_id"] = call_id
    cb.update(extra)
    return cb


# --- G8: cost attribution ---------------------------------------------------

@pytest.mark.asyncio
async def test_usage_totals_learns_terminal_entries(tmp_path):
    """R15: usage_totals re-sums subagent_terminal entries (advisor-loop mirror)."""
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=0.0)):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g8a"))
    await await_terminal(agent, ROOM, "tc_g8a")
    await settle()
    totals = bot.session_log.usage_totals(ROOM)
    assert totals.get("subagent_cost_usd", 0) > 0 or totals.get("unpriced_tokens", 0) > 0, (
        f"terminal-entry cost not re-summed into usage_totals: {totals}")


@pytest.mark.asyncio
async def test_live_accrual_at_terminal_persistence(tmp_path):
    """R15: async terminal accrues live — no tool-result append site exists."""
    bot, agent = build_bot(tmp_path)
    before = dict(agent._usage_for(ROOM)) if hasattr(agent, "_usage_for") else {}
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=0.0)):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g8b"))
    await await_terminal(agent, ROOM, "tc_g8b")
    await settle()
    after = agent._usage_for(ROOM)
    grew = (after.get("subagent_cost_usd", 0) > before.get("subagent_cost_usd", 0)) or \
           (after.get("unpriced_tokens", 0) > before.get("unpriced_tokens", 0))
    assert grew, f"terminal persistence did not accrue live usage: {before} -> {after}"


@pytest.mark.asyncio
async def test_status_totals_truthful_after_restart(tmp_path):
    """R15: _activate_room re-sum includes terminal entries → /status stays truthful."""
    from openalph.session import SessionLog
    sl = SessionLog(tmp_path, "@agent:matrix.local")
    sl.append(role="system", sender="@agent:matrix.local", room=ROOM, event_id=None,
              event="subagent_terminal",
              detail=json.dumps({"dispatch_id": "tc_g8c", "state": "completed",
                                 "input_tokens": 500, "output_tokens": 200,
                                 "cost_usd": 0.05, "unpriced_tokens": 0}))
    bot, agent = build_bot(tmp_path)
    await bot._activate_room(ROOM)
    totals = bot.session_log.usage_totals(ROOM)
    assert abs(totals.get("subagent_cost_usd", 0.0) - 0.05) < 1e-9, (
        f"restart re-sum missed subagent_terminal cost: {totals}")


# --- G9: keepalive lifecycle -------------------------------------------------

def _ka_config(**kw):
    """Anthropic config with subagent_cache_keepalive enabled on the provider."""
    from subledger_fixtures import make_provider
    prov = make_provider(subagent_cache_keepalive=True)
    return {"providers": {"anthropic": prov}}


@pytest.mark.asyncio
async def test_keepalive_outlives_receipt_while_dispatch_running(tmp_path):
    """THE re-gate red (anchors 'read first'): gather returns at receipt time;
    keepalive must NOT disarm while an async dispatch is still running."""
    recorder = KeepaliveRecorder()
    bot, agent = build_bot(tmp_path, **_ka_config())
    agent._maybe_arm_cache_keepalive = recorder.wrap(agent)
    tcs = [sub_tool_call_tc("tc_g9a")]
    stream_fn, _ = make_main_stream(tool_calls=tcs)
    cb = _cb(bot)
    with patch("openalph.agent.stream", side_effect=stream_fn), \
         patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=1.5)):
        await agent.handle_input("go", room_id=ROOM, callbacks=cb)
    assert recorder.arms, "keepalive never armed for a subagent batch"
    task, stop = recorder.arms[-1]
    assert task is not None and stop is not None
    await asyncio.sleep(0.05)
    assert not stop.is_set(), (
        "keepalive disarmed at receipt — must outlive the batch while async dispatch runs")
    assert not task.done()


@pytest.mark.asyncio
async def test_keepalive_disarms_at_last_terminal(tmp_path):
    recorder = KeepaliveRecorder()
    bot, agent = build_bot(tmp_path, **_ka_config())
    agent._maybe_arm_cache_keepalive = recorder.wrap(agent)
    tcs = [sub_tool_call_tc("tc_g9b")]
    stream_fn, _ = make_main_stream(tool_calls=tcs)
    with patch("openalph.agent.stream", side_effect=stream_fn), \
         patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=0.0)):
        await agent.handle_input("go", room_id=ROOM, callbacks=_cb(bot))
    task, stop = recorder.arms[-1]
    await await_terminal(agent, ROOM, "tc_g9b")
    await settle(0.4)
    assert stop.is_set(), (
        "keepalive not disarmed after the room's LAST async terminal event")


@pytest.mark.asyncio
async def test_sync_batch_disarms_as_today(tmp_path):
    """Non-goal guard: sync-only batches disarm at gather end (byte-identical)."""
    recorder = KeepaliveRecorder()
    bot, agent = build_bot(tmp_path, **_ka_config())
    agent._maybe_arm_cache_keepalive = recorder.wrap(agent)
    tcs = [sub_tool_call_tc("tc_g9c", background=False)]
    stream_fn, _ = make_main_stream(tool_calls=tcs)
    with patch("openalph.agent.stream", side_effect=stream_fn), \
         patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=0.0)):
        await agent.handle_input("go", room_id=ROOM, callbacks=_cb(bot))
    task, stop = recorder.arms[-1]
    assert stop.is_set(), "sync batch must disarm at the gather finally as today"


@pytest.mark.asyncio
async def test_non_anthropic_parent_never_arms(tmp_path):
    from subledger_fixtures import make_provider
    recorder = KeepaliveRecorder()
    bot, agent = build_bot(
        tmp_path, providers={"blackwell": make_provider(
            key="blackwell", type="openai", api_key="none",
            base_url="http://localhost:8000/v1", subagent_cache_keepalive=True)})
    agent._maybe_arm_cache_keepalive = recorder.wrap(agent)
    tcs = [sub_tool_call_tc("tc_g9d")]
    stream_fn, _ = make_main_stream(tool_calls=tcs)
    with patch("openalph.agent.stream", side_effect=stream_fn), \
         patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=0.0)):
        await agent.handle_input("go", room_id=ROOM, callbacks=_cb(bot))
    assert all(task is None for task, _ in recorder.arms), (
        "non-Anthropic parent must not arm the keepalive")


@pytest.mark.asyncio
async def test_stop_cancellation_disarms_keepalive(tmp_path):
    """Advisor: /stop cancels the room's dispatches → the re-gate clears."""
    recorder = KeepaliveRecorder()
    bot, agent = build_bot(tmp_path, **_ka_config())
    agent._maybe_arm_cache_keepalive = recorder.wrap(agent)
    tcs = [sub_tool_call_tc("tc_g9e")]
    stream_fn, _ = make_main_stream(tool_calls=tcs)
    with patch("openalph.agent.stream", side_effect=stream_fn), \
         patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=1.5)):
        await agent.handle_input("go", room_id=ROOM, callbacks=_cb(bot))
    task, stop = recorder.arms[-1]
    assert not stop.is_set()
    await bot._cancel_current(ROOM)
    await await_terminal(agent, ROOM, "tc_g9e")
    await settle(0.3)
    assert stop.is_set(), "cancellation is terminal — keepalive must disarm"


@pytest.mark.asyncio
async def test_failed_terminal_disarms_keepalive(tmp_path):
    """Advisor: disarm on exception paths, not just gather success."""
    recorder = KeepaliveRecorder()
    bot, agent = build_bot(tmp_path, **_ka_config())
    agent._maybe_arm_cache_keepalive = recorder.wrap(agent)
    tcs = [sub_tool_call_tc("tc_g9f")]
    stream_fn, _ = make_main_stream(tool_calls=tcs)

    class Boom(RuntimeError):
        pass

    with patch("openalph.agent.stream", side_effect=stream_fn), \
         patch("openalph.tools.subagent.complete", side_effect=Boom("kaboom")):
        await agent.handle_input("go", room_id=ROOM, callbacks=_cb(bot))
    task, stop = recorder.arms[-1]
    await await_terminal(agent, ROOM, "tc_g9f")
    await settle(0.3)
    assert stop.is_set(), "failed terminal must disarm the keepalive (exception path)"


@pytest.mark.asyncio
async def test_second_arm_does_not_double_disarm_first(tmp_path):
    """Advisor: multiple batches arming in one turn — one coherent lifecycle."""
    recorder = KeepaliveRecorder()
    bot, agent = build_bot(tmp_path, **_ka_config())
    agent._maybe_arm_cache_keepalive = recorder.wrap(agent)
    tcs = [sub_tool_call_tc("tc_g9g1"), sub_tool_call_tc("tc_g9g2")]
    stream_fn, _ = make_main_stream(tool_calls=tcs)
    with patch("openalph.agent.stream", side_effect=stream_fn), \
         patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=0.2)):
        await agent.handle_input("fan out", room_id=ROOM, callbacks=_cb(bot))
    await await_terminal(agent, ROOM, "tc_g9g1")
    await await_terminal(agent, ROOM, "tc_g9g2")
    await settle(0.4)
    assert recorder.arms, "keepalive never armed"
    live = [t for t, s in recorder.arms if t is not None and not t.done()]
    assert not live, f"orphan keepalive loop still running after last terminal: {live}"
