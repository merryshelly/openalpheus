"""kdsn.330 G1 — dispatch ledger core: state machine, persistence, scoping.

Spec: kdsn.330-async-subagents-spec.md D1/D4/D5; rulings R3-R5, R15.
Real path: real Agent + real SessionLog + real callbacks; only sub
provider completion is mocked.
"""

import json
from unittest.mock import patch

import pytest



from subledger_fixtures import (
    ROOM, DEFAULT_TASK,
    build_bot, real_callbacks, sub_complete_factory, sub_tool_call,
    await_terminal, settle, ledger_entries,
)

from subledger_fixtures import default_main_stream  # noqa: F401

pytestmark = pytest.mark.usefixtures("default_main_stream")


def _cb(bot, call_id, **extra):
    cb = real_callbacks(bot)
    cb["call_id"] = call_id
    cb.update(extra)
    return cb


@pytest.mark.asyncio
async def test_dispatch_creates_running_record(tmp_path):
    """Dispatching background ⇒ ledger record in `running` with provenance fields."""
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=1.0)):
        res = await execute_tool(
            "subagent", sub_tool_call(task=DEFAULT_TASK),
            {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g1a"))
    assert not res.is_error, f"receipt must not be an error: {res.content}"
    room_ledger = agent._dispatch_ledger.get(ROOM, {})
    assert room_ledger, "no ledger record created for dispatch"
    rec = room_ledger.get("tc_g1a")
    assert rec is not None, f"dispatch id not keyed by call_id: {list(room_ledger)}"
    assert rec.state == "running"
    assert rec.task == DEFAULT_TASK, "full task text must be stored"
    assert rec.task_head and len(rec.task_head) <= 80
    assert rec.model, "model must be recorded"
    assert rec.dispatched_at


@pytest.mark.asyncio
async def test_terminal_completed_persists_state_and_result(tmp_path):
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(content="REPORT-XYZ-9182")):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g1b"))
    rec = await await_terminal(agent, ROOM, "tc_g1b")
    assert rec.state == "completed"
    assert "REPORT-XYZ-9182" in (rec.result or "")
    assert rec.terminal_at


@pytest.mark.asyncio
async def test_terminal_state_protection_late_event_cannot_regress(tmp_path):
    """Once terminal, a late in-flight event must NOT regress the record."""
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory()):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g1c"))
    rec = await await_terminal(agent, ROOM, "tc_g1c")
    assert rec.state == "completed"
    rec.transition("running")  # late stray event via the public transition API
    assert rec.state == "completed", (
        "late event regressed a terminal record — terminal-state protection missing")


@pytest.mark.asyncio
async def test_persist_before_notify_jsonl_entry_exists(tmp_path):
    """D1/D5: the subagent_terminal system entry is durable, not just announced."""
    bot, agent = build_bot(tmp_path)
    fired = []
    cb = _cb(bot, "tc_g1d")
    orig = cb.get("subagent_terminal_notify")

    async def spying_notify(room_id, event):
        fired.append(event)
        if orig is not None:
            await orig(room_id, event)

    cb["subagent_terminal_notify"] = spying_notify
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory()):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=cb)
    await await_terminal(agent, ROOM, "tc_g1d")
    await settle()
    entries = ledger_entries(bot.session_log, ROOM, "subagent_terminal")
    assert entries, "no subagent_terminal system entry persisted"
    assert fired, "notify callback never fired"
    detail = entries[0].get("detail")
    payload = detail if isinstance(detail, dict) else json.loads(str(detail))
    assert payload.get("dispatch_id") == "tc_g1d", (
        "persisted terminal entry must identify the dispatch (persist-before-notify)")


@pytest.mark.asyncio
async def test_ledger_entries_are_system_role(tmp_path):
    """R5: ledger lifecycle entries are role=system — never user/assistant."""
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory()):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g1e"))
    await await_terminal(agent, ROOM, "tc_g1e")
    await settle()
    for e in bot.session_log.read(ROOM):
        if str(e.get("event", "")).startswith("subagent"):
            assert e.get("role") == "system", (
                f"ledger entry leaked as role={e.get('role')!r} — cache/byte-identity hazard")


@pytest.mark.asyncio
async def test_build_context_skips_ledger_entries(tmp_path):
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(content="NEEDLE-LEDGER-77")):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g1f"))
    await await_terminal(agent, ROOM, "tc_g1f")
    await settle()
    ctx = bot.session_log.build_context(ROOM)
    assert not any("NEEDLE-LEDGER-77" in str(m.get("content")) for m in ctx), (
        "ledger result text rendered into LLM context — system entries must be skipped")


@pytest.mark.asyncio
async def test_reset_room_clears_ledger(tmp_path):
    """D4: per-room ledger is cleared in Agent.reset_room."""
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory()):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g1g"))
    await await_terminal(agent, ROOM, "tc_g1g")
    agent.reset_room(ROOM)
    assert not agent._dispatch_ledger.get(ROOM), (
        "reset_room left ledger state behind — post-umbral room must start dispatch-free")


@pytest.mark.asyncio
async def test_usage_recorded_on_terminal_entry(tmp_path):
    """R15: per-sub usage rides the subagent_terminal entry."""
    bot, agent = build_bot(tmp_path)
    from openalph.provider import Usage
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(
                   usage=Usage(input_tokens=321, output_tokens=123))):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g1h"))
    await await_terminal(agent, ROOM, "tc_g1h")
    await settle()
    entries = ledger_entries(bot.session_log, ROOM, "subagent_terminal")
    assert entries, "terminal entry missing"
    detail = entries[0].get("detail")
    payload = detail if isinstance(detail, dict) else json.loads(str(detail))
    assert payload.get("input_tokens") == 321, f"usage not on terminal entry: {payload}"
    assert payload.get("output_tokens") == 123
    assert "cost_usd" in payload and "unpriced_tokens" in payload


@pytest.mark.asyncio
async def test_task_head_truncated_for_notices(tmp_path):
    bot, agent = build_bot(tmp_path)
    long_task = "x" * 300 + " tail-marker-9931"
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=1.0)):
        await execute_tool(
            "subagent", sub_tool_call(task=long_task),
            {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g1i"))
    rec = agent._dispatch_ledger[ROOM]["tc_g1i"]
    assert len(rec.task_head) <= 90, "task_head not truncated — notices will flood"
    assert rec.task == long_task, "full task text must still be stored"


@pytest.mark.asyncio
async def test_dispatch_entry_persisted_at_dispatch_time(tmp_path):
    """R5: subagent_dispatched entry written when the dispatch is created."""
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=1.0)):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g1j"))
    entries = ledger_entries(bot.session_log, ROOM, "subagent_dispatched")
    assert entries, "no subagent_dispatched entry at dispatch time"
    assert len(entries) == 1
