"""kdsn.330 G7 — restart semantics: orphans, pending_delivery, no re-arm. Spec D5; ruling R14.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest



from subledger_fixtures import (
    ROOM, DEFAULT_TASK,
    build_bot, real_callbacks, sub_tool_call,
    await_terminal, settle, ledger_entries, history_user_messages,
)

from subledger_fixtures import default_main_stream  # noqa: F401

pytestmark = pytest.mark.usefixtures("default_main_stream")

EVENT_FRAME = "[Automated sub-agent events"


def _write_dispatch_entry(session_log, room, call_id, task=DEFAULT_TASK, **detail):
    session_log.append(role="system", sender="@agent:matrix.local", room=room,
                       event_id=None, event="subagent_dispatched",
                       detail=json.dumps({"dispatch_id": call_id, "task": task,
                                          "task_head": task[:80], "state": "running",
                                          **detail}))


def _write_terminal_entry(session_log, room, call_id, state="completed",
                          result="OLD-REPORT-1", **detail):
    session_log.append(role="system", sender="@agent:matrix.local", room=room,
                       event_id=None, event="subagent_terminal",
                       detail=json.dumps({"dispatch_id": call_id, "state": state,
                                          "result": result, **detail}))


@pytest.mark.asyncio
async def test_running_without_terminal_resolves_orphaned_on_activate(tmp_path):
    """D5: kill-on-restart, never re-arm — running ⇒ orphaned_at_restart."""
    from openalph.session import SessionLog
    sl = SessionLog(tmp_path, "@agent:matrix.local")
    _write_dispatch_entry(sl, ROOM, "tc_g7a")
    bot, agent = build_bot(tmp_path)
    bot._run_heartbeat_turn = AsyncMock()
    await bot._activate_room(ROOM)
    rec = agent._dispatch_ledger[ROOM]["tc_g7a"]
    assert rec.state == "orphaned_at_restart", (
        f"in-flight dispatch must resolve to orphaned_at_restart, got {rec.state}")


@pytest.mark.asyncio
async def test_no_synthetic_fire_on_restart(tmp_path):
    from openalph.session import SessionLog
    sl = SessionLog(tmp_path, "@agent:matrix.local")
    _write_dispatch_entry(sl, ROOM, "tc_g7b")
    bot, agent = build_bot(tmp_path)
    bot._run_heartbeat_turn = AsyncMock()
    with patch("openalph.tools.subagent.complete") as m:
        await bot._activate_room(ROOM)
        await settle(0.4)
    bot._run_heartbeat_turn.assert_not_awaited()
    m.assert_not_called(), "restart re-armed a sub — D5 violation"


@pytest.mark.asyncio
async def test_terminal_undelivered_becomes_pending_delivery(tmp_path):
    """D5 advisor-completion ruling: durable-but-undelivered ⇒ pending_delivery."""
    from openalph.session import SessionLog
    sl = SessionLog(tmp_path, "@agent:matrix.local")
    _write_dispatch_entry(sl, ROOM, "tc_g7c")
    _write_terminal_entry(sl, ROOM, "tc_g7c")
    bot, agent = build_bot(tmp_path)
    bot._active_turns.add(ROOM)  # activate during a "turn" so nothing fires
    await bot._activate_room(ROOM)
    rec = agent._dispatch_ledger[ROOM]["tc_g7c"]
    assert rec.state == "pending_delivery", (
        f"terminal-but-undelivered must be pending_delivery, got {rec.state}")


@pytest.mark.asyncio
async def test_pending_delivery_drains_at_next_turn(tmp_path):
    from openalph.session import SessionLog
    sl = SessionLog(tmp_path, "@agent:matrix.local")
    _write_dispatch_entry(sl, ROOM, "tc_g7d")
    _write_terminal_entry(sl, ROOM, "tc_g7d", result="RESTART-REPORT-55")
    bot, agent = build_bot(tmp_path)
    bot._active_turns.add(ROOM)
    await bot._activate_room(ROOM)
    bot._active_turns.discard(ROOM)
    await agent.handle_input("next", room_id=ROOM, callbacks=real_callbacks(bot))
    drained = [m for m in history_user_messages(agent, ROOM)
               if EVENT_FRAME in str(m.get("content", ""))
               and "tc_g7d" in str(m.get("content"))]
    assert drained, "pending_delivery event never drained at the next turn"
    assert "RESTART-REPORT-55" in str(drained[0].get("content"))


@pytest.mark.asyncio
async def test_no_rearm_of_orphaned_dispatch(tmp_path):
    from openalph.session import SessionLog
    sl = SessionLog(tmp_path, "@agent:matrix.local")
    _write_dispatch_entry(sl, ROOM, "tc_g7e")
    bot, agent = build_bot(tmp_path)
    bot._run_heartbeat_turn = AsyncMock()
    with patch("openalph.tools.subagent.complete") as m:
        await bot._activate_room(ROOM)
    m.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_against_orphan_is_error_with_steering(tmp_path):
    from openalph.session import SessionLog
    sl = SessionLog(tmp_path, "@agent:matrix.local")
    _write_dispatch_entry(sl, ROOM, "tc_g7f")
    bot, agent = build_bot(tmp_path)
    await bot._activate_room(ROOM)
    from openalph.tools import execute_tool
    res = await execute_tool("subagent_status", {"action": "retrieve", "id": "tc_g7f"},
                             {}, agent.config, tools=None,
                             callbacks=real_callbacks(bot))
    assert res.is_error
    assert "re-dispatch" in res.content.lower() or "no result" in res.content.lower(), (
        "orphan retrieve must steer: no result exists; re-dispatch if still needed")


@pytest.mark.asyncio
async def test_orphan_resolution_recorded_in_ledger_chain(tmp_path):
    from openalph.session import SessionLog
    sl = SessionLog(tmp_path, "@agent:matrix.local")
    _write_dispatch_entry(sl, ROOM, "tc_g7g")
    bot, agent = build_bot(tmp_path)
    await bot._activate_room(ROOM)
    entries = ledger_entries(bot.session_log, ROOM)
    orphan_records = [e for e in entries
                      if "tc_g7g" in str(e.get("detail"))
                      and "orphaned_at_restart" in str(e.get("detail"))]
    assert orphan_records, (
        "orphan resolution not recorded in the JSONL causality chain")


@pytest.mark.asyncio
async def test_failed_terminal_still_carries_usage(tmp_path):
    """Spec §5: orphans'/failures' burn is captured too — presence, not value."""
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool

    class Boom(RuntimeError):
        pass

    with patch("openalph.tools.subagent.complete", side_effect=Boom("die")):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None,
                           callbacks={**real_callbacks(bot), "call_id": "tc_g7h"})
    rec = await await_terminal(agent, ROOM, "tc_g7h")
    assert rec.state == "failed"
    await settle()
    entries = ledger_entries(bot.session_log, ROOM, "subagent_terminal")
    assert entries, "failed terminal not persisted"
    detail = entries[0].get("detail")
    payload = detail if isinstance(detail, dict) else json.loads(str(detail))
    assert "input_tokens" in payload and "cost_usd" in payload, (
        "failed terminal must still record usage fields")
