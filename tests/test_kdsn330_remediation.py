"""kdsn.330 audit remediation — red tests for the converged 2-lineage findings.

Sources: tmp/kdsn330/audit-synkimi3.md + audit-synglm53.md. Each test is red
against HEAD (dcb9dc9) and names the finding it pins. Fixes: single
remediation pass, then a diff-only re-audit.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from subledger_fixtures import (
    ROOM,
    build_bot, real_callbacks, sub_complete_factory,
    sub_tool_call, sub_tool_call_tc, await_terminal, settle, pending_events,
)
from openalph.provider import Response, Usage, ToolCall, StreamEvent
from subledger_fixtures import default_main_stream  # noqa: F401  (fixture provider)

EVENT_FRAME = "[Automated sub-agent events"

pytestmark = pytest.mark.usefixtures("default_main_stream")


def _cb(bot, call_id=None, **extra):
    cb = real_callbacks(bot)
    if call_id:
        cb["call_id"] = call_id
    cb.update(extra)
    return cb


# --- R-A (HIGH, both lineages): lost wakeup — vetoed fires never re-armed ---

@pytest.mark.asyncio
async def test_terminal_during_final_model_call_still_fires(tmp_path):
    """audit-synglm53 #1 / audit-synkimi3 #2: a terminal landing during the
    parent turn's FINAL model call is never drained by that turn (drain runs
    pre-model-call) and the fire was vetoed at claim time — the event must be
    RE-ARMED at turn end, not stranded until some future turn."""
    bot, agent = build_bot(tmp_path)
    # Main stream: call 1 = background subagent tool call; call 2 = final text
    # BUT the final model call takes 0.5s (sub terminal lands inside it).
    payloads = []
    call_idx = [0]

    async def slow_final_stream(*, config=None, system=None, messages=None,
                                tools=None, model="test", thinking=None,
                                cache_ttl=None, **kw):
        payloads.append(list(messages))
        call_idx[0] += 1
        if call_idx[0] == 1:
            yield StreamEvent(
                type="done",
                response=Response(content="", tool_calls=[sub_tool_call_tc("tc_rw1")],
                                  model=model, usage=Usage(input_tokens=10, output_tokens=5),
                                  stop_reason="tool_use"),
                stop_reason="tool_use", model=model)
        else:
            await asyncio.sleep(0.5)  # the "final model call" the terminal lands inside
            yield StreamEvent(type="text", content="done")
            yield StreamEvent(
                type="done",
                response=Response(content="done", model=model,
                                  usage=Usage(input_tokens=10, output_tokens=5),
                                  stop_reason="end_turn"),
                stop_reason="end_turn", model=model)

    stream_fn = slow_final_stream
    bot._run_heartbeat_turn = AsyncMock()
    with patch("openalph.agent.stream", side_effect=stream_fn), \
         patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=0.2)):
        await agent.handle_input("go", room_id=ROOM, callbacks=_cb(bot))
    await await_terminal(agent, ROOM, "tc_rw1")
    await settle(0.6)
    # The turn is over; the room is idle; the event is pending — the fire
    # must (re)arm now. NOT your-own-next-terminal recovery: no further
    # dispatch happens.
    bot._run_heartbeat_turn.assert_awaited(), "no re-fire after turn end"
    # and the pending inbox is consumed by that fire
    await settle(0.3)
    assert not pending_events(agent, ROOM) or bot._run_heartbeat_turn.await_count >= 1


# --- R-B (HIGH, synkimi3): D8 operator-facing quiet setter ------------------

@pytest.mark.asyncio
async def test_quiet_slash_setter_roundtrip(tmp_path):
    """audit-synkimi3 #1: D8 promises an operator-controlled per-room flag,
    but nothing in src/ ever sets it. Contract: the room-scoped slash handler
    `MatrixBot._cmd_subagentquiet(room_id, arg)` — arg "on"/"off" — sets
    agent._room_quiet AND appends the quiet_override JSONL room override
    (so it restores on restart)."""
    bot, agent = build_bot(tmp_path)
    await bot._cmd_subagentquiet(ROOM, "on")
    assert agent._room_quiet.get(ROOM) is True, "slash 'on' did not set the flag"
    entries = [e for e in bot.session_log.read(ROOM)
               if e.get("role") == "system" and e.get("event") == "quiet_override"]
    assert entries, "quiet_override room-override entry not persisted"
    await bot._cmd_subagentquiet(ROOM, "off")
    assert agent._room_quiet.get(ROOM) is False, "slash 'off' did not clear the flag"
    # And a fresh agent rehydrates it:
    bot2, agent2 = build_bot(tmp_path)
    await bot2._activate_room(ROOM)
    assert agent2._room_quiet.get(ROOM) is False, "last-wins 'off' not restored"


# --- R-C (MEDIUM, both): cancelled subs carry their partial burn ------------

@pytest.mark.asyncio
async def test_cancelled_sub_usage_captured(tmp_path):
    """audit-synkimi3 #3 / synglm53 #5-adjacent: CancelledError bypasses the
    bridge write, so a cancelled sub's partial burn is lost. After the fix
    (bridge write in a finally), the cancelled terminal entry carries the
    accrued usage. Choreography: iteration 1 completes (accrues), iteration 2
    blocks, cancel lands mid-flight."""
    bot, agent = build_bot(tmp_path)
    calls = [0]

    async def two_iteration_sub(**kwargs):
        calls[0] += 1
        if calls[0] == 1:
            tc = ToolCall(id=f"subtc_{calls[0]}", name="shell",
                          input={"command": "echo burn"})
            return Response(content="", tool_calls=[tc], model="anthropic/claude-sonnet-4-20250514",
                            usage=Usage(input_tokens=250, output_tokens=120),
                            stop_reason="tool_use")
        await asyncio.sleep(30)  # iteration 2 blocks; cancel lands here
        return Response(content="never", model="anthropic/claude-sonnet-4-20250514",
                        usage=Usage(input_tokens=1, output_tokens=1),
                        stop_reason="end_turn")

    from openalph.tools import discover_tools, execute_tool
    tools = discover_tools(agent.config.workspace)
    with patch("openalph.tools.subagent.complete", side_effect=two_iteration_sub):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=tools, callbacks=_cb(bot, "tc_rc"))
    await asyncio.sleep(0.4)  # let iteration 1 accrue
    res = await execute_tool("subagent_status", {"action": "cancel", "id": "tc_rc"},
                             {}, agent.config, tools=None, callbacks=_cb(bot))
    assert not res.is_error, f"cancel failed: {res.content}"
    rec = await await_terminal(agent, ROOM, "tc_rc")
    assert rec.state == "cancelled"
    await settle()
    entries = [e for e in bot.session_log.read(ROOM)
               if e.get("role") == "system" and e.get("event") == "subagent_terminal"
               and "tc_rc" in str(e.get("detail"))]
    assert entries, "cancelled terminal entry missing"
    detail = entries[0].get("detail")
    payload = detail if isinstance(detail, dict) else json.loads(str(detail))
    assert payload.get("input_tokens", 0) >= 250, (
        f"cancelled sub's partial burn lost: {payload} — bridge must write in a finally")


# --- R-D (MEDIUM, synglm53): reconstitution delivery match is line-anchored -

def _w(sl, event, detail):
    sl.append(role="system", sender="@agent:matrix.local", room=ROOM,
              event_id=None, event=event, detail=json.dumps(detail))


@pytest.mark.asyncio
async def test_reconstitution_delivery_match_is_line_anchored(tmp_path):
    """audit-synglm53 #3: 'call_1' counts as delivered if 'call_12' drained
    (substring match over batch contents). ids must match LINE-anchored."""
    from openalph.session import SessionLog
    sl = SessionLog(tmp_path, "@agent:matrix.local")
    _w(sl, "subagent_dispatched", {"dispatch_id": "call_1", "task": "t1",
                                   "task_head": "t1", "state": "running"})
    _w(sl, "subagent_terminal", {"dispatch_id": "call_1", "state": "completed",
                                 "result": "r1"})
    _w(sl, "subagent_dispatched", {"dispatch_id": "call_12", "task": "t12",
                                   "task_head": "t12", "state": "running"})
    _w(sl, "subagent_terminal", {"dispatch_id": "call_12", "state": "completed",
                                 "result": "r12"})
    # ONLY call_12's line drained (the delivery user entry):
    sl.append(role="user", sender="@agent:matrix.local", room=ROOM, event_id=None,
              content="[Automated sub-agent events — harness notice, not operator input]\n"
                      "- call_12: completed — task: t12",
              source="subagent_event")
    bot, agent = build_bot(tmp_path)
    bot._active_turns.add(ROOM)
    await bot._activate_room(ROOM)
    led = agent._dispatch_ledger.get(ROOM, {})
    assert led.get("call_12") is not None and led["call_12"].state == "completed", (
        f"delivered dispatch must stay completed: { {k: v.state for k, v in led.items()} }")
    assert led.get("call_1") is not None, "call_1 missing from reconstituted ledger"
    assert led["call_1"].state == "pending_delivery", (
        f"undelivered terminal must be pending_delivery (substring match falsely "
        f"marked it delivered): got {led['call_1'].state}")


# --- R-E (MEDIUM, synglm53): parser applies terminal states unconditionally -

def test_parse_ledger_entries_refuses_terminal_regression(tmp_path):
    """audit-synglm53 #2: the rehydration parser must honor terminal-state
    protection — a later 'running' entry cannot regress a rebuilt 'completed'
    record (the in-memory transition() refuses; the parser must too)."""
    from openalph.tools.subledger import parse_ledger_entries
    entries = [
        {"role": "system", "event": "subagent_dispatched",
         "detail": json.dumps({"dispatch_id": "d1", "task": "t", "state": "running"})},
        {"role": "system", "event": "subagent_terminal",
         "detail": json.dumps({"dispatch_id": "d1", "state": "completed", "result": "ok"})},
        {"role": "system", "event": "subagent_terminal",
         "detail": json.dumps({"dispatch_id": "d1", "state": "running"})},  # late stray
    ]
    records = parse_ledger_entries(entries)
    assert records["d1"].state == "completed", (
        f"parser regressed a terminal record: {records['d1'].state}")
