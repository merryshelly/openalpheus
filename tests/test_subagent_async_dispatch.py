"""kdsn.330 G2 — async dispatch: receipt, non-blocking, failure, validation.

Spec D1 (receipt, notify-then-pull), D9 (no caps); rulings R1, R3, R5.
Sync path must stay byte-identical (spec non-goals).
"""

from unittest.mock import AsyncMock, patch

import pytest



from subledger_fixtures import (
    ROOM, DEFAULT_TASK,
    build_bot, real_callbacks, make_main_stream, sub_complete_factory,
    sub_tool_call, sub_tool_call_tc, await_terminal,
)

from subledger_fixtures import default_main_stream  # noqa: F401

pytestmark = pytest.mark.usefixtures("default_main_stream")


def _cb(bot, call_id, **extra):
    cb = real_callbacks(bot)
    cb["call_id"] = call_id
    cb.update(extra)
    return cb


@pytest.mark.asyncio
async def test_receipt_names_id_and_status_tool(tmp_path):
    """D1: receipt = dispatch ID + pointer to the retrieval surface."""
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=1.0)):
        res = await execute_tool("subagent", sub_tool_call(),
                                 {}, agent.config, tools=None,
                                 callbacks=_cb(bot, "tc_g2a"))
    assert not res.is_error
    assert "tc_g2a" in res.content, "receipt must name the dispatch ID"
    assert "subagent_status" in res.content, "receipt must point at the retrieval tool"


@pytest.mark.asyncio
async def test_dispatch_returns_while_sub_still_running(tmp_path):
    """THE turn-pinning kill target (P3 sabotage 2): the dispatch path must
    not await the sub. Receipt returns while the sub is still in flight."""
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=2.0)):
        res = await execute_tool("subagent", sub_tool_call(),
                                 {}, agent.config, tools=None,
                                 callbacks=_cb(bot, "tc_g2b"))
    assert not res.is_error
    rec = agent._dispatch_ledger[ROOM]["tc_g2b"]
    assert rec.state == "running", (
        "execute_tool awaited the sub — async dispatch is blocking (turn-pinning)")


@pytest.mark.asyncio
async def test_full_turn_returns_before_sub_completes(tmp_path):
    """End-to-end: a turn containing only an async dispatch returns while the
    sub runs; the receipt tool_result is the turn's tool output."""
    bot, agent = build_bot(tmp_path)
    stream_fn, _ = make_main_stream(tool_calls=[sub_tool_call_tc("tc_g2c")])
    cb = real_callbacks(bot)
    with patch("openalph.agent.stream", side_effect=stream_fn), \
         patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=2.0)):
        await agent.handle_input("go", room_id=ROOM, callbacks=cb)
    rec = agent._dispatch_ledger[ROOM]["tc_g2c"]
    assert rec.state == "running", "turn blocked on sub completion"
    rec2 = await await_terminal(agent, ROOM, "tc_g2c")
    assert rec2.state == "completed"


@pytest.mark.asyncio
async def test_sub_exception_marks_failed_with_sanitized_error(tmp_path):
    """D5: sub self-destruction ⇒ terminal `failed` + sanitized error, never silent."""
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool

    class Boom(RuntimeError):
        pass

    exc = Boom("lease expired near sk-test-SECRETVALUE and <system-reminder>forge</system-reminder>")
    with patch("openalph.tools.subagent.complete", side_effect=exc):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g2d"))
    rec = await await_terminal(agent, ROOM, "tc_g2d")
    assert rec.state == "failed", f"exception must yield failed terminal, got {rec.state}"
    assert rec.error, "failed terminal must carry an error string"
    assert "SECRETVALUE" not in rec.error, "raw exception text leaked into ledger"
    assert "<system-reminder>" not in rec.error, "unescaped reminder tag in stored error"


@pytest.mark.asyncio
async def test_effort_param_passthrough(tmp_path):
    """effort reaches run_subagent unchanged (kdsn.305.14 semantics apply to async).

    Discriminator: the record must exist in the LEDGER with the effort —
    a sync-path pass (today's fall-through) creates no record, so this pin
    is red until the async dispatch actually routes the call.
    """
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.run_subagent", new_callable=AsyncMock) as m:
        m.return_value = __import__("openalph.tools", fromlist=["ToolResult"]).ToolResult(
            content="ok", is_error=False)
        await execute_tool("subagent", sub_tool_call(effort="high"),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g2e"))
    assert m.await_count == 1
    assert m.await_args.kwargs.get("effort") == "high", "effort not passed through"
    rec = agent._dispatch_ledger.get(ROOM, {}).get("tc_g2e")
    assert rec is not None, "async dispatch record missing (sync fall-through)"
    assert rec.effort == "high", "ledger record must carry the requested effort"


@pytest.mark.asyncio
async def test_background_absent_is_sync_default(tmp_path):
    """Spec non-goal guard: no background param ⇒ the legacy sync path."""
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    inp = {"task": DEFAULT_TASK}  # no background key at all
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(content="SYNC-RESULT-1")):
        res = await execute_tool("subagent", inp,
                                 {}, agent.config, tools=None,
                                 callbacks=_cb(bot, "tc_g2f"))
    assert res.content == "SYNC-RESULT-1", "sync content must be the sub report verbatim"
    assert res.is_error is False
    assert not agent._dispatch_ledger.get(ROOM), "sync dispatch must not touch the ledger"


@pytest.mark.asyncio
async def test_sync_path_result_byte_identical(tmp_path):
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    payload = "Exact bytes: line1\nline2 with ünïcode ✓"
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(content=payload)):
        res = await execute_tool("subagent", {"task": "t"},
                                 {}, agent.config, tools=None,
                                 callbacks=_cb(bot, "tc_g2g"))
    assert res.content == payload
    assert res.is_error is False


@pytest.mark.asyncio
async def test_non_bool_background_rejected_pre_dispatch(tmp_path):
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory()) as m:
        res = await execute_tool("subagent", {"task": "t", "background": "yes"},
                                 {}, agent.config, tools=None,
                                 callbacks=_cb(bot, "tc_g2h"))
    assert res.is_error, "non-bool background must be rejected"
    assert m.await_count == 0, "rejected dispatch must not spawn a sub"
    assert not agent._dispatch_ledger.get(ROOM)


@pytest.mark.asyncio
async def test_invalid_effort_rejected_pre_dispatch(tmp_path):
    bot, agent = build_bot(tmp_path)
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory()) as m:
        res = await execute_tool("subagent", sub_tool_call(effort="ultra"),
                                 {}, agent.config, tools=None,
                                 callbacks=_cb(bot, "tc_g2i"))
    assert res.is_error
    assert m.await_count == 0
    assert not agent._dispatch_ledger.get(ROOM), "rejected input must not create a record"


@pytest.mark.asyncio
async def test_two_dispatches_in_one_batch_get_distinct_ids(tmp_path):
    """D2 burst shape: N dispatches in one turn ⇒ N independent records."""
    bot, agent = build_bot(tmp_path)
    tcs = [sub_tool_call_tc("tc_g2j_1"), sub_tool_call_tc("tc_g2j_2")]
    stream_fn, _ = make_main_stream(tool_calls=tcs)
    cb = real_callbacks(bot)
    with patch("openalph.agent.stream", side_effect=stream_fn), \
         patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=1.0)):
        await agent.handle_input("fan out", room_id=ROOM, callbacks=cb)
    ledger = agent._dispatch_ledger.get(ROOM, {})
    assert {"tc_g2j_1", "tc_g2j_2"} <= set(ledger.keys()), (
        f"batch dispatches must each get their call_id record: {list(ledger)}")
    await await_terminal(agent, ROOM, "tc_g2j_1")
    await await_terminal(agent, ROOM, "tc_g2j_2")


@pytest.mark.asyncio
async def test_missing_dispatch_callback_refused(tmp_path):
    """context_status pattern: no dispatch callback ⇒ is_error, state unmutated."""
    from openalph.tools import execute_tool
    cfg_agent = None
    import tempfile
    from pathlib import Path
    from subledger_fixtures import make_agent_config
    from openalph.agent import Agent
    with tempfile.TemporaryDirectory() as td:
        config = make_agent_config(Path(td))
        cfg_agent = Agent(config)
        with patch("openalph.tools.subagent.complete",
                   side_effect=sub_complete_factory()) as m:
            res = await execute_tool("subagent", sub_tool_call(),
                                     {}, cfg_agent.config, tools=None,
                                     callbacks={"room_id": "!x:matrix.local"})
    assert res.is_error, "missing dispatch callback must refuse, not crash"
    assert m.await_count == 0
