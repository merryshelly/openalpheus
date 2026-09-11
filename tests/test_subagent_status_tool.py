"""kdsn.330 G5 — companion `subagent_status` tool: list|status|retrieve|cancel.

Spec §5; rulings R2, R11, R16 (subs inherit but are sentinel-refused).
"""

from unittest.mock import patch

import pytest



from subledger_fixtures import (
    ROOM, SUB_SENTINEL, DEFAULT_TASK,
    build_bot, real_callbacks, sub_complete_factory, sub_tool_call,
    await_terminal, settle,
)

from subledger_fixtures import default_main_stream  # noqa: F401

pytestmark = pytest.mark.usefixtures("default_main_stream")


def _cb(bot, call_id=None, room=ROOM, **extra):
    cb = real_callbacks(bot, room=room)
    if call_id:
        cb["call_id"] = call_id
    cb.update(extra)
    return cb


async def _dispatch_one(tmp_path, bot, agent, call_id="tc_g5", delay=1.0,
                        content="Sub report: all done."):
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=delay, content=content)):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, call_id))
    return call_id


def _status(agent, action, cb, **inp):
    from openalph.tools import execute_tool
    payload = {"action": action}
    payload.update(inp)
    return execute_tool("subagent_status", payload, {}, agent.config,
                        tools=None, callbacks=cb)


@pytest.mark.asyncio
async def test_registry_entry_exists_with_actions(tmp_path):
    from openalph.tools import BUILTIN_TOOLS
    entry = BUILTIN_TOOLS.get("subagent_status")
    assert entry, "subagent_status not registered in BUILTIN_TOOLS"
    props = entry["parameters"]["properties"]
    assert "action" in props, "action param missing"
    enum = props["action"].get("enum") or []
    assert {"list", "status", "retrieve", "cancel"} <= set(enum), (
        f"actions enum incomplete: {enum}")


@pytest.mark.asyncio
async def test_list_reports_states(tmp_path):
    bot, agent = build_bot(tmp_path)
    await _dispatch_one(tmp_path, bot, agent, call_id="tc_g5_l1", delay=1.0)
    await _dispatch_one(tmp_path, bot, agent, call_id="tc_g5_l2", delay=0.0)
    await await_terminal(agent, ROOM, "tc_g5_l2")
    await settle()
    res = await _status(agent, "list", _cb(bot))
    assert not res.is_error
    assert "tc_g5_l1" in res.content and "running" in res.content
    assert "tc_g5_l2" in res.content and "completed" in res.content


@pytest.mark.asyncio
async def test_status_single_dispatch(tmp_path):
    bot, agent = build_bot(tmp_path)
    await _dispatch_one(tmp_path, bot, agent, call_id="tc_g5_s1", delay=1.0)
    res = await _status(agent, "status", _cb(bot), id="tc_g5_s1")
    assert not res.is_error
    assert "running" in res.content
    assert DEFAULT_TASK[:20] in res.content, "status should show the task head"


@pytest.mark.asyncio
async def test_retrieve_returns_completed_report(tmp_path):
    bot, agent = build_bot(tmp_path)
    await _dispatch_one(tmp_path, bot, agent, call_id="tc_g5_r1", delay=0.0,
                        content="RET-REPORT-4242 unique payload")
    await await_terminal(agent, ROOM, "tc_g5_r1")
    await settle()
    res = await _status(agent, "retrieve", _cb(bot), id="tc_g5_r1")
    assert not res.is_error, f"retrieve failed: {res.content}"
    assert "RET-REPORT-4242" in res.content, "retrieve must return the report content"


@pytest.mark.asyncio
async def test_retrieve_unknown_id_is_error_with_steering(tmp_path):
    bot, agent = build_bot(tmp_path)
    res = await _status(agent, "retrieve", _cb(bot), id="no_such_id")
    assert res.is_error
    assert "no_such_id" in res.content, "error should name the unknown id"


@pytest.mark.asyncio
async def test_retrieve_against_running_is_error(tmp_path):
    bot, agent = build_bot(tmp_path)
    await _dispatch_one(tmp_path, bot, agent, call_id="tc_g5_r2", delay=1.0)
    res = await _status(agent, "retrieve", _cb(bot), id="tc_g5_r2")
    assert res.is_error, "retrieve on a running dispatch must refuse"
    assert "running" in res.content.lower()


@pytest.mark.asyncio
async def test_retrieve_truncation_points_at_flight_recorder(tmp_path):
    """R11: over-limit reports truncate with the standard marker + recorder pointer."""
    bot, agent = build_bot(tmp_path, truncation_limit=200)
    big = "R" * 5000 + " end-marker-777"
    await _dispatch_one(tmp_path, bot, agent, call_id="tc_g5_r3", delay=0.0,
                        content=big)
    await await_terminal(agent, ROOM, "tc_g5_r3")
    await settle()
    res = await _status(agent, "retrieve", _cb(bot), id="tc_g5_r3")
    assert "truncated" in res.content.lower(), "no truncation marker"
    assert "sessions/subs" in res.content, (
        "truncation marker must point at the flight recorder for the full output")


@pytest.mark.asyncio
async def test_cancel_running_dispatch(tmp_path):
    """D7 via tool: cancel ⇒ terminal cancelled + sub task actually cancelled."""
    bot, agent = build_bot(tmp_path)
    call_id = await _dispatch_one(tmp_path, bot, agent, call_id="tc_g5_c1", delay=1.5)
    res = await _status(agent, "cancel", _cb(bot), id=call_id)
    assert not res.is_error, f"cancel failed: {res.content}"
    rec = await await_terminal(agent, ROOM, call_id)
    assert rec.state == "cancelled"
    await settle()


@pytest.mark.asyncio
async def test_cancel_terminal_dispatch_is_error(tmp_path):
    bot, agent = build_bot(tmp_path)
    call_id = await _dispatch_one(tmp_path, bot, agent, call_id="tc_g5_c2", delay=0.0)
    await await_terminal(agent, ROOM, call_id)
    res = await _status(agent, "cancel", _cb(bot), id=call_id)
    assert res.is_error, "cannot cancel an already-terminal dispatch"


@pytest.mark.asyncio
async def test_sub_sentinel_refused_for_all_actions(tmp_path):
    """R16: subs inherit the tool but the __sub__ sentinel refuses every action."""
    bot, agent = build_bot(tmp_path)
    call_id = await _dispatch_one(tmp_path, bot, agent, call_id="tc_g5_x", delay=0.0)
    await await_terminal(agent, ROOM, call_id)
    for action, inp in (("list", {}), ("status", {"id": call_id}),
                        ("retrieve", {"id": call_id}), ("cancel", {"id": call_id})):
        res = await _status(agent, action, _cb(bot, room=SUB_SENTINEL), **inp)
        assert res.is_error, f"sub-context {action} must be refused"


@pytest.mark.asyncio
async def test_missing_status_callback_refused(tmp_path):
    """context_status pattern: no callback ⇒ is_error (headless cage without wiring)."""
    from openalph.tools import BUILTIN_TOOLS, execute_tool
    assert "subagent_status" in BUILTIN_TOOLS, (
        "tool must exist first — an unknown-tool refusal is not this pin")
    res = await execute_tool("subagent_status", {"action": "list"}, {},
                             None, tools=None, callbacks={})
    assert res.is_error


@pytest.mark.asyncio
async def test_status_truth_not_suppressed_by_quiet(tmp_path):
    """D8: quiet suppresses fires/notices, NEVER status truth."""
    bot, agent = build_bot(tmp_path)
    call_id = await _dispatch_one(tmp_path, bot, agent, call_id="tc_g5_q", delay=0.0)
    await await_terminal(agent, ROOM, call_id)
    await settle()
    agent._room_quiet = getattr(agent, "_room_quiet", {})
    agent._room_quiet[ROOM] = True
    res = await _status(agent, "status", _cb(bot), id=call_id)
    assert not res.is_error
    assert "completed" in res.content, "quiet flag suppressed status truth"
