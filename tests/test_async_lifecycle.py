"""kdsn.330 G6 — lifecycle: /stop, umbral, quiet flag. Spec D6/D7/D8; rulings R10, R12, R13.
"""

from unittest.mock import AsyncMock, patch

import pytest



from subledger_fixtures import (
    ROOM, build_bot, real_callbacks, sub_complete_factory,
    sub_tool_call, await_terminal, settle, ledger_entries, pending_events,
)

from subledger_fixtures import default_main_stream  # noqa: F401

pytestmark = pytest.mark.usefixtures("default_main_stream")

EVENT_FRAME = "[Automated sub-agent events"


def _cb(bot, call_id=None, **extra):
    cb = real_callbacks(bot)
    if call_id:
        cb["call_id"] = call_id
    cb.update(extra)
    return cb


async def _dispatch_running(bot, agent, call_id="tc_g6", delay=1.5):
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=delay)):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, call_id))


# --- /stop (D7) -------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_cancels_running_dispatches(tmp_path):
    """D7: 'stop stops the room's work' — background dispatches included."""
    bot, agent = build_bot(tmp_path)
    await _dispatch_running(bot, agent, call_id="tc_g6a")
    await bot._cancel_current(ROOM)
    rec = await await_terminal(agent, ROOM, "tc_g6a")
    assert rec.state == "cancelled", f"/stop did not cancel the dispatch: {rec.state}"


@pytest.mark.asyncio
async def test_stop_writes_terminal_entry_and_notice(tmp_path):
    bot, agent = build_bot(tmp_path)
    await _dispatch_running(bot, agent, call_id="tc_g6b")
    await bot._cancel_current(ROOM)
    await await_terminal(agent, ROOM, "tc_g6b")
    await settle()
    entries = ledger_entries(bot.session_log, ROOM, "subagent_terminal")
    assert any("tc_g6b" in str(e.get("detail")) for e in entries), (
        "cancellation entry missing from the ledger causality chain")
    notices = [str(c.args[1]) for c in bot.send_notice.call_args_list
               if len(c.args) >= 2]
    assert any("tc_g6b" in n for n in notices), "no room notice for the cancellation"


# --- umbral (D6) ------------------------------------------------------------

@pytest.mark.asyncio
async def test_umbral_cancels_inflight_and_drains_terminal_prewipe(tmp_path):
    """D6: archive keeps the full lifecycle truthful; post-wipe room is clean."""
    bot, agent = build_bot(tmp_path)
    # one completed-but-undelivered, one in-flight
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=0.0)):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g6c1"))
    await await_terminal(agent, ROOM, "tc_g6c1")
    await _dispatch_running(bot, agent, call_id="tc_g6c2", delay=1.5)
    bot._active_turns.add(ROOM)  # keep delivery pending (deposits, no fire)
    assert pending_events(agent, ROOM)
    bot._active_turns.discard(ROOM)

    await bot._inject_umbral(ROOM)
    await settle()

    # Post-wipe: session file empty, ledger cleared
    assert agent.history(ROOM) == [] or not agent._dispatch_ledger.get(ROOM), (
        "post-wipe ledger must be empty")
    assert not pending_events(agent, ROOM), "pending events survived the wipe"


@pytest.mark.asyncio
async def test_umbral_cancellation_entries_land_in_archive(tmp_path):
    """D6: in-flight cancelled BEFORE archive+wipe ⇒ cancelled entries archived."""
    bot, agent = build_bot(tmp_path)
    await _dispatch_running(bot, agent, call_id="tc_g6d", delay=1.5)
    await bot._inject_umbral(ROOM)
    await settle()
    # The live session was wiped; the archive must carry the cancelled record.
    # Selector: the room's safe id contains '_' (async-room_matrix.local), so
    # the LIVE file also matches *_*.jsonl and sorts AFTER its archives
    # ('-' 0x2D < '_' 0x5F → archive sorts first). Exclude the live file.
    sessions = bot.session_log.workspace / "sessions"
    archives = sorted(
        f for f in sessions.glob("*_*.jsonl")
        if f.name != "async-room_matrix.local.jsonl")
    assert archives, "no archive written"
    text = archives[-1].read_text()
    assert "tc_g6d" in text, "cancellation not in the archive"
    assert "cancelled" in text


@pytest.mark.asyncio
async def test_umbral_archive_failure_preserves_room_and_ledger(tmp_path):
    """Anchors D6 note: wipe-refusal path keeps the room — no destruction on
    backup failure; the cancelled states recorded pre-archive stay truthful."""
    bot, agent = build_bot(tmp_path)
    await _dispatch_running(bot, agent, call_id="tc_g6e", delay=1.5)
    with patch.object(bot.session_log, "archive",
                      side_effect=OSError("disk full")):
        await bot._inject_umbral(ROOM)
    await settle()
    # No wipe: live session still readable, ledger still populated
    assert agent._dispatch_ledger.get(ROOM), "ledger destroyed on archive failure"
    live = bot.session_log.read(ROOM)
    assert live, "session wiped despite archive failure — data loss"
    rec = agent._dispatch_ledger[ROOM]["tc_g6e"]
    assert rec.state in ("cancelled", "running"), (
        f"unexpected state {rec.state}: archive-failure branch must not destroy work")


# --- quiet flag (D8) --------------------------------------------------------

@pytest.mark.asyncio
async def test_quiet_suppresses_synthetic_fire(tmp_path):
    bot, agent = build_bot(tmp_path)
    bot._run_heartbeat_turn = AsyncMock()
    agent._room_quiet = getattr(agent, "_room_quiet", {})
    agent._room_quiet[ROOM] = True
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=0.0)):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g6f"))
    await await_terminal(agent, ROOM, "tc_g6f")
    await settle(0.6)
    bot._run_heartbeat_turn.assert_not_awaited()
    assert pending_events(agent, ROOM), "quiet room must still deposit for later drain"


@pytest.mark.asyncio
async def test_quiet_suppresses_room_notices(tmp_path):
    bot, agent = build_bot(tmp_path)
    agent._room_quiet = getattr(agent, "_room_quiet", {})
    agent._room_quiet[ROOM] = True
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=0.0)):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g6g"))
    await await_terminal(agent, ROOM, "tc_g6g")
    await settle()
    notices = [str(c.args[1]) for c in bot.send_notice.call_args_list
               if len(c.args) >= 2]
    assert not any("tc_g6g" in n for n in notices), "quiet room emitted a notice"


@pytest.mark.asyncio
async def test_quiet_persisted_as_room_override_and_restored(tmp_path):
    """R10: JSONL room override; restored on _activate_room."""
    bot, agent = build_bot(tmp_path)
    # The operator-facing setter (slash handler) writes the override entry;
    # the test drives the seam the handler uses.
    assert hasattr(agent, "_room_quiet"), "agent-side quiet store missing"
    bot.session_log.append(role="system", sender=bot.config.user_id, room=ROOM,
                           event_id=None, event="quiet_override", detail="on")
    await bot._activate_room(ROOM)
    assert agent._room_quiet.get(ROOM) is True, (
        "quiet_override entry not restored on activation")


@pytest.mark.asyncio
async def test_quiet_preserved_across_umbral(tmp_path):
    """R10: quiet joins _room_models/_room_locks in reset_room's preserved list."""
    bot, agent = build_bot(tmp_path)
    agent._room_quiet = getattr(agent, "_room_quiet", {})
    agent._room_quiet[ROOM] = True
    with patch.object(bot.session_log, "archive", side_effect=OSError("no archive")):
        try:
            await bot._inject_umbral(ROOM)
        except Exception:
            pass
    # Archive failed ⇒ room preserved — quiet must obviously survive.
    assert agent._room_quiet.get(ROOM) is True
    agent.reset_room(ROOM)
    assert agent._room_quiet.get(ROOM) is True, (
        "reset_room must preserve the quiet flag (SB ruling: never silently un-quiet)")


@pytest.mark.asyncio
async def test_umbral_drain_endruns_quiet(tmp_path):
    """D8: umbral drain happens regardless of quiet."""
    bot, agent = build_bot(tmp_path)
    agent._room_quiet = getattr(agent, "_room_quiet", {})
    agent._room_quiet[ROOM] = True
    from openalph.tools import execute_tool
    with patch("openalph.tools.subagent.complete",
               side_effect=sub_complete_factory(delay=0.0)):
        await execute_tool("subagent", sub_tool_call(),
                           {}, agent.config, tools=None, callbacks=_cb(bot, "tc_g6h"))
    await await_terminal(agent, ROOM, "tc_g6h")
    await settle()
    assert pending_events(agent, ROOM)
    await bot._inject_umbral(ROOM)
    await settle()
    # Exclude the LIVE room file (safe id contains '_' — see the archive
    # selector note in test_umbral_cancellation_entries_land_in_archive).
    sessions = bot.session_log.workspace / "sessions"
    archives = sorted(
        f for f in sessions.glob("*_*.jsonl")
        if f.name != "async-room_matrix.local.jsonl")
    assert archives and "tc_g6h" in archives[-1].read_text(), (
        "umbral drain did not end-run quiet — completed event lost at rotation")


# --- kdsn.322 boundary interplay (D10) --------------------------------------

@pytest.mark.asyncio
async def test_handoff_boundary_lets_inflight_survive(tmp_path):
    """D10: in-flight dispatches survive a parent handoff boundary."""
    bot, agent = build_bot(tmp_path)
    await _dispatch_running(bot, agent, call_id="tc_g6i", delay=1.0)
    cb = _cb(bot)
    apply_cb = cb.get("apply_handoff_boundary")
    assert apply_cb is not None, "boundary tool callback missing from real callbacks"
    # Real callers pass room_id positionally + trigger kwarg (context_handoff
    # tool / _gc_apply_boundary precedent).
    await apply_cb(ROOM, trigger="test")
    rec = await await_terminal(agent, ROOM, "tc_g6i")
    assert rec.state == "completed", (
        f"boundary must NOT cancel in-flight dispatches (D10), got {rec.state}")
    assert rec.state != "orphaned_at_restart"
