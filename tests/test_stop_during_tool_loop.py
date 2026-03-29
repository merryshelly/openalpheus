"""Tests for /stop cancellation during active tool loops.

Verifies that /stop is processed even when an agent is mid-tool-loop,
which requires _process_message to run as a background task so
sync_forever can dispatch the /stop event.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.matrix import MatrixBot


def _make_config(user_id="@agent:matrix.local"):
    config = MagicMock()
    config.user_id = user_id
    config.homeserver = "https://matrix.local"
    config.device_id = "TEST"
    config.sync_timeout = 30000
    return config


def _make_room(room_id="!room:matrix.local", member_count=2):
    room = MagicMock()
    room.room_id = room_id
    room.member_count = member_count
    return room


def _make_event(sender="@sb:matrix.local", body="hello", event_id="$evt1"):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.source = {"content": {"body": body}}
    return event


def _make_bot():
    config = _make_config()
    agent = MagicMock()
    agent._rooms = {}
    agent.handle_input = AsyncMock(return_value="response")
    agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))
    agent.cancel = MagicMock(return_value=None)
    agent.tools = []
    agent.config = MagicMock()
    agent.config.workspace = "/tmp/test-workspace"

    bot = MatrixBot.__new__(MatrixBot)
    bot.config = config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock()
    bot.client.room_typing = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = {"!room:matrix.local"}
    bot._room_thinking = {}
    bot._halted_rooms = set()
    bot._background_tasks = set()
    bot._session_locks = {}
    bot.session_log = MagicMock()
    bot.session_log.append = MagicMock()
    bot.session_log.build_context = MagicMock(return_value=[])
    bot.heartbeat = MagicMock()

    return bot


class TestStopDuringToolLoop:

    @pytest.mark.asyncio
    async def test_handle_room_message_fires_background_task(self):
        """_handle_room_message should create a background task, not block."""
        bot = _make_bot()
        room = _make_room()
        event = _make_event()

        # Make handle_input block until we release it
        block = asyncio.Event()
        async def slow_handle(*args, **kwargs):
            await block.wait()
            return "done"
        bot.agent.handle_input = AsyncMock(side_effect=slow_handle)

        # _handle_room_message should return immediately (not block on handle_input)
        await bot._handle_room_message(room, event)

        # A background task should be running
        assert len(bot._background_tasks) == 1

        # handle_input should NOT have completed yet
        assert bot.agent.handle_input.await_count == 0 or not block.is_set()

        # Release and drain
        block.set()
        await asyncio.gather(*bot._background_tasks)

    @pytest.mark.asyncio
    async def test_stop_cancels_during_tool_loop(self):
        """Sending /stop while a tool loop is running should cancel it."""
        bot = _make_bot()
        room = _make_room()

        # Simulate a long-running handle_input (like a tool loop)
        cancel_event = asyncio.Event()
        async def long_tool_loop(*args, **kwargs):
            try:
                await asyncio.sleep(999)  # Will be cancelled
            except asyncio.CancelledError:
                cancel_event.set()
                raise
            return "should not reach"
        bot.agent.handle_input = AsyncMock(side_effect=long_tool_loop)

        # Send a regular message — starts background processing
        msg_event = _make_event(body="do some research")
        await bot._handle_room_message(room, msg_event)

        # Give the background task a chance to start
        await asyncio.sleep(0.01)

        # Now the task should be running
        assert len(bot._background_tasks) >= 1
        task = list(bot._background_tasks)[0]
        assert not task.done()

        # Wire up agent.cancel to actually cancel the task
        bot.agent.cancel = MagicMock(side_effect=lambda: (task.cancel(), task)[-1])

        # Send /stop — should be processable because _handle_room_message returned
        stop_event = _make_event(body="/stop", event_id="$stop1")
        await bot._handle_room_message(room, stop_event)

        # The task should have been cancelled
        assert cancel_event.is_set(), "handle_input was not cancelled by /stop"

    @pytest.mark.asyncio
    async def test_stop_sends_cancelled_notice(self):
        """After /stop, the bot should send 'Cancelled.' to the room."""
        bot = _make_bot()
        room = _make_room()

        async def blocking_handle(*args, **kwargs):
            await asyncio.sleep(999)
        bot.agent.handle_input = AsyncMock(side_effect=blocking_handle)

        # Start processing
        msg_event = _make_event(body="work")
        await bot._handle_room_message(room, msg_event)
        await asyncio.sleep(0.01)

        task = list(bot._background_tasks)[0]
        bot.agent.cancel = MagicMock(side_effect=lambda: (task.cancel(), task)[-1])

        # Set _current_room as _process_message would
        bot._current_room = room.room_id

        # Mock bot.send to capture messages
        sent_messages = []
        async def capture_send(room_id, text):
            sent_messages.append(text)
        bot.send = capture_send

        stop_event = _make_event(body="/stop", event_id="$stop2")
        await bot._handle_room_message(room, stop_event)

        # Check that "Cancelled." was sent
        assert any("Cancelled" in m for m in sent_messages),             f"No 'Cancelled.' message sent. Got: {sent_messages}"

    @pytest.mark.asyncio
    async def test_background_task_cleanup(self):
        """Completed background tasks should be removed from the tracking set."""
        bot = _make_bot()
        room = _make_room()
        event = _make_event()

        await bot._handle_room_message(room, event)
        assert len(bot._background_tasks) >= 1

        # Drain
        await asyncio.gather(*bot._background_tasks)
        # After completion, done_callback should have removed it
        # (may need a tick for the callback to fire)
        await asyncio.sleep(0)
        assert len(bot._background_tasks) == 0
