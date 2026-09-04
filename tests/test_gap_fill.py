"""Tests for gap-fill pagination in _activate_room.

Verifies that gap-fill pages backward through Matrix messages until it finds
overlap with known session history, respects the GAP_FILL_MAX cap, and handles
edge cases cleanly.
"""

import asyncio
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock

from openalph.matrix import MatrixBot
from openalph.config import MatrixConfig


def make_matrix_config(**kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id="@merry:matrix.local",
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def make_room_message(sender, body, event_id="$evt1"):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = 1000000
    return event


def make_room(room_id):
    room = MagicMock()
    room.room_id = room_id
    return room


def _make_bot_with_session_log(known_event_ids):
    """Create a MatrixBot with a mocked session_log containing known event IDs."""
    config = make_matrix_config(user_id="@merry:matrix.local")
    agent = MagicMock()
    agent._rooms = {}
    agent.handle_input = AsyncMock(return_value="response")
    agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))
    agent.cancel = MagicMock()

    bot = MatrixBot.__new__(MatrixBot)
    bot.config = config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock()
    bot.client.room_typing = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = set()
    bot._halted_rooms = set()
    bot._session_locks = {}

    entries = [{"event_id": eid, "role": "user", "content": "old"} for eid in known_event_ids]
    if not entries:
        entries = [{"role": "system", "event_id": None}]

    session_log = MagicMock()
    session_log.read.return_value = entries
    session_log.last_event_id.return_value = "$known_last"
    session_log.build_context.return_value = []
    bot.session_log = session_log

    return bot, session_log


class TestGapFillPagination:

    @pytest.mark.asyncio
    async def test_gap_fill_pages_until_overlap(self):
        """Gap-fill pages through messages until it finds a known event ID."""
        bot, session_log = _make_bot_with_session_log(["$known1", "$known2"])

        # Page 1: all new messages, no overlap
        page1 = MagicMock()
        page1.chunk = [
            make_room_message("@sb:local", "new msg 2", "$new2"),
            make_room_message("@sb:local", "new msg 1", "$new1"),
        ]
        page1.end = "token_p2"

        # Page 2: first message is new, second is known -> overlap
        page2 = MagicMock()
        page2.chunk = [
            make_room_message("@sb:local", "new msg 0", "$new0"),
            make_room_message("@sb:local", "known msg", "$known1"),
        ]
        page2.end = "token_p3"

        bot.client.room_messages = AsyncMock(side_effect=[page1, page2])

        room = make_room("!room:local")
        event = make_room_message("@sb:local", "trigger")
        await bot._handle_room_message(room, event)
        # Drain background tasks fired by _handle_room_message
        await asyncio.gather(*bot._background_tasks)
        # Drain background tasks fired by _handle_room_message
        await asyncio.gather(*bot._background_tasks)

        # Should have called room_messages twice (stopped on overlap)
        assert bot.client.room_messages.await_count == 2

        # session_log.append calls: 3 gap-fill messages + session_resume + trigger message
        append_calls = session_log.append.call_args_list
        gap_fill_appends = [
            c for c in append_calls
            if c.kwargs.get("role") == "user" and c.kwargs.get("content") != "trigger"
        ]
        assert len(gap_fill_appends) == 3
        # Verify chronological order (reversed from newest-first collection)
        assert gap_fill_appends[0].kwargs["content"] == "new msg 0"
        assert gap_fill_appends[1].kwargs["content"] == "new msg 1"
        assert gap_fill_appends[2].kwargs["content"] == "new msg 2"

    @pytest.mark.asyncio
    async def test_gap_fill_respects_max_cap(self):
        """Gap-fill stops at GAP_FILL_MAX even without overlap."""
        bot, session_log = _make_bot_with_session_log(["$old1"])

        call_count = [0]

        def make_page(*a, **kw):
            page = MagicMock()
            base = call_count[0] * 100
            page.chunk = [
                make_room_message("@sb:local", f"msg {base + i}", f"$new{base + i}")
                for i in range(100)
            ]
            page.end = f"token_{call_count[0] + 1}"
            call_count[0] += 1
            return page

        bot.client.room_messages = AsyncMock(side_effect=make_page)

        room = make_room("!room:local")
        event = make_room_message("@sb:local", "trigger")
        await bot._handle_room_message(room, event)
        # Drain background tasks fired by _handle_room_message
        await asyncio.gather(*bot._background_tasks)

        # Should have stopped after ~5-6 pages (500 messages cap)
        assert bot.client.room_messages.await_count <= 6
        # Gap-fill user appends should be capped at 500
        gap_fill_appends = [
            c for c in session_log.append.call_args_list
            if c.kwargs.get("role") == "user" and c.kwargs.get("content") != "trigger"
        ]
        assert len(gap_fill_appends) == 500

    @pytest.mark.asyncio
    async def test_gap_fill_handles_empty_response(self):
        """Gap-fill exits cleanly when room_messages returns empty chunk."""
        bot, session_log = _make_bot_with_session_log(["$old1"])

        empty_page = MagicMock()
        empty_page.chunk = []
        empty_page.end = ""
        bot.client.room_messages = AsyncMock(return_value=empty_page)

        room = make_room("!room:local")
        event = make_room_message("@sb:local", "trigger")
        await bot._handle_room_message(room, event)
        # Drain background tasks fired by _handle_room_message
        await asyncio.gather(*bot._background_tasks)

        assert bot.client.room_messages.await_count == 1
        # Only the trigger message and session_resume, no gap-fill user messages
        gap_fill_appends = [
            c for c in session_log.append.call_args_list
            if c.kwargs.get("role") == "user" and c.kwargs.get("content") != "trigger"
        ]
        assert len(gap_fill_appends) == 0

    @pytest.mark.asyncio
    async def test_gap_fill_first_call_uses_none_start_token(self):
        """Gap-fill must call room_messages with start=None (not '') on the first call."""
        bot, session_log = _make_bot_with_session_log(["$known1"])

        overlap_page = MagicMock()
        overlap_page.chunk = [
            make_room_message("@sb:local", "known msg", "$known1"),
        ]
        overlap_page.end = "token_p2"
        bot.client.room_messages = AsyncMock(return_value=overlap_page)

        room = make_room("!room:local")
        event = make_room_message("@sb:local", "trigger")
        await bot._handle_room_message(room, event)
        # Drain background tasks fired by _handle_room_message
        await asyncio.gather(*bot._background_tasks)

        first_call_kwargs = bot.client.room_messages.call_args_list[0].kwargs
        assert first_call_kwargs.get("start") is None, (
            f"Expected start=None on first gap-fill call, got {first_call_kwargs.get('start')!r}"
        )

    @pytest.mark.asyncio
    async def test_gap_fill_no_overlap_logs_warning(self, caplog):
        """When no overlap is found, a warning is logged."""
        bot, session_log = _make_bot_with_session_log(["$old1"])

        page = MagicMock()
        page.chunk = [
            make_room_message("@sb:local", "new msg", "$new1"),
        ]
        page.end = ""  # no more pages
        bot.client.room_messages = AsyncMock(return_value=page)

        room = make_room("!room:local")
        event = make_room_message("@sb:local", "trigger")

        with caplog.at_level(logging.WARNING, logger="openalph.matrix"):
            await bot._handle_room_message(room, event)
            # Drain background tasks fired by _handle_room_message
            await asyncio.gather(*bot._background_tasks)

        assert any("no overlap found" in r.message for r in caplog.records), \
            f"Expected 'no overlap found' warning, got: {[r.message for r in caplog.records]}"
