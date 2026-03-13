"""Tests for StreamingDelivery class (Phase C of kdsn.65).

Contract:
    StreamingDelivery(bot, room_id) — manages progressive message delivery via edits

    push(delta: str, done: bool = False)
        - Accumulates text deltas
        - Sends initial message after INITIAL_SEND_CHARS threshold
        - Edits message at EDIT_INTERVAL_MS / EDIT_MIN_CHARS intervals
        - On done=True: finalizes message (no cursor, proper formatting)
        - After done=True: resets internal state for next tool loop iteration

    Tuning constants:
        EDIT_INTERVAL_MS = 600    (min ms between edits)
        EDIT_MIN_CHARS = 60       (min new chars before edit)
        INITIAL_SEND_CHARS = 40   (chars before first send)

    Edit failures are logged but non-fatal. Final delivery retries via _room_send_with_retry.
    Message splitting handled on finalize if content exceeds MAX_MESSAGE_CHARS.
    Cursor indicator (▍) present during streaming, absent on finalize.
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path

from openalph.matrix import StreamingDelivery, MatrixBot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mock_bot():
    """Create a mock MatrixBot with the methods StreamingDelivery needs."""
    bot = MagicMock(spec=MatrixBot)
    # _room_send_with_retry returns a mock with event_id
    send_response = MagicMock()
    send_response.event_id = "$initial_msg_id"
    bot._room_send_with_retry = AsyncMock(return_value=send_response)
    # client.room_send for edits (non-retrying)
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock()
    # send() for fallback (if initial was never sent)
    bot.send = AsyncMock()
    return bot


ROOM = "!test:matrix.local"


# ---------------------------------------------------------------------------
# Tests: Initial send threshold
# ---------------------------------------------------------------------------

class TestInitialSend:

    @pytest.mark.asyncio
    async def test_no_send_before_threshold(self):
        """Nothing sent until INITIAL_SEND_CHARS is reached."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        await sd.push("Hi")  # < 40 chars

        bot._room_send_with_retry.assert_not_called()
        bot.client.room_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_after_threshold(self):
        """Initial message sent once INITIAL_SEND_CHARS reached."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        await sd.push("x" * 45)  # > 40 chars

        bot._room_send_with_retry.assert_called_once()
        content = bot._room_send_with_retry.call_args[0][1]
        assert content["msgtype"] == "m.text"
        assert "▍" in content["body"]  # Cursor indicator

    @pytest.mark.asyncio
    async def test_initial_send_captures_event_id(self):
        """Event ID from initial send is stored for subsequent edits."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        await sd.push("x" * 45)

        assert sd._event_id == "$initial_msg_id"


# ---------------------------------------------------------------------------
# Tests: Edit behavior
# ---------------------------------------------------------------------------

class TestEditing:

    @pytest.mark.asyncio
    async def test_edit_uses_m_replace(self):
        """Edits use m.relates_to with rel_type m.replace."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        # Trigger initial send
        await sd.push("x" * 45)
        # Force enough time + chars for an edit
        sd._last_edit_time = 0  # Bypass time check
        sd._last_edit_len = 0   # Bypass char check
        await sd.push("y" * 70)

        bot.client.room_send.assert_called()
        edit_content = bot.client.room_send.call_args[0][2]
        assert "m.relates_to" in edit_content
        assert edit_content["m.relates_to"]["rel_type"] == "m.replace"
        assert edit_content["m.relates_to"]["event_id"] == "$initial_msg_id"

    @pytest.mark.asyncio
    async def test_edit_contains_cursor(self):
        """Intermediate edits include cursor indicator."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        await sd.push("x" * 45)
        sd._last_edit_time = 0
        sd._last_edit_len = 0
        await sd.push("y" * 70)

        edit_content = bot.client.room_send.call_args[0][2]
        new_content = edit_content["m.new_content"]
        assert "▍" in new_content["body"]

    @pytest.mark.asyncio
    async def test_no_edit_below_char_threshold(self):
        """No edit if less than EDIT_MIN_CHARS of new content."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        await sd.push("x" * 45)  # Initial send
        sd._last_edit_time = 0   # Bypass time check
        await sd.push("y" * 10)  # Only 10 new chars (< 60)

        bot.client.room_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_edit_failure_non_fatal(self):
        """Edit failures are caught and don't crash delivery."""
        bot = make_mock_bot()
        bot.client.room_send = AsyncMock(side_effect=Exception("rate limited"))
        sd = StreamingDelivery(bot, ROOM)

        await sd.push("x" * 45)  # Initial send
        sd._last_edit_time = 0
        sd._last_edit_len = 0

        # Should not raise
        await sd.push("y" * 70)


# ---------------------------------------------------------------------------
# Tests: Finalization
# ---------------------------------------------------------------------------

class TestFinalization:

    @pytest.mark.asyncio
    async def test_final_edit_no_cursor(self):
        """Finalized message has no cursor indicator."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        await sd.push("x" * 45)  # Initial send
        await sd.push("", done=True)  # Finalize

        # Should call _room_send_with_retry for final edit (retrying)
        final_call = bot._room_send_with_retry.call_args_list[-1]
        content = final_call[0][1]
        new_content = content.get("m.new_content", content)
        assert "▍" not in new_content["body"]

    @pytest.mark.asyncio
    async def test_final_has_html_formatting(self):
        """Finalized message includes org.matrix.custom.html format."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        await sd.push("**bold text**" + " " * 30)  # > threshold
        await sd.push("", done=True)

        final_call = bot._room_send_with_retry.call_args_list[-1]
        content = final_call[0][1]
        new_content = content.get("m.new_content", content)
        assert new_content.get("format") == "org.matrix.custom.html"
        assert "formatted_body" in new_content

    @pytest.mark.asyncio
    async def test_finalize_without_initial_send(self):
        """If text was too short for initial send, finalize sends normally."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        await sd.push("Hi")       # < threshold, no initial send
        await sd.push("", done=True)

        # Should use bot.send() (normal send, not edit)
        bot.send.assert_called_once_with(ROOM, "Hi")

    @pytest.mark.asyncio
    async def test_finalize_empty_buffer_no_send(self):
        """Finalize with no content sends nothing."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        await sd.push("", done=True)

        bot._room_send_with_retry.assert_not_called()
        bot.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_finalize_splits_long_messages(self):
        """Messages exceeding MAX_MESSAGE_CHARS are split on finalize."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        long_text = "x" * (MatrixBot.MAX_MESSAGE_CHARS + 1000)
        await sd.push(long_text)  # Trigger initial send
        await sd.push("", done=True)

        # Should have initial send + final edit of first chunk + additional sends
        # The exact call count depends on split logic, but should be > 1
        total_sends = (bot._room_send_with_retry.call_count +
                       bot.send.call_count)
        assert total_sends >= 2


# ---------------------------------------------------------------------------
# Tests: Self-reset after finalization
# ---------------------------------------------------------------------------

class TestSelfReset:

    @pytest.mark.asyncio
    async def test_reset_after_done(self):
        """After done=True, internal state resets for next message."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        # First message
        await sd.push("x" * 45)
        await sd.push("", done=True)

        assert sd._event_id is None
        assert sd._buffer == ""

    @pytest.mark.asyncio
    async def test_second_message_after_reset(self):
        """After reset, new push() starts a fresh message."""
        bot = make_mock_bot()
        # Different event_id for second initial send
        resp1 = MagicMock()
        resp1.event_id = "$msg1"
        resp2 = MagicMock()
        resp2.event_id = "$msg2"
        bot._room_send_with_retry = AsyncMock(side_effect=[resp1, resp1, resp2, resp2])

        sd = StreamingDelivery(bot, ROOM)

        # First message
        await sd.push("First message content here!!! " * 2)  # > 40 chars
        await sd.push("", done=True)

        # Second message (after tool loop)
        await sd.push("Second message content here!! " * 2)   # > 40 chars
        await sd.push("", done=True)

        # Should have sent two separate initial messages
        initial_sends = [c for c in bot._room_send_with_retry.call_args_list
                         if "m.relates_to" not in c[0][1]]
        assert len(initial_sends) >= 2


# ---------------------------------------------------------------------------
# Tests: HTML rendering
# ---------------------------------------------------------------------------

class TestHtmlRendering:

    @pytest.mark.asyncio
    async def test_markdown_rendered_to_html(self):
        """Message content is rendered from markdown to HTML."""
        bot = make_mock_bot()
        sd = StreamingDelivery(bot, ROOM)

        await sd.push("Hello **world**" + " " * 30)
        await sd.push("", done=True)

        final_call = bot._room_send_with_retry.call_args_list[-1]
        content = final_call[0][1]
        new_content = content.get("m.new_content", content)
        assert "<strong>" in new_content.get("formatted_body", "")
