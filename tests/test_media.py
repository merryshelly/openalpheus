"""Tests for media attachment handling (.24).

Interface contract:
    MatrixBot receives media events (Image, Audio, Video, File),
    downloads files to workspace/media/<hash>/<filename>,
    and passes bracket-tag messages to agent.handle_input().

Spec: docs/media-handling-spec.md
"""

import pytest
import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from openalph.matrix import MatrixBot, MAX_MEDIA_BYTES, MEDIA_DIR
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig


# --- Fixtures ---


def make_matrix_config(**kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id="@agent:matrix.local",
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


def make_agent_config(tmp_path, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(key="anthropic", type="anthropic", api_key="sk-test", base_url=None, quirks=[])},
        workspace=tmp_path,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_media_event(sender, body, event_id="$media1", url="mxc://matrix.local/abcdef",
                     mimetype="image/jpeg", size=1024000, event_type="image"):
    """Create a mock Matrix media event.

    Args:
        sender: Event sender
        body: Original filename (or caption)
        event_id: Matrix event ID
        url: mxc:// URI
        mimetype: MIME type
        size: File size in bytes
        event_type: One of "image", "audio", "video", "file"
    """
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.url = url
    event.server_timestamp = 1000000
    # Matrix event source with content.info
    event.source = {
        "content": {
            "body": body,
            "url": url,
            "info": {
                "mimetype": mimetype,
                "size": size,
            }
        }
    }
    return event


def make_download_response(body: bytes, content_type="image/jpeg"):
    """Create a mock successful download response."""
    resp = MagicMock()
    resp.body = body
    resp.content_type = content_type
    resp.filename = None
    # Indicate success by NOT being a DownloadError
    resp.__class__.__name__ = "MemoryDownloadResponse"
    return resp


def make_download_error(message="Not found"):
    """Create a mock download error response."""
    from nio import DownloadError
    resp = DownloadError(message)
    return resp


def _event_id_hash(event_id: str) -> str:
    """Mirror the production hash logic for test assertions."""
    return hashlib.sha256(event_id.encode()).hexdigest()[:16]


async def _make_bot(tmp_path, agent_config=None, matrix_config=None):
    """Create a MatrixBot with mocked client, ready for testing."""
    config = matrix_config or make_matrix_config()
    agent = MagicMock()
    agent.config = agent_config or make_agent_config(tmp_path)
    agent.system_prompt = "test"
    agent.history = MagicMock(return_value=[])
    agent.handle_input = AsyncMock(return_value="Got it!")
    agent.tools = []

    with patch("openalph.matrix.AsyncClient") as MockClient:
        client = MockClient.return_value
        client.user_id = config.user_id
        client.login = AsyncMock(return_value=MagicMock())
        client.sync = AsyncMock()
        client.close = AsyncMock()
        client.joined_rooms = AsyncMock(return_value=MagicMock(rooms=[]))
        client.room_send = AsyncMock(return_value=MagicMock(event_id="$resp1"))
        client.download = AsyncMock()
        client.add_event_callback = MagicMock()

        bot = MatrixBot(agent, config)
        bot._synced = True
        bot.client = client

    return bot, agent, client


# --- Constants ---


class TestConstants:
    """Verify media constants are exported and sane."""

    def test_max_media_bytes_is_20mb(self):
        assert MAX_MEDIA_BYTES == 20_000_000

    def test_media_dir_name(self):
        assert MEDIA_DIR == "media"


# --- Filename Sanitization ---


class TestFilenameSanitization:
    """Test _sanitize_filename edge cases."""

    def test_normal_filename(self):
        from openalph.matrix import _sanitize_filename
        assert _sanitize_filename("photo.jpg") == "photo.jpg"

    def test_strips_path_separators(self):
        from openalph.matrix import _sanitize_filename
        assert "/" not in _sanitize_filename("../../etc/passwd")
        assert "\\" not in _sanitize_filename("..\\..\\windows\\system32")

    def test_strips_null_bytes(self):
        from openalph.matrix import _sanitize_filename
        assert "\0" not in _sanitize_filename("file\0name.jpg")

    def test_truncates_long_names(self):
        from openalph.matrix import _sanitize_filename
        long_name = "a" * 300 + ".jpg"
        result = _sanitize_filename(long_name)
        assert len(result) <= 200

    def test_empty_name_fallback(self):
        from openalph.matrix import _sanitize_filename
        assert _sanitize_filename("") == "attachment"

    def test_only_separators_fallback(self):
        from openalph.matrix import _sanitize_filename
        assert _sanitize_filename("///") == "attachment"


# --- Event ID Hashing ---


class TestEventIdHash:
    """Test _event_id_hash produces filesystem-safe, deterministic output."""

    def test_deterministic(self):
        from openalph.matrix import _event_id_hash
        h1 = _event_id_hash("$abc123:matrix.local")
        h2 = _event_id_hash("$abc123:matrix.local")
        assert h1 == h2

    def test_length_16(self):
        from openalph.matrix import _event_id_hash
        h = _event_id_hash("$anything")
        assert len(h) == 16

    def test_hex_chars_only(self):
        from openalph.matrix import _event_id_hash
        h = _event_id_hash("$evt:matrix.local")
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_ids_different_hashes(self):
        from openalph.matrix import _event_id_hash
        h1 = _event_id_hash("$event1")
        h2 = _event_id_hash("$event2")
        assert h1 != h2


# --- Event Registration ---


class TestMediaEventRegistration:
    """Verify media event types are registered in start()."""

    @pytest.mark.asyncio
    async def test_registers_all_media_types(self):
        """start() registers callbacks for Image, Audio, Video, File events."""
        from nio import RoomMessageImage, RoomMessageAudio, RoomMessageVideo, RoomMessageFile

        config = make_matrix_config()
        agent = MagicMock()
        agent.system_prompt = "test"
        agent.tools = []

        with patch("openalph.matrix.AsyncClient") as MockClient:
            client = MockClient.return_value
            client.login = AsyncMock(return_value=MagicMock())
            client.sync = AsyncMock()
            client.close = AsyncMock()
            client.joined_rooms = AsyncMock(return_value=MagicMock(rooms=[]))
            client.add_event_callback = MagicMock()
            # Prevent actual sync loop from running
            client.sync_forever = AsyncMock()

            bot = MatrixBot(agent, config)
            await bot.start()

            # Collect all registered event types
            registered_types = [
                call_args[0][1] for call_args in client.add_event_callback.call_args_list
            ]

            assert RoomMessageImage in registered_types, "RoomMessageImage not registered"
            assert RoomMessageAudio in registered_types, "RoomMessageAudio not registered"
            assert RoomMessageVideo in registered_types, "RoomMessageVideo not registered"
            assert RoomMessageFile in registered_types, "RoomMessageFile not registered"


# --- Download + Storage ---


class TestMediaDownload:
    """Test that media files are downloaded and stored correctly."""

    @pytest.mark.asyncio
    async def test_downloads_and_stores_file(self, tmp_path):
        """Media event triggers download and file is written to correct path."""
        bot, agent, client = await _make_bot(tmp_path)
        file_content = b"\xff\xd8\xff\xe0" + b"\x00" * 1000  # fake JPEG
        client.download = AsyncMock(return_value=make_download_response(file_content))

        event = make_media_event("@user:matrix.local", "photo.jpg", event_id="$dl1")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # File should exist at workspace/media/<hash>/photo.jpg
        expected_dir = tmp_path / MEDIA_DIR / _event_id_hash("$dl1")
        expected_file = expected_dir / "photo.jpg"
        assert expected_file.exists(), f"Expected file at {expected_file}"
        assert expected_file.read_bytes() == file_content

    @pytest.mark.asyncio
    async def test_creates_media_directory(self, tmp_path):
        """Media directory is created if it doesn't exist."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_response(b"data"))

        event = make_media_event("@user:matrix.local", "file.bin", event_id="$dl2")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        media_dir = tmp_path / MEDIA_DIR / _event_id_hash("$dl2")
        assert media_dir.is_dir()

    @pytest.mark.asyncio
    async def test_download_uses_correct_mxc_url(self, tmp_path):
        """Download is called with the event's mxc:// URL."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_response(b"data"))

        event = make_media_event("@user:matrix.local", "test.png",
                                 url="mxc://matrix.local/specific123")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        client.download.assert_awaited_once()
        call_kwargs = client.download.call_args
        # Check mxc URL was passed (positional or keyword)
        assert "mxc://matrix.local/specific123" in str(call_kwargs)


# --- Message Format ---


class TestMessageFormat:
    """Test the bracket-tag message passed to agent.handle_input()."""

    @pytest.mark.asyncio
    async def test_bracket_tag_format(self, tmp_path):
        """Message includes [media: path (mime, size)] format."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_response(b"\x00" * 2400000))

        event = make_media_event("@user:matrix.local", "sunset.jpg",
                                 mimetype="image/jpeg", size=2400000)
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Check what was passed to agent.handle_input
        agent.handle_input.assert_awaited_once()
        message = agent.handle_input.call_args[0][0]
        assert message.startswith("[media:")
        assert "image/jpeg" in message
        assert "sunset.jpg" in message or "media/" in message

    @pytest.mark.asyncio
    async def test_includes_human_readable_size(self, tmp_path):
        """Size is formatted as human-readable (KB, MB)."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_response(b"\x00" * 1500000))

        event = make_media_event("@user:matrix.local", "image.png",
                                 mimetype="image/png", size=1500000)
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        message = agent.handle_input.call_args[0][0]
        # Should contain MB representation (1.5 MB or similar)
        assert "MB" in message or "KB" in message

    @pytest.mark.asyncio
    async def test_caption_included_when_different_from_filename(self, tmp_path):
        """When body differs from filename, it's included as caption."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_response(b"\x00" * 100))

        # body="Check this out" but info has filename
        event = make_media_event("@user:matrix.local", "Check this out")
        # Set a different filename in the source info
        event.source["content"]["filename"] = "photo.jpg"
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        message = agent.handle_input.call_args[0][0]
        assert "Check this out" in message

    @pytest.mark.asyncio
    async def test_no_caption_when_body_equals_filename(self, tmp_path):
        """When body is just the filename, no caption line is added."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_response(b"\x00" * 100))

        event = make_media_event("@user:matrix.local", "photo.jpg")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        message = agent.handle_input.call_args[0][0]
        lines = [l for l in message.strip().split("\n") if l.strip()]
        # Should only be the bracket-tag line, no caption
        assert len(lines) == 1

    @pytest.mark.asyncio
    async def test_mime_type_fallback(self, tmp_path):
        """Missing mimetype falls back to application/octet-stream."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_response(b"\x00" * 100))

        event = make_media_event("@user:matrix.local", "mystery.bin")
        # Remove mimetype from source
        event.source["content"]["info"] = {}
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        message = agent.handle_input.call_args[0][0]
        assert "application/octet-stream" in message


# --- Size Limit ---


class TestSizeLimit:
    """Test that oversized files are rejected."""

    @pytest.mark.asyncio
    async def test_oversized_file_rejected(self, tmp_path):
        """Files exceeding MAX_MEDIA_BYTES are not stored."""
        bot, agent, client = await _make_bot(tmp_path)
        huge_content = b"\x00" * (MAX_MEDIA_BYTES + 1)
        client.download = AsyncMock(return_value=make_download_response(huge_content))

        event = make_media_event("@user:matrix.local", "big_video.mp4",
                                 mimetype="video/mp4", size=MAX_MEDIA_BYTES + 1)
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # File should NOT be stored
        media_dir = tmp_path / MEDIA_DIR
        if media_dir.exists():
            stored_files = list(media_dir.rglob("*"))
            file_files = [f for f in stored_files if f.is_file()]
            assert len(file_files) == 0, f"Oversized file was stored: {file_files}"

    @pytest.mark.asyncio
    async def test_oversized_file_sends_skip_notification(self, tmp_path):
        """Agent receives a skip notification for oversized files."""
        bot, agent, client = await _make_bot(tmp_path)
        huge_content = b"\x00" * (MAX_MEDIA_BYTES + 1)
        client.download = AsyncMock(return_value=make_download_response(huge_content))

        event = make_media_event("@user:matrix.local", "huge.mp4",
                                 mimetype="video/mp4", size=MAX_MEDIA_BYTES + 1)
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Agent should receive a skip message
        agent.handle_input.assert_awaited_once()
        message = agent.handle_input.call_args[0][0]
        assert "skipped" in message.lower() or "exceed" in message.lower()
        assert "huge.mp4" in message

    @pytest.mark.asyncio
    async def test_file_at_exact_limit_accepted(self, tmp_path):
        """File exactly at MAX_MEDIA_BYTES is accepted."""
        bot, agent, client = await _make_bot(tmp_path)
        content = b"\x00" * MAX_MEDIA_BYTES
        client.download = AsyncMock(return_value=make_download_response(content))

        event = make_media_event("@user:matrix.local", "big_but_ok.bin",
                                 size=MAX_MEDIA_BYTES)
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        # Should be stored, not skipped
        message = agent.handle_input.call_args[0][0]
        assert "skipped" not in message.lower()


# --- Download Errors ---


class TestDownloadErrors:
    """Test graceful handling of download failures."""

    @pytest.mark.asyncio
    async def test_download_error_sends_notification(self, tmp_path):
        """Download failure sends error notification to agent."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_error("404 Not Found"))

        event = make_media_event("@user:matrix.local", "missing.jpg")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        agent.handle_input.assert_awaited_once()
        message = agent.handle_input.call_args[0][0]
        assert "failed" in message.lower() or "error" in message.lower()
        assert "missing.jpg" in message

    @pytest.mark.asyncio
    async def test_download_error_does_not_crash(self, tmp_path):
        """Download failure doesn't crash the bot."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_error("Server error"))

        event = make_media_event("@user:matrix.local", "broken.png")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        # Should not raise
        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

    @pytest.mark.asyncio
    async def test_download_exception_handled(self, tmp_path):
        """Unexpected exception during download is handled gracefully."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(side_effect=Exception("Connection reset"))

        event = make_media_event("@user:matrix.local", "oops.jpg")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        # Should not raise — bot should catch and notify
        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)


# --- Guards ---


class TestMediaGuards:
    """Test that media events are filtered like text events."""

    @pytest.mark.asyncio
    async def test_skips_during_initial_sync(self, tmp_path):
        """Media events during initial sync are ignored."""
        bot, agent, client = await _make_bot(tmp_path)
        bot._synced = False

        event = make_media_event("@user:matrix.local", "photo.jpg")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        agent.handle_input.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_own_media_messages(self, tmp_path):
        """Media from the bot's own user_id is ignored."""
        bot, agent, client = await _make_bot(tmp_path)

        event = make_media_event("@agent:matrix.local", "selfie.jpg")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        agent.handle_input.assert_not_awaited()
        client.download.assert_not_awaited()


# --- Refactor: Shared Processing ---


class TestSharedProcessing:
    """Test that _process_message is shared between text and media handlers."""

    @pytest.mark.asyncio
    async def test_process_message_exists(self, tmp_path):
        """MatrixBot has a _process_message method."""
        bot, _, _ = await _make_bot(tmp_path)
        assert hasattr(bot, '_process_message'), "Missing _process_message method"
        assert callable(bot._process_message)

    @pytest.mark.asyncio
    async def test_media_handler_delegates_to_process_message(self, tmp_path):
        """_handle_media_message calls _process_message after download."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_response(b"data"))

        event = make_media_event("@user:matrix.local", "test.jpg")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        with patch.object(bot, '_process_message', new_callable=AsyncMock) as mock_process:
            await bot._handle_media_message(room, event)
            # Drain background tasks fired by handler
            if hasattr(bot, "_background_tasks"):
                await asyncio.gather(*bot._background_tasks)
            mock_process.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_text_handler_delegates_to_process_message(self, tmp_path):
        """_handle_room_message calls _process_message for regular messages."""
        bot, agent, client = await _make_bot(tmp_path)

        event = MagicMock()
        event.sender = "@user:matrix.local"
        event.body = "hello"
        event.event_id = "$text1"
        event.server_timestamp = 1000000

        room = MagicMock()
        room.room_id = "!test:matrix.local"

        with patch.object(bot, '_process_message', new_callable=AsyncMock) as mock_process:
            await bot._handle_room_message(room, event)
            # Drain background tasks fired by handler
            if hasattr(bot, "_background_tasks"):
                await asyncio.gather(*bot._background_tasks)
            mock_process.assert_awaited_once()


# --- Integration: Multiple Media Types ---


class TestMediaTypes:
    """Test that different media types are handled uniformly."""

    @pytest.mark.asyncio
    async def test_audio_event(self, tmp_path):
        """Audio events are downloaded and tagged correctly."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_response(b"audio"))

        event = make_media_event("@user:matrix.local", "voice.ogg",
                                 mimetype="audio/ogg", event_type="audio")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        message = agent.handle_input.call_args[0][0]
        assert "audio/ogg" in message

    @pytest.mark.asyncio
    async def test_video_event(self, tmp_path):
        """Video events are downloaded and tagged correctly."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_response(b"video"))

        event = make_media_event("@user:matrix.local", "clip.mp4",
                                 mimetype="video/mp4", event_type="video")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        message = agent.handle_input.call_args[0][0]
        assert "video/mp4" in message

    @pytest.mark.asyncio
    async def test_file_event(self, tmp_path):
        """Generic file events are downloaded and tagged correctly."""
        bot, agent, client = await _make_bot(tmp_path)
        client.download = AsyncMock(return_value=make_download_response(b"pdf"))

        event = make_media_event("@user:matrix.local", "report.pdf",
                                 mimetype="application/pdf", event_type="file")
        room = MagicMock()
        room.room_id = "!test:matrix.local"

        await bot._handle_media_message(room, event)
        # Drain background tasks fired by handler
        if hasattr(bot, "_background_tasks"):
            await asyncio.gather(*bot._background_tasks)

        message = agent.handle_input.call_args[0][0]
        assert "application/pdf" in message
