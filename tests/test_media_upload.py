"""Tests for media upload — agent sends files to Matrix rooms (kdsn.76).

Interface contract:
    send_media(path, caption, max_upload_bytes, upload_callback) -> ToolResult
    MatrixBot.upload_and_send(room_id, file_path, content_type, filename, caption)
    execute_tool(..., callbacks={"send_media": fn}) passes callback to send_media

Spec: memory/projects/openalph/media-upload-spec.md
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from openalph.tools import execute_tool, discover_tools, BUILTIN_TOOLS
from openalph.tools.media import send_media
from openalph.config import AgentConfig, ProviderConfig


# --- Fixtures ---


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


# --- send_media tool executor ---


class TestSendMediaFileNotFound:

    @pytest.mark.asyncio
    async def test_missing_file_returns_error(self, tmp_path):
        """Nonexistent file → is_error=True with path info."""
        result = await send_media(
            path=str(tmp_path / "does_not_exist.mp3"),
            upload_callback=AsyncMock(),
        )
        assert result.is_error is True
        assert "not found" in result.content.lower()


class TestSendMediaFileTooLarge:

    @pytest.mark.asyncio
    async def test_oversized_file_returns_error(self, tmp_path):
        """File exceeding max_upload_bytes → is_error with size info."""
        big_file = tmp_path / "huge.mp3"
        big_file.write_bytes(b"\x00" * 1000)

        result = await send_media(
            path=str(big_file),
            max_upload_bytes=500,
            upload_callback=AsyncMock(),
        )
        assert result.is_error is True
        assert "too large" in result.content.lower() or "limit" in result.content.lower()


class TestSendMediaEmptyFile:

    @pytest.mark.asyncio
    async def test_empty_file_returns_error(self, tmp_path):
        """Zero-byte file → is_error=True."""
        empty = tmp_path / "empty.mp3"
        empty.write_bytes(b"")

        result = await send_media(
            path=str(empty),
            upload_callback=AsyncMock(),
        )
        assert result.is_error is True
        assert "empty" in result.content.lower()


class TestSendMediaNotAFile:

    @pytest.mark.asyncio
    async def test_directory_returns_error(self, tmp_path):
        """Directory path → is_error=True."""
        subdir = tmp_path / "somedir"
        subdir.mkdir()

        result = await send_media(
            path=str(subdir),
            upload_callback=AsyncMock(),
        )
        assert result.is_error is True
        assert "not a file" in result.content.lower()


class TestSendMediaNoCallback:

    @pytest.mark.asyncio
    async def test_no_callback_returns_error_with_path(self, tmp_path):
        """CLI mode (no callback) → is_error=True, mentions saved path."""
        audio = tmp_path / "response.mp3"
        audio.write_bytes(b"\xff\xfb\x90\x00" * 100)

        result = await send_media(
            path=str(audio),
            upload_callback=None,
        )
        assert result.is_error is True
        assert str(audio) in result.content or "not available" in result.content.lower()


class TestSendMediaMimeDetection:

    @pytest.mark.asyncio
    async def test_mp3_detected(self, tmp_path):
        """*.mp3 → audio/mpeg content type passed to callback."""
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\xff\xfb\x90\x00" * 100)

        callback = AsyncMock()
        await send_media(path=str(audio), upload_callback=callback)
        callback.assert_awaited_once()
        _, kwargs = callback.call_args
        assert kwargs["content_type"] == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_png_detected(self, tmp_path):
        """*.png → image/png content type."""
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 100)

        callback = AsyncMock()
        await send_media(path=str(img), upload_callback=callback)
        _, kwargs = callback.call_args
        assert kwargs["content_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_unknown_extension_defaults_to_octet_stream(self, tmp_path):
        """Unknown extension → application/octet-stream."""
        data = tmp_path / "data.xyz123"
        data.write_bytes(b"\x00" * 100)

        callback = AsyncMock()
        await send_media(path=str(data), upload_callback=callback)
        _, kwargs = callback.call_args
        assert kwargs["content_type"] == "application/octet-stream"


class TestSendMediaSuccess:

    @pytest.mark.asyncio
    async def test_success_calls_callback_with_correct_args(self, tmp_path):
        """Successful send → callback called with file_path, content_type, filename."""
        audio = tmp_path / "hello.mp3"
        audio.write_bytes(b"\xff\xfb\x90\x00" * 100)

        callback = AsyncMock()
        result = await send_media(path=str(audio), upload_callback=callback)

        assert result.is_error is False
        assert "hello.mp3" in result.content
        callback.assert_awaited_once()
        _, kwargs = callback.call_args
        assert kwargs["file_path"] == audio
        assert kwargs["filename"] == "hello.mp3"

    @pytest.mark.asyncio
    async def test_success_message_includes_size(self, tmp_path):
        """Success message includes human-readable file size."""
        audio = tmp_path / "voice.mp3"
        audio.write_bytes(b"\x00" * 2048)

        callback = AsyncMock()
        result = await send_media(path=str(audio), upload_callback=callback)
        assert "KB" in result.content or "B" in result.content

    @pytest.mark.asyncio
    async def test_caption_passed_through(self, tmp_path):
        """Caption parameter is forwarded to callback."""
        audio = tmp_path / "note.mp3"
        audio.write_bytes(b"\xff" * 100)

        callback = AsyncMock()
        await send_media(path=str(audio), caption="Here you go!", upload_callback=callback)
        _, kwargs = callback.call_args
        assert kwargs["caption"] == "Here you go!"


class TestSendMediaCallbackFailure:

    @pytest.mark.asyncio
    async def test_callback_exception_returns_error(self, tmp_path):
        """Callback raises → is_error=True with error info."""
        audio = tmp_path / "broken.mp3"
        audio.write_bytes(b"\xff" * 100)

        callback = AsyncMock(side_effect=RuntimeError("Upload timeout"))
        result = await send_media(path=str(audio), upload_callback=callback)

        assert result.is_error is True
        assert "Upload timeout" in result.content or "failed" in result.content.lower()


# --- execute_tool callbacks integration ---


class TestExecuteToolCallbacks:

    @pytest.mark.asyncio
    async def test_callbacks_parameter_accepted(self, tmp_path):
        """execute_tool accepts optional callbacks dict without error."""
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\xff" * 100)

        callback = AsyncMock()
        config = make_agent_config(tmp_path)

        result = await execute_tool(
            name="send_media",
            input={"path": str(audio)},
            tool_config={},
            agent_config=config,
            callbacks={"send_media": callback},
        )
        assert not result.is_error
        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_callbacks_still_works_for_other_tools(self, tmp_path):
        """Existing tools work fine when callbacks=None (backward compat)."""
        config = make_agent_config(tmp_path)

        result = await execute_tool(
            name="shell",
            input={"command": "echo hello"},
            tool_config={"default_timeout": 5, "max_output": 1000},
            agent_config=config,
        )
        assert "hello" in result.content

    @pytest.mark.asyncio
    async def test_send_media_without_callback_returns_error(self, tmp_path):
        """send_media dispatched with callbacks=None → CLI mode error."""
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\xff" * 100)

        config = make_agent_config(tmp_path)

        result = await execute_tool(
            name="send_media",
            input={"path": str(audio)},
            tool_config={},
            agent_config=config,
            callbacks=None,
        )
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_path_resolved_against_workspace(self, tmp_path):
        """Relative path in input is resolved against config.workspace."""
        media_dir = tmp_path / "media"
        media_dir.mkdir()
        audio = media_dir / "test.mp3"
        audio.write_bytes(b"\xff" * 100)

        callback = AsyncMock()
        config = make_agent_config(tmp_path)

        result = await execute_tool(
            name="send_media",
            input={"path": "media/test.mp3"},
            tool_config={},
            agent_config=config,
            callbacks={"send_media": callback},
        )
        assert not result.is_error
        _, kwargs = callback.call_args
        assert kwargs["file_path"] == audio


# --- Tool discovery ---


class TestSendMediaDiscovery:

    def test_send_media_in_builtin_tools(self):
        """send_media is registered in BUILTIN_TOOLS."""
        assert "send_media" in BUILTIN_TOOLS

    def test_discoverable_via_toml(self, tmp_path):
        """send_media.toml enables the tool."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "send_media.toml").write_text("")

        tools = discover_tools(tmp_path)
        names = {t.name for t in tools}
        assert "send_media" in names

    def test_schema_has_required_path(self):
        """Tool schema requires 'path' parameter."""
        schema = BUILTIN_TOOLS["send_media"]["parameters"]
        assert "path" in schema["required"]
        assert "path" in schema["properties"]

    def test_schema_has_optional_caption(self):
        """Tool schema has optional 'caption' parameter."""
        schema = BUILTIN_TOOLS["send_media"]["parameters"]
        assert "caption" in schema["properties"]
        assert "caption" not in schema.get("required", [])


# --- MatrixBot.upload_and_send ---


class TestUploadAndSend:
    """Tests for MatrixBot.upload_and_send method.

    These test the Matrix-specific upload logic using mocked nio client.
    """

    def _make_bot(self, tmp_path):
        """Create a MatrixBot with mocked internals."""
        from openalph.matrix import MatrixBot
        from openalph.config import MatrixConfig

        matrix_config = MatrixConfig(
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

        agent_config = make_agent_config(tmp_path, matrix=matrix_config)

        # Use __new__ to avoid __init__ side effects, then set required attrs
        bot = MatrixBot.__new__(MatrixBot)
        bot.client = AsyncMock()
        bot.agent = MagicMock()
        bot.agent.config = agent_config
        bot.config = matrix_config
        return bot

    @pytest.mark.asyncio
    async def test_audio_sends_m_audio(self, tmp_path):
        """audio/* MIME → m.audio msgtype."""
        bot = self._make_bot(tmp_path)
        audio = tmp_path / "hello.mp3"
        audio.write_bytes(b"\xff\xfb\x90" * 100)

        # Mock successful upload
        upload_resp = MagicMock()
        upload_resp.content_uri = "mxc://matrix.local/abc123"
        bot.client.upload = AsyncMock(return_value=(upload_resp, None))
        bot._room_send_with_retry = AsyncMock()

        await bot.upload_and_send("!room:matrix.local", audio, "audio/mpeg", "hello.mp3")

        bot._room_send_with_retry.assert_awaited_once()
        _, kwargs = bot._room_send_with_retry.call_args
        content = kwargs.get("content") or bot._room_send_with_retry.call_args[0][1]
        assert content["msgtype"] == "m.audio"
        assert content["url"] == "mxc://matrix.local/abc123"

    @pytest.mark.asyncio
    async def test_image_sends_m_image(self, tmp_path):
        """image/* MIME → m.image msgtype."""
        bot = self._make_bot(tmp_path)
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 100)

        upload_resp = MagicMock()
        upload_resp.content_uri = "mxc://matrix.local/img456"
        bot.client.upload = AsyncMock(return_value=(upload_resp, None))
        bot._room_send_with_retry = AsyncMock()

        await bot.upload_and_send("!room:matrix.local", img, "image/png", "photo.png")

        content = bot._room_send_with_retry.call_args[0][1]
        assert content["msgtype"] == "m.image"

    @pytest.mark.asyncio
    async def test_unknown_mime_sends_m_file(self, tmp_path):
        """Unknown MIME → m.file msgtype."""
        bot = self._make_bot(tmp_path)
        data = tmp_path / "report.pdf"
        data.write_bytes(b"%PDF" + b"\x00" * 100)

        upload_resp = MagicMock()
        upload_resp.content_uri = "mxc://matrix.local/pdf789"
        bot.client.upload = AsyncMock(return_value=(upload_resp, None))
        bot._room_send_with_retry = AsyncMock()

        await bot.upload_and_send("!room:matrix.local", data, "application/pdf", "report.pdf")

        content = bot._room_send_with_retry.call_args[0][1]
        assert content["msgtype"] == "m.file"

    @pytest.mark.asyncio
    async def test_upload_error_raises(self, tmp_path):
        """nio upload error → RuntimeError raised."""
        from nio import UploadError

        bot = self._make_bot(tmp_path)
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\xff" * 100)

        error_resp = MagicMock(spec=UploadError)
        error_resp.message = "Server rejected upload"
        bot.client.upload = AsyncMock(return_value=(error_resp, None))

        with pytest.raises(RuntimeError, match="[Uu]pload failed"):
            await bot.upload_and_send("!room:matrix.local", audio, "audio/mpeg", "test.mp3")

    @pytest.mark.asyncio
    async def test_caption_sets_body_and_filename(self, tmp_path):
        """With caption → body=caption, filename=original name."""
        bot = self._make_bot(tmp_path)
        audio = tmp_path / "note.mp3"
        audio.write_bytes(b"\xff" * 100)

        upload_resp = MagicMock()
        upload_resp.content_uri = "mxc://matrix.local/cap123"
        bot.client.upload = AsyncMock(return_value=(upload_resp, None))
        bot._room_send_with_retry = AsyncMock()

        await bot.upload_and_send(
            "!room:matrix.local", audio, "audio/mpeg", "note.mp3",
            caption="Here's the voice note"
        )

        content = bot._room_send_with_retry.call_args[0][1]
        assert content["body"] == "Here's the voice note"
        assert content["filename"] == "note.mp3"

    @pytest.mark.asyncio
    async def test_no_caption_uses_filename_as_body(self, tmp_path):
        """Without caption → body=filename, no separate filename field."""
        bot = self._make_bot(tmp_path)
        audio = tmp_path / "output.mp3"
        audio.write_bytes(b"\xff" * 100)

        upload_resp = MagicMock()
        upload_resp.content_uri = "mxc://matrix.local/nc456"
        bot.client.upload = AsyncMock(return_value=(upload_resp, None))
        bot._room_send_with_retry = AsyncMock()

        await bot.upload_and_send("!room:matrix.local", audio, "audio/mpeg", "output.mp3")

        content = bot._room_send_with_retry.call_args[0][1]
        assert content["body"] == "output.mp3"
        assert "filename" not in content

    @pytest.mark.asyncio
    async def test_info_includes_mimetype_and_size(self, tmp_path):
        """Event content.info has mimetype and size."""
        bot = self._make_bot(tmp_path)
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\xff" * 500)

        upload_resp = MagicMock()
        upload_resp.content_uri = "mxc://matrix.local/info789"
        bot.client.upload = AsyncMock(return_value=(upload_resp, None))
        bot._room_send_with_retry = AsyncMock()

        await bot.upload_and_send("!room:matrix.local", audio, "audio/mpeg", "test.mp3")

        content = bot._room_send_with_retry.call_args[0][1]
        assert content["info"]["mimetype"] == "audio/mpeg"
        assert content["info"]["size"] == 500
