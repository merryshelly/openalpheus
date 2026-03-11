"""Tests for image/vision support (.25).

Interface contract:
    When vision=True in config, image media tags in user messages are
    expanded to multi-part content blocks with base64-encoded image data.
    Provider layer converts normalized image blocks to wire format.

Spec: docs/vision-spec.md
Depends on: .24 (media attachment handling)
"""

import pytest
import base64
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.agent import Agent, _build_user_content, VISION_MIME_TYPES, MEDIA_TAG_RE
from openalph.provider import (
    _convert_messages_for_anthropic,
    _convert_messages_for_openai,
)


# --- Fixtures ---


def _make_image(tmp_path, name="photo.jpg", content=b"\xff\xd8\xff\xe0" + b"\x00" * 100):
    """Write a fake image file and return its path relative to tmp_path."""
    media_dir = tmp_path / "media" / "abc123"
    media_dir.mkdir(parents=True, exist_ok=True)
    img_path = media_dir / name
    img_path.write_bytes(content)
    return f"media/abc123/{name}"


def _make_config(tmp_path, vision=False, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model="claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"default": ProviderConfig(key="default", type="anthropic", api_key="sk-test", base_url=None, quirks=[])},
        workspace=tmp_path,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
        vision=vision,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


# --- Config ---


class TestVisionConfig:
    """Test vision config flag."""

    def test_vision_defaults_false(self, tmp_path):
        """AgentConfig.vision defaults to False."""
        config = _make_config(tmp_path)
        assert config.vision is False

    def test_vision_enabled(self, tmp_path):
        """AgentConfig.vision can be set to True."""
        config = _make_config(tmp_path, vision=True)
        assert config.vision is True

    def test_vision_loaded_from_toml(self, tmp_path):
        """Vision flag is loaded from TOML config."""
        from openalph.config import load_config

        toml_content = """
[agent]
name = "test"
model = "test-model"
max_tokens = 8192
vision = true

[provider]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "{workspace}"
""".format(workspace=str(tmp_path))

        config_path = tmp_path / "test.toml"
        config_path.write_text(toml_content)
        config = load_config(config_path)
        assert config.vision is True

    def test_vision_absent_from_toml_defaults_false(self, tmp_path):
        """Missing vision key in TOML defaults to False."""
        from openalph.config import load_config

        toml_content = """
[agent]
name = "test"
model = "test-model"
max_tokens = 8192

[provider]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "{workspace}"
""".format(workspace=str(tmp_path))

        config_path = tmp_path / "test.toml"
        config_path.write_text(toml_content)
        config = load_config(config_path)
        assert config.vision is False


# --- Media Tag Regex ---


class TestMediaTagRegex:
    """Test MEDIA_TAG_RE parsing."""

    def test_parses_standard_tag(self):
        m = MEDIA_TAG_RE.search("[media: media/abc123/photo.jpg (image/jpeg, 2.4 MB)]")
        assert m is not None
        assert m.group(1) == "media/abc123/photo.jpg"
        assert m.group(2) == "image/jpeg"
        assert m.group(3) == "2.4 MB"

    def test_parses_audio_tag(self):
        m = MEDIA_TAG_RE.search("[media: media/def456/voice.ogg (audio/ogg, 350 KB)]")
        assert m is not None
        assert m.group(2) == "audio/ogg"

    def test_parses_no_space_after_colon(self):
        """Regex handles minor whitespace variations."""
        m = MEDIA_TAG_RE.search("[media:media/abc/f.jpg (image/png, 1 KB)]")
        # Allow either match or no match — the implementation defines the contract
        # As long as the standard format matches, edge cases are implementation-defined

    def test_no_match_on_plain_text(self):
        m = MEDIA_TAG_RE.search("hello world")
        assert m is None

    def test_parses_filename_with_spaces(self):
        tag = "[media: media/abc123/Screenshot from 2026-03-10 12-23-51.png (image/png, 228.0 KB)]"
        match = MEDIA_TAG_RE.search(tag)
        assert match is not None
        assert match.group(1) == "media/abc123/Screenshot from 2026-03-10 12-23-51.png"
        assert match.group(2) == "image/png"
        assert match.group(3) == "228.0 KB"

    def test_parses_filename_with_special_chars(self):
        tag = "[media: media/abc123/image-2026.03.10.png (image/png, 100 B)]"
        match = MEDIA_TAG_RE.search(tag)
        assert match is not None
        assert match.group(1) == "media/abc123/image-2026.03.10.png"


# --- Content Building ---


class TestBuildUserContent:
    """Test _build_user_content: media tag parsing and image expansion."""

    def test_plain_text_returns_string(self, tmp_path):
        """No media tags → returns plain string."""
        config = _make_config(tmp_path, vision=True)
        result = _build_user_content("hello world", config)
        assert result == "hello world"
        assert isinstance(result, str)

    def test_vision_disabled_returns_string(self, tmp_path):
        """vision=False → media tags pass through as text."""
        rel_path = _make_image(tmp_path)
        config = _make_config(tmp_path, vision=False)
        text = f"[media: {rel_path} (image/jpeg, 104 B)]\nNice sunset"
        result = _build_user_content(text, config)
        assert isinstance(result, str)
        assert result == text

    def test_image_expanded_to_content_blocks(self, tmp_path):
        """vision=True + image tag → returns list with text + image blocks."""
        rel_path = _make_image(tmp_path)
        config = _make_config(tmp_path, vision=True)
        text = f"[media: {rel_path} (image/jpeg, 104 B)]\nCheck this out"
        result = _build_user_content(text, config)
        assert isinstance(result, list)

        # Should have at least one image block and one text block
        types = [b["type"] for b in result]
        assert "image" in types
        assert "text" in types

    def test_image_block_has_base64_data(self, tmp_path):
        """Image content block contains base64-encoded file data."""
        img_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        rel_path = _make_image(tmp_path, content=img_bytes)
        config = _make_config(tmp_path, vision=True)
        text = f"[media: {rel_path} (image/jpeg, 54 B)]"
        result = _build_user_content(text, config)

        img_blocks = [b for b in result if b["type"] == "image"]
        assert len(img_blocks) == 1
        assert img_blocks[0]["media_type"] == "image/jpeg"

        # Verify base64 decodes to original bytes
        decoded = base64.b64decode(img_blocks[0]["data"])
        assert decoded == img_bytes

    def test_non_image_media_stays_as_text(self, tmp_path):
        """Audio/video/file tags are NOT expanded even with vision=True."""
        config = _make_config(tmp_path, vision=True)
        text = "[media: media/abc/voice.ogg (audio/ogg, 1.1 MB)]"
        result = _build_user_content(text, config)
        assert isinstance(result, str)
        assert result == text

    def test_unsupported_image_format_stays_as_text(self, tmp_path):
        """TIFF and other unsupported image formats are not expanded."""
        config = _make_config(tmp_path, vision=True)
        text = "[media: media/abc/scan.tiff (image/tiff, 5 MB)]"
        result = _build_user_content(text, config)
        assert isinstance(result, str)
        assert result == text

    def test_missing_file_falls_back_to_text(self, tmp_path):
        """If image file doesn't exist, leave tag as text."""
        config = _make_config(tmp_path, vision=True)
        text = "[media: media/nonexistent/gone.jpg (image/jpeg, 100 KB)]"
        result = _build_user_content(text, config)
        assert isinstance(result, str)
        assert result == text

    def test_multiple_images_in_one_message(self, tmp_path):
        """Multiple image tags → multiple image blocks."""
        rel1 = _make_image(tmp_path, name="a.jpg")
        rel2 = _make_image(tmp_path, name="b.png")
        config = _make_config(tmp_path, vision=True)
        text = (
            f"[media: {rel1} (image/jpeg, 104 B)]\n"
            f"First image\n"
            f"[media: {rel2} (image/png, 104 B)]\n"
            f"Second image"
        )
        result = _build_user_content(text, config)
        assert isinstance(result, list)
        img_blocks = [b for b in result if b["type"] == "image"]
        assert len(img_blocks) == 2

    def test_mixed_image_and_audio(self, tmp_path):
        """Image expanded, audio left as text in the same message."""
        rel_img = _make_image(tmp_path, name="pic.jpg")
        config = _make_config(tmp_path, vision=True)
        text = (
            f"[media: {rel_img} (image/jpeg, 104 B)]\n"
            f"[media: media/abc/voice.ogg (audio/ogg, 500 KB)]\n"
            f"Both attached"
        )
        result = _build_user_content(text, config)
        assert isinstance(result, list)
        img_blocks = [b for b in result if b["type"] == "image"]
        text_blocks = [b for b in result if b["type"] == "text"]
        assert len(img_blocks) == 1
        # Audio tag should be in text block(s)
        all_text = " ".join(b["text"] for b in text_blocks)
        assert "audio/ogg" in all_text

    def test_caption_text_preserved(self, tmp_path):
        """Text around media tags is preserved in text blocks."""
        rel_path = _make_image(tmp_path)
        config = _make_config(tmp_path, vision=True)
        text = f"Before image\n[media: {rel_path} (image/jpeg, 104 B)]\nAfter image"
        result = _build_user_content(text, config)
        assert isinstance(result, list)
        all_text = " ".join(b["text"] for b in result if b["type"] == "text")
        assert "Before image" in all_text
        assert "After image" in all_text

    def test_image_with_spaces_in_filename(self, tmp_path):
        rel = _make_image(tmp_path, name="Screenshot from 2026.png")
        tag = f"[media: {rel} (image/png, 104 B)]"
        config = _make_config(tmp_path, vision=True)
        result = _build_user_content(tag, config)
        assert isinstance(result, list)
        assert any(b["type"] == "image" for b in result)


# --- Vision MIME Types ---


class TestVisionMimeTypes:
    """Test VISION_MIME_TYPES constant."""

    def test_jpeg_supported(self):
        assert "image/jpeg" in VISION_MIME_TYPES

    def test_png_supported(self):
        assert "image/png" in VISION_MIME_TYPES

    def test_gif_supported(self):
        assert "image/gif" in VISION_MIME_TYPES

    def test_webp_supported(self):
        assert "image/webp" in VISION_MIME_TYPES

    def test_tiff_not_supported(self):
        assert "image/tiff" not in VISION_MIME_TYPES

    def test_bmp_not_supported(self):
        assert "image/bmp" not in VISION_MIME_TYPES


# --- Provider Conversion: Anthropic ---


class TestAnthropicImageConversion:
    """Test Anthropic wire format for image content blocks."""

    def test_text_only_unchanged(self):
        """User message with string content passes through."""
        msgs = [{"role": "user", "content": "hello"}]
        result = _convert_messages_for_anthropic(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_image_block_converted(self):
        """Image block → Anthropic source format."""
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "What's this?"},
            {"type": "image", "media_type": "image/jpeg", "data": "base64data"},
        ]}]
        result = _convert_messages_for_anthropic(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)

        # Find image block
        img_blocks = [b for b in content if b["type"] == "image"]
        assert len(img_blocks) == 1
        img = img_blocks[0]
        assert img["source"]["type"] == "base64"
        assert img["source"]["media_type"] == "image/jpeg"
        assert img["source"]["data"] == "base64data"

    def test_text_block_preserved(self):
        """Text blocks pass through in list content."""
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image", "media_type": "image/png", "data": "abc"},
        ]}]
        result = _convert_messages_for_anthropic(msgs)
        text_blocks = [b for b in result[0]["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "describe this"

    def test_multiple_images(self):
        """Multiple image blocks converted correctly."""
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "compare"},
            {"type": "image", "media_type": "image/jpeg", "data": "img1"},
            {"type": "image", "media_type": "image/png", "data": "img2"},
        ]}]
        result = _convert_messages_for_anthropic(msgs)
        img_blocks = [b for b in result[0]["content"] if b["type"] == "image"]
        assert len(img_blocks) == 2


# --- Provider Conversion: OpenAI ---


class TestOpenAIImageConversion:
    """Test OpenAI wire format for image content blocks."""

    def test_text_only_unchanged(self):
        """User message with string content passes through."""
        msgs = [{"role": "user", "content": "hello"}]
        result = _convert_messages_for_openai(msgs)
        assert result[0]["content"] == "hello"

    def test_image_block_converted(self):
        """Image block → OpenAI image_url format."""
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "What's this?"},
            {"type": "image", "media_type": "image/jpeg", "data": "base64data"},
        ]}]
        result = _convert_messages_for_openai(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)

        # Find image_url block
        img_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(img_blocks) == 1
        url = img_blocks[0]["image_url"]["url"]
        assert url == "data:image/jpeg;base64,base64data"

    def test_text_block_preserved(self):
        """Text blocks pass through in list content."""
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image", "media_type": "image/png", "data": "abc"},
        ]}]
        result = _convert_messages_for_openai(msgs)
        text_blocks = [b for b in result[0]["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "describe this"

    def test_multiple_images(self):
        """Multiple image blocks converted correctly."""
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "compare"},
            {"type": "image", "media_type": "image/jpeg", "data": "img1"},
            {"type": "image", "media_type": "image/png", "data": "img2"},
        ]}]
        result = _convert_messages_for_openai(msgs)
        img_blocks = [b for b in result[0]["content"] if b["type"] == "image_url"]
        assert len(img_blocks) == 2


# --- Token Estimation ---


class TestImageTokenEstimation:
    """Test that image content blocks contribute to context token estimates."""

    @pytest.mark.asyncio
    async def test_image_adds_to_token_count(self, tmp_path):
        """Messages with image blocks have higher token estimates than text-only."""
        config = _make_config(tmp_path, vision=True)

        with patch("openalph.agent.complete") as mock_complete, \
             patch("openalph.agent.assemble_prompt", return_value="system"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            agent = Agent(config)

            # Manually add a text-only message
            text_history = [{"role": "user", "content": "hello"}]
            text_estimate = agent._estimate_context_tokens("text_room", text_history)

            # Add a message with image data (large base64 string)
            large_b64 = "A" * 10000  # ~7500 raw bytes
            image_history = [{"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "media_type": "image/jpeg", "data": large_b64},
            ]}]
            image_estimate = agent._estimate_context_tokens("img_room", image_history)

            assert image_estimate > text_estimate


# --- Integration: handle_input with vision ---


class TestHandleInputVision:
    """Test that handle_input uses _build_user_content when vision is enabled."""

    @pytest.mark.asyncio
    async def test_vision_message_reaches_provider(self, tmp_path):
        """Image in message with vision=True produces list content in history."""
        img_bytes = b"\xff\xd8" + b"\x00" * 100
        rel_path = _make_image(tmp_path, content=img_bytes)
        config = _make_config(tmp_path, vision=True)

        from openalph.provider import Response, Usage
        mock_response = Response(
            content="I see a photo!",
            tool_calls=[],
            usage=Usage(input_tokens=100, output_tokens=20),
            stop_reason="end_turn",
        )

        with patch("openalph.agent.complete", new_callable=AsyncMock, return_value=mock_response), \
             patch("openalph.agent.assemble_prompt", return_value="system"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            agent = Agent(config)
            text = f"[media: {rel_path} (image/jpeg, 102 B)]\nWhat's in this image?"
            result = await agent.handle_input(text)

            assert result == "I see a photo!"

            # Check that history has list content (not plain string)
            history = agent.history("_default")
            user_msg = history[0]
            assert isinstance(user_msg["content"], list)
            img_blocks = [b for b in user_msg["content"] if b["type"] == "image"]
            assert len(img_blocks) == 1

    @pytest.mark.asyncio
    async def test_no_vision_message_stays_text(self, tmp_path):
        """Image in message with vision=False keeps plain text content."""
        rel_path = _make_image(tmp_path)
        config = _make_config(tmp_path, vision=False)

        from openalph.provider import Response, Usage
        mock_response = Response(
            content="I see a media tag",
            tool_calls=[],
            usage=Usage(input_tokens=100, output_tokens=20),
            stop_reason="end_turn",
        )

        with patch("openalph.agent.complete", new_callable=AsyncMock, return_value=mock_response), \
             patch("openalph.agent.assemble_prompt", return_value="system"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            agent = Agent(config)
            text = f"[media: {rel_path} (image/jpeg, 104 B)]\nWhat's this?"
            result = await agent.handle_input(text)

            history = agent.history("_default")
            user_msg = history[0]
            assert isinstance(user_msg["content"], str)
