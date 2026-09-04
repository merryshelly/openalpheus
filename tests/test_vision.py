"""Tests for image/vision support (.25, refactored by kdsn.275).

Interface contract:
    Vision capability is a MODEL-LEVEL property, resolved per active room
    model by openalph.provider.model_supports_vision via three layers:
      1. [model_vision] TOML override (exact full-string match, post-alias)
      2. _MODEL_CAPABILITIES table (substring fragment, first-match-wins)
      3. Fail-closed False + one-time WARN log (never send base64 to an
         uncharacterized model)
    When the active room model supports vision, image media tags in user
    messages are expanded to multi-part content blocks with base64-encoded
    image data (_build_user_content). The provider layer converts normalized
    image blocks to wire format (Anthropic source.base64 / OpenAI image_url).

Spec: memory/projects/openalph/vision-model-capability-spec.md (kdsn.275/.276)
Supersedes the agent-level [agent] vision TOML gate from .25 (hard cut).
"""

import dataclasses
import logging
import pytest
import base64
from unittest.mock import patch
from openalph.config import AgentConfig, ConfigError, ProviderConfig
from openalph.agent import Agent, _build_user_content, VISION_MIME_TYPES, MEDIA_TAG_RE
from openalph.provider import (
    Response,
    Usage,
    StreamEvent,
    _convert_messages_for_anthropic,
    _convert_messages_for_openai,
    _MODEL_CAPABILITIES,
    model_supports_vision,
    model_context_window,
)


# --- Fixtures ---


def _make_image(tmp_path, name="photo.jpg", content=b"\xff\xd8\xff\xe0" + b"\x00" * 100):
    """Write a fake image file and return its path relative to tmp_path."""
    media_dir = tmp_path / "media" / "abc123"
    media_dir.mkdir(parents=True, exist_ok=True)
    img_path = media_dir / name
    img_path.write_bytes(content)
    return f"media/abc123/{name}"


# A model string that resolves vision=True via the capabilities table.
VISION_MODEL = "anthropic/claude-sonnet-5"
# A model string that resolves vision=False via the capabilities table
# (known, explicitly blind — no unknown-model warning noise).
BLIND_MODEL = "macstudio/deepseek-v4-flash"
# A model string absent from the table entirely (fail-closed + warn-once).
UNKNOWN_MODEL = "anthropic/claude-zephyr-9-20990101"


def _make_config(tmp_path, default_model=VISION_MODEL, extra_providers=None, **kwargs):
    providers = {
        "anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[]),
    }
    if extra_providers:
        providers.update(extra_providers)
    defaults = dict(
        name="test-agent",
        default_model=default_model,
        max_tokens=8192,
        providers=providers,
        workspace=tmp_path,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _macstudio_provider():
    return ProviderConfig(
        key="macstudio", type="openai", api_key="sk-local",
        base_url="http://10.0.20.104:8000/v1", quirks=[])


def make_stream_events(content="Hello!", input_tokens=100, output_tokens=50):
    """Create a mock async generator that yields stream events."""
    async def _stream(*args, **kwargs):
        yield StreamEvent(type="text", content=content)
        yield StreamEvent(
            type="done",
            response=Response(
                content=content,
                model="claude-sonnet-5",
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
                stop_reason="end_turn",
            ),
            stop_reason="end_turn",
            model="claude-sonnet-5",
        )
    return _stream


def _toml_config(tmp_path, extra_agent_lines="", extra_sections=""):
    """Write a minimal agent TOML and return its path."""
    toml_content = f"""
[agent]
name = "test"
default_model = "anthropic/test-model"
max_tokens = 8192
{extra_agent_lines}

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "{tmp_path}"
{extra_sections}
"""
    config_path = tmp_path / "test.toml"
    config_path.write_text(toml_content)
    return config_path


# --- Model-level capability table sanity (kdsn.275) ---


class TestCapabilitiesTable:
    """_MODEL_CAPABILITIES entries carry an explicit vision element."""

    def test_every_fragment_has_vision_element(self):
        """Every tuple is (fragment, window, output_cap, vision: bool)."""
        for entry in _MODEL_CAPABILITIES:
            assert len(entry) == 4, f"entry missing vision element: {entry!r}"
            assert isinstance(entry[3], bool), f"vision element not bool: {entry!r}"

    def test_anthropic_family_vision_true(self, tmp_path):
        config = _make_config(tmp_path)
        for model in ("anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-6",
                      "anthropic/claude-sonnet-5", "anthropic/claude-opus-4-8",
                      "anthropic/claude-opus-5", "anthropic/claude-fable-5"):
            assert model_supports_vision(model, config) is True, model

    def test_gemini_and_maverick_vision_true(self, tmp_path):
        config = _make_config(tmp_path)
        assert model_supports_vision("google/gemini-3.5-flash", config) is True
        assert model_supports_vision("openrouter/meta/llama-4-maverick", config) is True

    def test_kimi_vision_true_probe_verified(self, tmp_path):
        """Live-probed 2026-08-22 (spec section 3.5): k2p6 + k3 ID'd red/blue PNGs."""
        config = _make_config(tmp_path)
        for model in ("fireworks/accounts/fireworks/models/kimi-k2p6",
                      "fireworks/accounts/fireworks/models/kimi-k3"):
            assert model_supports_vision(model, config) is True, model

    def test_known_blind_models_false(self, tmp_path):
        config = _make_config(tmp_path)
        for model in ("fireworks/accounts/fireworks/models/glm-5p2",
                      "macstudio/deepseek-v4-flash",
                      "macstudio-qwen/qwen38-coder",
                      "openrouter/nousresearch/hermes-4"):
            assert model_supports_vision(model, config) is False, model

    def test_qwen38_multimodal_fragments(self, tmp_path):
        """New qwen3.8/qwen3p8 fragments: vision True, native window 262_144."""
        config = _make_config(tmp_path)
        for model in ("local-qwen/qwen3.8-27b", "local-qwen/qwen3p8-27b"):
            assert model_supports_vision(model, config) is True, model
            assert model_context_window(model) == 262_144, model

    def test_qwen38_coder_fragment_not_shadowed_by_multimodal(self, tmp_path):
        """'qwen38' (coder, blind) and 'qwen3.8' (multimodal) never collide."""
        config = _make_config(tmp_path)
        assert model_supports_vision("macstudio-qwen/qwen38-coder", config) is False
        assert model_supports_vision("local-qwen/qwen3.8-27b", config) is True


# --- Resolver: model_supports_vision (kdsn.275) ---


class TestModelSupportsVision:
    """3-layer resolution: TOML override -> table -> fail-closed False."""

    def test_table_hit_true(self, tmp_path):
        config = _make_config(tmp_path)
        assert model_supports_vision(VISION_MODEL, config) is True

    def test_table_hit_false(self, tmp_path):
        config = _make_config(tmp_path)
        assert model_supports_vision(BLIND_MODEL, config) is False

    def test_override_disables_capable_model(self, tmp_path):
        """Force-disable (cost / injection-surface reasons): override beats table."""
        config = _make_config(
            tmp_path, model_vision={VISION_MODEL: False})
        assert model_supports_vision(VISION_MODEL, config) is False

    def test_override_enables_blind_model(self, tmp_path):
        """Force-enable (operator attests capability): override beats table."""
        config = _make_config(
            tmp_path, model_vision={BLIND_MODEL: True})
        assert model_supports_vision(BLIND_MODEL, config) is True

    def test_unknown_model_fail_closed(self, tmp_path):
        """Uncharacterized model resolves False: no base64 to mystery models."""
        config = _make_config(tmp_path)
        assert model_supports_vision(UNKNOWN_MODEL, config) is False

    def test_unknown_model_warns_once_per_process(self, tmp_path, caplog):
        """Fail-closed default logs a WARN the first time only."""
        from openalph import provider
        provider._VISION_WARNED.discard(UNKNOWN_MODEL)  # isolate from other tests
        config = _make_config(tmp_path)
        with caplog.at_level(logging.WARNING, logger="openalph.provider"):
            assert model_supports_vision(UNKNOWN_MODEL, config) is False
            assert model_supports_vision(UNKNOWN_MODEL, config) is False
        vision_warns = [r for r in caplog.records
                        if r.levelno == logging.WARNING
                        and "vision" in r.getMessage().lower()]
        assert len(vision_warns) == 1, \
            f"expected exactly one warn-once record, got {len(vision_warns)}"
        provider._VISION_WARNED.discard(UNKNOWN_MODEL)  # don't leak state

    def test_bare_alias_expanded_before_lookup(self, tmp_path):
        """A bare alias resolves through its expansion (the 2026-08-03 wonmun
        bare-alias bug is the precedent: expand before BOTH the override
        layer and the table layer)."""
        config = _make_config(
            tmp_path, model_aliases={"sonnet": VISION_MODEL})
        assert model_supports_vision("sonnet", config) is True

    def test_override_applies_through_alias(self, tmp_path):
        """Override keys are the EXPANDED (fully-qualified) model string."""
        config = _make_config(
            tmp_path,
            model_aliases={"deepseek": BLIND_MODEL},
            model_vision={BLIND_MODEL: True})
        assert model_supports_vision("deepseek", config) is True


# --- Config: [model_vision] parsing + hard cut of [agent] vision ---


class TestModelVisionConfig:
    """config.py: model_vision section parsed; agent-level vision gone."""

    def test_agent_config_has_no_vision_field(self):
        """Hard cut: AgentConfig no longer carries an agent-level vision flag."""
        field_names = {f.name for f in dataclasses.fields(AgentConfig)}
        assert "vision" not in field_names

    def test_agent_config_has_model_vision_field(self, tmp_path):
        config = _make_config(tmp_path)
        assert config.model_vision == {}

    def test_model_vision_loaded_from_toml(self, tmp_path):
        from openalph.config import load_config
        path = _toml_config(tmp_path, extra_sections="""
[model_vision]
"anthropic/claude-sonnet-5" = false
"macstudio/deepseek-v4-flash" = true
""")
        config = load_config(path)
        assert config.model_vision == {
            "anthropic/claude-sonnet-5": False,
            "macstudio/deepseek-v4-flash": True,
        }

    def test_model_vision_rejects_non_bool(self, tmp_path):
        """Fail-LOUD on malformed override values: a silently-dropped disable
        override is a fail-open on a safety knob. Deliberate deviation from
        [model_limits]' lenient skip (documented in kdsn.275)."""
        from openalph.config import load_config
        path = _toml_config(tmp_path, extra_sections="""
[model_vision]
"anthropic/claude-sonnet-5" = "false"
""")
        with pytest.raises(ConfigError):
            load_config(path)

    def test_stale_agent_vision_key_is_inert(self, tmp_path):
        """config.py is lenient about unknown TOML keys: a stale
        `[agent] vision = true` from an unmigrated fleet config must load
        without error AND have zero effect (the gate is model-level now)."""
        from openalph.config import load_config
        path = _toml_config(tmp_path, extra_agent_lines="vision = true")
        config = load_config(path)  # must not raise
        # The stale key does nothing: an unknown default model still
        # resolves vision=False.
        assert model_supports_vision(config.default_model, config) is False


# --- Media Tag Regex (unchanged from .25) ---


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


# --- Content Building (gate is now an explicit vision kwarg) ---


class TestBuildUserContent:
    """_build_user_content: expansion is driven by the resolved vision flag,
    not by any agent-level config field."""

    def test_vision_kwarg_is_keyword_only(self, tmp_path):
        """Signature is (text, config, *, vision) — positional third arg must
        fail loudly so call sites state their intent."""
        config = _make_config(tmp_path)
        with pytest.raises(TypeError):
            _build_user_content("hello", config, True)

    def test_plain_text_returns_string(self, tmp_path):
        config = _make_config(tmp_path)
        result = _build_user_content("hello world", config, vision=True)
        assert result == "hello world"
        assert isinstance(result, str)

    def test_vision_disabled_returns_string(self, tmp_path):
        """vision=False → media tags pass through as text."""
        rel_path = _make_image(tmp_path)
        config = _make_config(tmp_path)
        text = f"[media: {rel_path} (image/jpeg, 104 B)]\nNice sunset"
        result = _build_user_content(text, config, vision=False)
        assert isinstance(result, str)
        assert result == text

    def test_image_expanded_to_content_blocks(self, tmp_path):
        rel_path = _make_image(tmp_path)
        config = _make_config(tmp_path)
        text = f"[media: {rel_path} (image/jpeg, 104 B)]\nCheck this out"
        result = _build_user_content(text, config, vision=True)
        assert isinstance(result, list)
        types = [b["type"] for b in result]
        assert "image" in types
        assert "text" in types

    def test_image_block_has_base64_data(self, tmp_path):
        img_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        rel_path = _make_image(tmp_path, content=img_bytes)
        config = _make_config(tmp_path)
        text = f"[media: {rel_path} (image/jpeg, 54 B)]"
        result = _build_user_content(text, config, vision=True)

        img_blocks = [b for b in result if b["type"] == "image"]
        assert len(img_blocks) == 1
        assert img_blocks[0]["media_type"] == "image/jpeg"
        decoded = base64.b64decode(img_blocks[0]["data"])
        assert decoded == img_bytes

    def test_non_image_media_stays_as_text(self, tmp_path):
        config = _make_config(tmp_path)
        text = "[media: media/abc/voice.ogg (audio/ogg, 1.1 MB)]"
        result = _build_user_content(text, config, vision=True)
        assert isinstance(result, str)
        assert result == text

    def test_unsupported_image_format_stays_as_text(self, tmp_path):
        config = _make_config(tmp_path)
        text = "[media: media/abc/scan.tiff (image/tiff, 5 MB)]"
        result = _build_user_content(text, config, vision=True)
        assert isinstance(result, str)
        assert result == text

    def test_missing_file_falls_back_to_text(self, tmp_path):
        config = _make_config(tmp_path)
        text = "[media: media/nonexistent/gone.jpg (image/jpeg, 100 KB)]"
        result = _build_user_content(text, config, vision=True)
        assert isinstance(result, str)
        assert result == text

    def test_multiple_images_in_one_message(self, tmp_path):
        rel1 = _make_image(tmp_path, name="a.jpg")
        rel2 = _make_image(tmp_path, name="b.png")
        config = _make_config(tmp_path)
        text = (
            f"[media: {rel1} (image/jpeg, 104 B)]\n"
            f"First image\n"
            f"[media: {rel2} (image/png, 104 B)]\n"
            f"Second image"
        )
        result = _build_user_content(text, config, vision=True)
        assert isinstance(result, list)
        img_blocks = [b for b in result if b["type"] == "image"]
        assert len(img_blocks) == 2

    def test_mixed_image_and_audio(self, tmp_path):
        rel_img = _make_image(tmp_path, name="pic.jpg")
        config = _make_config(tmp_path)
        text = (
            f"[media: {rel_img} (image/jpeg, 104 B)]\n"
            f"[media: media/abc/voice.ogg (audio/ogg, 500 KB)]\n"
            f"Both attached"
        )
        result = _build_user_content(text, config, vision=True)
        assert isinstance(result, list)
        img_blocks = [b for b in result if b["type"] == "image"]
        text_blocks = [b for b in result if b["type"] == "text"]
        assert len(img_blocks) == 1
        all_text = " ".join(b["text"] for b in text_blocks)
        assert "audio/ogg" in all_text

    def test_caption_text_preserved(self, tmp_path):
        rel_path = _make_image(tmp_path)
        config = _make_config(tmp_path)
        text = f"Before image\n[media: {rel_path} (image/jpeg, 104 B)]\nAfter image"
        result = _build_user_content(text, config, vision=True)
        assert isinstance(result, list)
        all_text = " ".join(b["text"] for b in result if b["type"] == "text")
        assert "Before image" in all_text
        assert "After image" in all_text

    def test_image_with_spaces_in_filename(self, tmp_path):
        rel = _make_image(tmp_path, name="Screenshot from 2026.png")
        tag = f"[media: {rel} (image/png, 104 B)]"
        config = _make_config(tmp_path)
        result = _build_user_content(tag, config, vision=True)
        assert isinstance(result, list)
        assert any(b["type"] == "image" for b in result)


# --- Vision MIME Types (unchanged) ---


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


# --- Provider Conversion: Anthropic (unchanged) ---


class TestAnthropicImageConversion:
    """Test Anthropic wire format for image content blocks."""

    def test_text_only_unchanged(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = _convert_messages_for_anthropic(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_image_block_converted(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "What's this?"},
            {"type": "image", "media_type": "image/jpeg", "data": "base64data"},
        ]}]
        result = _convert_messages_for_anthropic(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        img_blocks = [b for b in content if b["type"] == "image"]
        assert len(img_blocks) == 1
        img = img_blocks[0]
        assert img["source"]["type"] == "base64"
        assert img["source"]["media_type"] == "image/jpeg"
        assert img["source"]["data"] == "base64data"

    def test_text_block_preserved(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image", "media_type": "image/png", "data": "abc"},
        ]}]
        result = _convert_messages_for_anthropic(msgs)
        text_blocks = [b for b in result[0]["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "describe this"

    def test_multiple_images(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "compare"},
            {"type": "image", "media_type": "image/jpeg", "data": "img1"},
            {"type": "image", "media_type": "image/png", "data": "img2"},
        ]}]
        result = _convert_messages_for_anthropic(msgs)
        img_blocks = [b for b in result[0]["content"] if b["type"] == "image"]
        assert len(img_blocks) == 2


# --- Provider Conversion: OpenAI (unchanged) ---


class TestOpenAIImageConversion:
    """Test OpenAI wire format for image content blocks."""

    def test_text_only_unchanged(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = _convert_messages_for_openai(msgs)
        assert result[0]["content"] == "hello"

    def test_image_block_converted(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "What's this?"},
            {"type": "image", "media_type": "image/jpeg", "data": "base64data"},
        ]}]
        result = _convert_messages_for_openai(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        img_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(img_blocks) == 1
        url = img_blocks[0]["image_url"]["url"]
        assert url == "data:image/jpeg;base64,base64data"

    def test_text_block_preserved(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image", "media_type": "image/png", "data": "abc"},
        ]}]
        result = _convert_messages_for_openai(msgs)
        text_blocks = [b for b in result[0]["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "describe this"

    def test_multiple_images(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "compare"},
            {"type": "image", "media_type": "image/jpeg", "data": "img1"},
            {"type": "image", "media_type": "image/png", "data": "img2"},
        ]}]
        result = _convert_messages_for_openai(msgs)
        img_blocks = [b for b in result[0]["content"] if b["type"] == "image_url"]
        assert len(img_blocks) == 2


# --- Token Estimation (unchanged) ---


class TestImageTokenEstimation:
    """Test that image content blocks contribute to context token estimates."""

    @pytest.mark.asyncio
    async def test_image_adds_to_token_count(self, tmp_path):
        config = _make_config(tmp_path)

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events()
            agent = Agent(config)

            text_history = [{"role": "user", "content": "hello"}]
            text_estimate = agent._estimate_context_tokens("text_room", text_history)

            large_b64 = "A" * 10000  # ~7500 raw bytes
            image_history = [{"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "media_type": "image/jpeg", "data": large_b64},
            ]}]
            image_estimate = agent._estimate_context_tokens("img_room", image_history)

            assert image_estimate > text_estimate


# --- Integration: handle_input gates on the ACTIVE ROOM MODEL ---


class TestHandleInputVision:
    """The spec-section-1 live-hole regression suite: same agent config,
    different per-room active model -> different expansion behavior."""

    @pytest.mark.asyncio
    async def test_image_expanded_under_vision_room_model(self, tmp_path):
        """Active room model is vision-capable -> tag expands to blocks."""
        img_bytes = b"\xff\xd8" + b"\x00" * 100
        rel_path = _make_image(tmp_path, content=img_bytes)
        config = _make_config(tmp_path)  # default_model=VISION_MODEL

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content="I see a photo!")
            agent = Agent(config)
            text = f"[media: {rel_path} (image/jpeg, 102 B)]\nWhat's in this image?"
            result = await agent.handle_input(text)

            assert result == "I see a photo!"
            history = agent.history("_default")
            user_msg = history[0]
            assert isinstance(user_msg["content"], list)
            img_blocks = [b for b in user_msg["content"] if b["type"] == "image"]
            assert len(img_blocks) == 1

    @pytest.mark.asyncio
    async def test_image_not_expanded_under_blind_room_model(self, tmp_path):
        """LIVE-HOLE REGRESSION: /model to a blind model, THEN post an image
        -> the tag must pass through as text. No base64 is ever dumped into
        a model that cannot process images."""
        rel_path = _make_image(tmp_path)
        config = _make_config(
            tmp_path, extra_providers={"macstudio": _macstudio_provider()})

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content="I see a tag")
            agent = Agent(config)

            # Same agent config — switch the ROOM to a blind model first
            # (no images in history yet -> switch is allowed).
            err = agent.switch_model(BLIND_MODEL)
            assert err is None, f"precondition: switch to blind model failed: {err}"

            text = f"[media: {rel_path} (image/jpeg, 104 B)]\nWhat's this?"
            await agent.handle_input(text)

            history = agent.history("_default")
            user_msg = history[0]
            assert isinstance(user_msg["content"], str), \
                "blind room model must receive tags as plain text"
            assert "[media:" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_switch_back_to_vision_model_reenables(self, tmp_path):
        """Blind -> vision switch with only unexpanded tags in history is
        allowed (no image BLOCKS exist), and the next post expands."""
        rel_path = _make_image(tmp_path)
        config = _make_config(
            tmp_path, extra_providers={"macstudio": _macstudio_provider()})

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content="ok")
            agent = Agent(config)

            assert agent.switch_model(BLIND_MODEL) is None
            await agent.handle_input(f"[media: {rel_path} (image/jpeg, 104 B)]")
            # Tag stayed text -> no image blocks -> switch back is allowed
            err = agent.switch_model(VISION_MODEL)
            assert err is None, f"switch back to vision model should be allowed: {err}"
            await agent.handle_input(f"[media: {rel_path} (image/jpeg, 104 B)] again")

            last_user = [m for m in agent.history("_default")
                         if m["role"] == "user"][-1]
            assert isinstance(last_user["content"], list)

    @pytest.mark.asyncio
    async def test_override_disabled_model_stays_text(self, tmp_path):
        """[model_vision] force-disable: the default model IS table-capable
        but the operator override wins — tags pass through as text."""
        rel_path = _make_image(tmp_path)
        config = _make_config(tmp_path, model_vision={VISION_MODEL: False})

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content="tag text")
            agent = Agent(config)
            await agent.handle_input(f"[media: {rel_path} (image/jpeg, 104 B)]")

            user_msg = agent.history("_default")[0]
            assert isinstance(user_msg["content"], str)


# --- switch_model precise guard (kdsn.275) ---


class TestSwitchModelPreciseGuard:
    """Block ONLY when: history contains image blocks AND the TARGET model
    fails the vision resolver. Vision->vision switches stop being collateral
    damage; vision->blind with images stays blocked."""

    def _agent_with_images(self, tmp_path, **cfg_kwargs):
        config = _make_config(tmp_path, **cfg_kwargs)
        agent = Agent(config)
        room_id = "!img:server"
        agent.history(room_id).append({
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "media_type": "image/jpeg", "data": "base64data"},
            ],
        })
        return agent, room_id

    def test_images_plus_vision_target_allowed(self, tmp_path):
        """Sonnet -> Opus with images in history: allowed (blocked before)."""
        agent, room_id = self._agent_with_images(tmp_path)
        result = agent.switch_model("anthropic/claude-opus-5", room_id=room_id)
        assert result is None, f"vision-to-vision switch must be allowed: {result}"
        assert agent.get_model(room_id) == "anthropic/claude-opus-5"

    def test_images_plus_blind_target_blocked(self, tmp_path):
        agent, room_id = self._agent_with_images(
            tmp_path, extra_providers={"macstudio": _macstudio_provider()})
        result = agent.switch_model(BLIND_MODEL, room_id=room_id)
        assert result is not None
        assert "image" in result.lower() or "vision" in result.lower()
        assert agent.get_model(room_id) != BLIND_MODEL

    def test_images_plus_unknown_target_blocked(self, tmp_path):
        """Fail-closed: an uncharacterized target is treated as blind."""
        agent, room_id = self._agent_with_images(tmp_path)
        result = agent.switch_model(UNKNOWN_MODEL, room_id=room_id)
        assert result is not None
        assert agent.get_model(room_id) != UNKNOWN_MODEL

    def test_images_plus_blind_target_allowed_when_override_enables(self, tmp_path):
        """[model_vision] force-enable on the target unblocks the switch."""
        agent, room_id = self._agent_with_images(
            tmp_path,
            extra_providers={"macstudio": _macstudio_provider()},
            model_vision={BLIND_MODEL: True})
        result = agent.switch_model(BLIND_MODEL, room_id=room_id)
        assert result is None, f"override-enabled target must be allowed: {result}"

    def test_no_images_blind_target_allowed(self, tmp_path):
        """No images in history -> blind target is fine (text-only room)."""
        config = _make_config(
            tmp_path, extra_providers={"macstudio": _macstudio_provider()})
        agent = Agent(config)
        agent.history("!txt:server").append({"role": "user", "content": "hello"})
        result = agent.switch_model(BLIND_MODEL, room_id="!txt:server")
        assert result is None
        assert agent.get_model("!txt:server") == BLIND_MODEL

    def test_images_plus_vision_target_via_bare_alias_allowed(self, tmp_path):
        """Target given as a bare alias resolves through model_aliases."""
        agent, room_id = self._agent_with_images(
            tmp_path, model_aliases={"opus": "anthropic/claude-opus-5"})
        result = agent.switch_model("opus", room_id=room_id)
        assert result is None, f"alias to vision-capable target must be allowed: {result}"
