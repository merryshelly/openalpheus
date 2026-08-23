"""Tests for the view_image tool (kdsn.276).

Interface contract:
    view_image(path) lets an agent attach a workspace image to its own
    context. The tool validates (containment, existence, image MIME from
    extension, size cap, model vision capability) and deposits a
    `[media: path (mime, size)]` tag into a per-room inbox via a
    `vision_deposit` callback. At the top of the NEXT tool-loop iteration a
    `drain_vision` callback returns ONE framed user message batching all
    queued tags (framing by openalph.tools.vision.frame_vision_batch);
    agent.py expands it via the EXISTING _build_user_content and appends it
    after ALL tool results of the pending batch. Provider-universal: images
    ride user messages, never tool results (vllm#43203).

    Callback keys (wired by matrix._make_vision_callbacks for interactive
    turns; by subagent.py per-sub): "vision_deposit" (async tag->None),
    "drain_vision" (async -> str), "active_model" (str, set at dispatch).

Spec: memory/projects/openalph/vision-model-capability-spec.md sections 2, 4, 5.
"""

import json
import logging
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.config import AgentConfig, ProviderConfig
from openalph.agent import Agent, MEDIA_TAG_RE
from openalph.provider import Response, Usage, StreamEvent, ToolCall
from openalph.provider import (
    _convert_messages_for_anthropic,
    _convert_messages_for_openai,
)
from openalph.tools import (
    ToolDef, ToolResult, discover_tools, execute_tool,
)
from openalph.tools.vision import view_image, frame_vision_batch


# --- Fixtures ---

VISION_MODEL = "anthropic/claude-sonnet-5"
BLIND_MODEL = "macstudio/deepseek-v4-flash"

JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def _make_image(tmp_path, name="photo.jpg", content=JPEG_BYTES):
    """Write a fake image at workspace root; return workspace-relative path."""
    img = tmp_path / name
    img.write_bytes(content)
    return name


def _make_config(tmp_path, default_model=VISION_MODEL, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model=default_model,
        max_tokens=8192,
        providers={
            "anthropic": ProviderConfig(
                key="anthropic", type="anthropic", api_key="sk-test",
                base_url=None, quirks=[]),
            "macstudio": ProviderConfig(
                key="macstudio", type="openai", api_key="sk-local",
                base_url="http://10.0.20.104:8000/v1", quirks=[]),
        },
        workspace=tmp_path,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _deposit_spy(record: list):
    """A vision_deposit double: records tags, returns None (accepted)."""
    async def _deposit(tag: str):
        record.append(tag)
    return _deposit


def _drain_for(inbox: list):
    """A drain_vision double over a plain list inbox (frames like prod)."""
    async def _drain() -> str:
        tags = list(inbox)
        inbox.clear()
        return frame_vision_batch(tags) if tags else ""
    return _drain


def _callbacks(tmp_path, inbox=None, active_model=VISION_MODEL):
    """Callbacks dict mirroring the production wiring for a room."""
    if inbox is None:
        inbox = []
    return {
        "room_id": "!test:server",
        "vision_deposit": _deposit_spy(inbox),
        "drain_vision": _drain_for(inbox),
        "active_model": active_model,
    }


def _stream_tool_then_text(tool_calls_list, final_text="Done"):
    """Stream factory: first call returns tool_use, second returns text."""
    calls = iter([(tool_calls_list, ""), ([], final_text)])

    async def _stream(*args, **kwargs):
        tc_list, text = next(calls)
        if text:
            yield StreamEvent(type="text", content=text)
        for i, tc in enumerate(tc_list):
            yield StreamEvent(type="tool_done", tool_index=i, tool_call=tc)
        yield StreamEvent(
            type="done",
            response=Response(
                content=text,
                tool_calls=tc_list,
                model="claude-sonnet-5",
                usage=Usage(input_tokens=10, output_tokens=5),
                stop_reason="tool_use" if tc_list else "end_turn",
            ),
            stop_reason="tool_use" if tc_list else "end_turn",
            model="claude-sonnet-5",
        )
    return _stream


# --- Tool validation (spec 5.6: each failure a distinct steerable error) ---


class TestViewImageValidation:
    """view_image refuses cleanly, with steering, on every bad input class."""

    @pytest.mark.asyncio
    async def test_missing_file(self, tmp_path):
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": "gone.jpg"},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=_callbacks(tmp_path))
        assert result.is_error
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_non_image_extension(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": "notes.txt"},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=_callbacks(tmp_path))
        assert result.is_error
        assert "image" in result.content.lower()

    @pytest.mark.asyncio
    async def test_unsupported_image_format(self, tmp_path):
        (tmp_path / "scan.tiff").write_bytes(b"II*\x00" + b"\x00" * 50)
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": "scan.tiff"},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=_callbacks(tmp_path))
        assert result.is_error
        assert "tiff" in result.content.lower() or "supported" in result.content.lower()

    @pytest.mark.asyncio
    async def test_oversize_rejected_with_downscale_steering(self, tmp_path):
        rel = _make_image(tmp_path, content=JPEG_BYTES * 10)  # 1040 bytes
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": rel},
            tool_config={"max_bytes": 512},  # tiny cap for the test
            agent_config=config, callbacks=_callbacks(tmp_path))
        assert result.is_error
        assert "exceed" in result.content.lower()
        assert "downscale" in result.content.lower() or "shell" in result.content.lower()

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self, tmp_path):
        rel = _make_image(tmp_path)
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": str(tmp_path / rel)},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=_callbacks(tmp_path))
        assert result.is_error
        assert "relative" in result.content.lower() or "absolute" in result.content.lower()

    @pytest.mark.asyncio
    async def test_traversal_rejected(self, tmp_path):
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": "../../etc/passwd"},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=_callbacks(tmp_path))
        assert result.is_error
        assert ("outside" in result.content.lower()
                or "escape" in result.content.lower()
                or "workspace" in result.content.lower())

    @pytest.mark.asyncio
    async def test_traversal_that_lands_inside_is_also_rejected(self, tmp_path):
        """Any '..' component is rejected outright — no normalization games."""
        _make_image(tmp_path)
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": "subdir/../../photo.jpg"},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=_callbacks(tmp_path))
        assert result.is_error

    @pytest.mark.asyncio
    async def test_empty_file_rejected(self, tmp_path):
        (tmp_path / "empty.png").write_bytes(b"")
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": "empty.png"},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=_callbacks(tmp_path))
        assert result.is_error
        assert "empty" in result.content.lower()

    @pytest.mark.asyncio
    async def test_tag_breaking_chars_rejected(self, tmp_path):
        """Paths containing ] ( ) < > or control chars would break the
        [media:] tag framing / open an injection seam — rejected with steering."""
        (tmp_path / "evil].png").write_bytes(JPEG_BYTES)
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": "evil].png"},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=_callbacks(tmp_path))
        assert result.is_error
        assert "rename" in result.content.lower() or "character" in result.content.lower()

    @pytest.mark.asyncio
    async def test_blind_model_refused_with_model_escape(self, tmp_path):
        rel = _make_image(tmp_path)
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": rel},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=_callbacks(tmp_path, active_model=BLIND_MODEL))
        assert result.is_error
        assert BLIND_MODEL in result.content
        assert "/model" in result.content

    @pytest.mark.asyncio
    async def test_active_model_falls_back_to_default(self, tmp_path):
        """No active_model in callbacks -> agent_config.default_model is used."""
        rel = _make_image(tmp_path)
        config = _make_config(tmp_path, default_model=BLIND_MODEL)
        cbs = _callbacks(tmp_path)
        del cbs["active_model"]
        result = await execute_tool(
            name="view_image", input={"path": rel},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=cbs)
        assert result.is_error
        assert BLIND_MODEL in result.content

    @pytest.mark.asyncio
    async def test_no_deposit_callback_is_clean_error(self, tmp_path):
        """Unwired runtime context (e.g. heartbeat/CLI in v1) -> honest error,
        never a silent drop."""
        rel = _make_image(tmp_path)
        config = _make_config(tmp_path)
        cbs = _callbacks(tmp_path)
        del cbs["vision_deposit"]
        result = await execute_tool(
            name="view_image", input={"path": rel},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=cbs)
        assert result.is_error
        assert "not" in result.content.lower()  # "not available/wired ..."


# --- Success path (spec 5.7 first half) ---


class TestViewImageSuccess:
    """Deposit + text-only ToolResult. The image NEVER rides the tool result."""

    @pytest.mark.asyncio
    async def test_deposit_and_text_result(self, tmp_path):
        rel = _make_image(tmp_path)
        inbox = []
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": rel},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=_callbacks(tmp_path, inbox=inbox))
        assert not result.is_error
        assert "will be provided before the next model call" in result.content
        # Exactly one tag deposited, workspace-relative path, correct MIME
        assert len(inbox) == 1
        m = MEDIA_TAG_RE.search(inbox[0])
        assert m is not None, f"deposited text is not a parseable media tag: {inbox[0]!r}"
        assert m.group(1) == rel
        assert m.group(2) == "image/jpeg"

    @pytest.mark.asyncio
    async def test_mime_derived_from_extension_case_insensitive(self, tmp_path):
        (tmp_path / "PIC.PNG").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        inbox = []
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": "PIC.PNG"},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=_callbacks(tmp_path, inbox=inbox))
        assert not result.is_error
        assert "(image/png," in inbox[0]

    @pytest.mark.asyncio
    async def test_subdirectory_image_keeps_relative_path(self, tmp_path):
        sub = tmp_path / "shots"
        sub.mkdir()
        (sub / "a.webp").write_bytes(b"RIFF" + b"\x00" * 30)
        inbox = []
        config = _make_config(tmp_path)
        result = await execute_tool(
            name="view_image", input={"path": "shots/a.webp"},
            tool_config={"max_bytes": 5_242_880}, agent_config=config,
            callbacks=_callbacks(tmp_path, inbox=inbox))
        assert not result.is_error
        m = MEDIA_TAG_RE.search(inbox[0])
        assert m and m.group(1) == "shots/a.webp"
        assert m.group(2) == "image/webp"

    @pytest.mark.asyncio
    async def test_size_cap_override_via_tool_config(self, tmp_path):
        """workspace/tools/view_image.toml [config] max_bytes overrides 5MB."""
        rel = _make_image(tmp_path, content=JPEG_BYTES * 100)  # ~10.4KB
        config = _make_config(tmp_path)
        # Generous override: accepts
        ok = await execute_tool(
            name="view_image", input={"path": rel},
            tool_config={"max_bytes": 20_971_520}, agent_config=config,
            callbacks=_callbacks(tmp_path))
        assert not ok.is_error
        # Default-capped: rejects (10.4KB > 64B test cap proves override works)
        no = await execute_tool(
            name="view_image", input={"path": rel},
            tool_config={"max_bytes": 64}, agent_config=config,
            callbacks=_callbacks(tmp_path))
        assert no.is_error


# --- Framing ---


class TestFrameVisionBatch:
    """frame_vision_batch: the ONE batched framed user message (spec 4.1.3)."""

    def test_single_image_frame(self):
        tags = ["[media: a.jpg (image/jpeg, 104 B)]"]
        framed = frame_vision_batch(tags)
        assert "[view_image tool output" in framed
        assert "a.jpg" in framed
        assert tags[0] in framed

    def test_multi_image_frame_batches_all(self):
        tags = ["[media: a.jpg (image/jpeg, 104 B)]",
                "[media: b.png (image/png, 200 B)]"]
        framed = frame_vision_batch(tags)
        assert "2" in framed  # image count in the header
        for t in tags:
            assert t in framed

    def test_empty_batch_is_empty(self):
        assert frame_vision_batch([]) == ""


# --- Discovery + description ---


class TestViewImageDiscovery:
    def test_builtin_default_cap_is_5mb(self, tmp_path):
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "view_image.toml").write_text("")
        tools = discover_tools(tmp_path)
        vi = [t for t in tools if t.name == "view_image"]
        assert len(vi) == 1
        assert vi[0].config["max_bytes"] == 5_242_880

    def test_toml_config_overrides_cap(self, tmp_path):
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "view_image.toml").write_text(
            '[config]\nmax_bytes = 1048576\n')
        tools = discover_tools(tmp_path)
        vi = [t for t in tools if t.name == "view_image"][0]
        assert vi.config["max_bytes"] == 1_048_576

    def test_description_steers_usage(self):
        """House-style description: when-NOT (send_media), supported types."""
        from openalph.tools import BUILTIN_TOOLS
        desc = BUILTIN_TOOLS["view_image"]["description"].lower()
        assert "send_media" in desc
        assert "jpeg" in desc or "png" in desc


# --- Drain integration: agent loop (spec 5.7/5.8/5.10) ---


class TestVisionDrainInAgentLoop:
    """handle_input: deposit during tool execution -> injected as ONE user
    message at the next loop top, AFTER all tool results of the batch."""

    def _agent_with_vision_tool(self, tmp_path):
        (tmp_path / "tools").mkdir(exist_ok=True)
        (tmp_path / "tools" / "view_image.toml").write_text("")
        config = _make_config(tmp_path)
        return config

    @pytest.mark.asyncio
    async def test_deposit_then_injected_next_iteration(self, tmp_path):
        rel = _make_image(tmp_path)
        config = self._agent_with_vision_tool(tmp_path)
        inbox = []
        cbs = _callbacks(tmp_path, inbox=inbox)

        tc = ToolCall(id="tc1", name="view_image", input={"path": rel})
        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system"):
            mock_stream.side_effect = _stream_tool_then_text([tc], "I see it")
            agent = Agent(config)
            result = await agent.handle_input("look at this", callbacks=cbs)

        assert result == "I see it"
        history = agent.history("_default")
        roles = [m["role"] for m in history]
        # user, assistant(tool_calls), tool(result), user(INJECTED), assistant
        assert roles[0] == "user"
        assert roles[1] == "assistant"
        assert roles[2] == "tool"
        assert roles[3] == "user", f"injected message must follow tool results: {roles}"
        injected = history[3]
        assert isinstance(injected["content"], list), \
            "vision-capable room model must expand the injected tag to blocks"
        img = [b for b in injected["content"] if b.get("type") == "image"]
        assert len(img) == 1
        assert img[0]["media_type"] == "image/jpeg"
        import base64 as _b64
        assert _b64.b64decode(img[0]["data"]) == JPEG_BYTES

    @pytest.mark.asyncio
    async def test_ordering_with_parallel_tool_batch(self, tmp_path):
        """Parallel batch of view_image + another tool: the injected message
        lands after ALL tool results of the batch (spec invariant 1)."""
        rel = _make_image(tmp_path)
        config = self._agent_with_vision_tool(tmp_path)
        inbox = []
        cbs = _callbacks(tmp_path, inbox=inbox)

        tc_view = ToolCall(id="tc1", name="view_image", input={"path": rel})
        # Second tool is NOT enabled in this workspace -> clean unknown-tool
        # error result; ordering assertions are unaffected.
        tc_other = ToolCall(id="tc2", name="file_read", input={"path": "x"})
        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system"):
            mock_stream.side_effect = _stream_tool_then_text([tc_view, tc_other], "ok")
            agent = Agent(config)
            await agent.handle_input("batch", callbacks=cbs)

        history = agent.history("_default")
        roles = [m["role"] for m in history]
        tool_idxs = [i for i, r in enumerate(roles) if r == "tool"]
        assert len(tool_idxs) == 2, f"expected 2 tool results, got {roles}"
        injected_idx = next(i for i, m in enumerate(history)
                            if m["role"] == "user" and i > 0)
        assert injected_idx > max(tool_idxs), \
            f"injected user message must follow ALL tool results: {roles}"

    @pytest.mark.asyncio
    async def test_multi_image_batching_one_message(self, tmp_path):
        """Two view_image calls in one turn -> ONE injected user message
        carrying TWO image blocks (spec 5.10)."""
        rel_a = _make_image(tmp_path, name="a.jpg")
        rel_b = _make_image(tmp_path, name="b.jpg")
        config = self._agent_with_vision_tool(tmp_path)
        inbox = []
        cbs = _callbacks(tmp_path, inbox=inbox)

        calls = [ToolCall(id="tc1", name="view_image", input={"path": rel_a}),
                 ToolCall(id="tc2", name="view_image", input={"path": rel_b})]
        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system"):
            mock_stream.side_effect = _stream_tool_then_text(calls, "both seen")
            agent = Agent(config)
            await agent.handle_input("compare", callbacks=cbs)

        injected = [m for m in agent.history("_default")
                    if m["role"] == "user"
                    and isinstance(m["content"], list)]
        assert len(injected) == 1, "two deposits must batch into ONE user message"
        imgs = [b for b in injected[0]["content"] if b.get("type") == "image"]
        assert len(imgs) == 2

    @pytest.mark.asyncio
    async def test_blind_room_model_tool_error_no_injection(self, tmp_path):
        """Room model blind at dispatch -> tool error result; NOTHING is
        deposited or injected (spec 5.6 + 5.12 override/disable behavior)."""
        rel = _make_image(tmp_path)
        config = self._agent_with_vision_tool(tmp_path)
        inbox = []
        cbs = _callbacks(tmp_path, inbox=inbox, active_model=BLIND_MODEL)
        # Room actually running the blind model:
        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system"):
            tc = ToolCall(id="tc1", name="view_image", input={"path": rel})
            mock_stream.side_effect = _stream_tool_then_text([tc], "cannot see")
            agent = Agent(config)
            agent.switch_model(BLIND_MODEL)  # no images yet -> allowed
            await agent.handle_input("look", callbacks=cbs)

        assert inbox == [], "blind model must refuse before deposit"
        history = agent.history("_default")
        tool_results = [m for m in history if m["role"] == "tool"]
        assert tool_results and tool_results[0]["is_error"] is True
        assert BLIND_MODEL in tool_results[0]["content"]
        # No injected user message beyond the original
        assert [m["role"] for m in history].count("user") == 1


# --- Matrix wiring: deposit/drain closures + JSONL (spec 4.1, 4.3.5) ---


class TestMatrixVisionCallbacks:
    """MatrixBot._make_vision_callbacks(room_id) builds the per-room deposit
    and drain closures; the drain logs ONE JSONL entry (source=view_image)
    and returns the framed text for the agent loop to expand."""

    def _bare_bot(self):
        from openalph.matrix import MatrixBot
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = MagicMock(user_id="@merry:test")
        bot.session_log = MagicMock()
        bot.send_notice = AsyncMock()
        return bot

    def test_callbacks_have_expected_keys(self):
        bot = self._bare_bot()
        cbs = bot._make_vision_callbacks("!r:test")
        assert callable(cbs["vision_deposit"])
        assert callable(cbs["drain_vision"])

    @pytest.mark.asyncio
    async def test_deposit_appends_to_per_room_inbox(self):
        bot = self._bare_bot()
        cbs = bot._make_vision_callbacks("!r:test")
        await cbs["vision_deposit"]("[media: a.jpg (image/jpeg, 1 B)]")
        await cbs["vision_deposit"]("[media: b.png (image/png, 2 B)]")
        assert bot._vision_inbox["!r:test"] == [
            "[media: a.jpg (image/jpeg, 1 B)]",
            "[media: b.png (image/png, 2 B)]",
        ]
        # A different room is isolated
        assert bot._vision_inbox.get("!other:test") in (None, [])

    @pytest.mark.asyncio
    async def test_drain_frames_logs_and_pops(self):
        bot = self._bare_bot()
        cbs = bot._make_vision_callbacks("!r:test")
        tags = ["[media: a.jpg (image/jpeg, 1 B)]", "[media: b.png (image/png, 2 B)]"]
        for t in tags:
            await cbs["vision_deposit"](t)

        framed = await cbs["drain_vision"]()
        assert framed == frame_vision_batch(tags)
        assert "[view_image tool output" in framed
        # ONE JSONL entry, role=user, source=view_image, tag text stored
        assert bot.session_log.append.call_count == 1
        _, kwargs = bot.session_log.append.call_args
        assert kwargs.get("role") == "user"
        assert kwargs.get("source") == "view_image"
        assert kwargs.get("content") == framed
        # Inbox drained; second drain is empty and logs nothing
        assert bot._vision_inbox.get("!r:test") in (None, [])
        assert await cbs["drain_vision"]() == ""
        assert bot.session_log.append.call_count == 1

    @pytest.mark.asyncio
    async def test_drain_without_session_log_still_frames(self):
        """No session_log handle (tests/CLI) -> drain still returns the frame."""
        bot = self._bare_bot()
        bot.session_log = None
        cbs = bot._make_vision_callbacks("!r:test")
        await cbs["vision_deposit"]("[media: a.jpg (image/jpeg, 1 B)]")
        framed = await cbs["drain_vision"]()
        assert "[media: a.jpg" in framed


class TestViewImageJSONL:
    """Spec 4.3.5: injected entry persists with source=view_image and the
    tag text; rehydration degrades to plain text (never re-expanded)."""

    def test_entry_survives_rehydration_as_text(self, tmp_path):
        from openalph.session import SessionLog
        sl = SessionLog(workspace=tmp_path, agent_user_id="@agent:x")
        framed = frame_vision_batch(["[media: a.jpg (image/jpeg, 104 B)]"])
        sl.append(role="user", sender="@agent:x", room="!r:x",
                  event_id=None, content=framed, source="view_image")
        ctx = sl.build_context("!r:x")
        assert ctx[0]["role"] == "user"
        assert isinstance(ctx[0]["content"], str), \
            "rehydrated view_image entry must be plain text (expansion is live-only)"
        assert "[media: a.jpg" in ctx[0]["content"]
        assert "[view_image tool output" in ctx[0]["content"]


# --- Provider wire formats for the injected message (spec 4.3.2) ---


class TestInjectedMessageWireFormat:
    """The injected user message lands after the tool results; Anthropic's
    converter tolerates the resulting consecutive user messages and converts
    the image block; OpenAI's emits a data-URI image_url."""

    def _history_shape(self):
        framed = frame_vision_batch(["[media: a.jpg (image/jpeg, 104 B)]"])
        return [
            {"role": "user", "content": "look at this"},
            {"role": "assistant", "content": "", "tool_calls": [
                ToolCall(id="tc1", name="view_image", input={"path": "a.jpg"})]},
            {"role": "tool", "tool_call_id": "tc1",
             "content": "Image attached — will be provided before the next model call."},
            # The injected message, post-_build_user_content expansion:
            {"role": "user", "content": [
                {"type": "text", "text": framed},
                {"type": "image", "media_type": "image/jpeg", "data": "AAAA"},
            ]},
        ]

    def test_anthropic_converter_no_alternation_assumption(self):
        out = _convert_messages_for_anthropic(self._history_shape())
        # No raise = the big one. Now pin ordering + image conversion.
        roles = [m.get("role") for m in out]
        assert roles[0] == "user"
        # The image must survive conversion, after the tool result
        flat = json.dumps(out)
        assert '"base64"' in flat and "AAAA" in flat
        img_pos = flat.index("AAAA")
        tool_pos = flat.index("will be provided before the next model call")
        assert tool_pos < img_pos, "image must land after the tool result"

    def test_openai_converter_emits_data_uri(self):
        result = _convert_messages_for_openai(self._history_shape())
        flat = json.dumps(result)
        assert "data:image/jpeg;base64,AAAA" in flat


# --- Subagent drain (spec 4.4, 5.11) ---


class TestSubagentVisionDrain:
    """Subagents get a per-sub vision inbox in v1: a sub's view_image call
    lands in the SUB's context (never the parent's)."""

    def _tools(self):
        return [ToolDef(
            name="view_image",
            description="d",
            parameters={"type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]},
            config={"max_bytes": 5_242_880},
        )]

    def _fake_complete(self, captured, final="I see the image"):
        async def _complete(*, config, system, messages, tools, max_tokens):
            captured.append([dict(m) for m in messages])
            if len(captured) == 1:
                return Response(
                    content="", model="claude-sonnet-5",
                    tool_calls=[ToolCall(id="tc1", name="view_image",
                                         input={"path": self._rel})],
                    usage=Usage(input_tokens=10, output_tokens=5),
                    stop_reason="tool_use")
            return Response(
                content=final, model="claude-sonnet-5", tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=5),
                stop_reason="end_turn")
        return _complete

    @pytest.mark.asyncio
    async def test_sub_view_image_lands_in_sub_context(self, tmp_path):
        from openalph.tools import subagent as sub_mod
        self._rel = _make_image(tmp_path)
        config = _make_config(tmp_path)  # default VISION_MODEL
        captured = []
        with patch("openalph.tools.subagent.complete",
                   side_effect=self._fake_complete(captured)):
            result = await sub_mod.run_subagent(
                task="view it", config=config, tools=self._tools())

        assert not result.is_error
        assert result.content == "I see the image"
        assert len(captured) == 2
        second = captured[1]
        injected = [m for m in second
                    if m["role"] == "user" and isinstance(m.get("content"), list)]
        assert len(injected) == 1, \
            f"expected ONE injected vision user message in sub context: {second}"
        imgs = [b for b in injected[0]["content"] if b.get("type") == "image"]
        assert len(imgs) == 1
        # Ordering invariant in the sub loop too: after the tool result
        roles = [m["role"] for m in second]
        assert "tool" in roles
        assert roles.index("tool") < second.index(injected[0])
        # Flight recorder (sub audit plane) captured the injection
        transcripts = list((tmp_path / "sessions" / "subs").glob("*.jsonl"))
        assert transcripts, "sub flight recorder transcript missing"
        assert '"event": "view_image"' in transcripts[0].read_text()

    @pytest.mark.asyncio
    async def test_sub_blind_model_refused(self, tmp_path):
        from openalph.tools import subagent as sub_mod
        self._rel = _make_image(tmp_path)
        config = _make_config(tmp_path, default_model=BLIND_MODEL)
        captured = []
        with patch("openalph.tools.subagent.complete",
                   side_effect=self._fake_complete(captured, final="no eyes")):
            result = await sub_mod.run_subagent(
                task="view it", config=config, tools=self._tools())

        assert not result.is_error
        assert result.content == "no eyes"
        second = captured[1]
        # No injected image message
        injected = [m for m in second
                    if m["role"] == "user" and isinstance(m.get("content"), list)]
        assert injected == []
        # The tool result carries the refusal naming the blind model
        tool_msgs = [m for m in second if m["role"] == "tool"]
        assert tool_msgs and tool_msgs[0].get("is_error") is True
        assert BLIND_MODEL in tool_msgs[0]["content"]
