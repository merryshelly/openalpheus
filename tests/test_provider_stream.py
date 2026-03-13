"""Tests for provider streaming (Phase A of kdsn.65).

Contract:
    stream(config, system, messages, tools, max_tokens, model, thinking)
        -> AsyncGenerator[StreamEvent, None]

    StreamEvent.type values:
        "text"       — text delta (content field)
        "thinking"   — thinking delta (content field)
        "signature"  — thinking block signature (content field)
        "tool_start" — tool_use block started (tool_index, tool_id, tool_name)
        "tool_delta" — partial tool JSON (tool_index, content)
        "tool_done"  — tool_use block complete (tool_index, tool_call: ToolCall)
        "usage"      — final usage stats (usage field)
        "done"       — stream complete (stop_reason, model, response: Response)

    complete() wraps stream() internally — same interface, same behavior.
    _ANTHROPIC_NONSTREAMING_MAX cap is removed (all paths use streaming).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import anthropic as anthropic_sdk
import openai as openai_sdk

from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import (
    StreamEvent,
    stream,
    complete,
    Response,
    Usage,
    ToolCall,
    ThinkingBlock,
    ProviderError,
    _DEGEN_CHAR_THRESHOLD,
    _DEGEN_WARNING,
)


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

def make_provider(key="anthropic", type="anthropic", api_key="sk-test",
                  base_url=None, quirks=None):
    return ProviderConfig(
        key=key, type=type, api_key=api_key,
        base_url=base_url, quirks=quirks or [],
    )


def make_config(**kwargs):
    defaults = {
        "name": "test-agent",
        "default_model": "anthropic/claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "providers": {"anthropic": make_provider(key="anthropic")},
        "workspace": Path("/tmp/test"),
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_openai_config(**kwargs):
    defaults = {
        "name": "test-agent",
        "default_model": "openrouter/moonshotai/kimi-k2.5",
        "max_tokens": 8192,
        "providers": {
            "openrouter": make_provider(
                key="openrouter", type="openai", api_key="sk-test",
                base_url="https://openrouter.ai/api/v1",
            )
        },
        "workspace": Path("/tmp/test"),
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


# --- Anthropic stream mocking ---

class MockAnthropicStream:
    """Mock for Anthropic's AsyncMessageStream context manager."""

    def __init__(self, events, final_message=None):
        self._events = events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final_message


def _anthropic_text(text):
    e = MagicMock()
    e.type = "text"
    e.text = text
    return e


def _anthropic_thinking(text):
    e = MagicMock()
    e.type = "thinking"
    e.thinking = text
    return e


def _anthropic_signature(signature):
    e = MagicMock()
    e.type = "signature"
    e.signature = signature
    return e


def _anthropic_input_json(partial_json):
    e = MagicMock()
    e.type = "input_json"
    e.partial_json = partial_json
    return e


def _anthropic_block_start(index, block_type="text", **kwargs):
    e = MagicMock()
    e.type = "content_block_start"
    e.index = index
    block = MagicMock()
    block.type = block_type
    if block_type == "tool_use":
        block.id = kwargs.get("tool_id", f"toolu_{index}")
        block.name = kwargs.get("tool_name", "test_tool")
    e.content_block = block
    return e


def _anthropic_block_stop(index, content_block):
    e = MagicMock()
    e.type = "content_block_stop"
    e.index = index
    e.content_block = content_block
    return e


def _anthropic_tool_block(tool_id, name, input_dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_dict
    return block


def _anthropic_message_stop():
    e = MagicMock()
    e.type = "message_stop"
    return e


def _anthropic_final_message(
    text="", model="claude-sonnet-4-20250514",
    input_tokens=100, output_tokens=50,
    cache_read=0, cache_create=0,
    stop_reason="end_turn", tool_calls=None, thinking=None,
):
    msg = MagicMock()
    msg.model = model
    msg.stop_reason = stop_reason
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    msg.usage.cache_read_input_tokens = cache_read
    msg.usage.cache_creation_input_tokens = cache_create

    content_blocks = []
    if thinking:
        for t in thinking:
            tb = MagicMock()
            tb.type = "thinking"
            tb.thinking = t["thinking"]
            tb.signature = t.get("signature", "sig-xxx")
            content_blocks.append(tb)
    if text:
        tb = MagicMock()
        tb.type = "text"
        tb.text = text
        content_blocks.append(tb)
    if tool_calls:
        for tc in tool_calls:
            tb = MagicMock()
            tb.type = "tool_use"
            tb.id = tc["id"]
            tb.name = tc["name"]
            tb.input = tc["input"]
            content_blocks.append(tb)
    msg.content = content_blocks
    return msg


# --- OpenAI stream mocking ---

class MockOpenAIStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for chunk in self._chunks:
            yield chunk


def _openai_text_chunk(content, finish_reason=None):
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _openai_tool_chunk(index, tool_id=None, name=None, arguments_delta=""):
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = None
    tc_delta = MagicMock()
    tc_delta.index = index
    tc_delta.id = tool_id
    tc_delta.function = MagicMock()
    tc_delta.function.name = name
    tc_delta.function.arguments = arguments_delta
    delta.tool_calls = [tc_delta]
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _openai_finish_chunk(finish_reason="stop"):
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = None
    delta.tool_calls = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _openai_usage_chunk(prompt_tokens=100, completion_tokens=50):
    chunk = MagicMock()
    chunk.choices = []
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    chunk.usage = usage
    return chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def collect_events(gen):
    events = []
    async for event in gen:
        events.append(event)
    return events


def events_of_type(events, type_name):
    return [e for e in events if e.type == type_name]


# ---------------------------------------------------------------------------
# Tests: StreamEvent dataclass
# ---------------------------------------------------------------------------

class TestStreamEvent:

    def test_text_event(self):
        e = StreamEvent(type="text", content="hello")
        assert e.type == "text"
        assert e.content == "hello"
        assert e.tool_call is None
        assert e.response is None

    def test_thinking_event(self):
        e = StreamEvent(type="thinking", content="reasoning...")
        assert e.type == "thinking"
        assert e.content == "reasoning..."

    def test_signature_event(self):
        e = StreamEvent(type="signature", content="sig-abc123")
        assert e.type == "signature"
        assert e.content == "sig-abc123"

    def test_tool_done_event(self):
        tc = ToolCall(id="tc1", name="shell", input={"command": "ls"})
        e = StreamEvent(type="tool_done", tool_index=0, tool_call=tc)
        assert e.type == "tool_done"
        assert e.tool_call.name == "shell"
        assert e.tool_call.input == {"command": "ls"}

    def test_done_event_with_response(self):
        resp = Response(content="done", model="test-model")
        e = StreamEvent(
            type="done", stop_reason="end_turn", model="test-model", response=resp,
        )
        assert e.type == "done"
        assert e.response.content == "done"
        assert e.stop_reason == "end_turn"

    def test_defaults(self):
        e = StreamEvent(type="text")
        assert e.content == ""
        assert e.tool_index == 0
        assert e.tool_id == ""
        assert e.tool_name == ""
        assert e.tool_call is None
        assert e.usage is None
        assert e.stop_reason == ""
        assert e.model == ""
        assert e.response is None


# ---------------------------------------------------------------------------
# Tests: Anthropic streaming
# ---------------------------------------------------------------------------

class TestAnthropicStream:

    @pytest.mark.asyncio
    async def test_text_only(self):
        """Pure text response yields text events then done."""
        config = make_config()
        sdk_events = [
            _anthropic_text("Hello "),
            _anthropic_text("world!"),
            _anthropic_message_stop(),
        ]
        final_msg = _anthropic_final_message(
            text="Hello world!", input_tokens=100, output_tokens=10,
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                sdk_events, final_msg,
            )

            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            ))

        texts = events_of_type(events, "text")
        assert len(texts) == 2
        assert texts[0].content == "Hello "
        assert texts[1].content == "world!"

        dones = events_of_type(events, "done")
        assert len(dones) == 1
        done = dones[0]
        assert done.response is not None
        assert done.response.content == "Hello world!"
        assert done.response.usage.input_tokens == 100
        assert done.response.usage.output_tokens == 10
        assert done.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_thinking_then_text(self):
        """Thinking events arrive before text events."""
        config = make_config()
        sdk_events = [
            _anthropic_thinking("Let me think..."),
            _anthropic_thinking(" carefully."),
            _anthropic_text("The answer is 42."),
            _anthropic_message_stop(),
        ]
        final_msg = _anthropic_final_message(
            text="The answer is 42.",
            thinking=[{"thinking": "Let me think... carefully.", "signature": "sig-abc"}],
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                sdk_events, final_msg,
            )

            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Think hard"}],
                thinking="high",
            ))

        thinking = events_of_type(events, "thinking")
        assert len(thinking) == 2
        assert thinking[0].content == "Let me think..."
        assert thinking[1].content == " carefully."

        texts = events_of_type(events, "text")
        assert len(texts) == 1
        assert texts[0].content == "The answer is 42."

        done = events_of_type(events, "done")[0]
        assert done.response.content == "The answer is 42."
        assert len(done.response.thinking) >= 1

    @pytest.mark.asyncio
    async def test_signature_event(self):
        """Signature events from thinking blocks are yielded."""
        config = make_config()
        sdk_events = [
            _anthropic_thinking("reasoning"),
            _anthropic_signature("sig-xyz789"),
            _anthropic_text("answer"),
            _anthropic_message_stop(),
        ]
        final_msg = _anthropic_final_message(
            text="answer",
            thinking=[{"thinking": "reasoning", "signature": "sig-xyz789"}],
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                sdk_events, final_msg,
            )

            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                thinking="high",
            ))

        sigs = events_of_type(events, "signature")
        assert len(sigs) == 1
        assert sigs[0].content == "sig-xyz789"

    @pytest.mark.asyncio
    async def test_tool_use(self):
        """Tool use blocks yield tool events."""
        config = make_config()
        tool_block = _anthropic_tool_block(
            "toolu_1", "shell", {"command": "ls -la"},
        )
        sdk_events = [
            _anthropic_text("Let me check."),
            _anthropic_block_start(
                1, "tool_use", tool_id="toolu_1", tool_name="shell",
            ),
            _anthropic_input_json('{"command": "ls -la"}'),
            _anthropic_block_stop(1, tool_block),
            _anthropic_message_stop(),
        ]
        final_msg = _anthropic_final_message(
            text="Let me check.",
            tool_calls=[{
                "id": "toolu_1", "name": "shell",
                "input": {"command": "ls -la"},
            }],
            stop_reason="tool_use",
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                sdk_events, final_msg,
            )

            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "List files"}],
            ))

        texts = events_of_type(events, "text")
        assert len(texts) == 1
        assert texts[0].content == "Let me check."

        tool_dones = events_of_type(events, "tool_done")
        assert len(tool_dones) == 1
        assert tool_dones[0].tool_call.name == "shell"
        assert tool_dones[0].tool_call.input == {"command": "ls -la"}
        assert tool_dones[0].tool_call.id == "toolu_1"

        done = events_of_type(events, "done")[0]
        assert done.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """Multiple tool_use blocks each yield a tool_done event."""
        config = make_config()
        block_1 = _anthropic_tool_block("toolu_1", "shell", {"command": "ls"})
        block_2 = _anthropic_tool_block("toolu_2", "file_read", {"path": "/tmp/x"})
        sdk_events = [
            _anthropic_block_start(0, "tool_use", tool_id="toolu_1", tool_name="shell"),
            _anthropic_block_stop(0, block_1),
            _anthropic_block_start(1, "tool_use", tool_id="toolu_2", tool_name="file_read"),
            _anthropic_block_stop(1, block_2),
            _anthropic_message_stop(),
        ]
        final_msg = _anthropic_final_message(
            tool_calls=[
                {"id": "toolu_1", "name": "shell", "input": {"command": "ls"}},
                {"id": "toolu_2", "name": "file_read", "input": {"path": "/tmp/x"}},
            ],
            stop_reason="tool_use",
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                sdk_events, final_msg,
            )

            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Do two things"}],
            ))

        tool_dones = events_of_type(events, "tool_done")
        assert len(tool_dones) == 2
        assert tool_dones[0].tool_call.name == "shell"
        assert tool_dones[1].tool_call.name == "file_read"

    @pytest.mark.asyncio
    async def test_error_during_stream_iteration(self):
        """SDK error mid-stream raises ProviderError."""
        config = make_config()

        class ErrorStream(MockAnthropicStream):
            async def _aiter_impl(self):
                yield _anthropic_text("partial")
                raise anthropic_sdk.APIStatusError(
                    message="overloaded",
                    response=MagicMock(status_code=529),
                    body=None,
                )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = ErrorStream([], None)

            with pytest.raises(ProviderError):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                ))

    @pytest.mark.asyncio
    async def test_error_on_stream_setup(self):
        """Error creating the stream raises ProviderError."""
        config = make_config()

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.side_effect = anthropic_sdk.APIStatusError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body=None,
            )

            with pytest.raises(ProviderError):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                ))

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """SDK timeout raises ProviderError."""
        config = make_config()

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.side_effect = anthropic_sdk.APITimeoutError(
                request=MagicMock(),
            )

            with pytest.raises(ProviderError, match="timed out"):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                ))

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """SDK connection error raises ProviderError."""
        config = make_config()

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.side_effect = anthropic_sdk.APIConnectionError(
                request=MagicMock(),
            )

            with pytest.raises(ProviderError, match="connection"):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                ))

    @pytest.mark.asyncio
    async def test_degeneration_detected(self):
        """Degenerate output is detected and flagged."""
        config = make_config()
        normal = "Hello "
        degenerate = "a" * (_DEGEN_CHAR_THRESHOLD + 10)
        full_text = normal + degenerate

        sdk_events = [
            _anthropic_text(normal),
            _anthropic_text(degenerate),
            _anthropic_message_stop(),
        ]
        final_msg = _anthropic_final_message(text=full_text)

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                sdk_events, final_msg,
            )

            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            ))

        done = events_of_type(events, "done")[0]
        assert done.response.degenerate is True
        assert _DEGEN_WARNING in done.response.content

    @pytest.mark.asyncio
    async def test_kwargs_system_caching(self):
        """System prompt gets cache_control in Anthropic kwargs."""
        config = make_config(max_tokens=4096)
        final_msg = _anthropic_final_message()

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_message_stop()], final_msg,
            )

            await collect_events(stream(
                config=config, system="You are helpful.",
                messages=[{"role": "user", "content": "Hi"}],
            ))

        client.messages.stream.assert_called_once()
        kw = client.messages.stream.call_args.kwargs
        assert kw["model"] == "claude-sonnet-4-20250514"
        assert kw["system"] == [
            {
                "type": "text",
                "text": "You are helpful.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        assert kw["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_kwargs_last_user_message_cached(self):
        """Last user message gets cache_control."""
        config = make_config()
        final_msg = _anthropic_final_message()

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_message_stop()], final_msg,
            )

            await collect_events(stream(
                config=config, system="Test",
                messages=[
                    {"role": "user", "content": "First"},
                    {"role": "assistant", "content": "Reply"},
                    {"role": "user", "content": "Second"},
                ],
            ))

        kw = client.messages.stream.call_args.kwargs
        last_msg = kw["messages"][-1]
        assert last_msg["content"] == [
            {
                "type": "text", "text": "Second",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    @pytest.mark.asyncio
    async def test_no_21k_cap(self):
        """max_tokens above 21000 is passed through in streaming."""
        config = make_config(max_tokens=50000)
        final_msg = _anthropic_final_message()

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_message_stop()], final_msg,
            )

            await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            ))

        kw = client.messages.stream.call_args.kwargs
        assert kw["max_tokens"] == 50000

    @pytest.mark.asyncio
    async def test_thinking_kwargs(self):
        """Thinking parameters are included in API kwargs."""
        config = make_config(
            default_model="anthropic/claude-sonnet-4-20250514",
        )
        final_msg = _anthropic_final_message()

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_message_stop()], final_msg,
            )

            await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                thinking="high",
            ))

        kw = client.messages.stream.call_args.kwargs
        assert "thinking" in kw

    @pytest.mark.asyncio
    async def test_tools_passed(self):
        """Tool definitions are converted and passed."""
        config = make_config()
        final_msg = _anthropic_final_message()

        tool_def = MagicMock()
        tool_def.name = "shell"
        tool_def.description = "Run a shell command"
        tool_def.parameters = {"type": "object", "properties": {}}

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_message_stop()], final_msg,
            )

            await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                tools=[tool_def],
            ))

        kw = client.messages.stream.call_args.kwargs
        assert "tools" in kw
        assert kw["tools"][0]["name"] == "shell"


# ---------------------------------------------------------------------------
# Tests: OpenAI streaming
# ---------------------------------------------------------------------------

class TestOpenAIStream:

    @pytest.mark.asyncio
    async def test_text_only(self):
        """Pure text response yields text events then done."""
        config = make_openai_config()
        chunks = [
            _openai_text_chunk("Hello "),
            _openai_text_chunk("world!"),
            _openai_finish_chunk("stop"),
            _openai_usage_chunk(prompt_tokens=100, completion_tokens=10),
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream(chunks),
            )

            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            ))

        texts = events_of_type(events, "text")
        assert len(texts) == 2
        assert texts[0].content == "Hello "
        assert texts[1].content == "world!"

        dones = events_of_type(events, "done")
        assert len(dones) == 1
        done = dones[0]
        assert done.response.content == "Hello world!"
        assert done.response.usage.input_tokens == 100
        assert done.response.usage.output_tokens == 10
        assert done.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_tool_calls_accumulated(self):
        """Tool call chunks are accumulated and yield tool_done events."""
        config = make_openai_config()
        chunks = [
            _openai_tool_chunk(0, tool_id="call_1", name="shell",
                               arguments_delta='{"com'),
            _openai_tool_chunk(0, arguments_delta='mand": "ls"}'),
            _openai_finish_chunk("tool_calls"),
            _openai_usage_chunk(),
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream(chunks),
            )

            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "List files"}],
            ))

        tool_dones = events_of_type(events, "tool_done")
        assert len(tool_dones) == 1
        assert tool_dones[0].tool_call.name == "shell"
        assert tool_dones[0].tool_call.id == "call_1"
        assert tool_dones[0].tool_call.input == {"command": "ls"}

        done = events_of_type(events, "done")[0]
        assert done.stop_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """Multiple parallel tool calls from interleaved chunks."""
        config = make_openai_config()
        chunks = [
            _openai_tool_chunk(0, tool_id="call_1", name="shell",
                               arguments_delta='{"command": "ls"}'),
            _openai_tool_chunk(1, tool_id="call_2", name="file_read",
                               arguments_delta='{"path": "/tmp/x"}'),
            _openai_finish_chunk("tool_calls"),
            _openai_usage_chunk(),
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream(chunks),
            )

            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Do things"}],
            ))

        tool_dones = events_of_type(events, "tool_done")
        assert len(tool_dones) == 2
        names = {td.tool_call.name for td in tool_dones}
        assert names == {"shell", "file_read"}

    @pytest.mark.asyncio
    async def test_text_before_tool_calls(self):
        """Text content can precede tool calls."""
        config = make_openai_config()
        chunks = [
            _openai_text_chunk("Let me check. "),
            _openai_tool_chunk(0, tool_id="call_1", name="shell",
                               arguments_delta='{"command": "ls"}'),
            _openai_finish_chunk("tool_calls"),
            _openai_usage_chunk(),
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream(chunks),
            )

            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Check files"}],
            ))

        texts = events_of_type(events, "text")
        assert len(texts) == 1
        assert texts[0].content == "Let me check. "

        tool_dones = events_of_type(events, "tool_done")
        assert len(tool_dones) == 1

    @pytest.mark.asyncio
    async def test_kwargs_system_prepended(self):
        """System message prepended, frequency_penalty set, stream=True."""
        config = make_openai_config(max_tokens=2048)
        chunks = [
            _openai_text_chunk("ok", finish_reason="stop"),
            _openai_usage_chunk(),
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream(chunks),
            )

            await collect_events(stream(
                config=config, system="Be helpful.",
                messages=[{"role": "user", "content": "Hi"}],
            ))

        kw = client.chat.completions.create.call_args.kwargs
        assert kw["stream"] is True
        assert kw["messages"][0] == {"role": "system", "content": "Be helpful."}
        assert kw["max_tokens"] == 2048
        assert kw["frequency_penalty"] == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_stream_options_include_usage(self):
        """stream_options with include_usage is passed."""
        config = make_openai_config()
        chunks = [
            _openai_text_chunk("ok", finish_reason="stop"),
            _openai_usage_chunk(),
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream(chunks),
            )

            await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            ))

        kw = client.chat.completions.create.call_args.kwargs
        assert kw.get("stream_options") == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_error_during_stream(self):
        """SDK error mid-stream raises ProviderError."""
        config = make_openai_config()

        class ErrorStream(MockOpenAIStream):
            async def _aiter_impl(self):
                yield _openai_text_chunk("partial")
                raise openai_sdk.APIStatusError(
                    message="server error",
                    response=MagicMock(status_code=500),
                    body=None,
                )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = MagicMock(
                return_value=ErrorStream([]),
            )

            with pytest.raises(ProviderError):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                ))

    @pytest.mark.asyncio
    async def test_degeneration_detected(self):
        """Degenerate output detected in OpenAI path."""
        config = make_openai_config()
        degen = "x" * (_DEGEN_CHAR_THRESHOLD + 10)
        chunks = [
            _openai_text_chunk("Hello "),
            _openai_text_chunk(degen),
            _openai_finish_chunk("stop"),
            _openai_usage_chunk(),
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream(chunks),
            )

            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            ))

        done = events_of_type(events, "done")[0]
        assert done.response.degenerate is True

    @pytest.mark.asyncio
    async def test_no_system_role_quirk(self):
        """no_system_role quirk folds system into first user message."""
        config = make_config(
            providers={
                "ollama": make_provider(
                    key="ollama", type="openai", api_key="sk-test",
                    base_url="http://localhost:11434/v1",
                    quirks=["no_system_role"],
                )
            },
            default_model="ollama/test-model",
        )
        chunks = [
            _openai_text_chunk("ok", finish_reason="stop"),
            _openai_usage_chunk(),
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream(chunks),
            )

            await collect_events(stream(
                config=config, system="System prompt",
                messages=[{"role": "user", "content": "Hi"}],
            ))

        kw = client.chat.completions.create.call_args.kwargs
        assert kw["messages"][0]["role"] == "user"
        assert "System prompt" in kw["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Tests: complete() wraps stream() — backward compatibility
# ---------------------------------------------------------------------------

class TestCompleteWrapsStream:

    @pytest.mark.asyncio
    async def test_anthropic_backward_compat(self):
        """complete() returns the same Response shape as before."""
        config = make_config()
        final_msg = _anthropic_final_message(
            text="Hello!", model="claude-sonnet-4-20250514",
            input_tokens=150, output_tokens=30,
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_text("Hello!"), _anthropic_message_stop()],
                final_msg,
            )

            response = await complete(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert isinstance(response, Response)
        assert response.content == "Hello!"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.usage.input_tokens == 150
        assert response.usage.output_tokens == 30
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_openai_backward_compat(self):
        """complete() via OpenAI path returns correct Response."""
        config = make_openai_config()
        chunks = [
            _openai_text_chunk("Hello!"),
            _openai_finish_chunk("stop"),
            _openai_usage_chunk(prompt_tokens=100, completion_tokens=20),
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream(chunks),
            )

            response = await complete(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert isinstance(response, Response)
        assert response.content == "Hello!"
        assert response.usage.input_tokens == 100
        assert response.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_no_21k_cap(self):
        """max_tokens above 21000 is NOT capped after refactor."""
        config = make_config(max_tokens=50000)
        final_msg = _anthropic_final_message()

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_message_stop()], final_msg,
            )

            await complete(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.messages.stream.call_args.kwargs
        assert kw["max_tokens"] == 50000

    @pytest.mark.asyncio
    async def test_tool_calls_in_response(self):
        """Tool calls from stream are in the complete() Response."""
        config = make_config()
        tool_block = _anthropic_tool_block(
            "toolu_1", "file_read", {"path": "/tmp/test"},
        )
        final_msg = _anthropic_final_message(
            text="Reading file.",
            tool_calls=[{
                "id": "toolu_1", "name": "file_read",
                "input": {"path": "/tmp/test"},
            }],
            stop_reason="tool_use",
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [
                    _anthropic_text("Reading file."),
                    _anthropic_block_start(
                        1, "tool_use", tool_id="toolu_1", tool_name="file_read",
                    ),
                    _anthropic_block_stop(1, tool_block),
                    _anthropic_message_stop(),
                ],
                final_msg,
            )

            response = await complete(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Read file"}],
            )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "file_read"
        assert response.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_thinking_blocks_in_response(self):
        """Thinking blocks present in complete() Response."""
        config = make_config()
        final_msg = _anthropic_final_message(
            text="42",
            thinking=[{"thinking": "Deep thought...", "signature": "sig-123"}],
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [
                    _anthropic_thinking("Deep thought..."),
                    _anthropic_text("42"),
                    _anthropic_message_stop(),
                ],
                final_msg,
            )

            response = await complete(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Think"}],
                thinking="high",
            )

        assert response.content == "42"
        assert len(response.thinking) == 1
        assert response.thinking[0].thinking == "Deep thought..."

    @pytest.mark.asyncio
    async def test_degeneration_flagged(self):
        """Degenerate output sets degenerate=True on Response."""
        config = make_config()
        degen_text = "Hello " + "x" * (_DEGEN_CHAR_THRESHOLD + 10)
        final_msg = _anthropic_final_message(text=degen_text)

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [
                    _anthropic_text(degen_text),
                    _anthropic_message_stop(),
                ],
                final_msg,
            )

            response = await complete(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert response.degenerate is True
        assert _DEGEN_WARNING in response.content

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """Empty response produces Response with empty content."""
        config = make_config()
        final_msg = _anthropic_final_message(text="")

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_message_stop()], final_msg,
            )

            response = await complete(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert response.content == ""
        assert response.tool_calls == []
        assert response.thinking == []

    @pytest.mark.asyncio
    async def test_provider_error_propagated(self):
        """ProviderError from stream setup propagates through complete()."""
        config = make_config()

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.side_effect = anthropic_sdk.APIStatusError(
                message="overloaded",
                response=MagicMock(status_code=529),
                body=None,
            )

            with pytest.raises(ProviderError):
                await complete(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                )


# ---------------------------------------------------------------------------
# Tests: Regression — existing behavior preserved
# ---------------------------------------------------------------------------

class TestExistingBehavior:

    @pytest.mark.asyncio
    async def test_model_override(self):
        """Explicit model parameter overrides config default."""
        config = make_config()
        final_msg = _anthropic_final_message(model="claude-haiku-4-5-20250514")

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_message_stop()], final_msg,
            )

            await complete(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                model="anthropic/claude-haiku-4-5-20250514",
            )

        kw = client.messages.stream.call_args.kwargs
        assert kw["model"] == "claude-haiku-4-5-20250514"

    @pytest.mark.asyncio
    async def test_max_tokens_override(self):
        """Explicit max_tokens overrides config default."""
        config = make_config(max_tokens=8192)
        final_msg = _anthropic_final_message()

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_message_stop()], final_msg,
            )

            await complete(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=2048,
            )

        kw = client.messages.stream.call_args.kwargs
        assert kw["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_openai_frequency_penalty(self):
        """OpenAI path includes frequency_penalty."""
        config = make_openai_config()
        chunks = [
            _openai_text_chunk("ok", finish_reason="stop"),
            _openai_usage_chunk(),
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream(chunks),
            )

            await complete(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.chat.completions.create.call_args.kwargs
        assert kw["frequency_penalty"] == pytest.approx(0.3)
