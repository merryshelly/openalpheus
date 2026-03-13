"""Tests for provider tool_use support.

Extends the existing provider adapter with:
    - tools parameter on complete()
    - ToolCall dataclass (id, name, input)
    - Response.tool_calls field (default [])
    - Message format conversion: normalized ↔ Anthropic/OpenAI native
    - Backward compatibility: no tools → same behavior as Phase 1

Provider converts normalized message history to/from native format on each call.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import complete, Response, Usage, ToolCall
from openalph.tools import ToolDef


def make_provider(key="anthropic", type="anthropic", api_key="sk-test", base_url=None, quirks=None):
    return ProviderConfig(
        key=key,
        type=type,
        api_key=api_key,
        base_url=base_url,
        quirks=quirks or [],
    )


def make_config(**kwargs):
    defaults = {
        "name": "test",
        "default_model": "anthropic/claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "providers": {"anthropic": make_provider(key="anthropic")},
        "workspace": Path("/tmp/test"),
        "max_iterations": 25,
        "truncation_limit": 50000,
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


SAMPLE_TOOL = ToolDef(
    name="shell",
    description="Execute a shell command",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
        },
        "required": ["command"],
    },
    config={},
)


# --- Mock helpers for streaming ---

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


def _anthropic_final_message(text="", model="test-model",
                             input_tokens=100, output_tokens=50,
                             cache_read=0, cache_create=0,
                             stop_reason="end_turn", tool_calls=None):
    msg = MagicMock()
    msg.model = model
    msg.stop_reason = stop_reason
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    msg.usage.cache_read_input_tokens = cache_read
    msg.usage.cache_creation_input_tokens = cache_create

    content_blocks = []
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


class MockOpenAIStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for chunk in self._chunks:
            yield chunk


def _openai_text_chunk(text, finish_reason=None):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = text
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = None
    return chunk


def _openai_usage_chunk(prompt_tokens=100, completion_tokens=50):
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = MagicMock()
    chunk.usage.prompt_tokens = prompt_tokens
    chunk.usage.completion_tokens = completion_tokens
    return chunk


# --- ToolCall dataclass ---


class TestToolCall:

    def test_has_required_fields(self):
        tc = ToolCall(id="toolu_123", name="shell", input={"command": "echo hi"})
        assert tc.id == "toolu_123"
        assert tc.name == "shell"
        assert tc.input == {"command": "echo hi"}


# --- Response backward compatibility ---


class TestResponseBackwardCompat:

    def test_tool_calls_defaults_empty(self):
        """Response.tool_calls defaults to empty list."""
        r = Response(
            content="Hello",
            tool_calls=[],
            model="test",
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
        )
        assert r.tool_calls == []

    def test_response_with_tool_calls(self):
        """Response can carry tool_calls."""
        tc = ToolCall(id="tc_1", name="shell", input={"command": "ls"})
        r = Response(
            content="",
            tool_calls=[tc],
            model="test",
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="tool_use",
        )
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "shell"


# --- Anthropic tool support ---


class TestAnthropicTools:

    def _mock_text_response_events(self, text="Hello"):
        """Create mock events for a text-only response."""
        final_msg = _anthropic_final_message(text=text)
        return [_anthropic_text(text), _anthropic_message_stop()], final_msg

    def _mock_tool_use_response(self, tool_id="toolu_123", tool_name="shell",
                                 tool_input=None, text=None):
        """Create mock events and final message for a tool_use response."""
        tool_input = tool_input or {"command": "echo hi"}
        
        events = []
        if text:
            events.append(_anthropic_text(text))
        
        tool_block = _anthropic_tool_block(tool_id, tool_name, tool_input)
        events.append(_anthropic_block_start(len(events), "tool_use", tool_id=tool_id, tool_name=tool_name))
        events.append(_anthropic_block_stop(len(events), tool_block))
        events.append(_anthropic_message_stop())
        
        final_msg = _anthropic_final_message(
            text=text or "",
            tool_calls=[{"id": tool_id, "name": tool_name, "input": tool_input}],
            stop_reason="tool_use",
        )
        
        return events, final_msg

    @pytest.mark.asyncio
    async def test_tools_sent_to_anthropic(self):
        """Tools are passed to Anthropic API in native format."""
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")}
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            
            events, final_msg = self._mock_text_response_events()
            client.messages.stream.return_value = MockAnthropicStream(events, final_msg)

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                tools=[SAMPLE_TOOL],
            )

        kw = client.messages.stream.call_args.kwargs
        assert "tools" in kw
        assert len(kw["tools"]) == 1
        assert kw["tools"][0]["name"] == "shell"
        assert "input_schema" in kw["tools"][0]

    @pytest.mark.asyncio
    async def test_no_tools_omits_parameter(self):
        """tools=None → tools not sent to API (or sent as empty)."""
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")}
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            
            events, final_msg = self._mock_text_response_events()
            client.messages.stream.return_value = MockAnthropicStream(events, final_msg)

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                tools=None,
            )

        kw = client.messages.stream.call_args.kwargs
        # Either tools not in kwargs, or empty
        tools = kw.get("tools")
        assert tools is None or tools == []

    @pytest.mark.asyncio
    async def test_tool_use_response_parsed(self):
        """Anthropic tool_use blocks are parsed into Response.tool_calls."""
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")}
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            
            events, final_msg = self._mock_tool_use_response(
                tool_id="toolu_abc",
                tool_name="shell",
                tool_input={"command": "ls -la"},
            )
            client.messages.stream.return_value = MockAnthropicStream(events, final_msg)

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "List files"}],
                tools=[SAMPLE_TOOL],
            )

        assert response.stop_reason == "tool_use"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "toolu_abc"
        assert response.tool_calls[0].name == "shell"
        assert response.tool_calls[0].input == {"command": "ls -la"}

    @pytest.mark.asyncio
    async def test_text_and_tool_use_combined(self):
        """Response with both text and tool_use preserves both."""
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")}
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            
            events, final_msg = self._mock_tool_use_response(
                text="Let me check that for you.",
                tool_name="shell",
                tool_input={"command": "pwd"},
            )
            client.messages.stream.return_value = MockAnthropicStream(events, final_msg)

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Where am I?"}],
                tools=[SAMPLE_TOOL],
            )

        assert response.content == "Let me check that for you."
        assert len(response.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """Multiple tool_use blocks in one response."""
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")}
        )

        # Build events with 2 tool_use blocks
        tool1_block = _anthropic_tool_block("toolu_1", "shell", {"command": "ls"})
        tool2_block = _anthropic_tool_block("toolu_2", "shell", {"command": "pwd"})
        
        events = [
            _anthropic_block_start(0, "tool_use", tool_id="toolu_1", tool_name="shell"),
            _anthropic_block_stop(0, tool1_block),
            _anthropic_block_start(1, "tool_use", tool_id="toolu_2", tool_name="shell"),
            _anthropic_block_stop(1, tool2_block),
            _anthropic_message_stop(),
        ]
        
        final_msg = _anthropic_final_message(
            tool_calls=[
                {"id": "toolu_1", "name": "shell", "input": {"command": "ls"}},
                {"id": "toolu_2", "name": "shell", "input": {"command": "pwd"}},
            ],
            stop_reason="tool_use",
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(events, final_msg)

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Do two things"}],
                tools=[SAMPLE_TOOL],
            )

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].id == "toolu_1"
        assert response.tool_calls[1].id == "toolu_2"


# --- Anthropic message format conversion ---


class TestAnthropicMessageConversion:

    def _mock_text_response(self):
        """Create mock for text response."""
        events = [_anthropic_text("OK"), _anthropic_message_stop()]
        final_msg = _anthropic_final_message(text="OK")
        return events, final_msg

    @pytest.mark.asyncio
    async def test_tool_result_in_history_converted(self):
        """Normalized tool result messages are converted to Anthropic format.

        Normalized: {"role": "tool", "tool_call_id": "x", "content": "result"}
        Anthropic: {"role": "user", "content": [{"type": "tool_result", ...}]}
        """
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")}
        )

        messages = [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": "I'll check",
                "tool_calls": [
                    ToolCall(id="toolu_1", name="shell", input={"command": "ls"})
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "content": "file1.txt\nfile2.txt",
                "is_error": False,
            },
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            
            events, final_msg = self._mock_text_response()
            client.messages.stream.return_value = MockAnthropicStream(events, final_msg)

            await complete(
                config=config,
                system="Test",
                messages=messages,
                tools=[SAMPLE_TOOL],
            )

        kw = client.messages.stream.call_args.kwargs
        api_messages = kw["messages"]

        # First message: user (unchanged)
        assert api_messages[0]["role"] == "user"
        assert api_messages[0]["content"] == "List files"

        # Second message: assistant with tool_use content blocks
        assert api_messages[1]["role"] == "assistant"
        assistant_content = api_messages[1]["content"]
        assert isinstance(assistant_content, list)
        # Should contain text block and tool_use block
        types = [block["type"] for block in assistant_content]
        assert "tool_use" in types

        # Third message: user with tool_result
        assert api_messages[2]["role"] == "user"
        tool_result_content = api_messages[2]["content"]
        assert isinstance(tool_result_content, list)
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "toolu_1"

    @pytest.mark.asyncio
    async def test_error_tool_result_converted(self):
        """Tool error results include is_error flag in Anthropic format."""
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")}
        )

        messages = [
            {"role": "user", "content": "Read missing file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    ToolCall(
                        id="toolu_1",
                        name="file_read",
                        input={"path": "/missing.txt"},
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "content": "File not found: /missing.txt",
                "is_error": True,
            },
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            
            events, final_msg = self._mock_text_response()
            client.messages.stream.return_value = MockAnthropicStream(events, final_msg)

            await complete(
                config=config,
                system="Test",
                messages=messages,
                tools=[SAMPLE_TOOL],
            )

        kw = client.messages.stream.call_args.kwargs
        api_messages = kw["messages"]
        tool_result = api_messages[2]["content"][0]
        assert tool_result["is_error"] is True


# --- OpenAI tool support ---


class TestOpenAITools:

    def _mock_text_chunks(self, text="Hello"):
        """Create mock OpenAI streaming chunks for text response."""
        chunks = [
            _openai_text_chunk(text, finish_reason="stop"),
            _openai_usage_chunk(),
        ]
        return chunks

    def _mock_tool_call_chunks(self, calls=None):
        """Create mock OpenAI streaming chunks with tool calls."""
        if calls is None:
            calls = [{"id": "call_123", "name": "shell", "arguments": '{"command": "echo hi"}'}]
        
        chunks = []
        for call in calls:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = None
            
            tc = MagicMock()
            tc.index = 0
            tc.id = call["id"]
            tc.function = MagicMock()
            tc.function.name = call["name"]
            tc.function.arguments = call["arguments"]
            chunk.choices[0].delta.tool_calls = [tc]
            chunk.choices[0].finish_reason = "tool_calls"  # Set finish_reason
            chunk.usage = None
            chunks.append(chunk)
        
        # Usage chunk
        chunks.append(_openai_usage_chunk())
        
        return chunks

    @pytest.mark.asyncio
    async def test_tools_sent_as_functions(self):
        """Tools are sent to OpenAI API in function format."""
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            },
            default_model="openrouter/test-model",
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(self._mock_text_chunks())
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                tools=[SAMPLE_TOOL],
            )

        kw = client.chat.completions.create.call_args.kwargs
        assert "tools" in kw
        assert len(kw["tools"]) == 1
        assert kw["tools"][0]["type"] == "function"
        assert kw["tools"][0]["function"]["name"] == "shell"
        assert "parameters" in kw["tools"][0]["function"]

    @pytest.mark.asyncio
    async def test_tool_call_response_parsed(self):
        """OpenAI tool_calls are parsed into Response.tool_calls."""
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            },
            default_model="openrouter/test-model",
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(self._mock_tool_call_chunks())
            )

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Do something"}],
                tools=[SAMPLE_TOOL],
            )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_123"
        assert response.tool_calls[0].name == "shell"
        assert response.tool_calls[0].input == {"command": "echo hi"}

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_empty(self):
        """Text-only response → empty tool_calls."""
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            },
            default_model="openrouter/test-model",
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(self._mock_text_chunks("Just text"))
            )

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert response.tool_calls == []
        assert response.content == "Just text"


# --- OpenAI message format conversion ---


class TestOpenAIMessageConversion:

    def _mock_text_chunks(self):
        """Create mock OpenAI streaming chunks."""
        return [
            _openai_text_chunk("OK", finish_reason="stop"),
            _openai_usage_chunk(prompt_tokens=10, completion_tokens=5),
        ]

    @pytest.mark.asyncio
    async def test_tool_result_in_history_converted(self):
        """Normalized tool result messages are converted to OpenAI format.

        Normalized: {"role": "tool", "tool_call_id": "x", "content": "result"}
        OpenAI: {"role": "tool", "tool_call_id": "x", "content": "result"}
        """
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            },
            default_model="openrouter/test-model",
        )

        messages = [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": "Checking",
                "tool_calls": [
                    ToolCall(id="call_1", name="shell", input={"command": "ls"})
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "file1.txt\nfile2.txt",
                "is_error": False,
            },
        ]

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(self._mock_text_chunks())
            )

            await complete(
                config=config,
                system="Test",
                messages=messages,
                tools=[SAMPLE_TOOL],
            )

        kw = client.chat.completions.create.call_args.kwargs
        api_messages = kw["messages"]

        # First: system (prepended by provider)
        assert api_messages[0]["role"] == "system"

        # Second: user
        assert api_messages[1]["role"] == "user"
        assert api_messages[1]["content"] == "List files"

        # Third: assistant with tool_calls in OpenAI format
        assert api_messages[2]["role"] == "assistant"
        assert "tool_calls" in api_messages[2]
        tc = api_messages[2]["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "shell"

        # Fourth: tool result
        assert api_messages[3]["role"] == "tool"
        assert api_messages[3]["tool_call_id"] == "call_1"
        assert "file1.txt" in api_messages[3]["content"]


# --- Backward compatibility ---


class TestBackwardCompat:

    @pytest.mark.asyncio
    async def test_no_tools_anthropic_unchanged(self):
        """Without tools, Anthropic path behaves exactly as Phase 1."""
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")}
        )

        events = [_anthropic_text("Hello"), _anthropic_message_stop()]
        final_msg = _anthropic_final_message(text="Hello", stop_reason="end_turn")

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream.return_value = MockAnthropicStream(events, final_msg)

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert response.content == "Hello"
        assert response.tool_calls == []
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_no_tools_openai_unchanged(self):
        """Without tools, OpenAI path behaves exactly as Phase 1."""
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            },
            default_model="openrouter/test-model",
        )

        chunks = [
            _openai_text_chunk("Hello", finish_reason="stop"),
            _openai_usage_chunk(),
        ]

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(chunks)
            )

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert response.content == "Hello"
        assert response.tool_calls == []
        assert response.stop_reason == "stop"
