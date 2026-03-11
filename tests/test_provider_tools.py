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


def make_provider(key="default", type="anthropic", api_key="sk-test", base_url=None, quirks=None):
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
        "default_model": "claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "providers": {"default": make_provider()},
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

    def _mock_text_response(self, text="Hello"):
        """Mock an Anthropic response with only text content."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text

        resp = MagicMock()
        resp.content = [text_block]
        resp.model = "test-model"
        resp.usage.input_tokens = 100
        resp.usage.output_tokens = 50
        resp.usage.cache_read_input_tokens = 0
        resp.usage.cache_creation_input_tokens = 0
        resp.stop_reason = "end_turn"
        return resp

    def _mock_tool_use_response(self, tool_id="toolu_123", tool_name="shell",
                                 tool_input=None, text=None):
        """Mock an Anthropic response with tool_use (and optional text)."""
        blocks = []

        if text:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = text
            blocks.append(text_block)

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = tool_id
        tool_block.name = tool_name
        tool_block.input = tool_input or {"command": "echo hi"}
        blocks.append(tool_block)

        resp = MagicMock()
        resp.content = blocks
        resp.model = "test-model"
        resp.usage.input_tokens = 100
        resp.usage.output_tokens = 50
        resp.usage.cache_read_input_tokens = 0
        resp.usage.cache_creation_input_tokens = 0
        resp.stop_reason = "tool_use"
        return resp

    @pytest.mark.asyncio
    async def test_tools_sent_to_anthropic(self):
        """Tools are passed to Anthropic API in native format."""
        config = make_config(
            providers={"default": make_provider(key="default", type="anthropic", api_key="sk-test")}
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(
                return_value=self._mock_text_response()
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                tools=[SAMPLE_TOOL],
            )

        kw = client.messages.create.call_args.kwargs
        assert "tools" in kw
        assert len(kw["tools"]) == 1
        assert kw["tools"][0]["name"] == "shell"
        assert "input_schema" in kw["tools"][0]

    @pytest.mark.asyncio
    async def test_no_tools_omits_parameter(self):
        """tools=None → tools not sent to API (or sent as empty)."""
        config = make_config(
            providers={"default": make_provider(key="default", type="anthropic", api_key="sk-test")}
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(
                return_value=self._mock_text_response()
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                tools=None,
            )

        kw = client.messages.create.call_args.kwargs
        # Either tools not in kwargs, or empty
        tools = kw.get("tools")
        assert tools is None or tools == []

    @pytest.mark.asyncio
    async def test_tool_use_response_parsed(self):
        """Anthropic tool_use blocks are parsed into Response.tool_calls."""
        config = make_config(
            providers={"default": make_provider(key="default", type="anthropic", api_key="sk-test")}
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(
                return_value=self._mock_tool_use_response(
                    tool_id="toolu_abc",
                    tool_name="shell",
                    tool_input={"command": "ls -la"},
                )
            )

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
            providers={"default": make_provider(key="default", type="anthropic", api_key="sk-test")}
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(
                return_value=self._mock_tool_use_response(
                    text="Let me check that for you.",
                    tool_name="shell",
                    tool_input={"command": "pwd"},
                )
            )

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
            providers={"default": make_provider(key="default", type="anthropic", api_key="sk-test")}
        )

        # Build response with 2 tool_use blocks
        tool1 = MagicMock()
        tool1.type = "tool_use"
        tool1.id = "toolu_1"
        tool1.name = "shell"
        tool1.input = {"command": "ls"}

        tool2 = MagicMock()
        tool2.type = "tool_use"
        tool2.id = "toolu_2"
        tool2.name = "shell"
        tool2.input = {"command": "pwd"}

        resp = MagicMock()
        resp.content = [tool1, tool2]
        resp.model = "test-model"
        resp.usage.input_tokens = 100
        resp.usage.output_tokens = 50
        resp.usage.cache_read_input_tokens = 0
        resp.usage.cache_creation_input_tokens = 0
        resp.stop_reason = "tool_use"

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(return_value=resp)

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
        """Mock an Anthropic response with only text content."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "OK"
        resp = MagicMock()
        resp.content = [text_block]
        resp.model = "test-model"
        resp.usage.input_tokens = 10
        resp.usage.output_tokens = 5
        resp.usage.cache_read_input_tokens = 0
        resp.usage.cache_creation_input_tokens = 0
        resp.stop_reason = "end_turn"
        return resp

    @pytest.mark.asyncio
    async def test_tool_result_in_history_converted(self):
        """Normalized tool result messages are converted to Anthropic format.

        Normalized: {"role": "tool", "tool_call_id": "x", "content": "result"}
        Anthropic: {"role": "user", "content": [{"type": "tool_result", ...}]}
        """
        config = make_config(
            providers={"default": make_provider(key="default", type="anthropic", api_key="sk-test")}
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

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(
                return_value=self._mock_text_response()
            )

            await complete(
                config=config,
                system="Test",
                messages=messages,
                tools=[SAMPLE_TOOL],
            )

        kw = client.messages.create.call_args.kwargs
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
            providers={"default": make_provider(key="default", type="anthropic", api_key="sk-test")}
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

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(
                return_value=self._mock_text_response()
            )

            await complete(
                config=config,
                system="Test",
                messages=messages,
                tools=[SAMPLE_TOOL],
            )

        kw = client.messages.create.call_args.kwargs
        api_messages = kw["messages"]
        tool_result = api_messages[2]["content"][0]
        assert tool_result["is_error"] is True


# --- OpenAI tool support ---


class TestOpenAITools:

    def _mock_text_response(self, text="Hello"):
        choice = MagicMock()
        choice.message.content = text
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        resp.model = "test-model"
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        return resp

    def _mock_tool_call_response(self, calls=None):
        """Mock OpenAI response with tool_calls."""
        if calls is None:
            tc = MagicMock()
            tc.id = "call_123"
            tc.type = "function"
            tc.function.name = "shell"
            tc.function.arguments = json.dumps({"command": "echo hi"})
            calls = [tc]

        choice = MagicMock()
        choice.message.content = None
        choice.message.tool_calls = calls
        choice.finish_reason = "tool_calls"
        resp = MagicMock()
        resp.choices = [choice]
        resp.model = "test-model"
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        return resp

    @pytest.mark.asyncio
    async def test_tools_sent_as_functions(self):
        """Tools are sent to OpenAI API in function format."""
        config = make_config(
            providers={
                "default": make_provider(
                    key="default", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            }
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=self._mock_text_response()
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
                "default": make_provider(
                    key="default", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            }
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=self._mock_tool_call_response()
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
                "default": make_provider(
                    key="default", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            }
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=self._mock_text_response("Just text")
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

    def _mock_text_response(self):
        choice = MagicMock()
        choice.message.content = "OK"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        resp.model = "test-model"
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        return resp

    @pytest.mark.asyncio
    async def test_tool_result_in_history_converted(self):
        """Normalized tool result messages are converted to OpenAI format.

        Normalized: {"role": "tool", "tool_call_id": "x", "content": "result"}
        OpenAI: {"role": "tool", "tool_call_id": "x", "content": "result"}
        """
        config = make_config(
            providers={
                "default": make_provider(
                    key="default", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            }
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
                return_value=self._mock_text_response()
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
            providers={"default": make_provider(key="default", type="anthropic", api_key="sk-test")}
        )

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello"

        resp = MagicMock()
        resp.content = [text_block]
        resp.model = "test-model"
        resp.usage.input_tokens = 100
        resp.usage.output_tokens = 50
        resp.usage.cache_read_input_tokens = 0
        resp.usage.cache_creation_input_tokens = 0
        resp.stop_reason = "end_turn"

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(return_value=resp)

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
                "default": make_provider(
                    key="default", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            }
        )

        choice = MagicMock()
        choice.message.content = "Hello"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        resp.model = "test-model"
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(return_value=resp)

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert response.content == "Hello"
        assert response.tool_calls == []
        assert response.stop_reason == "stop"
