"""Tests for extended thinking support (kdsn.62).

Covers:
    - Adaptive thinking (Opus/Sonnet 4-6): type=adaptive + output_config.effort
    - Budget-based thinking (older models): type=enabled + budget_tokens
    - Thinking block parsing from Anthropic responses
    - Thinking block round-tripping in conversation history
    - Prompt caching (cache_control markers)
    - OpenRouter reasoning (extra_body injection)
    - Config loading with thinking field
    - Session persistence with thinking blocks
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from openalph.config import AgentConfig, ProviderConfig, load_config
from openalph.provider import (
    complete,
    Response,
    Usage,
    ThinkingBlock,
    _supports_adaptive_thinking,
    _thinking_effort,
    _thinking_budget,
    _convert_messages_for_anthropic,
    _parse_anthropic_response,
)


# ---------------------------------------------------------------------------
# Helpers
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
        "default_model": "anthropic/claude-opus-4-6-20250605",
        "max_tokens": 8192,
        "providers": {"anthropic": make_provider(key="anthropic")},
        "workspace": Path("/tmp/test"),
        "thinking": "off",
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def mock_anthropic_response_with_thinking(
    text="Hello",
    thinking_text="Let me think...",
    thinking_signature="sig123abc",
    model="claude-opus-4-6-20250605",
    input_tokens=100,
    output_tokens=50,
):
    """Mock Anthropic response containing thinking + text blocks."""
    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    thinking_block.thinking = thinking_text
    thinking_block.signature = thinking_signature

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    resp = MagicMock()
    resp.content = [thinking_block, text_block]
    resp.model = model
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    resp.usage.cache_read_input_tokens = 0
    resp.usage.cache_creation_input_tokens = 0
    resp.stop_reason = "end_turn"
    return resp


def mock_anthropic_response_text_only(
    text="Hello",
    model="claude-opus-4-6-20250605",
    input_tokens=100,
    output_tokens=50,
):
    """Mock Anthropic response with text only (no thinking)."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    resp = MagicMock()
    resp.content = [text_block]
    resp.model = model
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    resp.usage.cache_read_input_tokens = 0
    resp.usage.cache_creation_input_tokens = 0
    resp.stop_reason = "end_turn"
    return resp



class MockOpenAIStream:
    """Mock for OpenAI streaming response."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for chunk in self._chunks:
            yield chunk


def _make_openai_chunk(content="Hello", finish_reason=None, usage=None):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = content
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = usage
    return chunk

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


def _make_text_event(text):
    e = MagicMock()
    e.type = "text"
    e.text = text
    return e


def _make_message_stop():
    e = MagicMock()
    e.type = "message_stop"
    return e


# ---------------------------------------------------------------------------
# _supports_adaptive_thinking
# ---------------------------------------------------------------------------


class TestSupportsAdaptiveThinking:
    """Model detection for adaptive vs budget-based thinking."""

    def test_opus_4_6(self):
        assert _supports_adaptive_thinking("claude-opus-4-6-20250605") is True

    def test_sonnet_4_6(self):
        assert _supports_adaptive_thinking("claude-sonnet-4-6-20250514") is True

    def test_opus_4_6_no_date(self):
        assert _supports_adaptive_thinking("claude-opus-4-6") is True

    def test_sonnet_4_6_no_date(self):
        assert _supports_adaptive_thinking("claude-sonnet-4-6") is True

    def test_sonnet_3_5(self):
        assert _supports_adaptive_thinking("claude-3-5-sonnet-20241022") is False

    def test_haiku_3_5(self):
        assert _supports_adaptive_thinking("claude-3-5-haiku-20241022") is False

    def test_sonnet_4_0(self):
        assert _supports_adaptive_thinking("claude-sonnet-4-20250514") is False

    def test_empty_string(self):
        assert _supports_adaptive_thinking("") is False


# ---------------------------------------------------------------------------
# Effort / budget mapping
# ---------------------------------------------------------------------------


class TestThinkingEffort:
    """Mapping from config thinking level to Anthropic effort."""

    def test_low(self):
        assert _thinking_effort("low") == "low"

    def test_medium(self):
        assert _thinking_effort("medium") == "medium"

    def test_high(self):
        assert _thinking_effort("high") == "high"


class TestThinkingBudget:
    """Budget-based thinking for older models."""

    def test_low_budget(self):
        budget, max_tok = _thinking_budget("low", base_max_tokens=8192, model_max_tokens=200000)
        assert budget == 2048
        assert max_tok == 8192 + 2048

    def test_medium_budget(self):
        budget, max_tok = _thinking_budget("medium", base_max_tokens=8192, model_max_tokens=200000)
        assert budget == 8192
        assert max_tok == 8192 + 8192

    def test_high_budget(self):
        budget, max_tok = _thinking_budget("high", base_max_tokens=8192, model_max_tokens=200000)
        assert budget == 16384
        assert max_tok == 8192 + 16384

    def test_budget_clamped_to_model_max(self):
        """When base + budget would exceed model max, clamp."""
        budget, max_tok = _thinking_budget("high", base_max_tokens=190000, model_max_tokens=200000)
        assert max_tok == 200000
        # budget should be adjusted so there's room for output
        assert budget <= 200000 - 1024  # at least 1024 for output


# ---------------------------------------------------------------------------
# Thinking block parsing
# ---------------------------------------------------------------------------


class TestParseThinkingBlocks:
    """_parse_anthropic_response with thinking blocks."""

    def test_thinking_and_text(self):
        raw = mock_anthropic_response_with_thinking(
            text="The answer is 42.",
            thinking_text="I need to calculate...",
            thinking_signature="sigABC",
        )
        response = _parse_anthropic_response(raw)
        assert response.content == "The answer is 42."
        assert len(response.thinking) == 1
        assert response.thinking[0].thinking == "I need to calculate..."
        assert response.thinking[0].signature == "sigABC"

    def test_text_only_no_thinking(self):
        raw = mock_anthropic_response_text_only(text="Simple answer")
        response = _parse_anthropic_response(raw)
        assert response.content == "Simple answer"
        assert response.thinking == []

    def test_multiple_thinking_blocks(self):
        """Interleaved thinking can produce multiple thinking blocks."""
        t1 = MagicMock(type="thinking", thinking="First thought", signature="sig1")
        text1 = MagicMock(type="text", text="Partial response")
        t2 = MagicMock(type="thinking", thinking="Second thought", signature="sig2")
        text2 = MagicMock(type="text", text="Final response")

        raw = MagicMock()
        raw.content = [t1, text1, t2, text2]
        raw.model = "claude-opus-4-6-20250605"
        raw.usage.input_tokens = 100
        raw.usage.output_tokens = 200
        raw.usage.cache_read_input_tokens = 0
        raw.usage.cache_creation_input_tokens = 0
        raw.stop_reason = "end_turn"

        response = _parse_anthropic_response(raw)
        assert response.content == "Partial response\nFinal response"
        assert len(response.thinking) == 2
        assert response.thinking[0].signature == "sig1"
        assert response.thinking[1].signature == "sig2"


# ---------------------------------------------------------------------------
# ThinkingBlock dataclass
# ---------------------------------------------------------------------------


class TestThinkingBlockDataclass:

    def test_fields(self):
        tb = ThinkingBlock(thinking="hello", signature="abc")
        assert tb.thinking == "hello"
        assert tb.signature == "abc"


# ---------------------------------------------------------------------------
# Thinking block round-tripping in Anthropic messages
# ---------------------------------------------------------------------------


class TestThinkingRoundTripping:
    """Thinking blocks in history must be sent back to the API."""

    def test_assistant_with_thinking_blocks_stripped(self):
        """Thinking blocks are stripped before conversion (JSONL can't guarantee fidelity)."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "4",
                "thinking": [
                    {"thinking": "Simple arithmetic", "signature": "sigABC"},
                ],
            },
            {"role": "user", "content": "And 3+3?"},
        ]
        result = _convert_messages_for_anthropic(messages)

        # Thinking stripped — assistant message should be plain string content
        assistant = result[1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "4"

    def test_empty_signature_stripped_with_rest(self):
        """Thinking blocks with empty signatures are stripped like all others."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi",
                "thinking": [
                    {"thinking": "Some aborted thought", "signature": ""},
                ],
            },
        ]
        result = _convert_messages_for_anthropic(messages)
        assistant = result[1]
        assert assistant["content"] == "Hi"

    def test_none_signature_stripped_with_rest(self):
        """Thinking blocks with None signatures are stripped like all others."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi",
                "thinking": [
                    {"thinking": "Incomplete thought", "signature": None},
                ],
            },
        ]
        result = _convert_messages_for_anthropic(messages)
        assistant = result[1]
        assert assistant["content"] == "Hi"

    def test_no_thinking_key_unchanged(self):
        """Assistant messages without thinking key work as before."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = _convert_messages_for_anthropic(messages)
        assert result[1] == {"role": "assistant", "content": "Hello"}

    def test_thinking_stripped_tool_calls_preserved(self):
        """Thinking stripped but tool_calls still convert to tool_use blocks."""
        from openalph.provider import ToolCall
        messages = [
            {"role": "user", "content": "Read my file"},
            {
                "role": "assistant",
                "content": "",
                "thinking": [
                    {"thinking": "I should read the file", "signature": "sig1"},
                ],
                "tool_calls": [
                    ToolCall(id="tc1", name="file_read", input={"path": "/tmp/test"}),
                ],
            },
        ]
        result = _convert_messages_for_anthropic(messages)
        assistant = result[1]
        assert isinstance(assistant["content"], list)

        types = [b["type"] for b in assistant["content"]]
        assert "thinking" not in types
        assert "tool_use" in types

# Anthropic API call: adaptive thinking
# ---------------------------------------------------------------------------


class TestAdaptiveThinkingAPICall:
    """Verify correct API parameters for adaptive thinking (Opus/Sonnet 4-6)."""

    @pytest.mark.asyncio
    async def test_adaptive_thinking_high(self):
        config = make_config(
            default_model="anthropic/claude-opus-4-6-20250605",
            thinking="high",
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            final_msg = mock_anthropic_response_with_thinking()
            client.messages.stream = MagicMock(
                return_value=MockAnthropicStream(
                    [_make_text_event("Hello"), _make_message_stop()],
                    final_message=final_msg
                )
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.messages.stream.call_args.kwargs
        assert kw["thinking"] == {"type": "adaptive"}
        assert kw["output_config"] == {"effort": "high"}
        # max_tokens should still be set
        assert "max_tokens" in kw

    @pytest.mark.asyncio
    async def test_adaptive_thinking_low(self):
        config = make_config(
            default_model="anthropic/claude-sonnet-4-6-20250514",
            thinking="low",
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            final_msg = mock_anthropic_response_text_only()
            client.messages.stream = MagicMock(
                return_value=MockAnthropicStream(
                    [_make_text_event("Hello"), _make_message_stop()],
                    final_message=final_msg
                )
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.messages.stream.call_args.kwargs
        assert kw["thinking"] == {"type": "adaptive"}
        assert kw["output_config"] == {"effort": "low"}

    @pytest.mark.asyncio
    async def test_thinking_off_no_thinking_param(self):
        config = make_config(
            default_model="anthropic/claude-opus-4-6-20250605",
            thinking="off",
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            final_msg = mock_anthropic_response_text_only()
            client.messages.stream = MagicMock(
                return_value=MockAnthropicStream(
                    [_make_text_event("Hello"), _make_message_stop()],
                    final_message=final_msg
                )
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.messages.stream.call_args.kwargs
        assert "thinking" not in kw
        assert "output_config" not in kw


# ---------------------------------------------------------------------------
# Anthropic API call: budget-based thinking
# ---------------------------------------------------------------------------


class TestBudgetThinkingAPICall:
    """Verify correct API parameters for budget-based thinking (older models)."""

    @pytest.mark.asyncio
    async def test_budget_thinking_high(self):
        config = make_config(
            default_model="anthropic/claude-3-5-sonnet-20241022",
            thinking="high",
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            final_msg = mock_anthropic_response_with_thinking()
            client.messages.stream = MagicMock(
                return_value=MockAnthropicStream(
                    [_make_text_event("Hello"), _make_message_stop()],
                    final_message=final_msg
                )
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.messages.stream.call_args.kwargs
        assert kw["thinking"] == {"type": "enabled", "budget_tokens": 16384}
        # max_tokens adjusted (base + budget) then capped for non-streaming
        assert kw["max_tokens"] == 24576  # base + budget, no longer capped

    @pytest.mark.asyncio
    async def test_budget_thinking_low(self):
        config = make_config(
            default_model="anthropic/claude-3-5-sonnet-20241022",
            thinking="low",
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            final_msg = mock_anthropic_response_text_only()
            client.messages.stream = MagicMock(
                return_value=MockAnthropicStream(
                    [_make_text_event("Hello"), _make_message_stop()],
                    final_message=final_msg
                )
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.messages.stream.call_args.kwargs
        assert kw["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert kw["max_tokens"] == 8192 + 2048


# ---------------------------------------------------------------------------
# Prompt caching
# ---------------------------------------------------------------------------


class TestPromptCaching:
    """cache_control markers on system prompt and last user message."""

    @pytest.mark.asyncio
    async def test_system_prompt_has_cache_control(self):
        config = make_config(
            default_model="anthropic/claude-opus-4-6-20250605",
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            final_msg = mock_anthropic_response_text_only()
            client.messages.stream = MagicMock(
                return_value=MockAnthropicStream(
                    [_make_text_event("Hello"), _make_message_stop()],
                    final_message=final_msg
                )
            )

            await complete(
                config=config,
                system="You are a test agent.",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.messages.stream.call_args.kwargs
        # System should be a list of content blocks with cache_control
        system = kw["system"]
        assert isinstance(system, list)
        assert len(system) == 1
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "You are a test agent."
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_last_user_message_has_cache_control(self):
        config = make_config(
            default_model="anthropic/claude-opus-4-6-20250605",
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            final_msg = mock_anthropic_response_text_only()
            client.messages.stream = MagicMock(
                return_value=MockAnthropicStream(
                    [_make_text_event("Hello"), _make_message_stop()],
                    final_message=final_msg
                )
            )

            await complete(
                config=config,
                system="Test",
                messages=[
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "Reply"},
                    {"role": "user", "content": "Second message"},
                ],
            )

        kw = client.messages.stream.call_args.kwargs
        messages = kw["messages"]
        last_user = messages[-1]
        # Last user message should have cache_control on its content
        assert isinstance(last_user["content"], list)
        last_block = last_user["content"][-1]
        assert last_block["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_openai_no_cache_control(self):
        """OpenAI/OpenRouter path should NOT add cache_control."""
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="https://openrouter.ai/api/v1",
                )
            },
            default_model="openrouter/kimi-k2.5",
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Hi"
            choice.message.tool_calls = None
            choice.finish_reason = "stop"
            resp.choices = [choice]
            resp.model = "kimi-k2.5"
            resp.usage.prompt_tokens = 100
            resp.usage.completion_tokens = 50
            usage = MagicMock()
            usage.prompt_tokens = 100
            usage.completion_tokens = 50
            chunk = _make_openai_chunk(content="Hello", finish_reason="stop", usage=usage)
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream([chunk])
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.chat.completions.create.call_args.kwargs
        # System should be a plain string in the first message, no cache_control
        system_msg = kw["messages"][0]
        assert system_msg["role"] == "system"
        assert isinstance(system_msg["content"], str)


# ---------------------------------------------------------------------------
# OpenRouter reasoning
# ---------------------------------------------------------------------------


class TestOpenRouterReasoning:
    """OpenRouter reasoning.effort via extra_body."""

    @pytest.mark.asyncio
    async def test_reasoning_effort_injected(self):
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="https://openrouter.ai/api/v1",
                )
            },
            default_model="openrouter/kimi-k2.5",
            thinking="high",
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Hello"
            choice.message.tool_calls = None
            choice.finish_reason = "stop"
            resp.choices = [choice]
            resp.model = "kimi-k2.5"
            resp.usage.prompt_tokens = 100
            resp.usage.completion_tokens = 50
            usage = MagicMock()
            usage.prompt_tokens = 100
            usage.completion_tokens = 50
            chunk = _make_openai_chunk(content="Hello", finish_reason="stop", usage=usage)
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream([chunk])
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.chat.completions.create.call_args.kwargs
        assert kw["extra_body"] == {"reasoning": {"effort": "high"}}

    @pytest.mark.asyncio
    async def test_reasoning_off_no_extra_body(self):
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="https://openrouter.ai/api/v1",
                )
            },
            default_model="openrouter/kimi-k2.5",
            thinking="off",
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Hello"
            choice.message.tool_calls = None
            choice.finish_reason = "stop"
            resp.choices = [choice]
            resp.model = "kimi-k2.5"
            resp.usage.prompt_tokens = 100
            resp.usage.completion_tokens = 50
            usage = MagicMock()
            usage.prompt_tokens = 100
            usage.completion_tokens = 50
            chunk = _make_openai_chunk(content="Hello", finish_reason="stop", usage=usage)
            client.chat.completions.create = MagicMock(
                return_value=MockOpenAIStream([chunk])
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kw


# ---------------------------------------------------------------------------
# complete() thinking parameter override
# ---------------------------------------------------------------------------


class TestThinkingOverride:
    """The thinking parameter to complete() overrides config."""

    @pytest.mark.asyncio
    async def test_override_to_high(self):
        config = make_config(
            default_model="anthropic/claude-opus-4-6-20250605",
            thinking="off",
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            final_msg = mock_anthropic_response_with_thinking()
            client.messages.stream = MagicMock(
                return_value=MockAnthropicStream(
                    [_make_text_event("Hello"), _make_message_stop()],
                    final_message=final_msg
                )
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                thinking="high",
            )

        kw = client.messages.stream.call_args.kwargs
        assert kw["thinking"] == {"type": "adaptive"}
        assert kw["output_config"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_override_to_off(self):
        config = make_config(
            default_model="anthropic/claude-opus-4-6-20250605",
            thinking="high",
        )

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            final_msg = mock_anthropic_response_text_only()
            client.messages.stream = MagicMock(
                return_value=MockAnthropicStream(
                    [_make_text_event("Hello"), _make_message_stop()],
                    final_message=final_msg
                )
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                thinking="off",
            )

        kw = client.messages.stream.call_args.kwargs
        assert "thinking" not in kw


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigThinking:
    """Config loading with thinking field."""

    def test_thinking_default_off(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.thinking == "off"

    def test_thinking_high(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"
thinking = "high"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.thinking == "high"

    def test_thinking_invalid_value(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"
thinking = "maximum"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        from openalph.config import ConfigError
        with pytest.raises(ConfigError, match="thinking"):
            load_config(tmp_path / "agent.toml")

    def test_thinking_all_valid_values(self, tmp_path):
        for val in ("off", "low", "medium", "high"):
            (tmp_path / "agent.toml").write_text(f"""
[agent]
name = "test"
default_model = "anthropic/test-model"
thinking = "{val}"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
            config = load_config(tmp_path / "agent.toml")
            assert config.thinking == val


# ---------------------------------------------------------------------------
# Session persistence with thinking blocks
# ---------------------------------------------------------------------------


class TestSessionThinkingPersistence:
    """Thinking blocks survive JSONL round-trip."""

    def test_assistant_with_thinking_serializes(self, tmp_path):
        from openalph.session import SessionLog
        log = SessionLog(workspace=tmp_path, agent_user_id="@test:matrix.local")
        room_id = "!testroom:matrix.local"

        # Log an assistant message with thinking (uses generic append API)
        log.append(
            role="assistant",
            sender="@test:matrix.local",
            room=room_id,
            event_id=None,
            content="The answer is 42.",
            thinking=[{"thinking": "Deep thought...", "signature": "sig123"}],
        )

        # Read it back
        entries = log.read(room_id)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["role"] == "assistant"
        assert entry["content"] == "The answer is 42."
        assert entry["thinking"] == [{"thinking": "Deep thought...", "signature": "sig123"}]

    def test_build_context_includes_thinking(self, tmp_path):
        from openalph.session import SessionLog
        log = SessionLog(workspace=tmp_path, agent_user_id="@test:matrix.local")
        room_id = "!testroom:matrix.local"

        log.append(
            role="user",
            sender="@sb:matrix.local",
            room=room_id,
            event_id="$evt1",
            content="What is the meaning of life?",
        )
        log.append(
            role="assistant",
            sender="@test:matrix.local",
            room=room_id,
            event_id=None,
            content="42",
            thinking=[{"thinking": "Let me consider...", "signature": "sigABC"}],
        )
        log.append(
            role="user",
            sender="@sb:matrix.local",
            room=room_id,
            event_id="$evt2",
            content="Why?",
        )

        context = log.build_context(room_id)
        # The assistant message in context should have thinking key
        assistant_msg = [m for m in context if m["role"] == "assistant"][0]
        assert "thinking" in assistant_msg
        assert assistant_msg["thinking"][0]["thinking"] == "Let me consider..."
        assert assistant_msg["thinking"][0]["signature"] == "sigABC"

    def test_build_context_without_thinking(self, tmp_path):
        """Messages without thinking still work."""
        from openalph.session import SessionLog
        log = SessionLog(workspace=tmp_path, agent_user_id="@test:matrix.local")
        room_id = "!testroom:matrix.local"

        log.append(
            role="user",
            sender="@sb:matrix.local",
            room=room_id,
            event_id="$evt1",
            content="Hello",
        )
        log.append(
            role="assistant",
            sender="@test:matrix.local",
            room=room_id,
            event_id=None,
            content="Hi there",
        )

        context = log.build_context(room_id)
        assistant_msg = [m for m in context if m["role"] == "assistant"][0]
        assert assistant_msg.get("thinking") is None or assistant_msg.get("thinking") == []


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


class TestResponseThinkingField:

    def test_response_default_thinking_empty(self):
        r = Response(content="Hello")
        assert r.thinking == []

    def test_response_with_thinking(self):
        r = Response(
            content="Hello",
            thinking=[ThinkingBlock(thinking="hmm", signature="sig1")],
        )
        assert len(r.thinking) == 1
        assert r.thinking[0].thinking == "hmm"
