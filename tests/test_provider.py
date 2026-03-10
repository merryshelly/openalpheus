"""Tests for the provider adapter.

Interface contract:
    complete(config, system, messages, max_tokens) -> Response

Routes to Anthropic SDK or OpenAI SDK based on config.provider.
Normalizes response format across both providers.

Response: content (str), model (str), usage (Usage), stop_reason (str)
Usage: input_tokens, output_tokens, cache_read_tokens (optional), cache_creation_tokens (optional)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from openalph.config import AgentConfig
from openalph.provider import complete, Response, Usage, _convert_messages_for_anthropic


def make_config(provider="anthropic", **kwargs):
    defaults = dict(
        name="test",
        model="test-model",
        max_tokens=8192,
        api_key="sk-test",
        base_url=None,
        workspace=Path("/tmp/test"),
    )
    defaults.update(kwargs)
    defaults["provider"] = provider
    if provider == "openai" and "base_url" not in kwargs:
        defaults["base_url"] = "http://localhost:11434/v1"
    return AgentConfig(**defaults)


def mock_anthropic_response(text="Hello", model="test-model",
                             input_tokens=100, output_tokens=50,
                             cache_read=0, cache_create=0):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.model = model
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    resp.usage.cache_read_input_tokens = cache_read
    resp.usage.cache_creation_input_tokens = cache_create
    resp.stop_reason = "end_turn"
    return resp


def mock_openai_response(text="Hello", model="test-model",
                          prompt_tokens=100, completion_tokens=50):
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = model
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


# --- Routing ---


class TestRouting:

    @pytest.mark.asyncio
    async def test_anthropic_uses_anthropic_sdk(self):
        config = make_config("anthropic", model="claude-sonnet-4-20250514")

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(
                return_value=mock_anthropic_response("Hello from Claude")
            )

            response = await complete(
                config=config,
                system="You are a test agent.",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert response.content == "Hello from Claude"
        client.messages.create.assert_awaited_once()

        # Verify Anthropic SDK call shape
        kw = client.messages.create.call_args.kwargs
        assert kw["model"] == "claude-sonnet-4-20250514"
        assert kw["system"] == "You are a test agent."
        assert kw["messages"] == [{"role": "user", "content": "Hi"}]
        assert kw["max_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_openai_uses_openai_sdk(self):
        config = make_config(
            "openai",
            model="moonshotai/kimi-k2.5",
            base_url="https://openrouter.ai/api/v1",
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response("Hello from Kimi")
            )

            response = await complete(
                config=config,
                system="You are a test agent.",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert response.content == "Hello from Kimi"

        # Verify client constructed with base_url
        MockClient.assert_called_once_with(
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
        )

        # Verify system message prepended to messages
        kw = client.chat.completions.create.call_args.kwargs
        assert kw["messages"][0] == {"role": "system", "content": "You are a test agent."}
        assert kw["messages"][1] == {"role": "user", "content": "Hi"}

    @pytest.mark.asyncio
    async def test_ollama_uses_openai_path(self):
        config = make_config(
            "openai",
            model="qwen3:235b-a22b",
            base_url="http://100.90.3.4:11434/v1",
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response("Hello from Qwen")
            )

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        MockClient.assert_called_once_with(
            api_key="sk-test",
            base_url="http://100.90.3.4:11434/v1",
        )
        assert response.content == "Hello from Qwen"


# --- Response normalization ---


class TestResponseNormalization:

    @pytest.mark.asyncio
    async def test_anthropic_response_fields(self):
        config = make_config("anthropic")

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(
                return_value=mock_anthropic_response(
                    text="Test",
                    model="claude-sonnet-4-20250514",
                    input_tokens=150,
                    output_tokens=75,
                    cache_read=120,
                    cache_create=30,
                )
            )

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert isinstance(response, Response)
        assert response.content == "Test"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.usage.input_tokens == 150
        assert response.usage.output_tokens == 75
        assert response.usage.cache_read_tokens == 120
        assert response.usage.cache_creation_tokens == 30
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_openai_response_fields(self):
        config = make_config("openai", base_url="http://localhost/v1")

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response(
                    text="Test",
                    model="kimi-k2.5",
                    prompt_tokens=200,
                    completion_tokens=60,
                )
            )

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert isinstance(response, Response)
        assert response.content == "Test"
        assert response.model == "kimi-k2.5"
        assert response.usage.input_tokens == 200
        assert response.usage.output_tokens == 60
        assert response.usage.cache_read_tokens is None
        assert response.usage.cache_creation_tokens is None
        assert response.stop_reason == "stop"


# --- Multi-turn ---


class TestMultiTurn:

    @pytest.mark.asyncio
    async def test_full_history_passed(self):
        config = make_config("anthropic")
        messages = [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(
                return_value=mock_anthropic_response("6")
            )

            await complete(config=config, system="Math tutor", messages=messages)

        kw = client.messages.create.call_args.kwargs
        assert len(kw["messages"]) == 3
        assert kw["messages"][2]["content"] == "And 3+3?"

    @pytest.mark.asyncio
    async def test_openai_history_with_system_prepended(self):
        """OpenAI path prepends system, then passes all history messages."""
        config = make_config("openai", base_url="http://localhost/v1")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response("Good!")
            )

            await complete(config=config, system="Be friendly", messages=messages)

        kw = client.chat.completions.create.call_args.kwargs
        assert len(kw["messages"]) == 4  # system + 3 history
        assert kw["messages"][0]["role"] == "system"
        assert kw["messages"][1]["content"] == "Hello"


# --- max_tokens ---


class TestMaxTokens:

    @pytest.mark.asyncio
    async def test_anthropic_max_tokens_from_config(self):
        config = make_config("anthropic", max_tokens=4096)

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(
                return_value=mock_anthropic_response()
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.messages.create.call_args.kwargs
        assert kw["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_openai_max_tokens_from_config(self):
        config = make_config("openai", max_tokens=2048, base_url="http://localhost/v1")

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response()
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.chat.completions.create.call_args.kwargs
        assert kw["max_tokens"] == 2048


# --- Error propagation ---


class TestErrors:

    @pytest.mark.asyncio
    async def test_anthropic_error_propagates(self):
        config = make_config("anthropic")

        with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
            client = MockClient.return_value
            client.messages.create = AsyncMock(
                side_effect=Exception("rate limit exceeded")
            )

            with pytest.raises(Exception, match="rate limit"):
                await complete(
                    config=config,
                    system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                )

    @pytest.mark.asyncio
    async def test_openai_error_propagates(self):
        config = make_config("openai", base_url="http://localhost/v1")

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                side_effect=Exception("connection refused")
            )

            with pytest.raises(Exception, match="connection refused"):
                await complete(
                    config=config,
                    system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                )


# --- Dataclasses ---


class TestDataclasses:

    def test_response(self):
        r = Response(
            content="Hello",
            model="test",
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
        )
        assert r.content == "Hello"
        assert r.usage.input_tokens == 10

    def test_usage_cache_defaults_none(self):
        u = Usage(input_tokens=10, output_tokens=5)
        assert u.cache_read_tokens is None
        assert u.cache_creation_tokens is None

    def test_usage_with_cache(self):
        u = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=80,
            cache_creation_tokens=20,
        )
        assert u.cache_read_tokens == 80
        assert u.cache_creation_tokens == 20


class TestConvertMessagesForAnthropic:
    """Tests for _convert_messages_for_anthropic tool result merging."""

    def test_single_tool_result(self):
        """Single tool result converts to user message with tool_result block."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "tool_call_id": "tc1", "content": "result1"},
        ]
        result = _convert_messages_for_anthropic(messages)
        assert result[-1] == {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tc1", "content": "result1"}],
        }

    def test_consecutive_tool_results_merged(self):
        """Multiple consecutive tool results merge into one user message."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "calling tools"},
            {"role": "tool", "tool_call_id": "tc1", "content": "result1"},
            {"role": "tool", "tool_call_id": "tc2", "content": "result2"},
            {"role": "tool", "tool_call_id": "tc3", "content": "result3"},
        ]
        result = _convert_messages_for_anthropic(messages)
        # Should be: user, assistant, user (merged 3 tool results)
        assert len(result) == 3
        merged_user = result[2]
        assert merged_user["role"] == "user"
        assert len(merged_user["content"]) == 3
        assert merged_user["content"][0]["tool_use_id"] == "tc1"
        assert merged_user["content"][1]["tool_use_id"] == "tc2"
        assert merged_user["content"][2]["tool_use_id"] == "tc3"

    def test_non_consecutive_tool_results_not_merged(self):
        """Tool results separated by assistant message stay separate."""
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": "result1"},
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "tool_call_id": "tc2", "content": "result2"},
        ]
        result = _convert_messages_for_anthropic(messages)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_plain_text_user_not_merged_with_tool_result(self):
        """Plain text user messages adjacent to tool results are NOT merged."""
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": "result1"},
            {"role": "user", "content": "follow up question"},
        ]
        result = _convert_messages_for_anthropic(messages)
        # tool result has list content, plain user has string content -> no merge
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "follow up question"
