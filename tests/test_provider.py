"""Tests for the provider adapter.

Interface contract:
    complete(config, system, messages, max_tokens, model) -> Response

Routes to Anthropic SDK or OpenAI SDK based on config.providers.
Normalizes response format across both providers.

Response: content (str), model (str), usage (Usage), stop_reason (str)
Usage: input_tokens, output_tokens, cache_read_tokens (optional), cache_creation_tokens (optional)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import complete, stream, Response, Usage, ThinkingBlock, StreamEvent, _convert_messages_for_anthropic, _build_openai_kwargs


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
        "name": "test-agent",
        "default_model": "anthropic/claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "providers": {"anthropic": make_provider(key="anthropic")},
        "workspace": Path("/tmp/test"),
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


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


def _anthropic_message_stop():
    e = MagicMock()
    e.type = "message_stop"
    return e


def _anthropic_final_message(text="", model="test-model",
                             input_tokens=100, output_tokens=50,
                             cache_read=0, cache_create=0,
                             stop_reason="end_turn"):
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
    msg.content = content_blocks
    return msg


def mock_openai_stream_chunks(text="Hello", model="test-model",
                               prompt_tokens=100, completion_tokens=50):
    """Create mock OpenAI streaming chunks."""
    chunks = []
    
    # Text chunk
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = text
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].finish_reason = "stop"
    chunk.usage = None
    chunks.append(chunk)
    
    # Usage chunk
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = MagicMock()
    chunk.usage.prompt_tokens = prompt_tokens
    chunk.usage.completion_tokens = completion_tokens
    chunks.append(chunk)
    
    return chunks


class MockOpenAIStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for chunk in self._chunks:
            yield chunk


# --- Routing ---


class TestRouting:

    @pytest.mark.asyncio
    async def test_anthropic_uses_anthropic_sdk(self):
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")},
            default_model="anthropic/claude-sonnet-4-20250514"
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            
            final_msg = _anthropic_final_message("Hello from Claude")
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_text("Hello from Claude"), _anthropic_message_stop()],
                final_msg,
            )

            response = await complete(
                config=config,
                system="You are a test agent.",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert response.content == "Hello from Claude"
        client.messages.stream.assert_called_once()

        # Verify Anthropic SDK call shape
        kw = client.messages.stream.call_args.kwargs
        assert kw["model"] == "claude-sonnet-4-20250514"
        assert kw["system"] == [{"type": "text", "text": "You are a test agent.", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
        assert kw["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]
        assert kw["max_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_openai_uses_openai_sdk(self):
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="https://openrouter.ai/api/v1"
                )
            },
            default_model="openrouter/moonshotai/kimi-k2.5"
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(mock_openai_stream_chunks("Hello from Kimi"))
            )

            response = await complete(
                config=config,
                system="You are a test agent.",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert response.content == "Hello from Kimi"

        # Verify client constructed with base_url and timeout
        MockClient.assert_called_once()
        call_kwargs = MockClient.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-test"
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert "timeout" in call_kwargs

        # Verify system message prepended to messages
        kw = client.chat.completions.create.call_args.kwargs
        assert kw["messages"][0] == {"role": "system", "content": "You are a test agent."}
        assert kw["messages"][1] == {"role": "user", "content": "Hi"}

    @pytest.mark.asyncio
    async def test_ollama_uses_openai_path(self):
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="http://100.90.3.4:11434/v1"
                )
            },
            default_model="openrouter/qwen3:235b-a22b"
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(mock_openai_stream_chunks("Hello from Qwen"))
            )

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        MockClient.assert_called_once()
        call_kwargs = MockClient.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-test"
        assert call_kwargs["base_url"] == "http://100.90.3.4:11434/v1"
        assert "timeout" in call_kwargs
        assert response.content == "Hello from Qwen"


# --- Response normalization ---


class TestResponseNormalization:

    @pytest.mark.asyncio
    async def test_anthropic_response_fields(self):
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")}
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            
            final_msg = _anthropic_final_message(
                text="Test",
                model="claude-sonnet-4-20250514",
                input_tokens=150,
                output_tokens=75,
                cache_read=120,
                cache_create=30,
            )
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_text("Test"), _anthropic_message_stop()],
                final_msg,
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
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            },
            default_model="openrouter/test-model"
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(mock_openai_stream_chunks(
                    text="Test",
                    model="test-model",
                    prompt_tokens=200,
                    completion_tokens=60,
                ))
            )

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert isinstance(response, Response)
        assert response.content == "Test"
        # In streaming mode, model comes from api_model (config default_model)
        assert response.model == "test-model"
        assert response.usage.input_tokens == 200
        assert response.usage.output_tokens == 60
        assert response.usage.cache_read_tokens == 0
        assert response.usage.cache_creation_tokens == 0
        assert response.stop_reason == "stop"


# --- Multi-turn ---


class TestMultiTurn:

    @pytest.mark.asyncio
    async def test_full_history_passed(self):
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")}
        )
        messages = [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            
            final_msg = _anthropic_final_message("6")
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_text("6"), _anthropic_message_stop()],
                final_msg,
            )

            await complete(config=config, system="Math tutor", messages=messages)

        kw = client.messages.stream.call_args.kwargs
        assert len(kw["messages"]) == 3
        assert kw["messages"][2]["content"] == [{"type": "text", "text": "And 3+3?", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]

    @pytest.mark.asyncio
    async def test_openai_history_with_system_prepended(self):
        """OpenAI path prepends system, then passes all history messages."""
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            },
            default_model="openrouter/test-model"
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(mock_openai_stream_chunks("Good!"))
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
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")},
            max_tokens=4096
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            
            final_msg = _anthropic_final_message()
            client.messages.stream.return_value = MockAnthropicStream(
                [_anthropic_message_stop()],
                final_msg,
            )

            await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        kw = client.messages.stream.call_args.kwargs
        assert kw["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_openai_max_tokens_from_config(self):
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            },
            max_tokens=2048,
            default_model="openrouter/test-model",
        )

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(mock_openai_stream_chunks())
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
        config = make_config(
            providers={"anthropic": make_provider(key="anthropic", type="anthropic", api_key="sk-test")}
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            
            # Create an error stream
            class ErrorStream:
                async def __aenter__(self):
                    raise Exception("rate limit exceeded")
                async def __aexit__(self, *args):
                    pass
            
            client.messages.stream.return_value = ErrorStream()

            with pytest.raises(Exception, match="rate limit"):
                await complete(
                    config=config,
                    system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                )

    @pytest.mark.asyncio
    async def test_openai_error_propagates(self):
        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            },
            default_model="openrouter/test-model"
        )

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


def mock_openai_reasoning_stream_chunks(reasoning_text='I think...', content_text='Hello',
                                         model='test-model', prompt_tokens=100, completion_tokens=50):
    """Create mock OpenAI streaming chunks with reasoning (OpenRouter format)."""
    chunks = []

    # Reasoning chunk
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock(spec=[])
    chunk.choices[0].delta.content = ''
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].delta.reasoning = reasoning_text
    chunk.choices[0].finish_reason = None
    chunk.usage = None
    chunks.append(chunk)

    # Text chunk
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock(spec=[])
    chunk.choices[0].delta.content = content_text
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].delta.reasoning = None
    chunk.choices[0].finish_reason = 'stop'
    chunk.usage = None
    chunks.append(chunk)

    # Usage chunk
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = MagicMock()
    chunk.usage.prompt_tokens = prompt_tokens
    chunk.usage.completion_tokens = completion_tokens
    chunks.append(chunk)

    return chunks


def mock_openai_complete_with_reasoning(content='Hello', reasoning='I think...'):
    """Create a mock non-streaming OpenAI response with reasoning."""
    response = MagicMock()
    response.model = 'test-model'
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock(spec=[])
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = None
    response.choices[0].message.reasoning = reasoning
    response.choices[0].finish_reason = 'stop'
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    return response


class TestOpenAIReasoning:

    def make_config_openai(self):
        return make_config(
            providers={'openai': make_provider(key='openai', type='openai', api_key='sk-test')},
            default_model='openai/gpt-4o',
        )

    @pytest.mark.asyncio
    async def test_streaming_reasoning_yields_thinking_events(self):
        config = self.make_config_openai()
        with patch('openalph.provider._get_client') as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(
                    mock_openai_reasoning_stream_chunks(reasoning_text='Deep thoughts', content_text='Answer')
                )
            )
            events = []
            async for event in stream(config=config, system='You are helpful.', messages=[{'role': 'user', 'content': 'Hello'}]):
                events.append(event)

        thinking_events = [e for e in events if e.type == 'thinking']
        done_events = [e for e in events if e.type == 'done']

        assert len(thinking_events) >= 1
        assert thinking_events[0].content == 'Deep thoughts'

        assert len(done_events) == 1
        resp = done_events[0].response
        assert len(resp.thinking) == 1
        assert isinstance(resp.thinking[0], ThinkingBlock)
        assert resp.thinking[0].thinking == 'Deep thoughts'
        assert resp.content == 'Answer'

    @pytest.mark.asyncio
    async def test_streaming_no_reasoning_no_thinking(self):
        config = self.make_config_openai()
        with patch('openalph.provider._get_client') as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(mock_openai_stream_chunks('Hello'))
            )
            events = []
            async for event in stream(config=config, system='You are helpful.', messages=[{'role': 'user', 'content': 'Hello'}]):
                events.append(event)

        thinking_events = [e for e in events if e.type == 'thinking']
        done_events = [e for e in events if e.type == 'done']

        assert thinking_events == []
        assert len(done_events) == 1
        assert done_events[0].response.thinking == []

    @pytest.mark.asyncio
    async def test_complete_reasoning_extracted(self):
        """complete() accumulates reasoning from streaming and returns it in response.thinking."""
        config = self.make_config_openai()
        with patch('openalph.provider._get_client') as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(
                    mock_openai_reasoning_stream_chunks(reasoning_text='My reasoning', content_text='Answer')
                )
            )
            response = await complete(config=config, system='You are helpful.', messages=[{'role': 'user', 'content': 'Hello'}])

        assert len(response.thinking) == 1
        assert isinstance(response.thinking[0], ThinkingBlock)
        assert response.thinking[0].thinking == 'My reasoning'
        assert response.content == 'Answer'

    @pytest.mark.asyncio
    async def test_complete_no_reasoning_no_thinking(self):
        """complete() without reasoning returns empty thinking list."""
        config = self.make_config_openai()
        with patch('openalph.provider._get_client') as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(mock_openai_stream_chunks('Hello'))
            )
            response = await complete(config=config, system='You are helpful.', messages=[{'role': 'user', 'content': 'Hello'}])

        assert response.thinking == []


class TestBuildOpenaiKwargs:
    """Tests for _build_openai_kwargs provider capability flags."""

    def _base_args(self, **overrides):
        defaults = dict(
            api_model="test/model",
            system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None,
            max_tokens=1024,
            thinking_level="off",
            quirks=[],
        )
        defaults.update(overrides)
        return defaults

    # --- Routing ---

    def test_routing_adds_provider_to_extra_body(self):
        """routing dict appears as extra_body.provider."""
        routing = {"quantizations": ["fp8", "fp16"]}
        kw = _build_openai_kwargs(**self._base_args(routing=routing))
        assert kw["extra_body"]["provider"] == routing

    def test_no_routing_no_extra_body(self):
        """Without routing or thinking, extra_body is absent."""
        kw = _build_openai_kwargs(**self._base_args())
        assert "extra_body" not in kw

    def test_routing_and_thinking_both_in_extra_body(self):
        """Both routing and thinking produce their keys in extra_body (OpenRouter)."""
        routing = {"quantizations": ["fp8"]}
        kw = _build_openai_kwargs(**self._base_args(
            thinking_level="high", routing=routing, provider_key="openrouter",
        ))
        assert kw["extra_body"]["reasoning"] == {"effort": "high"}
        assert kw["extra_body"]["provider"] == routing

    # --- max_tokens vs max_completion_tokens ---

    def test_openai_uses_max_completion_tokens(self):
        """Direct OpenAI provider sends max_completion_tokens, not max_tokens."""
        kw = _build_openai_kwargs(**self._base_args(provider_key="openai"))
        assert "max_completion_tokens" in kw
        assert kw["max_completion_tokens"] == 1024
        assert "max_tokens" not in kw

    def test_google_uses_max_tokens(self):
        """Direct Google provider sends max_tokens (not max_completion_tokens)."""
        kw = _build_openai_kwargs(**self._base_args(provider_key="google"))
        assert "max_tokens" in kw
        assert kw["max_tokens"] == 1024
        assert "max_completion_tokens" not in kw

    def test_openrouter_uses_max_tokens(self):
        """OpenRouter proxy sends max_tokens."""
        kw = _build_openai_kwargs(**self._base_args(provider_key="openrouter"))
        assert "max_tokens" in kw
        assert "max_completion_tokens" not in kw

    def test_default_provider_uses_max_tokens(self):
        """Unknown/empty provider_key defaults to max_tokens."""
        kw = _build_openai_kwargs(**self._base_args())
        assert "max_tokens" in kw
        assert "max_completion_tokens" not in kw

    # --- frequency_penalty ---

    def test_google_omits_frequency_penalty(self):
        """Direct Google provider does not send frequency_penalty."""
        kw = _build_openai_kwargs(**self._base_args(provider_key="google"))
        assert "frequency_penalty" not in kw

    def test_openai_omits_frequency_penalty_by_default(self):
        """Default sampling profile no longer sends frequency_penalty (kdsn.241.3).
        The old unconditional 0.3 was harmful to long-reasoning models; the
        default profile now omits penalties. See test_sampling_profiles.py."""
        kw = _build_openai_kwargs(**self._base_args(provider_key="openai"))
        assert "frequency_penalty" not in kw

    # --- reasoning extra_body ---

    def test_openrouter_includes_reasoning(self):
        """OpenRouter sends reasoning in extra_body when thinking is on."""
        kw = _build_openai_kwargs(**self._base_args(
            thinking_level="high", provider_key="openrouter",
        ))
        assert kw["extra_body"]["reasoning"] == {"effort": "high"}

    def test_openai_omits_reasoning(self):
        """Direct OpenAI does not send reasoning extra_body."""
        kw = _build_openai_kwargs(**self._base_args(
            thinking_level="high", provider_key="openai",
        ))
        assert "extra_body" not in kw

    def test_google_omits_reasoning(self):
        """Direct Google does not send reasoning extra_body."""
        kw = _build_openai_kwargs(**self._base_args(
            thinking_level="high", provider_key="google",
        ))
        assert "extra_body" not in kw

    def test_openrouter_reasoning_max(self):
        """OpenRouter sends reasoning=max in extra_body."""
        kw = _build_openai_kwargs(**self._base_args(
            thinking_level="max", provider_key="openrouter",
        ))
        assert kw["extra_body"]["reasoning"] == {"effort": "max"}

    def test_openrouter_reasoning_xhigh(self):
        """OpenRouter sends reasoning=xhigh in extra_body."""
        kw = _build_openai_kwargs(**self._base_args(
            thinking_level="xhigh", provider_key="openrouter",
        ))
        assert kw["extra_body"]["reasoning"] == {"effort": "xhigh"}


# --- B2: OpenAI/Fireworks cached-token normalization (RED) ---


class TestOpenAICacheNormalization:
    """Tests for the shared _openai_usage normalizer called at both parse sites.

    These are RED tests for a feature that does not exist yet. The helper
    `_openai_usage` is imported INSIDE each function so a missing symbol fails
    only that one test, not the whole file at collection time.
    """

    # --- 1) Pure helper tests (SimpleNamespace; MagicMock would auto-vivify) ---

    def test_usage_cached_present(self):
        from types import SimpleNamespace
        from openalph.provider import _openai_usage

        u = SimpleNamespace(
            prompt_tokens=200,
            completion_tokens=60,
            prompt_tokens_details=SimpleNamespace(cached_tokens=120),
        )
        usage = _openai_usage(u)
        assert usage.input_tokens == 80
        assert usage.output_tokens == 60
        assert usage.cache_read_tokens == 120
        assert usage.cache_creation_tokens == 0

    def test_usage_clamped(self):
        from types import SimpleNamespace
        from openalph.provider import _openai_usage

        u = SimpleNamespace(
            prompt_tokens=200,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=250),
        )
        usage = _openai_usage(u)
        assert usage.input_tokens == 0
        assert usage.cache_read_tokens == 200
        assert usage.cache_creation_tokens == 0

    def test_usage_details_absent(self):
        from types import SimpleNamespace
        from openalph.provider import _openai_usage

        u = SimpleNamespace(prompt_tokens=200, completion_tokens=60)
        usage = _openai_usage(u)
        assert usage.input_tokens == 200
        assert usage.cache_read_tokens == 0
        assert usage.cache_creation_tokens == 0

    def test_usage_garbage_cached_coerced_to_zero(self):
        # A non-int cached_tokens (malformed provider response, or a MagicMock
        # usage object in a mock-based test) must be treated as 0, never crash on
        # min(non-int, int). Matches the spec's "clamp against a buggy provider"
        # ethos; also keeps every existing MagicMock-usage streaming test green
        # once B2 wires _openai_usage into the parse sites.
        from types import SimpleNamespace
        from openalph.provider import _openai_usage

        u = SimpleNamespace(prompt_tokens=200, completion_tokens=60,
                            prompt_tokens_details=SimpleNamespace(cached_tokens="oops"))
        usage = _openai_usage(u)
        assert usage.input_tokens == 200
        assert usage.cache_read_tokens == 0
        assert usage.cache_creation_tokens == 0

    # --- 2) Non-streaming wiring: _parse_openai_response ---

    def test_parse_openai_normalizes_cache(self):
        from unittest.mock import MagicMock
        from openalph.provider import _parse_openai_response

        resp = MagicMock()
        msg = resp.choices[0].message
        msg.content = "hi"
        msg.tool_calls = None
        msg.reasoning = None
        msg.reasoning_content = None
        resp.model = "accounts/fireworks/models/kimi-k2p6"
        resp.usage.prompt_tokens = 200
        resp.usage.completion_tokens = 60
        resp.usage.prompt_tokens_details.cached_tokens = 120

        r = _parse_openai_response(resp)
        assert r.usage.input_tokens == 80
        assert r.usage.cache_read_tokens == 120
        assert r.usage.cache_creation_tokens == 0
        assert r.usage.output_tokens == 60

    # --- 3) Streaming wiring: via complete() (mirrors test_openai_response_fields) ---

    @pytest.mark.asyncio
    async def test_streaming_normalizes_cache(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from openalph.provider import complete

        config = make_config(
            providers={
                "openrouter": make_provider(
                    key="openrouter", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1"
                )
            },
            default_model="openrouter/test-model"
        )

        text_chunk = MagicMock()
        text_chunk.choices = [MagicMock()]
        text_chunk.choices[0].delta = MagicMock()
        text_chunk.choices[0].delta.content = "Test"
        text_chunk.choices[0].delta.tool_calls = None
        text_chunk.choices[0].finish_reason = "stop"
        text_chunk.usage = None

        usage_chunk = MagicMock()
        usage_chunk.choices = []
        usage_chunk.usage = MagicMock()
        usage_chunk.usage.prompt_tokens = 200
        usage_chunk.usage.completion_tokens = 60
        usage_chunk.usage.prompt_tokens_details.cached_tokens = 120

        with patch("openalph.provider.openai.AsyncOpenAI") as MockClient:
            client = MockClient.return_value
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream([text_chunk, usage_chunk])
            )

            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert response.usage.input_tokens == 80
        assert response.usage.cache_read_tokens == 120
        assert response.usage.cache_creation_tokens == 0
        assert response.usage.output_tokens == 60
