"""
OpenAlph LLM Provider Adapter

Routes completion requests to either Anthropic or OpenAI SDKs based on configuration.
The adapter handles the differences in API shapes and response formats between providers.
"""

import copy
import logging
from dataclasses import dataclass
from typing import AsyncGenerator
import json
import re
import anthropic
import httpx
import openai
from openalph.config import AgentConfig, ProviderConfig, resolve_model

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """User-surfaceable error from an LLM provider.

    Wraps SDK-specific exceptions with a sanitized message
    safe to display in chat.
    """
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


_API_KEY_PATTERN = re.compile(r'\b(sk-[a-zA-Z0-9_-]{10,})\b')

# ---------------------------------------------------------------------------
# Degeneration detection
# ---------------------------------------------------------------------------
# Minimum consecutive identical characters to trigger truncation.
_DEGEN_CHAR_THRESHOLD = 50
_DEGEN_WARNING = (
    "\n\n⚠️ *Output truncated — repetition collapse detected. "
    "Session context may be degraded; consider starting a new session (`/new`).*"
)

# Default frequency_penalty for OpenAI-compatible providers.
# Discourages token repetition at the sampling level.  Moderate value
# that shouldn't affect normal output but raises the cost of degenerate loops.
_DEFAULT_FREQUENCY_PENALTY = 0.3


def _detect_and_truncate_degeneration(text: str) -> tuple[str, bool]:
    """Detect degenerate repetition in model output.

    Checks for runs of identical characters >= _DEGEN_CHAR_THRESHOLD.
    When found, truncates at the start of the degenerate run and appends
    a warning.

    Returns (possibly_truncated_text, was_degenerate).
    """
    if not text or len(text) < _DEGEN_CHAR_THRESHOLD:
        return text, False

    run_start = 0
    run_len = 1

    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            run_len += 1
            if run_len >= _DEGEN_CHAR_THRESHOLD:
                truncated = text[:run_start].rstrip()
                if not truncated:
                    # Entire output is degenerate
                    return _DEGEN_WARNING.lstrip(), True
                return truncated + _DEGEN_WARNING, True
        else:
            run_start = i
            run_len = 1

    return text, False


def _sanitize_error(message: str) -> str:
    """Extract clean error message and strip sensitive data.

    SDK error messages come as 'Error code: NNN - {body_dict}'.
    Extract just the meaningful error text from the body when possible.
    Always strip API key patterns as a safety net.
    """
    # Try to extract the error message from SDK's formatted string
    # Format: "Error code: 400 - {'error': {'message': '...', ...}, ...}"
    import ast
    if " - " in message and message.startswith("Error code:"):
        _, _, body_str = message.partition(" - ")
        try:
            body = ast.literal_eval(body_str.strip())
            if isinstance(body, dict):
                # OpenRouter/OpenAI: {"error": {"message": "..."}}
                err = body.get("error", {})
                if isinstance(err, dict) and "message" in err:
                    message = err["message"]
                # Anthropic: {"error": {"message": "..."}} or {"message": "..."}
                elif "message" in body:
                    message = body["message"]
        except (ValueError, SyntaxError):
            pass  # Keep original message if parsing fails
    return _API_KEY_PATTERN.sub('[REDACTED]', message)

# Client cache: reuse HTTP clients for connection pooling.
# Keyed by (provider, api_key, base_url) so different configs get different clients.
_client_cache: dict[tuple, object] = {}


def _get_client(provider: ProviderConfig):
    """Get or create a cached provider client."""
    timeout = getattr(provider, "timeout", 600.0)
    if provider.type == "anthropic":
        key = ("anthropic", provider.api_key, None, timeout)
        if key not in _client_cache:
            _client_cache[key] = anthropic.AsyncAnthropic(
                api_key=provider.api_key,
                timeout=httpx.Timeout(timeout, connect=10.0),
            )
        return _client_cache[key]
    elif provider.type == "openai":
        key = ("openai", provider.api_key, provider.base_url, timeout)
        if key not in _client_cache:
            _client_cache[key] = openai.AsyncOpenAI(
                api_key=provider.api_key,
                base_url=provider.base_url,
                timeout=httpx.Timeout(timeout, connect=10.0),
            )
        return _client_cache[key]
    else:
        raise ValueError(f"Unsupported provider: {provider.type}")


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict

    def __post_init__(self):
        # Models occasionally emit tool names with leading/trailing whitespace
        self.name = self.name.strip()


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str


@dataclass
class Response:
    content: str
    model: str = ""
    usage: Usage = None
    stop_reason: str = ""
    tool_calls: list[ToolCall] = None
    thinking: list[ThinkingBlock] = None
    degenerate: bool = False
    generation_id: str = ""  # Provider-assigned ID (OpenRouter gen ID, Anthropic msg ID)

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.thinking is None:
            self.thinking = []
        if self.usage is None:
            self.usage = Usage(input_tokens=0, output_tokens=0)


@dataclass
class StreamEvent:
    """A single event from a streaming LLM response."""
    type: str  # "text", "thinking", "signature", "tool_start", "tool_delta", "tool_done", "usage", "done"
    content: str = ""
    tool_index: int = 0
    tool_id: str = ""
    tool_name: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    stop_reason: str = ""
    model: str = ""
    response: Response | None = None


def _supports_adaptive_thinking(model_id: str) -> bool:
    """Returns True for claude-opus-4-6 and claude-sonnet-4-6 variants."""
    return "opus-4-6" in model_id or "sonnet-4-6" in model_id


def _thinking_effort(level: str) -> str:
    """Map config thinking level to Anthropic effort. low→low, medium→medium, high→high."""
    return level  # Direct mapping


def _thinking_budget(level: str, base_max_tokens: int, model_max_tokens: int) -> tuple[int, int]:
    """Budget-based thinking for older models.
    Returns (budget_tokens, adjusted_max_tokens).
    Budget mapping: low=2048, medium=8192, high=16384.
    max_tokens = min(base + budget, model_max_tokens).
    If clamped, reduce budget to leave at least 1024 for output.
    """
    budgets = {"low": 2048, "medium": 8192, "high": 16384}
    budget = budgets.get(level, 16384)
    max_tokens = min(base_max_tokens + budget, model_max_tokens)
    if max_tokens <= budget:
        budget = max(0, max_tokens - 1024)
    return budget, max_tokens

def _convert_tools_for_provider(tools: list | None, provider_type: str) -> list[dict] | None:
    """Convert ToolDef list to provider-native format.
    
    Anthropic: [{"name": ..., "description": ..., "input_schema": ...}]
    OpenAI: [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
    """
    if tools is None:
        return None
    
    result = []
    for tool in tools:
        if provider_type == "anthropic":
            result.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        elif provider_type == "openai":
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
    return result


def _convert_messages_for_provider(messages: list[dict], provider_type: str) -> list[dict]:
    """Convert normalized message history to provider-native format.
    
    Normalized format:
    - {"role": "user", "content": "text"}
    - {"role": "assistant", "content": "text", "tool_calls": [ToolCall(...)]}
    - {"role": "tool", "tool_call_id": "...", "content": "...", "is_error": bool}
    
    Anthropic native:
    - Assistant: {"role": "assistant", "content": [{"type": "text", ...}, {"type": "tool_use", ...}]}
    - Tool result: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": ..., "content": ..., "is_error": ...}]}
    
    OpenAI native:
    - Assistant: {"role": "assistant", "content": "text", "tool_calls": [{"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}]}
    - Tool result: {"role": "tool", "tool_call_id": "...", "content": "..."}
    """
    if provider_type == "anthropic":
        return _convert_messages_for_anthropic(messages)
    elif provider_type == "openai":
        return _convert_messages_for_openai(messages)
    return messages


def _convert_messages_for_anthropic(messages: list[dict]) -> list[dict]:
    """Convert normalized messages to Anthropic format.

    Thinking blocks are stripped from all assistant messages.  Anthropic
    cryptographically signs thinking blocks and rejects any modification,
    but JSONL round-tripping cannot guarantee byte-perfect fidelity.
    Stripping is safe: the thinking already influenced the response and
    replaying it wastes context tokens.
    """
    # Strip thinking from all assistant messages before conversion
    messages = [
        {k: v for k, v in msg.items() if k != "thinking"}
        if msg.get("role") == "assistant" else msg
        for msg in messages
    ]

    result = []
    for msg in messages:
        role = msg.get("role")

        if role == "user":
            # User message may have list content (text + image blocks)
            content = msg.get("content")
            if isinstance(content, list):
                # Convert image blocks to Anthropic wire format
                converted_blocks = []
                for block in content:
                    if block.get("type") == "image":
                        # Convert to Anthropic image source format
                        converted_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.get("media_type", "image/jpeg"),
                                "data": block.get("data", ""),
                            },
                        })
                    else:
                        # Text blocks pass through unchanged
                        converted_blocks.append(block)
                result.append({"role": "user", "content": converted_blocks})
            else:
                # Simple user message - pass through
                result.append(msg)

        elif role == "assistant":
            # Assistant message may have tool_calls and thinking blocks
            tool_calls = msg.get("tool_calls", [])
            content = msg.get("content", "")
            thinking = msg.get("thinking", [])
            
            # If no thinking and no tool_calls, keep original behavior (string content)
            if not thinking and not tool_calls:
                result.append(msg)
                continue
            
            # Build content blocks: thinking blocks + text block + tool_use blocks
            content_blocks = []
            
            # Add thinking blocks first (if any)
            if thinking:
                for tb in thinking:
                    sig = tb.get("signature")
                    if sig:
                        # Valid signature - add as thinking block
                        content_blocks.append({
                            "type": "thinking",
                            "thinking": tb.get("thinking", ""),
                            "signature": sig,
                        })
                    else:
                        # Empty/None signature - demote to text block
                        content_blocks.append({
                            "type": "text",
                            "text": tb.get("thinking", ""),
                        })
            
            # Add text content if non-empty, or if there are no other blocks
            # (Anthropic rejects empty text blocks alongside tool_use)
            if content:
                content_blocks.append({"type": "text", "text": content})
            elif not content_blocks and not tool_calls:
                # No content at all and no tool calls — keep empty text to avoid
                # sending a message with zero content blocks
                content_blocks.append({"type": "text", "text": ""})
            
            # Add tool_use blocks
            if tool_calls:
                for tc in tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
            
            if content_blocks:
                result.append({"role": "assistant", "content": content_blocks})
            else:
                # No content at all - pass through minimal message
                result.append({"role": "assistant", "content": content})
        
        elif role == "tool":
            # Tool result -> user role with tool_result content block
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id"),
                "content": msg.get("content", ""),
            }
            if msg.get("is_error"):
                tool_result_block["is_error"] = True
            result.append({"role": "user", "content": [tool_result_block]})
        
        else:
            # Unknown role - pass through
            result.append(msg)
    
    # Merge consecutive user messages with list content (tool results from parallel calls)
    merged = []
    for msg in result:
        if (merged
                and merged[-1]["role"] == "user"
                and msg["role"] == "user"
                and isinstance(merged[-1].get("content"), list)
                and isinstance(msg.get("content"), list)):
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append(msg)
    return merged


def _convert_messages_for_openai(messages: list[dict]) -> list[dict]:
    """Convert normalized messages to OpenAI format."""
    result = []
    for msg in messages:
        role = msg.get("role")

        if role == "user":
            # User message may have list content (text + image blocks)
            content = msg.get("content")
            if isinstance(content, list):
                # Convert image blocks to OpenAI wire format
                converted_blocks = []
                for block in content:
                    if block.get("type") == "image":
                        # Convert to OpenAI image_url format
                        media_type = block.get("media_type", "image/jpeg")
                        data = block.get("data", "")
                        converted_blocks.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{data}",
                            },
                        })
                    else:
                        # Text blocks pass through unchanged
                        converted_blocks.append(block)
                result.append({"role": "user", "content": converted_blocks})
            else:
                # Simple user message - pass through
                result.append(msg)

        elif role == "assistant":
            # Assistant message may have tool_calls
            tool_calls = msg.get("tool_calls", [])
            content = msg.get("content", "")
            
            if tool_calls:
                # Convert ToolCall objects to OpenAI format
                openai_tool_calls = []
                for tc in tool_calls:
                    openai_tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.input),
                        },
                    })
                result.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": openai_tool_calls,
                })
            else:
                # No tool calls - simple text message
                result.append(msg)
        
        elif role == "tool":
            # Tool result -> OpenAI tool role
            # OpenAI has no native is_error field; prepend marker so the
            # model knows the tool call failed.
            tool_content = msg.get("content", "")
            if msg.get("is_error") and not tool_content.startswith("Error: "):
                tool_content = f"Error: {tool_content}"
            result.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id"),
                "content": tool_content,
            })
        
        else:
            # User and other roles - pass through
            result.append(msg)
    
    return result


def _parse_anthropic_response(response) -> Response:
    """Parse Anthropic response into normalized Response."""
    # Extract text content, tool_calls, and thinking blocks from content blocks
    text_parts = []
    tool_calls = []
    thinking_blocks = []
    
    for block in response.content:
        # Handle both MagicMock (no type attr) and real response blocks
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                input=block.input,
            ))
        elif block_type == "thinking":
            thinking_blocks.append(ThinkingBlock(
                thinking=block.thinking,
                signature=block.signature,
            ))
        else:
            # Backward compatibility: old mocks have .text but no .type="text"
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
    
    content = "\n".join(text_parts) if text_parts else ""
    
    return Response(
        content=content,
        tool_calls=tool_calls,
        thinking=thinking_blocks,
        model=response.model,
        usage=Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_input_tokens,
            cache_creation_tokens=response.usage.cache_creation_input_tokens,
        ),
        stop_reason=response.stop_reason,
        generation_id=getattr(response, "id", "") or "",
    )


def _parse_openai_response(response) -> Response:
    """Parse OpenAI response into normalized Response."""
    message = response.choices[0].message
    content = message.content or ""
    tool_calls = []
    
    if message.tool_calls:
        for tc in message.tool_calls:
            # Parse function arguments JSON
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=arguments,
            ))
    
    # Extract reasoning (OpenRouter extension) into thinking blocks
    thinking_blocks = []
    reasoning = getattr(message, 'reasoning', None)
    if isinstance(reasoning, str) and reasoning:
        thinking_blocks.append(ThinkingBlock(thinking=reasoning, signature=""))

    return Response(
        content=content,
        tool_calls=tool_calls,
        thinking=thinking_blocks,
        model=response.model,
        usage=Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        ),
        stop_reason=response.choices[0].finish_reason,
        generation_id=getattr(response, "id", "") or "",
    )


def _build_anthropic_kwargs(
    api_model: str,
    system: str,
    provider_messages: list[dict],
    provider_tools: list[dict] | None,
    max_tokens: int,
    thinking_level: str,
    model_max_tokens: int = 200000,
    temperature: float | None = None,
    top_p: float | None = None,
    cache_ttl: str | None = None,
) -> dict:
    """Build kwargs for Anthropic messages API."""
    _cc = {"type": "ephemeral", "ttl": cache_ttl} if cache_ttl else {"type": "ephemeral"}
    api_kwargs = {
        "model": api_model,
        "system": [{"type": "text", "text": system, "cache_control": _cc}],
        "messages": provider_messages,
        "max_tokens": max_tokens,
    }
    if provider_tools:
        api_kwargs["tools"] = provider_tools
    
    # Add prompt caching to last user message.
    # Deep copy the target message to avoid mutating the caller's history
    # dicts (shared references from agent.py's shallow list copy).
    if api_kwargs["messages"]:
        last_msg = api_kwargs["messages"][-1]
        if last_msg.get("role") == "user":
            last_msg = copy.deepcopy(last_msg)
            api_kwargs["messages"][-1] = last_msg
            msg_content = last_msg.get("content")
            if isinstance(msg_content, str):
                last_msg["content"] = [{"type": "text", "text": msg_content, "cache_control": _cc}]
            elif isinstance(msg_content, list) and msg_content:
                msg_content[-1]["cache_control"] = _cc
    
    # Add sampling parameters (only when thinking is off — Anthropic disallows with thinking)
    if thinking_level == "off":
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if top_p is not None:
            api_kwargs["top_p"] = top_p

    # Add thinking parameters if enabled
    if thinking_level != "off":
        if _supports_adaptive_thinking(api_model):
            # Adaptive thinking for Opus/Sonnet 4-6
            api_kwargs["thinking"] = {"type": "adaptive"}
            api_kwargs["output_config"] = {"effort": _thinking_effort(thinking_level)}
        else:
            # Budget-based thinking for older models
            budget, adjusted_max = _thinking_budget(thinking_level, max_tokens, model_max_tokens)
            api_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            api_kwargs["max_tokens"] = adjusted_max
    
    return api_kwargs


def _build_openai_kwargs(
    api_model: str,
    system: str,
    provider_messages: list[dict],
    provider_tools: list[dict] | None,
    max_tokens: int,
    thinking_level: str,
    quirks: list[str],
    temperature: float | None = None,
    top_p: float | None = None,
    routing: dict | None = None,
) -> dict:
    """Build kwargs for OpenAI chat completions API."""
    # Handle quirks
    if "no_system_role" in quirks:
        # Fold system into first user message instead of separate system role
        messages_with_system = provider_messages
        if messages_with_system and messages_with_system[0]["role"] == "user":
            first = messages_with_system[0].copy()
            content = first.get("content", "")
            if isinstance(content, str):
                first["content"] = f"{system}\n\n{content}"
            messages_with_system = [first] + messages_with_system[1:]
        else:
            messages_with_system = [{"role": "user", "content": system}] + messages_with_system
    else:
        # Normal: prepend system message to messages list
        messages_with_system = [{"role": "system", "content": system}] + provider_messages
    
    api_kwargs = {
        "model": api_model,
        "messages": messages_with_system,
        "max_tokens": max_tokens,
        "frequency_penalty": _DEFAULT_FREQUENCY_PENALTY,
    }
    if provider_tools:
        api_kwargs["tools"] = provider_tools
    
    # Add sampling parameters
    if temperature is not None:
        api_kwargs["temperature"] = temperature
    if top_p is not None:
        api_kwargs["top_p"] = top_p

    # Build extra_body incrementally
    extra_body = {}
    if thinking_level != "off":
        extra_body["reasoning"] = {"effort": thinking_level}
    if routing:
        extra_body["provider"] = routing
    if extra_body:
        api_kwargs["extra_body"] = extra_body
    
    return api_kwargs


async def stream(
    config: AgentConfig,
    system: str,
    messages: list[dict],
    tools: list | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    thinking: str | None = None,
    cache_ttl: str | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Stream completion events from Anthropic or OpenAI SDK based on config.providers.
    
    Yields StreamEvent objects for each event in the stream.
    Final event is always type="done" with the complete Response.
    """
    # Resolve model string to provider and API model name
    model_str = model or config.default_model
    provider_cfg, api_model = resolve_model(model_str, config.providers)
    
    # Convert messages to provider-native format
    provider_messages = _convert_messages_for_provider(messages, provider_cfg.type)
    
    # Convert tools to provider-native format
    provider_tools = _convert_tools_for_provider(tools, provider_cfg.type)
    
    # Use config.max_tokens if max_tokens is not provided
    tokens = max_tokens if max_tokens is not None else config.max_tokens
    
    # Resolve thinking level: param > config > "off"
    thinking_level = thinking if thinking is not None else getattr(config, "thinking", "off")
    
    if provider_cfg.type == "anthropic":
        client = _get_client(provider_cfg)
        
        api_kwargs = _build_anthropic_kwargs(
            api_model=api_model,
            system=system,
            provider_messages=provider_messages,
            provider_tools=provider_tools,
            max_tokens=tokens,
            thinking_level=thinking_level,
            model_max_tokens=getattr(config, "model_max_tokens", 200000),
            temperature=getattr(config, "temperature", None),
            top_p=getattr(config, "top_p", None),
            cache_ttl=cache_ttl,
        )
        
        try:
            async with client.messages.stream(**api_kwargs) as stream:
                accumulated_text = ""
                
                async for event in stream:
                    event_type = getattr(event, "type", None)
                    
                    if event_type == "text":
                        yield StreamEvent(type="text", content=event.text)
                        accumulated_text += event.text
                    elif event_type == "thinking":
                        yield StreamEvent(type="thinking", content=event.thinking)
                    elif event_type == "signature":
                        yield StreamEvent(type="signature", content=event.signature)
                    elif event_type == "input_json":
                        yield StreamEvent(
                            type="tool_delta",
                            content=event.partial_json,
                        )
                    elif event_type == "content_block_start":
                        block = event.content_block
                        if getattr(block, "type", None) == "tool_use":
                            yield StreamEvent(
                                type="tool_start",
                                tool_index=event.index,
                                tool_id=block.id,
                                tool_name=block.name,
                            )
                    elif event_type == "content_block_stop":
                        block = event.content_block
                        if getattr(block, "type", None) == "tool_use":
                            yield StreamEvent(
                                type="tool_done",
                                tool_index=event.index,
                                tool_call=ToolCall(
                                    id=block.id,
                                    name=block.name,
                                    input=block.input,
                                ),
                            )
                    elif event_type == "message_stop":
                        # Get final message and build response
                        final_message = await stream.get_final_message()
                        response = _parse_anthropic_response(final_message)
                        response.content, response.degenerate = _detect_and_truncate_degeneration(response.content)
                        
                        yield StreamEvent(
                            type="done",
                            stop_reason=final_message.stop_reason,
                            model=final_message.model,
                            response=response,
                        )
                        
        except anthropic.APIStatusError as e:
            raise ProviderError(
                _sanitize_error(e.message), status_code=e.status_code,
            ) from e
        except anthropic.APITimeoutError as e:
            raise ProviderError("Provider request timed out") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError("Provider unreachable — connection failed") from e
    
    elif provider_cfg.type == "openai":
        client = _get_client(provider_cfg)
        
        api_kwargs = _build_openai_kwargs(
            api_model=api_model,
            system=system,
            provider_messages=provider_messages,
            provider_tools=provider_tools,
            max_tokens=tokens,
            thinking_level=thinking_level,
            quirks=provider_cfg.quirks,
            temperature=getattr(config, "temperature", None),
            top_p=getattr(config, "top_p", None),
            routing=provider_cfg.routing,
        )
        
        # Add streaming-specific kwargs
        api_kwargs["stream"] = True
        api_kwargs["stream_options"] = {"include_usage": True}
        
        try:
            response = await client.chat.completions.create(**api_kwargs)
            
            accumulated_text = ""
            accumulated_reasoning = ""
            usage = None
            stop_reason = None
            generation_id = ""
            # Accumulate tool call data: index -> {"id": str, "name": str, "arguments": str}
            tool_call_accumulators: dict[int, dict] = {}
            
            async for chunk in response:
                # Capture generation ID from first chunk
                if not generation_id and getattr(chunk, "id", None):
                    generation_id = chunk.id

                # Handle usage chunk
                if chunk.usage:
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )
                
                # Process content deltas
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Handle text content
                    if delta.content is not None:
                        yield StreamEvent(type="text", content=delta.content)
                        accumulated_text += delta.content
                    
                    # Handle reasoning (OpenRouter extension)
                    reasoning_text = getattr(delta, 'reasoning', None)
                    if isinstance(reasoning_text, str) and reasoning_text:
                        yield StreamEvent(type="thinking", content=reasoning_text)
                        accumulated_reasoning += reasoning_text
                    
                    # Handle tool calls
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_accumulators:
                                tool_call_accumulators[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": tc_delta.function.name or "",
                                    "arguments": "",
                                }
                            if tc_delta.function.arguments:
                                tool_call_accumulators[idx]["arguments"] += tc_delta.function.arguments
                    
                    # Track finish reason
                    if choice.finish_reason:
                        stop_reason = choice.finish_reason
            
            # Yield tool_done events for accumulated tool calls
            for idx in sorted(tool_call_accumulators.keys()):
                tc_data = tool_call_accumulators[idx]
                try:
                    input_dict = json.loads(tc_data["arguments"])
                except json.JSONDecodeError:
                    input_dict = {}
                
                yield StreamEvent(
                    type="tool_done",
                    tool_index=idx,
                    tool_call=ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        input=input_dict,
                    ),
                )
            
            # Build tool_calls list for the response
            response_tool_calls = []
            for idx in sorted(tool_call_accumulators.keys()):
                tc_data = tool_call_accumulators[idx]
                try:
                    input_dict = json.loads(tc_data["arguments"])
                except json.JSONDecodeError:
                    input_dict = {}
                response_tool_calls.append(ToolCall(
                    id=tc_data["id"],
                    name=tc_data["name"],
                    input=input_dict,
                ))
            
            # Build and yield final done event
            thinking_blocks = []
            if accumulated_reasoning:
                thinking_blocks.append(ThinkingBlock(thinking=accumulated_reasoning, signature=""))

            response_obj = Response(
                content=accumulated_text,
                model=api_model,
                usage=usage or Usage(input_tokens=0, output_tokens=0),
                stop_reason=stop_reason or "",
                tool_calls=response_tool_calls,
                thinking=thinking_blocks,
                generation_id=generation_id,
            )
            response_obj.content, response_obj.degenerate = _detect_and_truncate_degeneration(response_obj.content)
            
            yield StreamEvent(
                type="done",
                stop_reason=stop_reason or "",
                model=api_model,
                response=response_obj,
            )
            
        except openai.APIStatusError as e:
            raise ProviderError(
                _sanitize_error(e.message), status_code=e.status_code,
            ) from e
        except openai.APITimeoutError as e:
            raise ProviderError("Provider request timed out") from e
        except openai.APIConnectionError as e:
            raise ProviderError("Provider unreachable — connection failed") from e
    
    else:
        raise ValueError(f"Unsupported provider type: {provider_cfg.type}")


async def complete(
    config: AgentConfig,
    system: str,
    messages: list[dict],
    tools: list | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    thinking: str | None = None,
) -> Response:
    """
    Route to Anthropic or OpenAI SDK based on config.providers.
    If max_tokens is None, use config.max_tokens.
    If model is None, use config.default_model.
    If thinking is None, use config.thinking (defaults to "off").
    
    Errors propagate directly - no wrapping, no retry.
    """
    response = None
    async for event in stream(
        config=config,
        system=system,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        model=model,
        thinking=thinking,
    ):
        if event.type == "done":
            response = event.response
    
    if response and response.generation_id:
        out_tokens = response.usage.output_tokens if response.usage else 0
        has_content = bool(response.content and response.content.strip())
        has_tools = bool(response.tool_calls)
        logger.info(
            "generation %s model=%s out=%d stop=%s content=%s tools=%s",
            response.generation_id, response.model, out_tokens,
            response.stop_reason, has_content, has_tools,
        )

    return response
