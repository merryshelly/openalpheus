"""
OpenAlph LLM Provider Adapter

Routes completion requests to either Anthropic or OpenAI SDKs based on configuration.
The adapter handles the differences in API shapes and response formats between providers.
"""

from dataclasses import dataclass
import json
import anthropic
import openai
from openalph.config import AgentConfig

# Client cache: reuse HTTP clients for connection pooling.
# Keyed by (provider, api_key, base_url) so different configs get different clients.
_client_cache: dict[tuple, object] = {}


def _get_client(config: AgentConfig):
    """Get or create a cached provider client."""
    if config.provider == "anthropic":
        key = ("anthropic", config.api_key, None)
        if key not in _client_cache:
            _client_cache[key] = anthropic.AsyncAnthropic(api_key=config.api_key)
        return _client_cache[key]
    elif config.provider == "openai":
        key = ("openai", config.api_key, config.base_url)
        if key not in _client_cache:
            _client_cache[key] = openai.AsyncOpenAI(
                api_key=config.api_key, base_url=config.base_url
            )
        return _client_cache[key]
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


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


@dataclass
class Response:
    content: str
    model: str
    usage: Usage
    stop_reason: str
    tool_calls: list[ToolCall] = None

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


def _convert_tools_for_provider(tools: list | None, provider: str) -> list[dict] | None:
    """Convert ToolDef list to provider-native format.
    
    Anthropic: [{"name": ..., "description": ..., "input_schema": ...}]
    OpenAI: [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
    """
    if tools is None:
        return None
    
    result = []
    for tool in tools:
        if provider == "anthropic":
            result.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        elif provider == "openai":
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
    return result


def _convert_messages_for_provider(messages: list[dict], provider: str) -> list[dict]:
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
    if provider == "anthropic":
        return _convert_messages_for_anthropic(messages)
    elif provider == "openai":
        return _convert_messages_for_openai(messages)
    return messages


def _convert_messages_for_anthropic(messages: list[dict]) -> list[dict]:
    """Convert normalized messages to Anthropic format."""
    result = []
    for msg in messages:
        role = msg.get("role")
        
        if role == "user":
            # Simple user message - pass through
            result.append(msg)
        
        elif role == "assistant":
            # Assistant message may have tool_calls
            tool_calls = msg.get("tool_calls", [])
            content = msg.get("content", "")
            
            if tool_calls:
                # Build content blocks: text block + tool_use blocks
                content_blocks = []
                if content:
                    content_blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                result.append({"role": "assistant", "content": content_blocks})
            else:
                # No tool calls - simple text message
                result.append(msg)
        
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
        
        if role == "assistant":
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
            result.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id"),
                "content": msg.get("content", ""),
            })
        
        else:
            # User and other roles - pass through
            result.append(msg)
    
    return result


def _parse_anthropic_response(response) -> Response:
    """Parse Anthropic response into normalized Response."""
    # Extract text content and tool_calls from content blocks
    text_parts = []
    tool_calls = []
    
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
        else:
            # Backward compatibility: old mocks have .text but no .type="text"
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
    
    content = "\n".join(text_parts) if text_parts else ""
    
    return Response(
        content=content,
        tool_calls=tool_calls,
        model=response.model,
        usage=Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_input_tokens,
            cache_creation_tokens=response.usage.cache_creation_input_tokens,
        ),
        stop_reason=response.stop_reason,
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
    
    return Response(
        content=content,
        tool_calls=tool_calls,
        model=response.model,
        usage=Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        ),
        stop_reason=response.choices[0].finish_reason,
    )


async def complete(
    config: AgentConfig,
    system: str,
    messages: list[dict],
    tools: list | None = None,
    max_tokens: int | None = None,
) -> Response:
    """
    Route to Anthropic or OpenAI SDK based on config.provider.
    If max_tokens is None, use config.max_tokens.
    
    Errors propagate directly - no wrapping, no retry.
    """
    # Use config.max_tokens if max_tokens is not provided
    tokens = max_tokens if max_tokens is not None else config.max_tokens
    
    # Convert messages to provider-native format
    provider_messages = _convert_messages_for_provider(messages, config.provider)
    
    # Convert tools to provider-native format
    provider_tools = _convert_tools_for_provider(tools, config.provider)
    
    if config.provider == "anthropic":
        # Anthropic path: use their specific API shape
        client = _get_client(config)
        
        # Build API call kwargs
        api_kwargs = {
            "model": config.model,
            "system": system,
            "messages": provider_messages,
            "max_tokens": tokens,
        }
        if provider_tools:
            api_kwargs["tools"] = provider_tools
        
        response = await client.messages.create(**api_kwargs)
        
        # Parse Anthropic response into normalized format
        return _parse_anthropic_response(response)
    
    elif config.provider == "openai":
        # OpenAI path: prepend system message and use their API shape
        client = _get_client(config)
        
        # Prepend system message to messages list
        messages_with_system = [{"role": "system", "content": system}] + provider_messages
        
        # Build API call kwargs
        api_kwargs = {
            "model": config.model,
            "messages": messages_with_system,
            "max_tokens": tokens,
        }
        if provider_tools:
            api_kwargs["tools"] = provider_tools
        
        response = await client.chat.completions.create(**api_kwargs)
        
        # Parse OpenAI response into normalized format
        return _parse_openai_response(response)
    
    else:
        # This shouldn't happen if config is validated, but we'll raise a clear error
        raise ValueError(f"Unsupported provider: {config.provider}")
