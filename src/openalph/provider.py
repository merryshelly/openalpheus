"""
OpenAlph LLM Provider Adapter

Routes completion requests to either Anthropic or OpenAI SDKs based on configuration.
The adapter handles the differences in API shapes and response formats between providers.
"""

from dataclasses import dataclass
import anthropic
import openai
from openalph.config import AgentConfig


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None


@dataclass
class Response:
    content: str
    model: str
    usage: Usage
    stop_reason: str


async def complete(config: AgentConfig, system: str, messages: list[dict], max_tokens: int | None = None) -> Response:
    """
    Route to Anthropic or OpenAI SDK based on config.provider.
    If max_tokens is None, use config.max_tokens.
    
    Errors propagate directly - no wrapping, no retry.
    """
    # Use config.max_tokens if max_tokens is not provided
    tokens = max_tokens if max_tokens is not None else config.max_tokens
    
    if config.provider == "anthropic":
        # Anthropic path: use their specific API shape
        client = anthropic.AsyncAnthropic(api_key=config.api_key)
        response = await client.messages.create(
            model=config.model,
            system=system,
            messages=messages,
            max_tokens=tokens
        )
        
        # Extract Anthropic-specific fields
        return Response(
            content=response.content[0].text,
            model=response.model,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=response.usage.cache_read_input_tokens,
                cache_creation_tokens=response.usage.cache_creation_input_tokens
            ),
            stop_reason=response.stop_reason
        )
    
    elif config.provider == "openai":
        # OpenAI path: prepend system message and use their API shape
        client = openai.AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        
        # Prepend system message to messages list
        messages_with_system = [{"role": "system", "content": system}] + messages
        
        response = await client.chat.completions.create(
            model=config.model,
            messages=messages_with_system,
            max_tokens=tokens
        )
        
        # Extract OpenAI-specific fields
        # Note: OpenAI doesn't provide cache token metrics, so they remain None
        return Response(
            content=response.choices[0].message.content,
            model=response.model,
            usage=Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
                # cache_read_tokens and cache_creation_tokens remain None (default)
            ),
            stop_reason=response.choices[0].finish_reason
        )
    
    else:
        # This shouldn't happen if config is validated, but we'll raise a clear error
        raise ValueError(f"Unsupported provider: {config.provider}")
