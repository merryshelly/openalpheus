"""Sub-agent executor for OpenAlph.

Multi-turn LLM call for focused, isolated tasks. Sub-agents get the parent's
tools (minus subagent itself, preventing recursion) and can iterate up to
a circuit breaker limit (default 10 iterations).
"""

import asyncio
import logging
from dataclasses import replace

from openalph.provider import complete
from openalph.tools import ToolDef, ToolResult, tool_schemas, truncate_result
from openalph.config import AgentConfig

logger = logging.getLogger("openalph.subagent")

MAX_ITERATIONS = 100


async def run_subagent(
    task: str,
    config: AgentConfig,
    tools: list[ToolDef] | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> ToolResult:
    """Execute a multi-turn LLM call as a sub-agent.

    Uses the parent agent's config for API key and provider. The sub-agent
    gets the parent's tools minus 'subagent' (max depth = 1). Iterates
    until a text response or the circuit breaker fires.

    Args:
        task: The task description for the sub-agent
        config: Parent agent's configuration (API key, provider, model)
        tools: Parent's tool list (subagent tool will be filtered out)
        system_prompt: Custom system prompt (default: "You are a helpful assistant.")
        model: Model override (default: use parent's model)
        max_tokens: Max tokens override (default: use parent's max_tokens)

    Returns:
        ToolResult with the LLM's response content, or error description on failure
    """
    # Use default system prompt if not provided
    system = system_prompt if system_prompt is not None else "You are a helpful assistant."

    # Handle model override by creating a new config with the overridden model
    if model is not None:
        config = replace(config, default_model=model)

    # Filter out subagent tool to prevent recursion
    sub_tools = [t for t in (tools or []) if t.name != "subagent"]
    tools_arg = sub_tools if sub_tools else None

    # Build conversation starting with task as user message
    messages = [{"role": "user", "content": task}]

    try:
        for iteration in range(MAX_ITERATIONS):
            response = await complete(
                config=config,
                system=system,
                messages=list(messages),
                tools=tools_arg,
                max_tokens=max_tokens,
            )

            # Text response — done
            if not response.tool_calls:
                return ToolResult(content=response.content, is_error=False)

            # Tool calls — execute and loop
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls,
            })

            # Import execute_tool here to avoid circular import
            from openalph.tools import execute_tool

            tool_coros = []
            for tc in response.tool_calls:
                tool_config = {}
                for t in sub_tools:
                    if t.name == tc.name:
                        tool_config = t.config
                        break
                tool_coros.append(execute_tool(
                    name=tc.name,
                    input=tc.input,
                    tool_config=tool_config,
                    agent_config=config,
                ))

            results = await asyncio.gather(*tool_coros)

            for tc, result in zip(response.tool_calls, results):
                truncated = truncate_result(result.content, config.truncation_limit)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": truncated,
                    "is_error": result.is_error,
                })
                logger.debug("Sub-agent tool %s: %s (%d chars)",
                             tc.name, "error" if result.is_error else "ok",
                             len(truncated))

        # Circuit breaker
        return ToolResult(
            content="[Sub-agent tool call limit reached after "
                    f"{MAX_ITERATIONS} iterations]",
            is_error=True,
        )

    except Exception as e:
        return ToolResult(content=f"Sub-agent error: {e}", is_error=True)
