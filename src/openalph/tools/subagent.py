"""Sub-agent executor for OpenAlph.

Multi-turn LLM call for focused, isolated tasks. Sub-agents get the parent's
tools (minus subagent itself, preventing recursion) and can iterate up to
a circuit breaker limit (default 100 iterations).
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import replace
from pathlib import Path

from openalph.provider import complete
from openalph.tools import ToolDef, ToolResult, tool_schemas, truncate_result, wrap_tool_result
from openalph.config import AgentConfig

logger = logging.getLogger("openalph.subagent")

MAX_ITERATIONS = 200

# Safety preamble loaded once at import time — shared across all subagent invocations.
# This file contains hard safety constraints that every subagent must follow.
_PREAMBLE_PATH = Path("/srv/openalph/shared/skills/subagent-preamble.md")
_SAFETY_PREAMBLE: str | None = None

def _load_safety_preamble() -> str:
    """Load the safety preamble from disk, caching after first read."""
    global _SAFETY_PREAMBLE
    if _SAFETY_PREAMBLE is None:
        try:
            _SAFETY_PREAMBLE = _PREAMBLE_PATH.read_text().strip()
            logger.info("Loaded subagent safety preamble (%d chars)", len(_SAFETY_PREAMBLE))
        except FileNotFoundError:
            logger.warning("Subagent safety preamble not found at %s", _PREAMBLE_PATH)
            _SAFETY_PREAMBLE = ""
        except Exception as e:
            logger.warning("Failed to load subagent safety preamble: %s", e)
            _SAFETY_PREAMBLE = ""
    return _SAFETY_PREAMBLE


def _build_system_prompt(custom_prompt: str | None) -> str:
    """Build the full system prompt: safety preamble + custom/default prompt."""
    preamble = _load_safety_preamble()
    user_part = custom_prompt if custom_prompt is not None else ""
    if preamble and user_part:
        return f"{preamble}\n\n---\n\n{user_part}"
    return preamble or user_part or "You are a helpful assistant."


def _sanitize_call_id(call_id: str) -> str:
    """Replace non-alphanumeric characters with underscores for safe filenames."""
    return re.sub(r"[^a-zA-Z0-9]", "_", call_id)


async def run_subagent(
    task: str,
    config: AgentConfig,
    tools: list[ToolDef] | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    max_iterations: int | None = None,
    call_id: str | None = None,
) -> ToolResult:
    """Execute a multi-turn LLM call as a sub-agent.

    Uses the parent agent's config for API key and provider. The sub-agent
    gets the parent's tools minus 'subagent' (max depth = 1). Iterates
    until a text response or the circuit breaker fires.

    The safety preamble from /srv/openalph/shared/skills/subagent-preamble.md
    is always prepended to the system prompt. Custom system prompts are
    appended after the preamble.

    Args:
        task: The task description for the sub-agent
        config: Parent agent's configuration (API key, provider, model)
        tools: Parent's tool list (subagent tool will be filtered out)
        system_prompt: Custom system prompt (appended after safety preamble)
        model: Model override (default: use parent's model)
        max_tokens: Max tokens override (default: use parent's max_tokens)
        max_iterations: Max tool-call iterations (default: MAX_ITERATIONS)
        call_id: Optional identifier for cross-referencing logs (default: generated from timestamp)

    Returns:
        ToolResult with the LLM's response content, or error description on failure
    """
    # Build system prompt: safety preamble + custom/default
    system = _build_system_prompt(system_prompt)

    # Handle model override by creating a new config with the overridden model
    if model is not None:
        config = replace(config, default_model=model)

    # Resolve iteration limit
    iteration_limit = max_iterations if max_iterations is not None else MAX_ITERATIONS

    # Filter out subagent tool to prevent recursion
    sub_tools = [t for t in (tools or []) if t.name != "subagent"]
    tools_arg = sub_tools if sub_tools else None

    # Set up JSONL log file
    run_start = time.time()
    ts = int(run_start)
    if call_id is None:
        safe_call_id = str(ts)
    else:
        safe_call_id = _sanitize_call_id(call_id)
    log_filename = f"{ts}-{safe_call_id}.jsonl"
    log_dir = Path(config.workspace) / "logs" / "subagents"
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir / log_filename

    def _append_log(entry: dict) -> None:
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as log_exc:
            logger.warning("Failed to write subagent log: %s", log_exc)

    # Build conversation starting with task as user message
    messages = [{"role": "user", "content": task}]

    # Tracking counters
    total_input_tokens = 0
    total_output_tokens = 0
    total_tool_calls = 0
    completed_iterations = 0

    try:
        for iteration in range(iteration_limit):
            iter_start = time.time()
            response = await complete(
                config=config,
                system=system,
                messages=list(messages),
                tools=tools_arg,
                max_tokens=max_tokens,
            )

            # Accumulate token counts
            if response.usage:
                total_input_tokens += response.usage.input_tokens or 0
                total_output_tokens += response.usage.output_tokens or 0

            # Text response — done
            if not response.tool_calls:
                elapsed = time.time() - run_start
                _append_log({
                    "event": "summary",
                    "status": "completed",
                    "total_iterations": completed_iterations,
                    "total_tool_calls": total_tool_calls,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "elapsed_seconds": round(elapsed, 3),
                    "model": config.default_model,
                    "task": task,
                })
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

            error_count = 0
            for tc, result in zip(response.tool_calls, results):
                if result.is_error:
                    error_count += 1
                truncated = truncate_result(result.content, config.truncation_limit)
                wrapped = wrap_tool_result(truncated, tc.name, tc.id)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": wrapped,
                    "is_error": result.is_error,
                })
                logger.debug("Sub-agent tool %s: %s (%d chars)",
                             tc.name, "error" if result.is_error else "ok",
                             len(wrapped))

            tools_called = [tc.name for tc in response.tool_calls]
            total_tool_calls += len(tools_called)
            iter_elapsed = time.time() - iter_start
            _append_log({
                "event": "iteration",
                "iteration": iteration,
                "tools_called": tools_called,
                "errors": error_count,
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
                "elapsed_seconds": round(iter_elapsed, 3),
            })
            completed_iterations += 1

        # Circuit breaker — request summary from the model
        logger.warning("Sub-agent tool call limit (%d) reached", iteration_limit)
        limit_notice = (
            "[SYSTEM: Tool call limit reached. You MUST now summarize your progress. "
            "State what you completed, what remains, and any partial results. "
            "Do NOT attempt further tool calls.]"
        )
        messages.append({"role": "user", "content": limit_notice})

        elapsed = time.time() - run_start
        _append_log({
            "event": "summary",
            "status": "circuit_breaker",
            "total_iterations": completed_iterations,
            "total_tool_calls": total_tool_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "elapsed_seconds": round(elapsed, 3),
            "model": config.default_model,
            "task": task,
        })

        try:
            summary = await complete(
                config=config,
                system=system,
                messages=list(messages),
                tools=None,  # no tools — force text response
                max_tokens=max_tokens,
            )
            return ToolResult(
                content=f"⚠️ Sub-agent hit tool call limit ({iteration_limit} iterations). "
                        f"Summary:\n\n{summary.content}",
                is_error=True,
            )
        except Exception as e:
            logger.warning("Sub-agent summary generation failed: %s", e)
            return ToolResult(
                content=f"⚠️ Sub-agent hit tool call limit ({iteration_limit} iterations). "
                        "Summary generation also failed.",
                is_error=True,
            )

    except Exception as e:
        elapsed = time.time() - run_start
        _append_log({
            "event": "summary",
            "status": "error",
            "total_iterations": completed_iterations,
            "total_tool_calls": total_tool_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "elapsed_seconds": round(elapsed, 3),
            "model": config.default_model,
            "task": task,
            "error": str(e),
        })
        return ToolResult(content=f"Sub-agent error: {e}", is_error=True)
