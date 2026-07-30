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

from openalph.provider import complete, compute_cost
from openalph.tools import ToolDef, ToolResult, truncate_result, wrap_tool_result
from openalph.config import AgentConfig

logger = logging.getLogger("openalph.subagent")

# BUG-3: the schema, config default and docstring all say 100, but the real
# fallback used to be this module constant at 200 -- double the documented cap
# and double the worst-case runaway cost. Aligned to the documented default;
# the config's `default_max_iterations` is now actually threaded through
# (see execute_tool), so operator tuning of that key takes effect.
MAX_ITERATIONS = 100

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
    parent_room_id: str | None = None,
    callbacks: dict | None = None,
) -> ToolResult:
    """Execute a multi-turn LLM call as a sub-agent.

    Uses the parent agent's config for API key and provider. The sub-agent
    gets the parent's tools minus 'subagent' (max depth = 1). Iterates
    until a text response or the circuit breaker fires.

    The safety preamble from /srv/openalph/shared/skills/subagent-preamble.md
    is always prepended to the system prompt. Custom system prompts are
    appended after the preamble.

    Two independent, append-only JSONL logs are written as the run
    progresses (workspace-kdsn.192):
      - workspace/logs/subagents/<ts>-<call_id>.jsonl — METRICS (iteration
        token counts, tool NAMES, summary). Pre-existing; untouched here.
      - workspace/sessions/subs/<date>-<call_id>.jsonl — FLIGHT RECORDER,
        a full-content transcript (assistant turns, post-redaction tool
        results, final response). New in this change. Both are pure,
        out-of-band I/O: neither can mutate `messages` or the returned
        ToolResult, and nothing either writes ever re-enters model context.

    Args:
        task: The task description for the sub-agent
        config: Parent agent's configuration (API key, provider, model)
        tools: Parent's tool list (subagent tool will be filtered out)
        system_prompt: Custom system prompt (appended after safety preamble)
        model: Model override (default: use parent's model)
        max_tokens: Max tokens override (default: use parent's max_tokens)
        max_iterations: Max tool-call iterations (default: MAX_ITERATIONS)
        call_id: Optional identifier for cross-referencing logs (default: generated from timestamp)
        parent_room_id: Optional parent Matrix room id, recorded in the flight
            recorder transcript header for cross-referencing only — never
            used for execution/dispatch decisions.

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

    # --- Flight recorder: full-content, append-only transcript (workspace-kdsn.192) ---
    #
    # SEPARATE from the metrics log above (_append_log / logs/subagents) — that log
    # records iteration token counts, tool NAMES, and a summary (metrics, not content).
    # This transcript captures the actual CONTENT of the run: full assistant turns,
    # the post-redaction/post-truncation tool_result bytes, and a final entry. It is
    # PURE OUT-OF-BAND I/O — it must never mutate `messages` or the returned
    # ToolResult, and nothing it writes may ever enter a model context. Every write
    # (including directory creation) is wrapped in try/except and logs-and-continues
    # on failure: a recorder failure must NEVER break the sub run.
    transcript_date = time.strftime("%Y-%m-%d", time.localtime(run_start))
    transcript_filename = f"{transcript_date}-{safe_call_id}.jsonl"
    transcript_dir = Path(config.workspace) / "sessions" / "subs"
    transcript_path = transcript_dir / transcript_filename

    def _append_transcript(entry: dict) -> None:
        try:
            os.makedirs(transcript_dir, exist_ok=True)
            with open(transcript_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as transcript_exc:
            logger.warning("Failed to write subagent flight recorder transcript: %s", transcript_exc)

    def _usage_snapshot() -> dict:
        """Point-in-time token usage snapshot for the flight recorder's final entry."""
        return {
            "uncached_input_tokens": uncached_input_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "total_output_tokens": total_output_tokens,
            "peak_context_tokens": peak_context_tokens,
        }

    _append_transcript({
        "event": "meta",
        "model": config.default_model,
        "parent_room_id": parent_room_id,
        "parent_call_id": call_id,
        "task": task,
        "ts_start": int(run_start),
    })

    def _estimate_context_tokens(msgs: list[dict]) -> int:
        """Estimate token count from messages list (1 token ≈ 4 chars)."""
        total_chars = 0
        for msg in msgs:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
        return total_chars // 4

    # Build conversation starting with task as user message
    messages = [{"role": "user", "content": task}]

    # Tracking counters
    uncached_input_tokens = 0
    cache_read_tokens = 0
    cache_creation_tokens = 0
    total_output_tokens = 0
    total_tool_calls = 0
    completed_iterations = 0
    peak_context_tokens = 0
    # Cost tracking (workspace-kdsn.218): single fixed model per sub-run
    # (config.default_model never changes mid-run -- no per-call switch
    # mechanism exists for subs), so cache_ttl_fallback="1h" is a safe
    # constant (subs have no room-level cache_ttl concept).
    sub_cost = 0.0
    sub_unpriced = 0
    # F2 (kdsn.218 remediation): provider-type gate for pricing. Resolved once
    # (single fixed model per run). Fail-soft: unknown -> not-Anthropic -> the
    # run's tokens are tallied unpriced rather than mispriced or crashing.
    _sub_api_model = config.default_model  # fallback if a response omits .model
    try:
        from openalph.config import resolve_model
        _sub_pcfg, _sub_api_model = resolve_model(config.default_model, config.providers,
                                                  aliases=config.model_aliases)
        _sub_provider_key = getattr(_sub_pcfg, "key", None)
        _sub_provider_type = getattr(_sub_pcfg, "type", None)
    except Exception:
        _sub_api_model = None
        _sub_provider_key = None
        _sub_provider_type = None

    def _write_subagent_bridge() -> None:
        """Fail-soft: write frozen sub-run cost to the callbacks bridge for
        the parent's tool-result JSONL entry. Must never raise (mirrors
        advisor.py's bridge at tools/advisor.py:~415-431)."""
        try:
            if callbacks and "subagent_results" in callbacks:
                _key = (parent_room_id, call_id)
                callbacks["subagent_results"][_key] = {
                    "cost_usd": sub_cost,
                    "unpriced_tokens": sub_unpriced,
                    "model": config.default_model,
                }
        except Exception:
            pass

    def _accrue_cost(model_s, usage_obj) -> None:
        """Fail-soft cost accrual (F3, kdsn.218 re-audit): a compute_cost failure
        must never convert a successful sub-run into an error — log and treat as
        $0, mirroring the main path's guard in agent._record_turn_usage."""
        nonlocal sub_cost, sub_unpriced
        try:
            # re-audit LOW-1: fall back to the resolved API model if a response
            # omits .model, mirroring the advisor path's `_api_model` fallback.
            _cr = compute_cost(model_s or _sub_api_model, usage_obj,
                               cache_ttl_fallback="1h",
                               provider_key=_sub_provider_key,
                               provider_type=_sub_provider_type)
            sub_cost += _cr.cost_usd
            sub_unpriced += _cr.unpriced_tokens
        except Exception:
            logger.warning("subagent cost: compute_cost failed for model %r; recording $0", model_s)

    # Per-sub-agent isolated read registry — prevents sub from using parent's read state
    _sub_read_registry: dict[str, float] = {}
    # Per-sub-agent isolated advisor consult counter — local to this dispatch,
    # not the parent room's (a parent near-cap must not gate the sub).
    _sub_advisor_uses: dict = {}
    # F6 (kdsn.218 remediation): local advisor-cost bridge. A sub keeps the
    # advisor tool; without a bridge here a sub->advisor consult would be costed
    # by neither sub nor parent. run_advisor writes frozen cost keyed by the
    # inner call's tc.id; we drain it after each tool batch into sub_cost.
    _sub_advisor_results: dict = {}

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
                uncached_input_tokens += response.usage.input_tokens or 0
                cache_read_tokens += response.usage.cache_read_tokens or 0
                cache_creation_tokens += response.usage.cache_creation_tokens or 0
                total_output_tokens += response.usage.output_tokens or 0
                _accrue_cost(response.model, response.usage)

            # Flight recorder: record this iteration's assistant turn verbatim
            # (content + tool call name/id/input) — out-of-band, does not touch messages.
            _append_transcript({
                "event": "assistant",
                "iteration": iteration,
                "content": response.content,
                "tool_calls": [
                    {"name": tc.name, "id": tc.id, "input": tc.input}
                    for tc in (response.tool_calls or [])
                ],
            })

            # Text response — check for truncation before accepting
            if not response.tool_calls:
                # If the model was cut off by max_tokens, it may have been
                # about to issue a tool call. Inject a continuation prompt
                # and loop instead of returning truncated output.
                if response.stop_reason in ("max_tokens", "length"):
                    logger.warning(
                        "Sub-agent response truncated (stop_reason=%s) at iteration %d, "
                        "injecting continuation prompt",
                        response.stop_reason, iteration,
                    )
                    _append_log({
                        "event": "truncation_recovery",
                        "iteration": iteration,
                        "stop_reason": response.stop_reason,
                        "truncated_content_length": len(response.content),
                    })
                    # Preserve the truncated text and ask the model to continue
                    messages.append({
                        "role": "assistant",
                        "content": response.content,
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            "[SYSTEM: Your previous response was truncated by the token limit "
                            "(stop_reason=" + response.stop_reason + "). You were cut off mid-output. "
                            "Do NOT repeat what you already said. Continue from where you left off, "
                            "and use tools (file_write, shell, etc.) for any large content instead of "
                            "generating it inline.]"
                        ),
                    })
                    completed_iterations += 1
                    continue

                elapsed = time.time() - run_start
                _append_log({
                    "event": "summary",
                    "status": "completed",
                    "stop_reason": response.stop_reason,
                    "total_iterations": completed_iterations,
                    "total_tool_calls": total_tool_calls,
                    "uncached_input_tokens": uncached_input_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_creation_tokens": cache_creation_tokens,
                    "total_output_tokens": total_output_tokens,
                    "peak_context_tokens": peak_context_tokens,
                    "elapsed_seconds": round(elapsed, 3),
                    "model": config.default_model,
                    "task": task,
                })
                _append_transcript({
                    "event": "final",
                    "status": "completed",
                    "response": response.content,
                    "iterations": completed_iterations,
                    "usage": _usage_snapshot(),
                    "elapsed_s": round(elapsed, 3),
                })
                _write_subagent_bridge()
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
                    callbacks={
                        "read_registry": _sub_read_registry,
                        "get_transcript": lambda: (system, list(messages)),
                        "advisor_uses": _sub_advisor_uses,
                        "advisor_results": _sub_advisor_results,
                        "room_id": "__sub__",
                        # Unique per inner call so concurrent advisor consults
                        # in one batch don't collide on the bridge key (F6).
                        "call_id": tc.id,
                    },
                ))

            results = await asyncio.gather(*tool_coros)

            # F6: fold any nested advisor-consult cost into this sub-run's cost.
            if _sub_advisor_results:
                for _adv in _sub_advisor_results.values():
                    try:
                        sub_cost += float(_adv.get("cost_usd", 0.0) or 0.0)
                        sub_unpriced += int(_adv.get("unpriced_tokens", 0) or 0)
                    except (TypeError, ValueError):
                        pass
                _sub_advisor_results.clear()

            error_count = 0
            for tc, result in zip(response.tool_calls, results):
                if result.is_error:
                    error_count += 1
                truncated = truncate_result(result.content, config.truncation_limit)
                wrapped = wrap_tool_result(truncated, tc.name, tc.id)
                # Flight recorder: record the SAME post-redaction, post-truncation
                # `wrapped` bytes that are about to be appended to `messages` below —
                # i.e. exactly what the sub-agent itself saw, never the raw pre-redaction
                # tool output. Out-of-band: this call cannot affect `messages`.
                _append_transcript({
                    "event": "tool_result",
                    "iteration": iteration,
                    "call_id": tc.id,
                    "name": tc.name,
                    "content": wrapped,
                    "is_error": result.is_error,
                })
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
            context_tokens = _estimate_context_tokens(messages)
            if context_tokens > peak_context_tokens:
                peak_context_tokens = context_tokens
            iter_elapsed = time.time() - iter_start
            _append_log({
                "event": "iteration",
                "iteration": iteration,
                "tools_called": tools_called,
                "errors": error_count,
                "stop_reason": response.stop_reason,
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
                "context_tokens": context_tokens,
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
            "uncached_input_tokens": uncached_input_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "total_output_tokens": total_output_tokens,
            "peak_context_tokens": peak_context_tokens,
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
            if summary.usage:
                _accrue_cost(summary.model, summary.usage)
            final_content = (
                f"⚠️ Sub-agent hit tool call limit ({iteration_limit} iterations). "
                f"Summary:\n\n{summary.content}"
            )
            _append_transcript({
                "event": "final",
                "status": "circuit_breaker",
                "response": final_content,
                "iterations": completed_iterations,
                "usage": _usage_snapshot(),
                "elapsed_s": round(elapsed, 3),
            })
            _write_subagent_bridge()
            return ToolResult(
                content=final_content,
                is_error=True,
            )
        except Exception as e:
            logger.warning("Sub-agent summary generation failed: %s", e)
            final_content = (
                f"⚠️ Sub-agent hit tool call limit ({iteration_limit} iterations). "
                "Summary generation also failed."
            )
            _append_transcript({
                "event": "final",
                "status": "circuit_breaker",
                "response": final_content,
                "iterations": completed_iterations,
                "usage": _usage_snapshot(),
                "elapsed_s": round(elapsed, 3),
            })
            _write_subagent_bridge()
            return ToolResult(
                content=final_content,
                is_error=True,
            )

    except Exception as e:
        elapsed = time.time() - run_start
        _append_log({
            "event": "summary",
            "status": "error",
            "total_iterations": completed_iterations,
            "total_tool_calls": total_tool_calls,
            "uncached_input_tokens": uncached_input_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "total_output_tokens": total_output_tokens,
            "peak_context_tokens": peak_context_tokens,
            "elapsed_seconds": round(elapsed, 3),
            "model": config.default_model,
            "task": task,
            "error": str(e),
        })
        final_content = f"Sub-agent error: {e}"
        _append_transcript({
            "event": "final",
            "status": "error",
            "response": final_content,
            "iterations": completed_iterations,
            "usage": _usage_snapshot(),
            "elapsed_s": round(elapsed, 3),
        })
        _write_subagent_bridge()
        return ToolResult(content=final_content, is_error=True)
