"""Conversation loop for an OpenAlph agent.

Ties together config, prompt assembly, and the provider adapter into a
stateful conversation agent. Keeps a running history and token counts
so the operator can check usage without external tooling.
"""

import asyncio
import base64
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from openalph.config import AgentConfig
from openalph.prompt import assemble_prompt
from openalph.provider import complete, stream, StreamEvent, ThinkingBlock
from openalph.tools import discover_tools, execute_tool, truncate_result, wrap_tool_result

logger = logging.getLogger(__name__)

# Supported image MIME types for vision support
VISION_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Media tag regex: [media: <path> (<mime>, <size>)]
MEDIA_TAG_RE = re.compile(r'\[media:\s*(.+?)\s+\(([^,]+),\s*([^)]+)\)\]')

# Approximate tokens per raw image byte (base64 decoded)
IMAGE_TOKENS_PER_BYTE = 1 / 750

# Wire format overhead per tool interaction (JSON envelope chars not in content)
_TOOL_CALL_OVERHEAD_CHARS = 80   # {"type":"tool_use","id":"...","name":"...","input":}
_TOOL_RESULT_OVERHEAD_CHARS = 80  # {"type":"tool_result","tool_use_id":"...","content":}


def _build_user_content(text: str, config: AgentConfig) -> str | list[dict]:
    """Build user message content, expanding image media tags when vision is enabled.

    Returns plain text string when no image expansion needed.
    Returns list of content blocks when images are present and vision is enabled.
    """
    # If vision is disabled, return text unchanged
    if not config.vision:
        return text

    # Find all media tags
    matches = list(MEDIA_TAG_RE.finditer(text))
    if not matches:
        return text

    content_blocks = []
    last_end = 0
    images_found = False

    for match in matches:
        path_str = match.group(1)
        mime_type = match.group(2)

        # Add text before this tag
        if match.start() > last_end:
            text_before = text[last_end:match.start()]
            if text_before.strip():
                content_blocks.append({"type": "text", "text": text_before})

        # Check if this is a supported image type
        if mime_type in VISION_MIME_TYPES:
            # Resolve path relative to workspace
            image_path = config.workspace / path_str
            try:
                image_bytes = image_path.read_bytes()
                if image_bytes:
                    base64_data = base64.b64encode(image_bytes).decode("ascii")
                    content_blocks.append({
                        "type": "image",
                        "media_type": mime_type,
                        "data": base64_data,
                    })
                    images_found = True
                else:
                    # Empty file - leave tag as text
                    content_blocks.append({"type": "text", "text": match.group(0)})
            except FileNotFoundError:
                logger.warning(f"Image file not found: {image_path}")
                content_blocks.append({"type": "text", "text": match.group(0)})
            except OSError as e:
                logger.warning(f"Failed to read image file {image_path}: {e}")
                content_blocks.append({"type": "text", "text": match.group(0)})
        else:
            # Not a supported image type - leave as text
            content_blocks.append({"type": "text", "text": match.group(0)})

        last_end = match.end()

    # Add any remaining text after the last tag
    if last_end < len(text):
        text_after = text[last_end:]
        if text_after.strip():
            content_blocks.append({"type": "text", "text": text_after})

    # If no images were expanded, return original text
    if not images_found:
        return text

    # Merge consecutive text blocks for efficiency
    merged_blocks = []
    current_text = ""

    for block in content_blocks:
        if block["type"] == "text":
            current_text += block["text"]
        else:
            if current_text:
                merged_blocks.append({"type": "text", "text": current_text})
                current_text = ""
            merged_blocks.append(block)

    if current_text:
        merged_blocks.append({"type": "text", "text": current_text})

    return merged_blocks


class ContextOverflowError(Exception):
    """Raised when conversation context exceeds model capacity."""

    def __init__(self, current_tokens: int, max_tokens: int):
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"Context overflow: ~{current_tokens:,} tokens exceeds model capacity "
            f"({max_tokens:,}). Start a new room."
        )


class Agent:
    """A single-conversation agent backed by an LLM provider."""

    def __init__(self, config: AgentConfig):
        self.config = config
        # Build the system prompt once at init — it won't change mid-conversation.
        # This reads workspace files (SOUL.md, OPERATOR.md, etc.) and builds a
        # skills index, all determined by the workspace directory in config.
        self.system_prompt = assemble_prompt(
            config.workspace,
            model_aliases=config.model_aliases,
        )
        self._rooms: dict[str, list[dict]] = {}  # room_id → history
        self.uncached_input_tokens = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
        self.total_output_tokens = 0
        self.total_tool_calls = 0
        # Per-room usage tracking (Workstream C)
        self._room_usage: dict[str, dict[str, int]] = {}
        self._last_turn_usage: dict[str, dict[str, int]] = {}
        self._last_stop_reason: dict[str, str] = {}
        # Discover tools from workspace/tools/ directory
        self.tools = discover_tools(config.workspace)
        # Pre-compute tool definition cost for token estimation.
        # Tool schemas are sent with every API call via the tools parameter.
        self._tool_defs_chars = 0
        if self.tools:
            for t in self.tools:
                params_chars = len(json.dumps(t.parameters)) if isinstance(t.parameters, dict) else 0
                self._tool_defs_chars += len(t.name) + len(t.description) + params_chars
        self._current_task: asyncio.Task | None = None
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._room_models: dict[str, str] = {}  # room_id → model override
        self._warned_models: set = set()  # warn-once for unknown model context windows
        self._truncation_retry = False


    def reset_room(self, room_id: str) -> None:
        """Clear in-memory history for a room.

        Preserves model overrides (_room_models) and room locks (_room_locks).
        Called by umbral after archive+wipe.
        """
        if room_id in self._rooms:
            self._rooms[room_id].clear()
        self._room_usage.pop(room_id, None)
        self._last_turn_usage.pop(room_id, None)
        self._last_stop_reason.pop(room_id, None)

    def _usage_for(self, room_id: str) -> dict[str, int]:
        """Lazily init + return the per-room counter record (5 keys, all int)."""
        if room_id not in self._room_usage:
            self._room_usage[room_id] = {
                "uncached_input_tokens": 0, "cache_read_tokens": 0,
                "cache_creation_tokens": 0, "total_output_tokens": 0,
                "total_tool_calls": 0,
            }
        return self._room_usage[room_id]

    def _record_turn_usage(self, room_id: str, usage) -> None:
        """Called once per API call (text + tool turns + summary). Updates globals
        AND per-room token counters, and stores the per-turn delta for the serializer.
        `usage` is a provider Usage object (.input_tokens, .output_tokens,
        .cache_read_tokens, .cache_creation_tokens; cache fields may be None)."""
        cr = usage.cache_read_tokens or 0
        cc = usage.cache_creation_tokens or 0
        # globals (unchanged semantics)
        self.uncached_input_tokens += usage.input_tokens
        self.cache_read_tokens += cr
        self.cache_creation_tokens += cc
        self.total_output_tokens += usage.output_tokens
        # per-room mirror
        r = self._usage_for(room_id)
        r["uncached_input_tokens"] += usage.input_tokens
        r["cache_read_tokens"] += cr
        r["cache_creation_tokens"] += cc
        r["total_output_tokens"] += usage.output_tokens
        # per-turn delta (serializer persists this; tool_calls added by serializer)
        self._last_turn_usage[room_id] = {
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "cache_read_tokens": cr, "cache_creation_tokens": cc,
        }

    def _record_tool_calls(self, room_id: str, n: int) -> None:
        self.total_tool_calls += n
        self._usage_for(room_id)["total_tool_calls"] += n

    def last_turn_usage(self, room_id: str) -> dict | None:
        """Public accessor for the matrix serializer."""
        return self._last_turn_usage.get(room_id)

    def last_stop_reason(self, room_id: str) -> str | None:
        """Return the stop_reason from the last completed turn in this room."""
        return self._last_stop_reason.get(room_id)

    def restore_usage(self, room_id: str, totals: dict) -> None:
        """Set the per-room counters from JSONL-summed totals (rehydration)."""
        r = self._usage_for(room_id)
        for k in ("uncached_input_tokens", "cache_read_tokens", "cache_creation_tokens",
                  "total_output_tokens", "total_tool_calls"):
            r[k] = int(totals.get(k, 0))

    def history(self, room_id: str) -> list[dict]:
        """Get or create history for a room."""
        if room_id not in self._rooms:
            self._rooms[room_id] = []
        return self._rooms[room_id]

    def get_model(self, room_id: str = "_default") -> str:
        """Return the active model for a room, falling back to config default."""
        return self._room_models.get(room_id, self.config.default_model)

    def _resolve_model_limit_for(self, model_str: str) -> int:
        """3-layer window resolution for an explicit model string (config override
        -> curated default -> model_max_tokens fallback + warn-once)."""
        if model_str in self.config.model_limits:
            return self.config.model_limits[model_str]
        from openalph.provider import model_context_window
        w = model_context_window(model_str)
        if w is not None:
            return w
        if model_str not in self._warned_models:
            self._warned_models.add(model_str)
            logger.warning("No curated context window for model %r; falling back to "
                           "model_max_tokens=%d. Add a [model_limits] override if its "
                           "window differs.", model_str, self.config.model_max_tokens)
        return self.config.model_max_tokens

    def _resolve_model_limit(self, room_id: str = "_default") -> int:
        """True context window for the room's active model.

        Layer 1: [model_limits] config override.
        Layer 2: curated code default (model_context_window).
        Layer 3: fallback to config.model_max_tokens (+ one-time WARN log).
        Delegates to _resolve_model_limit_for with the room's active model.
        """
        return self._resolve_model_limit_for(self.get_model(room_id))

    def switch_model(self, model_str: str, room_id: str = "_default") -> str | None:
        """Switch active model. Returns error string on failure, None on success."""
        from openalph.config import resolve_model

        # Validate the model can be resolved
        try:
            resolve_model(model_str, self.config.providers, aliases=self.config.model_aliases)
        except ValueError as e:
            return str(e)

        # Vision guard: if session has images, block switch to non-vision model
        history = self.history(room_id)
        has_images = any(
            isinstance(msg.get("content"), list) and
            any(block.get("type") == "image" for block in msg.get("content", []))
            for msg in history
        )
        if has_images:
            return f"Cannot switch to {model_str} — session contains images and model may not support vision."

        # Context window guard: check current context vs model limit
        # Use 3-layer resolution (same as _resolve_model_limit) to get the target model's window
        model_limit = self._resolve_model_limit_for(model_str)
        context_tokens = self._estimate_context_tokens(room_id)
        if context_tokens > model_limit - self.config.max_tokens:
            return f"Cannot switch to {model_str} — current context (~{context_tokens:,} tokens) exceeds model limit ({model_limit:,})."

        self._room_models[room_id] = model_str
        return None


    def _log_turn(
        self,
        room_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tool_calls: list[dict] | None,
        latency_ms: float,
        content_preview: str,
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
        stop_reason: str = "",
    ) -> None:
        """Write a JSONL entry for this LLM turn.

        Failures are logged but do not crash the agent.
        """
        try:
            log_dir = Path(self.config.workspace) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = log_dir / f"{self.config.name}-{date_str}.jsonl"

            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "room_id": room_id,
                "direction": "outbound",
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_creation_tokens": cache_creation_tokens,
                "tool_calls": tool_calls or [],
                "latency_ms": latency_ms,
                "stop_reason": stop_reason,
                "content_preview": content_preview[:200] if content_preview else "",
            }

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write JSONL log: {e}")

    async def handle_input(self, text: str, room_id: str = "_default", *,
                           on_tool_call=None, on_tool_intent=None, thinking: str | None = None,
                           callbacks: dict | None = None,
                           on_text_delta=None, on_thinking_delta=None,
                           on_cache_status=None, cache_ttl: str | None = None,
                           append_user: bool = True,
                           drain_steering=None) -> str:
        """Process a user message and return the assistant's response.

        Appends the user message to room history, calls the LLM, appends the
        assistant response, and accumulates token usage. The full room history
        is passed on every call so the model has conversation context.

        Args:
            text: User message
            room_id: Room identifier for per-room history isolation
            append_user: If True (default), append the user message to history
                before calling the LLM.  Set False when the caller (e.g. the
                gated-room path in matrix.py) has already written the message
                to JSONL and hydrated history from it, so we do not duplicate
                the entry in the wire payload.
            on_text_delta: Optional callback(text: str, done: bool) for text streaming
            on_thinking_delta: Optional callback(text: str, done: bool) for thinking streaming
        """
        if room_id not in self._room_locks:
            self._room_locks[room_id] = asyncio.Lock()
        async with self._room_locks[room_id]:
            self._current_task = asyncio.current_task()
            history = self.history(room_id)
            try:
                # Build user content (may expand image media tags if vision enabled)
                content = _build_user_content(text, self.config)

                # Check for context overflow before appending user message.
                # When append_user=False the message is already in history (hydrated
                # from JSONL by the gated-room path), so don't double-count it.
                content_tokens = self._estimate_content_tokens(content)
                context_tokens = self._estimate_context_tokens(room_id) + (
                    content_tokens if append_user else 0
                )
                limit = self._resolve_model_limit(room_id)
                available = limit - self.config.max_tokens
                if context_tokens > available:
                    raise ContextOverflowError(context_tokens, limit)

                if append_user:
                    history.append({"role": "user", "content": content})
                # else: caller already appended via JSONL → build_context → history.extend

                # Resolve the effective thinking level once (param overrides config).
                # This may be lowered to "off" mid-loop as a one-shot recovery when
                # extended thinking consumes the entire output budget (empty text with
                # stop_reason == "max_tokens"). See the retry logic in the text branch.
                effective_thinking = (
                    thinking if thinking is not None
                    else getattr(self.config, "thinking", "off")
                )
                retried_without_thinking = False

                # Tool loop: continue calling LLM until we get a text response
                for iteration in range(self.config.max_iterations):
                    # Drain steering inbox at the top of every iteration (before API call).
                    # Check both the direct kwarg and the callbacks dict (the latter allows
                    # _process_message to pass the closure without breaking existing test mocks
                    # that have explicit handle_input signatures without drain_steering).
                    _effective_drain = drain_steering or (callbacks or {}).get('drain_steering')
                    if _effective_drain:
                        _steer_notes = await _effective_drain()
                        for _note in _steer_notes:
                            if _note.strip():
                                history.append({
                                    "role": "user",
                                    "content": f"[Operator steering — mid-turn guidance]: {_note}",
                                })

                    # Check for context overflow before calling the API (tool results may push over)
                    context_tokens = self._estimate_context_tokens(room_id)
                    limit = self._resolve_model_limit(room_id)
                    available = limit - self.config.max_tokens
                    if context_tokens > available:
                        raise ContextOverflowError(context_tokens, limit)

                    # Pass tools=None if no tools discovered (backward compatibility)
                    tools_arg = self.tools if self.tools else None

                    # Record start time for latency measurement
                    start_time = time.monotonic()

                    # Use streaming
                    accumulated_text = ""
                    accumulated_thinking = ""
                    response = None
                    text_emitted = False
                    thinking_emitted = False
                    tool_calls = []
                    usage = None

                    async for event in stream(
                        config=self.config,
                        system=self.system_prompt,
                        messages=list(history),
                        tools=tools_arg,
                        model=self.get_model(room_id),
                        thinking=effective_thinking,
                        cache_ttl=cache_ttl,
                    ):
                        if event.type == "text":
                            accumulated_text += event.content
                            if on_text_delta:
                                await on_text_delta(event.content, done=False)
                            text_emitted = True
                        elif event.type == "thinking":
                            accumulated_thinking += event.content
                            if on_thinking_delta:
                                await on_thinking_delta(event.content, done=False)
                            thinking_emitted = True
                        elif event.type == "signature":
                            # Signature is part of thinking block, but we handle it at done
                            pass
                        elif event.type == "tool_done":
                            tool_calls.append(event.tool_call)
                        elif event.type == "done":
                            response = event.response
                            # Fire done signals
                            if text_emitted and on_text_delta:
                                await on_text_delta("", done=True)
                            if thinking_emitted and on_thinking_delta:
                                await on_thinking_delta("", done=True)

                    if response is None:
                        # Fallback if no done event
                        response = await complete(
                            config=self.config,
                            system=self.system_prompt,
                            messages=list(history),
                            tools=tools_arg,
                            model=self.get_model(room_id),
                            thinking=effective_thinking,
                        )

                    latency_ms = (time.monotonic() - start_time) * 1000

                    self._record_turn_usage(room_id, response.usage)
                    usage = response.usage

                    # Check if response has tool calls
                    if not tool_calls and not response.tool_calls:
                        # Text response - log and return
                        final_text = accumulated_text or response.content
                        self._log_turn(
                            room_id=room_id,
                            model=self.get_model(room_id),
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            tool_calls=None,
                            latency_ms=latency_ms,
                            content_preview=final_text,
                            cache_read_tokens=usage.cache_read_tokens,
                            cache_creation_tokens=usage.cache_creation_tokens,
                            stop_reason=response.stop_reason,
                        )
                        if on_cache_status:
                            try:
                                await on_cache_status(usage, self.get_model(room_id))
                            except Exception:
                                logger.error("on_cache_status callback failed", exc_info=True)

                        # Recovery: extended thinking can consume the entire output
                        # budget, yielding empty text with stop_reason == "max_tokens"
                        # (the model "thought" until it hit max_tokens and never wrote
                        # an answer). Retry once with thinking disabled so the full
                        # budget is available for output. Single attempt only; if
                        # thinking is already off we cannot reduce it further, and a
                        # non-empty (merely truncated) answer is kept as-is.
                        if (
                            not (final_text or "").strip()
                            and response.stop_reason == "max_tokens"
                            and effective_thinking != "off"
                            and not retried_without_thinking
                        ):
                            retried_without_thinking = True
                            effective_thinking = "off"
                            # This attempt's token usage and _log_turn entry already
                            # fired above — both reflect a real API call and are kept.
                            logger.warning(
                                "Empty response in %s (stop_reason=max_tokens — extended "
                                "thinking consumed the entire %d-token output budget); "
                                "retrying once with thinking disabled",
                                room_id, usage.output_tokens,
                            )
                            continue

                        assistant_msg = {"role": "assistant", "content": final_text}
                        if accumulated_thinking or response.thinking:
                            thinking_blocks = response.thinking if response.thinking else []
                            if accumulated_thinking and not thinking_blocks:
                                thinking_blocks = [ThinkingBlock(thinking=accumulated_thinking, signature="")]
                            assistant_msg["thinking"] = [
                                {"thinking": tb.thinking, "signature": tb.signature}
                                for tb in thinking_blocks
                            ]
                        # INVARIANT (RC1): assistant_msg must be history[-1] when matrix persists this turn after return.
                        history.append(assistant_msg)
                        self._last_stop_reason[room_id] = response.stop_reason
                        return final_text

                    # Use tool_calls from stream or from response
                    active_tool_calls = tool_calls if tool_calls else response.tool_calls

                    # Tool use - append assistant message with tool_calls to history
                    tool_msg = {
                        "role": "assistant",
                        "content": accumulated_text or response.content,
                        "tool_calls": active_tool_calls,
                    }
                    if accumulated_thinking or response.thinking:
                        thinking_blocks = response.thinking if response.thinking else []
                        if accumulated_thinking and not thinking_blocks:
                            thinking_blocks = [ThinkingBlock(thinking=accumulated_thinking, signature="")]
                        tool_msg["thinking"] = [
                            {"thinking": tb.thinking, "signature": tb.signature}
                            for tb in thinking_blocks
                        ]
                    # INVARIANT (RC1): tool_msg (carrying thinking) MUST remain history[-1]
                    # when on_tool_intent fires — matrix._persist_assistant_turn reads
                    # thinking from history[-1]. Do NOT insert any history mutation between
                    # this append and the on_tool_intent call below.
                    history.append(tool_msg)

                    # Emit tool intent before execution (for session logging / observability)
                    if on_tool_intent:
                        try:
                            await on_tool_intent(active_tool_calls, accumulated_text or response.content)
                        except Exception as e:
                            logger.warning("Tool intent callback failed: %s", e)

                    # Execute tool calls in parallel
                    tool_coros = []
                    for tc in active_tool_calls:
                        # Find the tool config for this tool
                        tool_config = {}
                        for t in self.tools:
                            if t.name == tc.name:
                                tool_config = t.config
                                break

                        # Thread call_id through for subagent log cross-referencing
                        tc_callbacks = {**(callbacks or {}), "call_id": tc.id}
                        tool_coros.append(execute_tool(
                            name=tc.name,
                            input=tc.input,
                            tool_config=tool_config,
                            agent_config=self.config,
                            tools=self.tools,
                            callbacks=tc_callbacks,
                        ))

                    results = await asyncio.gather(*tool_coros)

                    # Track total tool calls
                    self._record_tool_calls(room_id, len(active_tool_calls))

                    # Build tool_calls log entry with is_error from results
                    logged_tool_calls = []
                    for tc, result in zip(active_tool_calls, results):
                        logged_tool_calls.append({
                            "name": tc.name,
                            "input": tc.input,
                            "is_error": result.is_error,
                        })

                    # Log this tool-use turn
                    self._log_turn(
                        room_id=room_id,
                        model=self.get_model(room_id),
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        tool_calls=logged_tool_calls,
                        latency_ms=latency_ms,
                        content_preview=accumulated_text or response.content,
                        cache_read_tokens=usage.cache_read_tokens,
                        cache_creation_tokens=usage.cache_creation_tokens,
                        stop_reason=response.stop_reason,
                    )
                    if on_cache_status:
                        try:
                            await on_cache_status(usage, self.get_model(room_id))
                        except Exception:
                            logger.error("on_cache_status callback failed", exc_info=True)

                    # Append tool results to history (truncated + wrapped) and notify
                    for tc, result in zip(active_tool_calls, results):
                        truncated_content = truncate_result(result.content, self.config.truncation_limit)
                        wrapped_content = wrap_tool_result(truncated_content, tc.name, tc.id)
                        history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": wrapped_content,
                            "is_error": result.is_error,
                        })
                        if on_tool_call:
                            try:
                                await on_tool_call(
                                    tc.id, tc.name, tc.input, wrapped_content, result.is_error
                                )
                            except Exception as e:
                                logger.warning("Tool call callback failed: %s", e)

                # Hit max iterations — request a summary from the model
                logger.warning("Tool call limit (%d) reached in room %s",
                               self.config.max_iterations, room_id)
                limit_notice = (
                    "[SYSTEM: Tool call limit reached. You MUST now summarize your progress. "
                    "State what you completed, what remains, and any partial results. "
                    "Do NOT attempt further tool calls.]"
                )
                history.append({"role": "user", "content": limit_notice})
                # FIX 2: clear stale last-turn usage BEFORE the summary try, so that if
                # the summary fails or yields no done event, serializer sees no usage
                # (preventing double-count of the prior tool turn on restart).
                # On summary success, _record_turn_usage re-sets it correctly.
                self._last_turn_usage.pop(room_id, None)

                try:
                    summary_text = ""
                    summary_response = None
                    async for event in stream(
                        config=self.config,
                        system=self.system_prompt,
                        messages=list(history),
                        tools=None,  # no tools — force text response
                        model=self.get_model(room_id),
                        # Force thinking off for the forced summary: it must produce
                        # visible output, and extended thinking here could consume the
                        # whole budget and return empty with no recovery path (the same
                        # failure the in-loop retry guards against).
                        thinking="off",
                        cache_ttl=cache_ttl,
                    ):
                        if event.type == "text":
                            summary_text += event.content
                            if on_text_delta:
                                await on_text_delta(event.content, done=False)
                        elif event.type == "done":
                            summary_response = event.response
                            if on_text_delta:
                                await on_text_delta("", done=True)

                    if summary_response:
                        self._record_turn_usage(room_id, summary_response.usage)
                        if on_cache_status:
                            try:
                                await on_cache_status(summary_response.usage, self.get_model(room_id))
                            except Exception:
                                logger.error("on_cache_status callback failed", exc_info=True)

                    final_text = summary_text or (
                        summary_response.content if summary_response else
                        "⚠️ Tool call limit reached and summary generation failed."
                    )
                except Exception as e:
                    logger.warning("Summary generation failed after tool limit: %s", e)
                    final_text = (
                        f"⚠️ **Tool call limit reached** ({self.config.max_iterations} iterations). "
                        "Summary generation also failed — check logs for details."
                    )

                history.append({"role": "assistant", "content": final_text})
                self._last_stop_reason[room_id] = summary_response.stop_reason if summary_response else "max_iterations"
                return final_text
            except asyncio.CancelledError:
                # /stop or process shutdown cancelled us mid-tool-loop.
                # The in-memory history may have an assistant message with
                # tool_calls but no tool results (orphan). Strip orphans
                # to prevent bricking the session on the next message.
                self._repair_history(history)
                raise
            finally:
                self._current_task = None

    @staticmethod
    def _repair_history(history: list[dict]) -> None:
        """Strip orphaned tool_calls and their partial results from history.

        An orphan is an assistant message with tool_calls where not all
        tool_call IDs have matching tool results after it in the history.
        This happens when CancelledError interrupts asyncio.gather() during
        tool execution.

        Mutates the list in place. Also strips partial tool results that
        belong to orphaned assistant messages (they would reference
        non-existent tool_use blocks).
        """
        # Identify orphaned assistant messages
        orphaned_indices = set()
        orphaned_tc_ids = set()

        for i, msg in enumerate(history):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                needed_ids = {tc.id for tc in msg["tool_calls"]}
                found_ids = set()
                for j in range(i + 1, len(history)):
                    entry = history[j]
                    if entry.get("role") == "tool" and entry.get("tool_call_id") in needed_ids:
                        found_ids.add(entry["tool_call_id"])
                if needed_ids - found_ids:
                    logger.warning(
                        "Repairing history: stripping orphaned tool_calls %s",
                        needed_ids - found_ids,
                    )
                    orphaned_indices.add(i)
                    orphaned_tc_ids.update(needed_ids)

        if orphaned_indices:
            # Remove orphans and their partial results in reverse order
            to_remove = set()
            for i, msg in enumerate(history):
                if i in orphaned_indices:
                    to_remove.add(i)
                elif (msg.get("role") == "tool" and
                      msg.get("tool_call_id") in orphaned_tc_ids):
                    to_remove.add(i)

            for i in sorted(to_remove, reverse=True):
                history.pop(i)

    def cancel(self) -> asyncio.Task | None:
        """Cancel current processing and return the task for awaiting."""
        task = self._current_task
        if task:
            task.cancel()
        return task

    def _estimate_content_tokens(self, content: str | list[dict]) -> int:
        """Estimate tokens for a single content (string or list of blocks).

        Text: 1 token ≈ 4 chars.
        Images: base64 decoded size * 3/4 / 750.
        """
        if isinstance(content, str):
            return len(content) // 4

        if isinstance(content, list):
            total = 0
            for block in content:
                if block.get("type") == "text":
                    total += len(block.get("text", "")) // 4
                elif block.get("type") == "image":
                    # Estimate from base64 data length
                    base64_len = len(block.get("data", ""))
                    # base64 expands by 4/3, so decoded size = encoded * 3/4
                    decoded_bytes = base64_len * 3 // 4
                    total += int(decoded_bytes * IMAGE_TOKENS_PER_BYTE)
            return total

        return 0

    def _estimate_context_tokens(self, room_id: str = "_default", history: list[dict] | None = None) -> int:
        """Estimate current context size in tokens for a room.

        Includes system prompt + room history. Uses 1 token ≈ 4 chars.
        If history is provided, use that instead of looking up by room_id.
        """
        total_chars = len(self.system_prompt)
        total_chars += self._tool_defs_chars
        msgs = history if history is not None else self.history(room_id)
        for msg in msgs:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # List of content blocks
                for block in content:
                    if block.get("type") == "text":
                        total_chars += len(block.get("text", ""))
                    elif block.get("type") == "image":
                        # Estimate from base64 data length
                        base64_len = len(block.get("data", ""))
                        decoded_bytes = base64_len * 3 // 4
                        total_chars += int(decoded_bytes * IMAGE_TOKENS_PER_BYTE * 4)
            # Tool calls have input dicts — estimate their JSON size
            for tc in msg.get("tool_calls", []):
                total_chars += len(str(tc.input)) + _TOOL_CALL_OVERHEAD_CHARS
            # Thinking blocks can be large — include in estimate
            for tb in msg.get("thinking", []):
                total_chars += len(tb.get("thinking", ""))
                total_chars += len(tb.get("signature", ""))
            # Wire envelope overhead for tool result messages
            if msg.get("role") == "tool":
                total_chars += _TOOL_RESULT_OVERHEAD_CHARS
        return total_chars // 4

    def status(self, room_id: str = "_default", history: list[dict] | None = None) -> dict:
        """Snapshot of agent state for operator visibility.

        If history is provided (e.g. from session_log.build_context()),
        use it for the context estimate instead of raw in-memory history.
        This ensures toolstrip-aware context sizes.
        """
        context_tokens = self._estimate_context_tokens(room_id, history=history)
        model_max = self._resolve_model_limit(room_id)
        context_pct = round(context_tokens / model_max * 100) if model_max else 0
        _u = self._usage_for(room_id)
        return {
            "name": self.config.name,
            "model": self.get_model(room_id),
            "turns": sum(1 for m in self.history(room_id) if m["role"] == "user"),
            "context_tokens": context_tokens,
            "context_max": model_max,
            "context_pct": context_pct,
            "context_remaining": max(0, model_max - context_tokens),
            "uncached_input_tokens": _u["uncached_input_tokens"],
            "cache_read_tokens": _u["cache_read_tokens"],
            "cache_creation_tokens": _u["cache_creation_tokens"],
            "total_output_tokens": _u["total_output_tokens"],
            "total_tool_calls": _u["total_tool_calls"],
        }
