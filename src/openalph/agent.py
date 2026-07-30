"""Conversation loop for an OpenAlph agent.

Ties together config, prompt assembly, and the provider adapter into a
stateful conversation agent. Keeps a running history and token counts
so the operator can check usage without external tooling.
"""

import asyncio
import contextlib
import base64
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from openalph.config import AgentConfig
from openalph.prompt import assemble_prompt
from openalph.provider import complete, stream, ping_cache, ThinkingBlock, compute_cost
from openalph.tools import discover_tools, execute_tool, truncate_result, wrap_tool_result, escape_system_reminder_tags, _TODO_STATE
from openalph.reminders import ReminderEngine, ReminderState

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


# --- Subagent cache keepalive (workspace-kdsn.190) -------------------------
KEEPALIVE_REFRESH_FRACTION = 0.8   # refresh at 80% of the TTL window
KEEPALIVE_MIN_INTERVAL_S = 60      # floor (protects a 5m-TTL room)
KEEPALIVE_MIN_CACHE_READ = 1000    # a real prefix hit reads at least this many tokens
_KEEPALIVE_TTL_SECONDS = {"5m": 300, "1h": 3600}
KEEPALIVE_MAX_CONSECUTIVE_ERRORS = 3   # abort keepalive after this many back-to-back ping errors
KEEPALIVE_ERROR_BACKOFF_S = 5          # quick retry after a transient ping error (capped by interval)


def _keepalive_ttl_seconds(cache_ttl: str | None) -> int:
    """Resolve a cache_ttl label to seconds; unknown/None -> 1h (platform default)."""
    return _KEEPALIVE_TTL_SECONDS.get(cache_ttl or "1h", 3600)


def _keepalive_interval(cache_ttl: str | None) -> float:
    """Refresh interval: a fraction of the TTL, floored at KEEPALIVE_MIN_INTERVAL_S."""
    return max(_keepalive_ttl_seconds(cache_ttl) * KEEPALIVE_REFRESH_FRACTION,
               KEEPALIVE_MIN_INTERVAL_S)


def _keepalive_is_hit(read: int | None, write: int | None) -> bool:
    """A ping is a cache HIT when the read dominates the write and clears the floor."""
    read = read or 0
    write = write or 0
    return read > write and read >= KEEPALIVE_MIN_CACHE_READ


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
            injection_defense=config.injection_defense,
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
        # CORE-1: per-room, mirroring `_room_locks` below.
        #
        # This was a single `_current_task` slot while room turns run
        # CONCURRENTLY -- every message is dispatched through
        # `_fire_background`, and each turn holds only its own per-room lock,
        # so two rooms can be inside `handle_input` at once. Last writer won,
        # which broke `/stop` in both directions: a `/stop` in room A could
        # cancel room B's turn (whichever wrote the slot last), and the
        # `finally` that nulled the slot when ANY turn ended could leave a
        # `/stop` cancelling nothing while a turn was genuinely running.
        # `_halted_rooms` was already correctly per-room, so A's future
        # messages were dropped while A's actual in-flight turn kept going.
        self._current_tasks: dict[str, asyncio.Task] = {}
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._room_models: dict[str, str] = {}  # room_id → model override
        self._warned_models: set = set()  # warn-once for unknown model context windows
        self._truncation_retry = False
        # R1-4: Per-room reminder engines (replaces shared _reminder_engine)
        self._reminder_engines: dict[str, ReminderEngine] = {}
        # Per-room per-tool-name call counts (session scope)
        self._room_tool_counts: dict[str, dict[str, int]] = {}
        # R1-1: Per-room read registries for file_write guard
        self._read_registries: dict[str, dict] = {}
        # Per-room advisor consult counters (session cap tracking)
        self._advisor_uses: dict[str, int] = {}


    def _engine_for(self, room_id: str) -> ReminderEngine:
        """Lazily create and return a per-room ReminderEngine (R1-4)."""
        if room_id not in self._reminder_engines:
            self._reminder_engines[room_id] = ReminderEngine(self.config)
        return self._reminder_engines[room_id]

    def rehydrate_reminders(self, room_id: str, entries: list[dict]) -> None:
        """Rehydrate reminder fired-state for a room from JSONL entries (R1-4)."""
        self._engine_for(room_id).rehydrate(entries)

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
        self._room_tool_counts.pop(room_id, None)
        # R1-4: reset+drop only this room's engine (not all rooms)
        if room_id in self._reminder_engines:
            self._reminder_engines[room_id].reset()
            del self._reminder_engines[room_id]
        # R1-1: clear this room's read registry
        self._read_registries.pop(room_id, None)
        self._advisor_uses.pop(room_id, None)

    def _usage_for(self, room_id: str) -> dict[str, int]:
        """Lazily init + return the per-room counter record."""
        if room_id not in self._room_usage:
            self._room_usage[room_id] = {
                "uncached_input_tokens": 0, "cache_read_tokens": 0,
                "cache_creation_tokens": 0, "total_output_tokens": 0,
                "total_tool_calls": 0,
                "main_cost_usd": 0.0, "subagent_cost_usd": 0.0,
                "advisor_cost_usd": 0.0, "unpriced_tokens": 0,
            }
        return self._room_usage[room_id]

    def _provider_is_anthropic(self, model_str: str) -> bool:
        """Resolve whether `model_str` routes to an Anthropic-typed provider —
        the authoritative gate for USD cost pricing (F2, kdsn.218). Fail-soft:
        an unknown model or resolution error returns False (the call is tallied
        as unpriced, never mispriced at Anthropic rates, never a crash)."""
        try:
            from openalph.config import resolve_model
            pcfg, _ = resolve_model(model_str, self.config.providers,
                                    aliases=self.config.model_aliases)
            return getattr(pcfg, "type", None) == "anthropic"
        except Exception:
            return False

    def _record_turn_usage(self, room_id: str, usage, model: str, cache_ttl: str | None,
                           is_anthropic: bool | None = None) -> None:
        """Called once per API call (text + tool turns + summary). Updates globals
        AND per-room token counters, and stores the per-turn delta for the serializer.
        `usage` is a provider Usage object (.input_tokens, .output_tokens,
        .cache_read_tokens, .cache_creation_tokens; cache fields may be None).
        `model` is the resolved/authoritative model string for this call (used to
        freeze cost); `cache_ttl` is the room's cache TTL label, used only as the
        aggregate-cache-write fallback multiplier inside compute_cost.
        `is_anthropic` gates pricing by provider type (F2)."""
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
        # frozen cost for THIS call. F3 (kdsn.218): fail-soft — a bad model
        # string or usage shape must never raise through the hot turn loop;
        # on failure record $0 and move on.
        try:
            cr_result = compute_cost(model, usage, cache_ttl_fallback=cache_ttl or "1h",
                                     is_anthropic=is_anthropic)
            _cost_usd = cr_result.cost_usd
            _unpriced = cr_result.unpriced_tokens
        except Exception:
            logger.warning("cost: compute_cost failed for model %r; recording $0", model)
            _cost_usd = 0.0
            _unpriced = 0
        r["main_cost_usd"] += _cost_usd
        r["unpriced_tokens"] += _unpriced
        # per-turn delta (serializer persists this; tool_calls added by serializer)
        self._last_turn_usage[room_id] = {
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "cache_read_tokens": cr, "cache_creation_tokens": cc,
            "cost_usd": _cost_usd, "model": model,
            "cache_creation_5m": usage.cache_creation_5m_tokens or 0,
            "cache_creation_1h": usage.cache_creation_1h_tokens or 0,
            "unpriced_tokens": _unpriced,
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
        """Set the per-room counters from JSONL-summed totals (rehydration).

        Float-safe: cost counters are dollars-and-cents floats, NOT int-cast
        (a blanket int() cast would truncate them to whole dollars).
        Fail-soft (F3, kdsn.218): a malformed persisted value coerces to 0
        per-key rather than raising through _activate_room and bricking the
        room's wake."""
        def _int(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0
        def _float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0
        r = self._usage_for(room_id)
        for k in ("uncached_input_tokens", "cache_read_tokens", "cache_creation_tokens",
                  "total_output_tokens", "total_tool_calls"):
            r[k] = _int(totals.get(k, 0))
        for k in ("main_cost_usd", "subagent_cost_usd", "advisor_cost_usd"):
            r[k] = _float(totals.get(k, 0.0))
        r["unpriced_tokens"] = _int(totals.get("unpriced_tokens", 0))

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

    def _maybe_arm_cache_keepalive(self, *, room_id, active_tool_calls,
                                   request_messages, tools_arg, cache_ttl, callbacks,
                                   thinking, stream_start=None):
        """Arm a background cache-keepalive task iff a subagent is in the batch AND
        the active model is an Anthropic provider with subagent_cache_keepalive on.
        Returns (task, stop_event) or (None, None). Zero overhead in the default path.
        One task covers the parent's single cache prefix even for N parallel subagents.
        """
        if not any(getattr(tc, "name", None) == "subagent" for tc in active_tool_calls):
            return None, None
        try:
            from openalph.config import resolve_model
            provider_cfg, _ = resolve_model(
                self.get_model(room_id), self.config.providers,
                aliases=self.config.model_aliases,
            )
        except Exception:
            return None, None
        if provider_cfg.type != "anthropic" or not getattr(
            provider_cfg, "subagent_cache_keepalive", False
        ):
            return None, None
        stop = asyncio.Event()
        on_miss = (callbacks or {}).get("on_keepalive_miss")
        elapsed = (time.monotonic() - stream_start) if stream_start is not None else 0.0
        logger.info(
            "cache keepalive armed in %s (interval=%.0fs, stream_elapsed=%.1fs)",
            room_id, _keepalive_interval(cache_ttl), elapsed,
        )
        task = asyncio.create_task(self._cache_keepalive(
            system=self.system_prompt,
            messages=request_messages,
            tools=tools_arg,
            cache_ttl=cache_ttl,
            model=self.get_model(room_id),
            thinking=thinking,
            room_id=room_id,
            on_miss=on_miss,
            stop=stop,
            stream_elapsed=elapsed,
        ))
        return task, stop

    async def _cache_keepalive(self, *, system, messages, tools, cache_ttl, model,
                               thinking, room_id, on_miss, stop, stream_elapsed=0.0):
        """Periodically refresh the parent's Anthropic prompt cache while a subagent
        runs. Fires a cheap max_tokens=1 ping just inside the TTL; verifies each ping
        was a cache READ and aborts (with an operator notice) on a WRITE signature so
        a drifted/expired prefix cannot turn insurance into a cost bomb.

        The cache TTL starts at the *request* (stream start), not at arm time, so the
        first fire is shortened by the stream duration already elapsed. Transient ping
        errors get quick retries; KEEPALIVE_MAX_CONSECUTIVE_ERRORS in a row aborts with
        an operator notice instead of retrying blindly for the whole subagent run.
        """
        interval = _keepalive_interval(cache_ttl)
        # First fire accounts for time already consumed by the (possibly slow) stream.
        next_wait = max(interval - max(stream_elapsed, 0.0), 0.0)
        errors = 0
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=next_wait)
                return  # stop set (gather completed / cancelled) -> clean exit
            except asyncio.TimeoutError:
                pass  # time to refresh
            next_wait = interval  # subsequent fires use the full interval unless shortened below
            try:
                usage = await ping_cache(
                    self.config, system=system, messages=messages,
                    tools=tools, cache_ttl=cache_ttl, model=model,
                    thinking_level=thinking,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # A ping failure must NEVER crash the parent turn.
                errors += 1
                logger.warning("cache keepalive ping error #%d in %s: %s", errors, room_id, e)
                if errors >= KEEPALIVE_MAX_CONSECUTIVE_ERRORS:
                    logger.warning(
                        "cache keepalive aborting after %d consecutive errors in %s",
                        errors, room_id,
                    )
                    await self._keepalive_notify_miss(on_miss, room_id)
                    return
                next_wait = min(KEEPALIVE_ERROR_BACKOFF_S, interval)  # quick retry, not a full TTL window
                continue
            errors = 0  # a completed ping (hit or miss) clears the error streak
            if usage is None:
                return  # non-Anthropic (gated upstream) -> nothing to do
            read = usage.cache_read_tokens or 0
            write = usage.cache_creation_tokens or 0
            if _keepalive_is_hit(read, write):
                logger.info("cache keepalive hit in %s (read=%d write=%d)", room_id, read, write)
                continue
            logger.warning(
                "cache keepalive MISS in %s (read=%d write=%d) -- disabling for this turn",
                room_id, read, write,
            )
            await self._keepalive_notify_miss(on_miss, room_id)
            return  # never ping into a miss

    async def _keepalive_notify_miss(self, on_miss, room_id):
        """Fire the operator miss-notice callback, isolated from its own failures."""
        if on_miss is not None:
            try:
                await on_miss(room_id)
            except Exception:
                logger.error("cache keepalive miss notice failed in %s", room_id, exc_info=True)

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
            self._current_tasks[room_id] = asyncio.current_task()
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
                    # R2-A: Escape user-origin <system-reminder> tags in context
                    # to prevent spoofing.  JSONL stores raw text (audit fidelity);
                    # escaping is context-only (mirrors /timesense, /steer).
                    # Only genuine user content is escaped — harness-injected
                    # reminder/steer strings are trusted and appended elsewhere.
                    if isinstance(content, str):
                        _escaped = escape_system_reminder_tags(content)
                    elif isinstance(content, list):
                        # N1: vision/multimodal content — escape text blocks,
                        # preserve image blocks (keeps live/rebuild symmetric).
                        _escaped = [
                            {**_blk, "text": escape_system_reminder_tags(_blk["text"])}
                            if isinstance(_blk, dict) and _blk.get("type") == "text"
                            and isinstance(_blk.get("text"), str) else _blk
                            for _blk in content
                        ]
                    else:
                        _escaped = content
                    history.append({"role": "user", "content": _escaped})
                # else: caller already appended via JSONL → build_context → history.extend

                # Per-turn tool call counter (reset each handle_input call)
                _tool_calls_this_turn: dict[str, int] = {}
                # Reset per-turn reminder state (R1-4: per-room engine)
                self._engine_for(room_id).reset_turn()
                # Determine turn source from callbacks (heartbeat/umbral/None)
                _turn_source = (callbacks or {}).get("turn_source")

                # Compute completed turns (user-role messages in history)
                _completed_turns = sum(1 for m in history if m.get("role") == "user")

                # Ensure per-room session tool counts exist
                if room_id not in self._room_tool_counts:
                    self._room_tool_counts[room_id] = {}

                # Determine enabled tool names — all builtins are available
                # regardless of which tool TOMLs are present in workspace/tools/
                # NOTE (R1-6): ideally {t.name for t in self.tools} but existing
                # test_guidance_injection tests create workspaces without
                # todo_write.toml and rely on T1 firing. Cannot change those
                # tests (FORBIDDEN). R1-6 real-path test validates engine-level
                # suppression. See circuit-breaker report.
                _enabled_tools = {t.name for t in self.tools}

                # Turn-start reminder evaluation (T3 fires here).
                # I1 durability invariant: every injection must become a durable JSONL
                # entry.  At the turn-start site, log_reminder MUST be present before
                # we evaluate or inject.  Without a usable persistence seam, skip
                # evaluation entirely (debug-level log explains why).
                _log_reminder_fn = (callbacks or {}).get("log_reminder")
                if not _log_reminder_fn:
                    logger.debug(
                        "Skipping turn-start reminder evaluation in %s: "
                        "log_reminder callback absent (I1 durability seam required)",
                        room_id,
                    )
                    _turn_start_reminders = []
                else:
                    _turn_start_state = ReminderState(
                        evaluation_point="turn_start",
                        iteration=0,
                        max_iterations=self.config.max_iterations,
                        context_tokens=self._estimate_context_tokens(room_id),
                        context_limit=self._resolve_model_limit(room_id),
                        completed_turns=_completed_turns,
                        turn_source=_turn_source,
                        tool_calls_this_turn=_tool_calls_this_turn,
                        tool_calls_session=dict(self._room_tool_counts.get(room_id, {})),
                        todo_list=list(_TODO_STATE.get(room_id, [])),  # R1-3
                        enabled_tools=_enabled_tools,
                    )
                    _turn_start_reminders = self._engine_for(room_id).evaluate(_turn_start_state)
                for _rem in _turn_start_reminders:
                    history.append({"role": "user", "content": _rem.content})
                    # send_notice is best-effort: failure does not block injection or persistence.
                    _send_notice = (callbacks or {}).get("send_notice")
                    if _send_notice:
                        try:
                            # R2-B: summary line + exact framed content (I2 compliance)
                            import html as _html_mod
                            _notice_body = (
                                f"🔔 System reminder ({_rem.trigger})\n\n"
                                f"{_html_mod.escape(_rem.content)}"
                            )
                            await _send_notice(
                                room_id,
                                _notice_body,
                            )
                        except Exception:
                            logger.warning("send_notice callback failed for reminder")
                    # log_reminder is present (gated above) — persist to JSONL.
                    try:
                        await _log_reminder_fn(room_id, _rem)
                    except Exception:
                        logger.warning("log_reminder callback failed")

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

                    # Reminder evaluation at tool-loop boundary (after steering, before API call).
                    # Ordering: steering drains first, then reminders (operator outranks harness).
                    # R1-5: mirror turn-start durability gate — when production callbacks
                    # (identified by "room_id" key from _build_agent_callbacks) are present
                    # but log_reminder is absent, skip evaluate+inject entirely to enforce
                    # I1 durability invariant.  Without production wiring, no gate.
                    _boundary_log_reminder = (callbacks or {}).get("log_reminder")
                    _is_production_callbacks = callbacks is not None and "room_id" in callbacks
                    if _is_production_callbacks and not _boundary_log_reminder:
                        logger.debug(
                            "Skipping boundary reminder evaluation in %s: "
                            "log_reminder callback absent (I1 durability seam required)",
                            room_id,
                        )
                    else:
                        context_tokens = self._estimate_context_tokens(room_id)
                        limit = self._resolve_model_limit(room_id)
                        _boundary_state = ReminderState(
                            evaluation_point="tool_loop_boundary",
                            iteration=iteration,
                            max_iterations=self.config.max_iterations,
                            context_tokens=context_tokens,
                            context_limit=limit,
                            completed_turns=_completed_turns,
                            turn_source=_turn_source,
                            tool_calls_this_turn=dict(_tool_calls_this_turn),
                            tool_calls_session=dict(self._room_tool_counts.get(room_id, {})),
                            todo_list=list(_TODO_STATE.get(room_id, [])),  # R1-3
                            enabled_tools=_enabled_tools,
                        )
                        _boundary_reminders = self._engine_for(room_id).evaluate(_boundary_state)
                        for _rem in _boundary_reminders:
                            history.append({"role": "user", "content": _rem.content})
                            # send_notice is best-effort: failure must not alter model-visible behavior.
                            _send_notice = (callbacks or {}).get("send_notice")
                            if _send_notice:
                                try:
                                    # R2-B: summary line + exact framed content (I2 compliance)
                                    import html as _html_mod
                                    _notice_body = (
                                        f"🔔 System reminder ({_rem.trigger})\n\n"
                                        f"{_html_mod.escape(_rem.content)}"
                                    )
                                    await _send_notice(
                                        room_id,
                                        _notice_body,
                                    )
                                except Exception:
                                    logger.warning("send_notice callback failed for reminder")
                            # log_reminder is present (gated above) — persist to JSONL.
                            try:
                                await _boundary_log_reminder(room_id, _rem)
                            except Exception:
                                logger.warning("log_reminder callback failed")

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

                    # Cache-keepalive (spec §1.3): snapshot the EXACT request suffix
                    # sent this iteration -- it ends on a valid user/tool_result
                    # boundary. The post-response history ends on the assistant
                    # tool_use turn, which is not a replayable prefix.
                    # INVARIANT: history dicts (and tools_arg) are APPEND-ONLY /
                    # stable between this shallow snapshot and the keepalive's last
                    # ping. In-place mutation of an existing entry would drift the
                    # cache key. The hit-check catches drift (one bounded write) but
                    # preventing it is cheaper -- keep such mutations append-only.
                    _ka_request_messages = list(history)
                    _degenerate_notified = False

                    async for event in stream(
                        config=self.config,
                        system=self.system_prompt,
                        messages=_ka_request_messages,
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
                        elif event.type == "degenerate":
                            # Mid-stream degeneration notice (kdsn.241.21): fire
                            # on_degenerate immediately so the operator sees the
                            # notice while the generation is still in flight, not
                            # after it completes. The provider's monitor.tripped
                            # flag ensures this event fires exactly once.
                            _on_degenerate = (callbacks or {}).get("on_degenerate")
                            if _on_degenerate:
                                try:
                                    await _on_degenerate(
                                        model=event.model,
                                        generation_id=event.generation_id,
                                    )
                                except Exception:
                                    logger.debug("on_degenerate callback raised (ignored)", exc_info=True)
                            _degenerate_notified = True
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

                    # Post-stream degeneration backstop (kdsn.241.21): if the
                    # mid-stream degenerate event didn't fire (e.g. the complete()
                    # fallback path has no stream events), check response.degenerate
                    # here. Skipped if the mid-stream handler already notified.
                    if not _degenerate_notified and getattr(response, 'degenerate', False):
                        _on_degenerate = (callbacks or {}).get("on_degenerate")
                        if _on_degenerate:
                            try:
                                await _on_degenerate(model=self.get_model(room_id), generation_id=getattr(response, 'generation_id', None))
                            except Exception:
                                logger.debug("on_degenerate callback raised (ignored)", exc_info=True)

                    latency_ms = (time.monotonic() - start_time) * 1000

                    self._record_turn_usage(
                        room_id, response.usage, response.model, cache_ttl,
                        is_anthropic=self._provider_is_anthropic(self.get_model(room_id)))
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

                    _ka_task, _ka_stop = self._maybe_arm_cache_keepalive(
                        room_id=room_id,
                        active_tool_calls=active_tool_calls,
                        request_messages=_ka_request_messages,
                        tools_arg=tools_arg,
                        cache_ttl=cache_ttl,
                        callbacks=callbacks,
                        thinking=effective_thinking,
                        stream_start=start_time,
                    )
                    try:
                        results = await asyncio.gather(*tool_coros)
                    finally:
                        if _ka_task is not None:
                            _ka_stop.set()
                            _ka_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await _ka_task
                            logger.info("cache keepalive disarmed in %s", room_id)

                    # Track total tool calls (aggregate + per-tool-name + per-turn)
                    self._record_tool_calls(room_id, len(active_tool_calls))
                    if room_id not in self._room_tool_counts:
                        self._room_tool_counts[room_id] = {}
                    for tc in active_tool_calls:
                        self._room_tool_counts[room_id][tc.name] = (
                            self._room_tool_counts[room_id].get(tc.name, 0) + 1
                        )
                        _tool_calls_this_turn[tc.name] = (
                            _tool_calls_this_turn.get(tc.name, 0) + 1
                        )

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
                        self._record_turn_usage(
                            room_id, summary_response.usage, summary_response.model, cache_ttl,
                            is_anthropic=self._provider_is_anthropic(self.get_model(room_id)))
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
                # Only clear OUR entry, and only if it is still ours: another
                # turn for the same room must not have its task dropped by a
                # late-finishing predecessor.
                if self._current_tasks.get(room_id) is asyncio.current_task():
                    self._current_tasks.pop(room_id, None)

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

    def cancel(self, room_id: str | None = None) -> asyncio.Task | None:
        """Cancel the in-flight turn for `room_id` and return it for awaiting.

        CORE-1: `room_id` is required in practice -- the `/stop` handler has
        it in scope. It stays optional only so an out-of-tree caller does not
        break; passing None cancels every in-flight turn, which is the right
        behaviour for process shutdown and is what `MatrixBot.shutdown` wants.
        It is NOT a "cancel the current one" fallback: there is no such thing
        when rooms run concurrently, and pretending there was is what let a
        `/stop` in one room kill another room's work.
        """
        if room_id is not None:
            task = self._current_tasks.get(room_id)
            if task:
                task.cancel()
            return task

        last: asyncio.Task | None = None
        for task in list(self._current_tasks.values()):
            task.cancel()
            last = task
        return last

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
            "main_cost_usd": _u["main_cost_usd"],
            "subagent_cost_usd": _u["subagent_cost_usd"],
            "advisor_cost_usd": _u["advisor_cost_usd"],
            "unpriced_tokens": _u["unpriced_tokens"],
            "total_cost_usd": _u["main_cost_usd"] + _u["subagent_cost_usd"] + _u["advisor_cost_usd"],
        }
