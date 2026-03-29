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
from openalph.provider import complete, stream, StreamEvent
from openalph.tools import discover_tools, execute_tool, truncate_result, wrap_tool_result

logger = logging.getLogger(__name__)

# Supported image MIME types for vision support
VISION_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Media tag regex: [media: <path> (<mime>, <size>)]
MEDIA_TAG_RE = re.compile(r'\[media:\s*(.+?)\s+\(([^,]+),\s*([^)]+)\)\]')

# Approximate tokens per raw image byte (base64 decoded)
IMAGE_TOKENS_PER_BYTE = 1 / 750


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
        self.system_prompt = assemble_prompt(config.workspace)
        self._rooms: dict[str, list[dict]] = {}  # room_id → history
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tool_calls = 0
        # Discover tools from workspace/tools/ directory
        self.tools = discover_tools(config.workspace)
        self._current_task: asyncio.Task | None = None
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._room_models: dict[str, str] = {}  # room_id → model override
        self._truncation_retry = False


    def history(self, room_id: str) -> list[dict]:
        """Get or create history for a room."""
        if room_id not in self._rooms:
            self._rooms[room_id] = []
        return self._rooms[room_id]

    def get_model(self, room_id: str = "_default") -> str:
        """Return the active model for a room, falling back to config default."""
        return self._room_models.get(room_id, self.config.default_model)

    def switch_model(self, model_str: str, room_id: str = "_default") -> str | None:
        """Switch active model. Returns error string on failure, None on success."""
        from openalph.config import resolve_model

        # Validate the model can be resolved
        try:
            resolve_model(model_str, self.config.providers)
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
        # Use model_limits from config if available, else config.model_max_tokens
        model_limit = self.config.model_limits.get(model_str, self.config.model_max_tokens)
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
                "content_preview": content_preview[:200] if content_preview else "",
            }

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write JSONL log: {e}")

    async def handle_input(self, text: str, room_id: str = "_default", *,
                           on_tool_call=None, on_tool_intent=None, thinking: str | None = None,
                           callbacks: dict | None = None,
                           on_text_delta=None, on_thinking_delta=None) -> str:
        """Process a user message and return the assistant's response.

        Appends the user message to room history, calls the LLM, appends the
        assistant response, and accumulates token usage. The full room history
        is passed on every call so the model has conversation context.

        Args:
            text: User message
            room_id: Room identifier for per-room history isolation
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

                # Check for context overflow before appending user message
                content_tokens = self._estimate_content_tokens(content)
                context_tokens = self._estimate_context_tokens(room_id) + content_tokens
                available = self.config.model_max_tokens - self.config.max_tokens
                if context_tokens > available:
                    raise ContextOverflowError(context_tokens, self.config.model_max_tokens)

                history.append({"role": "user", "content": content})

                # Tool loop: continue calling LLM until we get a text response
                for iteration in range(self.config.max_iterations):
                    # Check for context overflow before calling the API (tool results may push over)
                    context_tokens = self._estimate_context_tokens(room_id)
                    available = self.config.model_max_tokens - self.config.max_tokens
                    if context_tokens > available:
                        raise ContextOverflowError(context_tokens, self.config.model_max_tokens)

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
                        thinking=thinking,
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
                            thinking=thinking,
                        )

                    latency_ms = (time.monotonic() - start_time) * 1000

                    self.total_input_tokens += response.usage.input_tokens
                    self.total_output_tokens += response.usage.output_tokens
                    usage = response.usage

                    # Check if response has tool calls
                    if not tool_calls and not response.tool_calls:
                        # Text response - log and return
                        self._log_turn(
                            room_id=room_id,
                            model=self.get_model(room_id),
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            tool_calls=None,
                            latency_ms=latency_ms,
                            content_preview=accumulated_text or response.content,
                            cache_read_tokens=usage.cache_read_tokens,
                            cache_creation_tokens=usage.cache_creation_tokens,
                        )
                        assistant_msg = {"role": "assistant", "content": accumulated_text or response.content}
                        if accumulated_thinking or response.thinking:
                            thinking_blocks = response.thinking if response.thinking else []
                            if accumulated_thinking and not thinking_blocks:
                                thinking_blocks = [ThinkingBlock(thinking=accumulated_thinking, signature="")]
                            assistant_msg["thinking"] = [
                                {"thinking": tb.thinking, "signature": tb.signature}
                                for tb in thinking_blocks
                            ]
                        history.append(assistant_msg)
                        return accumulated_text or response.content

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

                        tool_coros.append(execute_tool(
                            name=tc.name,
                            input=tc.input,
                            tool_config=tool_config,
                            agent_config=self.config,
                            tools=self.tools,
                            callbacks=callbacks,
                        ))

                    results = await asyncio.gather(*tool_coros)

                    # Track total tool calls
                    self.total_tool_calls += len(active_tool_calls)

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
                    )

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

                try:
                    summary_text = ""
                    summary_response = None
                    async for event in stream(
                        config=self.config,
                        system=self.system_prompt,
                        messages=list(history),
                        tools=None,  # no tools — force text response
                        model=self.get_model(room_id),
                        thinking=thinking,
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
                        self.total_input_tokens += summary_response.usage.input_tokens
                        self.total_output_tokens += summary_response.usage.output_tokens

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
                return final_text
            finally:
                self._current_task = None

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
                total_chars += len(str(tc.input))
            # Thinking blocks can be large — include in estimate
            for tb in msg.get("thinking", []):
                total_chars += len(tb.get("thinking", ""))
        return total_chars // 4

    def status(self, room_id: str = "_default") -> dict:
        """Snapshot of agent state for operator visibility."""
        context_tokens = self._estimate_context_tokens(room_id)
        model_max = self.config.model_max_tokens
        context_pct = round(context_tokens / model_max * 100) if model_max else 0
        return {
            "name": self.config.name,
            "model": self.get_model(room_id),
            "turns": sum(1 for m in self.history(room_id) if m["role"] == "user"),
            "context_tokens": context_tokens,
            "context_max": model_max,
            "context_pct": context_pct,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tool_calls": self.total_tool_calls,
        }
