"""Conversation loop for an OpenAlph agent.

Ties together config, prompt assembly, and the provider adapter into a
stateful conversation agent. Keeps a running history and token counts
so the operator can check usage without external tooling.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from openalph.config import AgentConfig
from openalph.prompt import assemble_prompt
from openalph.provider import complete
from openalph.tools import discover_tools, execute_tool, truncate_result

logger = logging.getLogger(__name__)


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


    def history(self, room_id: str) -> list[dict]:
        """Get or create history for a room."""
        if room_id not in self._rooms:
            self._rooms[room_id] = []
        return self._rooms[room_id]

    def _log_turn(
        self,
        room_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tool_calls: list[dict] | None,
        latency_ms: float,
        content_preview: str,
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
                "tool_calls": tool_calls or [],
                "latency_ms": latency_ms,
                "content_preview": content_preview[:200] if content_preview else "",
            }

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write JSONL log: {e}")

    async def handle_input(self, text: str, room_id: str = "_default", *,
                           on_tool_call=None, on_tool_intent=None) -> str:
        """Process a user message and return the assistant's response.

        Appends the user message to room history, calls the LLM, appends the
        assistant response, and accumulates token usage. The full room history
        is passed on every call so the model has conversation context.

        Args:
            text: User message
            room_id: Room identifier for per-room history isolation
        """
        if room_id not in self._room_locks:
            self._room_locks[room_id] = asyncio.Lock()
        async with self._room_locks[room_id]:
            self._current_task = asyncio.current_task()
            history = self.history(room_id)
            try:
                history.append({"role": "user", "content": text})

                # Tool loop: continue calling LLM until we get a text response
                for iteration in range(self.config.max_iterations):
                    # Check for context overflow before calling the API
                    context_tokens = self._estimate_context_tokens(room_id)
                    available = self.config.model_max_tokens - self.config.max_tokens
                    if context_tokens > available:
                        raise ContextOverflowError(context_tokens, self.config.model_max_tokens)

                    # Pass tools=None if no tools discovered (backward compatibility)
                    tools_arg = self.tools if self.tools else None

                    # Record start time for latency measurement
                    start_time = time.monotonic()

                    # Pass a snapshot of history, not the live list.
                    response = await complete(
                        config=self.config,
                        system=self.system_prompt,
                        messages=list(history),
                        tools=tools_arg,
                    )

                    latency_ms = (time.monotonic() - start_time) * 1000

                    self.total_input_tokens += response.usage.input_tokens
                    self.total_output_tokens += response.usage.output_tokens

                    # Check if response has tool calls
                    if not response.tool_calls:
                        # Text response - log and return
                        self._log_turn(
                            room_id=room_id,
                            model=self.config.model,
                            input_tokens=response.usage.input_tokens,
                            output_tokens=response.usage.output_tokens,
                            tool_calls=None,
                            latency_ms=latency_ms,
                            content_preview=response.content,
                        )
                        history.append({"role": "assistant", "content": response.content})
                        return response.content

                    # Tool use - append assistant message with tool_calls to history
                    history.append({
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": response.tool_calls,
                    })

                    # Emit tool intent before execution (for session logging / observability)
                    if on_tool_intent:
                        await on_tool_intent(response.tool_calls, response.content)

                    # Execute tool calls in parallel
                    tool_coros = []
                    for tc in response.tool_calls:
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
                        ))

                    results = await asyncio.gather(*tool_coros)

                    # Track total tool calls
                    self.total_tool_calls += len(response.tool_calls)

                    # Build tool_calls log entry with is_error from results
                    logged_tool_calls = []
                    for tc, result in zip(response.tool_calls, results):
                        logged_tool_calls.append({
                            "name": tc.name,
                            "input": tc.input,
                            "is_error": result.is_error,
                        })

                    # Log this tool-use turn
                    self._log_turn(
                        room_id=room_id,
                        model=self.config.model,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        tool_calls=logged_tool_calls,
                        latency_ms=latency_ms,
                        content_preview=response.content,
                    )

                    # Append tool results to history (truncated) and notify
                    for tc, result in zip(response.tool_calls, results):
                        truncated_content = truncate_result(result.content, self.config.truncation_limit)
                        history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": truncated_content,
                            "is_error": result.is_error,
                        })
                        if on_tool_call:
                            await on_tool_call(
                                tc.id, tc.name, tc.input, truncated_content, result.is_error
                            )

                # Hit max iterations - return limit message
                return "[Tool call limit reached. Please summarize your progress.]"
            finally:
                self._current_task = None

    def cancel(self) -> asyncio.Task | None:
        """Cancel current processing and return the task for awaiting."""
        task = self._current_task
        if task:
            task.cancel()
        return task

    def _estimate_context_tokens(self, room_id: str = "_default") -> int:
        """Estimate current context size in tokens for a room.

        Includes system prompt + room history. Uses 1 token ≈ 4 chars.
        """
        total_chars = len(self.system_prompt)
        for msg in self.history(room_id):
            content = msg.get("content", "")
            if content:
                total_chars += len(content)
            # Tool calls have input dicts — estimate their JSON size
            for tc in msg.get("tool_calls", []):
                total_chars += len(str(tc.input))
        return total_chars // 4

    def status(self, room_id: str = "_default") -> dict:
        """Snapshot of agent state for operator visibility."""
        context_tokens = self._estimate_context_tokens(room_id)
        model_max = self.config.model_max_tokens
        context_pct = round(context_tokens / model_max * 100) if model_max else 0
        return {
            "name": self.config.name,
            "model": self.config.model,
            "turns": sum(1 for m in self.history(room_id) if m["role"] == "user"),
            "context_tokens": context_tokens,
            "context_max": model_max,
            "context_pct": context_pct,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tool_calls": self.total_tool_calls,
        }
