"""Conversation loop for an OpenAlph agent.

Ties together config, prompt assembly, and the provider adapter into a
stateful conversation agent. Keeps a running history and token counts
so the operator can check usage without external tooling.
"""

import asyncio
from openalph.config import AgentConfig
from openalph.prompt import assemble_prompt
from openalph.provider import complete
from openalph.tools import discover_tools, execute_tool, truncate_result


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
        self._on_tool_call = None  # async callback(name, input, result, is_error)

    def history(self, room_id: str) -> list[dict]:
        """Get or create history for a room."""
        if room_id not in self._rooms:
            self._rooms[room_id] = []
        return self._rooms[room_id]

    def reset_room(self, room_id: str):
        """Clear history for a room."""
        self._rooms[room_id] = []

    async def handle_input(self, text: str, room_id: str = "_default") -> str:
        """Process a user message and return the assistant's response.

        Appends the user message to room history, calls the LLM, appends the
        assistant response, and accumulates token usage. The full room history
        is passed on every call so the model has conversation context.

        Args:
            text: User message
            room_id: Room identifier for per-room history isolation
        """
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

                # Pass a snapshot of history, not the live list.
                response = await complete(
                    config=self.config,
                    system=self.system_prompt,
                    messages=list(history),
                    tools=tools_arg,
                )

                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens

                # Check if response has tool calls
                if not response.tool_calls:
                    # Text response - append to history and return
                    history.append({"role": "assistant", "content": response.content})
                    return response.content

                # Tool use - append assistant message with tool_calls to history
                history.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                })

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
                    ))

                results = await asyncio.gather(*tool_coros)

                # Track total tool calls
                self.total_tool_calls += len(response.tool_calls)

                # Append tool results to history (truncated) and notify
                for tc, result in zip(response.tool_calls, results):
                    truncated_content = truncate_result(result.content, self.config.truncation_limit)
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": truncated_content,
                        "is_error": result.is_error,
                    })
                    if self._on_tool_call:
                        await self._on_tool_call(
                            tc.name, tc.input, truncated_content, result.is_error
                        )

            # Hit max iterations - return limit message
            return "[Tool call limit reached. Please summarize your progress.]"
        finally:
            self._current_task = None

    def cancel(self):
        """Cancel current processing."""
        if self._current_task:
            self._current_task.cancel()

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
