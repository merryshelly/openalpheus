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


class Agent:
    """A single-conversation agent backed by an LLM provider."""

    def __init__(self, config: AgentConfig):
        self.config = config
        # Build the system prompt once at init — it won't change mid-conversation.
        # This reads workspace files (SOUL.md, OPERATOR.md, etc.) and builds a
        # skills index, all determined by the workspace directory in config.
        self.system_prompt = assemble_prompt(config.workspace)
        self.history: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tool_calls = 0
        # Discover tools from workspace/tools/ directory
        self.tools = discover_tools(config.workspace)
        self._current_task: asyncio.Task | None = None

    async def handle_input(self, text: str) -> str:
        """Process a user message and return the assistant's response.

        Appends the user message to history, calls the LLM, appends the
        assistant response, and accumulates token usage. The full history
        is passed on every call so the model has conversation context.
        """
        self._current_task = asyncio.current_task()
        try:
            self.history.append({"role": "user", "content": text})

            # Tool loop: continue calling LLM until we get a text response
            for iteration in range(self.config.max_iterations):
                # Pass tools=None if no tools discovered (backward compatibility)
                tools_arg = self.tools if self.tools else None

                # Pass a snapshot of history, not the live list. The mock test suite
                # inspects call_args after the fact — if we pass the mutable list,
                # it will reflect later mutations (appended assistant response).
                response = await complete(
                    config=self.config,
                    system=self.system_prompt,
                    messages=list(self.history),
                    tools=tools_arg,
                )

                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens

                # Check if response has tool calls
                if not response.tool_calls:
                    # Text response - append to history and return
                    self.history.append({"role": "assistant", "content": response.content})
                    return response.content

                # Tool use - append assistant message with tool_calls to history
                self.history.append({
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

                # Append tool results to history (truncated)
                for tc, result in zip(response.tool_calls, results):
                    truncated_content = truncate_result(result.content, self.config.truncation_limit)
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": truncated_content,
                        "is_error": result.is_error,
                    })

            # Hit max iterations - return limit message
            return "[Tool call limit reached. Please summarize your progress.]"
        finally:
            self._current_task = None

    def cancel(self):
        """Cancel current processing."""
        if self._current_task:
            self._current_task.cancel()

    def status(self) -> dict:
        """Snapshot of agent state for operator visibility."""
        return {
            "name": self.config.name,
            "model": self.config.model,
            # Each turn is one user message + one assistant response.
            "turns": sum(1 for m in self.history if m["role"] == "user"),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tool_calls": self.total_tool_calls,
        }
