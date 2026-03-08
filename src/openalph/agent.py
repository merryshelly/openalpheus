"""Conversation loop for an OpenAlph agent.

Ties together config, prompt assembly, and the provider adapter into a
stateful conversation agent. Keeps a running history and token counts
so the operator can check usage without external tooling.
"""

from openalph.config import AgentConfig
from openalph.prompt import assemble_prompt
from openalph.provider import complete


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

    async def handle_input(self, text: str) -> str:
        """Process a user message and return the assistant's response.

        Appends the user message to history, calls the LLM, appends the
        assistant response, and accumulates token usage. The full history
        is passed on every call so the model has conversation context.
        """
        self.history.append({"role": "user", "content": text})

        # Pass a snapshot of history, not the live list. The mock test suite
        # inspects call_args after the fact — if we pass the mutable list,
        # it will reflect later mutations (appended assistant response).
        response = await complete(
            config=self.config,
            system=self.system_prompt,
            messages=list(self.history),
        )

        self.history.append({"role": "assistant", "content": response.content})
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        return response.content

    def status(self) -> dict:
        """Snapshot of agent state for operator visibility."""
        return {
            "name": self.config.name,
            "model": self.config.model,
            # Each turn is one user message + one assistant response.
            "turns": sum(1 for m in self.history if m["role"] == "user"),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
