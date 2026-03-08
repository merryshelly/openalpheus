"""Sub-agent executor for OpenAlph.

Single-turn LLM call for focused, isolated tasks. Sub-agents do not have
access to tools—they return a text response that the parent agent can use.
"""

from dataclasses import replace

from openalph.provider import complete
from openalph.tools import ToolResult
from openalph.config import AgentConfig


async def run_subagent(
    task: str,
    config: AgentConfig,
    system_prompt: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> ToolResult:
    """Execute a single-turn LLM call as a sub-agent.
    
    Uses the parent agent's config for API key and provider. The sub-agent
    has no access to tools—it returns a text response for the parent to use.
    
    Args:
        task: The task description for the sub-agent
        config: Parent agent's configuration (API key, provider, model)
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
        config = replace(config, model=model)
    
    # Build user message from task
    messages = [{"role": "user", "content": task}]
    
    try:
        # Call the LLM with no tools (single-turn, no tool use for sub-agents)
        response = await complete(
            config=config,
            system=system,
            messages=messages,
            tools=None,
            max_tokens=max_tokens,
        )
        
        # Return the response content as a successful ToolResult
        return ToolResult(content=response.content, is_error=False)
        
    except Exception as e:
        # On any LLM error, return an error ToolResult
        return ToolResult(content=f"Sub-agent error: {e}", is_error=True)
