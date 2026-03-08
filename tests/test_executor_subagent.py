"""Tests for sub-agent execution.

Interface contract:
    run_subagent(task, config, system_prompt=None, model=None, max_tokens=None) -> ToolResult

Single-turn LLM call via provider.complete(). No tools for sub-agents.
Uses parent's config for API key/provider. model/system_prompt can be overridden.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path
from openalph.tools.subagent import run_subagent
from openalph.tools import ToolResult
from openalph.config import AgentConfig
from openalph.provider import Response, Usage


def make_config(**kwargs):
    defaults = dict(
        name="test-parent",
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        provider="anthropic",
        api_key="sk-test",
        base_url=None,
        workspace=Path("/tmp/test"),
        max_iterations=25,
        truncation_limit=50000,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def mock_response(text="Sub-agent response", model="claude-sonnet-4-20250514"):
    return Response(
        content=text,
        tool_calls=[],
        model=model,
        usage=Usage(input_tokens=100, output_tokens=50),
        stop_reason="end_turn",
    )


class TestBasicExecution:

    @pytest.mark.asyncio
    async def test_returns_llm_response(self):
        """Sub-agent returns the LLM's text response as content."""
        config = make_config()
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=mock_response("The answer is 42"),
        ):
            result = await run_subagent("What is the meaning of life?", config)

        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert "42" in result.content

    @pytest.mark.asyncio
    async def test_task_sent_as_user_message(self):
        """Task string is sent as the user message to the LLM."""
        config = make_config()
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=mock_response(),
        ) as mock_complete:
            await run_subagent("Summarize this document", config)

        call_kwargs = mock_complete.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Summarize this document"


class TestSystemPrompt:

    @pytest.mark.asyncio
    async def test_default_system_prompt(self):
        """No system_prompt → uses a minimal default."""
        config = make_config()
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=mock_response(),
        ) as mock_complete:
            await run_subagent("Do something", config)

        call_kwargs = mock_complete.call_args.kwargs
        system = call_kwargs["system"]
        assert isinstance(system, str)
        assert len(system) > 0  # not empty

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self):
        """Explicit system_prompt is used instead of default."""
        config = make_config()
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=mock_response(),
        ) as mock_complete:
            await run_subagent(
                "Analyze this",
                config,
                system_prompt="You are a code reviewer.",
            )

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["system"] == "You are a code reviewer."


class TestModelOverride:

    @pytest.mark.asyncio
    async def test_default_uses_parent_model(self):
        """No model override → uses parent's model from config."""
        config = make_config(model="claude-opus-4-20250514")
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=mock_response(),
        ) as mock_complete:
            await run_subagent("Do something", config)

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["config"].model == "claude-opus-4-20250514"

    @pytest.mark.asyncio
    async def test_model_override(self):
        """Explicit model parameter overrides parent's model."""
        config = make_config(model="claude-opus-4-20250514")
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=mock_response(),
        ) as mock_complete:
            await run_subagent("Do something", config, model="claude-haiku-3-5-20241022")

        # The config passed to complete should use the overridden model
        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["config"].model == "claude-haiku-3-5-20241022"


class TestMaxTokens:

    @pytest.mark.asyncio
    async def test_default_max_tokens(self):
        """No max_tokens → uses parent config's max_tokens."""
        config = make_config(max_tokens=4096)
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=mock_response(),
        ) as mock_complete:
            await run_subagent("Do something", config)

        # Should use config default, not override
        call_kwargs = mock_complete.call_args.kwargs
        # max_tokens either not passed (uses config) or matches config
        max_tok = call_kwargs.get("max_tokens")
        if max_tok is not None:
            assert max_tok == 4096

    @pytest.mark.asyncio
    async def test_max_tokens_override(self):
        """Explicit max_tokens overrides config default."""
        config = make_config(max_tokens=8192)
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=mock_response(),
        ) as mock_complete:
            await run_subagent("Do something", config, max_tokens=1024)

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs.get("max_tokens") == 1024


class TestNoTools:

    @pytest.mark.asyncio
    async def test_sub_does_not_receive_tools(self):
        """Sub-agents are called without tools (single-turn, no tool use)."""
        config = make_config()
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=mock_response(),
        ) as mock_complete:
            await run_subagent("Do something", config)

        call_kwargs = mock_complete.call_args.kwargs
        # tools should be None or not passed
        tools = call_kwargs.get("tools")
        assert tools is None


class TestErrorHandling:

    @pytest.mark.asyncio
    async def test_llm_error_returns_tool_error(self):
        """LLM API error → is_error=True with error description."""
        config = make_config()
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            side_effect=Exception("Rate limit exceeded"),
        ):
            result = await run_subagent("Do something", config)

        assert result.is_error is True
        assert "rate limit" in result.content.lower() or "error" in result.content.lower()

    @pytest.mark.asyncio
    async def test_empty_response_not_error(self):
        """Empty LLM response is not an error (just empty content)."""
        config = make_config()
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=mock_response(""),
        ):
            result = await run_subagent("Do something", config)

        assert result.is_error is False
