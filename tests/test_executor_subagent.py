"""Tests for sub-agent execution.

Interface contract:
    run_subagent(task, config, tools=None, system_prompt=None, model=None,
                 max_tokens=None) -> ToolResult

Multi-turn LLM call via provider.complete(). Sub-agents get the parent's
tools minus 'subagent' (preventing recursion). Circuit breaker at 10 iterations.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path
from openalph.tools.subagent import run_subagent, MAX_ITERATIONS
from openalph.tools import ToolDef, ToolResult
from openalph.config import AgentConfig
from openalph.provider import Response, Usage, ToolCall


def make_provider(key="default", type="anthropic", api_key="sk-test", base_url=None, quirks=None):
    from openalph.config import ProviderConfig
    return ProviderConfig(key=key, type=type, api_key=api_key, base_url=base_url, quirks=quirks or [])


def make_provider(key="default", type="anthropic", api_key="sk-test", base_url=None, quirks=None):
    from openalph.config import ProviderConfig
    return ProviderConfig(key=key, type=type, api_key=api_key, base_url=base_url, quirks=quirks or [])


def make_config(**kwargs):
    defaults = dict(
        name="test-parent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider(key="anthropic")},
        workspace=Path("/tmp/test"),
        max_iterations=25,
        truncation_limit=50000,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_tools():
    """Build a realistic tool set including subagent (should be filtered)."""
    return [
        ToolDef(name="shell", description="Run shell", parameters={}, config={}),
        ToolDef(name="file_read", description="Read file", parameters={}, config={}),
        ToolDef(name="file_write", description="Write file", parameters={}, config={}),
        ToolDef(name="subagent", description="Spawn sub", parameters={}, config={}),
    ]


def text_response(text="Sub-agent response"):
    return Response(
        content=text,
        tool_calls=[],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=100, output_tokens=50),
        stop_reason="end_turn",
    )


def tool_response(tool_name="shell", tool_input=None, tool_id="tc_1", content=""):
    return Response(
        content=content,
        tool_calls=[ToolCall(id=tool_id, name=tool_name, input=tool_input or {})],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=100, output_tokens=50),
        stop_reason="tool_use",
    )


class TestBasicExecution:

    @pytest.mark.asyncio
    async def test_returns_llm_response(self):
        """Sub-agent returns the LLM's text response as content."""
        config = make_config()
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=text_response("The answer is 42"),
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
            return_value=text_response(),
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
            return_value=text_response(),
        ) as mock_complete:
            await run_subagent("Do something", config)

        call_kwargs = mock_complete.call_args.kwargs
        system = call_kwargs["system"]
        assert isinstance(system, str)
        assert len(system) > 0

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self):
        """Explicit system_prompt is used instead of default."""
        config = make_config()
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=text_response(),
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
        config = make_config(default_model="claude-opus-4-20250514")
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=text_response(),
        ) as mock_complete:
            await run_subagent("Do something", config)

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["config"].default_model == "claude-opus-4-20250514"

    @pytest.mark.asyncio
    async def test_model_override(self):
        """Explicit model parameter overrides parent's model."""
        config = make_config(default_model="claude-opus-4-20250514")
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=text_response(),
        ) as mock_complete:
            await run_subagent("Do something", config, model="claude-haiku-3-5-20241022")

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["config"].default_model == "claude-haiku-3-5-20241022"


class TestMaxTokens:

    @pytest.mark.asyncio
    async def test_max_tokens_override(self):
        """Explicit max_tokens overrides config default."""
        config = make_config(max_tokens=8192)
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=text_response(),
        ) as mock_complete:
            await run_subagent("Do something", config, max_tokens=1024)

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs.get("max_tokens") == 1024


class TestToolFiltering:

    @pytest.mark.asyncio
    async def test_subagent_tool_filtered_out(self):
        """Sub-agents receive parent tools minus 'subagent' (no recursion)."""
        config = make_config()
        tools = make_tools()
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=text_response(),
        ) as mock_complete:
            await run_subagent("Do something", config, tools=tools)

        call_kwargs = mock_complete.call_args.kwargs
        passed_tools = call_kwargs.get("tools", [])
        tool_names = [t.name for t in passed_tools]
        assert "subagent" not in tool_names
        assert "shell" in tool_names
        assert "file_read" in tool_names
        assert "file_write" in tool_names

    @pytest.mark.asyncio
    async def test_no_tools_passed_none(self):
        """No tools → tools=None passed to complete."""
        config = make_config()
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=text_response(),
        ) as mock_complete:
            await run_subagent("Do something", config, tools=None)

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs.get("tools") is None

    @pytest.mark.asyncio
    async def test_only_subagent_tool_results_in_none(self):
        """If parent only has subagent tool, sub gets tools=None."""
        config = make_config()
        tools = [ToolDef(name="subagent", description="Spawn", parameters={}, config={})]
        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=text_response(),
        ) as mock_complete:
            await run_subagent("Do something", config, tools=tools)

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs.get("tools") is None


class TestMultiTurnToolUse:

    @pytest.mark.asyncio
    async def test_tool_call_then_text(self):
        """Sub-agent calls a tool, gets result, then responds with text."""
        config = make_config()
        tools = make_tools()

        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            side_effect=[
                tool_response("shell", {"command": "date"}),
                text_response("Today is March 8, 2026"),
            ],
        ), patch(
            "openalph.tools.execute_tool",
            new_callable=AsyncMock,
            return_value=ToolResult(content="Sat Mar  8 19:00:00 EDT 2026"),
        ):
            result = await run_subagent("What's the date?", config, tools=tools)

        assert result.is_error is False
        assert "March 8" in result.content

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_before_text(self):
        """Sub-agent makes multiple tool iterations before final text."""
        config = make_config()
        tools = make_tools()

        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            side_effect=[
                tool_response("file_read", {"path": "/tmp/a.txt"}, "tc_1"),
                tool_response("file_write", {"path": "/tmp/b.txt"}, "tc_2"),
                text_response("Done, wrote the file."),
            ],
        ), patch(
            "openalph.tools.execute_tool",
            new_callable=AsyncMock,
            return_value=ToolResult(content="ok"),
        ):
            result = await run_subagent("Copy a.txt to b.txt", config, tools=tools)

        assert result.is_error is False
        assert "Done" in result.content


class TestCircuitBreaker:

    @pytest.mark.asyncio
    async def test_circuit_breaker_fires(self):
        """After MAX_ITERATIONS tool calls, returns error with summary."""
        config = make_config()
        tools = make_tools()

        call_count = 0

        async def _mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("tools") is not None:
                return tool_response("shell", {"command": "echo loop"})
            else:
                # Summary call (tools=None)
                return text_response("Here is what I did so far.")

        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            side_effect=_mock_complete,
        ), patch(
            "openalph.tools.execute_tool",
            new_callable=AsyncMock,
            return_value=ToolResult(content="looping"),
        ) as mock_exec:
            result = await run_subagent("Loop forever", config, tools=tools)

        assert result.is_error is True
        assert "limit" in result.content.lower()
        assert "what I did" in result.content  # summary content present
        assert mock_exec.call_count == MAX_ITERATIONS
        assert call_count == MAX_ITERATIONS + 1  # iterations + summary

    @pytest.mark.asyncio
    async def test_max_iterations_is_200(self):
        """Circuit breaker is set to 200 iterations."""
        assert MAX_ITERATIONS == 200

    @pytest.mark.asyncio
    async def test_circuit_breaker_summary_failure_returns_fallback(self):
        """If summary generation fails, a static fallback is returned."""
        config = make_config()
        tools = make_tools()

        async def _mock_complete(*args, **kwargs):
            if kwargs.get("tools") is not None:
                return tool_response("shell", {"command": "echo"})
            else:
                raise RuntimeError("Provider down")

        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            side_effect=_mock_complete,
        ), patch(
            "openalph.tools.execute_tool",
            new_callable=AsyncMock,
            return_value=ToolResult(content="ok"),
        ):
            result = await run_subagent("Loop forever", config, tools=tools)

        assert result.is_error is True
        assert "limit" in result.content.lower()
        assert "failed" in result.content.lower()

    @pytest.mark.asyncio
    async def test_custom_max_iterations(self):
        """max_iterations parameter overrides the default."""
        config = make_config()
        tools = make_tools()

        call_count = 0

        async def _mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("tools") is not None:
                return tool_response("shell", {"command": "echo"})
            else:
                return text_response("Done after custom limit.")

        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            side_effect=_mock_complete,
        ), patch(
            "openalph.tools.execute_tool",
            new_callable=AsyncMock,
            return_value=ToolResult(content="ok"),
        ) as mock_exec:
            result = await run_subagent("Work", config, tools=tools, max_iterations=5)

        assert result.is_error is True
        assert "5" in result.content  # mentions the custom limit
        assert mock_exec.call_count == 5
        assert call_count == 6  # 5 iterations + 1 summary


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
        assert "error" in result.content.lower()

    @pytest.mark.asyncio
    async def test_tool_error_continues_loop(self):
        """Tool execution error doesn't crash the sub — error fed back to LLM."""
        config = make_config()
        tools = make_tools()

        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            side_effect=[
                tool_response("shell", {"command": "bad_cmd"}),
                text_response("The command failed, here's what happened."),
            ],
        ), patch(
            "openalph.tools.execute_tool",
            new_callable=AsyncMock,
            return_value=ToolResult(content="command not found", is_error=True),
        ):
            result = await run_subagent("Run bad_cmd", config, tools=tools)

        assert result.is_error is False
        assert "failed" in result.content.lower()
