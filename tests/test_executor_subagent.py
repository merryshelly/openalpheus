"""Tests for sub-agent execution.

Interface contract:
    run_subagent(task, config, tools=None, system_prompt=None, model=None,
                 max_tokens=None) -> ToolResult

Multi-turn LLM call via provider.complete(). Sub-agents get the parent's
tools minus 'subagent' (preventing recursion). Circuit breaker at 10 iterations.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock
from pathlib import Path
from openalph.tools.subagent import run_subagent, MAX_ITERATIONS
from openalph.tools import ToolDef, ToolResult
from openalph.config import AgentConfig
from openalph.provider import Response, Usage, ToolCall


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
        """Explicit system_prompt is appended after safety preamble."""
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
        system = call_kwargs["system"]
        # Custom prompt is included in the system prompt
        assert "You are a code reviewer." in system
        # Safety preamble is also present (if the preamble file exists)
        # The custom prompt comes after the preamble, separated by ---
        if "\n\n---\n\n" in system:
            parts = system.split("\n\n---\n\n", 1)
            assert parts[1] == "You are a code reviewer."


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
    async def test_max_iterations_matches_documented_default(self):
        """BUG-3: the hard-fallback constant matches the documented/advertised
        default (100) — the tool schema's `default_max_iterations` and the
        subagent tool-call description both say 100; the module constant used
        to silently diverge at 200, doubling the real worst-case runaway cost."""
        assert MAX_ITERATIONS == 100

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



def truncated_response(content="I was about to use a tool but got cut off..."):
    """Response truncated by max_tokens — no tool calls emitted."""
    return Response(
        content=content,
        tool_calls=[],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=100, output_tokens=4096),
        stop_reason="max_tokens",
    )


class TestTruncationRecovery:

    def _make_workspace(self, tmp_path):
        """Create a workspace directory with log subdirs and return a config."""
        ws = tmp_path / "workspace"
        ws.mkdir()
        return make_config(workspace=ws)

    @pytest.mark.asyncio
    async def test_truncation_triggers_continuation(self, tmp_path):
        """Truncated response (max_tokens) triggers a continuation loop.

        First call returns truncated text. Second call returns normal text.
        Verify: result is successful, complete() called twice, and the
        second call's messages include the continuation prompt.
        """
        config = self._make_workspace(tmp_path)

        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            side_effect=[
                truncated_response("partial output"),
                text_response("Here is the full answer."),
            ],
        ) as mock_complete:
            result = await run_subagent("Do something", config)

        assert result.is_error is False
        assert "full answer" in result.content
        assert mock_complete.call_count == 2

        # Second call should have continuation prompt in messages
        second_call_kwargs = mock_complete.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]
        # Messages: user task, assistant (truncated), user (continuation prompt)
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "partial output"
        assert messages[2]["role"] == "user"
        assert "truncated" in messages[2]["content"].lower() or "stop_reason" in messages[2]["content"]

    @pytest.mark.asyncio
    async def test_truncation_length_stop_reason(self, tmp_path):
        """OpenAI-style stop_reason='length' also triggers recovery."""
        config = self._make_workspace(tmp_path)

        length_response = Response(
            content="cut off by length",
            tool_calls=[],
            model="claude-sonnet-4-20250514",
            usage=Usage(input_tokens=100, output_tokens=4096),
            stop_reason="length",
        )

        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            side_effect=[
                length_response,
                text_response("Continued successfully."),
            ],
        ) as mock_complete:
            result = await run_subagent("Do something", config)

        assert result.is_error is False
        assert "Continued successfully" in result.content
        assert mock_complete.call_count == 2

        # Verify continuation prompt references the stop_reason
        second_messages = mock_complete.call_args_list[1].kwargs["messages"]
        assert second_messages[2]["role"] == "user"
        assert "length" in second_messages[2]["content"]

    @pytest.mark.asyncio
    async def test_truncation_then_tool_use(self, tmp_path):
        """Truncated response → continuation → tool use → final text."""
        config = self._make_workspace(tmp_path)
        tools = make_tools()

        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            side_effect=[
                truncated_response("I need to write a file but—"),
                tool_response("file_write", {"path": "/tmp/out.txt", "content": "data"}, "tc_1"),
                text_response("Done, wrote the file."),
            ],
        ) as mock_complete, patch(
            "openalph.tools.execute_tool",
            new_callable=AsyncMock,
            return_value=ToolResult(content="ok"),
        ):
            result = await run_subagent("Write a file", config, tools=tools)

        assert result.is_error is False
        assert "Done" in result.content
        assert mock_complete.call_count == 3

    @pytest.mark.asyncio
    async def test_normal_completion_not_affected(self, tmp_path):
        """Normal end_turn response returns immediately — no extra calls."""
        config = self._make_workspace(tmp_path)

        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=text_response("Normal response"),
        ) as mock_complete:
            result = await run_subagent("Quick question", config)

        assert result.is_error is False
        assert "Normal response" in result.content
        assert mock_complete.call_count == 1

    @pytest.mark.asyncio
    async def test_truncation_counts_as_iteration(self, tmp_path):
        """Truncation recovery increments the iteration counter.

        With max_iterations=2: first response truncated (iteration 0,
        increments completed_iterations to 1), second response uses a tool
        (iteration 1, increments to 2) → circuit breaker fires.
        """
        config = self._make_workspace(tmp_path)
        tools = make_tools()

        async def _mock_complete(*args, **kwargs):
            call_num = _mock_complete.call_count
            _mock_complete.call_count += 1
            if call_num == 0:
                return truncated_response("truncated...")
            elif kwargs.get("tools") is not None:
                return tool_response("shell", {"command": "echo hi"})
            else:
                # Summary call (tools=None)
                return text_response("Here is what I did so far.")

        _mock_complete.call_count = 0

        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            side_effect=_mock_complete,
        ), patch(
            "openalph.tools.execute_tool",
            new_callable=AsyncMock,
            return_value=ToolResult(content="hi"),
        ):
            result = await run_subagent("Work", config, tools=tools, max_iterations=2)

        assert result.is_error is True
        assert "limit" in result.content.lower()

    @pytest.mark.asyncio
    async def test_completed_summary_includes_stop_reason(self, tmp_path):
        """Normal completion writes a summary log entry with stop_reason."""
        config = self._make_workspace(tmp_path)

        with patch(
            "openalph.tools.subagent.complete",
            new_callable=AsyncMock,
            return_value=text_response("All done."),
        ):
            await run_subagent("Quick task", config)

        # Find the JSONL log file
        log_dir = Path(config.workspace) / "logs" / "subagents"
        log_files = list(log_dir.glob("*.jsonl"))
        assert len(log_files) >= 1, f"Expected JSONL log file in {log_dir}"

        # Read the log and find the summary entry
        log_entries = []
        for lf in log_files:
            for line in lf.read_text().strip().splitlines():
                log_entries.append(json.loads(line))

        summary_entries = [e for e in log_entries if e.get("event") == "summary"]
        assert len(summary_entries) >= 1, "Expected a summary log entry"
        summary = summary_entries[-1]
        assert "stop_reason" in summary, f"Summary missing stop_reason: {summary}"
        assert summary["stop_reason"] == "end_turn"


# ===========================================================================
# BUG-14 — concurrent sub-agents must not share one todo list
# (todo state was keyed on the constant room_id "__sub__")
# ===========================================================================

class TestSubagentTodoIsolation:

    @pytest.mark.asyncio
    async def test_distinct_subagents_do_not_share_todo_state(self):
        from openalph.tools import _execute_todo_write, _TODO_STATE

        cb_a = {"room_id": "__sub__", "call_id": "tc_a"}
        cb_b = {"room_id": "__sub__", "call_id": "tc_b"}

        await _execute_todo_write(
            {"todos": [{"content": "A-only task", "status": "pending"}]}, cb_a
        )
        await _execute_todo_write(
            {"todos": [{"content": "B-only task", "status": "pending"}]}, cb_b
        )

        key_a = ("__sub__", "tc_a")
        key_b = ("__sub__", "tc_b")
        assert key_a in _TODO_STATE and key_b in _TODO_STATE
        assert _TODO_STATE[key_a] != _TODO_STATE[key_b]
        assert _TODO_STATE[key_a][0]["content"] == "A-only task"
        assert _TODO_STATE[key_b][0]["content"] == "B-only task"
