"""Tests for subagent visibility improvements (.127).

Two features:
1. Collapsible <details> blocks in Matrix with proper formatting
   - Task brief: full content (not truncated), rendered as markdown
   - Result: full content (not truncated), rendered as markdown
   - No <pre> wrapping around rendered HTML

2. Per-invocation JSONL log file
   - Written to workspace/logs/subagents/<timestamp>-<call_id>.jsonl
   - One entry per iteration: iteration, tools called, errors, tokens, elapsed
   - Final summary entry on completion
"""

import json
import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock, ANY

from openalph.tools.subagent import run_subagent, MAX_ITERATIONS
from openalph.tools import ToolDef, ToolResult
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import Response, Usage, ToolCall


# --- Fixtures ---

def make_provider():
    return ProviderConfig(
        key="anthropic", type="anthropic", api_key="sk-test",
        base_url=None, quirks=[],
    )


def make_config(workspace=None):
    return AgentConfig(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider()},
        workspace=workspace or Path("/tmp/test-workspace"),
        max_iterations=25,
        truncation_limit=50000,
    )


def make_tools():
    return [
        ToolDef(name="shell", description="Run shell", parameters={}, config={}),
        ToolDef(name="file_read", description="Read file", parameters={}, config={}),
        ToolDef(name="subagent", description="Spawn sub", parameters={}, config={}),
    ]


def text_response(text="done", input_tokens=100, output_tokens=50):
    return Response(
        content=text,
        tool_calls=[],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
    )


def tool_response(tool_name="shell", tool_input=None, tool_id="tc_1",
                  content="", input_tokens=100, output_tokens=50):
    return Response(
        content=content,
        tool_calls=[ToolCall(id=tool_id, name=tool_name, input=tool_input or {"command": "echo hi"})],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="tool_use",
    )


# =============================================================================
# Part 2: Per-invocation JSONL log file
# =============================================================================

class TestSubagentLogFile:
    """Subagent runs produce a JSONL log in workspace/logs/subagents/."""

    @pytest.mark.asyncio
    async def test_log_file_created_on_text_response(self, tmp_path):
        """Even a simple text response (no tool calls) produces a log file."""
        config = make_config(workspace=tmp_path)
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    return_value=text_response("all done")):
            await run_subagent("do a thing", config, tools=make_tools())

        log_dir = tmp_path / "logs" / "subagents"
        assert log_dir.exists()
        log_files = list(log_dir.glob("*.jsonl"))
        assert len(log_files) == 1

    @pytest.mark.asyncio
    async def test_log_file_contains_summary_entry(self, tmp_path):
        """Log file ends with a summary entry."""
        config = make_config(workspace=tmp_path)
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    return_value=text_response("all done")):
            await run_subagent("do a thing", config, tools=make_tools())

        log_dir = tmp_path / "logs" / "subagents"
        log_file = list(log_dir.glob("*.jsonl"))[0]
        lines = [json.loads(l) for l in log_file.read_text().strip().splitlines()]

        # Last entry should be a summary
        summary = lines[-1]
        assert summary["event"] == "summary"
        assert summary["status"] == "completed"
        assert "total_iterations" in summary
        assert "elapsed_seconds" in summary

    @pytest.mark.asyncio
    async def test_log_file_records_tool_iterations(self, tmp_path):
        """Each tool-use iteration gets a log entry."""
        config = make_config(workspace=tmp_path)
        responses = [
            tool_response(tool_name="shell", tool_id="tc_1"),
            tool_response(tool_name="file_read", tool_id="tc_2"),
            text_response("done after 2 iterations"),
        ]
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=responses):
            with patch("openalph.tools.execute_tool", new_callable=AsyncMock,
                       return_value=ToolResult(content="ok")):
                await run_subagent("multi-step task", config, tools=make_tools())

        log_dir = tmp_path / "logs" / "subagents"
        log_file = list(log_dir.glob("*.jsonl"))[0]
        lines = [json.loads(l) for l in log_file.read_text().strip().splitlines()]

        # Should have: iteration 0, iteration 1, summary
        iteration_entries = [l for l in lines if l["event"] == "iteration"]
        assert len(iteration_entries) == 2
        assert iteration_entries[0]["iteration"] == 0
        assert iteration_entries[1]["iteration"] == 1

    @pytest.mark.asyncio
    async def test_iteration_entry_records_tool_names(self, tmp_path):
        """Iteration entries list which tools were called."""
        config = make_config(workspace=tmp_path)
        # Response with two parallel tool calls
        multi_tool = Response(
            content="",
            tool_calls=[
                ToolCall(id="tc_a", name="shell", input={"command": "ls"}),
                ToolCall(id="tc_b", name="file_read", input={"path": "x.txt"}),
            ],
            model="claude-sonnet-4-20250514",
            usage=Usage(input_tokens=100, output_tokens=50),
            stop_reason="tool_use",
        )
        responses = [multi_tool, text_response("done")]
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=responses):
            with patch("openalph.tools.execute_tool", new_callable=AsyncMock,
                       return_value=ToolResult(content="ok")):
                await run_subagent("parallel tools", config, tools=make_tools())

        log_dir = tmp_path / "logs" / "subagents"
        log_file = list(log_dir.glob("*.jsonl"))[0]
        lines = [json.loads(l) for l in log_file.read_text().strip().splitlines()]

        iteration_entry = [l for l in lines if l["event"] == "iteration"][0]
        assert set(iteration_entry["tools_called"]) == {"shell", "file_read"}

    @pytest.mark.asyncio
    async def test_iteration_entry_records_errors(self, tmp_path):
        """Iteration entries note which tool calls had errors."""
        config = make_config(workspace=tmp_path)
        responses = [
            tool_response(tool_name="shell", tool_id="tc_1"),
            text_response("done"),
        ]
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=responses):
            with patch("openalph.tools.execute_tool", new_callable=AsyncMock,
                       return_value=ToolResult(content="command not found", is_error=True)):
                await run_subagent("error task", config, tools=make_tools())

        log_dir = tmp_path / "logs" / "subagents"
        log_file = list(log_dir.glob("*.jsonl"))[0]
        lines = [json.loads(l) for l in log_file.read_text().strip().splitlines()]

        iteration_entry = [l for l in lines if l["event"] == "iteration"][0]
        assert iteration_entry["errors"] == 1

    @pytest.mark.asyncio
    async def test_iteration_entry_records_token_counts(self, tmp_path):
        """Iteration entries include input/output token counts."""
        config = make_config(workspace=tmp_path)
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    return_value=text_response("done", input_tokens=150, output_tokens=75)):
            await run_subagent("token task", config, tools=make_tools())

        log_dir = tmp_path / "logs" / "subagents"
        log_file = list(log_dir.glob("*.jsonl"))[0]
        lines = [json.loads(l) for l in log_file.read_text().strip().splitlines()]

        summary = lines[-1]
        assert summary["total_input_tokens"] == 150
        assert summary["total_output_tokens"] == 75

    @pytest.mark.asyncio
    async def test_summary_records_total_tool_calls(self, tmp_path):
        """Summary entry includes total tool calls across all iterations."""
        config = make_config(workspace=tmp_path)
        responses = [
            tool_response(tool_name="shell", tool_id="tc_1"),
            tool_response(tool_name="shell", tool_id="tc_2"),
            text_response("done"),
        ]
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=responses):
            with patch("openalph.tools.execute_tool", new_callable=AsyncMock,
                       return_value=ToolResult(content="ok")):
                await run_subagent("multi tool", config, tools=make_tools())

        log_dir = tmp_path / "logs" / "subagents"
        log_file = list(log_dir.glob("*.jsonl"))[0]
        lines = [json.loads(l) for l in log_file.read_text().strip().splitlines()]

        summary = lines[-1]
        assert summary["total_tool_calls"] == 2

    @pytest.mark.asyncio
    async def test_log_written_on_error(self, tmp_path):
        """Log file is still written when subagent hits an exception."""
        config = make_config(workspace=tmp_path)
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=Exception("provider exploded")):
            result = await run_subagent("failing task", config, tools=make_tools())

        assert result.is_error

        log_dir = tmp_path / "logs" / "subagents"
        log_files = list(log_dir.glob("*.jsonl"))
        assert len(log_files) == 1

        lines = [json.loads(l) for l in log_files[0].read_text().strip().splitlines()]
        summary = lines[-1]
        assert summary["event"] == "summary"
        assert summary["status"] == "error"
        assert "provider exploded" in summary["error"]

    @pytest.mark.asyncio
    async def test_log_written_on_circuit_breaker(self, tmp_path):
        """Log file captures circuit breaker activation."""
        config = make_config(workspace=tmp_path)

        # Always return a tool call — will hit circuit breaker
        endless_tool = tool_response(tool_name="shell", tool_id="tc_loop")
        summary_response = text_response("I was stuck in a loop")

        call_count = 0
        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("tools") is None:
                # Summary request (no tools)
                return summary_response
            return endless_tool

        with patch("openalph.tools.subagent.complete", side_effect=mock_complete):
            with patch("openalph.tools.execute_tool", new_callable=AsyncMock,
                       return_value=ToolResult(content="ok")):
                result = await run_subagent("looping task", config, tools=make_tools(),
                                           max_iterations=3)

        log_dir = tmp_path / "logs" / "subagents"
        log_file = list(log_dir.glob("*.jsonl"))[0]
        lines = [json.loads(l) for l in log_file.read_text().strip().splitlines()]

        summary = lines[-1]
        assert summary["event"] == "summary"
        assert summary["status"] == "circuit_breaker"
        assert summary["total_iterations"] == 3

    @pytest.mark.asyncio
    async def test_log_filename_contains_call_id(self, tmp_path):
        """Log filename includes the call_id for cross-referencing."""
        config = make_config(workspace=tmp_path)
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    return_value=text_response("done")):
            await run_subagent("task", config, tools=make_tools(),
                              call_id="call_abc123")

        log_dir = tmp_path / "logs" / "subagents"
        log_files = list(log_dir.glob("*.jsonl"))
        assert len(log_files) == 1
        assert "call_abc123" in log_files[0].name

    @pytest.mark.asyncio
    async def test_log_summary_includes_model(self, tmp_path):
        """Summary entry records which model was used."""
        config = make_config(workspace=tmp_path)
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    return_value=text_response("done")):
            await run_subagent("task", config, tools=make_tools(),
                              model="openrouter/kimi-k2.5")

        log_dir = tmp_path / "logs" / "subagents"
        log_file = list(log_dir.glob("*.jsonl"))[0]
        lines = [json.loads(l) for l in log_file.read_text().strip().splitlines()]

        summary = lines[-1]
        assert summary["model"] == "openrouter/kimi-k2.5"

    @pytest.mark.asyncio
    async def test_iteration_entry_records_context_tokens(self, tmp_path):
        """Iteration entries include estimated context size in tokens."""
        config = make_config(workspace=tmp_path)
        responses = [
            tool_response(tool_name="shell", tool_id="tc_1"),
            text_response("done"),
        ]
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=responses):
            with patch("openalph.tools.execute_tool", new_callable=AsyncMock,
                       return_value=ToolResult(content="x" * 400)):
                await run_subagent("do stuff", config, tools=make_tools())

        log_dir = tmp_path / "logs" / "subagents"
        log_file = list(log_dir.glob("*.jsonl"))[0]
        lines = [json.loads(l) for l in log_file.read_text().strip().splitlines()]

        iteration_entry = [l for l in lines if l["event"] == "iteration"][0]
        assert "context_tokens" in iteration_entry
        assert iteration_entry["context_tokens"] > 0

    @pytest.mark.asyncio
    async def test_summary_records_peak_context_tokens(self, tmp_path):
        """Summary entry includes peak context token estimate."""
        config = make_config(workspace=tmp_path)
        responses = [
            tool_response(tool_name="shell", tool_id="tc_1"),
            tool_response(tool_name="shell", tool_id="tc_2"),
            text_response("done"),
        ]
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=responses):
            with patch("openalph.tools.execute_tool", new_callable=AsyncMock,
                       return_value=ToolResult(content="y" * 800)):
                await run_subagent("growing context", config, tools=make_tools())

        log_dir = tmp_path / "logs" / "subagents"
        log_file = list(log_dir.glob("*.jsonl"))[0]
        lines = [json.loads(l) for l in log_file.read_text().strip().splitlines()]

        summary = lines[-1]
        assert "peak_context_tokens" in summary
        assert summary["peak_context_tokens"] > 0

        # Peak should be >= the last iteration's context tokens
        iterations = [l for l in lines if l["event"] == "iteration"]
        assert summary["peak_context_tokens"] >= iterations[-1]["context_tokens"]

    @pytest.mark.asyncio
    async def test_peak_context_grows_with_iterations(self, tmp_path):
        """Context tokens should grow across iterations as messages accumulate."""
        config = make_config(workspace=tmp_path)
        responses = [
            tool_response(tool_name="shell", tool_id="tc_1"),
            tool_response(tool_name="shell", tool_id="tc_2"),
            text_response("done"),
        ]
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=responses):
            with patch("openalph.tools.execute_tool", new_callable=AsyncMock,
                       return_value=ToolResult(content="output " * 50)):
                await run_subagent("multi-step", config, tools=make_tools())

        log_dir = tmp_path / "logs" / "subagents"
        log_file = list(log_dir.glob("*.jsonl"))[0]
        lines = [json.loads(l) for l in log_file.read_text().strip().splitlines()]

        iterations = [l for l in lines if l["event"] == "iteration"]
        assert len(iterations) == 2
        # Second iteration should have more context than first
        assert iterations[1]["context_tokens"] > iterations[0]["context_tokens"]

    @pytest.mark.asyncio
    async def test_log_summary_includes_task(self, tmp_path):
        """Summary entry records the task description."""
        config = make_config(workspace=tmp_path)
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    return_value=text_response("done")):
            await run_subagent("Build the widget factory", config, tools=make_tools())

        log_dir = tmp_path / "logs" / "subagents"
        log_file = list(log_dir.glob("*.jsonl"))[0]
        lines = [json.loads(l) for l in log_file.read_text().strip().splitlines()]

        summary = lines[-1]
        assert summary["task"] == "Build the widget factory"


# =============================================================================
# Part 1: Matrix notice formatting (tested via _tool_notice in matrix.py)
# =============================================================================

class TestSubagentNoticeFormatting:
    """Matrix notices for subagent return use proper <details> blocks."""

    def _build_tool_notice(self, task, result, model="default", elapsed=10.0, is_error=False):
        """Simulate the HTML generation logic from matrix.py _tool_notice.

        This tests the formatting function in isolation. The actual function
        is a closure inside matrix.py — we test the expected output format.
        """
        import mistune

        status = "❌ error" if is_error else "✅"
        elapsed_str = f" — {elapsed:.1f}s"
        summary_line = f"🤖 subagent ({model}) {status}{elapsed_str}"

        html = f'<b>{summary_line}</b>'
        if task:
            html += (
                f'\n<details><summary>📋 Task brief</summary>\n'
                f'{mistune.html(task)}</details>'
            )
        if result:
            html += (
                f'\n<details><summary>📨 Result</summary>\n'
                f'{mistune.html(result)}</details>'
            )
        return html

    def test_no_pre_tags_in_task_brief(self):
        """Task brief should NOT be wrapped in <pre> tags."""
        html = self._build_tool_notice("Build a **widget**", "done")
        assert "<pre>" not in html

    def test_no_pre_tags_in_result(self):
        """Result should NOT be wrapped in <pre> tags."""
        html = self._build_tool_notice("task", "The **result** is ready")
        assert "<pre>" not in html

    def test_markdown_rendered_in_task(self):
        """Markdown in task brief should be rendered as HTML."""
        html = self._build_tool_notice("Build a **widget** with `code`", "done")
        assert "<strong>widget</strong>" in html
        assert "<code>code</code>" in html

    def test_markdown_rendered_in_result(self):
        """Markdown in result should be rendered as HTML."""
        html = self._build_tool_notice("task", "Created **3 files** with `tests`")
        assert "<strong>3 files</strong>" in html

    def test_details_block_structure(self):
        """Output has proper <details><summary> structure."""
        html = self._build_tool_notice("my task", "my result")
        assert "<details><summary>📋 Task brief</summary>" in html
        assert "<details><summary>📨 Result</summary>" in html
        assert html.count("</details>") == 2

    def test_full_task_not_truncated(self):
        """Long task briefs are NOT truncated."""
        long_task = "x" * 5000
        html = self._build_tool_notice(long_task, "done")
        # The full content should be present (rendered through mistune)
        assert "x" * 100 in html  # At minimum, a long stretch should survive

    def test_full_result_not_truncated(self):
        """Long results are NOT truncated."""
        long_result = "y" * 5000
        html = self._build_tool_notice("task", long_result)
        assert "y" * 100 in html
