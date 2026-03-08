"""Tests for agent loop with tool use integration.

Extends agent.py handle_input() to:
    - Discover tools from workspace on init
    - Execute tool_use responses: run tools → feed results → continue
    - Parallel execution of multiple tool calls
    - Error results fed back to LLM
    - Circuit breaker at max_iterations
    - Truncation applied to tool results
    - Status includes tool metrics

Mocks both provider (LLM responses) and tool executors.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path
from openalph.agent import Agent
from openalph.config import AgentConfig
from openalph.provider import Response, Usage, ToolCall
from openalph.tools import ToolDef, ToolResult


# --- Fixtures ---


def make_config(workspace=None, **kwargs):
    defaults = dict(
        name="test-agent",
        model="test-model",
        max_tokens=8192,
        provider="anthropic",
        api_key="sk-test",
        base_url=None,
        workspace=workspace or Path("/tmp/test-workspace"),
        max_iterations=25,
        truncation_limit=50000,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def text_response(text, input_tokens=100, output_tokens=50):
    """Simulate a normal text response (no tool calls)."""
    return Response(
        content=text,
        tool_calls=[],
        model="test-model",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
    )


def tool_use_response(tool_calls, text="", input_tokens=100, output_tokens=50):
    """Simulate a response with tool_use blocks."""
    return Response(
        content=text,
        tool_calls=tool_calls,
        model="test-model",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="tool_use",
    )


SHELL_TOOL = ToolDef(
    name="shell",
    description="Execute a shell command",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
        },
        "required": ["command"],
    },
    config={"default_timeout": 30},
)


# --- Tool discovery on init ---


class TestToolDiscovery:

    def test_agent_discovers_tools_from_workspace(self, tmp_path):
        """Agent loads tools from workspace/tools/ directory on init."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path)

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        assert len(agent.tools) >= 1
        assert any(t.name == "shell" for t in agent.tools)

    def test_agent_no_tools_directory(self, tmp_path):
        """No tools/ directory → agent has empty tools list."""
        config = make_config(workspace=tmp_path)

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        assert agent.tools == []


# --- Basic tool use flow ---


class TestToolUseFlow:

    @pytest.mark.asyncio
    async def test_single_tool_call_and_continue(self, tmp_path):
        """Agent executes tool, feeds result back, gets final text response."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path)

        # LLM returns: tool_use → (after result) → text
        responses = [
            tool_use_response([
                ToolCall(id="tc_1", name="shell", input={"command": "echo hello"})
            ]),
            text_response("The output was: hello"),
        ]

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        with patch("openalph.agent.complete", new_callable=AsyncMock, side_effect=responses):
            with patch(
                "openalph.agent.execute_tool",
                new_callable=AsyncMock,
                return_value=ToolResult(content="hello"),
            ):
                result = await agent.handle_input("Run echo hello")

        assert "hello" in result

    @pytest.mark.asyncio
    async def test_tool_result_fed_back_to_llm(self, tmp_path):
        """Tool execution result is included in the next LLM call."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path)

        responses = [
            tool_use_response([
                ToolCall(id="tc_1", name="shell", input={"command": "pwd"})
            ]),
            text_response("You're in /home"),
        ]

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        with patch("openalph.agent.complete", new_callable=AsyncMock, side_effect=responses) as mock_complete:
            with patch(
                "openalph.agent.execute_tool",
                new_callable=AsyncMock,
                return_value=ToolResult(content="/home/test"),
            ):
                await agent.handle_input("Where am I?")

        # Second call should have tool result in messages
        second_call_messages = mock_complete.call_args_list[1].kwargs["messages"]
        # Should contain a tool result message
        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "/home/test" in tool_msgs[0]["content"]


# --- Parallel tool execution ---


class TestParallelExecution:

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_executed(self, tmp_path):
        """Multiple tool_use blocks in one response → all executed."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path)

        responses = [
            tool_use_response([
                ToolCall(id="tc_1", name="shell", input={"command": "echo a"}),
                ToolCall(id="tc_2", name="shell", input={"command": "echo b"}),
            ]),
            text_response("Done: a and b"),
        ]

        call_count = 0

        async def mock_execute(name, input, tool_config, agent_config):
            nonlocal call_count
            call_count += 1
            cmd = input.get("command", "")
            return ToolResult(content=f"output of {cmd}")

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        with patch("openalph.agent.complete", new_callable=AsyncMock, side_effect=responses):
            with patch("openalph.agent.execute_tool", side_effect=mock_execute):
                result = await agent.handle_input("Run two commands")

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_all_results_returned_to_llm(self, tmp_path):
        """Results from parallel tool calls are all included in next LLM call."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path)

        responses = [
            tool_use_response([
                ToolCall(id="tc_1", name="shell", input={"command": "echo a"}),
                ToolCall(id="tc_2", name="shell", input={"command": "echo b"}),
            ]),
            text_response("Both done"),
        ]

        async def mock_execute(name, input, tool_config, agent_config):
            cmd = input.get("command", "")
            return ToolResult(content=f"result_{cmd[-1]}")

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        with patch("openalph.agent.complete", new_callable=AsyncMock, side_effect=responses) as mock_complete:
            with patch("openalph.agent.execute_tool", side_effect=mock_execute):
                await agent.handle_input("Two things")

        second_call_messages = mock_complete.call_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2


# --- Error handling ---


class TestToolErrors:

    @pytest.mark.asyncio
    async def test_tool_error_fed_back(self, tmp_path):
        """Tool error result (is_error=True) is fed back to LLM."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path)

        responses = [
            tool_use_response([
                ToolCall(id="tc_1", name="shell", input={"command": "bad_cmd"})
            ]),
            text_response("That command failed, let me try something else."),
        ]

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        with patch("openalph.agent.complete", new_callable=AsyncMock, side_effect=responses) as mock_complete:
            with patch(
                "openalph.agent.execute_tool",
                new_callable=AsyncMock,
                return_value=ToolResult(content="command not found", is_error=True),
            ):
                result = await agent.handle_input("Run bad command")

        # Error should be in the tool result message
        second_call_messages = mock_complete.call_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].get("is_error") is True


# --- Circuit breaker ---


class TestCircuitBreaker:

    @pytest.mark.asyncio
    async def test_max_iterations_stops_loop(self, tmp_path):
        """Agent stops after max_iterations tool calls."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path, max_iterations=3)

        # LLM always returns tool_use (infinite loop without breaker)
        infinite_tool_use = tool_use_response([
            ToolCall(id="tc_x", name="shell", input={"command": "loop"})
        ])

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        with patch(
            "openalph.agent.complete",
            new_callable=AsyncMock,
            return_value=infinite_tool_use,
        ):
            with patch(
                "openalph.agent.execute_tool",
                new_callable=AsyncMock,
                return_value=ToolResult(content="still going"),
            ):
                result = await agent.handle_input("Do something in a loop")

        # Should have stopped and returned a message about the limit
        assert "limit" in result.lower() or "iteration" in result.lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_respects_config(self, tmp_path):
        """max_iterations from config controls the limit."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path, max_iterations=2)

        infinite_tool_use = tool_use_response([
            ToolCall(id="tc_x", name="shell", input={"command": "loop"})
        ])

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        call_count = 0

        with patch("openalph.agent.complete", new_callable=AsyncMock, return_value=infinite_tool_use) as mock_complete:
            with patch(
                "openalph.agent.execute_tool",
                new_callable=AsyncMock,
                return_value=ToolResult(content="loop"),
            ):
                await agent.handle_input("Loop forever")
                call_count = mock_complete.await_count

        # Should have called complete at most max_iterations times
        assert call_count <= 2


# --- Truncation ---


class TestToolResultTruncation:

    @pytest.mark.asyncio
    async def test_large_result_truncated(self, tmp_path):
        """Tool results exceeding truncation_limit are truncated."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path, truncation_limit=100)

        responses = [
            tool_use_response([
                ToolCall(id="tc_1", name="shell", input={"command": "cat bigfile"})
            ]),
            text_response("Here's a summary of the file."),
        ]

        big_output = "x" * 10000

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        with patch("openalph.agent.complete", new_callable=AsyncMock, side_effect=responses) as mock_complete:
            with patch(
                "openalph.agent.execute_tool",
                new_callable=AsyncMock,
                return_value=ToolResult(content=big_output),
            ):
                await agent.handle_input("Show me the big file")

        # The tool result in the second call should be truncated
        second_call_messages = mock_complete.call_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        # Content should be much smaller than 10000
        assert len(tool_msgs[0]["content"]) < 1000

    @pytest.mark.asyncio
    async def test_small_result_not_truncated(self, tmp_path):
        """Tool results under truncation_limit pass through unchanged."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path, truncation_limit=50000)

        responses = [
            tool_use_response([
                ToolCall(id="tc_1", name="shell", input={"command": "echo hi"})
            ]),
            text_response("OK"),
        ]

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        with patch("openalph.agent.complete", new_callable=AsyncMock, side_effect=responses) as mock_complete:
            with patch(
                "openalph.agent.execute_tool",
                new_callable=AsyncMock,
                return_value=ToolResult(content="hi"),
            ):
                await agent.handle_input("echo")

        second_call_messages = mock_complete.call_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == "hi"


# --- Status tracking ---


class TestStatusWithTools:

    @pytest.mark.asyncio
    async def test_status_includes_tool_calls_count(self, tmp_path):
        """status() includes total tool call count."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path)

        responses = [
            tool_use_response([
                ToolCall(id="tc_1", name="shell", input={"command": "echo a"}),
                ToolCall(id="tc_2", name="shell", input={"command": "echo b"}),
            ]),
            text_response("Done"),
        ]

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        with patch("openalph.agent.complete", new_callable=AsyncMock, side_effect=responses):
            with patch(
                "openalph.agent.execute_tool",
                new_callable=AsyncMock,
                return_value=ToolResult(content="ok"),
            ):
                await agent.handle_input("Do two things")

        status = agent.status()
        assert "total_tool_calls" in status
        assert status["total_tool_calls"] == 2


# --- No tools → backward compatible ---


class TestNoToolsBackwardCompat:

    @pytest.mark.asyncio
    async def test_text_only_conversation(self, tmp_path):
        """Without tools, handle_input works exactly as Phase 1."""
        config = make_config(workspace=tmp_path)

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        with patch(
            "openalph.agent.complete",
            new_callable=AsyncMock,
            return_value=text_response("Hello there!"),
        ):
            result = await agent.handle_input("Hi")

        assert result == "Hello there!"

    @pytest.mark.asyncio
    async def test_history_maintained(self, tmp_path):
        """Conversation history works correctly with tool use interspersed."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path)

        # Turn 1: normal text
        # Turn 2: tool use → result → text
        responses_turn1 = text_response("Hi!")
        responses_turn2 = [
            tool_use_response([
                ToolCall(id="tc_1", name="shell", input={"command": "date"})
            ]),
            text_response("Today is March 8"),
        ]

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        # Turn 1
        with patch("openalph.agent.complete", new_callable=AsyncMock, return_value=responses_turn1):
            await agent.handle_input("Hello")

        # Turn 2
        with patch("openalph.agent.complete", new_callable=AsyncMock, side_effect=responses_turn2) as mock_complete:
            with patch(
                "openalph.agent.execute_tool",
                new_callable=AsyncMock,
                return_value=ToolResult(content="Sun Mar  8 10:00:00 EDT 2026"),
            ):
                result = await agent.handle_input("What day is it?")

        assert "March 8" in result

        # History should contain turn 1 + turn 2 (including tool interactions)
        status = agent.status()
        assert status["turns"] == 2


# --- Tools passed to provider ---


class TestToolsPassedToProvider:

    @pytest.mark.asyncio
    async def test_tools_included_in_complete_call(self, tmp_path):
        """Agent passes discovered tools to provider.complete()."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        config = make_config(workspace=tmp_path)

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        with patch(
            "openalph.agent.complete",
            new_callable=AsyncMock,
            return_value=text_response("Hi"),
        ) as mock_complete:
            await agent.handle_input("Hello")

        call_kwargs = mock_complete.call_args.kwargs
        assert "tools" in call_kwargs
        # Should pass the discovered ToolDef list
        tools = call_kwargs["tools"]
        assert len(tools) >= 1

    @pytest.mark.asyncio
    async def test_no_tools_passes_none(self, tmp_path):
        """Agent with no tools passes tools=None (or empty) to provider."""
        config = make_config(workspace=tmp_path)

        with patch("openalph.agent.assemble_prompt", return_value="system prompt"):
            agent = Agent(config)

        with patch(
            "openalph.agent.complete",
            new_callable=AsyncMock,
            return_value=text_response("Hi"),
        ) as mock_complete:
            await agent.handle_input("Hello")

        call_kwargs = mock_complete.call_args.kwargs
        tools = call_kwargs.get("tools")
        assert tools is None or tools == []
