"""Tests for agent streaming callbacks (Phase B of kdsn.65).

Contract:
    handle_input() gains on_text_delta and on_thinking_delta callbacks.
    Tool loop uses stream() instead of complete().

    on_text_delta(text: str, done: bool) -> None
        - text: partial content delta
        - done=True signals end of text for this LLM turn
        - done=True fires ONLY if text was produced in this turn
        - Fires once per stream iteration that produces text

    on_thinking_delta(text: str, done: bool) -> None
        - text: partial thinking delta
        - done=True signals end of thinking for this turn

    handle_input() still returns the complete response string.
    Token tracking, tool execution, circuit breaker, context overflow all unchanged.
    When no callbacks provided, behavior is identical to previous implementation.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path

from openalph.agent import Agent, ContextOverflowError
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import StreamEvent, Response, Usage, ToolCall, ThinkingBlock
from openalph.tools import ToolDef, ToolResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_provider(key="anthropic", type="anthropic", api_key="sk-test",
                  base_url=None, quirks=None):
    return ProviderConfig(
        key=key, type=type, api_key=api_key,
        base_url=base_url, quirks=quirks or [],
    )


def make_config(workspace, **kwargs):
    defaults = {
        "name": "test-agent",
        "default_model": "anthropic/claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "providers": {"anthropic": make_provider()},
        "max_iterations": 25,
        "truncation_limit": 50000,
    }
    defaults.update(kwargs)
    defaults["workspace"] = workspace
    return AgentConfig(**defaults)


def make_response(content="", input_tokens=100, output_tokens=50,
                  stop_reason="end_turn", tool_calls=None, thinking=None):
    return Response(
        content=content,
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
        tool_calls=tool_calls or [],
        thinking=thinking or [],
    )


SHELL_TOOL = ToolDef(
    name="shell",
    description="Execute a shell command",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
    config={"default_timeout": 30},
)


# ---------------------------------------------------------------------------
# Stream mock helpers
# ---------------------------------------------------------------------------

def make_stream_fn(calls):
    """Create a fake stream() that yields different events per invocation.

    Args:
        calls: list of lists of StreamEvent. Each inner list is one
               stream() invocation's events. Consumed in order.

    Usage:
        with patch("openalph.agent.stream", new=make_stream_fn([
            [StreamEvent(type="text", content="Hi"),
             StreamEvent(type="done", response=..., stop_reason="end_turn")],
        ])):
    """
    call_iter = iter(calls)

    async def _fake_stream(*args, **kwargs):
        events = next(call_iter)
        for event in events:
            yield event

    return _fake_stream


def text_events(text, input_tokens=100, output_tokens=50, chunks=None):
    """Build stream events for a simple text response.

    Args:
        text: full response text
        chunks: optional list of text chunk strings (defaults to [text])
    """
    parts = chunks or [text]
    events = [StreamEvent(type="text", content=c) for c in parts]
    events.append(StreamEvent(
        type="done",
        response=make_response(content=text, input_tokens=input_tokens,
                               output_tokens=output_tokens),
        stop_reason="end_turn",
        model="claude-sonnet-4-20250514",
    ))
    return events


def tool_events(text_before, tool_calls, input_tokens=100, output_tokens=50):
    """Build stream events for a tool-use response.

    Args:
        text_before: text emitted before tool calls (can be "")
        tool_calls: list of ToolCall objects
    """
    events = []
    if text_before:
        events.append(StreamEvent(type="text", content=text_before))
    for i, tc in enumerate(tool_calls):
        events.append(StreamEvent(
            type="tool_done", tool_index=i, tool_call=tc,
        ))
    events.append(StreamEvent(
        type="done",
        response=make_response(
            content=text_before, tool_calls=tool_calls,
            input_tokens=input_tokens, output_tokens=output_tokens,
            stop_reason="tool_use",
        ),
        stop_reason="tool_use",
        model="claude-sonnet-4-20250514",
    ))
    return events


def thinking_then_text_events(thinking_text, text, thinking_signature="sig-test",
                               input_tokens=100, output_tokens=50):
    """Build stream events for a thinking + text response."""
    events = [
        StreamEvent(type="thinking", content=thinking_text),
        StreamEvent(type="signature", content=thinking_signature),
        StreamEvent(type="text", content=text),
        StreamEvent(
            type="done",
            response=make_response(
                content=text, input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking=[ThinkingBlock(thinking=thinking_text,
                                        signature=thinking_signature)],
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-20250514",
        ),
    ]
    return events


# ---------------------------------------------------------------------------
# Tests: on_text_delta callback
# ---------------------------------------------------------------------------

class TestOnTextDelta:

    @pytest.mark.asyncio
    async def test_called_with_text_chunks(self, tmp_path):
        """on_text_delta fires for each text event."""
        config = make_config(tmp_path)
        agent = Agent(config)

        cb = AsyncMock()

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        text_events("Hello world!", chunks=["Hello ", "world!"]),
                    ])):
            result = await agent.handle_input("Hi", on_text_delta=cb)

        assert result == "Hello world!"

        # Should be called: ("Hello ", False), ("world!", False), ("", True)
        assert cb.call_count == 3
        cb.assert_any_call("Hello ", done=False)
        cb.assert_any_call("world!", done=False)
        cb.assert_any_call("", done=True)

    @pytest.mark.asyncio
    async def test_done_signal_at_end(self, tmp_path):
        """done=True is the last call after all text deltas."""
        config = make_config(tmp_path)
        agent = Agent(config)

        cb = AsyncMock()

        with patch("openalph.agent.stream",
                    new=make_stream_fn([text_events("Hi")])):
            await agent.handle_input("Hi", on_text_delta=cb)

        # Last call should be done=True
        last_call = cb.call_args_list[-1]
        assert last_call == call("", done=True)

    @pytest.mark.asyncio
    async def test_not_called_when_none(self, tmp_path):
        """No error when on_text_delta is not provided."""
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream",
                    new=make_stream_fn([text_events("Hello")])):
            result = await agent.handle_input("Hi")

        assert result == "Hello"

    @pytest.mark.asyncio
    async def test_done_not_fired_when_no_text_before_tools(self, tmp_path):
        """done=True is NOT fired if no text was produced before tool calls."""
        config = make_config(tmp_path)
        agent = Agent(config)

        cb = AsyncMock()
        tc = ToolCall(id="tc1", name="shell", input={"command": "ls"})

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        tool_events("", [tc]),           # no text, just tool
                        text_events("Files listed."),    # text response after tool
                    ])), \
             patch("openalph.agent.discover_tools", return_value=[SHELL_TOOL]), \
             patch("openalph.agent.execute_tool",
                   new_callable=AsyncMock,
                   return_value=ToolResult(content="file.txt", is_error=False)):

            result = await agent.handle_input("List", on_text_delta=cb)

        # First stream: no text → no done=True for that iteration
        # Second stream: text + done=True
        text_calls = [c for c in cb.call_args_list if c.kwargs.get("done") is not True
                      or c == call("", done=True)]
        done_calls = [c for c in cb.call_args_list if c == call("", done=True)]
        assert len(done_calls) == 1  # Only from the second stream
        assert result == "Files listed."

    @pytest.mark.asyncio
    async def test_done_fired_for_text_before_tools(self, tmp_path):
        """done=True IS fired when text precedes tool calls."""
        config = make_config(tmp_path)
        agent = Agent(config)

        cb = AsyncMock()
        tc = ToolCall(id="tc1", name="shell", input={"command": "ls"})

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        tool_events("Let me check.", [tc]),  # text then tool
                        text_events("Done."),                 # final response
                    ])), \
             patch("openalph.agent.discover_tools", return_value=[SHELL_TOOL]), \
             patch("openalph.agent.execute_tool",
                   new_callable=AsyncMock,
                   return_value=ToolResult(content="result", is_error=False)):

            await agent.handle_input("Check", on_text_delta=cb)

        # Two done=True signals: one after "Let me check.", one after "Done."
        done_calls = [c for c in cb.call_args_list if c == call("", done=True)]
        assert len(done_calls) == 2


# ---------------------------------------------------------------------------
# Tests: on_thinking_delta callback
# ---------------------------------------------------------------------------

class TestOnThinkingDelta:

    @pytest.mark.asyncio
    async def test_called_with_thinking_chunks(self, tmp_path):
        """on_thinking_delta fires for thinking events."""
        config = make_config(tmp_path)
        agent = Agent(config)

        thinking_cb = AsyncMock()

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        thinking_then_text_events("Deep thought", "42"),
                    ])):
            result = await agent.handle_input(
                "Think", on_thinking_delta=thinking_cb, thinking="high",
            )

        assert result == "42"
        thinking_cb.assert_any_call("Deep thought", done=False)
        # done=True should fire after thinking completes
        thinking_cb.assert_any_call("", done=True)

    @pytest.mark.asyncio
    async def test_not_called_when_none(self, tmp_path):
        """No error when on_thinking_delta is not provided."""
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        thinking_then_text_events("thoughts", "answer"),
                    ])):
            result = await agent.handle_input("Think", thinking="high")

        assert result == "answer"

    @pytest.mark.asyncio
    async def test_both_callbacks_independent(self, tmp_path):
        """on_text_delta and on_thinking_delta fire independently."""
        config = make_config(tmp_path)
        agent = Agent(config)

        text_cb = AsyncMock()
        thinking_cb = AsyncMock()

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        thinking_then_text_events("reasoning", "answer"),
                    ])):
            await agent.handle_input(
                "Think",
                on_text_delta=text_cb,
                on_thinking_delta=thinking_cb,
                thinking="high",
            )

        # text_cb gets text events
        text_cb.assert_any_call("answer", done=False)
        # thinking_cb gets thinking events
        thinking_cb.assert_any_call("reasoning", done=False)


# ---------------------------------------------------------------------------
# Tests: Tool loop with streaming
# ---------------------------------------------------------------------------

class TestToolLoopStreaming:

    @pytest.mark.asyncio
    async def test_single_tool_call(self, tmp_path):
        """Tool call → execute → next stream → return text."""
        config = make_config(tmp_path)
        agent = Agent(config)
        tc = ToolCall(id="tc1", name="shell", input={"command": "ls"})

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        tool_events("Checking.", [tc]),
                        text_events("Found 3 files."),
                    ])), \
             patch("openalph.agent.discover_tools", return_value=[SHELL_TOOL]), \
             patch("openalph.agent.execute_tool",
                   new_callable=AsyncMock,
                   return_value=ToolResult(content="a.txt\nb.txt\nc.txt", is_error=False)):

            result = await agent.handle_input("List files")

        assert result == "Found 3 files."

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self, tmp_path):
        """Multiple tool calls execute in parallel."""
        config = make_config(tmp_path)
        agent = Agent(config)

        tc1 = ToolCall(id="tc1", name="shell", input={"command": "ls"})
        tc2 = ToolCall(id="tc2", name="shell", input={"command": "pwd"})

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        tool_events("Running commands.", [tc1, tc2]),
                        text_events("Done."),
                    ])), \
             patch("openalph.agent.discover_tools", return_value=[SHELL_TOOL]), \
             patch("openalph.agent.execute_tool",
                   new_callable=AsyncMock,
                   return_value=ToolResult(content="output", is_error=False)) as mock_exec:

            await agent.handle_input("Run both")

        # Both tools should be executed
        assert mock_exec.call_count == 2

    @pytest.mark.asyncio
    async def test_multi_iteration_tool_loop(self, tmp_path):
        """Multiple rounds of tool use before final text response."""
        config = make_config(tmp_path)
        agent = Agent(config)

        tc1 = ToolCall(id="tc1", name="shell", input={"command": "ls"})
        tc2 = ToolCall(id="tc2", name="shell", input={"command": "cat file.txt"})

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        tool_events("", [tc1]),          # Round 1
                        tool_events("Reading.", [tc2]),   # Round 2
                        text_events("Contents: hello"),   # Final
                    ])), \
             patch("openalph.agent.discover_tools", return_value=[SHELL_TOOL]), \
             patch("openalph.agent.execute_tool",
                   new_callable=AsyncMock,
                   return_value=ToolResult(content="result", is_error=False)):

            result = await agent.handle_input("Read")

        assert result == "Contents: hello"

    @pytest.mark.asyncio
    async def test_tool_error_fed_back(self, tmp_path):
        """Tool error results are fed back to the LLM."""
        config = make_config(tmp_path)
        agent = Agent(config)
        tc = ToolCall(id="tc1", name="shell", input={"command": "fail"})

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        tool_events("", [tc]),
                        text_events("Command failed."),
                    ])), \
             patch("openalph.agent.discover_tools", return_value=[SHELL_TOOL]), \
             patch("openalph.agent.execute_tool",
                   new_callable=AsyncMock,
                   return_value=ToolResult(content="exit code 1", is_error=True)):

            result = await agent.handle_input("Run bad command")

        assert result == "Command failed."
        # History should contain the error tool result
        history = agent.history("_default")
        tool_msgs = [m for m in history if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["is_error"] is True

    @pytest.mark.asyncio
    async def test_on_tool_call_callback_still_works(self, tmp_path):
        """on_tool_call callback fires during streaming tool loop."""
        config = make_config(tmp_path)
        agent = Agent(config)
        tc = ToolCall(id="tc1", name="shell", input={"command": "ls"})

        tool_cb = AsyncMock()

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        tool_events("", [tc]),
                        text_events("Done."),
                    ])), \
             patch("openalph.agent.discover_tools", return_value=[SHELL_TOOL]), \
             patch("openalph.agent.execute_tool",
                   new_callable=AsyncMock,
                   return_value=ToolResult(content="output", is_error=False)):

            await agent.handle_input("List", on_tool_call=tool_cb)

        tool_cb.assert_called_once()
        args = tool_cb.call_args
        assert args[0][0] == "tc1"       # tool_call_id
        assert args[0][1] == "shell"     # tool name

    @pytest.mark.asyncio
    async def test_on_tool_intent_callback_still_works(self, tmp_path):
        """on_tool_intent callback fires before tool execution."""
        config = make_config(tmp_path)
        agent = Agent(config)
        tc = ToolCall(id="tc1", name="shell", input={"command": "ls"})

        intent_cb = AsyncMock()

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        tool_events("Checking.", [tc]),
                        text_events("Result."),
                    ])), \
             patch("openalph.agent.discover_tools", return_value=[SHELL_TOOL]), \
             patch("openalph.agent.execute_tool",
                   new_callable=AsyncMock,
                   return_value=ToolResult(content="output", is_error=False)):

            await agent.handle_input("Check", on_tool_intent=intent_cb)

        intent_cb.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    @pytest.mark.asyncio
    async def test_returns_complete_string(self, tmp_path):
        """handle_input returns the full text, not deltas."""
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        text_events("Hello world!", chunks=["Hello ", "world!"]),
                    ])):
            result = await agent.handle_input("Hi")

        assert result == "Hello world!"

    @pytest.mark.asyncio
    async def test_history_correct(self, tmp_path):
        """History contains complete messages, not deltas."""
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream",
                    new=make_stream_fn([text_events("Response")])):
            await agent.handle_input("User message")

        history = agent.history("_default")
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "User message"}
        assert history[1] == {"role": "assistant", "content": "Response"}

    @pytest.mark.asyncio
    async def test_thinking_in_history(self, tmp_path):
        """Thinking blocks are stored in history."""
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        thinking_then_text_events("deep thought", "answer", "sig-x"),
                    ])):
            await agent.handle_input("Think", thinking="high")

        history = agent.history("_default")
        assistant_msg = history[-1]
        assert assistant_msg["content"] == "answer"
        assert "thinking" in assistant_msg
        assert assistant_msg["thinking"][0]["thinking"] == "deep thought"
        assert assistant_msg["thinking"][0]["signature"] == "sig-x"

    @pytest.mark.asyncio
    async def test_token_tracking(self, tmp_path):
        """Token usage tracked from stream events."""
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        text_events("Hi", input_tokens=150, output_tokens=30),
                    ])):
            await agent.handle_input("Hello")

        assert agent.uncached_input_tokens == 150
        assert agent.total_output_tokens == 30

    @pytest.mark.asyncio
    async def test_token_tracking_across_tool_loops(self, tmp_path):
        """Token usage accumulates across tool loop iterations."""
        config = make_config(tmp_path)
        agent = Agent(config)
        tc = ToolCall(id="tc1", name="shell", input={"command": "ls"})

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        tool_events("", [tc], input_tokens=100, output_tokens=20),
                        text_events("Done.", input_tokens=200, output_tokens=40),
                    ])), \
             patch("openalph.agent.discover_tools", return_value=[SHELL_TOOL]), \
             patch("openalph.agent.execute_tool",
                   new_callable=AsyncMock,
                   return_value=ToolResult(content="out", is_error=False)):

            await agent.handle_input("Do it")

        assert agent.uncached_input_tokens == 300
        assert agent.total_output_tokens == 60

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, tmp_path):
        """max_iterations still enforced with streaming; summary is generated."""
        config = make_config(tmp_path, max_iterations=3)
        agent = Agent(config)

        tc = ToolCall(id="tc1", name="shell", input={"command": "loop"})
        # 3 tool iterations + 1 summary call (text response, tools=None)
        calls = [tool_events("", [tc]) for _ in range(3)]
        calls.append(text_events("Here is my progress summary."))

        with patch("openalph.agent.stream",
                    new=make_stream_fn(calls)), \
             patch("openalph.agent.discover_tools", return_value=[SHELL_TOOL]), \
             patch("openalph.agent.execute_tool",
                   new_callable=AsyncMock,
                   return_value=ToolResult(content="again", is_error=False)):

            result = await agent.handle_input("Loop forever")

        assert "progress" in result.lower() or "summary" in result.lower()

    @pytest.mark.asyncio
    async def test_context_overflow(self, tmp_path):
        """ContextOverflowError still raised on overflow."""
        config = make_config(tmp_path, model_max_tokens=1000, max_tokens=200)
        agent = Agent(config)

        # Pre-fill history near capacity
        agent._rooms["_default"] = [
            {"role": "user", "content": "x" * 3000},
            {"role": "assistant", "content": "y" * 100},
        ]

        with pytest.raises(ContextOverflowError):
            with patch("openalph.agent.stream",
                        new=make_stream_fn([text_events("nope")])):
                await agent.handle_input("z" * 2000)


# ---------------------------------------------------------------------------
# Tests: Room isolation with streaming
# ---------------------------------------------------------------------------

class TestRoomIsolation:

    @pytest.mark.asyncio
    async def test_different_rooms_independent(self, tmp_path):
        """Streaming callbacks per room don't interfere."""
        config = make_config(tmp_path)
        agent = Agent(config)

        cb_a = AsyncMock()
        cb_b = AsyncMock()

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        text_events("Room A response"),
                        text_events("Room B response"),
                    ])):
            await agent.handle_input("Hi A", room_id="room_a", on_text_delta=cb_a)
            await agent.handle_input("Hi B", room_id="room_b", on_text_delta=cb_b)

        # Each callback only received its room's content
        a_texts = [c.args[0] for c in cb_a.call_args_list if c.kwargs.get("done") is not True]
        b_texts = [c.args[0] for c in cb_b.call_args_list if c.kwargs.get("done") is not True]
        assert "Room A response" in "".join(a_texts)
        assert "Room B response" in "".join(b_texts)


# ---------------------------------------------------------------------------
# Tests: Status still works
# ---------------------------------------------------------------------------

class TestStatusWithStreaming:

    @pytest.mark.asyncio
    async def test_status_after_streaming_conversation(self, tmp_path):
        """Status reflects usage after streaming conversation."""
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream",
                    new=make_stream_fn([
                        text_events("Hello", input_tokens=50, output_tokens=20),
                    ])):
            await agent.handle_input("Hi")

        status = agent.status()
        assert status["turns"] == 1
        assert status["uncached_input_tokens"] == 50
        assert status["total_output_tokens"] == 20
