"""Tests for empty-response recovery when extended thinking exhausts the budget.

Background (incident 2026-06-28, bead workspace-kdsn.169):
    merry (Opus 4.8, thinking="max", max_tokens=16384) returned an empty
    response on a hard turn. For Anthropic extended thinking, max_tokens is a
    *combined* budget for thinking + output. At effort="max" the adaptive path
    let thinking consume the entire 16384-token budget (turn log showed
    output_tokens == 16384 exactly, all thinking, empty text), so the API
    returned stop_reason == "max_tokens" with no answer.

Contract added to Agent.handle_input():
    When a turn yields empty text AND no tool calls AND stop_reason == "max_tokens"
    AND thinking is currently enabled, retry ONCE with thinking disabled so the
    full budget is available for output. The retry is a normal tool-loop
    iteration (it may itself produce text or tool calls). Guards:
      - only one retry per handle_input() call
      - never retry when thinking is already "off" (cannot reduce further)
      - never retry on a non-"max_tokens" stop_reason (different cause)
      - never retry when text is present (a truncated answer is kept as-is)
    Observability: every logged turn records `stop_reason` in the JSONL turn log.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, patch

from openalph.agent import Agent
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import StreamEvent, Response, Usage, ToolCall, ThinkingBlock
from openalph.tools import ToolDef, ToolResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_provider(key="anthropic", type="anthropic", api_key="sk-test"):
    return ProviderConfig(key=key, type=type, api_key=api_key, base_url=None, quirks=[])


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


def make_recording_stream_fn(calls):
    """Patch for openalph.agent.stream: yields scripted events per invocation
    and records the kwargs of each call so tests can assert on `thinking`.

    `calls` is a list of event-lists; one list is consumed per stream() call.
    Calling stream() more times than scripted is an explicit test failure.
    """
    call_iter = iter(calls)
    recorded = []

    async def _fake_stream(*args, **kwargs):
        recorded.append(kwargs)
        try:
            events = next(call_iter)
        except StopIteration:  # pragma: no cover - guard for over-calling
            raise AssertionError("stream() called more times than scripted")
        for event in events:
            yield event

    _fake_stream.recorded = recorded
    return _fake_stream


def _done(content="", stop_reason="end_turn", output_tokens=50, thinking=None):
    return StreamEvent(
        type="done",
        response=make_response(content=content, output_tokens=output_tokens,
                               stop_reason=stop_reason, thinking=thinking or []),
        stop_reason=stop_reason,
        model="claude-sonnet-4-20250514",
    )


def empty_maxtokens_events(output_tokens=16384):
    """The bug signature: a thinking-only response that hit max_tokens."""
    return [
        StreamEvent(type="thinking", content="deliberating " * 40),
        StreamEvent(type="signature", content="sig-x"),
        _done(content="", stop_reason="max_tokens", output_tokens=output_tokens,
              thinking=[ThinkingBlock(thinking="deliberating", signature="sig-x")]),
    ]


def text_events(text, stop_reason="end_turn", output_tokens=50):
    return [
        StreamEvent(type="text", content=text),
        _done(content=text, stop_reason=stop_reason, output_tokens=output_tokens),
    ]


def tool_call_events(tc):
    return [
        StreamEvent(type="tool_done", tool_index=0, tool_call=tc),
        StreamEvent(
            type="done",
            response=make_response(content="", tool_calls=[tc], stop_reason="tool_use"),
            stop_reason="tool_use",
            model="claude-sonnet-4-20250514",
        ),
    ]


# ---------------------------------------------------------------------------
# Recovery behaviour
# ---------------------------------------------------------------------------

class TestEmptyMaxTokensRetry:

    @pytest.mark.asyncio
    async def test_retries_and_returns_recovered_text(self, tmp_path):
        """Empty text + stop_reason=max_tokens + thinking on -> retry yields the answer."""
        agent = Agent(make_config(tmp_path))
        fn = make_recording_stream_fn([
            empty_maxtokens_events(),
            text_events("Here is the real answer."),
        ])
        with patch("openalph.agent.stream", new=fn):
            result = await agent.handle_input("do the hard thing", thinking="max")

        assert result == "Here is the real answer."
        assert len(fn.recorded) == 2, "expected exactly one retry"

    @pytest.mark.asyncio
    async def test_retry_disables_thinking(self, tmp_path):
        """The retry call must set thinking='off'; the first keeps the configured level."""
        agent = Agent(make_config(tmp_path))
        fn = make_recording_stream_fn([
            empty_maxtokens_events(),
            text_events("answer"),
        ])
        with patch("openalph.agent.stream", new=fn):
            await agent.handle_input("x", thinking="max")

        assert fn.recorded[0]["thinking"] == "max"
        assert fn.recorded[1]["thinking"] == "off"

    @pytest.mark.asyncio
    async def test_thinking_from_config_is_resolved(self, tmp_path):
        """When handle_input thinking=None, the configured level drives the first call."""
        agent = Agent(make_config(tmp_path, thinking="high"))
        fn = make_recording_stream_fn([
            empty_maxtokens_events(),
            text_events("recovered"),
        ])
        with patch("openalph.agent.stream", new=fn):
            result = await agent.handle_input("x")  # no explicit thinking

        assert result == "recovered"
        assert fn.recorded[0]["thinking"] == "high"
        assert fn.recorded[1]["thinking"] == "off"

    @pytest.mark.asyncio
    async def test_no_empty_assistant_message_in_history(self, tmp_path):
        """After recovery, history holds only the real answer — not an empty turn."""
        agent = Agent(make_config(tmp_path))
        fn = make_recording_stream_fn([
            empty_maxtokens_events(),
            text_events("Here is the real answer."),
        ])
        with patch("openalph.agent.stream", new=fn):
            await agent.handle_input("x", room_id="!r:s", thinking="max")

        assistant_msgs = [m for m in agent.history("!r:s") if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "Here is the real answer."

    @pytest.mark.asyncio
    async def test_retry_can_produce_tool_calls(self, tmp_path):
        """The retry is a normal iteration: if it emits a tool call, the loop runs it."""
        agent = Agent(make_config(tmp_path))
        tc = ToolCall(id="tc1", name="shell", input={"command": "ls"})
        fn = make_recording_stream_fn([
            empty_maxtokens_events(),     # original: empty, hit max_tokens
            tool_call_events(tc),         # retry (thinking off): emits a tool call
            text_events("All done."),     # after tool result
        ])
        with patch("openalph.agent.stream", new=fn), \
             patch("openalph.agent.execute_tool", new_callable=AsyncMock,
                   return_value=ToolResult(content="file.txt", is_error=False)):
            result = await agent.handle_input("x", thinking="max")

        assert result == "All done."
        assert len(fn.recorded) == 3
        assert fn.recorded[1]["thinking"] == "off"   # retry
        assert fn.recorded[2]["thinking"] == "off"   # stays off for the rest of the turn


# ---------------------------------------------------------------------------
# Guards: when we must NOT retry
# ---------------------------------------------------------------------------

class TestRetryGuards:

    @pytest.mark.asyncio
    async def test_no_retry_when_thinking_already_off(self, tmp_path):
        """Cannot reduce thinking below off — return empty unchanged."""
        agent = Agent(make_config(tmp_path))
        fn = make_recording_stream_fn([empty_maxtokens_events()])
        with patch("openalph.agent.stream", new=fn):
            result = await agent.handle_input("x", thinking="off")

        assert result == ""
        assert len(fn.recorded) == 1, "must not retry when thinking is off"

    @pytest.mark.asyncio
    async def test_no_retry_on_non_maxtokens_empty(self, tmp_path):
        """Empty text with stop_reason=end_turn is a different failure — do not retry."""
        agent = Agent(make_config(tmp_path))
        fn = make_recording_stream_fn([
            [_done(content="", stop_reason="end_turn")],
        ])
        with patch("openalph.agent.stream", new=fn):
            result = await agent.handle_input("x", thinking="max")

        assert result == ""
        assert len(fn.recorded) == 1, "only stop_reason=max_tokens triggers the retry"

    @pytest.mark.asyncio
    async def test_no_retry_when_text_present_but_truncated(self, tmp_path):
        """A non-empty answer that hit max_tokens is kept; retrying would lose it."""
        agent = Agent(make_config(tmp_path))
        fn = make_recording_stream_fn([
            text_events("partial truncated answer", stop_reason="max_tokens",
                        output_tokens=16384),
        ])
        with patch("openalph.agent.stream", new=fn):
            result = await agent.handle_input("x", thinking="max")

        assert result == "partial truncated answer"
        assert len(fn.recorded) == 1

    @pytest.mark.asyncio
    async def test_retry_attempted_only_once(self, tmp_path):
        """If the retry is itself empty+max_tokens, give up (no infinite loop)."""
        agent = Agent(make_config(tmp_path))
        fn = make_recording_stream_fn([
            empty_maxtokens_events(),   # original
            empty_maxtokens_events(),   # retry ALSO empty (thinking now off)
        ])
        with patch("openalph.agent.stream", new=fn):
            result = await agent.handle_input("x", thinking="max")

        assert result == ""
        assert len(fn.recorded) == 2, "exactly one retry, then surface the empty result"

    @pytest.mark.asyncio
    async def test_normal_response_not_retried(self, tmp_path):
        """Regression: an ordinary thinking response returns directly, no retry."""
        agent = Agent(make_config(tmp_path))
        fn = make_recording_stream_fn([text_events("normal answer")])
        with patch("openalph.agent.stream", new=fn):
            result = await agent.handle_input("hi", thinking="max")

        assert result == "normal answer"
        assert len(fn.recorded) == 1
        assert fn.recorded[0]["thinking"] == "max"


# ---------------------------------------------------------------------------
# Edge cases / the forced-summary path
# ---------------------------------------------------------------------------

class TestAdditionalCases:

    @pytest.mark.asyncio
    async def test_whitespace_only_text_triggers_retry(self, tmp_path):
        """Whitespace-only output (not just '') counts as empty for the retry."""
        agent = Agent(make_config(tmp_path))
        fn = make_recording_stream_fn([
            [StreamEvent(type="text", content="   \n  "),
             _done(content="   \n  ", stop_reason="max_tokens", output_tokens=16384)],
            text_events("real answer"),
        ])
        with patch("openalph.agent.stream", new=fn):
            result = await agent.handle_input("x", thinking="max")

        assert result == "real answer"
        assert len(fn.recorded) == 2

    @pytest.mark.asyncio
    async def test_forced_summary_runs_with_thinking_off(self, tmp_path):
        """Hitting the tool-call limit forces a summary that must disable thinking,
        so the wrap-up can never be eaten by the same budget-exhaustion bug."""
        agent = Agent(make_config(tmp_path, max_iterations=2))
        tc = ToolCall(id="tc", name="shell", input={"command": "ls"})
        fn = make_recording_stream_fn([
            tool_call_events(tc),                  # iteration 0
            tool_call_events(tc),                  # iteration 1 -> exhausts max_iterations
            text_events("summary of progress"),    # forced summary call
        ])
        with patch("openalph.agent.stream", new=fn), \
             patch("openalph.agent.execute_tool", new_callable=AsyncMock,
                   return_value=ToolResult(content="ok", is_error=False)):
            result = await agent.handle_input("x", thinking="max")

        assert result == "summary of progress"
        assert len(fn.recorded) == 3
        assert fn.recorded[0]["thinking"] == "max"   # normal tool iterations
        assert fn.recorded[2]["thinking"] == "off"   # the forced summary


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

class TestStopReasonLogging:

    def _read_turn_log(self, config):
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = Path(config.workspace) / "logs" / f"{config.name}-{date_str}.jsonl"
        lines = [line for line in log_file.read_text().splitlines() if line.strip()]
        return [json.loads(line) for line in lines]

    @pytest.mark.asyncio
    async def test_stop_reason_logged_for_text_turn(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)
        fn = make_recording_stream_fn([text_events("answer")])
        with patch("openalph.agent.stream", new=fn):
            await agent.handle_input("hi")

        entries = self._read_turn_log(config)
        assert entries[-1]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_stop_reason_logs_both_attempts_on_recovery(self, tmp_path):
        """The empty max_tokens attempt and the recovery are both recorded."""
        config = make_config(tmp_path)
        agent = Agent(config)
        fn = make_recording_stream_fn([
            empty_maxtokens_events(),
            text_events("recovered"),
        ])
        with patch("openalph.agent.stream", new=fn):
            await agent.handle_input("x", thinking="max")

        stops = [e["stop_reason"] for e in self._read_turn_log(config)]
        assert "max_tokens" in stops
        assert "end_turn" in stops
