"""kdsn.315 — Length-stop normalization, visibility, and single continuation.

Contract (spec: memory/projects/openalph/specs/kdsn315-length-stop-spec.md):

P1  provider._normalize_stop_reason: OpenAI "length" -> canonical "max_tokens";
    applied at BOTH openai response sites (non-streaming _parse_openai_response,
    streaming accumulation). Anthropic vocabulary unchanged.
P2  One best-effort send_notice per handle_input turn when stop_reason is
    max_tokens (text branch AND tool branch, deduped per turn); notice failure
    never alters turn outcome.
P3  [agent] max_continuations (int >=0, not bool, default 1): non-empty
    text length-stop gets ONE request-local continuation; durable history sees
    ONE merged assistant message (RC1); usage merged; response2 tool_calls are
    abandoned (logged), never executed.
P4  Empty-text thinking-retry (agent.py) becomes live on openai-compat via P1
    — zero code change there, pinned by regression test.

Loop tests use an OPENAI-type provider so the vocabulary fix is what's proven.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.agent import Agent
from openalph.config import AgentConfig, ConfigError, ProviderConfig
from openalph.provider import (
    StreamEvent, Response, Usage, ToolCall, _normalize_stop_reason, _parse_openai_response,
)
from openalph.tools import ToolDef, ToolResult


# ---------------------------------------------------------------------------
# Fixtures (mirroring tests/test_agent_streaming.py)
# ---------------------------------------------------------------------------

def make_provider(key="blackwell", type="openai", api_key="none",
                  base_url="http://127.0.0.1:9999/v1"):
    return ProviderConfig(
        key=key, type=type, api_key=api_key, base_url=base_url, quirks=[],
    )


def make_config(workspace, **kwargs):
    defaults = {
        "name": "test-agent",
        "default_model": "blackwell/qwen38-27b-fp8",
        "max_tokens": 32768,
        "providers": {"blackwell": make_provider()},
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
        model="qwen38-27b-fp8",
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


def make_stream_fn(calls):
    """Fake stream(): one inner list per invocation, consumed in order."""
    call_iter = iter(calls)

    async def _fake_stream(*args, **kwargs):
        events = next(call_iter)
        for event in events:
            yield event

    return _fake_stream


def make_recording_stream_fn(calls):
    """Like make_stream_fn, but records (args, kwargs) of every invocation."""
    call_iter = iter(calls)
    recorded = []

    async def _fake_stream(*args, **kwargs):
        recorded.append({"args": args, "kwargs": kwargs})
        events = next(call_iter)
        for event in events:
            yield event

    return _fake_stream, recorded


def text_events(text, stop_reason="end_turn", input_tokens=100,
                output_tokens=50, thinking=None):
    events = [StreamEvent(type="text", content=text)]
    events.append(StreamEvent(
        type="done",
        response=make_response(content=text, stop_reason=stop_reason,
                               input_tokens=input_tokens,
                               output_tokens=output_tokens,
                               thinking=thinking),
        stop_reason=stop_reason,
        model="qwen38-27b-fp8",
    ))
    return events


def tool_events(text_before, tool_calls, stop_reason="tool_use",
                input_tokens=100, output_tokens=50):
    events = []
    if text_before:
        events.append(StreamEvent(type="text", content=text_before))
    for i, tc in enumerate(tool_calls):
        events.append(StreamEvent(type="tool_done", tool_index=i,
                                  tool_call=tc))
    events.append(StreamEvent(
        type="done",
        response=make_response(content=text_before, tool_calls=tool_calls,
                               stop_reason=stop_reason,
                               input_tokens=input_tokens,
                               output_tokens=output_tokens),
        stop_reason=stop_reason,
        model="qwen38-27b-fp8",
    ))
    return events


def notice_spy():
    return AsyncMock(return_value=None)


# ---------------------------------------------------------------------------
# P1: provider seam
# ---------------------------------------------------------------------------

class TestNormalizeStopReason:

    def test_length_maps_to_max_tokens(self):
        assert _normalize_stop_reason("length") == "max_tokens"

    @pytest.mark.parametrize("value", ["stop", "end_turn", "max_tokens",
                                       "tool_calls", "tool_use",
                                       "content_filter", "stop_sequence"])
    def test_passthrough(self, value):
        assert _normalize_stop_reason(value) == value

    def test_none_passthrough(self):
        assert _normalize_stop_reason(None) is None


class TestParseOpenAIResponse:

    def _stub_openai_response(self, finish_reason):
        msg = MagicMock()
        msg.content = "partial answer"
        msg.tool_calls = None
        msg.reasoning = None
        msg.reasoning_content = None
        choice = MagicMock()
        choice.finish_reason = finish_reason
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        # kdsn.315 stub-shape fix (reported): the real OpenAI wire always
        # carries a usage object; _openai_usage() reads prompt_tokens /
        # completion_tokens off it, so None crashes the (pre-existing)
        # parser for reasons unrelated to stop_reason normalization.
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        resp.model = "qwen38-27b-fp8"
        return resp

    def test_length_normalized(self):
        out = _parse_openai_response(self._stub_openai_response("length"))
        assert out.stop_reason == "max_tokens"

    def test_stop_untouched(self):
        out = _parse_openai_response(self._stub_openai_response("stop"))
        assert out.stop_reason == "stop"


# ---------------------------------------------------------------------------
# P3 config surface
# ---------------------------------------------------------------------------

class TestMaxContinuationsConfig:

    def test_default_is_1(self, tmp_path):
        config = make_config(tmp_path)
        assert config.max_continuations == 1

    def test_zero_accepted(self, tmp_path):
        config = make_config(tmp_path, max_continuations=0)
        assert config.max_continuations == 0

    @pytest.mark.parametrize("bad", ["2", True, -1, 1.5])
    def test_invalid_rejected(self, tmp_path, bad):
        with pytest.raises(ConfigError):
            make_config(tmp_path, max_continuations=bad)


# ---------------------------------------------------------------------------
# P4: empty-text thinking retry goes live on openai-compat (the discriminator)
# ---------------------------------------------------------------------------

class TestEmptyTextRetryOpenAI:

    @pytest.mark.asyncio
    async def test_length_stop_triggers_thinking_off_retry(self, tmp_path):
        """OpenAI-type wire 'length' + empty text + thinking=xhigh must retry
        with thinking dropped. RED on old code: stop_reason stays 'length',
        retry never fires, handle_input returns '' after ONE provider call."""
        config = make_config(tmp_path, thinking="xhigh")
        agent = Agent(config)

        stream_fn, recorded = make_recording_stream_fn([
            # call 1: model burned the budget on reasoning, emitted no text
            text_events("", stop_reason="length", output_tokens=32768),
            # call 2 (retry, thinking=off): produces the answer
            text_events("Recovered answer", stop_reason="end_turn"),
        ])

        with patch("openalph.agent.stream", new=stream_fn):
            result = await agent.handle_input("Hi")

        assert result == "Recovered answer"
        assert len(recorded) == 2, (
            f"expected retry (2 provider calls), got {len(recorded)}"
        )
        assert recorded[1]["kwargs"].get("thinking") == "off"


# ---------------------------------------------------------------------------
# P3: single continuation on non-empty length-stop
# ---------------------------------------------------------------------------

class TestContinuation:

    @pytest.mark.asyncio
    async def test_continuation_merges_text_and_usage(self, tmp_path):
        config = make_config(tmp_path, thinking="xhigh")
        agent = Agent(config)

        stream_fn, recorded = make_recording_stream_fn([
            text_events("The answer is ", stop_reason="length",
                        output_tokens=100),
            text_events("42.", stop_reason="end_turn", output_tokens=20),
        ])
        log_spy = AsyncMock()

        with patch("openalph.agent.stream", new=stream_fn), \
                patch.object(agent, "_log_turn", log_spy):
            result = await agent.handle_input("Hi")

        # direct concatenation, single returned string
        assert result == "The answer is 42."
        assert len(recorded) == 2

        # RC1: durable history ends with ONE merged assistant message and
        # contains NO continuation-framing user message
        history = agent.history("_default")
        assert history[-1]["role"] == "assistant"
        assert history[-1]["content"] == "The answer is 42."
        framing = [m for m in history
                   if m["role"] == "user" and "output-token limit" in str(
                       m.get("content", ""))]
        assert framing == [], "continuation framing must be request-local"

        # the continuation REQUEST carried the framing + partial text
        cont_msgs = recorded[1]["kwargs"].get("messages") or \
            (recorded[1]["args"][2] if len(recorded[1]["args"]) > 2 else [])
        cont_str = str(cont_msgs)
        assert "output-token limit" in cont_str
        assert "The answer is" in cont_str

        # usage from BOTH calls reaches the turn log
        logged = log_spy.call_args
        assert logged.kwargs.get("output_tokens") == 120

        # stop reason of the continuation is what the room records
        assert agent.last_stop_reason("_default") == "end_turn"

    @pytest.mark.asyncio
    async def test_continuation_also_truncated_stops_at_one(self, tmp_path):
        config = make_config(tmp_path, thinking="xhigh")
        agent = Agent(config)

        stream_fn, recorded = make_recording_stream_fn([
            text_events("Part one ", stop_reason="length", output_tokens=100),
            text_events("part two", stop_reason="length", output_tokens=100),
        ])
        cb = notice_spy()

        with patch("openalph.agent.stream", new=stream_fn):
            result = await agent.handle_input("Hi", callbacks={
                "send_notice": cb})

        assert result == "Part one part two"
        assert len(recorded) == 2, "must not attempt a second continuation"
        assert agent.last_stop_reason("_default") == "max_tokens"
        assert cb.call_count == 1

    @pytest.mark.asyncio
    async def test_max_continuations_zero_is_legacy(self, tmp_path):
        config = make_config(tmp_path, thinking="xhigh", max_continuations=0)
        agent = Agent(config)

        stream_fn, recorded = make_recording_stream_fn([
            text_events("Truncated answer", stop_reason="length",
                        output_tokens=100),
        ])

        with patch("openalph.agent.stream", new=stream_fn):
            result = await agent.handle_input("Hi")

        assert result == "Truncated answer"
        assert len(recorded) == 1, "max_continuations=0 must not continue"
        assert agent.last_stop_reason("_default") == "max_tokens"

    @pytest.mark.asyncio
    async def test_continuation_tool_calls_abandoned(self, tmp_path):
        """If the continuation answers with tool_calls, the attempt is
        abandoned: partial text kept, tool NOT executed, clean turn end."""
        config = make_config(tmp_path, thinking="xhigh")
        agent = Agent(config)
        tc = ToolCall(id="c1", name="shell", input={"command": "rm -rf /"})

        stream_fn, recorded = make_recording_stream_fn([
            text_events("Partial answer ", stop_reason="length",
                        output_tokens=100),
            tool_events("", [tc], stop_reason="tool_use", output_tokens=20),
        ])
        exec_mock = AsyncMock(return_value=ToolResult(content="x",
                                                      is_error=False))

        with patch("openalph.agent.stream", new=stream_fn), \
                patch("openalph.agent.execute_tool", new=exec_mock):
            result = await agent.handle_input("Hi", tools=[SHELL_TOOL])

        assert result == "Partial answer "
        exec_mock.assert_not_awaited()
        assert len(recorded) == 2


# ---------------------------------------------------------------------------
# P2: notice
# ---------------------------------------------------------------------------

class TestLengthStopNotice:

    @pytest.mark.asyncio
    async def test_notice_fires_once_on_text_length_stop(self, tmp_path):
        config = make_config(tmp_path, thinking="xhigh")
        agent = Agent(config)

        stream_fn, _ = make_recording_stream_fn([
            text_events("Truncated", stop_reason="length", output_tokens=100),
            text_events(" rest", stop_reason="end_turn", output_tokens=10),
        ])
        cb = notice_spy()

        with patch("openalph.agent.stream", new=stream_fn):
            await agent.handle_input("Hi", callbacks={"send_notice": cb})

        assert cb.call_count == 1
        body = str(cb.call_args)
        assert "max_tokens" in body

    @pytest.mark.asyncio
    async def test_notice_failure_never_breaks_turn(self, tmp_path):
        config = make_config(tmp_path, thinking="xhigh")
        agent = Agent(config)

        stream_fn, _ = make_recording_stream_fn([
            text_events("Truncated", stop_reason="length", output_tokens=100),
            text_events(" rest", stop_reason="end_turn", output_tokens=10),
        ])

        async def _boom(room_id, body):
            raise RuntimeError("matrix down")

        with patch("openalph.agent.stream", new=stream_fn):
            result = await agent.handle_input("Hi", callbacks={
                "send_notice": _boom})

        assert result == "Truncated rest"

    @pytest.mark.asyncio
    async def test_notice_fires_on_tool_branch_length_stop(self, tmp_path):
        config = make_config(tmp_path, thinking="xhigh")
        agent = Agent(config)
        tc = ToolCall(id="t1", name="shell",
                      input={"command": "ls"})

        stream_fn, _ = make_recording_stream_fn([
            tool_events("", [tc], stop_reason="length", output_tokens=100),
            text_events("done", stop_reason="end_turn", output_tokens=10),
        ])
        exec_mock = AsyncMock(return_value=ToolResult(content="a.txt",
                                                      is_error=False))
        cb = notice_spy()

        with patch("openalph.agent.stream", new=stream_fn), \
                patch("openalph.agent.execute_tool", new=exec_mock):
            await agent.handle_input("Hi", tools=[SHELL_TOOL],
                                     callbacks={"send_notice": cb})

        exec_mock.assert_awaited_once()   # tool still executes normally
        assert cb.call_count == 1
        assert "max_tokens" in str(cb.call_args)


# ---------------------------------------------------------------------------
# Audit remediation regressions (synkimi3 cold-read, 2026-09-02)
# ---------------------------------------------------------------------------

class TestAuditRemediations:

    @pytest.mark.asyncio
    async def test_continuation_tokens_reach_usage_accumulator(self, tmp_path):
        """H1: continuation usage must reach _record_turn_usage — global AND
        per-room counters include BOTH calls, not just the main response."""
        config = make_config(tmp_path, thinking="xhigh")
        agent = Agent(config)

        stream_fn, _ = make_recording_stream_fn([
            text_events("The answer is ", stop_reason="length",
                        output_tokens=100),
            text_events("42.", stop_reason="end_turn", output_tokens=20),
        ])

        with patch("openalph.agent.stream", new=stream_fn):
            await agent.handle_input("Hi")

        assert agent.total_output_tokens == 120, (
            "continuation tokens missing from the global usage accumulator"
        )
        room = agent._usage_for("_default")
        assert room["total_output_tokens"] == 120, (
            "continuation tokens missing from the per-room accumulator"
        )

    @pytest.mark.asyncio
    async def test_no_notice_on_recovered_empty_retry_turn(self, tmp_path):
        """M1: an empty-retry turn that RECOVERED (final text, end_turn, no
        continuation) must not fire the budget notice — the turn was not
        truncated; the retry succeeded."""
        config = make_config(tmp_path, thinking="xhigh")
        agent = Agent(config)

        stream_fn, _ = make_recording_stream_fn([
            text_events("", stop_reason="length", output_tokens=32768),
            text_events("Recovered answer", stop_reason="end_turn",
                        output_tokens=30),
        ])
        cb = notice_spy()

        with patch("openalph.agent.stream", new=stream_fn):
            result = await agent.handle_input("Hi", callbacks={
                "send_notice": cb})

        assert result == "Recovered answer"
        assert cb.call_count == 0
