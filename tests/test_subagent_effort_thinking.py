"""Red suite for kdsn.305.14 — subagent effort dispatch + thinking replay.

Covers (see memory/projects/openalph/specs/kdsn.305.14-subagent-effort-replay-spec.md):
  A — effort param: schema, dispatch threading, default medium, validation
  B — thinking attach/replay at the sub loop's message-construction sites
  C — _estimate_context_tokens counts thinking (GC honesty)
  D — flight recorder: full thinking text + effort in meta
  E — real-path dispatch through execute_tool (tool-management "one lesson")

Provider is mocked at openalph.tools.subagent.complete (house convention,
tests/test_executor_subagent.py). GC is exercised through the real
context_gc.transform via config-driven thresholds.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from openalph.config import AgentConfig, ContextGCConfig
from openalph.provider import Response, ThinkingBlock, ToolCall, Usage
from openalph.tools import BUILTIN_TOOLS, ToolDef, ToolResult, execute_tool
from openalph.tools.subagent import _estimate_context_tokens, run_subagent


def make_provider(key="anthropic", type="anthropic", api_key="sk-test", base_url=None, quirks=None):
    from openalph.config import ProviderConfig
    return ProviderConfig(key=key, type=type, api_key=api_key, base_url=base_url, quirks=quirks or [])


def make_config(workspace=Path("/tmp/test"), **kwargs):
    defaults = dict(
        name="test-parent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider(key="anthropic")},
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def text_response(text="Sub-agent response"):
    return Response(
        content=text,
        tool_calls=[],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=100, output_tokens=50),
        stop_reason="end_turn",
    )


def tool_response(tool_name="file_read", tool_input=None, tool_id="tc_1", content="",
                  thinking=None):
    return Response(
        content=content,
        tool_calls=[ToolCall(id=tool_id, name=tool_name, input=tool_input or {})],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=100, output_tokens=50),
        stop_reason="tool_use",
        thinking=thinking or [],
    )


def think(text, sig="sig9"):
    return ThinkingBlock(thinking=text, signature=sig)


COMPLETE_PATH = "openalph.tools.subagent.complete"


# ---------------------------------------------------------------- A: effort

class TestEffortParam:

    def test_A1_schema_exposes_effort(self):
        """Tool schema gains effort: string enum, description names default medium."""
        params = BUILTIN_TOOLS["subagent"]["parameters"]
        assert "effort" in params["properties"]
        eff = params["properties"]["effort"]
        assert eff["type"] == "string"
        assert eff["enum"] == ["off", "low", "medium", "high", "xhigh", "max"]
        assert "medium" in eff["description"]

    @pytest.mark.asyncio
    async def test_A2_dispatch_threads_effort(self):
        """execute_tool threads input.effort into run_subagent."""
        with patch("openalph.tools.subagent.run_subagent", new_callable=AsyncMock) as m:
            m.return_value = ToolResult(content="ok", is_error=False)
            await execute_tool(
                name="subagent",
                input={"task": "t", "effort": "high"},
                tool_config={},
                agent_config=make_config(),
                callbacks=None,
            )
        assert m.call_args.kwargs.get("effort") == "high"

    @pytest.mark.asyncio
    async def test_A3_default_effort_is_medium(self):
        """Param omitted → complete() carries thinking='medium' (config.thinking bypassed)."""
        config = make_config()  # AgentConfig default thinking='off' must NOT win
        with patch(COMPLETE_PATH, new_callable=AsyncMock, return_value=text_response()) as mc:
            result = await run_subagent("Do a thing", config)
        assert result.is_error is False
        assert mc.call_args.kwargs.get("thinking") == "medium"

    @pytest.mark.asyncio
    async def test_A4_explicit_high(self):
        with patch(COMPLETE_PATH, new_callable=AsyncMock, return_value=text_response()) as mc:
            await run_subagent("Do a thing", make_config(), effort="high")
        assert mc.call_args.kwargs.get("thinking") == "high"

    @pytest.mark.asyncio
    async def test_A5_explicit_off(self):
        with patch(COMPLETE_PATH, new_callable=AsyncMock, return_value=text_response()) as mc:
            await run_subagent("Do a thing", make_config(), effort="off")
        assert mc.call_args.kwargs.get("thinking") == "off"

    @pytest.mark.asyncio
    async def test_A6_invalid_value_rejected_pre_io(self):
        """Bad effort → is_error steering, complete never called, zero side effects."""
        config = make_config(workspace=Path("/tmp/kdsn30514-a6"))
        with patch(COMPLETE_PATH, new_callable=AsyncMock, return_value=text_response()) as mc:
            result = await run_subagent("Do a thing", config, effort="turbo")
        assert result.is_error is True
        assert "off" in result.content and "xhigh" in result.content
        mc.assert_not_called()
        assert not (Path("/tmp/kdsn30514-a6") / "sessions" / "subs").exists()
        assert not (Path("/tmp/kdsn30514-a6") / "logs" / "subagents").exists()

    @pytest.mark.asyncio
    async def test_A7_non_string_effort_rejected(self):
        for bad in (5, True):
            with patch(COMPLETE_PATH, new_callable=AsyncMock, return_value=text_response()) as mc:
                result = await run_subagent("Do a thing", make_config(), effort=bad)
            assert result.is_error is True, f"effort={bad!r} must be rejected"
            mc.assert_not_called()

    @pytest.mark.asyncio
    async def test_A8_breaker_summary_carries_effort(self):
        """Circuit-breaker summary complete() uses the same effective effort, tools=None."""
        config = make_config()
        side = [tool_response(), text_response("summary text")]
        with patch(COMPLETE_PATH, new_callable=AsyncMock, side_effect=side) as mc:
            await run_subagent("Do a thing", config, effort="high", max_iterations=1)
        assert mc.call_count == 2
        breaker_call = mc.call_args_list[1]
        assert breaker_call.kwargs.get("thinking") == "high"
        assert breaker_call.kwargs.get("tools") is None


# ------------------------------------------------- B: thinking attach/replay

class TestThinkingReplay:

    @pytest.mark.asyncio
    async def test_B1_tool_branch_attaches_thinking(self):
        """Tool-call assistant turn carries thinking+signature into iteration-2 messages."""
        r1 = tool_response(thinking=[think("why the tool", "sig9")])
        side = [r1, text_response("done")]
        with patch(COMPLETE_PATH, new_callable=AsyncMock, side_effect=side) as mc:
            await run_subagent("Do a thing", make_config())
        msgs = mc.call_args_list[1].kwargs["messages"]
        asst = [m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")]
        assert len(asst) == 1
        assert asst[0]["thinking"] == [{"thinking": "why the tool", "signature": "sig9"}]

    @pytest.mark.asyncio
    async def test_B2_no_thinking_emitted_no_key(self):
        """Attach-when-present: no thinking in response → no 'thinking' key at all."""
        side = [tool_response(), text_response("done")]
        with patch(COMPLETE_PATH, new_callable=AsyncMock, side_effect=side) as mc:
            await run_subagent("Do a thing", make_config())
        msgs = mc.call_args_list[1].kwargs["messages"]
        asst = [m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")]
        assert asst and "thinking" not in asst[0]

    @pytest.mark.asyncio
    async def test_B3_truncation_path_attaches_thinking(self):
        """Truncation-continuation assistant turn also carries thinking."""
        trunc = Response(
            content="partial output",
            tool_calls=[],
            model="claude-sonnet-4-20250514",
            usage=Usage(input_tokens=100, output_tokens=50),
            stop_reason="max_tokens",
            thinking=[think("cut off mid plan")],
        )
        side = [trunc, text_response("continued")]
        with patch(COMPLETE_PATH, new_callable=AsyncMock, side_effect=side) as mc:
            await run_subagent("Do a thing", make_config())
        msgs = mc.call_args_list[1].kwargs["messages"]
        asst = [m for m in msgs if m["role"] == "assistant"]
        assert len(asst) == 1
        assert asst[0]["thinking"] == [{"thinking": "cut off mid plan", "signature": "sig9"}]
        assert any(m["role"] == "user" and "truncated" in str(m.get("content", "")).lower()
                   for m in msgs)

    @pytest.mark.asyncio
    async def test_B4_thinking_accumulates_across_iterations(self):
        """Iteration-3 messages replay thinking from BOTH prior tool iterations."""
        r1 = tool_response(tool_id="tc_1", thinking=[think("plan one", "s1")])
        r2 = tool_response(tool_id="tc_2", thinking=[think("plan two", "s2")])
        side = [r1, r2, text_response("done")]
        with patch(COMPLETE_PATH, new_callable=AsyncMock, side_effect=side) as mc:
            await run_subagent("Do a thing", make_config())
        msgs = mc.call_args_list[2].kwargs["messages"]
        # Targeted extraction: the wire shape carries ToolCall OBJECTS in
        # tool_calls (in-process; the GC transform round-trips them via
        # dataclasses.replace), so never blob-serialize the message list.
        thinking_blob = "".join(
            tb.get("thinking", "")
            for m in msgs for tb in (m.get("thinking") or [])
        )
        assert "plan one" in thinking_blob
        assert "plan two" in thinking_blob
        asst = [m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")]
        assert len(asst) == 2

    @pytest.mark.asyncio
    async def test_B5_gc_retention_alive_in_subs(self):
        """Boundary in a sub retains the newest thinking and strips older (tail knobs).

        Window 9000 / max_tokens 8000 → usable 1000 → auto threshold 850 est-tokens.
        Two 1200-char thinking turns cross it at iteration-2 top; tail budget 512
        tokens (2048 chars) retains NEWCOT, strips OLDCOT. Task echo survives.
        """
        cfg = make_config(
            model_limits={"anthropic/claude-sonnet-4-20250514": 9000},
            context=ContextGCConfig(thinking_tail_max_tokens=512),
        )
        task = "T" * 1600
        old = tool_response(tool_id="tc_1", thinking=[think("OLDCOT" * 200, "s_old")])
        new = tool_response(tool_id="tc_2", thinking=[think("NEWCOT" * 200, "s_new")])
        side = [old, new, text_response("done")]
        with patch(COMPLETE_PATH, new_callable=AsyncMock, side_effect=side) as mc:
            result = await run_subagent(task, cfg, call_id="gctest", max_tokens=8000)
        assert result.is_error is False
        # the boundary fired (third call happened after a boundary)
        assert mc.call_count == 3
        final_msgs = mc.call_args_list[2].kwargs["messages"]
        # Targeted extraction — thinking fields only (never blob-serialize a
        # message list carrying ToolCall objects).
        thinking_blob = "".join(
            tb.get("thinking", "")
            for m in final_msgs for tb in (m.get("thinking") or [])
        )
        assert "NEWCOT" in thinking_blob, "newest thinking must survive the boundary"
        assert "OLDCOT" not in thinking_blob, \
            "older thinking beyond the tail budget must be stripped"
        # task echo is durable content — never reduced
        assert any(m["role"] == "user" and m.get("content") == task for m in final_msgs)


# ---------------------------------------------------------------- C: estimator

class TestEstimatorCountsThinking:

    def test_C1_thinking_chars_counted(self):
        base_msg = {"role": "assistant", "content": "x", "tool_calls": []}
        with_thinking = dict(base_msg, thinking=[{"thinking": "12345678", "signature": "s"}])
        est_without = _estimate_context_tokens([base_msg])
        est_with = _estimate_context_tokens([with_thinking])
        assert est_with >= est_without + 2  # 8 chars ≈ 2 tokens


# ---------------------------------------------------------------- D: transcript

def _read_transcript(workspace: Path, call_id: str) -> list[dict]:
    subs = workspace / "sessions" / "subs"
    files = sorted(subs.glob(f"*-{call_id}.jsonl"))
    assert files, f"no transcript for call_id {call_id}"
    return [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]


class TestFlightRecorder:

    @pytest.mark.asyncio
    async def test_D1_transcript_records_full_thinking(self, tmp_path):
        r1 = tool_response(thinking=[think("full reasoning text " * 10, "sigZ")])
        side = [r1, text_response("done")]
        with patch(COMPLETE_PATH, new_callable=AsyncMock, side_effect=side):
            await run_subagent("Do a thing", make_config(workspace=tmp_path),
                               call_id="dtest")
        entries = _read_transcript(tmp_path, "dtest")
        assistant_events = [e for e in entries if e.get("event") == "assistant"]
        assert assistant_events, "assistant transcript entries must exist"
        assert any(
            any("full reasoning text" in tb.get("thinking", "")
                and tb.get("signature") == "sigZ"
                for tb in (e.get("thinking") or []))
            for e in assistant_events
        ), "full thinking text must be recorded verbatim (ruling 3)"

    @pytest.mark.asyncio
    async def test_D2_meta_records_effort(self, tmp_path):
        with patch(COMPLETE_PATH, new_callable=AsyncMock, return_value=text_response()):
            await run_subagent("Do a thing", make_config(workspace=tmp_path),
                               effort="xhigh", call_id="mtest")
        entries = _read_transcript(tmp_path, "mtest")
        meta = [e for e in entries if e.get("event") == "meta"]
        assert meta and meta[0].get("effort") == "xhigh"


# ---------------------------------------------------------------- E: real path

class TestRealPathDispatch:

    @pytest.mark.asyncio
    async def test_E1_full_dispatch_real_execute_tool(self, tmp_path):
        """Real execute_tool → real run_subagent → mocked provider only.

        The sub calls file_read with empty input (deterministic error result, no
        side effects), then answers. Proves effort lands and the run completes
        through the real dispatch path with parent-shaped callbacks.
        """
        bridge = {}
        tools = [
            ToolDef(name="file_read", description="Read", parameters={}, config={}),
            ToolDef(name="subagent", description="Spawn sub", parameters={}, config={}),
        ]
        side = [tool_response(tool_name="file_read"), text_response("all done")]
        with patch(COMPLETE_PATH, new_callable=AsyncMock, side_effect=side) as mc:
            result = await execute_tool(
                name="subagent",
                input={"task": "Do a thing", "effort": "high"},
                tool_config={},
                agent_config=make_config(workspace=tmp_path),
                tools=tools,
                callbacks={"room_id": "!parent:x", "subagent_results": bridge},
            )
        assert result.is_error is False
        assert "all done" in result.content
        assert mc.call_count == 2
        assert mc.call_args_list[0].kwargs.get("thinking") == "high"
        assert mc.call_args_list[1].kwargs.get("thinking") == "high"
        assert bridge, "subagent cost bridge must be written by the real path"
