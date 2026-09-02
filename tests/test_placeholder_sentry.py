"""GC placeholder sentry — refuse model-regurgitated elision markers.

Florin crit (workspace-3ejn.2, 2026-09-01): after a GC boundary, qwen38-27b
emits the harness's own context-elision markers ("[stripped: N chars]") as
its own tool-call payload values, with self-estimated sizes (calibrated N =
the intended payload size — the model internalized the marker as a
*substitute for generation*, not a copy). The live dispatch path is
transform-free, so the marker value reaches the handler and executes
literally: shell calls fail loudly, code writes are caught by the syntax
validators, but a file_write carrying a marker onto a non-code file would
corrupt it SILENTLY.

The sentry sits at the trust boundary (``_execute_tool_inner``, pre-dispatch):
a string value that is a placeholder marker (whole-value match, never
substring) refuses execution and returns corrective steering. Uniform across
main sessions, subagents, and CLI — all dispatch through ``execute_tool``.

This file IS the spec:
  S1  whole-value marker in any string param -> refuse (is_error); handler
      never runs; no side effects; input dict unmutated.
  S2  marker as a substring of a longer legitimate value -> passes through.
  S3  non-string values / empty strings -> pass through (never tripped).
  S4  whitespace-padded marker -> still caught (strip before match).
  S5  all four placeholder families covered: legacy input, legacy result,
      GC pointer, GC media-expunge.
  S6  real path: REAL Agent + REAL MatrixBot._build_agent_callbacks, mocked
      provider only — a marker-bearing tool call is refused at the boundary,
      the corrective error reaches the model as the tool result, the victim
      file is never created, and the turn completes.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from openalph.agent import Agent
from openalph.provider import Response, StreamEvent, ToolCall, Usage
from openalph.tools import execute_tool

from test_guidance_integration import (
    ROOM_A,
    _cfg,
    _make_bot_with_real_agent,
    _setup_workspace,
)

MARKER_INPUT = "[stripped: 4760 chars]"
MARKER_RESULT = "[stripped: file_read result, 2043 chars]"
MARKER_POINTER = ("[expunged at GC boundary 304: shell command "
                  "(4760 chars) \u2014 re-run the tool if the result is needed]")
MARKER_MEDIA = ("[expunged at GC boundary 304: media attachment "
                "\u2014 re-share or re-generate the image if needed]")

STEERING_NEEDLE = "context-GC elision marker"


def _agent(tmp_path, **kw):
    ws = _setup_workspace(tmp_path)
    config = _cfg(ws, **kw)
    return Agent(config), ws


# ============================================================================
# S1 — whole-value markers refused, handler never runs
# ============================================================================

class TestS1Refusal:
    @pytest.mark.asyncio
    async def test_shell_marker_refused(self, tmp_path):
        agent, _ = _agent(tmp_path)
        result = await execute_tool(
            name="shell", input={"command": MARKER_INPUT, "timeout": 60},
            tool_config={}, agent_config=agent.config, tools=agent.tools,
        )
        assert result.is_error
        assert STEERING_NEEDLE in result.content
        assert "command" in result.content

    @pytest.mark.asyncio
    async def test_file_write_marker_victim_file_never_created(self, tmp_path):
        """The silent-corruption case: marker content on a fresh non-code file."""
        agent, ws = _agent(tmp_path)
        victim = ws / "victim.md"
        assert not victim.exists()
        result = await execute_tool(
            name="file_write",
            input={"path": "victim.md", "content": MARKER_INPUT},
            tool_config={}, agent_config=agent.config, tools=agent.tools,
        )
        assert result.is_error, f"expected refusal: {result.content}"
        assert STEERING_NEEDLE in result.content
        assert not victim.exists(), "sentry must refuse BEFORE the handler runs"

    @pytest.mark.asyncio
    async def test_input_dict_unmutated(self, tmp_path):
        agent, _ = _agent(tmp_path)
        inp = {"content": MARKER_INPUT, "path": "x.md"}
        snapshot = dict(inp)
        await execute_tool(
            name="file_write", input=inp, tool_config={},
            agent_config=agent.config, tools=agent.tools,
        )
        assert inp == snapshot, "sentry must not mutate the caller's input"

    @pytest.mark.asyncio
    async def test_all_families_refused(self, tmp_path):
        agent, ws = _agent(tmp_path)
        for i, marker in enumerate(
                (MARKER_INPUT, MARKER_RESULT, MARKER_POINTER, MARKER_MEDIA)):
            result = await execute_tool(
                name="file_write",
                input={"path": f"v{i}.md", "content": marker},
                tool_config={}, agent_config=agent.config, tools=agent.tools,
            )
            assert result.is_error, f"family {i} not refused: {result.content}"
            assert STEERING_NEEDLE in result.content
        # pointer family also as a shell command (Florin's original strike)
        result = await execute_tool(
            name="shell", input={"command": MARKER_POINTER},
            tool_config={}, agent_config=agent.config, tools=agent.tools,
        )
        assert result.is_error
        assert STEERING_NEEDLE in result.content


# ============================================================================
# S2/S3 — precision: never a substring match, never a shape raise
# ============================================================================

class TestS2S3Precision:
    @pytest.mark.asyncio
    async def test_substring_in_longer_content_passes(self, tmp_path):
        agent, ws = _agent(tmp_path)
        legit = f"journal line: {MARKER_INPUT} was in the transcript"
        result = await execute_tool(
            name="file_write",
            input={"path": "notes.md", "content": legit},
            tool_config={}, agent_config=agent.config, tools=agent.tools,
        )
        assert not result.is_error, f"substring must not trip: {result.content}"
        assert (ws / "notes.md").exists()

    @pytest.mark.asyncio
    async def test_substring_in_shell_command_passes(self, tmp_path):
        """A legitimate grep for the marker string itself must run."""
        agent, _ = _agent(tmp_path)
        result = await execute_tool(
            name="shell",
            input={"command": f"echo {MARKER_INPUT!r} >/dev/null && echo ok"},
            tool_config={}, agent_config=agent.config, tools=agent.tools,
        )
        assert not result.is_error, f"legit command must pass: {result.content}"

    @pytest.mark.asyncio
    async def test_int_values_never_trip(self, tmp_path):
        agent, _ = _agent(tmp_path)
        result = await execute_tool(
            name="shell", input={"command": "echo ok", "timeout": 60},
            tool_config={}, agent_config=agent.config, tools=agent.tools,
        )
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_empty_string_never_trips(self, tmp_path):
        agent, _ = _agent(tmp_path)
        result = await execute_tool(
            name="shell", input={"command": ""},
            tool_config={}, agent_config=agent.config, tools=agent.tools,
        )
        assert STEERING_NEEDLE not in result.content

    @pytest.mark.asyncio
    async def test_near_miss_shapes_never_trip(self, tmp_path):
        agent, ws = _agent(tmp_path)
        for i, val in enumerate((
                "[stripped: 4760 chars",            # unclosed bracket
                "[stripped: abc chars]",             # non-numeric
                "[stripped: 4760 char]",             # singular
                " [stripped: 4760 chars] extra",    # trailing content
                "[expunged at GC boundary: nope]",   # no boundary index
                "[Stripped: 4760 chars]",            # case
        )):
            result = await execute_tool(
                name="file_write",
                input={"path": f"nm{i}.md", "content": val},
                tool_config={}, agent_config=agent.config, tools=agent.tools,
            )
            assert not result.is_error, f"near-miss tripped: {val!r}"


# ============================================================================
# S4 — whitespace-padded marker still caught
# ============================================================================

class TestS4Padding:
    @pytest.mark.asyncio
    async def test_padded_marker_refused(self, tmp_path):
        agent, ws = _agent(tmp_path)
        victim = ws / "padded.md"
        result = await execute_tool(
            name="file_write",
            input={"path": "padded.md", "content": f"  {MARKER_INPUT}  "},
            tool_config={}, agent_config=agent.config, tools=agent.tools,
        )
        assert result.is_error, f"padded marker must be caught: {result.content}"
        assert not victim.exists()


# ============================================================================
# S6 — real path: REAL Agent + REAL callbacks, provider mocked
# ============================================================================

def _make_marker_stream(final_text="Understood — regenerating with full payload."):
    """Stream factory: iter 1 emits file_write with a marker value; then final."""
    call_idx = [0]

    async def _stream(*, config=None, system=None, messages=None,
                      tools=None, model="test", thinking=None,
                      cache_ttl=None, **kw):
        call_idx[0] += 1
        if call_idx[0] == 1:
            tc = ToolCall(id="tc_marker", name="file_write",
                          input={"path": "victim_s6.md", "content": MARKER_INPUT})
            yield StreamEvent(type="tool_done", tool_index=0, tool_call=tc)
            yield StreamEvent(
                type="done",
                response=Response(content="", tool_calls=[tc], model=model,
                                  usage=Usage(input_tokens=10, output_tokens=5),
                                  stop_reason="tool_use"),
                stop_reason="tool_use", model=model)
        else:
            yield StreamEvent(type="text", content=final_text)
            yield StreamEvent(
                type="done",
                response=Response(content=final_text, model=model,
                                  usage=Usage(input_tokens=10, output_tokens=5),
                                  stop_reason="end_turn"),
                stop_reason="end_turn", model=model)

    return _stream


class TestS6RealPath:
    @pytest.mark.asyncio
    async def test_real_agent_marker_call_refused_no_corruption(self, tmp_path):
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM_A, None)
        ws = Path(str(agent.config.workspace))
        victim = ws / "victim_s6.md"

        with patch("openalph.agent.stream",
                   side_effect=_make_marker_stream()):
            await agent.handle_input("work", room_id=ROOM_A, callbacks=cb)

        # 1) no corruption: the victim file was never written
        assert not victim.exists(), "real path: marker must not reach file_write"
        # 2) the corrective error reached the model as the tool result
        history = agent.history(ROOM_A)
        tool_msgs = [m for m in history if m.get("role") == "tool"]
        assert any(STEERING_NEEDLE in str(m.get("content", ""))
                   for m in tool_msgs), \
            "sentry steering must land in the model's context as the tool result"
        # 3) the turn completed (final text reached the terminal path)
        assert any("regenerating with full payload" in str(m.get("content", ""))
                   for m in history if m.get("role") == "assistant")

# ============================================================================
# S7 — nested scan: markers inside list-of-dict params (todo_write shape)
# ============================================================================

class TestS7Nested:
    @pytest.mark.asyncio
    async def test_marker_nested_in_todo_list_refused(self, tmp_path):
        agent, _ = _agent(tmp_path)
        result = await execute_tool(
            name="todo_write",
            input={"todos": [
                {"content": "real item", "status": "pending"},
                {"content": MARKER_INPUT, "status": "in_progress"},
            ]},
            tool_config={}, agent_config=agent.config, tools=agent.tools,
        )
        assert result.is_error, f"nested marker must be refused: {result.content}"
        assert STEERING_NEEDLE in result.content

    @pytest.mark.asyncio
    async def test_clean_nested_list_passes(self, tmp_path):
        agent, _ = _agent(tmp_path)
        result = await execute_tool(
            name="todo_write",
            input={"todos": [
                {"content": "check the [stripped: 10 chars] note in logs",
                 "status": "pending"},
            ]},
            tool_config={}, agent_config=agent.config, tools=agent.tools,
        )
        assert not result.is_error, f"clean nested list must pass: {result.content}"

    @pytest.mark.asyncio
    async def test_marker_in_string_list_item_refused(self, tmp_path):
        agent, _ = _agent(tmp_path)
        result = await execute_tool(
            name="shell",
            input={"command": "echo ok", "extra_list": [MARKER_POINTER]},
            tool_config={}, agent_config=agent.config, tools=agent.tools,
        )
        assert result.is_error, "list-item marker must be refused"


# ============================================================================
# S8 — generator sync pin: the sentry regex must match EVERY marker the
# production generators emit (a future wording edit must turn this red, not
# silently disarm a family).
# ============================================================================

class TestS8GeneratorSync:
    def test_regex_matches_live_generators(self):
        from openalph.context_gc import (
            legacy_input_placeholder, legacy_tool_placeholder, tool_pointer)
        from openalph.tools import _GC_PLACEHOLDER_RE
        for marker in (
                legacy_input_placeholder(4760),
                legacy_tool_placeholder("file_read", 2043),
                tool_pointer(304, "shell", {"command": "x"}, 4760),
                tool_pointer(304, "tool", None, 0),
        ):
            assert _GC_PLACEHOLDER_RE.match(marker.strip()), \
                f"sentry does not cover generator output: {marker!r}"

    def test_regex_matches_media_expunge_literal(self):
        from openalph.tools import _GC_PLACEHOLDER_RE
        marker = ("[expunged at GC boundary 304: media attachment "
                  "\u2014 re-share or re-generate the image if needed]")
        assert _GC_PLACEHOLDER_RE.match(marker.strip())

    @pytest.mark.asyncio
    async def test_real_agent_clean_call_untouched(self, tmp_path):
        """Positive control: a normal tool call passes the boundary unchanged."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM_A, None)
        ws = Path(str(agent.config.workspace))

        call_idx = [0]

        async def _stream(*, config=None, system=None, messages=None,
                          tools=None, model="test", thinking=None,
                          cache_ttl=None, **kw):
            call_idx[0] += 1
            if call_idx[0] == 1:
                tc = ToolCall(id="tc_clean", name="file_write",
                              input={"path": "clean_s6.md",
                                     "content": "real payload here"})
                yield StreamEvent(type="tool_done", tool_index=0, tool_call=tc)
                yield StreamEvent(
                    type="done",
                    response=Response(content="", tool_calls=[tc], model=model,
                                      usage=Usage(input_tokens=10, output_tokens=5),
                                      stop_reason="tool_use"),
                    stop_reason="tool_use", model=model)
            else:
                final = "done"
                yield StreamEvent(type="text", content=final)
                yield StreamEvent(
                    type="done",
                    response=Response(content=final, model=model,
                                      usage=Usage(input_tokens=10, output_tokens=5),
                                      stop_reason="end_turn"),
                    stop_reason="end_turn", model=model)

        with patch("openalph.agent.stream", side_effect=_stream):
            await agent.handle_input("work", room_id=ROOM_A, callbacks=cb)

        assert (ws / "clean_s6.md").read_text() == "real payload here", \
            "clean tool call must execute normally"
        history = agent.history(ROOM_A)
        assert not any(STEERING_NEEDLE in str(m.get("content", ""))
                       for m in history if m.get("role") == "tool"), \
            "sentry must not fire on a clean call"
