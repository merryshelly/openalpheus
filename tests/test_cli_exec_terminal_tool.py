"""Tests for the exec TERMINAL-TOOL mechanism (Stigmergy Decision 18,
bead workspace-e2uh.152).

The station-contract primitive: `openalph exec --submit-schema <path>` loads a
forced-tool-shaped JSON schema ({"name", "strict", "description",
"input_schema"} — the same shape as a BUILTIN_TOOLS entry plus `strict`) and
exposes the named tool to the model for this run. A call to that tool is
TERMINAL: the harness executes no tool implementation (there is none — the
tool acts on nothing), captures the arguments dict as the episode's return
value, and ends the loop cleanly (status "done", exit 0 — equivalent to a
natural turn end, NOT an error). The captured arguments become a top-level
`"result"` field on the exec result JSON line (additive: every existing field
and exit code is untouched), and the terminal call appears in `tool_trace`
like any tool call (provenance). When the model ends WITHOUT calling the
terminal tool, behavior is exactly today's — no `result` field; the CALLER
decides whether a missing result is a failure.

Determinism/guards: the terminal tool is per-run state (registered at exec
start on the fresh Agent, never cached across runs); the flag composes with
--tools (union), --model, --system-prompt-file, --max-turns; the iteration-cap
sentinel still fires if the model burns all turns before submitting; deny/
ceiling paths are unchanged (a relay deny is not a terminal call).

Written RED-first. Two levels, both without network:

* CLI level (TestExecSubmitSchema*): cmd_exec with the Agent symbol stubbed —
  same harness as tests/test_cli_exec.py (stubbed Agent, real
  load_agent_config patch, exactly-one-JSON-line stdout discipline). Pins flag
  parsing, schema loading/fail-loud, the union with --tools, the `result`
  field (present iff the agent reports a terminal submit), and per-run
  registration on the stubbed Agent.

* Agent level (TestAgentTerminalTool*): a REAL Agent (test_agent_tools.py
  style) with openalph.agent.stream/complete/execute_tool mocked. Pins the
  loop mechanics at the tool-dispatch site: terminal call ends the loop
  (exactly one model call), `strict` threading to the provider, terminal
  provenance in tool_trace + history, the text-only path (no submit), --tools
  composition (a normal builtin and the terminal tool across turns), and
  cap-before-submit (sentinel + forced summary unchanged, no terminal
  handling).
"""

import asyncio
import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import AsyncMock, patch

import pytest

from openalph.provider import Response, StreamEvent, ToolCall, Usage
from openalph.tools import ToolDef, ToolResult

from test_cli_exec import make_config, make_agent_stub, parse_single_json, run_exec


# The submit_validation schema — the forced-tool shape the spec cites as the
# canonical example (stigmergy critic substrate): name + strict + description
# + input_schema. Tests write a file in this exact shape.
SUBMIT_SCHEMA = {
    "name": "submit_validation",
    "strict": True,
    "description": "Submit the station's validation verdict.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string"},
            "score": {"type": "integer"},
        },
        "required": ["verdict", "score"],
        "additionalProperties": False,
    },
}


def write_schema(tmp_path, *, name="submit_validation", strict=True,
                 description=None, input_schema=None):
    """Write a forced-tool-shaped schema file and return its path.

    Defaults to SUBMIT_SCHEMA; the fail-loud tests pass explicit
    name/input_schema omissions to exercise each validation branch.
    """
    schema = {
        "name": name,
        "strict": strict,
        "description": description if description is not None
        else SUBMIT_SCHEMA["description"],
        "input_schema": input_schema if input_schema is not None
        else SUBMIT_SCHEMA["input_schema"],
    }
    p = tmp_path / f"submit_{name}.json"
    p.write_text(json.dumps(schema))
    return p


# ---------------------------------------------------------------------------
# CLI: flag parsing
# ---------------------------------------------------------------------------


class TestExecSubmitSchemaParse:
    def test_default_none(self):
        """Without the flag, args.submit_schema is None (nothing changes)."""
        from openalph.cli import parse_args

        args = parse_args(["exec", "--agent", "w", "--task-file", "p"])
        assert args.submit_schema is None

    def test_flag_parsed(self):
        from openalph.cli import parse_args

        args = parse_args(["exec", "--agent", "w", "--task-file", "p",
                           "--submit-schema", "/path/to/schema.json"])
        assert args.submit_schema == "/path/to/schema.json"

    def test_flag_composes_with_all_other_flags(self):
        """Decision 18: the flag composes with --tools, --model,
        --system-prompt-file, --max-turns (all parse together, no
        interference)."""
        from openalph.cli import parse_args

        args = parse_args([
            "exec", "--agent", "w", "--task-file", "-",
            "--model", "blackwell/qwen38-27b-fp8",
            "--max-turns", "40", "--tools", "shell,file_read",
            "--system-prompt-file", "/path/artifact.md",
            "--submit-schema", "/path/schema.json",
        ])
        assert args.model == "blackwell/qwen38-27b-fp8"
        assert args.max_turns == 40
        assert args.tools == "shell,file_read"
        assert args.system_prompt_file == "/path/artifact.md"
        assert args.submit_schema == "/path/schema.json"


# ---------------------------------------------------------------------------
# CLI: schema loading (fail loud, burn nothing)
# ---------------------------------------------------------------------------


class TestExecSubmitSchemaLoad:
    def test_valid_schema_loads(self, tmp_path):
        """Happy path: forced-tool-shaped JSON -> (name, ToolDef) where the
        ToolDef carries the schema's input_schema verbatim (the provider
        grammar-constrains required fields) and a config marking it terminal
        with the schema's `strict` honored."""
        from openalph.cli import _exec_load_submit_schema

        p = write_schema(tmp_path)
        name, td = _exec_load_submit_schema(str(p))
        assert name == "submit_validation"
        assert isinstance(td, ToolDef)
        assert td.name == "submit_validation"
        assert td.parameters == SUBMIT_SCHEMA["input_schema"]
        assert td.description == SUBMIT_SCHEMA["description"]
        assert td.config.get("terminal") is True
        assert td.config.get("strict") is True

    def test_strict_false_preserved(self, tmp_path):
        """`strict` is the schema's, not an operator assumption: a
        non-strict schema registers a non-strict terminal tool."""
        from openalph.cli import _exec_load_submit_schema

        p = write_schema(tmp_path, strict=False)
        _, td = _exec_load_submit_schema(str(p))
        assert td.config.get("strict") is False

    def test_missing_file_fails_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            from openalph.cli import _exec_load_submit_schema
            _exec_load_submit_schema(str(tmp_path / "nope.json"))

    def test_not_json_fails_loud(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("this is not json{")
        with pytest.raises(ValueError):
            from openalph.cli import _exec_load_submit_schema
            _exec_load_submit_schema(str(p))

    def test_non_object_json_fails_loud(self, tmp_path):
        p = tmp_path / "arr.json"
        p.write_text("[1, 2, 3]")
        with pytest.raises(ValueError):
            from openalph.cli import _exec_load_submit_schema
            _exec_load_submit_schema(str(p))

    def test_missing_name_fails_loud(self, tmp_path):
        p = tmp_path / "noname.json"
        p.write_text(json.dumps({"strict": True,
                                 "description": "d",
                                 "input_schema": {"type": "object"}}))
        with pytest.raises(ValueError):
            from openalph.cli import _exec_load_submit_schema
            _exec_load_submit_schema(str(p))

    def test_missing_input_schema_fails_loud(self, tmp_path):
        p = tmp_path / "nois.json"
        p.write_text(json.dumps({"name": "submit_x", "strict": True,
                                 "description": "d"}))
        with pytest.raises(ValueError):
            from openalph.cli import _exec_load_submit_schema
            _exec_load_submit_schema(str(p))


# ---------------------------------------------------------------------------
# CLI: install, exposure, union with --tools
# ---------------------------------------------------------------------------


class TestExecSubmitSchemaInstall:
    def test_no_schema_no_terminal_tool(self, tmp_path):
        """Without --submit-schema the agent gets NO terminal-tool state —
        the pre-Decision-18 byte-for-byte behavior."""
        config = make_config(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "t.md"
        task.write_text("t")

        run_exec(["exec", "--agent", "w", "--task-file", str(task)],
                 config=config, agent=agent)
        # No terminal tool installed: no tool in the inventory carries the
        # terminal config (a MagicMock auto-vivifies attribute access, so
        # probe the INSTALL surface — agent.tools — not hasattr).
        assert all(not (getattr(t, "config", None) or {}).get("terminal")
                   for t in agent.tools)

    def test_terminal_tool_registered_on_agent(self, tmp_path):
        """--submit-schema registers (name, terminal ToolDef) on the FRESH
        agent (per-run state, never cached across runs): name from the
        schema's `name` field, ToolDef with the schema's input_schema and a
        terminal config."""
        config = make_config(tmp_path)
        p = write_schema(tmp_path)
        task = tmp_path / "t.md"
        task.write_text("t")

        built = {}

        def _capture(cfg):
            agent = make_agent_stub()
            built["agent"] = agent
            return agent

        with (
            patch("openalph.cli.load_agent_config", return_value=config),
            patch("openalph.agent.Agent", side_effect=_capture),
            redirect_stdout(StringIO()), redirect_stderr(StringIO()),
        ):
            with pytest.raises(SystemExit):
                from openalph.cli import main
                main(["exec", "--agent", "w", "--task-file", str(task),
                      "--submit-schema", str(p)])
        agent = built["agent"]
        assert agent._terminal_tool is not None
        name, td = agent._terminal_tool
        assert name == "submit_validation"
        assert td.name == "submit_validation"
        assert td.parameters == SUBMIT_SCHEMA["input_schema"]
        assert td.config.get("terminal") is True
        assert td.config.get("strict") is True

    def test_exposed_to_model_via_tools(self, tmp_path):
        """The named tool is EXPOSED to the model: it is appended to the
        agent's tool inventory (tools_arg goes to the provider as-is)."""
        config = make_config(tmp_path)
        p = write_schema(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "t.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 0
        names = [t.name for t in agent.tools]
        assert "submit_validation" in names
        # Appended exactly once; the rest of the inventory is untouched.
        assert names.count("submit_validation") == 1

    def test_union_with_tools_flag(self, tmp_path):
        """The flag composes with --tools as a UNION: the resolved builtins
        stay installed AND the terminal tool is appended (not a
        replacement, not a de-dup of builtins)."""
        config = make_config(tmp_path)
        p = write_schema(tmp_path)
        agent = make_agent_stub(tools=[
            ToolDef(name="shell", description="s", parameters={}, config={}),
            ToolDef(name="file_read", description="f", parameters={}, config={}),
        ])
        task = tmp_path / "t.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--tools", "shell,file_read", "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 0
        names = [t.name for t in agent.tools]
        assert "shell" in names and "file_read" in names
        assert "submit_validation" in names

    def test_union_without_tools_flag(self, tmp_path):
        """Without --tools the terminal tool is the ONLY tool installed
        (discovery is bypassed by exec design; nothing else fills
        agent.tools on a fresh stub)."""
        config = make_config(tmp_path)
        p = write_schema(tmp_path)
        agent = make_agent_stub(tools=[])
        task = tmp_path / "t.md"
        task.write_text("t")

        run_exec(["exec", "--agent", "w", "--task-file", str(task),
                  "--submit-schema", str(p)],
                 config=config, agent=agent)
        names = [t.name for t in agent.tools]
        assert names == ["submit_validation"]


# ---------------------------------------------------------------------------
# CLI: fail-loud end-to-end (stderr + exit 1, empty stdout, no model call)
# ---------------------------------------------------------------------------


class TestExecSubmitSchemaFailLoud:
    @pytest.mark.parametrize(
        "fname", ["nope.json", "bad.json", "noname.json", "nois.json"])
    def test_bad_schema_exits1_before_model_call(self, tmp_path, fname):
        """Unreadable / not-JSON / missing `name` / missing `input_schema`
        -> stderr + exit 1 BEFORE any model call: stdout stays EMPTY (no
        partial JSON line) and handle_input never runs (burn nothing)."""
        config = make_config(tmp_path)
        p = tmp_path / fname
        if fname == "nope.json":
            pass  # left nonexistent
        elif fname == "bad.json":
            p.write_text("not json at all{")
        elif fname == "noname.json":
            p.write_text(json.dumps({"strict": True, "description": "d",
                                     "input_schema": {"type": "object"}}))
        else:
            p.write_text(json.dumps({"name": "submit_x", "strict": True,
                                     "description": "d"}))
        agent = make_agent_stub()
        task = tmp_path / "t.md"
        task.write_text("t")

        stdout, stderr, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 1
        assert stdout.strip() == ""  # no partial JSON on failure
        assert fname in stderr
        agent.handle_input.assert_not_awaited()


# ---------------------------------------------------------------------------
# CLI: result field (additive — present iff a terminal submit happened)
# ---------------------------------------------------------------------------


class TestExecSubmitSchemaResult:
    def test_submit_yields_result_field_done_exit0(self, tmp_path):
        """The model called the terminal tool: result == the arguments dict,
        status stays "done", exit 0, and the terminal call appears in
        tool_trace like any tool call (provenance)."""
        config = make_config(tmp_path)
        p = write_schema(tmp_path)
        args_dict = {"verdict": "pass", "score": 97}

        async def hi(text, room_id, *, on_tool_call=None, thinking=None,
                     on_tool_intent=None, callbacks=None, **kw):
            if on_tool_call is not None:
                # The real loop fires the callback with the captured
                # arguments; the stub reports the same shape.
                await on_tool_call("c1", "submit_validation", args_dict,
                                   "submitted", False)
            agent._terminal_submit = args_dict
            return "SUBMISSION ACK"

        agent = make_agent_stub(handle_input=hi)
        task = tmp_path / "t.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 0
        obj = parse_single_json(stdout)
        assert obj["result"] == args_dict
        assert obj["status"] == "done"
        assert obj["content"] == "SUBMISSION ACK"
        # Provenance: the terminal call is in the trace, like any tool call.
        # bead .162 audit fix: trace entries gained an additive "executed" flag
        # (False only for terminal-batch not-executed calls); a terminal call is
        # by definition executed.
        assert {
            "name": "submit_validation",
            "is_error": False,
            "executed": True,
        } in obj["tool_trace"]
        # Existing fields untouched (additive contract).
        assert obj["ceiling_trip"] is None
        assert obj["deny_reason"] is None

    def test_no_submit_no_result_field(self, tmp_path):
        """The model ends WITHOUT calling the terminal tool (text-only
        stop): behavior is exactly today's — NO `result` field (the caller
        decides whether a missing result is a failure), status done,
        exit 0."""
        config = make_config(tmp_path)
        p = write_schema(tmp_path)
        agent = make_agent_stub(response_text="just text, no submission")
        task = tmp_path / "t.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 0
        obj = parse_single_json(stdout)
        assert "result" not in obj
        assert obj["status"] == "done"
        assert obj["content"] == "just text, no submission"
        assert obj["tool_trace"] == []

    def test_result_absent_without_flag(self, tmp_path):
        """Without the flag the result field never appears (the field is
        meaningless without a terminal tool; pre-Decision-18 output)."""
        config = make_config(tmp_path)
        agent = make_agent_stub(response_text="plain turn")
        task = tmp_path / "t.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task)],
            config=config, agent=agent,
        )
        assert code == 0
        obj = parse_single_json(stdout)
        assert "result" not in obj


# ---------------------------------------------------------------------------
# Agent level: the real loop (stream/complete/execute_tool mocked, no
# network) — same provider/executor stubbing style as test_agent_tools.py.
# ---------------------------------------------------------------------------


def _provider():
    from openalph.config import ProviderConfig
    return ProviderConfig(key="default", type="openai", api_key="sk-test",
                          base_url="http://127.0.0.1:18081")


def _agent(tmp_path, **kwargs):
    """A REAL Agent (empty workspace: no discovery, no identity files) with
    prompt assembly stubbed — the test_agent_tools.py real-path pattern."""
    from openalph.agent import Agent
    from openalph.config import AgentConfig

    kwargs.setdefault("max_iterations", 25)
    config = AgentConfig(
        name="terminal-agent",
        default_model="default/claude-test",
        max_tokens=8192,
        providers={"default": _provider()},
        workspace=tmp_path,
        **kwargs,
    )
    with (
        patch("openalph.agent.assemble_prompt", return_value="sp"),
        patch("openalph.agent.discover_tools", return_value=[]),
    ):
        return Agent(config)


def _text_response(text="FINAL TEXT"):
    return Response(
        content=text, tool_calls=[], model="claude-test",
        usage=Usage(input_tokens=10, output_tokens=5), stop_reason="end_turn",
    )


def _tool_use_response(tool_calls):
    return Response(
        content="", tool_calls=tool_calls, model="claude-test",
        usage=Usage(input_tokens=10, output_tokens=5), stop_reason="tool_use",
    )


def _stream_responses(responses):
    """side_effect for openalph.agent.stream: one scripted Response per model
    call. Exhausting the list raises StopIteration -> test failure, which is
    how 'the loop ended after exactly N model calls' gets pinned."""
    it = iter(responses)

    async def _stream(*args, **kwargs):
        response = next(it)
        if response.content:
            yield StreamEvent(type="text", content=response.content)
        for i, tc in enumerate(response.tool_calls):
            yield StreamEvent(type="tool_done", tool_index=i, tool_call=tc)
        yield StreamEvent(
            type="done", response=response,
            stop_reason=response.stop_reason, model=response.model,
        )
    return _stream


def _submit_call(args, cid="call_submit"):
    return ToolCall(id=cid, name="submit_validation", input=args)


def _terminal_tooldef():
    return ToolDef(name="submit_validation",
                   description="Submit the station's validation verdict.",
                   parameters=SUBMIT_SCHEMA["input_schema"],
                   config={"terminal": True, "strict": True})


def _exec_tool_stub(**results):
    """execute_tool stub keyed by tool name. The terminal tool must never
    reach here — if it does, the KeyError fails the test loudly."""

    async def _execute_tool(name, input, **kwargs):
        return ToolResult(content=results[name], is_error=False)
    return _execute_tool


class TestAgentTerminalTool:
    def test_terminal_call_ends_loop_result_is_args(self, tmp_path):
        """The model emits the terminal call: the loop ends CLEANLY after
        exactly ONE model call (a second stream pop would StopIteration),
        handle_input returns (no exception), the terminal call is recorded
        in the on_tool_call trace with its arguments, and a tool result is
        appended to history (no orphan tool_calls)."""
        agent = _agent(tmp_path)
        agent._terminal_tool = ("submit_validation", _terminal_tooldef())
        args_dict = {"verdict": "pass", "score": 97}

        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([_submit_call(args_dict)]),
                  ])),
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool", side_effect=_exec_tool_stub()),
        ):
            calls = []

            async def on_tc(call_id, name, input_data, result, is_error):
                calls.append((call_id, name, input_data, is_error))

            result = asyncio.run(agent.handle_input(
                "do it", "r1", on_tool_call=on_tc))

        assert result is not None  # returned, not raised
        # Provenance: exactly the terminal call, no errors, args captured.
        assert calls == [("call_submit", "submit_validation", args_dict, False)]
        # History: user, assistant-with-tool-call, tool-result (closed).
        hist = agent.history("r1")
        assert [m.get("role") for m in hist] == ["user", "assistant", "tool"]
        assert hist[-1]["tool_call_id"] == "call_submit"

    def test_text_only_no_submit(self, tmp_path):
        """The model answers in plain text: the natural turn-end path is
        byte-for-byte today's behavior — no terminal state is produced and
        no extra model call is made."""
        agent = _agent(tmp_path)
        agent._terminal_tool = ("submit_validation", _terminal_tooldef())

        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([_text_response("just text")])),
            patch("openalph.agent.complete", new_callable=AsyncMock),
        ):
            calls = []

            async def on_tc(call_id, name, input_data, result, is_error):
                calls.append((call_id, name, input_data, is_error))

            result = asyncio.run(agent.handle_input(
                "do it", "r1", on_tool_call=on_tc))

        assert result == "just text"
        assert calls == []

    def test_strict_threaded_to_provider(self, tmp_path):
        """With a `strict` terminal tool registered, the model call on this
        turn carries strict=True (the provider grammar-constrains the
        schema — required fields cannot be omitted; kdsn.304 call-level
        threading). Without any terminal tool strict is False — the wire
        shapes stay byte-identical to pre-Decision-18 (parity pinned in
        tests/test_forced_tool.py::TestParity)."""
        agent = _agent(tmp_path)
        agent._terminal_tool = ("submit_validation", _terminal_tooldef())
        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([_text_response("ok")]))
            as mock_stream,
            patch("openalph.agent.complete", new_callable=AsyncMock),
        ):
            asyncio.run(agent.handle_input("do it", "r1"))
        assert mock_stream.call_args.kwargs.get("strict") is True

        agent2 = _agent(tmp_path)
        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([_text_response("ok")]))
            as mock_stream2,
            patch("openalph.agent.complete", new_callable=AsyncMock),
        ):
            asyncio.run(agent2.handle_input("do it", "r1"))
        assert mock_stream2.call_args.kwargs.get("strict") is False

    def test_per_run_state_two_turns(self, tmp_path):
        """Per-run determinism: the SAME agent (same room, warm history)
        runs a terminal turn, the registration is cleared (a fresh run
        starts unregistered — exec's contract), then a plain turn — the
        second turn behaves exactly as a pre-Decision-18 turn (no strict
        kwarg, no terminal handling)."""
        agent = _agent(tmp_path)
        agent._terminal_tool = ("submit_validation", _terminal_tooldef())
        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([_submit_call(
                          {"verdict": "pass", "score": 1})]),
                  ])),
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool", side_effect=_exec_tool_stub()),
        ):
            asyncio.run(agent.handle_input("turn one", "r1"))

        # The exec contract: registration is per RUN; a fresh run starts
        # unregistered.
        agent._terminal_tool = None
        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([_text_response("plain")]))
            as mock_stream,
            patch("openalph.agent.complete", new_callable=AsyncMock),
        ):
            result = asyncio.run(agent.handle_input("turn two", "r1"))
        assert result == "plain"
        # Unregistered: no strict threading (byte-parity wire, as above).
        assert mock_stream.call_args.kwargs.get("strict") is False

    def test_tools_composition_normal_then_terminal(self, tmp_path):
        """--tools composition: a NORMAL builtin (shell) and the terminal
        tool are both exposed; the model calls shell on turn 1 (real
        execution, result fed back) and submit_validation on turn 2 —
        exactly two model calls, the loop ends on the terminal call, and
        the trace shows BOTH calls in order."""
        agent = _agent(tmp_path)
        # Mirror cmd_exec: the resolved --tools builtins are installed AND
        # the terminal tool is APPENDED (union).
        _tdef = _terminal_tooldef()
        agent.tools = [ToolDef(name="shell", description="s",
                               parameters={}, config={}), _tdef]
        agent._terminal_tool = ("submit_validation", _tdef)
        args_dict = {"verdict": "pass", "score": 42}

        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([ToolCall(id="c1", name="shell",
                                                   input={"command": "ls"})]),
                      _tool_use_response([_submit_call(args_dict)]),
                  ])) as mock_stream,
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool",
                  side_effect=_exec_tool_stub(shell="file1\nfile2")),
        ):
            calls = []

            async def on_tc(call_id, name, input_data, result, is_error):
                calls.append((call_id, name, input_data, is_error))

            asyncio.run(agent.handle_input(
                "inspect then submit", "r1", on_tool_call=on_tc))

        # BOTH tools were exposed to the provider on EVERY model call
        # (union with --tools, wire level).
        for call_kwargs in mock_stream.call_args_list:
            sent = {t.name for t in call_kwargs.kwargs["tools"]}
            assert sent == {"shell", "submit_validation"}
        # shell executed (turn 1), terminal captured (turn 2), loop ended.
        assert calls == [
            ("c1", "shell", {"command": "ls"}, False),
            ("call_submit", "submit_validation", args_dict, False),
        ]
        # History: user, asst(shell), tool(shell result), asst(submit),
        # tool(submit result) — both tool calls closed, no orphan.
        hist = agent.history("r1")
        assert [m.get("role") for m in hist] == [
            "user", "assistant", "tool", "assistant", "tool"]
        assert hist[2]["tool_call_id"] == "c1"
        assert hist[-1]["tool_call_id"] == "call_submit"

    def test_cap_before_submit_unchanged(self, tmp_path):
        """The model burns ALL turns WITHOUT submitting (max_iterations=1):
        the iteration-cap path is UNCHANGED — the sentinel is appended, the
        forced no-tools summary call happens (tools=None, thinking=off), and
        the turn ends on the summary. Because the terminal tool was never
        called, NO submit is captured (the caller sees no `result`). The CLI
        maps the sentinel to ceiling_trip=driver_turns (pinned at CLI level
        in test_cli_exec.py); here the agent-level contract is pinned.

        (A terminal call on the final turn would end the loop cleanly on
        THAT call — see test_terminal_call_ends_loop_result_is_args — so
        'cap before submit' is, by construction, the model calling a NORMAL
        tool and exhausting the budget.)"""
        agent = _agent(tmp_path, max_iterations=1)
        agent._terminal_tool = ("submit_validation", _terminal_tooldef())
        agent.tools = [ToolDef(name="shell", description="s",
                               parameters={}, config={})]
        # Two model calls: the capped NORMAL tool call (not the terminal
        # tool), then the forced no-tools summary.
        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([ToolCall(id="c1", name="shell",
                                                   input={"command": "ls"})]),
                      _text_response("summary of partial work"),
                  ])) as mock_stream,
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool",
                  side_effect=_exec_tool_stub(shell="file1")),
        ):
            calls = []

            async def on_tc(call_id, name, input_data, result, is_error):
                calls.append((call_id, name, input_data, is_error))

            result = asyncio.run(agent.handle_input(
                "spin", "r1", on_tool_call=on_tc))

        # Turn ends on the forced summary (cap fired), NOT on a submit.
        assert result == "summary of partial work"
        # The sentinel is in history (the CLI's structural ceiling check).
        sentinel = [m for m in agent.history("r1")
                    if isinstance(m.get("content"), str)
                    and "[SYSTEM: Tool call limit reached." in m["content"]]
        assert sentinel, "cap sentinel missing from history"
        # Only the normal shell call happened — NO terminal call, so NO
        # submit was captured (exec would emit no `result` field).
        assert calls == [("c1", "shell", {"command": "ls"}, False)]
        assert getattr(agent, "_terminal_submit", None) is None
        # The forced summary call is no-tools (today's exact form).
        summary_kwargs = mock_stream.call_args_list[1].kwargs
        assert summary_kwargs["tools"] is None
        assert summary_kwargs["thinking"] == "off"
