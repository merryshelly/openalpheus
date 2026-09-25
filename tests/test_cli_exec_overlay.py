"""Tests for the declare-done exec OVERLAY MERGE (bead workspace-kdsn.350.4).

Design contract: memory/projects/openalph/declare-done/design-memo.md §6
(SB-ratified) + the orchestrator's ratified rulings for this bead:

- ONE wire name. `openalph exec --submit-schema FILE` exposes exactly ONE
  terminal tool, named `declare_done`, whose input_schema IS the run schema
  (plus its description). No second ToolDef is appended; the old
  `agent.tools.append(terminal_tool[1])` append path is retired for schema
  runs. If the run's tool set already carries a core zero-payload
  declare_done (workspace toml / --tools), the merged overlay ToolDef
  REPLACES it in place (overlay merge, memo §6).
- Camp-B exec preserved: a schema run in a workspace WITHOUT declare_done
  (or with --tools given) still gets the synthesized per-run declare_done
  (today's submit_result behavior, renamed).
- Coerce-then-validate-then-retry at the terminal intercept: ds4-leniency
  normalization (string "true"/"false" -> bool; stringified-JSON -> object /
  list — the cairn _normalize_submit_result pattern, imported per the
  spike step3 postmortem) runs BEFORE schema validation (required fields,
  declared property types). On failure a typed ERROR ToolResult names the
  offending field(s) + expected type; the turn does NOT end,
  `_terminal_submit` stays unset, no `declared` booking — the loop
  continues and a valid re-call ends it with the COERCED dict captured.
  Repetition is bounded by the iteration cap; NO interaction with the
  one-shot undeclared-text-end corrective (separate counters).
- Overlay undeclared = exit 1: a schema run that ends WITHOUT a successful
  terminal declaration (text ending, cap, corrective-exhausted) reports
  status "failed", exit 1, NO `result` key; `conclusion` still obeys the R4
  omission contract (absent unless status=="done"); the cause rides the
  existing `detail` field. Loop-observed endings (task-failure 1 / infra 2 /
  wedged 3) keep their existing codes.
- Zero-payload runs unchanged: no --submit-schema means declare_done{} as
  today; non-schema runs ending by undeclared text keep exit 0 / done /
  "undeclared" (diagnostics, not a gate).
- strict threads through overlay runs exactly as today (the schema file's
  `strict` flag -> the provider's strict kwarg).
- Ledger continuity: a valid overlay declaration books `declared` exactly as
  a zero-payload declaration does (same dispatch-site marker).

Written RED-first. Two levels, no network:
* CLI level: cmd_exec with the Agent symbol stubbed (test_cli_exec harness)
  — pins the wire shape (what the model is shown), the loader, and the
  result-JSON / exit-code mapping.
* Agent level: a REAL Agent (test_cli_exec_terminal_tool harness) with
  stream/complete/execute_tool mocked — pins the intercept mechanics
  (coercion, validation error loop, batch close, strict threading,
  corrective interplay).
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.provider import Response, ToolCall, Usage
from openalph.tools import ToolDef

from test_cli_exec import (
    CAP_SENTINEL,
    make_agent_stub,
    make_config,
    parse_single_json,
    run_exec,
)
from test_cli_exec_terminal_tool import (
    SUBMIT_SCHEMA,
    _agent,
    _exec_tool_stub,
    _stream_responses,
    _text_response,
    _tool_use_response,
    write_schema,
)


# The Cairn-class run schema (spike step3: accepted + data). This is the
# shape that motivated the coercion import: ds4 served accepted as the
# string "true" and data as stringified JSON.
CAIRN_SCHEMA = {
    "name": "submit_result",
    "strict": True,
    "description": "Submit the run result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "accepted": {"type": "boolean"},
            "data": {"type": "object"},
        },
        "required": ["accepted", "data"],
        "additionalProperties": False,
    },
}


def write_cairn_schema(tmp_path, *, name=None, strict=None,
                       input_schema=None):
    """Write the CAIRN_SCHEMA file (with optional field overrides)."""
    schema = {
        "name": name if name is not None else CAIRN_SCHEMA["name"],
        "strict": CAIRN_SCHEMA["strict"] if strict is None else strict,
        "description": CAIRN_SCHEMA["description"],
        "input_schema": input_schema if input_schema is not None
        else CAIRN_SCHEMA["input_schema"],
    }
    p = tmp_path / "submit_result.json"
    p.write_text(json.dumps(schema))
    return p


def _merged_declare_done(schema=CAIRN_SCHEMA):
    """The merged wire ToolDef a schema run must expose: named
    declare_done, run schema as input_schema, terminal config with the
    schema's own strict."""
    return ToolDef(
        name="declare_done",
        description=schema["description"],
        parameters=schema["input_schema"],
        config={"terminal": True, "strict": bool(schema.get("strict")),
                "run_schema_name": schema["name"]},
    )


def _declare_call(args, cid="call_declare"):
    return ToolCall(id=cid, name="declare_done", input=args)


def _text_tool_response(text, tool_calls, stop_reason="tool_use"):
    """A response carrying BOTH reply text and tool calls (the
    terminal-tool harness's _text_response is text-only)."""
    return Response(
        content=text, tool_calls=tool_calls, model="claude-test",
        usage=Usage(input_tokens=10, output_tokens=5), stop_reason=stop_reason,
    )


def _shell_def():
    return ToolDef(name="shell", description="s", parameters={}, config={})


def _overlay_agent(tmp_path, *, schema=CAIRN_SCHEMA, extra_tools=()):
    """A REAL Agent wired the way cmd_exec must wire a schema run: the
    merged declare_done on the wire (plus any --tools builtins) and
    agent._terminal_tool = ("declare_done", merged_def)."""
    agent = _agent(tmp_path)
    merged = _merged_declare_done(schema)
    agent.tools = list(extra_tools) + [merged]
    agent._terminal_tool = ("declare_done", merged)
    agent._terminal_submit = None
    return agent


# ---------------------------------------------------------------------------
# CLI: wire shape — exactly ONE terminal tool, named declare_done
# ---------------------------------------------------------------------------


class TestOverlayWireShape:
    def test_one_tool_on_wire_named_declare_done(self, tmp_path):
        """Ruling 1 — the model sees ONE tool named declare_done whose
        input_schema IS the run schema (run fields present in the schema
        the model was shown); the schema's own name never reaches the
        wire; agent._terminal_tool's name is declare_done."""
        config = make_config(tmp_path)
        p = write_cairn_schema(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "t.md"
        task.write_text("t")

        # The stub never declares -> the run ends undeclared: exit 1 /
        # failed (ruling 5) — the wire is still installed pre-turn, which
        # is what this test pins.
        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 1
        names = [t.name for t in agent.tools]
        assert names.count("declare_done") == 1, f"wire names: {names}"
        assert CAIRN_SCHEMA["name"] not in names, \
            f"schema name leaked to the wire: {names}"
        td = agent.tools[names.index("declare_done")]
        assert td.parameters == CAIRN_SCHEMA["input_schema"]
        assert td.description == CAIRN_SCHEMA["description"]
        assert td.config.get("terminal") is True
        # The run fields are what the model was shown.
        assert set(td.parameters["properties"]) == {"accepted", "data"}
        assert td.parameters["required"] == ["accepted", "data"]
        # One ToolDef, one terminal: the per-run state points at the same
        # merged def, under the canonical name.
        tt = agent._terminal_tool
        assert tt is not None
        assert tt[0] == "declare_done"
        assert tt[1] is td

    def test_merge_replaces_discovered_declare_done(self, tmp_path):
        """Overlay MERGE (memo §6): when the run's tool set already carries
        the core zero-payload declare_done, the merged overlay def
        REPLACES it in place — still exactly one declare_done on the wire,
        now carrying the run schema; the other tools are untouched."""
        config = make_config(tmp_path)
        p = write_cairn_schema(tmp_path)
        zero = ToolDef(
            name="declare_done",
            description="core zero-payload",
            parameters={"type": "object", "properties": {}},
            config={},
        )
        agent = make_agent_stub(tools=[_shell_def(), zero])
        task = tmp_path / "t.md"
        task.write_text("t")

        # The stub never declares -> exit 1 (ruling 5); the wire merge is
        # installed pre-turn, which is what this test pins.
        _, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 1
        names = [t.name for t in agent.tools]
        assert names.count("declare_done") == 1, f"wire names: {names}"
        td = agent.tools[names.index("declare_done")]
        assert td.parameters == CAIRN_SCHEMA["input_schema"]
        assert td.config.get("terminal") is True
        assert "shell" in names

    def test_tools_flag_with_declare_done_merges(self, tmp_path):
        """--tools carrying declare_done composes with --submit-schema as a
        MERGE: one declare_done (overlay params), the other resolved
        builtins intact, no second terminal tool."""
        config = make_config(tmp_path)
        p = write_cairn_schema(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "t.md"
        task.write_text("t")

        # The stub never declares -> exit 1 (ruling 5); the merged wire is
        # installed pre-turn, which is what this test pins.
        _, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--tools", "shell,declare_done", "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 1
        names = [t.name for t in agent.tools]
        assert names.count("declare_done") == 1, f"wire names: {names}"
        assert "shell" in names
        assert CAIRN_SCHEMA["name"] not in names
        td = agent.tools[names.index("declare_done")]
        assert td.parameters == CAIRN_SCHEMA["input_schema"]

    def test_loader_returns_merged_declare_done(self, tmp_path):
        """The loader keeps its fail-loud validation but returns the merged
        wire def: name declare_done, run input_schema verbatim, description
        preserved, terminal config with the schema's strict, and the
        original schema name retained as provenance only."""
        from openalph.cli import _exec_load_submit_schema

        p = write_cairn_schema(tmp_path)
        name, td = _exec_load_submit_schema(str(p))
        assert name == "declare_done"
        assert td.name == "declare_done"
        assert td.parameters == CAIRN_SCHEMA["input_schema"]
        assert td.description == CAIRN_SCHEMA["description"]
        assert td.config.get("terminal") is True
        assert td.config.get("strict") is True
        assert td.config.get("run_schema_name") == CAIRN_SCHEMA["name"]

    def test_strict_false_threads_through_wire(self, tmp_path):
        """Ruling 6 — strict comes from the run schema file's `strict` flag
        and lands on the merged wire def's config (the agent reads
        _terminal_strict from there, exactly as today)."""
        from openalph.cli import _exec_load_submit_schema

        p = write_schema(tmp_path, name="submit_result", strict=False,
                         input_schema=CAIRN_SCHEMA["input_schema"])
        name, td = _exec_load_submit_schema(str(p))
        assert name == "declare_done"
        assert td.name == "declare_done"
        assert td.config.get("strict") is False
        assert td.config.get("terminal") is True


# ---------------------------------------------------------------------------
# Agent level: coerce-then-validate-then-retry at the terminal intercept
# ---------------------------------------------------------------------------


class TestOverlayAgentLoop:
    def test_coerced_declaration_captured(self, tmp_path):
        """Pins b+c — the model submits the ds4-lenient shape (accepted as
        the string "true", data as stringified JSON): exactly ONE model
        call ends the loop, the COERCED dict is what gets captured as the
        run result, and the turn books `declared` (ledger continuity —
        same marker as a zero-payload declaration)."""
        agent = _overlay_agent(tmp_path)
        raw = {"accepted": "true", "data": '{"k": 1}'}

        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([_declare_call(raw)]),
                  ])),
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool", side_effect=_exec_tool_stub()),
        ):
            calls = []

            async def on_tc(call_id, name, input_data, result, is_error):
                calls.append((call_id, name, is_error))

            asyncio.run(agent.handle_input(
                "go", "r1", on_tool_call=on_tc))

        assert agent._terminal_submit == {"accepted": True, "data": {"k": 1}}
        assert agent.last_turn_declaration("r1") == "declared"
        # Exactly one model call (a second scripted pop would StopIteration).
        assert calls == [("call_declare", "declare_done", False)]
        # No orphan tool_calls: the captured call is closed in history.
        hist = agent.history("r1")
        assert [m.get("role") for m in hist] == ["user", "assistant", "tool"]
        assert hist[-1]["tool_call_id"] == "call_declare"

    def test_invalid_payload_typed_error_loop_continues(self, tmp_path):
        """Pin d — un-coercible value + missing required field: the turn
        does NOT end. The offending call gets a typed ERROR ToolResult
        naming the field(s) and the expected type, no `_terminal_submit`,
        no `declared` booking; the loop continues and a LATER valid re-call
        ends it with the valid payload captured (exactly 2 model calls)."""
        agent = _overlay_agent(tmp_path)
        bad = {"accepted": "maybe"}  # un-coercible bool; "data" missing
        good = {"accepted": True, "data": {}}

        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([_declare_call(bad, cid="call_bad")]),
                      _tool_use_response([_declare_call(good, cid="call_good")]),
                  ])),
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool", side_effect=_exec_tool_stub()),
        ):
            calls = []

            async def on_tc(call_id, name, input_data, result, is_error):
                calls.append((call_id, name, is_error))

            asyncio.run(agent.handle_input(
                "go", "r1", on_tool_call=on_tc))

        # The VALID re-call ends it; the coerced/clean dict is captured.
        assert agent._terminal_submit == good
        assert agent.last_turn_declaration("r1") == "declared"
        assert calls == [
            ("call_bad", "declare_done", True),
            ("call_good", "declare_done", False),
        ]
        tool_results = [m for m in agent.history("r1") if m.get("role") == "tool"]
        assert {m["tool_call_id"] for m in tool_results} == {
            "call_bad", "call_good"}
        bad_res = next(m for m in tool_results if m["tool_call_id"] == "call_bad")
        good_res = next(m for m in tool_results if m["tool_call_id"] == "call_good")
        # Typed error: names BOTH offending fields and the expected types.
        assert bad_res.get("is_error") is True
        content = str(bad_res.get("content", ""))
        assert "accepted" in content
        assert "boolean" in content
        assert "data" in content
        assert "object" in content
        assert good_res.get("is_error") is False

    def test_missing_required_field_named(self, tmp_path):
        """Pin d (verdict schema) — a missing REQUIRED field names the
        field and its expected type in the typed error; the re-call ends
        the turn (exactly 2 model calls)."""
        agent = _overlay_agent(tmp_path, schema=SUBMIT_SCHEMA)
        bad = {"verdict": "pass"}  # required "score" missing
        good = {"verdict": "pass", "score": 97}

        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([_declare_call(bad, cid="call_bad")]),
                      _tool_use_response([_declare_call(good, cid="call_good")]),
                  ])),
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool", side_effect=_exec_tool_stub()),
        ):
            asyncio.run(agent.handle_input("go", "r1"))

        assert agent._terminal_submit == good
        assert agent.last_turn_declaration("r1") == "declared"
        tool_results = [m for m in agent.history("r1") if m.get("role") == "tool"]
        bad_res = next(m for m in tool_results if m["tool_call_id"] == "call_bad")
        assert bad_res.get("is_error") is True
        content = str(bad_res.get("content", ""))
        assert "score" in content
        assert "integer" in content

    def test_first_call_wins_batch_closed_no_orphans(self, tmp_path):
        """Pin g — first-call-wins batch semantics preserved for the merged
        terminal: declare_done(valid) + a shell call in ONE batch end the
        turn on exactly ONE model call; the batchmate is provenanced +
        closed (no orphan) but NOT executed; the declare payload is what
        gets captured."""
        agent = _overlay_agent(tmp_path, extra_tools=[_shell_def()])
        good = {"accepted": False, "data": {"n": 2}}

        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([
                          _declare_call(good, cid="call_declare"),
                          ToolCall(id="call_shell", name="shell",
                                   input={"command": "ls"}),
                      ]),
                  ])),
            patch("openalph.agent.complete", new_callable=AsyncMock),
            # No "shell" key in the stub: an attempted execution would
            # KeyError and fail the test loudly.
            patch("openalph.agent.execute_tool", side_effect=_exec_tool_stub()),
        ):
            asyncio.run(agent.handle_input(
                "inspect then submit", "r1"))

        assert agent._terminal_submit == good
        assert agent.last_turn_declaration("r1") == "declared"
        tool_results = [m for m in agent.history("r1") if m.get("role") == "tool"]
        assert {m["tool_call_id"] for m in tool_results} == {
            "call_declare", "call_shell"}
        shell_res = next(m for m in tool_results
                         if m["tool_call_id"] == "call_shell")
        assert "not executed" in str(shell_res.get("content", ""))

    def test_strict_threaded_to_provider(self, tmp_path):
        """Pin f (agent level) — the merged def's config strict (from the
        run schema file) threads to the provider's strict kwarg, True and
        False alike (ruling 6)."""
        agent = _overlay_agent(tmp_path)
        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([_declare_call(
                          {"accepted": True, "data": {}})])]))
            as mock_stream,
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool", side_effect=_exec_tool_stub()),
        ):
            asyncio.run(agent.handle_input("go", "r1"))
        assert mock_stream.call_args.kwargs.get("strict") is True

        schema2 = dict(CAIRN_SCHEMA)
        schema2["strict"] = False
        agent2 = _overlay_agent(tmp_path, schema=schema2)
        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([_declare_call(
                          {"accepted": True, "data": {}})])]))
            as mock_stream2,
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool", side_effect=_exec_tool_stub()),
        ):
            asyncio.run(agent2.handle_input("go", "r1"))
        assert mock_stream2.call_args.kwargs.get("strict") is False

    def test_camp_b_wire_composition(self, tmp_path):
        """Pin h (agent level) — a schema run whose workspace carries NO
        declare_done: the synthesized merged declare_done + a --tools
        builtin (shell) are BOTH on the wire on every model call, the
        run fields are in the schema the model was shown, and the shell
        call runs for real before the declaration ends the turn (exactly
        two model calls)."""
        shell = _shell_def()
        agent = _overlay_agent(tmp_path, extra_tools=[shell])
        good = {"accepted": True, "data": {"hop": 3}}

        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([ToolCall(id="c1", name="shell",
                                                   input={"command": "ls"})]),
                      _tool_use_response([_declare_call(good)]),
                  ])) as mock_stream,
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool",
                  side_effect=_exec_tool_stub(shell="file1")),
        ):
            calls = []

            async def on_tc(call_id, name, input_data, result, is_error):
                calls.append((call_id, name, is_error))

            asyncio.run(agent.handle_input(
                "inspect then submit", "r1", on_tool_call=on_tc))

        for call_kwargs in mock_stream.call_args_list:
            sent = {t.name for t in call_kwargs.kwargs["tools"]}
            assert sent == {"shell", "declare_done"}, f"wire: {sent}"
            dd = next(t for t in call_kwargs.kwargs["tools"]
                      if t.name == "declare_done")
            assert set(dd.parameters["properties"]) == {"accepted", "data"}
        assert calls == [
            ("c1", "shell", False),
            ("call_declare", "declare_done", False),
        ]
        assert agent._terminal_submit == good
        assert agent.last_turn_declaration("r1") == "declared"

    def test_holdback_corrective_fires_in_overlay(self, tmp_path):
        """Pin i (carryover) — in an overlay run, an undeclared text ending
        with declare_done available still fires the hold-back corrective
        (existing .350.3 behavior): the text is held, ONE corrective, the
        model declares, the turn returns the final text (exactly 2 model
        calls)."""
        agent = _overlay_agent(tmp_path)
        good = {"accepted": True, "data": {}}

        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _text_response("prose, no declaration"),
                      _text_tool_response("final answer",
                                          [_declare_call(good)]),
                  ])),
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool", side_effect=_exec_tool_stub()),
        ):
            result = asyncio.run(agent.handle_input("go", "r1"))

        assert result == "final answer"
        assert agent._terminal_submit == good
        assert agent.last_turn_declaration("r1") == "declared"
        correctives = [
            m for m in agent.history("r1")
            if m.get("role") == "user"
            and "declare_done" in str(m.get("content", ""))
        ]
        assert len(correctives) == 1

    def test_corrective_independent_of_validation_retries(self, tmp_path):
        """Pin i — the payload-validation retry loop and the one-shot
        undeclared-text-end corrective are SEPARATE budgets: an invalid
        declaration (typed error, retry) followed by an undeclared text
        ending still fires the corrective, and the subsequent valid
        declaration ends the turn (exactly 3 model calls; ONE corrective;
        ONE payload error)."""
        agent = _overlay_agent(tmp_path)
        bad = {"accepted": "maybe"}
        good = {"accepted": True, "data": {}}

        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([_declare_call(bad, cid="call_bad")]),
                      _text_response("prose, no declaration"),
                      _text_tool_response("final answer",
                                          [_declare_call(good,
                                                         cid="call_good")]),
                  ])),
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool", side_effect=_exec_tool_stub()),
        ):
            result = asyncio.run(agent.handle_input("go", "r1"))

        assert result == "final answer"
        assert agent._terminal_submit == good
        assert agent.last_turn_declaration("r1") == "declared"
        correctives = [
            m for m in agent.history("r1")
            if m.get("role") == "user"
            and "declare_done" in str(m.get("content", ""))
        ]
        assert len(correctives) == 1
        tool_results = [m for m in agent.history("r1") if m.get("role") == "tool"]
        bad_res = next(m for m in tool_results if m["tool_call_id"] == "call_bad")
        assert bad_res.get("is_error") is True
        assert "accepted" in str(bad_res.get("content", ""))


# ---------------------------------------------------------------------------
# CLI: result-JSON / exit-code mapping (overlay undeclared = exit 1)
# ---------------------------------------------------------------------------


class TestOverlayExecResult:
    def test_valid_declaration_exit0_result_and_conclusion(self, tmp_path):
        """Pin b (CLI level) — a successful terminal declaration: exit 0,
        status "done", `result` = the declared payload, `conclusion`
        "declared", and the terminal call is in the tool trace under its
        canonical wire name (executed)."""
        config = make_config(tmp_path)
        p = write_cairn_schema(tmp_path)
        args_dict = {"accepted": True, "data": {"k": 1}}

        async def hi(text, room_id, *, on_tool_call=None, thinking=None,
                     on_tool_intent=None, callbacks=None, **kw):
            if on_tool_call is not None:
                await on_tool_call("c1", "declare_done", args_dict,
                                   "submitted", False)
            agent._terminal_submit = args_dict
            return "SUBMISSION ACK"

        agent = make_agent_stub(handle_input=hi)
        agent.last_turn_declaration = MagicMock(return_value="declared")
        task = tmp_path / "t.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 0
        obj = parse_single_json(stdout)
        assert obj["status"] == "done"
        assert obj["result"] == args_dict
        assert obj["conclusion"] == "declared"
        assert {
            "name": "declare_done",
            "is_error": False,
            "executed": True,
        } in obj["tool_trace"]

    def test_undeclared_text_end_failed_exit1_no_result(self, tmp_path):
        """Pin e / ruling 5 — a schema run ending by undeclared text:
        status "failed", exit 1, NO `result` key, NO `conclusion` key
        (R4 omission contract — the run did not land "done"), and the
        cause conveyed via the existing `detail` field."""
        config = make_config(tmp_path)
        p = write_cairn_schema(tmp_path)
        agent = make_agent_stub(response_text="just text, no submission")
        task = tmp_path / "t.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 1
        obj = parse_single_json(stdout)
        assert obj["status"] == "failed"
        assert "result" not in obj
        assert "conclusion" not in obj
        assert "declare_done" in obj["detail"]
        assert obj["tool_trace"] == []

    def test_ceiling_ending_keeps_existing_code(self, tmp_path):
        """Loop-observed endings keep their existing codes — a schema run
        that hits the iteration cap (structural ceiling) stays
        failed / driver_turns / exit 1, no result key (the cap is its own
        class; the overlay check must not double-classify it)."""
        config = make_config(tmp_path)
        p = write_cairn_schema(tmp_path)
        history = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "user", "content": CAP_SENTINEL + " rest of message"},
            {"role": "assistant", "content": "summary text"},
        ]
        agent = make_agent_stub(history=history, response_text="summary text")
        task = tmp_path / "t.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 1
        obj = parse_single_json(stdout)
        assert obj["status"] == "failed"
        assert obj["ceiling_trip"] == "driver_turns"
        assert "result" not in obj
        assert "conclusion" not in obj

    def test_infra_ending_keeps_code2(self, tmp_path):
        """A schema run that dies on a transport error stays infra /
        exit 2 (loop-observed class — existing code, not the overlay
        failure class), no result key."""
        import httpx

        config = make_config(tmp_path)
        p = write_cairn_schema(tmp_path)
        err = Exception("unreachable")
        err.__cause__ = httpx.ConnectError("connection refused")
        agent = make_agent_stub(handle_input=AsyncMock(side_effect=err))
        task = tmp_path / "t.md"
        task.write_text("t")

        stdout, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 2
        obj = parse_single_json(stdout)
        assert obj["status"] == "infra"
        assert "result" not in obj
        assert "conclusion" not in obj


# ---------------------------------------------------------------------------
# AUDIT REMEDIATION (3-model adversarial audit, reconciliation.md):
# hardening pins written RED-first (F-A, F-B, F-C, F-E, F-F, F-G).
# ---------------------------------------------------------------------------


def _write_loader_schema(tmp_path, input_schema, name="submit_x"):
    """Write a forced-tool-shaped schema file with the given input_schema
    (the F-A loader pins vary the INNER shape of input_schema)."""
    p = tmp_path / "loader_schema.json"
    p.write_text(json.dumps({
        "name": name, "strict": True, "description": "d",
        "input_schema": input_schema,
    }))
    return p


# ---------------------------------------------------------------------------
# F-A layer 1: the LOADER structurally validates the run schema — a
# malformed schema fails loud (ValueError, the existing _fail(...,1)
# path) instead of inverting the dispatch-site validation gate.
# ---------------------------------------------------------------------------


class TestOverlaySchemaLoaderHardening:
    def test_required_bare_string_fails_loud(self, tmp_path):
        """`"required": "accepted"` (bare string) must NOT load: the
        dispatch site would char-iterate it (the gate-inversion the
        audit demonstrated — garbage passes, real payloads fail)."""
        from openalph.cli import _exec_load_submit_schema

        p = _write_loader_schema(tmp_path, {
            "type": "object",
            "properties": {"accepted": {"type": "boolean"}},
            "required": "accepted",
        })
        with pytest.raises(ValueError):
            _exec_load_submit_schema(str(p))

    def test_required_dict_fails_loud(self, tmp_path):
        """`"required": {"a": 1}` (a mapping) must NOT load — `required`
        is a list of strings or absent, nothing else."""
        from openalph.cli import _exec_load_submit_schema

        p = _write_loader_schema(tmp_path, {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": {"a": 1},
        })
        with pytest.raises(ValueError):
            _exec_load_submit_schema(str(p))

    def test_required_non_string_member_fails_loud(self, tmp_path):
        """A `required` list with a non-string member is malformed JSON
        Schema (required: array of unique strings) — fail loud."""
        from openalph.cli import _exec_load_submit_schema

        p = _write_loader_schema(tmp_path, {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["accepted", 1],
        })
        with pytest.raises(ValueError):
            _exec_load_submit_schema(str(p))

    def test_properties_not_dict_fails_loud(self, tmp_path):
        """Non-dict `properties` (here: a bare string) must NOT load —
        today it silently no-ops the gate (no type checks, ever)."""
        from openalph.cli import _exec_load_submit_schema

        p = _write_loader_schema(tmp_path, {
            "type": "object",
            "properties": "accepted",
        })
        with pytest.raises(ValueError):
            _exec_load_submit_schema(str(p))

    def test_property_spec_not_dict_fails_loud(self, tmp_path):
        """Each property spec must be an object — a bare string spec is
        malformed and must fail loud, not be skipped."""
        from openalph.cli import _exec_load_submit_schema

        p = _write_loader_schema(tmp_path, {
            "type": "object",
            "properties": {"accepted": "boolean"},
        })
        with pytest.raises(ValueError):
            _exec_load_submit_schema(str(p))

    def test_unknown_type_keyword_fails_loud(self, tmp_path):
        """An unknown type keyword ("str" is not a JSON keyword) must NOT
        load — the gate would never type-check that field."""
        from openalph.cli import _exec_load_submit_schema

        p = _write_loader_schema(tmp_path, {
            "type": "object",
            "properties": {"accepted": {"type": "str"}},
        })
        with pytest.raises(ValueError):
            _exec_load_submit_schema(str(p))

    def test_unknown_type_in_list_fails_loud(self, tmp_path):
        """A type keyword LIST containing an unknown keyword fails loud
        (a list of known keywords is the only accepted list form)."""
        from openalph.cli import _exec_load_submit_schema

        p = _write_loader_schema(tmp_path, {
            "type": "object",
            "properties": {"accepted": {"type": ["string", "maybe"]}},
        })
        with pytest.raises(ValueError):
            _exec_load_submit_schema(str(p))

    def test_type_wrong_shape_fails_loud(self, tmp_path):
        """A dict-shaped `type` must NOT load — today it reaches the
        dispatch loop and raises TypeError (unhashable dict) on the
        first terminal call."""
        from openalph.cli import _exec_load_submit_schema

        p = _write_loader_schema(tmp_path, {
            "type": "object",
            "properties": {"accepted": {"type": {"a": 1}}},
        })
        with pytest.raises(ValueError):
            _exec_load_submit_schema(str(p))

    def test_valid_schema_with_null_and_list_types_still_loads(self, tmp_path):
        """Carryover + over-strictness guard: a valid schema — including
        a `null` keyword and a type-keyword list (both legal) — still
        loads with its input_schema verbatim."""
        from openalph.cli import _exec_load_submit_schema

        schema = {
            "type": "object",
            "properties": {
                "note": {"type": ["string", "null"]},
                "accepted": {"type": "boolean"},
            },
            "required": ["accepted"],
        }
        p = _write_loader_schema(tmp_path, schema)
        name, td = _exec_load_submit_schema(str(p))
        assert name == "declare_done"
        assert td.parameters == schema


# ---------------------------------------------------------------------------
# F-A layer 2 (defense-in-depth) + F-E + F-F + F-G: the dispatch-site
# helper NEVER raises on a degenerate schema / parse failure — it either
# returns honest errors or passes honestly.
# ---------------------------------------------------------------------------


class TestCoerceTerminalPayloadHardening:
    def test_required_bare_string_skipped_never_char_iterated(self):
        """`required` that is a bare string is SKIPPED (not char-iterated):
        the payload validates honestly — no spurious 'missing required
        field' errors for the letters of the string."""
        from openalph.agent import _coerce_terminal_payload

        out, errors = _coerce_terminal_payload(
            {"accepted": True},
            {"properties": {}, "required": "accepted"})
        assert errors is None
        assert out == {"accepted": True}

    def test_required_dict_skipped(self):
        """`required` that is a mapping is SKIPPED (never key-iterated)."""
        from openalph.agent import _coerce_terminal_payload

        out, errors = _coerce_terminal_payload({}, {"required": {"a": 1}})
        assert errors is None
        assert out == {}

    def test_required_non_string_list_skipped(self):
        """A `required` list with a non-string member is SKIPPED as a
        whole (no 'missing required field 1' nonsense)."""
        from openalph.agent import _coerce_terminal_payload

        out, errors = _coerce_terminal_payload({}, {"required": [1]})
        assert errors is None
        assert out == {}

    def test_type_wrong_shape_skipped_never_raises(self):
        """A dict-shaped property `type` is SKIPPED — no exception class
        (TypeError included) may escape the helper; the field passes
        honestly (undeclared-effective type)."""
        from openalph.agent import _coerce_terminal_payload

        out, errors = _coerce_terminal_payload(
            {"x": 5}, {"properties": {"x": {"type": {"a": 1}}}})
        assert errors is None
        assert out == {"x": 5}

    def test_type_list_with_non_str_member_skipped(self):
        """A type list containing a non-string member is SKIPPED (not a
        str, not a list of strs) — honest pass, no raise."""
        from openalph.agent import _coerce_terminal_payload

        out, errors = _coerce_terminal_payload(
            {"x": 5}, {"properties": {"x": {"type": [1, 2]}}})
        assert errors is None
        assert out == {"x": 5}

    def test_properties_not_dict_still_safe(self):
        """Non-dict `properties` degrades to 'no declared fields' (the
        existing guard) — a payload passes honestly with no checks."""
        from openalph.agent import _coerce_terminal_payload

        out, errors = _coerce_terminal_payload(
            {"a": 1}, {"properties": "a"})
        assert errors is None
        assert out == {"a": 1}

    def test_recursion_error_in_json_loads_is_rejection(self):
        """F-E — ANY exception class out of the coercion json.loads
        (here: RecursionError, via a string nested beyond the json
        parser's ~10000-level nesting cap) must land as un-coercible
        -> honest validation rejection (None, errors) — NEVER an
        exception escaping into the loop.

        Note: this interpreter's C-accelerated json parser caps nesting
        at its own S_MAX_DEPTH (~10000) and does NOT recurse into
        Python frames, so a reduced sys.setrecursionlimit cannot induce
        the RecursionError (verified empirically); depth 11000 is used
        instead, which raises RecursionError under BOTH the C and the
        pure-Python parsers. See REPORT §5 (deviation)."""
        from openalph.agent import _coerce_terminal_payload

        deep = "[" * 11000 + "]" * 11000
        schema = {"properties": {"data": {"type": "object"}}}
        out, errors = _coerce_terminal_payload({"data": deep}, schema)
        assert out is None
        assert errors, "deep-string field must be rejected (not coerced)"
        assert any("data" in e for e in errors)

    def test_any_of_object_array_coerces_stringified_array(self):
        """F-F — a property declared `["object", "array"]` (any-of) with
        a stringified ARRAY value must be coerced to the list: the
        coercion branches are independent (boolean, then object, then
        array), not an elif ladder that only tries the first listed
        non-string type."""
        from openalph.agent import _coerce_terminal_payload

        out, errors = _coerce_terminal_payload(
            {"v": "[1, 2]"},
            {"properties": {"v": {"type": ["object", "array"]}}})
        assert errors is None
        assert out == {"v": [1, 2]}

    def test_null_type_validates_none_value(self):
        """F-G — `"null"` is a satisfiable type keyword: a field
        declared `["string", "null"]` with the value None validates."""
        from openalph.agent import _coerce_terminal_payload

        out, errors = _coerce_terminal_payload(
            {"v": None},
            {"properties": {"v": {"type": ["string", "null"]}}})
        assert errors is None
        assert out == {"v": None}


# ---------------------------------------------------------------------------
# F-B: the CORE (discovered, zero-payload) declare_done keeps pre-350.4
# raw capture — non-dict ToolCall.input captured verbatim, turn ends
# declared. Overlay runs still validate (carried by the existing
# TestOverlayAgentLoop pins, which must stay green).
# ---------------------------------------------------------------------------


class TestOverlayHardeningLoop:
    def test_discovered_declare_done_non_dict_input_captured_raw(self,
                                                                  tmp_path):
        """F-B (agent level) — a DISCOVERED declare_done (no
        agent._terminal_tool — the core zero-payload path) called with a
        NON-DICT input (the provider `arguments: "null"` class) is
        captured VERBATIM as _terminal_submit and ends the turn
        declared: exactly ONE model call, no typed validation error, no
        loop continuation — pre-350.4 semantics restored."""
        agent = _agent(tmp_path)
        core = ToolDef(
            name="declare_done",
            description="Declare the turn complete (core zero-payload).",
            parameters={"type": "object", "properties": {}},
            config={},
        )
        agent.tools = [core]
        agent._terminal_submit = None

        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _tool_use_response([ToolCall(id="c1",
                                                   name="declare_done",
                                                   input="null")]),
                  ])),
            patch("openalph.agent.complete", new_callable=AsyncMock),
            patch("openalph.agent.execute_tool", side_effect=_exec_tool_stub()),
        ):
            calls = []

            async def on_tc(call_id, name, input_data, result, is_error):
                calls.append((call_id, name, is_error))

            result = asyncio.run(agent.handle_input(
                "go", "r1", on_tool_call=on_tc))

        # Captured VERBATIM: the raw non-dict value (not None, not a
        # dict, not a validation error).
        assert agent._terminal_submit == "null"
        assert agent.last_turn_declaration("r1") == "declared"
        # Exactly one model call (a second scripted pop would
        # StopIteration — the loop did NOT continue on a fake error),
        # closed without an error result.
        assert calls == [("c1", "declare_done", False)]
        hist = agent.history("r1")
        assert [m.get("role") for m in hist] == ["user", "assistant", "tool"]
        assert hist[-1]["tool_call_id"] == "c1"
        assert hist[-1].get("is_error") is False
        assert result is not None


# ---------------------------------------------------------------------------
# F-C: a single wire def under duplicate inputs —
# --tools declare_done,declare_done still yields exactly ONE
# declare_done after the overlay merge (the merged overlay def).
# ---------------------------------------------------------------------------


class TestOverlayHardeningCli:
    def test_duplicate_declare_done_tools_single_wire_def(self, tmp_path):
        """F-C — `--tools declare_done,declare_done` (a duplicated
        incoming list — _resolve_exec_tools does not de-dup) with
        --submit-schema yields EXACTLY ONE declare_done on the final
        wire: the merged overlay def (run schema as input_schema,
        terminal config) — no duplicate left behind by the
        first-occurrence replace."""
        config = make_config(tmp_path)
        p = write_cairn_schema(tmp_path)
        agent = make_agent_stub()
        task = tmp_path / "t.md"
        task.write_text("t")

        # The stub never declares -> exit 1 (ruling 5); the merged wire
        # is installed pre-turn, which is what this test pins.
        _, _, code = run_exec(
            ["exec", "--agent", "w", "--task-file", str(task),
             "--tools", "declare_done,declare_done",
             "--submit-schema", str(p)],
            config=config, agent=agent,
        )
        assert code == 1
        names = [t.name for t in agent.tools]
        assert names.count("declare_done") == 1, f"wire names: {names}"
        td = agent.tools[names.index("declare_done")]
        assert td.parameters == CAIRN_SCHEMA["input_schema"]
        assert td.config.get("terminal") is True
