"""Declare-done grammar + turn ledger — TDD RED suite (workspace-kdsn.350.2).

Design contract: memory/projects/openalph/declare-done/design-memo.md (SB-ratified
2026-09-24). This suite is the executable spec for:

  .350.3  declare_done zero-payload builtin + loop mechanics (grammar layer)
  .179    turn.started/turn.finished ledger + every-turn m.notice (ledger layer)
  (.350.4 exec overlay-merge authors its own red tests at kickoff; the CLI
          conclusion-field pins here are the shared seam.)

Core contract encoded here
--------------------------
- `declare_done` is a BUILTIN_TOOLS entry, zero-payload (properties {}), terminal
  semantics. Enabled by workspace discovery (`declare_done.toml`), NEVER
  core-registered. SB: standard discovery only.
- A response carrying a `declare_done` tool call ends the turn cleanly at the
  dispatch site (same seam the exec --submit-schema terminal tool uses today):
  the call is provenanced (counted/logged/on_tool_call'd), NO execution is
  attempted, every call in the batch gets a tool result (no orphans), the
  accompanying text is the turn's reply.
- Undeclared text-end (declare_done registered): the response is held (NOT
  returned), a fixed harness-authored corrective naming `declare_done` is
  appended as a user message, and the loop continues. The model may resume
  work, declare, or fail again. A SECOND consecutive undeclared text-end books
  `undeclared` and ends the turn. Bounded at one corrective per turn.
- Corrective is REGISTRATION-GATED: with the tool absent the old Camp-B
  behavior holds (text response returns immediately, no corrective) — the
  marker reports None and the funnel books `undeclared`.
- `agent.last_turn_declaration(room_id)` reports the turn's landing:
  "declared" | "undeclared" | "cap_exhausted" | None (unregistered / no turn).
- The forced-summary (iteration-cap) terminal path NEVER sees the corrective
  and reports "cap_exhausted".
- Ledger: every turn on every funnel books exactly one `turn.started` and one
  `turn.finished` JSONL system entry (shared turn_id; finished carries
  conclusion | elapsed_s | origin), plus ONE plain every-turn m.notice line.
  Emission is fail-soft; a stale `turn.started` is the crash signature — no
  fabricated success.
- Conclusion taxonomy: declared | undeclared | cap_exhausted | overflow |
  cancelled | error | abandoned.

Harness notes
-------------
- Agent-level pins reuse the test_cli_exec_terminal_tool scripted-stream
  pattern (StopIteration pins "exactly N model calls").
- Matrix-level pins reuse test_gapfill_trigger_dedup's make_bot (REAL
  SessionLog, mocked agent/transport).
- The real-path pin (tool-management "one lesson") runs a REAL Agent through
  the REAL `MatrixBot._build_agent_callbacks` construction path.

NOTE (streaming amendment, recorded in progress.md): hold-back governs the
RETURN value and final-message send; live-streamed deltas of a first,
undeclared response are not recalled (StreamingDelivery streams as it goes).
The corrective drives follow-through; there is no "exactly one message" pin.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.agent import Agent, ContextOverflowError
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import ProviderError, Response, StreamEvent, ToolCall, Usage
from openalph.tools import BUILTIN_TOOLS, ToolResult, discover_tools

# Reuse the proven harnesses (same tests/ dir, rootdir on sys.path).
from test_gapfill_trigger_dedup import (  # noqa: E402
    USER, ROOM_ID, drain, gapfill_page, make_bot, make_event,
    make_room, seed_prior_history,
)

ROOM_A = ROOM_ID


# ──────────────────────────────────────────────────────────────────────
# Agent-level helpers (mirrors tests/test_cli_exec_terminal_tool.py)
# ──────────────────────────────────────────────────────────────────────


def _provider():
    return ProviderConfig(key="default", type="openai", api_key="sk-test",
                          base_url="http://127.0.0.1:18081")


def _make_agent(tmp_path, *_, declare_done=True, extra_tools=(), max_iterations=25):
    """A REAL Agent on a REAL workspace. When declare_done is True the
    workspace carries declare_done.toml (standard discovery — no patching of
    discover_tools). Prompt assembly is the only stub."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(exist_ok=True)
    names = list(extra_tools) + (["declare_done"] if declare_done else [])
    for name in names:
        (tools_dir / f"{name}.toml").write_text("[config]\n")
    config = AgentConfig(
        name="dd-agent",
        default_model="default/claude-test",
        max_tokens=8192,
        providers={"default": _provider()},
        workspace=tmp_path,
        max_iterations=max_iterations,
    )
    with patch("openalph.agent.assemble_prompt", return_value="sp"):
        return Agent(config)


def _text_response(text, tool_calls=None, stop_reason="end_turn"):
    return Response(
        content=text, tool_calls=tool_calls or [], model="claude-test",
        usage=Usage(input_tokens=10, output_tokens=5), stop_reason=stop_reason,
    )


def _declare_call(args=None, cid="call_declare"):
    return ToolCall(id=cid, name="declare_done", input=args or {})


def _shell_call(cid="call_shell"):
    return ToolCall(id=cid, name="shell", input={"command": "echo hi"})


def _stream_responses(responses, calls=None):
    """side_effect for openalph.agent.stream: one scripted Response per model
    call. Exhausting the list raises StopIteration -> test failure, which is
    how 'the loop ended after exactly N model calls' gets pinned. When `calls`
    is a list, each invocation's kwargs are appended (tools=None detection)."""
    it = iter(responses)

    async def _stream(*args, **kwargs):
        if calls is not None:
            calls.append(kwargs)
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


def _exec_tool_stub(**results):
    """execute_tool stub keyed by tool name. declare_done must NEVER reach
    here — it's captured at the dispatch site; a dispatch attempt raises
    KeyError and fails the test loudly."""

    async def _execute_tool(name, input, **kwargs):
        return ToolResult(content=results[name], is_error=False)
    return _execute_tool


def _corrective_entries(agent, room_id):
    """User-role history entries that name declare_done — i.e. corrective
    injections (the real operator/user messages in these tests never mention
    the tool)."""
    return [
        m for m in agent.history(room_id)
        if m.get("role") == "user" and "declare_done" in str(m.get("content", ""))
    ]


def _tool_result_entries(agent, room_id):
    return [m for m in agent.history(room_id) if m.get("role") == "tool"]


def _drive(agent, room_id, responses, text="do the thing", calls=None, on_tool_call=None):
    with (
        patch("openalph.agent.stream",
              side_effect=_stream_responses(responses, calls=calls)),
        patch("openalph.agent.complete", new_callable=AsyncMock),
        patch("openalph.agent.execute_tool",
              side_effect=_exec_tool_stub(shell="ok")),
    ):
        return asyncio.run(
            agent.handle_input(text, room_id, on_tool_call=on_tool_call))


# ──────────────────────────────────────────────────────────────────────
# G — grammar layer (agent.py + tools/__init__.py) · bead .350.3
# ──────────────────────────────────────────────────────────────────────


class TestDeclareDoneRegistration:
    def test_declare_done_is_a_builtin(self):
        """G1 — the tool exists in the shared registry, zero-payload."""
        assert "declare_done" in BUILTIN_TOOLS
        entry = BUILTIN_TOOLS["declare_done"]
        assert entry["parameters"]["type"] == "object"
        assert entry["parameters"].get("properties", {}) == {}
        assert not entry["parameters"].get("required")
        assert isinstance(entry["description"], str) and entry["description"].strip()

    def test_discovery_enables_it(self, tmp_path):
        """G2 — standard discovery contract: declare_done.toml -> ToolDef."""
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "declare_done.toml").write_text("[config]\n")
        defs = discover_tools(tmp_path)
        names = [d.name for d in defs]
        assert "declare_done" in names
        dd = defs[names.index("declare_done")]
        assert dd.parameters.get("properties", {}) == {}


class TestDeclaredEnding:
    def test_text_plus_declare_ends_turn_cleanly(self, tmp_path):
        """G3 — one response with reply text AND declare_done{}: the turn
        returns the text after exactly ONE model call; the call is captured
        at the dispatch site (never executed), provenanced via on_tool_call,
        and no orphan tool_calls remain in history."""
        agent = _make_agent(tmp_path)
        seen = []

        async def on_tool_call(call_id, name, input_data, result, is_error):
            seen.append((call_id, name, input_data, is_error))

        result = _drive(agent, "r1", [
            _text_response("Here is your answer.",
                           tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ], on_tool_call=on_tool_call)

        assert result == "Here is your answer."
        assert agent.last_turn_declaration("r1") == "declared"
        # The declare_done call is provenanced like any tool call ...
        assert [(cid, n) for cid, n, _, _ in seen] == [("call_declare", "declare_done")]
        assert seen[0][2] == {} and seen[0][3] is False
        # ... and no orphan tool_calls: a tool result landed for it.
        assert any("call_declare" == m.get("tool_call_id")
                   for m in _tool_result_entries(agent, "r1"))

    def test_batch_mate_tools_not_executed(self, tmp_path):
        """G4 — a declare_done in a mixed batch ends the turn; sibling calls
        are provenanced but NOT executed (same semantics as the exec
        terminal tool)."""
        agent = _make_agent(tmp_path)
        result = _drive(agent, "r1", [
            _text_response("done", tool_calls=[_shell_call(), _declare_call()],
                           stop_reason="tool_use"),
        ])
        assert agent.last_turn_declaration("r1") == "declared"
        results = _tool_result_entries(agent, "r1")
        # both calls closed (no orphans) ...
        assert {m.get("tool_call_id") for m in results} == {"call_shell", "call_declare"}
        # ... and the sibling was not executed.
        shell_result = next(m for m in results if m.get("tool_call_id") == "call_shell")
        assert "not executed" in str(shell_result.get("content", ""))
        assert result == "done"

    def test_extraneous_args_are_tolerated(self, tmp_path):
        """G11 — zero-payload core is tolerant: stray args do not bounce the
        declaration into a corrective loop (core validation contract is
        absence-of-call only; the overlay layer owns strict validation)."""
        agent = _make_agent(tmp_path)
        result = _drive(agent, "r1", [
            _text_response("wrapped up",
                           tool_calls=[_declare_call({"note": "finished early"})],
                           stop_reason="tool_use"),
        ])
        assert result == "wrapped up"
        assert agent.last_turn_declaration("r1") == "declared"


class TestCorrectiveHoldBack:
    def test_text_then_declare_after_corrective(self, tmp_path):
        """G5 — first response is text with NO declare_done: held back (not
        returned), ONE corrective naming declare_done is appended, the loop
        continues, the model declares, and the turn returns the FINAL text
        exactly once (exactly 2 model calls)."""
        agent = _make_agent(tmp_path)
        result = _drive(agent, "r1", [
            _text_response("First attempt text"),
            _text_response("Final answer.",
                           tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ])
        assert result == "Final answer."
        assert agent.last_turn_declaration("r1") == "declared"
        assert len(_corrective_entries(agent, "r1")) == 1

    def test_double_text_books_undeclared(self, tmp_path):
        """G6 — two consecutive undeclared text-ends: ONE corrective total,
        exactly 2 model calls (no third), the second text returns, marker is
        'undeclared'. The loop can never wedge on a shape-unshapeable model."""
        agent = _make_agent(tmp_path)
        result = _drive(agent, "r1", [
            _text_response("still prose"),
            _text_response("prose again"),
        ])
        assert result == "prose again"
        assert len(_corrective_entries(agent, "r1")) == 1
        assert agent.last_turn_declaration("r1") == "undeclared"

    def test_no_corrective_when_tool_unregistered(self, tmp_path):
        """G7 — registration gate: declare_done absent from the workspace
        means Camp-B behavior holds — text response returns after exactly
        one model call, no corrective, marker is None (the funnel books
        'undeclared')."""
        agent = _make_agent(tmp_path, declare_done=False)
        result = _drive(agent, "r1", [_text_response("plain old ending")])
        assert result == "plain old ending"
        assert _corrective_entries(agent, "r1") == []
        assert agent.last_turn_declaration("r1") is None


class TestCapInterplay:
    def test_forced_summary_never_sees_corrective(self, tmp_path):
        """G8 — iteration-cap path is a loop-observed ending: sentinel +
        forced summary (tools=None), NO corrective anywhere in history,
        marker 'cap_exhausted', turn returns the summary text."""
        agent = _make_agent(tmp_path, max_iterations=1)
        calls = []
        result = _drive(agent, "r1", [
            _text_response("", tool_calls=[_shell_call()], stop_reason="tool_use"),
            _text_response("summary of the wreckage"),
        ], calls=calls)
        assert result == "summary of the wreckage"
        assert len(calls) == 2
        # The second (forced-summary) call is the no-tools call.
        assert calls[1].get("tools") is None
        history = agent.history("r1")
        assert any(m.get("role") == "user"
                   and "[SYSTEM: Tool call limit reached." in str(m.get("content", ""))
                   for m in history)
        assert _corrective_entries(agent, "r1") == []
        assert agent.last_turn_declaration("r1") == "cap_exhausted"


class TestRealPathIntegration:
    def test_marker_and_corrective_through_real_callbacks(self, tmp_path):
        """G10 — the tool-management 'one lesson' pin: a REAL Agent driven
        through the REAL MatrixBot._build_agent_callbacks construction path
        (mocking only provider + nio client). The corrective/hold-back
        mechanics and the marker must work with production-wired callbacks,
        not a hand-built dict."""
        from test_guidance_integration import _make_bot_with_real_agent

        bot, agent = _make_bot_with_real_agent(tmp_path, tools_list=("declare_done",))
        callbacks = bot._build_agent_callbacks(ROOM_A, None)

        with (
            patch("openalph.agent.stream",
                  side_effect=_stream_responses([
                      _text_response("mid prose"),
                      _text_response("and now I am done.",
                                     tool_calls=[_declare_call()],
                                     stop_reason="tool_use"),
                  ])),
            patch("openalph.agent.execute_tool",
                  side_effect=_exec_tool_stub(shell="ok")),
        ):
            result = asyncio.run(
                agent.handle_input("question?", ROOM_A, callbacks=callbacks))

        assert result == "and now I am done."
        assert len(_corrective_entries(agent, ROOM_A)) == 1
        assert agent.last_turn_declaration(ROOM_A) == "declared"


# ──────────────────────────────────────────────────────────────────────
# L — ledger layer (matrix.py funnels + session JSONL) · bead .179
# ──────────────────────────────────────────────────────────────────────


def _entries(bot, room_id=ROOM_ID):
    return bot.session_log.read(room_id)


def _turn_events(bot, name, room_id=ROOM_ID):
    return [e for e in _entries(bot, room_id) if e.get("event") == name]


def _notice_bodies(bot):
    bodies = []
    for call in bot.send_notice.await_args_list:
        if len(call.args) >= 2:
            bodies.append(call.args[1])
        elif "text" in call.kwargs:
            bodies.append(call.kwargs["text"])
    return bodies


def _prime_bot(tmp_path, marker="declared"):
    """gapfill harness: mocked agent + REAL SessionLog; the turn's landing
    is what the (mocked) agent reports via last_turn_declaration."""
    bot, agent = make_bot(tmp_path)
    seed_prior_history(bot.session_log)
    bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
    agent.last_turn_declaration = MagicMock(return_value=marker)
    return bot, agent


async def _run_user_turn(bot, body="hello"):
    await bot._process_message(
        make_room(ROOM_ID), make_event(USER, body, "$e-user"), body)
    await drain(bot)


class TestTurnLedger:
    @pytest.mark.asyncio
    async def test_declared_turn_bookended(self, tmp_path):
        """L1 — a declared turn books exactly one turn.started + one
        turn.finished with a shared turn_id, conclusion 'declared',
        origin 'user', a non-negative elapsed; and the every-turn plain
        m.notice line fires (SB: no threshold)."""
        bot, _ = _prime_bot(tmp_path, "declared")
        await _run_user_turn(bot)

        started = _turn_events(bot, "turn.started")
        finished = _turn_events(bot, "turn.finished")
        assert len(started) == 1 and len(finished) == 1
        assert started[0].get("turn_id")
        assert started[0]["turn_id"] == finished[0]["turn_id"]
        assert finished[0]["conclusion"] == "declared"
        assert finished[0].get("origin") == "user"
        assert isinstance(finished[0].get("elapsed_s"), (int, float))
        assert finished[0]["elapsed_s"] >= 0
        assert any("declared" in body for body in _notice_bodies(bot))

    @pytest.mark.asyncio
    async def test_undeclared_turn_booked(self, tmp_path):
        """L2 — agent reports 'undeclared': the ledger records it (and the
        notice shows it)."""
        bot, _ = _prime_bot(tmp_path, "undeclared")
        await _run_user_turn(bot)
        finished = _turn_events(bot, "turn.finished")
        assert len(finished) == 1
        assert finished[0]["conclusion"] == "undeclared"
        assert any("undeclared" in body for body in _notice_bodies(bot))

    @pytest.mark.asyncio
    async def test_unregistered_agent_books_undeclared(self, tmp_path):
        """L3 — marker None (tool absent, Camp-B ending) books 'undeclared'.
        Absence of declaration is always an anomaly class."""
        bot, agent = _prime_bot(tmp_path, "declared")
        agent.last_turn_declaration = MagicMock(return_value=None)
        await _run_user_turn(bot)
        assert _turn_events(bot, "turn.finished")[0]["conclusion"] == "undeclared"

    @pytest.mark.asyncio
    async def test_cap_exhausted_booked(self, tmp_path):
        """L4 — iteration-cap endings are their own class, not
        'undeclared' (declaration was impossible)."""
        bot, _ = _prime_bot(tmp_path, "cap_exhausted")
        await _run_user_turn(bot)
        assert (_turn_events(bot, "turn.finished")[0]["conclusion"]
                == "cap_exhausted")

    @pytest.mark.asyncio
    async def test_provider_error_booked(self, tmp_path):
        """L5 — ProviderError out of handle_input: conclusion 'error' with
        error_type 'provider', and the bookend pair still lands."""
        bot, agent = _prime_bot(tmp_path, "declared")
        agent.handle_input = AsyncMock(side_effect=ProviderError("boom"))
        await _run_user_turn(bot)
        started = _turn_events(bot, "turn.started")
        finished = _turn_events(bot, "turn.finished")
        assert len(started) == 1 and len(finished) == 1
        assert finished[0]["turn_id"] == started[0]["turn_id"]
        assert finished[0]["conclusion"] == "error"
        assert finished[0].get("error_type") == "provider"

    @pytest.mark.asyncio
    async def test_overflow_booked(self, tmp_path):
        """L6 — ContextOverflowError: conclusion 'overflow'."""
        bot, agent = _prime_bot(tmp_path, "declared")
        agent.handle_input = AsyncMock(side_effect=ContextOverflowError(150000, 200000))
        await _run_user_turn(bot)
        assert (_turn_events(bot, "turn.finished")[0]["conclusion"] == "overflow")

    @pytest.mark.asyncio
    async def test_cancel_books_cancelled_with_same_turn_id(self, tmp_path):
        """L7 — /stop mid-turn: turn.finished conclusion 'cancelled'
        carries the SAME turn_id as the in-flight turn.started."""
        bot, agent = _prime_bot(tmp_path, "declared")
        gate = asyncio.Event()

        async def slow(*args, **kwargs):
            await gate.wait()
            return "ok"

        agent.handle_input = AsyncMock(side_effect=slow)
        task = asyncio.create_task(bot._process_message(
            make_room(ROOM_ID), make_event(USER, "work", "$e-work"), "work"))

        started = None
        for _ in range(200):
            found = _turn_events(bot, "turn.started")
            if found:
                started = found[0]
                break
            await asyncio.sleep(0.01)
        assert started is not None, "turn.started never landed"

        agent.cancel = MagicMock()
        await bot._cancel_current(ROOM_ID)
        gate.set()
        await asyncio.gather(task, return_exceptions=True)
        await drain(bot)

        finished = _turn_events(bot, "turn.finished")
        assert len(finished) == 1
        assert finished[0]["conclusion"] == "cancelled"
        assert finished[0]["turn_id"] == started["turn_id"]

    @pytest.mark.asyncio
    async def test_emission_failure_never_blocks_turn(self, tmp_path):
        """L8 — ledger emission is fail-soft: turn.started/turn.finished
        write failures must not crash or block the turn; the reply is still
        delivered."""
        bot, agent = _prime_bot(tmp_path, "declared")

        orig_append = bot.session_log.append

        def flaky_append(*, role, **kwargs):
            if str(kwargs.get("event", "")).startswith("turn."):
                raise RuntimeError("ledger down")
            return orig_append(role=role, **kwargs)

        bot.session_log.append = flaky_append
        await _run_user_turn(bot)
        assert agent.handle_input.await_count == 1
        assert bot.send.await_count >= 1

    @pytest.mark.asyncio
    async def test_concurrent_rooms_distinct_turn_ids(self, tmp_path):
        """L9 — two rooms' turns book independently: distinct turn_ids, no
        cross-room contamination of entries."""
        bot, agent = _prime_bot(tmp_path, "declared")
        for room_id, eid in ((ROOM_A, "$e-a"), ("!room-b:matrix.local", "$e-b")):
            await bot._process_message(
                make_room(room_id), make_event(USER, "hi", eid), "hi")
        await drain(bot)
        a_finished = _turn_events(bot, "turn.finished", ROOM_A)
        b_finished = _turn_events(bot, "turn.finished", "!room-b:matrix.local")
        assert len(a_finished) == 1 and len(b_finished) == 1
        assert a_finished[0]["turn_id"] != b_finished[0]["turn_id"]

    @pytest.mark.asyncio
    async def test_heartbeat_turn_bookended(self, tmp_path):
        """L10 — the ledger covers EVERY funnel, not just _process_message:
        heartbeat turns book with origin 'heartbeat'."""
        bot, agent = _prime_bot(tmp_path, "declared")
        await bot._run_heartbeat_turn(ROOM_ID, "heartbeat content",
                                      turn_source="heartbeat")
        await drain(bot)
        finished = _turn_events(bot, "turn.finished")
        assert len(finished) == 1
        assert finished[0]["conclusion"] == "declared"
        assert finished[0].get("origin") == "heartbeat"


# ──────────────────────────────────────────────────────────────────────
# E — exec conclusion seam (cli.py) · shared by .179 and .350.4
# ──────────────────────────────────────────────────────────────────────


class TestExecConclusion:
    def test_exec_result_carries_conclusion_declared(self, tmp_path):
        """E1 — the one-JSON-line result gains a top-level 'conclusion'
        taken from the turn's landing (declared here)."""
        from test_cli_exec import make_agent_stub, make_config, parse_single_json, run_exec

        config = make_config(tmp_path)
        agent = make_agent_stub(response_text="done")
        agent.last_turn_declaration = MagicMock(return_value="declared")

        out, _err, code = run_exec(
            ["exec", "--agent", "test-agent", "--task-file", "-"], config=config, agent=agent, stdin="task")
        obj = parse_single_json(out)
        assert code == 0
        assert obj["conclusion"] == "declared"

    def test_exec_result_carries_conclusion_undeclared(self, tmp_path):
        """E2 — a text-ending exec run (Camp-B) surfaces 'undeclared' in
        the result JSON: downstream consumers stop guessing."""
        from test_cli_exec import make_agent_stub, make_config, parse_single_json, run_exec

        config = make_config(tmp_path)
        agent = make_agent_stub(response_text="done by text")
        agent.last_turn_declaration = MagicMock(return_value=None)

        out, _err, code = run_exec(
            ["exec", "--agent", "test-agent", "--task-file", "-"], config=config, agent=agent, stdin="task")
        obj = parse_single_json(out)
        assert obj["conclusion"] == "undeclared"

    def test_exec_result_carries_cap_exhausted(self, tmp_path):
        """E3 — cap endings carry their own class through the exec result."""
        from test_cli_exec import make_agent_stub, make_config, parse_single_json, run_exec

        config = make_config(tmp_path)
        agent = make_agent_stub(response_text="hit the wall")
        agent.last_turn_declaration = MagicMock(return_value="cap_exhausted")

        out, _err, code = run_exec(
            ["exec", "--agent", "test-agent", "--task-file", "-"], config=config, agent=agent, stdin="task")
        obj = parse_single_json(out)
        assert obj["conclusion"] == "cap_exhausted"
