"""Regression suite — 5 audit remediations (R1-R5) applied to the advisor tool.

Covers:
  R1 — the rendered transcript (user text + assistant tool-call INPUTS) is
       redacted BEFORE it is forwarded to the advisor provider (tools/advisor.py,
       run_advisor). Previously only tool OUTPUTS were redacted upstream.
  R2 — `advice` is redacted ONCE, immediately upon extraction, and that SAME
       redacted value is used for BOTH the returned ToolResult and the
       `advisor_results` notice stash (tools/advisor.py, run_advisor).
  R3 — the outer catch-all `except Exception as e` returns a GENERIC message
       to context (no raw `{e}` interpolation); a sanitized log line
       (exception type only) is kept (tools/advisor.py, run_advisor).
  R4 — `get_transcript` returns a shallow COPY of the room history list, not
       the live list object, so a future render-path mutation can't corrupt
       executor history (matrix.py, _build_agent_callbacks).
  R5 — `_advisor_results` is keyed by `(room_id, call_id)` on BOTH the write
       side (tools/advisor.py, run_advisor) and the read/pop side
       (matrix.py, _tool_notice's advisor branch) — a bot-wide dict keyed by
       call_id alone can collide across concurrent rooms.

Per the project's "one lesson": construct a REAL `Agent` and use the REAL
`MatrixBot._build_agent_callbacks`; mock ONLY the provider (LLM) and the nio
client. Helpers below are copied/adapted from test_advisor_integration.py
(the canonical real-path pattern for this feature) — that file is read-only
reference material, never imported from (each test file is self-contained).

All tests gate on `run_advisor is not None` so a red state yields clean
FAILED lines, not collection ERRORs.

Fixture secrets are NON-REAL throughout: "sk-ant-" + "A"*95 matches the
`anthropic_api_key` pattern in openalph.tools.security.CREDENTIAL_PATTERNS
without being a real credential.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.agent import Agent
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import Response, Usage, StreamEvent, ToolCall
from openalph.tools import _TODO_STATE

try:
    from openalph.tools.advisor import run_advisor, render_transcript
except ImportError:
    run_advisor = None
    render_transcript = None

from openalph.matrix import MatrixBot

NOT_IMPL = "advisor module not implemented"

ROOM_A = "!advisor-remediation-a:matrix.local"
ROOM_B = "!advisor-remediation-b:matrix.local"
AGENT_USER = "@agent:matrix.local"

# NON-REAL fixture secret — matches the anthropic_api_key pattern
# (r"sk-ant-[a-zA-Z0-9_-]{8,}") without being an actual credential.
FIXTURE_SECRET = "sk-ant-" + "A" * 95


# --- Helpers (copied/adapted from test_advisor_integration.py) -----------

def _cfg(workspace, **kw):
    defaults = dict(
        name="test-remediation",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[],
        )},
        workspace=workspace,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
        reminders=True,
        model_aliases={"adv": "anthropic/claude-opus-4-8"},
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _setup_workspace(tmp_path, tools=("shell", "advisor", "subagent",
                                      "file_read", "memory_search")):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(exist_ok=True)
    for name in tools:
        if name == "advisor":
            (tools_dir / "advisor.toml").write_text(
                "[config]\n"
                'model = "adv"\n'
                "max_uses = 10\n"
                "max_tokens = 8192\n"
                'thinking = "medium"\n'
                'cache_ttl = "5m"\n'
                "timeout = 300\n"
                "include_system_prompt = true\n"
                "transcript_max_chars = 0\n"
            )
        else:
            (tools_dir / f"{name}.toml").write_text("[config]\n")
    return tmp_path


def _make_bot_with_real_agent(tmp_path, agent=None, **agent_kw):
    """Build a MatrixBot with a REAL Agent and mocked nio client."""
    if "tools_list" in agent_kw:
        ws = _setup_workspace(tmp_path, tools=agent_kw.pop("tools_list"))
    else:
        ws = _setup_workspace(tmp_path)
    config = _cfg(ws, **agent_kw)
    if agent is None:
        agent = Agent(config)

    from openalph.config import MatrixConfig
    matrix_config = MatrixConfig(
        homeserver="https://matrix.local",
        user_id=AGENT_USER,
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )

    bot = MatrixBot.__new__(MatrixBot)
    bot.config = matrix_config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$resp1"))
    bot.client.room_typing = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = set()
    bot._room_effort = {}
    bot._room_cache_ttl = {}
    bot._room_timesense = {}
    bot._halted_rooms = set()
    bot._background_tasks = set()
    bot._session_locks = {}
    bot.session_log = MagicMock()
    bot.session_log.append = MagicMock()
    bot.session_log.build_context = MagicMock(return_value=[])
    bot.session_log.read = MagicMock(return_value=[])
    bot.session_log.last_event_id = MagicMock(return_value=None)
    bot.session_log.usage_totals = MagicMock(return_value={})
    bot.heartbeat = MagicMock()
    bot.heartbeat.is_active = MagicMock(return_value=False)
    bot.umbral = MagicMock()
    bot.umbral.is_active = MagicMock(return_value=False)
    bot._steering_inbox = {}
    bot._active_turns = set()
    if not hasattr(bot, "agent_user_id"):
        bot.agent_user_id = AGENT_USER

    return bot, agent


def _advice_response(text="THE_ADVICE_BYTES: refactor incrementally.",
                     thinking=None, cache_read=42, in_tok=1000, out_tok=120):
    return Response(
        content=text, model="claude-opus-4-8",
        usage=Usage(input_tokens=in_tok, output_tokens=out_tok, cache_read_tokens=cache_read),
        stop_reason="end_turn", thinking=thinking or [],
    )


def _one_consult_stream(final_text="fin"):
    """Executor stream: iteration 1 dispatches the advisor tool, iteration 2 finishes.
    Fresh call gets its own iteration counter, so every invocation independently
    yields the SAME tool_call id ("adv_1") on its first iteration — used by the
    R5 test to prove two different rooms reusing the same call_id don't collide."""
    idx = [0]

    async def _stream(*, messages=None, tools=None, model="m", **kw):
        idx[0] += 1
        if idx[0] == 1:
            tc = ToolCall(id="adv_1", name="advisor", input={"focus": "x"})
            yield StreamEvent(type="done", model=model, stop_reason="tool_use",
                              response=Response(content="", tool_calls=[tc], model=model,
                                                usage=Usage(input_tokens=10, output_tokens=5),
                                                stop_reason="tool_use"))
        else:
            yield StreamEvent(type="done", model=model, stop_reason="end_turn",
                              response=Response(content=final_text, model=model,
                                                usage=Usage(input_tokens=10, output_tokens=5),
                                                stop_reason="end_turn"))
    return _stream


def _wire_notice_capture(bot):
    """Patch a bot's notice-sending methods to capture (room_id, message) pairs."""
    sent = []
    bot._room_send_with_retry = AsyncMock(side_effect=lambda rid, msg, **kw: sent.append((rid, msg)))
    bot.send_notice = AsyncMock(side_effect=lambda rid, text, **kw: sent.append(
        (rid, {"msgtype": "m.notice", "body": text})))
    bot._set_typing = AsyncMock()
    return sent


@pytest.fixture(autouse=True)
def _cleanup_todo():
    yield
    _TODO_STATE.pop(ROOM_A, None)
    _TODO_STATE.pop(ROOM_B, None)


# ========================================================================
# R1 — transcript redacted before egress to the advisor provider
# ========================================================================

class TestR1TranscriptRedactedBeforeEgress:

    @pytest.mark.asyncio
    async def test_R1_transcript_redacted_before_egress(self, tmp_path):
        """A NON-REAL credential planted in BOTH a user message AND an assistant
        tool-call `input` must NOT reach `openalph.tools.advisor.complete` in the
        raw — the rendered transcript sent to the advisor provider must show
        the redaction marker and never the raw fixture secret."""
        assert run_advisor is not None, NOT_IMPL
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb = bot._build_agent_callbacks(ROOM_A, None)

        # Seed room history BEFORE the consult turn: one user message and one
        # assistant tool-call whose INPUT both carry the fixture secret.
        agent.history(ROOM_A).append({
            "role": "user",
            "content": f"here is my key {FIXTURE_SECRET} please use it",
        })
        agent.history(ROOM_A).append({
            "role": "assistant",
            "content": "",
            "tool_calls": [ToolCall(id="sh_1", name="shell",
                                     input={"command": f"echo {FIXTURE_SECRET}"})],
        })
        agent.history(ROOM_A).append({
            "role": "tool", "tool_call_id": "sh_1",
            "content": '<tool_result tool="shell" id="sh_1">\nok\n</tool_result>',
            "is_error": False,
        })

        captured_calls = []

        async def _capture_complete(**kw):
            captured_calls.append(kw)
            return _advice_response()

        with patch("openalph.agent.stream", side_effect=_one_consult_stream()), \
             patch("openalph.tools.advisor.complete", new=_capture_complete):
            await agent.handle_input("please advise", room_id=ROOM_A, callbacks=cb)

        assert captured_calls, "Advisor must have been consulted"
        adv_msgs = captured_calls[0]["messages"]
        sent_transcript = adv_msgs[0]["content"][0]["text"]

        assert "[REDACTED" in sent_transcript, \
            "Transcript sent to the advisor provider must carry a redaction marker"
        assert FIXTURE_SECRET not in sent_transcript, \
            "The raw fixture secret must NEVER reach the advisor provider call"


# ========================================================================
# R2 — advice redacted once; reused for BOTH the ToolResult and the notice stash
# ========================================================================

class TestR2AdviceRedactedInNoticeAndContext:

    @pytest.mark.asyncio
    async def test_R2_advice_redacted_in_notice_and_context(self, tmp_path):
        """Advisor returns advice containing a NON-REAL credential:
          (a) the advisor tool_result appended to executor history is redacted,
          (b) the value stashed in advisor_results (feeding the Matrix return
              notice) is ALSO redacted, and
          (c) the actual Matrix m.notice body/formatted_body sent on return
              never contains the raw secret either.
        No raw secret survives in any of the three surfaces.

        Two separate turns are used: `_tool_notice` POPS the advisor_results
        stash entry the instant it fires (real production behavior), so (a)+(b)
        are checked on a turn WITHOUT on_tool_call wired (stash left intact for
        inspection), and (c) is checked on a separate turn WITH on_tool_call
        wired (proving what actually reaches the room)."""
        assert run_advisor is not None, NOT_IMPL
        advice_text = f"Use this key: {FIXTURE_SECRET} for auth."

        # --- Turn 1: no on_tool_call wired -> advisor_results stash survives
        # the turn intact, so both the history AND the stash can be inspected.
        bot1, agent1 = _make_bot_with_real_agent(tmp_path)
        cb1 = bot1._build_agent_callbacks(ROOM_A, None)
        with patch("openalph.agent.stream", side_effect=_one_consult_stream()), \
             patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_advice_response(text=advice_text)):
            await agent1.handle_input("work", room_id=ROOM_A, callbacks=cb1)

        # (a) executor history tool_result content is redacted.
        history = agent1.history(ROOM_A)
        tool_entries = [m for m in history if m.get("role") == "tool"]
        assert tool_entries, "Advisor consult must append a tool_result to history"
        history_content = tool_entries[-1]["content"]
        assert FIXTURE_SECRET not in history_content, \
            "Raw secret must not appear in the executor history tool_result"
        assert "[REDACTED" in history_content, \
            "Redaction marker must be present in the executor history tool_result"

        # (b) the advisor_results stash (source of the Matrix return notice)
        # is ALSO redacted — keyed (room_id, call_id) per R5.
        assert cb1["advisor_results"], "advisor_results must have an entry after a consult"
        stashed = list(cb1["advisor_results"].values())[0]
        stashed_advice = stashed.get("advice", "")
        assert FIXTURE_SECRET not in stashed_advice, \
            "Raw secret must not appear in the advisor_results notice stash"
        assert "[REDACTED" in stashed_advice, \
            "Redaction marker must be present in the advisor_results notice stash"

        # --- Turn 2: on_tool_call/on_tool_intent wired -> proves what the
        # Matrix room actually receives (the stash entry gets popped here).
        bot2, agent2 = _make_bot_with_real_agent(tmp_path)
        sent = _wire_notice_capture(bot2)
        _tool_notice, _tool_intent = bot2._make_tool_callbacks(ROOM_A)
        cb2 = bot2._build_agent_callbacks(ROOM_A, None)
        with patch("openalph.agent.stream", side_effect=_one_consult_stream()), \
             patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_advice_response(text=advice_text)):
            await agent2.handle_input("work", room_id=ROOM_A, callbacks=cb2,
                                      on_tool_call=_tool_notice, on_tool_intent=_tool_intent)

        # (c) the actual Matrix notice sent never carries the raw secret.
        blobs = [str(m.get("body", "")) + str(m.get("formatted_body", "")) for _rid, m in sent]
        joined = "\n".join(blobs)
        assert FIXTURE_SECRET not in joined, \
            "Raw secret must never appear in any Matrix notice body/formatted_body"
        assert "[REDACTED" in joined, \
            "Redaction marker must appear in the Matrix return notice"


# ========================================================================
# R3 — generic error text on the catch-all exception handler
# ========================================================================

class TestR3UnexpectedExceptionGenericMessage:

    @pytest.mark.asyncio
    async def test_R3_unexpected_exception_generic_message(self, tmp_path):
        """Force an unexpected error (get_transcript raises KeyError('boom')):
        the returned ToolResult must be is_error=True with the GENERIC message,
        and must NOT contain the raw exception text 'boom'. No exception
        propagates out of run_advisor (A6 fail-soft)."""
        assert run_advisor is not None, NOT_IMPL
        ws = _setup_workspace(tmp_path)
        cfg = _cfg(ws)

        def _raiser():
            raise KeyError("boom")

        callbacks = {
            "get_transcript": _raiser,
            "advisor_uses": {},
            "room_id": ROOM_A,
            "call_id": "adv_err",
        }
        tool_config = {
            "model": "adv", "max_uses": 10, "max_tokens": 8192,
            "thinking": "medium", "cache_ttl": "5m", "timeout": 300,
            "include_system_prompt": True, "transcript_max_chars": 0,
        }

        # Must not raise.
        result = await run_advisor(
            focus=None, model=None, config=cfg,
            tool_config=tool_config, callbacks=callbacks,
        )

        assert result.is_error is True, "Unexpected exception must yield is_error=True"
        assert result.content == (
            "Advisor consult failed unexpectedly — proceed with your own judgment."
        ), f"Content must be the exact generic message, got: {result.content!r}"
        assert "boom" not in result.content, \
            "The raw exception text must NOT leak into the returned content"
        assert "KeyError" not in result.content, \
            "The raw exception type name must NOT leak into the returned content"


# ========================================================================
# R4 — get_transcript returns a copy of the history list
# ========================================================================

class TestR4GetTranscriptReturnsCopy:

    @pytest.mark.asyncio
    async def test_R4_get_transcript_returns_copy(self, tmp_path):
        """get_transcript()[1] must be a DIFFERENT list object than
        agent.history(room) — mutating the returned list must not affect the
        real executor history (cache-safety belt)."""
        assert run_advisor is not None, NOT_IMPL
        bot, agent = _make_bot_with_real_agent(tmp_path)
        agent.history(ROOM_A).append({"role": "user", "content": "seed message"})

        cb = bot._build_agent_callbacks(ROOM_A, None)
        t = cb["get_transcript"]()

        assert t[1] is not agent.history(ROOM_A), \
            "get_transcript must return a COPY of the history list, not the live object"

        before = len(agent.history(ROOM_A))
        t[1].append({"role": "user", "content": "MUTATION_SHOULD_NOT_LEAK"})
        after = len(agent.history(ROOM_A))

        assert after == before, \
            "Appending to the returned transcript list must NOT mutate the real executor history"
        assert not any(
            m.get("content") == "MUTATION_SHOULD_NOT_LEAK" for m in agent.history(ROOM_A)
        ), "The mutation must not be visible in the real history"


# ========================================================================
# R5 — advisor_results keyed by (room_id, call_id), not call_id alone
# ========================================================================

class TestR5AdvisorResultsRoomScoped:

    @pytest.mark.asyncio
    async def test_R5_advisor_results_room_scoped(self, tmp_path):
        """Two DIFFERENT rooms reuse the SAME call_id ('adv_1', from
        _one_consult_stream — a fresh instance per room independently starts
        its own iteration count at 1). Driving a consult in each room with
        distinct advice must not collide.

        Two-phase check:
          Phase 1 (dict-level, no on_tool_call wired so neither write is
          popped): after BOTH rooms' consults land in the ONE shared
          bot-wide advisor_results dict, BOTH (room, call_id) entries must
          coexist with their OWN distinct advice — proving the effective key
          is (room_id, call_id), since a call_id-only key would have had the
          second write silently clobber the first.
          Phase 2 (notice-level, on_tool_call wired on fresh bot/agent
          instances): each room's actual Matrix return notice shows only its
          own advice, never the other room's."""
        assert run_advisor is not None, NOT_IMPL
        advice_a = "ADVICE_FOR_ROOM_A_ONLY"
        advice_b = "ADVICE_FOR_ROOM_B_ONLY"

        # --- Phase 1: dict-level collision check. No on_tool_call wired, so
        # _tool_notice never pops — both writes accumulate in the ONE shared
        # bot._advisor_results dict and can be inspected together.
        bot1, agent1 = _make_bot_with_real_agent(tmp_path)
        cb_a1 = bot1._build_agent_callbacks(ROOM_A, None)
        with patch("openalph.agent.stream", side_effect=_one_consult_stream()), \
             patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_advice_response(text=advice_a)):
            await agent1.handle_input("work A", room_id=ROOM_A, callbacks=cb_a1)

        cb_b1 = bot1._build_agent_callbacks(ROOM_B, None)
        with patch("openalph.agent.stream", side_effect=_one_consult_stream()), \
             patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_advice_response(text=advice_b)):
            await agent1.handle_input("work B", room_id=ROOM_B, callbacks=cb_b1)

        # Both consults used the identical call_id "adv_1" (each stream's own
        # counter independently yields "adv_1" on its first iteration) and
        # share the ONE bot-wide dict.
        assert cb_a1["advisor_results"] is cb_b1["advisor_results"], \
            "Both rooms must share the SAME bot-wide advisor_results dict (the collision surface)"
        shared = cb_a1["advisor_results"]
        assert (ROOM_A, "adv_1") in shared and (ROOM_B, "adv_1") in shared, \
            "Both (room_id, call_id) entries must coexist — a call_id-only key " \
            "would have let Room B's write silently clobber Room A's"
        assert shared[(ROOM_A, "adv_1")]["advice"] == advice_a, \
            "Room A's entry must hold Room A's own advice, unclobbered by Room B"
        assert shared[(ROOM_B, "adv_1")]["advice"] == advice_b, \
            "Room B's entry must hold Room B's own advice, unclobbered by Room A"

        # --- Phase 2: notice-level check on fresh bot/agent instances, with
        # on_tool_call wired so each room's actual return notice is captured.
        bot2, agent2 = _make_bot_with_real_agent(tmp_path)
        sent = _wire_notice_capture(bot2)

        _tool_notice_a, _tool_intent_a = bot2._make_tool_callbacks(ROOM_A)
        cb_a2 = bot2._build_agent_callbacks(ROOM_A, None)
        with patch("openalph.agent.stream", side_effect=_one_consult_stream()), \
             patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_advice_response(text=advice_a)):
            await agent2.handle_input("work A", room_id=ROOM_A, callbacks=cb_a2,
                                      on_tool_call=_tool_notice_a, on_tool_intent=_tool_intent_a)

        _tool_notice_b, _tool_intent_b = bot2._make_tool_callbacks(ROOM_B)
        cb_b2 = bot2._build_agent_callbacks(ROOM_B, None)
        with patch("openalph.agent.stream", side_effect=_one_consult_stream()), \
             patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_advice_response(text=advice_b)):
            await agent2.handle_input("work B", room_id=ROOM_B, callbacks=cb_b2,
                                      on_tool_call=_tool_notice_b, on_tool_intent=_tool_intent_b)

        room_a_blobs = [str(m.get("body", "")) + str(m.get("formatted_body", ""))
                        for rid, m in sent if rid == ROOM_A]
        room_b_blobs = [str(m.get("body", "")) + str(m.get("formatted_body", ""))
                        for rid, m in sent if rid == ROOM_B]
        joined_a = "\n".join(room_a_blobs)
        joined_b = "\n".join(room_b_blobs)

        assert advice_a in joined_a, "Room A's notice must contain Room A's own advice"
        assert advice_b not in joined_a, "Room A's notice must NOT contain Room B's advice"
        assert advice_b in joined_b, "Room B's notice must contain Room B's own advice"
        assert advice_a not in joined_b, "Room B's notice must NOT contain Room A's advice"

        # And the executor histories of each room reflect their own advice only.
        hist_a = agent2.history(ROOM_A)
        hist_b = agent2.history(ROOM_B)
        tool_a = [m["content"] for m in hist_a if m.get("role") == "tool"]
        tool_b = [m["content"] for m in hist_b if m.get("role") == "tool"]
        assert any(advice_a in c for c in tool_a), "Room A history must carry Room A's advice"
        assert not any(advice_b in c for c in tool_a), "Room A history must NOT carry Room B's advice"
        assert any(advice_b in c for c in tool_b), "Room B history must carry Room B's advice"
        assert not any(advice_a in c for c in tool_b), "Room B history must NOT carry Room A's advice"


# ========================================================================
# N1 -- executor-authored `focus` redacted at BOTH egress points: the
# advisor provider request (tools/advisor.py, run_advisor) and the Matrix
# 🔮 spawn notice (matrix.py, _tool_intent's advisor branch). R1/R2 only
# covered the transcript and the returned advice -- `focus` itself was
# still forwarded/echoed raw.
# ========================================================================

def _one_consult_stream_with_focus(focus_value, final_text="fin"):
    """Like _one_consult_stream, but the dispatched advisor tool_use carries
    a caller-supplied `focus` value in its input (instead of the fixed "x")
    so tests can plant a NON-REAL secret there."""
    idx = [0]

    async def _stream(*, messages=None, tools=None, model="m", **kw):
        idx[0] += 1
        if idx[0] == 1:
            tc = ToolCall(id="adv_1", name="advisor", input={"focus": focus_value})
            yield StreamEvent(type="done", model=model, stop_reason="tool_use",
                              response=Response(content="", tool_calls=[tc], model=model,
                                                usage=Usage(input_tokens=10, output_tokens=5),
                                                stop_reason="tool_use"))
        else:
            yield StreamEvent(type="done", model=model, stop_reason="end_turn",
                              response=Response(content=final_text, model=model,
                                                usage=Usage(input_tokens=10, output_tokens=5),
                                                stop_reason="end_turn"))
    return _stream


class TestN1FocusRedactedInEgressAndSpawnNotice:

    @pytest.mark.asyncio
    async def test_N1_focus_redacted_in_egress_and_spawn_notice(self, tmp_path):
        """The executor-authored `focus` string (the advisor tool_use's
        `input["focus"]`) must be redacted BEFORE it reaches either of its two
        egress points:
          (a) the advisor provider request -- the focus/closing text block
              forwarded to `openalph.tools.advisor.complete` must show the
              redaction marker, never the raw fixture secret (advisor.py,
              run_advisor).
          (b) the Matrix room's 🔮 spawn notice -- the notice emitted by the
              `_tool_intent` advisor branch echoes `focus[:120]`; it must
              never carry the raw fixture secret either (matrix.py,
              _tool_intent).

        Non-vacuous: against the pre-fix code, (a) fails because
        closing_text embeds the raw `focus` verbatim, and (b) fails because
        the notice echoes `str(tc.input.get("focus", ""))` verbatim.
        """
        assert run_advisor is not None, NOT_IMPL
        bot, agent = _make_bot_with_real_agent(tmp_path)
        sent = _wire_notice_capture(bot)
        _tool_notice, _tool_intent = bot._make_tool_callbacks(ROOM_A)
        cb = bot._build_agent_callbacks(ROOM_A, None)

        captured_calls = []

        async def _capture_complete(**kw):
            captured_calls.append(kw)
            return _advice_response()

        with patch("openalph.agent.stream",
                   side_effect=_one_consult_stream_with_focus(FIXTURE_SECRET)), \
             patch("openalph.tools.advisor.complete", new=_capture_complete):
            await agent.handle_input(
                "please advise", room_id=ROOM_A, callbacks=cb,
                on_tool_call=_tool_notice, on_tool_intent=_tool_intent,
            )

        # --- (a) egress to the advisor provider request -----------------
        assert captured_calls, "Advisor must have been consulted"
        adv_msgs = captured_calls[0]["messages"]
        focus_block_text = adv_msgs[0]["content"][1]["text"]

        assert "[REDACTED" in focus_block_text, \
            "The focus/closing block sent to the advisor provider must carry a redaction marker"
        assert FIXTURE_SECRET not in focus_block_text, \
            "The raw fixture secret must NEVER reach the advisor provider via the focus block"

        # --- (b) the Matrix 🔮 spawn notice -------------------------------
        spawn_blobs = [
            str(m.get("body", "")) + str(m.get("formatted_body", ""))
            for _rid, m in sent
            if "Advisor consult" in str(m.get("body", ""))
        ]
        assert spawn_blobs, "The advisor spawn notice (🔮 Advisor consult -> ...) must have been sent"
        joined_spawn = "\n".join(spawn_blobs)
        assert FIXTURE_SECRET not in joined_spawn, \
            "The raw fixture secret must never appear in the Matrix spawn notice"
        assert "[REDACTED" in joined_spawn, \
            "Redaction marker must be present in the Matrix spawn notice"
