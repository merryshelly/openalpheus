"""RED test suite — kdsn.298 "session-orient" reminder trigger.

Implementation contract (these tests ARE the specification — spec §5 tests 1–22):

Module changes (src/openalph/reminders.py):
  ReminderState gains defaulted fields:
      model_resolved: str = ""   — post-room-override, post-alias-expansion
      model_vision:   bool = False
      orient_ts:      str = ""   — pre-rendered host-local timestamp string
  Reminder gains optional field:
      detail: str | None = None  — generic JSONL plumbing; session-orient sets
                                   detail = state.model_resolved at fire time
  ReminderEngine gains state:
      _oriented_model: str | None = None   — set in __init__ AND cleared in reset()
  New trigger (evaluate, turn_start only):
      predicate: state.evaluation_point == "turn_start"
                 and state.model_resolved != self._oriented_model
      fires Reminder(trigger="session-orient", detail=state.model_resolved)
      with the exact §4.3 four-line text, then _oriented_model = model_resolved.
  rehydrate() gains:
      elif trigger == "session-orient" and entry.get("detail") is not None:
          self._oriented_model = entry["detail"]     # last-wins across entries

Sink changes (src/openalph/callbacks.py):
  MatrixSinks.log_reminder and HeadlessSinks.log_reminder append
      detail=reminder.detail  CONDITIONALLY — only when reminder.detail is not
      None (T1–T6 JSONL entries must NOT gain a detail key).

Agent change (src/openalph/agent.py):
  The turn-start ReminderState site (~agent.py:739-751) populates the three new
  fields via one helper (_orient_inputs): resolve_model fail-soft, host-local
  astimezone() %A/%B/%d/%Y — %H:%M %Z timestamp, model_supports_vision.
  The tool-loop-boundary site (~agent.py:850) passes nothing for them.

Anchor notes: TAG_OPEN below uses REAL angle brackets (od-verified in the
precedent files; grep renders them as entities — display artifact only).
Local copies of the canonical helpers (no cross-test-module imports).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.agent import Agent
from openalph.callbacks import HeadlessSinks, MatrixSinks, build_callbacks
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.matrix import MatrixBot
from openalph.provider import Response, StreamEvent, ToolCall, Usage
from openalph.reminders import Reminder, ReminderEngine, ReminderState
from openalph.session import SessionLog
from openalph.tools import ToolResult, _TODO_STATE

# --- Constants ---

ROOM_A = "!orient-room-a:matrix.local"
ROOM_B = "!orient-room-b:matrix.local"
ROOM_SUB = "__sub__"
AGENT_USER = "@agent:matrix.local"
TAG_OPEN = "<system-reminder>"          # real angle brackets (od-verified)

ORIENT_ID = "session-orient"

MODEL_DEFAULT = "anthropic/claude-test-a"       # window 200000, vision=True (overrides)
MODEL_ALT = "fireworks/llama-test-b"            # window 131072, vision=False (overrides)
ORIENT_TS = "Friday, August 01, 2026 — 12:00 UTC"


# --- Helpers (local copies of the canonical test_guidance_* patterns) ---

def _cfg(workspace, **kw):
    """Build AgentConfig for tests.

    Defaults are hermetic: model_limits/model_vision pin the resolutions for
    the two fixture models so expectations never depend on the curated
    _MODEL_CAPABILITIES table.
    """
    defaults = dict(
        name="test-orient",
        default_model=MODEL_DEFAULT,
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
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _cfg_two_providers(workspace, **kw):
    """_cfg with a second (fireworks) provider and pinned limits/vision.

    Spec §5 test-13 fixture note: switch_model validates the target against
    config.providers (agent.py:444-453) — a second provider is required, and
    it varies the window/vision expectations too.
    """
    base = dict(
        providers={
            "anthropic": ProviderConfig(
                key="anthropic", type="anthropic", api_key="sk-test",
                base_url=None, quirks=[],
            ),
            "fireworks": ProviderConfig(
                key="fireworks", type="openai", api_key="sk-test-2",
                base_url="https://api.fireworks.test", quirks=[],
            ),
        },
        model_limits={MODEL_DEFAULT: 200000, MODEL_ALT: 131072},
        model_vision={MODEL_DEFAULT: True, MODEL_ALT: False},
    )
    base.update(kw)
    return _cfg(workspace, **base)


def _setup_workspace(tmp_path, tools=("shell", "file_read", "file_write",
                                       "file_edit", "todo_write", "memory_search")):
    """Create workspace/tools/ with tool TOMLs for discovery."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(exist_ok=True)
    for name in tools:
        (tools_dir / f"{name}.toml").write_text("[config]\n")
    return tmp_path


def _orient_turn_state(**kw):
    """ReminderState for turn_start evaluations, WITH the new §4.5 fields.

    Pre-implementation, constructing ReminderState with model_resolved /
    model_vision / orient_ts raises TypeError (unexpected keyword argument) —
    that IS the intended red signal for every engine-unit test in this file.
    """
    defaults = dict(
        evaluation_point="turn_start",
        iteration=0,
        max_iterations=100,
        context_tokens=10000,
        context_limit=200000,
        completed_turns=1,
        turn_source=None,
        tool_calls_this_turn={},
        tool_calls_session={},
        todo_list=[],
        enabled_tools={"shell", "file_read", "memory_search", "todo_write"},
        # --- new kdsn.298 §4.5 fields ---
        model_resolved=MODEL_DEFAULT,
        model_vision=True,
        orient_ts=ORIENT_TS,
    )
    defaults.update(kw)
    return ReminderState(**defaults)


def _make_capturing_stream(tool_iterations=6, final_text="Done"):
    """Stream factory: tool_use N times then text. Captures wire payloads."""
    payloads = []
    call_idx = [0]

    async def _stream(*, config=None, system=None, messages=None,
                      tools=None, model="test", thinking=None,
                      cache_ttl=None, **kw):
        payloads.append(list(messages))
        call_idx[0] += 1
        if tools is not None and call_idx[0] <= tool_iterations:
            tc = ToolCall(id=f"tc_{call_idx[0]}", name="shell",
                          input={"command": f"echo {call_idx[0]}"})
            yield StreamEvent(type="tool_done", tool_index=0, tool_call=tc)
            yield StreamEvent(
                type="done",
                response=Response(
                    content="", tool_calls=[tc], model=model,
                    usage=Usage(input_tokens=10, output_tokens=5),
                    stop_reason="tool_use"),
                stop_reason="tool_use", model=model)
        else:
            yield StreamEvent(type="text", content=final_text)
            yield StreamEvent(
                type="done",
                response=Response(
                    content=final_text, model=model,
                    usage=Usage(input_tokens=10, output_tokens=5),
                    stop_reason="end_turn"),
                stop_reason="end_turn", model=model)

    return _stream, payloads


def _make_bot_with_real_agent(tmp_path, agent=None, session_log=None, **agent_kw):
    """Build a MatrixBot with a REAL Agent and mocked nio client.

    Canonical pattern from tests/test_guidance_integration.py, extended so a
    REAL SessionLog may be passed in (bot._build_agent_callbacks
    prefers `session_log or self.session_log` at matrix.py:1334).
    """
    ws = _setup_workspace(tmp_path)
    config = _cfg(ws, **agent_kw)
    if agent is None:
        agent = Agent(config)

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
    if session_log is not None:
        bot.session_log = session_log
    else:
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

    return bot, agent


def _orient_entries(payload):
    """session-orient entries in a JSONL/dict payload (list of entry dicts)."""
    return [e for e in payload if e.get("trigger") == ORIENT_ID]


def _orient_history(history):
    """session-orient framed messages in a live history list."""
    return [m for m in history
            if m.get("role") == "user"
            and TAG_OPEN in str(m.get("content", ""))
            and "Session orientation" in str(m.get("content", ""))]



def _append_model_override(sl, room, model_str):
    """Append a system/model_override entry (umbral+restart restore edge)."""
    sl.append(role="system", sender=AGENT_USER, room=room, event_id=None,
              event="model_override", detail=model_str)


async def _run_turn(agent, callbacks, text="work", room_id=ROOM_A,
                    tool_iterations=0, payloads_sink=None):
    """Run one real handle_input turn with mocked provider + tool execution."""
    stream_fn, payloads = _make_capturing_stream(tool_iterations=tool_iterations)
    mock_exec = AsyncMock(return_value=ToolResult(content="ok", is_error=False))
    with patch("openalph.agent.stream", side_effect=stream_fn), \
         patch("openalph.agent.execute_tool", mock_exec):
        await agent.handle_input(text, room_id=room_id, callbacks=callbacks)
    if payloads_sink is not None:
        payloads_sink.extend(payloads)
    return payloads


@pytest.fixture(autouse=True)
def _cleanup_todo():
    """Clean up module-global todo state after each test."""
    yield
    _TODO_STATE.pop(ROOM_A, None)
    _TODO_STATE.pop(ROOM_B, None)
    _TODO_STATE.pop(ROOM_SUB, None)


# ============================================================================
# Spec §5 — Engine unit tests (1–7, plus 5b/5c named as the spec does)
# ============================================================================

class TestEngineUnit:
    """ReminderState built by hand; engine evaluated directly.

    Pre-implementation red signal: TypeError from the missing
    model_resolved/model_vision/orient_ts kwargs (intended — spec §4.5), or
    AttributeError on the missing _oriented_model field.
    """

    def test_t01_fires_on_fresh_engine_at_turn_start(self, tmp_path):
        """§5.1: fresh engine (oriented=None) fires once at turn_start;
        trigger id, four content lines, detail == model_resolved."""
        eng = ReminderEngine(_cfg(tmp_path))
        results = eng.evaluate(_orient_turn_state())
        orient = [r for r in results if r.trigger == ORIENT_ID]
        assert len(orient) == 1, (
            f"session-orient must fire exactly once on a fresh engine at "
            f"turn_start; got {len(orient)} (results: {[r.trigger for r in results]})"
        )
        rem = orient[0]
        content = rem.content
        assert content.startswith(TAG_OPEN + "\n"), \
            "content must be framed with the system-reminder tags"
        assert "Session orientation" in content, \
            "content must open with the §4.3 header line"
        assert f"Session began: {ORIENT_TS}" in content, \
            "content must carry the pre-rendered orient_ts string"
        assert f"Active model: {MODEL_DEFAULT}" in content, \
            "content must carry the resolved provider/model string"
        assert "Context window: 200,000 tokens" in content, \
            "content must carry the live-resolved window (thousands separator)"
        assert "Vision: yes" in content, \
            "content must state vision=yes when state.model_vision is True"
        assert rem.detail == MODEL_DEFAULT, \
            "session-orient must set detail = state.model_resolved at fire time"

    def test_t02_does_not_refire_with_equal_model(self, tmp_path):
        """§5.2: equal-model follow-up evaluation is suppressed (model-keyed cap)."""
        eng = ReminderEngine(_cfg(tmp_path))
        first = eng.evaluate(_orient_turn_state())
        assert any(r.trigger == ORIENT_ID for r in first), \
            "Precondition: fresh engine fires once"
        second = eng.evaluate(_orient_turn_state())
        assert not any(r.trigger == ORIENT_ID for r in second), \
            "session-orient must NOT re-fire for the same resolved model"

    def test_t03_does_not_fire_at_tool_loop_boundary(self, tmp_path):
        """§5.3: evaluation_point guard — boundary never fires session-orient,
        even on a fresh engine with a model set."""
        eng = ReminderEngine(_cfg(tmp_path))
        st = _orient_turn_state(evaluation_point="tool_loop_boundary",
                                iteration=0)
        results = eng.evaluate(st)
        assert not any(r.trigger == ORIENT_ID for r in results), \
            "session-orient must not fire at tool_loop_boundary"

    def test_t04_model_switch_refires_and_each_switch_keys_once(self, tmp_path):
        """§5.4: A fire → B fires → back to A fires; repeats of the same model
        stay suppressed in between."""
        eng = ReminderEngine(_cfg(tmp_path))
        r_a1 = eng.evaluate(_orient_turn_state())
        assert any(r.trigger == ORIENT_ID for r in r_a1), \
            "Precondition: first evaluation fires"
        r_a2 = eng.evaluate(_orient_turn_state())
        assert not any(r.trigger == ORIENT_ID for r in r_a2), \
            "Same model must stay suppressed"
        r_b = eng.evaluate(_orient_turn_state(model_resolved=MODEL_ALT,
                                              model_vision=False,
                                              context_limit=131072))
        assert any(r.trigger == ORIENT_ID for r in r_b), \
            "Switching to model B must re-fire session-orient"
        r_b2 = eng.evaluate(_orient_turn_state(model_resolved=MODEL_ALT,
                                               model_vision=False,
                                               context_limit=131072))
        assert not any(r.trigger == ORIENT_ID for r in r_b2), \
            "Model B must also be keyed once"
        r_a3 = eng.evaluate(_orient_turn_state())
        orient_a3 = [r for r in r_a3 if r.trigger == ORIENT_ID]
        assert orient_a3, "Switching BACK to model A must re-fire"
        assert orient_a3[0].detail == MODEL_DEFAULT, \
            "Back-switch fire must carry model A in detail"

    def test_t05_rehydrate_restores_oriented_model_from_detail(self, tmp_path):
        """§5.5: session-orient entry with detail → equal-model evaluation
        suppressed; legacy entry lacking detail → fires once (backfill)."""
        entries = [
            {"role": "user", "source": "reminder", "trigger": ORIENT_ID,
             "detail": MODEL_DEFAULT, "content": TAG_OPEN},
        ]
        eng = ReminderEngine(_cfg(tmp_path))
        eng.rehydrate(entries)
        res = eng.evaluate(_orient_turn_state())
        assert not any(r.trigger == ORIENT_ID for r in res), \
            "Rehydrated detail == current model must suppress the nudge"

        legacy = [
            {"role": "user", "source": "reminder", "trigger": ORIENT_ID,
             "content": TAG_OPEN},  # no detail key
        ]
        eng2 = ReminderEngine(_cfg(tmp_path))
        eng2.rehydrate(legacy)
        res2 = eng2.evaluate(_orient_turn_state())
        assert any(r.trigger == ORIENT_ID for r in res2), \
            "Legacy entry without detail must leave oriented=None → fires once"

    def test_t05b_rehydrate_last_wins(self, tmp_path):
        """§5.5b: details A, B, C in JSONL order → oriented == C; evaluation
        with C suppressed, with B fires (the /model-switch-then-restart path)."""
        entries = [
            {"role": "user", "source": "reminder", "trigger": ORIENT_ID,
             "detail": MODEL_DEFAULT, "content": TAG_OPEN},
            {"role": "user", "source": "reminder", "trigger": ORIENT_ID,
             "detail": MODEL_ALT, "content": TAG_OPEN},
            {"role": "user", "source": "reminder", "trigger": ORIENT_ID,
             "detail": "anthropic/claude-test-c", "content": TAG_OPEN},
        ]
        eng = ReminderEngine(_cfg(tmp_path))
        eng.rehydrate(entries)
        assert eng._oriented_model == "anthropic/claude-test-c", \
            "rehydrate must apply session-orient details last-wins"
        res_c = eng.evaluate(
            _orient_turn_state(model_resolved="anthropic/claude-test-c",
                               model_vision=True))
        assert not any(r.trigger == ORIENT_ID for r in res_c), \
            "Evaluation with the last-oriented model must be suppressed"
        res_b = eng.evaluate(_orient_turn_state(model_resolved=MODEL_ALT,
                                                model_vision=False,
                                                context_limit=131072))
        assert any(r.trigger == ORIENT_ID for r in res_b), \
            "Evaluation with an earlier (non-last) model must fire"

    def test_t05c_reset_clears_oriented_model(self, tmp_path):
        """§5.5c: _oriented_model initialized in __init__ AND cleared in
        reset() (review MED-1 — omitting reset() silently breaks the
        rehydrate-resets-first invariant)."""
        eng = ReminderEngine(_cfg(tmp_path))
        assert eng._oriented_model is None, \
            "Fresh engine must initialize _oriented_model = None"
        first = eng.evaluate(_orient_turn_state())
        assert any(r.trigger == ORIENT_ID for r in first), \
            "Precondition: fresh engine fires once"
        eng.reset()
        assert eng._oriented_model is None, \
            "reset() must clear _oriented_model to None"
        again = eng.evaluate(_orient_turn_state())
        assert any(r.trigger == ORIENT_ID for r in again), \
            "After reset() (umbral), session-orient must re-arm and fire"

    def test_t06_rehydrate_idempotent_and_reset_rearms(self, tmp_path):
        """§5.6: rehydrate() called twice yields the same oriented state;
        reset() re-arms."""
        entries = [
            {"role": "user", "source": "reminder", "trigger": ORIENT_ID,
             "detail": MODEL_DEFAULT, "content": TAG_OPEN},
        ]
        eng = ReminderEngine(_cfg(tmp_path))
        eng.rehydrate(entries)
        eng.rehydrate(entries)  # second call — must be idempotent
        assert eng._oriented_model == MODEL_DEFAULT, \
            "Double rehydrate must leave the same oriented model"
        res = eng.evaluate(_orient_turn_state())
        assert not any(r.trigger == ORIENT_ID for r in res), \
            "Still suppressed after the second rehydrate"
        eng.reset()
        res2 = eng.evaluate(_orient_turn_state())
        assert any(r.trigger == ORIENT_ID for r in res2), \
            "reset() must re-arm session-orient"

    def test_t07_reminders_kill_switch_suppresses(self, tmp_path):
        """§5.7: [agent] reminders = false → empty evaluation (no new toggle;
        the existing kill-switch line covers the new trigger)."""
        eng = ReminderEngine(_cfg(tmp_path, reminders=False))
        results = eng.evaluate(_orient_turn_state())
        assert results == [], \
            f"reminders=False must suppress session-orient; got {results}"


# ============================================================================
# Spec §5 — Sink / JSONL unit tests (8–10)
# ============================================================================

class TestSinksAndJsonl:
    @pytest.mark.asyncio
    async def test_t08_sinks_append_detail_conditionally(self, tmp_path):
        """§5.8: MatrixSinks + HeadlessSinks append detail=... when the
        Reminder carries one; omit the key entirely otherwise (no schema
        noise for T1–T6 — test_A04_jsonl_entry_format must stay green)."""
        plain = Reminder(trigger="todo-nudge", text="plain")
        # Constructing a Reminder with detail= is part of the contract —
        # pre-implementation this raises TypeError (missing field). That IS
        # the desired red signal for the engine-side dataclass change.
        orient_rem = Reminder(trigger=ORIENT_ID, text="orient",
                              detail=MODEL_DEFAULT)

        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)

        headless = HeadlessSinks(session_log=sl, agent_user_id=AGENT_USER)
        await headless.log_reminder(ROOM_A, orient_rem)
        await headless.log_reminder(ROOM_A, plain)

        entries = sl.read(ROOM_A)
        assert len(entries) == 2
        orient_entry = [e for e in entries if e.get("trigger") == ORIENT_ID]
        assert orient_entry, "Orient entry must be persisted by HeadlessSinks"
        assert orient_entry[0].get("detail") == MODEL_DEFAULT, (
            "HeadlessSinks.log_reminder must append detail when the Reminder "
            "carries one"
        )
        plain_entry = [e for e in entries if e.get("trigger") == "todo-nudge"]
        assert plain_entry
        assert "detail" not in plain_entry[0], (
            "Plain T1–T6 reminders must NOT gain a detail key in JSONL "
            "(conditional pass-through — review LOW-13)"
        )

        # MatrixSinks — mocked bot with a MagicMock session_log capturing kwargs.
        bot = MagicMock()
        bot.config = MagicMock()
        bot.config.user_id = AGENT_USER
        bot.session_log = MagicMock()
        matrix_sinks = MatrixSinks(bot, ROOM_A)
        await matrix_sinks.log_reminder(ROOM_A, orient_rem)
        await matrix_sinks.log_reminder(ROOM_A, plain)

        append_calls = bot.session_log.append.call_args_list
        assert len(append_calls) == 2
        orient_call = [c for c in append_calls
                       if c.kwargs.get("trigger") == ORIENT_ID]
        assert orient_call and orient_call[0].kwargs.get("detail") == MODEL_DEFAULT, (
            "MatrixSinks.log_reminder must append detail when the Reminder "
            "carries one"
        )
        plain_call = [c for c in append_calls
                      if c.kwargs.get("trigger") == "todo-nudge"]
        assert plain_call and "detail" not in plain_call[0].kwargs, (
            "MatrixSinks must omit detail for plain reminders"
        )

    def test_t09_replay_verbatim_and_rehydrate_detail(self, tmp_path):
        """§5.9 (A06-style): real SessionLog; orient entry with detail replays
        VERBATIM via build_context, role==user, source==reminder; the same
        JSONL rehydrates the oriented model without raising."""
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)
        sl.append(role="user", sender="@op:x", room=ROOM_A, event_id="$e1",
                  content="hello")
        framed = (TAG_OPEN + "\n"
                  "Session orientation (inserted automatically at context-epoch start):\n"
                  f"- Session began: {ORIENT_TS}\n"
                  f"- Active model: {MODEL_DEFAULT}\n"
                  "- Context window: 200,000 tokens\n"
                  "- Vision: yes\n"
                  "</system-reminder>")
        sl.append(role="user", sender=AGENT_USER, room=ROOM_A, event_id=None,
                  content=framed, source="reminder", trigger=ORIENT_ID,
                  detail=MODEL_DEFAULT)

        context = sl.build_context(ROOM_A)
        orient_msgs = [m for m in context
                       if TAG_OPEN in str(m.get("content", ""))]
        assert len(orient_msgs) == 1, "Orient entry must appear in rebuilt context"
        assert orient_msgs[0]["content"] == framed, \
            "build_context must replay the orient content VERBATIM (byte-identity)"
        assert orient_msgs[0]["role"] == "user"
        raw = sl.read(ROOM_A)
        orient_raw = [e for e in raw if e.get("trigger") == ORIENT_ID]
        assert orient_raw[0].get("source") == "reminder"
        assert orient_raw[0].get("detail") == MODEL_DEFAULT, \
            "detail must round-trip through the JSONL"

        eng = ReminderEngine(_cfg(tmp_path))
        eng.rehydrate(raw)  # must not raise
        res = eng.evaluate(_orient_turn_state())
        assert not any(r.trigger == ORIENT_ID for r in res), \
            "After rehydrate(detail), the equal-model evaluation is suppressed"

    @pytest.mark.asyncio
    async def test_t10_cache_safety_strict_prefix_with_orient_injection(self, tmp_path):
        """§5.10 (A05-style): a real-path turn WITH the turn-start orient
        injection — every call's messages are a strict object-identity prefix
        of the next call's."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        callbacks = bot._build_agent_callbacks(ROOM_A, None)

        stream_fn, payloads = _make_capturing_stream(tool_iterations=2)
        mock_exec = AsyncMock(return_value=ToolResult(content="ok", is_error=False))
        with patch("openalph.agent.stream", side_effect=stream_fn), \
             patch("openalph.agent.execute_tool", mock_exec):
            await agent.handle_input("work", room_id=ROOM_A, callbacks=callbacks)

        history = agent.history(ROOM_A)
        assert _orient_history(history), (
            "A session-orient reminder MUST have been injected at turn start "
            "for this cache-safety test to be meaningful "
            "(no orient found — turn-start site does not populate the state yet)"
        )
        assert len(payloads) >= 2, "Need ≥2 API calls for the prefix property"
        for i in range(len(payloads) - 1):
            cur, nxt = payloads[i], payloads[i + 1]
            assert len(nxt) >= len(cur), (
                f"Call {i + 1} has fewer messages ({len(nxt)}) than call {i} "
                f"({len(cur)})"
            )
            for j in range(len(cur)):
                assert cur[j] is nxt[j], (
                    f"Message {j} differs between call {i} and {i + 1} — "
                    f"cache bust risk from the orient injection"
                )


# ============================================================================
# Spec §5 — Real-path integration tests (11–18 + 20–21)
# ============================================================================

class TestRealPath:
    """Real Agent + real _build_agent_callbacks; mocked provider + mocked nio.

    Pre-implementation these must fail at BEHAVIORAL assertions (no
    session-orient anywhere), never in fixture setup.
    """

    @pytest.mark.asyncio
    async def test_t11_fresh_room_first_message_fires_exactly_once(self, tmp_path):
        """§5.11: fresh room, first user message ⇒ exactly one session-orient
        in payloads[0] AND in agent.history; the four values match the
        fixture config (model, pinned window, pinned vision)."""
        sl = SessionLog(workspace=tmp_path / "sess", agent_user_id=AGENT_USER)
        bot, agent = _make_bot_with_real_agent(
            tmp_path, session_log=sl,
            model_limits={MODEL_DEFAULT: 200000},
            model_vision={MODEL_DEFAULT: True})
        callbacks = bot._build_agent_callbacks(ROOM_A, None)

        payloads = []
        await _run_turn(agent, callbacks, payloads_sink=payloads)

        history = agent.history(ROOM_A)
        orient_hist = _orient_history(history)
        assert len(orient_hist) == 1, (
            f"Exactly one session-orient must land in history on the first "
            f"user message; got {len(orient_hist)}"
        )
        assert payloads and _orient_history(payloads[0]), (
            "The first wire payload (payloads[0]) must already contain the "
            "session-orient injection (it precedes the first API call)"
        )
        content = orient_hist[0]["content"]
        assert f"Active model: {MODEL_DEFAULT}" in content
        assert "Context window: 200,000 tokens" in content
        assert "Vision: yes" in content
        assert "Session began: " in content, \
            "Content must carry a 'Session began' timestamp line"

        entries = sl.read(ROOM_A)
        orient_entries = _orient_entries(entries)
        assert len(orient_entries) == 1, \
            "The orient fire must be persisted to the JSONL exactly once"
        assert orient_entries[0].get("detail") == MODEL_DEFAULT, \
            "Persisted entry must carry detail = resolved model"

    @pytest.mark.asyncio
    async def test_t12_second_message_same_model_no_new_fire(self, tmp_path):
        """§5.12: second user message, same model ⇒ NO new fire. Establishes
        the positive precondition (fires once) first."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        callbacks = bot._build_agent_callbacks(ROOM_A, None)

        payloads1 = await _run_turn(agent, callbacks, text="first")
        combined1 = _orient_history(payloads1[0]) + _orient_history(
            agent.history(ROOM_A))
        assert len(combined1) >= 1 and any(_orient_history(payloads1[0])), (
            "First turn must fire session-orient (positive precondition)"
        )

        payloads2 = await _run_turn(agent, callbacks, text="second")
        assert payloads2, "Second turn must produce a wire payload"
        history = agent.history(ROOM_A)
        assert len(_orient_history(history)) == 1, (
            f"Same-model second turn must NOT re-fire; history now has "
            f"{len(_orient_history(history))} orient messages"
        )
        jsonl_kwargs = [c.kwargs for c in bot.session_log.append.call_args_list]
        assert len(_orient_entries(jsonl_kwargs)) == 1, \
            "JSONL must contain exactly one orient entry after two turns"

    @pytest.mark.asyncio
    async def test_t13_model_switch_refires_with_new_model(self, tmp_path):
        """§5.13: switch_model(fireworks/...) then next user message ⇒ one
        new fire whose model line (and detail) reflect the SWITCHED model.
        Fixture: two providers + pinned limits/vision per spec §5 note."""
        sl = SessionLog(workspace=tmp_path / "sess", agent_user_id=AGENT_USER)
        bot, agent = _make_bot_with_real_agent(
            tmp_path, session_log=sl,
            providers={
                "anthropic": ProviderConfig(
                    key="anthropic", type="anthropic", api_key="sk-test",
                    base_url=None, quirks=[],
                ),
                "fireworks": ProviderConfig(
                    key="fireworks", type="openai", api_key="sk-test-2",
                    base_url="https://api.fireworks.test", quirks=[],
                ),
            },
            model_limits={MODEL_DEFAULT: 200000, MODEL_ALT: 131072},
            model_vision={MODEL_DEFAULT: True, MODEL_ALT: False})
        callbacks = bot._build_agent_callbacks(ROOM_A, None)

        await _run_turn(agent, callbacks, text="first")
        history = agent.history(ROOM_A)
        assert len(_orient_history(history)) == 1, \
            "Positive precondition: first turn fires once"

        err = agent.switch_model(MODEL_ALT, ROOM_A)
        assert err is None, f"switch_model must succeed with the two-provider fixture: {err}"
        await _run_turn(agent, callbacks, text="second")

        history = agent.history(ROOM_A)
        orient_msgs = _orient_history(history)
        assert len(orient_msgs) == 2, (
            f"Model switch must cause exactly one re-fire; history has "
            f"{len(orient_msgs)} orient messages"
        )
        refire = orient_msgs[-1]["content"]
        assert f"Active model: {MODEL_ALT}" in refire, \
            "Re-fired nudge must name the switched model"
        assert "Context window: 131,072 tokens" in refire, \
            "Re-fired nudge must carry the new model's window"
        assert "Vision: no" in refire, \
            "Re-fired nudge must reflect the new model's vision flag"

        orient_entries = _orient_entries(sl.read(ROOM_A))
        assert len(orient_entries) == 2
        assert orient_entries[-1].get("detail") == MODEL_ALT, \
            "Second orient entry's detail must carry the switched model"

    @pytest.mark.asyncio
    async def test_t14_umbral_reset_room_rearms_with_real_sessionlog(self, tmp_path):
        """§5.14: reset_room (umbral seam) re-arms; with a REAL SessionLog,
        archive+wipe then first message ⇒ fires again."""
        # In-memory seam only first
        bot, agent = _make_bot_with_real_agent(tmp_path)
        callbacks = bot._build_agent_callbacks(ROOM_A, None)
        await _run_turn(agent, callbacks, text="first")
        assert len(_orient_history(agent.history(ROOM_A))) == 1, \
            "Positive precondition: first turn fires once"
        agent.reset_room(ROOM_A)
        await _run_turn(agent, callbacks, text="post-umbral")
        assert len(_orient_history(agent.history(ROOM_A))) == 1, (
            "After reset_room (umbral seam), the first post-reset message "
            "must fire session-orient again (fresh epoch)"
        )

        # Real-SessionLog archive+wipe variant
        tmp2 = tmp_path / "umbral"
        tmp2.mkdir()
        sl = SessionLog(workspace=tmp2 / "sess", agent_user_id=AGENT_USER)
        bot2, agent2 = _make_bot_with_real_agent(tmp2, session_log=sl)
        cb2 = bot2._build_agent_callbacks(ROOM_A, None)
        await _run_turn(agent2, cb2, text="first")
        assert len(_orient_entries(sl.read(ROOM_A))) == 1, \
            "Positive precondition: orient persisted before umbral"
        sl.archive(ROOM_A)
        sl.wipe(ROOM_A)
        agent2.reset_room(ROOM_A)
        await _run_turn(agent2, cb2, text="post-umbral")
        entries = sl.read(ROOM_A)
        assert len(_orient_entries(entries)) == 1, (
            "Post-umbral fresh JSONL must contain a new session-orient fire"
        )

    @pytest.mark.asyncio
    async def test_t15_heartbeat_and_umbral_turn_sources_fire_identically(self, tmp_path):
        """§5.15: the turn-start evaluation site is source-agnostic —
        heartbeat and umbral turn_source fire the same nudge."""
        for source in ("heartbeat", "umbral"):
            ws_root = tmp_path / source
            ws_root.mkdir()
            ws = _setup_workspace(ws_root)
            config = _cfg(ws)
            agent = Agent(config)
            bot, _ = _make_bot_with_real_agent(ws_root, agent=agent)
            callbacks = bot._build_agent_callbacks(ROOM_A, source)
            await _run_turn(agent, callbacks, room_id=ROOM_A,
                            text=f"{source} directive")
            orient = _orient_history(agent.history(ROOM_A))
            assert len(orient) == 1, (
                f"turn_source={source!r} must fire session-orient on its "
                f"first turn exactly once; got {len(orient)}"
            )

    @pytest.mark.asyncio
    async def test_t16_stale_provider_override_fail_soft(self, tmp_path):
        """§5.16: room override pointing at an unconfigured provider —
        resolve_model raises ⇒ fail-soft: nudge still fires with the RAW
        model string, turn completes, detail carries the raw string."""
        stale = "zzz-notaconfiguredprovider/some-model-x"
        sl = SessionLog(workspace=tmp_path / "sess", agent_user_id=AGENT_USER)
        _append_model_override(sl, ROOM_A, stale)
        bot, agent = _make_bot_with_real_agent(tmp_path, session_log=sl)
        # Umbral+restart restore edge: matrix.py:1887 assigns directly.
        agent._room_models[ROOM_A] = stale
        callbacks = bot._build_agent_callbacks(ROOM_A, None)

        payloads = []
        await _run_turn(agent, callbacks, payloads_sink=payloads)  # must not raise

        orient = _orient_history(agent.history(ROOM_A))
        assert len(orient) == 1, (
            "Fail-soft: session-orient must still fire with the raw model "
            "string when resolve_model raises on a stale override"
        )
        assert f"Active model: {stale}" in orient[0]["content"], \
            "Fail-soft content must show the RAW override string"
        entries = _orient_entries(sl.read(ROOM_A))
        assert entries and entries[0].get("detail") == stale, (
            "detail must carry the RAW string under fail-soft (review LOW-2: "
            "the compared form flips fail-soft)"
        )

    def test_t23_bare_alias_resolves_to_provider_key(self, tmp_path):
        """Audit reconciliation HIGH-1 (kimi3+qwen38 convergence): a bare-alias
        /model override must resolve to '<provider.key>/<api_model>' using the
        RESOLVED provider's key — not '<alias>/<api_model>'. The model-keyed
        cap and the display line both depend on this canonical form; without
        it, '/model myalias' and '/model anthropic/claude-test-a' key as two
        DIFFERENT models and the nudge's model line is wrong."""
        ws = _setup_workspace(tmp_path)
        cfg = _cfg(ws, model_aliases={"myalias": MODEL_DEFAULT})
        agent = Agent(cfg)
        # Bare alias is the persisted form of '/model myalias'
        # (matrix restore assigns raw at matrix.py:1887).
        agent._room_models[ROOM_A] = "myalias"
        inputs = agent._orient_inputs(ROOM_A)
        assert inputs["model_resolved"] == MODEL_DEFAULT, (
            f"bare alias must resolve to '<provider.key>/<api>': "
            f"got {inputs['model_resolved']!r}"
        )

    @pytest.mark.asyncio
    async def test_t17_vision_yes_and_fail_closed_no(self, tmp_path):
        """§5.17: pinned vision=True model ⇒ 'Vision: yes'; an unknown model
        outside every table ⇒ fail-closed 'Vision: no'."""
        unknown = "anthropic/uncharacterized-xzqz-9000"
        sl = SessionLog(workspace=tmp_path / "sess", agent_user_id=AGENT_USER)
        bot, agent = _make_bot_with_real_agent(
            tmp_path, session_log=sl,
            model_limits={MODEL_DEFAULT: 200000},
            model_vision={MODEL_DEFAULT: True})
        callbacks = bot._build_agent_callbacks(ROOM_A, None)

        await _run_turn(agent, callbacks, text="first")
        orient = _orient_history(agent.history(ROOM_A))
        assert orient and "Vision: yes" in orient[0]["content"], (
            "model_vision override True must surface as 'Vision: yes' "
            "(positive precondition)"
        )

        err = agent.switch_model(unknown, ROOM_A)
        assert err is None, f"switch to an unknown anthropic/* model must validate: {err}"
        await _run_turn(agent, callbacks, text="second")
        orient = _orient_history(agent.history(ROOM_A))
        assert len(orient) == 2, "Switch must re-fire for the unknown model"
        assert "Vision: no" in orient[-1]["content"], (
            "Uncharacterized model must fail closed to 'Vision: no' "
            "(model_supports_vision Layer 3)"
        )

    @pytest.mark.asyncio
    async def test_t18_headless_path_jsonl_and_no_nio(self, tmp_path):
        """§5.18: build_callbacks with HeadlessSinks ⇒ JSONL entry has
        trigger + detail, nothing raises without a nio client."""
        ws = _setup_workspace(tmp_path)
        config = _cfg(ws)
        agent = Agent(config)
        sl = SessionLog(workspace=tmp_path / "sess", agent_user_id=AGENT_USER)
        sinks = HeadlessSinks(session_log=sl, agent_user_id=AGENT_USER)
        callbacks = build_callbacks(agent, ROOM_A, sinks,
                                    turn_source=None, session_log=sl,
                                    room_name="CLI")

        await _run_turn(agent, callbacks)

        entries = sl.read(ROOM_A)
        orient = _orient_entries(entries)
        assert len(orient) == 1, (
            "Headless path must persist the session-orient entry to JSONL "
            "(log_reminder via HeadlessSinks)"
        )
        assert orient[0].get("trigger") == ORIENT_ID
        assert orient[0].get("detail") == MODEL_DEFAULT, \
            "Headless sink must pass detail through"

    @pytest.mark.asyncio
    async def test_t19_sub_agent_sentinel_room_never_fires(self, tmp_path):
        """§5.19: the __sub__ sentinel room gets NO orientation fields at the
        turn-start site (decision 7) ⇒ orientation never fires there.
        Positive precondition in a normal room pins the non-vacuity."""
        bot, agent = _make_bot_with_real_agent(tmp_path)
        cb_normal = bot._build_agent_callbacks(ROOM_A, None)
        cb_sub = bot._build_agent_callbacks(ROOM_SUB, None)

        await _run_turn(agent, cb_normal, text="hello", room_id=ROOM_A)
        assert len(_orient_history(agent.history(ROOM_A))) == 1, (
            "Positive precondition: normal room fires once"
        )

        await _run_turn(agent, cb_sub, text="sub task", room_id=ROOM_SUB)
        orient_sub = _orient_history(agent.history(ROOM_SUB))
        assert not orient_sub, (
            "__sub__ sentinel room must NEVER receive a session-orient nudge — "
            "the helper must not populate the orientation fields there "
            "(defaults model_resolved='' == _oriented_model suppression)"
        )

    @pytest.mark.asyncio
    async def test_t20_per_room_isolation(self, tmp_path):
        """§5.20 (review LOW-5a): rooms A and B run different models; A's fire
        doesn't suppress B's; reset_room(B) re-arms B without disturbing A."""
        sl = SessionLog(workspace=tmp_path / "sess", agent_user_id=AGENT_USER)
        bot, agent = _make_bot_with_real_agent(
            tmp_path, session_log=sl,
            providers={
                "anthropic": ProviderConfig(
                    key="anthropic", type="anthropic", api_key="sk-test",
                    base_url=None, quirks=[],
                ),
                "fireworks": ProviderConfig(
                    key="fireworks", type="openai", api_key="sk-test-2",
                    base_url="https://api.fireworks.test", quirks=[],
                ),
            },
            model_limits={MODEL_DEFAULT: 200000, MODEL_ALT: 131072},
            model_vision={MODEL_DEFAULT: True, MODEL_ALT: False})
        err = agent.switch_model(MODEL_ALT, ROOM_B)
        assert err is None, f"Room B model override must validate: {err}"

        cb_a = bot._build_agent_callbacks(ROOM_A, None)
        cb_b = bot._build_agent_callbacks(ROOM_B, None)

        await _run_turn(agent, cb_a, text="work A", room_id=ROOM_A)
        await _run_turn(agent, cb_b, text="work B", room_id=ROOM_B)

        hist_a, hist_b = agent.history(ROOM_A), agent.history(ROOM_B)
        assert len(_orient_history(hist_a)) == 1, \
            "Room A must have its own session-orient fire"
        assert len(_orient_history(hist_b)) == 1, (
            "Room B must NOT be suppressed by room A's fire — engines are "
            "per-room and keyed to each room's own model"
        )
        assert f"Active model: {MODEL_DEFAULT}" in _orient_history(hist_a)[0]["content"]
        assert f"Active model: {MODEL_ALT}" in _orient_history(hist_b)[0]["content"]

        # reset_room(B) re-arms B without disturbing A
        agent.reset_room(ROOM_B)
        await _run_turn(agent, cb_b, text="work B2", room_id=ROOM_B)
        await _run_turn(agent, cb_a, text="work A2", room_id=ROOM_A)
        assert len(_orient_history(agent.history(ROOM_B))) == 1, \
            "Room B must re-fire after its own reset"
        assert len(_orient_history(agent.history(ROOM_A))) == 1, (
            "Room A's oriented state must survive room B's reset "
            "(no cross-room engine contamination)"
        )

    @pytest.mark.asyncio
    async def test_t21_mid_session_rebuild_no_refire(self, tmp_path):
        """§5.21 (review LOW-5b): after a fire, force the gated-room rebuild
        pattern (history.clear(); history.extend(build_context(...))) WITHOUT
        rehydrate ⇒ the next turn does NOT re-fire (in-memory engine state
        survives the rebuild)."""
        sl = SessionLog(workspace=tmp_path / "sess", agent_user_id=AGENT_USER)
        bot, agent = _make_bot_with_real_agent(tmp_path, session_log=sl)
        callbacks = bot._build_agent_callbacks(ROOM_A, None)

        await _run_turn(agent, callbacks, text="first")
        assert len(_orient_history(agent.history(ROOM_A))) == 1, \
            "Positive precondition: first turn fires once"

        # The gated-room rebuild pattern — WITHOUT rehydrate.
        history = agent.history(ROOM_A)
        history.clear()
        history.extend(sl.build_context(ROOM_A))

        await _run_turn(agent, callbacks, text="second")
        assert len(_orient_history(agent.history(ROOM_A))) == 1, (
            "Mid-session rebuild without rehydrate must NOT re-fire — the "
            "in-memory engine's _oriented_model survives history replacement"
        )


# ============================================================================
# Spec §5 — test 22 guard note
# ============================================================================
# Test 22 (boundary-site completeness, review LOW-9/GLM) is implicitly covered
# by the multi-iteration real-path tests above: tests t10 (tool_iterations=2)
# runs the turn through ≥2 tool-loop boundary iterations, exercising the
# boundary ReminderState construction site (~agent.py:850) with the new
# defaulted fields in place. A missing/default-less field there would raise
# TypeError inside those tests. No dedicated test exists by design — see the
# operator directive ("Test 22 guard: just make sure at least one real-path
# test uses tool_iterations >= 2"; satisfied by test_t10).
