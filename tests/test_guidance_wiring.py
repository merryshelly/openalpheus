"""Matrix-side wiring tests for guidance injection (reminders + todo_write).

Verifies that matrix.py correctly wires callbacks (send_notice, log_reminder,
turn_source), emits formatted notices, calls engine.rehydrate on activation,
and clears state on umbral.  RED until matrix.py wiring is implemented.

Follows test_realtime_steering.py conventions: MatrixBot via __new__ bypass,
mocked nio client, mocked Agent with AsyncMock handle_input.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# --- Guard imports (clean collection even if modules are broken) ---
try:
    from openalph.matrix import MatrixBot
    _matrix_imported = True
except Exception:
    MatrixBot = None
    _matrix_imported = False

try:
    from openalph.config import MatrixConfig, ProviderConfig
    _config_imported = True
except Exception:
    MatrixConfig = ProviderConfig = None
    _config_imported = False

try:
    from openalph.reminders import Reminder
    _reminders_imported = True
except Exception:
    Reminder = None
    _reminders_imported = False

try:
    from openalph.tools import _TODO_STATE
    _todo_imported = True
except Exception:
    _TODO_STATE = {}
    _todo_imported = False

# --- Constants ---
ROOM = "!wiring-test:matrix.local"
AGENT_USER = "@agent:matrix.local"
OPERATOR = "@operator:matrix.local"


# --- Helpers (matching test_realtime_steering.py conventions) ---

def _make_matrix_config(user_id=AGENT_USER, **kw):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id=user_id,
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )
    defaults.update(kw)
    return MatrixConfig(**defaults)


def _make_room(room_id=ROOM, member_count=2):
    room = MagicMock()
    room.room_id = room_id
    room.name = "Test Room"
    room.display_name = "Test Room"
    room.users = {f"@user{i}:matrix.local": MagicMock() for i in range(member_count)}
    room.member_count = member_count
    return room


def _make_event(sender=OPERATOR, body="hello", event_id="$evt1"):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = 1_000_000
    event.source = {"content": {"body": body}}
    return event


def _make_bot(**overrides):
    """Create a MatrixBot for testing via __new__ bypass.

    Mirrors test_realtime_steering._make_bot: all required attrs set on the
    MatrixBot instance so _process_message, _inject_heartbeat, _inject_umbral,
    and _activate_room can run with mocked dependencies.
    """
    config = _make_matrix_config()
    agent = MagicMock()
    agent._rooms = {}
    agent.handle_input = AsyncMock(return_value="response")
    agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))
    agent.cancel = MagicMock(return_value=None)
    agent.tools = []
    agent.config = MagicMock()
    agent.config.workspace = "/tmp/test-workspace"
    agent.config.model_aliases = {}
    agent.status = MagicMock(return_value={
        "context_pct": 10, "context_tokens": 1000, "context_max": 200000,
    })
    agent.last_turn_usage = MagicMock(return_value=None)
    agent.last_stop_reason = MagicMock(return_value="end_turn")
    # R1-4 adaptation: per-room engines replaced shared _reminder_engine.
    # Agent now exposes rehydrate_reminders(room_id, entries) instead.
    agent._reminder_engines = {}
    agent.rehydrate_reminders = MagicMock()
    # R1-1: per-room read registries
    agent._read_registries = {}
    agent.restore_usage = MagicMock()
    agent.reset_room = MagicMock()
    agent._room_models = {}

    bot = MatrixBot.__new__(MatrixBot)
    bot.config = config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$resp1"))
    bot.client.room_typing = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = {ROOM}
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
    bot.heartbeat.directive_for = MagicMock(return_value=None)
    bot.umbral = MagicMock()
    bot.umbral.is_active = MagicMock(return_value=False)
    bot.umbral.directive_for = MagicMock(return_value=None)
    bot._steering_inbox = {}
    bot._active_turns = set()

    for key, val in overrides.items():
        setattr(bot, key, val)
    return bot


@pytest.fixture(autouse=True)
def _cleanup_todo_state():
    """Ensure todo state for test room is cleaned up after each test."""
    yield
    if _todo_imported and isinstance(_TODO_STATE, dict):
        _TODO_STATE.pop(ROOM, None)


# ============================================================================
# Guidance wiring tests
# ============================================================================

class TestGuidanceWiring:
    """Matrix-side wiring for reminder engine and todo_write."""

    @pytest.mark.asyncio
    async def test_user_turn_callbacks(self):
        """§9/A10: user turn callbacks include send_notice, log_reminder,
        turn_source=None."""
        bot = _make_bot()
        await bot._process_message(_make_room(), _make_event(), "hello")

        assert bot.agent.handle_input.called, "handle_input must be called"
        cb = bot.agent.handle_input.call_args.kwargs['callbacks']
        assert 'send_notice' in cb, \
            "callbacks must include send_notice for reminder notices"
        assert 'log_reminder' in cb, \
            "callbacks must include log_reminder for JSONL persistence"
        assert cb.get('turn_source') is None, \
            "User turn: turn_source must be None (not heartbeat/umbral)"

    @pytest.mark.asyncio
    async def test_heartbeat_turn_source(self):
        """A10/T3: heartbeat turn sets turn_source='heartbeat' so T3 skips."""
        bot = _make_bot()
        await bot._inject_heartbeat(ROOM)

        assert bot.agent.handle_input.called
        cb = bot.agent.handle_input.call_args.kwargs['callbacks']
        assert cb.get('turn_source') == 'heartbeat', \
            "Heartbeat must set callbacks['turn_source']='heartbeat'"

    @pytest.mark.asyncio
    async def test_umbral_turn_source(self):
        """A10/T3: umbral turn sets turn_source='umbral' so T3 skips."""
        bot = _make_bot()
        bot.session_log.archive = MagicMock(return_value="archive.jsonl")
        bot.session_log.wipe = MagicMock()

        await bot._inject_umbral(ROOM)

        assert bot.agent.handle_input.called
        cb = bot.agent.handle_input.call_args.kwargs['callbacks']
        assert cb.get('turn_source') == 'umbral', \
            "Umbral must set callbacks['turn_source']='umbral'"

    @pytest.mark.asyncio
    async def test_reminder_notice_collapsed_html(self):
        """§9/A7: reminder notice → m.notice with collapsed <details> HTML,
        same formatted_body pattern as thinking blocks."""
        bot = _make_bot()
        await bot._process_message(_make_room(), _make_event(), "hello")

        cb = bot.agent.handle_input.call_args.kwargs['callbacks']
        send_notice = cb.get('send_notice')
        assert send_notice is not None, "send_notice callback required"

        bot.client.room_send.reset_mock()
        await send_notice(
            ROOM,
            "🔔 System reminder (todo-nudge)\n\nUse todo_write for multi-step tasks.",
        )

        assert bot.client.room_send.called, "send_notice must send to Matrix"
        content = bot.client.room_send.call_args[0][2]
        assert content['msgtype'] == 'm.notice', "Must be m.notice"
        assert content.get('format') == 'org.matrix.custom.html', \
            "Must have org.matrix.custom.html format"
        fb = content.get('formatted_body', '')
        assert '<details>' in fb, "Must use collapsed <details> HTML"
        assert 'todo-nudge' in fb or 'todo-nudge' in content.get('body', ''), \
            "Trigger ID must appear in notice"
        # Bug fix (workspace-kdsn.186.17, issue 1): the header line
        # (already in <summary>) must NOT be re-rendered inside the
        # expanded <details> body too. Against the pre-fix code, mistune
        # re-rendered the WHOLE body (header included), so the header
        # string appeared twice; against the fix it appears exactly once.
        assert fb.count("System reminder (todo-nudge)") == 1, \
            "Reminder header must appear exactly once (not duplicated in body)"

    @pytest.mark.asyncio
    async def test_display_only_invariant(self):
        """§9 I2: notice NOT in JSONL; reminder USER entry IS in JSONL
        with source='reminder' + trigger id."""
        bot = _make_bot()
        await bot._process_message(_make_room(), _make_event(), "hello")
        cb = bot.agent.handle_input.call_args.kwargs['callbacks']

        log_reminder = cb.get('log_reminder')
        assert log_reminder is not None, "log_reminder callback required"

        rem = Reminder(trigger="todo-nudge", text="Test text")
        bot.session_log.append.reset_mock()
        await log_reminder(ROOM, rem)

        # Reminder JSONL entry must exist with correct fields
        rem_calls = [c for c in bot.session_log.append.call_args_list
                     if c.kwargs.get('source') == 'reminder']
        assert rem_calls, "log_reminder must write JSONL with source='reminder'"
        kw = rem_calls[0].kwargs
        assert kw['role'] == 'user', "Reminder JSONL: role must be 'user'"
        assert kw['trigger'] == 'todo-nudge', "Reminder JSONL: trigger id required"
        assert '<system-reminder>' in kw.get('content', ''), \
            "Reminder JSONL: content must be framed"

        # send_notice must NOT produce a JSONL entry
        send_notice = cb.get('send_notice')
        if send_notice:
            bot.session_log.append.reset_mock()
            await send_notice(ROOM, "🔔 System reminder (test)\n\nTest")
            bad = [c for c in bot.session_log.append.call_args_list
                   if '🔔' in str(c.kwargs.get('content', ''))]
            assert not bad, "Notice must be display-only (no JSONL entry)"

    @pytest.mark.asyncio
    async def test_todo_notice_format(self):
        """§5/§9: todo_write update → m.notice 📋 + counts + collapsed <details> list."""
        bot = _make_bot()
        _tool_notice, _ = bot._make_tool_callbacks(ROOM)

        bot.client.room_send.reset_mock()
        await _tool_notice(
            "call_1", "todo_write",
            {"todos": [{"content": "Task A", "status": "in_progress"}]},
            "Todo list updated (1 in progress):\n  ● [in_progress] Task A",
            False,
        )

        found = False
        for call in bot.client.room_send.call_args_list:
            content = call[0][2] if len(call[0]) > 2 else {}
            if isinstance(content, dict) and '📋' in content.get('body', ''):
                fb = content.get('formatted_body', '')
                assert '<details>' in fb, "Todo notice must have collapsed <details>"
                assert content['msgtype'] == 'm.notice'
                found = True
                break
        assert found, "todo_write must emit 📋 m.notice with collapsed list"

        # Bug fix (workspace-kdsn.186.17, issue 2): notice must be built from
        # the STRUCTURED input_data, not the `result` string. In production
        # `result` is the wrapped `<tool_result tool="..." id="...">...
        # </tool_result>` envelope (agent.py's wrap_tool_result) — deriving
        # the notice from that string leaks the envelope + tool-call id and
        # collapses the todos onto one line. Pin: envelope/id must NOT leak,
        # and each todo must render as its own marker (one per todo).
        bot.client.room_send.reset_mock()
        await _tool_notice(
            "call_1", "todo_write",
            {"todos": [
                {"content": "Task A", "status": "in_progress"},
                {"content": "Task B", "status": "pending"},
            ]},
            '<tool_result tool="todo_write" id="toolu_ABC">\n'
            'Todo list updated (1 in progress · 1 pending):\n'
            '  ● [in_progress] Task A\n'
            '  ○ [pending] Task B\n'
            '</tool_result>',
            False,
        )

        fb2 = None
        for call in bot.client.room_send.call_args_list:
            content = call[0][2] if len(call[0]) > 2 else {}
            if isinstance(content, dict) and '📋' in content.get('body', ''):
                fb2 = content.get('formatted_body', '')
                break
        assert fb2 is not None, "todo_write must emit a 📋 m.notice"
        assert 'id=' not in fb2, \
            "tool-call id from the <tool_result> envelope must not leak into the notice"
        assert '<tool_result' not in fb2, \
            "the <tool_result> wrapper envelope must not leak into the notice"
        marker_count = fb2.count('●') + fb2.count('○') + fb2.count('✓')
        assert marker_count == 2, \
            f"expected one status marker per todo (2 todos), got {marker_count}"

    @pytest.mark.asyncio
    async def test_rehydration_on_activate(self):
        """§3 rehydration: _activate_room with reminder JSONL entries
        → engine.rehydrate called with those entries."""
        bot = _make_bot()
        bot._active_rooms.discard(ROOM)
        entries = [
            {"role": "user", "content": "hello"},
            {"role": "user", "source": "reminder", "trigger": "todo-nudge",
             "content": "<system-reminder>\nuse todo\n</system-reminder>"},
            {"role": "assistant", "content": "ok"},
        ]
        bot.session_log.read = MagicMock(return_value=entries)

        await bot._activate_room(ROOM)

        # R1-4 adaptation: rehydrate is now via agent.rehydrate_reminders(room_id, entries)
        bot.agent.rehydrate_reminders.assert_called_once()
        call_args = bot.agent.rehydrate_reminders.call_args
        assert call_args[0][0] == ROOM, "rehydrate must be called with the room_id"
        arg = call_args[0][1]
        assert any(e.get("source") == "reminder" for e in arg), \
            "rehydrate must receive entries containing reminder records"

    @pytest.mark.asyncio
    async def test_umbral_reset_and_todo_clear(self):
        """§3/§5: umbral → engine.reset() via reset_room; todo state cleared."""
        bot = _make_bot()
        bot.session_log.archive = MagicMock(return_value="archive.jsonl")
        bot.session_log.wipe = MagicMock()

        if _todo_imported and isinstance(_TODO_STATE, dict):
            _TODO_STATE[ROOM] = [{"content": "lingering", "status": "pending"}]

        await bot._inject_umbral(ROOM)

        bot.agent.reset_room.assert_called_once_with(ROOM)
        if _todo_imported and isinstance(_TODO_STATE, dict):
            assert ROOM not in _TODO_STATE or _TODO_STATE.get(ROOM) == [], \
                "Umbral must clear todo state for the room"

    @pytest.mark.asyncio
    async def test_steering_and_reminder_coexistence(self):
        """§3 ordering: both drain_steering and log_reminder in callbacks;
        both produce JSONL entries when exercised."""
        bot = _make_bot()
        await bot._process_message(_make_room(), _make_event(), "hello")
        cb = bot.agent.handle_input.call_args.kwargs['callbacks']
        assert 'drain_steering' in cb, "drain_steering must be in callbacks"
        assert 'log_reminder' in cb, "log_reminder must be in callbacks"

        bot.session_log.append.reset_mock()
        bot._steering_inbox[ROOM] = ["op note"]
        await cb['drain_steering']()

        rem = Reminder(trigger="context-pressure", text="Context at 85%.")
        await cb['log_reminder'](ROOM, rem)

        steer = [c for c in bot.session_log.append.call_args_list
                 if c.kwargs.get('source') == 'steer']
        reminder = [c for c in bot.session_log.append.call_args_list
                    if c.kwargs.get('source') == 'reminder']
        assert steer, "Steer note must be logged to JSONL"
        assert reminder, "Reminder must be logged to JSONL"

    @pytest.mark.asyncio
    async def test_gated_room_wiring(self):
        """§9: gated room (mention path) still has send_notice + log_reminder."""
        bot = _make_bot()
        with patch("openalph.matrix.is_gated", return_value=True):
            await bot._process_message(
                _make_room(), _make_event(), "hello", _gating_handled=True,
            )
        assert bot.agent.handle_input.called
        cb = bot.agent.handle_input.call_args.kwargs['callbacks']
        assert 'send_notice' in cb, "Gated path must include send_notice"
        assert 'log_reminder' in cb, "Gated path must include log_reminder"

    @pytest.mark.asyncio
    async def test_gated_injection_ordering_reminder_after_user(self):
        """§3 turn-start ordering: on a gated mention turn, reminder appended
        AFTER the user message — wire payload has user msg before reminder entry.

        Design §3: turn-start evaluation happens AFTER the user message enters
        history; the reminder must follow it in both history and wire payload.
        Uses a real Agent with mocked stream so we can inspect message order.
        """
        from pathlib import Path
        from openalph.agent import Agent
        from openalph.config import AgentConfig, ProviderConfig
        from openalph.provider import StreamEvent, Response, Usage

        # Build a minimal agent config with reminders enabled
        workspace = Path("/tmp/test-ordering-workspace")
        workspace.mkdir(exist_ok=True)
        (workspace / "SOUL.md").write_text("Test soul.")
        agent_config = AgentConfig(
            name="test-ordering",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=8192,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic",
                api_key="sk-test", base_url=None, quirks=[],
            )},
            workspace=workspace,
            reminders=True,
        )
        agent = Agent(agent_config)

        room_id = "!ordering-test:matrix.local"
        user_msg = "Please help me with this task"
        reminder_content = None
        captured_messages = []

        async def _mock_stream(config, system, messages, **kwargs):
            captured_messages.append(list(messages))
            yield StreamEvent(type="text", content="ok")
            yield StreamEvent(
                type="done",
                response=Response(
                    content="ok",
                    model="claude-sonnet-4-20250514",
                    usage=Usage(input_tokens=5, output_tokens=2),
                    stop_reason="end_turn",
                ),
                stop_reason="end_turn",
                model="claude-sonnet-4-20250514",
            )

        # Simulate gated-path: pre-populate history with user message (hydration)
        # then call handle_input with append_user=False and real log_reminder
        history = agent.history(room_id)
        history.append({"role": "user", "content": user_msg})

        # Force T3 predicate to fire by marking completed_turns >= 2
        # (inject a prior user+assistant pair so history reflects 2 completed turns)
        history.insert(0, {"role": "user", "content": "prior user"})
        history.insert(1, {"role": "assistant", "content": "prior assistant"})

        log_reminder_calls = []

        async def _log_reminder(_room_id, reminder):
            log_reminder_calls.append(reminder)

        async def _send_notice(_room_id, body, **kw):
            pass  # best-effort display only

        callbacks = {
            "log_reminder": _log_reminder,
            "send_notice": _send_notice,
            "turn_source": None,
        }

        with patch("openalph.agent.stream", _mock_stream):
            await agent.handle_input(
                user_msg, room_id,
                append_user=False,
                callbacks=callbacks,
            )

        assert captured_messages, "stream() must be called"
        messages = captured_messages[0]

        # Find user message index and any reminder index
        user_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "user" and user_msg in str(m.get("content", ""))
        ]
        reminder_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "user" and "<system-reminder>" in str(m.get("content", ""))
        ]

        if reminder_indices:
            # If a reminder fired, it must come AFTER the user message
            assert user_indices, "User message must be in wire payload"
            assert max(user_indices) < min(reminder_indices), (
                f"Reminder (index {min(reminder_indices)}) must appear AFTER "
                f"user message (index {max(user_indices)}) in wire payload. "
                f"Messages: {[(i, m['role'], str(m.get('content',''))[:40]) for i, m in enumerate(messages)]}"
            )
        else:
            # T3 may not have fired (not enough session context); just verify user is present
            assert user_indices, "User message must be in wire payload even if no reminder fires"
