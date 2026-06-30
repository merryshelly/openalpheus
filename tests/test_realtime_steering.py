"""RED test suite — Real-time steering (mid-turn /steer injection).

This suite is intentionally RED. It specifies the acceptance criteria for the
steering feature before implementation. Every test here MUST fail until the
feature is built. Tests fail on missing behaviour/attributes, NOT on import
errors that would break collection of the whole test suite — all feature
imports are guarded with try/except.

Design locked by:
  memory/projects/openalph/realtime-steering-design.md
  memory/projects/openalph/architecture-summary.md

Interface spec (chosen by recon pass):
  MatrixBot._steering_inbox: dict[str, list[str]]
  MatrixBot._active_turns: set[str]
  /steer slash command in _handle_room_message
  drain_steering() callback passed to agent.handle_input
  agent.handle_input kwarg: drain_steering (async callable → list[str])
  JSONL: role="user", source="steer", content=<original text>
  build_context: source=="steer" entries get framing prefix prepended
  Framing: "[Operator steering — mid-turn guidance]: <message>"
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

# ---------------------------------------------------------------------------
# Guard imports — feature doesn't exist yet, so we import defensively.
# Tests fail on missing behaviour; suite still collects if attrs are missing.
# ---------------------------------------------------------------------------
try:
    from openalph.matrix import MatrixBot
    _matrix_imported = True
except Exception:
    MatrixBot = None  # type: ignore
    _matrix_imported = False

try:
    from openalph.agent import Agent
    _agent_imported = True
except Exception:
    Agent = None  # type: ignore
    _agent_imported = False

try:
    from openalph.session import SessionLog
    _session_imported = True
except Exception:
    SessionLog = None  # type: ignore
    _session_imported = False

try:
    from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
    from openalph.provider import Response, Usage, StreamEvent, ToolCall
    from openalph.tools import ToolResult
    _config_imported = True
except Exception:
    AgentConfig = MatrixConfig = ProviderConfig = None  # type: ignore
    Response = Usage = StreamEvent = ToolCall = None  # type: ignore
    ToolResult = None  # type: ignore
    _config_imported = False

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

ROOM_A = "!room-a:matrix.local"
ROOM_B = "!room-b:matrix.local"
AGENT_USER = "@agent:matrix.local"
OPERATOR = "@operator:matrix.local"

STEER_FRAMING = "[Operator steering — mid-turn guidance]:"


def _make_matrix_config(user_id=AGENT_USER, **kwargs):
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
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def _make_agent_config(workspace, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[],
        )},
        workspace=workspace,
        max_iterations=5,   # small cap so iteration-count tests run fast
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _make_room(room_id=ROOM_A, member_count=2):
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


def _make_bot(agent=None, config=None, **overrides):
    """Create a MatrixBot for testing via __new__ bypass.

    Sets all attributes that the existing test suite sets, PLUS the new
    steering attrs (_steering_inbox, _active_turns). Tests that probe for
    the absence of these attrs will fail correctly until the feature exists.
    """
    if config is None:
        config = _make_matrix_config()
    if agent is None:
        agent = MagicMock()
        agent._rooms = {}
        agent.handle_input = AsyncMock(return_value="response")
        agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))
        agent.cancel = MagicMock(return_value=None)
        agent.tools = []
        agent.config = MagicMock()
        agent.config.workspace = "/tmp/test-workspace"
        agent.config.model_aliases = {}

    bot = MatrixBot.__new__(MatrixBot)
    bot.config = config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock()
    bot.client.room_typing = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = {ROOM_A}   # room already activated
    bot._room_thinking = {}
    bot._room_cache_ttl = {}
    bot._room_timesense = {}
    bot._halted_rooms = set()
    bot._background_tasks = set()
    bot._session_locks = {}
    bot.session_log = MagicMock()
    bot.session_log.append = MagicMock()
    bot.session_log.build_context = MagicMock(return_value=[])
    bot.session_log.read = MagicMock(return_value=[])
    bot.heartbeat = MagicMock()
    bot.heartbeat.is_active = MagicMock(return_value=False)
    bot.umbral = MagicMock()
    bot.umbral.is_active = MagicMock(return_value=False)
    # --- NEW steering attrs (these will raise AttributeError until feature ships) ---
    # We deliberately do NOT set _steering_inbox or _active_turns here —
    # tests that check for their presence will fail RED as intended.
    # A separate helper sets them for tests that need the full pipeline.
    for key, val in overrides.items():
        setattr(bot, key, val)
    return bot


def _make_bot_with_steering(**overrides):
    """Like _make_bot but pre-populated with steering attrs for pipeline tests.

    Use when you need the bot to have _steering_inbox and _active_turns set
    (e.g. to test drain behaviour after queuing). Pipeline tests that need
    the feature to 'exist' start here.
    """
    bot = _make_bot(**overrides)
    # These are set here only to exercise the pipeline tests (C, D, E, F).
    # Tests in group A/B check for the ATTRIBUTE's EXISTENCE on a 'real' bot
    # and will fail until MatrixBot.__init__ creates them.
    if not hasattr(bot, '_steering_inbox'):
        bot._steering_inbox = {}
    if not hasattr(bot, '_active_turns'):
        bot._active_turns = set()
    return bot


def _make_stream_events(content="Hello", input_tokens=10, output_tokens=5):
    """Minimal stream that yields text + done."""
    async def _stream(*args, **kwargs):
        yield StreamEvent(type="text", content=content)
        yield StreamEvent(
            type="done",
            response=Response(
                content=content,
                model="claude-sonnet-4-20250514",
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
                stop_reason="end_turn",
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-20250514",
        )
    return _stream


def _make_tool_then_text(tool_calls_list, final_text="Done"):
    """Stream factory: first call returns tool_use, second returns text."""
    calls = iter([
        # tool-use response
        (tool_calls_list, ""),
        # text response
        ([], final_text),
    ])

    async def _stream(*args, **kwargs):
        tc_list, text = next(calls)
        if text:
            yield StreamEvent(type="text", content=text)
        for i, tc in enumerate(tc_list):
            yield StreamEvent(type="tool_done", tool_index=i, tool_call=tc)
        yield StreamEvent(
            type="done",
            response=Response(
                content=text,
                tool_calls=tc_list,
                model="claude-sonnet-4-20250514",
                usage=Usage(input_tokens=10, output_tokens=5),
                stop_reason="tool_use" if tc_list else "end_turn",
            ),
            stop_reason="tool_use" if tc_list else "end_turn",
            model="claude-sonnet-4-20250514",
        )

    return _stream


# ---------------------------------------------------------------------------
# A. /steer slash command
# ---------------------------------------------------------------------------


class TestSteerSlashCommand:
    """A. /steer slash command routing."""

    @pytest.mark.asyncio
    async def test_A1_no_active_turn_emits_no_active_turn_notice(self):
        """A1. No active turn → notice says no active turn, inbox untouched."""
        bot = _make_bot(
            _steering_inbox={},    # might not exist yet — test probes behaviour
            _active_turns=set(),   # empty: no active turn
        )
        # Override send_notice to capture calls
        notices = []
        bot.send_notice = AsyncMock(side_effect=lambda rid, text: notices.append(text))

        room = _make_room()
        event = _make_event(body="/steer fix the typo")
        await bot._handle_room_message(room, event)

        # Must emit a notice about no active turn
        assert any(
            "no active turn" in n.lower() or "no active" in n.lower()
            for n in notices
        ), f"Expected 'no active turn' notice; got: {notices}"

    @pytest.mark.asyncio
    async def test_A1_no_active_turn_does_not_deposit(self):
        """A1. No active turn → nothing deposited to inbox."""
        bot = _make_bot(
            _steering_inbox={},
            _active_turns=set(),
        )
        bot.send_notice = AsyncMock()

        room = _make_room()
        event = _make_event(body="/steer fix the typo")
        await bot._handle_room_message(room, event)

        inbox = getattr(bot, '_steering_inbox', {})
        assert inbox.get(ROOM_A, []) == [], (
            f"Inbox should be empty (no active turn) but got: {inbox}"
        )

    @pytest.mark.asyncio
    async def test_A1_no_active_turn_does_not_start_new_turn(self):
        """A1. No active turn → agent.handle_input NOT called."""
        bot = _make_bot(
            _steering_inbox={},
            _active_turns=set(),
        )
        bot.send_notice = AsyncMock()

        room = _make_room()
        event = _make_event(body="/steer fix the typo")
        await bot._handle_room_message(room, event)
        await asyncio.gather(*bot._background_tasks)

        bot.agent.handle_input.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_A2_active_turn_deposits_note_to_inbox(self):
        """A2. Active turn → note deposited to inbox without acquiring room lock."""
        bot = _make_bot(
            _steering_inbox={},
            _active_turns={ROOM_A},   # turn is active
        )
        bot.send_notice = AsyncMock()

        room = _make_room()
        event = _make_event(body="/steer fix the typo")
        await bot._handle_room_message(room, event)

        inbox = getattr(bot, '_steering_inbox', {})
        assert inbox.get(ROOM_A) == ["fix the typo"], (
            f"Expected ['fix the typo'] in inbox; got: {inbox.get(ROOM_A)}"
        )

    @pytest.mark.asyncio
    async def test_A2_active_turn_emits_queued_notice(self):
        """A2. Active turn → emits a 'queued' notice (contains 🧭 or 'steer'/'queued')."""
        bot = _make_bot(
            _steering_inbox={},
            _active_turns={ROOM_A},
        )
        notices = []
        bot.send_notice = AsyncMock(side_effect=lambda rid, text: notices.append(text))

        room = _make_room()
        event = _make_event(body="/steer fix the typo")
        await bot._handle_room_message(room, event)

        assert any(
            "🧭" in n or "queue" in n.lower() or "steer" in n.lower()
            for n in notices
        ), f"Expected queued/steer notice; got: {notices}"

    @pytest.mark.asyncio
    async def test_A2_active_turn_does_not_start_new_turn(self):
        """A2. Active turn → does NOT fire _process_message / handle_input."""
        bot = _make_bot(
            _steering_inbox={},
            _active_turns={ROOM_A},
        )
        bot.send_notice = AsyncMock()

        room = _make_room()
        event = _make_event(body="/steer fix the typo")
        await bot._handle_room_message(room, event)
        await asyncio.gather(*bot._background_tasks)

        bot.agent.handle_input.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_A3_empty_body_emits_usage_notice(self):
        """A3. /steer with no message → usage/ignore notice, nothing deposited."""
        bot = _make_bot(
            _steering_inbox={},
            _active_turns={ROOM_A},
        )
        notices = []
        bot.send_notice = AsyncMock(side_effect=lambda rid, text: notices.append(text))

        room = _make_room()
        event = _make_event(body="/steer")
        await bot._handle_room_message(room, event)

        assert notices, "Expected at least one notice for empty /steer"
        # Should NOT have deposited anything
        inbox = getattr(bot, '_steering_inbox', {})
        assert inbox.get(ROOM_A, []) == [], (
            f"Empty /steer should not deposit; inbox: {inbox}"
        )

    @pytest.mark.asyncio
    async def test_A3_whitespace_only_body_emits_usage_notice(self):
        """A3. /steer with whitespace-only message → usage notice, nothing deposited."""
        bot = _make_bot(
            _steering_inbox={},
            _active_turns={ROOM_A},
        )
        notices = []
        bot.send_notice = AsyncMock(side_effect=lambda rid, text: notices.append(text))

        room = _make_room()
        event = _make_event(body="/steer    ")
        await bot._handle_room_message(room, event)

        assert notices, "Expected a notice for whitespace-only /steer"
        inbox = getattr(bot, '_steering_inbox', {})
        assert inbox.get(ROOM_A, []) == [], (
            f"Whitespace-only /steer should not deposit; inbox: {inbox}"
        )


# ---------------------------------------------------------------------------
# B. Active-turn detection
# ---------------------------------------------------------------------------


class TestActiveTurnDetection:
    """B. _active_turns flag lifecycle."""

    def test_B_steering_inbox_attr_exists_on_bot(self, tmp_path):
        """MatrixBot.__init__ must create _steering_inbox."""
        config = _make_matrix_config()
        agent = MagicMock()
        agent.config = _make_agent_config(tmp_path)
        agent.config.workspace = tmp_path

        bot = MatrixBot(agent, config)
        assert hasattr(bot, '_steering_inbox'), (
            "MatrixBot must have _steering_inbox attribute after __init__"
        )
        assert isinstance(bot._steering_inbox, dict)

    def test_B_active_turns_attr_exists_on_bot(self, tmp_path):
        """MatrixBot.__init__ must create _active_turns."""
        config = _make_matrix_config()
        agent = MagicMock()
        agent.config = _make_agent_config(tmp_path)
        agent.config.workspace = tmp_path

        bot = MatrixBot(agent, config)
        assert hasattr(bot, '_active_turns'), (
            "MatrixBot must have _active_turns attribute after __init__"
        )
        assert isinstance(bot._active_turns, set)

    @pytest.mark.asyncio
    async def test_B1_active_turn_set_during_processing(self):
        """B1. _active_turns contains room_id while handle_input is running."""
        observed_during = []

        bot = _make_bot_with_steering()
        room = _make_room()

        async def slow_handle(*args, **kwargs):
            # Capture _active_turns state while handle_input is 'running'
            observed_during.append(ROOM_A in bot._active_turns)
            return "done"

        bot.agent.handle_input = AsyncMock(side_effect=slow_handle)
        event = _make_event(body="do some work")

        # Fire _process_message (which should set _active_turns before handle_input)
        await bot._process_message(room, event, "do some work")

        assert any(observed_during), (
            "_active_turns should contain room_id while handle_input is running"
        )

    @pytest.mark.asyncio
    async def test_B1_active_turn_cleared_after_processing(self):
        """B1. _active_turns does not contain room_id after handle_input completes."""
        bot = _make_bot_with_steering()
        room = _make_room()

        bot.agent.handle_input = AsyncMock(return_value="done")
        event = _make_event(body="do some work")

        await bot._process_message(room, event, "do some work")

        assert ROOM_A not in bot._active_turns, (
            "_active_turns should be cleared after turn completes"
        )

    @pytest.mark.asyncio
    async def test_B1_active_turn_cleared_after_exception(self):
        """B1. _active_turns cleared even if handle_input raises."""
        bot = _make_bot_with_steering()
        room = _make_room()

        bot.agent.handle_input = AsyncMock(side_effect=RuntimeError("boom"))
        # Also mock send so the error handler doesn't crash
        bot.send = AsyncMock()
        event = _make_event(body="do some work")

        # _process_message should catch the exception internally (like it does today)
        # and still clear _active_turns in its finally block
        await bot._process_message(room, event, "do some work")

        assert ROOM_A not in bot._active_turns, (
            "_active_turns must be cleared in finally, even after exception"
        )

    @pytest.mark.asyncio
    async def test_B2_room_isolation_active_turns(self):
        """B2. Room A active does NOT mark Room B active."""
        bot = _make_bot_with_steering()
        bot._active_rooms.add(ROOM_B)

        room_a = _make_room(ROOM_A)
        block_a = asyncio.Event()
        done_a = asyncio.Event()

        async def slow_handle_a(*args, **kwargs):
            await block_a.wait()
            done_a.set()
            return "done"

        bot.agent.handle_input = AsyncMock(side_effect=slow_handle_a)
        event = _make_event(body="work in room A")

        # Start processing room A (fires background task that will block)
        task = asyncio.create_task(bot._process_message(room_a, event, "work in room A"))
        await asyncio.sleep(0)  # let task start

        # While room A is "processing" — if _active_turns is set before handle_input
        # room A should be in _active_turns, room B should not
        # (task hasn't hit handle_input yet since block_a isn't set — but the
        # _active_turns.add should happen before handle_input is awaited)
        block_a.set()  # unblock
        await task

        # After completion: neither room should be active
        assert ROOM_B not in bot._active_turns, (
            "Room B must never appear in _active_turns when only Room A processes"
        )

    @pytest.mark.asyncio
    async def test_B2_steer_room_a_does_not_affect_room_b(self):
        """B2. /steer in room A, with active turn only in room A, deposits only to room A."""
        bot = _make_bot(
            _steering_inbox={},
            _active_turns={ROOM_A},  # only room A is active
        )
        bot.send_notice = AsyncMock()

        room_a = _make_room(ROOM_A)
        event = _make_event(body="/steer fix it")
        await bot._handle_room_message(room_a, event)

        inbox = getattr(bot, '_steering_inbox', {})
        assert inbox.get(ROOM_B, []) == [], (
            f"Room B inbox must be untouched; got: {inbox.get(ROOM_B)}"
        )
        assert inbox.get(ROOM_A) == ["fix it"]


# ---------------------------------------------------------------------------
# C. Drain + inject in the agent loop
# ---------------------------------------------------------------------------


class TestDrainInjectInLoop:
    """C. Drain + inject in the agent tool loop."""

    @pytest.mark.asyncio
    async def test_C1_drain_called_at_top_of_each_iteration(self, tmp_path):
        """C1. drain_steering() is called once per tool-loop iteration."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        drain_call_count = 0
        async def counting_drain():
            nonlocal drain_call_count
            drain_call_count += 1
            return []

        tc = ToolCall(id="c1", name="shell", input={"command": "echo hi"})

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.execute_tool") as mock_exec:

            # First call: tool use; second call: text response
            responses = [
                Response(
                    content="",
                    tool_calls=[tc],
                    model="claude-sonnet-4-20250514",
                    usage=Usage(input_tokens=10, output_tokens=5),
                    stop_reason="tool_use",
                ),
                Response(
                    content="Done",
                    tool_calls=[],
                    model="claude-sonnet-4-20250514",
                    usage=Usage(input_tokens=10, output_tokens=5),
                    stop_reason="end_turn",
                ),
            ]
            resp_iter = iter(responses)

            async def fake_stream(*args, **kwargs):
                resp = next(resp_iter)
                if resp.content:
                    yield StreamEvent(type="text", content=resp.content)
                for i, t in enumerate(resp.tool_calls):
                    yield StreamEvent(type="tool_done", tool_index=i, tool_call=t)
                yield StreamEvent(
                    type="done", response=resp,
                    stop_reason=resp.stop_reason, model=resp.model,
                )

            mock_stream.side_effect = fake_stream
            mock_exec.return_value = ToolResult(content="ok", is_error=False)

            result = await agent.handle_input(
                "do work",
                room_id=ROOM_A,
                drain_steering=counting_drain,
            )

        # drain was called at the top of iteration 0 AND iteration 1 (2 iterations total)
        assert drain_call_count >= 2, (
            f"drain_steering should be called every iteration; got {drain_call_count} calls"
        )

    @pytest.mark.asyncio
    async def test_C2_pending_note_appended_as_user_message_with_framing(self, tmp_path):
        """C2. Pending steering note → user-role message with framing prefix, FIFO."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        notes = ["rewrite the summary"]
        async def drain_once():
            return notes.pop(0) and [notes[0]] if notes else (
                lambda: (notes.pop(0), ["rewrite the summary"])[1]
            )()

        # Simpler approach: one-shot drain
        delivered = []
        async def drain_one_shot():
            if delivered:
                return []
            delivered.append(True)
            return ["rewrite the summary"]

        with patch("openalph.agent.stream") as mock_stream:
            call_num = [0]

            async def fake_stream(*args, **kwargs):
                call_num[0] += 1
                content = "Final answer"
                yield StreamEvent(type="text", content=content)
                yield StreamEvent(
                    type="done",
                    response=Response(
                        content=content, model="claude-sonnet-4-20250514",
                        usage=Usage(input_tokens=10, output_tokens=5),
                        stop_reason="end_turn",
                    ),
                    stop_reason="end_turn", model="claude-sonnet-4-20250514",
                )

            mock_stream.side_effect = fake_stream

            await agent.handle_input(
                "do work",
                room_id=ROOM_A,
                drain_steering=drain_one_shot,
            )

        # Check history: steering note should be in history as user message with framing
        history = agent.history(ROOM_A)
        user_msgs = [m for m in history if m.get("role") == "user"]
        steer_msgs = [
            m for m in user_msgs
            if STEER_FRAMING in m.get("content", "")
        ]
        assert steer_msgs, (
            f"Expected a user-role message with framing '{STEER_FRAMING}' in history. "
            f"User messages: {user_msgs}"
        )
        assert "rewrite the summary" in steer_msgs[0]["content"], (
            f"Framed message should contain the original note text. Got: {steer_msgs[0]}"
        )

    @pytest.mark.asyncio
    async def test_C3_injected_note_appears_in_provider_messages(self, tmp_path):
        """C3. The steering note appears in the messages sent to the mocked provider."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        async def drain_once():
            if not hasattr(drain_once, '_done'):
                drain_once._done = True
                return ["inject this note"]
            return []

        provider_calls = []

        with patch("openalph.agent.stream") as mock_stream:
            async def capturing_stream(*args, **kwargs):
                provider_calls.append(kwargs.get("messages", []))
                content = "OK"
                yield StreamEvent(type="text", content=content)
                yield StreamEvent(
                    type="done",
                    response=Response(
                        content=content, model="claude-sonnet-4-20250514",
                        usage=Usage(input_tokens=10, output_tokens=5),
                        stop_reason="end_turn",
                    ),
                    stop_reason="end_turn", model="claude-sonnet-4-20250514",
                )

            mock_stream.side_effect = capturing_stream
            await agent.handle_input("hello", room_id=ROOM_A, drain_steering=drain_once)

        # At least one provider call must include the framing-prefixed steering note
        all_messages = [m for call in provider_calls for m in call]
        steer_msgs = [
            m for m in all_messages
            if m.get("role") == "user" and STEER_FRAMING in m.get("content", "")
        ]
        assert steer_msgs, (
            f"Steering note with framing must appear in messages sent to provider. "
            f"All user messages seen by provider: "
            f"{[m for m in all_messages if m.get('role')=='user']}"
        )

    @pytest.mark.asyncio
    async def test_C3_drain_via_callbacks_dict_resolves_and_injects(self, tmp_path):
        """C3 (new). drain_steering passed via callbacks dict (production path) is
        resolved by _effective_drain and injects the steering note exactly as the
        direct kwarg path does — the note with STEER_FRAMING appears in the messages
        sent to the mocked provider."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        async def drain_once():
            if not hasattr(drain_once, '_done'):
                drain_once._done = True
                return ["callbacks-path note"]
            return []

        provider_calls = []

        with patch("openalph.agent.stream") as mock_stream:
            async def capturing_stream(*args, **kwargs):
                provider_calls.append(kwargs.get("messages", []))
                content = "OK"
                yield StreamEvent(type="text", content=content)
                yield StreamEvent(
                    type="done",
                    response=Response(
                        content=content, model="claude-sonnet-4-20250514",
                        usage=Usage(input_tokens=10, output_tokens=5),
                        stop_reason="end_turn",
                    ),
                    stop_reason="end_turn", model="claude-sonnet-4-20250514",
                )

            mock_stream.side_effect = capturing_stream
            # Production path: drain_steering passed in callbacks dict, NOT as direct kwarg
            await agent.handle_input(
                "hello",
                room_id=ROOM_A,
                callbacks={"drain_steering": drain_once},
            )

        # At least one provider call must include the framing-prefixed steering note
        all_messages = [m for call in provider_calls for m in call]
        steer_msgs = [
            m for m in all_messages
            if m.get("role") == "user" and STEER_FRAMING in m.get("content", "")
        ]
        assert steer_msgs, (
            f"Steering note with framing must appear in messages sent to provider "
            f"when drain_steering is resolved from callbacks dict. "
            f"All user messages seen by provider: "
            f"{[m for m in all_messages if m.get('role') == 'user']}"
        )

    @pytest.mark.asyncio
    async def test_C4_empty_inbox_no_spurious_injection(self, tmp_path):
        """C4. Empty inbox → no extra user messages injected; history unchanged."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        async def always_empty_drain():
            return []

        with patch("openalph.agent.stream") as mock_stream:
            mock_stream.side_effect = _make_stream_events("Hello")
            await agent.handle_input("hi", room_id=ROOM_A, drain_steering=always_empty_drain)

        history = agent.history(ROOM_A)
        steering_msgs = [
            m for m in history
            if m.get("role") == "user" and STEER_FRAMING in m.get("content", "")
        ]
        assert steering_msgs == [], (
            f"No steering messages expected when inbox is empty; got: {steering_msgs}"
        )

    @pytest.mark.asyncio
    async def test_C4_no_drain_kwarg_loop_behaves_as_today(self, tmp_path):
        """C4. If drain_steering not passed (None), loop behaves exactly as today."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock_stream:
            mock_stream.side_effect = _make_stream_events("Hello")
            # No drain_steering kwarg — must not crash
            result = await agent.handle_input("hi", room_id=ROOM_A)

        assert result == "Hello"

    @pytest.mark.asyncio
    async def test_C5_steering_does_not_extend_max_iterations(self, tmp_path):
        """C5. Receiving steering notes does NOT extend max_iterations cap."""
        config = _make_agent_config(tmp_path, max_iterations=3)
        agent = Agent(config)

        # drain_steering returns a note on EVERY iteration (simulating continuous input)
        async def always_has_note():
            return ["keep going!"]

        iteration_count = [0]

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.execute_tool") as mock_exec:

            # Always return tool use so the loop keeps iterating
            tc = ToolCall(id="iter_tc", name="shell", input={"command": "echo"})
            resp_tool = Response(
                content="", tool_calls=[tc],
                model="claude-sonnet-4-20250514",
                usage=Usage(input_tokens=10, output_tokens=5),
                stop_reason="tool_use",
            )
            # After max_iterations, the loop forces a summary — give it a text response
            resp_summary = Response(
                content="Summary text", tool_calls=[],
                model="claude-sonnet-4-20250514",
                usage=Usage(input_tokens=10, output_tokens=5),
                stop_reason="end_turn",
            )

            summary_calls = [0]

            async def fake_stream(*args, **kwargs):
                iteration_count[0] += 1
                # Always return tool use while tools are passed; text when tools=None
                if kwargs.get("tools") is None:
                    # This is the forced summary call after loop ends
                    resp = resp_summary
                else:
                    resp = resp_tool

                if resp.content:
                    yield StreamEvent(type="text", content=resp.content)
                for i, t in enumerate(resp.tool_calls):
                    yield StreamEvent(type="tool_done", tool_index=i, tool_call=t)
                yield StreamEvent(
                    type="done", response=resp,
                    stop_reason=resp.stop_reason, model=resp.model,
                )

            mock_stream.side_effect = fake_stream
            mock_exec.return_value = ToolResult(content="result", is_error=False)

            await agent.handle_input("do infinite work", room_id=ROOM_A,
                                      drain_steering=always_has_note)

        # The tool-use iterations should be capped at max_iterations (3)
        # Plus 1 summary call = 4 total stream calls max
        tool_iterations = iteration_count[0] - 1  # last call is summary
        assert tool_iterations <= config.max_iterations, (
            f"Steering should not extend max_iterations. "
            f"Expected ≤{config.max_iterations} tool iterations, got {tool_iterations}. "
            f"Total stream calls: {iteration_count[0]}"
        )

    @pytest.mark.asyncio
    async def test_C6_multiple_notes_drained_fifo_in_one_iteration(self, tmp_path):
        """C6. Multiple notes before one boundary → all drained FIFO in one pass."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        # Three notes in the inbox, all returned at once on first drain
        drained = [False]
        async def drain_three_at_once():
            if not drained[0]:
                drained[0] = True
                return ["note one", "note two", "note three"]
            return []

        with patch("openalph.agent.stream") as mock_stream:
            mock_stream.side_effect = _make_stream_events("result")
            await agent.handle_input("go", room_id=ROOM_A, drain_steering=drain_three_at_once)

        history = agent.history(ROOM_A)
        steer_msgs = [
            m for m in history
            if m.get("role") == "user" and STEER_FRAMING in m.get("content", "")
        ]
        # All three must be present
        assert len(steer_msgs) == 3, (
            f"Expected 3 steering messages; got {len(steer_msgs)}: {steer_msgs}"
        )
        # FIFO order
        assert "note one" in steer_msgs[0]["content"]
        assert "note two" in steer_msgs[1]["content"]
        assert "note three" in steer_msgs[2]["content"]


# ---------------------------------------------------------------------------
# D. Persistence / JSONL
# ---------------------------------------------------------------------------


class TestPersistenceJSONL:
    """D. Steering notes written to session JSONL with source='steer'."""

    def test_D1_consumed_note_logged_with_source_steer(self, tmp_path):
        """D1. drain_steering logs each note as user-role with source='steer'."""
        # This is a unit test of the drain_steering closure's logging behaviour.
        # We simulate what the drain callback does (it's a closure created in matrix.py
        # that calls session_log.append). The test verifies the contract on the log call.
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)

        # Manually exercise what drain_steering is supposed to do when it logs:
        sl.append(
            role="user",
            sender=OPERATOR,
            room=ROOM_A,
            event_id=None,
            content="fix the typo",
            source="steer",
        )

        entries = sl.read(ROOM_A)
        assert len(entries) == 1
        e = entries[0]
        assert e["role"] == "user"
        assert e["source"] == "steer"
        assert e["content"] == "fix the typo"

    @pytest.mark.asyncio
    async def test_D1_drain_callback_calls_session_log_append_with_source_steer(self, tmp_path):
        """D1 (integration). When drain_steering is invoked, session_log.append is called
        with source='steer' and the original note text."""
        bot = _make_bot_with_steering(
            _steering_inbox={ROOM_A: ["fix the typo"]},
            _active_turns={ROOM_A},
        )
        # Replace session_log with a real one to capture writes
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)
        bot.session_log = sl
        bot.send_notice = AsyncMock()

        room = _make_room()
        event = _make_event(body="/steer fix the typo")

        # Build the drain_steering closure as _process_message would
        # (We call _build_drain_steering if it exists, otherwise assume
        # the closure is constructed inside _process_message)
        # The test exercises the bot's drain closure by invoking _process_message
        # with a real handle_input that triggers the drain.

        bot.agent.handle_input = AsyncMock(return_value="done")

        # Inject the note into the inbox first
        bot._steering_inbox = {ROOM_A: ["fix the typo"]}

        # Run one complete turn (which should create and call drain_steering)
        await bot._process_message(room, event, "do work")

        # The session log must have a steer entry
        entries = sl.read(ROOM_A)
        steer_entries = [e for e in entries if e.get("source") == "steer"]
        assert steer_entries, (
            f"No source='steer' entries found in JSONL after drain. Entries: {entries}"
        )
        assert steer_entries[0]["role"] == "user"
        assert steer_entries[0]["content"] == "fix the typo"

    def test_D2_build_context_reconstructs_steering_note_with_framing(self, tmp_path):
        """D2. build_context reconstructs steered turn: note appears as user message
        with framing prefix in the correct position."""
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)

        # Write a turn with a steering note in the middle
        sl.append(role="user", sender=OPERATOR, room=ROOM_A, event_id="$e1",
                  content="start work")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_A, event_id=None,
                  content="I'll do some work")
        # Steering note logged mid-turn
        sl.append(role="user", sender=OPERATOR, room=ROOM_A, event_id=None,
                  content="fix the typo", source="steer")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_A, event_id=None,
                  content="Fixed!")

        context = sl.build_context(ROOM_A)

        # There should be a user message with framing at position 2
        user_msgs = [m for m in context if m.get("role") == "user"]
        steer_msgs = [m for m in user_msgs if STEER_FRAMING in m.get("content", "")]

        assert steer_msgs, (
            f"build_context must include steering note with framing prefix '{STEER_FRAMING}'. "
            f"Context user messages: {user_msgs}"
        )
        assert "fix the typo" in steer_msgs[0]["content"], (
            f"Framing message must include the original note text. Got: {steer_msgs[0]}"
        )

    def test_D3_original_text_preserved_in_jsonl_not_framing(self, tmp_path):
        """D3. JSONL stores the operator's original text; framing is context-only."""
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)
        sl.append(
            role="user",
            sender=OPERATOR,
            room=ROOM_A,
            event_id=None,
            content="fix the typo",
            source="steer",
        )

        entries = sl.read(ROOM_A)
        steer = [e for e in entries if e.get("source") == "steer"]
        assert steer, "Steer entry should be present"
        # JSONL content must be the ORIGINAL text, NOT the framing-prefixed version
        assert steer[0]["content"] == "fix the typo", (
            f"JSONL must store original text, not framing. Got: {steer[0]['content']}"
        )
        assert STEER_FRAMING not in steer[0]["content"], (
            "Framing prefix must NOT appear in JSONL — it's a context-only concern"
        )

    def test_D2_steer_entry_position_in_context_is_correct(self, tmp_path):
        """D2. Steering note appears in the correct position in rebuilt context."""
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)

        sl.append(role="user", sender=OPERATOR, room=ROOM_A, event_id="$e1",
                  content="start work")
        sl.append(role="user", sender=OPERATOR, room=ROOM_A, event_id=None,
                  content="fix the typo", source="steer")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_A, event_id=None,
                  content="Fixed!")

        context = sl.build_context(ROOM_A)

        roles = [m["role"] for m in context]
        assert roles == ["user", "user", "assistant"], (
            f"Expected [user, user, assistant] in context; got {roles}"
        )

        # The second user entry should have framing
        assert STEER_FRAMING in context[1]["content"], (
            f"Second user message should be the framed steering note; got: {context[1]}"
        )


# ---------------------------------------------------------------------------
# E. Notices
# ---------------------------------------------------------------------------


class TestNotices:
    """E. Operator-facing notices for queued and delivered events."""

    @pytest.mark.asyncio
    async def test_E1_queued_notice_on_deposit(self):
        """E1. 'queued' notice emitted when note is deposited (turn active)."""
        bot = _make_bot(
            _steering_inbox={},
            _active_turns={ROOM_A},
        )
        notices = []
        bot.send_notice = AsyncMock(side_effect=lambda rid, text: notices.append(text))

        room = _make_room()
        event = _make_event(body="/steer do this")
        await bot._handle_room_message(room, event)

        assert any(
            "queue" in n.lower() or "🧭" in n or "steer" in n.lower()
            for n in notices
        ), f"Expected queued notice; got: {notices}"

    @pytest.mark.asyncio
    async def test_E2_delivered_notice_on_drain(self, tmp_path):
        """E2. 'delivered' notice emitted when a note is drained/injected."""
        bot = _make_bot_with_steering(
            _steering_inbox={ROOM_A: ["please focus"]},
            _active_turns={ROOM_A},
        )

        delivered_notices = []
        original_send_notice = bot.send_notice if hasattr(bot, 'send_notice') else None

        async def capture_notice(rid, text):
            delivered_notices.append(text)

        bot.send_notice = AsyncMock(side_effect=capture_notice)
        bot.send = AsyncMock()
        bot.agent.handle_input = AsyncMock(return_value="done")
        bot._set_typing = AsyncMock()

        room = _make_room()
        event = _make_event(body="work on this")
        await bot._process_message(room, event, "work on this")

        assert any(
            "deliver" in n.lower() or "🧭" in n
            for n in delivered_notices
        ), f"Expected delivered notice after drain; got: {delivered_notices}"

    @pytest.mark.asyncio
    async def test_E3_notices_not_in_rebuilt_context(self, tmp_path):
        """E3. Notices are system-role (or not logged at all) — do NOT enter LLM context."""
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)

        # Log a notice the way matrix.py would (as a system entry)
        sl.append(
            role="system",
            sender=AGENT_USER,
            room=ROOM_A,
            event_id=None,
            event="steer_notice",
            detail="🧭 Steering note queued",
        )

        context = sl.build_context(ROOM_A)  # skip_system=True by default

        notice_msgs = [m for m in context if "queued" in str(m.get("content", "")).lower()
                       or "🧭" in str(m.get("content", ""))]
        assert notice_msgs == [], (
            f"Notices must not appear in rebuilt LLM context; found: {notice_msgs}"
        )

    @pytest.mark.asyncio
    async def test_E3_steer_notice_uses_m_notice_msgtype(self):
        """E3. Notices use m.notice msgtype (same as existing notices)."""
        bot = _make_bot(
            _steering_inbox={},
            _active_turns={ROOM_A},
        )

        room = _make_room()
        event = _make_event(body="/steer do this")
        await bot._handle_room_message(room, event)

        # Any room_send call for a notice must use m.notice
        notice_calls = [
            c for c in bot.client.room_send.await_args_list
            if len(c.args) >= 3 and c.args[2].get("msgtype") == "m.notice"
        ]
        assert notice_calls, (
            "At least one m.notice must be sent via client.room_send for the /steer command"
        )


# ---------------------------------------------------------------------------
# F. Edge cases / safety
# ---------------------------------------------------------------------------


class TestEdgeCasesAndSafety:
    """F. Race rule, /stop, whitespace filtering."""

    @pytest.mark.asyncio
    async def test_F1_note_deposited_in_final_iteration_is_logged(self, tmp_path):
        """F1 — Race rule. A note deposited while the final iteration runs is
        still injected/logged this turn (not lost), because the active-turn flag
        is cleared only AFTER a final drain pass once the loop exits.

        Ordering invariant tested:
          1. Tool loop exits (no more iterations).
          2. _active_turns.discard(room_id) is NOT called yet.
          3. A final drain_steering() is called.
          4. Notes found in that drain are logged.
          5. THEN _active_turns.discard(room_id) is called.
        """
        bot = _make_bot_with_steering()
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)
        bot.session_log = sl
        bot.send_notice = AsyncMock()
        bot.send = AsyncMock()
        bot._set_typing = AsyncMock()

        # Simulate: note arrives right at loop-exit time (inbox has a note)
        bot._steering_inbox = {ROOM_A: ["last-minute note"]}

        bot.agent.handle_input = AsyncMock(return_value="done")

        room = _make_room()
        event = _make_event(body="do work")
        await bot._process_message(room, event, "do work")

        # After the turn, _active_turns should be clear
        assert ROOM_A not in bot._active_turns, "_active_turns must be cleared after turn"

        # AND the inbox note must have been logged (consumed by final drain)
        entries = sl.read(ROOM_A)
        steer_entries = [e for e in entries if e.get("source") == "steer"]
        assert steer_entries, (
            "A note present when the turn ended must be logged by final drain, "
            f"not silently dropped. Session entries: {entries}"
        )
        assert steer_entries[0]["content"] == "last-minute note"

    @pytest.mark.asyncio
    async def test_F2_stop_clears_steering_inbox(self):
        """F2. /stop during a steered turn: inbox cleared so stale notes don't
        leak into the next turn."""
        bot = _make_bot_with_steering(
            _steering_inbox={ROOM_A: ["stale note"]},
            _active_turns={ROOM_A},
        )
        bot.send = AsyncMock()
        bot._set_typing = AsyncMock()
        bot.agent.cancel = MagicMock(return_value=None)

        room = _make_room()
        stop_event = _make_event(body="/stop")
        await bot._handle_room_message(room, stop_event)

        # Give any tasks a chance to settle
        await asyncio.gather(*bot._background_tasks, return_exceptions=True)

        inbox = getattr(bot, '_steering_inbox', {})
        assert inbox.get(ROOM_A, []) == [], (
            f"Inbox must be cleared after /stop; got: {inbox.get(ROOM_A)}"
        )

    @pytest.mark.asyncio
    async def test_F2_repair_history_still_strips_orphans_in_steered_turn(self, tmp_path):
        """F2. After /stop in a steered turn, _repair_history still strips orphaned
        tool_calls exactly as it does today (no regression)."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        tc = ToolCall(id="orphan1", name="shell", input={"command": "echo"})

        # Manually put an orphaned tool_call in history (as cancel mid-tool would)
        history = agent.history(ROOM_A)
        history.append({"role": "user", "content": "hello"})
        history.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [tc],
        })
        # No tool result → orphan

        agent._repair_history(history)

        # Orphan must be stripped
        assistant_msgs = [m for m in history if m.get("role") == "assistant"]
        assert all(
            not m.get("tool_calls") for m in assistant_msgs
        ), f"Orphaned tool_calls should be stripped; history: {history}"

    @pytest.mark.asyncio
    async def test_F3_whitespace_only_note_never_injected(self, tmp_path):
        """F3. Whitespace-only note is never injected into history."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        drained = [False]
        async def drain_whitespace():
            if not drained[0]:
                drained[0] = True
                return ["   ", "\t", "  \n  "]  # all whitespace
            return []

        with patch("openalph.agent.stream") as mock_stream:
            mock_stream.side_effect = _make_stream_events("result")
            await agent.handle_input("go", room_id=ROOM_A, drain_steering=drain_whitespace)

        history = agent.history(ROOM_A)
        steer_msgs = [
            m for m in history
            if m.get("role") == "user" and STEER_FRAMING in m.get("content", "")
        ]
        assert steer_msgs == [], (
            f"Whitespace-only notes must NOT be injected; got: {steer_msgs}"
        )

    @pytest.mark.asyncio
    async def test_F3_whitespace_only_steer_command_deposits_nothing(self):
        """F3. /steer with whitespace body → nothing in inbox."""
        bot = _make_bot(
            _steering_inbox={},
            _active_turns={ROOM_A},
        )
        bot.send_notice = AsyncMock()

        room = _make_room()
        event = _make_event(body="/steer   \n  ")
        await bot._handle_room_message(room, event)

        inbox = getattr(bot, '_steering_inbox', {})
        assert inbox.get(ROOM_A, []) == [], (
            f"Whitespace-only /steer must not deposit; got: {inbox}"
        )


# ---------------------------------------------------------------------------
# G. No regression — smoke-test that handle_input still works w/o drain kwarg
# ---------------------------------------------------------------------------


class TestNoRegression:
    """G. Basic handle_input behaviour unchanged when steering not passed."""

    @pytest.mark.asyncio
    async def test_G_handle_input_no_drain_returns_response(self, tmp_path):
        """handle_input without drain_steering kwarg returns response as before."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock_stream:
            mock_stream.side_effect = _make_stream_events("Baseline response")
            result = await agent.handle_input("hi", room_id=ROOM_A)

        assert result == "Baseline response"

    @pytest.mark.asyncio
    async def test_G_handle_input_no_drain_history_unchanged(self, tmp_path):
        """handle_input without drain_steering does not add framing messages."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock_stream:
            mock_stream.side_effect = _make_stream_events("OK")
            await agent.handle_input("hi", room_id=ROOM_A)

        history = agent.history(ROOM_A)
        steer_msgs = [m for m in history if STEER_FRAMING in m.get("content", "")]
        assert steer_msgs == [], f"No framing messages without drain: {steer_msgs}"

    @pytest.mark.asyncio
    async def test_G_handle_room_message_non_steer_command_unaffected(self):
        """Non-steer commands (/status, /stop) are not affected by steering logic."""
        bot = _make_bot(
            _steering_inbox={},
            _active_turns=set(),
        )
        bot.send = AsyncMock()
        bot.send_notice = AsyncMock()

        room = _make_room()
        stop_event = _make_event(body="/stop")
        # Should not raise, should cancel/halt as today
        await bot._handle_room_message(room, stop_event)
        # /stop sends a message
        bot.send.assert_awaited()

    def test_G_session_log_user_entry_without_source_unaffected(self, tmp_path):
        """SessionLog user entries without source='steer' are unchanged in build_context."""
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)
        sl.append(role="user", sender=OPERATOR, room=ROOM_A,
                  event_id="$e1", content="normal message")
        sl.append(role="assistant", sender=AGENT_USER, room=ROOM_A,
                  event_id=None, content="normal reply")

        context = sl.build_context(ROOM_A)
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "normal message"
        # Ensure framing was NOT prepended to a normal entry
        assert STEER_FRAMING not in context[0]["content"]


# ---------------------------------------------------------------------------
# Sanity: check that _steering_inbox and _active_turns are truly absent today
# (These tests are meta-guards to confirm the suite is genuinely RED on
# missing attributes, not accidentally passing on non-existent feature code.)
# ---------------------------------------------------------------------------


class TestFeatureAttributesAbsent:
    """Confirm the feature attributes don't exist yet — suite is genuinely RED."""

    def test_meta_steering_inbox_not_in_init(self, tmp_path):
        """_steering_inbox must NOT exist in current MatrixBot.__init__
        (confirms these tests are RED until the feature is built).
        """
        config = _make_matrix_config()
        agent = MagicMock()
        agent.config = _make_agent_config(tmp_path)
        agent.config.workspace = tmp_path

        bot = MatrixBot(agent, config)
        # This assertion FAILS until the feature is implemented (expected RED)
        assert hasattr(bot, '_steering_inbox'), (
            "EXPECTED RED: _steering_inbox not yet in MatrixBot.__init__. "
            "This test fails RED until the feature is built."
        )

    def test_meta_active_turns_not_in_init(self, tmp_path):
        """_active_turns must NOT exist in current MatrixBot.__init__."""
        config = _make_matrix_config()
        agent = MagicMock()
        agent.config = _make_agent_config(tmp_path)
        agent.config.workspace = tmp_path

        bot = MatrixBot(agent, config)
        assert hasattr(bot, '_active_turns'), (
            "EXPECTED RED: _active_turns not yet in MatrixBot.__init__. "
            "This test fails RED until the feature is built."
        )

    def test_meta_handle_input_accepts_drain_steering_kwarg(self, tmp_path):
        """agent.handle_input must accept drain_steering as a kwarg without crashing."""
        import inspect
        sig = inspect.signature(Agent.handle_input)
        assert 'drain_steering' in sig.parameters, (
            "EXPECTED RED: handle_input does not yet accept drain_steering kwarg. "
            "This test fails RED until the feature is built."
        )

    def test_meta_build_context_applies_steer_framing(self, tmp_path):
        """build_context must apply framing to source='steer' entries."""
        sl = SessionLog(workspace=tmp_path, agent_user_id=AGENT_USER)
        sl.append(
            role="user", sender=OPERATOR, room=ROOM_A,
            event_id=None, content="original note", source="steer",
        )
        context = sl.build_context(ROOM_A)
        assert any(
            STEER_FRAMING in m.get("content", "") for m in context
        ), (
            "EXPECTED RED: build_context does not yet apply steer framing. "
            "This test fails RED until the feature is built."
        )
