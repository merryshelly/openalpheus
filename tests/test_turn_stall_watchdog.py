"""Tests for the per-turn stall watchdog (Bug B) — Round 2.

Bug (production room wedge, 2026-08-03 RCA):
    Provider clients use httpx.Timeout(600.0, connect=10.0) and the OpenAI SDK's
    transparent retry layer retries up to _MAX_SDK_RETRIES=20 times, honouring
    Retry-After up to 60s.  During a ~42-minute rate-limit storm the turn task
    sat inside that retry loop with NO socket held, NO room-visible progress and
    NO log above DEBUG -- while holding BOTH per-room locks
    (matrix._session_locks[room] and agent._room_locks[room]).  Follow-up
    messages queued silently and the room looked dead.  `except Exception` never
    fired because nothing raised.

Fix:
    A per-turn watchdog cancels the turn when no room-observable progress signal
    has fired for `[agent] turn_stall_timeout_seconds` (default 900, 0 disables),
    schedules a stall notice, and always re-raises so the existing `finally`
    blocks release the locks, clear typing and discard _active_turns.

Progress signals: on_text_delta / on_thinking_delta, the tool callbacks,
_drain_steering deliveries, and the explicit callbacks['turn_progress'] hook —
which is now emitted from REAL sub-run milestones inside run_subagent (each
provider response, each completed tool-call iteration), NOT from a blind
time-based pinger.

Round-2 additions (adversarial review remediation):
  F1  cancellation delivered exactly at the watchdog-disarm await must PROPAGATE
      (the Round-1 `contextlib.suppress(CancelledError)` swallowed it and went on
      to persist + send the response).
  F3  subagent liveness comes from real milestones; a sub parked in one silent
      provider call longer than the parent's stall timeout is SUPPOSED to be
      cancelled — that is the unwedge behaviour, not a bug.
  F4  the stall notice must not delay releasing the room locks.
  F5  non-finite turn_stall_timeout_seconds (nan / +inf / -inf) is rejected.
  F6  tests exercise a real room lock, assert cancellation actually propagated,
      and cover the disarm window.
"""

import asyncio
import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from openalph.config import AgentConfig, ConfigError, MatrixConfig, ProviderConfig, load_config
from openalph.matrix import MatrixBot

ROOM = "!stall:matrix.local"
AGENT_USER = "@watson:matrix.local"
USER = "@alice:matrix.local"


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_matrix_config(**kwargs):
    defaults = dict(
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
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def make_agent_config(**kwargs):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={
            "anthropic": ProviderConfig(
                key="anthropic", type="anthropic",
                api_key="sk-test", base_url=None, quirks=[],
            )
        },
        workspace=Path("/tmp/test"),
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_event(body="do the thing", event_id="$evt1", sender=USER):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = 1_000_000
    event.source = {"content": {"msgtype": "m.text", "body": body}}
    return event


def make_room(room_id=ROOM):
    room = MagicMock()
    room.room_id = room_id
    room.name = "Stall Room"
    room.display_name = "Stall Room"
    room.joined_count = 2          # ungated
    room.users = {USER: MagicMock(), AGENT_USER: MagicMock()}
    return room


def make_bot(stall_timeout=1, workspace=None):
    """MatrixBot via __new__ with an ACTIVE room (no lazy wake / gap-fill)."""
    agent = MagicMock()
    agent.handle_input = AsyncMock(return_value="done")
    agent.history = MagicMock(return_value=[])
    agent.config = make_agent_config(
        **({"workspace": workspace} if workspace is not None else {}))
    # The knob under test. Set explicitly (not a MagicMock attribute) so the
    # watchdog arming decision is a real number in every test.
    agent.config.turn_stall_timeout_seconds = stall_timeout
    agent.status = MagicMock(return_value={
        "context_pct": 1, "context_tokens": 10, "context_max": 200000,
    })
    agent.last_stop_reason = MagicMock(return_value="end_turn")
    agent.last_turn_usage = MagicMock(return_value={})
    agent._room_tool_counts = {}
    agent._advisor_uses = {}
    agent._read_registries = {}
    # F6: a REAL per-room lock dict, so a test can prove the agent-side lock a
    # wedged turn holds is actually released after the watchdog fires.
    agent._room_locks = {}

    bot = MatrixBot.__new__(MatrixBot)
    bot.config = make_matrix_config()
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$resp"))
    bot.client.room_typing = AsyncMock()
    bot.client.rooms = {}
    bot._set_typing = AsyncMock()
    bot.send = AsyncMock()
    bot.send_notice = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = {ROOM}
    bot._halted_rooms = set()
    bot._room_effort = {}
    bot._room_cache_ttl = {}
    bot._room_timesense = {}
    bot._background_tasks = set()
    bot._session_locks = {}
    bot._steering_inbox = {}
    bot._active_turns = set()
    bot._advisor_results = {}
    bot._subagent_results = {}
    bot._known_event_ids = {}
    bot.heartbeat = None
    bot.umbral = None
    bot.session_log = MagicMock()
    bot.session_log.append = MagicMock()
    bot.session_log.read = MagicMock(return_value=[])
    bot.session_log.build_context = MagicMock(return_value=[])
    bot.session_log.last_event_id = MagicMock(return_value=None)
    bot.session_log.usage_totals = MagicMock(return_value={})
    return bot, agent


async def drain_background(bot, timeout=5):
    """F4: the stall notice is sent from a separately-bounded background task so
    a Matrix outage cannot delay the lock release. Tests must therefore drain."""
    for _ in range(10):
        tasks = [t for t in getattr(bot, "_background_tasks", set()) if not t.done()]
        if not tasks:
            break
        await asyncio.wait(tasks, timeout=timeout)
    await asyncio.sleep(0)


def stall_notices(bot):
    """Every message sent to the room that looks like the stall notice."""
    out = []
    for c in bot.send.await_args_list:
        text = c.args[1] if len(c.args) > 1 else c.kwargs.get("body", "")
        if "stall" in str(text).lower():
            out.append(str(text))
    return out


def watchdog_tasks():
    return [
        t for t in asyncio.all_tasks()
        if not t.done() and "watchdog" in (t.get_name() or "")
    ]


# ── B1: a stalled turn is cancelled, announced, and unwedges the room ────────

class TestStalledTurnCancelled:

    @pytest.mark.asyncio
    async def test_stalled_turn_cancelled_with_notice_and_locks_released(self):
        """B1 — handle_input never returns and never signals progress.

        F6: the fake turn acquires a REAL agent-side per-room lock (exactly as
        Agent.handle_input does) so this proves the lock a wedged turn holds is
        genuinely released, and it requires CancelledError to PROPAGATE out of
        the turn task rather than merely checking that a notice was sent.
        """
        bot, agent = make_bot(stall_timeout=1)
        cancelled = asyncio.Event()
        agent._room_locks[ROOM] = asyncio.Lock()

        async def never_progresses(*args, **kwargs):
            # Mirror Agent.handle_input: hold the per-room lock for the turn.
            async with agent._room_locks[ROOM]:
                try:
                    await asyncio.sleep(999)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return "unreachable"

        agent.handle_input = AsyncMock(side_effect=never_progresses)

        task = asyncio.create_task(
            bot._process_message(make_room(), make_event(), "stall me")
        )
        # Watchdog polls on min(30, timeout/4); with timeout=1 it must fire well
        # inside 5s (>=4x margin, no 1s-exact race).
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

        assert task.cancelled(), (
            "the watchdog's CancelledError must propagate out of the turn task, "
            "not be swallowed by the handler"
        )
        assert cancelled.is_set(), "the turn's handle_input must have been cancelled"

        await drain_background(bot)
        notices = stall_notices(bot)
        assert notices, (
            f"a stall notice must be sent to the room; sent: "
            f"{[c.args for c in bot.send.await_args_list]}"
        )
        assert "re-send" in notices[0].lower(), (
            f"stall notice should tell the operator to re-send: {notices[0]!r}"
        )

        # Locks / turn bookkeeping must be clean so the room is usable again.
        assert ROOM not in bot._active_turns, "_active_turns must be discarded"
        lock = bot._session_locks.get(ROOM)
        assert lock is None or not lock.locked(), (
            "the per-room session lock must be released by the finally block"
        )
        assert not agent._room_locks[ROOM].locked(), (
            "the agent-side per-room lock the wedged turn held must be released"
        )

        # Prove it: a subsequent message processes normally.
        agent.handle_input = AsyncMock(return_value="second turn ok")
        await asyncio.wait_for(
            bot._process_message(make_room(), make_event(event_id="$evt2"), "again"),
            timeout=5,
        )
        assert agent.handle_input.await_count == 1, (
            "a follow-up message must process after the stalled turn was cancelled"
        )

    @pytest.mark.asyncio
    async def test_no_watchdog_task_leaks_after_stall(self):
        bot, agent = make_bot(stall_timeout=1)

        async def never_progresses(*args, **kwargs):
            await asyncio.sleep(999)

        agent.handle_input = AsyncMock(side_effect=never_progresses)
        task = asyncio.create_task(
            bot._process_message(make_room(), make_event(), "stall me")
        )
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            if not task.done():
                task.cancel()
        await asyncio.sleep(0.1)
        await drain_background(bot)
        assert watchdog_tasks() == [], (
            f"watchdog task leaked after the turn ended: {watchdog_tasks()}"
        )

    @pytest.mark.asyncio
    async def test_stall_notice_does_not_delay_lock_release(self):
        """F4 — a Matrix outage on the stall notice must NOT hold the room lock.

        `send` blocks for the whole test; the turn task must still finish
        (cancelled) with both locks released, because the notice is sent from a
        separately-bounded background task after the lock-release path.
        """
        bot, agent = make_bot(stall_timeout=1)
        agent._room_locks[ROOM] = asyncio.Lock()
        send_entered = asyncio.Event()
        release_send = asyncio.Event()

        async def wedged_send(room_id, text, *a, **kw):
            # Simulate the Matrix outage ONLY for the stall notice, so the
            # follow-up turn's ordinary response delivery still works — the
            # point is that the notice cannot hold the room hostage.
            if "stall" in str(text).lower():
                send_entered.set()
                await release_send.wait()

        bot.send = AsyncMock(side_effect=wedged_send)

        async def never_progresses(*args, **kwargs):
            async with agent._room_locks[ROOM]:
                await asyncio.sleep(999)

        agent.handle_input = AsyncMock(side_effect=never_progresses)

        task = asyncio.create_task(
            bot._process_message(make_room(), make_event(), "stall me")
        )
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

        # The turn is done and the locks are free even though `send` is wedged.
        lock = bot._session_locks.get(ROOM)
        assert lock is None or not lock.locked(), (
            "session lock must be released before/independently of the stall notice"
        )
        assert not agent._room_locks[ROOM].locked()
        assert ROOM not in bot._active_turns

        # A follow-up message must be processable while the notice is still stuck.
        agent.handle_input = AsyncMock(return_value="ok now")
        await asyncio.wait_for(
            bot._process_message(make_room(), make_event(event_id="$evt2"), "again"),
            timeout=5,
        )
        assert agent.handle_input.await_count == 1

        release_send.set()
        await drain_background(bot)
        assert send_entered.is_set(), "the stall notice must still be attempted"


# ── F1: cancellation delivered exactly at the watchdog-disarm await ──────────

class TestCancelAtDisarmWindow:
    """CRITICAL regression (Round-1 review): the disarm used
    `contextlib.suppress(asyncio.CancelledError)` around `await _watchdog`, which
    cannot tell the child watchdog's expected cancellation from a cancellation
    delivered to the TURN task at that same await. In the tight return/disarm
    window the latter was swallowed and the turn went on to persist and send the
    response with `current_task().cancelling() == 1`.
    """

    @pytest.mark.asyncio
    async def test_cancel_at_disarm_propagates_and_sends_nothing(self):
        bot, agent = make_bot(stall_timeout=300)   # armed, far from firing
        persisted = []
        bot._persist_assistant_turn = MagicMock(
            side_effect=lambda room_id, **kw: persisted.append(kw.get("content")))

        async def returns_then_cancelled(*args, **kwargs):
            # Schedule the cancel so it lands on the NEXT suspension point of
            # the turn task — which is the watchdog-disarm await in the tight
            # finally around handle_input.
            turn_task = asyncio.current_task()
            asyncio.get_running_loop().call_soon(turn_task.cancel)
            return "the answer the operator must NOT receive"

        agent.handle_input = AsyncMock(side_effect=returns_then_cancelled)

        task = asyncio.create_task(
            bot._process_message(make_room(), make_event(), "answer me")
        )
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

        assert task.cancelled(), (
            "cancellation delivered at the disarm await must propagate — the "
            "turn must NOT complete normally"
        )
        assert persisted == [], (
            f"a cancelled turn must not persist its response; persisted: {persisted}"
        )
        sent = [c.args for c in bot.send.await_args_list]
        assert not any("the answer the operator must NOT receive" in str(a) for a in sent), (
            f"a cancelled turn must not send its response; sent: {sent}"
        )
        # Not a watchdog fire -> no stall notice.
        await drain_background(bot)
        assert stall_notices(bot) == []
        # And the room is left clean.
        assert ROOM not in bot._active_turns
        lock = bot._session_locks.get(ROOM)
        assert lock is None or not lock.locked()

    @pytest.mark.asyncio
    async def test_watchdog_still_collected_when_cancel_races_disarm(self):
        """No leaked watchdog even when the parent is cancelled during the
        disarm collection."""
        bot, agent = make_bot(stall_timeout=300)

        async def returns_then_cancelled(*args, **kwargs):
            asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
            return "answer"

        agent.handle_input = AsyncMock(side_effect=returns_then_cancelled)
        task = asyncio.create_task(
            bot._process_message(make_room(), make_event(), "hi")
        )
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)
        assert watchdog_tasks() == [], (
            f"watchdog must not leak when cancel races disarm: {watchdog_tasks()}"
        )


# ── B2: a healthy streaming turn is never cancelled ──────────────────────────

class TestHealthyTurnSurvives:

    @pytest.mark.asyncio
    async def test_text_deltas_keep_turn_alive(self):
        """B2 — on_text_delta every 0.3s with timeout=2 must complete."""
        bot, agent = make_bot(stall_timeout=2)
        # Outcome recorded OUTSIDE the turn: _process_message wraps handle_input
        # in `except Exception`, so an assert raised inside the fake turn would
        # otherwise be swallowed and the test would pass vacuously.
        outcome = {"completed": False, "cancelled": False}

        async def streaming_turn(body, room_id, **kwargs):
            on_text_delta = kwargs.get("on_text_delta")
            try:
                # 10 * 0.3s = 3.0s total, which exceeds the 2s stall timeout, so
                # only a working progress reset keeps this turn alive.
                for _ in range(10):
                    await asyncio.sleep(0.3)
                    if on_text_delta:
                        await on_text_delta("tick ", False)
                if on_text_delta:
                    await on_text_delta("", True)
            except asyncio.CancelledError:
                outcome["cancelled"] = True
                raise
            outcome["completed"] = True
            return "healthy turn complete"

        agent.handle_input = AsyncMock(side_effect=streaming_turn)

        await asyncio.wait_for(
            bot._process_message(make_room(), make_event(), "stream to me"),
            timeout=15,
        )

        assert agent.handle_input.await_count == 1
        assert outcome["completed"] is True, "the healthy turn must run to completion"
        assert outcome["cancelled"] is False, "a streaming turn must never be cancelled"
        await drain_background(bot)
        assert stall_notices(bot) == [], (
            "a turn that streams text every 0.3s must NOT be declared stalled"
        )

    @pytest.mark.asyncio
    async def test_thinking_deltas_keep_turn_alive(self):
        bot, agent = make_bot(stall_timeout=2)
        outcome = {"completed": False, "cancelled": False}

        async def thinking_turn(body, room_id, **kwargs):
            on_thinking_delta = kwargs.get("on_thinking_delta")
            try:
                for _ in range(10):
                    await asyncio.sleep(0.3)
                    if on_thinking_delta:
                        await on_thinking_delta("pondering ", False)
            except asyncio.CancelledError:
                outcome["cancelled"] = True
                raise
            outcome["completed"] = True
            return "thought about it"

        agent.handle_input = AsyncMock(side_effect=thinking_turn)

        await asyncio.wait_for(
            bot._process_message(make_room(), make_event(), "think"),
            timeout=15,
        )
        assert outcome["completed"] is True
        assert outcome["cancelled"] is False
        await drain_background(bot)
        assert stall_notices(bot) == []


# ── B3: explicit turn_progress pings reset the window ────────────────────────

class TestTurnProgressCallback:

    @pytest.mark.asyncio
    async def test_turn_progress_callback_exposed(self):
        bot, agent = make_bot(stall_timeout=1)
        await bot._process_message(make_room(), make_event(), "hello")
        cb = agent.handle_input.await_args.kwargs["callbacks"]
        assert "turn_progress" in cb, (
            "callbacks must expose 'turn_progress' so tools can signal liveness"
        )
        assert callable(cb["turn_progress"])

    @pytest.mark.asyncio
    async def test_turn_progress_pings_prevent_cancellation(self):
        """B3 — a turn whose ONLY progress signal is callbacks['turn_progress']
        (the seam the subagent milestones use) must not be cancelled."""
        bot, agent = make_bot(stall_timeout=2)
        # Recorded outside the turn — see the note in B2 about `except Exception`
        # swallowing in-turn assertion errors.
        outcome = {"ping_available": False, "completed": False, "cancelled": False}

        async def pinging_turn(body, room_id, **kwargs):
            ping = (kwargs.get("callbacks") or {}).get("turn_progress")
            outcome["ping_available"] = callable(ping)
            try:
                for _ in range(10):
                    await asyncio.sleep(0.3)
                    if callable(ping):
                        res = ping()
                        if asyncio.iscoroutine(res):
                            await res
            except asyncio.CancelledError:
                outcome["cancelled"] = True
                raise
            outcome["completed"] = True
            return "kept alive by pings"

        agent.handle_input = AsyncMock(side_effect=pinging_turn)

        await asyncio.wait_for(
            bot._process_message(make_room(), make_event(), "ping me"),
            timeout=15,
        )
        assert agent.handle_input.await_count == 1
        assert outcome["ping_available"] is True, (
            "callbacks['turn_progress'] must be a callable the tool layer can ping"
        )
        assert outcome["cancelled"] is False, (
            "turn_progress pings must reset the stall window — the turn was cancelled"
        )
        assert outcome["completed"] is True
        await drain_background(bot)
        assert stall_notices(bot) == [], (
            "turn_progress pings must reset the stall window"
        )


# ── B4: the knob disables the watchdog ───────────────────────────────────────

class TestWatchdogDisabled:

    @pytest.mark.asyncio
    async def test_zero_timeout_creates_no_watchdog_task(self):
        """B4 — turn_stall_timeout_seconds=0 -> no watchdog task at all."""
        bot, agent = make_bot(stall_timeout=0)
        seen = []

        async def observe_tasks(*args, **kwargs):
            seen.extend(watchdog_tasks())
            return "no watchdog please"

        agent.handle_input = AsyncMock(side_effect=observe_tasks)

        await asyncio.wait_for(
            bot._process_message(make_room(), make_event(), "hi"), timeout=5
        )
        assert seen == [], f"watchdog must not be armed when disabled: {seen}"
        await drain_background(bot)
        assert stall_notices(bot) == []

    @pytest.mark.asyncio
    async def test_watchdog_armed_when_enabled(self):
        """Control for B4: with a positive timeout, a watchdog IS armed."""
        bot, agent = make_bot(stall_timeout=30)
        seen = []

        async def observe_tasks(*args, **kwargs):
            seen.extend(watchdog_tasks())
            return "ok"

        agent.handle_input = AsyncMock(side_effect=observe_tasks)

        await asyncio.wait_for(
            bot._process_message(make_room(), make_event(), "hi"), timeout=5
        )
        assert seen, "a positive turn_stall_timeout_seconds must arm a watchdog task"


# ── B5: operator /stop-style cancels behave exactly as before ────────────────

class TestOperatorCancelUnchanged:

    @pytest.mark.asyncio
    async def test_external_cancel_sends_no_stall_notice(self):
        """B5 — cancel not raised by the watchdog (no _stall_fired) must not
        produce a stall notice, and must still propagate CancelledError."""
        bot, agent = make_bot(stall_timeout=300)   # watchdog armed but far away
        started = asyncio.Event()

        async def blocking_turn(*args, **kwargs):
            started.set()
            await asyncio.sleep(999)

        agent.handle_input = AsyncMock(side_effect=blocking_turn)

        task = asyncio.create_task(
            bot._process_message(make_room(), make_event(), "long work")
        )
        await asyncio.wait_for(started.wait(), timeout=5)

        task.cancel()   # operator /stop path: _cancel_current -> task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await drain_background(bot)
        assert stall_notices(bot) == [], (
            f"an operator cancel must NOT emit a stall notice; sent: "
            f"{[c.args for c in bot.send.await_args_list]}"
        )
        # Prior behaviour: bookkeeping cleaned, locks released.
        assert ROOM not in bot._active_turns
        lock = bot._session_locks.get(ROOM)
        assert lock is None or not lock.locked()

    @pytest.mark.asyncio
    async def test_external_cancel_leaves_no_watchdog_task(self):
        bot, agent = make_bot(stall_timeout=300)
        started = asyncio.Event()

        async def blocking_turn(*args, **kwargs):
            started.set()
            await asyncio.sleep(999)

        agent.handle_input = AsyncMock(side_effect=blocking_turn)
        task = asyncio.create_task(
            bot._process_message(make_room(), make_event(), "long work")
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)
        await drain_background(bot)
        assert watchdog_tasks() == [], (
            f"watchdog must be cancelled in the turn's finally: {watchdog_tasks()}"
        )


# ── Config plumbing for the new knob ─────────────────────────────────────────

def _write_toml(tmp_path, value_line):
    (tmp_path / "agent.toml").write_text(f"""
[agent]
name = "test"
default_model = "anthropic/test-model"
{value_line}

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test-workspace"
""")
    return tmp_path / "agent.toml"


class TestStallTimeoutConfig:

    def test_agent_config_default_is_900(self):
        cfg = make_agent_config()
        assert cfg.turn_stall_timeout_seconds == 900

    def test_toml_parses_value(self, tmp_path):
        cfg = load_config(_write_toml(tmp_path, "turn_stall_timeout_seconds = 120"))
        assert cfg.turn_stall_timeout_seconds == 120

    def test_toml_omitted_defaults_to_900(self, tmp_path):
        cfg = load_config(_write_toml(tmp_path, ""))
        assert cfg.turn_stall_timeout_seconds == 900

    def test_toml_zero_disables(self, tmp_path):
        """0 is a legal value (disables the watchdog) — unlike max_iterations."""
        cfg = load_config(_write_toml(tmp_path, "turn_stall_timeout_seconds = 0"))
        assert cfg.turn_stall_timeout_seconds == 0

    def test_toml_rejects_negative(self, tmp_path):
        with pytest.raises(ConfigError, match="turn_stall_timeout_seconds"):
            load_config(_write_toml(tmp_path, "turn_stall_timeout_seconds = -5"))

    def test_toml_rejects_non_number(self, tmp_path):
        with pytest.raises(ConfigError, match="turn_stall_timeout_seconds"):
            load_config(_write_toml(tmp_path, 'turn_stall_timeout_seconds = "soon"'))

    # --- F5: non-finite values bypassed the promised validation --------------

    def test_toml_rejects_nan(self, tmp_path):
        """TOML `nan` is a float and `nan < 0` is False, so the naive check let
        it through and silently DISABLED the watchdog (`nan > 0` is also False)."""
        with pytest.raises(ConfigError, match="turn_stall_timeout_seconds"):
            load_config(_write_toml(tmp_path, "turn_stall_timeout_seconds = nan"))

    def test_toml_rejects_positive_inf(self, tmp_path):
        """`inf` arms a watchdog whose idle comparison can never fire."""
        with pytest.raises(ConfigError, match="turn_stall_timeout_seconds"):
            load_config(_write_toml(tmp_path, "turn_stall_timeout_seconds = inf"))

    def test_toml_rejects_negative_inf(self, tmp_path):
        with pytest.raises(ConfigError, match="turn_stall_timeout_seconds"):
            load_config(_write_toml(tmp_path, "turn_stall_timeout_seconds = -inf"))

    def test_finite_float_still_accepted(self, tmp_path):
        cfg = load_config(_write_toml(tmp_path, "turn_stall_timeout_seconds = 12.5"))
        assert cfg.turn_stall_timeout_seconds == 12.5
        assert math.isfinite(cfg.turn_stall_timeout_seconds)


# ── F3: subagent liveness comes from REAL sub-run milestones ─────────────────

def _sub_response(content="", tool_calls=None, stop_reason="end_turn"):
    from openalph.provider import Response, Usage
    return Response(
        content=content,
        tool_calls=tool_calls or [],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=10, output_tokens=5),
        stop_reason=stop_reason,
    )


class TestSubagentMilestoneProgress:
    """The Round-1 design pinged `turn_progress` every 60s merely because
    run_subagent had not returned. That is a BLIND heartbeat: a sub parked in
    the same provider retry storm resets the parent watchdog forever while both
    parent room locks stay held — i.e. it preserves the exact wedge the watchdog
    exists to break. Progress must come from REAL sub-run milestones, and a sub
    with no milestones for longer than the parent's stall timeout SHOULD be
    cancelled (same semantics as a stalled parent turn).
    """

    def test_no_time_based_pinger_remains(self):
        """The blind time-based pinger (and its cadence constant) must be gone."""
        import openalph.tools as tools_mod
        assert not hasattr(tools_mod, "_SUBAGENT_PROGRESS_PING_INTERVAL"), (
            "the blind time-based subagent pinger must be removed entirely — it "
            "masked the provider wedge the watchdog exists to catch"
        )

    @pytest.mark.asyncio
    async def test_milestones_emitted_from_sub_run(self, monkeypatch, tmp_path):
        """run_subagent fires callbacks['turn_progress'] on each provider
        response and each completed tool-call iteration."""
        from openalph.tools.subagent import run_subagent

        calls = {"n": 0}
        pings = []

        async def fake_complete(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                # stop_reason=max_tokens with no tool calls -> continuation loop,
                # i.e. another real provider round trip.
                return _sub_response(content="partial", stop_reason="max_tokens")
            return _sub_response(content="all done")

        monkeypatch.setattr(
            "openalph.tools.subagent.complete", fake_complete, raising=True)

        result = await run_subagent(
            task="do sub work",
            config=make_agent_config(workspace=tmp_path),
            callbacks={"turn_progress": lambda: pings.append(1)},
        )
        assert not result.is_error, result.content
        assert calls["n"] == 3
        assert len(pings) >= calls["n"], (
            f"every provider response is a milestone: {calls['n']} completions "
            f"produced only {len(pings)} progress signals"
        )

    @pytest.mark.asyncio
    async def test_milestone_emitted_per_tool_iteration(self, monkeypatch, tmp_path):
        """A completed tool-call iteration is also a real milestone."""
        from openalph.provider import ToolCall
        from openalph.tools.subagent import run_subagent

        calls = {"n": 0}
        pings = []

        async def fake_complete(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _sub_response(
                    content="calling a tool",
                    tool_calls=[ToolCall(id="tc1", name="shell",
                                         input={"command": "echo hi"})],
                    stop_reason="tool_use")
            return _sub_response(content="finished")

        async def fake_execute_tool(**kwargs):
            from openalph.tools import ToolResult
            return ToolResult(content="hi", is_error=False)

        monkeypatch.setattr(
            "openalph.tools.subagent.complete", fake_complete, raising=True)
        monkeypatch.setattr(
            "openalph.tools.execute_tool", fake_execute_tool, raising=True)

        result = await run_subagent(
            task="tool work",
            config=make_agent_config(workspace=tmp_path),
            callbacks={"turn_progress": lambda: pings.append(1)},
        )
        assert not result.is_error, result.content
        # 2 provider responses + 1 completed tool iteration
        assert len(pings) >= 3, (
            f"expected >=3 milestones (2 responses + 1 tool iteration); got {len(pings)}"
        )

    @pytest.mark.asyncio
    async def test_milestone_failure_never_breaks_sub_run(self, monkeypatch, tmp_path):
        """A raising turn_progress hook must not convert a good sub run into an
        error (fail-soft, mirroring the other out-of-band hooks)."""
        from openalph.tools.subagent import run_subagent

        async def fake_complete(**kwargs):
            return _sub_response(content="fine")

        monkeypatch.setattr(
            "openalph.tools.subagent.complete", fake_complete, raising=True)

        def boom():
            raise RuntimeError("hook exploded")

        result = await run_subagent(
            task="t", config=make_agent_config(workspace=tmp_path),
            callbacks={"turn_progress": boom},
        )
        assert not result.is_error
        assert "fine" in result.content

    @pytest.mark.asyncio
    async def test_missing_turn_progress_callback_is_harmless(self, monkeypatch, tmp_path):
        """Headless/CLI callers have no turn_progress; the sub run must work."""
        import openalph.tools as tools_mod
        from openalph.tools import ToolResult

        async def fast_run_subagent(**kwargs):
            return ToolResult(content="fine without pinger", is_error=False)

        monkeypatch.setattr(
            "openalph.tools.subagent.run_subagent", fast_run_subagent, raising=True
        )
        result = await tools_mod.execute_tool(
            name="subagent",
            input={"task": "no callbacks"},
            tool_config={},
            agent_config=make_agent_config(workspace=tmp_path),
            tools=None,
            callbacks={"room_id": ROOM},
        )
        assert not result.is_error
        assert "fine without pinger" in result.content

    @pytest.mark.asyncio
    async def test_subagent_cancel_at_return_does_not_continue_tool_loop(
            self, monkeypatch, tmp_path):
        """MAJOR #4 — the pinger's cleanup `await _pinger` was an AMBIGUOUS
        suspension point: cancellation delivered to the CALLER right at sub-run
        return was suppressed as if it had come from the deliberately cancelled
        pinger. Verified against the Round-1 pattern in isolation: the caller
        returned the sub result with `current_task().cancelling() == 1` and then
        ran ANOTHER model iteration — i.e. `/stop` or shutdown was acknowledged
        and ignored.

        Removing the pinger removes that suspension point, so the request is
        still pending when the caller's tool loop next suspends and cancellation
        strictly propagates. This test drives the caller shape that matters: a
        loop that continues after the tool returns.
        """
        import openalph.tools as tools_mod
        from openalph.tools import ToolResult

        async def run_then_cancel_caller(**kwargs):
            asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
            return ToolResult(content="sub done", is_error=False)

        monkeypatch.setattr(
            "openalph.tools.subagent.run_subagent", run_then_cancel_caller,
            raising=True)

        iterations = []

        async def fake_tool_loop():
            result = await tools_mod.execute_tool(
                name="subagent",
                input={"task": "t"},
                tool_config={},
                agent_config=make_agent_config(workspace=tmp_path),
                tools=None,
                callbacks={"turn_progress": lambda: None, "room_id": ROOM},
            )
            await asyncio.sleep(0)      # the next model iteration's first await
            iterations.append("continued past the cancel")
            return result

        task = asyncio.create_task(fake_tool_loop())
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled(), (
            "cancellation requested at sub-run return must propagate, not be "
            "swallowed as if it came from a deliberately cancelled helper task"
        )
        assert iterations == [], (
            "a cancelled tool loop must NOT continue into another model "
            f"iteration; it ran: {iterations}"
        )

    @pytest.mark.asyncio
    async def test_no_helper_task_suspension_at_sub_return(self, monkeypatch, tmp_path):
        """Corollary: with no pinger there is no helper task to collect, so the
        subagent tool leaks no task and adds no await after the sub returns."""
        import openalph.tools as tools_mod
        from openalph.tools import ToolResult

        async def fast(**kwargs):
            return ToolResult(content="quick", is_error=False)

        monkeypatch.setattr(
            "openalph.tools.subagent.run_subagent", fast, raising=True)

        await tools_mod.execute_tool(
            name="subagent",
            input={"task": "t"},
            tool_config={},
            agent_config=make_agent_config(workspace=tmp_path),
            tools=None,
            callbacks={"turn_progress": lambda: None, "room_id": ROOM},
        )
        await asyncio.sleep(0.1)
        leftover = [
            t for t in asyncio.all_tasks()
            if not t.done() and (
                "progress" in (t.get_name() or "") or "ping" in (t.get_name() or ""))
        ]
        assert leftover == [], f"subagent helper task leaked: {leftover}"


class TestSubagentWatchdogIntegration:
    """End-to-end: a watchdog-armed parent turn whose model calls the subagent
    tool, with the sub's REAL milestones as the only liveness signal."""

    def _wire_turn(self, bot, agent):
        """handle_input side effect that calls the subagent tool the way the real
        agent tool loop does — with the turn's own callbacks dict."""
        import openalph.tools as tools_mod

        async def turn(body, room_id, **kwargs):
            cbs = dict(kwargs.get("callbacks") or {})
            res = await tools_mod.execute_tool(
                name="subagent",
                input={"task": "long sub work"},
                tool_config={},
                agent_config=agent.config,
                tools=None,
                callbacks=cbs,
            )
            return res.content

        agent.handle_input = AsyncMock(side_effect=turn)

    @pytest.mark.asyncio
    async def test_sub_milestones_keep_parent_turn_alive(self, monkeypatch, tmp_path):
        bot, agent = make_bot(stall_timeout=1, workspace=tmp_path)
        self._wire_turn(bot, agent)

        calls = {"n": 0}

        async def fake_complete(**kwargs):
            # 6 * 0.3s = 1.8s of sub work, well past the 1s stall timeout: only
            # real per-response milestones can keep the parent alive.
            await asyncio.sleep(0.3)
            calls["n"] += 1
            if calls["n"] < 6:
                return _sub_response(content="partial", stop_reason="max_tokens")
            return _sub_response(content="sub finished cleanly")

        monkeypatch.setattr(
            "openalph.tools.subagent.complete", fake_complete, raising=True)

        await asyncio.wait_for(
            bot._process_message(make_room(), make_event(), "spawn a sub"),
            timeout=20,
        )
        await drain_background(bot)
        assert calls["n"] == 6, f"sub run did not complete: {calls['n']} calls"
        assert stall_notices(bot) == [], (
            "a sub emitting real milestones must keep the parent turn alive"
        )
        sent = " ".join(str(c.args) for c in bot.send.await_args_list)
        assert "sub finished cleanly" in sent

    @pytest.mark.asyncio
    async def test_silent_sub_is_cancelled_by_parent_watchdog(self, monkeypatch, tmp_path):
        """INTENDED unwedge behaviour: a sub parked in ONE silent provider call
        longer than the parent's stall timeout gets the whole turn cancelled."""
        bot, agent = make_bot(stall_timeout=1, workspace=tmp_path)
        self._wire_turn(bot, agent)
        sub_cancelled = asyncio.Event()

        async def wedged_complete(**kwargs):
            try:
                await asyncio.sleep(999)      # the provider retry storm
            except asyncio.CancelledError:
                sub_cancelled.set()
                raise

        monkeypatch.setattr(
            "openalph.tools.subagent.complete", wedged_complete, raising=True)

        task = asyncio.create_task(
            bot._process_message(make_room(), make_event(), "spawn a wedged sub")
        )
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=8)

        assert task.cancelled(), "the wedged turn must actually be cancelled"
        assert sub_cancelled.is_set(), (
            "cancellation must reach the sub's blocked provider call"
        )
        await drain_background(bot)
        assert stall_notices(bot), "the stall must be announced in the room"
        assert ROOM not in bot._active_turns
        lock = bot._session_locks.get(ROOM)
        assert lock is None or not lock.locked()
