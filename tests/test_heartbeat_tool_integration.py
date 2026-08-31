"""Real-path integration tests for the built-in `heartbeat` tool (kdsn.290).

Spec: memory/projects/openalph/specs/heartbeat-tool-spec.md §2/§5 (I01–I06).
Anchors: tmp/heartbeat-tool/anchors.md.

Unlike tests/test_heartbeat_tool.py (fake managers), these exercise the REAL
seams the spec wires:

  I01 — MatrixBot._build_agent_callbacks forwards a REAL HeartbeatManager so
        execute_tool("heartbeat", ...) mutates real heartbeats.json.
  I02 — build_callbacks returns dict keys "heartbeat"/"umbral" carrying the
        manager objects (anchors A1/A3 wiring decision).
  I03 — REAL UmbralManager active in the room excludes tool-start.
  I04 — self-stop pin: stop called via the tool FROM WITHIN the fired
        heartbeat turn completes without CancelledError and the manager
        stops refiring (hazard H2; nothing may wrap the turn in create_task).
  I05 — persistence parity: tool-start writes the slash-start JSON shape and
        a fresh manager resume()s it with the same interval + directive.
  I06 — headless: build_callbacks(..., HeadlessSinks(), turn_source=None,
        session_log=None) has the keys but both managers None → every action
        fails clean with steering.

RED suite: the dispatch branch and callbacks keys do not exist yet — every
I-test is expected to FAIL against current main. I02/I06 fail at meaningful
assertions (missing "heartbeat"/"umbral" keys), not on "Unknown tool".
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from openalph.agent import Agent
from openalph.callbacks import HeadlessSinks, build_callbacks
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.heartbeat import HeartbeatManager
from openalph.matrix import MatrixBot
from openalph.tools import execute_tool
from openalph.umbral import UmbralManager


ROOM_ID = "!hb-tool:matrix.local"

FLOOR_MSG = "Minimum interval is 5m."
UMBRAL_EXCLUSION = "Stop the umbral timer first"


# ---------------------------------------------------------------------------
# Helpers — mirror tests/test_guidance_integration.py:102
# (_make_bot_with_real_agent), with the MagicMock heartbeat/umbral managers
# REPLACED by real managers on tmp paths.
# ---------------------------------------------------------------------------


def _cfg(workspace, **kw):
    defaults = dict(
        name="hb-tool-integ",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={
            "anthropic": ProviderConfig(
                key="anthropic", type="anthropic", api_key="sk-test",
                base_url=None, quirks=[],
            )
        },
        workspace=workspace,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
        reminders=True,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _setup_workspace(tmp_path):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(exist_ok=True)
    for name in ("shell", "file_read", "file_write", "todo_write"):
        (tools_dir / f"{name}.toml").write_text("[config]\n")
    return tmp_path


def _make_bot_with_real_agent(tmp_path, agent=None, **agent_kw):
    """Mirror of _make_bot_with_real_agent (test_guidance_integration.py:102).

    Same shape, one substitution: bot.heartbeat is a REAL HeartbeatManager
    (and bot.umbral a REAL UmbralManager) on tmp paths, so
    _build_agent_callbacks → build_callbacks forwards REAL managers.
    Returns (bot, agent, heartbeat, umbral). Cleanup: await hb.shutdown() /
    um.shutdown() at test end.
    """
    ws = _setup_workspace(tmp_path)
    config = _cfg(ws, **agent_kw)
    if agent is None:
        agent = Agent(config)

    matrix_config = MatrixConfig(
        homeserver="https://matrix.local",
        user_id="@agent:matrix.local",
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )

    heartbeat = HeartbeatManager(tmp_path / "heartbeats.json", AsyncMock())
    umbral = UmbralManager(tmp_path / "umbral.json", AsyncMock())

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
    bot.heartbeat = heartbeat          # REAL manager (not a MagicMock)
    bot.umbral = umbral                # REAL manager (not a MagicMock)
    bot._steering_inbox = {}
    bot._active_turns = set()
    bot._advisor_results = {}
    bot._subagent_results = {}

    return bot, agent, heartbeat, umbral


async def _run_tool(input, callbacks, agent_config=None):
    return await execute_tool(
        name="heartbeat",
        input=input,
        tool_config={},
        agent_config=agent_config or MagicMock(
            workspace="/tmp/test", truncation_limit=50000
        ),
        tools=None,
        callbacks=callbacks,
    )


# ---------------------------------------------------------------------------
# I01 — start through the REAL callbacks seam persists + is visible
# ---------------------------------------------------------------------------


class TestI01RealCallbacksStartPersists:
    @pytest.mark.asyncio
    async def test_start_via_real_callbacks_writes_heartbeats_json(self, tmp_path):
        """I01: execute_tool run with the REAL _build_agent_callbacks output
        (which must forward the real managers) reaches the real manager:
        heartbeats.json written, status() shows the entry."""
        bot, agent, hb, um = _make_bot_with_real_agent(tmp_path)
        try:
            callbacks = bot._build_agent_callbacks(ROOM_ID, None)
            result = await _run_tool(
                {"action": "start", "interval": "15m"}, callbacks, agent.config
            )
            assert result.is_error is False, result.content
            assert "every 15m" in result.content

            hb_path = tmp_path / "heartbeats.json"
            assert hb_path.exists(), "tool start must persist via the real manager"
            entry = hb.status()[0]
            assert entry.room_id == ROOM_ID
            assert entry.interval_seconds == 900
        finally:
            await hb.shutdown()
            await um.shutdown()

    @pytest.mark.asyncio
    async def test_tool_floor_rejects_below_floor_even_with_real_manager(self, tmp_path):
        """I01 (anchoring the I04 preamble): the REAL manager accepts a 1s
        interval (manager-level allows any positive), but the TOOL must
        enforce the 5-minute floor itself."""
        bot, agent, hb, um = _make_bot_with_real_agent(tmp_path)
        try:
            callbacks = bot._build_agent_callbacks(ROOM_ID, None)

            # Manager-level: no floor enforcement.
            await hb.start(ROOM_ID, 1)
            assert hb.is_active(ROOM_ID)
            await hb.stop(ROOM_ID)
            assert not hb.is_active(ROOM_ID)

            # Tool-level: floor enforced.
            result = await _run_tool(
                {"action": "start", "interval": "60s"}, callbacks, agent.config
            )
            assert result.is_error is True
            assert FLOOR_MSG in result.content
            assert not hb.is_active(ROOM_ID)
        finally:
            await hb.shutdown()
            await um.shutdown()


# ---------------------------------------------------------------------------
# I02 — build_callbacks exposes the managers as dict keys
# ---------------------------------------------------------------------------


class TestI02BuildCallbacksManagerKeys:
    def test_build_callbacks_dict_carries_heartbeat_and_umbral(self, tmp_path):
        """I02: build_callbacks returns "heartbeat"/"umbral" keys carrying the
        manager objects passed as kwargs (anchors A1/A3 — 15 → 17 keys)."""
        hb = HeartbeatManager(tmp_path / "heartbeats.json", AsyncMock())
        um = UmbralManager(tmp_path / "umbral.json", AsyncMock())
        agent = MagicMock()
        sinks = MagicMock()
        cb = build_callbacks(
            agent, ROOM_ID, sinks,
            turn_source=None, session_log=None,
            heartbeat=hb, umbral=um,
        )
        assert "heartbeat" in cb, "callbacks dict must expose the heartbeat manager"
        assert "umbral" in cb, "callbacks dict must expose the umbral manager"
        assert cb["heartbeat"] is hb
        assert cb["umbral"] is um
        assert len(cb) == 19, f"build_callbacks must now return 19 keys (17 + spotter-era 18 + GC pair, kdsn.305), got {len(cb)}"

    @pytest.mark.asyncio
    async def test_bot_build_agent_callbacks_forwards_real_managers(self, tmp_path):
        """I02 (bot seam): MatrixBot._build_agent_callbacks forwards
        bot.heartbeat / bot.umbral through build_callbacks (matrix.py:1291
        already passes the kwargs — the dict keys are the new contract)."""
        bot, agent, hb, um = _make_bot_with_real_agent(tmp_path)
        try:
            callbacks = bot._build_agent_callbacks(ROOM_ID, "heartbeat")
            assert callbacks["heartbeat"] is hb
            assert callbacks["umbral"] is um
        finally:
            await hb.shutdown()
            await um.shutdown()


# ---------------------------------------------------------------------------
# I03 — REAL umbral active excludes tool-start
# ---------------------------------------------------------------------------


class TestI03RealUmbralExcludesStart:
    @pytest.mark.asyncio
    async def test_real_active_umbral_blocks_tool_start(self, tmp_path):
        """I03: with a REAL UmbralManager actually started in the room,
        the tool's start must fail with the slash-parity exclusion message
        and the real heartbeat manager must stay inactive."""
        bot, agent, hb, um = _make_bot_with_real_agent(tmp_path)
        try:
            await um.start(ROOM_ID, 1800)  # real umbral, now active
            assert um.is_active(ROOM_ID)

            callbacks = bot._build_agent_callbacks(ROOM_ID, None)
            result = await _run_tool(
                {"action": "start", "interval": "15m"}, callbacks, agent.config
            )
            assert result.is_error is True
            assert UMBRAL_EXCLUSION in result.content
            assert "umbral and heartbeat cannot run in the same room" in result.content
            assert not hb.is_active(ROOM_ID)
            # The exclusion must fire BEFORE any manager timer task runs —
            # the fake fire callbacks on both real managers stay untouched.
            hb.callback.assert_not_awaited()
            um.callback.assert_not_awaited()
        finally:
            await hb.shutdown()
            await um.shutdown()


# ---------------------------------------------------------------------------
# I04 — self-stop pin (hazard H2; kdsn.290 audit R1)
# ---------------------------------------------------------------------------


class TestI04SelfStopFromWithinFiredTurn:
    @staticmethod
    def _tool_callbacks(hb, room_id):
        return {
            "heartbeat": hb,
            "umbral": None,
            "room_id": room_id,
            "send_notice": AsyncMock(),
        }

    @staticmethod
    def _assert_stopped_state(hb, fired, tmp_path):
        """Shared assertion block: exactly one fire, manager inactive, and
        the removal persisted (the room vanishes from heartbeats.json)."""
        assert fired == [ROOM_ID], (
            "callback must fire exactly once (no refire after self-stop), "
            f"got {fired}"
        )
        assert not hb.is_active(ROOM_ID)
        persisted = json.loads((tmp_path / "heartbeats.json").read_text())
        assert all(e["room_id"] != ROOM_ID for e in persisted)

    @pytest.mark.asyncio
    async def test_tool_stop_from_inside_heartbeat_fire_completes(self, tmp_path):
        """I04 (production-faithful, R2 rewrite): stop called via the tool
        FROM WITHIN the fired heartbeat turn must complete WITHOUT
        CancelledError — even when the tool call runs through
        asyncio.gather, the way Agent.handle_input dispatches tool calls
        (a single call gathered, mirroring the tool-call batch). gather
        wraps the coroutine in a CHILD task, so the old task-identity
        self-stop detection inside stop() missed the gather child and
        cancel+awaited its own ancestor (cyclic cancel → RecursionError;
        audit R1). The manager's loop-context detection (contextvars
        propagate INTO gather children, but NOT into unrelated tasks like
        slash-command handling) must catch it instead: the callback fires
        exactly once over the ~3s observation window, the tool returns
        "Heartbeat stopped." non-error, the manager stops refiring, and the
        loop exits cleanly.

        Guarded by asyncio.wait_for(15s) so a hang FAILS fast instead of
        hanging the suite.
        """
        fired = []
        results = []
        hb = None

        async def fire_turn(room_id):
            """The fired turn: the tool call dispatched exactly like
            handle_input — through asyncio.gather (child-task wrapping,
            the production wiring)."""
            fired.append(room_id)
            [r] = await asyncio.gather(
                _run_tool({"action": "stop"}, self._tool_callbacks(hb, room_id))
            )
            results.append(r)

        hb = HeartbeatManager(tmp_path / "heartbeats.json", fire_turn)
        await hb.start(ROOM_ID, 1)  # manager-level: any positive interval ok

        async def _observe():
            # If stop() cancelled the current (timer) task, the
            # CancelledError would propagate into this awaiting test code.
            try:
                await asyncio.sleep(3)
            except asyncio.CancelledError:  # pragma: no cover - guard path
                pytest.fail(
                    "self-stop through the gather child cancelled the timer "
                    "task itself: loop-context detection failed (audit R1)"
                )

        try:
            await asyncio.wait_for(_observe(), timeout=15)
        except asyncio.TimeoutError:  # pragma: no cover - regression path
            pytest.fail(
                "self-stop observation window hung: the turn must EXIT "
                "cleanly inside 15s (fail-fast guard for R2)"
            )

        assert len(results) == 1, f"expected one stop result, got {results}"
        result = results[0]
        # ("unknown tool" would be the not-yet-dispatched RED state, kept
        # distinct from a wiring-level contract failure.)
        assert "unknown tool" not in result.content.lower(), result.content
        assert result.is_error is False, result.content
        assert "Heartbeat stopped." in result.content
        self._assert_stopped_state(hb, fired, tmp_path)
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_tool_stop_from_inside_fire_direct_path_completes(self, tmp_path):
        """I04 (cheap direct-path pin, R2 second test): pre-fix wiring — the
        tool call is awaited INLINE in the timer task (no gather). BOTH
        paths must work: the task-identity check catches this one directly,
        the contextvar check is a no-op match. Deliberately small; the
        regression tripwire for the timer-loop contract."""
        fired = []
        results = []
        hb = None

        async def fire_turn(room_id):
            fired.append(room_id)
            r = await _run_tool(
                {"action": "stop"}, self._tool_callbacks(hb, room_id)
            )
            results.append(r)

        hb = HeartbeatManager(tmp_path / "heartbeats.json", fire_turn)
        await hb.start(ROOM_ID, 1)

        try:
            await asyncio.wait_for(asyncio.sleep(3), timeout=15)
        except asyncio.TimeoutError:  # pragma: no cover - regression path
            pytest.fail(
                "direct-path self-stop observation window hung (fail fast for R2)"
            )
        except asyncio.CancelledError:  # pragma: no cover - guard path
            pytest.fail("direct-path self-stop cancelled the timer task (audit R1)")

        assert len(results) == 1, f"expected one stop result, got {results}"
        assert results[0].is_error is False, results[0].content
        assert "Heartbeat stopped." in results[0].content
        self._assert_stopped_state(hb, fired, tmp_path)
        await hb.shutdown()

    @pytest.mark.asyncio
    async def test_fire_turn_runs_under_manager_loop_context(self, tmp_path):
        """R1 coverage at the integration seam: inside the fired turn,
        manager.in_own_loop(room_id) is True (the contextvar was set, and
        this fired turn holds a superset of that context); in the OUTER
        test task it is False. This is the guard the tool's start-refusal
        policy consumes."""
        fired = asyncio.Event()
        seen_in_turn = []
        hb = None

        async def fire_turn(room_id):
            fired.set()
            seen_in_turn.append(hb.in_own_loop(room_id))
            await hb.stop(room_id)  # one fire then exit (self-stop)

        hb = HeartbeatManager(tmp_path / "heartbeats.json", fire_turn)
        try:
            await asyncio.wait_for(asyncio.sleep(0.05), timeout=15)
        except asyncio.TimeoutError:
            pass
        assert not hb.in_own_loop(ROOM_ID), (
            "outer (non-fire) task must not see the loop context"
        )

        await hb.start(ROOM_ID, 0.05, _initial_delay=0.01)
        try:
            await asyncio.wait_for(asyncio.sleep(1), timeout=15)
        except asyncio.TimeoutError:  # pragma: no cover - regression path
            pytest.fail("fire-turn context observation window hung")
        await hb.shutdown()

        assert fired.is_set()
        assert seen_in_turn == [True], (
            "the fired turn must see itself inside the loop context, got "
            f"{seen_in_turn}"
        )
        # After the turn completed, the context is reset for everyone.
        assert not hb.in_own_loop(ROOM_ID)


# ---------------------------------------------------------------------------
# I05 — persistence parity with the slash-start shape
# ---------------------------------------------------------------------------


class TestI05PersistenceParity:
    @pytest.mark.asyncio
    async def test_tool_start_shape_matches_slash_shape_and_resumes(self, tmp_path):
        """I05: a tool-start writes heartbeats.json in the slash-start shape
        and a fresh HeartbeatManager on the same file resume()s the entry
        with the same interval + directive."""
        bot, agent, hb, um = _make_bot_with_real_agent(tmp_path)
        try:
            callbacks = bot._build_agent_callbacks(ROOM_ID, None)
            result = await _run_tool(
                {"action": "start", "interval": "15m",
                 "directive": "watch the deploy"},
                callbacks, agent.config,
            )
            assert result.is_error is False, result.content

            data = json.loads((tmp_path / "heartbeats.json").read_bytes())
            assert isinstance(data, list) and len(data) == 1
            entry = data[0]
            assert set(entry.keys()) == {
                "room_id", "last_fired_at", "directive",
                "interval_seconds", "schedule",
            }, f"tool-start shape diverges from slash-start shape: {sorted(entry)}"
            assert entry["room_id"] == ROOM_ID
            assert entry["directive"] == "watch the deploy"
            assert entry["interval_seconds"] == 900
            assert entry["schedule"] is None
        finally:
            # Close the first manager WITHOUT persisting an empty state —
            # the entry must still be on disk for the resume below.
            await hb.shutdown()
            await um.shutdown()

        assert json.loads((tmp_path / "heartbeats.json").read_text()) != []

        # A NEW manager on the same file resume()s the entry.
        resumed = HeartbeatManager(tmp_path / "heartbeats.json", AsyncMock())
        try:
            await resumed.resume()
            entries = resumed.status()
            assert len(entries) == 1
            assert entries[0].room_id == ROOM_ID
            assert entries[0].interval_seconds == 900
            assert entries[0].directive == "watch the deploy"
        finally:
            await resumed.shutdown()


# ---------------------------------------------------------------------------
# I06 — headless transport fails clean
# ---------------------------------------------------------------------------


class TestI06HeadlessFailsClean:
    def test_headless_callbacks_have_none_managers(self):
        """I06: CLI/headless build_callbacks (cli.py:717 passes no
        heartbeat/umbral kwargs) has the keys present with value None."""
        agent = MagicMock()
        cb = build_callbacks(
            agent, ROOM_ID, HeadlessSinks(),
            turn_source=None, session_log=None,
        )
        assert "heartbeat" in cb, "headless callbacks must still carry the key"
        assert "umbral" in cb, "headless callbacks must still carry the key"
        assert cb["heartbeat"] is None
        assert cb["umbral"] is None

    @pytest.mark.asyncio
    async def test_headless_all_three_actions_fail_clean_with_steering(self):
        """I06: with the headless callbacks dict, every action returns
        is_error + steering (managers unavailable on this interface;
        heartbeats are managed in Matrix rooms via /heartbeat)."""
        agent = MagicMock()
        cb = build_callbacks(
            agent, ROOM_ID, HeadlessSinks(),
            turn_source=None, session_log=None,
        )
        for action in ("start", "stop", "status"):
            result = await _run_tool(
                {"action": action, "interval": "15m"}, cb
            )
            assert result.is_error is True, f"action={action}: {result.content}"
            assert "/heartbeat" in result.content, result.content


# ---------------------------------------------------------------------------
# F5/F9 — same-batch [stop, start] inside a fired turn: start is refused
# ---------------------------------------------------------------------------


class TestF5SameBatchStopStart:
    @pytest.mark.asyncio
    async def test_stop_then_start_same_batch_start_refused_no_new_timer(
        self, tmp_path
    ):
        """F5/F9 (re-audit) integration pin: an agent that "restarts" its
        heartbeat naturally batches [stop, start] into ONE tool-call batch,
        which handle_input runs as ONE asyncio.gather. The stop sibling
        (self-stop) removes the room's timer entry before the start sibling
        reads any state — but the start MUST still be refused on the
        loop-context mark ALONE (re-audit F5: the previous "entry exists"
        AND-term let this same-batch bypass arm a NEW timer mid-turn,
        violating the 'start from own turn is refused' policy). Real
        HeartbeatManager, both calls in ONE gather (production wiring).

        Asserts: stop returned non-error; start is is_error with the
        later-turn steering; the turn completes (wait_for fail-fast);
        the manager is inactive; NO new timer entry exists and NO refire
        over the ~3s observation window.
        """
        fired = []
        results = {}
        hb = None

        async def fire_turn(room_id):
            """ONE fired turn = ONE tool-call batch: both actions dispatched
            in a SINGLE asyncio.gather, mirroring agent.py's tool batch —
            contextvars copy into BOTH gather children (the exact
            interleaving the F5 bypass depended on)."""
            fired.append(room_id)
            stop_r, start_r = await asyncio.gather(
                _run_tool(
                    {"action": "stop"},
                    TestI04SelfStopFromWithinFiredTurn._tool_callbacks(
                        hb, room_id
                    ),
                ),
                _run_tool(
                    {"action": "start", "interval": 300},
                    TestI04SelfStopFromWithinFiredTurn._tool_callbacks(
                        hb, room_id
                    ),
                ),
            )
            results["stop"] = stop_r
            results["start"] = start_r

        hb = HeartbeatManager(tmp_path / "heartbeats.json", fire_turn)
        await hb.start(ROOM_ID, 1)  # fires after ~1s

        async def _observe():
            try:
                await asyncio.sleep(3)
            except asyncio.CancelledError:  # pragma: no cover - guard path
                pytest.fail(
                    "same-batch stop+start cancelled the timer task itself "
                    "(cyclic-cancel class must stay closed, audit R1)"
                )

        try:
            await asyncio.wait_for(_observe(), timeout=15)
        except asyncio.TimeoutError:  # pragma: no cover - regression path
            pytest.fail(
                "same-batch stop+start observation window hung: the turn "
                "must EXIT cleanly inside 15s (fail-fast guard)"
            )

        assert list(results) == ["stop", "start"], (
            f"the turn must complete and deliver both results, got {results}"
        )

        stop_r = results["stop"]
        assert stop_r.is_error is False, stop_r.content
        assert "Heartbeat stopped." in stop_r.content

        # Re-audit F5: the start sibling must be REFUSED even though the
        # stop sibling already removed the entry — the refusal keys on
        # in_own_loop (contextvar) ALONE.
        start_r = results["start"]
        assert start_r.is_error is True, (
            "same-batch start must be refused from the room's own fired "
            f"turn, got non-error: {start_r.content}"
        )
        assert "its own turn" in start_r.content, start_r.content
        assert "later turn" in start_r.content, start_r.content

        # Exactly ONE fire, no refire: no new timer may have been armed.
        assert fired == [ROOM_ID], (
            f"exactly one fire (a silently-armed new timer would refire "
            f"inside the ~3s window), got {fired}"
        )
        assert not hb.is_active(ROOM_ID)
        assert ROOM_ID not in hb._tasks, (
            "no timer entry may exist after the turn — the same-batch "
            "start must not have armed a replacement"
        )
        assert all(
            e.room_id != ROOM_ID for e in hb.status()
        ), "status() must show no entry for the room"
        persisted = json.loads((tmp_path / "heartbeats.json").read_text())
        assert all(e["room_id"] != ROOM_ID for e in persisted)
        await hb.shutdown()
