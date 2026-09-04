"""Unit tests for the built-in `heartbeat` tool (workspace-kdsn.290).

Spec: memory/projects/openalph/specs/heartbeat-tool-spec.md §2/§5 (T01–T23).
Anchors: tmp/heartbeat-tool/anchors.md (verified against /opt/openalph).

The tool lets an agent session manage its own per-room heartbeat timer via a
single built-in tool with an `action` param ("start" | "stop" | "status").
Room scoping: the tool ALWAYS operates on callbacks["room_id"] — never on any
input parameter (there is no cross-room surface in v1).

Production dispatch seam:

    await execute_tool(
        name="heartbeat",
        input={"action": ..., ...},
        tool_config={},
        agent_config=<config-shaped object>,
        tools=None,
        callbacks=<dict with "heartbeat"/"umbral"/"room_id"/"send_notice">,
    ) -> ToolResult            (tools/__init__.py:1149)

Managers arrive via the callbacks dict: "heartbeat" is a HeartbeatManager
(or None on headless transports), "umbral" an UmbralManager or None.

RED suite: the implementation does not exist yet — every test below (except
the anchored parse-helper guards in TestAnchoredParseHelpers) is expected to
FAIL against current main, most on "Unknown tool: heartbeat".
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openalph.heartbeat import HeartbeatEntry, HeartbeatManager, parse_interval
from openalph.tools import execute_tool


ROOM_ID = "!room:matrix.local"

# Contract-canonical strings (spec §2 result contract + slash-handler parity):
FLOOR_MSG = "Minimum interval is 5m."
UMBRAL_EXCLUSION = "Stop the umbral timer first"
NO_HEARTBEAT_ACTIVE = "No heartbeat active in this room."


def _agent_config():
    """Config-shaped stand-in for execute_tool's agent_config parameter.

    execute_tool reads agent_config only for credential collection
    (defensive: anything without a dict .providers treated as having none)
    and for sub-agent spawning; the heartbeat handler never touches it.
    """
    return MagicMock(workspace="/tmp/test", truncation_limit=50000)


def _entry(**overrides):
    """HeartbeatEntry with sensible defaults for status/stop mocking."""
    defaults = dict(
        room_id=ROOM_ID,
        interval_seconds=900,
        seconds_until_next=847,
        directive=None,
        schedule=None,
        tz=None,
    )
    defaults.update(overrides)
    return HeartbeatEntry(**defaults)


def _fake_hb(*, umbral_active=False):
    """Spec'd fake HeartbeatManager double + matching umbral fake.

    Spec-limited to MagicMock(spec=HeartbeatManager) so tests exercise only
    the anchored manager surface the tool uses.
    """
    hb = MagicMock(spec=HeartbeatManager)
    hb.start = AsyncMock()
    hb.start_schedule = AsyncMock()
    hb.stop = AsyncMock(return_value=True)
    hb.resume = AsyncMock(return_value=[])
    hb.shutdown = AsyncMock()
    hb.status.return_value = []
    hb.is_active.return_value = False
    # Not inside a fired turn by default: the spec'd default
    # in_own_loop() returns a truthy MagicMock(), which would trip the
    # start-from-own-turn refusal in every ordinary start test — the F5
    # refusal keys on in_own_loop ALONE (not on entry-exists), so the
    # double MUST state its loop context explicitly.
    hb.in_own_loop.return_value = False
    hb.directive_for.return_value = None

    um = MagicMock(spec=HeartbeatManager)  # duck-compatible with UmbralManager
    um.start = AsyncMock()
    um.start_schedule = AsyncMock()
    um.stop = AsyncMock(return_value=True)
    um.resume = AsyncMock(return_value=[])
    um.shutdown = AsyncMock()
    um.status.return_value = []
    um.is_active.return_value = umbral_active
    um.directive_for.return_value = None
    return hb, um


def _callbacks(hb, um, *, room_id=ROOM_ID):
    """The callbacks dict the tool contract expects from build_callbacks."""
    return {
        "heartbeat": hb,
        "umbral": um,
        "room_id": room_id,
        "send_notice": AsyncMock(),
    }


# Room-id keys the tool MUST NOT honor when present in the input — room
# scoping always comes from callbacks["room_id"], never from input (spec §2:
# "There is no cross-room parameter in v1").
_FOREIGN_ROOM = "!foreign:matrix.local"
_ALL_ROOM_KEYS = ("room", "room_id", "target_room", "target", "roomId")


async def _run(input, callbacks):
    return await execute_tool(
        name="heartbeat",
        input=input,
        tool_config={},
        agent_config=_agent_config(),
        tools=None,
        callbacks=callbacks,
    )


# ---------------------------------------------------------------------------
# Anchored helpers the tool MUST use exist today (heartbeat.py:48,93 and
# _timer.py class attrs). These guard assumptions behind contract literals
# and are expected to ALREADY PASS (anchored current behavior).
# ---------------------------------------------------------------------------


class TestAnchoredParseHelpers:
    def test_parse_interval_basic(self):
        assert parse_interval("15m") == 900
        assert parse_interval("1h") == 3600
        assert parse_interval("300s") == 300
        assert parse_interval("60S") == 60  # case-insensitive
        assert parse_interval(" 5m ") == 300  # strips whitespace

    def test_parse_interval_invalid_returns_none_never_raises(self):
        assert parse_interval("banana") is None
        assert parse_interval("") is None
        assert parse_interval("15x") is None
        assert parse_interval("0m") is None
        assert parse_interval("-5m") is None

    def test_floor_constant(self):
        """HeartbeatManager._floor is the canonical 5-minute floor (300)."""
        assert HeartbeatManager._floor == 300


# ---------------------------------------------------------------------------
# T01–T06: start
# ---------------------------------------------------------------------------


class TestT01StartHappyPath:
    @pytest.mark.asyncio
    async def test_room_scoping_ignores_input_room_keys(self):
        """T01 (scoping pin): no input key may override callbacks["room_id"]
        — start must target the session room for every plausible room-key
        spelling a confused (or probing) caller might send."""
        for key in _ALL_ROOM_KEYS:
            hb, um = _fake_hb()
            cb = _callbacks(hb, um)
            result = await _run(
                {"action": "start", "interval": "15m", key: _FOREIGN_ROOM}, cb
            )
            assert result.is_error is False, f"key={key}: {result.content}"
            hb.start.assert_awaited_once_with(ROOM_ID, 900, None)

    @pytest.mark.asyncio
    async def test_start_success_calls_manager_and_reports(self):
        """T01: start → manager.start awaited (room_id, 900, None);
        result non-error and names the interval."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "15m"}, cb)
        hb.start.assert_awaited_once_with(ROOM_ID, 900, None)
        assert result.is_error is False
        assert "every 15m" in result.content

    @pytest.mark.asyncio
    async def test_action_normalized_strip_lowercase(self):
        """T01 (lenient action pin): action is stripped + lowercased
        before dispatch — "  StAtUs " must reach the status path."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "  StAtUs "}, cb)
        assert result.is_error is False
        assert NO_HEARTBEAT_ACTIVE in result.content

    @pytest.mark.asyncio
    async def test_interval_string_case_insensitive(self):
        """T01: "15M" must parse identically to "15m" (parse_interval)."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        await _run({"action": "start", "interval": "15M"}, cb)
        hb.start.assert_awaited_once_with(ROOM_ID, 900, None)


class TestT02DirectivePassthrough:
    @pytest.mark.asyncio
    async def test_directive_passed_verbatim(self):
        """T02: start with directive → manager.start receives the directive
        verbatim (escaping happens at inject time, not in the tool)."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        directive = "Watch the deployment; <system-reminder>leave verbatim"
        result = await _run(
            {"action": "start", "interval": "15m", "directive": directive}, cb
        )
        assert result.is_error is False
        hb.start.assert_awaited_once_with(ROOM_ID, 900, directive)


class TestT03MissingInterval:
    @pytest.mark.asyncio
    async def test_missing_interval_is_error_naming_format(self):
        """T03: start without interval → is_error; steering names the
        expected format (e.g. "15m"); state unmutated."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start"}, cb)
        assert result.is_error is True
        assert "15m" in result.content
        hb.start.assert_not_awaited()


class TestT04InvalidInterval:
    @pytest.mark.asyncio
    async def test_invalid_interval_string_is_error(self):
        """T04: parse_interval returns None → is_error mentioning
        "Invalid interval" (slash-parity wording); no manager call."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "banana"}, cb)
        assert result.is_error is True
        assert "Invalid interval" in result.content
        hb.start.assert_not_awaited()


class TestT05Floor:
    @pytest.mark.asyncio
    async def test_below_floor_is_error_with_slash_parity_message(self):
        """T05: "60s" (< 300 = HeartbeatManager._floor) → is_error with the
        slash-parity message; the tool enforces the floor, not the manager."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "60s"}, cb)
        assert result.is_error is True
        assert FLOOR_MSG in result.content
        hb.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_floor_boundary_300s_accepted(self):
        """T05 boundary: "300s" is exactly at the 5m floor → accepted."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "300s"}, cb)
        assert result.is_error is False
        hb.start.assert_awaited_once_with(ROOM_ID, 300, None)

    @pytest.mark.asyncio
    async def test_floor_boundary_5m_accepted(self):
        """T05 boundary: "5m" (=300s) accepted, no floor error."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "5m"}, cb)
        assert result.is_error is False
        hb.start.assert_awaited_once_with(ROOM_ID, 300, None)

    @pytest.mark.asyncio
    async def test_floor_applies_to_integer_intervals(self):
        """T05: integer intervals are floored the same as strings."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": 60}, cb)
        assert result.is_error is True
        assert FLOOR_MSG in result.content
        hb.start.assert_not_awaited()


class TestT06IntegerInterval:
    @pytest.mark.asyncio
    async def test_positive_integer_is_seconds_directly(self):
        """T06: a positive JSON integer interval is treated as seconds."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": 900}, cb)
        assert result.is_error is False
        hb.start.assert_awaited_once_with(ROOM_ID, 900, None)

    @pytest.mark.asyncio
    async def test_bool_interval_rejected(self):
        """T06: bool is an int subclass but must NOT be treated as an
        interval — is_error, no manager call.

        Pinned against a PARTIAL implementation too: an implementor who
        registers the tool but forgets the bool guard makes this RED on
        "900" (True coerced to 1 second) instead of "Unknown tool"."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": True}, cb)
        assert result.is_error is True
        assert "unknown tool" not in result.content.lower()  # tool must exist
        assert "900" not in result.content  # bool must never coerce to int
        hb.start.assert_not_awaited()


# ---------------------------------------------------------------------------
# T07–T08: umbral interaction
# ---------------------------------------------------------------------------


class TestT07UmbralActiveExcludes:
    @pytest.mark.asyncio
    async def test_umbral_active_blocks_start(self):
        """T07: an active umbral in the room excludes heartbeat start —
        exclusion message mirrors the slash wording; start NOT called."""
        hb, um = _fake_hb(umbral_active=True)
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "15m"}, cb)
        assert result.is_error is True
        assert UMBRAL_EXCLUSION in result.content
        assert "umbral and heartbeat cannot run in the same room" in result.content
        hb.start.assert_not_awaited()


class TestT08UmbralManagerNone:
    @pytest.mark.asyncio
    async def test_missing_umbral_manager_does_not_block_start(self):
        """T08: umbral manager None (transport didn't wire one) + heartbeat
        manager present → start still works (exclusion check skipped)."""
        hb, _ = _fake_hb()
        cb = _callbacks(hb, None)
        result = await _run({"action": "start", "interval": "15m"}, cb)
        assert result.is_error is False
        hb.start.assert_awaited_once_with(ROOM_ID, 900, None)


# ---------------------------------------------------------------------------
# T09–T11: stop
# ---------------------------------------------------------------------------


class TestT09StopHappyPath:
    @pytest.mark.asyncio
    async def test_stop_success_message_and_notice(self):
        """T09: stop → "Heartbeat stopped." (slash-parity), non-error,
        exactly one notice fired mentioning "Heartbeat stopped"."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "stop"}, cb)
        hb.stop.assert_awaited_once_with(ROOM_ID)
        assert result.is_error is False
        assert "Heartbeat stopped." in result.content
        cb["send_notice"].assert_awaited_once()
        notice_args = cb["send_notice"].await_args.args
        assert notice_args[0] == ROOM_ID  # notices are room-scoped
        assert "Heartbeat stopped" in notice_args[1]
        assert "💓" in notice_args[1]


class TestT10StopNothingActive:
    @pytest.mark.asyncio
    async def test_stop_with_nothing_active_is_error(self):
        """T10: manager.stop returns False → is_error with the slash-parity
        "No heartbeat active in this room." message. (Contrast T12: the same
        message for status is NOT an error.)"""
        hb, um = _fake_hb()
        hb.stop = AsyncMock(return_value=False)
        cb = _callbacks(hb, um)
        result = await _run({"action": "stop"}, cb)
        assert result.is_error is True
        assert NO_HEARTBEAT_ACTIVE in result.content


class TestT11StopScheduleModeEntry:
    @pytest.mark.asyncio
    async def test_stop_on_schedule_entry_surfaces_spec(self):
        """T11: stopping an operator-started schedule-mode entry succeeds,
        and the result surfaces the entry's schedule spec."""
        hb, um = _fake_hb()
        hb.stop = AsyncMock(return_value=True)
        hb.status.return_value = [
            _entry(
                interval_seconds=None,
                schedule="30 6 * * 1-5",
                tz="UTC",
                seconds_until_next=43200,
            )
        ]
        cb = _callbacks(hb, um)
        result = await _run({"action": "stop"}, cb)
        assert result.is_error is False
        assert "Heartbeat stopped." in result.content
        assert "was schedule" in result.content
        assert '30 6 * * 1-5' in result.content


# ---------------------------------------------------------------------------
# T12–T15: status
# ---------------------------------------------------------------------------


class TestT12StatusInactive:
    @pytest.mark.asyncio
    async def test_status_with_nothing_active_is_not_an_error(self):
        """T12: status is a read — absence of a timer is information, so
        is_error stays False (contrast T10) and the message says so.
        Discriminates against a partial implementation that shares one
        "nothing active" path between stop and status."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "status"}, cb)
        assert NO_HEARTBEAT_ACTIVE in result.content
        assert result.is_error is False, result.content


class TestT13StatusIntervalEntry:
    @pytest.mark.asyncio
    async def test_status_renders_interval_and_next_fire(self):
        """T13: active interval entry → "every {fmt}, next in {fmt}" with
        a human-readable (duration-style, not raw-seconds) next-in."""
        hb, um = _fake_hb()
        hb.is_active.return_value = True
        hb.status.return_value = [_entry(interval_seconds=900, seconds_until_next=847)]
        cb = _callbacks(hb, um)
        result = await _run({"action": "status"}, cb)
        assert result.is_error is False
        assert "every 15m" in result.content, result.content
        assert "next in" in result.content
        assert "next in 847" not in result.content  # must be formatted
        assert "14m" in result.content  # 847s → "14m 7s"


class TestT14StatusScheduleEntry:
    @pytest.mark.asyncio
    async def test_status_renders_schedule_spec_and_tz(self):
        """T14: schedule-mode entry → spec + timezone surfaced."""
        hb, um = _fake_hb()
        hb.is_active.return_value = True
        hb.status.return_value = [
            _entry(
                interval_seconds=None,
                schedule="30 6 * * 1-5",
                tz="America/Denver",
                seconds_until_next=3900,
            )
        ]
        cb = _callbacks(hb, um)
        result = await _run({"action": "status"}, cb)
        assert result.is_error is False
        assert "30 6 * * 1-5" in result.content
        assert "America/Denver" in result.content
        assert "next in" in result.content


class TestT15StatusDirectiveRenderedTruncated:
    @pytest.mark.asyncio
    async def test_status_directive_suffix_is_truncated(self):
        """T15: a long directive is rendered as " · directive: {truncated}"
        — never verbatim."""
        hb, um = _fake_hb()
        long_directive = "d" * 500
        hb.is_active.return_value = True
        hb.status.return_value = [_entry(directive=long_directive)]
        cb = _callbacks(hb, um)
        result = await _run({"action": "status"}, cb)
        assert result.is_error is False
        assert "directive:" in result.content
        assert long_directive not in result.content
        assert "d" * 200 not in result.content  # truncation actually happened

    @pytest.mark.asyncio
    async def test_status_short_directive_rendered_verbatim(self):
        """T15 complement: truncation must not mangle ordinary directives
        (a comparator-eager implementation that over-truncates fails here)."""
        hb, um = _fake_hb()
        hb.is_active.return_value = True
        hb.status.return_value = [_entry(directive="watch the deploy")]
        cb = _callbacks(hb, um)
        result = await _run({"action": "status"}, cb)
        assert result.is_error is False
        assert " · directive: watch the deploy" in result.content


# ---------------------------------------------------------------------------
# T16–T17: transport/scope guards
# ---------------------------------------------------------------------------


class TestT16SubAgentSentinel:
    @staticmethod
    def _sub_callbacks():
        """Production-faithful sub-agent callbacks (R3 rewire): run_subagent
        builds its tool-call callbacks with room_id "__sub__" and does NOT
        forward heartbeat/umbral manager keys — so previous revisions of
        this test (which DID wire a manager) could never observe the guard
        ordering bug they claimed to pin."""
        return {"room_id": "__sub__"}

    @pytest.mark.asyncio
    async def test_sub_room_scoped_start_refused_mentions_sub_agents(self):
        """T16: callbacks["room_id"] == "__sub__" (the sub-agent sentinel)
        → start is refused, is_error, steering mentions sub-agents — even
        with NO heartbeat key in production wiring (the refusal must not be
        masked by the transport-unavailable guard)."""
        cb = self._sub_callbacks()
        assert "heartbeat" not in cb  # pin the production wiring itself
        result = await _run({"action": "start", "interval": "15m"}, cb)
        assert result.is_error is True
        assert "sub-agent" in result.content.lower(), result.content

    @pytest.mark.asyncio
    async def test_sub_room_scoped_stop_and_status_also_refused(self):
        """T16: the same sub-agent refusal applies to stop and status —
        the sentinel has no per-room timer semantics for ANY action."""
        for action in ("stop", "status"):
            cb = self._sub_callbacks()
            result = await _run({"action": action}, cb)
            assert result.is_error is True, f"action={action}"
            assert "sub-agent" in result.content.lower(), result.content


class TestT17TransportUnavailable:
    @pytest.mark.asyncio
    async def test_none_heartbeat_manager_fails_clean_all_actions(self):
        """T17: headless/CLI wires "heartbeat"=None → all three actions fail
        clean with is_error steering toward Matrix / the /heartbeat slash."""
        for action in ("start", "stop", "status"):
            hb, um = _fake_hb()
            cb = _callbacks(None, um)
            result = await _run({"action": action, "interval": "15m"}, cb)
            assert result.is_error is True, f"action={action}"
            assert "/heartbeat" in result.content or "matrix" in result.content.lower(), (
                result.content
            )
            um.stop.assert_not_awaited()  # nothing mutated

    @pytest.mark.asyncio
    async def test_missing_callbacks_entirely_fails_clean_no_raise(self):
        """T17: callbacks=None → a transport-unavailable error result
        (distinct from a dispatch-level unknown-tool rejection), never
        raises."""
        result = await _run({"action": "start", "interval": "15m"}, None)
        assert result.is_error is True
        assert "/heartbeat" in result.content or "unavailable" in result.content.lower(), (
            result.content
        )


# ---------------------------------------------------------------------------
# T18–T21: bad input + manager exceptions
# ---------------------------------------------------------------------------


class TestT18UnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action_lists_valid_actions(self):
        """T18: unknown action → is_error; steering lists the valid actions
        so the caller can self-correct."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "explode"}, cb)
        assert result.is_error is True
        for valid in ("start", "stop", "status"):
            assert valid in result.content, result.content
        hb.start.assert_not_awaited()


class TestT19MissingAction:
    @pytest.mark.asyncio
    async def test_missing_action_is_error(self):
        """T19: action is required — absent → is_error with steering."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"interval": "15m"}, cb)
        assert result.is_error is True
        assert "action" in result.content.lower()


class TestT20BadShapes:
    """T20 bad shapes → is_error steering FROM THE TOOL (not a dispatch-level
    unknown-tool rejection), state unmutated, never raises."""

    @staticmethod
    def _assert_tool_level_rejection(result):
        assert result.is_error is True
        assert "unknown tool" not in result.content.lower()

    @pytest.mark.asyncio
    async def test_stringified_action_is_not_silently_coerced(self):
        """T20 hedge: a non-str action must be rejected, not shoved through
        the lenient-normalization path with some coercion — "True" is not a
        valid action and must not silently dispatch to anything."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": 1}, cb)
        self._assert_tool_level_rejection(result)
        hb.start.assert_not_awaited()
        hb.stop.assert_not_awaited()
        hb.status.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_string_action_is_error_no_raise(self):
        """T20: non-str action (True) → is_error, never raises, no manager
        call (bool short-circuits the lenient str coercion)."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": True}, cb)
        self._assert_tool_level_rejection(result)
        hb.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_string_directive_is_error_state_unmutated(self):
        """T20: non-str directive → is_error, manager.start NEVER called
        (state unmutated — a coercing implementation would leak through)."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run(
            {"action": "start", "interval": "15m", "directive": 123}, cb
        )
        self._assert_tool_level_rejection(result)
        hb.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dict_interval_is_error_state_unmutated(self):
        """T20: dict interval → is_error, no manager call, no raise."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run(
            {"action": "start", "interval": {"value": 15, "unit": "m"}}, cb
        )
        self._assert_tool_level_rejection(result)
        hb.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_float_interval_is_error(self):
        """T20: non-integral numeric interval → is_error (only a positive
        JSON integer is treated as seconds directly)."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": 1.5}, cb)
        self._assert_tool_level_rejection(result)
        hb.start.assert_not_awaited()


class TestT21ManagerExceptionSanitized:
    @pytest.mark.asyncio
    async def test_manager_exception_on_start_surfaced_sanitised(self):
        """T21: manager raises → is_error result, never propagates, and the
        content carries only type(e).__name__-level detail (no traceback
        text, no exception message body)."""
        hb, um = _fake_hb()
        hb.start = AsyncMock(
            side_effect=RuntimeError("boom in db layer: /secret/path detail")
        )
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "15m"}, cb)
        assert result.is_error is True
        assert "RuntimeError" in result.content  # exception type surfaced
        assert "Traceback" not in result.content
        assert "boom in db layer" not in result.content  # message body sanitized

    @pytest.mark.asyncio
    async def test_manager_exception_on_stop_never_propagates(self):
        """T21: the same exception hygiene applies to the stop path."""
        hb, um = _fake_hb()
        hb.stop = AsyncMock(
            side_effect=OSError("disk explodes at /secret/path")
        )
        cb = _callbacks(hb, um)
        result = await _run({"action": "stop"}, cb)
        assert result.is_error is True
        assert "OSError" in result.content
        assert "disk explodes" not in result.content


# ---------------------------------------------------------------------------
# T22–T23: notices + replace semantics
# ---------------------------------------------------------------------------


class TestT22NoticeDiscipline:
    @pytest.mark.asyncio
    async def test_start_fires_exactly_one_notice(self):
        """T22: start → one room-scoped notice mentioning "Heartbeat started"."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        await _run({"action": "start", "interval": "15m"}, cb)
        cb["send_notice"].assert_awaited_once()
        notice_args = cb["send_notice"].await_args.args
        assert notice_args[0] == ROOM_ID
        assert "Heartbeat started" in notice_args[1]
        assert "💓" in notice_args[1]

    @pytest.mark.asyncio
    async def test_status_fires_no_notice(self):
        """T22: reads don't need timeline noise — status NEVER notices,
        active or not."""
        for active in (False, True):
            hb, um = _fake_hb()
            if active:
                hb.is_active.return_value = True
                hb.status.return_value = [_entry()]
            cb = _callbacks(hb, um)
            result = await _run({"action": "status"}, cb)
            assert result.is_error is False
            cb["send_notice"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_actions_fire_no_notice(self):
        """T22: error paths never emit notices (no timeline noise on
        failures the operator can't act on)."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start"}, cb)  # missing interval → error
        assert result.is_error is True
        assert "unknown tool" not in result.content.lower()  # tool-level error
        cb["send_notice"].assert_not_awaited()

        cb2_hb, _ = _fake_hb()
        cb2_hb.stop = AsyncMock(return_value=False)
        cb2 = _callbacks(cb2_hb, _fake_hb()[1])
        result2 = await _run({"action": "stop"}, cb2)  # nothing active → error
        assert result2.is_error is True
        assert "unknown tool" not in result2.content.lower()
        cb2["send_notice"].assert_not_awaited()


class TestT23ReplaceOnStart:
    @pytest.mark.asyncio
    async def test_reissued_start_delegates_to_manager(self):
        """T23: start while a timer is already active → delegates to
        manager.start anyway (replace is the tested manager contract:
        _start_common cancels and re-arms). No refusal, no special message,
        and the result still reports the NEW interval."""
        hb, um = _fake_hb()
        hb.is_active.return_value = True
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "30m"}, cb)
        assert result.is_error is False
        hb.start.assert_awaited_once_with(ROOM_ID, 1800, None)
        assert "every 30m" in result.content


# ---------------------------------------------------------------------------
# Remediation pins (kdsn.290 audit): R1 / R3 / R4 / R5
# ---------------------------------------------------------------------------


class TestR1StartFromOwnTurnInPlace:
    """kdsn.310: start from the room's own fired turn is an IN-PLACE
    cadence/directive update — never cancel-and-replace (the old loop task
    is the ancestor of the tool's gather child; cancelling it cyclically
    cancels the running turn), never a refusal (the old 'interval changes
    need a later turn' contract made autonomous heartbeat self-modification
    impossible: an agent that stopped its own heartbeat could never re-arm
    until an operator message created a non-heartbeat turn — live session
    d0Oh5MZ7PC2ahnOs41 lines 308-312, two consecutive refusals).

    Validation (interval parse, floor, umbral exclusion, shapes) applies
    EQUALLY in the own-turn path — own-turn is a different apply mechanism,
    not a policy exemption."""

    @pytest.mark.asyncio
    async def test_start_from_own_fired_turn_delegates_to_in_place_apply(self):
        """Own-turn start → hb.apply_in_own_turn(room, seconds, directive);
        the cancel-and-replace manager start must NEVER run from the room's
        own fired turn."""
        hb, um = _fake_hb()
        hb.in_own_loop.return_value = True
        hb.apply_in_own_turn = AsyncMock()
        cb = _callbacks(hb, um)
        result = await _run(
            {"action": "start", "interval": "30m", "directive": "phase 2"}, cb
        )
        assert result.is_error is False, result.content
        hb.apply_in_own_turn.assert_awaited_once_with(ROOM_ID, 1800, "phase 2")
        hb.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_start_from_own_turn_floor_still_enforced(self):
        hb, um = _fake_hb()
        hb.in_own_loop.return_value = True
        hb.apply_in_own_turn = AsyncMock()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "1m"}, cb)
        assert result.is_error is True
        assert FLOOR_MSG in result.content, result.content
        hb.apply_in_own_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_start_from_own_turn_invalid_interval_still_rejected(self):
        hb, um = _fake_hb()
        hb.in_own_loop.return_value = True
        hb.apply_in_own_turn = AsyncMock()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "abc"}, cb)
        assert result.is_error is True
        assert "Invalid interval" in result.content, result.content
        hb.apply_in_own_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_start_from_own_turn_missing_interval_is_error(self):
        hb, um = _fake_hb()
        hb.in_own_loop.return_value = True
        hb.apply_in_own_turn = AsyncMock()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start"}, cb)
        assert result.is_error is True
        assert "Missing required parameter 'interval'" in result.content
        hb.apply_in_own_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_start_from_own_turn_umbral_exclusion_still_enforced(self):
        """R5 fail-closed: umbral verification/exclusion runs in the own-turn
        path too — own-turn is not an exclusion bypass."""
        hb, um = _fake_hb(umbral_active=True)
        hb.in_own_loop.return_value = True
        hb.apply_in_own_turn = AsyncMock()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "30m"}, cb)
        assert result.is_error is True
        assert UMBRAL_EXCLUSION in result.content, result.content
        hb.apply_in_own_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_start_from_own_turn_omitted_directive_clears_and_says_so(self):
        """Directive semantics match the manager seam: None clears. When a
        standing directive existed and start omits one, the result must say
        it was cleared (silent clears are how agents lose standing
        routines)."""
        hb, um = _fake_hb()
        hb.in_own_loop.return_value = True
        hb.directive_for.return_value = "old standing directive"
        hb.apply_in_own_turn = AsyncMock()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "30m"}, cb)
        assert result.is_error is False, result.content
        hb.apply_in_own_turn.assert_awaited_once_with(ROOM_ID, 1800, None)
        assert "cleared" in result.content.lower(), result.content

    @pytest.mark.asyncio
    async def test_start_from_own_turn_notice_emitted(self):
        hb, um = _fake_hb()
        hb.in_own_loop.return_value = True
        hb.apply_in_own_turn = AsyncMock()
        cb = _callbacks(hb, um)
        await _run({"action": "start", "interval": "30m"}, cb)
        cb["send_notice"].assert_awaited_once()
        notice = cb["send_notice"].await_args.args[-1]
        assert "30m" in notice, notice

    @pytest.mark.asyncio
    async def test_start_from_own_turn_apply_failure_is_error_never_raises(self):
        hb, um = _fake_hb()
        hb.in_own_loop.return_value = True
        hb.apply_in_own_turn = AsyncMock(side_effect=RuntimeError("boom"))
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "30m"}, cb)
        assert result.is_error is True
        assert "Heartbeat update failed: RuntimeError" in result.content

    @pytest.mark.asyncio
    async def test_start_from_other_turn_still_replaces(self):
        """R1 contract guard: in_own_loop False (start issued from a NORMAL
        turn while the timer happens to be running) still delegates to the
        manager — T23 replace semantics are untouched by the policy."""
        hb, um = _fake_hb()
        hb.status.return_value = [_entry()]
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "30m"}, cb)
        assert result.is_error is False, result.content
        hb.start.assert_awaited_once_with(ROOM_ID, 1800, None)

    @pytest.mark.asyncio
    async def test_in_own_loop_probe_failure_falls_through_never_raises(self):
        """R1 robustness: a manager whose probe raises must not crash the
        turn — the own-turn check is advisory and wraps its probe; the
        ordinary start path then applies."""
        hb, um = _fake_hb()
        hb.in_own_loop.side_effect = RuntimeError("probe boom")
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "30m"}, cb)
        assert result.is_error is False, result.content
        hb.start.assert_awaited_once_with(ROOM_ID, 1800, None)


class TestR4MissingRoomId:
    """R4: room scoping reads callbacks.get("room_id") — a missing key is
    an is_error steering result, never a raised KeyError."""

    @pytest.mark.asyncio
    async def test_missing_room_id_key_is_error_never_raises(self):
        """R4: callbacks dict WITHOUT a "room_id" key (manager present) →
        is_error steering; never a KeyError escaping the tool; the manager
        is never touched."""
        hb, um = _fake_hb()
        cb = {"heartbeat": hb, "umbral": um, "send_notice": AsyncMock()}
        for action in ("start", "stop", "status"):
            result = await _run({"action": action, "interval": "15m"}, cb)
            assert result.is_error is True, f"action={action}: {result.content}"
        hb.start.assert_not_awaited()
        hb.stop.assert_not_awaited()
        hb.is_active.assert_not_called()


class TestR5UmbralFailClosed:
    """R5: if umbral state cannot be verified (is_active raises), start is
    refused fail-closed with the failure named."""

    @pytest.mark.asyncio
    async def test_raising_is_active_refuses_start_fail_closed(self):
        """R5: um.is_active(room_id) raising means umbral state is
        UNVERIFIABLE — the tool must REFUSE start (fail closed, is_error,
        naming the failure) rather than crash the turn or start blind."""
        hb, um = _fake_hb()
        um.is_active.side_effect = RuntimeError("umbral db wedged")
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "15m"}, cb)
        assert result.is_error is True
        assert "Cannot verify umbral state" in result.content, result.content
        assert "RuntimeError" in result.content  # failure named, type-only
        assert "umbral db wedged" not in result.content  # message body sanitized
        hb.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_false_is_active_after_wrap_still_starts(self):
        """R5 contract guard: the wrapped check returning False (inactive
        umbral) must let start through — fail-closed must not become
        always-closed."""
        hb, um = _fake_hb()
        cb = _callbacks(hb, um)
        result = await _run({"action": "start", "interval": "15m"}, cb)
        assert result.is_error is False, result.content
        hb.start.assert_awaited_once_with(ROOM_ID, 900, None)
