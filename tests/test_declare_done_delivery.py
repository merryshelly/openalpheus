"""Declare-done DELIVERY decoupling — TDD RED suite (bead workspace-kdsn.350.16).

Design contract: memory/projects/openalph/declare-done/design-memo.md §3
AMENDMENT #2 (SB-approved 2026-10-02, "go with your recommendations").

The defect this suite pins (SB-reported; mechanism code-verified @ d5679dd9)
--------------------------------------------------------------------------
The §3 hold-back withholds the model's text on every NON-streaming transport,
so the turn's report is destroyed whenever the model answers the corrective
with a BARE `declare_done{}` — which is exactly what the corrective invites.

Chain: hold-back (agent.py) appends the text to in-memory history + the
[SYSTEM:] corrective, then `continue` — `handle_input` never returns, so the
text never reaches the return-value/final-send path. Next iteration the bare
declare hits the terminal dispatch, which returns
`accumulated_text or response.content` == "". The funnel then sees an empty
response with conclusion "declared" and sends nothing.

Why it looked fine: the live interactive path is unaffected —
`StreamingDelivery` pushes deltas live and `_finalize` sets `_delivered`
before the corrective fires. This is a TRANSPORT-ASYMMETRY bug, not universal
withholding. Total loss on heartbeat/umbral (matrix.py funnel, no streaming by
design — the "fire report" class in workspace-im7t.61.45), exec/headless, and
any other return-value-driven consumer.

The amendment
-------------
D1  deliver at hold-back: a fail-soft `deliver_turn_text(text)` callback is
    fired at the hold-back site. Non-streaming room transports (heartbeat/
    umbral) wire it to send-and-record; the live path does NOT (streaming owns
    delivery); exec does NOT (no mid-turn channel — D2 carries it).
D2  return the turn's last text: a bare declaration immediately following a
    hold-back returns the HELD text, not "". A turn that emitted no text still
    returns "". A hold-back superseded by further work does not stand as the
    final reply.
D3  no double delivery: ONE shared predicate compares the transport's recorded
    delivered text against the turn-end response; both funnels use it.
D4  declaration semantics UNCHANGED: ledger classes, corrective gating on
    registration, the one-shot bound, exactly-two-model-calls, and the
    declared|undeclared|cap_exhausted discriminator all stand as ratified.
D5  the corrective is delivery-truthful: it states the text has been delivered
    and instructs the model NOT to repeat it.

Harness notes
-------------
- Agent-level pins reuse test_declare_done's harness (_make_agent/_drive
  pattern), adding a callbacks-carrying driver so the D1 seam can be observed.
- The real-path pins (tool-management "one lesson") run a REAL Agent through
  the REAL `MatrixBot._build_agent_callbacks` construction path and the REAL
  `_run_heartbeat_turn` funnel — mocking only provider + nio client. This is
  the coverage whose ABSENCE let the defect ship: the .350.2 red suite pinned
  the live path (where streaming masks the bug) and explicitly disclaimed a
  single-message pin, so nothing ever asserted heartbeat-path delivery.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from test_declare_done import (  # noqa: E402
    _corrective_entries,
    _declare_call,
    _exec_tool_stub,
    _make_agent,
    _shell_call,
    _stream_responses,
    _text_response,
)
from test_gapfill_trigger_dedup import (  # noqa: E402
    ROOM_ID,
    USER,
    drain,
    gapfill_page,
    make_event,
    make_room,
)

ROOM_A = ROOM_ID


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _delivery_recorder():
    """Collect the texts handed to the D1 seam."""
    seen = []

    async def _deliver(text):
        seen.append(text)

    return seen, _deliver


def _drive_cb(agent, room_id, responses, callbacks=None, text="do the thing",
              thinking=None):
    """`_drive` with a caller-supplied callbacks dict (the D1 seam rides it) and
    an optional thinking level (the empty-max_tokens recovery needs thinking on)."""
    with (
        patch("openalph.agent.stream",
              side_effect=_stream_responses(responses)),
        patch("openalph.agent.complete", new_callable=AsyncMock),
        patch("openalph.agent.execute_tool",
              side_effect=_exec_tool_stub(shell="ok")),
    ):
        return asyncio.run(
            agent.handle_input(text, room_id, callbacks=callbacks,
                               thinking=thinking))


def _make_heartbeat_bot(tmp_path):
    """A REAL Agent + REAL callbacks on a real workspace, driven through the
    REAL heartbeat funnel. Only the provider (stream) and the nio client are
    mocked. The room is pre-activated so the turn does not need gapfill
    plumbing (which is orthogonal to delivery)."""
    from test_guidance_integration import _make_bot_with_real_agent

    bot, agent = _make_bot_with_real_agent(
        tmp_path, tools_list=("declare_done", "shell"))
    bot._active_rooms.add(ROOM_A)
    bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
    return bot, agent


def _sent_bodies(bot):
    """Every body the transport actually sent, edits resolved to their new
    content. Order preserved (send order == call order)."""
    bodies = []
    for call in bot.client.room_send.await_args_list:
        content = call.args[2] if len(call.args) >= 3 else call.kwargs.get("content", {})
        if not isinstance(content, dict):
            continue
        new_content = content.get("m.new_content")
        if isinstance(new_content, dict):
            bodies.append(new_content.get("body", ""))
        else:
            bodies.append(content.get("body", ""))
    return bodies


def _hits(bodies, needle):
    return [b for b in bodies if needle in b]


async def _run_heartbeat(bot, responses, content="heartbeat content"):
    with (
        patch("openalph.agent.stream",
              side_effect=_stream_responses(responses)),
        patch("openalph.agent.complete", new_callable=AsyncMock),
        patch("openalph.agent.execute_tool",
              side_effect=_exec_tool_stub(shell="ok")),
    ):
        await bot._run_heartbeat_turn(ROOM_A, content, turn_source="heartbeat")
    await drain(bot)


REPORT = "All systems nominal. Nothing needs attention."


# ──────────────────────────────────────────────────────────────────────
# D1 — the delivery seam fires at the hold-back
# ──────────────────────────────────────────────────────────────────────


class TestDeliverySeam:
    def test_hold_back_fires_the_delivery_seam_once(self, tmp_path):
        """D1: the held text is emitted to the transport at the hold-back.
        Without this, a non-streaming transport never sees it at all."""
        agent = _make_agent(tmp_path)
        seen, deliver = _delivery_recorder()
        result = _drive_cb(agent, "r1", [
            _text_response("First attempt text"),
            _text_response("Final answer.",
                           tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ], callbacks={"deliver_turn_text": deliver})
        assert seen == ["First attempt text"], (
            "the hold-back must emit the held text exactly once (D1)")
        assert result == "Final answer."

    def test_seam_absent_is_a_noop(self, tmp_path):
        """D1: an absent seam changes nothing — CLI/subagent/test callers that
        wire no transport keep byte-identical behavior."""
        agent = _make_agent(tmp_path)
        result = _drive_cb(agent, "r1", [
            _text_response("First attempt text"),
            _text_response("Final answer.",
                           tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ], callbacks={})
        assert result == "Final answer."
        assert agent.last_turn_declaration("r1") == "declared"
        assert len(_corrective_entries(agent, "r1")) == 1

    def test_seam_failure_is_fail_soft(self, tmp_path):
        """D1: a raising seam is logged and swallowed — a transport failure
        must never change the turn's outcome or kill the loop."""
        agent = _make_agent(tmp_path)

        async def _boom(text):
            raise RuntimeError("transport down")

        result = _drive_cb(agent, "r1", [
            _text_response("First attempt text"),
            _text_response("Final answer.",
                           tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ], callbacks={"deliver_turn_text": _boom})
        assert result == "Final answer."
        assert agent.last_turn_declaration("r1") == "declared"

    def test_seam_not_fired_when_declared_with_text(self, tmp_path):
        """D1: no hold-back, no seam — a text+declare response is an ordinary
        declared ending."""
        agent = _make_agent(tmp_path)
        seen, deliver = _delivery_recorder()
        result = _drive_cb(agent, "r1", [
            _text_response("the report",
                           tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ], callbacks={"deliver_turn_text": deliver})
        assert seen == []
        assert result == "the report"

    def test_seam_not_fired_when_tool_unregistered(self, tmp_path):
        """D1 + D4: Camp-B (no declare_done.toml) has no corrective and no
        seam — behavior is unchanged from before declare-done existed."""
        agent = _make_agent(tmp_path, declare_done=False)
        seen, deliver = _delivery_recorder()
        result = _drive_cb(agent, "r1", [_text_response("plain old ending")],
                           callbacks={"deliver_turn_text": deliver})
        assert seen == []
        assert result == "plain old ending"
        assert agent.last_turn_declaration("r1") is None

    def test_seam_fires_once_per_turn_not_per_text_end(self, tmp_path):
        """D4: the corrective is one-shot, so the seam is one-shot. A second
        consecutive text-end returns normally (no second hold-back)."""
        agent = _make_agent(tmp_path)
        seen, deliver = _delivery_recorder()
        result = _drive_cb(agent, "r1", [
            _text_response("still prose"),
            _text_response("prose again"),
        ], callbacks={"deliver_turn_text": deliver})
        assert seen == ["still prose"]
        assert result == "prose again"
        assert agent.last_turn_declaration("r1") == "undeclared"


# ──────────────────────────────────────────────────────────────────────
# D2 — the turn returns its last text
# ──────────────────────────────────────────────────────────────────────


class TestBareDeclarationReturnsHeldText:
    def test_bare_declare_returns_the_held_text(self, tmp_path):
        """D2 — THE DEFECT PIN. The model emits its report as plain text, the
        corrective fires, the model does exactly what the corrective asks (a
        bare declare_done{}), and the turn must return the REPORT — not "".
        On a non-streaming transport this is the difference between the report
        reaching the room and vanishing (workspace-im7t.61.45)."""
        agent = _make_agent(tmp_path)
        result = _drive_cb(agent, "r1", [
            _text_response(REPORT),
            _text_response("", tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ])
        assert result == REPORT
        assert agent.last_turn_declaration("r1") == "declared"

    def test_bare_declare_returns_empty_when_no_text_was_emitted(self, tmp_path):
        """D2 guard: nothing was ever said, so "" is honest — the R7
        bare-declare-is-a-legitimate-ending semantics survive."""
        agent = _make_agent(tmp_path)
        result = _drive_cb(agent, "r1", [
            _text_response("", tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ])
        assert result == ""
        assert agent.last_turn_declaration("r1") == "declared"

    def test_declare_carrying_text_still_returns_that_text(self, tmp_path):
        """D2 guard: the ratified conversational shape (reply rides in the
        same message as the declaration) is unchanged."""
        agent = _make_agent(tmp_path)
        result = _drive_cb(agent, "r1", [
            _text_response("First attempt text"),
            _text_response("Final answer.",
                           tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ])
        assert result == "Final answer."

    def test_held_text_superseded_by_further_work_does_not_stand(self, tmp_path):
        """D2 guard — the over-reach guard. A hold-back that the model follows
        with real work is an ANNOUNCEMENT, not a report: it was delivered (D1)
        but must NOT be returned as the turn's final reply when the turn later
        ends on a bare declaration. Otherwise a stale "I'm going to do X"
        becomes the run's result text."""
        agent = _make_agent(tmp_path)
        result = _drive_cb(agent, "r1", [
            _text_response("I'm going to check the logs."),
            _text_response("", tool_calls=[_shell_call()],
                           stop_reason="tool_use"),
            _text_response("", tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ])
        assert result == ""
        assert agent.last_turn_declaration("r1") == "declared"

    def test_camp_b_ending_unchanged(self, tmp_path):
        """D2 + D4: with the tool unregistered there is no corrective and the
        text returns immediately — pre-declare-done behavior, untouched."""
        agent = _make_agent(tmp_path, declare_done=False)
        result = _drive_cb(agent, "r1", [_text_response("plain old ending")])
        assert result == "plain old ending"


# ──────────────────────────────────────────────────────────────────────
# D5 — the corrective is delivery-truthful
# ──────────────────────────────────────────────────────────────────────


class TestCorrectiveWording:
    def test_corrective_is_transport_truthful(self, tmp_path):
        """D5 — SPEC REFINED by the 2026-10-02 adversarial audit (findings E,
        3/3 auditors). The original D5 required the corrective to state the text
        "was delivered" — but on exec/`cli chat` the D1 seam is deliberately
        unwired, so nothing has been delivered when the corrective fires, and on
        wired transports the send may have failed and been swallowed. One fixed
        string therefore cannot be truthful on all transports, and a false
        "delivered" claim is load-bearing: it is what licenses the model to omit
        its report. The spec is corrected to a claim that is true everywhere —
        the text is the turn's reply and is kept.

        This pin asserts the corrected spec: names the tool, forbids repetition,
        and does NOT claim delivery."""
        agent = _make_agent(tmp_path)
        _drive_cb(agent, "r1", [
            _text_response(REPORT),
            _text_response("", tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ])
        entries = _corrective_entries(agent, "r1")
        assert len(entries) == 1
        body = str(entries[0]["content"]).lower()
        assert "declare_done" in body, (
            "the corrective must still name the terminal tool")
        assert "repeat" in body, (
            "the corrective must tell the model not to repeat the text (D5)")
        assert "delivered" not in body, (
            "the corrective must NOT claim delivery — false on exec/chat and "
            "after a swallowed seam failure (D5 as refined, finding E)")


# ──────────────────────────────────────────────────────────────────────
# D1/D3 — transport wiring and the shared no-double-send predicate
# ──────────────────────────────────────────────────────────────────────


class TestTransportWiring:
    def test_live_path_does_not_wire_the_delivery_seam(self, tmp_path):
        """D1: the live interactive path is excluded by design —
        StreamingDelivery already delivered the deltas, and wiring the seam
        there would double-post. The absence is the design statement."""
        bot, _agent = _make_heartbeat_bot(tmp_path)
        callbacks = bot._build_agent_callbacks(ROOM_A, None)
        assert "deliver_turn_text" not in callbacks, (
            "the streaming (live) path must NOT wire the delivery seam (D1)")


# ──────────────────────────────────────────────────────────────────────
# The headline real-path pins — REAL Agent + REAL funnel
# ──────────────────────────────────────────────────────────────────────


class TestHeartbeatDeliveryRealPath:
    @pytest.mark.asyncio
    async def test_heartbeat_delivers_the_held_report(self, tmp_path):
        """The defect, end to end, on the transport that actually loses it:
        a heartbeat turn whose model emits its report as plain text and then
        answers the corrective with a bare declare_done{} must PUT THE REPORT
        IN THE ROOM — exactly once, with the turn booked 'declared'.

        This is the pin whose absence let the defect ship: the .350.2 suite
        pinned the live (streaming) path, where the bug is invisible."""
        bot, agent = _make_heartbeat_bot(tmp_path)
        await _run_heartbeat(bot, [
            _text_response(REPORT),
            _text_response("", tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ])
        bodies = _sent_bodies(bot)
        assert len(_hits(bodies, "All systems nominal")) == 1, (
            f"the held report must reach the room exactly once; sent bodies={bodies!r}")
        assert agent.last_turn_declaration(ROOM_A) == "declared"

    @pytest.mark.asyncio
    async def test_heartbeat_does_not_double_send_when_declare_repeats_text(self, tmp_path):
        """D3: if the model re-emits its text in the declaration (the old
        prompt-level workaround), the transport must not post it twice — the
        shared delivered-text predicate suppresses the turn-end send."""
        bot, agent = _make_heartbeat_bot(tmp_path)
        await _run_heartbeat(bot, [
            _text_response(REPORT),
            _text_response(REPORT, tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ])
        bodies = _sent_bodies(bot)
        assert len(_hits(bodies, "All systems nominal")) == 1, (
            f"repeating the report in the declaration must not double-post; sent bodies={bodies!r}")
        assert agent.last_turn_declaration(ROOM_A) == "declared"

    @pytest.mark.asyncio
    async def test_heartbeat_normal_declared_turn_sends_exactly_once(self, tmp_path):
        """D3 regression guard: the ordinary heartbeat shape (reply text in
        the same message as the declaration) still delivers exactly once."""
        bot, agent = _make_heartbeat_bot(tmp_path)
        await _run_heartbeat(bot, [
            _text_response(REPORT, tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ])
        bodies = _sent_bodies(bot)
        assert len(_hits(bodies, "All systems nominal")) == 1, (
            f"a declared heartbeat turn must deliver its report once; sent bodies={bodies!r}")
        assert agent.last_turn_declaration(ROOM_A) == "declared"


# ──────────────────────────────────────────────────────────────────────
# REMEDIATION PINS — 2026-10-02 adversarial audit of 6a090dab
# (tmp/code-audit/declare-done-delivery/AUDIT-REPORT.md, findings A / B / G)
# ──────────────────────────────────────────────────────────────────────


class TestAuditFindingA_RetryDoubleDelivery:
    @pytest.mark.asyncio
    async def test_heartbeat_retry_does_not_duplicate_the_delivered_report(self, tmp_path):
        """FINDING A (GLM HIGH, Kimi HIGH — both reproduced it; orchestrator-
        verified at matrix.py:2499 and 2485-2493). The empty-response retry
        re-runs handle_input with the SAME callbacks dict — hence the same live
        seam and the same BulkDelivery — and sends the retry text
        UNCONDITIONALLY, with no _needs_send gate; the retry turn also carries a
        fresh corrective, so its hold-back posts the text a second time.

        The room must see the report exactly ONCE across the whole two-call
        funnel run. Fix-shape agnostic: passes either by skipping the retry when
        the turn already delivered, or by gating the retry send on the shared
        predicate — both leave exactly one copy."""
        bot, agent = _make_heartbeat_bot(tmp_path)
        await _run_heartbeat(bot, [
            _text_response(REPORT),   # hold-back → the D1 seam delivers REPORT
            _text_response(""),       # silent end → 'undeclared' → funnel retries
            _text_response(REPORT, tool_calls=[_declare_call()],
                           stop_reason="tool_use"),  # the retry turn
        ])
        bodies = _sent_bodies(bot)
        assert len(_hits(bodies, "All systems nominal")) == 1, (
            "the report must reach the room exactly once across the "
            f"empty-response retry (finding A); sent bodies={bodies!r}")


class TestAuditFindingB_AdjacencyIsNotIndexParity:
    def test_held_text_survives_a_harness_internal_retry(self, tmp_path):
        """FINDING B (3/3 auditors; orchestrator-verified at agent.py:2550).
        The D2 substitution was gated on `_hold_iteration == iteration - 1`, but
        the empty-max_tokens thinking recovery consumes an iteration with NO
        superseding work — it executes no tool and appends no work to history.
        A bare declaration after it must still return the held report: the rule
        is "no WORK since the hold-back", not "the immediately next iteration".

        The overlay-validation retry (agent.py:2993) is the same class and is
        covered by the same mechanism — the re-audit should confirm it."""
        agent = _make_agent(tmp_path)
        result = _drive_cb(agent, "r1", [
            _text_response(REPORT),                        # hold-back
            _text_response("", stop_reason="max_tokens"),  # thinking-burn → retry
            _text_response("", tool_calls=[_declare_call()],
                           stop_reason="tool_use"),         # bare declare
        ], thinking="high")
        assert result == REPORT, (
            "a harness-internal retry between the hold-back and the declaration "
            "must not supersede the held report (finding B)")
        assert agent.last_turn_declaration("r1") == "declared"

    def test_real_tool_work_still_supersedes_the_held_text(self, tmp_path):
        """FINDING B regression guard: the fix keys on WORK, not on the absence
        of a declaration. A hold-back followed by a real tool call is an
        ANNOUNCEMENT, not a report — a later bare declaration must NOT return
        it. (The original intent of the adjacency rule, which the fix must
        preserve; see also the pre-existing pin of the same shape.)"""
        agent = _make_agent(tmp_path)
        result = _drive_cb(agent, "r1", [
            _text_response("I'm going to check the logs."),
            _text_response("", tool_calls=[_shell_call()],
                           stop_reason="tool_use"),
            _text_response("", tool_calls=[_declare_call()],
                           stop_reason="tool_use"),
        ])
        assert result == "", (
            "real tool work between the hold-back and the declaration must "
            "still supersede the held text (finding B guard)")
        assert agent.last_turn_declaration("r1") == "declared"


class TestAuditFindingG_LiveCallSite:
    @pytest.mark.asyncio
    async def test_live_funnel_call_site_does_not_wire_the_seam(self, tmp_path):
        """FINDING G (Kimi, Qwen). The original live-path pin asserted the
        BUILDER default — `_build_agent_callbacks(ROOM_A, None)` called directly
        — and never exercised `_process_message`'s real call site, so a
        regression that started passing a BulkDelivery on the live path would
        have passed it while double-posting every held report in live rooms.
        This pin captures the callbacks the live funnel actually hands to
        handle_input."""
        bot, agent = _make_heartbeat_bot(tmp_path)
        captured = {}

        async def _spy(text, room_id="_default", **kw):
            captured.update(kw)
            return "ok"

        agent.handle_input = _spy
        await bot._process_message(
            make_room(ROOM_A), make_event(USER, "hi", "$e-live"), "hi")
        await drain(bot)
        assert "deliver_turn_text" not in (captured.get("callbacks") or {}), (
            "the live (streaming) path must not wire the delivery seam at its "
            "real call site (finding G)")
