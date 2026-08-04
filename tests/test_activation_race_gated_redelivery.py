"""Round-3 regression tests: concurrent activation race + gated/media redelivery.

Pass-2 adversarial review of the Round-2 wedge fixes found two MAJOR defects and
one MINOR one.  These tests pin all three.  They are written to FAIL against the
Round-2 tree and PASS after the Round-3 fix.

R3-1 (MAJOR) — CONCURRENT ACTIVATION RACE
    `_activate_room` runs BEFORE `_session_locks[room_id]` is acquired, at both
    live-event call sites (`_process_message`'s lazy wake and
    `_handle_room_message`'s pre-activation).  `_activate_room` is network-bound
    (it awaits `client.room_messages`) and only adds the room to `_active_rooms`
    at the very END, so two near-simultaneous events on a dormant room BOTH
    observe the room as inactive and BOTH activate:

      * both run gap-fill, so recent history is appended to the JSONL twice;
      * each assigns a FRESH `_known_event_ids[room_id]` set, clobbering the
        other's membership and letting real redeliveries slip past the gate;
      * one caller's live event is back-filled by the other's gap-fill, so its
        own dispatch is then eaten by the redelivery gate and its turn never
        runs.

    Fix: a per-room activation mutex with a double-checked inactivity test.
    Lock order is `_activate_locks[room] -> (released) -> _session_locks[room]`;
    the activation lock is leaf-level and is never held across the session-lock
    acquisition, so the two cannot deadlock.

R3-2 (MAJOR) — GATED ROOMS BYPASS THE REDELIVERY GATE
    The Round-2 gate in `_process_message` is `not gated and ...`, because a
    gated room's trigger is buffered (and claimed) by `_handle_room_message`
    BEFORE `_process_message` runs, so an undifferentiated membership check
    there would suppress every gated turn.  The consequence was that gated
    rooms — production 3+ member rooms — had NO redelivery protection at all: a
    sync-loop reconnect duplicate produced a second buffer entry, a second model
    turn and a second assistant reply.

    Fix: move the decision UPSTREAM for gated rooms.  `_handle_room_message`
    checks membership BEFORE the gating buffer append: already-known means
    redelivery (INFO log, skip entirely — no buffer append, no dispatch);
    first-seen buffers, claims and dispatches exactly as before.

R3-3 (MINOR) — MEDIA REDELIVERY DOWNLOADS BEFORE THE GATE
    `_handle_media_message` downloads and writes the file before
    `_process_message` can gate the redelivery, so every duplicate delivery of a
    media event re-ran an HTTP download and rewrote the file.  Fix: the same
    upstream membership check at the top of the media handler.

Like tests/test_gapfill_trigger_dedup.py these use a REAL SessionLog, so entry
counts are on-disk facts rather than mock bookkeeping.
"""

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.matrix import MatrixBot
from openalph.session import SessionLog

ROOM_ID = "!race:matrix.local"
AGENT_USER = "@watson:matrix.local"
USER = "@alice:matrix.local"

# Bound every "wait for the racing tasks" await so a regression fails loudly
# instead of hanging the suite.
JOIN_TIMEOUT = 5.0


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


def make_agent_config(workspace: Path, **kwargs):
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
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_event(sender, body, event_id, ts=1_000_000, mention=False):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = ts
    content = {"msgtype": "m.text", "body": body}
    if mention:
        content["m.mentions"] = {"user_ids": [AGENT_USER]}
    event.source = {"content": content}
    return event


def make_media_event(sender, body, event_id, ts=1_000_000,
                     url="mxc://matrix.local/abcdef", mimetype="image/jpeg",
                     size=1024):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.url = url
    event.server_timestamp = ts
    event.source = {
        "content": {
            "body": body,
            "url": url,
            "info": {"mimetype": mimetype, "size": size},
        }
    }
    return event


def make_room(room_id=ROOM_ID, member_count=2):
    """member_count >= 3 -> gated (mention required); 2 -> ungated."""
    room = MagicMock()
    room.room_id = room_id
    room.name = "Race Room"
    room.display_name = "Race Room"
    room.joined_count = member_count
    room.users = {f"@u{i}:matrix.local": MagicMock() for i in range(member_count)}
    return room


def make_bot(tmp_path):
    agent = MagicMock()
    agent.handle_input = AsyncMock(return_value="ok")
    agent._rooms = {}
    agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))
    agent.config = make_agent_config(tmp_path)
    # Keep the stall watchdog disarmed: these tests are about activation and
    # redelivery, not about the watchdog.
    agent.config.turn_stall_timeout_seconds = 0
    agent.restore_usage = MagicMock()
    agent.rehydrate_reminders = MagicMock()
    agent.status = MagicMock(return_value={
        "context_pct": 1, "context_tokens": 10, "context_max": 200000,
    })
    agent.last_stop_reason = MagicMock(return_value="end_turn")
    agent.last_turn_usage = MagicMock(return_value={})
    agent._room_tool_counts = {}
    agent._advisor_uses = {}
    agent._room_models = {}

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
    bot._active_rooms = set()          # dormant -> lazy wake fires
    bot._halted_rooms = set()
    bot._room_thinking = {}
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
    bot.session_log = SessionLog(tmp_path, AGENT_USER)
    return bot, agent


def seed_prior_history(session_log):
    """Gap-fill only runs when the JSONL already has entries."""
    session_log.append(
        role="user", sender=USER, room=ROOM_ID,
        event_id="$prior-user", content="an earlier message",
    )
    session_log.append(
        role="assistant", sender=AGENT_USER, room=ROOM_ID,
        event_id=None, content="an earlier reply",
    )


def gapfill_page(messages):
    page = MagicMock()
    page.chunk = messages
    page.end = ""
    return page


def user_entries_for(session_log, event_id, room_id=ROOM_ID):
    return [
        e for e in session_log.read(room_id)
        if e.get("role") == "user" and e.get("event_id") == event_id
    ]


def system_events(session_log, event_name, room_id=ROOM_ID):
    return [
        e for e in session_log.read(room_id)
        if e.get("role") == "system" and e.get("event") == event_name
    ]


def count_activations(bot):
    """Wrap the REAL `_activate_room` in a counter (never replace it — the test
    depends on genuine hydration and gap-fill running)."""
    calls = []
    original = bot._activate_room

    async def _counting(*args, **kwargs):
        calls.append((args, kwargs))
        return await original(*args, **kwargs)

    bot._activate_room = _counting
    return calls


def gate_gap_fill(bot, page, gate: asyncio.Event):
    """Park every `room_messages` call on `gate` so activation can be held
    open deterministically while a second event races in."""
    async def _room_messages(*args, **kwargs):
        await gate.wait()
        return page

    bot.client.room_messages = AsyncMock(side_effect=_room_messages)
    return bot.client.room_messages


async def settle(times=25):
    """Yield to the loop enough times for every runnable task to reach its next
    real await point (the parked gap-fill, or the activation lock)."""
    for _ in range(times):
        await asyncio.sleep(0)


async def drain(bot):
    """Await every background task, including ones spawned mid-drain."""
    for _ in range(10):
        pending = [t for t in list(getattr(bot, "_background_tasks", None) or ())
                   if not t.done()]
        if not pending:
            return
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True), timeout=JOIN_TIMEOUT,
        )


# ── R3-1: concurrent activation race ─────────────────────────────────────────

class TestConcurrentActivationRace:
    """Two live events land on a DORMANT room at once (batched wake, reconnect
    replay, a user pasting two messages).  Exactly one activation may run."""

    @pytest.mark.asyncio
    async def test_two_concurrent_first_messages_activate_once(self, tmp_path):
        """The production text path: `_handle_room_message` pre-activates."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        activations = count_activations(bot)

        e1 = make_event(USER, "first", "$E1", ts=2_000)
        e2 = make_event(USER, "second", "$E2", ts=3_000)
        gate = asyncio.Event()
        # The server back-fetch sees both live events plus the known prior one.
        room_messages = gate_gap_fill(bot, gapfill_page([
            make_event(USER, "second", "$E2", ts=3_000),
            make_event(USER, "first", "$E1", ts=2_000),
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]), gate)

        room = make_room()
        t1 = asyncio.create_task(bot._handle_room_message(room, e1))
        t2 = asyncio.create_task(bot._handle_room_message(room, e2))
        await settle()

        # Mid-flight: the first caller is parked inside gap-fill and the second
        # MUST be queued on the activation lock, not inside a second gap-fill.
        # Captured, not asserted, so a regression cannot strand parked tasks.
        in_flight_gap_fills = room_messages.await_count

        gate.set()
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=JOIN_TIMEOUT)
        await drain(bot)

        assert in_flight_gap_fills == 1, (
            f"only ONE activation may be in flight for a dormant room; "
            f"{in_flight_gap_fills} concurrent gap-fills were started"
        )
        assert len(activations) == 1, (
            f"_activate_room must run exactly once for a dormant room hit by two "
            f"concurrent live events; ran {len(activations)} times"
        )
        assert len(system_events(bot.session_log, "session_resume")) == 1, (
            "a duplicated activation writes a second session_resume marker and a "
            "second copy of the gap-filled history"
        )
        for eid in ("$E1", "$E2"):
            assert len(user_entries_for(bot.session_log, eid)) == 1, (
                f"{eid} must be persisted exactly once; got "
                f"{len(user_entries_for(bot.session_log, eid))} copies"
            )
        assert agent.handle_input.await_count == 2, (
            "both distinct live events must run their own turn — a racing "
            "gap-fill that back-fills the other caller's event makes its "
            f"dispatch look like a redelivery; got {agent.handle_input.await_count}"
        )
        known = bot._known_event_ids.get(ROOM_ID, set())
        assert {"$E1", "$E2", "$prior-user"} <= known, (
            "the membership set must not be clobbered by a second activation; "
            f"got {sorted(known)}"
        )

    @pytest.mark.asyncio
    async def test_two_concurrent_lazy_wakes_in_process_message(self, tmp_path):
        """The other call site: `_process_message`'s own lazy wake (the media
        path and any direct dispatch reach activation through it)."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        activations = count_activations(bot)

        gate = asyncio.Event()
        room_messages = gate_gap_fill(bot, gapfill_page([
            make_event(USER, "second", "$E2", ts=3_000),
            make_event(USER, "first", "$E1", ts=2_000),
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]), gate)

        room = make_room()
        t1 = asyncio.create_task(bot._process_message(
            room, make_event(USER, "first", "$E1", ts=2_000), "first"))
        t2 = asyncio.create_task(bot._process_message(
            room, make_event(USER, "second", "$E2", ts=3_000), "second"))
        await settle()

        in_flight_gap_fills = room_messages.await_count

        gate.set()
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=JOIN_TIMEOUT)
        await drain(bot)

        assert in_flight_gap_fills == 1, (
            f"the lazy wake inside _process_message must also be mutually "
            f"exclusive; {in_flight_gap_fills} concurrent gap-fills ran"
        )
        assert len(activations) == 1
        for eid in ("$E1", "$E2"):
            assert len(user_entries_for(bot.session_log, eid)) == 1
        assert agent.handle_input.await_count == 2

    @pytest.mark.asyncio
    async def test_second_message_waits_then_skips_activation(self, tmp_path):
        """Activation and an immediately-following message, interleaved
        deterministically: the second caller must observe the room ACTIVE after
        the lock hand-off and skip activation via the double check."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        activations = count_activations(bot)

        gate = asyncio.Event()
        gate_gap_fill(bot, gapfill_page([
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]), gate)

        room = make_room()
        t1 = asyncio.create_task(bot._handle_room_message(
            room, make_event(USER, "first", "$E1", ts=2_000)))
        await settle()
        active_while_activating = ROOM_ID in bot._active_rooms

        # Second live event arrives WHILE activation is in flight.
        t2 = asyncio.create_task(bot._handle_room_message(
            room, make_event(USER, "second", "$E2", ts=3_000)))
        await settle()
        activations_in_flight = len(activations)

        gate.set()
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=JOIN_TIMEOUT)
        await drain(bot)

        assert active_while_activating is False, (
            "activation was still parked in gap-fill — the room must not be "
            "marked active yet"
        )
        assert activations_in_flight == 1, (
            "the second caller must block on the activation lock rather than "
            f"starting its own activation; {activations_in_flight} activations started"
        )
        assert len(activations) == 1
        assert ROOM_ID in bot._active_rooms
        assert len(user_entries_for(bot.session_log, "$E1")) == 1
        assert len(user_entries_for(bot.session_log, "$E2")) == 1
        assert agent.handle_input.await_count == 2

    @pytest.mark.asyncio
    async def test_activation_lock_released_when_activation_raises(self, tmp_path):
        """A failed activation must not leave the room's activation lock held —
        the next message would wedge on it forever."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)

        boom = RuntimeError("activation exploded")

        async def _explode(*args, **kwargs):
            raise boom

        bot._activate_room = _explode
        room = make_room()

        with pytest.raises(RuntimeError):
            await bot._handle_room_message(
                room, make_event(USER, "first", "$E1", ts=2_000))

        lock = getattr(bot, "_activate_locks", {}).get(ROOM_ID)
        assert lock is None or not lock.locked(), (
            "a raising activation must still release the per-room activation lock"
        )

        # And the room can still be activated afterwards.
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]))
        del bot._activate_room       # restore the real bound method
        await bot._handle_room_message(
            room, make_event(USER, "second", "$E2", ts=3_000))
        await drain(bot)
        assert ROOM_ID in bot._active_rooms

    @pytest.mark.asyncio
    async def test_activation_lock_not_held_across_the_session_lock(self, tmp_path):
        """Lock-order invariant: the activation lock is leaf-level and is
        released BEFORE `_session_locks[room_id]` is acquired, so the two can
        never deadlock."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]))

        observed = {}

        async def _inspect(*args, **kwargs):
            session_lock = bot._session_locks.get(ROOM_ID)
            act_lock = getattr(bot, "_activate_locks", {}).get(ROOM_ID)
            observed["session_held"] = bool(session_lock and session_lock.locked())
            observed["activation_held"] = bool(act_lock and act_lock.locked())
            return "ok"

        agent.handle_input = AsyncMock(side_effect=_inspect)

        await bot._handle_room_message(
            make_room(), make_event(USER, "first", "$E1", ts=2_000))
        await drain(bot)

        assert observed.get("session_held") is True, "sanity: the turn holds the session lock"
        assert observed.get("activation_held") is False, (
            "the activation lock must NOT still be held while the session lock "
            "is held — that ordering is what makes the two locks deadlock-free"
        )


# ── R3-2: gated rooms honour the redelivery gate ─────────────────────────────

class TestGatedRedeliveryGate:
    """Gated (3+ member) rooms are the production case, and Round-2 left them
    with no redelivery protection at all."""

    @pytest.mark.asyncio
    async def test_gated_first_delivery_processes_normally(self, tmp_path):
        """Guard rail: the mention path must be completely intact."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))

        await bot._handle_room_message(
            make_room(member_count=3),
            make_event(USER, f"{AGENT_USER} hello there", "$G1", ts=2_000,
                       mention=True),
        )
        await drain(bot)

        assert len(user_entries_for(bot.session_log, "$G1")) == 1, (
            "the gated buffer append must still happen on first delivery"
        )
        assert agent.handle_input.await_count == 1, (
            "a first-seen gated mention must still run its turn"
        )
        assert agent.handle_input.await_args.kwargs["append_user"] is False
        assert "$G1" in bot._known_event_ids.get(ROOM_ID, set())

    @pytest.mark.asyncio
    async def test_gated_redelivery_no_second_buffer_no_second_turn(self, tmp_path):
        """The MAJOR: a redelivered gated mention must be inert."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        room = make_room(member_count=3)

        for _ in range(2):
            await bot._handle_room_message(
                room,
                make_event(USER, f"{AGENT_USER} hello there", "$G1", ts=2_000,
                           mention=True),
            )
            await drain(bot)

        assert len(user_entries_for(bot.session_log, "$G1")) == 1, (
            "a redelivered gated event must NOT produce a second buffer entry; "
            f"got {len(user_entries_for(bot.session_log, '$G1'))}"
        )
        assert agent.handle_input.await_count == 1, (
            "a redelivered gated event must NOT run a second model turn; ran "
            f"{agent.handle_input.await_count} turns"
        )
        replies = [c for c in bot.send.await_args_list if "ok" in str(c.args)]
        assert len(replies) == 1, (
            f"a redelivered gated event must not deliver a second reply; got {replies}"
        )

    @pytest.mark.asyncio
    async def test_gated_redelivery_when_room_already_active(self, tmp_path):
        """Same invariant with the room already active (sync-loop reconnect
        during a live session, not a restart)."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot._active_rooms.add(ROOM_ID)
        bot._known_event_ids[ROOM_ID] = {"$prior-user"}
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        room = make_room(member_count=3)
        ev = make_event(USER, f"{AGENT_USER} status?", "$G2", ts=2_000, mention=True)

        await bot._handle_room_message(room, ev)
        await drain(bot)
        await bot._handle_room_message(room, ev)
        await drain(bot)

        assert len(user_entries_for(bot.session_log, "$G2")) == 1
        assert agent.handle_input.await_count == 1

    @pytest.mark.asyncio
    async def test_gated_non_mentioned_redelivery_stays_inert(self, tmp_path):
        """A non-mentioned gated message is buffered once and never re-buffered:
        the duplicate would otherwise show up twice in hydrated context."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        room = make_room(member_count=3)
        ev = make_event(USER, "just chatting", "$G3", ts=2_000)

        await bot._handle_room_message(room, ev)
        await drain(bot)
        await bot._handle_room_message(room, ev)
        await drain(bot)

        assert len(user_entries_for(bot.session_log, "$G3")) == 1, (
            "a redelivered non-mentioned gated message must not be buffered twice"
        )
        assert agent.handle_input.await_count == 0, (
            "a non-mentioned gated message must never run a turn"
        )

    @pytest.mark.asyncio
    async def test_gated_redelivery_logged_at_info(self, tmp_path, caplog):
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        room = make_room(member_count=3)
        ev = make_event(USER, f"{AGENT_USER} hi", "$G4", ts=2_000, mention=True)

        await bot._handle_room_message(room, ev)
        await drain(bot)
        with caplog.at_level("INFO", logger="openalph.matrix"):
            await bot._handle_room_message(room, ev)
            await drain(bot)

        assert any("$G4" in r.getMessage() for r in caplog.records), (
            "a gated redelivery must be logged at INFO; records: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_gated_distinct_events_still_all_processed(self, tmp_path):
        """Over-suppression guard: distinct gated events must be unaffected."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        room = make_room(member_count=3)

        for i in range(3):
            await bot._handle_room_message(
                room,
                make_event(USER, f"{AGENT_USER} msg {i}", f"$D{i}", ts=2_000 + i,
                           mention=True),
            )
            await drain(bot)

        assert agent.handle_input.await_count == 3
        for i in range(3):
            assert len(user_entries_for(bot.session_log, f"$D{i}")) == 1

    @pytest.mark.asyncio
    async def test_gated_event_without_event_id_still_processed(self, tmp_path):
        """Falsy event_ids never match membership, so id-less gated events (test
        doubles, malformed events) must not be mistaken for redeliveries."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        room = make_room(member_count=3)

        for _ in range(2):
            await bot._handle_room_message(
                room,
                make_event(USER, f"{AGENT_USER} hi", None, ts=2_000, mention=True),
            )
            await drain(bot)

        assert agent.handle_input.await_count == 2, (
            "events without an event_id must keep the pre-fix behaviour"
        )


# ── R3-3: media redelivery is gated before the download ──────────────────────

class TestMediaRedeliveryGate:

    @staticmethod
    def _media_dir_hash(event_id):
        return hashlib.sha256(event_id.encode()).hexdigest()[:16]

    def _download_mock(self, body=b"jpegbytes"):
        download = MagicMock()
        download.body = body
        download.content_type = "image/jpeg"
        download.filename = None
        download.__class__.__name__ = "MemoryDownloadResponse"
        return AsyncMock(return_value=download)

    @pytest.mark.asyncio
    async def test_redelivered_media_performs_zero_downloads(self, tmp_path):
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        bot.client.download = self._download_mock()
        room = make_room()
        ev = make_media_event(USER, "photo.jpg", "$M1", ts=2_000)

        await bot._handle_media_message(room, ev)
        await drain(bot)
        assert bot.client.download.await_count == 1, "sanity: first delivery downloads"

        await bot._handle_media_message(room, ev)
        await drain(bot)

        assert bot.client.download.await_count == 1, (
            "a redelivered media event must perform ZERO further downloads; got "
            f"{bot.client.download.await_count} total"
        )
        assert len(user_entries_for(bot.session_log, "$M1")) == 1, (
            "a redelivered media event must not be appended again"
        )
        assert agent.handle_input.await_count == 1, (
            "a redelivered media event must not run a second turn"
        )

    @pytest.mark.asyncio
    async def test_media_redelivery_logged_at_info(self, tmp_path, caplog):
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        bot.client.download = self._download_mock()
        room = make_room()
        ev = make_media_event(USER, "photo.jpg", "$M2", ts=2_000)

        await bot._handle_media_message(room, ev)
        await drain(bot)
        with caplog.at_level("INFO", logger="openalph.matrix"):
            await bot._handle_media_message(room, ev)
            await drain(bot)

        assert any("$M2" in r.getMessage() for r in caplog.records), (
            "a media redelivery must be logged at INFO; records: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_first_media_delivery_unaffected(self, tmp_path):
        """Guard rail: the canonical media tag path is untouched."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        bot.client.download = self._download_mock()

        await bot._handle_media_message(
            make_room(), make_media_event(USER, "photo.jpg", "$M3", ts=2_000))
        await drain(bot)

        rel = f"media/{self._media_dir_hash('$M3')}/photo.jpg"
        assert (tmp_path / rel).exists(), "the media file must still be downloaded"
        entries = user_entries_for(bot.session_log, "$M3")
        assert len(entries) == 1
        assert rel in entries[0]["content"], (
            f"the canonical media tag must still be persisted; got {entries[0]['content']!r}"
        )
        assert agent.handle_input.await_count == 1

    @pytest.mark.asyncio
    async def test_media_event_without_event_id_not_gated(self, tmp_path):
        """Falsy ids must never match membership."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        bot.client.download = self._download_mock()
        room = make_room()

        for _ in range(2):
            await bot._handle_media_message(
                room, make_media_event(USER, "photo.jpg", None, ts=2_000))
            await drain(bot)

        assert bot.client.download.await_count == 2
