"""Regression tests for the lazy-wake gap-fill double-append bug (Fix A) — Round 2.

Bug (production room wedge, 2026-08-03 RCA):
    _process_message performs a lazy wake when the room is not yet active:
        await self._activate_room(room_id, ...)
    _activate_room's gap-fill back-fetches recent messages from the server and
    appends every message not already in the local session JSONL -- which
    INCLUDES the triggering live message itself (it arrived seconds ago and was
    never persisted).  Control returns to _process_message which, for ungated
    rooms, appends the SAME message to the JSONL again, and calls handle_input
    with append_user=True, putting a third copy into in-memory history.

    provider._dedup_trailing_user() masks the wire-payload symptom, but the
    JSONL is left permanently duplicated.

Round-1 fix (REMOVED — see the review findings) compared the trigger's event_id
to `SessionLog.last_event_id(room_id)` (a "tail equality" guard). Three defects:
  * batched wakes: with a chunk of [E2, E1] both are back-filled, E2 becomes the
    tail, so E1 misses the guard and is appended AGAIN — then E2 does too;
  * redelivery was not turn-idempotent: a tail match still ran the model with
    append_user=False, so a duplicate event produced a second assistant reply;
  * media triggers lost their canonical `[media: …]` tag (gap-fill serializes the
    raw Matrix body — usually a bare filename), and with append_user=False the
    canonical tag never reached hydrated history either, so the model saw only
    "photo.jpg" and could not locate the downloaded file;
  * plus an O(session-size) blocking whole-file scan on EVERY ungated message.

Round-2 fix (what these tests pin):
  (a) TRIGGER-AWARE ACTIVATION: `_activate_room(..., trigger_event_id=,
      trigger_ts=)` excludes the triggering live event AND anything newer (the
      live sync loop dispatches those separately) from back-fill persistence.
      The trigger is then persisted exactly once by _process_message's normal
      ungated append — which means media triggers keep their canonical
      `[media: media/<hash>/<file> (<mime>, <size>)]` tag and are hydrated into
      handle_input's history.
  (b) EVENT-ID MEMBERSHIP: a per-room known-event-ID set, hydrated ONCE from the
      JSONL at activation and updated on every append carrying an event_id. On
      _process_message entry (post-lock) a trigger already in the set is a
      REDELIVERY: log at INFO and return WITHOUT invoking the agent.
  (c) O(1) per message: no `last_event_id()` full-file scan on the hot path.

These tests use a REAL SessionLog (not a MagicMock) so the JSONL entry counts
are genuine on-disk facts rather than mock-call bookkeeping.
"""

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.matrix import MatrixBot
from openalph.session import SessionLog

ROOM_ID = "!wedge:matrix.local"
AGENT_USER = "@watson:matrix.local"
USER = "@alice:matrix.local"
TRIGGER_EVENT_ID = "$trigger-live-msg"
TRIGGER_BODY = "the live message that wedged the room"


# ── Fixtures (mirroring tests/test_session.py conventions) ────────────────────

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


def make_event(sender, body, event_id, ts=1_000_000):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = ts
    event.source = {"content": {"msgtype": "m.text", "body": body}}
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


def make_room(room_id, member_count=2):
    """A 2-member room is UNGATED (is_gated() -> False), which is the buggy path."""
    room = MagicMock()
    room.room_id = room_id
    room.name = "Wedge Room"
    room.display_name = "Wedge Room"
    room.joined_count = member_count
    room.users = {f"@u{i}:matrix.local": MagicMock() for i in range(member_count)}
    return room


def make_bot(tmp_path):
    """MatrixBot via __new__ with a REAL SessionLog, per test_session.py."""
    agent = MagicMock()
    agent.handle_input = AsyncMock(return_value="ok")
    # A real per-room history list, so hydration assertions are genuine.
    agent._rooms = {}
    agent.history = MagicMock(side_effect=lambda rid: agent._rooms.setdefault(rid, []))
    agent.config = make_agent_config(tmp_path)
    # turn_stall_timeout_seconds must not be a MagicMock: keep the watchdog off
    # so these Fix-A tests exercise only the gap-fill path.
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
    bot._active_rooms = set()          # room NOT active -> lazy wake fires
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
    """A room_messages response page: newest-first chunk, no further pages."""
    page = MagicMock()
    page.chunk = messages
    page.end = ""
    return page


def user_entries_for(session_log, event_id):
    return [
        e for e in session_log.read(ROOM_ID)
        if e.get("role") == "user" and e.get("event_id") == event_id
    ]


def user_entries(session_log):
    return [e for e in session_log.read(ROOM_ID) if e.get("role") == "user"]


def assistant_entries(session_log):
    return [e for e in session_log.read(ROOM_ID) if e.get("role") == "assistant"]


async def drain(bot):
    if getattr(bot, "_background_tasks", None):
        await asyncio.gather(*list(bot._background_tasks), return_exceptions=True)


# ── A1: gap-fill would otherwise back-fetch the triggering event ─────────────

class TestGapFillPersistedTrigger:
    """A1 — the core regression. RED against unpatched code."""

    @pytest.mark.asyncio
    async def test_trigger_appears_exactly_once_in_jsonl(self, tmp_path):
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)

        # The server's back-fetch returns the triggering live message (newest,
        # not yet persisted) followed by the known prior message (overlap).
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]))

        await bot._process_message(
            make_room(ROOM_ID),
            make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
            TRIGGER_BODY,
        )

        dupes = user_entries_for(bot.session_log, TRIGGER_EVENT_ID)
        assert len(dupes) == 1, (
            f"Triggering event must appear exactly ONCE in the session JSONL; "
            f"found {len(dupes)} entries: {[e.get('content') for e in dupes]}"
        )

    @pytest.mark.asyncio
    async def test_trigger_persisted_by_process_message_with_append_user_true(self, tmp_path):
        """Round-2 change of contract: activation EXCLUDES the trigger from
        back-fill, so _process_message performs the one canonical append and
        handle_input still appends the user turn to history (append_user=True).

        The Round-1 guard instead let gap-fill persist the trigger and passed
        append_user=False — which is what lost media triggers' canonical tag.
        """
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)

        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]))

        await bot._process_message(
            make_room(ROOM_ID),
            make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
            TRIGGER_BODY,
        )

        assert agent.handle_input.await_count == 1
        assert agent.handle_input.await_args.kwargs["append_user"] is True, (
            "Activation must exclude the live trigger from back-fill, leaving "
            "_process_message as its single writer — so handle_input still "
            "appends the user turn exactly once."
        )
        assert len(user_entries_for(bot.session_log, TRIGGER_EVENT_ID)) == 1


# ── A2: gap-fill did NOT back-fetch the triggering event ─────────────────────

class TestGapFillMissedTrigger:
    """A2 — pass-through guard: normal path must be unchanged by the fix."""

    @pytest.mark.asyncio
    async def test_trigger_appended_exactly_once_when_not_gapfilled(self, tmp_path):
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)

        # Back-fetch sees only the known prior message -> immediate overlap,
        # nothing new persisted. The trigger is still unpersisted.
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]))

        await bot._process_message(
            make_room(ROOM_ID),
            make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
            TRIGGER_BODY,
        )

        entries = user_entries_for(bot.session_log, TRIGGER_EVENT_ID)
        assert len(entries) == 1, (
            f"Trigger not seen by gap-fill must be appended exactly once by "
            f"_process_message; found {len(entries)}"
        )

    @pytest.mark.asyncio
    async def test_handle_input_called_with_append_user_true(self, tmp_path):
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)

        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]))

        await bot._process_message(
            make_room(ROOM_ID),
            make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
            TRIGGER_BODY,
        )

        assert agent.handle_input.await_args.kwargs["append_user"] is True, (
            "When gap-fill did not persist the trigger, handle_input must still "
            "append the user message (unchanged behaviour)."
        )


# ── A1b: the REAL production text path ───────────────────────────────────────

class TestTextPathEndToEnd:
    """A1b — drive the real entry point, _handle_room_message.

    This is the path the production wedge actually took, and it does NOT hit the
    lazy wake inside _process_message: _handle_room_message activates the room
    itself (so slash commands see history) BEFORE firing _process_message as a
    background task.  The trigger must therefore be threaded into activation at
    BOTH call sites.
    """

    @pytest.mark.asyncio
    async def test_text_path_logs_trigger_once(self, tmp_path):
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]))

        await bot._handle_room_message(
            make_room(ROOM_ID),
            make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
        )
        await drain(bot)

        dupes = user_entries_for(bot.session_log, TRIGGER_EVENT_ID)
        assert len(dupes) == 1, (
            f"Text path: triggering event must appear exactly ONCE in the JSONL; "
            f"found {len(dupes)}"
        )
        assert agent.handle_input.await_count == 1

    @pytest.mark.asyncio
    async def test_text_path_unaffected_when_gapfill_misses_trigger(self, tmp_path):
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]))

        await bot._handle_room_message(
            make_room(ROOM_ID),
            make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
        )
        await drain(bot)

        entries = user_entries_for(bot.session_log, TRIGGER_EVENT_ID)
        assert len(entries) == 1
        assert agent.handle_input.await_args.kwargs["append_user"] is True


# ── F2d: batched wake — the case tail-equality could not fix ─────────────────

class TestBatchedWake:
    """Round-1's tail-equality guard fixed only the single-trigger fixture.

    Real wakes are batched: the first live callback is for E1 while
    `room_messages` already returns [E2, E1] (newest-first). Round-1 back-filled
    BOTH, leaving E2 as `last_event_id`; E1 then missed the tail guard and was
    appended a second time, and when E2's own callback arrived it missed too —
    two JSONL copies of both events.
    """

    @pytest.mark.asyncio
    async def test_batched_e1_e2_each_persisted_once_in_order(self, tmp_path):
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)

        e1 = make_event(USER, "first rapid message", "$E1", ts=2_000)
        e2 = make_event(USER, "second rapid message", "$E2", ts=3_000)

        # The server already has BOTH E1 and E2 by the time E1's callback runs.
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, "second rapid message", "$E2", ts=3_000),
            make_event(USER, "first rapid message", "$E1", ts=2_000),
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]))

        # E1's callback (triggers activation), then E2's (room already active).
        await bot._handle_room_message(make_room(ROOM_ID), e1)
        await drain(bot)
        await bot._handle_room_message(make_room(ROOM_ID), e2)
        await drain(bot)

        assert len(user_entries_for(bot.session_log, "$E1")) == 1, (
            "E1 must be persisted exactly once: "
            f"{[e.get('content') for e in user_entries_for(bot.session_log, '$E1')]}"
        )
        assert len(user_entries_for(bot.session_log, "$E2")) == 1, (
            "E2 must be persisted exactly once — activation must not back-fill "
            "events NEWER than the trigger (the sync loop dispatches those)."
        )

        # Order preserved: E1 before E2.
        ids = [e.get("event_id") for e in user_entries(bot.session_log)]
        assert ids.index("$E1") < ids.index("$E2"), (
            f"batched events must be persisted in arrival order; got {ids}"
        )
        assert agent.handle_input.await_count == 2, (
            "both distinct events must run a turn"
        )

    @pytest.mark.asyncio
    async def test_batched_wake_newer_event_not_backfilled_early(self, tmp_path):
        """Immediately after E1's activation, E2 must NOT already be in the log —
        it is newer than the trigger and belongs to its own dispatch."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)

        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, "second rapid message", "$E2", ts=3_000),
            make_event(USER, "first rapid message", "$E1", ts=2_000),
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]))

        await bot._handle_room_message(
            make_room(ROOM_ID),
            make_event(USER, "first rapid message", "$E1", ts=2_000),
        )
        await drain(bot)

        assert user_entries_for(bot.session_log, "$E2") == [], (
            "an event NEWER than the trigger must not be back-filled by "
            "activation — the live sync loop will dispatch it separately"
        )


# ── F2d: redelivery is turn-idempotent, not just append-idempotent ───────────

class TestRedeliveryIdempotent:

    @pytest.mark.asyncio
    async def test_immediate_redelivery_runs_no_second_turn(self, tmp_path):
        """Round-1 asserted only the user-append count, so it passed while a
        redelivered event executed a SECOND model turn and delivered a second
        assistant reply."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot._active_rooms.add(ROOM_ID)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))

        for _ in range(2):
            await bot._process_message(
                make_room(ROOM_ID),
                make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
                TRIGGER_BODY,
            )

        assert len(user_entries_for(bot.session_log, TRIGGER_EVENT_ID)) == 1, (
            "a redelivered event must be logged once"
        )
        assert agent.handle_input.await_count == 1, (
            f"a redelivered event must NOT invoke the agent again; "
            f"handle_input ran {agent.handle_input.await_count} times"
        )
        assert len(assistant_entries(bot.session_log)) == 2, (
            "exactly one NEW assistant entry (plus the seeded one) — a "
            f"redelivery must not persist a second reply; got "
            f"{len(assistant_entries(bot.session_log))}"
        )
        replies = [c for c in bot.send.await_args_list if "ok" in str(c.args)]
        assert len(replies) == 1, (
            f"a redelivered event must not deliver a second response; got {replies}"
        )

    @pytest.mark.asyncio
    async def test_redelivery_after_intervening_event(self, tmp_path):
        """Round-1's tail guard only looked at the NEWEST id, so a redelivery
        after any intervening event missed entirely and was fully re-appended
        and re-run."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot._active_rooms.add(ROOM_ID)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        room = make_room(ROOM_ID)

        await bot._process_message(
            room, make_event(USER, "message A", "$A", ts=2_000), "message A")
        await bot._process_message(
            room, make_event(USER, "message B", "$B", ts=3_000), "message B")
        # Now $A is redelivered — it is NOT the tail any more.
        await bot._process_message(
            room, make_event(USER, "message A", "$A", ts=2_000), "message A")

        assert len(user_entries_for(bot.session_log, "$A")) == 1, (
            "a non-tail redelivery must still be recognised via event-ID "
            f"membership; got {len(user_entries_for(bot.session_log, '$A'))} copies"
        )
        assert agent.handle_input.await_count == 2, (
            f"only the two distinct events may run turns; got "
            f"{agent.handle_input.await_count}"
        )

    @pytest.mark.asyncio
    async def test_redelivery_across_activation_boundary(self, tmp_path):
        """An event already in the JSONL from a PREVIOUS process lifetime is a
        redelivery too: the membership set is hydrated once at activation."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        # Pretend a prior run already processed and logged $A.
        bot.session_log.append(
            role="user", sender=USER, room=ROOM_ID, event_id="$A",
            content="message A",
        )
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))

        await bot._process_message(
            make_room(ROOM_ID), make_event(USER, "message A", "$A", ts=2_000),
            "message A")

        assert len(user_entries_for(bot.session_log, "$A")) == 1
        assert agent.handle_input.await_count == 0, (
            "an event already durably logged must not re-run the model after a "
            "restart-time redelivery"
        )

    @pytest.mark.asyncio
    async def test_redelivery_releases_the_session_lock(self, tmp_path):
        """The early return happens post-lock; the lock must still be released."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot._active_rooms.add(ROOM_ID)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        room = make_room(ROOM_ID)
        ev = make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000)

        await bot._process_message(room, ev, TRIGGER_BODY)
        await bot._process_message(room, ev, TRIGGER_BODY)

        lock = bot._session_locks.get(ROOM_ID)
        assert lock is None or not lock.locked(), (
            "the redelivery early-return must not leak the per-room session lock"
        )
        assert ROOM_ID not in bot._active_turns

    @pytest.mark.asyncio
    async def test_redelivery_logged_at_info(self, tmp_path, caplog):
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot._active_rooms.add(ROOM_ID)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        room = make_room(ROOM_ID)
        ev = make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000)

        await bot._process_message(room, ev, TRIGGER_BODY)
        with caplog.at_level("INFO", logger="openalph.matrix"):
            await bot._process_message(room, ev, TRIGGER_BODY)

        assert any(TRIGGER_EVENT_ID in r.message % r.args if r.args else
                   TRIGGER_EVENT_ID in r.message
                   for r in caplog.records), (
            f"redelivery must be logged at INFO; records: "
            f"{[r.getMessage() for r in caplog.records]}"
        )


# ── F2d: dormant-room MEDIA trigger keeps its canonical tag ──────────────────

class TestDormantRoomMediaTrigger:
    """MAJOR #2 in the review. Gap-fill serialized a media event's raw Matrix
    `body` (usually a bare filename/caption), while _handle_media_message passes
    _process_message the canonical
    `[media: media/<hash>/<file> (<mime>, <size>)]` tag. Under the Round-1 guard
    the canonical append was SKIPPED and handle_input got append_user=False —
    and Agent.handle_input does not add the `text` argument to hydrated history.
    The model therefore saw only "photo.jpg", could not locate the downloaded
    file, and vision mode could not expand the image. The raw representation was
    also the only durable JSONL record.

    The internal lazy-wake branch is the production media wake path, so this is
    not dead code.
    """

    def _media_dir_hash(self, event_id):
        return hashlib.sha256(event_id.encode()).hexdigest()[:16]

    @pytest.mark.asyncio
    async def test_media_trigger_canonical_tag_in_jsonl_and_history(self, tmp_path):
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        # Room is DORMANT: the media path reaches _process_message's lazy wake.
        assert ROOM_ID not in bot._active_rooms

        media_event_id = "$media-trigger"
        # The server back-fetch includes the media event with its RAW body.
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, "photo.jpg", media_event_id, ts=2_000),
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]))

        download = MagicMock()
        download.body = b"\xff\xd8\xff\xd9jpegbytes"
        download.content_type = "image/jpeg"
        download.filename = None
        download.__class__.__name__ = "MemoryDownloadResponse"
        bot.client.download = AsyncMock(return_value=download)

        event = make_media_event(USER, "photo.jpg", media_event_id, ts=2_000)

        # Capture the history handle_input would see (hydrated by activation +
        # the canonical append), exactly as the real Agent does.
        seen_history = {}

        async def capture(body, room_id, **kwargs):
            seen_history["body"] = body
            seen_history["append_user"] = kwargs.get("append_user")
            seen_history["history"] = [
                dict(h) for h in agent.history(room_id)
            ]
            return "looked at your photo"

        agent.handle_input = AsyncMock(side_effect=capture)

        await bot._handle_media_message(make_room(ROOM_ID), event)
        await drain(bot)

        expected_tag = (
            f"[media: media/{self._media_dir_hash(media_event_id)}/photo.jpg "
            f"(image/jpeg, "
        )

        media_entries = user_entries_for(bot.session_log, media_event_id)
        assert len(media_entries) == 1, (
            f"the media trigger must be persisted exactly once; got "
            f"{[e.get('content') for e in media_entries]}"
        )
        assert media_entries[0]["content"].startswith(expected_tag), (
            "the durable JSONL record must be the CANONICAL media tag (with the "
            "downloaded-file reference), not the raw Matrix body: "
            f"{media_entries[0]['content']!r}"
        )

        # And the model must actually receive it.
        assert seen_history["body"].startswith(expected_tag)
        hydrated = "\n".join(
            str(h.get("content")) for h in seen_history["history"]
        )
        assert (seen_history["append_user"] is True
                or expected_tag in hydrated), (
            "handle_input must either append the canonical tag itself "
            "(append_user=True) or find it already hydrated in history; "
            f"append_user={seen_history['append_user']!r} history={hydrated!r}"
        )
        # The raw body must never be the record for this event.
        assert media_entries[0]["content"] != "photo.jpg"

    @pytest.mark.asyncio
    async def test_media_file_written_and_referenced(self, tmp_path):
        """The tag's path must point at the file that was actually downloaded."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        media_event_id = "$media-trigger-2"
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, "shot.png", media_event_id, ts=2_000),
            make_event(USER, "an earlier message", "$prior-user", ts=1_000),
        ]))
        download = MagicMock()
        download.body = b"pngbytes"
        download.content_type = "image/png"
        download.filename = None
        download.__class__.__name__ = "MemoryDownloadResponse"
        bot.client.download = AsyncMock(return_value=download)

        event = make_media_event(USER, "shot.png", media_event_id, ts=2_000,
                                 mimetype="image/png", size=8)
        await bot._handle_media_message(make_room(ROOM_ID), event)
        await drain(bot)

        rel = f"media/{self._media_dir_hash(media_event_id)}/shot.png"
        assert (tmp_path / rel).exists(), "the media file must be on disk"
        entries = user_entries_for(bot.session_log, media_event_id)
        assert len(entries) == 1
        assert rel in entries[0]["content"], (
            f"the persisted tag must reference the downloaded file; got "
            f"{entries[0]['content']!r}"
        )


# ── Edge cases the fix must not disturb ──────────────────────────────────────

class TestFixDoesNotOverTrigger:

    @pytest.mark.asyncio
    async def test_gated_room_path_unchanged(self, tmp_path):
        """Gated rooms buffer the trigger in _handle_room_message BEFORE
        _process_message runs and hydrate from JSONL; the ungated append never
        runs and append_user must stay False.

        Critically, the new event-ID membership gate must NOT mistake that
        legitimate pre-append for a redelivery and skip the turn: whichever path
        performs the append CLAIMS the event.
        """
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([
            make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
        ]))

        await bot._handle_room_message(
            make_room(ROOM_ID, member_count=3),
            make_event(USER, f"{AGENT_USER} {TRIGGER_BODY}", TRIGGER_EVENT_ID,
                       ts=2_000),
        )
        await drain(bot)

        entries = user_entries_for(bot.session_log, TRIGGER_EVENT_ID)
        assert len(entries) == 1, (
            f"Gated path must keep exactly the one pre-buffered entry; got {len(entries)}"
        )
        assert agent.handle_input.await_count == 1, (
            "the gated pre-append must NOT be misread as a redelivery — the turn "
            "must still run"
        )
        assert agent.handle_input.await_args.kwargs["append_user"] is False

    @pytest.mark.asyncio
    async def test_gated_direct_process_message_unchanged(self, tmp_path):
        """Same invariant via the direct _process_message entry (media in a
        gated room), where _process_message itself buffers the trigger."""
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))

        await bot._process_message(
            make_room(ROOM_ID, member_count=3),
            make_event(USER, f"{AGENT_USER} {TRIGGER_BODY}", TRIGGER_EVENT_ID,
                       ts=2_000),
            f"{AGENT_USER} {TRIGGER_BODY}",
        )

        assert len(user_entries_for(bot.session_log, TRIGGER_EVENT_ID)) == 1
        assert agent.handle_input.await_count == 1

    @pytest.mark.asyncio
    async def test_already_active_room_still_appends_new_event(self, tmp_path):
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot._active_rooms.add(ROOM_ID)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))

        await bot._process_message(
            make_room(ROOM_ID),
            make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
            TRIGGER_BODY,
        )

        entries = user_entries_for(bot.session_log, TRIGGER_EVENT_ID)
        assert len(entries) == 1, (
            "Active room, unseen event: the ungated append must still happen "
            f"exactly once; got {len(entries)}"
        )
        assert agent.handle_input.await_args.kwargs["append_user"] is True

    @pytest.mark.asyncio
    async def test_new_room_no_prior_history(self, tmp_path):
        """Brand-new room: gap-fill does not run (no existing entries), so the
        trigger must be appended exactly once with append_user=True."""
        bot, agent = make_bot(tmp_path)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))

        await bot._process_message(
            make_room(ROOM_ID),
            make_event(USER, TRIGGER_BODY, TRIGGER_EVENT_ID, ts=2_000),
            TRIGGER_BODY,
        )

        entries = user_entries_for(bot.session_log, TRIGGER_EVENT_ID)
        assert len(entries) == 1
        assert agent.handle_input.await_args.kwargs["append_user"] is True

    @pytest.mark.asyncio
    async def test_trigger_without_event_id_still_appended(self, tmp_path):
        """Synthetic/heartbeat-shaped events carry no event_id and must never be
        deduped against a None-valued log entry."""
        bot, agent = make_bot(tmp_path)
        bot.session_log.append(
            role="system", sender=AGENT_USER, room=ROOM_ID,
            event_id=None, event="session_start", detail="fresh",
        )
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))

        event = make_event(USER, TRIGGER_BODY, None, ts=2_000)
        await bot._process_message(make_room(ROOM_ID), event, TRIGGER_BODY)

        entries = [
            e for e in bot.session_log.read(ROOM_ID)
            if e.get("role") == "user" and e.get("content") == TRIGGER_BODY
        ]
        assert len(entries) == 1, (
            "An event with event_id=None must never be treated as already "
            f"processed; got {len(entries)} entries"
        )
        assert agent.handle_input.await_args.kwargs["append_user"] is True

    @pytest.mark.asyncio
    async def test_two_eventless_triggers_both_run(self, tmp_path):
        """Two distinct id-less (synthetic) messages must both run — the
        membership set must not collapse them."""
        bot, agent = make_bot(tmp_path)
        bot._active_rooms.add(ROOM_ID)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        room = make_room(ROOM_ID)

        await bot._process_message(room, make_event(USER, "one", None), "one")
        await bot._process_message(room, make_event(USER, "two", None), "two")

        assert agent.handle_input.await_count == 2
        assert len(user_entries(bot.session_log)) == 2


# ── F2c: the per-message path must stay O(1) ──────────────────────────────────

# Reads of the room JSONL performed per TURN by code that predates Round 2 and
# is NOT part of the dedup path: the post-turn context-capacity check
# (`_process_message` -> `session_log.build_context(room_id)` -> `read`), which
# warns when context passes 80%. Verified identical at git HEAD, so it is
# neither introduced nor removable by F2 — but it IS pinned here as an exact
# number so that any NEW per-message scan (a regression of the round-1 mistake)
# fails these tests loudly. Reducing it is tracked as a separate follow-up.
PREEXISTING_READS_PER_TURN = 1


class TestPerMessageCostIsConstant:
    """MINOR (review): Round-1 called `SessionLog.last_event_id(room_id)` on
    EVERY ungated message — a synchronous open + full-file JSON decode of the
    whole room JSONL, performed while the event loop and both room locks are
    held. Long sessions with large tool records could pause unrelated rooms.
    Membership must come from an in-memory set hydrated ONCE at activation.

    The invariant these tests pin is therefore twofold:
      * `last_event_id()` — the dedup accessor — is called ZERO times per
        message (it belongs to activation only);
      * total per-message JSONL reads are a CONSTANT that does not grow with
        session size or message count.

    NOTE ON MECHANISM: assertions are made on COUNTS collected by wrapper
    functions, never on a `side_effect=AssertionError`. `_process_message`
    wraps the capacity check in a bare `except Exception`, which silently
    swallows a raised AssertionError — a raising probe there passes whether or
    not the code under test is correct.
    """

    @pytest.mark.asyncio
    async def test_no_dedup_scan_per_message(self, tmp_path):
        """`last_event_id()` must never run on the per-message path.

        This is the direct F2c invariant and is RED against the round-1 tail
        guard, which called it for every ungated message.
        """
        bot, agent = make_bot(tmp_path)
        seed_prior_history(bot.session_log)
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        room = make_room(ROOM_ID)

        # Activate first (a scan THERE is fine and expected).
        await bot._process_message(
            room, make_event(USER, "first", "$m0", ts=2_000), "first")

        calls = {"last_event_id": 0}
        real_leid = SessionLog.last_event_id

        def counting_leid(self, room_id):
            calls["last_event_id"] += 1
            return real_leid(self, room_id)

        with patch.object(SessionLog, "last_event_id", counting_leid):
            for i in range(1, 4):
                await bot._process_message(
                    room, make_event(USER, f"msg {i}", f"$m{i}", ts=2_000 + i),
                    f"msg {i}")

        assert calls["last_event_id"] == 0, (
            "SessionLog.last_event_id() must not be called per message — "
            "redelivery membership comes from the in-memory set hydrated at "
            f"activation; got {calls['last_event_id']} calls"
        )
        assert agent.handle_input.await_count == 4
        ids = [e.get("event_id") for e in user_entries(bot.session_log)]
        assert ids == ["$prior-user", "$m0", "$m1", "$m2", "$m3"], ids

    @pytest.mark.asyncio
    async def test_large_log_activation_scans_once(self, tmp_path):
        """A big session must be scanned once at activation, not per message.

        Per-message reads must be a CONSTANT independent of log size, and must
        not include any dedup scan.
        """
        bot, agent = make_bot(tmp_path)
        for i in range(300):
            bot.session_log.append(
                role="user", sender=USER, room=ROOM_ID,
                event_id=f"$old{i}", content=f"old message {i}",
            )
        bot.client.room_messages = AsyncMock(return_value=gapfill_page([]))
        room = make_room(ROOM_ID)

        reads = {"n": 0}
        leid = {"n": 0}
        real_read = SessionLog.read
        real_leid = SessionLog.last_event_id

        def counting_read(self, room_id):
            reads["n"] += 1
            return real_read(self, room_id)

        def counting_leid(self, room_id):
            leid["n"] += 1
            return real_leid(self, room_id)

        with patch.object(SessionLog, "read", counting_read), \
                patch.object(SessionLog, "last_event_id", counting_leid):
            await bot._process_message(
                room, make_event(USER, "live one", "$live1", ts=9_000), "live one")
            scans_after_activation = reads["n"]
            leid_after_activation = leid["n"]

            n_messages = 4
            for i in range(2, 2 + n_messages):
                await bot._process_message(
                    room, make_event(USER, f"live {i}", f"$live{i}", ts=9_000 + i),
                    f"live {i}")

            per_message_reads = reads["n"] - scans_after_activation

            # The dedup path must contribute NOTHING per message.
            assert leid["n"] == leid_after_activation, (
                "SessionLog.last_event_id() must not be called per message; "
                f"{leid_after_activation} -> {leid['n']}"
            )
            # Reads per message must be an exact constant — no growth with log
            # size, and no new scan beyond the pre-existing capacity check.
            assert per_message_reads == n_messages * PREEXISTING_READS_PER_TURN, (
                f"per-message JSONL reads must stay at exactly "
                f"{PREEXISTING_READS_PER_TURN} (the pre-existing post-turn "
                f"context-capacity check, NOT a dedup scan); got "
                f"{per_message_reads} reads across {n_messages} messages"
            )

            # And an old event redelivered is still recognised from the hydrated
            # set — WITHOUT any further JSONL read, since the gate returns before
            # the capacity check runs.
            reads_before_redelivery = reads["n"]
            await bot._process_message(
                room, make_event(USER, "old message 7", "$old7", ts=1_000),
                "old message 7")

        assert len(user_entries_for(bot.session_log, "$old7")) == 1
        assert reads["n"] == reads_before_redelivery, (
            "a redelivery must be recognised from the in-memory set alone, "
            f"with no JSONL read at all; got "
            f"{reads['n'] - reads_before_redelivery} reads"
        )
