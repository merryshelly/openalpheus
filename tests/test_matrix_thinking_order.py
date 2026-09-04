"""Thinking-details ordering + fail-soft delivery (kdsn.328).

Pins the room-timeline contract for the 💭 Thinking furled block:

  1. The thinking details block must be SENT before the first room-visible
     byte of the answer — reasoning reads above, answer last (mobile
     clients render <details> unfurled, so late details land below the
     answer and the operator scrolls up past reasoning to find content).
  2. One details event per thinking segment per tool-loop iteration.
  3. A failed details send must never kill the turn: the answer still
     delivers and no internal-error notice fires.

The fakes below reproduce agent.py's real callback firing order: thinking
deltas stream first, then text deltas, then (at stream end) the text done
signal fires before the thinking done signal. That order is what makes the
ordering pins RED against the current implementation.

Event channels: initial sends / thinking details / finalize edits flow
through bot._room_send_with_retry; bulk responses (heartbeat path, short
stream finalize) through bot.send. Both are recorded into one ordered list.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.matrix import MatrixBot
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.agent import Agent

ROOM = "!test:matrix.local"


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_matrix_streaming.py)
# ---------------------------------------------------------------------------

def make_matrix_config(**kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id="@test:matrix.local",
        device_id="TEST",
        password="test-pw",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def make_agent_config(workspace, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
        )},
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
    )
    defaults.update(kwargs)
    defaults["workspace"] = workspace
    return AgentConfig(**defaults)


def make_bot(tmp_path, **kwargs):
    """Create a MatrixBot with mocked Matrix client."""
    agent_config = make_agent_config(tmp_path, **kwargs)
    agent_config.matrix = make_matrix_config()
    agent = MagicMock(spec=Agent)
    agent.config = agent_config
    agent.system_prompt = "test prompt"
    agent.history = MagicMock(return_value=[])

    matrix_config = make_matrix_config()
    with patch("openalph.matrix.AsyncClient"):
        bot = MatrixBot(agent, matrix_config)
        bot.agent = agent
        bot.config = matrix_config
        bot._synced = True
        bot._active_rooms = set()
        bot.session_log = None

        # Mock Matrix client methods
        bot.client = MagicMock()
        bot.client.room_typing = AsyncMock()
        send_resp = MagicMock()
        send_resp.event_id = "$evt_stream"
        bot.client.room_send = AsyncMock(return_value=send_resp)
        bot._room_send_with_retry = AsyncMock(return_value=send_resp)
        bot.send = AsyncMock()
        bot.send_notice = AsyncMock()

    return bot, agent


def make_room(room_id=ROOM, members=2):
    room = MagicMock()
    room.room_id = room_id
    room.users = {f"@user{i}:matrix.local": MagicMock() for i in range(members)}
    room.name = "Test Room"
    room.display_name = "Test Room"
    return room


def make_event(sender="@user:matrix.local", body="Hello", event_id="$evt1"):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = 1000000
    event.source = {}
    return event


# ---------------------------------------------------------------------------
# Order-recording helpers
# ---------------------------------------------------------------------------

def record_order(bot):
    """Route both send channels into one ordered list.

    Returns (order, send_resp). Entries: ("room_send", content-dict) for
    _room_send_with_retry traffic and ("bulk", text-str) for bot.send.
    """
    order = []
    send_resp = MagicMock()
    send_resp.event_id = "$evt_rec"

    async def rec_room_send(room_id, content, *a, **k):
        order.append(("room_send", content))
        return send_resp

    async def rec_send(room_id, text, *a, **k):
        order.append(("bulk", text))
        return send_resp

    bot._room_send_with_retry = AsyncMock(side_effect=rec_room_send)
    bot.send = AsyncMock(side_effect=rec_send)
    return order, send_resp


def kinds(order):
    """Classify recorded events in timeline order."""
    out = []
    for label, payload in order:
        if label == "bulk":
            out.append("bulk")
        elif isinstance(payload, dict):
            if payload.get("openalph.thinking") is True:
                out.append("thinking")
            elif isinstance(payload.get("m.relates_to"), dict) and \
                    payload["m.relates_to"].get("rel_type") == "m.replace":
                out.append("edit")
            elif payload.get("msgtype") == "m.text":
                out.append("text")
            else:
                out.append("other")
        else:
            out.append("other")
    return out


def thinking_indices(k):
    return [i for i, x in enumerate(k) if x == "thinking"]


def wire_fake_turn(agent, events, response=None):
    """Replace agent.handle_input with a fake that fires the streaming
    callbacks in the exact order listed. Callers reproduce agent.py's real
    firing order: thinking deltas, then text deltas, then (at stream end)
    the text done signal before the thinking done signal.
    """

    async def fake_handle_input(*args, **kwargs):
        t_cb = kwargs.get("on_thinking_delta")
        x_cb = kwargs.get("on_text_delta")
        for kind, payload, done in events:
            if kind == "thinking":
                await t_cb(payload, done)
            elif kind == "text":
                await x_cb(payload, done)
            else:
                raise ValueError(f"unknown event kind {kind!r}")
        return response

    agent.handle_input = fake_handle_input


# ---------------------------------------------------------------------------
# Ordering pins (live-message path)
# ---------------------------------------------------------------------------

class TestThinkingOrder:
    @pytest.mark.asyncio
    async def test_details_precede_streamed_answer(self, tmp_path):
        """kdsn.328 — RED before the fix.

        Thinking then text (>40 chars → StreamingDelivery initial send).
        Today the details block fires on the thinking done signal, which
        agent.py emits AFTER all text deltas — details land below the
        answer. Required: details are the first room-visible event.
        """
        bot, agent = make_bot(tmp_path)
        order, _ = record_order(bot)
        answer = "The answer is forty-two, delivered after careful reasoning."
        wire_fake_turn(agent, events=[
            ("thinking", "Deep reasoning here.", False),
            ("text", answer, False),
            ("text", "", True),
            ("thinking", "", True),
        ])

        await bot._process_message(make_room(), make_event(), "Hi")

        k = kinds(order)
        assert k.count("thinking") == 1, f"expected exactly 1 details event, got {k}"
        assert thinking_indices(k)[0] < k.index("text"), (
            f"thinking details must precede the streamed answer, got {k}"
        )

    @pytest.mark.asyncio
    async def test_thinking_only_iteration_still_sends_details(self, tmp_path):
        """Iteration with thinking but no text (tool-call iteration): the
        details block is still delivered (end-of-stream fallback)."""
        bot, agent = make_bot(tmp_path)
        order, _ = record_order(bot)
        wire_fake_turn(agent, events=[
            ("thinking", "plan the work", False),
            ("thinking", "", True),
        ])

        await bot._process_message(make_room(), make_event(), "Hi")

        k = kinds(order)
        assert k.count("thinking") == 1, f"details event missing, got {k}"

    @pytest.mark.asyncio
    async def test_multi_iteration_details_precede_answer(self, tmp_path):
        """kdsn.328 — RED before the fix.

        Iteration 1 emits thinking only; iteration 2 emits thinking then
        text. Iteration 2's details must flush before iteration 2's answer
        streams — not at final stream end.
        """
        bot, agent = make_bot(tmp_path)
        order, _ = record_order(bot)
        answer = "Final answer, reasoned across two passes."
        wire_fake_turn(agent, events=[
            ("thinking", "iter1 reasoning", False),
            ("thinking", "", True),
            ("thinking", "iter2 reasoning", False),
            ("text", answer, False),
            ("text", "", True),
            ("thinking", "", True),
        ])

        await bot._process_message(make_room(), make_event(), "Hi")

        k = kinds(order)
        assert k.count("thinking") == 2, f"expected 2 details events, got {k}"
        assert max(thinking_indices(k)) < k.index("text"), (
            f"iteration-2 details must precede the answer, got {k}"
        )

    @pytest.mark.asyncio
    async def test_interleaved_thinking_flushes_below_answer(self, tmp_path):
        """Accepted fallback (kdsn.328): thinking that arrives AFTER text
        started flushes at the next boundary — below the answer start.
        Pins the fallback so a refactor can't silently change it.
        """
        bot, agent = make_bot(tmp_path)
        order, _ = record_order(bot)
        answer = "The answer is forty-two, delivered after careful reasoning."
        wire_fake_turn(agent, events=[
            ("text", answer, False),
            ("thinking", "late thought", False),
            ("text", "", True),
            ("thinking", "", True),
        ])

        await bot._process_message(make_room(), make_event(), "Hi")

        k = kinds(order)
        assert k.count("thinking") == 1, f"details event missing, got {k}"
        assert thinking_indices(k)[0] > k.index("text"), (
            f"interleaved thinking is accepted below the answer start, got {k}"
        )

    @pytest.mark.asyncio
    async def test_no_thinking_no_details(self, tmp_path):
        """Text-only turn: zero details events, answer delivered once."""
        bot, agent = make_bot(tmp_path)
        order, _ = record_order(bot)
        answer = "Plain answer, no thinking involved."
        wire_fake_turn(agent, events=[
            ("text", answer, False),
            ("text", "", True),
        ], response=answer)

        await bot._process_message(make_room(), make_event(), "Hi")

        k = kinds(order)
        assert "thinking" not in k, f"no details expected, got {k}"
        payloads = [p for _, p in order]
        assert sum(1 for p in payloads if "Plain answer" in str(p)) >= 1


# ---------------------------------------------------------------------------
# Fail-soft delivery
# ---------------------------------------------------------------------------

class TestThinkingFailSoft:
    @pytest.mark.asyncio
    async def test_details_send_failure_does_not_kill_turn(self, tmp_path):
        """kdsn.328 — RED before the fix.

        A failed details send must be swallowed: the turn completes, the
        answer still delivers, and no internal-error notice fires. Today
        the send inside the done branch is unguarded, so a transient
        Matrix failure aborts the turn after the answer was streamed.
        """
        bot, agent = make_bot(tmp_path)
        order = []
        send_resp = MagicMock()
        send_resp.event_id = "$evt_rec"

        async def failing_room_send(room_id, content, *a, **k):
            order.append(("room_send", content))
            if isinstance(content, dict) and content.get("openalph.thinking") is True:
                raise Exception("network down")
            return send_resp

        async def rec_send(room_id, text, *a, **k):
            order.append(("bulk", text))
            return send_resp

        bot._room_send_with_retry = AsyncMock(side_effect=failing_room_send)
        bot.send = AsyncMock(side_effect=rec_send)

        answer = "The answer survives the thinking-send failure."
        wire_fake_turn(agent, events=[
            ("thinking", "Deep reasoning here.", False),
            ("text", answer, False),
            ("text", "", True),
            ("thinking", "", True),
        ])

        await bot._process_message(make_room(), make_event(), "Hi")  # must not raise

        payloads = [p for _, p in order]
        assert any("survives the thinking-send" in str(p) for p in payloads), (
            f"answer must still be delivered, got {payloads}"
        )
        texts = [p for p in payloads if isinstance(p, str)]
        assert not any("Internal error" in t for t in texts), (
            f"no error notice expected, got {texts}"
        )


# ---------------------------------------------------------------------------
# Heartbeat path (bulk response — ordering already correct, pinned)
# ---------------------------------------------------------------------------

class TestHeartbeatPath:
    @pytest.mark.asyncio
    async def test_heartbeat_details_precede_bulk_response(self, tmp_path):
        """Heartbeat flushes details at thinking-end, inside handle_input,
        before the bulk response send. Pin the existing contract."""
        bot, agent = make_bot(tmp_path)
        order, _ = record_order(bot)
        wire_fake_turn(agent, events=[
            ("thinking", "hb reasoning", False),
            ("thinking", "", True),
        ], response="HB answer")

        bot._active_rooms.add(ROOM)  # skip room activation
        await bot._run_heartbeat_turn(ROOM, "heartbeat prompt")

        k = kinds(order)
        assert k.count("thinking") == 1, f"details event missing, got {k}"
        assert thinking_indices(k)[0] < k.index("bulk"), (
            f"details must precede the bulk response, got {k}"
        )

    @pytest.mark.asyncio
    async def test_heartbeat_no_thinking_no_details(self, tmp_path):
        bot, agent = make_bot(tmp_path)
        order, _ = record_order(bot)
        wire_fake_turn(agent, events=[], response="HB plain")

        bot._active_rooms.add(ROOM)
        await bot._run_heartbeat_turn(ROOM, "heartbeat prompt")

        k = kinds(order)
        assert "thinking" not in k, f"no details expected, got {k}"
