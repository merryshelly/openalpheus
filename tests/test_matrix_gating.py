"""Integration tests for mention gating in MatrixBot._handle_room_message.

These test the full flow: message comes in from Matrix sync → gating check →
either skip (buffer only) or process through agent.

Interface contract:
    - 3+ member rooms: only process messages that @mention the agent
    - 2-member rooms: process all messages (DM behavior, unchanged from Phase 4)
    - TOML overrides: require_mention=true/false per room
    - Buffered messages: all messages logged to session JSONL regardless of mention
    - Context hydration: on mention, agent sees all buffered messages
    - Commands (/stop, /status) bypass gating
    - m.notice (tool notices) don't trigger _handle_room_message at all
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock, call
from pathlib import Path
from openalph.matrix import MatrixBot
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig


# --- Fixtures ---


def make_matrix_config(rooms=None, **kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id="@watson:matrix.local",
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
        rooms=rooms,
    )
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def make_agent_config(workspace, **kwargs):
    defaults = dict(
        name="watson",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(key="anthropic", type="anthropic", api_key="sk-test", base_url=None, quirks=[])},
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_room(room_id, member_count):
    """Create a mock room with N members."""
    room = MagicMock()
    room.room_id = room_id
    room.name = "Test Room"
    room.display_name = "Test Room"
    room.users = {f"@user{i}:matrix.local": MagicMock() for i in range(member_count)}
    return room


def make_event(sender, body, event_id="$evt1", mentions_user_ids=None):
    """Create a mock RoomMessageText event.
    
    Args:
        sender: Matrix user ID of sender
        body: Message text
        event_id: Matrix event ID
        mentions_user_ids: List of user IDs for m.mentions.user_ids (optional)
    """
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = 1000000
    
    # Build event.source for structured mention detection
    content = {"msgtype": "m.text", "body": body}
    if mentions_user_ids is not None:
        content["m.mentions"] = {"user_ids": mentions_user_ids}
    event.source = {"content": content}
    
    return event


def make_bot(tmp_path, rooms=None):
    """Create a MatrixBot with mocked internals, ready for testing.
    
    Returns (bot, agent_mock) tuple.
    """
    matrix_config = make_matrix_config(rooms=rooms)
    agent_config = make_agent_config(workspace=tmp_path)
    
    agent = MagicMock()
    agent.config = agent_config
    agent.handle_input = AsyncMock(return_value="Agent response")
    agent.status = MagicMock(return_value={
        "name": "watson",
        "model": "claude-sonnet-4-20250514",
        "context_tokens": 1000,
        "context_max": 200000,
        "context_pct": 0,
        "turns": 5,
        "total_input_tokens": 5000,
        "total_output_tokens": 2000,
        "total_tool_calls": 3,
    })
    agent.history = MagicMock(return_value=[])
    agent.cancel = MagicMock()
    
    with patch("openalph.matrix.AsyncClient"):
        bot = MatrixBot(agent, matrix_config)
    
    bot._synced = True
    bot.send = AsyncMock()
    bot.send_notice = AsyncMock()
    bot._set_typing = AsyncMock()
    
    return bot, agent


# --- Gated Room: Non-Mentioned Messages ---


class TestGatedRoomSkipsNonMention:
    """3+ member room: messages without @mention are buffered but not processed."""

    @pytest.mark.asyncio
    async def test_no_mention_no_api_call(self, tmp_path):
        """Non-mentioned message in gated room → agent.handle_input NOT called."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        event = make_event("@alice:matrix.local", "Hello everyone")
        
        await bot._handle_room_message(room, event)
        
        agent.handle_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_mention_no_response_sent(self, tmp_path):
        """Non-mentioned message → no message sent to room."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        event = make_event("@alice:matrix.local", "Hello everyone")
        
        await bot._handle_room_message(room, event)
        
        bot.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_mention_buffered_in_session_log(self, tmp_path):
        """Non-mentioned message is still written to session JSONL."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        event = make_event("@alice:matrix.local", "Hello everyone")
        
        await bot._handle_room_message(room, event)
        
        # Check session log was written
        entries = bot.session_log.read("!group:matrix.local")
        assert len(entries) >= 1
        last_user_entry = [e for e in entries if e["role"] == "user"][-1]
        assert last_user_entry["content"] == "Hello everyone"
        assert last_user_entry["mentioned"] is False

    @pytest.mark.asyncio
    async def test_no_mention_no_typing_indicator(self, tmp_path):
        """Non-mentioned message → no typing indicator set."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        event = make_event("@alice:matrix.local", "Hello everyone")
        
        await bot._handle_room_message(room, event)
        
        bot._set_typing.assert_not_called()


# --- Gated Room: Mentioned Messages ---


class TestGatedRoomProcessesMention:
    """3+ member room: @mentioned messages are processed through agent."""

    @pytest.mark.asyncio
    async def test_structured_mention_triggers_agent(self, tmp_path):
        """Structured m.mentions → agent.handle_input called."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        event = make_event(
            "@alice:matrix.local",
            "Hey watson what do you think?",
            mentions_user_ids=["@watson:matrix.local"],
        )
        
        # Pre-activate the room so we don't hit lazy wake
        bot._active_rooms.add("!group:matrix.local")
        
        await bot._handle_room_message(room, event)
        
        agent.handle_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_body_mention_triggers_agent(self, tmp_path):
        """@watson in message body → agent.handle_input called."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        event = make_event("@alice:matrix.local", "Hey @watson what do you think?")
        
        bot._active_rooms.add("!group:matrix.local")
        
        await bot._handle_room_message(room, event)
        
        agent.handle_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_id_mention_triggers_agent(self, tmp_path):
        """@watson:matrix.local in body → agent.handle_input called."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        event = make_event("@alice:matrix.local", "Hey @watson:matrix.local what's up?")
        
        bot._active_rooms.add("!group:matrix.local")
        
        await bot._handle_room_message(room, event)
        
        agent.handle_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_mention_sends_response(self, tmp_path):
        """Mentioned message → response sent to room."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        event = make_event(
            "@alice:matrix.local",
            "@watson help",
            mentions_user_ids=["@watson:matrix.local"],
        )
        
        bot._active_rooms.add("!group:matrix.local")
        
        await bot._handle_room_message(room, event)
        
        bot.send.assert_called_once_with("!group:matrix.local", "Agent response")

    @pytest.mark.asyncio
    async def test_mention_sets_typing(self, tmp_path):
        """Mentioned message → typing indicator ON then OFF."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        event = make_event(
            "@alice:matrix.local",
            "@watson help",
            mentions_user_ids=["@watson:matrix.local"],
        )
        
        bot._active_rooms.add("!group:matrix.local")
        
        await bot._handle_room_message(room, event)
        
        # Typing ON was called
        bot._set_typing.assert_any_call("!group:matrix.local", True)
        # Typing OFF was called (in finally block)
        bot._set_typing.assert_any_call("!group:matrix.local", False)


# --- Ungated Room (DM): All Messages Processed ---


class TestUngatedRoomProcessesAll:
    """2-member room: all messages processed (existing Phase 4 behavior)."""

    @pytest.mark.asyncio
    async def test_dm_processes_without_mention(self, tmp_path):
        """2-member room, no mention → agent.handle_input called (DM behavior)."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!dm:matrix.local", 2)
        event = make_event("@alice:matrix.local", "Hello")
        
        bot._active_rooms.add("!dm:matrix.local")
        
        await bot._handle_room_message(room, event)
        
        agent.handle_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_dm_sends_response(self, tmp_path):
        """2-member room → response sent."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!dm:matrix.local", 2)
        event = make_event("@alice:matrix.local", "Hello")
        
        bot._active_rooms.add("!dm:matrix.local")
        
        await bot._handle_room_message(room, event)
        
        bot.send.assert_called_once()


# --- TOML Override ---


class TestTomlOverride:
    """TOML [matrix.rooms] overrides auto-detection."""

    @pytest.mark.asyncio
    async def test_override_false_on_group(self, tmp_path):
        """require_mention=false on 3-member room → processes all messages."""
        bot, agent = make_bot(tmp_path, rooms={
            "!group:matrix.local": {"require_mention": False}
        })
        room = make_room("!group:matrix.local", 3)
        event = make_event("@alice:matrix.local", "Hello everyone")
        
        bot._active_rooms.add("!group:matrix.local")
        
        await bot._handle_room_message(room, event)
        
        agent.handle_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_override_true_on_dm(self, tmp_path):
        """require_mention=true on 2-member room → requires mention."""
        bot, agent = make_bot(tmp_path, rooms={
            "!dm:matrix.local": {"require_mention": True}
        })
        room = make_room("!dm:matrix.local", 2)
        event = make_event("@alice:matrix.local", "Hello")
        
        await bot._handle_room_message(room, event)
        
        agent.handle_input.assert_not_called()


# --- Commands Bypass Gating ---


class TestCommandsBypassGating:
    """/stop and /status work in gated rooms without @mention."""

    @pytest.mark.asyncio
    async def test_stop_works_in_gated_room(self, tmp_path):
        """"/stop" in gated room without mention → still cancels."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        event = make_event("@alice:matrix.local", "/stop")
        
        await bot._handle_room_message(room, event)
        
        # /stop should have triggered cancel, not agent processing
        agent.handle_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_status_works_in_gated_room(self, tmp_path):
        """"/status" in gated room without mention → still responds."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        event = make_event("@alice:matrix.local", "/status")
        
        bot._active_rooms.add("!group:matrix.local")
        
        await bot._handle_room_message(room, event)
        
        # /status should send status, not process through agent
        agent.handle_input.assert_not_called()
        bot.send.assert_called_once()


# --- Room Transition ---


class TestRoomTransition:
    """Gating responds dynamically to member count changes."""

    @pytest.mark.asyncio
    async def test_transition_2_to_3_activates_gating(self, tmp_path):
        """Room starts with 2, 3rd joins → non-mentioned message skipped."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room:matrix.local", 2)
        
        # First message: 2 members, no gating
        event1 = make_event("@alice:matrix.local", "Hello", event_id="$e1")
        bot._active_rooms.add("!room:matrix.local")
        await bot._handle_room_message(room, event1)
        assert agent.handle_input.call_count == 1
        
        # 3rd member joins
        room.users["@newcomer:matrix.local"] = MagicMock()
        
        # Second message: 3 members, gated, no mention → skipped
        agent.handle_input.reset_mock()
        event2 = make_event("@alice:matrix.local", "Anyone there?", event_id="$e2")
        await bot._handle_room_message(room, event2)
        agent.handle_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_transition_3_to_2_deactivates_gating(self, tmp_path):
        """Room starts with 3, member leaves → all messages processed."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!room:matrix.local", 3)
        
        # First message: 3 members, gated, no mention → skipped
        event1 = make_event("@alice:matrix.local", "Hello", event_id="$e1")
        await bot._handle_room_message(room, event1)
        agent.handle_input.assert_not_called()
        
        # Member leaves
        first_key = next(iter(room.users))
        del room.users[first_key]
        
        # Second message: 2 members, no gating → processed
        event2 = make_event("@alice:matrix.local", "Now it's a DM", event_id="$e2")
        bot._active_rooms.add("!room:matrix.local")
        await bot._handle_room_message(room, event2)
        agent.handle_input.assert_called_once()


# --- Context Hydration ---


class TestContextHydration:
    """On mention in gated room, agent sees all buffered messages."""

    @pytest.mark.asyncio
    async def test_buffered_messages_available_on_mention(self, tmp_path):
        """3 non-mentioned msgs buffered, 4th mentions agent → all 4 in context."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        bot._active_rooms.add("!group:matrix.local")
        
        # Send 3 non-mentioned messages (buffered, not processed)
        for i in range(3):
            event = make_event(
                "@alice:matrix.local",
                f"Message {i+1}",
                event_id=f"$e{i+1}",
            )
            await bot._handle_room_message(room, event)
        
        assert agent.handle_input.call_count == 0
        
        # 4th message mentions the agent
        event4 = make_event(
            "@alice:matrix.local",
            "@watson what do you think about all that?",
            event_id="$e4",
            mentions_user_ids=["@watson:matrix.local"],
        )
        await bot._handle_room_message(room, event4)
        
        assert agent.handle_input.call_count == 1
        
        # Verify session log has all 4 messages
        entries = bot.session_log.read("!group:matrix.local")
        user_entries = [e for e in entries if e["role"] == "user"]
        assert len(user_entries) == 4
        assert user_entries[0]["content"] == "Message 1"
        assert user_entries[3]["content"] == "@watson what do you think about all that?"


# --- Tool Notice Non-Triggering ---


class TestToolNoticeNonTrigger:
    """m.notice messages should not trigger _handle_room_message.
    
    This is verified structurally: the callback is registered for
    RoomMessageText only, and m.notice is RoomMessageNotice.
    We verify the imports are correct.
    """

    def test_nio_event_types_are_distinct(self):
        """RoomMessageText and RoomMessageNotice are different classes in nio."""
        from nio import RoomMessageText, RoomMessageNotice
        assert RoomMessageText is not RoomMessageNotice

    def test_callback_registered_for_text_only(self, tmp_path):
        """MatrixBot registers callback for RoomMessageText, not RoomMessageNotice."""
        # Inspect the source to verify — this is a structural test
        import inspect
        from openalph.matrix import MatrixBot
        source = inspect.getsource(MatrixBot.start)
        assert "RoomMessageText" in source
        assert "RoomMessageNotice" not in source


# --- Self-Message and Pre-Sync Guards ---


class TestExistingGuards:
    """Gating doesn't break existing pre-sync and self-message guards."""

    @pytest.mark.asyncio
    async def test_pre_sync_still_skipped(self, tmp_path):
        """Messages before initial sync → skipped (regardless of mention)."""
        bot, agent = make_bot(tmp_path)
        bot._synced = False
        
        room = make_room("!group:matrix.local", 3)
        event = make_event(
            "@alice:matrix.local",
            "@watson help",
            mentions_user_ids=["@watson:matrix.local"],
        )
        
        await bot._handle_room_message(room, event)
        
        agent.handle_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_self_message_still_skipped(self, tmp_path):
        """Own messages → skipped (regardless of gating)."""
        bot, agent = make_bot(tmp_path)
        room = make_room("!group:matrix.local", 3)
        event = make_event("@watson:matrix.local", "My own message")
        
        await bot._handle_room_message(room, event)
        
        agent.handle_input.assert_not_called()
