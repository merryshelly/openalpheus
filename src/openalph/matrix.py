"""Matrix integration for OpenAlph agents.

Connects an Agent to a Matrix room via matrix-nio. Handles:
- Login (password or access token)
- Message routing (skip own messages, commands, regular messages)
- Typing indicator management
- Lazy room activation (history loaded on first live message)
- Graceful error handling
"""

import asyncio
import logging
from nio import AsyncClient, InviteMemberEvent, RoomMessageText

from openalph.agent import ContextOverflowError as AgentOverflowError
from openalph.config import MatrixConfig
from openalph.session import SessionLog

logger = logging.getLogger(__name__)


class MatrixBot:
    """Matrix client wrapping an OpenAlph Agent.

    Public interface:
        start() — login, load history, start sync loop
        stop() — cancel work, shutdown
        send(room_id, text) — send message to room
    """

    def __init__(self, agent, config: MatrixConfig):
        """Initialize with an Agent and Matrix config.

        Args:
            agent: An OpenAlph Agent instance
            config: MatrixConfig with connection details
        """
        self.agent = agent
        self.config = config
        self.client = AsyncClient(config.homeserver, config.user_id, config.device_id)
        self.session_log = SessionLog(agent.config.workspace, config.user_id)
        self._running = False
        self._synced = False
        self._current_room = None
        self._active_rooms = set()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Conservative estimate: 1 token ≈ 4 characters.
        """
        return len(text) // 4

    async def _login(self):
        """Login to Matrix server.

        If access_token is provided, sets it directly.
        Otherwise, uses password login.
        """
        if self.config.access_token:
            # Set access token directly, skip password login
            self.client.access_token = self.config.access_token
        else:
            # Password login
            await self.client.login(self.config.password)

    async def _set_typing(self, room_id: str, state: bool):
        """Set typing indicator for a room.

        Args:
            room_id: Matrix room ID
            state: True for typing ON, False for typing OFF
        """
        await self.client.room_typing(room_id, typing_state=state)

    async def send_notice(self, room_id: str, text: str):
        """Send a notice (tool visibility) to a room.

        Notices are visually distinct from regular messages in most clients.
        """
        content = {
            "msgtype": "m.notice",
            "body": text,
        }
        await self.client.room_send(
            room_id,
            "m.room.message",
            content,
        )

    async def send(self, room_id: str, text: str):
        """Send a text message to a room.

        Args:
            room_id: Matrix room ID
            text: Message content (markdown supported)
        """
        content = {
            "msgtype": "m.text",
            "body": text,
            "format": "org.matrix.custom.html",
            "formatted_body": text,  # Simplified; could add markdown->HTML conversion
        }
        await self.client.room_send(
            room_id,
            "m.room.message",
            content,
        )

    async def _cancel_current(self):
        """Cancel any current in-flight work.

        - Cancel in-flight LLM call
        - Kill tool subprocesses
        - Send cancellation notice to room
        """
        if hasattr(self.agent, "cancel"):
            self.agent.cancel()

        if self._current_room:
            await self.send(self._current_room, "Cancelled.")
            await self._set_typing(self._current_room, False)

    async def _activate_room(self, room_id: str, room_name: str = ""):
        """Load session history on first message (lazy wake).

        New architecture: reads local JSONL instead of paginating Matrix history.
        If entries exist, does gap-fill from last known event. If new room, starts fresh.
        """
        session_log = getattr(self, 'session_log', None)

        if session_log:
            existing = session_log.read(room_id)

            if existing:
                # Gap-fill: fetch Matrix messages since the last known event
                last_id = session_log.last_event_id(room_id)
                if last_id:
                    try:
                        # Fetch recent messages to check for gaps
                        response = await self.client.room_messages(room_id, start="", limit=100)
                        recent = list(response.chunk)
                        # Recent messages come in reverse order (newest first)
                        # Find the cutoff: append only events after last_id
                        known_ids = {e.get("event_id") for e in existing if e.get("event_id")}
                        for msg in reversed(recent):
                            if not hasattr(msg, 'body'):
                                continue
                            ev_id = getattr(msg, 'event_id', None)
                            if ev_id and ev_id not in known_ids and msg.sender != self.config.user_id:
                                session_log.append(
                                    role="user",
                                    sender=msg.sender,
                                    room=room_id,
                                    event_id=ev_id,
                                    content=msg.body,
                                )
                        session_log.append(
                            role="system",
                            sender=self.agent_user_id if hasattr(self, 'agent_user_id') else self.config.user_id,
                            room=room_id,
                            event="session_resume",
                            detail=f"Resumed session with {len(existing)} prior entries",
                        )
                    except Exception as e:
                        logger.warning("Gap-fill failed for %s: %s", room_id, e)

                # Restore context from session log
                history = self.agent.history(room_id)
                history.clear()
                history.extend(session_log.build_context(room_id))
            else:
                # New room: start fresh, log session start
                session_log.append(
                    role="system",
                    sender=self.config.user_id,
                    room=room_id,
                    event="session_start",
                    detail="New room, starting fresh",
                )
        else:
            # Fallback: paginate Matrix history (legacy path for tests without session_log)
            all_messages = []
            response = await self.client.room_messages(room_id, start="", limit=100)
            all_messages.extend(response.chunk)
            while response.end:
                response = await self.client.room_messages(room_id, start=response.end, limit=100)
                all_messages.extend(response.chunk)

            all_messages.reverse()

            history = self.agent.history(room_id)
            for msg in all_messages:
                if not hasattr(msg, 'body'):
                    continue
                if msg.sender == self.config.user_id:
                    role = "assistant"
                else:
                    role = "user"
                history.append({"role": role, "content": msg.body})

        self._active_rooms.add(room_id)

    async def _handle_invite(self, room, event):
        """Auto-join rooms on invite.

        Args:
            room: Matrix room object (invited room)
            event: InviteMemberEvent
        """
        if event.state_key == self.config.user_id:
            logger.info("Invited to %s by %s — joining", room.room_id, event.sender)
            await self.client.join(room.room_id)

    async def _handle_room_message(self, room, event):
        """Handle a room message event.

        Routes messages:
        - Skip events from initial sync (lazy wake: no hydration)
        - Skip own messages
        - /stop → cancel current work
        - /status → post agent status
        - Otherwise → activate room if needed, process through agent

        Args:
            room: Matrix room object
            event: Room message event
        """
        # During initial sync, don't hydrate rooms — lazy wake on first live message
        if not self._synced:
            return

        # Skip own messages
        if event.sender == self.config.user_id:
            return

        room_id = room.room_id
        self._current_room = room_id

        try:
            # Check for commands
            body = event.body.strip()

            if body == "/stop":
                await self._cancel_current()
                return

            if body == "/status":
                status = self.agent.status(room_id)
                # Format status message
                ctx = status['context_tokens']
                ctx_max = status['context_max']
                ctx_pct = status['context_pct']
                lines = [
                    f"**{status['name']}**",
                    f"Model: {status['model']}",
                    f"Context: ~{ctx:,} / {ctx_max:,} tokens ({ctx_pct}%)",
                    f"Turns: {status['turns']}",
                    f"Cumulative: {status['total_input_tokens']:,} in / {status['total_output_tokens']:,} out",
                    f"Tool calls: {status['total_tool_calls']}",
                ]
                await self.send(room_id, "\n".join(lines))
                return

            # Lazy wake: activate room on first live message
            if room_id not in self._active_rooms:
                room_name = getattr(room, 'name', '') or getattr(room, 'display_name', '') or room_id
                await self._activate_room(room_id, room_name=room_name)

            # Append user message to session log
            session_log = getattr(self, 'session_log', None)
            if session_log:
                session_log.append(
                    role="user",
                    sender=event.sender,
                    room=room_id,
                    event_id=getattr(event, 'event_id', None),
                    content=body,
                )

            # Regular message: process through agent
            await self._set_typing(room_id, True)

            # Wire tool visibility for this turn
            async def _tool_notice(call_id, name, input_data, result, is_error):
                # Send abbreviated notice to Matrix room
                notice_body = f"🔧 {name}: {str(result)[:200]}"
                try:
                    await self.send_notice(room_id, notice_body)
                except Exception:
                    pass
                # Append tool result to session log
                _sl = getattr(self, 'session_log', None)
                if _sl:
                    _sl.append(
                        role="tool",
                        sender=self.config.user_id,
                        room=room_id,
                        event_id=None,
                        call_id=call_id,
                        name=name,
                        output=result,
                        is_error=is_error,
                    )

            # Wire tool intent logging (fires before tool execution)
            async def _tool_intent(tool_calls, content):
                _sl = getattr(self, 'session_log', None)
                if _sl:
                    _sl.append(
                        role="assistant",
                        sender=self.config.user_id,
                        room=room_id,
                        event_id=None,
                        content=content or "",
                        tool_calls=[
                            {"call_id": tc.id, "name": tc.name, "input": tc.input}
                            for tc in tool_calls
                        ],
                    )

            self.agent._on_tool_call = _tool_notice
            self.agent._on_tool_intent = _tool_intent

            try:
                response = await self.agent.handle_input(body, room_id)
                # Append assistant response to session log
                if session_log:
                    session_log.append(
                        role="assistant",
                        sender=self.config.user_id,
                        room=room_id,
                        event_id=None,
                        content=response,
                    )
                await self.send(room_id, response)
            except AgentOverflowError as e:
                logger.warning("Context overflow in %s: %s", room_id, e)
                await self.send(room_id,
                    f"⚠️ **Context overflow** — ~{e.current_tokens:,} / "
                    f"{e.max_tokens:,} tokens. Start a new room to continue.")
            except Exception as e:
                # Agent error: send error message, don't crash
                logger.exception("Agent error processing message")
                await self.send(room_id, f"Error: {e}")
            finally:
                self.agent._on_tool_call = None
                self.agent._on_tool_intent = None
                await self._set_typing(room_id, False)

        finally:
            self._current_room = None

    async def start(self):
        """Start the Matrix bot.

        1. Login
        2. Start sync loop (rooms hydrate on first message via lazy wake)
        """
        await self._login()
        self._running = True

        # Register event callbacks before initial sync so history events
        # are captured. _synced=False tells the handler to load them as
        # context instead of responding.
        self.client.add_event_callback(self._handle_room_message, RoomMessageText)
        self.client.add_event_callback(self._handle_invite, InviteMemberEvent)

        # Initial sync: populates rooms and loads timeline history via callback
        await self.client.sync(timeout=self.config.sync_timeout)
        self._synced = True
        logger.info("Initial sync complete (lazy wake: rooms will hydrate on first message)")

        # Sync loop
        delay = self.config.retry_base
        while self._running:
            try:
                await self.client.sync(timeout=self.config.sync_timeout)
                delay = self.config.retry_base  # Reset on success
            except Exception as e:
                logger.warning(f"Sync failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.retry_max)

    async def stop(self):
        """Stop the Matrix bot gracefully.

        1. Cancel any in-flight work
        2. Close Matrix client
        """
        self._running = False
        await self._cancel_current()
        await self.client.close()
