"""Matrix integration for OpenAlph agents.

Connects an Agent to a Matrix room via matrix-nio. Handles:
- Login (password or access token)
- Message routing (skip own messages, commands, regular messages)
- Typing indicator management
- History loading with context overflow detection
- Graceful error handling
"""

import asyncio
import logging
from nio import AsyncClient, InviteMemberEvent, RoomMessageText

from openalph.config import MatrixConfig

logger = logging.getLogger(__name__)


class ContextOverflowError(Exception):
    """Raised when room history exceeds model context capacity.

    This is a hard error, not a graceful degradation.
    Operator must switch to a larger model or start a new room.
    """

    def __init__(self, room_id: str, history_tokens: int, max_tokens: int):
        self.room_id = room_id
        self.history_tokens = history_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"Room {room_id}: history ({history_tokens} tokens) exceeds "
            f"model capacity ({max_tokens} - {max_tokens - history_tokens} reserve). "
            f"Switch to a larger model or start a new room."
        )


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
        self._running = False
        self._current_room = None

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

    def _load_history_into_agent(self, messages: list, room_id: str):
        """Load room messages into agent history.

        Converts Matrix events to agent message format.
        Raises ContextOverflowError if history exceeds model capacity.

        Args:
            messages: List of Matrix room message events
            room_id: Matrix room ID

        Raises:
            ContextOverflowError: If history tokens exceed available capacity
        """
        # Calculate available capacity
        model_max = self.agent.config.model_max_tokens
        reserve = self.config.context_reserve
        system_tokens = self._estimate_tokens(self.agent.system_prompt)
        available = model_max - reserve - system_tokens

        # Convert messages to agent format and estimate tokens
        history = []
        total_tokens = 0

        for msg in messages:
            # Determine role based on sender
            if msg.sender == self.config.user_id:
                role = "assistant"
            else:
                role = "user"

            content = msg.body
            history.append({"role": role, "content": content})
            total_tokens += self._estimate_tokens(content)

        # Check for overflow
        if total_tokens > available:
            raise ContextOverflowError(room_id, total_tokens, model_max)

        # Load into agent
        self.agent.history = history

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
        - Skip own messages
        - /stop → cancel current work
        - /status → post agent status
        - Otherwise → process through agent

        Args:
            room: Matrix room object
            event: Room message event
        """
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
                status = self.agent.status()
                # Format status message
                lines = [
                    f"**{status['name']}**",
                    f"Model: {status['model']}",
                    f"Turns: {status['turns']}",
                    f"Input tokens: {status['total_input_tokens']}",
                    f"Output tokens: {status['total_output_tokens']}",
                    f"Tool calls: {status['total_tool_calls']}",
                ]
                await self.send(room_id, "\n".join(lines))
                return

            # Regular message: process through agent
            await self._set_typing(room_id, True)

            try:
                response = await self.agent.handle_input(body)
                await self.send(room_id, response)
            except Exception as e:
                # Agent error: send error message, don't crash
                logger.exception("Agent error processing message")
                await self.send(room_id, f"Error: {e}")
            finally:
                await self._set_typing(room_id, False)

        finally:
            self._current_room = None

    async def start(self):
        """Start the Matrix bot.

        1. Login
        2. Load room history
        3. Start sync loop
        """
        await self._login()
        self._running = True

        # Register event callbacks
        self.client.add_event_callback(self._handle_room_message, RoomMessageText)
        self.client.add_event_callback(self._handle_invite, InviteMemberEvent)

        # Initial sync to get rooms
        await self.client.sync(timeout=self.config.sync_timeout)

        # TODO: Load history for each joined room
        # This would require room_messages() calls and pagination
        # For MVP, history is loaded on-demand or skipped

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
