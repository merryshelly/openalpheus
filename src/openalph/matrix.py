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
from pathlib import Path
from nio import (
    AsyncClient,
    InviteMemberEvent,
    MessageDirection,
    RoomMessageText,
    LoginResponse,
    RoomSendError,
)

from openalph.agent import ContextOverflowError as AgentOverflowError
from openalph.config import MatrixConfig
from openalph.session import SessionLog
from openalph.mention import mentions_me, is_gated, MentionCheckResult
from openalph.heartbeat import HeartbeatManager, parse_interval, format_interval

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
        self.heartbeat = HeartbeatManager(
            config_path=Path(agent.config.workspace) / "heartbeats.json",
            callback=self._inject_heartbeat,
        )

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
            response = await self.client.login(self.config.password)
            if not isinstance(response, LoginResponse):
                raise RuntimeError(f"Matrix login failed: {response}")

    async def _set_typing(self, room_id: str, state: bool):
        """Set typing indicator for a room.

        Args:
            room_id: Matrix room ID
            state: True for typing ON, False for typing OFF
        """
        await self.client.room_typing(room_id, typing_state=state)

    async def _room_send_with_retry(
        self,
        room_id: str,
        content: dict,
        *,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ):
        """Send a message to a room with retry on transient failures.

        Retries on RoomSendError responses and network exceptions with
        exponential backoff. Raises on permanent failures or exhaustion.

        Args:
            room_id: Matrix room ID
            content: Message content dict (must include msgtype)
            max_attempts: Maximum send attempts (default 3)
            base_delay: Initial backoff delay in seconds (doubles each retry)
        """
        delay = base_delay
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.client.room_send(
                    room_id,
                    "m.room.message",
                    content,
                )
                if isinstance(response, RoomSendError):
                    raise RuntimeError(
                        f"Matrix send failed: {response.status_code} "
                        f"{response.message}"
                    )
                return response
            except Exception as exc:
                last_error = exc
                if attempt < max_attempts:
                    logger.warning(
                        "Send to %s failed (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        room_id,
                        attempt,
                        max_attempts,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2

        raise RuntimeError(
            f"Failed to send to {room_id} after {max_attempts} attempts"
        ) from last_error

    async def send_notice(self, room_id: str, text: str):
        """Send a notice (tool visibility) to a room.

        Notices are visually distinct from regular messages in most clients.
        Retries on transient failures with exponential backoff.
        """
        content = {
            "msgtype": "m.notice",
            "body": text,
        }
        await self._room_send_with_retry(room_id, content)

    async def send(self, room_id: str, text: str):
        """Send a text message to a room.

        Args:
            room_id: Matrix room ID
            text: Message content (markdown supported)

        Retries on transient failures with exponential backoff.
        """
        content = {
            "msgtype": "m.text",
            "body": text,
            "format": "org.matrix.custom.html",
            "formatted_body": text,  # Simplified; could add markdown->HTML conversion
        }
        await self._room_send_with_retry(room_id, content)

    async def _cancel_current(self):
        """Cancel any current in-flight work.

        - Cancel in-flight LLM call
        - Kill tool subprocesses
        - Send cancellation notice to room
        """
        task = None
        if hasattr(self.agent, "cancel"):
            task = self.agent.cancel()

        # Wait for the cancelled task to actually finish
        if task:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass  # Task may raise other errors during cancellation

        if self._current_room:
            await self.send(self._current_room, "Cancelled.")
            await self._set_typing(self._current_room, False)

    async def _inject_heartbeat(self, room_id: str) -> None:
        """Process a heartbeat as if the agent received a wake message."""
        heartbeat_content = "Heartbeat: execute your WAKE instructions."

        # Log to session JSONL
        if self.session_log:
            self.session_log.append(
                role="system",
                sender=self.config.user_id,
                room=room_id,
                event_id=None,
                content=heartbeat_content,
                source="heartbeat",
            )

        # Activate room if not already active
        if room_id not in self._active_rooms:
            await self._activate_room(room_id)

        # Process through agent
        try:
            await self._set_typing(room_id, True)
            response = await self.agent.handle_input(heartbeat_content, room_id)
            if self.session_log:
                self.session_log.append(
                    role="assistant",
                    sender=self.config.user_id,
                    room=room_id,
                    event_id=None,
                    content=response,
                )
            await self.send(room_id, response)
        except AgentOverflowError as e:
            logger.warning("Context overflow in %s — auto-stopping heartbeat", room_id)
            await self.heartbeat.stop(room_id)
            await self.send(
                room_id,
                "⚠️ **Context overflow** — heartbeat auto-stopped for this room. "
                "Start a new room to continue.",
            )
        except Exception as e:
            logger.exception("Heartbeat processing error in %s", room_id)
            await self.send(room_id, "⚠️ Heartbeat error — check agent logs for details.")
        finally:
            await self._set_typing(room_id, False)

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
                        # Gap-fill: page backward through recent Matrix messages until
                        # we find overlap with known session history (or hit the cap).
                        GAP_FILL_MAX = 500  # safety cap to avoid infinite paging
                        known_ids = {e.get("event_id") for e in existing if e.get("event_id")}
                        new_messages = []
                        start_token = None  # None = start from current position
                        found_overlap = False

                        while len(new_messages) < GAP_FILL_MAX:
                            response = await self.client.room_messages(
                                room_id, start=start_token, limit=100,
                                direction=MessageDirection.back,
                            )
                            if not response.chunk:
                                break  # no more history

                            for msg in response.chunk:  # newest-first order
                                ev_id = getattr(msg, 'event_id', None)
                                if ev_id and ev_id in known_ids:
                                    found_overlap = True
                                    break
                                if hasattr(msg, 'body') and msg.sender != self.config.user_id:
                                    new_messages.append(msg)

                            if found_overlap or not response.end:
                                break
                            start_token = response.end

                        if not found_overlap and new_messages:
                            logger.warning(
                                "Gap-fill for %s: no overlap found after %d messages — "
                                "some history may be missing",
                                room_id, len(new_messages),
                            )

                        # Append in chronological order (we collected newest-first)
                        for msg in reversed(new_messages):
                            ev_id = getattr(msg, 'event_id', None)
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
            LEGACY_HISTORY_MAX = 500  # safety cap to avoid OOM on large rooms
            all_messages = []
            response = await self.client.room_messages(
                room_id, start=None, limit=100, direction=MessageDirection.back,
            )
            all_messages.extend(response.chunk)
            while response.end and len(all_messages) < LEGACY_HISTORY_MAX:
                response = await self.client.room_messages(
                    room_id, start=response.end, limit=100, direction=MessageDirection.back,
                )
                if not response.chunk:
                    break
                all_messages.extend(response.chunk)
            if len(all_messages) >= LEGACY_HISTORY_MAX:
                logger.warning("Legacy history load for %s capped at %d messages", room_id, LEGACY_HISTORY_MAX)

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
        logger.debug("Event received: room=%s sender=%s body=%r synced=%s",
                      room.room_id, event.sender, getattr(event, 'body', '')[:50], self._synced)

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

            if body == "/showprompt":
                parts = [self.agent.system_prompt]
                # Append tool list
                if self.agent.tools:
                    parts.append("\n## Available Tools (passed via API, not in prompt)\n")
                    for tool in self.agent.tools:
                        params = ", ".join(tool.parameters.get("properties", {}).keys())
                        line = f"- **{tool.name}**: {tool.description}"
                        if params:
                            line += f"\n  Parameters: {params}"
                        parts.append(line)
                output = "\n".join(parts)
                if len(output) > 15000:
                    output = output[:15000] + "\n\n... (truncated)"
                await self.send(room_id, f"```\n{output}\n```")
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

            if body.startswith("/heartbeat"):
                parts = body.split()
                if len(parts) >= 3 and parts[1] == "start":
                    interval = parse_interval(parts[2])
                    if interval is None:
                        await self.send(room_id, "Invalid interval. Use e.g. `15m`, `1h`, `6h`.")
                    elif interval < 300:
                        await self.send(room_id, "Minimum interval is 5m.")
                    else:
                        await self.heartbeat.start(room_id, interval)
                        human = format_interval(interval)
                        await self.send(room_id, f"Heartbeat started: every {human} in this room.")
                elif len(parts) >= 2 and parts[1] == "stop":
                    stopped = await self.heartbeat.stop(room_id)
                    if stopped:
                        await self.send(room_id, "Heartbeat stopped.")
                    else:
                        await self.send(room_id, "No heartbeat active in this room.")
                elif len(parts) >= 2 and parts[1] == "status":
                    entries = self.heartbeat.status()
                    if not entries:
                        await self.send(room_id, "No active heartbeats.")
                    else:
                        lines = ["Active heartbeats:"]
                        for e in entries:
                            lines.append(f"  {e.room_id} — every {format_interval(e.interval_seconds)} (next: {format_interval(e.seconds_until_next)})")
                        await self.send(room_id, "\n".join(lines))
                else:
                    await self.send(room_id, "Usage: `/heartbeat start <interval>` | `/heartbeat stop` | `/heartbeat status`")
                return

            # --- Mention gating ---
            gated = is_gated(self.config, room)

            if gated:
                event_source = getattr(event, 'source', {}) or {}
                mention = mentions_me(self.config.user_id, event_source, body)

                # Always buffer to session log
                session_log = getattr(self, 'session_log', None)
                if session_log:
                    session_log.append(
                        role="user",
                        sender=event.sender,
                        room=room_id,
                        event_id=getattr(event, 'event_id', None),
                        content=body,
                        mentioned=mention.mentioned,
                    )

                if not mention.mentioned:
                    logger.debug("Gated room %s: not mentioned (%s), skipping",
                                 room_id, mention.method)
                    return

                logger.info("Gated room %s: mentioned via %s, processing",
                             room_id, mention.method)

                # Context hydration: reload history from session_log to include
                # all buffered non-mentioned messages since last response
                if room_id in self._active_rooms and self.session_log:
                    history = self.agent.history(room_id)
                    history.clear()
                    history.extend(self.session_log.build_context(room_id))
                    logger.info("Hydrated context for %s: %d entries", room_id, len(history))
            # --- End mention gating ---

            # Lazy wake: activate room on first live message
            if room_id not in self._active_rooms:
                room_name = getattr(room, 'name', '') or getattr(room, 'display_name', '') or room_id
                await self._activate_room(room_id, room_name=room_name)

            # Append user message to session log (only for ungated rooms — gated already buffered above)
            if not gated:
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

            try:
                response = await self.agent.handle_input(
                    body, room_id,
                    on_tool_call=_tool_notice,
                    on_tool_intent=_tool_intent,
                )
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
                # Agent error: send generic message to avoid leaking exception details
                logger.exception("Agent error processing message in %s", room_id)
                await self.send(room_id, "⚠️ Internal error — check agent logs for details.")
            finally:
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
        import time as _time
        _sync_start = _time.monotonic()
        await self.client.sync(timeout=self.config.sync_timeout)
        _sync_ms = (_time.monotonic() - _sync_start) * 1000
        self._synced = True
        joined = len(self.client.rooms) if hasattr(self.client, 'rooms') else '?'
        logger.info("Initial sync complete in %.0fms (%s rooms joined, lazy wake active)", _sync_ms, joined)

        # Resume persisted heartbeats
        await self.heartbeat.resume()

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
        await self.heartbeat.shutdown()
        await self.client.close()
