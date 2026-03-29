"""Matrix integration for OpenAlph agents.

Connects an Agent to a Matrix room via matrix-nio. Handles:
- Login (password or access token)
- Message routing (skip own messages, commands, regular messages)
- Typing indicator management
- Lazy room activation (history loaded on first live message)
- Graceful error handling
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path

import mistune
from nio import (
    AsyncClient,
    DownloadError,
    InviteMemberEvent,
    MessageDirection,
    RoomMessageFile,
    RoomMessageImage,
    RoomMessageAudio,
    RoomMessageText,
    RoomMessageVideo,
    LoginResponse,
    RoomSendError,
    UploadError,
)

# Constants for media handling
MAX_MEDIA_BYTES = 20_000_000  # 20 MB
MEDIA_DIR = "media"

from openalph.agent import ContextOverflowError as AgentOverflowError
from openalph.provider import ProviderError
from openalph.config import MatrixConfig
from openalph.session import SessionLog
from openalph.mention import mentions_me, is_gated, strip_mention, MentionCheckResult
from openalph.heartbeat import HeartbeatManager, parse_interval, format_interval
from openalph.umbral import UmbralManager

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename for safe filesystem storage.

    Strips path separators and null bytes, truncates to 200 chars,
    and falls back to 'attachment' if empty or only separators.
    """
    # Strip path separators and null bytes
    name = name.replace("/", "").replace("\\", "").replace("\0", "")
    # Truncate
    name = name[:200]
    # Fallback if empty
    return name or "attachment"


_STREAMING_CURSOR = " ▍"


def _is_streaming_edit(event) -> bool:
    """Return True if a Matrix event is an intermediate streaming edit.

    Streaming delivery sends an initial message, edits it progressively
    (each with a trailing cursor ▍), then sends a final cursor-free edit.

    In multi-agent rooms, receivers should:
      - Skip intermediate edits (cursor present)
      - Skip the initial partial send (cursor present, not an edit)
      - Accept the final edit (cursor absent) — this carries the full content

    Returns True for events that should be **dropped** (intermediate edits
    and initial cursor-bearing sends from other agents).
    """
    source = getattr(event, 'source', None) or {}
    content = source.get("content", {})
    relates_to = content.get("m.relates_to", {})
    is_edit = relates_to.get("rel_type") == "m.replace"

    body = getattr(event, 'body', '') or ''

    if is_edit:
        # Edit event — accept only the final edit (no cursor)
        new_body = content.get("m.new_content", {}).get("body", "")
        return new_body.endswith(_STREAMING_CURSOR)
    else:
        # Original send — skip if it has the streaming cursor
        return body.endswith(_STREAMING_CURSOR)


def _event_id_hash(event_id: str) -> str:
    """Create a filesystem-safe hash from a Matrix event ID.

    Returns the first 16 characters of the SHA-256 hex digest.
    """
    return hashlib.sha256(event_id.encode()).hexdigest()[:16]

class StreamingDelivery:
    """Manages progressive message delivery via Matrix message edits."""

    EDIT_INTERVAL_MS = 600     # min ms between edits
    EDIT_MIN_CHARS = 60        # min new chars before edit
    INITIAL_SEND_CHARS = 40    # chars before first send

    def __init__(self, bot: 'MatrixBot', room_id: str):
        self.bot = bot
        self.room_id = room_id
        self._buffer = ""
        self._event_id: str | None = None
        self._last_edit_time: float = 0
        self._last_edit_len: int = 0
        self._delivered = False
        self._delivered_text = ""  # last finalized content, for duplicate detection

    async def push(self, delta: str, done: bool = False):
        """Accept a text delta and manage delivery."""
        # Accumulate
        self._buffer += delta

        if done:
            await self._finalize()
            # Reset state for next tool loop iteration
            self._event_id = None
            self._buffer = ""
            self._last_edit_time = 0
            self._last_edit_len = 0
            return

        # If no initial message sent yet
        if self._event_id is None:
            if len(self._buffer) >= self.INITIAL_SEND_CHARS:
                await self._send_initial()
        else:
            # Initial sent - check if we should edit
            chars_since_edit = len(self._buffer) - self._last_edit_len
            time_since_edit = (time.monotonic() * 1000) - self._last_edit_time

            if chars_since_edit >= self.EDIT_MIN_CHARS and time_since_edit >= self.EDIT_INTERVAL_MS:
                await self._edit()

    async def _send_initial(self):
        """Send initial message with cursor indicator."""
        display = self._buffer + " ▍"
        content = {
            "msgtype": "m.text",
            "body": display,
            "format": "org.matrix.custom.html",
            "formatted_body": mistune.html(display),
        }
        resp = await self.bot._room_send_with_retry(self.room_id, content)
        self._event_id = resp.event_id
        self._last_edit_time = time.monotonic() * 1000
        self._last_edit_len = len(self._buffer)

    async def _edit(self):
        """Edit existing message with new content."""
        display = self._buffer + " ▍"
        content = {
            "msgtype": "m.text",
            "body": f"* {display}",
            "m.new_content": {
                "msgtype": "m.text",
                "body": display,
                "format": "org.matrix.custom.html",
                "formatted_body": mistune.html(display),
            },
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": self._event_id,
            },
        }
        try:
            await self.bot.client.room_send(self.room_id, "m.room.message", content)
            self._last_edit_time = time.monotonic() * 1000
            self._last_edit_len = len(self._buffer)
        except Exception as exc:
            logger.warning("Edit failed (non-critical): %s", exc)

    async def _finalize(self):
        """Finalize message delivery - remove cursor, handle long messages."""
        if not self._buffer:
            return

        self._delivered = True
        self._delivered_text = self._buffer

        if self._event_id is None:
            # Never sent initial - use normal send
            await self.bot.send(self.room_id, self._buffer)
            return

        # Check if message needs splitting
        if len(self._buffer) > MatrixBot.MAX_MESSAGE_CHARS:
            chunks = MatrixBot._split_message(self._buffer)
            # Edit first chunk into existing message
            await self._edit_final(chunks[0])
            # Send remaining chunks
            for chunk in chunks[1:]:
                await self.bot.send(self.room_id, chunk)
        else:
            await self._edit_final(self._buffer)

    async def _edit_final(self, text: str):
        """Send final edit without cursor, with retry."""
        content = {
            "msgtype": "m.text",
            "body": f"* {text}",
            "m.new_content": {
                "msgtype": "m.text",
                "body": text,
                "format": "org.matrix.custom.html",
                "formatted_body": mistune.html(text),
            },
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": self._event_id,
            },
        }
        await self.bot._room_send_with_retry(self.room_id, content)


class MatrixBot:
    """Matrix client wrapping an OpenAlph Agent.

    Public interface:
        start() — login, load history, start sync loop
        stop() — cancel work, shutdown
        send(room_id, text) — send message to room
    """

    def __init__(self, agent, config: MatrixConfig = None):
        """Initialize with an Agent and Matrix config.

        Args:
            agent: An OpenAlph Agent instance
            config: MatrixConfig with connection details
        """
        self.agent = agent
        self.config = config
        self.client = AsyncClient(config.homeserver, config.user_id, config.device_id)
        # Handle test case where agent might not have .config yet
        workspace = getattr(getattr(agent, 'config', None), 'workspace', None)
        if workspace:
            self.session_log = SessionLog(workspace, config.user_id)
            self.heartbeat = HeartbeatManager(
                config_path=Path(workspace) / "heartbeats.json",
                callback=self._inject_heartbeat,
            )
            self.umbral = UmbralManager(
                config_path=Path(workspace) / "umbral.json",
                callback=self._inject_umbral,
            )
        else:
            self.session_log = None
            self.heartbeat = None
            self.umbral = None
        self._running = False
        self._synced = False
        self._current_room = None
        self._active_rooms = set()
        self._halted_rooms: set[str] = set()
        self._room_thinking = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._session_locks: dict[str, asyncio.Lock] = {}

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
            # Accept LoginResponse or duck-typed mocks with required attributes
            if not isinstance(response, LoginResponse) and not hasattr(response, 'access_token'):
                raise RuntimeError(f"Matrix login failed: {response}")

    async def _set_typing(self, room_id: str, state: bool):
        """Set typing indicator for a room.

        Args:
            room_id: Matrix room ID
            state: True for typing ON, False for typing OFF
        """
        try:
            # Pass state as positional arg for test compatibility
            await self.client.room_typing(room_id, state)
        except Exception:
            # Silently ignore typing errors (e.g., in tests without full mocking)
            pass

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

    async def upload_and_send(
        self,
        room_id: str,
        file_path: Path,
        content_type: str,
        filename: str,
        caption: str | None = None,
    ):
        """Upload a file to Matrix and send it to a room."""
        file_size = file_path.stat().st_size

        with open(file_path, "rb") as f:
            response, _ = await self.client.upload(
                f,
                content_type=content_type,
                filename=filename,
                filesize=file_size,
            )

        if isinstance(response, UploadError):
            raise RuntimeError(f"Upload failed: {response.message}")

        mxc_uri = response.content_uri

        major = content_type.split("/")[0]
        msgtype_map = {"audio": "m.audio", "image": "m.image", "video": "m.video"}
        msgtype = msgtype_map.get(major, "m.file")

        content = {
            "msgtype": msgtype,
            "url": mxc_uri,
            "body": caption or filename,
            "info": {
                "mimetype": content_type,
                "size": file_size,
            },
        }

        if caption:
            content["filename"] = filename

        await self._room_send_with_retry(room_id, content)

    # Matrix PDU limit is 65535 bytes.  HTML formatting roughly doubles
    # the size of markdown, so we split at 25K *characters* to stay well
    # under the wire-format limit in all cases.
    MAX_MESSAGE_CHARS = 25_000

    @staticmethod
    def _split_message(text: str, limit: int | None = None) -> list[str]:
        """Split a long message into chunks that fit within the PDU limit.

        Splits on paragraph boundaries (double-newline) first, then on
        single newlines, falling back to a hard cut only if a single
        unbreakable block exceeds the limit.

        Each chunk except the last gets a continuation footer and each
        chunk except the first gets a continuation header so readers
        know the message was split.

        Returns a list of 1+ chunks.
        """
        if limit is None:
            limit = MatrixBot.MAX_MESSAGE_CHARS

        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        remaining = text
        # Reserve room for the continuation markers
        marker_budget = len("\n\n[\u2026continued]")
        effective = limit - marker_budget

        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break

            # Try to split on a paragraph boundary
            candidate = remaining[:effective]
            split_pos = candidate.rfind("\n\n")

            # Fall back to single newline
            if split_pos < effective // 4:
                split_pos = candidate.rfind("\n")

            # Hard cut as last resort
            if split_pos < effective // 4:
                split_pos = effective

            chunk = remaining[:split_pos].rstrip()
            remaining = remaining[split_pos:].lstrip("\n")
            chunks.append(chunk)

        # Add continuation markers
        if len(chunks) > 1:
            for i in range(len(chunks)):
                if i < len(chunks) - 1:
                    chunks[i] += "\n\n[\u2026continued]"
                if i > 0:
                    chunks[i] = "[\u2026continued]\n\n" + chunks[i]

            # Repair any code fences broken by the split
            chunks = MatrixBot._repair_fences(chunks)

        return chunks

    @staticmethod
    def _repair_fences(chunks: list[str]) -> list[str]:
        """Fix markdown code fences broken by message splitting.

        If a split falls inside a fenced code block, the first chunk gets
        an unclosed fence (breaking all subsequent rendering) and the next
        chunk starts mid-block. This repairs both sides by closing the
        fence at the end of the chunk and re-opening it at the start of
        the next, preserving the language tag.
        """
        if not chunks:
            return chunks

        CONT_END = "\n\n[\u2026continued]"
        CONT_START = "[\u2026continued]\n\n"

        result = []
        fence_opener_for_next = None

        for chunk in chunks:
            # Extract continuation markers
            end_marker = ""
            start_marker = ""
            content = chunk

            if content.endswith(CONT_END):
                end_marker = CONT_END
                content = content[:-len(end_marker)]

            if content.startswith(CONT_START):
                start_marker = CONT_START
                content = content[len(start_marker):]

            # If previous chunk had open fence, prepend opener
            if fence_opener_for_next:
                content = fence_opener_for_next + content
                fence_opener_for_next = None

            # Track fence state through this chunk
            lines = content.split("\n")
            fence_count = 0
            last_opener = "```"

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("```"):
                    fence_count += 1
                    if fence_count % 2 == 1:  # opening fence
                        last_opener = stripped

            # If fence is open at end of chunk, close it and queue opener for next
            if fence_count % 2 == 1:
                content += "\n```"
                fence_opener_for_next = last_opener + "\n"

            result.append(start_marker + content + end_marker)

        return result

    async def send(self, room_id: str, text: str):
        """Send a text message to a room, splitting if too large.

        Messages exceeding MAX_MESSAGE_CHARS are split on paragraph
        boundaries and sent as sequential messages.

        Args:
            room_id: Matrix room ID
            text: Message content (markdown supported)

        Retries on transient failures with exponential backoff.
        """
        chunks = self._split_message(text)
        for chunk in chunks:
            content = {
                "msgtype": "m.text",
                "body": chunk,
                "format": "org.matrix.custom.html",
                "formatted_body": mistune.html(chunk),
            }
            await self._room_send_with_retry(room_id, content)

    def _fire_background(self, coro) -> asyncio.Task:
        """Schedule a coroutine as a background task.

        Allows sync_forever to continue dispatching events (including /stop)
        while the agent processes a message.
        """
        task = asyncio.create_task(coro)
        # Lazy-init for tests that construct MatrixBot via __new__
        if not hasattr(self, '_background_tasks'):
            self._background_tasks = set()
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _cancel_current(self):
        """Cancel any current in-flight work.

        - Cancel in-flight LLM call
        - Kill tool subprocesses
        - Send cancellation notice to room
        """
        # Capture room before cancellation — the task's finally block clears it
        room = self._current_room

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

        if room:
            await self.send(room, "Cancelled.")
            await self._set_typing(room, False)

    async def _run_heartbeat_turn(self, room_id: str, content: str) -> None:
        """Execute a heartbeat/umbral turn: activate room, process input, deliver response.

        Extracted from _inject_heartbeat for reuse by _inject_umbral.
        Raises on error — caller is responsible for error handling.
        """
        # Activate room if not already active
        if room_id not in self._active_rooms:
            await self._activate_room(room_id)

        # -- tool-use callbacks (same as normal message path) --

        async def _tool_notice(call_id, name, input_data, result, is_error):
            status = "❌ error" if is_error else "✅"
            detail = ""
            if isinstance(input_data, dict):
                for key in ("path", "file_path", "command", "query", "url"):
                    if key in input_data:
                        val = str(input_data[key])[:120]
                        detail = f" `{val}`"
                        break
            try:
                await self._set_typing(room_id, True)
            except Exception:
                pass
            if name == "subagent" and isinstance(input_data, dict):
                task_preview = input_data.get("task", "")[:500]
                model_info = input_data.get("model", "default")
                result_preview = str(result)[:2000] if result else ""
                summary_line = f"🤖 subagent ({model_info}) {status}"
                html = f'<b>{summary_line}</b>'
                if task_preview:
                    html += (
                        f'\n<details><summary>📋 Task brief</summary>\n'
                        f'<pre>{mistune.html(task_preview)}</pre></details>'
                    )
                if result_preview:
                    html += (
                        f'\n<details><summary>📨 Result</summary>\n'
                        f'<pre>{mistune.html(result_preview)}</pre></details>'
                    )
                body_text = f"{summary_line}\n\nTask: {task_preview[:200]}"
                content_msg = {
                    "msgtype": "m.notice",
                    "body": body_text,
                    "format": "org.matrix.custom.html",
                    "formatted_body": html,
                }
                try:
                    await self._room_send_with_retry(room_id, content_msg)
                except Exception:
                    pass
            else:
                notice_body = f"🔧 {name}{detail} {status}"
                try:
                    await self.send_notice(room_id, notice_body)
                except Exception:
                    pass
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

        async def _tool_intent(tool_calls, content_text):
            _sl = getattr(self, 'session_log', None)
            if _sl:
                _sl.append(
                    role="assistant",
                    sender=self.config.user_id,
                    room=room_id,
                    event_id=None,
                    content=content_text or "",
                    tool_calls=[
                        {"call_id": tc.id, "name": tc.name, "input": tc.input}
                        for tc in tool_calls
                    ],
                )

        # Process through agent
        try:
            await self._set_typing(room_id, True)

            # Resolve thinking level: room override > config
            _thinking_override = getattr(self, '_room_thinking', {}).get(room_id)
            _thinking_buffer = []
            _thinking_done = False

            async def _thinking_delta(text: str, done: bool):
                nonlocal _thinking_done
                if not done:
                    _thinking_buffer.append(text)
                else:
                    _thinking_done = True
                    # Send thinking as <details> block
                    full_thinking = "".join(_thinking_buffer)
                    _thinking_buffer.clear()  # reset for next tool-loop iteration
                    if full_thinking.strip():
                        if len(full_thinking) > self.MAX_MESSAGE_CHARS:
                            full_thinking = (
                                full_thinking[:self.MAX_MESSAGE_CHARS]
                                + "\n\n[truncated — full thinking in session JSONL]"
                            )
                        html = (
                            '<details>\n<summary>💭 Thinking</summary>\n'
                            f'{mistune.html(full_thinking)}'
                            '</details>'
                        )
                        thinking_content = {
                            "msgtype": "m.notice",
                            "body": f"💭 Thinking\n\n{full_thinking}",
                            "format": "org.matrix.custom.html",
                            "formatted_body": html,
                            "openalph.thinking": True,
                        }
                        await self._room_send_with_retry(room_id, thinking_content)

            response = await self.agent.handle_input(
                content,
                room_id,
                on_tool_call=_tool_notice,
                on_tool_intent=_tool_intent,
                thinking=_thinking_override,
                on_thinking_delta=_thinking_delta,
            )
            if response and response.strip():
                if self.session_log:
                    _thinking_data = None
                    _agent_history = self.agent.history(room_id)
                    if _agent_history and _agent_history[-1].get("role") == "assistant":
                        _thinking_data = _agent_history[-1].get("thinking")
                    _log_kwargs = dict(
                        role="assistant",
                        sender=self.config.user_id,
                        room=room_id,
                        event_id=None,
                        content=response,
                    )
                    if _thinking_data:
                        _log_kwargs["thinking"] = _thinking_data
                    self.session_log.append(**_log_kwargs)
                await self.send(room_id, response)
            else:
                # Model did tool work but returned empty text.  Retry once
                # with a nudge — the model sees its own tool results in
                # history and should produce the report it failed to emit.
                logger.warning("Empty heartbeat response in %s — retrying once", room_id)
                retry = await self.agent.handle_input(
                    "[SYSTEM: Your previous heartbeat response was empty. "
                    "Summarize your findings now.]",
                    room_id,
                    on_tool_call=_tool_notice,
                    on_tool_intent=_tool_intent,
                    thinking=_thinking_override,
                    on_thinking_delta=_thinking_delta,
                )
                if retry and retry.strip():
                    if self.session_log:
                        _thinking_data = None
                        _agent_history = self.agent.history(room_id)
                        if _agent_history and _agent_history[-1].get("role") == "assistant":
                            _thinking_data = _agent_history[-1].get("thinking")
                        _log_kwargs = dict(
                            role="assistant",
                            sender=self.config.user_id,
                            room=room_id,
                            event_id=None,
                            content=retry,
                        )
                        if _thinking_data:
                            _log_kwargs["thinking"] = _thinking_data
                        self.session_log.append(**_log_kwargs)
                    await self.send(room_id, retry)
                else:
                    logger.warning("Empty heartbeat response in %s after retry — giving up", room_id)
                    await self.send(room_id,
                        "⚠️ **Empty heartbeat response** — the model returned no content "
                        "after retry. This may indicate degeneration or a provider issue.")
        finally:
            await self._set_typing(room_id, False)

    async def _inject_heartbeat(self, room_id: str) -> None:
        """Process a heartbeat as if the agent received a wake message."""
        heartbeat_content = "Heartbeat: execute your WAKE instructions."

        # Post to Matrix so the operator can see heartbeat triggers
        await self.send_notice(room_id, "💓 Heartbeat")

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

        try:
            await self._run_heartbeat_turn(room_id, heartbeat_content)
        except AgentOverflowError:
            logger.warning("Context overflow in %s — auto-stopping heartbeat", room_id)
            await self.heartbeat.stop(room_id)
            await self.send(
                room_id,
                "⚠️ **Context overflow** — heartbeat auto-stopped for this room. "
                "Start a new room to continue.",
            )
        except ProviderError as e:
            code = f" ({e.status_code})" if e.status_code else ""
            logger.warning("Heartbeat provider error%s in %s: %s", code, room_id, e)
            await self.send(room_id, f"⚠️ **Provider error** (heartbeat): {e}")
        except Exception:
            logger.exception("Heartbeat processing error in %s", room_id)
            await self.send(room_id, "⚠️ Heartbeat error — check agent logs for details.")

    async def _inject_umbral(self, room_id: str) -> None:
        """Execute an umbral turn: heartbeat + context rotation."""
        heartbeat_content = "Heartbeat: execute your WAKE instructions."

        await self.send_notice(room_id, "🌑 Umbral turn beginning")

        if self.session_log:
            self.session_log.append(
                role="system",
                sender=self.config.user_id,
                room=room_id,
                event_id=None,
                content=heartbeat_content,
                source="umbral",
            )

        try:
            await self._run_heartbeat_turn(room_id, heartbeat_content)
        except AgentOverflowError:
            # Context was already too big — still archive+wipe (that's the point)
            logger.warning("Context overflow during umbral in %s — archiving anyway", room_id)
        except ProviderError as e:
            code = f" ({e.status_code})" if e.status_code else ""
            logger.warning("Umbral provider error%s in %s: %s", code, room_id, e)
            await self.send(room_id, f"⚠️ **Provider error** during umbral turn: {e}")
        except Exception:
            logger.exception("Umbral turn processing error in %s", room_id)
            await self.send(room_id, "⚠️ Umbral turn error — check agent logs. Archiving context.")

        # Archive + wipe regardless of turn success (failed turns still consume context)
        try:
            archive_name = self.session_log.archive(room_id)
            self.session_log.wipe(room_id)
            self.session_log.append(
                role="system",
                sender=self.config.user_id,
                room=room_id,
                event_id=None,
                event="umbral_reset",
                detail=f"Context reset. Previous session archived to sessions/{archive_name}",
            )
            self.agent.reset_room(room_id)
            self._active_rooms.discard(room_id)
            await self.send_notice(
                room_id,
                "🌑 Umbral turn concluded — context reset. "
                "Agent will not remember session history above this point.",
            )
        except FileNotFoundError:
            logger.warning("Umbral archive: no session file for %s — nothing to rotate", room_id)
            await self.send_notice(room_id, "🌑 Umbral turn concluded (no session to archive)")
        except OSError as e:
            # Archive failed — DO NOT wipe. Stop umbral. Alert operator.
            logger.critical("Umbral archive failed for %s: %s — NOT wiping, stopping umbral", room_id, e)
            await self.umbral.stop(room_id)
            await self.send(
                room_id,
                f"🚨 **Umbral archive failed** — session preserved, umbral stopped. "
                f"Manual intervention required. Error: {e}",
            )

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
                                if hasattr(msg, 'body') and msg.sender != self.config.user_id and not _is_streaming_edit(msg):
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
                            # For final edit events, use replacement content
                            msg_source = getattr(msg, 'source', {}) or {}
                            msg_content = msg_source.get("content", {})
                            if msg_content.get("m.relates_to", {}).get("rel_type") == "m.replace":
                                msg_body = msg_content.get("m.new_content", {}).get("body", msg.body)
                            else:
                                msg_body = msg.body
                            session_log.append(
                                role="user",
                                sender=msg.sender,
                                room=room_id,
                                event_id=ev_id,
                                content=msg_body,
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

                # Restore per-room overrides (model, thinking) from session log.
                # Scan all entries — last override wins (user may have switched multiple times).
                _restored_model = None
                _restored_thinking = None
                for entry in existing:
                    if entry.get("role") == "system":
                        ev = entry.get("event")
                        detail = entry.get("detail", "")
                        if ev == "model_override" and detail:
                            self.agent._room_models[room_id] = detail
                            _restored_model = detail
                        elif ev == "thinking_override" and detail:
                            if not hasattr(self, "_room_thinking"):
                                self._room_thinking = {}
                            self._room_thinking[room_id] = detail
                            _restored_thinking = detail

                # Send session resume notice to Matrix
                parts = [f"🔄 **Session resumed** — {len(existing)} prior entries"]
                if _restored_model:
                    parts.append(f"Model override: `{_restored_model}`")
                if _restored_thinking:
                    parts.append(f"Thinking: `{_restored_thinking}`")
                try:
                    await self.send_notice(room_id, " · ".join(parts))
                except Exception:
                    pass
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
            try:
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
            except TypeError:
                # Test environment with mock client - skip history loading
                pass

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

    def _format_size(self, size_bytes: int) -> str:
        """Format bytes as human-readable string."""
        if size_bytes >= 1_000_000:
            return f"{size_bytes / 1_000_000:.1f} MB"
        elif size_bytes >= 1_000:
            return f"{size_bytes / 1_000:.1f} KB"
        else:
            return f"{size_bytes} B"

    async def _process_message(self, room, event, body: str, _gating_handled: bool = False):
        """Process a message through the agent pipeline.

        Shared logic for both text and media messages. Handles:
        - Mention gating (for gated rooms)
        - Room activation
        - Session logging
        - Agent processing with tool visibility
        - Response sending

        Args:
            room: Matrix room object
            event: Matrix event (for metadata like sender, event_id)
            body: The message content to process
        """
        room_id = room.room_id
        self._current_room = room_id

        try:
            # --- Mention gating ---
            session_log = getattr(self, 'session_log', None)
            gated = is_gated(self.config, room)

            if gated and not _gating_handled:
                # Media messages still need gating here (text messages handled in _handle_room_message)
                event_source = getattr(event, 'source', {}) or {}
                mention = mentions_me(self.config.user_id, event_source, body)

                # Always buffer to session log
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

            # Context hydration for gated rooms (regardless of who handled gating)
            if gated and room_id in self._active_rooms and self.session_log:
                history = self.agent.history(room_id)
                history.clear()
                history.extend(self.session_log.build_context(room_id))
                logger.info("Hydrated context for %s: %d entries", room_id, len(history))
            # --- End mention gating ---

            # Lazy wake: activate room on first live message
            if room_id not in self._active_rooms:
                room_name = getattr(room, 'name', '') or getattr(room, 'display_name', '') or room_id
                await self._activate_room(room_id, room_name=room_name)

            # Acquire per-room session lock to prevent interleaved JSONL writes.
            # Holds from user-message append through handle_input (which triggers
            # tool_intent and tool_notice JSONL writes) so a concurrent message
            # cannot wedge its user entry between tool_calls and tool_results.
            if not hasattr(self, '_session_locks'):
                self._session_locks = {}
            if room_id not in self._session_locks:
                self._session_locks[room_id] = asyncio.Lock()
            _session_lock = self._session_locks[room_id]
            await _session_lock.acquire()

            # Append user message to session log (only for ungated rooms — gated already buffered above)
            if not gated:
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
                # Show tool name + brief input context, but NEVER output
                # (which may contain secrets from op read, API responses, etc.)
                status = "❌ error" if is_error else "✅"
                detail = ""
                if isinstance(input_data, dict):
                    # Pick the most informative input field per tool type
                    for key in ("path", "file_path", "command", "query", "url"):
                        if key in input_data:
                            val = str(input_data[key])[:120]
                            detail = f" `{val}`"
                            break
                # Refresh typing indicator — Matrix expires it after ~30s,
                # so long tool loops look dead without this.
                try:
                    await self._set_typing(room_id, True)
                except Exception:
                    pass
                # Subagent calls: show task brief (collapsed) on dispatch,
                # and result summary (collapsed) on return
                if name == "subagent" and isinstance(input_data, dict):
                    task_preview = input_data.get("task", "")[:500]
                    model_info = input_data.get("model", "default")
                    result_preview = str(result)[:2000] if result else ""
                    summary_line = f"🤖 subagent ({model_info}) {status}"
                    html = f'<b>{summary_line}</b>'
                    if task_preview:
                        html += (
                            f'\n<details><summary>📋 Task brief</summary>\n'
                            f'<pre>{mistune.html(task_preview)}</pre></details>'
                        )
                    if result_preview:
                        html += (
                            f'\n<details><summary>📨 Result</summary>\n'
                            f'<pre>{mistune.html(result_preview)}</pre></details>'
                        )
                    body_text = f"{summary_line}\n\nTask: {task_preview[:200]}"
                    content = {
                        "msgtype": "m.notice",
                        "body": body_text,
                        "format": "org.matrix.custom.html",
                        "formatted_body": html,
                    }
                    try:
                        await self._room_send_with_retry(room_id, content)
                    except Exception:
                        pass
                else:
                    notice_body = f"🔧 {name}{detail} {status}"
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

            # Create upload callback for send_media tool
            async def _upload_callback(file_path, content_type, filename, caption=None):
                await self.upload_and_send(room.room_id, file_path, content_type, filename, caption)

            try:
                # Resolve thinking level: room override > config
                _thinking_override = getattr(self, '_room_thinking', {}).get(room_id)
                callbacks = {"send_media": _upload_callback}

                # Set up streaming delivery
                streaming = StreamingDelivery(self, room_id)
                _thinking_buffer = []
                _thinking_done = False

                async def _text_delta(text: str, done: bool):
                    await streaming.push(text, done=done)

                async def _thinking_delta(text: str, done: bool):
                    nonlocal _thinking_done
                    if not done:
                        _thinking_buffer.append(text)
                    else:
                        _thinking_done = True
                        # Send thinking as <details> block
                        full_thinking = "".join(_thinking_buffer)
                        _thinking_buffer.clear()  # reset for next tool-loop iteration
                        if full_thinking.strip():
                            # Truncate thinking to avoid M_TOO_LARGE on Matrix PDU limit.
                            # HTML rendering roughly doubles size; cap raw text at MAX_MESSAGE_CHARS
                            # to keep formatted message well under 65535 bytes.
                            # Full thinking is preserved in session JSONL.
                            if len(full_thinking) > self.MAX_MESSAGE_CHARS:
                                full_thinking = (
                                    full_thinking[:self.MAX_MESSAGE_CHARS]
                                    + "\n\n[truncated — full thinking in session JSONL]"
                                )
                            html = (
                                '<details>\n<summary>💭 Thinking</summary>\n'
                                f'{mistune.html(full_thinking)}'
                                '</details>'
                            )
                            content = {
                                "msgtype": "m.notice",
                                "body": f"💭 Thinking\n\n{full_thinking}",
                                "format": "org.matrix.custom.html",
                                "formatted_body": html,
                                "openalph.thinking": True,
                            }
                            await self._room_send_with_retry(room_id, content)

                response = await self.agent.handle_input(
                        body, room_id,
                        on_tool_call=_tool_notice,
                        on_tool_intent=_tool_intent,
                        on_text_delta=_text_delta,
                        on_thinking_delta=_thinking_delta,
                        thinking=_thinking_override,
                        callbacks=callbacks,
                    )
                # Append assistant response to session log
                if response and response.strip():
                    if session_log:
                        # Capture thinking blocks from agent's history for session persistence
                        _thinking_data = None
                        _agent_history = self.agent.history(room_id)
                        if _agent_history and _agent_history[-1].get("role") == "assistant":
                            _thinking_data = _agent_history[-1].get("thinking")
                        _log_kwargs = dict(
                            role="assistant",
                            sender=self.config.user_id,
                            room=room_id,
                            event_id=None,
                            content=response,
                        )
                        if _thinking_data:
                            _log_kwargs["thinking"] = _thinking_data
                        session_log.append(**_log_kwargs)
                    # Send if streaming didn't deliver, or if the response
                    # differs from what was streamed (e.g. tool-limit summary
                    # generated after the last streamed tool-call text).
                    if not streaming._delivered:
                        await self.send(room_id, response)
                    elif streaming._delivered_text.strip() != response.strip():
                        logger.info("Response differs from streamed content — sending separately")
                        await self.send(room_id, response)
                else:
                    logger.warning("Empty response from agent in %s — not sending", room_id)
                    await self.send(room_id,
                        "⚠️ **Empty response** — the model returned no content. "
                        "This may indicate degeneration or a provider issue. "
                        "Try again or start a new room.")
                # Check context capacity after successful turn
                try:
                    _status = self.agent.status(room_id)
                    _pct = _status.get("context_pct", 0)
                    if _pct >= 80:
                        await self.send_notice(room_id,
                            f"⚠️ Context at **{_pct}%** — "
                            f"~{_status['context_tokens']:,} / {_status['context_max']:,} tokens. "
                            f"Consider starting a new room soon.")
                except Exception:
                    pass
            except AgentOverflowError as e:
                logger.warning("Context overflow in %s: %s", room_id, e)
                await self.send(room_id,
                    f"⚠️ **Context overflow** — ~{e.current_tokens:,} / "
                    f"{e.max_tokens:,} tokens. Start a new room to continue.")
            except ProviderError as e:
                code = f" ({e.status_code})" if e.status_code else ""
                logger.warning("Provider error%s in %s: %s", code, room_id, e)
                await self.send(room_id, f"⚠️ **Provider error:** {e}")
            except Exception as e:
                # Agent error: send generic message to avoid leaking exception details
                logger.exception("Agent error processing message in %s", room_id)
                await self.send(room_id, "⚠️ Internal error — check agent logs for details.")
            finally:
                await self._set_typing(room_id, False)

        finally:
            # Release per-room session lock so queued messages can proceed.
            # The lock was acquired before the user-message JSONL write and
            # held through handle_input (which writes tool_intent / tool_notice
            # / assistant entries), preventing interleaved JSONL writes.
            if hasattr(self, '_session_locks'):
                _sl = self._session_locks.get(room_id)
                if _sl and _sl.locked():
                    _sl.release()
            self._current_room = None

    async def _handle_media_message(self, room, event):
        """Handle a media message event (image, audio, video, file).

        Downloads the media file to the workspace and passes a bracket-tag
        message to the agent for processing.

        Args:
            room: Matrix room object
            event: RoomMessageImage, RoomMessageAudio, RoomMessageVideo, or RoomMessageFile
        """
        logger.debug("Media event received: room=%s sender=%s type=%s synced=%s",
                     room.room_id, event.sender, type(event).__name__, self._synced)

        # During initial sync, don't hydrate rooms
        if not self._synced:
            return

        # Skip own messages
        if event.sender == self.config.user_id:
            return

        # Get filename from event
        filename = _sanitize_filename(event.body)

        try:
            # Download the file
            response = await self.client.download(mxc=event.url, filename=event.body)

            # Check for download error
            if isinstance(response, DownloadError):
                error_msg = f"[media: download failed — {filename} ({response.message})]"
                logger.warning("Media download failed for %s: %s", event.url, response.message)
                self._fire_background(self._process_message(room, event, error_msg))
                return

            # Check size limit
            file_size = len(response.body)
            if file_size > MAX_MEDIA_BYTES:
                size_human = self._format_size(file_size)
                skip_msg = f"[media: skipped — {filename} exceeds 20 MB limit ({size_human})]"
                logger.warning("Media file too large: %s (%s)", filename, size_human)
                self._fire_background(self._process_message(room, event, skip_msg))
                return

            # Determine storage path
            event_hash = _event_id_hash(event.event_id)
            workspace = Path(self.agent.config.workspace)
            media_dir = workspace / MEDIA_DIR / event_hash
            media_dir.mkdir(parents=True, exist_ok=True)

            # Save the file
            file_path = media_dir / filename
            file_path.write_bytes(response.body)

            # Get MIME type from event metadata
            content_info = event.source.get("content", {}).get("info", {})
            mime_type = content_info.get("mimetype", "application/octet-stream")

            # Build the bracket-tag message
            relative_path = f"{MEDIA_DIR}/{event_hash}/{filename}"
            size_human = self._format_size(file_size)

            # Check if body differs from filename for caption
            event_content = event.source.get("content", {})
            original_filename = event_content.get("filename", event.body)
            has_caption = event.body != original_filename

            message = f"[media: {relative_path} ({mime_type}, {size_human})]"
            if has_caption:
                message += f"\n{event.body}"

            # Process through the shared pipeline
            self._fire_background(self._process_message(room, event, message))

        except Exception as e:
            # Handle unexpected errors (network issues, etc.)
            error_msg = f"[media: download failed — {filename} ({str(e)})]"
            logger.exception("Unexpected error downloading media from %s", event.url)
            self._fire_background(self._process_message(room, event, error_msg))

    async def _handle_room_message(self, room, event):
        """Handle a room message event.

        Routes messages through mention gating before command parsing.
        In gated rooms (3+ members), all commands require @mention.
        In DM rooms (2 members), all commands work without mention.

        Flow:
        1. Skip pre-sync events and own messages
        2. In gated rooms: check mention, buffer to session log, send hint for
           bare commands, strip mention from body
        3. Parse slash commands with (possibly stripped) body
        4. Regular messages → _process_message

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

        # Skip intermediate streaming edits and partial initial sends from
        # other agents.  Accept final edits (full content, no cursor).
        if _is_streaming_edit(event):
            return

        # Skip thinking blocks from other agents (custom content field)
        event_source = getattr(event, 'source', None) or {}
        if event_source.get("content", {}).get("openalph.thinking") is True:
            return

        room_id = room.room_id

        # If room is halted via /stop, drop all incoming messages silently
        if room_id in self._halted_rooms:
            return

        # For final edit events, use the replacement content (m.new_content)
        event_content = event_source.get("content", {})
        if event_content.get("m.relates_to", {}).get("rel_type") == "m.replace":
            body = event_content.get("m.new_content", {}).get("body", event.body).strip()
        else:
            body = event.body.strip()

        # --- Mention gating (kdsn.60) ---
        gated = is_gated(self.config, room)

        if gated:
            event_source = getattr(event, 'source', {}) or {}
            mention = mentions_me(self.config.user_id, event_source, body)

            # Buffer ALL messages to session log (mentioned or not)
            if self.session_log:
                self.session_log.append(
                    role="user",
                    sender=event.sender,
                    room=room_id,
                    event_id=getattr(event, 'event_id', None),
                    content=body,
                    mentioned=mention.mentioned,
                )

            if not mention.mentioned:
                # Not mentioned — send hint for bare commands, then skip
                if body.startswith("/"):
                    localpart = self.config.user_id.split(":")[0] if ":" in self.config.user_id else self.config.user_id
                    await self.send_notice(
                        room_id,
                        f"\U0001f4a1 Commands need a mention in shared rooms: "
                        f"`{localpart} {body.split()[0]}`",
                    )
                logger.debug("Gated room %s: not mentioned, skipping", room_id)
                return

            # Strip mention from body for command parsing
            body = strip_mention(self.config.user_id, body)
            logger.info("Gated room %s: mentioned via %s, processing (body=%r)",
                         room_id, mention.method, body[:50])
        # --- End mention gating ---

        # Lazy wake: ensure room is activated before processing any command so that
        # _process_message and slash commands both see full room history (kdsn.79).
        if not hasattr(self, '_active_rooms'):
            self._active_rooms = set()
        if room_id not in self._active_rooms:
            room_name = getattr(room, 'name', '') or getattr(room, 'display_name', '') or room_id
            await self._activate_room(room_id, room_name=room_name)

        # --- Slash commands ---
        if body == "/stop":
            self._halted_rooms.add(room_id)
            await self._cancel_current()
            await self.send(room_id, "Stopped. Room halted \u2014 use `/resume` to re-enable.")
            return

        if body == "/resume":
            if room_id in self._halted_rooms:
                self._halted_rooms.discard(room_id)
                await self.send(room_id, "Resumed.")
            else:
                await self.send(room_id, "Room wasn't halted.")
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
            if len(output) > 65000:
                output = output[:65000] + "\n\n... (truncated)"
            await self.send(room_id, f"```\n{output}\n```")
            return

        if body == "/status":
            status = self.agent.status(room_id)
            ctx = status['context_tokens']
            ctx_max = status['context_max']
            ctx_pct = status['context_pct']
            bar_len = 20
            filled = round(bar_len * ctx_pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            lines = [
                f"### {status['name']}",
                "",
                f"| | |",
                f"|---|---|",
                f"| **Model** | `{status['model']}` |",
                f"| **Turns** | {status['turns']} |",
                f"| **Context** | {bar} {ctx_pct}% (~{ctx:,} / {ctx_max:,}) |",
                f"| **Session in** | {status['total_input_tokens']:,} tokens |",
                f"| **Session out** | {status['total_output_tokens']:,} tokens |",
                f"| **Tool calls** | {status['total_tool_calls']} |",
            ]
            await self.send(room_id, "\n".join(lines))
            return

        if body.startswith("/model"):
            parts = body.split(None, 1)
            if len(parts) < 2:
                await self.send(room_id, "Usage: `/model <provider/model-name>`")
                return
            new_model = parts[1].strip()
            error = self.agent.switch_model(new_model, room_id)
            if error:
                await self.send(room_id, f"\u26a0\ufe0f {error}")
            else:
                # Persist override so it survives process restarts
                if self.session_log:
                    self.session_log.append(
                        role="system",
                        sender=event.sender,
                        room=room_id,
                        event_id=None,
                        event="model_override",
                        detail=new_model,
                    )
                await self.send(room_id, f"Model switched to **{new_model}**")
            return

        if body.startswith("/thinking"):
            parts = body.split(None, 1)
            if len(parts) < 2:
                # Show current thinking level
                current = self._room_thinking.get(room_id)
                if current is None:
                    current = getattr(self.agent.config, 'thinking', 'off')
                    source = "config"
                else:
                    source = "override"
                await self.send(room_id, f"Thinking: **{current}** ({source})")
                return
            level = parts[1].strip().lower()
            valid_levels = ("off", "low", "medium", "high")
            if level not in valid_levels:
                await self.send(room_id, f"Invalid level. Use: {', '.join(valid_levels)}")
                return
            self._room_thinking[room_id] = level
            # Persist override so it survives process restarts
            if self.session_log:
                self.session_log.append(
                    role="system",
                    sender=event.sender,
                    room=room_id,
                    event_id=None,
                    event="thinking_override",
                    detail=level,
                )
            await self.send(room_id, f"Thinking set to **{level}** for this room")
            return

        if body.startswith("/heartbeat"):
            parts = body.split()
            if len(parts) >= 3 and parts[1] == "start":
                interval = parse_interval(parts[2])
                if interval is None:
                    await self.send(room_id, "Invalid interval. Use e.g. `15m`, `1h`, `6h`.")
                elif interval < 300:
                    await self.send(room_id, "Minimum interval is 5m.")
                elif self.umbral and self.umbral.is_active(room_id):
                    await self.send(room_id,
                        "Stop the umbral timer first (`/umbral stop`) — "
                        "umbral and heartbeat cannot run in the same room.")
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
                    lines = ["**Active heartbeats:**", ""]
                    for e in entries:
                        # Resolve room name from nio client
                        nio_room = self.client.rooms.get(e.room_id)
                        name = (getattr(nio_room, 'name', '') or getattr(nio_room, 'display_name', '') or e.room_id) if nio_room else e.room_id
                        lines.append(f"- **{name}** — every {format_interval(e.interval_seconds)}, next in {format_interval(e.seconds_until_next)}")
                    await self.send(room_id, "\n".join(lines))
            else:
                await self.send(room_id, "Usage: `/heartbeat start <interval>` | `/heartbeat stop` | `/heartbeat status`")
            return

        if body.startswith("/umbral"):
            parts = body.split()
            if len(parts) >= 3 and parts[1] == "start":
                interval = parse_interval(parts[2])
                if interval is None:
                    await self.send(room_id, "Invalid interval. Use e.g. `30m`, `6h`.")
                elif interval < 1800:
                    await self.send(room_id, "Minimum interval is 30m.")
                elif self.heartbeat and self.heartbeat.is_active(room_id):
                    await self.send(room_id,
                        "Stop the heartbeat first (`/heartbeat stop`) — "
                        "umbral and heartbeat cannot run in the same room.")
                else:
                    await self.umbral.start(room_id, interval)
                    human = format_interval(interval)
                    await self.send(room_id, f"🌑 Umbral started: every {human} in this room.")
            elif len(parts) >= 2 and parts[1] == "stop":
                stopped = await self.umbral.stop(room_id)
                if stopped:
                    await self.send(room_id, "🌑 Umbral stopped.")
                else:
                    await self.send(room_id, "No umbral active in this room.")
            elif len(parts) >= 2 and parts[1] == "status":
                entries = self.umbral.status()
                if not entries:
                    await self.send(room_id, "No active umbral timers.")
                else:
                    lines = ["**Active umbral timers:**", ""]
                    for e in entries:
                        nio_room = self.client.rooms.get(e.room_id)
                        name = (getattr(nio_room, 'name', '') or
                                getattr(nio_room, 'display_name', '') or
                                e.room_id) if nio_room else e.room_id
                        lines.append(
                            f"- **{name}** — every {format_interval(e.interval_seconds)}, "
                            f"next in {format_interval(e.seconds_until_next)}")
                    await self.send(room_id, "\n".join(lines))
            else:
                await self.send(room_id,
                    "Usage: `/umbral start <interval>` | `/umbral stop` | `/umbral status`")
            return

        # Fire as background task so sync_forever can dispatch /stop during tool loops
        self._fire_background(self._process_message(room, event, body, _gating_handled=gated))

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

        # Register media event callbacks
        self.client.add_event_callback(self._handle_media_message, RoomMessageImage)
        self.client.add_event_callback(self._handle_media_message, RoomMessageAudio)
        self.client.add_event_callback(self._handle_media_message, RoomMessageVideo)
        self.client.add_event_callback(self._handle_media_message, RoomMessageFile)

        # Initial sync: populates rooms and loads timeline history via callback
        import time as _time
        _sync_start = _time.monotonic()
        await self.client.sync(timeout=self.config.sync_timeout)
        _sync_ms = (_time.monotonic() - _sync_start) * 1000
        self._synced = True
        joined = len(self.client.rooms) if hasattr(self.client, 'rooms') else '?'
        logger.info("Initial sync complete in %.0fms (%s rooms joined, lazy wake active)", _sync_ms, joined)

        # Resume persisted heartbeats and umbral timers
        await self.heartbeat.resume()
        await self.umbral.resume()

        # Use sync_forever for the main sync loop
        await self.client.sync_forever(timeout=self.config.sync_timeout)

    async def stop(self):
        """Stop the Matrix bot gracefully.

        1. Cancel any in-flight work
        2. Close Matrix client
        """
        self._running = False
        await self._cancel_current()
        await self.heartbeat.shutdown()
        await self.umbral.shutdown()
        await self.client.close()
