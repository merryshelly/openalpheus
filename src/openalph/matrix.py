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
import tempfile
import time
from html import escape as html_escape
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
from openalph.tools import escape_system_reminder_tags

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


def _escape_preserve_breaks(text: str) -> str:
    """html-escape tool/model-origin text for a Matrix notice while preserving
    line breaks. HTML whitespace-folds literal newlines to spaces, so escape
    first (the deliberate injection-defense choice for model-origin text — NOT
    mistune/markdown) then convert real newlines to <br>. Mirrors the
    '<br>'.join idiom already used by the todo-notice fold."""
    esc = html_escape(str(text))
    esc = esc.replace("\r\n", "\n").replace("\r", "\n")
    return esc.replace("\n", "<br>")


def format_model_list(aliases: dict[str, str], current_model: str) -> str:
    """Format model alias table for /model list output."""
    lines = [f"**Current model:** `{current_model}`\n"]
    if aliases:
        lines.append("| Alias | Model |")
        lines.append("|-------|-------|")
        for alias in sorted(aliases):
            marker = " ◀" if aliases[alias] == current_model else ""
            lines.append(f"| `{alias}` | `{aliases[alias]}`{marker} |")
    else:
        lines.append("No model aliases configured.")
    return "\n".join(lines)


def _trunc_directive(d: str, width: int = 120) -> str:
    one_line = " ".join(d.split())
    return one_line if len(one_line) <= width else one_line[:width - 1] + "…"


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
        # CORE-1: retained only as the shutdown-time fallback for the
        # "Cancelled." notice. It is NOT used to decide WHAT to cancel any
        # more -- `Agent._current_tasks` is keyed by room for that.
        self._current_room = None
        self._active_rooms = set()
        self._halted_rooms: set[str] = set()
        self._room_thinking = {}
        self._room_cache_ttl = {}   # Room-scoped cache TTL overrides (e.g. "5m"; default is "1h")
        self._room_timesense = {}   # Room-scoped timesense toggle (prepend timestamp to user messages)
        self._background_tasks: set[asyncio.Task] = set()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._steering_inbox: dict[str, list[str]] = {}
        self._active_turns: set[str] = set()
        self._advisor_results: dict[tuple, dict] = {}  # R5: keyed (room_id, call_id)
        self._subagent_results: dict[tuple, dict] = {}  # keyed (room_id, call_id)

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

    _CANCEL_TIMEOUT = 5  # seconds to wait for cancelled task before abandoning

    async def _cancel_current(self, room_id: str | None = None):
        """Cancel in-flight work for `room_id` (or all rooms if None).

        - Cancel in-flight LLM call
        - Kill tool subprocesses
        - Send cancellation notice to the room that asked

        CORE-1: this took no argument and cancelled whatever
        `Agent._current_task` happened to point at, announcing into
        `self._current_room` -- both single slots overwritten by whichever
        room started a turn most recently. With concurrent rooms that meant a
        `/stop` in room A could cancel room B's turn and print "Cancelled."
        into B. Callers now pass the room explicitly; only `shutdown()` omits
        it, where cancelling everything is what is actually wanted.

        Uses a timeout to prevent blocking the event loop if the cancelled
        task is stuck (e.g., hung httpx call to a slow inference API).
        """
        # Announce into the room that asked, NOT into whichever room happened
        # to start a turn last. `_current_room` remains only as the shutdown
        # fallback, where there is no requesting room.
        room = room_id if room_id is not None else self._current_room

        task = None
        if hasattr(self.agent, "cancel"):
            task = self.agent.cancel(room_id) if room_id is not None else self.agent.cancel()

        # Wait for the cancelled task to finish, but not forever.
        # If it doesn't die within _CANCEL_TIMEOUT, abandon it and move on.
        # The room is halted anyway — the orphaned task will eventually
        # finish or be cleaned up on process shutdown.
        if task:
            try:
                done, _ = await asyncio.wait({task}, timeout=self._CANCEL_TIMEOUT)
                if not done:
                    logger.warning(
                        "Cancelled task in %s did not finish within %ds — abandoning",
                        room, self._CANCEL_TIMEOUT,
                    )
                else:
                    # Retrieve the result to suppress "exception was never retrieved"
                    for t in done:
                        try:
                            t.result()
                        except (asyncio.CancelledError, Exception):
                            pass
            except Exception:
                pass  # Defensive — don't let cancel cleanup break the event loop

        if room:
            await self.send(room, "Cancelled.")
            await self._set_typing(room, False)

    def _persist_assistant_turn(self, room_id: str, *, content: str,
                             tool_calls=None) -> None:
        """Single serializer for assistant turns. Captures content + tool_calls +
        thinking (from agent.history[-1]) + usage (from agent.last_turn_usage).
        Used by BOTH the tool-use path (_tool_intent) and the final-text paths.

        INVARIANT (RC1): This method reads thinking from agent.history(room_id)[-1].
        It is correct ONLY because the caller (agent.handle_input) always appends the
        assistant message to history immediately before this serializer runs. Any future
        change that inserts a history mutation between that append and this call will
        silently break thinking capture.
        """
        _sl = getattr(self, "session_log", None)
        if not _sl:
            return
        # thinking: read from the current last assistant turn in history
        thinking = None
        try:
            hist = self.agent.history(room_id)
            if hist and hist[-1].get("role") == "assistant":
                thinking = hist[-1].get("thinking")
        except Exception:
            logger.debug("_persist_assistant_turn: thinking capture failed", exc_info=True)
            thinking = None
        # usage: per-turn delta; isinstance guard so MagicMock agents (tests) -> {}
        usage = {}
        try:
            lu = self.agent.last_turn_usage(room_id)
            if isinstance(lu, dict):
                usage = dict(lu)
                usage["tool_calls"] = len(tool_calls or [])
        except Exception:
            logger.debug("_persist_assistant_turn: usage capture failed", exc_info=True)
            usage = {}
        kwargs = dict(role="assistant", sender=self.config.user_id,
                      room=room_id, event_id=None, content=content or "")
        if tool_calls is not None:
            logged = []
            for tc in tool_calls:
                entry = {"call_id": tc.id, "name": tc.name, "input": tc.input}
                # Persist opaque provider metadata (e.g. Google's
                # extra_content.google.thought_signature) so it survives
                # rehydration and can be echoed back on a later turn — see
                # ToolCall.extra_content docstring / bead workspace-kdsn.186.18.
                extra_content = getattr(tc, "extra_content", None)
                if extra_content:
                    entry["extra_content"] = extra_content
                logged.append(entry)
            kwargs["tool_calls"] = logged
        if thinking:
            kwargs["thinking"] = thinking
        if usage:
            kwargs["usage"] = usage
        _sl.append(**kwargs)

    def _advisor_display_model(self, mdl) -> str:
        """Expand a model alias to its full provider/model for display in the
        advisor notice (idempotent for already-full strings and non-aliases).
        Aliases live on the AGENT's config, not the MatrixConfig — mirrors
        resolve_model's expansion so the notice names the model that ran."""
        aliases = getattr(getattr(self.agent, "config", None), "model_aliases", None) or {}
        key = str(mdl)
        return aliases.get(key, key)

    def _make_tool_callbacks(self, room_id: str):
        """Create tool-use callback closures bound to a specific room.

        Returns a ``(_tool_notice, _tool_intent)`` tuple suitable for passing
        to the agent's ``process`` call.  Both closures share a private
        ``_subagent_start_times`` dict so that elapsed-time tracking works
        across the intent→notice lifecycle.

        This factory exists to eliminate the duplicated closure definitions
        that previously lived in both the heartbeat and streaming message
        code paths.
        """
        _subagent_start_times: dict[str, float] = {}

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
                task_preview = input_data.get("task", "")
                model_info = input_data.get("model", "default")
                result_preview = str(result) if result else ""
                # R8: cap an ERROR result preview before it enters the notice —
                # parity with the generic error branch's detail-text cap below.
                if is_error and len(result_preview) > 2000:
                    result_preview = result_preview[:2000] + "\n[truncated]"
                # Calculate elapsed time if we have a start timestamp
                elapsed_str = ""
                start_ts = _subagent_start_times.pop(call_id, None)
                if start_ts is not None:
                    elapsed = time.monotonic() - start_ts
                    if elapsed >= 60:
                        mins, secs = divmod(int(elapsed), 60)
                        elapsed_str = f" — {mins}m{secs:02d}s"
                    else:
                        elapsed_str = f" — {elapsed:.1f}s"
                summary_line = f"🤖 subagent ({model_info}) {status}{elapsed_str}"
                html = f'<b>{summary_line}</b>'
                # R8: html.escape (not raw mistune.html) — task/result previews
                # are tool-influenced text and must never pass raw HTML to the
                # client. Cosmetic markdown loss here is an accepted tradeoff
                # (audit decision).
                if task_preview:
                    html += (
                        f'\n<details><summary>📋 Task brief</summary>\n'
                        f'{html_escape(task_preview)}</details>'
                    )
                if result_preview:
                    html += (
                        f'\n<details><summary>📨 Result</summary>\n'
                        f'{html_escape(result_preview)}</details>'
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
            elif name == "todo_write" and not is_error:
                # §9: todo_write → 📋 m.notice with summary + collapsed full list
                # Render from STRUCTURED input_data, NOT the `result` string:
                # in production `result` is the wrapped
                # `<tool_result tool="todo_write" id="...">...</tool_result>`
                # envelope built by agent.py's wrap_tool_result, so deriving
                # the notice from it leaked the envelope + call id and
                # markdown collapsed the single-\n-separated todo lines onto
                # one line. input_data is {"todos": [{"content","status",
                # "activeForm"?}, ...]} — build the notice from that instead.
                todos = input_data.get("todos", []) if isinstance(input_data, dict) else []
                if not isinstance(todos, list):
                    todos = []
                status_markers = {"in_progress": "●", "pending": "○", "completed": "✓"}
                n_in_progress = sum(1 for t in todos if isinstance(t, dict)
                                     and t.get("status") == "in_progress")
                n_pending = sum(1 for t in todos if isinstance(t, dict)
                                 and t.get("status") == "pending")
                n_completed = sum(1 for t in todos if isinstance(t, dict)
                                   and t.get("status") == "completed")
                summary_line = (
                    f"{n_in_progress} in progress · {n_pending} pending"
                    f" · {n_completed} completed"
                )
                active = ""
                for t in todos:
                    if isinstance(t, dict) and t.get("status") == "in_progress":
                        active = str(t.get("content", ""))
                        break
                if active:
                    summary_line += f" — {active}"
                todo_body = f"📋 Todos: {summary_line}"
                todo_lines = [
                    f"{status_markers.get(t.get('status'), '?')} "
                    f"{html_escape(str(t.get('content', '')))}"
                    for t in todos if isinstance(t, dict)
                ]
                todo_html = (
                    '<details>\n<summary>📋 Todos: ' + html_escape(summary_line) + '</summary>\n'
                    + '<br>'.join(todo_lines) +
                    '</details>'
                )
                todo_content = {
                    "msgtype": "m.notice",
                    "body": todo_body,
                    "format": "org.matrix.custom.html",
                    "formatted_body": todo_html,
                }
                try:
                    await self._room_send_with_retry(room_id, todo_content)
                except Exception:
                    pass
            elif name == "advisor":
                # R5 (audit remediation): room-scoped key — the bot-wide
                # _advisor_results dict keyed by call_id alone would collide
                # across concurrent rooms; room_id is in this closure's
                # scope (_make_tool_callbacks(self, room_id)).
                info = self._advisor_results.pop((room_id, call_id), {})
                advice = info.get("advice") or (str(result) if result else "")
                elapsed = info.get("elapsed_s", 0.0)
                itok = info.get("input_tokens", 0)
                otok = info.get("output_tokens", 0)
                crd = info.get("cache_read_tokens", 0)
                cct = info.get("cache_creation_tokens", 0)
                cost_usd = info.get("cost_usd", 0.0)
                adv_unpriced = info.get("unpriced_tokens", 0)
                mdl = self._advisor_display_model(info.get("model", "?"))
                if is_error:
                    summary_line = f"🔮 Advisor failed: {str(result)[:200]}"
                    adv_html = html_escape(summary_line)
                else:
                    summary_line = f"🔮 Advisor returned · {mdl} ({elapsed:.1f}s · {itok}/{otok} tokens · cache read {crd})"
                    adv_html = f'<b>{html_escape(summary_line)}</b>'
                    if advice:
                        # FIX 3 (kdsn.198.9): preserve advice line breaks — HTML folds
                        # literal newlines to spaces. Keep html_escape (injection defense
                        # for model-origin advice); NOT mistune markdown rendering.
                        adv_html += f'\n<details><summary>🔮 advice</summary>\n{_escape_preserve_breaks(advice)}</details>'
                try:
                    await self._room_send_with_retry(room_id, {
                        "msgtype": "m.notice", "body": summary_line,
                        "format": "org.matrix.custom.html", "formatted_body": adv_html,
                    })
                except Exception:
                    pass
                if not is_error:
                    _sl_adv = getattr(self, 'session_log', None)
                    if _sl_adv:
                        _sl_adv.append(
                            role="system", sender=self.config.user_id, room=room_id,
                            event_id=None, event="advisor_consult", model=mdl,
                            input_tokens=itok, output_tokens=otok,
                            cache_read_tokens=crd, elapsed_s=elapsed,
                            cache_creation_tokens=cct, cost_usd=cost_usd,
                            unpriced_tokens=adv_unpriced,
                        )
                        # F4 (kdsn.218): accumulate advisor cost LIVE so /status
                        # is accurate before restart. Disjoint from the restart
                        # re-sum (restore_usage assigns once at _activate_room,
                        # before any live turn). Kept INSIDE the `if _sl_adv:`
                        # persistence guard (re-audit LOW-2) so live and
                        # post-restart totals stay identical: no session log =>
                        # neither persisted nor live-counted.
                        try:
                            _uu = self.agent._usage_for(room_id)
                            _uu["advisor_cost_usd"] += cost_usd or 0.0
                            _uu["unpriced_tokens"] += adv_unpriced or 0
                        except Exception:
                            pass
            else:
                notice_body = f"🔧 {name}{detail} {status}"
                if is_error:
                    result_str = str(result) if result else ""
                    if len(result_str) > 2000:
                        detail_text = result_str[:2000] + "\n[error detail truncated]"
                    else:
                        detail_text = result_str
                    error_html = (
                        html_escape(notice_body)
                        + '\n<details>\n<summary>⚠️ Error detail</summary>\n'
                        + html_escape(detail_text)
                        + '</details>'
                    )
                    content_msg = {
                        "msgtype": "m.notice",
                        "body": notice_body,
                        "format": "org.matrix.custom.html",
                        "formatted_body": error_html,
                    }
                    try:
                        await self._room_send_with_retry(room_id, content_msg)
                    except Exception:
                        pass
                else:
                    try:
                        await self.send_notice(room_id, notice_body)
                    except Exception:
                        pass
            # Append tool result to session log
            _sl = getattr(self, 'session_log', None)
            if _sl:
                _tool_kwargs = dict(
                    role="tool",
                    sender=self.config.user_id,
                    room=room_id,
                    event_id=None,
                    call_id=call_id,
                    name=name,
                    output=result,
                    is_error=is_error,
                )
                if name == "subagent":
                    _sub_info = getattr(self, "_subagent_results", {}).pop((room_id, call_id), {})
                    _tool_kwargs["cost_usd"] = _sub_info.get("cost_usd", 0.0)
                    _tool_kwargs["unpriced_tokens"] = _sub_info.get("unpriced_tokens", 0)
                    # F4 (kdsn.218): accumulate subagent cost LIVE (same pop —
                    # one read of the bridge dict feeds both the JSONL kwargs and
                    # the live counter). Disjoint from the restart re-sum.
                    try:
                        _uu = self.agent._usage_for(room_id)
                        _uu["subagent_cost_usd"] += _sub_info.get("cost_usd", 0.0) or 0.0
                        _uu["unpriced_tokens"] += _sub_info.get("unpriced_tokens", 0) or 0
                    except Exception:
                        pass
                _sl.append(**_tool_kwargs)

        async def _tool_intent(tool_calls, content_text):
            # Emit Matrix notice for subagent dispatch
            for tc in tool_calls:
                if tc.name == "subagent" and isinstance(tc.input, dict):
                    _subagent_start_times[tc.id] = time.monotonic()
                    task_preview = tc.input.get("task", "")
                    model_info = tc.input.get("model", "default")
                    iters = tc.input.get("max_iterations", 200)
                    summary = f"⚙️ Spawning sub-agent ({model_info}, max {iters} iters)"
                    html = f'<b>{summary}</b>'
                    # R8: html.escape (not raw mistune.html) — the dispatch task
                    # brief is tool-influenced text; never pass raw HTML through.
                    if task_preview:
                        html += (
                            f'\n<details><summary>📋 Task brief</summary>\n'
                            f'{html_escape(task_preview)}</details>'
                        )
                    body_text = f"{summary}: {task_preview[:200]}"
                    dispatch_msg = {
                        "msgtype": "m.notice",
                        "body": body_text,
                        "format": "org.matrix.custom.html",
                        "formatted_body": html,
                    }
                    try:
                        await self._room_send_with_retry(room_id, dispatch_msg)
                    except Exception:
                        pass
                elif tc.name == "advisor" and isinstance(tc.input, dict):
                    # FIX 2 (kdsn.198.9): show the RESOLVED advisor model, not the
                    # literal word "advisor". The tool hasn't run yet at spawn, so
                    # mirror run_advisor's resolution (call-param > tool_config) and
                    # expand any alias — byte-identical to the model the return
                    # notice will show.
                    _adv_cfg = next(
                        (t.config for t in (getattr(self.agent, "tools", None) or [])
                         if getattr(t, "name", None) == "advisor"), {})
                    mdl = self._advisor_display_model(
                        str(tc.input.get("model") or (_adv_cfg or {}).get("model") or "advisor"))
                    # N1 (audit remediation): `focus` is executor-authored and
                    # is about to be echoed into the room's spawn notice --
                    # redact it before display, same as the egress-side fix
                    # in tools/advisor.py's run_advisor.
                    from openalph.tools.security import redact_credentials
                    focus = redact_credentials(str(tc.input.get("focus", "")))[0]
                    summary = f"🔮 Advisor consult → {mdl}"
                    if focus:
                        # FIX 1 (kdsn.198.9): short one-line preview on the summary
                        # line; full focus under a collapsed <details> fold (same
                        # pattern as the subagent task-brief notice).
                        preview = focus.replace("\n", " ").replace("\r", " ")[:120]
                        summary += f" — focus: {preview}"
                    formatted = f'<b>{html_escape(summary)}</b>'
                    if focus:
                        formatted += (f'\n<details><summary>🔮 full focus</summary>\n'
                                      f'{_escape_preserve_breaks(focus)}</details>')
                    try:
                        await self._room_send_with_retry(room_id, {
                            "msgtype": "m.notice", "body": summary,
                            "format": "org.matrix.custom.html",
                            "formatted_body": formatted,
                        })
                    except Exception:
                        pass
            self._persist_assistant_turn(room_id, content=content_text or "", tool_calls=tool_calls)

        return _tool_notice, _tool_intent

    def _build_agent_callbacks(self, room_id: str, turn_source: str | None) -> dict:
        """Build the reminder/registry/identity callbacks for handle_input (R1 refactor).

        Used by BOTH _process_message and _run_heartbeat_turn so the wiring
        is identical.  Returns a dict with at minimum: send_notice, log_reminder,
        turn_source, read_registry, room_id, context_status, send_media,
        on_redaction.
        """

        async def _reminder_send_notice(_room_id, body, **kw):
            """Emit collapsed <details> m.notice for reminders.

            Summary holds the header line (lines[0]); the details body
            renders only the remainder, so the header isn't re-rendered
            inside the expanded body (mirrors the thinking-block pattern).
            """
            lines = body.split('\n')
            html = (
                '<details>\n<summary>' + lines[0] + '</summary>\n'
                + mistune.html('\n'.join(lines[1:]).strip()) +
                '</details>'
            )
            content_msg = {
                "msgtype": "m.notice",
                "body": body,
                "format": "org.matrix.custom.html",
                "formatted_body": html,
            }
            await self._room_send_with_retry(_room_id, content_msg)

        async def _log_reminder(_room_id, reminder):
            """Log reminder to JSONL with source='reminder' + trigger."""
            _sl = getattr(self, 'session_log', None)
            if _sl:
                _sl.append(
                    role="user",
                    sender=self.config.user_id,
                    room=_room_id,
                    event_id=None,
                    content=reminder.content,
                    source="reminder",
                    trigger=reminder.trigger,
                )

        async def _context_status_callback(req_room_id=None):
            return self._build_context_status(req_room_id or room_id)

        async def _upload_callback(file_path, content_type, filename, caption=None):
            await self.upload_and_send(room_id, file_path, content_type, filename, caption)

        async def _redaction_notice(tool_name, events):
            """Emit in-room notice when credentials are redacted from tool output."""
            for event in events:
                notice = f"🔒 Credential redacted in {tool_name} output: {event.pattern_name} ({event.char_count} chars)"
                try:
                    await self.send_notice(room_id, notice)
                except Exception as exc:
                    logger.error("Redaction notice failed in %s: %s", room_id, exc, exc_info=True)
                _sl = getattr(self, 'session_log', None)
                if _sl:
                    _sl.append(
                        role="system",
                        sender=self.config.user_id,
                        room=room_id,
                        event_id=None,
                        event="credential_redaction",
                        detail=f"tool={tool_name} pattern={event.pattern_name} chars={event.char_count}",
                    )

        async def _keepalive_miss_notice(_room_id=None):
            """Emit notice + system log when cache keepalive detects a write (miss)."""
            rid = _room_id or room_id
            notice = ("⚠️ cache keepalive missed (wrote instead of read) -- "
                      "disabling for this turn; next resume may bust cache")
            try:
                await self.send_notice(rid, notice)
            except Exception as exc:
                logger.error("cache keepalive miss notice failed in %s: %s", rid, exc, exc_info=True)
            _sl = getattr(self, 'session_log', None)
            if _sl:
                _sl.append(
                    role="system",
                    sender=self.config.user_id,
                    room=rid,
                    event_id=None,
                    event="cache_keepalive_miss",
                    detail="ping wrote instead of read",
                )

        # R1-1: per-room read registry for file_write guard
        # Use getattr for compatibility with mocked agents in test suites
        _registries = getattr(self.agent, '_read_registries', None)
        if _registries is None:
            _registries = {}
            try:
                self.agent._read_registries = _registries
            except AttributeError:
                pass  # spec-mocked agent, read_registry will be empty dict
        _read_registry = _registries.setdefault(room_id, {})

        # advisor: per-room consult counter (getattr for mocked-agent compatibility)
        _advisor_uses = getattr(self.agent, '_advisor_uses', None)
        if _advisor_uses is None:
            _advisor_uses = {}
            try:
                self.agent._advisor_uses = _advisor_uses
            except AttributeError:
                pass
        # Lazy-init for tests that construct MatrixBot via __new__
        if not hasattr(self, '_advisor_results'):
            self._advisor_results = {}
        if not hasattr(self, '_subagent_results'):
            self._subagent_results = {}

        return {
            "send_media": _upload_callback,
            "on_redaction": _redaction_notice,
            "on_keepalive_miss": _keepalive_miss_notice,
            "context_status": _context_status_callback,
            "send_notice": _reminder_send_notice,
            "log_reminder": _log_reminder,
            "turn_source": turn_source,
            "read_registry": _read_registry,       # R1-1
            "room_id": room_id,                    # R1-2
            "get_transcript": lambda: (self.agent.system_prompt, list(self.agent.history(room_id))),
            "advisor_uses": _advisor_uses,
            "advisor_results": self._advisor_results,
            "subagent_results": self._subagent_results,
        }

    async def _run_heartbeat_turn(self, room_id: str, content: str, *, turn_source: str | None = None) -> None:
        """Execute a heartbeat/umbral turn: activate room, process input, deliver response.

        Extracted from _inject_heartbeat for reuse by _inject_umbral.
        Raises on error — caller is responsible for error handling.
        """
        # Activate room if not already active
        if room_id not in self._active_rooms:
            await self._activate_room(room_id)

        # -- tool-use callbacks (same as normal message path) --
        _tool_notice, _tool_intent = self._make_tool_callbacks(room_id)

        # Process through agent
        try:
            await self._set_typing(room_id, True)

            # Resolve thinking level: room override > config
            _thinking_override = getattr(self, '_room_thinking', {}).get(room_id)
            _cache_ttl = getattr(self, '_room_cache_ttl', {}).get(room_id)
            _thinking_buffer = []
            _thinking_done = False
            _thinking_notified = False

            async def _thinking_delta(text: str, done: bool):
                nonlocal _thinking_done, _thinking_notified
                if not done:
                    _thinking_buffer.append(text)
                    # Send a one-time notice on first thinking chunk
                    if not _thinking_notified:
                        _thinking_notified = True
                        try:
                            await self.send_notice(room_id, "💭 Thinking…")
                            await self._set_typing(room_id, True)
                        except Exception:
                            pass
                else:
                    _thinking_done = True
                    _thinking_notified = False  # reset for next tool-loop iteration
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

            async def _cache_status(usage, model_str):
                """Emit in-room notice on significant prompt cache miss if provider opted in."""
                cr = usage.cache_read_tokens or 0
                cc = usage.cache_creation_tokens or 0
                # Only fire when a significant amount was re-cached
                if cc < 10000:
                    return
                # Check if the provider has cache_bust_notices enabled
                try:
                    from openalph.config import resolve_model
                    provider_cfg, _ = resolve_model(model_str, self.agent.config.providers, aliases=self.agent.config.model_aliases)
                    if not provider_cfg.cache_bust_notices:
                        return
                except Exception:
                    return
                # Calculate miss percentage
                total = cr + cc
                miss_pct = (cc / total * 100) if total > 0 else 100
                # Format the notice
                notice = f"⚠️ Cache warning — {cc:,} tokens written, {cr:,} read ({miss_pct:.0f}% uncached)"
                try:
                    await self.send_notice(room_id, notice)
                except Exception as exc:
                    logger.error("Cache warning send_notice failed in %s: %s", room_id, exc, exc_info=True)
                # Log as system entry (excluded from LLM context)
                _sl = getattr(self, 'session_log', None)
                if _sl:
                    _sl.append(
                        role="system",
                        sender=self.config.user_id,
                        room=room_id,
                        event_id=None,
                        event="cache_warning",
                        detail=f"cache_read={cr} cache_creation={cc} miss_pct={miss_pct:.0f}",
                    )

            # Timesense: prepend timestamp to heartbeat content
            if getattr(self, '_room_timesense', {}).get(room_id):
                from datetime import datetime, timezone
                _ts = datetime.now(timezone.utc).astimezone().strftime("%A, %B %d, %Y — %H:%M %Z")
                content = f"[{_ts}] {content}"

            # R1 refactor: use shared _build_agent_callbacks for identical wiring
            callbacks = self._build_agent_callbacks(room_id, turn_source)

            response = await self.agent.handle_input(
                content,
                room_id,
                on_tool_call=_tool_notice,
                on_tool_intent=_tool_intent,
                thinking=_thinking_override,
                on_thinking_delta=_thinking_delta,
                on_cache_status=_cache_status,
                cache_ttl=_cache_ttl,
                callbacks=callbacks,
            )
            if response and response.strip():
                self._persist_assistant_turn(room_id, content=response)
                await self.send(room_id, response)
            else:
                _stop = self.agent.last_stop_reason(room_id)
                if _stop == "refusal":
                    logger.warning("Model refusal during heartbeat in %s", room_id)
                    await self.send(room_id,
                        "⚠️ **Model refusal** — heartbeat turn refused by model "
                        "(API returned `stop_reason: refusal`). Content policy "
                        "restrictions were triggered. Consider switching models "
                        "with `/model`.")
                else:
                    # Model did tool work but returned empty text.  Retry once
                    # with a nudge — the model sees its own tool results in
                    # history and should produce the report it failed to emit.
                    logger.warning("Empty heartbeat response in %s — retrying once", room_id)
                    # R2-D-retry: pass same callbacks as primary call
                    retry = await self.agent.handle_input(
                        "[SYSTEM: Your previous heartbeat response was empty. "
                        "Summarize your findings now.]",
                        room_id,
                        on_tool_call=_tool_notice,
                        on_tool_intent=_tool_intent,
                        thinking=_thinking_override,
                        on_thinking_delta=_thinking_delta,
                        on_cache_status=_cache_status,
                        cache_ttl=_cache_ttl,
                        callbacks=callbacks,
                    )
                    if retry and retry.strip():
                        self._persist_assistant_turn(room_id, content=retry)
                        await self.send(room_id, retry)
                    else:
                        logger.warning("Empty heartbeat response in %s after retry — giving up", room_id)
                        await self.send(room_id,
                            "⚠️ **Empty heartbeat response** — the model returned no content "
                            "after retry. This may indicate degeneration or a provider issue.")
        finally:
            await self._set_typing(room_id, False)

    def _build_context_status(self, rid: str) -> dict:
        """Assemble context status data dict for the context_status tool.

        Collects agent status, session age, heartbeat state, umbral state,
        and room identity for the given room ID. Sync-safe: calls only sync
        methods on self.agent, self.session_log, self.heartbeat, self.umbral.
        """
        from datetime import datetime, timezone
        _hist = self.session_log.build_context(rid) if getattr(self, 'session_log', None) else None
        status_data = self.agent.status(rid, history=_hist)

        # Session age
        if getattr(self, 'session_log', None):
            entries = self.session_log.read(rid)
            if entries:
                first_ts = entries[0].get("ts", "")
                try:
                    first_dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                    age = (datetime.now(timezone.utc) - first_dt).total_seconds() / 60
                    status_data["session_age_minutes"] = round(age)
                except (ValueError, TypeError):
                    status_data["session_age_minutes"] = None
            else:
                status_data["session_age_minutes"] = None
        else:
            status_data["session_age_minutes"] = None

        # Heartbeat state
        hb = getattr(self, 'heartbeat', None)
        if hb and hb.is_active(rid):
            hb_entries = hb.status()
            hb_entry = next((e for e in hb_entries if e.room_id == rid), None)
            if hb_entry:
                status_data["heartbeat_active"] = True
                status_data["heartbeat_interval_minutes"] = round(hb_entry.interval_seconds / 60)
                status_data["heartbeat_next_minutes"] = round(hb_entry.seconds_until_next / 60)
            else:
                status_data["heartbeat_active"] = False
                status_data["heartbeat_interval_minutes"] = None
                status_data["heartbeat_next_minutes"] = None
        else:
            status_data["heartbeat_active"] = False
            status_data["heartbeat_interval_minutes"] = None
            status_data["heartbeat_next_minutes"] = None

        # Umbral state (context rotation timer)
        um = getattr(self, 'umbral', None)
        if um and um.is_active(rid):
            um_entries = um.status()
            um_entry = next((e for e in um_entries if e.room_id == rid), None)
            if um_entry:
                status_data["umbral_active"] = True
                status_data["umbral_interval_minutes"] = round(um_entry.interval_seconds / 60)
                status_data["umbral_next_minutes"] = round(um_entry.seconds_until_next / 60)
            else:
                status_data["umbral_active"] = False
                status_data["umbral_interval_minutes"] = None
                status_data["umbral_next_minutes"] = None
        else:
            status_data["umbral_active"] = False
            status_data["umbral_interval_minutes"] = None
            status_data["umbral_next_minutes"] = None

        # Room identity
        status_data["room_id"] = rid
        room = self.client.rooms.get(rid) if getattr(self, 'client', None) else None
        status_data["room_name"] = room.named_room_name() if room else None

        return status_data

    async def _inject_heartbeat(self, room_id: str) -> None:
        """Process a heartbeat as if the agent received a wake message."""
        _d = self.heartbeat.directive_for(room_id) if self.heartbeat else None
        if _d:
            heartbeat_content = f"[Automated heartbeat turn — operator may not be present. Your standing directive for this room:]\n\n{escape_system_reminder_tags(_d)}"
        else:
            heartbeat_content = "[Automated heartbeat — operator may not be present. Execute your WAKE instructions.]"

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
            await self._run_heartbeat_turn(room_id, heartbeat_content, turn_source="heartbeat")
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
        _d = self.umbral.directive_for(room_id) if self.umbral else None
        if _d:
            heartbeat_content = f"[Automated umbral turn — operator may not be present. Your standing directive for this room:]\n\n{escape_system_reminder_tags(_d)}"
        else:
            heartbeat_content = "[Automated heartbeat — operator may not be present. Execute your WAKE instructions.]"

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
            await self._run_heartbeat_turn(room_id, heartbeat_content, turn_source="umbral")
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
            # Clear todo state for this room (session-scoped; umbral = new session)
            try:
                from openalph.tools import _TODO_STATE
                _TODO_STATE.pop(room_id, None)
            except Exception:
                pass
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
                self.agent.restore_usage(room_id, session_log.usage_totals(room_id))

                # R1-4: Rehydrate per-room reminder engine fired-state from JSONL
                # (e.g., a once-per-session trigger does not re-fire after restart)
                self.agent.rehydrate_reminders(room_id, existing)

                # R2-8: Rehydrate per-room tool counts from JSONL.
                # Scan assistant entries with tool_calls for per-tool name counts.
                # This ensures T3 suppression by prior memory_search survives restart.
                _tool_counts: dict[str, int] = {}
                _last_todo_write_args = None
                advisor_count = 0
                for entry in existing:
                    if entry.get("role") == "assistant" and entry.get("tool_calls"):
                        for _tc in entry["tool_calls"]:
                            _tc_name = _tc.get("name", "")
                            if _tc_name:
                                _tool_counts[_tc_name] = _tool_counts.get(_tc_name, 0) + 1
                            if _tc_name == "advisor":
                                advisor_count += 1
                            # R2-todo-rehydrate: capture last todo_write args
                            if _tc_name == "todo_write":
                                _tc_input = _tc.get("input")
                                if isinstance(_tc_input, dict) and "todos" in _tc_input:
                                    _last_todo_write_args = _tc_input["todos"]
                if _tool_counts:
                    self.agent._room_tool_counts[room_id] = _tool_counts
                # advisor: rehydrate per-room consult counter (survives restart)
                try:
                    self.agent._advisor_uses[room_id] = advisor_count
                except (AttributeError, TypeError):
                    pass

                # R2-todo-rehydrate: restore _TODO_STATE from last todo_write call
                if _last_todo_write_args is not None:
                    from openalph.tools import _TODO_STATE
                    if isinstance(_last_todo_write_args, list):
                        _TODO_STATE[room_id] = _last_todo_write_args
                    else:
                        _TODO_STATE[room_id] = []

                # Restore per-room overrides (model, thinking) from session log.
                # Scan all entries — last override wins (user may have switched multiple times).
                _restored_model = None
                _restored_thinking = None
                _restored_cache_ttl = None
                _restored_timesense = None
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
                        elif ev == "cache_ttl_override" and detail:
                            if not hasattr(self, "_room_cache_ttl"):
                                self._room_cache_ttl = {}
                            if detail != "1h":
                                self._room_cache_ttl[room_id] = detail
                            _restored_cache_ttl = detail
                        elif ev == "timesense_override" and detail:
                            if not hasattr(self, "_room_timesense"):
                                self._room_timesense = {}
                            if detail == "on":
                                self._room_timesense[room_id] = True
                            _restored_timesense = detail

                # Send session resume notice to Matrix
                parts = [f"🔄 **Session resumed** — {len(existing)} prior entries"]
                if _restored_model:
                    parts.append(f"Model override: `{_restored_model}`")
                if _restored_thinking:
                    parts.append(f"Thinking: `{_restored_thinking}`")
                if _restored_cache_ttl:
                    parts.append(f"Cache TTL: `{_restored_cache_ttl}`")
                if _restored_timesense:
                    parts.append(f"Timesense: `{_restored_timesense}`")
                try:
                    await self.send_notice(room_id, " · ".join(parts))
                except Exception:
                    pass
            else:
                # New room: start fresh, log session start + system prompt
                prompt = getattr(self.agent, "system_prompt", None)
                session_log.append(
                    role="system",
                    sender=self.config.user_id,
                    room=room_id,
                    event="session_start",
                    detail="New room, starting fresh",
                    **({"system_prompt": prompt} if isinstance(prompt, str) else {}),
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
        _drain_steering_fn = None  # set after session lock acquired; used in outer finally

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
            # After hydration, history already contains the current user message
            # (written to JSONL in _handle_room_message before this path).
            # Record that fact so we can skip the re-append in handle_input.
            user_already_in_history = False
            if gated and room_id in self._active_rooms and self.session_log:
                history = self.agent.history(room_id)
                history.clear()
                history.extend(self.session_log.build_context(room_id))
                logger.info("Hydrated context for %s: %d entries", room_id, len(history))
                user_already_in_history = True
            # --- End mention gating ---

            # Lazy wake: activate room on first live message
            if room_id not in self._active_rooms:
                room_name = getattr(room, 'name', '') or getattr(room, 'display_name', '') or room_id
                await self._activate_room(room_id, room_name=room_name)
                # _activate_room hydrates history from JSONL (which already includes
                # the current user message for gated rooms).  Mark it so handle_input
                # does not re-append.
                if gated and self.session_log:
                    user_already_in_history = True

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
            # Timesense: prepend timestamp to user message for LLM context
            if getattr(self, '_room_timesense', {}).get(room_id):
                from datetime import datetime, timezone
                _ts = datetime.now(timezone.utc).astimezone().strftime("%A, %B %d, %Y — %H:%M %Z")
                body = f"[{_ts}] {body}"
            await self._set_typing(room_id, True)

            # Wire tool visibility for this turn
            _tool_notice, _tool_intent = self._make_tool_callbacks(room_id)

            try:
                # Resolve thinking level: room override > config
                _thinking_override = getattr(self, '_room_thinking', {}).get(room_id)
                _cache_ttl = getattr(self, '_room_cache_ttl', {}).get(room_id)

                # R1 refactor: use shared _build_agent_callbacks for identical wiring
                callbacks = self._build_agent_callbacks(room_id, None)

                # Set up streaming delivery
                streaming = StreamingDelivery(self, room_id)
                _thinking_buffer = []
                _thinking_done = False
                _thinking_notified = False

                async def _text_delta(text: str, done: bool):
                    await streaming.push(text, done=done)

                async def _thinking_delta(text: str, done: bool):
                    nonlocal _thinking_done, _thinking_notified
                    if not done:
                        _thinking_buffer.append(text)
                        # Send a one-time notice on first thinking chunk
                        if not _thinking_notified:
                            _thinking_notified = True
                            try:
                                await self.send_notice(room_id, "💭 Thinking…")
                                await self._set_typing(room_id, True)
                            except Exception:
                                pass
                    else:
                        _thinking_done = True
                        _thinking_notified = False  # reset for next tool-loop iteration
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

                async def _cache_status(usage, model_str):
                    """Emit in-room notice on significant prompt cache miss if provider opted in."""
                    cr = usage.cache_read_tokens or 0
                    cc = usage.cache_creation_tokens or 0
                    # Only fire when a significant amount was re-cached
                    if cc < 10000:
                        return
                    # Check if the provider has cache_bust_notices enabled
                    try:
                        from openalph.config import resolve_model
                        provider_cfg, _ = resolve_model(model_str, self.agent.config.providers, aliases=self.agent.config.model_aliases)
                        if not provider_cfg.cache_bust_notices:
                            return
                    except Exception:
                        return
                    # Calculate miss percentage
                    total = cr + cc
                    miss_pct = (cc / total * 100) if total > 0 else 100
                    # Format the notice
                    notice = f"⚠️ Cache warning — {cc:,} tokens written, {cr:,} read ({miss_pct:.0f}% uncached)"
                    try:
                        await self.send_notice(room_id, notice)
                    except Exception as exc:
                        logger.error("Cache warning send_notice failed in %s: %s", room_id, exc, exc_info=True)
                    # Log as system entry (excluded from LLM context)
                    _sl = getattr(self, 'session_log', None)
                    if _sl:
                        _sl.append(
                            role="system",
                            sender=self.config.user_id,
                            room=room_id,
                            event_id=None,
                            event="cache_warning",
                            detail=f"cache_read={cr} cache_creation={cc} miss_pct={miss_pct:.0f}",
                        )

                # Build drain_steering closure and mark turn active.
                # The closure pops the inbox atomically (list.pop is GIL-safe),
                # logs each note to JSONL, emits a "delivered" notice, and
                # returns the note strings so the agent loop can inject them.
                # _active_turns is set here, BEFORE handle_input, so that a
                # concurrent /steer command can detect the active turn.
                if not hasattr(self, '_steering_inbox'):
                    self._steering_inbox = {}
                if not hasattr(self, '_active_turns'):
                    self._active_turns = set()
                self._active_turns.add(room_id)

                async def _drain_steering() -> list:
                    notes = self._steering_inbox.pop(room_id, [])
                    for _n in notes:
                        # Log to JSONL: role=user, source=steer, original text
                        _sl2 = getattr(self, 'session_log', None)
                        if _sl2:
                            _sl2.append(
                                role="user",
                                sender=self.config.user_id,
                                room=room_id,
                                event_id=None,
                                content=_n,
                                source="steer",
                            )
                        # Emit "delivered" notice to operator
                        try:
                            await self.send_notice(room_id, "🧭 Steering note delivered")
                        except Exception:
                            pass
                    return notes

                _drain_steering_fn = _drain_steering

                # Pass drain_steering via callbacks dict so that test mocks
                # with explicit handle_input signatures (no drain_steering kwarg)
                # are not broken. agent.handle_input extracts it from callbacks
                # when drain_steering= kwarg is None.
                callbacks['drain_steering'] = _drain_steering

                response = await self.agent.handle_input(
                        body, room_id,
                        on_tool_call=_tool_notice,
                        on_tool_intent=_tool_intent,
                        on_text_delta=_text_delta,
                        on_thinking_delta=_thinking_delta,
                        thinking=_thinking_override,
                        callbacks=callbacks,
                        on_cache_status=_cache_status,
                        cache_ttl=_cache_ttl,
                        append_user=not user_already_in_history,
                    )
                # Append assistant response to session log
                if response and response.strip():
                    self._persist_assistant_turn(room_id, content=response)
                    # Send if streaming didn't deliver, or if the response
                    # differs from what was streamed (e.g. tool-limit summary
                    # generated after the last streamed tool-call text).
                    if not streaming._delivered:
                        await self.send(room_id, response)
                    elif streaming._delivered_text.strip() != response.strip():
                        logger.info("Response differs from streamed content — sending separately")
                        await self.send(room_id, response)
                else:
                    _stop = self.agent.last_stop_reason(room_id)
                    if _stop == "refusal":
                        logger.warning("Model refusal in %s — stop_reason=refusal", room_id)
                        await self.send(room_id,
                            "⚠️ **Model refusal** — the model refused to generate a response "
                            "(API returned `stop_reason: refusal`). This usually means content "
                            "policy restrictions were triggered. Try rephrasing, or switch "
                            "models with `/model`.")
                    else:
                        logger.warning("Empty response from agent in %s — not sending", room_id)
                        await self.send(room_id,
                            "⚠️ **Empty response** — the model returned no content. "
                            "This may indicate degeneration or a provider issue. "
                            "Try again or start a new room.")
                # Check context capacity after successful turn
                try:
                    _sh = None
                    if getattr(self, 'session_log', None):
                        _sh = self.session_log.build_context(room_id)
                    _status = self.agent.status(room_id, history=_sh)
                    _pct = _status.get("context_pct", 0)
                    if _pct >= 80:
                        await self.send_notice(room_id,
                            f"⚠️ Context at **{_pct}%** — "
                            f"~{_status['context_tokens']:,} / {_status['context_max']:,} tokens. "
                            f"Consider starting a new room soon.")
                    # Output-headroom soft notice (Workstream B)
                    try:
                        _ctx = _status.get("context_tokens", 0)
                        _resolved = _status.get("context_max", 0)
                        _max_out = self.agent.config.max_tokens
                        if _resolved and _ctx > _resolved - _max_out:
                            await self.send_notice(room_id,
                                f"ℹ️ Output headroom low — context (~{_ctx:,}) is within "
                                f"the model's output reserve (~{_max_out:,} tokens) of the "
                                f"{_resolved:,} window. Responses may be truncated soon.")
                    except Exception:
                        pass
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
            # Race rule: do a final drain pass to catch notes that arrived
            # during the last iteration, THEN discard from _active_turns.
            # This ensures a note deposited while the loop was on its last
            # iteration is still logged this turn, not silently dropped.
            if _drain_steering_fn is not None:
                try:
                    await _drain_steering_fn()
                except Exception:
                    pass
            if hasattr(self, '_active_turns'):
                self._active_turns.discard(room_id)

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

        # For final edit events, use the replacement content (m.new_content)
        event_content = event_source.get("content", {})
        if event_content.get("m.relates_to", {}).get("rel_type") == "m.replace":
            body = event_content.get("m.new_content", {}).get("body", event.body).strip()
        else:
            body = event.body.strip()

        # --- Mention gating ---
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
        # _process_message and slash commands both see full room history.
        if not hasattr(self, '_active_rooms'):
            self._active_rooms = set()
        if room_id not in self._active_rooms:
            room_name = getattr(room, 'name', '') or getattr(room, 'display_name', '') or room_id
            await self._activate_room(room_id, room_name=room_name)

        # --- Slash commands ---
        if body == "/stop":
            self._halted_rooms.add(room_id)
            # Clear steering inbox so stale notes don't leak into the next turn
            if hasattr(self, '_steering_inbox'):
                self._steering_inbox.pop(room_id, None)
            await self._cancel_current(room_id)
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
            # Assemble the full prompt (system prompt + tool list).
            parts = [self.agent.system_prompt]
            if self.agent.tools:
                parts.append("\n## Available Tools (passed via API, not in prompt)\n")
                for tool in self.agent.tools:
                    params = ", ".join(tool.parameters.get("properties", {}).keys())
                    line = f"- **{tool.name}**: {tool.description}"
                    if params:
                        line += f"\n  Parameters: {params}"
                    parts.append(line)
            output = "\n".join(parts)

            # Count prompt files present, skills indexed, and tools available
            # for the inline summary. Use config.workspace as the source of truth.
            workspace = Path(self.agent.config.workspace)
            prompt_files = [
                "SAFETY.md", "SOUL.md", "OPERATOR.md",
                "WAKE.md", "ENVIRONMENT.md", "OPERATIONS.md",
            ]
            file_count = sum(1 for f in prompt_files if (workspace / f).exists())
            skills_dir = workspace / "skills"
            skill_count = (
                sum(1 for f in skills_dir.iterdir() if f.suffix == ".md")
                if skills_dir.exists() else 0
            )
            tool_count = len(self.agent.tools) if self.agent.tools else 0

            summary = (
                f"📋 System prompt: {len(output):,} chars · "
                f"{file_count} prompt files · "
                f"{skill_count} skills · "
                f"{tool_count} tools"
            )

            # Write prompt to a temp .md file and upload as attachment.
            # Uploading as a file bypasses Markdown rendering entirely, so
            # fenced code blocks inside the prompt render correctly on
            # download rather than colliding with the wrapper fence.
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".md",
                    prefix=f"system-prompt-{self.agent.config.name}-",
                    delete=False,
                    encoding="utf-8",
                ) as tf:
                    tf.write(output)
                    tmp_path = Path(tf.name)

                await self.send_notice(room_id, summary)
                try:
                    await self.upload_and_send(
                        room_id,
                        tmp_path,
                        "text/markdown",
                        f"system-prompt-{self.agent.config.name}.md",
                        caption=f"System prompt ({self.agent.config.name})",
                    )
                finally:
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
            except Exception as exc:
                logger.exception("showprompt failed: %s", exc)
                await self.send(room_id, f"{summary}\n\n⚠️ Upload failed: {exc}")
            return

        if body == "/status":
            # Pass build_context output so the estimate reflects toolstrip
            _status_history = None
            if getattr(self, 'session_log', None):
                _status_history = self.session_log.build_context(room_id)
            status = self.agent.status(room_id, history=_status_history)
            ctx = status['context_tokens']
            ctx_max = status['context_max']
            ctx_pct = status['context_pct']
            bar_len = 20
            filled = round(bar_len * ctx_pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            # Resolve room-scoped overrides
            _thinking = getattr(self, '_room_thinking', {}).get(room_id) or self.agent.config.thinking
            _cache_ttl = getattr(self, '_room_cache_ttl', {}).get(room_id) or "1h (default)"
            lines = [
                f"### {status['name']}",
                "",
                f"| | |",
                f"|---|---|",
                f"| **Model** | `{status['model']}` |",
                f"| **Turns** | {status['turns']} |",
                f"| **Context** | {bar} {ctx_pct}% (~{ctx:,} / {ctx_max:,}) |",
                f"| **Session in (uncached)** | {status['uncached_input_tokens']:,} tokens |",
                f"| **Session in (cache read)** | {status['cache_read_tokens']:,} tokens |",
                f"| **Session in (cache write)** | {status['cache_creation_tokens']:,} tokens |",
                f"| **Session out** | {status['total_output_tokens']:,} tokens |",
                f"| **Tool calls** | {status['total_tool_calls']} |",
            ]
            if status.get('total_cost_usd', 0.0) > 0:
                lines.append(f"| **Session cost (main)** | ${status.get('main_cost_usd', 0.0):.2f} |")
                if status.get('subagent_cost_usd', 0.0) > 0:
                    lines.append(f"| **Session cost (subagent)** | ${status.get('subagent_cost_usd', 0.0):.2f} |")
                if status.get('advisor_cost_usd', 0.0) > 0:
                    lines.append(f"| **Session cost (advisor)** | ${status.get('advisor_cost_usd', 0.0):.2f} |")
                lines.append(f"| **Session cost (total)** | ${status.get('total_cost_usd', 0.0):.2f} |")
            if status.get('unpriced_tokens', 0) > 0:
                lines.append(
                    f"| **Unpriced** | {status.get('unpriced_tokens', 0):,} tokens (non-Anthropic or unlisted model) |"
                )
            lines += [
                f"| **Thinking** | {_thinking} |",
                f"| **Cache TTL** | {_cache_ttl} |",
                f"| **Timesense** | {'on' if getattr(self, '_room_timesense', {}).get(room_id) else 'off'} |",
            ]
            # Add strippable stats if session log available
            if getattr(self, 'session_log', None):
                try:
                    s_count, s_chars = self.session_log.strippable_stats(room_id)
                    if s_count > 0:
                        s_tokens = s_chars // 4
                        lines.append(f"| **Strippable** | {s_count} tool results, ~{s_tokens:,} tokens |")
                except Exception:
                    pass
            await self.send(room_id, "\n".join(lines))
            return

        if body.startswith("/model"):
            parts = body.split(None, 1)
            if len(parts) < 2:
                await self.send(room_id, "Usage: `/model <name-or-alias>` or `/model list`")
                return
            arg = parts[1].strip()
            if arg == "list":
                current = self.agent.get_model(room_id)
                output = format_model_list(self.agent.config.model_aliases, current)
                await self.send(room_id, output)
                return
            new_model = arg
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
            valid_levels = ("off", "low", "medium", "high", "xhigh", "max")
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

        if body.startswith("/cache"):
            parts = body.split(None, 1)
            if len(parts) < 2:
                # Show current cache status: TTL + toolstrip state
                current = self._room_cache_ttl.get(room_id)
                ttl_line = f"Cache TTL: **{current}** (override)" if current else "Cache TTL: **1h** (default)"
                strip_line = "Toolstrip: none"
                if self.session_log:
                    entries = self.session_log.read(room_id)
                    strip_markers = [
                        e.get("entry_index", 0) for e in entries
                        if e.get("role") == "system" and e.get("event") == "toolstrip"
                    ]
                    if strip_markers:
                        boundary = max(strip_markers)
                        # Count what was stripped
                        stripped_count = sum(
                            1 for i, e in enumerate(entries)
                            if e.get("role") == "tool" and i < boundary
                        )
                        strip_line = f"Toolstrip: active at entry {boundary} ({stripped_count} tool results stripped)"
                await self.send(room_id, f"{ttl_line}\n{strip_line}")
                return
            value = parts[1].strip().lower()
            if value == "toolstrip":
                # /cache toolstrip — mark strip point
                if self.session_log:
                    entries = self.session_log.read(room_id)
                    entry_count = len(entries)
                    # Compute what will be stripped
                    # Respect existing strip boundary
                    existing_markers = [
                        e.get("entry_index", 0) for e in entries
                        if e.get("role") == "system" and e.get("event") == "toolstrip"
                    ]
                    existing_boundary = max(existing_markers) if existing_markers else -1
                    new_count = 0
                    new_chars = 0
                    for i, e in enumerate(entries):
                        if e.get("role") == "tool" and i > existing_boundary:
                            new_count += 1
                            new_chars += len(e.get("output", ""))
                    # Append the marker
                    self.session_log.append(
                        role="system",
                        sender=event.sender,
                        room=room_id,
                        event_id=None,
                        event="toolstrip",
                        entry_index=entry_count,
                    )
                    # Refresh in-memory history to reflect the strip
                    history = self.agent.history(room_id)
                    history.clear()
                    history.extend(self.session_log.build_context(room_id))
                    msg = f"Toolstrip applied. Stripped {new_count} tool results (~{new_chars:,} chars) from context."
                    msg += "\n⚠️ Previously loaded skills were stripped — re-read any skills needed for ongoing work."
                    await self.send(room_id, msg)
                else:
                    await self.send(room_id, "⚠️ No session log available.")
                return
            if value == "off":
                value = "1h"
            valid_values = ("5m", "1h")
            if value not in valid_values:
                await self.send(room_id, f"Invalid value. Use: `/cache 1h`, `/cache 5m`, or `/cache off`")
                return
            # Check if current model uses Anthropic provider
            try:
                from openalph.config import resolve_model
                model_str = self.agent.get_model(room_id)
                provider_cfg, _ = resolve_model(model_str, self.agent.config.providers, aliases=self.agent.config.model_aliases)
                if provider_cfg.type != "anthropic":
                    await self.send(room_id,
                        f"⚠️ Cache TTL only applies to Anthropic providers. "
                        f"Current model `{model_str}` uses **{provider_cfg.type}**.")
                    return
            except Exception:
                pass  # If we can't resolve, allow the command anyway
            if value == "1h":
                self._room_cache_ttl.pop(room_id, None)
            else:
                self._room_cache_ttl[room_id] = value
            # Persist to JSONL
            if self.session_log:
                self.session_log.append(
                    role="system",
                    sender=event.sender,
                    room=room_id,
                    event_id=None,
                    event="cache_ttl_override",
                    detail=value,
                )
            await self.send(room_id, f"Cache TTL set to **{value}** for this room")
            return

        if body.startswith("/timesense"):
            parts = body.split(None, 1)
            if len(parts) < 2:
                # Show current state
                current = self._room_timesense.get(room_id, False)
                state = "on" if current else "off (default)"
                await self.send(room_id, f"Timesense: **{state}**")
                return
            value = parts[1].strip().lower()
            if value not in ("on", "off"):
                await self.send(room_id, "Invalid value. Use: `/timesense on` or `/timesense off`")
                return
            enabled = (value == "on")
            if enabled:
                self._room_timesense[room_id] = True
            else:
                self._room_timesense.pop(room_id, None)
            # Persist to JSONL
            if self.session_log:
                self.session_log.append(
                    role="system",
                    sender=event.sender,
                    room=room_id,
                    event_id=None,
                    event="timesense_override",
                    detail=value,
                )
            await self.send(room_id, f"Timesense set to **{value}** for this room")
            return

        if body.startswith("/heartbeat"):
            parts = body.split(None, 3)
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
                    directive = parts[3] if len(parts) >= 4 else None
                    await self.heartbeat.start(room_id, interval, directive)
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
                        line = f"- **{name}** — every {format_interval(e.interval_seconds)}, next in {format_interval(e.seconds_until_next)}"
                        if e.directive:
                            line += f" · directive: {_trunc_directive(e.directive)}"
                        lines.append(line)
                    await self.send(room_id, "\n".join(lines))
            else:
                await self.send(room_id, "Usage: `/heartbeat start <interval>` | `/heartbeat stop` | `/heartbeat status`")
            return

        if body.startswith("/umbral"):
            parts = body.split(None, 3)
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
                    directive = parts[3] if len(parts) >= 4 else None
                    await self.umbral.start(room_id, interval, directive)
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
                        line = (
                            f"- **{name}** — every {format_interval(e.interval_seconds)}, "
                            f"next in {format_interval(e.seconds_until_next)}")
                        if e.directive:
                            line += f" · directive: {_trunc_directive(e.directive)}"
                        lines.append(line)
                    await self.send(room_id, "\n".join(lines))
            else:
                await self.send(room_id,
                    "Usage: `/umbral start <interval>` | `/umbral stop` | `/umbral status`")
            return

        if body.startswith("/steer"):
            parts = body.split(None, 1)
            note = parts[1].strip() if len(parts) > 1 else ""
            if not note:
                await self.send_notice(room_id, "Usage: /steer <message>")
                return
            if room_id not in getattr(self, '_active_turns', set()):
                await self.send_notice(room_id, "🧭 No active turn to steer")
                return
            if not hasattr(self, '_steering_inbox'):
                self._steering_inbox = {}
            self._steering_inbox.setdefault(room_id, []).append(note)
            await self.send_notice(room_id, "🧭 Steering note queued")
            return

        # If room is halted via /stop, drop regular messages but allow slash
        # commands through (they already returned above).  This prevents a
        # deadlock where /resume is blocked by the very halt it needs to clear.
        if room_id in self._halted_rooms:
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
