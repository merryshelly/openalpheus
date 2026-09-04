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
import json
import logging
import shlex
import tempfile
import time
from html import escape as html_escape
from pathlib import Path
from zoneinfo import ZoneInfo

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

from openalph.agent import ContextOverflowError as AgentOverflowError
from openalph.handoff import (
    HANDOFF_EVENT,
    ACTIVE_PROJECT_EVENT,
    apply_boundary_and_rebuild,
    project_echo_text,
    read_active_project,
)
from openalph.provider import ProviderError, ProviderUnavailableError, resolve_model_checked
from openalph.config import MatrixConfig
from openalph.session import SessionLog, persist_assistant_turn
from openalph.mention import mentions_me, is_gated, strip_mention
from openalph.heartbeat import HeartbeatManager, parse_interval, format_interval
from openalph.umbral import UmbralManager
from openalph.tools import escape_system_reminder_tags, truncate_result
from openalph.callbacks import build_callbacks, build_context_status, MatrixSinks
import openalph.schedule as schedule

# Constants for media handling
MAX_MEDIA_BYTES = 20_000_000  # 20 MB
MEDIA_DIR = "media"

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
    # `source` (and its content/m.relates_to members) are untrusted input —
    # malformed or synthetic events (test doubles carry e.g.
    # m.relates_to=None) must degrade to "not an edit" rather than raising
    # through the whole dispatch path.
    content = source.get("content")
    if not isinstance(content, dict):
        content = {}
    relates_to = content.get("m.relates_to")
    if not isinstance(relates_to, dict):
        relates_to = {}
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


# Per-notice furled-detail caps. The furled block exists to show MORE than the
# abbreviated one-line summary, so these are head+tail truncated (via
# truncate_result) rather than hard-cut — but both the raw budget and the
# post-escape ceiling are bounded so a single notice stays comfortably under
# the Matrix PDU limit (~25K chars) even when html_escape expands the content
# (worst case ~6x for all-`"`/`<`/`&` text; realistic tool output ~1.05x).
_FURL_INPUT_RAW_CAP = 3000        # pretty-printed input_data, pre-escape
_FURL_RESULT_RAW_CAP = 12000      # full result content, pre-escape
_FURL_INPUT_ESC_CAP = 4000        # post-escape ceiling for the input fold
_FURL_RESULT_ESC_CAP = 16000      # post-escape ceiling for the result fold

# kdsn.247.2: the furl reader is the OPERATOR, who cannot re-run anything —
# the default agent-facing re-run steering marker would be misaddressed here.
_FURL_TRUNCATION_MARKER_TEMPLATE = (
    "[{n} chars elided from this notice — full content in session JSONL]"
)


def _unwrap_tool_result(text: str) -> str:
    """Strip the ``<tool_result tool="..." id="...">\\n … \\n</tool_result>``
    envelope that agent.py's wrap_tool_result adds before handing content to
    on_tool_call. The envelope's provenance attributes (tool name + call id)
    are noise in an operator-facing notice — the same reasoning the todo_write
    branch used to render from structured input rather than the wrapped string.
    Non-envelope text (e.g. an early guard-refusal string) is returned as-is."""
    s = str(text)
    open_tag = "<tool_result "
    if s.startswith(open_tag) and s.rstrip().endswith("</tool_result>"):
        gt = s.find(">")
        if gt != -1:
            inner = s[gt + 1:s.rstrip().rfind("</tool_result>")]
            return inner.strip("\n")
    return s


def _escape_capped(text: str, raw_cap: int, esc_cap: int) -> str:
    """Head+tail truncate `text` to `raw_cap`, html-escape it, then bound the
    escaped result to `esc_cap` (entity-safe: never cuts inside an `&…;`).
    Both bounds matter — raw_cap keeps the common case readable, esc_cap keeps
    the PDU size bounded regardless of how far escaping expands the content."""
    esc = html_escape(truncate_result(
        str(text), raw_cap,
        marker_template=_FURL_TRUNCATION_MARKER_TEMPLATE))
    if len(esc) <= esc_cap:
        return esc
    cut = esc[:esc_cap]
    # Avoid slicing through an HTML entity (`&amp;` etc.): if an unterminated
    # `&` opened before the cut, drop back to just before it.
    amp = cut.rfind("&")
    if amp != -1 and ";" not in cut[amp:]:
        cut = cut[:amp]
    return cut + "\n[truncated]"


# --- Context handoff command helpers (workspace-kdsn.305.1) ----------------
#
# Shared by the /cache status line, the /cache handoff subcommand, the
# apply_handoff_boundary tool callback, and the /project operator command so
# every surface reads state identically. All readers are pure JSONL scans;
# writers go through handoff.apply_boundary_and_rebuild (the ONLY boundary
# writer).

def _handoff_boundary_state(entries):
    """(index, trigger) of the LATEST handoff boundary marker in raw JSONL
    entries; (None, None) when there is none.

    The trigger is read from the handoff_boundary manifest (detail JSON).
    Hard epoch: legacy gc_boundary/toolstrip markers are NOT handoff
    boundaries and are not counted here.
    """
    best = -1
    trigger = None
    for entry in entries:
        if entry.get("role") != "system" or entry.get("event") != HANDOFF_EVENT:
            continue
        try:
            idx = int(entry.get("entry_index", -1))
        except (TypeError, ValueError):
            idx = -1
        if idx > best:
            best = idx
            trigger = None
            detail = entry.get("detail")
            if isinstance(detail, str):
                try:
                    manifest = json.loads(detail)
                    if isinstance(manifest, dict) and isinstance(manifest.get("trigger"), str):
                        trigger = manifest["trigger"]
                except json.JSONDecodeError:
                    trigger = None
            if trigger is None:
                trigger = "unknown"
    if best < 0:
        return None, None
    return best, trigger






def _handoff_confirm_text(outcome, trigger):
    """Operator-facing confirmation for an APPLIED boundary."""
    manifest = outcome.get("manifest") or {}
    durable = manifest.get("durable") or {}
    used = durable.get("used_tokens", 0)
    budget = durable.get("budget_tokens", 0)
    files = durable.get("files") or []
    over = (" — over reinjection budget (informational)"
            if outcome.get("over_budget") else "")
    return (
        f"✅ Handoff boundary {manifest.get('boundary_index', '?')} applied "
        f"(trigger: {trigger}). "
        f"Full strip: all pre-boundary content dropped; "
        f"{len(files)} durable file(s) re-injected. "
        f"Tokens {manifest.get('tokens_before', 0)} → "
        f"{manifest.get('tokens_after_est', 0)} (est.). "
        f"Durable budget {used}/{budget} tokens{over}."
    )


def _furl_tool_call_detail(input_data, result, is_error: bool) -> str:
    """Build the collapsed <details> disclosure appended to a generic tool-call
    notice: the FULL structured input (all params, pretty-printed) and the FULL
    result content, each furled/closed by default — the same click-to-expand UX
    the subagent/advisor/todo_write notices already use.

    Security invariants carried from those precedents:
      * html_escape (never raw mistune/HTML) on all tool-influenced text —
        tool inputs and outputs are adversarial-input surface (R8).
      * `result` is already post-redaction here: execute_tool applies
        redact_credentials/redact_known_secrets before agent.py hands the
        wrapped content to on_tool_call, so the furled result cannot surface a
        raw credential the abbreviated line would have hidden.
    """
    parts = []
    if input_data is not None:
        try:
            dumped = json.dumps(input_data, indent=2, ensure_ascii=False,
                                sort_keys=False, default=str)
        except (TypeError, ValueError):
            dumped = repr(input_data)
        input_esc = _escape_capped(dumped, _FURL_INPUT_RAW_CAP, _FURL_INPUT_ESC_CAP)
        parts.append(
            "<details><summary>📥 Full call</summary>\n"
            f"<pre>{input_esc}</pre></details>"
        )
    result_str = _unwrap_tool_result(result) if result else ""
    if result_str:
        result_esc = _escape_capped(result_str, _FURL_RESULT_RAW_CAP, _FURL_RESULT_ESC_CAP)
        summary = "⚠️ Error output" if is_error else "📤 Full result"
        parts.append(
            f"<details><summary>{summary}</summary>\n"
            f"<pre>{result_esc}</pre></details>"
        )
    return "".join(f"\n{p}" for p in parts)


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
            self.session_log = SessionLog(
                workspace, config.user_id,
                # Audit: hydration/status/CLI renders must honor the
                # GC flag — build_context without the kwarg now follows
                # this default instead of silently resurrecting expunged
                # content after a restart.
                handoff_default=getattr(config, "context", None) is not None
                and config.context.handoff_enabled,
            )
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
        self._room_effort = {}
        self._room_cache_ttl = {}   # Room-scoped cache TTL overrides (e.g. "5m"; default is "1h")
        self._room_timesense = {}   # Room-scoped timesense toggle (prepend timestamp to user messages)
        self._background_tasks: set[asyncio.Task] = set()
        self._session_locks: dict[str, asyncio.Lock] = {}
        # R3-1: per-room ACTIVATION mutex. See `_activate_room_once` for the
        # lock-ordering contract (activation lock is leaf-level and is always
        # released before `_session_locks[room_id]` is acquired).
        self._activate_locks: dict[str, asyncio.Lock] = {}
        self._steering_inbox: dict[str, list[str]] = {}
        self._active_turns: set[str] = set()
        self._advisor_results: dict[tuple, dict] = {}  # R5: keyed (room_id, call_id)
        self._subagent_results: dict[tuple, dict] = {}  # keyed (room_id, call_id)

    async def _emit_provider_notice(self, room_id: str, error) -> None:
        """Send the ⚠️ Provider error notice with the kdsn.292 warn-once latch.

        ProviderUnavailableError (degraded-start / skipped provider): keyed by
        (room_id, provider_key) — the FIRST occurrence per pair posts the
        notice (naming provider + startup skip reason); repeats are log-only,
        so a degraded default hit by a 5m heartbeat or umbral turn doesn't
        spam the room forever. Plain ProviderError (transient API failure):
        always posted — those are per-incident, not per-startup-state.
        """
        if isinstance(error, ProviderUnavailableError):
            if not hasattr(self, "_unavailable_noticed"):
                self._unavailable_noticed = set()
            key = (room_id, error.provider_key or "")
            if key in self._unavailable_noticed:
                logger.info(
                    "Provider unavailable notice suppressed (already sent) in %s: %s",
                    room_id, error,
                )
                return
            self._unavailable_noticed.add(key)
            await self.send(room_id, f"⚠️ **Provider error:** {error}")
            return
        await self.send(room_id, f"⚠️ **Provider error:** {error}")

    async def _broadcast_degraded_start(self) -> None:
        """One m.notice per joined room when the process came up degraded
        (kdsn.292 §3c). Once per process, right after initial sync; FAIL-SOFT —
        a per-room send failure warns and continues, never crashes startup.
        """
        skipped = getattr(self.agent.config, "skipped_providers", None)
        if not skipped:
            return
        from openalph.notify import degraded_summary
        body = degraded_summary(self.agent.config)
        # Review M2 (accepted tradeoff, comment-only): these rooms are
        # awaited INLINE inside sync_forever, so a down-Matrix startup delays
        # the send attempts serially (each send_notice retries/blocks per
        # room). Accepted — the rooms list only exists after the initial sync
        # succeeds anyway, and in the failure mode that matters (Matrix down)
        # this loop runs at most once with the per-room try/except above
        # bounding any individual failure.
        for room_id in list(getattr(self.client, "rooms", {}) or {}):
            try:
                await self.send_notice(room_id, f"🚨 **Agent started degraded**\n{body}")
            except Exception:
                logger.warning(
                    "degraded-start broadcast failed for %s", room_id, exc_info=True,
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

    # Upper bound on the best-effort stall notice (F4). self.send makes three
    # Matrix attempts, each able to wait out the client's network timeout, so an
    # unbounded notice task could outlive the turn it describes by many minutes.
    _STALL_NOTICE_TIMEOUT = 30

    async def _send_stall_notice(self, room_id: str, stall_minutes: float) -> None:
        """Best-effort, separately-bounded stall notice (F4).

        Runs OUTSIDE the turn's locked section as a background task so a Matrix
        outage cannot delay releasing `_session_locks[room_id]` — the whole point
        of the watchdog's cancel is to make the room usable again immediately.
        Never raises: the turn has already been cancelled and re-raised, and a
        failed courtesy notice must not surface as a background-task error.
        """
        try:
            await asyncio.wait_for(
                self.send(
                    room_id,
                    f"⚠️ **Turn stalled** — no progress for "
                    f"{stall_minutes:.0f} min; cancelled to unwedge this "
                    f"room. Please re-send your message.",
                ),
                timeout=self._STALL_NOTICE_TIMEOUT,
            )
        except asyncio.CancelledError:
            # Shutdown or an explicit cancel of this helper: propagate, never
            # swallow (async invariant).
            raise
        except asyncio.TimeoutError:
            logger.warning(
                "Stall notice to %s timed out after %ss — room was already "
                "unwedged", room_id, self._STALL_NOTICE_TIMEOUT,
            )
        except Exception:
            logger.exception("Failed to send stall notice to %s", room_id)

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
        """Single serializer for assistant turns. Delegates to the shared
        module-level function in session.py (kdsn.237 Phase 1).
        """
        persist_assistant_turn(
            self.agent,
            getattr(self, "session_log", None),
            room_id,
            content=content,
            tool_calls=tool_calls,
        )

    # --- Durable Matrix-event membership (round-2 F2b) -----------------------
    #
    # `_known_event_ids[room_id]` is the set of Matrix event_ids this room has
    # already durably processed. It is hydrated ONCE per activation from the room
    # JSONL (`_activate_room`) and maintained incrementally at every append site
    # that carries an event_id, so the per-message redelivery check stays O(1).
    #
    # It replaces the tail-equality guard, which compared only the newest
    # non-null event_id and therefore missed both batched wakes (two events
    # back-filled, only the newer one matching) and any redelivery with an
    # intervening event. It also replaces the per-message
    # `SessionLog.last_event_id()` call, a whole-file JSON scan run while the
    # event loop and both room locks were held.
    #
    # Deliberately NOT persisted separately: the JSONL is already the durable
    # record, and the set is rebuilt from it on every activation (including after
    # an umbral archive+wipe, which discards the room from `_active_rooms` and so
    # forces re-activation against the now-empty log — stale ids cannot suppress
    # post-reset messages).

    def _note_event_id(self, room_id: str, event_id: str | None) -> None:
        """Record a Matrix event_id as durably processed for `room_id`.

        No-op for falsy ids: synthetic turns (heartbeat/umbral/steer) and
        assistant entries carry `event_id=None`, and a None must never become a
        membership key or every id-less message would look like a redelivery of
        every other.
        """
        if not event_id:
            return
        if not hasattr(self, '_known_event_ids'):
            self._known_event_ids = {}
        self._known_event_ids.setdefault(room_id, set()).add(event_id)

    @staticmethod
    def _event_ts(event) -> int | float | None:
        """`server_timestamp` of a Matrix event as a real number, else None.

        Guards the gap-fill boundary comparison: unit-test doubles (and, in
        principle, a malformed server event) can leave `server_timestamp` as a
        MagicMock or a string, and `int > MagicMock` raises TypeError rather
        than comparing. `bool` is excluded because it is an `int` subclass and a
        `True` timestamp is meaningless. None means "unknown age", which the
        boundary treats as old history — the conservative direction, since a
        back-filled message is still deduped by event_id.
        """
        ts = getattr(event, 'server_timestamp', None)
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            return None
        return ts

    def _is_known_event_id(self, room_id: str, event_id: str | None) -> bool:
        """True when `event_id` was already durably processed for this room.

        False for falsy ids (see `_note_event_id`) and for rooms with no
        hydrated set, so an un-activated/legacy path degrades to the pre-fix
        behaviour rather than dropping messages.
        """
        if not event_id:
            return False
        return event_id in getattr(self, '_known_event_ids', {}).get(room_id, ())

    # --- Serialized lazy wake (round-3 R3-1) --------------------------------
    #
    # LOCK ORDER, and why activation needs its own lock at all.
    #
    # `_activate_room` is network-bound: it awaits `client.room_messages` for
    # gap-fill, and only adds the room to `_active_rooms` at the very END. The
    # `room_id not in self._active_rooms` test therefore stays true across those
    # awaits, so two live events arriving on a DORMANT room (batched wake, sync
    # reconnect replay, a user sending twice in a row) both saw it as inactive
    # and both activated. That double activation:
    #
    #   * ran gap-fill twice, appending the same back-filled history to the
    #     JSONL twice and writing two `session_resume` markers;
    #   * assigned a FRESH `_known_event_ids[room_id]` set each time, so the
    #     second assignment clobbered the first and genuine redeliveries walked
    #     straight through the F2b gate;
    #   * let one caller's gap-fill back-fill the OTHER caller's live event, so
    #     that event's own dispatch then looked like a redelivery and its turn
    #     was silently skipped.
    #
    # The fix is a per-room activation mutex plus a double-checked inactivity
    # test: acquire, RE-CHECK `_active_rooms` (the previous holder may have just
    # finished activating), and only then activate.
    #
    # ORDERING CONTRACT: `_activate_locks[room_id]` is LEAF-LEVEL. It is
    # acquired and fully released inside this helper, and is never held across
    # the acquisition of `_session_locks[room_id]` (or of the Agent's per-room
    # lock, taken further in, inside `handle_input`). Every caller therefore
    # takes activation first and session second, never the reverse, and no
    # cycle can form. `async with` guarantees release on the exception path too,
    # so a failed activation cannot wedge the room's next message.

    async def _activate_room_once(self, room_id: str, **kwargs) -> None:
        """Activate `room_id` at most once, even under concurrent live events.

        No-op when the room is already active. `kwargs` are forwarded verbatim
        to `_activate_room` (room_name / trigger_event_id / trigger_ts) and are
        only used by the caller that actually performs the activation — which is
        correct: the winner's trigger boundary is the one that matches the
        gap-fill it runs, and any racing event is dispatched separately anyway.
        """
        if not hasattr(self, '_active_rooms'):
            self._active_rooms = set()
        # Fast path: no lock, no await — the overwhelmingly common case is an
        # already-active room, and taking a lock there would serialise every
        # message in the room behind a mutex it does not need.
        if room_id in self._active_rooms:
            return

        if not hasattr(self, '_activate_locks'):
            self._activate_locks = {}
        lock = self._activate_locks.get(room_id)
        if lock is None:
            # Created without awaiting, so two callers in the same event-loop
            # step cannot end up with different Lock objects for one room.
            lock = self._activate_locks[room_id] = asyncio.Lock()

        async with lock:
            # Double check: a racing caller may have completed activation while
            # we waited here. Re-testing membership (rather than trusting the
            # pre-lock read) is what makes this mutual exclusion, not just
            # serialisation.
            if room_id in self._active_rooms:
                return
            await self._activate_room(room_id, **kwargs)

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
        ``_tool_start_times`` dict so that elapsed-time tracking works
        across the intent→notice lifecycle.

        This factory exists to eliminate the duplicated closure definitions
        that previously lived in both the heartbeat and streaming message
        code paths.
        """
        _tool_start_times: dict[str, float] = {}

        def _format_elapsed(start_ts) -> str:
            """Shared elapsed suffix for dispatch→completion notices
            (subagent branch + shell path, kdsn.319):
            `` — {m}m{ss:02d}s`` when ≥60s, else `` — {s:.1f}s``.
            Empty string when no start timestamp was recorded (restart
            mid-loop, or a non-dispatched call) — no suffix, no crash.
            """
            if start_ts is None:
                return ""
            elapsed = time.monotonic() - start_ts
            if elapsed >= 60:
                mins, secs = divmod(int(elapsed), 60)
                return f" — {mins}m{secs:02d}s"
            return f" — {elapsed:.1f}s"

        # kdsn.319: snapshot of the agent's live secret VALUES at factory
        # time, for the value-based room-egress pass (parity with the
        # tool-output seam's L1 redact_known_secrets layer).
        try:
            from openalph.tools import _collect_known_secrets as _cks
            _known_secrets = _cks(getattr(self.agent, "config", None))
        except Exception:
            _known_secrets = set()

        def _redact_for_room(text) -> str:
            """kdsn.319: redact a tool-input string before it enters a
            room-visible notice. Three layers, parity with the tool-output
            redaction seam:
              1. canonical shape pass (redact_credentials),
              2. value pass over the agent's own live secrets
                 (redact_known_secrets, factory-time snapshot),
              3. egress-context pass masking credential slots the shape
                 pass cannot recognise: case-insensitive Bearer, --token
                 values, -u/--user and URL userinfo, secret-named env
                 assignments (value masked, name kept).
            Over-redaction at the room-egress seam is safe (a [REDACTED:…]
            marker only hides context); a raw secret in the room is not.
            """
            import re
            from openalph.tools.security import redact_credentials, redact_known_secrets
            # kdsn.319 re-audit R1: hard input cap BEFORE any regex runs — the
            # userinfo scan is superlinear on adversarial word-runs, and this
            # runs on every tool-input string (e.g. file_write content). Every
            # room sink (preview 120 / fold 4000 / furled dump) renders at most
            # a few KB, so redacting beyond this window loses nothing visible.
            s = str(text)
            if len(s) > 65536:
                s = s[:65536]
            redacted, _ = redact_credentials(s)
            if _known_secrets:
                redacted, _ = redact_known_secrets(redacted, _known_secrets)
            redacted = re.sub(r"(?i)\bbearer\s+\S+", "[REDACTED:bearer_token]", redacted)
            redacted = re.sub(r"(?i)\bbasic\s+[A-Za-z0-9+/=]{8,}", "[REDACTED:basic_auth]", redacted)
            redacted = re.sub(r"--token(?:\s*=\s*|\s+)\S+", "[REDACTED:token]", redacted)
            redacted = re.sub(r"(?:(?<=-u )|(?<=--user ))\S+:\S+", "[REDACTED:userinfo]", redacted)
            # Glued / equals forms: -uadmin:pass, --user=admin:pass
            redacted = re.sub(r"(?<![\w-])-[uU]\S*:\S+", "[REDACTED:userinfo]", redacted)
            redacted = re.sub(r"--user(?:\s*=\s*|\s+)\S+:\S+", "[REDACTED:userinfo]", redacted)
            # URL userinfo. Scheme quantifier bounded (re-audit R1: unbounded
            # \w+ scanning is quadratic on word-runs); 32 covers realistic
            # word-only schemes.
            redacted = re.sub(r"(\w{1,32}://)[^/\s:@]+:[^@\s]+@", r"\1[REDACTED:userinfo]@", redacted)
            # Env assignment: secret word anywhere in the name; value masked
            # (quoted values included), name kept for debuggability.
            redacted = re.sub(
                r"(?i)\b((?:export\s+)?[A-Za-z0-9_]*"
                r"(?:SECRET|TOKEN|PASSWD|PASSWORD|PASS|KEY|CRED(?:ENTIAL)?S?)"
                r"[A-Za-z0-9_]*)\s*=\s*(\"[^\"]*\"|\S+)",
                r"\1=[REDACTED:env_secret]", redacted)
            return redacted

        def _redact_input_values(data):
            """kdsn.319: run _redact_for_room over every string value in a
            tool input and return a copy (the furled 'Full call' dump
            renders input_data verbatim); non-string values pass through
            unchanged.
            """
            if isinstance(data, str):
                return _redact_for_room(data)
            if isinstance(data, dict):
                return {k: _redact_input_values(v) for k, v in data.items()}
            if isinstance(data, list):
                return [_redact_input_values(v) for v in data]
            return data

        async def _tool_notice(call_id, name, input_data, result, is_error):
            # Show tool name + brief input context, but NEVER output
            # (which may contain secrets from op read, API responses, etc.)
            status = "❌ error" if is_error else "✅"
            detail = ""
            if isinstance(input_data, dict):
                # Pick the most informative input field per tool type.
                # "pattern" leads: grep/glob's sole REQUIRED param is "pattern"
                # ("path" is optional and usually omitted), so without it those
                # notices rendered as a context-free "🔧 grep ✅". "task" trails
                # for defensive completeness (subagent is special-cased below
                # and never reaches this generic path, but a future task-bearing
                # tool would otherwise render blank).
                for key in ("pattern", "path", "file_path", "command",
                            "query", "url", "task"):
                    if key in input_data:
                        # kdsn.319: redact BEFORE the 120-char cap — a secret
                        # past position 120 is still a secret, and capping
                        # first can split a [REDACTED:…] marker. Applies to
                        # every tool's detail field, not just shell.
                        val = _redact_for_room(str(input_data[key]))[:120]
                        detail = f" `{val}`"
                        break
                # grep's optional glob filter is meaningful context for the
                # abbreviated line — append it after the pattern.
                if name == "grep" and input_data.get("glob"):
                    detail += (f" (glob: `{_redact_for_room(str(input_data['glob']))[:60]}`)")
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
                # (kdsn.319: shared _format_elapsed helper — the shell
                # completion path renders the same format).
                elapsed_str = _format_elapsed(_tool_start_times.pop(call_id, None))
                summary_line = f"🤖 subagent ({model_info}) {status}{elapsed_str}"
                html = f'<b>{summary_line}</b>'
                # R8: html.escape (not raw mistune.html) — task/result previews
                # are tool-influenced text and must never pass raw HTML to the
                # client. Cosmetic markdown loss here is an accepted tradeoff
                # (audit decision).
                # kdsn.257: _escape_preserve_breaks (html_escape + nl->br) —
                # parity with the advisor notice fix (kdsn.198.9, dc9b550).
                # Plain html_escape left multi-line briefs/results as a
                # run-on line (HTML folds literal '\n' to a space).
                if task_preview:
                    html += (
                        f'\n<details><summary>📋 Task brief</summary>\n'
                        f'{_escape_preserve_breaks(task_preview)}</details>'
                    )
                if result_preview:
                    html += (
                        f'\n<details><summary>📨 Result</summary>\n'
                        f'{_escape_preserve_breaks(result_preview)}</details>'
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
                # Generic tool-call notice. The plain-text `body` stays the
                # short abbreviated line (default/unfurled view); the
                # formatted_body adds the FULL call+result behind collapsed
                # <details> folds — matching the subagent/advisor/todo_write
                # click-to-expand pattern, and now applied on SUCCESS too (not
                # just errors). _furl_tool_call_detail carries the R8 escaping
                # and post-redaction guarantees (see its docstring).
                # kdsn.319: shell completions carry the dispatch→completion
                # elapsed (same format as subagent); only shell and subagent
                # record start times, so other tools never get the suffix.
                elapsed_str = ""
                if name == "shell":
                    elapsed_str = _format_elapsed(_tool_start_times.pop(call_id, None))
                notice_body = f"🔧 {name}{detail} {status}{elapsed_str}"
                # kdsn.319: the furled 'Full call' dump renders input_data
                # verbatim — redact every string value first (same seam).
                detail_html = _furl_tool_call_detail(_redact_input_values(input_data), result, is_error)
                formatted_body = html_escape(notice_body) + detail_html
                content_msg = {
                    "msgtype": "m.notice",
                    "body": notice_body,
                    "format": "org.matrix.custom.html",
                    "formatted_body": formatted_body,
                }
                try:
                    await self._room_send_with_retry(room_id, content_msg)
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
                    _tool_start_times[tc.id] = time.monotonic()
                    task_preview = tc.input.get("task", "")
                    model_info = tc.input.get("model", "default")
                    iters = tc.input.get("max_iterations", 200)
                    summary = f"⚙️ Spawning sub-agent ({model_info}, max {iters} iters)"
                    html = f'<b>{summary}</b>'
                    # R8: html.escape (not raw mistune.html) — the dispatch task
                    # brief is tool-influenced text; never pass raw HTML through.
                    # kdsn.257: _escape_preserve_breaks — parity with the
                    # advisor spawn-notice fix (kdsn.198.9).
                    if task_preview:
                        html += (
                            f'\n<details><summary>📋 Task brief</summary>\n'
                            f'{_escape_preserve_breaks(task_preview)}</details>'
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
                elif tc.name == "shell" and isinstance(tc.input, dict):
                    # kdsn.319: dispatch-time notice for shell — a long shell
                    # (sleep-poll, long build) otherwise produces zero
                    # in-room signal until it lands; the typing indicator
                    # expires after ~30s and the room reads as dead.
                    _tool_start_times[tc.id] = time.monotonic()
                    # Redact the FULL command before flattening/truncating:
                    # capping first can split a [REDACTED:…] marker, and a
                    # secret past position 120 is still a secret.
                    redacted_cmd = _redact_for_room(str(tc.input.get("command") or ""))
                    flat = redacted_cmd.replace("\n", " ").replace("\r", " ")
                    # Head 120 chars — the preview identifies the call
                    # (house convention: every other preview uses [:120]).
                    summary = f"🔧 shell ▶ — {flat[:120]}" if flat else "🔧 shell ▶"
                    formatted = f'<b>{html_escape(summary)}</b>'
                    if redacted_cmd:
                        # kdsn.257 parity: _escape_preserve_breaks (html_escape
                        # + nl->br) — the full command is tool-influenced text
                        # and must never pass raw HTML to the client.
                        # kdsn.319 audit L1: cap the fold — an uncapped huge
                        # command blows the Matrix PDU limit, the send fails
                        # all 3 retries, and the dispatch notice is lost
                        # entirely (the exact signal this feature exists for).
                        fold = redacted_cmd
                        if len(fold) > 4000:
                            fold = fold[:4000] + "\n[truncated]"
                        formatted += (f'\n<details><summary>🔧 full command</summary>\n'
                                      f'{_escape_preserve_breaks(fold)}</details>')
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
        # resolve room_name from nio for the context_status callback
        room = self.client.rooms.get(room_id) if getattr(self, 'client', None) else None
        room_name = room.named_room_name() if room else None
        sinks = MatrixSinks(self, room_id)
        # Lazy-init for tests that construct MatrixBot via __new__
        if not hasattr(self, '_advisor_results'):
            self._advisor_results = {}
        if not hasattr(self, '_subagent_results'):
            self._subagent_results = {}
        # kdsn.305.1: the boundary tool callbacks (apply_handoff_boundary,
        # set_active_project) are built at THE construction seam —
        # callbacks.build_callbacks — so every transport (Matrix live,
        # heartbeat/umbral, headless CLI) carries identical wiring. This
        # wrapper only adds Matrix-specific keys below.
        callbacks = build_callbacks(
            self.agent,
            room_id,
            sinks,
            turn_source=turn_source,
            session_log=getattr(self, 'session_log', None),
            heartbeat=getattr(self, 'heartbeat', None),
            umbral=getattr(self, 'umbral', None),
            advisor_results=self._advisor_results,
            subagent_results=self._subagent_results,
            room_name=room_name,
        )

        return callbacks

    def _make_steering_drain(self, room_id: str, turn_progress=None):
        """Build the per-turn steering drain closure (kdsn.311).

        ONE construction seam shared by _process_message and
        _run_heartbeat_turn so the live and synthetic turn paths cannot
        drift (duplicated callback construction was the root cause of ~5
        audit findings per the tool-management skill). The closure pops the
        room's steering inbox atomically (list.pop is GIL-safe), logs each
        note to JSONL as role=user/source=steer (build_context re-frames
        source==steer entries on replay, so live and rebuild match), emits
        a delivered notice, and returns the note strings for the agent loop
        to inject at the next tool-loop boundary.

        ``turn_progress`` is the stall-watchdog liveness hook; the
        heartbeat/umbral path runs inside the timer-loop task with no
        watchdog and passes None.
        """
        if not hasattr(self, '_steering_inbox'):
            self._steering_inbox = {}

        async def _drain_steering() -> list:
            notes = self._steering_inbox.pop(room_id, [])
            if notes and turn_progress is not None:
                turn_progress()
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

        return _drain_steering

    async def _run_heartbeat_turn(self, room_id: str, content: str, *, turn_source: str | None = None) -> None:
        """Execute a heartbeat/umbral turn: activate room, process input, deliver response.

        Extracted from _inject_heartbeat for reuse by _inject_umbral.
        Raises on error — caller is responsible for error handling.
        """
        # Activate room if not already active.
        # R3-1: shares the per-room activation mutex with the live-message
        # paths, so a heartbeat firing as a live message wakes the room cannot
        # produce two concurrent gap-fills.
        if room_id not in self._active_rooms:
            await self._activate_room_once(room_id)

        # -- tool-use callbacks (same as normal message path) --
        _tool_notice, _tool_intent = self._make_tool_callbacks(room_id)

        # Process through agent
        # kdsn.311: pre-initialized so the finally below can never hit an
        # UnboundLocalError if the try body raises before the drain is armed
        # (e.g. a _set_typing network failure) — same guard pattern as the
        # live path's _drain_steering_fn. A raise before arming leaves the
        # active-turn mark untouched (never added) and skips the final drain.
        _hb_drain = None
        try:
            await self._set_typing(room_id, True)

            # Resolve effort level: room override > config
            # Room effort override maps to the API `thinking` level.
            _effort_override = getattr(self, '_room_effort', {}).get(room_id)
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
                    provider_cfg, _ = resolve_model_checked(model_str, self.agent.config.providers, aliases=self.agent.config.model_aliases, skipped_providers=getattr(self.agent.config, "skipped_providers", {}))
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

            # kdsn.311: steering works on heartbeat/umbral turns too. Arm the
            # SAME active-turn mark + drain seam as the live path so /steer
            # can redirect a long-running autonomous turn ("No active turn
            # to steer" was inaccurate — the heartbeat turn IS a turn). The
            # timer-loop task runs this turn; the drain closure rides the
            # callbacks dict via the shared _make_steering_drain seam.
            if not hasattr(self, '_active_turns'):
                self._active_turns = set()
            self._active_turns.add(room_id)
            _hb_drain = self._make_steering_drain(room_id)
            callbacks['drain_steering'] = _hb_drain
            # NOTE (audit qwen MEDIUM-2): heartbeat turns do not hold the
            # per-room session lock, so a fired heartbeat turn CAN overlap a
            # live operator turn in this room. Both install drains over the
            # SAME per-room inbox; whichever reaches a tool-loop boundary
            # first consumes a deposited note (first-drainer-wins). A note
            # consumed by the "wrong" turn is still JSONL-logged, noticed,
            # and framed into context on the next rebuild — degraded
            # delivery, never lost. Per-turn inbox scoping is bead-tracked
            # follow-up work.

            response = await self.agent.handle_input(
                content,
                room_id,
                on_tool_call=_tool_notice,
                on_tool_intent=_tool_intent,
                thinking=_effort_override,
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
                        thinking=_effort_override,
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
            # kdsn.311 race rule (same as _process_message): final drain
            # pass to catch notes that arrived during the last iteration,
            # THEN discard the active-turn mark.
            if _hb_drain is not None:
                try:
                    await _hb_drain()
                except Exception:
                    pass
            if hasattr(self, '_active_turns'):
                self._active_turns.discard(room_id)

    def _build_context_status(self, rid: str) -> dict:
        """Assemble context status data dict for the context_status tool.

        Collects agent status, session age, heartbeat state, umbral state,
        and room identity for the given room ID. Sync-safe: calls only sync
        methods on self.agent, self.session_log, self.heartbeat, self.umbral.
        """
        room = self.client.rooms.get(rid) if getattr(self, 'client', None) else None
        room_name = room.named_room_name() if room else None
        return build_context_status(
            self.agent, rid,
            room_name=room_name,
            session_log=getattr(self, 'session_log', None),
            heartbeat=getattr(self, 'heartbeat', None),
            umbral=getattr(self, 'umbral', None),
        )

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
            # kdsn.292: warn-once latch lives in _emit_provider_notice —
            # a degraded default must not spam the room every heartbeat.
            await self._emit_provider_notice(room_id, e)
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
            # kdsn.292: warn-once latch lives in _emit_provider_notice.
            await self._emit_provider_notice(room_id, e)
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

    async def _activate_room(self, room_id: str, room_name: str = "", *,
                             trigger_event_id: str | None = None,
                             trigger_ts: int | None = None):
        """Load session history on first message (lazy wake).

        New architecture: reads local JSONL instead of paginating Matrix history.
        If entries exist, does gap-fill from last known event. If new room, starts fresh.

        TRIGGER-AWARE GAP-FILL (round-2 F2a). Activation is always caused by some
        live event, and that event is dispatched to `_process_message` separately
        — which is the ONE writer that persists it, in its canonical form. So the
        triggering event, and anything NEWER than it (the live sync loop will
        dispatch those too), form an UPPER BOUNDARY: gap-fill back-fills strictly
        older messages and must not persist the boundary itself.

        Without this, gap-fill wrote its own copy of the trigger from the raw
        Matrix `body`, and `_process_message` then wrote a second — the 2026-08-03
        JSONL duplication. It also corrupted media triggers, whose raw `body` is a
        bare filename while the canonical record is
        `[media: media/<hash>/<file> (<mime>, <size>)]` — the only form that lets
        the model locate the downloaded file or vision expand the image.

        Args:
            trigger_event_id: event_id of the live event that caused activation.
                Excluded from back-fill persistence (its own dispatch persists it).
            trigger_ts: `server_timestamp` of that event. Anything at-or-after it
                is also excluded, covering batched wakes where the server already
                returns events newer than the one whose callback we are inside.
        """
        session_log = getattr(self, 'session_log', None)

        # Per-room known-event-ID membership set (round-2 F2b). Hydrated ONCE
        # here from the JSONL and then maintained incrementally by
        # `_note_event_id` at every append site, so the per-message redelivery
        # check in `_process_message` is O(1). The previous design called
        # `SessionLog.last_event_id()` on EVERY ungated message: a synchronous
        # open + whole-file JSON decode performed while the event loop and both
        # room locks were held, which could pause unrelated rooms on long
        # sessions — and, being only the tail, it missed batched wakes and
        # non-tail redeliveries entirely.
        if not hasattr(self, '_known_event_ids'):
            self._known_event_ids = {}
        _known_set: set = set()
        self._known_event_ids[room_id] = _known_set

        if session_log:
            existing = session_log.read(room_id)

            for _e in existing:
                _eid = _e.get("event_id") if isinstance(_e, dict) else None
                if _eid:
                    _known_set.add(_eid)

            if existing:
                # Gap-fill: fetch Matrix messages since the last known event
                last_id = session_log.last_event_id(room_id)
                if last_id:
                    try:
                        # Gap-fill: page backward through recent Matrix messages until
                        # we find overlap with known session history (or hit the cap).
                        GAP_FILL_MAX = 500  # safety cap to avoid infinite paging
                        known_ids = {e.get("event_id") for e in existing if e.get("event_id")}

                        # Normalise once: a non-numeric trigger_ts (test double,
                        # malformed event) disables the timestamp half of the
                        # boundary rather than raising inside the hot loop.
                        _trigger_ts = (
                            None if isinstance(trigger_ts, bool)
                            or not isinstance(trigger_ts, (int, float))
                            else trigger_ts
                        )

                        def _at_or_after_trigger(msg, ev_id) -> bool:
                            """True when `msg` IS the activation trigger, or is
                            strictly newer than it — i.e. an event the live sync
                            loop dispatches itself (F2a upper boundary).

                            Two independent tests:

                            * exact event_id — always applied, and the only test
                              that matters for the trigger itself. It is exact,
                              so it cannot be defeated by clock weirdness.
                            * strictly-greater timestamp — covers BATCHED wakes:
                              when E1 and E2 arrive together we may be inside
                              E1's callback while `room_messages` already returns
                              E2. E2 is newer, has its own pending dispatch, and
                              must not be back-filled here or it would be
                              persisted twice.

                            STRICT `>`, not `>=`. Equal timestamps are common and
                            do NOT imply "sibling of this wake": Matrix
                            `server_timestamp` has millisecond resolution, so
                            ordinary older history can share the trigger's
                            millisecond, and `>=` would silently drop real gap
                            messages — the exact failure mode this whole fix
                            exists to prevent. The residual case (a genuine
                            same-millisecond sibling) degrades benignly: it is
                            back-filled here, recorded via `_note_event_id`, and
                            its own live dispatch is then absorbed by the
                            redelivery gate in `_process_message`. Its content is
                            durably logged and hydrated; only its separate model
                            turn is skipped.
                            """
                            if trigger_event_id and ev_id == trigger_event_id:
                                return True
                            if _trigger_ts is None:
                                return False
                            _ts = self._event_ts(msg)
                            if _ts is None:
                                return False   # unknown age -> treat as old history
                            return _ts > _trigger_ts
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
                                # F2a: skip the activation trigger and anything
                                # newer. `continue`, never `break` — the chunk is
                                # newest-first, so older messages we DO want
                                # follow the boundary, and a `break` here would
                                # silently drop the real gap.
                                if _at_or_after_trigger(msg, ev_id):
                                    continue
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
                            # F2b: a back-filled event is durably processed —
                            # record it so a later live redelivery of the same
                            # event is recognised without re-scanning the file.
                            self._note_event_id(room_id, ev_id)
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
                history.extend(session_log.build_context(
                    room_id))
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

                # Restore per-room overrides (model, effort) from session log.
                # Scan all entries — last override wins (user may have switched multiple times).
                _restored_model = None
                _restored_effort = None
                _restored_cache_ttl = None
                _restored_timesense = None
                for entry in existing:
                    if entry.get("role") == "system":
                        ev = entry.get("event")
                        detail = entry.get("detail", "")
                        if ev == "model_override" and detail:
                            self.agent._room_models[room_id] = detail
                            _restored_model = detail
                        # Legacy "thinking_override" accepted read-only (pre-rename
                        # data migration); the write side only emits "effort_override".
                        elif ev in ("effort_override", "thinking_override") and detail:
                            if not hasattr(self, "_room_effort"):
                                self._room_effort = {}
                            self._room_effort[room_id] = detail
                            _restored_effort = detail
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
                if _restored_effort:
                    parts.append(f"Effort: `{_restored_effort}`")
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
                    _buffered_event_id = getattr(event, 'event_id', None)
                    session_log.append(
                        role="user",
                        sender=event.sender,
                        room=room_id,
                        event_id=_buffered_event_id,
                        content=body,
                        mentioned=mention.mentioned,
                    )
                    # F2b: the gated buffer is this event's durable claim.
                    self._note_event_id(room_id, _buffered_event_id)

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
                history.extend(self.session_log.build_context(
                    room_id))
                logger.info("Hydrated context for %s: %d entries", room_id, len(history))
                user_already_in_history = True
            # --- End mention gating ---

            # Lazy wake: activate room on first live message.
            #
            # F2a: the event we are processing IS the activation trigger, so it
            # is threaded into gap-fill as the upper boundary. Gap-fill then
            # back-fills strictly older history and leaves this event (and
            # anything newer, which the sync loop dispatches separately) alone —
            # the ungated append below is its single writer, which is what keeps
            # a media trigger's canonical `[media: …]` tag as the durable record
            # instead of the raw Matrix body.
            # R3-1: serialized through `_activate_room_once`, so two live events
            # on a dormant room cannot both run the network-bound activation.
            if room_id not in self._active_rooms:
                room_name = getattr(room, 'name', '') or getattr(room, 'display_name', '') or room_id
                await self._activate_room_once(
                    room_id, room_name=room_name,
                    trigger_event_id=getattr(event, 'event_id', None),
                    trigger_ts=self._event_ts(event),
                )
                # Activation hydrates history from JSONL (which already includes
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

            # --- REDELIVERY GATE (round-2 F2b) ------------------------------
            #
            # Post-lock, pre-agent. If this event_id was already durably
            # processed for this room, this delivery is a DUPLICATE: log it and
            # return without running a turn. Matrix redelivers on reconnect and
            # after a restart resumes from a stale sync token, and the previous
            # (round-1) guard only suppressed the duplicate APPEND — the model
            # still ran and the room still received a second reply.
            #
            # Membership is the in-memory `_known_event_ids[room_id]` set,
            # hydrated once at activation and updated at every append site, so
            # this is O(1). The removed round-1 guard called
            # `SessionLog.last_event_id()` here on EVERY ungated message: a
            # synchronous whole-file JSON scan run while the event loop and both
            # room locks were held, which also missed batched wakes (only the
            # tail id was compared) and any redelivery with an intervening event.
            #
            # SCOPED TO UNGATED ROOMS, deliberately. The invariant is "whichever
            # path APPENDS the trigger claims it". In a gated room the trigger is
            # always buffered BEFORE this point — by `_handle_room_message` (and
            # then hydrated into the set by activation) or by the gated buffer
            # above — so the event is legitimately "known" on its FIRST delivery
            # and gating here would suppress every gated turn. On the ungated
            # path `_process_message` is the sole writer and appends just below,
            # after this gate, so membership can only mean a genuine earlier
            # dispatch or durable history from a previous process lifetime.
            #
            # Gated rooms are NOT unprotected: round-3 R3-2 moved their
            # redelivery decision UPSTREAM, to `_handle_room_message` immediately
            # BEFORE the gating buffer append (and, for media, to the top of
            # `_handle_media_message`), which is the only place the check can sit
            # ahead of the claim. See the comment there.
            #
            # Falsy event_ids never match (see `_is_known_event_id`), so
            # synthetic turns — heartbeat, umbral, steering — are unaffected.
            #
            # Early return is lock-safe: the outer `finally` releases
            # `_session_locks[room_id]` and discards `_active_turns` on every
            # path, and neither typing nor the steering drain has been armed yet.
            _trigger_event_id = getattr(event, 'event_id', None)
            if not gated and self._is_known_event_id(room_id, _trigger_event_id):
                # No logging args: the message is pre-formatted so downstream
                # record inspection (`record.message`) is complete on its own.
                logger.info(
                    f"Duplicate delivery of event {_trigger_event_id} in "
                    f"{room_id} — already processed; skipping turn"
                )
                return

            # Append user message to session log (only for ungated rooms — gated already buffered above)
            if not gated:
                if session_log:
                    session_log.append(
                        role="user",
                        sender=event.sender,
                        room=room_id,
                        event_id=_trigger_event_id,
                        content=body,
                    )
                    # F2b: this append is the durable claim on the event; record
                    # it so a later redelivery hits the gate above.
                    self._note_event_id(room_id, _trigger_event_id)

            # Regular message: process through agent
            # Timesense: prepend timestamp to user message for LLM context
            if getattr(self, '_room_timesense', {}).get(room_id):
                from datetime import datetime, timezone
                _ts = datetime.now(timezone.utc).astimezone().strftime("%A, %B %d, %Y — %H:%M %Z")
                body = f"[{_ts}] {body}"
            await self._set_typing(room_id, True)

            # Wire tool visibility for this turn
            _raw_tool_notice, _raw_tool_intent = self._make_tool_callbacks(room_id)

            # --- Turn stall watchdog state (RCA 2026-08-03) ---------------------
            # _last_progress is a mutable cell poked by every room-observable
            # progress signal (text/thinking deltas, tool callbacks, steering
            # deliveries, and the explicit callbacks['turn_progress'] hook the
            # subagent tool pings). The watchdog cancels the turn when nothing
            # has poked it for turn_stall_timeout_seconds, so a provider retry
            # storm can no longer hold both per-room locks indefinitely.
            _turn_task = asyncio.current_task()
            _last_progress = [time.monotonic()]
            _stall_fired = [False]

            def _turn_progress(*_args, **_kwargs):
                """Reset the stall window. Plain (non-async) callable so tools can
                poke it without awaiting."""
                _last_progress[0] = time.monotonic()

            async def _tool_notice(*args, **kwargs):
                _turn_progress()
                return await _raw_tool_notice(*args, **kwargs)

            async def _tool_intent(*args, **kwargs):
                _turn_progress()
                return await _raw_tool_intent(*args, **kwargs)

            try:
                # Resolve effort level: room override > config
                _effort_override = getattr(self, '_room_effort', {}).get(room_id)
                _cache_ttl = getattr(self, '_room_cache_ttl', {}).get(room_id)

                # R1 refactor: use shared _build_agent_callbacks for identical wiring
                callbacks = self._build_agent_callbacks(room_id, None)

                # Set up streaming delivery
                streaming = StreamingDelivery(self, room_id)
                _thinking_buffer = []
                _thinking_done = False
                _thinking_notified = False

                async def _text_delta(text: str, done: bool):
                    _turn_progress()
                    await streaming.push(text, done=done)

                async def _thinking_delta(text: str, done: bool):
                    nonlocal _thinking_done, _thinking_notified
                    _turn_progress()
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
                        provider_cfg, _ = resolve_model_checked(model_str, self.agent.config.providers, aliases=self.agent.config.model_aliases, skipped_providers=getattr(self.agent.config, "skipped_providers", {}))
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
                # The closure is built by the ONE shared seam
                # (_make_steering_drain, kdsn.311) also used by the
                # heartbeat/umbral path, so the two turn paths cannot
                # drift. _active_turns is set here, BEFORE handle_input,
                # so that a concurrent /steer command can detect the
                # active turn.
                if not hasattr(self, '_active_turns'):
                    self._active_turns = set()
                self._active_turns.add(room_id)

                _drain_steering = self._make_steering_drain(room_id, _turn_progress)
                _drain_steering_fn = _drain_steering

                # Pass drain_steering via callbacks dict so that test mocks
                # with explicit handle_input signatures (no drain_steering kwarg)
                # are not broken. agent.handle_input extracts it from callbacks
                # when drain_steering= kwarg is None.
                callbacks['drain_steering'] = _drain_steering

                # Liveness hook for the tool layer (e.g. the subagent tool pings
                # this while blocked on a long sub run). Added here rather than in
                # _build_agent_callbacks so that builder's pinned key set — and the
                # heartbeat/CLI paths that share it — stay unchanged.
                callbacks['turn_progress'] = _turn_progress

                # Arm the stall watchdog. Read defensively: many tests build bots
                # with MagicMock agents, where a bare attribute read would yield a
                # MagicMock and make the arithmetic below nonsense.
                _stall_timeout = getattr(
                    getattr(self.agent, 'config', None), 'turn_stall_timeout_seconds', 0)
                if isinstance(_stall_timeout, bool) or not isinstance(_stall_timeout, (int, float)):
                    _stall_timeout = 0
                _watchdog = None

                async def _stall_watchdog(timeout: float):
                    _poll = min(30.0, timeout / 4) or 0.05
                    while True:
                        await asyncio.sleep(_poll)
                        _idle = time.monotonic() - _last_progress[0]
                        if _idle > timeout:
                            _stall_fired[0] = True
                            logger.warning(
                                "Turn stall watchdog firing in %s — no observable "
                                "progress for %.0fs (timeout %.0fs); cancelling turn",
                                room_id, _idle, timeout,
                            )
                            if _turn_task is not None:
                                _turn_task.cancel()
                            return

                if _stall_timeout > 0:
                    _watchdog = asyncio.create_task(
                        _stall_watchdog(float(_stall_timeout)),
                        name=f"turn-stall-watchdog:{room_id}",
                    )

                try:
                    response = await self.agent.handle_input(
                        body, room_id,
                        on_tool_call=_tool_notice,
                        on_tool_intent=_tool_intent,
                        on_text_delta=_text_delta,
                        on_thinking_delta=_thinking_delta,
                        thinking=_effort_override,
                        callbacks=callbacks,
                        on_cache_status=_cache_status,
                        cache_ttl=_cache_ttl,
                        append_user=not user_already_in_history,
                    )
                finally:
                    # Disarm immediately — a tight finally around handle_input (not
                    # the outer one) minimises the window in which a watchdog that
                    # fires just as the turn returns could land its CancelledError
                    # on a later await and lose the response.
                    #
                    # F1 (CRITICAL, round-2 review): collect the child with
                    # `gather(..., return_exceptions=True)`, NOT
                    # `contextlib.suppress(CancelledError) + await _watchdog`.
                    # `suppress` cannot distinguish the child watchdog's EXPECTED
                    # cancellation from a cancellation delivered to THIS turn task
                    # at that very await, so in the tight return/disarm window the
                    # latter was swallowed and the turn went on to persist and
                    # send `response` with `current_task().cancelling() == 1` —
                    # a watchdog / operator `/stop` / shutdown cancel could be
                    # acknowledged and then ignored.
                    #
                    # gather() consumes the child's CancelledError into its
                    # results list (return_exceptions=True) while cancellation
                    # aimed at the CURRENT task still propagates out of the
                    # gather await, so the invariant "CancelledError always
                    # re-raises" holds. Verified both ways before adopting.
                    if _watchdog is not None:
                        _watchdog.cancel()
                        await asyncio.gather(_watchdog, return_exceptions=True)
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
                        _sh = self.session_log.build_context(
                            room_id)
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
            except asyncio.CancelledError:
                # Distinguish the watchdog's cancel from an operator /stop or a
                # process-shutdown cancel: only the former sets _stall_fired, and
                # only the former explains itself in-room. ALWAYS re-raise so the
                # finally blocks below run (locks release, typing clears,
                # _active_turns is discarded) and /stop behaviour is unchanged.
                if _stall_fired[0]:
                    _stall_minutes = float(_stall_timeout) / 60
                    logger.warning(
                        "Turn cancelled by stall watchdog in %s after %.0f min "
                        "without observable progress",
                        room_id, _stall_minutes,
                    )
                    # F4 (round-2 review): capture the notice and hand it to a
                    # SEPARATELY-BOUNDED background task, then re-raise with ZERO
                    # awaits in between. Awaiting self.send here delayed the very
                    # unwedge this branch performs: send makes three Matrix
                    # attempts and each request can wait out the client's network
                    # timeout, so a Matrix outage kept _session_locks[room] (and
                    # the agent's room lock, released only as handle_input
                    # unwinds) held for minutes AFTER the provider stall had been
                    # successfully cancelled — and `/stop` could not reliably
                    # interrupt that phase because the Agent has already dropped
                    # its current-task entry. Scheduling instead means the outer
                    # `finally` blocks reach the lock release immediately and the
                    # room is usable while the notice is still in flight.
                    self._fire_background(self._send_stall_notice(
                        room_id, _stall_minutes))
                raise
            except AgentOverflowError as e:
                logger.warning("Context overflow in %s: %s", room_id, e)
                await self.send(room_id,
                    f"⚠️ **Context overflow** — ~{e.current_tokens:,} / "
                    f"{e.max_tokens:,} tokens. Start a new room to continue.")
            except ProviderError as e:
                code = f" ({e.status_code})" if e.status_code else ""
                logger.warning("Provider error%s in %s: %s", code, room_id, e)
                # kdsn.292: warn-once latch lives in _emit_provider_notice.
                await self._emit_provider_notice(room_id, e)
            except Exception:
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

        # --- MEDIA REDELIVERY GATE (round-3 R3-3) --------------------------
        #
        # Ahead of the download, deliberately. The redelivery gate in
        # `_process_message` sits far downstream of this handler, so a
        # redelivered media event used to re-fetch the whole file over HTTP and
        # rewrite it to disk before anything could recognise it as a duplicate —
        # wasted bandwidth and disk on an event whose turn was then skipped
        # anyway. Matrix redelivers on reconnect, so this is routine, not rare.
        #
        # Falsy event_ids never match (see `_is_known_event_id`), so an event
        # without an id keeps the pre-fix behaviour rather than being dropped.
        _media_event_id = getattr(event, 'event_id', None)
        if self._is_known_event_id(room.room_id, _media_event_id):
            logger.info(
                f"Duplicate delivery of media event {_media_event_id} in "
                f"{room.room_id} — already processed; skipping download"
            )
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
        if not isinstance(event_source, dict):
            event_source = {}
        _ev_content = event_source.get("content")
        if isinstance(_ev_content, dict) and _ev_content.get("openalph.thinking") is True:
            return

        room_id = room.room_id

        # For final edit events, use the replacement content (m.new_content)
        event_content = event_source.get("content", {})
        if not isinstance(event_content, dict):
            event_content = {}
        _relates = event_content.get("m.relates_to")
        if isinstance(_relates, dict) and _relates.get("rel_type") == "m.replace":
            _new_content = event_content.get("m.new_content")
            if not isinstance(_new_content, dict):
                _new_content = {}
            body = _new_content.get("body", event.body).strip()
        else:
            body = event.body.strip()

        # --- Mention gating ---
        gated = is_gated(self.config, room)

        if gated:
            event_source = getattr(event, 'source', {}) or {}
            mention = mentions_me(self.config.user_id, event_source, body)

            # --- GATED REDELIVERY GATE (round-3 R3-2) --------------------
            #
            # UPSTREAM of the buffer append, and this position is the whole
            # point. The post-lock gate in `_process_message` is necessarily
            # `not gated`: a gated room's trigger is buffered and CLAIMED right
            # below, before `_process_message` ever runs, so by the time that
            # gate is reached the event is legitimately "known" on its FIRST
            # delivery and checking there would suppress every gated turn. That
            # left gated rooms — the production 3+ member rooms — with no
            # redelivery protection at all: a sync reconnect or a restart
            # resuming from a stale sync token produced a second buffer entry, a
            # second model turn and a second assistant reply.
            #
            # Checking BEFORE the append restores the invariant "whichever path
            # appends the trigger claims it": membership here can only mean an
            # EARLIER delivery already appended it (this process) or durable
            # history from a previous process lifetime (hydrated at activation).
            # Skipping entirely — no buffer append, no dispatch, not even the
            # bare-command hint — makes the duplicate completely inert, so it
            # cannot reappear in hydrated context either.
            #
            # Falsy event_ids never match (see `_is_known_event_id`), so
            # id-less events keep the pre-fix behaviour.
            #
            # KNOWN LIMITATION (dormant gated room): the membership set is
            # hydrated from the JSONL by `_activate_room`, which for a gated
            # room runs AFTER this point. A redelivery that is the very FIRST
            # event to wake a dormant room therefore still slips through this
            # check. It remains a strict improvement — every subsequent
            # redelivery in the room's lifetime is caught — and closing it needs
            # activation to move ahead of the gated buffer, which would make
            # non-mentioned messages activate rooms (a behaviour change out of
            # scope here).
            _gated_event_id = getattr(event, 'event_id', None)
            if self._is_known_event_id(room_id, _gated_event_id):
                # Pre-formatted: no logging args, so downstream record
                # inspection (`record.message`) is complete on its own.
                logger.info(
                    f"Duplicate delivery of event {_gated_event_id} in gated "
                    f"room {room_id} — already processed; skipping"
                )
                return

            # Buffer ALL messages to session log (mentioned or not)
            if self.session_log:
                _buffered_event_id = getattr(event, 'event_id', None)
                self.session_log.append(
                    role="user",
                    sender=event.sender,
                    room=room_id,
                    event_id=_buffered_event_id,
                    content=body,
                    mentioned=mention.mentioned,
                )
                # F2b: this buffer is the event's durable claim. Recorded here
                # too so that once the room is active a redelivery is visible in
                # the membership set without re-reading the JSONL.
                self._note_event_id(room_id, _buffered_event_id)

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
            # F2a: this is the production text path and it activates BEFORE
            # dispatching _process_message, so the trigger must be threaded in
            # here too — otherwise gap-fill back-fills the very event that is
            # about to be processed and the JSONL gets two copies.
            #
            # R3-1: `_activate_room_once` serialises this with the lazy wake in
            # `_process_message` (and with a second concurrent live event here),
            # so only one gap-fill ever runs per dormant room.
            await self._activate_room_once(
                room_id, room_name=room_name,
                trigger_event_id=getattr(event, 'event_id', None),
                trigger_ts=self._event_ts(event),
            )

        # --- Slash commands ---
        if body == "/stop":
            self._halted_rooms.add(room_id)
            # Clear steering inbox so stale notes don't leak into the next turn
            if hasattr(self, '_steering_inbox'):
                self._steering_inbox.pop(room_id, None)
            # Clear any staged view_image tags so a halted room can't leak them
            # into the next turn (kdsn.279: the per-room vision inbox lives on
            # the AGENT now; getattr-guarded for tests with MagicMock agents).
            _vi = getattr(getattr(self, 'agent', None), '_vision_inbox', None)
            if _vi is not None:
                _vi.pop(room_id, None)
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
                _status_history = self.session_log.build_context(
                    room_id)
            status = self.agent.status(room_id, history=_status_history)
            ctx = status['context_tokens']
            ctx_max = status['context_max']
            ctx_pct = status['context_pct']
            bar_len = 20
            filled = round(bar_len * ctx_pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            # Resolve room-scoped overrides
            _effort = getattr(self, '_room_effort', {}).get(room_id) or self.agent.config.thinking
            _cache_ttl = getattr(self, '_room_cache_ttl', {}).get(room_id) or "1h (default)"
            lines = [
                f"### {status['name']}",
                "",
                "| | |",
                "|---|---|",
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
                f"| **Effort** | {_effort} |",
                f"| **Cache TTL** | {_cache_ttl} |",
                f"| **Timesense** | {'on' if getattr(self, '_room_timesense', {}).get(room_id) else 'off'} |",
            ]
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

        if body.startswith("/effort"):
            parts = body.split(None, 1)
            if len(parts) < 2:
                # Show current effort level
                current = self._room_effort.get(room_id)
                if current is None:
                    current = getattr(self.agent.config, 'thinking', 'off')
                    source = "config"
                else:
                    source = "override"
                await self.send(room_id, f"Effort: **{current}** ({source})")
                return
            level = parts[1].strip().lower()
            valid_levels = ("off", "low", "medium", "high", "xhigh", "max")
            if level not in valid_levels:
                await self.send(room_id, f"Invalid level. Use: {', '.join(valid_levels)}")
                return
            self._room_effort[room_id] = level
            # Persist override so it survives process restarts
            if self.session_log:
                self.session_log.append(
                    role="system",
                    sender=event.sender,
                    room=room_id,
                    event_id=None,
                    event="effort_override",
                    detail=level,
                )
            await self.send(room_id, f"Effort set to **{level}** for this room")
            return

        if body.startswith("/cache"):
            parts = body.split(None, 1)
            if len(parts) < 2:
                # Show current cache status: TTL + handoff boundary state
                current = self._room_cache_ttl.get(room_id)
                ttl_line = f"Cache TTL: **{current}** (override)" if current else "Cache TTL: **1h** (default)"
                # kdsn.305.1: the boundary state line — latest handoff
                # boundary index + trigger over handoff_boundary system
                # entries (legacy markers are NOT handoff boundaries).
                handoff_line = "Handoff boundary: none"
                if self.session_log:
                    _entries = self.session_log.read(room_id)
                    _idx, _trigger = _handoff_boundary_state(_entries)
                    if _idx is not None:
                        handoff_line = f"Handoff boundary: {_idx} ({_trigger})"
                await self.send(room_id, f"{ttl_line}\n{handoff_line}")
                return
            value = parts[1].strip().lower()
            if value == "gc":
                # kdsn.322: the context-handoff rework retired /cache gc —
                # the boundary command is /cache handoff now. Steer to the
                # new spelling; NO boundary is applied on the deprecated
                # one.
                await self.send_notice(
                    room_id,
                    "⚠️ /cache gc is deprecated — use /cache handoff "
                    "(the context-handoff rework, kdsn.322).",
                )
                return
            if value == "handoff":
                # /cache handoff — apply a handoff boundary at a clean
                # break (kdsn.322; renamed from /cache gc, which before
                # that was /cache toolstrip — both names retired).
                # Flag-on: full handoff boundary (manifest + durable
                # snapshot + full-strip history rebuild). Flag-off: the
                # legacy toolstrip behavior (bare strip marker, legacy
                # message, legacy rebuild) — retained for handoff-disabled
                # agents.
                if self.session_log:
                    if self.agent.config.context.handoff_enabled:
                        outcome = apply_boundary_and_rebuild(
                            self.agent, self.session_log, room_id,
                            trigger="manual", exclude_inflight=False)
                        if outcome.get("applied"):
                            # kdsn.305.12 R3: this operator path applies the
                            # boundary DIRECTLY (not via the
                            # apply_handoff_boundary callback), so the
                            # agent's applied-boundary
                            # bookkeeping — engine reset, hard-tier strikes,
                            # runway-fraction cache, churn-guard re-arm — must
                            # be driven here through the SAME seam the
                            # callback consumer uses
                            # (Agent._note_handoff_boundary_applied).
                            # Without it the handoff-runway reminder would
                            # keep evaluating the STALE pre-boundary runway.
                            # Fail-soft: a mock/legacy agent without the seam
                            # (or a malformed outcome) must never break the
                            # operator command.
                            _note = getattr(self.agent,
                                            "_note_handoff_boundary_applied", None)
                            if callable(_note):
                                try:
                                    _note(room_id, outcome)
                                except Exception:
                                    logger.warning(
                                        "handoff /cache: applied-boundary "
                                        "note failed for %s (fail-soft)",
                                        room_id, exc_info=True)
                            await self.send_notice(room_id, _handoff_confirm_text(outcome, "manual"))
                        else:
                            await self.send_notice(
                                room_id,
                                f"⚠️ Handoff boundary not applied — "
                                f"{outcome.get('noop_reason') or 'no-op'}",
                            )
                    else:
                        # Flag OFF — legacy toolstrip path (behavior + message
                        # byte-equivalent to the pre-handoff /cache toolstrip).
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
                        history.extend(self.session_log.build_context(
                            room_id))
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
                await self.send(room_id, "Invalid value. Use: `/cache 1h`, `/cache 5m`, or `/cache off`")
                return
            # Check if current model uses Anthropic provider
            try:
                model_str = self.agent.get_model(room_id)
                provider_cfg, _ = resolve_model_checked(model_str, self.agent.config.providers, aliases=self.agent.config.model_aliases, skipped_providers=getattr(self.agent.config, "skipped_providers", {}))
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

        if body.startswith("/project"):
            # /project operator command — active-project declaration.
            # Room-command path: operator AUTHORITY. Declaring again is an
            # allowed override (announced); this deliberately bypasses the
            # tool path's declare-once refusal.
            parts = body.split()
            # Grammar: `/project set <name>` (canonical), `/project <name>`
            # (convenience), bare `/project` (status).
            project = None
            if len(parts) >= 2 and parts[1] == "set" and len(parts) < 3:
                await self.send(
                    room_id,
                    "Usage: `/project set <name>` — a project name is required.",
                )
                return
            if len(parts) >= 3 and parts[1] == "set":
                project = parts[2]
            elif len(parts) == 2 and parts[1] != "set":
                project = parts[1]
            if project is not None:
                project = project.strip()
                # Name guard: this detail feeds workspace-relative path joins
                # at boundary application — no separators, no traversal.
                if ("/" in project or "\\" in project or project.startswith(".")
                        or project in ("", ".", "..")):
                    await self.send(
                        room_id,
                        "⚠️ Invalid project name — use a bare directory name "
                        "like `foo` (no paths, no dots-prefix).",
                    )
                    return
                if not self.session_log:
                    await self.send(room_id, "⚠️ No session log available.")
                    return
                agent_config = self.agent.config
                workspace = Path(agent_config.workspace)
                proj_dir = workspace / "memory" / "projects" / project
                entries = self.session_log.read(room_id)
                existing = read_active_project(entries)
                self.session_log.append(
                    role="system",
                    sender=event.sender,
                    room=room_id,
                    event_id=None,
                    event=ACTIVE_PROJECT_EVENT,
                    detail=project,
                )
                text = project_echo_text(workspace, project)
                if not proj_dir.is_dir():
                    text = (
                        f"⚠️ Directory not found: {proj_dir} — create "
                        "memory/projects/<name>/ before the next boundary.\n"
                        + text
                    )
                if existing is not None and existing != project:
                    text = (
                        f"⚠️ Operator override: was **{existing}**, now "
                        f"**{project}**.\n{text}"
                    )
                elif existing == project:
                    text = f"Re-declared (already **{project}**).\n{text}"
                await self.send_notice(room_id, text)
                return
            # No args — status: current project + declaration count.
            if self.session_log:
                entries = self.session_log.read(room_id)
            else:
                entries = []
            current = read_active_project(entries)
            count = sum(
                1 for e in entries
                if e.get("role") == "system" and e.get("event") == ACTIVE_PROJECT_EVENT
            )
            if current is None:
                await self.send(
                    room_id,
                    "Project: none declared. Use `/project set <name>` "
                    "(memory/projects/<name>/ must exist).",
                )
            else:
                await self.send(
                    room_id,
                    f"Project: **{current}** ({count} declaration(s) this epoch). "
                    "Use `/project set <name>` to change it.",
                )
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
            if len(parts) >= 2 and parts[1] == "schedule":
                # Parse schedule command: /heartbeat schedule "<spec>" [directive...]
                rest = body[len("/heartbeat schedule"):].lstrip()
                try:
                    tokens = shlex.split(rest)
                except ValueError as e:
                    msg = (
                        f"Usage error: {e}. Format: "
                        "`/heartbeat schedule \"<5-field cron spec>\" [directive...]`"
                    )
                    await self.send(room_id, msg)
                    return

                if not tokens:
                    msg = (
                        "Usage: "
                        "`/heartbeat schedule \"<5-field cron spec>\" [directive...]`"
                    )
                    await self.send(room_id, msg)
                    return

                # Check if unquoted cron spec (multiple tokens like cron fields)
                is_unquoted_cron = (
                    len(tokens) >= 5 and
                    all(t and (t[0].isdigit() or t[0] in '*,/-') for t in tokens[:5])
                )
                if is_unquoted_cron:
                    msg = (
                        "Cron spec must be quoted. Usage: "
                        "`/heartbeat schedule \"<5-field cron spec>\" [directive...]` "
                        "(e.g., `/heartbeat schedule \"30 6 * * 1-5\"`)"
                    )
                    await self.send(room_id, msg)
                    return

                spec = tokens[0]
                directive = " ".join(tokens[1:]) if len(tokens) > 1 else None

                # Validate spec
                try:
                    schedule.validate_spec(spec)
                except schedule.ScheduleError:
                    msg = (
                        "Invalid cron spec. Use 5-field format: "
                        "`minute hour day month day-of-week` (e.g., `30 6 * * 1-5`)."
                    )
                    await self.send(room_id, msg)
                    return

                # Check minimum gap
                try:
                    tz = ZoneInfo(schedule.DEFAULT_TZ)
                    gap = schedule.min_gap(spec, tz)
                    if gap < 300:  # heartbeat floor
                        msg = (
                            f"Schedule gap too small ({gap}s < 5m minimum). "
                            "Use a less frequent spec."
                        )
                        await self.send(room_id, msg)
                        return
                except schedule.ScheduleError:
                    msg = (
                        "Invalid cron spec. Use 5-field format: "
                        "`minute hour day month day-of-week` (e.g., `30 6 * * 1-5`)."
                    )
                    await self.send(room_id, msg)
                    return

                # Check mutual exclusion
                if self.umbral and self.umbral.is_active(room_id):
                    await self.send(room_id,
                        "Stop the umbral timer first (`/umbral stop`) — "
                        "umbral and heartbeat cannot run in the same room.")
                    return

                # Escape directive through the standard path
                if directive:
                    directive = escape_system_reminder_tags(directive)

                # Arm the schedule
                await self.heartbeat.start_schedule(
                    room_id, spec, tz=schedule.DEFAULT_TZ, directive=directive
                )
                directive_part = (
                    f" · directive: {_trunc_directive(directive)}"
                    if directive
                    else ""
                )
                hb_msg = (
                    f"Heartbeat started: every \"{spec}\" "
                    f"({schedule.DEFAULT_TZ}){directive_part}."
                )
                await self.send(room_id, hb_msg)
            elif len(parts) >= 3 and parts[1] == "start":
                interval = parse_interval(parts[2])
                if interval is None:
                    msg = "Invalid interval. Use e.g. `15m`, `1h`, `6h`."
                    await self.send(room_id, msg)
                elif interval < 300:
                    await self.send(room_id, "Minimum interval is 5m.")
                elif self.umbral and self.umbral.is_active(room_id):
                    await self.send(room_id,
                        "Stop the umbral timer first (`/umbral stop`) — "
                        "umbral and heartbeat cannot run in the same room.")
                else:
                    directive = (
                        parts[3] if len(parts) >= 4 else None
                    )
                    await self.heartbeat.start(room_id, interval, directive)
                    human = format_interval(interval)
                    msg = f"Heartbeat started: every {human} in this room."
                    await self.send(room_id, msg)
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
                        if nio_room:
                            name = (
                                getattr(nio_room, 'name', '') or
                                getattr(nio_room, 'display_name', '') or
                                e.room_id
                            )
                        else:
                            name = e.room_id
                        if e.schedule:
                            # Schedule mode
                            tz_name = e.tz or schedule.DEFAULT_TZ
                            next_str = format_interval(e.seconds_until_next)
                            line = (
                                f"- **{name}** — every \"{e.schedule}\" "
                                f"({tz_name}), next in {next_str}"
                            )
                        else:
                            # Interval mode
                            interval_str = format_interval(e.interval_seconds)
                            next_str = format_interval(e.seconds_until_next)
                            line = (
                                f"- **{name}** — every {interval_str}, "
                                f"next in {next_str}"
                            )
                        if e.directive:
                            line += f" · directive: {_trunc_directive(e.directive)}"
                        lines.append(line)
                    await self.send(room_id, "\n".join(lines))
            else:
                msg = (
                    "Usage: `/heartbeat schedule \"<cron>\"` | "
                    "`/heartbeat start <interval>` | `/heartbeat stop` | "
                    "`/heartbeat status`"
                )
                await self.send(room_id, msg)
            return

        if body.startswith("/umbral"):
            parts = body.split(None, 3)
            if len(parts) >= 2 and parts[1] == "schedule":
                # Parse schedule command: /umbral schedule "<spec>" [directive...]
                rest = body[len("/umbral schedule"):].lstrip()
                try:
                    tokens = shlex.split(rest)
                except ValueError as e:
                    msg = (
                        f"Usage error: {e}. Format: "
                        "`/umbral schedule \"<5-field cron spec>\" [directive...]`"
                    )
                    await self.send(room_id, msg)
                    return

                if not tokens:
                    msg = (
                        "Usage: "
                        "`/umbral schedule \"<5-field cron spec>\" [directive...]`"
                    )
                    await self.send(room_id, msg)
                    return

                # Check if unquoted cron spec (multiple tokens like cron fields)
                is_unquoted_cron = (
                    len(tokens) >= 5 and
                    all(t and (t[0].isdigit() or t[0] in '*,/-') for t in tokens[:5])
                )
                if is_unquoted_cron:
                    msg = (
                        "Cron spec must be quoted. Usage: "
                        "`/umbral schedule \"<5-field cron spec>\" [directive...]` "
                        "(e.g., `/umbral schedule \"0 20 * * 0\"`)"
                    )
                    await self.send(room_id, msg)
                    return

                spec = tokens[0]
                directive = " ".join(tokens[1:]) if len(tokens) > 1 else None

                # Validate spec
                try:
                    schedule.validate_spec(spec)
                except schedule.ScheduleError:
                    msg = (
                        "Invalid cron spec. Use 5-field format: "
                        "`minute hour day month day-of-week` (e.g., `0 20 * * 0`)."
                    )
                    await self.send(room_id, msg)
                    return

                # Check minimum gap
                try:
                    tz = ZoneInfo(schedule.DEFAULT_TZ)
                    gap = schedule.min_gap(spec, tz)
                    if gap < 1800:  # umbral floor
                        msg = (
                            f"Schedule gap too small ({gap}s < 30m minimum). "
                            "Use a less frequent spec."
                        )
                        await self.send(room_id, msg)
                        return
                except schedule.ScheduleError:
                    msg = (
                        "Invalid cron spec. Use 5-field format: "
                        "`minute hour day month day-of-week` (e.g., `0 20 * * 0`)."
                    )
                    await self.send(room_id, msg)
                    return

                # Check mutual exclusion
                if self.heartbeat and self.heartbeat.is_active(room_id):
                    await self.send(room_id,
                        "Stop the heartbeat first (`/heartbeat stop`) — "
                        "umbral and heartbeat cannot run in the same room.")
                    return

                # Escape directive through the standard path
                if directive:
                    directive = escape_system_reminder_tags(directive)

                # Arm the schedule
                await self.umbral.start_schedule(
                    room_id, spec, tz=schedule.DEFAULT_TZ, directive=directive
                )
                directive_part = (
                    f" · directive: {_trunc_directive(directive)}"
                    if directive
                    else ""
                )
                um_msg = (
                    f"🌑 Umbral started: every \"{spec}\" "
                    f"({schedule.DEFAULT_TZ}){directive_part}."
                )
                await self.send(room_id, um_msg)
            elif len(parts) >= 3 and parts[1] == "start":
                interval = parse_interval(parts[2])
                if interval is None:
                    msg = "Invalid interval. Use e.g. `30m`, `6h`."
                    await self.send(room_id, msg)
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
                    msg = f"🌑 Umbral started: every {human} in this room."
                    await self.send(room_id, msg)
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
                        if nio_room:
                            name = (
                                getattr(nio_room, 'name', '') or
                                getattr(nio_room, 'display_name', '') or
                                e.room_id
                            )
                        else:
                            name = e.room_id
                        if e.schedule:
                            # Schedule mode
                            tz_name = e.tz or schedule.DEFAULT_TZ
                            line = (
                                f"- **{name}** — every \"{e.schedule}\" ({tz_name}), "
                                f"next in {format_interval(e.seconds_until_next)}")
                        else:
                            # Interval mode
                            line = (
                                f"- **{name}** — every {format_interval(e.interval_seconds)}, "
                                f"next in {format_interval(e.seconds_until_next)}")
                        if e.directive:
                            line += f" · directive: {_trunc_directive(e.directive)}"
                        lines.append(line)
                    await self.send(room_id, "\n".join(lines))
            else:
                msg = (
                    "Usage: `/umbral schedule \"<cron>\"` | "
                    "`/umbral start <interval>` | `/umbral stop` | "
                    "`/umbral status`"
                )
                await self.send(room_id, msg)
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

        if body.startswith("/spotter"):
            # Spotter v1 operator API (design §10). Operator-namespace command:
            # a getattr-guard (a MagicMock / pre-Spotter agent has no
            # _spotter) reports unavailability rather than erroring. Sits
            # BEFORE the halted-room drop so it still answers while a room is
            # /stop-halted.
            _spotter = getattr(self.agent, "_spotter", None)
            if _spotter is None:
                await self.send_notice(room_id, "Spotter not available on this agent.")
                return
            parts = body.split()
            sub = parts[1] if len(parts) > 1 else "status"
            if sub == "status":
                out = _spotter.op_status(room_id)
            elif sub == "start":
                out = _spotter.op_start(room_id)
            elif sub == "stop":
                out = _spotter.op_stop(room_id)
            elif sub == "model" and len(parts) > 2:
                out = _spotter.op_set_model(room_id, parts[2])
            elif sub == "model":
                out = "Usage: /spotter model <alias>"
            else:
                out = "Usage: /spotter status|start|stop|model <alias>"
            await self.send_notice(room_id, out)
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

        # kdsn.292 §3c: if startup was degraded, announce it once per joined
        # room (fail-soft — never blocks the sync loop).
        await self._broadcast_degraded_start()

        # Resume persisted heartbeats and umbral timers
        heartbeat_catchup_rooms = await self.heartbeat.resume()
        umbral_catchup_rooms = await self.umbral.resume()

        # Post caught-up notices for missed scheduled fires
        hb_catchup_msg = (
            "⚠️ heartbeat: caught up a missed scheduled fire for this "
            "room (process was down across the scheduled instant)"
        )
        um_catchup_msg = (
            "⚠️ umbral: caught up a missed scheduled fire for this "
            "room (process was down across the scheduled instant)"
        )
        for room_id in heartbeat_catchup_rooms:
            await self.send_notice(room_id, hb_catchup_msg)
        for room_id in umbral_catchup_rooms:
            await self.send_notice(room_id, um_catchup_msg)

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
