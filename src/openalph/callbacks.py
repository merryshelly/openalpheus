"""Shared callback assembly for OpenAlph tools layer (kdsn.237 Phase 0).

Hoists callback-building logic out of MatrixBot so that callback assembly
no longer requires a live Matrix connection.  The Matrix path delegates
to these shared functions; headless paths (CLI, Phase 1) will use them
directly with a different CommsSinks implementation.
"""

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import sys

import mistune


# ---------------------------------------------------------------------------
# CommsSinks protocol (structural typing)
# ---------------------------------------------------------------------------

@runtime_checkable
class CommsSinks(Protocol):
    """Side-effect sink contract for the callback dict.

    Implementations:
      - MatrixSinks   (Phase 0) — delegates to a MatrixBot instance.
      - HeadlessSinks (Phase 1) — stdout / no-op.
    """

    async def send_notice(self, room_id, body, **kw) -> None: ...
    async def log_reminder(self, room_id, reminder) -> None: ...
    async def send_media(self, file_path, content_type, filename, caption=None) -> None: ...
    async def on_redaction(self, tool_name, events) -> None: ...
    async def on_keepalive_miss(self, room_id=None) -> None: ...
    async def on_degenerate(self, model=None, generation_id=None, **kw) -> None: ...
    async def log_vision_injection(self, room_id, framed) -> None: ...


# ---------------------------------------------------------------------------
# MatrixSinks — wraps a MatrixBot instance
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# HeadlessSinks — CommsSinks for CLI/headless mode (kdsn.237 Phase 1)
# ---------------------------------------------------------------------------

class HeadlessSinks:
    """CommsSinks for headless/CLI mode. Notices → stderr, no file delivery."""

    def __init__(self, session_log=None, agent_user_id=None):
        self._sl = session_log
        self._uid = agent_user_id or (session_log.agent_user_id if session_log else "cli")

    async def send_notice(self, room_id, body, **kw):
        print(body, file=sys.stderr, flush=True)

    async def log_reminder(self, room_id, reminder):
        if self._sl:
            self._sl.append(
                role="user",
                sender=self._uid,
                room=room_id,
                event_id=None,
                content=reminder.content,
                source="reminder",
                trigger=reminder.trigger,
            )
        print(reminder.content, file=sys.stderr, flush=True)

    async def send_media(self, file_path, content_type, filename, caption=None):
        raise RuntimeError(f"No delivery sink available in CLI mode — file remains at: {file_path}")

    async def on_redaction(self, tool_name, events):
        for event in events:
            print(f"🔒 Credential redacted in {tool_name}: {event.pattern_name}", file=sys.stderr)

    async def on_keepalive_miss(self, room_id=None):
        print("⚠️ cache keepalive missed", file=sys.stderr)

    async def on_degenerate(self, model=None, generation_id=None, **kw):
        print(f"⚠️ Degeneration detected — model: {model or 'unknown'}", file=sys.stderr)

    async def log_vision_injection(self, room_id, framed):
        """Log a view_image injection (kdsn.279): JSONL source='view_image' + a
        stderr notice. The framed tag text (pre-expansion) is persisted as-is —
        rehydration degrades it to plain text. No session_log → print only."""
        if self._sl is not None:
            self._sl.append(
                role="user",
                sender=self._uid,
                room=room_id,
                event_id=None,
                content=framed,
                source="view_image",
            )
        print(f"👁️ view_image: {framed.count('[media:')} image(s) attached",
              file=sys.stderr, flush=True)

class MatrixSinks:
    """CommsSinks backed by a live MatrixBot + nio client."""

    def __init__(self, bot, room_id: str):
        self._bot = bot
        self._room_id = room_id

    async def send_notice(self, room_id, body, **kw):
        """Emit collapsed <details> m.notice for reminders."""
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
        await self._bot._room_send_with_retry(room_id, content_msg)

    async def log_reminder(self, room_id, reminder):
        """Log reminder to JSONL with source='reminder' + trigger."""
        _sl = getattr(self._bot, 'session_log', None)
        if _sl:
            _sl.append(
                role="user",
                sender=self._bot.config.user_id,
                room=room_id,
                event_id=None,
                content=reminder.content,
                source="reminder",
                trigger=reminder.trigger,
            )

    async def send_media(self, file_path, content_type, filename, caption=None):
        """Upload media file to Matrix and send to the bound room."""
        await self._bot.upload_and_send(self._room_id, file_path, content_type, filename, caption)

    async def on_redaction(self, tool_name, events):
        """Emit in-room notice when credentials are redacted from tool output."""
        for event in events:
            notice = f"🔒 Credential redacted in {tool_name} output: {event.pattern_name} ({event.char_count} chars)"
            try:
                await self._bot.send_notice(self._room_id, notice)
            except Exception as exc:
                import logging
                logger = logging.getLogger(__name__)
                logger.error("Redaction notice failed in %s: %s", self._room_id, exc, exc_info=True)
            _sl = getattr(self._bot, 'session_log', None)
            if _sl:
                _sl.append(
                    role="system",
                    sender=self._bot.config.user_id,
                    room=self._room_id,
                    event_id=None,
                    event="credential_redaction",
                    detail=f"tool={tool_name} pattern={event.pattern_name} chars={event.char_count}",
                )

    async def on_keepalive_miss(self, room_id=None):
        """Emit notice + system log when cache keepalive detects a write (miss)."""
        rid = room_id or self._room_id
        notice = ("⚠️ cache keepalive missed (wrote instead of read) -- "
                  "disabling for this turn; next resume may bust cache")
        try:
            await self._bot.send_notice(rid, notice)
        except Exception as exc:
            import logging
            logger = logging.getLogger(__name__)
            logger.error("cache keepalive miss notice failed in %s: %s", rid, exc, exc_info=True)
        _sl = getattr(self._bot, 'session_log', None)
        if _sl:
            _sl.append(
                role="system",
                sender=self._bot.config.user_id,
                room=rid,
                event_id=None,
                event="cache_keepalive_miss",
                detail="ping wrote instead of read",
            )

    async def log_vision_injection(self, room_id, framed):
        """Log a view_image injection (kdsn.279): ONE JSONL entry
        (role=user, source='view_image', framed tag text) + a best-effort
        👁 room notice via the bot's PLAIN send_notice (not self.send_notice —
        that one wraps the body in reminder-style <details> HTML)."""
        _sl = getattr(self._bot, 'session_log', None)
        if _sl is not None:
            try:
                _sl.append(
                    role="user",
                    sender=self._bot.config.user_id,
                    room=room_id,
                    event_id=None,
                    content=framed,
                    source="view_image",
                )
            except Exception as exc:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("view_image session-log append failed in %s: %s",
                               room_id, exc, exc_info=True)
        try:
            await self._bot.send_notice(
                room_id,
                f"👁️ view_image: {framed.count('[media:')} image(s) attached")
        except Exception:
            pass

    async def on_degenerate(self, model=None, generation_id=None, **kw):
        """Emit an m.notice when the degen detector flags a response."""
        body = f"⚠️ Degeneration detected — model: {model or 'unknown'}"
        if generation_id:
            body += f", generation: {generation_id}"
        content_msg = {
            "msgtype": "m.notice",
            "body": body,
        }
        await self._bot._room_send_with_retry(self._room_id, content_msg)


# ---------------------------------------------------------------------------
# build_context_status — headless
# ---------------------------------------------------------------------------

def build_context_status(agent, room_id, *, room_name=None, session_log=None, heartbeat=None, umbral=None) -> dict:
    """Assemble context status data dict for the context_status tool.

    Collects agent status, session age, heartbeat state, umbral state,
    and room identity for the given room ID. Sync-safe: calls only sync
    methods on agent, session_log, heartbeat, umbral.
    """
    _hist = session_log.build_context(room_id) if session_log else None
    status_data = agent.status(room_id, history=_hist)

    # Session age
    if session_log:
        entries = session_log.read(room_id)
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
    if heartbeat and heartbeat.is_active(room_id):
        hb_entries = heartbeat.status()
        hb_entry = next((e for e in hb_entries if e.room_id == room_id), None)
        if hb_entry:
            status_data["heartbeat_active"] = True
            schedule = getattr(hb_entry, 'schedule', None)
            # Check if schedule is a non-empty string (not just truthy, to avoid MagicMock objects in tests)
            if isinstance(schedule, str) and schedule:
                # Schedule mode
                status_data["heartbeat_schedule"] = schedule
                status_data["heartbeat_tz"] = getattr(hb_entry, 'tz', None)
                status_data["heartbeat_next_minutes"] = round(hb_entry.seconds_until_next / 60)
                status_data["heartbeat_interval_minutes"] = None
            else:
                # Interval mode
                status_data["heartbeat_interval_minutes"] = round(hb_entry.interval_seconds / 60)
                status_data["heartbeat_next_minutes"] = round(hb_entry.seconds_until_next / 60)
                status_data["heartbeat_schedule"] = None
                status_data["heartbeat_tz"] = None
        else:
            status_data["heartbeat_active"] = False
            status_data["heartbeat_interval_minutes"] = None
            status_data["heartbeat_next_minutes"] = None
            status_data["heartbeat_schedule"] = None
            status_data["heartbeat_tz"] = None
    else:
        status_data["heartbeat_active"] = False
        status_data["heartbeat_interval_minutes"] = None
        status_data["heartbeat_next_minutes"] = None
        status_data["heartbeat_schedule"] = None
        status_data["heartbeat_tz"] = None

    # Umbral state (context rotation timer)
    if umbral and umbral.is_active(room_id):
        um_entries = umbral.status()
        um_entry = next((e for e in um_entries if e.room_id == room_id), None)
        if um_entry:
            status_data["umbral_active"] = True
            schedule = getattr(um_entry, 'schedule', None)
            # Check if schedule is a non-empty string (not just truthy, to avoid MagicMock objects in tests)
            if isinstance(schedule, str) and schedule:
                # Schedule mode
                status_data["umbral_schedule"] = schedule
                status_data["umbral_tz"] = getattr(um_entry, 'tz', None)
                status_data["umbral_next_minutes"] = round(um_entry.seconds_until_next / 60)
                status_data["umbral_interval_minutes"] = None
            else:
                # Interval mode
                status_data["umbral_interval_minutes"] = round(um_entry.interval_seconds / 60)
                status_data["umbral_next_minutes"] = round(um_entry.seconds_until_next / 60)
                status_data["umbral_schedule"] = None
                status_data["umbral_tz"] = None
        else:
            status_data["umbral_active"] = False
            status_data["umbral_interval_minutes"] = None
            status_data["umbral_next_minutes"] = None
            status_data["umbral_schedule"] = None
            status_data["umbral_tz"] = None
    else:
        status_data["umbral_active"] = False
        status_data["umbral_interval_minutes"] = None
        status_data["umbral_next_minutes"] = None
        status_data["umbral_schedule"] = None
        status_data["umbral_tz"] = None

    # Room identity
    status_data["room_id"] = room_id
    status_data["room_name"] = room_name

    return status_data


# ---------------------------------------------------------------------------
# build_callbacks — headless
# ---------------------------------------------------------------------------

def build_callbacks(
    agent,
    room_id,
    sinks,
    *,
    turn_source=None,
    session_log=None,
    heartbeat=None,
    umbral=None,
    advisor_results=None,
    subagent_results=None,
    room_name=None,
) -> dict:
    """Build the callback dict for handle_input.

    Agent-state callbacks read directly from *agent*.  Side-effect callbacks
    delegate to *sinks* (a CommsSinks implementation).  Returns a dict with
    the 17 keys the tools layer expects.
    """

    async def _context_status_callback(req_room_id=None):
        return build_context_status(
            agent,
            req_room_id or room_id,
            room_name=room_name,
            session_log=session_log,
            heartbeat=heartbeat,
            umbral=umbral,
        )

    async def _send_media_callback(file_path, content_type, filename, caption=None):
        await sinks.send_media(file_path, content_type, filename, caption)

    async def _redaction_callback(tool_name, events):
        await sinks.on_redaction(tool_name, events)

    async def _send_notice_callback(_room_id, body, **kw):
        await sinks.send_notice(_room_id, body, **kw)

    async def _log_reminder_callback(_room_id, reminder):
        await sinks.log_reminder(_room_id, reminder)

    async def _log_vision_injection_callback(_room_id, framed):
        await sinks.log_vision_injection(_room_id, framed)

    async def _keepalive_miss_callback(_room_id=None):
        await sinks.on_keepalive_miss(_room_id or room_id)

    async def _on_degenerate_callback(model=None, generation_id=None, **kw):
        await sinks.on_degenerate(model=model, generation_id=generation_id, **kw)

    # Per-room read registry for file_write guard (getattr for mock compat)
    _registries = getattr(agent, '_read_registries', None)
    if _registries is None:
        _registries = {}
        try:
            agent._read_registries = _registries
        except AttributeError:
            pass  # spec-mocked agent, read_registry will be empty dict
    _read_registry = _registries.setdefault(room_id, {})

    # Per-room consult counter (getattr for mocked-agent compatibility)
    _advisor_uses = getattr(agent, '_advisor_uses', None)
    if _advisor_uses is None:
        _advisor_uses = {}
        try:
            agent._advisor_uses = _advisor_uses
        except AttributeError:
            pass

    return {
        "send_media": _send_media_callback,
        "on_redaction": _redaction_callback,
        "on_keepalive_miss": _keepalive_miss_callback,
        "on_degenerate": _on_degenerate_callback,
        "context_status": _context_status_callback,
        "send_notice": _send_notice_callback,
        "log_reminder": _log_reminder_callback,
        "log_vision_injection": _log_vision_injection_callback,
        "turn_source": turn_source,
        "read_registry": _read_registry,
        "room_id": room_id,
        "get_transcript": lambda: (agent.system_prompt, list(agent.history(room_id))),
        "advisor_uses": _advisor_uses,
        "advisor_results": {} if advisor_results is None else advisor_results,
        "subagent_results": {} if subagent_results is None else subagent_results,
        # kdsn.290: raw timer managers (or None on transports that don't wire
        # them — e.g. headless CLI). Forwarded so the heartbeat tool can reach
        # them via callbacks["heartbeat"] / callbacks["umbral"].
        "heartbeat": heartbeat,
        "umbral": umbral,
    }
