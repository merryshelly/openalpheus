"""Shared callback assembly for OpenAlph tools layer (kdsn.237 Phase 0).

Hoists callback-building logic out of MatrixBot so that callback assembly
no longer requires a live Matrix connection.  The Matrix path delegates
to these shared functions; headless paths (CLI, Phase 1) will use them
directly with a different CommsSinks implementation.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

import html
import logging
import sys

import mistune

from openalph.reminders import Reminder
from openalph.handoff import (
    ACTIVE_PROJECT_EVENT,
    apply_boundary_and_rebuild,
    current_boundary_index,
    project_echo_text,
    project_valid_name,
    read_active_project,
)

logger = logging.getLogger(__name__)


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
    async def log_spotter_flag(self, room_id, raw_payload, *, flag_class=None, flag_severity=None) -> None: ...


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
                # kdsn.298: conditional pass-through — plain T1–T6 reminders
                # must NOT gain a detail key in the JSONL (review LOW-13).
                # isinstance (not attr probing): duck-typed/mocked reminders
                # (e.g. MagicMock in legacy sink tests) auto-fabricate a
                # truthy .detail that would corrupt the JSONL.
                **({"detail": reminder.detail}
                   if isinstance(reminder, Reminder)
                   and reminder.detail is not None else {}),
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

    async def log_spotter_flag(self, room_id, raw_payload, *, flag_class=None, flag_severity=None):
        """Log a delivered Spotter flag (design §9): JSONL source='spotter' + a
        stderr line. The RAW FLAG block (pre-framing) is persisted — the
        advisory frame is context-only (build_context re-frames on rebuild).
        Fail-soft: a session-log write failure must never break the turn. No
        session_log → print only."""
        if self._sl is not None:
            try:
                self._sl.append(
                    role="user",
                    sender=self._uid,
                    room=room_id,
                    event_id=None,
                    content=raw_payload,
                    source="spotter",
                    flag_class=flag_class,
                    flag_severity=flag_severity,
                )
            except Exception:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("spotter session-log append failed in %s",
                               room_id, exc_info=True)
        print(f"🔍 spotter flag ({flag_severity}, {flag_class})",
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
                # kdsn.298: conditional pass-through — plain T1–T6 reminders
                # must NOT gain a detail key in the JSONL (review LOW-13).
                # isinstance (not attr probing): duck-typed/mocked reminders
                # (e.g. MagicMock in legacy sink tests) auto-fabricate a
                # truthy .detail that would corrupt the JSONL.
                **({"detail": reminder.detail}
                   if isinstance(reminder, Reminder)
                   and reminder.detail is not None else {}),
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

    async def log_spotter_flag(self, room_id, raw_payload, *, flag_class=None, flag_severity=None):
        """Log a delivered Spotter flag (design §9): ONE JSONL entry
        (role=user, source='spotter', RAW FLAG block + class/severity).
        The advisory frame is context-only — build_context re-frames it on
        rebuild, so live and rebuilt advisory bytes are identical.
        Fail-soft: a session-log write failure must never break the turn."""
        _sl = getattr(self._bot, 'session_log', None)
        if _sl is not None:
            try:
                _sl.append(
                    role="user",
                    sender=self._bot.config.user_id,
                    room=room_id,
                    event_id=None,
                    content=raw_payload,
                    source="spotter",
                    flag_class=flag_class,
                    flag_severity=flag_severity,
                )
            except Exception as exc:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("spotter session-log append failed in %s: %s",
                               room_id, exc, exc_info=True)

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
    # Full strip (kdsn.322 T1): build_context no longer takes thinking-tail
    # kwargs; the handoff render is driven by the room's boundary markers.
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

# kdsn.322.14: fold cap for the reinserted progress.md — past this the fold
# truncates with a pointer to the on-disk file. The snapshot in the JSONL
# remains the complete audit record regardless.
_PROGRESS_FOLD_CAP_CHARS = 16_000


def render_handoff_notice(trigger: str, outcome: dict) -> str:
    """Operator-facing notice for an APPLIED handoff boundary (§3.4).

    THE shared renderer — every applied-boundary path (tool, auto, hard,
    slash) routes through this; kdsn.322.14 killed the two-renderer drift.
    Line 1 is the headline (the Matrix sink renders it as the collapsed
    <details> summary); everything after is the fold.

    Format (SB-ruled 2026-09-04): composite before→after tokens — the SAME
    accounting as /status (system prompt + tool defs + render), with the
    after-number MEASURED (snapshot + surviving tail), never the retired
    pinned-0; checkpoint as the bare status value (no definitional gloss);
    the reinserted file list; and the FROZEN progress.md below the fold.
    Insert-language throughout — never "inject". Display-only: the JSONL
    marker is the audit record.
    """
    manifest = outcome.get("manifest") or {}
    durable = manifest.get("durable") or {}
    runway = manifest.get("runway") or {}
    ckpt = manifest.get("checkpoint")
    ck = ckpt.get("status", "unknown") if isinstance(ckpt, dict) else "unknown"
    project = durable.get("project")
    tb = manifest.get("tokens_before")
    ta = manifest.get("tokens_after")
    if ta is None:
        # Legacy manifest (pre-kdsn.322.14): fall back to the message-side
        # runway figure rather than showing nothing.
        ta = runway.get("tokens_after")
    dropped = manifest.get("tokens_dropped")
    files = durable.get("files") or []

    head = f"🪢 Handoff boundary applied ({trigger})"
    if tb is not None and ta is not None:
        head += f" — context ~{tb:,} → ~{ta:,} tok"
    elif tb is not None:
        head += f" — context ~{tb:,} tok"
    if project:
        head += f" · project: {project}"
    head += f" · checkpoint: {ck}"
    if files:
        names = []
        for f in files:
            base = str(f.get("path", "?")).rsplit("/", 1)[-1]
            if base not in names:
                names.append(base)
        head += f" · reinserted {len(files)} files: {', '.join(names)}"

    fold = []
    bits = []
    if dropped is not None:
        bits.append(f"dropped: ~{dropped:,} tok")
    if durable:
        bits.append(f"durable budget: "
                    f"{durable.get('used_tokens', 0):,} / "
                    f"{durable.get('budget_tokens', 0):,}")
        if durable.get("over_budget") or outcome.get("over_budget"):
            bits.append("over")
    if bits:
        fold.append(" · ".join(bits))
    fold.append(f"boundary index: {manifest.get('boundary_index', '?')} "
                f"(session JSONL marker — audit ref)")
    errs = manifest.get("errors") or []
    fold.append(f"errors: {len(errs)}" if errs else "errors: none")
    progress = outcome.get("progress_md")
    if progress:
        text = html.escape(str(progress))
        if len(text) > _PROGRESS_FOLD_CAP_CHARS:
            removed = len(text) - _PROGRESS_FOLD_CAP_CHARS
            pointer = (f"memory/projects/{project}/progress.md"
                       if project else "the project progress.md")
            text = (text[:_PROGRESS_FOLD_CAP_CHARS] +
                    f"\n[truncated: {removed} chars removed — full text: "
                    f"{pointer}]")
        fold.append("── progress.md as reinserted (frozen at boundary) ──")
        fold.append(text)
    return head + "\n" + "\n".join(fold)


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
    the 17 keys the tools layer expects — plus an 18th, `log_spotter_flag`,
    when the owning agent carries a real SpotterManager (Spotter v1, design
    §9: the delivery-time I1 JSONL seam). Agents without a wired spotter
    (pre-Spotter / transport-less) keep the historical 17-key wire format.
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

    async def _log_spotter_flag_callback(_room_id, raw_payload,
                                         flag_class=None, flag_severity=None):
        await sinks.log_spotter_flag(_room_id, raw_payload,
                                     flag_class=flag_class,
                                     flag_severity=flag_severity)

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

    _callbacks = {
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

    # 18th key — Spotter v1 delivery-time JSONL seam (design §9). Conditional
    # on the owning agent carrying a REAL SpotterManager so that pre-Spotter
    # and mock agents keep the historical 17-key wire format (the type-NAME
    # check is load-bearing: a MagicMock agent auto-fabricates a truthy
    # `_spotter` attribute that must not count, and callbacks.py must not
    # import spotter.py to ask properly — design §0 import graph).
    _spotter = getattr(agent, "_spotter", None)
    if type(_spotter).__name__ == "SpotterManager":
        _callbacks["log_spotter_flag"] = _log_spotter_flag_callback
    # kdsn.305.1: GC boundary + project declaration tool callbacks. Built at
    # THIS seam (the one construction seam) so every transport — Matrix live
    # turns, heartbeat/umbral turns, headless CLI — carries identical wiring.
    # session_log=None (no JSONL on this transport) → keys are None; the tool
    # handlers steer to the operator command and the agent loop skips
    # silently (durability-gate discipline).
    if session_log is not None:
        async def _gc_apply_cb(_room_id, *, trigger, exclude_inflight=True):
            # Opt-out is opt-out (audit): with context.handoff_enabled=false
            # the tool path must NOT apply boundaries either — the tiers and
            # the operator command respect the flag; the seam did not.
            _gc_cfg = getattr(agent.config, "context", None)
            if _gc_cfg is None or not getattr(_gc_cfg, "handoff_enabled", False):
                return {
                    "applied": False,
                    "noop_reason": "context.handoff_enabled is false — handoff disabled for this agent",
                    "manifest": None,
                    "over_budget": False,
                }
            try:
                entries = session_log.read(_room_id)
            except Exception as e:
                logger.warning("gc boundary read failed for %s: %s", _room_id, e)
                return {
                    "applied": False,
                    "noop_reason": f"boundary failed: {type(e).__name__}",
                    "manifest": None,
                    "over_budget": False,
                }
            # Turn cooldown — TOOL-TRIGGERED boundaries only. Operator
            # commands (/cache handoff) and the loop's auto/hard tiers
            # bypass it.
            # Count by list POSITION: SessionLog.append only writes
            # entry_index on marker events, never on assistant entries.
            # Post-boundary positions start at `boundary`; no boundary (-1)
            # counts everything.
            if trigger == "tool" and exclude_inflight:
                cfg = getattr(agent.config, "context", None)
                cooldown = getattr(cfg, "turn_cooldown", 3)
                boundary = current_boundary_index(entries)
                count = sum(
                    1 for p, e in enumerate(entries)
                    if e.get("role") == "assistant" and p >= boundary
                )
                if count < cooldown:
                    return {
                        "applied": False,
                        "noop_reason": (
                            f"cooldown: {count} of {cooldown} turns "
                            "since last boundary"
                        ),
                        "manifest": None,
                        "over_budget": False,
                    }
            outcome = apply_boundary_and_rebuild(
                agent, session_log, _room_id,
                trigger=trigger, exclude_inflight=exclude_inflight,
                # Tool path = LIVE turn: the rebuild must preserve the
                # in-flight assistant (audit: full-scan orphan repair would
                # strip the live pair and orphan its result mid-loop).
                live_turn=exclude_inflight,
            )
            # §3.4 operator visibility: a collapsed room notice on EVERY
            # applied boundary (auto / hard / tool through this seam; the
            # /cache handoff operator path sends its own confirmation).
            # No-op outcomes carry NO notice — the caller surfaces the
            # noop_reason. Fail-soft: a notice failure must never break the
            # boundary application.
            if isinstance(outcome, dict) and outcome.get("applied"):
                try:
                    await sinks.send_notice(
                        _room_id, render_handoff_notice(trigger, outcome))
                except Exception:
                    logger.warning(
                        "handoff visibility notice failed for %s (fail-soft)",
                        _room_id, exc_info=True)
            return outcome

        async def _gc_set_project_cb(project):
            entries = session_log.read(room_id)
            existing = read_active_project(entries)
            if existing is not None:
                if existing == project:
                    return {"ok": True, "text": "already declared (no-op)"}
                return {
                    "ok": False,
                    "text": (
                        f"Project '{existing}' already declared for this room — "
                        "one project per room; escalate to your operator"
                    ),
                }
            if not project_valid_name(project):
                return {
                    "ok": False,
                    "text": ("Invalid project name — use a bare directory "
                             "name like `foo` (no paths, no dots-prefix)."),
                }
            workspace = Path(agent.config.workspace)
            proj_dir = workspace / "memory" / "projects" / project
            if not proj_dir.is_dir():
                return {
                    "ok": False,
                    "text": f"Project directory does not exist: {proj_dir}",
                }
            session_log.append(
                role="system",
                sender=session_log.agent_user_id,
                room=room_id,
                event=ACTIVE_PROJECT_EVENT,
                detail=project,
            )
            return {
                "ok": True,
                "text": project_echo_text(workspace, project),
            }

        _callbacks["apply_handoff_boundary"] = _gc_apply_cb
        _callbacks["set_active_project"] = _gc_set_project_cb
    else:
        _callbacks["apply_handoff_boundary"] = None
        _callbacks["set_active_project"] = None

    return _callbacks
