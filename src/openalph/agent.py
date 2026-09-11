"""Conversation loop for an OpenAlph agent.

Ties together config, prompt assembly, and the provider adapter into a
stateful conversation agent. Keeps a running history and token counts
so the operator can check usage without external tooling.
"""

import asyncio
import contextlib
import base64
import functools
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from openalph.config import AgentConfig
from openalph.prompt import assemble_prompt
from openalph.provider import complete, stream, ping_cache, ThinkingBlock, Usage, compute_cost, model_supports_vision
from openalph.tools import discover_tools, execute_tool, truncate_result, wrap_tool_result, escape_system_reminder_tags, _TODO_STATE, ToolResult
# kdsn.330 (async subagents): the dispatch ledger core (pure module) and the
# sub runner, imported THROUGH THE MODULE so call-site attribute lookup
# resolves test patches of openalph.tools.subagent.run_subagent.
from openalph.tools import subledger
from openalph.tools import subagent as _tools_subagent
from openalph.reminders import ReminderEngine, ReminderState
from openalph.spotter import SpotterManager

logger = logging.getLogger(__name__)

# Supported image MIME types for vision support
VISION_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Media tag regex: [media: <path> (<mime>, <size>)]
MEDIA_TAG_RE = re.compile(r'\[media:\s*(.+?)\s+\(([^,]+),\s*([^)]+)\)\]')

# Approximate tokens per raw image byte (base64 decoded)
IMAGE_TOKENS_PER_BYTE = 1 / 750

# Wire format overhead per tool interaction (JSON envelope chars not in content)
_TOOL_CALL_OVERHEAD_CHARS = 80   # {"type":"tool_use","id":"...","name":"...","input":}
_TOOL_RESULT_OVERHEAD_CHARS = 80  # {"type":"tool_result","tool_use_id":"...","content":}


def _build_user_content(text: str, config: AgentConfig, *, vision: bool) -> str | list[dict]:
    """Build user message content, expanding image media tags when the room's
    active model supports vision.

    The expansion gate is the explicit keyword-only ``vision`` flag — resolved
    per call from the room's active model via
    ``openalph.provider.model_supports_vision`` (kdsn.275). There is no
    agent-level vision config field anymore.

    Returns plain text string when no image expansion needed.
    Returns list of content blocks when images are present and vision is enabled.
    """
    # If vision is disabled, return text unchanged
    if not vision:
        return text

    # Find all media tags
    matches = list(MEDIA_TAG_RE.finditer(text))
    if not matches:
        return text

    content_blocks = []
    last_end = 0
    images_found = False

    for match in matches:
        path_str = match.group(1)
        mime_type = match.group(2)

        # Add text before this tag
        if match.start() > last_end:
            text_before = text[last_end:match.start()]
            if text_before.strip():
                content_blocks.append({"type": "text", "text": text_before})

        # Check if this is a supported image type
        if mime_type in VISION_MIME_TYPES:
            # Resolve path relative to workspace
            image_path = config.workspace / path_str
            try:
                image_bytes = image_path.read_bytes()
                if image_bytes:
                    base64_data = base64.b64encode(image_bytes).decode("ascii")
                    content_blocks.append({
                        "type": "image",
                        "media_type": mime_type,
                        "data": base64_data,
                    })
                    images_found = True
                else:
                    # Empty file - leave tag as text
                    content_blocks.append({"type": "text", "text": match.group(0)})
            except FileNotFoundError:
                logger.warning(f"Image file not found: {image_path}")
                content_blocks.append({"type": "text", "text": match.group(0)})
            except OSError as e:
                logger.warning(f"Failed to read image file {image_path}: {e}")
                content_blocks.append({"type": "text", "text": match.group(0)})
        else:
            # Not a supported image type - leave as text
            content_blocks.append({"type": "text", "text": match.group(0)})

        last_end = match.end()

    # Add any remaining text after the last tag
    if last_end < len(text):
        text_after = text[last_end:]
        if text_after.strip():
            content_blocks.append({"type": "text", "text": text_after})

    # If no images were expanded, return original text
    if not images_found:
        return text

    # Merge consecutive text blocks for efficiency
    merged_blocks = []
    current_text = ""

    for block in content_blocks:
        if block["type"] == "text":
            current_text += block["text"]
        else:
            if current_text:
                merged_blocks.append({"type": "text", "text": current_text})
                current_text = ""
            merged_blocks.append(block)

    if current_text:
        merged_blocks.append({"type": "text", "text": current_text})

    return merged_blocks


class ContextOverflowError(Exception):
    """Raised when conversation context exceeds model capacity."""

    def __init__(self, current_tokens: int, max_tokens: int):
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"Context overflow: ~{current_tokens:,} tokens exceeds model capacity "
            f"({max_tokens:,}). Start a new room."
        )


# --- Subagent cache keepalive (workspace-kdsn.190) -------------------------
KEEPALIVE_REFRESH_FRACTION = 0.8   # refresh at 80% of the TTL window
KEEPALIVE_MIN_INTERVAL_S = 60      # floor (protects a 5m-TTL room)
KEEPALIVE_MIN_CACHE_READ = 1000    # a real prefix hit reads at least this many tokens
_KEEPALIVE_TTL_SECONDS = {"5m": 300, "1h": 3600}
KEEPALIVE_MAX_CONSECUTIVE_ERRORS = 3   # abort keepalive after this many back-to-back ping errors
KEEPALIVE_ERROR_BACKOFF_S = 5          # quick retry after a transient ping error (capped by interval)


def _keepalive_ttl_seconds(cache_ttl: str | None) -> int:
    """Resolve a cache_ttl label to seconds; unknown/None -> 1h (platform default)."""
    return _KEEPALIVE_TTL_SECONDS.get(cache_ttl or "1h", 3600)


def _keepalive_interval(cache_ttl: str | None) -> float:
    """Refresh interval: a fraction of the TTL, floored at KEEPALIVE_MIN_INTERVAL_S."""
    return max(_keepalive_ttl_seconds(cache_ttl) * KEEPALIVE_REFRESH_FRACTION,
               KEEPALIVE_MIN_INTERVAL_S)


def _keepalive_is_hit(read: int | None, write: int | None) -> bool:
    """A ping is a cache HIT when the read dominates the write and clears the floor."""
    read = read or 0
    write = write or 0
    return read > write and read >= KEEPALIVE_MIN_CACHE_READ


class Agent:
    """A single-conversation agent backed by an LLM provider."""

    def __init__(self, config: AgentConfig, tool_executors: dict | None = None):
        self.config = config
        # Per-tool executor override map (workspace-kdsn.317): {tool name:
        # async callable} used by the execute_tool dispatch below to redirect
        # individual tool executions (e.g. shell into a task container)
        # without monkey-patching. Stored DEFENSIVELY — later mutation of the
        # caller's dict must not affect this agent. Default None → every
        # built-in branch runs exactly as before (byte-identical path).
        if tool_executors is not None:
            if not isinstance(tool_executors, dict):
                raise ValueError(
                    "tool_executors must be a dict mapping tool name -> "
                    "async callable, or None")
            for _tx_name, _tx_fn in tool_executors.items():
                if not callable(_tx_fn):
                    raise ValueError(
                        f"tool_executors[{_tx_name!r}] must be a callable "
                        f"(got {type(_tx_fn).__name__})")
            tool_executors = dict(tool_executors)
        self.tool_executors = tool_executors
        # Build the system prompt once at init — it won't change mid-conversation.
        # This reads workspace files (SOUL.md, OPERATOR.md, etc.) and builds a
        # skills index, all determined by the workspace directory in config.
        self.system_prompt = assemble_prompt(
            config.workspace,
            model_aliases=config.model_aliases,
            injection_defense=config.injection_defense,
            # workspace-kdsn.305: the 8th operator file (CONTINUITY.md) is
            # assembled only when context handoff is on — handoff_enabled=False
            # (opt-out via [context]) appends nothing, ever.
            gc_enabled=config.context.handoff_enabled,
        )
        self._rooms: dict[str, list[dict]] = {}  # room_id → history
        self.uncached_input_tokens = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
        self.total_output_tokens = 0
        self.total_tool_calls = 0
        # Per-room usage tracking (Workstream C)
        self._room_usage: dict[str, dict[str, int]] = {}
        self._last_turn_usage: dict[str, dict[str, int]] = {}
        self._last_stop_reason: dict[str, str] = {}
        # Discover tools from workspace/tools/ directory
        self.tools = discover_tools(config.workspace)
        # Pre-compute tool definition cost for token estimation.
        # Tool schemas are sent with every API call via the tools parameter.
        self._tool_defs_chars = 0
        if self.tools:
            for t in self.tools:
                params_chars = len(json.dumps(t.parameters)) if isinstance(t.parameters, dict) else 0
                self._tool_defs_chars += len(t.name) + len(t.description) + params_chars
        # CORE-1: per-room, mirroring `_room_locks` below.
        #
        # This was a single `_current_task` slot while room turns run
        # CONCURRENTLY -- every message is dispatched through
        # `_fire_background`, and each turn holds only its own per-room lock,
        # so two rooms can be inside `handle_input` at once. Last writer won,
        # which broke `/stop` in both directions: a `/stop` in room A could
        # cancel room B's turn (whichever wrote the slot last), and the
        # `finally` that nulled the slot when ANY turn ended could leave a
        # `/stop` cancelling nothing while a turn was genuinely running.
        # `_halted_rooms` was already correctly per-room, so A's future
        # messages were dropped while A's actual in-flight turn kept going.
        self._current_tasks: dict[str, asyncio.Task] = {}
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._room_models: dict[str, str] = {}  # room_id → model override
        # kdsn.329: per-room provider-usage token anchor —
        # room_id → (true_prompt_tokens, char_cursor). char_cursor None =
        # floor-only (restart rehydration); int = the char total of the exact
        # payload the provider last measured (live record seam). The
        # estimator merges this with the chars//4 heuristic — see
        # _estimate_context_tokens. Anchoring only ever RAISES estimates.
        self._token_anchor: dict[str, tuple[int, int | None]] = {}
        # kdsn.333: per-room EPOCH record — the last applied handoff boundary
        # as (boundary_index, manifest ts). Read (NOT consumed) by
        # _orient_inputs on every populated turn-start so session-orient
        # labels the epoch for the epoch's whole lifetime: a mid-epoch /model
        # switch refires with the same "Epoch began" row (audit M1 —
        # pop-on-read silently dropped the label on refires). Overwritten by
        # the next applied boundary; cleared by reset_room.
        self._handoff_epoch: dict[str, tuple[int, str]] = {}
        self._warned_models: set = set()  # warn-once for unknown model context windows
        self._truncation_retry = False
        # R1-4: Per-room reminder engines (replaces shared _reminder_engine)
        self._reminder_engines: dict[str, ReminderEngine] = {}
        # Per-room per-tool-name call counts (session scope)
        self._room_tool_counts: dict[str, dict[str, int]] = {}
        # R1-1: Per-room read registries for file_write guard
        self._read_registries: dict[str, dict] = {}
        # Per-room advisor consult counters (session cap tracking)
        self._advisor_uses: dict[str, int] = {}
        # view_image staging (kdsn.279): per-room inbox of [media:] tags deposited
        # by the view_image tool, drained at the top of the next tool-loop
        # iteration into ONE framed user message. Agent-owned (moved off
        # MatrixBot) so heartbeat/umbral/CLI turns stage + inject too.
        self._vision_inbox: dict[str, list[str]] = {}
        # Context handoff (workspace-kdsn.322, supersedes kdsn.305):
        # per-room hard-tier strike counter (consecutive FAILED boundary
        # attempts at the send-time overflow guard; >=
        # _GC_HARD_STRIKE_LIMIT -> raise as today) and the last applied
        # boundary's post-boundary runway consumption fraction
        # (handoff-runway reminder input, kdsn.305.12 D5 — replaces the
        # retired durable-budget fraction). Both in-memory only: the
        # fraction cache starts empty after restart -> 0.0 ->
        # handoff-runway silent (fail-safe, acceptable v1).
        # The turn-start auto tier + hard tier consume the transport-wired
        # callbacks["apply_handoff_boundary"] seam (contract: async callable
        # (room_id, *, trigger, exclude_inflight=True) -> {"applied",
        # "noop_reason", "manifest", "over_budget"}; on applied=True it has
        # ALREADY appended the JSONL entries AND rebuilt the room's
        # in-memory history in place, so the caller re-reads history()).
        self._gc_fail_strikes: dict[str, int] = {}
        self._gc_last_runway: dict[str, float] = {}
        # Auto-tier churn guard (audit): True when the most recent auto
        # boundary failed to bring the estimate under the auto threshold —
        # re-applying every turn start would append a fresh snapshot per turn
        # (JSONL growth) without ever clearing. Reset by any CLEARED boundary
        # or reset_room.
        self._gc_auto_uncleared: dict[str, bool] = {}
        # Spotter v1 (design §2): an independent monitor that watches this
        # session turn by turn. Agent-owned — NOT transport-owned (kdsn.276
        # lesson: the transport must never own harness state) — so every real
        # Agent carries a SpotterManager and the /spotter operator API, the
        # turn-completion fire seam, and the loop-top drain all work from the
        # agent side on every seam (matrix, heartbeat, umbral, CLI). The
        # defensive getattr guards at the agent seams cover mock /
        # spotter-less agents (the _spotter / _spotter_inbox attributes may be
        # absent or deleted).
        self._spotter = SpotterManager(self.config, self)
        self._spotter_inbox: dict[str, list] = {}
        # kdsn.330 (async subagents, D4): per-room dispatch ledger —
        # room_id -> {dispatch_id -> subledger.DispatchRecord}. Agent-owned
        # (the transport must never own harness state) and CLEARED in
        # reset_room, so a post-umbral room starts dispatch-free. The
        # background dispatch tasks live in _dispatch_tasks (NOT
        # _current_tasks — that holds exactly ONE turn task per room, so
        # background work is invisible to the legacy cancel path; D7
        # reaches them from the ledger).
        self._dispatch_ledger: dict[str, dict] = {}
        self._dispatch_tasks: dict[str, dict[str, asyncio.Task]] = {}
        # kdsn.330 R7: per-room pending terminal-event inbox — finalizers
        # deposit terminal event dicts here; the tool-loop-top drain /
        # idle-room completion turn consumes them.
        self._pending_events: dict[str, list[dict]] = {}
        # kdsn.330 R9: keepalives RETAINED past the batch gather for rooms
        # with running async dispatches — room_id -> (task, stop). The
        # gather-finally stores it, the LAST async terminal
        # (_finalize_dispatch) disarms it, and reset_room disarms it
        # sync-safely. Sync-only batches never reach the store: they
        # disarm at gather end exactly as pre-kdsn.330.
        self._room_keepalive: dict[str, tuple[asyncio.Task, asyncio.Event]] = {}
        # kdsn.330 R10: per-room quiet flag (D8) — READ here by the
        # delivery fire/notice guards (delivery callback + drain
        # notice). The operator-facing setter and the quiet_override
        # persistence / _activate_room restore are phase 2e. It JOINS
        # _room_models / _room_locks in reset_room's preserved list
        # (reset_room never pops it) so an umbral rotation can never
        # silently un-quiet a room.
        self._room_quiet: dict[str, bool] = {}

    async def _vision_deposit(self, room_id: str, tag: str) -> None:
        """Append a staged [media:] tag to this room's vision inbox (kdsn.279).

        Wired as the DEFAULT ``vision_deposit`` tool callback at dispatch
        (setdefault — an explicit callback in the caller's dict wins)."""
        self._vision_inbox.setdefault(room_id, []).append(tag)

    def _engine_for(self, room_id: str) -> ReminderEngine:
        """Lazily create and return a per-room ReminderEngine (R1-4)."""
        if room_id not in self._reminder_engines:
            self._reminder_engines[room_id] = ReminderEngine(self.config)
        return self._reminder_engines[room_id]

    def _orient_inputs(self, room_id: str) -> dict:
        """session-orient ReminderState inputs for the turn-start site (kdsn.298).

        Fail-soft everywhere: a stale persisted /model override pointing at an
        unconfigured provider raises in resolve_model — caught, and the RAW
        get_model string is used (review LOW-2: under fail-soft the compared
        keyed form flips to raw; self-healing). Must never raise into a turn.
        model_resolved uses the RESOLVED provider's key (audit reconciliation
        HIGH-1, kimi3+qwen38): reconstructing the prefix from the input string
        mis-keyed bare aliases ("/model deepseek" → "deepseek/deepseek-v4-flash"
        instead of "macstudio/deepseek-v4-flash"), making the model-keyed cap
        spelling-sensitive and the nudge's model line wrong.
        """
        model = self.get_model(room_id)
        try:
            from openalph.config import resolve_model
            provider_cfg, api_name = resolve_model(
                model, self.config.providers,
                aliases=self.config.model_aliases)
            model_resolved = f"{provider_cfg.key}/{api_name}"
        except Exception as exc:
            logger.warning("session-orient: model resolution failed for %r "
                           "in %s — using raw string (%s: %s)",
                           model, room_id, type(exc).__name__, exc)
            model_resolved = model
        # kdsn.333: the epoch record (last applied boundary) is READ here —
        # persist-on-read, defaults 0/"" when absent (fresh epoch). A
        # mid-epoch model switch refires session-orient with the same Epoch
        # label; the record lives until the next boundary or reset_room.
        # The manifest ts ("YYYY-MM-DDTHH:MM:SSZ", UTC) renders host-local,
        # same shape as orient_ts. An unrenderable ts drops the whole epoch
        # pair (never a blank "- Epoch began:  (...)" row — audit L1/L3).
        _epoch = self._handoff_epoch.get(room_id)
        _epoch_index, _epoch_ts_raw = (0, "") if not _epoch else _epoch
        _epoch_ts = ""
        if _epoch_index and _epoch_ts_raw:
            try:
                _epoch_ts = (
                    datetime.strptime(_epoch_ts_raw, "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc)
                    .astimezone()
                    .strftime("%A, %B %d, %Y — %H:%M %Z")
                )
            except (TypeError, ValueError):
                _epoch_index, _epoch_ts = 0, ""
        return {
            "model_resolved": model_resolved,
            "model_vision": model_supports_vision(model, self.config),
            "orient_ts": datetime.now(timezone.utc).astimezone().strftime(
                "%A, %B %d, %Y — %H:%M %Z"),
            "handoff_epoch_index": _epoch_index,
            "handoff_epoch_ts": _epoch_ts,
        }

    def rehydrate_reminders(self, room_id: str, entries: list[dict]) -> None:
        """Rehydrate reminder fired-state for a room from JSONL entries (R1-4)."""
        self._engine_for(room_id).rehydrate(entries)

    def reset_room(self, room_id: str) -> None:
        """Clear in-memory history for a room.

        Preserves model overrides (_room_models) and room locks (_room_locks).
        Called by umbral after archive+wipe.
        """
        if room_id in self._rooms:
            self._rooms[room_id].clear()
        self._handoff_epoch.pop(room_id, None)
        self._room_usage.pop(room_id, None)
        self._last_turn_usage.pop(room_id, None)
        self._last_stop_reason.pop(room_id, None)
        self._room_tool_counts.pop(room_id, None)
        # R1-4: reset+drop only this room's engine (not all rooms)
        if room_id in self._reminder_engines:
            self._reminder_engines[room_id].reset()
            del self._reminder_engines[room_id]
        # R1-1: clear this room's read registry
        self._read_registries.pop(room_id, None)
        self._advisor_uses.pop(room_id, None)
        # kdsn.279: drop staged view_image tags too — a reset room must not
        # resurrect stale tags into the post-rotation context.
        self._vision_inbox.pop(room_id, None)
        # Spotter v1 (design §8): a reset room starts a FRESH watch — cancel the
        # in-flight watch task, drop state + inbox, archive+truncate the
        # spotters transcript/ledger. Defensive: agents without a wired
        # SpotterManager skip this (the inbox may still be present as a
        # transport-wired empty dict).
        _spotter = getattr(self, "_spotter", None)
        if _spotter is not None:
            try:
                _spotter.reset_room(room_id)
            except Exception:
                logger.warning("spotter: reset_room failed in %s (fail-soft)",
                               room_id, exc_info=True)
        _spotter_inbox = getattr(self, "_spotter_inbox", None)
        if isinstance(_spotter_inbox, dict):
            _spotter_inbox.pop(room_id, None)
        # kdsn.305: a reset room starts with no boundary history of its own.
        self._gc_fail_strikes.pop(room_id, None)
        self._gc_last_runway.pop(room_id, None)
        self._gc_auto_uncleared.pop(room_id, None)
        # kdsn.329: wholesale history reset — the anchor's cursor no longer
        # prices anything in this room.
        self._token_anchor.pop(room_id, None)
        # kdsn.330 D4: a reset room starts dispatch-free — the per-room
        # ledger, its dispatch task slots, and the pending terminal-event
        # inbox are wiped with the history (post-umbral / post-restart
        # rooms must not resurrect stale dispatches).
        self._dispatch_ledger.pop(room_id, None)
        self._dispatch_tasks.pop(room_id, None)
        self._pending_events.pop(room_id, None)
        # R9: a keepalive retained for this room's async dispatches dies
        # with the room. reset_room is SYNC — stop + cancel is safe here
        # (the cancelled loop task completes on the next loop iteration);
        # the storage is popped so no later terminal can re-disarm a
        # dead entry or double-cancel.
        _ka_retained = self._room_keepalive.pop(room_id, None)
        if _ka_retained is not None:
            _ka_retained[1].set()
            _ka_retained[0].cancel()

    # --- Context GC boundary consumption (workspace-kdsn.305) --------------
    # The agent owns the CHECKS (auto tier at turn start, hard tier at the
    # send-time overflow guard); the transport (MatrixBot) owns the WIRING —
    # it injects callbacks["apply_handoff_boundary"] into the per-turn callback
    # dict. Every turn path (interactive/heartbeat/umbral/CLI) flows through
    # handle_input, so the hooks below cover all seams; headless/CLI turns
    # simply never carry the callback and skip silently (fail-safe).

    # Consecutive failed hard-tier boundary attempts before we stop trying
    # and raise ContextOverflowError as pre-GC (3-strike breaker: a wedged
    # boundary writer must not turn the overflow guard into an infinite loop).
    _GC_HARD_STRIKE_LIMIT = 3

    async def _gc_apply_boundary(self, room_id: str, callbacks: dict | None, *,
                                 trigger: str, exclude_inflight: bool) -> bool | dict:
        """Consume the transport-wired apply_handoff_boundary callback (kdsn.305).

        Returns the APPLIED outcome dict (truthy; callers may read
        ``pending_protected``), or False when the boundary did not apply
        (the callback has already
        appended the JSONL manifest + snapshot entries and rebuilt the room's
        in-memory history IN PLACE — the caller must re-read
        self.history(room_id) before re-estimating).

        Contract guards, in order:
          - callback absent (headless/CLI) -> False, silently;
          - callback raises -> log a warning, return False (a failed GC must
            NEVER kill a turn — the turn proceeds exactly as if GC were off);
          - applied=True -> record the outcome via
            _note_handoff_boundary_applied
            (the single applied-boundary bookkeeping seam, shared with the
            matrix /cache gc operator path: ReminderEngine reset, hard-tier
            strikes to 0, runway-fraction cache refresh, churn-guard re-arm).
          - applied=False (noop) -> strikes are the CALLER's concern (hard
            tier only); auto tier simply skips.
        """
        cb = (callbacks or {}).get("apply_handoff_boundary")
        if cb is None:
            return False
        try:
            res = await cb(room_id, trigger=trigger,
                           exclude_inflight=exclude_inflight)
        except Exception:
            logger.warning(
                "gc boundary callback failed in %s (trigger=%s) — continuing "
                "without a boundary (fail-soft; a GC failure never kills a "
                "turn)", room_id, trigger, exc_info=True)
            return False
        if not (isinstance(res, dict) and res.get("applied")):
            return False
        # APPLIED: the callback has already appended the JSONL entries and
        # rebuilt the in-memory history in place.
        self._note_handoff_boundary_applied(room_id, res)
        return res

    def _note_handoff_boundary_applied(self, room_id: str, res: dict) -> None:
        """Record the APPLIED-boundary consequences on this agent (kdsn.305.12).

        THE single place that consumes an applied boundary's outcome
        (``apply_boundary_and_rebuild`` result: {"applied", "noop_reason",
        "manifest", "over_budget", "handoff_advised", "forced_handoff"}) for
        the agent's per-room GC bookkeeping — four state updates, one seam:
          - per-room ReminderEngine reset (coexist-with-reset ruling: the
            nudge ladder re-arms after a boundary);
          - hard-tier fail-strikes reset to 0 (ANY applied boundary clears
            them);
          - runway-fraction cache refresh (tokens_after / available from the
            manifest "runway" block) — feeds the handoff-runway reminder;
          - auto-tier churn-guard re-arm (a boundary never proven
            unproductive re-arms the auto tier).
        Called from BOTH boundary-consumption seams: the callback consumer
        (``_gc_apply_boundary`` — auto/hard tiers + context_handoff tool) and
        the matrix ``/cache handoff`` operator path (which applies the
        boundary directly and must not leave the runway cache stale).
        Fail-soft: a malformed outcome must never break the caller.
        """
        self._engine_for(room_id).reset()
        self._gc_fail_strikes[room_id] = 0
        self._gc_last_runway[room_id] = self._handoff_runway_fraction(res.get("manifest"))
        # kdsn.333: record this boundary as the room's epoch start. The
        # record PERSISTS for the epoch's lifetime — every populated
        # turn-start reads it via _orient_inputs (overwritten by the next
        # applied boundary, cleared by reset_room). A malformed outcome must
        # never break the seam.
        _manifest = res.get("manifest") if isinstance(res, dict) else None
        if isinstance(_manifest, dict):
            _idx = _manifest.get("boundary_index")
            _ts = _manifest.get("ts")
            # bool is an int subclass — a bool "index" would render as
            # "boundary True" (audit L2: same guard class as _set_anchor_floor).
            if (isinstance(_idx, int) and not isinstance(_idx, bool)
                    and isinstance(_ts, str) and _ts):
                self._handoff_epoch[room_id] = (_idx, _ts)
        # An APPLIED boundary re-arms the auto tier ONLY if it cleared the
        # auto threshold (audit kdsn.322.9: the unconditional pop let a
        # mid-turn hard/tool/slash boundary re-arm a churn-latched auto tier
        # whose wedge state persists — one boundary + snapshot per turn,
        # unbounded JSONL growth). Same arithmetic as the auto tier:
        # post-boundary inclusive estimate < auto threshold. Fail-soft — on
        # any estimation failure the pre-existing latch state stands.
        # kdsn.329: an APPLIED boundary rebuilt history wholesale — the
        # anchor's char cursor no longer prices the surviving payload. Pop
        # BEFORE the churn re-arm estimate below so the re-arm prices the
        # rebuilt history with a clean heuristic (a stale cursor must never
        # survive a rebuild).
        self._token_anchor.pop(room_id, None)
        try:
            _limit = self._resolve_model_limit(room_id)
            _available = self._effective_available(_limit)
            _threshold = int(_available
                             * self.config.context.auto_pct / 100)
            if self._estimate_context_tokens(room_id) < _threshold:
                self._gc_auto_uncleared.pop(room_id, None)
        except Exception as e:  # fail-soft: the latch state stands
            logger.debug("churn-guard re-arm check failed for %s: %s",
                         room_id, e)

    def _handoff_runway_fraction(self, manifest: object) -> float:
        """Post-boundary runway consumption fraction (kdsn.305.12 D5).

        tokens_after / available from the manifest's "runway" block — the
        same composite the boundary applier used to gate the handoff.
        Fail-soft: a missing/corrupt runway block, non-numeric values, or a
        zero-or-negative available all yield 0.0 (handoff-runway stays silent —
        never false urgency from a corrupt manifest).
        """
        if not isinstance(manifest, dict):
            return 0.0
        runway = manifest.get("runway")
        if not isinstance(runway, dict):
            return 0.0
        tokens_after = runway.get("tokens_after")
        available = runway.get("available")
        if (isinstance(tokens_after, bool) or isinstance(available, bool)
                or not isinstance(tokens_after, (int, float))
                or not isinstance(available, (int, float))):
            return 0.0
        if available <= 0:
            return 0.0
        return tokens_after / available

    def _gc_runway_cached(self, room_id: str) -> float:
        """Last applied boundary's runway fraction (0.0 pre-boundary/restart)."""
        return self._gc_last_runway.get(room_id, 0.0)

    def _boundary_reminder_state(self, room_id, *, iteration, context_tokens,
                                 limit, completed_turns, turn_source,
                                 tool_calls_this_turn, tool_calls_session,
                                 todo_list, enabled_tools):
        """ReminderState for the tool-loop-boundary evaluation point.

        audit-fix (kdsn.322.9, spec §3.3): the handoff checkpoint evaluates
        at BOTH turn_start and tool_loop_boundary. This site must pass the
        SAME checkpoint_threshold the turn-start site computes
        (``checkpoint_pct`` of the usable runway, 0 when handoff is
        disabled) — the predicate is gated on ``checkpoint_threshold > 0``
        in reminders.py, so the dataclass default 0 keeps the trigger
        silent mid-turn and defeats the ratified dual-point evaluation.
        """
        _gc_cfg = self.config.context
        available = self._effective_available(limit)
        return ReminderState(
            evaluation_point="tool_loop_boundary",
            iteration=iteration,
            max_iterations=self.config.max_iterations,
            # Post-tool-results, pre-call estimate — the same surface the
            # pre-call overflow guard uses.
            context_tokens=context_tokens,
            context_limit=limit,
            # D9: same expression as the overflow guard.
            available_tokens=available,
            completed_turns=completed_turns,
            turn_source=turn_source,
            tool_calls_this_turn=tool_calls_this_turn,
            tool_calls_session=tool_calls_session,
            todo_list=todo_list,
            enabled_tools=enabled_tools,
            # Mirrors the turn-start site exactly (kdsn.322.9): checkpoint
            # threshold from [context].checkpoint_pct of the usable runway
            # (0 when handoff disabled -> engine silent).
            checkpoint_threshold=(
                int(available * _gc_cfg.checkpoint_pct / 100)
                if _gc_cfg.handoff_enabled else 0),
            handoff_runway_fraction=self._gc_runway_cached(room_id),
        )

    async def _gc_hard_tier(self, room_id: str, callbacks: dict | None, *,
                            context_tokens: int, available: int) -> bool:
        """Hard tier at the send-time overflow guard (kdsn.305).

        Returns True when an applied boundary cleared the overflow (caller
        re-estimates and proceeds), False when the caller must raise
        ContextOverflowError. Never raises itself.

        Gated: handoff_enabled, callback present, and this room's consecutive
        FAILED-attempt strikes < _GC_HARD_STRIKE_LIMIT. On a failed/noop
        attempt the strike counter increments; at >= 3 attempts stop and
        raise as pre-GC (3-strike breaker). Any applied boundary resets the
        counter to 0 (in _gc_apply_boundary).
        """
        ctx = getattr(self.config, "context", None)
        if ctx is None or not ctx.handoff_enabled:
            return False
        if (callbacks or {}).get("apply_handoff_boundary") is None:
            return False
        strikes = self._gc_fail_strikes.get(room_id, 0)
        if strikes >= self._GC_HARD_STRIKE_LIMIT:
            return False
        applied = await self._gc_apply_boundary(
            room_id, callbacks, trigger="hard", exclude_inflight=False)
        if not applied:
            self._gc_fail_strikes[room_id] = strikes + 1
        return applied

    def _usage_for(self, room_id: str) -> dict[str, int]:
        """Lazily init + return the per-room counter record."""
        if room_id not in self._room_usage:
            self._room_usage[room_id] = {
                "uncached_input_tokens": 0, "cache_read_tokens": 0,
                "cache_creation_tokens": 0, "total_output_tokens": 0,
                "total_tool_calls": 0,
                "main_cost_usd": 0.0, "subagent_cost_usd": 0.0,
                "advisor_cost_usd": 0.0, "unpriced_tokens": 0,
            }
        return self._room_usage[room_id]

    def _provider_gate(self, model_str: str) -> tuple[str | None, str | None]:
        """Resolve the (provider_key, provider_type) pricing gate for `model_str`
        — the authoritative gate for USD cost pricing (F2, kdsn.218). The pair is
        passed to compute_cost, which namespaces pricing by them (anthropic by
        type, else by key). Fail-soft: an unknown model or resolution error
        returns (None, None), so the call is tallied unpriced rather than
        mispriced at another provider's rates, and never crashes."""
        try:
            from openalph.config import resolve_model
            pcfg, _ = resolve_model(model_str, self.config.providers,
                                    aliases=self.config.model_aliases)
            return getattr(pcfg, "key", None), getattr(pcfg, "type", None)
        except Exception:
            # kdsn.292: keep this blanket catch — it is INTENTIONAL. Pricing
            # gating must never crash a turn; on ANY resolution failure the
            # call is tallied unpriced (None, None) instead of mispriced or fatal.
            return None, None

    def _record_turn_usage(self, room_id: str, usage, model: str, cache_ttl: str | None,
                           provider_key: str | None = None,
                           provider_type: str | None = None,
                           pre_call_chars: int | None = None) -> None:
        """Called once per API call (text + tool turns + summary). Updates globals
        AND per-room token counters, and stores the per-turn delta for the serializer.
        `usage` is a provider Usage object (.input_tokens, .output_tokens,
        .cache_read_tokens, .cache_creation_tokens; cache fields may be None).
        `model` is the resolved/authoritative model string for this call (used to
        freeze cost); `cache_ttl` is the room's cache TTL label, used only as the
        aggregate-cache-write fallback multiplier inside compute_cost.
        `provider_key`/`provider_type` gate pricing by provider (F2).
        `pre_call_chars` (kdsn.329): the char total of the EXACT payload the
        provider just measured (captured at the pre-call guard's walk). When
        given AND the usage carries prompt tokens, this call anchors the
        room's estimate to the true prompt size (input + cache_read +
        cache_creation). Continuation/summary call sites omit it → they never
        update the anchor (we never anchor without the matching payload
        cursor)."""
        cr = usage.cache_read_tokens or 0
        cc = usage.cache_creation_tokens or 0
        # globals (unchanged semantics)
        self.uncached_input_tokens += usage.input_tokens
        self.cache_read_tokens += cr
        self.cache_creation_tokens += cc
        self.total_output_tokens += usage.output_tokens
        # per-room mirror
        r = self._usage_for(room_id)
        r["uncached_input_tokens"] += usage.input_tokens
        r["cache_read_tokens"] += cr
        r["cache_creation_tokens"] += cc
        r["total_output_tokens"] += usage.output_tokens
        # frozen cost for THIS call. F3 (kdsn.218): fail-soft — a bad model
        # string or usage shape must never raise through the hot turn loop;
        # on failure record $0 and move on.
        try:
            cr_result = compute_cost(model, usage, cache_ttl_fallback=cache_ttl or "1h",
                                     provider_key=provider_key,
                                     provider_type=provider_type)
            _cost_usd = cr_result.cost_usd
            _unpriced = cr_result.unpriced_tokens
        except Exception:
            logger.warning("cost: compute_cost failed for model %r; recording $0", model)
            _cost_usd = 0.0
            _unpriced = 0
        r["main_cost_usd"] += _cost_usd
        r["unpriced_tokens"] += _unpriced
        # per-turn delta (serializer persists this; tool_calls added by serializer)
        self._last_turn_usage[room_id] = {
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "cache_read_tokens": cr, "cache_creation_tokens": cc,
            "cost_usd": _cost_usd, "model": model,
            "cache_creation_5m": usage.cache_creation_5m_tokens or 0,
            "cache_creation_1h": usage.cache_creation_1h_tokens or 0,
            "unpriced_tokens": _unpriced,
        }
        # kdsn.329: live anchor set — the provider just measured this exact
        # payload's true prompt size. Uniform formula for both provider
        # types (openai puts cached in cache_read; anthropic splits
        # cache_read/cache_creation): true_prompt = input + cr + cc.
        # Zero/no usage must NOT trample an existing anchor.
        if pre_call_chars is not None:
            _true_prompt = (usage.input_tokens or 0) + cr + cc
            if _true_prompt > 0:
                self._token_anchor[room_id] = (_true_prompt, pre_call_chars)

    def _record_tool_calls(self, room_id: str, n: int) -> None:
        self.total_tool_calls += n
        self._usage_for(room_id)["total_tool_calls"] += n

    def last_turn_usage(self, room_id: str) -> dict | None:
        """Public accessor for the matrix serializer."""
        return self._last_turn_usage.get(room_id)

    def last_stop_reason(self, room_id: str) -> str | None:
        """Return the stop_reason from the last completed turn in this room."""
        return self._last_stop_reason.get(room_id)

    def restore_usage(self, room_id: str, totals: dict) -> None:
        """Set the per-room counters from JSONL-summed totals (rehydration).

        Float-safe: cost counters are dollars-and-cents floats, NOT int-cast
        (a blanket int() cast would truncate them to whole dollars).
        Fail-soft (F3, kdsn.218): a malformed persisted value coerces to 0
        per-key rather than raising through _activate_room and bricking the
        room's wake."""
        def _int(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0
        def _float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0
        r = self._usage_for(room_id)
        for k in ("uncached_input_tokens", "cache_read_tokens", "cache_creation_tokens",
                  "total_output_tokens", "total_tool_calls"):
            r[k] = _int(totals.get(k, 0))
        for k in ("main_cost_usd", "subagent_cost_usd", "advisor_cost_usd"):
            r[k] = _float(totals.get(k, 0.0))
        r["unpriced_tokens"] = _int(totals.get("unpriced_tokens", 0))

    def history(self, room_id: str) -> list[dict]:
        """Get or create history for a room."""
        if room_id not in self._rooms:
            self._rooms[room_id] = []
        return self._rooms[room_id]

    def get_model(self, room_id: str = "_default") -> str:
        """Return the active model for a room, falling back to config default."""
        return self._room_models.get(room_id, self.config.default_model)

    def _fire_spotter_turn_completion(self, room_id: str, history: list[dict],
                                      turn_source: str | None,
                                      callbacks: dict | None) -> None:
        """Spotter v1 turn-completion seam (design §3): fire a watch pass over
        the room's newly completed turn.

        Defensive + fail-soft: agents without a wired SpotterManager (mock /
        spotter-less agents — a deleted or absent _spotter attribute) make
        this a silent no-op. The manager's
        maybe_fire is SYNC, never raises, and never mutates history (RC1
        invariant: it snapshots indices and schedules the watch task only),
        so a spotter failure can never break or delay the main turn."""
        _spotter = getattr(self, "_spotter", None)
        if _spotter is None:
            return
        try:
            _spotter.maybe_fire(room_id, history, turn_source, callbacks)
        except Exception:
            logger.warning("spotter: maybe_fire failed in %s (fail-soft)",
                           room_id, exc_info=True)

    def _resolve_model_limit_for(self, model_str: str) -> int:
        """3-layer window resolution for an explicit model string (config override
        -> curated default -> model_max_tokens fallback + warn-once)."""
        # Room model strings may be bare aliases (e.g. a persisted `/model deepseek`
        # override): expand before lookup, otherwise Layer 1 (full-string keys)
        # and Layer 2 (substring fragments) both silently miss and the model
        # falls through to the model_max_tokens fallback (found 2026-08-03:
        # wonmun room ran deepseek-v4-flash at a 262K window instead of 1M).
        if model_str in self.config.model_aliases:
            model_str = self.config.model_aliases[model_str]
        if model_str in self.config.model_limits:
            return self.config.model_limits[model_str]
        from openalph.provider import model_context_window
        w = model_context_window(model_str)
        if w is not None:
            return w
        if model_str not in self._warned_models:
            self._warned_models.add(model_str)
            logger.warning("No curated context window for model %r; falling back to "
                           "model_max_tokens=%d. Add a [model_limits] override if its "
                           "window differs.", model_str, self.config.model_max_tokens)
        return self.config.model_max_tokens

    def _resolve_model_limit(self, room_id: str = "_default") -> int:
        """True context window for the room's active model.

        Layer 1: [model_limits] config override.
        Layer 2: curated code default (model_context_window).
        Layer 3: fallback to config.model_max_tokens (+ one-time WARN log).
        Delegates to _resolve_model_limit_for with the room's active model.
        """
        return self._resolve_model_limit_for(self.get_model(room_id))

    def _effective_available(self, limit: int) -> int:
        """Usable context runway: window limit minus the output reserve.

        The single expression the ContextOverflowError guards AND both
        ReminderState constructions use (context-nudge-ladder D9:
        single-source `available` — when the guard's formula changes
        (im7t.47 / kdsn.258), the reminder ladder follows automatically).
        """
        return limit - self.config.max_tokens

    def switch_model(self, model_str: str, room_id: str = "_default") -> str | None:
        """Switch active model. Returns error string on failure, None on success."""
        from openalph.provider import ProviderUnavailableError, resolve_model_checked

        # Validate the model can be resolved. kdsn.292: an explicit /model at a
        # dead provider must be LOUD — ProviderUnavailableError's message names
        # the provider AND the startup skip reason; the room model is left
        # UNCHANGED (the assignment below is never reached). Alias-typo
        # ValueErrors keep their original listing message.
        try:
            resolve_model_checked(
                model_str, self.config.providers,
                aliases=self.config.model_aliases,
                skipped_providers=getattr(self.config, "skipped_providers", {}),
            )
        except ProviderUnavailableError as e:
            return str(e)
        except ValueError as e:
            return str(e)

        # Vision guard: if session has images, block switch to non-vision model
        history = self.history(room_id)
        has_images = any(
            isinstance(msg.get("content"), list) and
            any(block.get("type") == "image" for block in msg.get("content", []))
            for msg in history
        )
        if has_images and not model_supports_vision(model_str, self.config):
            return f"Cannot switch to {model_str} — session contains images and model may not support vision."

        # Context window guard: check current context vs model limit.
        # kdsn.329: the estimate here is ANCHOR-AWARE on purpose — the
        # provider-reported true prompt size is exactly the ground truth a
        # too-small-target-window check should consume (audit F2). The pop
        # waits until the switch actually COMMITS below: a switch refused by
        # this guard must not destroy the old model's anchor while we stay
        # on it (spec; audit F2/H3). Pre-validation failures (bad model,
        # vision refusal) return above and retain the anchor.
        model_limit = self._resolve_model_limit_for(model_str)
        context_tokens = self._estimate_context_tokens(room_id)
        if context_tokens > model_limit - self.config.max_tokens:
            return f"Cannot switch to {model_str} — current context (~{context_tokens:,} tokens) exceeds model limit ({model_limit:,})."

        # Committed switch: tokenizer semantics change — the anchor's ground
        # truth (measured under the OLD model's tokenizer) no longer prices
        # this room's context.
        self._token_anchor.pop(room_id, None)
        self._room_models[room_id] = model_str
        return None


    def _log_turn(
        self,
        room_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tool_calls: list[dict] | None,
        latency_ms: float,
        content_preview: str,
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
        stop_reason: str = "",
    ) -> None:
        """Write a JSONL entry for this LLM turn.

        Failures are logged but do not crash the agent.
        """
        try:
            log_dir = Path(self.config.workspace) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = log_dir / f"{self.config.name}-{date_str}.jsonl"

            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "room_id": room_id,
                "direction": "outbound",
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_creation_tokens": cache_creation_tokens,
                "tool_calls": tool_calls or [],
                "latency_ms": latency_ms,
                "stop_reason": stop_reason,
                "content_preview": content_preview[:200] if content_preview else "",
            }

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write JSONL log: {e}")

    async def _maybe_arm_cache_keepalive(self, *, room_id, active_tool_calls,
                                         request_messages, tools_arg, cache_ttl, callbacks,
                                         thinking, stream_start=None):
        """Arm a background cache-keepalive task iff a subagent is in the batch AND
        the active model is an Anthropic provider with subagent_cache_keepalive on.
        Returns (task, stop_event) or (None, None). Zero overhead in the default path.
        One task covers the parent's single cache prefix even for N parallel subagents.

        kdsn.330: async so the keepalive-lifecycle recorder seam can wrap it
        (``await original(...)``); the body is unchanged — the arm decision
        and task creation are the same, only the call site now awaits.
        """
        if not any(getattr(tc, "name", None) == "subagent" for tc in active_tool_calls):
            return None, None
        try:
            from openalph.provider import resolve_model_checked
            provider_cfg, _ = resolve_model_checked(
                self.get_model(room_id), self.config.providers,
                aliases=self.config.model_aliases,
                skipped_providers=getattr(self.config, "skipped_providers", {}),
            )
        except Exception:
            return None, None
        if provider_cfg.type != "anthropic" or not getattr(
            provider_cfg, "subagent_cache_keepalive", False
        ):
            return None, None
        stop = asyncio.Event()
        on_miss = (callbacks or {}).get("on_keepalive_miss")
        elapsed = (time.monotonic() - stream_start) if stream_start is not None else 0.0
        logger.info(
            "cache keepalive armed in %s (interval=%.0fs, stream_elapsed=%.1fs)",
            room_id, _keepalive_interval(cache_ttl), elapsed,
        )
        task = asyncio.create_task(self._cache_keepalive(
            system=self.system_prompt,
            messages=request_messages,
            tools=tools_arg,
            cache_ttl=cache_ttl,
            model=self.get_model(room_id),
            thinking=thinking,
            room_id=room_id,
            on_miss=on_miss,
            stop=stop,
            stream_elapsed=elapsed,
        ))
        return task, stop

    async def _cache_keepalive(self, *, system, messages, tools, cache_ttl, model,
                               thinking, room_id, on_miss, stop, stream_elapsed=0.0):
        """Periodically refresh the parent's Anthropic prompt cache while a subagent
        runs. Fires a cheap max_tokens=1 ping just inside the TTL; verifies each ping
        was a cache READ and aborts (with an operator notice) on a WRITE signature so
        a drifted/expired prefix cannot turn insurance into a cost bomb.

        The cache TTL starts at the *request* (stream start), not at arm time, so the
        first fire is shortened by the stream duration already elapsed. Transient ping
        errors get quick retries; KEEPALIVE_MAX_CONSECUTIVE_ERRORS in a row aborts with
        an operator notice instead of retrying blindly for the whole subagent run.
        """
        interval = _keepalive_interval(cache_ttl)
        # First fire accounts for time already consumed by the (possibly slow) stream.
        next_wait = max(interval - max(stream_elapsed, 0.0), 0.0)
        errors = 0
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=next_wait)
                return  # stop set (gather completed / cancelled) -> clean exit
            except asyncio.TimeoutError:
                pass  # time to refresh
            next_wait = interval  # subsequent fires use the full interval unless shortened below
            try:
                usage = await ping_cache(
                    self.config, system=system, messages=messages,
                    tools=tools, cache_ttl=cache_ttl, model=model,
                    thinking_level=thinking,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # A ping failure must NEVER crash the parent turn.
                errors += 1
                logger.warning("cache keepalive ping error #%d in %s: %s", errors, room_id, e)
                if errors >= KEEPALIVE_MAX_CONSECUTIVE_ERRORS:
                    logger.warning(
                        "cache keepalive aborting after %d consecutive errors in %s",
                        errors, room_id,
                    )
                    await self._keepalive_notify_miss(on_miss, room_id)
                    return
                next_wait = min(KEEPALIVE_ERROR_BACKOFF_S, interval)  # quick retry, not a full TTL window
                continue
            errors = 0  # a completed ping (hit or miss) clears the error streak
            if usage is None:
                return  # non-Anthropic (gated upstream) -> nothing to do
            read = usage.cache_read_tokens or 0
            write = usage.cache_creation_tokens or 0
            if _keepalive_is_hit(read, write):
                logger.info("cache keepalive hit in %s (read=%d write=%d)", room_id, read, write)
                continue
            logger.warning(
                "cache keepalive MISS in %s (read=%d write=%d) -- disabling for this turn",
                room_id, read, write,
            )
            await self._keepalive_notify_miss(on_miss, room_id)
            return  # never ping into a miss

    async def _keepalive_notify_miss(self, on_miss, room_id):
        """Fire the operator miss-notice callback, isolated from its own failures."""
        if on_miss is not None:
            try:
                await on_miss(room_id)
            except Exception:
                logger.error("cache keepalive miss notice failed in %s", room_id, exc_info=True)

    # --- kdsn.330: async (background) subagent dispatch --------------------

    async def dispatch_background_subagent(
        self, *, room_id, dispatch_id, task, model=None, effort=None,
        system_prompt=None, max_tokens=None, max_iterations=None,
        tools=None, tool_executors=None, sub_callbacks=None, session_log=None,
    ):
        """Record a background subagent dispatch and spawn its runner task.

        Returns the immediate receipt ToolResult (dispatch ID + pointer at
        the ``subagent_status`` retrieval surface) — the sub itself runs as
        a background task, so the caller's batch gather completes at
        receipt time and the turn never blocks on sub duration (D1/D9).

        Ordering (R5, persist-before-notify): the ledger record is created
        and the ``subagent_dispatched`` system entry is durably written
        BEFORE the receipt returns.
        """
        if not dispatch_id:
            return ToolResult(
                is_error=True,
                content="subagent background dispatch refused: the harness "
                        "threaded no dispatch id (call_id).",
            )
        # The record's model is the resolved one (R4: model provenance);
        # the sub RUN receives the raw input model (None -> the sub's own
        # default resolution), byte-identical to the sync path.
        rec_model = model or self.get_model(room_id)
        rec = subledger.DispatchRecord(
            dispatch_id=dispatch_id,
            task=task,
            task_head=subledger.make_task_head(task),
            model=rec_model,
            effort=effort,
            dispatched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        self._dispatch_ledger.setdefault(room_id, {})[dispatch_id] = rec

        # R5: the dispatch entry is durable before the receipt returns.
        # Fail-soft: a ledger-write failure must never block the dispatch
        # (the record is live in memory); it is logged loudly.
        if session_log is not None:
            try:
                session_log.append(
                    role="system",
                    sender=session_log.agent_user_id,
                    room=room_id,
                    event_id=None,
                    event="subagent_dispatched",
                    detail=subledger.dispatched_detail(rec),
                )
            except Exception:
                logger.error(
                    "subagent dispatch: subagent_dispatched entry failed "
                    "for %s in %s", dispatch_id, room_id, exc_info=True)

        runner = asyncio.create_task(
            self._run_background_dispatch(
                room_id, dispatch_id,
                _tools_subagent.run_subagent(
                    task=task,
                    config=self.config,
                    tools=tools,
                    system_prompt=system_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    max_iterations=max_iterations,
                    call_id=dispatch_id,
                    parent_room_id=room_id,
                    callbacks=sub_callbacks,
                    effort=effort,
                    tool_executors=tool_executors,
                ),
                sub_callbacks=sub_callbacks,
                session_log=session_log,
            ),
            name=f"subagent-dispatch-{dispatch_id}",
        )
        self._dispatch_tasks.setdefault(room_id, {})[dispatch_id] = runner

        # One loop yield so the scheduled runner reaches its first await
        # (the sub's provider I/O) before the receipt returns: an
        # immediate spawn failure is already being handled by the runner
        # (terminal `failed`) rather than silently parked, and the
        # receipt's observable ledger state is the dispatch the runner
        # will report on.
        await asyncio.sleep(0)

        return ToolResult(
            content=(
                f"Background subagent dispatched: {dispatch_id}\n"
                f"Task: {rec.task_head}\n"
                f"Status: running — check subagent_status(action=\"status\","
                f" id=\"{dispatch_id}\") or retrieve it; the completion "
                f"report is delivered automatically when it lands."
            ),
            is_error=False,
        )

    async def _run_background_dispatch(self, room_id, dispatch_id, coro, *,
                                       sub_callbacks=None, session_log=None):
        """Background runner for one async dispatch: run the sub to
        terminal, then finalize the ledger record (D1/D5).

        - CancelledError (operator /stop, shutdown, umbral pre-wipe) is
          terminal ``cancelled``: the record is finalized, THEN the
          cancellation re-raises so the task ends cancelled (the harness
          task graph stays truthful).
        - Exceptions escaping the sub and error ToolResults are terminal
          ``failed`` with a SANITIZED error (never a raw exception repr,
          never reminder-tag or credential text) — a failed sub is never
          silent.
        """
        try:
            result = await coro
        except asyncio.CancelledError:
            await self._finalize_dispatch(
                room_id, dispatch_id, subledger.CANCELLED,
                sub_callbacks=sub_callbacks, session_log=session_log,
                error="cancelled (operator /stop or room lifecycle)")
            raise
        except Exception as exc:
            err = subledger.sanitize_dispatch_error(exc)
            await self._finalize_dispatch(
                room_id, dispatch_id, subledger.FAILED,
                sub_callbacks=sub_callbacks, session_log=session_log,
                error=err)
            return
        if result.is_error:
            # run_subagent wraps its own exceptions as an error
            # ToolResult ("Sub-agent error: …"); sanitize that content
            # for the durable ledger exactly like the exception path.
            await self._finalize_dispatch(
                room_id, dispatch_id, subledger.FAILED,
                sub_callbacks=sub_callbacks, session_log=session_log,
                error=subledger.sanitize_dispatch_error(result.content))
            return
        await self._finalize_dispatch(
            room_id, dispatch_id, subledger.COMPLETED,
            sub_callbacks=sub_callbacks, session_log=session_log,
            result=result.content)

    async def _finalize_dispatch(self, room_id, dispatch_id, state, *,
                                 sub_callbacks=None, session_log=None,
                                 result=None, error=None):
        """Terminal handling for one async dispatch (R5/R9/R15 ordering).

        Sequence: bridge pop (usage substrate) -> state transition
        (terminal-state protected) -> ``subagent_terminal`` system entry
        persisted BEFORE any notify -> live accrual into the room usage
        counters -> terminal event deposited in the room's pending-event
        inbox -> fail-soft ``subagent_terminal_notify`` -> keepalive
        disarm iff no running dispatches remain in the room.

        A missing record (post-reset room) skips the persist/deposit but
        still runs the keepalive re-gate, so a wiped room cannot pin a
        keepalive alive.
        """
        # kdsn.330 re-audit: a CancelledError injected at the notify await
        # below (operator `subagent_status cancel` / `/stop` on a runner
        # that has already reached its finalizer) is remembered here and
        # re-raised AFTER the safe cleanup at the end of this method —
        # the runner task must end cancelled (cancellation semantics hold),
        # but persistence has already happened and the keepalive re-gate
        # below must still run (a cancelled finalize must not leave a
        # keepalive pinned).
        _swallowed_cancel = None

        # Task-slot cleanup on EVERY path (incl. missing record): the
        # runner task must not linger in _dispatch_tasks.
        tasks_by_room = self._dispatch_tasks.get(room_id)
        if tasks_by_room is not None:
            tasks_by_room.pop(dispatch_id, None)

        room_ledger = self._dispatch_ledger.get(room_id)
        rec = room_ledger.get(dispatch_id) if room_ledger is not None else None
        if rec is None:
            logger.info(
                "subagent finalize: no ledger record for %s in %s — "
                "skipping persist (post-reset room)", dispatch_id, room_id)
        else:
            # kdsn.218 bridge (written by run_subagent at terminal): the
            # usage/cost substrate. One pop feeds the ledger entry and
            # the live accrual below (mirrors the sync consumer).
            usage = {}
            try:
                bridge = (sub_callbacks or {}).get("subagent_results")
                if isinstance(bridge, dict):
                    bridge = bridge.pop((room_id, dispatch_id), None)
                if isinstance(bridge, dict):
                    usage = {
                        "input_tokens": bridge.get("input_tokens", 0) or 0,
                        "output_tokens": bridge.get("output_tokens", 0) or 0,
                        "cost_usd": bridge.get("cost_usd", 0.0) or 0.0,
                        "unpriced_tokens": bridge.get("unpriced_tokens", 0) or 0,
                    }
            except Exception:
                logger.warning(
                    "subagent finalize: bridge pop failed for %s in %s",
                    dispatch_id, room_id, exc_info=True)
            rec.usage = usage
            if result is not None:
                rec.result = result
            if error is not None:
                rec.error = error

            if not rec.transition(state):
                # Terminal-state protection (R4): a late in-flight event
                # after a terminal record is logged and ignored — no
                # second terminal entry, no double accrual.
                logger.warning(
                    "subagent finalize: late %s for terminal record %s in "
                    "%s ignored", state, dispatch_id, room_id)
            else:
                # R5: persist the terminal entry BEFORE any notify.
                # Fail-soft like every ledger write (counters + record
                # still move; a failed audit row is logged).
                if session_log is not None:
                    try:
                        session_log.append(
                            role="system",
                            sender=session_log.agent_user_id,
                            room=room_id,
                            event_id=None,
                            event="subagent_terminal",
                            detail=subledger.terminal_detail(rec),
                        )
                    except Exception:
                        logger.error(
                            "subagent finalize: subagent_terminal entry "
                            "failed for %s in %s", dispatch_id, room_id,
                            exc_info=True)
                # R15: live accrual — the async path has no tool-result
                # append site, so terminal persistence IS the accrual
                # point. Disjoint from the restart re-sum (usage_totals
                # re-reads the same entries after wipe-and-rehydrate).
                try:
                    u = self._usage_for(room_id)
                    u["subagent_cost_usd"] += usage.get("cost_usd", 0.0) or 0.0
                    u["unpriced_tokens"] += usage.get("unpriced_tokens", 0) or 0
                except Exception:
                    logger.warning(
                        "subagent finalize: live accrual failed for %s "
                        "in %s", dispatch_id, room_id, exc_info=True)
                # R7: deposit the terminal event for the delivery drain /
                # idle-room completion turn (the drain consumes
                # _pending_events).
                self._pending_events.setdefault(room_id, []).append(
                    subledger.terminal_event_dict(rec))
                # R8: fail-soft notify — the transport MAY wire a
                # subagent_terminal_notify (the idle-fire decision is
                # the transport's); its failure never blocks the ledger.
                notify = (sub_callbacks or {}).get("subagent_terminal_notify")
                if notify is not None:
                    try:
                        await notify(room_id, subledger.terminal_event_dict(rec))
                    except asyncio.CancelledError as _cancel_exc:
                        # kdsn.330 re-audit: `except Exception` does NOT
                        # catch CancelledError — an operator cancel can
                        # land its injection exactly on this await. The
                        # durable subagent_terminal entry is ALREADY
                        # persisted (R5 persist-before-notify), so complete
                        # the remaining safe cleanup (the keepalive
                        # re-gate below) and re-raise there: the task
                        # still ends cancelled, nothing is lost.
                        _swallowed_cancel = _cancel_exc
                    except Exception:
                        logger.warning(
                            "subagent finalize: terminal notify failed for "
                            "%s in %s", dispatch_id, room_id, exc_info=True)

        # R9: a keepalive retained for this room's async dispatches dies
        # when the LAST running dispatch goes terminal — any terminal
        # state (completed/failed/cancelled), on every path.
        room_ledger = self._dispatch_ledger.get(room_id)
        still_running = (
            room_ledger is not None
            and any(r.state == subledger.RUNNING for r in room_ledger.values())
        )
        if not still_running:
            await self._disarm_room_keepalive(room_id)

        if _swallowed_cancel is not None:
            raise _swallowed_cancel

    async def _disarm_room_keepalive(self, room_id):
        """Disarm the keepalive retained for this room's async dispatches.

        Idempotent: the gather-finally and the terminal handler may both
        reach the room in either order — only the STORED entry is
        disarmed, exactly once (pop)."""
        ka = self._room_keepalive.pop(room_id, None)
        if ka is None:
            return
        ka_task, ka_stop = ka
        ka_stop.set()
        ka_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ka_task
        logger.info("cache keepalive disarmed in %s (last async terminal)", room_id)

    async def subagent_status(self, action: str, room_id: str,
                              dispatch_id: str | None = None) -> ToolResult:
        """kdsn.330 R2: the companion ``subagent_status`` tool actions
        (``list`` | ``status`` | ``retrieve`` | ``cancel``) over the
        room's dispatch ledger (Agent-owned, D4).

        Truth-telling (D8): the per-room quiet flag suppresses delivery
        fires and in-room notices, NEVER this — there is deliberately no
        ``_room_quiet`` read in this method. Reset-room interplay is free:
        ``reset_room`` clears the ledger, so a post-umbral room answers
        unknown-id steering exactly like an id that never existed — no
        special casing.
        """
        room_ledger = self._dispatch_ledger.get(room_id) or {}

        if action == "list":
            if not room_ledger:
                return ToolResult(
                    content="No background subagent dispatches in this room.",
                    is_error=False)
            lines = []
            for did, rec in room_ledger.items():
                line = f"- {did}: {rec.state} — task: {rec.task_head}"
                if rec.terminal_at is not None:
                    line += f" (terminal at {rec.terminal_at})"
                lines.append(line)
            return ToolResult(content="\n".join(lines), is_error=False)

        if not dispatch_id:
            return ToolResult(
                is_error=True,
                content=f"subagent_status action {action!r} requires an id "
                        "(the dispatch id from the background dispatch "
                        "receipt).",
            )
        rec = room_ledger.get(dispatch_id)
        if rec is None:
            # Unknown id — steer, never fabricate (also the post-umbral
            # answer: the room's ledger starts dispatch-free).
            return ToolResult(
                is_error=True,
                content=f"No dispatch {dispatch_id!r} in this room's "
                        "ledger. Use action=\"list\" to see the live "
                        "dispatch ids (a room that just went through an "
                        "umbral starts dispatch-free).",
            )

        if action == "status":
            fields = [
                f"dispatch_id: {rec.dispatch_id}",
                f"state: {rec.state}",
                f"task: {rec.task_head}",
                f"model: {rec.model}",
                f"effort: {rec.effort}",
                f"dispatched_at: {rec.dispatched_at}",
                f"terminal_at: {rec.terminal_at}",
            ]
            return ToolResult(content="\n".join(fields), is_error=False)

        if action == "retrieve":
            if rec.state == subledger.RUNNING:
                return ToolResult(
                    is_error=True,
                    content=f"Dispatch {dispatch_id!r} is still running — "
                            "no report exists yet. Check it with "
                            "subagent_status(action=\"status\", "
                            f"id=\"{dispatch_id}\") or wait for the "
                            "completion notice.",
                )
            if rec.state == subledger.ORPHANED_AT_RESTART:
                return ToolResult(
                    is_error=True,
                    content=f"Dispatch {dispatch_id!r} was orphaned at "
                            "restart — killed in flight, no result exists. "
                            "Re-dispatch the task if it is still needed.",
                )
            if rec.result is not None:
                # Report-only (R11): the stored result bytes, no metadata
                # block — and this ToolResult flows through the normal
                # pipeline (credential redaction in the execute_tool tail,
                # truncate+wrap at the call site), exactly like any other
                # tool result.
                report = rec.result
                limit = getattr(self.config, "truncation_limit", None)
                if limit and len(report) > limit:
                    # R11 over-truncation: the standard house marker,
                    # re-pointed at the flight recorder — the sub's FULL
                    # output lives in sessions/subs/.
                    removed = len(report) - limit
                    marker = (f"[truncated: {removed} chars removed — "
                              "full report in the flight recorder: "
                              "sessions/subs/]")
                    head_len = max(limit - len(marker), 0)
                    report = report[:head_len] + marker
                return ToolResult(content=report, is_error=False)
            # Terminal without a report (failed / cancelled): the
            # report-only surface has nothing to retrieve — the truth is
            # the state + the sanitized error line.
            return ToolResult(
                content=f"Dispatch {dispatch_id!r} reached {rec.state} "
                        f"with no report. Error: "
                        f"{rec.error or '(none recorded)'}",
                is_error=False,
            )

        if action == "cancel":
            if rec.state != subledger.RUNNING:
                return ToolResult(
                    is_error=True,
                    content=f"Cannot cancel an already-terminal dispatch "
                            f"{dispatch_id!r} (state: {rec.state}). "
                            "Re-dispatch if the work is still needed.",
                )
            tasks_by_room = self._dispatch_tasks.get(room_id) or {}
            task = tasks_by_room.get(dispatch_id)
            if task is None or task.done():
                return ToolResult(
                    is_error=True,
                    content=f"No live background task found for dispatch "
                            f"{dispatch_id!r} — its ledger state may be "
                            "stale; use action=\"status\" to check.",
                )
            # The same primitive Agent.cancel uses for /stop (D7): the
            # runner observes the cancellation, writes the durable
            # `cancelled` terminal entry via its finalizer, and re-raises
            # so the task graph stays truthful.
            task.cancel()
            return ToolResult(
                content=f"Cancelled dispatch {dispatch_id!r} — the runner "
                        "records it as terminal `cancelled`. Confirm with "
                        "action=\"status\".",
                is_error=False,
            )

        return ToolResult(
            is_error=True,
            content=f"Invalid subagent_status action {action!r} — must be "
                    "one of: list, status, retrieve, cancel",
        )

    async def handle_input(self, text: str, room_id: str = "_default", *,
                           on_tool_call=None, on_tool_intent=None, thinking: str | None = None,
                           callbacks: dict | None = None,
                           on_text_delta=None, on_thinking_delta=None,
                           on_cache_status=None, cache_ttl: str | None = None,
                           append_user: bool = True,
                           tools: list | None = None,
                           drain_steering=None) -> str:
        """Process a user message and return the assistant's response.

        Appends the user message to room history, calls the LLM, appends the
        assistant response, and accumulates token usage. The full room history
        is passed on every call so the model has conversation context.

        Args:
            text: User message
            room_id: Room identifier for per-room history isolation
            append_user: If True (default), append the user message to history
                before calling the LLM.  Set False when the caller (e.g. the
                gated-room path in matrix.py) has already written the message
                to JSONL and hydrated history from it, so we do not duplicate
                the entry in the wire payload.
            tools: Optional per-turn tool override (kdsn.315 test seam). When
                given (including an explicit []), it replaces the tools
                discovered at init for this turn's API calls and dispatch;
                None (default) keeps the discovered set — behavior unchanged.
            on_text_delta: Optional callback(text: str, done: bool) for text streaming
            on_thinking_delta: Optional callback(text: str, done: bool) for thinking streaming
        """
        if room_id not in self._room_locks:
            self._room_locks[room_id] = asyncio.Lock()
        async with self._room_locks[room_id]:
            self._current_tasks[room_id] = asyncio.current_task()
            history = self.history(room_id)
            try:
                # Build user content (may expand image media tags when the
                # room's active model supports vision, kdsn.275)
                content = _build_user_content(
                    text, self.config,
                    vision=model_supports_vision(self.get_model(room_id), self.config))

                # Check for context overflow before appending user message.
                # When append_user=False the message is already in history (hydrated
                # from JSONL by the gated-room path), so don't double-count it.
                content_tokens = self._estimate_content_tokens(content)
                context_tokens = self._estimate_context_tokens(room_id) + (
                    content_tokens if append_user else 0
                )
                limit = self._resolve_model_limit(room_id)
                available = self._effective_available(limit)
                # kdsn.305 turn-start AUTO tier: when the inclusive estimate
                # crosses auto_pct of the usable runway, attempt a boundary
                # BEFORE the overflow guard and ReminderState construction.
                # Callback absent (headless/CLI) -> skip silently; a callback
                # exception -> warn + continue the turn (never kill a turn
                # because GC failed). On applied the in-memory history was
                # rebuilt in place, so re-estimate for the guard below and
                # for the ReminderState (D9 single-source `available`).
                _gc_cfg = self.config.context
                _gc_cb = (callbacks or {}).get("apply_handoff_boundary")
                _gc_turn_start_applied = False
                _gc_pending_in_rebuild = False
                if _gc_cfg.handoff_enabled and _gc_cb is not None:
                    _gc_auto_threshold = int(available
                                             * _gc_cfg.auto_pct / 100)
                    _gc_auto_blocked = self._gc_auto_uncleared.get(room_id, False)
                    if context_tokens >= _gc_auto_threshold and not _gc_auto_blocked:
                        # kdsn.322.15 (SB ruling b) + audit fix: exclude_inflight
                        # follows append_user for TEXT turns - the just-persisted
                        # message is the LAST JSONL entry and stays above the
                        # boundary (live and rebuild agree; the operator's
                        # question is never silently swallowed). Vision turns
                        # (list content) revert to the strip: the JSONL raw form
                        # cannot carry the expanded image blocks, so the live
                        # append must run (pre-fix behavior). append_user=False
                        # paths (gated rooms, exec) keep the pinned strip: the
                        # message is stripped everywhere (spec 5(b)) - task-text
                        # survival is the SUB path and project-declared snapshots.
                        _res = await self._gc_apply_boundary(
                                room_id, callbacks, trigger="auto",
                                exclude_inflight=append_user
                                and isinstance(content, str))
                        if _res:
                            _gc_turn_start_applied = True
                            # audit H1: skip the live append ONLY when the
                            # rebuild actually carries the pending input - a
                            # user-role protected entry that rendered. A
                            # role='system' directive entry (heartbeat/umbral)
                            # never renders; its carrier is the live append.
                            _gc_pending_in_rebuild = bool(
                                _res.get("pending_protected"))
                            # D12 re-estimate, kdsn.322.15: when the rebuild
                            # carries the pending message it is already IN
                            # the history - adding content_tokens again
                            # would double-count.
                            context_tokens = self._estimate_context_tokens(room_id) + (
                                content_tokens
                                if (append_user and not _gc_pending_in_rebuild)
                                else 0
                            )
                            # Churn guard (audit): did this boundary actually
                            # clear the threshold? A failed clear latches the
                            # tier off until something clears it — re-applying
                            # every turn would append a fresh snapshot per
                            # turn without ever fitting.
                            if context_tokens < _gc_auto_threshold:
                                self._gc_auto_uncleared.pop(room_id, None)
                            else:
                                self._gc_auto_uncleared[room_id] = True
                if context_tokens > available:
                    raise ContextOverflowError(context_tokens, limit)

                # kdsn.322.15: after a turn-start boundary applied with the
                # pending message protected, the rebuild already carries it
                # (above the boundary) - a live re-append would duplicate it.
                # audit H1: skip the live append ONLY when the rebuild
                # carries the pending input (user-role protected span).
                # Heartbeat/umbral directives persist as system entries -
                # never rendered by the rebuild - so their live append
                # MUST run even on a boundary turn.
                if append_user and not _gc_pending_in_rebuild:
                    # R2-A: Escape user-origin <system-reminder> tags in context
                    # to prevent spoofing.  JSONL stores raw text (audit fidelity);
                    # escaping is context-only (mirrors /timesense, /steer).
                    # Only genuine user content is escaped — harness-injected
                    # reminder/steer strings are trusted and appended elsewhere.
                    if isinstance(content, str):
                        _escaped = escape_system_reminder_tags(content)
                    elif isinstance(content, list):
                        # N1: vision/multimodal content — escape text blocks,
                        # preserve image blocks (keeps live/rebuild symmetric).
                        _escaped = [
                            {**_blk, "text": escape_system_reminder_tags(_blk["text"])}
                            if isinstance(_blk, dict) and _blk.get("type") == "text"
                            and isinstance(_blk.get("text"), str) else _blk
                            for _blk in content
                        ]
                    else:
                        _escaped = content
                    history.append({"role": "user", "content": _escaped})
                # else: caller already appended via JSONL → build_context → history.extend

                # Per-turn tool call counter (reset each handle_input call)
                _tool_calls_this_turn: dict[str, int] = {}
                # Reset per-turn reminder state (R1-4: per-room engine)
                self._engine_for(room_id).reset_turn()
                # Determine turn source from callbacks (heartbeat/umbral/None)
                _turn_source = (callbacks or {}).get("turn_source")

                # Compute completed turns (user-role messages in history)
                _completed_turns = sum(1 for m in history if m.get("role") == "user")

                # Ensure per-room session tool counts exist
                if room_id not in self._room_tool_counts:
                    self._room_tool_counts[room_id] = {}

                # Turn-start tool set: per-turn `tools` override (kdsn.315)
                # wins over the init-discovered set; None keeps old behavior.
                _turn_tools = tools if tools is not None else self.tools

                # Determine enabled tool names — all builtins are available
                # regardless of which tool TOMLs are present in workspace/tools/
                # NOTE (R1-6): ideally {t.name for t in self.tools} but existing
                # test_guidance_injection tests create workspaces without
                # todo_write.toml and rely on T1 firing. Cannot change those
                # tests (FORBIDDEN). R1-6 real-path test validates engine-level
                # suppression. See circuit-breaker report.
                _enabled_tools = {t.name for t in _turn_tools}

                # Turn-start reminder evaluation (T3 fires here).
                # I1 durability invariant: every injection must become a durable JSONL
                # entry.  At the turn-start site, log_reminder MUST be present before
                # we evaluate or inject.  Without a usable persistence seam, skip
                # evaluation entirely (debug-level log explains why).
                _log_reminder_fn = (callbacks or {}).get("log_reminder")
                if not _log_reminder_fn:
                    logger.debug(
                        "Skipping turn-start reminder evaluation in %s: "
                        "log_reminder callback absent (I1 durability seam required)",
                        room_id,
                    )
                    _turn_start_reminders = []
                else:
                    _turn_start_state = ReminderState(
                        evaluation_point="turn_start",
                        iteration=0,
                        max_iterations=self.config.max_iterations,
                        # D12: the guard's already-computed inclusive estimate
                        # (context + content_tokens when append_user) — a large
                        # paste on a no-tool-call turn must not skip a tier.
                        context_tokens=context_tokens,
                        context_limit=limit,
                        # D9: same expression as the overflow guard above.
                        available_tokens=available,
                        # kdsn.322 handoff inputs (turn-start site only):
                        # checkpoint threshold from [context].checkpoint_pct
                        # of the usable runway (0 when handoff disabled ->
                        # engine silent), and the last boundary's
                        # post-boundary runway consumption fraction
                        # (kdsn.305.12 D5; 0.0 pre-boundary/restart ->
                        # engine silent).
                        checkpoint_threshold=(
                            int(available * _gc_cfg.checkpoint_pct / 100)
                            if _gc_cfg.handoff_enabled else 0),
                        handoff_runway_fraction=self._gc_runway_cached(room_id),
                        completed_turns=_completed_turns,
                        turn_source=_turn_source,
                        tool_calls_this_turn=_tool_calls_this_turn,
                        tool_calls_session=dict(self._room_tool_counts.get(room_id, {})),
                        todo_list=list(_TODO_STATE.get(room_id, [])),  # R1-3
                        enabled_tools=_enabled_tools,
                        # kdsn.298: orientation inputs — turn-start site ONLY;
                        # the boundary site flows the dataclass defaults.
                        # The __sub__ sentinel room is never orientable
                        # (decision 7): empty strings == the engine's
                        # default/un-oriented key suppress the predicate there.
                        **self._orient_inputs(room_id) if room_id != "__sub__" else {},
                    )
                    _turn_start_reminders = self._engine_for(room_id).evaluate(_turn_start_state)
                for _rem in _turn_start_reminders:
                    history.append({"role": "user", "content": _rem.content})
                    # send_notice is best-effort: failure does not block injection or persistence.
                    _send_notice = (callbacks or {}).get("send_notice")
                    if _send_notice:
                        try:
                            # R2-B: summary line + exact framed content (I2 compliance)
                            import html as _html_mod
                            _notice_body = (
                                f"🔔 System reminder ({_rem.trigger})\n\n"
                                f"{_html_mod.escape(_rem.content)}"
                            )
                            await _send_notice(
                                room_id,
                                _notice_body,
                            )
                        except Exception:
                            logger.warning("send_notice callback failed for reminder")
                    # log_reminder is present (gated above) — persist to JSONL.
                    try:
                        await _log_reminder_fn(room_id, _rem)
                    except Exception:
                        logger.warning("log_reminder callback failed")

                # Resolve the effective thinking level once (param overrides config).
                # This may be lowered to "off" mid-loop as a one-shot recovery when
                # extended thinking consumes the entire output budget (empty text with
                # stop_reason == "max_tokens"). See the retry logic in the text branch.
                effective_thinking = (
                    thinking if thinking is not None
                    else getattr(self.config, "thinking", "off")
                )
                retried_without_thinking = False

                # kdsn.315 P3: continuation budget consumption for this turn
                # (at most one attempt; capped by config.max_continuations).
                continuations_used = 0

                # kdsn.315 P2: one budget-exhaustion notice per turn, fired on
                # the FIRST length-stopped response seen (text or tool branch).
                _length_stopped_notified = False

                # kdsn.315 P2: did this turn SEE a max_tokens stop at any
                # point (the turn's first response, or a continuation's)?
                # The text-branch notice is gated on this rather than the
                # FINAL stop reason: after a successful continuation the
                # turn ends "end_turn" yet the turn still hit the budget and
                # must be surfaced exactly once.
                _turn_saw_max_tokens = False
                _clipped_output_tokens = 0

                # Tool loop: continue calling LLM until we get a text response
                for iteration in range(self.config.max_iterations):
                    # Drain steering inbox at the top of every iteration (before API call).
                    # Check both the direct kwarg and the callbacks dict (the latter allows
                    # _process_message to pass the closure without breaking existing test mocks
                    # that have explicit handle_input signatures without drain_steering).
                    _effective_drain = drain_steering or (callbacks or {}).get('drain_steering')
                    if _effective_drain:
                        _steer_notes = await _effective_drain()
                        for _note in _steer_notes:
                            if _note.strip():
                                history.append({
                                    "role": "user",
                                    "content": f"[Operator steering — mid-turn guidance]: {_note}",
                                })

                    # Drain the AGENT's OWN vision inbox at the top of every iteration —
                    # AFTER the steering drain, BEFORE reminder evaluation (placement
                    # unchanged from kdsn.276; state moved onto the Agent in kdsn.279,
                    # there is NO drain_vision callback key anymore — a stray one in
                    # callbacks is ignored). view_image deposits [media:] tags during
                    # the previous batch's tool execution; draining here injects them
                    # as ONE user message at the next loop top, AFTER ALL tool results
                    # of the pending batch (parallel batches included) BY CONSTRUCTION
                    # — do not move this inside the tool-result append loop. The framed
                    # tag text is expanded to image blocks via _build_user_content,
                    # gated on the room's active model's vision.
                    _tags = self._vision_inbox.pop(room_id, [])
                    if _tags:
                        from openalph.tools.vision import frame_vision_batch
                        _framed = frame_vision_batch(_tags)
                        history.append({"role": "user", "content": _build_user_content(
                            _framed, self.config,
                            vision=model_supports_vision(self.get_model(room_id), self.config))})
                        # Optional observability seam (kdsn.279): the transport logs
                        # the injection (JSONL source="view_image" + room notice).
                        # Fail-soft — injection is NEVER gated on logging.
                        _log_cb = (callbacks or {}).get("log_vision_injection")
                        if callable(_log_cb):
                            try:
                                await _log_cb(room_id, _framed)
                            except Exception:
                                logger.warning(
                                    "log_vision_injection callback failed in %s",
                                    room_id, exc_info=True)

                    # Drain this room's pending async-subagent terminal
                    # events (kdsn.330 R7) at the top of every iteration —
                    # AFTER the steering + vision drains, BEFORE the spotter
                    # (operator > harness > monitor; the harness tier keeps its
                    # vision-before-subagent order). Finalizers deposited the
                    # event dicts into _pending_events AFTER persisting the
                    # durable subagent_terminal entry (persist-before-notify),
                    # so this is pure consumption: N events => ONE user
                    # message, harness-framed (D3), content-free (ids/states/
                    # task heads + the sanitized error/result line). The RAW
                    # (unframed) lines are persisted as a
                    # source="subagent_event" user entry — build_context
                    # re-frames with the identical pure expression at rebuild
                    # (R6 store-raw/frame-at-build ⇒ A05/A06). Fail-soft like
                    # the vision drain: delivery is NEVER gated on logging.
                    _sub_events = self._pending_events.pop(room_id, [])
                    if _sub_events:
                        _sub_raw = subledger.subagent_event_lines(_sub_events)
                        history.append({
                            "role": "user",
                            "content": subledger.frame_subagent_event_content(
                                _sub_raw),
                        })
                        _log_ev_cb = (callbacks or {}).get("log_subagent_event")
                        if callable(_log_ev_cb):
                            try:
                                await _log_ev_cb(room_id, _sub_raw)
                            except Exception:
                                logger.warning(
                                    "log_subagent_event callback failed in %s",
                                    room_id, exc_info=True)
                        # Collapsed in-room notice carrying the exact event
                        # line bytes (inserts-visible; D1). Suppressed under
                        # the room's quiet flag (D8: quiet suppresses fires
                        # AND notices — never status truth). Best-effort:
                        # delivery is never gated on the notice.
                        if not getattr(self, "_room_quiet", {}).get(room_id):
                            _notice_ev_cb = (callbacks or {}).get("send_notice")
                            if callable(_notice_ev_cb):
                                try:
                                    await _notice_ev_cb(
                                        room_id,
                                        subledger.terminal_notice_line(_sub_events))
                                except Exception:
                                    logger.warning(
                                        "subagent event notice failed in %s",
                                        room_id, exc_info=True)

                    # Drain the SPOTTER's flag inbox at the top of every iteration —
                    # AFTER the steering + vision drains, BEFORE reminder evaluation
                    # (operator outranks harness outranks monitor; design §9/§12).
                    # A delivered flag enters history as ONE framed user message
                    # (advisory framing — a third-party claim to verify, never an
                    # operator instruction). Defensive: agents without a wired
                    # SpotterManager make this a no-op. The JSONL log callback is
                    # best-effort: injection is NEVER gated on logging (vision
                    # precedent), a failing log_spotter_flag only warns.
                    _spotter = getattr(self, "_spotter", None)
                    if _spotter is not None:
                        for _flag_framed, _flag_raw, _flag_class, _flag_sev in _spotter.drain_flags(room_id):
                            history.append({"role": "user", "content": _flag_framed})
                            _log_flag_cb = (callbacks or {}).get("log_spotter_flag")
                            if callable(_log_flag_cb):
                                try:
                                    await _log_flag_cb(room_id, _flag_raw,
                                                      flag_class=_flag_class,
                                                      flag_severity=_flag_sev)
                                except Exception:
                                    logger.warning(
                                        "log_spotter_flag callback failed in %s",
                                        room_id, exc_info=True)

                    # Spotter v2 (V1-C boundary firing): fire the watcher at the
                    # tool-loop iteration top — AFTER the drains, BEFORE reminder
                    # evaluation. Same sync fail-soft maybe_fire as turn
                    # completion; coalescing is the only cadence throttle, so a
                    # tight loop batches boundary segments into back-to-back
                    # passes. This is what delivers flags MID-TURN (the actual
                    # v2 product: course-correction while the watched agent
                    # still works).
                    self._fire_spotter_turn_completion(
                        room_id, history, _turn_source, callbacks)

                    # Reminder evaluation at tool-loop boundary (after steering, before API call).
                    # Ordering: steering drains first, then reminders (operator outranks harness).
                    # R1-5: mirror turn-start durability gate — when production callbacks
                    # (identified by "room_id" key from _build_agent_callbacks) are present
                    # but log_reminder is absent, skip evaluate+inject entirely to enforce
                    # I1 durability invariant.  Without production wiring, no gate.
                    _boundary_log_reminder = (callbacks or {}).get("log_reminder")
                    _is_production_callbacks = callbacks is not None and "room_id" in callbacks
                    if _is_production_callbacks and not _boundary_log_reminder:
                        logger.debug(
                            "Skipping boundary reminder evaluation in %s: "
                            "log_reminder callback absent (I1 durability seam required)",
                            room_id,
                        )
                    else:
                        context_tokens = self._estimate_context_tokens(room_id)
                        limit = self._resolve_model_limit(room_id)
                        _boundary_state = self._boundary_reminder_state(
                            room_id,
                            iteration=iteration,
                            context_tokens=context_tokens,
                            limit=limit,
                            completed_turns=_completed_turns,
                            turn_source=_turn_source,
                            tool_calls_this_turn=dict(_tool_calls_this_turn),
                            tool_calls_session=dict(
                                self._room_tool_counts.get(room_id, {})),
                            todo_list=list(_TODO_STATE.get(room_id, [])),  # R1-3
                            enabled_tools=_enabled_tools,
                        )
                        _boundary_reminders = self._engine_for(room_id).evaluate(_boundary_state)
                        for _rem in _boundary_reminders:
                            history.append({"role": "user", "content": _rem.content})
                            # send_notice is best-effort: failure must not alter model-visible behavior.
                            _send_notice = (callbacks or {}).get("send_notice")
                            if _send_notice:
                                try:
                                    # R2-B: summary line + exact framed content (I2 compliance)
                                    import html as _html_mod
                                    _notice_body = (
                                        f"🔔 System reminder ({_rem.trigger})\n\n"
                                        f"{_html_mod.escape(_rem.content)}"
                                    )
                                    await _send_notice(
                                        room_id,
                                        _notice_body,
                                    )
                                except Exception:
                                    logger.warning("send_notice callback failed for reminder")
                            # log_reminder is present (gated above) — persist to JSONL.
                            try:
                                await _boundary_log_reminder(room_id, _rem)
                            except Exception:
                                logger.warning("log_reminder callback failed")

                    # Check for context overflow before calling the API (tool results may push over)
                    context_tokens = self._estimate_context_tokens(room_id)
                    limit = self._resolve_model_limit(room_id)
                    available = self._effective_available(limit)
                    # kdsn.305 HARD tier: attempt a boundary once the
                    # estimate crosses the hard threshold (hard_pct of usable,
                    # default 92%) — a failed attempt must not kill a turn
                    # that still fits, so the RAISE stays tied to >available
                    # (audit: hard_pct was parsed-then-dead; the tier fired
                    # only at 100%). 3-strike breaker — after
                    # _GC_HARD_STRIKE_LIMIT consecutive failed attempts in
                    # this room, stop trying and raise as pre-GC (a wedged
                    # boundary writer must not wedge the guard either). On
                    # applied the history was rebuilt in place: re-estimate
                    # and proceed when the room is back under the limit.
                    _gc_hard_threshold = int(
                        available
                        * getattr(self.config.context, "hard_pct", 92) / 100
                    ) if getattr(self.config, "context", None) else available
                    if context_tokens >= _gc_hard_threshold:
                        _gc_cleared = False
                        if await self._gc_hard_tier(
                                room_id, callbacks,
                                context_tokens=context_tokens,
                                available=available):
                            context_tokens = self._estimate_context_tokens(room_id)
                            _gc_cleared = context_tokens <= available
                        if context_tokens > available and not _gc_cleared:
                            raise ContextOverflowError(context_tokens, limit)

                    # kdsn.329: char total of the EXACT payload about to be
                    # sent (the SAME single walk the estimator uses, so the
                    # merge's marginal term shares its semantics), captured
                    # AFTER the hard-tier block above — a mid-turn boundary
                    # rebuilds history wholesale, and a pre-boundary cursor
                    # would pair this call's (rebuilt-payload) true prompt
                    # with a stale pre-boundary char total (audit H1).
                    # Loop-local: refreshed every iteration; the record seam
                    # below anchors the provider's true prompt size to this
                    # cursor.
                    pre_call_chars = self._context_char_total(room_id)

                    # Pass tools=None if no tools available (backward compatibility)
                    tools_arg = _turn_tools if _turn_tools else None

                    # Terminal tool (Stigmergy Decision 18, bead
                    # workspace-e2uh.152): per-run state registered by the
                    # caller (exec --submit-schema) on the agent BEFORE
                    # handle_input — never a global, never cached across
                    # runs. When present, the provider grammar-constrains
                    # this turn's tool schemas (strict=True, kdsn.304) so
                    # the terminal call's arguments are schema-faithful.
                    _terminal_name = None
                    _terminal_strict = False
                    _tt = getattr(self, "_terminal_tool", None)
                    if _tt is not None:
                        _terminal_name = _tt[0]
                        _terminal_strict = bool(_tt[1].config.get("strict"))

                    # Record start time for latency measurement
                    start_time = time.monotonic()

                    # Use streaming
                    accumulated_text = ""
                    accumulated_thinking = ""
                    response = None
                    text_emitted = False
                    thinking_emitted = False
                    tool_calls = []
                    usage = None

                    # Cache-keepalive (spec §1.3): snapshot the EXACT request suffix
                    # sent this iteration -- it ends on a valid user/tool_result
                    # boundary. The post-response history ends on the assistant
                    # tool_use turn, which is not a replayable prefix.
                    # INVARIANT: history dicts (and tools_arg) are APPEND-ONLY /
                    # stable between this shallow snapshot and the keepalive's last
                    # ping. In-place mutation of an existing entry would drift the
                    # cache key. The hit-check catches drift (one bounded write) but
                    # preventing it is cheaper -- keep such mutations append-only.
                    _ka_request_messages = list(history)
                    _degenerate_notified = False

                    async for event in stream(
                        config=self.config,
                        system=self.system_prompt,
                        messages=_ka_request_messages,
                        tools=tools_arg,
                        model=self.get_model(room_id),
                        thinking=effective_thinking,
                        cache_ttl=cache_ttl,
                        room_id=room_id,
                        strict=_terminal_strict,
                    ):
                        if event.type == "text":
                            accumulated_text += event.content
                            if on_text_delta:
                                await on_text_delta(event.content, done=False)
                            text_emitted = True
                        elif event.type == "thinking":
                            accumulated_thinking += event.content
                            if on_thinking_delta:
                                await on_thinking_delta(event.content, done=False)
                            thinking_emitted = True
                        elif event.type == "signature":
                            # Signature is part of thinking block, but we handle it at done
                            pass
                        elif event.type == "tool_done":
                            tool_calls.append(event.tool_call)
                        elif event.type == "degenerate":
                            # Mid-stream degeneration notice (kdsn.241.21): fire
                            # on_degenerate immediately so the operator sees the
                            # notice while the generation is still in flight, not
                            # after it completes. The provider's monitor.tripped
                            # flag ensures this event fires exactly once.
                            _on_degenerate = (callbacks or {}).get("on_degenerate")
                            if _on_degenerate:
                                try:
                                    await _on_degenerate(
                                        model=event.model,
                                        generation_id=event.generation_id,
                                    )
                                except Exception:
                                    logger.debug("on_degenerate callback raised (ignored)", exc_info=True)
                            _degenerate_notified = True
                        elif event.type == "done":
                            response = event.response
                            # Fire done signals
                            if text_emitted and on_text_delta:
                                await on_text_delta("", done=True)
                            if thinking_emitted and on_thinking_delta:
                                await on_thinking_delta("", done=True)

                    if response is None:
                        # Fallback if no done event
                        response = await complete(
                            config=self.config,
                            system=self.system_prompt,
                            messages=list(history),
                            tools=tools_arg,
                            model=self.get_model(room_id),
                            thinking=effective_thinking,
                            room_id=room_id,
                            strict=_terminal_strict,
                        )

                    # Post-stream degeneration backstop (kdsn.241.21): if the
                    # mid-stream degenerate event didn't fire (e.g. the complete()
                    # fallback path has no stream events), check response.degenerate
                    # here. Skipped if the mid-stream handler already notified.
                    if not _degenerate_notified and getattr(response, 'degenerate', False):
                        _on_degenerate = (callbacks or {}).get("on_degenerate")
                        if _on_degenerate:
                            try:
                                await _on_degenerate(model=self.get_model(room_id), generation_id=getattr(response, 'generation_id', None))
                            except Exception:
                                logger.debug("on_degenerate callback raised (ignored)", exc_info=True)

                    latency_ms = (time.monotonic() - start_time) * 1000

                    _pkey, _ptype = self._provider_gate(self.get_model(room_id))
                    self._record_turn_usage(
                        room_id, response.usage, response.model, cache_ttl,
                        provider_key=_pkey, provider_type=_ptype,
                        pre_call_chars=pre_call_chars)
                    # kdsn.315 P1 (loop-level belt): canonicalize
                    # budget-exhaustion vocabulary at the point the loop sees
                    # the response. The provider seam already maps
                    # OpenAI-compat "length" -> "max_tokens" (no-op in
                    # production); it is repeated here so the downstream
                    # checks (P2 notice, P3 continuation, P4 empty-retry) are
                    # robust to a response that bypassed the provider seam —
                    # test fakes driving openalph.agent.stream directly, or a
                    # future non-provider source.
                    if getattr(response, "stop_reason", None) == "length":
                        response.stop_reason = "max_tokens"
                    # kdsn.315 P2: remember that the turn saw a budget
                    # exhaustion, even if a later continuation ends
                    # "end_turn" (the notice must still fire once per turn).
                    if response.stop_reason == "max_tokens":
                        _turn_saw_max_tokens = True
                        _clipped_output_tokens = (
                            getattr(response.usage, "output_tokens", 0) or 0)
                    usage = response.usage

                    # Check if response has tool calls
                    if not tool_calls and not response.tool_calls:
                        # Text response - log and return
                        final_text = accumulated_text or response.content

                        # Recovery: extended thinking can consume the entire output
                        # budget, yielding empty text with stop_reason == "max_tokens"
                        # (the model "thought" until it hit max_tokens and never wrote
                        # an answer). Retry once with thinking disabled so the full
                        # budget is available for output. Single attempt only; if
                        # thinking is already off we cannot reduce it further, and a
                        # non-empty (merely truncated) answer is kept as-is.
                        if (
                            not (final_text or "").strip()
                            and response.stop_reason == "max_tokens"
                            and effective_thinking != "off"
                            and not retried_without_thinking
                        ):
                            retried_without_thinking = True
                            effective_thinking = "off"
                            # Per-attempt observability (pre-kdsn315 contract,
                            # pinned by test_stop_reason_logs_both_attempts_on_
                            # recovery): the failed attempt is a REAL API call —
                            # log it with its own usage/stop_reason so the turn
                            # log shows BOTH the empty max_tokens attempt and the
                            # recovery. (A continuation, by contrast, merges into
                            # ONE entry below.)
                            self._log_turn(
                                room_id=room_id,
                                model=self.get_model(room_id),
                                input_tokens=usage.input_tokens,
                                output_tokens=usage.output_tokens,
                                tool_calls=None,
                                latency_ms=latency_ms,
                                content_preview=final_text,
                                cache_read_tokens=usage.cache_read_tokens,
                                cache_creation_tokens=usage.cache_creation_tokens,
                                stop_reason=response.stop_reason,
                            )
                            if on_cache_status:
                                try:
                                    await on_cache_status(usage, self.get_model(room_id))
                                except Exception:
                                    logger.error("on_cache_status callback failed", exc_info=True)
                            logger.warning(
                                "Empty response in %s (stop_reason=max_tokens — extended "
                                "thinking consumed the entire %d-token output budget); "
                                "retrying once with thinking disabled",
                                room_id, usage.output_tokens,
                            )
                            continue

                        # kdsn.315 P3: single continuation attempt on a
                        # NON-EMPTY length-stop (stop_reason=max_tokens).
                        # REQUEST-LOCAL: the continuation's framing user message
                        # and the partial assistant message are carried in the
                        # call's `messages=` list ONLY — never appended to
                        # durable history (same no-history-mutation
                        # philosophy as the retry-without-thinking path above).
                        # The durable history gains exactly ONE merged
                        # assistant message below (RC1).
                        if (
                            final_text
                            and response.stop_reason == "max_tokens"
                            # kdsn.315 P3 is scoped to OpenAI-compat providers:
                            # that is where the wire vocabulary is "length" and
                            # where the budget-exhaustion clips the spec targets
                            # (blackwell/synthetic/fireworks/macstudio). An
                            # Anthropic max_tokens text turn keeps the
                            # pre-kdsn315 "kept as-is" contract pinned by
                            # tests/test_empty_response_retry.py (no continuation
                            # for it).
                            and _ptype == "openai"
                            and continuations_used < self.config.max_continuations
                        ):
                            continuations_used += 1
                            _partial_assistant = {"role": "assistant", "content": final_text}
                            if response.thinking:
                                _partial_assistant["thinking"] = [
                                    {"thinking": tb.thinking, "signature": tb.signature}
                                    for tb in response.thinking
                                ]
                            _continuation_messages = history + [
                                _partial_assistant,
                                {
                                    "role": "user",
                                    "content": (
                                        "Your previous response was cut off at the "
                                        "output-token limit. Continue EXACTLY where "
                                        "you left off mid-sentence — no preamble, no "
                                        "repetition. You may review your partial "
                                        "answer in the assistant message above and "
                                        "finish the response."
                                    ),
                                },
                            ]
                            # Same stream()/complete() seam as the main loop,
                            # same effective thinking level (a continuation is
                            # a REAL generation — do NOT drop thinking).
                            _cont_response = None
                            async for _cont_event in stream(
                                config=self.config,
                                system=self.system_prompt,
                                messages=_continuation_messages,
                                tools=tools_arg,
                                model=self.get_model(room_id),
                                thinking=effective_thinking,
                                cache_ttl=cache_ttl,
                                room_id=room_id,
                                strict=_terminal_strict,
                            ):
                                if _cont_event.type == "done":
                                    _cont_response = _cont_event.response
                            if _cont_response is None:
                                _cont_response = await complete(
                                    config=self.config,
                                    system=self.system_prompt,
                                    messages=_continuation_messages,
                                    tools=tools_arg,
                                    model=self.get_model(room_id),
                                    thinking=effective_thinking,
                                    room_id=room_id,
                                    strict=_terminal_strict,
                                )
                            if _cont_response.tool_calls:
                                # The continuation answered with tool calls
                                # instead of text: ABANDON the attempt. Never
                                # execute those tools, never grow the loop —
                                # the partial text stands as the final answer.
                                _abandoned = [
                                    f"{tc.name}({json.dumps(tc.input, ensure_ascii=False)})"
                                    for tc in _cont_response.tool_calls
                                ]
                                logger.warning(
                                    "Continuation in %s produced tool calls "
                                    "(%s) instead of text; abandoning the "
                                    "continuation — keeping the partial "
                                    "response as the final answer",
                                    room_id, ", ".join(_abandoned),
                                )
                                # Keep the partial text; still record the
                                # continuation's stop_reason and merge its
                                # usage into the turn.
                                response = _cont_response
                            else:
                                # Direct concatenation (no separator): the
                                # model continues mid-sentence from its own
                                # partial.
                                final_text = final_text + (_cont_response.content or "")
                                response = _cont_response
                            _cont_usage = _cont_response.usage or Usage(
                                input_tokens=0, output_tokens=0)
                            # Merge response2.usage into the turn's usage
                            # accumulator so _log_turn / the serializer see
                            # the SUM, not just the continuation's slice.
                            usage = Usage(
                                input_tokens=usage.input_tokens + _cont_usage.input_tokens,
                                output_tokens=usage.output_tokens + _cont_usage.output_tokens,
                                cache_read_tokens=(
                                    (usage.cache_read_tokens or 0)
                                    + (_cont_usage.cache_read_tokens or 0)
                                ),
                                cache_creation_tokens=(
                                    (usage.cache_creation_tokens or 0)
                                    + (_cont_usage.cache_creation_tokens or 0)
                                ),
                            )
                            # kdsn.315 H1 (audit remediation): the
                            # continuation's tokens must reach the REAL
                            # usage accumulator — _record_turn_usage feeds
                            # global/per-room counters, frozen USD cost, and
                            # the serializer's per-turn delta. The merged
                            # `usage` above feeds only the turn's JSONL row;
                            # the main call's delta was already recorded at
                            # loop top, so record the continuation DELTA here.
                            self._record_turn_usage(
                                room_id, _cont_usage,
                                _cont_response.model
                                or self.get_model(room_id),
                                cache_ttl, provider_key=_pkey,
                                provider_type=_ptype)

                        # kdsn.315 P1 (belt, continuation): response2 arrives
                        # AFTER the loop-level normalization above, so its
                        # "length" would otherwise stay unnormalized and leak
                        # into the recorded stop reason and the P2 notice.
                        if (response.stop_reason == "length"):
                            response.stop_reason = "max_tokens"
                            _turn_saw_max_tokens = True
                            _clipped_output_tokens = (
                                getattr(response.usage, "output_tokens", 0) or 0)
                        assistant_msg = {"role": "assistant", "content": final_text}
                        # kdsn.315 P3: when a continuation resolved, the durable
                        # message carries the MERGED thinking — the partial's
                        # blocks first (stashed in _partial_assistant at
                        # continuation entry), then the continuation's; the
                        # partial's streamed text stands in as a single block
                        # when it had no structured blocks. The non-continuation
                        # path is byte-identical to the pre-kdsn315 assignment.
                        if (continuations_used or accumulated_thinking
                                or response.thinking):
                            if continuations_used:
                                _merged_thinking = list(
                                    _partial_assistant.get("thinking") or [])
                                if not _merged_thinking and accumulated_thinking:
                                    _merged_thinking.append(
                                        {"thinking": accumulated_thinking,
                                         "signature": ""})
                                for tb in (response.thinking or []):
                                    _merged_thinking.append(
                                        {"thinking": tb.thinking,
                                         "signature": tb.signature})
                                if _merged_thinking:
                                    assistant_msg["thinking"] = _merged_thinking
                            else:
                                thinking_blocks = response.thinking if response.thinking else []
                                if accumulated_thinking and not thinking_blocks:
                                    thinking_blocks = [ThinkingBlock(thinking=accumulated_thinking, signature="")]
                                assistant_msg["thinking"] = [
                                    {"thinking": tb.thinking, "signature": tb.signature}
                                    for tb in thinking_blocks
                                ]
                        # kdsn.315 P3: the text turn's log entry fires ONCE,
                        # AFTER continuation resolution, so it sees the MERGED
                        # turn usage (partial + continuation) and the FINAL
                        # stop reason. (Pre-.315 the entry fired before the
                        # retry/continuation points; a continuation-less turn
                        # is byte-identical to that old single entry.)
                        self._log_turn(
                            room_id=room_id,
                            model=self.get_model(room_id),
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            tool_calls=None,
                            latency_ms=latency_ms,
                            content_preview=final_text,
                            cache_read_tokens=usage.cache_read_tokens,
                            cache_creation_tokens=usage.cache_creation_tokens,
                            stop_reason=response.stop_reason,
                        )
                        if on_cache_status:
                            try:
                                await on_cache_status(usage, self.get_model(room_id))
                            except Exception:
                                logger.error("on_cache_status callback failed", exc_info=True)
                        # kdsn.315 P2: notice AFTER continuation resolution so
                        # the body is accurate ("Continuation also truncated."
                        # only when the attempt also clipped).
                        # kdsn.315 M1 (audit remediation): fire only when the
                        # turn's FINAL outcome is a truncation OR a
                        # continuation happened — a recovered empty-retry
                        # turn (no continuation, final end_turn) is NOT
                        # truncated and stays silent. L4: the count is the
                        # length-stopped response's, not the merged turn's.
                        if (final_text or "").strip() \
                                and _turn_saw_max_tokens \
                                and not _length_stopped_notified \
                                and (response.stop_reason == "max_tokens"
                                     or continuations_used):
                            _clipped = (_clipped_output_tokens
                                        or usage.output_tokens)
                            _notice_body = (
                                "⚠️ Turn hit the output-token budget "
                                "(stop_reason=max_tokens"
                                + (f", {_clipped} output tokens"
                                   if _clipped else "")
                                + ")."
                            )
                            if continuations_used and response.stop_reason == "max_tokens":
                                _notice_body += " Continuation also truncated."
                            _notice_cb = (callbacks or {}).get("send_notice")
                            if _notice_cb:
                                try:
                                    await _notice_cb(room_id, _notice_body)
                                except Exception:
                                    logger.warning(
                                        "send_notice callback failed for "
                                        "length-stop notice", exc_info=True)
                            _length_stopped_notified = True
                        # INVARIANT (RC1): assistant_msg must be history[-1] when matrix persists this turn after return.
                        history.append(assistant_msg)
                        self._last_stop_reason[room_id] = response.stop_reason
                        # Spotter v1 (design §3): the turn is complete — the monitor
                        # watches the entries that just landed. maybe_fire is sync,
                        # never raises, and does not mutate history (RC1 holds).
                        self._fire_spotter_turn_completion(
                            room_id, history, _turn_source, callbacks)
                        return final_text

                    # kdsn.315 P2: fire the turn's one length-stop notice on
                    # the FIRST length-stopped response in the loop (a turn
                    # may clip across several tool iterations — one notice
                    # total). Tool execution proceeds exactly as today; a
                    # notice failure never alters the turn.
                    if response.stop_reason == "max_tokens" and not _length_stopped_notified:
                        _length_stopped_notified = True
                        _notice_cb = (callbacks or {}).get("send_notice")
                        if _notice_cb:
                            try:
                                _body = "⚠️ Turn hit the output-token budget (stop_reason=max_tokens"
                                if usage.output_tokens:
                                    _body += f", {usage.output_tokens} output tokens"
                                _body += ")."
                                await _notice_cb(room_id, _body)
                            except Exception:
                                logger.warning(
                                    "send_notice callback failed for length-stop notice",
                                    exc_info=True)

                    # Use tool_calls from stream or from response
                    active_tool_calls = tool_calls if tool_calls else response.tool_calls

                    # Tool use - append assistant message with tool_calls to history
                    tool_msg = {
                        "role": "assistant",
                        "content": accumulated_text or response.content,
                        "tool_calls": active_tool_calls,
                    }
                    if accumulated_thinking or response.thinking:
                        thinking_blocks = response.thinking if response.thinking else []
                        if accumulated_thinking and not thinking_blocks:
                            thinking_blocks = [ThinkingBlock(thinking=accumulated_thinking, signature="")]
                        tool_msg["thinking"] = [
                            {"thinking": tb.thinking, "signature": tb.signature}
                            for tb in thinking_blocks
                        ]
                    # INVARIANT (RC1): tool_msg (carrying thinking) MUST remain history[-1]
                    # when on_tool_intent fires — matrix._persist_assistant_turn reads
                    # thinking from history[-1]. Do NOT insert any history mutation between
                    # this append and the on_tool_intent call below.
                    history.append(tool_msg)

                    # Emit tool intent before execution (for session logging / observability)
                    if on_tool_intent:
                        try:
                            await on_tool_intent(active_tool_calls, accumulated_text or response.content)
                        except Exception as e:
                            logger.warning("Tool intent callback failed: %s", e)

                    # TERMINAL tool (Stigmergy Decision 18, bead
                    # workspace-e2uh.152): a call to the run's registered
                    # terminal tool is the episode's return value — there is
                    # NO implementation to run. The check sits at the
                    # tool-dispatch site: AFTER the assistant tool-call
                    # message is appended and the intent fired (so the call
                    # is provenanced exactly like a normal tool call —
                    # counted, logged, on_tool_intent'd, and reported via
                    # on_tool_call, which is how the exec-level tool_trace
                    # picks it up) and BEFORE any tool execution. A tool
                    # result is appended for every call in the batch so
                    # history has no orphan tool_calls (RC1-consistent).
                    # Then the turn ends CLEANLY — the text-return path
                    # (final stop_reason recorded, spotter fires, return),
                    # not an error. Iteration cap is unaffected: this is
                    # inside the loop, so a run that burns all turns before
                    # submitting still falls through to the cap sentinel +
                    # forced summary (a relay deny never reaches this site —
                    # it raises out of stream/complete).
                    if _terminal_name is not None and any(
                            tc.name == _terminal_name for tc in active_tool_calls):
                        self._record_tool_calls(room_id, len(active_tool_calls))
                        if room_id not in self._room_tool_counts:
                            self._room_tool_counts[room_id] = {}
                        for tc in active_tool_calls:
                            self._room_tool_counts[room_id][tc.name] = (
                                self._room_tool_counts[room_id].get(tc.name, 0) + 1
                            )
                            _tool_calls_this_turn[tc.name] = (
                                _tool_calls_this_turn.get(tc.name, 0) + 1
                            )
                        _tt_logged = []
                        for tc in active_tool_calls:
                            _tt_logged.append({
                                "name": tc.name,
                                "input": tc.input,
                                "is_error": False,
                            })
                        self._log_turn(
                            room_id=room_id,
                            model=self.get_model(room_id),
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            tool_calls=_tt_logged,
                            latency_ms=latency_ms,
                            content_preview=accumulated_text or response.content,
                            cache_read_tokens=usage.cache_read_tokens,
                            cache_creation_tokens=usage.cache_creation_tokens,
                            stop_reason=response.stop_reason,
                        )
                        if on_cache_status:
                            try:
                                await on_cache_status(usage, self.get_model(room_id))
                            except Exception:
                                logger.error("on_cache_status callback failed", exc_info=True)
                        _tt_submit = None
                        for tc in active_tool_calls:
                            if tc.name == _terminal_name:
                                # First terminal call in the batch wins; the
                                # arguments dict IS the episode's return value.
                                _tt_submit = tc.input
                                _tt_result_content = "submitted"
                            else:
                                # Non-terminal calls in the same batch are
                                # provenanced + closed (no orphan) but NOT
                                # executed — the terminal call ends the turn.
                                _tt_result_content = (
                                    "not executed: turn ended by terminal tool")
                            history.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": wrap_tool_result(
                                    _tt_result_content, tc.name, tc.id),
                                "is_error": False,
                            })
                            if on_tool_call:
                                try:
                                    await on_tool_call(
                                        tc.id, tc.name, tc.input,
                                        _tt_result_content, False,
                                    )
                                except Exception as e:
                                    logger.warning("Tool call callback failed: %s", e)
                        self._terminal_submit = _tt_submit
                        self._last_stop_reason[room_id] = response.stop_reason
                        # Spotter v1 (design §3): the turn is complete — same
                        # sync, fail-soft, history-non-mutating maybe_fire as
                        # the text-return path (RC1 holds).
                        self._fire_spotter_turn_completion(
                            room_id, history, _turn_source, callbacks)
                        return accumulated_text or response.content

                    # Execute tool calls in parallel
                    tool_coros = []
                    for tc in active_tool_calls:
                        # Find the tool config for this tool
                        tool_config = {}
                        for t in _turn_tools:
                            if t.name == tc.name:
                                tool_config = t.config
                                break

                        # Thread call_id through for subagent log cross-referencing
                        tc_callbacks = {**(callbacks or {}), "call_id": tc.id,
                                        "active_model": self.get_model(room_id)}
                        # Default vision_deposit seam (kdsn.279): the agent owns the
                        # per-room inbox, so it wires the deposit for every turn shape
                        # (interactive/heartbeat/umbral/CLI). setdefault is load-bearing:
                        # an explicit vision_deposit in the caller's callbacks WINS
                        # (test/transport override stays possible).
                        tc_callbacks.setdefault(
                            "vision_deposit",
                            functools.partial(self._vision_deposit, room_id))
                        # workspace-kdsn.317: per-tool executor bridge.
                        # Passed only when the agent carries a map — no-map
                        # agents keep the byte-identical legacy call shape
                        # (several test doubles pin the legacy signature
                        # exactly and must keep working).
                        _tx_exec = ({"tool_executors": self.tool_executors}
                                    if self.tool_executors is not None else {})
                        tool_coros.append(execute_tool(
                            name=tc.name,
                            input=tc.input,
                            tool_config=tool_config,
                            agent_config=self.config,
                            tools=_turn_tools,
                            callbacks=tc_callbacks,
                            **_tx_exec,
                        ))

                    _ka_task, _ka_stop = await self._maybe_arm_cache_keepalive(
                        room_id=room_id,
                        active_tool_calls=active_tool_calls,
                        request_messages=_ka_request_messages,
                        tools_arg=tools_arg,
                        cache_ttl=cache_ttl,
                        callbacks=callbacks,
                        thinking=effective_thinking,
                        stream_start=start_time,
                    )
                    try:
                        results = await asyncio.gather(*tool_coros)
                    finally:
                        if _ka_task is not None:
                            _room_ledger = self._dispatch_ledger.get(room_id)
                            _async_running = (
                                _room_ledger is not None
                                and any(r.state == subledger.RUNNING
                                        for r in _room_ledger.values())
                            )
                            if _async_running:
                                # kdsn.330 R9: async dispatches are still in
                                # flight — the gather returned at RECEIPT
                                # time, not at sub completion, so the
                                # keepalive must outlive the batch. Retain
                                # the (task, stop) pair; the LAST async
                                # terminal (Agent._finalize_dispatch)
                                # disarms it. Supersede any prior retained
                                # entry for the room — stop + cancel its
                                # loop first so a second arm never leaves
                                # an orphan loop or double-disarms.
                                _prev_ka = self._room_keepalive.get(room_id)
                                if _prev_ka is not None:
                                    _prev_ka[1].set()
                                    _prev_ka[0].cancel()
                                    with contextlib.suppress(asyncio.CancelledError):
                                        await _prev_ka[0]
                                self._room_keepalive[room_id] = (_ka_task, _ka_stop)
                                logger.info(
                                    "cache keepalive retained for async "
                                    "dispatch in %s", room_id)
                            else:
                                _ka_stop.set()
                                _ka_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await _ka_task
                                logger.info("cache keepalive disarmed in %s", room_id)

                    # Track total tool calls (aggregate + per-tool-name + per-turn)
                    self._record_tool_calls(room_id, len(active_tool_calls))
                    if room_id not in self._room_tool_counts:
                        self._room_tool_counts[room_id] = {}
                    for tc in active_tool_calls:
                        self._room_tool_counts[room_id][tc.name] = (
                            self._room_tool_counts[room_id].get(tc.name, 0) + 1
                        )
                        _tool_calls_this_turn[tc.name] = (
                            _tool_calls_this_turn.get(tc.name, 0) + 1
                        )

                    # Build tool_calls log entry with is_error from results
                    logged_tool_calls = []
                    for tc, result in zip(active_tool_calls, results):
                        logged_tool_calls.append({
                            "name": tc.name,
                            "input": tc.input,
                            "is_error": result.is_error,
                        })

                    # Log this tool-use turn
                    self._log_turn(
                        room_id=room_id,
                        model=self.get_model(room_id),
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        tool_calls=logged_tool_calls,
                        latency_ms=latency_ms,
                        content_preview=accumulated_text or response.content,
                        cache_read_tokens=usage.cache_read_tokens,
                        cache_creation_tokens=usage.cache_creation_tokens,
                        stop_reason=response.stop_reason,
                    )
                    if on_cache_status:
                        try:
                            await on_cache_status(usage, self.get_model(room_id))
                        except Exception:
                            logger.error("on_cache_status callback failed", exc_info=True)

                    # Append tool results to history (truncated + wrapped) and notify
                    for tc, result in zip(active_tool_calls, results):
                        truncated_content = truncate_result(result.content, self.config.truncation_limit)
                        wrapped_content = wrap_tool_result(truncated_content, tc.name, tc.id)
                        history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": wrapped_content,
                            "is_error": result.is_error,
                        })
                        if on_tool_call:
                            try:
                                await on_tool_call(
                                    tc.id, tc.name, tc.input, wrapped_content, result.is_error
                                )
                            except Exception as e:
                                logger.warning("Tool call callback failed: %s", e)

                # Hit max iterations — request a summary from the model
                logger.warning("Tool call limit (%d) reached in room %s",
                               self.config.max_iterations, room_id)
                limit_notice = (
                    "[SYSTEM: Tool call limit reached. You MUST now summarize your progress. "
                    "State what you completed, what remains, and any partial results. "
                    "Do NOT attempt further tool calls.]"
                )
                history.append({"role": "user", "content": limit_notice})
                # FIX 2: clear stale last-turn usage BEFORE the summary try, so that if
                # the summary fails or yields no done event, serializer sees no usage
                # (preventing double-count of the prior tool turn on restart).
                # On summary success, _record_turn_usage re-sets it correctly.
                self._last_turn_usage.pop(room_id, None)

                try:
                    summary_text = ""
                    summary_response = None
                    async for event in stream(
                        config=self.config,
                        system=self.system_prompt,
                        messages=list(history),
                        tools=None,  # no tools — force text response
                        model=self.get_model(room_id),
                        room_id=room_id,
                        # Force thinking off for the forced summary: it must produce
                        # visible output, and extended thinking here could consume the
                        # whole budget and return empty with no recovery path (the same
                        # failure the in-loop retry guards against).
                        thinking="off",
                        cache_ttl=cache_ttl,
                    ):
                        if event.type == "text":
                            summary_text += event.content
                            if on_text_delta:
                                await on_text_delta(event.content, done=False)
                        elif event.type == "done":
                            summary_response = event.response
                            if on_text_delta:
                                await on_text_delta("", done=True)

                    if summary_response:
                        _pkey, _ptype = self._provider_gate(self.get_model(room_id))
                        self._record_turn_usage(
                            room_id, summary_response.usage, summary_response.model, cache_ttl,
                            provider_key=_pkey, provider_type=_ptype)
                        if on_cache_status:
                            try:
                                await on_cache_status(summary_response.usage, self.get_model(room_id))
                            except Exception:
                                logger.error("on_cache_status callback failed", exc_info=True)

                    final_text = summary_text or (
                        summary_response.content if summary_response else
                        "⚠️ Tool call limit reached and summary generation failed."
                    )
                except Exception as e:
                    logger.warning("Summary generation failed after tool limit: %s", e)
                    final_text = (
                        f"⚠️ **Tool call limit reached** ({self.config.max_iterations} iterations). "
                        "Summary generation also failed — check logs for details."
                    )

                history.append({"role": "assistant", "content": final_text})
                self._last_stop_reason[room_id] = summary_response.stop_reason if summary_response else "max_iterations"
                # Spotter v1 (design §3): the turn is complete — fire the monitor
                # over the max-iterations summary turn (same sync, fail-soft,
                # history-non-mutating maybe_fire as the text-return path).
                self._fire_spotter_turn_completion(
                    room_id, history, _turn_source, callbacks)
                return final_text
            except asyncio.CancelledError:
                # /stop or process shutdown cancelled us mid-tool-loop.
                # The in-memory history may have an assistant message with
                # tool_calls but no tool results (orphan). Strip orphans
                # to prevent bricking the session on the next message.
                self._repair_history(history)
                raise
            finally:
                # Only clear OUR entry, and only if it is still ours: another
                # turn for the same room must not have its task dropped by a
                # late-finishing predecessor.
                if self._current_tasks.get(room_id) is asyncio.current_task():
                    self._current_tasks.pop(room_id, None)

    @staticmethod
    def _repair_history(history: list[dict]) -> None:
        """Strip orphaned tool_calls and their partial results from history.

        An orphan is an assistant message with tool_calls where not all
        tool_call IDs have matching tool results after it in the history.
        This happens when CancelledError interrupts asyncio.gather() during
        tool execution.

        Mutates the list in place. Also strips partial tool results that
        belong to orphaned assistant messages (they would reference
        non-existent tool_use blocks).
        """
        # Identify orphaned assistant messages
        orphaned_indices = set()
        orphaned_tc_ids = set()

        for i, msg in enumerate(history):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                needed_ids = {tc.id for tc in msg["tool_calls"]}
                found_ids = set()
                for j in range(i + 1, len(history)):
                    entry = history[j]
                    if entry.get("role") == "tool" and entry.get("tool_call_id") in needed_ids:
                        found_ids.add(entry["tool_call_id"])
                if needed_ids - found_ids:
                    logger.warning(
                        "Repairing history: stripping orphaned tool_calls %s",
                        needed_ids - found_ids,
                    )
                    orphaned_indices.add(i)
                    orphaned_tc_ids.update(needed_ids)

        if orphaned_indices:
            # Remove orphans and their partial results in reverse order
            to_remove = set()
            for i, msg in enumerate(history):
                if i in orphaned_indices:
                    to_remove.add(i)
                elif (msg.get("role") == "tool" and
                      msg.get("tool_call_id") in orphaned_tc_ids):
                    to_remove.add(i)

            for i in sorted(to_remove, reverse=True):
                history.pop(i)

    def cancel(self, room_id: str | None = None) -> asyncio.Task | None:
        """Cancel the in-flight turn for `room_id` and return it for awaiting.

        CORE-1: `room_id` is required in practice -- the `/stop` handler has
        it in scope. It stays optional only so an out-of-tree caller does not
        break; passing None cancels every in-flight turn, which is the right
        behaviour for process shutdown and is what `MatrixBot.shutdown` wants.
        It is NOT a "cancel the current one" fallback: there is no such thing
        when rooms run concurrently, and pretending there was is what let a
        `/stop` in one room kill another room's work.

        kdsn.330 D7: ``/stop`` also stops the room's BACKGROUND WORK. Async
        subagent dispatches are separate background tasks invisible to
        ``_current_tasks`` (one turn task per room), so they are cancelled
        from the ledger (``_dispatch_tasks``): each runner observes the
        cancellation as terminal ``cancelled`` (durable ledger entry +
        notice via its finalizer), so "stop stops the room's work" holds
        after async dispatch exists.

        The ``_dispatch_tasks`` reads are getattr-defensive (default
        empty): unit tests build ``Agent.__new__(Agent)`` with only
        ``_current_tasks`` seeded — pre-dispatch agents must keep the
        exact legacy cancel behaviour.
        """
        _dispatch_tasks = getattr(self, "_dispatch_tasks", {}) or {}
        if room_id is not None:
            task = self._current_tasks.get(room_id)
            if task:
                task.cancel()
            for dtask in list(_dispatch_tasks.get(room_id, {}).values()):
                dtask.cancel()
            return task

        last: asyncio.Task | None = None
        for task in list(self._current_tasks.values()):
            task.cancel()
            last = task
        for droom in _dispatch_tasks.values():
            for dtask in list(droom.values()):
                dtask.cancel()
        return last

    def _estimate_content_tokens(self, content: str | list[dict]) -> int:
        """Estimate tokens for a single content (string or list of blocks).

        Text: 1 token ≈ 4 chars.
        Images: base64 decoded size * 3/4 / 750.
        """
        if isinstance(content, str):
            return len(content) // 4

        if isinstance(content, list):
            total = 0
            for block in content:
                if block.get("type") == "text":
                    total += len(block.get("text", "")) // 4
                elif block.get("type") == "image":
                    # Estimate from base64 data length
                    base64_len = len(block.get("data", ""))
                    # base64 expands by 4/3, so decoded size = encoded * 3/4
                    decoded_bytes = base64_len * 3 // 4
                    total += int(decoded_bytes * IMAGE_TOKENS_PER_BYTE)
            return total

        return 0

    def _context_char_total(self, room_id: str = "_default",
                            history: list[dict] | None = None) -> int:
        """Char total of the context payload for a room (kdsn.329 single walk).

        system prompt + tool definitions + messages (the explicit `history`
        arg when given, else the room's in-memory history). This is the ONE
        walk both `_estimate_context_tokens` (heuristic = chars//4) and the
        pre-call guard's anchor-cursor capture consume — one walk, one
        semantics, no drift between the estimator and the anchor cursor.
        """
        total_chars = len(self.system_prompt)
        total_chars += self._tool_defs_chars
        msgs = history if history is not None else self.history(room_id)
        for msg in msgs:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # List of content blocks
                for block in content:
                    if block.get("type") == "text":
                        total_chars += len(block.get("text", ""))
                    elif block.get("type") == "image":
                        # Estimate from base64 data length
                        base64_len = len(block.get("data", ""))
                        decoded_bytes = base64_len * 3 // 4
                        total_chars += int(decoded_bytes * IMAGE_TOKENS_PER_BYTE * 4)
            # Tool calls have input dicts — estimate their JSON size
            for tc in msg.get("tool_calls", []):
                total_chars += len(str(tc.input)) + _TOOL_CALL_OVERHEAD_CHARS
            # Thinking blocks can be large — include in estimate
            for tb in msg.get("thinking", []):
                total_chars += len(tb.get("thinking", ""))
                total_chars += len(tb.get("signature", ""))
            # Wire envelope overhead for tool result messages
            if msg.get("role") == "tool":
                total_chars += _TOOL_RESULT_OVERHEAD_CHARS
        return total_chars

    def _estimate_context_tokens(self, room_id: str = "_default", history: list[dict] | None = None) -> int:
        """Estimate current context size in tokens for a room.

        Includes system prompt + room history. Uses 1 token ≈ 4 chars.
        If history is provided, use that instead of looking up by room_id
        (the merge below applies for ANY history arg — ONE behavior).

        kdsn.329: when the room carries a provider-usage token anchor
        (`_token_anchor[room_id]` = (true_prompt_tokens, char_cursor)), the
        estimate is anchored to the provider's last reported true prompt
        size plus marginal chars//4 growth since that call:
            heuristic = chars // 4
            anchored  = tokens0 if chars0 is None
                        else tokens0 + max(0, chars - chars0) // 4
            return max(heuristic, anchored)
        `max(0, ...)` clamps shrink-without-invalidation to the floor
        (fail-high); the outer max() means anchoring only ever RAISES
        estimates — tiers fire earlier, never later. Without an anchor the
        result is byte-identical to the legacy heuristic.
        """
        total_chars = self._context_char_total(room_id, history=history)
        heuristic = total_chars // 4
        anchor = self._token_anchor.get(room_id)
        if anchor is None:
            return heuristic
        tokens0, chars0 = anchor
        anchored = (tokens0 if chars0 is None
                    else tokens0 + max(0, total_chars - chars0) // 4)
        return max(heuristic, anchored)

    def _set_anchor_floor(self, room_id: str, tokens: int) -> None:
        """Restart-rehydration floor (kdsn.329): store (tokens, None).

        The merge reduces to max(heuristic, tokens) — safe after a wholesale
        history rebuild where no live payload cursor exists. The next live
        call replaces the floor with a full-cursor anchor via the record
        seam.

        Clamped to the room's AUTO threshold (audit H2): the JSONL is
        append-only, so a usage entry PREDATING a handoff boundary
        rehydrates a phantom full-runway floor. The floor's job is exactly
        one bit — "real pressure existed at last measurement" — not an exact
        count:
          - handoff ENABLED: floor == auto threshold trips the turn-start
            protective boundary (`>=`); the phantom buys a boundary, not a
            wedge.
          - handoff DISABLED / headless: floor == auto threshold < available,
            so the pre-flight raise (`> available`, inclusive of pending
            content) cannot fire before the first live call re-anchors with
            a full cursor. An unclamped floor would wedge the room:
            ContextOverflowError raises on every turn and no call ever
            succeeds to replace the phantom (audit H2: permanent brick the
            pre-fix heuristic could not produce).

        Fail-soft: a non-numeric/stubbed `tokens` must never raise through
        wake — ignore and let the first live call anchor normally.
        """
        if isinstance(tokens, bool) or not isinstance(tokens, (int, float)):
            logger.debug("_set_anchor_floor: ignoring non-numeric %r", tokens)
            return
        available = self._effective_available(self._resolve_model_limit(room_id))
        auto_th = int(available * self.config.context.auto_pct / 100)
        self._token_anchor[room_id] = (int(min(tokens, auto_th)), None)

    def status(self, room_id: str = "_default", history: list[dict] | None = None) -> dict:
        """Snapshot of agent state for operator visibility.

        If history is provided (e.g. from session_log.build_context()),
        use it for the context estimate instead of raw in-memory history.
        This ensures toolstrip-aware context sizes.
        """
        context_tokens = self._estimate_context_tokens(room_id, history=history)
        model_max = self._resolve_model_limit(room_id)
        context_pct = round(context_tokens / model_max * 100) if model_max else 0
        _u = self._usage_for(room_id)
        return {
            "name": self.config.name,
            "model": self.get_model(room_id),
            "turns": sum(1 for m in self.history(room_id) if m["role"] == "user"),
            "context_tokens": context_tokens,
            "context_max": model_max,
            "context_pct": context_pct,
            "context_remaining": max(0, model_max - context_tokens),
            "uncached_input_tokens": _u["uncached_input_tokens"],
            "cache_read_tokens": _u["cache_read_tokens"],
            "cache_creation_tokens": _u["cache_creation_tokens"],
            "total_output_tokens": _u["total_output_tokens"],
            "total_tool_calls": _u["total_tool_calls"],
            "main_cost_usd": _u["main_cost_usd"],
            "subagent_cost_usd": _u["subagent_cost_usd"],
            "advisor_cost_usd": _u["advisor_cost_usd"],
            "unpriced_tokens": _u["unpriced_tokens"],
            "total_cost_usd": _u["main_cost_usd"] + _u["subagent_cost_usd"] + _u["advisor_cost_usd"],
        }
