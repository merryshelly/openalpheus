"""Context GC — integration seams (workspace-kdsn.305.2, .3 + tools/matrix).

The tests are the specification. Scope of this file:
  - [context] TOML config parsing (Sub C: config.py) — superseded by
    tests/test_context_handoff_config.py (kdsn.322 T0; the config-surface
    pins moved there with the ContextHandoffConfig rename)
  - CONTINUITY.md prompt assembly, 8th operator file (Sub C: prompt.py + template)
  - handoff-checkpoint / handoff-runway reminder triggers,
    coexist-with-reset (Sub C: reminders.py)
  - context_handoff + set_active_project tools (Sub D: tools/__init__.py)
  - /cache handoff + /cache status + /project matrix commands (Sub D: matrix.py)

Sub-agent implementors NEVER modify this file. They MAY append new test
classes for the real-path agent-loop tests named in their brief (turn-start
auto tier, overflow-guard hard tier) — appended tests must follow the
canonical real-path pattern (tests/test_guidance_integration.py) and never
weaken assertions here.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from openalph.agent import Agent, ContextOverflowError
from openalph.config import AgentConfig, ContextHandoffConfig, MatrixConfig, ProviderConfig
from openalph.matrix import MatrixBot
from openalph.provider import Response, StreamEvent, ToolCall, Usage
from unittest.mock import patch
import json as _json
from openalph.session import SessionLog
from openalph.tools import ToolResult, BUILTIN_TOOLS, execute_tool
from openalph.reminders import ReminderEngine, ReminderState
from openalph.prompt import assemble_prompt

ROOM = "!gci:matrix.local"
AGENT_ID = "@gci-agent:matrix.local"
TAG_OPEN = "&lt;system-reminder&gt;"

WINDOW_262K = 262144
MAX_TOKENS_8K = 8192
USABLE_262K = WINDOW_262K - MAX_TOKENS_8K  # 253952


def _cfg(workspace, **kw):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[],
        )},
        workspace=workspace,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _state(**kw):
    defaults = dict(
        evaluation_point="turn_start",
        iteration=0,
        max_iterations=100,
        context_tokens=10000,
        context_limit=200000,
        completed_turns=0,
        turn_source=None,
        tool_calls_this_turn={},
        tool_calls_session={},
        todo_list=[],
        enabled_tools={"shell", "file_read"},
    )
    defaults.update(kw)
    return ReminderState(**defaults)


# ============================================================================
# CONTINUITY.md prompt assembly (8th operator file)
# ============================================================================

class TestContinuityPrompt:
    def _ws(self, tmp_path, with_file=True):
        (tmp_path / "SAFETY.md").write_text("# Safety\nstay safe")
        if with_file:
            (tmp_path / "CONTINUITY.md").write_text("# CONTINUITY\nmaintain progress.md")
        return tmp_path

    def test_workspace_file_used_when_enabled(self, tmp_path):
        p = assemble_prompt(self._ws(tmp_path), gc_enabled=True)
        assert "## CONTINUITY.md" in p
        assert "maintain progress.md" in p

    def test_packaged_template_fallback(self, tmp_path):
        p = assemble_prompt(self._ws(tmp_path, with_file=False), gc_enabled=True)
        assert "## CONTINUITY.md" in p
        # the packaged template teaches both artifacts
        assert "durable-set.toml" in p and "progress.md" in p

    def test_absent_when_disabled(self, tmp_path):
        p = assemble_prompt(self._ws(tmp_path), gc_enabled=False)
        assert "CONTINUITY.md" not in p

    def test_packaged_template_ships(self):
        import openalph.prompt as pp
        tpl = Path(pp.__file__).parent / "templates" / "CONTINUITY.md"
        assert tpl.exists(), "openalph must ship templates/CONTINUITY.md as package data"
        body = tpl.read_text()
        assert "durable-set.toml" in body and "progress.md" in body
        assert "set_active_project" in body


# ============================================================================
# handoff-checkpoint / handoff-runway reminder triggers (coexist-with-reset)
# ============================================================================

class TestGCTriggers:
    def _eng(self, tmp_path):
        return ReminderEngine(_cfg(tmp_path))

    def test_handoff_checkpoint_fires_at_threshold_turn_start(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(context_tokens=USABLE_262K, context_limit=WINDOW_262K,
                    checkpoint_threshold=int(USABLE_262K * 0.75))
        out = eng.evaluate(st)
        warns = [r for r in out if r.trigger == "handoff-checkpoint"]
        assert len(warns) == 1
        assert "durable-set" in warns[0].text
        assert "checkpoint" in warns[0].text.lower()

    def test_handoff_checkpoint_silent_below_threshold(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(context_tokens=10000, checkpoint_threshold=int(USABLE_262K * 0.75))
        assert [r for r in eng.evaluate(st) if r.trigger == "handoff-checkpoint"] == []

    def test_handoff_checkpoint_silent_when_unknown(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(context_tokens=999999, checkpoint_threshold=0)
        assert [r for r in eng.evaluate(st) if r.trigger == "handoff-checkpoint"] == []

    def test_handoff_checkpoint_once_per_cycle(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(context_tokens=USABLE_262K, checkpoint_threshold=1000)
        assert eng.evaluate(st)
        assert [r for r in eng.evaluate(st) if r.trigger == "handoff-checkpoint"] == []

    # kdsn.305.12 D5: the durable-budget trigger (durable-set usage >= 50%
    # of budget) is REMOVED — replaced by handoff-runway (post-boundary
    # runway consumption >= 90%, once/session); subsumed by the red suite.
    #
    # kdsn.322 deletions (behavior retired, not renamed):
    #  - the enabled-tools-gated tool-mention note in the directive text is
    #    DELETED (spec §4 feedback item 4 — the directive is a fixed
    #    template that names no tool);
    #  - the turn_start-only pin is DELETED (spec §3.3 — the checkpoint
    #    tier now evaluates at BOTH evaluation points; dual-point firing is
    #    pinned by tests/test_context_handoff_checkpoint.py).

    def test_triggers_rehydrate(self, tmp_path):
        eng = self._eng(tmp_path)
        eng.rehydrate([
            {"role": "user", "source": "reminder",
             "trigger": "handoff-checkpoint",
             "content": f"{TAG_OPEN}\nfinal durable-set\n{TAG_OPEN.replace('<', '</')}"},
        ])
        st = _state(context_tokens=USABLE_262K, checkpoint_threshold=1000)
        assert [r for r in eng.evaluate(st) if r.trigger == "handoff-checkpoint"] == []

    def test_reset_rearms_both(self, tmp_path):
        eng = self._eng(tmp_path)
        eng.evaluate(_state(context_tokens=USABLE_262K, checkpoint_threshold=1000,
                            handoff_runway_fraction=0.95))
        eng.reset()
        assert eng.evaluate(_state(context_tokens=USABLE_262K, checkpoint_threshold=1000))
        assert eng.evaluate(_state(handoff_runway_fraction=0.95))


# ============================================================================
# context_handoff + set_active_project tools
# ============================================================================

class TestHandoffTool:
    def test_registered(self):
        assert "context_handoff" in BUILTIN_TOOLS
        schema = BUILTIN_TOOLS["context_handoff"]["parameters"]
        assert schema["type"] == "object"
        assert not schema.get("required")

    def test_missing_callback_steers_to_slash(self):
        res = asyncio.new_event_loop().run_until_complete(
            execute_tool("context_handoff", {}, None, {"room_id": ROOM}))
        # headless/no-wiring: clean is_error, names the operator command
        assert res.is_error and "/cache handoff" in res.content

    def test_sub_sentinel_refused(self):
        res = asyncio.new_event_loop().run_until_complete(
            execute_tool("context_handoff", {}, None,
                         {"room_id": "__sub__",
                          "apply_handoff_boundary": AsyncMock(return_value={})}))
        assert res.is_error and "subagent" in res.content.lower()

    def test_success_returns_summary(self):
        manifest = {
            "boundary_index": 12, "trigger": "tool",
            "classes": {"tools": 3, "thinking": 2, "inputs": 0},
            "tokens_before": 90000, "tokens_after_est": 40000,
            "durable": {"project": "foo", "budget_tokens": 48000,
                        "used_tokens": 12000, "over_budget": False},
        }
        cb = AsyncMock(return_value={"applied": True, "noop_reason": None,
                                     "manifest": manifest, "over_budget": False})
        res = asyncio.new_event_loop().run_until_complete(
            execute_tool("context_handoff", {}, None,
                         {"room_id": ROOM, "apply_handoff_boundary": cb}))
        assert not res.is_error
        # kdsn.322.14: the tool result carries the COMPOSITE before-figure
        # (system prompt + tool defs + render) and the MEASURED after —
        # the old pinned-after (40000) was the retired tokens_after_est=0
        # era's estimate figure.
        assert "12" in res.content and "~90,000" in res.content
        assert "read via file_read" in res.content and "(est.)" not in res.content
        cb.assert_awaited_once()
        assert cb.call_args.kwargs.get("trigger", cb.call_args.args[-1] if cb.call_args.args else None) in (None, "tool")

    def test_noop_is_error_with_reason(self):
        cb = AsyncMock(return_value={"applied": False,
                                     "noop_reason": "cooldown: 1 of 3 turns since last boundary",
                                     "manifest": None, "over_budget": False})
        res = asyncio.new_event_loop().run_until_complete(
            execute_tool("context_handoff", {}, None,
                         {"room_id": ROOM, "apply_handoff_boundary": cb}))
        assert res.is_error
        assert "cooldown" in res.content


class TestSetActiveProjectTool:
    def test_registered(self):
        assert "set_active_project" in BUILTIN_TOOLS
        assert BUILTIN_TOOLS["set_active_project"]["parameters"]["required"] == ["project"]

    def _run(self, project, ret):
        cb = AsyncMock(return_value=ret)
        res = asyncio.new_event_loop().run_until_complete(
            execute_tool("set_active_project", {"project": project}, None,
                         {"room_id": ROOM, "set_active_project": cb}))
        return res, cb

    def test_success(self):
        res, cb = self._run("foo", {"ok": True,
                                    "text": "Project set to foo. Durable entries: 2"})
        assert not res.is_error
        assert "foo" in res.content
        cb.assert_awaited_once_with("foo")

    def test_already_declared_errors_naming_project(self):
        res, _ = self._run("bar", {"ok": False,
                                   "text": "Project 'foo' already declared for this room"})
        assert res.is_error and "foo" in res.content

    def test_missing_callback_steers(self):
        res = asyncio.new_event_loop().run_until_complete(
            execute_tool("set_active_project", {"project": "foo"}, None,
                         {"room_id": ROOM}))
        assert res.is_error

    def test_non_string_input_refused(self):
        res = asyncio.new_event_loop().run_until_complete(
            execute_tool("set_active_project", {"project": ["foo"]}, None,
                         {"room_id": ROOM, "set_active_project": AsyncMock()}))
        assert res.is_error

    def test_missing_param_refused(self):
        res = asyncio.new_event_loop().run_until_complete(
            execute_tool("set_active_project", {}, None,
                         {"room_id": ROOM, "set_active_project": AsyncMock()}))
        assert res.is_error


# ============================================================================
# Matrix commands: /cache handoff, /cache status, /project
# ============================================================================

def _event(body):
    ev = MagicMock()
    ev.body = body
    ev.sender = "@sb:matrix.local"
    ev.event_id = "$ev1"
    ev.source = {"content": {"m.relates_to": None}}
    return ev


class TestMatrixCommands:
    def _bot(self, tmp_path):
        """Real MatrixBot shell with REAL SessionLog and REAL Agent (mocked nio).

        Fixture precondition note for implementors: adapt construction to the
        canonical real-path pattern (tests/test_guidance_integration.py) — the
        assertions below are the contract, not the construction mechanics.
        """
        from openalph.matrix import MatrixBot
        from openalph.agent import Agent
        from openalph.config import MatrixConfig

        ws = tmp_path
        (ws / "skills").mkdir(exist_ok=True)
        config = _cfg(ws)
        agent = Agent(config)
        mcfg = MatrixConfig(
            homeserver="https://matrix.local", user_id=AGENT_ID,
            device_id="TEST", password="p", access_token=None,
            context_reserve=16384, sync_timeout=30000,
            retry_base=1, retry_max=10,
        )
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = mcfg
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$r1"))
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = set()
        bot._room_effort = {}
        bot._room_cache_ttl = {}
        bot._room_timesense = {}
        bot._halted_rooms = set()
        bot._background_tasks = set()
        bot._session_locks = {}
        bot.session_log = SessionLog(ws, AGENT_ID)
        bot.heartbeat = None
        bot.umbral = None
        bot._degraded_provider_notice = {}
        return bot, agent

    def _send_texts(self, bot):
        out = []
        for c in bot.client.room_send.call_args_list:
            # room_send(room_id, "m.room.message", content) — content is the
            # 3rd positional in production style; tolerate kwarg form too.
            content = c.kwargs.get("content")
            if content is None and len(c.args) >= 3:
                content = c.args[2]
            content = content or {}
            if content.get("msgtype") in ("m.text", "m.notice"):
                out.append(content.get("body", ""))
        return out

    def test_cache_handoff_applies_boundary_and_rebuilds_in_place(self, tmp_path):
        bot, agent = self._bot(tmp_path)
        log = bot.session_log
        for e in [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "call_id": "c1", "name": "shell", "output": "o" * 4000},
        ]:
            log.append(room=ROOM, sender=AGENT_ID, **e)
        agent._rooms[ROOM] = list(agent.history(ROOM)) or []
        history_obj = agent.history(ROOM)
        # seed in-memory history so identity is observable
        history_obj.extend(log.build_context(ROOM))
        ident_before = id(history_obj)

        room = MagicMock()
        room.room_id = ROOM
        asyncio.new_event_loop().run_until_complete(
            bot._handle_room_message(room, _event("/cache handoff")))

        entries = log.read(ROOM)
        assert any(e.get("event") == "handoff_boundary" for e in entries)
        assert any(e.get("source") == "handoff_snapshot" for e in entries)
        assert id(agent.history(ROOM)) == ident_before, \
            "in-memory history must be rebuilt IN PLACE (same object identity)"
        sent = "\n".join(self._send_texts(bot))
        assert "GC" in sent or "boundary" in sent

    def test_cache_status_shows_boundary(self, tmp_path):
        bot, agent = self._bot(tmp_path)
        log = bot.session_log
        log.append(room=ROOM, sender=AGENT_ID, role="user", content="q")
        log.append(room=ROOM, sender=AGENT_ID, role="system",
                   event="handoff_boundary", entry_index=1, detail=json.dumps({
                       "ts": "T", "boundary_index": 1, "trigger": "slash",
                       "tokens_before": 10, "tokens_after_est": 0,
                       "durable": {}, "errors": []}))
        room = MagicMock()
        room.room_id = ROOM
        asyncio.new_event_loop().run_until_complete(
            bot._handle_room_message(room, _event("/cache")))
        sent = "\n".join(self._send_texts(bot))
        assert "boundary" in sent and "1" in sent

    def test_project_set_and_status(self, tmp_path):
        bot, agent = self._bot(tmp_path)
        room = MagicMock()
        room.room_id = ROOM
        loop = asyncio.new_event_loop()
        loop.run_until_complete(bot._handle_room_message(room, _event("/project set foo")))
        entries = bot.session_log.read(ROOM)
        assert any(e.get("event") == "active_project" and e.get("detail") == "foo"
                   for e in entries)
        loop.run_until_complete(bot._handle_room_message(room, _event("/project")))
        sent = "\n".join(self._send_texts(bot))
        assert "foo" in sent
        # operator override is allowed (explicit authority) and announced
        loop.run_until_complete(bot._handle_room_message(room, _event("/project set bar")))
        entries = bot.session_log.read(ROOM)
        assert any(e.get("event") == "active_project" and e.get("detail") == "bar"
                   for e in entries)


# ============================================================================
# audit-fix (kdsn.322.9): /cache slash surface behavior pins
# ============================================================================

class TestSlashAuditFixes:
    def _bot(self, tmp_path):
        """Same real MatrixBot shell as TestMatrixCommands._bot."""
        from openalph.matrix import MatrixBot
        from openalph.agent import Agent
        from openalph.config import MatrixConfig

        ws = tmp_path
        (ws / "skills").mkdir(exist_ok=True)
        config = _cfg(ws)
        agent = Agent(config)
        mcfg = MatrixConfig(
            homeserver="https://matrix.local", user_id=AGENT_ID,
            device_id="TEST", password="p", access_token=None,
            context_reserve=16384, sync_timeout=30000,
            retry_base=1, retry_max=10,
        )
        bot = MatrixBot.__new__(MatrixBot)
        bot.config = mcfg
        bot.agent = agent
        bot.client = MagicMock()
        bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$r1"))
        bot.client.room_typing = AsyncMock()
        bot._current_room = None
        bot._synced = True
        bot._active_rooms = set()
        bot._room_effort = {}
        bot._room_cache_ttl = {}
        bot._room_timesense = {}
        bot._halted_rooms = set()
        bot._background_tasks = set()
        bot._session_locks = {}
        bot.session_log = SessionLog(ws, AGENT_ID)
        bot.heartbeat = None
        bot.umbral = None
        bot._degraded_provider_notice = {}
        return bot, agent

    def _seeded_room(self, bot, agent):
        log = bot.session_log
        for e in [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "call_id": "c1", "name": "shell",
             "output": "o" * 4000},
        ]:
            log.append(room=ROOM, sender=AGENT_ID, **e)
        agent._rooms[ROOM] = list(agent.history(ROOM)) or []
        agent.history(ROOM).extend(log.build_context(ROOM))
        return log

    @staticmethod
    def _send_contents(bot):
        out = []
        for c in bot.client.room_send.call_args_list:
            content = c.kwargs.get("content")
            if content is None and len(c.args) >= 3:
                content = c.args[2]
            content = content or {}
            if content.get("msgtype") in ("m.text", "m.notice"):
                out.append(content)
        return out

    def test_handoff_flag_off_emits_notice_and_applies_nothing(self, tmp_path):
        """audit-fix (kdsn.322.9): the flag-off legacy toolstrip branch was a
        zombie — under the hard epoch a toolstrip marker is ignored by
        build_context, so NOTHING would be stripped while the legacy message
        told the operator it was. The branch must emit a notice and apply
        NOTHING: no marker append, no legacy rebuild."""
        from openalph.config import ContextHandoffConfig

        bot, agent = self._bot(tmp_path)
        agent.config.context = ContextHandoffConfig(handoff_enabled=False)
        log = self._seeded_room(bot, agent)
        n_before = len(log.read(ROOM))
        history_before = list(agent.history(ROOM))
        room = MagicMock()
        room.room_id = ROOM
        asyncio.new_event_loop().run_until_complete(
            bot._handle_room_message(room, _event("/cache handoff")))
        entries = log.read(ROOM)
        assert not any(e.get("event") == "toolstrip" for e in entries), (
            "hard epoch: a legacy toolstrip marker must NOT be appended — "
            "build_context ignores it, so the strip would be a lie")
        assert not any(e.get("event") == "handoff_boundary" for e in entries)
        assert len(entries) == n_before, "flag-off branch applies NOTHING"
        assert list(agent.history(ROOM)) == history_before, (
            "no legacy history rebuild on the flag-off path")
        sent = "\n".join(c.get("body", "") for c in self._send_contents(bot))
        assert "handoff is disabled" in sent
        assert "hard epoch" in sent

    def test_cache_gc_returns_error_usage_response(self, tmp_path):
        """audit-fix (kdsn.322.9, spec 3.4): /cache gc returns an is_error
        usage response naming the deprecation and steering to /cache
        handoff — the established error shape for slash errors (an
        m.text reply via self.send, like the other usage errors), not a
        bare m.notice. NO boundary of any kind is applied."""
        bot, agent = self._bot(tmp_path)
        log = self._seeded_room(bot, agent)
        n_before = len(log.read(ROOM))
        room = MagicMock()
        room.room_id = ROOM
        asyncio.new_event_loop().run_until_complete(
            bot._handle_room_message(room, _event("/cache gc")))
        dep = [c for c in self._send_contents(bot)
               if "deprecated" in c.get("body", "")]
        assert dep, "the deprecation response must reach the room"
        body = dep[0].get("body", "")
        assert "/cache handoff" in body, "steers to the new command"
        assert dep[0].get("msgtype") == "m.text", (
            "is_error usage responses are operator replies (m.text), not "
            "bare notices")
        entries = log.read(ROOM)
        assert len(entries) == n_before, "no boundary is applied"
        assert not any(e.get("event") in ("handoff_boundary", "toolstrip")
                       for e in entries)

    def test_cache_handoff_trigger_is_slash(self, tmp_path):
        """audit-fix (kdsn.322.9, spec 3.1): the /cache handoff path emits
        trigger='slash' (the spec §3.1 vocabulary is auto|tool|slash|
        exec-auto), not 'manual'. The operator-visible confirmation uses
        the same vocabulary."""
        bot, agent = self._bot(tmp_path)
        self._seeded_room(bot, agent)
        log = bot.session_log
        room = MagicMock()
        room.room_id = ROOM
        asyncio.new_event_loop().run_until_complete(
            bot._handle_room_message(room, _event("/cache handoff")))
        markers = [e for e in log.read(ROOM)
                   if e.get("event") == "handoff_boundary"]
        assert markers, "the boundary must be applied"
        manifest = json.loads(markers[-1].get("detail", "{}"))
        assert manifest.get("trigger") == "slash", (
            "spec 3.1 vocabulary: the slash path emits 'slash'")
        sent = "\n".join(c.get("body", "")
                         for c in self._send_contents(bot))
        assert "applied (slash)" in sent, (
            "the operator confirmation names the spec trigger")

    def test_churn_guard_rearm_only_on_cleared(self, tmp_path):
        """audit-fix (kdsn.322.9): the applied-boundary seam must re-arm the
        auto tier's churn latch ONLY when the boundary actually cleared the
        auto threshold. The unconditional pop let a mid-turn hard/tool/slash
        boundary re-arm a latched auto tier whose wedge state (durable
        package >= threshold) persists — one boundary + snapshot per turn,
        unbounded JSONL growth."""
        bot, agent = self._bot(tmp_path)
        room = ROOM
        agent.config.model_max_tokens = 8000
        agent.config.max_tokens = 100
        # Threshold must sit BETWEEN the empty-history base (system prompt +
        # tool defs — real context the boundary cannot strip) and base plus
        # the seeded message: base < threshold < base + message.
        base = agent._estimate_context_tokens(room)
        available = agent._effective_available(8000)
        agent.config.context.auto_pct = min(
            99.0, (base + 500) * 100.0 / available)
        # Wedge state: latch set, live history above the auto threshold.
        agent._gc_auto_uncleared[room] = True
        agent.history(room).append({"role": "user", "content": "x" * 4000})
        agent._note_handoff_boundary_applied(
            room, {"applied": True, "manifest": None})
        assert agent._gc_auto_uncleared.get(room) is True, (
            "a boundary that did NOT clear the auto threshold must leave "
            "the churn latch set (re-arm only on cleared)")
        # A later boundary that clears the threshold re-arms the tier.
        agent.history(room).clear()
        agent._note_handoff_boundary_applied(
            room, {"applied": True, "manifest": None})
        assert not agent._gc_auto_uncleared.get(room), (
            "a boundary that cleared the auto threshold must re-arm the "
            "auto tier (pop the churn latch)")

# ============================================================================
# Real-path agent-loop pinning (workspace-kdsn.305.2, authored post-305 build).
# Canonical pattern per tests/test_guidance_integration.py: REAL Agent + REAL
# SessionLog + REAL callbacks (build_callbacks seam); ONLY the provider stream
# and the nio client are mocked. These tests exist because a full green suite
# missed an unawaited coroutine at the in-loop hard tier — the wiring bug was
# invisible to every seam-mocked test (tool-management, "the one lesson").
# ============================================================================

GC_ROOM = "!gcloop:matrix.local"
AGENT_ID = "@gci-agent:matrix.local"


def _cfg(workspace, **kw):
    """Shorthand AgentConfig builder (same shape as the sibling suites)."""
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[],
        )},
        workspace=workspace,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _gc_real_bot(tmp_path, *, model_max_tokens, tools=("shell",)):
    """Real MatrixBot + real Agent + real SessionLog; nio client mocked.

    Mirrors TestMatrixCommands._bot and the guidance-integration canonical
    fixture. Workspace is minimal (no prompt files) so seeded tool outputs
    dominate the token estimate deterministically.
    """
    ws = tmp_path
    (ws / "tools").mkdir(exist_ok=True)
    for name in tools:
        (ws / "tools" / f"{name}.toml").write_text("[config]\n")
    config = _cfg(ws, model_max_tokens=model_max_tokens)
    agent = Agent(config)
    bot = MatrixBot.__new__(MatrixBot)
    bot.config = MatrixConfig(
        homeserver="https://matrix.local", user_id=AGENT_ID, device_id="TEST",
        password="p", access_token=None, context_reserve=16384,
        sync_timeout=30000, retry_base=1, retry_max=10,
    )
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$r1"))
    bot.client.room_typing = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = set()
    bot._room_effort = {}
    bot._room_cache_ttl = {}
    bot._room_timesense = {}
    bot._halted_rooms = set()
    bot._background_tasks = set()
    bot._session_locks = {}
    bot._degraded_provider_notice = {}
    bot.session_log = SessionLog(ws, AGENT_ID)
    bot.heartbeat = None
    bot.umbral = None
    return bot, agent


def _gc_stream(state):
    """Stream factory: call 1 emits a shell tool_use; call 2+ emits text.
    The mocked tool execution (see tests) injects the bulk into history."""
    calls = [0]

    async def _stream(*, config=None, system=None, messages=None, tools=None,
                      model="test", thinking=None, cache_ttl=None, **kw):
        calls[0] += 1
        state["stream_calls"] = calls[0]
        state.setdefault("payloads", []).append(list(messages))
        if calls[0] == 1:
            tc = ToolCall(id="gc_tc_1", name="shell",
                          input={"command": "bloat"})
            yield StreamEvent(type="tool_done", tool_index=0, tool_call=tc)
            yield StreamEvent(
                type="done",
                response=Response(content="", tool_calls=[tc], model=model,
                                  usage=Usage(input_tokens=10, output_tokens=5),
                                  stop_reason="tool_use"),
                stop_reason="tool_use", model=model)
        else:
            yield StreamEvent(type="text", content="DONE")
            yield StreamEvent(
                type="done",
                response=Response(content="DONE", model=model,
                                  usage=Usage(input_tokens=10, output_tokens=5),
                                  stop_reason="end_turn"),
                stop_reason="end_turn", model=model)

    return _stream


def _seed_room(bot, big_chars=60000, count=8):
    """Seed the room JSONL with big tool outputs (below the 64KB overflow
    threshold so they persist inline) + refresh in-memory history from disk."""
    log = bot.session_log
    log.wipe(GC_ROOM)
    log.append(role="user", sender=AGENT_ID, room=GC_ROOM, content="go")
    for i in range(count):
        log.append(role="assistant", sender=AGENT_ID, room=GC_ROOM, content="",
                   tool_calls=[{"call_id": f"seed_{i}", "name": "shell",
                                "input": {"command": f"seed {i}"}}])
        log.append(role="tool", sender=AGENT_ID, room=GC_ROOM,
                   call_id=f"seed_{i}", name="shell", output="x" * big_chars)
    h = bot.agent.history(GC_ROOM)
    h.clear()
    h.extend(log.build_context(GC_ROOM))


def _big_outputs(history, size):
    return sum(1 for m in history
               if m.get("role") == "tool" and len(m.get("content", "")) == size)


def _run_turn(bot):
    # Production shape (kdsn.322.15): _inject_heartbeat persists the
    # directive to the JSONL BEFORE the turn runs — the turn-start tier
    # protects the pending input, so the directive must be the last JSONL
    # entry, as it always is in production. Driving _run_heartbeat_turn
    # bare leaves no pending entry and the protection would land on the
    # last seed entry instead (fixture artifact, not production shape).
    bot.session_log.append(role="system", sender=AGENT_ID, room=GC_ROOM,
                           content="[Automated heartbeat]",
                           source="heartbeat")
    return asyncio.run(
        bot._run_heartbeat_turn(GC_ROOM, "[Automated heartbeat]",
                                turn_source="heartbeat"))


async def _huge_tool(*args, name=None, input=None, **kw):
    return ToolResult(content="y" * 200000, is_error=False)


class TestGCAutoTierRealPath:
    def test_auto_tier_applies_boundary_at_threshold_real_path(self, tmp_path):
        """Turn-start estimate over the auto threshold (85% usable) but under
        the hard limit: the REAL loop applies a boundary (trigger=auto),
        rebuilds history IN PLACE (object identity stable), pointer-izes the
        big outputs, and the turn completes."""
        # 8×60000 chars ≈ 30000 tokens of seed bulk. Window 150000,
        # max_tokens 8192 → usable 141808, auto ≈ 120537. System prompt +
        # tool defs + seed ≈ 33000+ tokens → over auto, under hard.
        bot, agent = _gc_real_bot(tmp_path, model_max_tokens=150000)
        _seed_room(bot)
        h_before = agent.history(GC_ROOM)
        ident_before = id(h_before)
        assert _big_outputs(h_before, 60000) >= 6, \
            "fixture must seed enough bulk to cross the threshold"
        est = agent._estimate_context_tokens(GC_ROOM)
        usable = agent._effective_available(150000)
        auto_th = int(usable * agent.config.context.auto_pct / 100)
        assert est >= auto_th and est <= usable, (est, auto_th, usable)

        state = {}
        with patch("openalph.agent.stream", _gc_stream(state)):
            _run_turn(bot)

        marks = [e for e in bot.session_log.read(GC_ROOM)
                 if e.get("role") == "system" and e.get("event") == "handoff_boundary"]
        assert len(marks) == 1
        assert _json.loads(marks[0]["detail"])["trigger"] == "auto"
        assert id(agent.history(GC_ROOM)) == ident_before, \
            "history must be rebuilt IN PLACE (same object identity)"
        h_after = agent.history(GC_ROOM)
        assert _big_outputs(h_after, 60000) == 0
        # Full strip (kdsn.322): pre-boundary bulk DROPPED wholesale, the
        # snapshot is the only carryover, no pointer placeholders exist.
        assert any("durable context snapshot" in str(m.get("content", ""))
                   for m in h_after)
        assert not any("expunged" in str(m.get("content", ""))
                       for m in h_after)
        # turn completed: second stream call happened, its text was delivered
        assert state.get("stream_calls", 0) >= 2
        sent = []
        for c in bot.client.room_send.call_args_list:
            content = c.kwargs.get("content")
            if content is None and len(c.args) >= 3:
                content = c.args[2]
            sent.append(str((content or {}).get("body", "")))
        assert any("DONE" in s for s in sent)

    def test_auto_tier_optout_no_boundary_when_gc_disabled(self, tmp_path):
        """handoff_enabled=False: same crossing seed, NO boundary, outputs intact."""
        bot, agent = _gc_real_bot(tmp_path, model_max_tokens=150000)
        agent.config.context = ContextHandoffConfig(handoff_enabled=False)
        _seed_room(bot)
        h_before = agent.history(GC_ROOM)
        ident_before = id(h_before)
        state = {}
        with patch("openalph.agent.stream", _gc_stream(state)):
            _run_turn(bot)
        assert not any(e.get("event") == "handoff_boundary"
                       for e in bot.session_log.read(GC_ROOM))
        assert id(agent.history(GC_ROOM)) == ident_before
        assert _big_outputs(agent.history(GC_ROOM), 60000) >= 6, \
            "flag-off must leave seeded outputs untouched"


class TestGCHardTierRealPath:
    def _trip_setup(self, tmp_path):
        """Seed well UNDER the auto threshold at turn start (pre-append guard
        and auto tier stay quiet) — the overflow is pushed IN-LOOP by a mocked
        huge tool result, tripping only the pre-LLM-call guard."""
        # window 30000, max_tokens 8192 → usable 21808, auto ≈ 18536.
        # Small seed (~2K tokens). In-loop: the mocked 200000-char tool result
        # must SURVIVE the framework's truncation_limit cut before the guard
        # sees it — raise truncation_limit above the injected size.
        bot, agent = _gc_real_bot(tmp_path, model_max_tokens=30000)
        agent.config.truncation_limit = 210000
        _seed_room(bot, big_chars=1000, count=1)
        return bot, agent

    def test_hard_tier_applies_boundary_when_inloop_guard_trips(self, tmp_path):
        bot, agent = self._trip_setup(tmp_path)
        usable = agent._effective_available(30000)
        auto_th = int(usable * agent.config.context.auto_pct / 100)
        assert agent._estimate_context_tokens(GC_ROOM) < auto_th, \
            "fixture: turn start must be under the auto threshold"

        state = {}
        with patch("openalph.agent.stream", _gc_stream(state)), \
             patch("openalph.agent.execute_tool", _huge_tool):
            _run_turn(bot)

        marks = [e for e in bot.session_log.read(GC_ROOM)
                 if e.get("role") == "system" and e.get("event") == "handoff_boundary"]
        assert marks, "in-loop guard trip must apply a hard boundary"
        assert _json.loads(marks[-1]["detail"])["trigger"] == "hard"
        assert state.get("stream_calls", 0) >= 2, "turn must continue to completion"
        payloads = state.get("payloads", [])
        assert len(payloads) >= 2
        second_texts = [str(m.get("content", "")) for m in payloads[1]
                        if isinstance(m.get("content"), str)]
        # Full strip: the continued turn carries the snapshot, not the
        # pointer-ized bulk (no placeholders) and not the huge in-loop result.
        assert any("durable context snapshot" in t2 for t2 in second_texts), \
            "the continued turn must see the handoff snapshot"
        assert not any("expunged" in t2 for t2 in second_texts)
        assert not any("y" * 1000 in t2 for t2 in second_texts)

    def test_hard_tier_three_strikes_then_raises(self, tmp_path):
        bot, agent = self._trip_setup(tmp_path)
        assert agent._gc_fail_strikes.get(GC_ROOM, 0) == 0

        import openalph.callbacks as cbmod
        attempts = [0]

        # NOTE: sync — matches the real apply_boundary_and_rebuild contract
        # (the seam closure returns its result verbatim; an async stand-in
        # here would leak an unawaited coroutine into _gc_apply_boundary).
        def _counting_failure(*a, **kw):
            attempts[0] += 1
            return {"applied": False, "noop_reason": "injected failure",
                    "manifest": None, "over_budget": False}

        # Three consecutive trips across three turns. The fixture is re-seeded
        # between turns so the PRE-APPEND guard stays quiet and only the
        # in-loop guard trips; the GC callback is forced to fail every time.
        for _turn in range(3):
            _seed_room(bot, big_chars=1000, count=1)
            with patch("openalph.agent.stream", _gc_stream({})), \
                 patch("openalph.agent.execute_tool", _huge_tool), \
                 patch.object(cbmod, "apply_boundary_and_rebuild",
                              _counting_failure):
                with pytest.raises(ContextOverflowError):
                    _run_turn(bot)
        assert agent._gc_fail_strikes.get(GC_ROOM) == 3, \
            "three consecutive failed attempts must accumulate 3 strikes"
        assert attempts[0] == 3, \
            "the breaker permits exactly 3 attempts (MAX_CONSECUTIVE_FAILURES=3 pattern)"
        # 4th trip: breaker engaged — the attempt is SKIPPED (attempts frozen),
        # the raise still happens (pre-GC behavior).
        _seed_room(bot, big_chars=1000, count=1)
        with patch("openalph.agent.stream", _gc_stream({})), \
             patch("openalph.agent.execute_tool", _huge_tool), \
             patch.object(cbmod, "apply_boundary_and_rebuild", _counting_failure):
            with pytest.raises(ContextOverflowError):
                _run_turn(bot)
        assert attempts[0] == 3, \
            "the breaker must skip the attempt once 3 consecutive failures are banked"
        assert agent._gc_fail_strikes.get(GC_ROOM) == 3

        # Recovery: the breaker gates ONLY the hard tier. The turn-start AUTO
        # tier is not strike-gated, and any hard-trip context also crosses the
        # auto threshold — so a banked room recovers at the next turn start
        # (one-turn delay, never a permanent deadlock). Seed big enough to
        # cross auto at turn start; the auto boundary resets the counter, and
        # the in-loop trip then runs the hard tier with a clean slate.
        _seed_room(bot, big_chars=60000, count=8)
        state = {}
        with patch("openalph.agent.stream", _gc_stream(state)), \
             patch("openalph.agent.execute_tool", _huge_tool):
            _run_turn(bot)
        assert state.get("stream_calls", 0) >= 2, "recovery turn must complete"
        assert agent._gc_fail_strikes.get(GC_ROOM, 0) == 0, \
            "an applied boundary (any trigger) must reset the strike counter"
        marks = [e for e in bot.session_log.read(GC_ROOM)
                 if e.get("role") == "system" and e.get("event") == "handoff_boundary"]
        assert _json.loads(marks[-1]["detail"])["trigger"] in ("auto", "hard")
