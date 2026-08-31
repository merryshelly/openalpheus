"""Context GC — integration seams (workspace-kdsn.305.2, .3 + tools/matrix).

The tests are the specification. Scope of this file:
  - [context] TOML config parsing (Sub C: config.py)
  - CONTINUITY.md prompt assembly, 8th operator file (Sub C: prompt.py + template)
  - gc-warn / gc-budget reminder triggers, coexist-with-reset (Sub C: reminders.py)
  - context_gc + set_active_project tools (Sub D: tools/__init__.py)
  - /cache gc + /cache status + /project matrix commands (Sub D: matrix.py)

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

from openalph.config import AgentConfig, ProviderConfig, ConfigError, load_config
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


def _toml(tmp_path, body):
    p = tmp_path / "agent.toml"
    p.write_text(body)
    return load_config(p)


BASE_TOML = '''
[agent]
name = "gc-agent"
default_model = "anthropic/claude-sonnet-4-20250514"
max_tokens = 8192

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
'''


# ============================================================================
# [context] config section
# ============================================================================

class TestConfigContextSection:
    def test_absent_section_full_defaults(self, tmp_path):
        cfg = _toml(tmp_path, BASE_TOML)
        c = cfg.context
        assert c.gc_enabled is True
        assert c.warn_pct == 75
        assert c.auto_pct == 85
        assert c.hard_pct == 92
        assert c.durable_budget_pct == 15.0
        assert c.durable_budget_min_tokens == 48000
        assert c.durable_paths == []
        assert c.turn_cooldown == 3

    def test_overrides(self, tmp_path):
        body = BASE_TOML + '''
[context]
gc_enabled = false
warn_pct = 70
auto_pct = 80
hard_pct = 90
durable_budget_pct = 20.0
durable_budget_min_tokens = 24000
durable_paths = ["skills/*.md"]
turn_cooldown = 5
'''
        c = _toml(tmp_path, body).context
        assert c.gc_enabled is False
        assert c.warn_pct == 70 and c.auto_pct == 80 and c.hard_pct == 90
        assert c.durable_budget_pct == 20.0
        assert c.durable_budget_min_tokens == 24000
        assert c.durable_paths == ["skills/*.md"]
        assert c.turn_cooldown == 5

    def test_invalid_type_fails_loud(self, tmp_path):
        with pytest.raises(ConfigError):
            _toml(tmp_path, BASE_TOML + '[context]\ngc_enabled = "yes"\n')

    def test_pct_out_of_range_fails_loud(self, tmp_path):
        with pytest.raises(ConfigError):
            _toml(tmp_path, BASE_TOML + "[context]\nauto_pct = 150\n")

    def test_negative_budget_fails_loud(self, tmp_path):
        with pytest.raises(ConfigError):
            _toml(tmp_path, BASE_TOML + "[context]\ndurable_budget_min_tokens = -1\n")


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
# gc-warn / gc-budget reminder triggers (coexist-with-reset)
# ============================================================================

class TestGCTriggers:
    def _eng(self, tmp_path):
        return ReminderEngine(_cfg(tmp_path))

    def test_gc_warn_fires_at_threshold_turn_start(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(context_tokens=USABLE_262K, context_limit=WINDOW_262K,
                    gc_warn_threshold=int(USABLE_262K * 0.75))
        out = eng.evaluate(st)
        warns = [r for r in out if r.trigger == "gc-warn"]
        assert len(warns) == 1
        assert "durable-set" in warns[0].text
        assert "continuity" in warns[0].text.lower()

    def test_gc_warn_silent_below_threshold(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(context_tokens=10000, gc_warn_threshold=int(USABLE_262K * 0.75))
        assert [r for r in eng.evaluate(st) if r.trigger == "gc-warn"] == []

    def test_gc_warn_silent_when_unknown(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(context_tokens=999999, gc_warn_threshold=0)
        assert [r for r in eng.evaluate(st) if r.trigger == "gc-warn"] == []

    def test_gc_warn_once_per_session(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(context_tokens=USABLE_262K, gc_warn_threshold=1000)
        assert eng.evaluate(st)
        assert [r for r in eng.evaluate(st) if r.trigger == "gc-warn"] == []

    def test_gc_warn_tool_mention_gated(self, tmp_path):
        eng = self._eng(tmp_path)
        st_with = _state(context_tokens=USABLE_262K, gc_warn_threshold=1000,
                         enabled_tools={"context_gc"})
        assert "context_gc" in eng.evaluate(st_with)[0].text
        eng2 = self._eng(tmp_path)
        st_without = _state(context_tokens=USABLE_262K, gc_warn_threshold=1000)
        assert "context_gc" not in eng2.evaluate(st_without)[0].text

    def test_gc_warn_turn_start_only(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(context_tokens=USABLE_262K, gc_warn_threshold=1000,
                    evaluation_point="tool_loop_boundary")
        assert [r for r in eng.evaluate(st) if r.trigger == "gc-warn"] == []

    def test_gc_budget_at_50pct(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(gc_budget_fraction=0.5)
        out = [r for r in eng.evaluate(st) if r.trigger == "gc-budget"]
        assert len(out) == 1 and "prune" in out[0].text.lower()

    def test_gc_budget_below_half_silent(self, tmp_path):
        eng = self._eng(tmp_path)
        st = _state(gc_budget_fraction=0.49)
        assert [r for r in eng.evaluate(st) if r.trigger == "gc-budget"] == []

    def test_triggers_rehydrate(self, tmp_path):
        eng = self._eng(tmp_path)
        eng.rehydrate([
            {"role": "user", "source": "reminder", "trigger": "gc-warn",
             "content": f"{TAG_OPEN}\nfinal durable-set\n{TAG_OPEN.replace('<', '</')}"},
        ])
        st = _state(context_tokens=USABLE_262K, gc_warn_threshold=1000)
        assert [r for r in eng.evaluate(st) if r.trigger == "gc-warn"] == []

    def test_reset_rearms_both(self, tmp_path):
        eng = self._eng(tmp_path)
        eng.evaluate(_state(context_tokens=USABLE_262K, gc_warn_threshold=1000,
                            gc_budget_fraction=0.9))
        eng.reset()
        assert eng.evaluate(_state(context_tokens=USABLE_262K, gc_warn_threshold=1000))
        assert eng.evaluate(_state(gc_budget_fraction=0.9))


# ============================================================================
# context_gc + set_active_project tools
# ============================================================================

class TestGCTool:
    def test_registered(self):
        assert "context_gc" in BUILTIN_TOOLS
        schema = BUILTIN_TOOLS["context_gc"]["parameters"]
        assert schema["type"] == "object"
        assert not schema.get("required")

    def test_missing_callback_steers_to_slash(self):
        res = asyncio.new_event_loop().run_until_complete(
            execute_tool("context_gc", {}, None, {"room_id": ROOM}))
        # headless/no-wiring: clean is_error, names the operator command
        assert res.is_error and "/cache gc" in res.content

    def test_sub_sentinel_refused(self):
        res = asyncio.new_event_loop().run_until_complete(
            execute_tool("context_gc", {}, None,
                         {"room_id": "__sub__",
                          "apply_gc_boundary": AsyncMock(return_value={})}))
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
            execute_tool("context_gc", {}, None,
                         {"room_id": ROOM, "apply_gc_boundary": cb}))
        assert not res.is_error
        assert "12" in res.content and "40000" in res.content
        cb.assert_awaited_once()
        assert cb.call_args.kwargs.get("trigger", cb.call_args.args[-1] if cb.call_args.args else None) in (None, "tool")

    def test_noop_is_error_with_reason(self):
        cb = AsyncMock(return_value={"applied": False,
                                     "noop_reason": "cooldown: 1 of 3 turns since last boundary",
                                     "manifest": None, "over_budget": False})
        res = asyncio.new_event_loop().run_until_complete(
            execute_tool("context_gc", {}, None,
                         {"room_id": ROOM, "apply_gc_boundary": cb}))
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
# Matrix commands: /cache gc, /cache status, /project
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

    def test_cache_gc_applies_boundary_and_rebuilds_in_place(self, tmp_path):
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
            bot._handle_room_message(room, _event("/cache gc")))

        entries = log.read(ROOM)
        assert any(e.get("event") == "gc_boundary" for e in entries)
        assert any(e.get("source") == "gc_snapshot" for e in entries)
        assert id(agent.history(ROOM)) == ident_before, \
            "in-memory history must be rebuilt IN PLACE (same object identity)"
        sent = "\n".join(self._send_texts(bot))
        assert "GC" in sent or "boundary" in sent

    def test_cache_status_shows_boundary(self, tmp_path):
        bot, agent = self._bot(tmp_path)
        log = bot.session_log
        log.append(room=ROOM, sender=AGENT_ID, role="user", content="q")
        log.append(room=ROOM, sender=AGENT_ID, role="system",
                   event="gc_boundary", entry_index=1, detail=json.dumps({
                       "ts": "T", "boundary_index": 1, "trigger": "manual",
                       "classes": {}, "tokens_before": 10, "tokens_after_est": 5,
                       "durable": {}, "errors": []}))
        room = MagicMock()
        room.room_id = ROOM
        asyncio.new_event_loop().run_until_complete(
            bot._handle_room_message(room, _event("/cache")))
        sent = "\n".join(self._send_texts(bot))
        assert "GC boundary" in sent and "1" in sent

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
# Real-path agent-loop pinning (workspace-kdsn.305.2, authored post-305 build).
# Canonical pattern per tests/test_guidance_integration.py: REAL Agent + REAL
# SessionLog + REAL callbacks (build_callbacks seam); ONLY the provider stream
# and the nio client are mocked. These tests exist because a full green suite
# missed an unawaited coroutine at the in-loop hard tier — the wiring bug was
# invisible to every seam-mocked test (tool-management, "the one lesson").
# ============================================================================

import asyncio
import json as _json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.agent import Agent, ContextOverflowError
from openalph.config import AgentConfig, ContextGCConfig, MatrixConfig, ProviderConfig
from openalph.matrix import MatrixBot
from openalph.session import SessionLog
from openalph.provider import Response, StreamEvent, ToolCall, Usage

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
    return asyncio.run(
        bot._run_heartbeat_turn(GC_ROOM, "[Automated heartbeat]",
                                turn_source="heartbeat"))


async def _huge_tool(*args, name=None, input=None, **kw):
    from openalph.tools import ToolResult
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
                 if e.get("role") == "system" and e.get("event") == "gc_boundary"]
        assert len(marks) == 1
        assert _json.loads(marks[0]["detail"])["trigger"] == "auto"
        assert id(agent.history(GC_ROOM)) == ident_before, \
            "history must be rebuilt IN PLACE (same object identity)"
        h_after = agent.history(GC_ROOM)
        assert _big_outputs(h_after, 60000) == 0
        assert any("expunged at GC boundary" in str(m.get("content", ""))
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
        """gc_enabled=False: same crossing seed, NO boundary, outputs intact."""
        bot, agent = _gc_real_bot(tmp_path, model_max_tokens=150000)
        agent.config.context = ContextGCConfig(gc_enabled=False)
        _seed_room(bot)
        h_before = agent.history(GC_ROOM)
        ident_before = id(h_before)
        state = {}
        with patch("openalph.agent.stream", _gc_stream(state)):
            _run_turn(bot)
        assert not any(e.get("event") == "gc_boundary"
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
                 if e.get("role") == "system" and e.get("event") == "gc_boundary"]
        assert marks, "in-loop guard trip must apply a hard boundary"
        assert _json.loads(marks[-1]["detail"])["trigger"] == "hard"
        assert state.get("stream_calls", 0) >= 2, "turn must continue to completion"
        payloads = state.get("payloads", [])
        assert len(payloads) >= 2
        second_texts = [str(m.get("content", "")) for m in payloads[1]
                        if isinstance(m.get("content"), str)]
        assert any("expunged at GC boundary" in t for t in second_texts), \
            "the continued turn must see the pointer-ized history"

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
                 if e.get("role") == "system" and e.get("event") == "gc_boundary"]
        assert _json.loads(marks[-1]["detail"])["trigger"] in ("auto", "hard")
