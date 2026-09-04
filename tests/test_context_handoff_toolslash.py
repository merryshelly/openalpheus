"""Context handoff — T4 tool/slash rename (kdsn.322, spec §3.4 + .305.16).

RED SUITE — orchestrator-authored (tdd-orchestration route). The tests ARE
the specification for the operator-facing renames: the ``context_handoff``
tool (replaces ``context_gc``), the ``/cache handoff`` slash command
(replaces ``/cache gc``, which becomes a deprecation error), the §3.4
operator-visibility notice on EVERY applied trigger, the /cache + /status
boundary line renames, and the strippable-stats removal (workspace-kdsn.305.16).

Interface contract:
  - ``BUILTIN_TOOLS["context_handoff"]`` (no ``context_gc`` key): the
    description describes FULL-STRIP semantics — no pointer language — and
    keeps the cooldown guidance + no-parameters + sub-agent refusal contract.
  - The handler (renamed ``_execute_context_handoff``): same guard ladder
    (room_id from callbacks, ``__sub__`` refusal BEFORE the transport
    guard, missing-callback steering — now steering to ``/cache handoff``),
    cooldown for tool-triggered boundaries (config ``turn_cooldown``,
    RETAINED per spec §3.1), success summary with boundary index + tokens
    before->after.
  - §3.4 visibility: the shared ``apply_handoff_boundary`` closure emits a
    collapsed room notice on EVERY APPLIED boundary (auto / tool / slash) —
    trigger, tokens before->after, durable summary, checkpoint status.
    Display-only (the JSONL marker is the audit record). Headless sinks
    render it to stderr (never stdout — the exec contract from T3).
  - ``/cache handoff`` applies the boundary (manual trigger, cooldown
    bypassed, the _note_handoff_boundary_applied seam driven); ``/cache gc``
    returns the deprecation error steering to ``/cache handoff``.
  - ``/cache`` status + ``/status``: "Handoff boundary: <idx> (<trigger>)";
    the strippable stats row and its session.py plumbing are GONE.
  - Legacy marker NAMES (gc_boundary/gc_snapshot) legitimately remain in
    session.py render + spotter (hard-epoch legacy sessions) — the scan
    pins the OPERATOR surfaces only: the registry/handler/dispatch names,
    strippable_stats, and the /cache gc command string.
"""

import asyncio
from pathlib import Path

from openalph.config import AgentConfig, ContextHandoffConfig, ProviderConfig
from openalph.tools import BUILTIN_TOOLS, execute_tool


# ---------------------------------------------------------------------------
# Harness — real agent + real SessionLog + build_callbacks, faked boundary?
# NO: the real closure + real apply_boundary_and_rebuild. Only the pieces
# the tool handler needs are real; there is no provider in this slice.
# ---------------------------------------------------------------------------

ROOM_ID = "!toolslash:test"


def _config(tmp_path, **ctx_kw):
    provider = ProviderConfig(key="p", type="openai", api_key="sk-test",
                              base_url="http://127.0.0.1:18081")
    return AgentConfig(
        name="tsl",
        default_model="p/model",
        max_tokens=100,
        model_max_tokens=100000,  # no ladder interference in this slice
        providers={"p": provider},
        workspace=Path(tmp_path),
        context=ContextHandoffConfig(**ctx_kw),
    )


class _RecordingSinks:
    """Records send_notice calls (the §3.4 visibility contract)."""

    def __init__(self):
        self.notices = []

    async def send_notice(self, room_id, body, **kw):
        self.notices.append((room_id, body))

    async def log_reminder(self, room_id, reminder):
        pass

    async def send_media(self, *a, **k):
        raise RuntimeError("no media in tests")

    async def on_redaction(self, *a, **k):
        pass

    async def on_keepalive_miss(self, *a, **k):
        pass

    async def on_degenerate(self, *a, **k):
        pass

    async def log_vision_injection(self, *a, **k):
        pass

    async def log_spotter_flag(self, *a, **k):
        pass

    async def log_turn(self, *a, **k):
        pass


def _scene(tmp_path, *, entries=6):
    """Real Agent + SessionLog seeded with `entries` assistant turns."""
    from openalph.agent import Agent
    from openalph.session import SessionLog

    config = _config(tmp_path)
    agent = Agent(config)
    log = SessionLog(config.workspace, "@tsl:test", handoff_default=True)
    for i in range(entries):
        log.append(role="user", sender="operator", room=ROOM_ID,
                   content=f"task {i}")
        log.append(role="assistant", sender="@tsl:test", room=ROOM_ID,
                   content=f"reply {i} " + "z" * 400)
    agent.history(ROOM_ID).extend([
        {"role": "user", "content": f"task {i}"} for i in range(entries)
    ] + [
        {"role": "assistant", "content": f"reply {i} " + "z" * 400}
        for i in range(entries)
    ])
    return agent, log


def _callbacks(agent, log, sinks):
    from openalph.callbacks import build_callbacks
    return build_callbacks(
        agent, ROOM_ID, sinks, turn_source=None, session_log=log,
        room_name="tsl")


# ===========================================================================
# Tool registry
# ===========================================================================

class TestToolRegistry:
    def test_context_handoff_registered(self):
        assert "context_handoff" in BUILTIN_TOOLS
        assert "context_gc" not in BUILTIN_TOOLS

    def test_description_full_strip_semantics(self):
        desc = BUILTIN_TOOLS["context_handoff"]["description"]
        assert "strip" in desc.lower()
        assert "handoff" in desc.lower()
        assert "checkpoint" in desc.lower(), (
            "the description steers to the checkpoint reminder contract")
        # full-strip semantics: no pointer/carving language survives
        assert "pointer" not in desc.lower()
        assert "reduced" not in desc.lower()
        # the load-bearing constraints stay
        assert "cooldown" in desc.lower()
        assert "sub-agent" in desc.lower() or "subagents" in desc.lower()

    def test_no_parameters(self):
        schema = BUILTIN_TOOLS["context_handoff"]["parameters"]
        assert schema.get("properties") == {}
        assert schema.get("type") == "object"


# ===========================================================================
# Tool handler (real-path through execute_tool)
# ===========================================================================

class TestToolHandler:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_happy_path_applies_boundary(self, tmp_path):
        agent, log = _scene(tmp_path)
        sinks = _RecordingSinks()
        cb = _callbacks(agent, log, sinks)
        res = self._run(execute_tool(
            "context_handoff", {}, agent.config, cb))
        assert not res.is_error
        assert "boundary" in res.content.lower()
        entries = log.read(ROOM_ID)
        assert any(e.get("event") == "handoff_boundary" for e in entries)
        assert any(e.get("source") == "handoff_snapshot" for e in entries)

    def test_cooldown_blocks_second_immediate_call(self, tmp_path):
        agent, log = _scene(tmp_path)
        sinks = _RecordingSinks()
        cb = _callbacks(agent, log, sinks)
        first = self._run(execute_tool(
            "context_handoff", {}, agent.config, cb))
        assert not first.is_error
        # zero assistant turns since the boundary -> cooldown refusal
        second = self._run(execute_tool(
            "context_handoff", {}, agent.config, cb))
        assert second.is_error
        assert "cooldown" in second.content.lower()

    def test_sub_agent_refused(self, tmp_path):
        agent, log = _scene(tmp_path)
        sinks = _RecordingSinks()
        cb = _callbacks(agent, log, sinks)
        cb = dict(cb)
        cb["room_id"] = "__sub__"
        res = self._run(execute_tool(
            "context_handoff", {}, agent.config, cb))
        assert res.is_error
        assert "sub" in res.content.lower()

    def test_missing_callback_steers_to_slash(self, tmp_path):
        agent, log = _scene(tmp_path)
        res = self._run(execute_tool(
            "context_handoff", {}, agent.config, {"room_id": ROOM_ID}))
        assert res.is_error
        assert "/cache handoff" in res.content, (
            "the steering names the NEW slash command")

    def test_missing_room_id_steers(self, tmp_path):
        res = self._run(execute_tool(
            "context_handoff", {}, _config(tmp_path), {}))
        assert res.is_error

    def test_kill_switch_noops(self, tmp_path):
        agent, log = _scene(tmp_path)
        agent.config.context = ContextHandoffConfig(handoff_enabled=False)
        sinks = _RecordingSinks()
        cb = _callbacks(agent, log, sinks)
        res = self._run(execute_tool(
            "context_handoff", {}, agent.config, cb))
        assert res.is_error or "not applied" in res.content.lower()
        entries = log.read(ROOM_ID)
        assert not any(e.get("event") == "handoff_boundary" for e in entries)


# ===========================================================================
# §3.4 operator visibility: a notice on EVERY applied trigger
# ===========================================================================

class TestVisibilityNotice:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_tool_trigger_emits_notice(self, tmp_path):
        agent, log = _scene(tmp_path)
        sinks = _RecordingSinks()
        cb = _callbacks(agent, log, sinks)
        self._run(execute_tool("context_handoff", {}, agent.config, cb))
        assert sinks.notices, (
            "spec §3.4: boundary application emits a collapsed room notice "
            "on EVERY trigger — the tool path included")
        room, body = sinks.notices[-1]
        assert room == ROOM_ID
        assert "boundary" in body.lower()

    def test_closure_trigger_emits_notice(self, tmp_path):
        # the auto/hard tiers apply through the same closure — the notice
        # contract covers them (the T3 exec path renders it via stderr).
        agent, log = _scene(tmp_path)
        sinks = _RecordingSinks()
        cb = _callbacks(agent, log, sinks)
        apply_cb = cb.get("apply_handoff_boundary")
        assert callable(apply_cb)
        res = asyncio.run(apply_cb(ROOM_ID, trigger="auto",
                                   exclude_inflight=False))
        assert res.get("applied") is True
        assert sinks.notices, (
            "the shared closure emits the visibility notice (auto tier)")

    def test_noop_emits_no_success_notice(self, tmp_path):
        agent, log = _scene(tmp_path)
        sinks = _RecordingSinks()
        cb = _callbacks(agent, log, sinks)
        asyncio.run(execute_tool("context_handoff", {}, agent.config, cb))
        n_after_first = len(sinks.notices)
        # cooldown noop -> no NEW success notice (the tool result carries it)
        asyncio.run(execute_tool("context_handoff", {}, agent.config, cb))
        assert len(sinks.notices) == n_after_first


# ===========================================================================
# Slash commands (matrix.py surface pins)
# ===========================================================================

class TestSlashSurface:
    def test_handoff_command_wired(self):
        import inspect

        import openalph.matrix as m
        src = inspect.getsource(m)
        assert 'value == "handoff"' in src, (
            "/cache handoff is the manual boundary command")
        assert '"/cache handoff"' in src or "/cache handoff" in src

    def test_gc_command_deprecated(self):
        import inspect

        import openalph.matrix as m
        src = inspect.getsource(m)
        assert "deprecated" in src.lower()
        assert "/cache handoff" in src, (
            "the /cache gc deprecation error steers to /cache handoff")

    def test_status_line_renamed(self):
        import inspect

        import openalph.matrix as m
        src = inspect.getsource(m)
        assert "Handoff boundary:" in src
        assert "GC boundary:" not in src

    def test_strippable_gone_from_status(self):
        import inspect

        import openalph.matrix as m
        src = inspect.getsource(m)
        assert "strippable" not in src.lower()
        from openalph.session import SessionLog
        assert not hasattr(SessionLog, "strippable_stats"), (
            "workspace-kdsn.305.16: the strippable plumbing is removed, "
            "not just unrendered")


# ===========================================================================
# Hard epoch: operator-surface spellings absent from src
# ===========================================================================

class TestOldSpellingsAbsentFromSrc:
    def test_no_operator_surface_tokens(self):
        from _handoff_helpers import scan_src_for_tokens
        hits = scan_src_for_tokens([
            "context_gc",
            "strippable_stats",
        ])
        # The legacy marker EVENT names (gc_boundary/gc_snapshot) stay —
        # hard-epoch legacy sessions render them; spotter watches both.
        real = [(f, i, ln) for f, i, ln in hits
                if "gc_boundary" not in ln and "gc_snapshot" not in ln]
        assert not real, (
            "hard epoch: operator-surface spellings remain in src:\n"
            + "\n".join(f"  {f}:{i}: {ln}" for f, i, ln in real[:20]))
