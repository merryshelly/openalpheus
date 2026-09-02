"""Spotter v2 contract tests (bead workspace-im7t.40.7).

THE SPECIFICATION for Spotter v2, per the SB-signed v2-spec.md (grill 2026-09-01,
G1-G16 + G10 gate amendment). Written RED-FIRST per the standing SB directive.

v2 contract headlines (each maps to spec sections):
  - Verdicts are `report_verdict(status, claim, class, severity, evidence)` TOOL
    CALLS — flat schema (G10 gate: null-unions are not grammar-safe on sglang);
    one tool (G1); silence is an explicit status value.
  - Termination = the verdict call ONLY (G2: mixed calls terminate-and-ignore);
    narration is legal mid-pass prose and PERSISTS (V1-B, G4).
  - Boundary firing: maybe_fire at the main loop's tool-call iteration top
    (V1-C); coalescing is the only throttle (G5).
  - Session shape: variable-length passes tracked by state.pass_start — the v1
    2*passes arithmetic DIES; investigation traffic persists (G4).
  - G10: code default OFF, explicit arming only (config yes or in-room start);
    model default qwen38blackwell (gate passed 2026-09-01).
  - G13: error streak unit = failed watch LOOPS + failure-path cooldown; no
    in-loop retries.
  - G15: spotter sessions inherit platform Context GC (uniform rule; deltas are
    user-class and not reduced; estimate-gate stays as backstop).
  - G16: report_verdict extends the read-only allowlist (termination-only).
  - G9: the canonical fallback toolset includes report_verdict.

Run: cd /opt/openalph && python3 -m pytest tests/test_spotter_v2.py -q
"""

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from openalph import spotter
from openalph.provider import Response, ToolCall, Usage
from openalph.spotter import (
    REPORT_VERDICT_STATUS_VALUES,
    REPORT_VERDICT_TOOL_NAME,
    SpotterManager,
    SpotterRoomState,
    validate_verdict_args,
)
from openalph.tools import ToolDef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path, *, spotter_enabled=False, spotter_model="qwen38blackwell",
                max_tokens=4096, model_max_tokens=200000, **spotter_kwargs):
    """Minimal AgentConfig-like stub — no TOML parsing needed for unit tests."""
    class Cfg:
        pass
    c = Cfg()
    c.workspace = str(tmp_path)
    c.max_tokens = max_tokens
    c.model_max_tokens = model_max_tokens
    c.spotter_enabled = spotter_enabled
    c.spotter_model = spotter_model
    c.spotter_thinking = "off"
    c.spotter_max_iterations = 8
    c.spotter_disabled_rooms = []
    for k, v in spotter_kwargs.items():
        setattr(c, k, v)
    c.providers = {}
    c.model_aliases = {}
    c.skipped_providers = {}
    return c


class FakeAgent:
    """Owning-agent stub: only the attributes SpotterManager consumes."""

    def __init__(self, config, window=100000):
        self.config = config
        self.system_prompt = "EXECUTOR SYSTEM PROMPT"
        self.tools = []
        self.history = {}
        self._spotter_inbox = {}
        self._window = window
        self._window_calls = []

    def _resolve_model_limit_for(self, model_str: str) -> int:
        self._window_calls.append(model_str)
        return self._window


def make_manager(tmp_path, *, config=None, agent=None, window=100000):
    cfg = config or make_config(tmp_path)
    ag = agent or FakeAgent(cfg, window=window)
    return SpotterManager(cfg, ag), cfg, ag


def make_tools(names):
    return [ToolDef(name=n, description=f"{n} desc", parameters={"type": "object"},
                    config={}) for n in names]


@dataclass
class FakeUsage:
    input_tokens: int = 10
    output_tokens: int = 5


def resp(content="", tool_calls=None, stop_reason="stop"):
    return Response(content=content, usage=Usage(input_tokens=10, output_tokens=5),
                    stop_reason=stop_reason, tool_calls=tool_calls or [])


def tc(id_, name, input_):
    return ToolCall(id=id_, name=name, input=input_)


SILENT_ARGS = {"status": "silent", "claim": "", "class": "", "severity": "", "evidence": ""}
FLAG_ARGS = {"status": "flag", "claim": "the deploy failed while the agent said it succeeded",
             "class": "contradiction", "severity": "high",
             "evidence": "tool output: exit code 1"}


class CompleteScript:
    """Queue of responses (or exceptions) for the mocked spotter.complete."""

    def __init__(self, *items):
        self.items = list(items)
        self.calls = []  # kwargs of each complete() call

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def run_loop():
    """Drain pending asyncio tasks (the watch task) to completion."""
    async def _run():
        for _ in range(200):
            pending = [t for t in asyncio.all_tasks()
                       if not t.done() and t is not asyncio.current_task()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)
    return _run


# ---------------------------------------------------------------------------
# G10 — config defaults & explicit arming
# ---------------------------------------------------------------------------

class TestG10ConfigDefaults:
    def test_code_default_enabled_false(self, tmp_path):
        cfg = make_config(tmp_path)
        assert cfg.spotter_enabled is False

    def test_code_default_model_qwen38blackwell(self, tmp_path):
        cfg = make_config(tmp_path)
        assert cfg.spotter_model == "qwen38blackwell"

    def test_config_module_defaults_flip(self):
        import inspect
        import openalph.config as config_mod
        src = inspect.getsource(config_mod)
        assert "spotter_enabled: bool = False" in src, (
            "config.py default must flip to False (G10: explicit arming only)")
        assert 'spotter_model: str = "qwen38blackwell"' in src

    def test_fresh_state_armed_seeded_from_config_disabled(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path, config=make_config(tmp_path, spotter_enabled=False))
        st = mgr.ensure_state("!r:1")
        assert st.armed is False

    def test_fresh_state_armed_seeded_from_config_enabled(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path, config=make_config(tmp_path, spotter_enabled=True))
        st = mgr.ensure_state("!r:1")
        assert st.armed is True

    def test_maybe_fire_config_disabled_no_state_no_fire(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path, config=make_config(tmp_path, spotter_enabled=False))
        ag.history["!r:1"] = [{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "hello"}]
        mgr.maybe_fire("!r:1", ag.history["!r:1"], None, {})
        assert "!r:1" not in mgr._states  # cheap path: no state created

    def test_op_start_arms_room_despite_config_disabled(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path, config=make_config(tmp_path, spotter_enabled=False))
        ag.history["!r:1"] = [{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "hello"}]
        out = mgr.op_start("!r:1")
        assert mgr.ensure_state("!r:1").armed is True
        assert "🟢" in out

    def test_op_stop_disarms_room(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path, config=make_config(tmp_path, spotter_enabled=True))
        mgr.op_stop("!r:1")
        assert mgr.ensure_state("!r:1").armed is False

    def test_maybe_fire_ignores_disarmed_room(self, tmp_path, run_loop):
        mgr, cfg, ag = make_manager(tmp_path, config=make_config(tmp_path, spotter_enabled=True))
        ag.history["!r:1"] = [{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "hello"}]
        mgr.ensure_state("!r:1").armed = False
        mgr.maybe_fire("!r:1", ag.history["!r:1"], None, {})
        assert mgr.ensure_state("!r:1").dirty is False


# ---------------------------------------------------------------------------
# Verdict tool: schema, validation, toolset membership (G1/G16/G9/G10-gate)
# ---------------------------------------------------------------------------

class TestVerdictTool:
    def test_tool_name_constant(self):
        assert REPORT_VERDICT_TOOL_NAME == "report_verdict"

    def test_tool_schema_is_flat_status_enum(self):
        from openalph.spotter import report_verdict_tooldef
        td = report_verdict_tooldef()
        assert td.name == "report_verdict"
        props = td.parameters["properties"]
        assert set(props) == {"status", "claim", "class", "severity", "evidence"}
        assert props["status"]["enum"] == ["silent", "flag"]
        assert props["class"]["enum"] == list(spotter._VALID_FLAG_CLASSES)
        assert props["severity"]["enum"] == ["low", "med", "high"]
        assert set(td.parameters["required"]) == {"status", "claim", "class", "severity", "evidence"}
        assert td.parameters.get("additionalProperties") is False
        # NO union/null anywhere — grammar compilers cannot enforce them (G10 gate).
        assert "oneOf" not in json.dumps(td.parameters)
        assert "anyOf" not in json.dumps(td.parameters)

    def test_status_values_constant(self):
        assert REPORT_VERDICT_STATUS_VALUES == ("silent", "flag")

    def test_validate_silent_ignores_padded_siblings(self):
        padded = {"status": "silent", "claim": "no issue at all, everything fine",
                  "class": "guidance", "severity": "low", "evidence": "nothing happened"}
        status, flag = validate_verdict_args(padded)
        assert status == "silent"
        assert flag is None

    def test_validate_flag_valid(self):
        status, flag = validate_verdict_args(dict(FLAG_ARGS))
        assert status == "flag"
        assert flag is not None
        assert flag.claim == FLAG_ARGS["claim"]
        assert flag.klass == "contradiction"
        assert flag.severity == "high"
        assert flag.evidence == FLAG_ARGS["evidence"]

    def test_validate_accepts_json_string(self):
        status, flag = validate_verdict_args(json.dumps(FLAG_ARGS))
        assert status == "flag" and flag is not None

    @pytest.mark.parametrize("bad", [
        {"status": "flag", "claim": "", "class": "contradiction", "severity": "high", "evidence": "e"},
        {"status": "flag", "claim": "c", "class": "bogus", "severity": "high", "evidence": "e"},
        {"status": "flag", "claim": "c", "class": "contradiction", "severity": "extreme", "evidence": "e"},
        {"status": "flag", "claim": "c", "class": "contradiction", "severity": "high", "evidence": ""},
        {"status": "bogus", "claim": "", "class": "", "severity": "", "evidence": ""},
        {"claim": "c"},                       # missing status
        {"status": "flag", "extra": 1, "claim": "c", "class": "contradiction", "severity": "high", "evidence": "e"},
        "not a dict",
    ])
    def test_validate_invalid_args_parse_error(self, bad):
        status, flag = validate_verdict_args(bad)
        assert status == "parse_error"
        assert flag is None

    def test_validate_malformed_json_string(self):
        status, flag = validate_verdict_args("{not json")
        assert status == "parse_error"

    def test_pass_tools_always_include_report_verdict(self, tmp_path):
        # G9: even a minimally provisioned agent (no allowlisted tools) gets a
        # toolset that can terminate — report_verdict MUST be present.
        mgr, cfg, ag = make_manager(tmp_path)
        ag.tools = []
        tools = spotter.spotter_pass_tools(ag.tools)
        names = [t.name for t in tools]
        assert "report_verdict" in names

    def test_pass_tools_allowlist_intersection_plus_verdict(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path)
        ag.tools = make_tools(["file_read", "shell", "subagent", "grep"])
        tools = spotter.spotter_pass_tools(ag.tools)
        names = [t.name for t in tools]
        assert "file_read" in names and "grep" in names
        assert "shell" not in names and "subagent" not in names
        assert "report_verdict" in names


# ---------------------------------------------------------------------------
# Termination semantics (V1-A/V1-B/G2) — the pass loop
# ---------------------------------------------------------------------------

class TestTermination:
    async def _run_pass(self, tmp_path, monkeypatch, script, history=None, config=None):
        mgr, cfg, ag = make_manager(tmp_path, config=config)
        ag.tools = make_tools(["file_read"])
        monkeypatch.setattr(spotter, "complete", script)
        hist = history or [{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "hello"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        return mgr, cfg, ag

    async def test_narration_continues_loop_and_persists(self, tmp_path, monkeypatch):
        script = CompleteScript(resp(content="Let me think about this delta."),
                                resp(content="Still weighing the evidence."),
                                resp(tool_calls=[tc("v1", "report_verdict", dict(SILENT_ARGS))]))
        mgr, cfg, ag = await self._run_pass(tmp_path, monkeypatch, script)
        await _drain()
        st = mgr.ensure_state("!r:1")
        # narration turns persisted as assistant messages (V1-B), then verdict pair
        narrations = [m for m in st.messages if m.get("role") == "assistant"
                      and "tool_calls" not in m]
        assert len(narrations) == 2
        assert st.passes == 1
        assert len(script.calls) == 3  # narration did NOT terminate the loop

    async def test_verdict_call_terminates_and_stores_tool_pair(self, tmp_path, monkeypatch):
        script = CompleteScript(resp(tool_calls=[tc("v1", "report_verdict", dict(SILENT_ARGS))]))
        mgr, cfg, ag = make_manager(tmp_path)
        monkeypatch.setattr(spotter, "complete", script)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        st = mgr.ensure_state("!r:1")
        # session = [delta, assistant-verdict-call, tool-result] — the REAL pair
        assert len(st.messages) == 3
        assert st.messages[1]["role"] == "assistant"
        assert st.messages[1]["tool_calls"][0]["name"] == "report_verdict"
        assert st.messages[2]["role"] == "tool"
        assert st.messages[2]["tool_call_id"] == "v1"
        assert st.passes == 1
        assert st.last_status == "silent"
        assert len(script.calls) == 1  # terminated immediately

    async def test_mixed_calls_terminate_and_ignore_siblings(self, tmp_path, monkeypatch):
        script = CompleteScript(resp(tool_calls=[
            tc("g1", "file_read", {"path": "/etc/passwd"}),
            tc("v1", "report_verdict", dict(FLAG_ARGS)),
        ]))
        mgr, cfg, ag = make_manager(tmp_path)
        ag.tools = make_tools(["file_read"])
        executed = []

        async def fake_execute(**kwargs):
            executed.append(kwargs["name"])
            class R:
                content = "root:x:0:0"
            return R()

        monkeypatch.setattr(spotter, "complete", script)
        monkeypatch.setattr(spotter, "execute_tool", fake_execute)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        st = mgr.ensure_state("!r:1")
        assert executed == []  # sibling NOT executed (G2 terminate-and-ignore)
        assert st.passes == 1
        assert st.last_status == "flag"
        led = [json.loads(l) for l in
               (tmp_path / "sessions/spotters/r_1.ledger.jsonl").read_text().splitlines()]
        flag_events = [e for e in led if e["event"] == "flag"]
        assert flag_events and flag_events[0].get("siblings_ignored") == 1

    async def test_investigation_traffic_persists_in_session(self, tmp_path, monkeypatch):
        script = CompleteScript(
            resp(tool_calls=[tc("g1", "file_read", {"path": "/tmp/x"})]),
            resp(tool_calls=[tc("v1", "report_verdict", dict(SILENT_ARGS))]))
        mgr, cfg, ag = make_manager(tmp_path)

        class R:
            content = "file contents here"

        monkeypatch.setattr(spotter, "complete", script)
        monkeypatch.setattr(spotter, "execute_tool", lambda **kw: _async_return(R()))
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        st = mgr.ensure_state("!r:1")
        # G4: the investigation pair PERSISTS (v1 compacted it away)
        roles = [m.get("role") for m in st.messages]
        assert "tool" in roles
        tool_msgs = [m for m in st.messages if m.get("role") == "tool" and m.get("tool_call_id") == "g1"]
        assert tool_msgs, "investigation result must persist in state.messages"

    async def test_iteration_cap_forces_report_verdict_with_tool_choice(self, tmp_path, monkeypatch):
        narr = resp(content="thinking out loud")
        verdict = resp(tool_calls=[tc("v9", "report_verdict", dict(FLAG_ARGS))])
        script = CompleteScript(*([narr] * 8), verdict)
        mgr, cfg, ag = make_manager(tmp_path)
        monkeypatch.setattr(spotter, "complete", script)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        forced = script.calls[-1]
        assert forced.get("tool_choice") == "report_verdict"
        assert forced.get("strict") is True
        assert [t.name for t in forced["tools"]] == ["report_verdict"]
        st = mgr.ensure_state("!r:1")
        assert st.last_status == "flag"

    async def test_forced_call_failure_falls_back_to_text_parse(self, tmp_path, monkeypatch):
        # cap → forced call RAISES → degraded path: no-tools text call → v1.1b parser
        narr = resp(content="thinking out loud")
        script = CompleteScript(*([narr] * 8), RuntimeError("forced call exploded"),
                                resp(content="FLAG\nclaim: c\nclass: contradiction\n"
                                             "severity: high\nevidence: e"))
        mgr, cfg, ag = make_manager(tmp_path)
        monkeypatch.setattr(spotter, "complete", script)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        st = mgr.ensure_state("!r:1")
        assert st.last_status == "flag"
        fallback_call = script.calls[-1]
        assert fallback_call["tools"] in (None, [])  # no-tools text call

    async def test_total_failure_stores_placeholder(self, tmp_path, monkeypatch):
        narr = resp(content="thinking")
        script = CompleteScript(*([narr] * 8), RuntimeError("forced failed"),
                                resp(content="complete garbage, no verdict here"))
        mgr, cfg, ag = make_manager(tmp_path)
        monkeypatch.setattr(spotter, "complete", script)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        st = mgr.ensure_state("!r:1")
        assert st.last_status == "parse_error"
        # placeholder (tool-referencing + SILENT tail, G3) stored as assistant text
        last = st.messages[-1]
        assert last["role"] == "assistant"
        assert "report_verdict" in last["content"]
        assert last["content"].rstrip().endswith("SILENT")

    async def test_invalid_verdict_args_store_placeholder_and_ledger(self, tmp_path, monkeypatch):
        script = CompleteScript(resp(tool_calls=[tc("v1", "report_verdict",
                                                    {"status": "flag", "claim": "", "class": "x",
                                                     "severity": "y", "evidence": ""})]))
        mgr, cfg, ag = make_manager(tmp_path)
        monkeypatch.setattr(spotter, "complete", script)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        st = mgr.ensure_state("!r:1")
        assert st.last_status == "parse_error"
        led = [json.loads(l) for l in
               (tmp_path / "sessions/spotters/r_1.ledger.jsonl").read_text().splitlines()]
        assert led[-1]["event"] == "pass" and led[-1]["status"] == "parse_error"

    async def test_redaction_applies_to_persisted_tool_results(self, tmp_path, monkeypatch):
        # Redact-before-store invariant (v2): investigation results that quote
        # credential-shaped strings must not persist raw.
        script = CompleteScript(
            resp(tool_calls=[tc("g1", "file_read", {"path": "/tmp/creds"})]),
            resp(tool_calls=[tc("v1", "report_verdict", dict(SILENT_ARGS))]))
        mgr, cfg, ag = make_manager(tmp_path)

        class R:
            content = "the key is sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

        async def fake_execute(**kw):
            return R()

        monkeypatch.setattr(spotter, "complete", script)
        monkeypatch.setattr(spotter, "execute_tool", fake_execute)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        blob = json.dumps(mgr.ensure_state("!r:1").messages)
        assert "sk-ant-api03-AAAA" not in blob


# ---------------------------------------------------------------------------
# Session shape: pass_start, cancel/rewind, variable length (V1-D/G4)
# ---------------------------------------------------------------------------

class TestSessionShape:
    def test_state_has_pass_start(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path)
        st = mgr.ensure_state("!r:1")
        assert hasattr(st, "pass_start")
        assert st.pass_start is None

    async def test_no_2p_assumption_cancel_rewinds_to_pass_start(self, tmp_path, monkeypatch):
        # A prior pass with tool traffic makes the session longer than 2*passes;
        # cancel must truncate to THIS pass's start, not any arithmetic.
        script = CompleteScript(
            resp(tool_calls=[tc("g0", "file_read", {"path": "/x"})]),
            resp(tool_calls=[tc("v0", "report_verdict", dict(SILENT_ARGS))]),
            resp(content="narration for pass two"),
        )
        mgr, cfg, ag = make_manager(tmp_path)

        class R:
            content = "data"

        async def fake_execute(**kw):
            return R()

        monkeypatch.setattr(spotter, "complete", script)
        monkeypatch.setattr(spotter, "execute_tool", fake_execute)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"},
                {"role": "user", "content": "more"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        st = mgr.ensure_state("!r:1")
        assert st.passes == 1
        pre_second = len(st.messages)
        assert pre_second > 3  # delta + narrated traffic + verdict pair

        # Second pass: cancel mid-pass (complete hangs) → truncate to pass_start
        async def hanging(**kw):
            await asyncio.sleep(3600)
            raise asyncio.CancelledError

        monkeypatch.setattr(spotter, "complete", hanging)
        hist.append({"role": "assistant", "content": "y"})
        mgr.maybe_fire("!r:1", hist, None, {})
        task = mgr.ensure_state("!r:1").watch_task
        await asyncio.sleep(0.2)  # let the pass reach the hanging complete
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        st2 = mgr.ensure_state("!r:1")
        assert len(st2.messages) == pre_second  # truncated to pass_start
        assert st2.last_index == len(hist) - 1  # rewound: cancelled segment re-watches


async def _drain():
    """Drain pending spotter watch tasks to completion (excluding self)."""
    for _ in range(100):
        pending = [t for t in asyncio.all_tasks()
                   if not t.done() and t is not asyncio.current_task()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


def _async_return(v):
    async def _f(**kw):
        return v
    return _f()


# ---------------------------------------------------------------------------
# Boundary firing + coalescing (V1-C/G5) — agent seam
# ---------------------------------------------------------------------------

class TestBoundaryFiringAgentSeam:
    def test_agent_has_boundary_call_site_in_tool_loop(self):
        import inspect
        import openalph.agent as agent_mod
        src = inspect.getsource(agent_mod)
        # The loop-top fire must exist AFTER the spotter drain, BEFORE reminders.
        drain_idx = src.find("drain_flags(room_id)")
        fire_idx = src.find("_fire_spotter_turn_completion", drain_idx)
        rem_idx = src.find("Reminder evaluation at tool-loop boundary", drain_idx)
        assert drain_idx != -1 and fire_idx != -1 and rem_idx != -1
        assert drain_idx < fire_idx < rem_idx, (
            "V1-C: maybe_fire must be called at the loop top after the spotter "
            "drain and before reminder evaluation")

    async def test_coalescing_single_in_flight_task(self, tmp_path, monkeypatch):
        script = CompleteScript(resp(tool_calls=[tc("v1", "report_verdict", dict(SILENT_ARGS))]))
        mgr, cfg, ag = make_manager(tmp_path)
        monkeypatch.setattr(spotter, "complete", script)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        t1 = mgr.ensure_state("!r:1").watch_task
        mgr.maybe_fire("!r:1", hist, None, {})
        mgr.maybe_fire("!r:1", hist, None, {})
        t2 = mgr.ensure_state("!r:1").watch_task
        assert t1 is t2  # single in-flight pass (coalescing via dirty flag)
        assert t1 is not None  # the fire actually spawned a task (non-vacuous)
        await _drain()
        assert len(script.calls) == 1  # one pass consumed the coalesced backlog


# ---------------------------------------------------------------------------
# G13 — error streak: loop unit, cooldown, no in-loop retries
# ---------------------------------------------------------------------------

class TestErrorStreakLoopUnit:
    async def test_failed_pass_sets_cooldown_and_no_inloop_retry(self, tmp_path, monkeypatch):
        script = CompleteScript(RuntimeError("provider down"))
        mgr, cfg, ag = make_manager(tmp_path)
        monkeypatch.setattr(spotter, "complete", script)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        st = mgr.ensure_state("!r:1")
        assert st.consecutive_errors == 1
        assert st.cooldown_until is not None
        assert len(script.calls) == 1  # NO in-loop retry

    async def test_maybe_fire_skipped_during_cooldown(self, tmp_path, monkeypatch):
        script = CompleteScript(RuntimeError("provider down"))
        mgr, cfg, ag = make_manager(tmp_path)
        monkeypatch.setattr(spotter, "complete", script)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        hist.append({"role": "assistant", "content": "y"})
        mgr.maybe_fire("!r:1", hist, None, {})
        st = mgr.ensure_state("!r:1")
        assert st.dirty is False  # cooldown suppresses the fire
        assert st.last_index == len(hist) - 1  # index untouched: backlog coalesces

    async def test_success_resets_streak_and_clears_cooldown(self, tmp_path, monkeypatch):
        mgr, cfg, ag = make_manager(tmp_path)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        st = mgr.op_start("!r:1") and mgr.ensure_state("!r:1")
        st.consecutive_errors = 2
        st.cooldown_until = 1.0  # long past
        script = CompleteScript(resp(tool_calls=[tc("v1", "report_verdict", dict(SILENT_ARGS))]))
        monkeypatch.setattr(spotter, "complete", script)
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        st = mgr.ensure_state("!r:1")
        assert st.consecutive_errors == 0
        assert st.cooldown_until is None

    async def test_three_failed_loops_disarm_with_loud_notice(self, tmp_path, monkeypatch):
        notices = []
        mgr, cfg, ag = make_manager(tmp_path)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")

        def fake_send_notice(callbacks, room_id, text):
            notices.append(text)

        monkeypatch.setattr(mgr, "_send_notice", fake_send_notice)
        for i in range(3):
            script = CompleteScript(RuntimeError(f"outage {i}"))
            monkeypatch.setattr(spotter, "complete", script)
            hist.append({"role": "assistant", "content": f"turn {i}"})
            mgr.maybe_fire("!r:1", hist, None, {})
            await _drain()
            # cooldown would suppress the next fire — expire it for the test
            mgr.ensure_state("!r:1").cooldown_until = None
        st = mgr.ensure_state("!r:1")
        assert st.armed is False
        assert any("stopped watching" in n for n in notices)


# ---------------------------------------------------------------------------
# Exhaustion window fix (folded refactor): model-aware window resolution
# ---------------------------------------------------------------------------

class TestExhaustionWindowFix:
    def test_resolve_window_uses_spotter_model_string(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path, config=make_config(tmp_path, spotter_model="qwen38blackwell"))
        st = mgr.ensure_state("!r:1")
        mgr._resolve_window("!r:1")
        assert ag._window_calls == ["qwen38blackwell"]

    def test_resolve_window_prefers_room_override(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path)
        mgr.op_set_model("!r:1", "deepseek") if False else None
        st = mgr.ensure_state("!r:1")
        st.model_override = "synglm53"
        mgr._resolve_window("!r:1")
        assert ag._window_calls == ["synglm53"]

    def test_resolve_window_falls_back_on_resolver_error(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path)
        def boom(model_str):
            raise TypeError("signature drift")
        ag._resolve_model_limit_for = boom
        w = mgr._resolve_window("!r:1")
        assert w == int(cfg.model_max_tokens)

    def test_no_typeerror_from_zero_arg_call(self, tmp_path):
        # The born-broken 08-30 bug: _resolve_model_limit_for() called with no
        # args against the real agent signature → TypeError every preflight.
        mgr, cfg, ag = make_manager(tmp_path)
        try:
            mgr._resolve_window("!r:1")
        except TypeError:
            pytest.fail("_resolve_window must pass the model string")


# ---------------------------------------------------------------------------
# G15 — Context GC inheritance
# ---------------------------------------------------------------------------

class TestContextGCInheritance:
    async def test_gc_boundary_reduces_tool_results_only(self, tmp_path, monkeypatch):
        # Build a session with investigation traffic, then force a GC boundary:
        # tool RESULTS → pointer placeholders; deltas/narration/verdicts untouched.
        script = CompleteScript(
            resp(tool_calls=[tc("g1", "file_read", {"path": "/x"})]),
            resp(tool_calls=[tc("v0", "report_verdict", dict(SILENT_ARGS))]),
        )
        mgr, cfg, ag = make_manager(tmp_path)

        class R:
            content = "X" * 5000

        async def fake_execute(**kw):
            return R()

        monkeypatch.setattr(spotter, "complete", script)
        monkeypatch.setattr(spotter, "execute_tool", fake_execute)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        st = mgr.ensure_state("!r:1")
        before = json.dumps(st.messages)
        assert "XXXX" in before

        mgr._apply_gc_boundary("!r:1")
        after = json.dumps(st.messages)
        assert "XXXXX" not in after, "pre-boundary tool results must reduce to pointers"
        # deltas survive untouched (user-class evidence — G15)
        assert any(m["role"] == "user" and "TRANSCRIPT DELTA" in m["content"]
                   for m in st.messages)
        # verdict pair survives
        assert any(m.get("role") == "assistant" and m.get("tool_calls")
                   for m in st.messages)
        # transcript gained a gc_boundary manifest (JSONL append-only)
        events = [json.loads(l) for l in
                  (tmp_path / "sessions/spotters/r_1.jsonl").read_text().splitlines()]
        assert any(e.get("event") == "gc_boundary" for e in events)

    def test_rehydration_honors_gc_boundary_manifest(self, tmp_path, monkeypatch):
        mgr, cfg, ag = make_manager(tmp_path)
        t_path = tmp_path / "sessions/spotters/r_1.jsonl"
        t_path.parent.mkdir(parents=True, exist_ok=True)
        delta = "[TRANSCRIPT DELTA — 2 new entries from the watched session]\nbody\n[end of delta]\nanchor"
        events = [
            {"event": "meta", "room": "!r:1", "model": "m", "started": "t"},
            {"event": "delta", "pass_index": 0, "history_len": 2, "content": delta},
            {"event": "narration", "pass_index": 0, "content": "investigating"},
            {"event": "gc_boundary", "boundary_messages": 3, "reduced": 1, "ts": "t"},
            {"event": "verdict", "pass_index": 0, "status": "silent",
             "args": dict(SILENT_ARGS), "content": "", "history_len": 2,
             "tool_names": [], "usage": {}},
        ]
        t_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        st = SpotterRoomState()
        mgr._rehydrate("!r:1", st)
        roles = [(m.get("role"), "tool_calls" in m) for m in st.messages]
        # delta user, narration assistant, verdict tool-call pair
        assert roles == [("user", False), ("assistant", False),
                         ("assistant", True), ("tool", False)]
        assert st.last_index == 2

    def test_estimate_gate_still_backstops_after_gc(self, tmp_path, monkeypatch):
        # Tiny window: even with GC applied, overflow → exhausted (fail loud, D9).
        cfg = make_config(tmp_path, spotter_enabled=True, model_max_tokens=500)
        mgr, cfg2, ag = make_manager(tmp_path, config=cfg, window=600)
        hist = [{"role": "user", "content": "word " * 5000},
                {"role": "assistant", "content": "y"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        called = {"n": 0}

        async def no_complete(**kw):
            called["n"] += 1
            return resp()

        monkeypatch.setattr(spotter, "complete", no_complete)
        mgr.maybe_fire("!r:1", hist, None, {})
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_drain())
        finally:
            loop.close()
        st = mgr.ensure_state("!r:1")
        assert st.exhausted is True
        assert called["n"] == 0  # no provider call on exhaustion


# ---------------------------------------------------------------------------
# Rehydration v2: narration + verdict pairs + legacy canonicalization
# ---------------------------------------------------------------------------

class TestRehydrationV2:
    def _write_transcript(self, tmp_path, events):
        t = tmp_path / "sessions/spotters/r_1.jsonl"
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    def test_full_v2_rebuild(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path)
        delta = "[TRANSCRIPT DELTA — 1 new entries from the watched session]\nb\n[end of delta]\nanchor"
        self._write_transcript(tmp_path, [
            {"event": "meta", "room": "!r:1", "model": "m", "started": "t"},
            {"event": "delta", "pass_index": 0, "history_len": 2, "content": delta},
            {"event": "narration", "pass_index": 0, "content": "reading the file"},
            {"event": "verdict", "pass_index": 0, "status": "flag",
             "args": dict(FLAG_ARGS), "content": "", "history_len": 2,
             "tool_names": [], "usage": {}},
        ])
        st = SpotterRoomState()
        mgr._rehydrate("!r:1", st)
        assert [m.get("role") for m in st.messages] == ["user", "assistant", "assistant", "tool"]
        vc = st.messages[2]["tool_calls"][0]
        assert vc["name"] == "report_verdict"
        assert vc["input"]["status"] == "flag"
        assert st.messages[3]["tool_call_id"] == vc["id"]
        assert st.last_index == 2

    def test_legacy_text_verdicts_still_canonicalize(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path)
        self._write_transcript(tmp_path, [
            {"event": "meta", "room": "!r:1", "model": "m", "started": "t"},
            {"event": "delta", "pass_index": 0, "history_len": 2, "content": "D1"},
            {"event": "verdict", "pass_index": 0, "status": "silent",
             "content": "SILENT", "history_len": 2, "tool_names": [], "usage": {}},
        ])
        st = SpotterRoomState()
        mgr._rehydrate("!r:1", st)
        assert st.messages == [{"role": "user", "content": "D1"},
                               {"role": "assistant", "content": "SILENT"}]

    def test_orphan_tail_dropped_index_rewound(self, tmp_path):
        mgr, cfg, ag = make_manager(tmp_path)
        self._write_transcript(tmp_path, [
            {"event": "meta", "room": "!r:1", "model": "m", "started": "t"},
            {"event": "delta", "pass_index": 0, "history_len": 2, "content": "D1"},
            {"event": "verdict", "pass_index": 0, "status": "silent",
             "args": dict(SILENT_ARGS), "content": "", "history_len": 2,
             "tool_names": [], "usage": {}},
            {"event": "delta", "pass_index": 1, "history_len": 5, "content": "D2"},
            {"event": "narration", "pass_index": 1, "content": "mid-crash"},
        ])
        st = SpotterRoomState()
        mgr._rehydrate("!r:1", st)
        # trailing delta+narration dropped (crash mid-pass): index at 2, re-watched
        assert st.last_index == 2
        assert len(st.messages) == 3  # D1 + the verdict tool-call pair


# ---------------------------------------------------------------------------
# Prompt v2 contract (G3/G5/G12/G15 prompt parts)
# ---------------------------------------------------------------------------

class TestPromptV2Contract:
    def test_prompt_references_tool_contract(self):
        p = spotter.SPOTTER_SYSTEM_PROMPT
        assert "report_verdict" in p
        assert "status" in p and ("silent" in p and "flag" in p)

    def test_prompt_narrate_tightly(self):
        assert "narrate tightly" in spotter.SPOTTER_SYSTEM_PROMPT.lower()

    def test_prompt_midturn_segment_amendment(self):
        p = spotter.SPOTTER_SYSTEM_PROMPT.lower()
        assert "incompleteness" in p or "incomplete" in p
        assert "mid-turn" in p or "mid-turn segment" in p or "boundary" in p

    def test_prompt_feedback_loop_rules(self):
        p = spotter.SPOTTER_SYSTEM_PROMPT.lower()
        assert "not new evidence" in p
        assert "material action" in p or "premised" in p

    def test_prompt_narration_is_legal(self):
        p = spotter.SPOTTER_SYSTEM_PROMPT.lower()
        assert "narration" in p

    def test_re_anchor_line_names_tool(self):
        assert "report_verdict" in spotter._RE_ANCHOR_LINE
        assert "silent" in spotter._RE_ANCHOR_LINE.lower()

    def test_delta_frame_ends_with_re_anchor(self, tmp_path):
        framed = spotter.render_delta_frame("body", initial=False, n_entries=1)
        assert framed.rstrip().endswith(spotter._RE_ANCHOR_LINE)

    def test_placeholder_references_tool_and_keeps_silent_tail(self):
        ph = spotter._UNPARSED_VERDICT_TEXT
        assert "report_verdict" in ph
        assert ph.rstrip().endswith("SILENT")


# ---------------------------------------------------------------------------
# Ledger / op_status telemetry (G11)
# ---------------------------------------------------------------------------

class TestTelemetry:
    async def test_pass_ledger_gains_v2_fields(self, tmp_path, monkeypatch):
        script = CompleteScript(resp(tool_calls=[tc("v1", "report_verdict", dict(SILENT_ARGS))]))
        mgr, cfg, ag = make_manager(tmp_path)
        monkeypatch.setattr(spotter, "complete", script)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        led = [json.loads(l) for l in
               (tmp_path / "sessions/spotters/r_1.ledger.jsonl").read_text().splitlines()]
        ev = led[-1]
        for field_ in ("fallback", "forced", "siblings_ignored", "narration_turns"):
            assert field_ in ev, f"ledger pass event must carry {field_} (G11)"

    async def test_toolcall_verdict_wrapper_telemetry(self, tmp_path, monkeypatch):
        script = CompleteScript(resp(tool_calls=[tc("v1", "report_verdict", dict(FLAG_ARGS))]))
        mgr, cfg, ag = make_manager(tmp_path)
        monkeypatch.setattr(spotter, "complete", script)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        led = [json.loads(l) for l in
               (tmp_path / "sessions/spotters/r_1.ledger.jsonl").read_text().splitlines()]
        flag_ev = [e for e in led if e["event"] == "flag"][0]
        assert flag_ev["wrapper_type"] == "tool-call"
        assert flag_ev["wrapper_chars"] == 0

    async def test_op_status_shows_flag_counts_and_armed_state(self, tmp_path, monkeypatch):
        script = CompleteScript(resp(tool_calls=[tc("v1", "report_verdict", dict(FLAG_ARGS))]))
        mgr, cfg, ag = make_manager(tmp_path)
        monkeypatch.setattr(spotter, "complete", script)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
        ag.history["!r:1"] = hist
        mgr.op_start("!r:1")
        mgr.maybe_fire("!r:1", hist, None, {})
        await _drain()
        out = mgr.op_status("!r:1")
        assert "contradiction" in out  # per-class count visible (G11)
        assert "armed" in out.lower()
