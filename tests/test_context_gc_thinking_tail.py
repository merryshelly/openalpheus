"""Context GC — thinking-tail preservation (workspace-kdsn.305.13).

Orchestrator-authored red suite (spec:
memory/projects/openalph/context-gc/spec-kdsn.305.13-thinking-tail.md).
Sub-agents implement against these assertions and NEVER modify this file.

Pinned contract:
  - think_tail selection: the last N ELIGIBLE pre-boundary assistant
    entries, newest->oldest, contiguous-stop on first ceiling overflow;
    eligible = role assistant + non-empty thinking + (content OR tool_calls)
    — thought-only entries are always stripped/deleted (T1 eligibility).
  - build_context + apply_boundary_to_messages kwargs
    (thinking_tail_turns=0, thinking_tail_max_tokens=0) — default full
    strip; production call sites pass config values explicitly (T4).
  - [context] thinking_tail_turns (default 8) / thinking_tail_max_tokens
    (default 32768), fail-loud parse (T2).
  - Manifest: classes["thinking"] = stripped count, new key
    classes["thinking_retained"]; retained chars contribute to
    tokens_after_est (and thus the .12 runway composite) (T3/T3a/T5).
"""

import json
import pytest

from openalph.config import ConfigError, load_config
from openalph.context_gc import (
    GC_EVENT,
    apply_boundary,
    apply_boundary_and_rebuild,
    apply_boundary_to_messages,
)

try:  # pre-implementation the helper doesn't exist; selection tests must
    # still fail with individual identities, not a collection error.
    from openalph.context_gc import thinking_tail_indices
except ImportError:  # pragma: no cover - red-phase shim
    def thinking_tail_indices(*a, **kw):
        raise NotImplementedError("thinking_tail_indices not implemented")
from openalph.session import SessionLog

ROOM = "!gc-tail:matrix.local"
AGENT_ID = "@gc-tail-agent:matrix.local"


# ---------------------------------------------------------------------------
# Entry builders (JSONL shapes)
# ---------------------------------------------------------------------------

def _user(content):
    return {"role": "user", "content": content}


def _assistant(content="", tool_calls=None, thinking=None):
    e = {"role": "assistant"}
    if content:
        e["content"] = content
    if tool_calls:
        e["tool_calls"] = tool_calls
    if thinking:
        e["thinking"] = thinking
    return e


def _tc(call_id, name, input=None):
    return {"call_id": call_id, "name": name, "input": input or {}}


def _tool(call_id, name, output):
    return {"role": "tool", "call_id": call_id, "name": name, "output": output}


def _marker(event, entry_index, detail=None):
    e = {"role": "system", "event": event, "entry_index": entry_index}
    if detail is not None:
        e["detail"] = detail
    return e


def _log(tmp_path):
    return SessionLog(tmp_path, AGENT_ID)


def _append_all(log, entries):
    for e in entries:
        kw = {"role": e["role"], "sender": e.get("sender", AGENT_ID), "room": ROOM}
        kw.update({k: v for k, v in e.items() if k not in ("role", "sender", "room")})
        log.append(**kw)


def _ctx(log, **kw):
    kw.setdefault("gc_enabled", True)
    return log.build_context(ROOM, **kw)


BASE_TOML = '''
[agent]
name = "gc-tail-agent"
default_model = "anthropic/claude-sonnet-4-20250514"
max_tokens = 8192

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
'''


def _toml(tmp_path, body):
    p = tmp_path / "agent.toml"
    p.write_text(body)
    return load_config(p)


# Standard six-entry fixture for selection tests: boundary at position 6.
# Pre-boundary eligible thinking entries at 1 (str) and 5 (list blocks);
# position 3 is a thought-only entry (never eligible).
def _entries():
    return [
        _user("start"),                                                # 0
        _assistant("answer one", thinking="S" * 400,
                   tool_calls=[_tc("c1", "file_read", {"path": "/tmp/a.md"})]),  # 1
        _tool("c1", "file_read", "X" * 5000),                          # 2
        _assistant("", thinking="thought-only block"),                 # 3
        _user("question two"),                                         # 4
        _assistant("answer two",
                   thinking=[{"type": "thinking",
                              "thinking": "B" * 600,
                              "signature": "sig-abc"}]),               # 5
    ]


# ============================================================================
# Config surface (T2)
# ============================================================================

class TestThinkingTailConfig:
    def test_defaults(self, tmp_path):
        c = _toml(tmp_path, BASE_TOML).context
        assert c.thinking_tail_turns == 8
        assert c.thinking_tail_max_tokens == 32768

    def test_overrides(self, tmp_path):
        body = BASE_TOML + '''
[context]
thinking_tail_turns = 3
thinking_tail_max_tokens = 12000
'''
        c = _toml(tmp_path, body).context
        assert c.thinking_tail_turns == 3
        assert c.thinking_tail_max_tokens == 12000

    @pytest.mark.parametrize("line", [
        "thinking_tail_turns = -1",
        'thinking_tail_turns = "lots"',
        "thinking_tail_turns = 1.5",
        "thinking_tail_max_tokens = -1",
        'thinking_tail_max_tokens = "many"',
        "thinking_tail_max_tokens = 3.5",
    ])
    def test_invalid_fails_loud(self, tmp_path, line):
        with pytest.raises(ConfigError):
            _toml(tmp_path, BASE_TOML + f"[context]\n{line}\n")


# ============================================================================
# Selection helper (T1)
# ============================================================================

class TestTailSelection:
    def test_last_n_eligible(self):
        idx = thinking_tail_indices(_entries(), 6, 3, 32768)
        assert idx == {1, 5}  # only two eligible exist; N=3 not an error

    def test_n_one_picks_newest(self):
        assert thinking_tail_indices(_entries(), 6, 1, 32768) == {5}

    def test_zero_n_and_zero_ceiling_empty(self):
        assert thinking_tail_indices(_entries(), 6, 0, 32768) == frozenset()
        assert thinking_tail_indices(_entries(), 6, 3, 0) == frozenset()

    def test_no_boundary_empty(self):
        assert thinking_tail_indices(_entries(), -1, 3, 32768) == frozenset()

    def test_post_boundary_never_eligible(self):
        # Thinking at position 7 (>= boundary 6) must not be selected even
        # if it is the newest thinking in the list.
        entries = _entries() + [_user("post"), _assistant("a3", thinking="late")]
        idx = thinking_tail_indices(entries, 6, 4, 32768)
        assert 7 not in idx
        assert idx == {1, 5}

    def test_thought_only_never_consumes_slot(self):
        # Position 3 is thought-only: newest eligible is 5, then 1.
        idx = thinking_tail_indices(_entries(), 6, 2, 32768)
        assert idx == {1, 5}
        assert 3 not in idx

    def test_ceiling_contiguous_stop(self):
        # Newest (pos 5: ~600 chars -> 150 tokens) fits under 200 ceiling;
        # adding pos 1 (~400 chars -> 100 tokens) would exceed -> pos 5 ONLY.
        idx = thinking_tail_indices(_entries(), 6, 8, 200)
        assert idx == {5}
        # Under a 400 ceiling both fit (150 + 100 = 250).
        idx2 = thinking_tail_indices(_entries(), 6, 8, 400)
        assert idx2 == {1, 5}

    def test_ceiling_newest_alone_exceeds_gives_empty(self):
        # Newest block alone (150 tokens) exceeds 100 ceiling -> EMPTY
        # (stop-on-first-overflow; never skip to the older smaller block).
        idx = thinking_tail_indices(_entries(), 6, 8, 100)
        assert idx == frozenset()

    def test_returns_frozenset(self):
        assert isinstance(thinking_tail_indices(_entries(), 6, 3, 32768),
                          frozenset)


# ============================================================================
# Main render (T1/T4)
# ============================================================================

class TestTailRender:
    def _bounded_log(self, tmp_path):
        log = _log(tmp_path)
        _append_all(log, _entries())
        # Boundary at position 6 (after all six entries).
        _append_all(log, [
            _marker(GC_EVENT, 6, detail=json.dumps({"boundary_index": 6})),
            _user("post-boundary"),
        ])
        return log

    def test_retains_verbatim_str_and_blocks(self, tmp_path):
        ctx = _ctx(self._bounded_log(tmp_path),
                   thinking_tail_turns=2, thinking_tail_max_tokens=32768)
        thinking_by_content = {}
        for m in ctx:
            if m.get("role") == "assistant" and m.get("thinking"):
                thinking_by_content[m.get("content", "")] = m["thinking"]
        # Newest eligible: list-of-blocks, verbatim incl. signature.
        assert thinking_by_content["answer two"] == [
            {"type": "thinking", "thinking": "B" * 600, "signature": "sig-abc"}]
        # Older eligible: str form, verbatim.
        assert thinking_by_content["answer one"] == "S" * 400

    def test_thought_only_entry_still_deleted(self, tmp_path):
        ctx = _ctx(self._bounded_log(tmp_path),
                   thinking_tail_turns=4, thinking_tail_max_tokens=32768)
        # The thought-only pre-boundary entry is NEVER retained; no assistant
        # message with empty content and no tool_calls survives pre-boundary.
        shells = [m for m in ctx
                  if m.get("role") == "assistant"
                  and not m.get("content")
                  and not m.get("tool_calls")]
        assert shells == []
        # Its thinking text does not leak anywhere either.
        assert all("thought-only block" not in json.dumps(m, default=str)
                   for m in ctx)

    def test_surrounding_transforms_intact_on_retained(self, tmp_path):
        # Fixture where the retained assistant has a >500-char string input:
        # thinking retained, input STILL compacted pre-boundary (T1 keeps
        # the entry whole; only stripping is skipped).
        entries = [
            _user("start"),                                              # 0
            _assistant("big input", thinking="K" * 200,
                       tool_calls=[_tc("c1", "shell",
                                       {"command": "Y" * 900})]),        # 1
            _tool("c1", "shell", "ok"),                                  # 2
            _marker(GC_EVENT, 3,
                    detail=json.dumps({"boundary_index": 3})),           # 3
            _user("post"),                                               # 4
        ]
        log = _log(tmp_path)
        _append_all(log, entries)
        ctx = _ctx(log, thinking_tail_turns=1, thinking_tail_max_tokens=32768)
        a = next(m for m in ctx
                 if m.get("role") == "assistant" and m.get("content") == "big input")
        assert a["thinking"] == "K" * 200
        tc = a["tool_calls"][0]
        compacted = tc.input if isinstance(tc.input, dict) else {}
        assert compacted.get("command") == "[stripped: 900 chars]"

    def test_default_kwargs_are_full_strip(self, tmp_path):
        # T4 safety net: no tail kwargs == legacy full strip. Renders over
        # the SAME built log (renders are pure; never re-append between).
        log = self._bounded_log(tmp_path)
        ctx = _ctx(log)
        pre_assistants = [m for m in ctx
                          if m.get("role") == "assistant" and m.get("thinking")]
        assert pre_assistants == []
        # ...and the same render is byte-identical to explicit zeros.
        ctx_explicit = _ctx(log, thinking_tail_turns=0,
                            thinking_tail_max_tokens=0)
        assert ctx == ctx_explicit
        ctx_zero = _ctx(log, thinking_tail_turns=0,
                        thinking_tail_max_tokens=32768)
        assert ctx == ctx_zero

    def test_legacy_mode_tail_is_inert(self, tmp_path):
        # gc_enabled=False has its own pre-.13 semantics: legacy toolstrip
        # reduction KEEPS thinking (verified against main: the thinking-strip
        # branch is guarded on gc_enabled). The tail must be INERT there —
        # tail kwargs change nothing in legacy mode.
        log = self._bounded_log(tmp_path)
        ctx_plain = log.build_context(ROOM, gc_enabled=False)
        ctx_tail = log.build_context(ROOM, gc_enabled=False,
                                     thinking_tail_turns=8,
                                     thinking_tail_max_tokens=32768)
        assert ctx_tail == ctx_plain

    def test_legacy_toolstrip_marker_also_gets_tail(self, tmp_path):
        # Uniform rule: legacy markers are honored as boundaries; the tail
        # applies to them too.
        entries = _entries()
        entries.append(_marker("toolstrip", 6))
        entries.append(_user("post"))
        log = _log(tmp_path)
        _append_all(log, entries)
        ctx = _ctx(log, thinking_tail_turns=1, thinking_tail_max_tokens=32768)
        kept = [m["thinking"] for m in ctx
                if m.get("role") == "assistant" and m.get("thinking")]
        assert kept == [[{"type": "thinking", "thinking": "B" * 600,
                          "signature": "sig-abc"}]]

    def test_render_deterministic(self, tmp_path):
        log = self._bounded_log(tmp_path)
        a = _ctx(log, thinking_tail_turns=2, thinking_tail_max_tokens=32768)
        b = _ctx(log, thinking_tail_turns=2, thinking_tail_max_tokens=32768)
        assert a == b


# ============================================================================
# Subagent path parity (T4)
# ============================================================================

def _msgs():
    return [
        {"role": "user", "content": "task context"},
        {"role": "assistant", "content": "one", "thinking": "S" * 400,
         "tool_calls": [{"id": "c1", "name": "file_read",
                         "input": {"path": "/tmp/a.md"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "X" * 5000},
        {"role": "assistant", "content": "", "thinking": "thought only"},
        {"role": "assistant", "content": "two", "thinking": "B" * 600},
        {"role": "user", "content": "unrelated"},
    ]


class TestTailSubagent:
    def test_retains_last_n_and_counts(self):
        out = apply_boundary_to_messages(
            _msgs(), boundary_index=5, task_text="t",
            thinking_tail_turns=2, thinking_tail_max_tokens=32768)
        msgs = out["messages"]
        # Eligible pre-boundary (positions < 5): "one"(1) and "two"(4);
        # position 3's thought-only entry is ineligible and gets deleted.
        kept = [m["content"] for m in msgs if m.get("role") == "assistant"
                and m.get("thinking")]
        assert kept == ["one", "two"]
        assert all(m.get("thinking") != "thought only"
                   for m in msgs)

    def test_n_one_newest_only(self):
        out = apply_boundary_to_messages(
            _msgs(), boundary_index=5, task_text="t",
            thinking_tail_turns=1, thinking_tail_max_tokens=32768)
        kept = [m["content"] for m in out["messages"]
                if m.get("role") == "assistant" and m.get("thinking")]
        assert kept == ["two"]
        assert out["classes"]["thinking_retained"] == 1
        # Stripped count: "one" + the thought-only entry = 2.
        assert out["classes"]["thinking"] == 2

    def test_default_zero_keeps_legacy(self):
        legacy = apply_boundary_to_messages(_msgs(), boundary_index=5,
                                            task_text="t")
        explicit = apply_boundary_to_messages(
            _msgs(), boundary_index=5, task_text="t",
            thinking_tail_turns=0, thinking_tail_max_tokens=0)
        assert legacy == explicit
        assert [m for m in legacy["messages"]
                if m.get("role") == "assistant" and m.get("thinking")] == []


# ============================================================================
# Estimator + manifest (T3/T3a/T5)
# ============================================================================

def _apply(log, tmp_path, **kw):
    defaults = dict(
        workspace=tmp_path, trigger="manual", window=1_000_000,
        budget_pct=0.25, budget_min=96000,
        max_tokens=0, handoff_pct=0.0000001, handoff_min=0,
        bd_path=None,
    )
    defaults.update(kw)
    return apply_boundary(log, ROOM, **defaults)


class TestTailEstimator:
    def test_retained_adds_to_after_estimate(self, tmp_path):
        # Same fixture applied twice on sibling logs; the tail-inclusive
        # estimate must rise by exactly the retained block's chars//4.
        log0 = _log(tmp_path / "a")
        (tmp_path / "a").mkdir()
        _append_all(log0, _entries())
        m0 = _apply(log0, tmp_path / "a", thinking_tail_turns=0)

        log1 = _log(tmp_path / "b")
        (tmp_path / "b").mkdir()
        _append_all(log1, _entries())
        m1 = _apply(log1, tmp_path / "b", thinking_tail_turns=1)

        delta = (m1["manifest"]["tokens_after_est"]
                 - m0["manifest"]["tokens_after_est"])
        # Retained: pos 5 list-block, json-repr chars // 4 (deterministic).
        # after_chars is a summed-then-divided quantity, so a shared division
        # carry can shift the delta by exactly one; accept {q, q+1}.
        block = [{"type": "thinking", "thinking": "B" * 600,
                  "signature": "sig-abc"}]
        q = len(json.dumps(block, sort_keys=True)) // 4
        assert delta in (q, q + 1)

    def test_classes_new_key(self, tmp_path):
        log = _log(tmp_path)
        _append_all(log, _entries())
        r = _apply(log, tmp_path, thinking_tail_turns=1)
        classes = r["manifest"]["classes"]
        assert classes["thinking_retained"] == 1
        # Stripped = eligible-not-retained (pos 1) + thought-only (pos 3) = 2.
        assert classes["thinking"] == 2
        # Subagent path reports the key too (parity, T5).
        out = apply_boundary_to_messages(_msgs(), boundary_index=5,
                                         task_text="t")
        assert "thinking_retained" in out["classes"]

    def test_runway_composite_includes_retained(self, tmp_path):
        # The .12 composite tokens_after must grow with retention, so the
        # handoff gate sees the retained load.
        log = _log(tmp_path)
        _append_all(log, _entries())
        r = _apply(log, tmp_path, thinking_tail_turns=1)
        m = r["manifest"]
        assert m["runway"]["tokens_after"] >= (
            m["tokens_after_est"]
            + len(log.read(ROOM)[-1]["content"]) // 4)
        # (tail empty here: boundary appended at the end)


# ============================================================================
# Real-path plumbing (T4a)
# ============================================================================

class _StubAgent:
    def __init__(self, workspace, window, available, history_box, tail_turns):
        from types import SimpleNamespace
        ctx = SimpleNamespace(
            gc_enabled=True, durable_paths=[],
            durable_budget_pct=25.0, durable_budget_min_tokens=96000,
            handoff_runway_pct=10.0, handoff_runway_min_tokens=24000,
            thinking_tail_turns=tail_turns, thinking_tail_max_tokens=32768,
        )
        self.config = SimpleNamespace(context=ctx, workspace=workspace,
                                      model_max_tokens=window)
        self._window = window
        self._available = available
        self._history_box = history_box

    def _resolve_model_limit(self, room_id):
        return self._window

    def _effective_available(self, limit):
        assert limit == self._window
        return self._available

    def history(self, room_id):
        return self._history_box


class TestTailRealPath:
    def test_config_flows_into_rebuild(self, tmp_path):
        log = _log(tmp_path)
        _append_all(log, _entries())
        hist = [e.copy() for e in log.read(ROOM)]
        agent = _StubAgent(tmp_path, window=1_000_000, available=900_000,
                           history_box=hist, tail_turns=1)
        out = apply_boundary_and_rebuild(agent, log, ROOM,
                                         trigger="manual",
                                         exclude_inflight=False)
        assert out["applied"] is True
        kept = [m for m in hist
                if m.get("role") == "assistant" and m.get("thinking")]
        # tail_turns=1 -> only the newest eligible (list-block) survives.
        assert len(kept) == 1
        assert kept[0]["thinking"] == [
            {"type": "thinking", "thinking": "B" * 600,
             "signature": "sig-abc"}]

    def test_control_zero_config_full_strip(self, tmp_path):
        log = _log(tmp_path)
        _append_all(log, _entries())
        hist = [e.copy() for e in log.read(ROOM)]
        agent = _StubAgent(tmp_path, window=1_000_000, available=900_000,
                           history_box=hist, tail_turns=0)
        out = apply_boundary_and_rebuild(agent, log, ROOM,
                                         trigger="manual",
                                         exclude_inflight=False)
        assert out["applied"] is True
        assert [m for m in hist
                if m.get("role") == "assistant" and m.get("thinking")] == []


# ============================================================================
# gc_thinking_tail_kwargs helper (audit remediation: the wiring seam itself)
# ============================================================================

class TestTailKwargsHelper:
    def test_real_config_defaults_pass_through(self, tmp_path):
        from openalph.context_gc import gc_thinking_tail_kwargs
        cfg = _toml(tmp_path, BASE_TOML)
        kw = gc_thinking_tail_kwargs(cfg)
        assert kw == {"thinking_tail_turns": 8,
                      "thinking_tail_max_tokens": 32768}

    def test_real_config_overrides_pass_through(self, tmp_path):
        from openalph.context_gc import gc_thinking_tail_kwargs
        cfg = _toml(tmp_path, BASE_TOML + '''
[context]
thinking_tail_turns = 2
thinking_tail_max_tokens = 4000
''')
        kw = gc_thinking_tail_kwargs(cfg)
        assert kw == {"thinking_tail_turns": 2,
                      "thinking_tail_max_tokens": 4000}

    def test_mock_and_absent_config_fail_closed(self, tmp_path):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from openalph.context_gc import gc_thinking_tail_kwargs
        # Full MagicMock agent config: NOT a real ContextGCConfig → 0/0.
        assert gc_thinking_tail_kwargs(MagicMock()) == {
            "thinking_tail_turns": 0, "thinking_tail_max_tokens": 0}
        # Context section absent entirely → 0/0.
        assert gc_thinking_tail_kwargs(SimpleNamespace()) == {
            "thinking_tail_turns": 0, "thinking_tail_max_tokens": 0}
        # Context exists but fields missing (partial mock) → 0/0.
        assert gc_thinking_tail_kwargs(
            SimpleNamespace(context=SimpleNamespace())) == {
            "thinking_tail_turns": 0, "thinking_tail_max_tokens": 0}

    def test_bad_field_types_fail_closed_per_field(self, tmp_path):
        from types import SimpleNamespace
        from openalph.context_gc import gc_thinking_tail_kwargs
        ctx = SimpleNamespace(thinking_tail_turns=True,
                              thinking_tail_max_tokens=32768)
        kw = gc_thinking_tail_kwargs(SimpleNamespace(context=ctx))
        assert kw["thinking_tail_turns"] == 0  # bool is not int here
        assert kw["thinking_tail_max_tokens"] == 32768
        ctx2 = SimpleNamespace(thinking_tail_turns=-3,
                               thinking_tail_max_tokens="many")
        kw2 = gc_thinking_tail_kwargs(SimpleNamespace(context=ctx2))
        assert kw2 == {"thinking_tail_turns": 0, "thinking_tail_max_tokens": 0}
