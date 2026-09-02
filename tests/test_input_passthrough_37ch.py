"""workspace-37ch — assistant tool-call INPUT compaction removed.

Pre-boundary assistant tool-call inputs (string values of ANY length,
incl. >500) now render VERBATIM in the context-GC / toolstrip render
transform.  Result-side placeholders, thinking stripping, and media
expunge are UNCHANGED.  The "inputs" manifest class is retired (pinned 0
for schema stability); the ``legacy_input_placeholder`` generator is
deleted.  Fixture patterns follow tests/test_context_gc.py and
tests/test_toolstrip.py.
"""

from pathlib import Path

from openalph.session import SessionLog
from openalph.context_gc import (
    apply_boundary,
    apply_boundary_to_messages,
    legacy_tool_placeholder,
    tool_pointer,
)

ROOM = "!37ch:matrix.local"
AGENT_ID = "@37ch-agent:matrix.local"
USER = "@sb:matrix.local"


# ---------------------------------------------------------------------------
# Entry builders (JSONL dict shapes — same conventions as test_context_gc.py)
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


def _log(tmp_path):
    return SessionLog(workspace=tmp_path, agent_user_id=AGENT_ID)


def _append_all(log, entries):
    for e in entries:
        kw = {"role": e["role"], "sender": e.get("sender", AGENT_ID), "room": ROOM}
        kw.update({k: v for k, v in e.items() if k not in ("role", "sender", "room")})
        log.append(**kw)


def _input_strings(msg):
    """All string values of a rendered msg's tool_call inputs."""
    out = []
    for tc in msg.get("tool_calls") or []:
        inp = tc.input if hasattr(tc, "input") else tc.get("input")
        if isinstance(inp, dict):
            out.extend(v for v in inp.values() if isinstance(v, str))
        elif isinstance(inp, str):
            out.append(inp)
    return out


class TestT1BuildContextGCBoundary:
    """build_context w/ GC boundary: 2000-char input verbatim, result is a
    GC pointer, thinking stripped per existing rules."""

    def test_pre_boundary_input_verbatim_result_pointer_thinking_stripped(self, tmp_path):
        big = "x" * 2000
        log = _log(tmp_path)
        _append_all(log, [
            _user("write it"),                                    # 0
            _assistant("", tool_calls=[_tc("c1", "file_write",
                                           {"path": "/tmp/t.md",
                                            "content": big})],
                       thinking="deep planning"),                 # 1
            _tool("c1", "file_write", "ok"),                      # 2
            _user("done?"),                                       # 3
        ])
        r = apply_boundary(log, ROOM, workspace=tmp_path, trigger="manual",
                           window=1_000_000, budget_pct=0.15, budget_min=48000)
        assert r["applied"] is True
        boundary = r["manifest"]["boundary_index"]

        ctx = log.build_context(ROOM, gc_enabled=True)
        a = [m for m in ctx if m.get("role") == "assistant"][0]
        # INPUT: byte-identical to the JSONL content — no input placeholder.
        assert a["tool_calls"][0].input == {"path": "/tmp/t.md", "content": big}
        assert a["tool_calls"][0].input["content"] == big
        for v in _input_strings(a):
            assert "[stripped:" not in v
        # THINKING: stripped pre-boundary (existing rule, unchanged).
        assert "thinking" not in a
        # RESULT: GC pointer, pairing cites the ORIGINAL input.
        t = [m for m in ctx if m.get("tool_call_id") == "c1"][0]
        expected = tool_pointer(boundary, "file_write",
                                {"path": "/tmp/t.md", "content": big}, 2)
        assert t["content"] == expected
        assert "expunged at GC boundary" in t["content"]


class TestT2LegacyToolstripBoundary:
    """Legacy toolstrip boundary (gc disabled): input verbatim; the legacy
    result placeholder family is unchanged."""

    def test_legacy_input_verbatim_result_placeholder_unchanged(self, tmp_path):
        big = "y" * 2000
        log = _log(tmp_path)
        _append_all(log, [
            _user("write it"),                                    # 0
            _assistant("", tool_calls=[_tc("c1", "file_write",
                                           {"path": "/tmp/t.md",
                                            "content": big})]),   # 1
            _tool("c1", "file_write", "wrote"),                   # 2
        ])
        log.append(role="system", sender=USER, room=ROOM,
                   event="toolstrip", entry_index=3)

        ctx = log.build_context(ROOM)  # default gc: legacy toolstrip mode
        a = [m for m in ctx if m.get("role") == "assistant"][0]
        assert a["tool_calls"][0].input == {"path": "/tmp/t.md", "content": big}
        assert a["tool_calls"][0].input["content"] == big
        for v in _input_strings(a):
            assert "[stripped:" not in v
        # Legacy result placeholder: byte-exact, UNCHANGED family.
        t = [m for m in ctx if m["role"] == "tool"][0]
        assert t["content"] == legacy_tool_placeholder("file_write", 5)


class TestT3ApplyBoundaryToMessages:
    """apply_boundary_to_messages: >500-char input verbatim; pointer pairing
    still cites the original identifying param (raw input); classes.inputs
    pinned 0; tokens_after_est counts the FULL input length (current
    semantics pinned — must not change)."""

    def test_verbatim_input_raw_pairing_inputs_zero_full_len_estimate(self):
        path = "p" * 700
        scene = [
            {"role": "assistant", "content": "",
             "tool_calls": [{"call_id": "c1", "name": "file_read",
                             "input": {"path": path}}]},            # 0
            {"role": "tool", "tool_call_id": "c1", "content": "O" * 100},  # 1
            {"role": "user", "content": "after"},                  # 2
        ]
        r = apply_boundary_to_messages(scene, boundary_index=2, task_text="t")
        rendered = r["messages"][0]["tool_calls"][0]
        # >500-char input value passes VERBATIM (dict shape preserved).
        assert isinstance(rendered, dict)
        assert rendered["input"] == {"path": path}
        assert "[stripped:" not in rendered["input"]["path"]
        # The paired tool pointer STILL cites the original identifying
        # param (pairing uses the raw input), truncated to 80.
        pointer = tool_pointer(2, "file_read", {"path": path}, 100)
        assert r["messages"][1]["content"] == pointer
        assert "p" * 80 in pointer
        assert "[stripped" not in pointer
        # "inputs" class retired: key retained, pinned 0.
        assert r["manifest"]["classes"]["inputs"] == 0
        # after-estimate counts the FULL input length (char//4).
        exp_after = (700 + len(pointer)) // 4
        assert r["manifest"]["tokens_after_est"] == exp_after


class TestT4NoLegacyInputGenerator:
    """Nothing in src/ calls legacy_input_placeholder; the function is
    absent from context_gc's module and __all__."""

    def test_function_absent_from_module_and_all(self):
        import openalph.context_gc as cg
        assert not hasattr(cg, "legacy_input_placeholder")
        assert "legacy_input_placeholder" not in cg.__all__

    def test_no_live_use_anywhere_in_src(self):
        import openalph.context_gc as cg
        src_root = Path(cg.__file__).resolve().parent
        live = []
        for py in sorted(src_root.rglob("*.py")):
            for i, line in enumerate(py.read_text().splitlines(), 1):
                if "legacy_input_placeholder" not in line:
                    continue
                # The [stripped: N chars] family survives in the sentry
                # regex for LEGACY pre-37ch contexts only; a comment
                # cross-reference is fine — any NON-comment occurrence
                # (a def, a call, an import) is a live use.
                if not line.strip().startswith("#"):
                    live.append(f"{py}:{i}: {line.strip()}")
        assert live == [], live

    def test_input_marker_regex_family_still_covered(self):
        # The deleted family is still matched at the trust boundary for
        # legacy render bytes and model regurgitation.
        from openalph.tools import _GC_PLACEHOLDER_RE
        assert _GC_PLACEHOLDER_RE.match("[stripped: 4760 chars]")
