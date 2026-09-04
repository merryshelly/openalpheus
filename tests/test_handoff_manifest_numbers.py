"""Red suite: handoff manifest token accounting (kdsn.322.14).

Pins the unified composite accounting ruled by SB 2026-09-04 — the
notice and /status must share one arithmetic:

- tokens_before: COMPOSITE — system prompt + tool defs + full pre-boundary
  render (content, tool outputs, thinking, signatures, wire overheads).
- tokens_after: MEASURED composite post-boundary — system prompt + tool
  defs + framed snapshot + surviving tail. Never the pinned 0.
- tokens_dropped: render-only pre-boundary figure (what the strip bought),
  kept as its own key.
- tokens_after_est is GONE from new manifests (legacy manifests with it
  are handled fail-soft by the renderer — see test_handoff_notice_render).
- runway block arithmetic UNCHANGED (ladder semantics preserved): its
  tokens_after stays the message-side snapshot+tail composite.
- outcome carries progress_md FROZEN at apply time (the notice fold
  renders what the agent received, not a live disk read).

The estimator rules replicated here are the documented spec: chars//4,
per-message tool-call overhead, tool-role wire overhead, thinking +
signature chars — the same rules agent._estimate_context_tokens applies.
"""

from pathlib import Path


from openalph import handoff as m
from openalph.config import ContextHandoffConfig

ROOM = "!manifest-numbers:matrix.local"


# --- fixtures -------------------------------------------------------------

def _make_log(tmp_path):
    from openalph.session import SessionLog
    return SessionLog(tmp_path, "@agent:matrix.local", handoff_default=True)


def _agent_stub(tmp_path, window=100_000):
    """Agent stub carrying the composite-estimator inputs.

    system_prompt 4000 chars -> 1000 tok; _tool_defs_chars 8000 -> 2000 tok.
    """

    class _Cfg:
        context = ContextHandoffConfig()
        workspace = Path(tmp_path)
        model_max_tokens = window
        max_tokens = 100

    class _Hist(list):
        pass

    class _Agent:
        config = _Cfg()
        system_prompt = "S" * 4000
        _tool_defs_chars = 8000
        _history = {}

        def history(self, room_id):
            if room_id not in self._history:
                self._history[room_id] = _Hist()
            return self._history[room_id]

        def _resolve_model_limit(self, room_id):
            return window

        def _effective_available(self, limit):
            return limit - 100

    return _Agent()


def _entries():
    """Small scene with text, thinking+signature, and a tool output."""
    return [
        {"role": "user", "content": "u" * 400},
        {"role": "assistant", "content": "a" * 200,
         "thinking": [{"thinking": "t" * 100, "signature": "s" * 20}]},
        {"role": "tool", "content": "", "output": "o" * 400},
    ]


def _render_chars_of_entries(entries):
    """Test-local replica of the documented render accounting (chars).

    Mirrors agent._estimate_context_tokens' per-message rules over the
    message-dict shape build_context produces: content (str or text
    blocks), tool_calls input JSON + overhead, thinking + signature,
    tool-role wire overhead. Tool OUTPUTS arrive as tool-role content.
    """
    from openalph.agent import _TOOL_CALL_OVERHEAD_CHARS, _TOOL_RESULT_OVERHEAD_CHARS
    total = 0
    for e in entries:
        role = e.get("role")
        if role == "tool":
            out = e.get("output", e.get("content", ""))
            total += (len(out) if isinstance(out, str) else 0)
            total += _TOOL_RESULT_OVERHEAD_CHARS
            continue
        c = e.get("content", "")
        total += len(c) if isinstance(c, str) else 0
        for tc in e.get("tool_calls") or []:
            total += len(str(tc.get("input", tc))) + _TOOL_CALL_OVERHEAD_CHARS
        for tb in e.get("thinking") or []:
            total += len(tb.get("thinking", "")) + len(tb.get("signature", ""))
    return total


# --- tests ----------------------------------------------------------------

class TestCompositeManifestNumbers:
    def test_tokens_before_is_composite(self, tmp_path):
        """tokens_before = sp + tool defs + full pre-boundary render."""
        agent = _agent_stub(tmp_path)
        log = _make_log(tmp_path)
        for e in _entries():
            log.append(role=e["role"], content=e.get("content", ""),
                       room=ROOM, sender="@op:matrix.local",
                       output=e.get("output"),
                       thinking=e.get("thinking"))
        res = m.apply_boundary_and_rebuild(
            agent, log, ROOM, trigger="tool", exclude_inflight=False)
        assert res["applied"] is True
        mf = res["manifest"]
        sp_td = (len(agent.system_prompt) + agent._tool_defs_chars) // 4
        render_tok = _render_chars_of_entries(_entries()) // 4
        assert mf["tokens_before"] == sp_td + render_tok, (
            f"tokens_before must be composite: sp+tools ({sp_td}) + render "
            f"({render_tok}); got {mf['tokens_before']}")

    def test_tokens_after_is_measured_composite(self, tmp_path):
        """tokens_after = sp + tool defs + snapshot (+ tail); settled
        exclude_inflight=False boundary -> tail is 0."""
        agent = _agent_stub(tmp_path)
        log = _make_log(tmp_path)
        for e in _entries():
            log.append(role=e["role"], content=e.get("content", ""),
                       room=ROOM, sender="@op:matrix.local",
                       output=e.get("output"), thinking=e.get("thinking"))
        res = m.apply_boundary_and_rebuild(
            agent, log, ROOM, trigger="tool", exclude_inflight=False)
        mf = res["manifest"]
        sp_td = (len(agent.system_prompt) + agent._tool_defs_chars) // 4
        # The snapshot is the only rendering entry at/after the boundary.
        entries = log.read(ROOM)
        snap = next(e for e in entries
                    if e.get("source") == "handoff_snapshot")
        assert mf["tokens_after"] == sp_td + len(snap.get("content", "")) // 4
        assert mf["tokens_after"] > 0, "measured, never the pinned 0"

    def test_tokens_dropped_is_render_only(self, tmp_path):
        """tokens_dropped = pre-boundary render (what the strip bought),
        WITHOUT the sp/tool-defs constant."""
        agent = _agent_stub(tmp_path)
        log = _make_log(tmp_path)
        for e in _entries():
            log.append(role=e["role"], content=e.get("content", ""),
                       room=ROOM, sender="@op:matrix.local",
                       output=e.get("output"), thinking=e.get("thinking"))
        res = m.apply_boundary_and_rebuild(
            agent, log, ROOM, trigger="tool", exclude_inflight=False)
        mf = res["manifest"]
        expected = _render_chars_of_entries(_entries()) // 4
        assert mf["tokens_dropped"] == expected
        assert mf["tokens_dropped"] < mf["tokens_before"]

    def test_tokens_after_est_gone(self, tmp_path):
        agent = _agent_stub(tmp_path)
        log = _make_log(tmp_path)
        log.append(role="user", content="u" * 400, room=ROOM,
                   sender="@op:matrix.local")
        res = m.apply_boundary_and_rebuild(
            agent, log, ROOM, trigger="tool", exclude_inflight=False)
        assert "tokens_after_est" not in res["manifest"]

    def test_tokens_before_counts_only_rendered_span(self, tmp_path):
        """SB canary bug (2026-09-04, ~104,597 inflation): tokens_before/
        tokens_dropped must count only the RENDERED pre-boundary span —
        entries below the PREVIOUS boundary are already dead (stripped,
        never rendered). The figure must not grow with every boundary."""
        agent = _agent_stub(tmp_path)
        log = _make_log(tmp_path)
        # bulk that dies at boundary 1
        log.append(role="user", content="D" * 8000, room=ROOM,
                   sender="@op:matrix.local")
        res1 = m.apply_boundary_and_rebuild(
            agent, log, ROOM, trigger="tool", exclude_inflight=False)
        assert res1["applied"] is True
        # fresh, small content above boundary 1, then boundary 2
        log.append(role="user", content="L" * 400, room=ROOM,
                   sender="@op:matrix.local")
        res2 = m.apply_boundary_and_rebuild(
            agent, log, ROOM, trigger="tool", exclude_inflight=False)
        assert res2["applied"] is True
        mf2 = res2["manifest"]
        # the 8000-char bulk is DEAD — it must not appear in the figures
        assert mf2["tokens_dropped"] < 400, (
            f"tokens_dropped counted dead pre-boundary content: "
            f"{mf2['tokens_dropped']} (rendered span is ~200 tok)")
        constant = (len(agent.system_prompt) + agent._tool_defs_chars) // 4
        assert mf2["tokens_before"] == constant + mf2["tokens_dropped"], (
            f"tokens_before must be constant + rendered-dropped: "
            f"{mf2['tokens_before']} != {constant} + {mf2['tokens_dropped']}")
        # and the rendered span IS counted: the new 400-char entry (~100)
        # + the old snapshot (~60) must dominate the figure
        assert mf2["tokens_dropped"] >= 100

    def test_runway_block_arithmetic_unchanged(self, tmp_path):
        """The runway block keeps its message-side composite (snapshot +
        tail, NO sp/tool defs) — ladder semantics must not drift."""
        agent = _agent_stub(tmp_path)
        log = _make_log(tmp_path)
        log.append(role="user", content="u" * 400, room=ROOM,
                   sender="@op:matrix.local")
        res = m.apply_boundary_and_rebuild(
            agent, log, ROOM, trigger="tool", exclude_inflight=False)
        entries = log.read(ROOM)
        snap = next(e for e in entries
                    if e.get("source") == "handoff_snapshot")
        mf = res["manifest"]
        assert mf["runway"]["tokens_after"] == len(snap.get("content", "")) // 4


class TestFrozenProgressMd:
    def _declare(self, log, project="p1"):
        log.append(role="system", event="active_project", room=ROOM,
                   sender="@agent:matrix.local", detail=project)

    def _scaffold(self, tmp_path, progress_text):
        pdir = Path(tmp_path) / "memory" / "projects" / "p1"
        pdir.mkdir(parents=True)
        (pdir / "progress.md").write_text(progress_text)
        (pdir / "durable-set.toml").write_text(
            "# durable set\n# progress.md is auto-inserted; do NOT list it\n")

    def test_outcome_carries_frozen_progress_md(self, tmp_path):
        agent = _agent_stub(tmp_path)
        self._scaffold(tmp_path, "FROZEN-PROGRESS-TEXT")
        log = _make_log(tmp_path)
        self._declare(log)
        log.append(role="user", content="u" * 400, room=ROOM,
                   sender="@op:matrix.local")
        res = m.apply_boundary_and_rebuild(
            agent, log, ROOM, trigger="tool", exclude_inflight=False)
        assert res.get("progress_md") == "FROZEN-PROGRESS-TEXT"

    def test_progress_md_frozen_not_live(self, tmp_path):
        """Editing the file AFTER the apply must not change the outcome —
        the fold renders what the agent received, not a live disk read."""
        agent = _agent_stub(tmp_path)
        self._scaffold(tmp_path, "FROZEN-PROGRESS-TEXT")
        log = _make_log(tmp_path)
        self._declare(log)
        log.append(role="user", content="u" * 400, room=ROOM,
                   sender="@op:matrix.local")
        res = m.apply_boundary_and_rebuild(
            agent, log, ROOM, trigger="tool", exclude_inflight=False)
        (Path(tmp_path) / "memory" / "projects" / "p1" / "progress.md") \
            .write_text("CHANGED-AFTER-APPLY")
        assert res.get("progress_md") == "FROZEN-PROGRESS-TEXT"

    def test_no_project_progress_md_is_none(self, tmp_path):
        agent = _agent_stub(tmp_path)
        log = _make_log(tmp_path)
        log.append(role="user", content="u" * 400, room=ROOM,
                   sender="@op:matrix.local")
        res = m.apply_boundary_and_rebuild(
            agent, log, ROOM, trigger="tool", exclude_inflight=False)
        assert res.get("progress_md") is None
        # composite numbers still present
        mf = res["manifest"]
        assert mf["tokens_before"] > 0 and mf["tokens_after"] > 0
