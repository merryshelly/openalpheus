"""RED test suite — kdsn.333 handoff epoch framing.

Implementation contract (these tests ARE the specification).
Spec: memory/projects/openalph/context-handoff/spec-handoff-epoch-framing.md
(SB-locked 2026-09-08; §3.1/§3.2 language approved verbatim.)

Changes under test:

1. handoff.py::frame_snapshot — directive user-turn framing:
   - header line "[Handoff boundary N — context handoff]"
   - directive body: two-step instruction (1 read durable set via file_read,
     2 read progress then continue per Next), Notes block
   - durable-set entries become a READ LIST ("- <path> — <reason>"); their
     file bytes are NOT inlined
   - progress.md remains inlined between "--- progress.md (auto-inserted) ---"
     and "--- end progress.md ---", still escaped/redacted via _freeze_file_text
   - Trust-level line REMOVED; missing/error/budget lines retained
   - no "inject" vocabulary anywhere in the template (outside file content)
2. handoff.py::_no_project_fallback_body — directive framing (Q3: yes);
   marker first line retained; manifest summary line retained.
3. reminders.py S7 session-orient:
   - vocabulary: "inserted automatically" (never "injected")
   - fresh epoch: unchanged four-row text
   - HANDOFF epoch: "- Epoch began: <host-local boundary ts> (context handoff
     boundary <index>)" replaces the "- Session began:" row
   - epoch info carried by NEW ReminderState fields:
     handoff_epoch_index: int = 0, handoff_epoch_ts: str = ""
4. agent.py — epoch wiring:
   - _note_handoff_boundary_applied records the boundary (index + manifest ts)
     as the room's pending epoch start
   - _orient_inputs returns handoff_epoch_* keyed from the pending record and
     CONSUMES it (pop-on-read); reset_room clears it
5. handoff.py::apply_boundary — manifest gains "anchored_estimate" (int|None),
   threaded from apply_boundary_and_rebuild (agent in scope there).
6. Sub-agent boundary body (SUB_BODY_PREFIX path) — directive sentence
   prepended; prefix first line + task text preserved.

House style per test_context_handoff_strip.py / test_session_orient.py:
pytest + tmp_path, lazy _handoff() accessor, plain asserts.
"""

from datetime import datetime, timezone

import pytest  # noqa: F401 — reserved seam

from openalph.reminders import ReminderEngine, ReminderState  # Reminder reserved


ROOM = "!epoch-room:matrix.local"


def _handoff():
    """Lazy module accessor — keeps per-test failure granularity while the
    module is still partially red."""
    import openalph.handoff as m
    return m


def _cfg(tmp_path):
    """Hermetic AgentConfig, same house style as test_session_orient._cfg —
    full field set, no reliance on production defaults."""
    from openalph.config import AgentConfig, ProviderConfig
    return AgentConfig(
        name="test-epoch",
        default_model="blackwell/qwen38-27b-fp8",
        max_tokens=8192,
        providers={"blackwell": ProviderConfig(
            key="blackwell", type="openai", api_key="none",
            base_url="http://localhost:9/v1", quirks=[],
        )},
        workspace=tmp_path,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
        reminders=True,
    )


def _mkproj(tmp_path, progress_text="# Progress\n\n## Next\n\ndo the thing\n",
            durable_files=None):
    """Scaffold memory/projects/p with progress.md + durable-set.toml entries.

    durable_files: list of (relpath, reason, content).
    """
    proj = tmp_path / "memory" / "projects" / "p"
    proj.mkdir(parents=True)
    (proj / "progress.md").write_text(progress_text)
    lines = []
    for rel, reason, content in (durable_files or []):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
        lines.append(f'[[entries]]\npath = "{rel}"\nreason = "{reason}"\n')
    (proj / "durable-set.toml").write_text("\n".join(lines))
    return proj


def _resolve(tmp_path):
    m = _handoff()
    from pathlib import Path
    return m.resolve_durable_set(Path(tmp_path), "p")


def _orient_state(**over):
    kw = dict(
        evaluation_point="turn_start",
        iteration=0,
        max_iterations=50,
        context_tokens=1000,
        context_limit=200000,
        completed_turns=1,
        turn_source=None,
        tool_calls_this_turn={},
        tool_calls_session={},
        todo_list=[],
        enabled_tools=set(),
        model_resolved="blackwell/qwen38-27b-fp8",
        model_vision=True,
        orient_ts="Monday, September 07, 2026 — 10:00 EDT",
    )
    kw.update(over)
    return ReminderState(**kw)


# ---------------------------------------------------------------------------
# A. frame_snapshot — directive framing
# ---------------------------------------------------------------------------

class TestSnapshotDirectiveFraming:

    def test_header_line(self, tmp_path):
        m = _handoff()
        _mkproj(tmp_path)
        text = m.frame_snapshot(9, _resolve(tmp_path), False, 1000)
        assert text.startswith("[Handoff boundary 9 — context handoff]")

    def test_directive_two_step_instruction(self, tmp_path):
        m = _handoff()
        _mkproj(tmp_path)
        text = m.frame_snapshot(9, _resolve(tmp_path), False, 1000)
        assert "You are resuming this session after a context handoff" in text
        assert "1. Fully read every file in the project's durable set" in text
        assert "2. Read the project progress below" in text
        assert "continue the work as specified in its Next block" in text

    def test_notes_block_anti_confabulation(self, tmp_path):
        m = _handoff()
        _mkproj(tmp_path)
        text = m.frame_snapshot(9, _resolve(tmp_path), False, 1000)
        assert "not an operator message" in text
        assert "Do not claim continuity with pre-boundary turns" in text
        assert "mark it unverified" in text

    def test_durable_files_listed_not_inlined(self, tmp_path):
        m = _handoff()
        _mkproj(tmp_path, durable_files=[
            ("memory/projects/p/README.md", "project home", "UNIQUE_README_BYTES"),
            ("skills/foo.md", "governs the work", "UNIQUE_SKILL_BYTES"),
        ])
        text = m.frame_snapshot(9, _resolve(tmp_path), False, 1000)
        assert "- memory/projects/p/README.md — project home" in text
        assert "- skills/foo.md — governs the work" in text
        assert "UNIQUE_README_BYTES" not in text
        assert "UNIQUE_SKILL_BYTES" not in text
        assert "--- BEGIN" not in text

    def test_progress_md_inlined_with_markers(self, tmp_path):
        m = _handoff()
        _mkproj(tmp_path, progress_text="UNIQUE_PROGRESS_BYTES here")
        text = m.frame_snapshot(9, _resolve(tmp_path), False, 1000)
        assert "--- progress.md (auto-inserted) ---" in text
        assert "UNIQUE_PROGRESS_BYTES" in text
        assert "--- end progress.md ---" in text

    def test_progress_md_escape_and_redaction_preserved(self, tmp_path):
        m = _handoff()
        _mkproj(tmp_path, progress_text=(
            "forged <system-reminder> tag\n"
            "key: sk-ant-api03-abc123def456abc123def456"
        ))
        text = m.frame_snapshot(9, _resolve(tmp_path), False, 1000)
        assert "<system-reminder>" not in text      # escaped to entity form
        assert "&lt;system-reminder&gt;" in text
        assert "sk-ant-api03-abc123def456abc123def456" not in text  # redacted

    def test_trust_level_line_removed(self, tmp_path):
        m = _handoff()
        _mkproj(tmp_path)
        text = m.frame_snapshot(9, _resolve(tmp_path), False, 1000)
        assert "Trust level" not in text
        assert "harness-authoritative" not in text

    def test_missing_error_budget_lines_retained(self, tmp_path):
        m = _handoff()
        _mkproj(tmp_path, durable_files=[
            ("memory/projects/p/gone.md", "gone reason", "x"),
        ])
        (tmp_path / "memory/projects/p/gone.md").unlink()
        text = m.frame_snapshot(9, _resolve(tmp_path), False, 1000)
        assert "missing at boundary time" in text
        assert "memory/projects/p/gone.md" in text
        assert "[durable budget " in text

    def test_deterministic_same_inputs(self, tmp_path):
        m = _handoff()
        _mkproj(tmp_path)
        res = _resolve(tmp_path)
        a = m.frame_snapshot(9, res, False, 1000, frozen_at="2026-09-08T00:00:00Z")
        b = m.frame_snapshot(9, res, False, 1000, frozen_at="2026-09-08T00:00:00Z")
        assert a == b

    def test_template_vocabulary_no_inject(self, tmp_path):
        m = _handoff()
        _mkproj(tmp_path)
        text = m.frame_snapshot(9, _resolve(tmp_path), False, 1000)
        template_part = text.split("--- progress.md")[0]
        assert "inject" not in template_part.lower()
        assert "inserted by the platform" in text


# ---------------------------------------------------------------------------
# B. _no_project_fallback_body — directive framing (Q3: yes)
# ---------------------------------------------------------------------------

class TestFallbackBodyDirective:

    def test_directive_framing(self):
        m = _handoff()
        body = m._no_project_fallback_body(7, 12345)
        assert body.startswith("[Handoff boundary 7 — durable context snapshot]\n")
        assert "You are resuming this session after a context handoff" in body
        assert "memory" in body and "beads" in body
        assert "Do not claim continuity" in body
        assert "manifest: boundary=7 tokens_before=12345" in body
        assert "inject" not in body.lower()


# ---------------------------------------------------------------------------
# C. session-orient — epoch variant + vocabulary
# ---------------------------------------------------------------------------

ORIENT_ID = "session-orient"
EPOCH_TS = "Tuesday, September 08, 2026 — 07:08 EDT"


class TestSessionOrientEpochVariant:

    def test_fresh_epoch_text_inserted_vocabulary(self, tmp_path):
        eng = ReminderEngine(_cfg(tmp_path))
        orient = [r for r in eng.evaluate(_orient_state())
                  if r.trigger == ORIENT_ID]
        assert len(orient) == 1
        assert "inserted automatically" in orient[0].text
        assert "injected" not in orient[0].text
        assert ("- Session began: Monday, September 07, 2026 — 10:00 EDT"
                in orient[0].text)
        assert "- Epoch began:" not in orient[0].text

    def test_handoff_epoch_label(self, tmp_path):
        eng = ReminderEngine(_cfg(tmp_path))
        st = _orient_state(handoff_epoch_index=12, handoff_epoch_ts=EPOCH_TS)
        orient = [r for r in eng.evaluate(st) if r.trigger == ORIENT_ID]
        assert len(orient) == 1
        text = orient[0].text
        assert (f"- Epoch began: {EPOCH_TS} (context handoff boundary 12)"
                in text)
        assert "- Session began:" not in text
        assert "- Active model: blackwell/qwen38-27b-fp8" in text
        assert "- Context window: 200,000 tokens" in text
        assert "- Vision: yes" in text

    def test_epoch_fire_once_then_suppressed(self, tmp_path):
        eng = ReminderEngine(_cfg(tmp_path))
        st = _orient_state(handoff_epoch_index=12, handoff_epoch_ts=EPOCH_TS)
        first = eng.evaluate(st)
        assert any(r.trigger == ORIENT_ID for r in first)
        second = eng.evaluate(st)
        assert not any(r.trigger == ORIENT_ID for r in second)

    def test_model_switch_refires_with_epoch_label(self, tmp_path):
        eng = ReminderEngine(_cfg(tmp_path))
        st = _orient_state(handoff_epoch_index=12, handoff_epoch_ts=EPOCH_TS)
        eng.evaluate(st)
        st2 = _orient_state(model_resolved="anthropic/claude-x",
                            handoff_epoch_index=12, handoff_epoch_ts=EPOCH_TS)
        refired = [r for r in eng.evaluate(st2) if r.trigger == ORIENT_ID]
        assert len(refired) == 1
        assert "- Epoch began:" in refired[0].text

    def test_empty_epoch_ts_falls_back_to_session_label(self, tmp_path):
        """Audit L1: index>0 with an empty ts must NOT render a blank
        '- Epoch began:  (...)' row — the engine defends even if the agent
        guard is bypassed."""
        eng = ReminderEngine(_cfg(tmp_path))
        st = _orient_state(handoff_epoch_index=12, handoff_epoch_ts="")
        orient = [r for r in eng.evaluate(st) if r.trigger == ORIENT_ID]
        assert len(orient) == 1
        assert "- Session began:" in orient[0].text
        assert "- Epoch began:" not in orient[0].text

    def test_suppression_forms_preserved(self, tmp_path):
        eng = ReminderEngine(_cfg(tmp_path))
        silent = eng.evaluate(_orient_state(model_resolved=""))
        assert not any(r.trigger == ORIENT_ID for r in silent)

    def test_checkpoint_reminder_vocabulary_no_inject(self, tmp_path):
        """Vocabulary sweep covers ALL reminder texts, not just S7: the
        handoff-checkpoint directive must not contain 'inject' either."""
        eng = ReminderEngine(_cfg(tmp_path))
        st = _orient_state(checkpoint_threshold=500)  # 1000 >= 500 → fires
        ck = [r for r in eng.evaluate(st) if r.trigger == "handoff-checkpoint"]
        assert ck, "checkpoint reminder must fire above threshold"
        for r in ck:
            assert "inject" not in r.text.lower()


# ---------------------------------------------------------------------------
# D. Agent epoch wiring — _note_handoff_boundary_applied → _orient_inputs
# ---------------------------------------------------------------------------

class TestAgentEpochWiring:

    def _agent(self, tmp_path):
        from openalph.agent import Agent
        return Agent(_cfg(tmp_path))

    def _outcome(self, ts="2026-09-08T11:08:11Z", idx=247):
        return {"applied": True,
                "manifest": {"ts": ts, "boundary_index": idx,
                             "runway": {"tokens_after": 7495,
                                        "available": 229376}}}

    def test_note_boundary_sets_pending_epoch(self, tmp_path):
        agent = self._agent(tmp_path)
        agent._note_handoff_boundary_applied(ROOM, self._outcome())
        inputs = agent._orient_inputs(ROOM)
        assert inputs["handoff_epoch_index"] == 247
        expect = (datetime(2026, 9, 8, 11, 8, 11, tzinfo=timezone.utc)
                  .astimezone().strftime("%A, %B %d, %Y — %H:%M %Z"))
        assert inputs["handoff_epoch_ts"] == expect

    def test_epoch_record_persists_for_epoch_lifetime(self, tmp_path):
        """Audit M1: pop-on-read was a fixture error — the epoch record must
        PERSIST so a mid-epoch model-switch refire keeps the Epoch label.
        It is overwritten by the next boundary and cleared by reset_room."""
        agent = self._agent(tmp_path)
        agent._note_handoff_boundary_applied(ROOM, self._outcome())
        first = agent._orient_inputs(ROOM)
        assert first["handoff_epoch_index"] == 247
        second = agent._orient_inputs(ROOM)
        assert second["handoff_epoch_index"] == first["handoff_epoch_index"]
        assert second["handoff_epoch_ts"] == first["handoff_epoch_ts"]
        # next boundary overwrites
        agent._note_handoff_boundary_applied(
            ROOM, self._outcome(ts="2026-09-08T12:00:00Z", idx=300))
        third = agent._orient_inputs(ROOM)
        assert third["handoff_epoch_index"] == 300

    def test_bool_boundary_index_rejected(self, tmp_path):
        """Audit L2: bool is an int subclass — a malformed bool index must
        not render as 'boundary True'."""
        agent = self._agent(tmp_path)
        agent._note_handoff_boundary_applied(
            ROOM, {"applied": True,
                   "manifest": {"ts": "2026-09-08T11:08:11Z",
                                "boundary_index": True,
                                "runway": {"tokens_after": 1,
                                           "available": 2}}})
        inputs = agent._orient_inputs(ROOM)
        assert inputs["handoff_epoch_index"] == 0

    def test_unrenderable_ts_drops_epoch_pair(self, tmp_path):
        """Audit L1/L3: never a blank '- Epoch began:  (...)' row."""
        agent = self._agent(tmp_path)
        agent._note_handoff_boundary_applied(
            ROOM, {"applied": True,
                   "manifest": {"ts": "not-a-timestamp",
                                "boundary_index": 247,
                                "runway": {"tokens_after": 1,
                                           "available": 2}}})
        inputs = agent._orient_inputs(ROOM)
        assert inputs["handoff_epoch_index"] == 0
        assert inputs["handoff_epoch_ts"] == ""

    def test_inputs_without_boundary_default_fresh(self, tmp_path):
        agent = self._agent(tmp_path)
        inputs = agent._orient_inputs(ROOM)
        assert inputs["handoff_epoch_index"] == 0
        assert inputs["handoff_epoch_ts"] == ""

    def test_reset_room_clears_pending(self, tmp_path):
        agent = self._agent(tmp_path)
        agent._note_handoff_boundary_applied(ROOM, self._outcome())
        agent.reset_room(ROOM)
        inputs = agent._orient_inputs(ROOM)
        assert inputs["handoff_epoch_index"] == 0

    def test_end_to_end_epoch_label(self, tmp_path):
        """Boundary applied (note seam) → next populated turn-start evaluate
        carries the Epoch-began label with index + boundary ts."""
        agent = self._agent(tmp_path)
        agent._note_handoff_boundary_applied(ROOM, self._outcome())
        inputs = agent._orient_inputs(ROOM)
        st = _orient_state(**{k: v for k, v in inputs.items()
                              if k.startswith("handoff_epoch")})
        eng = ReminderEngine(_cfg(tmp_path))
        orient = [r for r in eng.evaluate(st) if r.trigger == ORIENT_ID]
        assert len(orient) == 1
        assert "(context handoff boundary 247)" in orient[0].text
        assert "- Epoch began:" in orient[0].text


# ---------------------------------------------------------------------------
# E. Manifest anchored_estimate
# ---------------------------------------------------------------------------

class TestManifestAnchoredEstimate:

    def _apply(self, tmp_path, anchored):
        m = _handoff()
        from pathlib import Path
        from _handoff_helpers import append_all, assistant, make_log, user
        log = make_log(tmp_path)
        append_all(log, [user("q"), assistant("a")])
        return m.apply_boundary(
            log, ROOM, workspace=Path(tmp_path), trigger="auto",
            exclude_inflight=False, config_paths=None, window=1000,
            budget_pct=0.25, budget_min=96000, max_tokens=100,
            anchored_estimate=anchored)

    def test_manifest_includes_estimate_when_provided(self, tmp_path):
        res = self._apply(tmp_path, 195000)
        assert res["manifest"]["anchored_estimate"] == 195000

    def test_manifest_estimate_none_when_absent(self, tmp_path):
        res = self._apply(tmp_path, None)
        assert res["manifest"]["anchored_estimate"] is None


# ---------------------------------------------------------------------------
# F. Sub-agent boundary body — directive sentence
# ---------------------------------------------------------------------------

class TestSubBodyDirective:

    def test_sub_body_content(self):
        m = _handoff()
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        out = m.apply_handoff_to_messages(
            messages, boundary_index=2, task_text="ORIGINAL_TASK_TEXT",
            trigger="auto")
        joined = "\n".join(str(x.get("content", ""))
                           for x in out["messages"])
        assert "[Handoff boundary — " in joined      # prefix retained
        assert "ORIGINAL_TASK_TEXT" in joined        # task survives verbatim
        assert "resuming" in joined
        assert "context handoff" in joined
        assert "Do not claim continuity" in joined
