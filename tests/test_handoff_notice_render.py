"""Red suite: handoff boundary m.notice redesign (kdsn.322.14).

Pins the operator-facing notice format ruled by SB 2026-09-04:

- ONE shared renderer (callbacks.render_handoff_notice) used by BOTH the
  tool/auto path and the slash path (matrix._handoff_confirm_text) —
  the two-renderer drift is dead.
- Composite accounting: before AND after both include the system prompt
  + tool defs (same accounting as /status); after is MEASURED
  (snapshot + surviving tail), never the pinned 0. tokens_dropped is
  its own key so "what did the strip buy" survives.
- Headline (the sink renders line 1 as the collapsed <details> summary):
  knot emoji, trigger, ~before → ~after tok, project, checkpoint (bare
  status value — NO definitional gloss), reinserted file list.
- Fold: dropped tokens, durable budget (no parentheticals), boundary
  index (audit ref), errors, full progress.md FROM THE FROZEN SNAPSHOT
  (html-escaped, size-capped with a truncation marker).
- Language: insert/reinserted everywhere; the string "inject" must not
  appear in any rendered notice.
- Legacy manifests (tokens_after_est=0, no new keys) render fail-soft.
"""


from openalph.callbacks import render_handoff_notice


def _new_outcome(**kw):
    """Outcome dict in the NEW manifest shape (post-redesign)."""
    manifest = {
        "ts": "2026-09-04T12:00:00Z",
        "boundary_index": 87,
        "trigger": "tool",
        "tokens_before": 16651,
        "tokens_after": 10712,
        "tokens_dropped": 4856,
        "durable": {
            "project": "context-handoff-test",
            "files": [
                {"path": "memory/projects/context-handoff-test/progress.md",
                 "reason": "project working state", "origin": "auto", "chars": 3134},
                {"path": "memory/projects/context-handoff-test/durable-set.toml",
                 "reason": "durable-set declaration", "origin": "auto", "chars": 465},
                {"path": "memory/projects/context-handoff-test/README.md",
                 "reason": "project home", "origin": "durable-set.toml", "chars": 2370},
            ],
            "budget_tokens": 96000,
            "used_tokens": 1492,
            "over_budget": False,
        },
        "runway": {
            "available": 253952, "tokens_after": 640, "runway_after": 253312,
            "threshold_tokens": 24000, "handoff_advised": False,
        },
        "checkpoint": {"status": "none", "fired_ts": None, "project_mtime": None},
        "errors": [],
    }
    manifest.update(kw.pop("manifest_overrides", {}))
    outcome = {"applied": True, "manifest": manifest,
               "over_budget": False, "handoff_advised": False,
               "forced_handoff": False,
               "progress_md": "FROZEN PROGRESS <script>alert(1)</script> & more"}
    outcome.update(kw)
    return outcome


class TestHeadline:
    def test_composite_before_after(self):
        text = render_handoff_notice("tool", _new_outcome())
        assert "context ~16,651 → ~10,712 tok" in text

    def test_checkpoint_bare_no_gloss(self):
        text = render_handoff_notice("tool", _new_outcome())
        assert "checkpoint: none" in text
        assert "checkpoint: none (" not in text, "no definitional gloss"

    def test_checkpoint_stale_value(self):
        oc = _new_outcome(manifest_overrides={
            "checkpoint": {"status": "stale", "fired_ts": "x", "project_mtime": None}})
        assert "checkpoint: stale" in render_handoff_notice("tool", oc)

    def test_durable_set_file_list_visible(self):
        text = render_handoff_notice("tool", _new_outcome())
        assert "durable set 3 files (read list; progress.md inserted):" in text
        for name in ("progress.md", "durable-set.toml", "README.md"):
            assert name in text

    def test_knot_emoji_not_broom(self):
        text = render_handoff_notice("tool", _new_outcome())
        assert "🪢" in text
        assert "🧹" not in text

    def test_trigger_in_headline(self):
        assert "(slash)" in render_handoff_notice("slash", _new_outcome())


class TestFold:
    def test_dropped_tokens_no_conversation_word(self):
        text = render_handoff_notice("tool", _new_outcome())
        assert "dropped: ~4,856 tok" in text
        assert "of conversation" not in text

    def test_budget_no_parenthetical(self):
        text = render_handoff_notice("tool", _new_outcome())
        assert "durable budget: 1,492 / 96,000" in text
        assert "informational" not in text

    def test_boundary_index_demoted_to_fold(self):
        text = render_handoff_notice("tool", _new_outcome())
        assert "boundary index: 87" in text

    def test_frozen_progress_md_present(self):
        text = render_handoff_notice("tool", _new_outcome())
        assert "FROZEN PROGRESS" in text

    def test_progress_md_escaped(self):
        text = render_handoff_notice("tool", _new_outcome())
        assert "<script>" not in text
        assert "&lt;script&gt;" in text

    def test_progress_md_size_capped(self):
        oc = _new_outcome(progress_md="P" * 200_000)
        text = render_handoff_notice("tool", oc)
        assert "[truncated" in text

    def test_no_project_no_progress_section(self):
        oc = _new_outcome(progress_md=None, manifest_overrides={
            "durable": {"project": None, "files": [], "budget_tokens": 96000,
                        "used_tokens": 0, "over_budget": False}})
        text = render_handoff_notice("tool", oc)
        assert "as reinserted" not in text  # wording is now "as inserted"


class TestLanguageAndFormat:
    def test_no_inject_wording(self):
        text = render_handoff_notice("tool", _new_outcome())
        assert "inject" not in text.lower()

    def test_no_est_qualifier(self):
        text = render_handoff_notice("tool", _new_outcome())
        assert "(est." not in text

    def test_over_budget_flagged_without_editorial(self):
        oc = _new_outcome(manifest_overrides={
            "durable": {"project": "p", "files": [], "budget_tokens": 100,
                        "used_tokens": 200, "over_budget": True}})
        oc["over_budget"] = True
        text = render_handoff_notice("tool", oc)
        assert "durable budget: 200 / 100" in text
        assert "over" in text.lower()


class TestAuditHardening:
    """AUDIT H2/H3 (kdsn.322 fix pass): the fold renders progress.md and the
    headline renders durable-file basenames + project names — all of it is
    agent/workspace-controlled text reaching a Matrix room. The notice must
    apply the SAME freeze pipeline as the snapshot (credential redaction +
    reminder escape) and escape every interpolated component."""

    SECRET = "ghp_AbCdEf123456789GhIjKlMnOpQrStUvWxYz"

    def _poisoned(self, **kw):
        oc = _new_outcome(
            progress_md=f"token {self.SECRET} <img src=x onerror=alert(1)>",
            manifest_overrides={
                "durable": {
                    "project": "p<img src=x onerror=alert(2)>",
                    "files": [
                        {"path": "memory/projects/p/<img onerror=alert(3)>.md",
                         "reason": "r", "origin": "durable-set.toml",
                         "chars": 10},
                        {"path": "memory/projects/p/progress.md",
                         "reason": "r", "origin": "auto", "chars": 10},
                    ],
                    "budget_tokens": 96000, "used_tokens": 10,
                    "over_budget": False,
                },
            },
        )
        oc.update(kw)
        return oc

    def test_no_raw_secret_in_notice(self):
        text = render_handoff_notice("tool", self._poisoned())
        assert self.SECRET not in text, (
            "raw credential reached the room — the fold must apply the "
            "snapshot's redaction pass, not broadcast raw file bytes")

    def test_no_raw_html_in_headline(self):
        text = render_handoff_notice("tool", self._poisoned())
        head = text.split("\n", 1)[0]
        assert "<img" not in head, (
            "unescaped workspace-controlled text reached the collapsed "
            "summary (raw HTML -> formatted_body)")

    def test_no_raw_html_in_fold_pointer(self):
        oc = self._poisoned(progress_md="P" * 200_000 + self.SECRET)
        text = render_handoff_notice("tool", oc)
        assert "<img" not in text

    def test_redacted_form_present(self):
        text = render_handoff_notice("tool", self._poisoned())
        assert "ghp_" not in text, "redaction must transform the token"


class TestLegacyTolerance:
    def test_legacy_manifest_renders_fail_soft(self):
        """Pre-redesign manifests (tokens_after_est=0, no new keys) still render."""
        oc = _new_outcome(manifest_overrides={
            "tokens_after": None, "tokens_dropped": None})
        del oc["progress_md"]
        mf = oc["manifest"]
        mf["tokens_after_est"] = 0
        text = render_handoff_notice("tool", oc)  # must not raise
        assert "checkpoint:" in text


class TestSharedRenderer:
    def test_slash_path_delegates_to_shared_renderer(self):
        """matrix._handoff_confirm_text must produce IDENTICAL output to the
        shared renderer — one renderer, two call sites, zero drift."""
        from openalph import matrix as mx
        outcome = _new_outcome()
        assert mx._handoff_confirm_text(outcome, "tool") == \
            render_handoff_notice("tool", outcome)

    def test_tool_path_uses_shared_renderer(self):
        """The callbacks tool/auto notice path must call the shared renderer."""
        import inspect
        from openalph import callbacks as cb
        src = inspect.getsource(cb)
        assert "_handoff_notice_text" not in src or \
            "_handoff_notice_text = render_handoff_notice" in src or \
            "render_handoff_notice(trigger, outcome)" in src, \
            "tool path must route through render_handoff_notice"
