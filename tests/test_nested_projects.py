"""kdsn.331: nested project directories — per-initiative durable sets.

RED SUITE — orchestrator-authored (tdd-orchestration route). Real-path
discipline per tool-management "the one lesson": the declare pins drive the
REAL _gc_set_project_cb through the REAL _build_agent_callbacks with a REAL
Agent + REAL SessionLog (only nio is mocked); the CLI pins run the REAL
cmd_exec with only the provider seam faked.

Spec: memory/projects/openalph/specs/kdsn.331-nested-projects-spec.md
Bead: workspace-kdsn.331. SB rulings 2026-09-14:
  - exactly ONE nesting level (a/b) — bare segments, no traversal, no dots
  - DESCENT allowed: foo -> foo/bar re-declare mid-epoch (tool path);
    ascent/sibling/unrelated stay refused; operator /project keeps blanket
    override authority
  - no scaffold-on-declare; parent durable set intact

Sections:
  A. Name grammar — project_valid_name (spec D1)
  B. Descent rule — real-path declare callback (spec D2)
  C. Nested boundary resolution — resolve/checkpoint/frame (spec D4/D5;
     GREEN pre-fix by design: resolution joins the name unchanged, the
     validator was the only blocker — these are regression guards)
  D. CLI exec --project (spec D7)
  E. Matrix /project command (spec D3)
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from openalph.handoff import (
    checkpoint_status,
    frame_snapshot,
    project_valid_name,
    read_active_project,
    resolve_durable_set,
    TRIGGER_CHECKPOINT,
)

ROOM = "!nested:test-local"
AGENT_ID = "@merry:test-local"


# ---------------------------------------------------------------------------
# Shared real-path harness (real Agent + real SessionLog, mocked nio only)
# ---------------------------------------------------------------------------

def _cfg(workspace, **kw):
    from openalph.config import AgentConfig, ContextHandoffConfig, ProviderConfig
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[])},
        workspace=workspace,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        context=ContextHandoffConfig(),
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _make_bot(tmp_path, projects=("foo", "foo/bar", "foo/baz", "qux")):
    """Real MatrixBot shell (gc-integration pattern) + REAL SessionLog.

    Creates memory/projects/<p>/ for each name in `projects` so the
    dir-exists checks see them; callers create extra dirs as needed."""
    from openalph.matrix import MatrixBot
    from openalph.agent import Agent
    from openalph.config import MatrixConfig
    from openalph.session import SessionLog

    ws = Path(tmp_path)
    (ws / "skills").mkdir(parents=True, exist_ok=True)
    for p in projects:
        (ws / "memory" / "projects" / p).mkdir(parents=True, exist_ok=True)
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
    bot._room_quiet = {}
    bot._halted_rooms = set()
    bot._background_tasks = set()
    bot._session_locks = {}
    bot._advisor_results = {}
    bot._subagent_results = {}
    bot.session_log = SessionLog(ws, AGENT_ID)
    bot.heartbeat = None
    bot.umbral = None
    bot._degraded_provider_notice = {}
    return bot, agent


def _declare(bot, project):
    """Tool-path declare: the REAL _gc_set_project_cb via the real seam."""
    cb = bot._build_agent_callbacks(ROOM, None)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(cb["set_active_project"](project))
    finally:
        loop.close()


def _project_events(bot):
    return [e for e in bot.session_log.read(ROOM)
            if e.get("role") == "system"
            and e.get("event") == "active_project"]


class TestDescentNotice:
    def test_E6_descent_emits_room_visible_notice(self, tmp_path):
        """kdsn.331 audit R1 (kimi+glm+qwen converged): a descent moves the
        room's durable-set SOURCE — operators get a passive room-visible
        signal (boundary-notice fail-soft pattern), not just tool-result
        text buried in a collapsed detail."""
        bot, _ = _make_bot(tmp_path)
        assert _declare(bot, "foo")["ok"]
        bot.client.room_send.reset_mock()
        res = _declare(bot, "foo/bar")
        assert res["ok"], res
        sent = _send_texts(bot)
        assert "foo" in sent and "foo/bar" in sent, (
            "descent must emit a room notice naming old and new project")
        # flat first-declare stays quiet (parity with pre-existing behavior)
        bot.client.room_send.reset_mock()
        bot2, _ = _make_bot(tmp_path / "ws2")
        assert _declare(bot2, "foo")["ok"]
        assert _send_texts(bot2) == "", "first declare is not a descent"


# ===========================================================================
# A. Name grammar (spec D1) — segment-based, one nesting level, parity
# ===========================================================================

@pytest.mark.parametrize("name", [
    "proj",            # flat (unchanged)
    "proj/init",       # one nesting level — THE new legal form
    "Foo/bar",         # case parity (no new charset rules)
    "proj init",       # interior whitespace parity (single segment)
])
def test_A_name_accepted(name):
    assert project_valid_name(name), f"{name!r} must be a valid project name"


@pytest.mark.parametrize("name", [
    "",                # empty
    ".", "..",         # dot shorthands
    ".hidden",         # dot-prefix segment
    "proj/.bar",       # dot-prefix in nested segment
    "proj/..",         # traversal
    "proj/.",          # dot segment
    "proj/./sub",      # dot segment mid-path
    "a/b/c",           # depth > 2
    "/proj",           # leading slash (empty segment)
    "proj/",           # trailing slash (empty segment)
    "proj//sub",       # empty mid segment
    "proj\\bar",       # backslash separator
    "foo/bar/",        # trailing slash, nested
    # kdsn.331 audit R3 (glm+qwen converged): control chars rejected at the
    # validator — they render unescaped into notice/echo copy otherwise
    "proj\ninit",      # newline in segment
    "proj\tinit",      # tab in segment
    "foo/bar\nbaz",    # newline in nested segment
    "proj\x7f",        # DEL
])
def test_A_name_rejected(name):
    assert not project_valid_name(name), f"{name!r} must be rejected"


# ===========================================================================
# B. Descent rule (spec D2) — real-path declare callback
# ===========================================================================

class TestDescent:
    def test_B1_nested_first_declare_allowed(self, tmp_path):
        bot, _ = _make_bot(tmp_path)
        res = _declare(bot, "foo/bar")
        assert res["ok"], res
        assert read_active_project(bot.session_log.read(ROOM)) == "foo/bar"
        assert len(_project_events(bot)) == 1

    def test_B2_descent_allowed_and_announced(self, tmp_path):
        bot, _ = _make_bot(tmp_path)
        assert _declare(bot, "foo")["ok"]
        res = _declare(bot, "foo/bar")
        assert res["ok"], f"descent foo -> foo/bar must be allowed: {res}"
        assert "descend" in res["text"].lower(), (
            "descent must be announced for visibility")
        events = _project_events(bot)
        assert len(events) == 2
        assert read_active_project(bot.session_log.read(ROOM)) == "foo/bar"

    def test_B3_ascent_refused(self, tmp_path):
        bot, _ = _make_bot(tmp_path)
        assert _declare(bot, "foo")["ok"]
        assert _declare(bot, "foo/bar")["ok"]
        res = _declare(bot, "foo")
        assert not res["ok"], "ascent a/b -> a must be refused"
        assert "one project per room" in res["text"]
        assert read_active_project(bot.session_log.read(ROOM)) == "foo/bar"

    def test_B4_sibling_refused(self, tmp_path):
        bot, _ = _make_bot(tmp_path)
        assert _declare(bot, "foo")["ok"]
        assert _declare(bot, "foo/bar")["ok"]
        res = _declare(bot, "foo/baz")
        assert not res["ok"], "sibling a/b -> a/c must be refused"
        assert "one project per room" in res["text"]

    def test_B5_unrelated_refused(self, tmp_path):
        bot, _ = _make_bot(tmp_path)
        assert _declare(bot, "foo")["ok"]
        res = _declare(bot, "qux")
        assert not res["ok"], "unrelated re-declare must stay refused"
        assert "one project per room" in res["text"]

    def test_B6_same_project_noop(self, tmp_path):
        bot, _ = _make_bot(tmp_path)
        assert _declare(bot, "foo")["ok"]
        res = _declare(bot, "foo")
        assert res["ok"] and "no-op" in res["text"]
        assert len(_project_events(bot)) == 1

    def test_B7_invalid_descent_target_gets_invalid_refusal(self, tmp_path):
        """Validity FIRST (spec D2): foo -> foo/.. must be refused as an
        INVALID NAME, not fall through to the one-project-per-room branch."""
        bot, _ = _make_bot(tmp_path)
        assert _declare(bot, "foo")["ok"]
        res = _declare(bot, "foo/..")
        assert not res["ok"]
        assert "Invalid project name" in res["text"], res
        assert "one project per room" not in res["text"]

    def test_B8_descent_missing_dir_refused_pre_append(self, tmp_path):
        bot, _ = _make_bot(tmp_path)
        assert _declare(bot, "foo")["ok"]
        res = _declare(bot, "foo/missing")
        assert not res["ok"]
        assert "does not exist" in res["text"], res
        assert len(_project_events(bot)) == 1, "refusal must not append"

    def test_B9_descent_echo_lists_nested_durable_entries(self, tmp_path):
        ws = Path(tmp_path)
        (ws / "docs").mkdir(exist_ok=True)
        (ws / "docs" / "extra.md").write_text("extra context\n")
        # fixture precondition: the nested dir must exist before the toml
        # write (_make_bot creates it later) — orchestrator fixture fix,
        # assertion untouched
        (ws / "memory" / "projects" / "foo" / "bar").mkdir(parents=True,
                                                           exist_ok=True)
        toml = (ws / "memory" / "projects" / "foo" / "bar"
                / "durable-set.toml")
        toml.write_text(
            '[[entries]]\npath = "docs/extra.md"\nreason = "initiative doc"\n')
        bot, _ = _make_bot(tmp_path)
        assert _declare(bot, "foo")["ok"]
        res = _declare(bot, "foo/bar")
        assert res["ok"], res
        assert "docs/extra.md" in res["text"], (
            "echo must list the nested project's durable entries")

    def test_B10_three_segments_refused_as_invalid(self, tmp_path):
        bot, _ = _make_bot(tmp_path)
        res = _declare(bot, "a/b/c")
        assert not res["ok"]
        assert "Invalid project name" in res["text"], res


# ===========================================================================
# C. Nested boundary resolution (spec D4/D5) — regression guards
# ===========================================================================

class TestNestedResolution:
    def _ws(self, tmp_path):
        ws = Path(tmp_path)
        proj = ws / "memory" / "projects" / "foo" / "bar"
        proj.mkdir(parents=True)
        (proj / "progress.md").write_text(
            "# State\nnested-checkpoint-marker\n")
        (proj / "durable-set.toml").write_text(
            '[[entries]]\npath = "memory/projects/foo/README.md"\n'
            'reason = "parent docs ride the initiative set"\n')
        (ws / "memory" / "projects" / "foo" / "README.md").write_text(
            "parent readme\n")
        return ws

    def test_C1_resolve_carries_nested_artifacts(self, tmp_path):
        ws = self._ws(tmp_path)
        res = resolve_durable_set(ws, "foo/bar")
        assert res["project"] == "foo/bar"
        assert res["errors"] == []
        paths = [f["path"] for f in res["files"]]
        assert "memory/projects/foo/bar/progress.md" in paths
        assert "memory/projects/foo/bar/durable-set.toml" in paths
        assert "memory/projects/foo/README.md" in paths, (
            "curated entry under the PARENT dir must resolve (workspace-wide)")

    def test_C2_containment_unchanged_for_nested(self, tmp_path):
        ws = self._ws(tmp_path)
        (ws / "memory" / "projects" / "foo" / "bar" / "durable-set.toml"
         ).write_text(
            '[[entries]]\npath = "../../../outside.txt"\nreason = "escape"\n')
        (Path(tmp_path).parent / "outside.txt").write_text("secret\n")
        res = resolve_durable_set(ws, "foo/bar")
        assert any("escapes" in e for e in res["errors"]), res["errors"]

    def test_C3_checkpoint_freshness_points_at_nested_dir(self, tmp_path):
        """Canonical mtimes pinned via os.utime against the PARSED fired ts
        (strip-file TestCheckpointPredicate pattern — no sleep, no tz games)."""
        import os
        from openalph.handoff import _parse_ts_utc
        from openalph.session import SessionLog
        ws = self._ws(tmp_path)
        log = SessionLog(ws, AGENT_ID)
        log.append(room=ROOM, sender=AGENT_ID, role="user", content="q")
        log.append(room=ROOM, sender=AGENT_ID, role="system",
                   event="active_project", detail="foo/bar")
        log.append(room=ROOM, sender=AGENT_ID, role="user", content="w")
        log.append(room=ROOM, sender=AGENT_ID, role="user", content="cp",
                   source="reminder", trigger=TRIGGER_CHECKPOINT)
        entries = log.read(ROOM)
        rem = [e for e in entries
               if e.get("trigger") == TRIGGER_CHECKPOINT][-1]
        fired = _parse_ts_utc(rem["ts"])
        proj = ws / "memory" / "projects" / "foo" / "bar"
        # BOTH files pinned (predicate takes the max mtime; the toml write
        # races the second-truncated ts otherwise — strip-file pattern)
        os.utime(proj / "progress.md", (fired - 100, fired - 100))
        os.utime(proj / "durable-set.toml", (fired - 100, fired - 100))
        assert checkpoint_status(entries, ws, "foo/bar")["status"] == "stale"
        os.utime(proj / "progress.md", (fired + 100, fired + 100))
        st = checkpoint_status(entries, ws, "foo/bar")
        assert st["status"] == "fresh" and st["project_mtime"] is not None

    def test_C4_snapshot_inlines_nested_progress(self, tmp_path):
        ws = self._ws(tmp_path)
        res = resolve_durable_set(ws, "foo/bar")
        snap = frame_snapshot(1, res, False, 100000)
        assert "nested-checkpoint-marker" in snap, (
            "the nested project's progress.md is the ONE inlined artifact")


# ===========================================================================
# D. CLI exec --project (spec D7)
# ===========================================================================

from test_context_handoff_exec import (  # noqa: E402  (sibling harness reuse)
    make_config, read_log, run_exec_real, _StreamRecorder, _task_file)


class TestCliNestedProject:
    def test_D1_nested_project_happy_path(self, tmp_path):
        (Path(tmp_path) / "memory" / "projects" / "proj" / "init"
         ).mkdir(parents=True)
        config = make_config(tmp_path)
        rec = _StreamRecorder(final_text="done")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, "hello"),
             "--project", "proj/init"],
            config=config, recorder=rec)
        assert code == 0, (out, err)
        events = [e for e in read_log(config)
                  if e.get("event") == "active_project"]
        assert len(events) == 1 and events[0].get("detail") == "proj/init"

    def test_D2_traversal_project_exits_before_inference(self, tmp_path):
        (Path(tmp_path) / "memory" / "projects" / "proj").mkdir(parents=True)
        config = make_config(tmp_path)
        rec = _StreamRecorder(final_text="done")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, "hello"),
             "--project", "proj/.."],
            config=config, recorder=rec)
        assert code == 1
        assert rec.calls == [], "no model call may happen"
        assert out.strip() == "", "no stdout on validation failure"
        assert "invalid" in err.lower()

    def test_D3_missing_nested_dir_fails_loud(self, tmp_path):
        (Path(tmp_path) / "memory" / "projects" / "proj").mkdir(parents=True)
        config = make_config(tmp_path)
        rec = _StreamRecorder(final_text="done")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, "hello"),
             "--project", "proj/missing"],
            config=config, recorder=rec)
        assert code == 1
        assert rec.calls == [], "no model call may happen"
        assert "does not exist" in err, (
            "valid nested name + missing dir must report the DIR, "
            "not the name grammar")


# ===========================================================================
# E. Matrix /project command (spec D3 — operator authority preserved)
# ===========================================================================

def _event(body):
    ev = MagicMock()
    ev.body = body
    ev.sender = "@sb:matrix.local"
    ev.event_id = "$ev1"
    ev.source = {"content": {"m.relates_to": None}}
    return ev


def _send_texts(bot):
    out = []
    for c in bot.client.room_send.call_args_list:
        content = c.kwargs.get("content") or (c.args[2] if len(c.args) > 2
                                              else {})
        body = content.get("body", "") if isinstance(content, dict) else ""
        out.append(body)
    return "\n".join(out)


def _slash(bot, body):
    room = MagicMock()
    room.room_id = ROOM
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(bot._handle_room_message(room, _event(body)))
    finally:
        loop.close()


class TestProjectSlashCommand:
    def test_E1_nested_set_accepted(self, tmp_path):
        bot, _ = _make_bot(tmp_path)
        _slash(bot, "/project set foo/bar")
        sent = _send_texts(bot)
        assert "Project set to **foo/bar**" in sent, sent
        assert read_active_project(bot.session_log.read(ROOM)) == "foo/bar"

    def test_E2_traversal_set_refused(self, tmp_path):
        bot, _ = _make_bot(tmp_path)
        _slash(bot, "/project set foo/..")
        assert "Invalid project name" in _send_texts(bot)
        assert read_active_project(bot.session_log.read(ROOM)) is None

    def test_E3_operator_override_flat_to_nested_announced(self, tmp_path):
        bot, _ = _make_bot(tmp_path)
        _slash(bot, "/project set foo")
        _slash(bot, "/project set foo/bar")
        sent = _send_texts(bot)
        assert "Operator override" in sent, (
            "operator authority: flat -> nested re-declare succeeds, announced")
        assert read_active_project(bot.session_log.read(ROOM)) == "foo/bar"

    def test_E4_status_shows_latest_nested_name(self, tmp_path):
        bot, _ = _make_bot(tmp_path)
        _slash(bot, "/project set foo")
        _slash(bot, "/project set foo/bar")
        bot.client.room_send.reset_mock()
        _slash(bot, "/project")
        assert "foo/bar" in _send_texts(bot)

    def test_E5_operator_set_missing_dir_refused_pre_append(self, tmp_path):
        """kdsn.331 audit R2 (glm MEDIUM): the operator path must refuse
        BEFORE appending — a typo'd canonical-looking nested name must not
        persist as the room's project with an empty durable package."""
        bot, _ = _make_bot(tmp_path, projects=("foo",))  # foo/bar NOT created
        _slash(bot, "/project set foo/bar")
        sent = _send_texts(bot)
        assert "does not exist" in sent, sent
        assert read_active_project(bot.session_log.read(ROOM)) is None, (
            "refusal must not append the active_project event")
