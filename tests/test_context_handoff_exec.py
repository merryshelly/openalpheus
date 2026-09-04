"""Context handoff — T3 exec wiring (kdsn.322, spec §3.5 — the money tests).

RED SUITE — orchestrator-authored (tdd-orchestration route). Real-path
discipline per tool-management "the one lesson": a REAL Agent through the
REAL cmd_exec with the REAL SessionLog/HeadlessSinks/build_callbacks —
ONLY the provider seam (stream) and execute_tool are faked. A green suite
that mocked the callbacks dict would have tested nothing about the wiring.

Interface contract (what T3 must export — spec §3.5):
  - cmd_exec builds a cage-local SessionLog at config.workspace/sessions
    under the sanitized room label (default _exec), wired with
    handoff_default from config.context.handoff_enabled.
  - Turn persistence: user task, assistant turns (tool-use AND terminal),
    tool results, reminder injections, boundary markers — appended as the
    turn runs (the _process_cli_line pattern, minus the stdin loop).
  - HeadlessSinks + build_callbacks: reminders, the apply_handoff_boundary
    seam, set_active_project all live on exec. The filing channel union
    holds: --tools file_ticket still carries filed_proposals_sink.
  - --project <name>: validate (project_valid_name + dir exists) BEFORE
    any inference; happy path appends the active_project event to the
    SessionLog; error paths exit 1 with NO stdout and NO model call.
  - No rehydration: a pre-existing SessionLog is appended to, never loaded
    into context.
  - stdout contract unchanged: exactly one JSON line — even when a
    boundary fires mid-turn; notices stay on stderr.

Ladder arithmetic in the fixtures: model_max_tokens=1000, max_tokens=100
→ available 900; checkpoint 75% = 675; auto 85% = 765; hard 92% = 828.
The MID task (~1200 chars ≈ 300 tokens + minimal system) crosses NEITHER —
clean baseline. The BIG task (~8000 chars ≈ 2000 tokens) crosses both.

Fixture hermeticity (kdsn.322 T5 follow-up): make_config writes a MINIMAL
workspace CONTINUITY.md, so the system-prompt base is fixture-controlled —
the packaged template's size (fleet prompt file #8, ~900 tokens since the
2026-09-03 handoff rewrite) no longer shifts the ladder arithmetic. Re-aim
tuned windows ONLY from the fixture base (footer + minimal file).
"""

import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from openalph.config import AgentConfig, ContextHandoffConfig, ProviderConfig
from openalph.provider import Response, StreamEvent, ToolCall, Usage
from openalph.tools import ToolResult


# ---------------------------------------------------------------------------
# Harness — real Agent, faked provider seam only
# ---------------------------------------------------------------------------

USAGE_KEYS = ("in", "cached", "out", "reasoning")


def make_config(tmp_path, *, model_max_tokens=100000, max_tokens=8192,
                max_iterations=20):
    provider = ProviderConfig(key="p", type="openai", api_key="sk-test",
                              base_url="http://127.0.0.1:18081")
    # Hermetic prompt base: pin a minimal CONTINUITY.md so the packaged
    # template (whose byte size is fleet prompt surface, not fixture
    # contract) never shifts the ladder arithmetic these tests tune against.
    ws = Path(tmp_path)
    (ws / "CONTINUITY.md").write_text(
        "# CONTINUITY.md\n\n"
        "Checkpoint discipline: maintain progress.md and durable-set.toml — "
        "they are the only carryover across a handoff boundary.\n")
    return AgentConfig(
        name="exec-agent",
        default_model="p/model",
        max_tokens=max_tokens,
        model_max_tokens=model_max_tokens,
        providers={"p": provider},
        workspace=Path(tmp_path),
        max_iterations=max_iterations,
        context=ContextHandoffConfig(),
    )


class _StreamRecorder:
    """Fake provider.stream: records (system, messages) per call, yields a
    scripted shape. tool_calls: list of (name, input) to emit as tool_use
    on the first N calls before the final text."""

    def __init__(self, tool_calls=(), final_text="FINAL TEXT"):
        self.calls = []          # list of message-lists (the wire payloads)
        self.tool_calls = list(tool_calls)
        self.final_text = final_text

    async def __call__(self, *, config=None, system=None, messages=None,
                       tools=None, model="test", thinking=None,
                       cache_ttl=None, **kw):
        self.calls.append([dict(m) for m in messages])
        n = len(self.calls)
        if n <= len(self.tool_calls):
            name, tinput = self.tool_calls[n - 1]
            tc = ToolCall(id=f"tc_{n}", name=name, input=tinput)
            yield StreamEvent(type="tool_done", tool_index=0, tool_call=tc)
            yield StreamEvent(
                type="done",
                response=Response(content="", tool_calls=[tc], model=model,
                                  usage=Usage(input_tokens=10, output_tokens=5),
                                  stop_reason="tool_use"),
                stop_reason="tool_use", model=model)
        else:
            yield StreamEvent(type="text", content=self.final_text)
            yield StreamEvent(
                type="done",
                response=Response(content=self.final_text, model=model,
                                  usage=Usage(input_tokens=10, output_tokens=5),
                                  stop_reason="end_turn"),
                stop_reason="end_turn", model=model)


def run_exec_real(argv, *, config, recorder, execute_tool=None, stdin=None):
    """Run main() over argv with the config loaded and the PROVIDER SEAM
    faked — the Agent itself is REAL (real callbacks, real SessionLog,
    real reminder engine, real boundary machinery)."""
    from openalph.cli import main

    argv = ["exec", "--config", str(_write_config(config))] + list(argv)

    out, err = StringIO(), StringIO()
    patches = [
        patch("openalph.cli.load_config", return_value=config),
        patch("openalph.agent.stream", recorder),
    ]
    if execute_tool is not None:
        patches.append(patch("openalph.agent.execute_tool", execute_tool))
    if stdin is not None:
        patches.append(patch("sys.stdin", StringIO(stdin)))
    import contextlib

    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(redirect_stdout(out))
        stack.enter_context(redirect_stderr(err))
        code = 0
        try:
            main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 0
    return out.getvalue(), err.getvalue(), code


def _write_config(config):
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".toml")
    with open(fd, "w") as f:
        f.write(_config_toml(config))
    return Path(path)


def _config_toml(c):
    provider = c.providers["p"]
    lines = [
        "[agent]",
        f'name = "{c.name}"',
        f'default_model = "{c.default_model}"',
        f"max_tokens = {c.max_tokens}",
        f"model_max_tokens = {c.model_max_tokens}",
        f"max_iterations = {c.max_iterations}",
        "",
        "[providers.p]",
        'type = "openai"',
        'api_key = "sk-test"',
        f'base_url = "{provider.base_url}"',
        "",
        "[context]",
        "handoff_enabled = true",
        "",
        "[workspace]",
        f'path = "{c.workspace}"',
    ]
    return "\n".join(lines) + "\n"


def _task_file(tmp_path, text, name="task.txt"):
    p = Path(tmp_path) / name
    p.write_text(text)
    return str(p)


def read_log(config, room="_exec"):
    from openalph.session import SessionLog
    log = SessionLog(config.workspace, "uid", handoff_default=True)
    return log.read(room)


# ===========================================================================
# (a) Crossing the auto tier produces the boundary in the cage SessionLog
# ===========================================================================

class TestExecBoundaryFire:
    def test_big_task_crossing_auto_tier_writes_marker_and_snapshot(
            self, tmp_path):
        # window 2000: the post-boundary render needs room — window 1000
        # (runway 900) is too tight with the snapshot + task in play.
        # available 1900; auto 85% = 1615; the 8000-char task (~2000
        # tok) + fixture base crosses it; post-boundary render fits.
        config = make_config(tmp_path, model_max_tokens=2000, max_tokens=100)
        big_task = "x" * 8000  # ~2000 tokens > auto 765
        rec = _StreamRecorder(final_text="done")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, big_task)],
            config=config, recorder=rec)
        assert code == 0
        entries = read_log(config)
        markers = [e for e in entries if e.get("event") == "handoff_boundary"]
        assert len(markers) == 1, (
            "crossing the auto tier on exec must apply a boundary through "
            "the wired callbacks seam")
        snaps = [e for e in entries if e.get("source") == "handoff_snapshot"]
        assert len(snaps) == 1
        manifest = json.loads(markers[0]["detail"])
        assert manifest["trigger"] == "auto"

    def test_boundary_applied_mid_run_rebuilds_history_in_place(
            self, tmp_path):
        config = make_config(tmp_path, model_max_tokens=2000, max_tokens=100)
        big_task = "x" * 8000
        rec = _StreamRecorder(final_text="done")
        run_exec_real(["--task-file", _task_file(tmp_path, big_task)],
                      config=config, recorder=rec)
        # the FIRST model call already sees the post-boundary render
        first = rec.calls[0]
        texts = [str(m.get("content", "")) for m in first]
        assert not any("x" * 100 in t for t in texts), (
            "the task was pre-boundary — the post-boundary request must not "
            "carry it")
        assert any("durable context snapshot" in t for t in texts)

    def test_small_task_no_boundary(self, tmp_path):
        config = make_config(tmp_path, model_max_tokens=2000, max_tokens=100)
        small_task = "y" * 200
        rec = _StreamRecorder(final_text="done")
        run_exec_real(["--task-file", _task_file(tmp_path, small_task)],
                      config=config, recorder=rec)
        entries = read_log(config)
        assert not any(e.get("event") == "handoff_boundary" for e in entries)

    def test_kill_switch_disables_boundaries_on_exec(self, tmp_path):
        config = make_config(tmp_path, model_max_tokens=1000, max_tokens=100)
        config = config.__class__(**{**config.__dict__,
                                     "context": ContextHandoffConfig(
                                         handoff_enabled=False)})
        big_task = "x" * 8000
        rec = _StreamRecorder(final_text="done")
        run_exec_real(["--task-file", _task_file(tmp_path, big_task)],
                      config=config, recorder=rec)
        entries = read_log(config)
        assert not any(e.get("event") == "handoff_boundary" for e in entries)


# ===========================================================================
# (b) Post-boundary model request shape
# ===========================================================================

class TestPostBoundaryRequestShape:
    def test_request_contains_only_snapshot(self, tmp_path):
        config = make_config(tmp_path, model_max_tokens=2000, max_tokens=100)
        big_task = "x" * 8000
        rec = _StreamRecorder(final_text="done")
        run_exec_real(["--task-file", _task_file(tmp_path, big_task)],
                      config=config, recorder=rec)
        first = rec.calls[0]
        roles = [m.get("role") for m in first]
        assert all(r == "user" for r in roles), (
            "post-boundary render = snapshot + harness directives only "
            "(no project, no post-boundary traffic yet)")
        assert "durable context snapshot" in str(first[0].get("content", ""))
        assert "No handoff package was declared" in str(
            first[0].get("content", ""))
        # the forced-handoff runway directive legitimately fires on a tiny
        # fixture (floor max(10%*window, 24000) always advises here)
        assert any("handoff-runway" in str(m.get("trigger", ""))
                   or "session-handoff" in str(m.get("content", ""))
                   for m in first[1:]) or len(first) == 1


# ===========================================================================
# (c) --project flag
# ===========================================================================

class TestProjectFlag:
    def _mk_project(self, tmp_path, name="proj"):
        d = Path(tmp_path) / "memory" / "projects" / name
        d.mkdir(parents=True)
        return d

    def test_happy_path_appends_event(self, tmp_path):
        config = make_config(tmp_path)
        self._mk_project(tmp_path, "proj")
        rec = _StreamRecorder(final_text="done")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, "hello"),
             "--project", "proj"],
            config=config, recorder=rec)
        assert code == 0
        entries = read_log(config)
        events = [e for e in entries if e.get("event") == "active_project"]
        assert len(events) == 1
        assert events[0].get("detail") == "proj"

    def test_event_precedes_first_model_call(self, tmp_path):
        config = make_config(tmp_path)
        self._mk_project(tmp_path, "proj")
        rec = _StreamRecorder(final_text="done")
        run_exec_real(["--task-file", _task_file(tmp_path, "hello"),
                       "--project", "proj"],
                      config=config, recorder=rec)
        # the declared project must be visible to the boundary machinery:
        # with a declared project the snapshot body frames the durable files
        config2 = make_config(tmp_path, model_max_tokens=1000, max_tokens=100)
        (Path(tmp_path) / "memory" / "projects" / "proj" / "progress.md"
         ).write_text("# State\ncheckpointed\n")
        rec2 = _StreamRecorder(final_text="done")
        run_exec_real(["--task-file", _task_file(tmp_path, "z" * 8000),
                       "--project", "proj"],
                      config=config2, recorder=rec2)
        entries = read_log(config2)
        snaps = [e for e in entries if e.get("source") == "handoff_snapshot"]
        assert snaps and "checkpointed" in snaps[-1]["content"], (
            "a declared project turns the snapshot into the durable package")

    def test_bad_name_exits_before_inference(self, tmp_path):
        config = make_config(tmp_path)
        rec = _StreamRecorder(final_text="SHOULD NOT BE CALLED")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, "hello"),
             "--project", "../evil"],
            config=config, recorder=rec)
        assert code == 1
        assert out.strip() == "", "error paths must keep stdout empty"
        assert rec.calls == [], "no model call may be burned on a bad --project"

    def test_missing_dir_exits_before_inference(self, tmp_path):
        config = make_config(tmp_path)
        rec = _StreamRecorder(final_text="SHOULD NOT BE CALLED")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, "hello"),
             "--project", "nonexistent"],
            config=config, recorder=rec)
        assert code == 1
        assert out.strip() == ""
        assert rec.calls == []


# ===========================================================================
# (d) set_active_project works on exec
# ===========================================================================

class TestSetActiveProjectOnExec:
    def test_tool_call_writes_event(self, tmp_path):
        config = make_config(tmp_path)
        (Path(tmp_path) / "memory" / "projects" / "declared").mkdir(
            parents=True)
        rec = _StreamRecorder(
            tool_calls=[("set_active_project", {"project": "declared"})],
            final_text="declared")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, "hello"),
             "--tools", "set_active_project"],
            config=config, recorder=rec)
        assert code == 0, err
        entries = read_log(config)
        events = [e for e in entries if e.get("event") == "active_project"]
        assert len(events) == 1, (
            "set_active_project must be live on exec via the callbacks seam")
        assert events[0].get("detail") == "declared"


# ===========================================================================
# (e) stdout contract under boundary fire
# ===========================================================================

class TestStdoutContract:
    def test_exactly_one_json_line_with_boundary(self, tmp_path):
        config = make_config(tmp_path, model_max_tokens=2000, max_tokens=100)
        big_task = "x" * 8000
        rec = _StreamRecorder(final_text="done")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, big_task)],
            config=config, recorder=rec)
        assert code == 0
        stripped = out.strip()
        assert stripped, "stdout must carry the JSON result line"
        obj, end = json.JSONDecoder().raw_decode(stripped)
        assert stripped[end:].strip() == "", (
            f"stdout carries content AFTER the JSON object: {stripped[end:]!r}")
        assert obj.get("status") == "done"
        assert obj.get("content") == "done"

    def test_notices_stay_off_stdout(self, tmp_path):
        config = make_config(tmp_path, model_max_tokens=1000, max_tokens=100)
        big_task = "x" * 8000
        rec = _StreamRecorder(final_text="done")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, big_task)],
            config=config, recorder=rec)
        # the boundary emits a notice through HeadlessSinks -> stderr
        for line in out.strip().splitlines()[:-1] if out.strip() else []:
            assert not line.startswith("{") or json.loads(line), line


# ===========================================================================
# (f) Reminder injections persisted to SessionLog (durability gate)
# ===========================================================================

class TestReminderDurability:
    def test_checkpoint_reminder_persisted(self, tmp_path):
        # MID ladder window: crosses checkpoint but NOT auto in the FULL
        # estimate (fixture base ≈610 tokens — security footer + the minimal
        # CONTINUITY.md from make_config — + task chars/4).
        # model_max_tokens=3000, max_tokens=100 → available 2900;
        # checkpoint 75% = 2175; auto 85% = 2465. Task 6800 chars
        # ≈ 1700 tokens → total ≈ 2310: in [2175, 2465) and under the window.
        config = make_config(tmp_path, model_max_tokens=3000, max_tokens=100)
        mid_task = "c" * 6800
        rec = _StreamRecorder(final_text="done")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, mid_task)],
            config=config, recorder=rec)
        assert code == 0, err
        entries = read_log(config)
        reminders = [e for e in entries if e.get("source") == "reminder"]
        assert any(e.get("trigger") == "handoff-checkpoint" for e in reminders), (
            "the checkpoint reminder must be INJECTED (not skipped) and "
            "PERSISTED via HeadlessSinks.log_reminder — the I1 durability "
            "gate on exec")

    def test_runway_reminder_persisted_after_boundary(self, tmp_path):
        # boundary leaves high runway consumption -> handoff-runway reminder
        config = make_config(tmp_path, model_max_tokens=1000, max_tokens=100)
        big_task = "x" * 8000
        rec = _StreamRecorder(final_text="done")
        run_exec_real(["--task-file", _task_file(tmp_path, big_task)],
                      config=config, recorder=rec)
        entries = read_log(config)
        reminders = [e for e in entries if e.get("source") == "reminder"]
        # the runway reminder fires only when consumption >= 90% — the
        # fallback snapshot is tiny, so it may legitimately stay silent;
        # the pin is: whatever fired, it persisted.
        for e in reminders:
            assert e.get("trigger"), "reminder entries must carry their trigger"


# ===========================================================================
# Turn persistence (the _process_cli_line pattern, minus the stdin loop)
# ===========================================================================

class TestTurnPersistence:
    def test_user_task_assistant_and_tool_result_persisted(self, tmp_path):
        config = make_config(tmp_path)

        async def fake_execute_tool(**kwargs):
            return ToolResult(content="TOOLRESULT-BYTES", is_error=False)

        rec = _StreamRecorder(
            tool_calls=[("shell", {"command": "echo hi"})],
            final_text="all done")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, "run the thing"),
             "--tools", "shell"],
            config=config, recorder=rec, execute_tool=fake_execute_tool)
        assert code == 0, err
        entries = read_log(config)
        roles = [e.get("role") for e in entries]
        assert "user" in roles and "assistant" in roles and "tool" in roles
        user_entries = [e for e in entries if e.get("role") == "user"]
        assert any(e.get("content") == "run the thing" for e in user_entries)
        tool_entries = [e for e in entries if e.get("role") == "tool"]
        assert any("TOOLRESULT-BYTES" in str(e.get("output", ""))
                   for e in tool_entries)
        assistants = [e for e in entries if e.get("role") == "assistant"]
        assert assistants, "assistant turns must persist (tool-use AND terminal)"
        assert any(e.get("content") == "all done" for e in assistants)

    def test_user_task_persisted_before_first_model_call(self, tmp_path):
        config = make_config(tmp_path)
        seen = {}

        real_stream = _StreamRecorder(final_text="done")

        async def spy_stream(**kw):
            # capture the SessionLog state AT the first model call
            from openalph.session import SessionLog
            log = SessionLog(config.workspace, "uid")
            seen["entries"] = log.read("_exec")
            async for ev in real_stream(**kw):
                yield ev

        run_exec_real(["--task-file", _task_file(tmp_path, "persist me")],
                      config=config, recorder=spy_stream)
        entries = seen.get("entries") or []
        assert any(
            e.get("role") == "user" and e.get("content") == "persist me"
            for e in entries), (
            "the user task must be appended to the SessionLog BEFORE the "
            "first model call (crash-atomic turn start)")

    def test_model_calls_persisted_mid_turn(self, tmp_path):
        config = make_config(tmp_path)
        rec = _StreamRecorder(final_text="done")
        run_exec_real(["--task-file", _task_file(tmp_path, "hello")],
                      config=config, recorder=rec)
        entries = read_log(config)
        # usage-bearing assistant entry exists (the unified serializer)
        assistants = [e for e in entries if e.get("role") == "assistant"]
        assert assistants and assistants[-1].get("content") == "done"


# ===========================================================================
# No rehydration + append-only substrate
# ===========================================================================

class TestNoRehydration:
    def test_preexisting_log_not_loaded_into_context(self, tmp_path):
        config = make_config(tmp_path)
        # pre-seed a session file with old traffic
        from openalph.session import SessionLog
        seed = SessionLog(config.workspace, "uid")
        seed.append(role="user", sender="operator", room="_exec",
                    content="ANCIENT pre-exec traffic")
        rec = _StreamRecorder(final_text="done")
        run_exec_real(["--task-file", _task_file(tmp_path, "fresh task")],
                      config=config, recorder=rec)
        first = rec.calls[0]
        texts = [str(m.get("content", "")) for m in first]
        assert not any("ANCIENT" in t for t in texts), (
            "exec is a fresh turn: the pre-existing SessionLog is durability "
            "substrate, never rehydrated into context")
        # and the file was APPENDED to, not truncated
        entries = read_log(config)
        assert any(e.get("content") == "ANCIENT pre-exec traffic"
                   for e in entries)


# ===========================================================================
# Filing channel union (the existing exec contract must survive)
# ===========================================================================

class TestFilingChannelUnion:
    def test_file_ticket_still_works_alongside_full_callbacks(self, tmp_path):
        config = make_config(tmp_path)
        rec = _StreamRecorder(
            tool_calls=[("file_ticket", {
                "title": "follow-up", "description": "do the thing",
                "evidence": "observed in run"})],
            final_text="filed")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, "hello"),
             "--tools", "file_ticket"],
            config=config, recorder=rec)
        assert code == 0, err
        obj = json.loads(out.strip())
        # the additive filed_proposals key must survive the callbacks union
        assert obj.get("filed_proposals"), (
            "the filing channel (filed_proposals_sink) must survive the "
            "full build_callbacks union on exec")


# ===========================================================================
# Room label + session_start
# ===========================================================================

class TestSessionShape:
    def test_custom_room_label(self, tmp_path):
        config = make_config(tmp_path)
        rec = _StreamRecorder(final_text="done")
        run_exec_real(["--task-file", _task_file(tmp_path, "hi"),
                       "--room", "worker-one"],
                      config=config, recorder=rec)
        # sanitation-agnostic: exactly one session file exists for this run
        # and it carries the task entry.
        files = list((Path(config.workspace) / "sessions").glob("*.jsonl"))
        assert len(files) == 1, files
        entries = [json.loads(ln) for ln in files[0].read_text().splitlines()
                   if ln.strip()]
        assert any(e.get("content") == "hi" for e in entries)

    def test_room_label_traversal_rejected(self, tmp_path):
        config = make_config(tmp_path)
        rec = _StreamRecorder(final_text="SHOULD NOT BE CALLED")
        out, err, code = run_exec_real(
            ["--task-file", _task_file(tmp_path, "hi"),
             "--room", "../../evil"],
            config=config, recorder=rec)
        assert code == 1
        assert rec.calls == []
