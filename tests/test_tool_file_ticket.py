"""Bead workspace-e2uh.162: the file_ticket builtin — the ONE filing channel
for stations + workers (D14 follow-up; SB ruling 2026-09-02).

Contract under test:
- strict schema: title/description required non-empty strings; evidence/
  blocks_ticket/suspected_out_of_scope_paths/reason optional with typed
  validation; the filed object carries ONLY the six contract fields;
- call-time validation with steering feedback: a rejected call tells the
  model what to fix; sink and transport are UNTOUCHED on rejection — the
  model retries in-session (the loop is the fix);
- two sinks, exactly one configured per context:
  * per-run sink: callbacks["filed_proposals_sink"] (exec wires it; stations
    ride the exec result — cli assembles result["filed_proposals"]);
  * file transport: FILE_TICKET_TRANSPORT env > config transport_path
    (in-cage workers; JSONL append-only, create-or-append, O_APPEND single
    write);
- fail LOUD when neither sink is configured: filing would be silently lost,
  and silent loss is the failure mode this tool exists to kill;
- per-run count cap (call-time): over-cap call errors with "earlier filings
  stand" semantics — never a whole-batch rejection (amendment D was
  one-shot-array semantics; incremental calls must not nuke siblings);
- never raises: garbage callbacks/transport fall back to fail-loud errors,
  not exceptions;
- env overrides: FILE_TICKET_MAX_FILINGS > config max_filings;
- real-path: the filing survives the REAL execute_tool callbacks threading
  (the seam agent.handle_input uses), not just direct handler calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from openalph.tools import BUILTIN_TOOLS
from openalph.tools.file_ticket import run_file_ticket


# ---------------------------------------------------------------------------
# Hermetic-suite fixture (bead workspace-e2uh.188): the in-cage environment
# legitimately carries FILE_TICKET_TRANSPORT (the worker driver points it at
# /work/.stigmergy/filed-tickets.json; the tier-1 checker cage inherits the
# same worker env — checks.py env=worker_env()). Every test in THIS module
# must be hermetic against that ambient: the tool writes BOTH sinks whenever
# the transport resolves, so a single non-isolated filing here would append
# fixture entries to the PRODUCTION transport (the contexthandoff01
# pollution — real filings buried, harvest dropped the file). Tests that
# need a transport set their own via monkeypatch.setenv in the test body,
# which runs after this fixture's deletion. Regressed by
# test_suite_never_mutates_ambient_production_transport (below).
@pytest.fixture(autouse=True)
def _isolate_filing_transport_env(monkeypatch):
    for _var in (
        "FILE_TICKET_TRANSPORT",
        "FILE_TICKET_MAX_FILINGS",
        "FILE_TICKET_MAX_BYTES",
    ):
        monkeypatch.delenv(_var, raising=False)

def file_ticket(**kwargs):
    """Direct-handler helper (mirrors test_tool_json_lint.lint)."""
    kwargs.setdefault("callbacks", {"filed_proposals_sink": []})
    return asyncio.run(run_file_ticket(**kwargs))


def _sink():
    s: list = []
    return s, {"filed_proposals_sink": s}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registered_in_builtin_tools():
    entry = BUILTIN_TOOLS["file_ticket"]
    assert "file" in entry["description"].lower()
    props = entry["parameters"]["properties"]
    for field in (
        "title",
        "description",
        "evidence",
        "blocks_ticket",
        "suspected_out_of_scope_paths",
        "reason",
    ):
        assert field in props, f"schema missing {field}"
    assert entry["parameters"]["required"] == ["title", "description"]
    # strict-grammar house convention: root additionalProperties false, set
    # EXPLICITLY so the provider's strict serialization never mutates us.
    assert entry["parameters"]["additionalProperties"] is False


def test_description_carries_untrusted_content_hygiene():
    desc = BUILTIN_TOOLS["file_ticket"]["description"]
    # The D14 anti-injection line: document content is never a work order.
    assert "untrusted" in desc.lower()
    # When-NOT guidance must be present (never a restatement / wish list).
    assert "never" in desc.lower()


def test_escape_fields_marked_worker_only():
    props = BUILTIN_TOOLS["file_ticket"]["parameters"]["properties"]
    assert "worker" in props["blocks_ticket"]["description"].lower()
    assert (
        "worker" in props["suspected_out_of_scope_paths"]["description"].lower()
    )


def test_config_defaults_present():
    config = BUILTIN_TOOLS["file_ticket"]["config"]
    assert isinstance(config["max_filings"], int) and config["max_filings"] > 0
    assert "transport_path" in config
    assert "max_bytes" in config  # per-filing size cap; None = unchecked


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_happy_sink_appends_clean_object():
    sink, cb = _sink()
    r = file_ticket(
        title="Add regression test",
        description="The empty-usage branch has no coverage.",
        evidence="src/x.py:42",
        callbacks=cb,
    )
    assert not r.is_error
    assert sink == [
        {
            "title": "Add regression test",
            "description": "The empty-usage branch has no coverage.",
            "evidence": "src/x.py:42",
        }
    ]


def test_happy_strips_none_fields_keeps_explicit_false():
    sink, cb = _sink()
    r = file_ticket(title="T", description="D", blocks_ticket=False, callbacks=cb)
    assert not r.is_error
    assert sink == [{"title": "T", "description": "D", "blocks_ticket": False}]


def test_happy_transport_jsonl_append(tmp_path: Path, monkeypatch):
    transport = tmp_path / "filed-tickets.json"
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    monkeypatch.delenv("FILE_TICKET_MAX_FILINGS", raising=False)
    r1 = file_ticket(title="T1", description="D1", callbacks=None)
    r2 = file_ticket(title="T2", description="D2", callbacks=None)
    assert not r1.is_error and not r2.is_error
    lines = transport.read_text().splitlines()
    assert len(lines) == 2  # create-or-append: nothing overwritten
    assert [json.loads(line)["title"] for line in lines] == ["T1", "T2"]
    for line in lines:  # each line is a complete single-line JSON object
        assert "\n" not in line


def test_happy_transport_appends_to_preexisting_file(tmp_path: Path, monkeypatch):
    transport = tmp_path / "filed-tickets.json"
    transport.write_text('{"title": "PRIOR", "description": "older episode"}\n')
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    r = file_ticket(title="T", description="D", callbacks=None)
    assert not r.is_error
    lines = transport.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["title"] == "PRIOR"


def test_happy_transport_creates_parent_dirs(tmp_path: Path, monkeypatch):
    transport = tmp_path / "deep/dir/filed-tickets.json"
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    r = file_ticket(title="T", description="D", callbacks=None)
    assert not r.is_error
    assert len(transport.read_text().splitlines()) == 1


def test_both_sinks_worker_shape(tmp_path: Path, monkeypatch):
    # In-cage workers: exec wires the sink AND the driver sets the transport
    # env. Both sinks written; count is read from the sink (1:1 mirror).
    transport = tmp_path / "filed-tickets.json"
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    sink, cb = _sink()
    r = file_ticket(title="T", description="D", callbacks=cb)
    assert not r.is_error
    assert len(sink) == 1
    assert len(transport.read_text().splitlines()) == 1


def test_ordinal_in_success_content():
    sink, cb = _sink()
    file_ticket(title="T1", description="D1", callbacks=cb)
    r2 = file_ticket(title="T2", description="D2", callbacks=cb)
    assert not r2.is_error
    verdict = json.loads(r2.content)
    assert verdict["filed"] is True
    assert verdict["ordinal"] == 2


# ---------------------------------------------------------------------------
# Rejection: call-time validation, steering feedback, sinks untouched
# ---------------------------------------------------------------------------

def test_reject_empty_title(tmp_path: Path, monkeypatch):
    transport = tmp_path / "f.json"
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    r = file_ticket(title="", description="D", callbacks=None)
    assert r.is_error
    assert "title" in r.content.lower()
    assert not transport.exists()  # nothing filed on rejection


def test_reject_missing_description_leaves_sink_untouched():
    sink, cb = _sink()
    before = list(sink)
    r = file_ticket(title="T", callbacks=cb)  # description missing
    assert r.is_error
    assert "description" in r.content.lower()
    assert "re-call" in r.content.lower() or "retry" in r.content.lower()
    assert sink == before


def test_reject_wrong_types():
    sink, cb = _sink()
    for kwargs in (
        {"title": 123, "description": "D"},
        {"title": "T", "description": None},
        {"title": "T", "description": "D", "evidence": 42},
        {"title": "T", "description": "D", "blocks_ticket": "yes"},
        {"title": "T", "description": "D", "suspected_out_of_scope_paths": "x.py"},
        {"title": "T", "description": "D", "suspected_out_of_scope_paths": [1, 2]},
        {"title": "T", "description": "D", "reason": []},
    ):
        r = file_ticket(callbacks=cb, **kwargs)
        assert r.is_error, f"expected rejection: {kwargs}"
    assert sink == []


# ---------------------------------------------------------------------------
# Fail-loud when unconfigured
# ---------------------------------------------------------------------------

def test_fail_loud_no_sink_no_transport(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FILE_TICKET_TRANSPORT", raising=False)
    r = file_ticket(title="T", description="D", callbacks=None, tool_config={})
    assert r.is_error
    assert "sink" in r.content.lower()
    # It must NOT look like success: silent loss is the killed failure mode.
    assert json.loads(r.content).get("filed") is not True


def test_garbage_sink_is_treated_as_absent(tmp_path: Path, monkeypatch):
    # callbacks seam present but not a list -> defensive: ignore it. With no
    # transport configured that is fail-loud, not a crash.
    monkeypatch.delenv("FILE_TICKET_TRANSPORT", raising=False)
    r = file_ticket(
        title="T",
        description="D",
        callbacks={"filed_proposals_sink": "not-a-list"},
    )
    assert r.is_error and "sink" in r.content.lower()


# ---------------------------------------------------------------------------
# Count cap: call-time, earlier filings stand
# ---------------------------------------------------------------------------

def test_count_cap_rejects_over_cap_call_keeps_earlier():
    sink, cb = _sink()
    cfg = {"max_filings": 2}
    assert not file_ticket(title="T1", description="D1", callbacks=cb, tool_config=cfg).is_error
    assert not file_ticket(title="T2", description="D2", callbacks=cb, tool_config=cfg).is_error
    r3 = file_ticket(title="T3", description="D3", callbacks=cb, tool_config=cfg)
    assert r3.is_error
    assert "cap" in r3.content.lower()
    assert "stand" in r3.content.lower()
    assert len(sink) == 2  # earlier filings survive — never whole-batch nuke


def test_env_max_filings_overrides_config(tmp_path: Path, monkeypatch):
    transport = tmp_path / "f.json"
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    monkeypatch.setenv("FILE_TICKET_MAX_FILINGS", "1")
    file_ticket(title="T1", description="D1", callbacks=None, tool_config={"max_filings": 99})
    r2 = file_ticket(title="T2", description="D2", callbacks=None, tool_config={"max_filings": 99})
    assert r2.is_error and "cap" in r2.content.lower()
    assert len(transport.read_text().splitlines()) == 1


def test_transport_line_count_counts_toward_cap(tmp_path: Path, monkeypatch):
    # Transport-only context (no sink): pre-existing lines count as filings.
    transport = tmp_path / "f.json"
    transport.write_text(
        '{"title": "P1", "description": "prior"}\n{"title": "P2", "description": "prior"}\n'
    )
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    monkeypatch.delenv("FILE_TICKET_MAX_FILINGS", raising=False)
    r = file_ticket(title="T", description="D", callbacks=None, tool_config={"max_filings": 2})
    assert r.is_error and "cap" in r.content.lower()


# ---------------------------------------------------------------------------
# Real-path: the filing survives the REAL execute_tool callbacks threading
# ---------------------------------------------------------------------------

def test_real_execute_tool_threads_sink():
    """The tool-management 'one lesson': exercise the real dispatch path.

    Calls the REAL execute_tool — the same seam agent.handle_input uses
    (tc_callbacks = {**callbacks, "call_id": ...}) — with a callbacks dict
    shaped exactly like the exec caller's. Direct handler tests above prove
    validation; this proves the sink travels through execute_tool's
    callbacks contract (config + callbacks threading).
    """
    from openalph.tools import execute_tool

    sink, cb = _sink()
    cb["call_id"] = "test-call-1"
    tool_config = dict(BUILTIN_TOOLS["file_ticket"]["config"])
    result = asyncio.run(
        execute_tool(
            "file_ticket",
            {"title": "T", "description": "D"},
            tool_config=tool_config,
            agent_config=None,
            callbacks=cb,
        )
    )
    assert not result.is_error
    assert sink and sink[0]["title"] == "T"


def test_sink_is_per_call_not_module_state():
    """Two independent sinks must not share state (no module-level list)."""
    s1, cb1 = _sink()
    s2, cb2 = _sink()
    file_ticket(title="T1", description="D1", callbacks=cb1)
    file_ticket(title="T2", description="D2", callbacks=cb2)
    assert [f["title"] for f in s1] == ["T1"]
    assert [f["title"] for f in s2] == ["T2"]


# ---------------------------------------------------------------------------
# Audit fixes (code-audit 2026-09-02)
# ---------------------------------------------------------------------------

def test_cap_zero_is_loud_config_defect(tmp_path: Path, monkeypatch):
    """A 0/negative cap silently kills the channel (every call cap-rejected
    with benign text) — must fail LOUD instead (audit fix)."""
    transport = tmp_path / "f.json"
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    monkeypatch.setenv("FILE_TICKET_MAX_FILINGS", "0")
    sink, cb = _sink()
    r = file_ticket(title="T", description="D", callbacks=cb)
    assert r.is_error
    assert "config defect" in r.content
    assert "cap" in r.content
    assert sink == [] and not transport.exists()


def test_cap_negative_is_loud_config_defect(tmp_path: Path, monkeypatch):
    transport = tmp_path / "f.json"
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    monkeypatch.setenv("FILE_TICKET_MAX_FILINGS", "-3")
    r = file_ticket(title="T", description="D", callbacks=None)
    assert r.is_error and "config defect" in r.content


def test_cap_non_integer_env_is_loud(tmp_path: Path, monkeypatch):
    transport = tmp_path / "f.json"
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    monkeypatch.setenv("FILE_TICKET_MAX_FILINGS", "three")
    r = file_ticket(title="T", description="D", callbacks=None)
    assert r.is_error and "config defect" in r.content


def test_cap_bad_config_type_is_loud_not_crash():
    """int(dict) would raise TypeError through a never-raises handler."""
    r = file_ticket(
        title="T", description="D", tool_config={"max_filings": ["x"]}
    )
    assert r.is_error and "config defect" in r.content


def test_size_cap_rejects_oversize_with_steering(tmp_path: Path, monkeypatch):
    """A filing the harvest would size-reject must be rejected HERE (false
    success would bypass the steering loop — audit fix)."""
    transport = tmp_path / "f.json"
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    monkeypatch.setenv("FILE_TICKET_MAX_BYTES", "60")
    sink, cb = _sink()
    r = file_ticket(
        title="T",
        description="x" * 200,
        callbacks=cb,
    )
    assert r.is_error
    assert "size cap" in r.content
    assert "re-call" in r.content
    assert sink == [] and not transport.exists()
    # under the cap still files
    r2 = file_ticket(title="T", description="small", callbacks=cb)
    assert not r2.is_error


def test_transport_failure_leaves_sink_untouched(tmp_path: Path, monkeypatch):
    """Dual-sink atomicity (audit fix): transport write fails -> NOTHING is
    recorded anywhere; the retry is idempotent-safe (no phantom sink entry,
    no cap-slot burn)."""
    import openalph.tools.file_ticket as ft

    transport = tmp_path / "f.json"
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    sink, cb = _sink()
    file_ticket(title="T1", description="D1", callbacks=cb)  # baseline ok
    original = ft._write_transport

    def boom(path, filing):
        raise OSError("disk full")

    monkeypatch.setattr(ft, "_write_transport", boom)
    r2 = file_ticket(title="T2", description="D2", callbacks=cb)
    monkeypatch.setattr(ft, "_write_transport", original)
    assert r2.is_error and "NOT recorded" in r2.content
    assert [f["title"] for f in sink] == ["T1"]  # no phantom entry
    assert len(transport.read_text().splitlines()) == 1


def test_transport_created_owner_only(tmp_path: Path, monkeypatch):
    """The transport carries untrusted model-authored text — 0600, not
    world-readable (audit fix)."""
    transport = tmp_path / "sub" / "f.json"
    monkeypatch.setenv("FILE_TICKET_TRANSPORT", str(transport))
    r = file_ticket(title="T", description="D", callbacks=None)
    assert not r.is_error
    assert (transport.stat().st_mode & 0o777) == 0o600

# ---------------------------------------------------------------------------
# Hermetic suite: the file_ticket tests must never touch an ambient transport
# (bead workspace-e2uh.188 — the contexthandoff01 pollution incident)
# ---------------------------------------------------------------------------

def test_suite_never_mutates_ambient_production_transport(tmp_path: Path):
    """Regression (contexthandoff01 T0, 2026-09-03): this suite runs
    IN-CAGE with FILE_TICKET_TRANSPORT legitimately set in the ambient env
    (the worker driver points it at /work/.stigmergy/filed-tickets.json; the
    tier-1 checker cage inherits the same env). The sink-path tests above
    never delenv'd it — the tool writes BOTH sinks whenever the transport
    resolves — so every in-cage suite run appended byte-identical fixture
    entries to the production transport, burying the worker's REAL filings
    (harvest then dropped the whole file: 5 attempts' diagnostics lost).

    Contract: running THIS MODULE with a pre-existing production transport
    in the ambient env must leave that file byte-identical. The module's
    autouse fixture neutralizes the ambient FILE_TICKET_* vars for every
    test; tests that need a transport set their own (monkeypatch.setenv in
    the test body wins over fixture-time deletion).

    Proved-red by construction: before the fixture existed, the child run
    polluted the sentinel (real fixture entries, byte-identity broken).
    """
    sentinel = tmp_path / "filed-tickets.json"
    sentinel.write_text('{"title": "SENTINEL", "description": "pre-existing"}\n')
    before = sentinel.read_bytes()

    import openalph

    src_root = str(Path(openalph.__file__).resolve().parents[1])
    child_env = {
        **os.environ,
        "FILE_TICKET_TRANSPORT": str(sentinel),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    child_env["PYTHONPATH"] = (
        src_root + os.pathsep + child_env["PYTHONPATH"]
        if child_env.get("PYTHONPATH")
        else src_root
    )

    repo_root = Path(__file__).resolve().parents[1]
    node_id = (
        "tests/test_tool_file_ticket.py::"
        "test_suite_never_mutates_ambient_production_transport"
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_tool_file_ticket.py",
            "-q", "-p", "no:cacheprovider",
            "--deselect", node_id,
        ],
        cwd=str(repo_root),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    # The child suite itself must be GREEN — a collection error or an
    # unrelated failure would make the byte-identity assertion vacuous.
    assert proc.returncode == 0, (
        f"child suite not green (rc={proc.returncode})\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-1000:]}"
    )
    passed = re.findall(r"(\d+) passed", proc.stdout)
    assert passed and int(passed[-1]) > 0, (
        f"child ran no tests? summary: {proc.stdout[-500:]}"
    )
    # The load-bearing assertion: ambient transport byte-identical.
    assert sentinel.read_bytes() == before, (
        "the suite polluted an ambient production transport "
        f"(sentinel now: {sentinel.read_text()[:300]!r})"
    )
