"""RED suite — Bundle-2 audit remediation (epic workspace-kdsn.195, bead .195.8).

Pins the fix semantics for the 16 reconciled findings (R1–R16) from the
three-model adversarial audit (tmp/code-audit/file-search-refresh/AUDIT-REPORT.md;
fix spec: specs/file-search-remediation-plan.md).

RED honesty
-----------
Every test asserts the FIXED behavior, so it fails RED against today's
(2a1350c) unfixed source. All failures are ASSERTION failures, never
collection/import errors — every symbol imported here exists today
(execute_tool, run_grep, run_glob, MatrixBot, _display_path, etc.). Where a
finding's repro reproduces a *silent* wrong-write, the test also pins the
file byte-identical so the fix cannot merely change the error text.

R1 (ReDoS) special handling
---------------------------
Catastrophic backtracking runs SYNCHRONOUSLY and CPython's ``re`` engine
holds the GIL for the duration of a *single* match, so a naive in-process
``await execute_tool(...)`` on a pathological pattern would hang the WHOLE
pytest process until the outer suite timeout — a collection-adjacent hang,
not a clean assertion. We therefore run the offending call in a KILLABLE
child process with a hard parent-side join timeout: today the child hangs
(parent kills it → assertion fails RED, bounded wall-clock); once the fix
lands (per-file / per-line-batch ``time.monotonic()`` deadline under
``time_budget_seconds``), the child returns an ``is_error`` result naming
the time budget well within the join window. The fixtures distribute cost
across many moderate units (not one uninterruptible line) because the
spec's stdlib-only deadline can only fire BETWEEN units.

Helpers copied from the test_search_tools.py / test_file_patch.py
convention (no cross-test-module imports exist in this suite → copy per
convention; never modify an existing test file).
"""

import asyncio
import builtins
import multiprocessing
import os
import re
import stat
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from openalph.agent import Agent
from openalph.config import AgentConfig, ProviderConfig, MatrixConfig
from openalph.matrix import MatrixBot
from openalph.provider import ToolCall
from openalph.tools import execute_tool, ToolResult, BUILTIN_TOOLS
from openalph.tools import search as search_mod
from openalph.tools.security import redact_credentials as _redact


# ===========================================================================
# Shared helpers (copied real-path pattern)
# ===========================================================================

def _cfg(workspace, **kw):
    defaults = dict(
        name="test-remediation",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[],
        )},
        workspace=workspace,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
        reminders=True,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


async def _grep(tmp_path, pattern, tool_config=None, agent_config=None, **kw):
    cfg = agent_config if agent_config is not None else _cfg(tmp_path)
    inp = {"pattern": pattern}
    inp.update(kw)
    return await execute_tool(
        name="grep", input=inp, tool_config=tool_config or {},
        agent_config=cfg, tools=None, callbacks={"call_id": "tc"},
    )


async def _glob(tmp_path, pattern, tool_config=None, agent_config=None, **kw):
    cfg = agent_config if agent_config is not None else _cfg(tmp_path)
    inp = {"pattern": pattern}
    inp.update(kw)
    return await execute_tool(
        name="glob", input=inp, tool_config=tool_config or {},
        agent_config=cfg, tools=None, callbacks={"call_id": "tc"},
    )


async def _patch(tmp_path, path, patch_text, tool_config=None):
    cfg = _cfg(tmp_path)
    return await execute_tool(
        name="file_patch",
        input={"path": str(path), "patch": patch_text},
        tool_config=tool_config or {},
        agent_config=cfg, tools=None,
        callbacks={"call_id": "tc", "read_registry": {}},
    )


async def _write(tmp_path, path, content, tool_config=None):
    cfg = _cfg(tmp_path)
    tc = {"require_read_before_write": False}
    if tool_config:
        tc.update(tool_config)
    return await execute_tool(
        name="file_write",
        input={"path": str(path), "content": content},
        tool_config=tc, agent_config=cfg, tools=None,
        callbacks={"call_id": "tc", "read_registry": {}},
    )


async def _edit(tmp_path, path, old, new, tool_config=None, **extra):
    cfg = _cfg(tmp_path)
    inp = {"path": str(path), "old_text": old, "new_text": new}
    inp.update(extra)
    return await execute_tool(
        name="file_edit", input=inp, tool_config=tool_config or {},
        agent_config=cfg, tools=None,
        callbacks={"call_id": "tc", "read_registry": {}},
    )


# ---------------------------------------------------------------------------
# R1 killable-child grep/glob harness (see module docstring)
# ---------------------------------------------------------------------------

def _search_child_body(coro_factory, q):
    """Child-process body: run ONE search coroutine to completion, flush the
    result, then HARD-exit.

    We deliberately use a fresh loop + run_until_complete and then os._exit(0)
    instead of asyncio.run(): once the fix wraps the scan in
    asyncio.to_thread(...), a catastrophic pattern leaves a worker thread
    burning CPU with the GIL held; asyncio.run()'s shutdown_default_executor()
    would then block forever waiting on that thread even AFTER the result is
    known. os._exit(0) skips that cleanup (and pytest atexit), so the child
    exits the instant the (post-fix) is_error result is available.
    """
    payload = ("EXC", "no result")
    try:
        loop = asyncio.new_event_loop()
        r = loop.run_until_complete(coro_factory())
        payload = (r.is_error, r.content[:400])
    except Exception as e:  # pragma: no cover - defensive
        payload = ("EXC", f"{type(e).__name__}: {e}")
    try:
        q.put(payload)
        q.close()
        q.join_thread()  # ensure the feeder thread flushed to the pipe
    except Exception:  # pragma: no cover - defensive
        pass
    os._exit(0)


def _run_search_child(kind, workspace, inp, tool_config, q):
    """execute_tool (await path) child entrypoint."""
    def factory():
        cfg = _cfg(Path(workspace))
        return execute_tool(
            name=kind, input=inp, tool_config=tool_config,
            agent_config=cfg, tools=None, callbacks={"call_id": "tc"},
        )
    _search_child_body(factory, q)


def _run_grep_direct_child(workspace, pattern, config, q):
    """run_grep (direct scan-loop path) child entrypoint."""
    def factory():
        return search_mod.run_grep(
            pattern=pattern, path=None, glob=None, output_mode="content",
            head_limit=None, case_insensitive=False,
            config=config, workspace=Path(workspace),
        )
    _search_child_body(factory, q)


def _bounded_child(target, args, join_timeout=10.0):
    """Run ``target(*args, q)`` in a killable child; return
    (finished, is_error, content). finished=False → the call hung past
    join_timeout (the RED-today signature of an unbounded scan)."""
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(*args, q), daemon=True)
    p.start()
    p.join(join_timeout)
    if p.is_alive():
        p.terminate()
        p.join(3)
        if p.is_alive():
            p.kill()
            p.join(3)
        return (False, None, None)
    try:
        is_error, content = q.get_nowait()
    except Exception:
        return (True, None, None)  # exited but produced no result
    return (True, is_error, content)


# ---------------------------------------------------------------------------
# MatrixBot real-closure harness for R8 (copied real-path pattern)
# ---------------------------------------------------------------------------

ROOM = "!remediation:matrix.local"
AGENT_USER = "@agent:matrix.local"


def _setup_workspace(tmp_path, tools=("shell", "file_read", "file_write",
                                      "file_edit", "subagent", "todo_write")):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(exist_ok=True)
    for name in tools:
        (tools_dir / f"{name}.toml").write_text("[config]\n")
    return tmp_path


def _make_bot_with_real_agent(tmp_path):
    ws = _setup_workspace(tmp_path)
    config = _cfg(ws)
    agent = Agent(config)
    matrix_config = MatrixConfig(
        homeserver="https://matrix.local", user_id=AGENT_USER, device_id="TEST",
        password="test-password", access_token=None, context_reserve=16384,
        sync_timeout=30000, retry_base=1, retry_max=10,
    )
    bot = MatrixBot.__new__(MatrixBot)
    bot.config = matrix_config
    bot.agent = agent
    bot.client = MagicMock()
    bot.client.room_send = AsyncMock(return_value=MagicMock(event_id="$resp1"))
    bot.client.room_typing = AsyncMock()
    bot._current_room = None
    bot._synced = True
    bot._active_rooms = set()
    bot._room_thinking = {}
    bot._room_cache_ttl = {}
    bot._room_timesense = {}
    bot._halted_rooms = set()
    bot._background_tasks = set()
    bot._session_locks = {}
    bot.session_log = MagicMock()
    bot.session_log.append = MagicMock()
    bot.heartbeat = MagicMock()
    bot.heartbeat.is_active = MagicMock(return_value=False)
    bot.umbral = MagicMock()
    bot.umbral.is_active = MagicMock(return_value=False)
    bot._steering_inbox = {}
    bot._active_turns = set()
    return bot, agent


def _notice_contents(bot):
    out = []
    for call in bot.client.room_send.call_args_list:
        if len(call.args) >= 3:
            out.append(call.args[2])
    return out


# A 48-hex secret that matches the generic_hex redaction pattern
# ([0-9a-fA-F]{48,}). Never printed; asserted in-process only.
HEX_SECRET = "deadbeef" * 6  # 48 hex chars


def await_(coro):
    """Run a coroutine to completion on a fresh event loop.

    This module drives async tools synchronously (no @pytest.mark.asyncio) so
    the R1 killable-child harness and the plain assertions share one style.
    """
    return asyncio.run(coro)


# ===========================================================================
# R1 (CRITICAL) — grep/glob ReDoS: wall-clock time budget, no process hang
# ===========================================================================

class TestR1Redos:
    """grep/glob must bound regex scan time via time_budget_seconds and return
    a steering is_error naming the budget, never hang the process.

    Cost is spread over MANY moderate units (lines / filenames), each cheap on
    its own but ruinous in aggregate. A single re match holds the GIL, so the
    only stdlib-only interruption point the fix can use is a time.monotonic()
    deadline checked BETWEEN units (per file / per line-batch / per entry);
    the fixtures make that granularity sufficient to abort well under budget.
    Each call runs in a killable child (module docstring) so today's unbounded
    hang is a clean, bounded RED, never a whole-suite timeout.
    """

    # (a+)+$ vs 'a'*n + 'X' backtracks; n=28 ≈ 4.8s per line, so a handful of
    # such lines already blows a 0.5s budget many times over. Planted across
    # files so the per-file deadline check fires early.
    CATASTROPHIC = r"(a+)+$"

    def _plant_grep_fixture(self, root, n_files=20, n_lines=8, line_len=28):
        line = "a" * line_len + "X\n"
        for i in range(n_files):
            (root / f"redos{i:02d}.txt").write_text(line * n_lines)

    def test_grep_time_budget_via_execute_tool(self, tmp_path):
        """execute_tool(grep) with a catastrophic pattern returns is_error
        naming the time budget in bounded wall-clock (no event-loop hang)."""
        self._plant_grep_fixture(tmp_path)
        t0 = time.monotonic()
        finished, is_error, content = _bounded_child(
            _run_search_child,
            ("grep", str(tmp_path),
             {"pattern": self.CATASTROPHIC, "output_mode": "content"},
             {"time_budget_seconds": 0.5}),
            join_timeout=10.0,
        )
        elapsed = time.monotonic() - t0
        assert finished, (
            "grep with a catastrophic pattern hung past the 10s join window "
            "(no time budget enforced) — the ReDoS fix (per-file/per-line "
            "deadline under time_budget_seconds) is missing")
        assert elapsed < 10.0, f"grep must return well under 10s; took {elapsed:.1f}s"
        assert is_error, f"exceeding the time budget must be an is_error result; got {content!r}"
        assert content is not None and "time budget" in content.lower(), (
            f"budget-exceeded error must mention 'time budget'; got: {content!r}")

    def test_run_grep_time_budget_direct(self, tmp_path):
        """run_grep called directly (the await path's inner scan) also honors
        the deadline — the budget must live in the scan loop, not only in the
        dispatch wrapper."""
        self._plant_grep_fixture(tmp_path)
        finished, is_error, content = _bounded_child(
            _run_grep_direct_child,
            (str(tmp_path), self.CATASTROPHIC, {"time_budget_seconds": 0.5}),
            join_timeout=10.0,
        )
        assert finished, (
            "run_grep hung directly — the deadline must live in the scan loop "
            "so the await path (asyncio.to_thread) is bounded too")
        assert is_error and content is not None and "time budget" in content.lower(), (
            f"direct run_grep must return a time-budget is_error; got {content!r}")

    def test_glob_time_budget_via_execute_tool(self, tmp_path):
        """glob's _glob_pattern_to_regex backtracks on 'a*...b' vs a long
        all-'a' filename; the same budget must bound per-entry matching."""
        # Distinct long all-'a' names (no 'b'); ~1s each × many entries ≫ 0.5s.
        for i in range(20):
            (tmp_path / ("a" * 34 + f"{i:03d}")).write_text("x\n")
        t0 = time.monotonic()
        finished, is_error, content = _bounded_child(
            _run_search_child,
            ("glob", str(tmp_path),
             {"pattern": "a*" * 10 + "b"},
             {"time_budget_seconds": 0.5}),
            join_timeout=10.0,
        )
        elapsed = time.monotonic() - t0
        assert finished, (
            "glob with a backtracking pattern hung past 10s — the per-entry "
            "time budget is missing")
        assert elapsed < 10.0, f"glob must return well under 10s; took {elapsed:.1f}s"
        assert is_error, f"glob exceeding the time budget must be an is_error result; got {content!r}"
        assert content is not None and "time budget" in content.lower(), (
            f"glob budget-exceeded error must mention 'time budget'; got {content!r}")


# ===========================================================================
# R2 (HIGH) — file_patch: fence-shaped line INSIDE content → clean is_error
# ===========================================================================

class TestR2FenceContent:
    """A SEARCH/REPLACE body containing a fence-shaped line (e.g. '=======')
    must be a clean is_error naming the hunk and steering to file_edit — never
    a silent boundary-shifted wrong write. File must be byte-identical."""

    def test_divider_in_content_rejected_file_untouched(self, tmp_path):
        """opus T1/e2e-A: SEARCH intends 'Section\\n=======\\nBody' but the
        inner '=======' is misread as the divider → today a silent wrong
        write. Fix: reject naming the hunk; file byte-identical."""
        f = tmp_path / "r2.md"
        f.write_text("Section\n")
        before = f.read_bytes()
        patch = ("<<<<<<< SEARCH\nSection\n=======\nBody\n"
                 "=======\nX\n>>>>>>> REPLACE")
        res = await_(_patch(tmp_path, f, patch))
        assert res.is_error, (
            "a hunk whose body contains a fence-shaped line ('=======') must be "
            "rejected, not silently boundary-shifted into a wrong write")
        assert "hunk" in res.content.lower(), "error must name the hunk"
        assert f.read_bytes() == before, (
            "V1: fence-in-content rejection must leave the file byte-identical")

    def test_conflict_marker_resolution_not_corrupted(self, tmp_path):
        """Adjacent edge: resolving a git-conflict block whose SEARCH holds
        '=======' must not silently produce still-conflicted output."""
        f = tmp_path / "conflict.md"
        original = "<<<<<<< HEAD\nmine\n=======\ntheirs\n>>>>>>> br\nkeep\n"
        f.write_text(original)
        before = f.read_bytes()
        # SEARCH body legitimately contains a bare '=======' divider-shaped line.
        patch = ("<<<<<<< SEARCH\nmine\n=======\ntheirs\n"
                 "=======\nRESOLVED\n>>>>>>> REPLACE")
        res = await_(_patch(tmp_path, f, patch))
        assert res.is_error, (
            "a SEARCH body carrying a '=======' line must be rejected cleanly")
        assert f.read_bytes() == before, (
            "V1: file must be byte-identical after the fence-in-content rejection")

    def test_search_body_fence_shaped_line_rejected(self, tmp_path):
        """SEARCH-body half of the R2 check: patching a file that itself
        documents the patch format (its SEARCH body carries a
        '>>>>>>> REPLACE'-shaped line) must be rejected, not silently applied.

        Parser trace today: in the 'search' state a '>>>>>>> REPLACE' line is
        NOT the divider, so it is appended to the SEARCH body; the hunk parses
        as SEARCH='foo\\n>>>>>>> REPLACE' and silently applies. The fix rejects
        any SEARCH body containing a fence-shaped line."""
        f = tmp_path / "r2c.md"
        f.write_text("foo\n>>>>>>> REPLACE\nkeep\n")
        before = f.read_bytes()
        patch = ("<<<<<<< SEARCH\nfoo\n>>>>>>> REPLACE\n=======\n"
                 "replaced\n>>>>>>> REPLACE")
        res = await_(_patch(tmp_path, f, patch))
        assert res.is_error, (
            "a SEARCH body containing a fence-shaped line ('>>>>>>> REPLACE') "
            "must be rejected, not silently applied")
        assert f.read_bytes() == before, "V1: file byte-identical after rejection"


# ===========================================================================
# R3 (HIGH) — file_patch: open block never silently dropped
# ===========================================================================

class TestR3OpenBlock:
    """A block left open at EOF, or a nested SEARCH-start inside a block, is a
    hard error naming the hunk + missing fence — never a silent swallow or a
    partial apply. File byte-identical."""

    def test_missing_replace_fence_swallow_rejected(self, tmp_path):
        """opus T2: 2-hunk patch missing hunk-1's REPLACE close is today parsed
        as ONE hunk that writes literal fence text. Fix: hard error, untouched."""
        f = tmp_path / "r3a.txt"
        f.write_text("a\nc\nkeep\n")
        before = f.read_bytes()
        patch = ("<<<<<<< SEARCH\na\n=======\nb\n"
                 "<<<<<<< SEARCH\nc\n=======\nd\n>>>>>>> REPLACE")
        res = await_(_patch(tmp_path, f, patch))
        assert res.is_error, (
            "a nested '<<<<<<< SEARCH' inside an open block must be a hard error, "
            "not silently absorbed into the REPLACE body")
        assert "hunk" in res.content.lower(), "error must name the hunk index"
        assert f.read_bytes() == before, "V1: file byte-identical after the parse error"

    def test_open_block_at_eof_no_partial_apply(self, tmp_path):
        """codex: one valid hunk followed by a second hunk missing its REPLACE
        close → today applies hunk 1 only ('Applied 1 hunk(s)'). Fix: whole
        patch is a hard error (all-or-none, V1); file byte-identical."""
        f = tmp_path / "r3b.txt"
        f.write_text("x\nz\n")
        before = f.read_bytes()
        patch = ("<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n"
                 "<<<<<<< SEARCH\nz\n=======\nw\n")  # 2nd hunk unclosed
        res = await_(_patch(tmp_path, f, patch))
        assert res.is_error, (
            "an open block at EOF must fail the whole patch (all-or-none, V1) — "
            "not apply the earlier hunk and drop the trailing one")
        assert f.read_bytes() == before, (
            "V1: no partial application — file must be byte-identical")

    def test_open_block_error_names_hunk_index(self, tmp_path):
        """The open-block error must NAME the hunk index (steering), not fall
        back to the generic 'no valid blocks' message.

        RED-honesty: a single unclosed hunk already errors today, but with the
        GENERIC zero-block text — which happens to contain 'REPLACE' (from the
        worked example) yet never names the offending hunk. Pinning 'hunk 1'
        distinguishes the required open-block-specific error from today's
        generic one."""
        f = tmp_path / "r3c.txt"
        f.write_text("only\n")
        patch = "<<<<<<< SEARCH\nonly\n=======\nreplacement\n"  # no REPLACE close
        res = await_(_patch(tmp_path, f, patch))
        assert res.is_error, "a single unclosed hunk must error, not be dropped"
        assert "no valid SEARCH/REPLACE blocks found" not in res.content, (
            "an open block must yield the open-block-specific error, not the "
            "generic zero-block message")
        assert "hunk 1" in res.content.lower(), (
            "the open-block error must name the offending hunk index to steer recovery")


# ===========================================================================
# R4 (HIGH) — file_patch: split('\n'), not splitlines() (separator fidelity)
# ===========================================================================

class TestR4Splitlines:
    """The patch parser must split on '\\n' only, so exotic Unicode/control
    line separators (form-feed etc.) that open() preserves are matched
    verbatim (SEARCH round-trips) and written verbatim (REPLACE lands)."""

    def test_form_feed_in_search_matches(self, tmp_path):
        """A file with a form-feed (open() preserves \\x0c) must be matchable by
        a SEARCH containing the same form-feed — today splitlines() normalizes
        it to \\n so the match spuriously fails."""
        f = tmp_path / "ff.txt"
        f.write_text("alpha\x0cbeta\nkeep\n")
        patch = "<<<<<<< SEARCH\nalpha\x0cbeta\n=======\nreplaced\n>>>>>>> REPLACE"
        res = await_(_patch(tmp_path, f, patch))
        assert not res.is_error, (
            "form-feed in SEARCH must match a form-feed file (split('\\n'), not "
            f"splitlines()): {res.content}")
        assert f.read_text() == "replaced\nkeep\n", (
            "the form-feed line must be found and replaced")

    def test_form_feed_in_replace_lands_verbatim(self, tmp_path):
        """REPLACE content with a form-feed must land on disk verbatim — today
        splitlines()+'\\n'.join() silently rewrites \\x0c to \\n."""
        f = tmp_path / "ffr.txt"
        f.write_text("target\n")
        patch = "<<<<<<< SEARCH\ntarget\n=======\npart1\x0cpart2\n>>>>>>> REPLACE"
        res = await_(_patch(tmp_path, f, patch))
        assert not res.is_error, f"patch must apply: {res.content}"
        out = f.read_text()
        assert "\x0c" in out, (
            "form-feed in REPLACE must be written verbatim, not normalized to \\n")
        assert out == "part1\x0cpart2\n", f"REPLACE must land byte-exact; got {out!r}"

    def test_line_separator_u2028_in_search_matches(self, tmp_path):
        """Adjacent edge: U+2028 LINE SEPARATOR is another char splitlines()
        splits on but open() preserves — SEARCH must still match."""
        f = tmp_path / "ls.txt"
        f.write_text("one\u2028two\nkeep\n")
        patch = "<<<<<<< SEARCH\none\u2028two\n=======\ndone\n>>>>>>> REPLACE"
        res = await_(_patch(tmp_path, f, patch))
        assert not res.is_error, (
            f"U+2028 in SEARCH must match (split on '\\n' only): {res.content}")
        assert f.read_text() == "done\nkeep\n"


# ===========================================================================
# R5 (HIGH) — grep/glob: zero-match footer honesty (skips + bound-hit)
# ===========================================================================

class TestR5ZeroMatchFooter:
    """A zero-match result must NOT claim definitive absence when files were
    skipped or the scan was bounded — the skip/bound footer is computed before
    the no-match return, for grep (all modes) and glob."""

    def test_binary_only_zero_match_mentions_skip(self, tmp_path):
        """Workspace with ONLY a binary file containing the needle → grep must
        report the skip, not a bare 'No matches' (a false negative today)."""
        (tmp_path / "blob.bin").write_bytes(b"\x00\x01needle\x00\x02")
        res = await_(_grep(tmp_path, "needle"))
        assert not res.is_error, "zero matches is a hint, not an error"
        assert "skip" in res.content.lower(), (
            "no-match text must mention skipped files when all candidates were "
            "skipped (binary/oversize) — never a bare definitive absence")
        assert "No matches for pattern 'needle'. Try broadening" not in res.content, (
            "the misleading bare no-match text must not stand alone when skips "
            "exist (it claims absence of a present-but-unscanned needle)")

    def test_bound_hit_zero_match_mentions_incomplete(self, tmp_path):
        """max_scan_files reached before a later matching file → zero-match text
        must warn the scan was incomplete, not claim definitive absence."""
        # Only-first-scanned file lacks the needle; needle lives past the bound.
        (tmp_path / "aaa_first.txt").write_text("nothing here\n")
        (tmp_path / "zzz_last.txt").write_text("the needle is here\n")
        res = await_(_grep(tmp_path, "needle",
                           tool_config={"max_scan_files": 1}))
        assert not res.is_error, "zero-in-scanned-prefix is a hint, not an error"
        low = res.content.lower()
        assert ("incomplete" in low or "scan limit" in low or "narrow" in low), (
            "a bound-limited zero-match must warn results may be incomplete, "
            "not claim definitive absence")

    def test_glob_bound_hit_zero_match_mentions_incomplete(self, tmp_path):
        """glob variant: bounded scan with no match must also carry the footer
        rather than a bare 'No matches ... Try a broader pattern'."""
        for i in range(6):
            (tmp_path / f"present{i}.txt").write_text("x\n")
        res = await_(_glob(tmp_path, "*.md",
                          tool_config={"max_scan_files": 2}))
        assert not res.is_error, "glob zero-match is a hint, not an error"
        low = res.content.lower()
        assert ("incomplete" in low or "scan limit" in low or "narrow" in low), (
            "glob bound-limited zero-match must warn the scan was incomplete")


# ===========================================================================
# R6 (HIGH) — validate.py: atomic write (tempfile + os.replace), mode preserved
# ===========================================================================

class TestR6AtomicWrite:
    """_write_now must write via a tempfile + os.replace so a mid-write failure
    leaves the original byte-identical with no orphan tmp; overwriting preserves
    the prior file mode."""

    def test_replace_failure_leaves_original_and_no_orphan(self, tmp_path, monkeypatch):
        """Monkeypatch os.replace to raise mid-write → original byte-identical
        AND no orphan temp files left in the target directory."""
        f = tmp_path / "atomic.txt"
        f.write_text("ORIGINAL CONTENT\n")
        before = f.read_bytes()
        dir_before = set(os.listdir(tmp_path))

        import openalph.tools.validate as validate_mod

        def _boom(src, dst, *a, **k):
            raise OSError("simulated os.replace failure mid-write")
        monkeypatch.setattr(validate_mod.os, "replace", _boom, raising=False)

        res = await_(_write(tmp_path, f, "NEW CONTENT THAT MUST NOT LAND\n"))
        assert res.is_error, (
            "a failed atomic replace must surface as an error, not a silent success")
        assert f.read_bytes() == before, (
            "V1: original file must be byte-identical when os.replace fails")
        dir_after = set(os.listdir(tmp_path))
        orphans = dir_after - dir_before
        assert not orphans, (
            f"a failed atomic write must leave no orphan temp files: {orphans}")

    def test_mode_preserved_on_overwrite(self, tmp_path):
        """A successful overwrite of a 0o640 file must preserve mode 0o640
        (tempfile+replace must chmod to the prior file's mode)."""
        f = tmp_path / "modes.txt"
        f.write_text("orig\n")
        os.chmod(f, 0o640)
        before_mode = stat.S_IMODE(os.stat(f).st_mode)
        assert before_mode == 0o640, "fixture sanity: file starts 0o640"

        res = await_(_write(tmp_path, f, "new content\n"))
        assert not res.is_error, f"overwrite must succeed: {res.content}"
        after_mode = stat.S_IMODE(os.stat(f).st_mode)
        assert after_mode == 0o640, (
            f"atomic overwrite must preserve prior mode 0o640; got 0o{after_mode:o} "
            "(a fresh NamedTemporaryFile defaults to 0o600 — must chmod back)")


# ===========================================================================
# R7 (MEDIUM) — search.py: symlink files skipped + counted; roots resolved
# ===========================================================================

class TestR7Symlinks:
    """grep must not read through a symlinked FILE whose target is outside the
    workspace; skipped symlinks are counted in the footer. (Symlinked dir
    descent is already pinned green elsewhere; extended here for the file case.)"""

    def test_symlinked_file_content_not_returned(self, tmp_path):
        """A symlink inside the workspace pointing at an outside file with the
        needle must NOT have its content returned by grep."""
        outside_dir = Path(tempfile.mkdtemp())
        target = outside_dir / "target.txt"
        target.write_text("OUTSIDE_NEEDLE lives here\n")
        (tmp_path / "real.txt").write_text("just a normal line\n")
        os.symlink(target, tmp_path / "slink.txt")

        res = await_(_grep(tmp_path, "OUTSIDE_NEEDLE", output_mode="content"))
        assert not res.is_error, res.content
        assert "OUTSIDE_NEEDLE" not in res.content, (
            "grep must not read through a symlinked file to outside-workspace "
            "content (no-follow policy)")
        assert "slink.txt" not in res.content, (
            "the symlinked file must not appear as a match")

    def test_symlinked_file_counted_in_footer(self, tmp_path):
        """A skipped symlink file is reported in the footer (N symlink(s)
        skipped) so the result is honest about what it did not read."""
        outside_dir = Path(tempfile.mkdtemp())
        target = outside_dir / "t.txt"
        target.write_text("content\n")
        os.symlink(target, tmp_path / "slink.txt")
        (tmp_path / "real.txt").write_text("hello\n")

        res = await_(_grep(tmp_path, "hello"))
        assert not res.is_error, res.content
        assert "symlink" in res.content.lower(), (
            "the footer must report skipped symlink(s) when >0")


# ===========================================================================
# R8 (MEDIUM) — matrix.py: subagent notice escapes + caps tool-influenced text
# ===========================================================================

class TestR8SubagentNotice:
    """The subagent spawn/return notice branch must html-escape task_preview
    and result_preview (no raw mistune.html on tool-influenced text) and cap
    the result preview at 2000 chars with a truncation marker on error."""

    def test_subagent_result_script_escaped(self, tmp_path):
        """A subagent error result containing <script> must be html-escaped in
        formatted_body — never raw. Exercised through the REAL _tool_notice."""
        bot, _ = _make_bot_with_real_agent(tmp_path)
        tool_notice, _ = bot._make_tool_callbacks(ROOM)
        await_(tool_notice("c1", "subagent", {"task": "do x", "model": "m"},
                          "<script>alert(1)</script>", True))
        c = _notice_contents(bot)[0]
        fb = c.get("formatted_body", "")
        assert "<script>alert(1)</script>" not in fb, (
            "raw <script> from subagent result must never reach formatted_body")
        assert "&lt;script&gt;" in fb, (
            "the subagent result must be html-escaped (escape, not raw mistune.html)")

    def test_subagent_task_script_escaped_on_dispatch(self, tmp_path):
        """The dispatch (_tool_intent) task brief must also escape <script>."""
        bot, _ = _make_bot_with_real_agent(tmp_path)
        _, tool_intent = bot._make_tool_callbacks(ROOM)
        tc = ToolCall(id="c2", name="subagent",
                      input={"task": "<script>evil</script>", "model": "m"})
        await_(tool_intent([tc], ""))
        c = _notice_contents(bot)[0]
        fb = c.get("formatted_body", "")
        assert "<script>evil</script>" not in fb, (
            "raw <script> in a subagent task brief must never reach formatted_body")
        assert "&lt;script&gt;" in fb, "the task brief must be html-escaped"

    def test_subagent_error_result_capped_2000(self, tmp_path):
        """A subagent error result >2000 chars must be capped with a marker
        (parity with the generic error branch), not embedded uncapped."""
        bot, _ = _make_bot_with_real_agent(tmp_path)
        tool_notice, _ = bot._make_tool_callbacks(ROOM)
        big = "Z" * 5000
        await_(tool_notice("c3", "subagent", {"task": "t", "model": "m"}, big, True))
        c = _notice_contents(bot)[0]
        fb = c.get("formatted_body", "")
        assert fb.count("Z") <= 2000, (
            f"subagent error result must be capped at 2000 chars; got {fb.count('Z')}")
        assert "truncat" in fb.lower(), (
            "a capped subagent error result must carry a truncation marker")


# ===========================================================================
# kdsn.257 — subagent notice line-break parity with the advisor fix
# (kdsn.198.9, dc9b550): _escape_preserve_breaks instead of plain html_escape
# ===========================================================================

class TestSubagentNoticeLineBreaks:
    """The subagent spawn (task brief) and return (task brief + result) notice
    folds must preserve real newlines as <br> — the same fix already applied
    to the advisor spawn/return notices (kdsn.198.9) — instead of collapsing
    a multi-line brief/result to a run-on line via plain html_escape."""

    def test_return_notice_task_and_result_preserve_line_breaks(self, tmp_path):
        bot, _ = _make_bot_with_real_agent(tmp_path)
        tool_notice, _ = bot._make_tool_callbacks(ROOM)
        task = "Step one.\nStep two.\n\nFinal step."
        result = "Found A.\nFound B.\n\nDone."
        await_(tool_notice("c4", "subagent", {"task": task, "model": "m"},
                          result, False))
        c = _notice_contents(bot)[0]
        fb = c.get("formatted_body", "")
        assert "Step one.<br>Step two.<br><br>Final step." in fb, (
            "return-notice task brief must keep line breaks (<br>), not fold to a run-on")
        assert "Found A.<br>Found B.<br><br>Done." in fb, (
            "return-notice result must keep line breaks (<br>), not fold to a run-on")
        assert "Step one.\nStep two." not in fb, (
            "literal-newline run-on (folded to a space by clients) must be gone")
        # Still html.escape'd, not raw HTML / markdown-rendered.
        await_(tool_notice("c5", "subagent", {"task": "line1\n<b>x</b>", "model": "m"},
                          "res1\n<i>y</i>", False))
        fb2 = _notice_contents(bot)[1].get("formatted_body", "")
        assert "line1<br>&lt;b&gt;x&lt;/b&gt;" in fb2, (
            "task brief must be html.escape'd with breaks preserved (no raw <b>)")
        assert "res1<br>&lt;i&gt;y&lt;/i&gt;" in fb2, (
            "result must be html.escape'd with breaks preserved (no raw <i>)")

    def test_spawn_notice_task_brief_preserves_line_breaks(self, tmp_path):
        bot, _ = _make_bot_with_real_agent(tmp_path)
        _, tool_intent = bot._make_tool_callbacks(ROOM)
        task = "Do X.\nThen Y.\n\nReport back."
        tc = ToolCall(id="c6", name="subagent", input={"task": task, "model": "m"})
        await_(tool_intent([tc], ""))
        c = _notice_contents(bot)[0]
        fb = c.get("formatted_body", "")
        assert "Do X.<br>Then Y.<br><br>Report back." in fb, (
            "spawn-notice task brief must keep line breaks (<br>), not fold to a run-on")
        assert "Do X.\nThen Y." not in fb, (
            "literal-newline run-on (folded to a space by clients) must be gone")


# ===========================================================================
# R9 (MEDIUM) — tools/__init__.py: redaction on EVERY return path
# ===========================================================================

class TestR9RedactAllPaths:
    """The redaction tail must apply to early error returns too — unknown-tool,
    and the file_write guard refusal — so a secret-bearing input never echoes
    raw into the result (and thence a Matrix notice)."""

    def test_unknown_tool_name_redacted(self, tmp_path):
        """An unknown tool name containing a 48-hex secret must return content
        with [REDACTED:...], not the raw hex (early return bypasses redaction
        today)."""
        res = await_(execute_tool(
            name=HEX_SECRET, input={}, tool_config={},
            agent_config=_cfg(tmp_path), tools=None, callbacks={"call_id": "t"},
        ))
        assert res.is_error, "unknown tool must be an error"
        assert HEX_SECRET not in res.content, (
            "the raw secret in an unknown-tool name must be redacted before return")
        assert "[REDACTED" in res.content, (
            "the unknown-tool error must carry a [REDACTED:*] marker (redaction "
            "tail must cover early returns)")

    def test_write_guard_refusal_redacted(self, tmp_path):
        """The file_write read-before-write guard refusal echoes the path; a
        secret-bearing path must be redacted, not echoed raw."""
        f = tmp_path / (HEX_SECRET + ".txt")
        f.write_text("exists\n")  # exists, NOT read this session → guard fires
        res = await_(execute_tool(
            name="file_write",
            input={"path": str(f), "content": "new\n"},
            tool_config={"require_read_before_write": True},
            agent_config=_cfg(tmp_path), tools=None,
            callbacks={"call_id": "t", "read_registry": {}},
        ))
        assert res.is_error, "unread existing file must be refused by the guard"
        assert "not read" in res.content.lower(), "sanity: this is the guard path"
        assert HEX_SECRET not in res.content, (
            "a secret-bearing path in a guard refusal must be redacted before return")
        assert "[REDACTED" in res.content, (
            "the guard refusal must carry a [REDACTED:*] marker")


# ===========================================================================
# R10 (MEDIUM) — file.py/validate.py: no mkdir side effect on reject
# ===========================================================================

class TestR10RejectNoMkdir:
    """A rejected new-file write (broken code in new nested dirs) must leave no
    directory trace — mkdir happens only after the validation verdict."""

    def test_rejected_new_py_leaves_no_parent_dirs(self, tmp_path):
        """write_file('newpkg/sub/broken.py', broken) rejected → newpkg/sub must
        NOT exist afterward."""
        target = tmp_path / "newpkg" / "sub" / "broken.py"
        res = await_(_write(tmp_path, target, "def (:\n"))
        assert res.is_error, "broken new .py must be rejected by validation"
        assert not target.exists(), "the rejected file must not be created"
        assert not (tmp_path / "newpkg" / "sub").exists(), (
            "the reject path must leave no newly-created parent directories")
        assert not (tmp_path / "newpkg").exists(), (
            "no partial directory tree may remain after a rejected write")

    def test_rejected_new_json_leaves_no_parent_dirs(self, tmp_path):
        """Adjacent edge: a broken new .json in new dirs also leaves no trace."""
        target = tmp_path / "cfgdir" / "broken.json"
        res = await_(_write(tmp_path, target, "{bad json\n"))
        assert res.is_error, "broken new .json must be rejected"
        assert not (tmp_path / "cfgdir").exists(), (
            "rejected .json write must not leave its parent dir behind")


# ===========================================================================
# R11 (MEDIUM) — search.py: validate output_mode + head_limit
# ===========================================================================

class TestR11ParamValidation:
    """grep must reject an unknown output_mode (naming the valid set) and a
    head_limit < 1 (mirroring read_file's posture), instead of silently
    mis-moding or negative-slicing."""

    def test_unknown_output_mode_errors(self, tmp_path):
        """output_mode='kontent' must be an is_error naming valid modes — today
        it silently falls through to files_with_matches."""
        (tmp_path / "t.txt").write_text("needle\n")
        res = await_(_grep(tmp_path, "needle", output_mode="kontent"))
        assert res.is_error, "an unknown output_mode must be rejected, not silent"
        low = res.content.lower()
        assert "output_mode" in low or "content" in low, (
            "the error must name the valid output_mode values")

    def test_head_limit_zero_errors(self, tmp_path):
        """head_limit=0 must be an is_error, not an empty body + 'first 0 of N'."""
        (tmp_path / "t.txt").write_text("needle\n")
        res = await_(_grep(tmp_path, "needle", head_limit=0))
        assert res.is_error, "head_limit=0 must be rejected (mirror read_file)"

    def test_head_limit_negative_errors(self, tmp_path):
        """head_limit=-1 must be an is_error, not a from-the-end negative slice."""
        (tmp_path / "t.txt").write_text("needle\n")
        res = await_(_grep(tmp_path, "needle", head_limit=-1))
        assert res.is_error, "negative head_limit must be rejected, not negative-sliced"

    def test_glob_head_limit_negative_errors(self, tmp_path):
        """glob shares the head_limit posture — negative must also error."""
        (tmp_path / "a.txt").write_text("x\n")
        res = await_(_glob(tmp_path, "*.txt", head_limit=-1))
        assert res.is_error, "glob negative head_limit must be rejected too"


# ===========================================================================
# R12 (MEDIUM) — file.py: edit_file rejects empty old_text (both modes)
# ===========================================================================

class TestR12EmptyOldText:
    """edit_file must reject an empty old_text in BOTH modes with a steering
    'must not be empty' message, leaving the file untouched — today
    replace_all=True shreds the file per-character."""

    def test_empty_old_text_replace_all_rejected_untouched(self, tmp_path):
        """old_text='' with replace_all=True must error, file untouched — today
        it inserts new_text between every character."""
        f = tmp_path / "shred.txt"
        f.write_text("hello world\n")
        before = f.read_bytes()
        res = await_(_edit(tmp_path, f, "", "X", replace_all=True))
        assert res.is_error, (
            "empty old_text with replace_all=True must be rejected, not "
            "per-character-shredded")
        assert "must not be empty" in res.content.lower(), (
            "the rejection must steer 'old_text must not be empty'")
        assert f.read_bytes() == before, "file must be byte-identical after rejection"

    def test_empty_old_text_single_mode_rejected_untouched(self, tmp_path):
        """old_text='' with replace_all=False must also error with the empty
        steering — today it errors with a DIFFERENT ('appears N times') message
        so pin the specific text; file untouched."""
        f = tmp_path / "shred2.txt"
        f.write_text("abc def\n")
        before = f.read_bytes()
        res = await_(_edit(tmp_path, f, "", "X", replace_all=False))
        assert res.is_error, "empty old_text (single mode) must be rejected"
        assert "must not be empty" in res.content.lower(), (
            "single-mode empty old_text must use the 'must not be empty' steering, "
            "not the ambiguous 'appears N times' message")
        assert f.read_bytes() == before, "file must be byte-identical after rejection"


# ===========================================================================
# R13 (LOW) — validate.py: pre-check open() race → new-file semantics
# ===========================================================================

class TestR13Toctou:
    """If the pre-check open() races and raises FileNotFoundError (file deleted
    between the exists-check and the read), the write must be treated with
    NEW-FILE semantics (broken new content REJECTED), not fail-open-written."""

    def test_precheck_race_treated_as_new_file(self, tmp_path):
        """Patch builtins.open so the FIRST read of an existing clean .py raises
        FileNotFoundError → broken candidate must be REJECTED (new-file: born
        clean), not written through a generic fail-open."""
        f = tmp_path / "race.py"
        f.write_text("x = 1\n")  # exists and clean
        broken = "x = = 1\n"
        real_open = builtins.open
        target = str(f)
        state = {"raised": False}

        def fake_open(file, mode="r", *a, **k):
            # Intercept only the FIRST read-open of the target (the pre-check).
            if (str(file) == target and "r" in mode
                    and "w" not in mode and "a" not in mode
                    and not state["raised"]):
                state["raised"] = True
                raise FileNotFoundError(f"race: {file} vanished")
            return real_open(file, mode, *a, **k)

        builtins.open = fake_open
        try:
            res = await_(_write(tmp_path, f, broken))
        finally:
            builtins.open = real_open

        assert state["raised"], "fixture sanity: the pre-check open must have raced"
        assert res.is_error, (
            "a FileNotFoundError race on the pre-check must fall into NEW-FILE "
            "semantics (born-clean → broken candidate REJECTED), not a generic "
            "fail-open write")
        assert f.read_text() != broken, (
            "the broken candidate must not have been fail-open-written to disk")


# ===========================================================================
# R14 (LOW) — validate.py: checker detail names the real target, not /tmp
# ===========================================================================

class TestR14CheckerPathScrub:
    """A subprocess-checker rejection detail relayed to the model must reference
    the REAL target filename, not the NamedTemporaryFile /tmp path."""

    def test_bash_reject_detail_names_real_target(self, tmp_path, monkeypatch):
        """Broken .sh rejected via the real bash checker → the detail must name
        'checker.sh' (real target), and must NOT leak the NamedTemporaryFile
        basename (e.g. 'tmpXXXX.sh')."""
        import shutil
        if shutil.which("bash") is None:
            pytest.skip("bash not on PATH — subprocess checker unavailable")
        f = tmp_path / "checker.sh"
        # New file, broken syntax → pre_ok vacuously true → clean→broken reject.
        res = await_(_write(tmp_path, f, "if then\nfi\n"))
        assert res.is_error, "broken .sh must be rejected (drives the checker path)"
        assert "checker.sh" in res.content, (
            "the rejection detail must reference the real target filename")
        # The tmpfile basename bash prints looks like 'tmp<random>.sh'; the fix
        # rewrites it to the real target, so no such token may survive. (We key
        # on the tmp basename, not '/tmp/', because pytest's tmp_path itself
        # lives under /tmp — the real target legitimately contains '/tmp/'.)
        assert not re.search(r"\btmp\w{6,}\.sh\b", res.content), (
            "the NamedTemporaryFile basename must be scrubbed from the detail")

    def test_reject_detail_no_tmp_suffix_token(self, tmp_path, monkeypatch):
        """Adjacent: a fabricated checker stderr full of the tmp path must have
        that path rewritten out before relaying."""
        import subprocess as _sub
        monkeypatch.setattr("shutil.which", lambda name, *a, **k: "/usr/bin/" + name)

        captured = {}

        def _fake_run(cmd, *a, **k):
            tmp = cmd[-1]  # the NamedTemporaryFile path argv
            captured["tmp"] = tmp
            return _sub.CompletedProcess(
                cmd, 1, "", f"{tmp}: line 1: syntax error near unexpected token")
        monkeypatch.setattr("subprocess.run", _fake_run)

        f = tmp_path / "leaky.sh"
        res = await_(_write(tmp_path, f, "echo hi\n"))
        assert res.is_error, "fake nonzero exit must reject"
        assert captured.get("tmp"), "fixture sanity: checker ran with a tmp path"
        assert captured["tmp"] not in res.content, (
            "the tmp path must be rewritten out of the relayed checker detail")


# ===========================================================================
# R15 (LOW) — search.py: display paths lexical; grep/glob consistent
# ===========================================================================

class TestR15DisplayPath:
    """Display paths must be computed lexically (os.path.relpath), never
    resolve()d through symlinks; grep and glob must spell the SAME file
    identically."""

    def test_symlink_display_not_resolved_absolute(self, tmp_path):
        """_display_path on a symlink-inside-workspace pointing OUTSIDE must
        render the in-workspace name lexically, not the resolved absolute
        external target."""
        outside_dir = Path(tempfile.mkdtemp())
        target = outside_dir / "secret.txt"
        target.write_text("s\n")
        link = tmp_path / "link.txt"
        os.symlink(target, link)

        rendered = search_mod._display_path(link, tmp_path)
        assert not os.path.isabs(rendered), (
            "a symlink inside the workspace must render as a workspace-relative "
            f"name, not a resolved absolute external path; got {rendered!r}")
        assert "link.txt" in rendered, (
            "the symlink must be shown at its in-workspace name")
        assert str(outside_dir) not in rendered, (
            "the resolved external target path must not appear in the display")

    def test_grep_and_glob_spell_nested_file_identically(self, tmp_path):
        """A normal nested file must render workspace-relative and identically
        in grep content mode and glob (lexical consistency, pairs with R7)."""
        sub = tmp_path / "pkg" / "mod"
        sub.mkdir(parents=True)
        (sub / "code.py").write_text("needle here\n")

        rg = await_(_grep(tmp_path, "needle", output_mode="content"))
        rl = await_(_glob(tmp_path, "**/*.py"))
        assert not rg.is_error and not rl.is_error
        # grep row is 'path:lineno: line'; the path token is before the first ':'.
        grep_path = rg.content.splitlines()[0].split(":")[0]
        glob_path = rl.content.splitlines()[0]
        assert grep_path == "pkg/mod/code.py", (
            f"grep must render the nested file workspace-relative; got {grep_path!r}")
        assert glob_path == "pkg/mod/code.py", (
            f"glob must render the same file identically; got {glob_path!r}")
        assert grep_path == glob_path, (
            "grep and glob must spell the same file with one consistent path")


# ===========================================================================
# R16 (LOW) — tools/__init__.py: missing workspace attr → loud error
# ===========================================================================

class TestR16WorkspaceRequired:
    """grep/glob with an agent_config lacking a workspace attr must fail loudly
    (is_error naming workspace), never silently root at the process CWD."""

    def test_grep_missing_workspace_errors(self, tmp_path):
        """A bare agent_config without .workspace → grep is_error mentioning
        workspace, not a CWD scan."""
        class NoWorkspace:  # deliberately no .workspace attribute
            pass

        res = await_(execute_tool(
            name="grep", input={"pattern": "zzz_absent_token"},
            tool_config={}, agent_config=NoWorkspace(), tools=None,
            callbacks={"call_id": "t"},
        ))
        assert res.is_error, (
            "grep with no workspace configuration must fail loudly, not scan CWD")
        assert "workspace" in res.content.lower(), (
            "the error must name the missing workspace configuration")

    def test_glob_missing_workspace_errors(self, tmp_path):
        """glob shares the requirement — missing workspace must also error."""
        class NoWorkspace:
            pass

        res = await_(execute_tool(
            name="glob", input={"pattern": "*"},
            tool_config={}, agent_config=NoWorkspace(), tools=None,
            callbacks={"call_id": "t"},
        ))
        assert res.is_error, "glob with no workspace configuration must fail loudly"
        assert "workspace" in res.content.lower(), (
            "the error must name the missing workspace configuration")
