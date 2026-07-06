"""RED suite — Re-audit fixes N1-N6 (adjudicated scope, see BRIEF.md).

Pins the fix semantics for the new findings in
tmp/code-audit/file-search-remediation/findings-opus-reaudit.md's
``## New findings`` section (N1-N6), as adjudicated in
tmp/f4/BRIEF.md — where the brief narrows or otherwise deviates from the
auditor's suggested fix, the brief wins and this suite follows the brief.

Every test here asserts the FIXED behavior, so each one that covers a code
change (N1, N2, N3, N4) fails RED against the pre-fix source and turns GREEN
once the corresponding fix lands. N5 and N6 are docstring-only (no behavior
change), so they have no dedicated pass/fail test; a light structural check
pins that the new invariant text exists in the docstring, which is the only
thing there is to pin for a comment-only change.

Helpers copied from the test_bundle2_remediation.py / test_search_tools.py
convention (no cross-test-module imports exist in this suite → copy per
convention; per the brief, this is the ONLY new test file and no existing
test file is ever modified).
"""

import asyncio
import logging
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import Usage
from openalph.tools import execute_tool, ToolResult
from openalph.tools import search as search_mod
from openalph.tools import validate as validate_mod
import openalph.provider as provider_mod


# ===========================================================================
# Shared helpers (copied real-path pattern; see module docstring)
# ===========================================================================

def _cfg(workspace, **kw):
    defaults = dict(
        name="test-reaudit",
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


def await_(coro):
    """Run a coroutine to completion on a fresh event loop (mirrors
    test_bundle2_remediation.py's convention for non-@pytest.mark.asyncio
    modules driving async tools synchronously)."""
    return asyncio.run(coro)


# ===========================================================================
# N1 (HIGH) — honest belt comments + loud degradation + main-thread pinning
# ===========================================================================

class TestN1LoudDegradation:
    """The SIGALRM guard must WARN (not silently no-op) when it expected to
    arm but could not (off-main-thread), and must NOT warn on the normal
    main-thread path. The asyncio.wait_for dispatch belt comments must no
    longer claim it bounds an await-free scan."""

    def test_off_main_thread_grep_logs_warning(self, tmp_path, caplog):
        """run_grep executed off the main thread must log the N1 warning
        (SIGALRM cannot arm there) and must still return a result (the
        between-unit monotonic checks remain as the degraded-mode bound)."""
        (tmp_path / "a.txt").write_text("needle line\n")
        result_box = {}

        def worker():
            async def _factory():
                return await search_mod.run_grep(
                    pattern="needle", path=None, glob=None,
                    output_mode="files_with_matches", head_limit=None,
                    case_insensitive=False, config={}, workspace=tmp_path,
                )
            loop = asyncio.new_event_loop()
            try:
                result_box["result"] = loop.run_until_complete(_factory())
            finally:
                loop.close()

        with caplog.at_level(logging.WARNING, logger="openalph.tools.search"):
            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=15.0)

        assert not t.is_alive(), "off-main-thread run_grep must still terminate"
        assert "result" in result_box, "off-main-thread run_grep must produce a result"
        assert not result_box["result"].is_error, result_box["result"].content

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("without the primary" in r.message.lower()
                   and "redos" in r.message.lower() for r in warnings), (
            "off-main-thread scan must log a warning that it is running "
            f"without the primary ReDoS bound; got: {[r.message for r in warnings]}")

    def test_main_thread_execute_tool_does_not_log_warning(self, tmp_path, caplog):
        """The normal dispatch path (execute_tool -> run_grep, main thread)
        must NOT emit the off-main-thread degradation warning."""
        (tmp_path / "a.txt").write_text("needle line\n")
        with caplog.at_level(logging.WARNING, logger="openalph.tools.search"):
            res = await_(_grep(tmp_path, "needle"))
        assert not res.is_error, res.content
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("without the primary" in r.message.lower()
                       for r in warnings), (
            "main-thread execute_tool(grep) dispatch must not log the "
            f"off-main-thread degradation warning; got: {[r.message for r in warnings]}")

    def test_main_thread_execute_tool_glob_does_not_log_warning(self, tmp_path, caplog):
        """Same as above for glob's dispatch path."""
        (tmp_path / "a.txt").write_text("x\n")
        with caplog.at_level(logging.WARNING, logger="openalph.tools.search"):
            res = await_(_glob(tmp_path, "*.txt"))
        assert not res.is_error, res.content
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("without the primary" in r.message.lower()
                       for r in warnings), (
            "main-thread execute_tool(glob) dispatch must not log the "
            f"off-main-thread degradation warning; got: {[r.message for r in warnings]}")

    def test_dispatch_belt_comment_no_longer_claims_defense_in_depth(self):
        """The tools/__init__.py grep/glob dispatch belt comments must state
        the truth (wait_for cannot cancel an await-free coroutine) instead of
        AFFIRMATIVELY claiming 'defense in depth' backup for the SIGALRM
        guard. The corrected comment is allowed to still contain the phrase
        while explicitly DENYING it (e.g. 'is NOT "defense in depth"') --
        what must be gone is the old unqualified claim that the belt IS a
        defense-in-depth layer, and the old 'wait_for is compatible here'
        assertion (both taken from the pre-fix source verbatim)."""
        import inspect
        import openalph.tools as tools_pkg
        source = inspect.getsource(tools_pkg)
        lower = source.lower()
        assert "enforcement (defense in depth" not in lower, (
            "the old affirmative claim '...enforcement (defense in depth' "
            "must be removed/corrected")
        assert "wait_for is compatible here" not in lower, (
            "the old 'asyncio.wait_for is compatible here' claim must be "
            "removed/corrected")
        assert "not a scan bound" in lower or "is not a scan bound" in lower, (
            "the corrected comment must state the belt is NOT a scan bound")


# ===========================================================================
# N2 (MEDIUM) — refuse symlink-as-root (narrow fix only)
# ===========================================================================

class TestN2SymlinkAsRoot:
    """grep/glob must refuse a symlinked search ROOT (dir or file) rather
    than follow it out of the workspace. A normal (non-symlink) root must be
    completely unaffected."""

    def test_grep_symlinked_dir_root_refused(self, tmp_path):
        outside_dir = Path(tempfile.mkdtemp())
        (outside_dir / "secret.txt").write_text("SECRET_NEEDLE outside workspace\n")
        linkdir = tmp_path / "linkdir"
        os.symlink(outside_dir, linkdir)

        res = await_(_grep(tmp_path, "SECRET_NEEDLE", path="linkdir"))
        assert res.is_error, (
            "grep with a symlinked directory as the search root must be refused")
        assert "symlink" in res.content.lower(), (
            f"error must name the symlink no-follow policy; got: {res.content!r}")
        assert "SECRET_NEEDLE" not in res.content, (
            "refusal must not leak the outside content into the error message")

    def test_glob_symlinked_dir_root_refused(self, tmp_path):
        outside_dir = Path(tempfile.mkdtemp())
        (outside_dir / "secret.txt").write_text("content\n")
        linkdir = tmp_path / "linkdir"
        os.symlink(outside_dir, linkdir)

        res = await_(_glob(tmp_path, "**/*.txt", path="linkdir"))
        assert res.is_error, (
            "glob with a symlinked directory as the search root must be refused")
        assert "symlink" in res.content.lower(), (
            f"error must name the symlink no-follow policy; got: {res.content!r}")
        assert "secret.txt" not in res.content

    def test_grep_symlinked_file_root_refused(self, tmp_path):
        outside_dir = Path(tempfile.mkdtemp())
        target = outside_dir / "secret.txt"
        target.write_text("SECRET_NEEDLE outside workspace\n")
        linkfile = tmp_path / "linkfile.txt"
        os.symlink(target, linkfile)

        res = await_(_grep(tmp_path, "SECRET_NEEDLE", path="linkfile.txt"))
        assert res.is_error, (
            "grep with a symlinked FILE as the search root must be refused "
            "(today's root.is_file() follows the symlink and reads through it)")
        assert "symlink" in res.content.lower(), (
            f"error must name the symlink no-follow policy; got: {res.content!r}")
        assert "SECRET_NEEDLE" not in res.content

    def test_grep_normal_dir_root_unaffected(self, tmp_path):
        """A completely ordinary (non-symlink) directory root must still work
        exactly as before — the narrow fix must not over-trigger."""
        sub = tmp_path / "normal_subdir"
        sub.mkdir()
        (sub / "f.txt").write_text("needle here\n")

        res = await_(_grep(tmp_path, "needle", path="normal_subdir"))
        assert not res.is_error, f"a normal directory root must not be refused: {res.content}"
        assert "f.txt" in res.content

    def test_glob_normal_dir_root_unaffected(self, tmp_path):
        sub = tmp_path / "normal_subdir2"
        sub.mkdir()
        (sub / "g.txt").write_text("x\n")

        res = await_(_glob(tmp_path, "*.txt", path="normal_subdir2"))
        assert not res.is_error, f"a normal directory root must not be refused: {res.content}"
        assert "g.txt" in res.content


# ===========================================================================
# N3 (MEDIUM) — atomic write must not sever symlinks
# ===========================================================================

class TestN3WriteThroughSymlink:
    """_write_now (and file_write end-to-end) must write THROUGH a symlinked
    target: the link stays a symlink, and the REAL target file receives the
    new content — never silently detached into a private regular file."""

    def test_write_through_symlink_updates_real_target(self, tmp_path):
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        real_file = real_dir / "target.txt"
        real_file.write_text("ORIGINAL\n")
        link = tmp_path / "link.txt"
        os.symlink(real_file, link)

        res = await_(_write(tmp_path, link, "NEW CONTENT\n"))
        assert not res.is_error, f"write through a symlink must succeed: {res.content}"

        assert os.path.islink(link), (
            "the symlink itself must survive the write (must not be replaced "
            "by a regular file)")
        assert os.path.realpath(link) == os.path.realpath(real_file), (
            "the symlink must still point at the same real target")
        assert real_file.read_text() == "NEW CONTENT\n", (
            "the REAL target file must receive the new content, not just the "
            "symlink's own (now-detached) copy")

    def test_write_through_symlink_leaves_no_orphan_tmp(self, tmp_path):
        real_dir = tmp_path / "real_dir2"
        real_dir.mkdir()
        real_file = real_dir / "target2.txt"
        real_file.write_text("ORIGINAL\n")
        link = tmp_path / "link2.txt"
        os.symlink(real_file, link)
        dir_before = set(os.listdir(real_dir))

        res = await_(_write(tmp_path, link, "NEW\n"))
        assert not res.is_error, res.content

        dir_after = set(os.listdir(real_dir))
        orphans = dir_after - dir_before - {"target2.txt"}
        assert not orphans, f"write-through-symlink must leave no orphan tmp files: {orphans}"

    def test_write_through_symlink_preserves_mode_on_real_target(self, tmp_path):
        import stat as stat_mod
        real_dir = tmp_path / "real_dir3"
        real_dir.mkdir()
        real_file = real_dir / "target3.txt"
        real_file.write_text("ORIGINAL\n")
        os.chmod(real_file, 0o640)
        link = tmp_path / "link3.txt"
        os.symlink(real_file, link)

        res = await_(_write(tmp_path, link, "NEW\n"))
        assert not res.is_error, res.content

        after_mode = stat_mod.S_IMODE(os.stat(real_file).st_mode)
        assert after_mode == 0o640, (
            f"the REAL target's prior mode (0o640) must be preserved; got 0o{after_mode:o}")

    def test_write_now_direct_symlink_target(self, tmp_path):
        """Unit-level check directly against _write_now (bypassing the tool
        dispatch layer) — the seam the brief names explicitly."""
        real_file = tmp_path / "direct_target.txt"
        real_file.write_text("ORIG\n")
        link = tmp_path / "direct_link.txt"
        os.symlink(real_file, link)

        validate_mod._write_now(str(link), "DIRECT NEW\n")

        assert os.path.islink(link), "_write_now must not replace the symlink itself"
        assert real_file.read_text() == "DIRECT NEW\n"

    def test_write_non_symlink_target_unaffected(self, tmp_path):
        """Sanity/no-regression: a plain (non-symlink) path must still write
        in place exactly as before."""
        f = tmp_path / "plain.txt"
        f.write_text("ORIG\n")
        res = await_(_write(tmp_path, f, "PLAIN NEW\n"))
        assert not res.is_error, res.content
        assert f.read_text() == "PLAIN NEW\n"
        assert not os.path.islink(f)


# ===========================================================================
# N4 (MEDIUM) — keepalive ping must never authorize a thinking budget
# ===========================================================================

def _mock_client(cache_read=150_000, cache_creation=4, in_tok=8, out_tok=1):
    msg = MagicMock()
    msg.usage.input_tokens = in_tok
    msg.usage.output_tokens = out_tok
    msg.usage.cache_read_input_tokens = cache_read
    msg.usage.cache_creation_input_tokens = cache_creation
    msg.content = []
    msg.model = "claude-sonnet-4-20250514"
    msg.stop_reason = "end_turn"
    msg.id = "msg_ping"
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=msg)
    return client


def _keepalive_cfg(tmp_path, **kw):
    defaults = dict(
        name="test-reaudit-ka",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[], subagent_cache_keepalive=True,
        )},
        workspace=tmp_path,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


@pytest.mark.asyncio
class TestN4KeepaliveBudgetThinkingSkip:
    """A keepalive ping must never silently authorize a real thinking budget
    on a budget-based (non-adaptive) thinking model. Adaptive-thinking models
    (max_tokens=1 verified live) must be byte-for-byte unaffected."""

    BUDGET_MODEL = "anthropic/claude-sonnet-4-20250514"   # not on the adaptive allowlist
    ADAPTIVE_MODEL = "anthropic/claude-opus-4-8"          # on the adaptive allowlist

    async def test_budget_thinking_model_no_api_call_and_warns(self, tmp_path, caplog):
        cfg = _keepalive_cfg(tmp_path)
        client = _mock_client()
        with patch.object(provider_mod, "_get_client", return_value=client), \
             caplog.at_level(logging.WARNING, logger="openalph.provider"):
            usage = await provider_mod.ping_cache(
                cfg, system="SYS", messages=[{"role": "user", "content": "hi"}],
                tools=None, cache_ttl="1h", model=self.BUDGET_MODEL,
                thinking_level="high",
            )
        client.messages.create.assert_not_awaited()
        assert any("keepalive" in r.message.lower() and "budget" in r.message.lower()
                   for r in caplog.records if r.levelno == logging.WARNING), (
            f"must warn about skipping the ping for a budget-thinking model; "
            f"got: {[r.message for r in caplog.records]}")
        # Reuses the existing miss-shaped signal so the caller's normal
        # miss-handling (log + on_miss notice + abort loop) applies.
        assert usage is not None, "must return a Usage (miss-shaped), not None"
        read = usage.cache_read_tokens or 0
        write = usage.cache_creation_tokens or 0
        from openalph.agent import _keepalive_is_hit
        assert not _keepalive_is_hit(read, write), (
            "the returned Usage must read as a MISS to the caller's existing "
            "_keepalive_is_hit check, so the keepalive loop aborts via its "
            "current miss-handling path")

    async def test_budget_thinking_off_still_pings_normally(self, tmp_path):
        """thinking_level='off' on a budget-model is the ordinary (pre-N4)
        path -- must still ping normally (N4 only gates thinking-ON budget
        models, never thinking-off)."""
        cfg = _keepalive_cfg(tmp_path)
        client = _mock_client()
        with patch.object(provider_mod, "_get_client", return_value=client):
            usage = await provider_mod.ping_cache(
                cfg, system="SYS", messages=[{"role": "user", "content": "hi"}],
                tools=None, cache_ttl="1h", model=self.BUDGET_MODEL,
                thinking_level="off",
            )
        client.messages.create.assert_awaited_once()
        kw = client.messages.create.call_args.kwargs
        assert kw["max_tokens"] == 1
        assert usage.cache_read_tokens == 150_000

    async def test_adaptive_model_unaffected_max_tokens_stays_one(self, tmp_path):
        """Existing behavior for adaptive-thinking models must be completely
        unchanged: the ping still fires, still max_tokens=1, still threads
        the adaptive+effort block through."""
        cfg = _keepalive_cfg(tmp_path)
        client = _mock_client()
        with patch.object(provider_mod, "_get_client", return_value=client):
            usage = await provider_mod.ping_cache(
                cfg, system="SYS", messages=[{"role": "user", "content": "hi"}],
                tools=None, cache_ttl="1h", model=self.ADAPTIVE_MODEL,
                thinking_level="max",
            )
        client.messages.create.assert_awaited_once()
        kw = client.messages.create.call_args.kwargs
        assert kw["max_tokens"] == 1, "adaptive path must stay cheap (max_tokens=1), unchanged"
        assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kw["output_config"]["effort"] == "max"
        assert usage.cache_read_tokens == 150_000

    async def test_adaptive_model_thinking_off_unaffected(self, tmp_path):
        cfg = _keepalive_cfg(tmp_path)
        client = _mock_client()
        with patch.object(provider_mod, "_get_client", return_value=client):
            await provider_mod.ping_cache(
                cfg, system="SYS", messages=[{"role": "user", "content": "hi"}],
                tools=None, cache_ttl="1h", model=self.ADAPTIVE_MODEL,
                thinking_level="off",
            )
        client.messages.create.assert_awaited_once()
        kw = client.messages.create.call_args.kwargs
        assert kw["max_tokens"] == 1
        assert "thinking" not in kw

    async def test_non_anthropic_still_noop_unaffected(self, tmp_path):
        """N4's new gate must not disturb the pre-existing non-Anthropic
        no-op short-circuit (checked before the new gate can even run)."""
        cfg = _keepalive_cfg(
            tmp_path,
            providers={"openai": ProviderConfig(
                key="openai", type="openai", api_key="sk-test",
                base_url="http://local", quirks=[],
            )},
        )
        with patch.object(provider_mod, "_get_client") as gc:
            usage = await provider_mod.ping_cache(
                cfg, system="SYS", messages=[{"role": "user", "content": "hi"}],
                tools=None, cache_ttl="1h", model="openai/gpt-x",
                thinking_level="high",
            )
        assert usage is None
        gc.assert_not_called()


# ===========================================================================
# N5 (LOW) — guard invariant comment (docstring-only, no code change)
# ===========================================================================

class TestN5GuardInvariantDocstring:
    """_time_budget_guard's docstring must explicitly document the
    await-free / main-thread / non-reentrant correctness precondition."""

    def test_docstring_states_await_free_precondition(self):
        doc = (search_mod._time_budget_guard.__doc__ or "").lower()
        assert "await" in doc, (
            "docstring must mention the await-free precondition explicitly")
        assert "main-thread" in doc or "main thread" in doc, (
            "docstring must mention the main-thread precondition explicitly")


# ===========================================================================
# N6 (LOW) — docstring scope (docstring-only, no code change)
# ===========================================================================

class TestN6WriteNowDocstringScope:
    """_write_now's docstring must scope the atomicity claim to
    reader-visibility and explicitly carve out cross-crash rename durability
    (parent-dir fsync) as out of scope."""

    def test_docstring_scopes_atomicity_to_reader_visibility(self):
        raw_doc = validate_mod._write_now.__doc__ or ""
        doc = raw_doc.lower()
        # Normalize whitespace (docstring text wraps across lines) so a
        # phrase split by a line break is still matched as one phrase.
        doc_flat = " ".join(doc.split())
        assert "reader" in doc, (
            "docstring must scope the atomicity guarantee to reader-visibility")
        assert "durab" in doc, (
            "docstring must discuss (and disclaim) cross-crash durability")
        assert "out of scope" in doc_flat or "out-of-scope" in doc_flat, (
            "docstring must explicitly say cross-crash rename durability is "
            f"out of scope; got: {raw_doc!r}")


# ===========================================================================
# Zero-regression spot checks (cheap sanity — full suite is the real gate)
# ===========================================================================

class TestNoRegressionSpotChecks:
    """A couple of fast direct checks that the R1/R7 mechanisms this bundle
    touches still behave as before for the plain (non-symlink, main-thread)
    case. The full existing suite (test_bundle2_remediation.py,
    test_cache_keepalive.py, test_search_tools.py) is the authoritative
    regression gate and is run unmodified alongside this file.
    """

    def test_grep_zero_match_hint_unaffected(self, tmp_path):
        (tmp_path / "f.txt").write_text("nothing here\n")
        res = await_(_grep(tmp_path, "NOPE_NOT_PRESENT"))
        assert not res.is_error
        assert "No matches for pattern" in res.content

    def test_glob_star_listing_unaffected(self, tmp_path):
        (tmp_path / "x.txt").write_text("x\n")
        res = await_(_glob(tmp_path, "*.txt"))
        assert not res.is_error
        assert "x.txt" in res.content
