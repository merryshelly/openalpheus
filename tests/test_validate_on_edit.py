"""RED suite — Component C (validate-on-edit).

Bundle-2 (workspace-kdsn.195), design §5 + §10-bullet-2, invariant V2.

Validation is a NEW subsystem that does not exist yet.  Today file_write /
file_edit accept syntactically-broken content unconditionally and file_patch
is unregistered ("Unknown tool").  These tests therefore go RED on the
net-new reject/warn/fail-open behavior while a handful of cells pin
existing pass-through / fail-open invariants that MUST remain green (V2:
"an agent can always fix a broken file").

Monkeypatch targets are stdlib seams the future ``tools/validate.py`` will
call — ``shutil.which`` (subprocess-checker gating) and ``subprocess.run``
(the subprocess checker itself).  We never import ``tools.validate`` (it
does not exist → would be a collection error, FORBIDDEN); all behavior is
driven through the existing ``execute_tool`` dispatch.

Helpers copied from the test_guidance_integration.py convention (no
cross-test-module imports exist in this suite).
"""

import glob
import os
import subprocess
from pathlib import Path

import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.tools import execute_tool, ToolResult


VALID_PY = "x = 1\n"
BROKEN_PY = "x = = 1\n"          # SyntaxError under compile()
BROKEN_PY_2 = "y = = 2\n"        # different broken content
VALID_JSON = '{"a": 1}\n'
BROKEN_JSON = '{bad json\n'
VALID_TOML = "a = 1\n"
BROKEN_TOML = "a = = 1\n"


def _cfg(workspace, **kw):
    defaults = dict(
        name="test-validate",
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


async def _write(tmp_path, path, content, tool_config=None):
    """file_write via execute_tool, guard disabled to isolate validation."""
    cfg = _cfg(tmp_path)
    tc = {"require_read_before_write": False}
    if tool_config:
        tc.update(tool_config)
    return await execute_tool(
        name="file_write",
        input={"path": str(path), "content": content},
        tool_config=tc,
        agent_config=cfg,
        tools=None,
        callbacks={"call_id": "tc", "read_registry": {}},
    )


async def _edit(tmp_path, path, old, new, tool_config=None):
    cfg = _cfg(tmp_path)
    return await execute_tool(
        name="file_edit",
        input={"path": str(path), "old_text": old, "new_text": new},
        tool_config=tool_config or {},
        agent_config=cfg,
        tools=None,
        callbacks={"call_id": "tc", "read_registry": {}},
    )


async def _patch(tmp_path, path, search, replace, tool_config=None):
    cfg = _cfg(tmp_path)
    patch_text = f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"
    return await execute_tool(
        name="file_patch",
        input={"path": str(path), "patch": patch_text},
        tool_config=tool_config or {},
        agent_config=cfg,
        tools=None,
        callbacks={"call_id": "tc", "read_registry": {}},
    )


# ===========================================================================
# Checker registry — in-process (py / json / toml)
# ===========================================================================

class TestInProcessCheckers:
    @pytest.mark.asyncio
    async def test_py_broken_new_file_rejected(self, tmp_path):
        """.py checker (compile()) rejects broken new content (design §5)."""
        f = tmp_path / "s.py"
        res = await _write(tmp_path, f, BROKEN_PY)
        assert res.is_error, "broken .py must be rejected by compile() checker"
        assert not f.exists() or f.read_text() != BROKEN_PY, \
            "rejected broken .py must not be written to disk (V1/V2)"

    @pytest.mark.asyncio
    async def test_json_broken_new_file_rejected(self, tmp_path):
        """.json checker (json.loads) rejects broken new content."""
        f = tmp_path / "s.json"
        res = await _write(tmp_path, f, BROKEN_JSON)
        assert res.is_error, "broken .json must be rejected by json.loads checker"

    @pytest.mark.asyncio
    async def test_toml_broken_new_file_rejected(self, tmp_path):
        """.toml checker (tomllib.loads) rejects broken new content."""
        f = tmp_path / "s.toml"
        res = await _write(tmp_path, f, BROKEN_TOML)
        assert res.is_error, "broken .toml must be rejected by tomllib.loads checker"

    @pytest.mark.asyncio
    async def test_py_checker_creates_no_pycache(self, tmp_path):
        """.py checker uses compile(), NOT import — no __pycache__ side effect (§5)."""
        f = tmp_path / "clean.py"
        res = await _write(tmp_path, f, VALID_PY)
        assert not res.is_error, f"valid .py must write: {res.content}"
        pycache = list(tmp_path.rglob("__pycache__"))
        assert not pycache, \
            f"py validation must not create __pycache__ (compile, not import): {pycache}"

    @pytest.mark.asyncio
    async def test_unknown_extension_skips_silently(self, tmp_path):
        """Unknown extension → no checker → plain write, even if not valid code (§5)."""
        f = tmp_path / "notes.xyz"
        res = await _write(tmp_path, f, BROKEN_PY)  # python-invalid, but .xyz has no checker
        assert not res.is_error, "unknown extension must skip validation and write"
        assert f.read_text() == BROKEN_PY, "content must be written verbatim"


# ===========================================================================
# Subprocess checkers — bash -n / node --check, gated on shutil.which
# ===========================================================================

class TestSubprocessCheckers:
    @pytest.mark.asyncio
    async def test_bash_present_broken_rejected(self, tmp_path, monkeypatch):
        """bash on PATH → broken .sh rejected (design §5, subprocess checker)."""
        monkeypatch.setattr("shutil.which", lambda name, *a, **k: "/usr/bin/" + name)

        def _fake_run(*a, **k):
            return subprocess.CompletedProcess(a[0] if a else "bash", 1, "", "syntax error")
        monkeypatch.setattr("subprocess.run", _fake_run)

        f = tmp_path / "broken.sh"
        res = await _write(tmp_path, f, "fi\n")  # nonsense; fake_run says returncode 1
        assert res.is_error, "broken .sh must be rejected when bash is on PATH"
        assert "reject" in res.content.lower(), \
            "rejection message must say the edit was rejected (design §5 text)"

    @pytest.mark.asyncio
    async def test_bash_absent_fail_open(self, tmp_path, monkeypatch):
        """bash NOT on PATH → checker skipped, write proceeds (V4/V2 fail-open)."""
        monkeypatch.setattr("shutil.which", lambda name, *a, **k: None)
        f = tmp_path / "noscript.sh"
        res = await _write(tmp_path, f, "echo hi\n")
        assert not res.is_error, \
            "with bash absent, .sh write must proceed (best-effort gating, V4)"
        assert f.read_text() == "echo hi\n"

    @pytest.mark.asyncio
    async def test_node_present_broken_rejected(self, tmp_path, monkeypatch):
        """node on PATH → broken .js rejected (design §5)."""
        monkeypatch.setattr("shutil.which", lambda name, *a, **k: "/usr/bin/" + name)

        def _fake_run(*a, **k):
            return subprocess.CompletedProcess(a[0] if a else "node", 1, "", "SyntaxError")
        monkeypatch.setattr("subprocess.run", _fake_run)

        f = tmp_path / "broken.js"
        res = await _write(tmp_path, f, "function (\n")
        assert res.is_error, "broken .js must be rejected when node is on PATH"
        assert "reject" in res.content.lower()

    @pytest.mark.asyncio
    async def test_node_absent_fail_open(self, tmp_path, monkeypatch):
        """node NOT on PATH → checker skipped, write proceeds (V4/V2 fail-open)."""
        monkeypatch.setattr("shutil.which", lambda name, *a, **k: None)
        f = tmp_path / "app.js"
        res = await _write(tmp_path, f, "const x = 1;\n")
        assert not res.is_error, "with node absent, .js write must proceed (V4)"
        assert f.read_text() == "const x = 1;\n"


# ===========================================================================
# Fail-open on checker infrastructure failure (timeout / crash)
# ===========================================================================

class TestFailOpen:
    @pytest.mark.asyncio
    async def test_checker_timeout_fail_open(self, tmp_path, monkeypatch):
        """Checker timeout → fail-open write + skip note (design §5 pt 6)."""
        monkeypatch.setattr("shutil.which", lambda name, *a, **k: "/usr/bin/" + name)

        def _hang(*a, **k):
            raise subprocess.TimeoutExpired(cmd="bash", timeout=5)
        monkeypatch.setattr("subprocess.run", _hang)

        f = tmp_path / "slow.sh"
        res = await _write(tmp_path, f, "echo ok\n")
        assert not res.is_error, "checker timeout must fail-open (write proceeds), not error"
        assert "validation skipped" in res.content.lower(), \
            "timeout fail-open must append '(validation skipped: checker unavailable)'"

    @pytest.mark.asyncio
    async def test_checker_crash_fail_open(self, tmp_path, monkeypatch):
        """Checker crash → fail-open write + skip note (design §5 pt 6)."""
        monkeypatch.setattr("shutil.which", lambda name, *a, **k: "/usr/bin/" + name)

        def _boom(*a, **k):
            raise RuntimeError("checker exploded")
        monkeypatch.setattr("subprocess.run", _boom)

        f = tmp_path / "crash.sh"
        res = await _write(tmp_path, f, "echo ok\n")
        assert not res.is_error, "checker crash must fail-open (write proceeds), not error"
        assert "validation skipped" in res.content.lower(), \
            "crash fail-open must append '(validation skipped: checker unavailable)'"


# ===========================================================================
# Four-cell pre_ok × post_ok matrix — file_write
# ===========================================================================

class TestMatrixFileWrite:
    @pytest.mark.asyncio
    async def test_pre_ok_post_ok_write_proceeds(self, tmp_path):
        """(pre_ok, post_ok) → write proceeds unadorned. [existing-behavior pin]"""
        f = tmp_path / "cw_aa.py"
        f.write_text(VALID_PY)
        res = await _write(tmp_path, f, "x = 2\n")
        assert not res.is_error, res.content
        assert f.read_text() == "x = 2\n"

    @pytest.mark.asyncio
    async def test_not_pre_ok_post_ok_write_proceeds(self, tmp_path):
        """(¬pre_ok, post_ok) → write proceeds (fixing a broken file, V2). [pin]"""
        f = tmp_path / "cw_ba.py"
        f.write_text(BROKEN_PY)
        res = await _write(tmp_path, f, VALID_PY)
        assert not res.is_error, \
            f"fixing a broken file must always be allowed (V2): {res.content}"
        assert f.read_text() == VALID_PY

    @pytest.mark.asyncio
    async def test_pre_ok_not_post_ok_rejected_unchanged(self, tmp_path):
        """(pre_ok, ¬post_ok) → NO write, byte-identical, is_error, 'file unchanged'. [RED]"""
        f = tmp_path / "cw_ab.py"
        f.write_text(VALID_PY)
        before = f.read_bytes()
        res = await _write(tmp_path, f, BROKEN_PY)
        assert res.is_error, "clean→broken write must be rejected"
        assert f.read_bytes() == before, "V1: rejected write leaves file byte-identical"
        assert "unchanged" in res.content.lower(), \
            "reject message must use 'file unchanged' language (design §5)"

    @pytest.mark.asyncio
    async def test_not_pre_ok_not_post_ok_write_with_warning(self, tmp_path):
        """(¬pre_ok, ¬post_ok) → write proceeds + pre-existing warning. [RED on warning]"""
        f = tmp_path / "cw_bb.py"
        f.write_text(BROKEN_PY)
        res = await _write(tmp_path, f, BROKEN_PY_2)
        assert not res.is_error, "broken→broken must still write (V2)"
        assert "pre-existing" in res.content.lower(), \
            "broken→broken must append the pre-existing-errors warning (design §5)"


# ===========================================================================
# Four-cell matrix — file_edit
# ===========================================================================

class TestMatrixFileEdit:
    @pytest.mark.asyncio
    async def test_pre_ok_post_ok_write_proceeds(self, tmp_path):
        """(pre_ok, post_ok) → edit proceeds. [existing-behavior pin]"""
        f = tmp_path / "ce_aa.py"
        f.write_text(VALID_PY)
        res = await _edit(tmp_path, f, "1", "2")
        assert not res.is_error, res.content
        assert f.read_text() == "x = 2\n"

    @pytest.mark.asyncio
    async def test_not_pre_ok_post_ok_write_proceeds(self, tmp_path):
        """(¬pre_ok, post_ok) → edit that fixes a broken file proceeds (V2). [pin]"""
        f = tmp_path / "ce_ba.py"
        f.write_text(BROKEN_PY)  # "x = = 1\n"
        res = await _edit(tmp_path, f, "= = ", "= ")
        assert not res.is_error, \
            f"edit that repairs a broken file must be allowed (V2): {res.content}"
        assert f.read_text() == VALID_PY

    @pytest.mark.asyncio
    async def test_pre_ok_not_post_ok_rejected_unchanged(self, tmp_path):
        """(pre_ok, ¬post_ok) → NO write, byte-identical, is_error, 'file unchanged'. [RED]"""
        f = tmp_path / "ce_ab.py"
        f.write_text(VALID_PY)
        before = f.read_bytes()
        res = await _edit(tmp_path, f, "1", "= 2")  # "x = = 2\n" broken
        assert res.is_error, "edit that breaks a clean file must be rejected"
        assert f.read_bytes() == before, "V1: rejected edit leaves file byte-identical"
        assert "unchanged" in res.content.lower(), \
            "reject message must use 'file unchanged' language"

    @pytest.mark.asyncio
    async def test_not_pre_ok_not_post_ok_write_with_warning(self, tmp_path):
        """(¬pre_ok, ¬post_ok) → edit proceeds + pre-existing warning. [RED on warning]"""
        f = tmp_path / "ce_bb.py"
        f.write_text(BROKEN_PY)  # "x = = 1\n"
        res = await _edit(tmp_path, f, "1", "2")  # "x = = 2\n" still broken
        assert not res.is_error, "broken→broken edit must still write (V2)"
        assert "pre-existing" in res.content.lower(), \
            "broken→broken edit must append the pre-existing-errors warning"


# ===========================================================================
# Four-cell matrix — file_patch (all RED: tool unregistered today)
# ===========================================================================

class TestMatrixFilePatch:
    @pytest.mark.asyncio
    async def test_pre_ok_post_ok_write_proceeds(self, tmp_path):
        """(pre_ok, post_ok) → patch proceeds. [RED — file_patch unimplemented]"""
        f = tmp_path / "cp_aa.py"
        f.write_text(VALID_PY)
        res = await _patch(tmp_path, f, "1", "2")
        assert not res.is_error, f"valid→valid patch must proceed: {res.content}"
        assert f.read_text() == "x = 2\n"

    @pytest.mark.asyncio
    async def test_not_pre_ok_post_ok_write_proceeds(self, tmp_path):
        """(¬pre_ok, post_ok) → patch that repairs a broken file proceeds (V2). [RED]"""
        f = tmp_path / "cp_ba.py"
        f.write_text(BROKEN_PY)
        res = await _patch(tmp_path, f, "= = ", "= ")
        assert not res.is_error, f"repairing patch must proceed (V2): {res.content}"
        assert f.read_text() == VALID_PY

    @pytest.mark.asyncio
    async def test_pre_ok_not_post_ok_rejected_unchanged(self, tmp_path):
        """(pre_ok, ¬post_ok) → NO write, byte-identical, is_error, 'file unchanged'. [RED]"""
        f = tmp_path / "cp_ab.py"
        f.write_text(VALID_PY)
        before = f.read_bytes()
        res = await _patch(tmp_path, f, "1", "= 2")  # would yield "x = = 2\n" broken
        assert res.is_error, "clean→broken patch must be rejected"
        assert f.read_bytes() == before, "V1: rejected patch leaves file byte-identical"
        assert "unchanged" in res.content.lower(), \
            "reject message must use 'file unchanged' language (not 'Unknown tool')"

    @pytest.mark.asyncio
    async def test_not_pre_ok_not_post_ok_write_with_warning(self, tmp_path):
        """(¬pre_ok, ¬post_ok) → patch proceeds + pre-existing warning. [RED]"""
        f = tmp_path / "cp_bb.py"
        f.write_text(BROKEN_PY)
        res = await _patch(tmp_path, f, "1", "2")  # "x = = 2\n" still broken
        assert not res.is_error, "broken→broken patch must still write (V2)"
        assert "pre-existing" in res.content.lower(), \
            "broken→broken patch must append the pre-existing-errors warning"


# ===========================================================================
# Kill-switch, tempfile hygiene, output cap
# ===========================================================================

class TestKillSwitchAndHygiene:
    @pytest.mark.asyncio
    async def test_kill_switch_disables_validation(self, tmp_path):
        """validate_on_edit=false → plain write of broken content (design §5 pt 1). [pin]"""
        f = tmp_path / "kill.py"
        f.write_text(VALID_PY)
        res = await _write(tmp_path, f, BROKEN_PY,
                           tool_config={"validate_on_edit": False})
        assert not res.is_error, "with validate_on_edit=false, broken write must proceed"
        assert f.read_text() == BROKEN_PY, "kill-switch write must land verbatim"

    @pytest.mark.asyncio
    async def test_no_orphan_tempfiles_after_subprocess_checker(self, tmp_path):
        """Subprocess checker cleans up its NamedTemporaryFile (design §5 pt 4). [RED-driven]

        Uses the REAL bash on PATH.  Snapshots the system temp dir for
        ``tmp*.sh`` NamedTemporaryFile leftovers before/after and asserts the
        checker leaves no net-new orphan (cleaned up in ``finally``).
        """
        import tempfile
        tmpdir = Path(tempfile.gettempdir())
        before = set(glob.glob(str(tmpdir / "tmp*.sh")))

        f = tmp_path / "hyg.sh"
        # bash IS on PATH in this environment (real subprocess); broken script.
        res = await _write(tmp_path, f, "if then\n")
        assert res.is_error, \
            "broken .sh must be rejected (drives the subprocess checker path)"

        after = set(glob.glob(str(tmpdir / "tmp*.sh")))
        orphans = after - before
        assert not orphans, \
            f"subprocess checker must leave no orphan tempfiles: {orphans}"

    @pytest.mark.asyncio
    async def test_relayed_checker_output_capped_2000(self, tmp_path, monkeypatch):
        """Rejected-checker output relayed to the model is capped at 2000 chars (§5). [RED]"""
        monkeypatch.setattr("shutil.which", lambda name, *a, **k: "/usr/bin/" + name)

        huge = "X" * 5000

        def _fake_run(*a, **k):
            return subprocess.CompletedProcess(a[0] if a else "bash", 1, "", huge)
        monkeypatch.setattr("subprocess.run", _fake_run)

        f = tmp_path / "cap.sh"  # new file → pre_ok vacuously True → clean→broken reject
        res = await _write(tmp_path, f, "echo hi\n")
        assert res.is_error, "broken .sh must be rejected (drives the cap path)"
        assert res.content.count("X") <= 2000, \
            f"relayed checker output must be capped at 2000 chars; got {res.content.count('X')}"
