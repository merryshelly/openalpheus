"""New-file mode regression pin — workspace-kdsn.357.

``_write_now`` (the single atomic-write seam behind file_write / file_edit /
file_patch) creates its content in a ``tempfile.NamedTemporaryFile``, which
lands at 0o600. The seam re-applied a mode only when the target ALREADY
existed, so every file an agent CREATED was owner-only — regardless of the
service unit's ``UMask=0027`` (which gives 0o640 for a normal
``open(path, "w")``).

Live consequence: agent-written files that another Unix user has to read
failed with EACCES. It broke the fleet's off-site backup job, which runs as
oa-merry and reads files written by oa-saw under /srv/openalph/shared/briefs
(23 of 67 briefs were 0o600, unreadable to the backup job).

Fix under test: when there is no prior mode (new file), the tmp is chmod'd
to ``0o666 & ~umask`` — the mode a normal program's write would produce.
The umask is READ from /proc/self/status, never via the ``os.umask(0)``
idiom, which would set the process umask to 0 for the duration of the call
(any file created in that window by another thread or library would land
world-writable).
"""

import os
import stat

import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.tools import execute_tool, validate


def _cfg(workspace, **kw):
    defaults = dict(
        name="test-new-file-mode",
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
    """file_write via execute_tool, read-guard disabled to isolate the write."""
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


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


class TestNewFileMode:
    @pytest.mark.asyncio
    async def test_new_file_honors_umask_0027(self, tmp_path, monkeypatch):
        """The fleet's unit umask (UMask=0027) → a new file lands 0o640. [RED]"""
        monkeypatch.setattr(validate, "_process_umask", lambda: 0o027)
        target = tmp_path / "fresh.txt"
        res = await _write(tmp_path, target, "hello\n")
        assert not res.is_error, res.content
        assert _mode(target) == 0o640, (
            "a new file must land 0o666 & ~umask, not the atomic write's "
            f"tempfile default of 0o600 (got {oct(_mode(target))})"
        )

    @pytest.mark.asyncio
    async def test_new_file_tracks_umask_not_a_hardcoded_mode(self, tmp_path, monkeypatch):
        """The mode follows the umask; 0o077 legitimately yields 0o600. [pin]

        Paired with a 0o027 file so this test DISCRIMINATES the fix: the 0o077
        case alone also passes against the pre-fix code (a tempfile default of
        0o600 is indistinguishable from "umask 0o077 applied"). The pair fails
        for a hardcoded 0o600 or a hardcoded 0o640.

        0o077 is also the case where "follow the umask" legitimately produces
        an owner-only file — so a future "the backup can't read my files"
        report is a umask question, not automatically a regression here.
        """
        monkeypatch.setattr(validate, "_process_umask", lambda: 0o077)
        private = tmp_path / "private.txt"
        res = await _write(tmp_path, private, "hi\n")
        assert not res.is_error, res.content
        assert _mode(private) == 0o600

        monkeypatch.setattr(validate, "_process_umask", lambda: 0o027)
        fleet = tmp_path / "fleet.txt"
        res = await _write(tmp_path, fleet, "hi\n")
        assert not res.is_error, res.content
        assert _mode(fleet) == 0o640, \
            "same seam, different umask — a hardcoded mode fails this pair"

    @pytest.mark.asyncio
    async def test_world_bits_clamped_under_a_shell_umask(self, tmp_path, monkeypatch):
        """umask 0o022 (shell/CLI launch) → 0o640, never world-readable. [RED]"""
        monkeypatch.setattr(validate, "_process_umask", lambda: 0o022)
        target = tmp_path / "shell.txt"
        res = await _write(tmp_path, target, "hi\n")
        assert not res.is_error, res.content
        assert _mode(target) == 0o640
        assert not (_mode(target) & stat.S_IROTH), "world read must stay off"

    @pytest.mark.asyncio
    async def test_umask_zero_cannot_yield_a_world_writable_file(self, tmp_path, monkeypatch):
        """umask 0o000 → 0o660: group-writable at worst, never world. [RED]"""
        monkeypatch.setattr(validate, "_process_umask", lambda: 0o000)
        target = tmp_path / "open.txt"
        res = await _write(tmp_path, target, "hi\n")
        assert not res.is_error, res.content
        assert _mode(target) == 0o660
        assert not (_mode(target) & (stat.S_IROTH | stat.S_IWOTH))

    @pytest.mark.asyncio
    async def test_new_file_group_readable_for_cross_user_readers(self, tmp_path, monkeypatch):
        """The actual failure mode: group read must be set (0o640 ⊃ g+r). [RED]"""
        monkeypatch.setattr(validate, "_process_umask", lambda: 0o027)
        target = tmp_path / "cross-user.txt"
        await _write(tmp_path, target, "readable by the openalph group\n")
        assert _mode(target) & stat.S_IRGRP, \
            "group read is the permission the fleet backup job depends on"

    @pytest.mark.asyncio
    async def test_existing_file_mode_still_preserved(self, tmp_path, monkeypatch):
        """Overwriting an existing file keeps ITS mode — unchanged behavior. [pin]"""
        monkeypatch.setattr(validate, "_process_umask", lambda: 0o027)
        target = tmp_path / "existing.txt"
        target.write_text("old\n")
        os.chmod(target, 0o444)
        res = await _write(tmp_path, target, "new\n")
        assert not res.is_error, res.content
        assert _mode(target) == 0o444, \
            "prior-mode preservation must survive the new-file fix"
        assert target.read_text() == "new\n"

    @pytest.mark.asyncio
    async def test_new_file_under_a_symlinked_missing_target(self, tmp_path, monkeypatch):
        """A dangling symlink target is a NEW file → umask path, link intact. [RED]"""
        monkeypatch.setattr(validate, "_process_umask", lambda: 0o027)
        real = tmp_path / "real.txt"
        link = tmp_path / "link.txt"
        link.symlink_to(real)
        res = await _write(tmp_path, link, "via link\n")
        assert not res.is_error, res.content
        assert link.is_symlink(), "the symlink must survive the write"
        assert real.read_text() == "via link\n"
        assert _mode(real) == 0o640

    def test_process_umask_reports_the_live_umask_without_mutating_it(self):
        """_process_umask() is a pure read of the real process umask. [pin]

        The os.umask() idiom appears HERE only (a single-threaded test that
        restores immediately) to learn the expected value. The production
        helper must not use it: os.umask(0) sets the process umask to 0 for
        the duration of the call.
        """
        before = os.umask(0o077)
        os.umask(before)

        assert validate._process_umask() == before

        after = os.umask(0o077)
        os.umask(after)
        assert after == before, "_process_umask() must not mutate the process umask"

    @pytest.mark.asyncio
    async def test_unreadable_proc_umask_falls_back_and_still_writes(self, tmp_path, monkeypatch):
        """No /proc → degrade to 0o640 + warn; never refuse the write. [RED]"""
        monkeypatch.setattr(validate, "_read_proc_umask", lambda: None)
        # Reset the module-global warn flag: pytest runs the whole suite in one
        # process, so without this the fallback would silently disarm the
        # warn-once assertion in any test that runs after it.
        monkeypatch.setattr(validate, "_UMASK_FALLBACK_WARNED", False)
        target = tmp_path / "fallback.txt"
        res = await _write(tmp_path, target, "ok\n")
        assert not res.is_error, \
            "an unresolvable umask must never block a write"
        # 0o666 & ~(0o022 fallback | 0o007) — the same 0o640 the fleet's real
        # UMask=0027 produces, so a degraded environment stays fleet-consistent.
        assert _mode(target) == 0o640


class TestProcUmaskParsing:
    """The /proc parse is the fix's only untrusted input — pin its refusals."""

    def test_parses_a_real_proc_field(self):
        assert validate._parse_proc_umask("Name:\tpython\nUmask:\t0027\n") == 0o027
        assert validate._parse_proc_umask("Umask:\t077\n") == 0o077

    @pytest.mark.parametrize("text", [
        "",
        "Name:\tpython\n",              # field absent
        "Umask:\n",                      # value-less
        "Umask:\tgarbage\n",             # not octal
        "Umask:\t-1\n",                  # int("-1", 8) == -1 → 0o666 & ~-1 == 0o000 lockout
        "Umask:\t1777\n",                # out of range
        "Umask: 0027 extra\n",           # malformed field
        "Umask:\t+027\n",                # int(x, 8) would accept the sign
        "Umask:\t0_2_7\n",               # int(x, 8) would accept underscores
        "Umask:\t٠٢٧\n",                 # int(x, 8) would accept non-ASCII digits
    ])
    def test_refuses_untrustworthy_values(self, text):
        assert validate._parse_proc_umask(text) is None

    def test_name_field_cannot_spoof_a_umask_line(self):
        """/proc's first field is the comm-controlled Name: — a lookalike line
        must not be mistaken for the real Umask: field. [pin]"""
        text = "Name:\tUmask:\t0000\nUmask:\t0027\n"
        assert validate._parse_proc_umask(text) == 0o027

    def test_fallback_warns_once_not_per_write(self, monkeypatch, caplog):
        """A degraded environment must not log on every single write. [pin]"""
        monkeypatch.setattr(validate, "_read_proc_umask", lambda: None)
        monkeypatch.setattr(validate, "_UMASK_FALLBACK_WARNED", False)
        with caplog.at_level("WARNING"):
            assert validate._process_umask() == 0o022
            assert validate._process_umask() == 0o022
        hits = [r for r in caplog.records if "umask" in r.getMessage().lower()]
        assert len(hits) == 1, f"expected exactly one warning, got {len(hits)}"

    @pytest.mark.asyncio
    async def test_target_vanishing_before_the_write_is_treated_as_new(self, tmp_path, monkeypatch):
        """A target deleted between stat and write lands the new-file mode. [RED]"""
        monkeypatch.setattr(validate, "_process_umask", lambda: 0o027)
        real_stat = os.stat
        target = tmp_path / "gone.txt"
        target.write_text("original\n")

        def _vanish(path, *a, **k):
            if str(path) == str(target):
                raise FileNotFoundError(path)
            return real_stat(path, *a, **k)

        monkeypatch.setattr(validate.os, "stat", _vanish)
        try:
            res = await _write(tmp_path, target, "fresh\n")
        finally:
            monkeypatch.undo()

        assert not res.is_error, res.content
        assert target.read_text() == "fresh\n"
        assert _mode(target) == 0o640, \
            "a vanished target is a new file — it must not fail the write"
