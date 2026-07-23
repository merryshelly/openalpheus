"""RED suite — Component D (grep) + Component E (glob).

Bundle-2 (workspace-kdsn.195), design §6/§7 + §10-bullet-3, invariant V3.

grep and glob are NOT registered today, so ``execute_tool`` returns
``ToolResult("Unknown tool: grep ...", is_error=True)`` — an honest runtime
return, never a collection/import error.  Every success-expecting assertion
fails red; every error-path assertion pins the SPECIFIC steering/format text
from design §6/§7 so today's generic "Unknown tool" message cannot spuriously
satisfy a bare is_error check.

Helpers copied from the test_guidance_integration.py convention (no
cross-test-module imports exist in this suite).
"""

import os
from pathlib import Path

import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.tools import execute_tool, ToolResult, BUILTIN_TOOLS


# Exact overflow steering text (design §6, V3).  Em-dash is literal.
OVERFLOW = ('[truncated: showing first {n} of {m} — '
            'narrow with path/glob, or use output_mode="count"]')


def _cfg(workspace, **kw):
    defaults = dict(
        name="test-search",
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


async def _grep(tmp_path, pattern, tool_config=None, **kw):
    cfg = _cfg(tmp_path)
    inp = {"pattern": pattern}
    inp.update(kw)
    return await execute_tool(
        name="grep", input=inp, tool_config=tool_config or {},
        agent_config=cfg, tools=None, callbacks={"call_id": "tc"},
    )


async def _glob(tmp_path, pattern, tool_config=None, **kw):
    cfg = _cfg(tmp_path)
    inp = {"pattern": pattern}
    inp.update(kw)
    return await execute_tool(
        name="glob", input=inp, tool_config=tool_config or {},
        agent_config=cfg, tools=None, callbacks={"call_id": "tc"},
    )


def _mk_files(root, n, pattern_line="needle here", prefix="f"):
    """Create n text files each containing pattern_line; return their paths."""
    paths = []
    for i in range(n):
        p = root / f"{prefix}{i:03d}.txt"
        p.write_text(pattern_line + "\n")
        paths.append(p)
    return paths


# ===========================================================================
# Schema / registry contract
# ===========================================================================

class TestSearchSchemas:
    def test_grep_registered(self):
        assert "grep" in BUILTIN_TOOLS, "grep must be a registered builtin"

    def test_glob_registered(self):
        assert "glob" in BUILTIN_TOOLS, "glob must be a registered builtin"

    def test_grep_schema_shape(self):
        assert "grep" in BUILTIN_TOOLS, "grep not registered"
        p = BUILTIN_TOOLS["grep"]["parameters"]
        props = p.get("properties", {})
        assert props.get("pattern", {}).get("type") == "string"
        assert props.get("path", {}).get("type") == "string"
        assert props.get("glob", {}).get("type") == "string"
        assert props.get("output_mode", {}).get("type") == "string"
        assert props.get("head_limit", {}).get("type") == "integer"
        assert props.get("case_insensitive", {}).get("type") == "boolean"
        assert p.get("required", []) == ["pattern"], "only pattern is required"

    def test_grep_config_keys(self):
        assert "grep" in BUILTIN_TOOLS, "grep not registered"
        cfg = BUILTIN_TOOLS["grep"]["config"]
        assert "max_scan_files" in cfg, "grep config must expose max_scan_files"
        assert "max_file_bytes" in cfg, "grep config must expose max_file_bytes"

    def test_glob_schema_shape(self):
        assert "glob" in BUILTIN_TOOLS, "glob not registered"
        p = BUILTIN_TOOLS["glob"]["parameters"]
        props = p.get("properties", {})
        assert props.get("pattern", {}).get("type") == "string"
        assert props.get("path", {}).get("type") == "string"
        assert props.get("head_limit", {}).get("type") == "integer"
        assert p.get("required", []) == ["pattern"], "only pattern is required"

    def test_glob_config_keys(self):
        assert "glob" in BUILTIN_TOOLS, "glob not registered"
        cfg = BUILTIN_TOOLS["glob"]["config"]
        assert "max_scan_files" in cfg
        assert "max_file_bytes" in cfg


# ===========================================================================
# grep — output modes + defaults
# ===========================================================================

class TestGrepOutputModes:
    @pytest.mark.asyncio
    async def test_default_mode_is_files_with_matches(self, tmp_path):
        """Default output_mode = files_with_matches → returns filenames only (§6)."""
        (tmp_path / "hit.txt").write_text("the needle is here\n")
        (tmp_path / "miss.txt").write_text("nothing relevant\n")
        res = await _grep(tmp_path, "needle")
        assert not res.is_error, f"grep must succeed: {res.content}"
        assert "hit.txt" in res.content, "matching file must be listed"
        assert "miss.txt" not in res.content, "non-matching file must be absent"
        # files mode returns names, NOT 'path:lineno: line' content rows
        assert ":1:" not in res.content, "files_with_matches must not emit line rows"

    @pytest.mark.asyncio
    async def test_content_mode_line_format(self, tmp_path):
        """content mode: 'path:lineno: line' rows (design §6)."""
        (tmp_path / "c.txt").write_text("first line\nsecond needle line\n")
        res = await _grep(tmp_path, "needle", output_mode="content")
        assert not res.is_error, res.content
        assert "c.txt:2: second needle line" in res.content, \
            "content mode must render 'path:lineno: line' with correct lineno"

    @pytest.mark.asyncio
    async def test_content_mode_500_char_line_truncation(self, tmp_path):
        """content mode truncates each rendered line at 500 chars (§6)."""
        (tmp_path / "long.txt").write_text("A" * 1000 + "needle\n")
        res = await _grep(tmp_path, "needle", output_mode="content")
        assert not res.is_error, res.content
        assert res.content.count("A") <= 500, \
            "content-mode line must be truncated at 500 chars"

    @pytest.mark.asyncio
    async def test_count_mode_format(self, tmp_path):
        """count mode: 'path: N' per file + 'total: M' (design §6)."""
        (tmp_path / "a.txt").write_text("needle\nneedle\n")
        res = await _grep(tmp_path, "needle", output_mode="count")
        assert not res.is_error, res.content
        assert "a.txt: 2" in res.content, "count mode must report per-file count"
        assert "total: 2" in res.content.lower() or "total:2" in res.content.lower(), \
            "count mode must report a total"


class TestGrepHeadLimit:
    @pytest.mark.asyncio
    async def test_files_default_head_limit_50(self, tmp_path):
        """files_with_matches default head_limit = 50 (design §6)."""
        _mk_files(tmp_path, 55)
        res = await _grep(tmp_path, "needle")
        assert not res.is_error, res.content
        assert OVERFLOW.format(n=50, m=55) in res.content, \
            "files mode must cap at 50 and emit exact overflow steering"

    @pytest.mark.asyncio
    async def test_content_default_head_limit_100(self, tmp_path):
        """content default head_limit = 100 (design §6)."""
        (tmp_path / "many.txt").write_text("".join("needle\n" for _ in range(105)))
        res = await _grep(tmp_path, "needle", output_mode="content")
        assert not res.is_error, res.content
        assert OVERFLOW.format(n=100, m=105) in res.content, \
            "content mode must cap at 100 and emit exact overflow steering"

    @pytest.mark.asyncio
    async def test_count_default_head_limit_50(self, tmp_path):
        """count default head_limit = 50 (design §6)."""
        _mk_files(tmp_path, 55)
        res = await _grep(tmp_path, "needle", output_mode="count")
        assert not res.is_error, res.content
        assert OVERFLOW.format(n=50, m=55) in res.content, \
            "count mode must cap at 50 and emit exact overflow steering"

    @pytest.mark.asyncio
    async def test_explicit_head_limit_override(self, tmp_path):
        """Explicit head_limit overrides the default + emits exact steering (§6)."""
        _mk_files(tmp_path, 5)
        res = await _grep(tmp_path, "needle", head_limit=2)
        assert not res.is_error, res.content
        assert OVERFLOW.format(n=2, m=5) in res.content, \
            "explicit head_limit must cap and emit exact overflow steering"


class TestGrepOrderingAndBounds:
    @pytest.mark.asyncio
    async def test_mtime_desc_path_asc_ordering(self, tmp_path):
        """Deterministic order: mtime desc, path asc tiebreak (V3, design §6)."""
        old = tmp_path / "old.txt"; old.write_text("needle\n")
        new = tmp_path / "new.txt"; new.write_text("needle\n")
        # old has an older mtime than new
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        res = await _grep(tmp_path, "needle")
        assert not res.is_error, res.content
        assert res.content.index("new.txt") < res.content.index("old.txt"), \
            "newer file (mtime desc) must be listed before older"

    @pytest.mark.asyncio
    async def test_path_asc_tiebreak_same_mtime(self, tmp_path):
        """Equal mtime → path ascending tiebreak (V3)."""
        a = tmp_path / "aaa.txt"; a.write_text("needle\n")
        b = tmp_path / "bbb.txt"; b.write_text("needle\n")
        os.utime(a, (5_000_000, 5_000_000))
        os.utime(b, (5_000_000, 5_000_000))
        res = await _grep(tmp_path, "needle")
        assert not res.is_error, res.content
        assert res.content.index("aaa.txt") < res.content.index("bbb.txt"), \
            "equal-mtime files must be path-ascending"

    @pytest.mark.asyncio
    async def test_skip_list_pruning(self, tmp_path):
        """Skip-list dirs are pruned (design §6 walker)."""
        skip_dirs = [".git", "node_modules", "__pycache__", ".venv",
                     "venv", ".memory-index", ".cache"]
        for d in skip_dirs:
            sub = tmp_path / d
            sub.mkdir()
            (sub / "buried.txt").write_text("needle\n")
        (tmp_path / "keep.txt").write_text("needle\n")
        res = await _grep(tmp_path, "needle")
        assert not res.is_error, res.content
        assert "keep.txt" in res.content, "normal file must be found"
        for d in skip_dirs:
            assert d not in res.content, f"skip-list dir {d} must be pruned from results"

    @pytest.mark.asyncio
    async def test_binary_and_oversize_skipped_with_footer(self, tmp_path):
        """Binary + >5MB files skipped; footer reports skip count (§6, V3)."""
        (tmp_path / "text.txt").write_text("needle\n")
        (tmp_path / "blob.bin").write_bytes(b"\x00\x01needle\x00\x02")
        big = tmp_path / "big.txt"
        big.write_text("needle\n" + ("A" * (5 * 1024 * 1024 + 100)))
        res = await _grep(tmp_path, "needle")
        assert not res.is_error, res.content
        assert "text.txt" in res.content, "the small text file must match"
        assert "blob.bin" not in res.content, "binary file must be skipped"
        assert "big.txt" not in res.content, ">5MB file must be skipped"
        assert "skip" in res.content.lower(), \
            "footer must report skipped-file count when >0"

    @pytest.mark.asyncio
    async def test_max_scan_files_bound(self, tmp_path):
        """max_scan_files bound honored; footer suggests narrowing (§6, V3)."""
        _mk_files(tmp_path, 8)
        res = await _grep(tmp_path, "needle",
                          tool_config={"max_scan_files": 3})
        assert not res.is_error, res.content
        assert "narrow" in res.content.lower(), \
            "hitting max_scan_files must add a narrowing hint in the footer"

    @pytest.mark.asyncio
    async def test_case_insensitive_flag(self, tmp_path):
        """case_insensitive=true matches across case; default false does not (§6)."""
        (tmp_path / "cap.txt").write_text("Hello World\n")
        res_ci = await _grep(tmp_path, "hello", case_insensitive=True)
        assert not res_ci.is_error, res_ci.content
        assert "cap.txt" in res_ci.content, "case_insensitive=true must match 'Hello'"
        res_cs = await _grep(tmp_path, "hello")
        assert not res_cs.is_error, res_cs.content
        assert "No matches for pattern" in res_cs.content, \
            "case-sensitive default must NOT match 'Hello' → zero-match hint"


class TestGrepErrorsAndHints:
    @pytest.mark.asyncio
    async def test_invalid_regex_error(self, tmp_path):
        """Invalid regex → is_error with re.error text + Python-re steering (§6)."""
        (tmp_path / "x.txt").write_text("data\n")
        res = await _grep(tmp_path, "[unterminated")
        assert res.is_error, "invalid regex must be an error"
        assert "pattern uses Python re syntax" in res.content, \
            "invalid-regex error must include the Python-re steering text"

    @pytest.mark.asyncio
    async def test_zero_match_is_hint_not_error(self, tmp_path):
        """Zero matches → hint, NOT an error (design §6)."""
        (tmp_path / "x.txt").write_text("unrelated content\n")
        res = await _grep(tmp_path, "absent_pattern_xyz")
        assert not res.is_error, "zero matches must NOT be an error"
        assert "No matches for pattern 'absent_pattern_xyz'" in res.content, \
            "zero-match hint must name the pattern (design §6 text)"

    @pytest.mark.asyncio
    async def test_default_root_is_workspace(self, tmp_path):
        """With no path arg, grep defaults to the workspace root (§6)."""
        (tmp_path / "ws.txt").write_text("needle\n")
        res = await _grep(tmp_path, "needle")
        assert not res.is_error, res.content
        assert "ws.txt" in res.content, "default root must be the workspace"


# ===========================================================================
# glob — Component E
# ===========================================================================

class TestGlob:
    @pytest.mark.asyncio
    async def test_double_star_recursion(self, tmp_path):
        """glob '**' recurses into subdirectories (design §7)."""
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "deep.py").write_text("x\n")
        (tmp_path / "top.py").write_text("y\n")
        res = await _glob(tmp_path, "**/*.py")
        assert not res.is_error, res.content
        assert "deep.py" in res.content, "** must recurse to nested files"
        assert "top.py" in res.content, "** must also include top-level matches"

    @pytest.mark.asyncio
    async def test_directories_rendered_with_trailing_slash(self, tmp_path):
        """glob renders directories with a trailing '/' (ls replacement, §7)."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("x\n")
        res = await _glob(tmp_path, "*")
        assert not res.is_error, res.content
        assert "subdir/" in res.content, "directory must render with trailing slash"

    @pytest.mark.asyncio
    async def test_star_as_directory_listing(self, tmp_path):
        """glob '*' lists directory entries (the ls replacement, §7)."""
        (tmp_path / "one.txt").write_text("x\n")
        (tmp_path / "two.txt").write_text("y\n")
        res = await _glob(tmp_path, "*")
        assert not res.is_error, res.content
        assert "one.txt" in res.content and "two.txt" in res.content, \
            "glob '*' must list directory entries"

    @pytest.mark.asyncio
    async def test_mtime_desc_sort(self, tmp_path):
        """glob sorts mtime desc, path asc tiebreak (§7, V3)."""
        old = tmp_path / "old.md"; old.write_text("x\n")
        new = tmp_path / "new.md"; new.write_text("y\n")
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        res = await _glob(tmp_path, "*.md")
        assert not res.is_error, res.content
        assert res.content.index("new.md") < res.content.index("old.md"), \
            "glob must sort newest-first (mtime desc)"

    @pytest.mark.asyncio
    async def test_head_limit_and_steering(self, tmp_path):
        """glob head_limit caps results + emits exact overflow steering (§7)."""
        for i in range(5):
            (tmp_path / f"g{i}.txt").write_text("x\n")
        res = await _glob(tmp_path, "*.txt", head_limit=2)
        assert not res.is_error, res.content
        assert OVERFLOW.format(n=2, m=5) in res.content, \
            "glob overflow must use the same exact steering text as grep"

    @pytest.mark.asyncio
    async def test_nonexistent_root_error(self, tmp_path):
        """glob with a nonexistent root → error (design §7)."""
        res = await _glob(tmp_path, "*.txt", path="does_not_exist_dir")
        assert res.is_error, "nonexistent root must error"
        assert ("not found" in res.content.lower()
                or "does not exist" in res.content.lower()
                or "no such" in res.content.lower()), \
            "nonexistent-root error must indicate the root is missing (not 'Unknown tool')"

    @pytest.mark.asyncio
    async def test_zero_match_is_hint_not_error(self, tmp_path):
        """glob zero matches → hint, NOT an error (design §7)."""
        (tmp_path / "present.txt").write_text("x\n")
        res = await _glob(tmp_path, "*.nomatch")
        assert not res.is_error, "glob zero matches must NOT be an error"
        assert "no" in res.content.lower() and "match" in res.content.lower(), \
            "glob zero-match must return a hint result"

    @pytest.mark.asyncio
    async def test_default_root_is_workspace(self, tmp_path):
        """glob with no path defaults to the workspace root (§7)."""
        (tmp_path / "root.txt").write_text("x\n")
        res = await _glob(tmp_path, "*.txt")
        assert not res.is_error, res.content
        assert "root.txt" in res.content, "default root must be the workspace"
