"""RED suite — Component A (file_patch) + Component B (file_edit replace_all).

Bundle-2 (workspace-kdsn.195), design §3/§4/§10-bullet-1, invariants V1/V7.

These tests SPECIFY unimplemented behavior.  file_patch is not yet a
BUILTIN_TOOLS entry, so ``execute_tool(name="file_patch", ...)`` today
returns ``ToolResult(content="Unknown tool: file_patch ...", is_error=True)``
— an honest runtime return, NOT an import/collection error.  Every
success-expecting assertion therefore fails red today; every error-path
assertion pins the SPECIFIC steering text so that today's generic
"Unknown tool" message cannot spuriously satisfy a bare is_error check.

Helpers are COPIED (not imported) from the canonical real-path pattern
in test_guidance_integration.py — cross-test-module imports are not used
anywhere in this suite, so copying follows the existing convention and
avoids modifying an existing test file (FORBIDDEN).
"""

from pathlib import Path

import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.tools import execute_tool, BUILTIN_TOOLS


# --- Helpers (copied from test_guidance_integration.py convention) ---------

def _cfg(workspace, **kw):
    """Build a real AgentConfig for tests."""
    defaults = dict(
        name="test-filepatch",
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


def _block(search, replace, fence=7):
    """Build a single SEARCH/REPLACE hunk (both sections non-empty)."""
    o = "<" * fence
    d = "=" * fence
    c = ">" * fence
    return f"{o} SEARCH\n{search}\n{d}\n{replace}\n{c} REPLACE"


async def _patch(tmp_path, path, patch_text, tool_config=None, registry=None):
    """Invoke file_patch through the real execute_tool dispatch."""
    cfg = _cfg(tmp_path)
    callbacks = {"call_id": "tc-fp"}
    if registry is not None:
        callbacks["read_registry"] = registry
    return await execute_tool(
        name="file_patch",
        input={"path": str(path), "patch": patch_text},
        tool_config=tool_config or {},
        agent_config=cfg,
        tools=None,
        callbacks=callbacks,
    )


# ===========================================================================
# Schema / registry contract
# ===========================================================================

class TestFilePatchSchema:
    def test_file_patch_registered_as_builtin(self):
        """file_patch must be a registered builtin tool (design §3)."""
        assert "file_patch" in BUILTIN_TOOLS, \
            "file_patch must be registered in BUILTIN_TOOLS"

    def test_file_patch_schema_shape(self):
        """Schema: path:str required, patch:str required (design §3)."""
        assert "file_patch" in BUILTIN_TOOLS, "file_patch not registered"
        params = BUILTIN_TOOLS["file_patch"]["parameters"]
        props = params.get("properties", {})
        assert props.get("path", {}).get("type") == "string"
        assert props.get("patch", {}).get("type") == "string"
        assert set(params.get("required", [])) == {"path", "patch"}


# ===========================================================================
# Parser
# ===========================================================================

class TestFilePatchParser:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("width", [5, 6, 7, 8, 9])
    async def test_fence_width_variants(self, tmp_path, width):
        """Fence keywords accept 5–9 marker chars (design §3 regexes)."""
        f = tmp_path / "w.txt"
        f.write_text("alpha\n")
        res = await _patch(tmp_path, f, _block("alpha", "beta", fence=width))
        assert not res.is_error, f"width={width} must parse+apply: {res.content}"
        assert f.read_text() == "beta\n"

    @pytest.mark.asyncio
    async def test_fence_trailing_whitespace_tolerated(self, tmp_path):
        """Trailing whitespace on fence lines is tolerated (\\s*$ anchors)."""
        f = tmp_path / "tw.txt"
        f.write_text("one\n")
        patch = "<<<<<<< SEARCH   \none\n=======  \ntwo\n>>>>>>> REPLACE  "
        res = await _patch(tmp_path, f, patch)
        assert not res.is_error, f"trailing ws must be tolerated: {res.content}"
        assert f.read_text() == "two\n"

    @pytest.mark.asyncio
    async def test_prose_outside_blocks_ignored(self, tmp_path):
        """Prose outside SEARCH/REPLACE blocks is ignored (weak-model tolerance)."""
        f = tmp_path / "p.txt"
        f.write_text("keep\ntarget\n")
        patch = (
            "Sure, here is the change you asked for:\n"
            + _block("target", "changed")
            + "\nLet me know if you need anything else!"
        )
        res = await _patch(tmp_path, f, patch)
        assert not res.is_error, f"prose must be ignored: {res.content}"
        assert f.read_text() == "keep\nchanged\n"

    @pytest.mark.asyncio
    async def test_zero_blocks_error_echoes_worked_example(self, tmp_path):
        """Zero parseable blocks → error echoing the expected fence format."""
        f = tmp_path / "z.txt"
        f.write_text("data\n")
        res = await _patch(tmp_path, f, "this text has no fences at all")
        assert res.is_error, "no parseable block must error"
        assert "<<<<<<< SEARCH" in res.content, \
            "zero-block error must echo the SEARCH fence as a worked example"
        assert ">>>>>>> REPLACE" in res.content, \
            "zero-block error must echo the REPLACE fence as a worked example"

    @pytest.mark.asyncio
    async def test_empty_search_steers_to_file_write(self, tmp_path):
        """Empty SEARCH → exact steering error to file_write (design §3)."""
        f = tmp_path / "e.txt"
        f.write_text("stuff\n")
        patch = "<<<<<<< SEARCH\n=======\nnew content\n>>>>>>> REPLACE"
        res = await _patch(tmp_path, f, patch)
        assert res.is_error, "empty SEARCH must error"
        assert "SEARCH must not be empty — use file_write to create new files." \
            in res.content, "empty-SEARCH error text must match design §3 exactly"

    @pytest.mark.asyncio
    async def test_empty_replace_is_deletion(self, tmp_path):
        """Empty REPLACE = deletion of the SEARCH text (valid, design §3)."""
        f = tmp_path / "d.txt"
        f.write_text("keep1\ndelete_me\nkeep2\n")
        patch = "<<<<<<< SEARCH\ndelete_me\n=======\n>>>>>>> REPLACE"
        res = await _patch(tmp_path, f, patch)
        assert not res.is_error, f"empty REPLACE must delete, not error: {res.content}"
        out = f.read_text()
        assert "delete_me" not in out, "SEARCH text must be deleted"
        assert "keep1" in out and "keep2" in out, "surrounding text preserved"


# ===========================================================================
# Multi-hunk semantics
# ===========================================================================

class TestFilePatchSemantics:
    @pytest.mark.asyncio
    async def test_multi_hunk_sequential_apply(self, tmp_path):
        """Multiple hunks apply sequentially (design §3)."""
        f = tmp_path / "m.txt"
        f.write_text("aaa\nbbb\nccc\n")
        patch = _block("aaa", "AAA") + "\n" + _block("ccc", "CCC")
        res = await _patch(tmp_path, f, patch)
        assert not res.is_error, f"both hunks must apply: {res.content}"
        assert f.read_text() == "AAA\nbbb\nCCC\n"

    @pytest.mark.asyncio
    async def test_hunk_n_matches_text_created_by_hunk_n_minus_1(self, tmp_path):
        """Hunk N may match text produced by hunk N−1 (evolving buffer, §3)."""
        f = tmp_path / "chain.txt"
        f.write_text("original\n")
        # hunk1: original -> midway ; hunk2: midway (created by hunk1) -> final
        patch = _block("original", "midway") + "\n" + _block("midway", "final")
        res = await _patch(tmp_path, f, patch)
        assert not res.is_error, \
            f"hunk 2 must match hunk 1's output (sequential buffer): {res.content}"
        assert f.read_text() == "final\n"

    @pytest.mark.asyncio
    async def test_hunk_zero_match_names_index_and_steers(self, tmp_path):
        """A hunk with 0 matches → error naming hunk index + read-first steering."""
        f = tmp_path / "zm.txt"
        f.write_text("present\n")
        patch = _block("absent_text", "whatever")
        res = await _patch(tmp_path, f, patch)
        assert res.is_error, "0-match hunk must error"
        assert "hunk" in res.content.lower(), "error must name the hunk"
        assert "read the file" in res.content.lower(), \
            "0-match hunk error must steer 'read the file first'"

    @pytest.mark.asyncio
    async def test_hunk_multi_match_names_index_and_count(self, tmp_path):
        """A hunk matching >1 time → error naming hunk index + count + context steer."""
        f = tmp_path / "mm.txt"
        f.write_text("dup\ndup\n")
        patch = _block("dup", "solo")
        res = await _patch(tmp_path, f, patch)
        assert res.is_error, ">1-match hunk must error"
        assert "hunk" in res.content.lower(), "error must name the hunk"
        assert "context" in res.content.lower(), \
            ">1-match hunk error must steer 'add surrounding context'"

    @pytest.mark.asyncio
    async def test_mid_patch_failure_atomicity_v1(self, tmp_path):
        """V1: a failed multi-hunk patch leaves the file byte-identical (no partial)."""
        f = tmp_path / "atomic.txt"
        original = "AAA\nBBB\nCCC\n"
        f.write_text(original)
        # hunk1 (AAA->XXX) would succeed; hunk2 (NOPE) has 0 matches → whole patch fails
        patch = _block("AAA", "XXX") + "\n" + _block("NOPE_not_present", "ZZZ")
        res = await _patch(tmp_path, f, patch)
        assert res.is_error, "patch with a failing hunk must error"
        assert f.read_text() == original, \
            "V1 atomicity: file must be byte-identical after a failed patch"
        assert "hunk" in res.content.lower() and "2" in res.content, \
            "error must identify the failing hunk (index 2)"

    @pytest.mark.asyncio
    async def test_success_message_reports_hunk_count(self, tmp_path):
        """Success result reports the applied hunk count (design §3)."""
        f = tmp_path / "count.txt"
        f.write_text("p\nq\nr\n")
        patch = _block("p", "P") + "\n" + _block("r", "R")
        res = await _patch(tmp_path, f, patch)
        assert not res.is_error, res.content
        assert "Applied 2 hunk" in res.content, \
            "success message must report hunk count ('Applied N hunk(s)')"


# ===========================================================================
# Consistency with existing file tools (V7)
# ===========================================================================

class TestFilePatchConsistency:
    @pytest.mark.asyncio
    async def test_binary_file_refused(self, tmp_path):
        """Binary target refused like file_edit (V7 / anchors §7 trap #4)."""
        f = tmp_path / "bin.dat"
        f.write_bytes(b"\x00\x01\x02\x00binary\x00content\x00")
        patch = _block("binary", "text")
        res = await _patch(tmp_path, f, patch)
        assert res.is_error, "binary file must be refused"
        assert "binary" in res.content.lower(), \
            "binary refusal must mention binary"

    @pytest.mark.asyncio
    async def test_missing_file_error(self, tmp_path):
        """Missing target → error (existing-file-only, design §3)."""
        missing = tmp_path / "nope.txt"
        res = await _patch(tmp_path, missing, _block("x", "y"))
        assert res.is_error, "missing file must error"
        assert "not found" in res.content.lower() or "no such" in res.content.lower(), \
            "missing-file error must indicate the file was not found"

    @pytest.mark.asyncio
    async def test_workspace_relative_path_resolution(self, tmp_path):
        """Relative path resolves against workspace (anchors §7 trap #2)."""
        f = tmp_path / "rel.txt"
        f.write_text("hello\n")
        cfg = _cfg(tmp_path)
        res = await execute_tool(
            name="file_patch",
            input={"path": "rel.txt", "patch": _block("hello", "world")},
            tool_config={},
            agent_config=cfg,
            tools=None,
            callbacks={"call_id": "tc"},
        )
        assert not res.is_error, f"relative path must resolve to workspace: {res.content}"
        assert f.read_text() == "world\n", \
            "relative 'rel.txt' must resolve to <workspace>/rel.txt"

    @pytest.mark.asyncio
    async def test_read_registry_mtime_refresh_on_success(self, tmp_path):
        """Successful patch refreshes read-registry mtime (V6 / anchors §7 trap #2,#3)."""
        f = tmp_path / "reg.txt"
        f.write_text("before\n")
        registry: dict = {}
        res = await _patch(tmp_path, f, _block("before", "after"), registry=registry)
        assert not res.is_error, res.content
        resolved = str(Path(str(f)).resolve())
        assert resolved in registry, \
            "successful file_patch must refresh the read-registry mtime"


# ===========================================================================
# Component B — file_edit replace_all (design §4)
# ===========================================================================

async def _edit(tmp_path, path, old, new, **extra):
    cfg = _cfg(tmp_path)
    inp = {"path": str(path), "old_text": old, "new_text": new}
    inp.update(extra)
    return await execute_tool(
        name="file_edit", input=inp, tool_config={},
        agent_config=cfg, tools=None, callbacks={"call_id": "tc", "read_registry": {}},
    )


class TestFileEditReplaceAll:
    @pytest.mark.asyncio
    async def test_default_false_single_match_unchanged(self, tmp_path):
        """replace_all default false: exactly-one match still replaces (unchanged)."""
        f = tmp_path / "one.txt"
        f.write_text("solo occurrence here\n")
        res = await _edit(tmp_path, f, "solo", "SOLO")
        assert not res.is_error, res.content
        assert f.read_text() == "SOLO occurrence here\n"

    @pytest.mark.asyncio
    async def test_default_false_zero_match_error_unchanged(self, tmp_path):
        """replace_all default false: 0 matches → steering error (unchanged)."""
        f = tmp_path / "zero.txt"
        f.write_text("content\n")
        res = await _edit(tmp_path, f, "absent", "x")
        assert res.is_error, "0 matches must error"
        assert "read the file first" in res.content.lower()

    @pytest.mark.asyncio
    async def test_default_false_multi_match_ambiguous_unchanged(self, tmp_path):
        """replace_all default false: >1 matches → ambiguous error (unchanged)."""
        f = tmp_path / "multi.txt"
        f.write_text("x x x\n")
        res = await _edit(tmp_path, f, "x", "y")
        assert res.is_error, ">1 match without replace_all must error"
        assert f.read_text() == "x x x\n", "file must be unchanged on ambiguous error"

    @pytest.mark.asyncio
    async def test_replace_all_true_zero_match_error(self, tmp_path):
        """replace_all=true with 0 matches → same steering error (design §4)."""
        f = tmp_path / "ra0.txt"
        f.write_text("nothing to match\n")
        res = await _edit(tmp_path, f, "absent_token", "x", replace_all=True)
        assert res.is_error, "replace_all=true with 0 matches must error"
        assert "read the file first" in res.content.lower(), \
            "0-match error must retain read-first steering"

    @pytest.mark.asyncio
    async def test_replace_all_true_n_matches_all_replaced_with_count(self, tmp_path):
        """replace_all=true with N matches → all replaced + count in message (§4)."""
        f = tmp_path / "raN.txt"
        f.write_text("foo foo foo bar foo\n")
        res = await _edit(tmp_path, f, "foo", "baz", replace_all=True)
        assert not res.is_error, f"replace_all=true must succeed on N matches: {res.content}"
        assert f.read_text() == "baz baz baz bar baz\n", \
            "all occurrences must be replaced"
        assert "Replaced 4 occurrence" in res.content, \
            "success message must report the replacement count"

    def test_file_edit_schema_gains_replace_all(self):
        """file_edit schema gains replace_all:bool optional (design §4)."""
        props = BUILTIN_TOOLS["file_edit"]["parameters"]["properties"]
        assert "replace_all" in props, "file_edit must expose replace_all"
        assert props["replace_all"].get("type") == "boolean"
        assert "replace_all" not in BUILTIN_TOOLS["file_edit"]["parameters"].get("required", []), \
            "replace_all must be optional"
