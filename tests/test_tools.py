"""Tests for tool registry: discovery, schema generation, result truncation.

Interface contract:
    discover_tools(workspace: Path) -> list[ToolDef]
    tool_schemas(tools: list[ToolDef]) -> list[dict]
    truncate_result(text: str, max_chars: int) -> str

ToolDef: name (str), description (str), parameters (dict), config (dict)
ToolResult: content (str), is_error (bool, default False)
ToolError: raised for configuration/discovery errors
"""

import pytest
from pathlib import Path
from openalph.tools import (
    discover_tools,
    tool_schemas,
    truncate_result,
    ToolDef,
    ToolResult,
    ToolError,
)


# --- Data classes ---


class TestToolDef:

    def test_has_required_fields(self):
        td = ToolDef(
            name="shell",
            description="Execute a shell command",
            parameters={"type": "object", "properties": {}},
            config={"default_timeout": 30},
        )
        assert td.name == "shell"
        assert td.description == "Execute a shell command"
        assert td.parameters["type"] == "object"
        assert td.config["default_timeout"] == 30


class TestToolResult:

    def test_defaults_not_error(self):
        tr = ToolResult(content="hello")
        assert tr.content == "hello"
        assert tr.is_error is False

    def test_error_result(self):
        tr = ToolResult(content="file not found", is_error=True)
        assert tr.is_error is True


class TestToolError:

    def test_is_exception(self):
        with pytest.raises(ToolError):
            raise ToolError("bad config")


# --- Discovery ---


class TestDiscoverTools:

    def test_discovers_tools_from_toml_files(self, tmp_path):
        """Each .toml file in workspace/tools/ becomes a ToolDef."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\ndefault_timeout = 30\n")
        (tools_dir / "file_read.toml").write_text("[config]\n")

        tools = discover_tools(tmp_path)
        names = {t.name for t in tools}
        assert "shell" in names
        assert "file_read" in names

    def test_missing_tools_directory_returns_empty(self, tmp_path):
        """No tools/ directory → empty list, not an error."""
        tools = discover_tools(tmp_path)
        assert tools == []

    def test_empty_tools_directory_returns_empty(self, tmp_path):
        """tools/ exists but is empty → empty list."""
        (tmp_path / "tools").mkdir()
        tools = discover_tools(tmp_path)
        assert tools == []

    def test_unknown_tool_name_raises_error(self, tmp_path):
        """TOML file with unrecognized name → ToolError."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "not_a_real_tool.toml").write_text("[config]\n")

        with pytest.raises(ToolError):
            discover_tools(tmp_path)

    def test_invalid_toml_raises_error(self, tmp_path):
        """Malformed TOML → ToolError."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("this is not valid toml [[[")

        with pytest.raises(ToolError):
            discover_tools(tmp_path)

    def test_toml_config_merged_into_tooldef(self, tmp_path):
        """Config values from TOML are accessible on the ToolDef."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text(
            "[config]\ndefault_timeout = 60\nmax_output = 100000\n"
        )

        tools = discover_tools(tmp_path)
        shell = next(t for t in tools if t.name == "shell")
        assert shell.config["default_timeout"] == 60
        assert shell.config["max_output"] == 100000

    def test_tooldef_has_description_and_parameters(self, tmp_path):
        """ToolDef includes description and JSON Schema parameters from builtin definition."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        tools = discover_tools(tmp_path)
        shell = next(t for t in tools if t.name == "shell")
        assert isinstance(shell.description, str)
        assert len(shell.description) > 0
        assert isinstance(shell.parameters, dict)
        assert "properties" in shell.parameters

    def test_non_toml_files_ignored(self, tmp_path):
        """Non-.toml files in tools/ are silently ignored."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")
        (tools_dir / "README.md").write_text("# Notes\n")
        (tools_dir / ".gitkeep").write_text("")

        tools = discover_tools(tmp_path)
        assert len(tools) == 1
        assert tools[0].name == "shell"

    def test_empty_config_section_uses_defaults(self, tmp_path):
        """TOML with empty [config] → tool gets default config values."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        tools = discover_tools(tmp_path)
        shell = next(t for t in tools if t.name == "shell")
        # Should have defaults from BUILTIN_TOOLS, not crash
        assert isinstance(shell.config, dict)

    def test_all_builtin_tools_discoverable(self, tmp_path):
        """All 7 built-in tools can be discovered."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        for name in [
            "shell",
            "file_read",
            "file_write",
            "file_edit",
            "web_search",
            "web_fetch",
            "subagent",
        ]:
            (tools_dir / f"{name}.toml").write_text("[config]\n")

        tools = discover_tools(tmp_path)
        names = {t.name for t in tools}
        assert names == {
            "shell",
            "file_read",
            "file_write",
            "file_edit",
            "web_search",
            "web_fetch",
            "subagent",
        }


# --- Schema generation ---


class TestToolSchemas:

    def test_generates_api_ready_schemas(self, tmp_path):
        """Schemas are dicts ready to pass to the LLM API."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        tools = discover_tools(tmp_path)
        schemas = tool_schemas(tools)

        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["name"] == "shell"
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"

    def test_empty_tools_returns_empty_schemas(self):
        """No tools → empty schema list."""
        assert tool_schemas([]) == []

    def test_schema_includes_required_params(self, tmp_path):
        """Schema marks required parameters."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")

        tools = discover_tools(tmp_path)
        schemas = tool_schemas(tools)
        schema = schemas[0]
        assert "required" in schema["input_schema"]
        assert "command" in schema["input_schema"]["required"]

    def test_multiple_tools_produce_multiple_schemas(self, tmp_path):
        """Each tool produces a separate schema entry."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shell.toml").write_text("[config]\n")
        (tools_dir / "file_read.toml").write_text("[config]\n")
        (tools_dir / "file_write.toml").write_text("[config]\n")

        tools = discover_tools(tmp_path)
        schemas = tool_schemas(tools)
        assert len(schemas) == 3
        names = {s["name"] for s in schemas}
        assert names == {"shell", "file_read", "file_write"}


# --- Truncation ---


class TestTruncateResult:

    def test_under_limit_unchanged(self):
        """Text shorter than limit passes through unchanged."""
        text = "hello world"
        assert truncate_result(text, 1000) == text

    def test_exactly_at_limit_unchanged(self):
        """Text exactly at limit passes through unchanged."""
        text = "x" * 100
        assert truncate_result(text, 100) == text

    def test_over_limit_truncated_with_marker(self):
        """Text over limit gets head + marker + tail."""
        text = "A" * 10000
        result = truncate_result(text, 1000)
        assert len(result) < len(text)
        # Marker must be present
        assert "truncated" in result.lower()

    def test_preserves_head_and_tail(self):
        """Head and tail of original text are preserved."""
        head = "HEAD_CONTENT_" * 10
        middle = "M" * 10000
        tail = "TAIL_CONTENT_" * 10
        text = head + middle + tail

        result = truncate_result(text, 500)
        assert result.startswith("HEAD_CONTENT_")
        assert result.endswith("TAIL_CONTENT_")

    def test_marker_includes_char_count(self):
        """Truncation marker reports the fixed-point omitted-character count."""
        text = "x" * 10000
        result = truncate_result(text, 1000)
        import re

        match = re.search(r"\[truncated: (\d+) chars removed", result)
        assert match is not None
        # N = len(text) - max_chars + len(marker(N)); with the required Fix C
        # marker, N=9106 and len(marker)=106, so 9000 + 106 is stable.
        assert int(match.group(1)) == 9106

    def test_marker_counts_actual_gap(self):
        """R1: the marker count equals the real omitted gap at the 50k seam."""
        import re

        text = "w" * 50_087
        result = truncate_result(text, 50_000)
        match = re.search(r"\[truncated: (\d+) chars removed[^]]*\]", result)
        assert match is not None
        head = result[:match.start()]
        tail = result[match.end():]
        omitted = len(text) - len(head) - len(tail)
        assert int(match.group(1)) == omitted
        assert len(result) == 50_000

    def test_marker_digit_boundary(self):
        """R2: N0=999 crosses to a four-digit fixed point and stays exact."""
        import re

        text = "w" * 1_999
        result = truncate_result(text, 1_000)
        match = re.search(r"\[truncated: (\d+) chars removed[^]]*\]", result)
        assert match is not None
        head = result[:match.start()]
        tail = result[match.end():]
        omitted = len(text) - len(head) - len(tail)
        # N0 = 1999 - 1000 = 999 (three digits). With the exact Fix C marker,
        # marker(999) is 105 chars, producing N1=1104 (four digits); rebuilding
        # gives a 106-char marker and the stable fixed point N2=N3=1105. The
        # required ten-iteration cap therefore has ample margin (two updates).
        n0 = len(text) - 1_000
        n1 = n0 + len(
            f"[truncated: {n0} chars removed — re-run with a smaller limit or "
            "a narrower query/command to retrieve more]"
        )
        n2 = n0 + len(
            f"[truncated: {n1} chars removed — re-run with a smaller limit or "
            "a narrower query/command to retrieve more]"
        )
        n3 = n0 + len(
            f"[truncated: {n2} chars removed — re-run with a smaller limit or "
            "a narrower query/command to retrieve more]"
        )
        assert (n0, n1, n2, n3) == (999, 1_104, 1_105, 1_105)
        assert int(match.group(1)) == n2
        assert int(match.group(1)) == omitted
        assert len(result) == 1_000

    def test_under_limit_unchanged_fixed_point_regression(self):
        """R3: fixed-point accounting never changes an under-budget value."""
        text = "unchanged-window"
        assert truncate_result(text, len(text)) == text
        assert truncate_result(text, len(text) + 1) == text

    def test_marker_exceeds_budget_fallback(self):
        """R4: when the marker exceeds the budget, return marker(N_converged)[:limit].

        The fallback path (content_budget < 0) returns the *converged*-N marker
        sliced to the limit, not the legacy N0 marker. N is the documented
        fixed point: N = overflow + len(marker(N)), iterated up to 10 times.
        Pinned honestly across limits that span the pre-number, in-number, and
        near-full marker regions — never loosened (the N value is meaningless
        here by design, but the prefix the model receives is exact).
        """
        text = "w" * 200

        def _converged_marker(txt, limit):
            overflow = len(txt) - limit
            removed = overflow
            for _ in range(10):
                marker = (
                    f"[truncated: {removed} chars removed — re-run with a "
                    "smaller limit or a narrower query/command to retrieve "
                    "more]"
                )
                nxt = overflow + len(marker)
                if nxt == removed:
                    break
                removed = nxt
            marker = (
                f"[truncated: {removed} chars removed — re-run with a "
                "smaller limit or a narrower query/command to retrieve "
                "more]"
            )
            return marker

        for limit in (10, 16, 50, 102):
            marker = _converged_marker(text, limit)
            assert len(marker) > limit, (
                f"marker ({len(marker)} chars) must exceed limit {limit} to "
                "exercise the fallback path"
            )
            assert truncate_result(text, limit) == marker[:limit]

    def test_zero_length_text(self):
        """Empty string is unchanged."""
        assert truncate_result("", 1000) == ""

    def test_very_small_limit(self):
        """Even with a tiny limit, function doesn't crash."""
        text = "hello world, this is a test"
        result = truncate_result(text, 10)
        # Should produce something, even if mostly marker
        assert isinstance(result, str)

    def test_custom_marker_template_audience_text(self):
        """kdsn.247.2: marker_template swaps the agent-facing steering text."""
        text = "A" * 10000
        template = ("[{n} chars elided from this notice — "
                    "full content in session JSONL]")
        result = truncate_result(text, 1000, marker_template=template)
        assert "elided from this notice" in result
        assert "full content in session JSONL" in result
        assert "re-run with a smaller" not in result
        assert len(result) == 1000

    def test_custom_marker_template_counts_actual_gap(self):
        """kdsn.247.2: fixed-point budget math holds with a custom template —
        the count N still equals the real omitted gap."""
        import re

        text = "w" * 20000
        template = ("[{n} chars elided from this notice — "
                    "full content in session JSONL]")
        result = truncate_result(text, 5000, marker_template=template)
        match = re.search(
            r"\[(\d+) chars elided from this notice — "
            r"full content in session JSONL\]", result)
        assert match is not None
        head = result[:match.start()]
        tail = result[match.end():]
        omitted = len(text) - len(head) - len(tail)
        assert int(match.group(1)) == omitted
        assert result.startswith("w")
        assert result.endswith("w")

    def test_default_marker_template_unchanged(self):
        """kdsn.247.2: no marker_template → the agent-facing default persists."""
        result = truncate_result("A" * 10000, 1000)
        assert ("re-run with a smaller limit or a narrower query/command"
                in result)
