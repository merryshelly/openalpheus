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
        """Truncation marker indicates how many characters were removed."""
        text = "x" * 10000
        result = truncate_result(text, 1000)
        # Marker should contain a number (the removed char count)
        import re

        numbers = re.findall(r"\d+", result)
        assert len(numbers) > 0
        # At least one number should be in the thousands (chars removed)
        assert any(int(n) > 1000 for n in numbers)

    def test_zero_length_text(self):
        """Empty string is unchanged."""
        assert truncate_result("", 1000) == ""

    def test_very_small_limit(self):
        """Even with a tiny limit, function doesn't crash."""
        text = "hello world, this is a test"
        result = truncate_result(text, 10)
        # Should produce something, even if mostly marker
        assert isinstance(result, str)
