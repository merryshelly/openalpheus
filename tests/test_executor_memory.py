"""Tests for memory_search tool executor and tool registration.

Interface contract:
    run_memory_search(query, config, workspace) -> ToolResult
    BUILTIN_TOOLS contains "memory_search" key
    Tool enabled via workspace/tools/memory_search.toml
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from openalph.tools import BUILTIN_TOOLS, discover_tools, ToolResult


class TestToolRegistration:

    def test_memory_search_in_builtin_tools(self):
        """memory_search is registered in BUILTIN_TOOLS."""
        assert "memory_search" in BUILTIN_TOOLS

    def test_has_required_schema_fields(self):
        tool = BUILTIN_TOOLS["memory_search"]
        assert "description" in tool
        assert "parameters" in tool
        assert "config" in tool

    def test_query_is_required_parameter(self):
        params = BUILTIN_TOOLS["memory_search"]["parameters"]
        assert "query" in params["properties"]
        assert "query" in params["required"]

    def test_has_optional_parameters(self):
        params = BUILTIN_TOOLS["memory_search"]["parameters"]
        assert "max_results" in params["properties"]
        assert "min_score" in params["properties"]

    def test_config_has_embedding_settings(self):
        config = BUILTIN_TOOLS["memory_search"]["config"]
        assert "embedding_model" in config
        assert "embedding_base_url" in config

    def test_config_has_search_weights(self):
        config = BUILTIN_TOOLS["memory_search"]["config"]
        assert "vector_weight" in config
        assert "text_weight" in config

    def test_discovered_when_toml_exists(self, tmp_path):
        """Tool is discovered when memory_search.toml exists in tools dir."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "memory_search.toml").write_text("")
        tools = discover_tools(tmp_path)
        names = [t.name for t in tools]
        assert "memory_search" in names

    def test_config_override_from_toml(self, tmp_path):
        """TOML config values override defaults."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "memory_search.toml").write_text(
            '[config]\nembedding_model = "custom-model"\nvector_weight = 0.5\n'
        )
        tools = discover_tools(tmp_path)
        mem_tool = [t for t in tools if t.name == "memory_search"][0]
        assert mem_tool.config["embedding_model"] == "custom-model"
        assert mem_tool.config["vector_weight"] == 0.5


class TestToolExecution:

    @pytest.mark.asyncio
    async def test_returns_tool_result(self, tmp_path):
        """Executor returns ToolResult."""
        from openalph.tools.memory_search import run_memory_search

        # Create a minimal workspace with a memory dir
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "test.md").write_text(
            "## Test\nContent about validators that is long enough for minimum chunk size threshold."
        )

        config = BUILTIN_TOOLS["memory_search"]["config"].copy()
        result = await run_memory_search(
            query="validators",
            config=config,
            workspace=tmp_path,
        )
        assert isinstance(result, ToolResult)
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_result_contains_path_and_lines(self, tmp_path):
        """Results include file path and line numbers for follow-up reads."""
        from openalph.tools.memory_search import run_memory_search

        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "test.md").write_text(
            "## Validators\nInformation about validator monitoring that is detailed enough for a good match."
        )

        config = BUILTIN_TOOLS["memory_search"]["config"].copy()
        result = await run_memory_search(
            query="validator monitoring",
            config=config,
            workspace=tmp_path,
        )
        # Output should contain path reference
        assert "test.md" in result.content

    @pytest.mark.asyncio
    async def test_empty_query_returns_helpful_error(self, tmp_path):
        from openalph.tools.memory_search import run_memory_search

        config = BUILTIN_TOOLS["memory_search"]["config"].copy()
        result = await run_memory_search(
            query="",
            config=config,
            workspace=tmp_path,
        )
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_no_results_returns_informative_message(self, tmp_path):
        """When no results found, return a clear message (not error)."""
        from openalph.tools.memory_search import run_memory_search

        (tmp_path / "memory").mkdir()
        # Empty memory dir → no results for any query

        config = BUILTIN_TOOLS["memory_search"]["config"].copy()
        result = await run_memory_search(
            query="xyznonexistent",
            config=config,
            workspace=tmp_path,
        )
        assert result.is_error is False
        assert "no results" in result.content.lower() or "0 results" in result.content.lower()

    @pytest.mark.asyncio
    async def test_max_results_limits_output(self, tmp_path):
        """max_results parameter caps the number of returned results."""
        from openalph.tools.memory_search import run_memory_search

        (tmp_path / "memory").mkdir()
        for i in range(10):
            (tmp_path / "memory" / f"file{i}.md").write_text(
                f"## Topic {i}\nContent about topic {i} validator monitoring that is long enough."
            )

        config = BUILTIN_TOOLS["memory_search"]["config"].copy()
        result = await run_memory_search(
            query="validator monitoring",
            config=config,
            workspace=tmp_path,
            max_results=3,
        )
        # Count result entries (each starts with [N])
        import re
        matches = re.findall(r'\[\d+\]', result.content)
        assert len(matches) <= 3
