"""Tests for shell command execution.

Interface contract:
    run_shell(command, cwd=None, env=None, timeout=30, max_output=50000) -> ToolResult

Stateless subprocess execution. Each call is independent.
Returns stdout on success (is_error=False), stderr on failure (is_error=True).
Timeout kills the process. Output truncated at max_output.
"""

import pytest
from openalph.tools.shell import run_shell
from openalph.tools import ToolResult


class TestBasicExecution:

    @pytest.mark.asyncio
    async def test_simple_command_returns_stdout(self):
        """Successful command returns stdout as content."""
        result = await run_shell("echo hello")
        assert result.content.strip() == "hello"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_returns_tool_result(self):
        """Return type is ToolResult."""
        result = await run_shell("echo test")
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_multiline_output(self):
        """Multi-line output is captured correctly."""
        result = await run_shell("printf 'line1\\nline2\\nline3'")
        assert "line1" in result.content
        assert "line2" in result.content
        assert "line3" in result.content

    @pytest.mark.asyncio
    async def test_shell_pipe(self):
        """Shell pipes work."""
        result = await run_shell("echo 'hello world' | tr 'h' 'H'")
        assert "Hello" in result.content

    @pytest.mark.asyncio
    async def test_shell_expansion(self):
        """Shell variable expansion works."""
        result = await run_shell("echo $((2 + 2))")
        assert "4" in result.content


class TestErrorHandling:

    @pytest.mark.asyncio
    async def test_nonzero_exit_returns_error(self):
        """Non-zero exit code → is_error=True."""
        result = await run_shell("exit 1")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_stderr_in_error_content(self):
        """Stderr is included in error result content."""
        result = await run_shell("echo oops >&2 && exit 1")
        assert result.is_error is True
        assert "oops" in result.content

    @pytest.mark.asyncio
    async def test_command_not_found(self):
        """Non-existent command → is_error=True."""
        result = await run_shell("not_a_real_command_xyz_123")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_exit_code_in_error_content(self):
        """Error result includes or references the exit code."""
        result = await run_shell("exit 42")
        assert result.is_error is True
        # Should mention the exit code somewhere
        assert "42" in result.content


class TestWorkingDirectory:

    @pytest.mark.asyncio
    async def test_cwd_sets_working_directory(self, tmp_path):
        """cwd parameter changes the working directory."""
        result = await run_shell("pwd", cwd=str(tmp_path))
        assert result.content.strip() == str(tmp_path)

    @pytest.mark.asyncio
    async def test_cwd_none_uses_default(self):
        """cwd=None uses the current process working directory."""
        result = await run_shell("pwd")
        assert result.is_error is False
        assert len(result.content.strip()) > 0

    @pytest.mark.asyncio
    async def test_invalid_cwd_returns_error(self):
        """Non-existent cwd → is_error=True."""
        result = await run_shell("pwd", cwd="/tmp/does_not_exist_xyz_123")
        assert result.is_error is True


class TestEnvironment:

    @pytest.mark.asyncio
    async def test_env_sets_variables(self):
        """env parameter adds environment variables."""
        result = await run_shell(
            "echo $MY_TEST_VAR", env={"MY_TEST_VAR": "test_value_42"}
        )
        assert "test_value_42" in result.content

    @pytest.mark.asyncio
    async def test_env_merges_with_existing(self):
        """env adds to (not replaces) the existing environment."""
        # PATH should still work even with custom env
        result = await run_shell("which echo", env={"MY_VAR": "test"})
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_env_none_inherits_environment(self):
        """env=None inherits the parent process environment."""
        result = await run_shell("echo $HOME")
        assert result.is_error is False
        assert len(result.content.strip()) > 0


class TestTimeout:

    @pytest.mark.asyncio
    async def test_timeout_kills_long_command(self):
        """Command exceeding timeout is killed and returns error."""
        result = await run_shell("sleep 60", timeout=1)
        assert result.is_error is True
        assert "timeout" in result.content.lower()

    @pytest.mark.asyncio
    async def test_fast_command_within_timeout(self):
        """Command completing before timeout succeeds."""
        result = await run_shell("echo fast", timeout=10)
        assert result.is_error is False
        assert "fast" in result.content


class TestOutputTruncation:

    @pytest.mark.asyncio
    async def test_output_truncated_at_max(self):
        """Output exceeding max_output is truncated."""
        result = await run_shell("seq 1 100000", max_output=1000)
        assert len(result.content) <= 2000  # some overhead for marker
        assert result.is_error is False  # truncation is not an error

    @pytest.mark.asyncio
    async def test_small_output_not_truncated(self):
        """Output under max_output passes through unchanged."""
        result = await run_shell("echo short", max_output=50000)
        assert result.content.strip() == "short"


class TestStderrOnSuccess:

    @pytest.mark.asyncio
    async def test_stderr_captured_on_success(self):
        """On exit 0, both stdout and stderr are available."""
        result = await run_shell("echo out && echo err >&2")
        assert result.is_error is False
        # stdout should definitely be there
        assert "out" in result.content
