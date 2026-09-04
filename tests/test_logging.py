"""Tests for structured JSONL logging.

Phase 3.5 Feature: Per-agent JSONL log file for debugging and observability.

Contract:
    - Agent writes one JSONL entry per LLM turn to workspace/logs/<agent-name>-YYYY-MM-DD.jsonl
    - Each entry contains: ts, room_id, direction, model, input_tokens, output_tokens,
      tool_calls, latency_ms, content_preview
    - Log file rotates by date (new file per day)
    - Entries are valid JSON (one per line)
    - Tool calls within a turn are captured in the same entry
    - Log directory is created automatically if missing
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.agent import Agent
from openalph.config import AgentConfig
from openalph.provider import StreamEvent, Response, Usage, ToolCall


# --- Fixtures ---


def make_provider(key="default", type="anthropic", api_key="sk-test", base_url=None, quirks=None):
    from openalph.config import ProviderConfig
    return ProviderConfig(key=key, type=type, api_key=api_key, base_url=base_url, quirks=quirks or [])


def make_agent_config(tmp_path, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider(key="anthropic")},
        workspace=tmp_path,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_response(content="Hello!", input_tokens=100, output_tokens=50, tool_calls=None):
    """Create a mock LLM response."""
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = tool_calls or []
    resp.usage = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    return resp


def make_stream_events(content="Hello!", input_tokens=100, output_tokens=50, tool_calls=None):
    """Create a mock async generator that yields stream events."""
    async def _stream(*args, **kwargs):
        # Yield text delta
        yield StreamEvent(type="text", content=content)
        # Yield done event with response
        yield StreamEvent(
            type="done",
            response=Response(
                content=content,
                model="claude-sonnet-4-20250514",
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
                stop_reason="end_turn",
                tool_calls=tool_calls or [],
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-20250514",
        )
    return _stream


def make_stream_events_with_tool(content="", tool_call=None, input_tokens=100, output_tokens=50):
    """Create a mock async generator that yields a tool call."""
    async def _stream(*args, **kwargs):
        # Yield tool_start
        yield StreamEvent(
            type="tool_start",
            tool_index=0,
            tool_id=tool_call.id if tool_call else "tc_1",
            tool_name=tool_call.name if tool_call else "shell",
        )
        # Yield tool_done
        yield StreamEvent(
            type="tool_done",
            tool_index=0,
            tool_call=tool_call if tool_call else ToolCall(id="tc_1", name="shell", input={}),
        )
        # Yield done event with response
        yield StreamEvent(
            type="done",
            response=Response(
                content=content,
                model="claude-sonnet-4-20250514",
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
                stop_reason="end_turn",
                tool_calls=[tool_call] if tool_call else [],
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-20250514",
        )
    return _stream


def make_tool_call(name="shell", tool_input=None, call_id="tc_1"):
    """Create a mock tool call."""
    tc = MagicMock()
    tc.name = name
    tc.input = tool_input or {"command": "date"}
    tc.id = call_id
    return tc


def make_tool_result(content="Mon Mar 8 20:00:00 EDT 2026", is_error=False):
    """Create a mock tool execution result."""
    result = MagicMock()
    result.content = content
    result.is_error = is_error
    return result


# --- Log File Creation ---


class TestLogFileCreation:

    @pytest.mark.asyncio
    async def test_log_file_created_on_first_turn(self, tmp_path):
        """JSONL log file is created after the first LLM call."""
        config = make_agent_config(tmp_path)
        response = make_response()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        log_dir = tmp_path / "logs"
        assert log_dir.exists(), "logs/ directory should be created automatically"

        log_files = list(log_dir.glob("test-agent-*.jsonl"))
        assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"

    @pytest.mark.asyncio
    async def test_log_directory_created_if_missing(self, tmp_path):
        """Log directory is created automatically when it doesn't exist."""
        config = make_agent_config(tmp_path)
        response = make_response()

        # Ensure no logs dir exists
        log_dir = tmp_path / "logs"
        assert not log_dir.exists()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        assert log_dir.exists()

    @pytest.mark.asyncio
    async def test_log_filename_contains_agent_name_and_date(self, tmp_path):
        """Log file named <agent-name>-YYYY-MM-DD.jsonl."""
        config = make_agent_config(tmp_path, name="watson")
        response = make_response()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        log_dir = tmp_path / "logs"
        log_files = list(log_dir.glob("watson-*.jsonl"))
        assert len(log_files) == 1

        # Verify date format in filename
        filename = log_files[0].name
        assert filename.startswith("watson-")
        assert filename.endswith(".jsonl")
        # Extract date part and validate format
        date_part = filename.replace("watson-", "").replace(".jsonl", "")
        datetime.strptime(date_part, "%Y-%m-%d")  # Raises if invalid format


# --- Log Entry Format ---


class TestLogEntryFormat:

    @pytest.mark.asyncio
    async def test_entry_is_valid_json(self, tmp_path):
        """Each line in the log file is valid JSON."""
        config = make_agent_config(tmp_path)
        response = make_response()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        for line in log_file.read_text().strip().splitlines():
            entry = json.loads(line)  # Raises on invalid JSON
            assert isinstance(entry, dict)

    @pytest.mark.asyncio
    async def test_entry_contains_required_fields(self, tmp_path):
        """Log entry contains all required fields."""
        config = make_agent_config(tmp_path)
        response = make_response(input_tokens=1200, output_tokens=350)

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        entry = json.loads(log_file.read_text().strip())

        required_fields = {
            "ts", "room_id", "direction", "model",
            "input_tokens", "output_tokens", "tool_calls",
            "latency_ms", "content_preview",
        }
        assert required_fields.issubset(entry.keys()), \
            f"Missing fields: {required_fields - entry.keys()}"

    @pytest.mark.asyncio
    async def test_entry_room_id_matches(self, tmp_path):
        """Log entry room_id matches the room where the message was processed."""
        config = make_agent_config(tmp_path)
        response = make_response()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!myroom:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        entry = json.loads(log_file.read_text().strip())
        assert entry["room_id"] == "!myroom:local"

    @pytest.mark.asyncio
    async def test_entry_direction_is_outbound(self, tmp_path):
        """LLM response log entries have direction='outbound'."""
        config = make_agent_config(tmp_path)
        response = make_response()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        entry = json.loads(log_file.read_text().strip())
        assert entry["direction"] == "outbound"

    @pytest.mark.asyncio
    async def test_entry_model_matches_config(self, tmp_path):
        """Log entry model matches the agent's configured model."""
        config = make_agent_config(tmp_path, default_model="claude-opus-4-20250514")
        response = make_response()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        entry = json.loads(log_file.read_text().strip())
        assert entry["model"] == "claude-opus-4-20250514"

    @pytest.mark.asyncio
    async def test_entry_token_counts_match_response(self, tmp_path):
        """Log entry token counts match what the LLM reported."""
        config = make_agent_config(tmp_path)
        response = make_response(input_tokens=1500, output_tokens=400)

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        entry = json.loads(log_file.read_text().strip())
        assert entry["input_tokens"] == 1500
        assert entry["output_tokens"] == 400

    @pytest.mark.asyncio
    async def test_entry_latency_is_positive(self, tmp_path):
        """Log entry latency_ms is a positive number."""
        config = make_agent_config(tmp_path)
        response = make_response()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        entry = json.loads(log_file.read_text().strip())
        assert isinstance(entry["latency_ms"], (int, float))
        assert entry["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_entry_content_preview_truncated(self, tmp_path):
        """Content preview is truncated to 200 characters."""
        config = make_agent_config(tmp_path)
        long_content = "x" * 500
        response = make_response(content=long_content)

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        entry = json.loads(log_file.read_text().strip())
        assert len(entry["content_preview"]) <= 200

    @pytest.mark.asyncio
    async def test_entry_content_preview_not_truncated_when_short(self, tmp_path):
        """Short content is preserved in full in preview."""
        config = make_agent_config(tmp_path)
        response = make_response(content="Hello back!")

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        entry = json.loads(log_file.read_text().strip())
        assert entry["content_preview"] == "Hello back!"

    @pytest.mark.asyncio
    async def test_entry_timestamp_is_iso8601(self, tmp_path):
        """Log entry ts is ISO 8601 UTC format."""
        config = make_agent_config(tmp_path)
        response = make_response()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        entry = json.loads(log_file.read_text().strip())
        # Should parse as ISO 8601
        dt = datetime.fromisoformat(entry["ts"])
        assert dt.tzinfo is not None or entry["ts"].endswith("Z")


# --- Tool Call Logging ---


class TestToolCallLogging:

    @pytest.mark.asyncio
    async def test_tool_calls_logged_in_entry(self, tmp_path):
        """Tool calls within a turn are captured in the log entry."""
        config = make_agent_config(tmp_path)
        tc = make_tool_call("shell", {"command": "date"}, "tc_1")
        make_response(content="", input_tokens=100, output_tokens=50, tool_calls=[tc])
        make_response(content="It's Monday.", input_tokens=200, output_tokens=100)
        tool_result = make_tool_result("Mon Mar 8 20:00:00 EDT 2026")

        call_count = 0

        async def mock_stream_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call returns tool call
                async for event in make_stream_events_with_tool(content="", tool_call=tc, input_tokens=100, output_tokens=50)():
                    yield event
            else:
                # Second call returns final response
                async for event in make_stream_events(content="It's Monday.", input_tokens=200, output_tokens=100)():
                    yield event

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.execute_tool", AsyncMock(return_value=tool_result)), \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[MagicMock(name="shell", config={})]):
            mock_stream.side_effect = mock_stream_fn
            agent = Agent(config)
            await agent.handle_input("What day is it?", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        lines = log_file.read_text().strip().splitlines()
        # Should have entries for each LLM call
        assert len(lines) >= 1

        # Find the entry with tool calls
        tool_entry = None
        for line in lines:
            entry = json.loads(line)
            if entry.get("tool_calls"):
                tool_entry = entry
                break

        assert tool_entry is not None, "Should have an entry with tool_calls"
        assert len(tool_entry["tool_calls"]) == 1
        assert tool_entry["tool_calls"][0]["name"] == "shell"
        assert tool_entry["tool_calls"][0]["is_error"] is False

    @pytest.mark.asyncio
    async def test_no_tool_calls_logged_as_empty_list(self, tmp_path):
        """Turn without tool calls has empty tool_calls list."""
        config = make_agent_config(tmp_path)
        response = make_response()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        entry = json.loads(log_file.read_text().strip())
        assert entry["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_tool_error_logged(self, tmp_path):
        """Tool call errors are recorded in the log entry."""
        config = make_agent_config(tmp_path)
        tc = make_tool_call("shell", {"command": "bad-cmd"}, "tc_err")
        make_response(content="", input_tokens=100, output_tokens=50, tool_calls=[tc])
        make_response(content="That command failed.", input_tokens=200, output_tokens=100)
        tool_result = make_tool_result("command not found", is_error=True)

        call_count = 0

        async def mock_stream_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                async for event in make_stream_events_with_tool(content="", tool_call=tc, input_tokens=100, output_tokens=50)():
                    yield event
            else:
                async for event in make_stream_events(content="That command failed.", input_tokens=200, output_tokens=100)():
                    yield event

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.execute_tool", AsyncMock(return_value=tool_result)), \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[MagicMock(name="shell", config={})]):
            mock_stream.side_effect = mock_stream_fn
            agent = Agent(config)
            await agent.handle_input("Run bad-cmd", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        lines = log_file.read_text().strip().splitlines()

        tool_entry = None
        for line in lines:
            entry = json.loads(line)
            if entry.get("tool_calls"):
                tool_entry = entry
                break

        assert tool_entry is not None
        assert tool_entry["tool_calls"][0]["is_error"] is True


# --- Multiple Turns ---


class TestMultipleTurns:

    @pytest.mark.asyncio
    async def test_multiple_turns_append_to_same_file(self, tmp_path):
        """Multiple turns in same session append to the same JSONL file."""
        config = make_agent_config(tmp_path)
        response = make_response()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("First", "!room:local")
            await agent.handle_input("Second", "!room:local")
            await agent.handle_input("Third", "!room:local")

        log_file = next((tmp_path / "logs").glob("*.jsonl"))
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 3

        # Each line is valid JSON
        for line in lines:
            json.loads(line)

    @pytest.mark.asyncio
    async def test_different_rooms_log_to_same_file(self, tmp_path):
        """Messages from different rooms all log to the same agent log file."""
        config = make_agent_config(tmp_path)
        response = make_response()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)
            await agent.handle_input("Hello", "!room1:local")
            await agent.handle_input("Hello", "!room2:local")

        log_files = list((tmp_path / "logs").glob("*.jsonl"))
        assert len(log_files) == 1  # Single file per agent per day

        log_file = log_files[0]
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 2

        entries = [json.loads(line) for line in lines]
        assert entries[0]["room_id"] == "!room1:local"
        assert entries[1]["room_id"] == "!room2:local"


# --- Date Rotation ---


class TestDateRotation:

    @pytest.mark.asyncio
    async def test_log_rotates_by_date(self, tmp_path):
        """New log file created when date changes."""
        config = make_agent_config(tmp_path)
        response = make_response()

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)

            # First call: logs to today's file
            with patch("openalph.agent.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(2026, 3, 8, 12, 0, 0, tzinfo=timezone.utc)
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                await agent.handle_input("Day 1", "!room:local")

            # Second call: simulate next day
            with patch("openalph.agent.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                await agent.handle_input("Day 2", "!room:local")

        log_files = sorted((tmp_path / "logs").glob("*.jsonl"))
        assert len(log_files) == 2
        assert "2026-03-08" in log_files[0].name
        assert "2026-03-09" in log_files[1].name


# --- Logging Resilience ---


class TestLoggingResilience:

    @pytest.mark.asyncio
    async def test_logging_failure_does_not_crash_agent(self, tmp_path):
        """If logging fails (e.g., disk full), agent still returns response."""
        config = make_agent_config(tmp_path)
        response = make_response(content="I'm still working!")

        with patch("openalph.agent.stream") as mock_stream, \
             patch("openalph.agent.assemble_prompt", return_value="system prompt"), \
             patch("openalph.agent.discover_tools", return_value=[]):
            mock_stream.side_effect = make_stream_events(content=response.content, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            agent = Agent(config)

            # Make logs dir read-only to simulate write failure
            log_dir = tmp_path / "logs"
            log_dir.mkdir()
            log_dir.chmod(0o444)

            try:
                result = await agent.handle_input("Hello", "!room:local")
                assert result == "I'm still working!"
            finally:
                log_dir.chmod(0o755)
