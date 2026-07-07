"""Tests for the sub-agent FLIGHT RECORDER (workspace-kdsn.192).

Full-content, append-only transcript per subagent run, written to:

    workspace/sessions/subs/<YYYY-MM-DD>-<safe_call_id>.jsonl

This is ADDITIVE and fully separate from the EXISTING metrics log
(`_append_log` -> workspace/logs/subagents/<ts>-<call_id>.jsonl, covered by
test_subagent_visibility.py and test_executor_subagent.py). That log records
iteration token counts, tool NAMES, and a summary -- metrics, not content.
This file tests the SECOND recorder: the one that captures full CONTENT
(the "flight recorder"). Both logs must coexist, untouched by each other.

The recorder is pure out-of-band I/O:
  - it must never mutate `messages` passed to complete()
  - it must never mutate the returned ToolResult
  - nothing it writes may enter the model context
  - a recorder failure must never break the sub run (fail-safe, log-and-continue)

Interface contract exercised here:
    run_subagent(task, config, tools=None, system_prompt=None, model=None,
                 max_tokens=None, max_iterations=None, call_id=None,
                 parent_room_id=None) -> ToolResult
"""

import json
import re
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from openalph.tools.subagent import run_subagent, MAX_ITERATIONS
from openalph.tools import ToolDef, ToolResult
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import Response, Usage, ToolCall


# NON-REAL fixture secret -- matches the `anthropic_api_key` redaction pattern
# (r"sk-ant-[a-zA-Z0-9_-]{8,}" in openalph.tools.security.CREDENTIAL_PATTERNS)
# without being an actual credential.
FIXTURE_SECRET = "sk-ant-" + "A" * 95


# --- Fixtures / helpers (mirrors test_subagent_visibility.py / test_executor_subagent.py) ---

def make_provider():
    return ProviderConfig(
        key="anthropic", type="anthropic", api_key="sk-test",
        base_url=None, quirks=[],
    )


def make_config(workspace, **kwargs):
    defaults = dict(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider()},
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_tools():
    return [
        ToolDef(name="shell", description="Run shell", parameters={}, config={}),
        ToolDef(name="file_read", description="Read file", parameters={}, config={}),
        ToolDef(name="subagent", description="Spawn sub", parameters={}, config={}),
    ]


def text_response(text="done", input_tokens=100, output_tokens=50):
    return Response(
        content=text,
        tool_calls=[],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
    )


def tool_response(tool_name="shell", tool_input=None, tool_id="tc_1", content=""):
    return Response(
        content=content,
        tool_calls=[ToolCall(id=tool_id, name=tool_name, input=tool_input or {"command": "echo hi"})],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=100, output_tokens=50),
        stop_reason="tool_use",
    )


def _transcript_dir(workspace):
    return Path(workspace) / "sessions" / "subs"


def _metrics_dir(workspace):
    return Path(workspace) / "logs" / "subagents"


def _read_jsonl(path):
    return [json.loads(l) for l in path.read_text().strip().splitlines()]


# =============================================================================
# 1. Meta header
# =============================================================================

class TestMetaHeader:

    @pytest.mark.asyncio
    async def test_transcript_file_written_with_meta_header(self, tmp_path):
        """A run writes sessions/subs/<date>-<call_id>.jsonl with a meta header."""
        config = make_config(tmp_path)
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    return_value=text_response("all done")):
            await run_subagent(
                "do a thing", config, tools=make_tools(),
                call_id="call_abc123", parent_room_id="!room:matrix.local",
            )

        sub_dir = _transcript_dir(tmp_path)
        assert sub_dir.exists()
        files = list(sub_dir.glob("*.jsonl"))
        assert len(files) == 1

        fname = files[0].name
        assert re.match(r"^\d{4}-\d{2}-\d{2}-call_abc123\.jsonl$", fname), fname

        entries = _read_jsonl(files[0])
        header = entries[0]
        assert header["event"] == "meta"
        assert header["model"] == config.default_model
        assert header["parent_room_id"] == "!room:matrix.local"
        assert header["parent_call_id"] == "call_abc123"
        assert header["task"] == "do a thing"
        assert isinstance(header["ts_start"], int)

    @pytest.mark.asyncio
    async def test_meta_header_reflects_model_override(self, tmp_path):
        """If model= overrides the parent's default, the header records the resolved model."""
        config = make_config(tmp_path, default_model="anthropic/claude-opus-4-20250514")
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    return_value=text_response("done")):
            await run_subagent(
                "task", config, tools=make_tools(), call_id="call_model",
                model="anthropic/claude-haiku-3-5-20241022",
            )

        files = list(_transcript_dir(tmp_path).glob("*.jsonl"))
        entries = _read_jsonl(files[0])
        assert entries[0]["model"] == "anthropic/claude-haiku-3-5-20241022"

    @pytest.mark.asyncio
    async def test_meta_header_parent_room_id_defaults_to_none(self, tmp_path):
        """parent_room_id is optional; omitted -> recorded as null."""
        config = make_config(tmp_path)
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    return_value=text_response("done")):
            await run_subagent("task", config, tools=make_tools(), call_id="call_noroom")

        files = list(_transcript_dir(tmp_path).glob("*.jsonl"))
        entries = _read_jsonl(files[0])
        assert entries[0]["parent_room_id"] is None


# =============================================================================
# 2. assistant / tool_result / final entries
# =============================================================================

class TestTranscriptEntries:

    @pytest.mark.asyncio
    async def test_assistant_tool_result_and_final_entries(self, tmp_path):
        """Transcript records assistant turns, wrapped tool results, and a final entry."""
        config = make_config(tmp_path)
        tools = make_tools()

        responses = [
            tool_response(tool_name="shell", tool_input={"command": "date"},
                          tool_id="tc_1", content="thinking..."),
            text_response("Today is March 8, 2026"),
        ]
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=responses), \
             patch("openalph.tools.execute_tool", new_callable=AsyncMock,
                    return_value=ToolResult(content="Sat Mar 8 2026", is_error=False)):
            result = await run_subagent(
                "What's the date?", config, tools=tools, call_id="call_xyz",
            )

        assert result.is_error is False
        assert "March 8" in result.content

        files = list(_transcript_dir(tmp_path).glob("*.jsonl"))
        assert len(files) == 1
        entries = _read_jsonl(files[0])

        # --- assistant entries: one per iteration that produced a response ---
        assistant_entries = [e for e in entries if e["event"] == "assistant"]
        assert len(assistant_entries) == 2

        a0 = assistant_entries[0]
        assert a0["iteration"] == 0
        assert a0["content"] == "thinking..."
        assert a0["tool_calls"] == [{"name": "shell", "id": "tc_1", "input": {"command": "date"}}]

        a1 = assistant_entries[1]
        assert a1["iteration"] == 1
        assert a1["content"] == "Today is March 8, 2026"
        assert a1["tool_calls"] == []

        # --- tool_result entry: the wrapped, post-redaction/post-truncation bytes ---
        tool_result_entries = [e for e in entries if e["event"] == "tool_result"]
        assert len(tool_result_entries) == 1
        tr = tool_result_entries[0]
        assert tr["iteration"] == 0
        assert tr["call_id"] == "tc_1"
        assert tr["name"] == "shell"
        assert tr["is_error"] is False
        # Must be the SAME `wrapped` string appended to messages (XML tool_result wrapper)
        assert tr["content"].startswith('<tool_result tool="shell" id="tc_1">')
        assert "Sat Mar 8 2026" in tr["content"]

        # --- final entry ---
        final_entries = [e for e in entries if e["event"] == "final"]
        assert len(final_entries) == 1
        f = final_entries[0]
        assert f["status"] == "completed"
        assert f["response"] == "Today is March 8, 2026"
        assert f["iterations"] == 1
        assert isinstance(f["usage"], dict)
        assert isinstance(f["elapsed_s"], float)

    @pytest.mark.asyncio
    async def test_tool_result_entry_records_error_flag(self, tmp_path):
        """A failing tool call is recorded with is_error=True in the transcript."""
        config = make_config(tmp_path)
        tools = make_tools()

        responses = [
            tool_response(tool_name="shell", tool_input={"command": "bad"}, tool_id="tc_err"),
            text_response("it failed"),
        ]
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=responses), \
             patch("openalph.tools.execute_tool", new_callable=AsyncMock,
                    return_value=ToolResult(content="command not found", is_error=True)):
            await run_subagent("run bad", config, tools=tools, call_id="call_err")

        files = list(_transcript_dir(tmp_path).glob("*.jsonl"))
        entries = _read_jsonl(files[0])
        tr = [e for e in entries if e["event"] == "tool_result"][0]
        assert tr["is_error"] is True

    @pytest.mark.asyncio
    async def test_final_entry_written_on_error_exit(self, tmp_path):
        """A final entry (status=error) is written when run_subagent hits an exception."""
        config = make_config(tmp_path)
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=Exception("provider exploded")):
            result = await run_subagent("failing task", config, tools=make_tools(),
                                        call_id="call_boom")

        assert result.is_error is True

        files = list(_transcript_dir(tmp_path).glob("*.jsonl"))
        assert len(files) == 1
        entries = _read_jsonl(files[0])
        final = [e for e in entries if e["event"] == "final"][-1]
        assert final["status"] == "error"

    @pytest.mark.asyncio
    async def test_final_entry_written_on_circuit_breaker(self, tmp_path):
        """A final entry (status=circuit_breaker) is written when the loop limit fires."""
        config = make_config(tmp_path)
        tools = make_tools()
        endless_tool = tool_response(tool_name="shell", tool_id="tc_loop")
        summary_response = text_response("I was stuck in a loop")

        async def mock_complete(**kwargs):
            if kwargs.get("tools") is None:
                return summary_response
            return endless_tool

        with patch("openalph.tools.subagent.complete", side_effect=mock_complete), \
             patch("openalph.tools.execute_tool", new_callable=AsyncMock,
                    return_value=ToolResult(content="ok")):
            result = await run_subagent("looping task", config, tools=tools,
                                        max_iterations=3, call_id="call_loop")

        assert result.is_error is True

        files = list(_transcript_dir(tmp_path).glob("*.jsonl"))
        entries = _read_jsonl(files[0])
        final = [e for e in entries if e["event"] == "final"][-1]
        assert final["status"] == "circuit_breaker"
        assert final["iterations"] == 3


# =============================================================================
# 3. Redaction — transcript records what the sub SAW, post-redaction
# =============================================================================

class TestRedactionInTranscript:

    @pytest.mark.asyncio
    async def test_tool_result_shows_redacted_marker_not_raw_secret(self, tmp_path):
        """A tool result containing a credential is redacted before it hits the transcript.

        execute_tool() itself performs redaction (tools/__init__.py) before returning
        to the run_subagent loop, so leaving execute_tool UNMOCKED here (and instead
        mocking the underlying shell executor) exercises the real redaction path.
        """
        config = make_config(tmp_path)
        tools = make_tools()

        responses = [
            tool_response(tool_name="shell", tool_input={"command": "cat leaked.txt"},
                          tool_id="tc_secret"),
            text_response("done"),
        ]
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=responses), \
             patch("openalph.tools.shell.run_shell", new_callable=AsyncMock,
                    return_value=ToolResult(content=f"leaked key: {FIXTURE_SECRET}")):
            result = await run_subagent("leak a secret", config, tools=tools,
                                        call_id="call_redact")

        assert result.is_error is False

        files = list(_transcript_dir(tmp_path).glob("*.jsonl"))
        entries = _read_jsonl(files[0])
        tr = [e for e in entries if e["event"] == "tool_result"][0]

        assert FIXTURE_SECRET not in tr["content"]
        assert "[REDACTED" in tr["content"]


# =============================================================================
# 4. Never-context — recorder is out-of-band, pure I/O
# =============================================================================

class TestNeverEntersModelContext:

    @pytest.mark.asyncio
    async def test_returned_result_and_messages_carry_no_transcript_bytes(self, tmp_path):
        """The ToolResult and the messages sent to complete() are free of transcript data."""
        config = make_config(tmp_path)
        tools = make_tools()

        responses = [
            tool_response(tool_name="shell", tool_input={"command": "date"}, tool_id="tc_1"),
            text_response("Final answer only"),
        ]
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    side_effect=responses) as mock_complete, \
             patch("openalph.tools.execute_tool", new_callable=AsyncMock,
                    return_value=ToolResult(content="tool output")):
            result = await run_subagent(
                "task", config, tools=tools, call_id="call_ctx",
                parent_room_id="!room:matrix.local",
            )

        # The returned ToolResult content is EXACTLY the final response text.
        assert result.content == "Final answer only"

        # No transcript/meta markers leaked into the returned content.
        for marker in ('"event"', '"meta"', "parent_room_id", "parent_call_id", "ts_start"):
            assert marker not in result.content

        # Every messages= payload actually sent to the model is free of
        # transcript event/meta keys -- the recorder must never leak into
        # what the model sees.
        assert mock_complete.call_count == 2
        for call in mock_complete.call_args_list:
            messages = call.kwargs["messages"]
            serialized = json.dumps(messages, default=str)
            assert '"event"' not in serialized
            assert '"meta"' not in serialized
            assert "parent_room_id" not in serialized
            assert "ts_start" not in serialized


# =============================================================================
# 5. Path traversal — call_id is sanitized (reuses _sanitize_call_id)
# =============================================================================

class TestPathTraversal:

    @pytest.mark.asyncio
    async def test_call_id_traversal_is_sanitized_into_subs_dir(self, tmp_path):
        """A path-traversal call_id produces a file safely inside sessions/subs/."""
        config = make_config(tmp_path)
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    return_value=text_response("done")):
            await run_subagent(
                "task", config, tools=make_tools(), call_id="../../etc/evil",
            )

        sub_dir = _transcript_dir(tmp_path)
        files = list(sub_dir.glob("*.jsonl"))
        assert len(files) == 1
        assert files[0].parent == sub_dir
        assert "/" not in files[0].name
        assert ".." not in files[0].name

        # Nothing escaped upward out of the workspace via traversal.
        assert not (tmp_path / "etc").exists()


# =============================================================================
# 6. Fail-safe — a recorder I/O failure must never break the sub run
# =============================================================================

class TestFailSafe:

    @pytest.mark.asyncio
    async def test_transcript_write_failure_does_not_break_subagent(self, tmp_path):
        """If writing the transcript raises, the sub still completes normally."""
        config = make_config(tmp_path)
        real_open = open

        def _boom(file, *args, **kwargs):
            path_str = str(file)
            if "sessions" in path_str and "subs" in path_str:
                raise OSError("simulated disk failure")
            return real_open(file, *args, **kwargs)

        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    return_value=text_response("still works")), \
             patch("openalph.tools.subagent.open", side_effect=_boom, create=True):
            result = await run_subagent("task", config, tools=make_tools(),
                                        call_id="call_failsafe")

        assert result.is_error is False
        assert result.content == "still works"

    @pytest.mark.asyncio
    async def test_transcript_dir_unwritable_does_not_break_subagent(self, tmp_path):
        """If sessions/subs/ cannot be created/written, the sub still completes normally."""
        config = make_config(tmp_path)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True)
        sessions_dir.chmod(0o000)
        try:
            with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                        return_value=text_response("still works too")):
                result = await run_subagent("task", config, tools=make_tools(),
                                            call_id="call_failsafe2")
        finally:
            sessions_dir.chmod(0o755)

        assert result.is_error is False
        assert result.content == "still works too"


# =============================================================================
# 7. Co-existence — the pre-existing metrics log (.127) is untouched
# =============================================================================

class TestCoexistenceWithMetricsLog:

    @pytest.mark.asyncio
    async def test_both_flight_recorder_and_metrics_log_written(self, tmp_path):
        """Both logs/subagents/*.jsonl (metrics) AND sessions/subs/*.jsonl (transcript) exist."""
        config = make_config(tmp_path)
        with patch("openalph.tools.subagent.complete", new_callable=AsyncMock,
                    return_value=text_response("all done")):
            await run_subagent(
                "do a thing", config, tools=make_tools(), call_id="call_both",
            )

        metrics_files = list(_metrics_dir(tmp_path).glob("*.jsonl"))
        transcript_files = list(_transcript_dir(tmp_path).glob("*.jsonl"))

        assert len(metrics_files) == 1, "existing metrics log (.127) must still be written"
        assert len(transcript_files) == 1, "new flight recorder transcript must be written"

        # The metrics log keeps its own pre-existing shape, untouched.
        metrics_entries = _read_jsonl(metrics_files[0])
        assert metrics_entries[-1]["event"] == "summary"
        assert metrics_entries[-1]["status"] == "completed"

        # The transcript has its own independent shape.
        transcript_entries = _read_jsonl(transcript_files[0])
        assert transcript_entries[0]["event"] == "meta"
        assert transcript_entries[-1]["event"] == "final"

    @pytest.mark.asyncio
    async def test_max_iterations_constant_unchanged(self):
        """Sanity: this feature does not touch the existing circuit breaker constant."""
        assert MAX_ITERATIONS == 200
