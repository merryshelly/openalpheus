"""Tests for the agent conversation loop.

Interface contract:
    Agent(config: AgentConfig) — initializes with config, assembles system prompt, empty history
    Agent.handle_input(text: str) -> str — sends message, returns response content
    Agent.history — list of {"role": ..., "content": ...} dicts
    Agent.total_input_tokens / total_output_tokens — cumulative usage
    Agent.status() -> dict — model, turns, token counts
"""

import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from openalph.agent import Agent
from openalph.config import AgentConfig
from openalph.provider import Response, Usage, StreamEvent


def make_provider(key="default", type="anthropic", api_key="sk-test", base_url=None, quirks=None):
    from openalph.config import ProviderConfig
    return ProviderConfig(key=key, type=type, api_key=api_key, base_url=base_url, quirks=quirks or [])


def make_config(workspace, **kwargs):
    from openalph.config import ProviderConfig
    defaults = dict(
        name="test",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": make_provider(key="anthropic")},
    )
    defaults.update(kwargs)
    defaults["workspace"] = workspace
    return AgentConfig(**defaults)


def make_response(content="Hello", input_tokens=10, output_tokens=5):
    return Response(
        content=content,
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
    )


def make_stream_events(content="Hello", input_tokens=10, output_tokens=5):
    """Create a mock async generator that yields stream events."""
    async def _stream(*args, **kwargs):
        yield StreamEvent(type="text", content=content)
        yield StreamEvent(
            type="done",
            response=Response(
                content=content,
                model="claude-sonnet-4-20250514",
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
                stop_reason="end_turn",
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-20250514",
        )
    return _stream


# --- Initialization ---


class TestAgentInit:

    def test_init_assembles_prompt(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("I am test agent.")
        config = make_config(tmp_path)
        agent = Agent(config)

        assert "I am test agent." in agent.system_prompt

    def test_init_empty_history(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        assert agent.history("_default") == []

    def test_init_zero_tokens(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        assert agent.total_input_tokens == 0
        assert agent.total_output_tokens == 0


# --- Conversation ---


class TestHandleInput:

    @pytest.mark.asyncio
    async def test_returns_response_content(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("Test soul.")
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock:
            mock.side_effect = make_stream_events("Hi there!")
            result = await agent.handle_input("Hello")

        assert result == "Hi there!"

    @pytest.mark.asyncio
    async def test_history_accumulates(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock:
            mock.side_effect = make_stream_events("Response 1")
            await agent.handle_input("Message 1")

            mock.side_effect = make_stream_events("Response 2")
            await agent.handle_input("Message 2")

        assert len(agent.history("_default")) == 4
        assert agent.history("_default")[0] == {"role": "user", "content": "Message 1"}
        assert agent.history("_default")[1] == {"role": "assistant", "content": "Response 1"}
        assert agent.history("_default")[2] == {"role": "user", "content": "Message 2"}
        assert agent.history("_default")[3] == {"role": "assistant", "content": "Response 2"}

    @pytest.mark.asyncio
    async def test_system_prompt_passed_to_provider(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("I am the soul.")
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock:
            mock.side_effect = make_stream_events("OK")
            await agent.handle_input("Hi")

        kw = mock.call_args.kwargs
        assert "I am the soul." in kw["system"]

    @pytest.mark.asyncio
    async def test_config_passed_to_provider(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock:
            mock.side_effect = make_stream_events("OK")
            await agent.handle_input("Hi")

        kw = mock.call_args.kwargs
        assert kw["config"] is config

    @pytest.mark.asyncio
    async def test_full_history_passed_to_provider(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock:
            mock.side_effect = make_stream_events("First")
            await agent.handle_input("Hello")

            mock.side_effect = make_stream_events("Second")
            await agent.handle_input("Follow-up")

        # Second call should include full history + new message
        second_call = mock.call_args_list[1]
        messages = second_call.kwargs["messages"]
        assert len(messages) == 3
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "First"
        assert messages[2]["content"] == "Follow-up"


# --- Token tracking ---


class TestTokenTracking:

    @pytest.mark.asyncio
    async def test_tokens_accumulate(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock:
            mock.side_effect = make_stream_events("A", input_tokens=100, output_tokens=50)
            await agent.handle_input("First")

            mock.side_effect = make_stream_events("B", input_tokens=200, output_tokens=80)
            await agent.handle_input("Second")

        assert agent.total_input_tokens == 300
        assert agent.total_output_tokens == 130


# --- Status ---


class TestStatus:

    def test_status_initial(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        status = agent.status()
        assert status["model"] == "anthropic/claude-sonnet-4-20250514"
        assert status["turns"] == 0
        assert status["total_input_tokens"] == 0
        assert status["total_output_tokens"] == 0

    @pytest.mark.asyncio
    async def test_status_after_conversation(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock:
            mock.side_effect = make_stream_events("Hi", input_tokens=50, output_tokens=20)
            await agent.handle_input("Hello")

        status = agent.status()
        assert status["turns"] == 1
        assert status["total_input_tokens"] == 50
        assert status["total_output_tokens"] == 20
        assert status["model"] == "anthropic/claude-sonnet-4-20250514"
        assert "name" in status


# --- Fix 3: Context overflow check before user message appended ---


class TestContextOverflowPreAppend:

    @pytest.mark.asyncio
    async def test_overflow_rejects_without_appending(self, tmp_path):
        """When context is near-full, a large user message triggers
        ContextOverflowError WITHOUT the message being added to history."""
        from openalph.agent import ContextOverflowError
        config = make_config(tmp_path, model_max_tokens=1000, max_tokens=200)
        agent = Agent(config)

        # Pre-fill history to near capacity (800 available tokens = 3200 chars)
        # System prompt is empty-ish, so fill history close to limit
        agent._rooms["_default"] = [
            {"role": "user", "content": "x" * 3000},
            {"role": "assistant", "content": "y" * 100},
        ]

        big_message = "z" * 2000  # would push well over

        with pytest.raises(ContextOverflowError):
            with patch("openalph.agent.stream") as mock:
                await agent.handle_input(big_message)

        # The big message must NOT be in history
        contents = [m.get("content", "") for m in agent.history("_default")]
        assert big_message not in contents


# --- Usage Reconstruction ---


import json as _json


class TestUsageReconstruction:

    def _write_jsonl(self, path, entries):
        with open(path, "w") as f:
            for entry in entries:
                f.write(_json.dumps(entry) + "\n")

    def test_reconstruct_from_empty_logs_dir(self, tmp_path):
        (tmp_path / "logs").mkdir()
        config = make_config(tmp_path)
        agent = Agent(config)
        assert agent.total_input_tokens == 0
        assert agent.total_output_tokens == 0
        assert agent.total_tool_calls == 0

    def test_reconstruct_from_no_logs_dir(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)
        assert agent.total_input_tokens == 0
        assert agent.total_output_tokens == 0
        assert agent.total_tool_calls == 0

    def test_reconstruct_sums_across_files(self, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        self._write_jsonl(logs / "a.jsonl", [
            {"input_tokens": 10, "output_tokens": 20, "tool_calls": []},
            {"input_tokens": 5, "output_tokens": 15, "tool_calls": []},
        ])
        self._write_jsonl(logs / "b.jsonl", [
            {"input_tokens": 100, "output_tokens": 200, "tool_calls": []},
        ])
        config = make_config(tmp_path)
        agent = Agent(config)
        assert agent.total_input_tokens == 115
        assert agent.total_output_tokens == 235
        assert agent.total_tool_calls == 0

    def test_reconstruct_skips_malformed_lines(self, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        log_file = logs / "mixed.jsonl"
        with open(log_file, "w") as f:
            f.write(_json.dumps({"input_tokens": 7, "output_tokens": 3, "tool_calls": []}) + "\n")
            f.write("this is not json\n")
            f.write(_json.dumps({"input_tokens": 2, "output_tokens": 1, "tool_calls": []}) + "\n")
        config = make_config(tmp_path)
        agent = Agent(config)
        assert agent.total_input_tokens == 9
        assert agent.total_output_tokens == 4

    def test_reconstruct_counts_tool_calls(self, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        self._write_jsonl(logs / "tools.jsonl", [
            {"input_tokens": 1, "output_tokens": 1, "tool_calls": [{"name": "shell"}, {"name": "read"}]},
            {"input_tokens": 1, "output_tokens": 1, "tool_calls": [{"name": "write"}]},
            {"input_tokens": 1, "output_tokens": 1, "tool_calls": []},
        ])
        config = make_config(tmp_path)
        agent = Agent(config)
        assert agent.total_tool_calls == 3

    def test_status_reflects_reconstructed_stats(self, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        self._write_jsonl(logs / "run.jsonl", [
            {"input_tokens": 50, "output_tokens": 100, "tool_calls": [{"name": "shell"}]},
        ])
        config = make_config(tmp_path)
        agent = Agent(config)
        s = agent.status()
        assert s["total_input_tokens"] == 50
        assert s["total_output_tokens"] == 100
        assert s["total_tool_calls"] == 1

    @pytest.mark.asyncio
    async def test_new_usage_adds_to_reconstructed(self, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        self._write_jsonl(logs / "prior.jsonl", [
            {"input_tokens": 30, "output_tokens": 60, "tool_calls": []},
        ])
        config = make_config(tmp_path)
        agent = Agent(config)
        assert agent.total_input_tokens == 30

        with patch("openalph.agent.stream") as mock:
            mock.side_effect = make_stream_events("Hi", input_tokens=10, output_tokens=5)
            await agent.handle_input("hello")

        assert agent.total_input_tokens == 40
        assert agent.total_output_tokens == 65
