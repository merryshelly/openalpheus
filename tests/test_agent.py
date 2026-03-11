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
from openalph.provider import Response, Usage


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

        with patch("openalph.agent.complete", new_callable=AsyncMock) as mock:
            mock.return_value = make_response("Hi there!")
            result = await agent.handle_input("Hello")

        assert result == "Hi there!"

    @pytest.mark.asyncio
    async def test_history_accumulates(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.complete", new_callable=AsyncMock) as mock:
            mock.return_value = make_response("Response 1")
            await agent.handle_input("Message 1")

            mock.return_value = make_response("Response 2")
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

        with patch("openalph.agent.complete", new_callable=AsyncMock) as mock:
            mock.return_value = make_response("OK")
            await agent.handle_input("Hi")

        kw = mock.call_args.kwargs
        assert "I am the soul." in kw["system"]

    @pytest.mark.asyncio
    async def test_config_passed_to_provider(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.complete", new_callable=AsyncMock) as mock:
            mock.return_value = make_response("OK")
            await agent.handle_input("Hi")

        kw = mock.call_args.kwargs
        assert kw["config"] is config

    @pytest.mark.asyncio
    async def test_full_history_passed_to_provider(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.complete", new_callable=AsyncMock) as mock:
            mock.return_value = make_response("First")
            await agent.handle_input("Hello")

            mock.return_value = make_response("Second")
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

        with patch("openalph.agent.complete", new_callable=AsyncMock) as mock:
            mock.return_value = make_response("A", input_tokens=100, output_tokens=50)
            await agent.handle_input("First")

            mock.return_value = make_response("B", input_tokens=200, output_tokens=80)
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

        with patch("openalph.agent.complete", new_callable=AsyncMock) as mock:
            mock.return_value = make_response("Hi", input_tokens=50, output_tokens=20)
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
            with patch("openalph.agent.complete", new_callable=AsyncMock) as mock:
                await agent.handle_input(big_message)

        # The big message must NOT be in history
        contents = [m.get("content", "") for m in agent.history("_default")]
        assert big_message not in contents
