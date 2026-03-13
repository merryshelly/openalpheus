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


# --- Session-Scoped Usage Counters ---


class TestSessionScopedUsage:

    def test_counters_start_at_zero(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)
        assert agent.total_input_tokens == 0
        assert agent.total_output_tokens == 0
        assert agent.total_tool_calls == 0

    def test_counters_ignore_prior_logs(self, tmp_path):
        """Existing JSONL logs do NOT inflate session counters."""
        import json
        logs = tmp_path / "logs"
        logs.mkdir()
        with open(logs / "prior.jsonl", "w") as f:
            f.write(json.dumps({"input_tokens": 999, "output_tokens": 888, "tool_calls": []}) + "\n")
        config = make_config(tmp_path)
        agent = Agent(config)
        assert agent.total_input_tokens == 0
        assert agent.total_output_tokens == 0

    @pytest.mark.asyncio
    async def test_counters_accumulate_within_session(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.agent.stream") as mock:
            mock.side_effect = make_stream_events("A", input_tokens=100, output_tokens=50)
            await agent.handle_input("First")
            mock.side_effect = make_stream_events("B", input_tokens=200, output_tokens=80)
            await agent.handle_input("Second")

        assert agent.total_input_tokens == 300
        assert agent.total_output_tokens == 130

    def test_status_reflects_session_counters(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)
        s = agent.status()
        assert s["total_input_tokens"] == 0
        assert s["total_output_tokens"] == 0
        assert s["total_tool_calls"] == 0


# --- Per-Room Model Override (bead .83) ---


class TestPerRoomModelOverride:

    def test_default_fallback_no_override(self, tmp_path):
        config = make_config(tmp_path, default_model="test/default-model")
        agent = Agent(config)

        assert agent.get_model("room_a") == "test/default-model"
        assert agent.get_model("room_b") == "test/default-model"

    def test_switch_model_sets_per_room(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.config.resolve_model"):
            result = agent.switch_model("other/model-x", "room_a")

        assert result is None
        assert agent.get_model("room_a") == "other/model-x"

    def test_per_room_isolation(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.config.resolve_model"):
            agent.switch_model("other/model-a", "room_a")
            agent.switch_model("other/model-b", "room_b")

        assert agent.get_model("room_a") == "other/model-a"
        assert agent.get_model("room_b") == "other/model-b"
        assert agent.get_model("room_c") == config.default_model

    def test_status_shows_per_room_model(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.config.resolve_model"):
            agent.switch_model("other/room-model", "room_a")

        status_a = agent.status("room_a")
        status_b = agent.status("room_b")

        assert status_a["model"] == "other/room-model"
        assert status_b["model"] == config.default_model

    @pytest.mark.asyncio
    async def test_override_persists_across_calls(self, tmp_path):
        config = make_config(tmp_path)
        agent = Agent(config)

        with patch("openalph.config.resolve_model"):
            agent.switch_model("persistent/model", "test_room")

        with patch("openalph.agent.stream") as mock:
            mock.side_effect = make_stream_events("r1")
            await agent.handle_input("msg 1", "test_room")

            mock.side_effect = make_stream_events("r2")
            await agent.handle_input("msg 2", "test_room")

        for call in mock.call_args_list:
            assert call.kwargs["model"] == "persistent/model"

    def test_new_agent_has_no_overrides(self, tmp_path):
        """Process restart (new Agent instance) resets all rooms to default."""
        config = make_config(tmp_path, default_model="original/default")

        agent1 = Agent(config)
        with patch("openalph.config.resolve_model"):
            agent1.switch_model("override/model", "room_a")
        assert agent1.get_model("room_a") == "override/model"

        agent2 = Agent(config)
        assert agent2.get_model("room_a") == "original/default"

    def test_vision_guard_per_room(self, tmp_path):
        """Vision guard only blocks the room that has images, not other rooms."""
        config = make_config(tmp_path)
        agent = Agent(config)

        # Room A has images in history
        agent._rooms["room_a"] = [
            {"role": "user", "content": [
                {"type": "image", "media_type": "image/png", "data": "fakedata"},
            ]},
        ]
        # Room B has no images
        agent._rooms["room_b"] = [
            {"role": "user", "content": "just text"},
        ]

        with patch("openalph.config.resolve_model"):
            result_a = agent.switch_model("other/model", "room_a")
            result_b = agent.switch_model("other/model", "room_b")

        assert result_a is not None  # blocked
        assert "image" in result_a.lower() or "vision" in result_a.lower()
        assert result_b is None  # allowed
        assert agent.get_model("room_a") == config.default_model  # unchanged
        assert agent.get_model("room_b") == "other/model"  # set


# --- Tool Call Limit Summary (bead .83 follow-up) ---


class TestToolCallLimitSummary:

    @staticmethod
    def _give_agent_tools(agent):
        """Give agent a fake tool so tools_arg is not None in the loop."""
        from openalph.tools import ToolDef
        agent.tools = [ToolDef(name="shell", description="Run shell", parameters={}, config={})]

    @pytest.mark.asyncio
    async def test_limit_returns_summary_not_static_message(self, tmp_path):
        """When tool limit is hit, agent makes a final LLM call for a summary."""
        from openalph.tools import ToolResult
        from openalph.provider import ToolCall as TC

        config = make_config(tmp_path, max_iterations=1)
        agent = Agent(config)
        self._give_agent_tools(agent)
        tool_result = ToolResult(content="ok", is_error=False)
        tc = TC(id="tc_1", name="shell", input={"command": "echo hi"})
        call_count = 0

        async def _mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("tools") is not None:
                yield StreamEvent(type="tool_done", tool_call=tc)
                yield StreamEvent(
                    type="done",
                    response=Response(
                        content="", model="test",
                        usage=Usage(input_tokens=10, output_tokens=5),
                        stop_reason="tool_use", tool_calls=[tc],
                    ),
                    stop_reason="tool_use", model="test",
                )
            else:
                yield StreamEvent(type="text", content="Here is my summary.")
                yield StreamEvent(
                    type="done",
                    response=Response(
                        content="Here is my summary.", model="test",
                        usage=Usage(input_tokens=20, output_tokens=10),
                        stop_reason="end_turn",
                    ),
                    stop_reason="end_turn", model="test",
                )

        with patch("openalph.agent.stream", _mock_stream), \
             patch("openalph.agent.execute_tool", return_value=tool_result):
            result = await agent.handle_input("Do something", "_default")

        assert "summary" in result.lower()
        assert call_count == 2  # one tool iteration + one summary call

    @pytest.mark.asyncio
    async def test_limit_summary_streams_through_callback(self, tmp_path):
        """Summary turn fires on_text_delta so streaming delivery works."""
        from openalph.tools import ToolResult
        from openalph.provider import ToolCall as TC

        config = make_config(tmp_path, max_iterations=1)
        agent = Agent(config)
        self._give_agent_tools(agent)
        tool_result = ToolResult(content="ok", is_error=False)
        tc = TC(id="tc_1", name="shell", input={"command": "echo"})
        streamed_deltas = []

        async def _on_text_delta(text, done=False):
            streamed_deltas.append((text, done))

        async def _mock_stream(*args, **kwargs):
            if kwargs.get("tools") is not None:
                yield StreamEvent(type="tool_done", tool_call=tc)
                yield StreamEvent(
                    type="done",
                    response=Response(
                        content="", model="test",
                        usage=Usage(input_tokens=10, output_tokens=5),
                        stop_reason="tool_use", tool_calls=[tc],
                    ),
                    stop_reason="tool_use", model="test",
                )
            else:
                yield StreamEvent(type="text", content="Summary content")
                yield StreamEvent(
                    type="done",
                    response=Response(
                        content="Summary content", model="test",
                        usage=Usage(input_tokens=20, output_tokens=10),
                        stop_reason="end_turn",
                    ),
                    stop_reason="end_turn", model="test",
                )

        with patch("openalph.agent.stream", _mock_stream), \
             patch("openalph.agent.execute_tool", return_value=tool_result):
            result = await agent.handle_input(
                "Do work", "_default", on_text_delta=_on_text_delta
            )

        text_deltas = [t for t, d in streamed_deltas if t and not d]
        done_signals = [d for t, d in streamed_deltas if d]
        assert any("Summary" in t for t in text_deltas)
        assert len(done_signals) >= 1

    @pytest.mark.asyncio
    async def test_limit_summary_failure_returns_fallback(self, tmp_path):
        """If summary generation fails, a static error message is returned."""
        from openalph.tools import ToolResult
        from openalph.provider import ToolCall as TC

        config = make_config(tmp_path, max_iterations=1)
        agent = Agent(config)
        self._give_agent_tools(agent)
        tool_result = ToolResult(content="ok", is_error=False)
        tc = TC(id="tc_1", name="shell", input={"command": "echo"})

        async def _mock_stream(*args, **kwargs):
            if kwargs.get("tools") is not None:
                yield StreamEvent(type="tool_done", tool_call=tc)
                yield StreamEvent(
                    type="done",
                    response=Response(
                        content="", model="test",
                        usage=Usage(input_tokens=10, output_tokens=5),
                        stop_reason="tool_use", tool_calls=[tc],
                    ),
                    stop_reason="tool_use", model="test",
                )
            else:
                raise RuntimeError("Provider exploded")
                yield  # make it a generator

        with patch("openalph.agent.stream", _mock_stream), \
             patch("openalph.agent.execute_tool", return_value=tool_result):
            result = await agent.handle_input("Do work", "_default")

        assert "Tool call limit reached" in result
        assert "failed" in result.lower()
