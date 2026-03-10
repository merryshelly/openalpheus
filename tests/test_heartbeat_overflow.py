"""Tests for heartbeat auto-stop on context overflow.

When agent.handle_input raises ContextOverflowError during a heartbeat,
the heartbeat should auto-stop for that room and send a single clear notice.
Generic exceptions should NOT trigger auto-stop.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.heartbeat import HeartbeatManager
from openalph.matrix import MatrixBot
from openalph.config import AgentConfig, MatrixConfig
from openalph.agent import ContextOverflowError


# --- Helpers ---


def make_matrix_config(**kwargs):
    defaults = dict(
        homeserver="https://matrix.local",
        user_id="@watson:matrix.local",
        device_id="TEST",
        password="test-password",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
        rooms=None,
    )
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def make_agent_config(workspace, **kwargs):
    defaults = dict(
        name="watson",
        model="test-model",
        max_tokens=8192,
        provider="anthropic",
        api_key="sk-test",
        base_url=None,
        workspace=workspace,
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_bot(tmp_path):
    """Create a MatrixBot with mocked internals."""
    matrix_config = make_matrix_config()
    agent_config = make_agent_config(workspace=tmp_path)

    agent = MagicMock()
    agent.config = agent_config
    agent.handle_input = AsyncMock(return_value="Agent response")
    agent.cancel = MagicMock()
    agent._on_tool_call = None
    agent._on_tool_intent = None

    with patch("openalph.matrix.AsyncClient"):
        bot = MatrixBot(agent, matrix_config)

    bot._synced = True
    bot.send = AsyncMock()
    bot._set_typing = AsyncMock()
    bot.session_log = None

    return bot, agent


# --- Tests ---


class TestHeartbeatOverflow:
    @pytest.mark.asyncio
    async def test_heartbeat_auto_stops_on_overflow(self, tmp_path):
        """On ContextOverflowError, heartbeat.stop(room_id) is called and notice sent."""
        bot, agent = make_bot(tmp_path)
        room_id = "!overflow:matrix.local"

        agent.handle_input.side_effect = ContextOverflowError(150000, 128000)
        bot.heartbeat.stop = AsyncMock()
        bot._active_rooms.add(room_id)

        # Should not raise
        await bot._inject_heartbeat(room_id)

        bot.heartbeat.stop.assert_awaited_once_with(room_id)
        assert bot.send.await_count == 1
        sent_msg = bot.send.call_args[0][1]
        assert "auto-stopped" in sent_msg

    @pytest.mark.asyncio
    async def test_heartbeat_overflow_sends_single_notice(self, tmp_path):
        """Each _inject_heartbeat call on overflow sends exactly one notice and calls stop."""
        bot, agent = make_bot(tmp_path)
        room_id = "!overflow2:matrix.local"

        agent.handle_input.side_effect = ContextOverflowError(150000, 128000)
        bot.heartbeat.stop = AsyncMock()
        bot._active_rooms.add(room_id)

        # Simulate two rapid fires
        await bot._inject_heartbeat(room_id)
        await bot._inject_heartbeat(room_id)

        assert bot.heartbeat.stop.await_count == 2
        assert bot.send.await_count == 2
        for call in bot.send.call_args_list:
            assert "auto-stopped" in call[0][1]

    @pytest.mark.asyncio
    async def test_heartbeat_non_overflow_error_keeps_running(self, tmp_path):
        """Generic RuntimeError should NOT call heartbeat.stop."""
        bot, agent = make_bot(tmp_path)
        room_id = "!generic_err:matrix.local"

        agent.handle_input.side_effect = RuntimeError("something broke")
        bot.heartbeat.stop = AsyncMock()
        bot._active_rooms.add(room_id)

        await bot._inject_heartbeat(room_id)

        bot.heartbeat.stop.assert_not_awaited()
        assert bot.send.await_count == 1
        sent_msg = bot.send.call_args[0][1]
        assert "auto-stopped" not in sent_msg
