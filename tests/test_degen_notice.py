"""Tests: degen detector warn-mode scoping + in-room Matrix notice (kdsn.241.21).

Three components under test:

1. **Provider-level degen_detector config** (config.py + provider.py):
   ProviderConfig gains an optional ``degen_detector`` field.  Resolution
   order at monitor instantiation: provider-level → agent-level → default
   "off".  This lets ``[providers.macstudio] degen_detector = "warn"`` turn
   the monitor on for macstudio-path requests only, without affecting
   Fireworks/Anthropic/other providers on the same agent.

2. **Agent on_degenerate callback** (agent.py):
   When a streamed or completed Response has ``.degenerate == True``, the
   agent fires ``callbacks["on_degenerate"](model, generation_id)`` if
   present.  Exceptions in the callback are swallowed (a Matrix send
   failure must not kill a healthy response).  Missing callback (subagent
   path) is a silent no-op.

3. **Matrix notice** (matrix.py):
   ``_build_agent_callbacks`` includes an ``on_degenerate`` entry that sends
   an ``m.notice`` to the room.  All Matrix rendering lives in matrix.py;
   the agent knows nothing about Matrix.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.config import AgentConfig, ProviderConfig, ConfigError
from openalph.provider import stream, StreamEvent, Response, Usage
from openalph.agent import Agent
from openalph.tools import ToolDef


# ---------------------------------------------------------------------------
# Shared helpers (adapted from test_degen_wiring.py + test_agent_streaming.py)
# ---------------------------------------------------------------------------

def _macstudio_provider(degen_detector=None):
    kw = dict(key="macstudio", type="openai", api_key="sk-test",
              base_url="http://10.0.20.104:8000/v1")
    if degen_detector is not None:
        kw["degen_detector"] = degen_detector
    return ProviderConfig(**kw)


def _fireworks_provider(degen_detector=None):
    kw = dict(key="fireworks", type="openai", api_key="sk-test",
              base_url="https://api.fireworks.ai/inference/v1")
    if degen_detector is not None:
        kw["degen_detector"] = degen_detector
    return ProviderConfig(**kw)


def _cfg(agent_degen="off", provider=None, provider_degen=None,
         default_model="macstudio/mlx-community/MiniMax-M3-4bit"):
    if provider is None:
        provider = _macstudio_provider(degen_detector=provider_degen)
    pkey = provider.key
    return AgentConfig(
        name="t",
        default_model=default_model,
        max_tokens=4096,
        providers={pkey: provider},
        workspace=Path("/tmp/test"),
        degen_detector=agent_degen,
    )


def _text_chunk(content):
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    delta.reasoning = None
    delta.reasoning_content = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None
    chunk.choices = [choice]
    chunk.usage = None
    chunk.id = "gen-1"
    return chunk


def _finish_chunk():
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = None
    delta.tool_calls = None
    delta.reasoning = None
    delta.reasoning_content = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = "stop"
    chunk.choices = [choice]
    chunk.usage = None
    chunk.id = "gen-1"
    return chunk


class CountingStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for c in self._chunks:
            yield c

    async def close(self):
        self.closed = True


async def _collect(gen):
    return [e async for e in gen]


# 60 repetitions of a 6-word phrase → arms and trips the n-gram layer.
_LOOP = ["the build is completely broken now " for _ in range(60)]


async def _run_stream(config):
    """Run a degenerate loop through provider.stream() and return the done event."""
    chunks = [_text_chunk(t) for t in _LOOP] + [_finish_chunk()]
    st = CountingStream(chunks)
    with patch("openalph.provider._get_client") as gc:
        client = MagicMock()
        gc.return_value = client
        client.chat.completions.create = AsyncMock(return_value=st)
        events = await _collect(stream(
            config=config, system="sys",
            messages=[{"role": "user", "content": "go"}],
        ))
    return [e for e in events if e.type == "done"][0]


# ---------------------------------------------------------------------------
# 1. Provider-level degen_detector scoping
# ---------------------------------------------------------------------------

class TestProviderLevelScoping:
    """Provider-level degen_detector overrides agent-level; agent-level is
    used when provider-level is unset; default is off when neither is set."""

    @pytest.mark.asyncio
    async def test_provider_warn_overrides_agent_off(self):
        """macstudio provider with degen_detector='warn' + agent 'off' → warn."""
        cfg = _cfg(agent_degen="off", provider_degen="warn")
        done = await _run_stream(cfg)
        assert done.response.degenerate is True

    @pytest.mark.asyncio
    async def test_default_off_when_neither_set(self):
        """No provider-level, agent 'off' → off (no detection)."""
        cfg = _cfg(agent_degen="off", provider_degen=None)
        done = await _run_stream(cfg)
        assert done.response.degenerate is False

    @pytest.mark.asyncio
    async def test_agent_warn_when_provider_unset(self):
        """Agent 'warn', no provider-level → warn (agent-level carries)."""
        cfg = _cfg(agent_degen="warn", provider_degen=None)
        done = await _run_stream(cfg)
        assert done.response.degenerate is True

    @pytest.mark.asyncio
    async def test_provider_off_overrides_agent_warn(self):
        """Provider 'off' explicitly disables even if agent says 'warn'."""
        cfg = _cfg(agent_degen="warn", provider_degen="off")
        done = await _run_stream(cfg)
        assert done.response.degenerate is False

    @pytest.mark.asyncio
    async def test_non_macstudio_provider_unaffected_by_default(self):
        """A fireworks provider with no degen_detector + agent 'off' → off."""
        fw = _fireworks_provider(degen_detector=None)
        cfg = _cfg(agent_degen="off", provider=fw,
                   default_model="fireworks/accounts/fireworks/models/glm-5p2")
        done = await _run_stream(cfg)
        assert done.response.degenerate is False

    @pytest.mark.asyncio
    async def test_non_macstudio_provider_warn_via_provider_level(self):
        """A fireworks provider with degen_detector='warn' → warn (generalizes)."""
        fw = _fireworks_provider(degen_detector="warn")
        cfg = _cfg(agent_degen="off", provider=fw,
                   default_model="fireworks/accounts/fireworks/models/glm-5p2")
        done = await _run_stream(cfg)
        assert done.response.degenerate is True


# ---------------------------------------------------------------------------
# 2. Agent on_degenerate callback
# ---------------------------------------------------------------------------

def _make_stream_fn(calls):
    """Fake stream() yielding different event lists per invocation."""
    call_iter = iter(calls)

    async def _fake_stream(*args, **kwargs):
        for event in next(call_iter):
            yield event

    return _fake_stream


def _make_response(content="", degenerate=False,
                   model="macstudio/mlx-community/MiniMax-M3-4bit",
                   generation_id="gen-test-1"):
    r = Response(
        content=content,
        model=model,
        usage=Usage(input_tokens=100, output_tokens=50),
        stop_reason="end_turn",
        generation_id=generation_id,
    )
    r.degenerate = degenerate
    return r


def _make_agent_config(tmp_path):
    return AgentConfig(
        name="test-agent",
        default_model="macstudio/mlx-community/MiniMax-M3-4bit",
        max_tokens=8192,
        providers={"macstudio": _macstudio_provider()},
        max_iterations=25,
        truncation_limit=50000,
        workspace=tmp_path,
    )


class TestOnDegenerateCallback:

    @pytest.mark.asyncio
    async def test_fires_when_response_degenerate(self, tmp_path):
        """on_degenerate is called when response.degenerate is True."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        resp = _make_response(content="loop loop loop", degenerate=True)
        with patch("openalph.agent.stream", new=_make_stream_fn([
            [StreamEvent(type="text", content="loop loop loop"),
             StreamEvent(type="done", response=resp, stop_reason="end_turn",
                         model="macstudio/mlx-community/MiniMax-M3-4bit")],
        ])):
            cb = AsyncMock()
            await agent.handle_input(
                "Hi", callbacks={"on_degenerate": cb},
            )

        assert cb.called
        # Callback receives model and generation_id for log cross-referencing.
        call_kwargs = cb.call_args
        assert "macstudio" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_not_fired_when_clean(self, tmp_path):
        """on_degenerate is NOT called when response.degenerate is False."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        resp = _make_response(content="Hello world", degenerate=False)
        with patch("openalph.agent.stream", new=_make_stream_fn([
            [StreamEvent(type="text", content="Hello world"),
             StreamEvent(type="done", response=resp, stop_reason="end_turn",
                         model="macstudio/mlx-community/MiniMax-M3-4bit")],
        ])):
            cb = AsyncMock()
            result = await agent.handle_input(
                "Hi", callbacks={"on_degenerate": cb},
            )

        assert not cb.called
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_missing_callback_no_error(self, tmp_path):
        """No on_degenerate in callbacks (subagent path) → no error, response returned."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        resp = _make_response(content="loop", degenerate=True)
        with patch("openalph.agent.stream", new=_make_stream_fn([
            [StreamEvent(type="text", content="loop"),
             StreamEvent(type="done", response=resp, stop_reason="end_turn",
                         model="macstudio/mlx-community/MiniMax-M3-4bit")],
        ])):
            # No callbacks dict at all
            result = await agent.handle_input("Hi")

        assert result == "loop"

    @pytest.mark.asyncio
    async def test_callback_exception_doesnt_kill_turn(self, tmp_path):
        """If on_degenerate raises, the response is still returned."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        resp = _make_response(content="loop loop", degenerate=True)
        with patch("openalph.agent.stream", new=_make_stream_fn([
            [StreamEvent(type="text", content="loop loop"),
             StreamEvent(type="done", response=resp, stop_reason="end_turn",
                         model="macstudio/mlx-community/MiniMax-M3-4bit")],
        ])):
            cb = AsyncMock(side_effect=RuntimeError("matrix send failed"))
            result = await agent.handle_input(
                "Hi", callbacks={"on_degenerate": cb},
            )

        assert cb.called
        assert result == "loop loop"

    @pytest.mark.asyncio
    async def test_fires_on_nonstreaming_complete_fallback(self, tmp_path):
        """The complete() fallback path also sets .degenerate and fires the callback."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        resp = _make_response(content="loop loop", degenerate=True)
        with patch("openalph.agent.stream", new=_make_stream_fn([
            # stream() yields nothing → agent falls back to complete()
            [],
        ])), patch("openalph.agent.complete",
                   new=AsyncMock(return_value=resp)):
            cb = AsyncMock()
            result = await agent.handle_input(
                "Hi", callbacks={"on_degenerate": cb},
            )

        assert cb.called
        assert result == "loop loop"


# ---------------------------------------------------------------------------
# 3. Matrix notice delivery
# ---------------------------------------------------------------------------

class TestMatrixNotice:
    """_build_agent_callbacks includes on_degenerate, which sends an m.notice."""

    @pytest.mark.asyncio
    async def test_callbacks_include_on_degenerate(self):
        """The callbacks dict from _build_agent_callbacks has an on_degenerate key."""
        from openalph.matrix import MatrixBot
        from openalph.config import MatrixConfig

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = MagicMock()
        bot.config.user_id = "@test:server"
        bot.agent = MagicMock()
        bot.agent._read_registries = {}
        bot.agent._advisor_uses = {}
        bot._advisor_results = {}
        bot._subagent_results = {}
        bot.session_log = None
        bot._room_send_with_retry = AsyncMock()

        callbacks = MatrixBot._build_agent_callbacks(bot, "!room:server", None)
        assert "on_degenerate" in callbacks

    @pytest.mark.asyncio
    async def test_notice_sent_with_correct_msgtype(self):
        """Calling on_degenerate sends an m.notice to the room."""
        from openalph.matrix import MatrixBot

        bot = MatrixBot.__new__(MatrixBot)
        bot.config = MagicMock()
        bot.config.user_id = "@test:server"
        bot.agent = MagicMock()
        bot.agent._read_registries = {}
        bot.agent._advisor_uses = {}
        bot._advisor_results = {}
        bot._subagent_results = {}
        bot.session_log = None
        bot._room_send_with_retry = AsyncMock()

        callbacks = MatrixBot._build_agent_callbacks(bot, "!room:server", None)
        on_degenerate = callbacks["on_degenerate"]

        await on_degenerate(
            model="macstudio/mlx-community/MiniMax-M3-4bit",
            generation_id="gen-test-42",
        )

        bot._room_send_with_retry.assert_called_once()
        call_args = bot._room_send_with_retry.call_args
        room_id = call_args.args[0]
        content = call_args.args[1]
        assert room_id == "!room:server"
        assert content["msgtype"] == "m.notice"
        assert "macstudio" in content["body"] or "MiniMax" in content["body"]


# ---------------------------------------------------------------------------
# 4. Mid-stream degenerate event (kdsn.241.21 — real-time notice)
# ---------------------------------------------------------------------------

class TestMidStreamDegenerateEvent:
    """The provider yields a 'degenerate' StreamEvent at the trip point (not
    after done), so the agent can fire on_degenerate mid-stream — while there's
    still time for the operator to intervene on a runaway generation."""

    @pytest.mark.asyncio
    async def test_provider_yields_degenerate_event_mid_stream(self):
        """provider.stream() yields a type='degenerate' event when the monitor
        trips, BEFORE the done event."""
        cfg = _cfg(agent_degen="off", provider_degen="warn")
        events = []
        chunks = [_text_chunk(t) for t in _LOOP] + [_finish_chunk()]
        st = CountingStream(chunks)
        with patch("openalph.provider._get_client") as gc:
            client = MagicMock()
            gc.return_value = client
            client.chat.completions.create = AsyncMock(return_value=st)
            events = await _collect(stream(
                config=cfg, system="sys",
                messages=[{"role": "user", "content": "go"}],
            ))
        degen_events = [e for e in events if e.type == "degenerate"]
        done_events = [e for e in events if e.type == "done"]
        assert len(degen_events) == 1  # exactly once
        assert len(done_events) == 1
        # Degenerate event comes before done
        assert events.index(degen_events[0]) < events.index(done_events[0])
        # Carries model + generation_id for the notice
        assert "MiniMax" in degen_events[0].model

    @pytest.mark.asyncio
    async def test_agent_fires_callback_mid_stream(self, tmp_path):
        """When the agent receives a 'degenerate' stream event, it fires
        on_degenerate immediately — before the done event."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        degenerate_event = StreamEvent(
            type="degenerate",
            model="macstudio/mlx-community/MiniMax-M3-4bit",
            generation_id="gen-mid-1",
        )
        resp = _make_response(content="loop loop", degenerate=True,
                              generation_id="gen-mid-1")
        with patch("openalph.agent.stream", new=_make_stream_fn([
            [StreamEvent(type="text", content="loop loop"),
             degenerate_event,
             StreamEvent(type="done", response=resp, stop_reason="end_turn",
                         model="macstudio/mlx-community/MiniMax-M3-4bit")],
        ])):
            cb = AsyncMock()
            result = await agent.handle_input(
                "Hi", callbacks={"on_degenerate": cb},
            )

        assert cb.called
        assert cb.call_count == 1  # exactly once — mid-stream, not double-fired
        assert result == "loop loop"

    @pytest.mark.asyncio
    async def test_no_double_fire_mid_stream_then_backstop(self, tmp_path):
        """When both a mid-stream degenerate event AND response.degenerate=True
        are present, the callback fires exactly ONCE (mid-stream), not twice."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        degenerate_event = StreamEvent(
            type="degenerate",
            model="macstudio/mlx-community/MiniMax-M3-4bit",
            generation_id="gen-mid-2",
        )
        resp = _make_response(content="loop", degenerate=True,
                              generation_id="gen-mid-2")
        with patch("openalph.agent.stream", new=_make_stream_fn([
            [StreamEvent(type="text", content="loop"),
             degenerate_event,
             StreamEvent(type="done", response=resp, stop_reason="end_turn",
                         model="macstudio/mlx-community/MiniMax-M3-4bit")],
        ])):
            cb = AsyncMock()
            await agent.handle_input(
                "Hi", callbacks={"on_degenerate": cb},
            )

        assert cb.call_count == 1  # mid-stream fires, post-stream backstop skips

    @pytest.mark.asyncio
    async def test_backstop_still_fires_for_complete_fallback(self, tmp_path):
        """When NO mid-stream degenerate event is present (complete() fallback),
        the post-stream backstop still fires the callback."""
        config = _make_agent_config(tmp_path)
        agent = Agent(config)

        resp = _make_response(content="loop loop", degenerate=True,
                              generation_id="gen-back-1")
        with patch("openalph.agent.stream", new=_make_stream_fn([
            # stream() yields nothing → complete() fallback; no degenerate event
            [],
        ])), patch("openalph.agent.complete",
                   new=AsyncMock(return_value=resp)):
            cb = AsyncMock()
            result = await agent.handle_input(
                "Hi", callbacks={"on_degenerate": cb},
            )

        assert cb.called
        assert cb.call_count == 1
        assert result == "loop loop"
