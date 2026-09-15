"""Mid-stream provider retry with display reset (kdsn.345).

Contract under test — SB-ratified 2026-09-15 (reverses the kdsn.287
"mid-stream = fatal" residual; measured basis: 21/11,514 mid-stream
failures Aug 28–Sep 15, ALL with >=1 chunk yielded, all Synthetic):

  1. provider.stream() gains an optional ``on_stream_reset`` async
     callback param. On a mid-stream httpx transport error (RemoteProtocolError /
     TimeoutException and kin) with >=1 chunk consumed:
       - seam UNWIRED, ``retry_enabled=False``, retry budget exhausted,
         or a mid-stream SDK APIStatusError  -> ProviderError (today's behavior)
       - seam wired + budget remaining       -> log, sleep, fire
         on_stream_reset(info), re-issue the FULL request from scratch.
       - zero-yield failures are NEVER retried at the OA layer (the SDK's
         max_retries=20 owns establishment; kdsn.220).
  2. Info dict: {"error": str, "chunks": int, "attempt": int}.
  3. Per-attempt state is FULLY reset between attempts: stream counters,
     hardened byte counters, text/reasoning accumulators, usage,
     tool_call_accumulators, degen monitor (RE-INSTANTIATED — stale
     n-gram state from the dead attempt must not contaminate the retry).
  4. agent.py passes the provider param ONLY when the callbacks dict
     carries 'on_stream_reset' (live + heartbeat paths). Subagent / CLI
     / keepalive paths stay fatal-today.
  5. agent-side reset closure clears accumulated_text / accumulated_thinking /
     response / text_emitted / thinking_emitted / usage / tool_calls /
     _degenerate_notified, then forwards to callbacks['on_stream_reset'].
  6. matrix live path: callbacks['on_stream_reset'] annotates the in-flight
     partial Matrix message (cursor removed, " ⟳" appended) fail-soft,
     clears StreamingDelivery state (retry starts a FRESH message), and
     sends one retry notice. Heartbeat/umbral path: notice-only (no
     streaming display on background turns).

Hard-coded policy (deliberately minimal — kdsn.219 died on config surface):
  _MIDSTREAM_MAX_RETRIES = 1 extra attempt, _MIDSTREAM_RETRY_DELAY_S = 5.0,
  single per-provider ``retry_enabled`` kill switch (default True).
  CancelledError during the backoff sleep propagates (/stop stays live).
"""

import asyncio
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import openalph.provider as provider_module
from openalph.config import AgentConfig, MatrixConfig, ProviderConfig
from openalph.matrix import MatrixBot, StreamingDelivery
from openalph.provider import ProviderError, Response, Usage, stream
from openalph.agent import Agent

ROOM = "!test:matrix.local"


# ---------------------------------------------------------------------------
# Provider-level fixtures (mirrors test_provider.py conventions)
# ---------------------------------------------------------------------------

def make_provider(key="synthetic", type="openai", api_key="sk-test",
                  base_url="https://api.synthetic.new/openai/v1", **extra):
    return ProviderConfig(
        key=key, type=type, api_key=api_key, base_url=base_url, **extra,
    )


def make_config(**kwargs):
    defaults = {
        "name": "test-agent",
        "default_model": "synthetic/hf:moonshotai/Kimi-K3",
        "max_tokens": 8192,
        "providers": {"synthetic": make_provider()},
        "workspace": pathlib.Path("/tmp/test"),
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def openai_text_chunk(text, finish_reason=None):
    chunk = MagicMock()
    chunk.id = "gen-1"
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = text
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].delta.reasoning = None
    chunk.choices[0].delta.reasoning_content = None
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = None
    return chunk


def openai_usage_chunk(prompt_tokens=100, completion_tokens=50):
    chunk = MagicMock()
    chunk.id = "gen-1"
    chunk.choices = []
    chunk.usage = MagicMock()
    chunk.usage.prompt_tokens = prompt_tokens
    chunk.usage.completion_tokens = completion_tokens
    return chunk


def openai_tool_delta_chunk(name="shell", args='{"com'):
    chunk = MagicMock()
    chunk.id = "gen-1"
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = None
    chunk.choices[0].delta.reasoning = None
    chunk.choices[0].delta.reasoning_content = None
    chunk.choices[0].finish_reason = None
    chunk.choices[0].delta.tool_calls = [MagicMock()]
    tc = chunk.choices[0].delta.tool_calls[0]
    tc.index = 0
    tc.id = "call_1"
    tc.function.name = name
    tc.function.arguments = args
    tc.extra_content = None
    chunk.usage = None
    return chunk


class _FakeOpenAIStream:
    """Yields chunks then raises exc (kdsn.287 mock shape)."""

    def __init__(self, chunks, exc):
        self._chunks = chunks
        self._exc = exc

    def __aiter__(self):
        return self._impl()

    async def _impl(self):
        for chunk in self._chunks:
            yield chunk
        raise self._exc


class _SuccessOpenAIStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._impl()

    async def _impl(self):
        for chunk in self._chunks:
            yield chunk


# --- anthropic fixtures ---

def _anthropic_text(text):
    e = MagicMock()
    e.type = "text"
    e.text = text
    return e


def _anthropic_message_stop():
    e = MagicMock()
    e.type = "message_stop"
    return e


def _anthropic_final_message(text=""):
    msg = MagicMock()
    msg.model = "test-model"
    msg.stop_reason = "end_turn"
    msg.usage.input_tokens = 100
    msg.usage.output_tokens = 50
    msg.usage.cache_read_input_tokens = 0
    msg.usage.cache_creation_input_tokens = 0
    tb = MagicMock()
    tb.type = "text"
    tb.text = text
    msg.content = [tb] if text else []
    return msg


class _RaisingAnthropicStream:
    def __init__(self, events, exc, final_message=None):
        self._events = events
        self._exc = exc
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._impl()

    async def _impl(self):
        for event in self._events:
            yield event
        raise self._exc

    async def get_final_message(self):
        return self._final_message


class _SuccessAnthropicStream:
    def __init__(self, events, final_message):
        self._events = events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._impl()

    async def _impl(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final_message


async def collect_events(gen):
    return [e async for e in gen]


def _anthropic_cfg():
    return make_config(
        providers={
            "anthropic": make_provider(
                key="anthropic", type="anthropic", api_key="sk-test",
                base_url=None,
            )
        },
        default_model="anthropic/claude-sonnet-4-20250514",
    )


# ---------------------------------------------------------------------------
# Provider layer — openai family
# ---------------------------------------------------------------------------

class TestMidStreamRetryOpenAI:
    def _cfg(self, **prov_extra):
        return make_config(
            providers={"synthetic": make_provider(**prov_extra)},
        )

    @pytest.mark.asyncio
    async def test_retry_then_success_remote_protocol_error(self):
        """kdsn.345 — RED. Mid-stream RemoteProtocolError with the seam wired:
        reset fires once, request is re-issued, final response is attempt-2
        only (no attempt-1 partial text), done event present."""
        config = self._cfg()
        reset_cb = AsyncMock()
        failing = _FakeOpenAIStream(
            [openai_text_chunk("partial")],
            httpx.RemoteProtocolError("peer closed connection"),
        )
        ok = _SuccessOpenAIStream([openai_text_chunk("full answer", "stop"),
                                       openai_usage_chunk()])

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(
                side_effect=[failing, ok])
            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                on_stream_reset=reset_cb,
            ))

        assert client.chat.completions.create.call_count == 2
        reset_cb.assert_awaited_once()
        info = reset_cb.await_args.args[0]
        assert info["chunks"] == 1
        assert "RemoteProtocolError" in info["error"]
        assert [e.type for e in events][-1] == "done"
        response = events[-1].response
        assert response.content == "full answer"

    @pytest.mark.asyncio
    async def test_retry_then_success_read_timeout(self):
        """kdsn.345 — RED. Mid-stream ReadTimeout retries under the same policy."""
        config = self._cfg()
        reset_cb = AsyncMock()
        failing = _FakeOpenAIStream(
            [openai_text_chunk("part")], httpx.ReadTimeout("stalled"))
        ok = _SuccessOpenAIStream([openai_text_chunk("complete", "stop"),
                                       openai_usage_chunk()])

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(
                side_effect=[failing, ok])
            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                on_stream_reset=reset_cb,
            ))

        assert client.chat.completions.create.call_count == 2
        reset_cb.assert_awaited_once()
        assert events[-1].type == "done"
        assert events[-1].response.content == "complete"

    @pytest.mark.asyncio
    async def test_event_sequence_partial_reset_full_done(self):
        """kdsn.345 — RED. Caller sees attempt-1 partial events, then the
        reset callback, then attempt-2 events: text, reset, text, done."""
        config = self._cfg()
        timeline = []

        async def recording_reset(info):
            timeline.append("reset")

        failing = _FakeOpenAIStream(
            [openai_text_chunk("par")],
            httpx.RemoteProtocolError("dropped"))
        ok = _SuccessOpenAIStream([openai_text_chunk("done-text", "stop"),
                                       openai_usage_chunk()])

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(
                side_effect=[failing, ok])

            async def collecting():
                async for e in stream(config=config, system="Test",
                                      messages=[{"role": "user", "content": "Hi"}],
                                      on_stream_reset=recording_reset):
                    timeline.append(("event", e.type))

            await collecting()

        assert timeline == [("event", "text"), "reset",
                            ("event", "text"), ("event", "done")]

    @pytest.mark.asyncio
    async def test_backoff_sleep_before_retry(self):
        """kdsn.345 — RED. Retry sleeps _MIDSTREAM_RETRY_DELAY_S (sleep
        patched out) before re-issuing."""
        config = self._cfg()
        reset_cb = AsyncMock()
        failing = _FakeOpenAIStream(
            [openai_text_chunk("p")], httpx.RemoteProtocolError("x"))
        ok = _SuccessOpenAIStream([openai_text_chunk("ok", "stop"),
                                       openai_usage_chunk()])

        with patch("openalph.provider._get_client") as mock_gc, \
             patch("openalph.provider.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(
                side_effect=[failing, ok])
            await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                on_stream_reset=reset_cb,
            ))

        mock_sleep.assert_awaited_once_with(
            provider_module._MIDSTREAM_RETRY_DELAY_S)

    @pytest.mark.asyncio
    async def test_exhaustion_raises_after_second_failure(self):
        """kdsn.345 — RED. Both attempts fail mid-stream: ProviderError from
        attempt 2, reset fired exactly once (before attempt 2 only)."""
        config = self._cfg()
        reset_cb = AsyncMock()
        fail1 = _FakeOpenAIStream(
            [openai_text_chunk("one")], httpx.RemoteProtocolError("drop 1"))
        fail2 = _FakeOpenAIStream(
            [openai_text_chunk("two"), openai_text_chunk("x")],
            httpx.RemoteProtocolError("drop 2"))

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(
                side_effect=[fail1, fail2])
            with pytest.raises(ProviderError,
                               match=r"mid-stream after 2 chunk\(s\)"):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                    on_stream_reset=reset_cb,
                ))

        assert client.chat.completions.create.call_count == 2
        reset_cb.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fatal_when_seam_unwired(self):
        """kdsn.345 — guard (green pre/post). No on_stream_reset param:
        mid-stream failure stays fatal, single attempt."""
        config = self._cfg()
        failing = _FakeOpenAIStream(
            [openai_text_chunk("partial")],
            httpx.RemoteProtocolError("dropped"))

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(return_value=failing)
            with pytest.raises(ProviderError,
                               match=r"mid-stream after 1 chunk\(s\)"):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                ))

        assert client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_fatal_when_retry_enabled_false(self):
        """kdsn.345 — guard (green pre/post). Per-provider kill switch:
        retry_enabled=False with the seam wired still fails fast."""
        config = self._cfg(retry_enabled=False)
        reset_cb = AsyncMock()
        failing = _FakeOpenAIStream(
            [openai_text_chunk("partial")],
            httpx.RemoteProtocolError("dropped"))

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(return_value=failing)
            with pytest.raises(ProviderError, match="mid-stream"):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                    on_stream_reset=reset_cb,
                ))

        assert client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_zero_yield_never_retried_even_with_seam(self):
        """kdsn.345 — guard (green pre/post). Zero-yield failures stay
        OA-fatal: the SDK's max_retries=20 owns establishment (kdsn.220);
        an OA retry would stack on top of a ~2min SDK window."""
        config = self._cfg()
        reset_cb = AsyncMock()
        failing = _FakeOpenAIStream(
            [], httpx.RemoteProtocolError("peer closed connection"))

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(return_value=failing)
            with pytest.raises(ProviderError,
                               match=r"before streaming any data"):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                    on_stream_reset=reset_cb,
                ))

        assert client.chat.completions.create.call_count == 1
        reset_cb.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_midstream_api_status_error_not_retried(self):
        """kdsn.345 — guard (green pre/post). Mid-stream SDK APIStatusError
        (e.g. 529 delivered as an SSE error event) is out of scope: fatal,
        single attempt."""
        import openai as openai_sdk
        config = self._cfg()
        reset_cb = AsyncMock()
        failing = _FakeOpenAIStream(
            [openai_text_chunk("partial")],
            openai_sdk.APIStatusError(
                message="overloaded",
                response=MagicMock(status_code=529),
                body=None,
            ),
        )

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(return_value=failing)
            with pytest.raises(ProviderError):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                    on_stream_reset=reset_cb,
                ))

        assert client.chat.completions.create.call_count == 1
        reset_cb.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancellation_during_backoff_propagates(self):
        """kdsn.345 — RED. /stop during the backoff sleep must propagate
        CancelledError, never swallow or convert it."""
        config = self._cfg()
        reset_cb = AsyncMock()
        failing = _FakeOpenAIStream(
            [openai_text_chunk("p")], httpx.RemoteProtocolError("x"))

        async def cancelled_sleep(delay):
            raise asyncio.CancelledError()

        with patch("openalph.provider._get_client") as mock_gc, \
             patch("openalph.provider.asyncio.sleep", new=cancelled_sleep):
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(return_value=failing)
            with pytest.raises(asyncio.CancelledError):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                    on_stream_reset=reset_cb,
                ))

        assert client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_call_state_not_leaked_across_attempts(self):
        """kdsn.345 — RED. Attempt 1 accumulated a partial tool call before
        dying; attempt 2 succeeds text-only. The retried response must carry
        NO tool calls (per-attempt tool_call_accumulators reset)."""
        config = self._cfg()
        reset_cb = AsyncMock()
        failing = _FakeOpenAIStream(
            [openai_tool_delta_chunk()],
            httpx.RemoteProtocolError("dropped mid tool-args"))
        ok = _SuccessOpenAIStream([openai_text_chunk("clean", "stop"),
                                       openai_usage_chunk()])

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(
                side_effect=[failing, ok])
            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                on_stream_reset=reset_cb,
            ))

        assert [e.type for e in events if e.type == "tool_done"] == []
        assert events[-1].type == "done"
        assert events[-1].response.tool_calls == []

    @pytest.mark.asyncio
    async def test_degen_monitor_reinstantiated_per_attempt(self):
        """kdsn.345 — RED. A fresh DegenerationMonitor per attempt: stale
        n-gram state from the dead attempt must not contaminate the retry."""
        config = self._cfg()
        reset_cb = AsyncMock()
        instants = []
        real = provider_module.DegenerationMonitor

        class CountingMonitor(real):
            def __init__(self, *a, **k):
                instants.append(1)
                super().__init__(*a, **k)

        failing = _FakeOpenAIStream(
            [openai_text_chunk("par")], httpx.RemoteProtocolError("x"))
        ok = _SuccessOpenAIStream([openai_text_chunk("ok", "stop"),
                                       openai_usage_chunk()])

        with patch("openalph.provider._get_client") as mock_gc, \
             patch("openalph.provider.DegenerationMonitor", CountingMonitor):
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(
                side_effect=[failing, ok])
            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                on_stream_reset=reset_cb,
            ))

        assert len(instants) == 2, (
            f"expected a fresh monitor per attempt (2), got {len(instants)}")
        assert events[-1].type == "done"


# ---------------------------------------------------------------------------
# Provider layer — anthropic family
# ---------------------------------------------------------------------------

class TestMidStreamRetryAnthropic:
    @pytest.mark.asyncio
    async def test_retry_then_success_remote_protocol_error(self):
        """kdsn.345 — RED. Anthropic path: mid-stream RemoteProtocolError
        with the seam wired retries from scratch; final message is attempt-2."""
        config = _anthropic_cfg()
        reset_cb = AsyncMock()
        failing = _RaisingAnthropicStream(
            [_anthropic_text("partial ")],
            httpx.RemoteProtocolError("peer closed connection"))
        ok = _SuccessAnthropicStream(
            [_anthropic_text("full answer"), _anthropic_message_stop()],
            _anthropic_final_message("full answer"))

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream = MagicMock(side_effect=[failing, ok])
            events = await collect_events(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                on_stream_reset=reset_cb,
            ))

        assert client.messages.stream.call_count == 2
        reset_cb.assert_awaited_once()
        assert [e.type for e in events][-1] == "done"
        assert events[-1].response.content == "full answer"

    @pytest.mark.asyncio
    async def test_exhaustion_raises_single_reset(self):
        """kdsn.345 — RED. Anthropic path exhaustion: ProviderError after
        attempt 2, reset fired once."""
        config = _anthropic_cfg()
        reset_cb = AsyncMock()
        fail1 = _RaisingAnthropicStream(
            [_anthropic_text("one")], httpx.RemoteProtocolError("d1"))
        fail2 = _RaisingAnthropicStream(
            [_anthropic_text("two"), _anthropic_text("x")],
            httpx.RemoteProtocolError("d2"))

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream = MagicMock(side_effect=[fail1, fail2])
            with pytest.raises(ProviderError,
                               match=r"mid-stream after 2 chunk\(s\)"):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                    on_stream_reset=reset_cb,
                ))

        assert client.messages.stream.call_count == 2
        reset_cb.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fatal_when_seam_unwired(self):
        """kdsn.345 — guard (green pre/post)."""
        config = _anthropic_cfg()
        failing = _RaisingAnthropicStream(
            [_anthropic_text("partial ")],
            httpx.RemoteProtocolError("dropped"))

        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.messages.stream = MagicMock(return_value=failing)
            with pytest.raises(ProviderError, match="mid-stream"):
                await collect_events(stream(
                    config=config, system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                ))

        assert client.messages.stream.call_count == 1


# ---------------------------------------------------------------------------
# StreamingDelivery.reset()
# ---------------------------------------------------------------------------

def _delivery_bot():
    """Bot mock for StreamingDelivery tests; returns (bot, room_sends).

    room_sends records every _room_send_with_retry content dict; each call
    gets a fresh event id ($evt_0, $evt_1, ...).
    """
    bot = MagicMock()
    counter = {"n": 0}
    room_sends = []

    async def rec(room_id, content, *a, **k):
        room_sends.append(content)
        resp = MagicMock()
        resp.event_id = f"$evt_{counter['n']}"
        counter["n"] += 1
        return resp

    bot._room_send_with_retry = AsyncMock(side_effect=rec)
    return bot, room_sends


class TestStreamingDeliveryReset:
    @pytest.mark.asyncio
    async def test_reset_annotates_partial_and_clears_state(self):
        """kdsn.345 — RED. After an initial send, reset() edits the partial
        message (cursor removed, " ⟳" appended), clears buffer + event_id,
        and the next push starts a FRESH message."""
        bot, room_sends = _delivery_bot()
        d = StreamingDelivery(bot, ROOM)
        await d.push("x" * 60)          # crosses INITIAL_SEND_CHARS -> initial send
        assert d._event_id is not None

        await d.reset()

        # Annotation edit: replace on the partial's event id, contains ⟳, no cursor
        assert len(room_sends) >= 2
        reset_edit = room_sends[1]
        assert reset_edit["m.relates_to"]["rel_type"] == "m.replace"
        assert reset_edit["m.relates_to"]["event_id"] == "$evt_0"
        assert "⟳" in reset_edit["m.new_content"]["body"]
        assert "▍" not in reset_edit["m.new_content"]["body"]
        assert d._buffer == ""
        assert d._event_id is None

        # Next push starts a fresh message (new initial send, no m.relates_to)
        await d.push("y" * 60)
        assert len(room_sends) == 3
        assert not room_sends[2].get("m.relates_to"), (
            "retry text must start a fresh message, not edit the dead one")

    @pytest.mark.asyncio
    async def test_reset_without_initial_send_clears_buffer_only(self):
        """kdsn.345 — RED. Below the initial-send threshold there is nothing
        to annotate: reset() clears state, sends nothing."""
        bot, room_sends = _delivery_bot()
        d = StreamingDelivery(bot, ROOM)
        await d.push("tiny")
        assert d._event_id is None

        await d.reset()

        assert room_sends == []
        assert d._buffer == ""
        assert d._event_id is None

    @pytest.mark.asyncio
    async def test_reset_edit_failure_non_fatal(self):
        """kdsn.345 — RED. A failed annotation edit must not raise out of
        reset(): state is still cleared so the retry can proceed."""
        bot, room_sends = _delivery_bot()
        d = StreamingDelivery(bot, ROOM)
        await d.push("x" * 60)  # initial send succeeds
        assert d._event_id is not None
        # The annotation edit now fails
        bot._room_send_with_retry = AsyncMock(
            side_effect=RuntimeError("matrix down"))

        await d.reset()  # must not raise

        assert d._buffer == ""
        assert d._event_id is None


# ---------------------------------------------------------------------------
# agent.py wiring
# ---------------------------------------------------------------------------

class TestAgentStreamResetWiring:
    def _cfg(self, workspace):
        return AgentConfig(
            name="test-agent",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=8192,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic", api_key="sk-test")},
            max_iterations=25,
            truncation_limit=50000,
            workspace=workspace,
        )

    @pytest.mark.asyncio
    async def test_stream_receives_reset_and_accumulators_reset(self, tmp_path):
        """kdsn.345 — RED. callbacks carries 'on_stream_reset':
        (a) stream() is called with a non-None on_stream_reset kwarg;
        (b) firing it mid-stream clears the agent's accumulated state so the
            final result is attempt-2 text ONLY (not 'partial' + attempt-2);
        (c) the matrix-side callback is forwarded the info dict."""
        from openalph.provider import StreamEvent as SE

        captured = {}

        def fake_stream_fn(**kwargs):
            captured.update(kwargs)

            async def gen():
                yield SE(type="text", content="partial ")
                await kwargs["on_stream_reset"](
                    {"error": "RemoteProtocolError", "chunks": 3, "attempt": 1})
                yield SE(type="text", content="full answer")
                yield SE(type="done", stop_reason="end_turn",
                         model="claude-sonnet-4-20250514",
                         response=Response(
                             content="full answer",
                             model="claude-sonnet-4-20250514",
                             usage=Usage(input_tokens=10, output_tokens=5),
                             stop_reason="end_turn"))
            return gen()

        matrix_cb = AsyncMock()
        agent = Agent(self._cfg(tmp_path))

        with patch("openalph.agent.stream", new=fake_stream_fn):
            result = await agent.handle_input(
                "Hi", callbacks={"on_stream_reset": matrix_cb})

        assert captured.get("on_stream_reset") is not None, (
            "agent must pass on_stream_reset to provider when callbacks "
            "carry the key")
        assert result == "full answer", (
            f"accumulated attempt-1 text leaked into the retry: {result!r}")
        matrix_cb.assert_awaited_once()
        assert matrix_cb.await_args.args[0]["error"] == "RemoteProtocolError"

    @pytest.mark.asyncio
    async def test_stream_receives_none_when_callbacks_lack_key(self, tmp_path):
        """kdsn.345 — guard (green pre/post). Subagent/CLI-style callbacks
        (no 'on_stream_reset' key) -> provider param stays None (fatal-today
        path preserved for unwired consumers)."""
        from openalph.provider import StreamEvent as SE

        captured = {}

        def fake_stream_fn(**kwargs):
            captured.update(kwargs)

            async def gen():
                yield SE(type="text", content="hi")
                yield SE(type="done", stop_reason="end_turn",
                         model="claude-sonnet-4-20250514",
                         response=Response(
                             content="hi",
                             model="claude-sonnet-4-20250514",
                             usage=Usage(input_tokens=10, output_tokens=5),
                             stop_reason="end_turn"))
            return gen()

        agent = Agent(self._cfg(tmp_path))
        with patch("openalph.agent.stream", new=fake_stream_fn):
            await agent.handle_input("Hi", callbacks={"drain_steering": AsyncMock()})

        assert captured.get("on_stream_reset") is None

    @pytest.mark.asyncio
    async def test_stream_receives_none_when_callbacks_none(self, tmp_path):
        """kdsn.345 — guard (green pre/post). callbacks=None entirely (CLI
        path) -> provider param stays None."""
        from openalph.provider import StreamEvent as SE

        captured = {}

        def fake_stream_fn(**kwargs):
            captured.update(kwargs)

            async def gen():
                yield SE(type="done", stop_reason="end_turn",
                         model="claude-sonnet-4-20250514",
                         response=Response(
                             content="ok",
                             model="claude-sonnet-4-20250514",
                             usage=Usage(input_tokens=10, output_tokens=5),
                             stop_reason="end_turn"))
            return gen()

        agent = Agent(self._cfg(tmp_path))
        with patch("openalph.agent.stream", new=fake_stream_fn):
            await agent.handle_input("Hi")

        assert captured.get("on_stream_reset") is None


# ---------------------------------------------------------------------------
# matrix.py wiring
# ---------------------------------------------------------------------------

def make_matrix_config():
    return MatrixConfig(
        homeserver="https://matrix.local",
        user_id="@test:matrix.local",
        device_id="TEST",
        password="test-pw",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
    )


def make_bot(tmp_path):
    agent_config = AgentConfig(
        name="test-agent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test")},
        max_iterations=25,
        truncation_limit=50000,
        model_max_tokens=200000,
        workspace=tmp_path,
    )
    agent_config.matrix = make_matrix_config()
    agent = MagicMock(spec=Agent)
    agent.config = agent_config
    agent.system_prompt = "test prompt"
    agent.history = MagicMock(return_value=[])

    matrix_config = make_matrix_config()
    with patch("openalph.matrix.AsyncClient"):
        bot = MatrixBot(agent, matrix_config)
        bot.agent = agent
        bot.config = matrix_config
        bot._synced = True
        bot._active_rooms = set()
        bot.session_log = None
        bot.client = MagicMock()
        bot.client.room_typing = AsyncMock()
        bot.send = AsyncMock()
        bot.send_notice = AsyncMock()

        counter = {"n": 0}

        async def rec_room_send(room_id, content, *a, **k):
            resp = MagicMock()
            resp.event_id = f"$evt_{counter['n']}"
            counter["n"] += 1
            return resp

        bot._room_send_with_retry = AsyncMock(side_effect=rec_room_send)
    return bot, agent


def make_room():
    room = MagicMock()
    room.room_id = ROOM
    room.users = {"@user:matrix.local": MagicMock()}
    room.name = "Test Room"
    room.display_name = "Test Room"
    return room


def make_event(sender="@user:matrix.local", body="Hello", event_id="$evt1"):
    event = MagicMock()
    event.sender = sender
    event.body = body
    event.event_id = event_id
    event.server_timestamp = 1000000
    event.source = {}
    return event


class TestMatrixStreamResetWiring:
    @pytest.mark.asyncio
    async def test_live_path_wires_reset_and_room_contract(self, tmp_path):
        """kdsn.345 — RED. Live path: handle_input receives callbacks with
        'on_stream_reset'. Firing the captured callback mid-stream with a
        partial in flight: (1) partial message annotated with ⟳ (replace
        edit, cursor stripped), (2) one retry notice sent, (3) retry text
        starts a FRESH message, (4) the final answer still delivers."""
        bot, agent = make_bot(tmp_path)

        captured_callbacks = {}

        def fake_handle_input(*args, **kwargs):
            captured_callbacks.update(kwargs)

            async def run():
                text_cb = kwargs.get("on_text_delta")
                # Attempt 1: partial streams to the room (>= 40 chars -> initial send)
                await text_cb("The answer starts here but the connection ", done=False)
                # Mid-stream failure: provider fires the reset seam
                await kwargs["callbacks"]["on_stream_reset"](
                    {"error": "RemoteProtocolError", "chunks": 42, "attempt": 1})
                # Attempt 2: fresh full answer
                await text_cb("The answer is forty-two, complete and delivered.",
                              done=False)
                await text_cb("", done=True)
                return "The answer is forty-two, complete and delivered."
            return run()

        agent.handle_input = fake_handle_input

        await bot._process_message(make_room(), make_event(), "Hi")

        assert "on_stream_reset" in captured_callbacks.get("callbacks", {}), (
            "live path must wire callbacks['on_stream_reset']")

        sends = [a.args[1] for a in bot._room_send_with_retry.await_args_list]

        # 1: partial initial, 2: reset annotation (⟳, no cursor), 3: fresh
        # initial for the retry, 4: final edit without cursor
        assert len(sends) == 4, f"expected 4 room sends, got {len(sends)}: {sends}"
        assert "▍" in sends[0]["body"], "first send is the streaming partial"
        assert "⟳" in sends[1]["m.new_content"]["body"]
        assert "▍" not in sends[1]["m.new_content"]["body"]
        assert sends[1]["m.relates_to"]["event_id"] == "$evt_0", (
            "reset annotation must edit the partial message")
        assert "⟳" not in sends[2]["body"], "retry text starts fresh (no ⟳)"
        assert "forty-two" in sends[3]["m.new_content"]["body"]

        # exactly one retry notice
        notices = [c.args[1] for c in bot.send_notice.await_args_list]
        reset_notices = [n for n in notices if "retrying" in n.lower()]
        assert len(reset_notices) == 1, f"expected 1 retry notice, got {notices}"

    @pytest.mark.asyncio
    async def test_heartbeat_path_wires_reset_notice_only(self, tmp_path):
        """kdsn.345 — RED. Heartbeat turns (no StreamingDelivery) still wire
        the seam: invoking it sends the retry notice and never touches the
        room-send channel."""
        bot, agent = make_bot(tmp_path)

        captured_callbacks = {}

        def fake_handle_input(*args, **kwargs):
            captured_callbacks.update(kwargs)

            async def run():
                await kwargs["callbacks"]["on_stream_reset"](
                    {"error": "RemoteProtocolError", "chunks": 7, "attempt": 1})
                return "hb answer"
            return run()

        agent.handle_input = fake_handle_input

        bot._active_rooms.add(ROOM)  # skip room activation
        await bot._run_heartbeat_turn(ROOM, "heartbeat prompt")

        assert "on_stream_reset" in captured_callbacks.get("callbacks", {}), (
            "heartbeat path must wire callbacks['on_stream_reset']")
        notices = [c.args[1] for c in bot.send_notice.await_args_list]
        assert any("retrying" in n.lower() for n in notices), (
            f"expected a retry notice, got {notices}")
        # notice-only: no room_send traffic from the reset seam
        assert bot._room_send_with_retry.await_count == 0
