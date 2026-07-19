"""Integration tests: DegenerationMonitor wired into provider.stream() (kdsn.241.4).

Verifies the OpenAI-compat streaming path:
  - warn mode: a degenerate stream is flagged (response.degenerate) and logged,
    but output is NOT modified and the stream is fully consumed.
  - abort mode: the consume loop breaks, the stream is closed, and the emitted
    text is truncated with the warning appended.
  - off mode: no detection, no flag.
"""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import stream, _DEGEN_WARNING


def _cfg(degen_detector):
    return AgentConfig(
        name="t",
        default_model="fireworks/accounts/fireworks/models/glm-5p2",
        max_tokens=4096,
        providers={"fireworks": ProviderConfig(
            key="fireworks", type="openai", api_key="sk-test",
            base_url="https://api.fireworks.ai/inference/v1",
        )},
        workspace=Path("/tmp/test"),
        degen_detector=degen_detector,
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


def _tool_chunk(name, args_delta):
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = None
    delta.reasoning = None
    delta.reasoning_content = None
    tc = MagicMock()
    tc.index = 0
    tc.id = "call_1"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = args_delta
    tc.extra_content = None
    delta.tool_calls = [tc]
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
    """Async-iterable OpenAI stream mock that counts consumed chunks and
    records whether close() was awaited."""
    def __init__(self, chunks):
        self._chunks = chunks
        self.consumed = 0
        self.closed = False

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for c in self._chunks:
            self.consumed += 1
            yield c

    async def close(self):
        self.closed = True


async def _collect(gen):
    return [e async for e in gen]


# A phrase loop: 60 repetitions -> arms and trips the n-gram layer early.
_LOOP = ["the build is completely broken now " for _ in range(60)]


async def _run(mode):
    chunks = [_text_chunk(t) for t in _LOOP] + [_finish_chunk()]
    st = CountingStream(chunks)
    with patch("openalph.provider._get_client") as gc:
        client = MagicMock()
        gc.return_value = client
        client.chat.completions.create = AsyncMock(return_value=st)
        events = await _collect(stream(
            config=_cfg(mode), system="sys",
            messages=[{"role": "user", "content": "go"}],
        ))
    done = [e for e in events if e.type == "done"][0]
    return st, done


@pytest.mark.asyncio
async def test_warn_flags_but_does_not_truncate():
    st, done = await _run("warn")
    # Fully consumed (warn never breaks), degenerate flagged, output intact.
    assert st.consumed == len(_LOOP) + 1
    assert st.closed is False
    assert done.response.degenerate is True
    assert _DEGEN_WARNING not in done.response.content
    assert done.response.content.count("broken") > 5  # full loop text preserved


@pytest.mark.asyncio
async def test_abort_truncates_and_closes_stream():
    st, done = await _run("abort")
    # Broke early: not all chunks consumed; stream closed; text truncated + warned.
    assert st.consumed < len(_LOOP) + 1
    assert st.closed is True
    assert done.response.degenerate is True
    assert _DEGEN_WARNING in done.response.content


@pytest.mark.asyncio
async def test_off_mode_no_detection():
    st, done = await _run("off")
    assert st.consumed == len(_LOOP) + 1
    assert done.response.degenerate is False
    assert _DEGEN_WARNING not in done.response.content


@pytest.mark.asyncio
async def test_abort_drops_partial_tool_calls():
    """On abort, a partially-accumulated tool call (incomplete JSON args) must
    NOT be emitted or included in the response (kdsn.241.4 audit finding)."""
    # Degenerate text trips the monitor, then a tool call starts accumulating
    # partial (invalid) JSON args — abort must discard it.
    chunks = ([_text_chunk(t) for t in _LOOP]
              + [_tool_chunk("submit_review", '{"verdict": "pa')]  # truncated JSON
              + [_finish_chunk()])
    st = CountingStream(chunks)
    with patch("openalph.provider._get_client") as gc:
        client = MagicMock()
        gc.return_value = client
        client.chat.completions.create = AsyncMock(return_value=st)
        events = await _collect(stream(
            config=_cfg("abort"), system="sys",
            messages=[{"role": "user", "content": "go"}],
        ))
    done = [e for e in events if e.type == "done"][0]
    tool_done = [e for e in events if e.type == "tool_done"]
    assert done.response.tool_calls == []
    assert tool_done == []
