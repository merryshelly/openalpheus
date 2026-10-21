"""Tests for the subagent streaming path + retry-seam arming (workspace-im7t.83).

Problem (verified in source at 90f24d82):
  1. subagent.py calls provider.complete() — NON-streaming. Synthetic Infinite
     Mode (workspace-im7t.80-.82) REQUIRES stream:true (long pauses time out
     unstreamed requests), so a sub dispatched at an infinite alias cannot run
     any non-trivial generation.
  2. A sub parked in ONE silent provider call emits no milestone, so the
     parent's turn-stall watchdog (900s default) cancels the parent turn —
     documented as intended for silent calls, but a 950s slow-and-alive
     infinite call is NOT stalled; the parent watchdog must be fed DURING
     the provider call.
  3. kdsn.345's midstream retry is armed only when the caller passes
     on_stream_reset — sub and CLI paths keep fatal-on-drop behavior. With
     free infinite tokens, full-call retry is the cheap recovery.

Design under test (blast-radius-minimizing — ~126 existing test sites patch
'openalph.tools.subagent.complete' and MUST keep working):
  - subagent.complete() becomes a module-level wrapper with the SAME call
    shape (config/system/messages/tools/max_tokens/thinking/metrics_labels)
    that internally calls provider stream(), collects the done-event
    Response, and forwards the same kwargs (attribution labels + effort
    preserved).
  - The wrapper ALWAYS passes on_stream_reset (a no-op async reset — subs
    have no room display to reset) so the provider-level midstream retry is
    armed on every sub LLM call. Provider-side retry semantics are already
    covered by test_midstream_retry.py; here we pin that the seam is ARMED.
  - The wrapper pokes a per-run progress hook (ContextVar _SUB_PROGRESS_POKE,
    set by run_subagent to the parent's turn_progress reset, cleared in a
    finally) once per stream event — the RAW watchdog reset, deliberately
    NOT _milestone (whose documented contract is completions/iterations).
  - build_callbacks gains a no-op on_stream_reset default so headless CLI
    paths arm the retry too; matrix.py overrides it with the real display
    seam (assignment, unchanged).
"""

from unittest.mock import MagicMock, patch

import pytest

from openalph.callbacks import build_callbacks
from openalph.provider import ProviderError, Response, StreamEvent, Usage
from openalph.tools import subagent as subagent_mod
from subledger_fixtures import DEFAULT_TASK, build_bot, real_callbacks, sub_tool_call


def _done_response(content="REPORT", tool_calls=None):
    return Response(
        content=content,
        usage=Usage(input_tokens=10, output_tokens=5),
        stop_reason="tool_calls" if tool_calls else "stop",
        tool_calls=tool_calls or [],
    )


def _stream_events(response):
    return [
        StreamEvent(type="text", content="partial"),
        StreamEvent(type="thinking", content="hm"),
        StreamEvent(type="done", response=response),
    ]


def _minimal_config():
    """The wrapper forwards config opaquely; a stub suffices at this layer."""
    return MagicMock()


class TestSubCompleteWrapper:
    """subagent.complete() — same name/call-shape, streaming internals."""

    async def _call(self, monkeypatch, events, **extra):
        captured = {}

        async def fake_stream(**kwargs):
            captured.update(kwargs)
            for e in events:
                yield e

        monkeypatch.setattr(subagent_mod, "_provider_stream", fake_stream)
        kwargs = dict(
            config=_minimal_config(),
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=1024,
            thinking="medium",
            metrics_labels={"agent": "t", "session": "r", "kind": "sub"},
        )
        kwargs.update(extra)
        resp = await subagent_mod.complete(**kwargs)
        return resp, captured

    @pytest.mark.asyncio
    async def test_returns_done_event_response(self, monkeypatch):
        resp, _ = await self._call(monkeypatch, _stream_events(_done_response("R1")))
        assert resp.content == "R1"
        assert resp.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_forwards_call_shape_kwargs(self, monkeypatch):
        """Attribution labels + effort + tools must reach the provider stream
        call (test_metrics_labels / test_subagent_effort_thinking pin the
        call sites; the wrapper must not drop them)."""
        _, captured = await self._call(monkeypatch, _stream_events(_done_response()))
        assert captured["metrics_labels"] == {"agent": "t", "session": "r", "kind": "sub"}
        assert captured["thinking"] == "medium"
        assert captured["max_tokens"] == 1024
        assert captured["system"] == "sys"
        assert captured["messages"] == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_stream_reset_seam_armed(self, monkeypatch):
        """kdsn.345 retry is armed ONLY when on_stream_reset is passed — subs
        must ALWAYS pass one (no-op: no room display to reset)."""
        _, captured = await self._call(monkeypatch, _stream_events(_done_response()))
        reset = captured.get("on_stream_reset")
        assert reset is not None
        # Must be an async no-op: awaitable with the retry-info dict, no raise.
        await reset({"error": "boom", "chunks": 3, "attempt": 1})

    @pytest.mark.asyncio
    async def test_pokes_progress_per_event(self, monkeypatch):
        """Every stream event pokes the raw parent watchdog reset (NOT a
        milestone): a 950s slow-and-alive single provider call must not
        starve the parent's 900s stall watchdog."""
        pokes = []

        token = subagent_mod._SUB_PROGRESS_POKE.set(lambda: pokes.append(1))
        try:
            await self._call(monkeypatch, _stream_events(_done_response()))
        finally:
            subagent_mod._SUB_PROGRESS_POKE.reset(token)
        assert len(pokes) == 3  # text + thinking + done

    @pytest.mark.asyncio
    async def test_poke_absent_is_tolerated(self, monkeypatch):
        """Headless/no-parent paths run with the ContextVar unset — the
        wrapper must not raise."""
        assert subagent_mod._SUB_PROGRESS_POKE.get() is None
        resp, _ = await self._call(monkeypatch, _stream_events(_done_response()))
        assert resp.content == "REPORT"

    @pytest.mark.asyncio
    async def test_raises_provider_error_without_done_event(self, monkeypatch):
        """A stream that ends without a done event (transport died after
        retries were exhausted upstream) must raise ProviderError — the
        sub loop's existing failure handling keys on it; a silent None
        would corrupt the loop."""
        with pytest.raises(ProviderError):
            await self._call(monkeypatch, [StreamEvent(type="text", content="partial")])


class TestBuildCallbacksDefaultReset:
    def test_noop_stream_reset_in_default_dict(self):
        """CLI/headless callback dicts arm the midstream retry via a no-op
        reset; matrix.py's real display seam replaces it by assignment."""
        cb = build_callbacks(agent=MagicMock(), room_id="!r:local", sinks=MagicMock())
        assert "on_stream_reset" in cb
        reset = cb["on_stream_reset"]
        assert callable(reset)

    @pytest.mark.asyncio
    async def test_default_reset_is_awaitable_noop(self):
        cb = build_callbacks(agent=MagicMock(), room_id="!r:local", sinks=MagicMock())
        await cb["on_stream_reset"]({"error": "boom", "chunks": 1, "attempt": 0})


class TestSubRunWiresPokeContext:
    @pytest.mark.asyncio
    async def test_run_sets_and_clears_poke_contextvar(self, tmp_path):
        """run_subagent sets _SUB_PROGRESS_POKE to a callable (the parent's
        watchdog reset, no-op-safe) for the duration of the run and clears
        it afterwards. Concurrency-safe by ContextVar semantics: each sub
        task sees its own value."""
        bot, agent = build_bot(tmp_path)
        seen = []

        async def fake_stream(**kwargs):
            seen.append(subagent_mod._SUB_PROGRESS_POKE.get())
            yield StreamEvent(type="text", content="working")
            seen.append(subagent_mod._SUB_PROGRESS_POKE.get())
            yield StreamEvent(type="done",
                              response=_done_response("STREAMED-REPORT-1"))

        from openalph.tools import execute_tool
        cb = real_callbacks(bot)
        with patch.object(subagent_mod, "_provider_stream", fake_stream):
            res = await execute_tool(
                "subagent", sub_tool_call(task=DEFAULT_TASK, background=False),
                {}, agent.config, tools=None, callbacks=cb)
        assert not res.is_error, f"sub run failed: {res.content}"
        assert "STREAMED-REPORT-1" in res.content
        assert len(seen) == 2
        assert all(callable(v) for v in seen), "poke must be set during the run"
        assert subagent_mod._SUB_PROGRESS_POKE.get() is None, "poke must be cleared"
