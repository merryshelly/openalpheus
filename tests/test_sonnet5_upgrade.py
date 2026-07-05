"""Tests for the Sonnet 5 upgrade (workspace-snak.14).

Three framework changes in provider.py, all on the shared Anthropic inference path:

  1. _supports_adaptive_thinking(): claude-sonnet-5 uses adaptive thinking
     (type=adaptive + output_config.effort). Verified empirically 2026-07-04
     (live API, oa-babson key): Sonnet 5 REJECTS the legacy
     {"type":"enabled","budget_tokens":N} format with HTTP 400 and requires the
     adaptive format. Without this, every thinking inference 400s.

  2. _supports_sampling_params() allowlist: modern Anthropic models (Opus 4.7,
     Opus 4.8, Sonnet 5, Fable, Mythos) permanently removed temperature/top_p/
     top_k -- sending them 400s or is silently dropped. The sampling block is now
     gated on a fail-closed allowlist; the 4.x families are enumerated per-minor
     so a future in-family minor that drops sampling is NOT auto-accepted.
     Closes kdsn.134 (Opus 4.7 guard).

  3. _MODEL_CAPABILITIES: claude-sonnet-5 -> native 1M context / 128K output cap.

Hardening applied post code-audit (2026-07-05): per-minor allowlist granularity,
case-insensitive gate functions, and end-to-end config->complete() wiring coverage.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import (
    complete,
    _supports_adaptive_thinking,
    _supports_sampling_params,
    _model_output_cap,
    model_context_window,
    _build_anthropic_kwargs,
)


# ---------------------------------------------------------------------------
# Local helpers (self-contained; no cross-test-module imports)
# ---------------------------------------------------------------------------


def make_provider(key="anthropic", type="anthropic", api_key="sk-test",
                  base_url=None, quirks=None):
    return ProviderConfig(
        key=key, type=type, api_key=api_key,
        base_url=base_url, quirks=quirks or [],
    )


def make_config(**kwargs):
    defaults = {
        "name": "test-agent",
        "default_model": "anthropic/claude-sonnet-5",
        "max_tokens": 65536,
        "providers": {"anthropic": make_provider(key="anthropic")},
        "workspace": Path("/tmp/test"),
        "thinking": "high",
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


class MockAnthropicStream:
    """Mock for Anthropic's AsyncMessageStream context manager."""

    def __init__(self, events, final_message=None):
        self._events = events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final_message


def _make_text_event(text):
    e = MagicMock()
    e.type = "text"
    e.text = text
    return e


def _make_message_stop():
    e = MagicMock()
    e.type = "message_stop"
    return e


def mock_anthropic_response(text="ok", model="claude-sonnet-5"):
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    resp = MagicMock()
    resp.content = [text_block]
    resp.model = model
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 5
    resp.usage.cache_read_input_tokens = 0
    resp.usage.cache_creation_input_tokens = 0
    resp.stop_reason = "end_turn"
    return resp


async def _stream_kwargs(model, thinking, temperature=None, top_p=None):
    """Drive a real AgentConfig through complete() and return the outbound
    Anthropic SDK kwargs — exercises the production call-site wiring."""
    config = make_config(default_model=model, thinking=thinking,
                         temperature=temperature, top_p=top_p)
    with patch("openalph.provider.anthropic.AsyncAnthropic") as MockClient:
        client = MockClient.return_value
        client.messages.stream = MagicMock(
            return_value=MockAnthropicStream(
                [_make_text_event("ok"), _make_message_stop()],
                final_message=mock_anthropic_response(),
            )
        )
        await complete(config=config, system="Test",
                       messages=[{"role": "user", "content": "Hi"}])
    return client.messages.stream.call_args.kwargs


# ---------------------------------------------------------------------------
# Change 1: adaptive thinking for Sonnet 5
# ---------------------------------------------------------------------------


class TestSonnet5AdaptiveThinking:

    def test_sonnet5_supports_adaptive(self):
        assert _supports_adaptive_thinking("claude-sonnet-5") is True

    def test_sonnet5_with_provider_prefix(self):
        assert _supports_adaptive_thinking("anthropic/claude-sonnet-5") is True

    def test_sonnet5_dated(self):
        assert _supports_adaptive_thinking("claude-sonnet-5-20260630") is True

    def test_sonnet5_case_insensitive(self):
        # Gate must not depend on casing (matches sibling capability functions).
        assert _supports_adaptive_thinking("claude-Sonnet-5") is True

    def test_sonnet46_still_adaptive_regression(self):
        # Adding sonnet-5 must not disturb sonnet-4-6.
        assert _supports_adaptive_thinking("claude-sonnet-4-6") is True

    @pytest.mark.asyncio
    async def test_sonnet5_complete_uses_adaptive_not_legacy(self):
        """thinking=high on Sonnet 5 sends adaptive format, NOT legacy budget_tokens.

        The legacy {"type":"enabled","budget_tokens":N} format is what the API
        rejects with HTTP 400 for Sonnet 5. Lock the adaptive format in.
        """
        kw = await _stream_kwargs("anthropic/claude-sonnet-5", "high")
        assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kw["thinking"].get("type") == "adaptive"
        assert kw["thinking"].get("type") != "enabled"
        assert "budget_tokens" not in kw["thinking"]
        assert kw["output_config"] == {"effort": "high"}


# ---------------------------------------------------------------------------
# Change 2: sampling-param allowlist (fail-closed, per-minor, case-insensitive)
# ---------------------------------------------------------------------------


class TestSupportsSamplingParams:
    """Allowlist: which Anthropic models accept temperature/top_p/top_k."""

    # Reject set — modern models that removed sampling params.
    def test_sonnet5_rejects(self):
        assert _supports_sampling_params("claude-sonnet-5") is False

    def test_opus47_rejects(self):
        assert _supports_sampling_params("claude-opus-4-7") is False

    def test_opus48_rejects(self):
        assert _supports_sampling_params("claude-opus-4-8") is False

    def test_fable_rejects(self):
        assert _supports_sampling_params("claude-fable-5") is False

    def test_mythos_rejects(self):
        assert _supports_sampling_params("claude-mythos-5") is False

    # Accept set — older releases still take sampling params.
    def test_opus46_accepts(self):
        assert _supports_sampling_params("claude-opus-4-6-20250605") is True

    def test_opus45_accepts(self):
        assert _supports_sampling_params("claude-opus-4-5-20250220") is True

    def test_sonnet46_accepts(self):
        assert _supports_sampling_params("claude-sonnet-4-6") is True

    def test_haiku45_accepts(self):
        assert _supports_sampling_params("claude-haiku-4-5-20251001") is True

    def test_claude35_sonnet_accepts(self):
        assert _supports_sampling_params("claude-3-5-sonnet-20241022") is True

    def test_claude35_haiku_accepts(self):
        # Haiku 3.5 accepts sampling; matched via the frozen claude-3.x family.
        assert _supports_sampling_params("claude-3-5-haiku-20241022") is True

    # Fail-closed granularity: per-minor for 4.x families. A FUTURE in-family
    # minor that drops sampling (as Opus did at 4.6 -> 4.7) must NOT be accepted.
    def test_future_haiku_minor_failclosed(self):
        assert _supports_sampling_params("claude-haiku-4-7") is False

    def test_future_sonnet4_minor_failclosed(self):
        assert _supports_sampling_params("claude-sonnet-4-7") is False

    def test_sonnet40_failclosed(self):
        # Sonnet 4.0 is not enumerated (only sonnet-4-6). Fail-closed -> sampling
        # silently dropped (safe direction), never a 400. Not in our fleet.
        assert _supports_sampling_params("claude-sonnet-4-20250514") is False

    # Forward-safety: unknown/future defaults to False (never a 400).
    def test_unknown_model_defaults_false(self):
        assert _supports_sampling_params("claude-some-future-model-9") is False

    def test_empty_false(self):
        assert _supports_sampling_params("") is False

    # Case-insensitivity (matches model_context_window / _model_output_cap).
    def test_case_insensitive(self):
        assert _supports_sampling_params("claude-Sonnet-5") is False
        assert _supports_sampling_params("Claude-Opus-4-6") is True


class TestSamplingParamStripping:
    """Unit-level: _build_anthropic_kwargs (thinking off, params supplied)."""

    def _kw(self, model, thinking="off", temperature=0.7, top_p=0.9):
        return _build_anthropic_kwargs(
            model, "sys",
            [{"role": "user", "content": "hi"}], None,
            65536, thinking,
            model_max_tokens=1048576,
            temperature=temperature, top_p=top_p,
        )

    def test_sonnet5_off_strips_sampling(self):
        kw = self._kw("claude-sonnet-5", thinking="off")
        assert "temperature" not in kw
        assert "top_p" not in kw

    def test_opus47_off_strips_sampling(self):
        # kdsn.134 — Opus 4.7 guard, folded in here.
        kw = self._kw("claude-opus-4-7", thinking="off")
        assert "temperature" not in kw
        assert "top_p" not in kw

    def test_opus48_off_strips_sampling(self):
        kw = self._kw("claude-opus-4-8", thinking="off")
        assert "temperature" not in kw
        assert "top_p" not in kw

    def test_sonnet46_off_keeps_sampling_regression(self):
        # Regression guard: older Sonnet must still receive sampling params.
        kw = self._kw("claude-sonnet-4-6", thinking="off")
        assert kw["temperature"] == 0.7
        assert kw["top_p"] == 0.9

    def test_opus46_off_keeps_sampling_regression(self):
        kw = self._kw("claude-opus-4-6", thinking="off")
        assert kw["temperature"] == 0.7
        assert kw["top_p"] == 0.9

    def test_sonnet5_thinking_high_no_sampling(self):
        # Thinking on already gates sampling; assert adaptive + no leak.
        kw = self._kw("claude-sonnet-5", thinking="high")
        assert "temperature" not in kw
        assert "top_p" not in kw
        assert kw["thinking"]["type"] == "adaptive"


class TestSamplingStripEndToEnd:
    """Production wiring: real AgentConfig -> complete() -> stream() kwargs.

    Guards the specific failure mode the change exists to prevent: config-sourced
    sampling params leaking into a Sonnet 5 / Opus 4.7+ Anthropic request. The
    unit tests above poke the builder directly; these exercise the call site
    (provider.py getattr(config, "temperature"/"top_p")).
    """

    @pytest.mark.asyncio
    async def test_sonnet5_off_strips_sampling_wired(self):
        kw = await _stream_kwargs("anthropic/claude-sonnet-5", "off",
                                  temperature=0.7, top_p=0.9)
        assert "temperature" not in kw
        assert "top_p" not in kw

    @pytest.mark.asyncio
    async def test_opus48_off_strips_sampling_wired(self):
        kw = await _stream_kwargs("anthropic/claude-opus-4-8", "off",
                                  temperature=0.7, top_p=0.9)
        assert "temperature" not in kw
        assert "top_p" not in kw

    @pytest.mark.asyncio
    async def test_sonnet46_off_keeps_sampling_wired(self):
        # Mirror-positive: older Sonnet still gets config sampling params.
        kw = await _stream_kwargs("anthropic/claude-sonnet-4-6", "off",
                                  temperature=0.7, top_p=0.9)
        assert kw["temperature"] == 0.7
        assert kw["top_p"] == 0.9


# ---------------------------------------------------------------------------
# Change 3: capabilities (1M window / 128K output cap)
# ---------------------------------------------------------------------------


class TestSonnet5Capabilities:

    def test_context_window_1m(self):
        assert model_context_window("claude-sonnet-5") == 1_048_576

    def test_context_window_prefixed(self):
        assert model_context_window("anthropic/claude-sonnet-5") == 1_048_576

    def test_output_cap_128k(self):
        assert _model_output_cap("claude-sonnet-5") == 128_000

    def test_sonnet46_still_200k_regression(self):
        # Adding sonnet-5 must not shadow the sonnet-4-6 200K entry.
        assert model_context_window("claude-sonnet-4-6") == 200_000

    def test_build_kwargs_clamps_output_to_128k(self):
        kw = _build_anthropic_kwargs(
            "claude-sonnet-5", "sys",
            [{"role": "user", "content": "hi"}], None,
            200000, "max",
            model_max_tokens=1048576,
        )
        assert kw["max_tokens"] == 128000
