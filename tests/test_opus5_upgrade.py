"""Tests for the Opus 5 launch (2026-08, migration from Opus 4.8).

Anthropic's migration guide (docs.claude.com Opus 4.8 -> Opus 5) documents Opus 5
as a drop-in upgrade at identical pricing ($5/$25 per MTok), with two breaking
changes that don't affect this fleet (thinking-on-by-default when the `thinking`
field is omitted -- we never omit it while thinking!="off"; and
`thinking:{type:"disabled"}` capped at effort<=high -- we never send `type:
"disabled"` at all, we simply omit the `thinking` field when thinking_level=="off").

What DOES need a framework change, identical in kind to the Sonnet 5 launch
(see test_sonnet5_upgrade.py): claude-opus-5 is a brand-new model-ID fragment
that matches none of the existing substring allowlists in provider.py. Without
adding it:

  1. _supports_adaptive_thinking("claude-opus-5") would be False, so any request
     with thinking enabled (our fleet default everywhere) would fall through to
     the legacy `{"type":"enabled","budget_tokens":N}` branch instead of
     `{"type":"adaptive"}` + `output_config.effort` -- silently downgrading
     xhigh/max to a fixed ~16K budget (with a warning), or erroring outright if
     Opus 5 no longer accepts the legacy format.
  2. _MODEL_CAPABILITIES would have no entry -> unknown-model window/cap warning
     instead of the documented 1M context / 128K output cap.
  3. _MODEL_PRICING would have no entry -> Opus 5 usage silently reported as
     unpriced_tokens instead of costed at $5/$25 per MTok.

Fixes are additive to the existing allowlists (fragment "opus-5", chosen to
avoid colliding with "opus-4-5" et al, which contain "-4-" and not "opus-5" as a
contiguous substring -- covered by the collision-regression tests below).
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
    compute_cost,
    Usage,
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
        "default_model": "anthropic/claude-opus-5",
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


def mock_anthropic_response(text="ok", model="claude-opus-5"):
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
# Change 1: adaptive thinking for Opus 5
# ---------------------------------------------------------------------------


class TestOpus5AdaptiveThinking:

    def test_opus5_supports_adaptive(self):
        assert _supports_adaptive_thinking("claude-opus-5") is True

    def test_opus5_with_provider_prefix(self):
        assert _supports_adaptive_thinking("anthropic/claude-opus-5") is True

    def test_opus5_case_insensitive(self):
        assert _supports_adaptive_thinking("claude-Opus-5") is True

    def test_opus45_still_adaptive_regression(self):
        # Adding opus-5 must not disturb opus-4-5 (collision check: "opus-4-5"
        # does not contain the contiguous substring "opus-5").
        assert _supports_adaptive_thinking("claude-opus-4-5") is True

    def test_opus46_still_adaptive_regression(self):
        assert _supports_adaptive_thinking("claude-opus-4-6") is True

    def test_opus47_still_adaptive_regression(self):
        assert _supports_adaptive_thinking("claude-opus-4-7") is True

    def test_opus48_still_adaptive_regression(self):
        assert _supports_adaptive_thinking("claude-opus-4-8") is True

    def test_no_false_positive_on_opus4x(self):
        # Explicit collision guard: "opus-5" must not be a substring of any
        # opus-4.x fragment (they all have "-4-" between "opus" and the minor).
        for frag in ("opus-4-5", "opus-4-6", "opus-4-7", "opus-4-8"):
            assert "opus-5" not in frag

    @pytest.mark.asyncio
    async def test_opus5_complete_uses_adaptive_not_legacy(self):
        """thinking=high on Opus 5 sends adaptive format, NOT legacy budget_tokens.

        This is the failure mode the change exists to prevent: without the
        allowlist entry, this would send {"type":"enabled","budget_tokens":16384}
        instead, silently downgrading effort and possibly 400ing.
        """
        kw = await _stream_kwargs("anthropic/claude-opus-5", "high")
        assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kw["thinking"].get("type") != "enabled"
        assert "budget_tokens" not in kw["thinking"]
        assert kw["output_config"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_opus5_xhigh_effort_passthrough(self):
        # The whole point of adding opus5 as a "step up" option is xhigh/max
        # effort actually reaching the API, not getting silently capped.
        kw = await _stream_kwargs("anthropic/claude-opus-5", "xhigh")
        assert kw["output_config"] == {"effort": "xhigh"}

    @pytest.mark.asyncio
    async def test_opus5_max_effort_passthrough(self):
        kw = await _stream_kwargs("anthropic/claude-opus-5", "max")
        assert kw["output_config"] == {"effort": "max"}


# ---------------------------------------------------------------------------
# Change 2: sampling params -- Opus 5 must stay OUT of the allowlist
# ---------------------------------------------------------------------------


class TestOpus5SamplingParams:
    """Fail-closed by omission: Opus 5 is a newer generation than Opus 4.6 (the
    last sampling-accepting minor), so it must NOT be added to
    _supports_sampling_params. Locking this in explicitly rather than relying
    on silent omission, matching the Sonnet 5 test suite's convention."""

    def test_opus5_rejects_sampling_params(self):
        assert _supports_sampling_params("claude-opus-5") is False

    def test_opus5_off_strips_sampling(self):
        kw = _build_anthropic_kwargs(
            "claude-opus-5", "sys",
            [{"role": "user", "content": "hi"}], None,
            65536, "off",
            model_max_tokens=1048576,
            temperature=0.7, top_p=0.9,
        )
        assert "temperature" not in kw
        assert "top_p" not in kw


# ---------------------------------------------------------------------------
# Change 3: capabilities (1M window / 128K output cap)
# ---------------------------------------------------------------------------


class TestOpus5Capabilities:

    def test_context_window_1m(self):
        assert model_context_window("claude-opus-5") == 1_048_576

    def test_context_window_prefixed(self):
        assert model_context_window("anthropic/claude-opus-5") == 1_048_576

    def test_output_cap_128k(self):
        assert _model_output_cap("claude-opus-5") == 128_000

    def test_opus48_still_1m_regression(self):
        # Adding opus-5 must not shadow the opus-4-8 entry.
        assert model_context_window("claude-opus-4-8") == 1_048_576
        assert _model_output_cap("claude-opus-4-8") == 128_000

    def test_build_kwargs_clamps_output_to_128k(self):
        kw = _build_anthropic_kwargs(
            "claude-opus-5", "sys",
            [{"role": "user", "content": "hi"}], None,
            200000, "max",
            model_max_tokens=1048576,
        )
        assert kw["max_tokens"] == 128000


# ---------------------------------------------------------------------------
# Change 4: session cost pricing ($5/$25 per MTok, matching Opus 4.8)
# ---------------------------------------------------------------------------


class TestOpus5Pricing:

    def test_priced_not_unpriced(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        result = compute_cost("anthropic/claude-opus-5", usage, provider_type="anthropic")
        assert result.priced is True
        assert result.unpriced_tokens == 0
        # $5 input + $25 output per MTok, 1M tokens each => $5 + $25 = $30
        assert result.cost_usd == pytest.approx(30.0)

    def test_matches_opus48_rate(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        opus5 = compute_cost("anthropic/claude-opus-5", usage, provider_type="anthropic")
        opus48 = compute_cost("anthropic/claude-opus-4-8", usage, provider_type="anthropic")
        assert opus5.cost_usd == pytest.approx(opus48.cost_usd)
