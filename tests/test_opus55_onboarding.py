"""Opus 5.5 (claude-opus-5-5) onboarding pins — 2026-09-22, release day.

Vendor facts (platform.claude.com/docs/en/models/opus-5-5/overview):
  * Model ID: claude-opus-5-5 (no date suffix on the Claude API ID).
  * 1M context, 128K max output (Anthropic hard-400s over-cap -> clamp).
  * $4 input / $20 output per MTok; 5m cache write $5 (1.25x), 1h cache
    write $8 (2.0x), cache READ $0.20/MTok = 0.05x base input — the SECOND
    Anthropic model off the uniform CACHE_READ_MULT (fable-5-1 is 0.025x).
  * Adaptive thinking ALWAYS ON (can't be disabled), default effort medium;
    effort parameter controls depth.
  * Breaking changes shared with fable-5.1: forced tool use 400s; thinking
    blocks tied to model+conversation; text between tool calls returns in
    thinking blocks (empty text at default display — streamed inter-tool
    progress goes quiet unless a display value is set).

Code surface pinned here:
  1. _MODEL_CAPABILITIES: opus-5-5 row BEFORE the generic "opus-5" row
     (first-match-wins — "opus-5" is a substring of "opus-5-5").
  2. compute_cost: opus-5-5 cache reads at 0.05x ($0.20/MTok); opus-5 stays
     0.1x (contrast); date-suffix normalization prices identically.
  3. Gates: adaptive True, sampling-params False (fail-closed allowlist).
"""

from datetime import date

import pytest

from openalph.provider import (
    Usage,
    compute_cost,
    _MODEL_CAPABILITIES,
    _MODEL_PRICING,
    _supports_adaptive_thinking,
    _supports_sampling_params,
    _model_output_cap,
    model_context_window,
)

APPROX = 1e-9


def _u(**kw) -> Usage:
    return Usage(**kw)


def _fragments():
    return [frag for frag, *_ in _MODEL_CAPABILITIES]


class TestOpus55Capabilities:
    def test_window_and_output_cap(self):
        assert model_context_window("claude-opus-5-5") == 1_048_576
        assert _model_output_cap("claude-opus-5-5") == 128_000

    def test_row_ordered_before_generic_opus5(self):
        """Ordering pin: 'opus-5' substring-matches 'opus-5-5'; the explicit
        row must sit FIRST so a future opus-5 legacy edit can't leak specs."""
        frags = _fragments()
        assert frags.index("opus-5-5") < frags.index("opus-5")

    def test_adaptive_thinking(self):
        assert _supports_adaptive_thinking("claude-opus-5-5") is True
        assert _supports_adaptive_thinking("anthropic/claude-opus-5-5") is True

    def test_sampling_params_fail_closed(self):
        """Newer than the sampling-drop generation — no temp/top_p/top_k."""
        assert _supports_sampling_params("claude-opus-5-5") is False


class TestOpus55Pricing:
    def test_entry_shape(self):
        entry = _MODEL_PRICING["anthropic"]["claude-opus-5-5"]
        assert entry["input"] == 4.0
        assert entry["output"] == 20.0
        assert entry["cached_input"] == 0.20

    def test_full_cost_mix_with_005x_cache_read(self):
        usage = _u(input_tokens=1_000_000, output_tokens=100_000,
                   cache_read_tokens=200_000, cache_creation_tokens=20_000,
                   cache_creation_5m_tokens=8_000, cache_creation_1h_tokens=12_000)
        r = compute_cost("claude-opus-5-5", usage)
        # 4.0 (in) + 2.0 (out) + 200k*0.20/1e6 (0.04) + 8k*5/1e6 (0.04)
        #   + 12k*8/1e6 (0.096) = 6.176
        assert r.priced is True
        assert r.unpriced_tokens == 0
        assert r.cost_usd == pytest.approx(6.176, abs=APPROX)

    def test_opus5_still_uniform_01x(self):
        """Contrast: same usage on opus-5 — cache read at 0.1x of $5 = $0.50/MTok."""
        usage = _u(input_tokens=1_000_000, output_tokens=100_000,
                   cache_read_tokens=200_000, cache_creation_tokens=20_000,
                   cache_creation_5m_tokens=8_000, cache_creation_1h_tokens=12_000)
        r = compute_cost("claude-opus-5", usage)
        # 5.0 + 2.5 + 200k*0.5/1e6 (0.10) + 8k*6.25/1e6 (0.05) + 12k*10/1e6 (0.12)
        assert r.cost_usd == pytest.approx(7.77, abs=APPROX)

    def test_date_suffix_normalized(self):
        """compute_cost strips trailing -YYYYMMDD before lookup (house
        convention): a pinned snapshot ID prices identically."""
        usage = _u(input_tokens=1000, output_tokens=100)
        r = compute_cost("claude-opus-5-5-20260922", usage, now=date(2026, 9, 22))
        assert r.priced is True
        assert r.cost_usd == pytest.approx(1000 * 4.0 / 1e6 + 100 * 20.0 / 1e6,
                                           abs=APPROX)
