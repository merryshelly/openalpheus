"""Tests for the Fable 5.1 upgrade (onboarded 2026-09-13, SB-ordered).

Model facts verified against platform.claude.com docs (models overview +
Fable 5.1 page + pricing page, 2026-09-13):

  - claude-fable-5-1: 1M context, 128K output cap, vision, adaptive thinking
    always-on (default effort high), released 2026-09-01.
  - Pricing: $10/$50 per MTok — same as fable-5 — but cache HITS are 0.025x
    base input ($0.25/MTok), the ONLY Anthropic model breaking the uniform
    0.1x CACHE_READ_MULT. Cache writes stay standard (5m 1.25x / 1h 2.0x).
  - carried in _MODEL_PRICING as a per-model absolute "cached_input" rate,
    same field convention as the fireworks namespace.

Scope pinned here (provider.py only; 5.1 rides the existing shared Anthropic
inference path — _supports_adaptive_thinking's generic "fable" fragment and
the fail-closed sampling allowlist already cover it):

  1. _MODEL_CAPABILITIES: explicit fable-5-1 row BEFORE the generic "fable"
     row (first-match-wins ordering pin).
  2. compute_cost: 5.1 cache reads at 0.025x, fable-5 still 0.1x (contrast).
  3. Adaptive-thinking + sampling-allowlist gates for the 5-1 string.
"""

import pytest
from datetime import date

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
    defaults = dict(input_tokens=0, output_tokens=0, cache_read_tokens=0,
                    cache_creation_tokens=0, cache_creation_5m_tokens=0,
                    cache_creation_1h_tokens=0)
    defaults.update(kw)
    return Usage(**defaults)


class TestFable51Capabilities:
    def test_window_output_cap_vision_row(self):
        assert model_context_window("claude-fable-5-1") == 1_048_576
        assert model_context_window("anthropic/claude-fable-5-1") == 1_048_576
        assert _model_output_cap("claude-fable-5-1") == 128_000

    def test_specific_fragment_precedes_generic_fable(self):
        # First-match-wins: if "fable" ever ships legacy specs, the explicit
        # 5-1 row must still win for the 5-1 string.
        frags = [frag for frag, *_ in _MODEL_CAPABILITIES]
        assert frags.index("fable-5-1") < frags.index("fable")

    def test_adaptive_thinking_on(self):
        # 5.1 is adaptive always-on (default effort high); the generic "fable"
        # fragment must keep matching the 5-1 string.
        assert _supports_adaptive_thinking("claude-fable-5-1") is True

    def test_sampling_params_rejected(self):
        # Fail-closed allowlist: 5.1 (like fable-5) must never receive
        # temperature/top_p/top_k.
        assert _supports_sampling_params("claude-fable-5-1") is False


class TestFable51Pricing:
    def test_cache_read_0025x_not_01x(self):
        # input 100k, output 50k, cache_read 200k, 5m write 8k, 1h write 12k
        usage = _u(input_tokens=100_000, output_tokens=50_000,
                   cache_read_tokens=200_000, cache_creation_tokens=20_000,
                   cache_creation_5m_tokens=8_000, cache_creation_1h_tokens=12_000)
        r = compute_cost("claude-fable-5-1", usage)
        # 1.0 (in) + 2.5 (out) + 200k*0.25/1e6 (0.05) + 8k*12.5/1e6 (0.10)
        #   + 12k*20/1e6 (0.24) = 3.89
        assert r.priced is True
        assert r.unpriced_tokens == 0
        assert r.cost_usd == pytest.approx(3.89, abs=APPROX)

    def test_fable5_still_uniform_01x(self):
        # Contrast: same usage on fable-5 — cache read at 0.1x of $10 = $1/MTok.
        usage = _u(input_tokens=100_000, output_tokens=50_000,
                   cache_read_tokens=200_000, cache_creation_tokens=20_000,
                   cache_creation_5m_tokens=8_000, cache_creation_1h_tokens=12_000)
        r = compute_cost("claude-fable-5", usage)
        # 1.0 + 2.5 + 0.20 + 0.10 + 0.24 = 4.04
        assert r.priced is True
        assert r.cost_usd == pytest.approx(4.04, abs=APPROX)

    def test_prefix_normalized_and_effective_dates(self):
        usage = _u(input_tokens=10_000, output_tokens=5_000)
        r = compute_cost("anthropic/claude-fable-5-1", usage)
        assert r.priced is True
        assert r.cost_usd == pytest.approx(0.35, abs=APPROX)  # 0.1 + 0.25
        # Flat rates across the 5.1 lifecycle (released 2026-09-01).
        r2 = compute_cost("claude-fable-5-1", usage, now=date(2027, 9, 1))
        assert r2.cost_usd == pytest.approx(0.35, abs=APPROX)

    def test_entry_in_pricing_table_with_cached_rate(self):
        entry = _MODEL_PRICING["anthropic"]["claude-fable-5-1"]
        assert entry["input"] == 10.0 and entry["output"] == 50.0
        assert entry["cached_input"] == 0.25
