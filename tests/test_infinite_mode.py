"""Tests for Synthetic Infinite Mode registration (workspace-im7t.82).

Wire evidence (probe 2026-10-21, tmp/infinite-probe/, both accounts Big plan):
  - model string 'infinite:small:vision' (only infinite: id in /openai/v1/models;
    resolves vendor-side to hf:deepseek-ai/DeepSeek-V4.1-Flash)
  - ALL SIX reasoning_effort levels return 200 (probe1 A/B + probe3 matrix:
    none=0 reasoning tokens — genuine off; medium/high/xhigh/max all 200)
  - tool calling works (streamed tool_calls deltas, finish_reason "tool_calls")
  - reasoning_content replay in assistant turns: 200 (no no_reasoning_replay
    quirk needed on the synthetic-inf block)
  - prompt caching live (created_cache_tokens=31488 on a 31.5k prefill)
  - timing: ~9s trivial / 950.9s 3.6k-tok generation (max CONTENT gap 79s)
    / 20.5s 31.5k-token prefill; infinite calls consume NO quota counters

Registration under test (mirrors the syndsv41f onboarding pattern):
  1. _MODEL_CAPABILITIES  ("infinite:", 524288, None, False) — vision fail-closed
     (plain dsv41f is wire-probed image-broken; the infinite alias serves that
     same model despite the "vision" label in the alias name)
  2. _MODEL_PRICING       "synthetic-inf" namespace, zero-cost entry (free tier:
     priced at $0, NEVER unpriced — unpriced tokens pollute /status caveats)
  3. _SAMPLING_PROFILES   dsv41f-aligned anti-collapse pin (temp 1.0 / top_p 0.95)
  4. _SYNTHETIC_EFFORT_OVERRIDES  "infinite:" fragment, 1:1 all six levels
     (kills the xhigh->high remap warn observed live on wonmun's first exec)
"""

import logging

import pytest

from openalph import provider as provider_module
from openalph.provider import (
    _build_openai_kwargs,
    _model_output_cap,
    _MODEL_CAPABILITIES,
    _sampling_profile,
    _VISION_WARNED,
    compute_cost,
    model_context_window,
    model_supports_vision,
    Usage,
)

INF_MODEL = "infinite:small:vision"
INF_BASE_URL = "https://api.synthetic.new/openai/v1"


def _first_matching_fragment(api_model: str):
    """Simulate the table's first-match-wins scan (frag in api_model.lower())."""
    m = api_model.lower()
    for frag, *_rest in _MODEL_CAPABILITIES:
        if frag in m:
            return frag
    return None


class _VisionCfg:
    """Duck-typed config for model_supports_vision (provider.py avoids
    importing openalph.config)."""

    model_aliases = {}
    model_vision = {}


class TestInfiniteCapabilities:
    def test_context_window_512k(self):
        assert model_context_window(INF_MODEL) == 524_288

    def test_output_cap_none(self):
        """Synthetic is openai-compatible and tolerates over-cap requests —
        no clamp, same as the dsv41f rows."""
        assert _model_output_cap(INF_MODEL) is None

    def test_first_match_is_infinite_fragment(self):
        assert _first_matching_fragment(INF_MODEL) == "infinite:"

    def test_vision_fail_closed_known_model(self, caplog):
        """Vision False (matching plain dsv41f's wire-probed image-broken
        deployment), AND — because the model is now REGISTERED — no
        fail-closed warn-once (the warn the first wonmun exec emitted)."""
        caplog.set_level(logging.WARNING, logger="openalph.provider")
        _VISION_WARNED.clear()
        assert model_supports_vision(INF_MODEL, _VisionCfg()) is False
        assert INF_MODEL not in _VISION_WARNED
        assert "no curated vision capability" not in caplog.text

    def test_dsv41f_rows_unaffected(self):
        """Registration must not perturb the existing dsv41f rows."""
        assert model_context_window("hf:deepseek-ai/deepseek-v4.1-flash") == 524_288
        assert model_context_window("hf:deepseek-ai/deepseek-v4.1-flash-1017") == 524_288
        m = "hf:deepseek-ai/deepseek-v4.1-flash-1017".lower()
        for frag, _w, _c, vision in _MODEL_CAPABILITIES:
            if frag in m:
                assert vision is True  # -1017 wire-probed vision-true (kdsn.366)
                break
        m = "hf:deepseek-ai/deepseek-v4.1-flash".lower()
        for frag, _w, _c, vision in _MODEL_CAPABILITIES:
            if frag in m:
                assert vision is False
                break


class TestInfinitePricing:
    def test_free_tier_priced_at_zero(self):
        """Free tier must be PRICED at $0 — not unpriced. Unpriced tokens
        land in the /status caveat bucket; a registered free tier should
        read as a clean $0.00."""
        r = compute_cost(
            INF_MODEL,
            Usage(input_tokens=1000, output_tokens=2000),
            provider_key="synthetic-inf",
            provider_type="openai",
        )
        assert r.cost_usd == 0.0
        assert r.priced is True
        assert r.unpriced_tokens == 0

    def test_cache_tokens_price_at_zero(self):
        """The endpoint caches prompts (created_cache_tokens probed); cache
        reads/writes on the inf block price at the registered $0 cached_input
        rate — and must NOT trip the missing-cached_rate warn-once path."""
        r = compute_cost(
            INF_MODEL,
            Usage(input_tokens=500, output_tokens=100,
                  cache_read_tokens=31_488, cache_creation_tokens=31_488),
            provider_key="synthetic-inf",
            provider_type="openai",
        )
        assert r.cost_usd == 0.0
        assert r.priced is True
        assert r.unpriced_tokens == 0


class TestInfiniteSampling:
    def test_profile_matches_dsv41f_base(self):
        """Anti-collapse pin mirrors hf:deepseek-ai/deepseek-v4.1-flash:
        temperature 1.0, top_p 0.95, penalties omitted (None = not sent)."""
        p = _sampling_profile(INF_MODEL)
        assert p.temperature == 1.0
        assert p.top_p == 0.95
        assert p.frequency_penalty is None
        assert p.presence_penalty is None


class TestInfiniteEffort:
    @pytest.fixture(autouse=True)
    def _clear_warned(self):
        """Isolate the per-process warn-once sets (house convention)."""
        provider_module._SYNTHETIC_EFFORT_WARNED.clear()
        provider_module._SYNTHETIC_IDENTITY_WARNED.clear()
        yield
        provider_module._SYNTHETIC_EFFORT_WARNED.clear()
        provider_module._SYNTHETIC_IDENTITY_WARNED.clear()

    def _args(self, level, **overrides):
        args = dict(
            api_model=INF_MODEL,
            system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None,
            max_tokens=1024,
            thinking_level=level,
            quirks=["reasoning_replay"],
            provider_key="synthetic-inf",
            base_url=INF_BASE_URL,
        )
        args.update(overrides)
        return args

    def _effort(self, kw):
        return kw.get("extra_body", {}).get("reasoning_effort")

    @pytest.mark.parametrize(
        "level,expected",
        [("off", "none"), ("low", "low"), ("medium", "medium"),
         ("high", "high"), ("xhigh", "xhigh"), ("max", "max")],
    )
    def test_all_six_levels_1to1(self, level, expected):
        """Wire-probed 200 at every level (probe1 + probe3). Unlike the
        generic Synthetic default map (xhigh/max -> high + warn), the
        infinite endpoint accepts the full vocabulary — pass 1:1."""
        kw = _build_openai_kwargs(**self._args(level))
        assert self._effort(kw) == expected

    def test_off_sends_none_explicitly(self):
        """off must send reasoning_effort='none' explicitly — an omitted param
        lets the server default (reasoning ON) silently override operator
        intent (kdsn.271 class; probe1 call A showed ~35 reasoning tokens
        on a trivial prompt with no effort param sent)."""
        kw = _build_openai_kwargs(**self._args("off"))
        assert self._effort(kw) == "none"

    def test_no_remap_warning_on_xhigh(self, caplog):
        """The live wonmun exec warned 'Synthetic does not support
        reasoning_effort=xhigh; remapping to high' — the infinite override
        entry must eliminate that warn for this model."""
        caplog.set_level(logging.WARNING, logger="openalph.provider")
        _build_openai_kwargs(**self._args("xhigh"))
        assert "remapping" not in caplog.text

    def test_no_identity_mismatch_warning_for_renamed_block(self, caplog):
        """[providers.synthetic-inf] is a renamed Synthetic block; service
        identity is keyed on base_url host (kdsn.348.2), so the mapping
        applies AND the mis-wire warn (synthetic-shaped name, wrong host)
        never fires."""
        caplog.set_level(logging.WARNING, logger="openalph.provider")
        _build_openai_kwargs(**self._args("high"))
        assert "will NOT" not in caplog.text
