"""kdsn.362 fleet refresh onboarding pins — 2026-10-13: Sonnet 5.5, Haiku 5.5,
GLM-5.3 (non-flash, Synthetic).

Vendor facts (docs.anthropic.com models overview + pricing page, fetched
2026-10-13):
  * claude-sonnet-5-5: 1M ctx, 128K max output, vision. $2/$10 per MTok;
    5m write $2.50 (1.25x), 1h write $4 (2.0x); cache READ $0.10/MTok =
    0.05x base — the THIRD Anthropic model off uniform 0.1x (fable-5-1
    0.025x, opus-5-5 0.05x; pricing page: "0.05x on Claude Opus 5.5 and
    Claude Sonnet 5.5"). Adaptive always-on, default effort high.
    Retirement not before 2027-09-28.
  * claude-haiku-5-5: 1M ctx, 128K max output, vision. FIRST tiered-price
    Anthropic model — per-REQUEST prompt length (input + cache reads +
    cache writes, per the vendor doc) STRICTLY OVER 100,000 tokens pays
    the high tier ($0.50 in / $2.50 out), else the low tier ($0.10 in /
    $0.50 out). Cache reads are 0.1x base at BOTH tiers ($0.01 / $0.05),
    so the uniform CACHE_READ_MULT applies with no per-model pin. Writes
    1.25x/2.0x of the tier base ($0.125/$0.20 and $0.625/$1.0). Adaptive
    always-on, default effort medium. Retirement not before 2027-10-07.
  * hf:zai-org/GLM-5.3 (Synthetic — present on BOTH subscription accounts,
    vendor GET /models 2026-10-13): 524288 served, TEXT ONLY
    (input_modalities=["text"] — the first GLM catalog whose vision
    metadata matches the wire; fail-closed False agrees), card
    max_output_length 65536, declared effort vocab low/high/max.
    Wire-probed 2026-10-13 (tmp/glm53-probe/, acct-1): tool calls
    (finish=tool_calls, proper OpenAI shape), streaming reasoning deltas,
    reasoning_content present on assistant messages, over-cap
    max_tokens=70000 TOLERATED (200) -> output cap None. EFFORT HAZARD:
    'none' AND 'low' are pathological on this served deployment — both
    burned the entire 2000-token budget on reasoning with EMPTY content
    (finish_reason=length) on a 'reply with exactly: OK' prompt, while
    medium/high/xhigh/max terminate normally (~55-160 reasoning tokens).
    Differs from glm-5.3-flash, where 'none' genuinely zeroes reasoning
    and all six levels are sane (probe 2026-08-29). Served-template
    coercion quirk; no vendor fix noted as of onboarding.

Code surface pinned here:
  1. _MODEL_CAPABILITIES: ("sonnet-5-5", ...) BEFORE ("sonnet-5", ...)
     ("sonnet-5" is a substring of "claude-sonnet-5-5" — same trap class
     as opus-5-5/opus-5); ("haiku-5-5", ...) pinned 1M/128K (vs haiku-4-5's
     200K/64K); ("hf:zai-org/glm-5.3", ...) AFTER the "-flash" row
     ("hf:zai-org/glm-5.3" IS a substring of "hf:zai-org/glm-5.3-flash").
  2. _MODEL_PRICING: claude-sonnet-5-5 with explicit cached_input 0.10;
     claude-haiku-5-5 with the new "tiers" schema selected by prompt
     total (input + cache_read + cache_creation) strictly over 100K.
  3. _SYNTHETIC_EFFORT_OVERRIDES: hf:zai-org/glm-5.3 maps off/low/medium
     -> "high" (bounded; vendor-declared value), high 1:1, xhigh passed
     1:1 per the glm-5.3-flash convention (coercion above 'high'
     uncharacterized), max 1:1 (card-native). The flash entry keeps its
     all-six-1:1 map and stays FIRST (first-match-wins).
  4. Gates: _supports_adaptive_thinking True for both new Anthropic IDs;
     _supports_sampling_params False (fail-closed allowlist).
"""

from datetime import date
from types import SimpleNamespace

import pytest

from openalph.provider import (
    Usage,
    compute_cost,
    _MODEL_CAPABILITIES,
    _MODEL_PRICING,
    _SAMPLING_PROFILES,
    _SYNTHETIC_EFFORT_OVERRIDES,
    _supports_adaptive_thinking,
    _supports_sampling_params,
    _model_output_cap,
    model_context_window,
    model_supports_vision,
)

APPROX = 1e-9


def _u(**kw) -> Usage:
    return Usage(**kw)


def _cap_fragments():
    return [frag for frag, *_ in _MODEL_CAPABILITIES]


def _effort_map(frag):
    for f, m in _SYNTHETIC_EFFORT_OVERRIDES:
        if f == frag:
            return m
    raise AssertionError(f"no override entry for {frag}")


class TestSonnet55:
    def test_window_and_output_cap(self):
        assert model_context_window("claude-sonnet-5-5") == 1_048_576
        assert _model_output_cap("claude-sonnet-5-5") == 128_000

    def test_row_ordered_before_generic_sonnet5(self):
        """Ordering pin: 'sonnet-5' substring-matches 'claude-sonnet-5-5';
        the explicit row must sit FIRST (first-match-wins)."""
        frags = _cap_fragments()
        assert frags.index("sonnet-5-5") < frags.index("sonnet-5")
        # and the generic still resolves Sonnet 5 itself
        assert model_context_window("claude-sonnet-5") == 1_048_576

    def test_adaptive_thinking(self):
        assert _supports_adaptive_thinking("claude-sonnet-5-5") is True

    def test_sampling_params_fail_closed(self):
        assert _supports_sampling_params("claude-sonnet-5-5") is False

    def test_pricing_entry_shape(self):
        entry = _MODEL_PRICING["anthropic"]["claude-sonnet-5-5"]
        assert entry["input"] == 2.0
        assert entry["output"] == 10.0
        assert entry["cached_input"] == 0.10

    def test_full_cost_mix_005x_cache_read(self):
        usage = _u(input_tokens=1_000_000, output_tokens=100_000,
                   cache_read_tokens=200_000, cache_creation_tokens=20_000,
                   cache_creation_5m_tokens=8_000, cache_creation_1h_tokens=12_000)
        r = compute_cost("claude-sonnet-5-5", usage)
        # 2.0 (in) + 1.0 (out) + 200k*0.10/1e6 (0.02) + 8k*2.50/1e6 (0.02)
        #   + 12k*4/1e6 (0.048)
        assert r.priced is True
        assert r.cost_usd == pytest.approx(3.088, abs=APPROX)

    def test_date_suffix_normalized(self):
        usage = _u(input_tokens=1000, output_tokens=100)
        r = compute_cost("claude-sonnet-5-5-20261013", usage, now=date(2026, 10, 13))
        assert r.priced is True
        assert r.cost_usd == pytest.approx(1000 * 2.0 / 1e6 + 100 * 10.0 / 1e6,
                                           abs=APPROX)


class TestHaiku55:
    def test_window_and_output_cap(self):
        """1M/128K — larger than haiku-4-5's 200K window + 64K output cap."""
        assert model_context_window("claude-haiku-5-5") == 1_048_576
        assert _model_output_cap("claude-haiku-5-5") == 128_000
        assert model_context_window("claude-haiku-4-5") == 200_000
        assert _model_output_cap("claude-haiku-4-5") == 64_000

    def test_adaptive_thinking(self):
        """Haiku 5.5 is adaptive always-on — the whole point of the gate
        edit; haiku-4-5 stays on the legacy budget path (contrast)."""
        assert _supports_adaptive_thinking("claude-haiku-5-5") is True
        assert _supports_adaptive_thinking("claude-haiku-4-5") is False

    def test_sampling_params_fail_closed(self):
        assert _supports_sampling_params("claude-haiku-5-5") is False
        # contrast: haiku-4-5 is the last haiku on the legacy allowlist
        assert _supports_sampling_params("claude-haiku-4-5") is True

    def test_pricing_entry_is_tiered(self):
        entry = _MODEL_PRICING["anthropic"]["claude-haiku-5-5"]
        tiers = entry["tiers"]
        assert tiers["threshold_prompt_tokens"] == 100_000
        assert tiers["low"] == {"input": 0.10, "output": 0.50}
        assert tiers["high"] == {"input": 0.50, "output": 2.50}

    def test_low_tier_below_threshold(self):
        usage = _u(input_tokens=50_000, output_tokens=1_000)
        r = compute_cost("claude-haiku-5-5", usage)
        # 50k*0.10/1e6 + 1k*0.50/1e6 = 0.005 + 0.0005
        assert r.priced is True
        assert r.cost_usd == pytest.approx(0.0055, abs=APPROX)

    def test_high_tier_above_threshold(self):
        usage = _u(input_tokens=200_000, output_tokens=1_000)
        r = compute_cost("claude-haiku-5-5", usage)
        # 200k*0.50/1e6 + 1k*2.50/1e6 = 0.10 + 0.0025
        assert r.cost_usd == pytest.approx(0.1025, abs=APPROX)

    def test_threshold_is_strictly_over_100k(self):
        """Vendor: 'a request whose prompt is OVER 100,000 tokens pays
        higher prices' — exactly 100,000 stays on the low tier."""
        usage = _u(input_tokens=100_000, output_tokens=0)
        r = compute_cost("claude-haiku-5-5", usage)
        assert r.cost_usd == pytest.approx(100_000 * 0.10 / 1e6, abs=APPROX)

    def test_cache_tokens_count_toward_prompt_total(self):
        """The tier boundary counts input + cache reads + cache writes."""
        usage = _u(input_tokens=50_000, output_tokens=0,
                   cache_read_tokens=50_000, cache_creation_tokens=1)
        r = compute_cost("claude-haiku-5-5", usage)
        # total prompt = 100,001 > 100,000 -> high tier across ALL components:
        # 50k*0.50/1e6 + 50k*0.05/1e6(0.1x high base) + 1 token cache-write
        # at the fallback 2.0x (no 5m/1h split supplied) = 2.0*0.50/1e6
        expected = (50_000 * 0.50 + 50_000 * 0.05 + 1.0) / 1e6
        assert r.cost_usd == pytest.approx(expected, abs=1e-8)

    def test_uniform_cache_multiplier_both_tiers(self):
        """Cache reads are 0.1x of the tier base at BOTH tiers (vendor
        confirms $0.01 and $0.05) — no per-model cached_input pin."""
        low = compute_cost("claude-haiku-5-5",
                           _u(input_tokens=0, output_tokens=0,
                              cache_read_tokens=10_000))
        assert low.cost_usd == pytest.approx(10_000 * 0.01 / 1e6, abs=APPROX)
        high = compute_cost("claude-haiku-5-5",
                            _u(input_tokens=0, output_tokens=0,
                               cache_read_tokens=100_001))
        assert high.cost_usd == pytest.approx(100_001 * 0.05 / 1e6, abs=APPROX)

    def test_date_suffix_normalized(self):
        usage = _u(input_tokens=10_000, output_tokens=100)
        r = compute_cost("claude-haiku-5-5-20261013", usage, now=date(2026, 10, 13))
        assert r.priced is True
        assert r.cost_usd == pytest.approx(10_000 * 0.10 / 1e6 + 100 * 0.50 / 1e6,
                                           abs=APPROX)


class TestSyntheticGlm53NonFlash:
    def test_capabilities(self):
        assert model_context_window("hf:zai-org/GLM-5.3") == 524_288
        assert _model_output_cap("hf:zai-org/GLM-5.3") is None
        cfg = SimpleNamespace(model_vision={}, model_aliases={})
        assert model_supports_vision("hf:zai-org/GLM-5.3", cfg) is False

    def test_row_after_flash(self):
        """Ordering pin: 'hf:zai-org/glm-5.3' is a substring of
        'hf:zai-org/glm-5.3-flash' — flash must come FIRST so it keeps
        resolving to its own row (first-match-wins)."""
        frags = _cap_fragments()
        flash = frags.index("hf:zai-org/glm-5.3-flash")
        plain = frags.index("hf:zai-org/glm-5.3")
        assert flash < plain
        # flash still resolves as flash (vision False, 512K)
        assert model_context_window("hf:zai-org/GLM-5.3-Flash") == 524_288

    def test_sampling_profile_penalties_omitted(self):
        """GLM family pattern (kdsn.241.3): penalties pinned None = omit."""
        profs = {f: p for f, p in _SAMPLING_PROFILES}
        p = profs["hf:zai-org/glm-5.3"]
        assert p.frequency_penalty is None
        assert p.presence_penalty is None

    def test_effort_override_map(self):
        """off/low/medium -> high (bounded sane tier; 'none'/'low' are
        pathological at the wire — empty content after burning the whole
        budget). high/xhigh/max per the flash convention."""
        m = _effort_map("hf:zai-org/glm-5.3")
        assert m["off"] == "high"
        assert m["low"] == "high"
        assert m["medium"] == "high"
        assert m["high"] == "high"
        assert m["xhigh"] == "xhigh"
        assert m["max"] == "max"

    def test_effort_flash_row_first_and_untouched(self):
        frags = [f for f, _ in _SYNTHETIC_EFFORT_OVERRIDES]
        assert frags.index("hf:zai-org/glm-5.3-flash") < frags.index("hf:zai-org/glm-5.3")
        m = _effort_map("hf:zai-org/glm-5.3-flash")
        assert m == {"off": "none", "low": "low", "medium": "medium",
                     "high": "high", "xhigh": "xhigh", "max": "max"}
