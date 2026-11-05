"""Kimi-K3.1-Pro onboarding pins — 2026-11-04 (Synthetic, both accounts).

Vendor facts (live GET /models on BOTH accounts, 2026-11-04):
  * hf:moonshotai/Kimi-K3.1-Pro — 524288 served, card max_output_length
    65536, quantization mxfp4, catalog input_modalities ["text","image"],
    declared effort vocab low/high/max. Vendor-platform pricing
    $2/$10/$0.20-cached per MTok - informational only: Synthetic is
    subscription-metered, so K3.1 rides the unpriced-subscription path
    (no _MODEL_PRICING entry, same class as synkimi3/synglm53/syndsv41f).
  * Aliases synkimi31 / syn2kimi31 added to all 10 live fleet TOMLs.
    Advisor repointed fleet-wide to synkimi31 (SB steering 2026-11-04).

Wire probes (tmp/k31-probe/, BOTH accounts, 2026-11-04):
  * effort matrix: all six OA levels return 200 and terminate normally
    (reasoning tokens acct1 none/low/med/high/xhigh/max = 6/10/14/24/30/13;
    acct2 10/11/27/24/25/34). "none" is SANE - no glm-5.3-style empty-
    content budget burn. Differentiation above high is non-monotonic at
    single-sample scale (acct1 max 13 < xhigh 30) - do not cite xhigh/max
    as distinct tiers above high. Map: all six 1:1.
  * VISION REAL: 256x256 solid-red PNG -> prompt_tokens 208 vs 99 text-only
    baseline, answered "Red"; baseline answered "Unknown" (both accounts).
  * over-cap max_tokens=70000 tolerated (200, card cap 65536) -> cap None.
  * reasoning_content replay accepted (200; follow-up answered in-thread).

Code surface pinned here:
  1. _MODEL_CAPABILITIES: the k3.1-pro row (524288, None, vision=True)
     BEFORE the hf:moonshotai/kimi-k3 row (that fragment is a substring
     of the K3.1 id; first-match-wins).
  2. _SAMPLING_PROFILES: Kimi-family no-penalty pin, k3.1 row first.
  3. _SYNTHETIC_EFFORT_OVERRIDES: all-six 1:1 map for the k3.1 fragment.
"""

from types import SimpleNamespace

from openalph.provider import (
    _MODEL_CAPABILITIES,
    _SAMPLING_PROFILES,
    _SYNTHETIC_EFFORT_OVERRIDES,
    _model_output_cap,
    model_context_window,
    model_supports_vision,
)

K3 = "hf:moonshotai/kimi-k3"
K31 = "hf:moonshotai/kimi-k3.1-pro"
MODEL = "hf:moonshotai/Kimi-K3.1-Pro"


class TestK31Capabilities:
    def test_window_and_output_cap(self):
        assert model_context_window(MODEL) == 524_288
        assert _model_output_cap(MODEL) is None

    def test_vision_true(self):
        """Probed TRUE 2026-11-04 on both accounts (real catalog claim,
        unlike the glm-5.3-flash / dsv41f-base metadata lies)."""
        cfg = SimpleNamespace(model_vision={}, model_aliases={})
        assert model_supports_vision(MODEL, cfg) is True

    def test_row_before_kimi_k3(self):
        """Ordering pin: "hf:moonshotai/kimi-k3" is a substring of the
        K3.1 id; first-match-wins means a later K3 row would silently
        apply K3's facts to K3.1 (here both are 524288/None today, but
        the row must pin K3.1's facts independently of K3 edits)."""
        frags = [frag for frag, *_ in _MODEL_CAPABILITIES]
        assert frags.index(K31) < frags.index(K3)

    def test_kimik3_row_untouched(self):
        assert model_context_window("hf:moonshotai/Kimi-K3") == 524_288


class TestK31SamplingAndEffort:
    def test_sampling_profile_kimi_family_pin(self):
        profs = {f: p for f, p in _SAMPLING_PROFILES}
        p = profs[K31]
        assert p.frequency_penalty is None
        assert p.presence_penalty is None
        # temp/top_p left to served defaults (omit), mirroring the K3 row
        assert p.temperature is None
        assert p.top_p is None

    def test_sampling_row_before_kimi_k3(self):
        frags = [f for f, _ in _SAMPLING_PROFILES]
        assert frags.index(K31) < frags.index(K3)

    def test_effort_override_all_six_1to1(self):
        m = next(m for f, m in _SYNTHETIC_EFFORT_OVERRIDES if f == K31)
        assert m == {"off": "none", "low": "low", "medium": "medium",
                     "high": "high", "xhigh": "xhigh", "max": "max"}

    def test_no_synthetic_pricing_entry(self):
        """Subscription class: K3.1 must NOT appear in _MODEL_PRICING
        (unpriced-subscription path, same class as synkimi3)."""
        from openalph.provider import _MODEL_PRICING
        assert "synthetic" not in _MODEL_PRICING
        for namespace, table in _MODEL_PRICING.items():
            assert not any("k3.1" in k for k in table), \
                f"unexpected K3.1 pricing in {namespace}"
