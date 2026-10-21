"""kdsn.366 onboarding pins — 2026-10-21: DeepSeek-V4.1-Flash-1017 (Synthetic).

Vendor facts (live GET /models on BOTH accounts, 2026-10-21):
  * hf:deepseek-ai/DeepSeek-V4.1-Flash-1017 — new snapshot, SUPERSEDING the
    2026-09-22-onboarded hf:deepseek-ai/DeepSeek-V4.1-Flash snapshot. 524288
    served, card max_output_length 65536. Declared effort vocab
    none/low/high/xhigh/max (no medium) — identical to the base snapshot.
  * Aliases repointed same day: syndsv41f / syn2dsv41f now resolve to the
    -1017 id in all 12 agent/headless TOMLs. The base id stays in the
    catalog for now.

Wire probes (tmp/kdsn366/, acct-1, 2026-10-21):
  * tools: finish=tool_calls, proper OpenAI shape.
  * streaming: SSE reasoning+content chunks.
  * effort (bounded 1000–4000 max_tokens): all six OA levels + no-param
    return 200 and terminate; "none" genuinely silences (0 reasoning
    chars); medium works (99 reasoning chars) despite being undeclared;
    volumes monotonic-ish single-sample (max 272 > high 147 ~ xhigh 152 >
    medium 99 > low 0 = none 0).
  * over-cap max_tokens=70000 tolerated (200) -> output cap None.
  * VISION WORKS on the -1017 snapshot (in contrast to the base snapshot's
    catalog-lie 3x HTTP-500, 2026-09-22): 64x64 solid-red PNG -> "Red",
    prompt_tokens=224 counted the image (text-only control ~15-20).

Code surface pinned here:
  1. _MODEL_CAPABILITIES: the -1017 row (524288, None, vision=True) BEFORE
     the base hf:deepseek-ai/deepseek-v4.1-flash row (base id is a
     substring of the -1017 id; first-match-wins).
  2. _SAMPLING_PROFILES: same DSV4.1-family pin, -1017 row first.
  3. _SYNTHETIC_EFFORT_OVERRIDES: all-six 1:1 map for -1017, entry first.
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

BASE = "hf:deepseek-ai/deepseek-v4.1-flash"
SNAP = "hf:deepseek-ai/deepseek-v4.1-flash-1017"
MODEL = "hf:deepseek-ai/DeepSeek-V4.1-Flash-1017"


def _cap_fragments():
    return [frag for frag, *_ in _MODEL_CAPABILITIES]


class TestDs1017Capabilities:
    def test_window_and_output_cap(self):
        assert model_context_window(MODEL) == 524_288
        assert _model_output_cap(MODEL) is None

    def test_vision_true(self):
        """Probed TRUE on 2026-10-21 (unlike the base snapshot's 3x 500)."""
        cfg = SimpleNamespace(model_vision={}, model_aliases={})
        assert model_supports_vision(MODEL, cfg) is True

    def test_row_before_base_snapshot(self):
        """Ordering pin: the base id is a substring of the -1017 id."""
        frags = _cap_fragments()
        assert frags.index(SNAP) < frags.index(BASE)

    def test_base_snapshot_row_untouched(self):
        """The base snapshot rows stay (catalog still serves it; no alias
        points at it after the repoint — retained for deliberation)."""
        assert model_context_window("hf:deepseek-ai/DeepSeek-V4.1-Flash") == 524_288


class TestDs1017SamplingAndEffort:
    def test_sampling_profile_family_pin(self):
        profs = {f: p for f, p in _SAMPLING_PROFILES}
        p = profs[SNAP]
        assert p.temperature == 1.0
        # follows the -1017 card (top_p 1.0 only), not the base row's 0.95
        assert p.top_p == 1.0
        assert profs[BASE].top_p == 0.95
        assert p.frequency_penalty is None
        assert p.presence_penalty is None

    def test_sampling_row_before_base(self):
        frags = [f for f, _ in _SAMPLING_PROFILES]
        assert frags.index(SNAP) < frags.index(BASE)

    def test_effort_override_all_six_1to1(self):
        m = next(m for f, m in _SYNTHETIC_EFFORT_OVERRIDES if f == SNAP)
        assert m == {"off": "none", "low": "low", "medium": "medium",
                     "high": "high", "xhigh": "xhigh", "max": "max"}

    def test_effort_row_before_base(self):
        frags = [f for f, _ in _SYNTHETIC_EFFORT_OVERRIDES]
        assert frags.index(SNAP) < frags.index(BASE)

    def test_base_effort_override_unchanged(self):
        m = next(m for f, m in _SYNTHETIC_EFFORT_OVERRIDES if f == BASE)
        assert m["off"] == "none" and m["max"] == "max"
