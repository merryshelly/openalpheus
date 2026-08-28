"""Tests for blackwell (local SGLang rig) reasoning_effort support (workspace-kdsn.301).

Spec: memory/projects/openalph/specs/kdsn.301-blackwell-effort-spec.md
Wire evidence (curl probes 2026-08-28 vs router http://10.0.20.111:8000,
SGLang 0.5.18, served model qwen38-27b-fp8, same prompt per level):
  none -> 200, reasoning_tokens=0       (true thinking-off)
  low  -> 200, reasoning_tokens=212
  medium -> 200, reasoning_tokens=258
  xhigh -> 200, reasoning_tokens=2733
  high, max, garbage -> 400 (LOUD reject; never send)
  (omitted) -> 200, low/medium-like default (NOT vLLM's xhigh template default)

Problem under test: provider_key="blackwell" hits no effort branch in
_build_openai_kwargs (not _supports_reasoning_extra, not fireworks/synthetic,
not the DSv4 gate) -> '/effort' was a silent no-op on the fleet's speed-tier
default. kdsn.271 bug class: silent server-default override of operator intent.
"""

import logging

import pytest

from openalph import provider as provider_module
from openalph.provider import _build_openai_kwargs


class TestBlackwellEffortMapping:
    """Always-explicit top-level reasoning_effort for provider_key="blackwell"
    (kdsn.301). Map mirrors _SYNTHETIC_EFFORT_OVERRIDES['hf:qwen/qwen3.8']
    with ONE forced divergence: the SGLang backend 400s on 'high' (that
    backend accepted it), so high collapses DOWN to medium + warn-once —
    never UP to xhigh, which would silently ~10x the reasoning burn."""

    @pytest.fixture(autouse=True)
    def _clear_blackwell_warned(self):
        """Isolate the warn-once set between tests (house convention:
        _SYNTHETIC_EFFORT_WARNED in test_synthetic_provider.py)."""
        provider_module._BLACKWELL_EFFORT_WARNED.clear()
        yield
        provider_module._BLACKWELL_EFFORT_WARNED.clear()

    def _args(self, level, **overrides):
        args = dict(
            api_model="qwen38-27b-fp8",
            system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None,
            max_tokens=1024,
            thinking_level=level,
            quirks=["reasoning_replay"],
            provider_key="blackwell",
        )
        args.update(overrides)
        return args

    def _effort(self, kw):
        return kw.get("extra_body", {}).get("reasoning_effort")

    # --- discriminating tests (red pre-implementation) ---

    def test_off_maps_to_none(self):
        """off must send reasoning_effort='none' explicitly — an omitted param
        falls back to the server default (silent override, kdsn.271 class).
        Wire-verified: 'none' yields reasoning_tokens=0 (true thinking-off)."""
        kw = _build_openai_kwargs(**self._args("off"))
        assert self._effort(kw) == "none"

    @pytest.mark.parametrize("level", ["low", "medium", "xhigh"])
    def test_low_medium_xhigh_passthrough(self, level):
        """Native wire vocabulary: low/medium/xhigh pass through 1:1
        (200s, genuinely distinguished reasoning volume). NOTE: unlike the
        synthetic default map, xhigh does NOT collapse — it is card-native
        for qwen3.8 and wire-verified distinguished (2733 vs 258 reasoning
        tokens vs medium on the same prompt)."""
        kw = _build_openai_kwargs(**self._args(level))
        assert self._effort(kw) == level

    def test_high_maps_to_medium_with_warning(self, caplog):
        """'high' is a 400 on this backend. Collapse DOWN to medium + warn
        (downward tier-drop, lossy) — NEVER up to xhigh: mapping up would
        silently multiply reasoning volume ~10x against operator intent."""
        with caplog.at_level(logging.WARNING):
            kw = _build_openai_kwargs(**self._args("high"))
        assert self._effort(kw) == "medium"
        assert any(
            "high" in r.message and "reasoning_effort" in r.message
            for r in caplog.records
        ), f"expected remap warning, got: {[r.message for r in caplog.records]}"

    def test_max_maps_to_xhigh_with_warning(self, caplog):
        """'max' is a 400 here (accepted on synthetic's qwen3.8 backend —
        per-backend contract, not model-card truth). Ceiling-map to xhigh +
        warn-once, mirroring the synthetic qwen3.8 override."""
        with caplog.at_level(logging.WARNING):
            kw = _build_openai_kwargs(**self._args("max"))
        assert self._effort(kw) == "xhigh"
        assert any(
            "max" in r.message and "reasoning_effort" in r.message
            for r in caplog.records
        ), f"expected remap warning, got: {[r.message for r in caplog.records]}"

    def test_remap_warns_once_per_level(self, caplog):
        """Warn-once per OA level, not warn-never / warn-spam (fleet log
        hygiene). high x3 = exactly 1 warning; high+max = 2 distinct."""
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                _build_openai_kwargs(**self._args("high"))
            for _ in range(3):
                _build_openai_kwargs(**self._args("max"))
        warns = [r for r in caplog.records
                 if "reasoning_effort" in r.message and r.levelno >= logging.WARNING]
        assert len(warns) == 2, f"expected exactly 2 warnings, got {len(warns)}"

    def test_no_warning_for_exact_levels(self, caplog):
        """1:1 mappings (incl. xhigh passthrough) never warn; off->none is a
        disable-alias, not a tier drop."""
        with caplog.at_level(logging.WARNING):
            for level in ("off", "low", "medium", "xhigh"):
                _build_openai_kwargs(**self._args(level))
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warns == [], f"unexpected warnings: {[r.message for r in warns]}"

    def test_no_nested_reasoning_object(self):
        """Blackwell takes TOP-LEVEL reasoning_effort, never OpenRouter's
        nested reasoning.effort. The _supports_reasoning_extra gate must not
        capture provider_key="blackwell" even with thinking enabled."""
        kw = _build_openai_kwargs(**self._args("medium"))
        assert "reasoning" not in kw.get("extra_body", {})

    def test_reasoning_coexists_with_routing(self):
        """effort and a routing dict share extra_body (openrouter-style
        `provider` key must not clobber reasoning_effort)."""
        kw = _build_openai_kwargs(**self._args(
            "low", routing={"some": "route"}))
        assert kw["extra_body"]["reasoning_effort"] == "low"
        assert kw["extra_body"]["provider"] == {"some": "route"}

    def test_no_anthropic_thinking_param(self):
        """The openai-type path must never emit an Anthropic `thinking` kwarg."""
        kw = _build_openai_kwargs(**self._args("xhigh"))
        assert "thinking" not in kw
        assert "thinking" not in kw.get("extra_body", {})

    @pytest.mark.parametrize("level", ["garbage", "", None])
    def test_unknown_level_falls_back_to_medium_no_warning(self, caplog, level):
        """Spec'd fallback: out-of-enum values (typo'd advisor TOML, corrupted
        restore data) map to 'medium' silently — never sent verbatim, never
        warned (warnings are reserved for the two valid lossy tier-drops).
        Pins audit NOTE-3 2026-08-28."""
        with caplog.at_level(logging.WARNING):
            kw = _build_openai_kwargs(**self._args(level))
        assert self._effort(kw) == "medium"
        warns = [r for r in caplog.records if "reasoning_effort" in r.message]
        assert warns == [], f"unexpected warnings: {[r.message for r in warns]}"

    # --- guards (expected green already; their job is staying green) ---

    def test_other_providers_unaffected(self):
        """The blackwell branch must not capture other provider keys:
        openrouter keeps nested reasoning.effort, fireworks keeps its 1:1
        top-level passthrough, synthetic keeps its default collapse."""
        kw = _build_openai_kwargs(
            api_model="test/model", system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None, max_tokens=1024,
            thinking_level="high", quirks=[], provider_key="openrouter",
        )
        assert kw["extra_body"]["reasoning"] == {"effort": "high"}
        assert "reasoning_effort" not in kw["extra_body"]

        provider_module._SYNTHETIC_EFFORT_WARNED.clear()
        kw = _build_openai_kwargs(
            api_model="accounts/fireworks/models/kimi-k3", system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None, max_tokens=1024,
            thinking_level="high", quirks=[], provider_key="fireworks",
        )
        assert kw["extra_body"]["reasoning_effort"] == "high"

        provider_module._SYNTHETIC_EFFORT_WARNED.clear()
        kw = _build_openai_kwargs(
            api_model="hf:zai-org/GLM-5.2", system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None, max_tokens=1024,
            thinking_level="xhigh", quirks=[], provider_key="synthetic",
        )
        assert kw["extra_body"]["reasoning_effort"] == "high"

    def test_macstudio_dsv4_unchanged(self):
        """Guard: DSv4-flash on macstudio keeps its prefix + top-level-none
        behavior — the blackwell branch must not preempt the model-name gate."""
        kw = _build_openai_kwargs(
            api_model="macstudio/deepseek-v4-flash", system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None, max_tokens=1024,
            thinking_level="off", quirks=[], provider_key="macstudio",
        )
        assert kw["extra_body"]["reasoning_effort"] == "none"
        assert kw["messages"][0]["content"] == "sys"
