"""Tests for per-model sampling profiles on OpenAI-compatible providers (kdsn.241.3).

OpenAlph historically hardcoded ``frequency_penalty=0.3`` on EVERY OpenAI-compat
call. Vendor docs + the LZ-Penalty paper show this is corrosive to long-reasoning
models (GLM-5.2, Kimi K2.6): the penalty accumulates over tens of thousands of
reasoning tokens until common/structural tokens are banned, producing repetition
collapse and malformed tool calls. Vendors ship 0 (Moonshot hard-pins Kimi to 0;
Fireworks/Baseten quickstarts use 0/0; Z.AI omits the param entirely).

Fix: replace the unconditional 0.3 with roster-driven per-model sampling
profiles. Default now OMITS penalties (was 0.3). GLM/Kimi are pinned to omit,
independent of whatever the default becomes.
"""
import pytest

from openalph.provider import (
    _build_anthropic_kwargs,
    _build_openai_kwargs,
    _sampling_profile,
    SamplingProfile,
    _DEFAULT_SAMPLING_PROFILE,
)


def _base_args(**overrides):
    defaults = dict(
        api_model="test/model",
        system="sys",
        provider_messages=[{"role": "user", "content": "hi"}],
        provider_tools=None,
        max_tokens=1024,
        thinking_level="off",
        quirks=[],
    )
    defaults.update(overrides)
    return defaults


class TestDefaultOmitsFrequencyPenalty:
    """The old unconditional 0.3 is gone — default profile sends no penalties."""

    def test_openai_omits_frequency_penalty_by_default(self):
        kw = _build_openai_kwargs(**_base_args(provider_key="openai"))
        assert "frequency_penalty" not in kw

    def test_default_provider_omits_frequency_penalty(self):
        kw = _build_openai_kwargs(**_base_args())
        assert "frequency_penalty" not in kw

    def test_default_omits_presence_penalty(self):
        kw = _build_openai_kwargs(**_base_args(provider_key="openai"))
        assert "presence_penalty" not in kw

    def test_no_penalty_value_is_the_legacy_0_3(self):
        """Regression guard: the specific harmful value 0.3 must never appear."""
        kw = _build_openai_kwargs(**_base_args(provider_key="openai"))
        assert kw.get("frequency_penalty") != 0.3


class TestVendorPinnedModels:
    """GLM-5.2 and Kimi K2.6 must never receive penalties (vendor requirement)."""

    def test_glm_omits_penalties(self):
        kw = _build_openai_kwargs(**_base_args(
            api_model="accounts/fireworks/models/glm-5p2", provider_key="fireworks",
        ))
        assert "frequency_penalty" not in kw
        assert "presence_penalty" not in kw

    def test_kimi_omits_penalties(self):
        kw = _build_openai_kwargs(**_base_args(
            api_model="accounts/fireworks/models/kimi-k2p6", provider_key="fireworks",
        ))
        assert "frequency_penalty" not in kw
        assert "presence_penalty" not in kw


class TestSamplingProfileLookup:
    """_sampling_profile() resolves per-model profiles with fragment matching."""

    def test_unknown_model_returns_default(self):
        assert _sampling_profile("some/unknown-model") is _DEFAULT_SAMPLING_PROFILE

    def test_glm_fragment_matches(self):
        prof = _sampling_profile("accounts/fireworks/models/GLM-5p2")  # case-insensitive
        assert prof.frequency_penalty is None
        assert prof.presence_penalty is None

    def test_kimi_fragment_matches(self):
        prof = _sampling_profile("accounts/fireworks/models/kimi-k2p6")
        assert prof.frequency_penalty is None
        assert prof.presence_penalty is None

    def test_default_profile_has_no_penalties(self):
        assert _DEFAULT_SAMPLING_PROFILE.frequency_penalty is None
        assert _DEFAULT_SAMPLING_PROFILE.presence_penalty is None


class TestProfileApplicationMechanism:
    """The mechanism DOES send penalties when a profile specifies them (and the
    provider supports them) — proves the profile isn't just always-omit."""

    def test_nonzero_frequency_penalty_is_sent(self, monkeypatch):
        monkeypatch.setattr(
            "openalph.provider._sampling_profile",
            lambda m: SamplingProfile(frequency_penalty=0.5, presence_penalty=0.2),
        )
        kw = _build_openai_kwargs(**_base_args(provider_key="openai"))
        assert kw["frequency_penalty"] == 0.5
        assert kw["presence_penalty"] == 0.2

    def test_google_never_gets_penalties_even_if_profile_sets_them(self, monkeypatch):
        """Google rejects penalties — the support gate wins over the profile."""
        monkeypatch.setattr(
            "openalph.provider._sampling_profile",
            lambda m: SamplingProfile(frequency_penalty=0.5, presence_penalty=0.2),
        )
        kw = _build_openai_kwargs(**_base_args(provider_key="google"))
        assert "frequency_penalty" not in kw
        assert "presence_penalty" not in kw

    def test_zero_penalty_is_explicitly_sent_when_profile_sets_zero(self, monkeypatch):
        """A profile can assert an explicit 0.0 (distinct from None=omit)."""
        monkeypatch.setattr(
            "openalph.provider._sampling_profile",
            lambda m: SamplingProfile(frequency_penalty=0.0),
        )
        kw = _build_openai_kwargs(**_base_args(provider_key="openai"))
        assert kw["frequency_penalty"] == 0.0


class TestSamplingProfileTemperatureTopP:
    """kdsn.241.3.1: SamplingProfile extended with temperature/top_p.

    Root cause of the 2026-07-20 MiniMax-M3 runaway-generation incident
    (task ran 30+ min / 22k+ tokens with no EOS on !pySmWhiuIGZElv1Wul):
    openalph sent no temperature/top_p override for macstudio models, so
    the request fell back to llama.cpp server defaults (temp=0.80) instead
    of MiniMax's vendor-recommended 1.0 — a known mode-collapse setup.
    """

    def test_default_profile_omits_temperature_and_top_p(self):
        assert _DEFAULT_SAMPLING_PROFILE.temperature is None
        assert _DEFAULT_SAMPLING_PROFILE.top_p is None

    def test_minimax_m3_fragment_matches_full_alias(self):
        prof = _sampling_profile("macstudio/mlx-community/MiniMax-M3-4bit")
        assert prof.temperature == 1.0
        assert prof.top_p == 0.95

    def test_minimax_m3_fragment_matches_bare_model_id_case_insensitive(self):
        prof = _sampling_profile("MiniMax-M3")
        assert prof.temperature == 1.0
        assert prof.top_p == 0.95

    def test_minimax_m3_profile_flows_to_request(self):
        kw = _build_openai_kwargs(**_base_args(
            api_model="mlx-community/MiniMax-M3-4bit", provider_key="macstudio",
        ))
        assert kw["temperature"] == 1.0
        assert kw["top_p"] == 0.95

    def test_profile_temperature_wins_over_per_agent_override(self):
        """Profile is a vendor-pinned/hard requirement — it wins over
        whatever the agent's own TOML config (temperature/top_p) passes in."""
        kw = _build_openai_kwargs(**_base_args(
            api_model="mlx-community/MiniMax-M3-4bit",
            provider_key="macstudio",
            temperature=0.7,
            top_p=0.5,
        ))
        assert kw["temperature"] == 1.0
        assert kw["top_p"] == 0.95

    def test_unprofiled_model_keeps_per_agent_override(self):
        """No profile temperature set -> caller-supplied value passes
        through unchanged (existing config-driven behavior preserved)."""
        kw = _build_openai_kwargs(**_base_args(
            api_model="test/model", temperature=0.7, top_p=0.5,
        ))
        assert kw["temperature"] == 0.7
        assert kw["top_p"] == 0.5

    def test_no_temperature_anywhere_omits_the_param(self):
        kw = _build_openai_kwargs(**_base_args(api_model="test/model"))
        assert "temperature" not in kw
        assert "top_p" not in kw

    def test_profile_temperature_flows_even_to_google(self, monkeypatch):
        """Unlike penalties, Google DOES support temperature/top_p — the
        provider support gate (_supports_penalties) is penalty-specific and
        must not block profile-driven temperature/top_p."""
        monkeypatch.setattr(
            "openalph.provider._sampling_profile",
            lambda m: SamplingProfile(temperature=0.3, top_p=0.8),
        )
        kw = _build_openai_kwargs(**_base_args(provider_key="google"))
        assert kw["temperature"] == 0.3
        assert kw["top_p"] == 0.8

    def test_anthropic_builder_never_sees_sampling_profiles(self):
        """_build_anthropic_kwargs has no knowledge of _sampling_profile at
        all — profile values (e.g. minimax-m3's 1.0/0.95) must never leak
        into an Anthropic request just because the model string matches."""
        kw = _build_anthropic_kwargs(
            api_model="some-anthropic-model-that-happens-to-contain-minimax-m3",
            system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None,
            max_tokens=1024,
            thinking_level="off",
        )
        assert "temperature" not in kw
        assert "top_p" not in kw

