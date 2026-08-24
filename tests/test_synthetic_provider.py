"""Tests for Synthetic inference provider support (workspace-kdsn.281).

Spec: memory/projects/openalph/specs/synthetic-provider-spec.md
Probe evidence (2026-08-23, via op-run): tmp/synthetic-probe{4,5,6}.py —
  * reasoning_effort wire values: none/low/medium/high/xhigh -> 200,
    max -> 400 (backend), off -> 400 (gateway). Omitted param = reasoning ON.
  * Streaming: reasoning surfaces as `delta.reasoning`; stream_options
    include_usage honored; transparent prefix caching reported on the
    streaming path (cached=3840/3863 on 2nd identical call).
  * Anthropic surface (base_url=https://api.synthetic.new/anthropic): SDK
    streaming + thinking=budget + cache_control ttl=1h all accepted.
  * Catalog windows (live /models): Kimi-K3 524288, GLM-5.2 524288,
    GLM-4.7-Flash 196608, Qwen3.6/3.8-27B 262144, gpt-oss-120b 131072,
    Nemotron-3-Super-120B 262144. NO Kimi-K2.6 (404, rotated).
  * Vision: hf:moonshotai/Kimi-K3 identified a 64x64 red PNG via the
    openai image_url shape.

Code surface under test:
  C1  _get_client anthropic branch honors provider.base_url (generic patch —
      unblocks any Anthropic-compatible endpoint; adoption deferred, spec D3).
  C2  _build_openai_kwargs synthetic branch: off->none, low/medium/high 1:1,
      xhigh/max -> high + warn-once, ALWAYS explicit (kdsn.271 property).
      The warn-once set is the module-level ``_SYNTHETIC_EFFORT_WARNED``
      (house convention: ``_VISION_WARNED``, ``_warned_unpriced_anthropic``).
  C3  _MODEL_CAPABILITIES + _SAMPLING_PROFILES fragments for hf:-namespaced
      Synthetic IDs (ordered before generic rows — first-match-wins).
  C4  resolve_model locks: "synthetic/hf:org/Model" (colon + extra slash).
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from openalph import provider as provider_module
from openalph.config import ProviderConfig, resolve_model, load_config
from openalph.provider import (
    _get_client,
    _build_openai_kwargs,
    _sampling_profile,
    model_context_window,
    model_supports_vision,
)

SYNTH_OPENAI_BASE = "https://api.synthetic.new/openai/v1"
SYNTH_ANTH_BASE = "https://api.synthetic.new/anthropic"


def _provider(key="synthetic", type="anthropic", api_key="sk-syn", base_url=None):
    return ProviderConfig(
        key=key, type=type, api_key=api_key, base_url=base_url, quirks=[],
    )


# ---------------------------------------------------------------------------
# C1 — anthropic client base_url passthrough (generic; kdsn.281 blocker)
# ---------------------------------------------------------------------------

class TestAnthropicClientBaseUrl:
    """_get_client must thread provider.base_url into AsyncAnthropic and key
    the client cache on it (the openai branch at provider.py:143 is the
    template). Without this, an Anthropic-compatible gateway is unreachable:
    the SDK always dials api.anthropic.com."""

    def test_base_url_passed_to_anthropic_ctor(self):
        with patch.object(provider_module.anthropic, "AsyncAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            _get_client(_provider(base_url=SYNTH_ANTH_BASE))
        _, kwargs = mock_cls.call_args
        assert kwargs.get("base_url") == SYNTH_ANTH_BASE

    def test_no_base_url_anthropic_ctor_unchanged(self):
        """Legacy/default path: no base_url configured -> SDK default."""
        with patch.object(provider_module.anthropic, "AsyncAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            _get_client(_provider(key="anthropic", base_url=None))
        _, kwargs = mock_cls.call_args
        assert kwargs.get("base_url") is None

    def test_distinct_base_urls_get_distinct_clients(self):
        c1 = _get_client(_provider(base_url=SYNTH_ANTH_BASE))
        c2 = _get_client(_provider(base_url="https://other.example.com/anthropic"))
        assert id(c1) != id(c2)

    def test_same_base_url_reuses_client(self):
        c1 = _get_client(_provider(base_url=SYNTH_ANTH_BASE))
        c2 = _get_client(_provider(base_url=SYNTH_ANTH_BASE))
        assert id(c1) == id(c2)

    def test_base_url_client_does_not_collide_with_default(self):
        """A base_url'd gateway client must never be served the real-
        Anthropic cached client (and vice versa)."""
        c1 = _get_client(_provider(key="anthropic", base_url=None))
        c2 = _get_client(_provider(key="synthetic", base_url=SYNTH_ANTH_BASE))
        assert id(c1) != id(c2)


# ---------------------------------------------------------------------------
# C2 — synthetic reasoning_effort mapping in _build_openai_kwargs
# ---------------------------------------------------------------------------

class TestSyntheticEffortMapping:
    """Probe-validated mapping (spec D4). Always explicit: omitting the param
    leaves the server default (reasoning ON — measured 784 reasoning chars on
    a no-param call), which is the kdsn.271 silent-override bug class."""

    @pytest.fixture(autouse=True)
    def _clear_synthetic_warned(self):
        """Isolate the warn-once set between tests (conftest clears the
        _client_cache; this is the same isolation for the remap warn-set)."""
        provider_module._SYNTHETIC_EFFORT_WARNED.clear()
        yield
        provider_module._SYNTHETIC_EFFORT_WARNED.clear()

    def _args(self, level, model="hf:zai-org/GLM-5.2"):
        return dict(
            api_model=model,
            system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None,
            max_tokens=1024,
            thinking_level=level,
            quirks=[],
            provider_key="synthetic",
        )

    def _effort(self, kw):
        return kw.get("extra_body", {}).get("reasoning_effort")

    def test_off_maps_to_none(self):
        """`off` must send reasoning_effort='none' — probe: 'none' -> 200 with
        reasoning_len=0; literal 'off' -> gateway 400; omitting -> thinking ON."""
        kw = _build_openai_kwargs(**self._args("off"))
        assert self._effort(kw) == "none"

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_low_medium_high_passthrough(self, level):
        kw = _build_openai_kwargs(**self._args(level))
        assert self._effort(kw) == level

    def test_xhigh_maps_to_high_with_warning(self, caplog):
        """Probe: xhigh -> 200, but advisor ruling: likely gateway coercion
        with no evidence it differs from high — collapse, warn, don't trust."""
        with caplog.at_level(logging.WARNING):
            kw = _build_openai_kwargs(**self._args("xhigh"))
        assert self._effort(kw) == "high"
        assert any(
            "xhigh" in r.message and "reasoning_effort" in r.message
            for r in caplog.records
        ), f"expected remap warning, got: {[r.message for r in caplog.records]}"

    def test_max_maps_to_high_with_warning(self, caplog):
        """Probe: max -> 400 from the inference backend (literal_error:
        only low/medium/high). Remap + warn rather than hard-fail the turn."""
        with caplog.at_level(logging.WARNING):
            kw = _build_openai_kwargs(**self._args("max"))
        assert self._effort(kw) == "high"
        assert any(
            "max" in r.message and "reasoning_effort" in r.message
            for r in caplog.records
        )

    def test_remap_warns_once_per_level(self, caplog):
        """Warn-once, not warn-never / warn-spam (fleet log hygiene)."""
        with caplog.at_level(logging.WARNING):
            _build_openai_kwargs(**self._args("max"))
            _build_openai_kwargs(**self._args("max"))
            _build_openai_kwargs(**self._args("max"))
        warns = [r for r in caplog.records
                 if "reasoning_effort" in r.message and r.levelno >= logging.WARNING]
        assert len(warns) == 1, f"expected exactly 1 warning, got {len(warns)}"

    def test_no_warning_for_exact_levels(self, caplog):
        with caplog.at_level(logging.WARNING):
            for level in ("off", "low", "medium", "high"):
                _build_openai_kwargs(**self._args(level))
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warns == [], f"unexpected warnings: {[r.message for r in warns]}"

    def test_no_nested_reasoning_object(self):
        """Synthetic takes TOP-LEVEL reasoning_effort (like Fireworks), never
        OpenRouter's nested reasoning.effort."""
        kw = _build_openai_kwargs(**self._args("high"))
        assert "reasoning" not in kw.get("extra_body", {})

    def test_other_providers_unaffected(self):
        """Guard: the synthetic branch must not capture other keys."""
        kw = _build_openai_kwargs(
            api_model="test/model", system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None, max_tokens=1024,
            thinking_level="high", quirks=[], provider_key="openrouter",
        )
        assert kw["extra_body"]["reasoning"] == {"effort": "high"}
        assert "reasoning_effort" not in kw["extra_body"]


# ---------------------------------------------------------------------------
# C3 — capabilities + sampling fragments for hf:-namespaced Synthetic IDs
# ---------------------------------------------------------------------------

class TestSyntheticCapabilities:
    """Windows from live GET /openai/v1/models (2026-08-23). Synthetic serves
    SMALLER windows than Fireworks for the same weights (Kimi-K3: 512K vs
    1M) — fragment rows must be ordered so hf:-namespaced IDs match first."""

    @pytest.mark.parametrize("model,window", [
        ("hf:moonshotai/Kimi-K3", 524288),
        ("hf:zai-org/GLM-5.2", 524288),
        ("hf:zai-org/GLM-4.7-Flash", 196608),
        ("hf:Qwen/Qwen3.8-27B", 262144),
        ("hf:Qwen/Qwen3.6-27B", 262144),
        ("hf:openai/gpt-oss-120b", 131072),
        ("hf:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", 262144),
    ])
    def test_synthetic_windows(self, model, window):
        assert model_context_window(model) == window

    def test_fireworks_kimi_k3_window_unaffected(self):
        """Ordering regression guard: the generic kimi-k3 row (1M, Fireworks)
        must still win for Fireworks IDs after synthetic rows are added."""
        assert model_context_window("accounts/fireworks/models/kimi-k3") == 1_048_576

    def test_synthetic_kimi_k3_beats_generic_fragment(self):
        """First-match-wins: 'hf:moonshotai/kimi-k3' contains 'kimi-k3', so
        the synthetic row must sit earlier in the table (512K, not 1M)."""
        assert model_context_window("hf:moonshotai/Kimi-K3") == 524288

    def test_vision_flags(self):
        """Kimi-K3 vision live-probed on Synthetic (probe5, red PNG via
        image_url). Everything else ships fail-closed False until probed.
        Qwen3.8-27B inherits the generic qwen3.8 row (True) — it is the
        vendor's syn:small:vision target; live image probe is a wire-up
        smoke item (spec section 6)."""
        config = SimpleNamespace(model_aliases={}, model_vision={})
        assert model_supports_vision("hf:moonshotai/Kimi-K3", config) is True
        for blind in ("hf:zai-org/GLM-5.2", "hf:zai-org/GLM-4.7-Flash",
                      "hf:openai/gpt-oss-120b",
                      "hf:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"):
            assert model_supports_vision(blind, config) is False, blind

    def test_sampling_pins_explicit(self):
        """kdsn.241.3 invariant: GLM/Kimi must NEVER receive penalties
        (Moonshot hard-errors on nonzero). 'hf:zai-org/glm-5.2' does NOT
        substring-match the Fireworks 'glm-5p2' fragment, so synthetic IDs
        need their own explicit pins — the default profile matching by
        accident is not a pin."""
        for model in ("hf:zai-org/GLM-5.2", "hf:moonshotai/Kimi-K3"):
            p = _sampling_profile(model)
            assert p.frequency_penalty is None, model
            assert p.presence_penalty is None, model
            assert p.temperature is None, model
            assert p.top_p is None, model


# ---------------------------------------------------------------------------
# C4 — resolution locks (no code change expected; pin the behavior)
# ---------------------------------------------------------------------------

class TestSyntheticResolution:
    def _providers(self):
        return {
            "synthetic": ProviderConfig(
                key="synthetic", type="openai", api_key="sk-syn",
                base_url=SYNTH_OPENAI_BASE, quirks=["reasoning_replay"],
            ),
        }

    def test_hf_id_with_colon_and_slash_resolves(self):
        """resolve_model splits on the FIRST '/' — 'synthetic/hf:org/Model'
        must yield api_model 'hf:org/Model' intact (colon preserved)."""
        cfg, model = resolve_model(
            "synthetic/hf:moonshotai/Kimi-K3", self._providers())
        assert cfg.key == "synthetic"
        assert model == "hf:moonshotai/Kimi-K3"

    def test_alias_expansion(self):
        cfg, model = resolve_model(
            "synkimi3", self._providers(),
            aliases={"synkimi3": "synthetic/hf:moonshotai/Kimi-K3"})
        assert cfg.key == "synthetic"
        assert model == "hf:moonshotai/Kimi-K3"

    def test_alias_then_vision(self):
        """Alias-expansion-first (wonmun precedent): a persisted bare-alias
        room model must still resolve vision via the table."""
        config = SimpleNamespace(
            model_aliases={"synkimi3": "synthetic/hf:moonshotai/Kimi-K3"},
            model_vision={},
        )
        assert model_supports_vision("synkimi3", config) is True


# ---------------------------------------------------------------------------
# Config acceptance lock — deploy shape (spec D1) must load cleanly
# ---------------------------------------------------------------------------

class TestSyntheticConfigAcceptance:
    def _write_toml(self, tmp_path, provider_block):
        path = tmp_path / "test.toml"
        path.write_text(f"""
[agent]
name = "test"
default_model = "synthetic/hf:zai-org/GLM-4.7-Flash"
max_tokens = 8192

{provider_block}

[workspace]
path = "{tmp_path}"
""")
        return path

    def test_openai_surface_config_loads(self, tmp_path):
        cfg = load_config(self._write_toml(tmp_path, f"""
[providers.synthetic]
type = "openai"
base_url = "{SYNTH_OPENAI_BASE}"
api_key = "sk-syn"
quirks = ["reasoning_replay"]
"""))
        p = cfg.providers["synthetic"]
        assert p.type == "openai"
        assert p.base_url == SYNTH_OPENAI_BASE
        assert "reasoning_replay" in p.quirks

    def test_anthropic_type_with_base_url_loads_and_keeps_it(self, tmp_path):
        """The deferred anthropic surface (spec D3): config layer must accept
        and RETAIN base_url for type=anthropic (it is only REQUIRED for
        openai). Locks the deploy-time contract C1 relies on."""
        cfg = load_config(self._write_toml(tmp_path, f"""
[providers.synthetic]
type = "anthropic"
base_url = "{SYNTH_ANTH_BASE}"
api_key = "sk-syn"
"""))
        p = cfg.providers["synthetic"]
        assert p.type == "anthropic"
        assert p.base_url == SYNTH_ANTH_BASE
