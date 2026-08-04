"""Tests for Workstream B: model-aware context window.

Three groups:
1. _resolve_model_limit layering: config override, curated default, fallback+warn.
2. model-switched room reports active model window via agent.status().
3. model_context_window and _model_output_cap table correctness.
"""

import logging
import pytest

from openalph.agent import Agent
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import model_context_window, _model_output_cap, _MODEL_CAPABILITIES


# ---------------------------------------------------------------------------
# Helpers (mirrors test_token_estimation.py make_config / make_agent)
# ---------------------------------------------------------------------------

def make_config(tmp_path, **kwargs):
    defaults = dict(
        name="test",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        model_max_tokens=200_000,
        providers={
            "anthropic": ProviderConfig(
                key="anthropic",
                type="anthropic",
                api_key="sk-test",
                base_url=None,
                quirks=None,
            )
        },
        workspace=tmp_path,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_agent(tmp_path, **kwargs):
    config = make_config(tmp_path, **kwargs)
    return Agent(config)


# ---------------------------------------------------------------------------
# Group 1: _resolve_model_limit layering
# ---------------------------------------------------------------------------

class TestResolveModelLimitLayering:

    def test_config_override_wins(self, tmp_path):
        """[model_limits] entry takes priority over curated default and fallback."""
        # Use a known-curated model but override its limit in config
        agent = make_agent(
            tmp_path,
            default_model="anthropic/claude-opus-4-8",
            model_limits={"anthropic/claude-opus-4-8": 500_000},
        )
        room = "!test:override"
        # opus-4-8 curated = 1_048_576 but config override = 500_000
        assert agent._resolve_model_limit(room) == 500_000

    def test_curated_default_for_known_model(self, tmp_path):
        """Known model string with no config override returns curated window."""
        agent = make_agent(
            tmp_path,
            default_model="anthropic/claude-opus-4-8",
            model_limits={},
        )
        room = "!test:curated"
        # opus-4-8 fragment matches -> 1_048_576
        assert agent._resolve_model_limit(room) == 1_048_576

    def test_room_alias_override_resolves_expanded_window(self, tmp_path):
        """Room model stored as a bare alias (/model deepseek) resolves the
        expanded model's window, not the model_max_tokens fallback.
        Regression test for 2026-08-03 alias-blindness: wonmun room ran
        deepseek-v4-flash at a 262K window instead of 1M."""
        agent = make_agent(
            tmp_path,
            default_model="anthropic/claude-opus-4-8",
            model_max_tokens=200_000,
            model_limits={"macstudio/deepseek-v4-flash": 1_048_576},
            model_aliases={"deepseek": "macstudio/deepseek-v4-flash"},
        )
        # Direct lookup with the bare alias (what a /model override stores)
        assert agent._resolve_model_limit_for("deepseek") == 1_048_576
        # End-to-end via a room override
        agent._room_models["!test:alias"] = "deepseek"
        assert agent._resolve_model_limit("!test:alias") == 1_048_576

    def test_room_alias_override_hits_curated_without_config_entry(self, tmp_path):
        """Alias expansion also feeds Layer 2 (curated substring fragments)."""
        agent = make_agent(
            tmp_path,
            default_model="anthropic/claude-opus-4-8",
            model_max_tokens=200_000,
            model_limits={},
            model_aliases={"opus": "anthropic/claude-opus-4-8"},
        )
        assert agent._resolve_model_limit_for("opus") == 1_048_576

    def test_fallback_to_model_max_tokens_and_single_warn(self, tmp_path, caplog):
        """Unknown model falls back to model_max_tokens and logs exactly one WARNING."""
        agent = make_agent(
            tmp_path,
            default_model="anthropic/claude-sonnet-4-20250514",  # not in table
            model_max_tokens=200_000,
            model_limits={},
        )
        room = "!test:fallback"
        with caplog.at_level(logging.WARNING, logger="openalph.agent"):
            result1 = agent._resolve_model_limit(room)
            result2 = agent._resolve_model_limit(room)  # second call — no extra warn

        assert result1 == 200_000
        assert result2 == 200_000
        # Exactly one warning for this unknown model
        warns = [r for r in caplog.records
                 if r.levelno == logging.WARNING and "No curated context window" in r.message]
        assert len(warns) == 1

    def test_warn_once_per_model(self, tmp_path, caplog):
        """_warned_models deduplicate: two unknown models → two warnings (one each)."""
        agent = make_agent(
            tmp_path,
            default_model="anthropic/unknown-alpha",
            model_limits={},
        )
        room_a = "!room:a"
        room_b = "!room:b"
        agent._room_models[room_b] = "anthropic/unknown-beta"

        with caplog.at_level(logging.WARNING, logger="openalph.agent"):
            agent._resolve_model_limit(room_a)
            agent._resolve_model_limit(room_a)  # second call for alpha — still just 1
            agent._resolve_model_limit(room_b)
            agent._resolve_model_limit(room_b)  # second call for beta — still just 1

        warns = [r for r in caplog.records
                 if r.levelno == logging.WARNING and "No curated context window" in r.message]
        assert len(warns) == 2  # one per distinct unknown model


# ---------------------------------------------------------------------------
# Group 2: model-switched room reports active model window via status()
# ---------------------------------------------------------------------------

class TestStatusReportsActiveModelWindow:

    def test_switched_room_shows_opus_window(self, tmp_path):
        """After switching a room to opus-4-8, status()['context_max'] == 1_048_576."""
        agent = make_agent(tmp_path, model_max_tokens=200_000)
        room = "!switched:room"
        # Direct set avoids switch_model context-fit guard
        agent._room_models[room] = "anthropic/claude-opus-4-8"
        status = agent.status(room)
        assert status["context_max"] == 1_048_576

    def test_unswitched_room_keeps_default(self, tmp_path):
        """An un-switched room still uses model_max_tokens from config."""
        agent = make_agent(tmp_path, model_max_tokens=200_000)
        room_default = "!default:room"
        room_switched = "!switched:room"
        agent._room_models[room_switched] = "anthropic/claude-opus-4-8"

        status_default = agent.status(room_default)
        status_switched = agent.status(room_switched)

        assert status_default["context_max"] == 200_000  # fallback
        assert status_switched["context_max"] == 1_048_576

    def test_context_pct_uses_resolved_window(self, tmp_path):
        """context_pct is computed against the resolved (per-model) window."""
        agent = make_agent(tmp_path, model_max_tokens=200_000)
        room = "!pct:room"
        agent._room_models[room] = "anthropic/claude-opus-4-8"
        # Add a small amount of history to get nonzero tokens
        agent.history(room).append({"role": "user", "content": "hello"})
        status = agent.status(room)
        # context_pct must be consistent with context_max=1_048_576
        expected_pct = round(status["context_tokens"] / 1_048_576 * 100)
        assert status["context_pct"] == expected_pct


# ---------------------------------------------------------------------------
# Group 3: model_context_window + _model_output_cap table correctness
# ---------------------------------------------------------------------------

class TestCapabilitiesTable:

    # --- model_context_window ---

    def test_haiku_4_5_window(self):
        assert model_context_window("anthropic/claude-haiku-4-5") == 200_000

    def test_sonnet_4_6_window(self):
        assert model_context_window("anthropic/claude-sonnet-4-6") == 200_000

    def test_opus_4_6_window(self):
        assert model_context_window("anthropic/claude-opus-4-6") == 1_048_576

    def test_opus_4_7_window(self):
        assert model_context_window("anthropic/claude-opus-4-7") == 1_048_576

    def test_opus_4_8_window(self):
        assert model_context_window("anthropic/claude-opus-4-8") == 1_048_576

    def test_fable_window(self):
        assert model_context_window("accounts/fireworks/models/fable-3") == 1_048_576

    def test_glm_5p2_window(self):
        assert model_context_window("accounts/fireworks/models/glm-5p2") == 1_048_576

    def test_kimi_k2p6_window(self):
        assert model_context_window("accounts/fireworks/models/kimi-k2p6") == 262_144

    def test_qwen3_5_window(self):
        assert model_context_window("qwen3.5:latest") == 262_144

    def test_qwen3p5_window(self):
        assert model_context_window("qwen3p5-14b") == 262_144

    def test_maverick_window(self):
        assert model_context_window("accounts/fireworks/models/llama4-maverick") == 1_048_576

    def test_hermes_window(self):
        assert model_context_window("accounts/fireworks/models/hermes-3-llama") == 131_072

    def test_gemini_window(self):
        assert model_context_window("google/gemini-2.0-flash") == 1_048_576

    def test_unknown_model_returns_none(self):
        """Default test model string is NOT in the table — must return None."""
        assert model_context_window("anthropic/claude-sonnet-4-20250514") is None

    def test_case_insensitive(self):
        assert model_context_window("ANTHROPIC/CLAUDE-HAIKU-4-5") == 200_000

    # --- _model_output_cap (preserved values) ---

    def test_haiku_4_5_output_cap(self):
        assert _model_output_cap("anthropic/claude-haiku-4-5") == 64_000

    def test_sonnet_4_6_output_cap(self):
        assert _model_output_cap("anthropic/claude-sonnet-4-6") == 128_000

    def test_opus_4_6_output_cap(self):
        assert _model_output_cap("anthropic/claude-opus-4-6") == 128_000

    def test_opus_4_7_output_cap(self):
        assert _model_output_cap("anthropic/claude-opus-4-7") == 128_000

    def test_opus_4_8_output_cap(self):
        assert _model_output_cap("anthropic/claude-opus-4-8") == 128_000

    def test_fable_output_cap(self):
        assert _model_output_cap("accounts/fireworks/models/fable-3") == 128_000

    def test_fireworks_model_output_cap_none(self):
        assert _model_output_cap("accounts/fireworks/models/glm-5p2") is None

    def test_kimi_output_cap_none(self):
        assert _model_output_cap("accounts/fireworks/models/kimi-k2p6") is None

    def test_unknown_model_output_cap_none(self):
        """Unknown model (including default test model) → None."""
        assert _model_output_cap("anthropic/claude-sonnet-4-20250514") is None

    def test_capabilities_table_structure(self):
        """Every entry in _MODEL_CAPABILITIES is a 3-tuple (str, int|None, int|None)."""
        for entry in _MODEL_CAPABILITIES:
            assert len(entry) == 3
            frag, window, cap = entry
            assert isinstance(frag, str)
            assert window is None or isinstance(window, int)
            assert cap is None or isinstance(cap, int)


# ---------------------------------------------------------------------------
# FIX 1 test: switch_model must use curated 3-layer window resolution
# ---------------------------------------------------------------------------

class TestSwitchModelUsesCuratedWindow:
    """FIX 1 guard: switch_model must reject a switch when context exceeds
    the TARGET model's curated window, even if it fits within model_max_tokens."""

    def test_switch_model_uses_curated_window(self, tmp_path):
        """agent with model_max_tokens=1_048_576, no [model_limits].
        Context > 200_000 tokens → switch to sonnet-4-6 (curated 200_000) must FAIL.
        A clean room must succeed (proving the guard itself works)."""
        # Agent with a huge default model_max_tokens — no [model_limits] overrides.
        agent = make_agent(tmp_path, model_max_tokens=1_048_576, model_limits={})
        room = "!big_context:room"
        other_room = "!clean_room:room"

        # Put ~250_000 tokens of context in the room (250_000 * 4 = 1_000_000 chars).
        # This exceeds sonnet-4-6's curated window (200_000) but fits within
        # model_max_tokens=1_048_576, so the OLD logic (skipping the curated table)
        # would wrongly allow the switch.
        big_content = "x" * (250_000 * 4)
        agent.history(room).append({"role": "user", "content": big_content})

        # Switch must be REJECTED because sonnet-4-6 curated window = 200_000
        err = agent.switch_model("anthropic/claude-sonnet-4-6", room)
        assert err is not None, (
            "FIX 1 REGRESSION: switch_model should have rejected the switch "
            "(context ~250k tokens > sonnet-4-6 curated window 200_000), "
            "but returned None (success). The old logic skipped the curated table "
            "and used model_max_tokens=1_048_576 instead."
        )
        assert "200,000" in err or "200000" in err, (
            f"Error message should cite the 200_000 window; got: {err!r}"
        )

        # A clean room with minimal context must succeed
        err2 = agent.switch_model("anthropic/claude-sonnet-4-6", other_room)
        assert err2 is None, (
            f"Clean room should succeed but got error: {err2!r}"
        )
