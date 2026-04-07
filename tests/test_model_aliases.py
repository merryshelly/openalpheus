"""Tests for model alias resolution (snak epic — agent/operator ergonomics).

Covers:
- AgentConfig.model_aliases field
- resolve_model() alias expansion
- load_config() parsing of [model_aliases] TOML section
- assemble_prompt() alias table injection
- /model list slash command (via MatrixBridge)
- /model <alias> switching
- Sub-agent alias resolution (via config.default_model)

Design:
    model_aliases is a dict[str, str] on AgentConfig.
    resolve_model() accepts optional aliases dict. If model_str has no "/"
    and matches an alias key, it expands to the fully-qualified string before
    normal resolution. If model_str has a "/" it's treated as fully-qualified
    (aliases are never checked for strings containing "/").
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.config import (
    AgentConfig,
    ProviderConfig,
    load_config,
    resolve_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_provider(key="default", type="anthropic", api_key="sk-test",
                  base_url=None, quirks=None):
    return ProviderConfig(
        key=key, type=type, api_key=api_key,
        base_url=base_url, quirks=quirks or [],
    )


PROVIDERS = {
    "anthropic": make_provider(key="anthropic"),
    "openrouter": make_provider(
        key="openrouter", type="openai", api_key="sk-or",
        base_url="https://openrouter.ai/api/v1",
    ),
    "fireworks": make_provider(
        key="fireworks", type="openai", api_key="sk-fw",
        base_url="https://api.fireworks.ai/inference/v1",
    ),
}

ALIASES = {
    "opus": "anthropic/claude-opus-4-6",
    "sonnet": "anthropic/claude-sonnet-4-6",
    "kimi": "fireworks/accounts/fireworks/models/kimi-k2p5",
    "glm5": "openrouter/z-ai/glm-5",
}


# ===========================================================================
# SECTION 1: resolve_model with aliases
# ===========================================================================

class TestResolveModelAliases:

    def test_alias_expands_to_correct_provider_and_model(self):
        """An alias without '/' resolves through the alias table."""
        prov, api_model = resolve_model("opus", PROVIDERS, aliases=ALIASES)
        assert prov.key == "anthropic"
        assert api_model == "claude-opus-4-6"

    def test_alias_with_nested_model_name(self):
        """Alias pointing to a model with slashes (fireworks path) works."""
        prov, api_model = resolve_model("kimi", PROVIDERS, aliases=ALIASES)
        assert prov.key == "fireworks"
        assert api_model == "accounts/fireworks/models/kimi-k2p5"

    def test_alias_to_openrouter_nested(self):
        """Alias pointing to openrouter model with org/name works."""
        prov, api_model = resolve_model("glm5", PROVIDERS, aliases=ALIASES)
        assert prov.key == "openrouter"
        assert api_model == "z-ai/glm-5"

    def test_fully_qualified_string_bypasses_aliases(self):
        """A string with '/' is treated as fully-qualified, never alias-checked."""
        prov, api_model = resolve_model(
            "anthropic/claude-opus-4-6", PROVIDERS, aliases=ALIASES,
        )
        assert prov.key == "anthropic"
        assert api_model == "claude-opus-4-6"

    def test_fully_qualified_string_that_matches_alias_name(self):
        """Even if an alias name appears as a provider prefix, '/' triggers FQ path."""
        # This shouldn't happen in practice, but ensures '/' always means FQ.
        providers_with_opus = {**PROVIDERS, "opus": make_provider(key="opus")}
        prov, api_model = resolve_model(
            "opus/some-model", providers_with_opus, aliases=ALIASES,
        )
        assert prov.key == "opus"
        assert api_model == "some-model"

    def test_unknown_alias_raises_with_helpful_message(self):
        """A string with no '/' that isn't an alias raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model alias 'banana'"):
            resolve_model("banana", PROVIDERS, aliases=ALIASES)

    def test_unknown_string_no_aliases_raises_original_error(self):
        """With no aliases dict, a bare string raises the original no-slash error."""
        with pytest.raises(ValueError, match="must be fully qualified"):
            resolve_model("opus", PROVIDERS)

    def test_empty_aliases_dict_treated_as_no_aliases(self):
        """Passing empty aliases dict falls through to normal resolution."""
        with pytest.raises(ValueError, match="must be fully qualified"):
            resolve_model("opus", PROVIDERS, aliases={})

    def test_alias_pointing_to_unknown_provider_raises(self):
        """If alias target references a provider not in config, raise error."""
        bad_aliases = {"broken": "nonexistent/some-model"}
        with pytest.raises(ValueError, match="Unknown provider"):
            resolve_model("broken", PROVIDERS, aliases=bad_aliases)

    def test_alias_case_sensitive(self):
        """Aliases are case-sensitive — 'Opus' != 'opus'."""
        with pytest.raises(ValueError, match="Unknown model alias 'Opus'"):
            resolve_model("Opus", PROVIDERS, aliases=ALIASES)

    def test_no_aliases_param_defaults_to_none(self):
        """resolve_model works without aliases parameter (backward compat)."""
        prov, api_model = resolve_model("anthropic/claude-opus-4-6", PROVIDERS)
        assert prov.key == "anthropic"
        assert api_model == "claude-opus-4-6"


# ===========================================================================
# SECTION 2: AgentConfig.model_aliases field
# ===========================================================================

class TestAgentConfigAliases:

    def test_default_empty_dict(self):
        """model_aliases defaults to empty dict if not provided."""
        config = AgentConfig(
            name="test", default_model="anthropic/x",
            max_tokens=8192, providers=PROVIDERS,
            workspace=Path("/tmp/test"),
        )
        assert config.model_aliases == {}

    def test_explicit_aliases(self):
        """model_aliases can be set explicitly."""
        config = AgentConfig(
            name="test", default_model="anthropic/x",
            max_tokens=8192, providers=PROVIDERS,
            workspace=Path("/tmp/test"),
            model_aliases=ALIASES,
        )
        assert config.model_aliases == ALIASES
        assert config.model_aliases["opus"] == "anthropic/claude-opus-4-6"


# ===========================================================================
# SECTION 3: TOML loading parses [model_aliases]
# ===========================================================================

class TestLoadConfigAliases:

    def test_aliases_parsed_from_toml(self, tmp_path):
        """[model_aliases] section parsed into config.model_aliases."""
        ws = tmp_path / "workspace"
        ws.mkdir()
        toml_content = f"""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "{ws}"

[model_aliases]
opus = "anthropic/claude-opus-4-6"
kimi = "fireworks/accounts/fireworks/models/kimi-k2p5"
"""
        config_file = tmp_path / "test.toml"
        config_file.write_text(toml_content)

        config = load_config(config_file)
        assert config.model_aliases == {
            "opus": "anthropic/claude-opus-4-6",
            "kimi": "fireworks/accounts/fireworks/models/kimi-k2p5",
        }

    def test_missing_aliases_section_defaults_empty(self, tmp_path):
        """No [model_aliases] section → empty dict."""
        ws = tmp_path / "workspace"
        ws.mkdir()
        toml_content = f"""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "{ws}"
"""
        config_file = tmp_path / "test.toml"
        config_file.write_text(toml_content)

        config = load_config(config_file)
        assert config.model_aliases == {}

    def test_non_string_alias_values_ignored(self, tmp_path):
        """Non-string values in [model_aliases] are silently skipped."""
        ws = tmp_path / "workspace"
        ws.mkdir()
        toml_content = f"""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "{ws}"

[model_aliases]
opus = "anthropic/claude-opus-4-6"
bad_number = 42
bad_bool = true
"""
        config_file = tmp_path / "test.toml"
        config_file.write_text(toml_content)

        config = load_config(config_file)
        assert config.model_aliases == {"opus": "anthropic/claude-opus-4-6"}


# ===========================================================================
# SECTION 4: Prompt injection of alias table
# ===========================================================================

class TestPromptAliasInjection:

    def test_aliases_appear_in_prompt(self, tmp_path):
        """assemble_prompt includes alias table when aliases provided."""
        from openalph.prompt import assemble_prompt

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        aliases = {"opus": "anthropic/claude-opus-4-6", "kimi": "Fireworks Kimi K2.5"}
        prompt = assemble_prompt(workspace, model_aliases=aliases)

        assert "## Model Aliases" in prompt
        assert "opus" in prompt
        assert "kimi" in prompt

    def test_no_aliases_no_section(self, tmp_path):
        """No aliases → no Model Aliases section in prompt."""
        from openalph.prompt import assemble_prompt

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        prompt = assemble_prompt(workspace)
        assert "## Model Aliases" not in prompt

    def test_empty_aliases_no_section(self, tmp_path):
        """Empty aliases dict → no Model Aliases section in prompt."""
        from openalph.prompt import assemble_prompt

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        prompt = assemble_prompt(workspace, model_aliases={})
        assert "## Model Aliases" not in prompt


# ===========================================================================
# SECTION 5: /model list slash command
# ===========================================================================

class TestModelListCommand:
    """Tests for /model list slash command in MatrixBridge."""

    def _make_bridge(self, aliases=None):
        """Create a minimal mock MatrixBridge with config."""
        bridge = MagicMock()
        bridge.agent = MagicMock()
        bridge.agent.config = AgentConfig(
            name="test", default_model="anthropic/claude-opus-4-6",
            max_tokens=8192,
            providers=PROVIDERS,
            workspace=Path("/tmp/test"),
            model_aliases=aliases or {},
        )
        bridge.agent.get_model = MagicMock(return_value="anthropic/claude-opus-4-6")
        bridge.send = AsyncMock()
        return bridge

    @pytest.mark.asyncio
    async def test_model_list_shows_aliases(self):
        """'/model list' sends a message containing alias names and targets."""
        from openalph.matrix import format_model_list

        output = format_model_list(ALIASES, "anthropic/claude-opus-4-6")
        assert "opus" in output
        assert "sonnet" in output
        assert "kimi" in output
        assert "glm5" in output
        # Should show current model
        assert "claude-opus-4-6" in output

    @pytest.mark.asyncio
    async def test_model_list_empty_aliases(self):
        """'/model list' with no aliases shows only current model."""
        from openalph.matrix import format_model_list

        output = format_model_list({}, "anthropic/claude-opus-4-6")
        assert "No model aliases configured" in output
        assert "claude-opus-4-6" in output
