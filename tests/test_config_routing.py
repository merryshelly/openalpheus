"""Tests for provider routing configuration."""

from openalph.config import load_config


class TestRoutingConfig:

    def test_routing_parsed_from_toml(self, tmp_path):
        """[providers.openrouter.routing] is parsed into ProviderConfig.routing."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "openrouter/test-model"

[providers.openrouter]
type = "openai"
api_key = "sk-test"
base_url = "https://openrouter.ai/api/v1"

[providers.openrouter.routing]
quantizations = ["fp8", "fp16"]

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.providers["openrouter"].routing == {"quantizations": ["fp8", "fp16"]}

    def test_routing_invalid_type_skips_provider(self, tmp_path):
        """kdsn.292 flip: routing must be a dict; a non-dict now SKIPS the
        provider with reason (any per-provider authoring error degrades,
        never crash-loops). (Was: ..._raises, ConfigError.)"""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "openrouter/test-model"

[providers.openrouter]
type = "openai"
api_key = "sk-test"
base_url = "https://openrouter.ai/api/v1"
routing = "bad"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "openrouter" in config.skipped_providers
        assert "routing must be a table/dict" in config.skipped_providers["openrouter"]

    def test_routing_absent_defaults_to_none(self, tmp_path):
        """No routing section means ProviderConfig.routing is None."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "openrouter/test-model"

[providers.openrouter]
type = "openai"
api_key = "sk-test"
base_url = "https://openrouter.ai/api/v1"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.providers["openrouter"].routing is None
