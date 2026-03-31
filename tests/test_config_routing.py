"""Tests for provider routing configuration."""

import pytest
from pathlib import Path
from openalph.config import load_config, ConfigError


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

    def test_routing_invalid_type_raises(self, tmp_path):
        """routing must be a dict; non-dict raises ConfigError."""
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
        with pytest.raises(ConfigError, match="routing must be a table/dict"):
            load_config(tmp_path / "agent.toml")

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
