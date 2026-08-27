"""Tests for graceful provider degradation (kdsn.155).

When a single provider's api_key resolution fails, the agent should:
- Log a warning and skip that provider
- Continue loading remaining providers
- Only raise ConfigError if zero providers loaded, or if the default_model's provider failed

Acceptance criteria:
- [ ] Single bad provider doesn't crash the process
- [ ] Warning log clearly identifies which provider failed and why
- [ ] default_model's provider failure remains fatal
- [ ] Aliases pointing to skipped providers produce clear error on use, not silent failure
- [ ] Tests cover: one bad provider among good ones, default_model provider bad, all providers bad
"""

import logging
import pytest
from pathlib import Path
from openalph.config import load_config, ConfigError, resolve_model


def _write_config(tmp_path, toml_content):
    """Helper: write TOML config and ensure workspace dir exists."""
    config_path = tmp_path / "agent.toml"
    config_path.write_text(toml_content)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return config_path


class TestGracefulProviderDegradation:
    """Provider failures are isolated — one bad provider doesn't kill the process."""

    def test_one_bad_provider_skipped_good_ones_load(self, tmp_path):
        """A single provider with a broken api_key_cmd is skipped;
        the other provider loads normally."""
        config_path = _write_config(tmp_path, f"""
[agent]
name = "test"
default_model = "anthropic/claude-test"

[providers.anthropic]
type = "anthropic"
api_key = "sk-good-key"

[providers.broken]
type = "openai"
base_url = "http://localhost:9999"
api_key_cmd = "exit 1"

[workspace]
path = "{tmp_path / 'workspace'}"
""")
        config = load_config(config_path)
        assert "anthropic" in config.providers
        assert "broken" not in config.providers
        assert len(config.providers) == 1

    def test_multiple_bad_providers_all_skipped(self, tmp_path):
        """Multiple bad providers are all skipped; the one good provider loads."""
        config_path = _write_config(tmp_path, f"""
[agent]
name = "test"
default_model = "anthropic/claude-test"

[providers.anthropic]
type = "anthropic"
api_key = "sk-good-key"

[providers.broken1]
type = "openai"
base_url = "http://localhost:9999"
api_key_cmd = "exit 1"

[providers.broken2]
type = "openai"
base_url = "http://localhost:9998"
api_key_cmd = "echo ''"

[workspace]
path = "{tmp_path / 'workspace'}"
""")
        config = load_config(config_path)
        assert "anthropic" in config.providers
        assert "broken1" not in config.providers
        assert "broken2" not in config.providers
        assert len(config.providers) == 1

    def test_bad_provider_missing_api_key_entirely(self, tmp_path):
        """Provider with no api_key/api_key_env/api_key_cmd is skipped."""
        config_path = _write_config(tmp_path, f"""
[agent]
name = "test"
default_model = "anthropic/claude-test"

[providers.anthropic]
type = "anthropic"
api_key = "sk-good-key"

[providers.nokey]
type = "openai"
base_url = "http://localhost:9999"

[workspace]
path = "{tmp_path / 'workspace'}"
""")
        config = load_config(config_path)
        assert "anthropic" in config.providers
        assert "nokey" not in config.providers

    def test_default_model_provider_failure_is_degraded(self, tmp_path):
        """kdsn.292 flip: default_model's provider failing to load is now a
        DEGRADED start (skip recorded with reason), NOT fatal — see SB ruling
        2026-08-26. (Was: test_default_model_provider_failure_is_fatal.)"""
        config_path = _write_config(tmp_path, f"""
[agent]
name = "test"
default_model = "broken/some-model"

[providers.broken]
type = "openai"
base_url = "http://localhost:9999"
api_key_cmd = "exit 1"

[providers.anthropic]
type = "anthropic"
api_key = "sk-good-key"

[workspace]
path = "{tmp_path / 'workspace'}"
""")
        config = load_config(config_path)  # must NOT raise (kdsn.292)
        assert config.default_model == "broken/some-model"
        assert "broken" not in config.providers
        assert "broken" in config.skipped_providers
        assert "exit code" in config.skipped_providers["broken"]
        assert "anthropic" in config.providers  # healthy half still usable

    def test_default_model_alias_to_skipped_provider_is_degraded(self, tmp_path):
        """kdsn.292 flip: default_model aliasing a skipped provider is now a
        DEGRADED start, NOT fatal. (Was: ..._is_fatal.)"""
        config_path = _write_config(tmp_path, f"""
[agent]
name = "test"
default_model = "myalias"

[providers.broken]
type = "openai"
base_url = "http://localhost:9999"
api_key_cmd = "exit 1"

[providers.anthropic]
type = "anthropic"
api_key = "sk-good-key"

[model_aliases]
myalias = "broken/some-model"

[workspace]
path = "{tmp_path / 'workspace'}"
""")
        config = load_config(config_path)  # must NOT raise (kdsn.292)
        assert "broken" not in config.providers
        assert "broken" in config.skipped_providers
        assert "exit code" in config.skipped_providers["broken"]

    def test_all_providers_fail_is_degraded(self, tmp_path):
        """kdsn.292 flip: ALL providers failing is a DEGRADED start with
        empty providers dict, NOT ConfigError. (Was: ..._is_fatal.)"""
        config_path = _write_config(tmp_path, f"""
[agent]
name = "test"
default_model = "broken/some-model"

[providers.broken]
type = "openai"
base_url = "http://localhost:9999"
api_key_cmd = "exit 1"

[providers.also_broken]
type = "anthropic"
api_key_cmd = "echo ''"

[workspace]
path = "{tmp_path / 'workspace'}"
""")
        config = load_config(config_path)  # must NOT raise (kdsn.292)
        assert config.providers == {}
        assert set(config.skipped_providers) == {"broken", "also_broken"}
        assert all(config.skipped_providers.values())  # every skip has a reason

    def test_warning_logged_for_skipped_provider(self, tmp_path, caplog):
        """Skipped provider emits a WARNING log identifying which provider and why."""
        config_path = _write_config(tmp_path, f"""
[agent]
name = "test"
default_model = "anthropic/claude-test"

[providers.anthropic]
type = "anthropic"
api_key = "sk-good-key"

[providers.broken]
type = "openai"
base_url = "http://localhost:9999"
api_key_cmd = "exit 1"

[workspace]
path = "{tmp_path / 'workspace'}"
""")
        with caplog.at_level(logging.WARNING, logger="openalph.config"):
            config = load_config(config_path)

        # Must mention which provider failed
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("broken" in msg for msg in warning_msgs), \
            f"Expected warning mentioning 'broken' provider, got: {warning_msgs}"

    def test_info_logged_for_active_providers(self, tmp_path, caplog):
        """On successful load with skipped providers, log which providers are active."""
        config_path = _write_config(tmp_path, f"""
[agent]
name = "test"
default_model = "anthropic/claude-test"

[providers.anthropic]
type = "anthropic"
api_key = "sk-good-key"

[providers.broken]
type = "openai"
base_url = "http://localhost:9999"
api_key_cmd = "exit 1"

[workspace]
path = "{tmp_path / 'workspace'}"
""")
        with caplog.at_level(logging.INFO, logger="openalph.config"):
            config = load_config(config_path)

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        # Should log which providers loaded successfully
        assert any("anthropic" in msg for msg in info_msgs), \
            f"Expected info log mentioning active providers, got: {info_msgs}"

    def test_skipped_provider_not_in_providers_dict(self, tmp_path):
        """Skipped providers must not appear in the providers dict at all."""
        config_path = _write_config(tmp_path, f"""
[agent]
name = "test"
default_model = "anthropic/claude-test"

[providers.anthropic]
type = "anthropic"
api_key = "sk-good-key"

[providers.broken]
type = "openai"
base_url = "http://localhost:9999"
api_key_cmd = "exit 1"

[workspace]
path = "{tmp_path / 'workspace'}"
""")
        config = load_config(config_path)
        # Verify resolve_model fails cleanly for the skipped provider
        with pytest.raises(ValueError, match="Unknown provider.*broken"):
            resolve_model("broken/some-model", config.providers)

    def test_non_api_key_errors_skip_with_reason(self, tmp_path):
        """kdsn.292 flip: errors that aren't api_key failures (e.g. invalid
        provider type) SKIP THE PROVIDER WITH REASON — the SB ruling widened
        tolerance to ANY per-provider authoring problem (a mass-applied TOML
        slip must never crash-loop an agent). (Was: ..._still_fatal.)"""
        config_path = _write_config(tmp_path, f"""
[agent]
name = "test"
default_model = "anthropic/claude-test"

[providers.anthropic]
type = "anthropic"
api_key = "sk-good-key"

[providers.weird]
type = "invalid_type"
api_key = "sk-some-key"

[workspace]
path = "{tmp_path / 'workspace'}"
""")
        # kdsn.292: invalid provider type skips the provider (reason names
        # the type problem); the healthy provider still loads.
        config = load_config(config_path)
        assert "weird" not in config.providers
        assert "weird" in config.skipped_providers
        assert "type" in config.skipped_providers["weird"].lower()
        assert "anthropic" in config.providers
