"""Tests for agent configuration loading.

Interface contract:
    load_config(path: Path) -> AgentConfig
    AgentConfig: name, default_model, max_tokens, providers (dict[str, ProviderConfig]), workspace

Config is TOML. API key resolves from: api_key (direct), api_key_env (env var),
api_key_cmd (shell command). Precedence: api_key > api_key_env > api_key_cmd.
"""

import pytest
import subprocess
from pathlib import Path
from openalph.config import AgentConfig, ProviderConfig, load_config, load_agent_config, ConfigError, CONFIG_DIR


# --- Valid configs ---


class TestLoadValidConfig:

    def test_minimal_anthropic(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "merry"
default_model = "anthropic/claude-sonnet-4-20250514"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test-key"

[workspace]
path = "/tmp/test-workspace"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.name == "merry"
        assert config.default_model == "anthropic/claude-sonnet-4-20250514"
        assert "anthropic" in config.providers
        assert list(config.providers.values())[0].type == "anthropic"
        assert list(config.providers.values())[0].api_key == "sk-test-key"
        assert list(config.providers.values())[0].base_url is None
        assert config.workspace == Path("/tmp/test-workspace")

    def test_max_tokens_default(self, tmp_path):
        """max_tokens defaults to 8192 if not specified."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.max_tokens == 8192

    def test_max_tokens_custom(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"
max_tokens = 4096

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.max_tokens == 4096

    def test_openai_compatible(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "babson"
default_model = "openrouter/moonshotai/kimi-k2.5"

[providers.openrouter]
type = "openai"
api_key = "sk-or-test"
base_url = "https://openrouter.ai/api/v1"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert list(config.providers.values())[0].type == "openai"
        assert list(config.providers.values())[0].base_url == "https://openrouter.ai/api/v1"

    def test_ollama_via_openai(self, tmp_path):
        """Ollama uses the OpenAI-compatible path."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "local"
default_model = "ollama/qwen3:235b-a22b"

[providers.ollama]
type = "openai"
api_key = "ollama"
base_url = "http://localhost:11434/v1"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert list(config.providers.values())[0].type == "openai"
        assert config.default_model == "ollama/qwen3:235b-a22b"


# --- API key resolution ---


class TestApiKeyResolution:

    def test_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-from-env")
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key_env = "TEST_API_KEY"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert list(config.providers.values())[0].api_key == "sk-from-env"

    def test_from_command(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key_cmd = "echo sk-from-cmd"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert list(config.providers.values())[0].api_key == "sk-from-cmd"

    def test_precedence_direct_over_env(self, tmp_path, monkeypatch):
        """Direct api_key wins over api_key_env."""
        monkeypatch.setenv("TEST_KEY", "from-env")
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test"

[providers.anthropic]
type = "anthropic"
api_key = "from-direct"
api_key_env = "TEST_KEY"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert list(config.providers.values())[0].api_key == "from-direct"

    def test_precedence_env_over_cmd(self, tmp_path, monkeypatch):
        """api_key_env wins over api_key_cmd."""
        monkeypatch.setenv("TEST_KEY", "from-env")
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test"

[providers.anthropic]
type = "anthropic"
api_key_env = "TEST_KEY"
api_key_cmd = "echo from-cmd"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert list(config.providers.values())[0].api_key == "from-env"


# --- Validation errors ---


class TestConfigValidation:

    def test_missing_name(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
default_model = "anthropic/test"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        with pytest.raises(ConfigError, match="name"):
            load_config(tmp_path / "agent.toml")

    def test_missing_model(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        with pytest.raises(ConfigError, match="default_model"):
            load_config(tmp_path / "agent.toml")

    def test_missing_api_key_entirely(self, tmp_path):
        """No api_key, api_key_env, or api_key_cmd."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test"

[providers.anthropic]
type = "anthropic"

[workspace]
path = "/tmp/test"
""")
        # ARCH-2 consolidated api_key/password/access_token resolution into one
        # `_resolve_secret` helper — the "no source configured" message now
        # names the actual field consistently ("api_key", not "API key").
        # kdsn.292 flip: this is now a SKIP-WITH-REASON (degraded start), not
        # a fatal ConfigError — one stray provider block must never crash-loop.
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "anthropic" in config.skipped_providers
        assert "api_key" in config.skipped_providers["anthropic"]

    def test_invalid_provider_type(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test"

[providers.google]
type = "google"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        # kdsn.292 flip: invalid type provider is skipped with reason.
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "google" in config.skipped_providers
        assert "type" in config.skipped_providers["google"].lower()

    def test_openai_missing_base_url(self, tmp_path):
        """OpenAI-compatible provider requires base_url."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test"

[providers.openrouter]
type = "openai"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        # kdsn.292 flip: missing base_url provider is skipped with reason.
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "openrouter" in config.skipped_providers
        assert "base_url" in config.skipped_providers["openrouter"]

    def test_env_var_not_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test"

[providers.anthropic]
type = "anthropic"
api_key_env = "NONEXISTENT_VAR"

[workspace]
path = "/tmp/test"
""")
        # kdsn.292 flip: unset env var skips the provider; reason names the var.
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "anthropic" in config.skipped_providers
        assert "NONEXISTENT_VAR" in config.skipped_providers["anthropic"]

    def test_command_fails(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test"

[providers.anthropic]
type = "anthropic"
api_key_cmd = "false"

[workspace]
path = "/tmp/test"
""")
        # kdsn.292 flip: failing api_key_cmd skips the provider (single
        # attempt, no retry — the acceptance suite pins the attempt count).
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "anthropic" in config.skipped_providers
        assert "api_key_cmd" in config.skipped_providers["anthropic"]

    def test_nonexistent_file(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(tmp_path / "nonexistent.toml")

    def test_missing_workspace(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"
""")
        with pytest.raises(ConfigError):
            load_config(tmp_path / "agent.toml")

    def test_nonexistent_workspace_path(self, tmp_path):
        """workspace.path pointing to nonexistent dir raises ConfigError."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/nonexistent/path/here"
""")
        with pytest.raises(ConfigError, match="does not exist"):
            load_config(tmp_path / "agent.toml")

    def test_default_model_unknown_provider(self, tmp_path):
        """kdsn.292 flip: default_model referencing a provider that was never
        configured is a DEGRADED start — skip reason "not configured" —
        NOT ConfigError. (Was: raised ConfigError.)"""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "banana/some-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert "banana" in config.skipped_providers
        assert "not configured" in config.skipped_providers["banana"]
        assert "anthropic" in config.providers


# --- Dataclass ---


# --- Agent config discovery (Phase 4.4) ---

VALID_AGENT_TOML = """\
[agent]
name = "watson"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test-key"

[workspace]
path = "/home/oa-watson/workspace"
"""


class TestLoadAgentConfig:
    """Config discovery from /etc/openalph/agents/<name>.toml."""

    def test_discovers_and_loads(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "agents"
        config_dir.mkdir()
        toml = VALID_AGENT_TOML.replace(
            "/home/oa-watson/workspace", str(tmp_path))
        (config_dir / "watson.toml").write_text(toml)
        monkeypatch.setattr("openalph.config.CONFIG_DIR", config_dir)
        config = load_agent_config("watson")
        assert config.name == "watson"

    def test_missing_config_raises(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "agents"
        config_dir.mkdir()
        monkeypatch.setattr("openalph.config.CONFIG_DIR", config_dir)
        with pytest.raises(ConfigError):
            load_agent_config("nonexistent")

    def test_invalid_toml_raises(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "agents"
        config_dir.mkdir()
        (config_dir / "bad.toml").write_text("not valid toml {{{")
        monkeypatch.setattr("openalph.config.CONFIG_DIR", config_dir)
        with pytest.raises(ConfigError):
            load_agent_config("bad")

    def test_config_dir_constant(self):
        assert CONFIG_DIR == Path("/etc/openalph/agents")


# --- Dataclass ---


class TestAgentConfig:

    def test_fields(self):
        config = AgentConfig(
            name="merry",
            default_model="claude-sonnet-4-20250514",
            max_tokens=8192,
            model_max_tokens=200000,
            providers={
                "anthropic": ProviderConfig(
                    key="anthropic",
                    type="anthropic",
                    api_key="sk-test",
                )
            },
            workspace=Path("/tmp/test"),
            matrix=None,
        )
        assert config.name == "merry"
        assert config.max_tokens == 8192
        assert config.model_max_tokens == 200000
        assert config.workspace == Path("/tmp/test")
        assert config.matrix is None


# --- Fix 4: Subprocess timeout on cmd resolution ---


class TestSubprocessTimeout:

    def test_api_key_cmd_timeout_skips_provider(self, tmp_path):
        """When api_key_cmd hangs, the provider is SKIPPED with the "timed
        out" reason (kdsn.292 flip: single attempt, degraded start; was a
        fatal ConfigError). The _resolve_secret reason text is unchanged."""
        from unittest.mock import patch as _patch
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key_cmd = "sleep 999"

[workspace]
path = "/tmp/test"
""")
        with _patch("openalph.config.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 999", timeout=10)
            config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "anthropic" in config.skipped_providers
        assert "timed out" in config.skipped_providers["anthropic"].lower()
        assert mock_run.call_count == 1  # kdsn.292: exactly ONE attempt, never re-hammered

    def test_password_cmd_timeout_raises_config_error(self, tmp_path):
        """When matrix.password_cmd hangs, ConfigError is raised. The [matrix]
        section is NOT a provider block, so matrix credentials stay
        structural-FATAL: the kdsn.292 skip-with-reason flip applies only to
        [providers.*] and must not degrade the Matrix layer (verified against
        config.py: _parse_matrix_config calls _resolve_secret directly — its
        ConfigError on timeout is never caught by the provider-skip loop)."""
        from unittest.mock import patch as _patch
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"

[matrix]
homeserver = "https://matrix.test"
user_id = "@bot:matrix.test"
password_cmd = "sleep 999"
""")
        with _patch("openalph.config.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 999", timeout=10)
            with pytest.raises(ConfigError, match="timed out"):
                load_config(tmp_path / "agent.toml")

    def test_access_token_cmd_timeout_raises_config_error(self, tmp_path):
        """When matrix.access_token_cmd hangs, ConfigError is raised."""
        from unittest.mock import patch as _patch
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"

[matrix]
homeserver = "https://matrix.test"
user_id = "@bot:matrix.test"
access_token_cmd = "sleep 999"
""")
        with _patch("openalph.config.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 999", timeout=10)
            with pytest.raises(ConfigError, match="timed out"):
                load_config(tmp_path / "agent.toml")
