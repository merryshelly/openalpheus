"""Tests for agent configuration loading.

Interface contract:
    load_config(path: Path) -> AgentConfig
    AgentConfig: name, model, max_tokens, provider, api_key, base_url, workspace

Config is TOML. API key resolves from: api_key (direct), api_key_env (env var),
api_key_cmd (shell command). Precedence: api_key > api_key_env > api_key_cmd.
"""

import pytest
from pathlib import Path
from openalph.config import AgentConfig, load_config, load_agent_config, ConfigError, CONFIG_DIR


# --- Valid configs ---


class TestLoadValidConfig:

    def test_minimal_anthropic(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "merry"
model = "claude-sonnet-4-20250514"

[provider]
type = "anthropic"
api_key = "sk-test-key"

[workspace]
path = "/tmp/test-workspace"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.name == "merry"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.provider == "anthropic"
        assert config.api_key == "sk-test-key"
        assert config.base_url is None
        assert config.workspace == Path("/tmp/test-workspace")

    def test_max_tokens_default(self, tmp_path):
        """max_tokens defaults to 8192 if not specified."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
model = "test-model"

[provider]
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
model = "test-model"
max_tokens = 4096

[provider]
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
model = "moonshotai/kimi-k2.5"

[provider]
type = "openai"
api_key = "sk-or-test"
base_url = "https://openrouter.ai/api/v1"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.provider == "openai"
        assert config.base_url == "https://openrouter.ai/api/v1"

    def test_ollama_via_openai(self, tmp_path):
        """Ollama uses the OpenAI-compatible path."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "local"
model = "qwen3:235b-a22b"

[provider]
type = "openai"
api_key = "ollama"
base_url = "http://localhost:11434/v1"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.provider == "openai"
        assert config.model == "qwen3:235b-a22b"


# --- API key resolution ---


class TestApiKeyResolution:

    def test_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-from-env")
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
model = "test-model"

[provider]
type = "anthropic"
api_key_env = "TEST_API_KEY"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.api_key == "sk-from-env"

    def test_from_command(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
model = "test-model"

[provider]
type = "anthropic"
api_key_cmd = "echo sk-from-cmd"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.api_key == "sk-from-cmd"

    def test_precedence_direct_over_env(self, tmp_path, monkeypatch):
        """Direct api_key wins over api_key_env."""
        monkeypatch.setenv("TEST_KEY", "from-env")
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
model = "test"

[provider]
type = "anthropic"
api_key = "from-direct"
api_key_env = "TEST_KEY"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.api_key == "from-direct"

    def test_precedence_env_over_cmd(self, tmp_path, monkeypatch):
        """api_key_env wins over api_key_cmd."""
        monkeypatch.setenv("TEST_KEY", "from-env")
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
model = "test"

[provider]
type = "anthropic"
api_key_env = "TEST_KEY"
api_key_cmd = "echo from-cmd"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.api_key == "from-env"


# --- Validation errors ---


class TestConfigValidation:

    def test_missing_name(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
model = "test"

[provider]
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

[provider]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        with pytest.raises(ConfigError, match="model"):
            load_config(tmp_path / "agent.toml")

    def test_missing_api_key_entirely(self, tmp_path):
        """No api_key, api_key_env, or api_key_cmd."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
model = "test"

[provider]
type = "anthropic"

[workspace]
path = "/tmp/test"
""")
        with pytest.raises(ConfigError, match="api_key"):
            load_config(tmp_path / "agent.toml")

    def test_invalid_provider_type(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
model = "test"

[provider]
type = "google"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        with pytest.raises(ConfigError, match="provider"):
            load_config(tmp_path / "agent.toml")

    def test_openai_missing_base_url(self, tmp_path):
        """OpenAI-compatible provider requires base_url."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
model = "test"

[provider]
type = "openai"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        with pytest.raises(ConfigError, match="base_url"):
            load_config(tmp_path / "agent.toml")

    def test_env_var_not_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
model = "test"

[provider]
type = "anthropic"
api_key_env = "NONEXISTENT_VAR"

[workspace]
path = "/tmp/test"
""")
        with pytest.raises(ConfigError, match="NONEXISTENT_VAR"):
            load_config(tmp_path / "agent.toml")

    def test_command_fails(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
model = "test"

[provider]
type = "anthropic"
api_key_cmd = "false"

[workspace]
path = "/tmp/test"
""")
        with pytest.raises(ConfigError, match="api_key_cmd"):
            load_config(tmp_path / "agent.toml")

    def test_nonexistent_file(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(tmp_path / "nonexistent.toml")

    def test_missing_workspace(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
model = "test"

[provider]
type = "anthropic"
api_key = "sk-test"
""")
        with pytest.raises(ConfigError, match="workspace"):
            load_config(tmp_path / "agent.toml")


# --- Dataclass ---


# --- Agent config discovery (Phase 4.4) ---

VALID_AGENT_TOML = """\
[agent]
name = "watson"
model = "test-model"

[provider]
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
        (config_dir / "watson.toml").write_text(VALID_AGENT_TOML)
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
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            model_max_tokens=200000,
            provider="anthropic",
            api_key="sk-test",
            base_url=None,
            workspace=Path("/tmp/test"),
            matrix=None,
        )
        assert config.name == "merry"
        assert config.max_tokens == 8192
        assert config.model_max_tokens == 200000
        assert config.workspace == Path("/tmp/test")
        assert config.matrix is None
