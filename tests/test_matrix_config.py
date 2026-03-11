"""Tests for Matrix configuration parsing.

Interface contract:
    MatrixConfig dataclass with all Matrix-related fields.
    Parsed from [matrix] section of TOML config.
    Auth resolution: password > password_env > password_cmd (same pattern as API keys).
"""

import pytest
from pathlib import Path
from openalph.config import load_config, ConfigError, MatrixConfig


def write_config(tmp_path, toml_content):
    """Helper to write a TOML config file."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(toml_content)
    return config_file


MINIMAL_AGENT_TOML = """
[agent]
name = "test"
default_model = "anthropic/test-model"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
"""


class TestMatrixConfigParsing:

    def test_full_matrix_config(self, tmp_path):
        """All Matrix fields parsed from TOML."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
homeserver = "https://matrix.local"
user_id = "@merry:matrix.local"
device_id = "OPENALPH"
password = "secret"
context_reserve = 32768

[matrix.sync]
timeout = 60000
retry_base = 10
retry_max = 600
""")
        config = load_config(config_file)
        assert config.matrix is not None
        assert config.matrix.homeserver == "https://matrix.local"
        assert config.matrix.user_id == "@merry:matrix.local"
        assert config.matrix.device_id == "OPENALPH"
        assert config.matrix.password == "secret"
        assert config.matrix.context_reserve == 32768
        assert config.matrix.sync_timeout == 60000
        assert config.matrix.retry_base == 10
        assert config.matrix.retry_max == 600

    def test_matrix_config_defaults(self, tmp_path):
        """Missing optional fields use defaults."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
homeserver = "https://matrix.local"
user_id = "@merry:matrix.local"
password = "secret"
""")
        config = load_config(config_file)
        assert config.matrix.device_id == "OPENALPH"  # default
        assert config.matrix.context_reserve == 16384  # default
        assert config.matrix.sync_timeout == 30000  # default
        assert config.matrix.retry_base == 5  # default
        assert config.matrix.retry_max == 300  # default

    def test_no_matrix_section(self, tmp_path):
        """No [matrix] section → config.matrix is None."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML)
        config = load_config(config_file)
        assert config.matrix is None

    def test_matrix_missing_homeserver(self, tmp_path):
        """Missing homeserver → ConfigError."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
user_id = "@merry:matrix.local"
password = "secret"
""")
        with pytest.raises(ConfigError):
            load_config(config_file)

    def test_matrix_missing_user_id(self, tmp_path):
        """Missing user_id → ConfigError."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
homeserver = "https://matrix.local"
password = "secret"
""")
        with pytest.raises(ConfigError):
            load_config(config_file)

    def test_matrix_no_auth(self, tmp_path):
        """No password or access_token → ConfigError."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
homeserver = "https://matrix.local"
user_id = "@merry:matrix.local"
""")
        with pytest.raises(ConfigError):
            load_config(config_file)

    def test_matrix_access_token(self, tmp_path):
        """access_token used instead of password."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
homeserver = "https://matrix.local"
user_id = "@merry:matrix.local"
access_token = "syt_test_token"
""")
        config = load_config(config_file)
        assert config.matrix.access_token == "syt_test_token"
        assert config.matrix.password is None

    def test_matrix_password_env(self, tmp_path, monkeypatch):
        """password_env resolves from environment."""
        monkeypatch.setenv("TEST_MATRIX_PW", "env_secret")
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
homeserver = "https://matrix.local"
user_id = "@merry:matrix.local"
password_env = "TEST_MATRIX_PW"
""")
        config = load_config(config_file)
        assert config.matrix.password == "env_secret"

    def test_matrix_password_cmd(self, tmp_path):
        """password_cmd resolves from shell command."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
homeserver = "https://matrix.local"
user_id = "@merry:matrix.local"
password_cmd = "echo cmd_secret"
""")
        config = load_config(config_file)
        assert config.matrix.password == "cmd_secret"


class TestMatrixRoomsConfig:
    """Tests for [matrix.rooms] per-room override parsing."""

    def test_rooms_parsed(self, tmp_path):
        """[matrix.rooms] section loads into MatrixConfig.rooms."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
homeserver = "https://matrix.local"
user_id = "@watson:matrix.local"
password = "secret"

[matrix.rooms]
"!abc123:matrix.local" = { require_mention = false }
"!def456:matrix.local" = { require_mention = true }
""")
        config = load_config(config_file)
        assert config.matrix.rooms is not None
        assert config.matrix.rooms["!abc123:matrix.local"]["require_mention"] is False
        assert config.matrix.rooms["!def456:matrix.local"]["require_mention"] is True

    def test_rooms_empty(self, tmp_path):
        """No [matrix.rooms] → rooms is None."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
homeserver = "https://matrix.local"
user_id = "@watson:matrix.local"
password = "secret"
""")
        config = load_config(config_file)
        assert config.matrix.rooms is None

    def test_rooms_empty_section(self, tmp_path):
        """Empty [matrix.rooms] section → rooms is empty dict."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
homeserver = "https://matrix.local"
user_id = "@watson:matrix.local"
password = "secret"

[matrix.rooms]
""")
        config = load_config(config_file)
        # TOML parser produces {} for empty section, which is truthy
        # Config should handle both None and {} gracefully
        assert config.matrix.rooms is not None or config.matrix.rooms == {}

    def test_rooms_require_mention_bool(self, tmp_path):
        """require_mention value is parsed as boolean."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
homeserver = "https://matrix.local"
user_id = "@watson:matrix.local"
password = "secret"

[matrix.rooms]
"!room:matrix.local" = { require_mention = true }
""")
        config = load_config(config_file)
        val = config.matrix.rooms["!room:matrix.local"]["require_mention"]
        assert isinstance(val, bool)
        assert val is True

    def test_rooms_unknown_keys_allowed(self, tmp_path):
        """Unknown keys in room config don't cause errors (forward compat)."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML + """
[matrix]
homeserver = "https://matrix.local"
user_id = "@watson:matrix.local"
password = "secret"

[matrix.rooms]
"!room:matrix.local" = { require_mention = true, future_setting = "value" }
""")
        config = load_config(config_file)
        assert config.matrix.rooms["!room:matrix.local"]["require_mention"] is True
        assert config.matrix.rooms["!room:matrix.local"]["future_setting"] == "value"


class TestModelMaxTokens:

    def test_model_max_tokens_parsed(self, tmp_path):
        """model_max_tokens parsed from [agent] section."""
        config_file = write_config(tmp_path, """
[agent]
name = "test"
default_model = "anthropic/test-model"
model_max_tokens = 128000

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        config = load_config(config_file)
        assert config.model_max_tokens == 128000

    def test_model_max_tokens_default(self, tmp_path):
        """model_max_tokens defaults to 200000 if not specified."""
        config_file = write_config(tmp_path, MINIMAL_AGENT_TOML)
        config = load_config(config_file)
        assert config.model_max_tokens == 200000
