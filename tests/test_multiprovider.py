"""Tests for multi-provider routing (kdsn.27).

Covers:
- ProviderConfig dataclass and resolve_model()
- Config loading: [providers.*] (new) and [provider] (backward compat)
- provider.complete() with model routing
- Agent.active_model and switch_model()
- /model command guards (vision, context window)
- Sub-agent config derivation
- Quirks handling

Interface contracts:
    ProviderConfig: key, type, api_key, base_url, quirks
    AgentConfig: name, default_model, max_tokens, providers, workspace, ...
    resolve_model(model_str, providers) -> (ProviderConfig, api_model_name)
    complete(config, system, messages, tools, max_tokens, model) -> Response
    Agent.active_model: str (mutable, initialized from config.default_model)
    Agent.switch_model(model_str, room_id) -> str | None (error message)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from openalph.config import (
    AgentConfig,
    ConfigError,
    ProviderConfig,
    load_config,
    resolve_model,
)
from openalph.provider import Response, ToolCall, Usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_provider(key="default", type="anthropic", api_key="sk-test",
                  base_url=None, quirks=None):
    return ProviderConfig(
        key=key,
        type=type,
        api_key=api_key,
        base_url=base_url,
        quirks=quirks or [],
    )


def make_config(workspace="/tmp/test", **kwargs):
    """Create an AgentConfig with multi-provider defaults."""
    defaults = {
        "name": "test-agent",
        "default_model": "anthropic/claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "providers": {
            "anthropic": make_provider(
                key="anthropic", type="anthropic", api_key="sk-ant-test"
            ),
        },
        "workspace": Path(workspace),
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_multi_config(workspace="/tmp/test", **kwargs):
    """Create an AgentConfig with multiple providers."""
    defaults = {
        "name": "test-agent",
        "default_model": "anthropic/claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "providers": {
            "anthropic": make_provider(
                key="anthropic", type="anthropic", api_key="sk-ant-test"
            ),
            "openrouter": make_provider(
                key="openrouter", type="openai", api_key="sk-or-test",
                base_url="https://openrouter.ai/api/v1"
            ),
            "ollama": make_provider(
                key="ollama", type="openai", api_key="ollama",
                base_url="http://localhost:11434/v1"
            ),
        },
        "workspace": Path(workspace),
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def mock_anthropic_response(content="Hello", model="claude-sonnet-4-20250514",
                            stop_reason="end_turn"):
    """Create a mock Anthropic API response."""
    mock_resp = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = content
    mock_resp.content = [text_block]
    mock_resp.model = model
    mock_resp.stop_reason = stop_reason
    mock_resp.usage = MagicMock(
        input_tokens=100, output_tokens=50,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    )
    return mock_resp


def mock_openai_response(content="Hello", model="moonshotai/kimi-k2.5",
                         finish_reason="stop"):
    """Create a mock OpenAI API response."""
    mock_resp = MagicMock()
    mock_resp.model = model
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    mock_resp.choices[0].message.tool_calls = None
    mock_resp.choices[0].finish_reason = finish_reason
    mock_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    return mock_resp


# ===========================================================================
# SECTION 1: ProviderConfig + resolve_model
# ===========================================================================

class TestProviderConfig:

    def test_create_anthropic_provider(self):
        p = ProviderConfig(
            key="anthropic", type="anthropic",
            api_key="sk-ant-test", base_url=None, quirks=[],
        )
        assert p.key == "anthropic"
        assert p.type == "anthropic"
        assert p.api_key == "sk-ant-test"
        assert p.base_url is None
        assert p.quirks == []

    def test_create_openai_provider_with_base_url(self):
        p = ProviderConfig(
            key="openrouter", type="openai",
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
            quirks=[],
        )
        assert p.key == "openrouter"
        assert p.type == "openai"
        assert p.base_url == "https://openrouter.ai/api/v1"

    def test_create_provider_with_quirks(self):
        p = ProviderConfig(
            key="vllm-mlx", type="openai",
            api_key="sk-vllm", base_url="http://mac:8000/v1",
            quirks=["no_system_role", "no_streaming_tools"],
        )
        assert p.quirks == ["no_system_role", "no_streaming_tools"]


class TestResolveModel:

    def test_resolve_prefixed_anthropic(self):
        providers = {
            "anthropic": make_provider(key="anthropic"),
            "openrouter": make_provider(key="openrouter", type="openai",
                                        base_url="https://openrouter.ai/api/v1"),
        }
        prov, api_model = resolve_model("anthropic/claude-opus-4-6", providers)
        assert prov.key == "anthropic"
        assert api_model == "claude-opus-4-6"

    def test_resolve_prefixed_openrouter_nested_model_name(self):
        """OpenRouter model names contain slashes: moonshotai/kimi-k2.5"""
        providers = {
            "openrouter": make_provider(key="openrouter", type="openai",
                                        base_url="https://openrouter.ai/api/v1"),
        }
        prov, api_model = resolve_model(
            "openrouter/moonshotai/kimi-k2.5", providers
        )
        assert prov.key == "openrouter"
        assert api_model == "moonshotai/kimi-k2.5"

    def test_resolve_prefixed_ollama(self):
        providers = {
            "ollama": make_provider(key="ollama", type="openai",
                                    base_url="http://localhost:11434/v1"),
        }
        prov, api_model = resolve_model("ollama/devstral-2:123b", providers)
        assert prov.key == "ollama"
        assert api_model == "devstral-2:123b"

    def test_resolve_prefixed_vllm_mlx_nested(self):
        """vllm-mlx model names contain slashes too."""
        providers = {
            "vllm-mlx": make_provider(key="vllm-mlx", type="openai",
                                       base_url="http://mac:8000/v1"),
        }
        prov, api_model = resolve_model(
            "vllm-mlx/mlx-community/Qwen3.5-397B-A17B-4bit", providers
        )
        assert prov.key == "vllm-mlx"
        assert api_model == "mlx-community/Qwen3.5-397B-A17B-4bit"




    def test_resolve_unknown_prefix_raises(self):
        """Prefix that doesn't match any provider key raises ValueError."""
        providers = {
            "anthropic": make_provider(key="anthropic"),
            "openrouter": make_provider(key="openrouter", type="openai",
                                        base_url="https://openrouter.ai/api/v1"),
        }
        with pytest.raises(ValueError, match="Unknown provider"):
            resolve_model("cohere/some-model", providers)

    def test_resolve_empty_model_string_raises(self):
        providers = {"anthropic": make_provider(key="anthropic")}
        with pytest.raises(ValueError):
            resolve_model("", providers)


# ===========================================================================
# SECTION 2: Config loading — [providers.*] (new format)
# ===========================================================================

class TestLoadMultiProviderConfig:

    def test_load_multi_provider_toml(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "merry"
default_model = "anthropic/claude-opus-4-6"
max_tokens = 8192
vision = true

[providers.anthropic]
type = "anthropic"
api_key = "sk-ant-test"

[providers.openrouter]
type = "openai"
api_key = "sk-or-test"
base_url = "https://openrouter.ai/api/v1"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.name == "merry"
        assert config.default_model == "anthropic/claude-opus-4-6"
        assert len(config.providers) == 2
        assert "anthropic" in config.providers
        assert "openrouter" in config.providers
        assert config.providers["anthropic"].type == "anthropic"
        assert config.providers["anthropic"].api_key == "sk-ant-test"
        assert config.providers["openrouter"].type == "openai"
        assert config.providers["openrouter"].base_url == "https://openrouter.ai/api/v1"

    def test_load_multi_provider_with_quirks(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "vllm-mlx/some-model"
max_tokens = 4096

[providers.vllm-mlx]
type = "openai"
api_key = "placeholder"
base_url = "http://mac:8000/v1"
quirks = ["no_system_role", "no_streaming_tools"]

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        p = config.providers["vllm-mlx"]
        assert p.quirks == ["no_system_role", "no_streaming_tools"]

    def test_load_multi_provider_api_key_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_ANT_KEY", "sk-from-env")
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/claude-sonnet-4-20250514"

[providers.anthropic]
type = "anthropic"
api_key_env = "TEST_ANT_KEY"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.providers["anthropic"].api_key == "sk-from-env"

    def test_load_multi_provider_api_key_cmd(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/claude-sonnet-4-20250514"

[providers.anthropic]
type = "anthropic"
api_key_cmd = "echo sk-from-cmd"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.providers["anthropic"].api_key == "sk-from-cmd"

    def test_load_multi_provider_missing_api_key_raises(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/claude-sonnet-4-20250514"

[providers.anthropic]
type = "anthropic"

[workspace]
path = "/tmp/test"
""")
        with pytest.raises(ConfigError, match="[Aa]PI key"):
            load_config(tmp_path / "agent.toml")

    def test_load_multi_provider_openai_missing_base_url_raises(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "openrouter/model"

[providers.openrouter]
type = "openai"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        with pytest.raises(ConfigError, match="base_url"):
            load_config(tmp_path / "agent.toml")

    def test_load_multi_provider_invalid_type_raises(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "cohere/model"

[providers.cohere]
type = "cohere"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        with pytest.raises(ConfigError, match="[Ii]nvalid provider type"):
            load_config(tmp_path / "agent.toml")

    def test_load_multi_provider_with_ollama_placeholder_key(self, tmp_path):
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "ollama/devstral-2:123b"

[providers.ollama]
type = "openai"
api_key = "ollama"
base_url = "http://localhost:11434/v1"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.providers["ollama"].api_key == "ollama"

    def test_load_missing_default_model_raises(self, tmp_path):
        """New format requires default_model, not model."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        with pytest.raises(ConfigError):
            load_config(tmp_path / "agent.toml")


# (Legacy backward compat tests removed — [provider] format no longer supported)


# ===========================================================================
# SECTION 4: AgentConfig new shape
# ===========================================================================

class TestAgentConfigNewShape:

    def test_agentconfig_has_providers_dict(self):
        config = make_config()
        assert isinstance(config.providers, dict)
        assert "anthropic" in config.providers
        assert isinstance(config.providers["anthropic"], ProviderConfig)

    def test_agentconfig_has_default_model(self):
        config = make_config()
        assert config.default_model == "anthropic/claude-sonnet-4-20250514"

    def test_agentconfig_no_model_field(self):
        """AgentConfig should not have a 'model' field (replaced by default_model)."""
        config = make_config()
        # The dataclass should NOT have 'model' as a field
        field_names = [f.name for f in config.__dataclass_fields__.values()]
        assert "model" not in field_names

    def test_agentconfig_no_provider_field(self):
        """AgentConfig should not have 'provider' as a field (replaced by providers)."""
        config = make_config()
        field_names = [f.name for f in config.__dataclass_fields__.values()]
        assert "provider" not in field_names

    def test_agentconfig_no_api_key_field(self):
        config = make_config()
        field_names = [f.name for f in config.__dataclass_fields__.values()]
        assert "api_key" not in field_names

    def test_agentconfig_no_base_url_field(self):
        config = make_config()
        field_names = [f.name for f in config.__dataclass_fields__.values()]
        assert "base_url" not in field_names


# ===========================================================================
# SECTION 5: provider.complete() with multi-provider
# ===========================================================================

class TestCompleteMultiProvider:

    @pytest.mark.asyncio
    async def test_complete_routes_to_anthropic(self):
        """complete() with anthropic-prefixed model uses Anthropic SDK."""
        config = make_multi_config()
        mock_resp = mock_anthropic_response(content="Hi from Claude")

        with patch("openalph.provider._get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_client

            from openalph.provider import complete
            response = await complete(
                config=config,
                system="You are helpful.",
                messages=[{"role": "user", "content": "Hello"}],
                model="anthropic/claude-sonnet-4-20250514",
            )

            assert response.content == "Hi from Claude"
            # Verify the API was called with the un-prefixed model name
            call_kwargs = mock_client.messages.create.call_args
            assert call_kwargs.kwargs["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_complete_routes_to_openrouter(self):
        """complete() with openrouter-prefixed model uses OpenAI SDK."""
        config = make_multi_config()
        mock_resp = mock_openai_response(content="Hi from Kimi")

        with patch("openalph.provider._get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_client

            from openalph.provider import complete
            response = await complete(
                config=config,
                system="You are helpful.",
                messages=[{"role": "user", "content": "Hello"}],
                model="openrouter/moonshotai/kimi-k2.5",
            )

            assert response.content == "Hi from Kimi"
            call_kwargs = mock_client.chat.completions.create.call_args
            assert call_kwargs.kwargs["model"] == "moonshotai/kimi-k2.5"

    @pytest.mark.asyncio
    async def test_complete_uses_default_model_when_none(self):
        """complete() without model param uses config.default_model."""
        config = make_config(default_model="anthropic/claude-sonnet-4-20250514")
        mock_resp = mock_anthropic_response()

        with patch("openalph.provider._get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_client

            from openalph.provider import complete
            await complete(
                config=config,
                system="test",
                messages=[{"role": "user", "content": "hi"}],
            )

            call_kwargs = mock_client.messages.create.call_args
            assert call_kwargs.kwargs["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_complete_unknown_provider_raises(self):
        """complete() with model prefix not in providers raises ValueError."""
        config = make_config()  # only has "anthropic" provider
        from openalph.provider import complete
        with pytest.raises(ValueError, match="[Uu]nknown provider"):
            await complete(
                config=config,
                system="test",
                messages=[{"role": "user", "content": "hi"}],
                model="openrouter/moonshotai/kimi-k2.5",
            )


# ===========================================================================
# SECTION 6: Client cache with ProviderConfig
# ===========================================================================

class TestClientCacheMultiProvider:

    def test_get_client_uses_provider_config(self):
        """_get_client creates appropriate client based on ProviderConfig.type."""
        from openalph.provider import _get_client, _client_cache
        _client_cache.clear()

        prov = make_provider(key="anthropic", type="anthropic", api_key="sk-ant-123")
        client = _get_client(prov)
        assert client is not None

    def test_get_client_caches_by_type_key_url(self):
        """Same (type, api_key, base_url) returns same client."""
        from openalph.provider import _get_client, _client_cache
        _client_cache.clear()

        prov = make_provider(key="openrouter", type="openai", api_key="sk-or",
                             base_url="https://openrouter.ai/api/v1")
        client1 = _get_client(prov)
        client2 = _get_client(prov)
        assert client1 is client2

    def test_get_client_different_keys_different_clients(self):
        """Different API keys produce different clients."""
        from openalph.provider import _get_client, _client_cache
        _client_cache.clear()

        prov1 = make_provider(key="or1", type="openai", api_key="sk-key-1",
                              base_url="https://openrouter.ai/api/v1")
        prov2 = make_provider(key="or2", type="openai", api_key="sk-key-2",
                              base_url="https://openrouter.ai/api/v1")
        assert _get_client(prov1) is not _get_client(prov2)

    def test_get_client_different_base_urls_different_clients(self):
        """Different base URLs produce different clients."""
        from openalph.provider import _get_client, _client_cache
        _client_cache.clear()

        prov1 = make_provider(key="or", type="openai", api_key="sk-same",
                              base_url="https://openrouter.ai/api/v1")
        prov2 = make_provider(key="ollama", type="openai", api_key="sk-same",
                              base_url="http://localhost:11434/v1")
        assert _get_client(prov1) is not _get_client(prov2)


# ===========================================================================
# SECTION 7: Agent.active_model + switch_model
# ===========================================================================

class TestAgentActiveModel:

    def test_agent_active_model_initialized_from_config(self, tmp_path):
        """Agent.active_model starts as config.default_model."""
        (tmp_path / "SAFETY.md").write_text("")
        config = make_config(workspace=str(tmp_path),
                             default_model="anthropic/claude-sonnet-4-20250514")
        from openalph.agent import Agent
        agent = Agent(config)
        assert agent.active_model == "anthropic/claude-sonnet-4-20250514"

    def test_agent_status_reports_active_model(self, tmp_path):
        """Agent.status() includes the active model, not just default."""
        (tmp_path / "SAFETY.md").write_text("")
        config = make_config(workspace=str(tmp_path))
        from openalph.agent import Agent
        agent = Agent(config)
        status = agent.status()
        assert status["model"] == agent.active_model

    def test_switch_model_changes_active_model(self, tmp_path):
        """switch_model() updates active_model."""
        (tmp_path / "SAFETY.md").write_text("")
        config = make_multi_config(workspace=str(tmp_path))
        from openalph.agent import Agent
        agent = Agent(config)

        result = agent.switch_model("openrouter/moonshotai/kimi-k2.5")
        assert result is None  # no error
        assert agent.active_model == "openrouter/moonshotai/kimi-k2.5"

    def test_switch_model_unknown_provider_returns_error(self, tmp_path):
        """switch_model() returns error string for unknown provider prefix."""
        (tmp_path / "SAFETY.md").write_text("")
        config = make_config(workspace=str(tmp_path))
        from openalph.agent import Agent
        agent = Agent(config)

        result = agent.switch_model("cohere/some-model")
        assert result is not None
        assert "cohere" in result.lower() or "unknown" in result.lower()
        # Active model should NOT change on error
        assert agent.active_model == config.default_model

    def test_switch_model_vision_guard(self, tmp_path):
        """Block model switch when images exist in room history and target is non-vision."""
        (tmp_path / "SAFETY.md").write_text("")
        config = make_multi_config(workspace=str(tmp_path), vision=True)
        from openalph.agent import Agent
        agent = Agent(config)

        # Simulate images in history
        room_id = "!test:server"
        history = agent.history(room_id)
        history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "media_type": "image/jpeg", "data": "base64data"},
            ],
        })

        # Attempt switch — images are present so it should be blocked
        result = agent.switch_model("ollama/devstral-2:123b", room_id=room_id)
        assert result is not None
        assert "image" in result.lower() or "vision" in result.lower()
        # Active model unchanged
        assert agent.active_model == config.default_model

    def test_switch_model_context_window_guard(self, tmp_path):
        """Block model switch when context exceeds new model's max tokens."""
        (tmp_path / "SAFETY.md").write_text("")
        config = make_multi_config(
            workspace=str(tmp_path),
            model_max_tokens=200000,
            max_tokens=8192,
        )
        from openalph.agent import Agent
        agent = Agent(config)

        # Fill room with enough history to exceed a smaller model's window
        room_id = "!test:server"
        history = agent.history(room_id)
        # Add enough text to represent ~50K tokens (~200K chars)
        big_text = "x" * 200000
        history.append({"role": "user", "content": big_text})

        # Model limits: if target model has 32K max, this should fail
        result = agent.switch_model(
            "ollama/devstral-2:123b", room_id=room_id
        )
        # With a conservative 32K default for unknown models, 50K tokens should be blocked
        assert result is not None
        assert "context" in result.lower() or "exceed" in result.lower()

    def test_switch_model_requires_fully_qualified(self, tmp_path):
        """switch_model() with unqualified model string on multi-provider config errors."""
        (tmp_path / "SAFETY.md").write_text("")
        config = make_multi_config(workspace=str(tmp_path))
        from openalph.agent import Agent
        agent = Agent(config)

        # Bare model name should fail with multi-provider
        result = agent.switch_model("claude-sonnet-4-20250514")
        assert result is not None  # error message returned


# ===========================================================================
# SECTION 8: Quirks handling in provider
# ===========================================================================

class TestQuirksHandling:

    @pytest.mark.asyncio
    async def test_no_system_role_quirk_folds_system_into_user(self):
        """When provider has 'no_system_role' quirk, system prompt
        is prepended to first user message instead of separate system role."""
        config = make_config(
            default_model="vllm-mlx/some-model",
            providers={
                "vllm-mlx": make_provider(
                    key="vllm-mlx", type="openai",
                    api_key="placeholder",
                    base_url="http://mac:8000/v1",
                    quirks=["no_system_role"],
                ),
            },
        )

        mock_resp = mock_openai_response(content="response")

        with patch("openalph.provider._get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_client

            from openalph.provider import complete
            await complete(
                config=config,
                system="You are a test assistant.",
                messages=[{"role": "user", "content": "Hello"}],
                model="vllm-mlx/some-model",
            )

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            messages = call_kwargs["messages"]
            # No system role message
            assert all(m["role"] != "system" for m in messages)
            # System content should be in the first user message
            first_user = next(m for m in messages if m["role"] == "user")
            assert "You are a test assistant." in first_user["content"]


# ===========================================================================
# SECTION 9: Sub-agent config derivation
# ===========================================================================

class TestSubagentMultiProvider:

    @pytest.mark.asyncio
    async def test_subagent_model_override_uses_correct_provider(self):
        """Sub-agent with model override routes through the right provider."""
        config = make_multi_config()
        mock_resp = mock_openai_response(content="Research done")

        with patch("openalph.provider._get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_client

            from openalph.tools.subagent import run_subagent
            result = await run_subagent(
                task="Research this topic",
                config=config,
                model="openrouter/moonshotai/kimi-k2.5",
            )

            assert result.content == "Research done"
            assert not result.is_error
            # Verify OpenAI path was used (system as first message)
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "moonshotai/kimi-k2.5"

    @pytest.mark.asyncio
    async def test_subagent_inherits_full_provider_registry(self):
        """Sub-agent config has access to all parent providers."""
        config = make_multi_config()
        from openalph.config import resolve_model
        for prefix in config.providers:
            model_str = f"{prefix}/test-model"
            prov, api_model = resolve_model(model_str, config.providers)
            assert prov.key == prefix

    @pytest.mark.asyncio
    async def test_subagent_uses_default_model_when_no_override(self):
        """Sub-agent without model param uses config.default_model."""
        config = make_config(default_model="anthropic/claude-sonnet-4-20250514")
        mock_resp = mock_anthropic_response(content="Done")

        with patch("openalph.provider._get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_resp)
            mock_get.return_value = mock_client

            from openalph.tools.subagent import run_subagent
            result = await run_subagent(
                task="Do something",
                config=config,
            )

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["model"] == "claude-sonnet-4-20250514"


# ===========================================================================
# SECTION 10: handle_input passes active_model to complete()
# ===========================================================================

class TestHandleInputMultiProvider:

    @pytest.mark.asyncio
    async def test_handle_input_uses_active_model(self, tmp_path):
        """Agent.handle_input passes active_model to complete()."""
        (tmp_path / "SAFETY.md").write_text("")
        config = make_multi_config(
            workspace=str(tmp_path),
            default_model="anthropic/claude-sonnet-4-20250514",
        )

        from openalph.agent import Agent
        agent = Agent(config)

        mock_resp = Response(
            content="Hello!",
            model="claude-sonnet-4-20250514",
            usage=Usage(input_tokens=50, output_tokens=25),
            stop_reason="end_turn",
        )

        with patch("openalph.agent.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_resp
            await agent.handle_input("Hi", "room1")

            call_kwargs = mock_complete.call_args
            assert call_kwargs.kwargs.get("model") == "anthropic/claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_handle_input_after_model_switch(self, tmp_path):
        """After switch_model, handle_input uses the new model."""
        (tmp_path / "SAFETY.md").write_text("")
        config = make_multi_config(
            workspace=str(tmp_path),
            default_model="anthropic/claude-sonnet-4-20250514",
        )

        from openalph.agent import Agent
        agent = Agent(config)
        agent.switch_model("openrouter/moonshotai/kimi-k2.5")

        mock_resp = Response(
            content="Hi from Kimi",
            model="moonshotai/kimi-k2.5",
            usage=Usage(input_tokens=50, output_tokens=25),
            stop_reason="stop",
        )

        with patch("openalph.agent.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_resp
            await agent.handle_input("Hi", "room1")

            call_kwargs = mock_complete.call_args
            assert call_kwargs.kwargs.get("model") == "openrouter/moonshotai/kimi-k2.5"


# ===========================================================================
# SECTION 11: Model limits (optional TOML table)
# ===========================================================================

class TestModelLimits:

    def test_load_model_limits_from_toml(self, tmp_path):
        """Optional [model_limits] table parsed from config."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/claude-opus-4-6"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[providers.ollama]
type = "openai"
api_key = "ollama"
base_url = "http://localhost:11434/v1"

[model_limits]
"anthropic/claude-opus-4-6" = 200000
"anthropic/claude-haiku-4-5" = 200000
"ollama/devstral-2:123b" = 32000

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert hasattr(config, "model_limits")
        assert config.model_limits["anthropic/claude-opus-4-6"] == 200000
        assert config.model_limits["ollama/devstral-2:123b"] == 32000

    def test_model_limits_defaults_empty(self, tmp_path):
        """If no [model_limits] section, defaults to empty dict."""
        (tmp_path / "agent.toml").write_text("""
[agent]
name = "test"
default_model = "anthropic/claude-sonnet-4-20250514"

[providers.anthropic]
type = "anthropic"
api_key = "sk-test"

[workspace]
path = "/tmp/test"
""")
        config = load_config(tmp_path / "agent.toml")
        assert config.model_limits == {}


# --- ProviderError wrapping (kdsn.61) ---


class TestProviderErrorWrapping:
    """API errors from SDKs are wrapped as ProviderError with sanitized messages."""

    @staticmethod
    def _make_response(status_code, json_body=None):
        """Create an httpx.Response with a request attached (required by SDKs)."""
        import httpx
        request = httpx.Request("POST", "https://api.example.com/v1/messages")
        return httpx.Response(status_code, json=json_body or {}, request=request)

    def _make_config(self, provider_type="anthropic", provider_key="anthropic"):
        providers = {
            provider_key: ProviderConfig(
                key=provider_key, type=provider_type,
                api_key="sk-test-key-1234567890",
                base_url="https://openrouter.ai/api/v1" if provider_type == "openai" else None,
            )
        }
        return AgentConfig(
            name="test", default_model=f"{provider_key}/test-model",
            max_tokens=1024, providers=providers, workspace=Path("/tmp/test"),
        )

    @pytest.mark.asyncio
    async def test_anthropic_bad_model_raises_provider_error(self):
        """Anthropic NotFoundError → ProviderError with message."""
        import anthropic as anthropic_sdk
        from openalph.provider import complete, ProviderError

        config = self._make_config("anthropic")
        resp = self._make_response(404, {"error": {"message": "model: invalid model: bongo"}})
        exc = anthropic_sdk.NotFoundError(
            message="model: invalid model: bongo", response=resp, body=None
        )

        with patch("openalph.provider._get_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(side_effect=exc)
            with pytest.raises(ProviderError) as exc_info:
                await complete(config, "system", [{"role": "user", "content": "hi"}])
            assert "invalid model" in str(exc_info.value)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_openai_bad_model_raises_provider_error(self):
        """OpenAI NotFoundError → ProviderError with message."""
        import openai as openai_sdk
        from openalph.provider import complete, ProviderError

        config = self._make_config("openai", "openrouter")
        resp = self._make_response(404, {"error": {"message": "Model not found: bongo"}})
        exc = openai_sdk.NotFoundError(
            message="Model not found: bongo", response=resp, body=None
        )

        with patch("openalph.provider._get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(side_effect=exc)
            with pytest.raises(ProviderError) as exc_info:
                await complete(config, "system", [{"role": "user", "content": "hi"}])
            assert "Model not found" in str(exc_info.value)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_anthropic_auth_error_raises_provider_error(self):
        """Anthropic AuthenticationError → ProviderError with 401."""
        import anthropic as anthropic_sdk
        from openalph.provider import complete, ProviderError

        config = self._make_config("anthropic")
        resp = self._make_response(401, {"error": {"message": "Invalid API key"}})
        exc = anthropic_sdk.AuthenticationError(
            message="Invalid API key", response=resp, body=None
        )

        with patch("openalph.provider._get_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(side_effect=exc)
            with pytest.raises(ProviderError) as exc_info:
                await complete(config, "system", [{"role": "user", "content": "hi"}])
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_anthropic_connection_error_raises_provider_error(self):
        """Anthropic APIConnectionError → ProviderError with 'unreachable'."""
        import anthropic as anthropic_sdk
        from openalph.provider import complete, ProviderError

        config = self._make_config("anthropic")
        exc = anthropic_sdk.APIConnectionError(request=MagicMock())

        with patch("openalph.provider._get_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(side_effect=exc)
            with pytest.raises(ProviderError, match="unreachable"):
                await complete(config, "system", [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_anthropic_timeout_error_raises_provider_error(self):
        """Anthropic APITimeoutError → ProviderError with 'timed out'."""
        import anthropic as anthropic_sdk
        from openalph.provider import complete, ProviderError

        config = self._make_config("anthropic")
        exc = anthropic_sdk.APITimeoutError(request=MagicMock())

        with patch("openalph.provider._get_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(side_effect=exc)
            with pytest.raises(ProviderError, match="timed out"):
                await complete(config, "system", [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_openai_connection_error_raises_provider_error(self):
        """OpenAI APIConnectionError → ProviderError with 'unreachable'."""
        import openai as openai_sdk
        from openalph.provider import complete, ProviderError

        config = self._make_config("openai", "openrouter")
        exc = openai_sdk.APIConnectionError(request=MagicMock())

        with patch("openalph.provider._get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(side_effect=exc)
            with pytest.raises(ProviderError, match="unreachable"):
                await complete(config, "system", [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_openai_rate_limit_error_raises_provider_error(self):
        """OpenAI RateLimitError → ProviderError with 429."""
        import openai as openai_sdk
        from openalph.provider import complete, ProviderError

        config = self._make_config("openai", "openrouter")
        resp = self._make_response(429, {"error": {"message": "Rate limit exceeded"}})
        exc = openai_sdk.RateLimitError(
            message="Rate limit exceeded", response=resp, body=None
        )

        with patch("openalph.provider._get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(side_effect=exc)
            with pytest.raises(ProviderError) as exc_info:
                await complete(config, "system", [{"role": "user", "content": "hi"}])
            assert exc_info.value.status_code == 429


class TestSanitizeError:
    """API key patterns are stripped from error messages."""

    def test_strips_anthropic_key(self):
        from openalph.provider import _sanitize_error
        msg = "Failed with key sk-ant-api03-abc123xyz456def789"
        assert "sk-ant" not in _sanitize_error(msg)
        assert "[REDACTED]" in _sanitize_error(msg)

    def test_strips_openai_key(self):
        from openalph.provider import _sanitize_error
        msg = "Invalid key: sk-proj-1234567890abcdef"
        assert "sk-proj" not in _sanitize_error(msg)

    def test_preserves_normal_message(self):
        from openalph.provider import _sanitize_error
        msg = "model: invalid model: bongo"
        assert _sanitize_error(msg) == msg

    def test_strips_openrouter_key(self):
        from openalph.provider import _sanitize_error
        msg = "Auth failed: sk-or-v1-abc123def456ghi789jkl012"
        assert "sk-or" not in _sanitize_error(msg)
