"""Configuration loader for OpenAlph agents.

This module loads agent configuration from TOML files and resolves API keys
from multiple sources (direct value, environment variable, or shell command).

Design decisions:
- Use tomllib from stdlib (Python 3.11+) to avoid external dependencies
- API key resolution follows precedence: direct > env var > command
- All validation errors raise ConfigError with descriptive messages
- Workspace path is converted to Path object for consistency
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import subprocess
import tomllib


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


# System-wide config directory for per-agent TOML files
CONFIG_DIR = Path("/etc/openalph/agents")


@dataclass
class MatrixConfig:
    """Configuration for Matrix integration."""
    homeserver: str
    user_id: str
    device_id: str
    password: str | None
    access_token: str | None
    context_reserve: int
    sync_timeout: int
    retry_base: int
    retry_max: int
    rooms: dict[str, dict] | None = None  # Per-room overrides, e.g., {"!room:server": {"require_mention": True}}


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    key: str           # e.g., "openrouter", "anthropic", "default"
    type: str          # "anthropic" | "openai"
    api_key: str       # resolved value
    base_url: str | None = None
    quirks: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Configuration for an OpenAlph agent."""
    name: str
    default_model: str
    max_tokens: int
    providers: dict[str, ProviderConfig]
    workspace: Path
    model_max_tokens: int = 200000
    matrix: MatrixConfig | None = None
    max_iterations: int = 50
    truncation_limit: int = 50000
    vision: bool = False
    thinking: str = "off"
    model_limits: dict[str, int] = field(default_factory=dict)


def resolve_model(model_str: str, providers: dict[str, ProviderConfig]) -> tuple[ProviderConfig, str]:
    """Returns (provider_config, api_model_name).

    Model strings MUST be fully qualified: "<provider_key>/<api_model_name>".
    Split on the first "/". The prefix must match a key in providers.
    Everything after the first "/" is the API model name (may contain more slashes).
    """
    if not model_str:
        raise ValueError("Model string cannot be empty")
    prefix, sep, remainder = model_str.partition("/")
    if not sep:
        raise ValueError(
            f"Model '{model_str}' must be fully qualified as '<provider>/<model>'. "
            f"Available providers: {', '.join(sorted(providers.keys()))}"
        )
    if prefix not in providers:
        raise ValueError(
            f"Unknown provider '{prefix}' in model '{model_str}'. "
            f"Available providers: {', '.join(sorted(providers.keys()))}"
        )
    return providers[prefix], remainder
    raise ValueError(f"Cannot resolve model '{model_str}': ambiguous provider (multiple providers configured, none named 'default')")


def _resolve_api_key(section: dict) -> str:
    """Resolve API key from provider section using precedence: api_key > api_key_env > api_key_cmd.
    
    Args:
        section: Provider section dict from TOML
        
    Returns:
        Resolved API key string
        
    Raises:
        ConfigError: If no API key source is available or resolution fails
    """
    # Try direct api_key first
    if "api_key" in section:
        api_key = section["api_key"]
        if not api_key or not isinstance(api_key, str):
            raise ConfigError("api_key must be a non-empty string")
        return api_key
    
    # Try api_key_env if no direct key
    if "api_key_env" in section:
        env_var_name = section["api_key_env"]
        if not env_var_name or not isinstance(env_var_name, str):
            raise ConfigError("api_key_env must be a non-empty string")
        
        api_key = os.environ.get(env_var_name)
        if api_key is None:
            raise ConfigError(f"Environment variable {env_var_name} is not set")
        return api_key
    
    # Try api_key_cmd if no key yet
    if "api_key_cmd" in section:
        cmd = section["api_key_cmd"]
        if not cmd or not isinstance(cmd, str):
            raise ConfigError("api_key_cmd must be a non-empty string")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            api_key = result.stdout.strip()
            if not api_key:
                raise ConfigError("api_key_cmd produced empty output")
            return api_key
        except subprocess.TimeoutExpired:
            raise ConfigError("api_key_cmd timed out after 10 seconds")
        except subprocess.CalledProcessError as e:
            raise ConfigError(f"api_key_cmd failed with exit code {e.returncode}")
    
    # No API key source found
    raise ConfigError("No API key provided. One of api_key, api_key_env, or api_key_cmd must be set")


def load_config(path: Path) -> AgentConfig:
    """Load agent configuration from a TOML file.
    
    Args:
        path: Path to the TOML configuration file
        
    Returns:
        AgentConfig with all fields resolved and validated
        
    Raises:
        ConfigError: If configuration is invalid or cannot be loaded
    """
    # Validate file exists and is readable
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    
    try:
        with path.open("rb") as f:
            toml_data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        raise ConfigError(f"Failed to parse TOML: {e}")
    
    # Extract sections
    try:
        agent_section = toml_data["agent"]
    except KeyError as e:
        raise ConfigError(f"Missing required section: {e}")

    # Support both inline format (agent.provider, agent.api_key, agent.workspace.path)
    # and separate section format ([provider], [workspace])
    provider_section = toml_data.get("provider", {})
    workspace_section = toml_data.get("workspace", {})
    
    # Check for [providers.*] sections (new multi-provider format)
    # TOML parses [providers.anthropic] as nested dict under "providers"
    providers_sections = {}
    if "providers" in toml_data:
        providers_data = toml_data["providers"]
        if isinstance(providers_data, dict):
            for provider_key, provider_data in providers_data.items():
                if isinstance(provider_data, dict):
                    providers_sections[f"providers.{provider_key}"] = provider_data
    

    # If workspace section is missing, check for inline workspace settings in agent section
    if not workspace_section:
        agent_workspace = agent_section.get("workspace", {})
        if isinstance(agent_workspace, dict) and "path" in agent_workspace:
            workspace_section = agent_workspace

    # Validate and extract agent fields
    try:
        name = agent_section["name"]
        if not name or not isinstance(name, str):
            raise ConfigError("Agent name must be a non-empty string")
    except KeyError:
        raise ConfigError("Missing required field: agent.name")
    
    # max_tokens defaults to 8192 if not specified
    max_tokens = agent_section.get("max_tokens", 8192)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ConfigError("max_tokens must be a positive integer")

    # model_max_tokens defaults to 200000 if not specified
    model_max_tokens = agent_section.get("model_max_tokens", 200000)
    if not isinstance(model_max_tokens, int) or model_max_tokens <= 0:
        raise ConfigError("model_max_tokens must be a positive integer")

    # max_iterations defaults to 25 if not specified
    max_iterations = agent_section.get("max_iterations", 50)
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ConfigError("max_iterations must be a positive integer")
    
    # truncation_limit defaults to 50000 if not specified
    truncation_limit = agent_section.get("truncation_limit", 50000)
    if not isinstance(truncation_limit, int) or truncation_limit <= 0:
        raise ConfigError("truncation_limit must be a positive integer")

    # vision defaults to False if not specified
    vision = agent_section.get("vision", False)
    if not isinstance(vision, bool):
        raise ConfigError("vision must be a boolean")

    # thinking defaults to "off" if not specified
    thinking = agent_section.get("thinking", "off")
    valid_thinking = ("off", "low", "medium", "high")
    if thinking not in valid_thinking:
        raise ConfigError(f"thinking must be one of {valid_thinking}, got: {thinking!r}")

    # Validate workspace path
    try:
        workspace_path_str = workspace_section["path"]
        if not workspace_path_str or not isinstance(workspace_path_str, str):
            raise ConfigError("workspace.path must be a non-empty string")
        workspace_path = Path(workspace_path_str)
    except KeyError:
        raise ConfigError("Missing required field: workspace.path")

    # Parse providers configuration — requires [providers.*] sections
    providers: dict[str, ProviderConfig] = {}
    
    if not providers_sections:
        raise ConfigError("Missing [providers.*] section(s). At least one provider must be configured.")

    try:
        default_model = agent_section["default_model"]
        if not default_model or not isinstance(default_model, str):
            raise ConfigError("Agent default_model must be a non-empty string")
    except KeyError:
        raise ConfigError("Missing required field: agent.default_model")
    
    for section_name, section_data in providers_sections.items():
        provider_key = section_name.split(".", 1)[1]
        
        provider_type = section_data.get("type")
        if not provider_type:
            raise ConfigError(f"Missing required field: {section_name}.type")
        if provider_type not in ("anthropic", "openai"):
            raise ConfigError(f"Invalid provider type: {provider_type}. Must be 'anthropic' or 'openai'")
        
        api_key = _resolve_api_key(section_data)
        
        base_url = section_data.get("base_url")
        if provider_type == "openai":
            if base_url is None:
                raise ConfigError(f"base_url is required for {section_name} provider")
            if not isinstance(base_url, str) or not base_url:
                raise ConfigError(f"base_url must be a non-empty string")
        
        quirks = section_data.get("quirks", [])
        if not isinstance(quirks, list):
            quirks = []
        
        providers[provider_key] = ProviderConfig(
            key=provider_key,
            type=provider_type,
            api_key=api_key,
            base_url=base_url,
            quirks=quirks,
        )

    # Parse optional [model_limits] section
    model_limits = {}
    if "model_limits" in toml_data:
        model_limits_section = toml_data["model_limits"]
        if isinstance(model_limits_section, dict):
            for model_name, limit in model_limits_section.items():
                if isinstance(limit, int) and limit > 0:
                    model_limits[model_name] = limit

    # Parse optional [matrix] section
    matrix = _parse_matrix_config(toml_data)

    # Return resolved configuration
    return AgentConfig(
        name=name,
        default_model=default_model,
        max_tokens=max_tokens,
        model_max_tokens=model_max_tokens,
        providers=providers,
        workspace=workspace_path,
        matrix=matrix,
        max_iterations=max_iterations,
        truncation_limit=truncation_limit,
        vision=vision,
        thinking=thinking,
        model_limits=model_limits,
    )


def load_agent_config(name: str) -> AgentConfig:
    """Load configuration for a named agent from the system config directory.

    Resolves /etc/openalph/agents/<name>.toml and loads it via load_config().

    Args:
        name: Agent name (e.g., "watson"). Used as-is for path construction;
              name validation is the caller's responsibility (CLI or admin layer).

    Returns:
        AgentConfig with all fields resolved and validated

    Raises:
        ConfigError: If config file is missing, invalid, or cannot be loaded
    """
    config_path = CONFIG_DIR / f"{name}.toml"
    return load_config(config_path)


def _parse_matrix_config(toml_data: dict) -> MatrixConfig | None:
    """Parse the [matrix] section from TOML data.

    Returns None if [matrix] section is not present.
    Raises ConfigError if required fields are missing or invalid.
    """
    if "matrix" not in toml_data:
        return None

    matrix_section = toml_data["matrix"]

    # Required fields
    try:
        homeserver = matrix_section["homeserver"]
        if not homeserver or not isinstance(homeserver, str):
            raise ConfigError("matrix.homeserver must be a non-empty string")
    except KeyError:
        raise ConfigError("Missing required field: matrix.homeserver")

    try:
        user_id = matrix_section["user_id"]
        if not user_id or not isinstance(user_id, str):
            raise ConfigError("matrix.user_id must be a non-empty string")
    except KeyError:
        raise ConfigError("Missing required field: matrix.user_id")

    # Optional fields with defaults
    device_id = matrix_section.get("device_id", "OPENALPH")
    if not isinstance(device_id, str):
        raise ConfigError("matrix.device_id must be a string")

    context_reserve = matrix_section.get("context_reserve", 16384)
    if not isinstance(context_reserve, int) or context_reserve <= 0:
        raise ConfigError("matrix.context_reserve must be a positive integer")

    # Parse [matrix.sync] subsection with defaults
    sync_section = matrix_section.get("sync", {})
    sync_timeout = sync_section.get("timeout", 30000)
    if not isinstance(sync_timeout, int) or sync_timeout <= 0:
        raise ConfigError("matrix.sync.timeout must be a positive integer")

    retry_base = sync_section.get("retry_base", 5)
    if not isinstance(retry_base, int) or retry_base <= 0:
        raise ConfigError("matrix.sync.retry_base must be a positive integer")

    retry_max = sync_section.get("retry_max", 300)
    if not isinstance(retry_max, int) or retry_max <= 0:
        raise ConfigError("matrix.sync.retry_max must be a positive integer")

    # Resolve password with precedence: password > password_env > password_cmd
    password = None

    if "password" in matrix_section:
        password = matrix_section["password"]
        if not isinstance(password, str):
            raise ConfigError("matrix.password must be a string")

    if password is None and "password_env" in matrix_section:
        env_var_name = matrix_section["password_env"]
        if not env_var_name or not isinstance(env_var_name, str):
            raise ConfigError("matrix.password_env must be a non-empty string")
        password = os.environ.get(env_var_name)
        if password is None:
            raise ConfigError(f"Environment variable {env_var_name} is not set")

    if password is None and "password_cmd" in matrix_section:
        cmd = matrix_section["password_cmd"]
        if not cmd or not isinstance(cmd, str):
            raise ConfigError("matrix.password_cmd must be a non-empty string")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            password = result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise ConfigError("matrix.password_cmd timed out after 10 seconds")
        except subprocess.CalledProcessError as e:
            raise ConfigError(f"matrix.password_cmd failed with exit code {e.returncode}")

    # Resolve access_token with precedence: access_token > access_token_env > access_token_cmd
    access_token = None

    if "access_token" in matrix_section:
        access_token = matrix_section["access_token"]
        if not isinstance(access_token, str):
            raise ConfigError("matrix.access_token must be a string")

    if access_token is None and "access_token_env" in matrix_section:
        env_var_name = matrix_section["access_token_env"]
        if not env_var_name or not isinstance(env_var_name, str):
            raise ConfigError("matrix.access_token_env must be a non-empty string")
        access_token = os.environ.get(env_var_name)
        if access_token is None:
            raise ConfigError(f"Environment variable {env_var_name} is not set")

    if access_token is None and "access_token_cmd" in matrix_section:
        cmd = matrix_section["access_token_cmd"]
        if not cmd or not isinstance(cmd, str):
            raise ConfigError("matrix.access_token_cmd must be a non-empty string")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            access_token = result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise ConfigError("matrix.access_token_cmd timed out after 10 seconds")
        except subprocess.CalledProcessError as e:
            raise ConfigError(f"matrix.access_token_cmd failed with exit code {e.returncode}")

    # Must have either password or access_token
    if password is None and access_token is None:
        raise ConfigError("Matrix authentication required: one of password/password_env/password_cmd or access_token/access_token_env/access_token_cmd must be set")

    # Parse optional [matrix.rooms] section
    rooms_section = matrix_section.get("rooms")
    rooms = None
    if rooms_section is not None and isinstance(rooms_section, dict):
        rooms = dict(rooms_section)  # shallow copy

    return MatrixConfig(
        homeserver=homeserver,
        user_id=user_id,
        device_id=device_id,
        password=password,
        access_token=access_token,
        context_reserve=context_reserve,
        sync_timeout=sync_timeout,
        retry_base=retry_base,
        retry_max=retry_max,
        rooms=rooms
    )
