"""Configuration loader for OpenAlph agents.

This module loads agent configuration from TOML files and resolves API keys
from multiple sources (direct value, environment variable, or shell command).

Design decisions:
- Use tomllib from stdlib (Python 3.11+) to avoid external dependencies
- API key resolution follows precedence: direct > env var > command
- All validation errors raise ConfigError with descriptive messages
- Workspace path is converted to Path object for consistency
"""

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import tomllib


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


@dataclass
class AgentConfig:
    """Configuration for an OpenAlph agent."""
    name: str
    model: str
    max_tokens: int
    provider: str  # "anthropic" or "openai"
    api_key: str  # resolved value (not the reference)
    base_url: str | None
    workspace: Path
    max_iterations: int = 25
    truncation_limit: int = 50000


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
        provider_section = toml_data["provider"]
        workspace_section = toml_data["workspace"]
    except KeyError as e:
        raise ConfigError(f"Missing required section: {e}")
    
    # Validate and extract agent fields
    try:
        name = agent_section["name"]
        if not name or not isinstance(name, str):
            raise ConfigError("Agent name must be a non-empty string")
    except KeyError:
        raise ConfigError("Missing required field: agent.name")
    
    try:
        model = agent_section["model"]
        if not model or not isinstance(model, str):
            raise ConfigError("Agent model must be a non-empty string")
    except KeyError:
        raise ConfigError("Missing required field: agent.model")
    
    # max_tokens defaults to 8192 if not specified
    max_tokens = agent_section.get("max_tokens", 8192)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ConfigError("max_tokens must be a positive integer")
    
    # max_iterations defaults to 25 if not specified
    max_iterations = agent_section.get("max_iterations", 25)
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ConfigError("max_iterations must be a positive integer")
    
    # truncation_limit defaults to 50000 if not specified
    truncation_limit = agent_section.get("truncation_limit", 50000)
    if not isinstance(truncation_limit, int) or truncation_limit <= 0:
        raise ConfigError("truncation_limit must be a positive integer")
    
    # Validate and extract provider fields
    try:
        provider_type = provider_section["type"]
    except KeyError:
        raise ConfigError("Missing required field: provider.type")
    
    if provider_type not in ("anthropic", "openai"):
        raise ConfigError(f"Invalid provider type: {provider_type}. Must be 'anthropic' or 'openai'")
    
    # Resolve API key with precedence: api_key > api_key_env > api_key_cmd
    api_key = None
    
    # Try direct api_key first
    if "api_key" in provider_section:
        api_key = provider_section["api_key"]
        if not api_key or not isinstance(api_key, str):
            raise ConfigError("api_key must be a non-empty string")
    
    # Try api_key_env if no direct key
    if api_key is None and "api_key_env" in provider_section:
        env_var_name = provider_section["api_key_env"]
        if not env_var_name or not isinstance(env_var_name, str):
            raise ConfigError("api_key_env must be a non-empty string")
        
        api_key = os.environ.get(env_var_name)
        if api_key is None:
            raise ConfigError(f"Environment variable {env_var_name} is not set")
    
    # Try api_key_cmd if no key yet
    if api_key is None and "api_key_cmd" in provider_section:
        cmd = provider_section["api_key_cmd"]
        if not cmd or not isinstance(cmd, str):
            raise ConfigError("api_key_cmd must be a non-empty string")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            api_key = result.stdout.strip()
            if not api_key:
                raise ConfigError("api_key_cmd produced empty output")
        except subprocess.CalledProcessError as e:
            raise ConfigError(f"api_key_cmd failed with exit code {e.returncode}")
    
    # Final API key validation
    if api_key is None:
        raise ConfigError("No API key provided. One of api_key, api_key_env, or api_key_cmd must be set")
    
    # Validate base_url for openai provider
    base_url = provider_section.get("base_url")
    if provider_type == "openai":
        if base_url is None:
            raise ConfigError("base_url is required for openai provider")
        if not isinstance(base_url, str) or not base_url:
            raise ConfigError("base_url must be a non-empty string")
    
    # Validate workspace path
    try:
        workspace_path_str = workspace_section["path"]
        if not workspace_path_str or not isinstance(workspace_path_str, str):
            raise ConfigError("workspace.path must be a non-empty string")
        workspace_path = Path(workspace_path_str)
    except KeyError:
        raise ConfigError("Missing required field: workspace.path")
    
    # Return resolved configuration
    return AgentConfig(
        name=name,
        model=model,
        max_tokens=max_tokens,
        provider=provider_type,
        api_key=api_key,
        base_url=base_url,
        workspace=workspace_path,
        max_iterations=max_iterations,
        truncation_limit=truncation_limit
    )
