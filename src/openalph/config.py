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
import math
import os
import subprocess
import tomllib
import logging


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


# System-wide config directory for per-agent TOML files
CONFIG_DIR = Path("/etc/openalph/agents")
logger = logging.getLogger(__name__)


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
    timeout: float = 600.0  # HTTP read timeout in seconds (default matches SDK defaults)
    cache_bust_notices: bool = False  # Emit in-room notice on full prompt cache miss
    subagent_cache_keepalive: bool = False  # Refresh parent prompt cache during long subagent runs (Anthropic only)
    routing: dict | None = None  # OpenRouter provider routing preferences
    degen_detector: str | None = None  # streaming degeneration monitor mode override: off|warn|abort (provider-level takes precedence over agent-level; kdsn.241.21)


@dataclass
class NotificationsConfig:
    """Optional ntfy alerting configuration (kdsn.292, [notifications] section).

    The alerting path must never recreate the bug it reports: ntfy_token is
    fail-soft at config load (a token-cmd failure logs a warning and leaves
    token None; notifications then fire unauthenticated), and the send path
    (openalph.notify) swallows every network error.
    """
    ntfy_url: str
    ntfy_token: str | None = None


@dataclass
class AgentConfig:
    """Configuration for an OpenAlph agent."""
    name: str
    default_model: str
    max_tokens: int
    providers: dict[str, ProviderConfig]
    workspace: Path
    model_max_tokens: int = 200000
    user_id: str | None = None  # kdsn.237 Phase 1: CLI session identity independent of matrix.user_id
    matrix: MatrixConfig | None = None
    max_iterations: int = 100
    truncation_limit: int = 50000
    # Per-turn stall watchdog: cancel a turn that has made no room-observable
    # progress for this many seconds. 0 disables the watchdog entirely.
    # Guards against the provider-retry wedge (RCA 2026-08-03): the SDK retry
    # loop holds both per-room locks with no socket, no room output and no log
    # above DEBUG, so nothing raises and the room silently looks dead.
    turn_stall_timeout_seconds: float = 900
    thinking: str = "off"
    temperature: float | None = None
    top_p: float | None = None
    degen_detector: str = "off"  # streaming degeneration monitor mode: off|warn|abort (default off per Phase 1 code-audit -- H1/H2/H3 false-positive/truncation findings, workspace-kdsn.241.4 remediation pending)
    reminders: bool = True
    injection_defense: bool = True
    model_limits: dict[str, int] = field(default_factory=dict)
    # kdsn.275: vision capability is a MODEL-LEVEL property — [model_vision]
    # holds per-model overrides ("provider/api-model" = true|false) consulted
    # by provider.model_supports_vision BEFORE the curated table. The .25
    # agent-level `vision: bool` flag was hard-cut (no alias, no tombstone).
    model_vision: dict[str, bool] = field(default_factory=dict)
    model_aliases: dict[str, str] = field(default_factory=dict)
    # kdsn.292: providers that failed startup validation/key resolution are
    # held here (provider_key -> human-readable reason) instead of killing
    # the load. Empty == fully healthy start. Do NOT rename — several
    # call sites poke it defensively via getattr(config, "skipped_providers", {}).
    skipped_providers: dict[str, str] = field(default_factory=dict)
    notifications: NotificationsConfig | None = None
    # Spotter v1 (spotter-v1-design.md §1): an independent monitor that
    # watches this agent's live session, turn by turn. Optional [spotter]
    # TOML section; absent section → all defaults (enabled=True per D6
    # single knob). This is a CONFIDENTIALITY knob — [spotter] parsing
    # fails LOUD (ConfigError on bad type/value), the same discipline as
    # [model_vision]: a silently-dropped setting would be fail-open on a
    # setting that controls what leaves the process.
    spotter_enabled: bool = True
    spotter_model: str = "synglm53"
    spotter_thinking: str = "off"
    spotter_max_iterations: int = 8
    spotter_disabled_rooms: list[str] = field(default_factory=list)


def resolve_model(
    model_str: str,
    providers: dict[str, ProviderConfig],
    aliases: dict[str, str] | None = None,
) -> tuple[ProviderConfig, str]:
    """Returns (provider_config, api_model_name).

    If model_str contains no "/" and aliases is provided, look up alias first.
    Otherwise model_str MUST be fully qualified: "<provider_key>/<api_model_name>".
    Split on the first "/". The prefix must match a key in providers.
    Everything after the first "/" is the API model name (may contain more slashes).
    """
    if not model_str:
        raise ValueError("Model string cannot be empty")

    # Alias expansion: bare name (no "/") with aliases dict
    if "/" not in model_str and aliases:
        if model_str in aliases:
            model_str = aliases[model_str]
        else:
            available = ", ".join(sorted(aliases.keys()))
            raise ValueError(
                f"Unknown model alias '{model_str}'. "
                f"Available aliases: {available}"
            )

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


def _resolve_secret(section: dict, base_key: str, label: str, *, required: bool = True) -> str | None:
    """Resolve a secret from `<base>`, `<base>_env`, or `<base>_cmd` (in that
    precedence), with ONE consistent set of validation rules.

    ARCH-2: this logic existed three times -- `_resolve_api_key`, and the
    hand-rolled `password` and `access_token` ladders in matrix parsing -- with
    divergent validation. `_resolve_api_key` rejected empty strings and empty
    command output; the other two did not, so `password = ""` passed config
    load and failed later at Matrix login (a confusing, far-from-source error)
    instead of failing loudly here. Consolidated so a fix is written once and
    every secret is validated identically: a present-but-empty value, an empty
    env var, or empty command output is an error.

    Returns the resolved secret, or None when `required` is False and no source
    is configured (callers that accept one of several auth methods).
    """
    direct_key = base_key
    env_key = f"{base_key}_env"
    cmd_key = f"{base_key}_cmd"

    if direct_key in section:
        value = section[direct_key]
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{label} must be a non-empty string")
        return value

    if env_key in section:
        env_var_name = section[env_key]
        if not env_var_name or not isinstance(env_var_name, str):
            raise ConfigError(f"{label}_env must be a non-empty string")
        value = os.environ.get(env_var_name)
        if value is None:
            raise ConfigError(f"Environment variable {env_var_name} is not set")
        if not value:
            raise ConfigError(f"Environment variable {env_var_name} (for {label}) is empty")
        return value

    if cmd_key in section:
        cmd = section[cmd_key]
        if not cmd or not isinstance(cmd, str):
            raise ConfigError(f"{label}_cmd must be a non-empty string")
        try:
            result = subprocess.run(
                cmd, shell=True, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            raise ConfigError(f"{label}_cmd timed out after 10 seconds")
        except subprocess.CalledProcessError as e:
            raise ConfigError(f"{label}_cmd failed with exit code {e.returncode}")
        value = result.stdout.strip()
        if not value:
            raise ConfigError(f"{label}_cmd produced empty output")
        return value

    if required:
        raise ConfigError(
            f"No {label} provided. One of {base_key}, {env_key}, or {cmd_key} must be set"
        )
    return None


def _resolve_api_key(section: dict) -> str:
    """Resolve API key: api_key > api_key_env > api_key_cmd. See _resolve_secret."""
    return _resolve_secret(section, "api_key", "api_key", required=True)


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

    # ARCH-6: the old inline "[provider]" format is no longer supported (only
    # [providers.<key>] sections are), so the previously-read `provider_section`
    # was dead and its comment stale. Removed.
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

    # max_iterations defaults to 100 if not specified
    max_iterations = agent_section.get("max_iterations", 100)
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ConfigError("max_iterations must be a positive integer")
    
    # truncation_limit defaults to 50000 if not specified
    truncation_limit = agent_section.get("truncation_limit", 50000)
    if not isinstance(truncation_limit, int) or truncation_limit <= 0:
        raise ConfigError("truncation_limit must be a positive integer")

    # turn_stall_timeout_seconds defaults to 900 (15 min) if not specified.
    # 0 is legal and disables the stall watchdog (unlike max_iterations, where
    # 0 is meaningless), so the check is non-negative rather than positive.
    #
    # F5 (round-2 review): math.isfinite is REQUIRED, not decorative. TOML has
    # first-class `nan` / `inf` / `-inf` float literals, and neither compares
    # less than zero, so a naive `< 0` check accepts all three:
    #   nan  -> `nan > 0` is also False, so the watchdog is silently DISABLED
    #           while the operator believes a finite timeout is in force;
    #   inf  -> arms a watchdog whose `idle > timeout` can never be true;
    #   -inf -> `-inf < 0` actually does reject, but only by luck of ordering.
    # All three violate the documented finite-timeout semantics, so reject any
    # non-finite value outright.
    turn_stall_timeout_seconds = agent_section.get("turn_stall_timeout_seconds", 900)
    if (isinstance(turn_stall_timeout_seconds, bool)
            or not isinstance(turn_stall_timeout_seconds, (int, float))
            or not math.isfinite(turn_stall_timeout_seconds)
            or turn_stall_timeout_seconds < 0):
        raise ConfigError(
            "turn_stall_timeout_seconds must be a finite non-negative number")
    turn_stall_timeout_seconds = float(turn_stall_timeout_seconds)

    # reminders defaults to True if not specified
    reminders = agent_section.get("reminders", True)
    if not isinstance(reminders, bool):
        raise ConfigError("reminders must be a boolean")

    # PHIL-1: injection_defense defaults to True -- the security footer is
    # appended unless the operator turns it off. Documented and greppable,
    # where before it was an unconditional hardcoded string.
    injection_defense = agent_section.get("injection_defense", True)
    if not isinstance(injection_defense, bool):
        raise ConfigError("injection_defense must be a boolean")

    # thinking defaults to "off" if not specified
    thinking = agent_section.get("thinking", "off")
    valid_thinking = ("off", "low", "medium", "high", "xhigh", "max")
    if thinking not in valid_thinking:
        raise ConfigError(f"thinking must be one of {valid_thinking}, got: {thinking!r}")

    # temperature defaults to None (provider default) if not specified
    temperature = agent_section.get("temperature")
    if temperature is not None:
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            raise ConfigError("temperature must be a number between 0 and 2")
        temperature = float(temperature)

    # top_p defaults to None (provider default) if not specified
    top_p = agent_section.get("top_p")
    if top_p is not None:
        if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
            raise ConfigError("top_p must be a number between 0 and 1")
        top_p = float(top_p)

    # degen_detector: streaming degeneration monitor mode (kdsn.241.4).
    # off = disabled (default -- Phase 1 code-audit flagged warn-mode false
    # positives contaminating calibration telemetry, workspace-kdsn.241.4
    # remediation pending); warn = log trips, never modify output;
    # abort = tear down the stream mid-generation on a trip (DO NOT ARM --
    # audit found a truncation-position bug that destroys legitimate output).
    degen_detector = agent_section.get("degen_detector", "off")
    valid_degen = ("off", "warn", "abort")
    if degen_detector not in valid_degen:
        raise ConfigError(f"degen_detector must be one of {valid_degen}, got: {degen_detector!r}")

    # Validate workspace path
    try:
        workspace_path_str = workspace_section["path"]
        if not workspace_path_str or not isinstance(workspace_path_str, str):
            raise ConfigError("workspace.path must be a non-empty string")
        workspace_path = Path(workspace_path_str)
        if not workspace_path.is_dir():
            raise ConfigError(f"workspace.path does not exist or is not a directory: {workspace_path}")
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
    
    # kdsn.292 (SB ruling 2026-08-26): ANY per-provider problem — authoring
    # error (bad type/base_url/timeout/flags) or secret-resolution failure —
    # skips that provider WITH REASON into skipped_map. A stray config must
    # NEVER be ConfigError at load: exit 1 under Restart=on-failure crash-loops
    # and any `*_cmd` source re-hammers 1Password fleet-wide every cycle.
    # Resolution reasons carry _resolve_secret text verbatim (per spec §2);
    # authoring reasons constructed here name the field ("type", "base_url", …).
    skipped_map: dict[str, str] = {}

    def _skip_provider(provider_key: str, reason: str) -> None:
        logger.warning("Skipping provider '%s': %s", provider_key, reason)
        skipped_map[provider_key] = reason

    for section_name, section_data in providers_sections.items():
        provider_key = section_name.split(".", 1)[1]
        
        provider_type = section_data.get("type")
        if not provider_type:
            _skip_provider(provider_key, f"missing required field: {section_name}.type")
            continue
        if provider_type not in ("anthropic", "openai"):
            _skip_provider(provider_key, f"invalid provider type: {provider_type}. Must be 'anthropic' or 'openai'")
            continue

        try:
            api_key = _resolve_api_key(section_data)
        except ConfigError as e:
            _skip_provider(provider_key, str(e))
            continue

        base_url = section_data.get("base_url")
        if provider_type == "openai":
            if base_url is None:
                _skip_provider(provider_key, f"base_url is required for {section_name} (openai-compatible) provider")
                continue
            if not isinstance(base_url, str) or not base_url:
                _skip_provider(provider_key, "base_url must be a non-empty string")
                continue
        
        quirks = section_data.get("quirks", [])
        if not isinstance(quirks, list):
            quirks = []
        
        timeout = section_data.get("timeout", 600.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            _skip_provider(provider_key, f"timeout must be a positive number, got: {timeout!r}")
            continue
        timeout = float(timeout)

        cache_bust_notices = section_data.get("cache_bust_notices", False)
        if not isinstance(cache_bust_notices, bool):
            _skip_provider(provider_key, "cache_bust_notices must be a boolean")
            continue

        subagent_cache_keepalive = section_data.get("subagent_cache_keepalive", False)
        if not isinstance(subagent_cache_keepalive, bool):
            _skip_provider(provider_key, "subagent_cache_keepalive must be a boolean")
            continue

        routing = section_data.get("routing")
        if routing is not None and not isinstance(routing, dict):
            _skip_provider(provider_key, f"routing must be a table/dict, got: {type(routing).__name__}")
            continue

        provider_degen = section_data.get("degen_detector")
        if provider_degen is not None:
            if provider_degen not in ("off", "warn", "abort"):
                _skip_provider(provider_key, f"degen_detector must be one of ('off', 'warn', 'abort'), got: {provider_degen!r}")
                continue

        providers[provider_key] = ProviderConfig(
            key=provider_key,
            type=provider_type,
            api_key=api_key,
            base_url=base_url,
            quirks=quirks,
            timeout=timeout,
            cache_bust_notices=cache_bust_notices,
            subagent_cache_keepalive=subagent_cache_keepalive,
            routing=routing,
            degen_detector=provider_degen,
        )

    # kdsn.292: zero providers is DEGRADED, not fatal — slash commands, the
    # Matrix sync loop and local tools all function; every LLM invocation
    # fails loudly via provider.ProviderUnavailableError (pre-network).
    if not providers and skipped_map:
        logger.error(
            "DEGRADED START: no providers loaded — agent will start degraded; skipped: %s",
            ", ".join(f"{k} ({v})" for k, v in skipped_map.items()),
        )

    # Log provider status
    active = ", ".join(sorted(providers.keys()))
    if skipped_map:
        skipped = ", ".join(skipped_map.keys())
        logger.info("Providers loaded: %s (skipped: %s)", active, skipped)
    else:
        logger.info("Providers loaded: %s", active)

    # Parse optional [model_aliases] section (needed for default_model validation)
    model_aliases = {}
    if "model_aliases" in toml_data:
        aliases_section = toml_data["model_aliases"]
        if isinstance(aliases_section, dict):
            for alias, target in aliases_section.items():
                if isinstance(target, str):
                    model_aliases[alias] = target

    # Validate default_model references a configured provider
    # Resolve through aliases if default_model is a bare name
    resolved_default = default_model
    if "/" not in default_model and default_model in model_aliases:
        resolved_default = model_aliases[default_model]

    if "/" in resolved_default:
        provider_prefix = resolved_default.split("/", 1)[0]
        if provider_prefix not in providers:
            # kdsn.292: degraded, loud, never fatal (see SB ruling). If the
            # prefix was never even declared in [providers.*], record the
            # authoring-vs-resolution distinguishing reason.
            if provider_prefix not in skipped_map:
                skipped_map[provider_prefix] = "not configured in [providers.*]"
            logger.error(
                "DEGRADED START: default_model '%s' provider '%s' unavailable (%s)",
                default_model, provider_prefix, skipped_map[provider_prefix],
            )

    # Parse optional [model_limits] section
    model_limits = {}
    if "model_limits" in toml_data:
        model_limits_section = toml_data["model_limits"]
        if isinstance(model_limits_section, dict):
            for model_name, limit in model_limits_section.items():
                if isinstance(limit, int) and limit > 0:
                    model_limits[model_name] = limit

    # Parse optional [model_vision] section (kdsn.275). DELIBERATE deviation
    # from [model_limits]' lenient skip: a non-bool value raises ConfigError.
    # Fail-LOUD here — a silently-dropped `= false` disable override would be
    # fail-OPEN on a safety knob (images would flow to a model the operator
    # explicitly marked blind).
    model_vision = {}
    if "model_vision" in toml_data:
        model_vision_section = toml_data["model_vision"]
        if isinstance(model_vision_section, dict):
            for model_name, vis in model_vision_section.items():
                if not isinstance(vis, bool):
                    raise ConfigError(
                        f"model_vision[{model_name!r}] must be a boolean, "
                        f"got {vis!r} ({type(vis).__name__})")
                model_vision[model_name] = vis

    # Parse optional [spotter] section (Spotter v1, spotter-v1-design.md §1).
    # DELIBERATE deviation from [model_limits]' lenient skip: every bad type
    # or value RAISES ConfigError. Fail-LOUD — the spotter settings control
    # what leaves the process (an external model reads this agent's full
    # transcript), so a silently-dropped `enabled = false` or a mistyped
    # `disabled_rooms` would be fail-open on a confidentiality knob.
    # Absent section → all defaults (spotter_enabled=True per D6 single knob).
    spotter_enabled = True
    spotter_model = "synglm53"
    spotter_thinking = "off"
    spotter_max_iterations = 8
    spotter_disabled_rooms: list[str] = []
    if "spotter" in toml_data:
        spotter_section = toml_data["spotter"]
        if not isinstance(spotter_section, dict):
            raise ConfigError("[spotter] section must be a table")
        if "enabled" in spotter_section:
            if not isinstance(spotter_section["enabled"], bool):
                raise ConfigError(
                    f"[spotter] enabled must be a boolean, "
                    f"got {spotter_section['enabled']!r} "
                    f"({type(spotter_section['enabled']).__name__})")
            spotter_enabled = spotter_section["enabled"]
        if "model" in spotter_section:
            m = spotter_section["model"]
            if not isinstance(m, str) or not m:
                raise ConfigError(
                    f"[spotter] model must be a non-empty string, got {m!r}")
            spotter_model = m
        if "thinking" in spotter_section:
            th = spotter_section["thinking"]
            valid_spotter_thinking = ("off", "low", "medium", "high")
            # xhigh/max are NOT valid for the spotter (D1: qwen38/synthetic
            # mapping surprises — a future lever, not a v1 knob).
            if not isinstance(th, str) or th not in valid_spotter_thinking:
                raise ConfigError(
                    f"[spotter] thinking must be one of {valid_spotter_thinking}, "
                    f"got {th!r}")
            spotter_thinking = th
        if "max_iterations" in spotter_section:
            mi = spotter_section["max_iterations"]
            # bool is an int subclass in Python — reject it explicitly.
            if isinstance(mi, bool) or not isinstance(mi, int) or mi <= 0:
                raise ConfigError(
                    f"[spotter] max_iterations must be a positive integer, "
                    f"got {mi!r}")
            spotter_max_iterations = mi
        if "disabled_rooms" in spotter_section:
            dr = spotter_section["disabled_rooms"]
            if not isinstance(dr, list) or not all(
                    isinstance(r, str) and r for r in dr):
                raise ConfigError(
                    "[spotter] disabled_rooms must be a list of non-empty "
                    f"strings, got {dr!r}")
            spotter_disabled_rooms = list(dr)

    # Parse optional [matrix] section
    matrix = _parse_matrix_config(toml_data)

    # Parse optional [notifications] section (kdsn.292: ntfy degraded-start alert)
    notifications = _parse_notifications_config(toml_data)

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
        turn_stall_timeout_seconds=turn_stall_timeout_seconds,
        reminders=reminders,
        injection_defense=injection_defense,
        thinking=thinking,
        temperature=temperature,
        top_p=top_p,
        degen_detector=degen_detector,
        model_limits=model_limits,
        model_vision=model_vision,
        model_aliases=model_aliases,
        skipped_providers=skipped_map,
        notifications=notifications,
        spotter_enabled=spotter_enabled,
        spotter_model=spotter_model,
        spotter_thinking=spotter_thinking,
        spotter_max_iterations=spotter_max_iterations,
        spotter_disabled_rooms=spotter_disabled_rooms,
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


def _parse_notifications_config(toml_data: dict) -> NotificationsConfig | None:
    """Parse the optional [notifications] section (kdsn.292).

    ntfy_url is the section's only mandatory field: invalid → ConfigError
    (structural authoring error, never fires secret resolution).
    ntfy_token resolution is FAIL-SOFT: a token-cmd failure can never be
    allowed to crash an agent that is otherwise alive — the alerting path
    must never recreate the bug it reports — so it warns and fires
    unauthenticated instead of raising.
    """
    if "notifications" not in toml_data:
        return None

    section = toml_data["notifications"]
    if not isinstance(section, dict):
        raise ConfigError("[notifications] section must be a table")

    ntfy_url = section.get("ntfy_url")
    if not isinstance(ntfy_url, str) or not ntfy_url:
        raise ConfigError("notifications.ntfy_url must be a non-empty string")

    try:
        ntfy_token = _resolve_secret(section, "ntfy_token", "ntfy_token", required=False)
    except ConfigError as e:
        logger.warning("ntfy_token resolution failed (fail-soft): %s", e)
        ntfy_token = None

    return NotificationsConfig(ntfy_url=ntfy_url, ntfy_token=ntfy_token)


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

    # ARCH-2: password and access_token now use the SAME resolver as api_key,
    # so `matrix.password = ""` is rejected at load time instead of failing
    # later at Matrix login. Each is optional on its own (one of the two auth
    # methods is required, checked just below).
    password = _resolve_secret(matrix_section, "password", "matrix.password", required=False)
    access_token = _resolve_secret(matrix_section, "access_token", "matrix.access_token", required=False)

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
