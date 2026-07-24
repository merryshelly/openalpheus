"""
OpenAlph LLM Provider Adapter

Routes completion requests to either Anthropic or OpenAI SDKs based on configuration.
The adapter handles the differences in API shapes and response formats between providers.
"""

import copy
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import AsyncGenerator
import json
import re
import anthropic
import httpx
import openai
from openalph.config import AgentConfig, ProviderConfig, resolve_model
from openalph.degen import DegenerationMonitor

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """User-surfaceable error from an LLM provider.

    Wraps SDK-specific exceptions with a sanitized message
    safe to display in chat.
    """
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


_API_KEY_PATTERN = re.compile(r'\b(sk-[a-zA-Z0-9_-]{10,})\b')

# ---------------------------------------------------------------------------
# Degeneration detection
# ---------------------------------------------------------------------------
# Minimum consecutive identical characters to trigger post-hoc truncation.
# Raised 50 -> 4500 (kdsn.241.4): 50 false-fired on legitimate bounded runs
# (pytest dot output, separator lines). True single-token spam runs to the
# token cap (tens of thousands); 4500 clears the largest legitimate run while
# still catching pathological spam. Word/phrase and compression-based loops are
# now caught mid-stream by the DegenerationMonitor (openalph.degen), for which
# this remains an aligned post-hoc backstop on the final accumulated text.
_DEGEN_CHAR_THRESHOLD = 4500
_DEGEN_WARNING = (
    "\n\n⚠️ *Output truncated — repetition collapse detected. "
    "Session context may be degraded; consider starting a new session (`/new`).*"
)


def _detect_and_truncate_degeneration(text: str) -> tuple[str, bool]:
    """Detect degenerate repetition in model output.

    Checks for runs of identical characters >= _DEGEN_CHAR_THRESHOLD.
    When found, truncates at the start of the degenerate run and appends
    a warning.

    Returns (possibly_truncated_text, was_degenerate).
    """
    if not text or len(text) < _DEGEN_CHAR_THRESHOLD:
        return text, False

    run_start = 0
    run_len = 1

    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            run_len += 1
            if run_len >= _DEGEN_CHAR_THRESHOLD:
                truncated = text[:run_start].rstrip()
                if not truncated:
                    # Entire output is degenerate
                    return _DEGEN_WARNING.lstrip(), True
                return truncated + _DEGEN_WARNING, True
        else:
            run_start = i
            run_len = 1

    return text, False


def _sanitize_error(message: str) -> str:
    """Extract clean error message and strip sensitive data.

    SDK error messages come as 'Error code: NNN - {body_dict}'.
    Extract just the meaningful error text from the body when possible.
    Always strip API key patterns as a safety net.
    """
    # Try to extract the error message from SDK's formatted string
    # Format: "Error code: 400 - {'error': {'message': '...', ...}, ...}"
    import ast
    if " - " in message and message.startswith("Error code:"):
        _, _, body_str = message.partition(" - ")
        try:
            body = ast.literal_eval(body_str.strip())
            if isinstance(body, dict):
                # OpenRouter/OpenAI: {"error": {"message": "..."}}
                err = body.get("error", {})
                if isinstance(err, dict) and "message" in err:
                    message = err["message"]
                # Anthropic: {"error": {"message": "..."}} or {"message": "..."}
                elif "message" in body:
                    message = body["message"]
        except (ValueError, SyntaxError):
            pass  # Keep original message if parsing fails
    return _API_KEY_PATTERN.sub('[REDACTED]', message)

# Client cache: reuse HTTP clients for connection pooling.
# Keyed by (provider, api_key, base_url) so different configs get different clients.
_client_cache: dict[tuple, object] = {}

# SDK-level automatic retry budget for transient failures (408/409/429/5xx incl
# Anthropic 529 overload), applied at request AND stream-establishment time. The
# SDK default is 2 -- every 529 was already retried twice invisibly. Bump to 20
# (kdsn.220) to ride out transient overload blips. SDK backoff is 0.5->1->2->4->8s
# then capped 8s/attempt (honors Retry-After up to 60s), so 20 retries is ~2.25min
# worst case; a blip outlasting that is a sustained outage, not a blip -> fail the
# turn and let the next message/heartbeat pick it up. The turn holds its per-room
# asyncio lock for the whole retry window, which is why this is capped at 20 and
# not higher. /stop stays responsive: CancelledError propagates through the
# backoff sleep. Mid-stream failures cannot be resumed (accepted); history is
# unchanged since the assistant turn is appended only after the stream completes.
_MAX_SDK_RETRIES = 20


def _get_client(provider: ProviderConfig):
    """Get or create a cached provider client."""
    timeout = getattr(provider, "timeout", 600.0)
    if provider.type == "anthropic":
        key = ("anthropic", provider.api_key, None, timeout)
        if key not in _client_cache:
            _client_cache[key] = anthropic.AsyncAnthropic(
                api_key=provider.api_key,
                timeout=httpx.Timeout(timeout, connect=10.0),
                max_retries=_MAX_SDK_RETRIES,
            )
        return _client_cache[key]
    elif provider.type == "openai":
        key = ("openai", provider.api_key, provider.base_url, timeout)
        if key not in _client_cache:
            _client_cache[key] = openai.AsyncOpenAI(
                api_key=provider.api_key,
                base_url=provider.base_url,
                timeout=httpx.Timeout(timeout, connect=10.0),
                max_retries=_MAX_SDK_RETRIES,
            )
        return _client_cache[key]
    else:
        raise ValueError(f"Unsupported provider: {provider.type}")


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    # TTL-split cache-write breakdown (Anthropic only; None when the response
    # carries no cache_creation object or the provider doesn't surface a split).
    # Invariant when both present: 5m + 1h == cache_creation_tokens.
    cache_creation_5m_tokens: int | None = None
    cache_creation_1h_tokens: int | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict
    # Opaque provider-specific metadata that must be echoed back verbatim on
    # the next turn for multi-turn tool use to work correctly (e.g. Google
    # Gemini 3.x's `extra_content.google.thought_signature` — see
    # https://ai.google.dev/gemini-api/docs/thought-signatures and bead
    # workspace-kdsn.186.18). None for every other provider today. Anthropic
    # has its own, separate signature mechanism via ThinkingBlock.signature;
    # this field is not used for Anthropic.
    extra_content: dict | None = None

    def __post_init__(self):
        # Models occasionally emit tool names with leading/trailing whitespace
        self.name = self.name.strip()


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str


@dataclass
class Response:
    content: str
    model: str = ""
    usage: Usage = None
    stop_reason: str = ""
    tool_calls: list[ToolCall] = None
    thinking: list[ThinkingBlock] = None
    degenerate: bool = False
    generation_id: str = ""  # Provider-assigned ID (OpenRouter gen ID, Anthropic msg ID)

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.thinking is None:
            self.thinking = []
        if self.usage is None:
            self.usage = Usage(input_tokens=0, output_tokens=0)


@dataclass
class StreamEvent:
    """A single event from a streaming LLM response."""
    type: str  # "text", "thinking", "signature", "tool_start", "tool_delta", "tool_done", "usage", "done"
    content: str = ""
    tool_index: int = 0
    tool_id: str = ""
    tool_name: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    stop_reason: str = ""
    model: str = ""
    response: Response | None = None


def _supports_adaptive_thinking(model_id: str) -> bool:
    """Returns True for models that support adaptive thinking (type=adaptive + effort).

    Supported: opus-4-5, opus-4-6, opus-4-7, opus-4-8, opus-5, sonnet-4-6, sonnet-5,
    mythos, fable.

    "opus-5" is a distinct fragment from "opus-4-5"/etc (no "-4-" substring in
    common), so it cannot collide with the existing opus-4.x entries -- verified
    via substring regression tests in test_opus5_upgrade.py. Without this entry,
    claude-opus-5 would silently fall through to the legacy budget_tokens branch
    instead of adaptive+effort (the exact bug class test_sonnet5_upgrade.py exists
    to prevent).

    Case-insensitive (matches model_context_window / _model_output_cap).
    """
    model_id = model_id.lower()
    return (
        "opus-4-5" in model_id
        or "opus-4-6" in model_id
        or "sonnet-4-6" in model_id
        or "sonnet-5" in model_id
        or "opus-4-7" in model_id
        or "opus-4-8" in model_id
        or "opus-5" in model_id
        or "mythos" in model_id
        or "fable" in model_id
    )


def _supports_sampling_params(model_id: str) -> bool:
    """Returns True for Anthropic models that accept temperature/top_p/top_k.

    Modern Anthropic models removed sampling params: Opus 4.7, Opus 4.8, Sonnet 5,
    Fable, and Mythos 400 (or silently ignore) when temperature/top_p/top_k are
    sent. Only older releases still accept them.

    Fail-closed ALLOWLIST: the 4.x families are enumerated PER-MINOR, so a future
    in-family minor that drops sampling (as Opus did at 4.6 -> 4.7) is NOT
    auto-accepted; only the frozen legacy claude-3.x family is matched broadly.
    Unknown/future models default to False, so we never send a param that 400s (a
    dropped param on a model that would have accepted it is merely ignored, never
    an error). Case-insensitive, matching model_context_window/_model_output_cap.

    Verified empirically 2026-07-04 (live API, oa-babson key): claude-sonnet-5
    rejects the legacy sampling params. Folds in the Opus 4.7 guard (kdsn.134).
    """
    model_id = model_id.lower()
    return (
        "claude-3" in model_id       # frozen legacy family (3.x sonnet/opus/haiku)
        or "opus-4-5" in model_id
        or "opus-4-6" in model_id
        or "sonnet-4-6" in model_id
        or "haiku-4-5" in model_id
    )


def _thinking_effort(level: str) -> str:
    """Map config thinking level to Anthropic effort value.

    Direct 1:1 mapping: low→low, medium→medium, high→high, xhigh→xhigh, max→max.
    """
    return level  # Direct mapping


def _thinking_budget(level: str, base_max_tokens: int, model_max_tokens: int) -> tuple[int, int]:
    """Budget-based thinking for older models.
    Returns (budget_tokens, adjusted_max_tokens).
    Budget mapping: low=2048, medium=8192, high=16384.
    max_tokens = min(base + budget, model_max_tokens).
    If clamped, reduce budget to leave at least 1024 for output.
    """
    budgets = {"low": 2048, "medium": 8192, "high": 16384}
    budget = budgets.get(level, 16384)
    max_tokens = min(base_max_tokens + budget, model_max_tokens)
    if max_tokens <= budget:
        budget = max(0, max_tokens - 1024)
    return budget, max_tokens


# (context_window, output_cap)  — output_cap None = no clamp (Fireworks/local tolerate)
_MODEL_CAPABILITIES: list[tuple[str, int | None, int | None]] = [
    # Anthropic
    ("haiku-4-5",  200_000,   64_000),
    ("sonnet-4-6", 200_000,  128_000),
    ("sonnet-5",  1_048_576, 128_000),
    ("opus-4-6",  1_048_576, 128_000),
    ("opus-4-7",  1_048_576, 128_000),
    ("opus-4-8",  1_048_576, 128_000),
    ("opus-5",    1_048_576, 128_000),
    ("fable",     1_048_576, 128_000),
    # Fireworks / open
    ("glm-5p2",   1_048_576, None),
    ("kimi-k2p6",   262_144, None),
    # Local
    ("qwen3.5",     262_144, None),
    ("qwen3p5",     262_144, None),
    ("qwen3.6",     262_144, None),
    ("qwen3p6",     262_144, None),
    # Others (window only)
    ("maverick",  1_048_576, None),
    ("hermes",      131_072, None),
    ("gemini",    1_048_576, None),
]


def model_context_window(api_model: str) -> int | None:
    """Curated default context window for a model, or None if unknown."""
    m = api_model.lower()
    for frag, window, _cap in _MODEL_CAPABILITIES:
        if frag in m:
            return window
    return None


def _model_output_cap(api_model: str) -> int | None:
    """Maximum output tokens (max_tokens) a model's API will accept.

    Anthropic hard-400s when max_tokens exceeds the model's cap (verified
    2026-06-29: Haiku 4.5 = 64000). Fireworks / local OpenAI-compatible servers
    tolerate over-cap values (verified: glm-5p2 and kimi-k2p6 accepted
    max_tokens=200000), so they return None (no clamp needed).

    Returns the cap in tokens, or None if unknown / no clamp required.
    """
    m = api_model.lower()
    for frag, _window, cap in _MODEL_CAPABILITIES:
        if frag in m:
            return cap
    return None


@dataclass(frozen=True)
class SamplingProfile:
    """Per-model sampling params for OpenAI-compatible providers.

    A field set to ``None`` means "do not send this param" (omit); a numeric
    value means "send exactly this". Penalties (frequency/presence) and now
    temperature/top_p (kdsn.241.3.1) are both profile-driven. The vendor
    defaults for GLM-5.2/Kimi K2.6 are already correct (temp=1.0, top_p=0.95
    via generation_config.json on Fireworks, which auto-applies the model's
    own config when the client omits the param) — self-hosted models with no
    such vendor auto-fill (e.g. MiniMax-M3 on our own llama.cpp server) need
    an explicit pinned profile instead, since the serving stack's own
    defaults won't match the vendor's recommended values.

    Precedence: profile temperature/top_p WIN over a caller-supplied
    per-agent override (``AgentConfig.temperature``/``top_p``) — a profile
    entry represents a vendor-pinned or hard operational requirement, not a
    preference. The per-agent override only applies when the resolved
    profile leaves the field as ``None``.
    """
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    temperature: float | None = None
    top_p: float | None = None


# Default: OMIT penalties and temperature/top_p. This replaces the old
# unconditional frequency_penalty=0.3, which the LZ-Penalty paper + every
# vendor's own defaults show is harmful to long-reasoning models and
# unnecessary elsewhere (kdsn.241.3 / penalty-and-harness-practice.md).
_DEFAULT_SAMPLING_PROFILE = SamplingProfile()

# Per-model overrides, fragment-keyed (substring match on api_model.lower(),
# first match wins — same convention as _MODEL_CAPABILITIES). GLM-5.2 and
# Kimi K2.6 are PINNED to zero penalties per vendor requirement (Moonshot
# hard-errors on nonzero for Kimi; Z.AI omits the param). They currently equal
# the default, but are enumerated explicitly so a future change to the default
# can never silently start sending these models a penalty they must not get.
#
# MiniMax-M3 (kdsn.241.3.1): pinned to temp=1.0/top_p=0.95, the vendor value
# (Unsloth model card + vLLM recipes + Unsloth docs all converge on
# 1.0/0.95/40). Our llama.cpp server's own default is 0.80/0.95/40 (verified
# via /props + --help) and the GGUF carries no general.default_sampling.* KV
# pairs to auto-supply it — so unlike Fireworks-hosted GLM/Kimi, nothing
# upstream fills this in for us. Root cause of the 2026-07-20 runaway-
# generation incident (task ran 30+ min / 22k+ tokens with no EOS on
# !pySmWhiuIGZElv1Wul): temp 0.80 without a repeat penalty is a known
# mode-collapse setup for long structured/reasoning transcripts.
_SAMPLING_PROFILES: list[tuple[str, SamplingProfile]] = [
    ("glm-5p2",    SamplingProfile(frequency_penalty=None, presence_penalty=None)),
    ("kimi-k2p6",  SamplingProfile(frequency_penalty=None, presence_penalty=None)),
    ("minimax-m3", SamplingProfile(temperature=1.0, top_p=0.95)),
]


def _sampling_profile(api_model: str) -> SamplingProfile:
    """Resolve the sampling profile for a model, or the default if unmatched."""
    m = api_model.lower()
    for frag, profile in _SAMPLING_PROFILES:
        if frag in m:
            return profile
    return _DEFAULT_SAMPLING_PROFILE


def _cc_split(usage, attr: str) -> int | None:
    """Read a TTL-split cache-creation field off an Anthropic SDK usage object.

    Returns None when the response carries no `cache_creation` breakdown
    (older/edge responses, or providers that don't surface the split).
    """
    cc = getattr(usage, "cache_creation", None)
    if cc is None:
        return None
    return getattr(cc, attr, None)


# ---- Session USD cost pricing (Anthropic only) --------------------------
# Verified 2026-07-09 against platform.claude.com/docs/en/about-claude/pricing
# (official Anthropic docs). Base rates in USD per MTok. Cache costs derive
# from these uniform multipliers (confirmed uniform across every model row):
CACHE_READ_MULT = 0.1        # cache hit / refresh  = 0.1x base input
CACHE_WRITE_5M_MULT = 1.25   # 5-minute cache write = 1.25x base input
CACHE_WRITE_1H_MULT = 2.0    # 1-hour cache write   = 2.0x base input

# No >200K long-context premium tier exists for any current model — dropped
# per SB (2026-07-09); the historical surcharge was a Sonnet 4.5 1M-beta
# artifact, retired. Rates are flat across all context sizes.
#
# Keyed by BARE resolved model string. compute_cost() normalizes off a
# provider prefix ("anthropic/...") and a trailing -YYYYMMDD date suffix
# before lookup. Sonnet 5 carries an effective-date schedule (intro rates
# through 2026-08-31, standard from 2026-09-01).
_MODEL_PRICING: dict[str, dict] = {
    "claude-opus-5":     {"input": 5.0,  "output": 25.0},
    "claude-opus-4-8":   {"input": 5.0,  "output": 25.0},
    "claude-opus-4-7":   {"input": 5.0,  "output": 25.0},
    "claude-opus-4-6":   {"input": 5.0,  "output": 25.0},
    "claude-opus-4-5":   {"input": 5.0,  "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0,  "output": 15.0},
    "claude-sonnet-4-5": {"input": 3.0,  "output": 15.0},
    "claude-haiku-4-5":  {"input": 1.0,  "output": 5.0},
    "claude-fable-5":    {"input": 10.0, "output": 50.0},
    "claude-mythos-5":   {"input": 10.0, "output": 50.0},
    "claude-sonnet-5": {
        # (effective_date, rates) — pick the latest date <= call date.
        "effective": [
            (date(1970, 1, 1),  {"input": 2.0, "output": 10.0}),   # introductory
            (date(2026, 9, 1),  {"input": 3.0, "output": 15.0}),   # standard
        ],
    },
}


@dataclass
class CostResult:
    """Result of a single-call cost computation.

    priced=False means the model wasn't in the Anthropic pricing table: the
    call is never dollar-costed (cost_usd=0.0) and its whole token count is
    reported in `unpriced_tokens` instead (never a silent $0, never a crash).
    """
    cost_usd: float
    unpriced_tokens: int
    priced: bool


def _resolve_rates(entry: dict, now: date) -> dict:
    """Select base input/output rates, honoring an effective-date schedule."""
    if "effective" in entry:
        chosen = entry["effective"][0][1]
        for eff_date, rates in entry["effective"]:
            if now >= eff_date:
                chosen = rates
        return chosen
    return entry


_warned_unpriced_anthropic: set[str] = set()


def compute_cost(model: str, usage: Usage, *, cache_ttl_fallback: str = "1h",
                 now: date | datetime | None = None,
                 is_anthropic: bool | None = None) -> CostResult:
    """Frozen USD cost of one Anthropic API call. Pure (except a warn-once log).

    Cache-write tokens are costed per their TTL bucket using the SDK's 5m/1h
    split (Usage.cache_creation_5m_tokens / _1h_tokens). If only an aggregate
    is available (older/edge responses), it is costed at `cache_ttl_fallback`
    (default "1h", the platform default and the conservative choice).

    `is_anthropic` is the authoritative provider-type gate (kdsn.218 remediation
    F2): pass `False` to force unpriced when the serving provider is NOT
    Anthropic-typed — so a non-Anthropic provider returning a claude-shaped
    model id (e.g. an OpenAI-compatible router) is never dollar-costed at
    Anthropic rates. `None` (default) keeps pure string-shape classification for
    unit tests / legacy callers; production call sites always pass an explicit
    bool resolved from the provider config.

    Non-Anthropic or unlisted-Anthropic models are never dollar-costed: their
    whole token count goes to `unpriced_tokens`. `now` selects the effective
    rate for date-scheduled models (Sonnet 5); default is today's UTC date.
    """
    if now is None:
        now = datetime.now(timezone.utc).date()
    elif isinstance(now, datetime):
        now = now.date()

    in_tok = usage.input_tokens or 0
    out_tok = usage.output_tokens or 0
    cache_read = usage.cache_read_tokens or 0
    cc_total = usage.cache_creation_tokens or 0
    cc_5m = usage.cache_creation_5m_tokens
    cc_1h = usage.cache_creation_1h_tokens

    # Provider-type gate (F2): a caller that knows the serving provider is not
    # Anthropic-typed forces unpriced regardless of the model string's shape.
    if is_anthropic is False:
        unpriced = in_tok + out_tok + cache_read + cc_total
        logger.debug("cost: unpriced (non-Anthropic provider) model %r", model)
        return CostResult(cost_usd=0.0, unpriced_tokens=unpriced, priced=False)

    # Normalize: strip provider prefix (last path component), then date suffix.
    base = model.rsplit("/", 1)[-1] if "/" in model else model
    entry = _MODEL_PRICING.get(base)
    if entry is None:
        stripped = re.sub(r"-\d{8}$", "", base)
        entry = _MODEL_PRICING.get(stripped)
        if entry is not None:
            base = stripped

    if entry is None:
        unpriced = in_tok + out_tok + cache_read + cc_total
        if base.startswith("claude"):
            if base not in _warned_unpriced_anthropic:
                _warned_unpriced_anthropic.add(base)
                logger.warning(
                    "cost: unlisted Anthropic model %r — add it to _MODEL_PRICING",
                    model)
        else:
            logger.debug("cost: unpriced (non-Anthropic) model %r", model)
        return CostResult(cost_usd=0.0, unpriced_tokens=unpriced, priced=False)

    rates = _resolve_rates(entry, now)
    in_rate = rates["input"]
    out_rate = rates["output"]

    cost = in_tok * in_rate + out_tok * out_rate
    cost += cache_read * in_rate * CACHE_READ_MULT
    # Cache-write cost by TTL bucket. F8 (kdsn.218 remediation): treat a
    # present-but-zero split the same as an absent split, and cost any residual
    # (split sum < aggregate) at the fallback multiplier — so an SDK-inconsistent
    # cache_creation object can never silently under-cost the write.
    split_5m = cc_5m or 0
    split_1h = cc_1h or 0
    split_sum = split_5m + split_1h
    _fallback_mult = CACHE_WRITE_5M_MULT if cache_ttl_fallback == "5m" else CACHE_WRITE_1H_MULT
    if split_sum == 0 and cc_total > 0:
        cost += cc_total * in_rate * _fallback_mult
        logger.debug("cost: cache_creation split absent/zero; costed %d tokens at "
                     "%s fallback multiplier", cc_total, cache_ttl_fallback)
    else:
        cost += split_5m * in_rate * CACHE_WRITE_5M_MULT
        cost += split_1h * in_rate * CACHE_WRITE_1H_MULT
        residual = cc_total - split_sum
        if residual > 0:
            cost += residual * in_rate * _fallback_mult
            logger.debug("cost: cache_creation split (%d) < aggregate (%d); costed "
                         "residual %d at %s fallback", split_sum, cc_total, residual,
                         cache_ttl_fallback)
    cost /= 1_000_000.0
    return CostResult(cost_usd=cost, unpriced_tokens=0, priced=True)


def _convert_tools_for_provider(tools: list | None, provider_type: str) -> list[dict] | None:
    """Convert ToolDef list to provider-native format.
    
    Anthropic: [{"name": ..., "description": ..., "input_schema": ...}]
    OpenAI: [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
    """
    if tools is None:
        return None
    
    result = []
    for tool in tools:
        if provider_type == "anthropic":
            result.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        elif provider_type == "openai":
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
    return result


def _dedup_trailing_user(messages: list[dict]) -> list[dict]:
    """Belt-and-suspenders guard: drop a duplicate trailing user message.

    If the last two messages are both role="user" with identical content,
    remove the last one and log a WARNING.  This catches any regression where
    the gated-room path accidentally re-appends the triggering message.

    Non-duplicate consecutive user messages (different content or different
    roles) pass through unchanged.

    Args:
        messages: Normalised message list (not yet converted to provider format).

    Returns:
        Same list, minus the duplicate tail entry if one was detected.
    """
    if len(messages) < 2:
        return messages
    last = messages[-1]
    prev = messages[-2]
    if (
        last.get("role") == "user"
        and prev.get("role") == "user"
        and last.get("content") == prev.get("content")
    ):
        logger.warning(
            "Deduped trailing user message — possible gated-room race. "
            "Content snippet: %r",
            str(last.get("content", ""))[:120],
        )
        return messages[:-1]
    return messages


def _convert_messages_for_provider(
    messages: list[dict],
    provider_type: str,
    quirks: list[str] | None = None,
) -> list[dict]:
    """Convert normalized message history to provider-native format.
    
    Normalized format:
    - {"role": "user", "content": "text"}
    - {"role": "assistant", "content": "text", "tool_calls": [ToolCall(...)]}
    - {"role": "tool", "tool_call_id": "...", "content": "...", "is_error": bool}
    
    Anthropic native:
    - Assistant: {"role": "assistant", "content": [{"type": "text", ...}, {"type": "tool_use", ...}]}
    - Tool result: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": ..., "content": ..., "is_error": ...}]}
    
    OpenAI native:
    - Assistant: {"role": "assistant", "content": "text", "tool_calls": [{"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}]}
    - Tool result: {"role": "tool", "tool_call_id": "...", "content": "..."}
    """
    if provider_type == "anthropic":
        return _convert_messages_for_anthropic(messages)
    elif provider_type == "openai":
        reasoning_replay = "reasoning_replay" in (quirks or [])
        return _convert_messages_for_openai(messages, reasoning_replay=reasoning_replay)
    return messages


def _convert_messages_for_anthropic(messages: list[dict]) -> list[dict]:
    """Convert normalized messages to Anthropic format.

    Thinking blocks are preserved on all assistant messages and passed back
    to the API verbatim.  Per Anthropic docs, the API automatically filters
    thinking blocks, uses the relevant ones to preserve reasoning, and only
    bills for the blocks shown to Claude.  On Opus 4.5+ models, the server
    actively retains prior-turn thinking in context.

    Signatures are base64 strings (pure ASCII) and survive JSONL round-tripping
    without corruption.
    """
    # Belt-and-suspenders: drop duplicate trailing user message if present.
    # The gated-room path should prevent this via append_user=False, but we
    # guard here as a defence-in-depth measure against future regressions.
    messages = _dedup_trailing_user(messages)

    result = []
    for msg in messages:
        role = msg.get("role")

        if role == "user":
            # User message may have list content (text + image blocks)
            content = msg.get("content")
            if isinstance(content, list):
                # Convert image blocks to Anthropic wire format
                converted_blocks = []
                for block in content:
                    if block.get("type") == "image":
                        # Convert to Anthropic image source format
                        converted_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.get("media_type", "image/jpeg"),
                                "data": block.get("data", ""),
                            },
                        })
                    else:
                        # Text blocks pass through unchanged
                        converted_blocks.append(block)
                result.append({"role": "user", "content": converted_blocks})
            else:
                # Simple user message - pass through
                result.append(msg)

        elif role == "assistant":
            # Assistant message may have tool_calls and thinking blocks
            tool_calls = msg.get("tool_calls", [])
            content = msg.get("content", "")
            thinking = msg.get("thinking", [])
            
            # If no thinking and no tool_calls, keep original behavior (string content)
            if not thinking and not tool_calls:
                result.append(msg)
                continue
            
            # Build content blocks: thinking blocks + text block + tool_use blocks
            content_blocks = []
            
            # Add thinking blocks first (if any)
            if thinking:
                for tb in thinking:
                    sig = tb.get("signature")
                    if sig:
                        # Valid signature - add as thinking block
                        content_blocks.append({
                            "type": "thinking",
                            "thinking": tb.get("thinking", ""),
                            "signature": sig,
                        })
                    else:
                        # Empty/None signature - demote to text block
                        content_blocks.append({
                            "type": "text",
                            "text": tb.get("thinking", ""),
                        })
            
            # Add text content if non-empty, or if there are no other blocks
            # (Anthropic rejects empty text blocks alongside tool_use)
            if content:
                content_blocks.append({"type": "text", "text": content})
            elif not content_blocks and not tool_calls:
                # No content at all and no tool calls — keep empty text to avoid
                # sending a message with zero content blocks
                content_blocks.append({"type": "text", "text": ""})
            
            # Add tool_use blocks
            if tool_calls:
                for tc in tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
            
            if content_blocks:
                result.append({"role": "assistant", "content": content_blocks})
            else:
                # No content at all - pass through minimal message
                result.append({"role": "assistant", "content": content})
        
        elif role == "tool":
            # Tool result -> user role with tool_result content block
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id"),
                "content": msg.get("content", ""),
            }
            if msg.get("is_error"):
                tool_result_block["is_error"] = True
            result.append({"role": "user", "content": [tool_result_block]})
        
        else:
            # Unknown role - pass through
            result.append(msg)
    
    # Merge consecutive user messages with list content (tool results from parallel calls)
    merged = []
    for msg in result:
        if (merged
                and merged[-1]["role"] == "user"
                and msg["role"] == "user"
                and isinstance(merged[-1].get("content"), list)
                and isinstance(msg.get("content"), list)):
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append(msg)
    return merged


def _convert_messages_for_openai(
    messages: list[dict], reasoning_replay: bool = False,
) -> list[dict]:
    """Convert normalized messages to OpenAI format.

    reasoning_replay: when True, stored thinking blocks on assistant turns are
    re-emitted as a flat ``reasoning_content`` string (the OpenAI-compat wire
    field) instead of being dropped. Vendor thinking models (Kimi K2.6,
    GLM-5.2) require the reasoning to be sent back within a multi-step
    tool-calling loop or they degenerate/loop (kdsn.241.2; vendor-guidance.md).
    Opt-in per-provider via the ``reasoning_replay`` quirk, because some strict
    OpenAI-compatible endpoints reject the unknown field; Fireworks accepts it.
    When False (default), the historical strip behavior is preserved.
    """
    # Belt-and-suspenders: drop duplicate trailing user message if present.
    # The gated-room path should prevent this via append_user=False, but we
    # guard here as a defence-in-depth measure against future regressions.
    messages = _dedup_trailing_user(messages)

    # Normalize the provider-specific `thinking` field on assistant messages.
    # Default: strip it (most OpenAI-compatible APIs reject the unknown field).
    # reasoning_replay: convert it to a verbatim `reasoning_content` string so
    # the model keeps its reasoning state across tool-call steps.
    def _normalize_assistant_thinking(msg: dict) -> dict:
        if msg.get("role") != "assistant":
            return msg
        out = {k: v for k, v in msg.items() if k != "thinking"}
        if reasoning_replay:
            texts = [
                tb.get("thinking", "")
                for tb in (msg.get("thinking") or [])
                if tb.get("thinking")
            ]
            joined = "\n".join(texts)
            if joined:
                out["reasoning_content"] = joined
        return out

    messages = [_normalize_assistant_thinking(msg) for msg in messages]

    result = []
    for msg in messages:
        role = msg.get("role")

        if role == "user":
            # User message may have list content (text + image blocks)
            content = msg.get("content")
            if isinstance(content, list):
                # Convert image blocks to OpenAI wire format
                converted_blocks = []
                for block in content:
                    if block.get("type") == "image":
                        # Convert to OpenAI image_url format
                        media_type = block.get("media_type", "image/jpeg")
                        data = block.get("data", "")
                        converted_blocks.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{data}",
                            },
                        })
                    else:
                        # Text blocks pass through unchanged
                        converted_blocks.append(block)
                result.append({"role": "user", "content": converted_blocks})
            else:
                # Simple user message - pass through
                result.append(msg)

        elif role == "assistant":
            # Assistant message may have tool_calls
            tool_calls = msg.get("tool_calls", [])
            content = msg.get("content", "")
            
            if tool_calls:
                # Convert ToolCall objects to OpenAI format
                openai_tool_calls = []
                for tc in tool_calls:
                    oai_tc = {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.input),
                        },
                    }
                    # Echo back opaque provider metadata verbatim if present
                    # (e.g. Google's extra_content.google.thought_signature —
                    # required on the next turn for Gemini 3.x tool use, see
                    # ToolCall.extra_content docstring / bead
                    # workspace-kdsn.186.18). No-op (key omitted) for every
                    # provider that never populated it.
                    extra_content = getattr(tc, "extra_content", None)
                    if extra_content:
                        oai_tc["extra_content"] = extra_content
                    openai_tool_calls.append(oai_tc)
                built = {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": openai_tool_calls,
                }
                # Carry reasoning_content into the rebuilt tool-call turn.
                # _normalize_assistant_thinking set it on `msg`, but this branch
                # constructs a fresh dict, so it must be forwarded explicitly —
                # this is exactly the in-tool-loop turn the vendors require it on
                # (kdsn.241.2). No-op when reasoning_replay is off / absent.
                if msg.get("reasoning_content"):
                    built["reasoning_content"] = msg["reasoning_content"]
                result.append(built)
            else:
                # No tool calls - simple text message
                result.append(msg)
        
        elif role == "tool":
            # Tool result -> OpenAI tool role
            # OpenAI has no native is_error field; prepend marker so the
            # model knows the tool call failed.
            tool_content = msg.get("content", "")
            if msg.get("is_error") and not tool_content.startswith("Error: "):
                tool_content = f"Error: {tool_content}"
            result.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id"),
                "content": tool_content,
            })
        
        else:
            # User and other roles - pass through
            result.append(msg)
    
    return result


def _parse_anthropic_response(response) -> Response:
    """Parse Anthropic response into normalized Response."""
    # Extract text content, tool_calls, and thinking blocks from content blocks
    text_parts = []
    tool_calls = []
    thinking_blocks = []
    
    for block in response.content:
        # Handle both MagicMock (no type attr) and real response blocks
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                input=block.input,
            ))
        elif block_type == "thinking":
            thinking_blocks.append(ThinkingBlock(
                thinking=block.thinking,
                signature=block.signature,
            ))
        else:
            # Backward compatibility: old mocks have .text but no .type="text"
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
    
    content = "\n".join(text_parts) if text_parts else ""
    
    return Response(
        content=content,
        tool_calls=tool_calls,
        thinking=thinking_blocks,
        model=response.model,
        usage=Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_input_tokens,
            cache_creation_tokens=response.usage.cache_creation_input_tokens,
            cache_creation_5m_tokens=_cc_split(response.usage, "ephemeral_5m_input_tokens"),
            cache_creation_1h_tokens=_cc_split(response.usage, "ephemeral_1h_input_tokens"),
        ),
        stop_reason=response.stop_reason,
        generation_id=getattr(response, "id", "") or "",
    )


def _parse_openai_response(response) -> Response:
    """Parse OpenAI response into normalized Response."""
    message = response.choices[0].message
    content = message.content or ""
    tool_calls = []
    
    if message.tool_calls:
        for tc in message.tool_calls:
            # Parse function arguments JSON
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            # Capture opaque provider metadata that must be echoed back next
            # turn (currently: Google's extra_content.google.thought_signature
            # on Gemini 3.x — see ToolCall.extra_content docstring / bead
            # workspace-kdsn.186.18). Verified live (2026-07-16) that Gemini's
            # OpenAI-compat tool_call objects surface this as a plain
            # attribute (pydantic extra="allow"); harmless None on providers
            # that don't send it.
            extra_content = getattr(tc, "extra_content", None)

            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=arguments,
                extra_content=extra_content,
            ))
    
    # Extract reasoning (OpenRouter extension) into thinking blocks
    thinking_blocks = []
    reasoning = getattr(message, 'reasoning', None) or getattr(message, 'reasoning_content', None)
    if isinstance(reasoning, str) and reasoning:
        thinking_blocks.append(ThinkingBlock(thinking=reasoning, signature=""))

    return Response(
        content=content,
        tool_calls=tool_calls,
        thinking=thinking_blocks,
        model=response.model,
        usage=Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        ),
        stop_reason=response.choices[0].finish_reason,
        generation_id=getattr(response, "id", "") or "",
    )


def _build_anthropic_kwargs(
    api_model: str,
    system: str,
    provider_messages: list[dict],
    provider_tools: list[dict] | None,
    max_tokens: int,
    thinking_level: str,
    model_max_tokens: int = 200000,
    temperature: float | None = None,
    top_p: float | None = None,
    cache_ttl: str | None = None,
) -> dict:
    """Build kwargs for Anthropic messages API."""
    _cc = {"type": "ephemeral", "ttl": cache_ttl or "1h"}
    api_kwargs = {
        "model": api_model,
        "system": [{"type": "text", "text": system, "cache_control": _cc}],
        "messages": provider_messages,
        "max_tokens": max_tokens,
    }
    if provider_tools:
        api_kwargs["tools"] = provider_tools
    
    # Add prompt caching to last user message.
    # Deep copy the target message to avoid mutating the caller's history
    # dicts (shared references from agent.py's shallow list copy).
    #
    # Caller-placed breakpoint (advisor amendment, design §14 #1b): if a
    # caller has ALREADY placed a block-level cache_control anywhere in
    # provider_messages (e.g. the advisor handler puts one on its rendered-
    # transcript block so a later, cheap-to-vary block can follow it), honor
    # that breakpoint and SKIP the automatic last-block application below --
    # otherwise the auto-apply would move (or add a second) breakpoint onto
    # the trailing block, busting the cache prefix the caller deliberately
    # pinned earlier in the message. When no caller breakpoint is present,
    # behavior is UNCHANGED: the last block of the last user message is
    # still cached automatically (backward-compatible; every pre-existing
    # caller relies on this and places no cache_control of its own).
    _caller_has_breakpoint = any(
        isinstance(block, dict) and "cache_control" in block
        for msg in api_kwargs["messages"]
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
    )
    if not _caller_has_breakpoint and api_kwargs["messages"]:
        last_msg = api_kwargs["messages"][-1]
        if last_msg.get("role") == "user":
            last_msg = copy.deepcopy(last_msg)
            api_kwargs["messages"][-1] = last_msg
            msg_content = last_msg.get("content")
            if isinstance(msg_content, str):
                last_msg["content"] = [{"type": "text", "text": msg_content, "cache_control": _cc}]
            elif isinstance(msg_content, list) and msg_content:
                msg_content[-1]["cache_control"] = _cc
    
    # Add sampling parameters. Two gates:
    #   1. Anthropic disallows sampling params together with extended thinking.
    #   2. Modern Anthropic models (Opus 4.7+, Sonnet 5, Fable, Mythos) REMOVED
    #      temperature/top_p/top_k; sending them 400s. _supports_sampling_params()
    #      allowlists the older families (unknown/future models default to off).
    if thinking_level == "off" and _supports_sampling_params(api_model):
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if top_p is not None:
            api_kwargs["top_p"] = top_p

    # Add thinking parameters if enabled
    if thinking_level != "off":
        if _supports_adaptive_thinking(api_model):
            # Adaptive thinking for Opus/Sonnet 4-6+
            # Always request summarized display — Opus 4.7+ and Mythos
            # default to "omitted" (empty thinking field, signature only).
            api_kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            api_kwargs["output_config"] = {"effort": _thinking_effort(thinking_level)}
        else:
            # Budget-based thinking for older models
            if thinking_level in ("max", "xhigh"):
                logger.warning(
                    "Thinking level %r requires adaptive thinking but model %r "
                    "uses budget-based thinking; falling back to high budget (16384 tokens).",
                    thinking_level, api_model,
                )
            budget, adjusted_max = _thinking_budget(thinking_level, max_tokens, model_max_tokens)
            api_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            api_kwargs["max_tokens"] = adjusted_max

    # Clamp to the model output cap; Anthropic 400s if max_tokens exceeds it
    # (Haiku 4.5 = 64000). Matters for sub-agent dispatch, which inherits the
    # parent max_tokens (64K parent -> Haiku sub = 64K + 16K budget = 80K > 64K).
    _cap = _model_output_cap(api_model)
    if _cap is not None:
        api_kwargs["max_tokens"] = min(api_kwargs["max_tokens"], _cap)

    return api_kwargs


def _build_openai_kwargs(
    api_model: str,
    system: str,
    provider_messages: list[dict],
    provider_tools: list[dict] | None,
    max_tokens: int,
    thinking_level: str,
    quirks: list[str],
    temperature: float | None = None,
    top_p: float | None = None,
    routing: dict | None = None,
    provider_key: str = "",
) -> dict:
    """Build kwargs for OpenAI chat completions API."""
    # Provider capability flags — OpenRouter proxies handle unknown params gracefully,
    # but direct APIs (OpenAI, Google) reject params they don't support.
    _supports_penalties = provider_key not in ("google",)
    _supports_reasoning_extra = provider_key in ("openrouter", "macstudio")
    # OpenAI deprecated max_tokens in favor of max_completion_tokens (o1+, GPT-5+).
    # Google and OpenRouter still use max_tokens.
    _uses_max_completion_tokens = provider_key in ("openai",)

    # Handle quirks
    if "no_system_role" in quirks:
        # Fold system into first user message instead of separate system role
        messages_with_system = provider_messages
        if messages_with_system and messages_with_system[0]["role"] == "user":
            first = messages_with_system[0].copy()
            content = first.get("content", "")
            if isinstance(content, str):
                first["content"] = f"{system}\n\n{content}"
            messages_with_system = [first] + messages_with_system[1:]
        else:
            messages_with_system = [{"role": "user", "content": system}] + messages_with_system
    else:
        # Normal: prepend system message to messages list
        messages_with_system = [{"role": "system", "content": system}] + provider_messages
    
    _token_key = "max_completion_tokens" if _uses_max_completion_tokens else "max_tokens"
    api_kwargs = {
        "model": api_model,
        "messages": messages_with_system,
        _token_key: max_tokens,
    }
    # Sampling penalties: roster-driven per-model profile (kdsn.241.3).
    # Default profile omits them; GLM/Kimi pinned to omit. A profile value of
    # None means "don't send"; a number means "send exactly this". Gated by
    # provider support (Google rejects penalty params).
    _profile = _sampling_profile(api_model)
    if _supports_penalties:
        if _profile.frequency_penalty is not None:
            api_kwargs["frequency_penalty"] = _profile.frequency_penalty
        if _profile.presence_penalty is not None:
            api_kwargs["presence_penalty"] = _profile.presence_penalty
    if provider_tools:
        api_kwargs["tools"] = provider_tools
    
    # Add sampling parameters. Profile temperature/top_p (kdsn.241.3.1) win
    # over the caller-supplied per-agent override — a profile entry is a
    # vendor-pinned/hard requirement (e.g. MiniMax-M3's 1.0/0.95), not a
    # preference, so it must not be silently overridable by agent TOML.
    # Unlike penalties, temperature/top_p are NOT gated by _supports_penalties
    # (that gate is Google-penalty-specific; Google does support temp/top_p).
    _temperature = _profile.temperature if _profile.temperature is not None else temperature
    _top_p = _profile.top_p if _profile.top_p is not None else top_p
    if _temperature is not None:
        api_kwargs["temperature"] = _temperature
    if _top_p is not None:
        api_kwargs["top_p"] = _top_p

    # Build extra_body incrementally — reasoning and provider routing are
    # OpenRouter extensions, not part of the standard OpenAI API.
    extra_body = {}
    if thinking_level != "off" and _supports_reasoning_extra:
        extra_body["reasoning"] = {"effort": thinking_level}
    if routing:
        extra_body["provider"] = routing
    if extra_body:
        api_kwargs["extra_body"] = extra_body

    # Clamp to model output cap (defensive; OpenAI-compatible providers we use
    # tolerate over-cap max_tokens, so _model_output_cap returns None for them).
    _cap = _model_output_cap(api_model)
    if _cap is not None:
        api_kwargs[_token_key] = min(api_kwargs[_token_key], _cap)

    return api_kwargs


KEEPALIVE_MAX_OUTPUT_TOKENS = 1  # ping output is discarded; we only want the cache read


async def ping_cache(
    config: AgentConfig,
    *,
    system: str,
    messages: list[dict],
    tools: list | None,
    cache_ttl: str | None,
    thinking_level: str,
    model: str | None = None,
) -> Usage | None:
    """Refresh an Anthropic prompt-cache prefix by replaying the last request.

    Issues a SINGLE non-streaming Anthropic call replaying the parent's in-flight
    cached prefix (model + system + tools + messages + cache_ttl) AND its exact
    thinking config: thinking_level rebuilds the same adaptive+effort (or budget)
    block the parent used, capped at KEEPALIVE_MAX_OUTPUT_TOKENS. The parent's
    thinking mode is LOAD-BEARING -- Anthropic incorporates the extended-thinking
    mode AND effort into the prompt-cache key, so a thinking-OFF (or wrong-effort)
    replay against a thinking-ON prefix is a GUARANTEED total miss + full rewrite
    (verified live 2026-07-05; specs/subagent-cache-keepalive-prod-failure-2026-07-05.md).
    max_tokens=1 is fine even with ADAPTIVE thinking on: Anthropic accepts it and
    the response just truncates (stop_reason=max_tokens) AFTER the billed cache
    read, which is all the ping needs. The throwaway output is discarded; the
    returned Usage lets the caller verify the ping was a cache READ (hit) rather
    than a WRITE (prefix drift/expiry).

    N4: this max_tokens=1 cheapness guarantee holds ONLY for models on the
    adaptive-thinking path (``_supports_adaptive_thinking``). For BUDGET-based
    thinking models (thinking enabled, model not on that allowlist),
    ``_build_anthropic_kwargs`` unconditionally OVERRIDES max_tokens to
    base+budget_tokens (up to ~16384 for thinking_level="high") -- reusing
    that SAME decision (not a duplicated model-name list) below, this
    function refuses to ping at all in that case rather than silently
    authorize a real (and expensive) thinking generation every keepalive
    interval. The ADAPTIVE path is completely unreached by this new check
    and is byte-for-byte unchanged from before.

    Returns None for non-Anthropic providers (OpenAI-compat auto-caches; no TTL
    knob, no write premium) — nothing to refresh. For a budget-thinking model
    (see N4 above), returns a deliberately miss-shaped ``Usage`` (cache_read
    <= cache_creation, so ``_keepalive_is_hit`` reads it as a MISS) instead of
    attempting any API call -- this reuses the caller's EXISTING miss-handling
    branch (log + operator on_miss notice + abort the keepalive loop) as the
    closest available "abort keepalive" signal, without requiring any change
    to the caller's retry/error-counting logic.
    """
    model_str = model or config.default_model
    provider_cfg, api_model = resolve_model(
        model_str, config.providers, aliases=config.model_aliases,
    )
    if provider_cfg.type != "anthropic":
        return None

    # N4: a keepalive ping must NEVER authorize a real thinking budget. This
    # is the EXACT condition _build_anthropic_kwargs uses to pick its
    # "budget-based thinking for older models" branch (thinking enabled AND
    # the model is not on the adaptive-thinking allowlist) -- reusing
    # _supports_adaptive_thinking directly rather than re-deriving/duplicating
    # its model-name list. That branch overrides max_tokens to base+budget,
    # discarding the KEEPALIVE_MAX_OUTPUT_TOKENS=1 cap this ping is built
    # around, so a ping against such a model is never actually cheap. Refuse
    # to ping at all rather than let that happen silently.
    if thinking_level != "off" and not _supports_adaptive_thinking(api_model):
        logger.warning(
            "cache keepalive ping skipped for model %r: thinking_level=%r "
            "requires budget-based thinking on this model, which would "
            "override max_tokens to base+budget instead of the cheap "
            "max_tokens=%d the ping is designed around -- a cache-key-"
            "matching ping cannot be cheap for budget-thinking models.",
            api_model, thinking_level, KEEPALIVE_MAX_OUTPUT_TOKENS,
        )
        return Usage(
            input_tokens=0, output_tokens=0,
            cache_read_tokens=0, cache_creation_tokens=0,
        )

    provider_messages = _convert_messages_for_provider(messages, provider_cfg.type)
    provider_tools = _convert_tools_for_provider(tools, provider_cfg.type)
    client = _get_client(provider_cfg)

    api_kwargs = _build_anthropic_kwargs(
        api_model=api_model,
        system=system,
        provider_messages=provider_messages,
        provider_tools=provider_tools,
        max_tokens=KEEPALIVE_MAX_OUTPUT_TOKENS,
        thinking_level=thinking_level,
        model_max_tokens=getattr(config, "model_max_tokens", 200000),
        temperature=None,
        top_p=None,
        cache_ttl=cache_ttl,
    )

    response = await client.messages.create(**api_kwargs)
    u = response.usage
    return Usage(
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_tokens=u.cache_read_input_tokens,
        cache_creation_tokens=u.cache_creation_input_tokens,
    )


async def stream(
    config: AgentConfig,
    system: str,
    messages: list[dict],
    tools: list | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    thinking: str | None = None,
    cache_ttl: str | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Stream completion events from Anthropic or OpenAI SDK based on config.providers.
    
    Yields StreamEvent objects for each event in the stream.
    Final event is always type="done" with the complete Response.
    """
    # Resolve model string to provider and API model name
    model_str = model or config.default_model
    provider_cfg, api_model = resolve_model(
        model_str, config.providers, aliases=config.model_aliases,
    )
    
    # Convert messages to provider-native format
    provider_messages = _convert_messages_for_provider(
        messages, provider_cfg.type, quirks=provider_cfg.quirks,
    )
    
    # Convert tools to provider-native format
    provider_tools = _convert_tools_for_provider(tools, provider_cfg.type)
    
    # Use config.max_tokens if max_tokens is not provided
    tokens = max_tokens if max_tokens is not None else config.max_tokens
    
    # Resolve thinking level: param > config > "off"
    thinking_level = thinking if thinking is not None else getattr(config, "thinking", "off")

    if provider_cfg.type == "anthropic":
        client = _get_client(provider_cfg)
        
        api_kwargs = _build_anthropic_kwargs(
            api_model=api_model,
            system=system,
            provider_messages=provider_messages,
            provider_tools=provider_tools,
            max_tokens=tokens,
            thinking_level=thinking_level,
            model_max_tokens=getattr(config, "model_max_tokens", 200000),
            temperature=getattr(config, "temperature", None),
            top_p=getattr(config, "top_p", None),
            cache_ttl=cache_ttl,
        )
        
        try:
            async with client.messages.stream(**api_kwargs) as stream:
                accumulated_text = ""
                
                async for event in stream:
                    event_type = getattr(event, "type", None)
                    
                    if event_type == "text":
                        yield StreamEvent(type="text", content=event.text)
                        accumulated_text += event.text
                    elif event_type == "thinking":
                        yield StreamEvent(type="thinking", content=event.thinking)
                    elif event_type == "signature":
                        yield StreamEvent(type="signature", content=event.signature)
                    elif event_type == "input_json":
                        yield StreamEvent(
                            type="tool_delta",
                            content=event.partial_json,
                        )
                    elif event_type == "content_block_start":
                        block = event.content_block
                        if getattr(block, "type", None) == "tool_use":
                            yield StreamEvent(
                                type="tool_start",
                                tool_index=event.index,
                                tool_id=block.id,
                                tool_name=block.name,
                            )
                    elif event_type == "content_block_stop":
                        block = event.content_block
                        if getattr(block, "type", None) == "tool_use":
                            yield StreamEvent(
                                type="tool_done",
                                tool_index=event.index,
                                tool_call=ToolCall(
                                    id=block.id,
                                    name=block.name,
                                    input=block.input,
                                ),
                            )
                    elif event_type == "message_stop":
                        # Get final message and build response
                        final_message = await stream.get_final_message()
                        response = _parse_anthropic_response(final_message)
                        response.content, response.degenerate = _detect_and_truncate_degeneration(response.content)
                        
                        yield StreamEvent(
                            type="done",
                            stop_reason=final_message.stop_reason,
                            model=final_message.model,
                            response=response,
                        )
                        
        except anthropic.APIStatusError as e:
            raise ProviderError(
                _sanitize_error(e.message), status_code=e.status_code,
            ) from e
        except anthropic.APITimeoutError as e:
            raise ProviderError("Provider request timed out") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError("Provider unreachable — connection failed") from e
    
    elif provider_cfg.type == "openai":
        client = _get_client(provider_cfg)

        # Streaming degeneration monitor (kdsn.241.4) — OpenAI-compatible path only,
        # where the open-weight (GLM/Kimi) repetition collapse this targets occurs.
        # Warn-only by default: logs trips, never modifies output. "abort" mode is
        # config-gated and dormant (its mid-stream teardown + truncation precision
        # are validated in Phase 2 before it is armed in production).
        degen_monitor = DegenerationMonitor(mode=getattr(config, "degen_detector", "warn"))

        api_kwargs = _build_openai_kwargs(
            api_model=api_model,
            system=system,
            provider_messages=provider_messages,
            provider_tools=provider_tools,
            max_tokens=tokens,
            thinking_level=thinking_level,
            quirks=provider_cfg.quirks,
            temperature=getattr(config, "temperature", None),
            top_p=getattr(config, "top_p", None),
            routing=provider_cfg.routing,
            provider_key=provider_cfg.key,
        )
        
        # Add streaming-specific kwargs
        api_kwargs["stream"] = True
        api_kwargs["stream_options"] = {"include_usage": True}
        
        try:
            response = await client.chat.completions.create(**api_kwargs)
            
            accumulated_text = ""
            accumulated_reasoning = ""
            usage = None
            stop_reason = None
            generation_id = ""
            degen_aborted = False
            # Accumulate tool call data: index -> {"id": str, "name": str, "arguments": str}
            tool_call_accumulators: dict[int, dict] = {}
            
            async for chunk in response:
                # Capture generation ID from first chunk
                if not generation_id and getattr(chunk, "id", None):
                    generation_id = chunk.id

                # Handle usage chunk
                if chunk.usage:
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )
                
                # Process content deltas
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Handle text content
                    if delta.content is not None:
                        yield StreamEvent(type="text", content=delta.content)
                        accumulated_text += delta.content
                        if degen_monitor.feed(delta.content):
                            logger.warning(
                                "degeneration detected on stream: layer=%s pos=%s model=%s gen=%s mode=%s",
                                degen_monitor.trigger_layer, degen_monitor.trigger_pos,
                                api_model, generation_id, degen_monitor.mode,
                            )
                            if degen_monitor.mode == "abort":
                                degen_aborted = True
                                break
                    
                    # Handle reasoning (OpenRouter extension)
                    reasoning_text = getattr(delta, 'reasoning', None) or getattr(delta, 'reasoning_content', None)
                    if isinstance(reasoning_text, str) and reasoning_text:
                        yield StreamEvent(type="thinking", content=reasoning_text)
                        accumulated_reasoning += reasoning_text
                    
                    # Handle tool calls
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            # Google's Gemini OpenAI-compat streaming never
                            # populates tool_calls[].index (verified live
                            # 2026-07-16 — always None). Falling back to the
                            # call's own id keeps concurrent tool calls from
                            # colliding into the same accumulator slot (they
                            # would otherwise all land on the same `None` key,
                            # corrupting/losing all but one call in a
                            # multi-tool-call turn). Real index-bearing
                            # providers are unaffected — idx stays their int.
                            idx = tc_delta.index
                            if idx is None:
                                idx = tc_delta.id or f"__unindexed_{len(tool_call_accumulators)}"
                            if idx not in tool_call_accumulators:
                                tool_call_accumulators[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": tc_delta.function.name or "",
                                    "arguments": "",
                                    # Opaque metadata (e.g. Google's
                                    # extra_content.google.thought_signature)
                                    # that must be echoed back next turn —
                                    # see ToolCall.extra_content docstring.
                                    "extra_content": getattr(tc_delta, "extra_content", None),
                                }
                            if tc_delta.function.arguments:
                                tool_call_accumulators[idx]["arguments"] += tc_delta.function.arguments
                    
                    # Track finish reason
                    if choice.finish_reason:
                        stop_reason = choice.finish_reason
            
            # Mid-stream degeneration abort (kdsn.241.4). We broke out of the
            # consume loop; tear down the HTTP stream so we stop reading (and,
            # for providers that honor it, stop being billed for) the garbage
            # tail, then truncate the emitted text at the detection point and
            # append the standard warning. NOTE: real-network teardown semantics
            # are validated in Phase 2 before "abort" is armed in production;
            # the shipped default is "warn" (this branch is dormant).
            if degen_aborted:
                try:
                    await response.close()
                except Exception:
                    logger.debug("degen-abort: stream close raised (ignored)", exc_info=True)
                cut = degen_monitor.trigger_pos
                if cut is not None and 0 <= cut <= len(accumulated_text):
                    accumulated_text = accumulated_text[:cut].rstrip()
                accumulated_text = (accumulated_text + _DEGEN_WARNING) if accumulated_text else _DEGEN_WARNING.lstrip()
                stop_reason = stop_reason or "degenerate"
                # Drop any partially-accumulated tool calls: a mid-stream abort
                # means their JSON args are incomplete/garbage — never emit or
                # execute them.
                tool_call_accumulators.clear()
            
            # Yield tool_done events for accumulated tool calls
            for idx in sorted(tool_call_accumulators.keys()):
                tc_data = tool_call_accumulators[idx]
                try:
                    input_dict = json.loads(tc_data["arguments"])
                except json.JSONDecodeError:
                    input_dict = {}
                
                yield StreamEvent(
                    type="tool_done",
                    tool_index=idx,
                    tool_call=ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        input=input_dict,
                        extra_content=tc_data.get("extra_content"),
                    ),
                )
            
            # Build tool_calls list for the response
            response_tool_calls = []
            for idx in sorted(tool_call_accumulators.keys()):
                tc_data = tool_call_accumulators[idx]
                try:
                    input_dict = json.loads(tc_data["arguments"])
                except json.JSONDecodeError:
                    input_dict = {}
                response_tool_calls.append(ToolCall(
                    id=tc_data["id"],
                    name=tc_data["name"],
                    input=input_dict,
                    extra_content=tc_data.get("extra_content"),
                ))
            
            # Build and yield final done event
            thinking_blocks = []
            if accumulated_reasoning:
                thinking_blocks.append(ThinkingBlock(thinking=accumulated_reasoning, signature=""))

            response_obj = Response(
                content=accumulated_text,
                model=api_model,
                usage=usage or Usage(input_tokens=0, output_tokens=0),
                stop_reason=stop_reason or "",
                tool_calls=response_tool_calls,
                thinking=thinking_blocks,
                generation_id=generation_id,
            )
            response_obj.content, response_obj.degenerate = _detect_and_truncate_degeneration(response_obj.content)
            response_obj.degenerate = response_obj.degenerate or degen_monitor.tripped
            
            yield StreamEvent(
                type="done",
                stop_reason=stop_reason or "",
                model=api_model,
                response=response_obj,
            )
            
        except openai.APIStatusError as e:
            raise ProviderError(
                _sanitize_error(e.message), status_code=e.status_code,
            ) from e
        except openai.APITimeoutError as e:
            raise ProviderError("Provider request timed out") from e
        except openai.APIConnectionError as e:
            raise ProviderError("Provider unreachable — connection failed") from e
    
    else:
        raise ValueError(f"Unsupported provider type: {provider_cfg.type}")


async def complete(
    config: AgentConfig,
    system: str,
    messages: list[dict],
    tools: list | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    thinking: str | None = None,
    cache_ttl: str | None = None,
) -> Response:
    """
    Route to Anthropic or OpenAI SDK based on config.providers.
    If max_tokens is None, use config.max_tokens.
    If model is None, use config.default_model.
    If thinking is None, use config.thinking (defaults to "off").
    cache_ttl is forwarded to stream() unchanged (advisor amendment, design
    §14 #1a); omitting it (None) preserves prior behavior exactly, since
    stream() already treats a None cache_ttl as "use the default".
    
    Errors propagate directly - no wrapping, no retry.
    """
    response = None
    async for event in stream(
        config=config,
        system=system,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        model=model,
        thinking=thinking,
        cache_ttl=cache_ttl,
    ):
        if event.type == "done":
            response = event.response
    
    if response and response.generation_id:
        out_tokens = response.usage.output_tokens if response.usage else 0
        has_content = bool(response.content and response.content.strip())
        has_tools = bool(response.tool_calls)
        logger.info(
            "generation %s model=%s out=%d stop=%s content=%s tools=%s",
            response.generation_id, response.model, out_tokens,
            response.stop_reason, has_content, has_tools,
        )

    return response
