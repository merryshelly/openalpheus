"""RED suite — advisor handler `run_advisor` + config + provider extension.

Spec: specs/advisor-design.md §3 (config), §5 (request assembly + provider call),
§6 (counter/caps), §14 amendment #1 (provider extension), §10 test plan.
Recon: specs/advisor-anchors.md §2 (`from openalph.provider import complete`
seam — mirror subagent), §7 (resolve_model), §12 (dispatch branch).

PINNED SEAMS (identity contract — implementers must satisfy exactly these):
- `run_advisor` param NAMES: focus, model, config, tool_config, callbacks
  (all invoked by keyword; mirrors the subagent dispatch branch, anchors §12).
- provider seam: `from openalph.provider import complete` at advisor.py module
  scope → tests patch "openalph.tools.advisor.complete" (subagent precedent).
- transcript seam: `callbacks["get_transcript"]` is a SYNC callable returning
  `(system_prompt, messages)` (anchors §3 sub lambda `(system, list(messages))`).
- counter seam: `callbacks["advisor_uses"]` is a `dict[str,int]` keyed by
  `callbacks["room_id"]`; cap from `tool_config["max_uses"]` (default 10);
  check-and-increment is synchronous at handler entry (design §4/§6).
- config keys (advisor.toml [config]): model, max_uses, max_tokens, thinking,
  cache_ttl, timeout, include_system_prompt, transcript_max_chars.

All tests are gated on `run_advisor is not None` so red-state yields clean FAILED
lines (never collection ERRORs). The provider-extension tests target real
provider.py functions but are ALSO gated: they are part of the advisor bundle
(§14 amendment #1, Wave B) and must go green with it — gating keeps the whole
suite uniformly red now and the 2045/1-skip accounting clean.
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock

try:
    from openalph.tools.advisor import render_transcript, run_advisor
except ImportError:
    render_transcript = None
    run_advisor = None

from openalph.tools import ToolResult
from openalph.config import AgentConfig, ProviderConfig, resolve_model
from openalph.provider import (
    Response, Usage, ThinkingBlock, StreamEvent, ProviderError,
    complete, _build_anthropic_kwargs,
)

ROOM = "!advisor-room:matrix.local"
NOT_IMPL = "advisor module not implemented"
NOT_IMPL_BUNDLE = "advisor bundle (incl. provider cache_ttl/breakpoint extension) not implemented"


# --- Helpers -------------------------------------------------------------

def _provider(key="anthropic", type="anthropic"):
    return ProviderConfig(key=key, type=type, api_key="sk-test", base_url=None, quirks=[])


def _cfg(tmp_path, providers=None, aliases=None, **kw):
    defaults = dict(
        name="test-advisor",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers=providers or {"anthropic": _provider()},
        workspace=tmp_path,
        max_iterations=100,
        truncation_limit=50000,
        model_max_tokens=200000,
        matrix=None,
        reminders=True,
        model_aliases=aliases or {},
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _tc_cfg(model="anthropic/claude-opus-4-8", **kw):
    base = {
        "max_uses": 10, "max_tokens": 8192, "thinking": "medium",
        "cache_ttl": "5m", "timeout": 300,
        "include_system_prompt": True, "transcript_max_chars": 0,
    }
    if model is not None:
        base["model"] = model
    base.update(kw)
    return base


def _callbacks(system="CALLER_SYSPROMPT_XYZ", messages=None, uses=None,
               room=ROOM, call_id="tc_adv"):
    if messages is None:
        messages = [{"role": "user", "content": "TRANSCRIPT_USER_MARKER original request"}]
    return {
        "get_transcript": lambda: (system, list(messages)),
        "advisor_uses": {} if uses is None else uses,
        "room_id": room,
        "call_id": call_id,
    }


def _resp(content="ADVICE_TEXT here", thinking=None, in_tok=100, out_tok=50, cache_read=7):
    return Response(
        content=content, model="claude-opus-4-8",
        usage=Usage(input_tokens=in_tok, output_tokens=out_tok, cache_read_tokens=cache_read),
        stop_reason="end_turn", thinking=thinking or [],
    )


def _resolved_api_model(model_str, cfg):
    """Normalize whatever run_advisor passed to complete() into the api_model,
    robust to 'pass alias' vs 'pass resolved string' implementations."""
    return resolve_model(model_str, cfg.providers, aliases=cfg.model_aliases)[1]


def _blocks(msg):
    c = msg["content"]
    return c if isinstance(c, list) else [{"type": "text", "text": c}]


def _find_block(blocks, marker):
    for b in blocks:
        if marker in b.get("text", ""):
            return b
    return None


def _has_any_cache_control(messages):
    for m in messages:
        for b in _blocks(m):
            if isinstance(b, dict) and "cache_control" in b:
                return True
    return False


# ========================================================================
# Config + model resolution (§3, §5)
# ========================================================================

class TestConfigModel:

    @pytest.mark.asyncio
    async def test_missing_model_errors_with_steering_no_state_change(self, tmp_path):
        """Missing `model` config → is_error steering naming the fix; counter UNCHANGED."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        uses = {ROOM: 3}
        cb = _callbacks(uses=uses)
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            result = await run_advisor(
                focus=None, model=None, config=cfg,
                tool_config=_tc_cfg(model=None), callbacks=cb,
            )
        assert isinstance(result, ToolResult) and result.is_error, \
            "Missing advisor model must return is_error=True"
        low = result.content.lower()
        assert "model" in low and ("operator" in low or "config" in low), \
            f"Steering must name the fix (configure model / ask operator): {result.content!r}"
        assert uses[ROOM] == 3, "Missing-model error must NOT change the counter (no state change)"
        mock_complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_call_param_model_overrides_toml(self, tmp_path):
        """Call-param `model` OVERRIDES the TOML config model."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path, providers={"anthropic": _provider()})
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            result = await run_advisor(
                focus=None, model="anthropic/claude-param-override",
                config=cfg, tool_config=_tc_cfg(model="anthropic/claude-toml-model"),
                callbacks=cb,
            )
        assert not result.is_error, f"override consult should succeed: {result.content!r}"
        passed_model = mock_complete.call_args.kwargs["model"]
        assert _resolved_api_model(passed_model, cfg) == "claude-param-override", \
            "Call-param model must beat the TOML config model"

    @pytest.mark.asyncio
    async def test_alias_resolution(self, tmp_path):
        """A model alias resolves via resolve_model (no error)."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path, providers={"anthropic": _provider()},
                   aliases={"adv": "anthropic/claude-opus-4-8"})
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            result = await run_advisor(
                focus=None, model=None, config=cfg,
                tool_config=_tc_cfg(model="adv"), callbacks=cb,
            )
        assert not result.is_error, f"alias consult should succeed: {result.content!r}"
        passed_model = mock_complete.call_args.kwargs["model"]
        assert _resolved_api_model(passed_model, cfg) == "claude-opus-4-8", \
            "Alias 'adv' must resolve to its target api_model"

    @pytest.mark.asyncio
    async def test_unknown_provider_errors_with_steering(self, tmp_path):
        """Resolved model with an unconfigured provider → is_error steering."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path, providers={"anthropic": _provider()})
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            result = await run_advisor(
                focus=None, model="ghostprovider/some-model",
                config=cfg, tool_config=_tc_cfg(model="ghostprovider/some-model"),
                callbacks=cb,
            )
        assert result.is_error, "Unknown provider must return is_error=True"
        low = result.content.lower()
        assert "provider" in low or "not configured" in low, \
            f"Steering must explain the provider is not configured: {result.content!r}"
        mock_complete.assert_not_called()


# ========================================================================
# Counter + cap (§6)
# ========================================================================

class TestCap:

    @pytest.mark.asyncio
    async def test_counter_increments_on_dispatch(self, tmp_path):
        """A successful consult increments the per-room counter."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        uses = {}
        cb = _callbacks(uses=uses)
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()):
            result = await run_advisor(
                focus=None, model=None, config=cfg,
                tool_config=_tc_cfg(), callbacks=cb,
            )
        assert not result.is_error
        assert uses.get(ROOM) == 1, "Consult must increment advisor_uses[room] to 1"

    @pytest.mark.asyncio
    async def test_cap_hit_returns_steering_no_further_increment(self, tmp_path):
        """uses >= max_uses → is_error steering; counter not pushed past the cap; no LLM call."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        uses = {ROOM: 10}
        cb = _callbacks(uses=uses)
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            result = await run_advisor(
                focus=None, model=None, config=cfg,
                tool_config=_tc_cfg(max_uses=10), callbacks=cb,
            )
        assert result.is_error, "Cap hit must return is_error=True"
        low = result.content.lower()
        assert "cap" in low and "10" in result.content, \
            f"Cap steering must name the cap (10): {result.content!r}"
        assert uses[ROOM] == 10, "Cap-hit must not increment past the cap"
        mock_complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_two_consults_one_batch_both_count(self, tmp_path):
        """Two consults in ONE asyncio.gather batch BOTH count (shared counter → 2)."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        uses = {}
        cb1 = _callbacks(uses=uses, call_id="tc_a")
        cb2 = _callbacks(uses=uses, call_id="tc_b")
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()):
            r1, r2 = await asyncio.gather(
                run_advisor(focus=None, model=None, config=cfg,
                            tool_config=_tc_cfg(), callbacks=cb1),
                run_advisor(focus=None, model=None, config=cfg,
                            tool_config=_tc_cfg(), callbacks=cb2),
            )
        assert not r1.is_error and not r2.is_error
        assert uses.get(ROOM) == 2, "Both consults in one batch must count (counter → 2)"

    @pytest.mark.asyncio
    async def test_check_and_increment_before_first_await(self, tmp_path):
        """Cap race: uses=9, max=10, two-in-one-batch → exactly ONE succeeds, one caps,
        counter ends at 10 (proves the check-and-increment is SYNCHRONOUS at entry,
        before the first await — else both would read 9<10 and counter would hit 11)."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        uses = {ROOM: 9}
        cb1 = _callbacks(uses=uses, call_id="tc_a")
        cb2 = _callbacks(uses=uses, call_id="tc_b")
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            r1, r2 = await asyncio.gather(
                run_advisor(focus=None, model=None, config=cfg,
                            tool_config=_tc_cfg(max_uses=10), callbacks=cb1),
                run_advisor(focus=None, model=None, config=cfg,
                            tool_config=_tc_cfg(max_uses=10), callbacks=cb2),
            )
        errors = [r for r in (r1, r2) if r.is_error]
        assert len(errors) == 1, "Exactly one of the racing consults must be capped"
        assert uses[ROOM] == 10, "Counter must end at the cap (10), never 11"
        assert mock_complete.call_count == 1, "Only the non-capped consult may call the LLM"


# ========================================================================
# Fail-soft (§5, A6) — errors never kill the turn
# ========================================================================

class TestFailSoft:

    @pytest.mark.asyncio
    async def test_provider_error_is_soft(self, tmp_path):
        """ProviderError → is_error steering, NO exception propagates (turn survives)."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   side_effect=ProviderError("upstream 500 boom")):
            result = await run_advisor(
                focus=None, model=None, config=cfg,
                tool_config=_tc_cfg(), callbacks=cb,
            )
        assert isinstance(result, ToolResult) and result.is_error, \
            "Provider error must be a soft is_error ToolResult, not a raise"
        assert "advisor" in result.content.lower(), \
            f"Steering should mention the advisor: {result.content!r}"

    @pytest.mark.asyncio
    async def test_empty_response_is_soft_steering(self, tmp_path):
        """Empty advice → is_error steering ('proceed with your own judgment')."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp(content="")):
            result = await run_advisor(
                focus=None, model=None, config=cfg,
                tool_config=_tc_cfg(), callbacks=cb,
            )
        assert result.is_error, "Empty advisor response must return is_error=True"
        low = result.content.lower()
        assert "no advice" in low or "proceed" in low, \
            f"Empty-response steering must tell the caller to proceed: {result.content!r}"

    @pytest.mark.asyncio
    async def test_whitespace_only_response_is_soft(self, tmp_path):
        """Whitespace-only / refusal-shaped empty advice is treated as empty."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp(content="   \n  \t ")):
            result = await run_advisor(
                focus=None, model=None, config=cfg,
                tool_config=_tc_cfg(), callbacks=cb,
            )
        assert result.is_error, "Whitespace-only advice must be treated as empty (is_error)"

    @pytest.mark.asyncio
    async def test_timeout_is_soft_steering(self, tmp_path):
        """Per-consult timeout via asyncio.wait_for → is_error steering; turn survives."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()

        async def _slow(**kw):
            await asyncio.sleep(5)
            return _resp()

        with patch("openalph.tools.advisor.complete", new=_slow):
            result = await run_advisor(
                focus=None, model=None, config=cfg,
                tool_config=_tc_cfg(timeout=0.05), callbacks=cb,
            )
        assert isinstance(result, ToolResult) and result.is_error, \
            "Timeout must return a soft is_error ToolResult (no exception)"
        assert "advisor" in result.content.lower(), \
            f"Timeout steering should mention the advisor: {result.content!r}"


# ========================================================================
# Refusal handling (kdsn.198.10) — a provider content-policy refusal must
# surface as a DISTINCT, actionable message, not the generic empty-advice
# steering (which would be indistinguishable from a degenerate empty response,
# and invisible for sub consults that emit no Matrix notice).
# ========================================================================

class TestRefusalHandling:

    @pytest.mark.asyncio
    async def test_anthropic_refusal_surfaced_distinctly(self, tmp_path):
        """stop_reason='refusal' (Anthropic) → is_error message naming the refusal,
        NOT the generic 'no advice' empty-response message."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()
        refusal = Response(
            content="", model="claude-fable-5",
            usage=Usage(input_tokens=10, output_tokens=2),
            stop_reason="refusal",
        )
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=refusal):
            result = await run_advisor(
                focus=None, model="anthropic/claude-fable-5", config=cfg,
                tool_config=_tc_cfg(model="anthropic/claude-fable-5"), callbacks=cb,
            )
        assert result.is_error, "A refusal must be an is_error ToolResult"
        low = result.content.lower()
        assert "refus" in low, f"Refusal must be named as such: {result.content!r}"
        assert "no advice" not in low, \
            "A refusal must NOT collapse into the generic empty-advice message"

    @pytest.mark.asyncio
    async def test_openai_content_filter_surfaced_distinctly(self, tmp_path):
        """stop_reason='content_filter' (the OpenAI-family finish_reason) is
        treated the same as an Anthropic refusal — kdsn.198.11 enables
        openai-type advisors, so their refusal vocabulary must be covered too."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path, providers={"oai": _provider(key="oai", type="openai")})
        cb = _callbacks()
        filtered = Response(
            content="", model="gpt-strong",
            usage=Usage(input_tokens=10, output_tokens=0),
            stop_reason="content_filter",
        )
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=filtered):
            result = await run_advisor(
                focus=None, model="oai/gpt-strong", config=cfg,
                tool_config=_tc_cfg(model="oai/gpt-strong"), callbacks=cb,
            )
        assert result.is_error
        assert "refus" in result.content.lower(), \
            f"content_filter must surface as a refusal-class message: {result.content!r}"

    @pytest.mark.asyncio
    async def test_refusal_message_is_general_not_model_specific(self, tmp_path):
        """Operator scope note (kdsn.198.10): keep it GENERAL. The message names
        the refusing model and steers to the generic `model` parameter — it must
        NOT hardcode a specific model to switch to (no baked-in 'opus')."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()
        refusal = Response(
            content="", model="claude-fable-5",
            usage=Usage(input_tokens=10, output_tokens=2),
            stop_reason="refusal",
        )
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=refusal):
            result = await run_advisor(
                focus=None, model="anthropic/claude-fable-5", config=cfg,
                tool_config=_tc_cfg(model="anthropic/claude-fable-5"), callbacks=cb,
            )
        low = result.content.lower()
        assert "opus" not in low, \
            "Refusal steering must stay general — no hardcoded model name"
        assert "model" in low, \
            "Refusal steering should point at the `model` parameter"

    @pytest.mark.asyncio
    async def test_non_refusal_empty_still_generic(self, tmp_path):
        """A plain empty response (stop_reason='end_turn') is NOT a refusal —
        it must keep the generic empty-advice message, not the refusal one."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp(content="")):
            result = await run_advisor(
                focus=None, model=None, config=cfg,
                tool_config=_tc_cfg(), callbacks=cb,
            )
        assert result.is_error
        low = result.content.lower()
        assert "no advice" in low or "proceed" in low, \
            f"Non-refusal empty must keep the generic message: {result.content!r}"
        assert "refus" not in low, \
            "A non-refusal empty response must not claim a content-policy refusal"


# ========================================================================
# Thinking dropped (§4, §5)
# ========================================================================

class TestThinkingDropped:

    @pytest.mark.asyncio
    async def test_thinking_not_in_returned_result(self, tmp_path):
        """Advisor returns final text only — thinking content is NEVER in the ToolResult."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()
        resp = _resp(content="FINAL_ADVICE_ONLY",
                     thinking=[ThinkingBlock(thinking="SECRET_ADVISOR_THINKING", signature="sig==")])
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=resp):
            result = await run_advisor(
                focus=None, model=None, config=cfg,
                tool_config=_tc_cfg(), callbacks=cb,
            )
        assert not result.is_error
        assert "FINAL_ADVICE_ONLY" in result.content, "Final advice text must be returned"
        assert "SECRET_ADVISOR_THINKING" not in result.content, \
            "Advisor thinking content must be dropped from the returned ToolResult"


# ========================================================================
# Request shape (§5, §14) — mock the provider call, inspect what was sent
# ========================================================================

class TestRequestShape:

    @pytest.mark.asyncio
    async def test_advisor_system_prompt_present(self, tmp_path):
        """(a) An authored advisor system prompt is present (non-empty str) and is NOT
        merely the caller transcript. Exact wording is owned by the descriptions pass;
        cache_control on the system block is applied by the builder (see TestProviderExtension)."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            await run_advisor(focus=None, model=None, config=cfg,
                              tool_config=_tc_cfg(), callbacks=cb)
        system = mock_complete.call_args.kwargs["system"]
        assert isinstance(system, str) and len(system) >= 100, \
            "Advisor system prompt must be a substantial authored string"
        assert "TRANSCRIPT_USER_MARKER" not in system, \
            "The caller transcript must NOT be in the advisor system prompt (it goes in the user msg)"

    @pytest.mark.asyncio
    async def test_transcript_block_before_focus_with_breakpoint(self, tmp_path):
        """(b) user message = [transcript block WITH cache_control breakpoint][focus block
        AFTER the breakpoint], so focus variance never busts the transcript prefix."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path, providers={"anthropic": _provider()})
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            await run_advisor(focus="FOCUS_MARKER_Q should I refactor?",
                              model=None, config=cfg, tool_config=_tc_cfg(), callbacks=cb)
        messages = mock_complete.call_args.kwargs["messages"]
        assert len(messages) == 1 and messages[0]["role"] == "user", \
            "Advisor request is a single user message"
        blocks = _blocks(messages[0])
        tblock = _find_block(blocks, "TRANSCRIPT_USER_MARKER")
        fblock = _find_block(blocks, "FOCUS_MARKER_Q")
        assert tblock is not None, "Transcript block must be present in the user message"
        assert fblock is not None, "Focus block must be present in the user message"
        assert blocks.index(tblock) < blocks.index(fblock), \
            "Focus must come AFTER the transcript block (after the cache breakpoint)"
        assert tblock.get("cache_control") is not None, \
            "Transcript block must carry the cache_control breakpoint"
        assert "cache_control" not in fblock, \
            "Focus block must be AFTER the breakpoint (no cache_control) so focus variance is cheap"

    @pytest.mark.asyncio
    async def test_focus_optional_zero_required(self, tmp_path):
        """focus is optional — a consult with no focus still assembles a valid request."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            result = await run_advisor(focus=None, model=None, config=cfg,
                                       tool_config=_tc_cfg(), callbacks=cb)
        assert not result.is_error, "A focus-less consult must succeed (zero-required-params)"
        messages = mock_complete.call_args.kwargs["messages"]
        assert _find_block(_blocks(messages[0]), "TRANSCRIPT_USER_MARKER") is not None

    @pytest.mark.asyncio
    async def test_cache_control_present_for_anthropic_advisor(self, tmp_path):
        """(c) cache_control present when the advisor provider is Anthropic-type."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path, providers={"anthropic": _provider(type="anthropic")})
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            await run_advisor(focus="q", model="anthropic/claude-opus-4-8",
                              config=cfg, tool_config=_tc_cfg(model="anthropic/claude-opus-4-8"),
                              callbacks=cb)
        messages = mock_complete.call_args.kwargs["messages"]
        assert _has_any_cache_control(messages), \
            "Anthropic-type advisor must carry a cache_control breakpoint on the transcript block"

    @pytest.mark.asyncio
    async def test_cache_control_absent_for_openai_advisor(self, tmp_path):
        """(c) cache_control ABSENT/stripped when the advisor provider is openai-type."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path, providers={"oai": _provider(key="oai", type="openai")})
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            await run_advisor(focus="q", model="oai/gpt-strong",
                              config=cfg, tool_config=_tc_cfg(model="oai/gpt-strong"),
                              callbacks=cb)
        messages = mock_complete.call_args.kwargs["messages"]
        assert not _has_any_cache_control(messages), \
            "openai-type advisor must have cache_control stripped (unsupported there)"

    @pytest.mark.asyncio
    async def test_openai_advisor_content_is_flat_string(self, tmp_path):
        """kdsn.198.11: for an openai-type advisor provider the user message
        content must be a FLAT STRING, not a list of Anthropic-style text blocks
        (the openai wire format rejects the block-array for that field). The
        transcript + focus text must both still be present."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path, providers={"oai": _provider(key="oai", type="openai")})
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            await run_advisor(focus="FOCUS_MARKER_Q", model="oai/gpt-strong",
                              config=cfg, tool_config=_tc_cfg(model="oai/gpt-strong"),
                              callbacks=cb)
        messages = mock_complete.call_args.kwargs["messages"]
        assert len(messages) == 1 and messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, str), \
            f"openai-type advisor content must be a flat string, got {type(content).__name__}"
        assert "TRANSCRIPT_USER_MARKER" in content, "Transcript text must be present"
        assert "FOCUS_MARKER_Q" in content, "Focus text must be present"

    @pytest.mark.asyncio
    async def test_anthropic_advisor_content_stays_block_list(self, tmp_path):
        """Regression guard for kdsn.198.11: the anthropic path is UNCHANGED —
        still a list of content blocks carrying the cache_control breakpoint."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path, providers={"anthropic": _provider()})
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            await run_advisor(focus="q", model="anthropic/claude-opus-4-8",
                              config=cfg,
                              tool_config=_tc_cfg(model="anthropic/claude-opus-4-8"),
                              callbacks=cb)
        messages = mock_complete.call_args.kwargs["messages"]
        content = messages[0]["content"]
        assert isinstance(content, list), \
            "anthropic advisor content must stay a block list"
        assert _has_any_cache_control(messages), \
            "anthropic advisor must keep its cache_control breakpoint"

    @pytest.mark.asyncio
    async def test_max_tokens_and_thinking_forwarded(self, tmp_path):
        """Config max_tokens + thinking are forwarded to complete() (clamp is automatic)."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            await run_advisor(focus=None, model=None, config=cfg,
                              tool_config=_tc_cfg(max_tokens=8192, thinking="medium"),
                              callbacks=cb)
        kw = mock_complete.call_args.kwargs
        assert kw["max_tokens"] == 8192, "Configured max_tokens must be forwarded"
        assert kw["thinking"] == "medium", "Configured thinking effort must be forwarded"

    @pytest.mark.asyncio
    async def test_default_thinking_is_high(self, tmp_path):
        """kdsn.198.13: the advisor's fleet-wide CODE default thinking level is
        'high' (not 'medium') when a workspace advisor.toml does not set it.
        Reasoning depth is a property of the TOOL — 'advisors need to give sage
        advice.' Per-workspace TOML can still override up or down."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()
        tc = _tc_cfg()
        tc.pop("thinking", None)  # exercise the code default, not an explicit value
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            await run_advisor(focus=None, model=None, config=cfg,
                              tool_config=tc, callbacks=cb)
        assert mock_complete.call_args.kwargs["thinking"] == "high", \
            "Advisor default thinking level must be 'high' (kdsn.198.13)"

    @pytest.mark.asyncio
    async def test_cache_ttl_forwarded_to_complete(self, tmp_path):
        """The configured cache_ttl is forwarded to complete() (relies on §14 #1a passthrough)."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks()
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            await run_advisor(focus=None, model=None, config=cfg,
                              tool_config=_tc_cfg(cache_ttl="5m"), callbacks=cb)
        assert mock_complete.call_args.kwargs.get("cache_ttl") == "5m", \
            "Advisor must forward the configured cache_ttl to complete()"

    @pytest.mark.asyncio
    async def test_include_system_prompt_false_omits_caller_system(self, tmp_path):
        """include_system_prompt=false → caller system prompt NOT in the rendered transcript."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks(system="CALLER_SYSPROMPT_XYZ")
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            await run_advisor(focus=None, model=None, config=cfg,
                              tool_config=_tc_cfg(include_system_prompt=False), callbacks=cb)
        messages = mock_complete.call_args.kwargs["messages"]
        joined = "".join(b.get("text", "") for b in _blocks(messages[0]))
        assert "CALLER_SYSPROMPT_XYZ" not in joined, \
            "include_system_prompt=false must omit the caller system prompt from the transcript"

    @pytest.mark.asyncio
    async def test_include_system_prompt_true_includes_caller_system(self, tmp_path):
        """include_system_prompt=true (default) → caller system prompt IS in the transcript."""
        assert run_advisor is not None, NOT_IMPL
        cfg = _cfg(tmp_path)
        cb = _callbacks(system="CALLER_SYSPROMPT_XYZ")
        with patch("openalph.tools.advisor.complete", new_callable=AsyncMock,
                   return_value=_resp()) as mock_complete:
            await run_advisor(focus=None, model=None, config=cfg,
                              tool_config=_tc_cfg(include_system_prompt=True), callbacks=cb)
        messages = mock_complete.call_args.kwargs["messages"]
        joined = "".join(b.get("text", "") for b in _blocks(messages[0]))
        assert "CALLER_SYSPROMPT_XYZ" in joined, \
            "include_system_prompt=true must include the caller system prompt in the transcript"


# ========================================================================
# Provider extension (§14 amendment #1) — provider.py changes, pinned here
# ========================================================================

class TestProviderExtension:

    @pytest.mark.asyncio
    async def test_complete_forwards_cache_ttl_to_stream(self, tmp_path):
        """§14 #1a: complete() gains a cache_ttl passthrough forwarded to stream()."""
        assert run_advisor is not None, NOT_IMPL_BUNDLE
        cfg = _cfg(tmp_path)
        captured = {}

        async def _fake_stream(**kw):
            captured.update(kw)
            yield StreamEvent(type="done", model="m",
                              response=_resp(content="ok"))

        with patch("openalph.provider.stream", new=_fake_stream):
            await complete(config=cfg, system="s",
                           messages=[{"role": "user", "content": "hi"}],
                           model="anthropic/claude-opus-4-8", cache_ttl="5m")
        assert captured.get("cache_ttl") == "5m", \
            "complete() must forward caller-supplied cache_ttl to stream()"

    def test_builder_cache_ttl_reaches_kwargs(self):
        """§14 #1a: a caller cache_ttl='5m' lands on the Anthropic cache_control ttl."""
        assert run_advisor is not None, NOT_IMPL_BUNDLE
        kw = _build_anthropic_kwargs(
            api_model="claude-opus-4-8", system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None, max_tokens=1000, thinking_level="off",
            cache_ttl="5m",
        )
        assert kw["system"][0]["cache_control"]["ttl"] == "5m", \
            "cache_ttl must reach the system block's cache_control"

    def test_builder_system_block_has_cache_control(self):
        """The advisor system prompt is always cached (system block cache_control present)."""
        assert run_advisor is not None, NOT_IMPL_BUNDLE
        kw = _build_anthropic_kwargs(
            api_model="claude-opus-4-8", system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None, max_tokens=1000, thinking_level="off",
        )
        assert "cache_control" in kw["system"][0], \
            "System block must always carry cache_control (advisor prompt caching)"

    def test_builder_honors_caller_breakpoint_and_skips_autoapply(self):
        """§14 #1b: builder HONORS a caller-placed block-level cache_control and SKIPS
        its automatic last-block application when a caller breakpoint is present."""
        assert run_advisor is not None, NOT_IMPL_BUNDLE
        cc = {"type": "ephemeral", "ttl": "5m"}
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "RENDERED_TRANSCRIPT", "cache_control": cc},
            {"type": "text", "text": "FOCUS_AFTER_BREAKPOINT"},
        ]}]
        kw = _build_anthropic_kwargs(
            api_model="claude-opus-4-8", system="sys",
            provider_messages=msgs, provider_tools=None,
            max_tokens=1000, thinking_level="off", cache_ttl="5m",
        )
        content = kw["messages"][-1]["content"]
        tblock = _find_block(content, "RENDERED_TRANSCRIPT")
        fblock = _find_block(content, "FOCUS_AFTER_BREAKPOINT")
        assert tblock is not None and tblock.get("cache_control") is not None, \
            "Caller-placed cache_control on the transcript block must be honored"
        assert "cache_control" not in fblock, \
            "Auto last-block cache_control must be SKIPPED when a caller breakpoint is present"

    def test_builder_default_autoapply_unchanged_without_caller_breakpoint(self):
        """§14 #1b: default behavior unchanged — no caller breakpoint → builder applies
        cache_control to the last block of the last user message (backward-compatible)."""
        assert run_advisor is not None, NOT_IMPL_BUNDLE
        msgs = [{"role": "user", "content": "PLAIN_STRING_CONTENT"}]
        kw = _build_anthropic_kwargs(
            api_model="claude-opus-4-8", system="sys",
            provider_messages=msgs, provider_tools=None,
            max_tokens=1000, thinking_level="off", cache_ttl="5m",
        )
        content = kw["messages"][-1]["content"]
        assert isinstance(content, list) and content, "String content is wrapped into blocks"
        assert content[-1].get("cache_control") is not None, \
            "Without a caller breakpoint, the builder must still cache the last block (unchanged)"
