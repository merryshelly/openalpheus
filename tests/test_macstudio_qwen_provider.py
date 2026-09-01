"""Tests for macstudio-qwen (local llama.cpp qwen38 rig) reasoning_effort
support (2026-09-01, SB steering: server-side medium pin is no good — OA
passthrough).

Wire evidence (curl probes 2026-09-01 vs http://10.0.20.104:8010, llama.cpp
4df29be-era build, served model qwen38-coder, same prompt per level):
  none -> 200, zero reasoning content (server intercept: enable_thinking=false)
  low/medium/xhigh -> 200 with reasoning (tier ranking uncharacterized: n=1
      probe at temp 1.0 too noisy; legality is what the map encodes)
  high, max, garbage -> 500 LOUD (Jinja raise_exception; the template's own
      error text: "Supported types are xhigh (default), medium, and low")

Server source (tools/server/server-common.cpp, this build): TOP-LEVEL
reasoning_effort parsed natively on the OAI path; non-empty values are
written into chat_template_kwargs, overriding the CLI
--chat-template-kwargs pin per-key; "none" sets enable_thinking=false and
erases any pinned kwarg.

Problem under test: provider_key="macstudio-qwen" hit no effort branch in
_build_openai_kwargs (not _supports_reasoning_extra — and that gate emits
the nested reasoning.effort form llama.cpp ignores anyway) -> OA sent
nothing and the server-side pin was the only effort control (kdsn.271 bug
class: silent server-default override of operator intent). Map:
off->none, low/medium/xhigh 1:1, max ceiling->xhigh, high DOWN->medium,
each lossy remap warn-once — never up.
"""

import logging

import pytest

from openalph import provider as provider_module
from openalph.provider import _build_openai_kwargs


class TestMacstudioQwenEffortMapping:
    """Always-explicit top-level reasoning_effort for
    provider_key="macstudio-qwen". Same shape as the blackwell kdsn.301
    branch; the legality set differs per the template's own error text."""

    @pytest.fixture(autouse=True)
    def _clear_warned(self):
        """Isolate the warn-once set between tests (house convention:
        _BLACKWELL_EFFORT_WARNED in test_blackwell_provider.py)."""
        provider_module._MACSTUDIO_QWEN_EFFORT_WARNED.clear()
        yield
        provider_module._MACSTUDIO_QWEN_EFFORT_WARNED.clear()

    def _args(self, level, **overrides):
        args = dict(
            api_model="qwen38-coder",
            system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None,
            max_tokens=1024,
            thinking_level=level,
            quirks=["reasoning_replay"],
            provider_key="macstudio-qwen",
        )
        args.update(overrides)
        return args

    def _effort(self, kw):
        return kw.get("extra_body", {}).get("reasoning_effort")

    # --- discriminating tests ---

    def test_off_maps_to_none(self):
        """off must send reasoning_effort='none' explicitly — the server
        intercepts 'none' (enable_thinking=false + pinned-kwarg erase), so
        thinking-off works regardless of any server-side pin."""
        kw = _build_openai_kwargs(**self._args("off"))
        assert self._effort(kw) == "none"

    @pytest.mark.parametrize("level", ["low", "medium", "xhigh"])
    def test_native_levels_passthrough(self, level):
        """Template-native vocabulary per its own error text ("Supported
        types are xhigh (default), medium, and low"): 1:1 passthrough.
        Unlike blackwell's backend, xhigh is native here and does NOT
        collapse."""
        kw = _build_openai_kwargs(**self._args(level))
        assert self._effort(kw) == level

    def test_high_maps_to_medium_with_warning(self, caplog):
        """'high' is a 500 (Jinja raise_exception) on this template. Collapse
        DOWN to medium + warn-once — NEVER up to xhigh: mapping up would
        silently multiply reasoning volume against operator intent, and SB
        observed xhigh overthinking on this model (the reason the medium
        pin existed at all)."""
        with caplog.at_level(logging.WARNING):
            kw = _build_openai_kwargs(**self._args("high"))
        assert self._effort(kw) == "medium"
        assert any(
            "high" in r.message and "reasoning_effort" in r.message
            for r in caplog.records
        ), f"expected remap warning, got: {[r.message for r in caplog.records]}"

    def test_max_maps_to_xhigh_with_warning(self, caplog):
        """'max' is a 500 here (accepted on synthetic's qwen3.8 backend —
        per-backend contract, not model-card truth). Ceiling-map to xhigh
        (template top) + warn-once, mirroring blackwell max->xhigh."""
        with caplog.at_level(logging.WARNING):
            kw = _build_openai_kwargs(**self._args("max"))
        assert self._effort(kw) == "xhigh"
        assert any(
            "max" in r.message and "reasoning_effort" in r.message
            for r in caplog.records
        ), f"expected remap warning, got: {[r.message for r in caplog.records]}"

    def test_remap_warns_once_per_level(self, caplog):
        """Warn-once per OA level, not warn-never / warn-spam (fleet log
        hygiene). high x3 = exactly 1 warning; high+max = 2 distinct."""
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                _build_openai_kwargs(**self._args("high"))
            for _ in range(3):
                _build_openai_kwargs(**self._args("max"))
        warns = [r for r in caplog.records
                 if "reasoning_effort" in r.message and r.levelno >= logging.WARNING]
        assert len(warns) == 2, f"expected exactly 2 warnings, got {len(warns)}"

    def test_no_warning_for_exact_levels(self, caplog):
        """1:1 mappings (incl. xhigh passthrough) never warn; off->none is a
        disable-alias, not a tier drop."""
        with caplog.at_level(logging.WARNING):
            for level in ("off", "low", "medium", "xhigh"):
                _build_openai_kwargs(**self._args(level))
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warns == [], f"unexpected warnings: {[r.message for r in warns]}"

    def test_no_nested_reasoning_object(self):
        """macstudio-qwen takes TOP-LEVEL reasoning_effort, never OpenRouter's
        nested reasoning.effort. The _supports_reasoning_extra gate (key
        'macstudio') must not capture 'macstudio-qwen' — distinct keys, and
        the nested form is ignored by llama.cpp on the OAI path."""
        kw = _build_openai_kwargs(**self._args("medium"))
        assert "reasoning" not in kw.get("extra_body", {})

    def test_reasoning_coexists_with_routing(self):
        """effort and a routing dict share extra_body (openrouter-style
        `provider` key must not clobber reasoning_effort)."""
        kw = _build_openai_kwargs(**self._args(
            "low", routing={"some": "route"}))
        assert kw["extra_body"]["reasoning_effort"] == "low"
        assert kw["extra_body"]["provider"] == {"some": "route"}

    def test_no_anthropic_thinking_param(self):
        """The openai-type path must never emit an Anthropic `thinking` kwarg."""
        kw = _build_openai_kwargs(**self._args("xhigh"))
        assert "thinking" not in kw
        assert "thinking" not in kw.get("extra_body", {})

    @pytest.mark.parametrize("level", ["garbage", "", None])
    def test_unknown_level_falls_back_to_medium_no_warning(self, caplog, level):
        """Spec'd fallback: out-of-enum values (typo'd advisor TOML,
        corrupted restore data) map to 'medium' silently — never sent
        verbatim, never warned (warnings are reserved for the two valid
        lossy tier-drops). Mirrors the blackwell NOTE-3 convention."""
        with caplog.at_level(logging.WARNING):
            kw = _build_openai_kwargs(**self._args(level))
        assert self._effort(kw) == "medium"
        warns = [r for r in caplog.records if "reasoning_effort" in r.message]
        assert warns == [], f"unexpected warnings: {[r.message for r in warns]}"

    # --- guards (expected green already; their job is staying green) ---

    def test_dsv4_prefix_branch_not_triggered(self):
        """Collision guard: macstudio-qwen requests must not hit the DSv4F
        text-prefix branch (different provider key AND different model
        string) — the system message must pass through unmutated while the
        effort param is still sent."""
        kw = _build_openai_kwargs(**self._args("xhigh"))
        assert kw["messages"][0]["content"] == "sys"
        assert kw["extra_body"]["reasoning_effort"] == "xhigh"

    def test_macstudio_dsv4_unchanged(self):
        """Guard: DSv4-flash on 'macstudio' keeps its prefix + top-level-none
        behavior — the macstudio-qwen branch must not preempt the
        model-name gate."""
        kw = _build_openai_kwargs(
            api_model="macstudio/deepseek-v4-flash", system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None, max_tokens=1024,
            thinking_level="off", quirks=[], provider_key="macstudio",
        )
        assert kw["extra_body"]["reasoning_effort"] == "none"
        assert kw["messages"][0]["content"] == "sys"
