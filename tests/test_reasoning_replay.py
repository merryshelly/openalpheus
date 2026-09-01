"""Tests for reasoning_content replay on OpenAI-compatible providers (kdsn.241.2, kdsn.308).

Vendor requirement (Moonshot Kimi K2.6, Z.AI GLM-5.2, Fireworks): within a
multi-step tool-calling loop, the client MUST send stored reasoning_content back
on replayed assistant turns, or the model degenerates / errors. OpenAlph
historically STRIPPED thinking from all assistant messages before replay to
OpenAI-compatible providers. kdsn.241.2 restored it gated behind the opt-in
`reasoning_replay` quirk; kdsn.308 flips the default ON for every openai-type
provider (passback is self-gating — a model that emits no thinking has nothing
replayed), with the `no_reasoning_replay` quirk as the opt-out for strict
endpoints. The legacy `reasoning_replay` quirk is still tolerated (it now means
the default) but logs a warn-once deprecation.
"""
import json
import logging

import pytest

import openalph.provider as provider_module
from openalph.provider import (
    _convert_messages_for_openai,
    _convert_messages_for_provider,
    ToolCall,
)


@pytest.fixture(autouse=True)
def _clear_reasoning_replay_warn_latch():
    """Each test starts with the deprecation warn-once latch empty.

    getattr-tolerant: the latch set must exist in provider.py for the suite to
    be meaningful, but a missing attribute must not convert assertion reds into
    fixture-setup errors during red-first development."""
    latch = getattr(provider_module, "_REASONING_REPLAY_DEPRECATION_WARNED", None)
    if latch is not None:
        latch.clear()
    yield
    if latch is not None:
        latch.clear()


def _assistant_with_thinking(content="", thinking_texts=(), tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if thinking_texts:
        msg["thinking"] = [{"thinking": t, "signature": ""} for t in thinking_texts]
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


_DEPRECATION_FRAGMENT = "deprecated"
_CONFLICT_FRAGMENT = "conflict"


def _deprecation_warnings(caplog):
    return [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and _DEPRECATION_FRAGMENT in r.getMessage().lower()
    ]


def _conflict_warnings(caplog):
    return [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and _CONFLICT_FRAGMENT in r.getMessage().lower()
    ]


class TestReplayDefault:
    """Default (no quirk): kdsn.308-inverted behavior — thinking replayed as reasoning_content."""

    def test_thinking_replayed_on_plain_turn_by_default(self):
        msgs = [_assistant_with_thinking(content="hi", thinking_texts=["secret reasoning"])]
        out = _convert_messages_for_openai(msgs)  # reasoning_replay defaults True (kdsn.308)
        assert out[0]["reasoning_content"] == "secret reasoning"
        assert "thinking" not in out[0]
        assert out[0]["content"] == "hi"

    def test_thinking_replayed_with_tool_calls_by_default(self):
        tc = ToolCall(id="call_1", name="do_thing", input={"x": 1})
        msgs = [_assistant_with_thinking(content="", thinking_texts=["reasoning"], tool_calls=[tc])]
        out = _convert_messages_for_openai(msgs)
        assert out[0]["reasoning_content"] == "reasoning"
        assert "thinking" not in out[0]
        assert out[0]["tool_calls"][0]["function"]["name"] == "do_thing"


class TestReplayEnabled:
    """With reasoning_replay=True: thinking -> reasoning_content, thinking key removed."""

    def test_reasoning_content_set_on_plain_assistant(self):
        msgs = [_assistant_with_thinking(content="answer", thinking_texts=["step 1 then step 2"])]
        out = _convert_messages_for_openai(msgs, reasoning_replay=True)
        assert out[0]["reasoning_content"] == "step 1 then step 2"
        assert "thinking" not in out[0]
        assert out[0]["content"] == "answer"

    def test_reasoning_content_verbatim(self):
        """Reasoning text is preserved verbatim (vendor requirement)."""
        reasoning = 'I need to call capture_payment().\nCheck the TOCTOU race at line {42}.'
        msgs = [_assistant_with_thinking(content="", thinking_texts=[reasoning])]
        out = _convert_messages_for_openai(msgs, reasoning_replay=True)
        assert out[0]["reasoning_content"] == reasoning

    def test_reasoning_content_carried_into_tool_call_turn(self):
        """CRITICAL multi-step case: assistant turn WITH tool_calls must carry
        reasoning_content into the rebuilt OpenAI dict (this is the degeneration
        failure mode — reasoning stripped inside a tool loop)."""
        tc = ToolCall(id="call_1", name="submit_review", input={"verdict": "pass"})
        msgs = [_assistant_with_thinking(content="", thinking_texts=["deep reasoning"], tool_calls=[tc])]
        out = _convert_messages_for_openai(msgs, reasoning_replay=True)
        assert out[0]["reasoning_content"] == "deep reasoning"
        assert "thinking" not in out[0]
        assert out[0]["tool_calls"][0]["id"] == "call_1"
        assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"verdict": "pass"}

    def test_multiple_thinking_blocks_joined(self):
        msgs = [_assistant_with_thinking(content="x", thinking_texts=["block a", "block b"])]
        out = _convert_messages_for_openai(msgs, reasoning_replay=True)
        assert out[0]["reasoning_content"] == "block a\nblock b"

    def test_no_thinking_no_reasoning_content(self):
        """Assistant with no thinking: no reasoning_content field added, no crash."""
        msgs = [{"role": "assistant", "content": "plain"}]
        out = _convert_messages_for_openai(msgs, reasoning_replay=True)
        assert "reasoning_content" not in out[0]
        assert out[0]["content"] == "plain"

    def test_empty_thinking_text_omitted(self):
        """Empty/whitespace-only thinking blocks don't produce a reasoning_content field."""
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["", ""])]
        out = _convert_messages_for_openai(msgs, reasoning_replay=True)
        assert "reasoning_content" not in out[0]

    def test_user_and_tool_messages_untouched(self):
        tc = ToolCall(id="c1", name="f", input={})
        msgs = [
            {"role": "user", "content": "hello"},
            _assistant_with_thinking(content="", thinking_texts=["r"], tool_calls=[tc]),
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        out = _convert_messages_for_openai(msgs, reasoning_replay=True)
        assert out[0] == {"role": "user", "content": "hello"}
        # tool result role preserved
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert tool_msgs and tool_msgs[0]["content"] == "result"


class TestQuirkThreadingViaProvider:
    """_convert_messages_for_provider resolves reasoning_replay from the quirks list (kdsn.308:
    default ON, `no_reasoning_replay` opts out, legacy `reasoning_replay` tolerated + warn-once)."""

    def test_no_quirk_replays_default(self):
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["reasoning here"])]
        out = _convert_messages_for_provider(msgs, "openai", quirks=[])
        assert out[0]["reasoning_content"] == "reasoning here"
        assert "thinking" not in out[0]

    def test_quirks_none_replays_default(self):
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["r"])]
        out = _convert_messages_for_provider(msgs, "openai")
        assert out[0]["reasoning_content"] == "r"

    def test_legacy_quirk_replays_and_warns(self, caplog):
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["reasoning here"])]
        with caplog.at_level(logging.WARNING, logger="openalph.provider"):
            out = _convert_messages_for_provider(msgs, "openai", quirks=["reasoning_replay"])
        assert out[0]["reasoning_content"] == "reasoning here"
        deprecation = _deprecation_warnings(caplog)
        assert len(deprecation) == 1
        assert "reasoning_replay" in deprecation[0].getMessage()

    def test_optout_strips(self, caplog):
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["reasoning here"])]
        with caplog.at_level(logging.WARNING, logger="openalph.provider"):
            out = _convert_messages_for_provider(msgs, "openai", quirks=["no_reasoning_replay"])
        assert "reasoning_content" not in out[0]
        assert "thinking" not in out[0]

    def test_optout_no_warning(self, caplog):
        """Deliberate operator opt-out: stripped, and no deprecation warning emitted."""
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["r"])]
        with caplog.at_level(logging.WARNING, logger="openalph.provider"):
            out = _convert_messages_for_provider(msgs, "openai", quirks=["no_reasoning_replay"])
        assert "reasoning_content" not in out[0]
        assert _deprecation_warnings(caplog) == []
        assert _conflict_warnings(caplog) == []

    def test_legacy_warn_latches(self, caplog):
        """Warn-once: two conversions with the legacy quirk produce exactly one warning."""
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["r"])]
        with caplog.at_level(logging.WARNING, logger="openalph.provider"):
            _convert_messages_for_provider(msgs, "openai", quirks=["reasoning_replay"])
            _convert_messages_for_provider(msgs, "openai", quirks=["reasoning_replay"])
        assert len(_deprecation_warnings(caplog)) == 1

    def test_both_quirks_optout_wins_and_warns(self, caplog):
        """Both quirks present: opt-out wins (strip) + a DISTINCT conflict warning."""
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["reasoning here"])]
        with caplog.at_level(logging.WARNING, logger="openalph.provider"):
            out = _convert_messages_for_provider(
                msgs, "openai", quirks=["reasoning_replay", "no_reasoning_replay"])
        assert "reasoning_content" not in out[0]
        assert "thinking" not in out[0]
        conflict = _conflict_warnings(caplog)
        assert len(conflict) == 1
        # Distinct message: not the plain deprecation text
        assert _deprecation_warnings(caplog) == []

    def test_convert_openai_default_param_flipped(self):
        """_convert_messages_for_openai's reasoning_replay default flipped False -> True."""
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["r"])]
        assert _convert_messages_for_openai(msgs)[0]["reasoning_content"] == "r"
        # Explicit False still strips (strict endpoints can force the old behavior).
        out = _convert_messages_for_openai(msgs, reasoning_replay=False)
        assert "reasoning_content" not in out[0]
        assert "thinking" not in out[0]

    def test_reasoning_replay_quirk_ignored_for_anthropic(self):
        """Anthropic path has its own thinking mechanism; the openai-only quirk
        must not corrupt Anthropic conversion (thinking preserved as blocks there)."""
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["r"])]
        out = _convert_messages_for_provider(msgs, "anthropic", quirks=["reasoning_replay"])
        # Anthropic conversion does not produce a flat reasoning_content field.
        assert all("reasoning_content" not in m for m in out)
