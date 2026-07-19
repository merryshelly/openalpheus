"""Tests for reasoning_content replay on OpenAI-compatible providers (kdsn.241.2).

Vendor requirement (Moonshot Kimi K2.6, Z.AI GLM-5.2, Fireworks): within a
multi-step tool-calling loop, the client MUST send stored reasoning_content back
on replayed assistant turns, or the model degenerates / errors. OpenAlph
historically STRIPPED thinking from all assistant messages before replay to
OpenAI-compatible providers. This restores it, gated behind a per-provider
`reasoning_replay` quirk (opt-in; strict endpoints that reject the field keep
the strip behavior by default).
"""
import json
import pytest

from openalph.provider import (
    _convert_messages_for_openai,
    _convert_messages_for_provider,
    ToolCall,
)


def _assistant_with_thinking(content="", thinking_texts=(), tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if thinking_texts:
        msg["thinking"] = [{"thinking": t, "signature": ""} for t in thinking_texts]
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


class TestReplayDisabledByDefault:
    """Default (no quirk): current behavior preserved — thinking stripped, no reasoning_content."""

    def test_thinking_stripped_when_replay_off(self):
        msgs = [_assistant_with_thinking(content="hi", thinking_texts=["secret reasoning"])]
        out = _convert_messages_for_openai(msgs)  # reasoning_replay defaults False
        assert "thinking" not in out[0]
        assert "reasoning_content" not in out[0]
        assert out[0]["content"] == "hi"

    def test_thinking_stripped_with_tool_calls_when_replay_off(self):
        tc = ToolCall(id="call_1", name="do_thing", input={"x": 1})
        msgs = [_assistant_with_thinking(content="", thinking_texts=["reasoning"], tool_calls=[tc])]
        out = _convert_messages_for_openai(msgs)
        assert "thinking" not in out[0]
        assert "reasoning_content" not in out[0]
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
    """_convert_messages_for_provider derives reasoning_replay from the quirks list."""

    def test_quirk_enables_replay(self):
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["reasoning here"])]
        out = _convert_messages_for_provider(msgs, "openai", quirks=["reasoning_replay"])
        assert out[0]["reasoning_content"] == "reasoning here"

    def test_no_quirk_strips(self):
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["reasoning here"])]
        out = _convert_messages_for_provider(msgs, "openai", quirks=[])
        assert "reasoning_content" not in out[0]
        assert "thinking" not in out[0]

    def test_quirks_none_defaults_to_strip(self):
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["r"])]
        out = _convert_messages_for_provider(msgs, "openai")
        assert "reasoning_content" not in out[0]

    def test_reasoning_replay_quirk_ignored_for_anthropic(self):
        """Anthropic path has its own thinking mechanism; the openai-only quirk
        must not corrupt Anthropic conversion (thinking preserved as blocks there)."""
        msgs = [_assistant_with_thinking(content="a", thinking_texts=["r"])]
        out = _convert_messages_for_provider(msgs, "anthropic", quirks=["reasoning_replay"])
        # Anthropic conversion does not produce a flat reasoning_content field.
        assert all("reasoning_content" not in m for m in out)
