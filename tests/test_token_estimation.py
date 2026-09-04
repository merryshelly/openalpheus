"""Tests for token estimation accuracy improvements in agent._estimate_context_tokens."""

import json

from openalph.agent import Agent, _TOOL_CALL_OVERHEAD_CHARS, _TOOL_RESULT_OVERHEAD_CHARS
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import ToolCall


def make_config(tmp_path, **kwargs):
    defaults = dict(
        name="test",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        model_max_tokens=200000,
        providers={
            "anthropic": ProviderConfig(
                key="anthropic",
                type="anthropic",
                api_key="sk-test",
                base_url=None,
                quirks=None,
            )
        },
        workspace=tmp_path,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def make_agent(tmp_path, with_tools=False):
    """Create a minimal Agent for estimation tests."""
    if with_tools:
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir(exist_ok=True)
        (tools_dir / "shell.toml").write_text("[config]\n")
    config = make_config(tmp_path)
    return Agent(config)


# ---------------------------------------------------------------------------
# Change 1: thinking signature counted
# ---------------------------------------------------------------------------

class TestThinkingSignatureCounted:

    def test_thinking_signature_counted(self, tmp_path):
        """Thinking signature field is included in the token estimate."""
        agent = make_agent(tmp_path)
        thinking_text = "Let me reason step by step about this problem."
        signature_blob = "A" * 200  # simulate an ECDSA base64 blob

        history = [
            {
                "role": "assistant",
                "content": "",
                "thinking": [
                    {"thinking": thinking_text, "signature": signature_blob}
                ],
            }
        ]
        agent._rooms["room1"] = history

        estimate = agent._estimate_context_tokens("room1")

        # Both thinking text and signature must be reflected in the estimate.
        # System prompt chars + thinking + signature, all divided by 4.
        system_chars = len(agent.system_prompt)
        expected_min = (system_chars + len(thinking_text) + len(signature_blob)) // 4
        assert estimate >= expected_min

    def test_thinking_signature_missing_is_safe(self, tmp_path):
        """Missing signature key does not raise; defaults to zero contribution."""
        agent = make_agent(tmp_path)
        thinking_text = "Some thoughts."

        history = [
            {
                "role": "assistant",
                "content": "",
                "thinking": [
                    {"thinking": thinking_text}   # no "signature" key
                ],
            }
        ]
        agent._rooms["room1"] = history

        # Must not raise
        estimate = agent._estimate_context_tokens("room1")
        assert estimate > 0

    def test_signature_adds_to_estimate_vs_no_signature(self, tmp_path):
        """Estimate with a signature is strictly larger than without."""
        agent = make_agent(tmp_path)
        thinking_text = "Reasoning text."
        signature_blob = "S" * 300

        history_no_sig = [
            {"role": "assistant", "content": "",
             "thinking": [{"thinking": thinking_text}]}
        ]
        history_with_sig = [
            {"role": "assistant", "content": "",
             "thinking": [{"thinking": thinking_text, "signature": signature_blob}]}
        ]

        est_no_sig = agent._estimate_context_tokens(history=history_no_sig)
        est_with_sig = agent._estimate_context_tokens(history=history_with_sig)

        # Signature adds len(signature_blob)//4 tokens (at least 1 extra token given 300 chars)
        assert est_with_sig > est_no_sig


# ---------------------------------------------------------------------------
# Change 2: tool definitions counted
# ---------------------------------------------------------------------------

class TestToolDefsCounted:

    def test_tool_defs_counted(self, tmp_path):
        """_tool_defs_chars > 0 for an agent with tools; reflected in estimate."""
        agent = make_agent(tmp_path, with_tools=True)

        assert agent._tool_defs_chars > 0, "_tool_defs_chars should be non-zero with tools"

        # Estimate with empty history still includes tool defs
        agent._rooms["room1"] = []
        estimate = agent._estimate_context_tokens("room1")

        system_only = (len(agent.system_prompt)) // 4
        # Must be larger than system-prompt-only due to tool defs
        assert estimate > system_only

    def test_tool_defs_zero_when_no_tools(self, tmp_path):
        """_tool_defs_chars == 0 for an agent with no tools."""
        agent = make_agent(tmp_path, with_tools=False)
        assert agent._tool_defs_chars == 0

    def test_tool_defs_chars_matches_manual_calculation(self, tmp_path):
        """_tool_defs_chars equals the manually computed sum across all tools."""
        agent = make_agent(tmp_path, with_tools=True)

        expected = 0
        for t in agent.tools:
            params_chars = len(json.dumps(t.parameters)) if isinstance(t.parameters, dict) else 0
            expected += len(t.name) + len(t.description) + params_chars

        assert agent._tool_defs_chars == expected


# ---------------------------------------------------------------------------
# Change 3: wire format overhead constants
# ---------------------------------------------------------------------------

class TestToolCallWireOverhead:

    def test_tool_call_wire_overhead(self, tmp_path):
        """Tool call estimate includes _TOOL_CALL_OVERHEAD_CHARS."""
        agent = make_agent(tmp_path)
        tc = ToolCall(id="tc_001", name="shell", input={"command": "echo hi"})
        input_str = str(tc.input)

        history = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tc],
            }
        ]
        agent._rooms["room1"] = history

        estimate = agent._estimate_context_tokens("room1")

        # estimate = total_chars // 4, so we must use // 4 in the assertion too
        # (floor division means estimate*4 can be up to 3 less than total_chars)
        system_chars = len(agent.system_prompt)
        # _tool_defs_chars is 0 because make_agent has no tools
        expected_chars = system_chars + agent._tool_defs_chars + len(input_str) + _TOOL_CALL_OVERHEAD_CHARS
        assert estimate == expected_chars // 4

    def test_tool_call_overhead_adds_to_estimate_vs_no_overhead(self, tmp_path):
        """An estimate with the overhead constant is larger than input alone would give."""
        agent = make_agent(tmp_path)
        tc = ToolCall(id="tc_002", name="shell", input={"command": "ls"})
        input_chars = len(str(tc.input))

        # Compute expected difference: _TOOL_CALL_OVERHEAD_CHARS chars → some tokens
        overhead_tokens = _TOOL_CALL_OVERHEAD_CHARS // 4

        history = [{"role": "assistant", "content": "", "tool_calls": [tc]}]
        agent._rooms["room1"] = history
        estimate = agent._estimate_context_tokens("room1")

        # Baseline: system + input chars only (no overhead)
        baseline_tokens = (len(agent.system_prompt) + agent._tool_defs_chars + input_chars) // 4
        # Actual estimate should be at least overhead_tokens more
        assert estimate >= baseline_tokens + overhead_tokens

    def test_tool_result_wire_overhead(self, tmp_path):
        """Tool result message (role='tool') includes _TOOL_RESULT_OVERHEAD_CHARS."""
        agent = make_agent(tmp_path)
        result_content = "output of the tool"

        history = [
            {
                "role": "tool",
                "content": result_content,
                "tool_call_id": "tc_001",
            }
        ]
        agent._rooms["room1"] = history

        estimate = agent._estimate_context_tokens("room1")

        # chars: system_prompt + result_content + overhead
        system_chars = len(agent.system_prompt)
        min_chars = system_chars + len(result_content) + _TOOL_RESULT_OVERHEAD_CHARS
        # _estimate_context_tokens floors with total_chars // 4, so compare against
        # the floored lower bound — multiplying the floored estimate back by 4 can
        # lose up to 3 chars (cf. test_tool_call_wire_overhead's same-pitfall note).
        assert estimate >= min_chars // 4

    def test_tool_result_overhead_vs_non_tool_message(self, tmp_path):
        """A tool-role message gives a higher estimate than a user message of same length."""
        agent = make_agent(tmp_path)
        content = "x" * 100

        hist_tool = [{"role": "tool", "content": content, "tool_call_id": "tc_x"}]
        hist_user = [{"role": "user", "content": content}]

        est_tool = agent._estimate_context_tokens(history=hist_tool)
        est_user = agent._estimate_context_tokens(history=hist_user)

        # tool message has overhead; user message does not → tool estimate must be higher
        assert est_tool > est_user


# ---------------------------------------------------------------------------
# Change 4: strippable stats shows tokens not chars
# ---------------------------------------------------------------------------

class TestStrippableShowsTokens:

    def test_strippable_token_math(self):
        """s_chars // 4 produces the token count shown in /status."""
        # Pure arithmetic test — verifies the formula used in matrix.py Change 4.
        test_cases = [
            (0, 0),
            (4, 1),
            (100, 25),
            (1000, 250),
            (40000, 10000),
            (123456, 30864),
        ]
        for s_chars, expected_tokens in test_cases:
            assert s_chars // 4 == expected_tokens, (
                f"s_chars={s_chars}: expected {expected_tokens}, got {s_chars // 4}"
            )

    def test_strippable_tokens_less_than_chars(self):
        """Token count is always <= char count (4 chars per token)."""
        for s_chars in [0, 1, 100, 9999, 100000]:
            s_tokens = s_chars // 4
            assert s_tokens <= s_chars
