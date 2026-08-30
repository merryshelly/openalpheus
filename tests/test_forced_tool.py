"""Forced-tool provider support (workspace-kdsn.304).

Spec: memory/projects/openalph/specs/kdsn.304-forced-tool-spec.md

Provider-generic forced-tool contract: name-forced tool_choice +
strict/additionalProperties:false schema threading on all three wire
surfaces Stigmergy needs (Anthropic, OpenAI-compatible incl. blackwell
SGLang, Synthetic), plus the opt-in "hardened" key-bearing-call transport
flag (no redirect following, no proxy-env inheritance, bounded read).

Test conventions mirror test_blackwell_provider.py (kwargs-builder unit
tests, direct import of private builders, provider_module import) and
test_provider_tools.py (factories, SAMPLE_TOOL, stream mocks). No real
network calls anywhere in this suite.

Prove-can-fail: against unpatched provider.py, the discriminating tests
(TestAnthropicToolChoice / TestOpenAIToolChoice emission + requires-tools,
TestStrictSerialization, TestHardenedClient, TestHardenedBoundedRead) are
RED (missing param / missing keys / missing hardened construction); the
guards (TestParity, TestDefensiveId, test_tool_choice_absent_when_none)
are GREEN already — that is their job.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from openalph import provider as provider_module
from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import (
    _build_anthropic_kwargs,
    _build_openai_kwargs,
    _client_cache,
    _convert_tools_for_provider,
    _get_client,
    complete,
)
from openalph.tools import ToolDef

SAMPLE_TOOL = ToolDef(
    name="shell",
    description="Execute a shell command",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
        },
        "required": ["command"],
    },
    config={},
)


def make_provider(key="anthropic", type="anthropic", api_key="sk-test",
                  base_url=None, quirks=None, timeout=600.0):
    return ProviderConfig(
        key=key,
        type=type,
        api_key=api_key,
        base_url=base_url,
        quirks=quirks or [],
        timeout=timeout,
    )


def make_config(**kwargs):
    defaults = {
        "name": "test",
        "default_model": "anthropic/claude-sonnet-5",
        "max_tokens": 8192,
        "providers": {"anthropic": make_provider(key="anthropic")},
        "workspace": Path("/tmp/test"),
        "max_iterations": 25,
        "truncation_limit": 50000,
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


# --- OpenAI stream mocks (pattern from test_provider_tools.py) ---

class MockOpenAIStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for chunk in self._chunks:
            yield chunk


def _openai_usage_chunk(prompt_tokens=100, completion_tokens=50):
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = MagicMock()
    chunk.usage.prompt_tokens = prompt_tokens
    chunk.usage.completion_tokens = completion_tokens
    return chunk


def _openai_tool_delta_chunk(tool_id, name, arguments, finish_reason=None):
    """One stream chunk carrying a single tool-call delta."""
    chunk = MagicMock()
    chunk.id = None
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = None
    tc = MagicMock()
    tc.index = 0
    tc.id = tool_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    chunk.choices[0].delta.tool_calls = [tc]
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = None
    return chunk


# ===========================================================================
# Parity (guards — green BEFORE implementation; their job is staying green)
# ===========================================================================

class TestParity:

    def test_anthropic_kwargs_byte_identical_without_new_params(self):
        """_build_anthropic_kwargs with today's exact arg set (no
        tool_choice) returns a dict equal to today's byte-for-byte."""
        kw = _build_anthropic_kwargs(
            api_model="claude-sonnet-5",
            system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=[
                {"name": "shell", "description": "d", "input_schema": {"type": "object"}},
            ],
            max_tokens=1024,
            thinking_level="off",
            model_max_tokens=200000,
            temperature=None,
            top_p=None,
            cache_ttl="1h",
        )
        assert kw == {
            "model": "claude-sonnet-5",
            "system": [
                {"type": "text", "text": "sys",
                 "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi",
                     "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                ]},
            ],
            "max_tokens": 1024,
            "tools": [
                {"name": "shell", "description": "d", "input_schema": {"type": "object"}},
            ],
        }
        assert "tool_choice" not in kw
        # No strict key creep on the tools.
        assert "strict" not in kw["tools"][0]

    def test_openai_kwargs_byte_identical_without_new_params(self):
        """_build_openai_kwargs on the kdsn.301 blackwell branch (off level)
        returns a dict equal to today's byte-for-byte."""
        kw = _build_openai_kwargs(
            api_model="qwen38-27b-fp8",
            system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=None,
            max_tokens=1024,
            thinking_level="off",
            quirks=["reasoning_replay"],
            provider_key="blackwell",
        )
        assert kw == {
            "model": "qwen38-27b-fp8",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 1024,
            "temperature": 1.0,
            "top_p": 0.95,
            "extra_body": {"reasoning_effort": "none"},
        }
        assert "tool_choice" not in kw

    def test_convert_tools_identical_without_strict(self):
        """_convert_tools_for_provider without strict returns the current
        docstring shapes exactly — no strict key, no additionalProperties,
        and the caller's schema dict is not mutated."""
        anthropic_tools = _convert_tools_for_provider([SAMPLE_TOOL], "anthropic")
        assert anthropic_tools == [{
            "name": "shell",
            "description": "Execute a shell command",
            "input_schema": SAMPLE_TOOL.parameters,
        }]
        assert "strict" not in anthropic_tools[0]

        openai_tools = _convert_tools_for_provider([SAMPLE_TOOL], "openai")
        assert openai_tools == [{
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Execute a shell command",
                "parameters": SAMPLE_TOOL.parameters,
            },
        }]
        # SAMPLE_TOOL.parameters itself must not be mutated by conversion.
        assert "additionalProperties" not in SAMPLE_TOOL.parameters


# ===========================================================================
# tool_choice emission (discriminating — red pre-implementation)
# ===========================================================================

class TestAnthropicToolChoice:

    def test_tool_choice_emitted_anthropic_form(self):
        """AC1: tool_choice="submit_verdict" emits the Anthropic wire form
        {"type": "tool", "name": X} (critic_client.py:297-298 shape)."""
        tools = _convert_tools_for_provider([SAMPLE_TOOL], "anthropic")
        kw = _build_anthropic_kwargs(
            api_model="claude-sonnet-5",
            system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=tools,
            max_tokens=1024,
            thinking_level="off",
            model_max_tokens=200000,
            temperature=None,
            top_p=None,
            cache_ttl="1h",
            tool_choice="submit_verdict",
        )
        assert kw["tool_choice"] == {"type": "tool", "name": "submit_verdict"}
        # Strict off by default: no key creep on the tools.
        assert "strict" not in kw["tools"][0]

    def test_tool_choice_requires_tools_anthropic(self):
        """AC5: tool_choice without tools raises ValueError from the builder
        (before any network call)."""
        with pytest.raises(ValueError, match="tool_choice requires tools"):
            _build_anthropic_kwargs(
                api_model="claude-sonnet-5",
                system="sys",
                provider_messages=[{"role": "user", "content": "hi"}],
                provider_tools=None,
                max_tokens=1024,
                thinking_level="off",
                model_max_tokens=200000,
                temperature=None,
                top_p=None,
                cache_ttl="1h",
                tool_choice="submit_verdict",
            )


class TestOpenAIToolChoice:

    @pytest.mark.parametrize("provider_key", ["blackwell", "synthetic", "openrouter"])
    def test_tool_choice_emitted_openai_form(self, provider_key):
        """AC2: every openai-type provider (incl. blackwell/synthetic) emits
        the OpenAI structured-outputs form."""
        tools = _convert_tools_for_provider([SAMPLE_TOOL], "openai")
        kw = _build_openai_kwargs(
            api_model="parity-test-model",
            system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=tools,
            max_tokens=1024,
            thinking_level="off",
            quirks=[],
            provider_key=provider_key,
            tool_choice="submit_verdict",
        )
        assert kw["tool_choice"] == {
            "type": "function",
            "function": {"name": "submit_verdict"},
        }
        # Strict off by default: no key creep on the tools.
        assert "strict" not in kw["tools"][0]["function"]
        assert "additionalProperties" not in kw["tools"][0]["function"]["parameters"]

    def test_tool_choice_absent_when_none(self):
        """Guard (green pre-implementation): omitted tool_choice emits NO
        tool_choice key."""
        kw = _build_openai_kwargs(
            api_model="parity-test-model",
            system="sys",
            provider_messages=[{"role": "user", "content": "hi"}],
            provider_tools=_convert_tools_for_provider([SAMPLE_TOOL], "openai"),
            max_tokens=1024,
            thinking_level="off",
            quirks=[],
            provider_key="blackwell",
        )
        assert "tool_choice" not in kw

    def test_tool_choice_requires_tools_openai(self):
        """AC5: tool_choice without tools raises ValueError from the builder."""
        with pytest.raises(ValueError, match="tool_choice requires tools"):
            _build_openai_kwargs(
                api_model="parity-test-model",
                system="sys",
                provider_messages=[{"role": "user", "content": "hi"}],
                provider_tools=None,
                max_tokens=1024,
                thinking_level="off",
                quirks=[],
                provider_key="blackwell",
                tool_choice="submit_verdict",
            )


# ===========================================================================
# strict serialization (discriminating — red pre-implementation)
# ===========================================================================

class TestStrictSerialization:

    def test_anthropic_strict_true_on_tool(self):
        """AC3: strict=True adds "strict": True (top level, sibling of
        name/description/input_schema) to every tool; schema untouched."""
        params = {
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"],
        }
        tool = ToolDef(name="submit_verdict", description="v", parameters=params, config={})
        tools = _convert_tools_for_provider([tool], "anthropic", strict=True)
        assert tools[0]["strict"] is True
        assert set(tools[0].keys()) == {"name", "description", "input_schema", "strict"}
        # Schema untouched — caller-supplied additionalProperties stays the
        # single source of truth (portable house convention).
        assert tools[0]["input_schema"] is params
        assert "additionalProperties" not in params

    def test_openai_strict_true_and_additional_properties(self):
        """AC3: strict=True adds function["strict"] and guarantees
        additionalProperties:false at the schema root ONLY when the caller
        didn't already set it (injection is idempotent, non-mutating)."""
        tool = ToolDef(
            name="submit_verdict", description="v",
            parameters={"type": "object", "properties": {}}, config={},
        )
        tools = _convert_tools_for_provider([tool], "openai", strict=True)
        fn = tools[0]["function"]
        assert fn["strict"] is True
        assert fn["parameters"]["additionalProperties"] is False
        # The caller's original schema dict must not be mutated in place.
        assert "additionalProperties" not in tool.parameters

        # Untouched when the caller already set it (idempotent, no override).
        tool2 = ToolDef(
            name="submit_verdict", description="v",
            parameters={
                "type": "object", "properties": {},
                "additionalProperties": {"keep": "caller-value"},
            },
            config={},
        )
        tools2 = _convert_tools_for_provider([tool2], "openai", strict=True)
        assert tools2[0]["function"]["parameters"]["additionalProperties"] == {
            "keep": "caller-value",
        }

    def test_strict_false_no_key_creep(self):
        """AC4: strict=False (default) keeps neither key on either surface."""
        tool = ToolDef(
            name="submit_verdict", description="v",
            parameters={"type": "object", "properties": {}}, config={},
        )
        anthropic_tools = _convert_tools_for_provider([tool], "anthropic", strict=False)
        assert "strict" not in anthropic_tools[0]

        openai_tools = _convert_tools_for_provider([tool], "openai", strict=False)
        fn = openai_tools[0]["function"]
        assert "strict" not in fn
        assert "additionalProperties" not in fn["parameters"]


# ===========================================================================
# Hardened client (discriminating — red pre-implementation)
# ===========================================================================

class TestHardenedClient:

    def test_hardened_and_unhardened_clients_are_distinct(self):
        """AC8/AC9: hardened calls get their OWN cached client — two
        constructions, two distinct cache keys; the unhardened constructor
        call keeps today's exact kwargs (no http_client), and the pooled
        unhardened client is never touched by a hardened call."""
        prov = make_provider(key="anthropic", type="anthropic", api_key="sk-test")
        # Each SDK construction yields its OWN client object (side_effect),
        # mirroring the real SDK: two constructions = two distinct clients.
        with patch.object(
            provider_module.anthropic, "AsyncAnthropic",
            side_effect=lambda **kw: MagicMock(name="AsyncAnthropic()"),
        ) as mock_ctor, \
             patch.object(provider_module.httpx, "AsyncClient") as mock_httpx:
            unhardened = _get_client(prov)
            hardened = _get_client(prov, hardened=True)

            assert unhardened is not hardened
            assert mock_ctor.call_count == 2

            # Cache-key split: two distinct keys, each a 5-tuple ending in
            # the hardened flag (False / True).
            assert len(_client_cache) == 2
            keys = list(_client_cache.keys())
            assert len(set(keys)) == 2
            assert all(
                len(k) == 5 and k[0] == "anthropic" and k[1] == "sk-test"
                for k in keys
            )
            assert sorted(k[4] for k in keys) == [False, True]
            assert _client_cache[[k for k in keys if k[4] is False][0]] is unhardened
            assert _client_cache[[k for k in keys if k[4] is True][0]] is hardened

            # Unhardened construction is today-exact: NO http_client kwarg;
            # the hardened twin received one (share nothing), built by
            # constructing exactly one hardened httpx transport.
            unhardened_ctor_kwargs = mock_ctor.call_args_list[0].kwargs
            hardened_ctor_kwargs = mock_ctor.call_args_list[1].kwargs
            assert "http_client" not in unhardened_ctor_kwargs
            assert "http_client" in hardened_ctor_kwargs
            assert mock_httpx.call_count == 1
            assert hardened_ctor_kwargs["http_client"] is mock_httpx.return_value

            # AC9: after the hardened call, an unhardened lookup still
            # returns the ORIGINAL pooled client (same object, no new
            # construction).
            again = _get_client(prov)
            assert again is unhardened
            assert mock_ctor.call_count == 2

    def test_hardened_client_transport_flags_anthropic(self):
        """AC8: the hardened client's SDK constructor received an
        httpx.AsyncClient with follow_redirects=False and trust_env=False
        (inspected at mock level — no live sockets)."""
        prov = make_provider(key="anthropic", type="anthropic", api_key="sk-test")
        with patch.object(provider_module.anthropic, "AsyncAnthropic"), \
             patch.object(provider_module.httpx, "AsyncClient") as mock_httpx:
            _get_client(prov, hardened=True)

            assert mock_httpx.call_count == 1
            http_kwargs = mock_httpx.call_args.kwargs
            assert http_kwargs["follow_redirects"] is False
            assert http_kwargs["trust_env"] is False
            assert isinstance(http_kwargs["timeout"], httpx.Timeout)

    def test_hardened_client_transport_flags_openai(self):
        """AC8 (openai surface): same transport flags for AsyncOpenAI."""
        prov = make_provider(key="synthetic", type="openai", api_key="sk-test")
        with patch.object(provider_module.openai, "AsyncOpenAI"), \
             patch.object(provider_module.httpx, "AsyncClient") as mock_httpx:
            _get_client(prov, hardened=True)

            assert mock_httpx.call_count == 1
            http_kwargs = mock_httpx.call_args.kwargs
            assert http_kwargs["follow_redirects"] is False
            assert http_kwargs["trust_env"] is False
            assert isinstance(http_kwargs["timeout"], httpx.Timeout)


# ===========================================================================
# Defensive id (guard — green pre-implementation; no id-format parsing)
# ===========================================================================

class TestDefensiveId:

    @pytest.mark.asyncio
    async def test_kimi_style_tool_call_id_roundtrip(self):
        """AC10: a Kimi-style tool-call id (name:N, not call_<hex>) must
        round-trip verbatim through complete() on the OpenAI stream path.
        Guard against any future id-format branching/normalization — feeds a
        name:0-form id through the mock stream and asserts the id is echoed
        verbatim into Response.tool_calls[0].id."""
        config = make_config(
            providers={
                "synthetic": make_provider(
                    key="synthetic", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1",
                )
            },
            default_model="synthetic/hf:moonshotai/Kimi-K3",
        )

        chunks = [
            _openai_tool_delta_chunk(
                "submit_verdict:0", "submit_verdict", '{"verdict": "pass"}',
                finish_reason="tool_calls",
            ),
            _openai_usage_chunk(),
        ]

        with patch.object(provider_module.openai, "AsyncOpenAI") as mock_client_cls:
            client = mock_client_cls.return_value
            client.chat.completions.create = AsyncMock(
                return_value=MockOpenAIStream(chunks),
            )
            response = await complete(
                config=config,
                system="Test",
                messages=[{"role": "user", "content": "judge"}],
                tools=[SAMPLE_TOOL],
            )

        assert response.stop_reason == "tool_calls"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "submit_verdict:0"
        assert response.tool_calls[0].name == "submit_verdict"
        assert response.tool_calls[0].input == {"verdict": "pass"}


# ===========================================================================
# Hardened bounded read (discriminating — red pre-implementation, AC12)
# ===========================================================================

class _BigChunk:
    """A stand-in for an OpenAI stream chunk whose serialized size is
    ``nbytes``.

    Behaves like an empty chunk (no usage, no choices) so the generic
    consumption code parses nothing. Mirrors the real SDK contract: chunks
    are pydantic models (NO __len__) exposing ``model_dump_json()`` — which
    is exactly what the hardened byte accumulator measures. (A ``__len__``
    mock masked a live-call TypeError in the first implementation; this
    shape pins against that regression class.)
    """
    id = None
    usage = None
    choices = []

    def __init__(self, nbytes):
        self._nbytes = nbytes

    def model_dump_json(self) -> str:
        return "x" * self._nbytes


class TestHardenedBoundedRead:

    @pytest.mark.asyncio
    async def test_hardened_stream_over_cap_raises_providerror(self):
        """AC12: a hardened streamed response whose accumulated body exceeds
        _HARDENED_MAX_RESPONSE_BYTES aborts with ProviderError (mock stream
        emitting >10 MiB)."""
        assert provider_module._HARDENED_MAX_RESPONSE_BYTES == 10 * 1024 * 1024

        config = make_config(
            providers={
                "synthetic": make_provider(
                    key="synthetic", type="openai", api_key="sk-test",
                    base_url="http://localhost/v1",
                )
            },
            default_model="synthetic/hf:zai-org/GLM-5.3-Flash",
        )

        # 6 chunks x 2 MiB = 12 MiB > 10 MiB cap.
        big = [_BigChunk(2 * 1024 * 1024) for _ in range(6)]
        stream = MockOpenAIStream(big)

        with patch.object(provider_module.openai, "AsyncOpenAI") as mock_client_cls:
            client = mock_client_cls.return_value
            client.chat.completions.create = AsyncMock(return_value=stream)
            with pytest.raises(
                provider_module.ProviderError,
                match="hardened call: response body exceeded byte cap",
            ):
                await complete(
                    config=config,
                    system="Test",
                    messages=[{"role": "user", "content": "go"}],
                    hardened=True,
                )
