"""Regression tests for bead workspace-kdsn.186.18: Gemini 3.x tool-loop
support (thought_signature echo-back) + an adjacent bug found while fixing
it (Gemini's OpenAI-compat streaming never populates tool_calls[].index).

Background
----------
Google's Gemini 3.x models require the opaque `thought_signature` returned
on a function-call part to be echoed back verbatim on the next turn, or the
API 400s with INVALID_ARGUMENT ("Function call is missing a
thought_signature in functionCall parts"). Via OA's OpenAI-compat provider
path, this signature is surfaced on the tool_call object itself as
`extra_content.google.thought_signature` (verified live against the real
Gemini API, 2026-07-16 — see workspace-kdsn.186.18 and
memory/daily/2026-07-16.md for the probe transcript).

Separately, while probing the live API it was discovered that Gemini's
OpenAI-compat streaming NEVER populates `tool_calls[].index` (always None).
OA's streaming accumulator previously keyed exclusively on `index`, so two
simultaneous tool calls in one Gemini turn would collide into the same
dict slot and corrupt/lose one of them. Fixed alongside the signature work
since it's the same code region and the same root cause (Gemini's
OpenAI-compat layer not conforming to the index contract other providers
honor).

This file is provider-agnostic in spirit: ToolCall.extra_content is `None`
by default and only ever populated when a provider's response actually
supplies it, so every assertion here doubles as a regression guard that
non-Google providers are unaffected.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import (
    ToolCall,
    _parse_openai_response,
    _convert_messages_for_openai,
    stream,
)
from openalph.session import SessionLog
from openalph.matrix import MatrixBot


# A realistic-looking signature (not a real captured value) — long opaque
# base64-ish blob, matching the shape Google actually returns.
FAKE_SIGNATURE = "Eq0DCqoD" + ("A" * 200) + "=="
FAKE_EXTRA_CONTENT = {"google": {"thought_signature": FAKE_SIGNATURE}}


# ---------------------------------------------------------------------------
# ToolCall dataclass
# ---------------------------------------------------------------------------

class TestToolCallExtraContent:
    def test_defaults_to_none(self):
        tc = ToolCall(id="1", name="shell", input={})
        assert tc.extra_content is None

    def test_accepts_extra_content(self):
        tc = ToolCall(id="1", name="shell", input={}, extra_content=FAKE_EXTRA_CONTENT)
        assert tc.extra_content == FAKE_EXTRA_CONTENT


# ---------------------------------------------------------------------------
# Non-streaming response parsing
# ---------------------------------------------------------------------------

def _fake_openai_message(tool_calls):
    """Build a fake `response.choices[0].message`-shaped object using
    SimpleNamespace (not MagicMock) so that attribute access for fields we
    did NOT set genuinely raises AttributeError -> getattr(..., default)
    correctly falls back, exactly like a real (schema'd) SDK response
    object would for a field the provider never sent."""
    return SimpleNamespace(content=None, tool_calls=tool_calls, reasoning=None,
                            reasoning_content=None)


def _fake_openai_response(message):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model="gemini-3.5-flash",
        id="resp_1",
    )


def _fake_tool_call(id, name, arguments, extra_content=None):
    fn = SimpleNamespace(name=name, arguments=json.dumps(arguments))
    kwargs = dict(id=id, function=fn)
    if extra_content is not None:
        kwargs["extra_content"] = extra_content
    return SimpleNamespace(**kwargs)


class TestParseOpenAIResponseExtraContent:
    def test_captures_extra_content_when_present(self):
        message = _fake_openai_message([
            _fake_tool_call("call_1", "get_weather", {"city": "Boston"},
                             extra_content=FAKE_EXTRA_CONTENT),
        ])
        response = _fake_openai_response(message)
        parsed = _parse_openai_response(response)
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].extra_content == FAKE_EXTRA_CONTENT

    def test_none_when_absent(self):
        """A provider that never sends extra_content (Fireworks, OpenAI,
        Together, ...) must not crash and must yield extra_content=None."""
        message = _fake_openai_message([
            _fake_tool_call("call_1", "shell", {"command": "ls"}),
        ])
        response = _fake_openai_response(message)
        parsed = _parse_openai_response(response)
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].extra_content is None


# ---------------------------------------------------------------------------
# Outbound reconstruction (_convert_messages_for_openai)
# ---------------------------------------------------------------------------

class TestConvertMessagesExtraContent:
    def test_extra_content_echoed_when_present(self):
        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    ToolCall(id="call_1", name="get_weather", input={"city": "Boston"},
                             extra_content=FAKE_EXTRA_CONTENT),
                ],
            },
        ]
        result = _convert_messages_for_openai(messages)
        assistant = [m for m in result if m["role"] == "assistant"][0]
        assert assistant["tool_calls"][0]["extra_content"] == FAKE_EXTRA_CONTENT

    def test_extra_content_key_omitted_when_none(self):
        """Regression guard: every other provider's tool_calls must NOT
        gain an extra_content key — clean wire format, no behavior change."""
        messages = [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [ToolCall(id="call_1", name="shell", input={"command": "ls"})],
            },
        ]
        result = _convert_messages_for_openai(messages)
        assistant = [m for m in result if m["role"] == "assistant"][0]
        assert "extra_content" not in assistant["tool_calls"][0]


# ---------------------------------------------------------------------------
# Streaming: end-to-end via stream() with a mocked OpenAI-compat client
# ---------------------------------------------------------------------------

def make_google_config(**kwargs):
    defaults = {
        "name": "test-agent",
        "default_model": "google/gemini-3.5-flash",
        "max_tokens": 8192,
        "providers": {
            "google": ProviderConfig(
                key="google", type="openai", api_key="sk-test",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                quirks=[],
            )
        },
        "workspace": Path("/tmp/test"),
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


class MockOpenAIStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for chunk in self._chunks:
            yield chunk


def _gemini_tool_chunk(tool_id, name, arguments, extra_content=None):
    """A Gemini-style tool_call delta: full call arrives in one chunk,
    index is always None (verified live), extra_content present only on
    Gemini 3.x (absent -> not set at all, mirroring a real schema'd SDK
    object rather than an auto-vivifying MagicMock)."""
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = None
    tc_delta = SimpleNamespace(
        index=None,
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    if extra_content is not None:
        tc_delta.extra_content = extra_content
    delta.tool_calls = [tc_delta]
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _finish_chunk(finish_reason="tool_calls"):
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = None
    delta.tool_calls = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _usage_chunk():
    chunk = MagicMock()
    chunk.choices = []
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 20
    chunk.usage = usage
    return chunk


async def _collect(gen):
    events = []
    async for event in gen:
        events.append(event)
    return events


class TestGeminiStreamingToolCalls:

    @pytest.mark.asyncio
    async def test_single_tool_call_carries_extra_content(self):
        config = make_google_config()
        chunks = [
            _gemini_tool_chunk("call_1", "get_weather", {"city": "Boston"},
                               extra_content=FAKE_EXTRA_CONTENT),
            _finish_chunk(),
            _usage_chunk(),
        ]
        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(return_value=MockOpenAIStream(chunks))
            events = await _collect(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "weather?"}],
            ))

        tool_dones = [e for e in events if e.type == "tool_done"]
        assert len(tool_dones) == 1
        assert tool_dones[0].tool_call.extra_content == FAKE_EXTRA_CONTENT

        done = [e for e in events if e.type == "done"][0]
        assert len(done.response.tool_calls) == 1
        assert done.response.tool_calls[0].extra_content == FAKE_EXTRA_CONTENT

    @pytest.mark.asyncio
    async def test_two_simultaneous_tool_calls_do_not_collide(self):
        """Regression test for the index=None collision bug: two tool
        calls in one Gemini turn, both with index=None, must NOT merge
        into a single accumulator slot."""
        config = make_google_config()
        chunks = [
            _gemini_tool_chunk("call_1", "get_weather", {"city": "Boston"},
                               extra_content=FAKE_EXTRA_CONTENT),
            _gemini_tool_chunk("call_2", "get_weather", {"city": "Chicago"},
                               extra_content={"google": {"thought_signature": "sig-2"}}),
            _finish_chunk(),
            _usage_chunk(),
        ]
        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(return_value=MockOpenAIStream(chunks))
            events = await _collect(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "weather in both?"}],
            ))

        tool_dones = [e for e in events if e.type == "tool_done"]
        assert len(tool_dones) == 2, (
            "Two Gemini tool calls with index=None collided into one slot"
        )
        calls_by_id = {td.tool_call.id: td.tool_call for td in tool_dones}
        assert calls_by_id["call_1"].input == {"city": "Boston"}
        assert calls_by_id["call_2"].input == {"city": "Chicago"}
        assert calls_by_id["call_1"].extra_content["google"]["thought_signature"] == FAKE_SIGNATURE
        assert calls_by_id["call_2"].extra_content["google"]["thought_signature"] == "sig-2"

    @pytest.mark.asyncio
    async def test_normal_indexed_provider_unaffected(self):
        """Regression guard: a provider that DOES send a real int index
        (e.g. Fireworks/OpenAI) must be completely unaffected by the
        None-fallback logic — still keyed/ordered by its real index."""
        config = make_google_config(default_model="fireworks/accounts/fireworks/models/kimi-k2p6")
        config.providers["fireworks"] = ProviderConfig(
            key="fireworks", type="openai", api_key="sk-test",
            base_url="https://api.fireworks.ai/inference/v1", quirks=[],
        )

        def indexed_chunk(index, tool_id, name, arguments):
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = None
            tc_delta = SimpleNamespace(
                index=index, id=tool_id,
                function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
            )
            delta.tool_calls = [tc_delta]
            choice = MagicMock()
            choice.delta = delta
            choice.finish_reason = None
            chunk.choices = [choice]
            chunk.usage = None
            return chunk

        chunks = [
            indexed_chunk(0, "call_1", "shell", {"command": "ls"}),
            indexed_chunk(1, "call_2", "file_read", {"path": "/tmp/x"}),
            _finish_chunk(),
            _usage_chunk(),
        ]
        with patch("openalph.provider._get_client") as mock_gc:
            client = MagicMock()
            mock_gc.return_value = client
            client.chat.completions.create = AsyncMock(return_value=MockOpenAIStream(chunks))
            events = await _collect(stream(
                config=config, system="Test",
                messages=[{"role": "user", "content": "do things"}],
            ))

        tool_dones = [e for e in events if e.type == "tool_done"]
        assert len(tool_dones) == 2
        assert {td.tool_call.extra_content for td in tool_dones} == {None}
        names = {td.tool_call.name for td in tool_dones}
        assert names == {"shell", "file_read"}


# ---------------------------------------------------------------------------
# JSONL persistence round-trip (matrix.py write side, session.py read side)
# ---------------------------------------------------------------------------

class TestMatrixPersistExtraContent:
    def test_persist_includes_extra_content_when_present(self, tmp_path):
        bot = MagicMock(spec=MatrixBot)
        bot.config = SimpleNamespace(user_id="@bot:example.com")
        bot.session_log = MagicMock()
        bot.agent = MagicMock()
        bot.agent.history.return_value = [{"role": "assistant", "content": "", "thinking": None}]
        bot.agent.last_turn_usage.return_value = {}

        tool_calls = [ToolCall(id="call_1", name="get_weather", input={"city": "Boston"},
                                extra_content=FAKE_EXTRA_CONTENT)]

        MatrixBot._persist_assistant_turn(bot, "!room:example.com", content="", tool_calls=tool_calls)

        appended = bot.session_log.append.call_args.kwargs
        assert appended["tool_calls"][0]["extra_content"] == FAKE_EXTRA_CONTENT

    def test_persist_omits_extra_content_when_absent(self, tmp_path):
        bot = MagicMock(spec=MatrixBot)
        bot.config = SimpleNamespace(user_id="@bot:example.com")
        bot.session_log = MagicMock()
        bot.agent = MagicMock()
        bot.agent.history.return_value = [{"role": "assistant", "content": "", "thinking": None}]
        bot.agent.last_turn_usage.return_value = {}

        tool_calls = [ToolCall(id="call_1", name="shell", input={"command": "ls"})]

        MatrixBot._persist_assistant_turn(bot, "!room:example.com", content="", tool_calls=tool_calls)

        appended = bot.session_log.append.call_args.kwargs
        assert "extra_content" not in appended["tool_calls"][0]


class TestSessionRehydrationExtraContent:
    def _write_and_load(self, tmp_path, entries):
        log = SessionLog(tmp_path, "@bot:example.com")
        room = "!room:example.com"
        for e in entries:
            log.append(room=room, **e)
        return log.build_context(room)

    def test_rehydrates_extra_content(self, tmp_path):
        entries = [
            dict(role="user", sender="@sb:example.com", event_id="e1", content="weather?"),
            dict(role="assistant", sender="@bot:example.com", event_id="e2", content="",
                 tool_calls=[{"call_id": "call_1", "name": "get_weather",
                              "input": {"city": "Boston"}, "extra_content": FAKE_EXTRA_CONTENT}]),
            dict(role="tool", sender="@bot:example.com", event_id=None,
                 call_id="call_1", name="get_weather", output="cloudy, 41F"),
        ]
        context = self._write_and_load(tmp_path, entries)
        assistant_msgs = [m for m in context if m.get("role") == "assistant" and m.get("tool_calls")]
        assert len(assistant_msgs) == 1
        tc = assistant_msgs[0]["tool_calls"][0]
        assert tc.extra_content == FAKE_EXTRA_CONTENT

    def test_rehydrates_legacy_entries_without_extra_content(self, tmp_path):
        """Old JSONL entries written before this fix have no extra_content
        key at all — must rehydrate cleanly with extra_content=None, not KeyError."""
        entries = [
            dict(role="user", sender="@sb:example.com", event_id="e1", content="hi"),
            dict(role="assistant", sender="@bot:example.com", event_id="e2", content="",
                 tool_calls=[{"call_id": "call_1", "name": "shell", "input": {"command": "ls"}}]),
            dict(role="tool", sender="@bot:example.com", event_id=None,
                 call_id="call_1", name="shell", output="file1 file2"),
        ]
        context = self._write_and_load(tmp_path, entries)
        assistant_msgs = [m for m in context if m.get("role") == "assistant" and m.get("tool_calls")]
        assert len(assistant_msgs) == 1
        tc = assistant_msgs[0]["tool_calls"][0]
        assert tc.extra_content is None
