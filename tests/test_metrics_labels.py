"""Tests for the per-session metrics-labels header stamp (im7t.36.38).

Contract under test — SB-ratified 2026-09-25 (default-on for blackwell):

  1. The stamp carries per-request SGLang attribution labels as a JSON dict
     in one HTTP header on /v1/chat/completions:
       {"agent": <config.name>, "session": <room id / exec --room label>,
        "kind": "main"|"sub"}
     Header-only — never a body field (D1-clean: never logged either).
  2. HEADER NAME IS LOAD-BEARING: sgl-router 0.3.2 forwards ONLY an allowlist
     on the typed chat path (header_utils.rs should_forward_request_header:
     authorization, x-request-id, x-correlation-id,
     traceparent/tracestate, x-smg-routing-key, and the x-request-id-*
     PREFIX). The default header name therefore rides the prefix:
     x-request-id-oa-labels. Renaming it breaks fleet-wide labeling.
  3. Blackwell is DEFAULT-ON, no configuration needed. Kill flag:
     ``metrics_labels = false``. Header name overridable via
     ``metrics_labels_header``. Other openai-compatible providers opt in
     with ``metrics_labels = true``. Fireworks keeps its own path (no
     metrics-labels stamp unless explicitly opted in).
  4. Identity source: room_id for main turns -> derived
     {"agent", "session": room_id, "kind": "main"}. room_id=None and no
     explicit kwarg -> NO header (never invent identity).
  5. Subagent dispatch: the sub passes an explicit metrics_labels kwarg —
     its sub calls carry {"agent": <parent agent>, "session":
     <parent_room_id>, "kind": "sub"} WITHOUT reusing room_id (the kdsn.353
     routing-key design keeps subs unkeyed / deflectable; this invariant is
     pinned by the session label traveling ONLY in this header).
     Explicit kwarg wins over derivation. Breaker-summary call carries the
     same labels.
  6. Malformed kwarg: non-dict / empty -> treated as absent (warn, never
     crash a live turn); values coerced to str.
  7. Merge semantics: the stamp merges INTO existing extra_headers — it
     must coexist with the kdsn.353 routing-key stamp (both headers on
     blackwell calls).
"""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalph.config import AgentConfig, ProviderConfig, load_config
from openalph.provider import (
    DEFAULT_METRICS_LABELS_HEADER,
    complete,
    stream,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers (mirrors test_routing_key.py conventions)
# ---------------------------------------------------------------------------

ROOM = "!room:tuwunel.local"
ROOM_B = "!other:tuwunel.local"


def make_provider(key="blackwell", type="openai", api_key="sk-test",
                  base_url="http://10.0.20.111:8000/v1", **extra):
    kwargs = dict(key=key, type=type, api_key=api_key, base_url=base_url)
    kwargs.update(extra)
    return ProviderConfig(**kwargs)


def make_config(**kwargs):
    defaults = {
        "name": "test-agent",
        "default_model": "blackwell/qwen38-27b-fp8",
        "max_tokens": 8192,
        "providers": {"blackwell": make_provider(key="blackwell")},
        "workspace": Path("/tmp/test"),
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


class MockOpenAIStream:
    """Mock async-iterable for client.chat.completions.create results."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iter_impl()

    async def _iter_impl(self):
        for c in self._chunks:
            yield c


def _openai_text_chunk(content, finish_reason=None):
    choice = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.reasoning = None
    delta.reasoning_content = None
    delta.tool_calls = None
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.id = "gen-test"
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _openai_usage_chunk(prompt_tokens=100, completion_tokens=50):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    chunk = MagicMock()
    chunk.id = None
    chunk.choices = []
    chunk.usage = usage
    return chunk


async def collect_events(gen):
    async for event in gen:
        if event.type == "done":
            return event
    return None


def _chunks():
    return [
        _openai_text_chunk("ok", finish_reason="stop"),
        _openai_usage_chunk(),
    ]


def _openai_config(provider, default_model):
    """Single-provider openai config around `provider`."""
    return make_config(providers={provider.key: provider},
                       default_model=default_model)


def _mock_client_stream():
    """Fresh mock OpenAI client whose create() yields a 2-chunk stream."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=MockOpenAIStream(_chunks()),
    )
    return client


async def _drive_stream(config, room_id=None, metrics_labels=None):
    """Drive stream() against a fresh mock client; return the mock."""
    client = _mock_client_stream()
    with patch("openalph.provider._get_client", return_value=client):
        await collect_events(stream(
            config=config, system="Test",
            messages=[{"role": "user", "content": "Hi"}],
            room_id=room_id,
            metrics_labels=metrics_labels,
        ))
    return client


def _extra_headers(client):
    return client.chat.completions.create.call_args.kwargs.get(
        "extra_headers") or {}


def _labels_value(client):
    v = _extra_headers(client).get(DEFAULT_METRICS_LABELS_HEADER)
    return json.loads(v) if v is not None else None


DERIVED_ROOM = {"agent": "test-agent", "session": ROOM, "kind": "main"}


# ---------------------------------------------------------------------------
# Header-name snapshot — load-bearing router-prefix ride (§2)
# ---------------------------------------------------------------------------


class TestHeaderNameSnapshot:

    def test_default_header_name_rides_router_prefix(self):
        """sgl-router 0.3.2 forwards x-request-id-* only; the default name
        MUST keep that prefix or blackwell-wide labeling silently dies."""
        assert DEFAULT_METRICS_LABELS_HEADER == "x-request-id-oa-labels"
        assert DEFAULT_METRICS_LABELS_HEADER.startswith("x-request-id-")


# ---------------------------------------------------------------------------
# (a) Default-on derivation for blackwell — stream and complete (§1, §3)
# ---------------------------------------------------------------------------


class TestBlackwellDefaultOn:

    @pytest.mark.asyncio
    async def test_stream_stamps_derived_labels(self):
        config = make_config()
        client = await _drive_stream(config, room_id=ROOM)

        assert _labels_value(client) == DERIVED_ROOM
        # Header-only: never a body field.
        kw = client.chat.completions.create.call_args.kwargs
        assert "user" not in kw or kw.get("user") != DERIVED_ROOM

    @pytest.mark.asyncio
    async def test_complete_stamps_derived_labels(self):
        config = make_config()
        client = _mock_client_stream()
        with patch("openalph.provider._get_client", return_value=client):
            await complete(
                config=config, system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                room_id=ROOM,
            )
        assert _labels_value(client) == DERIVED_ROOM

    @pytest.mark.asyncio
    async def test_serialization_is_compact_and_canonical(self):
        """Wire format: compact JSON, canonical key order agent/session/kind."""
        config = make_config()
        client = await _drive_stream(config, room_id=ROOM)
        raw = _extra_headers(client)[DEFAULT_METRICS_LABELS_HEADER]
        assert raw == json.dumps(DERIVED_ROOM, separators=(",", ":"))


# ---------------------------------------------------------------------------
# (b) room_id=None + no kwarg -> no header (never invent identity) (§4)
# ---------------------------------------------------------------------------


class TestNoIdentityNoHeader:

    @pytest.mark.asyncio
    async def test_room_none_no_kwarg_no_labels_header(self):
        config = make_config()
        client = await _drive_stream(config, room_id=None)

        assert DEFAULT_METRICS_LABELS_HEADER not in _extra_headers(client)


# ---------------------------------------------------------------------------
# (c) Kill flag / opt-in / header-name override (§3)
# ---------------------------------------------------------------------------


class TestGating:

    @pytest.mark.asyncio
    async def test_kill_flag_disables_blackwell_default(self):
        config = make_config(providers={
            "blackwell": make_provider(key="blackwell", metrics_labels=False),
        })
        client = await _drive_stream(config, room_id=ROOM)
        assert DEFAULT_METRICS_LABELS_HEADER not in _extra_headers(client)

    @pytest.mark.asyncio
    async def test_kill_flag_beats_explicit_kwarg(self):
        """metrics_labels=false kills even an explicit per-call kwarg."""
        config = make_config(providers={
            "blackwell": make_provider(key="blackwell", metrics_labels=False),
        })
        client = await _drive_stream(
            config, room_id=ROOM,
            metrics_labels={"agent": "x", "session": "y", "kind": "sub"})
        assert DEFAULT_METRICS_LABELS_HEADER not in _extra_headers(client)

    @pytest.mark.asyncio
    async def test_non_blackwell_off_by_default(self):
        config = _openai_config(
            make_provider(key="macstudio"), "macstudio/glm-5.3-flash")
        client = await _drive_stream(config, room_id=ROOM)
        assert "extra_headers" not in \
            client.chat.completions.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_non_blackwell_opt_in_stamps(self):
        config = _openai_config(
            make_provider(key="macstudio", metrics_labels=True),
            "macstudio/glm-5.3-flash")
        client = await _drive_stream(config, room_id=ROOM)
        assert _labels_value(client) == DERIVED_ROOM

    @pytest.mark.asyncio
    async def test_header_name_override(self):
        config = make_config(providers={
            "blackwell": make_provider(
                key="blackwell", metrics_labels=True,
                metrics_labels_header="X-Custom-Labels"),
        })
        client = await _drive_stream(config, room_id=ROOM)
        kw = client.chat.completions.create.call_args.kwargs
        assert json.loads(kw["extra_headers"]["X-Custom-Labels"]) == \
            DERIVED_ROOM
        assert DEFAULT_METRICS_LABELS_HEADER not in kw["extra_headers"]


# ---------------------------------------------------------------------------
# (d) Explicit kwarg (sub path) (§5)
# ---------------------------------------------------------------------------


class TestExplicitKwarg:

    @pytest.mark.asyncio
    async def test_explicit_labels_stamped_verbatim(self):
        """Sub semantics: parent room + kind=sub, no room_id (never keyed)."""
        config = make_config()
        labels = {"agent": "test-agent", "session": ROOM, "kind": "sub"}
        client = await _drive_stream(config, room_id=None,
                                     metrics_labels=labels)
        assert _labels_value(client) == labels

    @pytest.mark.asyncio
    async def test_explicit_wins_over_derivation(self):
        """Explicit kwarg beats the room_id derivation when both exist."""
        config = make_config()
        labels = {"agent": "test-agent", "session": ROOM, "kind": "sub"}
        client = await _drive_stream(config, room_id=ROOM_B,
                                     metrics_labels=labels)
        got = _labels_value(client)
        assert got == labels
        assert got["kind"] == "sub"
        assert got["session"] == ROOM  # not ROOM_B

    @pytest.mark.asyncio
    async def test_partial_labels_not_augmented(self):
        """A two-key dict carries exactly two keys — no kind synthesized."""
        config = make_config()
        client = await _drive_stream(
            config, room_id=None,
            metrics_labels={"agent": "test-agent", "session": ROOM})
        assert _labels_value(client) == {"agent": "test-agent",
                                         "session": ROOM}


# ---------------------------------------------------------------------------
# (e) Merge semantics — coexists with the kdsn.353 routing-key stamp (§7)
# ---------------------------------------------------------------------------


class TestMergeSemantics:

    @pytest.mark.asyncio
    async def test_labels_merge_with_routing_key(self):
        """Blackwell default-on: BOTH the routing key and the labels header
        ride one call — neither clobbers the other."""
        config = make_config()
        client = await _drive_stream(config, room_id=ROOM)
        extra = _extra_headers(client)
        assert "X-SMG-Routing-Key" in extra
        assert DEFAULT_METRICS_LABELS_HEADER in extra

    @pytest.mark.asyncio
    async def test_labels_merge_with_fireworks_affinity(self):
        """A provider opted into BOTH fireworks affinity and labels keeps all
        three surfaces (user, x-session-affinity, labels)."""
        config = make_config(
            providers={"fireworks": make_provider(
                key="fireworks", type="openai",
                base_url="https://api.fireworks.ai/inference/v1",
                metrics_labels=True)},
            default_model="fireworks/accounts/fireworks/models/glm-5p2",
        )
        client = await _drive_stream(config, room_id=ROOM)
        kw = client.chat.completions.create.call_args.kwargs
        assert "user" in kw
        assert "x-session-affinity" in kw["extra_headers"]
        assert _labels_value(client) == DERIVED_ROOM


# ---------------------------------------------------------------------------
# (f) Malformed kwarg: warn-tier, never crash (§6)
# ---------------------------------------------------------------------------


class TestMalformedKwarg:

    @pytest.mark.asyncio
    async def test_non_dict_kwargs_treated_as_absent(self):
        """Non-dict metrics_labels -> derived-from-room fallback, no crash."""
        config = make_config()
        client = await _drive_stream(config, room_id=ROOM,
                                     metrics_labels="not-a-dict")
        assert _labels_value(client) == DERIVED_ROOM

    @pytest.mark.asyncio
    async def test_non_dict_kwargs_with_no_room_no_header(self):
        config = make_config()
        client = await _drive_stream(config, room_id=None,
                                     metrics_labels=42)
        assert DEFAULT_METRICS_LABELS_HEADER not in _extra_headers(client)

    @pytest.mark.asyncio
    async def test_non_string_values_coerced(self):
        config = make_config()
        labels = {"agent": "a", "session": 7, "kind": "main"}
        client = await _drive_stream(config, room_id=None,
                                     metrics_labels=labels)
        got = _labels_value(client)
        assert got == {"agent": "a", "session": "7", "kind": "main"}

    @pytest.mark.asyncio
    async def test_empty_dict_treated_as_absent(self):
        config = make_config()
        client = await _drive_stream(config, room_id=None,
                                     metrics_labels={})
        assert DEFAULT_METRICS_LABELS_HEADER not in _extra_headers(client)


# ---------------------------------------------------------------------------
# D1-clean: never logged (§1)
# ---------------------------------------------------------------------------


class TestNotLogged:

    @pytest.mark.asyncio
    async def test_labels_never_logged(self, caplog):
        config = make_config()
        with caplog.at_level(logging.DEBUG, logger="openalph.provider"):
            await _drive_stream(config, room_id=ROOM)

        value = json.dumps(DERIVED_ROOM, separators=(",", ":"))
        for record in caplog.records:
            assert DEFAULT_METRICS_LABELS_HEADER not in record.getMessage()
            assert value not in record.getMessage()
            assert ROOM not in record.getMessage()


# ---------------------------------------------------------------------------
# Resolver unit tests (the gating decision lives here)
# ---------------------------------------------------------------------------


class TestResolver:

    def test_blackwell_default_on(self):
        from openalph.provider import _resolved_metrics_labels_header
        assert _resolved_metrics_labels_header(
            make_provider(key="blackwell")) == "x-request-id-oa-labels"

    def test_other_provider_default_off(self):
        from openalph.provider import _resolved_metrics_labels_header
        assert _resolved_metrics_labels_header(
            make_provider(key="macstudio")) is None
        assert _resolved_metrics_labels_header(
            make_provider(key="fireworks")) is None

    def test_kill_flag_beats_default(self):
        from openalph.provider import _resolved_metrics_labels_header
        assert _resolved_metrics_labels_header(
            make_provider(key="blackwell", metrics_labels=False)) is None

    def test_opt_in_and_override(self):
        from openalph.provider import _resolved_metrics_labels_header
        assert _resolved_metrics_labels_header(
            make_provider(key="macstudio", metrics_labels=True)) == \
            "x-request-id-oa-labels"
        assert _resolved_metrics_labels_header(make_provider(
            key="macstudio", metrics_labels=True,
            metrics_labels_header="X-Other")) == "X-Other"


# ---------------------------------------------------------------------------
# Config parsing (fail-loud, same discipline as neighboring knobs)
# ---------------------------------------------------------------------------

TOML_BASE = """
[agent]
name = "test"
default_model = "blackwell/qwen38-27b-fp8"

[providers.blackwell]
type = "openai"
api_key = "sk-test"
base_url = "http://10.0.20.111:8000/v1"
{knobs}
[workspace]
path = "/tmp/test"
"""


class TestConfigParsing:

    def test_absent_defaults_none(self, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(knobs=""))
        config = load_config(tmp_path / "agent.toml")
        p = config.providers["blackwell"]
        assert p.metrics_labels is None
        assert p.metrics_labels_header is None

    def test_parsed(self, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs='metrics_labels = true\n'
                  'metrics_labels_header = "X-Custom"\n'))
        config = load_config(tmp_path / "agent.toml")
        p = config.providers["blackwell"]
        assert p.metrics_labels is True
        assert p.metrics_labels_header == "X-Custom"

    def test_kill_flag_parsed(self, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs="metrics_labels = false\n"))
        config = load_config(tmp_path / "agent.toml")
        assert config.providers["blackwell"].metrics_labels is False

    def test_invalid_metrics_labels_skips_provider(self, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs='metrics_labels = "yes"\n'))
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "metrics_labels must be a boolean" in \
            config.skipped_providers["blackwell"]

    def test_invalid_metrics_labels_header_skips_provider(self, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs="metrics_labels_header = 42\n"))
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "metrics_labels_header must be a non-empty string" in \
            config.skipped_providers["blackwell"]

    def test_empty_metrics_labels_header_skips_provider(self, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs='metrics_labels_header = ""\n'))
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "metrics_labels_header must be a non-empty string" in \
            config.skipped_providers["blackwell"]


# ---------------------------------------------------------------------------
# Subagent dispatch: explicit sub labels at the complete() call sites (§5)
# ---------------------------------------------------------------------------

from openalph.provider import Response, Usage  # noqa: E402
from openalph.tools.subagent import run_subagent  # noqa: E402

COMPLETE_PATH = "openalph.tools.subagent.complete"


def _sub_config(**kwargs):
    defaults = dict(
        name="test-parent",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test")},
        workspace=Path("/tmp/test"),
        max_iterations=25,
        truncation_limit=50000,
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _text_response(text="Sub-agent response"):
    return Response(
        content=text,
        tool_calls=[],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=100, output_tokens=50),
        stop_reason="end_turn",
    )


class TestSubagentLabels:

    @pytest.mark.asyncio
    async def test_sub_calls_carry_parent_session_and_kind_sub(self):
        """run_subagent passes metrics_labels with parent room + kind=sub —
        arriving via the explicit kwarg, never room_id (stays deflectable)."""
        with patch(COMPLETE_PATH, new_callable=AsyncMock) as m:
            m.return_value = _text_response()
            await run_subagent(
                task="Do a thing",
                model=None, tools=None, config=_sub_config(),
                parent_room_id="!parent:tuwunel.local",
                callbacks=None,
            )
        kw = m.call_args.kwargs
        assert kw["metrics_labels"] == {
            "agent": "test-parent",
            "session": "!parent:tuwunel.local",
            "kind": "sub",
        }
        # kdsn.353 invariant: subs are never keyed — room_id is not even
        # threaded into the sub's provider call (absent key, not None).
        assert "room_id" not in kw or kw["room_id"] is None

    @pytest.mark.asyncio
    async def test_sub_labels_without_parent_room(self):
        """No parent room (headless/CLI dispatch): labels omit session."""
        with patch(COMPLETE_PATH, new_callable=AsyncMock) as m:
            m.return_value = _text_response()
            await run_subagent(
                task="Do a thing",
                model=None, tools=None, config=_sub_config(),
                parent_room_id=None,
                callbacks=None,
            )
        kw = m.call_args.kwargs
        assert kw["metrics_labels"] == {"agent": "test-parent",
                                        "kind": "sub"}
        assert "session" not in kw["metrics_labels"]


# ---------------------------------------------------------------------------
# Audit remediation (2026-09-25 cold-read, synkimi3): M1 reserved-name
# guards, M2 coerce hardening, breaker-summary pin
# ---------------------------------------------------------------------------

from openalph.provider import ToolCall  # noqa: E402


class TestReservedHeaderNames:
    """M1: *_header override knobs must never name a reserved header — the
    extra_headers merge is last-writer-wins and a collision silently
    clobbers affinity stamps or auth."""

    @pytest.mark.parametrize("bad", ["X-SMG-Routing-Key", "authorization",
                                     "x-session-affinity",
                                     "X-REQUEST-ID-OA-LABELS"])
    def test_metrics_labels_header_reserved_rejected(self, bad, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs=f'metrics_labels_header = "{bad}"\n'))
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "must not collide with a reserved header name" in \
            config.skipped_providers["blackwell"]

    @pytest.mark.parametrize("bad", ["AUTHORIZATION", "content-type",
                                     "x-session-affinity"])
    def test_routing_key_header_reserved_rejected(self, bad, tmp_path):
        """Same guard on the kdsn.353 knob (same bug shape, audit M1)."""
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs=f'routing_key_header = "{bad}"\n'))
        config = load_config(tmp_path / "agent.toml")
        assert config.providers == {}
        assert "must not collide with a reserved header name" in \
            config.skipped_providers["blackwell"]

    def test_custom_names_still_parse(self, tmp_path):
        (tmp_path / "agent.toml").write_text(TOML_BASE.format(
            knobs='metrics_labels_header = "X-Custom-Labels"\n'
                  'routing_key_header = "X-Custom-Affinity"\n'))
        config = load_config(tmp_path / "agent.toml")
        p = config.providers["blackwell"]
        assert p.metrics_labels_header == "X-Custom-Labels"
        assert p.routing_key_header == "X-Custom-Affinity"


class TestCoerceHardening:
    """M2: malformed-per-call values must degrade cleanly."""

    @pytest.mark.asyncio
    async def test_none_value_dropped_not_stringified(self):
        config = make_config()
        client = await _drive_stream(
            config, room_id=None,
            metrics_labels={"agent": "a", "session": None, "kind": "sub"})
        got = _labels_value(client)
        assert got == {"agent": "a", "kind": "sub"}
        assert "None" not in str(got)

    @pytest.mark.asyncio
    async def test_invalid_prom_key_dropped_valid_kept(self):
        config = make_config()
        client = await _drive_stream(
            config, room_id=None,
            metrics_labels={"agent": "a", "not a key": "x", "kind": "main"})
        got = _labels_value(client)
        assert got == {"agent": "a", "kind": "main"}

    @pytest.mark.asyncio
    async def test_all_invalid_keys_no_stamp(self):
        config = make_config()
        client = await _drive_stream(
            config, room_id=None,
            metrics_labels={"not a key": "x", "1bad": "y"})
        assert DEFAULT_METRICS_LABELS_HEADER not in _extra_headers(client)


def _tool_response(tool_name="file_read", tool_input=None, tool_id="tc_1"):
    return Response(
        content="",
        tool_calls=[ToolCall(id=tool_id, name=tool_name,
                             input=tool_input or {"path": "/nonexistent"})],
        model="claude-sonnet-4-20250514",
        usage=Usage(input_tokens=100, output_tokens=50),
        stop_reason="tool_use",
    )


class TestBreakerSummaryLabels:

    @pytest.mark.asyncio
    async def test_breaker_summary_carries_sub_labels(self):
        """The breaker-summary complete() call carries the SAME sub labels
        (tools=None round trip after iteration exhaustion)."""
        with patch(COMPLETE_PATH, new_callable=AsyncMock) as m:
            m.return_value = _tool_response()
            await run_subagent(
                task="Do a thing",
                model=None, tools=None, config=_sub_config(),
                max_iterations=1,
                parent_room_id="!parent:tuwunel.local",
                callbacks=None,
            )
        assert m.call_count == 2  # loop call + breaker summary
        kw = m.call_args_list[1].kwargs
        assert kw["tools"] is None
        assert kw["metrics_labels"] == {
            "agent": "test-parent",
            "session": "!parent:tuwunel.local",
            "kind": "sub",
        }
