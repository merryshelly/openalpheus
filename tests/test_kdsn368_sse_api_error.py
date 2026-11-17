"""kdsn.368 — safe OpenAI SDK streaming SSE ``APIError`` diagnostics.

The trust contract is deliberately fail-closed: private prompt/credential-shaped
canaries inside upstream prose must never reach journald or ProviderError.
Only structured metadata is retained; free-form error prose is represented by
presence/length. Tests drive the real OpenAI SDK SSE parser (zero network),
public stream()/complete(), and pinned no-retry behavior.
"""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import ProviderError, StreamEvent, complete, stream


FAKE_API_KEY = "sk-proj-" + ("f" * 24)
PRIVATE_PROMPT = "PRIVATE_PROMPT_NEVER_LOG_71c04db1"
PRIVATE_DEBUG_VALUE = "PRIVATE_DEBUG_NEVER_LOG_a4257fd2"
OPAQUE_SECRET = "opaque-unrecognized-credential-never-log-d19ec6f7"
FREEFORM = "FREEFORM_UPSTREAM_PROSE_NEVER_LOG_a026e33a"
GENERIC = "Provider stream error: upstream returned an error (diagnostics logged)"
SENSITIVE_CANARIES = (FAKE_API_KEY, PRIVATE_PROMPT, PRIVATE_DEBUG_VALUE, OPAQUE_SECRET, FREEFORM)


def make_config() -> AgentConfig:
    return AgentConfig(
        name="kdsn368-test",
        default_model="synthetic/hf:moonshotai/Kimi-K3.1-Pro",
        max_tokens=256,
        model_max_tokens=524_288,
        providers={
            "synthetic": ProviderConfig(
                key="synthetic", type="openai", api_key=FAKE_API_KEY,
                base_url="https://api.synthetic.new/openai/v1", quirks=[],
            )
        },
        workspace=Path("/tmp/kdsn368-test"),
    )


def sdk_client_for_sse(payload: str, requests: list[httpx.Request]):
    """Real AsyncOpenAI client with a deterministic zero-network SSE transport."""
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text=payload, request=request,
        )

    return openai.AsyncOpenAI(
        api_key=FAKE_API_KEY,
        base_url="https://unit.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def openai_chunk(text: str) -> str:
    return "data: " + json.dumps({
        "id": "chatcmpl-kdsn368", "object": "chat.completion.chunk", "created": 1,
        "model": "hf:moonshotai/Kimi-K3.1-Pro",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }) + "\n\n"


def api_record(caplog) -> tuple[str, dict]:
    """Extract exactly one matching provider WARNING and parse its JSON payload."""
    records = [
        record for record in caplog.records
        if record.name == "openalph.provider"
        and record.getMessage().startswith("openai stream APIError ")
    ]
    assert len(records) == 1, [record.getMessage() for record in caplog.records]
    message = records[0].getMessage()
    prefix, separator, raw_json = message.partition(": ")
    assert separator, message
    return prefix, json.loads(raw_json)


def assert_canaries_absent(*texts: str):
    for text in texts:
        for canary in SENSITIVE_CANARIES:
            assert canary not in text


class RaisingCreate:
    def __init__(self, error):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = AsyncMock(side_effect=error)


async def consume_direct_error(error: openai.APIError, *, reset_cb=None, use_complete=False):
    """Drive the OpenAI branch with a direct residual APIError and return evidence."""
    client = RaisingCreate(error)
    events: list[StreamEvent] = []
    try:
        with patch("openalph.provider._get_client", return_value=client):
            with pytest.raises(ProviderError) as exc_info:
                if use_complete:
                    await complete(
                        config=make_config(), system="test",
                        messages=[{"role": "user", "content": PRIVATE_PROMPT}],
                    )
                else:
                    async for event in stream(
                        config=make_config(), system="test",
                        messages=[{"role": "user", "content": PRIVATE_PROMPT}],
                        on_stream_reset=reset_cb,
                    ):
                        events.append(event)
    except Exception as exc:  # pragma: no cover — only on fixture failure
        raise AssertionError(f"expected ProviderError, got {type(exc).__name__}: {exc}") from exc
    return exc_info.value, client, events


TEST_REQUEST = httpx.Request(
    "POST", "https://unit.invalid/v1/chat/completions",
    headers={"Authorization": "Bearer " + FAKE_API_KEY},
)


def direct_error(body, *, request=TEST_REQUEST) -> openai.APIError:
    return openai.APIError(
        "An error occurred during streaming", request=request, body=body,
    )


@pytest.mark.asyncio
async def test_mapping_error_zero_chunks_metadata_only_no_retry(caplog):
    """Real SDK {error: mapping}: retain metadata, suppress all upstream prose."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    reset_cb = AsyncMock()
    requests: list[httpx.Request] = []
    freeform = f"{FREEFORM}; echoed prompt={PRIVATE_PROMPT}; credential={OPAQUE_SECRET}"
    payload = "data: " + json.dumps({"error": {
        "code": "worker_down", "message": freeform, "type": "server_error",
        "param": "model", "request_id": "req_kdsn368_123", "id": "gen-safe-123",
        "debug": {"prompt": PRIVATE_PROMPT, "value": PRIVATE_DEBUG_VALUE},
        "authorization": FAKE_API_KEY,
    }}) + "\n\n"
    client = sdk_client_for_sse(payload, requests)
    try:
        with patch("openalph.provider._get_client", return_value=client):
            with pytest.raises(ProviderError) as exc_info:
                async for _ in stream(
                    config=make_config(), system="SYSTEM_NEVER_REQUEST_LOG",
                    messages=[{"role": "user", "content": PRIVATE_PROMPT}],
                    thinking="xhigh", on_stream_reset=reset_cb,
                ):
                    pass
    finally:
        await client.close()

    assert str(exc_info.value) == GENERIC
    assert str(exc_info.value).splitlines() == [GENERIC]
    assert type(exc_info.value.__cause__).__name__ == "APIError"
    assert len(requests) == 1, "diagnostics/wrapping must not add a retry"
    reset_cb.assert_not_awaited()
    prefix, record = api_record(caplog)
    assert "after 0 chunk(s)" in prefix
    assert record["exception"] == "APIError"
    assert record["body_type"] == "dict"
    assert record["code"] == "worker_down"
    assert record["type"] == "server_error"
    assert record["param"] == "model"
    assert record["id"] == "gen-safe-123"
    assert record["request_id"] == "req_kdsn368_123"
    assert record["message_present"] is True
    assert record["message_chars"] == len(freeform)
    assert "message" not in record, "free-form message must never be logged"
    assert_canaries_absent(caplog.text, str(exc_info.value))
    assert "Authorization" not in caplog.text


@pytest.mark.asyncio
async def test_string_error_after_chunk_metadata_only_no_retry(caplog):
    """Real SDK {error: string}: retain type/length only after partial output."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    reset_cb = AsyncMock()
    requests: list[httpx.Request] = []
    freeform = f"{FREEFORM}; prompt={PRIVATE_PROMPT}; keys={OPAQUE_SECRET} {FAKE_API_KEY}"
    payload = openai_chunk("partial") + "data: " + json.dumps({"error": freeform}) + "\n\n"
    client = sdk_client_for_sse(payload, requests)
    events: list[StreamEvent] = []
    try:
        with patch("openalph.provider._get_client", return_value=client):
            with pytest.raises(ProviderError) as exc_info:
                async for event in stream(
                    config=make_config(), system="SYSTEM_NEVER_REQUEST_LOG",
                    messages=[{"role": "user", "content": PRIVATE_PROMPT}],
                    thinking="xhigh", on_stream_reset=reset_cb,
                ):
                    events.append(event)
    finally:
        await client.close()

    assert [e.content for e in events if e.type == "text"] == ["partial"]
    assert str(exc_info.value) == GENERIC
    assert type(exc_info.value.__cause__).__name__ == "APIError"
    assert len(requests) == 1, "no retry even after output began"
    reset_cb.assert_not_awaited()
    prefix, record = api_record(caplog)
    assert "after 1 chunk(s)" in prefix
    assert record["body_type"] == "str"
    assert record["body_present"] is True
    assert record["body_chars"] == len(freeform)
    assert "body" not in record, "string body content must never be logged"
    assert_canaries_absent(caplog.text, str(exc_info.value))
    assert "Authorization" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [None, ["unlogged"], {"nested": {"drop": PRIVATE_DEBUG_VALUE}}])
async def test_unsupported_body_shapes_fail_closed(caplog, body):
    """Null, list, and nested mappings produce omission metadata + generic error."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    error = direct_error(body)
    wrapped, client, _ = await consume_direct_error(error)

    assert str(wrapped) == GENERIC
    assert wrapped.__cause__ is error
    assert client.chat.completions.create.await_count == 1
    _, record = api_record(caplog)
    assert record["body_type"] == type(body).__name__
    assert record["body_omitted"] is True
    assert_canaries_absent(caplog.text, str(wrapped))


@pytest.mark.asyncio
async def test_long_unicode_body_bounded_and_generic_surface_is_single_line(caplog):
    """Huge prose + Unicode separators cannot flood logs or forge room lines."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    body = (FREEFORM + "x") * 2500 + "\u0085\u2028\u2029FORGED_LOG_LINE"
    error = direct_error(body)
    wrapped, _, _ = await consume_direct_error(error)

    assert str(wrapped) == GENERIC
    assert len(str(wrapped).splitlines()) == 1
    assert len(caplog.text) < 1500
    _, record = api_record(caplog)
    assert record["body_chars"] == len(body)
    assert_canaries_absent(caplog.text, str(wrapped))


@pytest.mark.asyncio
async def test_available_exception_request_id_is_logged(caplog):
    """Short format-valid request IDs survive; request auth headers stay unread."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    error = direct_error({"message": FREEFORM})
    error.request_id = "req_kdsn368_attr"
    wrapped, _, _ = await consume_direct_error(error)

    _, record = api_record(caplog)
    assert record["request_id"] == "req_kdsn368_attr"
    assert record["message_present"] is True
    assert_canaries_absent(caplog.text, str(wrapped))
    assert "Authorization" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", [
    None, "", "x" * 129, "whitespace is not valid", "line\nbreak",
    "unicode\u2028separator", 1234, ["nested"], {"nested": PRIVATE_DEBUG_VALUE},
    {"unserializable"}, FAKE_API_KEY,
])
async def test_non_string_or_unsafe_request_id_is_omitted(caplog, bad_id):
    """Unsupported/unbounded/credential IDs cannot break or taint diagnostics."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    error = direct_error({"code": "upstream_error"})
    error.request_id = bad_id
    wrapped, _, _ = await consume_direct_error(error)

    assert str(wrapped) == GENERIC
    assert str(wrapped).splitlines() == [GENERIC]
    _, record = api_record(caplog)
    assert record["request_id"] is None
    assert_canaries_absent(caplog.text, str(wrapped))


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_number_name", ["huge_int", "nan", "inf", "-inf"])
async def test_unsafe_numeric_metadata_cannot_replace_provider_error(caplog, bad_number_name):
    """Huge/non-finite numerics are dropped; serialization never raises instead."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    # Keep the giant integer out of the decorator: pytest itself stringifies
    # parameter values when generating test IDs, which would mask collection.
    bad_number = {
        "huge_int": 10 ** 5000,
        "nan": float("nan"), "inf": float("inf"), "-inf": float("-inf"),
    }[bad_number_name]
    error = direct_error({"code": bad_number, "request_id": "req_numeric_safe"})
    wrapped, _, _ = await consume_direct_error(error)

    assert str(wrapped) == GENERIC
    assert wrapped.__cause__ is error
    _, record = api_record(caplog)  # Strict JSON parsing proves no NaN/Infinity token.
    assert "code" not in record
    assert record["request_id"] == "req_numeric_safe"


@pytest.mark.asyncio
async def test_safe_numeric_metadata_is_retained(caplog):
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    error = direct_error({"code": 529, "id": 123456789, "type": "server_error"})
    wrapped, _, _ = await consume_direct_error(error)

    _, record = api_record(caplog)
    assert record["code"] == 529
    assert record["id"] == 123456789
    assert str(wrapped) == GENERIC


@pytest.mark.asyncio
async def test_credential_flood_has_one_bounded_log_record(caplog):
    """Thousands of secret-shaped tokens create zero plaintext/log amplification."""
    caplog.set_level(logging.WARNING)
    body = " ".join([FAKE_API_KEY] * 2000)
    wrapped, _, _ = await consume_direct_error(direct_error(body))

    _, record = api_record(caplog)
    assert record["body_chars"] == len(body)
    assert len(caplog.records) == 1, "no per-credential-match secondary WARNING flood"
    assert len(caplog.text) < 1500
    assert_canaries_absent(caplog.text, str(wrapped))


@pytest.mark.asyncio
async def test_serializer_failure_still_raises_provider_error(caplog):
    """Final fail-safe: even a future record-construction bug cannot escape raw."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    error = direct_error({"code": "worker_down"})
    with patch("openalph.provider.json.dumps", side_effect=ValueError("forced serialization failure")):
        wrapped, _, _ = await consume_direct_error(error)

    assert str(wrapped) == GENERIC
    assert wrapped.__cause__ is error
    assert "diagnostic serialization failed" in caplog.text
    assert "forced serialization failure" not in caplog.text


@pytest.mark.asyncio
async def test_complete_uses_same_safe_wrapper(caplog):
    """complete() delegates to stream(): advisor/non-room callers are covered too."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    error = direct_error({"code": "worker_down", "message": FREEFORM})
    wrapped, client, _ = await consume_direct_error(error, use_complete=True)

    assert str(wrapped) == GENERIC
    assert wrapped.__cause__ is error
    assert client.chat.completions.create.await_count == 1
    assert_canaries_absent(caplog.text, str(wrapped))
