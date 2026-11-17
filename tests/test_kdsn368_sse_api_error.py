"""kdsn.368 — OpenAI SDK streaming SSE ``APIError`` diagnostics.

These tests drive the REAL OpenAI SDK 2.26.0 SSE parser through an httpx
MockTransport (zero network), then exercise OA's public provider.stream().
The original provider code lets the SDK APIError escape; the acceptance
contract is safe diagnostics + ProviderError, with NO new retry behavior.
"""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.provider import ProviderError, StreamEvent, stream


FAKE_API_KEY = "sk-proj-" + ("f" * 24)
PRIVATE_PROMPT = "PRIVATE_PROMPT_NEVER_LOG_71c04db1"
PRIVATE_DEBUG_VALUE = "PRIVATE_DEBUG_NEVER_LOG_a4257fd2"


def make_config() -> AgentConfig:
    return AgentConfig(
        name="kdsn368-test",
        default_model="synthetic/hf:moonshotai/Kimi-K3.1-Pro",
        max_tokens=256,
        model_max_tokens=524_288,
        providers={
            "synthetic": ProviderConfig(
                key="synthetic",
                type="openai",
                api_key=FAKE_API_KEY,
                base_url="https://unit.invalid/v1",
                quirks=[],
            )
        },
        workspace=Path("/tmp/kdsn368-test"),
    )


def sdk_client_for_sse(payload: str, requests: list[httpx.Request]):
    """Real AsyncOpenAI client whose only transport is the supplied SSE body."""
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=payload,
            request=request,
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    return openai.AsyncOpenAI(
        api_key=FAKE_API_KEY,
        base_url="https://unit.invalid/v1",
        http_client=http_client,
        max_retries=0,
    )


def openai_chunk(text: str) -> str:
    return "data: " + json.dumps({
        "id": "chatcmpl-kdsn368",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "hf:moonshotai/Kimi-K3.1-Pro",
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": None,
        }],
    }) + "\n\n"


@pytest.mark.asyncio
async def test_mapping_error_zero_chunks_safe_log_and_provider_error(caplog):
    """Real SDK {error: mapping} at zero chunks: capture diagnostics, no retry."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    reset_cb = AsyncMock()
    requests: list[httpx.Request] = []
    payload = "data: " + json.dumps({"error": {
        "code": "worker_down",
        "message": "worker unavailable",
        "type": "server_error",
        "request_id": "req_kdsn368_123",
        # Unknown/debug branches must never be dumped into journald.
        "debug": {"prompt": PRIVATE_PROMPT, "value": PRIVATE_DEBUG_VALUE},
        "authorization": FAKE_API_KEY,
    }}) + "\n\n"
    client = sdk_client_for_sse(payload, requests)
    try:
        with patch("openalph.provider._get_client", return_value=client):
            with pytest.raises(ProviderError, match="worker unavailable") as exc_info:
                async for _ in stream(
                    config=make_config(), system="SYSTEM_NEVER_REQUEST_LOG",
                    messages=[{"role": "user", "content": PRIVATE_PROMPT}],
                    thinking="xhigh", on_stream_reset=reset_cb,
                ):
                    pass
    finally:
        await client.close()

    assert type(exc_info.value.__cause__).__name__ == "APIError"
    assert len(requests) == 1, "diagnostics/wrapping must not add a retry"
    reset_cb.assert_not_awaited()
    text = caplog.text
    assert "after 0 chunk(s)" in text
    assert "APIError" in text
    assert "worker_down" in text
    assert "worker unavailable" in text
    assert "server_error" in text
    assert "req_kdsn368_123" in text
    assert "dict" in text
    # No request dump and no arbitrary body branch: no prompt, credentials,
    # headers, or embedded debug payload in journald.
    assert PRIVATE_PROMPT not in text
    assert PRIVATE_DEBUG_VALUE not in text
    assert FAKE_API_KEY not in text
    assert "Authorization" not in text


@pytest.mark.asyncio
async def test_string_error_after_chunk_safe_log_no_retry(caplog):
    """Real SDK {error: string} after one content chunk: body text is useful,
    credential-shaped content is redacted, and kdsn.345 stays disengaged."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    reset_cb = AsyncMock()
    requests: list[httpx.Request] = []
    payload = (
        openai_chunk("partial")
        + "data: " + json.dumps({
            "error": "upstream worker closed; rejected credential " + FAKE_API_KEY,
        }) + "\n\n"
    )
    client = sdk_client_for_sse(payload, requests)
    events: list[StreamEvent] = []
    try:
        with patch("openalph.provider._get_client", return_value=client):
            with pytest.raises(ProviderError, match="upstream worker closed") as exc_info:
                async for event in stream(
                    config=make_config(), system="SYSTEM_NEVER_REQUEST_LOG",
                    messages=[{"role": "user", "content": PRIVATE_PROMPT}],
                    thinking="xhigh", on_stream_reset=reset_cb,
                ):
                    events.append(event)
    finally:
        await client.close()

    assert [e.content for e in events if e.type == "text"] == ["partial"]
    assert type(exc_info.value.__cause__).__name__ == "APIError"
    assert FAKE_API_KEY not in str(exc_info.value), "surface must be sanitized too"
    assert len(requests) == 1, "no new retry even after output began"
    reset_cb.assert_not_awaited()
    text = caplog.text
    assert "after 1 chunk(s)" in text
    assert "upstream worker closed" in text
    assert "str" in text
    assert PRIVATE_PROMPT not in text
    assert FAKE_API_KEY not in text
    assert "Authorization" not in text


class RaisingStream:
    def __init__(self, chunks, error):
        self._chunks = chunks
        self._error = error

    def __aiter__(self):
        return self._impl()

    async def _impl(self):
        for chunk in self._chunks:
            yield chunk
        raise self._error


class RaisingCreate:
    def __init__(self, error):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = AsyncMock(side_effect=error)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [None, ["unlogged"], {"nested": {"drop": PRIVATE_DEBUG_VALUE}}])
async def test_unsupported_body_shapes_fail_closed(caplog, body):
    """No raw repr/dump for null, list, or unexpected nested mappings."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    request = httpx.Request("POST", "https://unit.invalid/v1/chat/completions")
    error = openai.APIError(
        "An error occurred during streaming", request=request, body=body,
    )
    client = RaisingCreate(error)
    with patch("openalph.provider._get_client", return_value=client):
        with pytest.raises(ProviderError):
            async for _ in stream(
                config=make_config(), system="test",
                messages=[{"role": "user", "content": PRIVATE_PROMPT}],
            ):
                pass

    assert client.chat.completions.create.await_count == 1
    assert PRIVATE_PROMPT not in caplog.text
    assert PRIVATE_DEBUG_VALUE not in caplog.text


@pytest.mark.asyncio
async def test_long_string_body_bounded_and_single_line(caplog):
    """Adjoint of real-SSE body-string coverage: bound raw diagnostic volume
    and JSON-escape control characters instead of forging multiline logs."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    request = httpx.Request("POST", "https://unit.invalid/v1/chat/completions")
    error = openai.APIError(
        "An error occurred during streaming", request=request,
        body=("x" * 2500) + "\nFORGED_LOG_LINE",
    )
    client = RaisingCreate(error)
    with patch("openalph.provider._get_client", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            async for _ in stream(
                config=make_config(), system="test",
                messages=[{"role": "user", "content": "hello"}],
            ):
                pass

    assert len(str(exc_info.value)) <= 1100
    assert "\n" not in str(exc_info.value)
    assert len(caplog.text) < 5000
    assert "x" * 1100 not in caplog.text
    assert "\\nFORGED_LOG_LINE" in caplog.text


@pytest.mark.asyncio
async def test_available_exception_request_id_is_logged(caplog):
    """Generic SSE APIError normally lacks request_id; if a subclass/caller
    exposes one, preserve it without inspecting request authorization headers."""
    caplog.set_level(logging.WARNING, logger="openalph.provider")
    request = httpx.Request(
        "POST", "https://unit.invalid/v1/chat/completions",
        headers={"Authorization": "Bearer " + FAKE_API_KEY},
    )
    error = openai.APIError("bad stream", request=request, body={"message": "bad stream"})
    error.request_id = "req_kdsn368_attr"
    client = RaisingCreate(error)
    with patch("openalph.provider._get_client", return_value=client):
        with pytest.raises(ProviderError):
            async for _ in stream(
                config=make_config(), system="test",
                messages=[{"role": "user", "content": PRIVATE_PROMPT}],
            ):
                pass

    assert "req_kdsn368_attr" in caplog.text
    assert FAKE_API_KEY not in caplog.text
    assert PRIVATE_PROMPT not in caplog.text
    assert "Authorization" not in caplog.text
