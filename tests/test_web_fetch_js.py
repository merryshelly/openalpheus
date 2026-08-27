"""Tests for web_fetch_js (Tabstack JS-rendered extraction) and the web_fetch
JS-shell handoff nudge.

Spec: memory/projects/openalph/web-fetch-js-design.md
  - §4  API contract (base URL, auth, endpoints, effort levels)
  - §5  web_fetch_js tool spec (schema, dispatch, description, handler)
  - §6  config coupling + rate-limit safety (_resolve_cached_api_key reuse)
  - §7  web_fetch handoff nudge (js_fallback_tool gate)
  - §9  this test plan (16 tests, numbered below to match)

All httpx calls are mocked -- no live network/API calls. The json-mode success
fixture below is the EXACT envelope captured from a real POST /extract/json
call against example.com (see BUILD STEP 1 in the task): the structured
extraction is the whole response body at top level, no .data/.content
wrapper. json.dumps(resp.json(), indent=2) is therefore the schema-mode
return value.

Tests 1-12 exercise web_fetch_js directly (and once through execute_tool for
the dispatch/config seam, test 12). Tests 13-16 exercise the web_fetch nudge.
"""

import json

import httpx
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from openalph.tools.web import web_fetch_js, web_fetch
from openalph.tools import ToolResult, execute_tool


BASE = "https://api.tabstack.ai/v1"

# The exact envelope captured live against example.com with schema {title, heading}
# (BUILD STEP 1, task preamble) -- confirms JSON mode has NO wrapper.
JSON_ENVELOPE = {"heading": "Example Domain", "title": "Example Domain"}


# ---------------------------------------------------------------------------
# Mocking helpers (POST-based; web_fetch_js is request/response, not streaming)
# ---------------------------------------------------------------------------

def mock_post_response(status_code=200, json_data=None):
    """Build a mock httpx.Response for a plain (non-streaming) POST call."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    return resp


def make_post_client(resp=None, side_effect=None):
    """Build a mocked httpx.AsyncClient context manager whose .post() returns
    resp, or raises side_effect if given (e.g. httpx.TimeoutException)."""
    client = AsyncMock()
    if side_effect is not None:
        client.post = AsyncMock(side_effect=side_effect)
    else:
        client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def mock_streaming_response(status_code=200, content=b""):
    """Mock for client.stream() used by web_fetch (mirrors test_executor_web.py)."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )

    captured = content

    async def _aiter_bytes():
        yield captured

    resp.aiter_bytes = _aiter_bytes

    stream_ctx = AsyncMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=resp)
    stream_ctx.__aexit__ = AsyncMock(return_value=False)
    return stream_ctx


def make_fetch_client(stream_ctx):
    """Wrap a streaming context in a patched AsyncClient (mirrors test_executor_web.py)."""
    client = MagicMock()
    client.stream = MagicMock(return_value=stream_ctx)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _completed(stdout="", returncode=0, stderr=""):
    """Build a subprocess.CompletedProcess like subprocess.run would return
    (mirrors test_api_key_cache.py's helper -- no cross-test-module imports)."""
    import subprocess
    return subprocess.CompletedProcess(
        args="cmd", returncode=returncode, stdout=stdout, stderr=stderr
    )


# ===========================================================================
# Tests 1-12: web_fetch_js tool
# ===========================================================================

class TestWebFetchJsFailSafes:

    @pytest.mark.asyncio
    async def test_1_no_api_key_returns_error_naming_fix(self):
        """§9 test 1: no api_key -> is_error, message names the config fix."""
        result = await web_fetch_js(url="https://example.com", api_key="")
        assert result.is_error is True
        assert "web_fetch_js not configured" in result.content
        assert "api_key_cmd" in result.content
        assert "workspace/tools/web_fetch_js.toml" in result.content

    @pytest.mark.asyncio
    async def test_2_empty_url_returns_error(self):
        """§9 test 2: empty url -> is_error, without needing a live call."""
        result = await web_fetch_js(url="", api_key="fake-key", base_url=BASE)
        assert result.is_error is True
        assert "url is required" in result.content


class TestWebFetchJsSuccessPaths:

    @pytest.mark.asyncio
    async def test_3_markdown_success(self):
        """§9 test 3: markdown mode POSTs to /extract/markdown with url/effort/
        nocache in the body and a Bearer auth header; returns .content verbatim."""
        mock_resp = mock_post_response(json_data={"content": "# Hello"})
        client = make_post_client(mock_resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE
            )

        assert result.is_error is False
        assert result.content == "# Hello"

        call = client.post.call_args
        assert call.args[0] == f"{BASE}/extract/markdown"
        body = call.kwargs["json"]
        assert body["url"] == "https://example.com"
        assert "effort" in body
        assert "nocache" in body
        assert call.kwargs["headers"]["Authorization"] == "Bearer fake-key"

    @pytest.mark.asyncio
    async def test_4_json_schema_success(self):
        """§9 test 4: schema provided -> POST /extract/json with json_schema in
        body; returns json.dumps(resp.json(), indent=2) of the TOP-LEVEL envelope
        (confirmed live: no .data/.content wrapper in schema mode)."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "heading": {"type": "string"},
            },
        }
        mock_resp = mock_post_response(json_data=JSON_ENVELOPE)
        client = make_post_client(mock_resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
                schema=schema,
            )

        assert result.is_error is False
        assert result.content == json.dumps(JSON_ENVELOPE, indent=2)

        call = client.post.call_args
        assert call.args[0] == f"{BASE}/extract/json"
        body = call.kwargs["json"]
        assert body["json_schema"] == schema
        assert body["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_5_schema_as_json_string_parsed_or_rejected(self):
        """§9 test 5: schema given as a JSON string is parsed and forwarded as
        a dict; an invalid JSON string -> is_error (defensive)."""
        schema_dict = {"type": "object", "properties": {"a": {"type": "string"}}}
        schema_str = json.dumps(schema_dict)
        mock_resp = mock_post_response(json_data={"a": "x"})
        client = make_post_client(mock_resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
                schema=schema_str,
            )

        assert result.is_error is False
        body = client.post.call_args.kwargs["json"]
        assert body["json_schema"] == schema_dict

        # Invalid JSON string -> is_error, no network call needed to fail this
        result2 = await web_fetch_js(
            url="https://example.com", api_key="fake-key", base_url=BASE,
            schema="not valid json {",
        )
        assert result2.is_error is True
        assert "schema must be a JSON Schema object" in result2.content

    @pytest.mark.asyncio
    async def test_6_effort_default_max_and_invalid_coerced(self):
        """§9 test 6: effort omitted -> "max"; an invalid effort value is
        coerced to "max" rather than erroring."""
        mock_resp = mock_post_response(json_data={"content": "ok"})

        client1 = make_post_client(mock_resp)
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client1
            result1 = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE
            )
        assert result1.is_error is False
        assert client1.post.call_args.kwargs["json"]["effort"] == "max"

        client2 = make_post_client(mock_resp)
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client2
            result2 = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
                effort="ludicrous",
            )
        assert result2.is_error is False
        assert client2.post.call_args.kwargs["json"]["effort"] == "max"

    @pytest.mark.asyncio
    async def test_7_max_chars_truncates_markdown(self):
        """§9 test 7: markdown longer than max_chars is truncated head+tail
        with a marker (same shape as web_fetch's own truncation)."""
        long_text = "A" * 1000
        mock_resp = mock_post_response(json_data={"content": long_text})
        client = make_post_client(mock_resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
                max_chars=200,
            )

        assert result.is_error is False
        assert "[truncated" in result.content
        assert len(result.content) < 300

    @pytest.mark.asyncio
    async def test_11_nocache_passed_through(self):
        """§9 test 11: nocache=True is forwarded verbatim in the request body."""
        mock_resp = mock_post_response(json_data={"content": "ok"})
        client = make_post_client(mock_resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
                nocache=True,
            )

        assert result.is_error is False
        assert client.post.call_args.kwargs["json"]["nocache"] is True


class TestWebFetchJsErrorSteering:

    @pytest.mark.asyncio
    async def test_8_http_401_403_auth_steering(self):
        """§9 test 8: 401/403 -> is_error with auth-fix steering text."""
        for code in (401, 403):
            mock_resp = mock_post_response(status_code=code)
            client = make_post_client(mock_resp)
            with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
                MockClient.return_value = client
                result = await web_fetch_js(
                    url="https://example.com", api_key="fake-key", base_url=BASE
                )
            assert result.is_error is True, f"HTTP {code} should be is_error"
            assert "authentication failed" in result.content, code
            assert "api_key_cmd" in result.content, code

    @pytest.mark.asyncio
    async def test_9_http_402_429_quota_rate_limit_steering(self):
        """§9 test 9: 402 -> quota-exhausted steering; 429 -> rate-limited steering."""
        cases = {402: "quota exhausted", 429: "rate-limited"}
        for code, phrase in cases.items():
            mock_resp = mock_post_response(status_code=code)
            client = make_post_client(mock_resp)
            with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
                MockClient.return_value = client
                result = await web_fetch_js(
                    url="https://example.com", api_key="fake-key", base_url=BASE
                )
            assert result.is_error is True, f"HTTP {code} should be is_error"
            assert phrase in result.content, f"HTTP {code}: {result.content!r}"
            assert "Tabstack" in result.content, code

    @pytest.mark.asyncio
    async def test_10_timeout_steering(self):
        """§9 test 10: an httpx timeout (or HTTP 408) -> 'try effort=standard' steering."""
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE
            )
        assert result.is_error is True
        assert "effort='standard'" in result.content

        # HTTP 408 is steered identically per spec ("408 or httpx.TimeoutException")
        mock_resp_408 = mock_post_response(status_code=408)
        client2 = make_post_client(mock_resp_408)
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client2
            result2 = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE
            )
        assert result2.is_error is True
        assert "effort='standard'" in result2.content


class TestWebFetchJsDispatchSeam:

    @pytest.mark.asyncio
    async def test_12_dispatch_uses_cached_resolver_not_per_call_op_read(self):
        """§9 test 12: through execute_tool, the api_key_cmd is resolved via the
        TTL-cached _resolve_cached_api_key -- NOT a fresh `op read` per call
        (the load-bearing rate-limit-safety invariant, §6). Two calls with the
        same tool_config must hit the resolver subprocess only once, and the
        handler must receive the resolved key (visible in the Bearer header)."""
        import openalph.tools as tools_mod
        tools_mod._api_key_cache.clear()
        try:
            mock_resp = mock_post_response(json_data={"content": "rendered"})
            client = make_post_client(mock_resp)

            with patch("openalph.tools.subprocess.run",
                       return_value=_completed(stdout="RESOLVED-KEY")) as run, \
                 patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
                MockClient.return_value = client
                cfg = {"api_key_cmd": "op read x", "base_url": BASE}
                r1 = await execute_tool(
                    "web_fetch_js", {"url": "https://a.example.com"}, cfg, None
                )
                r2 = await execute_tool(
                    "web_fetch_js", {"url": "https://b.example.com"}, cfg, None
                )

            assert r1.is_error is False
            assert r2.is_error is False
            assert run.call_count == 1, (
                "api_key_cmd must be resolved via the TTL cache, not per call"
            )
            headers = client.post.call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer RESOLVED-KEY"
        finally:
            tools_mod._api_key_cache.clear()


# ===========================================================================
# Tests 13-16: web_fetch JS-shell handoff nudge (workspace-kdsn.202.1)
# ===========================================================================

class TestWebFetchJsShellNudge:

    @pytest.mark.asyncio
    async def test_13_nudge_appended_on_js_shell_page(self):
        """§9 test 13: an unrendered-SPA-shaped page (shell marker present,
        negligible readable text) with js_fallback_tool configured gets the
        nudge appended; the fetch itself is still a success (is_error=False)."""
        html = '<html><body><div id="root"></div></body></html>'

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = make_fetch_client(
                mock_streaming_response(content=html.encode())
            )
            result = await web_fetch(
                "https://spa.example.com",
                tool_config={"js_fallback_tool": "web_fetch_js"},
            )

        assert result.is_error is False
        assert "[web_fetch note:" in result.content
        assert "JavaScript-rendered" in result.content
        assert "web_fetch_js" in result.content

    @pytest.mark.asyncio
    async def test_14_no_nudge_on_normal_content_page(self):
        """§9 test 14: a normal, richly-texted page gets NO nudge even with
        js_fallback_tool configured."""
        paragraph = (
            "This is a perfectly ordinary, richly informative article about "
            "widgets and their history. " * 5
        )
        html = f"<html><body><p>{paragraph}</p></body></html>"

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = make_fetch_client(
                mock_streaming_response(content=html.encode())
            )
            result = await web_fetch(
                "https://normal.example.com",
                tool_config={"js_fallback_tool": "web_fetch_js"},
            )

        assert result.is_error is False
        assert "[web_fetch note:" not in result.content

    @pytest.mark.asyncio
    async def test_15_nudge_gated_off_when_config_unset(self):
        """§9 test 15: js_fallback_tool unset (no tool_config, or tool_config
        without the key) -> never appends the nudge, even on a shell-like page."""
        html = '<html><body><div id="root"></div></body></html>'

        # No tool_config at all
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = make_fetch_client(
                mock_streaming_response(content=html.encode())
            )
            result = await web_fetch("https://spa.example.com")
        assert result.is_error is False
        assert "[web_fetch note:" not in result.content

        # tool_config present but without js_fallback_tool
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = make_fetch_client(
                mock_streaming_response(content=html.encode())
            )
            result2 = await web_fetch("https://spa.example.com", tool_config={})
        assert result2.is_error is False
        assert "[web_fetch note:" not in result2.content

    @pytest.mark.asyncio
    async def test_16_nudge_preserves_content_and_truncation(self):
        """§9 test 16: note is appended (content preserved) and max_chars
        truncation still applies; the note itself must survive intact (not be
        sliced by the head+tail truncation math)."""
        filler = "Loading please wait. " * 50
        html = f'<html><body><div id="__next">{filler}</div></body></html>'

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = make_fetch_client(
                mock_streaming_response(content=html.encode())
            )
            result = await web_fetch(
                "https://spa.example.com",
                max_chars=100,
                tool_config={"js_fallback_tool": "web_fetch_js"},
            )

        assert result.is_error is False
        assert "[truncated" in result.content
        assert "[web_fetch note:" in result.content
        assert "web_fetch_js" in result.content
        # The note must appear intact, after the truncation marker -- never cut.
        assert result.content.index("[web_fetch note:") > result.content.index("[truncated")
        assert result.content.endswith("full browser rendering.]")


# ===========================================================================
# Audit-remediation regression tests (workspace-kdsn.202 cold-read audit).
# Each of these fails against the pre-fix handler -- see the fix commit for
# the audit finding IDs (H1/H2/H3/M1/M2/L1/L2/L3) referenced in each test.
# ===========================================================================

class TestWebFetchJsBaseUrlPin:
    """H3: base_url must be pinned to the canonical Tabstack host BEFORE any
    request is made, so a misconfigured/overridden base_url can never
    exfiltrate the Bearer api_key to an arbitrary host."""

    @pytest.mark.asyncio
    async def test_17_noncanonical_base_url_rejected_no_request_sent(self):
        mock_resp = mock_post_response(json_data={"content": "should never be seen"})
        client = make_post_client(mock_resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com",
                api_key="fake-key",
                base_url="https://evil.example.com/v1",
            )

        assert result.is_error is True
        assert "non-canonical base_url" in result.content
        assert "https://api.tabstack.ai" in result.content
        # No request may have been made -- the key must never leave the process.
        assert MockClient.called is False
        assert client.post.called is False

    @pytest.mark.asyncio
    async def test_17b_http_scheme_on_canonical_host_also_rejected(self):
        """Belt-and-suspenders: the pin checks scheme too, not just host --
        plain http to the right host must still be refused (no downgrade)."""
        client = make_post_client(mock_post_response(json_data={"content": "x"}))

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com",
                api_key="fake-key",
                base_url="http://api.tabstack.ai/v1",
            )

        assert result.is_error is True
        assert "non-canonical base_url" in result.content
        assert MockClient.called is False

    @pytest.mark.asyncio
    async def test_17c_canonical_base_url_still_works(self):
        """Sanity: the exact canonical base_url (as used by every other test
        in this file) is unaffected by the pin."""
        mock_resp = mock_post_response(json_data={"content": "ok"})
        client = make_post_client(mock_resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE
            )

        assert result.is_error is False
        assert result.content == "ok"


class TestWebFetchJsApiKeyHygiene:
    """H2: an api_key containing a header-invalid control char must be
    rejected before any request, and the raw key must never leak through a
    generic exception message."""

    @pytest.mark.asyncio
    async def test_18_api_key_with_newline_rejected_no_request_sent(self):
        client = make_post_client(mock_post_response(json_data={"content": "x"}))

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com",
                api_key="valid-looking-key\nX-Evil: 1",
                base_url=BASE,
            )

        assert result.is_error is True
        assert "invalid api_key format" in result.content
        assert MockClient.called is False
        assert client.post.called is False

    @pytest.mark.asyncio
    async def test_18b_api_key_with_carriage_return_rejected(self):
        client = make_post_client(mock_post_response(json_data={"content": "x"}))
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="key\rwith-cr", base_url=BASE
            )
        assert result.is_error is True
        assert "invalid api_key format" in result.content
        assert MockClient.called is False
        assert client.post.called is False

    @pytest.mark.asyncio
    async def test_19_generic_exception_scrubs_raw_api_key_from_message(self):
        """H2: a low-level (non-HTTPStatusError, non-Timeout) exception that
        happens to echo the raw key in its message must have the key scrubbed
        before it reaches the caller. Uses a short key that the outer
        redact_credentials generic patterns (20+ char bearer/hex) would NOT
        independently catch, so this pins web_fetch_js's own scrubbing."""
        key = "sk-abc123"
        client = AsyncMock()
        client.post = AsyncMock(
            side_effect=RuntimeError(f"protocol error near header: Bearer {key}")
        )
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(url="https://example.com", api_key=key, base_url=BASE)

        assert result.is_error is True
        assert key not in result.content
        assert "[REDACTED]" in result.content


class TestWebFetchJsMalformedEnvelope:
    """H1/L1: a malformed markdown envelope (non-dict body, or "content"
    that isn't a string) must fail soft, and the result must survive the
    outer execute_tool -> redact_credentials tail without raising (a non-str
    ToolResult.content reaching redact_credentials raises TypeError there,
    which would otherwise kill the agent's turn)."""

    @pytest.mark.asyncio
    async def test_20_non_string_content_field_is_error_not_crash(self):
        mock_resp = mock_post_response(json_data={"content": ["x"]})
        client = make_post_client(mock_resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE
            )

        assert result.is_error is True
        assert "malformed response" in result.content
        assert isinstance(result.content, str)

    @pytest.mark.asyncio
    async def test_21_missing_content_key_is_error_not_empty_success(self):
        mock_resp = mock_post_response(json_data={"notcontent": 1})
        client = make_post_client(mock_resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE
            )

        # Must NOT be a silent empty success (pre-fix: is_error False, content "").
        assert result.is_error is True
        assert "malformed response" in result.content

    @pytest.mark.asyncio
    async def test_22_non_json_body_is_error_not_raise(self):
        """resp.json() raising (non-JSON body) must fail soft too."""
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.side_effect = json.JSONDecodeError("bad", "doc", 0)
        client = make_post_client(resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE
            )

        assert result.is_error is True
        assert "malformed response" in result.content

    @pytest.mark.asyncio
    async def test_23_malformed_envelope_survives_execute_tool_redaction_tail(self):
        """The load-bearing end-to-end check: through execute_tool (which
        unconditionally pipes result.content through redact_credentials),
        a non-string content field must come out as a clean is_error --
        never an uncaught TypeError killing the turn."""
        mock_resp = mock_post_response(json_data={"content": ["x"]})
        client = make_post_client(mock_resp)
        cfg = {"api_key": "fake-key", "base_url": BASE}

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            # No try/except here: if this raises, the test fails loudly --
            # which is exactly the pre-fix behaviour (TypeError propagates
            # out of execute_tool).
            result = await execute_tool(
                "web_fetch_js", {"url": "https://example.com"}, cfg, None
            )

        assert isinstance(result, ToolResult)
        assert isinstance(result.content, str)
        assert result.is_error is True
        assert "malformed response" in result.content


class TestWebFetchJsUrlValidation:
    """M1: url must be a full http(s) URL with a real host and no embedded
    credentials."""

    @pytest.mark.asyncio
    async def test_24_non_http_scheme_rejected(self):
        client = make_post_client(mock_post_response(json_data={"content": "x"}))
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(url="ftp://x", api_key="fake-key", base_url=BASE)
        assert result.is_error is True
        assert "embedded credentials" in result.content
        assert MockClient.called is False

    @pytest.mark.asyncio
    async def test_25_schemeless_url_rejected(self):
        client = make_post_client(mock_post_response(json_data={"content": "x"}))
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(url="example.com", api_key="fake-key", base_url=BASE)
        assert result.is_error is True
        assert "embedded credentials" in result.content
        assert MockClient.called is False

    @pytest.mark.asyncio
    async def test_26_url_with_embedded_credentials_rejected(self):
        client = make_post_client(mock_post_response(json_data={"content": "x"}))
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://user:pass@host/p", api_key="fake-key", base_url=BASE
            )
        assert result.is_error is True
        assert "embedded credentials" in result.content
        assert MockClient.called is False

    @pytest.mark.asyncio
    async def test_26b_empty_url_message_unchanged(self):
        """Regression guard: the pre-existing empty-url message/behaviour
        (test 2) must be untouched by the new validation."""
        result = await web_fetch_js(url="", api_key="fake-key", base_url=BASE)
        assert result.is_error is True
        assert "url is required" in result.content


class TestWebFetchJsMaxCharsBounds:
    """M2: max_chars truncation only applies for a real positive int; a bad
    optional value must never error and must never produce output larger
    than the untruncated source."""

    @pytest.mark.asyncio
    async def test_27_max_chars_zero_no_truncation_no_oversized_output(self):
        body = "A" * 1000
        client = make_post_client(mock_post_response(json_data={"content": body}))
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE, max_chars=0
            )
        assert result.is_error is False
        assert len(result.content) <= len(body)
        assert result.content == body
        assert "[truncated" not in result.content

    @pytest.mark.asyncio
    async def test_28_max_chars_negative_no_truncation_no_oversized_output(self):
        body = "A" * 1000
        client = make_post_client(mock_post_response(json_data={"content": body}))
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE, max_chars=-5
            )
        assert result.is_error is False
        assert len(result.content) <= len(body)
        assert result.content == body
        assert "[truncated" not in result.content

    @pytest.mark.asyncio
    async def test_28b_max_chars_positive_still_truncates(self):
        """Regression guard: a sane positive max_chars (test 7's behaviour)
        must be unaffected by the bounds check."""
        body = "A" * 1000
        client = make_post_client(mock_post_response(json_data={"content": body}))
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE, max_chars=200
            )
        assert result.is_error is False
        assert "[truncated" in result.content
        assert len(result.content) < len(body)


class TestWebFetchJsSchemaPresence:
    """L2: JSON mode is selected by schema is not None (presence), not
    truthiness -- an intentional empty schema {} must still route to
    /extract/json and be included in the request body."""

    @pytest.mark.asyncio
    async def test_29_empty_dict_schema_uses_json_endpoint(self):
        mock_resp = mock_post_response(json_data={"a": "b"})
        client = make_post_client(mock_resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE, schema={}
            )

        assert result.is_error is False
        call = client.post.call_args
        assert call.args[0] == f"{BASE}/extract/json"
        body = call.kwargs["json"]
        assert "json_schema" in body
        assert body["json_schema"] == {}
        # Schema-mode return shape (json.dumps of the envelope), not markdown.
        assert result.content == json.dumps({"a": "b"}, indent=2)


class TestWebFetchJsNocacheRealBool:
    """L3: nocache must be a real bool; any non-bool (notably the string
    "false", which Python's bool() would wrongly coerce to True) is treated
    as False. Covers both the handler directly and the __init__.py dispatch
    seam (which must forward the raw value, not force-bool() it)."""

    @pytest.mark.asyncio
    async def test_30_string_false_handler_direct(self):
        mock_resp = mock_post_response(json_data={"content": "ok"})
        client = make_post_client(mock_resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
                nocache="false",
            )

        assert result.is_error is False
        body = client.post.call_args.kwargs["json"]
        assert body["nocache"] is False

    @pytest.mark.asyncio
    async def test_31_string_false_through_dispatch(self):
        """Same as test 30 but through execute_tool, so a force-bool() left
        behind in __init__.py's dispatch (upstream of the handler's own
        coercion) would still be caught."""
        mock_resp = mock_post_response(json_data={"content": "ok"})
        client = make_post_client(mock_resp)
        cfg = {"api_key": "fake-key", "base_url": BASE}

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await execute_tool(
                "web_fetch_js",
                {"url": "https://example.com", "nocache": "false"},
                cfg,
                None,
            )

        assert result.is_error is False
        body = client.post.call_args.kwargs["json"]
        assert body["nocache"] is False

    @pytest.mark.asyncio
    async def test_31b_real_true_still_true(self):
        """Regression guard: an actual bool True (test 11's behaviour) must
        still be forwarded as True, not flattened by the new coercion."""
        mock_resp = mock_post_response(json_data={"content": "ok"})
        client = make_post_client(mock_resp)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
                nocache=True,
            )

        assert result.is_error is False
        assert client.post.call_args.kwargs["json"]["nocache"] is True


# ===========================================================================
# BUG-4 — web_fetch with max_chars <= 0 must not return MORE than the input
# ===========================================================================

class _FakeStreamResp:
    def __init__(self, body: bytes):
        self._body = body
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    def raise_for_status(self): return None
    async def aiter_bytes(self):
        yield self._body


class _FakeAsyncClient:
    def __init__(self, body: bytes):
        self._body = body
    def __call__(self, *a, **k): return self
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    def stream(self, method, url):
        return _FakeStreamResp(self._body)


def _fetch(body_html: str, max_chars):
    import asyncio
    from unittest.mock import patch
    client = _FakeAsyncClient(body_html.encode("utf-8"))
    with patch("openalph.tools.web.httpx.AsyncClient", client):
        # asyncio.run() always spins up (and tears down) a fresh event loop.
        # The previous asyncio.get_event_loop().run_until_complete(...) pattern
        # depends on an ambient loop existing on the main thread — order-
        # dependent, and Python 3.13 raises RuntimeError outright when no
        # loop is current (surfaced here as a full-suite-only failure,
        # invisible when this test file was run in isolation).
        return asyncio.run(web_fetch("https://example.com", max_chars=max_chars))


class TestWebFetchMaxCharsGuard:

    def test_zero_max_chars_does_not_expand_output(self):
        src = "abcdefghij" * 60  # 600 chars
        out = _fetch(f"<html><body>{src}</body></html>", 0)
        # BUG: max_chars=0 used to return the whole page + a "truncated" marker,
        # i.e. LONGER than the source. It must not exceed the source length.
        assert len(out.content) <= len(src) + 10

    def test_negative_max_chars_does_not_expand_output(self):
        src = "klmnopqrst" * 60
        out = _fetch(f"<html><body>{src}</body></html>", -10)
        assert len(out.content) <= len(src) + 10

    def test_positive_max_chars_truncates(self):
        src = "z" * 600
        out = _fetch(f"<html><body>{src}</body></html>", 100)
        assert "truncated" in out.content
        assert len(out.content) < len(src)


class TestWebFetchJsOffset:
    """kdsn.291: char-windowing offset for web_fetch_js markdown mode.
    Schema mode + offset -> is_error (silently-ignored would mislead);
    schema mode + max_chars stays ignored (existing documented behavior)."""

    @pytest.mark.asyncio
    async def test_markdown_window(self):
        text = "M" * 1000
        mock_resp = mock_post_response(json_data={"content": text})
        client = make_post_client(mock_resp)
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
                offset=400, max_chars=200,
            )
        assert result.is_error is False
        assert result.content.startswith(
            "[web_fetch_js window: chars 400-599 of 1000 | 400 before | 400 after — continue with offset=600]"
        )
        assert result.content.endswith("M" * 200)

    @pytest.mark.asyncio
    async def test_schema_mode_offset_errors_without_request(self):
        mock_resp = mock_post_response(json_data={"content": "x"})
        client = make_post_client(mock_resp)
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
                schema={"type": "object", "properties": {}}, offset=10,
            )
        assert result.is_error is True
        assert "markdown-mode only" in result.content
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_schema_mode_max_chars_still_ignored(self):
        long_json = {"a": "x" * 5000}
        mock_resp = mock_post_response(json_data=long_json)
        client = make_post_client(mock_resp)
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
                schema={"type": "object", "properties": {}}, max_chars=10,
            )
        assert result.is_error is False
        assert "x" * 5000 in result.content  # no truncation in schema mode

    @pytest.mark.asyncio
    async def test_markdown_offset_beyond_end(self):
        mock_resp = mock_post_response(json_data={"content": "M" * 1000})
        client = make_post_client(mock_resp)
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
                offset=5000,
            )
        assert result.is_error is True
        assert "1000" in result.content


class TestWebFetchJsOffsetReview:
    """kdsn.291 review fixes for web_fetch_js: invalid offsets rejected
    pre-request, offset flows through the real dispatch seam, huge markdown
    full text carries the head-anchored navigation note."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", ["x", True, -5])
    async def test_markdown_invalid_offset_errors_without_request(self, bad):
        client = make_post_client(mock_post_response(json_data={"content": "M" * 1000}))
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
                offset=bad,
            )
        assert result.is_error is True
        assert "offset must be a non-negative integer" in result.content
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_execute_tool_windows_markdown(self):
        from openalph.config import AgentConfig, ProviderConfig

        cfg = AgentConfig(
            name="test-agent",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=8192,
            providers={
                "anthropic": ProviderConfig(
                    key="anthropic", type="anthropic", api_key="sk-test",
                    base_url=None, quirks=[],
                )
            },
            workspace="/tmp",
            max_iterations=25,
            truncation_limit=50000,
        )
        client = make_post_client(mock_post_response(json_data={"content": "M" * 1000}))
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await execute_tool(
                name="web_fetch_js",
                input={"url": "https://example.com", "offset": 400, "max_chars": 200},
                tool_config={"api_key": "fake-key", "base_url": BASE},
                agent_config=cfg,
            )
        assert result.is_error is False
        assert result.content.startswith("[web_fetch_js window: chars 400-599 of 1000")

    @pytest.mark.asyncio
    async def test_huge_markdown_full_text_note(self):
        client = make_post_client(
            mock_post_response(json_data={"content": "M" * 60_000})
        )
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com", api_key="fake-key", base_url=BASE,
            )
        assert result.is_error is False
        assert result.content.startswith("[web_fetch_js note: full markdown is 60000 chars")
        assert "offset=30000 for the middle" in result.content
