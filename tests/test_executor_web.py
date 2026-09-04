"""Tests for web tools (search + fetch).

Tests mock httpx.AsyncClient to avoid real network calls while testing
the actual formatting, parsing, error handling, and truncation logic.
"""

import inspect
import re

import httpx
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from openalph.tools.web import (
    web_search,
    web_fetch,
    web_fetch_js,
    _strip_html_tags,
    _budgeted_window_text,
    MAX_RESPONSE_BYTES,
)
from openalph.tools import execute_tool, truncate_result


def mock_httpx_response(status_code=200, json_data=None, text="", content=None):
    """Create a mock httpx response (for non-streaming calls, e.g. web_search)."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    # content is the raw bytes; default to UTF-8 encoding of text
    resp.content = content if content is not None else text.encode("utf-8")
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    return resp


def mock_streaming_response(status_code=200, content=b""):
    """Create a mock for client.stream() used by web_fetch.

    Returns an async context manager whose __aenter__ yields a response with
    raise_for_status() and aiter_bytes() async generator.
    """
    import httpx

    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )

    # aiter_bytes must be an async generator; yield content in one chunk
    captured = content  # capture for closure

    async def _aiter_bytes():
        yield captured

    resp.aiter_bytes = _aiter_bytes

    # Wrap in an async context manager
    stream_ctx = AsyncMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=resp)
    stream_ctx.__aexit__ = AsyncMock(return_value=False)
    return stream_ctx


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


def brave_results(*items):
    """Build a Brave-format search response."""
    results = []
    for title, url, snippet in items:
        results.append({"title": title, "url": url, "description": snippet})
    return {"web": {"results": results}}


class TestWebSearch:

    @pytest.mark.asyncio
    async def test_returns_formatted_results(self):
        """Search results are formatted with title, URL, snippet."""
        data = brave_results(("Test Page", "https://test.com", "A test snippet"))
        mock_resp = mock_httpx_response(json_data=data)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get.return_value = mock_resp
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await web_search("test", api_key="fake-key")

        assert result.is_error is False
        assert "Test Page" in result.content
        assert "https://test.com" in result.content
        assert "A test snippet" in result.content

    @pytest.mark.asyncio
    async def test_passes_query_and_count(self):
        """Query and count are sent as params."""
        data = brave_results(("R", "https://r.com", "r"))
        mock_resp = mock_httpx_response(json_data=data)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get.return_value = mock_resp
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            await web_search("Grace Hopper", count=3, api_key="fake-key")

        call_kwargs = client.get.call_args
        assert call_kwargs.kwargs["params"]["q"] == "Grace Hopper"
        assert call_kwargs.kwargs["params"]["count"] == 3

    @pytest.mark.asyncio
    async def test_no_api_key_returns_error(self):
        """Missing API key returns error without making a request."""
        result = await web_search("test", api_key="")
        assert result.is_error is True
        assert "no api key" in result.content.lower()

    @pytest.mark.asyncio
    async def test_api_error_returns_tool_error(self):
        """HTTP error from API returns is_error=True."""
        mock_resp = mock_httpx_response(status_code=429)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get.return_value = mock_resp
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await web_search("test", api_key="fake-key")

        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """No results → informative message, not an error."""
        data = {"web": {"results": []}}
        mock_resp = mock_httpx_response(json_data=data)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get.return_value = mock_resp
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await web_search("test", api_key="fake-key")

        assert result.is_error is False
        assert "no results" in result.content.lower()

    @pytest.mark.asyncio
    async def test_multiple_results_formatted(self):
        """Multiple results are all included and separated."""
        data = brave_results(
            ("Page A", "https://a.com", "Snippet A"),
            ("Page B", "https://b.com", "Snippet B"),
        )
        mock_resp = mock_httpx_response(json_data=data)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get.return_value = mock_resp
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await web_search("test", api_key="fake-key")

        assert "Page A" in result.content
        assert "Page B" in result.content
        assert "Snippet A" in result.content
        assert "Snippet B" in result.content

    @pytest.mark.asyncio
    async def test_html_stripped_from_snippets(self):
        """HTML tags in Brave snippets are cleaned."""
        data = brave_results(
            ("Page", "https://x.com", "This is <strong>bold</strong> text"),
        )
        mock_resp = mock_httpx_response(json_data=data)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get.return_value = mock_resp
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await web_search("test", api_key="fake-key")

        assert "<strong>" not in result.content
        assert "bold" in result.content


def _make_fetch_client(stream_ctx):
    """Wrap a streaming context in a patched AsyncClient."""
    client = MagicMock()
    # stream() must be a plain callable (not a coroutine) that returns the ctx manager
    client.stream = MagicMock(return_value=stream_ctx)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestWebFetch:

    @pytest.mark.asyncio
    async def test_returns_readable_content(self):
        """HTML is stripped to readable text."""
        html = "<html><body><p>Hello world</p></body></html>"

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(
                mock_streaming_response(content=html.encode())
            )
            result = await web_fetch("https://test.com")

        assert result.is_error is False
        assert "Hello world" in result.content
        assert "<p>" not in result.content

    @pytest.mark.asyncio
    async def test_strips_scripts_and_nav(self):
        """Script, style, and nav tags are removed with content."""
        html = "<html><body><script>evil()</script><nav>menu</nav><p>Content</p></body></html>"

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(
                mock_streaming_response(content=html.encode())
            )
            result = await web_fetch("https://test.com")

        assert "evil" not in result.content
        assert "menu" not in result.content
        assert "Content" in result.content

    @pytest.mark.asyncio
    async def test_max_chars_truncates(self):
        """Content longer than max_chars is truncated with marker."""
        long_text = "A" * 1000

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(
                mock_streaming_response(content=long_text.encode())
            )
            result = await web_fetch("https://test.com", max_chars=200)

        assert result.is_error is False
        # kdsn.291: head+tail body preserved, steering marker carries total + midpoint
        assert result.content.startswith("A" * 100)
        assert result.content.endswith("A" * 100)
        assert (
            "[truncated: 800 chars removed — text is 1000 chars total; "
            "re-fetch with offset=500 to read a specific region]"
            in result.content
        )

    @pytest.mark.asyncio
    async def test_max_chars_none_returns_full(self):
        """No max_chars → full content returned."""
        text = "X" * 500

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(
                mock_streaming_response(content=text.encode())
            )
            result = await web_fetch("https://test.com")

        assert len(result.content) == 500

    @pytest.mark.asyncio
    async def test_fetch_error_returns_tool_error(self):
        """HTTP error returns is_error=True."""
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(
                mock_streaming_response(status_code=404)
            )
            result = await web_fetch("https://test.com/nope")

        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_plain_text_passed_through(self):
        """Non-HTML content passes through unchanged."""
        text = "Just plain text, no HTML here."

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(
                mock_streaming_response(content=text.encode())
            )
            result = await web_fetch("https://test.com/file.txt")

        assert result.content == text

    @pytest.mark.asyncio
    async def test_web_fetch_caps_large_response(self):
        """Streaming stops once MAX_RESPONSE_BYTES have been received; no OOM."""
        # Deliver the oversized payload in two chunks to exercise the mid-stream cap
        big_first_chunk = b"X" * MAX_RESPONSE_BYTES
        overflow_chunk = b"Y" * 512


        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()

        async def _aiter_bytes():
            yield big_first_chunk
            yield overflow_chunk  # should never be appended

        resp.aiter_bytes = _aiter_bytes

        stream_ctx = AsyncMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=resp)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(stream_ctx)
            result = await web_fetch("https://test.com/huge")

        assert result.is_error is False
        # kdsn.291: full huge text carries a head-anchored navigation note;
        # the 2MB cap applies to the fetched body, not the annotation.
        assert "Y" not in result.content
        assert result.content.startswith("[web_fetch note: full text is")
        body = result.content.split("\n", 1)[1]
        assert len(body.encode("utf-8")) == MAX_RESPONSE_BYTES

    @pytest.mark.asyncio
    async def test_web_fetch_partial_last_chunk_trimmed(self):
        """When the cap is hit mid-chunk, only the bytes up to the cap are kept."""
        # First chunk: 1MB, second chunk: 1.5MB (crosses the 2MB boundary)
        one_mb = 1024 * 1024
        chunk1 = b"A" * one_mb
        chunk2 = b"B" * (one_mb + 512 * 1024)  # 1.5 MB — pushes total over 2MB

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()

        async def _aiter_bytes():
            yield chunk1
            yield chunk2

        resp.aiter_bytes = _aiter_bytes

        stream_ctx = AsyncMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=resp)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(stream_ctx)
            result = await web_fetch("https://test.com/partial")

        assert result.is_error is False
        # kdsn.291: body is exactly the 2MB cap (navigation note rides ahead)
        body = result.content.split("\n", 1)[1]
        assert len(body.encode("utf-8")) == MAX_RESPONSE_BYTES
        # First chunk's characters must all be present
        assert "A" in body

    @pytest.mark.asyncio
    async def test_web_fetch_normal_response_unchanged(self):
        """Responses well under MAX_RESPONSE_BYTES pass through without truncation."""
        text = "Normal sized response content."

        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(
                mock_streaming_response(content=text.encode())
            )
            result = await web_fetch("https://test.com/small")

        assert result.is_error is False
        assert result.content == text


class TestHtmlStripping:
    """Unit tests for _strip_html_tags — no mocking needed."""

    def test_basic_tags(self):
        assert _strip_html_tags("<p>Hello</p>") == "Hello"

    def test_script_removed(self):
        assert "alert" not in _strip_html_tags("<script>alert('x')</script>Content")

    def test_style_removed(self):
        assert "color" not in _strip_html_tags("<style>.x{color:red}</style>Content")

    def test_entities_decoded(self):
        assert _strip_html_tags("&amp; &lt; &gt;") == "& < >"

    def test_whitespace_collapsed(self):
        result = _strip_html_tags("  lots   of    spaces  ")
        assert result == "lots of spaces"


# ---------------------------------------------------------------------------
# kdsn.291: char-windowing offset (window mode) + legacy-mode steering marker
# Spec: memory/projects/openalph/specs/kdsn.291-web-fetch-offset-windowing-spec.md
# ---------------------------------------------------------------------------


class TestWebFetchOffset:
    """kdsn.291: offset (0-based char position) turns max_chars from a head/tail
    cap into a window size. No offset -> legacy behavior byte-identical, with
    the legacy marker now carrying total size + midpoint steering."""

    @staticmethod
    async def _fetch(text, **kwargs):
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(
                mock_streaming_response(content=text.encode())
            )
            result = await web_fetch("https://test.com", **kwargs)
        return result, MockClient

    @pytest.mark.asyncio
    async def test_window_exact(self):
        """T1: offset=400 max_chars=200 on 1000 chars -> text[400:600] + nav marker."""
        text = "z" * 1000
        result, _ = await self._fetch(text, offset=400, max_chars=200)
        assert result.is_error is False
        assert result.content.startswith(
            "[web_fetch window: chars 400-599 of 1000 | 400 before | 400 after — continue with offset=600]"
        )
        assert result.content.endswith("z" * 200)

    @pytest.mark.asyncio
    async def test_window_clamps_to_end(self):
        """T2: window reaching the end clamps; 'end of text', no continue clause."""
        text = "z" * 1000
        result, _ = await self._fetch(text, offset=900, max_chars=200)
        assert result.is_error is False
        assert result.content.startswith(
            "[web_fetch window: chars 900-999 of 1000 | 900 before | end of text]"
        )
        assert result.content.endswith("z" * 100)

    @pytest.mark.asyncio
    async def test_window_default_max_chars(self):
        """T3: offset without max_chars -> 50k default window (clamped here)."""
        text = "z" * 1000
        result, _ = await self._fetch(text, offset=400)
        assert result.is_error is False
        assert result.content.startswith(
            "[web_fetch window: chars 400-999 of 1000 | 400 before | end of text]"
        )
        assert result.content.endswith("z" * 600)

    @pytest.mark.asyncio
    async def test_offset_zero_is_window_not_truncation(self):
        """T4: explicit offset=0 is window mode (not head/tail); 'before' omitted at 0."""
        text = "z" * 1000
        result, _ = await self._fetch(text, offset=0, max_chars=200)
        assert result.is_error is False
        assert result.content.startswith(
            "[web_fetch window: chars 0-199 of 1000 | 800 after — continue with offset=200]"
        )
        assert result.content.endswith("z" * 200)
        assert "before" not in result.content

    @pytest.mark.asyncio
    async def test_offset_beyond_end_errors_with_total(self):
        """T5: offset past end of text -> is_error naming the total length."""
        text = "z" * 1000
        result, _ = await self._fetch(text, offset=5000, max_chars=200)
        assert result.is_error is True
        assert "5000" in result.content
        assert "1000" in result.content

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", [-1, "400", True])
    async def test_offset_invalid_types_error_without_network(self, bad):
        """T6: invalid offset -> is_error BEFORE any network request."""
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            result = await web_fetch("https://test.com", offset=bad, max_chars=200)
        assert result.is_error is True
        assert "offset must be a non-negative integer" in result.content
        MockClient.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_offset_legacy_full(self):
        """T7: no offset, no max_chars -> full text, no marker (regression)."""
        text = "z" * 500
        result, _ = await self._fetch(text)
        assert result.is_error is False
        assert result.content == text

    @pytest.mark.asyncio
    async def test_no_offset_max_chars_legacy_marker_steers(self):
        """T8: legacy head/tail truncation marker carries total + midpoint steer."""
        text = "G" * 1000
        result, _ = await self._fetch(text, max_chars=200)
        assert result.is_error is False
        assert result.content.startswith("G" * 100)
        assert result.content.endswith("G" * 100)
        assert (
            "[truncated: 800 chars removed — text is 1000 chars total; "
            "re-fetch with offset=500 to read a specific region]"
            in result.content
        )

    @pytest.mark.asyncio
    async def test_dispatch_execute_tool_windows(self):
        """T9: offset flows through the real dispatch seam (schema wiring)."""
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
        text = "z" * 1000
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(
                mock_streaming_response(content=text.encode())
            )
            result = await execute_tool(
                name="web_fetch",
                input={"url": "https://test.com", "offset": 400, "max_chars": 200},
                tool_config={},
                agent_config=cfg,
            )
        assert result.is_error is False
        assert result.content.startswith("[web_fetch window: chars 400-599 of 1000")

    @pytest.mark.asyncio
    async def test_window_into_2mb_capped_text(self):
        """T10: windowing works on a 2MB-capped extraction."""
        text = "X" * MAX_RESPONSE_BYTES
        result, _ = await self._fetch(text, offset=1_500_000, max_chars=1000)
        assert result.is_error is False
        total = MAX_RESPONSE_BYTES
        assert (
            f"chars 1500000-1500999 of {total} | 1500000 before | "
            f"{total - 1500999 - 1} after — continue with offset=1501000"
            in result.content
        )
        assert result.content.endswith("X" * 1000)


class TestWebFetchPipeline:
    """kdsn.291 review fixes: windows must survive the framework truncation
    pipeline (truncate_result), and huge uncapped full text must carry a
    head-anchored navigation note that survives the head/tail cut."""

    @staticmethod
    async def _fetch(text, **kwargs):
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(
                mock_streaming_response(content=text.encode())
            )
            result = await web_fetch("https://test.com", **kwargs)
        return result

    @pytest.mark.asyncio
    async def test_default_window_survives_framework_truncation(self):
        """P1: marker + 49k default window < 50k limit -> byte-identical
        through the real truncate_result pipeline (no double-truncation)."""
        from openalph.tools import truncate_result

        result = await self._fetch("z" * 200_000, offset=0)
        assert result.is_error is False
        assert result.content.startswith("[web_fetch window: chars 0-48999 of 200000")
        truncated = truncate_result(result.content, 50_000)
        assert truncated == result.content  # intact — no framework marker

    @pytest.mark.asyncio
    async def test_window_larger_than_lower_limit_shaves_visibly(self):
        """P2: with a configured lower truncation_limit the window IS shaved —
        the framework marker appears (visible + self-correcting, never silent)."""
        from openalph.tools import truncate_result

        result = await self._fetch("z" * 200_000, offset=0)
        truncated = truncate_result(result.content, 20_000)
        assert "[truncated:" in truncated
        assert truncated != result.content

    @pytest.mark.asyncio
    async def test_huge_full_text_note_survives_truncation(self):
        """P3: uncapped huge full text -> head-anchored note with total size
        + midpoint offset; the note survives the framework head/tail cut."""
        from openalph.tools import truncate_result

        result = await self._fetch("z" * 200_000)
        assert result.is_error is False
        assert result.content.startswith("[web_fetch note: full text is 200000 chars")
        assert "offset=100000 for the middle" in result.content
        truncated = truncate_result(result.content, 50_000)
        assert truncated.startswith("[web_fetch note: full text is 200000 chars")
        assert "offset=100000" in truncated

    @pytest.mark.asyncio
    async def test_note_absent_when_legacy_cap_applies(self):
        """P4: max_chars head/tail truncation already carries navigation —
        the full-text note must not also appear (no double steering)."""
        result = await self._fetch("z" * 200_000, max_chars=2000)
        assert result.is_error is False
        assert "[web_fetch note:" not in result.content
        assert "re-fetch with offset=100000 to read a specific region" in result.content


# ---------------------------------------------------------------------------
# kdsn.293: result-budget-aware windowing for web_fetch and web_fetch_js
# ---------------------------------------------------------------------------


_WINDOW_HEADER_RE = re.compile(
    r"^\[(?P<tool>web_fetch(?:_js)?) window: chars "
    r"(?P<start>\d+)-(?P<end>\d+) of (?P<total>\d+)"
    r"(?P<clauses>[^\n]*)\]\n"
)
_JS_NOTE_PREFIX = "\n\n[web_fetch note:"


def _indexed_text(length):
    """Deterministic, non-hex filler whose position can be checked exactly."""
    # Each 20-char block starts with its absolute position, so equal-length
    # slices at different offsets cannot alias as they could with a pure cycle.
    blocks = []
    for start in range(0, length, 20):
        label = f"p{start:012d}:"
        blocks.append(label + ("w" * (20 - len(label))))
    return "".join(blocks)[:length]


def _split_window_output(content, *, js_note=False):
    """Return parsed header fields and the exact delivered source-text body."""
    match = _WINDOW_HEADER_RE.match(content)
    assert match is not None, f"missing/invalid window header: {content[:200]!r}"
    body = content[match.end():]
    note = ""
    if js_note:
        body, separator, note_tail = body.partition(_JS_NOTE_PREFIX)
        assert separator == _JS_NOTE_PREFIX
        note = separator + note_tail
    return (
        int(match.group("start")),
        int(match.group("end")),
        int(match.group("total")),
        match.group("clauses"),
        body,
        note,
    )


def _agent_config(truncation_limit):
    """Build agent_config exactly as the existing T9/J6 dispatch tests do."""
    from openalph.config import AgentConfig, ProviderConfig

    return AgentConfig(
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
        truncation_limit=truncation_limit,
    )


class TestWebFetchWindowBudget:
    """W1-W11: web_fetch window headers must describe exactly what arrives."""

    LENGTH = 141_529

    @staticmethod
    async def _fetch(text, **kwargs):
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(
                mock_streaming_response(content=text.encode())
            )
            result = await web_fetch("https://test.com", **kwargs)
        return result, MockClient

    @pytest.mark.asyncio
    async def test_window_at_budget_no_second_truncation(self):
        """W1: a requested 50k window is contiguous and pipeline-safe at 50k."""
        text = _indexed_text(self.LENGTH)
        result, _ = await self._fetch(
            text, offset=0, max_chars=50_000, result_limit=50_000
        )
        assert result.is_error is False
        assert "[truncated:" not in result.content
        start, end, total, clauses, body, _ = _split_window_output(result.content)
        assert (start, total) == (0, self.LENGTH)
        assert body == text[start:end + 1]
        assert f"continue with offset={end + 1}" in clauses
        assert truncate_result(result.content, 50_000) == result.content

    @pytest.mark.asyncio
    async def test_pipeline_noop(self):
        """W2: representative windows are byte-identical through truncation."""
        text = _indexed_text(self.LENGTH)
        for offset in (0, self.LENGTH // 2, self.LENGTH - 20_000):
            result, _ = await self._fetch(
                text, offset=offset, max_chars=50_000, result_limit=50_000
            )
            assert result.is_error is False
            assert truncate_result(result.content, 50_000) == result.content

    @pytest.mark.asyncio
    async def test_tiling_header_is_truth(self):
        """W3: the advertised A-B range is the exact contiguous delivered body."""
        text = _indexed_text(self.LENGTH)
        for offset in (0, self.LENGTH // 2, self.LENGTH - 20_000):
            result, _ = await self._fetch(
                text, offset=offset, max_chars=50_000, result_limit=50_000
            )
            assert result.is_error is False
            start, end, total, clauses, body, _ = _split_window_output(result.content)
            assert (start, total) == (offset, self.LENGTH)
            assert body == text[start:end + 1]
            assert len(body) == end - start + 1
            assert ("end of text" in clauses) is (end + 1 == total)
            assert ("continue with offset=" in clauses) is (end + 1 < total)

    @pytest.mark.asyncio
    async def test_window_trimmed_to_fit_lower_limit(self):
        """W4: a lower result budget trims visibly without cutting the body."""
        text = _indexed_text(self.LENGTH)
        result, _ = await self._fetch(
            text, offset=0, max_chars=50_000, result_limit=40_000
        )
        assert result.is_error is False
        start, end, total, clauses, body, _ = _split_window_output(result.content)
        assert "window trimmed to fit the 40000-char result budget" in clauses
        assert (start, total) == (0, self.LENGTH)
        assert body == text[start:end + 1]
        assert end + 1 < 50_000
        assert f"continue with offset={end + 1}" in clauses
        assert len(result.content) <= 40_000

    @pytest.mark.asyncio
    async def test_js_note_counts_against_budget(self):
        """W5: web_fetch's JS-shell nudge consumes the same result budget."""
        # The root marker forces the nudge while the text node remains large.
        visible = _indexed_text(60_000)
        html = f'<html><body><div id="root">{visible}</div></body></html>'
        result, _ = await self._fetch(
            html,
            offset=0,
            max_chars=50_000,
            result_limit=50_000,
            tool_config={"js_fallback_tool": "web_fetch_js"},
        )
        assert result.is_error is False
        start, end, total, _, body, note = _split_window_output(
            result.content, js_note=True
        )
        assert "Retry with web_fetch_js" in note
        assert (start, total) == (0, len(visible))
        assert body == visible[start:end + 1]
        assert len(result.content) <= 50_000
        assert truncate_result(result.content, 50_000) == result.content

    @pytest.mark.asyncio
    async def test_floor_is_error(self):
        """W6: a budget below header + 1,000 chars fails loud post-fetch."""
        tool_config = {"sentinel": "unchanged"}
        before = dict(tool_config)
        result, MockClient = await self._fetch(
            _indexed_text(self.LENGTH),
            offset=0,
            max_chars=50_000,
            result_limit=1_000,
            tool_config=tool_config,
        )
        assert result.is_error is True
        assert "1000" in result.content.replace(",", "")
        assert "window" in result.content.lower()
        assert tool_config == before
        MockClient.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_window_byte_identical(self):
        """W7 guard: default 49k direct window remains current-code exact."""
        text = _indexed_text(200_000)
        kwargs = {"offset": 0}
        if "result_limit" in inspect.signature(web_fetch).parameters:
            kwargs["result_limit"] = 50_000
        result, _ = await self._fetch(text, **kwargs)
        expected = (
            "[web_fetch window: chars 0-48999 of 200000 | 151000 after — "
            "continue with offset=49000]\n" + text[:49_000]
        )
        assert result.is_error is False
        assert result.content == expected

    @pytest.mark.asyncio
    async def test_none_limit_legacy_parity(self):
        """W8 guard: result_limit=None preserves the oversize legacy window."""
        text = _indexed_text(self.LENGTH)
        kwargs = {"offset": 0, "max_chars": 50_000}
        if "result_limit" in inspect.signature(web_fetch).parameters:
            kwargs["result_limit"] = None
        result, _ = await self._fetch(text, **kwargs)
        expected = (
            "[web_fetch window: chars 0-49999 of 141529 | 91529 after — "
            "continue with offset=50000]\n" + text[:50_000]
        )
        assert result.is_error is False
        assert result.content == expected
        assert len(result.content) > 50_000

    @pytest.mark.asyncio
    async def test_end_of_text_boundary(self):
        """W9: the 'end of text' / 'continue' / 'trimmed' boundary, all three
        sides, each pinned exactly (drops the old vacuous trimmed-to-EOF case).

        (a) natural-EOF untrimmed window -> 'end of text' present, no
            'continue with', no trimmed clause, body == full text.
        (b) one char before EOF (untrimmed) -> 'continue with offset' present,
            no 'end of text', body == text minus the last char.
        (c) a trimmed window -> trimmed clause AND 'continue with' present,
            NEVER 'end of text' (a trim always ends below total).
        """
        text = _indexed_text(self.LENGTH)

        # (a) natural-EOF: window reaches exactly the end, untrimmed.
        # Page is 45k (well outside the 1k headroom band of the 50k cap) so
        # the explicit window still short-circuits untrimmed.
        eof_text = _indexed_text(45_000)
        result, _ = await self._fetch(
            eof_text, offset=0, max_chars=45_000, result_limit=50_000
        )
        assert result.is_error is False
        start, end, total, clauses, body, _ = _split_window_output(result.content)
        assert (start, total) == (0, len(eof_text))
        assert end + 1 == total
        assert body == eof_text[start:end + 1]
        assert "end of text" in clauses
        assert "continue with offset=" not in clauses
        assert "window trimmed" not in clauses

        # (b) one char before EOF: window ends one short of the end, untrimmed.
        result, _ = await self._fetch(
            eof_text, offset=0, max_chars=45_000 - 1, result_limit=50_000
        )
        assert result.is_error is False
        start, end, total, clauses, body, _ = _split_window_output(result.content)
        assert (start, total) == (0, len(eof_text))
        assert end + 1 == total - 1
        assert body == eof_text[start:end + 1]
        assert "continue with offset=" in clauses
        assert "end of text" not in clauses
        assert "window trimmed" not in clauses

        # (c) a trimmed window: a lower result budget forces a trim, so the
        # trimmed clause and the continue clause are both present and 'end of
        # text' is never claimed (a trim always ends below total).
        result, _ = await self._fetch(
            text, offset=0, max_chars=50_000, result_limit=40_000
        )
        assert result.is_error is False
        start, end, total, clauses, body, _ = _split_window_output(result.content)
        assert (start, total) == (0, self.LENGTH)
        assert end + 1 < total
        assert "window trimmed to fit the 40000-char result budget" in clauses
        assert "continue with offset=" in clauses
        assert "end of text" not in clauses
        assert body == text[start:end + 1]

    @pytest.mark.asyncio
    async def test_dispatch_passes_result_limit(self):
        """W10: real dispatch forwards agent_config.truncation_limit=40000."""
        text = _indexed_text(self.LENGTH)
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_fetch_client(
                mock_streaming_response(content=text.encode())
            )
            result = await execute_tool(
                name="web_fetch",
                input={
                    "url": "https://test.com", "offset": 0, "max_chars": 50_000,
                },
                tool_config={},
                agent_config=_agent_config(40_000),
            )
        assert result.is_error is False
        start, end, total, clauses, body, _ = _split_window_output(result.content)
        assert "window trimmed to fit the 40000-char result budget" in clauses
        assert (start, total) == (0, self.LENGTH)
        assert body == text[start:end + 1]
        assert len(result.content) <= 40_000
        assert truncate_result(result.content, 40_000) == result.content

    @pytest.mark.asyncio
    async def test_digit_width_oscillation_converges(self):
        """W11: sizing converges when effective end crosses 99999/100000."""
        text = _indexed_text(100_100)
        result, _ = await self._fetch(
            text, offset=50_000, max_chars=50_100, result_limit=50_000
        )
        assert result.is_error is False
        start, end, total, clauses, body, _ = _split_window_output(result.content)
        assert "window trimmed to fit the 50000-char result budget" in clauses
        assert (start, total) == (50_000, len(text))
        assert end + 1 < 100_000 < 50_000 + 50_100
        assert body == text[start:end + 1]
        assert f"continue with offset={end + 1}" in clauses
        assert len(result.content) <= 50_000
        assert truncate_result(result.content, 50_000) == result.content

    @pytest.mark.asyncio
    async def test_maximal_end_audit_repro_unit(self):
        """W12a (FIX-2 MED): the maximal feasible end at the audit's exact
        digit-boundary repro, called directly on _budgeted_window_text (no
        mock). The pre-shrink floor once rejected a window that fits once a
        shorter header is realized; the enumeration now lands the maximal end.

        text = 'x'*10100; (text, 8948, 1052, 'web_fetch', 1151) -> not None;
        body exactly 1000 chars; header carries 'chars 8948-9947',
        '1151-char result budget', 'continue with offset=9948';
        len(output) == 1151 EXACTLY. The adjacent limit 1140 -> None.
        """
        text = "x" * 10100
        output = _budgeted_window_text(text, 8948, 1052, "web_fetch", 1151)
        assert output is not None
        assert len(output) == 1151
        start, end, total, clauses, body, _ = _split_window_output(output)
        assert (start, total) == (8948, 10100)
        assert "chars 8948-9947" in output
        assert "1151-char result budget" in clauses
        assert "continue with offset=9948" in clauses
        assert len(body) == 1000
        assert body == text[8948:9948]
        # Adjacent no-fit: one char less budget cannot fit the 1k window.
        assert _budgeted_window_text(text, 8948, 1052, "web_fetch", 1140) is None

    @pytest.mark.asyncio
    async def test_maximal_end_audit_repro_tool(self):
        """W12b: the same maximal-end repro through web_fetch with the file's
        mocked-httpx convention -> header truth, body == fixture[8948:9948],
        total output <= 1151."""
        text = "x" * 10100
        result, _ = await self._fetch(
            text, offset=8948, max_chars=1052, result_limit=1151
        )
        assert result.is_error is False
        start, end, total, clauses, body, _ = _split_window_output(result.content)
        assert (start, total) == (8948, 10100)
        assert "chars 8948-9947" in result.content
        assert "1151-char result budget" in clauses
        assert "continue with offset=9948" in clauses
        assert body == text[8948:9948]
        assert len(result.content) <= 1151

    @pytest.mark.asyncio
    async def test_redaction_headroom_pipeline_noop(self):
        """W13 (FIX-1 HIGH): a window sized with REDACTION_HEADROOM reserved
        stays a byte-identical no-op through the real
        handler -> redact_credentials -> truncate_result pipeline when the
        page carries credential-shaped tokens that EXPAND under redaction.

        The expanding pattern is the anthropic_api_key shape (``sk-ant-`` +
        8 chars = 15-char match -> 18-char ``[REDACTED:api_key]``, +3 each),
        verified empirically. N is picked so total expansion (~N*3) is
        ~50-80% of min(1000, 50000//10) = 1000 (i.e. ~500-800 chars).
        """
        from openalph.tools.security import redact_credentials

        # 200 tokens * +3 = +600 chars expansion = 60% of the 1000-char headroom.
        token = "sk-ant-" + "a" * 8  # 15-char match -> 18-char replacement
        unit = token + "!" * 80  # '!' is NOT in the token char class, so it
        #                          does not extend the match (verified: the
        #                          match stays 15 chars, not the whole unit).
        n_tokens = 200
        # +35k inert filler so the 50k window HITS the budget (untrimmed
        # header+50k > 50k) and the REDACTION_HEADROOM sizing is exercised.
        page = unit * n_tokens + "z" * 35_000  # 54000 total

        result, _ = await self._fetch(
            page, offset=0, max_chars=50_000, result_limit=50_000
        )
        assert result.is_error is False
        redacted, events = redact_credentials(result.content)
        # Redaction really happened (expansion occurred).
        assert redacted != result.content
        assert len(events) == n_tokens
        # I-B with headroom: the expanded output still fits the limit, so the
        # framework truncate_result is a byte-identical no-op.
        assert len(redacted) <= 50_000
        assert truncate_result(redacted, 50_000) == redacted

    @pytest.mark.asyncio
    async def test_redaction_overflow_marked(self):
        """W14 (FIX-1 residual): when the page carries enough expanding
        credential tokens that the total expansion EXCEEDS the reserved
        REDACTION_HEADROOM (~1000 chars), the post-redaction output
        overflows the limit and the framework truncate_result marks it with
        an accurate '[truncated:' marker — marked, never silent.

        ~500 tokens * +3 = ~1500 chars expansion > 1000-char headroom.
        """
        from openalph.tools.security import redact_credentials

        token = "sk-ant-" + "a" * 8  # 15-char match -> 18-char replacement
        unit = token + "!" * 80
        n_tokens = 480  # 480 * +3 = +1440 chars > 1000-char headroom;
        # 480 * 95 = 45600-char token region stays INSIDE the
        # headroom-reserved window so all tokens are delivered.
        # +10k inert filler so the 50k window hits the budget and the
        # overflow is due to expansion beating the reserved headroom.
        page = unit * n_tokens + "z" * 10_000  # 55600 total

        result, _ = await self._fetch(
            page, offset=0, max_chars=50_000, result_limit=50_000
        )
        assert result.is_error is False
        redacted, events = redact_credentials(result.content)
        assert redacted != result.content
        assert len(events) == n_tokens
        # Expansion exceeds the headroom -> overflow.
        assert len(redacted) > 50_000
        final = truncate_result(redacted, 50_000)
        # Documented residual: the overflow is MARKED, never silent.
        assert "[truncated:" in final
        assert len(final) == 50_000

    @pytest.mark.asyncio
    async def test_shortcircuit_band_keeps_headroom(self):
        """W15 (audit2 MED): an explicit window that fits UNTRIMMED but lands
        inside the headroom band (clamped short page, ~49.6k at the 50k cap)
        must still reserve headroom, else post-redaction expansion
        middle-cuts it. The default window (no max_chars) stays exempt —
        W7 pins its byte identity."""
        from openalph.tools.security import redact_credentials

        token = "sk-ant-" + "a" * 8  # +3 under redaction
        unit = token + "!" * 80
        n_tokens = 200  # +600 total, inside the 1000-char headroom
        page = unit * n_tokens + "z" * 30_500  # 49500 total

        result, _ = await self._fetch(
            page, offset=0, max_chars=50_000, result_limit=50_000
        )
        assert result.is_error is False
        assert "window trimmed" in result.content
        redacted, events = redact_credentials(result.content)
        assert len(events) == n_tokens
        assert len(redacted) <= 50_000
        assert truncate_result(redacted, 50_000) == redacted

    @pytest.mark.asyncio
    async def test_default_window_trimmed_path_keeps_headroom(self):
        """W16: the legacy exemption covers only the byte-identical
        SHORTCUT. When the default 49k window must be trimmed (lower custom
        result_limit), the trimmed path still reserves the full redaction
        headroom -> no-op after real redact + truncate."""
        from openalph.tools.security import redact_credentials

        token = "sk-ant-" + "a" * 8  # +3 under redaction
        unit = token + "!" * 80
        n_tokens = 200  # +600 total, inside the 1000-char headroom
        page = unit * n_tokens + "z" * 35_000  # 54000 total

        result, _ = await self._fetch(
            page, offset=0, result_limit=45_000
        )
        assert result.is_error is False
        assert "window trimmed" in result.content
        redacted, events = redact_credentials(result.content)
        assert len(events) == n_tokens
        assert len(redacted) <= 45_000
        assert truncate_result(redacted, 45_000) == redacted

    @pytest.mark.asyncio
    async def test_explicit_zero_max_chars_keeps_headroom(self):
        """W17 (audit3 N-1): explicit max_chars=0 (schema-valid; dispatch
        does not range-validate) must NOT receive the omission-only
        shortcut exemption — the default window it selects still reserves
        headroom, so the real redact -> truncate pipeline stays a no-op."""
        from openalph.tools.security import redact_credentials

        token = "sk-ant-" + "a" * 8  # +3 under redaction
        unit = token + "!" * 80
        n_tokens = 310  # +930, inside the 1000-char headroom
        page = unit * n_tokens + "z" * 30_000  # 59450 total

        result, _ = await self._fetch(
            page, offset=0, max_chars=0, result_limit=50_000
        )
        assert result.is_error is False
        assert "window trimmed" in result.content
        redacted, events = redact_credentials(result.content)
        assert len(events) == n_tokens
        assert len(redacted) <= 50_000
        assert truncate_result(redacted, 50_000) == redacted

    @pytest.mark.asyncio
    async def test_exact_floor_window_fits_without_headroom(self):
        """W18 (audit3 N-2): an EXACT 1,000-char window whose header+body
        fits the limit exactly is returned (floor wins over the shortcut
        headroom clamp); one char less budget is an accurate error."""
        text = "x" * 2000
        result, _ = await self._fetch(
            text, offset=0, max_chars=1000, result_limit=1081
        )
        assert result.is_error is False
        assert "chars 0-999 of 2000" in result.content
        assert len(result.content) == 1081
        result, _ = await self._fetch(
            text, offset=0, max_chars=1000, result_limit=1080
        )
        assert result.is_error is True
        assert "1080" in result.content


class TestWebFetchJsWindowBudget:
    """J8-J11: the same budget contract for Tabstack markdown windows."""

    @staticmethod
    async def _fetch(text, **kwargs):
        client = make_post_client(mock_post_response(json_data={"content": text}))
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com",
                api_key="fake-key",
                base_url="https://api.tabstack.ai/v1",
                **kwargs,
            )
        return result, client

    @pytest.mark.asyncio
    async def test_markdown_window_at_budget(self):
        """J8: a 50k markdown request is contiguous and pipeline-safe."""
        text = _indexed_text(141_529)
        result, _ = await self._fetch(
            text, offset=0, max_chars=50_000, result_limit=50_000
        )
        assert result.is_error is False
        assert "[truncated:" not in result.content
        start, end, total, clauses, body, _ = _split_window_output(result.content)
        assert (start, total) == (0, len(text))
        assert body == text[start:end + 1]
        assert f"continue with offset={end + 1}" in clauses
        assert truncate_result(result.content, 50_000) == result.content

    @pytest.mark.asyncio
    async def test_schema_offset_still_errors(self):
        """J9 guard: schema+offset remains a pre-request markdown-only error."""
        client = make_post_client(mock_post_response(json_data={"content": "unused"}))
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await web_fetch_js(
                url="https://example.com",
                api_key="fake-key",
                base_url="https://api.tabstack.ai/v1",
                schema={"type": "object", "properties": {}},
                offset=10,
            )
        assert result.is_error is True
        assert "markdown-mode only" in result.content
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_markdown_floor_is_error(self):
        """J10: too-small markdown result budget fails loud after the POST."""
        text = _indexed_text(141_529)
        result, client = await self._fetch(
            text, offset=0, max_chars=50_000, result_limit=1_000
        )
        assert result.is_error is True
        assert "1000" in result.content.replace(",", "")
        assert "window" in result.content.lower()
        client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dispatch_passes_result_limit_js(self):
        """J11: real dispatch forwards the 40k agent result budget to JS."""
        text = _indexed_text(141_529)
        client = make_post_client(mock_post_response(json_data={"content": text}))
        with patch("openalph.tools.web.httpx.AsyncClient") as MockClient:
            MockClient.return_value = client
            result = await execute_tool(
                name="web_fetch_js",
                input={
                    "url": "https://example.com", "offset": 0,
                    "max_chars": 50_000,
                },
                tool_config={
                    "api_key": "fake-key",
                    "base_url": "https://api.tabstack.ai/v1",
                },
                agent_config=_agent_config(40_000),
            )
        assert result.is_error is False
        start, end, total, clauses, body, _ = _split_window_output(result.content)
        assert "window trimmed to fit the 40000-char result budget" in clauses
        assert (start, total) == (0, len(text))
        assert body == text[start:end + 1]
        assert len(result.content) <= 40_000
        assert truncate_result(result.content, 40_000) == result.content

    @pytest.mark.asyncio
    async def test_maximal_end_audit_repro_js(self):
        """J12 (FIX-2 MED, web_fetch_js parity): the codexsol report's JS
        variant — limit 1154 (confirmed; web_fetch_js has no js_note suffix) —
        same maximal-end assertions as W12a. Called directly on
        _budgeted_window_text with tool='web_fetch_js'.

        (text, 8948, 1052, 'web_fetch_js', 1154) -> not None; body exactly
        1000 chars; header 'chars 8948-9947', '1154-char result budget',
        'continue with offset=9948'; len(output) == 1154 EXACTLY. The
        adjacent limit 1143 -> None.
        """
        text = "x" * 10100
        output = _budgeted_window_text(text, 8948, 1052, "web_fetch_js", 1154)
        assert output is not None
        assert len(output) == 1154
        start, end, total, clauses, body, _ = _split_window_output(output)
        assert (start, total) == (8948, 10100)
        assert "chars 8948-9947" in output
        assert "1154-char result budget" in clauses
        assert "continue with offset=9948" in clauses
        assert len(body) == 1000
        assert body == text[8948:9948]
        # Adjacent no-fit: one char less budget cannot fit the 1k window.
        assert _budgeted_window_text(text, 8948, 1052, "web_fetch_js", 1143) is None

    @pytest.mark.asyncio
    async def test_explicit_zero_max_chars_js(self):
        """J13 (audit3 N-1 parity): web_fetch_js, explicit max_chars=0
        still reserves shortcut headroom (same guard as W17)."""
        from openalph.tools.security import redact_credentials

        token = "sk-ant-" + "a" * 8  # +3 under redaction
        unit = token + "!" * 80
        n_tokens = 310  # +930, inside the 1000-char headroom
        page = unit * n_tokens + "z" * 30_000  # 59450 total

        result, _ = await self._fetch(
            page, offset=0, max_chars=0, result_limit=50_000
        )
        assert result.is_error is False
        assert "window trimmed" in result.content
        redacted, events = redact_credentials(result.content)
        assert len(events) == n_tokens
        assert len(redacted) <= 50_000
        assert truncate_result(redacted, 50_000) == redacted

    @pytest.mark.asyncio
    async def test_exact_floor_window_js(self):
        """J14 (audit3 N-2 parity): web_fetch_js header is 3 chars longer,
        so the exact-fit limit is 1084 (84-char header + 1000 body)."""
        text = "x" * 2000
        result, _ = await self._fetch(
            text, offset=0, max_chars=1000, result_limit=1084
        )
        assert result.is_error is False
        assert len(result.content) == 1084
