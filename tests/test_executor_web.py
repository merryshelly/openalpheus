"""Tests for web tools (search + fetch).

Tests mock httpx.AsyncClient to avoid real network calls while testing
the actual formatting, parsing, error handling, and truncation logic.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from openalph.tools.web import web_search, web_fetch, _strip_html_tags, MAX_RESPONSE_BYTES
from openalph.tools import ToolResult


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

        assert "[truncated" in result.content
        # Result should be roughly max_chars + marker length
        assert len(result.content) < 300

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

        import httpx as _httpx

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
        # Must be capped — overflow bytes must not appear
        assert "Y" not in result.content
        assert len(result.content.encode("utf-8")) <= MAX_RESPONSE_BYTES

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
        assert len(result.content.encode("utf-8")) <= MAX_RESPONSE_BYTES
        # First chunk's characters must all be present
        assert "A" in result.content

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
