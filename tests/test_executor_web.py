"""Tests for web operations: search and fetch.

Interface contract:
    web_search(query, count=5, api_key="", endpoint="") -> ToolResult
    web_fetch(url, max_chars=None) -> ToolResult

Testing seam: implementations must define these mockable internal functions:
    _call_search_api(query, count, api_key, endpoint) -> dict
    _fetch_url(url) -> str

Tests mock these internal functions to avoid real network calls.
"""

import pytest
from unittest.mock import patch, AsyncMock
from openalph.tools.web import web_search, web_fetch
from openalph.tools import ToolResult


# --- Fixtures ---


MOCK_SEARCH_RESPONSE = {
    "web": {
        "results": [
            {
                "title": "Python Documentation",
                "url": "https://docs.python.org",
                "description": "Official Python documentation and tutorials.",
            },
            {
                "title": "Real Python",
                "url": "https://realpython.com",
                "description": "Python tutorials and articles for all levels.",
            },
            {
                "title": "Python Package Index",
                "url": "https://pypi.org",
                "description": "Repository of software for Python.",
            },
        ]
    }
}

MOCK_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
<h1>Hello World</h1>
<p>This is a test paragraph with some content.</p>
<p>Another paragraph here.</p>
<script>var x = 'ignore this';</script>
<nav>Navigation stuff to ignore</nav>
</body>
</html>
"""


# --- web_search ---


class TestWebSearch:

    @pytest.mark.asyncio
    async def test_returns_formatted_results(self):
        """Search returns formatted results with title, url, snippet."""
        with patch(
            "openalph.tools.web._call_search_api",
            new_callable=AsyncMock,
            return_value=MOCK_SEARCH_RESPONSE,
        ):
            result = await web_search("python tutorials")

        assert result.is_error is False
        assert isinstance(result, ToolResult)
        # Results should contain titles and URLs from mock
        assert "Python Documentation" in result.content
        assert "https://docs.python.org" in result.content

    @pytest.mark.asyncio
    async def test_passes_query_to_api(self):
        """Query string is passed to the search API."""
        with patch(
            "openalph.tools.web._call_search_api",
            new_callable=AsyncMock,
            return_value=MOCK_SEARCH_RESPONSE,
        ) as mock_api:
            await web_search("test query", count=3, api_key="key", endpoint="ep")

        mock_api.assert_awaited_once_with("test query", 3, "key", "ep")

    @pytest.mark.asyncio
    async def test_count_parameter(self):
        """count parameter is forwarded to the API."""
        with patch(
            "openalph.tools.web._call_search_api",
            new_callable=AsyncMock,
            return_value=MOCK_SEARCH_RESPONSE,
        ) as mock_api:
            await web_search("test", count=10)

        args = mock_api.call_args
        assert args[0][1] == 10  # count is second positional arg

    @pytest.mark.asyncio
    async def test_api_error_returns_tool_error(self):
        """API failure → is_error=True."""
        with patch(
            "openalph.tools.web._call_search_api",
            new_callable=AsyncMock,
            side_effect=Exception("API rate limit exceeded"),
        ):
            result = await web_search("test")

        assert result.is_error is True
        assert "rate limit" in result.content.lower() or "error" in result.content.lower()

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Empty search results → not an error, just empty content."""
        with patch(
            "openalph.tools.web._call_search_api",
            new_callable=AsyncMock,
            return_value={"web": {"results": []}},
        ):
            result = await web_search("obscure query no results")

        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_multiple_results_formatted(self):
        """Multiple results are each formatted with title/url/snippet."""
        with patch(
            "openalph.tools.web._call_search_api",
            new_callable=AsyncMock,
            return_value=MOCK_SEARCH_RESPONSE,
        ):
            result = await web_search("python")

        # All three mock results should appear
        assert "Python Documentation" in result.content
        assert "Real Python" in result.content
        assert "Python Package Index" in result.content


# --- web_fetch ---


class TestWebFetch:

    @pytest.mark.asyncio
    async def test_returns_readable_content(self):
        """Fetched HTML is converted to readable text."""
        with patch(
            "openalph.tools.web._fetch_url",
            new_callable=AsyncMock,
            return_value=MOCK_HTML,
        ):
            result = await web_fetch("https://example.com")

        assert result.is_error is False
        assert isinstance(result, ToolResult)
        # Should contain the meaningful content
        assert "Hello World" in result.content
        assert "test paragraph" in result.content

    @pytest.mark.asyncio
    async def test_strips_scripts_and_nav(self):
        """Script and navigation content should be stripped or minimized."""
        with patch(
            "openalph.tools.web._fetch_url",
            new_callable=AsyncMock,
            return_value=MOCK_HTML,
        ):
            result = await web_fetch("https://example.com")

        # Script content should not appear
        assert "ignore this" not in result.content

    @pytest.mark.asyncio
    async def test_max_chars_truncates(self):
        """max_chars parameter limits output length."""
        long_html = "<html><body>" + "<p>Content. </p>" * 10000 + "</body></html>"
        with patch(
            "openalph.tools.web._fetch_url",
            new_callable=AsyncMock,
            return_value=long_html,
        ):
            result = await web_fetch("https://example.com", max_chars=500)

        assert len(result.content) <= 600  # some overhead OK

    @pytest.mark.asyncio
    async def test_max_chars_none_returns_full(self):
        """max_chars=None returns full content."""
        with patch(
            "openalph.tools.web._fetch_url",
            new_callable=AsyncMock,
            return_value=MOCK_HTML,
        ):
            result = await web_fetch("https://example.com", max_chars=None)

        assert result.is_error is False
        assert "Hello World" in result.content

    @pytest.mark.asyncio
    async def test_fetch_error_returns_tool_error(self):
        """Network/HTTP error → is_error=True."""
        with patch(
            "openalph.tools.web._fetch_url",
            new_callable=AsyncMock,
            side_effect=Exception("Connection refused"),
        ):
            result = await web_fetch("https://unreachable.example.com")

        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_passes_url_to_fetcher(self):
        """URL is passed through to the fetch function."""
        with patch(
            "openalph.tools.web._fetch_url",
            new_callable=AsyncMock,
            return_value="<html><body>OK</body></html>",
        ) as mock_fetch:
            await web_fetch("https://specific-url.example.com/page")

        mock_fetch.assert_awaited_once_with("https://specific-url.example.com/page")

    @pytest.mark.asyncio
    async def test_plain_text_passed_through(self):
        """Non-HTML content (plain text) is returned as-is."""
        with patch(
            "openalph.tools.web._fetch_url",
            new_callable=AsyncMock,
            return_value="Just plain text, no HTML tags here.",
        ):
            result = await web_fetch("https://example.com/plain.txt")

        assert "Just plain text" in result.content
