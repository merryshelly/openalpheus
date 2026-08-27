"""Tests for web tools (search + fetch).

Tests mock httpx.AsyncClient to avoid real network calls while testing
the actual formatting, parsing, error handling, and truncation logic.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from openalph.tools.web import web_search, web_fetch, _strip_html_tags, MAX_RESPONSE_BYTES
from openalph.tools import ToolResult, execute_tool


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
