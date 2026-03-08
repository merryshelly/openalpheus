"""Web operations: search and fetch.

Interface contract:
    web_search(query, count=5, api_key="", endpoint="") -> ToolResult
    web_fetch(url, max_chars=None) -> ToolResult

Testing seams (mockable internal functions):
    _call_search_api(query, count, api_key, endpoint) -> dict
    _fetch_url(url) -> str
"""

import re
from dataclasses import dataclass

from openalph.tools import ToolResult


# --- Internal functions (testing seams) ---


async def _call_search_api(
    query: str, count: int, api_key: str, endpoint: str
) -> dict:
    """Make raw search API call. Mockable for testing.

    In production, this would call a search API (Brave, etc.).
    Returns the raw JSON response as a dict.
    """
    # Placeholder implementation - would use httpx/aiohttp in production
    raise NotImplementedError("_call_search_api must be mocked for tests")


async def _fetch_url(url: str) -> str:
    """Fetch raw content from URL. Mockable for testing.

    In production, this would make an HTTP GET request.
    Returns the raw response body as a string.
    """
    # Placeholder implementation - would use httpx/aiohttp in production
    raise NotImplementedError("_fetch_url must be mocked for tests")


# --- HTML processing ---


def _strip_html_tags(html: str) -> str:
    """Strip HTML tags and extract readable text.

    Removes script, style, nav tags and their content.
    Converts remaining HTML to plain text by stripping tags.
    """
    # Remove script and style tags with their content
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


# --- Public API ---


async def web_search(
    query: str, count: int = 5, api_key: str = "", endpoint: str = ""
) -> ToolResult:
    """Search the web and return formatted results.

    Args:
        query: Search query string
        count: Number of results to return (default 5)
        api_key: API key for the search service
        endpoint: API endpoint URL

    Returns:
        ToolResult with formatted search results (title, url, snippet per result)
        or ToolResult with is_error=True on failure.
    """
    try:
        response = await _call_search_api(query, count, api_key, endpoint)

        # Extract results from response
        results = response.get("web", {}).get("results", [])

        if not results:
            return ToolResult(content="No results found.", is_error=False)

        # Format results
        lines = []
        for result in results:
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            snippet = result.get("description", result.get("snippet", ""))

            lines.append(f"Title: {title}")
            lines.append(f"URL: {url}")
            lines.append(f"Snippet: {snippet}")
            lines.append("")  # Blank line between results

        return ToolResult(content="\n".join(lines).strip(), is_error=False)

    except Exception as e:
        return ToolResult(content=f"Search error: {e}", is_error=True)


async def web_fetch(url: str, max_chars: int | None = None) -> ToolResult:
    """Fetch URL and extract readable content.

    Args:
        url: URL to fetch
        max_chars: Maximum characters to return (default None = no limit)

    Returns:
        ToolResult with readable text extracted from HTML.
        Plain text content passes through unchanged.
        Returns is_error=True on failure.
    """
    try:
        content = await _fetch_url(url)

        # Check if content looks like HTML
        if "<" in content and ">" in content:
            # Process HTML to extract readable text
            text = _strip_html_tags(content)
        else:
            # Plain text - pass through unchanged
            text = content

        # Apply max_chars limit if specified
        if max_chars is not None and len(text) > max_chars:
            # Truncate with head/tail preservation
            head_budget = max_chars // 2
            tail_budget = max_chars - head_budget

            head = text[:head_budget]
            tail = text[-tail_budget:]

            text = head + f"\n[truncated: {len(text) - max_chars} chars removed]\n" + tail

        return ToolResult(content=text, is_error=False)

    except Exception as e:
        return ToolResult(content=f"Fetch error: {e}", is_error=True)