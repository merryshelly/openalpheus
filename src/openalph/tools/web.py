"""Web operations: search and fetch.

Interface contract:
    web_search(query, count=5, api_key="", endpoint="") -> ToolResult
    web_fetch(url, max_chars=None) -> ToolResult

Search uses the Brave Search API. Fetch uses httpx with HTML stripping.
"""

import re
import ssl

import httpx

from openalph.tools import ToolResult

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
REQUEST_TIMEOUT = 15.0
USER_AGENT = "OpenAlph/0.1 (https://codeberg.org/merryshelly/openalph)"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2MB cap to prevent OOM on large responses

# Use the system SSL context so httpx picks up system CA certs
_ssl_context = ssl.create_default_context()


# --- HTML processing ---


def _strip_html_tags(html: str) -> str:
    """Strip HTML tags and extract readable text."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --- Public API ---


async def web_search(
    query: str, count: int = 5, api_key: str = "", endpoint: str = ""
) -> ToolResult:
    """Search the web via Brave Search API.

    Args:
        query: Search query string
        count: Number of results to return (default 5)
        api_key: Brave API key
        endpoint: API endpoint URL (default: Brave)

    Returns:
        ToolResult with formatted search results or error.
    """
    if not api_key:
        return ToolResult(content="web_search error: no API key configured", is_error=True)

    url = endpoint or BRAVE_ENDPOINT

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, verify=_ssl_context) as client:
            resp = await client.get(
                url,
                params={"q": query, "count": count},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("web", {}).get("results", [])
        if not results:
            return ToolResult(content="No results found.", is_error=False)

        lines = []
        for r in results:
            title = r.get("title", "Untitled")
            link = r.get("url", "")
            snippet = r.get("description", r.get("snippet", ""))
            # Strip HTML from snippets (Brave sometimes returns <strong> etc.)
            snippet = _strip_html_tags(snippet)
            lines.append(f"Title: {title}")
            lines.append(f"URL: {link}")
            lines.append(f"Snippet: {snippet}")
            lines.append("")

        return ToolResult(content="\n".join(lines).strip(), is_error=False)

    except httpx.HTTPStatusError as e:
        return ToolResult(content=f"Search API error: {e.response.status_code}", is_error=True)
    except Exception as e:
        return ToolResult(content=f"Search error: {e}", is_error=True)


async def web_fetch(url: str, max_chars: int | None = None) -> ToolResult:
    """Fetch URL and extract readable content.

    Args:
        url: URL to fetch
        max_chars: Maximum characters to return (default None = no limit)

    Returns:
        ToolResult with readable text extracted from HTML, or error.
    """
    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            verify=_ssl_context,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            raw_bytes = resp.content
            if len(raw_bytes) > MAX_RESPONSE_BYTES:
                raw_bytes = raw_bytes[:MAX_RESPONSE_BYTES]
            content = raw_bytes.decode("utf-8", errors="replace")

        # Strip HTML if it looks like HTML
        if "<html" in content.lower() or "<body" in content.lower():
            text = _strip_html_tags(content)
        else:
            text = content

        # Apply max_chars limit
        if max_chars is not None and len(text) > max_chars:
            head_budget = max_chars // 2
            tail_budget = max_chars - head_budget
            text = (
                text[:head_budget]
                + f"\n[truncated: {len(text) - max_chars} chars removed]\n"
                + text[-tail_budget:]
            )

        return ToolResult(content=text, is_error=False)

    except httpx.HTTPStatusError as e:
        return ToolResult(content=f"Fetch error: HTTP {e.response.status_code}", is_error=True)
    except Exception as e:
        return ToolResult(content=f"Fetch error: {e}", is_error=True)
