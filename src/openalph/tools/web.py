"""Web operations: search and fetch.

Interface contract:
    web_search(query, count=5, api_key="", endpoint="") -> ToolResult
    web_fetch(url, max_chars=None, tool_config=None) -> ToolResult
    web_fetch_js(url, schema=None, effort="max", max_chars=None, nocache=False,
                 api_key="", base_url="https://api.tabstack.ai/v1",
                 timeout=90) -> ToolResult

Search uses the Brave Search API. Fetch uses httpx with HTML stripping.
web_fetch_js wraps Tabstack's cloud-browser extraction API (POST
/extract/markdown or /extract/json) for JS-rendered pages that web_fetch
returns as an unrendered shell. See
memory/projects/openalph/web-fetch-js-design.md for the full design.
"""

import json
import re
import ssl
from urllib.parse import urlsplit

import httpx

from openalph.tools import ToolResult

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
REQUEST_TIMEOUT = 15.0
USER_AGENT = "OpenAlph/0.1 (https://codeberg.org/merryshelly/openalph)"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2MB cap to prevent OOM on large responses

TABSTACK_BASE_URL = "https://api.tabstack.ai/v1"
TABSTACK_TIMEOUT = 90.0  # Tabstack's max-effort render runs 15-60s; 15s (web_fetch's
                         # timeout) is far too short.
TABSTACK_EFFORT_LEVELS = ("min", "standard", "max")
TABSTACK_CANONICAL_NETLOC = "api.tabstack.ai"  # H3: base_url host pin -- never send
                                                # the Tabstack Bearer key to any other host.

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


# Known unrendered-SPA-shell markers (design §7). Matched case-insensitively
# against the raw HTML. These are the telltale signs of a JS app that never
# ran: a "please enable JavaScript" notice, or a framework mount point that's
# still empty because no JS executed.
_JS_SHELL_MARKERS = (
    "enable javascript",
    "please enable js",
    'id="root"',
    'id="__next"',
    "data-reactroot",
    "ng-app",
    "<app-root",
)

# Below this many characters of stripped text, an HTML response is treated as
# "effectively empty" for JS-shell detection purposes (design §7).
_JS_SHELL_MIN_TEXT_CHARS = 200


def _looks_like_unrendered_spa(raw_html: str, stripped_text: str) -> bool:
    """Heuristic: does this HTML response look like an unrendered JS-SPA shell?

    Per design §7 (workspace-kdsn.202.1): true if EITHER
      - the stripped, readable text is under _JS_SHELL_MIN_TEXT_CHARS chars, OR
      - the raw HTML contains a known shell marker (case-insensitive).

    Callers should only invoke this for responses that already look like HTML
    (this function does not itself re-check that).

    Args:
        raw_html: The original, undecoded-from-tags response body.
        stripped_text: The readable text already extracted from raw_html.

    Returns:
        True if the response smells like an unrendered SPA shell.
    """
    if len(stripped_text) < _JS_SHELL_MIN_TEXT_CHARS:
        return True
    lowered = raw_html.lower()
    return any(marker in lowered for marker in _JS_SHELL_MARKERS)


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


async def web_fetch(
    url: str, max_chars: int | None = None, tool_config: dict | None = None
) -> ToolResult:
    """Fetch URL and extract readable content.

    Args:
        url: URL to fetch
        max_chars: Maximum characters to return (default None = no limit)
        tool_config: web_fetch's own tool config (optional). Only used to read
            "js_fallback_tool" (workspace-kdsn.202.1): when set, a successful
            fetch that looks like an unrendered JS-SPA shell gets a steering
            note appended pointing at that tool. Unset (default) = no note,
            so this never dangles for an agent without a JS-render tool.

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
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                chunks = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_RESPONSE_BYTES:
                        chunks.append(chunk[: MAX_RESPONSE_BYTES - (total - len(chunk))])
                        break
                    chunks.append(chunk)
                raw_bytes = b"".join(chunks)
            content = raw_bytes.decode("utf-8", errors="replace")

        # Strip HTML if it looks like HTML
        is_html = "<html" in content.lower() or "<body" in content.lower()
        if is_html:
            text = _strip_html_tags(content)
        else:
            text = content

        # --- JS-shell handoff nudge (workspace-kdsn.202.1) ---
        # Never blocks, never errors -- appends prose to an already-successful
        # fetch. Only considered for HTML responses, and only emitted when
        # web_fetch's own config names a fallback tool (gate keeps this from
        # dangling on a reference to a tool that isn't enabled).
        js_note = ""
        fallback_tool = (tool_config or {}).get("js_fallback_tool") if is_html else None
        if fallback_tool and _looks_like_unrendered_spa(content, text):
            js_note = (
                f"\n\n[web_fetch note: this page appears to be JavaScript-rendered "
                f"(little/no readable text extracted). Retry with {fallback_tool} "
                f"for full browser rendering.]"
            )

        # Apply max_chars limit (to the extracted text only; the note is
        # appended AFTER truncation so it's never cut into by the head/tail
        # split below).
        # BUG-4: with max_chars<=0, len(text) > max_chars was true, tail_budget
        # was 0 and text[-0:] returned the WHOLE string, so "truncation"
        # produced the full page plus a marker (a negative value duplicated
        # content). Use the same guard web_fetch_js already had: only truncate
        # for a genuine positive int (and not a bool, which is an int subclass).
        if (
            isinstance(max_chars, int)
            and not isinstance(max_chars, bool)
            and max_chars > 0
            and len(text) > max_chars
        ):
            head_budget = max_chars // 2
            tail_budget = max_chars - head_budget
            text = (
                text[:head_budget]
                + f"\n[truncated: {len(text) - max_chars} chars removed]\n"
                + text[-tail_budget:]
            )

        text += js_note

        return ToolResult(content=text, is_error=False)

    except httpx.HTTPStatusError as e:
        return ToolResult(content=f"Fetch error: HTTP {e.response.status_code}", is_error=True)
    except Exception as e:
        return ToolResult(content=f"Fetch error: {e}", is_error=True)


async def web_fetch_js(
    url: str,
    schema: dict | str | None = None,
    effort: str = "max",
    max_chars: int | None = None,
    nocache: bool = False,
    api_key: str = "",
    base_url: str = TABSTACK_BASE_URL,
    timeout: float = TABSTACK_TIMEOUT,
) -> ToolResult:
    """Fetch a JavaScript-rendered page via Tabstack's cloud browser.

    Wraps POST /extract/markdown (default) or POST /extract/json (when
    `schema` is given) -- see design §4/§5.4. Defensive throughout: a bad
    input shape returns is_error instead of raising, so a malformed call
    never kills the agent's turn.

    Args:
        url: The publicly accessible URL to fetch and render.
        schema: Optional JSON Schema (dict, or a JSON string to be parsed).
            If given, switches to /extract/json and returns structured JSON
            instead of markdown.
        effort: "min" | "standard" | "max" (default "max"). Any other value
            is coerced to "max" rather than erroring.
        max_chars: Truncate returned markdown to this many chars (head+tail).
            Ignored in schema mode.
        nocache: Bypass Tabstack's cache for real-time data. Default False.
            Only a real bool is honored; any other type is coerced to False.
        api_key: Tabstack API key (Bearer auth). Empty -> fail-safe is_error.
            Rejected (is_error, no request) if it contains a control/
            whitespace char invalid in an HTTP header.
        base_url: Tabstack API base URL. Must resolve to https://api.tabstack.ai
            (path/version, e.g. "/v1", is preserved) -- pinned so the Bearer
            api_key is never sent to a non-canonical host.
        timeout: Request timeout in seconds (Tabstack's max-effort render can
            take up to ~60s; default matches TABSTACK_TIMEOUT).

    Returns:
        ToolResult with markdown content, structured JSON (schema mode), or
        an is_error result carrying point-of-need steering text.
    """
    if not api_key:
        return ToolResult(
            is_error=True,
            content=(
                "web_fetch_js not configured: set api_key_cmd in "
                "workspace/tools/web_fetch_js.toml → op read "
                "'op://<vault>/API Credentials - tabstack/credential' "
                "(see browser-automation skill)."
            ),
        )

    # H2: reject an api_key containing control/whitespace chars that are not
    # valid in an HTTP header value (e.g. \r, \n) BEFORE ever attempting the
    # request -- a raw newline in the Authorization header is a header-
    # injection vector, and letting httpx choke on it risks the raw key
    # surfacing in an exception message instead of a clean is_error.
    if any(ord(c) < 32 or ord(c) == 127 for c in api_key):
        return ToolResult(is_error=True, content="web_fetch_js: invalid api_key format")

    # M1: url must be a non-empty string (existing "url is required" message
    # preserved verbatim for the empty/missing case), and -- if non-empty --
    # a full http(s) URL with a real host and no embedded credentials
    # (rejects "user:pass@host" URL-credential smuggling).
    if not isinstance(url, str) or not url:
        return ToolResult(content="web_fetch_js error: url is required", is_error=True)

    try:
        _url_parts = urlsplit(url)
    except ValueError:
        return ToolResult(
            is_error=True,
            content=(
                "web_fetch_js: url must be a full http(s) URL without "
                "embedded credentials"
            ),
        )
    if (
        _url_parts.scheme not in ("http", "https")
        or not _url_parts.netloc
        or "@" in _url_parts.netloc
    ):
        return ToolResult(
            is_error=True,
            content=(
                "web_fetch_js: url must be a full http(s) URL without "
                "embedded credentials"
            ),
        )

    if schema is not None:
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                return ToolResult(
                    is_error=True,
                    content=(
                        "web_fetch_js error: schema must be a JSON Schema "
                        f"object (invalid JSON string: {e})"
                    ),
                )
        if not isinstance(schema, dict):
            return ToolResult(
                is_error=True,
                content="web_fetch_js error: schema must be a JSON Schema object.",
            )

    if effort not in TABSTACK_EFFORT_LEVELS:
        effort = "max"

    # L3: accept only a real bool for nocache; coerce any other type (e.g.
    # the string "false", which bool("false") would wrongly make True) to
    # False rather than forwarding it verbatim.
    if not isinstance(nocache, bool):
        nocache = False

    # H3: pin base_url to the canonical Tabstack host BEFORE building the
    # endpoint or making any request. base_url is caller/config-supplied;
    # without this pin a misconfigured (or maliciously overridden) base_url
    # would send the Bearer api_key to an arbitrary host (credential exfil).
    # Path/version (e.g. "/v1") is preserved -- only the scheme+host is
    # pinned.
    _base_parts = urlsplit(base_url)
    if _base_parts.scheme != "https" or _base_parts.netloc != TABSTACK_CANONICAL_NETLOC:
        return ToolResult(
            is_error=True,
            content=(
                "web_fetch_js: refusing to send credentials to non-canonical "
                "base_url (must be https://api.tabstack.ai)"
            ),
        )

    # L2: select JSON mode by presence (schema is not None), not truthiness,
    # so an intentional empty schema {} still routes to /extract/json.
    endpoint = f"{base_url}/extract/json" if schema is not None else f"{base_url}/extract/markdown"
    body: dict = {"url": url, "effort": effort, "nocache": nocache}
    if schema is not None:
        body["json_schema"] = schema

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=_ssl_context) as client:
            resp = await client.post(
                endpoint,
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            # H1/L1: a non-JSON body must fail soft (is_error), never raise
            # past this point -- an uncaught JSONDecodeError would otherwise
            # be caught below by the generic `except Exception`, but that
            # path stringifies the raw parser exception into the message
            # instead of giving a clean, specific steer.
            try:
                data = resp.json()
            except Exception:
                return ToolResult(
                    is_error=True,
                    content="web_fetch_js: malformed response (expected markdown content)",
                )

        if schema is not None:
            text = json.dumps(data, indent=2)
        else:
            # H1/L1: require the markdown envelope to actually be
            # {"content": "<str>", ...} before trusting it. A non-dict body,
            # or a "content" field that isn't a string (e.g. a list or an
            # absent key defaulting oddly), must never reach
            # ToolResult.content -- the outer execute_tool's
            # redact_credentials(result.content) assumes a str and raises
            # TypeError on anything else, which would kill the agent's turn
            # instead of surfacing a clean tool error.
            if not isinstance(data, dict) or not isinstance(data.get("content"), str):
                return ToolResult(
                    is_error=True,
                    content="web_fetch_js: malformed response (expected markdown content)",
                )
            text = data["content"]
            # M2: only truncate for a real, positive int max_chars. A bad
            # optional value (0, negative, non-int, or a bool -- True/False
            # are technically int in Python but never a sane char count)
            # must never error, and must never be used to compute a
            # negative tail slice that produces MORE output than the
            # untruncated source (the old text[-0:] / text[-negative:] bug).
            if (
                isinstance(max_chars, int)
                and not isinstance(max_chars, bool)
                and max_chars > 0
                and len(text) > max_chars
            ):
                head_budget = max_chars // 2
                tail_budget = max_chars - head_budget
                text = (
                    text[:head_budget]
                    + f"\n[truncated: {len(text) - max_chars} chars removed]\n"
                    + text[-tail_budget:]
                )

        return ToolResult(content=text, is_error=False)

    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code in (401, 403):
            return ToolResult(
                is_error=True,
                content="authentication failed — check api_key_cmd / 1Password item.",
            )
        if code == 402:
            return ToolResult(
                is_error=True,
                content="Tabstack quota exhausted — check console at tabstack.ai.",
            )
        if code == 429:
            return ToolResult(
                is_error=True,
                content="Tabstack rate-limited — back off and retry.",
            )
        if code == 408:
            return ToolResult(
                is_error=True,
                content=(
                    "page took too long to render — try effort='standard' "
                    "or a more specific URL."
                ),
            )
        return ToolResult(is_error=True, content=f"web_fetch_js error: HTTP {code}")
    except httpx.TimeoutException:
        return ToolResult(
            is_error=True,
            content=(
                "page took too long to render — try effort='standard' "
                "or a more specific URL."
            ),
        )
    except Exception as e:
        # H2: scrub the raw api_key out of the exception message before it
        # ever reaches the caller -- a low-level httpx/protocol error can
        # otherwise echo request internals (headers included) verbatim,
        # which would leak the Bearer credential through a plain error string.
        msg = str(e)
        if api_key:
            msg = msg.replace(api_key, "[REDACTED]")
        return ToolResult(is_error=True, content=f"web_fetch_js error: {msg}")
