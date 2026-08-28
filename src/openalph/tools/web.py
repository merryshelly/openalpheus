"""Web operations: search and fetch.

Interface contract:
    web_search(query, count=5, api_key="", endpoint="") -> ToolResult
    web_fetch(url, max_chars=None, tool_config=None, offset=None) -> ToolResult
    web_fetch_js(url, schema=None, effort="max", max_chars=None, nocache=False,
                 api_key="", base_url="https://api.tabstack.ai/v1",
                 timeout=90, offset=None) -> ToolResult

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

# kdsn.291: default window size for offset windowing. Just UNDER the
# fleet-default truncation_limit (50_000 chars): the navigation marker
# (<=~120 chars) rides ahead of the window, so marker+window must fit inside
# the downstream framework head/tail truncation or the window's own middle
# gets shaved. 49_000 + marker < 50_000 -> default windows arrive intact.
DEFAULT_WINDOW_CHARS = 49_000

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


# --- kdsn.291: offset windowing helpers ---


def _is_pos_int(v) -> bool:
    """Genuine positive int (bool excluded — it is an int subclass)."""
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _validate_offset(offset, tool: str):
    """kdsn.291: validate an optional 0-based char offset.

    Returns None when valid (absent or a non-negative int), else an error
    message string. bool is explicitly rejected (isinstance(True, int) is
    True). Validation happens BEFORE any network request.
    """
    if offset is None:
        return None
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return (
            f"{tool} error: offset must be a non-negative integer "
            f"(got {type(offset).__name__})"
        )
    return None


FLOOR_WINDOW_CHARS = 1_000


def _window_header(
    offset: int,
    end: int,
    total: int,
    tool: str,
    result_limit: int | None = None,
    *,
    trimmed: bool = False,
) -> str:
    """Render a window navigation header for an already-selected end offset.

    ``trimmed`` is explicit because a result limit alone does not establish
    that a delivered window was shortened. Only the budgeted-window path may
    set it, and only for an end below the requested end.
    """
    parts = [f"chars {offset}-{end - 1} of {total}"]
    if offset > 0:
        parts.append(f"{offset} before")
    if trimmed:
        if result_limit is None:
            raise ValueError("a trimmed window header requires result_limit")
        parts.append(
            f"window trimmed to fit the {result_limit}-char result budget"
        )
    if end < total:
        parts.append(f"{total - end} after — continue with offset={end}")
    else:
        parts.append("end of text")
    return f"[{tool} window: " + " | ".join(parts) + "]\n"


def _trimmed_header_intervals(
    lower: int,
    upper: int,
    total: int,
) -> list[tuple[int, int]]:
    """Return bounded end intervals with stable trimmed-header digit widths.

    Within an interval, only ``end - 1``, ``end``, and ``total - end`` can
    affect a trimmed header's length. Their decimal widths change only at
    powers of ten (plus the one-character ``end - 1`` transition immediately
    after a power), so these endpoints cover every realized header length
    without walking the requested window.
    """
    starts = {lower, upper + 1}
    for exponent in range(1, len(str(total)) + 1):
        power = 10 ** exponent
        for start in (power, power + 1, total - power + 1):
            if lower < start <= upper:
                starts.add(start)
    if lower < total <= upper:
        starts.add(total)

    points = sorted(starts)
    return [
        (start, next_start - 1)
        for start, next_start in zip(points, points[1:])
    ]


def _window_text(text: str, offset: int, window: int, tool: str) -> str:
    """kdsn.291: navigation marker + text[offset:offset+window] (clamped to
    end of text). The marker is PREFIXED so it survives the downstream
    framework head/tail truncation, which keeps the head."""
    total = len(text)
    end = min(offset + window, total)
    return _window_header(offset, end, total, tool) + text[offset:end]


def _budgeted_window_text(
    text: str,
    offset: int,
    window: int,
    tool: str,
    result_limit: int,
    suffix: str = "",
    preserve_legacy_untrimmed: bool = False,
) -> str | None:
    """Return the maximal fitting window, or ``None`` below the 1k floor.

    An untrimmed window that already fits is returned byte-for-byte unchanged
    (for explicit windows, only with the redaction headroom still reserved).
    Otherwise, evaluate the finite set of decimal-width header intervals and
    select the largest real end that fits; this avoids a shrink/re-render loop
    whose endpoint can be stale at a digit boundary.
    """
    total = len(text)
    requested_end = min(offset + window, total)
    untrimmed_header = _window_header(offset, requested_end, total, tool)

    # Redaction occurs after this handler. Reserve its realistic expansion.
    # The byte-identical shortcut for already-fitting windows must keep the
    # headroom reserved for EXPLICITLY supplied max_chars values, or a
    # near-limit window (e.g. a clamped short page) could expand past the
    # limit post-redaction and be middle-cut. The default 49k window
    # (max_chars omitted — callers pass preserve_legacy_untrimmed=True only
    # for a true None) is exempt from the shortcut's headroom requirement,
    # to keep kdsn.291 byte identity; its ~900-char margin at the 50k cap
    # is a documented marked-cut residual. At the 1k floor the reserve
    # clamps to the surplus, matching the trimmed path below. The trimmed
    # path ALWAYS reserves the full (floor-clamped) headroom, for default
    # and explicit windows alike.
    REDACTION_HEADROOM = min(1_000, result_limit // 10)
    if preserve_legacy_untrimmed:
        shortcut_headroom = 0
    else:
        untrimmed_body_budget = (
            result_limit - len(untrimmed_header) - len(suffix)
        )
        shortcut_headroom = min(
            REDACTION_HEADROOM, max(0, untrimmed_body_budget - FLOOR_WINDOW_CHARS)
        )

    # Preserve byte identity for all already-fitting windows.
    if (
        len(untrimmed_header)
        + (requested_end - offset)
        + len(suffix)
        + shortcut_headroom
        <= result_limit
    ):
        return untrimmed_header + text[offset:requested_end] + suffix

    # Every rendered trimmed candidate must actually be shorter than the
    # requested window. A shorter request that cannot fit needs no artificial
    # sub-1k result: fail loud instead.
    lower = offset + FLOOR_WINDOW_CHARS
    upper = requested_end - 1
    if lower > upper:
        return None

    # The reserve above applies to BODY capacity only, and is never a
    # pre-redaction here: execute_tool is the single choke point that also
    # emits operator redaction notices. At the 1k floor the reserve is
    # clamped to the surplus so a known feasible floor window is not
    # rejected solely by the reserve.

    best_end: int | None = None
    best_content: str | None = None
    for interval_start, interval_end in _trimmed_header_intervals(lower, upper, total):
        # This header length is realized throughout the interval. The raw
        # candidate is exactly offset + result_limit - h - len(suffix); then
        # reserve headroom from its body budget before choosing the endpoint.
        interval_header = _window_header(
            offset,
            interval_start,
            total,
            tool,
            result_limit,
            trimmed=True,
        )
        header_length = len(interval_header)
        raw_candidate_end = offset + result_limit - header_length - len(suffix)
        raw_body_budget = raw_candidate_end - offset
        if raw_body_budget < FLOOR_WINDOW_CHARS:
            continue
        reserved_headroom = min(
            REDACTION_HEADROOM,
            raw_body_budget - FLOOR_WINDOW_CHARS,
        )
        candidate_end = min(interval_end, raw_candidate_end - reserved_headroom)
        if candidate_end < interval_start:
            continue

        # Render the candidate's real header, rather than trusting an assumed
        # digit width, before accepting it. This is a bounded enumeration of
        # the realized header-length range, not a monotone shrink loop.
        header = _window_header(
            offset,
            candidate_end,
            total,
            tool,
            result_limit,
            trimmed=True,
        )
        body_length = candidate_end - offset
        actual_body_budget = result_limit - len(header) - len(suffix)
        actual_reserved_headroom = min(
            REDACTION_HEADROOM,
            max(0, actual_body_budget - FLOOR_WINDOW_CHARS),
        )
        if (
            body_length < FLOOR_WINDOW_CHARS
            or body_length + actual_reserved_headroom > actual_body_budget
        ):
            continue

        if best_end is None or candidate_end > best_end:
            best_end = candidate_end
            best_content = header + text[offset:candidate_end] + suffix

    return best_content


def _window_budget_error(tool: str, result_limit: int) -> str:
    """Compact post-fetch floor error; tiny configured caps may still cut it."""
    return (
        f"{tool} error: {result_limit}-char result budget cannot fit the header "
        "plus a 1,000-char window; raise truncation_limit."
    )


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
    url: str,
    max_chars: int | None = None,
    tool_config: dict | None = None,
    offset: int | None = None,
    result_limit: int | None = None,
) -> ToolResult:
    """Fetch URL and extract readable content.

    Args:
        url: URL to fetch
        max_chars: Without offset: head+tail truncation cap (default None =
            no limit). With offset: window size — chars to read starting at
            offset (default DEFAULT_WINDOW_CHARS).
        offset: 0-based char offset into the extracted text (kdsn.291).
            Enables window mode: read a specific region of a large page
            (e.g. the middle, which head/tail truncation makes unreachable).
            None (default) = legacy behavior, byte-identical.
        tool_config: web_fetch's own tool config (optional). Only used to read
            "js_fallback_tool" (workspace-kdsn.202.1): when set, a successful
            fetch that looks like an unrendered JS-SPA shell gets a steering
            note appended pointing at that tool. Unset (default) = no note,
            so this never dangles for an agent without a JS-render tool.

    Returns:
        ToolResult with readable text extracted from HTML, or error.
    """
    offset_error = _validate_offset(offset, "web_fetch")
    if offset_error:
        return ToolResult(content=offset_error, is_error=True)

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

        # kdsn.291: window mode (offset) takes precedence over the legacy
        # head/tail cap. Legacy path: BUG-4 guard preserved — only truncate
        # for a genuine positive int (bool is an int subclass; max_chars<=0
        # used to make text[-0:] return the WHOLE string).
        if offset is not None:
            total = len(text)
            if total == 0:
                return ToolResult(
                    content=(
                        "web_fetch error: extracted text is empty (0 chars) — "
                        "nothing to window"
                    ),
                    is_error=True,
                )
            if offset >= total:
                return ToolResult(
                    content=(
                        f"web_fetch error: offset {offset} is beyond the end of "
                        f"the extracted text ({total} chars) — use offset < {total}"
                    ),
                    is_error=True,
                )
            window = max_chars if _is_pos_int(max_chars) else DEFAULT_WINDOW_CHARS
            if _is_pos_int(result_limit):
                budgeted = _budgeted_window_text(
                    text, offset, window, "web_fetch", result_limit, js_note,
                    preserve_legacy_untrimmed=max_chars is None,
                )
                if budgeted is None:
                    return ToolResult(
                        content=_window_budget_error("web_fetch", result_limit),
                        is_error=True,
                    )
                return ToolResult(content=budgeted, is_error=False)
            text = _window_text(text, offset, window, "web_fetch")
        elif _is_pos_int(max_chars) and len(text) > max_chars:
            head_budget = max_chars // 2
            tail_budget = max_chars - head_budget
            total = len(text)
            text = (
                text[:head_budget]
                + f"\n[truncated: {total - max_chars} chars removed — text is "
                  f"{total} chars total; re-fetch with offset={total // 2} "
                  f"to read a specific region]\n"
                + text[-tail_budget:]
            )
        elif len(text) > DEFAULT_WINDOW_CHARS:
            # kdsn.291: full text larger than the typical framework cap will be
            # head/tail-truncated downstream with only a generic marker. The
            # note is PREPENDED so it survives that truncation (the head is
            # kept) and carries the total size + windowing steer at the point
            # of need. Only the no-window/no-cap path gets it — window mode
            # and the legacy cap already carry their own navigation.
            total = len(text)
            text = (
                f"[web_fetch note: full text is {total} chars — larger than the "
                f"typical 50,000-char tool-result limit, so expect head+tail "
                f"truncation if a [truncated:] marker appears. To read a specific "
                f"region, re-fetch with offset (0-based char position) and "
                f"max_chars (window size), e.g. offset={total // 2} for the "
                f"middle.]\n" + text
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
    offset: int | None = None,
    result_limit: int | None = None,
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
        max_chars: Without offset: truncate returned markdown to this many
            chars (head+tail). With offset: window size (chars to read
            starting at offset, default DEFAULT_WINDOW_CHARS). Ignored in
            schema mode.
        offset: 0-based char offset into the rendered markdown (kdsn.291).
            Window mode: read a specific region of long rendered pages.
            Markdown mode ONLY — errors when a schema is given (schema mode
            returns structured JSON, which windowing does not apply to).
            None (default) = legacy behavior, byte-identical.
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

    # kdsn.291: offset is markdown-mode only. A window request in schema mode
    # would be silently ignored (schema output is structured JSON) — an
    # explicit misdirected call, so it errors before any request.
    if offset is not None and schema is not None:
        return ToolResult(
            is_error=True,
            content=(
                "web_fetch_js error: offset is markdown-mode only — omit schema "
                "(or drop offset) to window the rendered markdown"
            ),
        )
    offset_error = _validate_offset(offset, "web_fetch_js")
    if offset_error:
        return ToolResult(content=offset_error, is_error=True)

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
            # kdsn.291: window mode (offset) takes precedence; legacy
            # head/tail truncation only when no offset was given.
            if offset is not None:
                total = len(text)
                if total == 0:
                    return ToolResult(
                        is_error=True,
                        content=(
                            "web_fetch_js error: rendered markdown is empty "
                            "(0 chars) — nothing to window"
                        ),
                    )
                if offset >= total:
                    return ToolResult(
                        is_error=True,
                        content=(
                            f"web_fetch_js error: offset {offset} is beyond the "
                            f"end of the markdown ({total} chars) — use "
                            f"offset < {total}"
                        ),
                    )
                window = max_chars if _is_pos_int(max_chars) else DEFAULT_WINDOW_CHARS
                if _is_pos_int(result_limit):
                    budgeted = _budgeted_window_text(
                        text, offset, window, "web_fetch_js", result_limit,
                        preserve_legacy_untrimmed=max_chars is None,
                    )
                    if budgeted is None:
                        return ToolResult(
                            content=_window_budget_error("web_fetch_js", result_limit),
                            is_error=True,
                        )
                    return ToolResult(content=budgeted, is_error=False)
                text = _window_text(text, offset, window, "web_fetch_js")
            # M2: only truncate for a real, positive int max_chars. A bad
            # optional value (0, negative, non-int, or a bool -- True/False
            # are technically int in Python but never a sane char count)
            # must never error, and must never be used to compute a
            # negative tail slice that produces MORE output than the
            # untruncated source (the old text[-0:] / text[-negative:] bug).
            elif (
                isinstance(max_chars, int)
                and not isinstance(max_chars, bool)
                and max_chars > 0
                and len(text) > max_chars
            ):
                head_budget = max_chars // 2
                tail_budget = max_chars - head_budget
                # kdsn.291 (review 2): same total + midpoint steering as
                # web_fetch's legacy marker — an agent paginating rendered
                # markdown gets the offset steer at the point of need.
                total = len(text)
                text = (
                    text[:head_budget]
                    + f"\n[truncated: {total - max_chars} chars removed — text is "
                      f"{total} chars total; re-fetch with offset={total // 2} "
                      f"to read a specific region]\n"
                    + text[-tail_budget:]
                )
            elif len(text) > DEFAULT_WINDOW_CHARS:
                # kdsn.291: as in web_fetch — full rendered markdown beyond the
                # typical framework cap gets a head-anchored navigation note.
                # LAST in the chain: window mode and the legacy cap take
                # precedence (they carry their own navigation).
                total = len(text)
                text = (
                    f"[web_fetch_js note: full markdown is {total} chars — larger "
                    f"than the typical 50,000-char tool-result limit, so expect "
                    f"head+tail truncation if a [truncated:] marker appears. To "
                    f"read a specific region, re-fetch with offset (0-based char "
                    f"position) and max_chars (window size), e.g. "
                    f"offset={total // 2} for the middle.]\n" + text
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
