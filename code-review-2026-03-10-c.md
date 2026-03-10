# OpenAlph Code Review — 2026-03-10 (Review #3)

**Scope:** Full source review (~3,860 lines across 17 files)  
**Focus:** New findings only (excludes items in code-review-advisories.md)

---

## Summary Table

| # | Severity | Category | File | Issue |
|---|----------|----------|------|-------|
| 1 | CRITICAL | Security | `tools/shell.py` | Shell injection via unescaped command strings |
| 2 | CRITICAL | Reliability | `matrix.py` | Gap-fill loop logic error causes missed overlap detection |
| 3 | IMPORTANT | Correctness | `agent.py` | Context overflow check happens after user message added |
| 4 | IMPORTANT | Reliability | `tools/web.py` | HTML stripping regex vulnerable to ReDoS |
| 5 | IMPORTANT | Security | `tools/file.py` | No path traversal protection in file operations |
| 6 | IMPORTANT | Reliability | `matrix.py` | `on_tool_intent` async errors silently swallowed |
| 7 | IMPORTANT | Correctness | `heartbeat.py` | Zero interval accepted, causes busy-loop |
| 8 | IMPORTANT | Reliability | `session.py` | `_safe_call_id` collision risk on empty/sanitized IDs |
| 9 | IMPORTANT | Performance | `tools/file.py` | `read_file` loads entire file even with offset/limit |
| 10 | IMPORTANT | Architecture | `matrix.py` | `send()` marks content as HTML without escaping |
| 11 | MINOR | Correctness | `agent.py` | `cancel()` doesn't clear task reference after cancellation |
| 12 | MINOR | Reliability | `provider.py` | No timeout on LLM API calls |
| 13 | MINOR | Correctness | `config.py` | Subprocess commands lack timeout for API key resolution |
| 14 | MINOR | Reliability | `agent.py` | Typing indicator may not clear on exception paths |
| 15 | MINOR | Correctness | `matrix.py` | Gap-fill warning logs even when overlap found with no new messages |

---

## Detailed Findings

### Finding 1: Shell Injection via Unescaped Commands

**Severity:** CRITICAL  
**File:** `src/openalph/tools/shell.py` (lines 31-38)  
**Category:** Security

**What's Wrong:**
The `run_shell` function passes user-provided `command` directly to `asyncio.create_subprocess_shell()` without any sanitization or escaping:

```python
proc = await asyncio.create_subprocess_shell(
    command,  # Direct user input from LLM tool call
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    cwd=cwd,
    env=process_env,
    start_new_session=True,
)
```

While the advisories note that "Unix user isolation IS the sandbox boundary," this doesn't address the risk of accidental command corruption from special characters in filenames, or the fact that a compromised LLM API could inject shell metacharacters through tool inputs.

**Suggested Fix:**
Use `shlex.quote()` on any user-provided path components before shell execution, or better yet, use `asyncio.create_subprocess_exec()` with a list of arguments instead of shell mode:

```python
# Option 1: Quote the command
import shlex
proc = await asyncio.create_subprocess_shell(
    shlex.quote(command),  # Escape metacharacters
    ...
)

# Option 2: Use exec instead of shell (breaks shell features like pipes)
proc = await asyncio.create_subprocess_exec(
    "/bin/sh", "-c", command,
    ...
)
```

---

### Finding 2: Gap-Fill Loop Logic Error

**Severity:** CRITICAL  
**File:** `src/openalph/matrix.py` (lines 248-290)  
**Category:** Reliability / Correctness

**What's Wrong:**
In `_activate_room`, the gap-fill loop has a logic error where `found_overlap` is set inside the inner message loop, but the outer while loop only checks it after processing the entire chunk. If overlap is found mid-chunk, the remaining messages in that chunk are still processed and added to `new_messages`, even though they predate the known history:

```python
for msg in response.chunk:  # newest-first order
    ev_id = getattr(msg, 'event_id', None)
    if ev_id and ev_id in known_ids:
        found_overlap = True
        break  # Only breaks inner for loop
    if hasattr(msg, 'body') and msg.sender != self.config.user_id:
        new_messages.append(msg)  # Already added before overlap check!

if found_overlap or not response.end:  # Checked after for loop
    break
```

This causes messages between the overlap point and end of chunk to be incorrectly added as "new" messages, potentially duplicating history or creating causality violations.

**Suggested Fix:**
Restructure the loop to check for overlap before adding messages:

```python
for msg in response.chunk:
    ev_id = getattr(msg, 'event_id', None)
    if ev_id and ev_id in known_ids:
        found_overlap = True
        break
    if hasattr(msg, 'body') and msg.sender != self.config.user_id:
        new_messages.append(msg)

if found_overlap:
    break  # Exit while loop immediately
if not response.end:
    break
start_token = response.end
```

---

### Finding 3: Context Overflow Check Timing Bug

**Severity:** IMPORTANT  
**File:** `src/openalph/agent.py` (lines 114-130)  
**Category:** Correctness

**What's Wrong:**
The context overflow check happens AFTER the user message has already been appended to history, but the check doesn't account for the message just added. If the message pushes context over the limit, the agent will still attempt an LLM call:

```python
history.append({"role": "user", "content": text})  # Added first

for iteration in range(self.config.max_iterations):
    context_tokens = self._estimate_context_tokens(room_id)  # Check after
    available = self.config.model_max_tokens - self.config.max_tokens
    if context_tokens > available:
        raise ContextOverflowError(...)
```

Additionally, the overflow check happens at the start of each iteration, so if a tool result pushes context over, it's caught before the next LLM call—but the user message that caused overflow in the first iteration is still in history.

**Suggested Fix:**
Move the overflow check before appending the user message, or include the new message in the estimate:

```python
# Check BEFORE adding to history
test_context = self._estimate_context_tokens(room_id) + len(text) // 4
if test_context > available:
    raise ContextOverflowError(test_context, self.config.model_max_tokens)

history.append({"role": "user", "content": text})
```

---

### Finding 4: ReDoS Vulnerability in HTML Stripping

**Severity:** IMPORTANT  
**File:** `src/openalph/tools/web.py` (lines 35-51)  
**Category:** Security / Performance

**What's Wrong:**
The `_strip_html_tags` function uses regex with `re.DOTALL` to match HTML tags, which can be vulnerable to Regular Expression Denial of Service (ReDoS) on maliciously crafted input:

```python
def _strip_html_tags(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    ...
```

The `.*?` patterns with `re.DOTALL` can exhibit catastrophic backtracking on inputs with many `<` or `>` characters near each other.

**Suggested Fix:**
Use an HTML parser (like BeautifulSoup or html.parser) instead of regex, or at minimum add a length limit before regex processing:

```python
def _strip_html_tags(html: str) -> str:
    # Defensive length limit
    if len(html) > MAX_RESPONSE_BYTES:
        html = html[:MAX_RESPONSE_BYTES]
    
    # Use a proper parser
    from html.parser import HTMLParser
    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = []
        def handle_data(self, data):
            self.text.append(data)
    parser = TextExtractor()
    parser.feed(html)
    return " ".join(parser.text)
```

---

### Finding 5: Path Traversal in File Operations

**Severity:** IMPORTANT  
**File:** `src/openalph/tools/file.py` (lines 12-29, 56-70, 88-104)  
**Category:** Security

**What's Wrong:**
The file tools (`read_file`, `write_file`, `edit_file`) accept arbitrary paths without validating they stay within the agent's workspace. A malicious or compromised LLM could request:

```
file_read(path="../../../../etc/passwd")
file_write(path="../../../../etc/cron.d/backdoor", content="...")
```

While the tools/__init__.py has some path resolution for relative paths, it doesn't prevent traversal attacks:

```python
if not os.path.isabs(file_path) and hasattr(agent_config, "workspace"):
    input = dict(input)
    input["path"] = str(agent_config.workspace / file_path)
```

This only prepends the workspace, so `workspace / "../etc/passwd"` still resolves outside.

**Suggested Fix:**
Resolve and validate the path stays within workspace:

```python
def _resolve_path(path: str, workspace: Path) -> Path:
    """Resolve path and ensure it stays within workspace."""
    resolved = (workspace / path).resolve()
    workspace_resolved = workspace.resolve()
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        raise ValueError(f"Path {path} escapes workspace")
    return resolved
```

---

### Finding 6: Silent Failure in Tool Intent Callback

**Severity:** IMPORTANT  
**File:** `src/openalph/matrix.py` (lines 374-384)  
**Category:** Reliability

**What's Wrong:**
The `on_tool_intent` callback is awaited but any exception is silently swallowed because it's not in a try-except block and the error just propagates up to be caught by a generic handler much later:

```python
# Emit tool intent before execution (for session logging / observability)
if on_tool_intent:
    await on_tool_intent(response.tool_calls, response.content)  # Errors lost
```

If session logging fails, the agent continues without knowing the log is incomplete. This could cause compliance/auditing issues.

**Suggested Fix:**
Wrap the callback in try-except with explicit logging:

```python
if on_tool_intent:
    try:
        await on_tool_intent(response.tool_calls, response.content)
    except Exception as e:
        logger.error(f"Failed to log tool intent: {e}")
        # Optionally: continue or fail based on requirements
```

---

### Finding 7: Zero Interval Accepted for Heartbeat

**Severity:** IMPORTANT  
**File:** `src/openalph/heartbeat.py` (lines 154-162, 203-221)  
**Category:** Correctness

**What's Wrong:**
`parse_interval` rejects negatives but accepts zero, and `start()` doesn't validate the interval:

```python
# In parse_interval
if num < 0:  # Only rejects negative, not zero
    return None

# In start()
self._intervals[room_id] = float(interval_seconds)  # No validation
```

A zero interval causes `asyncio.sleep(0)` in a tight loop, consuming 100% CPU.

**Suggested Fix:**
Add validation in both places:

```python
# In parse_interval
if num <= 0:
    return None

# In start()
if interval_seconds <= 0:
    raise ValueError("Interval must be positive")
if interval_seconds < 300:  # Also enforce minimum
    raise ValueError("Minimum interval is 5 minutes")
```

---

### Finding 8: Call ID Collision Risk

**Severity:** IMPORTANT  
**File:** `src/openalph/session.py` (lines 64-70)  
**Category:** Reliability

**What's Wrong:**
The `_safe_call_id` function can produce collisions:

```python
def _safe_call_id(self, call_id: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', call_id) or "unknown"
    return sanitized
```

If two different call_ids sanitize to the same value (e.g., `abc:123` and `abc;123` both become `abc_123`), their overflow files will collide.

**Suggested Fix:**
Include a hash of the original call_id or use a counter:

```python
def _safe_call_id(self, call_id: str) -> str:
    import hashlib
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', call_id) or "unknown"
    # Add hash suffix to prevent collisions
    short_hash = hashlib.sha256(call_id.encode()).hexdigest()[:8]
    return f"{safe}_{short_hash}"
```

---

### Finding 9: Inefficient File Reading with Offset/Limit

**Severity:** IMPORTANT  
**File:** `src/openalph/tools/file.py` (lines 23-29)  
**Category:** Performance

**What's Wrong:**
`read_file` loads the entire file into memory even when only a small portion is requested via offset/limit:

```python
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()  # Reads entire file

# Then slices
selected_lines = lines[start_idx:end_idx]
```

For a 100MB log file where only lines 1-10 are requested, this still allocates 100MB.

**Suggested Fix:**
Use line iteration for large files:

```python
if offset is not None or limit is not None:
    # Use iteration for partial reads
    selected_lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if offset and i < offset:
                continue
            if limit and len(selected_lines) >= limit:
                break
            selected_lines.append(line)
    content = ''.join(selected_lines)
else:
    # Full read for common case
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
```

---

### Finding 10: HTML Content Without Escaping

**Severity:** IMPORTANT  
**File:** `src/openalph/matrix.py` (lines 171-185)  
**Category:** Architecture / Security

**What's Wrong:**
The `send()` method sets `format: org.matrix.custom.html` but puts raw text (potentially containing HTML) in `formatted_body` without escaping:

```python
content = {
    "msgtype": "m.text",
    "body": text,
    "format": "org.matrix.custom.html",
    "formatted_body": text,  # Raw text with HTML markers!
}
```

If the agent outputs text like `<script>alert('xss')</script>`, Matrix clients may interpret it as HTML.

**Suggested Fix:**
Either escape HTML entities or don't claim the format is HTML:

```python
import html
content = {
    "msgtype": "m.text",
    "body": text,
    # Either escape:
    "format": "org.matrix.custom.html",
    "formatted_body": html.escape(text),
    # Or remove format claim for plain text:
    # "format": "org.matrix.custom.html",
    # "formatted_body": text,
}
```

---

### Finding 11: Cancel Doesn't Clear Task Reference

**Severity:** MINOR  
**File:** `src/openalph/agent.py` (lines 163-168)  
**Category:** Correctness

**What's Wrong:**
After `cancel()` is called, the `_current_task` reference is not cleared until `handle_input` exits (via its finally block). If someone calls `cancel()` twice, the second call operates on a stale task reference.

**Suggested Fix:**
Clear the reference immediately:

```python
def cancel(self) -> asyncio.Task | None:
    task = self._current_task
    if task:
        self._current_task = None  # Clear immediately
        task.cancel()
    return task
```

---

### Finding 12: No Timeout on LLM API Calls

**Severity:** MINOR  
**File:** `src/openalph/provider.py` (lines 234-277)  
**Category:** Reliability

**What's Wrong:**
The `complete()` function makes HTTP requests through the Anthropic/OpenAI SDKs without any explicit timeout. A hung connection could block the agent indefinitely.

**Suggested Fix:**
Add timeout parameters to SDK calls:

```python
# For Anthropic
response = await client.messages.create(
    ...,
    timeout=60.0,  # Add timeout
)

# For OpenAI
response = await client.chat.completions.create(
    ...,
    timeout=60.0,
)
```

---

### Finding 13: No Timeout on API Key Resolution Commands

**Severity:** MINOR  
**File:** `src/openalph/config.py` (lines 91-102, 215-226, 243-254)  
**Category:** Reliability

**What's Wrong:**
The subprocess calls for `api_key_cmd`, `password_cmd`, and `access_token_cmd` have no timeout:

```python
result = subprocess.run(
    cmd,
    shell=True,
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
    # No timeout!
)
```

A hung command (e.g., waiting for user input) blocks agent startup forever.

**Suggested Fix:**
Add a reasonable timeout (e.g., 10 seconds):

```python
result = subprocess.run(
    cmd,
    shell=True,
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=10,  # Prevent indefinite hang
)
```

---

### Finding 14: Typing Indicator May Not Clear

**Severity:** MINOR  
**File:** `src/openalph/matrix.py` (lines 399-437)  
**Category:** Reliability

**What's Wrong:**
The typing indicator is set to True at the start of message processing but cleared in a `finally` block. However, if `_set_typing` itself raises an exception (network error), the error propagates and the finally block may not execute properly depending on exception handling.

**Suggested Fix:**
Ensure typing is always cleared with a nested try-finally:

```python
await self._set_typing(room_id, True)
try:
    response = await self.agent.handle_input(...)
    ...
finally:
    try:
        await self._set_typing(room_id, False)
    except Exception:
        logger.warning("Failed to clear typing indicator")
```

---

### Finding 15: Incorrect Gap-Fill Warning

**Severity:** MINOR  
**File:** `src/openalph/matrix.py` (lines 270-276)  
**Category:** Correctness

**What's Wrong:**
The gap-fill warning is logged when `not found_overlap and new_messages`, but this triggers even if overlap WAS found but there just happened to be no new messages after it. The condition should check if we actually missed messages.

**Suggested Fix:**
Only warn if we hit the message cap without finding overlap:

```python
if not found_overlap and len(new_messages) >= GAP_FILL_MAX:
    logger.warning(
        "Gap-fill for %s: no overlap found after %d messages",
        room_id, len(new_messages),
    )
```

---

## End of Review
