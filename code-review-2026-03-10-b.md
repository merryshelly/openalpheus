# OpenAlph Code Review — 2026-03-10 (Round 2)

Cold review of ~3,800 lines across 17 files. Prior review fixed 23/25 issues.

## Summary Table

| # | Severity | File | Finding |
|---|----------|------|---------|
| 1 | CRITICAL | tools/__init__.py | Path traversal via symlink — `resolve()` check is bypassable |
| 2 | CRITICAL | matrix.py | Race condition: `_on_tool_call`/`_on_tool_intent` shared across rooms |
| 3 | IMPORTANT | config.py | `shell=True` for `api_key_cmd` without input validation |
| 4 | IMPORTANT | matrix.py | Gap-fill paginates with empty `start=""` — undefined behavior |
| 5 | IMPORTANT | session.py | `call_id` from LLM used directly as filename (path injection) |
| 6 | IMPORTANT | agent.py | `_room_locks` dict grows without bound (memory leak) |
| 7 | IMPORTANT | provider.py | `_client_cache` keyed on API key string — secrets as dict keys |
| 8 | IMPORTANT | matrix.py | Consecutive `tool_result` messages not merged for Anthropic |
| 9 | IMPORTANT | tools/web.py | `web_fetch` streams entire response into memory before size check |
| 10 | MINOR | heartbeat.py | `_persist` uses sync I/O in async context — blocks event loop |
| 11 | MINOR | agent.py | Token estimation counts `str(tc.input)` — Python repr, not JSON |
| 12 | MINOR | matrix.py | `_activate_room` gap-fill: `new_messages` collected in wrong order |
| 13 | MINOR | provider.py | Anthropic `cache_read_input_tokens` may not exist on all responses |
| 14 | MINOR | tools/__init__.py | `execute_tool` ignores caller-provided `timeout` parameter |
| 15 | MINOR | session.py | `read()` loads entire JSONL into memory — unbounded for long sessions |
| 16 | MINOR | matrix.py | `send()` sets `formatted_body` to raw text, not HTML |
| 17 | MINOR | heartbeat.py | `shutdown()` doesn't persist — heartbeats lost if `shutdown()` without prior `stop()` |
| 18 | STYLE | admin.py | `execute_plan` swallows `useradd` exit code 9 silently |

---

## Findings

### 1. CRITICAL — Path traversal via symlink in file tools

**File:** `tools/__init__.py`, `execute_tool` function

**Problem:** The path containment check uses `Path.resolve()` which follows symlinks. If an attacker (the LLM) creates a symlink inside the workspace pointing outside it, then references that symlink, `resolve()` will resolve through the symlink and the `relative_to()` check passes because the *symlink itself* is inside the workspace. But `resolve()` actually resolves to the *target*, so the check would catch this case.

Wait — re-reading: `resolve()` resolves the symlink to its final target, then checks that target is under workspace. So the symlink attack doesn't work directly. However, there's a TOCTOU issue: the path is validated, then passed to the file tool which opens it separately. Between check and use, a symlink could be created (by a concurrent shell tool call running in parallel via `asyncio.gather`).

**Severity upgrade rationale:** Tool calls execute in parallel (`asyncio.gather` in `agent.py`). A malicious LLM could issue a `shell` call to create a symlink and a `file_read` call targeting it simultaneously. The race window is real.

**Fix:** Resolve the path again inside the file operation functions themselves, or use `os.open` with `O_NOFOLLOW` and file descriptor-based operations. Alternatively, serialize file tool execution (don't run file tools in parallel with shell tools).

---

### 2. CRITICAL — Shared mutable callback creates cross-room data leak

**File:** `matrix.py`, `_handle_room_message`

**Problem:** `self.agent._on_tool_call` and `self.agent._on_tool_intent` are set per-message, but the `Agent` is shared across all rooms. If two rooms send messages near-simultaneously, the second message's callback overwrites the first's. Tool results from room A could be sent to room B via the wrong callback closure (which captures `room_id`).

The `_room_locks` in `agent.py` prevent concurrent `handle_input` for the *same* room, but different rooms can run concurrently. The callbacks are set *before* `handle_input` is called and cleared in `finally`, but a second room's message handler can overwrite them mid-flight.

**Fix:** Move callbacks into per-call parameters on `handle_input()` instead of setting them as instance attributes. Or pass them through the method signature:
```python
async def handle_input(self, text, room_id, on_tool_call=None, on_tool_intent=None):
```

---

### 3. IMPORTANT — `api_key_cmd` runs arbitrary shell commands from config

**File:** `config.py`, lines around `api_key_cmd` resolution

**Problem:** `api_key_cmd` from the TOML config is passed directly to `subprocess.run(..., shell=True)`. The config files live in `/etc/openalph/agents/` which should be root-owned, but if an agent can write to its own config (or if config is loaded from a user-writable path in dev mode), this is command injection.

**Fix:** Document the trust boundary clearly. Consider using `shlex.split()` and `shell=False` to reduce attack surface — most key commands are simple (`op read ...`, `cat /path/to/key`). Same issue applies to `password_cmd` and `access_token_cmd` in matrix config.

---

### 4. IMPORTANT — Gap-fill uses empty string as sync token

**File:** `matrix.py`, `_activate_room`, gap-fill section

**Problem:** `start_token = ""` is passed to `self.client.room_messages(room_id, start=start_token, limit=100)`. The `room_messages` API requires a valid pagination token. An empty string is not a valid token — behavior is undefined per the Matrix spec and depends on the homeserver implementation. nio may return the room's most recent messages or error.

**Fix:** Use `self.client.room_messages(room_id, start=self.client.rooms[room_id].prev_batch, limit=100)` or similar to get a valid sync token. Alternatively, use `since` from the sync response.

---

### 5. IMPORTANT — LLM-controlled `call_id` used as filename

**File:** `session.py`, `_overflow_path` and `append` method

**Problem:** `call_id` comes from the LLM's tool call response (`tc.id`) and is used directly to construct a filename: `overflow/{call_id}.txt`. A malicious model response could include `../../../etc/cron.d/evil` as a call_id, writing arbitrary files.

**Fix:** Sanitize `call_id` before using as filename:
```python
import re
safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', call_id)
```

---

### 6. IMPORTANT — `_room_locks` dict grows without bound

**File:** `agent.py`, `handle_input`

**Problem:** `self._room_locks[room_id] = asyncio.Lock()` — new locks are created for every unique room_id but never removed. Over a long-running agent lifetime with many rooms, this leaks memory. The `_rooms` dict has the same issue.

**Fix:** Implement an LRU eviction or explicit cleanup when rooms are no longer active. Or accept the leak as bounded by the number of rooms the agent has ever joined (which in practice may be fine — document the assumption).

---

### 7. IMPORTANT — API key used as dictionary key in `_client_cache`

**File:** `provider.py`, `_get_client`

**Problem:** The raw API key string is stored as part of the cache dictionary key: `("anthropic", config.api_key, None)`. This means the secret persists in memory as a dict key indefinitely and would appear in any heap dump, `repr()`, or debug output of the dict.

**Fix:** Use a hash of the key for the cache lookup:
```python
import hashlib
key_hash = hashlib.sha256(config.api_key.encode()).hexdigest()[:16]
key = ("anthropic", key_hash, None)
```

---

### 8. IMPORTANT — Anthropic message merging: consecutive `user` role messages

**File:** `provider.py`, `_convert_messages_for_anthropic`

**Problem:** Anthropic's API requires alternating user/assistant messages. When multiple tool results come back, each becomes a `{"role": "user", ...}` message. If there are multiple tool calls, multiple consecutive user-role messages are generated. The Anthropic API will reject this.

Looking more carefully: tool results are appended individually in `agent.py`'s tool loop, each as `{"role": "tool", ...}`. The converter maps each to a separate `{"role": "user", ...}`. Multiple tools = multiple consecutive user messages = API error.

**Fix:** Merge consecutive tool result entries into a single `{"role": "user", "content": [...tool_result blocks...]}` message:
```python
# After converting, merge consecutive user messages that contain tool_result blocks
merged = []
for msg in result:
    if (merged and merged[-1]["role"] == "user" and msg["role"] == "user"
            and isinstance(merged[-1].get("content"), list) and isinstance(msg.get("content"), list)):
        merged[-1]["content"].extend(msg["content"])
    else:
        merged.append(msg)
return merged
```

---

### 9. IMPORTANT — `web_fetch` buffers entire response before size check

**File:** `tools/web.py`, `web_fetch`

**Problem:** `resp = await client.get(url)` reads the entire response into memory. Only afterward is `len(raw_bytes) > MAX_RESPONSE_BYTES` checked. A malicious or very large URL could cause OOM before the check runs.

**Fix:** Use streaming:
```python
async with client.stream("GET", url) as resp:
    resp.raise_for_status()
    chunks = []
    total = 0
    async for chunk in resp.aiter_bytes():
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            break
        chunks.append(chunk)
    raw_bytes = b"".join(chunks)
```

---

### 10. MINOR — Sync file I/O in async `_persist`

**File:** `heartbeat.py`, `_persist`

**Problem:** `tmp_path.write_text(...)` and `os.replace(...)` are synchronous blocking calls inside an `async def` method. This blocks the event loop. The data is small so it's fast in practice, but it violates async discipline.

**Fix:** Use `asyncio.to_thread()` or `aiofiles`, or accept the tradeoff with a comment.

---

### 11. MINOR — Token estimation uses Python `str()` not JSON for tool input

**File:** `agent.py`, `_estimate_context_tokens`

**Problem:** `total_chars += len(str(tc.input))` — `str()` on a dict produces Python repr (`{'key': 'value'}`) which differs from JSON (`{"key": "value"}`). The estimate will be slightly off. Not dangerous but misleading.

**Fix:** Use `json.dumps(tc.input)` or accept the inaccuracy.

---

### 12. MINOR — Gap-fill message ordering assumption

**File:** `matrix.py`, `_activate_room`

**Problem:** The code assumes `response.chunk` returns messages in newest-first order (comment says "newest-first order"), but the Matrix spec says `room_messages` with `dir=b` returns newest-first. The code doesn't specify `dir`, so it depends on nio's default. If the default is forward, messages are collected in wrong order.

**Fix:** Explicitly pass `direction=MessageDirection.back` (or equivalent).

---

### 13. MINOR — Anthropic usage attribute may not exist

**File:** `provider.py`, `_parse_anthropic_response`

**Problem:** `response.usage.cache_read_input_tokens` and `cache_creation_input_tokens` are accessed directly but may not exist on all Anthropic responses (they're only present when prompt caching is used). If the SDK returns an object without these attributes, this raises `AttributeError`.

**Fix:** Use `getattr(response.usage, 'cache_read_input_tokens', None)`.

---

### 14. MINOR — Shell tool ignores user-provided `timeout`

**File:** `tools/__init__.py`, `execute_tool`

**Problem:** The shell tool schema includes a `timeout` parameter, but `execute_tool` passes `timeout=tool_config.get("default_timeout", 30)` — it ignores `input.get("timeout")`. The LLM can request a custom timeout but it's silently discarded.

**Fix:**
```python
timeout = input.get("timeout") or tool_config.get("default_timeout", 30)
```

---

### 15. MINOR — `SessionLog.read()` loads entire file into memory

**File:** `session.py`, `read` method

**Problem:** For long-running rooms, the JSONL file could grow to hundreds of MB. `read()` loads every line into a list of dicts. This is called by `build_context()` which is called on every message in gated rooms (context rehydration).

**Fix:** Add a `tail` parameter to `read()` that only loads the last N entries (seek to end, read backward). Or implement a separate `build_context_streaming` that processes the file without holding all entries in memory.

---

### 16. MINOR — `send()` sets `formatted_body` to raw text

**File:** `matrix.py`, `send` method

**Problem:** `"formatted_body": text` — the code comments "Simplified; could add markdown->HTML conversion" but setting `format: org.matrix.custom.html` with a non-HTML body means clients may render raw text as HTML (or not render it properly). Markdown syntax like `**bold**` won't render as bold.

**Fix:** Either remove the `format`/`formatted_body` fields (plain text is fine), or add actual markdown-to-HTML conversion. The current state is the worst of both worlds.

---

### 17. MINOR — `heartbeat.shutdown()` doesn't persist empty state

**File:** `heartbeat.py`, `shutdown`

**Problem:** `shutdown()` clears all internal state but doesn't call `_persist()`. If the agent shuts down cleanly via `shutdown()`, the heartbeats.json still contains the old entries. On next startup, `resume()` will restart all heartbeats. This may be intentional (persist-across-restart), but if `stop()` is the intended way to remove a heartbeat, then `shutdown()` behaving differently is surprising.

**Fix:** Document the behavior explicitly, or add a `persist` parameter to `shutdown()`.

---

### 18. STYLE — `useradd` exit code 9 silently accepted

**File:** `admin.py`, `execute_plan`

**Problem:** Exit code 9 means "username already in use" — silently continuing is probably intentional (idempotent setup). But there's no logging or documentation of why 9 is accepted. Other non-zero codes would raise `AdminError` but this special case is unexplained.

**Fix:** Add a comment: `# Exit code 9: user already exists (idempotent)`.

---

## Additional Observations (not findings)

- **Shell sandboxing:** Shell tool runs commands as the agent's Unix user with no additional sandboxing. Per prior review, this is by design — the Unix user is the sandbox boundary.

- **No rate limiting on tool calls:** A runaway LLM could make hundreds of shell calls in rapid succession. The `max_iterations` config caps total turns but not tool calls per turn (parallel execution via `asyncio.gather`).

- **Provider client cache never evicts:** `_client_cache` grows if configs change (key rotation). In practice bounded by number of distinct configs used.

- **`discover_tools` raises on unknown tool:** This means a stray `.toml` file in `tools/` crashes startup. Consider warning instead of raising.
