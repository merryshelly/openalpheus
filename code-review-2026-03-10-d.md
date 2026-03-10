# OpenAlph Code Review — 2026-03-10 (Review 4)

**Reviewer:** Sub-agent (cold review, advisory-aware)
**Source:** ~3,886 lines across 17 files

---

## Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | MINOR | agent.py:176-178 | History left in inconsistent state on max_iterations |
| 2 | MINOR | tools/file.py:84-110 | edit_file doesn't check for binary files |
| 3 | MINOR | heartbeat.py:143-155 | Race condition in concurrent heartbeat start/stop persistence |
| 4 | MINOR | session.py:107-117 | Overflow files never cleaned up |
| 5 | MINOR | tools/__init__.py:175-184 | api_key_cmd failure silently ignored |
| 6 | MINOR | session.py:211-246 | build_context doesn't validate role sequence |

---

## Findings

### Finding 1: History left in inconsistent state on max_iterations

**Severity:** MINOR
**File:** `src/openalph/agent.py`, lines 176-178

**What's wrong:**
When `max_iterations` is reached, the function returns the limit message without appending an assistant response to history:

```python
# Hit max iterations - return limit message
return "[Tool call limit reached. Please summarize your progress.]"
```

At this point, history contains:
- User message (appended at line 71)
- Assistant messages with tool_calls (appended in each iteration)
- Tool result messages (appended after each tool call)

But no final assistant text response. The next call to `handle_input` would append a new user message, resulting in history that transitions from `tool` role directly to `user` role without an intervening assistant response. This violates the expected conversation structure.

**Suggested fix:**
Append an assistant message before returning:

```python
# Hit max iterations - append limit message to history and return
limit_msg = "[Tool call limit reached. Please summarize your progress.]"
history.append({"role": "assistant", "content": limit_msg})
return limit_msg
```

**Impact:** The LLM can technically handle this pattern (it sees tool results followed by a new user message), but it's inconsistent with the expected conversation flow. Minor correctness issue with low practical impact.

---

### Finding 2: edit_file doesn't check for binary files

**Severity:** MINOR
**File:** `src/openalph/tools/file.py`, lines 84-110

**What's wrong:**
`read_file` calls `_is_binary(path)` and returns an error for binary files, but `edit_file` does not:

```python
async def edit_file(path: str, old_text: str, new_text: str) -> ToolResult:
    try:
        if not os.path.exists(path):
            return ToolResult(content=f"Error: File not found: {path}", is_error=True)
        
        if not os.path.isfile(path):
            return ToolResult(content=f"Error: Path is not a file: {path}", is_error=True)
        
        # No binary check here!
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
```

If an LLM attempts to edit a binary file, `content = f.read()` could fail with `UnicodeDecodeError` (which is caught and returns an error), or worse, could partially succeed with corrupted data. If the replacement somehow matches binary bytes represented as UTF-8, the file could be corrupted.

**Suggested fix:**
Add the same binary check as `read_file`:

```python
if _is_binary(path):
    return ToolResult(content=f"Error: Binary file cannot be edited: {path}", is_error=True)
```

**Impact:** Low probability in practice (LLM would need to try editing a binary file), but could corrupt files if it happens.

---

### Finding 3: Race condition in concurrent heartbeat start/stop persistence

**Severity:** MINOR
**File:** `src/openalph/heartbeat.py`, lines 67-90 and 92-112

**What's wrong:**
Both `start()` and `stop()` call `await self._persist()` after updating the internal state dicts (`_tasks`, `_start_times`, `_intervals`). If two operations run concurrently:

```python
# Task A: start room1
_tasks[room1] = task1
_start_times[room1] = time1
_intervals[room1] = interval1
# Task B: stop room2 (interleaves)
del _tasks[room2]
del _start_times[room2]
del _intervals[room2]
# Task A: persist
await self._persist()  # Persists state including room1 but NOT room2
# Task B: persist
await self._persist()  # Overwrites with state NOT including room2
```

The persisted state can become inconsistent with the in-memory state. After a crash and resume, heartbeats might not be restored correctly.

**Suggested fix:**
Use an asyncio.Lock for state mutations:

```python
def __init__(self, ...):
    ...
    self._state_lock = asyncio.Lock()

async def start(self, room_id: str, interval_seconds: int | float) -> None:
    async with self._state_lock:
        # ... existing logic ...
        await self._persist()
```

**Impact:** In practice, heartbeats are managed serially by the Matrix event loop, so concurrent access is unlikely. But the API permits it and could lead to inconsistent state after crash recovery.

---

### Finding 4: Overflow files never cleaned up

**Severity:** MINOR
**File:** `src/openalph/session.py`, lines 107-117

**What's wrong:**
When tool output exceeds `OVERFLOW_THRESHOLD` (64KB), the full output is written to an overflow file:

```python
overflow_file = overflow_dir / f"{self._safe_call_id(call_id)}.txt"
overflow_file.write_text(output, encoding="utf-8")
```

There is no mechanism to clean up these files. For long-running agents with large tool outputs, the `sessions/overflow/` directory will grow unbounded.

**Suggested fix:**
Add a cleanup mechanism, either:
1. On session start, delete overflow files older than N days
2. Provide a CLI command to clean overflow files
3. Or document that overflow files should be periodically cleaned by the operator

**Impact:** Disk usage grows over time for agents with large tool outputs. Low priority for typical usage patterns.

---

### Finding 5: api_key_cmd failure silently ignored

**Severity:** MINOR
**File:** `src/openalph/tools/__init__.py`, lines 175-184

**What's wrong:**
When resolving `api_key` from `api_key_cmd` for `web_search`:

```python
if not api_key and "api_key_cmd" in tool_config:
    import subprocess
    try:
        result = subprocess.run(
            tool_config["api_key_cmd"], shell=True,
            capture_output=True, text=True, timeout=10,
        )
        api_key = result.stdout.strip()
    except Exception:
        pass
```

If the command fails (exception, timeout, or non-zero exit code with `check=True`), the exception is silently caught. No logging occurs, making debugging difficult. Note: `subprocess.run` without `check=True` does NOT raise for non-zero exit codes, so a failing command would set `api_key` to empty stdout.

**Suggested fix:**
Log the failure:

```python
try:
    result = subprocess.run(
        tool_config["api_key_cmd"], shell=True,
        capture_output=True, text=True, timeout=10,
    )
    api_key = result.stdout.strip()
    if result.returncode != 0:
        logger.warning("api_key_cmd returned exit code %d: %s",
                       result.returncode, result.stderr)
except subprocess.TimeoutExpired:
    logger.warning("api_key_cmd timed out after 10 seconds")
except Exception as e:
    logger.warning("api_key_cmd failed: %s", e)
```

**Impact:** Debugging difficulty when API key resolution fails silently. The search will fail with "no API key configured" but the root cause won't be logged.

---

### Finding 6: build_context doesn't validate role sequence

**Severity:** MINOR
**File:** `src/openalph/session.py`, lines 211-246

**What's wrong:**
`build_context` converts JSONL entries to LLM message format without validating the role sequence. The LLM expects:
- Conversation starts with `user` or `system`
- `tool` messages follow `assistant` messages with `tool_calls`
- `assistant` messages with `tool_calls` are followed by `tool` results

If the session log is corrupted or manually edited (unlikely but possible), `build_context` would produce invalid context that could confuse the LLM or cause API errors.

**Suggested fix:**
Add validation or at least logging for unexpected sequences:

```python
def build_context(self, room_id: str, *, skip_system: bool = True) -> list[dict]:
    entries = self.read(room_id)
    context = []
    last_role = None
    
    for entry in entries:
        role = entry.get("role")
        
        # Log unexpected transitions
        if last_role == "tool" and role == "user":
            logger.debug("Unexpected transition: tool -> user in %s", room_id)
        if last_role == "assistant" and role == "user":
            # Check if previous assistant had tool_calls
            if context and context[-1].get("role") == "assistant":
                if context[-1].get("tool_calls"):
                    logger.warning("Assistant with tool_calls followed by user message in %s", room_id)
        
        # ... rest of the logic ...
        last_role = role
```

**Impact:** Low risk in normal operation. Session logs are append-only and only written by the agent code. But if corruption occurs, this would make debugging harder.

---

## Files Reviewed

- `src/openalph/admin.py` (agent setup)
- `src/openalph/agent.py` (conversation loop)
- `src/openalph/cli.py` (CLI commands)
- `src/openalph/config.py` (configuration loading)
- `src/openalph/heartbeat.py` (recurring timers)
- `src/openalph/__init__.py` (empty)
- `src/openalph/__main__.py` (entry point)
- `src/openalph/matrix.py` (Matrix integration)
- `src/openalph/mention.py` (mention detection)
- `src/openalph/prompt.py` (system prompt assembly)
- `src/openalph/provider.py` (LLM provider adapter)
- `src/openalph/session.py` (JSONL session log)
- `src/openalph/tools/__init__.py` (tool registry)
- `src/openalph/tools/file.py` (file operations)
- `src/openalph/tools/shell.py` (shell executor)
- `src/openalph/tools/subagent.py` (sub-agent executor)
- `src/openalph/tools/web.py` (web search/fetch)

---

## Notes

All findings in this review are MINOR. The codebase is well-structured with clear separation of concerns. Previous reviews and the advisory process have already addressed the significant security and correctness issues. The remaining findings are edge cases and minor reliability improvements that could be addressed opportunistically.

No CRITICAL or IMPORTANT findings were identified that are not already documented in the advisories.
