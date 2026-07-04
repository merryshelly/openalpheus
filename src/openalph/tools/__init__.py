"""OpenAlph tools package.

Tool registry, discovery, schema generation, and result truncation.
"""

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

logger = logging.getLogger(__name__)


# --- API key resolution cache ---
#
# web_search resolves its api_key via api_key_cmd (typically `op read op://...`),
# which makes a network call to the secret store (e.g. 1Password) on every
# invocation. Without caching, high search volume — e.g. parallel sub-agents each
# firing web_search — exhausts the service-account rate limit; the command then
# returns empty and web_search fails with "no API key configured" even though the
# credential is correct. Provider keys avoid this by resolving once at config load
# (config.py); this cache gives the same protection to per-call tool resolution.
_DEFAULT_API_KEY_CACHE_TTL = 3600.0  # seconds; also bounds max staleness after a credential rotation
_API_KEY_RETRY_BACKOFF = 60.0  # seconds to keep serving a stale key before retrying a failing api_key_cmd

# api_key_cmd string -> (resolved_key, expiry_monotonic)
_api_key_cache: dict[str, tuple[str, float]] = {}


def _resolve_cached_api_key(cmd: str, ttl: float | None = None) -> str:
    """Resolve an API key via a shell command, with in-memory TTL caching.

    Behaviour:
      * Cache hit (within TTL): return the cached key without running the command.
      * Miss/expired: run the command; cache and return stdout only if the command
        SUCCEEDED (exit 0) with non-empty output.
      * Empty/failed resolution: if a previous value is cached (even if expired),
        return it (stale-while-error) so a transient throttle does not break the
        tool, and back off re-resolution for _API_KEY_RETRY_BACKOFF seconds so a
        sustained outage doesn't re-run a doomed (event-loop-blocking) subprocess
        on every call; otherwise return "".

    A nonzero exit code is treated as failure: the command's stdout is never used
    or cached as a key (avoids caching error text), and a known-good stale value is
    preferred. Only successful, non-empty results are cached, so a transient failure
    is never negative-cached. The clock is read exactly once per call (before the
    subprocess), so the TTL is measured from call start.

    Note: if the credential is rotated while resolution is throttled, stale-while-
    error serves the old (now-invalid) key until re-resolution succeeds, producing
    API auth errors rather than "no API key configured". Under sustained throttle
    during a planned rotation, lower api_key_cache_ttl or restart the agent.

    Args:
        cmd: Shell command that prints the API key to stdout.
        ttl: Cache lifetime in seconds. None uses _DEFAULT_API_KEY_CACHE_TTL.
             ttl <= 0 disables hit-caching (re-resolves each call); negative values
             are clamped to 0. The stale-while-error fallback always applies.

    Returns:
        The resolved API key, a cached value, or "" if unavailable.
    """
    if ttl is None:
        ttl = _DEFAULT_API_KEY_CACHE_TTL
    if ttl < 0:
        logger.warning("api_key_cache_ttl=%s is negative; treating as 0", ttl)
        ttl = 0

    now = time.monotonic()
    cached = _api_key_cache.get(cmd)
    if cached is not None and cached[1] > now:
        return cached[0]

    key = ""
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            logger.warning(
                "api_key_cmd returned exit code %d: %s",
                proc.returncode, proc.stderr.strip(),
            )
        else:
            key = proc.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning("api_key_cmd timed out after 10 seconds")
    except Exception as e:
        logger.warning("api_key_cmd failed: %s", e)

    if key:
        _api_key_cache[cmd] = (key, now + ttl)
        return key

    # Resolution failed or returned empty. Fall back to a prior value if we have
    # one (stale-while-error) so a transient throttle doesn't break the tool.
    if cached is not None:
        if ttl > 0:
            # Back off: keep serving the stale key for a short window instead of
            # re-running the failing (event-loop-blocking) subprocess every call.
            _api_key_cache[cmd] = (cached[0], now + min(_API_KEY_RETRY_BACKOFF, ttl))
        logger.warning(
            "api_key_cmd resolution failed; serving cached key "
            "(stale-while-error, %.0fs past expiry)",
            max(0.0, now - cached[1]),
        )
        return cached[0]
    return ""


@dataclass
class ToolDef:
    """Definition of a tool available to the agent."""
    name: str
    description: str
    parameters: dict
    config: dict


@dataclass
class ToolResult:
    """Result of a tool execution."""
    content: str
    is_error: bool = False


class ToolError(Exception):
    """Raised for tool configuration/discovery errors."""
    pass


# Built-in tool definitions
# Each tool has: description, parameters (JSON Schema), and default config

BUILTIN_TOOLS: dict[str, dict[str, Any]] = {
    "shell": {
        "description": (
            "Run a shell command and return stdout (success) or stderr (failure). "
            "IMPORTANT: prefer bounded commands — pipe through head/tail/grep rather than "
            "dumping unlimited output; large output is truncated. "
            "NEVER run interactive commands (ssh, vim, python REPL, anything requiring stdin input) "
            "— they will hang until timeout. "
            "NEVER use for file reads/edits when file_read/file_edit are available; "
            "those tools are safer and register reads in the session. "
            "Timeout kills the process; set timeout explicitly for long-running tasks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute. Pipe through head/tail/grep to bound output."
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command (optional; defaults to workspace root)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (optional; uses default if not specified — set explicitly for slow commands)"
                },
                "env": {
                    "type": "object",
                    "description": "Extra environment variables to set for this command (optional)",
                    "additionalProperties": {"type": "string"}
                }
            },
            "required": ["command"]
        },
        "config": {
            "default_timeout": 30,
            "max_output": 50000
        }
    },
    "file_read": {
        "description": (
            "Read a file and return its text content. "
            "IMPORTANT: reading a file registers it in the session — required before file_write can overwrite it. "
            "For large files use offset+limit to read in sections rather than loading the whole file at once. "
            "Binary files return an error; use shell for binary inspection."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative paths resolve from workspace root — use skills/foo.md, not workspace/skills/foo.md)"
                },
                "offset": {
                    "type": "integer",
                    "description": "1-indexed line number to start reading from (optional; for large files, read in sections with offset+limit)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (optional; combine with offset to page through large files)"
                }
            },
            "required": ["path"]
        },
        "config": {}
    },
    "file_write": {
        "description": (
            "Write content to a file, creating parent directories as needed. "
            "NEVER write to an existing file you have not read this session — "
            "the guard will refuse it and the file will be unchanged. "
            "Read the file first (any offset/limit counts), then write. "
            "Prefer file_edit for targeted changes to existing files; "
            "file_write is for new files or complete replacements after reading."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (relative paths resolve from workspace root — use skills/foo.md, not workspace/skills/foo.md)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        },
        "config": {
            "require_read_before_write": True
        }
    },
    "file_edit": {
        "description": (
            "Replace one exact occurrence of old_text with new_text in a file. "
            "ALWAYS read the file first — exact-match fails against content you imagine rather than what is on disk. "
            "IMPORTANT: preserve indentation and whitespace exactly in old_text; even a single space difference causes no-match. "
            "Fails if old_text appears zero times (read the file first) or more than once "
            "(add surrounding context lines to old_text to make it unique). "
            "Prefer this over file_write for targeted changes to existing files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative paths resolve from workspace root — use skills/foo.md, not workspace/skills/foo.md)"
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace the old_text with"
                }
            },
            "required": ["path", "old_text", "new_text"]
        },
        "config": {}
    },
    "web_search": {
        "description": (
            "Search the web via Brave Search and return ranked results with title, URL, and snippet. "
            "IMPORTANT: prefer many small targeted searches over one broad query — "
            "narrow queries return more precise results. "
            "Use web_fetch to retrieve full content from a result URL. "
            "NOT for workspace/memory lookups — use memory_search for prior session knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)"
                }
            },
            "required": ["query"]
        },
        "config": {
            "api_key": "",
            "endpoint": ""
        }
    },
    "web_fetch": {
        "description": (
            "Fetch a URL and return its readable text content (HTML converted to plain text). "
            "Use max_chars to limit response size for large pages; if content is truncated, "
            "increase max_chars or fetch a more specific anchor URL. "
            "NOT for local files — use file_read instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "HTTP or HTTPS URL to fetch"
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (optional; reduce for large pages, increase if content is cut off)"
                }
            },
            "required": ["url"]
        },
        "config": {}
    },
    "subagent": {
        "description": (
            "Delegate a focused, bounded task to an isolated sub-agent. "
            "The sub-agent runs a full multi-turn tool loop and inherit the parent's enabled tools "
            "(minus subagent itself, preventing recursion). "
            "IMPORTANT: use for parallelisable or self-contained work that would consume "
            "many of the parent's iterations; do not delegate for simple single-tool calls. "
            "NEVER assume the sub-agent shares parent state — it starts with a clean context. "
            "Specify model to route to a cheaper or more capable model for the sub-task; "
            "set max_iterations conservatively to prevent runaway loops."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Complete self-contained task description for the sub-agent (include all context it needs)"
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Custom system prompt for the sub-agent (optional; defaults to helpful assistant)"
                },
                "model": {
                    "type": "string",
                    "description": "Model identifier to use (optional; defaults to parent model — override to use a cheaper or stronger model)"
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens per response turn (optional)"
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum tool-call iterations before the sub-agent stops (optional; default 100 — set lower for bounded tasks)"
                }
            },
            "required": ["task"]
        },
        "config": {
            "default_max_iterations": 100
        }
    },
    "memory_search": {
        "description": (
            "Search workspace memory files using hybrid semantic + keyword search. "
            "Returns ranked snippets with file paths and line numbers. "
            "IMPORTANT: search BEFORE asserting anything about prior work, decisions, dates, "
            "people, preferences, or todos — do not rely on recall alone. "
            "Run multiple targeted queries rather than one broad search. "
            "Use file_read to expand context around a returned snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 10)"
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum relevance score threshold 0-1 (default: 0.1)"
                }
            },
            "required": ["query"]
        },
        "config": {
            "embedding_model": "/opt/openalph/models/nomic-embed-text-v1.5.Q8_0.gguf",
            "embedding_base_url": "http://localhost:11434",
            "vector_weight": 0.7,
            "text_weight": 0.3,
            "mmr_enabled": True,
            "mmr_lambda": 0.7,
            "temporal_decay_enabled": True,
            "temporal_decay_half_life_days": 30,
            "extra_paths": []
        }
    },
    "todo_write": {
        "description": (
            "Maintain a session-scoped task list for multi-step work. "
            "Use for tasks with more than 3 steps; track exactly one in_progress item at a time. "
            "Mark items completed ONLY when fully done — if blocked, keep in_progress and add a new item. "
            "WHEN NOT TO USE: single-step tasks, trivial commands, or conversational replies. "
            "IMPORTANT: This list dies with the session — promote anything durable "
            "(blocked, deferred, or newly-discovered work) to your issue tracker before session end. "
            "Replaces the entire list on every call (full-array replacement); empty array clears."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Complete replacement list of todo items (empty array clears all)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Task description (required, must be non-empty)"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Task status: pending, in_progress, or completed"
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Optional form or context identifier for the active task"
                            }
                        },
                        "required": ["content", "status"]
                    }
                }
            },
            "required": ["todos"]
        },
        "config": {}
    },
    "context_status": {
        "description": (
            "Return current agent self-monitoring data as JSON: context window usage, "
            "session age, model info, token stats, and heartbeat state. "
            "Use before delegating or when approaching context limits to inform handoff decisions. "
            "No parameters required — room_id is injected by the framework."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "room_id": {
                    "type": "string",
                    "description": "Room ID (injected by framework, do not set manually)"
                }
            },
            "required": []
        },
        "config": {}
    },
    "send_media": {
        "description": (
            "Upload and send a file to the current Matrix room. "
            "Supports audio, images, video, and generic files. "
            "IMPORTANT: the file must already exist in your workspace before calling this. "
            "NOT for text replies — send text as normal message content, not as a file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to send (relative paths resolve from workspace root — use tmp/foo.png, not workspace/tmp/foo.png)"
                },
                "caption": {
                    "type": "string",
                    "description": "Optional caption/description for the file"
                },
            },
            "required": ["path"]
        },
        "config": {
            "max_upload_bytes": 20971520
        }
    }
}


def discover_tools(workspace: Path) -> list[ToolDef]:
    """Scan workspace/tools/ for .toml files and build ToolDef list.
    
    Each .toml file in workspace/tools/ enables a tool. The tool name is
    derived from the filename (without .toml extension). The file must
    correspond to a known tool in BUILTIN_TOOLS.
    
    Args:
        workspace: Path to the workspace directory
        
    Returns:
        List of ToolDef objects for enabled tools
        
    Raises:
        ToolError: If an unknown tool name or invalid TOML is encountered
    """
    tools_dir = workspace / "tools"
    
    # Missing or empty directory returns empty list
    if not tools_dir.exists() or not tools_dir.is_dir():
        return []
    
    tools: list[ToolDef] = []
    
    for toml_file in tools_dir.iterdir():
        # Only process .toml files
        if not toml_file.is_file() or not toml_file.suffix == ".toml":
            continue
        
        tool_name = toml_file.stem
        
        # Check if tool is known
        if tool_name not in BUILTIN_TOOLS:
            raise ToolError(f"Unknown tool: {tool_name}")
        
        # Parse TOML
        try:
            with open(toml_file, "rb") as f:
                toml_data = tomllib.load(f)
        except Exception as e:
            raise ToolError(f"Invalid TOML in {toml_file}: {e}")
        
        # Get builtin definition
        builtin = BUILTIN_TOOLS[tool_name]
        
        # Merge config: start with defaults, overlay TOML values
        config = builtin["config"].copy()
        if "config" in toml_data:
            config.update(toml_data["config"])
        
        tools.append(ToolDef(
            name=tool_name,
            description=builtin["description"],
            parameters=builtin["parameters"].copy(),
            config=config
        ))
    
    return tools


def tool_schemas(tools: list[ToolDef]) -> list[dict]:
    """Convert ToolDefs to API-ready schema dicts.
    
    Format: {"name": str, "description": str, "input_schema": dict}
    
    Args:
        tools: List of ToolDef objects
        
    Returns:
        List of schema dicts ready for LLM API calls
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters
        }
        for tool in tools
    ]


def wrap_tool_result(content: str, tool_name: str, tool_call_id: str) -> str:
    """Wrap tool result content in XML-style delimiter tags.

    Gives the LLM a structural signal that the content is tool output
    (data), not instructions.  Literal <system-reminder> and </system-reminder>
    tags (case-insensitive) are escaped to entity form before wrapping to prevent
    injection via tool output.  No other content is modified.

    Args:
        content: Raw tool result text (already truncated if needed)
        tool_name: Name of the tool that produced this result
        tool_call_id: Unique tool call identifier

    Returns:
        Content wrapped in ``<tool_result>`` tags with provenance attributes
    """
    import re as _re
    # Escape <system-reminder> and </system-reminder> tags (case-insensitive)
    # Only these specific tags; no other angle-bracket content is touched.
    content = _re.sub(
        r'<(/?)system-reminder>',
        lambda m: f'&lt;{m.group(1)}system-reminder&gt;',
        content,
        flags=_re.IGNORECASE,
    )
    return (
        f'<tool_result tool="{tool_name}" id="{tool_call_id}">\n'
        f"{content}\n"
        f"</tool_result>"
    )


def truncate_result(text: str, max_chars: int) -> str:
    """Truncate text to max_chars with head+tail and marker.
    
    If len(text) > max_chars: return head + '[truncated: N chars removed]' + tail.
    Head and tail each get roughly half the budget. Under limit: unchanged.
    
    Args:
        text: Text to potentially truncate
        max_chars: Maximum characters allowed
        
    Returns:
        Original text if under limit, or truncated text with marker
    """
    if len(text) <= max_chars:
        return text
    
    # Calculate removal
    removed_count = len(text) - max_chars
    
    # Budget for head and tail (leave room for marker)
    # Marker includes continuation steering so the agent knows how to recover.
    marker = f"[truncated: {removed_count} chars removed — re-run with offset/limit or a narrower command to retrieve more]"
    marker_len = len(marker)
    
    # Available space for content after accounting for marker
    content_budget = max_chars - marker_len
    
    # Handle edge case where marker itself is larger than max_chars
    if content_budget < 0:
        # Just return a truncated marker
        return marker[:max_chars]
    
    # Split remaining budget between head and tail
    head_len = content_budget // 2
    tail_len = content_budget - head_len
    
    head = text[:head_len]
    tail = text[-tail_len:] if tail_len > 0 else ""
    
    return head + marker + tail


# ---------------------------------------------------------------------------
# todo_write: per-room/per-instance in-memory state
# ---------------------------------------------------------------------------
# State is kept in a dict keyed by room_id (from callbacks["room_id"]).
# When callbacks has no room_id (e.g. sub-agent with empty callbacks dict),
# the key is the id() of the callbacks dict, giving each caller-instance its
# own isolated state — consistent with context_status/send_media precedents.
_TODO_STATE: dict[Any, list[dict]] = {}

_VALID_STATUSES = {"pending", "in_progress", "completed"}


async def _execute_todo_write(input: dict, callbacks: dict | None) -> "ToolResult":
    """Execute the todo_write tool.

    Full-array replacement semantics: every call replaces the entire list.
    Empty array clears. Validates before mutating state.

    Args:
        input: Tool input dict (must contain 'todos' key).
        callbacks: Callbacks dict used for state isolation (room_id or obj identity).

    Returns:
        ToolResult with formatted list + counts on success, or error without
        mutating state on validation failure.
    """
    todos = input.get("todos", [])

    # Validate all items before touching state
    in_progress_count = 0
    for i, item in enumerate(todos):
        content = item.get("content", "")
        status = item.get("status", "")

        if not isinstance(content, str) or not content.strip():
            return ToolResult(
                content=(
                    f"Validation error: item {i} has empty or missing content. "
                    "Each todo item must have a non-empty content string."
                ),
                is_error=True,
            )

        if status not in _VALID_STATUSES:
            return ToolResult(
                content=(
                    f"Validation error: item {i} has invalid status {status!r}. "
                    f"Valid status values are: {', '.join(sorted(_VALID_STATUSES))}."
                ),
                is_error=True,
            )

        if status == "in_progress":
            in_progress_count += 1

    if in_progress_count > 1:
        return ToolResult(
            content=(
                f"Validation error: {in_progress_count} items have status in_progress. "
                "At most 1 item may be in_progress at a time."
            ),
            is_error=True,
        )

    # Determine state key: room_id from callbacks, else id(callbacks) for isolation
    if callbacks and "room_id" in callbacks:
        state_key = callbacks["room_id"]
    elif callbacks is not None:
        state_key = id(callbacks)
    else:
        # No callbacks at all (e.g. bare execute_tool call in tests without room scoping)
        state_key = None

    # Full replacement: store validated list (defensive copy, strip unknown fields)
    clean_todos = []
    for item in todos:
        clean_item: dict = {
            "content": item["content"],
            "status": item["status"],
        }
        if "activeForm" in item:
            clean_item["activeForm"] = item["activeForm"]
        clean_todos.append(clean_item)

    _TODO_STATE[state_key] = clean_todos

    # Build result echo: formatted list + counts
    counts: dict[str, int] = {"pending": 0, "in_progress": 0, "completed": 0}
    lines = []
    status_symbols = {"pending": "○", "in_progress": "●", "completed": "✓"}
    for item in clean_todos:
        st = item["status"]
        counts[st] = counts.get(st, 0) + 1
        sym = status_symbols.get(st, "?")
        lines.append(f"  {sym} [{st}] {item['content']}")

    if not clean_todos:
        list_text = "  (empty)"
    else:
        list_text = "\n".join(lines)

    summary_parts = []
    if counts["in_progress"]:
        summary_parts.append(f"{counts['in_progress']} in progress")
    if counts["pending"]:
        summary_parts.append(f"{counts['pending']} pending")
    if counts["completed"]:
        summary_parts.append(f"{counts['completed']} completed")
    if not summary_parts:
        summary_parts = ["0 items"]

    summary = " · ".join(summary_parts)
    result_text = f"Todo list updated ({summary}):\n{list_text}"

    return ToolResult(content=result_text, is_error=False)


async def execute_tool(
    name: str,
    input: dict,
    tool_config: dict,
    agent_config: Any,
    tools: list[ToolDef] | None = None,
    callbacks: dict | None = None,
) -> ToolResult:
    """Dispatch to the named tool executor.
    
    Routes input + config to the appropriate tool implementation.
    
    Args:
        name: Tool name to execute
        input: Tool input parameters
        tool_config: Tool-specific configuration
        agent_config: Agent configuration (for sub-agents, etc.)
        
    Returns:
        ToolResult with content and error status
        
    Note:
        This is a stub that dispatches to tool-specific modules.
        Full implementation will be in Phase 2.
    """
    # Validate tool name exists
    if name not in BUILTIN_TOOLS:
        return ToolResult(
            content=f"Unknown tool: {name}. Available tools: {', '.join(BUILTIN_TOOLS.keys())}",
            is_error=True,
        )
    
    # Validate required parameters against schema
    schema = BUILTIN_TOOLS[name]["parameters"]
    required = schema.get("required", [])
    missing = [p for p in required if p not in input]
    if missing:
        return ToolResult(
            content=f"Missing required parameter(s): {', '.join(missing)}. "
                    f"Expected: {', '.join(required)}. Got: {', '.join(input.keys())}",
            is_error=True,
        )
    
    # Never mutate caller's input dict
    input = dict(input)

    # Resolve relative paths for file tools against workspace
    if name in ("file_read", "file_write", "file_edit", "send_media") and "path" in input:
        file_path = input["path"]
        if not os.path.isabs(file_path) and hasattr(agent_config, "workspace"):
            input["path"] = str(agent_config.workspace / file_path)

    # Normalize path to resolved form for registry keys (symlinks, .., relative spellings)
    # so that read via relative and write via absolute always hit the same registry entry.
    _resolved_path: str | None = None
    if name in ("file_read", "file_write", "file_edit") and "path" in input:
        try:
            _resolved_path = str(Path(input["path"]).resolve())
        except Exception:
            _resolved_path = input["path"]

    if name == "shell":
        from .shell import run_shell
        # Default cwd to workspace so relative paths match file tools
        shell_cwd = input.get("cwd")
        if shell_cwd is None and hasattr(agent_config, "workspace"):
            shell_cwd = str(agent_config.workspace)
        result = await run_shell(
            command=input["command"],
            cwd=shell_cwd,
            env=input.get("env"),
            timeout=input.get("timeout") or tool_config.get("default_timeout", 30),
            max_output=tool_config.get("max_output", 50000),
        )
    elif name == "file_read":
        from .file import read_file
        result = await read_file(
            path=input["path"],
            offset=input.get("offset"),
            limit=input.get("limit"),
        )
        # On successful read, record path+mtime in read_registry
        if not result.is_error and _resolved_path is not None and callbacks is not None:
            _registry = callbacks.get("read_registry")
            if _registry is not None:
                try:
                    import os as _os
                    _mtime = _os.stat(_resolved_path).st_mtime
                    _registry[_resolved_path] = _mtime
                except Exception:
                    pass  # best effort; guard will conservatively deny if stat fails
    elif name == "file_write":
        from .file import write_file
        import os as _os_fw

        # --- Read-before-write guard ---
        _fw_path = input["path"]
        _guard_enabled = tool_config.get("require_read_before_write", True)
        if _guard_enabled and _resolved_path is not None and _os_fw.path.exists(_fw_path):
            # File exists — apply guard
            _registry = (callbacks or {}).get("read_registry") if callbacks else None
            if _registry is None:
                # No registry provided (guard cannot be satisfied) — block to be safe
                # unless guard is explicitly disabled via config
                return ToolResult(
                    content=(
                        f"File exists and was not read this session: {_fw_path}. "
                        "Read it first, or use file_edit for targeted changes."
                    ),
                    is_error=True,
                )
            if _resolved_path not in _registry:
                # File exists but was not read — refuse
                return ToolResult(
                    content=(
                        f"File exists and was not read this session: {_fw_path}. "
                        "Read it first, or use file_edit for targeted changes."
                    ),
                    is_error=True,
                )
            # File was read — check if mtime has changed since read
            try:
                _current_mtime = _os_fw.stat(_fw_path).st_mtime
                _recorded_mtime = _registry[_resolved_path]
                if _current_mtime > _recorded_mtime:
                    return ToolResult(
                        content=(
                            f"File changed on disk since you last read it: {_fw_path}. "
                            "Re-read before overwriting."
                        ),
                        is_error=True,
                    )
            except Exception:
                pass  # stat failed; proceed (new file or disappeared — let write handle it)

        result = await write_file(
            path=input["path"],
            content=input["content"],
        )
        # On successful write, update read_registry with new mtime
        if not result.is_error and _resolved_path is not None and callbacks is not None:
            _registry = callbacks.get("read_registry")
            if _registry is not None:
                try:
                    _mtime = _os_fw.stat(_fw_path).st_mtime
                    _registry[_resolved_path] = _mtime
                except Exception:
                    pass
    elif name == "file_edit":
        from .file import edit_file
        result = await edit_file(
            path=input["path"],
            old_text=input["old_text"],
            new_text=input["new_text"],
        )
        # On successful edit, update read_registry with new mtime (keeps registry fresh)
        if not result.is_error and _resolved_path is not None and callbacks is not None:
            _registry = callbacks.get("read_registry")
            if _registry is not None:
                try:
                    import os as _os_fe
                    _mtime = _os_fe.stat(_resolved_path).st_mtime
                    _registry[_resolved_path] = _mtime
                except Exception:
                    pass
    elif name == "web_search":
        from .web import web_search
        # Resolve api_key: direct value, or via api_key_cmd with TTL caching.
        # Caching avoids a per-call hit to the secret store (e.g. 1Password),
        # which under burst load exhausts the service-account rate limit and
        # makes web_search fail with "no API key configured".
        api_key = tool_config.get("api_key", "")
        if not api_key and "api_key_cmd" in tool_config:
            api_key = _resolve_cached_api_key(
                tool_config["api_key_cmd"],
                tool_config.get("api_key_cache_ttl"),
            )
        result = await web_search(
            query=input["query"],
            count=input.get("count", 5),
            api_key=api_key,
            endpoint=tool_config.get("endpoint", ""),
        )
    elif name == "web_fetch":
        from .web import web_fetch
        result = await web_fetch(
            url=input["url"],
            max_chars=input.get("max_chars"),
        )
    elif name == "subagent":
        from .subagent import run_subagent
        result = await run_subagent(
            task=input["task"],
            config=agent_config,
            tools=tools,
            system_prompt=input.get("system_prompt"),
            model=input.get("model"),
            max_tokens=input.get("max_tokens"),
            max_iterations=input.get("max_iterations"),
            call_id=callbacks.get("call_id") if callbacks else None,
        )
    elif name == "memory_search":
        from .memory_search import run_memory_search
        result = await run_memory_search(
            query=input["query"],
            config=tool_config,
            workspace=agent_config.workspace if hasattr(agent_config, "workspace") else Path("."),
            max_results=input.get("max_results", 10),
            min_score=input.get("min_score", 0.1),
        )
    elif name == "context_status":
        import json as _json
        cb = callbacks.get("context_status") if callbacks else None
        if not cb:
            return ToolResult(
                content="context_status requires a callback from the matrix layer.",
                is_error=True,
            )
        try:
            status_data = await cb(input.get("room_id"))
            result = ToolResult(content=_json.dumps(status_data, indent=2))
        except Exception as e:
            result = ToolResult(content=f"Failed to get context status: {e}", is_error=True)
    elif name == "send_media":
        from .media import send_media
        result = await send_media(
            path=input["path"],
            caption=input.get("caption"),
            max_upload_bytes=tool_config.get("max_upload_bytes", 20_971_520),
            upload_callback=callbacks.get("send_media") if callbacks else None,
        )
    elif name == "todo_write":
        result = await _execute_todo_write(input, callbacks)
    else:
        return ToolResult(
            content=f"Unknown tool: {name}",
            is_error=True,
        )

    # Redact credentials from tool output
    from .security import redact_credentials as _redact_credentials
    redacted_content, redaction_events = _redact_credentials(result.content)
    if redaction_events:
        result = ToolResult(content=redacted_content, is_error=result.is_error)
        for event in redaction_events:
            logger.warning(
                "Credential redacted in %s output: %s (%d chars)",
                name, event.pattern_name, event.char_count,
            )
        if callbacks and "on_redaction" in callbacks:
            try:
                await callbacks["on_redaction"](name, redaction_events)
            except Exception as e:
                logger.warning("on_redaction callback failed: %s", e)

    return result
