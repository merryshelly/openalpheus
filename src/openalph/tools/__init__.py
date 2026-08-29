"""OpenAlph tools package.

Tool registry, discovery, schema generation, and result truncation.
"""

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

logger = logging.getLogger(__name__)

# NOTE: there is deliberately no subagent progress-ping cadence constant here.
# Parent-turn liveness during a sub run comes from REAL milestones emitted by
# run_subagent (see tools/subagent.py), never from a blind elapsed-time
# heartbeat — which would mask the provider wedge the turn stall watchdog exists
# to catch (RCA 2026-08-03, round-2 review).


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
            "For content or filename search across files, prefer the grep/glob tools "
            "(bounded, structured output) over shell grep/find pipelines. "
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
            "Binary files return an error; use shell for binary inspection. "
            "To locate files or content first, use glob/grep."
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
            "Written files are syntax-validated for known code types "
            "(a clean file that would become broken is rejected, unchanged; "
            "new files must be born clean). "
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
            "(add surrounding context lines to old_text to make it unique) — unless replace_all is set. "
            "Set replace_all=true to replace every occurrence in one call (e.g. renaming a symbol "
            "or string across the whole file) instead of disambiguating a single match. "
            "For multiple distinct edits to the same file in one call, prefer file_patch "
            "(multi-hunk, atomic). "
            "The resulting file is syntax-validated for known code types; an edit that would "
            "turn a clean file broken is rejected, unchanged. "
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
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence instead of requiring exactly one (optional; default false) — use for renaming a symbol/string across the file"
                }
            },
            "required": ["path", "old_text", "new_text"]
        },
        "config": {}
    },
    "file_patch": {
        "description": (
            "Apply one or more SEARCH/REPLACE hunks to an existing file in a single, atomic call. "
            "Fence syntax (each hunk):\n"
            "<<<<<<< SEARCH\n"
            "exact existing lines\n"
            "=======\n"
            "replacement lines\n"
            ">>>>>>> REPLACE\n"
            "Worked example — change 'foo = 1' to 'foo = 2':\n"
            "<<<<<<< SEARCH\n"
            "foo = 1\n"
            "=======\n"
            "foo = 2\n"
            ">>>>>>> REPLACE\n"
            "IMPORTANT: read the file first — each hunk's SEARCH must match the current file "
            "content EXACTLY (whitespace and indentation included) exactly once. "
            "Prefer ONE file_patch call with multiple hunks over many separate file_edit calls "
            "on the same file. Hunks apply in order against the file as edited by earlier hunks "
            "in the same call; ALL-OR-NOTHING — if any hunk fails to match, no hunk is written "
            "and the file is left byte-identical. "
            "The resulting file is syntax-validated for known code types; a patch that would "
            "turn a clean file broken is rejected, unchanged. "
            "When NOT to use: creating a new file (use file_write); a single trivial replacement "
            "in a file (use file_edit)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to patch (relative paths resolve from workspace root — use skills/foo.md, not workspace/skills/foo.md); file must already exist"
                },
                "patch": {
                    "type": "string",
                    "description": "One or more SEARCH/REPLACE fenced hunks (text outside blocks is ignored)"
                }
            },
            "required": ["path", "patch"]
        },
        "config": {}
    },
    "grep": {
        "description": (
            "Search file contents for a pattern across the workspace (or a "
            "narrower path/glob) using Python re syntax, matched per line. "
            "IMPORTANT: prefer many small targeted searches (narrow path/glob, "
            "specific pattern) over one broad search across the whole tree. "
            "Default output_mode='files_with_matches' returns matching filenames "
            "only; use output_mode='content' to see the actual matching lines, "
            "or output_mode='count' for per-file/total match counts. "
            "NOT for memory or prior-session lookups — use memory_search for "
            "recalling past work, decisions, or preferences; grep only sees "
            "files that exist on disk right now."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python re regular expression, matched against each line individually"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search (optional; default: workspace root)"
                },
                "glob": {
                    "type": "string",
                    "description": "Filename filter, fnmatch syntax (e.g. \"*.py\") narrowing which files are searched (optional)"
                },
                "output_mode": {
                    "type": "string",
                    "description": "\"files_with_matches\" (default, filenames only) | \"content\" (matching lines as 'path:lineno: line') | \"count\" (per-file + total match counts)"
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Cap on returned entries (optional; default 50 for files_with_matches/count, 100 for content)"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Match case-insensitively (optional; default false)"
                }
            },
            "required": ["pattern"]
        },
        "config": {
            "max_scan_files": 10000,
            "max_file_bytes": 5242880
        }
    },
    "glob": {
        "description": (
            "Find files and directories by name pattern (pathlib glob syntax, "
            "including \"**\" for recursion) under the workspace or a given path. "
            "pattern=\"*\" lists a directory's entries — this is the directory-"
            "listing tool; results are sorted newest-first (by mtime) and "
            "directories are rendered with a trailing \"/\". "
            "NOT for searching file contents — use grep for matching lines "
            "within files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (pathlib syntax, e.g. \"*.py\", \"**/*.md\", \"*\" for a directory listing)"
                },
                "path": {
                    "type": "string",
                    "description": "Directory root to search (optional; default: workspace root)"
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Cap on returned entries (optional; default 100)"
                }
            },
            "required": ["pattern"]
        },
        "config": {
            "max_scan_files": 10000,
            "max_file_bytes": 5242880
        }
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
            "For large pages: without offset, max_chars is a head+tail cap — the middle is "
            "unreachable. To read a specific region (e.g. the middle), pass offset (0-based "
            "char position) with max_chars as the window size; the result carries a "
            "navigation marker (total size, continue offset). Or fetch a more specific "
            "anchor URL. NOT for local files — use file_read instead."
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
                    "description": (
                        "Without offset: head+tail truncation cap. With offset: window "
                        "size (chars to read starting at offset; default 50000)."
                    )
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "0-based character offset into the extracted text (window mode). "
                        "Reads a specific region of a large page, e.g. the middle. The "
                        "result carries a navigation marker with total size and the "
                        "continue offset."
                    )
                }
            },
            "required": ["url"]
        },
        "config": {}
    },
    "web_fetch_js": {
        "description": (
            "Fetch a JavaScript-rendered web page via Tabstack's cloud browser and return clean "
            "content. Use when web_fetch returns an empty shell, a \"please enable JavaScript\" "
            "notice, or obviously incomplete text from a JS-heavy site (SPAs, dashboards, "
            "infinite-scroll, dynamically-loaded data). "
            "IMPORTANT: try web_fetch FIRST — it's faster and free. Reach for web_fetch_js only "
            "when web_fetch's result is clearly unrendered; this runs a real browser (slower, up "
            "to ~60s at effort=max). "
            "IMPORTANT: provide a schema (JSON Schema) to get structured JSON back instead of "
            "markdown — ideal when you need specific fields (prices, listings, table rows). "
            "Omit it for clean readable markdown. "
            "NEVER send credentials, cookies, or authenticated URLs through this tool — Tabstack "
            "fetches the page from its own cloud and cannot use your session. For logged-in "
            "flows, use local Playwright (see browser-automation skill). "
            "For rendered markdown longer than the context window: pass offset (0-based char "
            "position) with max_chars as the window size — markdown mode only. "
            "When NOT to use: static/simple pages web_fetch already handles; multi-step "
            "interaction (clicking, form flows) or multi-page research — those stay in the "
            "browser-automation skill via the tabstack CLI (/automate, /research)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The publicly accessible URL to fetch and render."
                },
                "schema": {
                    "type": "object",
                    "description": (
                        "Optional JSON Schema. If provided, returns structured JSON extraction "
                        "matching the schema instead of markdown. Use for specific fields "
                        "(prices, listings, table rows)."
                    )
                },
                "effort": {
                    "type": "string",
                    "enum": ["min", "standard", "max"],
                    "description": (
                        "Browser rendering effort. Default max (full render). Drop to "
                        "standard/min for speed on lighter pages."
                    )
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "Without offset: truncate returned markdown to this many chars "
                        "(head+tail), like web_fetch. With offset: window size (default "
                        "50000). Ignored in schema mode."
                    )
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "0-based char offset into the rendered markdown (window mode). "
                        "Markdown mode only — errors when a schema is given."
                    )
                },
                "nocache": {
                    "type": "boolean",
                    "description": "Bypass Tabstack's cache for real-time data. Default false."
                }
            },
            "required": ["url"]
        },
        "config": {
            "api_key": "",
            "base_url": "https://api.tabstack.ai/v1",
            "default_effort": "max",
            "timeout": 90
        }
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
    "advisor": {
        "description": (
            "Consult a second, typically stronger model for strategic guidance mid-task. "
            "The advisor sees your full transcript (system prompt, conversation, and tool "
            "activity so far) and returns advice as text. "
            "IMPORTANT: consult before committing to an approach on a non-obvious design "
            "decision, before your first substantive write on a multi-step task, and before "
            "declaring complex work done. "
            "The advisor has no tools and cannot act — its advice is guidance to weigh "
            "against your own context; you remain responsible for the outcome. "
            "Use focus to ask a specific question. "
            "NOT for: trivial or single-step tasks, factual lookups (use web_search), or "
            "delegating work (use subagent)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "Specific question or area to direct the advisor's attention (optional)"
                },
                "model": {
                    "type": "string",
                    "description": "Override the configured advisor model (alias or provider/model; optional)"
                }
            },
            "required": []
        },
        "config": {
            "model": "",
            "max_uses": 10,
            "max_tokens": 8192,
            "thinking": "medium",
            "include_system_prompt": True,
            "transcript_max_chars": 0,
            "cache_ttl": "5m",
            "timeout": 300
        }
    },
    "memory_search": {
        "description": (
            "Search workspace memory files using hybrid semantic + keyword search. "
            "Returns ranked results with file paths and line numbers. "
            "Results for memory atoms (files whose parent directory is named 'atoms') "
            "return complete content inline; other files return short truncated snippets "
            "to expand with file_read. "
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
            "atom_full_max_chars": 2500,
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
    "heartbeat": {
        "description": (
            "Start, check, or stop this session's per-room heartbeat — an "
            "automated wake-up timer that fires a turn in this room on an "
            "interval. Use it to supervise long-running background work "
            "(builds, deploys, watch-loops) instead of sleep hacks: start a "
            "heartbeat, keep working or let the room idle, then act on the "
            "next fire. "
            "IMPORTANT: Heartbeats persist across process restarts and burn "
            "tokens on every fire — ALWAYS stop the heartbeat when the "
            "supervised job completes. Minimum interval is 5 minutes. "
            "Re-issuing start replaces the existing timer for this room. "
            "Cannot start while an umbral timer is active in the room. "
            "IMPORTANT: interval mode only (\"15m\", \"1h\", or seconds as an "
            "integer) — cron schedules remain operator slash-command territory "
            "(/heartbeat schedule), as do recurring standing routines. "
            "NEVER use this for umbrals (context rotation) — that timer is "
            "operator-only by design. Not available to sub-agents or on the "
            "headless CLI. "
            "status reports this room's current entry (interval, next fire, "
            "directive). stop is idempotent-cheap but errors when nothing is "
            "active."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: \"start\", \"stop\", or \"status\""
                },
                "interval": {
                    "description": "Interval for start: a string like \"15m\"/\"1h\"/\"300s\" (case-insensitive) or a positive integer (seconds directly). Minimum 5 minutes."
                },
                "directive": {
                    "type": "string",
                    "description": "Optional free-text directive (start only), passed verbatim."
                }
            },
            "required": ["action"]
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
    },
    "view_image": {
        "description": (
            "View an image that already exists in your workspace by attaching it to "
            "your own context. The image is staged now and provided to you as a user "
            "message before the NEXT model call (never inside this tool result). "
            "Supported types: JPEG, PNG, GIF, WebP. "
            "WHEN TO USE: you need to actually see a workspace image (a photo, plot, "
            "screenshot, diagram) to reason about it. "
            "WHEN NOT TO USE: to SEND an image to the operator use send_media instead; "
            "for non-image files use shell/file_read; if the image is not on disk yet, "
            "download it via shell first. "
            "COST: images consume context — roughly 1 token per 750 bytes — and large "
            "images may exceed the per-image cap (default 5 MB, max_bytes); downscale "
            "oversized images via shell first. "
            "If the active model has no vision capability the tool returns a clear "
            "error naming the model and the /model escape — switch models and retry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path to the image (e.g. shots/a.jpg). Absolute paths, '..', and shell/framed-control characters are rejected."
                },
            },
            "required": ["path"]
        },
        "config": {
            "max_bytes": 5_242_880
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


def escape_system_reminder_tags(text: str) -> str:
    """Escape <system-reminder> tags in text to prevent spoofing (R2-A/R2-9).

    Catches optional surrounding whitespace, attributes, newlines, and
    mixed-case variants.  Normalizes to entity form: &lt;system-reminder&gt;
    or &lt;/system-reminder&gt;.  Idempotent by construction — entity-escaped
    tags (&lt;…&gt;) will not re-match the angle-bracket regex.

    Used by wrap_tool_result (tool output security, §8) and by both
    user-content escaping paths: agent.handle_input live-append and
    session.build_context replay (R2-A).
    """
    import re as _re
    return _re.sub(
        r'<\s*(/?)\s*system-reminder\b[^>]*>',
        lambda m: f'&lt;{m.group(1)}system-reminder&gt;',
        text,
        flags=_re.IGNORECASE,
    )


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
    # R2-9: Use shared helper for broadened regex (whitespace/attributes/case)
    content = escape_system_reminder_tags(content)
    return (
        f'<tool_result tool="{tool_name}" id="{tool_call_id}">\n'
        f"{content}\n"
        f"</tool_result>"
    )


DEFAULT_TRUNCATION_MARKER_TEMPLATE = (
    "[truncated: {n} chars removed — re-run with a smaller "
    "limit or a narrower query/command to retrieve more]"
)


def truncate_result(text: str, max_chars: int,
                    *, marker_template: str | None = None) -> str:
    """Truncate text to max_chars with head+tail and marker.
    
    If len(text) > max_chars: return head + '[truncated: N chars removed]' + tail.
    Head and tail each get roughly half the budget. Under limit: unchanged.
    
    Args:
        text: Text to potentially truncate
        max_chars: Maximum characters allowed
        marker_template: Optional override for the elision marker; must
            contain ``{n}``, replaced with the omitted-character count.
            Defaults to the agent-facing steering string; operator-facing
            renders (e.g. Matrix notices) pass their own (kdsn.247.2).

    Returns:
        Original text if under limit, or truncated text with marker
    """
    if len(text) <= max_chars:
        return text

    # N must count the actual gap between the retained head and tail, not
    # merely the amount by which the original value exceeded max_chars. The
    # marker itself consumes part of the budget, and its digit width can in
    # turn change its own length, so solve the small fixed point explicitly.
    overflow = len(text) - max_chars
    template = marker_template or DEFAULT_TRUNCATION_MARKER_TEMPLATE
    removed_count = overflow
    for _ in range(10):
        marker = template.format(n=removed_count)
        next_removed_count = overflow + len(marker)
        if next_removed_count == removed_count:
            break
        removed_count = next_removed_count

    marker = template.format(n=removed_count)
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

    # R2-C: validate todos is a list of dicts before field access.
    # Violations → is_error=True, state unchanged, no exception raised.
    if not isinstance(todos, list):
        return ToolResult(
            content=(
                "Validation error: todos must be a list (array) of todo items, "
                f"got {type(todos).__name__}. Each item must be a dict with "
                "'content' (str) and 'status' (str) fields."
            ),
            is_error=True,
        )
    for i, item in enumerate(todos):
        if not isinstance(item, dict):
            return ToolResult(
                content=(
                    f"Validation error: item {i} must be a dict, "
                    f"got {type(item).__name__}. Each todo item must be a dict with "
                    "'content' (str) and 'status' (str) fields."
                ),
                is_error=True,
            )

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
    # BUG-14: every sub-agent invocation passes the CONSTANT room_id "__sub__",
    # so keying on room_id alone made two concurrently-running sub-agents share
    # (and overwrite) one todo list. When the room is the sub-agent sentinel and
    # a per-call id is present, key on that id so each sub-agent's todos stay
    # isolated; top-level rooms are unchanged.
    if callbacks and callbacks.get("room_id") == "__sub__" and callbacks.get("call_id"):
        state_key = ("__sub__", callbacks["call_id"])
    elif callbacks and "room_id" in callbacks:
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


async def _execute_heartbeat_tool(input: dict, callbacks: dict | None) -> "ToolResult":
    """Execute the built-in heartbeat tool (kdsn.290).

    Lets an agent session manage its own per-room heartbeat timer (start /
    stop / status). Room scoping ALWAYS comes from ``callbacks["room_id"]``
    — never from any input key (there is no cross-room surface in v1).

    Guards (all return ``ToolResult(is_error=True)`` without mutating state,
    never raising):
      - missing callbacks / None heartbeat manager → transport-unavailable
        steering (headless/CLI has no timer manager).
      - sub-agent sentinel room ("__sub__") → refused (no timer semantics).
      - bad input shapes (non-str action, non-str/non-int interval, non-str
        directive) → steering.
      - manager exceptions → caught and sanitized (type name only).

    Reuses ``openalph.heartbeat.parse_interval`` (returns None, never raises)
    and ``HeartbeatManager._floor`` (300) rather than re-hardcoding literals.
    Room scoping reads ``callbacks["room_id"]`` via .get() — a missing key
    yields steering, never a KeyError (R4).
    """
    from openalph.heartbeat import (
        HeartbeatManager,
        format_interval,
        parse_interval,
    )

    # --- Transport / scope guards -----------------------------------------
    # R4: never bare-subscript the room id. A malformed caller (or a future
    # callback wiring that omits the key) must get steering, not a KeyError
    # escaping the tool and killing the turn.
    room_id = callbacks.get("room_id") if callbacks else None
    if room_id is None:
        return ToolResult(
            content=(
                "Heartbeat tool unavailable: no room-scoped session context "
                "(callbacks carry no \"room_id\"); heartbeat timers are "
                "per-room and cannot be managed without a room context."
            ),
            is_error=True,
        )

    # R3: the sub-agent sentinel refusal is checked BEFORE the transport
    # guard. Sub-agent tool-call callbacks carry no "heartbeat" key, so under
    # the previous order they hit the generic transport-unavailable message
    # instead of the sub-specific refusal the spec pins.
    if room_id == "__sub__":
        return ToolResult(
            content=(
                "Sub-agents cannot manage heartbeats — heartbeat timers are "
                "per-room and only the owning session may start/stop them."
            ),
            is_error=True,
        )

    # Headless/CLI wires "heartbeat"=None (and possibly callbacks=None) — all
    # actions fail clean with steering toward the /heartbeat slash command.
    hb = callbacks.get("heartbeat") if callbacks else None
    if hb is None:
        return ToolResult(
            content=(
                "Heartbeats are managed in Matrix rooms via the "
                "`/heartbeat` slash command — the heartbeat tool is "
                "unavailable interface (no timer manager wired)."
            ),
            is_error=True,
        )

    # --- Action normalization ---------------------------------------------
    action_raw = input.get("action")
    # Lenient: strip + lowercase a STRING action. Non-str (incl. bool) is a
    # bad shape → reject without coercion (never raise).
    if not isinstance(action_raw, str):
        return ToolResult(
            content=(
                "Invalid action: action must be a string. "
                "Valid actions: start, stop, status."
            ),
            is_error=True,
        )
    action = action_raw.strip().lower()

    um = callbacks.get("umbral")
    send_notice = callbacks.get("send_notice")

    # =====================================================================
    # start
    # =====================================================================
    if action == "start":
        # R1 (tool policy): start issued from within the room's OWN fired
        # heartbeat turn is refused. Re-issue=start replaces the timer, and
        # replacement cancels the current loop task — the ancestor of the
        # gather child this call runs in — recreating the cyclic-cancel the
        # manager-side R1 fix prevents for stop. A deferred-replacement
        # mechanism ("apply at end of the turn") is out of scope for v1, so
        # the contract is: interval changes must come from a LATER turn.
        # The refusal keys on in_own_loop ALONE (re-audit F5): a same-batch
        # [stop, start] pair inside one fired turn runs as parallel gather
        # siblings, and the stop sibling removes the room's entry BEFORE
        # this probe can read it — any additional "entry exists" AND-term
        # would silently skip the refusal and arm a NEW timer mid-turn,
        # violating the steering's "the current timer keeps running"
        # promise. Keying exclusively on the contextvar closes the race:
        # the mark survives the sibling's bookkeeping removal for the
        # whole turn.
        # Checked early, before interval validation: fired turns only enter
        # here with a complete tool-call input (action is required by the
        # schema), and a fixed refusal beats a confusing per-field error
        # sample for a call that could never have succeeded.
        _in_own_turn = False
        # getattr defends against contract drift (a mock or partial manager
        # lacking the helper); a REAL manager always exposes both. Any error
        # reading state here MUST NOT crash the turn (never-raise rule) —
        # fall through to the ordinary start path, which has its own
        # exception hygiene around hb.start.
        try:
            _has = getattr(hb, "in_own_loop", None)
            if _has is not None:
                _in_own_turn = bool(_has(room_id))
        except Exception:
            logger.warning(
                "in_own_loop probe failed for %s; treating as not-own-turn",
                room_id,
                exc_info=True,
            )
        if _in_own_turn:
            return ToolResult(
                content=(
                    "Heartbeat is running its own turn right now — interval "
                    "changes take effect only if issued from a later turn; "
                    "the current timer keeps running."
                ),
                is_error=True,
            )

        # interval required for start.
        interval_raw = input.get("interval")
        if interval_raw is None:
            return ToolResult(
                content=(
                    "Missing required parameter 'interval' for start. "
                    "Use a string like \"15m\", \"1h\", or \"300s\" "
                    "(case-insensitive), or a positive integer (seconds)."
                ),
                is_error=True,
            )

        # Resolve interval to seconds:
        #  - str → parse_interval (returns None on invalid, never raises)
        #  - positive JSON integer → seconds directly (bool is an int subclass
        #    and MUST be rejected; floats/dicts/etc. are bad shapes)
        #  - never re-parse a parsed value.
        if isinstance(interval_raw, bool):
            return ToolResult(
                content=(
                    "Invalid interval: a boolean is not a valid interval. "
                    "Use a string like \"15m\" or a positive integer (seconds)."
                ),
                is_error=True,
            )
        if isinstance(interval_raw, int):
            seconds = interval_raw
        elif isinstance(interval_raw, str):
            seconds = parse_interval(interval_raw)
            if seconds is None:
                return ToolResult(
                    content=(
                        f"Invalid interval: {interval_raw!r}. "
                        "Use e.g. `15m`, `1h`, `6h`, or a positive integer."
                    ),
                    is_error=True,
                )
        else:
            # float, dict, list, etc. — bad shape.
            return ToolResult(
                content=(
                    "Invalid interval: must be a string (e.g. \"15m\") or a "
                    "positive integer (seconds)."
                ),
                is_error=True,
            )

        # Floor (canonical 5-minute floor, slash-parity wording).
        if seconds < HeartbeatManager._floor:
            return ToolResult(
                content="Minimum interval is 5m.",
                is_error=True,
            )

        # directive (optional, start only) — must be a string if present.
        directive = input.get("directive")
        if directive is not None and not isinstance(directive, str):
            return ToolResult(
                content=(
                    "Invalid directive: must be a string. Omit it if you have "
                    "no directive to attach."
                ),
                is_error=True,
            )

        # Umbral mutual exclusion — checked BEFORE start is called.
        # R5 fail-closed: if umbral state can't be VERIFIED (is_active
        # raises), refuse the start and name the failure — starting with an
        # unknown umbral state could violate the exclusion, which is worse
        # than a spurious refusal.
        if um is not None:
            try:
                um_active = bool(um.is_active(room_id))
            except Exception as e:
                logger.warning(
                    "um.is_active raised for %s; failing closed", room_id,
                    exc_info=True,
                )
                return ToolResult(
                    content=(
                        "Cannot verify umbral state "
                        f"({type(e).__name__}) — refusing start; "
                        "check /umbral status and retry."
                    ),
                    is_error=True,
                )
            if um_active:
                return ToolResult(
                    content=(
                        "Stop the umbral timer first (`/umbral stop`) — "
                        "umbral and heartbeat cannot run in the same room."
                    ),
                    is_error=True,
                )

        # Delegate to the manager (replace is the tested manager contract —
        # no refusal, no special message for an already-active timer).
        try:
            await hb.start(room_id, seconds, directive)
        except Exception as e:
            return ToolResult(
                content=f"Heartbeat start failed: {type(e).__name__}.",
                is_error=True,
            )

        human = format_interval(seconds)
        # Notice (one, room-scoped) — only if a sink is wired.
        if send_notice is not None:
            try:
                await send_notice(room_id, f"💓 Heartbeat started — every {human}")
            except Exception:
                logger.warning("heartbeat start notice failed in %s", room_id)

        return ToolResult(
            content=(
                f"Heartbeat started: every {human} in this room.\n"
                "Persists across process restarts; auto-stops on context "
                "overflow. Stop with action=\"stop\" when done."
            ),
            is_error=False,
        )

    # =====================================================================
    # stop
    # =====================================================================
    if action == "stop":
        # Read the entry BEFORE stopping so we can surface schedule-mode
        # provenance in the result (the tool is interval-only but may stop an
        # operator-started schedule entry). Mirrors the slash handler's
        # read-before-mutate ordering.
        try:
            entries = hb.status()
        except Exception:
            entries = []
        entry = next((e for e in entries if e.room_id == room_id), None)

        try:
            stopped = await hb.stop(room_id)
        except Exception as e:
            return ToolResult(
                content=f"Heartbeat stop failed: {type(e).__name__}.",
                is_error=True,
            )
        if not stopped:
            return ToolResult(
                content="No heartbeat active in this room.",
                is_error=True,
            )

        suffix = ""
        if entry is not None and getattr(entry, "schedule", None):
            suffix = f' (was schedule "{entry.schedule}")'

        if send_notice is not None:
            try:
                await send_notice(room_id, "💓 Heartbeat stopped")
            except Exception:
                logger.warning("heartbeat stop notice failed in %s", room_id)

        return ToolResult(
            content=f"Heartbeat stopped.{suffix}",
            is_error=False,
        )

    # =====================================================================
    # status (a read — fires NO notice, absence is information not an error)
    # =====================================================================
    if action == "status":
        try:
            active = hb.is_active(room_id)
            entries = hb.status() if active else []
        except Exception as e:
            return ToolResult(
                content=f"Heartbeat status failed: {type(e).__name__}.",
                is_error=True,
            )
        entry = next((e for e in entries if e.room_id == room_id), None)
        if entry is None:
            return ToolResult(
                content="No heartbeat active in this room.",
                is_error=False,
            )

        # Format the entry — mirror the slash status line shape, this room only.
        next_str = format_interval(int(entry.seconds_until_next))
        schedule = getattr(entry, "schedule", None)
        if schedule:
            tz_name = getattr(entry, "tz", None) or "UTC"
            line = (
                f'every "{schedule}" ({tz_name}), next in {next_str}'
            )
        else:
            interval_str = format_interval(int(entry.interval_seconds))
            line = f"every {interval_str}, next in {next_str}"

        directive = getattr(entry, "directive", None)
        if directive:
            one_line = " ".join(directive.split())
            trunc = one_line if len(one_line) <= 120 else one_line[:119] + "…"
            line += f" · directive: {trunc}"

        return ToolResult(content=line, is_error=False)

    # Unknown action — steer the caller toward the valid actions.
    return ToolResult(
        content=(
            f"Unknown action: {action!r}. Valid actions: start, stop, status."
        ),
        is_error=True,
    )


def _update_read_registry(resolved_path: str | None, callbacks: dict | None) -> None:
    """Refresh the read-registry mtime entry for a resolved path after a mutation.

    Shared by file_read/file_write/file_edit/file_patch (all four call sites use
    this exact same best-effort pattern) so that a file freshly written/edited/
    patched/read is considered "read this session" for the write-before-read
    guard (V6). Best effort: any failure (missing registry, stat failure, no
    callbacks) is silently ignored — the guard falls back to conservative
    denial rather than raising through a successful tool call.

    Args:
        resolved_path: Absolute, resolved path whose registry entry should be
            refreshed, or None if no path was resolved for this call.
        callbacks: The callbacks dict passed to execute_tool (may be None, or
            may lack a "read_registry" key).
    """
    if resolved_path is None or callbacks is None:
        return
    _registry = callbacks.get("read_registry")
    if _registry is None:
        return
    try:
        _mtime = os.stat(resolved_path).st_mtime
        _registry[resolved_path] = _mtime
    except Exception:
        pass  # best effort; guard will conservatively deny if stat fails


def _collect_known_secrets(agent_config: Any) -> set[str]:
    """Assemble the set of currently-known secret VALUES for L1 value-based
    redaction (``redact_known_secrets`` in ``security.py``).

    Unions two populations:
      - ``agent_config.providers[*].api_key`` — the harness's own LLM
        provider keys, resolved once at config-load time (config.py).
      - ``_api_key_cache`` values (``v[0]``) — per-tool keys resolved via a
        tool's ``api_key_cmd`` (e.g. an ``op read`` command), cached with a
        TTL above.

    Defensive: tolerates a missing/non-dict ``.providers`` attribute on
    ``agent_config`` (explicit ``isinstance`` guard — treated as empty
    rather than raising), and skips None/empty-string values from either
    source. No length filter here — the length/entropy threshold lives
    entirely in ``redact_known_secrets`` (its ``min_length`` parameter),
    keeping this collector a pure "what values currently exist" assembly
    step.

    Args:
        agent_config: The agent's config object (or anything config-shaped;
            only ``.providers`` is read, defensively).

    Returns:
        A deduplicated ``set[str]`` of known secret values. Empty when there
        are no providers and the api-key cache is empty (the "works without
        1Password" no-op case).
    """
    known: set[str] = set()

    providers = getattr(agent_config, "providers", None)
    if not isinstance(providers, dict):
        providers = {}
    for provider in providers.values():
        api_key = getattr(provider, "api_key", None)
        if api_key:
            known.add(api_key)

    for cached in _api_key_cache.values():
        value = cached[0] if cached else None
        if value:
            known.add(value)

    return known


async def execute_tool(
    name: str,
    input: dict,
    tool_config: dict,
    agent_config: Any,
    tools: list[ToolDef] | None = None,
    callbacks: dict | None = None,
) -> ToolResult:
    """Validate name/params against BUILTIN_TOOLS, dispatch to the tool's
    executor module, then redact credentials from the result before
    returning (truncate + wrap happen later, at the call site).

    R9: this outer function is a thin wrapper — ALL dispatch logic (including
    the early unknown-tool / missing-param / guard-refusal returns) lives in
    ``_execute_tool_inner``, so every return path (early or late) flows
    through the SAME credential-redaction tail below before reaching the
    caller. Message formats are unchanged from before the refactor; only the
    control flow was moved.

    Args:
        name: Tool name to execute
        input: Tool input parameters
        tool_config: Tool-specific configuration
        agent_config: Agent configuration (for sub-agents, etc.)
        
    Returns:
        ToolResult with content and error status
    """
    result = await _execute_tool_inner(
        name=name,
        input=input,
        tool_config=tool_config,
        agent_config=agent_config,
        tools=tools,
        callbacks=callbacks,
    )

    # Redact credentials from tool output — applies to EVERY return path
    # above (unknown tool, missing param, guard refusal, and normal
    # dispatch results alike), not just the successful-dispatch tail.
    #
    # L1 value-based pass FIRST (redact known secrets in full, before the
    # pattern pass could fragment one that embeds a pattern-shaped
    # substring) — remediation round 1 §B: value-first makes the L1
    # backstop guarantee actually hold ("the value IS the value —
    # redacted in full"). Cost is cosmetic: a KNOWN shaped key (e.g. a
    # live sk-ant-... provider key) now gets [REDACTED:known_secret]
    # instead of [REDACTED:api_key]. An UNKNOWN shaped secret (not in the
    # known-set) still gets [REDACTED:api_key] from the pattern pass below.
    known_secrets = _collect_known_secrets(agent_config)
    if known_secrets:
        from .security import redact_known_secrets as _redact_known_secrets
        content_after_value, value_events = _redact_known_secrets(result.content, known_secrets)
    else:
        content_after_value, value_events = result.content, []

    # Pattern pass SECOND (catches shaped UNKNOWN secrets not in the
    # known-set — proves value-first did not weaken pattern coverage).
    from .security import redact_credentials as _redact_credentials
    redacted_content, pattern_events = _redact_credentials(content_after_value)
    redaction_events = value_events + pattern_events

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


async def _execute_tool_inner(
    name: str,
    input: dict,
    tool_config: dict,
    agent_config: Any,
    tools: list[ToolDef] | None = None,
    callbacks: dict | None = None,
) -> ToolResult:
    """Dispatch body moved out of execute_tool (R9). Every return here —
    early (unknown tool / missing param / guard refusal) or from a tool's
    executor module — is just a ToolResult; execute_tool applies the
    redaction tail uniformly to whatever this function returns.

    MESSAGE FORMATS UNCHANGED from the pre-refactor execute_tool: this is a
    pure code-motion, not a rewording.
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
    # grep/glob join this tuple ONLY (not the _resolved_path registry tuple
    # below) — a match is not a file read for write-guard purposes (anchors §7.2).
    #
    # NOTE: this is a convenience default-root join, NOT a security boundary.
    # Containment is enforced at the OS layer by the systemd sandbox (Unix user
    # per agent + ProtectHome=tmpfs + BindPaths=/home/oa-%i /srv/openalph/shared),
    # which deliberately places /srv/openalph/shared INSIDE the sandbox so the
    # shared-skill/shared-doc symlink convention works. An app-level path
    # allowlist here (the reverted SEC-9 guard, commit a194f8b) only knew about
    # the agent's own workspace, drifted out of sync with the real boundary, and
    # rejected legitimate shared reads. Do not reintroduce it — see
    # memory/projects/openalph/session-brief-next.md (kdsn.252).
    if name in ("file_read", "file_write", "file_edit", "file_patch", "send_media",
                "grep", "glob") and "path" in input:
        file_path = input["path"]
        if not os.path.isabs(file_path) and hasattr(agent_config, "workspace"):
            input["path"] = str(agent_config.workspace / file_path)

    # Normalize path to resolved form for registry keys (symlinks, .., relative spellings)
    # so that read via relative and write via absolute always hit the same registry entry.
    _resolved_path: str | None = None
    if name in ("file_read", "file_write", "file_edit", "file_patch") and "path" in input:
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
        if not result.is_error:
            _update_read_registry(_resolved_path, callbacks)
    elif name == "file_write":
        from .file import write_file
        import os as _os_fw

        # --- Read-before-write guard ---
        _fw_path = input["path"]
        _guard_enabled = tool_config.get("require_read_before_write", True)
        if _guard_enabled and _resolved_path is not None and _os_fw.path.exists(_resolved_path):
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
                _current_mtime = _os_fw.stat(_resolved_path).st_mtime
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
            tool_config=tool_config,
        )
        # On successful write, update read_registry with new mtime
        if not result.is_error:
            _update_read_registry(_resolved_path, callbacks)
    elif name == "file_edit":
        from .file import edit_file
        result = await edit_file(
            path=input["path"],
            old_text=input["old_text"],
            new_text=input["new_text"],
            replace_all=input.get("replace_all", False),
            tool_config=tool_config,
        )
        # On successful edit, update read_registry with new mtime (keeps registry fresh)
        if not result.is_error:
            _update_read_registry(_resolved_path, callbacks)
    elif name == "file_patch":
        from .file import patch_file
        result = await patch_file(
            path=input["path"],
            patch=input["patch"],
            tool_config=tool_config,
        )
        # On successful patch, update read_registry with new mtime (keeps registry fresh)
        if not result.is_error:
            _update_read_registry(_resolved_path, callbacks)
    elif name == "grep":
        from .search import run_grep
        # R16: search tools require an explicit workspace — never silently
        # fall back to scanning the process CWD.
        if not hasattr(agent_config, "workspace"):
            return ToolResult(
                content=(
                    "grep requires workspace configuration: agent_config has "
                    "no 'workspace' attribute. Refusing to fall back to the "
                    "process working directory."
                ),
                is_error=True,
            )
        # N1 (honesty correction): this asyncio.wait_for is NOT a scan bound
        # and must never be relied on as one. wait_for can only cancel
        # run_grep at an `await` point, but run_grep's scan body (walk /
        # stat / open / read / rx.search) is entirely SYNCHRONOUS -- zero
        # awaits between entry and return. A synchronous coroutine that
        # hangs holds the event loop, so wait_for's timeout callback cannot
        # be delivered until the coroutine yields the loop, which a hung
        # scan never does (verified: wrapping a busy sync loop in
        # asyncio.wait_for lets it run to full completion regardless of
        # timeout). This wrapper is kept only because it is a harmless
        # no-op belt for the ordinary case (run_grep returns/raises before
        # ever needing to be cancelled); it is NOT "defense in depth" against
        # a stuck scan.
        #
        # The REAL and ONLY bound on scan wall-clock time is search.py's
        # in-scan SIGALRM+deadline enforcement (_time_budget_guard), which
        # requires running on the main thread of the main interpreter
        # (signal.signal/setitimer raise ValueError off-main-thread). That
        # guard now logs a warning (once per scan entry) if it is ever
        # unable to arm -- e.g. because a future refactor dispatches grep/
        # glob via asyncio.to_thread or a worker-thread event loop -- so an
        # operator can see the ReDoS bound was silently lost; this
        # wait_for wrapper will NOT catch that condition either way.
        budget = float(tool_config.get("time_budget_seconds", 10))
        try:
            result = await asyncio.wait_for(
                run_grep(
                    pattern=input["pattern"],
                    path=input.get("path"),
                    glob=input.get("glob"),
                    output_mode=input.get("output_mode", "files_with_matches"),
                    head_limit=input.get("head_limit"),
                    case_insensitive=input.get("case_insensitive", False),
                    config=tool_config,
                    workspace=agent_config.workspace,
                ),
                timeout=budget + 5.0,
            )
        except asyncio.TimeoutError:
            result = ToolResult(
                content=(
                    f"grep exceeded its time budget ({budget:g}s) at the "
                    "dispatch level — narrow with path/glob or simplify the "
                    "pattern."
                ),
                is_error=True,
            )
        # grep is read-only over file CONTENT for search purposes, not a
        # file_read — it does NOT touch the read-registry (anchors §7.2).
    elif name == "glob":
        from .search import run_glob
        # R16: search tools require an explicit workspace — never silently
        # fall back to scanning the process CWD.
        if not hasattr(agent_config, "workspace"):
            return ToolResult(
                content=(
                    "glob requires workspace configuration: agent_config has "
                    "no 'workspace' attribute. Refusing to fall back to the "
                    "process working directory."
                ),
                is_error=True,
            )
        # N1 (honesty correction): see the grep branch above -- this
        # asyncio.wait_for is NOT a scan bound (run_glob's scan body is also
        # entirely await-free, so wait_for cannot cancel a hung scan). The
        # real and only bound is search.py's in-scan SIGALRM+deadline
        # enforcement, which is main-thread-only and now logs a warning if
        # it fails to arm.
        budget = float(tool_config.get("time_budget_seconds", 10))
        try:
            result = await asyncio.wait_for(
                run_glob(
                    pattern=input["pattern"],
                    path=input.get("path"),
                    head_limit=input.get("head_limit"),
                    config=tool_config,
                    workspace=agent_config.workspace,
                ),
                timeout=budget + 5.0,
            )
        except asyncio.TimeoutError:
            result = ToolResult(
                content=(
                    f"glob exceeded its time budget ({budget:g}s) at the "
                    "dispatch level — narrow with path/glob or simplify the "
                    "pattern."
                ),
                is_error=True,
            )
        # glob is a directory/name listing, not a file_read — it does NOT
        # touch the read-registry (anchors §7.2).
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
            offset=input.get("offset"),
            tool_config=tool_config,
            result_limit=getattr(agent_config, "truncation_limit", None),
        )
    elif name == "web_fetch_js":
        from .web import web_fetch_js
        # Reuse the TTL-cached resolver — same rate-limit safety invariant as
        # web_search (§6, load-bearing): a per-call `op read` under burst load
        # (e.g. an agent scraping a paginated JS site) exhausts the 1Password
        # service-account rate limit; the TTL cache is what prevents that.
        api_key = tool_config.get("api_key", "")
        if not api_key and "api_key_cmd" in tool_config:
            api_key = _resolve_cached_api_key(
                tool_config["api_key_cmd"],
                tool_config.get("api_key_cache_ttl"),
            )
        result = await web_fetch_js(
            url=input.get("url", ""),
            schema=input.get("schema"),
            effort=input.get("effort") or tool_config.get("default_effort", "max"),
            max_chars=input.get("max_chars"),
            offset=input.get("offset"),
            # L3: pass the raw value through -- do NOT force bool() here.
            # bool("false") is True, so a force-bool() at the dispatch seam
            # would turn a stringly-typed "false" into True before the
            # handler ever sees it. web_fetch_js itself validates
            # isinstance(nocache, bool) and coerces any non-bool to False.
            nocache=input.get("nocache", False),
            api_key=api_key,
            base_url=tool_config.get("base_url", "https://api.tabstack.ai/v1"),
            timeout=tool_config.get("timeout", 90),
            result_limit=getattr(agent_config, "truncation_limit", None),
        )
    elif name == "subagent":
        from .subagent import run_subagent
        # BUG-3: honour the documented `default_max_iterations` config key when
        # the call omits max_iterations, instead of silently falling back to the
        # module constant. Explicit per-call max_iterations still wins.
        _sub_max_iters = input.get("max_iterations")
        if _sub_max_iters is None:
            _sub_max_iters = tool_config.get("default_max_iterations")
        # Parent-turn liveness while we are blocked here is emitted by
        # run_subagent itself, from REAL sub-run milestones (each provider
        # response, each completed tool-call iteration) via
        # callbacks['turn_progress'].
        #
        # There is deliberately NO time-based pinger here (removed round 2). A
        # blind "still running" heartbeat reports elapsed time, not progress: a
        # sub parked in the SAME provider retry storm the parent's stall
        # watchdog exists to break would have reset that watchdog forever while
        # both parent room locks stayed held — recreating the exact incident.
        # A sub with no milestone for longer than the parent's
        # turn_stall_timeout_seconds SHOULD be cancelled; see the note above
        # the milestone hook in tools/subagent.py.
        result = await run_subagent(
            task=input["task"],
            config=agent_config,
            tools=tools,
            system_prompt=input.get("system_prompt"),
            model=input.get("model"),
            max_tokens=input.get("max_tokens"),
            max_iterations=_sub_max_iters,
            call_id=callbacks.get("call_id") if callbacks else None,
            # Flight recorder (workspace-kdsn.192): the PARENT room, so the
            # sub's transcript header can cross-reference where it was
            # dispatched from. callbacks["room_id"] here is the main-loop's
            # own room_id (set by MatrixBot._build_agent_callbacks), i.e.
            # the parent — never the sub's own dispatch-time "__sub__" id
            # used internally inside run_subagent's tool-call callbacks.
            parent_room_id=callbacks.get("room_id") if callbacks else None,
            callbacks=callbacks,
        )
    elif name == "advisor":
        from .advisor import run_advisor
        result = await run_advisor(
            focus=input.get("focus"),
            model=input.get("model"),
            config=agent_config,
            tool_config=tool_config,
            callbacks=callbacks,
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
    elif name == "view_image":
        from .vision import view_image
        result = await view_image(
            path=input["path"],
            tool_config=tool_config,
            agent_config=agent_config,
            callbacks=callbacks,
        )
    elif name == "todo_write":
        result = await _execute_todo_write(input, callbacks)
    elif name == "heartbeat":
        result = await _execute_heartbeat_tool(input, callbacks)
    else:
        return ToolResult(
            content=f"Unknown tool: {name}",
            is_error=True,
        )

    # R9: redaction now applied uniformly by the outer execute_tool wrapper
    # (covers this return AND the early returns above) — nothing to do here.
    return result
