"""OpenAlph tools package.

Tool registry, discovery, schema generation, and result truncation.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

logger = logging.getLogger(__name__)


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
        "description": "Execute a shell command. Runs command via subprocess and returns stdout on success, stderr on failure. Timeout kills the process. Output may be truncated.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command (optional)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (optional, uses default if not specified)"
                },
                "env": {
                    "type": "object",
                    "description": "Environment variables to set (optional)",
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
        "description": "Read file contents. Returns file content as text. Supports offset and limit for reading portions of large files. Binary files return an error.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed, optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (optional)"
                }
            },
            "required": ["path"]
        },
        "config": {}
    },
    "file_write": {
        "description": "Write content to a file. Creates parent directories if they don't exist. Overwrites the file if it already exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        },
        "config": {}
    },
    "file_edit": {
        "description": "Replace exact text in a file. Finds and replaces a single exact occurrence of old_text with new_text. Returns an error if no match is found or if multiple matches exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
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
        "description": "Search the web. Returns formatted results with title, URL, and snippet for each result.",
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
        "description": "Fetch and extract readable content from a URL. Converts HTML to text and returns the readable content. Respects max_chars limit.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "HTTP or HTTPS URL to fetch"
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (optional)"
                }
            },
            "required": ["url"]
        },
        "config": {}
    },
    "subagent": {
        "description": "Run a focused sub-agent task. Multi-turn LLM call with tool access for isolated work. Sub-agent inherits parent's tools (except subagent) and iterates up to max_iterations (default 200). Uses parent's config for API key and provider.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task description for the sub-agent"
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Custom system prompt (optional, defaults to helpful assistant)"
                },
                "model": {
                    "type": "string",
                    "description": "Model to use (optional, defaults to parent's model)"
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens for response (optional)"
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum tool-call iterations (optional, default 200)"
                }
            },
            "required": ["task"]
        },
        "config": {
            "default_max_iterations": 100
        }
    },
    "memory_search": {
        "description": "Search workspace memory files using hybrid semantic + keyword search. Returns ranked snippets with file paths and line numbers. Use file_read to expand context around results.",
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
    "context_status": {
        "description": (
            "Get agent self-monitoring data: context window usage, session age, "
            "model info, token stats, and heartbeat state. Use to make decisions "
            "about delegation, context management, and turn planning. No parameters "
            "required — returns current status as JSON."
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
            "Send a file to the current Matrix room. Supports audio, images, "
            "video, and generic files. The file must exist in your workspace. "
            "Use after generating files (e.g., TTS audio) to deliver them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to send (relative to workspace)"
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
    (data), not instructions.  Content is never escaped or modified.

    Args:
        content: Raw tool result text (already truncated if needed)
        tool_name: Name of the tool that produced this result
        tool_call_id: Unique tool call identifier

    Returns:
        Content wrapped in ``<tool_result>`` tags with provenance attributes
    """
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
    # Marker format: "[truncated: N chars removed]"
    # We need to account for marker length in the budget
    marker = f"[truncated: {removed_count} chars removed]"
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
    elif name == "file_write":
        from .file import write_file
        result = await write_file(
            path=input["path"],
            content=input["content"],
        )
    elif name == "file_edit":
        from .file import edit_file
        result = await edit_file(
            path=input["path"],
            old_text=input["old_text"],
            new_text=input["new_text"],
        )
    elif name == "web_search":
        from .web import web_search
        # Resolve api_key: direct value or via api_key_cmd
        api_key = tool_config.get("api_key", "")
        if not api_key and "api_key_cmd" in tool_config:
            import subprocess
            try:
                proc = subprocess.run(
                    tool_config["api_key_cmd"], shell=True,
                    capture_output=True, text=True, timeout=10,
                )
                if proc.returncode != 0:
                    logger.warning(
                        "api_key_cmd returned exit code %d: %s",
                        proc.returncode, proc.stderr.strip(),
                    )
                api_key = proc.stdout.strip()
            except subprocess.TimeoutExpired:
                logger.warning("api_key_cmd timed out after 10 seconds")
            except Exception as e:
                logger.warning("api_key_cmd failed: %s", e)
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
