"""OpenAlph tools package.

Tool registry, discovery, schema generation, and result truncation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


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
        "description": "Run a focused sub-agent task. Single-turn LLM call for isolated work. No tools available to the sub-agent. Uses parent's config for API key and provider.",
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
                }
            },
            "required": ["task"]
        },
        "config": {
            "default_max_iterations": 10
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
    agent_config: Any
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
    # Stub implementation - actual execution in tool modules
    return ToolResult(
        content=f"Tool '{name}' execution not yet implemented",
        is_error=True
    )
