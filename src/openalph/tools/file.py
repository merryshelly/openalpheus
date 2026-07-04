"""File operations: read, write, edit."""

import os
from pathlib import Path

from openalph.tools import ToolResult


def _is_binary(filepath: str) -> bool:
    """Check if a file is binary by reading the first chunk."""
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(8192)
            if not chunk:
                return False
            # Check for null bytes or high ratio of non-printable chars
            if b'\x00' in chunk:
                return True
            # Use a simple heuristic: if >30% non-printable, consider binary
            non_printable = sum(1 for b in chunk if b < 32 and b not in (9, 10, 13))
            return non_printable / len(chunk) > 0.3
    except Exception:
        return False


async def read_file(path: str, offset: int | None = None, limit: int | None = None) -> ToolResult:
    """Read file contents.
    
    Args:
        path: Path to the file
        offset: 1-indexed line number to start from
        limit: Maximum number of lines to return
    
    Returns:
        ToolResult with content or error
    """
    try:
        if not os.path.exists(path):
            return ToolResult(content=f"Error: File not found: {path}", is_error=True)
        
        if not os.path.isfile(path):
            return ToolResult(content=f"Error: Path is not a file: {path}", is_error=True)
        
        if _is_binary(path):
            return ToolResult(content=f"Error: Binary file cannot be read: {path}", is_error=True)
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Handle offset (1-indexed)
        if offset is not None:
            if offset < 1:
                return ToolResult(content=f"Error: offset must be >= 1", is_error=True)
            start_idx = offset - 1
        else:
            start_idx = 0
        
        # Handle limit
        if limit is not None:
            if limit < 0:
                return ToolResult(content=f"Error: limit must be >= 0", is_error=True)
            end_idx = start_idx + limit
        else:
            end_idx = len(lines)
        
        # Slice the lines
        selected_lines = lines[start_idx:end_idx]
        content = ''.join(selected_lines)
        
        return ToolResult(content=content, is_error=False)
    
    except UnicodeDecodeError:
        return ToolResult(content=f"Error: Binary file cannot be read: {path}", is_error=True)
    except Exception as e:
        return ToolResult(content=f"Error reading file: {e}", is_error=True)


async def write_file(path: str, content: str) -> ToolResult:
    """Write content to a file, creating parent directories if needed.
    
    Args:
        path: Path to the file
        content: Content to write
    
    Returns:
        ToolResult with confirmation or error
    """
    try:
        # Create parent directories
        parent = Path(path).parent
        if parent:
            parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return ToolResult(content=f"Successfully wrote to {path}", is_error=False)
    
    except Exception as e:
        return ToolResult(content=f"Error writing file: {e}", is_error=True)


async def edit_file(path: str, old_text: str, new_text: str) -> ToolResult:
    """Replace exact occurrence of old_text with new_text.
    
    Requires exactly one match. 0 or 2+ matches results in error.
    File is unchanged on error.
    
    Args:
        path: Path to the file
        old_text: Text to find
        new_text: Text to replace with
    
    Returns:
        ToolResult with confirmation or error
    """
    try:
        if not os.path.exists(path):
            return ToolResult(content=f"Error: File not found: {path}", is_error=True)
        
        if not os.path.isfile(path):
            return ToolResult(content=f"Error: Path is not a file: {path}", is_error=True)

        if _is_binary(path):
            return ToolResult(content=f"Error: Binary file cannot be edited: {path}", is_error=True)
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count occurrences
        count = content.count(old_text)
        
        if count == 0:
            return ToolResult(
                content=(
                    f"Error: old_text not found in {path}. "
                    "Read the file first — its content may differ from what you expect. "
                    "Check spacing, indentation, and line endings exactly."
                ),
                is_error=True,
            )
        
        if count > 1:
            return ToolResult(
                content=(
                    f"Error: old_text appears {count} times in {path} (ambiguous). "
                    "Add more surrounding context lines to old_text to disambiguate "
                    "and uniquely identify the target occurrence."
                ),
                is_error=True,
            )
        
        # Exactly one match - perform replacement
        new_content = content.replace(old_text, new_text, 1)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return ToolResult(content=f"Successfully edited {path}", is_error=False)
    
    except Exception as e:
        return ToolResult(content=f"Error editing file: {e}", is_error=True)
