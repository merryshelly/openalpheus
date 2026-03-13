"""Media upload tool for sending files to Matrix rooms."""

import mimetypes
from pathlib import Path

from openalph.tools import ToolResult


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string (e.g., "2.5 MB", "100 KB", "500 B")
    """
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    elif size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    else:
        return f"{size_bytes} B"


async def send_media(
    path: str,
    caption: str | None = None,
    max_upload_bytes: int = 20_971_520,
    upload_callback = None,
) -> ToolResult:
    """Send a file to the current Matrix room.
    
    Args:
        path: Path to the file to send (relative to workspace or absolute)
        caption: Optional caption/description for the file
        max_upload_bytes: Maximum allowed file size in bytes (default: 20MB)
        upload_callback: Async callback function to handle the upload.
                        Signature: async callback(file_path, content_type, filename, caption)
    
    Returns:
        ToolResult with success message or error details
    """
    file_path = Path(path)
    
    # Validate file exists
    if not file_path.exists():
        return ToolResult(
            content=f"File not found: {path}",
            is_error=True,
        )
    
    # Validate it's a file (not a directory)
    if not file_path.is_file():
        return ToolResult(
            content=f"Path is not a file: {path}",
            is_error=True,
        )
    
    # Get file size
    file_size = file_path.stat().st_size
    
    # Validate file is not empty
    if file_size == 0:
        return ToolResult(
            content=f"File is empty: {path}",
            is_error=True,
        )
    
    # Validate file size limit
    if file_size > max_upload_bytes:
        size_human = _format_size(file_size)
        limit_human = _format_size(max_upload_bytes)
        return ToolResult(
            content=f"File too large: {size_human} (limit: {limit_human})",
            is_error=True,
        )
    
    # Detect MIME type
    content_type, _ = mimetypes.guess_type(str(file_path))
    if content_type is None:
        content_type = "application/octet-stream"
    
    filename = file_path.name
    
    # If no callback (CLI mode), return error with file path
    if upload_callback is None:
        return ToolResult(
            content=f"Media upload not available in CLI mode. File saved at: {file_path}",
            is_error=True,
        )
    
    # Call the upload callback
    try:
        await upload_callback(
            file_path=file_path,
            content_type=content_type,
            filename=filename,
            caption=caption,
        )
    except Exception as e:
        return ToolResult(
            content=f"Upload failed: {e}",
            is_error=True,
        )
    
    # Success
    size_human = _format_size(file_size)
    return ToolResult(
        content=f"Sent {filename} ({size_human})",
        is_error=False,
    )
