# Media Attachment Handling — Specification (.24)

## Overview

Detect Matrix media events (images, audio, video, files), download them to the agent's workspace, and pass file paths to the agent in message content. The agent decides what to do with the files — this feature is purely transport-layer.

## Scope

**In scope:**
- Unencrypted media events: `RoomMessageImage`, `RoomMessageAudio`, `RoomMessageVideo`, `RoomMessageFile`
- Download via `client.download()`
- Storage to agent workspace
- Passing file path + metadata to `agent.handle_input()`
- Size limits (reject files exceeding threshold)

**Out of scope:**
- Encrypted media (E2EE — no olm/megolm setup)
- Vision/image understanding (.25)
- Audio transcription (.28)
- Media cleanup/pruning (operator responsibility)

## Design

### Event Registration

Register callbacks for four additional event types in `MatrixBot.start()`:

```python
from nio import RoomMessageImage, RoomMessageAudio, RoomMessageVideo, RoomMessageFile

self.client.add_event_callback(self._handle_media_message, RoomMessageImage)
self.client.add_event_callback(self._handle_media_message, RoomMessageAudio)
self.client.add_event_callback(self._handle_media_message, RoomMessageVideo)
self.client.add_event_callback(self._handle_media_message, RoomMessageFile)
```

### Media Handler

New method `_handle_media_message(room, event)` that:

1. **Guards** — same as `_handle_room_message`: skip if not synced, skip own messages
2. **Checks size** — `event.body` contains the filename; content length comes from the download response. We check after download headers arrive. If >20MB (`MAX_MEDIA_BYTES = 20_000_000`), skip and send a text note to the agent.
3. **Downloads** — `client.download(mxc=event.url, filename=event.body)`
4. **Stores** — Save to `{workspace}/media/{event_id_safe}/{sanitized_filename}`
5. **Formats message** — Construct a text message with bracket-tag metadata
6. **Delegates** — Pass to the same message-processing pipeline as `_handle_room_message` (room activation, typing indicator, agent input, response sending)

### Storage Path

```
{agent.config.workspace}/media/{event_id_hash}/{sanitized_filename}
```

- `event_id_hash`: First 16 chars of SHA-256 hex digest of `event.event_id`. Avoids filesystem issues with Matrix event ID characters (`$`, `:`, etc.)
- `sanitized_filename`: `event.body` with path separators and null bytes stripped, truncated to 200 chars. Falls back to `attachment` if empty after sanitization.
- Directory created with `parents=True, exist_ok=True`

### Message Format

The message passed to `agent.handle_input()`:

```
[media: {relative_path} ({mime_type}, {size_human})]
{caption}
```

Examples:
```
[media: media/a1b2c3d4e5f6a1b2/photo.jpg (image/jpeg, 2.4 MB)]
Check out this sunset
```

```
[media: media/f9e8d7c6b5a4f9e8/report.pdf (application/pdf, 1.1 MB)]
```

- `relative_path`: Path relative to workspace root
- `mime_type`: From `event.source.get("content", {}).get("info", {}).get("mimetype", "application/octet-stream")`
- `size_human`: Human-readable size (e.g., "2.4 MB", "350 KB")
- `caption`: `event.body` if it differs from the filename (Matrix clients set body=filename when no caption). If body == filename, omit caption line.

### Size Limit

- `MAX_MEDIA_BYTES = 20_000_000` (20 MB, matches conduwuit default)
- Configurable? Not initially — hardcoded constant in `matrix.py`. Can be moved to config later.
- Check: After download, check `len(response.body)`. If exceeds limit, delete partial file, log warning, and send notification to agent:

```
[media: skipped — {filename} exceeds 20 MB limit ({actual_size_human})]
```

### Filename Sanitization

```python
def _sanitize_filename(name: str) -> str:
    """Sanitize a filename for safe filesystem storage."""
    # Strip path separators and null bytes
    name = name.replace("/", "_").replace("\\", "_").replace("\0", "")
    # Truncate
    name = name[:200]
    # Fallback
    return name or "attachment"
```

### Download Error Handling

If `client.download()` returns `DownloadError`:
- Log the error
- Send text to agent: `[media: download failed — {filename} ({error})]`
- Do NOT crash the bot or skip the event silently

### Refactoring: Shared Message Processing

`_handle_room_message` and `_handle_media_message` share logic:
- Sync/sender guards
- Room activation
- Typing indicator
- Agent input → response → send
- Session logging
- Error handling (overflow, generic)

Extract common logic into `_process_message(room, event, body: str)` that both handlers call. `_handle_room_message` passes `event.body` directly; `_handle_media_message` downloads the file and constructs the bracket-tag message first, then calls `_process_message`.

## Constants

```python
MAX_MEDIA_BYTES = 20_000_000  # 20 MB
MEDIA_DIR = "media"           # Subdirectory under workspace
```

## Config Changes

None. This is a zero-config feature — if the bot receives media, it handles it.

## File Changes

| File | Changes |
|------|---------|
| `matrix.py` | Add imports, `_handle_media_message()`, `_process_message()`, `_sanitize_filename()`, `_event_id_hash()`, media event registration in `start()` |
| (no other files) | Agent, config, provider unchanged |

## Test Plan

See `tests/test_media.py`. Coverage:

1. **Event registration** — all 4 media types registered
2. **Download + storage** — file lands at correct path
3. **Message format** — bracket tag with correct mime/size/path
4. **Caption vs filename** — caption included only when different from filename
5. **Size limit** — oversized files rejected with notification
6. **Filename sanitization** — path traversal, empty, long names
7. **Download error** — graceful handling, notification sent
8. **Sync guard** — media events during initial sync are skipped
9. **Own message guard** — own media events are skipped
10. **Refactor** — `_process_message` called by both text and media handlers
