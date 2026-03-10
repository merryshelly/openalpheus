# Image/Vision Support — Specification (.25)

## Overview

When `vision: true` is set in agent config, image attachments from .24 are converted to image content blocks in LLM messages instead of text-only bracket tags. The agent's model sees the actual pixels. Non-image media (audio, video, files) remains text-only regardless. Non-vision agents (`vision: false`, the default) are unchanged.

## Scope

**In scope:**
- `vision` config flag (bool, default false)
- Parse `[media: ...]` tags to detect image attachments
- Read image files, base64 encode, construct multi-part content blocks
- Provider-specific wire format conversion (Anthropic + OpenAI)
- Graceful fallback for non-vision configs and non-image media
- Image token estimation for context overflow checking
- Supported formats: JPEG, PNG, GIF, WebP

**Out of scope:**
- Specialist vision model routing (deferred to .27)
- Image resizing/compression
- Video frame extraction
- Encrypted media

## Design

### Config Change

Add `vision: bool = False` to `AgentConfig`:

```python
@dataclass
class AgentConfig:
    ...
    vision: bool = False
```

Load from TOML:
```toml
[agent]
vision = true
```

### Architecture: Three-Layer Split

1. **matrix.py (.24)** — Downloads media, writes `[media: path (mime, size)]` tag. Unchanged.
2. **agent.py (.25)** — Parses media tags, reads images, builds normalized multi-part content. New.
3. **provider.py (.25)** — Converts normalized image blocks to provider wire format. New.

### Agent Layer: Content Construction

In `agent.py`, new function `_build_user_content(text, config)`:

```python
def _build_user_content(text: str, config: AgentConfig) -> str | list[dict]:
    """Build user message content, expanding image media tags when vision is enabled.
    
    Returns plain text string when no image expansion needed.
    Returns list of content blocks when images are present and vision is enabled.
    """
```

**Logic:**
1. If `config.vision` is False → return `text` unchanged (plain string)
2. Parse all `[media: <path> (<mime>, <size>)]` tags in the text
3. For each tag where mime starts with `image/` and mime is in SUPPORTED_IMAGE_TYPES:
   a. Resolve path relative to `config.workspace`
   b. Read file, base64 encode
   c. Replace the tag with an image content block
4. Non-image media tags remain as text
5. If no images found → return `text` unchanged (plain string)
6. If images found → return list of content blocks:
   ```python
   [
       {"type": "text", "text": "remaining text including non-image tags"},
       {"type": "image", "media_type": "image/jpeg", "data": "<base64>"},
       ...
   ]
   ```

**Media tag regex:**
```python
MEDIA_TAG_RE = re.compile(
    r'\[media:\s*(\S+)\s+\(([^,]+),\s*([^)]+)\)\]'
)
# Groups: (1) relative_path, (2) mime_type, (3) human_size
```

**Supported image MIME types:**
```python
VISION_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
```

### Normalized Content Format

The normalized message format extends to support image blocks:

```python
# Text-only (unchanged):
{"role": "user", "content": "hello"}

# With images (new):
{"role": "user", "content": [
    {"type": "text", "text": "Check this out"},
    {"type": "image", "media_type": "image/jpeg", "data": "base64..."},
]}
```

This is the **internal** format. Provider conversion translates to wire format.

### Provider Layer: Wire Format Conversion

Extend `_convert_messages_for_anthropic` and `_convert_messages_for_openai` to handle image content blocks in user messages.

**Anthropic wire format:**
```python
{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}
```

**OpenAI wire format:**
```python
{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
```

**Conversion rules:**
- User message with `content` as string → pass through (unchanged behavior)
- User message with `content` as list → iterate blocks:
  - `{"type": "text", ...}` → pass through
  - `{"type": "image", "media_type": M, "data": D}` → convert to provider format
- All other message types → unchanged

### Context Token Estimation

Images consume tokens. Add image token cost to `_estimate_context_tokens`:

- For image content blocks: `len(base64_data) * 3 / 4 / 750` (decode size / 750)
- This approximates Anthropic's formula without needing to decode dimensions
- Conservative enough for overflow prevention

### Fallback Behavior

| Config | Image media tag | Result |
|--------|----------------|--------|
| `vision: false` | `[media: path (image/jpeg, 2.4 MB)]` | Text passed as-is (bracket tag) |
| `vision: true` | `[media: path (image/jpeg, 2.4 MB)]` | Image block + caption text |
| `vision: true` | `[media: path (audio/ogg, 1.1 MB)]` | Text passed as-is (not an image) |
| `vision: true` | `[media: path (image/tiff, 5 MB)]` | Text passed as-is (unsupported format) |
| `vision: true` | Image file missing/unreadable | Text passed as-is + warning logged |

### Error Handling

- **File not found:** Log warning, leave media tag as text. Don't crash.
- **File read error:** Same — log and leave as text.
- **Unsupported image format:** Leave as text. Only JPEG/PNG/GIF/WebP get expanded.
- **Empty file:** Leave as text, log warning.

## File Changes

| File | Changes |
|------|---------|
| `config.py` | Add `vision: bool = False` to AgentConfig, load from TOML |
| `agent.py` | New `_build_user_content()`, call it in `handle_input()` before appending to history, update `_estimate_context_tokens` for image blocks |
| `provider.py` | Extend `_convert_messages_for_anthropic()` and `_convert_messages_for_openai()` to convert image blocks |

## Constants

```python
# agent.py
VISION_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MEDIA_TAG_RE = re.compile(r'\[media:\s*(\S+)\s+\(([^,]+),\s*([^)]+)\)\]')

# Approximate tokens per raw image byte (base64 decoded)
IMAGE_TOKENS_PER_BYTE = 1 / 750
```

## Test Plan

See `tests/test_vision.py`. Coverage:

1. **Config:** `vision` flag loads from TOML, defaults to False
2. **Content building:** `_build_user_content` parses tags and expands images
3. **No-op when vision disabled:** Tags pass through unchanged
4. **Non-image media:** Audio/video/file tags always pass through as text
5. **Unsupported image formats:** TIFF, BMP etc. left as text
6. **Missing files:** Graceful fallback to text
7. **Provider conversion — Anthropic:** Image blocks → Anthropic wire format
8. **Provider conversion — OpenAI:** Image blocks → OpenAI wire format
9. **Mixed content:** Text + multiple images in one message
10. **Caption preservation:** Non-media text around tags is preserved
11. **Token estimation:** Image blocks contribute to context size
12. **Integration:** Full flow from media tag → history → provider
