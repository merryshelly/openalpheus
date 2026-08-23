"""view_image tool: agent-initiated image viewing via staged user-message
injection (workspace-kdsn.276).

``view_image(path)`` lets an agent attach a workspace image to its own
context. The tool validates (containment, existence, image MIME from
extension, size cap, model vision capability) and deposits a
``[media: path (mime, size)]`` tag into the AGENT's per-room inbox
(``Agent._vision_inbox``) via a ``vision_deposit`` callback — wired by default
at agent.py tool dispatch (setdefault; callers may override), with
tools/subagent.py wiring its own per-sub deposit. At the top of the NEXT
tool-loop iteration the agent drains its OWN inbox (there is no drain
callback — kdsn.279 removed MatrixBot's ``_vision_inbox`` +
``_make_vision_callbacks``), frames ALL queued tags into ONE user message
(:func:`frame_vision_batch`), expands it via the EXISTING
``_build_user_content``, and appends it after ALL tool results of the pending
batch. The optional ``log_vision_injection(room_id, framed)`` callback is an
observability seam only (JSONL + notice) — injection is never gated on it.

Provider-universal by design: images ride user messages, never tool results
(vllm#43203). The image bytes NEVER appear in the tool result — the result is
text-only.

Spec: memory/projects/openalph/vision-model-capability-spec.md §2, §4, §5.
"""

from __future__ import annotations

import logging
from pathlib import Path

from openalph.provider import model_supports_vision

logger = logging.getLogger(__name__)

# Local extension -> MIME map (lowercase, no leading dot). Kept local to this
# module per the kdsn.276 contract — not shared with agent.VISION_MIME_TYPES.
_EXT_TO_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}

# Characters that would break the [media: ... ] tag framing or open an
# injection seam (brackets/parens/angle brackets delimit the tag and the
# (mime, size) trailer; control chars can smuggle framing). Rejected with
# "rename the file" steering.
_FORBIDDEN_PATH_CHARS = frozenset("[]()<>")


def _err(content: str):
    """Build an error ToolResult (local import: openalph.agent imports this
    package, so a top-level import of openalph.tools would be circular)."""
    from openalph.tools import ToolResult
    return ToolResult(content=content, is_error=True)


async def view_image(
    path: str,
    tool_config: dict,
    agent_config,
    callbacks: dict | None,
):
    """Validate a workspace image and stage it for injection into context.

    Validation order (each failure is a distinct, steerable error):
      1. vision_deposit callback wired (runtime context supports staging).
      2. Path safety: relative, no ``..``, no framing/control chars, and the
         resolved path must land under ``agent_config.workspace``.
      3. File exists, is a file, non-empty.
      4. Extension maps to a supported image MIME (JPEG/PNG/GIF/WebP).
      5. Size <= tool_config["max_bytes"] (default 5 MB).
      6. Active room model supports vision (else name the model + /model).

    On success: deposit the tag via ``vision_deposit`` and return a text-only
    ToolResult. The image bytes never enter the tool result.
    """
    # 1. Callback wiring — the runtime context must provide a deposit seam.
    deposit = (callbacks or {}).get("vision_deposit")
    if not callable(deposit):
        return _err(
            "view_image is not available in this runtime context: no "
            "vision_deposit callback is wired (the agent tool loop wires one "
            "by default since kdsn.279; this error means the tool was invoked "
            "outside a wired runtime). The image cannot be staged."
        )

    # 2. Path safety — reject absolute paths, traversal, framing chars, and
    # anything that resolves outside the workspace.
    if not isinstance(path, str) or not path:
        return _err("view_image: 'path' must be a non-empty workspace-relative string.")

    if Path(path).is_absolute():
        return _err(
            f"view_image: absolute paths are rejected — pass a path RELATIVE to "
            f"the workspace root (e.g. 'shots/a.jpg'), not '{path}'."
        )

    parts = Path(path).parts
    if ".." in parts:
        return _err(
            f"view_image: '..' components are rejected outright (no normalization "
            f"games) — the path must stay inside the workspace. Got '{path}'. "
            f"Pass a workspace-relative path without '..'."
        )

    bad_chars = sorted({c for c in path if c in _FORBIDDEN_PATH_CHARS})
    has_control = any(ord(c) < 0x20 or ord(c) == 0x7F for c in path)
    if bad_chars or has_control:
        which = ", ".join(repr(c) for c in bad_chars)
        if has_control:
            which = (which + ", " if which else "") + "control character(s)"
        return _err(
            f"view_image: the path contains character(s) that would break the "
            f"[media:] tag framing / open an injection seam ({which}). "
            f"Rename the file to use only plain filename characters "
            f"(letters, digits, '.', '-', '_', '/') and retry."
        )

    workspace = Path(agent_config.workspace).resolve()
    resolved = (workspace / path).resolve()
    if not (resolved == workspace or workspace in resolved.parents):
        return _err(
            f"view_image: '{path}' resolves outside the workspace "
            f"({resolved}); refusing. Keep images inside the workspace."
        )

    # 3. Existence / file-ness / non-empty.
    if not resolved.exists():
        return _err(
            f"view_image: file not found: '{path}' (resolved: {resolved}). "
            f"Check the path with shell/glob and retry."
        )
    if not resolved.is_file():
        return _err(
            f"view_image: '{path}' is not a regular file (resolved: {resolved})."
        )
    try:
        size = resolved.stat().st_size
    except OSError as e:
        return _err(f"view_image: could not stat '{path}': {e}")
    if size == 0:
        return _err(
            f"view_image: '{path}' is empty (0 bytes) — nothing to view. "
            f"Re-create or re-download the image."
        )

    # 4. Extension -> MIME (case-insensitive).
    ext = resolved.suffix.lstrip(".").lower()
    mime = _EXT_TO_MIME.get(ext)
    if mime is None:
        supported = ", ".join(sorted({m.split("/")[1].upper() for m in _EXT_TO_MIME.values()}))
        return _err(
            f"view_image: '{path}' is not a supported image "
            f"(extension '{ext or '(none)'}'). Supported types: {supported}. "
            f"Convert it first (e.g. via shell: convert/magick) or pass a JPEG/PNG/GIF/WebP."
        )

    # 5. Size cap.
    max_bytes = int(tool_config.get("max_bytes", 5_242_880))
    if size > max_bytes:
        mb = max_bytes / 1_048_576
        return _err(
            f"view_image: image exceeds the {mb:g} MB cap ({size} bytes > "
            f"{max_bytes}). Downscale/convert it via shell first (e.g. "
            f"convert/magick to a smaller JPEG), then view the smaller file."
        )

    # 6. Model vision gate — refuse cleanly on a blind model, naming it.
    active_model = (callbacks or {}).get("active_model") or agent_config.default_model
    if not model_supports_vision(active_model, agent_config):
        return _err(
            f"view_image: the active model '{active_model}' does not support "
            f"image input — the image would be dropped or rejected by the provider. "
            f"Switch to a vision-capable model with /model, then retry."
        )

    # Success — stage the tag (workspace-relative path, as given).
    tag = f"[media: {path} ({mime}, {size} B)]"
    await deposit(tag)
    from openalph.tools import ToolResult
    return ToolResult(
        content=(
            f"Image staged: {path} ({mime}, {size} B). The image will be provided "
            f"before the next model call as a user message. Do not call view_image "
            f"again for the same image this turn — it is already queued."
        ),
        is_error=False,
    )


def frame_vision_batch(tags: list[str]) -> str:
    """Frame a drained batch of [media:] tags into ONE user-message string.

    Returns "" for an empty batch. Otherwise a header line naming the count and
    the workspace-relative paths (extracted from the tags), followed by one tag
    per line. agent.py expands the tags via ``_build_user_content``; the header
    survives as the text block alongside the image blocks.
    """
    if not tags:
        return ""
    from openalph.agent import MEDIA_TAG_RE  # local: agent imports tools package

    paths = []
    for t in tags:
        m = MEDIA_TAG_RE.search(t)
        paths.append(m.group(1) if m else t)
    header = f"[view_image tool output — {len(tags)} image(s): {', '.join(paths)}]"
    return header + "\n" + "\n".join(tags)
