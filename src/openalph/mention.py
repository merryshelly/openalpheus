"""Mention detection and room gating for shared rooms.

Pure logic — no Matrix client or async dependencies.

Public API:
    MentionCheckResult — dataclass with (mentioned: bool, method: str)
    mentions_me(user_id, event_source, body) -> MentionCheckResult
    is_gated(config, room) -> bool
"""

import re
from dataclasses import dataclass


@dataclass
class MentionCheckResult:
    """Result of checking whether an event mentions this agent.
    
    Attributes:
        mentioned: True if the agent was mentioned
        method: How the mention was detected:
            "structured" — m.mentions.user_ids
            "body_user_id" — full @user:server in body
            "body_localpart" — @localpart with word boundary in body
            "none" — not mentioned
    """
    mentioned: bool
    method: str


def mentions_me(user_id: str, event_source: dict, body: str | None) -> MentionCheckResult:
    """Check whether a Matrix event mentions the given user.
    
    Two-layer detection, checked in order:
    1. Structured: m.mentions.user_ids in event content
    2. Body text: full user_id or @localpart with word boundary
    
    Args:
        user_id: The agent's Matrix user ID (e.g. "@watson:matrix.local")
        event_source: The raw event source dict (event.source)
        body: The message body text (may be None or empty)
        
    Returns:
        MentionCheckResult with mentioned=True/False and detection method
    """
    # Layer 1 — Structured mentions
    content = event_source.get("content", {})
    mentions = content.get("m.mentions", {})
    user_ids = mentions.get("user_ids", [])
    if user_id in user_ids:
        return MentionCheckResult(mentioned=True, method="structured")
    
    # Layer 2 — Body text fallback
    if body:
        # Check if full user_id appears in body
        if user_id in body:
            return MentionCheckResult(mentioned=True, method="body_user_id")
        
        # Extract localpart (e.g., "@watson" from "@watson:matrix.local")
        if ":" in user_id:
            localpart = user_id.split(":")[0]  # Everything before the first colon
        else:
            localpart = user_id
        
        # Check if @localpart appears with word boundary
        # Word boundary: end-of-string, whitespace, or punctuation [,;:!?.]
        escaped_localpart = re.escape(localpart)
        pattern = rf"{escaped_localpart}(?=$|\s|[,;:!?.])"
        if re.search(pattern, body):
            return MentionCheckResult(mentioned=True, method="body_localpart")
    
    return MentionCheckResult(mentioned=False, method="none")


def is_gated(config, room) -> bool:
    """Determine if mention gating is active for a room.
    
    Priority:
    1. TOML override: config.rooms[room.room_id]["require_mention"]
    2. Auto-detect: len(room.users) >= 3 → gated
    
    Args:
        config: MatrixConfig instance (needs .rooms attribute)
        room: Room object (needs .room_id and .users attributes)
        
    Returns:
        True if the room requires @mention for agent response
    """
    room_id = room.room_id
    
    # 1. Check TOML override
    room_overrides = config.rooms or {}
    if room_id in room_overrides:
        override = room_overrides[room_id]
        if "require_mention" in override:
            return override["require_mention"]
    
    # 2. Auto-detect from member count
    member_count = len(room.users) if hasattr(room, 'users') else 0
    return member_count >= 3
