"""Tests for room gating logic.

Interface contract:
    is_gated(config: MatrixConfig, room) -> bool

    Priority:
    1. TOML override: config.rooms["!roomid:server"]["require_mention"] = true/false
    2. Auto-detect: len(room.users) >= 3 → gated

    The function takes a MatrixConfig and a room object (anything with
    .room_id: str and .users: dict attributes).
"""

import pytest
from unittest.mock import MagicMock
from openalph.mention import is_gated
from openalph.config import MatrixConfig


def make_matrix_config(rooms=None, **kwargs):
    """Create a MatrixConfig with optional room overrides."""
    defaults = dict(
        homeserver="https://matrix.local",
        user_id="@watson:matrix.local",
        device_id="TEST",
        password="test",
        access_token=None,
        context_reserve=16384,
        sync_timeout=30000,
        retry_base=1,
        retry_max=10,
        rooms=rooms,
    )
    defaults.update(kwargs)
    return MatrixConfig(**defaults)


def make_room(room_id, member_count):
    """Create a mock room with N members.
    
    room.users is a dict (Matrix nio convention: {user_id: Member}).
    """
    room = MagicMock()
    room.room_id = room_id
    room.users = {f"@user{i}:matrix.local": MagicMock() for i in range(member_count)}
    return room


class TestAutoDetection:
    """Auto-detect gating from room member count."""

    def test_2_members_not_gated(self):
        """2-member room (DM) → not gated."""
        config = make_matrix_config()
        room = make_room("!dm:matrix.local", 2)
        assert is_gated(config, room) is False

    def test_3_members_gated(self):
        """3-member room → gated."""
        config = make_matrix_config()
        room = make_room("!group:matrix.local", 3)
        assert is_gated(config, room) is True

    def test_10_members_gated(self):
        """Large room → gated."""
        config = make_matrix_config()
        room = make_room("!large:matrix.local", 10)
        assert is_gated(config, room) is True

    def test_1_member_not_gated(self):
        """1-member room (edge case) → not gated."""
        config = make_matrix_config()
        room = make_room("!solo:matrix.local", 1)
        assert is_gated(config, room) is False

    def test_0_members_not_gated(self):
        """0-member room (shouldn't happen) → not gated."""
        config = make_matrix_config()
        room = make_room("!empty:matrix.local", 0)
        assert is_gated(config, room) is False


class TestTomlOverride:
    """TOML [matrix.rooms] overrides take priority over auto-detection."""

    def test_override_true_on_2_member_room(self):
        """require_mention=true on DM → gated (override wins)."""
        config = make_matrix_config(rooms={
            "!dm:matrix.local": {"require_mention": True}
        })
        room = make_room("!dm:matrix.local", 2)
        assert is_gated(config, room) is True

    def test_override_false_on_3_member_room(self):
        """require_mention=false on group → not gated (override wins)."""
        config = make_matrix_config(rooms={
            "!group:matrix.local": {"require_mention": False}
        })
        room = make_room("!group:matrix.local", 3)
        assert is_gated(config, room) is False

    def test_override_only_affects_matching_room(self):
        """Override for room A doesn't affect room B."""
        config = make_matrix_config(rooms={
            "!roomA:matrix.local": {"require_mention": False}
        })
        room_b = make_room("!roomB:matrix.local", 3)
        assert is_gated(config, room_b) is True  # auto-detect

    def test_override_none_rooms(self):
        """rooms=None → auto-detect only."""
        config = make_matrix_config(rooms=None)
        room = make_room("!group:matrix.local", 3)
        assert is_gated(config, room) is True

    def test_override_empty_rooms(self):
        """rooms={} (empty dict) → auto-detect only."""
        config = make_matrix_config(rooms={})
        room = make_room("!group:matrix.local", 3)
        assert is_gated(config, room) is True

    def test_override_room_without_require_mention(self):
        """Room entry exists but has no require_mention key → auto-detect."""
        config = make_matrix_config(rooms={
            "!group:matrix.local": {"some_other_key": "value"}
        })
        room = make_room("!group:matrix.local", 3)
        assert is_gated(config, room) is True


class TestDynamicMemberChanges:
    """Gating responds to live member count changes."""

    def test_member_join_activates_gating(self):
        """Room goes from 2 → 3 members → gating activates."""
        config = make_matrix_config()
        room = make_room("!room:matrix.local", 2)
        assert is_gated(config, room) is False

        # Simulate 3rd member joining
        room.users["@newcomer:matrix.local"] = MagicMock()
        assert is_gated(config, room) is True

    def test_member_leave_deactivates_gating(self):
        """Room goes from 3 → 2 members → gating deactivates."""
        config = make_matrix_config()
        room = make_room("!room:matrix.local", 3)
        assert is_gated(config, room) is True

        # Simulate member leaving
        first_key = next(iter(room.users))
        del room.users[first_key]
        assert is_gated(config, room) is False
