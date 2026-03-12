"""Tests for mention detection logic.

Interface contract:
    MentionCheckResult(mentioned: bool, method: str)
    mentions_me(user_id: str, event_source: dict, body: str) -> MentionCheckResult

    Detection layers (checked in order):
    1. Structured: m.mentions.user_ids contains agent user_id
    2. Body full ID: body contains "@watson:matrix.local"
    3. Body localpart: body contains "@watson" with word boundary
    
    The function is pure — no Matrix client or room state needed.
"""

import pytest
from openalph.mention import MentionCheckResult, mentions_me, strip_mention


USER_ID = "@watson:matrix.local"


class TestMentionCheckResult:
    """MentionCheckResult is a simple dataclass."""

    def test_mentioned_true(self):
        result = MentionCheckResult(mentioned=True, method="structured")
        assert result.mentioned is True
        assert result.method == "structured"

    def test_mentioned_false(self):
        result = MentionCheckResult(mentioned=False, method="none")
        assert result.mentioned is False
        assert result.method == "none"


class TestStructuredMentions:
    """Layer 1: m.mentions.user_ids in event content."""

    def test_structured_mention_present(self):
        """Agent user_id in m.mentions.user_ids → mentioned, method=structured."""
        event_source = {
            "content": {
                "m.mentions": {
                    "user_ids": [USER_ID]
                }
            }
        }
        result = mentions_me(USER_ID, event_source, "Hey watson")
        assert result.mentioned is True
        assert result.method == "structured"

    def test_structured_mention_multiple_users(self):
        """Agent user_id among multiple mentioned users → mentioned."""
        event_source = {
            "content": {
                "m.mentions": {
                    "user_ids": ["@alice:matrix.local", USER_ID, "@bob:matrix.local"]
                }
            }
        }
        result = mentions_me(USER_ID, event_source, "Hey everyone")
        assert result.mentioned is True
        assert result.method == "structured"

    def test_structured_mention_other_user(self):
        """Different user in m.mentions.user_ids → falls through to body check."""
        event_source = {
            "content": {
                "m.mentions": {
                    "user_ids": ["@alice:matrix.local"]
                }
            }
        }
        result = mentions_me(USER_ID, event_source, "Hey alice")
        assert result.mentioned is False
        assert result.method == "none"

    def test_structured_mention_empty_user_ids(self):
        """Empty user_ids list → falls through to body check."""
        event_source = {
            "content": {
                "m.mentions": {
                    "user_ids": []
                }
            }
        }
        result = mentions_me(USER_ID, event_source, "Hello")
        assert result.mentioned is False
        assert result.method == "none"

    def test_structured_mention_no_m_mentions(self):
        """No m.mentions key → falls through to body check."""
        event_source = {
            "content": {}
        }
        result = mentions_me(USER_ID, event_source, "Hello")
        assert result.mentioned is False
        assert result.method == "none"

    def test_structured_mention_no_content(self):
        """No content key in event source → falls through to body check."""
        event_source = {}
        result = mentions_me(USER_ID, event_source, "Hello")
        assert result.mentioned is False
        assert result.method == "none"


class TestBodyFullIdMention:
    """Layer 2a: full user_id string in message body."""

    def test_body_full_id_present(self):
        """Body contains full user_id → mentioned, method=body_user_id."""
        event_source = {"content": {}}
        body = "Hey @watson:matrix.local what do you think?"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_user_id"

    def test_body_full_id_at_start(self):
        """Full user_id at start of body → mentioned."""
        event_source = {"content": {}}
        body = "@watson:matrix.local can you help?"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_user_id"

    def test_body_full_id_at_end(self):
        """Full user_id at end of body → mentioned."""
        event_source = {"content": {}}
        body = "What do you think @watson:matrix.local"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_user_id"


class TestBodyLocalpartMention:
    """Layer 2b: @localpart with word boundary in message body."""

    def test_body_localpart_with_space(self):
        """@localpart followed by space → mentioned, method=body_localpart."""
        event_source = {"content": {}}
        body = "Hey @watson what do you think?"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_localpart"

    def test_body_localpart_at_end(self):
        """@localpart at end of string → mentioned."""
        event_source = {"content": {}}
        body = "What do you think @watson"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_localpart"

    def test_body_localpart_with_comma(self):
        """@localpart followed by comma → mentioned."""
        event_source = {"content": {}}
        body = "@watson, what's the status?"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_localpart"

    def test_body_localpart_with_exclamation(self):
        """@localpart followed by ! → mentioned."""
        event_source = {"content": {}}
        body = "Great work @watson!"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_localpart"

    def test_body_localpart_with_question(self):
        """@localpart followed by ? → mentioned."""
        event_source = {"content": {}}
        body = "Can you help @watson?"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_localpart"

    def test_body_localpart_with_colon(self):
        """@localpart followed by : → mentioned."""
        event_source = {"content": {}}
        body = "@watson: please check this"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_localpart"

    def test_body_localpart_with_semicolon(self):
        """@localpart followed by ; → mentioned."""
        event_source = {"content": {}}
        body = "Ask @watson; they know"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_localpart"

    def test_body_localpart_with_period(self):
        """@localpart followed by . → mentioned."""
        event_source = {"content": {}}
        body = "Thanks @watson."
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_localpart"

    def test_body_localpart_partial_no_match(self):
        """@watsonville should NOT match @watson (partial word)."""
        event_source = {"content": {}}
        body = "Talk to @watsonville about this"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is False
        assert result.method == "none"

    def test_body_localpart_substring_no_match(self):
        """watson without @ prefix should NOT match."""
        event_source = {"content": {}}
        body = "Ask watson about this"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is False
        assert result.method == "none"


class TestEdgeCases:
    """Edge cases and priority ordering."""

    def test_empty_body(self):
        """Empty body → not mentioned."""
        event_source = {"content": {}}
        result = mentions_me(USER_ID, event_source, "")
        assert result.mentioned is False
        assert result.method == "none"

    def test_none_body(self):
        """None body → not mentioned (should not crash)."""
        event_source = {"content": {}}
        result = mentions_me(USER_ID, event_source, None)
        assert result.mentioned is False
        assert result.method == "none"

    def test_structured_takes_priority_over_body(self):
        """When both structured and body match, method=structured (primary wins)."""
        event_source = {
            "content": {
                "m.mentions": {
                    "user_ids": [USER_ID]
                }
            }
        }
        body = "Hey @watson:matrix.local what do you think?"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "structured"

    def test_full_id_takes_priority_over_localpart(self):
        """When body contains full ID, method=body_user_id (not body_localpart)."""
        event_source = {"content": {}}
        body = "Hey @watson:matrix.local"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_user_id"

    def test_case_sensitive_structured(self):
        """Structured mentions are case-sensitive (Matrix user IDs are case-sensitive)."""
        event_source = {
            "content": {
                "m.mentions": {
                    "user_ids": ["@Watson:matrix.local"]  # capital W
                }
            }
        }
        result = mentions_me(USER_ID, event_source, "Hello")
        assert result.mentioned is False
        assert result.method == "none"

    def test_case_sensitive_body(self):
        """Body mentions are case-sensitive."""
        event_source = {"content": {}}
        body = "Hey @Watson what do you think?"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is False
        assert result.method == "none"

    def test_body_with_only_whitespace(self):
        """Whitespace-only body → not mentioned."""
        event_source = {"content": {}}
        result = mentions_me(USER_ID, event_source, "   \n\t  ")
        assert result.mentioned is False
        assert result.method == "none"

    def test_mention_in_multiline_body(self):
        """Mention can be on any line of a multiline message."""
        event_source = {"content": {}}
        body = "First line\nSecond line\n@watson what about this?\nFourth line"
        result = mentions_me(USER_ID, event_source, body)
        assert result.mentioned is True
        assert result.method == "body_localpart"


# --- strip_mention tests ---

class TestStripMentionFullId:
    """Strip full user_id (@watson:matrix.local) from body."""

    def test_full_id_at_start(self):
        """@watson:matrix.local /status → /status"""
        result = strip_mention(USER_ID, "@watson:matrix.local /status")
        assert result == "/status"

    def test_full_id_at_start_with_colon(self):
        """@watson:matrix.local: /status → /status"""
        result = strip_mention(USER_ID, "@watson:matrix.local: /status")
        assert result == "/status"

    def test_full_id_only(self):
        """@watson:matrix.local → empty string."""
        result = strip_mention(USER_ID, "@watson:matrix.local")
        assert result == ""

    def test_full_id_with_text_after(self):
        """@watson:matrix.local what do you think? → what do you think?"""
        result = strip_mention(USER_ID, "@watson:matrix.local what do you think?")
        assert result == "what do you think?"

    def test_full_id_in_middle(self):
        """hey @watson:matrix.local help → hey help"""
        result = strip_mention(USER_ID, "hey @watson:matrix.local help")
        assert result == "hey help"


class TestStripMentionLocalpart:
    """Strip @localpart from body."""

    def test_localpart_at_start(self):
        """@watson /status → /status"""
        result = strip_mention(USER_ID, "@watson /status")
        assert result == "/status"

    def test_localpart_at_start_with_colon(self):
        """@watson: /status → /status"""
        result = strip_mention(USER_ID, "@watson: /status")
        assert result == "/status"

    def test_localpart_at_start_with_comma(self):
        """@watson, /status → /status"""
        result = strip_mention(USER_ID, "@watson, /status")
        assert result == "/status"

    def test_localpart_only(self):
        """@watson → empty string."""
        result = strip_mention(USER_ID, "@watson")
        assert result == ""

    def test_localpart_with_text(self):
        """@watson what is the status? → what is the status?"""
        result = strip_mention(USER_ID, "@watson what is the status?")
        assert result == "what is the status?"


class TestStripMentionNoMatch:
    """Body with no mention of this agent."""

    def test_no_mention_returns_unchanged(self):
        """/status without mention → /status unchanged."""
        result = strip_mention(USER_ID, "/status")
        assert result == "/status"

    def test_other_agent_mention(self):
        """@babson /status → @babson /status (not stripped)."""
        result = strip_mention(USER_ID, "@babson /status")
        assert result == "@babson /status"

    def test_empty_body(self):
        """Empty string → empty string."""
        result = strip_mention(USER_ID, "")
        assert result == ""


class TestStripMentionEdgeCases:
    """Edge cases for strip_mention."""

    def test_full_id_preferred_over_localpart(self):
        """When full ID present, strip that (not just localpart)."""
        result = strip_mention(USER_ID, "@watson:matrix.local /model foo")
        assert result == "/model foo"

    def test_only_first_occurrence_stripped(self):
        """Only the first mention is removed."""
        result = strip_mention(USER_ID, "@watson @watson help")
        assert result == "@watson help"

    def test_multiword_command(self):
        """@watson /model openrouter/kimi-k2.5 → /model openrouter/kimi-k2.5"""
        result = strip_mention(USER_ID, "@watson /model openrouter/kimi-k2.5")
        assert result == "/model openrouter/kimi-k2.5"

    def test_heartbeat_command(self):
        """@watson /heartbeat start 6h → /heartbeat start 6h"""
        result = strip_mention(USER_ID, "@watson /heartbeat start 6h")
        assert result == "/heartbeat start 6h"

    def test_whitespace_normalized(self):
        """Extra whitespace around mention is cleaned up."""
        result = strip_mention(USER_ID, "  @watson   /status  ")
        assert result == "/status"

    def test_partial_localpart_not_stripped(self):
        """@watsonville should NOT be stripped for @watson."""
        result = strip_mention(USER_ID, "@watsonville /status")
        assert result == "@watsonville /status"
