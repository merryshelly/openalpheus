"""Credential redaction module for OpenAlph.

Scans tool output text for credential patterns and replaces them with
[REDACTED:<type>] markers before the text is returned to the model.
The original secret value never enters the model's context window.

Architecture:
    CREDENTIAL_PATTERNS — ordered list of compiled regex patterns
    RedactionEvent      — dataclass recording each individual redaction
    redact_credentials  — applies patterns sequentially; specific first,
                          generic last, preventing double-redaction
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("openalph.security")


@dataclass
class RedactionEvent:
    """Records a single credential redaction performed by redact_credentials."""
    pattern_name: str
    redaction_label: str
    char_count: int
    position: int


# Patterns are applied in order.  More-specific patterns come first so that
# their replacements prevent the later generic patterns from firing on the
# same text.
CREDENTIAL_PATTERNS: list[dict] = [
    {
        "name": "pem_private_key",
        "pattern": re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
        ),
        "redaction_label": "[REDACTED:private_key]",
    },
    {
        "name": "anthropic_api_key",
        "pattern": re.compile(r"sk-ant-[a-zA-Z0-9_-]{8,}"),
        "redaction_label": "[REDACTED:api_key]",
    },
    {
        "name": "openrouter_api_key",
        "pattern": re.compile(r"sk-or-v1-[a-f0-9]{64}"),
        "redaction_label": "[REDACTED:api_key]",
    },
    {
        "name": "openai_api_key",
        "pattern": re.compile(r"\bsk-(?!ant-)(?!or-)[a-zA-Z0-9_-]{20,}"),
        "redaction_label": "[REDACTED:api_key]",
    },
    {
        "name": "github_token",
        "pattern": re.compile(
            r"(?:ghp_|gho_|ghs_|ghr_|github_pat_)[a-zA-Z0-9_]{20,}"
        ),
        "redaction_label": "[REDACTED:token]",
    },
    {
        "name": "onepassword_service_token",
        "pattern": re.compile(r"ops_[a-zA-Z0-9_-]{20,}"),
        "redaction_label": "[REDACTED:service_token]",
    },
    {
        "name": "age_secret_key",
        "pattern": re.compile(r"AGE-SECRET-KEY-[A-Z0-9]{56,}"),
        "redaction_label": "[REDACTED:age_secret_key]",
    },
    {
        "name": "bearer_token",
        "pattern": re.compile(r"Bearer [a-zA-Z0-9_/+.\-]{20,}"),
        "redaction_label": "[REDACTED:bearer_token]",
    },
    {
        "name": "ethereum_private_key",
        "pattern": re.compile(r"0x[0-9a-fA-F]{64}\b"),
        "redaction_label": "[REDACTED:private_key]",
    },
    {
        "name": "generic_hex",
        "pattern": re.compile(r"(?<!<)\b[0-9a-fA-F]{48,}\b"),
        "redaction_label": "[REDACTED:hex_secret]",
    },
]


def redact_credentials(text: str) -> tuple[str, list[RedactionEvent]]:
    """Scan text for credential patterns and replace with redaction markers.

    Patterns are applied sequentially in order so that specific patterns
    fire before generic ones, preventing double-redaction.  Each pass
    operates on the text already modified by earlier passes.

    Args:
        text: Text to scan and redact.

    Returns:
        Tuple of (redacted_text, list_of_RedactionEvent).
        If no credentials are found the original text and an empty list
        are returned without any modification.
    """
    if not text:
        return (text, [])

    all_events: list[RedactionEvent] = []
    current_text = text

    for pattern_def in CREDENTIAL_PATTERNS:
        pattern_name: str = pattern_def["name"]
        pattern: re.Pattern = pattern_def["pattern"]
        label: str = pattern_def["redaction_label"]

        matches = list(pattern.finditer(current_text))
        if not matches:
            continue

        for match in matches:
            char_count = len(match.group())
            position = match.start()
            all_events.append(RedactionEvent(
                pattern_name=pattern_name,
                redaction_label=label,
                char_count=char_count,
                position=position,
            ))
            # NEVER include the actual secret value in log messages.
            logger.warning(
                "Credential redacted: %s (%d chars at position %d)",
                pattern_name,
                char_count,
                position,
            )

        # Replace all matches in the current text before moving to the next
        # pattern, so later (more generic) patterns cannot re-match the same
        # content after it has already been redacted.
        current_text = pattern.sub(label, current_text)

    return (current_text, all_events)
