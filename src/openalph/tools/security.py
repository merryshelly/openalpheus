"""Credential redaction module for OpenAlph.

Scans tool output text for credential patterns and replaces them with
[REDACTED:<type>] markers before the text is returned to the model.
The original secret value never enters the model's context window.

Architecture:
    CREDENTIAL_PATTERNS      — ordered list of compiled regex patterns
    RedactionEvent           — dataclass recording each individual redaction
    redact_credentials       — applies patterns sequentially; specific first,
                               generic last, preventing double-redaction
    redact_known_secrets     — L1 value-based backstop: redacts literal
                               occurrences of already-known secret values
                               (e.g. resolved provider API keys) that the
                               pattern pass above cannot recognise by shape.
                               Stateless — the known-value set is passed in
                               by the caller (tools/__init__.py); this module
                               never imports config or holds secrets itself.
    op_egress_block_reason   — L2 pure predicate (round-5, deny-by-default):
                               splits COMMAND SUBSTITUTIONS off the raw
                               string first (`$(...)` and backticks, quote-
                               aware), trusting one and only one shape —
                               an assignment capture, `NAME=$(...)`, whose
                               value lands in a shell variable rather than
                               on stdout. EVERY other substitution is
                               re-analysed as a command in its own right,
                               as is shell code passed to a `-c`-shaped
                               flag or to `eval`. What remains is tokenized
                               quote-aware, merged shell-punctuation tokens
                               are split, the result is cut into top-level
                               segments on `;`/`&&`/`||`/`&`/`|`/newline,
                               and any SEGMENT containing an op-egress
                               keyword combination (op read / opread / op
                               document get / op item get --fields|--field|
                               --format|--reveal|--otp) is BLOCKED, with a
                               redirect message returned in place of the
                               secret. Pipe is just another separator: a
                               piped op-egress command is BLOCKED.

                               Round-5 replaces round-4's excise-and-trust
                               stance, which fixed the guard's failures one
                               decoy shape at a time and still fell open
                               four ways (SEC-1 `echo $(op read …)`, SEC-2
                               backticks, SEC-3 `sh -c '…'`, and quoted
                               `echo "$(op read …)"`). The inversion is the
                               point: round-4 asked "is this construct one
                               I recognise as dangerous?", round-5 asks "is
                               this construct provably a capture?" — and
                               scans everything else.
"""

import logging
import os
import re
import shlex
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


_KNOWN_SECRET_LABEL = "[REDACTED:known_secret]"


def redact_known_secrets(
    text: str,
    known_values: set[str],
    min_length: int = 12,
) -> tuple[str, list[RedactionEvent]]:
    """Redact literal occurrences of already-known secret values (L1 backstop).

    This is a value-based pass, distinct from (and run AFTER) the shape-based
    ``redact_credentials`` pass above: it does not recognise a credential by
    what it looks like, it recognises it because the caller already knows the
    exact string is live (a resolved provider API key, a cached ``op read``
    result, etc). It exists to catch secrets the pattern list above cannot
    possibly match by shape (e.g. an arbitrary password or token with no
    recognisable prefix), so it deliberately runs on text that has already
    had ``redact_credentials`` applied to it.

    Stateless: this function holds no config/secret state itself — the
    known-value set is supplied in full by the caller on every call.

    Backstop semantics: a known value is redacted even when it appears only
    as a substring of a larger token. The value IS the value; the min_length
    floor plus the caller's own high-entropy secrets make accidental
    collisions negligible. This is intentional, not a bug.

    Args:
        text: Text to scan and redact (normally already pattern-redacted).
        known_values: Set of literal secret strings currently considered
            live/sensitive (e.g. provider API keys, cached resolved keys).
        min_length: Minimum length a known value must have to be eligible
            for redaction (default 12). Shorter values are skipped so that
            small/common substrings (e.g. a short fixture value) are never
            over-redacted.

    Returns:
        Tuple of (redacted_text, list_of_RedactionEvent). Empty text or an
        empty known_values set is a no-op: the original text and an empty
        event list are returned unchanged.
    """
    if not text or not known_values:
        return (text, [])

    # Longest-first so that a shorter secret which happens to be a substring
    # of a longer one cannot leave fragments of the longer secret behind
    # (redact the longer match first, consuming the shorter one within it).
    eligible = sorted(
        (v for v in known_values if v and len(v) >= min_length),
        key=len,
        reverse=True,
    )
    if not eligible:
        return (text, [])

    all_events: list[RedactionEvent] = []
    current_text = text

    for value in eligible:
        if value not in current_text:
            continue

        # Record one event per literal occurrence, positions computed against
        # the text as it stands before this value's replacements are applied.
        start = 0
        while True:
            idx = current_text.find(value, start)
            if idx == -1:
                break
            all_events.append(RedactionEvent(
                pattern_name="known_secret",
                redaction_label=_KNOWN_SECRET_LABEL,
                char_count=len(value),
                position=idx,
            ))
            # NEVER include the actual secret value in log messages.
            logger.warning(
                "Credential redacted: %s (%d chars at position %d)",
                "known_secret",
                len(value),
                idx,
            )
            start = idx + len(value)

        current_text = current_text.replace(value, _KNOWN_SECRET_LABEL)

    return (current_text, all_events)


# --- L2: shell op-egress guard (round-2 — presence-based rewrite) ----------
#
# Round-1 (see remediation-report.md) segmented a command into COMMANDS
# (split on `;`/`&&`/`||`/`&`/newline) then PIPE STAGES (split on `|`),
# evaluated only each command's LAST stage, and decided egress by stripping
# a fixed set of wrapper shapes (env-var assignments, then a leading
# `sudo`/`env`) before checking a hard-coded POSITIONAL binary/subcommand
# shape. A round-2 re-audit found this still fails open in several corners:
# shlex merging adjacent punctuation into one token (e.g. `"&&\n"`) hid a
# real separator from the splitter; a `sudo -u root op read ...` (an extra
# `-u root` pair of tokens before the real binary) was never unwrapped
# because only a FIXED wrapper shape was stripped; boolean global `op`
# flags, `time`/`nohup`/other wrappers, and `(op read x)` subshell groups
# all defeated the positional assumption in one way or another; and the
# pipe-allow policy (skip the last-but-one stage) meant a piped secret could
# still resurface via the pipe's consumer.
#
# This rewrite inverts the failure mode: instead of trying to strip away
# every possible wrapper shape down to "the" binary and "the" subcommand
# tokens (an open-ended, always-incomplete list), it scans an entire
# top-level SEGMENT for the PRESENCE of an op-egress keyword combination,
# after only two structural operations: (1) splitting any shlex token that
# is itself a merged run of several punctuation operators back into its
# individual operators, and (2) excising a genuine `$(...)` substitution
# span (whose captured value lands in a shell variable or another
# command's stdin, never directly in stdout) before segmenting on
# `;`/`&&`/`||`/`&`/`|`/newline. Pipe is now just another separator — a
# piped op-egress command BLOCKS like any other segment.
#
# Round-4 (H1) update: `<(...)` process substitution is NO LONGER treated
# as an excision opener at all (round-2/round-3 both excised it exactly
# like a genuine `$(...)` span). `<(` is dash/POSIX-sh syntax the real
# shell doesn't even support as a command substitution — a shell running
# under `/bin/sh -c` either rejects it outright or (bash) treats it as a
# process substitution whose OWN stdout is a separate fd, never this
# command's stdout — but a quoted DECOY (`echo '<('`) is emitted by shlex
# as an indistinguishable bare `<(` token, and letting it act as an
# opener let a later literal `)` (from an unrelated, subsequent `echo
# ')'`) close a synthetic span across a real top-level `op read` in
# between, excising (hiding) it (round-4 H1, empirically confirmed
# fail-open). Leaving `<(` as an ordinary, always-scannable token is
# strictly fail-closed and does not touch the dominant safe
# `VAR=$(opread ...)` pattern (which never uses `<(` at all). Round-5 makes
# this moot for `$(...)`/backticks -- they are now split off the raw string
# before tokenization by `_split_command_substitutions`, which reads quote
# state directly and so cannot be fooled by a quoted decoy at all -- but
# `<(` is still deliberately left as an ordinary, scannable token.

# op item get flags that dump the full/raw field set (rather than a masked
# overview) to stdout. Matched whole-token or as the LHS of a `--flag=value`
# form — never a substring, so `--show-all-fields` does not trigger on
# `--fields`. `--field` (singular) is a real, separate `op item get` flag
# (verified against `op item get --help` on the real binary at
# `/usr/local/bin/op` — round-4 H3; `--help`'s own flag list only documents
# `--fields` (plural), but `--field` singular is ALSO accepted by the real
# binary and dumps the same way, confirmed empirically: `op item get X
# --field password` reaches the same account-check dispatch as `--fields`,
# and an invalid unknown flag like `--fieldx`/`--fie` is rejected by `op`
# itself with "unknown flag" — proving `--field` is matched as its own
# exact flag, not a fuzzy/prefix match of `--fields`).
_OP_ITEM_GET_DUMP_FLAGS = frozenset({
    "--fields", "--field", "--format", "--reveal", "--otp",
})

# Round-3 (M-1): the set of KNOWN op subcommand keywords used to locate THE
# subcommand precisely, rather than flagging egress on ANY later `read`
# token regardless of what it actually is (an item name, a template path, a
# child-command argument, ...). Deliberately a superset of what this guard
# actually classifies below — `run`/`inject`/`vault`/`account`/etc. all just
# need to be recognised as "yes, this is where the real op subcommand is",
# so THE FIRST known-subcommand token found ends the scan and every token
# before it (global flags like `--account`/`--debug`, their values,
# `env`-style assignments) is correctly ignored without needing a
# flag-arity table (this also fixes the round-2 `op --debug read x` /
# `op --account prod read x` cases — those already worked because `read`
# happened to still be present in "later tokens", but this scan makes that
# robust rather than incidental).
_KNOWN_OP_SUBCOMMANDS = frozenset({
    "read", "item", "document", "inject", "run", "vault", "account", "user",
    "group", "connect", "plugin", "signin", "signout", "whoami", "completion",
    "update", "service-account", "events-api", "environment",
})

# Round-3 (M-1 ambiguous-flag safety net): GLOBAL (pre-subcommand) op flags
# that consume a following VALUE token, whose value could — in the worst
# case — coincidentally collide with a `_KNOWN_OP_SUBCOMMANDS` keyword and
# so be mistaken for the real subcommand (e.g. `op --account item read
# op://x`: `--account`'s VALUE is `item`, not a subcommand at all; the real
# subcommand is `read`, one token further along). Deliberately a SMALL,
# explicit set of just the value-taking GLOBAL flags (not a general
# per-subcommand flag-arity table, which is exactly the kind of
# always-incomplete machinery the spec says to avoid) — `--account` is the
# one already covered by this guard's own existing test suite
# (`op_global_flag_read`). See `_resolve_op_subcommand_index`'s use of this
# set for how it's applied: only to re-resolve past a value immediately
# following one of these flags, never to model any other flag's arity.
#
# Round-4 (H2): `--config` and `--session` added (`--account` kept).
# Verified against the real binary's own `op --help` Global Flags section
# (`/usr/local/bin/op --help`, `op` version 2.32.0):
#
#   --account account    Select the account ... (already modelled)
#   --config directory   Use this configuration directory.
#   --encoding type       Use this character encoding type. ...
#   --session token       Authenticate with this session token. ...
#
# All three of `--account`/`--config`/`--session` are documented as taking
# a space-separated value (`account`/`directory`/`token` placeholders, not
# a boolean), and this was independently confirmed by actually invoking the
# real binary: `op --config item read op://x` dispatches to `read` (not
# `item`) — stderr shows `Using configuration at non-standard location
# "item"` before the (unrelated) secret-reference-shape error for `read`'s
# own argument, proving `--config` consumed the literal token `item` as
# its OWN value, leaving `read` as the actual subcommand one token later;
# `op --session x read op://x` likewise reaches `read`'s own dispatch
# (`could not read secret 'op://x': ...`), proving `--session` consumed
# `x` as its value the same way. Both are exactly the M-1
# flag-value/subcommand-keyword collision shape `--account` was already
# modelled for (`op --account item read op://x`) — `--config`/`--session`
# were simply missing from the set, so a value that happened to spell a
# `_KNOWN_OP_SUBCOMMANDS` keyword (e.g. `--config`'s value `item`, or
# `--config`'s value `run`) would stop `_resolve_op_subcommand_index`'s
# scan there instead of at the real subcommand one token further along —
# hiding a real `read`/`item get --fields` behind it (round-4 H2,
# empirically confirmed fail-open).
#
# `--encoding` was also checked (`op --encoding zzzMARKERzzz item list` →
# `zzzMARKERzzz is not a supported character-encoding`, proving it too
# consumes a following value token) but is NOT added here: its only
# documented values (`UTF-8`, `SHIFT_JIS`, `gbk`) can never collide with a
# `_KNOWN_OP_SUBCOMMANDS` keyword, so modelling it would add surface
# without closing any reachable fail-open — left out per the spec's own
# "add `--encoding` if it space-takes a value" being conditioned on actual
# need, and to keep this set the tightest one that closes a real gap
# (matches the guard's stated anti-flag-arity-table posture). `--format`
# (also global, also value-taking: `human-readable`|`json` only) is
# excluded for the identical reason — checked, no collision is reachable,
# and confirmed empirically: `op --format item read op://x` never reaches
# `read`'s dispatch at all because `op` itself rejects `item` as an
# invalid `--format` value before dispatch (`invalid argument "item" for
# "--format" flag: Value must be one of 'human-readable,json'`) — real
# `op` rejects the command outright, so there is no way for this shape to
# ever actually leak a secret; both are documented residuals, not gaps.
_OP_VALUE_TAKING_GLOBAL_FLAGS = frozenset({"--account", "--config", "--session"})

# The exact character set shlex is configured with as `punctuation_chars`
# (see _tokenize_shell_command). A token composed ENTIRELY of characters
# from this set is a "punctuation run" — possibly a shlex-MERGED sequence of
# several distinct shell operators glued into one token (shlex emits `&&`
# immediately followed by a newline, with no space between them, as ONE
# token `"&&\n"` — empirically verified, see remediation-report-r2.md) —
# and must be split back into its individual operators before segmenting.
_PUNCTUATION_CHARS = frozenset(";&|()<>\n")

# Two-character control operators that ARE segment separators.
_TWO_CHAR_SEPARATOR_OPS = frozenset({"&&", "||"})

# The two-character process-substitution "open" marker. Recognised by
# `_split_punctuation_run` as its own atomic 2-char operator token (so a
# merged run like `<(true)` still tokenizes sanely) but — round-4, H1 — NEVER
# a structural opener for substitution handling: a quoted decoy `<(` is
# indistinguishable from a real one once shlex has erased quote provenance,
# and letting it open a synthetic span let a later, unrelated `)` close it
# across real content in between. It is simply left as an ordinary,
# always-scannable token. (Round-5 moved `$(...)`/backtick handling ahead of
# shlex entirely — see `_split_command_substitutions` — so no token-level
# span matching remains for a decoy to attack.)
_PROCESS_SUBSTITUTION_OPEN = "<("

# The two paren characters: structural tokens (a substitution boundary, or —
# when bare, no preceding `$` — a subshell group whose contents stay
# depth-0 and inspectable). Handled as their own single-char tokens by
# `_split_punctuation_run`; NOTE this is NOT the redirect-operator handling
# (`<`/`>` runs, e.g. `>&`, `>>`, `<<<`) — those are deliberately consumed
# as opaque multi-char units instead of decomposed; see
# `_split_punctuation_run`'s docstring for the empirical reasoning (a naive
# per-character split of a merged redirect token like `">&"` would
# manufacture a spurious `&` segment-separator out of a plain fd-duplication
# redirect — e.g. `op 2>&1 read x` — that isn't really a command boundary).
_SINGLE_CHAR_PARENS = frozenset({"(", ")"})

# Final separator set used to split depth-0 (post-excision) tokens into
# top-level segments: the two-char logical ops, the single-char ops, and a
# bare newline token (matched separately in `_split_into_segments`, since a
# merged multi-newline run like `"\n\n"` must also count as one separator).
# Pipe (`|`) is included — round-2 drops the round-1 pipe-allow carve-out.
_SEGMENT_SEPARATOR_TOKENS = frozenset({";", "&&", "||", "&", "|"})

# Recursion budget for shell code nested inside command substitutions and
# code-taking flags. Exceeding it BLOCKS (fail closed) rather than allowing:
# a command nested this deep is pathological, not legitimate operator work.
_MAX_NESTING_DEPTH = 12

# Flags that take a string of SHELL CODE as their very next argument.
# Deliberately a FLAG-SHAPE rule, not a list of interpreter binary names:
# `sh -c`, `bash -lc`, `dash -c`, `zsh -c`, `busybox sh -c`, `python3 -c`,
# `xargs -I{} sh -c ...` and any future runner all match `-c` (alone or
# bundled, where `c` must be last since it is the option that consumes the
# following argument), so the guard does not need to keep a binary list in
# sync with reality -- the round-2/3 "always one flag behind" trap.
_CODE_TAKING_FLAG_RE = re.compile(r"^(?:-[A-Za-z]*c|--command|--eval)$")

_OP_EGRESS_BLOCK_MESSAGE = (
    "\u26d4 Blocked: this command prints a 1Password secret to stdout, "
    "placing it in your context. To USE a secret without seeing it: "
    "op-run -- <cmd that references the op:// ref as an env var>. To write "
    "it into a file: op inject -i <template> -o <file>. To capture it into "
    "a variable for a later command, use VAR=$(opread \"op://...\") (the "
    "value stays out of your context — it lands in the shell variable, "
    "never printed to stdout)."
)


def _conservative_fallback(command: str) -> str | None:
    """Begins-with fallback used only when the command has unparseable quotes.

    Rare path (unbalanced quotes are unusual in legitimate commands) — errs
    conservative (may over-block relative to the full tokenizer, never
    under-blocks the plain prefix forms) rather than attempting to guess
    tokenization of malformed input.
    """
    s = command.strip()
    if s.startswith(("op read", "opread", "op document get")):
        return _OP_EGRESS_BLOCK_MESSAGE
    if s.startswith("op item get") and any(
        flag in s for flag in _OP_ITEM_GET_DUMP_FLAGS
    ):
        return _OP_EGRESS_BLOCK_MESSAGE
    return None


def _tokenize_shell_command(command: str) -> list[str] | None:
    """Quote-aware tokenization of a shell command line.

    Uses ``shlex`` with punctuation-token support so ``;``, ``&&``, ``||``,
    ``|`` etc. are emitted as their own tokens while quoted content (which
    may itself contain any of those characters) stays intact as a single
    token. Returns None if the command has unbalanced quotes (caller falls
    back to the conservative begins-with check).

    Two deliberate departures from a bare ``punctuation_chars=True`` config,
    both empirically verified against the guard's own test cases before
    shipping (see remediation-report.md for the verification transcript):

    1. ``shlex``'s default ``whitespace`` set is ``" \\t\\r\\n"`` — it
       includes a bare newline, which means an UNQUOTED newline between two
       commands (e.g. a heredoc-free multi-line shell block) is silently
       swallowed as plain whitespace and produces no token at all. That
       would make ``"cd /app\\nop read ...\\n"`` tokenize as a single
       unsplit command list (first token ``cd``), letting a smuggled `op
       read` after a literal newline sail through unblocked — the same
       fail-open shape as the compound-command (`;`/`&&`/`||`) finding this
       rewrite closes. Fix: newline is added to ``punctuation_chars`` AND
       stripped from ``whitespace``, so it becomes its own separator token
       (quoted newlines are unaffected — they stay embedded in their quoted
       token, verified empirically).
    2. ``shlex``'s default ``commenters`` is ``"#"`` — a bare ``#`` starts a
       comment that silently truncates the rest of the tokenized line. A
       command like ``true # ; op read x`` would then vanish after the
       ``#``, hiding a live command from this guard (fail-open again, and
       untested by any spec case, but the same class of bug). Fix:
       ``commenters`` is disabled entirely; the (fail-closed, strictly
       safer) cost is that a *genuinely* shell-commented ``op read`` after
       a real ``#`` gets evaluated too and may be over-blocked — acceptable
       per the guard's anti-fumble posture (block a few extra safe things
       rather than miss a dangerous one).
    """
    lex = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>\n")
    lex.whitespace_split = True
    lex.whitespace = " \t\r"  # newline excluded — see docstring point 1
    lex.commenters = ""  # disabled — see docstring point 2
    try:
        return list(lex)
    except ValueError:
        return None


def _split_punctuation_run(tok: str) -> list[str]:
    """Split ONE shlex token composed entirely of ``_PUNCTUATION_CHARS`` back
    into its individual shell operators.

    shlex (as configured in ``_tokenize_shell_command``) merges several
    ADJACENT punctuation characters with no separating whitespace into a
    single token — e.g. ``"&&\\n"`` (an ``&&`` immediately followed by a
    newline, no space between them) comes back as ONE token, not two
    (empirically verified — see remediation-report-r2.md). Left unsplit,
    the segmenter below would never see the newline as its own separator
    token, hiding a real command boundary (H-1: ``cd /app &&\\nop read
    ...`` would tokenize its trailing ``op read ...`` as part of the SAME
    merged operator token's neighbour list, not a new segment). This walks
    the token left to right and emits the individual operators bash itself
    would recognise:

      - ``&&`` / ``||`` — two-char logical separators.
      - ``<(`` — process-substitution open, recognised as its own atomic
        2-char token (so a merged run like ``<(true)`` still tokenizes
        sanely); it stays an ordinary, always-scannable token and is never
        a structural opener for substitution handling (round-5 does that
        on the raw string — see ``_split_command_substitutions``).
      - ``(`` / ``)`` / ``;`` / newline — the single-char structural/
        separator tokens, always meaningful alone.
      - Any run starting with ``<`` or ``>`` (redirect operators — ``>>``,
        ``<<``, ``<<<``, ``>|``, ``>&``, ``<&``, ...) or ``&`` immediately
        followed by ``>`` (``&>``, ``&>>``) is consumed WHOLE as one opaque,
        meaningless-to-this-guard unit, rather than decomposed character by
        character. This is a deliberate, empirically-verified departure
        from naive per-character splitting: shlex merges ``2>&1``'s
        redirect into a token ``">&"`` (with ``2``/``1`` as separate plain
        tokens) — decomposing THAT into ``">"`` then ``"&"`` would
        manufacture a SPURIOUS ``&`` segment-separator out of a plain
        fd-duplication redirect, splitting (for example) ``op 2>&1 read
        x`` into two fake segments (``[op, 2, >]`` and ``[1, read, x]``)
        that individually don't contain both ``op`` and ``read`` — a
        fail-open regression versus even the round-1 guard. (Confirmed via
        a fake-``op`` argv-echo test that bash itself strips the redirect
        and invokes one real command, ``op`` with args ``read x``.) Treating
        the whole redirect run as one atomic token sidesteps this: it is
        never equal to any separator token, so it can only ever cause two
        segments to be treated as one (fail closed — more presence
        detection surface, never less), never fabricate a false split.
      - A bare ``&`` or ``|`` not part of one of the above multi-char forms
        is its own one-char separator token.
    """
    ops: list[str] = []
    i, n = 0, len(tok)
    while i < n:
        two = tok[i:i + 2]
        if two in _TWO_CHAR_SEPARATOR_OPS or two == _PROCESS_SUBSTITUTION_OPEN:
            ops.append(two)
            i += 2
            continue
        ch = tok[i]
        if ch in _SINGLE_CHAR_PARENS or ch == ";" or ch == "\n":
            ops.append(ch)
            i += 1
            continue
        if ch == "&" and tok[i + 1:i + 2] == ">":
            # &> / &>> — redirect-both-streams, never a separator.
            j = i + 1
            while j < n and tok[j] in "<>&":
                j += 1
            ops.append(tok[i:j])
            i = j
            continue
        if ch in "<>":
            # >>, <<, <<<, >|, >&, <&, and any further-glued run of these —
            # consumed whole, see docstring.
            j = i + 1
            while j < n and tok[j] in "<>&|":
                j += 1
            ops.append(tok[i:j])
            i = j
            continue
        # Bare '&' or '|' with none of the above multi-char forms matching.
        ops.append(ch)
        i += 1
    return ops


def _split_all_punctuation_runs(tokens: list[str]) -> list[str]:
    """Post-process a token list: any token composed ENTIRELY of characters
    in ``_PUNCTUATION_CHARS`` is split into its individual operators via
    ``_split_punctuation_run``; every other token (words, quoted content,
    flags, refs, ``VAR=$``-style assignments, etc.) passes through
    untouched. Order is preserved."""
    result: list[str] = []
    for tok in tokens:
        if tok and set(tok) <= _PUNCTUATION_CHARS:
            result.extend(_split_punctuation_run(tok))
        else:
            result.append(tok)
    return result


def _find_unescaped(source: str, target: str, start: int) -> int | None:
    """Index of the next unescaped ``target`` char at/after ``start``, or None."""
    i, n = start, len(source)
    while i < n:
        if source[i] == "\\":
            i += 2
            continue
        if source[i] == target:
            return i
        i += 1
    return None


def _find_substitution_close(source: str, start: int) -> int | None:
    """Index of the ``)`` closing a ``$(`` whose body begins at ``start``.

    Tracks nesting depth over unquoted ``(``/``)`` and skips over quoted
    spans, so an inner ``$(...)``, a literal ``")"`` inside a double-quoted
    argument, and an arithmetic ``$((...))`` all match correctly. Returns
    None when the substitution is unterminated."""
    depth = 1
    i, n = start, len(source)
    in_single = in_double = False
    while i < n:
        ch = source[i]
        if ch == "\\" and not in_single:
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return None


# A command substitution is a SAFE CAPTURE only when it is the right-hand
# side of a shell assignment -- `NAME=$(...)`, `NAME="$(...)"`, `arr[k]=$(...)`
# -- optionally preceded by other assignments or a keyword like `export`.
# The regex is matched against the text to the LEFT of the substitution's
# opener, anchored at its end, with an optional opening double quote allowed
# between the `=` and the opener.
_ASSIGNMENT_CAPTURE_PREFIX_RE = re.compile(
    r"""(?:^|[\s;&|(<>\n])          # start of string or a word/segment boundary
        [A-Za-z_][A-Za-z0-9_]*      # variable name
        (?:\[[^\]]*\])?             # optional array subscript
        =                           # the assignment
        \"?$                        # optional opening double quote, then the opener
    """,
    re.VERBOSE,
)


_BACKTICK_UNESCAPE_RE = re.compile(r"\\([`\\\\$])")


def _split_command_substitutions(command: str) -> tuple[str, list[str]]:
    """Separate a command string into (residual_text, inner_command_strings).

    This replaces round-4's token-level ``_excise_command_substitutions``,
    which was fail-OPEN in three ways that this string-level pass closes by
    construction (SEC-1, SEC-2, and a fourth shape found while fixing them):

    1. **SEC-1 -- ``echo $(op read op://x)`` leaked.** Round-4 excised ANY
       ``$(...)`` span whose opener token was preceded by a ``$``, on the
       stated premise that "the captured value lands in a shell variable ...
       never directly in stdout." That premise holds ONLY for an assignment.
       In ``echo $(op read ...)`` / ``printf %s $(op read ...)`` the
       substitution's stdout becomes the OUTER command's argument and is
       printed straight to the agent's context. Here, a substitution is
       excised only when ``_ASSIGNMENT_CAPTURE_PREFIX_RE`` matches the text
       to its left; in every other position its body is returned as an inner
       command for the caller to analyse recursively -- deny by default.
    2. **SEC-2 -- backticks were invisible.** ``shlex`` (as configured in
       ``_tokenize_shell_command``) treats a backtick as an ordinary word
       character, so `` echo `op read op://x` `` tokenized as
       ``['echo', '`op', 'read', 'op://x`']`` -- no token's basename was
       ever ``op``, and there was no backtick excision path at all, so it
       ALLOWED. Backtick substitution prints to stdout identically to
       ``$(...)``; both are handled by the same code path below.
    3. **Quoted substitutions were invisible.** ``echo "$(op read op://x)"``
       and ``echo "`op read op://x`"`` collapse to a SINGLE shlex token
       whose basename is neither ``op`` nor ``opread``, so the segment scan
       never saw inside them (ALLOW, empirically confirmed). A token-level
       pass cannot fix this -- by the time shlex has run, the quoting that
       glued the substitution into one word is gone. Scanning the RAW string
       before tokenization is what makes these visible.

    Quoting is honoured exactly as the shell does: inside SINGLE quotes a
    ``$(`` or a backtick is a literal, no substitution occurs, and the text
    is passed through to the residual untouched (which is what keeps the
    round-4 quoted-decoy regressions -- ``echo '<('; op read op://x`` and
    ``echo '$' '('; ...`` -- blocking). Inside double quotes both forms DO
    substitute, and are handled.

    Each substitution is replaced in the residual by a single space, so the
    remaining text still tokenizes and segments normally and a
    ``VAR=$(opread "op://x"); psql`` keeps its ``;`` boundary.

    Fail-closed properties: an UNTERMINATED substitution contributes the
    rest of the string as an inner command (rather than being dropped); a
    substitution in any position that is not provably an assignment RHS is
    analysed rather than trusted; and concatenated substitutions
    (``X=$(a)$(op read y)``) fall on the deny side because the second
    opener's left-context is no longer a bare ``NAME=`` -- over-blocking a
    rare safe shape rather than under-blocking a leak.
    """
    residual: list[str] = []
    inner: list[str] = []
    i, n = 0, len(command)
    in_single = in_double = False

    while i < n:
        ch = command[i]

        if ch == "\\" and not in_single and i + 1 < n:
            residual.append(command[i:i + 2])
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            residual.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            residual.append(ch)
            i += 1
            continue

        is_backtick = ch == "`" and not in_single
        is_dollar_paren = (
            ch == "$" and not in_single and i + 1 < n and command[i + 1] == "("
        )
        if is_backtick or is_dollar_paren:
            body_start = i + 1 if is_backtick else i + 2
            if is_backtick:
                close = _find_unescaped(command, "`", body_start)
            else:
                close = _find_substitution_close(command, body_start)
            body = command[body_start:close] if close is not None else command[body_start:]
            if is_backtick:
                # Inside a backtick substitution the shell strips ONE level of
                # backslash escaping before executing the body, which is how
                # backticks nest: `` echo `echo \`op read op://x\`` ``. Without
                # this the inner `\`` stays escaped, is never recognised as a
                # substitution opener on recursion, and the nested `op read`
                # leaks (reproduced). `$(...)` needs no equivalent -- it nests
                # by paren depth, which `_find_substitution_close` tracks.
                body = _BACKTICK_UNESCAPE_RE.sub(r"\1", body)
            if not _ASSIGNMENT_CAPTURE_PREFIX_RE.search("".join(residual)):
                inner.append(body)
            residual.append(" ")
            i = (close + 1) if close is not None else n
            continue

        residual.append(ch)
        i += 1

    return "".join(residual), inner


def _split_into_segments(tokens: list[str]) -> list[list[str]]:
    """Split depth-0 tokens into top-level segments on the separator set
    ``;``, ``&&``, ``||``, ``&``, ``|``, and a bare newline token.

    Pipe (``|``) is a segment separator like all the others — round-2 drops
    the round-1 "pipe-allow" carve-out (a piped op-egress command's stdout
    still reaches a consumer process and can resurface; fail closed)."""
    segments: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        is_newline_token = bool(tok) and set(tok) <= {"\n"}
        if tok in _SEGMENT_SEPARATOR_TOKENS or is_newline_token:
            segments.append(current)
            current = []
        else:
            current.append(tok)
    segments.append(current)
    return segments


def _resolve_op_subcommand_index(later: list[str]) -> int | None:
    """Find the index (within ``later``, the tokens after an ``op`` token)
    of THE op subcommand: the FIRST token that is a member of
    ``_KNOWN_OP_SUBCOMMANDS``. Returns None if no known-subcommand token is
    present at all.

    This needs no general flag-arity table — boolean global flags
    (``--debug``, ``--no-color``) and their values (an ``OP_ACCOUNT=x``
    style assignment, an env-wrapper's own args) are simply skipped over
    because they never match a keyword in the known set, not because
    their arity was modelled and consumed positionally (the round-2-era
    bug this sidesteps: a fixed "skip N tokens for this flag" table is
    always one flag behind reality).

    ONE narrow exception, tracked explicitly rather than left to chance:
    ``_OP_VALUE_TAKING_GLOBAL_FLAGS`` (currently just ``--account``) DOES
    consume exactly one following token as its value, and that value is
    skipped over even if it happens to collide with a known-subcommand
    keyword (e.g. ``op --account item read op://x``: ``--account``'s
    value is ``item`` — not a subcommand, just a value that happens to
    spell one — the real subcommand is ``read``, found right after it).
    Without this, the naive "first known-keyword" scan would stop at that
    coincidental ``item`` and hide a real ``read`` one token later — a
    fail-open regression versus round-2 on this one contrived shape
    (confirmed via the differential harness, see remediation-report-r3.md).
    This is deliberately scoped to GLOBAL pre-subcommand flags only (a
    fixed, tiny, explicit set) — not a general per-subcommand argument
    arity table for ``read``/``item get``/etc., which is exactly the kind
    of always-incomplete machinery the spec says to avoid.
    """
    i, n = 0, len(later)
    while i < n:
        tok = later[i]
        if tok in _OP_VALUE_TAKING_GLOBAL_FLAGS:
            i += 2  # skip the flag AND its value token
            continue
        if tok in _KNOWN_OP_SUBCOMMANDS:
            return i
        i += 1
    return None


def _segment_is_op_egress(segment_tokens: list[str], _depth: int = 0) -> bool:
    """Decide whether ONE top-level segment's tokens are an op-egress
    command (round-3, M-1 — rewritten to locate THE subcommand precisely
    instead of flagging on the presence of ANY later ``read`` token
    regardless of what it actually is).

    Round-2 flagged egress whenever a ``read`` token appeared ANYWHERE
    after an ``op``/``opread`` token — which over-blocked plenty of
    sanctioned, safe commands where ``read`` is merely an ARGUMENT VALUE
    to a non-egress subcommand: ``op run -- make read`` (a child command
    literally named ``read``), ``op inject -i read -o out.conf`` (a
    template file named ``read``), ``op item get read`` (an item named
    ``read``), ``op item create --title read`` (a title of ``read``).
    ``op run``/``op inject`` in particular are the L3 SAFE paths this
    guard's own redirect message recommends — over-blocking them defeats
    the guard's own escape hatch.

    Fix: after locating an ``op``/``opread`` token, resolve THE subcommand
    by known-keyword scan (``_resolve_op_subcommand_index`` over
    ``_KNOWN_OP_SUBCOMMANDS`` — which also transparently absorbs
    ``_OP_VALUE_TAKING_GLOBAL_FLAGS``' values, see that function's
    docstring for the ``op --account item read x`` collision this closes)
    rather than testing for bare presence of ``read``, then classify by
    THAT subcommand:
      - ``read`` → egress (block) — unconditionally; no other subcommand
        can ever override this once resolved as the FIRST known keyword.
      - ``document`` with ``get`` present anywhere in ``later`` → egress.
      - ``item`` with ``get`` present AND a dump-flag token
        (``--fields``/``--field``/``--format``/``--reveal``/``--otp``, matched
        whole-token or as the LHS of ``--flag=value``) present anywhere in
        ``later`` → egress; ``item get`` with no dump flag is the masked
        overview → not egress.
      - any OTHER first-resolved subcommand (``run``, ``inject``,
        ``create``, ``edit``, ``list``, ``vault``, ``signin``, ``whoami``,
        ``account``, ...) → not egress; a later ``read``-looking argument
        value is deliberately ignored (that's the whole fix).
      - no known-subcommand token found at all → not egress (e.g. a bare
        ``op`` with no subcommand, or one still being typed).

    Basename matching (not raw token equality) means an absolute path like
    ``/usr/bin/op``/``/usr/bin/opread`` is still recognised. ``opread``
    anywhere is still unconditionally egress, unchanged from round-2.
    """
    # -- SEC-3: recurse into shell code handed to a command runner ---------
    # Round-4 detected egress only when a TOKEN's basename was `op`/`opread`.
    # With `sh -c 'op read op://x'`, shlex emits the whole inner command as
    # ONE quoted token whose basename is `x` (from `op://x`), so no token
    # matched and the command ALLOWED -- as did `bash -c "..."`, `eval '...'`
    # and `env FOO=1 sh -c '...'` (all three reproduced before this fix).
    # A quoted argument is opaque to a token scan, so it has to be re-entered
    # and analysed as a command in its own right.
    #
    # `eval` concatenates ALL its arguments into one command line, so every
    # following token is code. Everything else is reached through the
    # flag-shape rule in `_CODE_TAKING_FLAG_RE`.
    #
    # Known limit, stated rather than left implicit: a runner that takes
    # shell code WITHOUT a `-c`-shaped flag -- `ssh host 'op read x'`,
    # `perl -e`, `find . -exec op read {} \;` -- is still not re-entered.
    # Closing that generally would require treating every quoted argument of
    # every unknown command as code, which would block ordinary things like
    # `grep "op read" notes.md`. The boundary is drawn here deliberately.
    for i, tok in enumerate(segment_tokens):
        if os.path.basename(tok) == "eval":
            rest = " ".join(segment_tokens[i + 1:]).strip()
            if rest and op_egress_block_reason(rest, _depth=_depth + 1):
                return True
            continue
        if _CODE_TAKING_FLAG_RE.match(tok) and i + 1 < len(segment_tokens):
            if op_egress_block_reason(segment_tokens[i + 1], _depth=_depth + 1):
                return True

    for tok in segment_tokens:
        if os.path.basename(tok) == "opread":
            return True

    for i, tok in enumerate(segment_tokens):
        if os.path.basename(tok) != "op":
            continue
        later = segment_tokens[i + 1:]

        subcommand_index = _resolve_op_subcommand_index(later)
        if subcommand_index is None:
            continue
        subcommand = later[subcommand_index]

        if subcommand == "read":
            return True
        if subcommand == "document":
            if "get" in later:
                return True
            continue
        if subcommand == "item":
            has_get = "get" in later
            has_dump_flag = any(
                t.split("=")[0] in _OP_ITEM_GET_DUMP_FLAGS for t in later
            )
            if has_get and has_dump_flag:
                return True
            continue
        # Any other first-resolved subcommand (run/inject/vault/account/
        # signin/whoami/create/edit/list/...) is sanctioned/not-egress —
        # a later `read`-looking argument value is deliberately ignored.

    return False


def op_egress_block_reason(command: str, _depth: int = 0) -> str | None:
    """Return a redirect message if `command` would leak a 1Password secret.

    Pure predicate, no I/O, no state. Detects shell commands that print a
    1Password secret to stdout — which would land the raw value in the
    agent's context — and returns a redirect message pointing at the safe
    alternatives (``op-run``, ``op inject``) instead. Returns None when the
    command is safe to run as-is (including when ``command`` isn't a
    non-blank string at all — the guard is total, never raises).

    Algorithm (round-5, deny-by-default — see the module-level comment above
    ``_OP_ITEM_GET_DUMP_FLAGS`` for why presence-scanning replaced the
    round-1 wrapper-stripping/positional design, and
    ``_split_command_substitutions`` for why substitution handling moved off
    the token stream and onto the raw string):
      1. Delete backslash-newline line continuations, exactly as the shell
         does before it tokenizes anything.
      2. Split command substitutions off the RAW string, quote-aware
         (``_split_command_substitutions``). A substitution that is provably
         an assignment capture (``NAME=$(…)``, ``NAME="$(…)"``) is dropped —
         its value lands in a shell variable, never on this command's
         stdout. EVERY other substitution's body, in any position and in
         either syntax (``$(…)`` or backticks, bare or double-quoted), is
         returned for step 3.
      3. Recurse: any inner command that is itself op-egress blocks the
         whole command. Depth is bounded by ``_MAX_NESTING_DEPTH``, and
         exceeding it BLOCKS rather than allows.
      4. Quote-aware tokenize the residual via ``shlex``
         (``_tokenize_shell_command``). On unbalanced quotes
         (``ValueError``), fall back to ``_conservative_fallback``'s
         begins-with check.
      5. Split any MERGED punctuation token (e.g. shlex's ``"&&\n"``) back
         into its individual operators (``_split_all_punctuation_runs``).
      6. Split what remains into top-level SEGMENTS on ``;``, ``&&``,
         ``||``, ``&``, ``|`` (pipe is just another separator — pipe-allow
         was dropped in round-2), and a bare newline
         (``_split_into_segments``).
      7. A segment is op-egress by PRESENCE of an op-egress keyword
         combination anywhere in its tokens (``_segment_is_op_egress``) —
         no wrapper-stripping, no flag-arity guessing, no positional-binary
         assumption, so ``sudo -u root op read x``, ``time op read x``,
         ``op --debug read x``, and ``(op read x)`` are all caught by the
         same presence scan. That scan also re-enters shell code handed to
         a ``-c``-shaped flag or to ``eval`` (round-5, SEC-3).
      8. If ANY segment is op-egress, the command as a whole is BLOCKED.

    Residual risk, recorded rather than implied: a captured secret can still
    be printed by a LATER command (``V=$(op read x); echo "$V"``). This
    guard is a stdout-shape predicate, not dataflow analysis, and the
    capture carve-out is what makes the sanctioned ``VAR=$(opread …)``
    pattern usable at all. The redaction layer, not this predicate, is the
    control that covers that path.

    BLOCKS (examples): ``op read <ref>``, ``opread <ref>``,
    ``op document get <ref>``, ``op item get ... --fields|--field|--format|
    --reveal|--otp``, any of the above wrapped in ``sudo``/``sudo -u
    user``/``env``/``time``/``nohup``/an absolute path/an env-var
    assignment/a boolean global ``op`` flag (``--debug``, ``--no-color``)/
    a value-taking global ``op`` flag (``--account``/``--config``/
    ``--session``) whose value coincidentally spells a subcommand keyword,
    any of the above reached through a command substitution in a NON-capture
    position (``echo $(op read x)``, ``printf %s $(op read x)``, backticks
    ``echo `op read x` `` including nested/escaped ones, and their double-
    quoted forms ``echo "$(op read x)"`` — round-5, SEC-1/SEC-2), any of the
    above passed as shell code to a command runner (``sh -c 'op read x'``,
    ``bash -lc "…"``, ``eval '…'``, ``env FOO=1 sh -c '…'``, ``xargs -I{} sh
    -c '…'`` — round-5, SEC-3), any of the above as a top-level segment of a
    ``;``/``&&``/``||``/``&``/``|``/newline-separated sequence (including
    when the separator is glued to adjacent punctuation, e.g. ``&&\\n``,
    or backgrounded, e.g. ``echo x & op read y``), inside a subshell
    ``(...)``/brace ``{...}``/``if`` group, as the LAST stage of a pipe
    too (``op read x | jq .`` BLOCKS — pipe-allow is dropped), and a
    quoted decoy ``'<('``/``'$' '('`` that a later, unrelated ``)`` would
    otherwise appear to "close" (round-4: never excised, so the real
    ``op read``/etc. in between stays scannable).

    ALLOWS (examples): ``op item get`` without a dump flag (masked
    overview), all management commands (``op item create/edit/list``,
    ``op vault ...``, ``op inject``, ``op run``, ``op signin``,
    ``op whoami``, ``op account ...``), ``opsudo ...``, any non-op command,
    ``op read``/dump-flag text appearing only inside a quoted argument of
    another command (e.g. ``echo "op read x"``, ``op item get "Item
    --format"``), a flag whose name merely CONTAINS a dump-flag substring
    (e.g. ``--show-all-fields``, not a whole-token/``--flag=value`` match),
    ``op readme``/``opreadiness`` (distinct binaries/subcommands — basename
    and whole-token matching, never substring), and — the dominant safe
    pattern — a value captured via command substitution,
    ``VAR=$(opread "op://...")`` / ``VAR=$(op read "op://...")`` /
    ``VAR="$(op read "op://...")"`` / ``VAR=`op read op://...` `` (dropped
    before the segment scan; the value never reaches this command's stdout,
    landing in the shell variable instead). Note that a SINGLE-quoted
    ``VAR='$(op read x)'`` is a literal string to the shell, not a
    substitution, and is likewise not egress.

    Args:
        command: The raw shell command string as given to the shell tool.

    Returns:
        The redirect message (contains the substrings ``op-run`` and
        ``op inject``) if the command would leak a secret to stdout;
        otherwise None.
    """
    if not isinstance(command, str) or not command.strip():
        return None

    # Fail closed on pathological nesting rather than unwinding into an
    # allow (see `_MAX_NESTING_DEPTH`).
    if _depth > _MAX_NESTING_DEPTH:
        return _OP_EGRESS_BLOCK_MESSAGE

    # Round-3 (H2): the real shell (dash/bash) deletes a backslash-newline
    # line continuation BEFORE it ever tokenizes the command — `op \<NL>read
    # x` is exactly the same command as `op read x` to the shell. Without
    # this pre-pass, shlex instead emits the newline as its OWN separator
    # token (see `_tokenize_shell_command`'s docstring point 1) and the
    # backslash stays glued to `op` as one token `"op\\"` — so the exact
    # token `"read"` is still present, but the presence scan below (and,
    # more importantly, `_segment_is_op_egress`'s "AFTER that `op` token"
    # walk) would see it in a NEW segment after a spurious split, not
    # anchored to the same segment as the `op`/`opread` token — hiding a
    # top-level `op read`/`--fields` from the scan entirely (empirically
    # confirmed fail-open, see remediation-report-r3.md). Normalizing this
    # first, on the raw string, before any tokenization, makes the guard see
    # exactly the same token stream the shell itself will execute. Doing it
    # unconditionally is fail-closed, never fail-open: a backslash-newline is
    # only ever a line continuation, or (extremely rare) part of an already
    # atomic quoted-string token that this replace cannot partially unglue
    # (a quoted `'\<NL>'` stays one token either way) — at worst this
    # over-normalizes and over-blocks, never hides egress.
    command = command.replace("\\\r\n", "").replace("\\\n", "")

    # Separate command substitutions from the surrounding text BEFORE
    # tokenizing (see `_split_command_substitutions`): shlex destroys the
    # quoting that distinguishes `echo "$(op read x)"` from `echo 'op read
    # x'`, so this cannot be done on tokens. Every substitution that is not
    # provably an assignment capture is analysed as a command in its own
    # right -- deny by default, where round-4 excised and trusted it.
    residual, inner_commands = _split_command_substitutions(command)

    for inner in inner_commands:
        if op_egress_block_reason(inner, _depth=_depth + 1):
            return _OP_EGRESS_BLOCK_MESSAGE

    tokens = _tokenize_shell_command(residual)
    if tokens is None:
        return _conservative_fallback(residual)
    if not tokens:
        return None

    tokens = _split_all_punctuation_runs(tokens)

    for segment in _split_into_segments(tokens):
        if not segment:
            continue
        if _segment_is_op_egress(segment, _depth=_depth):
            return _OP_EGRESS_BLOCK_MESSAGE

    return None
