"""Degraded-start notifications (kdsn.292).

Two surfaces, one composer:

- ``degraded_summary(config)`` — PURE: builds the shared one-glance summary of
  which providers failed startup and why (reasons run through the credential
  redactor BEFORE composing — a skip reason might quote command output).
  Used by the matrix startup broadcast, ``/status``, and the ntfy alert.
- ``maybe_notify_default_degraded(config)`` — sends ONE ntfy POST at startup
  iff the RESOLVED default_model's provider was skipped AND [notifications]
  is configured. stdlib urllib only; calls _resolve_secret's resolved token
  if one loaded. FAIL-SOFT BY DESIGN: any network/HTTP error logs and
  returns False — the alerting path must never recreate the crash-loop bug
  it exists to report.
"""

import logging
import urllib.request
from urllib.request import Request, urlopen  # urlopen module-level so the
# acceptance suite (and ops smoke tests) can monkeypatch openalph.notify.urlopen

from openalph.tools.security import redact_credentials

logger = logging.getLogger(__name__)


def degraded_summary(config) -> str:
    """Compose the human summary of a degraded start.

    Names the agent, its default model, and every skipped provider with its
    (redacted) startup-failure reason. Ends with the operator-facing recovery
    hint: slash commands still work; `/model` to a loaded provider recovers
    without a restart.
    """
    lines = [
        f"**{config.name}** started DEGRADED "
        f"(default model: `{config.default_model}`).",
    ]
    skipped = getattr(config, "skipped_providers", {}) or {}
    if skipped:
        lines.append("Providers skipped at startup:")
        for key in sorted(skipped):
            reason, _events = redact_credentials(str(skipped[key]))
            lines.append(f"- **{key}**: {reason}")
    else:
        lines.append("No providers were skipped, but none are loaded.")
    loaded = sorted(getattr(config, "providers", {}) or {})
    if loaded:
        lines.append(f"Loaded providers: {', '.join(loaded)}.")
    lines.append(
        "Slash commands and local tools still work; "
        "`/model <loaded-provider>/<model>` recovers without a restart."
    )
    return "\n".join(lines)


def maybe_notify_default_degraded(config) -> bool:
    """POST one ntfy alert iff the default model's provider is degraded.

    Returns True only when a POST was actually issued. Silent (False) when
    notifications are unconfigured OR the default's provider loaded fine —
    non-default skips are surfaced in-room by the startup broadcast instead.
    Never raises (see module docstring): on ANY error logs and returns False.
    """
    notifications = getattr(config, "notifications", None)
    if notifications is None:
        return False

    skipped = getattr(config, "skipped_providers", {}) or {}
    if not skipped:
        return False

    # Resolve the default through aliases exactly as the (deleted) fatal
    # validation block at config load did: a bare default_model expands via
    # model_aliases before prefix matching.
    default = getattr(config, "default_model", "") or ""
    aliases = getattr(config, "model_aliases", {}) or {}
    if "/" not in default and default in aliases:
        default = aliases[default]
    if "/" not in default:
        return False  # unqualified & unaliased: no provider to check
    prefix = default.partition("/")[0]
    if prefix not in skipped:
        return False

    # Review M1 (accepted tradeoff, comment-only): the POST below runs
    # SYNCHRONOUSLY in the cli startup path with urlopen(timeout=5), so a
    # wedged ntfy endpoint stalls degraded startup by up to ~5s. That is
    # accepted — deterministic ordering beats threading a once-per-process
    # alert (no races against startup logs, no fire-and-forget task to keep
    # alive across the event-loop handoff), and the timeout bounds it.
    try:
        body = degraded_summary(config)
        headers = {
            "Title": f"\U0001F6A8 {config.name} degraded start",
            "Priority": "high",
            "Tags": "rotating_light",
            "Content-Type": "text/plain; charset=utf-8",
        }
        token = getattr(notifications, "ntfy_token", None)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = Request(
            notifications.ntfy_url,
            data=body.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            logger.info(
                "ntfy degraded-start alert sent for %s (HTTP %s)",
                config.name, getattr(resp, "status", "?"),
            )
        return True
    except Exception as e:
        # Fail-soft is THE point: never let the alert path take the agent down.
        logger.error(
            "ntfy degraded-start notification failed for %s: %s",
            getattr(config, "name", "?"), e,
        )
        return False
