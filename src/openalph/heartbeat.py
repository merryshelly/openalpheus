"""Heartbeat timer for OpenAlph.

Per-room recurring timer with persistence. The manager itself is a thin
subclass of RecurringTimerManager (see _timer.py); only the interval-parsing
helpers below are heartbeat-specific. Communicates via an async callback and
has no Matrix-specific knowledge.

Cadence: heartbeat fires on a FIXED interval and is NOT drift-corrected -- a
slow callback pushes the next fire later (a 15m heartbeat with 5m turns drifts
~33%). This is preserved from the original for behavioural compatibility; the
drift-corrected variant is UmbralManager. Both now share one implementation.
"""

from dataclasses import dataclass

from openalph._timer import RecurringTimerManager


@dataclass
class HeartbeatEntry:
    room_id: str
    interval_seconds: int | None  # or float for test compatibility
    seconds_until_next: int
    directive: str | None = None
    schedule: str | None = None
    tz: str | None = None


class HeartbeatManager(RecurringTimerManager):
    """Per-room recurring heartbeats with persistence.

    Args:
        config_path: Path to heartbeats.json
        callback: async function called with room_id when a heartbeat fires
    """

    _entry_cls = HeartbeatEntry
    _drift_correct = False
    _overlap_guard = False
    _floor = 300  # heartbeat minimum gap in seconds (§7 of DESIGN.md)
    _loop_name = "Heartbeat"


def parse_interval(s: str) -> int | None:
    """Parse '15m', '1h', '6h' etc. to seconds.

    Returns None on invalid input.
    Case-insensitive. Strips whitespace.
    Rejects decimals and negatives.
    """
    if not s:
        return None

    s = s.strip().lower()

    if not s:
        return None

    # Extract numeric part and unit
    num_str = ""
    unit = ""

    for i, ch in enumerate(s):
        if ch.isdigit():
            num_str += ch
        else:
            unit = s[i:]
            break

    if not num_str or not unit:
        return None

    # Reject decimal numbers (must be whole digits only)
    if "." in s:
        return None

    try:
        num = int(num_str)
    except ValueError:
        return None

    # Reject negatives and zero
    if num <= 0:
        return None

    # Parse unit
    if unit == "m":
        return num * 60
    elif unit == "h":
        return num * 3600
    elif unit == "s":
        return num
    else:
        return None


def format_interval(seconds: int) -> str:
    """Format seconds as human-readable, using mixed units when needed.

    3600 -> '1h', 900 -> '15m', 45 -> '45s'
    7393 -> '2h 3m', 3661 -> '1h 1m', 90 -> '1m 30s'
    """
    if seconds < 0:
        return "0s"
    if seconds == 0:
        return "0s"

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    # Only show seconds if no larger unit, or if total < 5 minutes (precision matters)
    if s and (not parts or seconds < 300):
        parts.append(f"{s}s")

    return " ".join(parts) if parts else "0s"
