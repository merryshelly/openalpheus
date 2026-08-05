"""Calendar-scheduled timer wrapper around the vendored cronsim engine (kdsn.210.6).

This module exports three pure functions and one exception class:
  - next_fire(spec, after, tz) → datetime: next matching instant strictly after `after`
  - validate_spec(spec) → None: raises ScheduleError on malformed specs
  - min_gap(spec, tz, samples=50) → int: smallest gap in seconds across the next `samples` fires
  - ScheduleError: exception raised on invalid cron specs

The vendored engine (openalph.vendor.cronsim) is kept internal; callers never import it.
DEFAULT_TZ is a module-level string constant holding the host-local IANA zone name.

Design: specs/scheduled-timers/DESIGN.md §5, §8; TEST-PLAN §A
"""

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from openalph.vendor.cronsim import CronSim, CronSimError


class ScheduleError(Exception):
    """Raised when a cron schedule spec is malformed."""
    pass


def _detect_default_tz() -> str:
    """Detect the host-local IANA timezone name.

    Tries (in order):
    1. TZ environment variable
    2. /etc/localtime symlink target under /usr/share/zoneinfo/
    3. /etc/timezone file contents

    Falls back to America/New_York if detection fails.
    Returns the IANA zone name as a string, e.g. 'America/New_York'.
    """
    # Try TZ environment variable
    tz = os.environ.get("TZ")
    if tz:
        return tz

    # Try /etc/localtime symlink
    try:
        localtime = Path("/etc/localtime")
        if localtime.is_symlink():
            target = localtime.resolve()
            zoneinfo_base = Path("/usr/share/zoneinfo")
            try:
                relative = target.relative_to(zoneinfo_base)
                return str(relative)
            except ValueError:
                # Not under /usr/share/zoneinfo
                pass
    except (OSError, RuntimeError):
        pass

    # Try /etc/timezone file
    try:
        tz_file = Path("/etc/timezone")
        if tz_file.exists():
            contents = tz_file.read_text().strip()
            if contents:
                return contents
    except (OSError, RuntimeError):
        pass

    # Fallback
    return "America/New_York"


DEFAULT_TZ: str = _detect_default_tz()


def validate_spec(spec: str) -> None:
    """Validate a 5-field cron specification.

    Accepts exactly 5 fields: minute hour day-of-month month day-of-week.
    Raises ScheduleError if the spec is empty, wrong field count, out-of-range,
    garbage, or injection-y input. No raw cronsim exception leaks out.
    """
    if not spec:
        raise ScheduleError("Spec cannot be empty")

    parts = spec.split()
    if len(parts) != 5:
        raise ScheduleError(
            f"Spec must be exactly 5 fields (minute hour day month day-of-week), got {len(parts)}"
        )

    try:
        # Use a dummy datetime in UTC to validate the spec
        dummy = datetime(2026, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("UTC"))
        CronSim(spec, dummy)
    except CronSimError as e:
        raise ScheduleError(f"Invalid cron spec: {e}") from e


def next_fire(spec: str, after: datetime, tz: ZoneInfo) -> datetime:
    """Compute the next matching instant strictly after `after`.

    Args:
        spec: 5-field cron expression (minute hour day month day-of-week)
        after: reference datetime (must be timezone-aware)
        tz: target ZoneInfo for the cron schedule

    Returns:
        A timezone-aware datetime in `tz` representing the next matching instant
        strictly after `after`. If `after` itself is a matching instant, returns
        the next match, never `after`.

    Raises:
        ScheduleError: if spec is malformed
    """
    try:
        # Convert 5-field spec to 6-field (add seconds=0) to avoid cronsim's
        # problematic fixup_tz mechanism that was designed for Debian cron.
        # fixup_tz is only applied to 5-field specs, so using 6-field avoids it.
        spec_6field = f"0 {spec}"

        # Use the target timezone; cronsim will correctly handle DST without fixup_tz
        after_tz = after.astimezone(tz)
        cron = CronSim(spec_6field, after_tz)
        result = next(cron)

        # Ensure result has the correct timezone
        if result.tzinfo is None:
            result = result.replace(tzinfo=tz)

        # Ensure strictly-after semantics: if result <= after, get the next occurrence
        while result <= after_tz:
            result = next(cron)
            if result.tzinfo is None:
                result = result.replace(tzinfo=tz)

        return result
    except CronSimError as e:
        raise ScheduleError(f"Invalid cron spec: {e}") from e


def min_gap(spec: str, tz: ZoneInfo, samples: int = 50) -> int:
    """Compute the smallest gap in seconds between consecutive fires.

    Iterates through the next `samples` fire times and returns the minimum
    gap (in seconds) between consecutive fires. Used for floor enforcement.

    Args:
        spec: 5-field cron expression
        tz: target ZoneInfo for the cron schedule
        samples: number of fires to examine (default 50)

    Returns:
        Smallest gap in seconds (as an int)

    Raises:
        ScheduleError: if spec is malformed
    """
    try:
        # Convert 5-field spec to 6-field (add seconds=0) to avoid cronsim's fixup_tz
        spec_6field = f"0 {spec}"

        # Start from now in the target timezone
        start = datetime.now(tz)
        cron = CronSim(spec_6field, start)

        # Collect the next `samples+1` fire times to compute `samples` gaps
        fires = []
        for _ in range(samples + 1):
            fire = next(cron)
            if fire.tzinfo is None:
                fire = fire.replace(tzinfo=tz)
            fires.append(fire)

        # Compute gaps and find minimum
        gaps = []
        for i in range(len(fires) - 1):
            gap = (fires[i + 1] - fires[i]).total_seconds()
            gaps.append(int(gap))

        return min(gaps) if gaps else 60
    except CronSimError as e:
        raise ScheduleError(f"Invalid cron spec: {e}") from e
