"""Tests for openalph.schedule — the cron-grammar wrapper (kdsn.210.6).

Pins the three pure functions over the vendored engine and their DST/tz
semantics. These are the correctness core of the feature: pure, no async, no
Agent — cheap and exhaustive.

    next_fire(spec, after, tz)   — next instant STRICTLY after `after`
    validate_spec(spec)          — None on valid, ScheduleError on malformed
    min_gap(spec, tz, samples)   — smallest gap (s) across the next `samples` fires

The vendored engine is openalph/vendor/cronsim.py (cuu508/cronsim, BSD-3).
Callers never import cronsim — ScheduleError wraps its parse errors.

A13 note: the vendored directory is resolved through the importable `openalph`
package (openalph.__file__/vendor), NOT via Path(__file__).parents or cwd —
this file is staged outside the repo tree, and the importable-package path is
correct both during staging and after the file lands in tests/.

Design:  memory/projects/openalph/scheduled-timers/DESIGN.md (§5, §8)
Plan:    memory/projects/openalph/scheduled-timers/TEST-PLAN.md §A
"""

import ast
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from openalph.schedule import next_fire, validate_spec, min_gap, ScheduleError


NY = ZoneInfo("America/New_York")
UTC = timezone.utc


# ======================================================================
# A. next_fire — strictly-after semantics + cron grammar (DESIGN §5)
# ======================================================================

class TestNextFire:
    """next_fire returns the next matching instant strictly after `after`."""

    def test_a01_weekday_skips_weekend(self):
        # A01 (§A): Fri 06:31 → next Mon 06:30 (the weekend is skipped).
        spec = "30 6 * * 1-5"
        fri_0631 = datetime(2026, 3, 6, 6, 31, tzinfo=NY)  # Friday
        nxt = next_fire(spec, fri_0631, NY)
        assert nxt == datetime(2026, 3, 9, 6, 30, tzinfo=NY)  # Monday
        assert nxt.weekday() == 0  # Monday

    def test_a02_same_day_before_target(self):
        # A02: Fri 06:29 → same-day 06:30 (target not yet reached).
        spec = "30 6 * * 1-5"
        fri_0629 = datetime(2026, 3, 6, 6, 29, tzinfo=NY)
        nxt = next_fire(spec, fri_0629, NY)
        assert nxt == datetime(2026, 3, 6, 6, 30, tzinfo=NY)

    def test_a03_sunday_evening(self):
        # A03: "0 20 * * 0" → next Sunday 20:00.
        spec = "0 20 * * 0"
        wed = datetime(2026, 3, 4, 12, 0, tzinfo=NY)  # Wednesday
        nxt = next_fire(spec, wed, NY)
        assert nxt == datetime(2026, 3, 8, 20, 0, tzinfo=NY)
        assert nxt.weekday() == 6  # Sunday

    def test_a04_step_values(self):
        # A04: "*/15 * * * *" from 10:07 → 10:15 (step values).
        spec = "*/15 * * * *"
        at_1007 = datetime(2026, 3, 6, 10, 7, tzinfo=NY)
        nxt = next_fire(spec, at_1007, NY)
        assert nxt == datetime(2026, 3, 6, 10, 15, tzinfo=NY)

    def test_a05_strictly_after_boundary(self):
        # A05 (§5): T that exactly matches → returns the NEXT occurrence,
        # never T itself (prevents a double-fire at the boundary).
        spec = "30 6 * * 1-5"
        exact = datetime(2026, 3, 6, 6, 30, tzinfo=NY)  # Friday 06:30 exactly
        nxt = next_fire(spec, exact, NY)
        assert nxt != exact
        assert nxt == datetime(2026, 3, 9, 6, 30, tzinfo=NY)  # next weekday 06:30

    def test_a05_strictly_after_every_minute(self):
        # A05 (every-minute spec): exact match → +60s, never the same instant.
        spec = "* * * * *"
        exact = datetime(2026, 3, 6, 6, 30, tzinfo=NY)
        nxt = next_fire(spec, exact, NY)
        assert nxt == exact + timedelta(minutes=1)


# ======================================================================
# A. validate_spec — accept canonical, reject malformed (DESIGN §5, §7)
# ======================================================================

class TestValidateSpec:
    """validate_spec returns None on valid specs, raises ScheduleError on bad."""

    def test_a06_accepts_canonical(self):
        # A06: canonical 5-field specs are accepted (return None, no raise).
        for spec in ("0 0 * * *", "30 6 * * 1-5", "15 14 1 * *", "0 22 * * 1-5"):
            assert validate_spec(spec) is None

    def test_a07_rejects_malformed(self):
        # A07: empty, wrong field count, out-of-range, garbage, injection —
        # every one raises ScheduleError; no raw cronsim error leaks out.
        bad = [
            "",                 # empty
            "0 0 * *",          # 4-field
            "0 0 0 * * *",      # 6-field (seconds form not supported)
            "60 * * * *",       # minute out of range
            "* 25 * * *",       # hour out of range
            "abc",              # garbage
            "; rm -rf",         # injection-y input
        ]
        for spec in bad:
            with pytest.raises(ScheduleError):
                validate_spec(spec)


# ======================================================================
# A. min_gap — smallest inter-fire gap, for floor enforcement (DESIGN §7)
# ======================================================================

class TestMinGap:
    """min_gap returns the smallest gap in seconds across the next N fires."""

    def test_a08_every_minute(self):
        # A08: "* * * * *" → 60s (below the heartbeat floor — caught up front).
        assert min_gap("* * * * *", NY) == 60

    def test_a08_weekday_spans_weekend(self):
        # A08: weekday spec — the Fri→Mon jump is the LARGER gap; the floor
        # check needs the SMALLEST, which is the intra-week 24h.
        assert min_gap("30 6 * * 1-5", NY) == 86400

    def test_a09_at_heartbeat_floor(self):
        # A09: "*/5 * * * *" → exactly 300s (the heartbeat floor — boundary).
        assert min_gap("*/5 * * * *", NY) == 300


# ======================================================================
# A. DST handling — cronsim convention (DESIGN §8)
# ======================================================================
# America/New_York 2026:
#   spring-forward  March 8 2026  (02:00 → 03:00; 02:00–02:59 do not exist)
#   fall-back       November 1 2026 (02:00 → 01:00; 01:00–01:59 occur twice)
# Convention (cronsim): nonexistent times are SKIPPED; ambiguous times fire ONCE.

class TestDst:
    def test_a10_spring_forward_skips_nonexistent(self):
        # A10: "30 2 * * *" across the 2026 spring-forward — 02:30 does not
        # exist on March 8; the iterator must not raise and must skip it.
        spec = "30 2 * * *"
        before = datetime(2026, 3, 7, 3, 0, tzinfo=NY)  # day before transition
        nxt = next_fire(spec, before, NY)
        # The next 02:30 is March 9 — March 8 02:30 is skipped, not raised.
        assert nxt == datetime(2026, 3, 9, 2, 30, tzinfo=NY)

    def test_a10_spring_forward_practical_case_unaffected(self):
        # A10: a 06:30 daily job is outside the 02:00–03:00 gap and is
        # unaffected by the same transition (the practical case).
        spec = "30 6 * * *"
        before = datetime(2026, 3, 7, 12, 0, tzinfo=NY)
        nxt = next_fire(spec, before, NY)
        assert nxt == datetime(2026, 3, 8, 6, 30, tzinfo=NY)

    def test_a11_fall_back_fires_once(self):
        # A11: a schedule at the ambiguous hour fires ONCE, not twice.
        spec = "30 1 * * *"
        before = datetime(2026, 10, 31, 12, 0, tzinfo=NY)  # day before fall-back
        first = next_fire(spec, before, NY)
        assert (first.month, first.day, first.hour, first.minute) == (11, 1, 1, 30)
        # The following occurrence is Nov 2 — the ambiguous 01:30 is not doubled.
        second = next_fire(spec, first, NY)
        assert (second.month, second.day, second.hour) == (11, 2, 1)
        assert (second - first).total_seconds() == 86400

    def test_a12_tz_actually_applied(self):
        # A12: same spec, tz=UTC vs America/New_York — instants offset by the
        # zone (proves tz is applied, not ignored). March 6 2026 is EST
        # (UTC-5 — spring-forward is March 8), so the offset is 5h.
        spec = "30 6 * * *"
        after_utc = datetime(2026, 3, 6, 12, 0, tzinfo=UTC)
        utc_fire = next_fire(spec, after_utc, ZoneInfo("UTC"))
        after_ny = datetime(2026, 3, 6, 7, 0, tzinfo=NY)  # same instant as 12:00Z
        ny_fire = next_fire(spec, after_ny, NY)
        # Same wall-clock spec → the NY fire is 5h behind the UTC fire.
        # (utc_fire = Mar 7 06:30Z; ny_fire = Mar 7 06:30 EST = 11:30Z.)
        diff = abs((utc_fire - ny_fire).total_seconds())
        assert diff == 5 * 3600


# ======================================================================
# A. Provenance of the vendored engine (DESIGN §5; TEST-PLAN A13)
# ======================================================================

class TestVendoredEngine:
    """The vendored cronsim.py is pure-stdlib; explain.py is NOT vendored."""

    def _vendor_dir(self) -> Path:
        # Resolve through the importable package (see module docstring), so the
        # path is correct both staged-outside-the-repo and after landing in tests/.
        import openalph
        return Path(openalph.__file__).resolve().parent / "vendor"

    def test_a13_cronsim_imports_only_stdlib(self):
        # A13: ast-scan cronsim.py — every MODULE-LEVEL import resolves to
        # stdlib (importing the module must require no third-party package).
        # We scan only top-level statements: `explain()` carries an intentional
        # function-local `from cronsim.explain import Expression` (dead on the
        # vendored path, since explain.py is NOT vendored) — a conditional
        # import, not a module-load dependency.
        cronsim = self._vendor_dir() / "cronsim.py"
        assert cronsim.exists(), f"vendored cronsim not found at {cronsim}"
        tree = ast.parse(cronsim.read_text())
        imported_roots = set()
        for node in tree.body:  # top-level (module-scope) statements only
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                # `from __future__ import ...` is not a runtime dependency.
                if node.module and node.module != "__future__":
                    imported_roots.add(node.module.split(".")[0])
        assert imported_roots, "expected cronsim.py to import something"
        for root in imported_roots:
            assert root in sys.stdlib_module_names, (
                f"cronsim.py imports non-stdlib module {root!r} "
                f"(vendored engine must be pure-stdlib at import time)"
            )

    def test_a13_explain_py_not_vendored(self):
        # A13: only cronsim.py (parser + forward iterator) is vendored;
        # explain.py is NOT (DESIGN §5: "We copy only cronsim.py, NOT explain.py").
        assert not (self._vendor_dir() / "explain.py").exists()
