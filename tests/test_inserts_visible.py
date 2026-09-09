"""Structural pin: the inserts-visible invariant (kdsn.333 follow-up).

Invariant (operator hard requirement 2026-07-04, violated silently until
2026-09-08): every byte the HARNESS inserts into the model's context must
also be visible to the operator in the Matrix room, collapsed.

The failure class this guards: a framing redesign changes what the model
sees (kdsn.333's directive snapshot) while the operator notice keeps the old
shape — nobody notices until the operator reports missing content.

Two layers:
1. DISCOVERY (lint): scan src/openalph for JSONL `source=` tags. Every tag
   must be either REGISTERED (harness-authored content → has a declared
   notice surface, pinned below) or EXEMPT (operator/config-authored or
   non-content metadata → visible by authorship). An unregistered tag fails
   with instructions.
2. SURFACE (behavioral/lint hybrid): each registered source's notice path
   must actually exist and carry the inserted bytes.
"""

import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "openalph"

# Harness-AUTHORED content sources → their room-notice surface. Any source
# tag on a user-role JSONL entry whose content the harness composed MUST be
# registered here with a working surface.
REGISTERED = {
    "reminder": "🔔 System reminder notice (agent.py emit path)",
    "handoff_snapshot": "🪢 render_handoff_notice (callbacks.py) — fold carries the full framed snapshot",
    "spotter": "spotter._send_notice advisory",
    "view_image": "log_vision_injection callback + room notice",
}

# Non-authored or non-content tags — visible by authorship, no surface owed:
EXEMPT = {
    "steer": "operator-authored (the operator wrote it in the room)",
    "heartbeat": "operator-configured timer directive (authored at /heartbeat start)",
    "umbral": "operator-configured timer directive (authored at /umbral start)",
}

_SOURCE_RE = re.compile(r'source="([a-z_]+)"')  # kwarg-only: JSONL append call sites; spaced assignments (locals) are not tags


def _discover_sources():
    found = {}
    for py in sorted(SRC.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        text = py.read_text(errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            for tag in _SOURCE_RE.findall(line):
                found.setdefault(tag, []).append(f"{py.name}:{i}")
    # The snapshot source rides a named constant, not a literal:
    if "HANDOFF_SNAPSHOT_SOURCE" in (SRC / "handoff.py").read_text():
        found.setdefault("handoff_snapshot", []).append("handoff.py (constant)")
    return found


class TestDiscovery:
    """Any new source tag entering context fails until registered."""

    def test_no_unregistered_source_tags(self):
        known = set(REGISTERED) | set(EXEMPT)
        unknown = {tag: locs for tag, locs in _discover_sources().items()
                   if tag not in known}
        assert not unknown, (
            "New context source tag(s) with no declared notice surface:\n"
            + "\n".join(f"  {tag!r}: {locs[:3]}" for tag, locs in unknown.items())
            + "\nIf the content is harness-authored and enters model context, "
            "add a room-notice surface and register it in REGISTERED "
            "(tests/test_inserts_visible.py). If operator/config-authored, "
            "add to EXEMPT with a one-line justification.")

    def test_registered_sources_actually_exist(self):
        found = _discover_sources()
        for tag in REGISTERED:
            assert tag in found, (
                f"REGISTERED source {tag!r} not found in the tree — "
                "renamed? update the registry, don't leave a stale entry")


class TestNoticeSurfaces:
    """Each registered source's declared surface must exist in the tree."""

    def test_reminder_notice_emit(self):
        text = (SRC / "agent.py").read_text()
        assert "🔔 System reminder" in text

    def test_handoff_snapshot_notice_renders_full_snapshot(self):
        text = (SRC / "callbacks.py").read_text()
        assert 'outcome.get("snapshot")' in text, (
            "render_handoff_notice must prefer the full framed snapshot — "
            "folding only progress.md hides the directive block (the "
            "2026-09-08 incident this invariant was raised over)")

    def test_spotter_notice_surface(self):
        text = (SRC / "spotter.py").read_text()
        assert "_send_notice" in text

    def test_view_image_notice_surface(self):
        text = (SRC / "agent.py").read_text()
        assert "log_vision_injection" in text

    def test_snapshot_surface_is_behavioral_not_decorative(self):
        """The surface must carry the inserted bytes end-to-end: boundary
        outcome carries the framed snapshot AND the notice renders it. See
        tests/test_handoff_notice_render.py::TestSnapshotDirectiveVisibility
        for the behavioral pins — this asserts the wiring seam between them
        isn't a stub."""
        from openalph.callbacks import render_handoff_notice
        out = render_handoff_notice("tool", {
            "applied": True,
            "manifest": {"boundary_index": 1, "durable": {"project": "p",
                         "files": [], "budget_tokens": 100, "used_tokens": 0,
                         "over_budget": False},
                         "errors": []},
            "snapshot": "UNIQUE_DIRECTIVE_NEEDLE_1234",
        })
        assert "UNIQUE_DIRECTIVE_NEEDLE_1234" in out
