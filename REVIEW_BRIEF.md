# Review brief — PR2 "shipped artifacts" (OpenAlpheus)

You are reviewing a single git commit for correctness before it is merged into
the production `main` of OpenAlpheus (a multi-agent platform). Your working
directory IS a checkout of the PR2 branch. The commit is also captured as a diff
at `./THE_PATCH.patch` (read it first to see exactly what changed).

## Critical context (scoping)
OpenAlpheus is **primarily internal-use** — the operator has said "we're the only
ones who currently use it." It was bootstrapped once; the live agents run under
hand-maintained systemd units. So weight every finding by **"does this affect us,
who already run the platform?"** vs **"this is pure public-release cosmetics."**
Label each change PRACTICAL (affects our running system or a real correctness bug)
or COSMETIC (only matters for a fresh public install we don't do). Do NOT dismiss
correctness bugs just because they're in install paths — flag them, but scope them.

## What the commit claims to do (verify each independently)
- **BUG-1/ARCH-3** — one canonical systemd unit at `src/openalph/data/openalph@.service`
  (shipped as package data), `ExecStart` fixed so it no longer crash-loops; `etc/openalph@.service`
  becomes a symlink to it; installer copies the packaged file. `tests/test_systemd.py`
  rewritten. VERIFY: is the new ExecStart correct? Does the symlink resolve? Does the
  test assert the *right* behavior (not just pass)?
- **SEC-4** — agent Matrix password no longer written to a world-readable bootstrap log;
  goes to /dev/tty or a 0600 file; log created 0600. VERIFY in install.sh: is the password
  truly kept out of any world-readable artifact on every path?
- **SEC-5** — an EXIT/ERR trap re-locks Matrix registration on every exit path. VERIFY:
  does the trap actually fire on all exits (including error/early-exit)? Any path that
  leaves registration unlocked?
- **PHIL-2** — `httpx` declared as a dep; `llama-cpp-python`/`sqlite-vec` moved to a
  `[memory]` extra installed by default with a checksum-gated model fetch; memory search
  emits an in-room notice when semantic search is unavailable instead of degrading silently.
  VERIFY: the `src/openalph/tools/memory_search.py` change — is the "unavailable" notice
  correct, and does it regress normal (available) operation? Check pyproject.toml/requirements.
- **BUG-16** — installer reads version from package metadata instead of a nonexistent
  `--version` flag. VERIFY it reads the right thing.

## Also flag
- The `OPENALPH_MODEL_SHA256` is intentionally EMPTY (`TODO(release)`) → installer SKIPS
  the model download rather than trust an unverified file. Is that a safe default? Does it
  leave memory-search broken on a fresh install (relevant only if we ever re-bootstrap)?
- Any place the patch is internally inconsistent (e.g. installer references the old etc/
  path, or the symlink + package-data don't line up).

## Verify the tests
Run the full suite in this worktree:
`cd <this dir> && PYTHONPATH=src /opt/openalph/.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -5`
It should be green. Then specifically read `tests/test_systemd.py` and confirm it asserts
correct behavior, not a broken contract.

## Output
Write your review to `./REVIEW_FINDINGS.md` (in this worktree). Structure:
- Per-change verdict: CORRECT / BUG / RISKY, with PRACTICAL|COSMETIC label, file:line, and
  a one-line why.
- Any CRITICAL/HIGH correctness or security issue called out at top.
- Bottom line: is PR2 safe to merge as-is? If not, what must change?
Cite specific file:line for every claim. Don't pad; if a change is fine, say so briefly.

---
Begin by restating what you're reviewing, then summarize your verdict. Keep the announcement short — detail goes in the file.
