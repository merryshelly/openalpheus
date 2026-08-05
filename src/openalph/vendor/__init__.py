"""Vendored third-party modules (copied source, no pip dependency).

Provenance policy: each vendored file carries a header pinning the upstream
repo + commit SHA it was copied from, and the upstream license text lives
alongside it. Vendored code is excluded from ruff (pyproject extend-exclude).

Current contents:
- cronsim.py -- cuu508/cronsim @ 375c57a0dc725a636fbf8322bf90f8092f255330,
  BSD-3-Clause (LICENSE.cronsim). Cron expression parser + next-match
  iterator backing openalph/schedule.py (kdsn.210.6).
"""
