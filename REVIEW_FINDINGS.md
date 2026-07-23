# Review findings — PR2 "shipped artifacts" (OpenAlpheus)

Reviewing: commit `299c1417` ("fix: make the shipped artifacts match the code
(BUG-1, SEC-4, SEC-5, PHIL-2)") on the PR2 branch, checked out at
`/home/oa-merry/workspace/tmp/pr2-tree`. Diff reviewed from `THE_PATCH.patch`
and cross-checked against the actual working-tree files. Scope note: this
platform is primarily internal-use (the operator already runs it under
hand-maintained systemd units), so findings are labeled PRACTICAL (affects
the running system / a real correctness bug) or COSMETIC (only matters for a
fresh public install we don't do).

## Top-level critical/high findings

(filled in below as investigation proceeds)

## Per-change verdicts

### BUG-1 / ARCH-3 — canonical systemd unit

- TBD

### SEC-4 — password no longer logged

- TBD

### SEC-5 — registration re-lock trap

- TBD

### PHIL-2 — deps + memory_search degradation notice

- TBD

### BUG-16 — version from package metadata

- TBD

### Internal consistency / "also flag" items

- TBD

## Test suite

- TBD

## Bottom line

- TBD
