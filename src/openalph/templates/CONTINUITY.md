# CONTINUITY.md

You are responsible for maintaining two continuity artifacts: `progress.md` and `durable-set.toml`
These artifacts are critically important to persist context across **context-handoff boundaries** — they are the ONLY thing that survives a boundary.

## How the boundary works

Context pressure is measured against usable runway (model window minus max_tokens):

- **75% — checkpoint.** The harness injects a checkpoint directive: stop starting new work and update your artifacts NOW.
- **85% — boundary.** ALL pre-boundary conversation is dropped from your context. The handoff package — your declared project's `progress.md` + `durable-set.toml` snapshot — is the only carryover. Tool outputs, reasoning, and conversation history do not survive. The snapshot is frozen at the boundary, so artifact updates must land BEFORE it fires.
- **92% — overflow guard.** Emergency floor with a 3-strike breaker, not a grace band — a single fast-growing turn can cross it between checkpoint evaluations.

If no project is declared, the boundary strips anyway and the snapshot is a bare rehydration pointer: your workspace, `memory/`, and beads are the recovery path. In isolated environments where the template files below are unreadable, the checkpoint directive embeds condensed skeletons — reproduce them. If the checkpoint is skipped, the boundary fires anyway and the carryover is whatever the files last contained — the manifest records it `stale`.

## The Artifacts

When working on any project, i.e., any work or conversation that will extend beyond a single session, you must scaffold TWO files:

- `memory/projects/<project>/progress.md` — the human-readable artifact, from
  `/srv/openalph/shared/templates/progress-template.md`
- `memory/projects/<project>/durable-set.toml` — the machine contract, from
  `/srv/openalph/shared/templates/durable-set-template.toml`

Do this at session start or the first milestone. 
Commit both files to git immediately after creation and after every milestone update.

### progress.md MUST contain exactly four blocks, in order:

1. **State** — what exists and works now. Every claim must carry observable verification (command, expected output, exit code). Unverified claims must be marked unverified.
2. **Story** — how the current State was reached: a chronological list of 5–15 one-line factual bullets, oldest first. What the operator asked, what was built or dispatched, what returned, what is in flight right now, and what is explicitly not done. Rules:
   - First-person factual: "SB ordered X; I dispatched Y (model, time); it returned Z." No rationale (that's Decisions), no status prose (that's State).
   - **Close every loop in writing.** Every in-flight or not-yet-started item carries an explicit status word — DISPATCHED, IN FLIGHT, NOT STARTED, BLOCKED. A post-boundary session has no memory of this narrative; an open loop left implicit invites it to be closed by invention.
   - Completion claims must point at their verification evidence in State; the Story never substitutes for it.
   - Prune mercilessly: compress settled arcs into one line ("kdsn.329 shipped + promoted — see README Session Log") rather than deleting bullets outright.
3. **Decisions** — what was decided and *why*.
4. **Next** — immediate next steps, pointing at beads where relevant. Any item that cannot be started right now carries its unblock condition and its owner — `[blocked: SB]` / `[blocked: external]` / `[blocked: agent-name]`; an implicit wait is the one thing the next epoch most needs to not walk past.

Keep it ~1–2K tokens (Story ≤ ~500 of it). It is a carrier bag for critical context, not a substitute for
documentation (README) or history (daily notes) or tasks (beads).

### durable-set.toml

The re-inject list: a TOML array of `[[entries]]` with `path` and `reason`. After a
context-handoff boundary, the harness re-injects the listed files verbatim (live-read at boundary
time). `progress.md` and `durable-set.toml` are auto-injected; do not list them.

Required categories:
- Project README
- Spec, architecture, and roadmap docs for the active workstream
- Skills governing the active work

## Discipline

- Declare the project to the harness at session start with the `set_active_project`
  tool. One project per room; if the work genuinely needs a second project, escalate to your operator rather
  than switching or blending.
- Update `progress.md` after every milestone. "Done" means verified complete, all tests green.
- Update `durable-set.toml` as needed — treat the 75% checkpoint as the deadline, not the boundary.
- **Boundary readiness for manual fires.** The 75% checkpoint fires on pressure, not on operator-requested
  handoffs. Before any tool- or slash-fired boundary: run bead hygiene (`bd` statuses current, discovered
  work filed), then update the Story block so every in-flight item carries its status word — the Story
  references truth, not stale memory. A boundary fired on a stale Story hands the next epoch a vacuum.
- Actively update relevant documentation so that you receive accurate information
  after a context-handoff boundary.
- If conducting session-handoff: **load the `session-handoff` skill first** and follow it in full —
  its artifact steps (durable-set review, progress.md Story sweep) are the manual-boundary version of this
  discipline. The session brief references the artifacts — do not duplicate.
- The durable-set budget is advisory (an informational pruning marker, not a cap). Act on it only when the
  harness reports it exceeded — never anticipate or infer an overage. When reviewing the durable set, the
  question is "what does the NEXT epoch need injected?" not "how do I shrink this?" — files that govern
  imminent work (e.g. deployment skills before a deploy phase) belong in the set even when large.
