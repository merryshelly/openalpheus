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

### progress.md MUST contain exactly three blocks, in order:

1. **State** — what exists and works now. Every claim must carry observable verification (command, expected output, exit code). Unverified claims must be marked unverified.
2. **Decisions** — what was decided and *why*.
3. **Next** — immediate next steps, pointing at beads where relevant.

Keep it ~1–2K tokens. It is a carrier bag for critical context, not a substitute for
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
- Actively update relevant documentation so that you receive accurate information
  after a context-handoff boundary.
- If conducting session-handoff: the session brief references the artifacts — do not duplicate.
