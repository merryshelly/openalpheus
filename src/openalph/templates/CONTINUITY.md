# CONTINUITY.md

You are responsible for maintaining two continuity artifacts: `progress.md` and `durable-set.toml`
These artifacts are critically important to persist important context across garbage-collection boundaries, therefore enabling autonomous long-horizon work.

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
garbage-collection boundary, the harness re-injects the listed files verbatim (live-read at boundary
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
- Update `durable-set.toml` as needed.
- Actively update relevant documentation so that you receive accurate information
  after a garbage-collection boundary.
- If conducting session-handoff: the session brief references the artifacts — do not duplicate.