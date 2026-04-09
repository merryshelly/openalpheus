<!-- Shipped with OpenAlph. Customize for your setup. -->
# Task Briefs for Sub-Agents

Sub-agents build the shape of what you ask for, then claim success. Your job is to make failure **observable to the sub itself**.

---

## OA Subagent Model

- **Tool:** `subagent` — multi-turn, not a separate session
- **Inheritance:** Parent's tools minus `subagent` itself (no recursion)
- **Circuit breaker:** Configurable iteration limit (default 100)
- **Model:** Inherits parent's provider/API key, supports model override
- **Workspace access:** Can read/write files in parent's workspace
- **No separate session:** Single turn-based conversation with tool isolation

---

## Checklist

### ✅ Define "done" as an observable state, not a claim

| ❌ Claim | ✅ Observable |
|---|---|
| "Tests pass" | "This command exits 0 in under 30s" |
| "Deployed and running" | "`curl` returns 200" |
| "File was updated" | "`grep 'expected string' path/to/file` exits 0" |

### ✅ Write self-checking verification commands

Bad:
> "Run the tests and make sure they pass."

Good:
> "Run: `timeout 30 node --test test/**/*.test.js 2>&1; echo EXIT_CODE: $?` — if output does not end with `EXIT_CODE: 0`, you are not done. Iterate until it does."

### ✅ Name anticipated failure modes

Subs fix problems they're told to look for. One sentence of priming saves a full round-trip:
- "Watch for dangling async handles that keep the event loop alive."
- "This API returns 200 even on auth failure — check the response body for `error`."

### ✅ Provide explicit schemas at integration boundaries

Don't let subs infer field names — they'll guess wrong (`attacker_ip` vs `source_ip`). Include the actual JSON shape or a sample response.

```json
// Include this in the brief:
{ "source_ip": "1.2.3.4", "severity": "high", "ts": 1700000000 }
```

### ✅ Include full tool paths for non-standard environments

The subagent inherits your environment, but explicit paths avoid ambiguity:
- `/srv/openalph/shared/bin/bd` for beads
- Absolute paths for any custom scripts

### ✅ Include all cross-reference IDs

Subs fabricate plausible-looking IDs rather than flag uncertainty. Either:
- Include every ID the deliverable might reference (bead IDs, epic IDs, issue numbers), **or**
- Instruct the sub to look them up: `bd search "keyword"` rather than guess.

---

## Required Prompt Structure

Every subagent task prompt must include:

1. **Full absolute output path** — place in the right home:
   - `memory/projects/<project>/` — project deliverables
   - `memory/ad-hoc-research/` — one-off research
   - `tmp/` — ephemeral/scratch
   - External git repo — code projects

2. **Self-contained announcement block** at the end:

```
---
Begin your response by briefly restating the task you were given, then summarize your findings. The receiving session may have lost context.
```

---

## Subs Write, You Read

Sub-agents have `file_write`, `file_read`, `shell`, and all other parent tools (minus `subagent`). **Always have the sub write its output to a file** rather than returning it as response text. This:

- Avoids bloating main session context with large outputs
- Lets you verify the output exists and spot-check with `file_read` + `offset`/`limit`
- Survives if the sub hits its iteration limit mid-work (partial file > no file)

❌ **Don't:** "Produce the markdown content for a project README" (sub returns it as text, you have to copy-paste it into a file)

✅ **Do:** "Write a project README to `memory/projects/foo/README.md`" (sub writes it directly, you verify)

---

## What Subs Get Automatically

Every subagent receives a **safety preamble** as its system prompt. This is injected by the framework — you don't need to include safety rules in your brief.

If you pass a custom `system_prompt`, it's appended after the safety preamble.

## Subagent Skills

Compressed skills for specific task types. When briefing a sub, **read the relevant skill and include it in the `system_prompt` parameter**:

| Task type | Skill file | When to use |
|-----------|-----------|-------------|
| `subagent/implement.md` | TDD coding | Feature implementation with tests |
| `subagent/deploy.md` | Deploy to host | Copying binaries, restarting services |
| `subagent/debug.md` | Fix bugs | Debugging failures, fixing tests |

**How to include:** Read the skill, pass it as `system_prompt`. The framework prepends the safety preamble automatically.

Don't include skills for mechanical tasks (file moves, simple lookups). Only include when the task type matches and failure modes are relevant.

## What Subs Don't Have

Sub-agents do NOT read workspace files or receive the parent's system prompt. They get:
- The **task string** you provide
- The **system prompt** you provide (with safety preamble auto-prepended)
- The parent's **tools** (minus subagent)
- The parent's **provider/API key**

They do **not** get: workspace prompt files, memory files, or session history.

Include any context the task requires directly in the task string — the sub has no other source of truth.

---

## Model Selection

The `subagent` tool's `model` parameter requires **fully qualified `provider/model` strings** — e.g., `anthropic/claude-sonnet-4-6`, not `sonnet` or `claude-sonnet`. The error message will tell you if you get it wrong, but save yourself the round-trip.

Consult your `docs/model-roster.md` for the full list of available models and their canonical strings.

When in doubt, omit the `model` parameter entirely — the sub-agent inherits the parent's model.

## Gotchas

| Issue | Solution |
|-------|----------|
| **No recursion** | Sub can't spawn another subagent — must complete task directly |
| **Iteration limit** | Complex tasks may hit this; break into smaller pieces or provide more context upfront |
| **Shared workspace** | Sub can read/write parent's files — coordinate paths to avoid conflicts |
| **Tool isolation** | Sub doesn't have `subagent` tool — can't delegate further |

---

## Example Brief

```
You are a subagent tasked with analyzing API response formats.

**Task:** Fetch the API documentation from https://api.example.com/docs and extract the schema for the `/users` endpoint. Write a TypeScript interface that matches the response shape.

**Output path:** `tmp/user-api-interface.ts`

**Verification:**
1. File must exist at the output path
2. File must contain `interface User`
3. Interface must have at least: `id`, `email`, `created_at` fields

**Expected failure modes:**
- API docs may require authentication — check for 401 response
- Response structure may be nested under a `data` key — check the actual shape

**Schema hint:**
```json
{ "id": 123, "email": "user@example.com", "created_at": "2024-01-01T00:00:00Z" }
```

---
Begin your response by briefly restating the task you were given, then summarize your findings. The receiving session may have lost context.
```
