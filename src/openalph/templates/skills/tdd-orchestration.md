<!-- Shipped with OpenAlph. Customize for your setup. -->
# TDD Orchestration

For any multi-component build (not quick fixes or single-file changes), use TDD orchestration.

## The Process

### 1. Main session designs architecture + writes the test suite

This is the highest-leverage step. The orchestrator defines:
- Component interfaces
- Edge cases and failure modes
- Integration tests between components
- Acceptance criteria

**The tests are the specification.** A complete test suite means implementors can work independently and bugs are caught before they reach the operator.

### 2. Sub-agents implement against the tests

Each sub-agent gets:
- A task brief (what to build)
- The test file(s) (pass/fail criteria)
- Clear scope boundaries

They iterate until green. This is parallelizable.

Use the `subagent` tool to spawn implementors. Subagents are multi-turn, inherit parent tools (except subagent itself — no recursion), and have a circuit breaker iteration limit.

### 3. Bugs are caught by tests, not by the operator

If a bug reaches the operator, the test suite was incomplete. Fix the test gap first, then fix the bug.

## Why It Works

| Scenario | Cost |
|----------|------|
| Orchestrator debugging downstream | Most expensive failure mode |
| Orchestrator writing tests upfront | Highest-leverage use of main session |
| Many sub-agent turns against a test suite | Fraction of the cost of main-session debugging |

As inference costs trend toward zero, value concentrates in **architecture and test design** — not implementation.

## When to Skip TDD

- True spikes/experiments where requirements aren't clear yet
- Single-file changes
- Anything where writing the test takes longer than writing the code

Use judgment. TDD is a force multiplier, not a ritual.

## Checklist

Before handing off to subs, verify:
- [ ] All component interfaces are defined
- [ ] Integration tests cover component boundaries
- [ ] Edge cases and error paths are tested
- [ ] Each test has a clear pass/fail signal
- [ ] Task briefs reference the exact test file(s)
