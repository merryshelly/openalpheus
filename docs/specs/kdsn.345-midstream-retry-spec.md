# Spec: Mid-Stream Provider Retry with Display Reset (kdsn.345)

**Bead:** workspace-kdsn.345 (parent: workspace-kdsn)
**Status:** Implemented — 23/23 suite green, 4319 passed full suite, ruff clean, sabotage cycle proven
**Author:** Merry (orchestrator); implementation via TDD sub (qwen38blackwell) + orchestrator verification
**Date:** 2026-09-15
**Ratification:** SB, this session ("Got it. Build workspace-kdsn.345") — formally reverses the
operator-stated kdsn.287 atom "mid-stream = fatal" and reopens kdsn.219 territory on measured evidence.

## Measured basis

- 21 mid-stream failures Aug 28–Sep 15 fleet-wide (15 × `RemoteProtocolError`, 6 × mid-stream
  timeout); ALL on Synthetic (merry GLM-5.3-Flash rooms + babson Kimi-K3 room), zero on
  Fireworks/blackwell/local. ≈0.18% of 11,514 model calls, clustered (6 on Sep 13 alone).
- **All 21 had ≥1 chunk yielded** (min 1, max 1141) — the kdsn.287 zero-yield retry gate would
  have retried zero of them. Each failure stalled the room until operator jumpstart; worst cost
  is unattended heartbeat/umbral turns.
- Raw journal extract: `workspace tmp/midstream-errors.txt` (session 2026-09-15).

## Design (minimal, advisor-reviewed)

`provider.stream()` gains `on_stream_reset: async (info: dict) -> None | None = None`.

On a mid-stream `httpx.TimeoutException`/`httpx.HTTPError` (≥1 chunk consumed):
- seam unwired, `retry_enabled=False`, budget exhausted, or mid-stream SDK `APIStatusError`
  → `ProviderError` exactly as today (kdsn.287 messages preserved).
- otherwise → `logger.warning`, `await asyncio.sleep(5.0)`, `await on_stream_reset(info)`,
  re-issue the FULL request from scratch. One extra attempt (`_MIDSTREAM_MAX_RETRIES = 1`).

Info dict: `{"error": "<ExcName>: <msg>", "chunks": int, "attempt": int}` (attempt is 1-based
retry number). A raise inside `on_stream_reset` aborts the retry (fail-safe against a garbled
display). `CancelledError` during the backoff sleep propagates — /stop stays live.

**Zero-yield failures are NEVER retried at the OA layer** — the SDK's `max_retries=20` owns
establishment (kdsn.220, ~2.25 min worst-case window); an OA retry would stack on top of it.
(Divergence from the original bead wording "2 zero-yield attempts" — deliberate simplification,
commented on the bead.)

**History safety:** the assistant turn is appended to history only after the stream completes and
tools execute only after completion — a mid-stream death leaves zero side effects, so restarting
the full request is state-safe. Identical request prefix → provider prompt-cache read on retry.

### Per-attempt state (both provider families)

Re-initialized at the top of each attempt: `_stream_count`, `_hardened_bytes`, `accumulated_text`,
`accumulated_reasoning`, `usage`, `stop_reason`, `generation_id`, `degen_aborted`,
`tool_call_accumulators`, and a **fresh `DegenerationMonitor`** (stale n-gram state from the dead
attempt must not contaminate the retry).

### Wiring (the gate)

- `agent.py` (main tool-loop call site ONLY — continuation ~2295 and forced-summary ~2788 stay
  fatal-today, noted residual): defines a `nonlocal`-resetting closure and passes
  `on_stream_reset=...` **only when `callbacks` carries `'on_stream_reset'`** — live + heartbeat
  paths. Subagent / CLI / keepalive consumers stay fatal-today.
- `matrix.py` live path: `_stream_reset` = `StreamingDelivery.reset()` (annotate the partial
  message — cursor stripped, " ⟳" appended, replace-edit, fail-soft) + one
  `⏳ Stream dropped mid-flight (… ) — retrying once…` notice. Retry text starts a FRESH message.
- `matrix.py` heartbeat/umbral path: notice-only (background turns have no streaming display).
- `config.py`: single per-provider kill switch `retry_enabled` (default `True`); invalid type
  skips the provider per the established `_skip_provider` pattern. NO other config surface —
  kdsn.219 died on config-scope scope creep; this build is deliberately hard-coded.

## Tests

`tests/test_midstream_retry.py` (23 tests, self-contained): provider-layer retry/success,
timeout-class retry, event ordering (text → reset → text → done), backoff sleep, exhaustion,
unwired-fatal guard, `retry_enabled=False` guard, zero-yield-never-retried guard,
mid-stream-APIStatusError-not-retried guard, cancellation propagation, tool-call state leak,
degen re-instantiation, anthropic path (success + exhaustion + unwired), `StreamingDelivery.reset()`
(annotation/clear/fail-soft), agent wiring (passes seam + resets accumulators + forwards info;
None when callbacks lack the key; None when callbacks=None), matrix wiring (live room contract:
partial → ⟳-edit → notice → fresh message → final answer; heartbeat notice-only).

Note: this suite flips the kdsn.287 spec's test-7 invariant ("mid-stream failure is fatal") for
the wired case — the fatal contract is preserved for unwired consumers.

## Accepted residuals

- Continuation-call and forced-summary mid-stream failures stay fatal (rare paths; same fix shape
  if ever needed).
- Mid-stream SDK `APIStatusError` (e.g. 529 as SSE error event) not retried — out of measured scope.
- Double-billing on retry after server-side completion: accepted noise at 0.18%.
- Subagent/CLI consumers: fatal-today (stigmergy has its own retry layer; keepalive has its own
  abort logic).

## Deploy

Per OPERATIONS.md order: **babson canary first** (Synthetic consumer — the measured failure
population), verify, then remaining domain agents, then Merry (operator-executed restart).
Kill switch: `retry_enabled = false` on the provider block in any agent TOML.
