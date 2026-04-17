# Week-1 Commits Independent Review (2026-04-17)

## Summary

All four commits deliver what their messages claim, and the scoped edits
to `minimal_alphazero_loop.py` (A1, A2) are minimal, non-destructive, and
behind tracker-or-None guards so disabling the quality gate still works.
I did not find any CRITICAL or HIGH issues. A handful of MEDIUM/LOW/NIT
items are listed below, primarily around misleading test/comment
semantics in the plateau detector, a subtle race in how the A2 detector
reads its own just-written JSONL line, and a few label/telemetry edge
cases worth tightening before they matter in production.

---

## Per-commit findings

### 98736c566 — A1 per-seat WR tracking

**Overall:** Correct, well-scoped, and does exactly what the plan asks
for. The per-seat outcome tracking is wired into both `evaluate()` and
`staged_evaluate()`, the gate is **diagnostic-only** (never returns
critical), and the `tracker is not None` guard preserves
`--skip-quality-gate` behavior.

- **Correctness (OK).** `candidate_player = (i % num_p) + 1` guarantees
  fair seat rotation across games, so with ≥50 games the per-seat sample
  threshold (`SEAT_FAIRNESS_MIN_GAMES_PER_SEAT = 10`) is met for all
  currently used `_EVAL_STAGES_*` first-stage counts (17/13/25 for
  3p/4p/2p). For 3p at 10 games the gate correctly skips with
  `min_games_per_seat < 10`.
- **Correctness (OK).** Draws are treated as "not a win" for the
  candidate, matching the docstring. This folds draws into each seat's
  loss column symmetrically, so the ratio is still an apples-to-apples
  signal across seats.
- **Correctness (OK).** `eff_min = max(min_wr, 1e-3)` prevents divide-
  by-zero for zero-win seats while still producing a ratio comfortably
  above the 1.5 threshold — this is exactly the canonical square8_3p
  failure mode and `test_zero_wins_handled` exercises it.
- **Safety (OK).** Touches to `minimal_alphazero_loop.py` are two
  six-line blocks at the end of the per-game loops, behind
  `if tracker is not None`, after the existing winner-accounting is
  already done. No state mutation that the loop relies on downstream.
  Respects the CLAUDE.local.md "do not modify the minimal loop" spirit
  — these are the narrowly scoped edits called out as intentional in
  the plan (item A1).
- **Observability (OK).** The new block lives entirely in
  `verdict.details["seat_fairness"]` and a warning line; existing
  quality-gate telemetry keys are unchanged.
- **Tests (good).** Nine tests cover: canonical square8_3p imbalance,
  healthy 3p balance, mild variance tolerance, zero-wins edge case,
  single-seat 2p (no ratio), below-min sample skip, verdict surfacing,
  and a regression guard on the defaults. Happy-path coverage and the
  three interesting edge cases (zero wins, single seat, under-sample)
  are all tested.

**Findings:**

- **NIT** — The docstring on `record_game_outcome` says "loss or draw"
  is False, but callers only call it on games that have been scored or
  declared a draw by the eval loop. Unrecorded-abort games (where
  `w is None` AND `state.game_status != COMPLETED`, e.g. `MAX_MOVES`
  exceeded) are still recorded as "candidate_won=False" by the new
  code, i.e. charged as a loss to the candidate's current seat. This
  matches the existing `dr += 1` draw accounting but slightly biases
  the per-seat WR downward for configs that hit the move cap often.
  Probably fine for the diagnostic but worth a line of comment.
- **LOW** — `_check_seat_fairness` reads `tracker._seat_games` /
  `tracker._seat_wins` directly (private attributes). The existing
  `_check_behavioral_diversity` / `_check_value_head_health` functions
  follow the same pattern so this is consistent with the module's
  conventions, but it does mean any future refactor of
  `QualityGateTracker` must update all three call sites in lockstep.
  Consider a read-only accessor later.
- **NIT** — The warning string `SEAT_WR_IMBALANCE: max/min per-seat WR
ratio 2.60 > 1.5 (seat1=58% (29/50), seat2=22% (11/50), seat3=28%
(14/50))` is informative but quite long; likely fine since it is a
  warning, not an error, and is only emitted for the known failure
  mode.

### 24bf557b9 — A2 plateau detector

**Overall:** Clean, pure-logic module with a good unit-test suite. The
loop integration is diagnostic-only as claimed (`if plateau.detected:
logger.warning(...)`) — no threshold adjustments yet. Defaults match
the plan doc.

- **Correctness (OK).** Both triggers gated by `rate_fired AND
staleness_fired`. Min-iterations gate prevents cold-start false
  positives. Never-promoted configs are handled explicitly
  (`iter_since_promotion = last_iter_seen`).
- **Correctness (OK).** Malformed-history handling is reasonable:
  missing `iteration` falls back to 1-based index, missing `promoted`
  key excludes the entry from the decided-window rate calculation
  (`decided = [m for m in recent if "promoted" in m]`).
- **Safety (OK).** The loop re-reads `logf` _after_ appending the
  current iteration's metrics, so the just-written entry is included
  in the history and `detect_plateau` sees the latest state. The
  `(OSError, ValueError, json.JSONDecodeError)` guard is appropriately
  narrow and demoted to `logger.debug`, so a transient read failure
  will never break the training loop.
- **Observability (OK).** When detected, the structured warning
  includes `config`, `iter`, `last_promoted`, `total_iters` — all the
  fields needed to correlate with `metrics.jsonl` after the fact.

**Findings:**

- **MEDIUM** — **Reads its own write every iteration.** The loop does
  `open(logf, "a"); f.write(...)` and then `open(logf) as f; history =
[json.loads(line) for line in f if line.strip()]`. This works in
  practice because POSIX `fsync` is not required for the read to see a
  newly appended line from the same process, but:
  1. On slow disk / NFS backends, there is a small window where the
     trailing line can be observed truncated (empty line filter
     helps, but a partial line will raise `json.JSONDecodeError` and
     silently skip detection).
  2. The loop already has `metrics` in memory — re-reading the whole
     file per iteration when the in-memory `history` could be
     maintained as a growing list is mildly wasteful.
     Not a correctness bug; in the worst case, detection is delayed by
     one iteration. Worth a comment at least, and considering a
     `history.append(metrics)` in-memory approach.
- **LOW** — `test_rate_not_high_enough_does_not_fire` has a misleading
  block comment: "the 'or' semantics require BOTH triggers to fire
  strictly above / at thresholds" — but the code uses `>=`, so boundary
  rate _does_ fire. The test passes because the staleness gate fails
  (`iter_since_promotion=2 < 15`), not because of "or semantics". The
  test name is also misleading; it should be
  `test_high_rate_but_fresh_promotion_does_not_fire` or similar. Only
  a comment / naming issue, not a behavior bug.
- **LOW** — `test_missing_promoted_key_excluded_from_rate` constructs
  a history where the last 10 entries have no `promoted` key →
  `window_size=0`. Because `min_iterations=20` check happens _before_
  the trigger-fire decision but _after_ rate/staleness computation,
  the result's `reason` returns the "no plateau" string with no parts,
  even though `window_size=0` and `rate=0.0` look uninformative. Not
  wrong, just slightly surprising. A trivial `reason` branch for
  `window_size == 0` would improve log clarity.
- **NIT** — `PlateauResult` is `frozen=True`, but
  `test_result_is_frozen_dataclass` catches `(AttributeError,
Exception)` — `Exception` covers `FrozenInstanceError` but the
  double-catch is redundant.
- **NIT** — `staleness_threshold <= 0` raises, but the message says
  "must be > 0" while the other validators use lowercased `must be`.
  Cosmetic.

### cbcd73baa — D1 model-version telemetry

**Overall:** Does what the commit message claims, with defensive
guards that make it safe to drop into the hot path. Cardinality is
bounded. Backward compatible.

- **Correctness (OK).** `_extract_model_version` walks
  `ai.neural_net.model` → `ARCHITECTURE_VERSION` → registry. Every
  layer is wrapped in a `try/except`. Empty-string
  `ARCHITECTURE_VERSION` is correctly rejected by
  `if isinstance(version, str) and version:` before falling through.
  The fallback uses `get_model_version(model)` which returns
  `"v0.0.0"` for unknown classes — a _single_ bounded value, so it
  does not pollute label cardinality.
- **Correctness (OK).** `nn_model_version = _extract_model_version(ai)
if effective_use_neural_net else None` short-circuits correctly when
  the request intentionally disables neural nets.
- **Observability / cardinality (OK).** Labels: `model_version`
  (≤10 known strings + `"none"` + `"init"`), `ai_type` (≤8 AIType enum
  values + `"init"`), `difficulty` (≤10 tiers + `"init"`). Worst-case
  ≈ 12 × 9 × 11 ≈ 1,200 series, well within Prometheus budget and
  bounded by finite enums on every label.
- **Observability (OK).** The `X-RingRift-Model-Checkpoint` header
  uses `os.path.basename(path)`, so the full filesystem path is not
  leaked — filenames only. No PII concerns; no user identifiers in
  any label or header.
- **Safety (OK).** `response: Response` is a new required signature
  parameter, but FastAPI injects it automatically via DI — no caller
  signature changes. The try/except around
  `AI_MOVES_BY_MODEL_VERSION.labels(...).inc()` means a metrics-lib
  issue cannot break move selection.
- **Tests (good).** The helper test suite covers: missing `neural_net`,
  `neural_net=None`, `model=None`, explicit `ARCHITECTURE_VERSION`,
  fallback to registry, registry exception, and empty-string rejection.
  The MoveResponse schema round-trip and counter label shape are
  asserted.

**Findings:**

- **LOW** — `response.headers["X-RingRift-Model-Version"] =
version_label` is emitted even for random/heuristic traffic with
  value `"none"`. That is what the commit message claims, but it does
  mean gateways now see a custom header on every `/ai/move` response,
  including the ~50% of traffic that isn't NN-backed. Harmless, but
  any WAF / CDN rules that normalize custom headers should be
  spot-checked.
- **LOW** — `X-RingRift-Model-Checkpoint` is set _only_ when
  `nn_checkpoint` is truthy, so a request that uses a neural net but
  whose loaded_checkpoint_path is missing will have
  `X-RingRift-Model-Version: v4.0.0` and no checkpoint header. That is
  correct but a bit asymmetric — consider always setting
  `X-RingRift-Model-Checkpoint: unknown` for consistency in log
  pipelines that join on the pair.
- **LOW** — `test_ignores_empty_version_string` accepts two possible
  outcomes (`None` or registry default `"v0.0.0"`). With the current
  registry logic for `SimpleNamespace` models the actual result is
  `None`, and the test tolerates either. This is fine defensively but
  the test is not strict enough to catch a future regression where
  the fallback starts returning `""`. The stricter assertion would be
  `assert result in (None, "v0.0.0")`.
- **NIT** — The comment on the `AI_MOVES_BY_MODEL_VERSION` init line
  in `metrics_base.py` uses `("none", "init", "0")` — three different
  dummy label values from the other init counters. Intentional (none
  is the real "no NN" sentinel) but could briefly confuse anyone
  greping for `"init"` initialisers.

### 6ec5c8e82 — D5 fallback observability

**Overall:** Correct, well-scoped, and mostly additive. The new
`aiFallbackMovesCounter` sits alongside the existing `aiFallbackCounter`
so dashboards that key off the old metric continue to work. The
circuit-breaker state machine now emits a gauge + transition counter
that makes flapping vs stably-open distinguishable, which is the
feature's entire point.

- **Correctness (OK).** `configuredAiType = config.aiType ??
this.selectAITypeForDifficulty(config.difficulty)` is computed once
  at the top so every fallback site uses the _intended_ tier type, not
  the fallback engine. This matches the commit message's "D10 should be
  Gumbel MCTS, not local heuristic" alertability requirement.
- **Correctness (OK).** The CircuitBreaker state-machine transitions:
  - `closed → open` after `failureCount >= threshold`, only when
    `state !== 'open'` (prevents duplicate transitions if
    `recordFailure` is re-entered during the same bad window).
  - `open → half_open` after timeout elapsed, in `execute()` before
    the trial request runs.
  - `half_open → closed` via `reset()` on success.
  - `half_open → open` via `recordFailure` on trial failure — single
    failure re-opens because `failureCount` is still at threshold,
    which is the **correct** half-open behaviour and a subtle
    improvement over the previous `reset()` semantics.
- **Correctness (OK).** `emitState()` in the constructor primes the
  gauge to 0 at process start so Alertmanager's
  `max_over_time(ai_circuit_breaker_state[5m]) >= 1` rule does not
  misfire on an uninitialised series.
- **Contract (OK).** The inner `CircuitBreaker.getStatus()` return
  type was widened to `{ isOpen, failureCount, state }`. The only
  caller is `AIServiceClient.getCircuitBreakerStatus()` whose
  declared return type is unchanged (`{ isOpen, failureCount }`) —
  TypeScript's structural typing happily assigns the wider type to the
  narrower declaration, so no external caller can break. I also
  grepped the repo: no other code reads `CircuitBreaker.getStatus()`.
- **Observability / cardinality (OK).** `aiFallbackMovesCounter`
  labels: `reason` (finite set: connection_refused, timeout,
  service_unavailable, server_error, client_error, overloaded,
  circuit_open, python_error, validation_failed, no_move_from_service,
  service_degraded — 11 values), `ai_type` (8 AIType values + "unknown"),
  `difficulty` (10 tiers + "unknown" + "n/a"). Worst-case ≈ 11 × 9 × 12
  ≈ 1,200 series. Bounded; no PII.
- **Observability (OK).** `aiCircuitBreakerTransitionsCounter` has
  `from_state × to_state` labels = 3 × 3 = 9 max, well bounded.
- **Tests (good).** Seven tests exercise label shape, per-tier
  differentiation, gauge round-trip, and transition semantics. Uses
  `metric.get()` which is the correct `prom-client` API for reading
  values in tests.

**Findings:**

- **MEDIUM** — **Open `isOpen` flag never transitions to `false` during
  half-open.** In `execute()`, after `transitionTo('half_open')`,
  `this.isOpen` remains `true`. A concurrent second call that arrives
  before the trial completes will:
  1. Enter `if (this.isOpen)` → true,
  2. Re-check `now - lastFailureTime > timeout` → still true in this
     window,
  3. Call `transitionTo('half_open')` again — no-op via the
     `from === to` early return,
  4. Fall through and call `fn()`.
     So the "one trial request only" comment in the `transitionTo(...)`
     call site is not enforced: any number of concurrent trial requests
     can hit the real service during the half-open window. This is a
     **pre-existing** concurrency bug (the old `reset()` code had the
     same issue because `isOpen=false` after reset still allowed
     unlimited traffic). Not a regression, but worth a ticket: the new
     state field makes it a one-line fix (`if (this.state !== 'closed')
return` inside the `execute()` guard, or a `Semaphore(1)` for the
     half-open trial). I would not block the commit on this, but I would
     not let the comment mislead future readers.
- **MEDIUM** — The `'service_degraded'` fallback site in
  `getLocalFallbackMove` labels both `ai_type` and `difficulty` as
  `'unknown'`. The `reason` is useful but any alert that tries to
  partition "D10 fallbacks" by difficulty will silently undercount
  this path. The comment acknowledges this but it is a genuine
  observability gap: any call through `getLocalFallbackMove()` that
  could feasibly know its tier should thread `configuredAiTypeLabel`
  - `difficultyLabel` in. Worth flagging for a follow-up rather than
    fixing now, since `getLocalFallbackMove` sits on a different call
    path that does not currently receive config. The alert rule
    `AIFallbackMovesElevated` uses `sum(...)` over all labels so it
    _will_ still fire on these events; per-tier drill-down is what
    degrades.
- **LOW** — `AIFallbackMovesElevated` uses
  `sum(rate(ai_move_latency_ms_count[5m]))` as the denominator. That
  is the _latency histogram count_ on the TS side, which is
  incremented in the TS AIEngine's happy + fallback paths (see
  `aiMoveLatencyHistogram.labels(...).observe(...)`). This is the
  right denominator _only if_ every fallback move also observes the
  histogram. A quick grep shows the heuristic and random fallback
  branches do observe latency, so the arithmetic checks out, but it
  is fragile: if a future fallback path forgets to observe latency,
  the alert denominator shrinks and the rate inflates. An explicit
  "total moves served" counter would be more robust.
- **LOW** — `AICircuitBreakerOpen` uses
  `max_over_time(ai_circuit_breaker_state[5m]) >= 1`. Because
  half-open is `0.5`, a breaker that flaps closed → open → half-open
  every minute will still satisfy `>= 1`. Combined with `for: 5m`
  this is probably fine (we _want_ to alert on a process that has
  even briefly been open in the window), but it does coalesce
  "flapping" and "sustained open" into one critical alert; the
  separate `AICircuitBreakerFlapping` alert is meant to distinguish
  them but is `warning`, not `critical`. Reasonable trade-off, just
  noting.
- **NIT** — The `recordFailure` / state interleaving:
  ```typescript
  if (this.failureCount >= this.threshold) {
    if (this.state !== 'open') {
      this.transitionTo('open');
      logger.warn(...)
    }
    this.isOpen = true;
  }
  ```
  The `this.isOpen = true` after the nested transition block is now
  redundant on the first crossing (transitionTo updates the state
  already) but is still needed because `transitionTo` does not touch
  `isOpen`. The dual state variables (`isOpen` boolean + `state`
  enum) are now redundant; consider collapsing `isOpen` into a
  computed getter `this.state === 'open'` in a follow-up.

---

## Cross-cutting concerns

1. **Dual state in CircuitBreaker.** Both commits D5 and the future
   tasks C1/D2 will touch the circuit breaker. Collapsing `isOpen` into
   `state === 'open'` before further changes would reduce the surface
   area for concurrency bugs.
2. **"Diagnostic first, auto-act later" pattern is consistent.** A1
   (seat fairness) and A2 (plateau) both log only; neither changes
   thresholds. This is the right cadence given the 2026-04-10 training
   snapshot in `CLAUDE.local.md` and matches the plan's Week 1 scope.
   Confirm in Week 2 that the diagnostic signals were valuable before
   shipping auto-response code (the working tree already has a tentative
   auto-relax code path in `minimal_alphazero_loop.py` — keep it gated
   on the `--auto-plateau-relax` flag and observe the first few
   triggers in production before flipping defaults).
3. **Counter cardinality is healthy across D1 + D5.** Combined, the
   new counters add ~2,400 series worst case; existing AI metrics are
   already larger. No PII, no unbounded user ids, no free-form strings
   leak into labels.
4. **TS ↔ Python parity and canonical rules.** None of the four
   commits touch rules semantics, phases, FE, or replay contracts.
   They are purely observability + eval diagnostic changes. Nothing in
   `src/shared/engine`, `ai-service/app/game_engine/`, or
   `TRAINING_DATA_REGISTRY.md`.
5. **minimal_alphazero_loop.py sensitivity.** Both A1 and A2 edits are
   additive, gated, and at well-defined late-iteration hook points.
   They do not change training, self-play, promote logic, or
   `staged_evaluate` thresholds. No tracker-state mutation that the
   loop downstream reads. CLAUDE.local.md spirit respected.

## Verification you were not asked to do

Recommended follow-ups to validate the Week-1 landing:

1. Run `pytest ai-service/tests/unit/scripts/test_model_quality_gate.py
ai-service/tests/unit/scripts/test_plateau_detector.py
ai-service/tests/unit/test_main_model_version_telemetry.py` and
   `npm test tests/unit/aiFallbackTelemetry.test.ts
tests/unit/AIEngine.fallback.test.ts` to confirm green on this
   exact SHA (I did not execute tests per instructions).
2. On the first square8_3p iteration after the A1 change lands on
   gh200-12, scrape `quality_gate` details out of the loop log and
   confirm the `seat_wr` block is populated and the
   `SEAT_WR_IMBALANCE` warning line appears. That is the whole point
   of A1 and deserves a one-line "it worked" PR comment on issue #78.
3. On the next hex8_2p restart on gh200-8/gh200-11, confirm
   `PLATEAU_DETECTED` fires in the logs around iter 50
   (≥ 15 rejections past the last promotion at iter 33 / 36). If it
   does _not_ fire within 20 iterations of continued rejection, the
   window or staleness defaults may be too strict.
4. Curl `/ai/move` on staging and confirm both
   `X-RingRift-Model-Version` and (when applicable)
   `X-RingRift-Model-Checkpoint` are set, and that
   `ai_moves_by_model_version_total` is emitted from `/metrics`.
5. Validate `monitoring/prometheus/alerts.yml` with
   `promtool check rules` before loading (the new ai-service group
   uses `$value | humanizePercentage` which is valid but some older
   promtool versions reject empty expressions inside `(> 0)` guards).
6. Double-check the `AICircuitBreakerOpen` expression `>=1` against
   half-open (`0.5`): with `for: 5m` and a cleanly half-open-then-open
   cycle every 60s, max_over_time saturates on the open samples and
   the alert fires correctly; with a stable half-open (no trials
   arriving) it does not — verify against a synthetic `scripts/` test
   that drives the breaker.

## Reviewer

Droid (Claude Opus 4.7) via factory.ai
