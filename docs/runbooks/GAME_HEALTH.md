# Game Health Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for investigating unusual game duration patterns and stuck/"zombie" games when the `LongRunningGames` alert fires.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rules in `monitoring/prometheus/alerts.yml` (alert `LongRunningGames` on `ringrift_game_duration_seconds_bucket`), and scrape configuration in `monitoring/prometheus/prometheus.yml`.
> - **Game session lifecycle:** `GameSession` and `GameSessionManager` (`src/server/game/GameSession.ts`, `src/server/game/GameSessionManager.ts`), WebSocket server (`src/server/websocket/server.ts`), and game session state machines under `src/shared/stateMachines/gameSession.ts` and `src/shared/stateMachines/connection.ts`.
> - **Rules semantics:** Shared engine helpers, aggregates, and orchestrator in `src/shared/engine/**`, plus supporting rules docs (`RULES_CANONICAL_SPEC.md`, `../rules/COMPLETE_RULES.md`, `../rules/COMPACT_RULES.md`).
> - **Persistence & retention:** `GamePersistenceService` and data lifecycle logic (`src/server/services/GamePersistenceService.ts`, `src/server/services/DataRetentionService.ts`, `docs/DATA_LIFECYCLE_AND_PRIVACY.md`).
> - **Parity & invariants:** Historical incident and parity docs that relate to stalled/degenerate game states, such as `docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`, `docs/PARITY_SEED_TRIAGE.md`, and `docs/STRICT_INVARIANT_SOAKS.md`.
>
> **Precedence:** Alert definitions, metrics, rules code, and lifecycle code are authoritative. This runbook explains **how to investigate and remediate**; if it conflicts with code/config/tests or the rules specs, **code + specs + tests win** and this document should be updated.
>
> For a high-level “rules vs AI vs infra” classification, see `AI_ARCHITECTURE.md` §0 (AI Incident Overview).

### Orchestrator posture & key metrics

- The **shared TypeScript rules engine + orchestrator** is the single source of truth for game semantics; backend, sandbox, and Python AI-service are adapters over this SSoT.
- Runtime rules selection is controlled by:
  - `ORCHESTRATOR_ADAPTER_ENABLED` (hardcoded to `true`)
  - `RINGRIFT_RULES_MODE` (`ts` default, `python` diagnostic/authoritative)
  - Legacy rollout/shadow flags were removed; adapter is always 100%.
- For generic **game-health** issues (long games, stalls, abnormal completion rates), keep orchestrator‑ON by default and treat these flags as **rules‑engine levers**, not first-line mitigations. Adjust them only when shared‑engine/.shared/contract tests indicate a true rules defect and follow `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` for any rollback.
- Key metrics to consult alongside `LongRunningGames`:
  - Game latency and progress:
    - `game_move_latency_ms` (backend move latency)
    - S‑invariant / progress metrics from invariant soaks (see `docs/STRICT_INVARIANT_SOAKS.md`)
  - Orchestrator health:
    - `ringrift_orchestrator_error_rate`
    - `ringrift_orchestrator_shadow_mismatch_rate`
    - `ringrift_orchestrator_circuit_breaker_state`
  - AI contribution (when games vs AI dominate the signal):
    - `ringrift_ai_request_duration_seconds_bucket`
    - `ringrift_ai_requests_total` / `ringrift_ai_fallback_total` (see `AI_PERFORMANCE.md`, `AI_ERRORS.md`, `AI_FALLBACK.md`, `AI_SERVICE_DOWN.md`)

When triaging a `LongRunningGames` incident:

- If games are **legally progressing but semantically wrong** (e.g. illegal moves accepted, victory/territory/LPS behaviour clearly incorrect, or invariant violations surfaced), treat this as a **rules/orchestrator incident** and:
  - Switch to `docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md` for rollout/flag levers and environment phases.
  - Use `docs/runbooks/RULES_PARITY.md`, contract vectors, and `docs/STRICT_INVARIANT_SOAKS.md` to investigate parity/invariant issues.
- If games are simply **slow due to move latency or AI slowness**, keep orchestrator flags unchanged and:
  - Follow `GAME_PERFORMANCE.md` and `HIGH_LATENCY.md` for backend/HTTP latency.
  - Follow `AI_PERFORMANCE.md` / `AI_ERRORS.md` when AI latency or errors dominate the signal.

---

## 1. When This Alert Fires

**Alert (from `monitoring/prometheus/alerts.yml`, `business` group):**

- `LongRunningGames` (severity: `info`, team: `product`)

**Conceptual behaviour (see `alerts.yml` for the canonical expression and thresholds):**

```promql
histogram_quantile(0.5,
  sum(rate(ringrift_game_duration_seconds_bucket[1h])) by (le, type)
) > <threshold_seconds>
```

Where:

- The underlying metric (`ringrift_game_duration_seconds_bucket`) tracks **completed game durations** (per `type`, e.g. board or game type).
- The alert fires when the **median** game duration (P50) for at least one `type` is above a configured threshold (on the order of an hour) for a sustained period.

**Intended semantics:**

- Surface situations where many games are **taking unusually long to finish**, which may mean:
  - Games are stalling in mid-play (no progress but sessions never end).
  - Certain board types or configurations produce degenerate loops or unreachable victory conditions.
  - Cleanup/timeout logic for abandoned games is not working as expected.
- The severity is `info` because some long games are legitimate (e.g. very slow, exploratory players), but **sustained elevation** is a signal to check for pathology.

**Related signals and docs:**

- `NoActiveGames` / `NoWebSocketConnections` (see `NO_ACTIVITY.md`, `WEBSOCKET_ISSUES.md`).
- `HighGameMoveLatency` and HTTP latency alerts (see `GAME_PERFORMANCE.md`, `HIGH_LATENCY.md`).
- AI and resource alerts (`AI_*`, `HighMemoryUsage*`, `HighEventLoopLag*`), plus incident docs under `docs/incidents/AVAILABILITY.md`, `docs/incidents/LATENCY.md`, `docs/incidents/RESOURCES.md`.

---

## 2. Quick Triage (First 5–10 Minutes)

> Goal: Decide whether we are seeing **legitimate long games** (e.g. certain board types or modes) vs **stuck/zombie games** or rules/lifecycle bugs.

### 2.1 Confirm the alert, board types, and time range

In Alertmanager / your monitoring UI:

1. Confirm that `LongRunningGames` is firing and note:
   - Environment (staging vs production).
   - Start time and duration.
   - Any annotations (summary, description, impact).
2. In Prometheus, inspect the median durations by game type:

```promql
# Median game duration by type over the last hour
histogram_quantile(0.5,
  sum(rate(ringrift_game_duration_seconds_bucket[1h])) by (le, type)
)
```

Check:

- Which `type` label(s) are elevated.
- Whether the elevation is recent or has been trending up for some time.

### 2.2 Correlate with move latency, AI, and resource alerts

Latency/resource issues can indirectly prolong games by making moves extremely slow or causing repeated retries:

- Check for:
  - `HighGameMoveLatency` / HTTP latency alerts → see `GAME_PERFORMANCE.md`, `HIGH_LATENCY.md`.
  - AI alerts (`AIServiceDown`, `AIRequestHighLatency`, `AIFallbackRateHigh`, `AIErrorsIncreasing`) → see `AI_SERVICE_DOWN.md`, `AI_PERFORMANCE.md`, `AI_FALLBACK.md`, `AI_ERRORS.md`.
  - Resource alerts (`HighMemoryUsage*`, `HighEventLoopLag*`, `HighActiveHandles`) → see `HIGH_MEMORY.md`, `EVENT_LOOP_LAG.md`, `RESOURCE_LEAK.md`.

If several of these are firing concurrently, treat `LongRunningGames` as **secondary** and prioritise the underlying latency/resource issues.

### 2.3 Sanity-check health, readiness, and game activity

From an operator shell, against `APP_BASE`:

```bash
# Liveness
curl -sS APP_BASE/health | jq . || curl -sS APP_BASE/health

# Readiness (includes dependency and service-status breakdown)
curl -sS APP_BASE/ready | jq . || curl -sS APP_BASE/ready
```

Look for:

- Degraded DB/Redis/AI entries (if so, pivot to those runbooks first).
- Whether **game services** (WebSocket server, rules engine) appear healthy.

In Prometheus, optionally check:

```promql
# Current number of active games
ringrift_games_active
```

- A large number of active games with high median durations suggests **zombie or stalled sessions**.
- Normal or low active games but high median durations may indicate **a small set of extremely long games** (e.g. edge-case board types or seeds).

---

## 3. Deep Diagnosis

> Goal: Determine whether long-running games are primarily due to **product behaviour**, **rules semantics**, **AI/stall bugs**, or **lifecycle/cleanup issues**.

### 3.1 Identify specific game types and patterns

Work with product/ops to answer:

- Which `type` values are elevated? Are these specific board types, time controls, or match formats?
- Has there been a recent change to:
  - Rules semantics (see `RULES_CANONICAL_SPEC.md`, `RULES_ENGINE_ARCHITECTURE.md`).
  - AI behaviour (see `ai-service/AI_IMPROVEMENT_PLAN.md`, `docs/AI_TRAINING_AND_DATASETS.md`).
  - Game session lifecycle (e.g. new reconnection behaviour, timeouts, or draw rules)?

If elevation is confined to one new mode/configuration, suspect product/config changes first.

### 3.2 Inspect candidate stalled or zombie games

Use your usual database access or internal tooling (see `docs/OPERATIONS_DB.md`) to:

1. Query for **very old active games**, for example:
   - Games with `status` still `IN_PROGRESS` but `updatedAt` older than some threshold.
   - Games whose recorded duration or last-move timestamp is extremely large relative to typical sessions.
2. For a few representative games, inspect:
   - Game state (board, rings, territory, last move).
   - Players’ last activity (are both players disconnected?).
   - Whether game status should have been a win/loss/draw but never transitioned.

Compare findings with known incidents/edge-cases:

- Territory/forced-elimination issues – see `docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`, territory parity tests, and invariant soak docs.
- Long LPS/capture chains – see `docs/PARITY_SEED_TRIAGE.md`, `docs/STRICT_INVARIANT_SOAKS.md`, and associated tests under `tests/unit/**LPS**` and `ai-service/tests/parity/**`.

### 3.3 Check lifecycle and timeout behaviour

Review and, if needed, instrument:

- `GameSession` and `GameSessionManager` for:
  - How long an idle game is allowed to stay `IN_PROGRESS` without moves.
  - How reconnection and abandonment are handled (e.g. explicit resign/timeout vs long-lived inactive sessions).
  - Any per-environment configuration controlling idle or maximum game duration.
- Data retention / cleanup processes:
  - `DataRetentionService` and related scheduled jobs that archive or clean up old sessions.

Questions to answer:

- Are **abandoned games** being kept indefinitely as active sessions?
- Do we have **max-duration or max-move safeguards** in code for pathological but technically legal games?
- Are there known reconnection edge-cases where a game never flips to a terminal state even when no players can rejoin?

### 3.4 Rules / AI-driven stalemates

If game snapshots show states where neither player can realistically force a win but the engine never declares a draw:

- Review victory semantics in `src/shared/engine/aggregates/VictoryAggregate.ts`, `VictoryAggregate.ts`, and related documentation in `RULES_CANONICAL_SPEC.md`.
- Check whether global stalemate / tie-breaker logic (e.g. R172 last-player-standing, territory thresholds, ring caps) is behaving as designed.
- For AI games, confirm that AI is not **repeating meaningless moves** due to evaluation plateaus or insufficient draw detection (see AI and training docs under `ai-service/AI_ASSESSMENT_REPORT.md` and `ai-service/AI_IMPROVEMENT_PLAN.md`).

If you suspect a rules bug or insufficient stalemate handling:

- Capture a minimal reproducible scenario (seed, sequence) and record it in `docs/PARITY_SEED_TRIAGE.md` or a dedicated incident doc.
- Work with rules maintainers to decide whether to:
  - Tighten draw/stalemate rules.
  - Add invariant checks to prevent known degenerate loops.

---

## 4. Remediation

> Goal: Reduce the prevalence of pathological long games while preserving valid strategic depth and respecting the canonical rules SSoT.

### 4.1 Fix lifecycle and cleanup issues

If diagnosis indicates **abandoned or zombie games** are the main driver:

1. **Improve idle/abandonment handling:**
   - Introduce or refine **max idle time** for inactive sessions, after which a game is auto-terminated (e.g. as a loss for the disconnected player or a draw per product policy).
   - Ensure reconnection windows are bounded and documented.
2. **Strengthen cleanup jobs:**
   - Verify that scheduled tasks or background jobs in `DataRetentionService` and related services are:
     - Running reliably in the target environment.
     - Correctly updating game status and cleaning up stale records.
3. **Keep behaviour per SSoT:**
   - All lifecycle changes should be implemented in code and captured in tests/state-machine docs (`docs/STATE_MACHINES.md`, `docs/DATA_LIFECYCLE_AND_PRIVACY.md`), **not** patched in this runbook.

### 4.2 Address rules or AI-driven stalemates

If rules semantics are technically respected but allow **practically endless** games in some configurations:

1. **Decide on product-level constraints:**
   - Introduce maximum move counts or repetition rules.
   - Offer or enforce draws after N moves without material progress, consistent with rules decisions documented in `RULES_CANONICAL_SPEC.md` and supplementary clarifications.
2. **Implement draw / stalemate detection:**
   - Extend victory and stalemate logic (`victoryLogic.ts`, aggregates, and orchestrator) to detect agreed-upon stalemate patterns.
   - Add regression tests and, where appropriate, parity fixtures to enforce the new behaviour.
3. **Update AI behaviour (for AI games):**
   - For modes that are clearly prone to looped games, adjust AI evaluation or exploration settings (in the AI service) to favour progress and game termination.

### 4.3 Treat long games caused by systemic latency as a latency issue

If game durations are inflated primarily because **each move is slow** (rather than because many moves are played):

- Prioritise the latency and resource runbooks:
  - `GAME_PERFORMANCE.md` and `HIGH_LATENCY.md` for move/HTTP latency.
  - Resource runbooks (`HIGH_MEMORY.md`, `EVENT_LOOP_LAG.md`, `RESOURCE_LEAK.md`).
  - AI performance/fallback runbooks where AI is contributing significantly.

Once move latency and system health are back to normal, re-check `LongRunningGames` to see if game durations naturally fall back to baseline.

---

## 5. Validation

Before considering the `LongRunningGames` incident resolved:

### 5.1 Metrics and alerts

- [ ] The `LongRunningGames` alert has cleared and remained clear for at least one full evaluation window.
- [ ] Median game duration (P50) for each `type` has returned to **expected baselines** for that environment (normal peak/off-peak variation is acceptable).
- [ ] If other related alerts (move latency, AI, resources) fired, they are also resolved.

### 5.2 Session and data checks

- [ ] Database queries or internal tools show **no large backlog** of clearly abandoned or zombie `IN_PROGRESS` games at extreme ages.
- [ ] Manual inspection of a few previously problematic game types confirms games now end via win/loss/draw or are cleaned up in a bounded time.

### 5.3 Tests and documentation

- [ ] Any lifecycle or rules changes are covered by unit/integration tests (e.g. `GameSession.*.test.ts`, relevant rules/territory/victory tests, soak tests, or invariants under `docs/STRICT_INVARIANT_SOAKS.md`).
- [ ] If stalemate/draw semantics were updated, the rules markdowns (`RULES_CANONICAL_SPEC.md`, `../rules/COMPLETE_RULES.md`) and parity docs (`docs/PYTHON_PARITY_REQUIREMENTS.md`, `docs/PARITY_SEED_TRIAGE.md`) have been refreshed accordingly.

---

## 6. TODO / Environment-Specific Notes

Populate these per environment (staging, production, etc.) and keep them updated:

- [ ] Links to dashboards showing game duration distributions (P50/P90/P99) by `type`, alongside move latency, AI metrics, and active game counts.
- [ ] Typical baseline values for median and tail game durations for key board types or modes.
- [ ] Known modes or configurations that are expected to produce long games (documented so on-call can distinguish expected vs unexpected behaviour).
- [ ] Internal tools or queries used to inspect individual game sessions and their lifecycle state (e.g. operations DB dashboards, ad-hoc SQL, or admin UIs).
