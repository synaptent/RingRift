# Game Performance Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for investigating high game move latency and sluggish in-game behaviour when `HighGameMoveLatency` fires.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rules in `monitoring/prometheus/alerts.yml` (alert `HighGameMoveLatency` on `ringrift_game_move_latency_seconds_bucket`) and scrape configuration in `monitoring/prometheus/prometheus.yml`.
> - **Metrics & instrumentation:** `MetricsService` and move latency recording in `src/server/services/MetricsService.ts` (histogram `ringrift_game_move_latency_seconds` with `board_type` and `phase` labels) and HTTP metrics driven by `metricsMiddleware` (`src/server/middleware/metricsMiddleware.ts`).
> - **Game orchestration:** Shared TS rules engine and orchestrator in `src/shared/engine/**` (turn orchestrator, aggregates, movement/capture/territory/victory logic), plus `GameEngine` / `RuleEngine` on the server (`src/server/game/GameEngine.ts`, `src/server/game/RuleEngine.ts`, `src/server/game/turn/TurnEngine.ts`).
> - **Session & real-time layer:** `GameSession` / `GameSessionManager` (`src/server/game/GameSession.ts`, `src/server/game/GameSessionManager.ts`), WebSocket server (`src/server/websocket/server.ts`), and associated state machines in `src/shared/stateMachines/**`.
> - **AI integrations (when AI is contributing to move latency):** `AIEngine`, `AIPlayer`, `AIInteractionHandler`, and `AIServiceClient` (`src/server/game/ai/AIEngine.ts`, `src/server/game/ai/AIPlayer.ts`, `src/server/game/ai/AIInteractionHandler.ts`, `src/server/services/AIServiceClient.ts`), and the FastAPI AI service in `ai-service/app/main.py`.
>
> **Precedence:** Alert definitions, metrics, and rules/game server code are authoritative for what is measured and how moves are processed. This runbook describes **how to investigate and remediate**; if it conflicts with code/config/tests, **code + configs + tests win** and this document should be updated.
>
> For a high-level “rules vs AI vs infra” classification, see `AI_ARCHITECTURE.md` §0 (AI Incident Overview).

### Orchestrator posture & key metrics

- The **shared TypeScript rules engine + orchestrator** is the single source of truth for game semantics; backend, sandbox, and Python AI-service are adapters over this SSoT.
- Runtime rules selection is controlled by:
  - `ORCHESTRATOR_ADAPTER_ENABLED`
  - `ORCHESTRATOR_ROLLOUT_PERCENTAGE`
  - `ORCHESTRATOR_SHADOW_MODE_ENABLED`
  - `RINGRIFT_RULES_MODE`
- For game‑performance issues (slow moves, spikes in move latency), keep orchestrator‑ON by default and treat these flags as **rules‑engine levers**, not first-line mitigations. Adjust them only when shared‑engine/.shared/contract tests indicate a true rules defect and follow `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` for any rollback.
- Key metrics to consult alongside `HighGameMoveLatency`:
  - Game and move latency:
    - `ringrift_game_move_latency_seconds_bucket` / derived `game_move_latency_ms` (move latency)
    - HTTP latency histograms used by `HIGH_LATENCY.md`
  - Orchestrator health:
    - `ringrift_orchestrator_error_rate`
    - `ringrift_orchestrator_shadow_mismatch_rate`
    - `ringrift_orchestrator_circuit_breaker_state`
    - `ringrift_orchestrator_rollout_percentage`
  - AI contribution (when AI is on the hot path):
    - `ringrift_ai_request_duration_seconds_bucket`
    - `ringrift_ai_requests_total` / `ringrift_ai_fallback_total` (see `AI_PERFORMANCE.md`, `AI_ERRORS.md`, `AI_FALLBACK.md`, `AI_SERVICE_DOWN.md`)

---

## 1. When This Alert Fires

**Alert (from `monitoring/prometheus/alerts.yml`, `latency` group):**

- `HighGameMoveLatency` (severity: `warning`)

**Conceptual behaviour (see `alerts.yml` for the canonical expression and thresholds):**

```promql
histogram_quantile(0.99,
  sum(rate(ringrift_game_move_latency_seconds_bucket[5m])) by (le, board_type)
) > <threshold_seconds>
```

Where:

- The underlying metric `ringrift_game_move_latency_seconds_bucket` is emitted by `MetricsService.gameMoveLatency` and captures **server‑side processing time for game moves**, labelled by `board_type` and `phase`.
- The alert fires when the **P99** move latency for at least one `board_type` exceeds a configured threshold over a sustained window.

**Intended semantics:**

- Detect when moves (placements, movements, captures, territory/victory processing) are taking too long on the backend.
- This is more granular than the generic HTTP latency alerts and focused on the **game pipeline** rather than arbitrary endpoints.

**Impact:**

- Players on the affected board types see **sluggish game responsiveness**: long delays after making a move before the board updates, especially in complex phases (chains, territory processing, victory resolution).
- If severe and sustained, move timeouts may surface as errors in WebSocket clients or HTTP endpoints, and may correlate with `HighP99Latency` / `HighMedianLatency` or even `HighErrorRate`.

**Related signals and docs:**

- HTTP latency runbook: `HIGH_LATENCY.md`.
- Game health/duration runbook: `GAME_HEALTH.md` (long‑running games).
- AI performance/fallback/error runbooks: `AI_PERFORMANCE.md`, `AI_FALLBACK.md`, `AI_ERRORS.md`.
- Resource runbooks: `HIGH_MEMORY.md`, `EVENT_LOOP_LAG.md`, `RESOURCE_LEAK.md`.

---

## 2. Quick Triage (First 5–10 Minutes)

> Goal: Confirm that move latency is genuinely elevated, identify which **board types** and (where available) **phases** are slow, and decide whether this is primarily a **rules/CPU hotspot**, **AI dependency issue**, or a **systemic latency/resource problem**.

### 2.1 Confirm which board types are affected

In Alertmanager / your monitoring UI:

1. Confirm `HighGameMoveLatency` is firing and capture:
   - Environment (staging vs production).
   - The affected `board_type` reported in annotations.
   - Start time and duration.

In Prometheus, inspect P99 move latency:

```promql
# P99 game move latency by board type
histogram_quantile(0.99,
  sum(rate(ringrift_game_move_latency_seconds_bucket[5m])) by (le, board_type)
)
```

Check:

- Which `board_type` values have elevated P99.
- Whether the elevation is widespread or isolated to a small set of boards/modes.

If you have dashboards that expose the `phase` label, also inspect:

```promql
# P99 move latency by board type and phase (placement/movement/capture/territory/victory)
histogram_quantile(0.99,
  sum(rate(ringrift_game_move_latency_seconds_bucket[5m])) by (le, board_type, phase)
)
```

Identify whether the slowness is concentrated in particular phases (e.g. massive capture chains, territory processing, victory evaluation).

### 2.2 Correlate with HTTP latency, AI, and resource alerts

Move latency rarely exists in isolation. In Alertmanager / dashboard, check for concurrent:

- HTTP latency alerts: `HighP99Latency`, `HighP99LatencyCritical`, `HighP95Latency`, `HighMedianLatency` → see `HIGH_LATENCY.md`.
- AI alerts: `AIRequestHighLatency`, `AIServiceDown`, `AIFallbackRateHigh`, `AIErrorsIncreasing` → see AI runbooks.
- Resource alerts: `HighMemoryUsage*`, `HighEventLoopLag*`, `HighActiveHandles` → see resource runbooks.
- Dependency performance: `DatabaseResponseTimeSlow`, `RedisResponseTimeSlow` → see DB/Redis performance runbooks.

Interpretation:

- If global HTTP latency is high and move latency is only one of many symptoms, treat `HighGameMoveLatency` as **part of a larger performance/availability incident**.
- If HTTP latency is fine but move latency is high only for certain board types/phases, suspect **rules‑ or AI‑specific hotspots**.

### 2.3 Quick health check from the app

From an operator shell, against `APP_BASE`:

```bash
# Liveness
curl -sS APP_BASE/health | jq . || curl -sS APP_BASE/health

# Readiness (dependency breakdown)
curl -sS APP_BASE/ready | jq . || curl -sS APP_BASE/ready
```

Check:

- That overall readiness is healthy.
- Whether any dependencies (DB, Redis, AI service) are marked degraded/unhealthy.

If AI or DB/Redis appear unhealthy, pivot to those runbooks first – slow dependencies often dominate move latency.

### 2.4 Inspect logs around slow moves

Use application logs to find examples of slow moves and any associated warnings/errors. Under docker‑compose from the repo root:

```bash
cd /path/to/ringrift

docker compose logs app --tail=500 2>&1 \
  | grep -Ei 'move latency|ringrift_game_move|AI Service|AI move|rules_parity_mismatch|GameSession' \
  | tail -n 100
```

Look for:

- Explicit logging of slow move processing or timeouts.
- AI‑related delays (`AI Service` errors, `aiErrorType` fields, fallback/circuit‑breaker messages).
- Evidence of repeated retries or long decision phases (e.g. extended capture/territory/victory sequences).

---

## 3. Deep Diagnosis

> Goal: Narrow down **where** the time is being spent in the move pipeline and which layer(s) need changes.

### 3.1 Break down move latency by phase and context

Use the `phase` label and any dashboard you have for move phases to see:

- Which phases (e.g. `placement`, `movement`, `capture`, `territory`, `victory`) are slow.
- Whether slowness is correlated with particular game situations (large board configurations, many stacks, complex territories).

If your observability stack exposes additional labels (game type, AI vs PvP), compare:

- AI games vs PvP games on the same `board_type`.
- Specific modes or match types vs general traffic.

### 3.2 Determine whether AI is on the critical path

For AI games, check:

- `ringrift_ai_request_duration_seconds_bucket` (via `AIRequestHighLatency` runbook) to see if AI requests are slow.
- `ringrift_ai_requests_total` outcomes and `ringrift_ai_fallback_total` (via `AI_FALLBACK.md`, `AI_ERRORS.md`).

If AI latency is elevated and aligns with move latency spikes:

- Treat AI performance as a primary suspect and follow `AI_PERFORMANCE.md` and `AI_SERVICE_DOWN.md` in parallel.
- Confirm that local heuristic fallbacks are being used appropriately when AI is slow or unavailable.

### 3.3 Look for rules/engine CPU hotspots

If AI and external dependencies are healthy but move latency is high:

- Review recent changes to:
  - Shared engine core and helpers (`src/shared/engine/core.ts`, `movementLogic.ts`, `captureLogic.ts`, `territoryProcessing.ts`, `victoryLogic.ts`, aggregates under `aggregates/**`).
  - Server‑side GameEngine and RuleEngine (`src/server/game/GameEngine.ts`, `src/server/game/RuleEngine.ts`, `src/server/game/turn/TurnEngine.ts`).
- Consider the complexity of known heavy scenarios:
  - Large capture chains and complex territories.
  - End‑game LPS scenarios where many candidate moves are evaluated.
  - Boards or modes that exercise dense graphs of positions.

If you have profiling tools enabled (e.g. node inspector, CPU profiles in staging), capture a short profile on a slow move path and confirm whether time is dominated by:

- Rules enumeration and validation.
- Data structure manipulation (e.g. copies of large game states).
- Logging or serialization.

### 3.4 Cross‑check with game health and duration patterns

Use `GAME_HEALTH.md` guidance to see whether long move latency is translating into:

- Very long game durations (`LongRunningGames`).
- Evidence of stalled/zombie games.
- Stalemate conditions or rules edge cases.

If move latency spikes are clustered around certain end‑game states, you may need to:

- Improve stalemate/draw detection.
- Add short‑circuit logic for obviously resolved or dead positions.

---

## 4. Remediation

> Goal: Reduce move latency back under thresholds while preserving rules correctness and respecting the rules/monitoring SSoT.

### 4.1 If AI / external dependencies are the bottleneck

When AI or other services are clearly slow:

1. **Follow AI and dependency runbooks:**
   - `AI_PERFORMANCE.md`, `AI_SERVICE_DOWN.md`, `AI_FALLBACK.md`, `AI_ERRORS.md` for AI.
   - Database / Redis performance runbooks if those alerts are firing.
2. **Ensure clean fallback behaviour:**
   - Confirm `AIEngine` and `AIInteractionHandler` route through local heuristic fallbacks when the AI service is degraded, so move processing does not simply stall.
   - Verify that these fallbacks still respect the rules SSoT (shared TS engine + orchestrator) and do not change rules semantics.

### 4.2 If rules / engine code is CPU‑bound

When profiling and logs point to the rules engine or game orchestration as the bottleneck:

1. **Targeted optimisation, not semantic shortcuts:**
   - Optimise data structures or algorithms in hot paths (e.g. reuse computed structures, reduce allocations, prune obviously illegal moves early) while keeping behaviour identical.
   - Avoid “optimisations” that change rule semantics; any such change must go through the formal rules/spec process and parity checks.
2. **Leverage existing tests and parity harnesses:**
   - Use shared engine unit tests, contract vectors (`tests/fixtures/contract-vectors/v2/**`), and parity suites (TS and Python) to ensure optimisations don’t introduce divergence.
   - Where optimisation targets a previously slow corner case, add a regression test so future changes don’t re‑introduce the issue.

### 4.3 If lifecycle or WebSocket behaviour is contributing

- Confirm that `GameSession` and WebSocket flows are not performing unnecessary work on every move (e.g. redundant full‑state recomputations, heavy logging, or large payload serialization).
- Review `GameSession` and `WebSocketInteractionHandler` logic to ensure incremental updates are used where possible and that per‑move state transitions are efficient.

### 4.4 Capacity and configuration considerations

If metrics show that latency spikes correlate strongly with **load** rather than a specific bug:

- Evaluate whether the current deployment matches the expected load profile:
  - Instance counts, CPU/memory, and connection pool sizes per `docs/DEPLOYMENT_REQUIREMENTS.md`.
  - Any rate limiting or degradation controls configured via `src/server/middleware/rateLimiter.ts` and `DegradationHeaders`.
- Coordinate with infra/product teams to:
  - Scale out the backend (and AI service, if applicable) for sustained higher usage.
  - Tune rate‑limit or backoff policies (via normal config and change management, not by editing this doc).

---

## 5. Validation

Before considering the `HighGameMoveLatency` incident resolved:

### 5.1 Metrics and alerts

- [ ] The `HighGameMoveLatency` alert has cleared and remained clear for at least one full evaluation window.
- [ ] P99 (and ideally P95/P50) move latencies for affected `board_type` values have returned to **expected baselines** for that environment.
- [ ] If related HTTP, AI, or resource alerts were firing, they are also resolved.

### 5.2 Behavioural checks

- [ ] Manual playthroughs (for the affected board types and modes) show **responsive move processing** without noticeable stalls.
- [ ] WebSocket and HTTP clients do not exhibit repeated timeouts or “move taking forever” UX.
- [ ] There is no new wave of `LongRunningGames` or zombie sessions linked to the same board types.

### 5.3 Tests and documentation

- [ ] Any rules/engine optimisations added to address the issue are covered by appropriate unit, integration, and parity tests on both TS and Python sides where relevant.
- [ ] If semantics or stalemate/draw behaviour was changed, the rules docs (`RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`, `ringrift_compact_rules.md`) and parity docs (`docs/PYTHON_PARITY_REQUIREMENTS.md`, `docs/PARITY_SEED_TRIAGE.md`) have been updated accordingly.

---

## 6. TODO / Environment-Specific Notes

Populate these per environment (staging, production, etc.) and keep them updated:

- [ ] Links to dashboards that correlate `ringrift_game_move_latency_seconds` with HTTP latency, AI metrics, resource usage, and error rates.
- [ ] Typical baseline P50/P95/P99 move latencies per `board_type` (and per `phase` where available).
- [ ] Known “expensive” board types or modes and what is considered acceptable latency for them.
- [ ] Any standard profiling tools or workflows used to capture and analyse slow moves in staging/production.
