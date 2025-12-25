# Game Performance Runbook

> **Doc Status (2025-12-01): Active Runbook**  
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
  - `ORCHESTRATOR_ADAPTER_ENABLED` (hardcoded to `true`)
  - `RINGRIFT_RULES_MODE` (`ts` default, `python` diagnostic/authoritative)
  - Legacy rollout/shadow flags were removed; adapter is always 100%.
- As of PASS20/Phase 4, production/staging environments run with the orchestrator adapter **fully enabled**. For game‑performance issues (slow moves, spikes in move latency), keep `RINGRIFT_RULES_MODE=ts` by default and use circuit breaker / infra levers first; only switch to `python` mode for explicit parity diagnostics per `docs/ORCHESTRATOR_ROLLOUT_PLAN.md`.
- Key metrics to consult alongside `HighGameMoveLatency`:
  - Game and move latency:
    - `ringrift_game_move_latency_seconds_bucket` / derived `game_move_latency_ms` (move latency)
    - HTTP latency histograms used by `HIGH_LATENCY.md`
  - Orchestrator health:
    - `ringrift_orchestrator_error_rate`
    - `ringrift_orchestrator_shadow_mismatch_rate`
    - `ringrift_orchestrator_circuit_breaker_state`
    - `ringrift_orchestrator_rollout_percentage` (telemetry-only; remains 100% with the flag removed)
  - Rules correctness and engine health (PASS22 metrics):
    - `ringrift_parity_checks_total` (TS ↔ Python parity checks; success vs failure)
    - `ringrift_contract_tests_passing` / `ringrift_contract_tests_total` (contract vector coverage)
    - `ringrift_rules_errors_total` (rules validation/mutation/internal errors by type)
    - `ringrift_line_detection_duration_ms` (line/territory detection performance)
    - `ringrift_capture_chain_depth` (capture chain depth distribution)
  - AI contribution (when AI is on the hot path):
    - `ringrift_ai_request_duration_seconds_bucket`
    - `ringrift_ai_requests_total` / `ringrift_ai_fallback_total` (see `AI_PERFORMANCE.md`, `AI_ERRORS.md`, `AI_FALLBACK.md`, `AI_SERVICE_DOWN.md`)
  - System health and cache behaviour:
    - `ringrift_cache_hits_total` / `ringrift_cache_misses_total` (Redis cache effectiveness)

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
   - Verify that these fallbacks still respect the canonical rules SSoT (canonical rules spec plus shared TS engine/orchestrator) and do not change rules semantics.

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
- [ ] If semantics or stalemate/draw behaviour was changed, the rules docs (`RULES_CANONICAL_SPEC.md`, `../rules/COMPLETE_RULES.md`, `../rules/COMPACT_RULES.md`) and parity docs (`docs/PYTHON_PARITY_REQUIREMENTS.md`, `docs/PARITY_SEED_TRIAGE.md`) have been updated accordingly.

---

## 6. TODO / Environment-Specific Notes

Populate these per environment (staging, production, etc.) and keep them updated:

- [ ] Links to dashboards that correlate `ringrift_game_move_latency_seconds` with HTTP latency, AI metrics, resource usage, and error rates.
- [ ] Typical baseline P50/P95/P99 move latencies per `board_type` (and per `phase` where available).
- [ ] Known “expensive” board types or modes and what is considered acceptable latency for them.
- [ ] Any standard profiling tools or workflows used to capture and analyse slow moves in staging/production.

## 7. Baseline Metrics – Game Creation (PASS22)

> **Scope:** Baseline load test for POST `/api/games` in a Docker-based stack during PASS22, used as a production-readiness reference point for game creation throughput and latency.

### 7.1 Scenario and Load Pattern

- **Scenario script:** [`game-creation.js`](../../tests/load/scenarios/game-creation.js:25)
- **Command:** `BASE_URL=http://localhost:3000 npm run test:smoke:game-creation`
- **Load pattern (k6 stages):**
  - 30s ramp-up to 10 VUs
  - 1m ramp-up to 50 VUs
  - 2m hold at 50 VUs
  - 30s ramp-down (total ~4 minutes)
- **Observed load:**
  - Max VUs: 50
  - Iterations: 2,916
  - HTTP requests: 5,834 (~12 iterations/s, ~24 HTTP req/s)

### 7.2 Baseline Metrics Summary

| Metric                       | Value                               |
| ---------------------------- | ----------------------------------- |
| VUs (peak)                   | 50                                  |
| Game creations per second    | ≈ 12 /s                             |
| HTTP requests per second     | ≈ 24 /s                             |
| Game creation p50 latency    | ≈ 8 ms (`game_creation_latency_ms`) |
| Game creation p95 latency    | ≈ 13 ms                             |
| Game creation p99 latency    | ≈ 16 ms                             |
| Global http_req_duration p95 | ≈ 11 ms                             |
| Global http_req_duration p99 | ≈ 15 ms                             |
| Game creation error rate     | 0% (`game_creation_success_rate`)   |
| Global http_req_failed       | ≈ 50% (see note below)              |

**Notes:**

- Game creation latency is measured via the custom metric `game_creation_latency_ms` derived from k6 output for POST `/api/games`.
- The aggregate HTTP latency metrics (`http_req_duration`, `http_req_failed`) include both POST `/api/games` and subsequent GET `/api/games/:gameId` calls generated by the scenario.

### 7.3 SLO Comparison (POST `/api/games`)

Staging SLO for POST `/api/games` (from [`tests/load/README.md`](../../tests/load/README.md:124) and [`STRATEGIC_ROADMAP.md`](../planning/STRATEGIC_ROADMAP.md:1)):

- p95 latency ≤ 800 ms
- p99 latency ≤ 1500 ms
- Error rate < 1%

Baseline results from the PASS22 `game-creation` scenario:

- `game_creation_latency_ms`:
  - p95 ≈ 13 ms
  - p99 ≈ 16 ms
- `game_creation_success_rate`:
  - 2,916 / 2,916 successful creations
  - Error rate ≈ 0%

Interpretation:

- At ~12 game creations/s and ~24 HTTP req/s with 50 VUs, POST `/api/games` latency is **~60–100x faster than the SLO thresholds** and the error budget is untouched.
- For this load profile, the game-creation endpoint **strongly meets** the staging SLOs with a large safety margin.

These baselines should be treated as the initial reference point for future capacity planning and regression detection for game creation.

### 7.4 Known Issue – GET `/api/games/:gameId` returns 400 GAME_INVALID_ID

During the PASS22 `game-creation` load test, every GET `/api/games/:gameId` request returned `400 GAME_INVALID_ID` due to an ID-format/validation mismatch between:

- The ID returned by `POST /api/games`, and
- The `GameIdParamSchema` used by [`router.get('/:gameId')`](../../src/server/routes/game.ts:438).

Impact on metrics:

- `http_req_failed` for the scenario is ≈ 49.98%, entirely due to these GET `/api/games/:gameId` 400 responses.
- Creation-specific metrics remain healthy:
  - `game_creation_success_rate = 100%` (2,916 / 2,916 successful game creations)
  - Creation latency metrics (p95/p99) are well within SLOs.

Classification:

- This is a **contract/validation bug** in the GET `/api/games/:gameId` endpoint, not a capacity or latency problem in the backend stack.
- It is tracked separately and should **not** be interpreted as an infrastructure regression or stability issue when reviewing k6 summaries for this scenario.

Operators and engineers should treat the elevated `http_req_failed` in this specific scenario as a known artifact of the GET endpoint contract bug until that issue is resolved.

## 8. PASS24.1 – k6 baselines after HTTP/WebSocket stabilization

&gt; **Context:** PASS24.1 re-ran all four k6 scenarios against the Docker stack after fixing HTTP/WebSocket availability issues by routing load through nginx on port 80 instead of hitting the Node app container directly on port 3000. All runs used `BASE_URL=http://127.0.0.1` (nginx → `app:3000`) and, where applicable, `WS_URL=ws://127.0.0.1`.

### 8.1 Scenario 1 – Game Creation

**Environment**

- `BASE_URL=http://127.0.0.1` (nginx listening on port 80 and proxying to the Node `app` service on port 3000).
- k6 scenario script: [`game-creation.js`](../../tests/load/scenarios/game-creation.js:1).
- Auth and health probes as in §7 (shared login helper and `/health`).

**Infra / availability status**

- `/health` and `POST /api/auth/login` were consistently 200 throughout the run.
- No socket-level connection failures were observed (`http_req_failed` due to `status=0` / `connect: connection refused` dropped to 0).
- The app container remained reachable for the full duration of the test; no k6 VUs were forced to back off due to transport errors.

**Application behaviour and SLOs**

- `POST /api/games` continued to behave like the PASS22 baseline:
  - All game-creation failures were HTTP-level responses (validation errors or `429 Too Many Requests` under higher RPS), not transport errors.
  - Creation latency remained far below the staging SLOs (p95 ≤ 800 ms, p99 ≤ 1500 ms).
- The dedicated PASS22 baseline run (see §7) remains representative after PASS24.1:
  - `game_creation_latency_ms`: p95 ≈ 13 ms, p99 ≈ 16 ms.
  - `game_creation_success_rate`: ≈ 100% (all creations succeed while GET failures are tracked separately).
- GETs issued by the scenario against `GET /api/games/:gameId` still surface the `GAME_INVALID_ID` contract issue described in §7.4, so the aggregate `http_req_failed` metric for the scenario stays elevated even though infrastructure availability is now healthy.

**Summary**

- **Infra / availability:** Healthy. No `ECONNREFUSED` / status=0 failures when routing through nginx.
- **Application correctness / SLOs:** Latency SLOs for `POST /api/games` are strongly met; remaining work is to fix the `GET /api/games/:gameId` ID/validation contract so that aggregate k6 error-rate thresholds reflect true service health.

### 8.2 Scenario 2 – Concurrent Games

**Environment**

- Same as Scenario 1 (`BASE_URL=http://127.0.0.1` via nginx → `app:3000`).
- k6 scenario script: [`concurrent-games.js`](../../tests/load/scenarios/concurrent-games.js:1).

**Infra / availability status**

- Setup phase (`/health`, `/api/auth/login`) was stable; no `status=0` or connection-refused errors.
- `POST /api/games` and `GET /api/games/:gameId` remained reachable for the full test window.
- Node and nginx stayed responsive under the 100+ concurrent game load.

**Application behaviour and SLOs**

- The dominant failures were HTTP 4xx responses rather than transport errors:
  - Many `GET /api/games/:gameId` calls returned `400 GAME_INVALID_ID` because the scenario still assumes an older ID shape / URL contract.
  - A minority of rate-limit responses (`429`) appear when ramping quickly to 100+ concurrent games.
- From an infra perspective the scenario demonstrates that the backend can maintain 100+ active games without dropping connections; however the high 4xx rate prevents SLO thresholds for error rate from passing until the contract between k6 and the game routes is corrected.

**Summary**

- **Infra / availability:** Healthy. No socket-level errors; k6 can sustain the target concurrency against nginx.
- **Application correctness / SLOs:** Failing due to contract/ID issues in `GET /api/games/:gameId` and some rate limiting, not due to capacity or availability problems.

### 8.3 Scenario 3 – Player Moves

**Environment**

- Same nginx-fronted HTTP topology as Scenarios 1–2.
- k6 scenario script: [`player-moves.js`](../../tests/load/scenarios/player-moves.js:1).
- HTTP move endpoint flag: `MOVE_HTTP_ENDPOINT_ENABLED=false` in the script; the scenario currently exercises game creation and polling only.

**Infra / availability status**

- `/health` and `/api/auth/login` remained stable (200) throughout the run.
- Repeated `GET /api/games/:gameId` polling did not produce any socket-level failures; no `ECONNREFUSED` or `status=0` errors were observed.
- Overall HTTP availability under this pattern is acceptable after PASS24.1.

**Application behaviour and SLOs**

- Because `MOVE_HTTP_ENDPOINT_ENABLED=false`, this scenario primarily stresses:
  - Creation of small AI-backed games via `POST /api/games`.
  - Polling reads via `GET /api/games/:gameId` while the game progresses via AI and WebSockets.
- Under the current implementation:
  - `GET /api/games/:gameId` shows a high 4xx rate driven by validation/contract assumptions (IDs, lifecycle timing), not transport issues.
  - Latency for successful GETs is low and well within the SLOs; the failures are functional (bad IDs / timing) rather than performance-related.
- This scenario will become more representative once an HTTP move endpoint is introduced and the script is updated to submit legal moves rather than only poll state.

**Summary**

- **Infra / availability:** Healthy. No socket or connection-refused errors when polling game state under load.
- **Application correctness / SLOs:** Thresholds currently fail because a large fraction of GETs return functional 4xx responses; fixing the `GET /api/games/:gameId` contract and aligning the scenario with the canonical move API are required before using this scenario as an SLO gate.

### 8.4 Scenario 4 – WebSocket Stress

**Environment**

- `BASE_URL=http://127.0.0.1`.
- `WS_URL=ws://127.0.0.1`.
- k6 scenario script: [`websocket-stress.js`](../../tests/load/scenarios/websocket-stress.js:1).
- WebSocket connections target the same Socket.IO endpoint as the browser client: `/socket.io/?EIO=4&amp;transport=websocket&amp;token=…`.

**Infra / availability status**

- After routing through nginx, k6 is able to establish WebSocket connections reliably:
  - Handshakes to `/socket.io/` complete successfully (HTTP 101 switching protocols) across all VUs.
  - No `ECONNREFUSED` or low-level TCP failures were observed.
- This confirms that nginx + WebSocketServer can absorb the connection load without dropping handshakes.

**Application behaviour and SLOs**

- Connections are short-lived:
  - The server closes many sockets shortly after handshake with "message parse error" / invalid Socket.IO frame semantics.
  - As a result, connection-duration and message-latency thresholds in the k6 script are not met.
- The root cause is **protocol mismatch**:
  - The k6 script speaks plain WebSocket frames with simple JSON envelopes.
  - The production client uses the full Socket.IO protocol and a richer message envelope.
  - WebSocketServer expects well-formed Socket.IO messages and terminates connections that do not conform.
- From an infra perspective, this is acceptable—the handshakes and nginx proxying behave correctly—but the scenario cannot yet be used as a strict SLO gate for steady-state WebSocket behaviour until its message format is aligned with the real client protocol.

**Summary**

- **Infra / availability:** Healthy. 100% of attempted WebSocket handshakes succeed via nginx and `/socket.io/`.
- **Application correctness / SLOs:** Failing due to WebSocket protocol/message-format mismatch between the k6 script and the production client; connection duration and message-latency thresholds currently reflect this mismatch, not infrastructure instability.

&gt; For up-to-date guidance on how these PASS24.1 scenario baselines relate to P21.4‑2 (production-scale load test) and P22.10 ("Document baseline metrics from load test"), see:
&gt;
&gt; - [`docs/PASS21_ASSESSMENT_REPORT.md`](../archive/assessments/PASS21_ASSESSMENT_REPORT.md:393) – updated Follow-up Status (PASS24.1) note under Production Validation.
&gt; - [`docs/PASS22_COMPLETION_SUMMARY.md`](../archive/assessments/PASS22_COMPLETION_SUMMARY.md:205) – Load Test Baselines section with PASS24.1 follow-up summary.
&gt; - [`docs/PASS22_ASSESSMENT_REPORT.md`](../archive/assessments/PASS22_ASSESSMENT_REPORT.md:393) – Production Validation section with infra vs functional k6 scenario status.
