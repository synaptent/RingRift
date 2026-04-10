# AI Service Degradation Drill Runbook

> **Doc Status (2025-12-05): Active Runbook**  
> **Role:** PASS-style drill for simulating AI service degradation in a **staging/non-production** environment and validating alerts, fallbacks, and user-visible behaviour.
>
> **SSoT alignment:** This runbook is **derived operational guidance** over:
>
> - **Incidents and alerts:** `docs/incidents/AI_SERVICE.md`, `docs/incidents/LATENCY.md`, `docs/incidents/AVAILABILITY.md`, and alert rules in `monitoring/prometheus/alerts.yml` (especially `AIServiceDown`, `AIFallbackRateHigh`, `AIFallbackRateCritical`, `AIRequestHighLatency`, `AIErrorsIncreasing`, and `ServiceDegraded`).
> - **Metrics and SLO documentation:** `docs/operations/ALERTING_THRESHOLDS.md` (AI service alerts and thresholds) and the Grafana dashboards under `monitoring/grafana/dashboards/` (Game Performance & System Health dashboards for AI latency/fallbacks and overall health).
> - **AI integration & fallbacks:** `AIServiceClient` and `AIEngine` (`src/server/services/AIServiceClient.ts`, `src/server/game/ai/AIEngine.ts`), including concurrency caps, circuit breaker, and local heuristic fallback logic plus counters such as `ringrift_ai_requests_total` and `ringrift_ai_fallback_total`.
> - **Environment and deployment:** `docs/operations/ENVIRONMENT_VARIABLES.md` (AI- and load-related env vars such as `AI_SERVICE_URL`, `AI_SERVICE_REQUEST_TIMEOUT_MS`, `AI_MAX_CONCURRENT_REQUESTS`, `ENABLE_HTTP_MOVE_HARNESS`) and `docs/planning/DEPLOYMENT_REQUIREMENTS.md` (monitoring stack expectations).
> - **Load / gameplay harnesses:** `tests/load/scenarios/player-moves.js` and its thresholds as mapped in `docs/operations/ALERTING_THRESHOLDS.md` §“AI turn SLOs (service and end‑to‑end turn latency)”.
> - **Existing drills:** `docs/runbooks/SECRETS_ROTATION_DRILL.md`, `docs/runbooks/DATABASE_BACKUP_AND_RESTORE_DRILL.md`, and the summary in `docs/archive/historical/OPERATIONAL_DRILLS_RESULTS_2025_12_03.md` (Drill 7.3.3 AI outage simulation).

> **Precedence:** Alert rules, metrics, backend/AI code, and tests are authoritative for **what counts as AI degradation** and how fallbacks behave. This runbook defines a **repeatable drill** for staging; if any step disagrees with code, configs, or tests, **code + configs + tests win** and this document must be updated.

---

## 1. Goal & Scope

This drill exercises a **controlled AI service degradation** in a **staging / non‑production** environment and validates that:

- Alerts for AI and aggregate degradation behave as expected:
  - `AIServiceDown` (AI service availability)
  - `AIFallbackRateHigh` / `AIFallbackRateCritical` (fallback share)
  - Optionally `AIRequestHighLatency`, `AIErrorsIncreasing`, and `ServiceDegraded`
- The backend **falls back gracefully** to local heuristics without breaking rules semantics:
  - AI turns continue to resolve.
  - `ringrift_ai_fallback_total` increments with appropriate `reason` labels.
- User-facing flows remain acceptable:
  - Players can still create and play games vs AI.
  - Degradation appears as **weaker/slower AI**, not hard failures.
- Existing incident/runbook guidance is usable under live conditions:
  - `docs/incidents/AI_SERVICE.md`, `docs/runbooks/AI_SERVICE_DOWN.md`, `docs/runbooks/AI_FALLBACK.md`, `docs/runbooks/AI_PERFORMANCE.md`, `docs/runbooks/SERVICE_DEGRADATION.md`.

**Environments:**

- Primary target: **staging** (or an equivalent non‑prod stack) that mirrors production topology:
  - App + Postgres + Redis + AI service via `docker-compose.yml` + `docker-compose.staging.yml`, or equivalent.
  - Monitoring stack (Prometheus, Alertmanager, Grafana) deployed and scraping the app.

**Non-goals:**

- Do **not** change SLO numbers in `STRATEGIC_ROADMAP.md` or `PROJECT_GOALS.md`.
- Do **not** modify alert thresholds in `monitoring/prometheus/alerts.yml` as part of this drill.
- Do **not** introduce new ad-hoc debug flags or one-off degradation mechanisms; reuse:
  - Docker Compose controls (`docker compose stop/start/restart ai-service`).
  - Existing AI integration, fallbacks, and monitoring.

---

## 2. Preconditions & Safety

### 2.1 Safety constraints

- **Environment:** This drill must be run only in **staging or non‑production**.  
  Do **not** run it against production without a separate, explicit production drill plan and change‑management approval.
- **Data:** AI degradation affects **quality and latency**, not data integrity.  
  No destructive actions against Postgres or Redis are required.
- **Topology assumptions:**
  - App reachable at `http://localhost:3000` (or environment-specific base URL).
  - AI service reachable at `http://localhost:8001` from the host and as `http://ai-service:8001` from the app container.
  - Monitoring stack reachable (Prometheus, Grafana, Alertmanager) per `docs/planning/DEPLOYMENT_REQUIREMENTS.md`.
- **Monitoring:** Prometheus must be scraping:
  - Node `/metrics` (for `ringrift_ai_*`, `ringrift_service_status`, `ringrift_degradation_level`).
  - Optionally the AI service `/metrics` if configured.

### 2.2 Required tooling and config

- Docker and Docker Compose.
- `npm` / Node toolchain on the operator host.
- [`k6`](https://k6.io/docs/getting-started/installation/) installed on the operator host.
- Staging stack configuration consistent with:
  - `docs/operations/ENVIRONMENT_VARIABLES.md` (especially AI service and monitoring variables).
  - `docs/planning/DEPLOYMENT_REQUIREMENTS.md`.

**For the canonical drill path below:**

- The backend’s **HTTP move harness** is enabled in staging:
  - `ENABLE_HTTP_MOVE_HARNESS=true` in the staging env (`.env.staging` or equivalent).
  - This exposes the internal `POST /api/games/:gameId/moves` endpoint used by `tests/load/scenarios/player-moves.js`.
- Prometheus alerts for AI and degradation are deployed from `monitoring/prometheus/alerts.yml`.

### 2.3 Recommended pre‑drill validation commands

From the project root (on the host where staging is deployed):

```bash
# Validate docker-compose, env schema, and deployment wiring
npm run validate:deployment

# (Optional but recommended) Validate monitoring configs and docs/SSoT alignment
npm run validate:monitoring      # wrapper around ./scripts/validate-monitoring-configs.sh
npm run ssot-check               # docs/config SSoT checks, including alert/docs cross-links
```

Fix any **hard failures** here before running the drill. Mild doc warnings can be noted but should not block.

### 2.4 PASS framing (staging drill overview)

- **Purpose:**  
  Practise detection, triage, and recovery for **AI service degradation** in staging:
  - Induce a controlled AI outage on the Python service.
  - Generate realistic AI traffic via existing load harnesses.
  - Observe that alerts, dashboards, and fallbacks behave as designed.

- **Preconditions:**
  - Staging stack healthy:
    - `/health` and `/ready` green for the app.
    - `ringrift_service_status{service="ai_service"} == 1`.
  - Monitoring online:
    - Prometheus scraping Node and AI service.
    - Grafana dashboards “System Health” and “Game Performance” configured.
  - HTTP move harness enabled (`ENABLE_HTTP_MOVE_HARNESS=true`) and reachable (see §3.2).

- **Actions (high level):**
  1. Establish a baseline for AI metrics and alerts under light AI traffic.
  2. Induce AI service degradation in staging (canonical: **stop the `ai-service` container**).
  3. Drive AI traffic via `tests/load/scenarios/player-moves.js` (k6 scenario).
  4. Observe alerts (`AIServiceDown`, `AIFallbackRateHigh`/`Critical`, optionally `ServiceDegraded`) and Grafana panels.
  5. Verify user-visible behaviour via manual AI games (fallback moves, no hard failures).
  6. Restore the AI service and confirm metrics/alerts return to baseline.

- **Signals & KPIs:**
  - `ringrift_service_status{service="ai_service"}` flips from `1 → 0` during the induced outage, then back to `1`.
  - `ringrift_ai_fallback_total / ringrift_ai_requests_total` increases during the outage then returns near baseline.
  - `AIServiceDown`, `AIFallbackRateHigh` (and potentially `AIFallbackRateCritical`) fire and clear as expected.
  - `/ready` marks the AI dependency as degraded during the outage and healthy afterwards.
  - AI games remain playable with lower move quality but without systemic 5xx or timeouts once fallbacks are active.

---

## 3. Step-by-step Drill (staging with docker-compose)

All shell commands below assume you are on the staging host in the RingRift deployment directory (for example `/opt/ringrift`) and using docker‑compose based deployment. Adapt hostnames and ports as needed for your environment.

### 3.1 Baseline: confirm healthy AI service and monitoring

1. **Check app health and readiness**

   ```bash
   APP_BASE=${APP_BASE:-http://localhost:3000}

   echo "== App health =="
   curl -sS "${APP_BASE}/health" | jq . || curl -sS "${APP_BASE}/health"

   echo "== App readiness (dependency breakdown) =="
   curl -sS "${APP_BASE}/ready" | jq . || curl -sS "${APP_BASE}/ready"
   ```

   - Confirm overall status is healthy.
   - Confirm the AI dependency entry (e.g. `aiService`) is present and marked as `healthy`.

2. **Confirm AI service container is up and healthy**

   ```bash
   docker compose ps ai-service

   echo "== AI service /health from host =="
   curl -sS http://localhost:8001/health | jq . || curl -sS http://localhost:8001/health

   echo "== AI service /health from app container =="
   docker compose exec app curl -sS http://ai-service:8001/health | jq . \
     || docker compose exec app curl -sS http://ai-service:8001/health
   ```

3. **Baseline AI metrics and fallback fraction**

   From a terminal with access to the app’s `/metrics`:

   ```bash
   APP_METRICS_BASE=${APP_METRICS_BASE:-${APP_BASE:-http://localhost:3000}}
   echo "== AI request duration histogram =="
   curl -sS "${APP_METRICS_BASE}/metrics" | grep ringrift_ai_request_duration_seconds || true

   echo "== AI request + fallback counters =="
   curl -sS "${APP_METRICS_BASE}/metrics" | grep 'ringrift_ai_\(requests_total\|fallback_total\)' || true
   ```

   In Prometheus (UI), sanity-check:

   ```promql
   ringrift_service_status{service="ai_service"}

   (
     sum(rate(ringrift_ai_fallback_total[10m]))
   )
   /
   (
     sum(rate(ringrift_ai_requests_total[10m]))
   )
   ```

   Expect:
   - `ringrift_service_status{service="ai_service"} == 1`.
   - Fallback fraction close to baseline (often near zero in idle staging).

4. **Optional: baseline AI dashboard view**

   In Grafana:
   - Open the **Game Performance** dashboard:
     - Confirm there are panels for **AI Request Latency** and **AI Request Outcomes & Fallbacks** (as described in `docs/operations/ALERTING_THRESHOLDS.md` under “AI turn SLOs”).
   - Open **System Health**:
     - Confirm HTTP error and latency panels are healthy (no sustained 5xx or high P99).

### 3.2 Confirm HTTP move harness & k6 scenario wiring

1. **Check that the HTTP move harness is enabled**
   - Ensure staging env has:

     ```env
     ENABLE_HTTP_MOVE_HARNESS=true
     ```

     as described in `docs/operations/ENVIRONMENT_VARIABLES.md` (`ENABLE_HTTP_MOVE_HARNESS`).

   - From the app host, a simple 404 vs 405 check is usually enough; exact status codes may vary between versions, so rely on your API reference or k6 behaviour. If in doubt, you can proceed; the k6 scenario will report harness‑availability via its thresholds.

2. **Verify `player-moves.js` is available**

   From the repo root:

   ```bash
   ls tests/load/scenarios/player-moves.js
   ```

   If this file is missing, stop here and review `tests/load/README.md` and `docs/operations/ALERTING_THRESHOLDS.md` for the current load harness.

### 3.3 Induce AI service degradation (canonical: stop the AI container)

For this drill, the **canonical degradation mode** is a **full AI service outage** (from the app’s perspective) while the rest of the stack stays healthy. This is intentionally simple, uses only docker‑compose, and exercises:

- `AIServiceDown` alert (via `ringrift_service_status{service="ai_service"} == 0`).
- Elevated fallback fraction (`AIFallbackRateHigh` / `Critical`).
- Overall `ServiceDegraded` (level 1) via `ringrift_degradation_level`.

> **Do not perform this step in production.** For production incidents, follow `AI_SERVICE_DOWN.md` and `SERVICE_DEGRADATION.md` instead.

1. **Inject degradation – stop the AI service**

   ```bash
   cd /path/to/ringrift

   echo "== Stopping AI service container to simulate outage =="
   docker compose stop ai-service

   echo "== Confirm AI service container is stopped =="
   docker compose ps ai-service
   ```

2. **Observe immediate impact on app readiness**

   ```bash
   APP_BASE=${APP_BASE:-http://localhost:3000}

   echo "== App readiness after AI service stop =="
   curl -sS "${APP_BASE}/ready" | jq . || curl -sS "${APP_BASE}/ready"
   ```

   - Expect the AI dependency entry to be `degraded` or `unhealthy` with a clear error message (e.g. connection failure).
   - Overall readiness may report `degraded`, but database and Redis should remain healthy.

3. **Confirm AI service status metric transitions**

   In Prometheus:

   ```promql
   ringrift_service_status{service="ai_service"}

   ringrift_degradation_level
   ```

   - Within a few minutes, expect `ringrift_service_status{service="ai_service"} == 0`.
   - `ringrift_degradation_level` should be > 0 (typically `1` – DEGRADED), triggering `ServiceDegraded`.

4. **Watch for `AIServiceDown` alert**
   - In Alertmanager or your alert UI, confirm `AIServiceDown` is firing with the expected annotations (runbook URL, impact text).
   - Keep this alert active while you proceed to the load step so you can see it correlate with AI usage.

### 3.4 Drive AI traffic via k6 player-moves scenario

With the AI service stopped, generate traffic that **would normally call the Python AI**, so you can:

- Exercise fallback logic in `AIEngine`.
- Increase `ringrift_ai_requests_total` and `ringrift_ai_fallback_total`.
- Observe dashboards and alerts under sustained AI load.

1. **Run the `player-moves` scenario against staging**

   From the repo root:

   ```bash
   APP_BASE=${APP_BASE:-http://localhost:3000}

   THRESHOLD_ENV=staging \
   BASE_URL="${APP_BASE}" \
   MOVE_HTTP_ENDPOINT_ENABLED=true \
   npx k6 run tests/load/scenarios/player-moves.js
   ```

   Notes:
   - The scenario:
     - Logs in via `/api/auth/login` (using a helper).
     - Creates AI games via `POST /api/games` with `aiOpponents` configured (AI vs human).
     - Optionally submits moves via the **HTTP move harness** when `MOVE_HTTP_ENDPOINT_ENABLED=true` and `ENABLE_HTTP_MOVE_HARNESS=true`.
   - Thresholds for this scenario are wired to the AI and WebSocket SLOs described in `docs/operations/ALERTING_THRESHOLDS.md` (HTTP & game move latency, stall rate, etc.).

2. **Keep the scenario running for at least one full alert window**
   - The AI alerts use windows on the order of **5–10 minutes** (see `monitoring/prometheus/alerts.yml`, `docs/operations/ALERTING_THRESHOLDS.md`).
   - Allow the k6 run to proceed long enough for:
     - `AIFallbackRateHigh` (and possibly `AIFallbackRateCritical`) to fire.
     - `AIServiceDown` to remain active throughout.
     - `ServiceDegraded` to fire due to `ringrift_degradation_level > 0`.

### 3.5 Observe alerts, metrics, and dashboards

While `ai-service` is stopped and `player-moves.js` is running:

1. **Prometheus checks**

   In PromQL (adjust windows to match your alert config; examples assume 5–10m windows):

   ```promql
   # AI availability
   ringrift_service_status{service="ai_service"}

   # Fallback fraction
   (
     sum(rate(ringrift_ai_fallback_total[10m]))
   )
   /
   (
     sum(rate(ringrift_ai_requests_total[10m]))
   )

   # AI request latency (even failures will have some duration)
   histogram_quantile(
     0.99,
     sum(rate(ringrift_ai_request_duration_seconds_bucket[5m])) by (le)
   )

   # Degradation level
   ringrift_degradation_level
   ```

   Expect during the drill:
   - `ringrift_service_status{service="ai_service"} == 0`.
   - Fallback fraction rising toward **1.0** (nearly all AI requests using fallback) while k6 is active.
   - `ringrift_degradation_level > 0` (and `ServiceDegraded` alert firing).
   - `AIServiceDown` and `AIFallbackRateHigh` active in Alertmanager.

2. **Grafana – Game Performance dashboard**
   - Open the **Game Performance** dashboard (`monitoring/grafana/dashboards/game-performance.json` is the source).
   - Locate panels for:
     - **AI Request Latency** (P50/P95/P99 from `ringrift_ai_request_duration_seconds_bucket`).
     - **AI Request Outcomes & Fallbacks** (`ringrift_ai_requests_total` vs `ringrift_ai_fallback_total`).
   - Verify that:
     - The AI latency panel shows elevated or erratic P99 when the service is down (timeouts/connection attempts).
     - The fallback panel shows a large share of requests marked as fallback during the outage, then converges back afterward.

3. **Grafana – System Health dashboard**
   - Open the **System Health** dashboard (`monitoring/grafana/dashboards/system-health.json`).
   - Confirm:
     - HTTP request rate and error share remain acceptable (no large 5xx spike purely due to AI failures).
     - General latency panels (`HighP99Latency`, `HighP95Latency` surfaces) only degrade within the tolerance described in `docs/operations/ALERTING_THRESHOLDS.md`.

4. **Alertmanager**

   During the drill window, verify that:
   - `AIServiceDown` is firing with the `runbook_url` pointing to `AI_SERVICE_DOWN.md`.
   - `AIFallbackRateHigh` (and possibly `AIFallbackRateCritical`) are firing with `runbook_url` pointing to `AI_FALLBACK.md`.
   - `ServiceDegraded` is firing with `runbook_url` pointing to `SERVICE_DEGRADATION.md`.
   - No **unexpected** alerts are firing (if they are, note them as findings).

### 3.6 Observe app behaviour from a player’s perspective

In parallel with k6 (or just before/after), use the staging frontend or API to verify **user‑visible effects**:

1. **Start an AI game**
   - Log in to staging via the normal UI.
   - Create a game vs AI (e.g. 1 AI opponent at difficulty 5, `mode: "service"`).
   - Begin playing moves.

2. **Observe behaviour while AI service is down**

   Expected behaviour:
   - AI moves **still arrive**, but:
     - They may be simpler / weaker (local heuristic fallback).
     - There may be a short initial degradation period while fallbacks ramp up.
   - The UI should **not** show repeated hard errors for AI moves.
   - Game rules remain enforced by the shared TS engine (no illegal moves).

3. **Cross-check logs and diagnostics**
   - `AIEngine` diagnostics are validated by tests such as:
     - `tests/unit/AIEngine.fallback.test.ts`
     - `tests/unit/AIServiceClient.concurrency.test.ts` (where present)
   - During the drill, the behaviour you see should match the expectations encoded in those tests:
     - Service failures and timeouts increment service‑failure counters.
     - Local heuristics are applied when the Python service fails or is unavailable.
     - Node’s concurrency cap and circuit breaker avoid thrashing a downed service.

### 3.7 Recovery and rollback

Once you are satisfied with the degraded-state observations:

1. **Restore the AI service container**

   ```bash
   cd /path/to/ringrift

   echo "== Restarting AI service container =="
   docker compose start ai-service

   echo "== Tail AI service logs during startup =="
   docker compose logs -f ai-service
   ```

   - Wait for FastAPI to start (`uvicorn` logs).
   - Confirm no repeated startup errors.

2. **Confirm AI health and service status**

   ```bash
   APP_BASE=${APP_BASE:-http://localhost:3000}

   echo "== AI service /health from host =="
   curl -sS http://localhost:8001/health | jq . || curl -sS http://localhost:8001/health

   echo "== App readiness after AI service restart =="
   curl -sS "${APP_BASE}/ready" | jq . || curl -sS "${APP_BASE}/ready"
   ```

   In Prometheus:

   ```promql
   ringrift_service_status{service="ai_service"}

   ringrift_degradation_level
   ```

   - Expect `ringrift_service_status{service="ai_service"} == 1` again.
   - `ringrift_degradation_level` should converge back to `0` once `ServiceStatusManager` sees healthy dependencies.

3. **Re-run a short k6 player-moves run (optional)**

   To confirm that AI calls now use the Python service again:

   ```bash
   THRESHOLD_ENV=staging \
   BASE_URL="${APP_BASE}" \
   MOVE_HTTP_ENDPOINT_ENABLED=true \
   npx k6 run tests/load/scenarios/player-moves.js
   ```

   - Check that `ringrift_ai_fallback_total / ringrift_ai_requests_total` returns close to baseline.
   - In Grafana, confirm AI latency and outcome panels reflect normal behaviour.

4. **Manual smoke test**
   - Start a fresh AI game and play several moves.
   - Confirm AI moves are again being served by the remote service (check logs if necessary) and that the in‑game experience feels consistent with pre-drill behaviour.

---

### 3.8 Capture an automated drill report (optional)

To make the drill repeatable and to attach concrete artefacts to release/ops tickets, you can run the lightweight **AI degradation drill harness** from the monorepo. This script does **not** induce degradation itself; it only verifies basic health wiring and writes a JSON report under `results/ops/`.

From the repo root on the staging host:

```bash
# Baseline (before stopping ai-service)
APP_BASE=${APP_BASE:-http://localhost:3000}

./node_modules/.bin/ts-node scripts/run-ai-degradation-drill.ts \
  --env staging \
  --phase baseline

# During induced degradation (ai-service stopped)
./node_modules/.bin/ts-node scripts/run-ai-degradation-drill.ts \
  --env staging \
  --phase degraded

# After recovery (ai-service restarted and healthy)
./node_modules/.bin/ts-node scripts/run-ai-degradation-drill.ts \
  --env staging \
  --phase recovery
```

Notes:

- The harness:
  - Calls `APP_BASE/health` (defaulting to `http://localhost:3000/health` when `APP_BASE` is unset).
  - Uses the server’s `HealthCheckService` readiness check to inspect the `aiService` dependency.
  - Runs a placeholder check for AI fallback behaviour (documented in the JSON report).
- Each run produces `results/ops/ai_degradation.<env>.<phase>.<timestamp>.json` with:
  - `drillType: "ai_service_degradation"`.
  - `environment`, `operator` (when passed via `--operator`), and `phase`.
  - A list of named checks (`backend_http_health`, `ai_service_health`, `ai_fallback_behaviour`) and `overallPass`.
- When filing drill results, you can attach or link these JSON reports as evidence alongside Grafana/Prometheus screenshots.

## 4. Validation Checklist

Use this checklist to decide whether the drill has successfully validated AI degradation handling in staging.

- [ ] **Preconditions:** Staging stack was healthy before the drill (`/health`, `/ready`, `ringrift_service_status{service="ai_service"} == 1`).
- [ ] **Degradation induction:**
  - [ ] `docker compose stop ai-service` successfully stopped only the AI container.
  - [ ] `/ready` showed AI dependency as degraded/unhealthy while DB and Redis remained healthy.
  - [ ] `ringrift_service_status{service="ai_service"} == 0` for the duration of the induced outage.
- [ ] **Alerts:**
  - [ ] `AIServiceDown` fired with correct annotations and cleared after recovery.
  - [ ] `AIFallbackRateHigh` (and, if applicable, `AIFallbackRateCritical`) fired while k6 was generating AI load.
  - [ ] `ServiceDegraded` fired (via `ringrift_degradation_level > 0`) during the outage and cleared after recovery.
- [ ] **Metrics / dashboards:**
  - [ ] `ringrift_ai_requests_total` and `ringrift_ai_fallback_total` increased as expected during the drill.
  - [ ] Fallback fraction was significantly elevated while the AI service was down and returned near baseline afterward.
  - [ ] The Game Performance dashboard showed elevated fallback share and appropriate AI request latency behaviour during the outage.
  - [ ] The System Health dashboard did **not** show unacceptable global HTTP error or latency degradation (beyond the expected impact on AI moves).
- [ ] **User-visible behaviour:**
  - [ ] AI games remained playable during the outage (moves still resolved via fallback).
  - [ ] Moves respected the shared TS rules semantics (no obvious illegal moves or crashes).
  - [ ] After recovery, AI behaviour returned to expected strength and responsiveness.
- [ ] **Post-drill clean-up:**
  - [ ] `docker compose start ai-service` (or equivalent) restored the AI service.
  - [ ] All AI- and degradation-related alerts have cleared in Alertmanager.
  - [ ] Any unexpected alerts or anomalies observed during the drill have been captured as findings (with follow-up issues as appropriate).
  - [ ] The drill has been logged in your internal ops log, optionally cross-referencing `docs/archive/historical/OPERATIONAL_DRILLS_RESULTS_2025_12_03.md`.

---

## 5. Notes for Production Adaptation

This runbook is explicitly scoped to **staging/non‑production** environments. For production:

- Treat this drill as a **template**, not a plug‑and‑play procedure.
- Any production exercise that intentionally degrades AI must:
  - Be coordinated with on-call / SRE and product.
  - Follow your organisation’s change‑management and incident‑simulation policies.
  - Include explicit communication plans (status page, Slack/Teams, support playbooks).

In particular:

- Use `docs/incidents/AI_SERVICE.md` as the primary guide when a **real** AI incident occurs.
- Use `AI_SERVICE_DOWN.md`, `AI_FALLBACK.md`, `AI_PERFORMANCE.md`, and `SERVICE_DEGRADATION.md` as **live runbooks** for diagnosis and mitigation under production conditions.
- If you design a production drill:
  - Prefer techniques that are:
    - Reversible (e.g. controlled `ai-service` rollout or feature-flag based AI disablement).
    - Time-bounded (short, scheduled maintenance window).
    - Clearly labelled and announced as a drill.
  - Capture the drill outcome and findings in your incident/post‑mortem tracking system.

This document remains **non‑SSoT operational guidance** and defers to:

- `monitoring/prometheus/alerts.yml` and `docs/operations/ALERTING_THRESHOLDS.md` for alerting rules and thresholds.
- `docs/operations/ENVIRONMENT_VARIABLES.md` for authoritative environment variable definitions.
- `src/server/services/AIServiceClient.ts`, `src/server/game/ai/AIEngine.ts`, and the associated `tests/unit/*.test.ts` files for AI integration and fallback semantics.
- `docs/incidents/AI_SERVICE.md` and `docs/runbooks/SERVICE_DEGRADATION.md` for broader incident handling and service-wide degradation semantics.
