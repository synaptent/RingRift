# Service Degradation Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for responding to `ServiceDegraded`, `ServiceMinimalMode`, and related degradation-level alerts.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - Prometheus alert rules in `monitoring/prometheus/alerts.yml` (e.g. `ServiceDegraded`, `ServiceMinimalMode`, `ServiceOffline`, `ringrift_degradation_level`).
> - Prometheus configuration in `monitoring/prometheus.yml`.
> - Backend degradation implementation in `src/server/services/ServiceStatusManager.ts` and `src/server/middleware/degradationHeaders.ts`.
> - Health / readiness plumbing in `src/server/services/HealthCheckService.ts` and the Express routing layer.
>
> **Precedence:** Alert thresholds, metric names, and degradation semantics are defined by monitoring configs + backend code above. **Do not treat this file as a source of truth** for semantics or thresholds; use it only as a playbook for investigation and mitigation.

---

## 1. When These Alerts Fire

**Prometheus alerts (see `monitoring/prometheus/alerts.yml`):**

- `ServiceDegraded` (warning)
  - `expr: ringrift_degradation_level > 0` (sustained for 5m).
  - Indicates the service is no longer in full mode.
- `ServiceMinimalMode` (critical)
  - `expr: ringrift_degradation_level >= 2` (sustained for 1m).
  - Indicates we are in **minimal** or **offline** mode.
- `ServiceOffline` (critical)
  - `expr: ringrift_degradation_level == 3` (30s).
  - Indicates complete outage; handled in more detail in `SERVICE_OFFLINE.md`.

Current degradation level semantics (authoritatively implemented in `ServiceStatusManager` and described in alert annotations):

- `FULL` (level 0): All services available, full functionality.
- `DEGRADED` (level 1): Non-critical services unavailable (e.g. AI service or Redis unhealthy/degraded).
- `MINIMAL` (level 2): Only core functionality available (local AI, limited matchmaking, in-memory fallbacks).
- `OFFLINE` (level 3): Critical services unavailable (database down), maintenance / outage mode.

When these alerts fire you should assume:

- The **backend has already evaluated dependency health** via `ServiceStatusManager` and surfaced an aggregate level into the `ringrift_degradation_level` metric.
- The **degradation headers middleware** (`degradationHeadersMiddleware`) is adding:
  - `X-Service-Status: degraded|minimal|offline`
  - `X-Degraded-Services: database,redis,aiService` (comma-separated, when applicable)
    to HTTP responses from the app.
- Requests may be blocked by `offlineModeMiddleware` for non-health paths when level is `OFFLINE`.

---

## 2. Triage

Work through these steps before changing anything:

1. **Confirm which degradation alert(s) are firing**
   - In Alertmanager / UI, note:
     - Which of `ServiceDegraded`, `ServiceMinimalMode`, `ServiceOffline` is active.
     - The current `ringrift_degradation_level` value and how long it has been non-zero.
   - In Prometheus, run:
     - `ringrift_degradation_level`
     - `max_over_time(ringrift_degradation_level[30m])`  
       This shows how long we have been degraded and whether the level is increasing.

2. **Check application health and readiness endpoints**
   - Against the app base URL (replace `APP_BASE` appropriately; for local docker-compose this is typically `http://localhost:3000`):

     ```bash
     # Overall health (lightweight)
     curl -sS APP_BASE/health | jq . || curl -sS APP_BASE/health

     # Readiness / dependency checks
     curl -sS APP_BASE/ready | jq . || curl -sS APP_BASE/ready
     ```

   - Inspect the JSON from `/ready` (shaped by `HealthCheckService`):
     - Overall status (e.g. `status: "ok" | "degraded" | "unhealthy"`).
     - Per-check details for dependencies such as **database**, **redis**, and **aiService**.
   - If `/ready` fails or returns non-2xx, treat this as a strong signal of deeper dependency problems.

3. **Inspect degradation headers on a normal API endpoint**
   - Pick a simple JSON endpoint such as `/api` or `/health` and inspect only headers:

     ```bash
     curl -sD- -o /dev/null APP_BASE/api | grep -E 'X-Service-Status|X-Degraded-Services' || true
     ```

   - Confirm:
     - `X-Service-Status` matches the Prometheus `ringrift_degradation_level` classification (`degraded`, `minimal`, `offline`).
     - `X-Degraded-Services` lists the same dependencies that show as unhealthy/degraded in `/ready`.

4. **Correlate with other alerts (root cause hunting)**

   Check the following alert groups and associated runbooks:
   - **Infrastructure / dependencies:**
     - `DatabaseDown`, `DatabaseResponseTimeSlow` → see `DATABASE_DOWN.md`, `DATABASE_PERFORMANCE.md`.
     - `RedisDown`, `RedisResponseTimeSlow` → see `REDIS_DOWN.md`, `REDIS_PERFORMANCE.md`.
   - **AI service:**
     - `AIServiceDown`, `AIFallbackRateHigh`, `AIRequestHighLatency`, `AIErrorsIncreasing`  
       → see `AI_SERVICE_DOWN.md`, `AI_FALLBACK.md`, `AI_PERFORMANCE.md`, `AI_ERRORS.md`.
   - **Traffic / user experience:**
     - `HighErrorRate`, `ElevatedErrorRate` → `HIGH_ERROR_RATE.md`.
     - `HighP95Latency` / `HighP99Latency*` → `HIGH_LATENCY.md`.
     - `NoHTTPTraffic` / `NoWebSocketConnections` / `HighWebSocketConnections`  
       → `NO_TRAFFIC.md`, `WEBSOCKET_ISSUES.md`, `WEBSOCKET_SCALING.md`.

   The `ServiceDegraded` / `ServiceMinimalMode` alerts are **aggregate symptoms**; root cause will almost always be visible in one of the above categories.

5. **Inspect application logs for degradation and dependency errors**

   In the host running the app (docker-compose by default):

   ```bash
   cd /path/to/ringrift

   # Check container state
   docker compose ps

   # Tail recent app logs, looking for dependency failures or degradation transitions
   docker compose logs app --tail=300 | sed -n '1,200p'
   ```

   Look for log lines emitted by `ServiceStatusManager` and `degradationLoggingMiddleware`, for example:
   - "Service status polling started" / "Service status manager reset".
   - "Service became unavailable" with `service=database|redis|aiService`.
   - "System degradation level changed" with `oldLevel` / `newLevel` and degraded services.
   - Repeated `SERVICE_UNAVAILABLE` responses from `offlineModeMiddleware`.

6. **Classify the current mode based on findings**
   - **Level 1 – DEGRADED:**
     - Database healthy; Redis and/or AI service degraded/unhealthy.
     - Most HTTP endpoints still available; some non-core features impacted.
   - **Level 2 – MINIMAL:**
     - Database degraded or intermittently failing **and** at least one non-critical service degraded/unhealthy, **or** all non-critical services down.
     - Only core gameplay flows may be enabled; heavy features (matchmaking, analytics, etc.) reduced.
   - **Level 3 – OFFLINE:**
     - Database unhealthy → overall `OFFLINE` (requests blocked except health / ready).
     - Treat as full outage and pivot quickly to `SERVICE_OFFLINE.md`.

---

## 3. Remediation (High Level)

Once you have identified which services are degraded, follow these paths.

1. **If level is DEGRADED (warning-level impact)**

   Focus on restoring non-critical services while keeping the core game stable:
   - **Database healthy, Redis or AI degraded:**
     - For Redis issues, follow `REDIS_DOWN.md` / `REDIS_PERFORMANCE.md`:
       - Check Redis container (`docker compose ps redis`).
       - Inspect Redis logs (`docker compose logs redis --tail=200`).
       - Validate Redis connectivity from the app container (e.g. using `redis-cli` if available or app logs).
     - For AI service issues, follow `AI_SERVICE_DOWN.md`, `AI_FALLBACK.md`, `AI_PERFORMANCE.md`, `AI_ERRORS.md`:
       - Confirm AI health endpoints as described there.
       - Check AI container logs and resource usage.
   - **Application-level symptoms:**
     - If `HighErrorRate` / `HighLatency` are also firing, work through those runbooks in parallel.
   - Avoid toggling degradation levels manually. Instead:
     - Fix the underlying service (restart containers if needed, resolve network / configuration issues).
     - Let `ServiceStatusManager` recompute the level via its health checks.

2. **If level is MINIMAL (critical impact)**

   Treat this as a serious incident where **only core flows** should remain available:
   - Expect at least one of the following:
     - Database degraded (intermittent failures or high latency).
     - Multiple non-critical services (Redis, AI) degraded/unhealthy.
   - Actions:
     - Prioritize **database health** first via `DATABASE_DOWN.md` / `DATABASE_PERFORMANCE.md`.
     - In parallel, stabilize Redis and AI per their runbooks so that, once DB is healthy, the system can quickly auto-promote out of minimal mode.
     - Coordinate with product / support to communicate reduced capabilities to users.

3. **If level is OFFLINE (ServiceOffline alert)**
   - Immediately follow the `SERVICE_OFFLINE.md` runbook for **full outage response**.
   - Use this `SERVICE_DEGRADATION.md` runbook mainly to:
     - Understand what caused the offline transition (which services failed, in what order).
     - Cross-check that any attempted recovery smoothly transitions through `MINIMAL` / `DEGRADED` back to `FULL`.

4. **Deployment / config considerations**
   - If degradation correlates strongly with a recent deploy:
     - Use `DEPLOYMENT_ROLLBACK.md` / `DEPLOYMENT_ROUTINE.md` as appropriate.
     - Verify that degradation clears after rollback before attempting re-rollout.
   - Configuration and thresholds for degradation are defined in code/monitoring:
     - If thresholds in alerts or health checks are too aggressive/lenient, **update `alerts.yml`, `ALERTING_THRESHOLDS.md`, and/or the health check implementation in code**, then run validation scripts and the SSoT harness rather than editing this runbook.

---

## 4. Validation

Before declaring the incident resolved:

- **Metrics / alerts:**
  - [ ] `ringrift_degradation_level` returns to `0` and remains there for at least one full alerting window:  
         `max_over_time(ringrift_degradation_level[10m]) == 0`.
  - [ ] `ServiceDegraded`, `ServiceMinimalMode`, and `ServiceOffline` alerts have cleared in Alertmanager.
  - [ ] Any correlated alerts used to identify the root cause (e.g. `DatabaseDown`, `RedisDown`, `AIServiceDown`, `HighErrorRate`, `HighP99Latency`) have also cleared.

- **Health / readiness / headers:**
  - [ ] `curl APP_BASE/health` returns 200 with healthy status.
  - [ ] `curl APP_BASE/ready` returns 200 and all dependency checks show healthy.
  - [ ] `curl -sD- -o /dev/null APP_BASE/api | grep X-Service-Status` shows **no** degradation headers (either header absent or status indicating `full`).

- **User flows / smoke tests:**
  - [ ] Basic authentication, lobby navigation, and game start flows succeed.
  - [ ] New games can be created and moves can be played without unexpected errors or timeouts.
  - [ ] If available, relevant CI or Playwright smoke tests pass when run against the environment.

---

## 5. TODO / Environment-Specific Notes

These items should be filled in and kept up to date **per environment** (staging, production, etc.):

- [ ] Document the exact mapping between `ringrift_degradation_level` numeric values and `DegradationLevel` enum, linking to `docs/operations/ALERTING_THRESHOLDS.md` and the relevant sections in `ServiceStatusManager`.
- [ ] List which **features / endpoints** are intentionally disabled or restricted at each level (`DEGRADED`, `MINIMAL`, `OFFLINE`) for your environment.
- [ ] Capture any **operator controls** used to influence degradation behavior (feature flags, admin endpoints, maintenance toggles) and where they are configured.
- [ ] Note any environment-specific dashboards (Grafana, etc.) used to visualize degradation and dependency health for quick triage.
