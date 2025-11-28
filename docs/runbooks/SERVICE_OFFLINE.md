# Service Offline Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for handling `ServiceOffline` alerts indicating a full application outage.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - Prometheus alert rules in `monitoring/prometheus/alerts.yml` (e.g. `ServiceOffline`, `ServiceMinimalMode`, `ServiceDegraded`, `ringrift_degradation_level`, `DatabaseDown`).
> - Prometheus configuration in `monitoring/prometheus/prometheus.yml`.
> - Backend degradation and offline plumbing in `src/server/services/ServiceStatusManager.ts` and `src/server/middleware/degradationHeaders.ts`.
> - Health / readiness and dependency checks implemented by `HealthCheckService` and related routes.
>
> **Precedence:** Alert thresholds, degradation semantics, health check logic, and deployment topology are defined by monitoring **configs + code**. This runbook is a playbook for investigation and remediation only and **must not be treated as a source of truth** for metrics or thresholds.

---

## 1. When This Alert Fires

**Primary alert (see `monitoring/prometheus/alerts.yml`):**

- `ServiceOffline` (critical)
  - `expr: ringrift_degradation_level == 3` (for 30s).
  - Indicates the backend has entered **OFFLINE** mode and is treating the situation as a full outage.

Related degradation alerts that often accompany `ServiceOffline`:

- `ServiceMinimalMode` (critical): `ringrift_degradation_level >= 2` (sustained).
- `ServiceDegraded` (warning): `ringrift_degradation_level > 0`.

Common correlated dependency alerts:

- `DatabaseDown`: `ringrift_service_status{service="database"} == 0` (critical).
- `RedisDown`: `ringrift_service_status{service="redis"} == 0`.
- AI-service alerts (`AIServiceDown`, `AIFallbackRateHigh`, `AIRequestHighLatency`, `AIErrorsIncreasing`).

When `ServiceOffline` is firing you should assume:

- `ServiceStatusManager` has classified the system as `DegradationLevel.OFFLINE` based on dependency failures (typically database).
- `offlineModeMiddleware` is **blocking most requests** with HTTP 503 (`SERVICE_UNAVAILABLE`) responses, except for health/ready endpoints.
- The system is in **major-incident** territory: user-facing gameplay and/or authentication are unavailable.

---

## 2. Triage

> Goal: Confirm that this is a real, user-impacting outage, identify which critical dependency failed (most often the database), and gather context before taking action.

1. **Confirm the alert and scope in monitoring**
   - In Alertmanager / UI:
     - Confirm `ServiceOffline` is active and note **start time** and **duration**.
     - Check whether `ServiceMinimalMode` / `ServiceDegraded` are also active.
     - Look for correlated alerts in the **availability**, **service-response**, and **ai-service** groups, especially `DatabaseDown`.
   - In Prometheus, run:

     ```promql
     # Current degradation level
     ringrift_degradation_level

     # Has the service been offline for a while?
     max_over_time(ringrift_degradation_level[30m])

     # Critical dependency status
     ringrift_service_status
     ```

   - Verify that `ringrift_degradation_level` is consistently at the OFFLINE level (3) and that `ringrift_service_status{service="database"}` is unhealthy if `DatabaseDown` is firing.

2. **Check health and readiness endpoints from a client perspective**

   Replace `APP_BASE` appropriately for the environment (e.g. `http://localhost:3000` for local docker-compose, or the production/staging base URL):

   ```bash
   # Overall liveness
   curl -v APP_BASE/health

   # Readiness / dependency checks
   curl -v APP_BASE/ready
   ```

   - Expected behaviour while `ServiceOffline` is active:
     - `/health` may still return 200 (container is running) or 503 depending on how health is wired.
     - `/ready` is likely failing or reporting dependencies as unhealthy.
   - Capture:
     - HTTP status codes.
     - Response bodies (especially from `/ready`, which should include checks for **database**, **redis**, **aiService**, etc.).

3. **Inspect degradation headers and offline behaviour**
   - Call a normal API endpoint (e.g. `/api/health`):

     ```bash
     curl -sD- -o /dev/null APP_BASE/api/health | grep -E 'HTTP/|X-Service-Status|X-Degraded-Services|Retry-After' || true
     ```

   - Confirm:
     - Most non-health endpoints are returning **503 Service Unavailable** with a JSON body containing `SERVICE_UNAVAILABLE` as the error code (from `offlineModeMiddleware`).
     - Headers:
       - `X-Service-Status: offline` (or equivalent for offline level).
       - `X-Degraded-Services` listing critical services such as `database`.
       - `Retry-After` header (typically 60 seconds) on 503s.

4. **Check docker-compose / container status**

   On the host where the RingRift stack is running:

   ```bash
   cd /path/to/ringrift

   # High-level container status
   docker compose ps

   # Focus on app + core dependencies
   docker compose ps app database redis ai-service

   # Last logs for the app and database
   docker compose logs app --tail=200
   docker compose logs database --tail=200
   ```

   - For the **database** container, look for:
     - Crash loops, repeated restarts.
     - Startup errors (migrations, schema, authentication, disk full).
     - Connection refused / timeout errors.
   - For the **app** container, look for:
     - Log messages from `ServiceStatusManager` indicating database is unhealthy.
     - "System degradation level changed" messages showing transition to `OFFLINE`.
     - Repeated 503 responses logged by `offlineModeMiddleware`.

5. **Classify the root cause category**

   Based on the above:
   - **Database outage or severe degradation**
     - `DatabaseDown` and/or `DatabaseResponseTimeSlow` firing.
     - Database container unhealthy or unreachable.  
       → Primary reference: `DATABASE_DOWN.md`, `DATABASE_PERFORMANCE.md`.

   - **Infrastructure / platform failure**
     - Containers failing to start, host node issues, disk full, network segmentation.  
       → Use deployment runbooks (`DEPLOYMENT_INITIAL.md`, `DEPLOYMENT_ROUTINE.md`, `DEPLOYMENT_ROLLBACK.md`) and infrastructure playbooks outside this repo.

   - **Configuration or rollout regression**
     - Offline started immediately after a new deploy or config change.
     - Migration errors, schema mismatch, or secrets/env misconfiguration.  
       → Cross-reference `DEPLOYMENT_REQUIREMENTS.md`, `OPERATIONS_DB.md`, and relevant deployment runbooks.

   - **Cascading dependency failure**
     - Multiple dependencies failing (Redis, AI) plus database issues or high error/latency alerts.  
       → Correlate with `REDIS_DOWN.md`, `AI_SERVICE_DOWN.md`, `HIGH_ERROR_RATE.md`, `HIGH_LATENCY.md`.

6. **Log the incident context**

   Before attempting remediation, capture at least:
   - Time `ServiceOffline` first fired.
   - Snapshot of key metrics/alerts (screenshots or copied queries).
   - Current versions / image tags deployed for app, database, and AI service.
   - Any recent changes (deployments, migrations, infra changes).

   These will be used later with `docs/incidents/AVAILABILITY.md`, `TRIAGE_GUIDE.md`, and the `POST_MORTEM_TEMPLATE.md`.

---

## 3. Remediation (High Level)

> Goal: Restore the system from **OFFLINE** mode to **MINIMAL/DEGRADED** and then to **FULL**, with a clear understanding of what changed.

### 3.1 Stabilise the database (likely root cause)

If there is any sign of database trouble, treat it as **P0**:

1. **Follow the `DATABASE_DOWN.md` runbook:**
   - Verify database container health (`docker compose ps database`).
   - Inspect logs in detail (`docker compose logs database`).
   - Validate connectivity from the app host/cluster.
   - Confirm disk usage, connection limits, and any migration failures.

2. **Check schema / migration state**
   - Review `docs/OPERATIONS_DB.md` for expected migration flows.
   - If a migration was recently applied, confirm it completed successfully and did not leave the DB in a partially applied state.

3. **If necessary, roll back or roll forward a deployment**
   - Use `DEPLOYMENT_ROLLBACK.md` if the current app build is incompatible with the DB state or is clearly causing offline mode.
   - Use `DEPLOYMENT_ROUTINE.md` if you need a clean re-deploy after fixing infra.

> Do **not** attempt to override `ringrift_degradation_level` directly; fix the underlying DB problem and allow `ServiceStatusManager` + health checks to promote the system out of offline.

### 3.2 Recover other critical dependencies

Once the database is stable (no more `DatabaseDown` / `DatabaseResponseTimeSlow`), ensure other dependencies are not keeping the system in OFFLINE or MINIMAL mode:

- **Redis:**
  - Follow `REDIS_DOWN.md` / `REDIS_PERFORMANCE.md` to restore service and latency.
  - Confirm `ringrift_service_status{service="redis"}` returns healthy.

- **AI Service:**
  - Follow `AI_SERVICE_DOWN.md`, `AI_FALLBACK.md`, `AI_PERFORMANCE.md`, `AI_ERRORS.md` as appropriate.
  - Ensure that AI failures alone are not driving the system into offline (they should typically map to degraded/minimal instead).

- **Platform / environment:**
  - If containers are failing due to host-level constraints (CPU/memory/disk), resolve those via infra playbooks.
  - Confirm network connectivity between app and database/Redis/AI service.

### 3.3 Clear bad deployments and configuration issues

If the outage started immediately after a change:

1. **Freeze further changes** until stability is restored.
2. **Roll back** the app or AI service to the last known good version using `DEPLOYMENT_ROLLBACK.md`.
3. Validate that `ServiceOffline` clears once you are back on the previous version and dependencies are healthy.
4. If the issue is due to environment variables or secrets, cross-check `ENVIRONMENT_VARIABLES.md` and `SECRETS_MANAGEMENT.md` before re-attempting rollout.

### 3.4 Communication and coordination

While remediation is in progress:

- **Treat this as a major incident:**
  - Follow your organisation’s incident management process (on-call escalation, incident channel, status page updates).
  - Refer to `docs/incidents/AVAILABILITY.md` and `TRIAGE_GUIDE.md` for guidance on documenting the incident.

- **Set expectations for users:**
  - If you control an external load balancer or reverse proxy, consider serving a static maintenance page while the backend is in OFFLINE mode.
  - Communicate expected timelines and next update times.

---

## 4. Validation

> Only close the incident once metrics, health surfaces, and smoke tests all confirm recovery.

**Metrics / alerts:**

- [ ] `ringrift_degradation_level` returns to the non-offline levels, then to `0` and remains there for at least one full alerting window:  
       `max_over_time(ringrift_degradation_level[10m]) == 0`.
- [ ] `ServiceOffline` alert has cleared in Alertmanager and stays clear.
- [ ] `ServiceMinimalMode` / `ServiceDegraded` alerts also clear, or stabilise at an acceptable level if you are intentionally running in a temporary degraded mode.
- [ ] Correlated alerts (`DatabaseDown`, `RedisDown`, `HighErrorRate`, `HighP99Latency`, `AIServiceDown`, etc.) are resolved.

**Health / readiness / headers:**

- [ ] `curl APP_BASE/health` returns 200 and indicates healthy state.
- [ ] `curl APP_BASE/ready` returns 200 and all dependency checks show healthy.
- [ ] A request to `/api/health` returns 2xx with **no** `X-Service-Status` offline indicator, and no `Retry-After` header from `offlineModeMiddleware`.

**User flows / end-to-end tests:**

- [ ] Users can log in and reach the lobby page.
- [ ] Users can start a new game and play several moves without unexpected errors or timeouts.
- [ ] Any automated smoke tests / Playwright suites for the environment pass (see existing e2e tests).

After validation, update the relevant incident record in `docs/incidents/AVAILABILITY.md` and, if the impact was material, complete a post-mortem using `docs/incidents/POST_MORTEM_TEMPLATE.md`.

---

## 5. TODO / Environment-Specific Notes

Populate the items below for each environment (staging, production, etc.) and keep them current:

- [ ] Document who owns on-call for **database**, **Redis**, **AI service**, and **infrastructure** in this environment (teams, escalation paths, contact methods).
- [ ] List links to environment-specific dashboards (Grafana or equivalent) that show `ringrift_degradation_level`, `ringrift_service_status`, error rates, and latency.
- [ ] Capture any special maintenance-mode or traffic-control mechanisms (load balancer features, feature flags, canary routes) that operators can use during offline events.
- [ ] Note any environment-specific caveats (e.g. read-only replicas, scheduled maintenance windows, backup/restore runbooks) that interact with offline behaviour.
