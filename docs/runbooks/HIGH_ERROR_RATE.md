# High Error Rate Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for diagnosing and mitigating elevated HTTP 5xx error rates when `HighErrorRate` or `ElevatedErrorRate` alerts fire.
>
> **SSoT alignment:** This runbook is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rules in `monitoring/prometheus/alerts.yml` (alerts `HighErrorRate` and `ElevatedErrorRate` based on `http_requests_total`).
> - **Metrics surface:** HTTP metrics emitted by `metricsMiddleware` / `MetricsService` (`src/server/middleware/metricsMiddleware.ts`, `src/server/services/MetricsService.ts`) and exposed via `/metrics`.
> - **Health/readiness surface:** `HealthCheckService` and `ServiceStatusManager` (`src/server/services/HealthCheckService.ts`, `src/server/services/ServiceStatusManager.ts`) behind `/health` and `/ready`.
> - **Availability incident docs:** `docs/incidents/AVAILABILITY.md` ("Alert: HighErrorRate" section) and related dependency runbooks (`DATABASE_DOWN.md`, `REDIS_DOWN.md`, `AI_SERVICE_DOWN.md`).
>
> **Precedence:**
>
> - Prometheus alert definitions, metrics code, and backend tests are authoritative for **what is measured and when alerts fire**.
> - This document only describes **how to investigate and mitigate**. If anything here disagrees with code/config/tests, **code + configs + tests win** and this runbook should be updated.

---

## 1. When These Alerts Fire

**Alerts (from `monitoring/prometheus/alerts.yml`, `availability` group):**

- `HighErrorRate` (critical)
- `ElevatedErrorRate` (warning)

**PromQL (conceptual, do not edit here):**

```promql
# HighErrorRate (critical)
(
  sum(rate(http_requests_total{status=~"5.."}[5m]))
  /
  sum(rate(http_requests_total[5m]))
) > 0.05

# ElevatedErrorRate (warning)
(
  sum(rate(http_requests_total{status=~"5.."}[5m]))
  /
  sum(rate(http_requests_total[5m]))
) > 0.01
```

**Impact (from alert annotations + availability doc):**

- Users see failures on a non-trivial fraction of requests (1–5%+).
- Often correlates with:
  - Dependency issues (`DatabaseDown`, `RedisDown`, `AIServiceDown`).
  - Deployment/config regressions.
  - Resource saturation (see `HighMemoryUsage`, `HighEventLoopLag`).

**Related incident docs:**

- Availability overview: `docs/incidents/AVAILABILITY.md` (sections **HighErrorRate** / **ElevatedErrorRate**).
- Dependency-specific: `DATABASE_DOWN.md`, `REDIS_DOWN.md`, `AI_SERVICE_DOWN.md`.

---

## 2. Quick Triage (First 5–10 Minutes)

Goal: Confirm the signal is real, scope it, and determine if it’s **infra/dep**, **app code**, or **external traffic**.

### 2.1 Confirm the alert and basic health

1. **In Alertmanager / Grafana**
   - Confirm which alert is firing (`HighErrorRate` vs `ElevatedErrorRate`).
   - Check **duration** and whether it is:
     - A short spike (deployment, rollout, transient dep issue), or
     - Sustained (likely ongoing incident).
   - Look at **correlated alerts**:
     - `DatabaseDown`, `RedisDown`, `AIServiceDown`.
     - `HighMemoryUsage`, `HighEventLoopLag`, `ServiceDegraded`, `ServiceOffline`.

2. **Check health and readiness endpoints** on the app host:

   ```bash
   # Liveness
   curl -s http://localhost:3000/health | jq

   # Readiness (includes dependency breakdown via HealthCheckService)
   curl -s http://localhost:3000/ready | jq
   ```

   - Look for `status` and `checks.database / checks.redis / checks.aiService` fields.
   - If `database.status === "unhealthy"`, this is likely a dependency issue → pivot to `DATABASE_DOWN.md`.

### 2.2 Inspect HTTP error metrics

Use Prometheus UI (or Grafana with Prometheus as data source).

1. **Overall error rate (sanity check)**

   ```promql
   # Overall 5xx rate and percentage
   sum(rate(http_requests_total{status=~"5.."}[5m]))
   /
   sum(rate(http_requests_total[5m]))
   ```

2. **Break down by status code**

   ```promql
   sum(rate(http_requests_total{status=~"5.."}[5m])) by (status)
   ```

   - A spike in **500** usually means _application error_.
   - A spike in **502/503** often means _upstream/dependency_ or _load balancer/orchestrator_ issues.

3. **Break down by route** (paths are normalized by `metricsMiddleware`):

   ```promql
   sum(rate(http_requests_total{status=~"5.."}[5m])) by (method, path)
   ```

   - Identify whether a **single endpoint** is responsible, or errors are widespread.

4. **Sanity check that metrics are flowing**

   ```bash
   # From the app host
   curl -s http://localhost:3000/metrics | grep http_requests_total | head
   ```

   If `/metrics` is not reachable, investigate **service availability** first (`AVAILABILITY.md`).

### 2.3 Check logs for representative failures

From the app host:

```bash
# Tail recent app logs, filter error-level lines
docker compose logs --tail 500 app 2>&1 | grep -i "error" | tail -n 50

# Optionally, look for stack traces around 5xx responses
docker compose logs --tail 1000 app 2>&1 | grep -E '"statusCode":5[0-9]{2}' | tail -n 50
```

- Look for **repeated stack traces**, **ECONNREFUSED/ETIMEDOUT** to dependencies, or **validation/runtime exceptions**.
- Cross-reference the failing **URL/path** with the Prometheus breakdown above.

---

## 3. Deep Diagnosis

Use this section once you’ve confirmed the alert is real and not a brief spike.

### 3.1 Classify the failure mode

Use combinations of health, metrics, and logs to categorize the error:

1. **Dependency failures (DB / Redis / AI service)**
   - Signs:
     - `/ready` shows `database.status === "unhealthy"` or Redis/AI errors.
     - Logs show `ECONNREFUSED`, timeouts, or connection pool exhaustion.
     - Correlated alerts: `DatabaseDown`, `RedisDown`, `AIServiceDown`, `DatabaseResponseTimeSlow`, `RedisResponseTimeSlow`.
   - Action: **Pivot to dependency-specific runbooks**:
     - `DATABASE_DOWN.md` / `DATABASE_PERFORMANCE.md`.
     - `REDIS_DOWN.md` / `REDIS_PERFORMANCE.md`.
     - `AI_SERVICE_DOWN.md` / `AI_PERFORMANCE.md` / `AI_ERRORS.md`.

2. **Application code / regression**
   - Signs:
     - `/ready` is healthy, dependencies look fine.
     - Error rate spikes immediately after a deployment or feature flag change.
     - Logs show consistent stack traces from domain routes (e.g. `src/server/routes/game.ts`, `src/server/routes/auth.ts`).
     - `http_requests_total` breakdown shows a single or small set of endpoints dominating 5xx.
   - Action:
     - Identify the failing route and owning team.
     - Compare behaviour against tests / expected contracts.
     - Consider **rolling back** the last deployment if clearly correlated.

3. **Resource pressure / saturation**
   - Signs:
     - Correlated alerts: `HighMemoryUsage`, `HighEventLoopLag`, `HighGameMoveLatency`, `ServiceDegraded`.
     - Logs show `ENOMEM`, timeouts, or slow responses.
     - Latency metrics (`http_request_duration_seconds_bucket`) show elevated P95/P99.
   - Action:
     - Follow `docs/incidents/RESOURCES.md` and runbooks for high memory and event loop lag.
     - Consider temporary **traffic throttling** or **scaling out**.

4. **External client / abuse patterns**
   - Signs:
     - Error rate isolated to specific endpoints with unusual request patterns.
     - High rate-limit hits (`HighRateLimitHits`, `SustainedRateLimiting`).
   - Action:
     - Inspect rate limiting metrics and logs (`ringrift_rate_limit_hits_total`, middleware in `src/server/middleware/rateLimiter.ts`).
     - Adjust application-level mitigations or WAF rules as per your environment.

### 3.2 Prometheus queries for targeted diagnosis

These are **examples**; use them as a starting point in the Prometheus UI:

```promql
# 5xx by endpoint and status (last 10 minutes)
sum(rate(http_requests_total{status=~"5.."}[10m])) by (method, path, status)

# Compare 5xx rate vs total per endpoint
sum(rate(http_requests_total{status=~"5.."}[10m])) by (path)
/
sum(rate(http_requests_total[10m])) by (path)

# Check for concurrent latency issues
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, path))
```

Use these to answer:

- Is the problem **one endpoint** or **multiple**?
- Are errors **co-located with high latency**?
- Do errors line up with **dependency alerts** or **resource alerts**?

---

## 4. Remediation Playbooks

> **Important:** These actions should be guided by your environment’s change-management and rollback policies. For production, coordinate with the on-call lead.

### 4.1 If a dependency is failing

1. Confirm via readiness response:

   ```bash
   curl -s http://localhost:3000/ready | jq '.checks'
   ```

   - If `database.status === "unhealthy"` → follow `DATABASE_DOWN.md`.
   - If Redis or AI service is degraded → follow `REDIS_DOWN.md` or `AI_SERVICE_DOWN.md`.

2. Once the dependency is restored, **do not immediately declare success**:
   - Watch `http_requests_total` 5xx fraction for at least one alert window (5–10 minutes).
   - Confirm `HighErrorRate` / `ElevatedErrorRate` alerts clear.

3. If the service has entered degraded or offline mode (`ServiceDegraded`, `ServiceOffline`):
   - See `docs/incidents/AVAILABILITY.md` → **ServiceDegraded / ServiceOffline** sections.

### 4.2 If this is a bad deployment / code regression

1. Identify the offending change:

   ```bash
   # On the app host
   cd /opt/ringrift   # or your deployment root
   git log --oneline -5
   ```

2. Correlate error spike time with the most recent deployment:
   - Check CI/CD logs and the `DEPLOYMENT_ROUTINE.md` history if you track releases.

3. Consider a **rollback** using the deployment runbooks:
   - `DEPLOYMENT_ROLLBACK.md` for full, documented rollback.
   - Ensure DB migrations / schema changes are compatible with rolling back the app.

4. After rollback:
   - Re-run a small smoke test (see below) and confirm error rate returns to baseline.

### 4.3 If errors are localized to a small set of endpoints

1. Use Prometheus to identify the top failing routes:

   ```promql
   topk(5, sum(rate(http_requests_total{status=~"5.."}[10m])) by (method, path))
   ```

2. Use logs to capture **representative requests**:

   ```bash
   docker compose logs --tail 1000 app 2>&1 \
     | grep -E '"statusCode":5[0-9]{2}' \
     | head -n 50
   ```

3. For each failing endpoint:
   - Validate request/response contracts against `docs/API_REFERENCE.md` (or your API docs).
   - Check relevant route handlers and middleware:
     - `src/server/routes/*.ts` for handlers.
     - `src/server/middleware/errorHandler.ts` for standardized error responses.
     - Any feature-flag or rollout logic touching that endpoint.

4. If safe, **temporarily disable or gate** the problematic feature via:
   - Config flags (as defined in `src/server/config/**/*.ts`).
   - API-level soft-fail behaviour, if implemented.

---

## 5. Validation and Recovery

You are done when **all** of the following hold:

1. **Prometheus metrics**
   - Overall error rate back under warning threshold:

     ```promql
     (
       sum(rate(http_requests_total{status=~"5.."}[5m]))
       /
       sum(rate(http_requests_total[5m]))
     ) < 0.01
     ```

   - No sustained spikes for target endpoints in the last alert window.

2. **Alerts**
   - `HighErrorRate` / `ElevatedErrorRate` alerts have **cleared** in Alertmanager.
   - Any correlated dependency or resource alerts (`DatabaseDown`, `HighMemoryUsage`, etc.) are resolved.

3. **Health and readiness**

   ```bash
   curl -s http://localhost:3000/health | jq
   curl -s http://localhost:3000/ready | jq
   ```

   - `status` is `"healthy"` or at worst `"degraded"` with understood cause.

4. **Synthetic checks / smoke tests**
   - Run a minimal smoke flow in the target environment:
     - Login → start a game → make a few moves → view game history.
   - For automated setups, run your CI smoke/e2e suite or a subset (see `tests/README.md`).

5. **User impact**
   - Support channels show error reports subsiding.
   - No new spikes appear after mitigation.

---

## 6. Post‑Incident Follow‑Up

After stabilizing, schedule a short follow-up within 24–48 hours:

- [ ] Capture a brief incident summary using `docs/incidents/POST_MORTEM_TEMPLATE.md`.
- [ ] Identify whether monitoring thresholds in `docs/ALERTING_THRESHOLDS.md` are appropriate.
- [ ] Add or refine tests around the affected routes (unit/integration/e2e) so similar regressions trip earlier.
- [ ] If dependencies were the cause, verify capacity / configuration against `docs/DEPLOYMENT_REQUIREMENTS.md` and `docs/OPERATIONS_DB.md`.

---

## 7. Related Documentation

- Incident docs
  - `docs/incidents/AVAILABILITY.md`
  - `docs/incidents/LATENCY.md`
  - `docs/incidents/RESOURCES.md`
  - `docs/incidents/TRIAGE_GUIDE.md`
- Runbooks
  - `docs/runbooks/DATABASE_DOWN.md`
  - `docs/runbooks/REDIS_DOWN.md`
  - `docs/runbooks/AI_SERVICE_DOWN.md`
  - `docs/runbooks/DEPLOYMENT_ROUTINE.md`
  - `docs/runbooks/DEPLOYMENT_ROLLBACK.md`
- Ops / config SSoT
  - `docs/ALERTING_THRESHOLDS.md`
  - `docs/DEPLOYMENT_REQUIREMENTS.md`
  - `docs/OPERATIONS_DB.md`
  - `docs/ENVIRONMENT_VARIABLES.md`
