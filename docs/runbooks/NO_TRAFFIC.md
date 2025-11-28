# No HTTP Traffic Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for investigating the `NoHTTPTraffic` alert when HTTP request volume drops to zero.
>
> **SSoT alignment:** This runbook is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rule `NoHTTPTraffic` in `monitoring/prometheus/alerts.yml` (based on `http_requests_total`).
> - **Metrics surface:** HTTP metrics emitted by `metricsMiddleware` / `MetricsService` (`src/server/middleware/metricsMiddleware.ts`, `src/server/services/MetricsService.ts`) and exposed via `/metrics`.
> - **Health/readiness surface:** `HealthCheckService` and `ServiceStatusManager` (`src/server/services/HealthCheckService.ts`, `src/server/services/ServiceStatusManager.ts`) behind `/health` and `/ready`.
> - **Availability incident docs:** `docs/incidents/AVAILABILITY.md` (sections **NoHTTPTraffic**, **NoWebSocketConnections**, **NoActiveGames**) and deployment runbooks under `docs/runbooks/`.
>
> **Precedence:** Prometheus rules, deployment topology, ingress/load balancer configuration, and server code are authoritative for **what constitutes traffic** and **how reachability is defined**. If this runbook conflicts with them, **code + configs + tests win** and this document should be updated.

---

## 1. When This Alert Fires

**Alert:** `NoHTTPTraffic`  
**Group:** `availability` in `monitoring/prometheus/alerts.yml`

**Condition (conceptual PromQL):**

```promql
sum(rate(http_requests_total[5m])) == 0
```

**For:** 10m (see `alerts.yml` for exact duration and labels).

**Impact:**

- Could be **expected** (off-peak hours, non-production environment with no users).
- Could indicate that **no HTTP requests** are reaching the app:
  - DNS, load balancer, or ingress misconfiguration.
  - All app instances down or removed from rotation.
  - Network partition or firewall change.
  - Metrics ingestion issues (less common—check `/metrics`).

**Related alerts / signals:**

- `NoActiveGames` (info) – may be normal off-peak.
- `NoWebSocketConnections` (warning) – suggests real-time connectivity issues.
- `ServiceOffline` / `ServiceDegraded` – internal degradation mode via `ringrift_degradation_level`.

---

## 2. Quick Triage (First 5–10 Minutes)

Goal: Decide whether this is **benign zero traffic** or a **reachability incident**.

### 2.1 Check whether zero traffic is expected

1. **Time of day / environment**
   - Off-peak hours in low-usage environments (e.g. staging) may legitimately have no traffic.
   - In production, 10+ minutes of zero traffic during normal business/peak hours is suspicious.

2. **Planned maintenance / deployments**
   - Verify whether there is an ongoing maintenance window or deployment freeze.
   - Check your org’s change calendar / CI pipelines for releases that might have taken the service temporarily offline.

If this appears **expected**, you can:

- Acknowledge the alert and consider adjusting thresholds or silencing during known idle windows (see `docs/ALERTING_THRESHOLDS.md`).

If this is **not expected**, continue.

### 2.2 Check application health from within the cluster/host

From the app host:

```bash
# Check container status
docker compose ps app

# Check health endpoints locally
curl -s http://localhost:3000/health | jq
curl -s http://localhost:3000/ready | jq
```

Interpretation:

- If `/health` or `/ready` **fails** or returns `status: "unhealthy"`, this is likely an **app/service outage** → see `docs/incidents/AVAILABILITY.md` and `SERVICE_OFFLINE.md` / `SERVICE_DEGRADATION.md`.
- If health is **OK** but `NoHTTPTraffic` is firing, focus on **ingress / load balancer / DNS**.

### 2.3 Verify that metrics are being scraped

From the app host:

```bash
# Confirm that metrics are exposed
curl -s http://localhost:3000/metrics | head
```

- If `/metrics` is unreachable, treat this as an **app availability** issue.
- If `/metrics` is OK, but `NoHTTPTraffic` is firing, this likely means **Prometheus is scraping** but seeing zero `http_requests_total`, i.e. truly no incoming HTTP requests.

---

## 3. Deep Diagnosis

### 3.1 Are **any** HTTP requests reaching the app?

Use Prometheus to confirm:

```promql
# Overall HTTP request rate
sum(rate(http_requests_total[5m]))

# Requests by status (if any)
sum(rate(http_requests_total[5m])) by (status)

# Requests by path (if any)
sum(rate(http_requests_total[5m])) by (method, path)
```

- If these all evaluate to **0**, no HTTP requests are being recorded by `metricsMiddleware`.
- If some non-zero values exist, but the alert is still firing, inspect time windows and Prometheus evaluation/for durations.

### 3.2 Check ingress / load balancer / DNS

From **outside** the cluster (simulate a client):

```bash
# Replace with your public base URL
BASE_URL="https://your-ringrift-host"   # or http://... for non-TLS

# Health endpoint from outside
curl -v "$BASE_URL/health"

# Basic app page
curl -v "$BASE_URL/" | head
```

Look for:

- DNS resolution errors.
- TLS certificate issues.
- `503`/`502` responses from a fronting proxy (nginx, cloud LB, API gateway).

On the host / ingress node (if applicable):

```bash
# Check any reverse proxy / nginx logs (if used)
docker compose logs --tail 200 nginx 2>&1 || true

# Confirm app port is listening
docker compose exec app sh -c "netstat -tlnp | grep 3000 || ss -tlnp | grep 3000" || true
```

Common issues:

- DNS records updated to the wrong IP.
- Load balancer health checks failing and removing all backends from rotation.
- Ingress rules misconfigured (wrong host/path, missing TLS cert, etc.).

### 3.3 Cross-check higher-level activity signals

Even if HTTP traffic is zero, you may see other metrics:

- `ringrift_websocket_connections` – if non-zero while HTTP is zero, it may indicate:
  - Long-lived WebSocket connections still active while **new HTTP connections** are blocked.
  - See `WEBSOCKET_ISSUES.md` / `WEBSOCKET_SCALING.md` and `docs/incidents/AVAILABILITY.md` (NoWebSocketConnections).

- `ringrift_games_active` – if non-zero, some games may still be active, but HTTP traffic for new sessions may be blocked.

Use these to differentiate **"platform entirely down"** from **"new HTTP sessions blocked"**.

---

## 4. Remediation Playbooks

> **Note:** For production, coordinate with the on-call lead and follow your organization’s change-management procedures.

### 4.1 If the application is unhealthy or offline

If `/health` or `/ready` fails, or the app container is not running:

1. **Check app container status and logs**

   ```bash
   docker compose ps app
   docker compose logs --tail 200 app
   ```

   - Look for crashes, startup exceptions, or port binding errors.

2. **Restart the app container** (non-destructive)

   ```bash
   cd /opt/ringrift   # or your deployment root
   docker compose restart app
   ```

3. If a recent deployment caused the outage, follow **deployment runbooks**:
   - `DEPLOYMENT_ROUTINE.md` for normal deploy flow.
   - `DEPLOYMENT_ROLLBACK.md` if you need to roll back to a known good version.

4. After restart/rollback, verify:
   - `/health` and `/ready` return healthy.
   - `http_requests_total` begins to rise again (see validation below).

### 4.2 If ingress / load balancer / DNS is misconfigured

1. **DNS**
   - Verify that your public hostname resolves to the correct load balancer / ingress IP.
   - Check for recent changes in DNS records.

2. **Load balancer / ingress**
   - Ensure health checks target the correct path (often `/health` or `/ready`).
   - Confirm backend pool configuration matches the active app instances.
   - Review TLS configuration if errors indicate certificate/handshake failures.

3. **Reverse proxy (if using nginx or similar)**
   - Confirm that the proxy forwards to the correct upstream (`localhost:3000` or your app service name).
   - Fix any misrouted paths or host-based rules.

4. Once corrected, re-test from outside:

   ```bash
   curl -v "$BASE_URL/health"
   curl -v "$BASE_URL/" | head
   ```

### 4.3 If this is expected but noisy

If you confirm that **zero traffic is acceptable** (e.g. staging environment at night):

1. Document the scenario in your incident tracker and this runbook’s **TODO** section.
2. Consider adjusting alert thresholds or routes:
   - Tuning `for:` duration in `alerts.yml` for `NoHTTPTraffic`.
   - Restricting the alert to specific environments (e.g. production-only using label filters).
   - Adding a maintenance/quiet-hours silence.
3. Update `docs/ALERTING_THRESHOLDS.md` to record the rationale for any changes.

---

## 5. Validation

You are done when **all** of the following hold:

1. **HTTP metrics show non-zero traffic**

   In Prometheus:

   ```promql
   sum(rate(http_requests_total[5m]))
   ```

   - This should be **> 0** during known active periods.
   - If appropriate, confirm by breakdown:

     ```promql
     sum(rate(http_requests_total[5m])) by (status)
     sum(rate(http_requests_total[5m])) by (method, path)
     ```

2. **Alerts**
   - `NoHTTPTraffic` has cleared in Alertmanager and remains green over at least one alert window.
   - Any correlated availability alerts (`ServiceOffline`, `NoWebSocketConnections`, etc.) are resolved.

3. **Health / readiness and external reachability**

   ```bash
   # Inside the cluster/host
   curl -s http://localhost:3000/health | jq
   curl -s http://localhost:3000/ready | jq

   # From external client
   curl -v "$BASE_URL/health"
   curl -v "$BASE_URL/" | head
   ```

   - Both internal and external checks succeed with expected responses.

4. **User visibility**
   - Synthetic probes (if configured) succeed.
   - No new user reports of reachability issues.

---

## 6. Post‑Incident Follow‑Up

- [ ] If this was a genuine outage, complete a brief write-up using `docs/incidents/POST_MORTEM_TEMPLATE.md`.
- [ ] If this was an **expected idle** period, decide whether to:
  - Tune thresholds for `NoHTTPTraffic` in `monitoring/prometheus/alerts.yml`.
  - Add environment-specific silences or schedules.
- [ ] If ingress/DNS was the cause, document the exact misconfiguration and fix in your infra runbooks.
- [ ] If app unavailability was the cause, verify deployment and health check configuration in `docs/DEPLOYMENT_REQUIREMENTS.md`.

---

## 7. Related Documentation

- Incident docs
  - `docs/incidents/AVAILABILITY.md`
  - `docs/incidents/LATENCY.md`
  - `docs/incidents/RESOURCES.md`
  - `docs/incidents/TRIAGE_GUIDE.md`
- Runbooks
  - `docs/runbooks/SERVICE_OFFLINE.md`
  - `docs/runbooks/SERVICE_DEGRADATION.md`
  - `docs/runbooks/DEPLOYMENT_INITIAL.md`
  - `docs/runbooks/DEPLOYMENT_ROUTINE.md`
  - `docs/runbooks/DEPLOYMENT_ROLLBACK.md`
- Ops / config SSoT
  - `docs/ALERTING_THRESHOLDS.md`
  - `docs/DEPLOYMENT_REQUIREMENTS.md`
  - `docs/OPERATIONS_DB.md`
  - `docs/ENVIRONMENT_VARIABLES.md`
