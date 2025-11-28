# High Latency Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for diagnosing and reducing elevated request/game latency when HTTP or move-latency alerts fire.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - Prometheus alert rules in `monitoring/prometheus/alerts.yml` (e.g. `HighP99Latency`, `HighP99LatencyCritical`, `HighP95Latency`, `HighMedianLatency`, `HighGameMoveLatency`).
> - Prometheus configuration in `monitoring/prometheus/prometheus.yml`.
> - HTTP and game-move metrics exposed by the backend (via `metricsMiddleware` and related instrumentation).
> - Resource and dependency alerts (memory, event loop lag, DB/Redis response time) defined alongside latency alerts.
>
> **Precedence:** Thresholds, bucket definitions, and metric names are defined in **monitoring configs + backend code** and may evolve over time. This runbook describes how to investigate and mitigate; it is **not** a source of truth for alert thresholds or semantics.

---

## 1. When These Alerts Fire

Latency-related alerts defined in `monitoring/prometheus/alerts.yml` (HTTP-level):

- `HighP99Latency` (warning)
  - Uses `histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) > 2` for 5 minutes.
- `HighP99LatencyCritical` (critical)
  - Same metric, threshold ~5s over a shorter window.
- `HighP95Latency` (warning)
  - `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) > 1` for 10 minutes.
- `HighMedianLatency` (warning)
  - `histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) > 0.5` for 15 minutes.

Game-specific latency alert:

- `HighGameMoveLatency` (warning)
  - `histogram_quantile(0.99, sum(rate(ringrift_game_move_latency_seconds_bucket[5m])) by (le, board_type)) > 1` for 5 minutes.
  - Indicates slow move processing for affected `board_type`.

When these fire, assume:

- The metrics pipeline (`metricsMiddleware` and game move metrics) is functioning and has determined that recent latency percentiles cross configured thresholds.
- Latency elevation may be due to:
  - Dependency slowness (DB/Redis/AI).
  - Resource pressure (CPU, memory, event loop lag).
  - Application-level hotspots or regressions.

---

## 2. Triage

> Goal: Localise **where** latency is coming from (which endpoints/board types and which layer: network, app, DB/Redis/AI, or Node.js runtime) before changing anything.

### 2.1 Confirm which alert(s) are active

In Alertmanager / UI:

- Identify which of `HighP99Latency`, `HighP99LatencyCritical`, `HighP95Latency`, `HighMedianLatency`, `HighGameMoveLatency` is firing.
- Note severity, duration, and whether this is isolated to a single environment.

In Prometheus, run (adjust windows to match `alerts.yml` if needed):

```promql
# Overall HTTP latency profiles
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# Game move latency by board type
histogram_quantile(0.99, sum(rate(ringrift_game_move_latency_seconds_bucket[5m])) by (le, board_type))
```

Confirm that the quantiles line up with the alert summary values.

### 2.2 Identify hot endpoints and operations

Break down HTTP latency by attributes where available:

```promql
# P99 latency by endpoint (if `path` or `route` label is present)
histogram_quantile(0.99,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le, path)
)

# P99 latency by method
histogram_quantile(0.99,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le, method)
)

# Optionally: tail latency for 5xx-only (to detect slow failures)
histogram_quantile(0.99,
  sum(rate(http_request_duration_seconds_bucket{status=~"5.."}[5m])) by (le, path)
)
```

For game move latency:

```promql
# High move latency by board type
histogram_quantile(0.99,
  sum(rate(ringrift_game_move_latency_seconds_bucket[5m])) by (le, board_type)
)
```

Use these to answer:

- Is latency elevated **globally**, or only for specific routes (e.g. matchmaking, game history, AI requests)?
- Is latency elevated for **all board types** or only particular ones?
- Are slow operations also returning 5xx (see `HIGH_ERROR_RATE.md`) or “just” slow 2xx?

### 2.3 Correlate with dependency, resource, and error alerts

Check for alerts in other groups that might explain high latency:

- **Dependencies / service response:**
  - `DatabaseResponseTimeSlow`, `RedisResponseTimeSlow`  
    → See `DATABASE_PERFORMANCE.md`, `REDIS_PERFORMANCE.md`.
- **Availability / degradation:**
  - `ServiceDegraded`, `ServiceMinimalMode`, `ServiceOffline`  
    → See `SERVICE_DEGRADATION.md`, `SERVICE_OFFLINE.md`.
- **Error rates:**
  - `HighErrorRate`, `ElevatedErrorRate`  
    → See `HIGH_ERROR_RATE.md`.
- **Resources:**
  - `HighMemoryUsage`, `HighMemoryUsageCritical` → `HIGH_MEMORY.md`.
  - `HighEventLoopLag`, `HighEventLoopLagCritical` → `EVENT_LOOP_LAG.md`.
  - `HighActiveHandles` → `RESOURCE_LEAK.md`.

If any of these are firing, treat them as **candidate root causes** of latency and pivot to those runbooks for deeper remediation.

### 2.4 Check health/readiness and basic behaviour

From an operator workstation against `APP_BASE` (per environment):

```bash
# Overall health (liveness)
curl -sS APP_BASE/health | jq . || curl -sS APP_BASE/health

# Readiness, including dependency checks
curl -sS APP_BASE/ready | jq . || curl -sS APP_BASE/ready
```

- If `/ready` shows **database** or **redis** degraded/unhealthy, prioritise those runbooks.
- If `/ready` is OK but latency remains high, suspect:
  - Application hotspots (expensive endpoints, N+1 queries, heavy CPU work).
  - Node.js event loop lag or memory pressure.

### 2.5 Inspect logs around hot endpoints

On the host where the stack is running (docker-compose by default):

```bash
cd /path/to/ringrift

# Tail app logs
docker compose logs app --tail=300 | sed -n '1,200p'
```

Look for:

- Slow request logging, timeouts, or upstream timeouts (DB, Redis, AI).
- Repeated warnings about degradation or resource issues.
- Application exceptions or long-running operations aligning with hot endpoints.

If game move latency is high, also correlate with logs around game session handling and AI interactions.

---

## 3. Remediation (High Level)

> Goal: Reduce latency back under thresholds **without** blindly changing alert configs; focus on root-cause fixes.

### 3.1 If dependency latency is the primary driver

When `DatabaseResponseTimeSlow` or `RedisResponseTimeSlow` are firing or `/ready` shows slow dependencies:

- **Database-focused remediation:**
  - Follow `DATABASE_PERFORMANCE.md` and `DATABASE_DOWN.md` as needed.
  - Investigate slow queries, lock contention, or insufficient resources.
  - Verify that connection pool sizes and timeouts are appropriate for current load.

- **Redis-focused remediation:**
  - Follow `REDIS_PERFORMANCE.md` and `REDIS_DOWN.md`.
  - Look for key-level hotspots, large values, or network issues to the cache.

- **AI-related delays:**
  - If AI requests dominate the slow endpoints or `HighGameMoveLatency` correlates with AI-heavy board types, cross-check AI runbooks (`AI_SERVICE_DOWN.md`, `AI_PERFORMANCE.md`, `AI_FALLBACK.md`).

### 3.2 If resource pressure / runtime issues are involved

When event loop lag, memory, or handle-count alerts are active:

- Follow `EVENT_LOOP_LAG.md`, `HIGH_MEMORY.md`, and `RESOURCE_LEAK.md` to:
  - Identify blocking operations on the main thread.
  - Address memory leaks or unbounded data structures.
  - Reduce the number of long-lived handles (e.g. open sockets, file descriptors).

Latency may improve automatically once the Node.js runtime is no longer saturated.

### 3.3 If latency is localised to specific endpoints or operations

When high latency is limited to a subset of routes or game operations and dependencies/resources look healthy:

- **Profile the hot endpoints** (using logs + code inspection):
  - Identify any expensive loops, large payload processing, or synchronous work in those request handlers.
  - Check whether queries for those endpoints are missing indexes or doing more work than expected.

- **Reduce work on the critical path:**
  - Move heavy, non-essential work to background jobs where feasible.
  - Introduce or tighten caching (within the existing caching framework).
  - Avoid per-request recomputation of expensive data.

- **Consider load/rate adjustments:**
  - If legitimate traffic has spiked, coordinate with product/ops to:
    - Apply or adjust rate limits (using existing `rateLimiter` behaviour).
    - Scale the application/database/cache tiers according to your deployment runbooks.

Changes to code, cache policies, or deployment topology should be implemented via normal development + review + CI pipelines; do not try to encode them in this runbook.

### 3.4 Coordinate with availability runbooks

If latency is so severe that it is effectively causing **outage-level** symptoms (e.g. timeouts, users unable to play):

- Treat in tandem with:
  - `SERVICE_DEGRADATION.md` (degraded/minimal modes).
  - `SERVICE_OFFLINE.md` if the system tips into offline mode.

---

## 4. Validation

Before considering the incident resolved:

- **Metrics / alerts:**
  - [ ] Relevant HTTP latency alerts have cleared in Alertmanager and stayed clear for at least one full evaluation window.
  - [ ] P50/P95/P99 HTTP latencies have returned to expected baselines for your environment and traffic levels.
  - [ ] `HighGameMoveLatency` (if it fired) is no longer active, and per-`board_type` move latencies are acceptable.

- **Health and dependencies:**
  - [ ] `/health` and `/ready` report healthy statuses for all dependencies.
  - [ ] No concurrent dependency or resource alerts (DB/Redis response time, event loop lag, memory) remain active.

- **User experience:**
  - [ ] Manual testing of key flows (login → lobby → start game → play moves) feels responsive and does not exhibit timeouts or long stalls.
  - [ ] Any e2e or smoke tests that cover hot endpoints or high-traffic paths pass reliably.

---

## 5. TODO / Environment-Specific Notes

Populate these per environment (staging, production, etc.) and keep them updated:

- [ ] Links to dashboards that show HTTP and game-move latency alongside error rates, dependency metrics, and resource usage.
- [ ] Typical baseline values for P50/P95/P99 latency during normal operation (per environment and time-of-day, high level is fine).
- [ ] Known high-cost endpoints or operations that are safe to temporarily throttle or offload in emergencies.
- [ ] Any environment-specific tuning levers (worker counts, connection pools, cache TTLs) and where they are configured (in code / env / infra), to be used via normal change management processes rather than ad-hoc edits.
