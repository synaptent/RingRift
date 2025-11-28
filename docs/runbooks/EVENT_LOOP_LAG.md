# Event Loop Lag Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for investigating and mitigating high Node.js event loop lag alerts.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rules in `monitoring/prometheus/alerts.yml` (group `resources`, alerts `HighEventLoopLag`, `HighEventLoopLagCritical`) and scrape configuration in `monitoring/prometheus/prometheus.yml`.
> - **Runtime metrics:** The `nodejs_eventloop_lag_seconds` metric exported by the Node.js process (via `prom-client` / runtime observers) and exposed on the `/metrics` endpoint.
> - **Service implementation & request lifecycle:**
>   - HTTP middleware and routing in `src/server/middleware/*` and `src/server/routes/**`.
>   - Game orchestration and state machines in `src/server/game/**` and `src/shared/stateMachines/**`.
>   - WebSocket server and handlers in `src/server/websocket/server.ts` and `src/server/game/WebSocketInteractionHandler.ts`.
>   - Database and cache clients in `src/server/database/connection.ts` and `src/server/cache/redis.ts`.
> - **Performance & resource docs:** `monitoring/README.md`, `docs/ALERTING_THRESHOLDS.md`, and related incident docs under `docs/incidents/LATENCY.md` and `docs/incidents/RESOURCES.md`.
>
> **Precedence:**
>
> - The **service code, runtime configuration, and deployment manifests** are authoritative for how work is scheduled and where blocking can occur.
> - `monitoring/prometheus/alerts.yml` is authoritative for **alert names, thresholds, and PromQL expressions** for event loop lag.
> - This runbook explains **how to interpret and act on** those alerts. If it conflicts with code/config/alerts, **code + configs + `alerts.yml` win**, and this document should be updated.

---

## 1. When These Alerts Fire

**Alerts (from `monitoring/prometheus/alerts.yml`, `resources` group):**

- `HighEventLoopLag` (warning)
- `HighEventLoopLagCritical` (critical)

**Conceptual behaviour (exact thresholds live in `alerts.yml`):**

- `nodejs_eventloop_lag_seconds` measures how far the Node.js event loop is **behind schedule** (i.e. how long the main thread is blocked).
- `HighEventLoopLag` fires when lag exceeds a **warning threshold** (e.g. ~100ms) for a sustained period.
- `HighEventLoopLagCritical` fires when lag exceeds a **critical threshold** (e.g. ~500ms) for a shorter window, indicating the process is effectively not keeping up with asynchronous work.

**Impact:**

- All asynchronous operations (HTTP requests, WebSocket message handling, timers, callbacks) are delayed.
- Users may experience:
  - Slow responses or timeouts on HTTP endpoints.
  - Delayed or stalled WebSocket updates and game moves.
  - Intermittent errors if upstream timeouts fire while the event loop is blocked.

Event loop lag is a **symptom** of blocked CPU (synchronous work) or overloaded process, not a root cause by itself.

---

## 2. Quick Triage (First 5–10 Minutes)

> Goal: Confirm that event loop lag is genuinely elevated, identify which instances are affected, and see whether the pattern correlates with CPU, memory, or specific workloads.

### 2.1 Identify affected instances and magnitude

In Alertmanager / your monitoring UI:

- Note which instances (pods/containers) are impacted by `HighEventLoopLag*` alerts.
- Check how long the alerts have been firing and whether they’re flapping.

In Prometheus:

```promql
# Current event loop lag by instance
nodejs_eventloop_lag_seconds

# Recent event loop lag trend for the last 15 minutes
nodejs_eventloop_lag_seconds{job="<your-backend-job>"}
```

Look for:

- **Single hot instance** with much higher lag than peers → likely skewed load or a localised issue.
- **All instances** showing elevated lag → systemic blocking work or under-provisioned CPU.

### 2.2 Correlate with latency, errors, and resource usage

Check other key metrics:

- **HTTP latency and errors:**
  - Alerts: `HighP99Latency`, `HighP95Latency`, `HighMedianLatency`, `HighErrorRate`, `ElevatedErrorRate`.
  - Runbooks: `HIGH_LATENCY.md`, `HIGH_ERROR_RATE.md`.
- **Resource alerts:**
  - `HighMemoryUsage` / `HighMemoryUsageCritical` — see `HIGH_MEMORY.md`.
  - `HighActiveHandles` — see `RESOURCE_LEAK.md`.

If event loop lag, HTTP latency, and error rate are all elevated, the node is likely CPU-bound or doing too much synchronous work.

### 2.3 Check recent deployments and configuration changes

Determine whether anything changed shortly before the lag alerts started:

- New backend release (especially touching:
  - Request handlers in `src/server/routes/**`.
  - Game orchestration and heavy rules/AI integrations in `src/server/game/**`.
  - Logging, tracing, or instrumentation code that might now run synchronously on the request path.)
- Changes in **traffic patterns** (launches, campaigns, new integrations).
- Configuration changes that reduce CPU limits or co-locate additional workloads on the same nodes.

If a deployment is strongly correlated with the onset of lag, prioritize inspecting the code paths introduced in that release.

### 2.4 Confirm overall health

- Check `/health` and `/ready` for the backend service.
- In cluster dashboards, look at **CPU utilization** for affected pods or nodes.
  - High CPU utilization aligned with lag suggests pure load/CPU bottleneck.
  - Moderate CPU with high lag suggests long synchronous or blocking operations (e.g. big JSON work, crypto, compression, or blocking I/O).

---

## 3. Deep Diagnosis

> Goal: Identify **where** the main thread is getting blocked: CPU-bound code, large synchronous operations, blocking library calls, or pathological workloads.

### 3.1 Characterize the pattern (steady vs bursty)

In Prometheus:

- Examine `nodejs_eventloop_lag_seconds` over a longer window (1–6 hours).
  - **Steady high lag**: suggests a persistent heavy loop or continuous CPU load.
  - **Periodic spikes**: may align with scheduled tasks, batch jobs, GC cycles, or heavy endpoints.

Attempt to line up lag spikes with:

- Cron-like or scheduled jobs.
- Known maintenance tasks (e.g. cleanup, migrations, backfills).
- Spikes in particular endpoint traffic (via `http_requests_total` broken down by path).

### 3.2 Inspect logs around lag windows

Using application logs (for example via docker-compose):

```bash
# Look at logs around the time of lag alerts
docker compose logs app --since=15m 2>&1 \
  | sed -n '1,200p'
```

Look for:

- Long-running operations logged with start/end timestamps.
- Warnings or errors from DB/Redis that might cause blocking retries.
- Heavy debug logging or synchronous console/file logging added recently.

### 3.3 Check for known sources of blocking work

Common sources of event loop blocking in a service like RingRift include:

- **CPU-heavy work on the request path:**
  - Complex rules evaluations or simulations performed synchronously in Node.js instead of being delegated appropriately to the shared engine/orchestrator or AI service.
  - Large in-memory transformations (deep cloning, serialisation/deserialisation of big objects).
- **Blocking or slow I/O operations:**
  - Using synchronous filesystem calls (`fs.*Sync`) on hot paths.
  - Misconfigured or blocking network libraries that don’t yield control to the event loop.
- **Excessive synchronous logging:**
  - Very verbose logging executed synchronously for every request or move, especially if writing to slow storage.
- **Pathological loops or retries:**
  - Tight retry loops in request handlers or background jobs that spin without yielding.

For any suspicious code paths introduced recently, review whether they:

- Use asynchronous APIs correctly.
- Avoid CPU-heavy work on latency-sensitive paths (consider delegating to worker processes or offloading to the AI service where appropriate).
- Use throttled/batched patterns for high-frequency operations.

### 3.4 Use profiling in a safe environment

In a staging or load-test environment that can reproduce the issue:

1. Run Node with `--inspect` and attach a profiler (Chrome DevTools, VS Code) to:
   - Capture **CPU profiles** during periods of artificially induced load.
   - Identify functions with high self or total time.
2. Use tools like `clinic` or `0x` (if allowed) to produce flamegraphs that show blocking hot paths.

Focus on:

- Functions that appear at the top of stacks where lag is worst.
- Code within `src/server/game/**`, `src/server/routes/**`, and heavy library wrappers.

### 3.5 Consider interaction with memory and handles

If `HighMemoryUsage` and/or `HighActiveHandles` are also firing:

- High memory can amplify lag via GC pauses.
- A large number of active handles (open sockets, timers, DB connections) can increase work per tick.

Use `HIGH_MEMORY.md` and `RESOURCE_LEAK.md` to investigate whether leaks or excessive handle counts are part of the problem.

---

## 4. Remediation

> Goal: Reduce event loop lag by removing/isolating blocking work and ensuring the process has sufficient CPU for expected workloads.

### 4.1 Short-term mitigations

1. **Scale out and/or up:**
   - Temporarily increase the number of backend instances and/or CPU resources, within cluster capacity and normal change management.
   - This does not fix blocking code, but can reduce per-instance load.
2. **Throttle heavy workloads:**
   - Use rate limiting (see `RATE_LIMITING.md`) or feature flags to reduce the frequency of heavy endpoints or debug/diagnostic flows.
   - Consider temporarily disabling non-essential batch jobs or debug endpoints.

These should be accompanied by deeper fixes below; do not rely on scaling alone if blocking work remains on the main thread.

### 4.2 Remove or offload blocking work

For code paths identified as problematic:

- **Move CPU-bound work off the request path:**
  - Where feasible, push heavy computations to background workers or the AI service, and expose them via asynchronous APIs.
  - Use queues or job systems rather than doing large computations inline.
- **Use asynchronous APIs for I/O:**
  - Replace `*Sync` filesystem or network calls in hot paths with asynchronous equivalents.
  - Ensure callbacks/promises are used correctly so events can interleave.
- **Tighten logging and diagnostics:**
  - Reduce log volume on hot paths, especially for info/debug-level logs.
  - Avoid synchronous logging libraries that block on I/O.

All such changes should preserve **rules semantics and game correctness**; when refactoring, ensure invariants and tests remain green.

### 4.3 Optimise hot paths in game and WebSocket logic

If profiling shows that game orchestration or WebSocket handling is hot:

- Review `GameSession`, `GameSessionManager`, and `WebSocketInteractionHandler` for any heavy per-message processing.
- Ensure that:
  - Game state transitions are efficient and do not repeatedly recompute expensive derived state.
  - WebSocket broadcasts are batched or carefully filtered when possible.
  - Expensive validations or computations are cached safely or performed ahead of time where appropriate.

Coordinate with rules/engine owners to ensure any optimisations remain SSoT-compliant with the shared engine semantics.

### 4.4 Adjust deployment and capacity when necessary

Once blocking code is removed or reduced, you may still need to adjust capacity:

- Increase CPU requests/limits for the backend in line with observed CPU usage under expected peak load.
- Ensure auto-scaling policies (if any) respond sensibly to CPU load and lag-induced latency.

Any capacity change should follow normal deployment and performance testing procedures.

---

## 5. Validation

Before considering an event-loop-lag incident resolved, confirm:

### 5.1 Metrics and alerts

- [ ] `HighEventLoopLag` and `HighEventLoopLagCritical` (if they fired) have cleared and stayed green across multiple evaluation windows.
- [ ] `nodejs_eventloop_lag_seconds` has returned to low, stable values under representative load.
- [ ] HTTP latency (`HighP99Latency`, `HighP95Latency`, `HighMedianLatency`) and error-rate alerts are green or tracked with their respective runbooks.

### 5.2 Behavioural checks

- [ ] Under normal and peak load tests, the service:
  - Responds promptly to HTTP and WebSocket events without noticeable stalls.
  - Maintains acceptable latency for game moves and critical endpoints (see `GAME_PERFORMANCE.md`).
- [ ] Users no longer report sporadic stalls or timeouts attributable to backend slowness.

### 5.3 Code and configuration

- [ ] Identified blocking hot paths have been refactored or moved off the main thread, and relevant tests remain green.
- [ ] Any logging/diagnostic changes maintain sufficient observability without reintroducing blocking behaviour.
- [ ] If CPU or resource limits were adjusted, they are recorded in capacity planning docs and verified in staging/production.

---

## 6. Related Documentation & Runbooks

- **Monitoring & thresholds:**
  - `monitoring/prometheus/alerts.yml`
  - `monitoring/prometheus/prometheus.yml`
  - `monitoring/README.md`
  - `docs/ALERTING_THRESHOLDS.md`

- **Resources & performance:**
  - `HIGH_MEMORY.md` — for memory pressure that can exacerbate lag.
  - `RESOURCE_LEAK.md` — for high active handles (`HighActiveHandles`).
  - `HIGH_LATENCY.md`, `GAME_PERFORMANCE.md` — user-facing latency and move performance.

- **Game lifecycle & state machines:**
  - `src/server/game/GameSession.ts`
  - `src/server/game/GameSessionManager.ts`
  - `src/server/game/WebSocketInteractionHandler.ts`
  - `src/shared/stateMachines/gameSession.ts`, `connection.ts`, `aiRequest.ts`
  - `GAME_HEALTH.md` — stalled or zombie games.

- **Incidents & resources:**
  - `docs/incidents/LATENCY.md`
  - `docs/incidents/RESOURCES.md`

Use this runbook as a **playbook** for diagnosing and remediating event loop lag. Always defer to the implementation, deployment configuration, and `alerts.yml` for the exact semantics and thresholds of event loop lag in each environment.
