# Resource Leak Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for investigating a high number of active handles or suspected Node.js resource leaks when `HighActiveHandles` alerts fire.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rules in `monitoring/prometheus/alerts.yml` (group `resources`, alert `HighActiveHandles`) and scrape configuration in `monitoring/prometheus/prometheus.yml`.
> - **Runtime metrics:** The `nodejs_active_handles_total` metric exported by the Node.js process and exposed on the `/metrics` endpoint.
> - **Service implementation & lifecycle:**
>   - Connection and session management in `src/server/websocket/server.ts`, `src/server/game/WebSocketInteractionHandler.ts`, `src/server/game/GameSession.ts`, `src/server/game/GameSessionManager.ts`.
>   - Database and cache connectivity in `src/server/database/connection.ts` and `src/server/cache/redis.ts`.
>   - Timer and background job usage in `src/server/middleware/*`, `src/server/services/**`, and any periodic jobs.
> - **Data lifecycle & retention:** `docs/DATA_LIFECYCLE_AND_PRIVACY.md`, `docs/OPERATIONS_DB.md` (how long we keep resources, and how they are cleaned up).
>
> **Precedence:**
>
> - The **Node.js runtime, service code, and connection management strategies** are authoritative for what counts as an active handle and how it should be managed.
> - `monitoring/prometheus/alerts.yml` is authoritative for the **thresholds** and **time windows** for `HighActiveHandles`.
> - This runbook describes **how to diagnose and fix** handle leaks. If it conflicts with code/config/alerts, **code + configs + `alerts.yml` win**, and this document should be updated.

---

## 1. When This Alert Fires

**Alert (from `monitoring/prometheus/alerts.yml`, `resources` group):**

- `HighActiveHandles` (warning)

**Conceptual behaviour (exact expression lives in `alerts.yml`):**

- `HighActiveHandles` fires when `nodejs_active_handles_total` stays above a configured threshold (e.g. > ~10,000 active handles) for a sustained period.
- Active handles include internal Node.js objects such as:
  - Open TCP sockets (HTTP, WebSocket, DB, Redis connections).
  - Timers and intervals.
  - File descriptors and other resource handles.

**Impact:**

- A very high handle count can indicate **leaks**:
  - WebSocket connections not being cleaned up.
  - DB or Redis connections not returned to pools.
  - Timers/intervals never cleared.
  - Streams or file descriptors left open.
- Left unchecked, this can lead to:
  - Resource exhaustion (e.g. hitting OS limits on file descriptors).
  - Increased memory usage (`HIGH_MEMORY.md`), GC pressure, and event loop lag (`EVENT_LOOP_LAG.md`).

---

## 2. Quick Triage (First 5–10 Minutes)

> Goal: Confirm that the high handle count is real, identify the **scope** (single vs many instances), and see whether it correlates with traffic or specific features.

### 2.1 Confirm which instances are affected

In Alertmanager / your monitoring UI:

- Identify which backend instances (pods/containers) have `HighActiveHandles` firing.
- Determine when the alert started and whether multiple instances exhibit similar behaviour.

In Prometheus:

```promql
# Active handles by instance
nodejs_active_handles_total

# Trend over the last 1–2 hours by instance
nodejs_active_handles_total{job="<your-backend-job>"}
```

Look for:

- A single instance with significantly more handles than peers → likely a localized leak or skewed traffic.
- All instances drifting upward together → systemic leak pattern or a global configuration change.

### 2.2 Correlate with traffic and other resource alerts

Check:

- **HTTP traffic and WebSockets:**
  - `sum(rate(http_requests_total[5m]))`
  - `ringrift_websocket_connections` from `MetricsService`.
- **Game sessions:**
  - `ringrift_games_active` and any session-related metrics.
- **Related alerts:**
  - `HighMemoryUsage` / `HighMemoryUsageCritical` → see `HIGH_MEMORY.md`.
  - `HighEventLoopLag` / `HighEventLoopLagCritical` → see `EVENT_LOOP_LAG.md`.

Key questions:

- Does the handle count rise with traffic and then **fall** when traffic subsides (suggesting capacity issues rather than leaks)?
- Or does it climb continuously even when traffic flat-lines (suggesting a leak)?

### 2.3 Check recent deployments and config changes

Establish if there were recent changes to:

- WebSocket server or connection handling logic.
- Database or Redis client configuration (pool sizes, timeouts, reconnection logic).
- Background jobs or schedulers that might start and never stop timers.

If a specific deployment aligns with the start of the leak, focus your investigation on code paths introduced in that change.

---

## 3. Deep Diagnosis

> Goal: Determine which **types of handles** are leaking or accumulating, and map them back to concrete code paths.

### 3.1 Identify handle types using Node diagnostics (staging or controlled env)

In a non-production environment that can reproduce the behaviour, use Node.js diagnostics to inspect active handles.

Typical approaches include:

- Using `node --inspect` and DevTools to inspect active handles and their stacks.
- Leveraging libraries like `why-is-node-running` or built-in diagnostics tools to print handle summaries.

Look for whether handles are predominantly:

- **Sockets** (network connections): HTTP, WebSocket, DB, Redis.
- **Timers/Intervals**.
- **File descriptors** or other exotic handles.

Even if you cannot run these tools in production, reproducing the pattern in staging with similar traffic can reveal the dominant handle type.

### 3.2 Map suspected handles to high-risk code paths

Based on handle type, map to likely sources:

- **Sockets / connections:**
  - HTTP/WebSocket: `src/server/websocket/server.ts`, `src/server/game/WebSocketInteractionHandler.ts`.
  - Database: `src/server/database/connection.ts` and any custom connection wrappers.
  - Redis: `src/server/cache/redis.ts`.
  - Third-party APIs: any custom HTTP clients or SDKs.
- **Timers / intervals:**
  - Recurring jobs in `src/server/services/**` or `src/server/middleware/**` that use `setInterval`, `setTimeout`, or scheduling libraries without teardown.
- **File descriptors / streams:**
  - Unbounded use of file I/O or streaming APIs without proper `close`/`end`.

Review recent changes in these areas for missing cleanup logic:

- Are WebSocket connections fully cleaned up on disconnect, error, and timeout?
- Are DB/Redis clients reused via pools and closed on process shutdown, but **not** per-request?
- Are timers cleared when no longer needed, especially in error or edge-case branches?

### 3.3 Compare handle growth with application state

Correlate `nodejs_active_handles_total` with:

- `ringrift_websocket_connections` — a mismatch (many handles but few active WebSocket connections) may indicate leaked sockets or timers.
- `ringrift_games_active` — a mismatch (few active games but many handles) suggests leakage in per-game resources.

Use `docs/OPERATIONS_DB.md` to inspect DB state for:

- Long-lived, apparently completed games still associated with live sessions.
- Orphaned game sessions that never cleaned up their connections.

### 3.4 Investigate per-feature diagnostics and debug tooling

Ensure that **diagnostic tools** and debugging utilities are not active in production in a way that keeps handles open:

- Long-lived debugging WebSocket subscriptions.
- Custom inspectors or trace streamers that attach and never detach.
- Simple scripts or tooling migrated from `scripts/` or `tests/utils/` into production code paths without proper teardown.

If such tools are present, confirm whether they:

- Have explicit lifecycles and termination conditions.
- Are guarded by feature flags or environment checks so that they don’t run in production unintentionally.

---

## 4. Remediation

> Goal: Close leaks by ensuring all handles are closed when no longer needed, and keep total handle counts within safe, predictable bounds.

### 4.1 Fix connection leaks (HTTP, WebSocket, DB, Redis)

1. **WebSocket connections:**
   - Review `src/server/websocket/server.ts` and `WebSocketInteractionHandler.ts` to ensure:
     - Connection close events (client disconnects, errors, heartbeats/timeouts) always trigger cleanup.
     - Game session teardown removes associations with WebSocket connections.
     - No additional listeners are attached repeatedly without removal.
2. **Database connections:**
   - Ensure `src/server/database/connection.ts` uses a connection pool rather than creating per-request connections.
   - Check for any code paths that create clients ad hoc and never close them.
3. **Redis connections:**
   - Inspect `src/server/cache/redis.ts` and rate limiting initialization (`src/server/middleware/rateLimiter.ts`) for correct reuse of shared clients.
   - Avoid creating new Redis clients for every request or background task.

### 4.2 Fix timer/interval leaks

For periodic jobs and timeouts:

- Audit `setInterval`, `setTimeout`, and any scheduler usage in `src/server/services/**`, `src/server/middleware/**`, and game/session lifecycles.
- Ensure that:
  - Intervals are cleared when their owning object or feature is shut down.
  - Timeouts are cleared when work completes earlier.
  - Re-scheduling does not create unbounded chains of timers.

Add tests (unit or integration) that simulate lifecycle transitions and assert that no unexpected timers remain.

### 4.3 Remove or gate problematic diagnostics

If diagnostics or debugging tools are responsible for handle growth:

- Disable them entirely in production environments, or
- Guard them behind explicit feature flags and ensure those flags are **off** outside of controlled experiments.

Ensure any experiment or debugging mode has a clear teardown path that closes connections and stops timers.

### 4.4 Consider capacity and configuration (secondary)

If, after fixing leaks, handle counts are still consistently high but stable, you may need to:

- Re-evaluate what a “reasonable” handle count is given the number of concurrent games and connections.
- Adjust alert thresholds in `alerts.yml` **only after** verifying that high handle counts are expected, controlled, and do not lead to memory or latency issues.

Threshold adjustments should be treated as configuration changes, tested in staging, and documented in `docs/ALERTING_THRESHOLDS.md`.

---

## 5. Validation

Before considering a resource-leak incident resolved, confirm:

### 5.1 Metrics and alerts

- [ ] `HighActiveHandles` has cleared and remains green for multiple evaluation windows.
- [ ] `nodejs_active_handles_total` has:
  - Returned to a stable baseline that scales sensibly with traffic and number of active games/WebSocket connections.
  - Stopped exhibiting monotonic, leak-like growth over time.
- [ ] Related alerts (`HighMemoryUsage*`, `HighEventLoopLag*`) are green or tracked via their own runbooks.

### 5.2 Behavioural checks

- [ ] Under representative traffic, the backend shows:
  - No evidence of resource exhaustion (no `EMFILE` or similar errors, no OOMs caused by runaway handles).
  - Stable memory usage and acceptable latency (see `HIGH_MEMORY.md`, `HIGH_LATENCY.md`).
- [ ] WebSocket and game behaviours are normal (connections close and clean up when players leave or games end).

### 5.3 Code and configuration

- [ ] Identified leaks in connection, timer, or diagnostic code have been fixed with appropriate tests.
- [ ] Any long-lived connections are now clearly owned by lifecycle-managed components (e.g. global DB pool, Redis client) and not created per-request.
- [ ] If alert thresholds were adjusted, the changes are reflected in `monitoring/prometheus/alerts.yml` and `docs/ALERTING_THRESHOLDS.md` with rationale.

---

## 6. Related Documentation & Runbooks

- **Monitoring & thresholds:**
  - `monitoring/prometheus/alerts.yml`
  - `monitoring/prometheus/prometheus.yml`
  - `monitoring/README.md`
  - `docs/ALERTING_THRESHOLDS.md`

- **Lifecycle & connection management:**
  - `src/server/websocket/server.ts`
  - `src/server/game/WebSocketInteractionHandler.ts`
  - `src/server/game/GameSession.ts`
  - `src/server/game/GameSessionManager.ts`
  - `src/server/database/connection.ts`
  - `src/server/cache/redis.ts`

- **Related runbooks:**
  - `HIGH_MEMORY.md` — for memory pressure resulting from leaks.
  - `EVENT_LOOP_LAG.md` — if leaks lead to event loop lag.
  - `GAME_HEALTH.md` — for long-running or zombie games that may drive handle growth.

- **Incidents & resources:**
  - `docs/incidents/RESOURCES.md`
  - `docs/incidents/AVAILABILITY.md`

Use this runbook as a **playbook** for diagnosing and fixing resource leaks. Always defer to the implementation, runtime configuration, and `alerts.yml` for the ground truth on resource usage expectations and thresholds.
