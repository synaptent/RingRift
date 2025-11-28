# High Memory Usage Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for diagnosing and mitigating high process memory usage when `HighMemoryUsage` alerts fire.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rules in `monitoring/prometheus/alerts.yml` (group `resources`, alerts `HighMemoryUsage`, `HighMemoryUsageCritical`) and scrape configuration in `monitoring/prometheus/prometheus.yml`.
> - **Metrics & instrumentation:** The `process_resident_memory_bytes` metric exported by the Node.js process via `prom-client` and exposed through the server’s `/metrics` endpoint (`src/server/services/MetricsService.ts`).
> - **Service implementation & lifecycle:**
>   - Game/session lifecycle and retention in `src/server/game/GameSession.ts`, `src/server/game/GameSessionManager.ts`, `src/server/services/GamePersistenceService.ts`, `src/server/services/DataRetentionService.ts`.
>   - WebSocket connection handling in `src/server/websocket/server.ts` and `src/server/game/WebSocketInteractionHandler.ts`.
>   - Caching / connection management in `src/server/database/connection.ts`, `src/server/cache/redis.ts`, and other long-lived services.
> - **Data & privacy SSoT:** `docs/DATA_LIFECYCLE_AND_PRIVACY.md` (what we store, for how long, and cleanup expectations).
> - **AI & training memory (where applicable):** Python-side memory configuration helpers in `ai-service/app/utils/memory_config.py` and tests in `ai-service/tests/test_memory_config.py` inform how training/AI jobs bound memory; Node alerts for the main backend should still defer to `alerts.yml` for exact scopes.
>
> **Precedence:**
>
> - Runtime behaviour and limits are defined by **code, container/resource configuration, and deployment manifests**.
> - `monitoring/prometheus/alerts.yml` is authoritative for alert expressions and thresholds (e.g. GiB limits, evaluation windows).
> - This runbook explains **how to investigate and mitigate** memory pressure. If it conflicts with code/config/alerts, **code + configs + `alerts.yml` win** and this document should be updated.

---

## 1. When These Alerts Fire

**Alerts (from `monitoring/prometheus/alerts.yml`, `resources` group):**

- `HighMemoryUsage` (warning)
- `HighMemoryUsageCritical` (critical)

**Conceptual behaviour (exact thresholds live in `alerts.yml`):**

- Both alerts watch `process_resident_memory_bytes` for the Node.js process.
- `HighMemoryUsage` fires when resident memory exceeds a **warning threshold** for a sustained period (e.g. around 1.5 GiB).
- `HighMemoryUsageCritical` fires when resident memory exceeds a **critical threshold** (e.g. around 2 GiB), indicating high risk of out-of-memory (OOM) kills from the container runtime or host.

**Impact:**

- Elevated memory usage reduces headroom for spikes and garbage collection.
- If unaddressed, the process may be killed by the runtime (Kubernetes / docker) or the OS, causing:
  - Dropped WebSocket connections and in-flight games being interrupted.
  - Increased latency and error rates before the crash.
  - Potential data loss if in-memory state isn’t durably persisted.

---

## 2. Quick Triage (First 5–10 Minutes)

> Goal: Confirm that the alert reflects a real problem (not a transient spike), identify the affected instances, and decide whether you’re looking at **leak / runaway growth** vs **legitimate high load**.

### 2.1 Identify which service instance is affected

In Alertmanager / your monitoring UI, note for each firing alert:

- **Service / job / instance labels** (e.g. `instance`, `pod`, `container`).
- Whether alerts affect **all** instances of the service or only a subset.

In Prometheus:

```promql
# Current memory usage by instance
process_resident_memory_bytes

# Recent trend over the last hour by instance
process_resident_memory_bytes{job="<your-backend-job>"}
```

Look for:

- One instance steadily climbing while others are stable → likely **leak or skewed load**.
- All instances trending up in parallel → likely **overall workload increase**, a configuration shift, or heavier per-request memory.

### 2.2 Correlate with load and other alerts

- Check **HTTP traffic and latency**:
  - `sum(rate(http_requests_total[5m]))`
  - Latency alerts (`HighP99Latency`, `HighP95Latency`, `HighMedianLatency` — see `HIGH_LATENCY.md`).
- Check **game and WebSocket activity**:
  - `ringrift_games_active`, `ringrift_websocket_connections` from `MetricsService`.
- Check **resource alerts**:
  - `HighEventLoopLag` / `HighEventLoopLagCritical` (see `EVENT_LOOP_LAG.md`).
  - `HighActiveHandles` (see `RESOURCE_LEAK.md`).

If memory is high **and**:

- Event loop lag and latency are high → memory pressure may be causing GC thrash or swapping.
- Active handles are high → unclosed connections or leaks may be holding memory.
- Load is unusually high → memory usage may be justified but the deployment may need scaling.

### 2.3 Check recent deployments and configuration changes

Confirm whether, shortly before the alert began:

- A new backend version was deployed (changes to `GameSession`, `GameSessionManager`, `GamePersistenceService`, `DataRetentionService`, WebSocket handling, caching, etc.).
- Data retention or session settings were changed (e.g. keeping more games in memory or in the DB for longer).
- Container memory limits or Kubernetes resource requests/limits were adjusted.

If a suspect deployment exists, prioritize understanding its impact on memory (see **3.3**).

### 2.4 Confirm system health at a high level

- Check `/health` and `/ready` endpoints for the backend service; if they’re failing, also follow `SERVICE_DEGRADATION.md` / `SERVICE_OFFLINE.md` as needed.
- In the orchestrator/cluster dashboard (Kubernetes, docker, etc.), confirm whether pods are being **restarted due to OOM kills**.

---

## 3. Deep Diagnosis

> Goal: Determine whether you are dealing with a **memory leak**, **unbounded data structures**, **too many live sessions**, or simply an under-provisioned deployment for the current workload.

### 3.1 Distinguish leak vs load

In Prometheus, examine the time-series shape of `process_resident_memory_bytes`:

- **Leak-like behaviour:** memory climbs steadily over hours (or days) without returning to a baseline, even during low-traffic periods.
- **Load-correlated behaviour:** memory rises and falls with traffic or number of active games (`ringrift_games_active`, `ringrift_websocket_connections`).

If memory only stays high when traffic is high and falls after, it may be **expected but mis-sized**. If it keeps ratcheting upward, suspect a leak.

### 3.2 Investigate game/session lifecycle and retention

Because games and sessions are often the largest objects in memory, inspect:

- **Session lifecycle:**
  - `src/server/game/GameSession.ts` and `GameSessionManager.ts` — ensure sessions are destroyed when games end or disconnects are final.
  - State machines in `src/shared/stateMachines/gameSession.ts` and `connection.ts` — confirm terminal states trigger cleanup.
- **Persistence and cleanup:**
  - `GamePersistenceService` and `DataRetentionService` — confirm old games are persisted and removed from memory; verify retention jobs are running as expected.
- **Orphaned/zombie sessions:**
  - Cross-check `ringrift_games_active` and DB state (see `docs/OPERATIONS_DB.md`) for games that never transition to a terminal state.

If active games and sessions remain high long after traffic drops, memory pressure may be a symptom of **lifecycle bugs** (see also `GAME_HEALTH.md`).

### 3.3 Examine code paths that hold large in-memory structures

Look for recent or suspicious changes that:

- Build large in-memory arrays, maps, or caches without bounded size.
- Keep full game histories, traces, or debug snapshots in memory for too long.
- Cache heavy objects across requests without eviction (e.g. in module-level variables or global maps).

Likely hotspots include:

- Game engine and trace tooling (`tests/utils/traceReplayer.ts`, `scripts/safe-view.js`, debugging helpers) when ported into production paths.
- WebSocket broadcast logic that holds references to closed sockets or subscriptions.
- Any new diagnostics added for AI, rules parity, or orchestrator rollout that might store full traces or seeds in memory.

Ensure such tools are either:

- Disabled in production, or
- Bounded with explicit size/age limits and eviction.

### 3.4 Use heap snapshots and profiling (when safe)

In a non-production or **staging** environment that reproduces the issue:

1. Run the Node process with `--inspect` or `--inspect-brk` and use Chrome DevTools or VS Code to capture **heap snapshots** during and after a test load.
2. Compare snapshots to identify which object types and retaining paths are growing.
3. Focus on long-lived objects associated with:
   - Sessions, games, WebSocket connections.
   - DB/Redis clients or custom caches.
   - Large logs or buffers kept in memory.

In production-like environments, consider safe sampling via tools like `clinic` or `0x` if allowed by your operational policies.

### 3.5 Consider Python/AI workloads (if co-located)

If the Node memory alert coincides with heavy AI or training activity:

- Confirm which services share the host/container with the Node backend.
- If any Python/AI processes are co-located, refer to:
  - `ai-service/app/utils/memory_config.py` and `ai-service/tests/test_memory_config.py`.
  - AI training/serving docs: `docs/AI_TRAINING_AND_DATASETS.md`, `docs/AI_TRAINING_PREPARATION_GUIDE.md`, `docs/AI_TRAINING_ASSESSMENT_FINAL.md`.

Where possible, **avoid co-locating** memory-heavy training workloads with the main backend Node process.

---

## 4. Remediation

> Goal: Reduce memory pressure in the short term to avoid OOMs, and implement structural fixes so memory stays within safe bounds under expected load.

### 4.1 Immediate safety measures

If `HighMemoryUsageCritical` is firing or pods are being OOM-killed:

1. **Scale out horizontally (if capacity exists):**
   - Increase the number of backend instances to share the load, using your normal deployment procedures.
2. **Reduce non-essential load:**
   - Temporarily disable or throttle:
     - Non-critical background jobs.
     - Heavy diagnostics or debugging endpoints.
   - Coordinate with AI/training teams if they share infrastructure.
3. **Avoid repeated manual restarts:**
   - Restarts can temporarily clear memory but hide leaks; use them only as a stopgap while you implement a real fix.

### 4.2 Fix lifecycle and retention bugs

If diagnosis points to long-lived sessions or data:

1. **Ensure sessions are torn down properly:**
   - Review `GameSession`, `GameSessionManager`, and related state machines to ensure all terminal states (victory, resignation, timeout, disconnect) trigger cleanup.
2. **Strengthen cleanup jobs:**
   - Confirm that `DataRetentionService` is running and enforcing retention windows as described in `docs/DATA_LIFECYCLE_AND_PRIVACY.md`.
   - Add metrics or logs for retention runs and any failures.
3. **Guard against pathological cases:**
   - Consider reasonable **max durations** or **max moves** for games to avoid unbounded sessions (coordinated with product and rules teams).
   - If new rules/AI work created extremely long games, coordinate with the rules engine SSoT docs and runbooks (`RULES_CANONICAL_SPEC.md`, `GAME_HEALTH.md`).

### 4.3 Bound caches and in-memory data structures

If the leak is tied to custom caches or collections:

- Introduce explicit size and TTL policies for caches.
- Use LRU or similar eviction strategies rather than unbounded `Map`/`Set`.
- Ensure debug traces and logs are streamed to disk/observability systems, not retained in large in-memory arrays.

Any such change should be accompanied by **tests** that verify behaviour under sustained load.

### 4.4 Adjust resource limits and deployment sizing (when justified)

If memory usage is predictable, correlated with legitimate load, and there is **no ongoing leak**:

1. **Revisit deployment sizing:**
   - Review current container memory limits and requests in your deployment manifests.
   - Compare them to observed typical and peak memory usage.
2. **Increase limits cautiously:**
   - Only adjust limits **after** you’ve confirmed memory growth is expected and controlled.
   - Ensure the underlying nodes/cluster can safely support the higher usage.
3. **Document the new baseline:**
   - Update capacity planning docs and internal SLOs to match.

---

## 5. Validation

Before considering a high-memory incident resolved, confirm:

### 5.1 Metrics and alerts

- [ ] `HighMemoryUsage` and `HighMemoryUsageCritical` (if they fired) have cleared and remained clear across multiple evaluation windows.
- [ ] `process_resident_memory_bytes` has returned to a stable baseline that:
  - Does not approach configured limits under normal load, and
  - Shows no steady leak-like upward trend during off-peak periods.
- [ ] Related alerts (`HighEventLoopLag*`, `HighActiveHandles`, latency and error alerts) are green or handled via their respective runbooks.

### 5.2 Behavioural checks

- [ ] Under representative traffic, the backend remains healthy:
  - No OOM kills or unexpected restarts.
  - Latency and error rates within SLOs.
- [ ] For any lifecycle or retention fixes, manual spot checks confirm that:
  - Completed/abandoned games are cleaned up in memory and in the database as expected.
  - No large accumulation of zombie sessions.

### 5.3 Documentation and configuration

- [ ] Any changes to data retention, session lifecycle, or caches are:
  - Covered by automated tests.
  - Documented in `docs/DATA_LIFECYCLE_AND_PRIVACY.md`, `docs/OPERATIONS_DB.md`, or relevant architecture docs if they materially change behaviour.
- [ ] Any updated memory limits or deployment sizing are reflected in:
  - `docs/DEPLOYMENT_REQUIREMENTS.md` or environment-specific ops documentation.
  - Runbooks or incident notes, if they were part of the remediation.

---

## 6. Related Documentation & Runbooks

- **Monitoring & thresholds:**
  - `monitoring/prometheus/alerts.yml`
  - `monitoring/prometheus/prometheus.yml`
  - `monitoring/README.md`
  - `docs/ALERTING_THRESHOLDS.md`

- **Lifecycle & data retention:**
  - `src/server/game/GameSession.ts`
  - `src/server/game/GameSessionManager.ts`
  - `src/server/services/GamePersistenceService.ts`
  - `src/server/services/DataRetentionService.ts`
  - `docs/DATA_LIFECYCLE_AND_PRIVACY.md`
  - `GAME_HEALTH.md` (for long-running games / zombie sessions)

- **Resources & performance:**
  - `EVENT_LOOP_LAG.md` (for main-thread blocking).
  - `RESOURCE_LEAK.md` (for high active handles).
  - `HIGH_LATENCY.md`, `GAME_PERFORMANCE.md`.

- **AI / training context:**
  - `ai-service/app/utils/memory_config.py`
  - `ai-service/tests/test_memory_config.py`
  - `docs/AI_TRAINING_AND_DATASETS.md`
  - `docs/AI_TRAINING_PREPARATION_GUIDE.md`
  - `docs/AI_TRAINING_ASSESSMENT_FINAL.md`

Use this runbook as a **playbook** for investigating and remediating high memory usage. Always defer to the actual implementation, deployment configuration, and `alerts.yml` for the ground truth on thresholds and expected behaviour.
