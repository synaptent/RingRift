# AI Service Down Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for diagnosing and restoring the RingRift AI service when the `AIServiceDown` alert fires.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rules in `monitoring/prometheus/alerts.yml` (alert `AIServiceDown` based on `ringrift_service_status{service="ai_service"}`) and scrape configuration in `monitoring/prometheus/prometheus.yml`.
> - **Health and status surfaces:** `ServiceStatusManager` and `HealthCheckService` (`src/server/services/ServiceStatusManager.ts`, `src/server/services/HealthCheckService.ts`) behind `/health` and `/ready`, plus the `ringrift_service_status` metric.
> - **AI integration:** `AIServiceClient` and AI orchestration (`src/server/services/AIServiceClient.ts`, `src/server/game/ai/AIEngine.ts`, `src/server/game/ai/AIPlayer.ts`) including circuit breaker, timeouts, and fallback behaviour.
> - **AI service implementation:** FastAPI app in `ai-service/app/main.py` (endpoints `/`, `/health`, `/metrics`, `/ai/move`, `/ai/evaluate`, `/ai/choice/*`, `/ai/cache`).
> - **Incident process:** `docs/incidents/AI_SERVICE.md` (AI incidents overview and narrative guidance).
>
> **Precedence:** Alert definitions, metrics, backend code, and tests are authoritative for **what is measured and how failure is handled**. This runbook explains **how to investigate and remediate**; if it conflicts with code/config/tests, **code + configs + tests win** and this document should be updated.
>
> For a high-level “rules vs AI vs infra” classification, see `AI_ARCHITECTURE.md` §0 (AI Incident Overview).

### Orchestrator posture & rules semantics

- The **shared TypeScript rules engine + orchestrator** (`src/shared/engine/**`, contract vectors, and orchestrator adapters) is the single source of truth for rules semantics; backend, sandbox, and Python AI-service are adapters over this SSoT.
- Key runtime flags for rules selection are: `ORCHESTRATOR_ADAPTER_ENABLED`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE`, `ORCHESTRATOR_SHADOW_MODE_ENABLED`, and `RINGRIFT_RULES_MODE`. When the remote AI service is down, these should normally stay in the **orchestrator‑ON** posture; legacy / SHADOW modes are for diagnostics only.
- **Losing the remote AI service should not change game rules.** Node should route AI turns through documented fallback paths (local heuristics, simplified evaluation, or explicit “no‑AI” modes) that still apply moves via the shared TS rules engine.
- If symptoms look like **rules-engine issues** rather than “remote AI is unavailable”, escalate via `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` (Safe rollback checklist) instead of altering rules or switching to legacy engines in response to AI outages.

---

## 1. When This Alert Fires

**Alert (from `monitoring/prometheus/alerts.yml`, `availability` group):**

- `AIServiceDown` (severity: `warning`, team: `ai`)

**Conceptual condition (do not edit thresholds here; see `alerts.yml`):**

```promql
ringrift_service_status{service="ai_service"} == 0
```

`ringrift_service_status` is emitted from the Node backend via `ServiceStatusManager` and updated by:

- Successful / failed calls in `AIServiceClient` (e.g. `getAIMove`, `evaluatePosition`).
- Periodic health checks via `AIServiceClient.healthCheck()` and `HealthCheckService`.

**Intended semantics:**

- `1` → AI service healthy (remote Python service reachable and responding).
- `0` → AI service unavailable or consistently failing (circuit breaker open, health checks failing, or repeated request failures).

**Impact (expected):**

- Remote AI service is **not usable** for game traffic.
- `AIEngine` should route AI turns through **local heuristic fallback** instead of the Python service.
- Games vs AI remain playable, but **move quality is degraded** (simpler moves, less search depth).
- This is **not** a full availability outage, but a **quality / experience degradation** for AI games.

**Related alerts and docs:**

- `AIFallbackRateHigh`, `AIFallbackRateCritical` → see `AI_FALLBACK.md`.
- `AIRequestHighLatency` → see `AI_PERFORMANCE.md`.
- `AIErrorsIncreasing` → see `AI_ERRORS.md`.
- Availability overview and AI-specific narrative: `docs/incidents/AI_SERVICE.md`.

---

## 2. Quick Triage (First 5–10 Minutes)

> Goal: Confirm whether this is a **real loss of the Python AI service** vs. a transient blip or a Node-side classification issue.

### 2.1 Confirm the alert and look for correlated signals

In Alertmanager / your monitoring UI:

1. Confirm that **`AIServiceDown`** is firing and note:
   - Environment (staging vs production).
   - Start time and duration.
   - Any annotations (summary, description, impact).
2. Look for **related AI alerts** in the same time window:
   - `AIFallbackRateHigh` / `AIFallbackRateCritical`.
   - `AIRequestHighLatency`.
   - `AIErrorsIncreasing`.
3. Check for **broader availability/resource issues**:
   - `HighErrorRate`, `ServiceDegraded`, `ServiceOffline`.
   - `HighMemoryUsage`, `HighEventLoopLag`, `HighWebSocketConnections`, dependency alerts (`DatabaseDown`, `RedisDown`).

If many non-AI alerts are firing, treat this as part of a **wider incident** and coordinate with availability / resources runbooks.

### 2.2 Verify app health and AI status from the Node backend

From an operator shell with access to the app:

```bash
# Overall liveness
curl -sS APP_BASE/health | jq . || curl -sS APP_BASE/health

# Readiness with dependency breakdown (HealthCheckService + ServiceStatusManager)
curl -sS APP_BASE/ready | jq . || curl -sS APP_BASE/ready
```

Inspect the readiness payload for the AI service entry (naming may vary slightly with implementation, e.g. `aiService`):

- Check that the AI entry is present.
- Note its `status` (e.g. `healthy`, `degraded`, `unhealthy`) and any `error` message.

If `/ready` is unhealthy for AI and other dependencies (DB, Redis) at the same time, prioritise **availability** and dependency runbooks (`SERVICE_DEGRADATION.md`, `DATABASE_DOWN.md`, `REDIS_DOWN.md`).

### 2.3 Check the `ringrift_service_status` metric

In Prometheus (or Grafana with Prometheus as a source):

```promql
# Current AI service status over time
ringrift_service_status{service="ai_service"}

# Optional: recent status history
avg_over_time(ringrift_service_status{service="ai_service"}[30m])
```

Confirm that the value has actually transitioned to `0` and stayed there for the alert’s `for` duration (see `alerts.yml` for exact timings).

### 2.4 Verify AI service from inside the stack (docker‑compose example)

From the deployment root (where `docker-compose.yml` lives):

```bash
cd /path/to/ringrift

# 1. Check AI service container status
docker compose ps ai-service

# 2. Direct health check from the host
curl -sS http://localhost:8001/health | jq . || curl -sS http://localhost:8001/health

# 3. Health check from the app container (tests connectivity + DNS)
docker compose exec app curl -sS http://ai-service:8001/health | jq . \
  || docker compose exec app curl -sS http://ai-service:8001/health

# 4. Recent AI service logs
docker compose logs --tail 200 ai-service
```

Interpretation:

- If the container is **not running** or restarting repeatedly → see **3.1 Container not running / crashing**.
- If the container is running but `/health` is not reachable or not returning `{"status": "healthy"}` → see **3.2 Unhealthy or misbehaving AI service**.
- If both `/ready` (Node) and `/health` (Python) look healthy but `AIServiceDown` persists → see **3.3 Node-side classification / circuit breaker issues**.

---

## 3. Deep Diagnosis

### 3.1 Container not running / crashing

Check container status and crash reasons:

```bash
# Container status
docker compose ps ai-service

# Logs around the last exit
docker compose logs ai-service | tail -100

# Inspect last container state for OOM / exit codes
docker inspect $(docker compose ps -q ai-service) | jq '.[0].State'
```

Common patterns:

- **Out of memory (OOMKilled):**
  - `State.OOMKilled == true`, logs show memory errors loading models.
  - Correlate with host/container memory usage and `HighMemoryUsage` alerts.
- **Import / configuration errors:**
  - Tracebacks about missing modules, bad environment variables, invalid paths.
- **Startup exceptions in `ai-service/app/main.py`:**
  - Model loading issues, rules engine initialisation errors, or FastAPI boot problems.

If the container is not running at all, skip ahead to **4.1 Restore the AI service container**.

### 3.2 Unhealthy or misbehaving AI service

If the container is up but `/health` is not reporting healthy, or `AIServiceClient.healthCheck()` is failing:

```bash
# From the host
curl -sS http://localhost:8001/health | jq . || curl -sS http://localhost:8001/health

# From within the ai-service container for more direct debugging
docker compose exec ai-service curl -sS http://localhost:8001/health | jq . \
  || docker compose exec ai-service curl -sS http://localhost:8001/health

# Inspect detailed logs for errors and model loading
docker compose logs --tail 500 ai-service | grep -Ei 'ERROR|Exception|Traceback|model|load'
```

Look for:

- Repeated **500 errors** or unhandled exceptions in `/ai/move` or `/ai/evaluate`.
- Messages about failing to load models, rules, or configuration.
- Health endpoint returning something other than `{ "status": "healthy" }`.

### 3.3 Node-side classification / circuit breaker / connectivity issues

Even if the Python service is up, the Node backend may treat it as “down” due to:

- Timeouts in `AIServiceClient` (`requestTimeoutMs` too low vs actual latency).
- Connection failures (`ECONNREFUSED`, DNS issues).
- Circuit breaker in `AIServiceClient` opening after repeated failures.
- Local concurrency caps preventing new AI calls (max concurrent AI requests exceeded).

Observe Node logs and circuit breaker state:

```bash
# App logs with AI-related errors
cd /path/to/ringrift
docker compose logs --tail 500 app 2>&1 | grep -Ei 'AI Service|AI move|aiErrorType|AI_SERVICE_' | tail -n 100
```

Look for log entries from `AIServiceClient` such as:

- `AI Service error:` (categorised errors from the axios interceptor).
- `Failed to get AI move` (includes `aiErrorType`, latency, player, difficulty).
- `AI Service concurrency limit reached, rejecting request` (overload protection).
- `Circuit breaker opened after repeated failures`.

If you can attach a small diagnostic endpoint or shell inside the app container, you can also query:

- `getAIServiceClient().getCircuitBreakerStatus()` (if you have a debug harness) to confirm if the breaker is open.
- `ringrift_service_status{service="ai_service"}` trend to see if status is flipping between 0 and 1.

If `AIRequestHighLatency` is also firing, follow `AI_PERFORMANCE.md` in parallel—high latency often triggers timeouts → AI treated as down.

---

## 4. Remediation

> Goal: Restore the AI service to a **healthy** and **reachable** state, then ensure Node correctly recognises that state and resumes using remote AI instead of fallback.

### 4.1 Restore the AI service container

#### Graceful restart

```bash
cd /path/to/ringrift

# Restart only the AI service container
docker compose restart ai-service

# Watch logs during startup
docker compose logs -f ai-service
```

Wait for:

- FastAPI to report that it has started.
- Any model/rules initialisation messages to complete.
- No repeating exceptions in startup.

Then re-check health:

```bash
curl -sS http://localhost:8001/health | jq . || curl -sS http://localhost:8001/health
```

#### Full recreate (if restart does not help)

```bash
cd /path/to/ringrift

# Stop and remove the existing container
docker compose stop ai-service
docker compose rm -f ai-service

# Optionally pull the latest image (if using registry images)
docker compose pull ai-service

# Start fresh
docker compose up -d ai-service

# Tail logs to confirm healthy startup
docker compose logs -f ai-service
```

If model weights or configuration live on volumes or bind mounts, ensure they are present and compatible with the current image.

### 4.2 Address resource constraints or environment issues

If logs/metrics indicate resource pressure on the AI host:

```bash
# Resource snapshot for the AI container
docker stats --no-stream ai-service

# Basic memory view inside container
docker compose exec ai-service head -40 /proc/meminfo
```

Potential actions (via your normal infra/deployment process):

- Increase CPU/memory limits or move AI service to a more capable node.
- Ensure model size and number of concurrent inferences are appropriate.
- If using accelerators (e.g. GPU), verify drivers and `nvidia-smi` (where available).

Coordinate with your infra team / deployment requirements documented in `docs/DEPLOYMENT_REQUIREMENTS.md`.

### 4.3 Triage and adjust Node‑side timeout / overload behaviour (if needed)

If the AI container is healthy but Node still reports `AIServiceDown`:

1. **Check for timeouts:**
   - Node logs from `AIServiceClient` with `aiErrorType: "timeout"`.
   - `AIRequestHighLatency` alert is likely also firing.
2. **Check for overload:**
   - Log message `AI Service concurrency limit reached, rejecting request` (AI_SERVICE_OVERLOADED).
   - Spikes in AI traffic volume.

Mitigations (through normal config/change process):

- Review AI service timeout and concurrency configuration (see `src/server/config/**/*.ts`, environment variables documented in `docs/ENVIRONMENT_VARIABLES.md`).
- Ensure `requestTimeoutMs` is consistent with realistic AI response times (driven by `AI_REQUEST_HIGH_LATENCY` thresholds and typical `ai-service` behaviour).
- If sustained, consider:
  - Scaling out `ai-service` instances.
  - Reducing effective think time in the Python service difficulty profiles.
  - Adjusting max concurrent AI requests per Node process when safe.

### 4.4 Coordinate with fallback and error-rate runbooks

While `AIServiceDown` is active, you will usually also see:

- Increased fallback usage → `AIFallbackRateHigh` / `AIFallbackRateCritical` (see `AI_FALLBACK.md`).
- Potential `AIErrorsIncreasing` if failures are being counted as errors (see `AI_ERRORS.md`).
- Elevated HTTP 5xx if AI failures bubble up to client‑visible errors instead of clean fallbacks (see `HIGH_ERROR_RATE.md`).

Work these in tandem to ensure that:

- Game servers continue to **serve AI players via local heuristics** (not hard‑failing).
- Once the AI service is restored, fallbacks and error rates converge back to baseline.

### 4.5 Fallback semantics & allowed degradations when AI is down

While `AIServiceDown` is active:

- **Use only documented fallback modes**
  - Local heuristic AI inside Node (`AIEngine` / `AIPlayer`) that still validates moves via the shared TS rules engine.
  - Simplified TS evaluation or “no‑AI” modes that clearly communicate reduced strength to users.
- **Keep rules semantics invariant**
  - All fallback moves must remain legal under the shared engine + contracts; do **not** introduce alternate legality checks or host‑specific rule tweaks to compensate for AI outages.
  - Avoid flipping `RINGRIFT_RULES_MODE` or disabling orchestrator adapters as a response to AI service loss; that conflates rules rollback with AI availability.
- **Separate rules rollbacks from AI mitigations**
  - If you have evidence that the orchestrator or shared engine is itself misbehaving, follow `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` for any rules‑engine rollback, and record that as a separate decision from AI fallback handling.

---

## 5. Validation

Before considering the incident resolved:

### 5.1 Metrics and alerts

- [ ] `ringrift_service_status{service="ai_service"}` has returned to `1` and stayed at `1` for at least one full alert evaluation window.
- [ ] The `AIServiceDown` alert has cleared and **does not re-fire** after restart.
- [ ] If previously firing, `AIFallbackRateHigh` / `AIFallbackRateCritical`, `AIRequestHighLatency`, and `AIErrorsIncreasing` have cleared as well.

Example PromQL checks:

```promql
# Current AI service status
ringrift_service_status{service="ai_service"}

# Fallback fraction (sanity check)
sum(rate(ringrift_ai_fallback_total[10m]))
/
sum(rate(ringrift_ai_requests_total[10m]))
```

### 5.2 Health / readiness and user experience

- [ ] `/health` and `/ready` both report overall healthy, and the AI section in readiness shows a healthy status with no persistent error message.
- [ ] Manual smoke test of an AI game (login → start game vs AI → make several moves) shows:
  - No repeated AI failures in the UI.
  - Moves arrive within expected time.
  - Logs confirm that at least some moves are served by the remote AI service again (not 100% fallback).

### 5.3 Post‑incident hygiene

- [ ] If this was production‑impacting, capture a brief incident summary using `docs/incidents/POST_MORTEM_TEMPLATE.md` and link under `docs/incidents/AI_SERVICE.md`.
- [ ] If configuration or capacity changes were required, update any relevant environment docs (`docs/ENVIRONMENT_VARIABLES.md`, `docs/DEPLOYMENT_REQUIREMENTS.md`).
- [ ] Consider adding/adjusting tests or synthetic checks that would have caught this before or earlier (e.g., periodic integration test hitting `/ai/move` end‑to‑end).

---

## 6. Related Documentation

- **Incident docs:**
  - `docs/incidents/AI_SERVICE.md`
  - `docs/incidents/AVAILABILITY.md`
  - `docs/incidents/LATENCY.md`
  - `docs/incidents/RESOURCES.md`
  - `docs/incidents/TRIAGE_GUIDE.md`
- **Related runbooks:**
  - `docs/runbooks/AI_FALLBACK.md`
  - `docs/runbooks/AI_PERFORMANCE.md`
  - `docs/runbooks/AI_ERRORS.md`
  - `docs/runbooks/HIGH_ERROR_RATE.md`
  - `docs/runbooks/SERVICE_DEGRADATION.md`
  - `docs/runbooks/SERVICE_OFFLINE.md`
- **Ops / config SSoT:**
  - `monitoring/prometheus/alerts.yml`
  - `monitoring/prometheus/prometheus.yml`
  - `docs/ALERTING_THRESHOLDS.md`
  - `docs/DEPLOYMENT_REQUIREMENTS.md`
  - `docs/ENVIRONMENT_VARIABLES.md`
  - `docs/OPERATIONS_DB.md`
- **Orchestrator rollout:**
  - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` – orchestrator‑everywhere posture and Safe rollback checklist when issues are truly rules‑engine related.
