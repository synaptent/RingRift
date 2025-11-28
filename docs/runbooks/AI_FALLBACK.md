# AI Fallback Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for investigating and mitigating elevated rates of fallback from the remote AI service to local heuristics when `AIFallbackRateHigh` / `AIFallbackRateCritical` alerts fire.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rules in `monitoring/prometheus/alerts.yml` (alerts `AIFallbackRateHigh`, `AIFallbackRateCritical` based on `ringrift_ai_fallback_total` and `ringrift_ai_requests_total`) and scrape configuration in `monitoring/prometheus/prometheus.yml`.
> - **Metrics surfaces:** Backend metrics that count AI requests and fallbacks (e.g. `ringrift_ai_requests_total`, `ringrift_ai_fallback_total`, AI latency metrics) exposed via `/metrics` on the Node app and, where configured, on the Python AI service (`ai-service/app/main.py`).
> - **AI integration and fallback logic:** `AIServiceClient`, `AIEngine`, and related components (`src/server/services/AIServiceClient.ts`, `src/server/game/ai/AIEngine.ts`, `src/server/game/ai/AIPlayer.ts`) that decide when to call the remote AI vs. local heuristic fallback and which metrics to increment.
> - **AI incidents:** `docs/incidents/AI_SERVICE.md` (sections on fallback behaviour, AI quality, and incident handling).
>
> **Precedence:** Metrics definitions, alert rules, backend/AI code, and tests are authoritative for **how fallback is implemented and measured**. This runbook describes **how to interpret and act on the alerts**. If anything here diverges from code/config/tests, **code + configs + tests win** and this document should be updated.
>
> For a high-level “rules vs AI vs infra” classification, see `AI_ARCHITECTURE.md` §0 (AI Incident Overview).

### Orchestrator posture & rules semantics

- The **shared TypeScript rules engine + orchestrator** (`src/shared/engine/**`, contract vectors, and orchestrator adapters) is the single source of truth for rules semantics; backend, sandbox, and Python AI-service are adapters over this SSoT.
- Key runtime flags for rules selection are: `ORCHESTRATOR_ADAPTER_ENABLED`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE`, `ORCHESTRATOR_SHADOW_MODE_ENABLED`, and `RINGRIFT_RULES_MODE`. For AI fallback incidents, these should normally stay in the **orchestrator‑ON** posture; legacy / SHADOW modes are for diagnostics only.
- **Fallbacks may reduce AI strength, but they must not change game rules.** All fallback paths (local heuristics, simplified evaluation, or “no‑AI” modes) must continue to produce moves that are legal under the shared TS rules engine + contracts.
- If symptoms look like **rules-engine issues** rather than “remote AI is unavailable/slow/erroring”, escalate via `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` (Safe rollback checklist) instead of altering fallback logic to accommodate legacy rules behaviour.

---

## 1. When These Alerts Fire

**Alerts (from `monitoring/prometheus/alerts.yml`, `ai-service` group):**

- `AIFallbackRateHigh` (severity: `warning`)
- `AIFallbackRateCritical` (severity: `critical`)

Both alerts are based on the **fraction of AI requests that fall back** to local heuristics over a sliding time window, using:

- `ringrift_ai_requests_total` → total AI requests observed by the backend.
- `ringrift_ai_fallback_total` → subset of requests served via fallback instead of the Python AI service.

> **Note:** Exact thresholds, windows, and PromQL expressions are defined in `monitoring/prometheus/alerts.yml`. Treat that file as the SSoT for numeric values.

**Intended semantics:**

- `AIFallbackRateHigh` → A significant minority of AI turns are using fallback.
- `AIFallbackRateCritical` → The majority of AI turns are using fallback; the remote AI service may be down, slow, or erroring for most requests.

**Impact (expected):**

- Games vs AI remain playable (fallback preserves functionality).
- **AI move quality is degraded** (simpler heuristics, less search depth).
- Users may report that AI feels weaker or more random than usual, but not that the game is down.

**Common causes:**

- AI service genuinely **down** → see `AI_SERVICE_DOWN.md`.
- AI service **slow** (requests time out) → see `AI_PERFORMANCE.md`.
- AI service **erroring** (500s, crashes, bad inputs) → see `AI_ERRORS.md`.
- **Overload / concurrency caps** in `AIServiceClient` causing proactive fallback under heavy load.

---

## 2. Quick Triage (First 5–10 Minutes)

> Goal: Confirm whether high fallback is expected (e.g. AI maintenance), due to AI service unavailability, or a symptom of performance / error spikes.

### 2.1 Confirm alert scope and correlation

In Alertmanager / your monitoring UI:

1. Confirm which alert is firing:
   - `AIFallbackRateHigh` vs `AIFallbackRateCritical`.
   - Start time, duration, and affected environment(s).
2. Check for **related AI alerts** in the same period:
   - `AIServiceDown`.
   - `AIRequestHighLatency`.
   - `AIErrorsIncreasing`.
3. Check for **global availability or resource issues**:
   - `HighErrorRate`, `ServiceDegraded`, `ServiceOffline`.
   - `HighMemoryUsage`, `HighEventLoopLag`, dependency alerts.

If you see widespread non-AI alerts, treat this as part of a **broader incident**; coordinate with `AVAILABILITY.md`, `RESOURCES.md`, and the corresponding runbooks.

### 2.2 Inspect fallback metrics in Prometheus

In Prometheus, run (mirroring the signals used in `alerts.yml`, but not redefining them):

```promql
# Overall fallback fraction over a 10m window (example)
(
  sum(rate(ringrift_ai_fallback_total[10m]))
)
/
(
  sum(rate(ringrift_ai_requests_total[10m]))
)

# Fallback rate by reason (if instrumented with a `reason` label)
sum(rate(ringrift_ai_fallback_total[10m])) by (reason)
```

Use this to answer:

- Is fallback fraction **close to 100%** (service effectively unusable) or just elevated?
- Do fallbacks cluster around a particular **`reason`** (e.g. `timeout`, `connection_refused`, `overloaded`) if such labels exist?
- Is the total AI request rate (`ringrift_ai_requests_total`) normal or spiking (increased traffic)?

> Always refer back to `alerts.yml` for exact windows and thresholds; adjust the `[10m]` example windows and ratios to match that configuration.

### 2.3 Check app readiness and AI status

From an operator shell against the Node backend:

```bash
# Liveness
curl -sS APP_BASE/health | jq . || curl -sS APP_BASE/health

# Readiness with dependency checks (includes AI via ServiceStatusManager)
curl -sS APP_BASE/ready | jq . || curl -sS APP_BASE/ready
```

Verify:

- The AI dependency entry exists (e.g. `aiService`).
- Its `status` (`healthy` / `degraded` / `unhealthy`) and any associated `error` field.
- Whether **other dependencies** (DB, Redis) are also degraded.

If AI is reported `unhealthy` and `ringrift_service_status{service="ai_service"} == 0`, treat `AIServiceDown` as the primary driver and pivot to `AI_SERVICE_DOWN.md`.

### 2.4 Spot‑check AI request behaviour in logs

From the app host (docker‑compose example):

```bash
cd /path/to/ringrift

# Recent AI-related logs
docker compose logs --tail 500 app 2>&1 \
  | grep -Ei 'AI move|AI Service|aiErrorType|AI_SERVICE_|ai_fallback' \
  | tail -n 100
```

Look for:

- Repeated `Failed to get AI move` entries with an `aiErrorType` (`timeout`, `connection_refused`, `service_unavailable`, etc.).
- `AI Service concurrency limit reached, rejecting request` (local overload → fallbacks).
- Structured error codes (`AI_SERVICE_TIMEOUT`, `AI_SERVICE_UNAVAILABLE`, `AI_SERVICE_ERROR`, `AI_SERVICE_OVERLOADED`).

This helps classify fallbacks by **cause**, which drives remediation.

---

## 3. Deep Diagnosis

### 3.1 Classify dominant fallback causes

Using metrics + logs, identify which failure mode dominates:

1. **Service fully down / unreachable**
   - `AIServiceDown` also firing.
   - `ringrift_service_status{service="ai_service"} == 0`.
   - Logs show `aiErrorType: "connection_refused"` or `"service_unavailable"`.
   - AI container not running or `/health` failing.  
     → Primary runbook: **`AI_SERVICE_DOWN.md`**.

2. **Timeouts / high latency**
   - `AIRequestHighLatency` also firing.
   - Logs show `aiErrorType: "timeout"` and structured code `AI_SERVICE_TIMEOUT`.
   - `ringrift_ai_request_duration_seconds_bucket` shows elevated P95/P99.  
     → Primary runbook: **`AI_PERFORMANCE.md`**.

3. **Internal AI errors**
   - `AIErrorsIncreasing` also firing.
   - AI logs (`ai-service` container) show Python exceptions or HTTP 500s in `/ai/move`.
   - Node logs show `AI_SERVICE_ERROR` with non-timeout, non-connection failures.  
     → Primary runbook: **`AI_ERRORS.md`**.

4. **Overload / concurrency limits**
   - Node logs show `AI Service concurrency limit reached, rejecting request` with `AI_SERVICE_OVERLOADED`.
   - AI host may be fine, but `AIServiceClient` max concurrent cap is reached frequently.
   - Total AI request rate is high; fallback uses a distinct `reason` (if instrumented).  
     → Consider load/capacity tuning and coordinated scaling.

5. **Caller mis‑use / unexpected patterns**
   - Spike in AI requests from specific flows (e.g. stress tests, experimental features).
   - Fallback mostly for low-priority or non-critical AI evaluations.  
     → Work with feature owners to balance AI traffic vs capacity.

### 3.2 Correlate with traffic and game behaviour

Check whether AI usage itself has changed:

```promql
# Overall AI request rate
sum(rate(ringrift_ai_requests_total[10m]))

# Optional: rate by game/board type or difficulty if labelled
sum(rate(ringrift_ai_requests_total[10m])) by (board_type, difficulty)
```

Questions to answer:

- Did a new event or feature **increase AI traffic sharply**?
- Are fallbacks concentrated in particular **board types** or **difficulty levels**?
- Does the time of day or tournament activity explain the spike?

### 3.3 Check AI service container and metrics (Python side)

From the deployment root:

```bash
# Container status and logs
cd /path/to/ringrift

docker compose ps ai-service
docker compose logs --tail 200 ai-service

# If metrics are exposed on the Python service
curl -sS http://localhost:8001/metrics | head
```

If the Python service is handling requests slowly or erroring frequently, follow `AI_PERFORMANCE.md` / `AI_ERRORS.md` for deeper diagnosis.

---

## 4. Remediation

> Goal: Reduce the fallback fraction to its normal baseline by addressing the underlying cause (service down, slow, erroring, or overloaded), **without** masking real problems by weakening alerts.

### 4.1 If the AI service is down / failing health checks

Follow **`AI_SERVICE_DOWN.md`**:

- Restart or recreate the `ai-service` container.
- Fix configuration or startup issues surfaced in logs.
- Ensure `/health` returns `{ "status": "healthy" }` and `ringrift_service_status{service="ai_service"}` returns to `1`.

Fallback alerts should clear automatically once the backend resumes using the remote AI successfully.

### 4.2 If fallbacks are driven by high latency / timeouts

Follow **`AI_PERFORMANCE.md`** in parallel:

- Optimise or scale the AI service (model choice, think time, resource allocation).
- Review timeout configuration (`config.aiService.requestTimeoutMs` in Node) versus real AI latencies and `AIRequestHighLatency` thresholds.
- Avoid simply stretching timeouts until they hide real performance regressions; adjust based on realistic SLA for AI move times.

As AI P95/P99 latency returns to normal, timeouts (and thus fallbacks) should subside.

### 4.3 If fallbacks are driven by AI internal errors

Follow **`AI_ERRORS.md`**:

- Use AI logs and metrics to classify and fix errors (e.g. invalid input, model issues, resource failures).
- Add or tighten validation at the Node layer to avoid sending impossible states to AI.
- If needed, temporarily route high‑risk scenarios more aggressively to fallback to preserve game stability while you fix the root cause.

### 4.4 If fallbacks are caused by overload / concurrency caps

If Node logs and metrics indicate many fallbacks due to overload (`AI_SERVICE_OVERLOADED`, concurrency limit reached):

1. **Confirm capacity pressure:**
   - Check AI container CPU/memory (`docker stats ai-service`).
   - Inspect AI host resource dashboards (if available).
   - Verify total AI request volume vs historical baseline.

2. **Mitigation options (via normal change process):**
   - **Scale out** AI service instances if your infra supports it.
   - Reduce per‑request `think_time_ms` in difficulty profiles (see `_CANONICAL_DIFFICULTY_PROFILES` in `ai-service/app/main.py`) to lighten per‑move cost.
   - Adjust Node’s `aiService.maxConcurrent` configuration if documented as tunable and capacity allows (see `config.aiService.maxConcurrent` / `ENVIRONMENT_VARIABLES.md`).
   - Coordinate with product/feature owners to **rate‑limit heavy AI features** or turn down experimental loads.

3. **Avoid** simply increasing concurrency caps without validating that AI hosts can handle the extra load; that will move the bottleneck elsewhere or cause instability.

### 4.5 If fallbacks are expected (maintenance / experiments)

In some controlled scenarios you may intentionally:

- Take AI service offline for maintenance.
- Route a subset of games to fallback for experiments.

In those cases:

- Verify that the elevated fallback aligns with the planned window and scope.
- Ensure player communication accurately reflects **reduced AI strength**.
- Consider temporarily muting non-critical fallback alerts if documented in your ops policy (do **not** change `alerts.yml` ad hoc).

### 4.6 Fallback semantics & allowed degradations

When you rely on fallback, keep these constraints in mind:

- **Allowed forms of fallback**
  - Local heuristic AI in Node (simpler evaluation, reduced search depth).
  - Simplified TS evaluation passes that still call into the shared rules engine.
  - In the extreme, “no‑AI” modes that pick trivially legal moves, while clearly communicating reduced strength to users.
- **Rules semantics must remain unchanged**
  - All fallback paths must still validate and apply moves via the shared TS engine + orchestrator; do **not** introduce alternate rule interpretations, special‑case legality checks, or host‑specific rule tweaks.
  - If behaviour appears to require “bending the rules” to keep AI working, stop and treat this as a rules‑engine/host integration issue, not a fallback tweak.
- **Never change rules as an AI mitigation**
  - Do not flip `RINGRIFT_RULES_MODE` or disable orchestrator adapters purely to hide AI bugs or performance problems.
  - If you have evidence that the orchestrator or shared engine is itself misbehaving, follow `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` for any rules‑engine rollback, and keep that decision separate from AI fallback tuning.

---

## 5. Validation

You are done when:

### 5.1 Metrics / alerts

- [ ] `AIFallbackRateHigh` / `AIFallbackRateCritical` have **cleared** and remain clear for at least one alert window.
- [ ] The fallback fraction, measured via metrics, has returned to the usual low baseline:

  ```promql
  (
    sum(rate(ringrift_ai_fallback_total[10m]))
  )
  /
  (
    sum(rate(ringrift_ai_requests_total[10m]))
  )
  ```

- [ ] Related AI alerts (`AIServiceDown`, `AIRequestHighLatency`, `AIErrorsIncreasing`) are either resolved or understood and stable at an acceptable level.

### 5.2 Health / readiness and experience

- [ ] `/ready` shows the AI dependency as `healthy` or, at worst, `degraded` with a known, temporary reason.
- [ ] Manual AI game smoke tests (login → start game vs AI → multiple moves) show:
  - No frequent AI failures.
  - Move times in a reasonable range per `AI_PERFORMANCE.md`.
  - Observed AI strength matches expectations for the configured difficulty.

### 5.3 Post‑incident follow‑up

- [ ] If this was a user‑visible incident, record a brief summary using `docs/incidents/POST_MORTEM_TEMPLATE.md` and link from `docs/incidents/AI_SERVICE.md`.
- [ ] Update any internal runbooks or dashboards that operators use to monitor AI quality vs fallback usage.
- [ ] Consider adding alerting or dashboards that segment fallback by **reason**, difficulty, or board type if that would improve future triage.

---

## 6. Related Documentation

- **Incident docs:**
  - `docs/incidents/AI_SERVICE.md`
  - `docs/incidents/AVAILABILITY.md`
  - `docs/incidents/LATENCY.md`
  - `docs/incidents/RESOURCES.md`
  - `docs/incidents/TRIAGE_GUIDE.md`
- **Related runbooks:**
  - `docs/runbooks/AI_SERVICE_DOWN.md`
  - `docs/runbooks/AI_PERFORMANCE.md`
  - `docs/runbooks/AI_ERRORS.md`
  - `docs/runbooks/HIGH_ERROR_RATE.md`
  - `docs/runbooks/SERVICE_DEGRADATION.md`
- **Ops / config SSoT:**
  - `monitoring/prometheus/alerts.yml`
  - `monitoring/prometheus/prometheus.yml`
  - `docs/ALERTING_THRESHOLDS.md`
  - `docs/DEPLOYMENT_REQUIREMENTS.md`
  - `docs/ENVIRONMENT_VARIABLES.md`
  - `docs/OPERATIONS_DB.md`
- **Orchestrator rollout:**
  - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` – orchestrator‑everywhere posture and Safe rollback checklist when issues are truly rules‑engine related.
