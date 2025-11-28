# AI Errors Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for investigating elevated AI error rates when `AIErrorsIncreasing` fires and for distinguishing between input, infrastructure, and model/code issues.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rules in `monitoring/prometheus/alerts.yml` (alert `AIErrorsIncreasing` based on `ringrift_ai_requests_total{outcome="error"}`) and scrape configuration in `monitoring/prometheus/prometheus.yml`.
> - **Metrics surfaces:** AI request outcome metrics emitted by the backend (e.g. `ringrift_ai_requests_total` with labels such as `outcome`) and associated latency/fallback metrics.
> - **AI integration & error handling:** `AIServiceClient`, `AIEngine`, and related backend components that classify AI failures into structured error codes (`AI_SERVICE_TIMEOUT`, `AI_SERVICE_UNAVAILABLE`, `AI_SERVICE_ERROR`, `AI_SERVICE_OVERLOADED`) and decide when to fall back (`src/server/services/AIServiceClient.ts`, `src/server/game/ai/AIEngine.ts`).
> - **AI service implementation:** FastAPI AI service in `ai-service/app/main.py` and underlying AI engines, which may raise errors surfaced as HTTP 5xx to the backend.
> - **Incident process:** `docs/incidents/AI_SERVICE.md` (sections on AI error handling and debugging), plus general incident docs.
>
> **Precedence:** Metric definitions, alert expressions, backend and AI service code, and tests are authoritative for **what counts as an error** and how it is reported. This runbook focuses on **triage and remediation**. If discrepancies appear, **code + configs + tests win** and this document should be revised.
>
> For a high-level “rules vs AI vs infra” classification, see `AI_ARCHITECTURE.md` §0 (AI Incident Overview).

### Orchestrator posture & rules semantics

- The **shared TypeScript rules engine + orchestrator** (`src/shared/engine/**`, contract vectors, and orchestrator adapters) is the single source of truth for rules semantics; backend, sandbox, and Python AI-service are adapters over this SSoT.
- Key runtime flags for rules selection are: `ORCHESTRATOR_ADAPTER_ENABLED`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE`, `ORCHESTRATOR_SHADOW_MODE_ENABLED`, and `RINGRIFT_RULES_MODE`. For AI incidents, these should normally stay in the **orchestrator‑ON** posture; legacy / SHADOW modes are for diagnostics only.
- **Do not respond to AI errors by changing game rules or flipping back to legacy rules engines.** Investigate AI infrastructure, models, or integration first; rules semantics remain anchored in the shared TS engine + contracts.
- If symptoms look like **rules-engine issues** (illegal moves accepted, incorrect scoring/turn sequencing) rather than AI‑only behaviour, escalate via `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` (Safe rollback checklist) instead of improvising rules changes here.

---

## 1. When This Alert Fires

**Alert (from `monitoring/prometheus/alerts.yml`, `ai-service` group):**

- `AIErrorsIncreasing` (severity: `warning`)

**Conceptual condition (do not edit thresholds here; see `alerts.yml`):**

```promql
sum(rate(ringrift_ai_requests_total{outcome="error"}[5m])) > 0.1
```

This indicates that the backend is observing AI requests classified as **errors** at a sustained rate above the configured threshold.

**Impact (expected):**

- Some AI turns fail outright or require fallback.
- Players may experience occasional AI failures (e.g. error messages, stalling turns, or visible fallbacks), especially under load.
- If unaddressed, error rates may cascade into `AIFallbackRateHigh`, `AIServiceDown`, or elevated HTTP 5xx (`HIGH_ERROR_RATE.md`).

**Common causes:**

- **Transport / infra issues:** timeouts, connection failures, AI service restarts.
- **Invalid or unexpected inputs:** Node sending game states or choices the AI does not handle well.
- **Model / code bugs inside the Python service:** unhandled exceptions in `/ai/move` or choice endpoints.
- **Resource problems:** OOMs or CPU spikes causing intermittent failures.

---

## 2. Quick Triage (First 5–10 Minutes)

> Goal: Determine whether AI errors are primarily infra/timeout‑driven, input/contract issues, or internal AI bugs.

### 2.1 Confirm alert and look for related AI and availability signals

In Alertmanager / your monitoring UI:

1. Confirm `AIErrorsIncreasing` is firing and note:
   - Environment, start time, duration.
   - Any annotations (summary, description, impact).
2. Check for correlated alerts:
   - `AIServiceDown`.
   - `AIFallbackRateHigh` / `AIFallbackRateCritical`.
   - `AIRequestHighLatency`.
   - Global alerts: `HighErrorRate`, `HighP99Latency`, dependency/resource alerts.

If many non-AI alerts are active, treat this as part of a **wider incident** and work alongside availability / latency runbooks.

### 2.2 AI triage checklist (metrics, saturation, semantics)

- **Quantify errors and saturation**
  - Check `ringrift_ai_requests_total{outcome="error"}` and related slices (by `reason`/`code` where available) to understand absolute and relative error rates.
  - Cross‑check `ringrift_ai_request_duration_seconds_bucket` for elevated P95/P99 latency; many timeouts will present as both high latency and errors.
  - Confirm whether AI traffic volume itself is normal or spiking (overall request rate, per‑difficulty load).
- **Check AI-service health and resources**
  - Verify `/health` and `/ready` on both the Node app and `ai-service`, and inspect container/host CPU and memory usage.
  - If the Python service is saturated or unhealthy, prioritise capacity / availability remediation (`AI_PERFORMANCE.md`, `AI_SERVICE_DOWN.md`) before deep bug hunts.
- **Validate semantics against contract vectors (rules vs AI)**
  - If errors appear to reflect “impossible” game states or rule violations, run or inspect the status of the shared contract‑vector suites instead of assuming an AI bug:
    - TS: `tests/contracts/contractVectorRunner.test.ts`
    - Python: `ai-service/tests/contracts/test_contract_vectors.py`
  - Treat these contracts and `.shared` rules tests as the authority for rules behaviour; AI incidents should not drive changes to canonical rules semantics.
- **Choose the remediation path**
  - Use the classification in §3.1 (transport/infra vs input vs internal errors vs overload) together with the above signals to decide whether to:
    - Scale/tune AI performance (`AI_PERFORMANCE.md`),
    - Fix input/contracts and validation,
    - Address internal AI bugs,
    - Or adjust capacity/limits and lean temporarily on documented fallbacks (`AI_FALLBACK.md`, `AI_SERVICE_DOWN.md`).

### 2.3 Inspect AI error metrics in Prometheus

Use Prometheus to quantify and slice AI error rates:

```promql
# Overall AI error rate (matches alert semantics)
sum(rate(ringrift_ai_requests_total{outcome="error"}[5m]))

# Optional: error rate by failure reason if labelled (e.g. reason, code)
sum(rate(ringrift_ai_requests_total{outcome="error"}[5m])) by (reason, code)

# Compare errors to total AI requests
sum(rate(ringrift_ai_requests_total{outcome="error"}[5m]))
/
sum(rate(ringrift_ai_requests_total[5m]))
```

Questions:

- Is the **absolute error rate** high, or is this primarily a high fraction at low traffic?
- Are errors heavily concentrated in a specific **reason/code** (timeouts, unavailable, validation, internal)?
- Have errors been steadily increasing, or was there a sudden spike?

> Exact labels (e.g. `reason`, `code`) depend on how metrics are wired in the backend. Refer to the metrics instrumentation in `AIServiceClient` / `AIEngine` and to `alerts.yml` for authoritative label sets.

### 2.4 Check readiness and AI dependency status

From an operator shell against Node:

```bash
# Liveness
curl -sS APP_BASE/health | jq . || curl -sS APP_BASE/health

# Readiness (shows dependency checks including AI via ServiceStatusManager)
curl -sS APP_BASE/ready | jq . || curl -sS APP_BASE/ready
```

Look for:

- AI dependency status (e.g. `aiService`): `healthy` vs `degraded` / `unhealthy`.
- Any `error` string associated with the AI status.
- Other dependencies (DB, Redis) being degraded concurrently.

If AI is `unhealthy` and `AIServiceDown` is also firing, treat that as the primary axis (`AI_SERVICE_DOWN.md`) and use this runbook to classify lingering errors after recovery.

### 2.5 Gather representative Node and AI logs

From the app host (docker‑compose example):

```bash
cd /path/to/ringrift

# Node-side AI error logs
docker compose logs --tail 500 app 2>&1 \
  | grep -Ei 'AI Service error|Failed to get AI move|AI_SERVICE_' \
  | tail -n 100

# Python AI service errors
docker compose logs --tail 500 ai-service 2>&1 \
  | grep -Ei 'Error generating AI move|ERROR|Exception|Traceback' \
  | tail -n 100
```

These logs will often show:

- Node‑side categorisation: `aiErrorType` (`timeout`, `connection_refused`, `service_unavailable`, `client_error`, `server_error`, `unknown`).
- HTTP status codes returned by the AI service (`500`, `503`, `4xx`).
- Python stack traces and exception types.

---

## 3. Deep Diagnosis

### 3.1 Classify dominant error types

Use a combination of metrics and logs to understand **why** errors are occurring:

1. **Timeout / connectivity errors**
   - Node logs show `aiErrorType: "timeout"` or "connection_refused" / "service_unavailable".
   - AI service may be under heavy load or intermittently unreachable.
   - Often correlated with `AIRequestHighLatency`, `AIServiceDown`, and elevated fallback.  
     → Treat these primarily as **performance / availability** issues (`AI_PERFORMANCE.md`, `AI_SERVICE_DOWN.md`, `AI_FALLBACK.md`).

2. **Client errors (4xx) from AI service**
   - Node logs show `aiErrorType: "client_error"` (HTTP 4xx).
   - Python logs show exceptions triggered by input validation or unsupported request shapes.  
     → Indicates **contract or input issues** between Node and AI; the AI service is rejecting requests as invalid.

3. **Server errors (5xx) from AI service**
   - Node logs show `aiErrorType: "server_error"` with structured code `AI_SERVICE_ERROR`.
   - Python logs show stack traces inside AI or rules code.  
     → Indicates **bugs or runtime failures** in AI logic, models, or rules integration.

4. **Overload‑driven errors**
   - Node logs include `AI Service concurrency limit reached, rejecting request` and code `AI_SERVICE_OVERLOADED`.
   - Errors may appear as explicit overload responses or as timeouts/failures due to resource exhaustion.  
     → Indicates **capacity mismatch** between demand and configured concurrency / resources.

Understanding which of these dominates will guide remediation.

### 3.2 Check where failures occur in the request flow

Recall the high-level flow (see `docs/incidents/AI_SERVICE.md` and `AI_ARCHITECTURE.md`):

1. Game logic decides an AI move is needed (via `AIEngine` / `AIPlayer`).
2. `AIServiceClient` sends a request to FastAPI `/ai/move` with `MoveRequest`.
3. FastAPI’s `get_ai_move` builds or fetches an AI instance and calls into the underlying engine.
4. A `MoveResponse` is returned or an exception is raised.

Failures can appear at each layer:

- **Before HTTP:** Node rejects due to local concurrency cap or pre‑flight cancellation.
- **Transport:** connection errors, timeouts, circuit breaker.
- **FastAPI validation:** Pydantic rejects request payloads (400).
- **AI logic:** Python exceptions when computing/serialising moves.

The combination of Node and Python logs usually makes the failing step clear.

### 3.3 Look for deployment or configuration changes

If errors began recently, check for:

- New AI models or profile changes (`_CANONICAL_DIFFICULTY_PROFILES` in `main.py`, model versioning docs, `AI_IMPROVEMENT_PLAN.md`).
- Code changes on the Python side (new heuristics, new board support, rules changes).
- Changes to request/response contracts in the Node backend or shared types.
- Environment variable or topology updates affecting AI (see `docs/ENVIRONMENT_VARIABLES.md`, `docs/DEPLOYMENT_REQUIREMENTS.md`).

A clear alignment with a particular change typically favours **rollback or targeted fixes**.

---

## 4. Remediation

> Goal: Reduce AI errors to baseline by addressing the dominating failure cause, while preserving game stability via fallbacks where appropriate.

### 4.1 If errors are primarily timeouts / connectivity

Treat as **performance/availability** problems first:

- Follow `AI_PERFORMANCE.md` to improve latency and capacity.
- Follow `AI_SERVICE_DOWN.md` if the AI service is intermittently or fully down.
- Use `AI_FALLBACK.md` to ensure that games continue via heuristics while you fix the issue.

Once latency and connectivity are stabilised, re-evaluate `AIErrorsIncreasing`; many timeout‑classified errors should disappear.

### 4.2 If errors are primarily client/input issues (4xx)

1. **Identify error types from Python logs and responses**:

   ```bash
   cd /path/to/ringrift

   docker compose logs --tail 500 ai-service 2>&1 \
     | grep -Ei '422 Unprocessable|400 Bad Request|ValidationError|ValueError' \
     | tail -n 50
   ```

2. **Cross-check request models vs shared types:**
   - Python `MoveRequest`, `LineRewardChoiceRequest`, `RingEliminationChoiceRequest`, `RegionOrderChoiceRequest` in `ai-service/app/main.py` and model modules.
   - Corresponding TypeScript payloads in `AIServiceClient` and AI integration paths.

3. **Mitigations (through normal code/change process):**
   - Fix mismatches between TS and Python models (e.g. missing fields, wrong naming, invalid ranges).
   - Add/strengthen validation on the Node side so invalid data is never sent to AI; surface clean 4xx/5xx to clients instead of mysterious AI failures.
   - Add regression tests for the problematic input shapes using existing AI integration tests (`tests/unit/AIEngine.serviceClient.test.ts`, Python AI tests).

### 4.3 If errors are primarily AI internal/server errors (5xx)

1. **Inspect full stack traces in AI logs:**

   ```bash
   cd /path/to/ringrift

   docker compose logs --tail 500 ai-service 2>&1 \
     | grep -Ei 'Error generating AI move|Traceback|ERROR' \
     | tail -n 100
   ```

2. **Classify by exception type and location:**
   - Logic errors in AI engines (`RandomAI`, `HeuristicAI`, `MinimaxAI`, `MCTSAI`, `DescentAI`).
   - Rules or board operations invoked from AI.
   - Serialization / type errors when encoding/decoding game state or moves.

3. **Mitigations:**
   - Fix underlying AI bugs in code; add tests reproducing the failing scenarios.
   - Where appropriate, catch specific exceptions and translate them into predictable **fallback paths** rather than raw 500s, keeping metrics and logs explicit about cause.
   - If a particular difficulty or AI engine is unstable, consider temporarily disabling or downgrading it and routing those requests to a more stable engine.

### 4.4 If errors are overload‑related

Follow the capacity recommendations in `AI_PERFORMANCE.md` and `AI_FALLBACK.md`:

- Scale `ai-service` capacity where possible.
- Verify Node concurrency caps and adjust only when you know AI hosts can cope.
- Coordinate with product/ops to throttle or limit heavy AI usage until capacity is improved.

### 4.5 Temporary mitigations to protect gameplay

While addressing root causes, you may:

- Increase reliance on heuristic fallback for clearly problematic scenarios to avoid hard failures.
- Temporarily lower the maximum selectable difficulty or disable experimental modes that trigger unstable behaviour.
- Communicate to users that AI strength may be temporarily reduced while issues are being resolved.

All such mitigations should be reversible and tracked in an incident summary.

---

## 5. Validation

You are done when:

### 5.1 Metrics / alerts

- [ ] `AIErrorsIncreasing` has **cleared** and remains clear for at least one alert window.
- [ ] The error portion of AI requests has returned to a low, stable baseline:

  ```promql
  sum(rate(ringrift_ai_requests_total{outcome="error"}[5m]))
  /
  sum(rate(ringrift_ai_requests_total[5m]))
  ```

- [ ] Related AI alerts (`AIFallbackRateHigh`, `AIRequestHighLatency`, `AIServiceDown`) are resolved or stable and understood.

### 5.2 Health / readiness and user experience

- [ ] `/ready` shows AI as `healthy` or acceptably `degraded` with known, temporary reasons.
- [ ] Manual AI game smoke tests (login → AI game → multiple moves) show **no repeated AI failures** or unexplained error states.

### 5.3 Post‑incident follow‑up

- [ ] If this represented a meaningful incident, capture a short post‑mortem using `docs/incidents/POST_MORTEM_TEMPLATE.md` and link from `docs/incidents/AI_SERVICE.md`.
- [ ] If bugs or contract issues were fixed, ensure regression tests exist (Node + Python) and are documented in relevant AI design / test planning docs.
- [ ] Consider enhancing AI metrics (e.g. error categorisation by exception type or calling context) if debugging required manual log spelunking.

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
  - `docs/runbooks/AI_FALLBACK.md`
  - `docs/runbooks/AI_PERFORMANCE.md`
  - `docs/runbooks/HIGH_ERROR_RATE.md`
- **Ops / config SSoT:**
  - `monitoring/prometheus/alerts.yml`
  - `monitoring/prometheus/prometheus.yml`
  - `docs/ALERTING_THRESHOLDS.md`
  - `docs/DEPLOYMENT_REQUIREMENTS.md`
  - `docs/ENVIRONMENT_VARIABLES.md`
- **Orchestrator rollout:**
  - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` – orchestrator‑everywhere posture and Safe rollback checklist when issues are truly rules‑engine related.
