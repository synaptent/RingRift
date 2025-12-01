# AI Performance Runbook

> **Doc Status (2025-11-28): Active Runbook**  
> **Role:** Guide for diagnosing and reducing high AI request latency when `AIRequestHighLatency` fires, and for understanding its impact on gameplay and fallback behaviour.
>
> **SSoT alignment:** This is **derived operational guidance** over:
>
> - **Monitoring SSoT:** Prometheus alert rules in `monitoring/prometheus/alerts.yml` (alert `AIRequestHighLatency` based on `ringrift_ai_request_duration_seconds_bucket`) and scrape configuration in `monitoring/prometheus/prometheus.yml`.
> - **Metrics surfaces:** AI latency and request metrics emitted by the backend (e.g. `ringrift_ai_request_duration_seconds_bucket`, `ringrift_ai_requests_total`, `ringrift_ai_fallback_total`) and, where configured, Python AI metrics (`AI_MOVE_LATENCY`, `AI_MOVE_REQUESTS` in `ai-service/app/metrics.py`, exposed from `ai-service/app/main.py` via `/metrics`).
> - **AI integration & timeouts:** `AIServiceClient` configuration and behaviour (`src/server/services/AIServiceClient.ts`), especially `requestTimeoutMs`, concurrency limits, and circuit breaker; AI orchestration in `AIEngine` and related components.
> - **AI service implementation:** FastAPI AI service in `ai-service/app/main.py`, including difficulty profiles (`_CANONICAL_DIFFICULTY_PROFILES`) and move-selection endpoints.
> - **Incident process:** `docs/incidents/AI_SERVICE.md` and `docs/incidents/LATENCY.md` for narrative handling of AI/latency incidents.
>
> **Precedence:** Metric names, bucket definitions, thresholds, and AI timeouts are defined in **monitoring configs + backend/AI code** and may evolve. This runbook explains **how to investigate and mitigate**; it is **not** a source of truth for thresholds or semantics.
>
> For a high-level “rules vs AI vs infra” classification, see `AI_ARCHITECTURE.md` §0 (AI Incident Overview).

### Orchestrator posture & rules semantics

- The **shared TypeScript rules engine + orchestrator** (`src/shared/engine/**`, contract vectors, and orchestrator adapters) is the single source of truth for rules semantics; backend, sandbox, and Python AI-service are adapters over this SSoT.
- Key runtime flags for rules selection are: `ORCHESTRATOR_ADAPTER_ENABLED`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE`, `ORCHESTRATOR_SHADOW_MODE_ENABLED`, and `RINGRIFT_RULES_MODE`. For AI latency incidents, these should normally stay in the **orchestrator‑ON** posture; legacy / SHADOW modes are for diagnostics only.
- **Do not respond to slow AI by changing game rules or flipping back to legacy rules engines.** Investigate AI infrastructure, models, difficulty profiles, or integration first; rules semantics remain anchored in the shared TS engine + contracts.
- If symptoms look like **rules-engine or orchestrator issues** (illegal moves accepted, incorrect scoring/turn sequencing) rather than “AI is slow”, treat this as a **rules/orchestrator incident** and:
  - Switch to `docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md` for rollout/flag levers and environment phases.
  - Use `docs/runbooks/RULES_PARITY.md` and contract vectors for TS↔Python parity checks.
  - Refer back to `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` (Safe rollback checklist) instead of continuing with AI-only tuning in this runbook.

---

## 1. When This Alert Fires

**Alert (from `monitoring/prometheus/alerts.yml`, `ai-service` group):**

- `AIRequestHighLatency` (severity: `warning`)

**Conceptual condition (do not edit thresholds here; see `alerts.yml`):**

```promql
histogram_quantile(
  0.99,
  sum(rate(ringrift_ai_request_duration_seconds_bucket[5m])) by (le)
) > 5
```

This means:

- The **99th percentile** latency for AI requests observed by the backend exceeds the configured threshold (on the order of seconds).
- AI move generation is **slow for a non-trivial tail** of requests.

**Impact (expected):**

- Games vs AI feel sluggish; players wait several seconds for AI moves.
- If latency crosses Node’s `AIServiceClient` timeout, requests may fail and **trigger fallback**, raising `AIFallbackRateHigh` / `AIFallbackRateCritical`.
- Prolonged high latency can contribute to **elevated error rates** and overload in the Node process if requests pile up.

**Common correlations:**

- `AIFallbackRateHigh` / `AIFallbackRateCritical` (timeouts causing fallbacks).
- `AIErrorsIncreasing` (long-running operations failing).
- Resource alerts on the AI host (CPU/memory) if monitored.
- `HighP99Latency` / `HighGameMoveLatency` if AI latency impacts overall game responsiveness.

---

## 2. Quick Triage (First 5–10 Minutes)

> Goal: Confirm that AI latency is genuinely high, understand whether it is capacity-, model-, or configuration-driven, and check for immediate user impact.

### 2.1 Confirm alert and check for related AI signals

In Alertmanager / your monitoring UI:

1. Confirm `AIRequestHighLatency` is active and note:
   - Environment (staging vs production).
   - Start time and duration.
2. Check for related AI alerts in the same time window:
   - `AIFallbackRateHigh` / `AIFallbackRateCritical`.
   - `AIServiceDown`.
   - `AIErrorsIncreasing`.
3. Check general latency & availability:
   - `HighP99Latency`, `HighGameMoveLatency` (see `HIGH_LATENCY.md`, `GAME_PERFORMANCE.md`).
   - `ServiceDegraded` / `ServiceOffline` if AI issues have triggered broader degradation.

### 2.2 Inspect AI latency metrics in Prometheus

In Prometheus, start with the same metric used by the alert (tune windows to match `alerts.yml`):

```promql
# Overall AI P99 latency
histogram_quantile(
  0.99,
  sum(rate(ringrift_ai_request_duration_seconds_bucket[5m])) by (le)
)

# Optionally inspect other quantiles
histogram_quantile(
  0.95,
  sum(rate(ringrift_ai_request_duration_seconds_bucket[5m])) by (le)
)

histogram_quantile(
  0.50,
  sum(rate(ringrift_ai_request_duration_seconds_bucket[5m])) by (le)
)
```

Questions to answer:

- Is high latency **tail-only** (P99) or also affecting P95/P50?
- Is this a **short spike** (e.g. deployment, burst traffic) or **sustained** over multiple alert windows?

If `ringrift_ai_request_duration_seconds_bucket` carries labels (e.g. `difficulty`, `ai_type`), you can further slice latency:

```promql
# P99 latency by difficulty (if labelled)
histogram_quantile(
  0.99,
  sum(rate(ringrift_ai_request_duration_seconds_bucket[5m])) by (le, difficulty)
)

# P99 latency by AI type (if labelled)
histogram_quantile(
  0.99,
  sum(rate(ringrift_ai_request_duration_seconds_bucket[5m])) by (le, ai_type)
)
```

Use these to identify whether particular **difficulty levels** or **AI engines** (MINIMAX/MCTS/DESCENT) are responsible.

### 2.3 Check app readiness and AI dependency status

From an operator shell against the Node backend:

```bash
# Liveness
curl -sS APP_BASE/health | jq . || curl -sS APP_BASE/health

# Readiness including AI via ServiceStatusManager
curl -sS APP_BASE/ready | jq . || curl -sS APP_BASE/ready
```

- If the AI dependency is marked `unhealthy` or if `ringrift_service_status{service="ai_service"} == 0`, treat `AIServiceDown` as primary and follow `AI_SERVICE_DOWN.md` alongside this runbook.
- If AI is `healthy` but latency is elevated, the service is up but **slow**; continue below.

### 2.4 Quick resource and health check on the AI service (docker‑compose example)

```bash
cd /path/to/ringrift

# AI service container status and logs
docker compose ps ai-service
docker compose logs --tail 200 ai-service

# Health endpoint
docker compose exec ai-service curl -sS http://localhost:8001/health | jq . \
  || docker compose exec ai-service curl -sS http://localhost:8001/health

# Resource snapshot (CPU, memory)
docker stats --no-stream ai-service
```

If `/health` is not `{"status": "healthy"}` or logs show repeated slow operations / timeouts, you likely have a **capacity or model performance** issue.

### 2.5 AI triage checklist (metrics, saturation, semantics)

- **Quantify latency and fallback**
  - Inspect `ringrift_ai_request_duration_seconds_bucket` for P50/P95/P99 against historical baselines to confirm how severe the slowdown is.
  - Check `ringrift_ai_fallback_total` vs `ringrift_ai_requests_total` to see whether timeouts are already triggering fallbacks (`AIFallbackRateHigh`).
- **Check AI-service health and resources**
  - Confirm `/health` on `ai-service` and readiness on the Node app; inspect container/host CPU and memory via `docker stats` or host dashboards.
  - If the Python service is healthy but saturated, prioritise capacity and difficulty‑profile tuning before code‑level changes.
- **Validate semantics against contract vectors when behaviour looks “wrong”, not just slow**
  - If players report obviously illegal or nonsensical moves, validate the shared‑engine contract vectors:
    - TS: `tests/contracts/contractVectorRunner.test.ts`
    - Python: `ai-service/tests/contracts/test_contract_vectors.py`
  - Treat these contracts and `.shared` rules tests as the authority for rules; AI latency incidents should not drive changes to canonical rules semantics.
- **Choose the remediation path**
  - Use the diagnosis in §3 (capacity, timeouts, Python behaviour, model/config regressions) together with the above signals to decide whether to:
    - Scale out or up the AI service,
    - Reduce effective think time / complexity at higher difficulties,
    - Tune Node timeouts and concurrency limits,
    - Or temporarily lean on documented fallbacks (`AI_FALLBACK.md`, `AI_SERVICE_DOWN.md`) while preserving rules semantics.

---

## 3. Deep Diagnosis

### 3.1 Capacity and traffic profile

First, check whether AI workloads have increased:

```promql
# Overall AI request rate
sum(rate(ringrift_ai_requests_total[10m]))

# Optional: rate by difficulty (if labelled)
sum(rate(ringrift_ai_requests_total[10m])) by (difficulty)
```

Questions:

- Has AI traffic **spiked** relative to typical baselines (events, promotions, tests)?
- Are high difficulties (e.g. 7–10, MCTS/DESCENT) dominating the load?
- Are particular board types or scenarios significantly more frequent?

### 3.2 Node‑side timeouts and overload signals

`AIServiceClient` enforces a per‑request timeout (`config.aiService.requestTimeoutMs`) and a Node‑local concurrency cap. These can drive perceived latency and fallback.

From app logs:

```bash
cd /path/to/ringrift

docker compose logs --tail 500 app 2>&1 \
  | grep -Ei 'AI Service|AI move|aiErrorType|AI_SERVICE_' \
  | tail -n 100
```

Look for:

- `aiErrorType: "timeout"` → AI requests **exceed Node’s timeout**; may drive `AI_SERVICE_TIMEOUT` and fallbacks.
- `AI Service concurrency limit reached, rejecting request` → local **concurrency cap** reached, even if AI host could handle more.
- Circuit breaker messages (`Circuit breaker opened after repeated failures`) → persistent high latency or errors causing Node to back off.

If timeouts / overload are the dominant cause, consider **capacity** and **timeout tuning** (see 4.2, 4.3) rather than immediately changing alert thresholds.

### 3.3 Python AI service behaviour

Inspect what the AI service is doing when handling `/ai/move`:

```bash
cd /path/to/ringrift

# Detailed AI logs
docker compose logs --tail 500 ai-service 2>&1 \
  | grep -Ei 'AI move|duration|time=|Error generating AI move|Exception' \
  | tail -n 100

# Optional: metrics directly from Python (if wired into your Prometheus stack)
curl -sS http://localhost:8001/metrics | grep -E 'AI_MOVE_LATENCY|AI_MOVE_REQUESTS' | head
```

The FastAPI handler in `ai-service/app/main.py`:

- Uses difficulty profiles (`_CANONICAL_DIFFICULTY_PROFILES`) to select AI engine and `think_time_ms`.
- Emits `AI_MOVE_LATENCY` and `AI_MOVE_REQUESTS` metrics per AI type and difficulty.
- Logs move generation time and evaluation.

Look for:

- `AI move: ... time=XYZms` showing **much higher times** than usual.
- Repeated `Error generating AI move` exceptions that correlate with long-running or retried operations.
- Changes in difficulty profiles or model usage that might explain longer inference.

### 3.4 Model / configuration regressions

If AI latency issues start after a model or configuration change:

- Check recent AI deployments and model upgrades (per your deployment logs / `AI_IMPROVEMENT_PLAN.md`, `AI_ASSESSMENT_REPORT.md`).
- Confirm that model sizes, think times, and heuristics are consistent with expectations documented in AI design docs.
- Verify environment variables used by AI (e.g. model paths, device selection) against `docs/ENVIRONMENT_VARIABLES.md`.

---

## 4. Remediation

> Goal: Bring AI latency back within expected bounds by adjusting capacity, configuration, or models—**not** by weakening alert thresholds.

### 4.1 Address obvious service health or dependency issues

If `AIServiceDown` or other severe alerts are also firing:

- Prioritise restoring **basic health** of the AI service (follow `AI_SERVICE_DOWN.md`).
- Once `/health` is stable and `ringrift_service_status{service="ai_service"} == 1`, revisit latency metrics to see if they normalise.

If DB/Redis/other dependencies of Node are degraded, they may indirectly slow AI orchestration; follow `DATABASE_PERFORMANCE.md`, `REDIS_PERFORMANCE.md`, and general latency runbooks (`HIGH_LATENCY.md`).

### 4.2 Capacity and scaling options

If metrics and logs indicate **high traffic + resource saturation**:

1. **Verify resource pressure on the AI container/host:**

   ```bash
   docker stats --no-stream ai-service
   docker compose exec ai-service head -40 /proc/meminfo
   ```

2. **Mitigation options (via your standard deployment/infra process):**
   - **Scale out** AI service instances (horizontal scaling) if your environment supports multiple `ai-service` replicas behind a load balancer.
   - Allocate more CPU/memory to the AI container where feasible.
   - If using heavier engines (e.g. MCTS/DESCENT at high difficulty), consider temporarily reducing their usage (feature flags, difficulty caps, or matchmaking choices).

3. **Coordinate with product / game design:**
   - For short-term relief, limit extremely high difficulty games or large board types that incur the highest cost.
   - If tournaments or special modes drive load, schedule them with capacity in mind.

### 4.3 Tune timeouts and think times safely

AI performance is a balance between **player experience** and **AI strength**:

- Node timeouts (`config.aiService.requestTimeoutMs`) define when the backend gives up and falls back.
- Python difficulty profiles define `think_time_ms` and AI engine choice per difficulty.

Mitigations (implemented via code/config + normal review/CI):

- Ensure `requestTimeoutMs` is **greater than** typical AI P95 latency, but tight enough to avoid hanging the game.
- If P99 / worst-case latency is consistently high due to model cost, consider **reducing `think_time_ms`** for one or more difficulty levels in `_CANONICAL_DIFFICULTY_PROFILES` (documenting changes in AI design docs).
- Avoid increasing timeout so far that user experience is obviously degraded; adjust both service capacity and timeouts together where possible.

Any such changes should be reviewed and tested; do **not** hand-edit timeouts in production containers.

### 4.4 Optimise or roll back AI model / code changes

If AI latency increased **after a model or code change**:

- Evaluate reverting to the previous, known-good model or configuration for the affected difficulty levels.
- Profile AI inference paths (using existing offline tooling / scripts in `ai-service/scripts/*.py`) to identify hotspots.
- Where feasible, apply optimisations (e.g. reduced search depth, caching within AI) behind feature flags and roll them out gradually.

Coordinate with the AI owners, referencing `AI_IMPROVEMENT_PLAN.md`, `AI_TRAINING_AND_DATASETS.md`, and any recent experimental notes.

### 4.5 Coordinate with fallback and error-rate mitigation

High latency often **feeds** other alerts:

- Timeouts → `AIFallbackRateHigh` / `AIFallbackRateCritical` (see `AI_FALLBACK.md`).
- Long-running operations that then fail → `AIErrorsIncreasing` (see `AI_ERRORS.md`).
- End-to-end request/game latency → `HIGH_LATENCY.md`, `GAME_PERFORMANCE.md`.

Treat these runbooks as complementary:

- Use **this** runbook to reduce **raw AI latency**.
- Use `AI_FALLBACK.md` to manage the impact of fallbacks and ensure the game remains playable.
- Use `AI_ERRORS.md` to fix underlying AI logic errors that may indirectly increase latency (e.g. retries, repeated failures).

---

## 5. Validation

You are done when:

### 5.1 Metrics / alerts

- [ ] `AIRequestHighLatency` has **cleared** and remains clear for at least one full evaluation window.
- [ ] AI latency quantiles have returned to expected baselines for your environment:

  ```promql
  histogram_quantile(
    0.99,
    sum(rate(ringrift_ai_request_duration_seconds_bucket[5m])) by (le)
  )
  ```

- [ ] If previously firing, `AIFallbackRateHigh` / `AIFallbackRateCritical` and `AIErrorsIncreasing` have stabilised or cleared.

### 5.2 Health / readiness and player experience

- [ ] `/health` and `/ready` report healthy for the AI dependency, with no ongoing error annotations.
- [ ] Manual AI game smoke tests (login → start AI game → multiple moves) show:
  - AI moves arriving within acceptable time (aligned with design expectations per difficulty).
  - No frequent fallbacks or visible AI failures.

### 5.3 Post‑incident follow‑up

- [ ] If this was a noticeable incident, capture a short write‑up using `docs/incidents/POST_MORTEM_TEMPLATE.md` and link it from `docs/incidents/AI_SERVICE.md`.
- [ ] If capacity or timeout settings were changed, update any relevant documentation (`docs/ENVIRONMENT_VARIABLES.md`, `docs/DEPLOYMENT_REQUIREMENTS.md`, AI design docs) so they remain the SSoT for expected behaviour.
- [ ] Consider adding or refining dashboards that show AI latency alongside fallback fraction and AI error rate.

---

## 6. Related Documentation

- **Incident docs:**
  - `docs/incidents/AI_SERVICE.md`
  - `docs/incidents/LATENCY.md`
  - `docs/incidents/AVAILABILITY.md`
  - `docs/incidents/RESOURCES.md`
  - `docs/incidents/TRIAGE_GUIDE.md`
- **Related runbooks:**
  - `docs/runbooks/AI_SERVICE_DOWN.md`
  - `docs/runbooks/AI_FALLBACK.md`
  - `docs/runbooks/AI_ERRORS.md`
  - `docs/runbooks/HIGH_LATENCY.md`
- **Ops / config SSoT:**
  - `monitoring/prometheus/alerts.yml`
  - `monitoring/prometheus/prometheus.yml`
  - `docs/ALERTING_THRESHOLDS.md`
  - `docs/DEPLOYMENT_REQUIREMENTS.md`
  - `docs/ENVIRONMENT_VARIABLES.md`
- **Orchestrator rollout:**
  - `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` – orchestrator‑everywhere posture and Safe rollback checklist when issues are truly rules‑engine related.
