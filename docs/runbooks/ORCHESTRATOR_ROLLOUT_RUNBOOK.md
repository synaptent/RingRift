# Orchestrator Rollout Runbook

> **Audience:** On-call engineers and backend maintainers  
> **Scope:** Managing rollout, shadow mode, and incident response for the shared turn orchestrator

---

## 1. Quick Status Checklist

When investigating an orchestrator-related page or alert, check:

- `ringrift_orchestrator_circuit_breaker_state` (0 = closed, 1 = open)
- `ringrift_orchestrator_error_rate` (0.0–1.0 fraction)
- `ringrift_orchestrator_sessions_total{engine,selection_reason}`
- `ringrift_orchestrator_shadow_mismatch_rate`
- Admin API snapshot:
  - `GET /api/admin/orchestrator/status` (requires admin auth)

**Target SLO-style thresholds (steady state):**

- Orchestrator error rate: `ringrift_orchestrator_error_rate < 0.02` (2%) over a 5–10 minute window.
- Shadow mismatch rate: `ringrift_orchestrator_shadow_mismatch_rate < 0.01` (1%) when SHADOW is enabled.
- Circuit breaker closed: `ringrift_orchestrator_circuit_breaker_state == 0` during normal operation.

Use `/metrics` plus the admin API to decide whether to:

- Increase / pause / roll back rollout percentage.
- Enable / disable shadow mode.
- Trip or reset the circuit breaker.

---

## 2. Reading the Admin Status Endpoint

**Endpoint:** `GET /api/admin/orchestrator/status`  
**Auth:** Bearer token, user with `role = "admin"`

Response shape (summary):

```json
{
  "success": true,
  "data": {
    "config": {
      "adapterEnabled": true,
      "rolloutPercentage": 50,
      "shadowModeEnabled": false,
      "allowlistUsers": ["staff-1"],
      "denylistUsers": [],
      "circuitBreaker": {
        "enabled": true,
        "errorThresholdPercent": 5,
        "errorWindowSeconds": 300
      }
    },
    "circuitBreaker": {
      "isOpen": false,
      "errorCount": 3,
      "requestCount": 200,
      "windowStart": "2025-11-28T12:34:56.000Z",
      "errorRatePercent": 1.5
    },
    "shadow": {
      "totalComparisons": 1200,
      "matches": 1188,
      "mismatches": 12,
      "mismatchRate": 0.01,
      "orchestratorErrors": 0,
      "orchestratorErrorRate": 0,
      "avgLegacyLatencyMs": 8.5,
      "avgOrchestratorLatencyMs": 9.2
    }
  }
}
```

Key questions:

- **Is the circuit breaker open?** (`circuitBreaker.isOpen === true`)
- **Is the error rate elevated?** (`errorRatePercent > 2–5%`)
- **Are shadow mismatches non-trivial?** (`shadow.mismatchRate > 0.01`)

If any of these are true, stop rollout increases and follow the incident steps below.

---

## 3. Normal Rollout Operations

### 3.1 Increasing Rollout Percentage

**Goal:** Gradually move traffic from legacy to orchestrator.

1. Confirm error and mismatch rates are low:
   - `ringrift_orchestrator_error_rate < 0.02`
   - `ringrift_orchestrator_shadow_mismatch_rate` near 0 (ideally < 0.01)
   - Circuit breaker closed: `ringrift_orchestrator_circuit_breaker_state == 0`
2. Adjust environment (e.g. Kubernetes):

```bash
kubectl set env deployment/ringrift-api ORCHESTRATOR_ROLLOUT_PERCENTAGE=25
```

3. Wait at least one error window (default 5 minutes) plus a few minutes of traffic.
4. Re-check:
   - `/metrics` for error rate and breaker state.
   - `/api/admin/orchestrator/status` for shadow metrics.
5. Repeat for 10% → 25% → 50% → 100% as confidence allows.

### 3.2 Enabling / Disabling Shadow Mode

**Shadow mode** runs both engines but keeps legacy authoritative.

- Enable shadow mode:

```bash
kubectl set env deployment/ringrift-api ORCHESTRATOR_SHADOW_MODE_ENABLED=true
```

- Disable shadow mode:

```bash
kubectl set env deployment/ringrift-api ORCHESTRATOR_SHADOW_MODE_ENABLED=false
```

Use shadow mode when:

- You want to validate orchestrator semantics under real traffic.
- You are preparing for a rollout increase or post-incident regression testing.

> **CI / Pre‑prod Profiles:** Gating CI jobs (unit/coverage, TS rules‑engine, and E2E)
> now run with `RINGRIFT_RULES_MODE=ts`,
> `ORCHESTRATOR_ADAPTER_ENABLED=true`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE=100`,
> and `ORCHESTRATOR_SHADOW_MODE_ENABLED=false`, treating the shared TS engine
>
> - orchestrator as the sole rules path. Any use of `RINGRIFT_RULES_MODE=shadow`
>   or `ORCHESTRATOR_SHADOW_MODE_ENABLED=true` should be confined to explicit
>   diagnostic jobs and post‑deploy parity experiments, not to CI gates.

---

## 4. Incident Scenarios & Actions

### 4.1 Alert: OrchestratorCircuitBreakerOpen

**Signal:**  
Prometheus alert `OrchestratorCircuitBreakerOpen` fired.  
Metric: `ringrift_orchestrator_circuit_breaker_state == 1`.

**Impact:**  
New orchestrator traffic is effectively disabled; legacy engine is used instead.

**Actions:**

1. Confirm via admin API:
   - `GET /api/admin/orchestrator/status` → `circuitBreaker.isOpen === true`.
2. Inspect:
   - `ringrift_orchestrator_error_rate`
   - Application logs around orchestrator adapter (`GameEngine.processMoveViaAdapter`) for repeated errors.
3. Short-term:
   - Leave `ORCHESTRATOR_ADAPTER_ENABLED=true` but keep rollout stable.
   - Do **not** increase `ORCHESTRATOR_ROLLOUT_PERCENTAGE`.
4. Remediation:
   - Identify and fix underlying orchestrator issue (e.g. specific move types, board types, or phases).
   - Once fixed and deployed, manually reset the breaker via an admin task (if added) or by restarting the API pods.
5. Post-fix:
   - Monitor `ringrift_orchestrator_error_rate` returning to near zero.
   - Confirm breaker state closes:
     - `ringrift_orchestrator_circuit_breaker_state == 0`.

### 4.2 Alert: OrchestratorErrorRateWarning

**Signal:**  
`ringrift_orchestrator_error_rate > 0.02` for > 2m.

**Actions:**

1. Pause rollout increases; keep `ORCHESTRATOR_ROLLOUT_PERCENTAGE` constant.
2. Check logs for adapter or orchestrator-level errors.
3. If error rate approaches threshold configured in `ORCHESTRATOR_ERROR_THRESHOLD_PERCENT`:
   - Consider manually lowering rollout (e.g. 50% → 10%).
   - If severe, temporarily set `ORCHESTRATOR_ADAPTER_ENABLED=false` (kill switch).

### 4.3 Alert: OrchestratorShadowMismatches

**Signal:**  
`ringrift_orchestrator_shadow_mismatch_rate > 0.01` for > 5m.

**Actions:**

1. Identify mismatch patterns via:
   - Logs from `ShadowModeComparator` (`ENGINE MISMATCH` entries).
2. Short-term:
   - Stop rollout increases until mismatch rate returns to near zero.
3. Mid-term:
   - Add or extend parity/contract tests for the mismatched scenarios.
   - Fix orchestrator or legacy engine divergence, then redeploy and re-run shadow.

---

## 5. Emergency Rollback Procedures

### 5.1 Immediate Kill Switch

Use when orchestrator behaviour is clearly wrong (e.g. incorrect winners, crashes).

```bash
kubectl set env deployment/ringrift-api ORCHESTRATOR_ADAPTER_ENABLED=false
```

Effect:

- New sessions will use the legacy engine only.
- Existing games running on the orchestrator may continue until they complete or the pods restart.

### 5.2 Gradual Rollback

Use when error rates or mismatches are elevated but not catastrophic.

```bash
kubectl set env deployment/ringrift-api ORCHESTRATOR_ROLLOUT_PERCENTAGE=10
```

Then monitor:

- `ringrift_orchestrator_error_rate`
- `ringrift_orchestrator_shadow_mismatch_rate`
- HTTP/E2E SLOs (latency, error rates).

---

## 6. Verification After Changes

After any rollout or rollback action:

1. Confirm configuration:
   - `GET /api/admin/orchestrator/status`
   - Environment variables in deployment (`ORCHESTRATOR_*`).
2. Confirm metrics:
   - `/metrics`:
     - `ringrift_orchestrator_rollout_percentage`
     - `ringrift_orchestrator_circuit_breaker_state`
     - `ringrift_orchestrator_error_rate`
     - `ringrift_orchestrator_shadow_mismatch_rate`
3. Run a small set of smoke tests:
   - Create a few games with AI/human players.
   - Exercise capture, lines, territory, and victory conditions.
   - Verify no spike in 5xx or orchestrator errors.

---

## 7. References

- **Design & Feature Flags:**  
  `docs/drafts/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md`

- **Shared Engine & Adapters:**
  - `src/shared/engine/orchestration/turnOrchestrator.ts`
  - `src/server/game/turn/TurnEngineAdapter.ts`
  - `src/client/sandbox/SandboxOrchestratorAdapter.ts`

- **Rollout & Shadow Services:**
  - `src/server/services/OrchestratorRolloutService.ts`
  - `src/server/services/ShadowModeComparator.ts`

- **Metrics:**
  - `src/server/services/MetricsService.ts`
  - `monitoring/prometheus/alerts.yml`
