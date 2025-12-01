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

To **drop orchestrator rollout to 0%** while keeping the adapter available for diagnostics:

```bash
kubectl set env deployment/ringrift-api ORCHESTRATOR_ROLLOUT_PERCENTAGE=0
```

This forces all new sessions onto the legacy rules path without changing other flags; use this when you want to pause orchestrator usage but retain the ability to re-enable it quickly.

To switch into a **shadow-only diagnostic posture** (legacy authoritative, orchestrator in shadow for comparison):

```bash
kubectl set env deployment/ringrift-api \
  ORCHESTRATOR_ADAPTER_ENABLED=true \
  ORCHESTRATOR_ROLLOUT_PERCENTAGE=0 \
  ORCHESTRATOR_SHADOW_MODE_ENABLED=true \
  RINGRIFT_RULES_MODE=shadow
```

In this mode, all sessions run legacy rules for gameplay while the orchestrator processes the same turns in parallel for mismatch detection.

For an equivalent **developer-focused shadow-mode parity harness** using the same flag posture, see the “Shadow-mode TS↔Python parity profile (`RINGRIFT_RULES_MODE=shadow`)" example in `tests/README.md`.

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
   - Optionally run an orchestrator soak to re-check core invariants under the TS engine:
     - `npm run soak:orchestrator:smoke` (single short backend game on square8, fails on invariant violation).
     - `npm run soak:orchestrator:short` (deterministic short backend soak on square8 with multiple games, `--failOnViolation=true`; this is the concrete CI implementation of `SLO-CI-ORCH-SHORT-SOAK`).
     - For deeper offline runs, see `npm run soak:orchestrator:nightly` and [`docs/STRICT_INVARIANT_SOAKS.md`](../STRICT_INVARIANT_SOAKS.md) §2.3–2.4 for details on orchestrator soak profiles and related SLOs.
   - Optionally run a **backend HTTP load smoke** to exercise `/api` and WebSocket paths under orchestrator‑ON:
     ```bash
     TS_NODE_PROJECT=tsconfig.server.json npm run load:orchestrator:smoke
     ```
     This script:
     - Registers a small number of throwaway users via `/api/auth/register`.
     - Creates short games via `/api/games` and fetches game lists/details.
     - Samples `/metrics` for orchestrator rollout metrics.
     Use it as a quick check that backend HTTP + orchestrator wiring behave sensibly at low concurrency before increasing rollout or enabling shadow mode.
   - Optionally run a **metrics & observability smoke** to confirm `/metrics` is exposed and key orchestrator gauges are present:
     ```bash
     npm run test:e2e -- tests/e2e/metrics.e2e.spec.ts
     ```
     This Playwright spec:
     - Waits for `/ready` via `E2E_API_BASE_URL` (or `http://localhost:3000`).
     - Scrapes `/metrics` and asserts Prometheus output.
     - Verifies the presence of orchestrator metrics such as:
       - `ringrift_orchestrator_error_rate`
       - `ringrift_orchestrator_rollout_percentage`
       - `ringrift_orchestrator_circuit_breaker_state`
     Use it as a fast guardrail that observability wiring is intact before and after major orchestrator rollouts.

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

---

## 8. Step-by-step operator playbook

This section gives an end-to-end, phase-oriented playbook for orchestrator rollout
and rollback. It is a human-facing view over the environment phases and SLOs
defined in `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` (§6 and §8).

### 8.1 Pre-deploy checklist (before promoting a build)

Run this checklist **before** promoting a new build to staging or production.

1. **CI gates (SLO-CI-ORCH-PARITY)**
   - Confirm the latest `main` build has a green `orchestrator-parity` job in
     CI (see `ORCHESTRATOR_ROLLOUT_PLAN.md` §6.2):
     - `npm run test:orchestrator-parity:ts`
     - `./scripts/run-python-contract-tests.sh --verbose`
   - The `orchestrator-parity` job is documented alongside other
     orchestrator-related lanes in `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`
     under **Orchestrator Parity (TS orchestrator + Python contracts)**; treat
     that section as the CI SLO SSoT when adjusting or debugging the job.
   - If this job is red or flaky:
     - Do **not** promote the build.
     - Triage and fix tests or underlying rules changes first.

2. **Short orchestrator soak (SLO-CI-ORCH-SHORT-SOAK)**
   - On the exact commit to be deployed, ensure the canonical short soak profile has run and passed either:
     - via CI (the `orchestrator-short-soak` job in `.github/workflows/ci.yml`), or
     - locally, using:
       ```bash
       npm run soak:orchestrator:short
       ```
   - Verify (for the run corresponding to the candidate commit):
     - Exit code is `0`.
     - The short-soak summary reports `totalInvariantViolations == 0` (see
       [`docs/STRICT_INVARIANT_SOAKS.md`](../STRICT_INVARIANT_SOAKS.md) §2.3 for output details).
   - If any violation is reported:
     - Treat this as an SLO breach.
     - Do not promote the build until the invariant bug is understood and
       addressed (often via a regression test).
   - For deeper offline or scheduled runs (not required for this SLO), use:
     ```bash
     npm run soak:orchestrator:nightly
     ```
     and inspect `results/orchestrator_soak_nightly.json` for invariant summaries.

3. **Staging readiness**
   - Confirm staging configuration supports orchestrator-only posture:
     - `ORCHESTRATOR_ADAPTER_ENABLED=true`
     - `ORCHESTRATOR_ROLLOUT_PERCENTAGE=100`
     - `RINGRIFT_RULES_MODE=ts`
   - Check that staging dashboards and alerts for:
     - `ringrift_orchestrator_error_rate`
     - `ringrift_orchestrator_shadow_mismatch_rate`
     - `ringrift_orchestrator_circuit_breaker_state`
     - rules-parity metrics
       are present and show reasonable baselines.

4. **Monitoring and alert configuration**
   - Ensure the following alerts are defined and routed correctly (see
     `monitoring/prometheus/alerts.yml` and `docs/ALERTING_THRESHOLDS.md`):
     - `OrchestratorCircuitBreakerOpen`
     - `OrchestratorErrorRateWarning`
     - `OrchestratorShadowMismatches`
     - `RulesParity*` alerts
   - Confirm on-call knows which Slack channels and runbooks are linked.

5. **Feature flag / env var plan**
   - For the target environment (staging or production), decide and document:
     - Desired `ORCHESTRATOR_ROLLOUT_PERCENTAGE` for the next step.
     - Desired `ORCHESTRATOR_SHADOW_MODE_ENABLED` value.
     - Any allow/deny list changes in `OrchestratorRolloutService`.
   - Ensure you have a **rollback plan** (see §8.5) before making any change.
   - When in doubt about which combination of `NODE_ENV`, `RINGRIFT_APP_TOPOLOGY`,
     `RINGRIFT_RULES_MODE`, and orchestrator flags to use for a given phase,
     refer to the canonical presets in
     [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md` §8.1.1](../ORCHESTRATOR_ROLLOUT_PLAN.md#811-environment-and-flag-presets-by-phase).

### 8.2 Phase 1 – Staging-only orchestrator

**Goal:** Run orchestrator as the only rules path in staging and hold SLOs
steady before touching production.

**Steps:**

1. **Enable orchestrator-only in staging**
   - Set:
     ```bash
     kubectl set env deployment/ringrift-api \
       ORCHESTRATOR_ADAPTER_ENABLED=true \
       ORCHESTRATOR_ROLLOUT_PERCENTAGE=100 \
       RINGRIFT_RULES_MODE=ts \
       ORCHESTRATOR_SHADOW_MODE_ENABLED=false
     ```
   - Verify via:
     ```bash
     curl -sS APP_BASE/health
     curl -sS APP_BASE/ready
     curl -sS APP_BASE/api/admin/orchestrator/status
     ```

2. **Bake-in period**
   - Allow at least **24 hours** (preferably 24–72h) of normal staging traffic.
   - Monitor:
     - `ringrift_orchestrator_error_rate{environment="staging"}`
     - `ringrift_orchestrator_circuit_breaker_state`
     - `ringrift_orchestrator_shadow_mismatch_rate` (if you enable shadow
       experiments in staging)
     - rules-parity dashboards scoped to staging.

3. **SLO check**
   - Confirm the staging SLOs from `ORCHESTRATOR_ROLLOUT_PLAN.md` §6.3:
     - Error rate well below 0.1% over the bake-in window.
     - No `OrchestratorCircuitBreakerOpen` alerts.
     - No new invariant or game-status parity incidents.

4. **Decision**
   - If all staging SLOs are green → you may move to Phase 2 (production shadow).
   - If SLOs are breached:
     - Hold in Phase 1 and debug (rules parity, invariants, game health).
     - If necessary, roll back to a pre-orchestrator image or disable the
       adapter in staging (returning to Phase 0 posture).

### 8.3 Phase 2 – Production shadow mode

**Goal:** Run orchestrator in **shadow** alongside the legacy engine in
production, compare decisions, and observe parity and performance.

**Steps:**

1. **Enable shadow mode in production**
   - Ensure legacy remains authoritative:
     - `RINGRIFT_RULES_MODE=shadow` (or equivalent configuration)
   - Enable orchestrator in shadow:
     ```bash
     kubectl set env deployment/ringrift-api \
       ORCHESTRATOR_ADAPTER_ENABLED=true \
       ORCHESTRATOR_SHADOW_MODE_ENABLED=true
     ```

2. **Observe parity and latency**
   - Monitor:
     - `ringrift_orchestrator_shadow_mismatch_rate`
     - `ringrift_orchestrator_error_rate`
     - Any `RulesParity*` alerts.
     - Orchestrator vs legacy latency in the admin status endpoint
       (`avgLegacyLatencyMs` vs `avgOrchestratorLatencyMs`).

3. **SLO check**
   - Over at least **24 hours** of production traffic:
     - Shadow mismatch rate stays below ~0.1%, no
       `OrchestratorShadowMismatches` alerts.
     - No `RulesParityGameStatusMismatch` alerts in production.
     - Orchestrator and legacy show comparable latency.

4. **Decision**
   - If SLOs remain green → proceed to Phase 3 (incremental rollout).
   - If mismatches or parity incidents appear:
     - Disable shadow (`ORCHESTRATOR_SHADOW_MODE_ENABLED=false`) and return
       to Phase 1 posture (orchestrator-only in staging only).
     - Investigate via `RULES_PARITY.md` and contract vectors.

### 8.4 Phase 3 – Incremental production rollout

**Goal:** Move production from legacy to orchestrator as the authoritative
engine in controlled steps.

**Steps:**

1. **Confirm preconditions**
   - Phase 1 and Phase 2 have been stable.
   - No open P0/P1 incidents involving orchestrator, invariants, or rules parity.
   - On-call is aware of the planned change window.

2. **Increase rollout percentage in small steps**
   - Example sequence: 0% → 10% → 25% → 50% → 100%.
   - At each step:
     ```bash
     kubectl set env deployment/ringrift-api ORCHESTRATOR_ROLLOUT_PERCENTAGE=25
     ```
   - Wait at least one or two error windows (e.g. 15–30m) at the new level.

3. **Observe metrics and SLOs**
   - During each observation window, check:
     - `ringrift_orchestrator_error_rate`
     - HTTP 5xx rate on game endpoints.
     - `ringrift_orchestrator_circuit_breaker_state`
     - `ringrift_orchestrator_shadow_mismatch_rate` (if shadow still enabled).
     - Rules-parity and game-health dashboards.

4. **Per-step decision**
   - If metrics and SLOs are healthy → optionally advance to the next
     percentage.
   - If error rate or mismatches rise:
     - Pause further increases and hold at the current percentage.
     - For serious issues, roll back to a lower percentage (or 0%) and
       investigate per §§4–5.

5. **Reaching 100%**
   - Only move to 100% rollout when:
     - Several lower-percentage steps have been stable.
     - Orchestrator-specific and global SLOs remain comfortably within
       budgets.

### 8.5 Rollback procedure by phase

Use this as a quick map for **where to roll back to**:

- **From Phase 3 (incremental rollout)**:
  - **Minor degradation (warning alerts only):**
    - Reduce `ORCHESTRATOR_ROLLOUT_PERCENTAGE` (e.g. `50 → 10 → 0`) and
      continue monitoring.
  - **Major incident (P0/P1, circuit breaker open, invariant violations):**
    - Set:
      ```bash
      kubectl set env deployment/ringrift-api \
        ORCHESTRATOR_ROLLOUT_PERCENTAGE=0 \
        ORCHESTRATOR_ADAPTER_ENABLED=false
      ```
    - This returns production effectively to Phase 2 or Phase 1 posture,
      depending on whether you keep shadow enabled for diagnostics.

- **From Phase 2 (shadow mode issues)**:
  - If shadow mismatches or parity alerts are firing:
    - Disable shadow only:
      ```bash
      kubectl set env deployment/ringrift-api ORCHESTRATOR_SHADOW_MODE_ENABLED=false
      ```
    - Keep staging orchestrator-only and continue debugging under the
      parity runbook.

- **From Phase 1 (staging-only issues)**:
  - If staging SLOs are breached:
    - Option 1 (softer): keep orchestrator enabled but stop promoting
      builds until fixed.
    - Option 2 (harder rollback): disable orchestrator in staging and
      redeploy a known-good image, effectively returning to Phase 0.

After any rollback, follow §6 **Verification After Changes** and the relevant
incident runbooks (availability, latency, resources, rules parity).

### 8.6 Incident handling checklist (orchestrator-focused)

Use this as a quick supplement to §§4–5 when you suspect an orchestrator-specific
problem:

1. **Classify the incident**
   - Is the primary signal:
     - Global HTTP error rate (`HighErrorRate` / `ElevatedErrorRate`)?
     - Orchestrator-specific (`OrchestratorCircuitBreakerOpen`,
       `OrchestratorErrorRateWarning`, `OrchestratorShadowMismatches`)?
     - Rules parity (`RulesParity*` alerts)?
   - Use `GAME_HEALTH.md` and `AI_ARCHITECTURE.md` to distinguish:
     - Rules/orchestrator defects vs.
     - AI/inference problems vs.
     - Infra (DB/Redis/latency) issues.

2. **Check orchestrator posture**
   - Confirm current flags via `/api/admin/orchestrator/status`:
     - `adapterEnabled`
     - `rolloutPercentage`
     - `shadowModeEnabled`
   - Map the environment to Phase 1/2/3 using the environment → phase table in `ORCHESTRATOR_ROLLOUT_PLAN.md` (§8.7).

3. **Run targeted diagnostics (when safe)**
   - For suspected invariant or rules issues:
     - Run a short orchestrator soak against the current image
       (`npm run soak:orchestrator -- --boardTypes=square8 --gamesPerBoard=5 --failOnViolation=true`).
   - For suspected parity issues:
     - Consult `RULES_PARITY.md` and consider running a small parity subset
       or targeted contract vectors.

4. **Apply the appropriate rollback**
   - Use §8.5 or §§4–5 to choose:
     - Percentage reduction.
     - Shadow disablement.
     - Full adapter kill switch.
   - Avoid ad-hoc combinations; always aim to land in a clearly defined
     phase (1, 2, or 3).

5. **Close the loop**
   - After stabilising the system:
     - Ensure SLOs from `ORCHESTRATOR_ROLLOUT_PLAN.md` §6 are back within
       budget.
     - Update any incident docs under `docs/incidents/**` with:
       - Phase before/after.
       - Flag values used.
       - Whether short soaks or parity tests caught the issue.
