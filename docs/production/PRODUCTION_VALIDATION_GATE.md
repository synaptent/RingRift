# Production Validation Gate Checklist

> **Doc Status (2025-12-23): Active**
>
> **Purpose:** Executable checklist for RingRift v1.0 production release validation. This consolidates all validation criteria from the PV-01 through PV-11 production validation remediation work.
>
> **Last Validated:** 2025-12-23
>
> **References:**
>
> - [`PRODUCTION_VALIDATION_REMEDIATION_PLAN.md`](../planning/PRODUCTION_VALIDATION_REMEDIATION_PLAN.md) - Full remediation details
> - [`PRODUCTION_READINESS_CHECKLIST.md`](./PRODUCTION_READINESS_CHECKLIST.md) - Comprehensive readiness checklist
> - [`PROJECT_GOALS.md`](../../PROJECT_GOALS.md) §4.1, §4.4 - SLO requirements
> - [`SLO_GATE_CI_RUNBOOK.md`](../operations/SLO_GATE_CI_RUNBOOK.md) - Automated SLO gate CI/CD integration

---

## Quick Reference: Validation Status Summary

| Category              | Status     | Key Metric                              |
| --------------------- | ---------- | --------------------------------------- |
| Pre-Deploy Tests      | ✅ PASS    | All test suites passing                 |
| Load Test Validation  | ✅ PASS    | 14.68ms p95, 0% true errors             |
| AI Service Validation | ✅ PASS    | 10s recovery time                       |
| Observability         | ⚠️ PARTIAL | Grafana works, web app not instrumented |
| Security              | ✅ PASS    | Rate limiting, TLS configured           |
| Infrastructure        | ✅ PASS    | Docker builds successful                |

---

## 1. Pre-Deploy Checklist

Complete these validation steps before any deployment.

### 1.1 Core Test Suites

| Check                     | Command                            | Status | Notes                    |
| ------------------------- | ---------------------------------- | ------ | ------------------------ |
| Unit tests passing        | `npm test`                         | [ ]    | 2,987+ TS tests expected |
| Core profile passing      | `npm run test:core`                | [ ]    | Fast PR gate profile     |
| Contract tests passing    | `npm run test:orchestrator-parity` | [ ]    | 90 contract vectors      |
| Integration tests passing | `npm run test:integration`         | [ ]    | API and WebSocket tests  |
| E2E tests passing         | `npm run test:e2e`                 | [ ]    | Playwright suite         |

### 1.2 Python AI Service Tests

| Check                | Command                                                  | Status | Notes                 |
| -------------------- | -------------------------------------------------------- | ------ | --------------------- |
| Python tests passing | `cd ai-service && pytest`                                | [ ]    | 1,824+ tests expected |
| Parity tests passing | `cd ai-service && pytest tests/parity/ tests/contracts/` | [ ]    | TS/Python parity      |

### 1.3 Build Verification

| Check                | Command                       | Status | Notes                  |
| -------------------- | ----------------------------- | ------ | ---------------------- |
| TypeScript builds    | `npm run build`               | [ ]    | No compilation errors  |
| Docker image builds  | `docker compose build`        | [ ]    | All services build     |
| Staging stack starts | `./scripts/deploy-staging.sh` | [ ]    | All containers healthy |

**PV Reference:** PV-01 through PV-05 validated these prerequisites.

---

## 2. Load Test Validation

### 2.1 Baseline Load Test (20G/60P)

**Scenario:** `BCAP_STAGING_BASELINE_20G_60P`

| Metric                     | Target  | Last Result (PV-07) | Status  |
| -------------------------- | ------- | ------------------- | ------- |
| HTTP p95 latency           | < 800ms | 12.74ms             | ✅ PASS |
| True error rate            | < 1%    | 0%                  | ✅ PASS |
| Rate limit bypass verified | 0 hits  | ✅ Working          | ✅ PASS |
| Contract failures          | 0       | 0                   | ✅ PASS |
| Lifecycle mismatches       | 0       | 0                   | ✅ PASS |

**Execution Commands:**

```bash
# Configure environment
export BASE_URL=http://localhost:3000
export WS_URL=ws://localhost:3001
export RATE_LIMIT_BYPASS_ENABLED=true
export RATE_LIMIT_BYPASS_TOKEN="<staging_bypass_token>"

# Run preflight check
npm run load:preflight -- --expected-vus 60

# Run baseline scenario
SEED_LOADTEST_USERS=true tests/load/scripts/run-baseline.sh staging

# Verify SLOs
npm run slo:verify tests/load/results/<baseline_file>.json -- --env staging
```

**Checklist:**

- [ ] Preflight health check passes
- [ ] Load test completes without k6 errors
- [ ] HTTP p95 < 800ms
- [ ] True error rate < 1%
- [ ] No contract failures
- [ ] No lifecycle mismatches
- [ ] Rate limit bypass working (0 429 responses from load test users)

**PV Reference:** PV-07 - Baseline validation passed 2025-12-20

---

### 2.2 Target-Scale Validation (100G/300P) - REQUIRED FOR v1.0

**Scenario:** `BCAP_SQ8_3P_TARGET_100G_300P`

| Metric               | Target  | Last Result (PV-08) | Status               |
| -------------------- | ------- | ------------------- | -------------------- |
| HTTP p95 latency     | < 500ms | 14.68ms             | ✅ PASS (97% margin) |
| HTTP p90 latency     | -       | 11.37ms             | ✅                   |
| True error rate      | < 0.5%  | 0%                  | ✅ PASS              |
| Rate limit hits      | 0       | 0                   | ✅ PASS              |
| Contract failures    | 0       | 0                   | ✅ PASS              |
| Lifecycle mismatches | 0       | 0                   | ✅ PASS              |
| All checks           | 100%    | 100% (48,917)       | ✅ PASS              |
| Game state checks    | 100%    | 100% (16,238)       | ✅ PASS              |

**Execution Commands:**

```bash
# Run target-scale scenario
SEED_LOADTEST_USERS=true tests/load/scripts/run-target-scale.sh staging

# Verify SLOs with PRODUCTION thresholds
npm run slo:verify tests/load/results/<target_file>.json -- --env production
```

**Checklist:**

- [ ] Load test completes at 100+ VUs for 10+ minutes
- [ ] HTTP p95 < 500ms
- [ ] True error rate < 0.5%
- [ ] 100% check success rate
- [ ] All containers stable (no restarts)
- [ ] Resource usage within limits (CPU < 80%, Memory < 80%)

**PV Reference:** PV-08 - Target-scale validation passed 2025-12-20

---

### 2.3 AI-Heavy Validation (Optional for v1.0)

**Scenario:** `BCAP_SQ8_4P_AI_HEAVY_75G_300P`

| Metric                 | Target   | Last Result (PV-09) | Status  |
| ---------------------- | -------- | ------------------- | ------- |
| WebSocket move RTT p95 | < 2000ms | 11ms                | ✅ PASS |
| WebSocket move success | > 95%    | 100%                | ✅ PASS |
| WS connection success  | > 99%    | 100%                | ✅ PASS |
| WS stalls              | < 10     | 0                   | ✅ PASS |
| True error rate        | < 0.5%   | 0%                  | ✅ PASS |

**Known Gap:** Python AI service was not exercised during the test. Heuristic AI runs locally in TypeScript. This is acceptable for v1.0.

**Checklist:**

- [ ] WebSocket move RTT p95 < 2000ms
- [ ] Move success rate > 95%
- [ ] Connection success rate > 99%
- [ ] No move stalls

**PV Reference:** PV-09 - WebSocket validation passed, AI service exercise deferred

---

## 3. AI Service Validation

### 3.1 AI Service Health

| Check                        | Method                              | Status |
| ---------------------------- | ----------------------------------- | ------ |
| AI health endpoint responds  | `curl http://localhost:8001/health` | [ ]    |
| AI service in Docker healthy | `docker ps` shows healthy           | [ ]    |
| No AI service restarts       | Container uptime stable             | [ ]    |

### 3.2 AI Degradation Drill (PV-10)

**Drill phases validated:**

| Phase    | Expected Behavior                           | Last Result                  | Status  |
| -------- | ------------------------------------------- | ---------------------------- | ------- |
| Baseline | All dependencies healthy                    | ✅ All checks pass           | ✅ PASS |
| Degraded | App detects AI down, shows "degraded"       | ✅ Status correctly reported | ✅ PASS |
| Recovery | AI service recovers, app returns to healthy | ✅ 10 second recovery        | ✅ PASS |

**Drill Execution:**

```bash
# Run automated drill
npx ts-node scripts/run-ai-degradation-drill.ts

# Manual drill steps:
# 1. Verify baseline: curl http://localhost:3000/ready
# 2. Stop AI: docker compose stop ai-service
# 3. Verify degraded: curl http://localhost:3000/ready (status: "degraded")
# 4. Restart AI: docker compose start ai-service
# 5. Verify recovery: curl http://localhost:3000/ready (status: "healthy")
```

**Checklist:**

- [ ] Baseline phase passes (all dependencies healthy)
- [ ] Degraded phase detected (AI shows as degraded, not unhealthy)
- [ ] No cascading failures (DB and Redis remain healthy)
- [ ] Recovery completes in < 60 seconds
- [ ] Core app remains available during AI outage

**PV Reference:** PV-10 - AI degradation drill passed 2025-12-20 (10s recovery time)

---

## 4. Observability Validation

### 4.1 Grafana Dashboard Access

| Check              | URL                                 | Status |
| ------------------ | ----------------------------------- | ------ |
| Grafana accessible | http://localhost:3002               | [ ]    |
| Login works        | admin credentials from .env.staging | [ ]    |
| Dashboards load    | 10 dashboards available             | [ ]    |

### 4.2 Prometheus Targets

| Target                | Expected Status  | Notes            |
| --------------------- | ---------------- | ---------------- |
| prometheus (self)     | Up               | Self-monitoring  |
| node-exporter-cluster | Up               | System metrics   |
| ringrift-p2p-cluster  | Up/Down (varies) | AI cluster nodes |

### 4.3 Dashboard Data Status

| Dashboard                   | Data Status  | Notes                              |
| --------------------------- | ------------ | ---------------------------------- |
| RingRift AI Cluster         | ✅ Live data | Self-play metrics, GPU utilization |
| RingRift - Game Performance | ⚠️ Partial   | AI cluster data only               |
| RingRift - System Health    | ⚠️ No data   | Web app not instrumented           |
| Integration Health          | ⚠️ Partial   | Some metrics available             |

**Known Gap (Non-Blocking for v1.0):**

- Web app not instrumented with Prometheus metrics
- System Health dashboard shows "No data" for HTTP/WebSocket metrics
- k6 load test results provide required SLO data as alternative

**Checklist:**

- [ ] Grafana accessible at configured URL
- [ ] At least 3 dashboards load without errors
- [ ] AI Cluster dashboard shows live metrics
- [ ] Prometheus receiving metrics from AI cluster

**PV Reference:** PV-11 - Dashboard validation passed 2025-12-20

---

## 5. Security Validation

### 5.1 Rate Limiting Configuration

| Setting                     | Production Value | Verification                |
| --------------------------- | ---------------- | --------------------------- |
| `RATE_LIMIT_BYPASS_ENABLED` | `false`          | ⚠️ CRITICAL                 |
| `RATE_LIMIT_BYPASS_TOKEN`   | (not set)        | Must be unset in production |
| Rate limiting active        | Enabled          | All endpoints protected     |

**Checklist:**

- [ ] `RATE_LIMIT_BYPASS_ENABLED=false` in production environment
- [ ] `RATE_LIMIT_BYPASS_TOKEN` is NOT set in production
- [ ] Rate limiting active on all public endpoints

### 5.2 Authentication & Secrets

| Check                                        | Status |
| -------------------------------------------- | ------ |
| JWT secrets rotated from development         | [ ]    |
| Access token TTL configured (15 min default) | [ ]    |
| Refresh token mechanism working              | [ ]    |
| AWS Secrets Manager configured               | [ ]    |

### 5.3 TLS/HTTPS

| Check                                   | Status |
| --------------------------------------- | ------ |
| HTTPS enforced in production            | [ ]    |
| SSL certificate valid                   | [ ]    |
| Certificate not expiring within 30 days | [ ]    |

**Verification Commands:**

```bash
# Check certificate expiration
curl -vI https://ringrift.ai 2>&1 | grep "expire date"

# Verify HTTPS redirect
curl -I http://ringrift.ai
```

**PV Reference:** PV-03 established rate limit bypass for staging only

---

## 6. Infrastructure Validation

### 6.1 Docker Infrastructure

| Check                  | Command                | Status |
| ---------------------- | ---------------------- | ------ |
| All services build     | `docker compose build` | [ ]    |
| All containers start   | `docker compose up -d` | [ ]    |
| All containers healthy | `docker ps`            | [ ]    |
| No container restarts  | Uptime stable          | [ ]    |

### 6.2 Staging Deployment

| Check                                  | Status |
| -------------------------------------- | ------ |
| `./scripts/deploy-staging.sh` succeeds | [ ]    |
| All health endpoints return 200        | [ ]    |
| Database migrations applied            | [ ]    |
| Load test users can be seeded          | [ ]    |

### 6.3 Production Deployment Verification

| Check                          | Status |
| ------------------------------ | ------ |
| Deployment runbook reviewed    | [ ]    |
| Rollback procedure tested      | [ ]    |
| Health check endpoints working | [ ]    |
| Monitoring receiving metrics   | [ ]    |

**Verification Commands:**

```bash
# Check container status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check container resource usage
docker stats --no-stream

# Verify health endpoints
curl -s http://localhost:3000/health | jq
curl -s http://localhost:3000/ready | jq
curl -s http://localhost:8001/health | jq
```

---

## 7. Known Gaps (Non-Blocking for v1.0)

The following gaps were identified during validation but are **not blocking** for v1.0 release:

### 7.1 Python AI Service Not Load Tested

**Gap:** PV-09 revealed that the WebSocket gameplay test uses local TypeScript heuristic AI rather than the Python AI service.

**Impact:** AI-specific SLOs (response latency p95, fallback rate) could not be validated through load testing.

**Mitigation:**

- AI service health and degradation drill (PV-10) passed
- Local heuristic AI performs well under load
- Python AI service can be validated in future iterations

**Status:** Acceptable for v1.0

---

### 7.2 Web App Prometheus Metrics Not Instrumented

**Gap:** PV-11 found that the Node.js web app does not expose a `/metrics` endpoint for Prometheus scraping.

**Impact:**

- System Health dashboard panels show "No data" for HTTP/WebSocket metrics
- Game Performance dashboard missing web game metrics

**Mitigation:**

- k6 load test results provide all required SLO data
- AI cluster monitoring is fully functional
- Can be instrumented post-launch

**Status:** Acceptable for v1.0

---

### 7.3 Production Load Test Not Yet Executed

**Gap:** All load tests have been run against staging Docker environment, not production infrastructure.

**Impact:** Production-specific performance characteristics not validated.

**Mitigation:**

- Staging tests validate application behavior
- Production infrastructure is similar (same Docker images)
- Can run abbreviated smoke test after deployment

**Status:** Acceptable for v1.0 with post-deploy smoke test

---

## 8. Gate Decision Workflow

```
                     ┌──────────────────────┐
                     │  Start Validation    │
                     └──────────────────────┘
                               │
                               ▼
                     ┌──────────────────────┐
                     │ 1. Pre-Deploy Tests  │
                     │    All passing?      │
                     └──────────────────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                    Yes                  No → Fix tests
                     │
                     ▼
                     ┌──────────────────────┐
                     │ 2. Load Test Gates   │
                     │    PV-07, PV-08      │
                     └──────────────────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                   PASS              FAIL/CONDITIONAL
                     │                   │
                     ▼                   ▼
                     │              Investigate & Remediate
                     │
                     ▼
                     ┌──────────────────────┐
                     │ 3. AI Degradation    │
                     │    Drill (PV-10)     │
                     └──────────────────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                   PASS                FAIL
                     │                   │
                     ▼                   ▼
                     │              Review AI service config
                     │
                     ▼
                     ┌──────────────────────┐
                     │ 4. Security Checks   │
                     │    Rate limits, TLS  │
                     └──────────────────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                 Verified             Not Verified
                     │                   │
                     ▼                   ▼
                     │              Update configuration
                     │
                     ▼
              ┌────────────────┐
              │  GATE: PASS    │
              │  Ready for     │
              │  Production    │
              └────────────────┘
```

---

## 9. Validation Execution Summary

### Required Validations (Must Pass)

| Validation             | PV Task | Status  | Last Run   |
| ---------------------- | ------- | ------- | ---------- |
| Pre-deploy tests       | -       | ✅      | Continuous |
| Baseline load test     | PV-07   | ✅ PASS | 2025-12-20 |
| Target-scale load test | PV-08   | ✅ PASS | 2025-12-20 |
| AI degradation drill   | PV-10   | ✅ PASS | 2025-12-20 |
| Security configuration | PV-03   | ✅ PASS | 2025-12-20 |

### Optional/Informational Validations

| Validation         | PV Task | Status     | Notes                                      |
| ------------------ | ------- | ---------- | ------------------------------------------ |
| AI-heavy load test | PV-09   | ⚠️ PARTIAL | WebSocket passed, Python AI not tested     |
| Grafana dashboards | PV-11   | ⚠️ PARTIAL | AI cluster works, web app not instrumented |

---

## 10. Command Reference

### Quick Validation Commands

```bash
# Run all pre-deploy tests
npm test && npm run test:core && npm run test:orchestrator-parity

# Deploy staging and run preflight
./scripts/deploy-staging.sh
npm run load:preflight

# Run baseline load test
SEED_LOADTEST_USERS=true tests/load/scripts/run-baseline.sh staging

# Run target-scale load test
SEED_LOADTEST_USERS=true tests/load/scripts/run-target-scale.sh staging

# Run AI degradation drill
npx ts-node scripts/run-ai-degradation-drill.ts

# Verify SLOs
npm run slo:verify tests/load/results/<file>.json -- --env production

# Check Grafana
open http://localhost:3002
```

### Health Check Commands

```bash
# Application health
curl -s http://localhost:3000/health | jq
curl -s http://localhost:3000/ready | jq

# AI service health
curl -s http://localhost:8001/health | jq

# Container status
docker ps --format "table {{.Names}}\t{{.Status}}"

# Resource usage
docker stats --no-stream
```

---

## Revision History

| Version | Date       | Changes                                                    |
| ------- | ---------- | ---------------------------------------------------------- |
| 1.0     | 2025-12-20 | Initial creation consolidating PV-01 through PV-11 results |

---

_This checklist should be executed before each production deployment. All required validations must pass before release._
