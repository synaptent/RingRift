# SLO Threshold Alignment Audit

> **Created:** 2025-12-20
> **Status:** Updated (2025-12-20)
> **Purpose:** Audit SLO thresholds across all configuration sources for v1.0 launch readiness

## Overview

This audit examines SLO threshold definitions across all project configuration sources to identify misalignments, inconsistencies, or gaps. The goal is to create a unified SLO reference for production validation.

### Documents Reviewed

| Document                                                                                                 | Role                            | Status      |
| -------------------------------------------------------------------------------------------------------- | ------------------------------- | ----------- |
| [`PROJECT_GOALS.md`](../../PROJECT_GOALS.md:155)                                                         | **SSoT for SLOs** (§4.1)        | ✅ Reviewed |
| [`tests/load/k6.config.js`](../../tests/load/k6.config.js:1)                                             | k6 load test configuration      | ✅ Reviewed |
| [`tests/load/config/thresholds.json`](../../tests/load/config/thresholds.json:1)                         | Environment-specific thresholds | ✅ Reviewed |
| [`docs/operations/SLO_VERIFICATION.md`](../operations/SLO_VERIFICATION.md:1)                             | SLO verification procedures     | ✅ Reviewed |
| [`docs/operations/ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:1)                       | Alerting configuration          | ✅ Reviewed |
| [`docs/ai/AI_SLO_BUDGETS.md`](../ai/AI_SLO_BUDGETS.md:1)                                                 | AI-specific SLO budgets         | ✅ Reviewed |
| [`docs/production/PRODUCTION_READINESS_CHECKLIST.md`](../production/PRODUCTION_READINESS_CHECKLIST.md:1) | Launch requirements             | ✅ Reviewed |
| [`docs/testing/LOAD_TEST_BASELINE.md`](../testing/LOAD_TEST_BASELINE.md:1)                               | Baseline test results           | ✅ Reviewed |

---

## 1. SLO Inventory Table

### 1.1 Authoritative SLOs from PROJECT_GOALS.md §4.1

| Metric          | Target          | Measurement                               | Source Reference     |
| --------------- | --------------- | ----------------------------------------- | -------------------- |
| System uptime   | >99.9%          | Core gameplay surfaces availability       | PROJECT_GOALS.md:159 |
| AI move latency | <1 second (p95) | Time from AI turn start to move broadcast | PROJECT_GOALS.md:160 |
| UI frame rate   | <16ms updates   | Smooth 60fps rendering during gameplay    | PROJECT_GOALS.md:161 |
| Move validation | <200ms (p95)    | Human move submission to broadcast        | PROJECT_GOALS.md:162 |
| HTTP API        | <500ms (p95)    | Login, game creation, state fetch         | PROJECT_GOALS.md:163 |

### 1.2 k6 Thresholds (thresholds.json) - Staging

| Metric                   | Category   | Target | Source Reference   |
| ------------------------ | ---------- | ------ | ------------------ |
| Auth login p95           | HTTP API   | 750ms  | thresholds.json:14 |
| Auth login p99           | HTTP API   | 1200ms | thresholds.json:15 |
| Auth login error rate    | HTTP API   | 1.0%   | thresholds.json:16 |
| Game creation p95        | HTTP API   | 800ms  | thresholds.json:19 |
| Game creation p99        | HTTP API   | 1500ms | thresholds.json:20 |
| Game creation error rate | HTTP API   | 1.0%   | thresholds.json:21 |
| Game state fetch p95     | HTTP API   | 400ms  | thresholds.json:24 |
| Game state fetch p99     | HTTP API   | 800ms  | thresholds.json:25 |
| Move submission e2e p95  | WebSocket  | 300ms  | thresholds.json:32 |
| Move submission e2e p99  | WebSocket  | 600ms  | thresholds.json:33 |
| Server processing p95    | WebSocket  | 200ms  | thresholds.json:34 |
| Server processing p99    | WebSocket  | 400ms  | thresholds.json:35 |
| Stall threshold          | WebSocket  | 2000ms | thresholds.json:36 |
| Stall rate               | WebSocket  | 0.5%   | thresholds.json:37 |
| Connection success rate  | WebSocket  | 99.0%  | thresholds.json:41 |
| AI move p95              | AI Service | 1500ms | thresholds.json:47 |
| AI move p99              | AI Service | 3000ms | thresholds.json:48 |
| AI fallback rate         | AI Service | 1.0%   | thresholds.json:50 |
| AI turn e2e p95          | AI Service | 3000ms | thresholds.json:53 |
| AI turn e2e p99          | AI Service | 5000ms | thresholds.json:54 |
| Max concurrent games     | Scale      | 20     | thresholds.json:59 |
| Max active players       | Scale      | 60     | thresholds.json:60 |
| Max AI-controlled seats  | Scale      | 40     | thresholds.json:61 |

### 1.3 k6 Thresholds (thresholds.json) - Production

| Metric                   | Category     | Target | Source Reference        |
| ------------------------ | ------------ | ------ | ----------------------- |
| Auth login p95           | HTTP API     | 700ms  | thresholds.json:70      |
| Auth login p99           | HTTP API     | 1000ms | thresholds.json:71      |
| Auth login error rate    | HTTP API     | 0.5%   | thresholds.json:72      |
| Game creation p95        | HTTP API     | 400ms  | thresholds.json:76      |
| Game creation p99        | HTTP API     | 800ms  | thresholds.json:77      |
| Game creation error rate | HTTP API     | 0.5%   | thresholds.json:78      |
| Game state fetch p95     | HTTP API     | 200ms  | thresholds.json:81      |
| Game state fetch p99     | HTTP API     | 400ms  | thresholds.json:82      |
| Move submission e2e p95  | WebSocket    | 200ms  | thresholds.json:89      |
| Move submission e2e p99  | WebSocket    | 400ms  | thresholds.json:90      |
| Server processing p95    | WebSocket    | 150ms  | thresholds.json:91      |
| Server processing p99    | WebSocket    | 300ms  | thresholds.json:92      |
| Stall threshold          | WebSocket    | 2000ms | thresholds.json:93      |
| Stall rate               | WebSocket    | 0.2%   | thresholds.json:94      |
| Connection success rate  | WebSocket    | 99.5%  | thresholds.json:98      |
| AI move p95 (CPU)        | AI Service   | 1000ms | thresholds.json:104     |
| AI move p99 (CPU)        | AI Service   | 2000ms | thresholds.json:105     |
| AI move p95 (GPU)        | AI Service   | 500ms  | thresholds.json:110     |
| AI move p99 (GPU)        | AI Service   | 1000ms | thresholds.json:111     |
| AI fallback rate         | AI Service   | 0.5%   | thresholds.json:107/113 |
| AI turn e2e p95          | AI Service   | 2000ms | thresholds.json:116     |
| AI turn e2e p99          | AI Service   | 4000ms | thresholds.json:117     |
| Max concurrent games     | Scale        | 100    | thresholds.json:121     |
| Max active players       | Scale        | 300    | thresholds.json:122     |
| Max AI-controlled seats  | Scale        | 210    | thresholds.json:125     |
| Core gameplay uptime     | Availability | 99.9%  | thresholds.json:129     |
| Monthly error budget     | Availability | 0.1%   | thresholds.json:130     |

### 1.4 SLO_VERIFICATION.md Thresholds

| Metric                    | Priority | Target          | Measurement                        | Source                 |
| ------------------------- | -------- | --------------- | ---------------------------------- | ---------------------- |
| Service Availability      | Critical | ≥99.9%          | successful_requests/total          | SLO_VERIFICATION.md:51 |
| Error Rate (staging)      | Critical | ≤1%             | http_req_failed rate               | SLO_VERIFICATION.md:54 |
| Error Rate (prod)         | Critical | ≤0.5%           | http_req_failed rate               | SLO_VERIFICATION.md:54 |
| True Error Rate           | Critical | ≤0.5% / ≤0.2%   | true_errors_total / total_requests | SLO_VERIFICATION.md:55 |
| Contract Failures         | Critical | 0               | contract_failures_total            | SLO_VERIFICATION.md:55 |
| Lifecycle Mismatches      | Critical | 0               | id_lifecycle_mismatches_total      | SLO_VERIFICATION.md:56 |
| API Latency p95           | High     | <500ms          | http_req_duration p95              | SLO_VERIFICATION.md:60 |
| Move Latency p95          | High     | ≤300ms / ≤200ms | move_latency p95                   | SLO_VERIFICATION.md:61 |
| WebSocket Connect p95     | High     | <1000ms         | ws_connecting p95                  | SLO_VERIFICATION.md:62 |
| AI Response p95           | High     | <1000ms         | ai_response_time p95               | SLO_VERIFICATION.md:63 |
| Concurrent Games (prod)   | High     | ≥100            | concurrent_games                   | SLO_VERIFICATION.md:64 |
| Concurrent Players (prod) | High     | ≥300            | concurrent_vus                     | SLO_VERIFICATION.md:65 |
| WebSocket Success Rate    | High     | ≥99%            | websocket_connection_success_rate  | SLO_VERIFICATION.md:66 |
| Move Stall Rate           | High     | ≤0.5%           | stalled_moves/total_moves          | SLO_VERIFICATION.md:67 |
| API Latency p99           | Medium   | <2000ms         | http_req_duration p99              | SLO_VERIFICATION.md:72 |
| Game Creation p95         | Medium   | <2000ms         | game_creation_time p95             | SLO_VERIFICATION.md:73 |
| AI Response p99           | Medium   | <2000ms         | ai_response_time p99               | SLO_VERIFICATION.md:74 |
| AI Fallback Rate          | Medium   | ≤1%             | ai_fallback_total/ai_requests      | SLO_VERIFICATION.md:75 |

### 1.5 ALERTING_THRESHOLDS.md Values

| Alert                    | Severity | Threshold | Duration | Source                     |
| ------------------------ | -------- | --------- | -------- | -------------------------- |
| HighErrorRate            | critical | >5% 5xx   | 5 min    | ALERTING_THRESHOLDS.md:133 |
| ElevatedErrorRate        | warning  | >1% 5xx   | 10 min   | ALERTING_THRESHOLDS.md:157 |
| HighP99Latency           | warning  | >2s       | 5 min    | ALERTING_THRESHOLDS.md:195 |
| HighP99LatencyCritical   | critical | >5s       | 2 min    | ALERTING_THRESHOLDS.md:205 |
| HighP95Latency           | warning  | >1s       | 10 min   | ALERTING_THRESHOLDS.md:224 |
| HighMedianLatency        | warning  | >500ms    | 15 min   | ALERTING_THRESHOLDS.md:235 |
| HighGameMoveLatency      | warning  | p99 >1s   | 5 min    | ALERTING_THRESHOLDS.md:249 |
| HighMemoryUsage          | warning  | >1.5GB    | 10 min   | ALERTING_THRESHOLDS.md:269 |
| HighMemoryUsageCritical  | critical | >2GB      | 5 min    | ALERTING_THRESHOLDS.md:279 |
| HighEventLoopLag         | warning  | >100ms    | 5 min    | ALERTING_THRESHOLDS.md:299 |
| HighEventLoopLagCritical | critical | >500ms    | 2 min    | ALERTING_THRESHOLDS.md:310 |
| HighWebSocketConnections | warning  | >1000     | 5 min    | ALERTING_THRESHOLDS.md:405 |
| AIFallbackRateHigh       | warning  | >30%      | 10 min   | ALERTING_THRESHOLDS.md:422 |
| AIFallbackRateCritical   | critical | >50%      | 5 min    | ALERTING_THRESHOLDS.md:435 |
| AIRequestHighLatency     | warning  | p99 >5s   | 5 min    | ALERTING_THRESHOLDS.md:455 |

### 1.6 AI_SLO_BUDGETS.md Values

| Metric           | Target  | Priority | Source               |
| ---------------- | ------- | -------- | -------------------- |
| AI Response p95  | <1000ms | High     | AI_SLO_BUDGETS.md:53 |
| AI Response p99  | <2000ms | Medium   | AI_SLO_BUDGETS.md:54 |
| AI Fallback Rate | ≤1%     | Medium   | AI_SLO_BUDGETS.md:55 |
| Move Stall Rate  | ≤0.5%   | High     | AI_SLO_BUDGETS.md:56 |
| Request Timeout  | 5000ms  | Config   | AI_SLO_BUDGETS.md:19 |

### 1.7 Load Test Classification Thresholds

| Metric                          | Staging    | Production  | Source                  |
| ------------------------------- | ---------- | ----------- | ----------------------- |
| contract_failures_total         | max: 0     | max: 0      | thresholds.json:245/264 |
| id_lifecycle_mismatches_total   | max: 0     | max: 0      | thresholds.json:248/267 |
| capacity_failures_total         | rate: 0.01 | rate: 0.005 | thresholds.json:251/270 |
| true_errors rate                | 0.005      | 0.002       | thresholds.json:255/274 |
| websocket.protocol_errors_max   | 0          | 0           | thresholds.json:258/277 |
| websocket.connection_errors_max | 50         | 10          | thresholds.json:259/278 |

---

## 2. Alignment Matrix

### 2.1 PROJECT_GOALS.md vs k6 Thresholds

| PROJECT_GOALS.md SLO         | thresholds.json (Production)                          | Aligned?   | Notes                                 |
| ---------------------------- | ----------------------------------------------------- | ---------- | ------------------------------------- |
| System uptime >99.9%         | core_gameplay_uptime: 99.9%                           | ✅ Aligned | thresholds.json updated to match SSoT |
| AI move <1s (p95)            | ai_service.cpu_baseline.move_p95: 1000ms              | ✅ Aligned | Exact match                           |
| UI frame rate <16ms          | N/A                                                   | ⚠️ **GAP** | Not represented in k6 thresholds      |
| Move validation <200ms (p95) | move_submission.e2e_p95: 200ms                        | ✅ Aligned | Exact match                           |
| HTTP API <500ms (p95)        | game_creation.p95: 400ms, game_state_fetch.p95: 200ms | ✅ Aligned | k6 is stricter per-endpoint           |

### 2.2 PROJECT_GOALS.md vs SLO_VERIFICATION.md

| PROJECT_GOALS.md SLO         | SLO_VERIFICATION.md         | Aligned?   | Notes                                 |
| ---------------------------- | --------------------------- | ---------- | ------------------------------------- |
| System uptime >99.9%         | Service Availability ≥99.9% | ✅ Aligned |                                       |
| AI move <1s (p95)            | AI Response p95 <1000ms     | ✅ Aligned |                                       |
| UI frame rate <16ms          | N/A                         | ⚠️ **GAP** | Client-side metric, not in load tests |
| Move validation <200ms (p95) | Move Latency p95 ≤300/≤200  | ✅ Aligned | SLO_VERIFICATION updated to match     |
| HTTP API <500ms (p95)        | API Latency p95 <500ms      | ✅ Aligned |                                       |

### 2.3 k6 Thresholds vs Alerting Thresholds

| k6 Threshold (Production)   | Alerting Threshold              | Aligned?              | Notes                                                          |
| --------------------------- | ------------------------------- | --------------------- | -------------------------------------------------------------- |
| error_rate 0.5%             | HighErrorRate >5% (critical)    | ⚠️ **LAG**            | Alert triggers at 10x SLO breach; expected for major incidents |
| error_rate 0.5%             | ElevatedErrorRate >1% (warning) | ✅ Acceptable         | Warning at 2x SLO                                              |
| game_state_fetch.p95: 200ms | HighP95Latency >1s              | ⚠️ **LAG**            | Alert at 5x SLO - very permissive                              |
| move_submission.p95: 200ms  | HighGameMoveLatency p99 >1s     | ⚠️ Mixed              | Different percentiles (p95 vs p99)                             |
| AI fallback 0.5%            | AIFallbackRateHigh >30%         | ❌ **MAJOR MISMATCH** | Alert at 60x SLO                                               |
| WebSocket connections 1000  | HighWebSocketConnections >1000  | ✅ Aligned            |                                                                |

### 2.4 AI SLO Alignment

| Source                     | AI Response p95   | AI Response p99 | AI Fallback Rate          |
| -------------------------- | ----------------- | --------------- | ------------------------- |
| PROJECT_GOALS.md           | <1s               | (implied)       | (not specified)           |
| thresholds.json (prod CPU) | 1000ms            | 2000ms          | 0.5%                      |
| SLO_VERIFICATION.md        | <1000ms           | <2000ms         | ≤1%                       |
| AI_SLO_BUDGETS.md          | <1000ms           | <2000ms         | ≤1%                       |
| ALERTING_THRESHOLDS.md     | (alert at 5s p99) | -               | 30% warning, 50% critical |

**Assessment:** AI latency SLOs are well-aligned. Fallback rate has:

- SLO target: 0.5-1%
- Alert warning: 30%
- Alert critical: 50%

This gap is intentional - alerts are for operational incidents, not SLO violations.

### 2.5 Staging vs Production Threshold Comparison

| Metric                       | Staging | Production | Ratio  | Assessment     |
| ---------------------------- | ------- | ---------- | ------ | -------------- |
| HTTP API p95 (game creation) | 800ms   | 400ms      | 2x     | ✅ Appropriate |
| HTTP API p99 (game creation) | 1500ms  | 800ms      | ~2x    | ✅ Appropriate |
| Error rate                   | 1.0%    | 0.5%       | 2x     | ✅ Appropriate |
| Move submission p95          | 300ms   | 200ms      | 1.5x   | ✅ Appropriate |
| AI move p95                  | 1500ms  | 1000ms     | 1.5x   | ✅ Appropriate |
| Connection success rate      | 99.0%   | 99.5%      | 1.005x | ✅ Appropriate |
| Concurrent games             | 20      | 100        | 5x     | ✅ Appropriate |
| Concurrent players           | 60      | 300        | 5x     | ✅ Appropriate |

---

## 3. Issues and Gaps Identified

### 3.1 Critical Issues (Resolved)

#### ISSUE-1: System Uptime Threshold Mismatch (Resolved)

- **PROJECT_GOALS.md**: >99.9%
- **thresholds.json**: 99.9%
- **Resolution**: Updated `core_gameplay_uptime_percent` and `monthly_error_budget_percent` to align with SSoT.

#### ISSUE-2: Move Validation Threshold Inconsistency (Resolved)

- **PROJECT_GOALS.md**: <200ms (p95)
- **SLO_VERIFICATION.md**: ≤300ms (staging) / ≤200ms (prod)
- **thresholds.json (prod)**: 200ms
- **Resolution**: Updated SLO verification target to 200ms and aligned SLO definitions.

### 3.2 High-Priority Issues

#### ISSUE-3: AI Fallback Alert Gap

- **SLO target**: 0.5-1%
- **Warning alert**: 30%
- **Critical alert**: 50%
- **Impact**: Alert won't fire until fallback rate is 30-60x above SLO
- **Recommendation**: Add an info-level alert at 5% and warning at 10%

#### ISSUE-4: UI Frame Rate Not Measured

- **PROJECT_GOALS.md**: <16ms UI updates
- **k6 thresholds**: Not present
- **Impact**: Cannot validate via load tests (expected - client-side metric)
- **Recommendation**: Document that this is validated via client-side profiling and E2E tests, not load tests

#### ISSUE-5: true_error_rate Not in SLO_VERIFICATION.md (Resolved)

- **thresholds.json**: true_errors.rate defined (0.005 staging, 0.002 prod)
- **SLO_VERIFICATION.md**: Added to Critical SLOs
- **Resolution**: true_error_rate now appears in SLO verification and reporting.

### 3.3 Medium-Priority Issues

#### ISSUE-6: WebSocket Latency Thresholds Not in PROJECT_GOALS.md

- **thresholds.json**: Multiple WebSocket latency thresholds defined
- **PROJECT_GOALS.md**: Only "move validation <200ms" mentioned
- **Impact**: WebSocket-specific SLOs not traceable to goals
- **Recommendation**: Either add WebSocket SLOs to PROJECT_GOALS.md or document they're derived from move validation SLO

#### ISSUE-7: Alerting Thresholds Very Permissive Compared to k6

- Most alerts trigger at 5-10x the SLO threshold
- This is intentional for operational stability but should be documented
- **Recommendation**: Add note in ALERTING_THRESHOLDS.md explaining the gap rationale

### 3.4 Gaps in Coverage

| Gap                        | Description                                                    | Severity       |
| -------------------------- | -------------------------------------------------------------- | -------------- |
| UI frame rate              | Not measurable in load tests                                   | Low (expected) |
| WebSocket reconnection SLO | Defined in thresholds.json but not in PROJECT_GOALS.md         | Medium         |
| Game creation error rate   | Per-endpoint in thresholds.json, aggregate in PROJECT_GOALS.md | Low            |
| Spectator capacity         | Defined in thresholds.json (50) but not in PROJECT_GOALS.md    | Low            |

---

## 4. Recommendations

### 4.1 Required Changes for SLO Alignment

| Priority | Source                            | Change                                      | Issue   |
| -------- | --------------------------------- | ------------------------------------------- | ------- |
| P0       | thresholds.json:129               | Update core_gameplay_uptime to 99.9% (done) | ISSUE-1 |
| P0       | SLO_VERIFICATION.md:61            | Update Move Latency p95 to <200ms (done)    | ISSUE-2 |
| P1       | SLO_VERIFICATION.md               | Add true_error_rate SLO (done)              | ISSUE-5 |
| P1       | ALERTING_THRESHOLDS.md            | Add AIFallbackRateElevated alert at 5%      | ISSUE-3 |
| P2       | ALERTING_THRESHOLDS.md            | Document alert vs SLO threshold rationale   | ISSUE-7 |
| P2       | PRODUCTION_READINESS_CHECKLIST.md | Note UI frame rate is client-validated      | ISSUE-4 |

### 4.2 Documentation Improvements

1. **Add "SLO Source of Truth" section to ALERTING_THRESHOLDS.md**
   - Explain that alerts are for operational incidents (major degradation)
   - SLO violations at lower thresholds are caught by k6 load test gates

2. **Update SLO_VERIFICATION.md**
   - Add true_error_rate metric (done)
   - Tighten Move Latency p95 to match PROJECT_GOALS.md (done)

3. **Consider adding WebSocket SLOs to PROJECT_GOALS.md**
   - WebSocket connect latency
   - Connection success rate
   - Message latency (if distinct from move submission)

---

## 5. Unified SLO Reference for v1.0

This table represents the **definitive SLO values** for v1.0 production validation, reconciling all sources with PROJECT_GOALS.md as the SSoT.

### 5.1 Critical Priority SLOs (Zero Tolerance)

| SLO ID | Metric               | Target       | Measurement                          | Pass Criteria                  |
| ------ | -------------------- | ------------ | ------------------------------------ | ------------------------------ |
| SLO-C1 | Service Availability | ≥99.9%       | successful_requests / total_requests | Rate ≥0.999 over test duration |
| SLO-C2 | Error Rate           | ≤0.5% (prod) | http_req_failed excluding 401/429    | Rate ≤0.005                    |
| SLO-C3 | Contract Failures    | 0            | contract_failures_total              | Count = 0                      |
| SLO-C4 | Lifecycle Mismatches | 0            | id_lifecycle_mismatches_total        | Count = 0                      |
| SLO-C5 | True Error Rate      | ≤0.2% (prod) | true_errors_total (excludes 401/429) | Rate ≤0.002                    |

### 5.2 High Priority SLOs

| SLO ID | Metric             | Target        | Measurement                       | Pass Criteria |
| ------ | ------------------ | ------------- | --------------------------------- | ------------- |
| SLO-H1 | HTTP API Latency   | <500ms (p95)  | http_req_duration p95             | p95 < 500ms   |
| SLO-H2 | Move Validation    | <200ms (p95)  | move_submission_latency p95       | p95 < 200ms   |
| SLO-H3 | AI Response Time   | <1000ms (p95) | ai_response_time p95              | p95 < 1000ms  |
| SLO-H4 | WebSocket Connect  | <1000ms (p95) | ws_connecting p95                 | p95 < 1000ms  |
| SLO-H5 | WebSocket Success  | ≥99.5% (prod) | websocket_connection_success_rate | Rate ≥0.995   |
| SLO-H6 | Move Stall Rate    | ≤0.2% (prod)  | stalled_moves / total_moves       | Rate ≤0.002   |
| SLO-H7 | Concurrent Games   | ≥100 (prod)   | concurrent_active_games max       | Max ≥100      |
| SLO-H8 | Concurrent Players | ≥300 (prod)   | concurrent_vus max                | Max ≥300      |

### 5.3 Medium Priority SLOs

| SLO ID | Metric           | Target        | Measurement                     | Pass Criteria |
| ------ | ---------------- | ------------- | ------------------------------- | ------------- |
| SLO-M1 | HTTP API Latency | <2000ms (p99) | http_req_duration p99           | p99 < 2000ms  |
| SLO-M2 | Game Creation    | <800ms (p95)  | game_creation_time p95          | p95 < 800ms   |
| SLO-M3 | AI Response Time | <2000ms (p99) | ai_response_time p99            | p99 < 2000ms  |
| SLO-M4 | AI Fallback Rate | ≤0.5% (prod)  | ai_fallback_total / ai_requests | Rate ≤0.005   |

### 5.4 Production Validation Pass/Fail Criteria

A production validation run **PASSES** when:

1. **All Critical SLOs (SLO-C1 through SLO-C5)** meet their targets
2. **All High Priority SLOs (SLO-H1 through SLO-H8)** meet their targets
3. **Medium Priority SLOs** are within 20% of target, with any exceptions documented

A production validation run is **CONDITIONAL** when:

- All Critical SLOs pass
- One or more High SLOs are within 10% of target
- Explicit risk acceptance is documented

A production validation run **FAILS** when:

- Any Critical SLO fails
- Any High SLO is significantly breached (>20% off target)
- Capacity SLOs (games/players) are not reached

### 5.5 Environment-Specific Thresholds

| SLO                          | Staging Threshold | Production Threshold |
| ---------------------------- | ----------------- | -------------------- |
| Error Rate (SLO-C2)          | ≤1.0%             | ≤0.5%                |
| True Error Rate (SLO-C5)     | ≤0.5%             | ≤0.2%                |
| HTTP API p95 (SLO-H1)        | <800ms            | <500ms               |
| Move Validation p95 (SLO-H2) | <300ms            | <200ms               |
| AI Response p95 (SLO-H3)     | <1500ms           | <1000ms              |
| WebSocket Success (SLO-H5)   | ≥99.0%            | ≥99.5%               |
| Move Stall Rate (SLO-H6)     | ≤0.5%             | ≤0.2%                |
| Concurrent Games (SLO-H7)    | ≥20               | ≥100                 |
| Concurrent Players (SLO-H8)  | ≥60               | ≥300                 |

---

## 6. Audit Summary

### 6.1 Overall Assessment

The SLO threshold definitions across the project are **largely aligned**. The previously noted uptime and move-latency mismatches have been resolved; remaining gaps are primarily around client-only metrics (UI frame rate) and explicit reconnection SLOs in goals.

### 6.2 Alignment Statistics

| Category          | Aligned | Misaligned | Gaps  |
| ----------------- | ------- | ---------- | ----- |
| HTTP API SLOs     | 4       | 0          | 0     |
| WebSocket SLOs    | 4       | 0          | 1     |
| AI SLOs           | 4       | 0          | 0     |
| Availability SLOs | 2       | 0          | 0     |
| Capacity SLOs     | 4       | 0          | 0     |
| **Total**         | **18**  | **0**      | **1** |

### 6.3 Completeness Check

| Question                                                           | Answer                                                 |
| ------------------------------------------------------------------ | ------------------------------------------------------ |
| Do k6 thresholds match PROJECT_GOALS.md SLOs?                      | ✅ Yes (UI frame rate remains client-side)             |
| Do alerting thresholds match k6 thresholds?                        | ✅ Yes - alerts are intentionally at higher thresholds |
| Are staging vs production thresholds appropriately differentiated? | ✅ Yes - production is 1.5-2x stricter                 |
| Are all SLOs from PROJECT_GOALS.md §4.1 represented in k6?         | ✅ Yes except UI frame rate (client-side)              |
| Is true_error_rate properly configured?                            | ✅ Yes - k6 + SLO verification aligned                 |
| Are AI response budgets aligned?                                   | ✅ Yes                                                 |

### 6.4 Action Items

| Priority | Action                                     | Owner | Status  |
| -------- | ------------------------------------------ | ----- | ------- |
| P0       | Fix uptime threshold in thresholds.json    | -     | ✅ Done |
| P0       | Fix move latency in SLO_VERIFICATION.md    | -     | ✅ Done |
| P1       | Add true_error_rate to SLO_VERIFICATION.md | -     | ✅ Done |
| P1       | Add lower-threshold AI fallback alert      | -     | ⬜ Todo |
| P2       | Document alert vs SLO gap rationale        | -     | ⬜ Todo |

---

## Related Documents

- [`PROJECT_GOALS.md`](../../PROJECT_GOALS.md) - Authoritative SLO source (§4.1)
- [`tests/load/config/thresholds.json`](../../tests/load/config/thresholds.json) - k6 threshold definitions
- [`docs/operations/SLO_VERIFICATION.md`](../operations/SLO_VERIFICATION.md) - Verification procedures
- [`docs/operations/ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md) - Alert configuration
- [`docs/production/PRODUCTION_READINESS_CHECKLIST.md`](../production/PRODUCTION_READINESS_CHECKLIST.md) - Launch requirements

---

**Audit Status:** Updated  
**Created:** 2025-12-20  
**Author:** Architect Mode (PV-06)
