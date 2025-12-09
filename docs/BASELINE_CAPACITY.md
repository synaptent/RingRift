# RingRift Baseline Capacity

## Overview

This document tracks the baseline capacity measurements for the RingRift system.
Baselines are established through load testing and used to:

- Validate system performance against SLO targets
- Track capacity improvements over time
- Identify regressions before they reach production
- Plan capacity for scaling decisions

## Current Baseline

> ⚠️ **No baseline established yet**
>
> Run baseline tests with:
>
> ```bash
> npm run load:baseline:local   # For local testing
> npm run load:baseline:staging # For staging environment
> ```

Baseline runner notes (Dec 2025):

- `SCENARIO_ID` defaults to `BCAP_STAGING_BASELINE_20G_60P` and is threaded into filenames/tags.
- Optional WebSocket companion run (`websocket-stress.js`, preset=baseline ~60 conns) runs automatically unless `SKIP_WS_COMPANION=true` is set; companion result files are named `websocket_<SCENARIO_ID>_...json`.
- Optional user seeding via `SEED_LOADTEST_USERS=true` (with `LOADTEST_USER_*` overrides) seeds a load-test account pool before k6 starts.
- Smoke profile: set `SMOKE=1` for a short/local wiring check; otherwise the `load` profile is used.

Common baseline commands:

```bash
# Staging baseline with seeding (default SCENARIO_ID, WS companion on)
SEED_LOADTEST_USERS=true tests/load/scripts/run-baseline.sh --staging

# Local smoke baseline without WS companion
SMOKE=1 SKIP_WS_COMPANION=1 tests/load/scripts/run-baseline.sh --local
```

Recent baseline run artifacts (Dec 2025):

| Date (UTC) | Scenario ID                   | Notes                                    | Paths                                                                                                                                                                                                                     |
| ---------- | ----------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2025-12-08 | BCAP_STAGING_BASELINE_20G_60P | Smoke-level run; baseline + WS companion | `tests/load/results/baseline_staging_20251208_144949.json`, `tests/load/results/baseline_staging_20251208_144949_summary.json`, `tests/load/results/websocket_BCAP_STAGING_BASELINE_20G_60P_staging_20251208_144949.json` |

Post-run validation checklist:

- Run SLO verifier against the main and WS outputs: `node tests/load/scripts/verify-slos.js <results.json> console --env staging`.
- If passes, append a row to the table above with scenario ID, date, notes, and result paths.
- If fails, attach SLO summary JSON and open a tracking issue with failing metrics.

Target-scale run tracking:

- **Not yet executed** (as of 2025-12-08). First target-scale run should use `SCENARIO_ID=BCAP_SQ8_3P_TARGET_100G_300P`, include the WebSocket companion (unless skipped), and record both main and WS result paths here after completion.

Latest staging baseline run for pipeline validation (smoke-level only, not target scale): see
`tests/load/results/baseline_staging_20251208_144949.json` and the corresponding
`baseline_staging_20251208_144949_summary.json` and
`baseline_staging_20251208_144949_summary_slo_summary.json`.

### Production Targets (from PROJECT_GOALS.md)

| Metric             | Target  | Current Baseline | Status | Notes                                  |
| ------------------ | ------- | ---------------- | ------ | -------------------------------------- |
| Concurrent Games   | 100     | TBD              | ⏳     | Target for production-scale deployment |
| Concurrent Players | 300     | TBD              | ⏳     | 3 players per game average             |
| p95 Latency        | <500ms  | TBD              | ⏳     | HTTP request response time             |
| p99 Latency        | <2000ms | TBD              | ⏳     | Tail latency target                    |
| Error Rate         | <1%     | TBD              | ⏳     | HTTP 5xx error rate                    |
| WebSocket Success  | >99%    | TBD              | ⏳     | Connection establishment rate          |
| Contract Failures  | 0       | TBD              | ⏳     | API contract violations                |

### Staging Targets (from thresholds.json)

| Metric             | Target | Current Baseline | Status |
| ------------------ | ------ | ---------------- | ------ |
| Concurrent Games   | 20     | TBD              | ⏳     |
| Concurrent Players | 60     | TBD              | ⏳     |
| p95 Latency        | <800ms | TBD              | ⏳     |
| Error Rate         | <1%    | TBD              | ⏳     |

---

## Immediate Actions to Reconcile Baselines with Wave 7 Status

The improvement plan marks production validation as complete, but this doc still lacks current baselines and target/AI-heavy entries. Execute and record the following runs (staging preferred) to bring this file in sync:

1. **Baseline (20G/60P + WS companion)**
   - Run: `SEED_LOADTEST_USERS=true tests/load/scripts/run-baseline.sh --staging`
   - Verify: `node tests/load/scripts/verify-slos.js <baseline_results.json> console --env staging`
   - WS verify: `node tests/load/scripts/verify-slos.js <websocket_results.json> console --env staging`
   - Record main + summary + WS paths in the Baseline table above.

2. **Target Scale (100G/300P + WS companion)**
   - Run: `SEED_LOADTEST_USERS=true tests/load/scripts/run-target-scale.sh --staging`
   - Skip prompt for CI/non-interactive: add `SKIP_CONFIRM=true`
   - Verify: `node tests/load/scripts/verify-slos.js <target_results.json> console --env production`
   - WS verify: `node tests/load/scripts/verify-slos.js <target_ws_results.json> console --env production`
   - Add a row to the Target-scale table with date, scenario ID, notes, and paths.

3. **AI-Heavy Probe (75G/300P, 3 AI seats per game)**
   - Run: `SEED_LOADTEST_USERS=true tests/load/scripts/run-ai-heavy.sh --staging`
   - Optional smoke/local: `SMOKE=1 SKIP_WS_COMPANION=1 SKIP_CONFIRM=true --local`
   - Verify: `node tests/load/scripts/verify-slos.js <ai_heavy_results.json> console --env staging`
   - WS verify if run: `node tests/load/scripts/verify-slos.js <ai_heavy_ws_results.json> console --env staging`
   - Add a row to the AI-heavy table with date, scenario ID, notes, and paths.

Recording template (replace placeholders):

| Date (UTC) | Scenario ID   | Notes                                       | Paths                                                               |
| ---------- | ------------- | ------------------------------------------- | ------------------------------------------------------------------- |
| 2026-01-XX | <SCENARIO_ID> | <baseline/target/ai-heavy, env, smoke/full> | `<main_results.json>`, `<summary.json>`, `<websocket_results.json>` |

After each run, if any SLO verifier fails, attach the SLO summary JSON and open a tracking issue with failing metrics and remediation plan before marking the table row as canonical.

## Target Scale Validation

Target scale testing validates the system at full production capacity:
**100 concurrent games with 300 concurrent players**.

Target-scale runner notes (Dec 2025):

- `SCENARIO_ID` defaults to `BCAP_SQ8_3P_TARGET_100G_300P` and is threaded into filenames/tags.
- Optional WebSocket companion run (`websocket-stress.js`, preset=target ~300 conns) runs automatically unless `SKIP_WS_COMPANION=true` is set; companion result files are named `websocket_<SCENARIO_ID>_...json`.
- Optional user seeding via `SEED_LOADTEST_USERS=true` (with `LOADTEST_USER_*` overrides) seeds a load-test account pool before k6 starts.
- Confirmation prompt can be skipped with `SKIP_CONFIRM=true` for CI/non-interactive runs.

Common target-scale commands:

```bash
# Staging target-scale with seeding and WS companion
SEED_LOADTEST_USERS=true tests/load/scripts/run-target-scale.sh --staging

# Local dry-run without WS companion (still uses target-scale stages)
SKIP_WS_COMPANION=1 SKIP_CONFIRM=true tests/load/scripts/run-target-scale.sh --local
```

Target-scale run tracking:

| Date (UTC) | Scenario ID                  | Notes                            | Paths                                          |
| ---------- | ---------------------------- | -------------------------------- | ---------------------------------------------- |
| _TBD_      | BCAP_SQ8_3P_TARGET_100G_300P | _Pending first target-scale run_ | _Add main + summary + WS companion paths here_ |

Post-run validation checklist (target-scale):

- Run SLO verifier on main and WS outputs: `node tests/load/scripts/verify-slos.js <results.json> console --env production`.
- If passes, record paths (main, summary, WS companion) in the table above with date/notes.
- If fails, attach SLO summary JSON and open a tracking issue with failing metrics and remediation plan.

## AI-Heavy Capacity Probe (75 games / ~300 players, 3 AI seats per game)

AI-heavy runner notes (Dec 2025):

- Scenario ID defaults to `BCAP_SQ8_4P_AI_HEAVY_75G_300P` and is threaded into filenames/tags.
- Uses target-scale-like stages but at 75 VUs (steady) with 4p/3 AI seats; optional WebSocket companion (`WS_SCENARIO_PRESET=target`, ~300 conns) runs automatically unless `SKIP_WS_COMPANION=true`.
- Optional user seeding via `SEED_LOADTEST_USERS=true` (with `LOADTEST_USER_*` overrides).
- Confirmation prompt can be skipped with `SKIP_CONFIRM=true` for CI/non-interactive runs; smoke mode available via `SMOKE=1`.

Common AI-heavy commands:

```bash
# Staging AI-heavy probe with seeding and WS companion
SEED_LOADTEST_USERS=true tests/load/scripts/run-ai-heavy.sh --staging

# Local smoke AI-heavy probe without WS companion
SMOKE=1 SKIP_WS_COMPANION=1 SKIP_CONFIRM=true tests/load/scripts/run-ai-heavy.sh --local
```

AI-heavy run tracking:

| Date (UTC) | Scenario ID                   | Notes                          | Paths                                          |
| ---------- | ----------------------------- | ------------------------------ | ---------------------------------------------- |
| _TBD_      | BCAP_SQ8_4P_AI_HEAVY_75G_300P | _Pending first AI-heavy probe_ | _Add main + summary + WS companion paths here_ |

Post-run validation checklist (AI-heavy):

- Run SLO verifier on main and WS outputs: `node tests/load/scripts/verify-slos.js <results.json> console --env staging`.
- If passes, record paths (main, summary, WS companion) in the table above with date/notes.
- If fails, attach SLO summary JSON and open a tracking issue with failing metrics and remediation plan.

---

## Stress Testing (Beyond Target)

Stress testing goes beyond the target scale to find the system's breaking point.
This helps identify capacity ceilings and plan for scaling.

### How to Run Stress Test

```bash
# Run stress test (~30 minutes)
npm run load:stress:breaking

# With custom max VUs
MAX_VUS=600 npm run load:stress:breaking
```

### Stress Test Phases

| Phase       | Duration | VUs     | Description          |
| ----------- | -------- | ------- | -------------------- |
| Baseline    | 5 min    | 0→100   | Establish baseline   |
| Near Target | 5 min    | 100→200 | Approach target      |
| AT TARGET   | 5 min    | 200→300 | At production target |
| Beyond      | 5 min    | 300→400 | Push beyond target   |
| Stress      | 5 min    | 400→500 | Find limits          |
| Recovery    | 5 min    | 500→0   | Graceful ramp down   |

**Total Duration:** ~30 minutes

### Interpreting Stress Results

Look for these indicators of the breaking point:

| Indicator           | Meaning                          |
| ------------------- | -------------------------------- |
| Error rate spike    | System capacity exceeded         |
| p95 latency >2s     | Performance degradation          |
| Connection failures | Server connection pool exhausted |
| 503 responses       | Backend overloaded               |

The breaking point is the VU count just before significant degradation.
Use this information to:

- Set appropriate capacity limits
- Configure autoscaling thresholds
- Plan infrastructure upgrades

---

## How to Establish Baseline

### Prerequisites

1. **Local Development**:

   ```bash
   # Start the development server
   npm run dev

   # Optionally start AI service
   cd ai-service && python -m uvicorn app.main:app --port 8000
   ```

2. **Staging Environment**:
   ```bash
   # Deploy staging with Docker Compose
   docker-compose -f docker-compose.staging.yml up -d
   ```

### Running Baseline Tests

#### Option 1: Using npm scripts

```bash
# Local baseline (against localhost:3001)
npm run load:baseline:local

# Staging baseline
npm run load:baseline:staging
```

#### Option 2: Using the runner script directly

```bash
cd tests/load
./scripts/run-baseline.sh --local
./scripts/run-baseline.sh --staging
```

#### Option 3: Running k6 manually

```bash
# Basic run
k6 run --env BASE_URL=http://localhost:3001 tests/load/scenarios/concurrent-games.js

# With JSON output
k6 run \
  --env BASE_URL=http://localhost:3001 \
  --env LOAD_PROFILE=load \
  --out json=results/baseline.json \
  tests/load/scenarios/concurrent-games.js
```

### Analyzing Results

After a baseline run completes:

```bash
# Analyze the results
node tests/load/scripts/analyze-results.js tests/load/results/baseline_local_*.json

# View summary
cat tests/load/results/baseline_local_*_summary.json | jq .
```

## Recording Baseline Measurements

When you establish a new baseline, update the tables above with your measurements.
Use the following template:

```markdown
| Concurrent Games | 100 | **85** | ⚠️ | 85% of target, needs optimization |
```

**Status Icons:**

- ✅ Meeting or exceeding target
- ⚠️ Within 80% of target
- ❌ Below 80% of target
- ⏳ Not yet measured

## Historical Baselines

Track baseline measurements over time to monitor progress and detect regressions.

| Date | Version | Test Type | VUs | p95 (ms) | Error Rate | Concurrent Games | Notes             |
| ---- | ------- | --------- | --- | -------- | ---------- | ---------------- | ----------------- |
| -    | -       | Initial   | -   | -        | -          | -                | Pending first run |

### Recording Historical Data

After each significant baseline test:

1. Add a row to the table above
2. Include the git commit hash or version
3. Note any significant changes (infrastructure, code, configuration)
4. Archive the full results JSON for detailed analysis

## Test Configuration

The baseline test uses the following configuration (from `tests/load/configs/baseline.json`):

### Test Phases

| Phase        | Duration  | Target VUs | Description                    |
| ------------ | --------- | ---------- | ------------------------------ |
| Warmup       | 1 minute  | 10         | Warm up caches and connections |
| Ramp Up      | 3 minutes | 50         | Gradually increase load        |
| Steady State | 5 minutes | 50         | Hold for accurate measurement  |
| Ramp Down    | 1 minute  | 0          | Graceful shutdown              |

**Total Duration:** ~10 minutes

### Scenarios

The baseline test runs the `concurrent-games` scenario which:

1. Creates games with varying configurations (board types, player counts)
2. Monitors game state via polling
3. Tracks concurrent active games
4. Measures API response times

## Interpreting Results

### Latency Categories

| p95 Range  | Status        | Action                           |
| ---------- | ------------- | -------------------------------- |
| <200ms     | 🟢 Excellent  | No action needed                 |
| 200-500ms  | 🟡 Acceptable | Within SLO, monitor              |
| 500-1000ms | 🟠 Warning    | Investigate bottlenecks          |
| >1000ms    | 🔴 Critical   | Immediate investigation required |

### Error Rate Categories

| Rate   | Status        | Action                           |
| ------ | ------------- | -------------------------------- |
| <0.1%  | 🟢 Excellent  | No action needed                 |
| 0.1-1% | 🟡 Acceptable | Within SLO, monitor              |
| 1-5%   | 🟠 Warning    | Investigate errors               |
| >5%    | 🔴 Critical   | Immediate investigation required |

### Failure Classification

The test framework classifies failures to help with root cause analysis:

| Type                     | Description                             | Investigation Steps                    |
| ------------------------ | --------------------------------------- | -------------------------------------- |
| **Contract Failures**    | API contract violations (400, 401, 403) | Check API request format, auth headers |
| **Capacity Failures**    | Server overload (429, 500+, timeouts)   | Check resource utilization, scale up   |
| **Lifecycle Mismatches** | Game IDs disappear unexpectedly         | Check game cleanup logic, TTL settings |

## Bottleneck Investigation Guide

If baseline doesn't meet targets, investigate systematically:

### 1. Database (PostgreSQL)

Check for database bottlenecks:

```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity;

-- Check slow queries
SELECT query, calls, mean_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

**Common issues:**

- Connection pool exhaustion
- Missing indexes
- Lock contention

### 2. Redis Cache

Check Redis performance:

```bash
redis-cli info stats
redis-cli info memory
```

**Common issues:**

- Memory pressure
- High eviction rate
- Connection limits

### 3. AI Service

Check AI service performance:

```bash
# Check health and queue depth
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

**Common issues:**

- Slow model inference
- Request queue backup
- Memory exhaustion

### 4. WebSocket Server

Check WebSocket metrics in Prometheus:

**Common issues:**

- Connection limit reached
- Event loop lag
- Memory leaks

### 5. Application Server

Check Node.js metrics:

**Common issues:**

- Event loop blocking
- Memory leaks
- CPU saturation

## Integration with Monitoring

### Prometheus Metrics

The load test results should correlate with Prometheus metrics:

- `ringrift_games_active` - Current active games
- `ringrift_websocket_connections` - WebSocket connection count
- `ringrift_game_move_latency_seconds` - Move processing time
- `http_request_duration_seconds` - HTTP request latency

### Grafana Dashboards

Use the "RingRift Load Test" dashboard during baseline runs to:

- Visualize real-time metrics
- Correlate load test events with system behavior
- Capture screenshots for documentation

## CI/CD Integration

### Smoke Tests

Run smoke tests on every PR:

```yaml
- name: Load Test Smoke
  run: npm run load:smoke:all
```

### Nightly Baselines

Run full baseline tests nightly:

```yaml
- name: Nightly Baseline
  run: |
    npm run load:baseline:staging
    # Archive results
    cp tests/load/results/*.json artifacts/
```

### Regression Detection

Compare baseline results against previous runs:

```javascript
// Example regression detection logic
const previousBaseline = require('./baselines/previous.json');
const currentResults = require('./results/latest_summary.json');

if (currentResults.latency.p95 > previousBaseline.latency.p95 * 1.2) {
  console.error('REGRESSION: p95 latency increased by >20%');
  process.exit(1);
}
```

## Related Documents

- [STAGING_ENVIRONMENT.md](./STAGING_ENVIRONMENT.md) - Staging deployment guide
- [DEPLOYMENT_REQUIREMENTS.md](./DEPLOYMENT_REQUIREMENTS.md) - Production requirements
- [PROJECT_GOALS.md](../PROJECT_GOALS.md) - Overall project targets
- [STRATEGIC_ROADMAP.md](../STRATEGIC_ROADMAP.md) - Performance optimization plans
- [tests/load/README.md](../tests/load/README.md) - Load test documentation
- [tests/load/configs/thresholds.json](../tests/load/config/thresholds.json) - SLO definitions

## Appendix: Quick Commands

```bash
# Run local baseline
npm run load:baseline:local

# Run staging baseline
npm run load:baseline:staging

# Run target scale test (100 games / 300 players)
npm run load:target:staging

# Run local target scale test
npm run load:target:local

# Run stress test (find breaking point)
npm run load:stress:breaking

# Analyze baseline results
node tests/load/scripts/analyze-results.js tests/load/results/baseline*.json

# Analyze target scale results
node tests/load/scripts/analyze-target-scale.js tests/load/results/target-scale*.json

# View latest summary
cat tests/load/results/*_summary.json | jq '.slo.all_passed'

# Check if all SLOs passed
jq -e '.passed == true' tests/load/results/*_summary.json

# Check target scale validation
jq -e '.all_targets_met == true' tests/load/results/target-scale*_summary.json
```
