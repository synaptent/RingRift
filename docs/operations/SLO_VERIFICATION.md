# RingRift SLO Verification Framework

> **Doc Status (2025-12-07): Active**
>
> This document describes the SLO verification framework for validating Service Level Objectives during load testing.

## Overview

Service Level Objectives (SLOs) define the targets for system performance and reliability. The SLO verification framework validates these targets during load testing, providing actionable reports for production readiness assessment.

> **Current status (2025-12-08):** Baseline and target-scale k6 JSON results are checked in under `tests/load/results/`. SLO verification consumes the **raw k6 JSON output** from these runs. Analyzer summaries (for example `*_summary.json`) are used for human-readable capacity reporting (for example in `docs/BASELINE_CAPACITY.md`), not as inputs to the SLO verifier.
>
> For a quick smoke run against a local dev server, start the app, then run:
>
> ```bash
> K6_EXTRA_ARGS="--vus 1 --duration 30s" tests/load/scripts/run-baseline.sh --local
> tests/load/scripts/run-slo-verification.sh local --skip-test --results-file tests/load/results/<baseline_file>.json
> ```
>
> or pass an existing raw results file via `--skip-test --results-file`.
>
> Load can be generated from a developer machine, CI runner, or cloud-hosted instances (for example AWS EC2/ECS tasks or equivalents on other providers), as long as the runner can reach the target environment and emits standard k6 JSON. The SLO verification scripts are environment-agnostic: they consume result files and configuration (for example the `--env` flag and thresholds) without depending on where the load generator itself is hosted.

## Quick Start

```bash
# Run full SLO verification pipeline (load test + verification + dashboard)
npm run slo:check

# Verify SLOs from an existing baseline results file (raw k6 JSON)
npm run slo:verify tests/load/results/baseline_staging_20251208_144949.json

# Generate HTML dashboard from the corresponding SLO report
npm run slo:dashboard tests/load/results/baseline_staging_20251208_144949_slo_report.json
```

`npm run slo:check` uses `tests/load/scripts/run-slo-verification.sh` under the hood; it automatically selects the latest **raw** `baseline_*.json` in `tests/load/results/` (excluding `*_summary.json` and `*_slo_*.json` artifacts).

If you already have a results JSON, you can skip the load test:

```bash
tests/load/scripts/run-slo-verification.sh --skip-test --results-file tests/load/results/baseline_staging_20251208_144949.json
```

## Defined SLOs

The following SLOs are derived from [`PROJECT_GOALS.md`](../PROJECT_GOALS.md) §4.1 and [`tests/load/config/thresholds.json`](../tests/load/config/thresholds.json):

### Critical Priority (Zero Tolerance)

| SLO                  | Target                       | Measurement                          | Source           |
| -------------------- | ---------------------------- | ------------------------------------ | ---------------- |
| Service Availability | ≥99.9%                       | successful_requests / total_requests | PROJECT_GOALS.md |
| Error Rate           | ≤1% (staging) / ≤0.5% (prod) | http_req_failed rate                 | thresholds.json  |
| Contract Failures    | 0                            | contract_failures_total              | thresholds.json  |
| Lifecycle Mismatches | 0                            | id_lifecycle_mismatches_total        | thresholds.json  |

### High Priority

| SLO                     | Target                      | Measurement                       | Source            |
| ----------------------- | --------------------------- | --------------------------------- | ----------------- |
| API Latency (p95)       | <500ms                      | http_req_duration p95             | PROJECT_GOALS.md  |
| Move Latency (p95)      | <500ms                      | move_latency p95                  | thresholds.json   |
| WebSocket Connect (p95) | <1000ms                     | ws_connecting p95                 | baseline.json     |
| AI Response (p95)       | <1000ms                     | ai_response_time p95              | PROJECT_GOALS.md  |
| Concurrent Games        | ≥100 (prod) / ≥20 (staging) | concurrent_games                  | target-scale.json |
| Concurrent Players      | ≥300 (prod) / ≥60 (staging) | concurrent_vus                    | target-scale.json |
| WebSocket Success Rate  | ≥99%                        | websocket_connection_success_rate | thresholds.json   |
| Move Stall Rate         | ≤0.5%                       | stalled_moves / total_moves       | thresholds.json   |

### Medium Priority

| SLO                 | Target  | Measurement                           | Source          |
| ------------------- | ------- | ------------------------------------- | --------------- |
| API Latency (p99)   | <2000ms | http_req_duration p99                 | baseline.json   |
| Game Creation (p95) | <2000ms | game_creation_time p95                | baseline.json   |
| AI Response (p99)   | <2000ms | ai_response_time p99                  | thresholds.json |
| AI Fallback Rate    | ≤1%     | ai_fallback_total / ai_requests_total | thresholds.json |

## Usage

### npm Scripts

```bash
# Verify SLOs from any k6 JSON results file
npm run slo:verify <results.json>

# Generate HTML dashboard from SLO report
npm run slo:dashboard <slo_report.json>

# Full pipeline: run load test + verify + dashboard (staging environment)
npm run slo:check

# Full pipeline against local server
npm run slo:check:local

# Full pipeline with production SLO thresholds
npm run slo:check:production
```

### Direct Script Usage

```bash
# Verify SLOs with console output (default)
node tests/load/scripts/verify-slos.js results/baseline_staging_20251208_144949.json

# Verify with JSON output
node tests/load/scripts/verify-slos.js results/baseline_staging_20251208_144949.json json

# Verify with Markdown output
node tests/load/scripts/verify-slos.js results/baseline_staging_20251208_144949.json markdown

# Verify with production thresholds
node tests/load/scripts/verify-slos.js results/baseline_staging_20251208_144949.json --env production

# Generate HTML dashboard
node tests/load/scripts/generate-slo-dashboard.js results/baseline_staging_20251208_144949_slo_report.json

# Run full verification pipeline
cd tests/load && ./scripts/run-slo-verification.sh staging
```

## Output Formats

### Console Output

Human-readable verification report with pass/fail status:

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      RingRift SLO Verification Report                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Environment: staging      Overall: 14/15 SLOs PASSED                         ║
║  ✅ All Critical SLOs Met                                                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  [CRITICAL]                                                                   ║
║  ✅ Service Availability                 99.95% /     99.9%                   ║
║  ✅ Error Rate                           0.05% /        1%                    ║
...
```

### JSON Output

Machine-readable report for CI/CD integration:

```json
{
  "timestamp": "2025-12-07T11:41:00.000Z",
  "environment": "staging",
  "overall_passed": true,
  "passed_count": 14,
  "total_count": 15,
  "breaches_by_priority": {
    "critical": 0,
    "high": 1,
    "medium": 0
  },
  "slos": { ... }
}
```

### Markdown Output

Documentation-friendly table for reports:

```markdown
# SLO Verification Report

**Date:** 2025-12-07T11:41:00.000Z
**Environment:** staging
**Overall:** 14/15 SLOs Passed

| SLO                  | Target | Actual | Status  | Priority |
| -------------------- | ------ | ------ | ------- | -------- |
| Service Availability | 99.9%  | 99.95% | ✅ Pass | critical |

...
```

### HTML Dashboard

Visual dashboard with:

- Overall status summary
- Breach count by priority
- Individual SLO cards with progress bars
- Color-coded pass/fail indicators

## Environment-specific Thresholds

The framework supports different thresholds for staging and production:

### Staging (Default)

- More permissive latency targets
- Lower capacity requirements (20 games, 60 players)
- Higher acceptable error rates (1%)

### Production

- Stricter latency targets
- Full capacity requirements (100 games, 300 players)
- Tighter error budgets (0.5%)

Select environment with `--env` flag:

```bash
node tests/load/scripts/verify-slos.js results.json --env production
```

## Priority Levels

### Critical

- **Impact:** Service unavailability or data corruption risk
- **Action on Breach:** Immediate alert (PagerDuty)
- **Error Budget:** 0.001% (practically zero tolerance)
- **SLOs:** Availability, Error Rate, Contract Failures, Lifecycle Mismatches

### High

- **Impact:** Degraded user experience
- **Action on Breach:** Team notification (Slack)
- **Error Budget:** 1%
- **SLOs:** Latency metrics, Capacity, Connection success

### Medium

- **Impact:** Suboptimal performance
- **Action on Breach:** Log warning for review
- **Error Budget:** 5%
- **SLOs:** p99 latencies, Game creation time, AI fallback

## CI/CD Integration

### Exit Codes

- `0`: All SLOs passed
- `1`: One or more SLOs breached
- `2`: Error (file not found, parse error)

### GitHub Actions Example

```yaml
jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install k6
        run: |
          sudo gpg -k
          sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6

      - name: Run Load Test
        run: npm run load:baseline:staging

      - name: Verify SLOs
        id: slo
        run: |
          RESULT_FILE=$(ls -t tests/load/results/baseline_*.json | head -1)
          npm run slo:verify $RESULT_FILE -- --env staging

      - name: Generate Dashboard
        if: always()
        run: |
          REPORT_FILE=$(ls -t tests/load/results/*_slo_report.json | head -1)
          npm run slo:dashboard $REPORT_FILE

      - name: Upload Artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: slo-report
          path: |
            tests/load/results/*_slo_report.json
            tests/load/results/*_slo_report.html
```

## File Structure

```
tests/load/
├── configs/
│   ├── slo-definitions.json    # Canonical SLO definitions
│   ├── baseline.json           # Baseline test config
│   └── target-scale.json       # Target scale test config
├── config/
│   └── thresholds.json         # Environment thresholds
├── scripts/
│   ├── verify-slos.js          # SLO verification script
│   ├── generate-slo-dashboard.js  # Dashboard generator
│   └── run-slo-verification.sh # Full pipeline runner
└── results/
    ├── baseline_staging_*.json      # k6 raw results
    ├── *_slo_report.json            # SLO verification reports
    └── *_slo_report.html            # HTML dashboards
```

## Extending the Framework

### Adding New SLOs

1. Add the SLO definition to [`tests/load/configs/slo-definitions.json`](../tests/load/configs/slo-definitions.json):

```json
{
  "slos": {
    "new_slo_key": {
      "name": "New SLO Name",
      "target": 100,
      "unit": "ms",
      "measurement": "k6_metric_name",
      "priority": "high",
      "source": "reference document"
    }
  }
}
```

2. Add metric extraction logic to [`verify-slos.js`](../tests/load/scripts/verify-slos.js) in `extractMetrics()` function.

3. Add verification logic in `verifySLOs()` function.

4. Add environment overrides if needed:

```json
{
  "environments": {
    "staging": {
      "overrides": {
        "new_slo_key": { "target": 200 }
      }
    }
  }
}
```

### Custom k6 Metrics

Ensure your k6 scenarios emit the required metrics:

```javascript
import { Trend, Counter } from 'k6/metrics';

const customLatency = new Trend('custom_metric_name');
const customCounter = new Counter('custom_count');

export default function () {
  customLatency.add(response.timings.duration);
  customCounter.add(1);
}
```

## Related Documentation

- [`PROJECT_GOALS.md`](../PROJECT_GOALS.md) - Project SLO targets (§4.1)
- [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md) - Phase-based SLO roadmap
- [`tests/load/config/thresholds.json`](../tests/load/config/thresholds.json) - Detailed thresholds
- [`docs/ALERTING_THRESHOLDS.md`](ALERTING_THRESHOLDS.md) - Production alerting thresholds
- [`docs/BASELINE_CAPACITY.md`](BASELINE_CAPACITY.md) - Baseline capacity documentation

## Troubleshooting

### No Metrics Data

If SLOs show "No data collected", ensure:

1. The load test scenario emits the required metrics
2. The test ran long enough to generate data points
3. The results file is in valid k6 JSON format

### All SLOs Failing

Check:

1. Server health before running tests
2. Correct BASE_URL configuration
3. Authentication is working
4. AI service is running (if testing AI metrics)

### Dashboard Not Generating

Ensure:

1. SLO verification ran successfully (check for `*_slo_report.json`)
2. Node.js is available
3. Write permissions on results directory
