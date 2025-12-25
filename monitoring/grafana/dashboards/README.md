# RingRift Grafana Dashboards

## Overview

This directory contains Grafana dashboard definitions for monitoring the RingRift game server. Dashboards are automatically provisioned when the monitoring stack is started.

## Available Dashboards

### 1. Game Performance Dashboard (`game-performance.json`)

**Purpose:** Provides visibility into core game metrics and performance.

**Panels:**

1. **Active Games** (Time Series)
   - Metric: `ringrift_games_active`
   - Description: Number of currently active games
   - Alert Threshold: >100 games (approaching capacity - yellow at 80, red at 100)
   - Related Alert: Lines 304-314 in `monitoring/prometheus/alerts.yml` (NoActiveGames)

2. **Game Creation Rate** (Time Series)
   - Metric: `rate(ringrift_games_total[1m]) * 60`
   - Description: Games created per minute by type (pvp/ai)
   - Use Case: Capacity planning and trend analysis

3. **Game Duration Distribution** (Heatmap)
   - Metric: `ringrift_game_duration_seconds_bucket`
   - Description: Distribution of game lengths
   - Alert Threshold: >3600s (1 hour) indicates stuck games
   - Related Alert: Lines 349-362 in `monitoring/prometheus/alerts.yml` (LongRunningGames)

4. **Players Per Game** (Time Series)
   - Metric: `ringrift_users_active / ringrift_games_active`
   - Description: Average players per active game
   - Alert Threshold: <2 players (indicates configuration issue - red below 2)
   - Use Case: Detect misconfigured games or player dropout

5. **Turn Processing Time** (Time Series)
   - Metrics: P50, P95, P99 of `ringrift_game_move_latency_seconds`
   - Description: Latency percentiles for turn processing
   - Alert Thresholds:
     - Yellow: P95 >250ms
     - Red: P95 >500ms (performance degradation)
   - Related Alert: Lines 202-212 in `monitoring/prometheus/alerts.yml` (HighGameMoveLatency)

6. **Move Rate by Type** (Time Series)
   - Metric: `rate(ringrift_moves_total[1m]) * 60`
   - Description: Move activity per minute by type (placement/movement)
   - Use Case: Identify game flow anomalies and phase distribution

7. **Abnormal Game Terminations by Reason** (Time Series)
   - Metric: `sum(rate(ringrift_game_session_abnormal_termination_total[5m])) by (reason) * 60`
   - Description: Abnormal game session terminations per minute by reason (e.g., timeout, resignation, abandonment)
   - Use Case: High-level view of game completions driven by timeouts/abandonments vs normal victories
   - Related Alert: `AIErrorsIncreasing` / `LongRunningGames` (via `ringrift_game_session_abnormal_termination_total{reason=~"timeout|resignation|abandonment"}`)

8. **WebSocket Reconnection Attempts** (Time Series)
   - Metric: `sum(rate(ringrift_websocket_reconnection_total[5m])) by (result) * 60`
   - Description: WebSocket reconnection attempts per minute by result (success/failed/timeout)
   - Use Case: Detect elevated reconnect churn that may correlate with network instability or backend restarts

9. **AI Request Latency** (Time Series)
   - Metrics: P50, P95, P99 of `ringrift_ai_request_duration_seconds_bucket`
   - Description: Backend-observed AI request latency percentiles in milliseconds
   - Alert Threshold: P99 >5s (see `AIRequestHighLatency` in `alerts.yml` lines 414-422)
   - Use Case: Correlate slow AI responses with degraded game experience and timeout/fallback behaviour

10. **AI Request Outcomes & Fallbacks** (Time Series)
    - Metrics:
      - `sum(rate(ringrift_ai_requests_total[5m])) by (outcome)`
      - `sum(rate(ringrift_ai_fallback_total[5m])) by (reason)`
    - Description: AI request outcome mix (success/fallback/error/timeout) plus fallback rates by reason
    - Use Case: Quickly see when AI starts erroring or relying on fallbacks, and which failure modes dominate

**Refresh Rate:** 30 seconds (auto-refresh enabled)

**Time Range:** Last 1 hour (default)

### 2. Rules Correctness Dashboard (`rules-correctness.json`)

**Purpose:** Monitors rules engine behavior, parity between TS/Python implementations, and correctness metrics.

**Panels:**

1. **Parity Check Success Rate** (Time Series)
   - Metric: `sum(rate(ringrift_parity_checks_total{result="success"}[5m])) / sum(rate(ringrift_parity_checks_total[5m]))`
   - Description: Percentage of TS/Python parity checks passing
   - Alert Threshold: <99.9% (correctness issue - red below 0.999, yellow 0.99-0.999)
   - Related Alerts: Lines 545-558, 564-576 in `monitoring/prometheus/alerts.yml` (ParityCheckFailures)

2. **Contract Vector Pass Rate** (Stat Panel)
   - Metric: `ringrift_contract_tests_passing / ringrift_contract_tests_total`
   - Description: Current pass rate of 90 contract vectors
   - Alert Threshold: <100% (regression detected - red below 1.0, yellow 0.99-1.0)
   - Use Case: Immediate visibility into contract test regressions

3. **Rules Engine Errors** (Time Series)
   - Metric: `rate(ringrift_rules_errors_total[5m])` by error_type
   - Description: Rate of validation/mutation errors
   - Alert Threshold: >0 errors/sec (unexpected errors - red above 0.01)
   - Use Case: Detect unexpected rules engine failures by category

4. **Line/Territory Detection Time** (Time Series)
   - Metrics: P50, P95, P99 of `ringrift_line_detection_duration_ms`
   - Description: Performance of complex rule calculations
   - Alert Thresholds:
     - Yellow: P95 >50ms
     - Red: P95 >100ms (performance regression)
   - Use Case: Monitor computational complexity of territory/line detection

5. **Capture Chain Depth** (Heatmap)
   - Metric: `ringrift_capture_chain_depth_bucket`
   - Description: Distribution of chain capture lengths
   - Use Case: Identify infinite loop risks and unusual capture patterns

6. **Move Validation Failures** (Time Series)
   - Metric: `rate(ringrift_moves_rejected_total[5m])` by reason
   - Description: Why moves are being rejected
   - Use Case: Debug rule validation issues and identify patterns

**Refresh Rate:** 30 seconds (auto-refresh enabled)

**Time Range:** Last 1 hour (default)

### 3. System Health Dashboard (`system-health.json`)

**Purpose:** Monitors overall system health, infrastructure metrics, and service dependencies.

**Panels:**

1. **HTTP Request Rate** (Time Series)
   - Metric: `rate(http_requests_total[5m])` by status code
   - Description: Requests per second, color-coded by HTTP status
   - Alert Threshold: 5xx errors >1% of total requests
   - Related Alert: Lines 81-96 in `monitoring/prometheus/alerts.yml` (HighErrorRate >5%)

2. **HTTP Latency** (Time Series)
   - Metrics: P50, P95, P99 of `http_request_duration_seconds`
   - Description: API response time percentiles
   - Alert Threshold: P95 >1000ms (user experience degradation)
   - Related Alert: Lines 172-182 in `monitoring/prometheus/alerts.yml` (HighP95Latency >1s)

3. **WebSocket Connections** (Time Series)
   - Metric: `ringrift_websocket_connections`
   - Description: Currently active WebSocket connections
   - Alert Threshold: >500 (capacity approaching)
   - Related Alert: Lines 334-344 in `monitoring/prometheus/alerts.yml` (HighWebSocketConnections >1000)

4. **Database Query Performance** (Time Series)
   - Metrics: P50, P95, P99 of `ringrift_service_response_time_seconds{service="database"}`
   - Description: Database operation latency percentiles
   - Alert Threshold: P95 >100ms (database overload)
   - Related Alert: Lines 717-727 in `monitoring/prometheus/alerts.yml` (DatabaseResponseTimeSlow >500ms)

5. **Redis Cache Hit Rate** (Stat Panel + Time Series)
   - Metric: `ringrift_cache_hits_total / (ringrift_cache_hits_total + ringrift_cache_misses_total)`
   - Description: Percentage of cache hits (both current value and trend)
   - Alert Threshold: <80% (cache ineffective)
   - **Note:** Requires cache metrics implementation in MetricsService (currently placeholder)

6. **AI Service Health** (Stat Panel + Time Series)
   - Metric: `ringrift_service_status{service="ai_service"}`
   - Description: AI service availability (1=healthy, 0=degraded)
   - Alert Threshold: 0 (AI service down)
   - Related Alert: Lines 65-76 in `monitoring/prometheus/alerts.yml` (AIServiceDown)

7. **Memory Usage** (Time Series)
   - Metric: `process_resident_memory_bytes / (1024*1024)` (MB)
   - Description: Node.js process memory consumption
   - Alert Thresholds:
     - Yellow: >1536MB (1.5GB)
     - Red: >2048MB (2GB, memory leak suspected)
   - Related Alerts: Lines 223-248 in `monitoring/prometheus/alerts.yml` (HighMemoryUsage >1.5GB warning, >2GB critical)

8. **Event Loop Lag** (Time Series)
   - Metric: `nodejs_eventloop_lag_seconds * 1000` (convert to ms)
   - Description: Node.js event loop responsiveness
   - Alert Thresholds:
     - Yellow: >50ms
     - Red: >100ms (thread blocking)
   - Related Alerts: Lines 253-278 in `monitoring/prometheus/alerts.yml` (HighEventLoopLag >100ms warning, >500ms critical)

**Refresh Rate:** 30 seconds (auto-refresh enabled)

**Time Range:** Last 1 hour (default)

## Alert Threshold Alignment

All dashboard thresholds are aligned with Prometheus alert rules defined in `monitoring/prometheus/alerts.yml`:

| Dashboard Panel            | Alert Rule                    | Threshold        | Lines in alerts.yml |
| :------------------------- | :---------------------------- | :--------------- | :------------------ |
| **Game Performance**       |                               |                  |                     |
| Active Games               | NoActiveGames                 | 0 for 30min      | 304-314             |
| Game Duration              | LongRunningGames              | Median >3600s    | 349-362             |
| Turn Processing            | HighGameMoveLatency           | P99 >1s          | 202-212             |
| **Rules Correctness**      |                               |                  |                     |
| Parity Check Success Rate  | RulesParityValidationMismatch | >5 mismatches/hr | 545-558             |
| Contract Vector Pass Rate  | ContractTestRegressions       | <100%            | Not yet implemented |
| Rules Engine Errors        | HighRulesEngineErrors         | >10/min          | Not yet implemented |
| **System Health**          |                               |                  |                     |
| HTTP Request Rate          | HighErrorRate                 | 5xx >5%          | 81-96               |
| HTTP Latency               | HighP95Latency                | P95 >1s          | 172-182             |
| WebSocket Connections      | HighWebSocketConnections      | >1000            | 334-344             |
| Database Query Performance | DatabaseResponseTimeSlow      | P99 >500ms       | 717-727             |
| Redis Cache Hit Rate       | N/A (capacity planning)       | <80%             | Not yet implemented |
| AI Service Health          | AIServiceDown                 | 0 for 2min       | 65-76               |
| Memory Usage               | HighMemoryUsage / Critical    | >1.5GB / >2GB    | 223-248             |
| Event Loop Lag             | HighEventLoopLag / Critical   | >100ms / >500ms  | 253-278             |

**Note:** Some thresholds (e.g., contract test failures, rules engine errors, cache hit rate) are capacity planning indicators or correctness metrics not yet formalized as alerts.

## Usage

### Starting Grafana with Dashboards

```bash
# Start the monitoring stack (includes Prometheus, Grafana, Alertmanager)
docker-compose --profile monitoring up -d

# Access Grafana
open http://localhost:3002

# Default credentials
# Username: admin
# Password: admin (or value of GRAFANA_PASSWORD env var)
```

### Dashboard Access

After logging in to Grafana:

1. Navigate to **Dashboards** → **Browse**
2. Look for folder: **RingRift Dashboards**
3. Available dashboards:
   - **RingRift - Game Performance** - Core game metrics and performance
   - **RingRift - Rules Correctness** - Rules engine behavior and parity
   - **RingRift - System Health** - Infrastructure and service health

Dashboards will auto-load from `/etc/grafana/provisioning/dashboards/` within the container.

## Provisioning

Dashboards are automatically provisioned via:

- **Dashboard definitions:**
  - `monitoring/grafana/dashboards/game-performance.json`
  - `monitoring/grafana/dashboards/rules-correctness.json`
  - `monitoring/grafana/dashboards/system-health.json`
- **Provisioning config:** `monitoring/grafana/provisioning/dashboards.yml`
- **Datasource config:** `monitoring/grafana/provisioning/datasources.yml`
- **Docker volume mount:** Defined in `docker-compose.yml` lines 179-181

Changes to dashboard JSON files are picked up within 10 seconds (see `updateIntervalSeconds` in `dashboards.yml`).

## Metrics Reference

All metrics are exposed by [`src/server/services/MetricsService.ts`](../../../src/server/services/MetricsService.ts).

**Key Metrics Used:**

### Game Performance Metrics

- `ringrift_games_active` (Gauge) - Line 278
- `ringrift_games_total` (Counter) - Line 272
- `ringrift_game_duration_seconds` (Histogram) - Line 289
- `ringrift_users_active` (Gauge) - Line 297
- `ringrift_game_move_latency_seconds` (Histogram) - Line 365
- `ringrift_moves_total` (Counter) - Line 283

### Rules Correctness Metrics

- `ringrift_parity_checks_total` (Counter) - Implemented in `MetricsService`; instrumentation wired in select parity/soak harnesses
- `ringrift_contract_tests_passing` (Gauge) - Implemented in `MetricsService`; updated by contract/vector runners
- `ringrift_contract_tests_total` (Gauge) - Implemented in `MetricsService`; updated by contract/vector runners
- `ringrift_rules_errors_total` (Counter) - Implemented in `MetricsService`; reserved for hard validation/mutation failures
- `ringrift_line_detection_duration_ms` (Histogram) - Implemented in `MetricsService`; ready for line/territory detection timing
- `ringrift_capture_chain_depth` (Histogram) - Not yet implemented
- `ringrift_moves_rejected_total` (Counter) - Implemented and wired for WebSocket move rejections and decision timeouts

### System Health Metrics

- `http_requests_total` (Counter) - Line 248
- `http_request_duration_seconds` (Histogram) - Line 241
- `ringrift_websocket_connections` (Gauge) - Line 301
- `ringrift_service_response_time_seconds` (Histogram) - Line 320
- `ringrift_cache_hits_total` (Counter) - Implemented and wired via `CacheService` Redis helpers
- `ringrift_cache_misses_total` (Counter) - Implemented and wired via `CacheService` Redis helpers
- `ringrift_service_status` (Gauge) - Line 310
- `process_resident_memory_bytes` (Gauge) - Standard Node.js metric
- `nodejs_eventloop_lag_seconds` (Gauge) - Standard Node.js metric

## Implementation Notes

### Missing / Partially Wired Metrics

The following metrics are referenced in dashboards and now exist in
[`MetricsService.ts`](../../../src/server/services/MetricsService.ts), but their
instrumentation is either limited to specific harnesses or still being rolled
out across all hosts:

**Rules Correctness Dashboard:**

- `ringrift_parity_checks_total` - Runtime parity check results (parity/soak harnesses)
- `ringrift_contract_tests_passing` / `ringrift_contract_tests_total` - Contract vector pass rate (contract/vector runners)
- `ringrift_rules_errors_total` - Rules engine validation/mutation errors (reserved for hard failures)
- `ringrift_line_detection_duration_ms` - Line/territory detection performance (ready for adoption)
- `ringrift_moves_rejected_total` - Move validation failure reasons (wired for WebSocket move rejections and decision timeouts)
- `ringrift_capture_chain_depth` - Capture chain length distribution (**still TODO**)

**System Health Dashboard:**

- `ringrift_cache_hits_total` / `ringrift_cache_misses_total` - Redis cache metrics (wired via `CacheService`)

Future PASS22+ work can extend instrumentation for the partially wired metrics
and add the remaining `ringrift_capture_chain_depth` histogram where capture
chains are computed.

## Troubleshooting

### Dashboard Not Loading

1. Verify Grafana is running:

   ```bash
   docker-compose ps grafana
   ```

2. Check Grafana logs:

   ```bash
   docker-compose logs grafana
   ```

3. Verify volume mounts:
   ```bash
   docker-compose config | grep -A 5 grafana
   ```

### No Data in Panels

1. Verify Prometheus is scraping metrics:

   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

2. Check if app is exposing metrics:

   ```bash
   curl http://localhost:3000/metrics
   ```

3. Verify Prometheus datasource in Grafana:
   - Navigate to **Configuration** → **Data Sources**
   - Select **Prometheus**
   - Click **Test** button

### Metrics Not Updating

1. Check Prometheus scrape interval (default: 15s)
2. Verify app is generating metrics (make some game moves)
3. Check dashboard refresh interval (top-right corner, should be 30s)

## Troubleshooting for Rules Correctness Dashboard

### Metrics Missing for Rules Correctness

**Parity Checks:**

- Metric: `ringrift_parity_checks_total`
- Source: Shadow mode comparisons and runtime parity tests
- To trigger: Run games in shadow mode or execute parity test suites

**Contract Tests:**

- Metrics: `ringrift_contract_tests_passing`, `ringrift_contract_tests_total`
- Source: MetricsService (lines 156-180)
- To trigger: Run contract test suite

**Rules Errors:**

- Metric: `ringrift_rules_errors_total`
- Source: Rules engine validation/mutation failures
- To trigger: Invalid move attempts will increment these counters

**Line Detection:**

- Metric: `ringrift_line_detection_duration_ms`
- Source: Territory and line calculation timings
- To trigger: Make moves that trigger territory/line detection

**Capture Chains:**

- Metric: `ringrift_capture_chain_depth`
- Source: Capture chain processing
- To trigger: Execute captures in games

**Move Rejections:**

- Metric: `ringrift_moves_rejected_total`
- Source: Move validation failures
- To trigger: Attempt invalid moves

## Runbook Links

Each dashboard panel references specific runbooks for alert investigation:

- [High Error Rate](../../../docs/runbooks/HIGH_ERROR_RATE.md) - 5xx errors >5%
- [High Latency](../../../docs/runbooks/HIGH_LATENCY.md) - API response times
- [WebSocket Issues](../../../docs/runbooks/WEBSOCKET_ISSUES.md) - Connection problems
- [WebSocket Scaling](../../../docs/runbooks/WEBSOCKET_SCALING.md) - High connection counts
- [Database Performance](../../../docs/runbooks/DATABASE_PERFORMANCE.md) - Slow queries
- [AI Service Down](../../../docs/runbooks/AI_SERVICE_DOWN.md) - AI unavailability
- [High Memory](../../../docs/runbooks/HIGH_MEMORY.md) - Memory usage alerts
- [Event Loop Lag](../../../docs/runbooks/EVENT_LOOP_LAG.md) - Event loop blocking
- [Rules Parity](../../../docs/runbooks/RULES_PARITY.md) - TS/Python divergence
- [Game Performance](../../../docs/runbooks/GAME_PERFORMANCE.md) - Game metrics
- [Game Health](../../../docs/runbooks/GAME_HEALTH.md) - Long-running games
- [No Activity](../../../docs/runbooks/NO_ACTIVITY.md) - Zero active games

## Related Documentation

- Alert Rules: [`monitoring/prometheus/alerts.yml`](../../prometheus/alerts.yml)
- Alerting Thresholds: [`docs/operations/ALERTING_THRESHOLDS.md`](../../../docs/operations/ALERTING_THRESHOLDS.md)
- PASS21 Assessment: [`docs/PASS21_ASSESSMENT_REPORT.md`](../../../docs/archive/assessments/PASS21_ASSESSMENT_REPORT.md) (Section 2.1)
- Metrics Service: [`src/server/services/MetricsService.ts`](../../../src/server/services/MetricsService.ts)
- Contract Vectors: [`docs/CONTRACT_VECTORS_DESIGN.md`](../../../docs/rules/CONTRACT_VECTORS_DESIGN.md)
- Invariants Framework: [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](../../../docs/rules/INVARIANTS_AND_PARITY_FRAMEWORK.md)

## Version History

- **v1.0 (2025-12-01):** Initial Game Performance Dashboard
  - 6 panels covering core game metrics
  - Auto-provisioning configured
  - Alert threshold alignment documented
- **v1.1 (2025-12-01):** Rules Correctness Dashboard
  - 6 panels monitoring rules engine correctness
  - Parity check success rate tracking
  - Contract vector pass rate monitoring
  - Rules error tracking by type
  - Line/territory detection performance
  - Capture chain depth distribution
  - Move validation failure analysis
- **v1.2 (2025-12-01):** System Health Dashboard (PASS21 P0 Completion)
  - 8 panels monitoring infrastructure and service health
  - HTTP request rate and latency percentiles
  - WebSocket connection tracking
  - Database query performance
  - Redis cache hit rate (metrics TBD)
  - AI service availability
  - Memory usage monitoring
  - Event loop lag tracking
  - Completes PASS21 P0 critical dashboard suite
