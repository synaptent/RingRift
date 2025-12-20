# RingRift Load Testing Framework (k6)

> **Created:** 2025-12-01  
> **Status:** Ready for execution (prior runs may exist under `results/`)  
> **Purpose:** Production-scale validation and performance SLO verification

## Overview

This directory contains the k6 load testing framework for validating RingRift at production scale (100+ concurrent games, 200-300 players). The framework implements the load testing scenarios defined in [`STRATEGIC_ROADMAP.md §3`](../../STRATEGIC_ROADMAP.md) and validates SLOs from [`monitoring/prometheus/alerts.yml`](../../monitoring/prometheus/alerts.yml).

### Key Objectives

- **Validate production scale assumptions** - Confirm system can handle 100+ concurrent games
- **Verify SLO compliance** - Ensure latency and error rate thresholds are met
- **Identify bottlenecks** - Discover performance limits before production deployment
- **Establish baselines** - Document "healthy system" metric ranges for operations

## Directory Structure

```
tests/load/
├── README.md                 # This file
├── package.json              # npm scripts for running tests
├── auth/
│   └── helpers.js            # Shared login helper for all k6 scenarios
├── scenarios/
│   ├── game-creation.js      # Tests game creation rate and latency
│   ├── concurrent-games.js   # Tests 100+ simultaneous games
│   ├── player-moves.js       # Tests move polling + (optional) HTTP moves
│   ├── websocket-stress.js   # Tests WebSocket/Socket.IO connection limits (500+)
│   └── websocket-gameplay.js # Tests end-to-end Socket.IO gameplay RTT/stalls
└── config/
    ├── thresholds.json       # Performance SLO thresholds by environment
    └── scenarios.json        # Test scenario configurations (smoke/load/stress)
```

## Prerequisites

### 1. Install k6

**macOS (Homebrew):**

```bash
brew install k6
```

**Linux (Debian/Ubuntu):**

```bash
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

**Docker (any platform):**

```bash
docker pull grafana/k6:latest
```

**Verify installation:**

```bash
k6 version
# Should show k6 v0.46.0 or newer
```

### 2. Running Infrastructure

Load tests require a running RingRift instance:

```bash
# From project root
docker-compose up -d

# Verify services are healthy
curl http://localhost:3001/health
```

## Quick Start

### 1. Validate Syntax (No Execution)

Before running tests, validate that all scripts are syntactically correct:

```bash
cd tests/load
npm run validate:syntax
```

### 2. Dry Run (Show What Would Execute)

Preview test execution plan without actually running. Note that k6 does not
support a `--dry-run` flag in v1.x; this step uses `k6 inspect` under the hood
to validate options, thresholds, and module resolution without sending traffic:

```bash
npm run validate:dry-run
```

### 3. Run Smoke Tests (Quick Validation)

Execute minimal load tests for quick sanity checking:

```bash
npm run test:smoke:game-creation
npm run test:smoke:concurrent-games
```

### 4. Run Full Load Tests

Execute comprehensive load tests at target production scale:

```bash
# Run all scenarios sequentially
npm run test:load:all

# Or run individually
npm run test:load:game-creation
npm run test:load:concurrent-games
npm run test:load:player-moves
npm run test:load:websocket-stress
```

## Test Scenarios

### Shared Auth Helper

All k6 scenarios share a common login helper:

- [`auth/helpers.js`](auth/helpers.js:1)

This helper:

- Calls `POST /api/auth/login` with the canonical payload shape:

  ```json
  { "email": "loadtest_user_1@loadtest.local", "password": "TestPassword123!" }
  ```

- Allows overriding credentials via environment:
  - `LOADTEST_EMAIL`
  - `LOADTEST_PASSWORD`

- Returns `{ token, userId }`. Scenarios should use `getValidToken(...)` to cache per-VU auth and refresh shortly before expiry for long-running runs.

Using a single helper keeps the load tests aligned with the production auth contract and avoids re-registering users under load.

Auth refresh controls (optional):

- `LOADTEST_AUTH_TOKEN_TTL_S` – default JWT TTL (seconds) when the API does not return `expiresIn`.
- `LOADTEST_AUTH_REFRESH_WINDOW_S` – safety window before expiry to refresh a token.

Preflight checks (recommended for long runs):

- `npm run load:preflight -- --expected-vus 300 --expected-duration-s 1800`
- Validates backend health, auth `expiresIn`/TTL alignment, and user pool sizing.
- You can set `LOADTEST_EXPECTED_VUS` / `LOADTEST_EXPECTED_DURATION_S` instead of flags.
- Runner scripts under `tests/load/scripts/` call this automatically unless `SKIP_PREFLIGHT_CHECKS=true`.

Load-test user pool controls (optional):

- `LOADTEST_USER_POOL_SIZE`, `LOADTEST_USER_POOL_PASSWORD`
- `LOADTEST_USER_POOL_PREFIX`, `LOADTEST_USER_POOL_DOMAIN`

### Scenario 1: Game Creation (`game-creation.js`)

**Purpose:** Validate game creation latency and throughput under load.

**Load Pattern:**

- Ramp: 10 → 50 users over 1.5 minutes
- Sustain: 50 users for 2 minutes
- Total Duration: ~4 minutes

**SLO Targets (Staging):**

- `POST /api/games`: p95 ≤ 800ms, p99 ≤ 1500ms
- Error rate: < 1.0%

**Run:**

```bash
k6 run --env BASE_URL=http://localhost:3001 scenarios/game-creation.js
```

**Expected Output:**

```
✓ game created successfully
✓ game ID returned
✓ game config matches

checks.........................: 99.50% ✓ 2985  ✗ 15
http_req_duration..............: avg=342ms  p95=756ms  p99=1234ms
game_creation_latency_ms.......: avg=338ms  p95=748ms  p99=1198ms
game_creation_success_rate.....: 99.80%
```

### Scenario 2: Concurrent Games (`concurrent-games.js`)

**Purpose:** Test system behavior with 100+ simultaneous active games.

**Load Pattern:**

- Ramp: 50 → 100 VUs over 5 minutes (each VU = 1 game)
- Sustain: 100 games for 5 minutes
- Total Duration: ~13 minutes

**SLO Targets (Staging):**

- `GET /api/games/:id`: p95 ≤ 400ms, p99 ≤ 800ms
- Concurrent games: ≥ 100
- Error rate: < 1.0%

**Run:**

```bash
k6 run --env BASE_URL=http://localhost:3001 scenarios/concurrent-games.js
```

**Monitor:**

- Check Grafana "Game Performance" dashboard
- Watch `ringrift_games_active` metric in Prometheus
- Monitor memory usage: `process_resident_memory_bytes`

### Scenario 3: Player Moves (`player-moves.js`)

**Purpose:** Test move submission latency and turn processing throughput.

**Load Pattern:**

- Ramp: 20 → 40 VUs over 4 minutes
- Sustain: 40 games for 5 minutes
- Total Duration: ~10 minutes

**SLO Targets (Staging):**

- Move latency: p95 ≤ 300ms, p99 ≤ 600ms
- Stalled moves (>2s): < 0.5%
- Success rate: > 99%

**Run:**

```bash
k6 run --env BASE_URL=http://localhost:3001 scenarios/player-moves.js
```

**Notes:**

- This scenario:
  - Authenticates via the shared helper.
  - Creates AI-backed games via `POST /api/games` using the same payload shape as the frontend.
  - Polls `GET /api/games/:gameId` for state, exercising the read path and turn progression under load.
- HTTP move submission is currently **guarded behind a feature flag** in the script:

  ```js
  const MOVE_HTTP_ENDPOINT_ENABLED = false;
  ```

  This reflects the fact that, in production, moves are carried exclusively over WebSockets. When an HTTP move endpoint is introduced, this flag can be enabled and the payload adjusted to match the new contract.

For full WebSocket real-time move testing today, supplement with Playwright E2E tests.

### Scenario 4: WebSocket Stress (`websocket-stress.js`)

**Purpose:** Test WebSocket connection limits and stability.

This scenario connects to the same Socket.IO endpoint used by the frontend client, including:

- Path: `/socket.io/` with `EIO=4&transport=websocket`
- JWT access token provided in the `token` query parameter

so that `WebSocketServer` authentication and connection metrics are exercised under realistic, production-like handshake conditions.

**Socket.IO v4 Protocol Implementation:**

Unlike basic k6 WebSocket scenarios that send raw JSON payloads, this scenario fully implements the Socket.IO v4 / Engine.IO v4 wire protocol:

| Stage                 | k6 Action                                                           | Server Response                                          |
| :-------------------- | :------------------------------------------------------------------ | :------------------------------------------------------- |
| 1. Transport          | Open WebSocket to `/socket.io/?EIO=4&transport=websocket&token=...` | HTTP 101 Upgrade                                         |
| 2. Engine.IO OPEN     | Wait for `0{...}` (open packet with sid/pingInterval)               | Server sends `0{"sid":"...", "pingInterval":25000, ...}` |
| 3. Socket.IO CONNECT  | Send `40{}` (connect to default namespace)                          | Server sends `40{"sid":"..."}` (connect ACK)             |
| 4. Application Events | Send `42["lobby:subscribe", {...}]`                                 | Server sends `42["lobby:game_created", ...]` etc.        |
| 5. Keepalive          | Respond to `2` (ping) with `3` (pong)                               | Connection maintained                                    |

This protocol-correct implementation eliminates the "message parse error" failures that occurred with earlier versions of this scenario.

**Custom Metrics:**

| Metric                              | Description                                                    |
| :---------------------------------- | :------------------------------------------------------------- |
| `websocket_connection_success_rate` | Rate of successful WebSocket transport connections (HTTP 101)  |
| `websocket_handshake_success_rate`  | Rate of successful Socket.IO handshakes (CONNECT ACK received) |
| `websocket_protocol_errors`         | Count of protocol-level parse/framing errors                   |
| `websocket_connection_errors`       | Count of connection-level errors                               |
| `websocket_message_latency_ms`      | Latency trend for timestamped events                           |
| `websocket_connection_duration_ms`  | Connection duration trend                                      |

**Load Pattern:**

- Ramp: 100 → 500 connections over 7 minutes
- Sustain: 500+ connections for 5 minutes
- Total Duration: ~15 minutes

**SLO Targets (Staging):**

- Connection success: > 99%
- Message latency: p95 < 200ms
- Connection duration: p50 > 5 minutes

**Run:**

```bash
k6 run \
  --env BASE_URL=http://localhost:3001 \
  --env WS_URL=ws://localhost:3001 \
  scenarios/websocket-stress.js
```

**Monitor:**

- `ringrift_websocket_connections` gauge
- Alert threshold: >1000 connections (warning)
- **Lifecycle metrics to watch** (see `docs/operations/ALERTING_THRESHOLDS.md` and Grafana "Game Performance" / "System Health" dashboards):
  - `ringrift_websocket_reconnection_total{result}` – reconnection outcomes (success/failed/timeout); sustained `result="timeout"` spikes may trigger `WebSocketReconnectionTimeouts`.
  - `ringrift_game_session_status_current{status}` – in‑memory sessions by derived status; watch for all sessions stuck outside `active_turn` / `active_decision_phase` (see `GameSessionStatusSkew`).
  - `ringrift_game_session_abnormal_termination_total{reason}` – abnormal endings (for example `disconnect_timeout`, `internal_error`, `session_cleanup`); spikes may indicate lifecycle/infra issues (`AbnormalGameSessionTerminationSpike`).

### Scenario 5: WebSocket Gameplay (`websocket-gameplay.js`)

**Purpose:** Exercise the canonical WebSocket gameplay path end to end:

- Auth via `POST /api/auth/login` using the shared helper [`auth/helpers.js`](auth/helpers.js:1).
- Game creation via `POST /api/games` (human vs AI) using the same payload shape as the frontend.
- WebSocket connect to `/socket.io` using the Socket.IO v4 / Engine.IO v4 framing model.
- `join_game` followed by repeated `player_move_by_id` events over WebSockets.
- Tracks end-to-end move RTT, stalls, and success rates via:
  - `ws_move_rtt_ms`
  - `ws_moves_attempted_total`
  - `ws_move_success_rate`
  - `ws_move_stalled_total`
  - `ws_connection_success_rate`
  - `ws_handshake_success_rate`

This scenario is implemented in [`websocket-gameplay.js`](scenarios/websocket-gameplay.js:1) and follows the strategy described in [`LOAD_TEST_WEBSOCKET_MOVE_STRATEGY.md`](../../docs/LOAD_TEST_WEBSOCKET_MOVE_STRATEGY.md:1) and the transport decision ADR [`PLAYER_MOVE_TRANSPORT_DECISION.md`](../../docs/PLAYER_MOVE_TRANSPORT_DECISION.md:1).

**Usage Modes:**

- **Smoke mode (dev/CI):**
  - Selected with `WS_GAMEPLAY_MODE=smoke` (or by leaving both `WS_GAMEPLAY_MODE` and `ENABLE_WS_GAMEPLAY_THROUGHPUT` unset).
  - Low VU count (1–2) and short duration.
  - Each VU plays a small number of human-vs-AI games to completion.
  - Intended as a quick sanity check that:
    - The canonical WebSocket move path (auth → game creation → WebSocket connect → `join_game` → `player_move_by_id`) is healthy.
    - [`websocket-gameplay.js`](scenarios/websocket-gameplay.js:1) is emitting the expected custom metrics.

- **Baseline mode (staging baselines, 10–25 games):**
  - Selected with `WS_GAMEPLAY_MODE=baseline`.
  - Uses a ramped VU profile that clamps concurrent VUs into the 10–25 range.
  - Intended for 10–25 game WebSocket gameplay baselines in staging (aligned with `load-test:baseline`).

- **Throughput mode (staging/perf):**
  - Selected with `WS_GAMEPLAY_MODE=throughput`, or when `ENABLE_WS_GAMEPLAY_THROUGHPUT=true` (or `1`, `yes`, `on`) in the environment for legacy compatibility.
  - Higher VU counts (20–40) with a longer steady-state duration.
  - Each VU plays multiple short games, generating many overlapping move streams.
  - Intended for P‑01-style concurrency testing of WebSocket gameplay in staging or dedicated perf environments.

- **Target-scale mode (~100 games / 300 players, perf/pre‑prod):**
  - Selected with `WS_GAMEPLAY_MODE=target`.
  - Uses a VU shape aligned with [`target-scale.json`](config/../configs/target-scale.json:1) (≈300 concurrent VUs at peak).
  - Intended for explicit target-scale validation of WebSocket gameplay without changing default smoke/baseline runs.

**Environment Variables:**

This scenario uses the same base variables as other WebSocket tests plus a few gameplay tunables:

- `BASE_URL` – HTTP base URL for API calls (for example `http://localhost:3000` or `http://localhost:3001` depending on how you run dev / Docker).
- `WS_URL` – WebSocket base URL used to derive the `/socket.io` endpoint. When omitted, [`websocket-gameplay.js`](scenarios/websocket-gameplay.js:1) derives this from `BASE_URL` by swapping `http` → `ws` and `https` → `wss` (for example `ws://localhost:3000`).
- `WS_GAMEPLAY_MODE` – Explicit mode selector: one of `smoke`, `baseline`, `throughput`, or `target`. When not set, the scenario defaults to `smoke`, unless `ENABLE_WS_GAMEPLAY_THROUGHPUT` is truthy (in which case it falls back to `throughput` for legacy flows).
- Optional tuning:
  - `GAME_MAX_MOVES` – Maximum moves per game before the scenario retires it (default `40`).
  - `GAME_MAX_LIFETIME_S` – Maximum lifetime of a single game in seconds before forced retirement (default `600`).
  - `VU_MAX_GAMES` – Maximum games per VU before it idles (default `3` in smoke mode, higher in throughput/target modes).
  - `ENABLE_WS_GAMEPLAY_THROUGHPUT` – Legacy boolean toggle. When set to a truthy value (for example `true`, `1`, `yes`, `on`) and `WS_GAMEPLAY_MODE` is not set, enables the throughput scenario alongside the default smoke configuration.

**Example Commands:**

Smoke (local dev / CI, from the repository root):

```bash
BASE_URL=http://localhost:3000 \
WS_URL=ws://localhost:3000 \
WS_GAMEPLAY_MODE=smoke \
npx k6 run tests/load/scenarios/websocket-gameplay.js
```

Baseline (10–25 games, staging baseline run):

```bash
THRESHOLD_ENV=staging \
RINGRIFT_ENV=staging \
BASE_URL=http://localhost:3000 \
WS_URL=ws://localhost:3000 \
WS_GAMEPLAY_MODE=baseline \
npx k6 run tests/load/scenarios/websocket-gameplay.js
```

Throughput (staging/perf):

```bash
BASE_URL=https://staging-api.ringrift.example \
WS_URL=wss://staging-ws.ringrift.example \
WS_GAMEPLAY_MODE=throughput \
npx k6 run tests/load/scenarios/websocket-gameplay.js
```

Target-scale WebSocket gameplay (~300 players / 100 games, perf/pre‑prod):

```bash
BASE_URL=https://perf-api.ringrift.example \
WS_URL=wss://perf-ws.ringrift.example \
WS_GAMEPLAY_MODE=target \
npx k6 run tests/load/scenarios/websocket-gameplay.js
```

By default, only the smoke configuration is active; higher-intensity modes are enabled explicitly via `WS_GAMEPLAY_MODE` (preferred) or the legacy `ENABLE_WS_GAMEPLAY_THROUGHPUT` toggle.

The concrete URLs and TLS configuration depend on your deployment topology (for example, whether WebSocket and HTTP traffic are served from the same origin). See [`DEPLOYMENT_REQUIREMENTS.md`](../../docs/DEPLOYMENT_REQUIREMENTS.md:1) and the topology notes in [`TOPOLOGY_MODES.md`](../../docs/architecture/TOPOLOGY_MODES.md:1) for environment-specific details.

## Configuration

### Environment Variables

All scenarios support these environment variables:

| Variable   | Default                 | Description          |
| :--------- | :---------------------- | :------------------- |
| `BASE_URL` | `http://localhost:3001` | HTTP API base URL    |
| `WS_URL`   | `ws://localhost:3001`   | WebSocket server URL |

**Examples:**

```bash
# Test against staging environment
BASE_URL=https://staging.ringrift.com k6 run scenarios/game-creation.js

# Test against local Docker with custom port
BASE_URL=http://localhost:8080 k6 run scenarios/concurrent-games.js
```

### Threshold Configuration

SLO thresholds are defined in [`config/thresholds.json`](config/thresholds.json) with separate profiles for:

- **Staging** - Permissive thresholds for testing (p95 < 800ms)
- **Production** - Strict thresholds for user-facing deployment (p95 < 400ms)

To validate against production thresholds, manually edit the script's `thresholds` section or use k6's `--env` flag:

```bash
# Override threshold in scenario
k6 run --env THRESHOLD_P95=400 scenarios/game-creation.js
```

### Scenario Profiles

[`config/scenarios.json`](config/scenarios.json) defines 5 test profiles:

1. **smoke** - Quick validation (5 VUs, 1 minute)
2. **load** - Realistic production load (50 VUs, 5 minutes)
3. **stress** - Maximum capacity test (100 VUs, 13 minutes)
4. **spike** - Sudden traffic surge (100 VUs in 30s)
5. **soak** - Extended stability test (40 VUs, 1 hour)

## Interpreting Results

### Success Criteria

A test passes when:

- ✓ All `checks` show > 99% pass rate
- ✓ `http_req_duration` p95/p99 within SLO thresholds
- ✓ `http_req_failed` rate < 1.0% (staging) or < 0.5% (production)
- ✓ Custom metrics (e.g., `game_creation_success_rate`) meet targets

### Example Output Analysis

```
✓ game created successfully
✓ game ID returned

checks.........................: 99.50% ✓ 2985  ✗ 15   <-- 99.5% success (PASS)
http_req_duration..............: avg=342ms  p95=756ms  p99=1234ms  <-- p95 under 800ms (PASS)
http_req_failed................: 0.75%   ✓ 15   ✗ 1985   <-- < 1% error rate (PASS)
game_creation_latency_ms.......: avg=338ms  p95=748ms  p99=1198ms  <-- Within SLO (PASS)
```

### Failure Indicators

🔴 **FAIL** - Immediate action required:

```
http_req_duration..............: p95=1850ms  <-- Exceeds 800ms threshold
http_req_failed................: 5.2%       <-- > 1% error rate
```

🟡 **WARNING** - Investigate but not blocking:

```
http_req_duration..............: p95=750ms  <-- Close to threshold (800ms)
websocket_reconnections_total..: 47         <-- High reconnection count
```

### Common Failure Causes

| Symptom               | Likely Cause                               | Investigation                                           |
| :-------------------- | :----------------------------------------- | :------------------------------------------------------ |
| High p95/p99 latency  | Database queries slow, event loop lag      | Check `nodejs_eventloop_lag_seconds`, DB query logs     |
| High error rate (5xx) | Service crashes, OOM, dependency failures  | Check logs, memory usage, AI service health             |
| Stalled moves         | AI timeout, deadlock, rules processing bug | Check `ai_move_latency_ms`, AI service logs             |
| Connection failures   | WebSocket limit reached, nginx timeout     | Check `ringrift_websocket_connections`, connection pool |

## Integration with Monitoring

### Prometheus Metrics

k6 results correlate with these Prometheus metrics:

| k6 Metric                      | Prometheus Metric                         | Dashboard Panel                       |
| :----------------------------- | :---------------------------------------- | :------------------------------------ |
| `http_req_duration`            | `http_request_duration_seconds`           | Infrastructure → HTTP Latency         |
| `game_creation_latency_ms`     | `ringrift_game_creation_duration_seconds` | Game Performance → Creation Time      |
| `move_submission_latency_ms`   | `ringrift_game_move_latency_seconds`      | Game Performance → Move Latency       |
| `websocket_connections_active` | `ringrift_websocket_connections`          | System Health → WebSocket Connections |
| `ai_move_latency_ms`           | `ringrift_ai_request_duration_seconds`    | AI Service Health → Request Latency   |

### Grafana Dashboards

After running load tests, review these dashboards (requires `docker-compose --profile monitoring up`):

1. **Game Performance** (`http://localhost:3000/d/game-performance`)
   - Active games over time
   - Game creation latency
   - Move processing latency
   - Abnormal terminations

2. **AI Service Health** (`http://localhost:3000/d/ai-service-health`)
   - AI request latency
   - Fallback rate
   - Concurrent AI requests

3. **Infrastructure Metrics** (`http://localhost:3000/d/system-health`)
   - HTTP request latency
   - WebSocket connections
   - Memory usage
   - Event loop lag

### Alerting

Load tests may trigger these alerts (defined in [`monitoring/prometheus/alerts.yml`](../../monitoring/prometheus/alerts.yml)):

- `HighP95Latency` - p95 > 1s sustained
- `HighMemoryUsage` - Process memory > 1.5GB
- `HighWebSocketConnections` - > 1000 connections
- `AIFallbackRateHigh` - > 30% AI requests falling back
- `WebSocketReconnectionTimeouts` - sustained `ringrift_websocket_reconnection_total{result="timeout"}` spikes (reconnect window failures)
- `AbnormalGameSessionTerminationSpike` - elevated `ringrift_game_session_abnormal_termination_total{reason=~"disconnect_timeout|internal_error|session_cleanup"}` during tests
- `GameSessionStatusSkew` - in‑memory sessions present but none in `active_turn` / `active_decision_phase` (possible stuck lifecycle)

**Expected behavior:** Alerts should NOT fire during normal load tests at target scale.

## Troubleshooting

### Issue: "connection refused" errors

**Symptom:**

```
ERRO[0002] dial tcp 127.0.0.1:3001: connect: connection refused
```

**Solution:**

```bash
# Verify services are running
docker-compose ps
docker-compose up -d

# Check health
curl http://localhost:3001/health
```

### Issue: High error rates (>5%)

**Symptom:**

```
http_req_failed: 12.5% ✓ 250 ✗ 1750
```

**Investigation:**

1. Check server logs: `docker-compose logs backend`
2. Check database connections: `docker-compose logs database`
3. Reduce load (lower VU count) to isolate issue
4. Check Grafana for resource exhaustion (memory, CPU)

### Issue: WebSocket connection failures

**Symptom:**

```
websocket_connection_errors: 87
websocket_connection_success_rate: 82.6%
```

**Solutions:**

- Verify `WS_URL` environment variable is correct
- Check nginx WebSocket proxy configuration (if applicable)
- Review WebSocket server connection limits
- Inspect `ringrift_websocket_connections` metric

### Issue: k6 crashes or hangs

**Symptom:** k6 process terminates unexpectedly or becomes unresponsive

**Solutions:**

- Reduce VU count: Edit scenario `target` values in scripts
- Increase system limits:
  ```bash
  ulimit -n 10000  # macOS/Linux file descriptor limit
  ```
- Use Docker k6 with resource limits:
  ```bash
  docker run --rm -v $(pwd):/scripts grafana/k6 run /scripts/scenarios/game-creation.js
  ```

## Best Practices

### 1. Progressive Load Testing

Execute tests in this order:

```bash
# Step 1: Validate infrastructure
npm run validate:syntax
npm run validate:dry-run

# Step 2: Smoke test
npm run test:smoke:game-creation

# Step 3: Baseline load
npm run test:load:game-creation
npm run test:load:concurrent-games

# Step 4: Production scale
npm run test:load:websocket-stress  # 500+ connections

# Step 5: Stress test (optional)
# Manually run with stress profile from scenarios.json
```

### 2. Monitor While Testing

Always have Grafana open during load tests:

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Open Grafana
open http://localhost:3000

# Run test in another terminal
npm run test:load:concurrent-games
```

### 3. Document Baseline Metrics

After successful load tests, document "healthy" metric ranges:

```markdown
## Baseline Metrics (2025-12-01)

- Game creation p95: 650ms (target <800ms)
- Move submission p95: 245ms (target <300ms)
- Peak concurrent games: 107 (target 100+)
- Memory usage at peak: 1.2GB (limit 2GB)
- AI fallback rate: 0.3% (target <1%)
```

### 4. Environment-Specific Testing

**Staging:**

- Run full suite (smoke + load + stress)
- Validate against staging thresholds
- Test with production-like data volumes

**Production:**

- **ONLY run smoke tests**
- Never run stress/spike tests in production
- Use synthetic monitoring for ongoing validation

## Advanced Usage

### Custom VU Ramp Patterns

Edit scenario files to customize load patterns:

```javascript
export const options = {
  stages: [
    { duration: '2m', target: 20 }, // Your custom ramp
    { duration: '5m', target: 75 }, // Your custom sustain
    { duration: '1m', target: 0 }, // Ramp down
  ],
};
```

### Output to File

Save results for later analysis:

```bash
k6 run --out json=results.json scenarios/game-creation.js

# View summary
k6 inspect results.json
```

### Integration with CI/CD

Example GitHub Actions workflow:

```yaml
- name: Run Load Tests
  run: |
    npm run validate:syntax
    npm run test:smoke:game-creation
```

## References

### Documentation

- [`STRATEGIC_ROADMAP.md §3`](../../STRATEGIC_ROADMAP.md) - Load testing scenarios and SLOs
- [`monitoring/prometheus/alerts.yml`](../../monitoring/prometheus/alerts.yml) - Alert definitions
- [`docs/PASS21_ASSESSMENT_REPORT.md`](../../docs/PASS21_ASSESSMENT_REPORT.md) - Operations readiness assessment

### External Resources

- [k6 Documentation](https://k6.io/docs/)
- [k6 WebSocket Testing](https://k6.io/docs/using-k6/protocols/websockets/)
- [k6 Thresholds](https://k6.io/docs/using-k6/thresholds/)
- [Grafana k6 Dashboard](https://grafana.com/grafana/dashboards/)

## Support

For issues or questions:

1. Check this README troubleshooting section
2. Review [`KNOWN_ISSUES.md`](../../KNOWN_ISSUES.md) for active blockers
3. Consult [`docs/runbooks/GAME_PERFORMANCE.md`](../../docs/runbooks/GAME_PERFORMANCE.md)
4. Open an issue in the project repository

---

**Last Updated:** 2025-12-01  
**Framework Version:** 1.0.0  
**k6 Version Required:** ≥ 0.46.0
