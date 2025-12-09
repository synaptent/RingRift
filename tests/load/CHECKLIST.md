# Load Test Pre-Flight Checklist

Use this checklist before running baseline or stress load tests to ensure accurate measurements.

## Environment Preparation

### Infrastructure

- [ ] **Server Running** - Target server is up and healthy

  ```bash
  curl -f http://localhost:3001/health
  ```

- [ ] **Database Ready** - PostgreSQL is running and accessible

  ```bash
  docker ps | grep postgres
  ```

- [ ] **Redis Available** - Redis cache is running (if used)

  ```bash
  docker ps | grep redis
  ```

- [ ] **AI Service Active** (optional) - AI service responds to health checks
  ```bash
  curl -f http://localhost:8000/health
  ```

### For Staging Environment

- [ ] **Docker Compose Up** - All containers running

  ```bash
  docker-compose -f docker-compose.staging.yml ps
  ```

- [ ] **Migrations Applied** - Database schema is current

  ```bash
  npm run db:migrate
  ```

- [ ] **Seed Data Present** - Load test users exist
  ```bash
  # Check if load test user exists
  # User: loadtest@example.com
  ```

### Resource Baseline

- [ ] **No Other Load** - System is idle, no other tests running
- [ ] **Memory Available** - Sufficient RAM for test (minimum 2GB free)
- [ ] **CPU Idle** - CPU utilization <20% before starting
- [ ] **Disk Space** - At least 1GB free for results

## Test Configuration

### k6 Installation

- [ ] **k6 Installed** - k6 binary is available

  ```bash
  k6 version
  # Expected: k6 v0.46.0 or later
  ```

- [ ] **Node.js Available** - For results analysis
  ```bash
  node --version
  # Expected: v18.0.0 or later
  ```

### Validate Test Syntax

- [ ] **Scenarios Valid** - No syntax errors in test files

  ```bash
  cd tests/load && npm run validate:syntax
  ```

- [ ] **Dry Run Passes** - Scenarios can parse and initialize
  ```bash
  cd tests/load && npm run validate:dry-run
  ```

## Pre-Run Smoke Test

Before running full baseline, verify with a quick smoke test:

```bash
# Quick smoke test (< 1 minute)
npm run load:smoke:all

# Or individual scenario
k6 run --env BASE_URL=http://localhost:3001 \
  --env LOAD_PROFILE=smoke \
  tests/load/scenarios/concurrent-games.js
```

- [ ] **Smoke Test Passes** - No errors or contract failures
- [ ] **Server Responds** - Health check still healthy after smoke

## Running the Baseline Test

### Option 1: npm scripts (Recommended)

```bash
# Local baseline
npm run load:baseline:local

# Staging baseline
npm run load:baseline:staging
```

Note: The baseline (`npm run load:baseline:*`) and target-scale (`npm run load:target:*`) harness scripts perform a login pre-flight against `${BASE_URL}/api/auth/login` using the configured `LOADTEST_EMAIL` and `LOADTEST_PASSWORD`. If this login fails, the script aborts before starting k6 and will typically suggest running `npm run load:seed-users` and verifying staging auth environment variables (for example `JWT_SECRET`, `JWT_REFRESH_SECRET`) and network connectivity to `${BASE_URL}`.

### Option 2: Direct script execution

```bash
cd tests/load
./scripts/run-baseline.sh --local
./scripts/run-baseline.sh --staging
```

### Option 3: Manual k6 run

```bash
k6 run \
  --env BASE_URL=http://localhost:3001 \
  --env LOAD_PROFILE=load \
  --out json=tests/load/results/baseline.json \
  tests/load/scenarios/concurrent-games.js
```

## During Test Execution

- [ ] **Monitor Server Logs** - Watch for errors

  ```bash
  tail -f logs/server.log
  ```

- [ ] **Watch Metrics** - If Prometheus/Grafana available
  - CPU utilization
  - Memory usage
  - Active connections
  - Request rate

- [ ] **No Interventions** - Don't restart services during test

## Post-Test Validation

### Results Analysis

- [ ] **Results File Exists** - JSON output was created

  ```bash
  ls -la tests/load/results/baseline_*.json
  ```

- [ ] **Run Analyzer** - Generate summary report

  ```bash
  npm run load:analyze tests/load/results/baseline_*.json
  ```

- [ ] **Check SLO Status** - Verify pass/fail status
  ```bash
  cat tests/load/results/baseline_*_summary.json | jq '.passed'
  ```

### Record Results

- [ ] **Update Baseline Doc** - Record measurements in `docs/BASELINE_CAPACITY.md`
- [ ] **Archive Results** - Keep JSON files for comparison
- [ ] **Note Any Issues** - Document unexpected behavior

## Troubleshooting Quick Reference

### Test Won't Start

| Symptom                 | Cause              | Fix                    |
| ----------------------- | ------------------ | ---------------------- |
| "k6: command not found" | k6 not installed   | `brew install k6`      |
| "Connection refused"    | Server not running | `npm run dev`          |
| "401 Unauthorized"      | Missing auth setup | Check test user exists |

### Test Fails Early

| Symptom           | Cause             | Fix                      |
| ----------------- | ----------------- | ------------------------ |
| Many 5xx errors   | Server overloaded | Reduce VU count          |
| Contract failures | API changes       | Update test expectations |
| Timeouts          | Network issues    | Check connectivity       |

### Results Look Wrong

| Symptom           | Cause          | Fix                     |
| ----------------- | -------------- | ----------------------- |
| 0 requests        | Wrong BASE_URL | Check env variable      |
| All requests fail | Auth issues    | Verify token generation |
| Very high latency | Debug mode on  | Disable logging/tracing |

## Quick Commands Reference

```bash
# Check everything is ready
npm run dev:doctor         # If available
curl http://localhost:3001/health

# Validate load tests
cd tests/load && npm run validate:syntax

# Run smoke test
npm run load:smoke:all

# Run baseline
npm run load:baseline:local

# Analyze results
npm run load:analyze tests/load/results/baseline*.json

# View summary
cat tests/load/results/*_summary.json | jq .
```

## Related Documentation

- [docs/BASELINE_CAPACITY.md](../../docs/BASELINE_CAPACITY.md) - Baseline tracking
- [tests/load/README.md](./README.md) - Load test overview
- [tests/load/configs/baseline.json](./configs/baseline.json) - Test configuration
- [tests/load/config/thresholds.json](./config/thresholds.json) - SLO definitions
