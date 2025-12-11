# RingRift Staging Environment

> **Doc Status (2025-12-07): Active**
>
> **Role:** Operational guide for the staging environment configured to mirror production topology for meaningful load testing and validation.
>
> **Related docs:** [`docs/planning/DEPLOYMENT_REQUIREMENTS.md`](planning/DEPLOYMENT_REQUIREMENTS.md), [`docs/runbooks/DEPLOYMENT_INITIAL.md`](runbooks/DEPLOYMENT_INITIAL.md), [`PROJECT_GOALS.md`](../PROJECT_GOALS.md), [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md)

## Overview

The staging environment is configured to mirror production topology, enabling meaningful load testing and validation before production deployment. It targets the scale requirements from `PROJECT_GOALS.md`:

- **100 concurrent games**
- **300 concurrent players**

### SLOs to Validate

| Metric           | Target       | Measurement Method                                |
| ---------------- | ------------ | ------------------------------------------------- |
| HTTP API latency | <500ms (p95) | Prometheus `http_request_duration_seconds`        |
| Error rate       | <1%          | Prometheus `http_requests_total{status=~"5.."}`   |
| AI move latency  | <1s (p95)    | Prometheus `ringrift_ai_request_duration_seconds` |
| System uptime    | >99.9%       | Prometheus `up{}`                                 |

---

## Prerequisites

### System Requirements

| Requirement        | Minimum | Recommended |
| ------------------ | ------- | ----------- |
| **RAM**            | 8GB     | 16GB        |
| **CPU**            | 4 cores | 8 cores     |
| **Disk**           | 20GB    | 50GB        |
| **Docker**         | v20+    | Latest      |
| **Docker Compose** | v2+     | Latest      |

### Port Availability

Ensure these ports are available before deployment:

| Port | Service      | Description                  |
| ---- | ------------ | ---------------------------- |
| 3000 | API Server   | HTTP API endpoints           |
| 3001 | WebSocket    | Real-time game communication |
| 3002 | Grafana      | Monitoring dashboards        |
| 5432 | PostgreSQL   | Database                     |
| 6379 | Redis        | Cache and sessions           |
| 8001 | AI Service   | Python AI backend            |
| 9090 | Prometheus   | Metrics collection           |
| 9093 | Alertmanager | Alert routing                |

---

## Quick Start

### 1. Configure Secrets

Copy and configure the staging environment file:

```bash
# The .env.staging file is version-controlled as a template
# Edit it to replace all placeholders
nano .env.staging
```

**Required Placeholders to Replace:**

| Placeholder                               | Description               | How to Generate           |
| ----------------------------------------- | ------------------------- | ------------------------- |
| `<STAGING_DB_PASSWORD>`                   | PostgreSQL password       | `openssl rand -base64 32` |
| `<STAGING_JWT_SECRET_REPLACE_ME>`         | JWT signing key           | `openssl rand -base64 48` |
| `<STAGING_JWT_REFRESH_SECRET_REPLACE_ME>` | JWT refresh key           | `openssl rand -base64 48` |
| `<STAGING_REDIS_PASSWORD>`                | Redis password (optional) | `openssl rand -base64 32` |
| `<STAGING_GRAFANA_PASSWORD>`              | Grafana admin password    | Choose a secure password  |

### 2. Deploy Staging

```bash
# Full deployment with health checks
./scripts/deploy-staging.sh

# Force rebuild images
./scripts/deploy-staging.sh --build

# Clean start (removes all data)
./scripts/deploy-staging.sh --clean
```

### 3. Verify Deployment

```bash
# Check application health
curl http://localhost:3000/health

# Check AI service health
curl http://localhost:8001/health

# View all service status
docker compose -f docker-compose.staging.yml ps
```

### 4. Teardown

```bash
# Stop services (preserve data)
./scripts/teardown-staging.sh

# Remove everything including data
./scripts/teardown-staging.sh --all

# Force teardown (no prompts)
./scripts/teardown-staging.sh --all --force
```

---

## Service Topology

```
                          ┌──────────────┐
                          │    nginx     │
                          │   :80/:443   │
                          └──────┬───────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │     app      │
                          │  :3000/:3001 │
                          └──────┬───────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │   postgres   │      │    redis     │      │  ai-service  │
    │    :5432     │      │    :6379     │      │    :8001     │
    └──────────────┘      └──────────────┘      └──────────────┘

                    MONITORING STACK
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  prometheus  │──│ alertmanager │  │   grafana    │
    │    :9090     │  │    :9093     │  │    :3002     │
    └──────────────┘  └──────────────┘  └──────────────┘
```

---

## Resource Allocation

The staging environment is configured with production-like resource limits:

| Component        | CPU Limit | Memory Limit | Notes                     |
| ---------------- | --------- | ------------ | ------------------------- |
| **app**          | 1         | 1GB          | Node.js API server        |
| **ai-service**   | 2         | 2GB          | Python AI backend         |
| **postgres**     | 2         | 2GB          | Tuned for 200 connections |
| **redis**        | 0.5       | 512MB        | 256MB cache with LRU      |
| **prometheus**   | 0.5       | 512MB        | 15-day retention          |
| **grafana**      | 0.5       | 512MB        | With dashboards           |
| **alertmanager** | 0.25      | 128MB        | Alert routing             |
| **nginx**        | 0.5       | 256MB        | Reverse proxy             |
| **Total**        | ~7.25     | ~7.4GB       | Fits on 8GB+ system       |

---

## PostgreSQL Configuration

The staging PostgreSQL is tuned for production-like performance:

```ini
max_connections=200
shared_buffers=512MB
effective_cache_size=1536MB
maintenance_work_mem=128MB
work_mem=16MB
wal_buffers=16MB
checkpoint_completion_target=0.9
random_page_cost=1.1
effective_io_concurrency=200
min_wal_size=1GB
max_wal_size=4GB
log_min_duration_statement=500
```

These settings mirror production recommendations for a system handling 100+ concurrent games.

---

## Redis Configuration

Redis is configured for production-like caching behavior:

```conf
maxmemory 256mb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
save 900 1
save 300 10
save 60 10000
tcp-keepalive 300
```

---

## Cloud-hosted staging environments and load-testing instances

Beyond the localhost Docker-based staging described above, operators may optionally run RingRift in a remote staging environment hosted on AWS or another cloud provider. A typical setup includes:

- An HTTPS-accessible staging API base URL, for example `https://staging.example.com`.
- A corresponding WebSocket endpoint, for example `wss://staging.example.com` (often the same origin as the HTTP API).
- One or more cloud-hosted load-generator instances (for example, EC2 instances, ECS tasks, or equivalent on other providers) that run the same k6-based load tests found under `tests/load/**`.

In this model:

- The game backend, AI service, and supporting services run in a cloud network (for example, a VPC) behind a public or private load balancer.
- Load generators are pointed at the remote staging base URLs via environment variables such as `BASE_URL` and `WS_URL`, matching the patterns described in `tests/load/README.md`.
- Orchestration can be handled by CI/CD pipelines or infrastructure-as-code (for example, Terraform-managed ECS services), depending on the operator's preferences.

Cloud-hosted staging is intentionally **deployment-specific**:

- This repository does not encode any particular cloud account, region, or hostname.
- Domains like `staging.example.com` are placeholders; teams should configure their own DNS entries, TLS certificates, network topology, and secrets.
- Environment variables and CI secrets are the primary mechanism for passing URLs and credentials into both the application and k6 runners.

For examples of how the AI service and training workloads can take advantage of cloud infrastructure, see `ai-service/docs/CLOUD_TRAINING_INFRASTRUCTURE_PLAN.md`. That plan is illustrative and not a requirement for basic game staging.

## Load Testing

The staging environment is the canonical target for the **Production Validation Contract** scenarios defined in
[`docs/PRODUCTION_READINESS_CHECKLIST.md`](PRODUCTION_READINESS_CHECKLIST.md:104):

- Baseline smoke: `BCAP_STAGING_BASELINE_20G_60P`
- Target-scale: `BCAP_SQ8_3P_TARGET_100G_300P`
- AI-heavy: `BCAP_SQ8_4P_AI_HEAVY_75G_300P`

These scenarios are driven by the k6 runners under `tests/load/scripts/**` and assume that the Docker-based
staging stack is already up and **healthy** (app, AI service, Postgres, Redis).

### How to bring up staging for production-validation load tests

From a clean clone with Docker running:

```bash
# 1) Deploy/refresh the local Docker-based staging stack
./scripts/deploy-staging.sh --clean
```

This script:

- Builds all images using [`docker-compose.staging.yml`](../docker-compose.staging.yml:1).
- Starts `postgres`, `redis`, `ai-service`, `app`, Prometheus, Grafana, Alertmanager, and nginx.
- Waits for:
  - PostgreSQL (`pg_isready` against the `postgres` service).
  - Redis (`redis-cli ping` inside the `redis` service).
  - AI service at `http://localhost:8001/health`.
  - App liveness at `http://localhost:3000/health`.
  - App readiness at `http://localhost:3000/ready`.

If any of these checks fail to become healthy within the timeout window, the script exits non-zero and prints
hints for inspecting logs so CI/automation can short-circuit.

To stop and clean up the stack when you are done:

```bash
# Stop services but preserve data volumes
./scripts/teardown-staging.sh

# Destroy containers + volumes (full reset)
./scripts/teardown-staging.sh --volumes
```

For a full reset (including volumes) followed by a clean deployment, run:

```bash
./scripts/teardown-staging.sh --volumes --force
./scripts/deploy-staging.sh --clean
```

### Running the BCAP baseline, target-scale, and AI-heavy runners against staging

Once staging is healthy on `http://localhost:3000` with AI service on `http://localhost:8001`, you can run the
three contract scenarios using the k6 runners. These commands assume you have a working load-test user account
(`LOADTEST_EMAIL` / `LOADTEST_PASSWORD`) that can log in to the target environment.

> **Tip:** You can seed a pool of `loadtest_user_*@loadtest.local` accounts via
> `npm run load:seed-users`. When running against the Docker-based staging DB, be sure your `DATABASE_URL`
> points at `localhost:5432` with the correct `DB_PASSWORD` from `.env.staging`.

#### 1. Baseline smoke: `BCAP_STAGING_BASELINE_20G_60P`

```bash
# From repo root, with staging already deployed
export LOADTEST_EMAIL="<your_loadtest_user_email>"
export LOADTEST_PASSWORD="<your_loadtest_user_password>"

SEED_LOADTEST_USERS=true \
  tests/load/scripts/run-baseline.sh --staging
```

- Targets the local Docker-based staging stack at `http://localhost:3000` by default.
- Writes raw k6 JSON and summaries under `tests/load/results/` with filenames prefixed by the
  scenario ID, for example `BCAP_STAGING_BASELINE_20G_60P_staging_*.json`.

#### 2. Target-scale: `BCAP_SQ8_3P_TARGET_100G_300P`

```bash
# From repo root, with staging already deployed and healthy
export LOADTEST_EMAIL="<your_loadtest_user_email>"
export LOADTEST_PASSWORD="<your_loadtest_user_password>"

SEED_LOADTEST_USERS=true \
  tests/load/scripts/run-target-scale.sh --staging
```

- Uses **production** thresholds for k6 (`THRESHOLD_ENV=production`) while still pointing at staging.
- Emits results under `tests/load/results/BCAP_SQ8_3P_TARGET_100G_300P_staging_*.json` and corresponding
  `*_summary.json` files for SLO aggregation.

#### 3. AI-heavy: `BCAP_SQ8_4P_AI_HEAVY_75G_300P`

```bash
# From repo root, with staging already deployed and healthy
export LOADTEST_EMAIL="<your_loadtest_user_email>"
export LOADTEST_PASSWORD="<your_loadtest_user_password>"

SEED_LOADTEST_USERS=true \
  tests/load/scripts/run-ai-heavy.sh --staging
```

- Uses staging thresholds for k6 (`THRESHOLD_ENV=staging`) and applies stricter, production-level SLOs
  for AI latency/fallback at the SLO verification layer, as described in
  [`docs/PRODUCTION_READINESS_CHECKLIST.md`](PRODUCTION_READINESS_CHECKLIST.md:176) and
  [`docs/SLO_VERIFICATION.md`](SLO_VERIFICATION.md:45).

### Where results and SLO reports are written

All three runners write their primary artifacts under `tests/load/results/`:

- Raw k6 JSON:
  - `tests/load/results/BCAP_STAGING_BASELINE_20G_60P_ENV_TIMESTAMP.json`
  - `tests/load/results/BCAP_SQ8_3P_TARGET_100G_300P_ENV_TIMESTAMP.json`
  - `tests/load/results/BCAP_SQ8_4P_AI_HEAVY_75G_300P_ENV_TIMESTAMP.json`
- Per-scenario k6 summaries produced via `handleSummary`:
  - `tests/load/results/*_summary.json`
- SLO verification reports generated by [`tests/load/scripts/verify-slos.js`](../tests/load/scripts/verify-slos.js:1)
  (invoked via `npm run slo:verify`):
  - `tests/load/results/*_slo_report.json`

After running the three BCAP scenarios you can aggregate the summary artifacts into a single go/no-go
view using:

```bash
npx ts-node scripts/analyze-load-slos.ts
```

which writes `load_slo_summary.json` and prints a compact table of scenario statuses, as described in
[`docs/PRODUCTION_READINESS_CHECKLIST.md`](PRODUCTION_READINESS_CHECKLIST.md:245).

### Target Metrics During Load Testing

See the canonical SLO catalogue and environment-specific thresholds in
[`docs/SLO_VERIFICATION.md`](SLO_VERIFICATION.md:45) and the JSON configs under
`tests/load/configs/` and `tests/load/config/thresholds.json`. For quick reference:

- **Baseline smoke (staging):** ≥20 concurrent games, ≥60 concurrent players.
- **Target-scale (production thresholds):** ≥100 concurrent games, ≥300 concurrent players.
- **AI-heavy (staging thresholds + production AI SLOs):** ≈75 concurrent games, ≈300 concurrent seats
  with 3 AI seats per game.

---

## Monitoring

### Accessing Dashboards

| Dashboard        | URL                   | Credentials                 |
| ---------------- | --------------------- | --------------------------- |
| **Grafana**      | http://localhost:3002 | admin / `$GRAFANA_PASSWORD` |
| **Prometheus**   | http://localhost:9090 | None                        |
| **Alertmanager** | http://localhost:9093 | None                        |

### Pre-configured Dashboards

The staging environment includes three Grafana dashboards:

1. **Game Performance** - Moves, AI latency, game outcomes
2. **Rules Correctness** - Parity metrics, invariant violations
3. **System Health** - HTTP, WebSocket, infrastructure metrics

### Key Metrics to Watch

```promql
# HTTP latency p95
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))

# Active games
ringrift_games_active

# WebSocket connections
ringrift_websocket_connections

# AI fallback rate
sum(rate(ringrift_ai_fallback_total[5m])) / sum(rate(ringrift_ai_requests_total[5m]))
```

### Staging-Specific Alerts

Alerts are configured with staging-appropriate thresholds:

| Alert                 | Threshold            | Severity |
| --------------------- | -------------------- | -------- |
| StagingHighLatency    | p95 > 500ms for 2m   | warning  |
| StagingHighErrorRate  | >1% errors for 1m    | critical |
| StagingAIFallbackHigh | >30% fallback for 5m | warning  |

---

## Troubleshooting

### Application Won't Start

```bash
# Check container logs
docker compose -f docker-compose.staging.yml logs app

# Common issues:
# - Placeholder secrets not replaced
# - Database not ready (wait longer)
# - Port already in use
```

### Database Connection Failures

```bash
# Check PostgreSQL status
docker compose -f docker-compose.staging.yml exec postgres pg_isready

# Check connection from app container
docker compose -f docker-compose.staging.yml exec app node -e "
const { PrismaClient } = require('@prisma/client');
const p = new PrismaClient();
p.\$queryRaw\`SELECT 1\`.then(() => console.log('OK')).catch(console.error);
"
```

### AI Service Not Responding

```bash
# Check AI service logs
docker compose -f docker-compose.staging.yml logs ai-service

# Test health endpoint directly
curl -v http://localhost:8001/health

# AI fallback mode is enabled automatically when service is unhealthy
```

### High Latency During Load Tests

1. Check PostgreSQL connection pool:

   ```bash
   docker compose -f docker-compose.staging.yml exec postgres \
     psql -U ringrift -c "SELECT count(*) FROM pg_stat_activity;"
   ```

2. Check Redis memory:

   ```bash
   docker compose -f docker-compose.staging.yml exec redis redis-cli info memory
   ```

3. Check app container resources:
   ```bash
   docker stats --no-stream
   ```

### Resetting the Environment

```bash
# Full reset (destroys all data)
./scripts/teardown-staging.sh --all --force
./scripts/deploy-staging.sh --clean
```

---

## CI/CD Integration

### Running Staging Tests in CI

```yaml
# Example GitHub Actions snippet
staging-test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Set up Docker
      uses: docker/setup-buildx-action@v3

    - name: Configure staging secrets
      run: |
        sed -i 's/<STAGING_DB_PASSWORD>/testpassword/' .env.staging
        sed -i 's/<STAGING_JWT_SECRET_REPLACE_ME>/testsecret12345678901234567890/' .env.staging
        sed -i 's/<STAGING_JWT_REFRESH_SECRET_REPLACE_ME>/testrefresh12345678901234567890/' .env.staging
        sed -i 's/<STAGING_GRAFANA_PASSWORD>/admin/' .env.staging
        sed -i 's/<STAGING_REDIS_PASSWORD>//' .env.staging

    - name: Deploy staging
      run: ./scripts/deploy-staging.sh --skip-health

    - name: Run load tests
      run: npm run load:baseline -- -e BASE_URL=http://localhost:3000

    - name: Teardown
      if: always()
      run: ./scripts/teardown-staging.sh --all --force
```

---

## Comparison with Production

| Aspect         | Staging        | Production            |
| -------------- | -------------- | --------------------- |
| **Replicas**   | 1 app instance | 1-N (with LB)         |
| **SSL/TLS**    | Optional       | Required              |
| **Secrets**    | Config file    | Vault/Secrets Manager |
| **Database**   | Docker volume  | Managed PostgreSQL    |
| **Redis**      | Docker volume  | Managed Redis         |
| **Monitoring** | Same stack     | Same stack + alerting |
| **Network**    | Docker bridge  | VPC/Private network   |

### Configuration Differences

| Setting                 | Staging    | Production        |
| ----------------------- | ---------- | ----------------- |
| `NODE_ENV`              | production | production        |
| `LOG_FORMAT`            | json       | json              |
| `LOG_LEVEL`             | info       | info              |
| `RINGRIFT_APP_TOPOLOGY` | single     | single (v1.0)     |
| Rate limits             | Elevated   | Production values |
| DB connections          | 200        | 200               |

---

## Version History

| Version | Date       | Changes                                   |
| ------- | ---------- | ----------------------------------------- |
| 1.0     | 2025-12-07 | Initial creation with production topology |

---

## Related Documentation

- [`docs/planning/DEPLOYMENT_REQUIREMENTS.md`](planning/DEPLOYMENT_REQUIREMENTS.md) - Full deployment requirements
- [`docs/runbooks/DEPLOYMENT_INITIAL.md`](runbooks/DEPLOYMENT_INITIAL.md) - Initial deployment runbook
- [`docs/testing/LOAD_TEST_BASELINE.md`](testing/LOAD_TEST_BASELINE.md) - Load testing procedures
- [`docs/operations/ALERTING_THRESHOLDS.md`](operations/ALERTING_THRESHOLDS.md) - Alert configuration
- [`PROJECT_GOALS.md`](../PROJECT_GOALS.md) - Scale targets and SLOs
