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

## Load Testing

The staging environment supports the k6 load testing scenarios defined in `tests/load/`:

### Baseline Test

Tests sustained load at target scale:

```bash
# Run from project root
npm run load:baseline -- -e BASE_URL=http://localhost:3000

# Or directly with k6
k6 run tests/load/scenarios/baseline.js \
  -e BASE_URL=http://localhost:3000 \
  -e VUS=100 \
  -e DURATION=5m
```

### Stress Test

Tests system behavior beyond target scale:

```bash
npm run load:stress -- -e BASE_URL=http://localhost:3000
```

### WebSocket Load Test

Tests WebSocket connection scaling:

```bash
npm run load:websocket -- -e WS_URL=ws://localhost:3001
```

### Target Metrics During Load Testing

| Metric                | Baseline Target | Stress Target |
| --------------------- | --------------- | ------------- |
| Concurrent VUs        | 100             | 300           |
| Requests/sec          | 500             | 1500          |
| p95 latency           | <500ms          | <1s           |
| Error rate            | <1%             | <5%           |
| WebSocket connections | 300             | 500           |

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
