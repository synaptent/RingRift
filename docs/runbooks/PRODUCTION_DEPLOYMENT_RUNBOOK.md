# Production Deployment Runbook

> **Doc Status (2025-12-03): Active Runbook**
> **Role:** Step-by-step guide for deploying RingRift to staging or production environments
>
> **SSoT alignment:** This runbook is derived from:
>
> - `docker-compose.yml` - Base Docker Compose configuration
> - `docker-compose.staging.yml` - Staging overlay
> - `.env.staging` - Staging environment template
> - `docs/runbooks/DEPLOYMENT_INITIAL.md` - Initial deployment reference

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Pre-Deployment Checklist](#2-pre-deployment-checklist)
3. [Environment Configuration](#3-environment-configuration)
4. [Build Process](#4-build-process)
5. [Deployment Steps](#5-deployment-steps)
6. [Smoke Tests](#6-smoke-tests)
7. [Load Test Verification](#7-load-test-verification)
8. [Post-Deployment](#8-post-deployment)
9. [Rollback Procedure](#9-rollback-procedure)
10. [Reference Baselines](#10-reference-baselines)

---

## 1. Prerequisites

### Infrastructure Requirements

| Component      | Minimum | Recommended | Notes                     |
| -------------- | ------- | ----------- | ------------------------- |
| Docker Engine  | v20+    | v24+        | Required                  |
| Docker Compose | v2+     | v2.20+      | Required                  |
| Memory         | 4GB     | 8GB         | For all services          |
| Storage        | 20GB    | 50GB        | For database and logs     |
| CPU            | 2 cores | 4 cores     | For concurrent operations |

### Required Ports

| Port | Service                | Protocol |
| ---- | ---------------------- | -------- |
| 80   | nginx (HTTP)           | TCP      |
| 443  | nginx (HTTPS)          | TCP      |
| 3000 | Backend API (internal) | TCP      |
| 3001 | WebSocket (internal)   | TCP      |
| 5432 | PostgreSQL             | TCP      |
| 6379 | Redis                  | TCP      |
| 8001 | AI Service             | TCP      |
| 9090 | Prometheus             | TCP      |
| 3002 | Grafana                | TCP      |

### Access Requirements

- [ ] SSH access to target server
- [ ] Docker registry credentials (if private)
- [ ] DNS management access
- [ ] SSL certificates (production only)

---

## 2. Pre-Deployment Checklist

```bash
# Verify Docker installation
docker --version  # Expected: 20.x or higher
docker compose version  # Expected: v2.x

# Verify disk space (minimum 20GB free)
df -h /var/lib/docker

# Verify network connectivity
curl -I https://registry.docker.io/v2/

# Create deployment directory
sudo mkdir -p /opt/ringrift
sudo chown $USER:$USER /opt/ringrift
```

---

## 3. Environment Configuration

### Generate Secrets

**CRITICAL: Never use placeholder values in production.**

```bash
cd /opt/ringrift

# Generate JWT secret (minimum 48 bytes base64)
JWT_SECRET=$(openssl rand -base64 48)
echo "JWT_SECRET=$JWT_SECRET"

# Generate JWT refresh secret (different from above)
JWT_REFRESH_SECRET=$(openssl rand -base64 48)
echo "JWT_REFRESH_SECRET=$JWT_REFRESH_SECRET"

# Generate database password (32 bytes)
DB_PASSWORD=$(openssl rand -base64 32)
echo "DB_PASSWORD=$DB_PASSWORD"
```

### Create Environment File

```bash
# Copy staging template
cp .env.staging .env

# Edit with actual values
nano .env
```

**Critical Variables:**

| Variable             | Description           | Example                                                 |
| -------------------- | --------------------- | ------------------------------------------------------- |
| `NODE_ENV`           | Environment mode      | `production`                                            |
| `DATABASE_URL`       | Postgres connection   | `postgresql://ringrift:PASSWORD@postgres:5432/ringrift` |
| `JWT_SECRET`         | Access token signing  | Generated above                                         |
| `JWT_REFRESH_SECRET` | Refresh token signing | Generated above                                         |
| `REDIS_URL`          | Redis connection      | `redis://redis:6379`                                    |
| `AI_SERVICE_URL`     | AI service endpoint   | `http://ai-service:8001`                                |

---

## 4. Build Process

**IMPORTANT:** The staging overlay requires a production build.

```bash
# Clone repository
git clone https://github.com/your-org/ringrift.git /opt/ringrift
cd /opt/ringrift

# Install dependencies
npm ci

# Build production assets
npm run build

# Verify build output exists
ls -la dist/server/index.js
ls -la dist/client/

# Build Docker images
docker compose build --no-cache
```

---

## 5. Deployment Steps

### Option A: Docker Compose (Staging/Small Production)

```bash
cd /opt/ringrift

# Deploy with staging overlay
docker compose -f docker-compose.yml -f docker-compose.staging.yml up -d

# Wait for services to be healthy
docker compose ps

# Expected: All services showing "healthy" or "running"
```

### Option B: Fresh Database

```bash
# First deployment only - run migrations
docker compose exec app npx prisma migrate deploy

# Verify migration status
docker compose exec app npx prisma migrate status
```

### Option C: Update Existing Deployment

```bash
# Pull latest images/code
git pull origin main

# Rebuild and restart
docker compose build app ai-service
docker compose up -d --force-recreate app ai-service

# Run any pending migrations
docker compose exec app npx prisma migrate deploy
```

---

## 6. Smoke Tests

Run these tests immediately after deployment:

```bash
# Set base URL
BASE_URL=http://localhost  # or your domain

# Test 1: Health endpoint
curl -s $BASE_URL/health
# Expected: {"status":"healthy",...}

# Test 2: Readiness with dependency checks
curl -s $BASE_URL/ready
# Expected: {"status":"healthy","checks":{"database":{"status":"healthy"},...}}

# Test 3: AI service health
curl -s $BASE_URL:8001/health
# Expected: {"status":"healthy"}

# Test 4: Authentication flow
curl -s -X POST $BASE_URL/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpassword"}'
# Expected: {"success":true,...} or auth error (expected if user doesn't exist)

# Test 5: Prometheus
curl -s http://localhost:9090/-/healthy
# Expected: "Prometheus Server is Healthy."

# Test 6: Grafana
curl -s http://localhost:3002/api/health
# Expected: {"database":"ok",...}
```

### Automated Smoke Test Script

```bash
#!/bin/bash
# save as: scripts/smoke-test.sh

BASE_URL=${1:-http://localhost}
PASS=0
FAIL=0

check() {
  local name=$1
  local expected=$2
  local actual=$3

  if echo "$actual" | grep -q "$expected"; then
    echo "✅ $name"
    ((PASS++))
  else
    echo "❌ $name - Expected: $expected, Got: $actual"
    ((FAIL++))
  fi
}

echo "Running smoke tests against $BASE_URL..."

# Health check
HEALTH=$(curl -s $BASE_URL/health)
check "Health endpoint" "healthy" "$HEALTH"

# Ready check
READY=$(curl -s $BASE_URL/ready)
check "Ready endpoint" "healthy" "$READY"
check "Database healthy" "database.*healthy" "$READY"
check "Redis healthy" "redis.*healthy" "$READY"

# AI service
AI=$(curl -s $BASE_URL:8001/health 2>/dev/null || echo "error")
check "AI service" "healthy" "$AI"

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit $FAIL
```

---

## 7. Load Test Verification

After smoke tests pass, run abbreviated load tests:

```bash
cd /opt/ringrift/tests/load

# Install k6 if not present
# brew install k6  # macOS
# sudo apt install k6  # Ubuntu

# Create test user first
curl -X POST $BASE_URL/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"loadtest@test.local","password":"LoadTest123","username":"loadtest"}'

# Run abbreviated load test (1 minute, 10 VUs)
k6 run --duration 1m --vus 10 \
  --env BASE_URL=$BASE_URL \
  --env LOADTEST_EMAIL=loadtest@test.local \
  --env LOADTEST_PASSWORD=LoadTest123 \
  scenarios/game-creation.js
```

### Expected Results

| Metric            | Target | Baseline (2025-12-03) |
| ----------------- | ------ | --------------------- |
| Success rate      | >99%   | 100%                  |
| Game creation p95 | <100ms | 17ms                  |
| HTTP p95          | <50ms  | 16.94ms               |
| Error rate        | <1%    | 0%                    |

---

## 8. Post-Deployment

### Configure Monitoring

```bash
# Verify Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[].health'
# Expected: all "up"

# Access Grafana dashboards
# URL: http://localhost:3002
# Default login: admin / (check GRAFANA_PASSWORD in .env)
```

### Enable Alerts

```bash
# Verify Alertmanager
curl -s http://localhost:9093/-/healthy
# Expected: "OK"

# Check for firing alerts
curl -s http://localhost:9093/api/v1/alerts | jq '.data'
```

### Database Backup

```bash
# Schedule backup cron job
echo "0 */6 * * * docker compose exec -T postgres pg_dump -U ringrift -d ringrift > /backups/ringrift_$(date +\%Y\%m\%d_\%H\%M\%S).sql" | crontab -
```

---

## 9. Rollback Procedure

### Quick Rollback

```bash
# Stop current deployment
docker compose down

# Restore previous version
git checkout HEAD~1

# Rebuild and deploy
docker compose build
docker compose up -d
```

### Database Rollback

**CAUTION: Test in staging first.**

```bash
# List available backups
ls -la /backups/

# Restore from backup (creates new DB, doesn't overwrite)
docker compose exec -T postgres createdb -U ringrift ringrift_restore
docker compose exec -T postgres psql -U ringrift -d ringrift_restore < /backups/BACKUP_FILE.sql

# If verified, switch DATABASE_URL to use ringrift_restore
# Or drop/recreate ringrift from backup
```

---

## 10. Reference Baselines

### Validated Performance (2025-12-03)

| Scenario         | VUs | Duration | Success Rate | p95 Latency |
| ---------------- | --- | -------- | ------------ | ----------- |
| Game Creation    | 50  | 4min     | 100%         | 13ms        |
| Concurrent Games | 40  | 10min    | 100%         | 12ms        |
| Player Moves     | 40  | 10min    | 100%         | 15ms        |
| WebSocket Stress | 500 | 15min    | 100%         | 2ms         |

### Capacity Model

| Resource              | Single Instance Capacity |
| --------------------- | ------------------------ |
| Concurrent games      | 100+                     |
| Active players        | 200+                     |
| WebSocket connections | 500+                     |
| Game creations/min    | 200+                     |

### Health Check Endpoints

| Endpoint           | Purpose             | Expected Response                     |
| ------------------ | ------------------- | ------------------------------------- |
| `/health`          | Basic liveness      | `{"status":"healthy"}`                |
| `/ready`           | Readiness with deps | `{"status":"healthy","checks":{...}}` |
| `:8001/health`     | AI service          | `{"status":"healthy"}`                |
| `:9090/-/healthy`  | Prometheus          | `Prometheus Server is Healthy.`       |
| `:3002/api/health` | Grafana             | `{"database":"ok"}`                   |

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker compose logs SERVICE_NAME --tail 50

# Common issues:
# - Missing .env file
# - Port already in use
# - Database not ready (wait longer)
```

### Database Connection Failed

```bash
# Verify postgres is healthy
docker compose ps postgres

# Test connection
docker compose exec postgres pg_isready -U ringrift

# Check DATABASE_URL format
# postgresql://USER:PASSWORD@HOST:PORT/DATABASE
```

### Redis Connection Failed

```bash
# Verify redis is healthy
docker compose exec redis redis-cli ping
# Expected: PONG

# Check REDIS_URL format
# redis://HOST:PORT
```

### AI Service Unhealthy

```bash
# Check AI service logs
docker compose logs ai-service --tail 50

# Verify health endpoint directly
curl http://localhost:8001/health

# Common issues:
# - Python dependencies missing
# - Port conflict
# - Database connection (if required)
```

---

**Document Maintainer:** Claude Code
**Last Updated:** December 3, 2025
**Based on validated deployment:** Local Docker Compose environment
