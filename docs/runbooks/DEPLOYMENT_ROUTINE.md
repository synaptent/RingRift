# Routine Deployment Runbook

## Overview

Standard procedure for deploying a new version of RingRift to an existing environment. Use this for regular releases after the initial deployment is complete.

**Estimated Time**: 15-30 minutes  
**Required Access**: SSH, CI/CD system, deployment notifications channel

---

## Prerequisites

Before starting deployment:

- [ ] Version tag created in Git (e.g., `v1.2.3`)
- [ ] CI pipeline passed (all tests green)
- [ ] Database migrations tested in staging
- [ ] Release notes reviewed for breaking changes
- [ ] Rollback plan reviewed (see [DEPLOYMENT_ROLLBACK.md](DEPLOYMENT_ROLLBACK.md))
- [ ] Deployment window confirmed (low-traffic period preferred)

---

## Pre-Deployment Checklist

### 1. Notify Team

```bash
# Post in #deployments channel
# Template:
🚀 Deployment starting: RingRift v1.2.3
Environment: [staging/production]
Deployer: @your-name
Expected duration: ~20 minutes
Release notes: <link>
```

### 2. Check Current System Health

```bash
# Check application health
curl -s https://api.ringrift.com/health | jq
# Verify: status is "healthy"

# Check readiness (all dependencies)
curl -s https://api.ringrift.com/ready | jq
# Verify: all checks show "healthy"

# Check current version
curl -s https://api.ringrift.com/health | jq '.version'
# Note: Current version for rollback reference
```

### 3. Verify Staging Deployment

```bash
# Confirm staging is on the target version
curl -s https://staging.ringrift.com/health | jq '.version'
# Expected: "1.2.3" (target version)

# Run health check verification against staging
curl -s https://staging.ringrift.com/health | jq
curl -s https://staging.ringrift.com/ready | jq
```

### 4. Review Release Notes

Check for:

- [ ] Database migrations required
- [ ] Environment variable changes
- [ ] Breaking API changes
- [ ] Configuration changes

---

## Deployment Steps

### Step 1: Connect to Deployment Environment

```bash
# SSH to deployment server
ssh deploy@ringrift-prod

# Navigate to application directory
cd /opt/ringrift

# Verify current state
docker compose ps
git log --oneline -1
```

### Step 2: Pull Latest Code/Image

**Option A: From Git (development/staging)**

```bash
# Fetch latest changes
git fetch origin

# Checkout target version
git checkout v1.2.3

# Verify checkout
git log --oneline -1
# Expected: Shows v1.2.3 commit
```

**Option B: From Registry (production)**

```bash
# Pull specific version
docker pull ringrift/app:v1.2.3
docker pull ringrift/ai-service:v1.2.3

# Verify images
docker images | grep ringrift
```

### Step 3: Check for Database Migrations

```bash
# Check migration status
docker compose run --rm app npx prisma migrate status

# If migrations pending, proceed to Step 4
# If no migrations, skip to Step 5
```

### Step 4: Apply Database Migrations (if needed)

⚠️ **Important**: For migrations with breaking changes, see [DATABASE_MIGRATION.md](DATABASE_MIGRATION.md).

```bash
# Backup database before migration (production)
docker compose exec postgres pg_dump -U ringrift -d ringrift > backups/pre_v1.2.3_$(date +%Y%m%d_%H%M%S).sql

# Apply migrations
docker compose run --rm app npx prisma migrate deploy

# Verify migration success
docker compose run --rm app npx prisma migrate status
# Expected: All migrations applied
```

### Step 5: Deploy New Version

**Rolling Update (recommended for production):**

```bash
# Update app service with zero downtime
docker compose up -d --no-deps --build app

# Wait for new container to be healthy (30 seconds)
sleep 30

# Verify new container is running
docker compose ps app
```

**Full Stack Update:**

```bash
# For staging or when all services need update
docker compose -f docker-compose.yml -f docker-compose.staging.yml up -d --build
```

### Step 6: Verify Deployment

```bash
# Check new version
curl -s http://localhost:3000/health | jq '.version'
# Expected: "1.2.3"

# Check health status
curl -s http://localhost:3000/health | jq '.status'
# Expected: "healthy"

# Check all dependencies
curl -s http://localhost:3000/ready | jq
# Expected: All checks "healthy"
```

### Step 7: Run Post-Deployment Tests

```bash
# Quick smoke test - health check
curl -s http://localhost:3000/health | jq '.status' | grep -q "healthy" && echo "✅ Health OK" || echo "❌ Health FAILED"

# Test API responsiveness
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health
# Expected: 200

# Test WebSocket connectivity
APP_PORT=${APP_PORT:-3000}
timeout 5 wscat -c ws://localhost:${APP_PORT} -x "ping" 2>/dev/null && echo "✅ WebSocket OK" || echo "⚠️ WebSocket check failed"

# Test game creation (requires valid auth token)
# curl -X POST http://localhost:3000/api/games \
#   -H "Authorization: Bearer $TOKEN" \
#   -H "Content-Type: application/json" \
#   -d '{"boardType":"standard","playerCount":2}'
```

---

## Post-Deployment Verification

### Automated Verification

```bash
#!/bin/bash
# Quick verification script

echo "🔍 Running post-deployment verification..."

# Health check
HEALTH=$(curl -s http://localhost:3000/health | jq -r '.status')
if [ "$HEALTH" = "healthy" ]; then
    echo "✅ Health check: PASSED"
else
    echo "❌ Health check: FAILED"
    exit 1
fi

# Ready check
READY=$(curl -s http://localhost:3000/ready | jq -r '.status')
if [ "$READY" != "unhealthy" ]; then
    echo "✅ Readiness check: PASSED"
else
    echo "❌ Readiness check: FAILED"
    exit 1
fi

# Version check
VERSION=$(curl -s http://localhost:3000/health | jq -r '.version')
echo "📦 Deployed version: $VERSION"

# Container health
CONTAINER_STATUS=$(docker compose ps app --format json | jq -r '.Health')
if [ "$CONTAINER_STATUS" = "healthy" ]; then
    echo "✅ Container health: PASSED"
else
    echo "⚠️ Container health: $CONTAINER_STATUS"
fi

echo "✅ Deployment verification complete!"
```

### Manual Verification Checklist

- [ ] `/health` returns 200 with correct version
- [ ] `/ready` returns 200 with healthy dependencies
- [ ] WebSocket connections work
- [ ] Can login with existing account
- [ ] Can create new game
- [ ] Can join existing game
- [ ] Game moves execute correctly
- [ ] Metrics endpoint accessible (`/metrics`, honoring ENABLE_METRICS/METRICS_PORT)

### Notify Team of Completion

```bash
# Post in #deployments channel
# Template:
✅ Deployment complete: RingRift v1.2.3
Environment: [staging/production]
Duration: X minutes
Status: Healthy
Verification: All checks passed
```

---

## Rollback Procedure

If issues are detected during or after deployment, follow [DEPLOYMENT_ROLLBACK.md](DEPLOYMENT_ROLLBACK.md).

**Quick rollback command:**

```bash
# If deployment just completed and issues found:
docker compose up -d --no-deps app:v1.2.2  # Previous version
```

---

## Troubleshooting

### Container won't start

```bash
# Check container logs
docker compose logs app --tail=100

# Common issues:
# 1. Secret validation failed - check .env for placeholder values
# 2. Port already in use - check for orphaned containers
# 3. Database connectivity - verify DATABASE_URL
```

### Health check failing

```bash
# Get detailed health info
curl -s http://localhost:3000/ready | jq

# Check specific dependency:

# Database
docker compose exec postgres pg_isready -U ringrift -d ringrift

# Redis
docker compose exec redis redis-cli ping

# AI Service
curl -s http://localhost:8001/health | jq
```

### WebSocket not connecting

```bash
# Check WebSocket server logs
docker compose logs app | grep -i websocket

# Verify port is exposed
APP_PORT=${APP_PORT:-3000}
netstat -tlnp | grep ${APP_PORT}

# Test with wscat
wscat -c ws://localhost:${APP_PORT}
```

### Migration failed

```bash
# Check migration status
docker compose run --rm app npx prisma migrate status

# View failed migration
docker compose run --rm app npx prisma migrate resolve --rolled-back <migration_name>

# Restore from backup if needed
docker compose exec -T postgres psql -U ringrift -d ringrift < backups/pre_v1.2.3_TIMESTAMP.sql
```

### Response time degradation

```bash
# Check container resource usage
docker stats app

# Check for memory pressure
docker compose exec app node -e "console.log(process.memoryUsage())"

# Check connection pools
curl -s http://localhost:3000/ready | jq '.checks.database.latency'
```

---

## Blue-Green Deployment (Optional)

For zero-downtime deployments with instant rollback capability:

### Setup Blue-Green

```bash
# Start new version on different ports (green)
docker compose -f docker-compose.green.yml up -d

# Verify green is healthy
curl -s http://localhost:3010/health | jq

# Switch traffic at load balancer
# (Update nginx upstream or cloud LB target group)

# After verification, stop old version (blue)
docker compose -f docker-compose.blue.yml down
```

### Rollback Blue-Green

```bash
# Revert load balancer to blue
# Start blue if stopped
docker compose -f docker-compose.blue.yml up -d
```

---

## Deployment Schedule Best Practices

| Environment | Recommended Time        | Avoid                       |
| ----------- | ----------------------- | --------------------------- |
| Staging     | Any time                | -                           |
| Production  | Tue-Thu, 10am-2pm local | Fridays, weekends, holidays |

### Deployment Freeze Periods

- Major game tournaments
- Peak usage hours (evenings in target timezone)
- Company events
- Before/after holidays

---

## Metrics to Monitor Post-Deployment

Watch these metrics for 15-30 minutes after deployment:

| Metric              | Normal Range  | Alert Threshold |
| ------------------- | ------------- | --------------- |
| Response time (p95) | <200ms        | >500ms          |
| Error rate          | <0.1%         | >1%             |
| Active connections  | Baseline ±20% | >50% change     |
| Memory usage        | <400MB        | >450MB          |
| CPU usage           | <30%          | >60%            |

```bash
# Quick metrics check
curl -s http://localhost:9090/api/v1/query?query=http_request_duration_seconds_bucket | jq
```

---

## Related Documentation

- [DEPLOYMENT_INITIAL.md](DEPLOYMENT_INITIAL.md) - First-time deployment
- [DEPLOYMENT_ROLLBACK.md](DEPLOYMENT_ROLLBACK.md) - Rollback procedures
- [DATABASE_MIGRATION.md](DATABASE_MIGRATION.md) - Migration procedures
- [DEPLOYMENT_REQUIREMENTS.md](../planning/DEPLOYMENT_REQUIREMENTS.md) - Environment requirements

---

**Last Updated**: 2024-01  
**Owner**: Platform Team  
**Review Cycle**: Quarterly
