# Scaling Procedures Runbook

## Overview

Procedures for scaling RingRift services to handle changes in load. This covers both vertical scaling (increasing resources) and horizontal scaling (adding instances), with important considerations for the current single-instance architecture.

**Important**: RingRift currently operates in single-instance mode (`RINGRIFT_APP_TOPOLOGY=single`). Multi-instance scaling requires careful consideration of session affinity and state management.

---

## Current Architecture Constraints

### Single-Instance Mode (Default)

```
┌─────────────────────────────────────────────────┐
│                  Load Balancer                   │
└─────────────────────┬───────────────────────────┘
                      │
              ┌───────▼───────┐
              │   App (1)     │ ◄── All game state in memory
              │   Port 3000   │
              └───────┬───────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
┌───▼───┐      ┌─────▼─────┐     ┌─────▼─────┐
│ Redis │      │ PostgreSQL │     │ AI Service│
└───────┘      └───────────┘     └───────────┘
```

**Key Limitation**: Game session state is stored in the single app instance's memory. Scaling to multiple instances requires infrastructure-enforced sticky sessions.

### Multi-Instance Mode (Advanced)

```
┌───────────────────────────────────────────────────┐
│          Load Balancer (Sticky Sessions)          │
└─────────┬─────────────────┬───────────────────────┘
          │                 │
   ┌──────▼──────┐   ┌──────▼──────┐
   │   App #1    │   │   App #2    │  ◄── Each has own game sessions
   │  Port 3000  │   │  Port 3000  │
   └──────┬──────┘   └──────┬──────┘
          │                 │
          └────────┬────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐   ┌─────▼─────┐  ┌─────▼─────┐
│ Redis │   │ PostgreSQL │  │ AI Service│
└───────┘   └───────────┘  └───────────┘
```

**Requirements for Multi-Instance**:

- `RINGRIFT_APP_TOPOLOGY=multi-sticky` environment variable
- Load balancer with sticky session support
- Session affinity based on user/game ID

---

## Section 1: Vertical Scaling (Increase Resources)

### When to Use

- High CPU usage (>70% sustained)
- High memory usage (>80%)
- Response time degradation without traffic increase
- Database connection pool exhaustion

### 1.1 Scale App Container Resources

**Check Current Resource Usage:**

```bash
# Real-time container stats
docker stats app

# Alternative: one-time snapshot
docker stats --no-stream app
```

**Increase Memory Limit:**

```bash
# Edit docker-compose.yml
# Change:
#   deploy:
#     resources:
#       limits:
#         memory: 512M
# To:
#   deploy:
#     resources:
#       limits:
#         memory: 1024M

# Apply changes
docker compose up -d app
```

**Or apply directly with Docker run:**

```bash
# Stop current container
docker compose stop app

# Run with increased resources
docker run -d \
  --name ringrift-app \
  --memory=1024m \
  --cpus=2 \
  --env-file .env \
  -p 3000:3000 \
  ringrift/app:latest
```

**Verify New Limits:**

```bash
docker inspect app | jq '.[0].HostConfig.Memory'
# Expected: 1073741824 (1GB in bytes)
```

### 1.2 Scale PostgreSQL Resources

**Check Current Usage:**

```bash
# Database connection count
docker compose exec postgres psql -U ringrift -d ringrift -c "SELECT count(*) FROM pg_stat_activity;"

# Database size
docker compose exec postgres psql -U ringrift -d ringrift -c "SELECT pg_size_pretty(pg_database_size('ringrift'));"
```

**Increase Connection Pool:**

```bash
# Edit .env
DATABASE_POOL_MIN=5
DATABASE_POOL_MAX=20

# Restart app to apply
docker compose restart app
```

**Increase PostgreSQL Resources:**

```bash
# Edit docker-compose.yml postgres service
# deploy:
#   resources:
#     limits:
#       memory: 512M  # Increase from 256M

docker compose up -d postgres
```

### 1.3 Scale Redis Resources

**Check Current Usage:**

```bash
# Redis memory usage
docker compose exec redis redis-cli INFO memory | grep used_memory_human
```

**Increase Redis Memory:**

```bash
# Edit docker-compose.yml
# deploy:
#   resources:
#     limits:
#       memory: 256M  # Increase from 128M

docker compose up -d redis
```

### 1.4 Scale AI Service Resources

**Check Current Usage:**

```bash
docker stats ai-service --no-stream
```

**Increase AI Service Concurrency:**

```bash
# Edit .env
AI_MAX_CONCURRENT_REQUESTS=32  # Increase from 16

# Increase memory if needed (edit docker-compose.yml)
# deploy:
#   resources:
#     limits:
#       memory: 1024M  # Increase from 512M

docker compose up -d ai-service
```

---

## Section 2: Horizontal Scaling (Add Instances)

⚠️ **Warning**: Before scaling horizontally, ensure you understand the session affinity requirements.

### 2.1 Prerequisites for Multi-Instance

1. **Configure topology mode:**

   ```bash
   # Edit .env
   RINGRIFT_APP_TOPOLOGY=multi-sticky
   ```

2. **Configure load balancer with sticky sessions:**

   Example nginx configuration:

   ```nginx
   upstream ringrift_app {
       ip_hash;  # Sticky sessions based on client IP
       server app1:3000;
       server app2:3000;
   }

   # Or use cookie-based persistence:
   upstream ringrift_app {
       server app1:3000;
       server app2:3000;
       sticky cookie srv_id expires=1h domain=.ringrift.com path=/;
   }
   ```

3. **Ensure Redis is configured for shared state.**

### 2.2 Scale App Service (Docker Compose)

**Not Recommended for Production** - Use orchestration platform instead.

```bash
# Scale app to 2 instances (requires sticky sessions!)
docker compose up -d --scale app=2

# ⚠️ This creates port conflicts without load balancer
# Only use with proper load balancer configuration
```

### 2.3 Scale AI Service

AI service is stateless and can be scaled more easily:

```bash
# Scale AI service to multiple instances
docker compose up -d --scale ai-service=3

# Update app to use multiple AI endpoints (requires custom config)
```

### 2.4 GPU Proxy Configuration (Staging)

For staging environments without GPU (e.g., AWS r5.xlarge), proxy AI requests to a GPU cluster:

**Architecture:**

```
┌──────────────────────┐        ┌───────────────────────────┐
│  Staging Server      │        │  Runpod GPU Cluster       │
│  (CPU only)          │        │  (GH200 with 96GB GPU)    │
│                      │        │                           │
│  ┌────────────────┐  │        │  ┌─────────────────────┐  │
│  │  Game Server   │  │        │  │  AI Inference       │  │
│  │  Port 3000     ├──┼──────►─┼──►  Port 8765          │  │
│  └────────────────┘  │        │  └─────────────────────┘  │
│                      │ Tailscale│                         │
└──────────────────────┘   VPN  └───────────────────────────┘
```

**Configuration Steps:**

1. Start AI inference server on GPU cluster node:

```bash
# SSH to GPU node (via Tailscale)
ssh ubuntu@100.88.35.19

# Start AI service
cd ~/ringrift/ai-service
source venv/bin/activate
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8765 > logs/inference_server.log 2>&1 &
```

2. Update staging .env to proxy to cluster:

```bash
# On staging server
AI_SERVICE_URL=http://100.88.35.19:8765
AI_SERVICE_PORT=8765
```

3. Restart game server to pick up new config:

```bash
pm2 restart ringrift-server --update-env
```

**Verification:**

```bash
# From staging, verify cluster connectivity
curl -s http://100.88.35.19:8765/health
# Should return {"status":"healthy",...}
```

**Active Cluster Nodes (Dec 2025):**

- GH200 nodes: 100.88.35.19 (active), 100.123.183.70, 100.88.176.74
- Each has 96GB GPU memory, ideal for neural network inference

### 2.5 Add Read Replicas for PostgreSQL

For read-heavy workloads:

```bash
# Add to docker-compose.yml:
# postgres-replica:
#   image: postgres:15-alpine
#   environment:
#     - POSTGRES_DB=ringrift
#     - POSTGRES_USER=ringrift
#     - POSTGRES_PASSWORD=${DB_PASSWORD}
#   command: >
#     postgres -c hot_standby=on
#   volumes:
#     - postgres_replica_data:/var/lib/postgresql/data
#   depends_on:
#     - postgres

# Configure replication on primary PostgreSQL
```

---

## Section 3: Kubernetes/Container Orchestration Scaling

For production deployments using Kubernetes:

### 3.1 Horizontal Pod Autoscaler

```yaml
# ringrift-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ringrift-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ringrift-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### 3.2 Manual Scaling in Kubernetes

```bash
# Scale deployment manually
kubectl scale deployment ringrift-app --replicas=5

# Check scaling status
kubectl get deployment ringrift-app
kubectl get pods -l app=ringrift-app
```

### 3.3 Session Affinity in Kubernetes

```yaml
# Ensure session affinity in Service
apiVersion: v1
kind: Service
metadata:
  name: ringrift-app
spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600 # 1 hour
  selector:
    app: ringrift-app
  ports:
    - port: 3000
```

---

## Section 4: Scaling Decision Guide

### Response Time Degradation

```
Response Time High?
    │
    ├─► CPU > 70%?
    │       └─► Vertical scale: Add CPU/instances
    │
    ├─► Memory > 80%?
    │       └─► Vertical scale: Add memory
    │
    ├─► DB connections maxed?
    │       └─► Increase pool size or add replicas
    │
    └─► Check network latency between services
```

### Memory Pressure

```bash
# Check app memory
docker stats app --no-stream

# If consistent growth (memory leak suspicion):
# - Restart app (temporary fix)
# - Scale memory limits
# - Investigate leak

# If stable but high:
# - Increase memory limit
# - Profile memory usage
```

### Traffic Spike Handling

```bash
# Immediate response:
# 1. Scale AI service (most common bottleneck)
docker compose up -d --scale ai-service=2

# 2. Enable aggressive caching
# Edit .env
CACHE_TTL_SECONDS=300  # If applicable

# 3. If using rate limiting, temporarily increase limits
# (Only if verified legitimate traffic)
```

---

## Section 5: Scaling Down Procedures

### 5.1 Scale Down App Instances

⚠️ **Graceful shutdown required to avoid dropping active games.**

```bash
# Check for active games
curl -s http://localhost:3000/api/admin/stats | jq '.activeGames'

# Graceful shutdown: wait for games to complete or migrate
# Then scale down:
docker compose up -d --scale app=1
```

### 5.2 Scale Down Resources

```bash
# Reduce memory limits after load decreases
# Edit docker-compose.yml to reduce limits

# Apply changes
docker compose up -d app

# Monitor for stability
docker stats app
```

### 5.3 Scheduled Scaling

For predictable traffic patterns:

```bash
#!/bin/bash
# scale-by-time.sh

HOUR=$(date +%H)

if [ $HOUR -ge 18 ] && [ $HOUR -le 23 ]; then
    # Peak hours: scale up
    docker compose up -d --scale ai-service=3
elif [ $HOUR -ge 2 ] && [ $HOUR -le 6 ]; then
    # Off-peak: scale down
    docker compose up -d --scale ai-service=1
fi
```

---

## Section 6: Monitoring During Scaling

### Key Metrics to Watch

| Metric              | Normal    | Warning   | Critical       |
| ------------------- | --------- | --------- | -------------- |
| App CPU             | <60%      | >70%      | >85%           |
| App Memory          | <70%      | >80%      | >90%           |
| Response Time (p95) | <200ms    | >500ms    | >1000ms        |
| Error Rate          | <0.1%     | >1%       | >5%            |
| DB Connections      | <80% pool | >80% pool | Pool exhausted |
| AI Service Latency  | <100ms    | >300ms    | >1000ms        |

### Monitoring Commands

```bash
# Base URLs (adjust as needed)
APP_BASE=${APP_BASE:-http://localhost:3000}
APP_METRICS_BASE=${APP_METRICS_BASE:-$APP_BASE}

# Real-time resource monitoring
docker stats

# Application metrics (if Prometheus enabled)
curl -s $APP_METRICS_BASE/metrics | grep -E "(request_duration|active_connections)"

# Database query performance
docker compose exec postgres psql -U ringrift -d ringrift -c "
SELECT
  calls,
  total_exec_time / calls as avg_time,
  query
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
"
```

### Post-Scaling Verification

```bash
# After any scaling operation:

# 1. Health check
# Requires ENABLE_HEALTH_CHECKS=true
curl -s $APP_BASE/health | jq

# 2. Readiness check
# Requires ENABLE_HEALTH_CHECKS=true
curl -s $APP_BASE/ready | jq

# 3. Load test (optional)
# ab -n 1000 -c 50 http://localhost:3000/api

# 4. Check error rates
docker compose logs app --since 5m | grep -c ERROR
```

---

## Capacity Planning Guidelines

### Resource Estimates per Component

| Component      | Per 100 Users | Per 1000 Users |
| -------------- | ------------- | -------------- |
| App Memory     | 128MB         | 256MB          |
| App CPU        | 0.2 cores     | 1 core         |
| DB Connections | 10            | 50             |
| Redis Memory   | 16MB          | 64MB           |
| AI Service     | 1 instance    | 2+ instances   |

### Load Testing Before Production Scaling

```bash
# Install k6 for load testing
# brew install k6

# Basic load test script
# k6-test.js
# import http from 'k6/http';
# export default function() {
#   http.get('http://localhost:3000/health');
# }

# Run load test
# k6 run --vus 100 --duration 30s k6-test.js
```

---

## Related Documentation

- [DEPLOYMENT_REQUIREMENTS.md](../planning/DEPLOYMENT_REQUIREMENTS.md) - Resource limits reference
- [DEPLOYMENT_ROUTINE.md](DEPLOYMENT_ROUTINE.md) - Standard deployment procedures
- [ALERTING_THRESHOLDS.md](../operations/ALERTING_THRESHOLDS.md) - Alert configuration

---

**Last Updated**: 2024-01  
**Owner**: Platform Team  
**Review Cycle**: Quarterly
