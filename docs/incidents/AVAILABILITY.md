# Availability Incidents

This guide covers incidents related to service availability, including service outages, high error rates, and degradation states.

## Alerts Covered

| Alert                         | Severity | Threshold             | Duration |
| ----------------------------- | -------- | --------------------- | -------- |
| DatabaseDown                  | Critical | Service status = 0    | 1 min    |
| RedisDown                     | Critical | Service status = 0    | 1 min    |
| HighErrorRate                 | Critical | >5% 5xx errors        | 5 min    |
| ElevatedErrorRate             | Warning  | >1% 5xx errors        | 10 min   |
| NoHTTPTraffic                 | Warning  | 0 requests            | 10 min   |
| ServiceDegraded               | Warning  | Degradation level > 0 | 5 min    |
| ServiceMinimalMode            | Critical | Degradation level ≥ 2 | 1 min    |
| ServiceOffline                | Critical | Degradation level = 3 | 30 sec   |
| NoActiveGames                 | Info     | 0 active games        | 30 min   |
| NoWebSocketConnections        | Warning  | 0 connections         | 15 min   |
| LongRunningGames              | Info     | Median > 1 hour       | 30 min   |
| RulesParityValidationMismatch | Warning  | >5/hour               | 5 min    |
| RulesParityHashMismatch       | Warning  | >5/hour               | 5 min    |
| RulesParityGameStatusMismatch | Critical | >0/hour               | 5 min    |

---

## Alert: DatabaseDown

### Severity

**P1 Critical** - Immediate response required

### Symptoms

- `/ready` endpoint returns database failure
- All API requests returning 500 errors
- Users cannot log in, create games, or perform any action
- `ringrift_service_status{service="database"} == 0`

### Impact

- **Complete service outage** - All data operations unavailable
- Users cannot authenticate
- Game state cannot be persisted
- Leaderboards/profiles unavailable

### Initial Triage (2-5 min)

```bash
# 1. Check database container status
docker compose ps postgres

# 2. Check database logs
docker compose logs --tail 200 postgres

# 3. Test database connectivity directly
docker exec ringrift-postgres-1 pg_isready -U ringrift

# 4. Test from app container
docker exec ringrift-app-1 sh -c "nc -zv postgres 5432"
```

### Diagnosis

#### Database Container Not Running

```bash
# Check container status
docker compose ps postgres

# Check why it stopped
docker compose logs postgres | tail -100

# Check for OOM kill
docker inspect ringrift-postgres-1 | jq '.[0].State'
```

**Common causes:**

- Out of memory kill
- Disk space exhaustion
- Configuration error
- Docker daemon issues

#### Database Running But Not Accepting Connections

```bash
# Check PostgreSQL state
docker exec ringrift-postgres-1 psql -U ringrift -c "SELECT 1;"

# Check connection count
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"

# Check if in recovery mode
docker exec ringrift-postgres-1 psql -U ringrift -c "SELECT pg_is_in_recovery();"
```

**Common causes:**

- Max connections reached
- Authentication issues
- WAL corruption
- Disk full

#### Connection Pool Exhausted

```bash
# Check active connections
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# Check waiting connections
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT count(*) FROM pg_stat_activity WHERE wait_event IS NOT NULL;"
```

### Mitigation

#### Restart Database Container

```bash
# Graceful restart
docker compose restart postgres

# Wait for healthy
sleep 10
docker compose exec postgres pg_isready -U ringrift

# Then restart app to reconnect
docker compose restart app
```

#### Clear Stuck Connections

```bash
# Terminate idle connections older than 10 minutes
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
   WHERE state = 'idle' AND query_start < now() - interval '10 minutes';"
```

#### Free Disk Space

```bash
# Check disk usage
docker system df

# Clean unused Docker resources
docker system prune -f

# Check database size
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT pg_database_size('ringrift') / 1024 / 1024 AS size_mb;"
```

### Communication

- **Status Page**: Update to "Database connectivity issues - Investigating"
- **Slack**: Post in #incidents with severity and impact
- **Escalation**: If not resolved in 15 min, page database admin

### Post-Incident

- Analyze what caused the database failure
- Review connection pool settings
- Check disk space alerts/monitoring
- Update runbook with any new failure modes

---

## Alert: RedisDown

### Severity

**P1 Critical** - Immediate response required

### Symptoms

- Rate limiting not working (429 errors stop)
- Session management issues
- `ringrift_service_status{service="redis"} == 0`

### Impact

- Rate limiting disabled (abuse vulnerability)
- Increased load on database
- Session caching unavailable

### Initial Triage (2-5 min)

```bash
# 1. Check Redis container status
docker compose ps redis

# 2. Check Redis logs
docker compose logs --tail 200 redis

# 3. Test Redis connectivity
docker exec ringrift-redis-1 redis-cli ping

# 4. Check Redis info
docker exec ringrift-redis-1 redis-cli info server
```

### Diagnosis

#### Redis Container Not Running

```bash
# Check why it stopped
docker compose logs redis | tail -100

# Check for OOM
docker inspect ringrift-redis-1 | jq '.[0].State'
```

#### Redis Running But Not Responding

```bash
# Check memory usage
docker exec ringrift-redis-1 redis-cli info memory

# Check slow log
docker exec ringrift-redis-1 redis-cli slowlog get 10

# Check client connections
docker exec ringrift-redis-1 redis-cli client list
```

### Mitigation

#### Restart Redis

```bash
# Graceful restart
docker compose restart redis

# Verify
docker exec ringrift-redis-1 redis-cli ping

# Restart app to reconnect
docker compose restart app
```

#### Free Redis Memory

```bash
# Check memory
docker exec ringrift-redis-1 redis-cli info memory | grep used_memory_human

# Flush expired keys
docker exec ringrift-redis-1 redis-cli --scan --pattern "*" | head -100

# If necessary, flush all (CAUTION: clears cache)
docker exec ringrift-redis-1 redis-cli flushall
```

### Communication

- **Status Page**: Update to "Cache service issues - Rate limiting degraded"
- **Slack**: Post in #incidents
- **Note**: Rate limiting is disabled, monitor for abuse

---

## Alert: HighErrorRate

### Severity

**P1 Critical** - Immediate response required when >5% of requests return 5xx

### Symptoms

- Users seeing "Something went wrong" errors
- Prometheus showing `http_requests_total{status=~"5.."}` spike
- Application logs full of errors

### Impact

- > 5% of user requests failing
- Degraded user experience
- Potential data inconsistency

### Initial Triage (2-5 min)

```bash
# 1. Check application logs for errors
docker compose logs --tail 500 app 2>&1 | grep -i error | tail -50

# 2. Check health endpoints
curl -s http://localhost:3000/health | jq
curl -s http://localhost:3000/ready | jq

# 3. Identify failing endpoints
curl -s http://localhost:3000/metrics | grep http_requests_total | grep 'status="5'

# 4. Check dependency status
docker compose ps
```

### Diagnosis

#### Identify the Failing Endpoint

```bash
# Group errors by endpoint
docker compose logs --tail 1000 app 2>&1 | \
  grep -E "\"statusCode\":5[0-9]{2}" | \
  jq -r '.url' | sort | uniq -c | sort -rn
```

#### Common Error Patterns

| Error                     | Likely Cause    | Action                   |
| ------------------------- | --------------- | ------------------------ |
| Database connection error | DB down         | Check DatabaseDown       |
| Redis connection error    | Redis down      | Check RedisDown          |
| ECONNREFUSED              | Service down    | Restart affected service |
| ENOMEM                    | Out of memory   | Check HighMemoryUsage    |
| Timeout                   | Slow dependency | Check latency alerts     |

### Mitigation

#### Quick Restart

```bash
# Restart application
docker compose restart app

# Monitor error rate
watch -n 5 'curl -s http://localhost:3000/metrics | grep http_requests_total | grep status=\"5'
```

#### Rollback Recent Deployment

If error spike correlates with recent deployment:

```bash
# See deployment runbook
# Quick version:
git log --oneline -5
docker compose pull app:previous-tag
docker compose up -d app
```

### Communication

- **Status Page**: "Service experiencing errors - Investigating"
- **Slack**: Post error rate and affected functionality
- **Escalation**: Page on-call if error rate doesn't decrease in 10 min

---

## Alert: ElevatedErrorRate

### Severity

**P3 Medium** - Investigation needed (>1% errors for 10 min)

### Symptoms

- Lower rate of errors than HighErrorRate
- May be intermittent
- May affect specific endpoints only

### Initial Triage

Same as HighErrorRate but less urgent. Focus on identifying the specific endpoint or user flow affected.

### Mitigation

1. Identify specific failing endpoints
2. Check for recent changes affecting those endpoints
3. Monitor closely - escalate if it increases toward 5%

---

## Alert: NoHTTPTraffic

### Severity

**P2 High** - Investigate urgently (no requests for 10 min)

### Symptoms

- No HTTP requests recorded in metrics
- Service appears up but unreachable

### Diagnosis

```bash
# Check if app is running
docker compose ps app

# Check if port is listening
docker compose exec app sh -c "netstat -tlnp | grep 3000"

# Check nginx/load balancer
docker compose logs --tail 100 nginx

# Test directly
curl -v http://localhost:3000/health
```

### Common Causes

- Load balancer misconfigured
- Firewall blocking traffic
- DNS issues
- Nginx down
- Network partition

### Mitigation

1. Check load balancer/ingress configuration
2. Verify DNS resolution
3. Check firewall rules
4. Restart nginx if needed

---

## Alert: ServiceDegraded

### Severity

**P3 Medium** - Some features unavailable

### Symptoms

- Application reporting degraded status
- `ringrift_degradation_level > 0`
- Some features returning 503

### Degradation Levels

| Level | Status   | Meaning                   |
| ----- | -------- | ------------------------- |
| 0     | FULL     | All features available    |
| 1     | DEGRADED | Some features unavailable |
| 2     | MINIMAL  | Only core features        |
| 3     | OFFLINE  | Service down              |

### Diagnosis

```bash
# Check which service is causing degradation
curl -s http://localhost:3000/ready | jq

# This shows status of each dependency
# Look for "healthy: false"
```

### Mitigation

Address the underlying service issue. The application auto-recovers when dependencies are restored.

---

## Alert: ServiceMinimalMode

### Severity

**P1 Critical** - Most features unavailable

### Symptoms

- Degradation level ≥ 2
- Only basic endpoints working
- AI service, complex queries unavailable

### Mitigation

1. Identify which dependency is down
2. Restore that dependency
3. Application should auto-recover

---

## Alert: ServiceOffline

### Severity

**P1 Critical** - Complete outage

### Symptoms

- Degradation level = 3
- All user-facing features unavailable

### Mitigation

1. Check all dependencies immediately
2. Identify root cause
3. Restore services in order: Database → Redis → AI Service → App

---

## Alert: NoWebSocketConnections

### Severity

**P2 High** - Real-time features broken

### Symptoms

- No WebSocket connections for 15 min
- Real-time game updates not working
- `ringrift_websocket_connections == 0`

### Diagnosis

```bash
# Check if WebSocket server is running
docker compose logs --tail 100 app | grep -i websocket

# Test WebSocket connection
# (Requires wscat: npm install -g wscat)
APP_PORT=${APP_PORT:-3000}
wscat -c ws://localhost:${APP_PORT}
```

### Common Causes

- WebSocket port not exposed
- Nginx WebSocket upgrade not configured
- All clients disconnected (off-peak?)

### Mitigation

1. Check nginx WebSocket configuration
2. Verify the app port is accessible (HTTP + WebSocket share `PORT`)
3. Restart app if WebSocket server crashed

---

## Alert: NoActiveGames

### Severity

**P4 Low** - Informational

### Symptoms

- No games running for 30 min
- May be normal during off-peak

### Response

- If during peak hours (10 AM - 10 PM local), investigate
- If off-peak, likely normal - no action needed

---

## Alert: LongRunningGames

### Severity

**P4 Low** - Investigate stalled games

### Symptoms

- Median game duration > 1 hour
- May indicate orphaned game sessions

### Diagnosis

```bash
# Check for stalled games in database
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT id, status, created_at, updated_at
   FROM games
   WHERE status = 'in_progress'
   AND updated_at < now() - interval '1 hour';"
```

### Mitigation

Consider implementing game timeout logic if not already present.

---

## Alert: RulesParityValidationMismatch

### Severity

**P3 Medium** - Rules engine consistency issue

### Symptoms

- TypeScript and Python rules engines disagreeing on move validity
- > 5 mismatches per hour

### Diagnosis

```bash
# Check logs for parity errors
docker compose logs --tail 500 app | grep -i parity

# Check AI service logs
docker compose logs --tail 500 ai-service | grep -i parity
```

### Response

- This is a code issue - file a bug
- Game functionality continues (TypeScript engine is authoritative)
- Schedule investigation for next sprint

---

## Alert: RulesParityGameStatusMismatch

### Severity

**P1 Critical** - Game integrity issue

### Symptoms

- Different win/loss outcomes between engines
- Critical game integrity violation

### Immediate Actions

1. **Capture affected game IDs** from logs
2. **Document the state** at time of mismatch
3. **Disable Python rules** if feature-flagged
4. **File critical bug** with reproduction steps

### Post-Incident

- Root cause analysis mandatory
- Review all affected games
- Consider player compensation if outcomes affected

---

## Quick Reference: Restart Commands

| Service       | Command                                       |
| ------------- | --------------------------------------------- |
| Application   | `docker compose restart app`                  |
| Database      | `docker compose restart postgres`             |
| Redis         | `docker compose restart redis`                |
| All services  | `docker compose restart`                      |
| Full recreate | `docker compose down && docker compose up -d` |

---

## Related Documentation

- [Initial Triage](TRIAGE_GUIDE.md)
- [Latency Incidents](LATENCY.md)
- [Resource Incidents](RESOURCES.md)
- [Deployment Runbooks](../runbooks/INDEX.md)
- [Database Operations](../operations/OPERATIONS_DB.md)
