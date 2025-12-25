# Latency Incidents

This guide covers incidents related to response time degradation and performance issues.

## Alerts Covered

| Alert                    | Severity | Threshold               | Duration |
| ------------------------ | -------- | ----------------------- | -------- |
| HighP99Latency           | Warning  | P99 > 2s                | 5 min    |
| HighP99LatencyCritical   | Critical | P99 > 5s                | 2 min    |
| HighP95Latency           | Warning  | P95 > 1s                | 10 min   |
| HighMedianLatency        | Warning  | P50 > 500ms             | 15 min   |
| HighGameMoveLatency      | Warning  | P99 > 1s per board type | 5 min    |
| DatabaseResponseTimeSlow | Warning  | P99 > 500ms             | 5 min    |
| RedisResponseTimeSlow    | Warning  | P99 > 100ms             | 5 min    |

---

## Alert: HighP99Latency

### Severity

**P3 Medium** - 1% of users experiencing slow responses (>2s)

### Symptoms

- Some users reporting slow page loads
- Metrics showing P99 latency > 2 seconds
- Game moves feel sluggish for some players

### Impact

- 1% of requests taking >2 seconds
- User experience degraded
- Potential for users abandoning actions

### Initial Triage (5 min)

```bash
# 1. Check current latency metrics
curl -s http://localhost:3000/metrics | grep http_request_duration

# 2. Identify slow endpoints
curl -s http://localhost:3000/metrics | grep http_request_duration_seconds_bucket | \
  grep -v "le=\"" | sort -t'"' -k4 -n | tail -10

# 3. Check event loop lag
curl -s http://localhost:3000/metrics | grep nodejs_eventloop_lag

# 4. Check service dependencies
curl -s http://localhost:3000/ready | jq

# 5. Check database response times
curl -s http://localhost:3000/metrics | grep ringrift_service_response_time
```

### Diagnosis

#### Identify Slow Endpoints

```bash
# Check which endpoints are slow
docker compose logs --tail 500 app 2>&1 | \
  jq -r 'select(.responseTime > 2000) | "\(.method) \(.url) \(.responseTime)ms"' | \
  sort | uniq -c | sort -rn
```

#### Check Database Performance

```bash
# Check slow queries
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT query, calls, mean_exec_time, total_exec_time
   FROM pg_stat_statements
   ORDER BY mean_exec_time DESC LIMIT 10;"

# Check active queries
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT pid, query_start, state, query
   FROM pg_stat_activity
   WHERE state = 'active';"
```

#### Check Event Loop

```bash
# Check event loop lag in metrics
curl -s http://localhost:3000/metrics | grep nodejs_eventloop_lag_seconds

# Check for blocking operations in logs
docker compose logs --tail 500 app | grep -i "blocked\|sync\|timeout"
```

#### Check Memory/GC

```bash
# Check garbage collection metrics
curl -s http://localhost:3000/metrics | grep nodejs_gc

# Check heap usage
curl -s http://localhost:3000/metrics | grep nodejs_heap
```

### Common Causes

| Cause                     | Indicators                  | Solution                      |
| ------------------------- | --------------------------- | ----------------------------- |
| Slow database queries     | Database response time high | Optimize queries, add indexes |
| Event loop blocking       | Event loop lag elevated     | Identify sync code, optimize  |
| Memory pressure           | High GC time, heap usage    | Profile memory, restart       |
| Network latency           | All external calls slow     | Check network, DNS            |
| Connection pool exhausted | Connection wait times high  | Increase pool size            |

### Mitigation

#### Restart Application (Quick Fix)

```bash
# Restart application to clear any accumulated state
docker compose restart app

# Monitor latency improvement
watch -n 5 'curl -s http://localhost:3000/metrics | grep http_request_duration_seconds_sum'
```

#### Enable Query Logging

```bash
# Temporarily enable slow query logging
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "ALTER SYSTEM SET log_min_duration_statement = 500;"
docker exec ringrift-postgres-1 psql -U ringrift -c "SELECT pg_reload_conf();"

# Watch for slow queries
docker compose logs -f postgres | grep duration
```

#### Scale Horizontally (if available)

```bash
# Increase app instances
docker compose up -d --scale app=3
```

### Communication

- **Status Page**: "Some users may experience slow performance"
- **Slack**: Post in #alerts with latency metrics
- **Monitor**: Watch for escalation to P99 > 5s Critical

### Post-Incident

- Identify the specific slow endpoints/queries
- Create optimization tickets
- Consider adding caching

---

## Alert: HighP99LatencyCritical

### Severity

**P1 Critical** - 1% of users experiencing timeouts (>5s)

### Symptoms

- Users seeing timeout errors
- P99 latency > 5 seconds
- Significant user complaints

### Impact

- Requests timing out
- Users unable to complete actions
- Game moves may fail

### Initial Triage (2-3 min)

```bash
# 1. Immediately check for event loop blocking
curl -s http://localhost:3000/metrics | grep nodejs_eventloop_lag

# 2. Check database
docker exec ringrift-postgres-1 pg_isready

# 3. Check active queries (may be blocking)
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT pid, now() - query_start as runtime, state, query
   FROM pg_stat_activity
   WHERE state != 'idle' ORDER BY runtime DESC LIMIT 5;"
```

### Immediate Actions

#### Kill Long-Running Queries

```bash
# Find and kill queries running > 30 seconds
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
   WHERE state = 'active' AND query_start < now() - interval '30 seconds';"
```

#### Emergency Restart

If latency doesn't improve within 5 minutes:

```bash
# Restart app (preserves database)
docker compose restart app

# If still slow, restart database connection by restarting app again
docker compose stop app && sleep 5 && docker compose start app
```

### Diagnosis (in parallel with mitigation)

Follow same diagnosis steps as HighP99Latency but with more urgency. Look for:

1. **Event loop lag > 500ms**: Something is blocking
2. **Database query > 10s**: Query needs optimization or killing
3. **Memory > 2GB**: Memory leak, restart needed
4. **GC pause > 500ms**: Memory pressure, restart

### Communication

- **Status Page**: "Service degraded - Some requests timing out"
- **Slack**: Escalate to #incidents
- **Escalation**: If not improving in 10 min, page secondary on-call

---

## Alert: HighP95Latency

### Severity

**P3 Medium** - 5% of users experiencing slow responses (>1s)

### Symptoms

- More widespread slowness than P99
- General sense that the app is "sluggish"

### Response

Similar to HighP99Latency but indicates broader performance issue. Focus on:

1. Database query optimization
2. Caching effectiveness
3. Network latency to dependencies
4. Consider scaling

---

## Alert: HighMedianLatency

### Severity

**P3 Medium** - 50% of users affected (median >500ms)

### Symptoms

- **Majority of users** experiencing slow responses
- General performance degradation
- Not isolated to specific endpoints

### Impact

- This is systemic - affects half of all requests
- User experience broadly degraded

### Diagnosis Focus

This typically indicates:

1. **Database is slow** - most requests hit DB
2. **Event loop is blocked** - affects all requests
3. **Infrastructure issue** - network, disk I/O

```bash
# Check overall system
docker stats --no-stream

# Check database wait events
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT wait_event_type, wait_event, count(*)
   FROM pg_stat_activity
   WHERE state = 'active'
   GROUP BY wait_event_type, wait_event;"
```

### Communication

- This affects most users - status page update needed
- "Performance degraded - investigating"

---

## Alert: HighGameMoveLatency

### Severity

**P3 Medium** - Game-specific latency issue

### Symptoms

- Game moves taking >1 second to process
- Players experiencing lag between moves
- Alert includes board_type label

### Impact

- Game experience degraded
- Competitive play affected
- Players may abandon games

### Diagnosis

```bash
# Check game move latency by board type
curl -s http://localhost:3000/metrics | grep ringrift_game_move_latency

# Check rules engine performance
docker compose logs --tail 200 app | grep -i "move\|rules\|apply"

# Check AI service latency (if AI games)
curl -s http://localhost:3000/metrics | grep ringrift_ai_request_duration
```

### Common Causes

| Board Type    | Possible Cause                   |
| ------------- | -------------------------------- |
| All types     | Rules engine performance         |
| Larger boards | Territory calculation complexity |
| AI games      | AI service response time         |

### Mitigation

If specific to board type, may need code optimization. For immediate relief:

```bash
# Restart app to clear any accumulated state
docker compose restart app

# If AI-related, check AI service
curl -s http://localhost:8001/health
docker compose logs --tail 100 ai-service
```

---

## Alert: DatabaseResponseTimeSlow

### Severity

**P3 Medium** - Database queries slow (P99 > 500ms)

### Symptoms

- Database queries taking longer than expected
- Affects all endpoints that touch database

### Diagnosis

```bash
# Check database performance
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT relname, seq_scan, idx_scan, n_tup_ins, n_tup_upd
   FROM pg_stat_user_tables ORDER BY seq_scan DESC LIMIT 10;"

# Check for missing indexes (high seq_scan)
# Check for table bloat
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
   FROM pg_stat_user_tables ORDER BY pg_total_relation_size(relid) DESC;"

# Check connection count
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT count(*) FROM pg_stat_activity;"
```

### Common Fixes

```bash
# Run VACUUM ANALYZE
docker exec ringrift-postgres-1 psql -U ringrift -c "VACUUM ANALYZE;"

# Check for lock contention
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT * FROM pg_locks WHERE NOT granted;"
```

---

## Alert: RedisResponseTimeSlow

### Severity

**P3 Medium** - Redis operations slow (P99 > 100ms)

### Symptoms

- Cache operations taking too long
- Rate limiting checks slow
- Session operations slow

### Diagnosis

```bash
# Check Redis latency
docker exec ringrift-redis-1 redis-cli --latency

# Check Redis slow log
docker exec ringrift-redis-1 redis-cli slowlog get 10

# Check memory usage
docker exec ringrift-redis-1 redis-cli info memory

# Check connected clients
docker exec ringrift-redis-1 redis-cli info clients
```

### Common Fixes

```bash
# Check for large keys
docker exec ringrift-redis-1 redis-cli --bigkeys

# Clear expired keys
docker exec ringrift-redis-1 redis-cli keys "*" | xargs -n 100 redis-cli del
```

---

## Diagnostic Commands Reference

### Quick Latency Check

```bash
# All-in-one latency check
echo "=== Event Loop ===" && \
curl -s http://localhost:3000/metrics | grep nodejs_eventloop_lag_seconds && \
echo "=== HTTP Latency ===" && \
curl -s http://localhost:3000/metrics | grep http_request_duration_seconds_sum && \
echo "=== Database ===" && \
curl -s http://localhost:3000/metrics | grep 'ringrift_service_response_time.*database' && \
echo "=== Redis ===" && \
curl -s http://localhost:3000/metrics | grep 'ringrift_service_response_time.*redis'
```

### Profile Application

```bash
# If Node.js profiling is enabled
docker compose exec app node --prof-process isolate-*.log > profile.txt

# Check GC impact
curl -s http://localhost:3000/metrics | grep nodejs_gc_duration_seconds
```

### Database Query Analysis

```bash
# Enable pg_stat_statements if not already
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"

# Get top slow queries
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT query, calls, mean_exec_time, total_exec_time
   FROM pg_stat_statements
   ORDER BY mean_exec_time DESC LIMIT 10;"
```

---

## Related Documentation

- [Initial Triage](TRIAGE_GUIDE.md)
- [Availability Incidents](AVAILABILITY.md)
- [Resource Incidents](RESOURCES.md)
- [Database Operations](../operations/OPERATIONS_DB.md)
- [Alerting Thresholds](../operations/ALERTING_THRESHOLDS.md)
