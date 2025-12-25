# Resource Incidents

This guide covers incidents related to resource exhaustion including memory, CPU, event loop blocking, and connection limits.

## Alerts Covered

| Alert                    | Severity | Threshold | Duration |
| ------------------------ | -------- | --------- | -------- |
| HighMemoryUsage          | Warning  | > 1.5 GB  | 10 min   |
| HighMemoryUsageCritical  | Critical | > 2 GB    | 5 min    |
| HighEventLoopLag         | Warning  | > 100ms   | 5 min    |
| HighEventLoopLagCritical | Critical | > 500ms   | 2 min    |
| HighActiveHandles        | Warning  | > 10,000  | 10 min   |
| HighWebSocketConnections | Warning  | > 1,000   | 5 min    |

---

## Alert: HighMemoryUsage

### Severity

**P3 Medium** - Memory usage over 1.5 GB (warning threshold)

### Symptoms

- Memory approaching container limit
- May see increased GC activity
- Response times may be slightly elevated
- `process_resident_memory_bytes / 1024 / 1024 / 1024 > 1.5`

### Impact

- Risk of OOM kill if memory continues to grow
- GC pauses may affect latency
- Service may become unstable

### Initial Triage (5 min)

```bash
# 1. Check current memory usage
curl -s http://localhost:3000/metrics | grep process_resident_memory_bytes

# 2. Check heap usage
curl -s http://localhost:3000/metrics | grep nodejs_heap

# 3. Check GC metrics
curl -s http://localhost:3000/metrics | grep nodejs_gc

# 4. Check container stats
docker stats --no-stream ringrift-app-1
```

### Diagnosis

#### Memory Growth Pattern

```bash
# Watch memory over time (every 10 seconds)
watch -n 10 'curl -s http://localhost:3000/metrics | grep process_resident_memory_bytes'

# Check if it's growing or stable
# Growing = likely leak
# Stable high = may need more memory or optimization
```

#### Identify Memory Consumers

```bash
# Check heap breakdown (if exposed)
curl -s http://localhost:3000/metrics | grep nodejs_heap_space_size_used_bytes

# Check external memory
curl -s http://localhost:3000/metrics | grep nodejs_external_memory_bytes
```

#### Check for Connection/Handle Leaks

```bash
# Check active handles
curl -s http://localhost:3000/metrics | grep nodejs_active_handles_total

# Check active requests
curl -s http://localhost:3000/metrics | grep nodejs_active_requests_total

# Check WebSocket connections
curl -s http://localhost:3000/metrics | grep ringrift_websocket_connections
```

### Common Causes

| Indicator                 | Likely Cause       | Solution                                 |
| ------------------------- | ------------------ | ---------------------------------------- |
| Heap growing continuously | Memory leak        | Requires code fix, restart as mitigation |
| High external memory      | Buffer/stream leak | Check file/network streams               |
| Many active handles       | Connection leak    | Check DB/Redis connections               |
| Correlated with traffic   | Under-provisioned  | Scale up or optimize                     |

### Mitigation

#### Monitor and Prepare

```bash
# If memory is high but stable, monitor closely
watch -n 30 'docker stats --no-stream ringrift-app-1'

# Set up for quick restart if needed
# Prepare rolling restart command
```

#### Trigger Garbage Collection (if exposed)

```bash
# If the app exposes GC endpoint (check if available)
curl -X POST http://localhost:3000/admin/gc
```

#### Restart Application (if growing)

```bash
# Graceful restart
docker compose restart app

# Verify memory after restart
sleep 10
curl -s http://localhost:3000/metrics | grep process_resident_memory_bytes
```

### Communication

- **Status Page**: No update needed for warning
- **Slack**: Post in #alerts for awareness
- **Monitor**: Watch for escalation to critical (2GB)

---

## Alert: HighMemoryUsageCritical

### Severity

**P1 Critical** - Memory over 2 GB, OOM risk imminent

### Symptoms

- Memory at or near container limit
- Possible increased latency due to GC
- Risk of container being killed

### Impact

- **OOM kill imminent** if not addressed
- Service will crash if container limit reached
- All in-flight requests will fail

### Immediate Actions (First 2 min)

```bash
# 1. Check if still running
docker compose ps app

# 2. Get current memory
docker stats --no-stream ringrift-app-1

# 3. Immediate restart to prevent OOM
docker compose restart app
```

### Post-Restart Investigation

```bash
# 1. Verify memory is lower
sleep 15
curl -s http://localhost:3000/metrics | grep process_resident_memory_bytes

# 2. Monitor growth rate
for i in 1 2 3 4 5; do
  echo "$(date): $(curl -s http://localhost:3000/metrics | grep process_resident_memory_bytes)"
  sleep 60
done
```

### Diagnosis (After Stabilization)

#### Enable Heap Dump (if available)

```bash
# Trigger heap dump before next restart (if endpoint exists)
curl -X POST http://localhost:3000/admin/heapdump

# Or from within container
docker compose exec app node --heapsnapshot-signal=SIGUSR2 &
docker compose exec app kill -SIGUSR2 1
```

#### Check for Patterns

```bash
# Correlate with traffic
curl -s http://localhost:3000/metrics | grep http_requests_total

# Check game counts
curl -s http://localhost:3000/metrics | grep ringrift_games_active

# WebSocket connections
curl -s http://localhost:3000/metrics | grep ringrift_websocket_connections
```

### Long-Term Actions

- File bug for memory leak investigation
- Review recent code changes
- Consider increasing container memory limit temporarily
- Add memory profiling to staging environment

### Communication

- **Status Page**: Update if restart causes downtime
- **Slack**: Post in #incidents with memory stats
- **Escalation**: If memory grows back to critical in <1 hour

---

## Alert: HighEventLoopLag

### Severity

**P3 Medium** - Event loop blocked > 100ms

### Symptoms

- All async operations delayed
- API responses slower than usual
- WebSocket messages delayed
- `nodejs_eventloop_lag_seconds > 0.1`

### Impact

- All requests affected (Node.js is single-threaded for JS)
- User-perceived latency increases
- WebSocket real-time updates delayed

### Initial Triage (5 min)

```bash
# 1. Check event loop lag
curl -s http://localhost:3000/metrics | grep nodejs_eventloop_lag

# 2. Check CPU usage
docker stats --no-stream ringrift-app-1

# 3. Check for long-running operations
docker compose logs --tail 200 app | grep -i "slow\|timeout\|blocked"

# 4. Check concurrent requests
curl -s http://localhost:3000/metrics | grep nodejs_active_requests
```

### Diagnosis

#### Identify Blocking Operations

Common blocking operations in Node.js:

| Type               | Indicator                          | Check                 |
| ------------------ | ---------------------------------- | --------------------- |
| Synchronous I/O    | Correlated with specific endpoints | Check endpoint logs   |
| Large JSON parsing | Spikes with large payloads         | Check request sizes   |
| CPU computation    | High CPU with lag                  | Profile CPU           |
| Complex regex      | Specific input patterns            | Check validation code |
| Crypto operations  | Auth/encryption endpoints          | Check auth flow       |

```bash
# Check CPU usage pattern
docker stats ringrift-app-1

# Check for CPU-intensive operations in logs
docker compose logs --tail 500 app | grep -E "processing|compute|parse|JSON"
```

#### Check for External Causes

```bash
# High traffic?
curl -s http://localhost:3000/metrics | grep http_requests_total

# Many concurrent games?
curl -s http://localhost:3000/metrics | grep ringrift_games_active

# AI service causing delays?
curl -s http://localhost:3000/metrics | grep ringrift_ai_request_duration
```

### Mitigation

#### Restart Application

```bash
# Quick restart to clear any stuck operations
docker compose restart app

# Monitor event loop after restart
watch -n 5 'curl -s http://localhost:3000/metrics | grep nodejs_eventloop_lag_seconds'
```

#### Scale if Traffic-Related

```bash
# Add more instances
docker compose up -d --scale app=2
```

### Communication

- **Slack**: Post in #alerts if sustained
- **Monitor**: Watch for escalation to critical (500ms)

---

## Alert: HighEventLoopLagCritical

### Severity

**P1 Critical** - Event loop blocked > 500ms

### Symptoms

- Application effectively unresponsive
- Requests timing out
- WebSocket connections may be dropped
- Health checks may fail

### Impact

- **Service nearly unusable**
- Requests timing out
- Users seeing errors

### Immediate Actions

```bash
# 1. Immediate restart - app is effectively down anyway
docker compose restart app

# 2. Verify recovery
sleep 10
curl -s http://localhost:3000/health

# 3. Monitor event loop
curl -s http://localhost:3000/metrics | grep nodejs_eventloop_lag_seconds
```

### Post-Restart Investigation

```bash
# Check what was happening before the lag
docker compose logs --since 10m app | head -200

# Look for patterns
docker compose logs --since 10m app | grep -E "error|fail|timeout|slow"
```

### Common Causes and Fixes

| Cause         | Evidence                    | Fix               |
| ------------- | --------------------------- | ----------------- |
| Infinite loop | 100% CPU                    | Code fix required |
| Large payload | Specific request before lag | Add size limits   |
| Regex DoS     | Specific input              | Fix regex         |
| Sync file I/O | File operation logs         | Make async        |

### Communication

- **Status Page**: "Service degraded - Requests may be slow or fail"
- **Slack**: Escalate to #incidents
- **Escalation**: If happens repeatedly, page team lead

---

## Alert: HighActiveHandles

### Severity

**P3 Medium** - Potential resource leak (>10,000 handles)

### Symptoms

- High number of Node.js active handles
- May indicate connection or timer leaks
- `nodejs_active_handles_total > 10000`

### Impact

- Potential resource exhaustion
- May lead to memory growth
- Could affect stability

### Diagnosis

```bash
# Check handle count
curl -s http://localhost:3000/metrics | grep nodejs_active_handles_total

# Check WebSocket connections (common source)
curl -s http://localhost:3000/metrics | grep ringrift_websocket_connections

# Check database connections
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT count(*) FROM pg_stat_activity WHERE application_name LIKE '%ringrift%';"

# Check Redis connections
docker exec ringrift-redis-1 redis-cli client list | wc -l
```

### Common Handle Sources

| Source                | Normal Count    | Check                          |
| --------------------- | --------------- | ------------------------------ |
| WebSocket connections | ~1 per user     | ringrift_websocket_connections |
| Database connections  | Pool size (~20) | pg_stat_activity               |
| Redis connections     | Pool size (~10) | redis client list              |
| Timers                | Varies          | Often from setInterval leaks   |
| File handles          | Should be low   | Check for file stream leaks    |

### Mitigation

```bash
# Restart to clear leaked handles
docker compose restart app

# Monitor after restart
watch -n 30 'curl -s http://localhost:3000/metrics | grep nodejs_active_handles_total'
```

### Long-Term Fix

- Audit code for:
  - Timers not being cleared
  - Connections not being closed
  - Event listeners not being removed

---

## Alert: HighWebSocketConnections

### Severity

**P3 Medium** - Many WebSocket connections (>1,000)

### Symptoms

- High concurrent WebSocket connections
- May indicate successful scaling OR connection leaks

### Impact

- Higher memory usage
- May approach connection limits
- If leaking, will exhaust resources

### Diagnosis

```bash
# Check connection count
curl -s http://localhost:3000/metrics | grep ringrift_websocket_connections

# Compare to active games/users
curl -s http://localhost:3000/metrics | grep ringrift_games_active
curl -s http://localhost:3000/metrics | grep ringrift_users_online
```

### Assessment

**Normal if:**

- High traffic period
- Connection count correlates with active games
- Count is stable

**Suspicious if:**

- Connections growing without traffic
- Much higher than expected for game count
- Growing even during low traffic

### Mitigation (if leak suspected)

```bash
# Restart to clear leaked connections
docker compose restart app

# Monitor growth rate after restart
watch -n 60 'curl -s http://localhost:3000/metrics | grep ringrift_websocket_connections'
```

---

## Resource Diagnostics Reference

### Quick System Check

```bash
# All-in-one resource check
echo "=== Container Stats ===" && \
docker stats --no-stream ringrift-app-1 && \
echo "=== Memory ===" && \
curl -s http://localhost:3000/metrics | grep process_resident_memory && \
echo "=== Event Loop ===" && \
curl -s http://localhost:3000/metrics | grep nodejs_eventloop_lag && \
echo "=== Handles ===" && \
curl -s http://localhost:3000/metrics | grep nodejs_active_handles && \
echo "=== Connections ===" && \
curl -s http://localhost:3000/metrics | grep ringrift_websocket_connections
```

### Docker Resource Commands

```bash
# Container stats
docker stats ringrift-app-1

# Container resource limits
docker inspect ringrift-app-1 | jq '.[0].HostConfig.Memory'

# Check for OOM kills
docker inspect ringrift-app-1 | jq '.[0].State.OOMKilled'

# Container logs for kill events
dmesg | grep -i "killed process" | tail -10
```

### Node.js Memory Commands

```bash
# Heap statistics
curl -s http://localhost:3000/metrics | grep nodejs_heap

# GC statistics
curl -s http://localhost:3000/metrics | grep nodejs_gc

# External memory
curl -s http://localhost:3000/metrics | grep nodejs_external_memory
```

---

## Scaling Considerations

When resource alerts indicate capacity issues (not leaks):

### Vertical Scaling

```bash
# Increase container memory limit
# Edit docker-compose.yml:
#   deploy:
#     resources:
#       limits:
#         memory: 4G
docker compose up -d
```

### Horizontal Scaling

```bash
# Add more app instances
docker compose up -d --scale app=3

# Ensure load balancer is configured
```

See [DEPLOYMENT_SCALING.md](../runbooks/DEPLOYMENT_SCALING.md) for full scaling procedures.

---

## Related Documentation

- [Initial Triage](TRIAGE_GUIDE.md)
- [Availability Incidents](AVAILABILITY.md)
- [Latency Incidents](LATENCY.md)
- [Scaling Procedures](../runbooks/DEPLOYMENT_SCALING.md)
- [Alerting Thresholds](../operations/ALERTING_THRESHOLDS.md)
