# AI Service Incidents

This guide covers incidents related to the AI service, including service unavailability, high fallback rates, and performance issues.

## Alerts Covered

| Alert                  | Severity | Threshold            | Duration |
| ---------------------- | -------- | -------------------- | -------- |
| AIServiceDown          | Warning  | Service status = 0   | 2 min    |
| AIFallbackRateHigh     | Warning  | > 30% fallback       | 10 min   |
| AIFallbackRateCritical | Critical | > 50% fallback       | 5 min    |
| AIRequestHighLatency   | Warning  | P99 > 5s             | 5 min    |
| AIErrorsIncreasing     | Warning  | Error rate > 0.1/sec | 5 min    |

---

## AI Service Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI REQUEST FLOW                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Game Move Request                                                       │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────┐    HTTP/8001    ┌─────────────────┐                   │
│  │ Node.js App  │ ──────────────▶ │   AI Service    │                   │
│  │ AIEngine.ts  │                 │   Python/FastAPI │                   │
│  └──────────────┘                 └─────────────────┘                   │
│        │                                  │                              │
│        │ (fallback if AI fails)           │                              │
│        ▼                                  ▼                              │
│  ┌──────────────┐              ┌─────────────────┐                      │
│  │   Local      │              │  Neural Network │                      │
│  │  Heuristic   │              │     Model       │                      │
│  │   Fallback   │              └─────────────────┘                      │
│  └──────────────┘                                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**

- AI service is **optional** - games continue with fallback if down
- Local heuristic fallback provides reduced but functional AI
- AI service typically responds in 1-3 seconds

---

## Alert: AIServiceDown

### Severity

**P2 High** - AI service unavailable (2 min without response)

### Symptoms

- AI service health check failing
- All AI requests using fallback
- `ringrift_service_status{service="ai_service"} == 0`

### Impact

- AI games continue with **reduced move quality**
- Local heuristic fallback activated
- Not a full outage - games still playable

### Initial Triage (5 min)

```bash
# 1. Check AI service container
docker compose ps ai-service

# 2. Test AI service health endpoint
curl -s http://localhost:8001/health | jq

# 3. Check AI service logs
docker compose logs --tail 200 ai-service

# 4. Check if app can reach AI service
docker compose exec app curl -s http://ai-service:8001/health
```

### Diagnosis

#### Container Not Running

```bash
# Check container status
docker compose ps ai-service

# Check why it stopped
docker compose logs ai-service | tail -100

# Check for OOM
docker inspect ringrift-ai-service-1 | jq '.[0].State'
```

**Common causes:**

- Out of memory (models are memory-intensive)
- Python crash
- GPU issues (if GPU-enabled)
- Configuration error

#### Container Running But Unhealthy

```bash
# Check health endpoint directly
docker compose exec ai-service curl -s localhost:8001/health

# Check if model is loaded
docker compose logs ai-service | grep -i "model\|load"

# Check Python errors
docker compose logs ai-service | grep -E "Error|Exception|Traceback"
```

**Common causes:**

- Model not loaded correctly
- Dependency issue
- Environment variable missing
- Port not binding

#### Network Connectivity Issue

```bash
# Test from app to AI service
docker compose exec app nc -zv ai-service 8001

# Check Docker network
docker network inspect ringrift_default
```

### Mitigation

#### Restart AI Service

```bash
# Graceful restart
docker compose restart ai-service

# Wait for model to load (can take 30-60 seconds)
sleep 60

# Verify health
curl -s http://localhost:8001/health | jq
```

#### Check Model Loading

```bash
# Watch AI service logs during startup
docker compose logs -f ai-service

# Look for "Model loaded" or similar message
```

#### Emergency: Force New Container

```bash
# Remove and recreate
docker compose stop ai-service
docker compose rm -f ai-service
docker compose up -d ai-service

# Monitor startup
docker compose logs -f ai-service
```

### Communication

- **Status Page**: "AI features operating in reduced mode"
- **Slack**: Post in #alerts
- **Note**: Users playing against AI will see simpler moves

### Post-Incident

- Document what caused the failure
- Check model loading reliability
- Review memory limits

---

## Alert: AIFallbackRateHigh

### Severity

**P3 Medium** - More than 30% of AI requests falling back to heuristics

### Symptoms

- AI service responding but with many failures
- Noticeable difference in AI game quality
- Alert indicates elevated fallback rate

### Impact

- ~30% of AI moves are from simpler local heuristic
- AI game quality degraded
- Service is functional but suboptimal

### Initial Triage (5 min)

```bash
# 1. Check fallback metrics
curl -s http://localhost:3000/metrics | grep ringrift_ai_fallback

# 2. Check AI request success/failure
curl -s http://localhost:3000/metrics | grep ringrift_ai_requests_total

# 3. Check AI service health
curl -s http://localhost:8001/health | jq

# 4. Check AI service logs for errors
docker compose logs --tail 200 ai-service | grep -E "error|fail|timeout"
```

### Diagnosis

#### Calculate Current Fallback Rate

```bash
# Get raw metrics
TOTAL=$(curl -s http://localhost:3000/metrics | grep -oP 'ringrift_ai_requests_total \K[0-9.]+')
FALLBACK=$(curl -s http://localhost:3000/metrics | grep -oP 'ringrift_ai_fallback_total \K[0-9.]+')

echo "Fallback rate: $(echo "scale=2; $FALLBACK / $TOTAL * 100" | bc)%"
```

#### Check AI Service Performance

```bash
# Check AI request latency
curl -s http://localhost:3000/metrics | grep ringrift_ai_request_duration

# Check if timeouts are causing fallbacks
docker compose logs --tail 500 app | grep -i "ai.*timeout"
```

#### Common Causes

| Cause                | Evidence                  | Solution                     |
| -------------------- | ------------------------- | ---------------------------- |
| AI service timeouts  | Timeout errors in logs    | Increase timeout or optimize |
| AI service errors    | Error messages in AI logs | Fix error condition          |
| Network issues       | Connection errors         | Check network                |
| Resource constraints | High CPU/memory on AI     | Scale AI service             |
| Model issues         | Prediction errors         | Check model configuration    |

### Mitigation

#### Restart AI Service

```bash
# Restart to clear any stuck state
docker compose restart ai-service

# Wait for full initialization
sleep 60

# Monitor fallback rate
watch -n 30 'curl -s http://localhost:3000/metrics | grep ringrift_ai_fallback'
```

#### Increase Timeout (if timeout-related)

```bash
# Check current timeout configuration
# In environment variables or config
echo $AI_REQUEST_TIMEOUT

# If timeouts are causing fallbacks, may need to increase
# Edit docker-compose.yml or .env and restart app
```

### Communication

- **Slack**: Post in #alerts with fallback percentage
- **Monitor**: Watch closely for escalation to 50% (Critical)

---

## Alert: AIFallbackRateCritical

### Severity

**P1 Critical** - More than 50% of AI requests falling back

### Symptoms

- Majority of AI moves using local heuristics
- AI service is effectively failing for most requests
- High error/timeout rate

### Impact

- AI game quality significantly degraded
- Most AI games not using trained model
- User experience for AI games is poor

### Immediate Actions (First 5 min)

```bash
# 1. Check if AI service is responding at all
curl -s http://localhost:8001/health | jq

# 2. Check error patterns
docker compose logs --tail 300 ai-service | grep -E "ERROR|Exception" | tail -20

# 3. Try restarting AI service
docker compose restart ai-service

# 4. Monitor recovery
watch -n 10 'curl -s http://localhost:3000/metrics | grep ringrift_ai_fallback'
```

### Diagnosis

If restart doesn't help:

```bash
# Check resource usage
docker stats --no-stream ringrift-ai-service-1

# Check for Python process issues
docker compose exec ai-service ps aux

# Check if model is loaded
docker compose logs ai-service | grep -i "model loaded"

# Check for dependency issues
docker compose exec ai-service pip check
```

### Mitigation

#### Full Service Recreate

```bash
# Stop and remove container
docker compose stop ai-service
docker compose rm -f ai-service

# Pull latest image (if using image)
docker compose pull ai-service

# Start fresh
docker compose up -d ai-service

# Monitor startup
docker compose logs -f ai-service
```

#### Scale AI Service (if capacity issue)

```bash
# If multiple instances are supported
docker compose up -d --scale ai-service=2
```

#### Disable AI Service (emergency)

If AI service is causing cascading issues:

```bash
# Optionally route all games to fallback temporarily
# This requires feature flag support
# Set AI_ENABLED=false in config if available
```

### Communication

- **Status Page**: "AI game quality degraded - Games using simplified AI"
- **Slack**: Escalate to #incidents
- **User Impact**: "AI opponents may make simpler moves than usual"

### Post-Incident

- Root cause analysis for high failure rate
- Review AI service resource limits
- Consider redundancy for AI service

---

## Alert: AIRequestHighLatency

### Severity

**P3 Medium** - AI requests taking >5 seconds at P99

### Symptoms

- Slow AI move responses
- Games feel sluggish when playing against AI
- `histogram_quantile(0.99, ringrift_ai_request_duration_seconds) > 5`

### Impact

- AI turn feels slow
- User waiting for AI moves
- May cause timeout-based fallbacks

### Diagnosis

```bash
# Check AI request duration distribution
curl -s http://localhost:3000/metrics | grep ringrift_ai_request_duration

# Check AI service internal performance
docker compose logs --tail 200 ai-service | grep -i "duration\|time\|seconds"

# Check AI service resource usage
docker stats --no-stream ringrift-ai-service-1

# Check for long-running inference
docker compose exec ai-service ps aux --sort=-time
```

### Common Causes

| Cause               | Evidence                    | Solution                  |
| ------------------- | --------------------------- | ------------------------- |
| Model too large     | High memory, slow inference | Use smaller model         |
| CPU bound           | High CPU usage              | Add resources or optimize |
| Complex board state | Large boards slower         | Optimize for board size   |
| Batch queuing       | Multiple requests waiting   | Scale AI service          |

### Mitigation

```bash
# Restart AI service
docker compose restart ai-service

# If resource-bound, consider scaling
docker stats --no-stream ringrift-ai-service-1
```

### Long-Term Fixes

- Optimize model inference
- Consider GPU acceleration
- Implement request batching
- Add caching for common positions

---

## Alert: AIErrorsIncreasing

### Severity

**P3 Medium** - AI service returning errors at >0.1/sec

### Symptoms

- Error rate increasing in AI requests
- Not yet causing high fallback, but trending poorly

### Diagnosis

```bash
# Check error types
docker compose logs --tail 500 ai-service | grep -E "ERROR|Exception" | sort | uniq -c

# Check specific error messages
docker compose logs --tail 500 ai-service | grep -A5 "Traceback"

# Check if specific operations failing
curl -s http://localhost:3000/metrics | grep 'ringrift_ai_requests_total{outcome="error"}'
```

### Common Error Types

| Error        | Cause          | Fix                          |
| ------------ | -------------- | ---------------------------- |
| ValueError   | Invalid input  | Fix input validation         |
| MemoryError  | OOM            | Increase memory limit        |
| TimeoutError | Slow inference | Optimize or increase timeout |
| ModelError   | Model issue    | Check model configuration    |

### Mitigation

```bash
# Restart AI service
docker compose restart ai-service

# Monitor error rate
watch -n 30 'docker compose logs --tail 50 ai-service | grep -c ERROR'
```

---

## AI Service Operations Reference

### Health Check Commands

```bash
# Full AI service health check
curl -s http://localhost:8001/health | jq

# Check model status
curl -s http://localhost:8001/model/status | jq  # If endpoint exists

# Check AI service metrics
curl -s http://localhost:8001/metrics  # If Prometheus metrics exposed
```

### Log Analysis

```bash
# Recent errors
docker compose logs --tail 500 ai-service 2>&1 | grep -E "ERROR|WARN" | tail -20

# Startup issues
docker compose logs ai-service 2>&1 | head -100

# Model loading
docker compose logs ai-service | grep -i "load\|model\|ready"

# Request patterns
docker compose logs ai-service | grep -i "request\|predict" | tail -50
```

### Resource Management

```bash
# Check AI service resources
docker stats --no-stream ringrift-ai-service-1

# Check memory specifically (models use lots of memory)
docker compose exec ai-service cat /proc/meminfo | head -5

# Check for GPU (if applicable)
docker compose exec ai-service nvidia-smi  # If GPU enabled
```

### Restart Procedures

```bash
# Quick restart
docker compose restart ai-service

# Full recreate
docker compose stop ai-service
docker compose rm -f ai-service
docker compose up -d ai-service

# Monitor startup
docker compose logs -f ai-service

# Verify health after restart (allow 60s for model loading)
sleep 60 && curl -s http://localhost:8001/health | jq
```

---

## Fallback Behavior

When AI service is unavailable or fails:

1. **AIEngine** detects failure (timeout, error, no response)
2. **Local heuristic** is used instead
3. **Move is returned** with `source: "fallback"` indicator
4. **Metric** `ringrift_ai_fallback_total` is incremented

### Testing Fallback

```bash
# Simulate AI service down (for testing)
docker compose stop ai-service

# Create a game against AI and verify it works
# Moves should come from fallback

# Restore AI service
docker compose start ai-service
```

---

## Related Documentation

- [Initial Triage](TRIAGE_GUIDE.md)
- [Availability Incidents](AVAILABILITY.md)
- [Latency Incidents](LATENCY.md)
- [AI Architecture](../architecture/AI_ARCHITECTURE.md)
- [Alerting Thresholds](../operations/ALERTING_THRESHOLDS.md)
