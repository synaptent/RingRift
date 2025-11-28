# RingRift Alerting Thresholds Documentation

> **Doc Status (2025-11-27): Active**  
> Canonical reference for Prometheus/Alertmanager alert rules, thresholds, and response procedures. This describes current monitoring configuration (as implemented in `monitoring/prometheus/alerts.yml`), not rules or lifecycle semantics.

This document describes all alerting rules configured for RingRift, their thresholds, rationale, and response procedures.

## Table of Contents

- [Overview](#overview)
- [Severity Levels](#severity-levels)
- [Alert Categories](#alert-categories)
  - [Availability Alerts](#availability-alerts)
  - [Latency Alerts](#latency-alerts)
  - [Resource Alerts](#resource-alerts)
  - [Business Metric Alerts](#business-metric-alerts)
  - [AI Service Alerts](#ai-service-alerts)
  - [Degradation Alerts](#degradation-alerts)
  - [Rate Limiting Alerts](#rate-limiting-alerts)
  - [Rules Parity Alerts](#rules-parity-alerts)
  - [Service Response Time Alerts](#service-response-time-alerts)
- [Threshold Rationale Summary](#threshold-rationale-summary)
- [Escalation Procedures](#escalation-procedures)
- [Tuning Guidelines](#tuning-guidelines)

---

## Overview

RingRift uses Prometheus for metrics collection and alerting. Alerts are defined in [`monitoring/prometheus/alerts.yml`](../monitoring/prometheus/alerts.yml) and routed through Alertmanager ([`monitoring/alertmanager/alertmanager.yml`](../monitoring/alertmanager/alertmanager.yml)).

### Metrics Endpoint

The application exposes metrics at `http://localhost:3000/metrics` in Prometheus format. Key metrics are collected by [`MetricsService`](../src/server/services/MetricsService.ts).

### Alert Flow

```
Application → MetricsService → /metrics endpoint → Prometheus (scrape)
                                                         ↓
                                                  Alert Rules
                                                         ↓
                                                  Alertmanager
                                                         ↓
                                          Notifications (Slack/Email/PagerDuty)
```

---

## Severity Levels

| Severity     | Response Time            | Description                                           |
| ------------ | ------------------------ | ----------------------------------------------------- |
| **critical** | Immediate (< 15 min)     | Service outage or severe degradation. Wake people up. |
| **warning**  | Business hours (< 4 hrs) | Degraded experience, investigation needed.            |
| **info**     | Next business day        | Informational, no immediate action required.          |

---

## Alert Categories

### Availability Alerts

These alerts detect service outages and critical failures.

#### DatabaseDown

| Property      | Value                                              |
| ------------- | -------------------------------------------------- |
| **Severity**  | critical                                           |
| **Threshold** | `ringrift_service_status{service="database"} == 0` |
| **Duration**  | 1 minute                                           |
| **Impact**    | All data operations unavailable                    |

**Rationale**: Database is the most critical dependency. A 1-minute threshold balances quick detection with avoiding alert noise from brief network blips.

**Response**:

1. Check database container status: `docker ps | grep postgres`
2. Check database logs: `docker logs ringrift-postgres-1`
3. Verify network connectivity from app container
4. Check disk space on database volume
5. If crash, restart: `docker-compose restart postgres`

---

#### RedisDown

| Property      | Value                                               |
| ------------- | --------------------------------------------------- |
| **Severity**  | critical                                            |
| **Threshold** | `ringrift_service_status{service="redis"} == 0`     |
| **Duration**  | 1 minute                                            |
| **Impact**    | Rate limiting disabled, session management degraded |

**Rationale**: Redis is critical for rate limiting and caching. Without it, the application becomes vulnerable to abuse.

**Response**:

1. Check Redis container status
2. Verify memory usage (Redis can OOM)
3. Check AOF/RDB persistence status
4. Restart if necessary: `docker-compose restart redis`

---

#### AIServiceDown

| Property      | Value                                                |
| ------------- | ---------------------------------------------------- |
| **Severity**  | warning                                              |
| **Threshold** | `ringrift_service_status{service="ai_service"} == 0` |
| **Duration**  | 2 minutes                                            |
| **Impact**    | AI moves use local fallback heuristics               |

**Rationale**: AI service has graceful fallback, so this is warning not critical. 2-minute duration allows for container restarts.

**Response**:

1. Check AI service container: `docker ps | grep ai-service`
2. Check logs: `docker logs ringrift-ai-service-1`
3. Verify Python environment is healthy
4. Check model loading status
5. Restart: `docker-compose restart ai-service`

---

#### HighErrorRate

| Property      | Value                          |
| ------------- | ------------------------------ |
| **Severity**  | critical                       |
| **Threshold** | > 5% of requests returning 5xx |
| **Duration**  | 5 minutes                      |
| **Impact**    | Users experiencing failures    |

**Rationale**: 5% error rate over 5 minutes indicates a serious issue. This is an industry-standard SLO threshold.

**Response**:

1. Check application logs for errors
2. Identify failing endpoints via metrics
3. Check recent deployments
4. Verify all dependencies are healthy
5. Consider rollback if deployment-related

---

#### ElevatedErrorRate

| Property      | Value                            |
| ------------- | -------------------------------- |
| **Severity**  | warning                          |
| **Threshold** | > 1% of requests returning 5xx   |
| **Duration**  | 10 minutes                       |
| **Impact**    | Some users experiencing failures |

**Rationale**: 1% sustained error rate warrants investigation before it escalates.

---

#### NoHTTPTraffic

| Property      | Value                                     |
| ------------- | ----------------------------------------- |
| **Severity**  | warning                                   |
| **Threshold** | `sum(rate(http_requests_total[5m])) == 0` |
| **Duration**  | 10 minutes                                |
| **Impact**    | Service may be unreachable                |

**Rationale**: No HTTP traffic for 10 minutes indicates the service may be down or unreachable from clients.

**Response**:

1. Verify the application is running: `docker compose ps`
2. Check network connectivity and loadbalancer health
3. Verify DNS resolution
4. Check firewall rules
5. Review application logs for startup errors

---

### Latency Alerts

These alerts detect response time degradation.

#### HighP99Latency

| Property      | Value                                   |
| ------------- | --------------------------------------- |
| **Severity**  | warning                                 |
| **Threshold** | P99 > 2 seconds                         |
| **Duration**  | 5 minutes                               |
| **Impact**    | 1% of users experiencing slow responses |

**Rationale**: 2-second P99 is acceptable for most operations but indicates performance issues that should be investigated.

---

#### HighP99LatencyCritical

| Property      | Value                             |
| ------------- | --------------------------------- |
| **Severity**  | critical                          |
| **Threshold** | P99 > 5 seconds                   |
| **Duration**  | 2 minutes                         |
| **Impact**    | 1% of users experiencing timeouts |

**Rationale**: 5-second latency approaches browser timeout thresholds and indicates severe issues.

**Response**:

1. Check event loop lag (may indicate blocking code)
2. Check database query performance
3. Check Redis performance
4. Review recent code changes
5. Check for memory pressure / GC issues

---

#### HighP95Latency

| Property      | Value                                   |
| ------------- | --------------------------------------- |
| **Severity**  | warning                                 |
| **Threshold** | P95 > 1 second                          |
| **Duration**  | 10 minutes                              |
| **Impact**    | 5% of users experiencing slow responses |

---

#### HighMedianLatency

| Property      | Value                                    |
| ------------- | ---------------------------------------- |
| **Severity**  | warning                                  |
| **Threshold** | P50 > 500ms                              |
| **Duration**  | 15 minutes                               |
| **Impact**    | 50% of users experiencing slow responses |

**Rationale**: If the median is slow, most users are affected. This indicates systemic performance issues.

---

#### HighGameMoveLatency

| Property      | Value                           |
| ------------- | ------------------------------- |
| **Severity**  | warning                         |
| **Threshold** | P99 > 1 second (per board type) |
| **Duration**  | 5 minutes                       |
| **Impact**    | Game responsiveness degraded    |

**Rationale**: Game moves should feel instant. 1-second processing time is too slow for good UX.

---

### Resource Alerts

These alerts detect resource exhaustion.

#### HighMemoryUsage

| Property      | Value                                   |
| ------------- | --------------------------------------- |
| **Severity**  | warning                                 |
| **Threshold** | > 1.5 GB                                |
| **Duration**  | 10 minutes                              |
| **Impact**    | Risk of OOM if memory continues to grow |

**Rationale**: 1.5GB is ~75% of typical 2GB container limit. Provides warning before OOM.

---

#### HighMemoryUsageCritical

| Property      | Value             |
| ------------- | ----------------- |
| **Severity**  | critical          |
| **Threshold** | > 2 GB            |
| **Duration**  | 5 minutes         |
| **Impact**    | OOM kill imminent |

**Rationale**: At 2GB, the container is at or near its limit. Immediate action required.

**Response**:

1. Check for memory leaks (heap dump if possible)
2. Identify memory-intensive operations
3. Consider restarting the application
4. Review recent code changes

---

#### HighEventLoopLag

| Property      | Value                        |
| ------------- | ---------------------------- |
| **Severity**  | warning                      |
| **Threshold** | > 100ms                      |
| **Duration**  | 5 minutes                    |
| **Impact**    | All async operations delayed |

**Rationale**: 100ms event loop lag indicates the main thread is being blocked, which is bad in Node.js.

---

#### HighEventLoopLagCritical

| Property      | Value                                |
| ------------- | ------------------------------------ |
| **Severity**  | critical                             |
| **Threshold** | > 500ms                              |
| **Duration**  | 2 minutes                            |
| **Impact**    | Application effectively unresponsive |

**Rationale**: 500ms lag means the event loop is severely blocked. The application cannot respond to requests properly.

**Response**:

1. Check for synchronous CPU-intensive operations
2. Look for blocking I/O or large JSON parsing
3. Check for infinite loops or recursive calls
4. Take CPU profile if possible
5. Restart as immediate mitigation

---

#### HighActiveHandles

| Property      | Value                   |
| ------------- | ----------------------- |
| **Severity**  | warning                 |
| **Threshold** | > 10,000 active handles |
| **Duration**  | 10 minutes              |
| **Impact**    | Potential resource leak |

**Rationale**: More than 10,000 active Node.js handles may indicate a resource leak (unclosed connections, file handles, timers).

**Response**:

1. Check for connection leaks (database, WebSocket)
2. Verify file handles are being closed
3. Review timer/interval usage
4. Take heap dump for analysis
5. Restart as immediate mitigation

---

### Business Metric Alerts

These alerts track game-specific business metrics.

#### NoActiveGames

| Property      | Value                                               |
| ------------- | --------------------------------------------------- |
| **Severity**  | info                                                |
| **Threshold** | `ringrift_games_active == 0`                        |
| **Duration**  | 30 minutes                                          |
| **Impact**    | None if during off-peak, investigate if during peak |

**Rationale**: This is informational. Zero games during 3 AM is normal; zero games during peak hours is unusual.

---

#### NoWebSocketConnections

| Property      | Value                                 |
| ------------- | ------------------------------------- |
| **Severity**  | warning                               |
| **Threshold** | `ringrift_websocket_connections == 0` |
| **Duration**  | 15 minutes                            |
| **Impact**    | Real-time features unavailable        |

**Rationale**: If the service is running but has zero WebSocket connections, there may be a connectivity issue.

---

#### LongRunningGames

| Property      | Value                              |
| ------------- | ---------------------------------- |
| **Severity**  | info                               |
| **Threshold** | Median game duration > 1 hour      |
| **Duration**  | 30 minutes                         |
| **Impact**    | Possible stalled or orphaned games |

**Rationale**: Games running unusually long may indicate stalled game states or unusual gameplay patterns.

**Response**:

1. Check for stalled game sessions
2. Review game state for stuck players
3. Verify AI service is responding
4. Consider implementing game timeout cleanup
5. Monitor for resource consumption from orphaned games

---

#### HighWebSocketConnections

| Property      | Value                          |
| ------------- | ------------------------------ |
| **Severity**  | warning                        |
| **Threshold** | > 1000 connections             |
| **Duration**  | 5 minutes                      |
| **Impact**    | May approach connection limits |

**Rationale**: High connection count may indicate scaling needs or connection leaks.

---

### AI Service Alerts

These alerts track AI service health and quality.

#### AIFallbackRateHigh

| Property      | Value                    |
| ------------- | ------------------------ |
| **Severity**  | warning                  |
| **Threshold** | > 30%                    |
| **Duration**  | 10 minutes               |
| **Impact**    | AI move quality degraded |

**Rationale**: 30% fallback rate means nearly 1/3 of AI games are using heuristic fallback. Quality is noticeably degraded.

---

#### AIFallbackRateCritical

| Property      | Value                        |
| ------------- | ---------------------------- |
| **Severity**  | critical                     |
| **Threshold** | > 50%                        |
| **Duration**  | 5 minutes                    |
| **Impact**    | Most AI games using fallback |

**Rationale**: At 50% fallback, the AI service is effectively broken.

**Response**:

1. Check AI service health
2. Verify model is loaded correctly
3. Check for timeout issues between app and AI service
4. Review AI service logs for errors
5. Restart AI service

---

#### AIRequestHighLatency

| Property      | Value            |
| ------------- | ---------------- |
| **Severity**  | warning          |
| **Threshold** | P99 > 5 seconds  |
| **Duration**  | 5 minutes        |
| **Impact**    | AI moves delayed |

---

#### AIErrorsIncreasing

| Property      | Value                       |
| ------------- | --------------------------- |
| **Severity**  | warning                     |
| **Threshold** | Error rate > 0.1/sec        |
| **Duration**  | 5 minutes                   |
| **Impact**    | AI game experience degraded |

---

### Degradation Alerts

These alerts track the application's graceful degradation status.

#### ServiceDegraded

| Property      | Value                            |
| ------------- | -------------------------------- |
| **Severity**  | warning                          |
| **Threshold** | `ringrift_degradation_level > 0` |
| **Duration**  | 5 minutes                        |
| **Impact**    | Some features unavailable        |

**Degradation Levels**:

- 0 = FULL: All features available
- 1 = DEGRADED: Some features unavailable
- 2 = MINIMAL: Only core features available
- 3 = OFFLINE: Service unavailable

---

#### ServiceMinimalMode

| Property      | Value                             |
| ------------- | --------------------------------- |
| **Severity**  | critical                          |
| **Threshold** | `ringrift_degradation_level >= 2` |
| **Duration**  | 1 minute                          |
| **Impact**    | Most features unavailable         |

---

#### ServiceOffline

| Property      | Value                             |
| ------------- | --------------------------------- |
| **Severity**  | critical                          |
| **Threshold** | `ringrift_degradation_level == 3` |
| **Duration**  | 30 seconds                        |
| **Impact**    | Complete service outage           |

**Response**:

1. Check all dependent services (DB, Redis, AI)
2. Check application logs for cause
3. Identify which service failure triggered offline mode
4. Restore failed services
5. Application should auto-recover

---

### Rate Limiting Alerts

These alerts track rate limiting activity.

#### HighRateLimitHits

| Property      | Value                   |
| ------------- | ----------------------- |
| **Severity**  | warning                 |
| **Threshold** | > 1/sec per endpoint    |
| **Duration**  | 5 minutes               |
| **Impact**    | Some users rate limited |

**Response**:

1. Identify affected endpoints
2. Check for abuse patterns (same IP, user agent)
3. Review rate limit thresholds
4. Consider IP blocking if abuse detected

---

#### SustainedRateLimiting

| Property      | Value                   |
| ------------- | ----------------------- |
| **Severity**  | warning                 |
| **Threshold** | > 10/sec total          |
| **Duration**  | 15 minutes              |
| **Impact**    | Multiple users affected |

---

### Rules Parity Alerts

These alerts track consistency between TypeScript and Python rules engines.

#### RulesParityValidationMismatch

| Property      | Value                              |
| ------------- | ---------------------------------- |
| **Severity**  | warning                            |
| **Threshold** | > 5 mismatches/hour                |
| **Duration**  | 5 minutes                          |
| **Impact**    | Rule engine inconsistency detected |

---

#### RulesParityHashMismatch

| Property      | Value                                  |
| ------------- | -------------------------------------- |
| **Severity**  | warning                                |
| **Threshold** | > 5 mismatches/hour                    |
| **Duration**  | 5 minutes                              |
| **Impact**    | Game state may diverge between engines |

---

#### RulesParityGameStatusMismatch

| Property      | Value                                    |
| ------------- | ---------------------------------------- |
| **Severity**  | critical                                 |
| **Threshold** | > 0 mismatches/hour                      |
| **Duration**  | 5 minutes                                |
| **Impact**    | Win/loss outcomes differ between engines |

**Response**:

1. This is a critical game integrity issue
2. Identify the affected game(s) via logs
3. Capture game state for analysis
4. Compare TypeScript vs Python rule evaluation
5. File bug report with reproduction steps

---

### Service Response Time Alerts

These alerts track database and cache response times.

#### DatabaseResponseTimeSlow

| Property      | Value                                           |
| ------------- | ----------------------------------------------- |
| **Severity**  | warning                                         |
| **Threshold** | P99 > 500ms                                     |
| **Duration**  | 5 minutes                                       |
| **Impact**    | Slow database queries affecting request latency |

**Rationale**: Database query latency above 500ms at the 99th percentile indicates query performance issues or database overload.

**Response**:

1. Check database connection pool utilization
2. Look for slow queries using `pg_stat_statements`
3. Check for missing indexes
4. Verify database resource usage (CPU, memory, disk I/O)
5. Consider query optimization or connection pool tuning

---

#### RedisResponseTimeSlow

| Property      | Value                                            |
| ------------- | ------------------------------------------------ |
| **Severity**  | warning                                          |
| **Threshold** | P99 > 100ms                                      |
| **Duration**  | 5 minutes                                        |
| **Impact**    | Cache operations slow, affecting overall latency |

**Rationale**: Redis operations should be sub-millisecond. P99 above 100ms indicates network issues or Redis overload.

**Response**:

1. Check Redis memory usage
2. Verify network latency between app and Redis
3. Look for large key operations or blocking commands
4. Check Redis client connection count
5. Consider Redis memory optimization or scaling

---

## Threshold Rationale Summary

| Metric            | Warning | Critical | Rationale                           |
| ----------------- | ------- | -------- | ----------------------------------- |
| Error Rate (5xx)  | 1%      | 5%       | Industry standard SLO thresholds    |
| P99 Latency       | 2s      | 5s       | Game responsiveness expectations    |
| P95 Latency       | 1s      | -        | Most users affected at this level   |
| P50 Latency       | 500ms   | -        | Median reflects overall health      |
| Memory Usage      | 1.5GB   | 2GB      | 75%/100% of typical container limit |
| Event Loop Lag    | 100ms   | 500ms    | Node.js responsiveness thresholds   |
| AI Fallback Rate  | 30%     | 50%      | Quality degradation indicator       |
| Degradation Level | >0      | ≥2       | Feature availability stages         |
| Database Down     | -       | 1m       | Critical dependency                 |
| Redis Down        | -       | 1m       | Critical for rate limiting          |
| AI Service Down   | 2m      | -        | Has graceful fallback               |
| Active Handles    | 10K     | -        | Resource leak indicator             |
| Database Response | 500ms   | -        | Query performance threshold         |
| Redis Response    | 100ms   | -        | Cache latency threshold             |

---

## Escalation Procedures

### Critical Alert Escalation

1. **Immediate** (0-5 min):
   - Alert fires to #ringrift-critical Slack channel
   - Email to oncall@ringrift.io
   - PagerDuty notification (if configured)

2. **Acknowledgement** (5-15 min):
   - On-call engineer acknowledges alert
   - Initial triage begins
   - Update incident channel

3. **Escalation** (15-30 min):
   - If unacknowledged, escalate to secondary on-call
   - If unresolved, involve senior engineers

4. **Resolution**:
   - Document root cause
   - Update runbooks if needed
   - Post-mortem for P0/P1 incidents

### Warning Alert Escalation

1. Alert fires to team-specific Slack channel
2. Engineer investigates during business hours
3. If persists > 4 hours, escalate to critical

### Info Alert Handling

1. Logged to #ringrift-info channel
2. Reviewed in weekly operations meeting
3. Used for capacity planning and trending

---

## Tuning Guidelines

### When to Adjust Thresholds

1. **Too Many Alerts (Alert Fatigue)**:
   - Review false positive rate
   - Consider relaxing thresholds
   - Increase `for` duration

2. **Missing Real Issues**:
   - Tighten thresholds
   - Reduce `for` duration
   - Add new alert rules

3. **Seasonal Variations**:
   - Consider time-based thresholds
   - Adjust for expected traffic patterns

### Baseline Metrics

After deployment, establish baselines for:

- Normal error rate (should be < 0.1%)
- Normal P99 latency (varies by endpoint)
- Normal memory usage over 24h
- Normal game/connection counts

### Threshold Adjustment Process

1. Propose change in PR
2. Review with ops team
3. Test in staging (if possible)
4. Deploy to production
5. Monitor for 1 week
6. Adjust if needed

---

## Files Reference

- **Alert Rules**: [`monitoring/prometheus/alerts.yml`](../monitoring/prometheus/alerts.yml)
- **Prometheus Config**: [`monitoring/prometheus/prometheus.yml`](../monitoring/prometheus/prometheus.yml)
- **Alertmanager Config**: [`monitoring/alertmanager/alertmanager.yml`](../monitoring/alertmanager/alertmanager.yml)
- **Metrics Service**: [`src/server/services/MetricsService.ts`](../src/server/services/MetricsService.ts)

---

## Related Documentation

- [Operations Database Guide](./OPERATIONS_DB.md)
- [Security Threat Model](./SECURITY_THREAT_MODEL.md)
- [Incident Response](./INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md)
