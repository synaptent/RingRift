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
  - [Orchestrator Rollout Alerts](#orchestrator-rollout-alerts)
  - [Rules Parity Alerts](#rules-parity-alerts)
  - [Service Response Time Alerts](#service-response-time-alerts)
- [Threshold Rationale Summary](#threshold-rationale-summary)
- [Dashboards & Observability Checklist](#dashboards--observability-checklist)
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

### Orchestrator Rollout Alerts

These alerts track orchestrator-specific health, rollout posture, and invariants. They are the on-call surface for the orchestrator SLOs defined in
[`ORCHESTRATOR_ROLLOUT_PLAN.md`](./ORCHESTRATOR_ROLLOUT_PLAN.md#6-orchestrator-specific-slos).

#### OrchestratorCircuitBreakerOpen

| Property      | Value                                                    |
| ------------- | -------------------------------------------------------- |
| **Severity**  | critical                                                 |
| **Threshold** | `ringrift_orchestrator_circuit_breaker_state == 1`       |
| **Duration**  | 30 seconds                                               |
| **Impact**    | Orchestrator path disabled; all traffic on legacy engine |

**Rationale**: A tripped circuit breaker means orchestrator error rate exceeded the configured threshold (default 5% over a 5‑minute window). This is a direct breach of `SLO-STAGE-ORCH-ERROR` / `SLO-PROD-ORCH-ERROR`.

**Response**:

1. Check orchestrator error logs and recent deploys (backend `GameEngine` / turn adapter changes).
2. Verify Python rules and AI services are healthy to rule out dependency cascades.
3. Follow `docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`:
   - Freeze rollout (do not increase `ORCHESTRATOR_ROLLOUT_PERCENTAGE`).
   - Investigate and fix the underlying error pattern.
   - Manually reset the circuit breaker via the admin APIs or config once resolved.

---

#### OrchestratorErrorRateWarning

| Property      | Value                                          |
| ------------- | ---------------------------------------------- |
| **Severity**  | warning                                        |
| **Threshold** | `ringrift_orchestrator_error_rate > 0.02` (2%) |
| **Duration**  | 2 minutes                                      |
| **Impact**    | Elevated orchestrator-specific failure rate    |

**Rationale**: At 2% orchestrator error rate, the system is approaching the 5% circuit‑breaker threshold. This is an early warning aligned with the orchestrator error‑rate SLOs.

**Response**:

1. Check `/metrics` and logs for specific orchestrator failure types (validation errors, timeouts, invariant violations).
2. Correlate with recent code changes or rollout percentage changes.
3. If errors are increasing, pre‑emptively pause rollout and reduce `ORCHESTRATOR_ROLLOUT_PERCENTAGE` in staging/production as per the rollout runbook.

---

#### OrchestratorShadowMismatches

| Property      | Value                                                    |
| ------------- | -------------------------------------------------------- |
| **Severity**  | warning                                                  |
| **Threshold** | `ringrift_orchestrator_shadow_mismatch_rate > 0.01` (1%) |
| **Duration**  | 5 minutes                                                |
| **Impact**    | Possible semantic divergence in shadow mode              |

**Rationale**: When running with `ORCHESTRATOR_SHADOW_MODE_ENABLED=true`, more than 1% mismatches between orchestrator and legacy paths indicates a likely rules or host‑integration bug. This maps to `SLO-STAGE-ORCH-PARITY` / `SLO-PROD-ORCH-PARITY`.

**Response**:

1. Use the shadow comparison dashboards (backed by `ringrift_orchestrator_shadow_mismatch_rate` and related gauges) to inspect mismatch patterns by board type and phase.
2. Correlate with rules parity alerts (`RulesParity*`) to distinguish orchestrator vs Python divergence.
3. Hold rollout at current percentage or revert to a safer phase until mismatches are understood and fixed.

---

#### OrchestratorInvariantViolationsStaging

| Property      | Value                                                                                 |
| ------------- | ------------------------------------------------------------------------------------- |
| **Severity**  | warning                                                                               |
| **Threshold** | `increase(ringrift_orchestrator_invariant_violations_total[1h]) > 0`                  |
| **Duration**  | 5 minutes                                                                             |
| **Impact**    | Invariant violations under CI or staging traffic                                      |
| **Labels**    | `environment`, `type`, `invariant_id` (see `docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`) |

**Rationale**: Any new orchestrator invariant violation (ACTIVE-no-move, S‑invariant decrease, elimination accounting regression) in staging or CI is treated as a staging SLO breach (`SLO-STAGE-ORCH-INVARIANTS`).

**Response**:

1. Inspect invariant violation logs and orchestrator soak summaries (see `docs/STRICT_INVARIANT_SOAKS.md` and `results/orchestrator_soak_summary.json` artifacts).
2. Promote the failing seed/geometry into a dedicated Jest regression under `tests/unit/OrchestratorSInvariant.regression.test.ts` where applicable.
3. Block promotion to production until the violation is fixed or explicitly waived with documented rationale.

---

#### OrchestratorInvariantViolationsProduction

| Property      | Value                                                                                 |
| ------------- | ------------------------------------------------------------------------------------- |
| **Severity**  | critical                                                                              |
| **Threshold** | `increase(ringrift_orchestrator_invariant_violations_total[1h]) > 0`                  |
| **Duration**  | 5 minutes                                                                             |
| **Impact**    | Invariants violated under real player traffic                                         |
| **Labels**    | `environment`, `type`, `invariant_id` (see `docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`) |

**Rationale**: Any orchestrator invariant violation in production is a direct breach of `SLO-PROD-ORCH-INVARIANTS` and must halt rollout.

**Response**:

1. Immediately freeze orchestrator rollout and consider reducing `ORCHESTRATOR_ROLLOUT_PERCENTAGE` or disabling the adapter per the rollout runbook.
2. Triage the violation using logs and, if available, invariant soak results against the production image.
3. Add or update regression tests and only resume rollout once the issue is fully understood and fixed.

---

### Connection & Session Lifecycle Alerts

These alerts connect the decision/reconnection lifecycle SSoT in
[`docs/P18.3-1_DECISION_LIFECYCLE_SPEC.md`](./P18.3-1_DECISION_LIFECYCLE_SPEC.md)
to concrete Prometheus signals for WebSocket reconnection, session state, and abnormal termination.

#### WebSocketReconnectionTimeouts

| Property      | Value                                                                           |
| ------------- | ------------------------------------------------------------------------------- |
| **Severity**  | warning                                                                         |
| **Threshold** | `sum(rate(ringrift_websocket_reconnection_total{result="timeout"}[5m])) > 0.05` |
| **Duration**  | 10 minutes                                                                      |
| **Impact**    | Many players are failing to reconnect within the defined window                 |

**Rationale**: Occasional reconnect timeouts are expected (users closing laptops, transient network loss). A sustained rate above ~0.05/sec over 10 minutes (≈30 timeouts) indicates systemic issues with reconnect window handling, network conditions, or client reconnect logic. This corresponds to the reconnect/abandonment semantics in P18.3‑1 §§2.4/4.3 and is exercised by:

- `tests/e2e/reconnection.simulation.test.ts`
- `tests/e2e/decision-phase-timeout.e2e.spec.ts`

**Example PromQL**:

```promql
sum(rate(ringrift_websocket_reconnection_total{result="timeout"}[5m])) > 0.05
```

**Response**:

1. Check WebSocket logs and dashboards for elevated disconnect rates and reconnection latency.
2. Correlate with deployment changes to `WebSocketServer`, GameConnection, and connection state machines.
3. Inspect `ringrift_game_session_status_current{status=~"disconnected_.*"}` and `ringrift_game_session_status_transitions_total` to determine whether reconnect windows are expiring as expected or prematurely.
4. If isolated to a region or environment, consider temporarily increasing the reconnection window and updating runbooks accordingly.

---

#### AbnormalGameSessionTerminationSpike

| Property      | Value                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------- | -------------- | -------------------------------- |
| **Severity**  | warning                                                                                     |
| **Threshold** | `sum(rate(ringrift_game_session_abnormal_termination_total{reason=~"disconnect_timeout      | internal_error | session_cleanup"}[10m])) > 0.02` |
| **Duration**  | 10 minutes                                                                                  |
| **Impact**    | More games ending via abnormal paths (disconnect timeouts, internal errors, forced cleanup) |

**Rationale**: `ringrift_game_session_abnormal_termination_total{reason}` is incremented when `GameSession.terminate(reason)` ends an `active` game for non‑standard reasons (see P18.3‑1 §4.3). A sustained increase indicates players are losing games to infrastructure or lifecycle bugs rather than normal victory conditions. This metric is backed by:

- `tests/unit/GameSession.abnormalTermination.metrics.test.ts`
- `tests/unit/WebSocketServer.sessionTermination.test.ts`

**Example PromQL**:

```promql
sum(rate(ringrift_game_session_abnormal_termination_total{reason=~"disconnect_timeout|internal_error|session_cleanup"}[10m])) > 0.02
```

**Response**:

1. Use Grafana panels over `ringrift_game_session_abnormal_termination_total` broken down by `reason` to identify which abnormal path is spiking.
2. Correlate with AI and database health:
   - `ringrift_ai_requests_total{outcome="error"}`
   - `ringrift_service_status{service=~"database|ai_service"}`
3. Inspect recent logs for `GameSession.terminate()` invocations with the same `reason` and sample game IDs; replay representative games in sandbox or replay tools.
4. If driven by infrastructure issues (DB, AI, or WebSocket outages), follow the relevant runbooks; if driven by logic bugs, add a focused Jest regression anchored to P18.3‑1.

---

#### GameSessionStatusSkew

| Property      | Value                                                                                   |
| ------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Severity**  | info                                                                                    |
| **Threshold** | `sum(ringrift_game_session_status_current{status=~"active_decision_phase                | active_turn"}) == 0 and sum(ringrift_game_session_status_current) > 0` |
| **Duration**  | 15 minutes                                                                              |
| **Impact**    | All in‑memory sessions stuck in non‑active states (e.g. waiting, abandoned, or cleanup) |

**Rationale**: `ringrift_game_session_status_current{status}` provides a coarse projection over the `GameSessionStatus` state machine. If there are live sessions but none in `active_turn` or `active_decision_phase` for an extended period, it may indicate a stuck lifecycle (for example, timers not firing or decisions never resolving). This complements the decision‑phase timeout alerts and is validated by:

- `tests/unit/GameSession.aiDiagnostics.test.ts`
- `tests/unit/WebSocketServer.connectionState.test.ts`

**Example PromQL**:

```promql
sum(ringrift_game_session_status_current{status=~"active_decision_phase|active_turn"}) == 0
and sum(ringrift_game_session_status_current) > 0
```

**Response**:

1. Check WebSocket decision‑phase metrics and logs for pending choices that never resolve.
2. Inspect `decision_phase_timed_out` events and `ringrift_ai_turn_request_terminal_total{kind="timed_out"}` to see if timers are firing but sessions are not progressing.
3. Sample affected game IDs from logs and load them via `/games/:gameId/diagnostics/session` to inspect `sessionStatus` and `lastAIRequestState`.
4. If a systemic state‑machine issue is suspected, roll back recent lifecycle changes and add regression tests under `tests/unit/stateMachines/*.test.ts`.

---

### Python Invariant Alerts (Self‑Play / AI Service)

These alerts track strict‑invariant violations observed by Python self‑play soaks and the AI service, using the same invariant IDs (`INV-*`) as the TS/orchestrator metrics.

#### PythonInvariantViolations

| Property      | Value                                                                  |
| ------------- | ---------------------------------------------------------------------- |
| **Severity**  | warning                                                                |
| **Threshold** | `increase(ringrift_python_invariant_violations_total[1h]) > 0`         |
| **Duration**  | 5 minutes                                                              |
| **Impact**    | Python self‑play invariants violated (training / AI rules regressions) |

**Rationale**: Any strict‑invariant violation in Python self‑play soaks (for example `INV-S-MONOTONIC` or `INV-ACTIVE-NO-MOVES`) indicates potential divergence between Python rules/AI training behaviour and the TS orchestrator invariants. This maps to the Python invariant surfaces described in `INVARIANTS_AND_PARITY_FRAMEWORK.md` and `STRICT_INVARIANT_SOAKS.md`.

**Response**:

1. Inspect the most recent Python soak summaries produced by `run_self_play_soak.py` (including `invariant_violations_by_id` and sample entries).
2. Identify whether the violation corresponds to a known triaged edge case or a new invariant break.
3. Promote representative seeds/snapshots into dedicated TS or Python regression tests, and align invariant semantics if TS vs Python differ.
4. For production‑adjacent training pipelines, pause or roll back to a last‑known‑good rules/AI configuration until the invariant is restored.

---

### Rules Parity Alerts

These alerts track consistency between TypeScript and Python rules engines.

Underlying metrics:

- **Unified counter:** `ringrift_rules_parity_mismatches_total{mismatch_type, suite}`
  - `mismatch_type` ∈ {`validation`, `hash`, `s_invariant`, `game_status`}
  - `suite` identifies the parity context (for example `runtime_shadow`, `runtime_python_mode`, or future contract-vector suites such as `contract_vectors_v2` corresponding to `PARITY-TS-PY-*` IDs in
    [`INVARIANTS_AND_PARITY_FRAMEWORK.md`](./INVARIANTS_AND_PARITY_FRAMEWORK.md).
- **Legacy counters (still exported for dashboards):**
  - `ringrift_rules_parity_valid_mismatch_total`
  - `ringrift_rules_parity_hash_mismatch_total`
  - `ringrift_rules_parity_s_mismatch_total`
  - `ringrift_rules_parity_game_status_mismatch_total`

#### RulesParityValidationMismatch

| Property      | Value                              |
| ------------- | ---------------------------------- |
| **Severity**  | warning                            |
| **Threshold** | > 5 validation mismatches/hour     |
| **Duration**  | 5 minutes                          |
| **Impact**    | Rule engine inconsistency detected |

Backed by:

```promql
increase(ringrift_rules_parity_mismatches_total{mismatch_type="validation"}[1h]) > 5
```

---

#### RulesParityHashMismatch

| Property      | Value                                  |
| ------------- | -------------------------------------- |
| **Severity**  | warning                                |
| **Threshold** | > 5 hash mismatches/hour               |
| **Duration**  | 5 minutes                              |
| **Impact**    | Game state may diverge between engines |

Backed by:

```promql
increase(ringrift_rules_parity_mismatches_total{mismatch_type="hash"}[1h]) > 5
```

---

#### RulesParityGameStatusMismatch

| Property      | Value                                    |
| ------------- | ---------------------------------------- |
| **Severity**  | critical                                 |
| **Threshold** | > 0 game-status mismatches/hour          |
| **Duration**  | 5 minutes                                |
| **Impact**    | Win/loss outcomes differ between engines |

Backed by:

```promql
increase(ringrift_rules_parity_mismatches_total{mismatch_type="game_status"}[1h]) > 0
```

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

## Dashboards & Observability Checklist

This section describes the **minimum recommended dashboards** to keep rules/orchestrator, AI, and infra health observable in staging and production. It complements the alert rules in this document and the orchestrator rollout SLOs in [`ORCHESTRATOR_ROLLOUT_PLAN.md`](./ORCHESTRATOR_ROLLOUT_PLAN.md).

### Rules / Orchestrator Dashboard

Every environment that is on, or preparing for, orchestrator‑first rollout should have a dedicated dashboard that shows at least:

- **Error and invariants**
  - `ringrift_orchestrator_error_rate{environment=...}`
  - `ringrift_orchestrator_invariant_violations_total{environment=...}` (where present, sliced by `type` and `invariant_id` matching `INV-*` IDs from
    [`INVARIANTS_AND_PARITY_FRAMEWORK.md`](./INVARIANTS_AND_PARITY_FRAMEWORK.md)).
  - (For AI/self‑play environments) `ringrift_python_invariant_violations_total{invariant_id=...,type=...}` for Python strict‑invariant soaks (`INV-*` IDs).
  - `OrchestratorErrorRateWarning`, `OrchestratorInvariantViolations*`, and `PythonInvariantViolations` alert state.
- **Rollout posture**
  - `ringrift_orchestrator_rollout_percentage{environment=...}`
  - `ringrift_orchestrator_circuit_breaker_state{environment=...}`
  - Shadow mismatch metrics:
    - `ringrift_orchestrator_shadow_mismatch_rate{environment=...}`
    - Comparison counters used by `OrchestratorShadowMismatches`.
- **Game health**
  - Game move latency histograms (P50/P95/P99) broken down by board type or route.
  - HTTP 5xx share on game endpoints (as used in `HighErrorRate` alerts).
  - Rules parity metrics and `RulesParity*` alert state, backed by `ringrift_rules_parity_mismatches_total{mismatch_type, suite}` where suites map onto
    parity contexts (for example `runtime_shadow`, `runtime_python_mode`, or future `PARITY-*` contract suites).

For Phase‑based expectations and thresholds, see §§6.2–6.4 and §8 of `ORCHESTRATOR_ROLLOUT_PLAN.md`.

### AI Service Dashboard

To distinguish AI incidents from rules/orchestrator issues (see `AI_ARCHITECTURE.md` and the AI runbooks), maintain a separate AI‑focused dashboard with:

- AI service availability and error rate:
  - `ringrift_service_status{service="ai_service"}`
  - HTTP 5xx/timeout fraction for AI endpoints.
- Latency:
  - AI request latency histograms (P50/P95/P99).
  - Any model‑specific latency or queue‑depth metrics.
- Fallback behaviour:
  - AI fallback rate (as used in AI degradation alerts).
  - Counts of degraded moves vs fully‑powered AI moves.

This dashboard should be the primary reference during `AIServiceDown`, `AI*Degraded`, and AI‑performance incidents.

### Infrastructure / Platform Dashboard

Finally, maintain a core infra dashboard that surfaces:

- Container and node health:
  - CPU, memory, and disk utilisation for app, database, and Redis.
  - `HighMemoryUsage*`, `HighEventLoopLag*`, and `HighActiveHandles` alert state.
- Dependency health:
  - `ringrift_service_status{service="database"}` and `{service="redis"}`.
  - Database and Redis response‑time summaries corresponding to their alerts.
- Traffic and saturation:
  - HTTP traffic volume (`http_requests_total`) per route.
  - Connection counts (WebSockets, DB connections, Redis clients).

During incident reviews, verify that these three dashboards together make it easy to answer:

1. Is the problem **rules/orchestrator**, **AI**, or **infra**?
2. Are current alert thresholds still appropriate for observed load and variability?
3. Does the current orchestrator rollout phase (from `ORCHESTRATOR_ROLLOUT_PLAN.md`) match the live flag and metric posture?

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
