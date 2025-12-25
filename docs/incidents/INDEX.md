# RingRift Incident Response Guide

This directory contains incident response procedures for RingRift. Each guide is aligned with alerting rules defined in [`monitoring/prometheus/alerts.yml`](../../monitoring/prometheus/alerts.yml).

## Quick Reference: Alert to Response Mapping

| Alert                         | Severity | Response Guide                                                                                       | Response Time |
| ----------------------------- | -------- | ---------------------------------------------------------------------------------------------------- | ------------- |
| **Availability**              |          |                                                                                                      |               |
| DatabaseDown                  | Critical | [AVAILABILITY.md#databasedown](AVAILABILITY.md#alert-databasedown)                                   | Immediate     |
| RedisDown                     | Critical | [AVAILABILITY.md#redisdown](AVAILABILITY.md#alert-redisdown)                                         | Immediate     |
| AIServiceDown                 | Warning  | [AI_SERVICE.md#aiservicedown](AI_SERVICE.md#alert-aiservicedown)                                     | 1 hour        |
| HighErrorRate                 | Critical | [AVAILABILITY.md#higherrorrate](AVAILABILITY.md#alert-higherrorrate)                                 | Immediate     |
| ElevatedErrorRate             | Warning  | [AVAILABILITY.md#elevatederrorrate](AVAILABILITY.md#alert-elevatederrorrate)                         | 4 hours       |
| NoHTTPTraffic                 | Warning  | [AVAILABILITY.md#nohttptraffic](AVAILABILITY.md#alert-nohttptraffic)                                 | 1 hour        |
| **Latency**                   |          |                                                                                                      |               |
| HighP99Latency                | Warning  | [LATENCY.md#highp99latency](LATENCY.md#alert-highp99latency)                                         | 4 hours       |
| HighP99LatencyCritical        | Critical | [LATENCY.md#highp99latencycritical](LATENCY.md#alert-highp99latencycritical)                         | Immediate     |
| HighP95Latency                | Warning  | [LATENCY.md#highp95latency](LATENCY.md#alert-highp95latency)                                         | 4 hours       |
| HighMedianLatency             | Warning  | [LATENCY.md#highmedianlatency](LATENCY.md#alert-highmedianlatency)                                   | 4 hours       |
| HighGameMoveLatency           | Warning  | [LATENCY.md#highgamemovelatency](LATENCY.md#alert-highgamemovelatency)                               | 4 hours       |
| **Resources**                 |          |                                                                                                      |               |
| HighMemoryUsage               | Warning  | [RESOURCES.md#highmemoryusage](RESOURCES.md#alert-highmemoryusage)                                   | 4 hours       |
| HighMemoryUsageCritical       | Critical | [RESOURCES.md#highmemoryusagecritical](RESOURCES.md#alert-highmemoryusagecritical)                   | Immediate     |
| HighEventLoopLag              | Warning  | [RESOURCES.md#higheventlooplag](RESOURCES.md#alert-higheventlooplag)                                 | 4 hours       |
| HighEventLoopLagCritical      | Critical | [RESOURCES.md#higheventlooplagcritical](RESOURCES.md#alert-higheventlooplagcritical)                 | Immediate     |
| HighActiveHandles             | Warning  | [RESOURCES.md#highactivehandles](RESOURCES.md#alert-highactivehandles)                               | 4 hours       |
| **AI Service**                |          |                                                                                                      |               |
| AIFallbackRateHigh            | Warning  | [AI_SERVICE.md#aifallbackratehigh](AI_SERVICE.md#alert-aifallbackratehigh)                           | 4 hours       |
| AIFallbackRateCritical        | Critical | [AI_SERVICE.md#aifallbackratecritical](AI_SERVICE.md#alert-aifallbackratecritical)                   | 1 hour        |
| AIRequestHighLatency          | Warning  | [AI_SERVICE.md#airequesthighlatency](AI_SERVICE.md#alert-airequesthighlatency)                       | 4 hours       |
| AIErrorsIncreasing            | Warning  | [AI_SERVICE.md#aierrorsincreasing](AI_SERVICE.md#alert-aierrorsincreasing)                           | 4 hours       |
| **Degradation**               |          |                                                                                                      |               |
| ServiceDegraded               | Warning  | [AVAILABILITY.md#servicedegraded](AVAILABILITY.md#alert-servicedegraded)                             | 4 hours       |
| ServiceMinimalMode            | Critical | [AVAILABILITY.md#serviceminimalmode](AVAILABILITY.md#alert-serviceminimalmode)                       | Immediate     |
| ServiceOffline                | Critical | [AVAILABILITY.md#serviceoffline](AVAILABILITY.md#alert-serviceoffline)                               | Immediate     |
| **Security/Rate Limiting**    |          |                                                                                                      |               |
| HighRateLimitHits             | Warning  | [SECURITY.md#highratelimithits](SECURITY.md#alert-highratelimithits)                                 | 4 hours       |
| SustainedRateLimiting         | Warning  | [SECURITY.md#sustainedratelimiting](SECURITY.md#alert-sustainedratelimiting)                         | 4 hours       |
| **Rules Parity**              |          |                                                                                                      |               |
| RulesParityValidationMismatch | Warning  | [AVAILABILITY.md#rulesparityvalidationmismatch](AVAILABILITY.md#alert-rulesparityvalidationmismatch) | 4 hours       |
| RulesParityHashMismatch       | Warning  | [AVAILABILITY.md#rulesparityhashmismatch](AVAILABILITY.md#alert-rulesparityhashmismatch)             | 4 hours       |
| RulesParityGameStatusMismatch | Critical | [AVAILABILITY.md#rulesparitygamestatusmismatch](AVAILABILITY.md#alert-rulesparitygamestatusmismatch) | Immediate     |
| **Service Response**          |          |                                                                                                      |               |
| DatabaseResponseTimeSlow      | Warning  | [LATENCY.md#databaseresponsetimeslow](LATENCY.md#alert-databaseresponsetimeslow)                     | 4 hours       |
| RedisResponseTimeSlow         | Warning  | [LATENCY.md#redisresponsetimeslow](LATENCY.md#alert-redisresponsetimeslow)                           | 4 hours       |
| **Business Metrics**          |          |                                                                                                      |               |
| NoActiveGames                 | Info     | [AVAILABILITY.md#noactivegames](AVAILABILITY.md#alert-noactivegames)                                 | Next day      |
| NoWebSocketConnections        | Warning  | [AVAILABILITY.md#nowebsocketconnections](AVAILABILITY.md#alert-nowebsocketconnections)               | 4 hours       |
| HighWebSocketConnections      | Warning  | [RESOURCES.md#highwebsocketconnections](RESOURCES.md#alert-highwebsocketconnections)                 | 4 hours       |
| LongRunningGames              | Info     | [AVAILABILITY.md#longrunninggames](AVAILABILITY.md#alert-longrunninggames)                           | Next day      |

---

## Severity Levels

| Level           | Priority          | Response Time | Notification                 | Examples                                         |
| --------------- | ----------------- | ------------- | ---------------------------- | ------------------------------------------------ |
| **P1 Critical** | Immediate         | 15 minutes    | PagerDuty + Slack #incidents | Database down, >5% error rate, service offline   |
| **P2 High**     | Urgent            | 1 hour        | Slack #incidents             | AI service down, high fallback rate              |
| **P3 Medium**   | Business Hours    | 4 hours       | Slack #alerts                | High latency, elevated errors, resource warnings |
| **P4 Low**      | Next Business Day | 24 hours      | Slack #info                  | Informational alerts, business metrics           |

---

## Response Guides

| Guide                                              | Contents                                       |
| -------------------------------------------------- | ---------------------------------------------- |
| [TRIAGE_GUIDE.md](TRIAGE_GUIDE.md)                 | Initial triage procedures for any incident     |
| [AVAILABILITY.md](AVAILABILITY.md)                 | Service availability and degradation incidents |
| [LATENCY.md](LATENCY.md)                           | Performance and latency incidents              |
| [RESOURCES.md](RESOURCES.md)                       | Memory, CPU, and resource exhaustion incidents |
| [AI_SERVICE.md](AI_SERVICE.md)                     | AI service failures and fallback incidents     |
| [SECURITY.md](SECURITY.md)                         | Rate limiting and security-related incidents   |
| [POST_MORTEM_TEMPLATE.md](POST_MORTEM_TEMPLATE.md) | Post-incident review template                  |

---

## Incident Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INCIDENT LIFECYCLE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐│
│  │  DETECT  │───▶│  TRIAGE  │───▶│ MITIGATE │───▶│  RESOLVE │───▶│ REVIEW ││
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └────────┘│
│       │               │               │               │               │     │
│       ▼               ▼               ▼               ▼               ▼     │
│  Alert fires    Assess severity  Restore service  Fix root cause  Post-    │
│  On-call paged  Identify scope   Communicate      Verify fix      mortem   │
│                 Find root cause  Update status    Monitor         within   │
│                                                                   48 hours │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## On-Call Responsibilities

### During an Incident

1. **Acknowledge** the alert within SLA (15 min for critical)
2. **Assess** severity using [TRIAGE_GUIDE.md](TRIAGE_GUIDE.md)
3. **Communicate** in #incidents channel with status
4. **Mitigate** using the appropriate response guide
5. **Escalate** if unable to resolve within escalation window
6. **Document** actions taken in the incident channel

### After Resolution

1. Update status page to "Resolved"
2. Post summary in #incidents channel
3. Create post-mortem issue (P1/P2 incidents)
4. Complete post-mortem within 48 hours
5. Update runbooks if new information discovered

---

## Communication Templates

### Initial Communication (When Incident Starts)

```
🔴 INCIDENT: [Alert Name]
Status: Investigating
Impact: [Brief description of user impact]
Start Time: [HH:MM UTC]
On-Call: @[name]
Channel: #incidents

Updates will be posted every 15 minutes.
```

### Update Communication

```
🟡 UPDATE: [Alert Name]
Status: [Investigating/Identified/Mitigating]
Impact: [Current impact]
Duration: [X minutes/hours]
Next Update: [HH:MM UTC]

[Brief description of current actions]
```

### Resolution Communication

```
🟢 RESOLVED: [Alert Name]
Duration: [Total duration]
Impact: [Summary of impact]
Root Cause: [Brief description]

Post-mortem will be scheduled within 48 hours.
```

---

## Related Documentation

- **Alerting Configuration**: [monitoring/prometheus/alerts.yml](../../monitoring/prometheus/alerts.yml)
- **Alerting Thresholds**: [docs/operations/ALERTING_THRESHOLDS.md](../operations/ALERTING_THRESHOLDS.md)
- **Deployment Runbooks**: [docs/runbooks/INDEX.md](../runbooks/INDEX.md)
- **Operations Guide**: [docs/operations/OPERATIONS_DB.md](../operations/OPERATIONS_DB.md)
- **Secrets Management**: [docs/operations/SECRETS_MANAGEMENT.md](../operations/SECRETS_MANAGEMENT.md)

---

## Contacts

| Role                | Contact                  | When to Escalate                    |
| ------------------- | ------------------------ | ----------------------------------- |
| Primary On-Call     | [Configure in PagerDuty] | First responder                     |
| Secondary On-Call   | [Configure in PagerDuty] | If primary unavailable after 15 min |
| Backend Team Lead   | [Configure]              | Critical issues persisting > 30 min |
| AI Team Lead        | [Configure]              | AI-specific critical issues         |
| Infrastructure Lead | [Configure]              | Infrastructure/database issues      |

---

_Last Updated: 2025-11-25_
