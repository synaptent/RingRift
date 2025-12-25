# Incident Triage Guide

This guide provides initial triage procedures for any RingRift incident. Use this as your first step when any alert fires.

## Initial Response Checklist (First 5 Minutes)

When you receive an alert, follow this checklist:

- [ ] **Acknowledge** the alert (PagerDuty/Slack)
- [ ] **Open** the incident channel (#incidents)
- [ ] **Post** initial status message
- [ ] **Assess** severity (see below)
- [ ] **Check** service health dashboard
- [ ] **Identify** affected component

---

## Step 1: Assess Severity

### Quick Severity Assessment

| Ask This Question                        | If Yes → Severity |
| ---------------------------------------- | ----------------- |
| Is the service completely down?          | **P1 Critical**   |
| Are >5% of users seeing errors?          | **P1 Critical**   |
| Is the database unreachable?             | **P1 Critical**   |
| Is latency >5s for P99?                  | **P1 Critical**   |
| Is the AI service down with no fallback? | **P2 High**       |
| Are >1% of users seeing errors?          | **P3 Medium**     |
| Is latency elevated but functional?      | **P3 Medium**     |
| Is it informational with no user impact? | **P4 Low**        |

### Severity Definitions

| Severity        | User Impact                                  | Response Time     | Escalation                   |
| --------------- | -------------------------------------------- | ----------------- | ---------------------------- |
| **P1 Critical** | Major outage, all/most users affected        | 15 minutes        | Immediate                    |
| **P2 High**     | Significant degradation, many users affected | 1 hour            | Within 30 min if unresolved  |
| **P3 Medium**   | Minor degradation, some users affected       | 4 hours           | Within 2 hours if unresolved |
| **P4 Low**      | Minimal/no user impact                       | Next business day | N/A                          |

---

## Step 2: Quick Health Check

Run these commands to get an immediate picture:

### Application Health

```bash
# Check overall health
curl -s http://localhost:3000/health | jq

# Check readiness (includes dependency checks)
curl -s http://localhost:3000/ready | jq

# Check metrics endpoint
curl -s http://localhost:3000/metrics | grep -E "^ringrift_" | head -20
```

### Service Status

```bash
# View all containers
docker compose ps

# Check container logs (last 100 lines)
docker compose logs --tail 100 app
docker compose logs --tail 100 postgres
docker compose logs --tail 100 redis
docker compose logs --tail 100 ai-service

# Check container resource usage
docker stats --no-stream
```

### Quick Database Check

```bash
# Test database connectivity
docker exec ringrift-postgres-1 pg_isready -U ringrift

# Check active connections
docker exec ringrift-postgres-1 psql -U ringrift -c "SELECT count(*) FROM pg_stat_activity;"
```

### Quick Redis Check

```bash
# Test Redis connectivity
docker exec ringrift-redis-1 redis-cli ping

# Check Redis memory
docker exec ringrift-redis-1 redis-cli info memory | grep used_memory_human
```

---

## Step 3: Identify the Problem Area

### Which Service is Affected?

| Symptom                             | Likely Cause                | Go To                              |
| ----------------------------------- | --------------------------- | ---------------------------------- |
| All API requests failing            | Database or App down        | [AVAILABILITY.md](AVAILABILITY.md) |
| Slow responses across all endpoints | Event loop blocked, DB slow | [LATENCY.md](LATENCY.md)           |
| AI moves using fallback             | AI service issue            | [AI_SERVICE.md](AI_SERVICE.md)     |
| High memory or CPU                  | Resource exhaustion         | [RESOURCES.md](RESOURCES.md)       |
| Rate limit alerts                   | Traffic spike or abuse      | [SECURITY.md](SECURITY.md)         |
| 403/429 errors increasing           | Rate limiting active        | [SECURITY.md](SECURITY.md)         |

### Check Grafana Dashboards

If Grafana is available:

1. **Overview Dashboard**: `http://localhost:3002/d/ringrift-overview`
2. **Node.js Dashboard**: `http://localhost:3002/d/nodejs`
3. **PostgreSQL Dashboard**: `http://localhost:3002/d/postgresql`

### Check Prometheus Alerts

```bash
# View active alerts
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | {alertname: .labels.alertname, state: .state, severity: .labels.severity}'
```

---

## Step 4: Determine Scope

### Questions to Answer

1. **When did it start?**
   - Check alert firing time in Prometheus
   - Look at metrics graphs for the change point
2. **What changed recently?**
   - Check recent deployments
   - Check infrastructure changes
   - Check traffic patterns

3. **Who is affected?**
   - All users?
   - Specific region/users?
   - Only AI games?

4. **What is the blast radius?**
   - Single service?
   - Multiple services?
   - Full outage?

### Useful Queries

```bash
# Check recent deployments
git log --oneline -10

# Check when containers were started
docker compose ps --format "table {{.Name}}\t{{.Status}}"

# Check error rate in logs
docker compose logs --tail 500 app 2>&1 | grep -c "ERROR"
```

---

## Step 5: Communicate

### Post Initial Status

Post in #incidents channel immediately:

```
🔴 INCIDENT: [Alert Name]
Severity: [P1/P2/P3/P4]
Status: Investigating
Start Time: [HH:MM UTC]
On-Call: @[your name]

Initial Assessment:
- [What we know so far]
- [What we're checking]

Impact:
- [User impact description]

Next Update: [HH:MM UTC] (in 15 min for P1, 30 min for P2)
```

### Update Status Page (if P1/P2)

1. Go to status page admin
2. Create incident
3. Set appropriate status (Investigating/Identified/Monitoring)
4. Post public update

---

## Step 6: Route to Appropriate Guide

Based on your assessment, proceed to the appropriate response guide:

| Alert Category                                       | Go To                              |
| ---------------------------------------------------- | ---------------------------------- |
| DatabaseDown, RedisDown, HighErrorRate, Service Down | [AVAILABILITY.md](AVAILABILITY.md) |
| HighP99Latency, HighMedianLatency, Slow Responses    | [LATENCY.md](LATENCY.md)           |
| HighMemoryUsage, HighEventLoopLag, Resource Issues   | [RESOURCES.md](RESOURCES.md)       |
| AIServiceDown, AIFallbackRate, AI Issues             | [AI_SERVICE.md](AI_SERVICE.md)     |
| RateLimitHits, Security Concerns                     | [SECURITY.md](SECURITY.md)         |

---

## Quick Mitigation Actions

If you need to act immediately before full diagnosis:

### Restart Application

```bash
# Graceful restart
docker compose restart app

# Hard restart (if unresponsive)
docker compose stop app && docker compose up -d app
```

### Restart All Services

```bash
# Restart everything
docker compose restart

# Nuclear option - recreate
docker compose down && docker compose up -d
```

### Scale Application (if supported)

```bash
# Scale app instances
docker compose up -d --scale app=3
```

### Rollback Deployment

See [DEPLOYMENT_ROLLBACK.md](../runbooks/DEPLOYMENT_ROLLBACK.md) for full procedure.

```bash
# Quick rollback to previous version
# (Assumes you have tagged releases)
docker compose pull app:previous
docker compose up -d app
```

---

## Escalation Matrix

### When to Escalate

| Condition                     | Action                        |
| ----------------------------- | ----------------------------- |
| P1 not acknowledged in 15 min | Escalate to secondary on-call |
| P1 not resolved in 30 min     | Escalate to team lead         |
| P2 not resolved in 2 hours    | Escalate to team lead         |
| Multiple services affected    | Consider all-hands            |
| Security incident suspected   | Involve security team         |
| Database corruption suspected | Involve DBA                   |

### How to Escalate

1. **PagerDuty**: Use the escalation button in the incident
2. **Slack**: Tag the appropriate team lead in #incidents
3. **Phone**: Use the on-call phone tree (see team directory)

---

## Post-Triage Checklist

Before diving into detailed diagnosis, confirm:

- [ ] Alert acknowledged
- [ ] Severity assessed
- [ ] Initial status posted
- [ ] Scope determined
- [ ] Appropriate response guide identified
- [ ] Status page updated (if P1/P2)
- [ ] Timeline started for post-mortem

---

## Related Documentation

- [Incident Index](INDEX.md)
- [Availability Incidents](AVAILABILITY.md)
- [Latency Incidents](LATENCY.md)
- [Resource Incidents](RESOURCES.md)
- [AI Service Incidents](AI_SERVICE.md)
- [Security Incidents](SECURITY.md)
- [Post-Mortem Template](POST_MORTEM_TEMPLATE.md)
- [Alerting Thresholds](../operations/ALERTING_THRESHOLDS.md)
