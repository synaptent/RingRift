# RingRift Deployment Runbooks

This directory contains operational runbooks for deploying, managing, and troubleshooting RingRift in various environments.

## Runbook Index

| Runbook                                                                        | Purpose                                               | When to Use                                                                       |
| ------------------------------------------------------------------------------ | ----------------------------------------------------- | --------------------------------------------------------------------------------- |
| [DEPLOYMENT_INITIAL.md](./DEPLOYMENT_INITIAL.md)                               | First-time deployment                                 | Setting up a new environment                                                      |
| [DEPLOYMENT_ROUTINE.md](./DEPLOYMENT_ROUTINE.md)                               | Standard release procedure                            | Regular version deployments                                                       |
| [DEPLOYMENT_ROLLBACK.md](./DEPLOYMENT_ROLLBACK.md)                             | Rollback procedures                                   | When a deployment causes issues                                                   |
| [DEPLOYMENT_SCALING.md](./DEPLOYMENT_SCALING.md)                               | Scaling operations                                    | Adjusting capacity up/down                                                        |
| [DATABASE_MIGRATION.md](./DATABASE_MIGRATION.md)                               | Database schema changes                               | Prisma migration procedures                                                       |
| [DATABASE_BACKUP_AND_RESTORE_DRILL.md](./DATABASE_BACKUP_AND_RESTORE_DRILL.md) | Backup/restore drill for Postgres                     | Periodic staging drills and pre-change rehearsals                                 |
| [SECRETS_ROTATION_DRILL.md](./SECRETS_ROTATION_DRILL.md)                       | JWT and DB secrets rotation drill                     | Periodic security drills and pre-launch hardening                                 |
| [AI_SERVICE_DEGRADATION_DRILL.md](./AI_SERVICE_DEGRADATION_DRILL.md)           | AI service degradation drill (staging)                | Staging drills for AI availability, fallbacks, dashboards, and alert validation   |
| [ORCHESTRATOR_ROLLOUT_RUNBOOK.md](./ORCHESTRATOR_ROLLOUT_RUNBOOK.md)           | Orchestrator rollout, rollback, and incident handling | Changing orchestrator rollout phase or responding to orchestrator-specific alerts |
| [FSM_VALIDATION_ROLLOUT.md](./FSM_VALIDATION_ROLLOUT.md)                       | FSM validation mode rollout and monitoring            | Enabling FSM validation (shadow or active mode) in production                     |

## Quick Reference

### Common Operations

| Task                | Command                                      |
| ------------------- | -------------------------------------------- |
| Check system health | `curl -s http://localhost:3000/health \| jq` |
| Check readiness     | `curl -s http://localhost:3000/ready \| jq`  |
| View service logs   | `docker compose logs -f app`                 |
| View all logs       | `docker compose logs -f`                     |
| Validate config     | `npm run validate:deployment`                |
| Apply migrations    | `npx prisma migrate deploy`                  |

### Service Ports

| Service         | Internal Port | External Port |
| --------------- | ------------- | ------------- |
| App (HTTP)      | 3000          | 3000          |
| App (WebSocket) | 3001          | 3001          |
| PostgreSQL      | 5432          | 5432          |
| Redis           | 6379          | 6379          |
| AI Service      | 8001          | 8001          |
| Nginx           | 80/443        | 80/443        |
| Prometheus      | 9090          | 9090          |
| Grafana         | 3000          | 3002          |
| Alertmanager    | 9093          | 9093          |

### Health Check Endpoints

```bash
# Liveness probe (is the process alive?)
curl -s http://localhost:3000/health | jq

# Readiness probe (can it serve traffic?)
curl -s http://localhost:3000/ready | jq

# AI service health
curl -s http://localhost:8001/health | jq
```

### Environment Files

| Environment | Base Config        | Override                   | Env File        |
| ----------- | ------------------ | -------------------------- | --------------- |
| Development | docker-compose.yml | -                          | .env            |
| Staging     | docker-compose.yml | docker-compose.staging.yml | .env.staging    |
| Production  | docker-compose.yml | docker-compose.prod.yml    | secrets manager |

## Related Documentation

- [DEPLOYMENT_REQUIREMENTS.md](../DEPLOYMENT_REQUIREMENTS.md) - Infrastructure requirements per environment
- [OPERATIONS_DB.md](../OPERATIONS_DB.md) - Database operations guide
- [SECRETS_MANAGEMENT.md](../SECRETS_MANAGEMENT.md) - Secrets handling procedures
- [ENVIRONMENT_VARIABLES.md](../ENVIRONMENT_VARIABLES.md) - Complete environment variable reference
- [ORCHESTRATOR_ROLLOUT_PLAN.md](../architecture/ORCHESTRATOR_ROLLOUT_PLAN.md) - Orchestrator rollout phases, SLOs, and environment profiles
- [tests/load/README.md](../../tests/load/README.md) - k6 load harness and scenario definitions
- [PASS22_ASSESSMENT_REPORT.md](../archive/assessments/PASS22_ASSESSMENT_REPORT.md) - Production polish and validation plan

## Emergency Contacts

| Role             | Contact              | Escalation         |
| ---------------- | -------------------- | ------------------ |
| On-Call Engineer | [Define in your org] | PagerDuty/OpsGenie |
| Database Admin   | [Define in your org] | -                  |
| Infrastructure   | [Define in your org] | -                  |
| Security         | [Define in your org] | -                  |

---

**Note**: These runbooks assume familiarity with Docker, Docker Compose, and basic command-line operations. For environment setup instructions, see [QUICKSTART.md](../../QUICKSTART.md).
