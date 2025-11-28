# Database Down Runbook

> **Doc Status (2025-11-27): Active Runbook**  
> **Role:** Step-by-step operational guide for diagnosing and recovering from a RingRift database outage.
>
> **SSoT alignment:** This runbook is a derived operational procedure over:
>
> - The **Operational SSoT** for deployment and infrastructure (`docs/DEPLOYMENT_REQUIREMENTS.md`, `docker-compose*.yml`, `src/server/config/**/*.ts`).
> - Data and persistence guidance in `docs/OPERATIONS_DB.md` and `docs/DATA_LIFECYCLE_AND_PRIVACY.md`.
>
> **Precedence:** Docker/Kubernetes configs, environment variable definitions, and `DataRetentionService` / Prisma migrations are authoritative for behaviour. If this runbook disagrees with them, **code + configs win** and this document must be updated.

---

## 1. When This Alert Fires

**Alert:** `DatabaseDown`  
**Source:** `monitoring/prometheus/alerts.yml` (`availability` group)  
**Condition:** `ringrift_service_status{service="database"} == 0` for more than 1 minute.

**User impact:**

- New logins, game creation, and persistence operations fail.
- Existing WebSocket sessions may begin to error on state updates.

---

## 2. Triage Checklist

1. **Confirm alert and scope**
   - Check Alertmanager and Grafana dashboards for:
     - `DatabaseDown` firing across all instances vs a single node.
     - Correlated alerts: `HighErrorRate`, `ServiceOffline`, `DatabaseResponseTimeSlow`.

2. **Verify process and container health**
   - On the primary database host:
     ```bash
     docker compose ps postgres
     docker compose logs -f postgres
     ```
   - If running in another orchestrator (Kubernetes, managed Postgres), use the provider's status/health tooling.

3. **Check basic connectivity**
   - From the app host:
     ```bash
     docker compose exec app nc -vz postgres 5432 || true
     ```
   - If this fails, suspect networking, DNS, or security group changes.

4. **Look for resource exhaustion**
   - On the DB node:
     ```bash
     df -h
     free -m
     ```
   - Address full disks or exhausted memory before restarting services.

---

## 3. Remediation Steps

> **Note:** Coordinate with your DBA or infra team before taking disruptive actions in production.

1. **Restart or fail over the database**
   - For single-node Docker Compose:
     ```bash
     cd /opt/ringrift
     docker compose restart postgres
     ```
   - For managed Postgres / HA setups, follow your platform's documented failover procedure.

2. **Validate schema health**
   - Once the database is accepting connections:
     ```bash
     docker compose exec postgres psql -U ringrift -d ringrift -c '\dt'
     ```
   - Optionally re-run Prisma migrations if corruption or partial deploy is suspected:
     ```bash
     docker compose run --rm app npx prisma migrate deploy
     ```

3. **Verify app recovery**
   - After database recovers, confirm the app transitions back to healthy:
     ```bash
     curl -s http://localhost:3000/ready | jq
     ```
   - Check logs for any repeated connection errors:
     ```bash
     docker compose logs -f app | grep -i "database" | tail -n 50
     ```

---

## 4. Validation

- [ ] `/ready` reports database as healthy with reasonable latency.
- [ ] New user registration and login succeed.
- [ ] New games can be created and moves are persisted.
- [ ] `DatabaseDown` alert clears in Prometheus/Alertmanager.

---

## 5. Follow-Up and Prevention

- [ ] Review database logs for root cause (OOM, disk full, connection limits, configuration changes).
- [ ] Verify `DATABASE_URL` and related pool settings match `docs/ENVIRONMENT_VARIABLES.md` and environment constraints.
- [ ] Ensure backups and retention policies follow `docs/OPERATIONS_DB.md`.
- [ ] If this was triggered by load, consider capacity planning or connection pool tuning.

---

## 6. TODO / Local Adaptation

This stub is intentionally generic. Adapt the following to your environment:

- [ ] Replace `/opt/ringrift` and Docker commands with your actual deployment topology.
- [ ] Document managed Postgres console / support runbooks if applicable.
- [ ] Link to any organisation-specific incident templates or escalation policies.
