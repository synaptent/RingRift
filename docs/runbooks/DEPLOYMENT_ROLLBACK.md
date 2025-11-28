# Deployment Rollback Runbook

## Overview

Procedures for reverting RingRift to a previous version when a deployment causes issues. This runbook covers multiple rollback scenarios from quick container rollback to full database restore.

**Severity Levels:**

- **P0 (Critical)**: Service completely down, immediate rollback required
- **P1 (High)**: Major functionality broken, rollback within minutes
- **P2 (Medium)**: Degraded performance or minor features broken, planned rollback

---

## Decision Tree: When to Rollback

```
Issue Detected
    │
    ├─► Service Completely Down (P0)
    │       └─► Immediate Rollback (Section 1)
    │
    ├─► Major Feature Broken (P1)
    │       ├─► New in this release? → Rollback (Section 2)
    │       └─► Pre-existing? → Investigate, don't rollback
    │
    ├─► Performance Degraded (P2)
    │       ├─► >50% degradation? → Consider rollback
    │       └─► <50% degradation? → Investigate first
    │
    └─► Minor Issues
            └─► Fix forward (don't rollback)
```

---

## Section 1: Immediate Application Rollback (No Database Changes)

**Use When**: Application crashes, API errors, but no database schema changes in the release.

**Time**: 2-5 minutes

### Step 1.1: Identify Previous Version

```bash
# Check deployment history
docker compose logs app | head -50 | grep version

# Or check git history
git log --oneline -5

# Note the previous version (e.g., v1.2.2)
PREVIOUS_VERSION="v1.2.2"
```

### Step 1.2: Stop Current Deployment

```bash
# Stop the problematic containers
docker compose stop app ai-service

# Verify stopped
docker compose ps
```

### Step 1.3: Rollback Application

**Option A: Git-based rollback (staging)**

```bash
# Checkout previous version
git checkout v${PREVIOUS_VERSION}

# Rebuild and deploy
docker compose up -d --build app ai-service
```

**Option B: Image-based rollback (production)**

```bash
# Pull previous image
docker pull ringrift/app:${PREVIOUS_VERSION}
docker pull ringrift/ai-service:${PREVIOUS_VERSION}

# Update compose to use previous version
# Edit docker-compose.yml or use:
docker compose up -d --no-build app ai-service

# Or specify image directly
docker run -d --name ringrift-app-rollback \
  --env-file .env \
  -p 3000:3000 \
  ringrift/app:${PREVIOUS_VERSION}
```

### Step 1.4: Verify Rollback

```bash
# Check version
curl -s http://localhost:3000/health | jq '.version'
# Expected: Previous version number

# Check health
curl -s http://localhost:3000/health | jq '.status'
# Expected: "healthy"

# Check readiness
curl -s http://localhost:3000/ready | jq
# Expected: All checks passing
```

### Step 1.5: Notify Team

```bash
# Post in #deployments and #incidents
# Template:
⚠️ ROLLBACK EXECUTED
Environment: [staging/production]
Rolled back from: v1.2.3
Rolled back to: v1.2.2
Reason: [brief description]
Status: Service restored
Next steps: [investigating root cause]
```

---

## Section 2: Rollback with Database Migration Reversal

**Use When**: Release included database migrations that need to be undone.

**Time**: 15-30 minutes

⚠️ **Warning**: Database rollbacks can cause data loss. Proceed with caution.

### Step 2.1: Assess Migration Impact

```bash
# Check migration status
docker compose run --rm app npx prisma migrate status

# List recent migrations
ls -la prisma/migrations/

# Review what the migration changed
cat prisma/migrations/YYYYMMDD_migration_name/migration.sql
```

### Step 2.2: Determine Rollback Strategy

| Migration Type          | Strategy                                 | Risk        |
| ----------------------- | ---------------------------------------- | ----------- |
| Added column (nullable) | Leave in place, rollback app only        | Low         |
| Added column (required) | May need data backfill or column removal | Medium      |
| Added table             | Leave in place, rollback app only        | Low         |
| Dropped column          | Restore from backup                      | High        |
| Dropped table           | Restore from backup                      | Critical    |
| Modified column         | Assess, may need backup restore          | Medium-High |

### Step 2.3: Backup Current State

```bash
# Always backup before attempting any DB changes
docker compose exec postgres pg_dump -U ringrift -d ringrift > backups/pre_rollback_$(date +%Y%m%d_%H%M%S).sql
```

### Step 2.4: Rollback Options

**Option A: Additive migrations (safe to leave in place)**

If the migration only added tables/columns that the old app ignores:

```bash
# Just rollback the application (see Section 1)
# Database changes remain but are unused
```

**Option B: Prisma mark migration as rolled back**

```bash
# Mark the failed migration as rolled back in Prisma's history
docker compose run --rm app npx prisma migrate resolve --rolled-back MIGRATION_NAME

# Verify status
docker compose run --rm app npx prisma migrate status
```

**Option C: Manual SQL reversal**

```bash
# Connect to database
docker compose exec postgres psql -U ringrift -d ringrift

# Execute reversal SQL (must be written based on migration)
# Example: If migration added a column
ALTER TABLE users DROP COLUMN IF EXISTS new_column;

# Exit psql
\q

# Rollback application
# (See Section 1)
```

**Option D: Restore from backup (data loss possible)**

```bash
# ⚠️ This will lose all data since the backup
# Use only for critical situations (dropped tables/columns with data)

# Stop all services
docker compose stop app ai-service

# Restore database
docker compose exec -T postgres psql -U ringrift -d ringrift < backups/pre_v1.2.3_TIMESTAMP.sql

# Rollback application to version that matches the backup
git checkout v${PREVIOUS_VERSION}
docker compose up -d --build app ai-service
```

### Step 2.5: Verify Database State

```bash
# Check tables exist
docker compose exec postgres psql -U ringrift -d ringrift -c '\dt'

# Check recent data integrity
docker compose exec postgres psql -U ringrift -d ringrift -c 'SELECT COUNT(*) FROM users;'
docker compose exec postgres psql -U ringrift -d ringrift -c 'SELECT COUNT(*) FROM games;'

# Application verification
curl -s http://localhost:3000/ready | jq
```

---

## Section 3: Blue-Green Rollback

**Use When**: Using blue-green deployment with traffic switching.

**Time**: 1-2 minutes (instant traffic switch)

### Step 3.1: Identify Active Environment

```bash
# Check which environment is receiving traffic
# This depends on your load balancer configuration

# Example for nginx:
grep upstream /etc/nginx/nginx.conf
```

### Step 3.2: Switch Traffic Back

```bash
# Update load balancer to point to previous (blue) environment
# Example nginx configuration change:

# Edit nginx config
# Change: upstream backend { server green:3000; }
# To:     upstream backend { server blue:3000; }

# Reload nginx
docker compose exec nginx nginx -s reload
```

### Step 3.3: Verify Traffic Switch

```bash
# Check requests are hitting blue environment
curl -s http://localhost:3000/health | jq '.version'
# Should show previous version

# Monitor error rates
# Check metrics/logs for errors stopping
```

### Step 3.4: Keep Green Running (for investigation)

```bash
# Don't delete green environment immediately
# Useful for debugging

# Mark it as not receiving traffic
docker compose -f docker-compose.green.yml exec app touch /app/ROLLED_BACK
```

---

## Section 4: Emergency Full System Restore

**Use When**: Multiple components affected, need to restore entire system to known good state.

**Time**: 30-60 minutes

### Step 4.1: Declare Incident

```bash
# Post immediately
🚨 INCIDENT DECLARED - Full System Rollback in Progress
Environment: [production]
Severity: P0
Status: Service degraded/down
ETA to resolution: ~45 minutes
Incident commander: @your-name
```

### Step 4.2: Stop All Services

```bash
# Stop application services (keep database running for now)
docker compose stop app ai-service nginx

# Verify stopped
docker compose ps
```

### Step 4.3: Identify Last Known Good State

```bash
# Find last successful deployment
git log --oneline --grep="deploy" -10

# Find database backup from that time
ls -la backups/ | head -20

# Note:
# Last good version: v1.2.0
# Last good backup: pre_v1.2.1_20240115_100000.sql
```

### Step 4.4: Restore Database (if needed)

```bash
# Create fresh backup of current state (for forensics)
docker compose exec postgres pg_dump -U ringrift -d ringrift > backups/incident_$(date +%Y%m%d_%H%M%S).sql

# Restore to known good state
# ⚠️ Data loss warning
docker compose exec -T postgres psql -U ringrift -d ringrift < backups/pre_v1.2.1_20240115_100000.sql
```

### Step 4.5: Restore Application

```bash
# Checkout known good version
git checkout v1.2.0

# Rebuild everything
docker compose build

# Start services in order
docker compose up -d postgres redis
sleep 10

docker compose up -d ai-service
sleep 20

docker compose up -d app
sleep 30

docker compose up -d nginx
```

### Step 4.6: Validate Full System

```bash
# Run comprehensive health checks
# Manual checks:
curl -s http://localhost:3000/health | jq
curl -s http://localhost:3000/ready | jq
curl -s http://localhost:8001/health | jq

# Test critical flows
# - User login
# - Game creation
# - Game joining
# - Making moves
```

### Step 4.7: Restore Traffic

```bash
# If using maintenance mode:
docker compose exec nginx rm /etc/nginx/maintenance.flag
docker compose exec nginx nginx -s reload

# Monitor error rates and user reports
```

---

## Post-Rollback Actions

### Immediate (within 1 hour)

- [ ] Notify stakeholders of rollback completion
- [ ] Update incident status
- [ ] Collect logs from failed deployment
- [ ] Document timeline of events

### Short-term (within 24 hours)

- [ ] Root cause analysis
- [ ] Create fix for failed deployment
- [ ] Test fix in staging
- [ ] Schedule re-deployment
- [ ] Update runbooks if gaps found

### Long-term (within 1 week)

- [ ] Post-incident review meeting
- [ ] Document lessons learned
- [ ] Implement preventive measures
- [ ] Update deployment checklist if needed

---

## Rollback Verification Checklist

After any rollback, verify:

- [ ] `/health` returns correct (previous) version
- [ ] `/ready` shows all dependencies healthy
- [ ] Error rates returned to baseline
- [ ] Response times returned to baseline
- [ ] Users can login
- [ ] Users can create games
- [ ] Users can join games
- [ ] Game moves execute correctly
- [ ] WebSocket connections work
- [ ] No data corruption detected

---

## Common Rollback Scenarios

### Scenario: Secret Validation Failure

```bash
# Symptom: Container exits immediately with secret validation error
# Check logs:
docker compose logs app | grep -i secret

# Fix: Verify .env has proper (non-placeholder) secrets
# Then restart:
docker compose up -d app
```

### Scenario: Database Connection Refused

```bash
# Symptom: App starts but can't connect to database
# Rollback is typically not needed - fix connection

# Check database is running
docker compose ps postgres

# Check DATABASE_URL
grep DATABASE_URL .env

# Restart with correct config
docker compose up -d app
```

### Scenario: AI Service Unavailable

```bash
# Symptom: Games work but AI moves fail
# Rollback may not be needed if AI_FALLBACK_ENABLED=true

# Check AI service
curl -s http://localhost:8001/health

# Restart AI service
docker compose restart ai-service

# If persistent, rollback AI service only:
docker pull ringrift/ai-service:${PREVIOUS_VERSION}
docker compose up -d ai-service
```

### Scenario: Memory Leak Causing OOM

```bash
# Symptom: Service works initially then crashes
# Check container stats
docker stats app

# Immediate mitigation: restart
docker compose restart app

# Rollback if leak is in new version
git checkout v${PREVIOUS_VERSION}
docker compose up -d --build app
```

---

## Emergency Contacts

| Role             | Contact  | Escalation         |
| ---------------- | -------- | ------------------ |
| On-Call Engineer | [Define] | First response     |
| Database Admin   | [Define] | Database restore   |
| Infrastructure   | [Define] | System-wide issues |
| Engineering Lead | [Define] | P0 incidents       |

---

## Related Documentation

- [DEPLOYMENT_ROUTINE.md](./DEPLOYMENT_ROUTINE.md) - Standard deployment
- [DATABASE_MIGRATION.md](./DATABASE_MIGRATION.md) - Migration procedures
- [OPERATIONS_DB.md](../OPERATIONS_DB.md) - Database backup/restore details

---

**Last Updated**: 2024-01  
**Owner**: Platform Team  
**Review Cycle**: Quarterly
