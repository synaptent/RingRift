# Database Migration Runbook

## Overview

Procedures for safely applying Prisma database migrations to RingRift environments. This runbook complements [OPERATIONS_DB.md](../operations/OPERATIONS_DB.md) with step-by-step operational procedures.

**Critical Rules:**

- Never run `prisma migrate dev` against staging or production
- Never run `prisma db push` against staging or production
- Always backup before applying migrations to production
- Always test migrations in staging before production

---

## Migration Types and Risk Assessment

| Migration Type      | Risk Level | Requires Backup | Downtime             |
| ------------------- | ---------- | --------------- | -------------------- |
| Add nullable column | Low        | Recommended     | None                 |
| Add table           | Low        | Recommended     | None                 |
| Add required column | Medium     | Required        | Brief                |
| Add index           | Medium     | Recommended     | None\*               |
| Modify column type  | High       | Required        | Possible             |
| Drop column         | High       | Required        | None\*\*             |
| Drop table          | Critical   | Required        | None\*\*             |
| Rename column/table | High       | Required        | May require downtime |

\*Large tables may lock during index creation  
\*\*After backwards-compatibility period

---

## Section 1: Development - Creating Migrations

### 1.1 Local Development Workflow

**Step 1: Make Schema Changes**

```bash
# Edit the Prisma schema
nano prisma/schema.prisma

# Example: Add a new field
# model User {
#   ...
#   preferredLanguage String @default("en")
# }
```

**Step 2: Generate Migration**

```bash
# Create migration with descriptive name
npm run db:migrate -- --name add_user_preferred_language

# Or directly:
npx prisma migrate dev --name add_user_preferred_language
```

**Step 3: Review Generated SQL**

```bash
# Check the migration file
cat prisma/migrations/YYYYMMDD_*_add_user_preferred_language/migration.sql

# Verify it matches expectations
# Look for:
# - Correct table name
# - Correct column definition
# - Appropriate defaults
# - No destructive operations
```

**Step 4: Test Locally**

```bash
# Run all tests
npm test

# Run integration tests that touch database
npm run test:integration

# Manually verify the change works
# Restart app and test functionality
npm run dev
```

**Step 5: Commit Migration**

```bash
# Commit both schema and migration
git add prisma/schema.prisma prisma/migrations/
git commit -m "feat: add preferredLanguage field to User model"
```

### 1.2 Backwards-Compatible Migration Pattern

For zero-downtime deployments, use expand-contract pattern:

**Phase 1: Expand (add new column as nullable)**

```sql
-- Migration 1: Add nullable column
ALTER TABLE users ADD COLUMN preferred_language VARCHAR(10);
```

**Phase 2: Migrate Data**

```sql
-- Migration 2 or application code: Populate existing rows
UPDATE users SET preferred_language = 'en' WHERE preferred_language IS NULL;
```

**Phase 3: Contract (make required after all apps updated)**

```sql
-- Migration 3: Add constraint after all app versions handle it
ALTER TABLE users ALTER COLUMN preferred_language SET NOT NULL;
ALTER TABLE users ALTER COLUMN preferred_language SET DEFAULT 'en';
```

---

## Section 2: Staging Migration Procedure

### Prerequisites

- [ ] Migration tested and committed locally
- [ ] Code merged to staging branch
- [ ] CI tests passed
- [ ] Access to staging environment

### 2.1 Pre-Migration Steps

```bash
# Connect to staging server
ssh deploy@staging-server
cd /opt/ringrift

# Check current migration status
docker compose run --rm app npx prisma migrate status

# Verify which migrations are pending
docker compose run --rm app npx prisma migrate diff \
  --from-schema-datasource prisma/schema.prisma \
  --to-migrations ./prisma/migrations
```

### 2.2 Backup Staging Database (Recommended)

```bash
# Create backup
docker compose exec postgres pg_dump -U ringrift -d ringrift > backups/staging_pre_migration_$(date +%Y%m%d_%H%M%S).sql

# Verify backup
ls -la backups/staging_pre_migration_*.sql | tail -1
```

### 2.3 Apply Migration

**Option A: Through Docker Compose restart (automatic)**

```bash
# With staging compose, migrations run on startup
docker compose -f docker-compose.yml -f docker-compose.staging.yml up -d --build
```

**Option B: Manual application**

```bash
# Apply migrations manually
docker compose run --rm app npx prisma migrate deploy

# Expected output:
# X migrations found in prisma/migrations
# X migrations already applied
# Applying migration `YYYYMMDD_migration_name`
# Applied migration `YYYYMMDD_migration_name`
```

### 2.4 Verify Migration Success

```bash
# Check migration status
docker compose run --rm app npx prisma migrate status
# Expected: All migrations applied

# Verify schema change exists
docker compose exec postgres psql -U ringrift -d ringrift -c '\d users'
# Should show new column

# Check application health
curl -s http://localhost:3000/health | jq
curl -s http://localhost:3000/ready | jq
```

### 2.5 Staging Smoke Tests

```bash
# Verify deployment via health checks
curl -s http://localhost:3000/health | jq
curl -s http://localhost:3000/ready | jq

# Manual verification:
# - Login works
# - Create game works
# - Join game works
# - Play a few moves
```

---

## Section 3: Production Migration Procedure

### Prerequisites

- [ ] Migration successfully deployed to staging
- [ ] Staging smoke tests passed
- [ ] Production backup confirmed recent
- [ ] Deployment window confirmed
- [ ] Rollback plan reviewed
- [ ] Team notified

### 3.1 Pre-Migration Notification

```bash
# Post in #deployments channel
# Template:
🗄️ DATABASE MIGRATION STARTING
Environment: Production
Migration: add_user_preferred_language
Estimated duration: 5 minutes
Operator: @your-name
Rollback plan: Restore from backup if needed
```

### 3.2 Create Production Backup

```bash
# SSH to production server (or use managed DB console)
ssh deploy@production-server

# Creating backup
docker compose exec postgres pg_dump -U ringrift -d ringrift > backups/prod_pre_migration_$(date +%Y%m%d_%H%M%S).sql

# Verify backup size and integrity
ls -lh backups/prod_pre_migration_*.sql | tail -1
head -20 backups/prod_pre_migration_*.sql | tail -1  # Should show SQL commands
```

**For managed databases (AWS RDS, GCP CloudSQL, etc.):**

```bash
# Create snapshot through provider console or CLI
# AWS example:
aws rds create-db-snapshot \
  --db-instance-identifier ringrift-prod \
  --db-snapshot-identifier ringrift-pre-migration-$(date +%Y%m%d)
```

### 3.3 Check Current State

```bash
# Verify current migration status
docker compose run --rm app npx prisma migrate status

# Check active connections
docker compose exec postgres psql -U ringrift -d ringrift -c "
SELECT count(*) as active_connections,
       state,
       wait_event_type
FROM pg_stat_activity
WHERE datname = 'ringrift'
GROUP BY state, wait_event_type;"
```

### 3.4 Apply Production Migration

**From controlled environment (CI job or ops runner):**

```bash
# Set production DATABASE_URL
export DATABASE_URL="postgresql://ringrift:$PROD_DB_PASSWORD@prod-db-host:5432/ringrift"

# Apply migration (dry run first if available)
npx prisma migrate deploy

# Watch for output:
# ✓ X migrations found in prisma/migrations
# ✓ Applying migration `YYYYMMDD_migration_name`
# ✓ Applied migration `YYYYMMDD_migration_name`
```

**Alternative: Through application container:**

```bash
docker compose run --rm app npx prisma migrate deploy
```

### 3.5 Verify Production Migration

```bash
# Check migration status
docker compose run --rm app npx prisma migrate status
# Expected: All migrations applied

# Check database schema
docker compose exec postgres psql -U ringrift -d ringrift -c '\d users'

# Check application health
curl -s https://api.ringrift.com/health | jq
curl -s https://api.ringrift.com/ready | jq
```

### 3.6 Deploy Application (if needed)

If application code changes depend on migration:

```bash
# Deploy new application version
docker compose up -d app

# Verify new version is running
curl -s https://api.ringrift.com/health | jq '.version'
```

### 3.7 Post-Migration Verification

```bash
# Run production smoke tests
# - Login
# - Create game
# - Join game
# - Make moves
# - Check affected functionality

# Monitor error rates for 15 minutes
docker compose logs app --since 15m | grep -c ERROR
# Expected: Near zero errors

# Monitor latency
# Check metrics endpoint or Grafana
```

### 3.8 Post-Migration Notification

```bash
# Post in #deployments channel
# Template:
✅ DATABASE MIGRATION COMPLETE
Environment: Production
Migration: add_user_preferred_language
Duration: X minutes
Status: Success
Verification: All smoke tests passed
```

---

## Section 4: Migration Rollback Procedures

### 4.1 Rollback Decision Tree

```
Migration Failed?
    │
    ├─► Migration didn't apply
    │       └─► Check logs, fix issue, retry
    │
    ├─► Migration applied but app broken
    │       ├─► Additive change? → Rollback app only
    │       └─► Destructive change? → Restore from backup
    │
    └─► Data integrity issues
            └─► Restore from backup
```

### 4.2 Mark Migration as Rolled Back

When migration needs to be re-created or skipped:

```bash
# Mark migration as rolled back in Prisma history
docker compose run --rm app npx prisma migrate resolve --rolled-back YYYYMMDD_migration_name

# Verify status
docker compose run --rm app npx prisma migrate status
```

### 4.3 Manual Schema Rollback (Additive Changes)

For migrations that only added columns/tables:

```bash
# Connect to database
docker compose exec postgres psql -U ringrift -d ringrift

# Reverse the migration manually
# Example: Drop added column
ALTER TABLE users DROP COLUMN IF EXISTS preferred_language;

# Exit
\q

# Mark migration as rolled back
docker compose run --rm app npx prisma migrate resolve --rolled-back YYYYMMDD_migration_name
```

### 4.4 Full Database Restore

For destructive migrations or data corruption:

```bash
# Stop application to prevent further writes
docker compose stop app ai-service

# Restore from backup
# ⚠️ THIS WILL LOSE ALL DATA SINCE BACKUP ⚠️

# For Docker Compose setup:
docker compose exec -T postgres psql -U ringrift -d ringrift < backups/prod_pre_migration_TIMESTAMP.sql

# For managed databases, use provider restore functionality

# Restart application with previous version
git checkout v<previous-version>
docker compose up -d --build app ai-service

# Verify restoration
docker compose run --rm app npx prisma migrate status
```

---

## Section 5: Common Migration Scenarios

### 5.1 Adding a Required Column to Existing Table

**Problem**: New column is required but existing rows have no value.

**Solution**: Two-phase migration

```bash
# Migration 1: Add as nullable
# prisma/schema.prisma
# newField String?  // Nullable initially

# Generate and apply migration 1
npx prisma migrate dev --name add_new_field_nullable

# Data population (in application or separate script)
docker compose exec postgres psql -U ringrift -d ringrift -c "
UPDATE table_name SET new_field = 'default_value' WHERE new_field IS NULL;
"

# Migration 2: Make required (after all data populated)
# prisma/schema.prisma
# newField String  // Now required

# Generate and apply migration 2
npx prisma migrate dev --name make_new_field_required
```

### 5.2 Renaming a Column

**Problem**: Direct rename can break running application.

**Solution**: Expand-contract with temporary column

```bash
# Migration 1: Add new column, copy data
# ALTER TABLE users ADD COLUMN new_name VARCHAR;
# UPDATE users SET new_name = old_name;

# Application update: Read from both, write to both

# Migration 2: Drop old column (after app updated everywhere)
# ALTER TABLE users DROP COLUMN old_name;
```

### 5.3 Adding Index to Large Table

**Problem**: Index creation can lock table.

**Solution**: Use `CONCURRENTLY` (PostgreSQL specific)

```sql
-- In a custom migration SQL file
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
```

**Note**: Prisma doesn't support CONCURRENTLY directly. May need manual SQL migration.

### 5.4 Dropping a Table or Column

**Pre-requisites:**

1. Verify no application code references the column/table
2. Deploy application without references
3. Wait for all old instances to be replaced
4. Only then drop the column/table

```bash
# Migration to drop column (only after app no longer uses it)
# prisma/schema.prisma
# Remove the field

npx prisma migrate dev --name drop_deprecated_column
```

---

## Section 6: Migration Troubleshooting

### Migration Timeout

```bash
# Symptom: Migration hangs or times out

# Check for blocking queries
docker compose exec postgres psql -U ringrift -d ringrift -c "
SELECT pid, state, query, now() - query_start AS duration
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;"

# Kill blocking query if safe
docker compose exec postgres psql -U ringrift -d ringrift -c "
SELECT pg_terminate_backend(<pid>);"
```

### Migration Already Applied Error

```bash
# Symptom: Prisma says migration already applied but schema doesn't match

# Check migration history
docker compose exec postgres psql -U ringrift -d ringrift -c "
SELECT * FROM _prisma_migrations ORDER BY finished_at DESC LIMIT 5;"

# If migration was partially applied:
# 1. Manually complete the schema changes
# 2. Mark as applied:
docker compose run --rm app npx prisma migrate resolve --applied MIGRATION_NAME
```

### Shadow Database Error

```bash
# Symptom: Error about shadow database in development

# Create shadow database manually
docker compose exec postgres psql -U ringrift -c "CREATE DATABASE ringrift_shadow;"

# Or use --skip-seed flag
npx prisma migrate dev --skip-seed
```

### Drift Detected

```bash
# Symptom: Schema drift detected between migrations and database

# Check current database schema
docker compose run --rm app npx prisma db pull --print

# If drift is intentional (manual changes):
# NOT RECOMMENDED - migrations should be source of truth

# If drift is from failed partial migration:
# Restore from backup and re-apply migrations
```

---

## Quick Reference Commands

| Task                       | Command                                           |
| -------------------------- | ------------------------------------------------- |
| Check migration status     | `npx prisma migrate status`                       |
| Apply pending migrations   | `npx prisma migrate deploy`                       |
| Create new migration (dev) | `npx prisma migrate dev --name <name>`            |
| Reset database (dev only!) | `npx prisma migrate reset`                        |
| Mark as applied            | `npx prisma migrate resolve --applied <name>`     |
| Mark as rolled back        | `npx prisma migrate resolve --rolled-back <name>` |
| View current schema        | `\d table_name` (in psql)                         |
| View migration SQL         | `cat prisma/migrations/*/migration.sql`           |

---

## Related Documentation

- [OPERATIONS_DB.md](../operations/OPERATIONS_DB.md) - Comprehensive database operations guide
- [DEPLOYMENT_ROUTINE.md](DEPLOYMENT_ROUTINE.md) - Standard deployment procedures
- [DEPLOYMENT_ROLLBACK.md](DEPLOYMENT_ROLLBACK.md) - Rollback procedures
- [prisma/schema.prisma](../../prisma/schema.prisma) - Database schema

---

**Last Updated**: 2024-01  
**Owner**: Platform Team  
**Review Cycle**: Quarterly
