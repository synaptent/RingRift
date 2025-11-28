# RingRift Database Operations & Migrations

> **Doc Status (2025-11-27): Active**  
> Canonical operational reference for running the Postgres database and Prisma migrations across local, staging, and production environments. This is not a rules or lifecycle SSoT; it complements `ENVIRONMENT_VARIABLES.md`, `DEPLOYMENT_REQUIREMENTS.md`, and `DOCUMENTATION_INDEX.md` for overall deployment architecture.

This playbook describes how to operate the Postgres database that backs RingRift across local development, staging, and production.

It is aimed at engineers and operators and is the canonical reference for:

- How Postgres instances are expected to run in each environment
- How Prisma migrations are created, reviewed, and applied
- How backups and restores are handled at a conceptual level
- How to respond when a migration or DB incident goes wrong

For general environment setup, see [`QUICKSTART.md`](../QUICKSTART.md) and [`README.md`](../README.md). For the Prisma schema itself, see [`prisma/schema.prisma`](../prisma/schema.prisma).

---

## 1. Environments & Database Expectations

RingRift uses PostgreSQL as the primary persistence layer and Prisma as the ORM. All schema changes must flow through checked-in Prisma migrations under [`prisma/migrations`](../prisma/migrations/20251119080345_init/migration.sql).

At a high level:

- **Local development** uses a Postgres container from [`docker-compose.yml`](../docker-compose.yml) or a local Postgres instance.
- **Staging** is a single-node Docker Compose stack using [`docker-compose.yml`](../docker-compose.yml) + [`docker-compose.staging.yml`](../docker-compose.staging.yml) and the [`.env.staging`](../.env.staging) template.
- **Production** is expected to use a managed or self-operated Postgres instance; the exact hosting platform is out of scope for this repo, but the workflows below assume:
  - A stable `DATABASE_URL` for the primary database.
  - Regular automated backups or snapshots managed by infra/ops.

### 1.1 Local Development

**Postgres instance**

- Start the local DB (and Redis) using Docker:

  ```bash
  docker compose up -d postgres redis
  ```

- By default, the Postgres container exposes:
  - Host: `localhost:5432`
  - DB name: `ringrift`
  - User: `ringrift`
  - Password: from `DB_PASSWORD` (default `password`) in [`docker-compose.yml`](../docker-compose.yml) / [`.env.example`](../.env.example).
- The local Prisma `DATABASE_URL` usually matches [`DATABASE_URL` in .env.example](../.env.example):  
  `postgresql://ringrift:password@localhost:5432/ringrift`.

**Applying migrations (schema changes)**

- Use **development-only** Prisma commands:
  - Apply and generate migrations via the npm script:

    ```bash
    npm run db:migrate   # wraps: npx prisma migrate dev
    npm run db:generate  # regenerate Prisma client
    ```

  - To reset the local database from scratch (drops and recreates the DB, re-applies all migrations, and optionally seeds):

    ```bash
    npx prisma migrate reset
    ```

- `prisma migrate dev` **must never** point at staging or production databases.

**Backups & restores (local)**

For local development, the database is considered disposable. Developers can usually reset instead of restoring.

When you need an ad-hoc backup (for example, before experimenting with destructive local migrations), you can use `pg_dump` against the `postgres` container:

```bash
# From the repo root
docker compose exec postgres pg_dump -U ringrift -d ringrift -f /backups/dev_YYYYMMDD_HHMM.sql
```

The `postgres` service mounts `./backups` into `/backups` (see [`docker-compose.yml`](../docker-compose.yml)), so the file will appear under `./backups/` on the host.

To restore a local backup:

```bash
# Drop and recreate the database (dev only)
docker compose exec postgres dropdb -U ringrift ringrift || true
docker compose exec postgres createdb -U ringrift ringrift

# Restore from a dump file in ./backups
docker compose exec -T postgres psql -U ringrift -d ringrift < backups/dev_YYYYMMDD_HHMM.sql
```

### 1.2 Staging

**Topology**

Staging is intended to look like a small production deployment while still living on a developer or CI host:

- Single-node stack defined by [`docker-compose.yml`](../docker-compose.yml) + [`docker-compose.staging.yml`](../docker-compose.staging.yml).
- Uses `.env` based on [`.env.staging`](../.env.staging) for secrets and connection strings.
- The `app` service runs:

  ```bash
  npx prisma migrate deploy && node dist/server/index.js
  ```

  on startup (see [`docker-compose.staging.yml`](../docker-compose.staging.yml)).

**Migrations in staging**

- Staging **only** ever runs `prisma migrate deploy` against its database.
- Migrations are applied automatically when you start the stack:

  ```bash
  docker-compose -f docker-compose.yml -f docker-compose.staging.yml up --build
  ```

- If you need to re-run migrations manually (for example, from CI or an ops shell), target the staging `DATABASE_URL` from `.env`:

  ```bash
  # From the host, with .env pointing at staging DB
  npx prisma migrate deploy
  ```

**Backups & restores (staging)**

- Staging data is useful for reproducing production-like behaviour, but it is **not** source-of-truth data.
- For the default local-only staging stack:
  - You may treat the DB as disposable and recreate it with a fresh `docker compose down && docker compose up`.
  - For important staging datasets, take manual `pg_dump` backups using the same pattern as local development, but using staging credentials from `.env.staging`.

### 1.3 Production

**Topology**

Production is expected to use:

- A managed or self-hosted Postgres instance with:
  - Regular automated backups or snapshots (for example, daily full backups plus point-in-time recovery, depending on provider capabilities).
  - Restricted network access (app servers only, VPN or private network).
- One or more app containers or processes, configured with a production `DATABASE_URL`.

The repo does **not** prescribe a specific cloud provider. Operators must ensure:

- There is a documented way to:
  - Trigger an on-demand backup/snapshot before migrations.
  - Restore a backup into either the primary instance or a new instance.
- Access to run `npx prisma migrate deploy` against the production `DATABASE_URL` is controlled (usually via CI/CD or an ops-only runner).

**Migrations in production**

- Production migrations **must** be applied via:

  ```bash
  npx prisma migrate deploy
  ```

  pointed at the production `DATABASE_URL`.

- Recommended pattern:
  1. Ensure the target commit (schema + code) has passed tests and staging validation.
  2. Take or confirm a recent backup/snapshot of the production DB.
  3. Run `prisma migrate deploy` once, from a controlled environment (CI job or ops shell).
  4. Deploy the new app version.

- For zero/low-downtime changes, design migrations to be **backwards compatible** with the previous app version (additive/expand-and-contract changes) so that:
  - The old app can still run against the new schema during rollout or rollback.
  - Cleanup migrations (dropping old columns, etc.) happen only after all app instances are updated.

**Backups & restores (production)**

- Assumptions:
  - The primary backup mechanism is provider-managed snapshots and/or continuous PITR.
  - For self-hosted Postgres, an equivalent schedule using `pg_basebackup` and/or regular `pg_dump` + `pg_restore` exists.
- Before any structural migration, confirm:
  - The last automated backup completed successfully.
  - You know which snapshot or backup you would restore if needed.
- In an incident, prefer:
  - Restoring a snapshot into a **new** database instance.
  - Pointing the app at that new instance after validation, rather than performing in-place destructive restore on the live DB.

---

## 2. Safe Prisma Migration Workflow

This workflow applies to any schema change in [`prisma/schema.prisma`](../prisma/schema.prisma). All changes must go through migrations checked into Git.

### 2.1 Design & local development

1. Edit the schema in [`prisma/schema.prisma`](../prisma/schema.prisma).
2. From a **local dev** environment, generate a new migration:

   ```bash
   npm run db:migrate -- --name meaningful_migration_name
   # or directly:
   npx prisma migrate dev --name meaningful_migration_name
   ```

3. Run local tests (at minimum):

   ```bash
   npm test
   ```

4. Inspect the generated SQL under [`prisma/migrations`](../prisma/migrations/20251119080345_init/migration.sql) to confirm it matches intent (especially for destructive changes).
5. Commit **both**:
   - The updated [`prisma/schema.prisma`](../prisma/schema.prisma).
   - The new directory under [`prisma/migrations`](../prisma/migrations/20251119080345_init/migration.sql).

### 2.2 Staging rollout

1. Merge the migration branch into the staging branch (for example, `main` or `staging`), following your team’s normal review process.
2. Deploy or start the staging stack:

   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.staging.yml up --build
   ```

   This runs `npx prisma migrate deploy` inside the `app` container on boot.

3. Verify migrations succeeded:
   - Check container logs for `prisma migrate deploy` success.
   - Optionally run:

     ```bash
     npx prisma migrate status
     ```

     against the staging `DATABASE_URL`.

4. Run staging smoke tests:
   - Basic flows: registration, login, game creation, joining, playing a short game.
   - Any specific features touched by the schema change.

### 2.3 Production rollout

1. Confirm staging deployment is healthy and tests have passed.
2. Coordinate a release window with whoever operates production.
3. Confirm a recent production backup/snapshot exists (and note its identifier).
4. From a controlled environment (CI job or ops shell with `DATABASE_URL` pointing at production), apply migrations:

   ```bash
   npx prisma migrate deploy
   ```

5. Deploy the new app version.
6. Monitor:
   - `/health` and metrics endpoints.
   - Error logs for DB or migration-related issues.
   - Core flows: login, lobby, game creation/join, short play session.

### 2.4 Things you must **never** do in staging or production

- Do **not** run:

  ```bash
  npx prisma migrate dev
  ```

  against any shared (staging or production) database.

- Do **not** run:

  ```bash
  npx prisma db push
  ```

  against staging or production. This bypasses the migrations history and risks drift.

- Do **not** manually edit migration SQL files that have already been applied to any shared environment. Instead, create a **new** migration that corrects or reverts the previous one.
- Avoid running `prisma migrate deploy` directly from arbitrary developer laptops against production. Prefer CI or dedicated ops tooling.

---

## 3. Rollback & Disaster-Recovery Playbooks

This section describes how to respond when migrations or DB incidents go wrong. Exact commands depend on your hosting provider; treat these as checklists and adapt provider-specific steps (console, CLI, etc.) accordingly.

### 3.1 Bad migration that breaks the app (schema is intact)

**Examples**

- A new column is required by the app but is not populated correctly.
- A constraint blocks legitimate writes but does not corrupt existing data.

**Response checklist**

1. **Stabilize the system**
   - If only some requests fail, keep the system up but avoid further deployments.
   - If the app is unusable, temporarily roll back to the previous app version while keeping the new schema (if backwards-compatible).
2. **Diagnose**
   - Inspect logs for specific constraint or nullability errors.
   - Confirm that the underlying tables and data are still intact (e.g., via provider console or `psql`).
3. **Fix forward with a corrective migration**
   - Make the necessary schema change locally in [`prisma/schema.prisma`](../prisma/schema.prisma) (for example, relax a constraint, add a default, or add a compatibility column).
   - Generate a new migration with `prisma migrate dev`, test locally, and roll it through staging.
   - Apply it to production with `prisma migrate deploy`.
4. **Clean up (if needed)**
   - Once the system is stable and old paths are no longer used, you can design follow-up migrations to tighten constraints again.

### 3.2 Bad migration that corrupts or deletes data

**Examples**

- Dropping a critical column or table (`DROP TABLE`, `DROP COLUMN`).
- Accidental mass update or delete without a proper `WHERE` clause.

**Response checklist**

1. **Immediately stop new writes**
   - Scale app instances down to zero or enable maintenance mode, depending on infrastructure.
   - In Docker Compose-based deployments, stop the `app` service:

     ```bash
     docker compose stop app
     ```

2. **Assess impact**
   - Identify which tables/rows are affected.
   - Determine the approximate time window when corruption occurred.
3. **Choose a restore point**
   - For managed Postgres:
     - Select a pre-incident snapshot or PITR timestamp.
   - For self-managed Postgres:
     - Select the appropriate `pg_dump` / physical backup to restore from.
4. **Restore into a safe environment**
   - Prefer restoring the backup into a **new** database instance.
   - Apply migrations up to, but **excluding**, the bad migration (using `prisma migrate deploy` against that instance).
5. **Validate**
   - Run smoke tests (login, lobby, game creation/join, short game).
   - Spot-check critical data (recent users, recent games).
6. **Cut over**
   - Point the app’s `DATABASE_URL` at the restored instance.
   - Gradually bring traffic back online, monitoring error rates and logs.

### 3.3 Production DB incident (infrastructure or operator error)

**Examples**

- Underlying storage failure or provider outage affecting the primary instance.
- Accidental drop of the entire database.

**Response checklist**

1. **Declare an incident**
   - Notify relevant stakeholders and freeze deployments.
2. **Stabilize**
   - If the DB is partially available, consider switching the app to a maintenance page to prevent further damage.
3. **Engage provider or DB team**
   - Open a support case with the cloud DB provider, or engage the internal DB team for self-hosted setups.
4. **Restore from backup**
   - Work with the provider/DBA to restore the most recent safe backup to a new instance.
   - Apply `prisma migrate deploy` if necessary to bring schema to the expected version.
5. **Validate and cut over**
   - Run core smoke tests and targeted checks on critical tables.
   - Update application configuration or secrets so `DATABASE_URL` points at the restored instance.
6. **Post-incident review**
   - Capture a timeline of events, root cause, and follow-up actions (for example, improved backup cadence, additional access controls, safer migration patterns).

---

## 4. Quick Reference

**Local development**

- Start DB: `docker compose up -d postgres redis`
- Apply migrations: `npm run db:migrate` (Prisma `migrate dev`)
- Reset DB: `npx prisma migrate reset`

**Staging**

- Start stack + apply migrations:  
  `docker-compose -f docker-compose.yml -f docker-compose.staging.yml up --build`
- Manual migrate: `npx prisma migrate deploy` (with staging `DATABASE_URL` in `.env`)

**Production**

- Apply migrations (from CI/ops runner): `npx prisma migrate deploy`
- Always confirm a recent backup/snapshot before running structural migrations.

**Never in staging/production**

- `npx prisma migrate dev`
- `npx prisma db push`
- Manual edits to already-applied migration SQL files

This document should be kept in sync with [`QUICKSTART.md`](../QUICKSTART.md), [`docker-compose.yml`](../docker-compose.yml), and [`docker-compose.staging.yml`](../docker-compose.staging.yml) whenever deployment or database practices change.
