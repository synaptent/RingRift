# Database Backup & Restore Drill Runbook

> **Doc Status (2025-11-30): Active Runbook**  
> **Role:** Step-by-step practice drill for verifying that Postgres backups and restores actually work in staging or a non-production environment.  
>  
> **SSoT alignment:** This runbook is a derived operational procedure over:  
> - `docs/OPERATIONS_DB.md` – canonical database operations and migration workflows  
> - `docs/DATA_LIFECYCLE_AND_PRIVACY.md` – data retention, minimisation, and recovery expectations  
> - `docs/DEPLOYMENT_REQUIREMENTS.md` / `docker-compose*.yml` – deployment topology and volumes  
>  
> **Precedence:** Postgres configuration, Prisma migrations, and deployment manifests are authoritative for actual behaviour. If this runbook disagrees with them, **code + configs win** and this document must be updated.

---

## 1. Goal & Scope

This runbook defines a **non-destructive backup and restore drill** for the RingRift Postgres database.

The goal is to regularly prove that you can:

- Take a usable logical backup (`pg_dump`) of the main database.
- Restore that backup into a **separate database** on the same Postgres instance (or a throwaway instance).
- Run a small application-level smoke test against the restored database.

This should be performed in **staging** or an equivalent non-production environment. Do **not** run this drill directly against production without adapting it to your provider’s backup/restore mechanisms.

---

## 2. Preconditions & Safety

- You are operating in **staging** (or another non-production environment).
- The Postgres container is named `postgres` and mounts `./backups` → `/backups`  
  (see `docker-compose.yml` and `docs/OPERATIONS_DB.md` §1.1–1.2).
- The primary database is named `ringrift` and owned by user `ringrift`.
- You have shell access to the host where Docker Compose is running.

> **Safety principle:** This drill restores into a **new database name** (for example `ringrift_restore_drill`) and never drops or overwrites the primary `ringrift` database.

If your topology or naming differs, adapt the commands accordingly but keep the “separate restore DB” principle.

---

## 3. Step-by-step Drill (staging with docker-compose)

All commands below assume you are on the staging host in the RingRift deployment directory (for example `/opt/ringrift`).

### 3.1 Take a fresh logical backup

1. Confirm the `postgres` container is healthy:

   ```bash
   docker compose ps postgres
   ```

2. Take a timestamped logical backup of the `ringrift` database:

   ```bash
   TIMESTAMP=$(date +%Y%m%d_%H%M%S)
   docker compose exec postgres \
     pg_dump -U ringrift -d ringrift \
     -f /backups/staging_drill_${TIMESTAMP}.sql
   ```

3. Verify the backup file exists on the host:

   ```bash
   ls -lh backups/staging_drill_*.sql | tail -1
   ```

4. (Optional) Spot-check the file header:

   ```bash
   head -20 backups/staging_drill_${TIMESTAMP}.sql
   ```

---

### 3.2 Create a separate restore database

Create a new database for the drill restore so that the primary `ringrift` database is untouched:

```bash
RESTORE_DB=ringrift_restore_drill

docker compose exec postgres \
  createdb -U ringrift "${RESTORE_DB}"

# Confirm it exists
docker compose exec postgres \
  psql -U ringrift -lqt | grep "${RESTORE_DB}" || true
```

If `createdb` fails because the database already exists, either drop it explicitly (only if you are certain it is safe), or choose a different name such as `ringrift_restore_drill_YYYYMMDD`.

---

### 3.3 Restore from the backup into the drill database

1. Restore the most recent drill backup into the new database:

   ```bash
   RESTORE_DB=ringrift_restore_drill
   LATEST_BACKUP=$(ls -1 backups/staging_drill_*.sql | sort | tail -1)

   echo "Restoring from ${LATEST_BACKUP} into ${RESTORE_DB}…"

   docker compose exec -T postgres \
     psql -U ringrift -d "${RESTORE_DB}" < "${LATEST_BACKUP}"
   ```

2. Verify core tables exist:

   ```bash
   docker compose exec postgres \
     psql -U ringrift -d "${RESTORE_DB}" -c '\dt'
   ```

   You should see the same schema tables you expect from `ringrift`.

---

### 3.4 Application-level smoke test against the restored DB

The goal of this step is to prove that the restored database is not only structurally valid, but also **usable by the application**.

1. Construct a temporary `DATABASE_URL` pointing at the restore DB. For a Compose-based Postgres, this often looks like:

   ```bash
   export DATABASE_URL="postgresql://ringrift:password@postgres:5432/ringrift_restore_drill"
   ```

   Adapt user, password, and host to match your staging configuration and `ENVIRONMENT_VARIABLES.md`.

2. Run a Prisma status check against the restored database:

   ```bash
   docker compose run --rm \
     -e DATABASE_URL="${DATABASE_URL}" \
     app npx prisma migrate status
   ```

   This should report that all expected migrations are applied.

3. (Optional but recommended) Run a light application smoke:

   - Start a one-off app container pointed at the restore DB:

     ```bash
     docker compose run --rm \
       -e DATABASE_URL="${DATABASE_URL}" \
       -p 4000:3000 \
       app npm run start
     ```

   - In another shell, hit health endpoints:

     ```bash
     curl -s http://localhost:4000/health | jq
     curl -s http://localhost:4000/ready | jq
     ```

   - Optionally create a throwaway user and a short game to ensure reads/writes succeed.

Stop the temporary app container once you are satisfied.

---

### 3.5 Cleanup

After the drill is complete and validated:

1. Drop the restore database (if no longer needed):

   ```bash
   docker compose exec postgres \
     dropdb -U ringrift ringrift_restore_drill
   ```

   Adjust the name if you used a timestamped variant.

2. Optionally prune old drill backups, keeping at least the most recent successful one:

   ```bash
   ls -1 backups/staging_drill_*.sql
   # Remove only files you are sure are safe to delete
   # rm backups/staging_drill_YYYYMMDD_HHMMSS.sql
   ```

Never delete production backups as part of this drill.

---

## 4. Validation Checklist

- [ ] `postgres` container healthy and reachable.
- [ ] New logical backup file created under `./backups/` for the drill run.
- [ ] Separate restore database (for example `ringrift_restore_drill`) created successfully.
- [ ] `psql '\dt'` against the restore DB shows expected tables.
- [ ] `prisma migrate status` passes against the restore DB.
- [ ] Optional: temporary app instance can start against the restore DB and pass basic health checks.
- [ ] Restore database dropped (or clearly labelled and left for further analysis).

Record the date, environment, and any issues found in your internal incident / ops log so you have a history of completed drills.

---

## 5. Adapting to Managed Postgres Providers

If staging or production uses a managed Postgres service (for example AWS RDS, GCP Cloud SQL, Azure Database for PostgreSQL):

- Replace the `docker compose exec postgres …` commands with provider-native mechanisms:
  - Snapshot creation / PITR configuration from the cloud console or CLI.
  - Restoring a snapshot into a **new instance** and pointing an app instance at it.
- Still perform an **application-level smoke** against the restored instance:
  - Run `npx prisma migrate status` against the restored `DATABASE_URL`.
  - Run the same login / game creation / short-play flow used in `docs/OPERATIONS_DB.md` §2.4.

The core principles remain the same:

- Backups are only useful if you can restore them.
- Restores should be proven **before** an incident, not during one.

