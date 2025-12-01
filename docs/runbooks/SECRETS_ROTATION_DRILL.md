# Secrets Rotation Drill Runbook

> **Doc Status (2025-11-30): Active Runbook**  
> **Role:** Step-by-step practice drill for rotating critical RingRift secrets (JWT signing keys and database credentials) in a **non‑production** environment and verifying application health.  
>  
> **SSoT alignment:** This runbook is a derived operational procedure over:  
> - `docs/SECRETS_MANAGEMENT.md` – canonical secrets inventory and rotation guidance  
> - `docs/ENVIRONMENT_VARIABLES.md` – authoritative list of env vars and types  
> - `docs/OPERATIONS_DB.md` – database operations and migration workflow  
> - `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md` and `docs/SECURITY_THREAT_MODEL.md` – supply‑chain and security posture  
>  
> **Precedence:** Env schema/validation (`src/server/config/env.ts`, `src/server/config/unified.ts`), secrets validation (`src/server/utils/secretsValidation.ts`), and deployment manifests (Docker/Kubernetes) are authoritative. If this runbook conflicts with them, **code + configs win** and this document must be updated.

---

## 1. Goal & Scope

This drill is designed to **prove that secrets can be safely rotated** without surprises, starting in **staging** or an equivalent non‑production environment.

The focus is on:

- `JWT_SECRET` and `JWT_REFRESH_SECRET` – access/refresh token signing keys.
- `DATABASE_URL` – database user/password component.

For production, treat this runbook as a template and adapt it to your secret manager and change‑management process.

> **Safety principle:** Perform the full drill in staging first. Only consider production after you are confident in the procedure and have sign‑off from your security/ops owners.

---

## 2. Preconditions

- You are operating in **staging** (or a non‑production stack) with:
  - App + Postgres running via `docker-compose.yml` + `docker-compose.staging.yml`, or an equivalent deployment.
  - Access to the secrets store used for that environment (for example `.env.staging`, Docker/Kubernetes secrets, or a managed secret manager).
- You understand the impact:
  - Rotating JWT keys will invalidate existing sessions (users must re‑login).
  - Rotating DB credentials may cause short‑lived connection errors during switch‑over.

Before starting, skim:

- `docs/SECRETS_MANAGEMENT.md` – especially “Secret Rotation Procedures”.
- `docs/OPERATIONS_DB.md` – database expectations and backup patterns.

---

## 3. JWT Secret Rotation Drill (staging)

This drill rotates `JWT_SECRET` and `JWT_REFRESH_SECRET` in staging using a simple, single‑key pattern.

### 3.1 Generate new keys

On your workstation or staging host:

```bash
NEW_JWT_SECRET=$(openssl rand -base64 48)
NEW_JWT_REFRESH_SECRET=$(openssl rand -base64 48)
echo "JWT_SECRET=${NEW_JWT_SECRET}"
echo "JWT_REFRESH_SECRET=${NEW_JWT_REFRESH_SECRET}"
```

Store these values in a secure temporary location (for example an encrypted notes tool) until they are injected into the environment.

### 3.2 Update staging secrets

Update the staging secrets store in one of the following ways:

- **Env file–based staging** (`.env.staging`):

  ```bash
  # Edit .env.staging and replace the JWT entries
  JWT_SECRET="...NEW_JWT_SECRET..."
  JWT_REFRESH_SECRET="...NEW_JWT_REFRESH_SECRET..."
  ```

- **Docker/Kubernetes secrets**: update the relevant secret entries (for example `jwt_secret`, `jwt_refresh_secret`) via your platform tooling, keeping the variable names `JWT_SECRET` and `JWT_REFRESH_SECRET`.

Do **not** commit real secrets to Git; ensure `.env`/`.env.staging` remain in `.gitignore`.

### 3.3 Restart the app

Apply the change and restart the app in staging:

```bash
# Docker Compose example
docker compose -f docker-compose.yml -f docker-compose.staging.yml up -d --build
```

Or, for Kubernetes:

```bash
kubectl rollout restart deployment/ringrift-api
```

### 3.4 Verify behaviour

1. Confirm startup and config validation:

   ```bash
   curl -s http://localhost:3000/health | jq
   curl -s http://localhost:3000/ready | jq
   ```

2. Attempt to use an **existing** session/cookie:
   - Expect it to be rejected (forced logout or auth failure).

3. Log in again and create a fresh session:
   - Confirm new login works.
   - Create a game and play a few moves.

4. Check logs for repeated auth/crypto errors:

   ```bash
   docker compose logs -f app | grep -i "jwt" | tail -n 50
   ```

If health checks pass, new logins work, and no repeated JWT errors are present, the JWT rotation drill is considered successful for staging.

---

## 4. Database Credential Rotation Drill (staging)

This drill rotates the database user/password used by the app by introducing a new user, switching the app over, then retiring the old user.

> **Note:** Adjust usernames/roles to match your staging configuration from `docs/OPERATIONS_DB.md` and `docs/ENVIRONMENT_VARIABLES.md`.

### 4.1 Create a new DB user

Connect to the staging Postgres instance (for Compose‑based stacks):

```bash
docker compose exec postgres psql -U postgres -d ringrift
```

From the `psql` shell:

```sql
-- Use a distinct username to avoid confusion
CREATE USER ringrift_staging_new WITH PASSWORD 'new_secure_password';

GRANT ALL PRIVILEGES ON DATABASE ringrift TO ringrift_staging_new;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ringrift_staging_new;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ringrift_staging_new;
```

For managed Postgres, use the provider console/CLI to create an equivalent user.

### 4.2 Update `DATABASE_URL` in staging

Construct a new `DATABASE_URL` using the new user:

```bash
export NEW_DATABASE_URL="postgresql://ringrift_staging_new:new_secure_password@postgres:5432/ringrift"
echo "${NEW_DATABASE_URL}"
```

Update the staging secrets store (for example `.env.staging`, Docker/Kubernetes secret) to use `NEW_DATABASE_URL` as the value for `DATABASE_URL`.

### 4.3 Restart the app and verify DB connectivity

Restart the app as in §3.3, then:

```bash
curl -s http://localhost:3000/health | jq
curl -s http://localhost:3000/ready | jq
```

Exercise basic flows:

- Register/login a test user.
- Create a game, join, and play several moves.

Check app logs for database errors:

```bash
docker compose logs -f app | grep -i "database" | tail -n 50
```

If health checks and game flows work normally, the app is successfully using the new DB credentials.

### 4.4 Retire the old DB user (optional, after validation)

Once you are confident the new user is working and no other services depend on the old user, you can retire the old user (example assumes `ringrift` was the previous staging user):

```sql
REVOKE ALL PRIVILEGES ON DATABASE ringrift FROM ringrift;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM ringrift;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM ringrift;
-- Optionally:
-- DROP USER ringrift;
```

Only perform this step in staging once you are sure no other components still use the old credentials.

---

## 5. Validation Checklist

- [ ] New JWT secrets configured in staging and validated by app startup.
- [ ] Existing sessions invalidated as expected; new logins succeed.
- [ ] No repeated JWT or crypto errors in logs after rotation.
- [ ] New DB user created with appropriate privileges.
- [ ] `DATABASE_URL` updated to use the new user/password.
- [ ] Health checks pass and core game flows work after DB rotation.
- [ ] (Optional) Old DB user privileges revoked or user dropped after validation.

Record the date, environment, and any issues found in your internal security/ops log. Aim to perform this drill **periodically** (for example quarterly) and before any public launch.

---

## 6. Notes for Production Adaptation

For production environments:

- Prefer a managed secret manager (for example AWS Secrets Manager, HashiCorp Vault) and use its rotation workflows.
- Coordinate rotations with:
  - Shortened token lifetimes where necessary.
  - Planned maintenance windows or blue/green deployments.
- For JWT keys, consider implementing the **key‑versioning** pattern described in `docs/SECRETS_MANAGEMENT.md` (“Zero‑Downtime Rotation (Advanced)”) before attempting live rotation.
- Ensure database user changes are compatible with your migration/deployment workflows in `docs/OPERATIONS_DB.md` and `docs/runbooks/DATABASE_MIGRATION.md`.

The same core principle applies as with backups: rotations should be **practised and boring** in staging before they are attempted in production.

