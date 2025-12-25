# Secrets Management Guide

> **Doc Status (2025-12-13): Active**
>
> **Role:** Canonical guide for how RingRift manages application secrets across environments (development, staging, production), including required environment variables, validation, rotation, and operational procedures.
>
> **Not a semantics SSoT:** This document does not define game rules or lifecycle semantics. Rules semantics are owned by the shared TypeScript rules engine under `src/shared/engine/**` plus contracts and vectors (see `RULES_CANONICAL_SPEC.md`, `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`). Lifecycle semantics are owned by `docs/CANONICAL_ENGINE_API.md` together with shared types/schemas in `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, and `src/shared/validation/websocketSchemas.ts`.
>
> **Related docs:** `docs/ENVIRONMENT_VARIABLES.md`, `docs/DEPLOYMENT_REQUIREMENTS.md`, `docs/SECURITY_THREAT_MODEL.md`, `docs/OPERATIONS_DB.md`, `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`, and `DOCUMENTATION_INDEX.md`.

This document describes best practices for managing secrets in RingRift deployments.

## Table of Contents

- [Overview](#overview)
- [Secrets Inventory](#secrets-inventory)
- [Development Environment](#development-environment)
- [Production Environment](#production-environment)
- [Secret Rotation Procedures](#secret-rotation-procedures)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

RingRift uses several types of secrets:

1. **Database credentials** - PostgreSQL connection strings
2. **Authentication secrets** - JWT signing keys
3. **Cache credentials** - Redis passwords (optional)
4. **Service URLs** - AI service endpoints

### Secret Categories

| Secret               | Required (Prod) | Required (Dev) | Min Length | Purpose               |
| -------------------- | --------------- | -------------- | ---------- | --------------------- |
| `JWT_SECRET`         | ✅ Yes          | ❌ No          | 32 chars   | Access token signing  |
| `JWT_REFRESH_SECRET` | ✅ Yes          | ❌ No          | 32 chars   | Refresh token signing |
| `DATABASE_URL`       | ✅ Yes          | ❌ No          | -          | PostgreSQL connection |
| `REDIS_URL`          | ✅ Yes          | ❌ No          | -          | Redis connection      |
| `REDIS_PASSWORD`     | ❌ No           | ❌ No          | 8 chars    | Redis authentication  |
| `AI_SERVICE_URL`     | ✅ Yes          | ❌ No          | -          | AI service endpoint   |

## Development Environment

### Quick Start

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. The default values work for local development with Docker:
   ```bash
   docker-compose up -d postgres redis
   npm run dev
   ```

### Default Development Secrets

In development mode (`NODE_ENV=development`), the application uses fallback secrets:

- `JWT_SECRET` defaults to `dev-access-token-secret`
- `JWT_REFRESH_SECRET` defaults to `dev-refresh-token-secret`
- `DATABASE_URL` is optional (uses local PostgreSQL)
- `REDIS_URL` defaults to `redis://localhost:6379`

**⚠️ Warning:** These defaults are detected and rejected in production mode.

## Production Environment

### Requirements

In production (`NODE_ENV=production`), the following are **mandatory**:

1. **JWT_SECRET** - Minimum 32 characters, cryptographically random
2. **JWT_REFRESH_SECRET** - Minimum 32 characters, different from JWT_SECRET
3. **DATABASE_URL** - Valid PostgreSQL connection string
4. **REDIS_URL** - Valid Redis connection string
5. **AI_SERVICE_URL** - Valid URL to AI service

### Generating Secure Secrets

```bash
# Generate a 48-character base64 secret (recommended)
openssl rand -base64 48

# Alternative using Node.js
node -e "console.log(require('crypto').randomBytes(48).toString('base64'))"

# Alternative using Python
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

### Validation

The application validates secrets at startup. In production, it will:

1. **Fail fast** if required secrets are missing
2. **Reject placeholder values** from `.env.example` or `docker-compose.yml`
3. **Enforce minimum length** for JWT secrets (32+ characters)
4. **Validate URL format** for service URLs

### Recommended Secrets Management Solutions

For production deployments, consider:

- **Kubernetes Secrets** - Native secret management for K8s
- **AWS Secrets Manager** - AWS-native with rotation support
- **HashiCorp Vault** - Enterprise-grade secret management
- **Docker Swarm Secrets** - For Docker Swarm deployments

**Current Infrastructure Status (2025-12-13):**

- AWS infrastructure exists for AI training (ECS, SQS, S3) in `ai-service/infrastructure/terraform/`
- AWS Secrets Manager is **not yet configured** in terraform
- For production deployment, add AWS Secrets Manager resources to the terraform configuration
- See `ai-service/infrastructure/README.md` for existing AWS setup

Example with Docker Secrets:

```yaml
services:
  app:
    secrets:
      - jwt_secret
      - jwt_refresh_secret
    environment:
      - JWT_SECRET_FILE=/run/secrets/jwt_secret
      - JWT_REFRESH_SECRET_FILE=/run/secrets/jwt_refresh_secret
```

## Secret Rotation Procedures

### Secrets Rotation Drill (staging)

For an operator-focused, step-by-step practice drill in **staging** or another non-production environment, use the runbook [`docs/runbooks/SECRETS_ROTATION_DRILL.md`](../runbooks/SECRETS_ROTATION_DRILL.md). The drill applies the JWT and database patterns below in a **PASS-style** structure (Purpose, Preconditions, Actions, Signals) and is intended to be boring and repeatable.

**Scope (initial drill):**

- **Environment:** `NODE_ENV=production` staging stack using `docker-compose.yml` + `docker-compose.staging.yml` or an equivalent deployment topology.
- **Secrets:**
  - `JWT_SECRET` and `JWT_REFRESH_SECRET` (token signing keys).
  - The credential component of `DATABASE_URL` (database user/password).
- **Out of scope for the first drill, but covered by the patterns below:**
  - `REDIS_PASSWORD`
  - Any future third-party API keys or service credentials.

**Preconditions:**

- **Backups exist and are tested** for the target environment:
  - See [`DATA_LIFECYCLE_AND_PRIVACY.md`](../security/DATA_LIFECYCLE_AND_PRIVACY.md) §3.5 (Backups and offline copies).
  - See [`OPERATIONS_DB.md`](OPERATIONS_DB.md) and the `DATABASE_BACKUP_AND_RESTORE_DRILL` runbook (`docs/runbooks/DATABASE_BACKUP_AND_RESTORE_DRILL.md`) for concrete backup/restore procedures. That runbook is the **canonical operational drill** for validating database credential changes together with backup/restore flows in staging or other non‑production environments.
- **Monitoring and alerts are active**:
  - Prometheus + Alertmanager and Grafana dashboards from [`monitoring/`](../../monitoring).
  - See [`DEPLOYMENT_REQUIREMENTS.md`](../planning/DEPLOYMENT_REQUIREMENTS.md) “Monitoring Stack” and “Validation”.
- **Config and manifests pass validation** for the staged configuration:
  - From the project root, run `npm run validate:deployment` to execute [`scripts/validate-deployment-config.ts`](../../scripts/validate-deployment-config.ts).
  - Optionally run `./scripts/validate-monitoring-configs.sh` (from the project root) to validate Prometheus/Alertmanager configs before and after the drill.

**High-level drill flow (staging):**

1. **Plan**
   - Identify the environment/cluster and exact secret set you will rotate.
   - Generate new secret values using the commands in this document or [`.env.example`](../../.env.example).
   - Stage the new values in the secret store for **staging only** (for example `.env.staging`, Docker/Kubernetes secrets). Do not change production secrets as part of the drill.
2. **Apply / Rotate**
   - Update the secrets in the staging secret store.
   - Restart or redeploy the app so the new values are picked up (for example `docker compose -f docker-compose.yml -f docker-compose.staging.yml up -d --build` or `kubectl rollout restart deployment/ringrift-api`).
3. **Seal / Verify**
   - Run preflight checks:
     - `npm run validate:deployment`
     - Health and readiness probes: `GET /health`, `GET /ready`.
   - Run targeted functional checks:
     - `./scripts/test-auth.sh` (or an equivalent auth smoke test) against the staging base URL.
     - Optional load/invariants smoke such as `npm run load:orchestrator:smoke` to exercise the orchestrator under light load.
   - Watch monitoring during and after rotation:
     - System/HTTP error-rate and latency panels in Grafana (`system-health`, `game-performance`, `rules-correctness` dashboards).
     - Relevant alerts defined in `monitoring/prometheus/alerts.yml` (for example auth error rate or database connectivity alerts).
4. **Steady-state / Rollback**
   - If any step fails, roll back to the previous known-good secret values and restart the app.
   - Capture findings (what broke, how it manifested, how you diagnosed it) in your internal ops/security log so the next drill is faster and safer.

The subsections below describe the individual rotation mechanics (JWT, database, Redis). The drill runbook applies them in a concrete, repeatable sequence for staging.

### JWT Secret Rotation

**Impact:** All existing sessions will be invalidated. Users must re-login.

**Steps:**

1. Generate new secrets:

   ```bash
   JWT_SECRET=$(openssl rand -base64 48)
   JWT_REFRESH_SECRET=$(openssl rand -base64 48)
   ```

2. Update secrets in your secrets management system

3. Restart the application:

   ```bash
   # Docker Compose
   docker-compose restart app

   # Kubernetes
   kubectl rollout restart deployment/ringrift-app
   ```

4. Monitor for authentication errors in logs

**Zero-Downtime Rotation (Advanced):**

For zero-downtime JWT rotation, implement key versioning:

1. Add new key with a version identifier
2. Configure app to accept both old and new keys for verification
3. Sign new tokens with new key only
4. Wait for old tokens to expire (refresh token lifetime)
5. Remove old key

### Database Password Rotation

**Impact:** Brief connection disruption during restart.

**Steps:**

1. Create new database user (or update password):

   ```sql
   -- Option A: Create new user
   CREATE USER ringrift_new WITH PASSWORD 'new_secure_password';
   GRANT ALL PRIVILEGES ON DATABASE ringrift TO ringrift_new;
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ringrift_new;

   -- Option B: Update existing password
   ALTER USER ringrift WITH PASSWORD 'new_secure_password';
   ```

2. Update `DATABASE_URL` with new credentials

3. Restart the application

4. Verify database connectivity:

   ```bash
   curl http://localhost:3000/health
   ```

5. (If using Option A) Revoke old user after confirming:
   ```sql
   REVOKE ALL PRIVILEGES ON DATABASE ringrift FROM ringrift_old;
   DROP USER ringrift_old;
   ```

### Redis Password Rotation

**Impact:** Brief cache disruption during restart.

**Steps:**

1. Update Redis password:

   ```bash
   redis-cli CONFIG SET requirepass "new_secure_password"
   ```

2. Update `REDIS_PASSWORD` environment variable

3. Restart the application

4. Verify Redis connectivity

**Note:** Redis password changes via CONFIG SET are not persisted. Also update your `redis.conf` or Docker Compose configuration.

## Security Best Practices

### Do's ✅

- **Use strong, unique secrets** for each environment
- **Rotate secrets regularly** (quarterly minimum, monthly recommended)
- **Use secrets management tools** in production
- **Monitor for secret exposure** in logs and error messages
- **Encrypt secrets at rest** in CI/CD systems
- **Use environment variables** rather than config files for secrets
- **Implement secret rotation procedures** before go-live

### Don'ts ❌

- **Never commit secrets** to version control
- **Never log secrets** even partially masked
- **Never share secrets** across environments
- **Never use placeholder values** in production
- **Never hardcode secrets** in source code
- **Never expose secrets** in error messages or stack traces
- **Never include secrets** in health check responses

### Audit Checklist

- [ ] All secrets are stored in a secrets management system
- [ ] `.env` file is in `.gitignore`
- [ ] No secrets in `docker-compose.yml` (use variable interpolation)
- [ ] No secrets in `.env.staging` or `.env.example`
- [ ] Secret rotation procedures documented and tested
- [ ] Monitoring alerts for authentication failures
- [ ] Secrets access is logged and auditable

## Troubleshooting

### "JWT secrets must not use placeholder values"

**Cause:** You're running in production mode with placeholder secrets from `.env.example`.

**Solution:** Generate new, unique secrets:

```bash
export JWT_SECRET=$(openssl rand -base64 48)
export JWT_REFRESH_SECRET=$(openssl rand -base64 48)
```

### "JWT_SECRET must be at least 32 characters"

**Cause:** Your JWT secret is too short for production security.

**Solution:** Generate a longer secret (48+ characters recommended):

```bash
openssl rand -base64 48
```

### "DATABASE_URL is required when NODE_ENV=production"

**Cause:** Missing database connection string in production.

**Solution:** Set the `DATABASE_URL` environment variable:

```bash
export DATABASE_URL="postgresql://user:password@host:5432/ringrift"
```

### "Secrets validation failed"

**Cause:** One or more secrets failed validation.

**Solution:** Check the error message for specific issues. Common causes:

- Missing required secret
- Placeholder value detected
- Secret too short
- Invalid URL format

View the full error details:

```bash
NODE_ENV=production node -e "require('./dist/server/config')"
```

## Related Documentation

- [`.env.example`](../../.env.example) - Environment variable template
- [`src/server/config.ts`](../../src/server/config.ts) - Configuration loading
- [`src/server/utils/secretsValidation.ts`](../../src/server/utils/secretsValidation.ts) - Validation logic
- [SECURITY_THREAT_MODEL.md](../security/SECURITY_THREAT_MODEL.md) - Security analysis
