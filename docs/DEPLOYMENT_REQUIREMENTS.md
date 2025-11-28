# RingRift Deployment Requirements

> **Doc Status (2025-11-27): Active**
>
> **Role:** Canonical deployment requirements and environment-configuration guide for RingRift across development, staging, and production. Covers infra prerequisites, topology flags, health checks, resource limits, and validation/monitoring expectations for operators.
>
> **Not a semantics SSoT:** This document does not define game rules or lifecycle semantics. Rules semantics are owned by the shared TypeScript rules engine under `src/shared/engine/**` plus contracts and vectors (see `RULES_CANONICAL_SPEC.md`, `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `docs/RULES_ENGINE_SURFACE_AUDIT.md`). Lifecycle semantics are owned by `docs/CANONICAL_ENGINE_API.md` together with shared types/schemas in `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, and `src/shared/validation/websocketSchemas.ts`.
>
> **Related docs:** `docs/ENVIRONMENT_VARIABLES.md`, `docs/SECRETS_MANAGEMENT.md`, `docs/OPERATIONS_DB.md`, `docs/SECURITY_THREAT_MODEL.md`, `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`, `docs/ALERTING_THRESHOLDS.md`, and `DOCUMENTATION_INDEX.md`.

This document outlines the requirements and configuration for deploying RingRift in different environments.

## Table of Contents

- [Environment Overview](#environment-overview)
- [Development Environment](#development-environment)
- [Staging Environment](#staging-environment)
- [Production Environment](#production-environment)
- [Service Dependencies](#service-dependencies)
- [Environment Variables](#environment-variables)
- [Health Checks](#health-checks)
- [Resource Limits](#resource-limits)
- [Security Requirements](#security-requirements)
- [Validation](#validation)

---

## Environment Overview

| Environment | `NODE_ENV`    | Purpose                | Secret Handling        |
| ----------- | ------------- | ---------------------- | ---------------------- |
| Development | `development` | Local development      | Placeholder secrets OK |
| Staging     | `production`  | Pre-production testing | Real secrets required  |
| Production  | `production`  | Live application       | Secrets from vault     |
| Test        | `test`        | Automated testing      | In-memory/mocked       |

---

## Development Environment

### Prerequisites

- **Node.js**: v22+ (as specified in package.json engines)
- **npm**: v9+
- **Docker**: v20+ (for containerized services)
- **Docker Compose**: v2+

### Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd ringrift
npm install

# 2. Set up environment
cp .env.example .env
# Edit .env with local settings if needed

# 3. Start database (via Docker)
docker-compose up -d postgres redis

# 4. Run migrations
npm run db:migrate

# 5. Start development server
npm run dev
```

### Optional Services

| Service    | Required | Fallback Behavior                           |
| ---------- | -------- | ------------------------------------------- |
| PostgreSQL | Yes      | No fallback - required for data persistence |
| Redis      | No       | In-memory rate limiting and caching         |
| AI Service | No       | Local heuristic-based AI moves              |

### Development-Specific Settings

```env
NODE_ENV=development
LOG_FORMAT=pretty
LOG_LEVEL=debug
AI_FALLBACK_ENABLED=true
```

---

## Staging Environment

### Prerequisites

- All development prerequisites
- Access to staging infrastructure
- Real secrets (not placeholders)

### Deployment

```bash
# Use staging compose file (extends base docker-compose.yml)
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
```

### Requirements

1. **Database**: PostgreSQL with health check
2. **Redis**: With health check and optional authentication
3. **AI Service**: Running and healthy
4. **Resource Limits**: Configured per service (see below)

### Staging-Specific Settings

```env
NODE_ENV=production
LOG_FORMAT=json
LOG_LEVEL=info

# Real secrets - these MUST NOT be placeholder values
JWT_SECRET=<real-32+-char-secret>
JWT_REFRESH_SECRET=<real-32+-char-secret>
DB_PASSWORD=<real-database-password>
```

### Service Health Checks

All services must pass health checks before the app starts:

| Service    | Health Check                                   | Timeout     |
| ---------- | ---------------------------------------------- | ----------- |
| postgres   | `pg_isready -U $POSTGRES_USER -d $POSTGRES_DB` | 30s startup |
| redis      | `redis-cli ping`                               | 10s startup |
| ai-service | HTTP `/health`                                 | 20s startup |
| app        | HTTP `/health`                                 | 5s startup  |

---

## Production Environment

### Additional Prerequisites

- TLS certificates
- Secrets management solution (Kubernetes Secrets, AWS Secrets Manager, HashiCorp Vault)
- Load balancer for multi-instance deployments
- Monitoring infrastructure

### Architecture

```
                    ┌──────────────────┐
                    │   Load Balancer  │
                    │   (TLS Term.)    │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
     ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
     │   App #1    │  │   App #2    │  │   App #N    │
     │  (Backend)  │  │  (Backend)  │  │  (Backend)  │
     └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     │                       │                       │
┌────▼────┐           ┌──────▼──────┐          ┌─────▼─────┐
│  Redis  │           │ PostgreSQL  │          │ AI Service│
│ (Cache) │           │ (Primary)   │          │           │
└─────────┘           └─────────────┘          └───────────┘
```

### Production-Specific Requirements

1. **Single Instance Mode (Default)**
   - `RINGRIFT_APP_TOPOLOGY=single`
   - All game state in single instance memory
   - Simple deployment, no session affinity needed

2. **Multi-Instance Mode (Advanced)**
   - `RINGRIFT_APP_TOPOLOGY=multi-sticky`
   - **REQUIRES** infrastructure-enforced sticky sessions
   - Load balancer must route all requests from same client to same backend
   - WebSocket connections must be sticky

### Secrets Management

**CRITICAL**: Production MUST NOT use placeholder secrets.

```env
# These will cause startup failure in production:
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production  # ❌ REJECTED
JWT_SECRET=Tg7k3X9p2Qm8Yw6Rh4Bv1Fn5Jc0LdSeAiUo...             # ✅ OK
```

Generate production secrets:

```bash
# JWT secrets (minimum 32 characters)
openssl rand -base64 48

# Database password
openssl rand -base64 32
```

### Resource Limits

| Service    | Memory Limit | Memory Reservation | CPU (if applicable)   |
| ---------- | ------------ | ------------------ | --------------------- |
| app        | 512MB        | 256MB              | Based on load testing |
| postgres   | 256MB        | 128MB              | Based on load testing |
| redis      | 128MB        | 64MB               | -                     |
| ai-service | 512MB        | 256MB              | Based on load testing |
| prometheus | 512MB        | 256MB              | -                     |
| grafana    | 256MB        | 128MB              | -                     |

---

## Service Dependencies

### Startup Order

```
postgres (healthy) ──┐
                     ├──► app
redis (healthy) ─────┤
                     │
ai-service (healthy) ┘
```

### Internal DNS (Docker)

| Service    | Internal URL                                                |
| ---------- | ----------------------------------------------------------- |
| PostgreSQL | `postgresql://ringrift:$DB_PASSWORD@postgres:5432/ringrift` |
| Redis      | `redis://redis:6379`                                        |
| AI Service | `http://ai-service:8001`                                    |

---

## Environment Variables

### Required in All Environments

| Variable       | Description                                              |
| -------------- | -------------------------------------------------------- |
| `NODE_ENV`     | Environment mode: development, staging, production, test |
| `DATABASE_URL` | PostgreSQL connection string                             |

### Required in Production/Staging

| Variable             | Description                                          |
| -------------------- | ---------------------------------------------------- |
| `JWT_SECRET`         | JWT signing secret (min 32 chars, no placeholders)   |
| `JWT_REFRESH_SECRET` | Refresh token secret (min 32 chars, no placeholders) |
| `REDIS_URL`          | Redis connection string                              |
| `AI_SERVICE_URL`     | AI service base URL                                  |

### Optional with Defaults

See [`.env.example`](../.env.example) for full list with descriptions.

---

## Health Checks

### Application Health Endpoint

```
GET /health
```

Response (200 OK):

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 12345,
  "checks": {
    "database": "ok",
    "redis": "ok",
    "aiService": "ok"
  }
}
```

### Docker Health Check

Defined in Dockerfile:

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1) })"
```

---

## Resource Limits

### Docker Compose Resource Configuration

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
```

### Kubernetes Resource Configuration

```yaml
resources:
  requests:
    memory: '256Mi'
    cpu: '100m'
  limits:
    memory: '512Mi'
    cpu: '500m'
```

---

## Security Requirements

### TLS/SSL

- All production traffic MUST use HTTPS
- Use nginx or load balancer for TLS termination
- Redirect HTTP to HTTPS

### Network Isolation

- Database should not be directly accessible from internet
- AI service should not be directly accessible from internet
- Use internal Docker network for service-to-service communication

### Secret Rotation

1. **JWT Secrets**: Rotation invalidates all sessions (users must re-login)
2. **Database Password**: Create new user → Update env → Restart → Revoke old user
3. **Redis Password**: Update Redis config → Update env → Restart

See [`SECRETS_MANAGEMENT.md`](./SECRETS_MANAGEMENT.md) for detailed procedures.

---

## Validation

### Automated Validation

Run deployment config validation before deploying:

```bash
npm run validate:deployment
```

This checks:

- All env vars in docker-compose are documented in .env.example
- No hardcoded secrets in compose files
- Health checks are configured
- Resource limits are set
- Volumes are properly defined

### CI Integration

Validation runs automatically on push/PR via `.github/workflows/validate-config.yml`.

### Manual Checklist

Before any production deployment:

- [ ] All secrets are real (not placeholders)
- [ ] Health checks pass for all services
- [ ] Resource limits are configured
- [ ] TLS is configured
- [ ] Backup strategy is in place
- [ ] Monitoring is configured
- [ ] Rollback plan is documented

---

## Monitoring Stack (Optional)

Enable monitoring with Docker Compose profiles:

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Or set in environment
export COMPOSE_PROFILES=monitoring
docker-compose up -d
```

### Monitoring Services

| Service      | Port | Purpose            |
| ------------ | ---- | ------------------ |
| Prometheus   | 9090 | Metrics collection |
| Alertmanager | 9093 | Alert routing      |
| Grafana      | 3002 | Dashboards         |

See [`monitoring/`](../monitoring/) for configuration files.

---

## Troubleshooting

### Common Issues

1. **App fails to start in production**
   - Check for placeholder secrets in environment
   - Verify DATABASE_URL is correct
   - Check database health

2. **AI service connection refused**
   - Verify AI_SERVICE_URL is correct
   - Check ai-service health
   - Enable AI_FALLBACK_ENABLED=true for graceful degradation

3. **Health check fails**
   - Increase start_period in health check config
   - Check service logs for errors
   - Verify port mappings

### Useful Commands

```bash
# View service logs
docker-compose logs -f app

# Check service health
docker-compose ps

# Run validation
npm run validate:deployment

# Interactive shell in container
docker-compose exec app sh
```
