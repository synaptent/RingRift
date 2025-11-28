# Initial Deployment Runbook

## Overview

This runbook covers the first-time deployment of RingRift to a new environment. Use this when setting up staging or production infrastructure from scratch.

**Estimated Time**: 45-60 minutes  
**Required Access**: SSH, Docker registry, secrets manager, DNS management

---

## Prerequisites

### Infrastructure Requirements

- [ ] Docker Engine v20+ installed on target host(s)
- [ ] Docker Compose v2+ available
- [ ] PostgreSQL instance provisioned (managed or self-hosted)
- [ ] Redis instance provisioned (optional for single-node, required for multi-node)
- [ ] SSL/TLS certificates obtained (production only)
- [ ] DNS records configured or planned

### Access Requirements

- [ ] SSH access to target server(s)
- [ ] Docker registry credentials (if using private registry)
- [ ] Database connection credentials
- [ ] Secrets manager access (production)
- [ ] DNS management access

### Files Prepared

- [ ] `.env` file with environment-specific values
- [ ] SSL certificates (for production)
- [ ] nginx.conf (if using nginx)

---

## Pre-Deployment Checklist

### 1. Verify Infrastructure

```bash
# SSH to target server
ssh user@target-server

# Verify Docker version
docker --version
# Expected: Docker version 20.x or higher

# Verify Docker Compose
docker compose version
# Expected: Docker Compose version v2.x

# Verify disk space (need at least 10GB free)
df -h /var/lib/docker
```

### 2. Generate Secrets

Generate secure secrets for production/staging. **Never use placeholder values**.

```bash
# Generate JWT secret (minimum 32 characters)
openssl rand -base64 48
# Save output for JWT_SECRET

# Generate JWT refresh secret (different from above)
openssl rand -base64 48
# Save output for JWT_REFRESH_SECRET

# Generate database password
openssl rand -base64 32
# Save output for DB_PASSWORD
```

### 3. Prepare Environment File

```bash
# Clone repository or copy files to server
# Replace with your organization's repository URL
git clone https://github.com/your-org/ringrift.git /opt/ringrift
cd /opt/ringrift

# Create environment file from template
cp .env.example .env

# Edit with production values
nano .env
```

**Critical variables to configure:**

| Variable             | Description                | Notes                         |
| -------------------- | -------------------------- | ----------------------------- |
| `NODE_ENV`           | Environment mode           | `production` for staging/prod |
| `DATABASE_URL`       | Postgres connection string | Include real password         |
| `JWT_SECRET`         | JWT signing key            | Generated above               |
| `JWT_REFRESH_SECRET` | Refresh token key          | Generated above               |
| `REDIS_URL`          | Redis connection string    | Required for production       |
| `AI_SERVICE_URL`     | AI service endpoint        | `http://ai-service:8001`      |

### 4. Validate Configuration

```bash
# Run deployment validation
npm run validate:deployment

# Expected output: ✅ Validation PASSED
```

---

## Deployment Steps

### Step 1: Create Required Directories

```bash
cd /opt/ringrift

# Create data directories
mkdir -p uploads logs backups ssl

# Set permissions
chmod 755 uploads logs backups
chmod 700 ssl  # Restrict SSL directory
```

### Step 2: Configure SSL (Production Only)

```bash
# Copy certificates to ssl directory
cp /path/to/fullchain.pem ssl/
cp /path/to/privkey.pem ssl/

# Set restrictive permissions
chmod 600 ssl/*.pem
```

### Step 3: Build Docker Images

```bash
# For staging (builds locally)
docker compose build

# For production (pull from registry)
docker compose pull
```

### Step 4: Start Database Services First

```bash
# Start PostgreSQL and Redis
docker compose up -d postgres redis

# Wait for services to be healthy
docker compose ps

# Verify PostgreSQL is ready
docker compose exec postgres pg_isready -U ringrift -d ringrift
# Expected: /var/run/postgresql:5432 - accepting connections

# Verify Redis is ready
docker compose exec redis redis-cli ping
# Expected: PONG
```

### Step 5: Initialize Database

```bash
# Run database migrations
docker compose run --rm app npx prisma migrate deploy

# Verify migration status
docker compose run --rm app npx prisma migrate status
# Expected: All migrations applied
```

**Verification:**

```bash
# Check tables exist
docker compose exec postgres psql -U ringrift -d ringrift -c '\dt'
# Expected: List of tables including users, games, moves, etc.
```

### Step 6: Start AI Service

```bash
# Start AI service
docker compose up -d ai-service

# Wait for health check
sleep 20

# Verify AI service is healthy
curl -s http://localhost:8001/health | jq
# Expected: {"status": "healthy", ...}
```

### Step 7: Start Main Application

```bash
# For development/staging
docker compose up -d app

# For staging with overrides
docker compose -f docker-compose.yml -f docker-compose.staging.yml up -d

# Wait for application startup
sleep 30
```

### Step 8: Start Nginx (Production)

```bash
# Verify nginx.conf exists
ls -la nginx.conf

# Start nginx
docker compose up -d nginx

# Verify nginx is running
docker compose ps nginx
```

### Step 9: Optional - Start Monitoring Stack

```bash
# Start Prometheus, Grafana, Alertmanager
docker compose --profile monitoring up -d

# Verify monitoring services
docker compose ps prometheus grafana alertmanager
```

---

## Post-Deployment Verification

### Health Check Verification

```bash
# Check application health
curl -s http://localhost:3000/health | jq
# Expected:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "uptime": <seconds>
# }

# Check application readiness
curl -s http://localhost:3000/ready | jq
# Expected:
# {
#   "status": "healthy",
#   "checks": {
#     "database": { "status": "healthy", "latency": <ms> },
#     "redis": { "status": "healthy", "latency": <ms> },
#     "aiService": { "status": "healthy", "latency": <ms> }
#   }
# }
```

### Service Connectivity Tests

```bash
# Test database connection from app
docker compose exec app node -e "
const { PrismaClient } = require('@prisma/client');
const p = new PrismaClient();
p.\$queryRaw\`SELECT 1\`.then(() => console.log('DB OK')).catch(console.error);
"

# Test Redis connection from app
docker compose exec app node -e "
const Redis = require('ioredis');
const r = new Redis(process.env.REDIS_URL);
r.ping().then(console.log).catch(console.error);
"
```

### Functional Tests

```bash
# Test API endpoint
curl -s http://localhost:3000/api/health | jq

# Test user registration (creates test user)
curl -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","username":"testuser","password":"TestPass123!"}' | jq

# Test login
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"TestPass123!"}' | jq
```

### WebSocket Test

```bash
# Install wscat if not available
npm install -g wscat

# Test WebSocket connection
wscat -c ws://localhost:3001
# Should connect without errors
# Type "ping" and expect a response
```

### Full Verification Checklist

- [ ] `/health` returns 200 with status "healthy"
- [ ] `/ready` returns 200 with all checks passing
- [ ] Database queries execute successfully
- [ ] Redis commands work
- [ ] AI service responds to health checks
- [ ] User registration works
- [ ] User login works
- [ ] WebSocket connections establish
- [ ] SSL/TLS works (production)
- [ ] DNS resolves correctly (production)

---

## Rollback Procedure

If initial deployment fails, clean up with:

```bash
# Stop all services
docker compose down

# Remove volumes if database needs recreation
docker compose down -v

# Review logs for issues
docker compose logs > deployment-failure.log

# Fix issues and retry from Step 4
```

---

## Troubleshooting

### Database connection refused

```bash
# Check PostgreSQL is running
docker compose ps postgres

# Check PostgreSQL logs
docker compose logs postgres

# Verify DATABASE_URL format
echo $DATABASE_URL
# Should be: postgresql://user:password@host:5432/database

# Test connection manually
docker compose exec postgres psql -U ringrift -d ringrift -c "SELECT 1"
```

### AI Service won't start

```bash
# Check AI service logs
docker compose logs ai-service

# Verify Python dependencies
docker compose exec ai-service pip list

# Check port binding
netstat -tlnp | grep 8001
```

### Application crashes on startup

```bash
# Check for secret validation errors
docker compose logs app | grep -i secret

# If secrets rejected, verify they are not placeholder values
# Generate new secrets and update .env

# Check for missing environment variables
docker compose logs app | grep -i "required"
```

### Port already in use

```bash
# Find process using port
lsof -i :3000

# Kill conflicting process or change port in .env
kill -9 <PID>
```

### Out of disk space

```bash
# Check disk usage
df -h

# Clean Docker resources
docker system prune -a

# Remove unused volumes
docker volume prune
```

---

## Next Steps

After successful initial deployment:

1. **Configure monitoring alerts** - See [ALERTING_THRESHOLDS.md](../ALERTING_THRESHOLDS.md)
2. **Set up backup schedule** - See [OPERATIONS_DB.md](../OPERATIONS_DB.md)
3. **Configure log aggregation** - Forward Docker logs to your logging system
4. **Document access** - Record credentials in secrets manager
5. **Schedule security review** - Verify firewall rules and exposed ports

---

## Emergency Contacts

| Role             | Contact  | When to Call               |
| ---------------- | -------- | -------------------------- |
| On-Call Engineer | [Define] | Any deployment issues      |
| Database Admin   | [Define] | Database connection issues |
| Security Team    | [Define] | Secret or access issues    |

---

**Last Updated**: 2024-01  
**Owner**: Platform Team  
**Review Cycle**: Quarterly
