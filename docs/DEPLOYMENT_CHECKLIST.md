# RingRift Deployment Checklist

Last Updated: January 12, 2026

## Overview

This checklist covers deploying ringrift.ai to production. The production server runs on AWS EC2 (r5.4xlarge in us-east-1).

## Production Infrastructure

| Component       | Details                                         |
| --------------- | ----------------------------------------------- |
| Web Server      | ringrift-web-prod (54.198.219.106)              |
| Process Manager | PM2                                             |
| Services        | ringrift-server (web), ringrift-ai (AI service) |
| Database        | PostgreSQL                                      |
| Cache           | Redis                                           |
| Proxy           | nginx                                           |
| DNS             | Cloudflare                                      |
| Domains         | ringrift.ai, staging.ringrift.ai                |

## Pre-Deployment Checklist

### 1. Code Quality

- [ ] All tests passing locally: `npm test`
- [ ] TypeScript compiles: `npm run build`
- [ ] Lint passes: `npm run lint`
- [ ] No console errors in browser dev tools
- [ ] AI service tests pass: `cd ai-service && pytest tests/ -v`

### 2. Database Migrations

- [ ] Check for pending migrations: `npm run migration:status`
- [ ] Review migration SQL for destructive changes
- [ ] Test migration on staging first
- [ ] Backup production database before applying

### 3. Environment Variables

Ensure these are set in production `.env`:

```bash
# Required for web server
NODE_ENV=production
DATABASE_URL=postgres://...
REDIS_URL=redis://...
SESSION_SECRET=<random-string>
PORT=3000

# Required for AI service
RINGRIFT_AI_SERVICE_URL=http://localhost:8000
RINGRIFT_NODE_ID=ringrift-web-prod
RINGRIFT_LOG_LEVEL=INFO
```

### 4. Model Files

- [ ] Canonical models exist in `ai-service/models/`
- [ ] Models are accessible to AI service
- [ ] Check model versions match production config

## Deployment Steps

### Option A: Use Deploy Script (Recommended)

The deploy script handles git pull, npm install, Prisma, build, and PM2 restart automatically.

```bash
ssh -i ~/.ssh/ringrift-staging-key.pem ubuntu@54.198.219.106
./deploy.sh
```

Options:

- `./deploy.sh --skip-pull` - Skip git pull (use when deploying local changes)
- `./deploy.sh --skip-install` - Skip npm install (faster when deps unchanged)

### Option B: Manual Deployment

#### Step 1: SSH to Production

```bash
ssh -i ~/.ssh/ringrift-staging-key.pem ubuntu@54.198.219.106
```

#### Step 2: Pull Latest Code

```bash
cd ~/ringrift
git fetch origin main
git status  # Check for local changes
git pull origin main
```

**Note:** The EC2 directory is lowercase `ringrift`, not `RingRift`.

#### Step 3: Install Dependencies

```bash
npm ci --production
cd ai-service && pip install -r requirements.txt
```

#### Step 4: Build Client

```bash
npm run build
# Verify dist/client exists and has files
ls -la dist/client/
```

#### Step 5: Run Migrations (if needed)

```bash
npm run migration:run
```

#### Step 6: Restart Services

```bash
pm2 restart ringrift-server --update-env
pm2 restart ringrift-ai --update-env
pm2 status
```

#### Step 7: Verify Deployment

```bash
# Check services are running
pm2 logs ringrift-server --lines 20
pm2 logs ringrift-ai --lines 20

# Check health endpoints
curl -s http://localhost:3000/ | head -5
curl -s http://localhost:8000/health
```

## Post-Deployment Verification

### 1. Web Application

- [ ] Homepage loads: https://ringrift.ai
- [ ] Can create new game (sandbox mode)
- [ ] Can start multiplayer game
- [ ] Rules modal displays correctly
- [ ] Mobile view works

### 2. AI Service

- [ ] AI opponent responds in sandbox
- [ ] AI difficulty selector works
- [ ] No timeout errors in browser console

### 3. Multiplayer

- [ ] Two users can join same game
- [ ] Moves sync between players
- [ ] WebSocket reconnection works

## Rollback Procedure

### Quick Rollback (code only)

```bash
ssh -i ~/.ssh/ringrift-staging-key.pem ubuntu@54.198.219.106
cd ~/ringrift
git checkout <previous-commit>
npm ci --production
npm run build
pm2 restart all
```

### Full Rollback (with database)

```bash
# 1. Restore database from backup
pg_restore -d ringrift production_backup.sql

# 2. Checkout previous code
git checkout <previous-commit>

# 3. Rebuild and restart
npm ci --production
npm run build
pm2 restart all
```

## Monitoring

### PM2 Commands

```bash
pm2 status              # Service status
pm2 logs                # All logs
pm2 logs ringrift-server --lines 100
pm2 monit               # Real-time monitor
```

### Health Checks

```bash
# Web server
curl -s https://ringrift.ai/ -o /dev/null -w "%{http_code}"

# AI service (internal)
curl -s http://localhost:8000/health

# Database
psql $DATABASE_URL -c "SELECT 1"

# Redis
redis-cli ping
```

### Key Metrics to Watch

- Response time < 200ms
- WebSocket connections active
- Memory usage < 80%
- CPU usage < 70%
- No 5xx errors in nginx logs

## Troubleshooting

### Service won't start

```bash
pm2 logs ringrift-server --err --lines 50
# Check for missing env vars or port conflicts
```

### AI service returns 500

```bash
pm2 logs ringrift-ai --err --lines 50
# Check model files exist and are readable
# Check CUDA/GPU if using GPU inference
```

### WebSocket disconnections

```bash
# Check nginx config for websocket timeout
grep -r "proxy_read_timeout" /etc/nginx/
# Should be at least 86400 (24 hours)
```

### High memory usage

```bash
pm2 restart ringrift-server --max-memory-restart 2G
# Or check for memory leaks in logs
```

## Emergency Contacts

- AWS Console: console.aws.amazon.com
- Cloudflare: dash.cloudflare.com
- PM2 Dashboard: app.pm2.io (if configured)

## Related Documents

- [Redis Performance Runbook](runbooks/REDIS_PERFORMANCE.md)
- [Database Performance](runbooks/DATABASE_PERFORMANCE.md)
- [Redis Down Procedure](runbooks/REDIS_DOWN.md)
