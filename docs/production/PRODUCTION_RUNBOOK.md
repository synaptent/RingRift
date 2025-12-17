# RingRift Production Runbook

## Overview

This runbook documents operational procedures for the RingRift production environment deployed on AWS.

**Infrastructure:**

- **Server:** EC2 r5.4xlarge (128GB RAM, 16 vCPU) in us-east-1
- **Domain:** ringrift.ai (via Route53)
- **SSL:** Let's Encrypt via Certbot
- **Process Manager:** PM2

## Services

| Service        | Port   | PM2 Name        | Description                       |
| -------------- | ------ | --------------- | --------------------------------- |
| Node.js Server | 3001   | ringrift-server | Main API and WebSocket server     |
| AI Service     | 8765   | ringrift-ai     | Python FastAPI AI move service    |
| nginx          | 80/443 | (system)        | Reverse proxy and SSL termination |

## Common Commands

### Check Service Status

```bash
pm2 status
pm2 logs ringrift-server --lines 50
pm2 logs ringrift-ai --lines 50
```

### Restart Services

```bash
pm2 restart ringrift-server
pm2 restart ringrift-ai
pm2 restart all
```

### View Real-time Logs

```bash
pm2 logs
pm2 logs ringrift-server
pm2 logs ringrift-ai
```

### Health Checks

```bash
curl http://localhost:3001/health
curl http://localhost:8765/health
/home/ubuntu/ringrift/scripts/ringrift-health-check.sh
```

## Monitoring

### CloudWatch Alarms

- **RingRift-AI-Service-Unhealthy**: Triggers if AI service health check fails for 3+ minutes
- **RingRift-Node-Server-Unhealthy**: Triggers if Node.js server health check fails for 3+ minutes

Alarms visible in AWS Console → CloudWatch → Alarms

### Custom Metrics

Health check pushes metrics every minute:

- Namespace: `RingRift/Production`
- Metrics: `AIServiceHealthy`, `NodeServerHealthy`

### Slack Alerts

Configured to send alerts to Slack when:

- Services go DOWN (red alert)
- Services recover (green notification)

Health check runs every minute via cron.

## Logs

### PM2 Logs

Location: `~/.pm2/logs/`

- `ringrift-server-out-*.log` - Server stdout
- `ringrift-server-error-*.log` - Server stderr
- `ringrift-ai-out-*.log` - AI service stdout
- `ringrift-ai-error-*.log` - AI service stderr

### Log Rotation

PM2-logrotate installed:

- Max size: 50MB per file
- Retention: 7 days
- Compression: enabled

### nginx Logs

- Access: `/var/log/nginx/access.log`
- Error: `/var/log/nginx/error.log`

## Backups

### Automated Backups

Daily backup cron runs at 3 AM UTC:

- Database dump
- PM2 config
- Recent logs (compressed)
- Uploaded to S3: `s3://ringrift-backups/daily/`
- Retention: 30 days

### Manual Backup

```bash
/home/ubuntu/ringrift/scripts/daily-backup.sh
```

### Restore from Backup

```bash
# Download backup
aws s3 sync s3://ringrift-backups/daily/YYYY-MM-DD/ /tmp/restore/

# Restore database
psql "$DATABASE_URL" < /tmp/restore/database.sql

# Restore PM2 config
cp /tmp/restore/pm2-dump.json ~/.pm2/dump.pm2
pm2 resurrect
```

## Deployment

### Deploy from Local

```bash
# Build
npm run build

# Deploy
rsync -avz --delete dist/ ubuntu@ringrift.ai:/home/ubuntu/ringrift/dist/

# Restart server
ssh ubuntu@ringrift.ai "pm2 restart ringrift-server"
```

### Deploy AI Service Changes

```bash
rsync -avz ai-service/app/ ubuntu@ringrift.ai:/home/ubuntu/ringrift/ai-service/app/
ssh ubuntu@ringrift.ai "pm2 restart ringrift-ai"
```

## Troubleshooting

### AI Service Not Responding

1. Check if service is running: `pm2 status`
2. Check logs: `pm2 logs ringrift-ai --lines 100`
3. Check memory: `free -h`
4. Restart: `pm2 restart ringrift-ai`

### High Latency

1. Check server load: `htop`
2. Check database connections: Check PM2 logs for connection pool errors
3. Check AI service queue: Look for timeout errors in logs

### Rate Limiting Issues

Rate limit configuration (defaults → current load test values):

| Limiter                  | Default       | Load Test | Window            |
| ------------------------ | ------------- | --------- | ----------------- |
| Sandbox AI               | 1000 requests | 10000     | per minute        |
| Authenticated API        | 200 requests  | 2000      | per minute        |
| Anonymous API            | 50 requests   | 50        | per minute        |
| Game API                 | 200 requests  | 2000      | per minute per IP |
| WebSocket connections    | 10            | 500       | per minute per IP |
| Game creation (per user) | 20 games      | 20        | per 10 minutes    |
| Game creation (per IP)   | 50 games      | 500       | per 10 minutes    |

If hitting limits, check logs for `"Adaptive rate limit exceeded"` or `"Rate limit exceeded"` messages.

**Load Testing Configuration:**

To support load testing where all k6 VUs originate from a single IP, the following `.env` variables are set:

```bash
RATE_LIMIT_WS_POINTS=500
RATE_LIMIT_WS_DURATION=60
RATE_LIMIT_GAME_CREATE_IP_POINTS=500
RATE_LIMIT_GAME_CREATE_IP_DURATION=600
RATE_LIMIT_GAME_POINTS=2000
RATE_LIMIT_API_AUTH_POINTS=2000
RATE_LIMIT_SANDBOX_AI_POINTS=10000
RATE_LIMIT_BYPASS_ENABLED=true
```

The `RATE_LIMIT_BYPASS_ENABLED=true` setting allows loadtest users (matching `loadtest.*@loadtest\.local`) to bypass per-user rate limits entirely.

### Circuit Breaker Tripped

If AI service has repeated failures, circuit breaker opens. Signs:

- Logs show "Circuit breaker opened after repeated failures"
- AI moves fall back to local selection

Fix: Restart both services to reset circuit breaker state.

### Memory Issues

AI service may consume significant memory (~400-800MB). If memory grows:

1. Check for stuck searches: Long-running minimax/MCTS
2. Restart AI service: `pm2 restart ringrift-ai`

### WebSocket Connection Limits

System limits that affect WebSocket capacity (defaults → tuned values):

| Limit                        | Default | Tuned | Effect                           |
| ---------------------------- | ------- | ----- | -------------------------------- |
| ulimit -n (file descriptors) | 1024    | 65535 | Limits total open connections    |
| nginx worker_connections     | 768     | 4096  | Max connections per nginx worker |
| proxy_read_timeout           | 60s     | 300s  | WebSocket idle timeout           |
| proxy_send_timeout           | 60s     | 300s  | WebSocket send timeout           |

**System Tuning Applied (2025-12-14):**

In `/etc/security/limits.conf`:

```
ubuntu soft nofile 65535
ubuntu hard nofile 65535
```

In `/etc/nginx/conf.d/websocket-tuning.conf`:

```nginx
proxy_read_timeout 300s;
proxy_send_timeout 300s;
```

In `/etc/nginx/nginx.conf` events block:

```nginx
worker_connections 4096;
```

**Note:** Requires `sudo systemctl reload nginx` and new SSH session (or server reboot) for ulimit changes to take effect.

**To increase capacity**, add to `/etc/security/limits.conf`:

```
ubuntu soft nofile 65535
ubuntu hard nofile 65535
```

And in `/etc/nginx/nginx.conf`:

```nginx
events {
    worker_connections 4096;
}
http {
    proxy_read_timeout 300;
    proxy_send_timeout 300;
}
```

## Configuration

### Environment Variables

Key environment variables for ringrift-server:

```
NODE_ENV=production
AI_SERVICE_URL=http://localhost:8765
AI_SERVICE_REQUEST_TIMEOUT_MS=30000
ENABLE_SANDBOX_AI_ENDPOINTS=true

# Load testing rate limit overrides (higher than defaults)
RATE_LIMIT_WS_POINTS=500              # Default: 10
RATE_LIMIT_WS_DURATION=60
RATE_LIMIT_GAME_CREATE_IP_POINTS=500  # Default: 50
RATE_LIMIT_GAME_CREATE_IP_DURATION=600
RATE_LIMIT_GAME_POINTS=2000           # Default: 200
RATE_LIMIT_API_AUTH_POINTS=2000       # Default: 200
RATE_LIMIT_SANDBOX_AI_POINTS=10000    # Default: 1000
RATE_LIMIT_BYPASS_ENABLED=true        # Bypass for loadtest users
```

### AI Difficulty Ladder

Think times by difficulty:
| Difficulty | AI Type | Think Time |
|------------|---------|------------|
| D1 | Random | ~50ms |
| D2 | Heuristic | ~200ms |
| D3 | Minimax | ~1.8s |
| D4 | Minimax+NNUE | ~2.8s |
| D5 | MCTS | ~4s |
| D6-8 | MCTS+Neural | 5-10s |
| D9-10 | Descent+Neural | 12-16s |

## Contacts

- Infrastructure issues: Check AWS Console / CloudWatch
- AI Service issues: Review logs, check neural net model loading
- Database issues: Check PostgreSQL connection string in secrets

## Version History

| Date       | Change                                                        |
| ---------- | ------------------------------------------------------------- |
| 2025-12-15 | Initial runbook created                                       |
| 2025-12-15 | Added AI timeout fix (5s → 30s)                               |
| 2025-12-15 | Added sandbox AI rate limiter (1000/min)                      |
| 2025-12-15 | Fixed minimax time check (1000 → 100 nodes)                   |
| 2025-12-15 | Increased WS/game rate limits for load test                   |
| 2025-12-15 | Documented WebSocket capacity limits                          |
| 2025-12-15 | Complete load test rate limit config (sandbox AI 10k, bypass) |
| 2025-12-15 | Applied system tuning (ulimit 65535, nginx 4096 workers)      |
