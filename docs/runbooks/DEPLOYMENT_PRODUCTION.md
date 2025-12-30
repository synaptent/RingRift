# Production Deployment Guide

## Overview

This guide covers deploying RingRift to a production environment. It assumes you have already tested the deployment in staging using `docker-compose.staging.yml`.

**Target SLOs:**

- <500ms p95 latency
- <1% error rate
- > 99.9% uptime
- 100 concurrent games, 300 concurrent players

---

## Pre-Deployment Checklist

### Infrastructure Ready

- [ ] Production server provisioned (minimum: 4 vCPU, 8GB RAM, 100GB SSD)
- [ ] Docker Engine v20+ and Docker Compose v2+ installed
- [ ] SSL certificates obtained (Let's Encrypt or commercial)
- [ ] DNS records configured pointing to server IP
- [ ] Firewall configured (ports 80, 443 open; 5432, 6379 internal only)
- [ ] Backup storage configured (S3, GCS, or local)

### Secrets Generated

Generate all secrets **before** deployment:

```bash
# JWT secrets (minimum 48 characters each)
echo "JWT_SECRET=$(openssl rand -base64 48)"
echo "JWT_REFRESH_SECRET=$(openssl rand -base64 48)"

# Database password
echo "POSTGRES_PASSWORD=$(openssl rand -base64 32)"

# Redis password
echo "REDIS_PASSWORD=$(openssl rand -base64 32)"

# Grafana admin password
echo "GRAFANA_PASSWORD=$(openssl rand -base64 24)"
```

### Configuration Files Prepared

- [ ] `.env.production` created from `.env.production.example`
- [ ] All placeholder values replaced with real secrets
- [ ] `nginx.production.conf` configured with your domain
- [ ] SSL certificates placed in `./ssl/` directory

---

## Deployment Steps

### Step 1: Clone and Configure

```bash
# Clone repository
git clone https://github.com/an0mium/RingRift.git /opt/ringrift
cd /opt/ringrift

# Create production environment file
cp .env.production.example .env.production
nano .env.production  # Edit with real values

# Create required directories
mkdir -p ssl backups logs uploads
chmod 700 ssl
```

### Step 2: Configure SSL

```bash
# Option A: Let's Encrypt (recommended)
certbot certonly --standalone -d your-domain.com
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/
cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/

# Option B: Commercial certificate
cp your-certificate.pem ssl/fullchain.pem
cp your-private-key.pem ssl/privkey.pem

# Secure permissions
chmod 600 ssl/*.pem
```

### Step 3: Configure Nginx

Create `nginx.production.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:3000;
    }

    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/ssl/fullchain.pem;
        ssl_certificate_key /etc/ssl/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Strict-Transport-Security "max-age=31536000" always;

        # API and static content
        location / {
            proxy_pass http://app;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket
        location /socket.io/ {
            proxy_pass http://app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 86400;
        }

        # Health check (no auth required)
        location /health {
            proxy_pass http://app;
            access_log off;
        }
    }
}
```

### Step 4: Deploy Services

```bash
# Pull latest images (if using registry)
docker compose -f docker-compose.production.yml --env-file .env.production pull

# Or build locally
docker compose -f docker-compose.production.yml --env-file .env.production build

# Start database first
docker compose -f docker-compose.production.yml --env-file .env.production up -d postgres redis

# Wait for database to be ready
sleep 30

# Run migrations
docker compose -f docker-compose.production.yml --env-file .env.production run --rm app npx prisma migrate deploy

# Start all services
docker compose -f docker-compose.production.yml --env-file .env.production up -d
```

### Step 5: Verify Deployment

```bash
# Check all services are running
docker compose -f docker-compose.production.yml ps

# Check application health
curl -k https://your-domain.com/health

# Check readiness (all dependencies)
curl -k https://your-domain.com/ready

# Test WebSocket connection
wscat -c wss://your-domain.com/socket.io/

# Check logs for errors
docker compose -f docker-compose.production.yml logs --tail=100 app
```

---

## Post-Deployment Tasks

### Configure Backups

```bash
# Add to crontab
0 2 * * * /opt/ringrift/scripts/backup-database.sh
```

### Set Up Monitoring Alerts

1. Access Grafana at `https://your-domain.com:3002`
2. Login with admin credentials from `.env.production`
3. Configure notification channels (Slack, email, PagerDuty)
4. Enable alerting rules

### Configure Log Rotation

```bash
# /etc/logrotate.d/ringrift
/opt/ringrift/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
}
```

---

## Rollback Procedure

If deployment fails:

```bash
# Stop all services
docker compose -f docker-compose.production.yml down

# Restore previous version
git checkout previous-release-tag

# Restart with previous config
docker compose -f docker-compose.production.yml --env-file .env.production up -d
```

---

## Maintenance Windows

### Planned Maintenance

```bash
# Enable maintenance mode
docker compose -f docker-compose.production.yml exec app node scripts/maintenance-mode.js enable

# Perform maintenance...

# Disable maintenance mode
docker compose -f docker-compose.production.yml exec app node scripts/maintenance-mode.js disable
```

### Zero-Downtime Updates

```bash
# Pull new images
docker compose -f docker-compose.production.yml pull

# Rolling restart (app only)
docker compose -f docker-compose.production.yml up -d --no-deps app

# Verify health
curl https://your-domain.com/health
```

---

## Troubleshooting

### Service won't start

```bash
# Check logs
docker compose -f docker-compose.production.yml logs <service-name>

# Check resource usage
docker stats

# Verify environment variables
docker compose -f docker-compose.production.yml config
```

### Database connection issues

```bash
# Test connection from app container
docker compose -f docker-compose.production.yml exec app node -e "
const { PrismaClient } = require('@prisma/client');
const p = new PrismaClient();
p.\$queryRaw\`SELECT 1\`.then(() => console.log('OK'));
"
```

### High latency

1. Check Grafana dashboard for bottlenecks
2. Review PostgreSQL slow query log
3. Check Redis memory usage
4. Verify AI service response times

---

## Security Checklist

- [ ] All secrets are unique and secure (not from examples)
- [ ] SSL/TLS properly configured (test with ssllabs.com)
- [ ] Database not exposed externally
- [ ] Redis password protected
- [ ] Rate limiting enabled
- [ ] Security headers configured in nginx
- [ ] Firewall rules restrictive
- [ ] Regular security updates scheduled

---

## Contact

| Role     | Contact | When               |
| -------- | ------- | ------------------ |
| On-Call  | [TBD]   | Production issues  |
| Security | [TBD]   | Security incidents |
| Database | [TBD]   | Data issues        |

---

**Last Updated:** 2025-12-22
**Owner:** Platform Team
