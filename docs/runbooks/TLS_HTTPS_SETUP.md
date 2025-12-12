# TLS/HTTPS Setup Guide

> **Doc Status (2025-12-11): Runbook**
>
> Step-by-step guide for enabling HTTPS on RingRift production deployment.

## Overview

This guide covers setting up TLS/HTTPS for RingRift using:

- **Let's Encrypt** for free SSL certificates
- **Caddy** as reverse proxy (simplest option - automatic HTTPS)
- Alternative: **nginx** with Certbot

**Time required:** 30-60 minutes

---

## Prerequisites

Before starting, you need:

1. **A domain name** pointing to your server's IP address
   - Example: `ringrift.example.com`
   - DNS A record: `ringrift.example.com → your.server.ip`

2. **A server** with:
   - Docker and Docker Compose installed
   - Ports 80 and 443 open in firewall
   - RingRift application running (default port 3001)

3. **DNS propagation complete** (can take up to 48 hours, usually minutes)
   - Verify with: `dig ringrift.example.com` or `nslookup ringrift.example.com`

---

## Option A: Caddy (Recommended - Simplest)

Caddy automatically obtains and renews Let's Encrypt certificates.

### Step 1: Create Caddyfile

Create a file named `Caddyfile` in your deployment directory:

```
# Caddyfile
ringrift.example.com {
    # Proxy all traffic to the RingRift backend
    reverse_proxy localhost:3001

    # Enable compression
    encode gzip

    # WebSocket support (automatic with reverse_proxy)

    # Security headers (Caddy adds sensible defaults)
    header {
        # Prevent clickjacking
        X-Frame-Options "SAMEORIGIN"
        # Prevent MIME sniffing
        X-Content-Type-Options "nosniff"
        # Enable XSS filter
        X-XSS-Protection "1; mode=block"
    }
}
```

Replace `ringrift.example.com` with your actual domain.

### Step 2: Create docker-compose.production.yml

```yaml
version: '3.8'

services:
  # Caddy reverse proxy with automatic HTTPS
  caddy:
    image: caddy:2-alpine
    restart: unless-stopped
    ports:
      - '80:80'
      - '443:443'
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    depends_on:
      - app
    networks:
      - ringrift

  # RingRift application
  app:
    image: ringrift:latest
    restart: unless-stopped
    environment:
      - NODE_ENV=production
      - DATABASE_URL=${DATABASE_URL}
      - JWT_SECRET=${JWT_SECRET}
      - REDIS_URL=${REDIS_URL}
      # Add other env vars from .env.production
    ports:
      - '127.0.0.1:3001:3001' # Only expose to localhost
    networks:
      - ringrift
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - ringrift

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data
    networks:
      - ringrift

volumes:
  caddy_data:
  caddy_config:
  postgres_data:
  redis_data:

networks:
  ringrift:
```

### Step 3: Deploy

```bash
# 1. Ensure your domain DNS is pointing to this server
#    Verify: dig +short ringrift.example.com

# 2. Stop any existing deployment
docker-compose down

# 3. Start the production stack
docker-compose -f docker-compose.production.yml up -d

# 4. Check Caddy logs for certificate acquisition
docker-compose -f docker-compose.production.yml logs caddy

# 5. Verify HTTPS is working
curl -I https://ringrift.example.com/health
```

### Step 4: Verify

1. **Browser test:** Visit `https://ringrift.example.com`
   - Should show green padlock
   - No certificate warnings

2. **SSL Labs test:** Visit https://www.ssllabs.com/ssltest/
   - Enter your domain
   - Should score A or A+

3. **WebSocket test:** Open browser console on your site
   ```javascript
   const ws = new WebSocket('wss://ringrift.example.com/ws');
   ws.onopen = () => console.log('WSS connected!');
   ```

---

## Option B: nginx + Certbot

More control, but requires manual certificate renewal setup.

### Step 1: Install Certbot

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install certbot python3-certbot-nginx nginx

# Or using snap (recommended by Certbot)
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot
```

### Step 2: Create nginx Configuration

Create `/etc/nginx/sites-available/ringrift`:

```nginx
# HTTP - redirect to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name ringrift.example.com;

    # Let's Encrypt challenge directory
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    # Redirect all other traffic to HTTPS
    location / {
        return 301 https://$host$request_uri;
    }
}

# HTTPS
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name ringrift.example.com;

    # SSL certificates (will be created by Certbot)
    ssl_certificate /etc/letsencrypt/live/ringrift.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ringrift.example.com/privkey.pem;

    # SSL configuration (modern settings)
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Proxy to RingRift backend
    location / {
        proxy_pass http://127.0.0.1:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket timeout (important for game connections)
        proxy_read_timeout 86400;
    }
}
```

### Step 3: Enable Site and Get Certificate

```bash
# 1. Enable the site
sudo ln -s /etc/nginx/sites-available/ringrift /etc/nginx/sites-enabled/

# 2. Test nginx configuration
sudo nginx -t

# 3. Reload nginx (without SSL first - comment out ssl_certificate lines)
sudo systemctl reload nginx

# 4. Get certificate from Let's Encrypt
sudo certbot --nginx -d ringrift.example.com

# 5. Certbot will:
#    - Verify domain ownership
#    - Obtain certificate
#    - Update nginx config automatically
#    - Set up auto-renewal

# 6. Verify auto-renewal is configured
sudo systemctl status certbot.timer
```

### Step 4: Test Certificate Renewal

```bash
# Dry run renewal test
sudo certbot renew --dry-run

# If successful, renewal will happen automatically via systemd timer
```

---

## Troubleshooting

### Certificate acquisition fails

**Symptom:** Certbot/Caddy can't get certificate

**Possible causes:**

1. **DNS not propagated:** Wait or check with `dig ringrift.example.com`
2. **Port 80 blocked:** Check firewall: `sudo ufw status` or cloud provider firewall
3. **Wrong domain:** Ensure domain in config matches DNS exactly

**Debug:**

```bash
# Check if port 80 is reachable from outside
curl -I http://ringrift.example.com

# Check Caddy logs
docker logs caddy 2>&1 | tail -50

# Check Certbot logs
sudo cat /var/log/letsencrypt/letsencrypt.log
```

### WebSocket connections fail over HTTPS

**Symptom:** Game works over HTTP but not HTTPS

**Fix:** Ensure your client connects to `wss://` not `ws://`:

```javascript
// Wrong
const ws = new WebSocket('ws://ringrift.example.com/ws');

// Correct
const ws = new WebSocket('wss://ringrift.example.com/ws');
```

### Mixed content warnings

**Symptom:** Browser console shows "Mixed Content" errors

**Fix:** Ensure all resources use HTTPS:

- Update `VITE_API_URL` in client build
- Check for hardcoded `http://` URLs

---

## Environment Variables for Production

Update your `.env.production`:

```bash
# Server
NODE_ENV=production
PORT=3001

# URLs (use HTTPS)
VITE_API_URL=https://ringrift.example.com
VITE_WS_URL=wss://ringrift.example.com

# Trust proxy (tells Express it's behind a reverse proxy)
TRUST_PROXY=true

# ... other secrets via secrets manager
```

---

## Verification Checklist

After setup, verify:

- [ ] `https://ringrift.example.com` loads without warnings
- [ ] HTTP automatically redirects to HTTPS
- [ ] SSL Labs test scores A or higher
- [ ] WebSocket connections work (`wss://`)
- [ ] Game functionality works end-to-end
- [ ] Certificate renewal test passes (`certbot renew --dry-run`)

---

## Certificate Renewal

### Caddy

Automatic - no action needed. Caddy renews certificates before expiry.

### Certbot

Automatic via systemd timer. Verify it's active:

```bash
sudo systemctl status certbot.timer
```

To manually renew:

```bash
sudo certbot renew
sudo systemctl reload nginx
```

---

## Related Documents

- [DEPLOYMENT_INITIAL.md](DEPLOYMENT_INITIAL.md) - Initial deployment guide
- [STAGING_ENVIRONMENT.md](../operations/STAGING_ENVIRONMENT.md) - Staging setup
- [SECURITY.md](../../SECURITY.md) - Security policy
