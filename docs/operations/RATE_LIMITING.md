# Rate Limiting Configuration Guide

> **Doc Status:** Active (2025-12-20)
>
> **Purpose:** Document production rate limit settings, bypass mechanisms for staging load tests, and security requirements.
>
> **References:**
>
> - [`src/server/middleware/rateLimiter.ts`](../../src/server/middleware/rateLimiter.ts) - Implementation
> - [`.env.example`](../../.env.example) - Environment variable documentation
> - [`docs/planning/PRODUCTION_VALIDATION_REMEDIATION_PLAN.md`](../planning/PRODUCTION_VALIDATION_REMEDIATION_PLAN.md) - PV-02/PV-03 context

## Table of Contents

- [Overview](#overview)
- [Security Requirements](#security-requirements)
- [Environment Variables](#environment-variables)
- [Rate Limit Configurations](#rate-limit-configurations)
- [Production vs Staging Configuration](#production-vs-staging-configuration)
- [Bypass Mechanism for Load Testing](#bypass-mechanism-for-load-testing)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Troubleshooting](#troubleshooting)

---

## Overview

RingRift uses a comprehensive rate limiting system to protect against abuse and ensure fair resource allocation. The system supports:

- **Redis-backed rate limiting** for distributed deployments
- **In-memory fallback** for development and single-instance deployments
- **Per-endpoint configuration** with different limits for auth, game, API, and WebSocket endpoints
- **User-aware limiting** with higher limits for authenticated users
- **Staging bypass mechanism** for load testing (disabled in production)

### Key Components

| Component                                                      | Location                 | Purpose                       |
| -------------------------------------------------------------- | ------------------------ | ----------------------------- |
| [`rateLimiter.ts`](../../src/server/middleware/rateLimiter.ts) | `src/server/middleware/` | Core rate limiting middleware |
| Rate limit headers                                             | HTTP responses           | Client quota visibility       |
| Audit logging                                                  | Application logs         | Security and debugging        |
| Prometheus metrics                                             | `/metrics` endpoint      | Monitoring and alerting       |

---

## Security Requirements

### ⚠️ CRITICAL: Production Security Checklist

Before deploying to production, verify these security requirements:

| Requirement           | Production Setting                                | Verification Command                  |
| --------------------- | ------------------------------------------------- | ------------------------------------- |
| Bypass disabled       | `RATE_LIMIT_BYPASS_ENABLED=false`                 | `grep RATE_LIMIT_BYPASS_ENABLED .env` |
| Bypass token empty    | `RATE_LIMIT_BYPASS_TOKEN=` (empty)                | `grep RATE_LIMIT_BYPASS_TOKEN .env`   |
| Bypass IPs empty      | `RATE_LIMIT_BYPASS_IPS=` (empty)                  | `grep RATE_LIMIT_BYPASS_IPS .env`     |
| Audit logging enabled | Logs contain `rate_limit_bypass_triggered` events | Check application logs                |

### Security Implications

1. **Rate limiting protects against:**
   - Brute-force authentication attacks
   - DDoS and resource exhaustion
   - Account enumeration
   - API abuse and scraping

2. **Bypass mechanism risks:**
   - If enabled in production, attackers could bypass DDoS protection
   - The bypass token, if leaked, provides unlimited API access
   - IP-based bypass can be exploited via IP spoofing

3. **Audit trail:**
   All bypass events are logged with:
   - Trigger reason (`ip`, `user_pattern`, `bypass_token`)
   - Request path and method
   - Client IP address
   - User ID and email (if authenticated)

### Startup Warnings

The server logs security warnings at startup when bypass is enabled:

```
SECURITY: Rate limit bypass is ENABLED. This is dangerous in production!
```

In production (`NODE_ENV=production`), this is logged at **ERROR** level and should trigger alerts.

---

## Environment Variables

### Core Rate Limit Settings

All rate limit configurations follow the pattern:

- `RATE_LIMIT_{TYPE}_POINTS` - Maximum requests allowed in the window
- `RATE_LIMIT_{TYPE}_DURATION` - Window duration in seconds
- `RATE_LIMIT_{TYPE}_BLOCK_DURATION` - Block duration when exceeded (seconds)

### Complete Variable Reference

#### General API Endpoints

| Variable                             | Default | Description                      |
| ------------------------------------ | ------- | -------------------------------- |
| `RATE_LIMIT_API_POINTS`              | 50      | Requests for anonymous users     |
| `RATE_LIMIT_API_DURATION`            | 60      | Window in seconds                |
| `RATE_LIMIT_API_BLOCK_DURATION`      | 300     | Block duration (5 min)           |
| `RATE_LIMIT_API_AUTH_POINTS`         | 200     | Requests for authenticated users |
| `RATE_LIMIT_API_AUTH_DURATION`       | 60      | Window in seconds                |
| `RATE_LIMIT_API_AUTH_BLOCK_DURATION` | 300     | Block duration (5 min)           |

#### Authentication Endpoints

| Variable                                   | Default | Description             |
| ------------------------------------------ | ------- | ----------------------- |
| `RATE_LIMIT_AUTH_POINTS`                   | 10      | General auth requests   |
| `RATE_LIMIT_AUTH_DURATION`                 | 900     | Window: 15 minutes      |
| `RATE_LIMIT_AUTH_BLOCK_DURATION`           | 1800    | Block: 30 minutes       |
| `RATE_LIMIT_AUTH_LOGIN_POINTS`             | 5       | Login attempts          |
| `RATE_LIMIT_AUTH_LOGIN_DURATION`           | 900     | Window: 15 minutes      |
| `RATE_LIMIT_AUTH_LOGIN_BLOCK_DURATION`     | 1800    | Block: 30 minutes       |
| `RATE_LIMIT_AUTH_REGISTER_POINTS`          | 3       | Registration attempts   |
| `RATE_LIMIT_AUTH_REGISTER_DURATION`        | 3600    | Window: 1 hour          |
| `RATE_LIMIT_AUTH_REGISTER_BLOCK_DURATION`  | 3600    | Block: 1 hour           |
| `RATE_LIMIT_AUTH_PWD_RESET_POINTS`         | 3       | Password reset attempts |
| `RATE_LIMIT_AUTH_PWD_RESET_DURATION`       | 3600    | Window: 1 hour          |
| `RATE_LIMIT_AUTH_PWD_RESET_BLOCK_DURATION` | 3600    | Block: 1 hour           |

#### Game Endpoints

| Variable                                     | Default | Description                |
| -------------------------------------------- | ------- | -------------------------- |
| `RATE_LIMIT_GAME_POINTS`                     | 200     | Game management operations |
| `RATE_LIMIT_GAME_DURATION`                   | 60      | Window in seconds          |
| `RATE_LIMIT_GAME_BLOCK_DURATION`             | 300     | Block: 5 minutes           |
| `RATE_LIMIT_GAME_MOVES_POINTS`               | 100     | Active gameplay moves      |
| `RATE_LIMIT_GAME_MOVES_DURATION`             | 60      | Window in seconds          |
| `RATE_LIMIT_GAME_MOVES_BLOCK_DURATION`       | 60      | Block: 1 minute            |
| `RATE_LIMIT_GAME_CREATE_USER_POINTS`         | 20      | Game creation per user     |
| `RATE_LIMIT_GAME_CREATE_USER_DURATION`       | 600     | Window: 10 minutes         |
| `RATE_LIMIT_GAME_CREATE_USER_BLOCK_DURATION` | 600     | Block: 10 minutes          |
| `RATE_LIMIT_GAME_CREATE_IP_POINTS`           | 50      | Game creation per IP       |
| `RATE_LIMIT_GAME_CREATE_IP_DURATION`         | 600     | Window: 10 minutes         |
| `RATE_LIMIT_GAME_CREATE_IP_BLOCK_DURATION`   | 600     | Block: 10 minutes          |

#### WebSocket Endpoints

| Variable                       | Default | Description       |
| ------------------------------ | ------- | ----------------- |
| `RATE_LIMIT_WS_POINTS`         | 10      | New connections   |
| `RATE_LIMIT_WS_DURATION`       | 60      | Window in seconds |
| `RATE_LIMIT_WS_BLOCK_DURATION` | 300     | Block: 5 minutes  |

#### Specialized Endpoints

| Variable                                    | Default | Description               |
| ------------------------------------------- | ------- | ------------------------- |
| `RATE_LIMIT_DATA_EXPORT_POINTS`             | 1       | GDPR data export per user |
| `RATE_LIMIT_DATA_EXPORT_DURATION`           | 3600    | Window: 1 hour            |
| `RATE_LIMIT_DATA_EXPORT_BLOCK_DURATION`     | 3600    | Block: 1 hour             |
| `RATE_LIMIT_TELEMETRY_POINTS`               | 100     | Client telemetry events   |
| `RATE_LIMIT_TELEMETRY_DURATION`             | 60      | Window in seconds         |
| `RATE_LIMIT_TELEMETRY_BLOCK_DURATION`       | 300     | Block: 5 minutes          |
| `RATE_LIMIT_CLIENT_ERRORS_POINTS`           | 20      | Client error reports      |
| `RATE_LIMIT_CLIENT_ERRORS_DURATION`         | 60      | Window in seconds         |
| `RATE_LIMIT_CLIENT_ERRORS_BLOCK_DURATION`   | 300     | Block: 5 minutes          |
| `RATE_LIMIT_INTERNAL_HEALTH_POINTS`         | 30      | Health check probes       |
| `RATE_LIMIT_INTERNAL_HEALTH_DURATION`       | 60      | Window in seconds         |
| `RATE_LIMIT_INTERNAL_HEALTH_BLOCK_DURATION` | 60      | Block: 1 minute           |
| `RATE_LIMIT_ALERT_WEBHOOK_POINTS`           | 10      | Alert webhooks            |
| `RATE_LIMIT_ALERT_WEBHOOK_DURATION`         | 60      | Window in seconds         |
| `RATE_LIMIT_ALERT_WEBHOOK_BLOCK_DURATION`   | 300     | Block: 5 minutes          |
| `RATE_LIMIT_USER_RATING_POINTS`             | 30      | User rating lookups       |
| `RATE_LIMIT_USER_RATING_DURATION`           | 60      | Window in seconds         |
| `RATE_LIMIT_USER_RATING_BLOCK_DURATION`     | 120     | Block: 2 minutes          |
| `RATE_LIMIT_USER_SEARCH_POINTS`             | 20      | User search queries       |
| `RATE_LIMIT_USER_SEARCH_DURATION`           | 60      | Window in seconds         |
| `RATE_LIMIT_USER_SEARCH_BLOCK_DURATION`     | 120     | Block: 2 minutes          |
| `RATE_LIMIT_SANDBOX_AI_POINTS`              | 1000    | Sandbox AI moves          |
| `RATE_LIMIT_SANDBOX_AI_DURATION`            | 60      | Window in seconds         |
| `RATE_LIMIT_SANDBOX_AI_BLOCK_DURATION`      | 60      | Block: 1 minute           |

#### Bypass Configuration (STAGING ONLY)

| Variable                         | Default                            | Description                  |
| -------------------------------- | ---------------------------------- | ---------------------------- |
| `RATE_LIMIT_BYPASS_ENABLED`      | `false`                            | Master switch for bypass     |
| `RATE_LIMIT_BYPASS_TOKEN`        | (empty)                            | Bypass token (min 16 chars)  |
| `RATE_LIMIT_BYPASS_USER_PATTERN` | `^loadtest[._].+@loadtest\.local$` | Regex for load test users    |
| `RATE_LIMIT_BYPASS_IPS`          | (empty)                            | Comma-separated IP whitelist |

---

## Rate Limit Configurations

### Endpoint-Specific Limits

The system defines 19 rate limiter configurations:

```typescript
// From rateLimiter.ts
const rateLimiterConfigs = {
  api, // General API (anonymous)
  apiAuthenticated, // General API (authenticated)
  auth, // All auth endpoints
  authLogin, // Login specifically
  authRegister, // Registration
  authPasswordReset, // Password reset
  game, // Game management
  gameMoves, // Active gameplay
  gameCreateUser, // Game creation per user
  gameCreateIp, // Game creation per IP
  websocket, // WS connections
  dataExport, // GDPR export
  telemetry, // Client telemetry
  clientErrors, // Error reporting
  internalHealth, // Health checks
  alertWebhook, // Alert webhooks
  userRating, // Rating lookups
  userSearch, // User search
  sandboxAi, // Sandbox AI
};
```

### Rate Limit Headers

All responses include quota information:

| Header                  | Description                             |
| ----------------------- | --------------------------------------- |
| `X-RateLimit-Limit`     | Maximum requests in the window          |
| `X-RateLimit-Remaining` | Remaining requests in current window    |
| `X-RateLimit-Reset`     | Unix timestamp when window resets       |
| `Retry-After`           | Seconds to wait (only on 429 responses) |

### 429 Response Format

When rate limit is exceeded:

```json
{
  "success": false,
  "error": {
    "message": "Too many requests, please try again later",
    "code": "RATE_LIMIT_EXCEEDED",
    "retryAfter": 300,
    "timestamp": "2025-12-20T05:30:00.000Z"
  }
}
```

---

## Production vs Staging Configuration

### Configuration Comparison

| Setting                          | Production        | Staging (Load Test)                |
| -------------------------------- | ----------------- | ---------------------------------- |
| `RATE_LIMIT_BYPASS_ENABLED`      | `false` ❌        | `true`                             |
| `RATE_LIMIT_BYPASS_TOKEN`        | (empty)           | `<secure-random-token>`            |
| `RATE_LIMIT_BYPASS_USER_PATTERN` | N/A               | `^loadtest[._].+@loadtest\.local$` |
| `RATE_LIMIT_BYPASS_IPS`          | (empty)           | (empty or load test IPs)           |
| Redis backing                    | Required          | Optional                           |
| Rate limit values                | Standard defaults | Standard defaults                  |

### Production Configuration

```bash
# .env.production
NODE_ENV=production

# ⚠️  CRITICAL: Bypass must be disabled
RATE_LIMIT_BYPASS_ENABLED=false
RATE_LIMIT_BYPASS_TOKEN=
RATE_LIMIT_BYPASS_IPS=

# Standard rate limits (use defaults or tune as needed)
RATE_LIMIT_API_POINTS=50
RATE_LIMIT_API_AUTH_POINTS=200
RATE_LIMIT_AUTH_LOGIN_POINTS=5
RATE_LIMIT_GAME_POINTS=200
```

### Staging Configuration (for Load Testing)

```bash
# .env.staging
NODE_ENV=staging

# ⚠️  STAGING ONLY: Enable bypass for load tests
RATE_LIMIT_BYPASS_ENABLED=true

# Token-based bypass (preferred method)
# Generate with: openssl rand -base64 32
RATE_LIMIT_BYPASS_TOKEN=your-secure-32-char-token-here

# User pattern bypass (fallback)
RATE_LIMIT_BYPASS_USER_PATTERN=^loadtest[._].+@loadtest\.local$

# IP bypass (if load test server has fixed IP)
# RATE_LIMIT_BYPASS_IPS=10.0.0.50,10.0.0.51

# Higher limits for staging stress testing
RATE_LIMIT_GAME_POINTS=10000
RATE_LIMIT_GAME_MOVES_POINTS=10000
```

### Load Test Configuration

When running k6 load tests, configure the bypass token:

```javascript
// tests/load/auth/helpers.js
export function getBypassHeaders() {
  const token = __ENV.RATE_LIMIT_BYPASS_TOKEN;
  if (token) {
    return { 'X-RateLimit-Bypass-Token': token };
  }
  return {};
}
```

Run load tests with bypass:

```bash
# Set environment variables
export BASE_URL=http://localhost:3000
export RATE_LIMIT_BYPASS_TOKEN="your-staging-token"

# Run load test
k6 run tests/load/scenarios/concurrent-games.js
```

---

## Bypass Mechanism for Load Testing

### How Bypass Works

The bypass mechanism is implemented in [`shouldBypassRateLimit()`](../../src/server/middleware/rateLimiter.ts):

```typescript
function shouldBypassRateLimit(req: Request): boolean {
  if (!isRateLimitBypassEnabled()) return false;

  // Check bypass token header (highest priority)
  const bypassToken = getBypassToken();
  if (bypassToken) {
    const headerToken = req.headers['x-ratelimit-bypass-token'];
    if (headerToken === bypassToken) {
      logBypassTriggered(req, 'bypass_token', '(token)');
      return true;
    }
  }

  // Check IP whitelist
  const bypassIPs = getBypassIPs();
  if (bypassIPs.has(normalizedIp)) {
    logBypassTriggered(req, 'ip', normalizedIp);
    return true;
  }

  // Check user email pattern
  if (userEmail && pattern.test(userEmail)) {
    logBypassTriggered(req, 'user_pattern', userEmail);
    return true;
  }

  return false;
}
```

### Bypass Priority Order

1. **Token bypass** (`X-RateLimit-Bypass-Token` header) - Recommended for load tests
2. **IP whitelist** (`RATE_LIMIT_BYPASS_IPS`) - For fixed load test servers
3. **User pattern** (`RATE_LIMIT_BYPASS_USER_PATTERN`) - For load test user accounts

### Token Requirements

- Minimum 16 characters for security
- Should be cryptographically random
- Generate with: `openssl rand -base64 32`

### Audit Logging

All bypass events are logged:

```json
{
  "event": "rate_limit_bypass_triggered",
  "reason": "bypass_token",
  "identifier": "(token)",
  "path": "/api/games",
  "method": "POST",
  "ip": "10.0.0.50",
  "userId": "abc123",
  "userEmail": "loadtest.user1@loadtest.local"
}
```

---

## Monitoring and Alerting

### Prometheus Metrics

Rate limit hits are recorded via the MetricsService:

```typescript
getMetricsService().recordRateLimitHit(req.path, limiterKey);
```

### Grafana Dashboard Queries

```promql
# Rate limit hits by endpoint
sum(rate(rate_limit_hits_total[5m])) by (path)

# Rate limit hits by limiter type
sum(rate(rate_limit_hits_total[5m])) by (limiter)

# Percentage of requests rate limited
sum(rate(rate_limit_hits_total[5m])) / sum(rate(http_requests_total[5m])) * 100
```

### Recommended Alerts

| Alert                        | Condition                                                  | Severity |
| ---------------------------- | ---------------------------------------------------------- | -------- |
| High rate limit rate         | Rate limit hits > 5% of requests                           | Warning  |
| Auth endpoint abuse          | Login rate limits > 100/5min                               | Warning  |
| Bypass enabled in production | `rate_limit_bypass_enabled=true` AND `NODE_ENV=production` | Critical |

### Log-Based Monitoring

Monitor these log patterns:

```bash
# Rate limit exceeded events
grep "Rate limit exceeded" /var/log/ringrift/app.log

# Bypass triggered events (should be zero in production)
grep "rate_limit_bypass_triggered" /var/log/ringrift/app.log

# Security warning on startup
grep "Rate limit bypass is ENABLED" /var/log/ringrift/app.log
```

---

## Troubleshooting

### Common Issues

#### 1. Users Getting 429 Too Many Requests

**Symptoms:** Legitimate users see rate limit errors during normal usage.

**Diagnosis:**

```bash
# Check which limiter is triggering
grep "Rate limit exceeded" /var/log/ringrift/app.log | jq '.limiter' | sort | uniq -c
```

**Solutions:**

- Increase the rate limit for the affected endpoint type
- Check for misconfigured clients making excessive requests
- Consider adding the user to the authenticated tier (higher limits)

#### 2. Load Tests Being Rate Limited

**Symptoms:** k6 load tests show many 429 responses.

**Diagnosis:**

```bash
# Check if bypass is enabled
grep RATE_LIMIT_BYPASS_ENABLED .env.staging

# Check if token is being sent
k6 run --http-debug tests/load/scenarios/concurrent-games.js
```

**Solutions:**

- Ensure `RATE_LIMIT_BYPASS_ENABLED=true` in staging
- Verify the bypass token is set and matches in both env and k6 config
- Check that load test users match the bypass pattern

#### 3. Bypass Not Working

**Symptoms:** Bypass configured but load tests still rate limited.

**Diagnosis:**

```typescript
// Add debug logging in rateLimiter.ts
console.log('Bypass check:', {
  enabled: isRateLimitBypassEnabled(),
  token: !!getBypassToken(),
  headerToken: req.headers['x-ratelimit-bypass-token'],
});
```

**Checklist:**

- [ ] `RATE_LIMIT_BYPASS_ENABLED=true`
- [ ] Token is at least 16 characters
- [ ] Token in `.env` matches token in k6 config
- [ ] Header name is exactly `x-ratelimit-bypass-token` (lowercase)
- [ ] If using user pattern, verify user email format

#### 4. Redis Connection Failures

**Symptoms:** Rate limiting falls back to memory, limits not shared across instances.

**Diagnosis:**

```bash
# Check Redis connectivity
redis-cli -u $REDIS_URL ping
```

**Solutions:**

- Verify Redis is running and accessible
- Check Redis credentials and TLS configuration
- The system will automatically use in-memory limiting as fallback

#### 5. IP Normalization Issues

**Symptoms:** Same client gets different rate limit buckets.

**Cause:** IPv6 vs IPv4 representation (e.g., `::1` vs `127.0.0.1`).

**Solution:** The middleware normalizes IPs automatically:

- `::1` → `127.0.0.1`
- `::ffff:192.168.1.1` → `192.168.1.1`

### Debugging Commands

```bash
# View current rate limit configuration
grep RATE_LIMIT .env

# Test rate limit response headers
curl -i http://localhost:3000/api

# Simulate rate limit exceeded
for i in {1..100}; do curl -s -o /dev/null -w "%{http_code}\n" http://localhost:3000/api/games; done

# Check Redis rate limit keys
redis-cli -u $REDIS_URL keys "*_limit*"
```

### Emergency Procedures

#### Temporarily Disable Rate Limiting

In emergencies, you can disable rate limiting by setting very high limits:

```bash
RATE_LIMIT_API_POINTS=1000000
RATE_LIMIT_API_AUTH_POINTS=1000000
# ... for all limit types
```

**⚠️ WARNING:** This removes abuse protection. Only use temporarily and restore limits immediately after resolving the issue.

#### Clear Rate Limit State

To reset all rate limit counters in Redis:

```bash
redis-cli -u $REDIS_URL KEYS "*_limit*" | xargs redis-cli -u $REDIS_URL DEL
```

---

## Revision History

| Version | Date       | Changes                         |
| ------- | ---------- | ------------------------------- |
| 1.0     | 2025-12-20 | Initial documentation for PV-14 |
