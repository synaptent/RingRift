# Environment Variables Reference

This document provides comprehensive documentation for all environment variables used by RingRift.

## Quick Start

1. Copy `.env.example` to `.env`
2. For development, the defaults should work out of the box
3. For production, ensure all required variables are set with secure values

## Environment Modes

| Mode          | Description                                    |
| ------------- | ---------------------------------------------- |
| `development` | Local development with sensible defaults       |
| `staging`     | Pre-production with production-like validation |
| `production`  | Production with strict validation              |
| `test`        | Automated testing (auto-detected in Jest)      |

## Variable Categories

- [Server Configuration](#server-configuration)
- [Database](#database)
- [Redis](#redis)
- [Authentication](#authentication)
- [AI Service](#ai-service)
- [Rate Limiting](#rate-limiting)
- [Logging](#logging)
- [CORS](#cors)
- [Feature Flags](#feature-flags)
- [Game Configuration](#game-configuration)
- [File Storage](#file-storage)
- [Email](#email)
- [Rules Engine](#rules-engine)
- [Application Topology](#application-topology)
- [Testing Configuration](#testing-configuration)
- [Debug Flags](#debug-flags)

---

## Server Configuration

### `NODE_ENV`

| Property | Value                                          |
| -------- | ---------------------------------------------- |
| Type     | `enum`                                         |
| Values   | `development`, `staging`, `production`, `test` |
| Default  | `development`                                  |
| Required | No                                             |

Application environment mode. Controls validation strictness and default behaviors.

### `PORT`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Range    | 1-65535  |
| Default  | `3000`   |
| Required | No       |

HTTP server port for the main API.

### `HOST`

| Property | Value     |
| -------- | --------- |
| Type     | `string`  |
| Default  | `0.0.0.0` |
| Required | No        |

Server bind address. Use `0.0.0.0` to listen on all interfaces.

---

## Database

### `DATABASE_URL`

| Property | Value                                           |
| -------- | ----------------------------------------------- |
| Type     | `string` (PostgreSQL URL)                       |
| Format   | `postgresql://USER:PASSWORD@HOST:PORT/DATABASE` |
| Default  | None                                            |
| Required | **Yes in production**                           |

PostgreSQL connection URL. Contains credentials - never log or expose.

**Security Considerations:**

- Use strong passwords (16+ characters, mixed characters)
- Use TLS connections in production
- Rotate credentials regularly

**Secret Rotation:**

1. Create new database user with new password
2. Grant same permissions as old user
3. Update `DATABASE_URL` environment variable
4. Restart application
5. Verify connectivity, then revoke old user

### `DATABASE_POOL_MIN`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `2`      |
| Required | No       |

Minimum database connection pool size.

### `DATABASE_POOL_MAX`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `10`     |
| Required | No       |

Maximum database connection pool size.

---

## Redis

### `REDIS_URL`

| Property | Value                                             |
| -------- | ------------------------------------------------- |
| Type     | `string` (Redis URL)                              |
| Format   | `redis://HOST:PORT` or `rediss://HOST:PORT` (TLS) |
| Default  | `redis://localhost:6379` (dev only)               |
| Required | **Yes in production**                             |

Redis connection URL for caching, sessions, and pub/sub.

**Security Considerations:**

- Use `rediss://` (TLS) in production
- Set `REDIS_PASSWORD` for authenticated access
- Never expose Redis to the internet

### `REDIS_PASSWORD`

| Property   | Value             |
| ---------- | ----------------- |
| Type       | `string`          |
| Default    | None              |
| Required   | No                |
| Min Length | 8 (in production) |

Redis authentication password.

### `REDIS_TLS`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `false`                   |
| Required | No                        |

Enable TLS for Redis connections.

---

## Authentication

### `JWT_SECRET`

| Property   | Value                                |
| ---------- | ------------------------------------ |
| Type       | `string`                             |
| Default    | `dev-access-token-secret` (dev only) |
| Required   | **Yes in production**                |
| Min Length | 32                                   |

Secret key for signing JWT access tokens.

**Security Considerations:**

- Generate with `openssl rand -base64 48`
- Never reuse across environments
- Placeholder values are rejected in production

### `JWT_REFRESH_SECRET`

| Property   | Value                                 |
| ---------- | ------------------------------------- |
| Type       | `string`                              |
| Default    | `dev-refresh-token-secret` (dev only) |
| Required   | **Yes in production**                 |
| Min Length | 32                                    |

Secret key for signing JWT refresh tokens. Should be different from `JWT_SECRET`.

### `JWT_EXPIRES_IN`

| Property | Value               |
| -------- | ------------------- |
| Type     | `string` (duration) |
| Format   | `15m`, `1h`, `7d`   |
| Default  | `15m`               |
| Required | No                  |

Access token expiration time. Keep short for security.

### `JWT_REFRESH_EXPIRES_IN`

| Property | Value               |
| -------- | ------------------- |
| Type     | `string` (duration) |
| Format   | `15m`, `1h`, `7d`   |
| Default  | `7d`                |
| Required | No                  |

Refresh token expiration time.

### `AUTH_MAX_FAILED_LOGIN_ATTEMPTS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `10`     |
| Required | No       |

Maximum failed login attempts before lockout.

### `AUTH_FAILED_LOGIN_WINDOW_SECONDS`

| Property | Value              |
| -------- | ------------------ |
| Type     | `number`           |
| Default  | `900` (15 minutes) |
| Required | No                 |

Time window for counting failed login attempts.

### `AUTH_LOCKOUT_DURATION_SECONDS`

| Property | Value              |
| -------- | ------------------ |
| Type     | `number`           |
| Default  | `900` (15 minutes) |
| Required | No                 |

Duration of login lockout.

### `AUTH_LOGIN_LOCKOUT_ENABLED`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `true`                    |
| Required | No                        |

Enable/disable login lockout feature.

### `BCRYPT_ROUNDS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Range    | 4-31     |
| Default  | `12`     |
| Required | No       |

bcrypt rounds for password hashing. Higher = more secure but slower.

---

## AI Service

### `AI_SERVICE_URL`

| Property | Value                              |
| -------- | ---------------------------------- |
| Type     | `string` (URL)                     |
| Default  | `http://localhost:8001` (dev only) |
| Required | **Yes in production**              |

AI service base URL.

**Docker Compose:** Use `http://ai-service:8001`

### `AI_SERVICE_REQUEST_TIMEOUT_MS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `5000`   |
| Required | No       |

AI service request timeout in milliseconds.

### `AI_RULES_REQUEST_TIMEOUT_MS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `5000`   |
| Required | No       |

AI rules evaluation request timeout in milliseconds.

### `AI_MAX_CONCURRENT_REQUESTS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `16`     |
| Required | No       |

Maximum concurrent AI requests (backpressure control).

### `AI_FALLBACK_ENABLED`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `true`    |
| Required | No        |

Enable AI fallback to local heuristics when service unavailable.

---

## Rate Limiting

Rate limiting uses Redis for distributed deployments or in-memory storage for development.

Response headers:

- `X-RateLimit-Limit`: Maximum requests in the window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Unix timestamp when the window resets

### General API (Anonymous)

| Variable                        | Default | Description               |
| ------------------------------- | ------- | ------------------------- |
| `RATE_LIMIT_API_POINTS`         | 50      | Requests per window       |
| `RATE_LIMIT_API_DURATION`       | 60      | Window duration (seconds) |
| `RATE_LIMIT_API_BLOCK_DURATION` | 300     | Block duration (seconds)  |

### General API (Authenticated)

| Variable                             | Default | Description               |
| ------------------------------------ | ------- | ------------------------- |
| `RATE_LIMIT_API_AUTH_POINTS`         | 200     | Requests per window       |
| `RATE_LIMIT_API_AUTH_DURATION`       | 60      | Window duration (seconds) |
| `RATE_LIMIT_API_AUTH_BLOCK_DURATION` | 300     | Block duration (seconds)  |

### Authentication Endpoints

| Variable                         | Default | Description               |
| -------------------------------- | ------- | ------------------------- |
| `RATE_LIMIT_AUTH_POINTS`         | 10      | Requests per window       |
| `RATE_LIMIT_AUTH_DURATION`       | 900     | Window duration (seconds) |
| `RATE_LIMIT_AUTH_BLOCK_DURATION` | 1800    | Block duration (seconds)  |

### Login Endpoint (Stricter)

| Variable                               | Default | Description               |
| -------------------------------------- | ------- | ------------------------- |
| `RATE_LIMIT_AUTH_LOGIN_POINTS`         | 5       | Requests per window       |
| `RATE_LIMIT_AUTH_LOGIN_DURATION`       | 900     | Window duration (seconds) |
| `RATE_LIMIT_AUTH_LOGIN_BLOCK_DURATION` | 1800    | Block duration (seconds)  |

### Registration Endpoint

| Variable                                  | Default | Description               |
| ----------------------------------------- | ------- | ------------------------- |
| `RATE_LIMIT_AUTH_REGISTER_POINTS`         | 3       | Requests per window       |
| `RATE_LIMIT_AUTH_REGISTER_DURATION`       | 3600    | Window duration (seconds) |
| `RATE_LIMIT_AUTH_REGISTER_BLOCK_DURATION` | 3600    | Block duration (seconds)  |

### Password Reset Endpoint

| Variable                                   | Default | Description               |
| ------------------------------------------ | ------- | ------------------------- |
| `RATE_LIMIT_AUTH_PWD_RESET_POINTS`         | 3       | Requests per window       |
| `RATE_LIMIT_AUTH_PWD_RESET_DURATION`       | 3600    | Window duration (seconds) |
| `RATE_LIMIT_AUTH_PWD_RESET_BLOCK_DURATION` | 3600    | Block duration (seconds)  |

### Game Endpoints

| Variable                         | Default | Description               |
| -------------------------------- | ------- | ------------------------- |
| `RATE_LIMIT_GAME_POINTS`         | 200     | Requests per window       |
| `RATE_LIMIT_GAME_DURATION`       | 60      | Window duration (seconds) |
| `RATE_LIMIT_GAME_BLOCK_DURATION` | 300     | Block duration (seconds)  |

### Game Moves

| Variable                               | Default | Description               |
| -------------------------------------- | ------- | ------------------------- |
| `RATE_LIMIT_GAME_MOVES_POINTS`         | 100     | Requests per window       |
| `RATE_LIMIT_GAME_MOVES_DURATION`       | 60      | Window duration (seconds) |
| `RATE_LIMIT_GAME_MOVES_BLOCK_DURATION` | 60      | Block duration (seconds)  |

### WebSocket Connections

| Variable                       | Default | Description               |
| ------------------------------ | ------- | ------------------------- |
| `RATE_LIMIT_WS_POINTS`         | 10      | Connections per window    |
| `RATE_LIMIT_WS_DURATION`       | 60      | Window duration (seconds) |
| `RATE_LIMIT_WS_BLOCK_DURATION` | 300     | Block duration (seconds)  |

### Game Creation (Per User)

| Variable                                     | Default | Description               |
| -------------------------------------------- | ------- | ------------------------- |
| `RATE_LIMIT_GAME_CREATE_USER_POINTS`         | 20      | Games per window          |
| `RATE_LIMIT_GAME_CREATE_USER_DURATION`       | 600     | Window duration (seconds) |
| `RATE_LIMIT_GAME_CREATE_USER_BLOCK_DURATION` | 600     | Block duration (seconds)  |

### Game Creation (Per IP)

| Variable                                   | Default | Description               |
| ------------------------------------------ | ------- | ------------------------- |
| `RATE_LIMIT_GAME_CREATE_IP_POINTS`         | 50      | Games per window          |
| `RATE_LIMIT_GAME_CREATE_IP_DURATION`       | 600     | Window duration (seconds) |
| `RATE_LIMIT_GAME_CREATE_IP_BLOCK_DURATION` | 600     | Block duration (seconds)  |

### Fallback Limiter

| Variable                           | Default | Description            |
| ---------------------------------- | ------- | ---------------------- |
| `RATE_LIMIT_FALLBACK_WINDOW_MS`    | 900000  | Window duration (ms)   |
| `RATE_LIMIT_FALLBACK_MAX_REQUESTS` | 100     | Max requests in window |

---

## Logging

### `LOG_LEVEL`

| Property | Value                                     |
| -------- | ----------------------------------------- |
| Type     | `enum`                                    |
| Values   | `error`, `warn`, `info`, `debug`, `trace` |
| Default  | `info`                                    |
| Required | No                                        |

Application log level.

### `LOG_FORMAT`

| Property | Value            |
| -------- | ---------------- |
| Type     | `enum`           |
| Values   | `json`, `pretty` |
| Default  | `json`           |
| Required | No               |

Log output format. Use `json` in production, `pretty` for development.

### `LOG_FILE`

| Property | Value           |
| -------- | --------------- |
| Type     | `string` (path) |
| Default  | None            |
| Required | No              |

Log file path. Logs also go to stdout.

---

## CORS

### `CORS_ORIGIN`

| Property | Value                   |
| -------- | ----------------------- |
| Type     | `string`                |
| Default  | `http://localhost:5173` |
| Required | No                      |

Primary CORS origin for API requests.

### `CLIENT_URL`

| Property | Value                   |
| -------- | ----------------------- |
| Type     | `string`                |
| Default  | `http://localhost:3000` |
| Required | No                      |

Public client URL for redirects.

### `ALLOWED_ORIGINS`

| Property | Value                                         |
| -------- | --------------------------------------------- |
| Type     | `string` (comma-separated)                    |
| Default  | `http://localhost:5173,http://localhost:3000` |
| Required | No                                            |

Comma-separated list of allowed CORS origins.

---

## Feature Flags

### `ENABLE_METRICS`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `true`    |
| Required | No        |

Enable Prometheus metrics endpoint.

### `ENABLE_HEALTH_CHECKS`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `true`    |
| Required | No        |

Enable health check endpoints.

### `METRICS_PORT`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Range    | 1-65535  |
| Default  | `9090`   |
| Required | No       |

Prometheus metrics server port.

---

## Game Configuration

### `MAX_CONCURRENT_GAMES`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `1000`   |
| Required | No       |

Maximum concurrent games allowed.

### `GAME_TIMEOUT_MINUTES`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `60`     |
| Required | No       |

Game inactivity timeout (minutes). Inactive games are cleaned up.

### `AI_THINK_TIME_MS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `2000`   |
| Required | No       |

AI thinking time in milliseconds.

### `MAX_SPECTATORS_PER_GAME`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `50`     |
| Required | No       |

Maximum spectators per game.

---

## File Storage

### `UPLOAD_DIR`

| Property | Value           |
| -------- | --------------- |
| Type     | `string` (path) |
| Default  | `./uploads`     |
| Required | No              |

Directory for user uploads (relative to project root).

### `MAX_FILE_SIZE`

| Property | Value            |
| -------- | ---------------- |
| Type     | `number` (bytes) |
| Default  | `5242880` (5MB)  |
| Required | No               |

Maximum upload file size in bytes.

---

## Email

### `SMTP_HOST`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

SMTP server host.

### `SMTP_PORT`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | None     |
| Required | No       |

SMTP server port (typically 587 for TLS, 465 for SSL).

### `SMTP_USER`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

SMTP authentication username.

### `SMTP_PASSWORD`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

SMTP authentication password.

### `SMTP_TLS`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `false`   |
| Required | No        |

Enable TLS for SMTP.

### `SMTP_FROM`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Email sender address.

---

## Rules Engine

### `RINGRIFT_RULES_MODE`

| Property | Value                    |
| -------- | ------------------------ |
| Type     | `enum`                   |
| Values   | `ts`, `python`, `shadow` |
| Default  | `ts`                     |
| Required | No                       |

Rules engine mode:

- `ts`: TypeScript engine only (default)
- `shadow`: Python runs in parallel for validation (development)
- `python`: Python engine is authoritative

---

## Application Topology

### `RINGRIFT_APP_TOPOLOGY`

| Property | Value                                    |
| -------- | ---------------------------------------- |
| Type     | `enum`                                   |
| Values   | `single`, `multi-unsafe`, `multi-sticky` |
| Default  | `single`                                 |
| Required | No                                       |

Deployment topology:

- `single`: Single instance, all state in memory
- `multi-unsafe`: Multiple instances without sticky sessions (NOT RECOMMENDED)
- `multi-sticky`: Multiple instances with infrastructure-enforced sticky sessions

**Warning:** Do not use `multi-*` without proper session affinity configuration.

---

## Testing Configuration

Test-related environment variables for controlling test execution and behavior.

### `ENABLE_MCTS_TESTS`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `false`                   |
| Required | No                        |

Enable MCTS (Monte Carlo Tree Search) AI tests. These tests are computationally expensive and are disabled by default to speed up CI.

### `MCTS_TEST_TIMEOUT`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `60`     |
| Required | No       |

Timeout in seconds for MCTS tests. Increase this value if MCTS tests are timing out in your environment.

### `E2E_BASE_URL`

| Property | Value                   |
| -------- | ----------------------- |
| Type     | `string` (URL)          |
| Default  | `http://localhost:3000` |
| Required | No                      |

Base URL for E2E (Playwright) tests. Set this to the URL where your application is running during E2E testing.

### `PLAYWRIGHT_WORKERS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `1`      |
| Required | No       |

Number of parallel workers for Playwright E2E tests. Increase for faster test execution on multi-core systems.

---

## Debug Flags

These flags are for development and debugging only. **Do not enable in production.**

| Variable                                       | Default | Description                             |
| ---------------------------------------------- | ------- | --------------------------------------- |
| `RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS` | `false` | Enable sandbox AI stall diagnostics     |
| `RINGRIFT_SANDBOX_CAPTURE_DEBUG`               | `false` | Enable sandbox capture debug logging    |
| `RINGRIFT_SANDBOX_AI_CAPTURE_DEBUG`            | `false` | Enable sandbox AI capture debug logging |
| `RINGRIFT_SANDBOX_AI_TRACE_MODE`               | `false` | Enable sandbox AI trace mode            |
| `RINGRIFT_SANDBOX_AI_PARITY_MODE`              | `false` | Enable sandbox AI parity mode           |
| `RINGRIFT_LOCAL_AI_HEURISTIC_MODE`             | `false` | Enable local AI heuristic mode          |

---

## Production Checklist

Before deploying to production, ensure:

- [ ] `NODE_ENV=production`
- [ ] `DATABASE_URL` is set with production database credentials
- [ ] `REDIS_URL` is set with production Redis URL
- [ ] `JWT_SECRET` is randomly generated (32+ characters)
- [ ] `JWT_REFRESH_SECRET` is randomly generated (32+ characters, different from JWT_SECRET)
- [ ] `AI_SERVICE_URL` is set to the internal AI service URL
- [ ] `CORS_ORIGIN` and `ALLOWED_ORIGINS` are set to production domains
- [ ] `LOG_FORMAT=json` for structured logging
- [ ] All debug flags are disabled
- [ ] No placeholder values are in use

## Generating Secure Secrets

```bash
# Generate a 48-byte random secret (64 characters in base64)
openssl rand -base64 48

# Or using Node.js
node -e "console.log(require('crypto').randomBytes(48).toString('base64'))"
```

## See Also

- [Secrets Management Guide](./SECRETS_MANAGEMENT.md)
- [Security Threat Model](./SECURITY_THREAT_MODEL.md)
- [Operations Database Reference](./OPERATIONS_DB.md)
