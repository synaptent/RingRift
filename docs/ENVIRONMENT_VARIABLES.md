# Environment Variables Reference

> **Doc Status (2025-11-27): Active**
>
> **Role:** Canonical reference for all environment variables used by RingRift across development, staging, and production, including defaults, ranges, and security considerations. Intended for operators and developers wiring config into Docker, Kubernetes, and CI.
>
> **Not a semantics SSoT:** This document does not define game rules or lifecycle semantics. Rules semantics are owned by the shared TypeScript rules engine under `src/shared/engine/**` plus contracts and vectors (see `RULES_CANONICAL_SPEC.md`, `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `docs/RULES_ENGINE_SURFACE_AUDIT.md`). Lifecycle semantics are owned by `docs/CANONICAL_ENGINE_API.md` together with shared types/schemas in `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, and `src/shared/validation/websocketSchemas.ts`.
>
> **Related docs:** `docs/SECRETS_MANAGEMENT.md`, `docs/DEPLOYMENT_REQUIREMENTS.md`, `docs/OPERATIONS_DB.md`, `docs/SECURITY_THREAT_MODEL.md`, `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`, and `DOCUMENTATION_INDEX.md`.

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
- [Data Retention](#data-retention)
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

### `ORCHESTRATOR_ADAPTER_ENABLED`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `true`                    |
| Required | No                        |

Enable the orchestrator adapter for unified turn processing.

When enabled (default), the server and sandbox engines use the canonical orchestrator
(`src/shared/engine/orchestration/`) via adapter wrappers for turn processing.
This is the production default after Phase 3 Rules Engine Consolidation completed
with 1179 tests passing under the orchestrator.

**Production Status (as of 2025-11-27):**

- Orchestrator adapter is now the default (`true`)
- 7 bugs were fixed to enable production readiness
- 1179 tests pass with orchestrator enabled
- Legacy tests that manipulate internal state are skipped by design

**Monitoring Checklist:**

- [x] No increase in game errors/exceptions
- [x] No divergence in move validation (cross-platform parity)
- [x] No performance regression (AI response time)
- [x] No difference in game outcomes (victory distribution)

**Rollback:**
Set `ORCHESTRATOR_ADAPTER_ENABLED=false` to immediately revert to legacy path.
No data migration required - flag is purely behavioral.

**Related Documentation:**

- [Legacy Code Elimination Plan](./drafts/LEGACY_CODE_ELIMINATION_PLAN.md)
- [Phase 3 Adapter Migration Report](./drafts/PHASE3_ADAPTER_MIGRATION_REPORT.md)

### Orchestrator rollout controls

These variables control **gradual rollout** and **automatic rollback** of the
canonical TS orchestrator in production. They are evaluated by
`OrchestratorRolloutService` and should typically be adjusted only in staging
or by on-call operators following a runbook.

#### `ORCHESTRATOR_ROLLOUT_PERCENTAGE`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Range    | 0-100    |
| Default  | `100`    |
| Required | No       |

Percentage of eligible sessions to route through the orchestrator when
`ORCHESTRATOR_ADAPTER_ENABLED=true`.

- `0` → orchestrator adapter is effectively disabled (all traffic on legacy path).
- `100` → all eligible sessions use the orchestrator (current default).
- Intermediate values (e.g. `10`, `50`) enable **canary-style rollout**.

#### `ORCHESTRATOR_SHADOW_MODE_ENABLED`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `false`   |
| Required | No        |

When `true`, the orchestrator runs in **shadow mode**:

- Legacy turn processing remains authoritative for gameplay.
- The orchestrator runs in parallel for the same turns and its results are
  compared against the legacy engine.
- Divergences are logged and surfaced via metrics for safe production testing.

Use this in staging / early rollout phases to validate orchestrator behaviour
under real traffic before switching to full production mode.

#### `ORCHESTRATOR_ALLOWLIST_USERS`

| Property | Value                      |
| -------- | -------------------------- |
| Type     | `string` (comma-separated) |
| Default  | `""` (empty string)        |
| Required | No                         |

Comma-separated list of **user IDs** that are **always routed through the
orchestrator**, regardless of `ORCHESTRATOR_ROLLOUT_PERCENTAGE` or other
heuristics.

Use this to force-enable orchestrator behaviour for internal accounts or
specific test users.

#### `ORCHESTRATOR_DENYLIST_USERS`

| Property | Value                      |
| -------- | -------------------------- |
| Type     | `string` (comma-separated) |
| Default  | `""` (empty string)        |
| Required | No                         |

Comma-separated list of **user IDs** that are **never routed through the
orchestrator**, even if the global rollout percentage would otherwise include
them.

Use this to exclude sensitive or high-value accounts from orchestrator
experiments while rollout is in progress.

#### `ORCHESTRATOR_CIRCUIT_BREAKER_ENABLED`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `true`    |
| Required | No        |

Enables the orchestrator **circuit breaker**. When enabled, sustained error
rates or latency regressions for orchestrator-processed sessions will
automatically reduce or disable orchestrator usage according to the thresholds
below.

Set to `false` only for controlled experiments where automatic rollback is not
desired (for example, during development or in tightly monitored staging
runs).

#### `ORCHESTRATOR_ERROR_THRESHOLD_PERCENT`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Range    | 0-100    |
| Default  | `5`      |
| Required | No       |

Percentage of orchestrator-processed requests within the error window that may
fail **before** the circuit breaker trips.

- At `5` (default), if more than 5% of orchestrator turns fail within the
  configured window, the orchestrator is considered unhealthy and may be
  disabled or scaled back.

#### `ORCHESTRATOR_ERROR_WINDOW_SECONDS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `300`    |
| Required | No       |

Time window, in seconds, over which orchestrator error rates are measured for
the circuit breaker.

#### `ORCHESTRATOR_LATENCY_THRESHOLD_MS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `500`    |
| Required | No       |

P99 latency threshold for orchestrator-processed requests, in milliseconds.
Used by `OrchestratorRolloutService` to detect performance regressions during
rollout.

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

## Data Retention

These variables control the data retention policies enforced by [`DataRetentionService`](../src/server/services/DataRetentionService.ts). The service provides configurable retention periods for different data types and can be scheduled via cron for automated cleanup.

### `DATA_RETENTION_DELETED_USERS_DAYS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `30`     |
| Required | No       |

Days to retain soft-deleted user accounts before permanent deletion. After this period, [`hardDeleteExpiredUsers()`](../src/server/services/DataRetentionService.ts) will permanently remove the user record and associated data.

### `DATA_RETENTION_INACTIVE_USERS_DAYS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `365`    |
| Required | No       |

Days of inactivity before considering a user for cleanup operations. Used for identifying dormant accounts.

### `DATA_RETENTION_UNVERIFIED_DAYS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `7`      |
| Required | No       |

Days to retain unverified user accounts before soft-deleting them. [`cleanupUnverifiedAccounts()`](../src/server/services/DataRetentionService.ts) marks these accounts as deleted with `isActive: false`.

### `DATA_RETENTION_GAME_DATA_MONTHS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `24`     |
| Required | No       |

Months to retain game data (moves, game records). Game data is preserved longer than user data for rating integrity and historical analysis.

### `DATA_RETENTION_SESSION_HOURS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `24`     |
| Required | No       |

Hours to retain session data in Redis. Affects WebSocket session cleanup.

### `DATA_RETENTION_EXPIRED_TOKEN_DAYS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `7`      |
| Required | No       |

Days to retain expired refresh tokens before cleanup. [`cleanupExpiredTokens()`](../src/server/services/DataRetentionService.ts) removes tokens that have been expired or revoked beyond this threshold.

### Data Retention Service Usage

The retention service can be initialized and run as follows:

```typescript
import { createDataRetentionService } from './services/DataRetentionService';

// Create service with environment-based configuration
const retentionService = createDataRetentionService(prisma);

// Run all retention tasks
await retentionService.runRetentionTasks();

// Or run individual tasks
await retentionService.hardDeleteExpiredUsers();
await retentionService.cleanupExpiredTokens();
await retentionService.cleanupUnverifiedAccounts();
```

**Recommended Setup:** Schedule `runRetentionTasks()` via cron job (e.g., daily at 3 AM):

```typescript
import cron from 'node-cron';

cron.schedule('0 3 * * *', async () => {
  await retentionService.runRetentionTasks();
});
```

**Related Documentation:**

- [Data Lifecycle and Privacy](./DATA_LIFECYCLE_AND_PRIVACY.md) - Full S-05.E implementation details
- [API Reference](./API_REFERENCE.md) - Account deletion and data export endpoints

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
