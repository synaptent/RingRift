# Environment Variables Reference

> **Doc Status (2025-12-01): Active**
>
> **Role:** Canonical reference for all environment variables used by RingRift across development, staging, and production, including defaults, ranges, and security considerations. Intended for operators and developers wiring config into Docker, Kubernetes, and CI.
>
> **Not a semantics SSoT:** This document does not define game rules or lifecycle semantics. Rules semantics are owned by the shared TypeScript rules engine under `src/shared/engine/**` plus contracts and vectors (see `RULES_CANONICAL_SPEC.md`, `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `docs/rules/RULES_ENGINE_SURFACE_AUDIT.md`). Lifecycle semantics are owned by `docs/architecture/CANONICAL_ENGINE_API.md` together with shared types/schemas in `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, and `src/shared/validation/websocketSchemas.ts`.
>
> **Related docs:** `docs/operations/SECRETS_MANAGEMENT.md`, `docs/planning/DEPLOYMENT_REQUIREMENTS.md`, `docs/operations/OPERATIONS_DB.md`, `docs/security/SECURITY_THREAT_MODEL.md`, `docs/security/SUPPLY_CHAIN_AND_CI_SECURITY.md`, and `DOCUMENTATION_INDEX.md`.

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
- [Client/Frontend](#clientfrontend)
- [Feature Flags](#feature-flags)
- [Game Configuration](#game-configuration)
- [File Storage](#file-storage)
- [Email](#email)
- [Rules Engine](#rules-engine)
- [Application Topology](#application-topology)
- [Testing Configuration](#testing-configuration)
- [Data Retention](#data-retention)
- [Python Training Flags](#python-training-flags)
- [Debug Flags](#debug-flags)
- [Deprecated Variables](#deprecated-variables)
- [Parity & Replay Tooling](#parity--replay-tooling)

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

Server bind address for the main API (and the optional metrics server when enabled). Use `0.0.0.0` to listen on all interfaces.

### `npm_package_version`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Application version automatically injected by npm during build. Used for version display and telemetry.

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

### `DB_HOST`

| Property | Value       |
| -------- | ----------- |
| Type     | `string`    |
| Default  | `localhost` |
| Required | No          |

Database host component used by docker-compose and deployment tooling. The
server runtime reads `DATABASE_URL` only; `DB_HOST` is ignored by Prisma/server.

### `DB_PORT`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `5432`   |
| Required | No       |

Database port component used by docker-compose and deployment tooling. The
server runtime reads `DATABASE_URL` only; `DB_PORT` is ignored by Prisma/server.

### `DB_NAME`

| Property | Value      |
| -------- | ---------- |
| Type     | `string`   |
| Default  | `ringrift` |
| Required | No         |

Database name component used by docker-compose and deployment tooling. The
server runtime reads `DATABASE_URL` only; `DB_NAME` is ignored by Prisma/server.

### `DB_USER`

| Property | Value      |
| -------- | ---------- |
| Type     | `string`   |
| Default  | `ringrift` |
| Required | No         |

Database user component used by docker-compose and deployment tooling. The
server runtime reads `DATABASE_URL` only; `DB_USER` is ignored by Prisma/server.

### `DB_PASSWORD`

| Property | Value                        |
| -------- | ---------------------------- |
| Type     | `string`                     |
| Default  | `password` (dev placeholder) |
| Required | No                           |

Database password component used by docker-compose and deployment tooling.
The server runtime reads `DATABASE_URL` only; `DB_PASSWORD` is ignored by
Prisma/server. Use strong secrets in production and prefer `DATABASE_URL`.

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

### `REDIS_HOST`

| Property | Value       |
| -------- | ----------- |
| Type     | `string`    |
| Default  | `localhost` |
| Required | No          |

Redis host component used by deployment tooling. The server runtime reads
`REDIS_URL` only; `REDIS_HOST` is ignored by the server.

### `REDIS_PORT`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `6379`   |
| Required | No       |

Redis port component used by deployment tooling. The server runtime reads
`REDIS_URL` only; `REDIS_PORT` is ignored by the server.

---

## Authentication

### `JWT_SECRET`

| Property   | Value                                     |
| ---------- | ----------------------------------------- |
| Type       | `string`                                  |
| Default    | None (set explicitly; see `.env.example`) |
| Required   | **Yes in production**                     |
| Min Length | 32                                        |

Secret key for signing JWT access tokens.

**Security Considerations:**

- Generate with `openssl rand -base64 48`
- Never reuse across environments
- Placeholder values are rejected in production

### `JWT_REFRESH_SECRET`

| Property   | Value                                     |
| ---------- | ----------------------------------------- |
| Type       | `string`                                  |
| Default    | None (set explicitly; see `.env.example`) |
| Required   | **Yes in production**                     |
| Min Length | 32                                        |

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

### `AI_SERVICE_PORT`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `8001`   |
| Required | No       |

Port the Python AI service binds to when started directly (FastAPI/uvicorn).

### `AI_LOG_LEVEL`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | `INFO`   |
| Required | No       |

Log level for the Python AI service container. In `docker-compose.*`, this is mapped to the
container's `LOG_LEVEL` env var (the TypeScript server does not read `AI_LOG_LEVEL`).

### `RINGRIFT_ENV`

| Property | Value                                          |
| -------- | ---------------------------------------------- |
| Type     | `string`                                       |
| Values   | `development`, `staging`, `production`, `test` |
| Default  | `development`                                  |
| Required | No                                             |

Environment mode for the Python AI service. Used for error sanitization and
assertion behavior inside `ai-service/app/main.py` and helpers.

### `CORS_ORIGINS`

| Property | Value                      |
| -------- | -------------------------- |
| Type     | `string` (comma-separated) |
| Default  | `*`                        |
| Required | No                         |

CORS allow-list for the Python AI service (`ai-service/app/main.py`). When unset or empty,
the service falls back to `*`. This is separate from the backend's `CORS_ORIGIN`.

### `ADMIN_API_KEY`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Admin key for protected AI-service endpoints (for example cache management).
When unset, the AI service generates an ephemeral key at startup.

### `RINGRIFT_NODE_ROLE`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | (empty)  |
| Required | No       |

When set, the AI service auto-starts the matching daemon profile on boot (see
`ai-service/app/coordination/daemon_manager.py`).

### `RINGRIFT_IS_COORDINATOR`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `auto`                    |
| Required | No                        |

Force coordinator-only mode. When enabled, selfplay/training/gauntlet/export are disabled
by default; when unset, the role is derived from `distributed_hosts.yaml`.

### `RINGRIFT_SELFPLAY_ENABLED`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `auto`                    |
| Required | No                        |

Explicit override for selfplay on this node. When unset, defaults are resolved from
`distributed_hosts.yaml` (coordinator nodes default to disabled).

### `RINGRIFT_TRAINING_ENABLED`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `auto`                    |
| Required | No                        |

Explicit override for training on this node. When unset, defaults are resolved from
`distributed_hosts.yaml` (coordinator nodes default to disabled).

### `RINGRIFT_GAUNTLET_ENABLED`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `auto`                    |
| Required | No                        |

Explicit override for gauntlet/evaluation on this node. When unset, defaults are resolved from
`distributed_hosts.yaml` (coordinator nodes default to disabled).

### `RINGRIFT_EXPORT_ENABLED`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `auto`                    |
| Required | No                        |

Explicit override for export jobs on this node. When unset, defaults are resolved from
`distributed_hosts.yaml` (coordinator nodes default to disabled).

### `RINGRIFT_IDLE_RESOURCE_ENABLED`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `true`                    |
| Required | No                        |

Enable the idle-resource daemon that spawns or scales down selfplay based on GPU idleness.

### `RINGRIFT_IDLE_THRESHOLD`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `10.0`   |
| Required | No       |

GPU utilization threshold (%) below which a node is considered idle. Legacy alias:
`RINGRIFT_IDLE_GPU_THRESHOLD`.

### `RINGRIFT_IDLE_DURATION`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `120`    |
| Required | No       |

Seconds a GPU must remain idle before the node is treated as idle.

### `RINGRIFT_JOB_GRACE_PERIOD`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `60`     |
| Required | No       |

Seconds to wait before SIGKILL after SIGTERM when shutting down jobs.

### `RINGRIFT_GPU_IDLE_THRESHOLD`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `600`    |
| Required | No       |

Seconds of GPU idleness before killing stuck processes.

### `RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `128`    |
| Required | No       |

Max selfplay processes per node before cleanup kicks in.

### `RINGRIFT_SSH_USER`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | `ubuntu` |
| Required | No       |

Default SSH user for cluster operations.

### `RINGRIFT_SSH_KEY`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Default SSH key path for cluster operations.

### `RINGRIFT_LAMBDA_IDLE_ENABLED`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `true`                    |
| Required | No                        |

Enable the Lambda idle shutdown daemon. (Lambda account currently suspended; kept for restoration.)

### `RINGRIFT_LAMBDA_IDLE_INTERVAL`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `300`    |
| Required | No       |

Lambda idle check interval in seconds.

### `RINGRIFT_LAMBDA_IDLE_THRESHOLD`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `5.0`    |
| Required | No       |

Lambda GPU idle threshold (%).

### `RINGRIFT_LAMBDA_IDLE_DURATION`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `1800`   |
| Required | No       |

Lambda idle duration threshold in seconds.

### `RINGRIFT_LAMBDA_MIN_NODES`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `1`      |
| Required | No       |

Minimum Lambda nodes to keep running.

### `RINGRIFT_START_DAEMONS`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `0`                       |
| Required | No                        |

Legacy toggle to start the core daemon set on AI-service boot.

### `RINGRIFT_AI_TIMEOUT`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `30.0`   |
| Required | No       |

Hard timeout in seconds for AI move/evaluation operations inside the Python service.

### `RINGRIFT_AI_INSTANCE_CACHE`

| Property | Value                     |
| -------- | ------------------------- |
| Type     | `boolean`                 |
| Values   | `true`, `false`, `1`, `0` |
| Default  | `true`                    |
| Required | No                        |

Enable the AI instance cache (re-uses tree search instances for active games).

### `RINGRIFT_AI_INSTANCE_CACHE_TTL_SEC`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `1800`   |
| Required | No       |

Time-to-live in seconds for cached AI instances.

### `RINGRIFT_AI_INSTANCE_CACHE_MAX`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `512`    |
| Required | No       |

Maximum number of cached AI instances before eviction.

### `RINGRIFT_AI_SERVICE_URL`

| Property | Value                              |
| -------- | ---------------------------------- |
| Type     | `string` (URL)                     |
| Default  | `http://localhost:8001` (dev only) |
| Required | No                                 |

Client-side AI service URL used by the React app (for example, by `ReplayService` to talk to the GameReplayDB replay API). When set at build/runtime, this value is propagated via Vite and should typically match `AI_SERVICE_URL` in environments where the browser can reach the AI service directly.

### `AI_SERVICE_REQUEST_TIMEOUT_MS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `30000`  |
| Required | No       |

AI service request timeout in milliseconds.

On the TS backend, this value is surfaced as `config.aiService.requestTimeoutMs`
and used by `GameSession` as the **hard host-level budget** for per-move AI
service calls via `runWithTimeout` (see
`GameSession.getAIMoveWithTimeout()` in `src/server/game/GameSession.ts` and
`docs/archive/assessments/P18.3-1_DECISION_LIFECYCLE_SPEC.md` §3.2.1.2). It bounds AI _search
time_ only; there is no additional artificial “thinking delay” layered on
top of this timeout.

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

### `GAME_REPLAY_DB_PATH`

| Property | Value                      |
| -------- | -------------------------- |
| Type     | `string` (filesystem path) |
| Default  | `data/games/selfplay.db`   |
| Required | No                         |

Path to the SQLite **GameReplayDB** file used by the AI service replay API (`/api/replay/*`).  
When set, this overrides the default `data/games/selfplay.db` used by `ai-service/app/routes/replay.py`.  
For `/sandbox`:

- The **ReplayPanel** can auto-load games whose `gameId` exists in this DB.
- If a self-play game is loaded from a different DB and the `gameId` is not present here, the ReplayPanel shows a banner:
  `Requested game not found in replay DB (check GAME_REPLAY_DB_PATH). Using local sandbox replay instead.`

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

### Specialized Endpoints

| Variable                                    | Default | Description                       |
| ------------------------------------------- | ------- | --------------------------------- |
| `RATE_LIMIT_DATA_EXPORT_POINTS`             | 1       | GDPR export requests per window   |
| `RATE_LIMIT_DATA_EXPORT_DURATION`           | 3600    | GDPR export window (seconds)      |
| `RATE_LIMIT_DATA_EXPORT_BLOCK_DURATION`     | 3600    | GDPR export block (seconds)       |
| `RATE_LIMIT_TELEMETRY_POINTS`               | 100     | Telemetry events per window       |
| `RATE_LIMIT_TELEMETRY_DURATION`             | 60      | Telemetry window (seconds)        |
| `RATE_LIMIT_TELEMETRY_BLOCK_DURATION`       | 300     | Telemetry block (seconds)         |
| `RATE_LIMIT_CLIENT_ERRORS_POINTS`           | 20      | Client error reports per window   |
| `RATE_LIMIT_CLIENT_ERRORS_DURATION`         | 60      | Client error window (seconds)     |
| `RATE_LIMIT_CLIENT_ERRORS_BLOCK_DURATION`   | 300     | Client error block (seconds)      |
| `RATE_LIMIT_INTERNAL_HEALTH_POINTS`         | 30      | Internal health probes per window |
| `RATE_LIMIT_INTERNAL_HEALTH_DURATION`       | 60      | Health probe window (seconds)     |
| `RATE_LIMIT_INTERNAL_HEALTH_BLOCK_DURATION` | 60      | Health probe block (seconds)      |
| `RATE_LIMIT_ALERT_WEBHOOK_POINTS`           | 10      | Alert webhook calls per window    |
| `RATE_LIMIT_ALERT_WEBHOOK_DURATION`         | 60      | Alert webhook window (seconds)    |
| `RATE_LIMIT_ALERT_WEBHOOK_BLOCK_DURATION`   | 300     | Alert webhook block (seconds)     |
| `RATE_LIMIT_USER_RATING_POINTS`             | 30      | User rating lookups per window    |
| `RATE_LIMIT_USER_RATING_DURATION`           | 60      | User rating window (seconds)      |
| `RATE_LIMIT_USER_RATING_BLOCK_DURATION`     | 120     | User rating block (seconds)       |
| `RATE_LIMIT_USER_SEARCH_POINTS`             | 20      | User search queries per window    |
| `RATE_LIMIT_USER_SEARCH_DURATION`           | 60      | User search window (seconds)      |
| `RATE_LIMIT_USER_SEARCH_BLOCK_DURATION`     | 120     | User search block (seconds)       |
| `RATE_LIMIT_SANDBOX_AI_POINTS`              | 1000    | Sandbox AI requests per window    |
| `RATE_LIMIT_SANDBOX_AI_DURATION`            | 60      | Sandbox AI window (seconds)       |
| `RATE_LIMIT_SANDBOX_AI_BLOCK_DURATION`      | 60      | Sandbox AI block (seconds)        |

### Rate Limit Bypass (staging/dev only)

| Variable                         | Default                            | Description                             |
| -------------------------------- | ---------------------------------- | --------------------------------------- |
| `RATE_LIMIT_BYPASS_ENABLED`      | `false`                            | Master switch for bypass                |
| `RATE_LIMIT_BYPASS_TOKEN`        | None                               | Shared bypass token (min 16 chars)      |
| `RATE_LIMIT_BYPASS_USER_PATTERN` | `^loadtest[._].+@loadtest\.local$` | Regex for load-test users               |
| `RATE_LIMIT_BYPASS_IPS`          | None                               | Comma-separated IP allowlist for bypass |

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

| Property | Value                                                       |
| -------- | ----------------------------------------------------------- |
| Type     | `enum`                                                      |
| Values   | `json`, `pretty`                                            |
| Default  | `json` (production), `pretty` (development/test when unset) |
| Required | No                                                          |

Log output format. Use `json` in production, `pretty` for development.

`LOG_FORMAT` controls **console** formatting. File transports (`logs/error.log` and the combined log file) always emit structured JSON.

### `LOG_FILE`

| Property | Value           |
| -------- | --------------- |
| Type     | `string` (path) |
| Default  | None            |
| Required | No              |

Log file path. Logs also go to stdout.

When set, this overrides the combined log file path (default: `logs/combined.log`). The error log remains at `logs/error.log`.

**Log fields and runbook alignment**

Structured logs emitted by the backend include a common set of fields used throughout the incident runbooks (`HIGH_ERROR_RATE.md`, `RATE_LIMITING.md`, `RULES_PARITY.md`), for example:

- `requestId` – per-request correlation ID (from `requestContext` / `requestContextStorage`).
- `userId` – authenticated user identifier when available.
- `method`, `path`, `statusCode` – HTTP verb, normalized path, and response status.
- `code` – standardized error / parity / rate-limit code when present (for example `AUTH_INVALID_CREDENTIALS`, `RATE_LIMIT_EXCEEDED`, `SERVER_INTERNAL_ERROR`, `RULES_PARITY_*`).

Operators can reliably filter logs on these fields when following the runbooks to correlate HTTP errors, rate limiting, and rules-parity incidents with specific requests and users.

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

## Client/Frontend

These variables are injected into the Vite-built client bundle. **Never**
store secrets in `VITE_*` variables; they are public.

### `VITE_API_URL`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Optional override for the client API base URL. When unset, the client uses
same-origin `/api` (with a dev fallback to `http://localhost:3000` when
running on `localhost:5173`).

### `VITE_WS_URL`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Optional override for the client WebSocket base URL. When unset, the client
derives a WebSocket origin from `VITE_API_URL` or the browser origin.

### `VITE_ERROR_REPORTING_ENABLED`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `false`   |
| Required | No        |

Enable lightweight client-side error reporting to the backend.

### `VITE_ERROR_REPORTING_ENDPOINT`

| Property | Value                |
| -------- | -------------------- |
| Type     | `string`             |
| Default  | `/api/client-errors` |
| Required | No                   |

Endpoint path used for client error reports when enabled.

### `VITE_ERROR_REPORTING_MAX_EVENTS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `50`     |
| Required | No       |

Maximum client error reports per browser session.

### `VITE_DIFFICULTY_CALIBRATION_TELEMETRY_ENABLED`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `true`    |
| Required | No        |

Enable AI difficulty calibration telemetry. Set to `false` to disable.

### `VITE_RULES_UX_TELEMETRY_ENABLED`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `true`    |
| Required | No        |

Enable rules UX telemetry. Set to `false` to disable.

### `VITE_RULES_UX_HELP_OPEN_SAMPLE_RATE`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `1.0`    |
| Required | No       |

Sampling rate (0.0-1.0) for `rules_help_open` telemetry events.

### `VITE_CLIENT_BUILD`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Optional client build identifier used in telemetry/debugging.

### `VITE_GIT_SHA`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Optional git SHA used in telemetry/debugging.

### `VITE_APP_VERSION`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Optional app version string used in telemetry/debugging.

---

## Feature Flags

### `ENABLE_METRICS`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `true`    |
| Required | No        |

Enable Prometheus metrics endpoint.

When `false`, the `/metrics` endpoint is not exposed and HTTP metrics middleware is disabled.

### `ENABLE_HEALTH_CHECKS`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `true`    |
| Required | No        |

Enable health check endpoints.

When `false`, `/health`, `/ready`, and `/api/internal/health/*` are not registered.

### `METRICS_PORT`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Range    | 1-65535  |
| Default  | `9090`   |
| Required | No       |

Prometheus metrics server port.

If `METRICS_PORT` is **explicitly set** and differs from `PORT`, the `/metrics` endpoint is served on that port instead of the main API port. When `METRICS_PORT` is unset, `/metrics` is served on the main port.

### `METRICS_API_KEY`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

When set, requests to `/metrics` must include the key in either the
`Authorization: Bearer <key>` header or the `X-Metrics-Key` header.

### `WS_RECONNECTION_TIMEOUT_MS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `30000`  |
| Required | No       |

Maximum duration (in milliseconds) that a disconnected WebSocket client is allowed to reconnect to an active game before its pending choices are cleared and abandonment handling may apply. This controls the reconnection window described in the WebSocket lifecycle sections of [`docs/architecture/CANONICAL_ENGINE_API.md`](../architecture/CANONICAL_ENGINE_API.md).

In production, this should typically remain at the default `30000` (30 seconds) unless coordinated changes are made to the documented lifecycle and associated tests. Shorter windows may be appropriate for load tests or CI harnesses; longer windows increase server-side resource retention and the risk of stale sessions.

### `ORCHESTRATOR_ADAPTER_ENABLED`

| Property | Value                             |
| -------- | --------------------------------- |
| Type     | `boolean` (hardcoded)             |
| Values   | `true` (always)                   |
| Default  | `true` (hardcoded)                |
| Required | No                                |
| Since    | PASS20 (2025-12-01)               |
| Modified | **Hardcoded to `true` in PASS20** |

**PERMANENTLY ENABLED** as of PASS20 (2025-12-01). The orchestrator adapter is now the canonical turn processor. This variable is **hardcoded to `true`** and no longer reads from environment variables.

The `useOrchestratorAdapter` property on [`GameEngine`](../../src/server/game/GameEngine.ts) and [`ClientSandboxEngine`](../../src/client/sandbox/ClientSandboxEngine.ts) remains for internal state management but always evaluates to true. The legacy turn processing path has been completely removed.

**Migration Completed:**

- Phase 3 migration complete (PASS20)
- ~1,118 lines of legacy code removed
- All 2,987+ TypeScript tests passing with orchestrator
- Zero invariant violations in soak tests

**Related Documentation:**

- [`ORCHESTRATOR_MIGRATION_COMPLETION_PLAN.md`](../archive/ORCHESTRATOR_MIGRATION_COMPLETION_PLAN.md) (archived)
- [`PASS20_COMPLETION_SUMMARY.md`](../archive/assessments/PASS20_COMPLETION_SUMMARY.md)

### `ENABLE_ANALYSIS_MODE`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `false`   |
| Required | No        |

Enable AI analysis mode for position evaluation streaming. When enabled, allows clients to request continuous position analysis from the AI service.

### `ENABLE_SANDBOX_AI_ENDPOINTS`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `false`   |
| Required | No        |

Enable sandbox AI helper endpoints (for example `/api/games/sandbox/ai/move`). These are development/test tools by default but may be enabled in staging-like environments for QA and sandbox validation.

### `ENABLE_HTTP_MOVE_HARNESS`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `false`   |
| Required | No        |

Controls availability of the **internal HTTP move harness** endpoint
(`POST /api/games/:gameId/moves`) on the backend. When set to `true`,
the route is registered as a thin adapter over the same canonical move
pipeline used by WebSocket move handlers. When `false` or unset, the
harness route is disabled and will typically respond with `404 Not Found`
for move submissions.

This flag is intended only for **internal/test** environments such as
local development, CI, dedicated load-testing, or tightly scoped staging
environments. It is **not** a general public move API for interactive
clients and must not be exposed as such.

For full semantics, security constraints, and recommended environment
defaults, see:

- [`PLAYER_MOVE_TRANSPORT_DECISION.md`](../architecture/PLAYER_MOVE_TRANSPORT_DECISION.md)
- The "Internal / Test harness APIs" section of
  [`API_REFERENCE.md`](../architecture/API_REFERENCE.md)

### `HTTP_MOVE_HARNESS_TIMEOUT_MS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `30000`  |
| Required | No       |

Timeout (milliseconds) for HTTP move harness requests (`POST /api/games/:gameId/moves`). If a request exceeds this threshold, the endpoint returns `504 Gateway Timeout` with `SERVER_GATEWAY_TIMEOUT`. This prevents unbounded hangs when the game lock or rules engine deadlocks.

### FSM Validation Mode

These variables control the Finite State Machine (FSM) validation layer that enforces phase/type correctness for all moves.

> **Note:** FSM is now the canonical game state orchestrator (RR-CANON compliance).
> Shadow mode has been fully removed. See `CANONICAL_ENGINE_API.md` for details.

#### `RINGRIFT_FSM_VALIDATION_MODE`

| Property | Value           |
| -------- | --------------- |
| Type     | `enum`          |
| Values   | `off`, `active` |
| Default  | `active`        |
| Required | No              |

Controls FSM validation behavior:

- `off`: No FSM validation (legacy mode, not recommended)
- `active`: FSM validation is authoritative, rejects invalid moves (default, canonical)

> **Note:** The `shadow` value was removed when FSM became canonical. FSM is now the
> single source of truth for game state validation. See `CANONICAL_ENGINE_API.md`.

**Related:**

- Validation script: `scripts/validate-fsm-active-mode.ts`
- FSM implementation: `src/shared/engine/fsm/`

#### `RINGRIFT_FSM_STRUCTURED_LOGGING`

| Property | Value       |
| -------- | ----------- |
| Type     | `boolean`   |
| Values   | `1`, `true` |
| Default  | `false`     |
| Required | No          |

Enable JSON structured logging for FSM validation events. When enabled, emits one JSON line per validated move with fields:

```json
{
  "event": "fsm_validation",
  "timestamp": "2025-12-07T...",
  "mode": "active",
  "gameId": "...",
  "moveNumber": 42,
  "moveType": "movement",
  "currentPhase": "main_actions",
  "fsmValid": true
}
```

Use for production monitoring and metrics collection.

#### `RINGRIFT_FSM_SHADOW_VALIDATION` (Removed)

| Property | Value       |
| -------- | ----------- |
| Status   | **Removed** |

This legacy flag has been removed. FSM is now the canonical game state orchestrator
and shadow validation is no longer supported. Use `RINGRIFT_FSM_VALIDATION_MODE=active`
(the new default) for authoritative FSM validation.

---

### Orchestrator rollout controls

These variables are retained for **diagnostics and metrics classification** only.
The orchestrator adapter is permanently enabled, so these settings do **not**
change live routing or roll back to legacy paths. `OrchestratorRolloutService`
still uses them to label engine-selection reasons in metrics and the
`/api/admin/orchestrator/status` endpoint, which can help triage incidents.

For recommended combinations of `NODE_ENV`, `RINGRIFT_APP_TOPOLOGY`,
`RINGRIFT_RULES_MODE`, and the orchestrator flags across CI, staging, and
production phases, see the env/phase presets table in
[`docs/archive/ORCHESTRATOR_ROLLOUT_PHASES.md` §8.1.1](../archive/ORCHESTRATOR_ROLLOUT_PHASES.md#811-environment-and-flag-presets-by-phase).

#### `ORCHESTRATOR_SHADOW_MODE_ENABLED` (REMOVED)

> **DEPRECATED (2025-12-09):** This environment variable has been removed.
> FSM is now the canonical game state orchestrator (RR-CANON compliance).
> Shadow mode is no longer available.

#### `ORCHESTRATOR_ALLOWLIST_USERS`

| Property | Value                      |
| -------- | -------------------------- |
| Type     | `string` (comma-separated) |
| Default  | `""` (empty string)        |
| Required | No                         |

Comma-separated list of **user IDs** that are always **classified** as
orchestrator-selected for metrics/logging. This does **not** change runtime
behaviour (the orchestrator is always on), but it helps tag internal/test
accounts during diagnostics.

#### `ORCHESTRATOR_DENYLIST_USERS`

| Property | Value                      |
| -------- | -------------------------- |
| Type     | `string` (comma-separated) |
| Default  | `""` (empty string)        |
| Required | No                         |

Comma-separated list of **user IDs** that are classified as **legacy-selected**
for metrics/logging. This does **not** change runtime behaviour, but it helps
label sensitive accounts when reviewing rollout metrics or circuit-breaker
events.

#### `ORCHESTRATOR_CIRCUIT_BREAKER_ENABLED`

| Property | Value     |
| -------- | --------- |
| Type     | `boolean` |
| Default  | `true`    |
| Required | No        |

Enables the orchestrator **circuit breaker state tracking**. When enabled,
`OrchestratorRolloutService` will open/close the breaker based on the thresholds
below and emit the `ringrift_orchestrator_circuit_breaker_state` metric. This
does **not** auto-disable the orchestrator (routing is fixed), but it provides
an incident signal.

Set to `false` only for controlled experiments where circuit-breaker telemetry
is not desired (for example, during development or in tightly monitored staging
runs).

#### `ORCHESTRATOR_ERROR_THRESHOLD_PERCENT`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Range    | 0-100    |
| Default  | `5`      |
| Required | No       |

Percentage of orchestrator-processed requests within the error window that may
fail **before** the circuit breaker trips. This updates breaker state and
metrics; it does not change routing.

- At `5` (default), if more than 5% of orchestrator turns fail within the
  configured window, the circuit breaker is marked open and recorded in metrics.

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
Used by `OrchestratorRolloutService` to detect regressions and annotate
circuit-breaker metrics (no routing changes are applied).

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

Legacy AI thinking-time budget in milliseconds.

This value is currently **not used to introduce any artificial delay** in AI
responses. Historical code paths used it to pad AI moves for UX; those hooks
have been removed so that AI moves are returned as soon as they are computed.
The variable is kept in the schema for backwards compatibility and may be
repurposed as a soft search-budget hint in future orchestration layers.

### `MAX_SPECTATORS_PER_GAME`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `50`     |
| Required | No       |

Maximum spectators per game.

### Decision Phase Timeouts

Optional overrides for decision-phase timeouts. Primarily intended for non-production environments and specialized test harnesses (e.g., Playwright E2E) that need shorter timeouts to exercise decision timeout behavior end-to-end.

When unset, the server falls back to hard-coded defaults in [`unified.ts`](../../src/server/config/unified.ts): 30s total timeout, 5s warning, 15s extension.

#### `DECISION_PHASE_TIMEOUT_MS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | None     |
| Required | No       |

Total timeout for decision phase in milliseconds. Default: 30000ms (30s) when unset.

#### `DECISION_PHASE_TIMEOUT_WARNING_MS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | None     |
| Required | No       |

Warning threshold for decision phase in milliseconds. Default: 5000ms (5s) when unset.

#### `DECISION_PHASE_TIMEOUT_EXTENSION_MS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | None     |
| Required | No       |

Extension duration for decision phase in milliseconds. Default: 15000ms (15s) when unset.

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

| Property | Value          |
| -------- | -------------- |
| Type     | `enum`         |
| Values   | `ts`, `python` |
| Default  | `ts`           |
| Required | No             |

Rules engine mode:

- `ts`: TypeScript engine only (default)
- `python`: Python engine is authoritative

**Note:** Legacy `shadow` mode was removed; any `shadow` value is treated as `ts`.

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

These variables control the data retention policies enforced by [`DataRetentionService`](../../src/server/services/DataRetentionService.ts). The service provides configurable retention periods for different data types and can be scheduled via cron for automated cleanup.

### `DATA_RETENTION_DELETED_USERS_DAYS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | `30`     |
| Required | No       |

Days to retain soft-deleted user accounts before permanent deletion. After this period, [`hardDeleteExpiredUsers()`](../../src/server/services/DataRetentionService.ts) will permanently remove the user record and associated data.

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

Days to retain unverified user accounts before soft-deleting them. [`cleanupUnverifiedAccounts()`](../../src/server/services/DataRetentionService.ts) marks these accounts as deleted with `isActive: false`.

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

Days to retain expired refresh tokens before cleanup. [`cleanupExpiredTokens()`](../../src/server/services/DataRetentionService.ts) removes tokens that have been expired or revoked beyond this threshold.

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

- [Data Lifecycle and Privacy](../security/DATA_LIFECYCLE_AND_PRIVACY.md) - Full S-05.E implementation details
- [API Reference](../architecture/API_REFERENCE.md) - Account deletion and data export endpoints

---

## Python Training Flags

These flags configure the standalone Python training and self-play stack under
`ai-service/`. They are consumed by `RingRiftEnv` and data-generation scripts
and **do not affect the live TypeScript game server**.

### `RINGRIFT_TRAINING_DISABLE_SWAP_RULE`

| Property | Value                                       |
| -------- | ------------------------------------------- |
| Type     | `boolean` (string flag)                     |
| Values   | `1`, `true`, `yes`, `on` (case-insensitive) |
| Default  | `false` (flag unset or any other value)     |
| Required | No                                          |

Controls whether the **swap (pie) rule** is available in 2-player Python
training and evaluation environments.

Behaviour:

- **When unset / falsey** (default):
  - 2-player environments created via `create_initial_state()` / `RingRiftEnv`
    default to `rulesOptions.swapRuleEnabled = True`.
  - The `swap_sides` meta-move is available to player 2 after player 1's first
    non-swap move (subject to the usual gate and "at most once" invariant).
- **When set to a truthy value** (`1`, `true`, `yes`, `on`):
  - 2-player environments are created with the pie rule **disabled**
    (`rulesOptions.swapRuleEnabled` is omitted), and `swap_sides` will not be
    emitted by the Python rules engine.
- **3-player and 4-player games** never enable the pie rule, regardless of this
  flag.

Historical behaviour:

- Training previously used an opt-in `RINGRIFT_TRAINING_ENABLE_SWAP_RULE` flag
  to enable the pie rule.
- That opt-in is now effectively **superseded**: 2-player training defaults to
  swap **enabled** when `RINGRIFT_TRAINING_DISABLE_SWAP_RULE` is absent.
- New experiments should prefer this single opt-out flag; the implementation in
  `create_initial_state()` is the canonical source of truth.

**References:**

- Implementation: `ai-service/app/training/generate_data.py::create_initial_state`
- Tests: `ai-service/tests/test_env_interface.py`

### `RINGRIFT_TRAINED_HEURISTIC_PROFILES`

| Property | Value           |
| -------- | --------------- |
| Type     | `string` (path) |
| Default  | None            |
| Required | No              |

Path to a JSON bundle of trained heuristic weights (for example `data/trained_heuristic_profiles.json`).
When set, the AI service loads the file at startup and merges profiles into
`HEURISTIC_WEIGHT_PROFILES` so ladder tiers can point at updated weights without code changes.

Notes:

- `ai-service/run.sh` auto-sets this to `data/trained_heuristic_profiles.json` if the file exists.
- Produced by CMA-ES / heuristic training flows (see `ai-service/scripts/run_iterative_cmaes.py`).

### `RINGRIFT_DISABLE_NEURAL_NET`

| Property | Value                                       |
| -------- | ------------------------------------------- |
| Type     | `boolean` (string flag)                     |
| Values   | `1`, `true`, `yes`, `on` (case-insensitive) |
| Default  | `false`                                     |
| Required | No                                          |

Globally disable neural network usage in the Python AI service (Minimax/MCTS/Descent).
When enabled, tiers that would normally use NN fall back to heuristic evaluation.

### `RINGRIFT_FORCE_CPU`

| Property | Value                                       |
| -------- | ------------------------------------------- |
| Type     | `boolean` (string flag)                     |
| Values   | `1`, `true`, `yes`, `on` (case-insensitive) |
| Default  | `false`                                     |
| Required | No                                          |

Force neural-net inference/training to use CPU even when CUDA is available.
Used by some training utilities and debugging flows.

### `RINGRIFT_MINIMAX_ZERO_SUM_EVAL`

| Property | Value                                       |
| -------- | ------------------------------------------- |
| Type     | `boolean` (string flag)                     |
| Values   | `1`, `true`, `yes`, `on` (case-insensitive) |
| Default  | `true`                                      |
| Required | No                                          |

Controls whether Minimax uses zero-sum evaluation (`my_eval - opp_eval`) when
computing scores. Disable only for debugging.

### `RINGRIFT_CMAES_WORKERS`

| Property | Value    |
| -------- | -------- |
| Type     | `number` |
| Default  | None     |
| Required | No       |

Number of CMA-ES workers for distributed heuristic training. Used by
`scripts/run_tier_training_pipeline.py` when selecting CMA-ES modes.

## Debug Flags

These flags are for development and debugging only. **Do not enable in production.**

| Variable                                       | Default | Description                                                                                              |
| ---------------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS` | `false` | Enable sandbox AI stall diagnostics                                                                      |
| `RINGRIFT_SANDBOX_CAPTURE_DEBUG`               | `false` | Enable sandbox capture debug logging                                                                     |
| `RINGRIFT_SANDBOX_AI_CAPTURE_DEBUG`            | `false` | Enable sandbox AI capture debug logging                                                                  |
| `RINGRIFT_SANDBOX_AI_TRACE_MODE`               | `false` | Enable sandbox AI trace mode                                                                             |
| `RINGRIFT_SANDBOX_AI_PARITY_MODE`              | `false` | Enable sandbox AI parity mode                                                                            |
| `RINGRIFT_LOCAL_AI_HEURISTIC_MODE`             | `false` | Enable local AI heuristic mode                                                                           |
| `RINGRIFT_TRACE_DEBUG`                         | `false` | Enable high-detail trace logging (capture, territory, orchestrator S-invariant) to logs/console in tests |
| `RINGRIFT_AI_DEBUG`                            | `false` | Mirror AI diagnostics from logs/ai/\*.log to the console for local debugging                             |

---

## Parity & Replay Tooling

These variables are optional and primarily used for deep parity analysis,
replay debugging, and golden-game regression checks. They do **not**
affect core game semantics.

### `RINGRIFT_RULES_ENGINE_VERSION`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Optional label for the current shared rules engine / ruleset version.

- When set, the AI service’s recording helpers attach this value as
  `rules_engine_version` in `games.metadata_json` for all newly
  recorded games (self-play soaks, training harnesses, CMA‑ES, etc.).
- Intended usage:
  - Tag recordings by rules rollout (e.g. `"2025-12-chain-capture-fix"`).
  - Slice replay/parity analyses by engine version.

### `RINGRIFT_PARITY_VALIDATION`

| Property | Value                   |
| -------- | ----------------------- |
| Type     | `enum`                  |
| Values   | `off`, `warn`, `strict` |
| Default  | `off`                   |
| Required | No                      |

Controls parity validation mode when recording games in the Python service.
`strict` raises on mismatches, `warn` logs discrepancies, `off` disables parity checks.

### `RINGRIFT_PARITY_DUMP_DIR`

| Property | Value             |
| -------- | ----------------- |
| Type     | `string`          |
| Default  | `parity_failures` |
| Required | No                |

Directory where parity tooling writes failure bundles and dumps.

### `RINGRIFT_SKIP_SHADOW_CONTRACTS`

| Property | Value                                       |
| -------- | ------------------------------------------- |
| Type     | `boolean` (string flag)                     |
| Values   | `1`, `true`, `yes`, `on` (case-insensitive) |
| Default  | `true`                                      |
| Required | No                                          |

Skip mutator shadow validation in the Python rules engine. Used for faster self-play;
set to `false` for debugging or when validating rule changes.

### `RINGRIFT_TS_ENGINE_VERSION`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Optional label for the TypeScript/shared engine build version that
produced the recordings.

- When set, recorded games include `ts_engine_version` in
  `games.metadata_json`.
- Useful for correlating TS replay/parity divergences with specific
  backend/sandbox engine revisions.

### `RINGRIFT_AI_SERVICE_VERSION`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Optional label for the Python AI service build or deployment version.

- When set, recorded games include `ai_service_version` in
  `games.metadata_json`.
- Intended for:
  - Tracking self-play / training runs across AI-service releases.
  - Post-hoc analysis of parity or invariant regressions.

### `RINGRIFT_PARITY_GOLDEN_DB`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

Path to a **golden** `GameReplayDB` SQLite file used by the Python
**differential replay** tests.

- Consumed by `ai-service/tests/parity/test_differential_replay.py`.
- When set together with `RINGRIFT_PARITY_GOLDEN_GAME_ID` and the DB
  exists, the golden-game test:
  - Replays the configured game via Python and TS,
  - Fails if any divergences are detected.

### `RINGRIFT_PARITY_GOLDEN_GAME_ID`

| Property | Value    |
| -------- | -------- |
| Type     | `string` |
| Default  | None     |
| Required | No       |

`game_id` within `RINGRIFT_PARITY_GOLDEN_DB` to treat as a golden
replay/parity case.

- Must correspond to a row in the `games` table of the DB configured via
  `RINGRIFT_PARITY_GOLDEN_DB`.
- Intended for:
  - Locking in previously-debugged problem games (e.g. specific
    chain-capture or pie-rule sequences) as permanent parity
    regression tests.

---

## Deprecated Variables

These variables were removed during project evolution. They are listed here for historical reference; setting them today has no effect.

### `ORCHESTRATOR_ROLLOUT_PERCENTAGE`

| Property | Value                                    |
| -------- | ---------------------------------------- |
| Type     | `number` (0-100)                         |
| Removed  | PASS20 (2025-12-01)                      |
| Reason   | Orchestrator permanently enabled at 100% |

**Status:** REMOVED

Percentage of eligible sessions to route through the orchestrator. This variable controlled gradual rollout during the orchestrator migration phases.

**Migration Path:**

- Phase 1-2: Variable controlled canary rollout (0-100%)
- PASS20 Phase 3: Orchestrator hardcoded to 100%, variable deprecated
- The orchestrator is now permanently enabled and cannot be disabled

**If Present:** This variable is ignored. The orchestrator always processes 100% of traffic.

**Related Changes:**

- [`ORCHESTRATOR_ADAPTER_ENABLED`](#orchestrator_adapter_enabled) - Now hardcoded to `true`
- See [`PASS20_COMPLETION_SUMMARY.md`](../archive/assessments/PASS20_COMPLETION_SUMMARY.md) for migration details

### `SOCKET_PORT`

| Property | Value                             |
| -------- | --------------------------------- |
| Type     | `number`                          |
| Removed  | Legacy WebSocket split (pre-2025) |
| Reason   | WebSocket now shares `PORT`       |

**Status:** REMOVED

Legacy WebSocket port configuration. The server now binds WebSocket traffic to the main `PORT`, and `SOCKET_PORT` is ignored.

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

- [Secrets Management Guide](SECRETS_MANAGEMENT.md)
- [Security Threat Model](../security/SECURITY_THREAT_MODEL.md)
- [Operations Database Reference](OPERATIONS_DB.md)
