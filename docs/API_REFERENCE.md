# RingRift REST API Reference

This document provides an overview of the RingRift REST API. For interactive documentation with live testing capabilities, visit the Swagger UI.

## 📖 Interactive Documentation

| Resource         | URL              | Description                                             |
| ---------------- | ---------------- | ------------------------------------------------------- |
| **Swagger UI**   | `/api/docs`      | Interactive API explorer with request/response examples |
| **OpenAPI Spec** | `/api/docs.json` | Raw OpenAPI 3.0 specification (JSON)                    |

When running locally: `http://localhost:3000/api/docs`

---

## 🔐 Authentication

The API uses **JWT Bearer tokens** for authentication. Most endpoints require authentication.

### Authentication Flow

1. **Register** a new account via `POST /api/auth/register`
2. **Login** to receive access and refresh tokens via `POST /api/auth/login`
3. Include the access token in requests: `Authorization: Bearer <token>`
4. **Refresh** expired tokens via `POST /api/auth/refresh`

### Token Types

| Token         | Purpose                  | Expiration |
| ------------- | ------------------------ | ---------- |
| Access Token  | API authentication       | 15 minutes |
| Refresh Token | Obtain new access tokens | 7 days     |

---

## 📁 API Endpoints Overview

### Authentication (`/api/auth`)

| Method | Endpoint                | Description                 | Auth Required              |
| ------ | ----------------------- | --------------------------- | -------------------------- |
| POST   | `/auth/register`        | Register a new user account | ❌                         |
| POST   | `/auth/login`           | Login and obtain tokens     | ❌                         |
| POST   | `/auth/refresh`         | Refresh access token        | ❌ (refresh token in body) |
| POST   | `/auth/logout`          | Logout from current device  | ✅                         |
| POST   | `/auth/logout-all`      | Logout from all devices     | ✅                         |
| POST   | `/auth/verify-email`    | Verify email address        | ❌                         |
| POST   | `/auth/forgot-password` | Request password reset      | ❌                         |
| POST   | `/auth/reset-password`  | Reset password with token   | ❌                         |

### User (`/api/users`)

| Method | Endpoint                | Description                        | Auth Required |
| ------ | ----------------------- | ---------------------------------- | ------------- |
| GET    | `/users/profile`        | Get current user's profile         | ✅            |
| PUT    | `/users/profile`        | Update current user's profile      | ✅            |
| GET    | `/users/stats`          | Get current user's game statistics | ✅            |
| GET    | `/users/games`          | Get current user's game history    | ✅            |
| GET    | `/users/search`         | Search users by username           | ✅            |
| GET    | `/users/leaderboard`    | Get leaderboard rankings           | ✅            |
| GET    | `/users/:userId/rating` | Get rating and rank for a user     | ✅            |
| DELETE | `/users/me`             | Delete current user's account      | ✅            |
| GET    | `/users/me/export`      | Export current user's data (GDPR)  | ✅            |

### Games (`/api/games`)

> **Move transport:** Interactive move submission is performed over the WebSocket API, not via a general-purpose HTTP move endpoint. For the canonical transport decision (WebSocket vs HTTP, and the scope of any HTTP move harness), see [`PLAYER_MOVE_TRANSPORT_DECISION.md`](docs/PLAYER_MOVE_TRANSPORT_DECISION.md:1).

| Method | Endpoint                             | Description                       | Auth Required |
| ------ | ------------------------------------ | --------------------------------- | ------------- |
| GET    | `/games`                             | List user's games                 | ✅            |
| POST   | `/games`                             | Create a new game                 | ✅            |
| GET    | `/games/:gameId`                     | Get game details                  | ✅            |
| POST   | `/games/:gameId/join`                | Join an existing game             | ✅            |
| POST   | `/games/:gameId/leave`               | Leave a game                      | ✅            |
| GET    | `/games/:gameId/moves`               | Get game move history             | ✅            |
| GET    | `/games/:gameId/history`             | Get structured move history       | ✅            |
| GET    | `/games/:gameId/diagnostics/session` | Get in-memory session diagnostics | ✅            |
| GET    | `/games/lobby/available`             | List available games to join      | ✅            |
| GET    | `/games/user/:userId`                | Get games for a specific user     | ✅            |

> **Lifecycle / spectator invariant:** The game detail, history, and diagnostics
> routes (`GET /games/:gameId`, `GET /games/:gameId/history`,
> `GET /games/:gameId/diagnostics/session`) all share the same authorization
> rule derived from the lifecycle SSoT (`docs/CANONICAL_ENGINE_API.md`): callers
> must either be **seated participants** in the game or, when
> `allowSpectators=true`, permitted **spectators**. Violations surface as
> standardized `RESOURCE_ACCESS_DENIED` / `GAME_NOT_FOUND` errors and are
> exercised by `tests/unit/gameHistory.routes.test.ts` alongside the core
> WebSocket lifecycle suites.

### Utility Endpoints

| Method | Endpoint         | Description                         | Auth Required |
| ------ | ---------------- | ----------------------------------- | ------------- |
| GET    | `/`              | API info and available endpoints    | ❌            |
| POST   | `/client-errors` | Report client-side errors (for SPA) | ❌            |

### Internal / Test harness APIs

> An internal HTTP move harness endpoint (`POST /api/games/:gameId/moves`) **may be enabled** in certain environments (for example, local, CI, dedicated loadtest, or tightly scoped staging) to support load testing and internal tools. It is a thin adapter over the same shared domain `applyMove` API used by WebSocket move handlers, is typically gated by feature flags such as `ENABLE_HTTP_MOVE_HARNESS`, and is **not** a general public HTTP move API for interactive clients. See [`PLAYER_MOVE_TRANSPORT_DECISION.md`](docs/PLAYER_MOVE_TRANSPORT_DECISION.md:1) for the canonical scope and constraints.

---

## Error Response Format

All errors follow a standardized format:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      // Optional additional context
    }
  },
  "requestId": "uuid-request-identifier"
}
```

### Common Error Codes

The authoritative error code catalog lives in `src/server/errors/errorCodes.ts` and is
surfaced via the `ApiError` helper and the global error handler. The tables
below summarise the **public-facing subsets** of that catalog. Where legacy
codes still exist in logs or historical responses, they are normalised to
these canonical values via `LegacyCodeMapping` before being returned to
clients.

#### Authentication Errors (`AUTH_*`)

| Code                               | HTTP Status | Description                                                                                   |
| ---------------------------------- | ----------- | --------------------------------------------------------------------------------------------- |
| `AUTH_INVALID_CREDENTIALS`         | 401         | Invalid username or password                                                                  |
| `AUTH_TOKEN_INVALID`               | 401         | JWT access token is malformed or has an invalid signature                                     |
| `AUTH_TOKEN_EXPIRED`               | 401         | JWT access token has expired                                                                  |
| `AUTH_REFRESH_TOKEN_INVALID`       | 401         | Refresh token is invalid or not recognised                                                    |
| `AUTH_REFRESH_TOKEN_EXPIRED`       | 401         | Refresh token has expired                                                                     |
| `AUTH_REFRESH_TOKEN_REUSED`        | 401         | Refresh token reuse detected; entire token family revoked for safety                          |
| `AUTH_REFRESH_TOKEN_REQUIRED`      | 400         | Refresh token required but not provided                                                       |
| `AUTH_ACCOUNT_DEACTIVATED`         | 401         | Account has been deactivated (e.g. after GDPR deletion or admin action)                       |
| `AUTH_REQUIRED`                    | 401         | Authentication required for this endpoint                                                     |
| `AUTH_TOKEN_REQUIRED`              | 401         | Authorization header missing or empty                                                         |
| `AUTH_FORBIDDEN`                   | 403         | Authenticated but lacking permission for this action                                          |
| `AUTH_LOGIN_LOCKED_OUT`            | 429         | Too many failed login attempts; temporary lockout in effect                                   |
| `AUTH_VERIFICATION_INVALID`        | 400         | Email verification token is invalid or has expired                                            |
| `AUTH_VERIFICATION_TOKEN_REQUIRED` | 400         | Email verification token is required                                                          |
| `AUTH_RESET_TOKEN_INVALID`         | 400         | Password reset token is invalid or has expired                                                |
| `AUTH_WEAK_PASSWORD`               | 400         | Password does not meet strength requirements (e.g. minimum length enforced by shared schemas) |

#### Validation Errors (`VALIDATION_*`)

| Code                               | HTTP Status | Description                                                 |
| ---------------------------------- | ----------- | ----------------------------------------------------------- |
| `VALIDATION_FAILED`                | 400         | Generic validation failure                                  |
| `VALIDATION_INVALID_REQUEST`       | 400         | Request body is malformed or missing required fields        |
| `VALIDATION_INVALID_FORMAT`        | 400         | Invalid data format (e.g. email, UUID, or enum values)      |
| `VALIDATION_INVALID_QUERY_PARAMS`  | 400         | Invalid query parameters                                    |
| `VALIDATION_INVALID_ID`            | 400         | Invalid ID format                                           |
| `VALIDATION_EMAIL_REQUIRED`        | 400         | Email address is required                                   |
| `VALIDATION_SEARCH_QUERY_REQUIRED` | 400         | Search query (`q`) is required                              |
| `VALIDATION_INVALID_PROFILE_DATA`  | 400         | Invalid profile update payload                              |
| `VALIDATION_INVALID_AI_CONFIG`     | 400         | Invalid AI configuration (e.g. mismatched difficulty array) |
| `VALIDATION_INVALID_DIFFICULTY`    | 400         | AI difficulty is out of the supported range                 |

#### Resource & User Errors (`RESOURCE_*`)

| Code                       | HTTP Status | Description                                |
| -------------------------- | ----------- | ------------------------------------------ |
| `RESOURCE_NOT_FOUND`       | 404         | Generic resource not found                 |
| `RESOURCE_ALREADY_EXISTS`  | 409         | Generic resource already exists (conflict) |
| `RESOURCE_USER_NOT_FOUND`  | 404         | User does not exist                        |
| `RESOURCE_GAME_NOT_FOUND`  | 404         | Game does not exist                        |
| `RESOURCE_EMAIL_EXISTS`    | 409         | Email already registered                   |
| `RESOURCE_USERNAME_EXISTS` | 409         | Username already taken                     |
| `RESOURCE_ROUTE_NOT_FOUND` | 404         | API route not found                        |
| `RESOURCE_ACCESS_DENIED`   | 403         | Access to this resource is denied          |

> Legacy codes such as `USER_NOT_FOUND`, `EMAIL_EXISTS`, and `USERNAME_EXISTS`
> are still recognised internally but are normalised to the `RESOURCE_*` codes
> above before being returned to clients.

#### Game Errors (`GAME_*`)

| Code                  | HTTP Status | Description                                  |
| --------------------- | ----------- | -------------------------------------------- |
| `GAME_NOT_FOUND`      | 404         | Game does not exist                          |
| `GAME_INVALID_ID`     | 400         | Invalid game ID format                       |
| `GAME_NOT_JOINABLE`   | 400         | Game is not accepting new players            |
| `GAME_ALREADY_JOINED` | 400         | Player has already joined this game          |
| `GAME_FULL`           | 400         | Game is at maximum players                   |
| `GAME_ACCESS_DENIED`  | 403         | Access to this game is denied                |
| `GAME_INVALID_MOVE`   | 400         | Invalid game move for the current game state |
| `GAME_NOT_YOUR_TURN`  | 400         | Not your turn to move                        |
| `GAME_ALREADY_ENDED`  | 400         | Game has already ended                       |
| `GAME_AI_UNRATED`     | 400         | AI games cannot be rated                     |

#### Rate Limiting Errors (`RATE_LIMIT_*`)

| Code                     | HTTP Status | Description                                         |
| ------------------------ | ----------- | --------------------------------------------------- |
| `RATE_LIMIT_EXCEEDED`    | 429         | Generic rate limit exceeded                         |
| `RATE_LIMIT_GAME_CREATE` | 429         | Too many games created in a short period            |
| `RATE_LIMIT_AUTH`        | 429         | Authentication endpoints are currently rate limited |

#### Server Errors (`SERVER_*`)

| Code                          | HTTP Status | Description                     |
| ----------------------------- | ----------- | ------------------------------- |
| `SERVER_INTERNAL_ERROR`       | 500         | Internal server error           |
| `SERVER_DATABASE_UNAVAILABLE` | 503         | Database not available          |
| `SERVER_SERVICE_UNAVAILABLE`  | 503         | Service temporarily unavailable |
| `SERVER_EMAIL_SEND_FAILED`    | 500         | Failed to send email            |

#### AI Service Errors (`AI_*`)

| Code                     | HTTP Status | Description              |
| ------------------------ | ----------- | ------------------------ |
| `AI_SERVICE_TIMEOUT`     | 503         | AI service timed out     |
| `AI_SERVICE_UNAVAILABLE` | 503         | AI service unavailable   |
| `AI_SERVICE_ERROR`       | 502         | AI service error         |
| `AI_SERVICE_OVERLOADED`  | 503         | AI service is overloaded |

---

## ⏱️ Rate Limiting

Certain endpoints are rate-limited to prevent abuse:

| Category       | Limit        | Window     |
| -------------- | ------------ | ---------- |
| Authentication | 5 requests   | 15 minutes |
| Password Reset | 3 requests   | 1 hour     |
| General API    | 100 requests | 1 minute   |

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699999999
```

---

## 📝 Request/Response Examples

### Register a New User

**Request:**

```http
POST /api/auth/register
Content-Type: application/json

{
  "username": "newplayer",
  "email": "player@example.com",
  "password": "SecurePassword123!"
}
```

**Success Response (201):**

```json
{
  "success": true,
  "data": {
    "user": {
      "id": "uuid-user-id",
      "username": "newplayer",
      "email": "player@example.com",
      "emailVerified": false,
      "rating": 1200,
      "createdAt": "2024-01-15T10:30:00Z"
    },
    "message": "Registration successful. Please verify your email."
  }
}
```

### Login

**Request:**

```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "player@example.com",
  "password": "SecurePassword123!"
}
```

**Success Response (200):**

```json
{
  "success": true,
  "data": {
    "user": {
      "id": "uuid-user-id",
      "username": "newplayer",
      "email": "player@example.com",
      "emailVerified": true,
      "rating": 1250
    },
    "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

### Create a Game (legacy example)

**Request:**

```http
POST /api/games
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "boardType": "standard",
  "isRanked": false
}
```

**Success Response (201):**

```json
{
  "success": true,
  "data": {
    "game": {
      "id": "uuid-game-id",
      "status": "waiting",
      "boardType": "standard",
      "isRanked": false,
      "players": [
        {
          "id": "uuid-user-id",
          "username": "newplayer",
          "color": "blue"
        }
      ],
      "createdAt": "2024-01-15T11:00:00Z"
    }
  }
}
```

> Note: The legacy example above is preserved for backward compatibility with older docs that referenced `boardType`/`isRanked` only.

### Delete Account (GDPR Compliant)

**Request:**

```http
DELETE /api/users/me
Authorization: Bearer <access_token>
```

**Success Response (200):**

```json
{
  "success": true,
  "message": "Account deleted successfully"
}
```

**Error Response (401):**

```json
{
  "success": false,
  "error": {
    "code": "AUTH_TOKEN_INVALID",
    "message": "Invalid authentication token"
  }
}
```

**Behavior:**

- Soft-deletes the account (sets `isActive: false`, `deletedAt` timestamp)
- Anonymizes PII (email becomes `deleted+<id>@example.invalid`, username becomes `DeletedPlayer_<short-id>`)
- Revokes all authentication tokens (increments `tokenVersion`)
- Deletes all refresh tokens
- Terminates active WebSocket sessions
- Preserves game history and ratings for integrity (but displayed as "Deleted Player")

### Export User Data (GDPR Data Portability)

**Request:**

```http
GET /api/users/me/export
Authorization: Bearer <access_token>
```

**Success Response (200):**

Response includes `Content-Disposition: attachment; filename="ringrift-data-export-<timestamp>.json"` header.

```json
{
  "exportedAt": "2024-01-15T12:00:00.000Z",
  "exportFormat": "1.0",
  "profile": {
    "id": "uuid-user-id",
    "username": "newplayer",
    "email": "player@example.com",
    "createdAt": "2024-01-01T10:00:00.000Z",
    "emailVerified": true,
    "role": "player",
    "isActive": true,
    "lastLoginAt": "2024-01-15T09:00:00.000Z",
    "updatedAt": "2024-01-15T09:00:00.000Z"
  },
  "statistics": {
    "rating": 1250,
    "gamesPlayed": 42,
    "wins": 25,
    "losses": 17,
    "winRate": 0.595
  },
  "games": [
    {
      "id": "uuid-game-id",
      "createdAt": "2024-01-14T14:00:00.000Z",
      "startedAt": "2024-01-14T14:01:00.000Z",
      "completedAt": "2024-01-14T15:30:00.000Z",
      "status": "finished",
      "result": "win",
      "boardType": "standard",
      "isRated": true,
      "opponent": "opponent_username",
      "moves": [
        {
          "moveNumber": 1,
          "data": { "type": "placement", "to": "a1" },
          "isUserMove": true
        }
      ]
    }
  ],
  "preferences": {}
}
```

**Notes:**

- Opponents who have deleted their accounts appear as `"Deleted Player"` in the export
- Sensitive fields are excluded: `passwordHash`, `tokenVersion`, `deletedAt`, tokens, etc.
- Move history includes all moves from games where the user participated
- The `isUserMove` flag indicates which moves were made by the exporting user

#### Worked example: `GET /api/users/me/export`

The data export endpoint is driven by the same OpenAPI source as other user routes (see [`src/server/openapi/config.ts`](../src/server/openapi/config.ts)) and implemented in the Users routes/handlers (`src/server/routes/user.ts` and associated controllers). Its concrete behaviour is validated by:

- `tests/integration/dataLifecycle.test.ts` – end-to-end tests under “GET /api/users/me/export” that assert:
  - The response uses `Content-Disposition` with a `ringrift-data-export-<timestamp>.json` filename.
  - The JSON body includes `profile`, `statistics`, `games`, and `preferences` sections shaped as shown above.
  - Deleted opponents are rendered as `"Deleted Player"` while preserving game integrity.
- Additional route wiring and auth behaviour is covered by `tests/unit/server.health-and-routes.test.ts`.

Clients calling `GET /api/users/me/export` should therefore expect:

- A 200 success response with a downloadable JSON attachment containing all user-visible data for the authenticated account.
- A 401 error (for example with code `AUTH_TOKEN_INVALID` or `AUTH_TOKEN_REQUIRED`) when the access token is missing or invalid, matching the standard error envelope described earlier in this document.

#### Worked example: `DELETE /api/users/me`

The account deletion endpoint is defined and documented via OpenAPI annotations in the user routes and aggregated through [`src/server/openapi/config.ts`](../src/server/openapi/config.ts). Its behaviour is enforced by both unit and integration tests:

- Route and controller logic live under the Users routes (`src/server/routes/user.ts` and related handlers).
- HTTP-level behaviour (status codes and error codes such as `AUTH_TOKEN_INVALID` or `AUTH_TOKEN_REQUIRED`) is covered by:
  - `tests/unit/user.delete.routes.test.ts` – unit-style tests for authentication and soft-delete semantics.
  - `tests/unit/server.health-and-routes.test.ts` – route wiring and 401 behaviour for unauthenticated requests.
  - `tests/integration/accountDeletion.test.ts` – end-to-end verification that deletion anonymizes PII, revokes tokens, and terminates WebSocket sessions.

Clients calling `DELETE /api/users/me` should therefore expect:

- A 200 success response with the structure shown above when the authenticated user is deleted successfully.
- A 401 error (for example with code `AUTH_TOKEN_INVALID` or `AUTH_TOKEN_REQUIRED`) when the request is missing or presents an invalid or expired access token.

---

## 🔌 WebSocket API

Real-time game communication uses Socket.IO over WebSockets. Authentication is required via JWT token passed in the handshake.

WebSocket is the canonical move transport for interactive clients; any HTTP move endpoint that submits moves is an internal/test harness over the same shared domain API, as documented in [`PLAYER_MOVE_TRANSPORT_DECISION.md`](docs/PLAYER_MOVE_TRANSPORT_DECISION.md:1).

For the **authoritative Move / PendingDecision / PlayerChoice / WebSocket lifecycle** (including concrete type definitions and a worked example), see [`docs/CANONICAL_ENGINE_API.md` §3.9–3.10](./CANONICAL_ENGINE_API.md). This section focuses on transport-level events and error codes; it assumes that orchestrator-centric lifecycle as its rules SSoT.

### Connection

```javascript
import { io } from 'socket.io-client';

const socket = io('http://localhost:3000', {
  auth: { token: 'your-jwt-access-token' },
  transports: ['websocket', 'polling'],
});
```

### Client → Server Events

| Event                    | Payload                                      | Description                        |
| ------------------------ | -------------------------------------------- | ---------------------------------- |
| `join_game`              | `{ gameId: string }`                         | Join a game room                   |
| `leave_game`             | `{ gameId: string }`                         | Leave a game room                  |
| `player_move`            | `{ gameId, move: { from?, to } }`            | Submit a move (geometry-based)     |
| `player_move_by_id`      | `{ gameId, moveId: string }`                 | Submit a move by canonical Move.id |
| `chat_message`           | `{ gameId, text: string }`                   | Send chat message                  |
| `player_choice_response` | `{ choiceId, playerNumber, selectedOption }` | Respond to a player choice prompt  |
| `lobby:subscribe`        | (none)                                       | Subscribe to lobby updates         |
| `lobby:unsubscribe`      | (none)                                       | Unsubscribe from lobby updates     |

### Server → Client Events

| Event                            | Description                                                                                                                                                                                                                                                             |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `game_state`                     | Full game state update (on join and after moves)                                                                                                                                                                                                                        |
| `game_over`                      | Game ended (includes final state and result)                                                                                                                                                                                                                            |
| `game_error`                     | Fatal game error (e.g., AI service failure)                                                                                                                                                                                                                             |
| `player_joined`                  | Another player joined the game room                                                                                                                                                                                                                                     |
| `player_left`                    | A player left the game room                                                                                                                                                                                                                                             |
| `player_disconnected`            | A player disconnected (may reconnect)                                                                                                                                                                                                                                   |
| `player_reconnected`             | A player reconnected to the same game within the configured reconnection window (see connection state machine in `docs/STATE_MACHINES.md` and `tests/unit/WebSocketServer.connectionState.test.ts`)                                                                     |
| `chat_message`                   | Broadcast chat message                                                                                                                                                                                                                                                  |
| `time_update`                    | Time control update for a player                                                                                                                                                                                                                                        |
| `player_choice_required`         | Server requests a decision from a player                                                                                                                                                                                                                                |
| `player_choice_canceled`         | A pending choice was canceled                                                                                                                                                                                                                                           |
| `decision_phase_timeout_warning` | Warning before auto-resolution of a decision. Emitted a short time (for example, ~5 seconds) before the server will auto-resolve a pending decision due to timeout; carries phase, player, and remaining time metadata so clients can surface a countdown.              |
| `decision_phase_timed_out`       | Decision phase timed out; auto-resolved. Emitted when a pending decision has exceeded its timeout and the engine has applied a default Move (for example, a line reward or region order) on behalf of the player; the auto-selected move id is included in the payload. |
| `error`                          | Structured error payload                                                                                                                                                                                                                                                |

#### Worked example: `player_choice_required` → `player_choice_response`

When the engine reaches an interactive decision (for example a line reward or region order choice), the backend emits a `player_choice_required` event to the relevant player, carrying a typed `PlayerChoice` object as defined in [`src/shared/types/game.ts`](../src/shared/types/game.ts) and surfaced on the socket via [`ServerToClientEvents.player_choice_required`](../src/shared/types/websocket.ts).

- The `WebSocketServer` delivers this event from the current `GameSession` interaction handler; see the choice-response wiring in `src/server/websocket/server.ts` (the `player_choice_response` handler).
- The client listens for `player_choice_required`, renders a choice UI (for example `ChoiceDialog` / `GameContext` in the React app), and replies with a `player_choice_response` event that selects exactly one of the provided options.
- On receipt, the server validates the response against the original `PlayerChoice` (type, player number, option validity) using `WebSocketInteractionHandler.handleChoiceResponse`, which drives the shared `choice.ts` state machine and either:
  - resolves the pending choice and continues the game, or
  - rejects the response and emits a `CHOICE_REJECTED` error when validation fails.

End-to-end behaviour for this flow is exercised by tests such as:

- `tests/unit/WebSocketInteractionHandler.test.ts` – lifecycle and validation of `PlayerChoice` requests/responses.
- `tests/unit/GameEngine.lineRewardChoiceWebSocketIntegration.test.ts` – full WebSocket + engine integration for line reward decisions.

### Lobby Events (Server → Client)

| Event                  | Payload                                      | Description                |
| ---------------------- | -------------------------------------------- | -------------------------- |
| `lobby:game_created`   | Game object                                  | New game available to join |
| `lobby:game_joined`    | `{ gameId, playerCount }`                    | Player count updated       |
| `lobby:game_started`   | `{ gameId, status, startedAt, playerCount }` | Game has started           |
| `lobby:game_cancelled` | `{ gameId }`                                 | Game was cancelled         |

### Diagnostic / Load‑Testing Events

These events are transport-only and do not mutate game state. They are primarily used by k6 scenarios (see `tests/load/scenarios/websocket-stress.js`) and internal diagnostics.

| Direction       | Event             | Payload                                                                                    | Description                                                                                       |
| --------------- | ----------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Client → Server | `diagnostic:ping` | `{ timestamp: number, vu?: string \| number, sequence?: number }`                          | Lightweight ping used to measure WebSocket round‑trip latency without touching rules or database. |
| Server → Client | `diagnostic:pong` | `{ timestamp: number, vu?: string \| number, sequence?: number, serverTimestamp: string }` | Echo of the original ping payload with an added `serverTimestamp`; used for latency calculations. |

### WebSocket Error Codes

| Code                     | Description                          |
| ------------------------ | ------------------------------------ |
| `INVALID_PAYLOAD`        | Malformed event payload              |
| `GAME_NOT_FOUND`         | Game does not exist                  |
| `ACCESS_DENIED`          | Not authorized for this action       |
| `RATE_LIMITED`           | Too many requests (e.g., chat spam)  |
| `MOVE_REJECTED`          | Move not valid in current game state |
| `CHOICE_REJECTED`        | Choice response not valid            |
| `DECISION_PHASE_TIMEOUT` | Decision phase timed out             |
| `INTERNAL_ERROR`         | Server-side error                    |

---

## 🔗 Related Documentation

- [WebSocket Types](../src/shared/types/websocket.ts) - TypeScript type definitions
- [Game Rules](../ringrift_complete_rules.md) - Complete game rulebook
- [Environment Variables](./ENVIRONMENT_VARIABLES.md) - Server configuration
- [OpenAPI Spec](../src/server/openapi/config.ts) - OpenAPI schema definitions

---

## 📋 Changelog

| Version | Date       | Changes                                                      |
| ------- | ---------- | ------------------------------------------------------------ |
| 1.2.0   | 2024-11-27 | Added DELETE /api/users/me and GET /api/users/me/export docs |
| 1.1.0   | 2024-11-26 | Added WebSocket API, fixed /users path, added missing codes  |
| 1.0.0   | 2024-11-25 | Initial OpenAPI documentation                                |
