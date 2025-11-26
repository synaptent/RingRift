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

### User (`/api/user`)

| Method | Endpoint            | Description                        | Auth Required |
| ------ | ------------------- | ---------------------------------- | ------------- |
| GET    | `/user/profile`     | Get current user's profile         | ✅            |
| PUT    | `/user/profile`     | Update current user's profile      | ✅            |
| GET    | `/user/stats`       | Get current user's game statistics | ✅            |
| GET    | `/user/games`       | Get current user's game history    | ✅            |
| GET    | `/user/search`      | Search users by username           | ✅            |
| GET    | `/user/leaderboard` | Get leaderboard rankings           | ✅            |
| DELETE | `/user/me`          | Delete current user's account      | ✅            |

### Games (`/api/games`)

| Method | Endpoint                 | Description                  | Auth Required |
| ------ | ------------------------ | ---------------------------- | ------------- |
| GET    | `/games`                 | List user's games            | ✅            |
| POST   | `/games`                 | Create a new game            | ✅            |
| GET    | `/games/:gameId`         | Get game details             | ✅            |
| POST   | `/games/:gameId/join`    | Join an existing game        | ✅            |
| POST   | `/games/:gameId/leave`   | Leave a game                 | ✅            |
| GET    | `/games/:gameId/moves`   | Get game move history        | ✅            |
| GET    | `/games/lobby/available` | List available games to join | ✅            |

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

#### Authentication Errors (AUTH\_\*)

| Code                         | HTTP Status | Description                   |
| ---------------------------- | ----------- | ----------------------------- |
| `AUTH_INVALID_CREDENTIALS`   | 401         | Invalid username or password  |
| `AUTH_TOKEN_EXPIRED`         | 401         | Access token has expired      |
| `AUTH_TOKEN_INVALID`         | 401         | Token is malformed or invalid |
| `AUTH_UNAUTHORIZED`          | 401         | Authentication required       |
| `AUTH_FORBIDDEN`             | 403         | Insufficient permissions      |
| `AUTH_EMAIL_NOT_VERIFIED`    | 403         | Email verification required   |
| `AUTH_REFRESH_TOKEN_INVALID` | 401         | Invalid refresh token         |
| `AUTH_REFRESH_TOKEN_EXPIRED` | 401         | Refresh token has expired     |
| `AUTH_USER_NOT_FOUND`        | 404         | User account not found        |

#### Validation Errors (VALIDATION\_\*)

| Code                        | HTTP Status | Description               |
| --------------------------- | ----------- | ------------------------- |
| `VALIDATION_ERROR`          | 400         | Request validation failed |
| `VALIDATION_INVALID_FORMAT` | 400         | Invalid data format       |
| `VALIDATION_REQUIRED_FIELD` | 400         | Required field missing    |
| `VALIDATION_INVALID_VALUE`  | 400         | Invalid field value       |

#### Resource Errors (RESOURCE\_\*)

| Code                 | HTTP Status | Description                  |
| -------------------- | ----------- | ---------------------------- |
| `RESOURCE_NOT_FOUND` | 404         | Resource does not exist      |
| `RESOURCE_CONFLICT`  | 409         | Resource already exists      |
| `RESOURCE_GONE`      | 410         | Resource no longer available |

#### Game Errors (GAME\_\*)

| Code                   | HTTP Status | Description                |
| ---------------------- | ----------- | -------------------------- |
| `GAME_NOT_FOUND`       | 404         | Game does not exist        |
| `GAME_ALREADY_STARTED` | 400         | Game has already started   |
| `GAME_FULL`            | 400         | Game is at maximum players |
| `GAME_NOT_STARTED`     | 400         | Game has not started       |
| `GAME_ALREADY_ENDED`   | 400         | Game has already ended     |
| `GAME_NOT_YOUR_TURN`   | 400         | Not your turn to move      |
| `GAME_INVALID_MOVE`    | 400         | Invalid game move          |

#### User Errors (USER\_\*)

| Code                  | HTTP Status | Description              |
| --------------------- | ----------- | ------------------------ |
| `USER_NOT_FOUND`      | 404         | User does not exist      |
| `USER_ALREADY_EXISTS` | 409         | Username/email taken     |
| `USER_EMAIL_TAKEN`    | 409         | Email already registered |
| `USER_USERNAME_TAKEN` | 409         | Username already taken   |

#### Rate Limiting Errors (RATE\_\*)

| Code                  | HTTP Status | Description       |
| --------------------- | ----------- | ----------------- |
| `RATE_LIMIT_EXCEEDED` | 429         | Too many requests |

#### Server Errors (INTERNAL\_\*)

| Code                  | HTTP Status | Description                     |
| --------------------- | ----------- | ------------------------------- |
| `INTERNAL_ERROR`      | 500         | Internal server error           |
| `SERVICE_UNAVAILABLE` | 503         | Service temporarily unavailable |

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

---

## � Related Documentation

- [WebSocket API](../src/shared/types/websocket.ts) - Real-time game events
- [Game Rules](../ringrift_complete_rules.md) - Complete game rulebook
- [Environment Variables](./ENVIRONMENT_VARIABLES.md) - Server configuration

---

## 📋 Changelog

| Version | Date       | Changes                       |
| ------- | ---------- | ----------------------------- |
| 1.0.0   | 2024-11-25 | Initial OpenAPI documentation |
