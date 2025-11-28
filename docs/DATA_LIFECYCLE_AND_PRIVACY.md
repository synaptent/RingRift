# RingRift Data Lifecycle, Retention, and Privacy (S-05.E)

> **Doc Status (2025-11-27): Active (IMPLEMENTED)**
> Data lifecycle, retention, and privacy features are now fully implemented. This document describes the **current production behavior** for S-05.E.x features. This is not a rules or lifecycle SSoT; it complements `SECURITY_THREAT_MODEL.md`, `OPERATIONS_DB.md`, and `DOCUMENTATION_INDEX.md` for overall security/ops posture.

**Related docs:** [`docs/SECURITY_THREAT_MODEL.md`](./SECURITY_THREAT_MODEL.md), [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md), [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md), [`schema.prisma`](../prisma/schema.prisma)

This document defines how RingRift handles user and game data over time:

- What we store and how sensitive it is.
- How long different data classes are retained.
- What happens when a user deletes their account.
- What a privacy-aware data export includes.

All S-05.E.x features described in this document are **implemented and tested** as of Pass 6 (2025-11-27).

## 1. Principles and Scope

- **Pragmatic and implementable:** Policies are designed to fit the existing schema and architecture (for example [`User`](../prisma/schema.prisma), [`Game`](../prisma/schema.prisma), [`Move`](../prisma/schema.prisma), [`RefreshToken`](../prisma/schema.prisma)) without major rewrites.
- **Minimise PII exposure:** Store the minimum necessary personal data; prefer pseudonyms and IDs in logs, metrics, and long-lived stores.
- **Favour anonymisation over hard delete:** For game history and ratings, keep records but strip or pseudonymise identifiers when accounts are deleted.
- **Provider-agnostic:** Retention windows are expressed in days; exact log or metric retention is configured in whatever logging/metrics stack operators choose.
- **Future-regime ready:** The design intentionally avoids mentioning specific regulations (GDPR, CCPA) but should be adaptable to them.

Scope covers:

- Core account and auth data in [`User`](../prisma/schema.prisma) and [`RefreshToken`](../prisma/schema.prisma).
- Game and move data in [`Game`](../prisma/schema.prisma) and [`Move`](../prisma/schema.prisma).
- Logs and metrics produced by [`logger`](../src/server/utils/logger.ts) and metrics utilities such as [`rulesParityMetrics`](../src/server/utils/rulesParityMetrics.ts).
- Client error reports from [`errorReporting`](../src/client/utils/errorReporting.ts) via `/api/client-errors` in [`index` routes](../src/server/routes/index.ts).
- Backups and restores as operated via [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md).

## 2. Data Inventory and Sensitivity

This section summarises the main data classes RingRift persists today and classifies their sensitivity and purpose. It is intentionally high-level; see [`schema.prisma`](../prisma/schema.prisma) for exact fields.

### 2.1 Per-user account and authentication data

| Category                 | Examples                                                                                                            | Sensitivity | Primary purpose                                    | Notes                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------- | ----------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Account identity         | `email`, `username`, `createdAt`, `lastLoginAt` in [`User`](../prisma/schema.prisma)                                | **High**    | Login, account recovery, basic UX display          | PII; email must never appear in logs, metrics, or exports for other users.                |
| Credentials & auth state | `passwordHash`, `emailVerified`, `tokenVersion`, reset and verification tokens in [`User`](../prisma/schema.prisma) | **High**    | Authentication, abuse protection, token revocation | Secrets; only stored in DB and never logged. Reset/verification tokens are short-lived.   |
| Session artefacts        | Rows in [`RefreshToken`](../prisma/schema.prisma) and any server-side lockout counters                              | **High**    | Long-lived sessions, login abuse throttling        | Tied to device/session; contain no plaintext credentials but must be pruned after expiry. |
| Basic profile & rating   | `rating`, `gamesPlayed`, `gamesWon` and similar fields in [`User`](../prisma/schema.prisma)                         | **Medium**  | Matchmaking, fairness, leaderboards                | Not PII alone but linked to identity; exposed to other users in limited contexts.         |

### 2.2 Game and gameplay data

| Category                  | Examples                                                                                                                            | Sensitivity    | Primary purpose                                    | Notes                                                                            |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------- | -------------------------------------------------- | -------------------------------------------------------------------------------- |
| Game configuration        | Board type, max players, rated flag, time control fields in [`Game`](../prisma/schema.prisma)                                       | **Low–Medium** | Running games, replay, analysis, matchmaking       | Not PII by itself, but combined with timestamps can act as quasi-identifier.     |
| Player assignments        | `player1Id`–`player4Id`, `winnerId` in [`Game`](../prisma/schema.prisma)                                                            | **Medium**     | Ownership, permissions, ratings, leaderboards      | Links between users and games; should be treated as personal data when exported. |
| Move history              | `Move` rows (position, moveType, timestamp) in [`Move`](../prisma/schema.prisma)                                                    | **Medium**     | Rules enforcement, replay, anti-cheat, AI training | Not PII alone, but linked via `playerId`; valuable for fairness and diagnostics. |
| Derived stats and ratings | Rating changes, win/loss streaks, derived aggregates in services such as [`RatingService`](../src/server/services/RatingService.ts) | **Medium**     | Fair matchmaking, competitive integrity            | Long-lived by design; do not need to expose opponent PII in exports.             |

### 2.3 Logs, metrics, and client error reports

| Category             | Examples                                                                                                                               | Sensitivity     | Primary purpose                                   | Notes                                                                                                        |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Application logs     | Structured logs from [`logger`](../src/server/utils/logger.ts), including request IDs, game IDs, and redacted emails via `redactEmail` | **Medium–High** | Debugging, incident response, limited audit trail | Must avoid raw payloads, tokens, and unredacted PII; retention should be bounded.                            |
| Metrics              | Server-side metrics (for example from [`rulesParityMetrics`](../src/server/utils/rulesParityMetrics.ts)) and Python metrics            | **Low**         | Performance, capacity planning, health checks     | Should not contain PII; primarily counters, histograms, and gauges.                                          |
| Client error reports | Payloads sent via [`errorReporting`](../src/client/utils/errorReporting.ts) to `/api/client-errors` and stored/logged server-side      | **Medium**      | Frontend diagnostics and crash analysis           | May include stack traces and user agent data; must not include tokens or raw PII. Retention should be short. |

### 2.4 Backups and derived datasets

| Category                      | Examples                                                                       | Sensitivity     | Primary purpose                          | Notes                                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------ | --------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Database backups              | Snapshots and dumps produced per [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md) | **High**        | Disaster recovery, migration rollback    | Contain complete copies of PII and game data; retention controlled at infra level; access must be tightly limited. |
| Analytics or research exports | Future offline datasets built from game history and ratings                    | **Medium–High** | Balance and AI analysis, product metrics | Should use pseudonymous IDs and avoid including direct PII where possible.                                         |

## 3. Retention and Anonymisation Policy

The policies below describe target behaviour for a production deployment. Where current behaviour differs (for example, no deletion endpoint yet), the gap is noted and scheduled into S-05.E implementation tracks.

### 3.1 User accounts and auth artefacts

| Data class                                                                                                                     | Target retention behaviour                                                                           | Anonymisation / deletion strategy                                                                                                                                                                                                        | Rationale                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Active user accounts (`User` rows with `isActive = true`)                                                                      | Retained indefinitely while the account exists.                                                      | None; account is live. PII protected via existing auth and logging controls.                                                                                                                                                             | Needed for ongoing login, gameplay, and fair ratings.                                    |
| Soft-deleted user accounts (`isActive = false`, planned `deletedAt` field)                                                     | Keep account row indefinitely but treat it as **logically deleted**.                                 | Immediately on deletion: revoke tokens via `tokenVersion`, clear reset/verification tokens, and replace `email` / `username` with non-identifying pseudonyms (for example, generated placeholders not derived from the original values). | Preserves referential integrity for games and ratings while removing direct identifiers. |
| Password reset and verification tokens (`verificationToken`, `passwordResetToken` fields in [`User`](../prisma/schema.prisma)) | Maximum **24 hours** after creation, then cleared by background job or on use.                       | Tokens are removed (set to `null`) as soon as they are used or expire; associated email addresses remain.                                                                                                                                | Limits the window for token theft and reduces stale auth artefacts in the DB.            |
| Refresh tokens (rows in [`RefreshToken`](../prisma/schema.prisma))                                                             | Rows with `expiresAt` in the past should be removed within **7–30 days** via a periodic cleanup job. | Simple hard delete of expired rows; no anonymisation needed as they do not contain user PII beyond the foreign key.                                                                                                                      | Keeps the table small and limits the blast radius of historic session identifiers.       |
| Login lockout or abuse-tracking state                                                                                          | Retain only as long as needed to enforce lockouts (for example **24–72 hours**).                     | Implemented via short-lived DB or Redis state; no export or long-term retention.                                                                                                                                                         | Balances abuse protection with privacy by avoiding long-lived behavioural profiling.     |

_Implemented in S-05.E.4: [`DataRetentionService`](../src/server/services/DataRetentionService.ts) provides background cleanup for soft-deleted users, expired tokens, and unverified accounts. See S-05.E.1 for account deletion endpoint._

### 3.2 Game data and ratings

| Data class                                                                                             | Target retention behaviour                                                   | Anonymisation / deletion strategy                                                                                                                                                                                    | Rationale                                                                            |
| ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Game records in [`Game`](../prisma/schema.prisma)                                                      | Retained indefinitely.                                                       | Keep internal `playerId` and `winnerId` references for integrity, but when a user is deleted, client-facing APIs should substitute a generic label (for example `DeletedPlayer`) instead of the historical username. | Game outcomes are required for rating integrity, anti-cheat, and historical stats.   |
| Move history in [`Move`](../prisma/schema.prisma)                                                      | Retained indefinitely.                                                       | No changes at DB level; moves refer to the anonymised or pseudonymous user record. Exports must avoid leaking other users’ PII.                                                                                      | Needed for replay, diagnostics, rules parity, and potential AI training.             |
| Ratings and derived stats (for example via [`RatingService`](../src/server/services/RatingService.ts)) | Retained indefinitely, but only exposed in aggregate once a user is deleted. | For deleted accounts, preserve rating history internally but hide them from public leaderboards or display them under anonymised labels.                                                                             | Maintains fairness and historical consistency while respecting user deletion intent. |

When in doubt, **keep gameplay data but sever or anonymise direct identifiers** rather than hard-deleting rows that other entities depend on.

### 3.3 Logs and metrics

| Data class                | Target retention behaviour                                                            | Anonymisation / deletion strategy                                                                                                                                                          | Rationale                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Application logs          | Production logs retained for **30 days** by default; staging/dev logs may be shorter. | Logs must already avoid PII and secrets (for example via `redactEmail` in [`logger`](../src/server/utils/logger.ts)). No per-user deletion is attempted; instead, retention is time-based. | Supports debugging and incident response while bounding the volume and privacy exposure. |
| Metrics (Node and Python) | Retention per metrics backend (commonly **30–90 days**).                              | Metrics must remain PII-free and keyed by IDs or labels only. No per-user deletion.                                                                                                        | Metrics are aggregate and low-sensitivity when designed correctly.                       |

_S-05.E.4 partially implemented: [`DataRetentionService`](../src/server/services/DataRetentionService.ts) handles database-level retention. Log and metric retention should be configured at the infrastructure level (e.g., log rotation, Prometheus retention settings)._

### 3.4 Client error reports

| Data class                                        | Target retention behaviour                                      | Anonymisation / deletion strategy                                                                                                       | Rationale                                                  |
| ------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Client error reports sent to `/api/client-errors` | Retain for **7–30 days** in production; shorter in staging/dev. | Payloads should be scrubbed of tokens and obvious PII before storage. No per-user deletion; rely on time-based expiry and storage caps. | Short-lived diagnostics only; not a permanent audit trail. |

_S-05.E.4 note: Client errors are written to application logs. Retention is controlled by log rotation settings in the logging infrastructure._

### 3.5 Backups and offline copies

- Backups created per [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md) inevitably contain full historical data, including PII and game history.
- Retention of backups is governed by ops policy (for example **30–180 days**, plus periodic long-term snapshots).
- Data-deletion workflows **do not retroactively edit historical backups**; instead, operators must ensure:
  - Access to backup storage is tightly controlled.
  - Restores from backups are followed by a re-application of deletion/anonymisation routines where appropriate.
- Any derived analytics datasets created from production data should:
  - Use pseudonymous user IDs.
  - Exclude direct identifiers such as email addresses.
  - Document retention and access controls alongside their creation.

## 4. Account Deletion Workflow ✅ IMPLEMENTED

This section describes the account deletion feature implemented in S-05.E.1 and S-05.E.2.

**Implementation:** [`DELETE /api/users/me`](../src/server/routes/user.ts) endpoint with full soft-delete support.

### 4.1 API surface ✅

- REST endpoint: [`DELETE /api/users/me`](../src/server/routes/user.ts)
- Authentication: required; uses existing JWT auth and the `tokenVersion` mechanism in [`User`](../prisma/schema.prisma).
- Responses:
  - `200 OK` with `{ success: true, message: "Account deleted successfully" }`
  - `401` for unauthenticated requests
  - `404` if user not found

### 4.2 Backend behaviour ✅

When a deletion request is accepted, the backend performs the following steps **atomically** within a database transaction:

1. **Mark account as deleted / inactive** ✅
   - Sets `isActive = false` on the [`User`](../prisma/schema.prisma) row.
   - Sets `deletedAt` timestamp to the current time.
   - Implementation: [`user.ts:980-995`](../src/server/routes/user.ts)

2. **Revoke auth tokens and sessions** ✅
   - Increments `tokenVersion` so all existing JWTs become invalid.
   - Deletes all [`RefreshToken`](../prisma/schema.prisma) rows for that user.
   - Clears `verificationToken`, `verificationTokenExpires`, `passwordResetToken`, `passwordResetExpires`.
   - Implementation: [`user.ts:986-1001`](../src/server/routes/user.ts)

3. **Anonymise PII in the account record** ✅
   - Replaces `email` with `deleted+<user-id>@example.invalid` via [`anonymizedEmail()`](../src/server/routes/user.ts)
   - Replaces `username` with `DeletedPlayer_<first-8-chars-of-id>` via [`anonymizedUsername()`](../src/server/routes/user.ts)
   - Preserves non-PII fields: `rating`, `gamesPlayed`, `gamesWon` for rating integrity.
   - Implementation: [`user.ts:992-994`](../src/server/routes/user.ts)

4. **Handle live sessions** ✅
   - Terminates active WebSocket connections via [`terminateUserSessions()`](../src/server/websocket/server.ts)
   - Sends `ACCESS_DENIED` error to client before disconnecting.
   - Implementation: [`user.ts:1009-1016`](../src/server/routes/user.ts)

5. **Update user-facing projections** ✅
   - Helper functions detect anonymized users: [`isDeletedUserUsername()`](../src/server/routes/user.ts)
   - Display formatting: [`getDisplayUsername()`](../src/server/routes/user.ts) returns "Deleted Player" for anonymized users
   - Constants: [`DELETED_USER_PREFIX`](../src/server/routes/user.ts), [`DELETED_USER_DISPLAY_NAME`](../src/server/routes/user.ts)

**Checklist – S-05.E account deletion behaviour** ✅ ALL IMPLEMENTED

- [x] Endpoint accepts only authenticated, non-deleted users.
- [x] `isActive` set to `false` and `deletedAt` populated.
- [x] `tokenVersion` incremented; all refresh tokens deleted.
- [x] Reset/verification tokens cleared.
- [x] Email and username replaced with non-identifying placeholders.
- [x] Active WebSocket sessions closed via [`terminateUserSessions()`](../src/server/websocket/server.ts).
- [x] Game and rating history preserved but presented under anonymised labels ("Deleted Player").

**Test Coverage:** [`tests/integration/accountDeletion.test.ts`](../tests/integration/accountDeletion.test.ts) - 10 test cases covering all checklist items.

### 4.3 Effects on related features

- **Login and auth:**
  - Subsequent login attempts for deleted accounts should fail deterministically (for example a 403 with a stable error code) without leaking whether the email previously existed.
- **Client auth context:**
  - The frontend `AuthContext` should treat specific auth errors (for example `ACCOUNT_DEACTIVATED`) as a signal to clear local state and show an appropriate message.
- **Rating and leaderboards:**
  - Leaderboards should either exclude deleted users or show them as anonymised entries.
  - Internal rating math continues to treat the user record as a valid historical participant.

## 5. Basic Data Export Workflow ✅ IMPLEMENTED

This section describes the data export feature that allows a user to retrieve their own data without exposing other users' PII.

**Implementation:** [`GET /api/users/me/export`](../src/server/routes/user.ts) endpoint.

### 5.1 API surface ✅

- REST endpoint: [`GET /api/users/me/export`](../src/server/routes/user.ts)
- Authentication: required; same auth path as other `/api/users/me` endpoints.
- Response format: JSON document with Content-Disposition header for file download.
- Response structure:
  - `exportedAt`: ISO timestamp of export
  - `exportFormat`: Version string (currently "1.0")
  - `profile`: User profile data (id, username, email, createdAt, emailVerified, role, isActive, lastLoginAt, updatedAt)
  - `statistics`: Rating and win/loss statistics (rating, gamesPlayed, wins, losses, winRate)
  - `games`: Array of games the user participated in, each with:
    - `id`, `createdAt`, `startedAt`, `completedAt`, `status`
    - `result`: win, loss, draw, in_progress, or abandoned
    - `boardType`, `isRated`
    - `opponent`: Anonymized as "Deleted Player" for deleted users
    - `moves`: Array of move data with `isUserMove` flag
  - `preferences`: User preferences (placeholder for future expansion)

### 5.2 Scope and privacy constraints ✅

- The export **does not** include other users' email addresses, password hashes, or internal IDs.
- Opponent usernames are included; deleted users appear as "Deleted Player" via [`getDisplayUsername()`](../src/server/routes/user.ts).
- Move histories include all moves from games the user participated in, with `isUserMove` flag.
- Sensitive fields explicitly excluded from export:
  - `passwordHash`, `tokenVersion`, `deletedAt`
  - `verificationToken`, `verificationTokenExpires`
  - `passwordResetToken`, `passwordResetExpires`

### 5.3 Implementation details ✅

- Implementation: [`user.ts:1169-1346`](../src/server/routes/user.ts)
- Uses Prisma queries to fetch user data and all games where user participated (as player1-4).
- Transforms game data to include user-perspective result (win/loss/draw).
- Sets `Content-Disposition: attachment` header for browser download.

**Checklist – S-05.E data export behaviour** ✅ ALL IMPLEMENTED

- [x] Export endpoint requires authentication.
- [x] Response includes account profile and a summary of games the user participated in.
- [x] No other users' email addresses or password hashes are present.
- [x] Deleted users appear as "Deleted Player" via [`getDisplayUsername()`](../src/server/routes/user.ts).
- [x] Response formatted as downloadable JSON file.

**Test Coverage:** [`tests/integration/dataLifecycle.test.ts`](../tests/integration/dataLifecycle.test.ts) - Tests for export profile data, game history, and sensitive data exclusion.

## 6. Implementation Tracks (S-05.E.x) - ALL COMPLETE

The following tracks were completed in Pass 6 (2025-11-27).

### S-05.E.1 – Soft-delete model and account deletion endpoint ✅ IMPLEMENTED

- **Status:** ✅ Complete (2025-11-27)
- **Goal:** Introduce a soft-delete model for users and implement an authenticated account deletion endpoint that revokes tokens and anonymises PII.
- **Implementation:**
  - [`DELETE /api/users/me`](../src/server/routes/user.ts) endpoint
  - `deletedAt` field in User model (Prisma schema)
  - Token revocation via `tokenVersion` increment
  - PII anonymization via [`anonymizedEmail()`](../src/server/routes/user.ts) and [`anonymizedUsername()`](../src/server/routes/user.ts)
  - WebSocket session termination via [`terminateUserSessions()`](../src/server/websocket/server.ts)
- **Tests:** [`tests/integration/accountDeletion.test.ts`](../tests/integration/accountDeletion.test.ts) - 10 tests

### S-05.E.2 – Historical game anonymisation and presentation ✅ IMPLEMENTED

- **Status:** ✅ Complete (2025-11-27)
- **Goal:** Ensure deleted users are anonymised in all game and rating-related outputs while preserving internal integrity.
- **Implementation:**
  - Helper functions: [`isDeletedUserUsername()`](../src/server/routes/user.ts), [`getDisplayUsername()`](../src/server/routes/user.ts)
  - Constants: [`DELETED_USER_PREFIX`](../src/server/routes/user.ts) = "DeletedPlayer\_", [`DELETED_USER_DISPLAY_NAME`](../src/server/routes/user.ts) = "Deleted Player"
  - Data export shows opponents as "Deleted Player" for anonymized users
- **Tests:** [`tests/integration/dataLifecycle.test.ts`](../tests/integration/dataLifecycle.test.ts) - Tests for game history with deleted players

### S-05.E.3 – Data export endpoint ✅ IMPLEMENTED

- **Status:** ✅ Complete (2025-11-27)
- **Goal:** Implement `GET /api/users/me/export` to return a privacy-respecting snapshot of a user's own data.
- **Implementation:**
  - [`GET /api/users/me/export`](../src/server/routes/user.ts) endpoint
  - Returns profile, statistics, games with moves, preferences
  - Excludes sensitive fields (passwordHash, tokens, etc.)
  - Anonymizes deleted opponents in game history
- **Tests:** [`tests/integration/dataLifecycle.test.ts`](../tests/integration/dataLifecycle.test.ts) - 4 tests for export functionality

### S-05.E.4 – Data retention service ✅ IMPLEMENTED

- **Status:** ✅ Complete (2025-11-27)
- **Goal:** Implement background cleanup service for enforcing data retention policies.
- **Implementation:** [`DataRetentionService`](../src/server/services/DataRetentionService.ts) (420 lines)
- **Features:**
  - [`runRetentionTasks()`](../src/server/services/DataRetentionService.ts): Runs all cleanup operations
  - [`hardDeleteExpiredUsers()`](../src/server/services/DataRetentionService.ts): Permanently removes soft-deleted users past retention period
  - [`cleanupExpiredTokens()`](../src/server/services/DataRetentionService.ts): Removes expired/revoked refresh tokens
  - [`cleanupUnverifiedAccounts()`](../src/server/services/DataRetentionService.ts): Soft-deletes unverified accounts past threshold
- **Configuration:** Environment variables (see [`docs/ENVIRONMENT_VARIABLES.md`](./ENVIRONMENT_VARIABLES.md)):
  - `DATA_RETENTION_DELETED_USERS_DAYS` (default: 30)
  - `DATA_RETENTION_INACTIVE_USERS_DAYS` (default: 365)
  - `DATA_RETENTION_UNVERIFIED_DAYS` (default: 7)
  - `DATA_RETENTION_GAME_DATA_MONTHS` (default: 24)
  - `DATA_RETENTION_SESSION_HOURS` (default: 24)
  - `DATA_RETENTION_EXPIRED_TOKEN_DAYS` (default: 7)
- **Factory Function:** [`createDataRetentionService()`](../src/server/services/DataRetentionService.ts) for environment-based configuration
- **Tests:** [`tests/integration/dataLifecycle.test.ts`](../tests/integration/dataLifecycle.test.ts) - 7 tests for retention service
- **TODO:** Schedule via cron job (e.g., `node-cron` at 3 AM daily)

### S-05.E.5 – Data lifecycle validation and tests ✅ IMPLEMENTED

- **Status:** ✅ Complete (2025-11-27)
- **Goal:** Add comprehensive tests to validate that data lifecycle policies work correctly.
- **Implementation:**
  - [`tests/integration/accountDeletion.test.ts`](../tests/integration/accountDeletion.test.ts) - 10 tests for account deletion
  - [`tests/integration/dataLifecycle.test.ts`](../tests/integration/dataLifecycle.test.ts) - 15 tests for data export and retention
- **Test Coverage:**
  - Authentication requirements for deletion
  - Token invalidation after deletion
  - PII anonymization verification
  - Login prevention after deletion
  - Email reuse after anonymization
  - Refresh token cleanup
  - Data export structure and privacy
  - Retention service cleanup operations
  - Game history with deleted players

**Total Test Count:** 25 tests across 2 test files.
