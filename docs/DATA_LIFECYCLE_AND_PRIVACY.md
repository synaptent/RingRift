# RingRift Data Lifecycle, Retention, and Privacy (S-05.E)

**Status:** Design-only (to be implemented in S-05.E.x tracks)  
**Related docs:** [`docs/SECURITY_THREAT_MODEL.md`](./SECURITY_THREAT_MODEL.md:1), [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:155), [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md:1), [`schema.prisma`](../prisma/schema.prisma:1)

This document defines how RingRift should handle user and game data over time:

- What we store and how sensitive it is.
- How long different data classes should be retained.
- What should happen when a user deletes their account.
- What a minimal, privacy-aware data export should include.

It is a **design and plan**, not a description of current behaviour. Where relevant, planned changes are grouped under S-05.E.x implementation tracks.

## 1. Principles and Scope

- **Pragmatic and implementable:** Policies are designed to fit the existing schema and architecture (for example [`User`](../prisma/schema.prisma:13), [`Game`](../prisma/schema.prisma:47), [`Move`](../prisma/schema.prisma:84), [`RefreshToken`](../prisma/schema.prisma:131)) without major rewrites.
- **Minimise PII exposure:** Store the minimum necessary personal data; prefer pseudonyms and IDs in logs, metrics, and long-lived stores.
- **Favour anonymisation over hard delete:** For game history and ratings, keep records but strip or pseudonymise identifiers when accounts are deleted.
- **Provider-agnostic:** Retention windows are expressed in days; exact log or metric retention is configured in whatever logging/metrics stack operators choose.
- **Future-regime ready:** The design intentionally avoids mentioning specific regulations (GDPR, CCPA) but should be adaptable to them.

Scope covers:

- Core account and auth data in [`User`](../prisma/schema.prisma:13) and [`RefreshToken`](../prisma/schema.prisma:131).
- Game and move data in [`Game`](../prisma/schema.prisma:47) and [`Move`](../prisma/schema.prisma:84).
- Logs and metrics produced by [`logger`](../src/server/utils/logger.ts:1) and metrics utilities such as [`rulesParityMetrics`](../src/server/utils/rulesParityMetrics.ts:1).
- Client error reports from [`errorReporting`](../src/client/utils/errorReporting.ts:1) via `/api/client-errors` in [`index` routes](../src/server/routes/index.ts:1).
- Backups and restores as operated via [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md:1).

## 2. Data Inventory and Sensitivity

This section summarises the main data classes RingRift persists today and classifies their sensitivity and purpose. It is intentionally high-level; see [`schema.prisma`](../prisma/schema.prisma:1) for exact fields.

### 2.1 Per-user account and authentication data

| Category                 | Examples                                                                                                               | Sensitivity | Primary purpose                                    | Notes                                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------- | ----------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Account identity         | `email`, `username`, `createdAt`, `lastLoginAt` in [`User`](../prisma/schema.prisma:13)                                | **High**    | Login, account recovery, basic UX display          | PII; email must never appear in logs, metrics, or exports for other users.                |
| Credentials & auth state | `passwordHash`, `emailVerified`, `tokenVersion`, reset and verification tokens in [`User`](../prisma/schema.prisma:13) | **High**    | Authentication, abuse protection, token revocation | Secrets; only stored in DB and never logged. Reset/verification tokens are short-lived.   |
| Session artefacts        | Rows in [`RefreshToken`](../prisma/schema.prisma:131) and any server-side lockout counters                             | **High**    | Long-lived sessions, login abuse throttling        | Tied to device/session; contain no plaintext credentials but must be pruned after expiry. |
| Basic profile & rating   | `rating`, `gamesPlayed`, `gamesWon` and similar fields in [`User`](../prisma/schema.prisma:13)                         | **Medium**  | Matchmaking, fairness, leaderboards                | Not PII alone but linked to identity; exposed to other users in limited contexts.         |

### 2.2 Game and gameplay data

| Category                  | Examples                                                                                                                              | Sensitivity    | Primary purpose                                    | Notes                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------- | -------------------------------------------------- | -------------------------------------------------------------------------------- |
| Game configuration        | Board type, max players, rated flag, time control fields in [`Game`](../prisma/schema.prisma:47)                                      | **Low–Medium** | Running games, replay, analysis, matchmaking       | Not PII by itself, but combined with timestamps can act as quasi-identifier.     |
| Player assignments        | `player1Id`–`player4Id`, `winnerId` in [`Game`](../prisma/schema.prisma:47)                                                           | **Medium**     | Ownership, permissions, ratings, leaderboards      | Links between users and games; should be treated as personal data when exported. |
| Move history              | `Move` rows (position, moveType, timestamp) in [`Move`](../prisma/schema.prisma:84)                                                   | **Medium**     | Rules enforcement, replay, anti-cheat, AI training | Not PII alone, but linked via `playerId`; valuable for fairness and diagnostics. |
| Derived stats and ratings | Rating changes, win/loss streaks, derived aggregates in services such as [`RatingService`](../src/server/services/RatingService.ts:1) | **Medium**     | Fair matchmaking, competitive integrity            | Long-lived by design; do not need to expose opponent PII in exports.             |

### 2.3 Logs, metrics, and client error reports

| Category             | Examples                                                                                                                                 | Sensitivity     | Primary purpose                                   | Notes                                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Application logs     | Structured logs from [`logger`](../src/server/utils/logger.ts:1), including request IDs, game IDs, and redacted emails via `redactEmail` | **Medium–High** | Debugging, incident response, limited audit trail | Must avoid raw payloads, tokens, and unredacted PII; retention should be bounded.                            |
| Metrics              | Server-side metrics (for example from [`rulesParityMetrics`](../src/server/utils/rulesParityMetrics.ts:1)) and Python metrics            | **Low**         | Performance, capacity planning, health checks     | Should not contain PII; primarily counters, histograms, and gauges.                                          |
| Client error reports | Payloads sent via [`errorReporting`](../src/client/utils/errorReporting.ts:1) to `/api/client-errors` and stored/logged server-side      | **Medium**      | Frontend diagnostics and crash analysis           | May include stack traces and user agent data; must not include tokens or raw PII. Retention should be short. |

### 2.4 Backups and derived datasets

| Category                      | Examples                                                                         | Sensitivity     | Primary purpose                          | Notes                                                                                                              |
| ----------------------------- | -------------------------------------------------------------------------------- | --------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Database backups              | Snapshots and dumps produced per [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md:1) | **High**        | Disaster recovery, migration rollback    | Contain complete copies of PII and game data; retention controlled at infra level; access must be tightly limited. |
| Analytics or research exports | Future offline datasets built from game history and ratings                      | **Medium–High** | Balance and AI analysis, product metrics | Should use pseudonymous IDs and avoid including direct PII where possible.                                         |

## 3. Retention and Anonymisation Policy

The policies below describe target behaviour for a production deployment. Where current behaviour differs (for example, no deletion endpoint yet), the gap is noted and scheduled into S-05.E implementation tracks.

### 3.1 User accounts and auth artefacts

| Data class                                                                                                                        | Target retention behaviour                                                                           | Anonymisation / deletion strategy                                                                                                                                                                                                        | Rationale                                                                                |
| --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Active user accounts (`User` rows with `isActive = true`)                                                                         | Retained indefinitely while the account exists.                                                      | None; account is live. PII protected via existing auth and logging controls.                                                                                                                                                             | Needed for ongoing login, gameplay, and fair ratings.                                    |
| Soft-deleted user accounts (`isActive = false`, planned `deletedAt` field)                                                        | Keep account row indefinitely but treat it as **logically deleted**.                                 | Immediately on deletion: revoke tokens via `tokenVersion`, clear reset/verification tokens, and replace `email` / `username` with non-identifying pseudonyms (for example, generated placeholders not derived from the original values). | Preserves referential integrity for games and ratings while removing direct identifiers. |
| Password reset and verification tokens (`verificationToken`, `passwordResetToken` fields in [`User`](../prisma/schema.prisma:13)) | Maximum **24 hours** after creation, then cleared by background job or on use.                       | Tokens are removed (set to `null`) as soon as they are used or expire; associated email addresses remain.                                                                                                                                | Limits the window for token theft and reduces stale auth artefacts in the DB.            |
| Refresh tokens (rows in [`RefreshToken`](../prisma/schema.prisma:131))                                                            | Rows with `expiresAt` in the past should be removed within **7–30 days** via a periodic cleanup job. | Simple hard delete of expired rows; no anonymisation needed as they do not contain user PII beyond the foreign key.                                                                                                                      | Keeps the table small and limits the blast radius of historic session identifiers.       |
| Login lockout or abuse-tracking state                                                                                             | Retain only as long as needed to enforce lockouts (for example **24–72 hours**).                     | Implemented via short-lived DB or Redis state; no export or long-term retention.                                                                                                                                                         | Balances abuse protection with privacy by avoiding long-lived behavioural profiling.     |

_To be implemented in S-05.E.1 and S-05.E.4: background jobs and DB helpers that enforce the above TTLs._

### 3.2 Game data and ratings

| Data class                                                                                               | Target retention behaviour                                                   | Anonymisation / deletion strategy                                                                                                                                                                                    | Rationale                                                                            |
| -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Game records in [`Game`](../prisma/schema.prisma:47)                                                     | Retained indefinitely.                                                       | Keep internal `playerId` and `winnerId` references for integrity, but when a user is deleted, client-facing APIs should substitute a generic label (for example `DeletedPlayer`) instead of the historical username. | Game outcomes are required for rating integrity, anti-cheat, and historical stats.   |
| Move history in [`Move`](../prisma/schema.prisma:84)                                                     | Retained indefinitely.                                                       | No changes at DB level; moves refer to the anonymised or pseudonymous user record. Exports must avoid leaking other users’ PII.                                                                                      | Needed for replay, diagnostics, rules parity, and potential AI training.             |
| Ratings and derived stats (for example via [`RatingService`](../src/server/services/RatingService.ts:1)) | Retained indefinitely, but only exposed in aggregate once a user is deleted. | For deleted accounts, preserve rating history internally but hide them from public leaderboards or display them under anonymised labels.                                                                             | Maintains fairness and historical consistency while respecting user deletion intent. |

When in doubt, **keep gameplay data but sever or anonymise direct identifiers** rather than hard-deleting rows that other entities depend on.

### 3.3 Logs and metrics

| Data class                | Target retention behaviour                                                            | Anonymisation / deletion strategy                                                                                                                                                            | Rationale                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Application logs          | Production logs retained for **30 days** by default; staging/dev logs may be shorter. | Logs must already avoid PII and secrets (for example via `redactEmail` in [`logger`](../src/server/utils/logger.ts:1)). No per-user deletion is attempted; instead, retention is time-based. | Supports debugging and incident response while bounding the volume and privacy exposure. |
| Metrics (Node and Python) | Retention per metrics backend (commonly **30–90 days**).                              | Metrics must remain PII-free and keyed by IDs or labels only. No per-user deletion.                                                                                                          | Metrics are aggregate and low-sensitivity when designed correctly.                       |

_To be implemented in S-05.E.4: document and, where possible, codify log and metric retention in deployment configuration._

### 3.4 Client error reports

| Data class                                        | Target retention behaviour                                      | Anonymisation / deletion strategy                                                                                                       | Rationale                                                  |
| ------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Client error reports sent to `/api/client-errors` | Retain for **7–30 days** in production; shorter in staging/dev. | Payloads should be scrubbed of tokens and obvious PII before storage. No per-user deletion; rely on time-based expiry and storage caps. | Short-lived diagnostics only; not a permanent audit trail. |

_To be implemented in S-05.E.4: ensure the storage mechanism for client errors enforces TTLs (for example via log rotation or table-level pruning)._

### 3.5 Backups and offline copies

- Backups created per [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md:1) inevitably contain full historical data, including PII and game history.
- Retention of backups is governed by ops policy (for example **30–180 days**, plus periodic long-term snapshots).
- Data-deletion workflows **do not retroactively edit historical backups**; instead, operators must ensure:
  - Access to backup storage is tightly controlled.
  - Restores from backups are followed by a re-application of deletion/anonymisation routines where appropriate.
- Any derived analytics datasets created from production data should:
  - Use pseudonymous user IDs.
  - Exclude direct identifiers such as email addresses.
  - Document retention and access controls alongside their creation.

## 4. Account Deletion Workflow (Design Only)

This section specifies the desired behaviour of a future account deletion feature. It does **not** describe current behaviour; implementation will be tracked under S-05.E.1 and S-05.E.2.

### 4.1 API surface

- REST endpoint: `DELETE /api/users/me` (or, alternatively, `POST /api/users/me/delete`).
- Authentication: required; uses existing JWT auth and the `tokenVersion` mechanism in [`User`](../prisma/schema.prisma:13).
- Optional re-authentication step (password confirmation or recent-login requirement) can be added later for defence in depth.
- Expected responses:
  - `200 OK` with a small payload indicating that the account has been scheduled for deletion and that all sessions are being revoked.
  - `401/403` for unauthenticated or already-deleted accounts.

### 4.2 Backend behaviour

When a deletion request is accepted, the backend should perform the following steps **atomically** for that user:

1. **Mark account as deleted / inactive**
   - Set `isActive = false` on the [`User`](../prisma/schema.prisma:13) row.
   - Add a `deletedAt` timestamp field in a future migration (S-05.E.1) and set it to the current time.
2. **Revoke auth tokens and sessions**
   - Increment `tokenVersion` so all existing JWTs become invalid (aligned with logout-all semantics).
   - Delete all [`RefreshToken`](../prisma/schema.prisma:131) rows for that user.
   - Clear any password reset or verification tokens (`verificationToken`, `passwordResetToken`, and their expiry fields).
3. **Anonymise PII in the account record**
   - Replace `email` with a non-identifying placeholder (for example a randomised `deleted+<opaque-id>@example.invalid` string that does not encode the original email).
   - Replace `username` with a generated pseudonym (for example `DeletedPlayer_<short-id>`).
   - Preserve non-PII fields needed for ratings and history (for example `rating`, `gamesPlayed`, `gamesWon`).
4. **Handle live sessions and games**
   - For active WebSocket connections, treat the account as logged out and close sockets gracefully.
   - For in-progress games, treat the user as having resigned or abandoned the game (consistent with existing `abandoned` / `finished` semantics in [`GameStatus`](../prisma/schema.prisma:111)).
5. **Update user-facing projections**
   - Ensure future API responses and WebSocket payloads that include this user use the anonymised label instead of the historical username.
   - Remove the user from friend lists, invites, or social features once those exist (out of scope for this design).

**Checklist – S-05.E account deletion behaviour**

- [ ] Endpoint accepts only authenticated, non-deleted users.
- [ ] `isActive` set to `false` and `deletedAt` populated.
- [ ] `tokenVersion` incremented; all refresh tokens deleted.
- [ ] Reset/verification tokens cleared.
- [ ] Email and username replaced with non-identifying placeholders.
- [ ] Active WebSocket sessions closed; future auth attempts fail with a specific error (for example `ACCOUNT_DEACTIVATED`).
- [ ] Game and rating history preserved but presented under anonymised labels.

### 4.3 Effects on related features

- **Login and auth:**
  - Subsequent login attempts for deleted accounts should fail deterministically (for example a 403 with a stable error code) without leaking whether the email previously existed.
- **Client auth context:**
  - The frontend `AuthContext` should treat specific auth errors (for example `ACCOUNT_DEACTIVATED`) as a signal to clear local state and show an appropriate message.
- **Rating and leaderboards:**
  - Leaderboards should either exclude deleted users or show them as anonymised entries.
  - Internal rating math continues to treat the user record as a valid historical participant.

## 5. Basic Data Export Workflow (Design Only)

This section defines a minimal, machine-readable export that allows a user to retrieve their own data without exposing other users’ PII.

### 5.1 API surface

- REST endpoint: `GET /api/users/me/export`.
- Authentication: required; same auth path as other `/api/users/me` endpoints.
- Response format: JSON document with at least the following top-level sections:
  - `account`: profile fields such as username, email, `createdAt`, current rating, `gamesPlayed`, `gamesWon`.
  - `games`: a list of games where the user participated, including:
    - `gameId`, `createdAt`, `endedAt`, board type, rated flag.
    - Seat (for example `playerIndex` 1–4).
    - Result from the user’s perspective (win, loss, draw, abandoned).
  - Optional: a list of recent moves in those games where the user was the acting player (for example move numbers and basic move descriptors).

### 5.2 Scope and privacy constraints

- The export **must not** include other users’ email addresses, password hashes, or internal IDs.
- Opponent information should be limited to what is already visible in normal gameplay UI (for example historical usernames at the time of export, or anonymised labels for deleted users).
- If full move histories are included, they should be restricted to games the requesting user participated in.
- Large historical exports may be scoped to a recent window (for example the last N games or last M days) if needed for performance; this window should be clearly documented in the response metadata.

### 5.3 Implementation hints

- Reuse existing user and game queries exposed from the `user` and `game` HTTP routes (for example [`user` routes](../src/server/routes/user.ts:1) and [`game` routes](../src/server/routes/game.ts:1)).
- Prefer a single, well-defined DTO for export responses to avoid tight coupling to internal DB models.
- Add tests that assert:
  - No other users’ PII appears in the export payload.
  - Deleted users appear in a form consistent with the anonymisation rules in Section 3.

**Checklist – S-05.E data export behaviour**

- [ ] Export endpoint requires authentication.
- [ ] Response includes account profile and a summary of games the user participated in.
- [ ] No other users’ email addresses or password hashes are present.
- [ ] Deleted users appear as anonymised labels where relevant.
- [ ] Response size and time remain within acceptable bounds (for example via pagination or time-windowing).

## 6. Implementation Tracks (S-05.E.x)

The following tracks break S-05.E into concrete, incremental tasks suitable for future code-mode work.

### S-05.E.1 – Soft-delete model and account deletion endpoint

- **Goal:** Introduce a soft-delete model for users and implement an authenticated account deletion endpoint that revokes tokens and anonymises PII.
- **Scope:** Backend (user routes, auth middleware), DB schema migration for `deletedAt` or equivalent flag, tests, and minimal client handling of `ACCOUNT_DEACTIVATED`-style errors.
- **Risk level:** Medium – touches auth flows and user data but is conceptually straightforward.
- **Dependencies:** Relies on existing `tokenVersion` and logout-all semantics, and on the rules in Section 3.1.

### S-05.E.2 – Historical game anonymisation and presentation

- **Goal:** Ensure deleted users are anonymised in all game and rating-related outputs while preserving internal integrity.
- **Scope:** Backend responses and serializers for game history, lobby listings, leaderboards, and any future replay APIs; tests to confirm anonymisation behaviour; minor client updates to display anonymised labels consistently.
- **Risk level:** Medium – affects user-visible data but does not change core game logic.
- **Dependencies:** Builds on the deletion model from S-05.E.1 and the retention rules for game data in Section 3.2.

### S-05.E.3 – Data export endpoint

- **Goal:** Implement `GET /api/users/me/export` to return a privacy-respecting snapshot of a user’s own data.
- **Scope:** Backend user/game routes, DTOs, and tests; optional small client entry point (for example a settings-page button).
- **Risk level:** Low–Medium – mainly data plumbing and filtering; careful test coverage is important.
- **Dependencies:** Should respect anonymisation rules from S-05.E.1 and S-05.E.2 and rely on established DB queries.

### S-05.E.4 – Log, metrics, and client-error retention configuration

- **Goal:** Align infra and application configuration with the retention expectations in Section 3 for logs, metrics, and client error reports.
- **Scope:** Deployment configuration and operator docs; optional lightweight code changes (for example table cleanup jobs, log rotation settings).
- **Risk level:** Low – primarily configuration and documentation.
- **Dependencies:** Builds on existing logging and metrics pipelines documented in [`docs/SECURITY_THREAT_MODEL.md`](./SECURITY_THREAT_MODEL.md:1) and [`STRATEGIC_ROADMAP.md`](../STRATEGIC_ROADMAP.md:155).

### S-05.E.5 – Data lifecycle validation and ops runbooks

- **Goal:** Add operator-facing checks and runbooks to validate that data lifecycle policies are being followed in production.
- **Scope:** Documentation updates (for example extensions to [`docs/OPERATIONS_DB.md`](./OPERATIONS_DB.md:1)), admin scripts or dashboards if needed, and periodic reviews of backup and retention settings.
- **Risk level:** Low – process-oriented, with minimal code changes.
- **Dependencies:** Depends on earlier S-05.E tracks being at least partially implemented so there is a concrete policy to validate.
