# Database Schema Documentation

RingRift uses PostgreSQL with Prisma ORM for data persistence. This document describes the database schema, relationships, and query patterns.

## Overview

| Table              | Purpose                            | Key Fields                   |
| ------------------ | ---------------------------------- | ---------------------------- |
| `users`            | Player accounts and authentication | email, username, rating      |
| `games`            | Game sessions and state            | boardType, status, gameState |
| `moves`            | Move history for replay            | gameId, moveNumber, moveData |
| `refresh_tokens`   | JWT refresh token management       | token, familyId, revokedAt   |
| `chat_messages`    | In-game chat                       | gameId, userId, message      |
| `rematch_requests` | Post-game rematch handling         | gameId, status, expiresAt    |
| `rating_history`   | Elo rating changes over time       | userId, oldRating, newRating |

## Entity Relationship Diagram

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│    users     │───┬───│    games     │───────│    moves     │
│              │   │   │              │       │              │
│ id (PK)      │   │   │ id (PK)      │       │ id (PK)      │
│ email        │   │   │ player1Id    │◄──┐   │ gameId (FK)  │
│ username     │   │   │ player2Id    │◄──┤   │ playerId(FK) │
│ passwordHash │   │   │ player3Id    │◄──┤   │ moveNumber   │
│ rating       │   │   │ player4Id    │◄──┘   │ moveData     │
│ tokenVersion │   │   │ winnerId     │       └──────────────┘
└──────┬───────┘   │   │ boardType    │
       │           │   │ status       │       ┌──────────────┐
       │           │   │ gameState    │───────│chat_messages │
       │           │   └──────────────┘       │              │
       │           │                          │ id (PK)      │
       │           └──────────────────────────│ gameId (FK)  │
       │                                      │ userId (FK)  │
       ▼                                      └──────────────┘
┌──────────────┐       ┌──────────────┐
│refresh_tokens│       │rating_history│
│              │       │              │
│ id (PK)      │       │ id (PK)      │
│ userId (FK)  │       │ userId (FK)  │
│ token        │       │ gameId       │
│ familyId     │       │ oldRating    │
│ revokedAt    │       │ newRating    │
└──────────────┘       └──────────────┘
```

## Tables

### users

Player accounts with authentication and game statistics.

| Column          | Type          | Description                |
| --------------- | ------------- | -------------------------- |
| `id`            | String (CUID) | Primary key                |
| `email`         | String        | Unique email address       |
| `username`      | String        | Unique display name        |
| `passwordHash`  | String        | bcrypt hashed password     |
| `role`          | Enum          | USER, ADMIN, MODERATOR     |
| `rating`        | Int           | Elo rating (default: 1200) |
| `gamesPlayed`   | Int           | Total games played         |
| `gamesWon`      | Int           | Total games won            |
| `emailVerified` | Boolean       | Email verification status  |
| `isActive`      | Boolean       | Account active status      |
| `tokenVersion`  | Int           | JWT invalidation counter   |
| `deletedAt`     | DateTime?     | Soft delete timestamp      |

**Auth tokens (nullable):**

- `verificationToken`, `verificationTokenExpires`: Email verification
- `passwordResetToken`, `passwordResetExpires`: Password reset

**Indexes:**

- `@@index([deletedAt])` - Efficient soft-delete filtering

### games

Game sessions with state and player assignments.

| Column                    | Type          | Description                               |
| ------------------------- | ------------- | ----------------------------------------- |
| `id`                      | String (CUID) | Primary key                               |
| `boardType`               | Enum          | square8, square19, hex8, hexagonal        |
| `maxPlayers`              | Int           | 2, 3, or 4                                |
| `timeControl`             | JSON          | `{ initialTime, increment }`              |
| `isRated`                 | Boolean       | Affects Elo calculation                   |
| `allowSpectators`         | Boolean       | Spectator access                          |
| `status`                  | Enum          | waiting, active, completed, etc.          |
| `gameState`               | JSON          | Current serialized game state             |
| `rngSeed`                 | Int?          | Deterministic replay seed                 |
| `player1Id` - `player4Id` | String?       | Player slot assignments                   |
| `winnerId`                | String?       | Winner reference                          |
| `finalState`              | JSON?         | Complete state at game end                |
| `finalScore`              | JSON?         | Score breakdown per player                |
| `outcome`                 | String?       | ring_elimination, territory_control, etc. |
| `recordMetadata`          | JSON?         | Training pipeline metadata                |

**Timestamps:**

- `createdAt`, `updatedAt`, `startedAt`, `endedAt`

**Indexes:**

- `@@index([status])` - Filter by game status
- `@@index([createdAt])` - Sort by creation time
- `@@index([status, createdAt])` - Combined filtering

### moves

Move history for game replay and analysis.

| Column       | Type          | Description                              |
| ------------ | ------------- | ---------------------------------------- |
| `id`         | String (CUID) | Primary key                              |
| `gameId`     | String        | FK to games                              |
| `playerId`   | String        | FK to users                              |
| `moveNumber` | Int           | Sequential move number                   |
| `position`   | JSON          | Legacy `{ from?, to }` position data     |
| `moveType`   | Enum          | place_ring, move_stack, etc.             |
| `moveData`   | JSON?         | Rich data: captured stacks, lines formed |
| `timestamp`  | DateTime      | When move was made                       |

**Constraints:**

- `@@unique([gameId, moveNumber])` - One move per number per game

**Move Types:**

```
Ring placement: place_ring, skip_placement
Movement: move_ring, move_stack, build_stack
Capture: overtaking_capture, continue_capture_segment
Decisions: process_line, choose_line_reward, process_territory_region, eliminate_rings_from_stack
Recovery: recovery_slide
```

### refresh_tokens

JWT refresh token management with rotation and reuse detection.

| Column       | Type          | Description                        |
| ------------ | ------------- | ---------------------------------- |
| `id`         | String (CUID) | Primary key                        |
| `token`      | String        | Unique token value                 |
| `userId`     | String        | FK to users                        |
| `expiresAt`  | DateTime      | Token expiration                   |
| `familyId`   | String?       | Token family for rotation tracking |
| `revokedAt`  | DateTime?     | When token was revoked             |
| `rememberMe` | Boolean       | Extended expiry (30d vs 7d)        |

**Security Features:**

- **Token families**: All rotated tokens share `familyId`
- **Reuse detection**: If revoked token is reused, entire family is invalidated
- **Soft revocation**: Tokens marked revoked instead of deleted

### chat_messages

In-game chat with 500 character limit.

| Column      | Type          | Description     |
| ----------- | ------------- | --------------- |
| `id`        | String (CUID) | Primary key     |
| `gameId`    | String        | FK to games     |
| `userId`    | String        | FK to users     |
| `message`   | VARCHAR(500)  | Message content |
| `createdAt` | DateTime      | Timestamp       |

### rematch_requests

Post-game rematch handling with 30-second timeout.

| Column        | Type          | Description                          |
| ------------- | ------------- | ------------------------------------ |
| `id`          | String (CUID) | Primary key                          |
| `gameId`      | String        | FK to original game                  |
| `requesterId` | String        | FK to requesting user                |
| `status`      | Enum          | pending, accepted, declined, expired |
| `expiresAt`   | DateTime      | 30-second timeout                    |
| `newGameId`   | String?       | ID of new game if accepted           |

**Constraints:**

- `@@unique([gameId, status])` - One pending request per game

### rating_history

Elo rating changes for analytics and display.

| Column      | Type          | Description                        |
| ----------- | ------------- | ---------------------------------- |
| `id`        | String (CUID) | Primary key                        |
| `userId`    | String        | FK to users                        |
| `gameId`    | String?       | FK to game (null for initial)      |
| `oldRating` | Int           | Rating before change               |
| `newRating` | Int           | Rating after change                |
| `change`    | Int           | Difference (newRating - oldRating) |
| `timestamp` | DateTime      | When change occurred               |

## Common Query Patterns

### Get User's Active Games

```typescript
const games = await prisma.game.findMany({
  where: {
    OR: [
      { player1Id: userId },
      { player2Id: userId },
      { player3Id: userId },
      { player4Id: userId },
    ],
    status: { in: ['waiting', 'active'] },
  },
  orderBy: { createdAt: 'desc' },
});
```

### Get Game with Moves

```typescript
const game = await prisma.game.findUnique({
  where: { id: gameId },
  include: {
    moves: { orderBy: { moveNumber: 'asc' } },
    player1: { select: { username: true, rating: true } },
    player2: { select: { username: true, rating: true } },
  },
});
```

### Get User's Rating History

```typescript
const history = await prisma.ratingHistory.findMany({
  where: { userId },
  orderBy: { timestamp: 'desc' },
  take: 50,
});
```

### Token Family Revocation

```typescript
// Revoke entire token family on reuse detection
await prisma.refreshToken.updateMany({
  where: { familyId: token.familyId },
  data: { revokedAt: new Date() },
});
```

## Migrations

Prisma handles migrations automatically:

```bash
# Create migration from schema changes
npx prisma migrate dev --name description_of_change

# Apply migrations in production
npx prisma migrate deploy

# Generate Prisma client
npx prisma generate
```

## Performance Considerations

1. **JSON columns** (`gameState`, `moveData`): Not directly queryable, use for storage only
2. **Soft deletes** (`deletedAt`): Always filter `WHERE deletedAt IS NULL` for active records
3. **Game state snapshots**: `finalState` stored only for completed games to save space
4. **Index usage**: Composite index on `[status, createdAt]` optimizes game list queries

## Related Documentation

- [Phase Orchestration Architecture](PHASE_ORCHESTRATION_ARCHITECTURE.md)
- [State Machines](STATE_MACHINES.md)
- [Prisma Documentation](https://www.prisma.io/docs)
