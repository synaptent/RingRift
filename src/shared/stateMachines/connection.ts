/**
 * Explicit lifecycle model for a player's WebSocket connection to a game.
 *
 * This is intentionally dependency-light and backend-agnostic so it can be
 * used by WebSocketServer and tests as a shared diagnostic view of
 * connection/reconnection handling.
 */

export type PlayerConnectionState =
  | {
      kind: 'connected';
      gameId: string;
      userId: string;
      /** Numeric player seat when the user is a player; omitted for spectators. */
      playerNumber?: number;
      /** First time we observed this user as connected for this game. */
      connectedAt: number;
      /** Last time we saw an explicit connection event for this game/user. */
      lastSeenAt: number;
    }
  | {
      kind: 'disconnected_pending_reconnect';
      gameId: string;
      userId: string;
      playerNumber: number;
      /** When the disconnect was observed. */
      disconnectedAt: number;
      /** Deadline after which we will treat the disconnect as expired. */
      deadlineAt: number;
    }
  | {
      kind: 'disconnected_expired';
      gameId: string;
      userId: string;
      playerNumber: number;
      /** When the disconnect was originally observed. */
      disconnectedAt: number;
      /** When the reconnection window actually expired. */
      expiredAt: number;
    };

/**
 * Mark a user as connected to a given game. If they were already connected,
 * we preserve the original connectedAt timestamp and only bump lastSeenAt.
 */
export function markConnected(
  gameId: string,
  userId: string,
  playerNumber: number | undefined,
  previous: PlayerConnectionState | undefined,
  now: number = Date.now()
): PlayerConnectionState {
  const connectedAt =
    previous &&
    previous.kind === 'connected' &&
    previous.gameId === gameId &&
    previous.userId === userId
      ? previous.connectedAt
      : now;

  return {
    kind: 'connected',
    gameId,
    userId,
    ...(playerNumber !== undefined ? { playerNumber } : {}),
    connectedAt,
    lastSeenAt: now,
  };
}

/**
 * Mark a player as disconnected but still within a reconnection window.
 */
export function markDisconnectedPendingReconnect(
  previous: PlayerConnectionState | undefined,
  gameId: string,
  userId: string,
  playerNumber: number,
  timeoutMs: number,
  now: number = Date.now()
): PlayerConnectionState {
  const disconnectedAt =
    previous &&
    previous.gameId === gameId &&
    previous.userId === userId &&
    (previous.kind === 'connected' || previous.kind === 'disconnected_pending_reconnect')
      ? previous.kind === 'disconnected_pending_reconnect'
        ? previous.disconnectedAt
        : now
      : now;

  return {
    kind: 'disconnected_pending_reconnect',
    gameId,
    userId,
    playerNumber,
    disconnectedAt,
    deadlineAt: now + timeoutMs,
  };
}

/**
 * Mark a player's pending reconnection window as expired.
 */
export function markDisconnectedExpired(
  previous: PlayerConnectionState | undefined,
  gameId: string,
  userId: string,
  playerNumber: number,
  now: number = Date.now()
): PlayerConnectionState {
  const disconnectedAt =
    previous &&
    previous.gameId === gameId &&
    previous.userId === userId &&
    previous.kind === 'disconnected_pending_reconnect'
      ? previous.disconnectedAt
      : now;

  return {
    kind: 'disconnected_expired',
    gameId,
    userId,
    playerNumber,
    disconnectedAt,
    expiredAt: now,
  };
}
