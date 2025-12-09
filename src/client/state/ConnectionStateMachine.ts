/**
 * ConnectionStateMachine - Discriminated union state machine for game connections
 *
 * This module provides a type-safe state machine for managing WebSocket connection
 * lifecycle in the game client. It consolidates connection-related state that was
 * previously scattered across multiple useState calls in GameContext.
 *
 * State Flow:
 * ```
 * idle ──connect()──> connecting ──onConnected()──> connected
 *                          │                            │
 *                          v                            v
 *                      error <──reconnect()───── reconnecting
 *                          │                            │
 *                          v                            v
 *                      (retry)                    disconnected
 * ```
 *
 * @module state/ConnectionStateMachine
 */

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Discriminated union of all connection states.
 */
export type ConnectionState =
  | ConnectionIdleState
  | ConnectionConnectingState
  | ConnectionConnectedState
  | ConnectionReconnectingState
  | ConnectionDisconnectedState
  | ConnectionErrorState;

/**
 * No active connection. Initial state.
 */
export interface ConnectionIdleState {
  readonly kind: 'idle';
}

/**
 * Attempting initial connection.
 */
export interface ConnectionConnectingState {
  readonly kind: 'connecting';
  /** Target game ID being connected to */
  readonly gameId: string;
  /** Timestamp when connection attempt started (ms since epoch) */
  readonly startedAt: number;
}

/**
 * Successfully connected and receiving game state.
 */
export interface ConnectionConnectedState {
  readonly kind: 'connected';
  /** Connected game ID */
  readonly gameId: string;
  /** Timestamp when connection was established */
  readonly connectedAt: number;
  /** Timestamp of most recent heartbeat */
  readonly lastHeartbeatAt: number;
}

/**
 * Lost connection, attempting automatic reconnection.
 */
export interface ConnectionReconnectingState {
  readonly kind: 'reconnecting';
  /** Game ID attempting to reconnect to */
  readonly gameId: string;
  /** Timestamp when reconnection started */
  readonly reconnectStartedAt: number;
  /** Number of reconnection attempts made */
  readonly attemptCount: number;
  /** Last known heartbeat before disconnect */
  readonly lastHeartbeatAt: number | null;
}

/**
 * Cleanly disconnected (user-initiated or game ended).
 */
export interface ConnectionDisconnectedState {
  readonly kind: 'disconnected';
  /** Reason for disconnect */
  readonly reason: DisconnectReason;
  /** Previous game ID if any */
  readonly previousGameId?: string;
  /** Timestamp when disconnected */
  readonly disconnectedAt: number;
}

/**
 * Connection failed with error.
 */
export interface ConnectionErrorState {
  readonly kind: 'error';
  /** Error message */
  readonly message: string;
  /** Error code if available */
  readonly code?: string;
  /** Game ID that failed */
  readonly gameId?: string;
  /** Timestamp of error */
  readonly errorAt: number;
  /** Whether retry is possible */
  readonly canRetry: boolean;
}

/**
 * Reasons for disconnection.
 */
export type DisconnectReason =
  | 'user_initiated'
  | 'game_ended'
  | 'server_closed'
  | 'network_error'
  | 'timeout';

// ═══════════════════════════════════════════════════════════════════════════
// STATE CONSTRUCTORS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create idle state.
 */
export function createIdleState(): ConnectionIdleState {
  return { kind: 'idle' };
}

/**
 * Create connecting state.
 */
export function createConnectingState(gameId: string, nowMs?: number): ConnectionConnectingState {
  return {
    kind: 'connecting',
    gameId,
    startedAt: nowMs ?? Date.now(),
  };
}

/**
 * Create connected state.
 */
export function createConnectedState(gameId: string, nowMs?: number): ConnectionConnectedState {
  const now = nowMs ?? Date.now();
  return {
    kind: 'connected',
    gameId,
    connectedAt: now,
    lastHeartbeatAt: now,
  };
}

/**
 * Create reconnecting state.
 */
export function createReconnectingState(
  gameId: string,
  lastHeartbeatAt: number | null,
  attemptCount: number = 1,
  nowMs?: number
): ConnectionReconnectingState {
  return {
    kind: 'reconnecting',
    gameId,
    reconnectStartedAt: nowMs ?? Date.now(),
    attemptCount,
    lastHeartbeatAt,
  };
}

/**
 * Create disconnected state.
 */
export function createDisconnectedState(
  reason: DisconnectReason,
  previousGameId?: string,
  nowMs?: number
): ConnectionDisconnectedState {
  return {
    kind: 'disconnected',
    reason,
    ...(previousGameId !== undefined ? { previousGameId } : {}),
    disconnectedAt: nowMs ?? Date.now(),
  };
}

/**
 * Create error state.
 */
export function createErrorState(
  message: string,
  options: {
    code?: string;
    gameId?: string;
    canRetry?: boolean;
    nowMs?: number;
  } = {}
): ConnectionErrorState {
  return {
    kind: 'error',
    message,
    ...(options.code ? { code: options.code } : {}),
    ...(options.gameId ? { gameId: options.gameId } : {}),
    errorAt: options.nowMs ?? Date.now(),
    canRetry: options.canRetry ?? true,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// STATE TRANSITIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Start connecting to a game.
 *
 * Valid from: idle, disconnected, error
 */
export function startConnecting(
  state: ConnectionState,
  gameId: string,
  nowMs?: number
): ConnectionConnectingState {
  if (state.kind !== 'idle' && state.kind !== 'disconnected' && state.kind !== 'error') {
    console.warn(
      `[ConnectionStateMachine] startConnecting called from invalid state: ${state.kind}`
    );
  }
  return createConnectingState(gameId, nowMs);
}

/**
 * Mark connection as established.
 *
 * Valid from: connecting, reconnecting
 */
export function markConnected(
  state: ConnectionConnectingState | ConnectionReconnectingState,
  nowMs?: number
): ConnectionConnectedState {
  return createConnectedState(state.gameId, nowMs);
}

/**
 * Update heartbeat timestamp.
 *
 * Valid from: connected
 */
export function updateHeartbeat(
  state: ConnectionConnectedState,
  nowMs?: number
): ConnectionConnectedState {
  return {
    ...state,
    lastHeartbeatAt: nowMs ?? Date.now(),
  };
}

/**
 * Start reconnection attempt.
 *
 * Valid from: connected
 */
export function startReconnecting(
  state: ConnectionConnectedState,
  nowMs?: number
): ConnectionReconnectingState {
  return createReconnectingState(state.gameId, state.lastHeartbeatAt, 1, nowMs);
}

/**
 * Increment reconnection attempt counter.
 *
 * Valid from: reconnecting
 */
export function incrementReconnectAttempt(
  state: ConnectionReconnectingState,
  nowMs?: number
): ConnectionReconnectingState {
  return {
    ...state,
    attemptCount: state.attemptCount + 1,
    reconnectStartedAt: nowMs ?? Date.now(),
  };
}

/**
 * Mark as disconnected.
 *
 * Valid from: any active state
 */
export function markDisconnected(
  state: ConnectionState,
  reason: DisconnectReason,
  nowMs?: number
): ConnectionDisconnectedState {
  const previousGameId =
    state.kind === 'connecting' || state.kind === 'connected' || state.kind === 'reconnecting'
      ? state.gameId
      : undefined;

  return createDisconnectedState(reason, previousGameId, nowMs);
}

/**
 * Mark as error.
 *
 * Valid from: connecting, reconnecting
 */
export function markError(
  state: ConnectionConnectingState | ConnectionReconnectingState,
  message: string,
  options: { code?: string; canRetry?: boolean; nowMs?: number } = {}
): ConnectionErrorState {
  return createErrorState(message, {
    ...options,
    gameId: state.gameId,
  });
}

/**
 * Reset to idle state.
 */
export function resetConnection(): ConnectionIdleState {
  return createIdleState();
}

// ═══════════════════════════════════════════════════════════════════════════
// QUERY HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if connection is in an active state (connecting or connected).
 */
export function isConnectionActive(
  state: ConnectionState
): state is ConnectionConnectingState | ConnectionConnectedState | ConnectionReconnectingState {
  return state.kind === 'connecting' || state.kind === 'connected' || state.kind === 'reconnecting';
}

/**
 * Check if connection is usable for game actions.
 */
export function isConnectionUsable(state: ConnectionState): state is ConnectionConnectedState {
  return state.kind === 'connected';
}

/**
 * Check if connection is in a failed/error state.
 */
export function isConnectionFailed(
  state: ConnectionState
): state is ConnectionErrorState | ConnectionDisconnectedState {
  return state.kind === 'error' || state.kind === 'disconnected';
}

/**
 * Get the game ID from the current state (if any).
 */
export function getConnectionGameId(state: ConnectionState): string | null {
  switch (state.kind) {
    case 'connecting':
    case 'connected':
    case 'reconnecting':
      return state.gameId;
    case 'error':
      return state.gameId ?? null;
    case 'disconnected':
      return state.previousGameId ?? null;
    case 'idle':
    default:
      return null;
  }
}

/**
 * Get time since last heartbeat (if connected/reconnecting).
 */
export function getTimeSinceHeartbeat(state: ConnectionState, nowMs?: number): number | null {
  const now = nowMs ?? Date.now();

  if (state.kind === 'connected') {
    return now - state.lastHeartbeatAt;
  }

  if (state.kind === 'reconnecting' && state.lastHeartbeatAt !== null) {
    return now - state.lastHeartbeatAt;
  }

  return null;
}

/**
 * Check if heartbeat is stale (exceeds threshold).
 */
export function isHeartbeatStale(
  state: ConnectionState,
  thresholdMs: number,
  nowMs?: number
): boolean {
  const timeSince = getTimeSinceHeartbeat(state, nowMs);
  return timeSince !== null && timeSince > thresholdMs;
}

/**
 * Map state kind to legacy ConnectionStatus for backward compatibility.
 */
export function toLegacyConnectionStatus(
  state: ConnectionState
): 'connected' | 'connecting' | 'reconnecting' | 'disconnected' {
  switch (state.kind) {
    case 'connected':
      return 'connected';
    case 'connecting':
      return 'connecting';
    case 'reconnecting':
      return 'reconnecting';
    case 'idle':
    case 'disconnected':
    case 'error':
    default:
      return 'disconnected';
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// STATE SUMMARY
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Summary of connection state for debugging/logging.
 */
export interface ConnectionStateSummary {
  kind: ConnectionState['kind'];
  gameId?: string;
  isActive: boolean;
  isUsable: boolean;
  error?: string;
  reconnectAttempts?: number;
  timeSinceHeartbeatMs?: number;
}

/**
 * Get a summary of the current connection state.
 */
export function getConnectionSummary(
  state: ConnectionState,
  nowMs?: number
): ConnectionStateSummary {
  const summary: ConnectionStateSummary = {
    kind: state.kind,
    isActive: isConnectionActive(state),
    isUsable: isConnectionUsable(state),
  };

  const gameId = getConnectionGameId(state);
  if (gameId) {
    summary.gameId = gameId;
  }

  if (state.kind === 'error') {
    summary.error = state.message;
  }

  if (state.kind === 'reconnecting') {
    summary.reconnectAttempts = state.attemptCount;
  }

  const heartbeatTime = getTimeSinceHeartbeat(state, nowMs);
  if (heartbeatTime !== null) {
    summary.timeSinceHeartbeatMs = heartbeatTime;
  }

  return summary;
}
