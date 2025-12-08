/**
 * RematchStateMachine - Discriminated union state machine for rematch flow
 *
 * This module provides a type-safe state machine for managing the rematch
 * request/response lifecycle after a game ends. It consolidates rematch-related
 * state that was previously scattered across multiple useState calls.
 *
 * State Flow:
 * ```
 * idle ──requestRematch()──> pending_request ──onResponse(accepted)──> accepted
 *   │                                │                                     │
 *   │                                v                                     v
 *   │<──onResponse(declined)───── declined                           navigate
 *   │
 *   └──onRequestReceived()──> pending_response ──respond(accept)──> accepted
 *                                    │
 *                                    v
 *                                declined
 * ```
 *
 * @module state/RematchStateMachine
 */

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Discriminated union of all rematch states.
 */
export type RematchState =
  | RematchIdleState
  | RematchPendingRequestState
  | RematchPendingResponseState
  | RematchAcceptedState
  | RematchDeclinedState
  | RematchExpiredState;

/**
 * No active rematch flow. Initial state.
 */
export interface RematchIdleState {
  readonly kind: 'idle';
}

/**
 * Local user has requested a rematch, waiting for opponent response.
 */
export interface RematchPendingRequestState {
  readonly kind: 'pending_request';
  /** ID of the completed game */
  readonly gameId: string;
  /** Timestamp when request was sent */
  readonly requestedAt: number;
}

/**
 * Opponent has requested a rematch, waiting for local user response.
 */
export interface RematchPendingResponseState {
  readonly kind: 'pending_response';
  /** Rematch request ID */
  readonly requestId: string;
  /** ID of the completed game */
  readonly gameId: string;
  /** Username of the requester */
  readonly requesterUsername: string;
  /** Timestamp when request was received */
  readonly receivedAt: number;
}

/**
 * Rematch was accepted, new game created.
 */
export interface RematchAcceptedState {
  readonly kind: 'accepted';
  /** ID of the new game */
  readonly newGameId: string;
  /** ID of the original game */
  readonly originalGameId: string;
  /** Timestamp when accepted */
  readonly acceptedAt: number;
}

/**
 * Rematch was declined.
 */
export interface RematchDeclinedState {
  readonly kind: 'declined';
  /** Who declined: 'local' or 'opponent' */
  readonly declinedBy: 'local' | 'opponent';
  /** ID of the original game */
  readonly gameId: string;
  /** Timestamp when declined */
  readonly declinedAt: number;
}

/**
 * Rematch request expired (timeout).
 */
export interface RematchExpiredState {
  readonly kind: 'expired';
  /** ID of the original game */
  readonly gameId: string;
  /** Timestamp when expired */
  readonly expiredAt: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// STATE CONSTRUCTORS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create idle state.
 */
export function createRematchIdleState(): RematchIdleState {
  return { kind: 'idle' };
}

/**
 * Create pending request state (local user initiated).
 */
export function createPendingRequestState(
  gameId: string,
  nowMs?: number
): RematchPendingRequestState {
  return {
    kind: 'pending_request',
    gameId,
    requestedAt: nowMs ?? Date.now(),
  };
}

/**
 * Create pending response state (opponent initiated).
 */
export function createPendingResponseState(
  requestId: string,
  gameId: string,
  requesterUsername: string,
  nowMs?: number
): RematchPendingResponseState {
  return {
    kind: 'pending_response',
    requestId,
    gameId,
    requesterUsername,
    receivedAt: nowMs ?? Date.now(),
  };
}

/**
 * Create accepted state.
 */
export function createRematchAcceptedState(
  newGameId: string,
  originalGameId: string,
  nowMs?: number
): RematchAcceptedState {
  return {
    kind: 'accepted',
    newGameId,
    originalGameId,
    acceptedAt: nowMs ?? Date.now(),
  };
}

/**
 * Create declined state.
 */
export function createRematchDeclinedState(
  declinedBy: 'local' | 'opponent',
  gameId: string,
  nowMs?: number
): RematchDeclinedState {
  return {
    kind: 'declined',
    declinedBy,
    gameId,
    declinedAt: nowMs ?? Date.now(),
  };
}

/**
 * Create expired state.
 */
export function createRematchExpiredState(
  gameId: string,
  nowMs?: number
): RematchExpiredState {
  return {
    kind: 'expired',
    gameId,
    expiredAt: nowMs ?? Date.now(),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// STATE TRANSITIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Local user requests a rematch.
 *
 * Valid from: idle
 */
export function requestRematch(
  state: RematchState,
  gameId: string,
  nowMs?: number
): RematchPendingRequestState {
  if (state.kind !== 'idle') {
    console.warn(
      `[RematchStateMachine] requestRematch called from invalid state: ${state.kind}`
    );
  }
  return createPendingRequestState(gameId, nowMs);
}

/**
 * Opponent requests a rematch.
 *
 * Valid from: idle
 */
export function receiveRematchRequest(
  state: RematchState,
  requestId: string,
  gameId: string,
  requesterUsername: string,
  nowMs?: number
): RematchPendingResponseState {
  if (state.kind !== 'idle') {
    console.warn(
      `[RematchStateMachine] receiveRematchRequest called from invalid state: ${state.kind}`
    );
  }
  return createPendingResponseState(requestId, gameId, requesterUsername, nowMs);
}

/**
 * Rematch is accepted (by either party).
 *
 * Valid from: pending_request, pending_response
 */
export function acceptRematch(
  state: RematchPendingRequestState | RematchPendingResponseState,
  newGameId: string,
  nowMs?: number
): RematchAcceptedState {
  return createRematchAcceptedState(newGameId, state.gameId, nowMs);
}

/**
 * Local user declines a rematch request.
 *
 * Valid from: pending_response
 */
export function declineRematchLocally(
  state: RematchPendingResponseState,
  nowMs?: number
): RematchDeclinedState {
  return createRematchDeclinedState('local', state.gameId, nowMs);
}

/**
 * Opponent declines the rematch request.
 *
 * Valid from: pending_request
 */
export function receiveRematchDecline(
  state: RematchPendingRequestState,
  nowMs?: number
): RematchDeclinedState {
  return createRematchDeclinedState('opponent', state.gameId, nowMs);
}

/**
 * Rematch request expires.
 *
 * Valid from: pending_request, pending_response
 */
export function expireRematch(
  state: RematchPendingRequestState | RematchPendingResponseState,
  nowMs?: number
): RematchExpiredState {
  return createRematchExpiredState(state.gameId, nowMs);
}

/**
 * Reset to idle state.
 */
export function resetRematch(): RematchIdleState {
  return createRematchIdleState();
}

// ═══════════════════════════════════════════════════════════════════════════
// QUERY HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if there's an active rematch flow.
 */
export function isRematchActive(
  state: RematchState
): state is RematchPendingRequestState | RematchPendingResponseState {
  return state.kind === 'pending_request' || state.kind === 'pending_response';
}

/**
 * Check if waiting for opponent's response.
 */
export function isAwaitingOpponentResponse(
  state: RematchState
): state is RematchPendingRequestState {
  return state.kind === 'pending_request';
}

/**
 * Check if waiting for local user's response.
 */
export function isAwaitingLocalResponse(
  state: RematchState
): state is RematchPendingResponseState {
  return state.kind === 'pending_response';
}

/**
 * Check if rematch was accepted and has a new game ID.
 */
export function hasNewGame(state: RematchState): state is RematchAcceptedState {
  return state.kind === 'accepted';
}

/**
 * Get the new game ID if rematch was accepted.
 */
export function getNewGameId(state: RematchState): string | null {
  return state.kind === 'accepted' ? state.newGameId : null;
}

/**
 * Get terminal status for legacy compatibility.
 */
export function getLegacyRematchStatus(
  state: RematchState
): 'accepted' | 'declined' | 'expired' | null {
  switch (state.kind) {
    case 'accepted':
      return 'accepted';
    case 'declined':
      return 'declined';
    case 'expired':
      return 'expired';
    default:
      return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// STATE SUMMARY
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Summary of rematch state for UI display.
 */
export interface RematchStateSummary {
  kind: RematchState['kind'];
  isActive: boolean;
  /** User-facing status message */
  message: string;
  /** Request ID if pending response */
  requestId?: string;
  /** New game ID if accepted */
  newGameId?: string;
  /** Requester username if pending response */
  requesterUsername?: string;
}

/**
 * Get a summary of the current rematch state.
 */
export function getRematchSummary(state: RematchState): RematchStateSummary {
  const base = {
    kind: state.kind,
    isActive: isRematchActive(state),
  };

  switch (state.kind) {
    case 'idle':
      return { ...base, message: 'No rematch in progress' };

    case 'pending_request':
      return { ...base, message: 'Waiting for opponent to respond...' };

    case 'pending_response':
      return {
        ...base,
        message: `${state.requesterUsername} wants a rematch`,
        requestId: state.requestId,
        requesterUsername: state.requesterUsername,
      };

    case 'accepted':
      return {
        ...base,
        message: 'Rematch accepted! Starting new game...',
        newGameId: state.newGameId,
      };

    case 'declined':
      return {
        ...base,
        message:
          state.declinedBy === 'local'
            ? 'You declined the rematch'
            : 'Opponent declined the rematch',
      };

    case 'expired':
      return { ...base, message: 'Rematch request expired' };
  }
}
