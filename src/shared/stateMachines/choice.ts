import type { PlayerChoice, PlayerChoiceType } from '../types/game';

/**
 * Explicit lifecycle model for a single PlayerChoice.
 *
 * This is intentionally dependency-light and backend-agnostic so it can be
 * used by WebSocketInteractionHandler, future HTTP transports, and tests as a
 * shared diagnostic view of choice handling.
 */

export type ChoiceRejectionReason = 'INVALID_OPTION' | 'PLAYER_MISMATCH';

export type ChoiceCancelReason = 'SERVER_CANCEL' | 'DISCONNECT';

export type ChoiceStatus =
  | {
      kind: 'pending';
      gameId: string;
      choiceId: string;
      playerNumber: number;
      choiceType?: PlayerChoiceType | undefined;
      requestedAt: number;
      /** Wall-clock deadline when the choice will time out, in ms since epoch. */
      deadlineAt: number;
    }
  | {
      kind: 'fulfilled';
      gameId: string;
      choiceId: string;
      playerNumber: number;
      choiceType?: PlayerChoiceType | undefined;
      completedAt: number;
    }
  | {
      kind: 'rejected';
      gameId: string;
      choiceId: string;
      playerNumber: number;
      choiceType?: PlayerChoiceType | undefined;
      completedAt: number;
      reason: ChoiceRejectionReason;
    }
  | {
      kind: 'canceled';
      gameId: string;
      choiceId: string;
      playerNumber: number;
      choiceType?: PlayerChoiceType | undefined;
      completedAt: number;
      reason: ChoiceCancelReason;
    }
  | {
      kind: 'expired';
      gameId: string;
      choiceId: string;
      playerNumber: number;
      choiceType?: PlayerChoiceType | undefined;
      requestedAt: number;
      deadlineAt: number;
      completedAt: number;
    };

function baseFromChoice(choice: PlayerChoice) {
  return {
    gameId: choice.gameId,
    choiceId: choice.id,
    playerNumber: choice.playerNumber,
    choiceType: choice.type,
  } as const;
}

export function makePendingChoiceStatus(
  choice: PlayerChoice,
  timeoutMs: number,
  now: number = Date.now()
): ChoiceStatus {
  const deadlineAt = now + timeoutMs;
  return {
    kind: 'pending',
    ...baseFromChoice(choice),
    requestedAt: now,
    deadlineAt,
  };
}

export function markChoiceFulfilled(
  previous: ChoiceStatus,
  now: number = Date.now()
): ChoiceStatus {
  return {
    kind: 'fulfilled',
    gameId: previous.gameId,
    choiceId: previous.choiceId,
    playerNumber: previous.playerNumber,
    choiceType: previous.choiceType,
    completedAt: now,
  };
}

export function markChoiceRejected(
  previous: ChoiceStatus,
  reason: ChoiceRejectionReason,
  now: number = Date.now()
): ChoiceStatus {
  return {
    kind: 'rejected',
    gameId: previous.gameId,
    choiceId: previous.choiceId,
    playerNumber: previous.playerNumber,
    choiceType: previous.choiceType,
    completedAt: now,
    reason,
  };
}

export function markChoiceCanceled(
  previous: ChoiceStatus,
  reason: ChoiceCancelReason,
  now: number = Date.now()
): ChoiceStatus {
  return {
    kind: 'canceled',
    gameId: previous.gameId,
    choiceId: previous.choiceId,
    playerNumber: previous.playerNumber,
    choiceType: previous.choiceType,
    completedAt: now,
    reason,
  };
}

export function markChoiceExpired(previous: ChoiceStatus, now: number = Date.now()): ChoiceStatus {
  // For diagnostics it is useful to preserve both the original requestedAt and
  // deadlineAt when transitioning to an expired state. If the previous status
  // was not pending (e.g. defensive call), fall back to treating `now` as both
  // requestedAt and deadlineAt.
  const requestedAt = previous.kind === 'pending' ? previous.requestedAt : now;
  const deadlineAt = previous.kind === 'pending' ? previous.deadlineAt : now;

  return {
    kind: 'expired',
    gameId: previous.gameId,
    choiceId: previous.choiceId,
    playerNumber: previous.playerNumber,
    choiceType: previous.choiceType,
    requestedAt,
    deadlineAt,
    completedAt: now,
  };
}
