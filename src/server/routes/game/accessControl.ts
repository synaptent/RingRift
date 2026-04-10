import { createError } from '../../middleware/errorHandler';

/**
 * Lightweight view of game participants used for HTTP-level authorization
 * checks. This intentionally mirrors only the player slots and spectator
 * flag that are required to enforce access control invariants.
 */
export type GameParticipantSnapshot = {
  player1Id: string | null;
  player2Id: string | null;
  player3Id: string | null;
  player4Id: string | null;
  allowSpectators?: boolean | null;
};

export const isUserParticipantInGame = (userId: string, game: GameParticipantSnapshot): boolean => {
  return [game.player1Id, game.player2Id, game.player3Id, game.player4Id]
    .filter(Boolean)
    .includes(userId);
};

/**
 * Enforce the invariant that only participants (or, when enabled, permitted
 * spectators) may inspect game-scoped HTTP resources.
 */
export const assertUserCanViewGame = (
  userId: string,
  game: GameParticipantSnapshot & { allowSpectators: boolean }
): void => {
  const isParticipant = isUserParticipantInGame(userId, game);

  if (!isParticipant && !game.allowSpectators) {
    throw createError('Access denied', 403, 'ACCESS_DENIED');
  }
};

/**
 * Enforce the invariant that only seated human participants in a game may
 * perform HTTP mutations that change its state.
 */
export const assertUserIsGameParticipant = (
  userId: string,
  game: GameParticipantSnapshot
): void => {
  if (!isUserParticipantInGame(userId, game)) {
    throw createError('Access denied', 403, 'ACCESS_DENIED');
  }
};
