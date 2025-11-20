import { GameResult, GameState, Position } from '../../shared/types/game';
import { checkSandboxVictory } from './sandboxVictory';

/**
 * Hooks for sandbox game-end processing. The host engine (e.g. ClientSandboxEngine)
 * owns the authoritative GameState and placement helpers; this module provides
 * pure-ish helpers that operate via those hooks.
 */
export interface SandboxGameEndHooks {
  /**
   * Enumerate all legal ring placements for the given player under the
   * sandbox no-dead-placement rule. This is used by the global stalemate
   * resolver to determine whether a player with rings in hand could
   * theoretically re-enter play on a bare board.
   */
  enumerateLegalRingPlacements(playerNumber: number): Position[];
}

/**
 * Detect global structural stalemate in the sandbox when the board has no
 * stacks remaining and no player has any legal ring placements left, even
 * if some players still hold rings in hand.
 *
 * In this situation the compact rules' stalemate ladder treats all rings
 * remaining in hand as eliminated (hand ≔ E) for tie-break purposes. We
 * mirror the backend resolveBlockedStateForCurrentPlayerForTesting helper
 * by converting those rings-in-hand into eliminated counts before running
 * checkSandboxVictory, which then applies the usual territory / elimination
 * tie-breakers.
 */
export function resolveGlobalStalemateIfNeededSandbox(
  state: GameState,
  hooks: SandboxGameEndHooks
): GameState {
  // Require an active game and a bare board.
  if (state.gameStatus !== 'active') {
    return state;
  }

  if (state.board.stacks.size !== 0) {
    return state;
  }

  const players = state.players;
  const anyRingsInHand = players.some((p) => p.ringsInHand > 0);

  // If nobody holds rings in hand, checkSandboxVictory already handles the
  // structural terminal case via its noStacksLeft && !anyRingsInHand branch.
  if (!anyRingsInHand) {
    return state;
  }

  // If any player with rings in hand still has at least one legal placement
  // under the sandbox no-dead-placement rule, the game is not yet globally
  // terminal: they may be able to re-enter play once phases advance.
  const anyLegalPlacementForAnyPlayer = players.some((p) => {
    if (p.ringsInHand <= 0) {
      return false;
    }
    const placements = hooks.enumerateLegalRingPlacements(p.playerNumber);
    return placements.length > 0;
  });

  if (anyLegalPlacementForAnyPlayer) {
    return state;
  }

  // At this point the board has no stacks, multiple players may still have
  // rings in hand, and none of them has a legal placement. Convert all
  // rings-in-hand into eliminated counts so that the sandbox victory helper
  // can apply its structural terminal tie-breakers.
  let handEliminations = 0;
  const updatedEliminatedRings: { [player: number]: number } = {
    ...state.board.eliminatedRings,
  };

  const updatedPlayers = players.map((p) => {
    if (p.ringsInHand <= 0) {
      return p;
    }

    const delta = p.ringsInHand;
    handEliminations += delta;

    const nextEliminated = p.eliminatedRings + delta;
    const existingBoardElims = updatedEliminatedRings[p.playerNumber] || 0;
    updatedEliminatedRings[p.playerNumber] = existingBoardElims + delta;

    return {
      ...p,
      ringsInHand: 0,
      eliminatedRings: nextEliminated,
    };
  });

  if (handEliminations === 0) {
    return state;
  }

  return {
    ...state,
    players: updatedPlayers,
    totalRingsEliminated: state.totalRingsEliminated + handEliminations,
    board: {
      ...state.board,
      eliminatedRings: updatedEliminatedRings,
    },
  };
}

/**
 * Apply ring-elimination and territory-control victory checks after
 * post-movement processing. When a winner is found, the returned state is
 * marked as completed and the associated GameResult is returned alongside
 * it; otherwise the original game remains active.
 */
export function checkAndApplyVictorySandbox(
  state: GameState,
  hooks: SandboxGameEndHooks
): { state: GameState; result: GameResult | null } {
  if (state.gameStatus !== 'active') {
    return { state, result: null };
  }

  const afterStalemate = resolveGlobalStalemateIfNeededSandbox(state, hooks);
  const result = checkSandboxVictory(afterStalemate);

  if (!result) {
    return { state: afterStalemate, result: null };
  }

  const nextState: GameState = {
    ...afterStalemate,
    gameStatus: 'completed',
    winner: result.winner,
  };

  return { state: nextState, result };
}
