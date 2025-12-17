import type { GameState, GameResult } from '../../shared/types/game';
import { computeGlobalLegalActionsSummary, isANMState } from '../../shared/engine/globalActions';

/**
 * Discriminated union describing high-level "weird" rules states that the
 * HUD and teaching surfaces can expose to players. This helper is
 * intentionally client-only and does not modify any engine semantics.
 */
export type WeirdStateBanner =
  | { type: 'none' }
  | { type: 'active-no-moves-movement'; playerNumber: number }
  | { type: 'active-no-moves-line'; playerNumber: number }
  | { type: 'active-no-moves-territory'; playerNumber: number }
  | { type: 'forced-elimination'; playerNumber: number }
  | { type: 'last-player-standing'; winner?: number; reason: GameResult['reason'] | null }
  | { type: 'structural-stalemate'; winner?: number; reason: GameResult['reason'] | null };

/**
 * Compute a coarse-grained weird-state classification for the current
 * snapshot. This is a thin adapter over the shared engine helpers:
 *
 * - computeGlobalLegalActionsSummary
 * - isANMState
 *
 * and is used purely for UX. It **must not** be used to drive rules
 * semantics or host / engine behaviour.
 */
export function getWeirdStateBanner(
  gameState: GameState,
  opts: { victoryState?: GameResult | null } = {}
): WeirdStateBanner {
  const { victoryState } = opts;

  // Completed / terminal games: surface structural stalemate when the
  // engine reports a non-standard terminal reason.
  if (gameState.gameStatus !== 'active') {
    if (victoryState && victoryState.reason === 'game_completed') {
      return {
        type: 'structural-stalemate',
        ...(victoryState.winner !== undefined ? { winner: victoryState.winner } : {}),
        reason: victoryState.reason,
      };
    }
    if (victoryState && victoryState.reason === 'last_player_standing') {
      return {
        type: 'last-player-standing',
        ...(victoryState.winner !== undefined ? { winner: victoryState.winner } : {}),
        reason: victoryState.reason,
      };
    }
    return { type: 'none' };
  }

  const currentPlayer = gameState.currentPlayer;
  const playerState = gameState.players.find((p) => p.playerNumber === currentPlayer);

  if (!playerState) {
    return { type: 'none' };
  }

  // ANM(state) in the shared engine encodes "has material but no global
  // legal actions (placements, phase-local moves, or forced elimination)".
  // In canonical flows this should only appear transiently, but we still
  // classify it explicitly for UX and tests.
  const isAnm = isANMState(gameState);

  if (isAnm) {
    switch (gameState.currentPhase) {
      case 'line_processing':
        return { type: 'active-no-moves-line', playerNumber: currentPlayer };
      case 'territory_processing':
        return { type: 'active-no-moves-territory', playerNumber: currentPlayer };
      case 'forced_elimination':
        // In forced_elimination phase, the player has forced_elimination moves
        // available, so ANM should not occur. If it does, treat as movement-blocked.
        return { type: 'active-no-moves-movement', playerNumber: currentPlayer };
      case 'movement':
      case 'capture':
      case 'chain_capture':
      case 'ring_placement':
      default:
        return { type: 'active-no-moves-movement', playerNumber: currentPlayer };
    }
  }

  const summary = computeGlobalLegalActionsSummary(gameState, currentPlayer);

  // Forced-elimination availability is a special global action surface:
  // the player controls stacks but has no legal placements or movement /
  // capture actions. This is the canonical "blocked with stacks" state
  // used by FE invariants and is exactly the situation we want the HUD
  // to explain to players.
  if (
    summary.hasTurnMaterial &&
    summary.hasForcedEliminationAction &&
    !summary.hasGlobalPlacementAction &&
    !summary.hasPhaseLocalInteractiveMove
  ) {
    return { type: 'forced-elimination', playerNumber: currentPlayer };
  }

  // No recognised weird state for this snapshot.
  return { type: 'none' };
}
