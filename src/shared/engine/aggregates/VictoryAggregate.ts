/**
 * ═══════════════════════════════════════════════════════════════════════════
 * VictoryAggregate - Consolidated Victory Domain
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This aggregate consolidates all victory condition evaluation and tie-breaker
 * logic from:
 *
 * - victoryLogic.ts → victory evaluation, game end detection
 *
 * Rule Reference: Section 13 - Victory Conditions
 *
 * Key Rules:
 * - RR-CANON-R170: Score threshold (1 point per line collapse, win at 3/4/5 points based on board)
 * - RR-CANON-R171: Marker threshold (win with 7+ markers on board)
 * - RR-CANON-R172: Last player standing (all opponents eliminated)
 * - RR-CANON-R173: Game termination conditions
 * - RR-CANON-R174: Tie-breaking rules (most markers → most rings → shared victory)
 * - Full-round requirement for Last Player Standing
 *
 * Design principles:
 * - Pure functions: No side effects, return new state
 * - Type safety: Full TypeScript typing
 * - Backward compatibility: Source files continue to export their functions
 */

import type { GameState, BoardState, BoardType, Position, Player } from '../../types/game';
import { positionToString } from '../../types/game';
import type { MovementBoardView } from '../core';
import { hasAnyLegalMoveOrCaptureFromOnBoard } from '../core';
import { hasGlobalPlacementAction, hasForcedEliminationAction } from '../globalActions';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Reason for game ending.
 */
export type VictoryReason =
  | 'ring_elimination'
  | 'territory_control'
  | 'last_player_standing'
  | 'game_completed';

/**
 * Result of evaluating victory conditions.
 */
export interface VictoryResult {
  /** True if the game has ended */
  isGameOver: boolean;
  /** Player number of the winner (undefined if draw or no winner) */
  winner?: number;
  /** Reason for the victory (undefined if game not over) */
  reason?: VictoryReason;
  /**
   * True when the result arises from a bare-board global stalemate where
   * rings remaining in hand are conceptually treated as eliminated for
   * tie-breaking purposes (FAQ Q11, §13.4).
   *
   * Hosts that persist elimination statistics (e.g. sandbox UI) may use
   * this flag to apply a hand→eliminated conversion before building a
   * final GameResult while keeping evaluateVictory itself side-effect free.
   */
  handCountsAsEliminated?: boolean;
}

/**
 * Victory evaluation context (for extended queries).
 */
export interface VictoryEvaluationContext {
  /** Whether to include tie-breaker details */
  includeTieBreakers?: boolean;
  /** Whether to check for stalemate conditions */
  checkStalemate?: boolean;
}

/**
 * Detailed victory evaluation result with standings.
 */
export interface DetailedVictoryResult extends VictoryResult {
  /** Player standings in order (winner first) */
  standings?: Player[];
  /** Score breakdown by player */
  scores?: {
    [playerNumber: number]: {
      eliminatedRings: number;
      territorySpaces: number;
      markerCount: number;
    };
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check whether the specified player has at least one legal ring placement
 * on a bare board (no stacks present) that satisfies the no-dead-placement
 * rule. This is used only for structural terminality detection; it is not
 * a general-purpose placement enumerator.
 */
function hasAnyLegalPlacementOnBareBoard(state: GameState, playerNumber: number): boolean {
  const player = state.players.find((p) => p.playerNumber === playerNumber);
  if (!player || player.ringsInHand <= 0) {
    return false;
  }

  const board = state.board;
  const boardType: BoardType = board.type;

  // Precondition: callers only invoke this when there are no stacks on the
  // board. Defensively handle unexpected stacks by delegating to the shared
  // movement helper anyway.
  const hasStacks = board.stacks.size > 0;

  let found = false;
  forEachBoardPosition(board, (pos) => {
    if (found) {
      return;
    }

    const key = positionToString(pos);

    // Do not allow placement on collapsed territory or markers; stacks and
    // markers are mutually exclusive with territory under the board
    // invariants.
    if (board.collapsedSpaces.has(key)) {
      return;
    }

    if (board.markers.has(key)) {
      return;
    }

    // Build a minimal MovementBoardView over a hypothetical board that has
    // a single new stack at `pos` for the given player. When the board
    // unexpectedly contains stacks, we also expose them to the view so
    // no-dead-placement remains conservative.
    const hypotheticalKey = key;

    const view: MovementBoardView = {
      isValidPosition: (p) => isValidBoardPosition(boardType, board.size, p),
      isCollapsedSpace: (p) => {
        const k = positionToString(p);
        return board.collapsedSpaces.has(k);
      },
      getStackAt: (p) => {
        const k = positionToString(p);
        if (k === hypotheticalKey) {
          return {
            controllingPlayer: playerNumber,
            capHeight: 1,
            stackHeight: 1,
          };
        }
        if (!hasStacks) {
          return undefined;
        }
        const existing = board.stacks.get(k);
        if (!existing) {
          return undefined;
        }
        return {
          controllingPlayer: existing.controllingPlayer,
          capHeight: existing.capHeight,
          stackHeight: existing.stackHeight,
        };
      },
      getMarkerOwner: (p) => {
        const k = positionToString(p);
        const marker = board.markers.get(k);
        return marker?.player;
      },
    };

    if (hasAnyLegalMoveOrCaptureFromOnBoard(boardType, pos, playerNumber, view)) {
      found = true;
    }
  });

  return found;
}

/**
 * Iterate over all valid positions on the board.
 */
function forEachBoardPosition(board: BoardState, fn: (pos: Position) => void): void {
  const size = board.size;

  if (board.type === 'hexagonal') {
    const radius = size - 1;
    for (let q = -radius; q <= radius; q++) {
      const r1 = Math.max(-radius, -q - radius);
      const r2 = Math.min(radius, -q + radius);
      for (let r = r1; r <= r2; r++) {
        const s = -q - r;
        fn({ x: q, y: r, z: s });
      }
    }
  } else {
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        fn({ x, y });
      }
    }
  }
}

/**
 * Check if a position is valid for the given board type and size.
 */
function isValidBoardPosition(boardType: BoardType, size: number, position: Position): boolean {
  if (boardType === 'hexagonal') {
    const radius = size - 1;
    const q = position.x;
    const r = position.y;
    const s = position.z ?? -q - r;
    return (
      Math.abs(q) <= radius && Math.abs(r) <= radius && Math.abs(s) <= radius && q + r + s === 0
    );
  }

  return position.x >= 0 && position.x < size && position.y >= 0 && position.y < size;
}

/**
 * Count markers on the board for each player.
 */
function countMarkersByPlayer(state: GameState): Map<number, number> {
  const markerCounts = new Map<number, number>();

  for (const player of state.players) {
    markerCounts.set(player.playerNumber, 0);
  }

  for (const marker of state.board.markers.values()) {
    const owner = marker.player;
    const current = markerCounts.get(owner) ?? 0;
    markerCounts.set(owner, current + 1);
  }

  return markerCounts;
}

// ═══════════════════════════════════════════════════════════════════════════
// Query Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Canonical, side-effect-free victory evaluator for RingRift.
 *
 * Responsibilities:
 * - Primary victories:
 *   - Ring-elimination (Section 13.1) via GameState.victoryThreshold.
 *   - Territory-control (Section 13.2) via GameState.territoryVictoryThreshold.
 * - Structural terminality on a bare board:
 *   - If any player still has a legal ring placement (respecting the
 *     no-dead-placement rule via hasAnyLegalMoveOrCaptureFromOnBoard),
 *     the game is *not* over.
 *   - Otherwise apply the stalemate ladder (Section 13.4):
 *     territory → eliminated rings (including hand when appropriate) →
 *     markers → last actor → game_completed.
 *
 * This helper is shared by the backend RuleEngine, the shared GameEngine,
 * and the client sandbox so that victory semantics are defined in a single
 * place.
 *
 * Rule Reference: Section 13 - Victory Conditions
 *
 * @param state The current game state to evaluate
 * @returns Victory result indicating if game is over and why
 */
export function evaluateVictory(state: GameState): VictoryResult {
  const players = state.players;

  if (!players || players.length === 0) {
    return { isGameOver: false };
  }

  // 1) Ring-elimination victory: strictly more than 50% of total rings in
  // play have been eliminated for a single player.
  const ringWinner = players.find((p) => p.eliminatedRings >= state.victoryThreshold);
  if (ringWinner) {
    return {
      isGameOver: true,
      winner: ringWinner.playerNumber,
      reason: 'ring_elimination',
      handCountsAsEliminated: false,
    };
  }

  // 2) Territory-control victory: strictly more than 50% of the board's
  // spaces are controlled as territory by a single player.
  const territoryWinner = players.find((p) => p.territorySpaces >= state.territoryVictoryThreshold);
  if (territoryWinner) {
    return {
      isGameOver: true,
      winner: territoryWinner.playerNumber,
      reason: 'territory_control',
      handCountsAsEliminated: false,
    };
  }

  // 3) Early Last-Player-Standing (R172): if exactly one player has stacks
  // on the board AND all other players have neither stacks nor rings in hand,
  // that player wins immediately. This matches Python's Early LPS check.
  // Note: We only trigger this when there ARE stacks on board. If no stacks
  // exist, we fall through to the bare-board stalemate logic which handles
  // global structural terminality.
  const playersWithStacks = new Set<number>();
  for (const stack of state.board.stacks.values()) {
    playersWithStacks.add(stack.controllingPlayer);
  }

  // Only consider Early LPS when exactly one player has stacks
  if (playersWithStacks.size === 1) {
    const stackOwner = Array.from(playersWithStacks)[0];
    // Check if ALL other players have no material (no stacks, no rings in hand)
    const othersHaveMaterial = players.some(
      (p) =>
        p.playerNumber !== stackOwner &&
        (playersWithStacks.has(p.playerNumber) || p.ringsInHand > 0)
    );
    if (!othersHaveMaterial) {
      return {
        isGameOver: true,
        winner: stackOwner,
        reason: 'last_player_standing',
        handCountsAsEliminated: false,
      };
    }
  }

  // 4) Trapped position stalemate: All active players have stacks but no legal actions.
  // This handles AI vs AI games that stall when both players are blocked.
  if (state.board.stacks.size > 0) {
    // Check if any player with material can still make progress
    let somePlayerCanAct = false;

    for (const p of players) {
      // Check if player has any stacks
      let playerHasStacks = false;
      for (const stack of state.board.stacks.values()) {
        if (stack.controllingPlayer === p.playerNumber) {
          playerHasStacks = true;
          break;
        }
      }

      // Check if player has material (stacks or rings in hand)
      const hasMaterial = playerHasStacks || p.ringsInHand > 0;
      if (!hasMaterial) {
        continue; // Player eliminated, skip
      }

      // Check if player can place a ring
      if (hasGlobalPlacementAction(state, p.playerNumber)) {
        somePlayerCanAct = true;
        break;
      }

      // If player has stacks, check if they can move/capture
      // hasForcedEliminationAction returns true only if player has stacks AND cannot place AND cannot move
      // Since we already checked placement and it's false, if hasForcedEliminationAction is false,
      // that means they can move/capture
      if (playerHasStacks && !hasForcedEliminationAction(state, p.playerNumber)) {
        somePlayerCanAct = true;
        break;
      }

      // If playerHasStacks && hasForcedEliminationAction is true: player is trapped
      // If !playerHasStacks && cannot place: player has rings but nowhere to place = trapped
    }

    if (somePlayerCanAct) {
      return { isGameOver: false };
    }

    // All players with material are trapped - fall through to stalemate ladder below
  }

  // 5) Bare-board structural terminality & global stalemate.
  const noStacksLeft = state.board.stacks.size === 0;

  const anyRingsInHand = players.some((p) => p.ringsInHand > 0);
  let treatHandAsEliminated = false;
  let handCountsAsEliminated = false;

  if (anyRingsInHand) {
    // Check whether any player with rings in hand still has at least one
    // legal placement under the full no-dead-placement rule. If so, the
    // game is not yet structurally terminal: they may be able to re-enter
    // play once phases advance.
    const anyLegalPlacementForAnyPlayer = players.some(
      (p) => p.ringsInHand > 0 && hasAnyLegalPlacementOnBareBoard(state, p.playerNumber)
    );

    if (anyLegalPlacementForAnyPlayer) {
      return { isGameOver: false };
    }

    // Global stalemate on a bare board: some players hold rings in hand but
    // none of them has a legal placement. For tie-break purposes we
    // conceptually treat those rings as eliminated (hand → E) without
    // mutating the underlying GameState; hosts that care about persistent
    // elimination totals can mirror this via a separate helper.
    treatHandAsEliminated = true;
    handCountsAsEliminated = true;
  }

  // At this point the board has no stacks and either nobody holds rings in
  // hand or no legal placements exist for any player. Apply the stalemate
  // ladder.

  // First tie-breaker: territory spaces.
  const maxTerritory = Math.max(...players.map((p) => p.territorySpaces));
  const territoryLeaders = players.filter((p) => p.territorySpaces === maxTerritory);

  if (territoryLeaders.length === 1 && maxTerritory > 0) {
    return {
      isGameOver: true,
      winner: territoryLeaders[0].playerNumber,
      reason: 'territory_control',
      handCountsAsEliminated,
    };
  }

  // Second tie-breaker: eliminated rings, including rings remaining in hand
  // when we are in a global-stalemate case.
  const eliminationScores = players.map(
    (p) => p.eliminatedRings + (treatHandAsEliminated ? p.ringsInHand : 0)
  );
  const maxEliminated = Math.max(...eliminationScores);
  const eliminationLeaders = players.filter((_player, idx) => {
    return eliminationScores[idx] === maxEliminated;
  });

  if (eliminationLeaders.length === 1 && maxEliminated > 0) {
    return {
      isGameOver: true,
      winner: eliminationLeaders[0].playerNumber,
      reason: 'ring_elimination',
      handCountsAsEliminated,
    };
  }

  // Third tie-breaker: remaining markers on the board.
  const markerCountsByPlayer: { [player: number]: number } = {};
  for (const p of players) {
    markerCountsByPlayer[p.playerNumber] = 0;
  }
  for (const marker of state.board.markers.values()) {
    const owner = marker.player;
    if (markerCountsByPlayer[owner] !== undefined) {
      markerCountsByPlayer[owner] += 1;
    }
  }

  const markerCounts = players.map((p) => markerCountsByPlayer[p.playerNumber] ?? 0);
  const maxMarkers = Math.max(...markerCounts);
  const markerLeaders = players.filter(
    (p) => (markerCountsByPlayer[p.playerNumber] ?? 0) === maxMarkers
  );

  if (markerLeaders.length === 1 && maxMarkers > 0) {
    return {
      isGameOver: true,
      winner: markerLeaders[0].playerNumber,
      reason: 'last_player_standing',
      handCountsAsEliminated,
    };
  }

  // Final tie-breaker: last player to complete a valid turn action.
  const lastActor = getLastActor(state);
  if (lastActor !== undefined) {
    return {
      isGameOver: true,
      winner: lastActor,
      reason: 'last_player_standing',
      handCountsAsEliminated,
    };
  }

  // Safety fallback: in degenerate cases where no last actor can be
  // determined (e.g. malformed game state), mark the game as completed
  // without a specific winner.
  return {
    isGameOver: true,
    reason: 'game_completed',
    handCountsAsEliminated,
  };
}

/**
 * Check if the game has ended due to last player standing condition.
 *
 * @param state Current game state
 * @returns Victory result if last player standing condition met, null otherwise
 */
export function checkLastPlayerStanding(state: GameState): VictoryResult | null {
  const players = state.players;
  if (!players || players.length < 2) {
    return null;
  }

  // Count players who still have stacks on the board
  const playersWithStacks = new Set<number>();
  for (const stack of state.board.stacks.values()) {
    playersWithStacks.add(stack.controllingPlayer);
  }

  // Count players who still have rings (on board or in hand)
  const playersWithRings = players.filter(
    (p) => p.ringsInHand > 0 || playersWithStacks.has(p.playerNumber)
  );

  // If only one player has rings remaining, they win
  if (playersWithRings.length === 1) {
    return {
      isGameOver: true,
      winner: playersWithRings[0].playerNumber,
      reason: 'last_player_standing',
      handCountsAsEliminated: false,
    };
  }

  return null;
}

/**
 * Check if any player has reached the score threshold for victory.
 *
 * @param state Current game state
 * @returns Victory result if score threshold met, null otherwise
 */
export function checkScoreThreshold(state: GameState): VictoryResult | null {
  const players = state.players;
  if (!players || players.length === 0) {
    return null;
  }

  // Check ring elimination threshold
  const ringWinner = players.find((p) => p.eliminatedRings >= state.victoryThreshold);
  if (ringWinner) {
    return {
      isGameOver: true,
      winner: ringWinner.playerNumber,
      reason: 'ring_elimination',
      handCountsAsEliminated: false,
    };
  }

  // Check territory control threshold
  const territoryWinner = players.find((p) => p.territorySpaces >= state.territoryVictoryThreshold);
  if (territoryWinner) {
    return {
      isGameOver: true,
      winner: territoryWinner.playerNumber,
      reason: 'territory_control',
      handCountsAsEliminated: false,
    };
  }

  return null;
}

/**
 * Calculate a player's current score.
 *
 * @param state Current game state
 * @param playerNumber Player to calculate score for
 * @returns Player's score (eliminated rings count)
 */
export function getPlayerScore(state: GameState, playerNumber: number): number {
  const player = state.players.find((p) => p.playerNumber === playerNumber);
  if (!player) {
    return 0;
  }

  return player.eliminatedRings;
}

/**
 * Get the list of players still active in the game.
 *
 * @param state Current game state
 * @returns Array of players who still have rings in play
 */
export function getRemainingPlayers(state: GameState): Player[] {
  const players = state.players;
  if (!players || players.length === 0) {
    return [];
  }

  // Get players who have stacks on board
  const playersWithStacks = new Set<number>();
  for (const stack of state.board.stacks.values()) {
    playersWithStacks.add(stack.controllingPlayer);
  }

  // Return players who have either stacks on board or rings in hand
  return players.filter((p) => p.ringsInHand > 0 || playersWithStacks.has(p.playerNumber));
}

/**
 * Check if a specific player has been eliminated.
 *
 * @param state Current game state
 * @param playerNumber Player to check
 * @returns True if player has no rings remaining
 */
export function isPlayerEliminated(state: GameState, playerNumber: number): boolean {
  const player = state.players.find((p) => p.playerNumber === playerNumber);
  if (!player) {
    return true; // Unknown player is treated as eliminated
  }

  // Check if player has rings in hand
  if (player.ringsInHand > 0) {
    return false;
  }

  // Check if player has stacks on board
  for (const stack of state.board.stacks.values()) {
    if (stack.controllingPlayer === playerNumber) {
      return false;
    }
  }

  return true;
}

/**
 * Determine the last player to complete a valid turn action, used as the
 * final rung of the stalemate tie-break ladder. Preference order:
 *
 * 1. The actor of the last structured history entry, when available.
 * 2. The player of the last legacy moveHistory entry.
 * 3. The player immediately preceding currentPlayer in turn order.
 *
 * @param state Current game state
 * @returns Player number of last actor, or undefined if cannot be determined
 */
export function getLastActor(state: GameState): number | undefined {
  if (state.history && state.history.length > 0) {
    const lastEntry = state.history[state.history.length - 1];
    if (lastEntry && typeof lastEntry.actor === 'number') {
      return lastEntry.actor;
    }
  }

  if (state.moveHistory && state.moveHistory.length > 0) {
    const lastMove = state.moveHistory[state.moveHistory.length - 1];
    if (lastMove && typeof lastMove.player === 'number') {
      return lastMove.player;
    }
  }

  const players = state.players;
  if (!players || players.length === 0) {
    return undefined;
  }

  const currentIdx = players.findIndex((p) => p.playerNumber === state.currentPlayer);
  if (currentIdx === -1) {
    return players[0].playerNumber;
  }

  const lastIdx = (currentIdx - 1 + players.length) % players.length;
  return players[lastIdx].playerNumber;
}

/**
 * Evaluate victory with detailed standings and score breakdown.
 *
 * @param state Current game state
 * @param _ctx Optional context for detailed evaluation (reserved for future use)
 * @returns Detailed victory result with standings
 */
export function evaluateVictoryDetailed(
  state: GameState,
  _ctx?: VictoryEvaluationContext
): DetailedVictoryResult {
  const baseResult = evaluateVictory(state);
  const players = state.players;

  if (!players || players.length === 0) {
    return baseResult;
  }

  // Calculate marker counts
  const markerCounts = countMarkersByPlayer(state);

  // Build score breakdown
  const scores: DetailedVictoryResult['scores'] = {};
  for (const player of players) {
    scores[player.playerNumber] = {
      eliminatedRings: player.eliminatedRings,
      territorySpaces: player.territorySpaces,
      markerCount: markerCounts.get(player.playerNumber) ?? 0,
    };
  }

  // Build standings (sorted by victory criteria)
  const standings = [...players].sort((a, b) => {
    // First by territory
    if (a.territorySpaces !== b.territorySpaces) {
      return b.territorySpaces - a.territorySpaces;
    }
    // Then by eliminated rings
    if (a.eliminatedRings !== b.eliminatedRings) {
      return b.eliminatedRings - a.eliminatedRings;
    }
    // Then by marker count
    const aMarkers = markerCounts.get(a.playerNumber) ?? 0;
    const bMarkers = markerCounts.get(b.playerNumber) ?? 0;
    return bMarkers - aMarkers;
  });

  return {
    ...baseResult,
    standings,
    scores,
  };
}

/**
 * Get the count of eliminated rings for a specific player.
 *
 * @param state Current game state
 * @param playerNumber Player to get elimination count for
 * @returns Number of eliminated rings
 */
export function getEliminatedRingCount(state: GameState, playerNumber: number): number {
  const player = state.players.find((p) => p.playerNumber === playerNumber);
  return player?.eliminatedRings ?? 0;
}

/**
 * Get the count of territory spaces controlled by a specific player.
 *
 * @param state Current game state
 * @param playerNumber Player to get territory count for
 * @returns Number of territory spaces controlled
 */
export function getTerritoryCount(state: GameState, playerNumber: number): number {
  const player = state.players.find((p) => p.playerNumber === playerNumber);
  return player?.territorySpaces ?? 0;
}

/**
 * Get the count of markers on the board for a specific player.
 *
 * @param state Current game state
 * @param playerNumber Player to get marker count for
 * @returns Number of markers on board
 */
export function getMarkerCount(state: GameState, playerNumber: number): number {
  let count = 0;
  for (const marker of state.board.markers.values()) {
    if (marker.player === playerNumber) {
      count++;
    }
  }
  return count;
}

/**
 * Check if the victory threshold has been reached by any player.
 *
 * @param state Current game state
 * @returns True if any victory condition is met
 */
export function isVictoryThresholdReached(state: GameState): boolean {
  const result = evaluateVictory(state);
  return result.isGameOver;
}
