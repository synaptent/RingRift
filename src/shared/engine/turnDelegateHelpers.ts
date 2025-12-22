import type { GameState, BoardType, Position } from '../types/game';
import { BOARD_CONFIGS, positionToString } from '../types/game';
import type { PerTurnState, TurnLogicDelegates } from './turnLogic';
import { countRingsOnBoardForPlayer } from './core';
import { validatePlacementOnBoard } from './validators/PlacementValidator';
import { playerHasAnyRings } from './globalActions';
import { enumerateSimpleMovesForPlayer } from './aggregates/MovementAggregate';
import { enumerateAllCaptureMoves } from './aggregates/CaptureAggregate';
import { getEffectiveRingsPerPlayer } from './rulesConfig';

/**
 * Shared helpers and factory for {@link TurnLogicDelegates} used by the
 * phase/turn sequencer in [`turnLogic.ts`](src/shared/engine/turnLogic.ts:1).
 *
 * This module provides the **canonical** way to answer the three action-availability
 * questions used by the sequencer:
 *
 * - "Does this player have any legal placement?" → {@link hasAnyPlacementForPlayer}
 * - "Does this player have any legal non-capturing movement?" → {@link hasAnyMovementForPlayer}
 * - "Does this player have any legal overtaking capture?" → {@link hasAnyCaptureForPlayer}
 *
 * It also provides a factory {@link createDefaultTurnLogicDelegates} that backend
 * and sandbox hosts can use to build {@link TurnLogicDelegates} instances that
 * share the same underlying reachability semantics.
 *
 * Implementation notes:
 * - Placement: Uses validatePlacementOnBoard for no-dead-placement rule
 * - Movement: Delegates to enumerateSimpleMovesForPlayer from MovementAggregate
 * - Capture: Delegates to enumerateAllCaptureMoves from CaptureAggregate
 * - All three respect mustMoveFromStackKey constraints from PerTurnState
 */

/**
 * Canonical "has any placement?" predicate for the shared turn/phase
 * sequencer.
 *
 * Target semantics (to be implemented in a future task):
 *
 * - Only placements that satisfy the no-dead-placement rule encoded by
 *   [`validatePlacementOnBoard`](src/shared/engine/validators/PlacementValidator.ts:76)
 *   are counted as "legal".
 * - Ring supply and per-player caps (BOARD_CONFIGS.ringsPerPlayer) are
 *   honoured.
 * - The current phase is **not** considered here; hosts remain responsible
 *   for calling this only when placement is relevant (usually from
 *   `'ring_placement'` or at turn start).
 *
 * This helper is pure and must not mutate `state`.
 */
export function hasAnyPlacementForPlayer(state: GameState, player: number): boolean {
  const playerObj = state.players.find((p) => p.playerNumber === player);
  if (!playerObj || playerObj.ringsInHand <= 0) {
    return false;
  }

  const board = state.board;
  const boardConfig = BOARD_CONFIGS[board.type as BoardType];

  if (!boardConfig) {
    return false;
  }

  // Global own-colour supply cap: if the player has no remaining capacity
  // under ringsPerPlayer, no placements are possible regardless of board
  // geometry.
  const ringsOnBoard = countRingsOnBoardForPlayer(board, player);
  const ringsPerPlayerCap = getEffectiveRingsPerPlayer(board.type as BoardType, state.rulesOptions);
  const remainingByCap = ringsPerPlayerCap - ringsOnBoard;
  const remainingBySupply = playerObj.ringsInHand;
  const maxAvailableGlobal = Math.min(remainingByCap, remainingBySupply);

  if (maxAvailableGlobal <= 0) {
    return false;
  }

  const baseContext = {
    boardType: board.type as BoardType,
    player,
    ringsInHand: playerObj.ringsInHand,
    ringsPerPlayerCap,
    ringsOnBoard,
    maxAvailableGlobal,
  };

  const hasPlacementAt = (pos: Position): boolean => {
    const result = validatePlacementOnBoard(board, pos, 1, baseContext);
    return result.valid;
  };

  if (board.type === 'square8' || board.type === 'square19') {
    const size = board.size;
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const pos: Position = { x, y };
        if (hasPlacementAt(pos)) {
          return true;
        }
      }
    }
    return false;
  }

  if (board.type === 'hexagonal' || board.type === 'hex8') {
    // For hex boards (hexagonal and hex8), iterate over all cube coordinates
    // within the board radius. validatePlacementOnBoard will discard off-board
    // positions. boardSize is the bounding box (2*radius + 1).
    const radius = (board.size - 1) / 2;
    for (let q = -radius; q <= radius; q += 1) {
      for (let r = -radius; r <= radius; r += 1) {
        const s = -q - r;
        const pos: Position = { x: q, y: r, z: s };
        if (hasPlacementAt(pos)) {
          return true;
        }
      }
    }
    return false;
  }

  // Unknown board type: conservatively report no placements.
  return false;
}

/**
 * Canonical "has any non-capturing movement?" predicate for the shared
 * turn/phase sequencer.
 *
 * Semantics:
 *
 * - Returns true if at least one legal **non-capturing** stack move exists
 *   for `player` in `state`, respecting:
 *   - board geometry and collapsed spaces,
 *   - stack ownership and height rules, and
 *   - any per-turn constraints encoded in {@link PerTurnState}
 *     (e.g. must-move-from-stack requirements).
 * - Capture opportunities are **not** considered here; those are handled by
 *   {@link hasAnyCaptureForPlayer}.
 *
 * Implementation:
 * - Delegates to `enumerateSimpleMovesForPlayer` from MovementAggregate
 * - Filters by mustMoveFromStackKey constraint if present in PerTurnState
 */
export function hasAnyMovementForPlayer(
  state: GameState,
  player: number,
  turn: PerTurnState
): boolean {
  // Get all simple (non-capturing) moves for the player
  const allMoves = enumerateSimpleMovesForPlayer(state, player);

  if (allMoves.length === 0) {
    return false;
  }

  // If there's a must-move-from constraint, filter to only those moves
  if (turn.mustMoveFromStackKey) {
    const mustMoveFrom = turn.mustMoveFromStackKey;
    const constrainedMoves = allMoves.filter((move) => {
      if (!move.from) return false;
      return positionToString(move.from) === mustMoveFrom;
    });
    return constrainedMoves.length > 0;
  }

  return true;
}

/**
 * Canonical "has any overtaking capture?" predicate for the shared
 * turn/phase sequencer.
 *
 * Semantics:
 *
 * - Returns true if at least one legal overtaking capture (initial or
 *   continuation) exists for `player` in `state`, respecting any per-turn
 *   constraints encoded in {@link PerTurnState}.
 * - Builds on the same capture geometry used everywhere else via
 *   `enumerateAllCaptureMoves` from CaptureAggregate.
 *
 * Implementation:
 * - Delegates to `enumerateAllCaptureMoves` from CaptureAggregate
 * - Filters by mustMoveFromStackKey constraint if present in PerTurnState
 *
 * This helper is pure and does not mutate `state`.
 */
export function hasAnyCaptureForPlayer(
  state: GameState,
  player: number,
  turn: PerTurnState
): boolean {
  // Get all capture moves for the player
  const allCaptures = enumerateAllCaptureMoves(state, player);

  if (allCaptures.length === 0) {
    return false;
  }

  // If there's a must-move-from constraint, filter to only captures from that stack
  if (turn.mustMoveFromStackKey) {
    const mustMoveFrom = turn.mustMoveFromStackKey;
    const constrainedCaptures = allCaptures.filter((move) => {
      if (!move.from) return false;
      return positionToString(move.from) === mustMoveFrom;
    });
    return constrainedCaptures.length > 0;
  }

  return true;
}

/**
 * Configuration required to build a {@link TurnLogicDelegates} instance using
 * the shared action-availability helpers defined in this module.
 *
 * Hosts remain responsible for:
 *
 * - how player numbers advance in seat order, and
 * - how forced-elimination is applied (which stack to eliminate from,
 *   whether to allow elimination-from-hand, when to run victory checks, etc.).
 */
export interface DefaultTurnDelegatesConfig {
  /**
   * Host-specific implementation of the delegate used by
   * [`advanceTurnAndPhase`](src/shared/engine/turnLogic.ts:135) to rotate
   * the active player number in seat order.
   */
  getNextPlayerNumber: TurnLogicDelegates['getNextPlayerNumber'];

  /**
   * Host-specific implementation of forced elimination for a blocked player.
   * This is threaded through without modification so that:
   *
   * - the backend can continue using its existing elimination logic wrapped
   *   around `GameEngine.eliminatePlayerRingOrCap`, and
   * - the sandbox can keep its UI-/history-specific elimination behaviour.
   */
  applyForcedElimination: TurnLogicDelegates['applyForcedElimination'];
}

/**
 * Build a {@link TurnLogicDelegates} instance that:
 *
 * - Answers `hasAnyPlacement`, `hasAnyMovement`, and `hasAnyCapture` via the
 *   shared predicates declared in this module, and
 * - Defers `getNextPlayerNumber` and `applyForcedElimination` to host
 *   implementations supplied via {@link DefaultTurnDelegatesConfig}.
 *
 * This factory is intended as the long-term entry point for both:
 *
 * - the backend TurnEngine (wrapping its existing next-player / forced-
 *   elimination semantics), and
 * - the sandbox turn engine, once both are refactored to drive the shared
 *   [`advanceTurnAndPhase`](src/shared/engine/turnLogic.ts:135) helper.
 */
export function createDefaultTurnLogicDelegates(
  config: DefaultTurnDelegatesConfig
): TurnLogicDelegates {
  return {
    getPlayerStacks: (state: GameState, player: number) => {
      // Minimal adapter returning just the fields required by the sequencer.
      const stacks: Array<{ position: Position; stackHeight: number }> = [];
      for (const stack of state.board.stacks.values()) {
        if (stack.controllingPlayer !== player) continue;
        stacks.push({ position: stack.position, stackHeight: stack.stackHeight });
      }
      return stacks;
    },
    hasAnyPlacement: (state: GameState, player: number) => hasAnyPlacementForPlayer(state, player),
    hasAnyMovement: (state: GameState, player: number, turn: PerTurnState) =>
      hasAnyMovementForPlayer(state, player, turn),
    hasAnyCapture: (state: GameState, player: number, turn: PerTurnState) =>
      hasAnyCaptureForPlayer(state, player, turn),
    applyForcedElimination: config.applyForcedElimination,
    getNextPlayerNumber: config.getNextPlayerNumber,
    playerHasAnyRings: (state: GameState, player: number) => playerHasAnyRings(state, player),
  };
}
