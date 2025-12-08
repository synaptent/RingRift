import type { GameState, Move, Position, BoardState } from '../types/game';
import { positionToString } from '../types/game';
import { hasAnyLegalMoveOrCaptureFromOnBoard, MovementBoardView, calculateCapHeight } from './core';
import { isValidPosition } from './validators/utils';

/**
 * Shared helpers for placement application and skip-placement decisions.
 *
 * This module is intended to absorb the remaining duplication between:
 *
 * - Backend:
 *   - `GameEngine.applyMove` branches for `place_ring` and `skip_placement`,
 *   - `RuleEngine.validateSkipPlacement`, and
 * - Sandbox:
 *   - `ClientSandboxEngine.tryPlaceRings`,
 *   - `sandboxPlacement.enumerateLegalRingPlacements`, and
 *   - sandbox skip-placement gating in `ClientSandboxEngine`.
 *
 * Core legality checks and board-level placement semantics are already
 * centralised in:
 *
 * - [`validatePlacementOnBoard()`](src/shared/engine/validators/PlacementValidator.ts:76)
 * - [`applyPlacementOnBoard()`](src/shared/engine/mutators/PlacementMutator.ts:16)
 *
 * The helpers here sit one level above those primitives and:
 *
 * - operate directly on the canonical {@link GameState} / {@link Move} types
 *   used by backend and sandbox engines, and
 * - define a single source of truth for:
 *   - how `place_ring` moves update `ringsInHand` and board stacks, and
 *   - when `skip_placement` is a legal decision for the active player.
 *
 * Implementations will be introduced in later P0 tasks once hosts are
 * refactored to call into this module. For P0 Task #21, the functions are
 * specified and documented but left as design-time stubs.
 */

/**
 * Result of applying a canonical `place_ring` move.
 */
export interface PlacementApplicationOutcome {
  /**
   * Next GameState after applying the placement, including:
   *
   * - updated board stacks/markers in accordance with
   *   [`applyPlacementOnBoard()`](src/shared/engine/mutators/PlacementMutator.ts:16),
   * - decremented `ringsInHand` for the acting player, and
   * - updated `lastMoveAt` and `moveHistory` (when hosts choose to record
   *   the move here rather than in a higher-level GameEngine wrapper).
   */
  nextState: GameState;

  /**
   * Number of rings actually placed as part of this move. This will usually
   * equal `move.placementCount ?? 1` but is modelled explicitly so callers
   * can distinguish between:
   *
   * - requested count (from the wire payload), and
   * - effective count (clamped by supply / per-cell caps).
   */
  placementCount: number;

  /**
   * True when the placement landed on an existing stack (merge) rather than
   * creating a brand-new stack on an empty cell. This mirrors the
   * `placedOnStack` diagnostic flag sometimes recorded by backend engines.
   */
  placedOnStack: boolean;
}

/**
 * Apply a canonical `place_ring` move to the given GameState.
 *
 * Responsibilities (once implemented):
 *
 * - Interpret the {@link Move}:
 *   - require `move.type === 'place_ring'`,
 *   - use `move.to` as the destination position, and
 *   - treat `move.placementCount ?? 1` as the requested number of rings.
 * - Derive a placement context for the acting player from the GameState
 *   (ringsInHand, per-player cap from BOARD_CONFIGS, boardType).
 * - Delegate per-cell legality and no-dead-placement checks to
 *   [`validatePlacementOnBoard()`](src/shared/engine/validators/PlacementValidator.ts:76).
 * - Call [`applyPlacementOnBoard()`](src/shared/engine/mutators/PlacementMutator.ts:16)
 *   to update the BoardState, then project the board-level result back into
 *   GameState:
 *   - decrement `ringsInHand` for the acting player,
 *   - update board stacks and markers,
 *   - leave `totalRingsInPlay` unchanged to match current backend semantics
 *     (global pool initialised from BOARD_CONFIGS), and
 *   - refresh `lastMoveAt`.
 *
 * Error handling:
 *
 * - This helper assumes the move has already been validated by the shared
 *   placement validators or an equivalent rules layer.
 * - It is permitted to throw when structural invariants are violated
 *   (e.g. referenced player not found), but should not re-encode placement
 *   legality checks.
 */
export function applyPlacementMove(state: GameState, move: Move): PlacementApplicationOutcome {
  if (move.type !== 'place_ring') {
    throw new Error(`applyPlacementMove: Expected 'place_ring' move, got '${move.type}'`);
  }

  const playerObj = state.players.find((p) => p.playerNumber === move.player);
  if (!playerObj) {
    throw new Error('applyPlacementMove: Player not found');
  }

  const requestedCount = move.placementCount ?? 1;
  const effectiveCount = Math.min(requestedCount, playerObj.ringsInHand);

  if (effectiveCount <= 0) {
    // No rings to place - return state unchanged with zero placed
    return {
      nextState: state,
      placementCount: 0,
      placedOnStack: false,
    };
  }

  const posKey = positionToString(move.to);
  const existingStack = state.board.stacks.get(posKey);
  const isPlacingOnStack = !!(existingStack && existingStack.rings.length > 0);

  // Update board state with new placement
  const newBoard: BoardState = {
    ...state.board,
    stacks: new Map(state.board.stacks),
    markers: new Map(state.board.markers),
  };

  // Maintain stack/marker exclusivity: clear any marker at the destination
  newBoard.markers.delete(posKey);

  const placementRings = new Array(effectiveCount).fill(move.player);

  if (isPlacingOnStack && existingStack) {
    const rings = [...placementRings, ...existingStack.rings];
    newBoard.stacks.set(posKey, {
      ...existingStack,
      rings,
      stackHeight: rings.length,
      capHeight: calculateCapHeight(rings),
      controllingPlayer: rings[0],
    });
  } else {
    const rings = placementRings;
    newBoard.stacks.set(posKey, {
      position: move.to,
      rings,
      stackHeight: rings.length,
      capHeight: calculateCapHeight(rings),
      controllingPlayer: rings[0],
    });
  }

  // Update player's ringsInHand
  const players = state.players.map((p) => {
    if (p.playerNumber === move.player) {
      return { ...p, ringsInHand: p.ringsInHand - effectiveCount };
    }
    return { ...p };
  });

  const nextState: GameState = {
    ...state,
    board: newBoard,
    players,
    moveHistory: [...state.moveHistory],
    lastMoveAt: new Date(),
  };

  return {
    nextState,
    placementCount: effectiveCount,
    placedOnStack: isPlacingOnStack,
  };
}

/**
 * Result of evaluating whether the active player is allowed to perform a
 * `skip_placement` decision in the current state.
 *
 * This helper is intended to become the canonical definition of
 * skip-eligibility for both backend and sandbox engines, replacing bespoke
 * `hasValidPlacements` / `validateSkipPlacement` implementations.
 */
export interface SkipPlacementEligibilityResult {
  /**
   * True when a `skip_placement` move is legal for the specified player in
   * the current state; false otherwise.
   */
  canSkip: boolean;
  /**
   * Optional human-readable explanation for why skip is disallowed. Provided
   * only when `canSkip === false`.
   */
  reason?: string;
  /**
   * Optional machine-readable code describing the reason. Intended to align
   * with the error codes used by
   * [`validateSkipPlacement()`](src/shared/engine/validators/PlacementValidator.ts:295)
   * and the backend `RuleEngine.validateSkipPlacement` helper.
   */
  code?: string;
}

/**
 * Evaluate whether the specified player is allowed to perform an explicit
 * `skip_placement` move in the current GameState.
 *
 * Target semantics (subject to clarification in future rules updates):
 *
 * - Phase / turn:
 *   - Only meaningful during the `ring_placement` phase.
 *   - Only the active player may attempt to skip.
 * - Supply:
 *   - Player must have at least one ring in hand; when `ringsInHand === 0`
 *     placement is impossible and the engine should advance phases without
 *     requiring an explicit skip.
 * - Action availability:
 *   - Player must control at least one stack on the board.
 *   - At least one controlled stack must have a legal movement or overtaking
 *     capture according to
 *     [`hasAnyLegalMoveOrCaptureFromOnBoard()`](src/shared/engine/core.ts:367).
 * - Placement exhaustion (dead-placement alignment):
 *   - **Design assumption for this helper:** the player may skip placement
 *     *only when no legal placements remain* that satisfy the no-dead-
 *     placement rule implemented by
 *     [`validatePlacementOnBoard()`](src/shared/engine/validators/PlacementValidator.ts:76).
 *   - This means that if at least one legal placement exists, hosts should
 *     not surface `skip_placement` as an explicit decision; the player
 *     instead chooses between specific placements and starting movement/
 *     capture after any desired placement.
 *
 * NOTE:
 *
 * - Existing backend and sandbox implementations differ slightly in how they
 *   treat skip-eligibility (some variants allow skipping even when legal
 *   placements remain). This helper encodes the stricter "no legal
 *   placements remaining" interpretation to match the dead-placement rule,
 *   and the design doc
 *   [`P0_TASK_21_SHARED_HELPER_MODULES_DESIGN.md`](P0_TASK_21_SHARED_HELPER_MODULES_DESIGN.md:1)
 *   calls out this divergence as an open question.
 *
 * Implementation sketch (not yet wired in this stub):
 *
 * - If structural/phase/turn checks fail, return `canSkip = false` with an
 *   appropriate `code`.
 * - Use a lightweight placement enumerator based on
 *   [`validatePlacementOnBoard()`](src/shared/engine/validators/PlacementValidator.ts:76)
 *   to detect whether *any* legal placement exists for the player.
 * - Use a MovementBoardView + `hasAnyLegalMoveOrCaptureFromOnBoard` to
 *   detect whether any movement/capture from a controlled stack exists.
 */
export function evaluateSkipPlacementEligibility(
  state: GameState,
  player: number
): SkipPlacementEligibilityResult {
  // Phase check
  if (state.currentPhase !== 'ring_placement') {
    return {
      canSkip: false,
      reason: 'Not in ring placement phase',
      code: 'INVALID_PHASE',
    };
  }

  // Turn check
  if (player !== state.currentPlayer) {
    return {
      canSkip: false,
      reason: 'Not your turn',
      code: 'NOT_YOUR_TURN',
    };
  }

  const playerObj = state.players.find((p) => p.playerNumber === player);
  if (!playerObj) {
    return {
      canSkip: false,
      reason: 'Player not found',
      code: 'PLAYER_NOT_FOUND',
    };
  }

  // Supply gate: skip_placement is only valid when the player has rings in hand.
  if (playerObj.ringsInHand <= 0) {
    return {
      canSkip: false,
      reason: 'Cannot skip placement with no rings in hand; use no_placement_action',
      code: 'NO_RINGS_IN_HAND',
    };
  }

  // Build a movement board view for checking legal moves/captures
  const boardView: MovementBoardView = {
    isValidPosition: (pos: Position) => isValidPosition(pos, state.board.type, state.board.size),
    isCollapsedSpace: (pos: Position) => state.board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      const stack = state.board.stacks.get(key);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos: Position) => {
      const marker = state.board.markers.get(positionToString(pos));
      return marker?.player;
    },
  };

  let hasControlledStack = false;
  let hasLegalActionFromStack = false;

  for (const stack of state.board.stacks.values()) {
    if (stack.controllingPlayer !== player || stack.stackHeight <= 0) {
      continue;
    }
    hasControlledStack = true;

    if (hasAnyLegalMoveOrCaptureFromOnBoard(state.board.type, stack.position, player, boardView)) {
      hasLegalActionFromStack = true;
      break;
    }
  }

  if (!hasControlledStack) {
    return {
      canSkip: false,
      reason: 'Cannot skip placement when you control no stacks on the board',
      code: 'NO_CONTROLLED_STACKS',
    };
  }

  if (!hasLegalActionFromStack) {
    return {
      canSkip: false,
      reason: 'Cannot skip placement when no legal moves or captures are available',
      code: 'NO_LEGAL_ACTIONS',
    };
  }

  return { canSkip: true };
}
