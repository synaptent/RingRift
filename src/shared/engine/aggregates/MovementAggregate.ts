/**
 * ═══════════════════════════════════════════════════════════════════════════
 * MovementAggregate - Consolidated Non-Capturing Movement Domain
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This aggregate consolidates all non-capturing movement validation, mutation,
 * and enumeration logic from:
 *
 * - movementLogic.ts → target enumeration, constraints
 * - movementApplication.ts → state application
 * - validators/MovementValidator.ts → validation
 * - mutators/MovementMutator.ts → mutation
 *
 * Rule Reference: Section 8 - Movement
 *
 * Key Rules:
 * - RR-CANON-R062: Minimum move distance equal to stack height
 * - RR-CANON-R063: Stack height determines max distance
 * - RR-CANON-R067: Must leave marker at origin (via SharedCore)
 * - Path must be unobstructed along straight line
 * - Cannot land on occupied position (except for captures - different aggregate)
 *
 * Design principles:
 * - Pure functions: No side effects, return new state
 * - Type safety: Full TypeScript typing
 * - Backward compatibility: Source files continue to export their functions
 */

import type { GameState, BoardState, Position, Move, BoardType } from '../../types/game';
import { positionToString } from '../../types/game';

import type { ValidationResult, MoveStackAction } from '../types';

import {
  getMovementDirectionsForBoardType,
  getPathPositions,
  calculateDistance,
  calculateCapHeight,
  applyMarkerEffectsAlongPathOnBoard,
  MovementBoardView,
  MarkerPathHelpers,
} from '../core';
import { enumerateSimpleMoveTargetsFromStack as enumerateSimpleMoveTargetsFromStackCore } from '../movementLogic';

import { isValidPosition } from '../validators/utils';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Represents a valid simple (non-capturing) move target.
 */
export interface SimpleMoveTarget {
  /** Origin position of the moving stack. */
  from: Position;
  /** Landing position for a simple (non-capturing) move. */
  to: Position;
}

/**
 * Adapter alias for the movement/capture reachability view used by the
 * shared core helpers. Callers typically construct this from a
 * GameState/BoardState using lightweight stack/marker projections.
 */
export type MovementBoardAdapters = MovementBoardView;

/**
 * Parameters for applying a simple (non-capturing) movement.
 */
export interface SimpleMovementParams {
  /** Origin of the moving stack. Must currently contain a stack. */
  from: Position;
  /** Landing position for the non-capturing move. */
  to: Position;
  /** Numeric player index expected to control the moving stack. */
  player: number;
  /**
   * When false, suppresses the default behaviour of leaving a departure
   * marker at `from`. This is mainly useful for hypothetical simulations
   * that want to reuse marker semantics but defer marker placement.
   *
   * Default: true.
   */
  leaveDepartureMarker?: boolean;
}

/**
 * Result of applying a simple movement mutation.
 */
export interface MovementApplicationOutcome {
  /**
   * Next GameState after applying the movement, including all stack,
   * marker, collapsed-space, and elimination bookkeeping.
   *
   * Implementations treat the input state as immutable and return a
   * shallow clone with cloned board/players/maps, matching the pattern used
   * by the existing shared mutators.
   */
  nextState: GameState;

  /**
   * Whether marker effects were applied during this move (flipping, collapsing).
   */
  markerEffectsApplied: boolean;

  /**
   * Per-player elimination deltas caused directly by this operation.
   * For simple non-capturing movement this is non-empty only when the move
   * lands on the mover's own marker and triggers top-ring elimination.
   * Empty object when no eliminations occurred.
   */
  eliminatedRingsByPlayer: { [player: number]: number };
}

/**
 * Result of validating a movement action.
 */
export type MovementValidationResult = ValidationResult;

/**
 * Result of applying a movement mutation.
 */
export type MovementMutationResult =
  | { success: true; newState: GameState }
  | { success: false; reason: string };

// ═══════════════════════════════════════════════════════════════════════════
// Internal Helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a MovementBoardView adapter from a BoardState.
 */
function createBoardView(board: BoardState, boardType: BoardType, size: number): MovementBoardView {
  return {
    isValidPosition: (pos: Position) => isValidPosition(pos, boardType, size),
    isCollapsedSpace: (pos: Position) => board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos: Position) => {
      const marker = board.markers.get(positionToString(pos));
      return marker?.player;
    },
  };
}

/**
 * Create marker path helpers for board mutation.
 */
function createMarkerHelpers(): MarkerPathHelpers {
  return {
    setMarker(position: Position, playerNumber: number, board: BoardState): void {
      const key = positionToString(position);
      board.markers.set(key, {
        player: playerNumber,
        position,
        type: 'regular',
      });
    },
    collapseMarker(position: Position, playerNumber: number, board: BoardState): void {
      const key = positionToString(position);
      board.markers.delete(key);
      board.collapsedSpaces.set(key, playerNumber);
    },
    flipMarker(position: Position, playerNumber: number, board: BoardState): void {
      const key = positionToString(position);
      const existing = board.markers.get(key);
      if (existing) {
        board.markers.set(key, {
          ...existing,
          player: playerNumber,
        });
      }
    },
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Validation Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Validate a MOVE_STACK action against full GameState.
 *
 * Rule Reference: Section 8 - Movement
 *
 * Checks:
 * - Phase must be 'movement'
 * - Must be the player's turn
 * - From/to positions must be valid
 * - Player must control the stack at 'from'
 * - Cannot move to collapsed space
 * - Must move in a valid direction (8 for square, 6 for hex)
 * - Distance must be >= stack height
 * - Path must be clear of stacks and collapsed spaces
 * - Cannot land on opponent marker or existing stack
 */
export function validateMovement(
  state: GameState,
  action: MoveStackAction
): MovementValidationResult {
  // 1. Phase Check
  if (state.currentPhase !== 'movement') {
    return { valid: false, reason: 'Not in movement phase', code: 'INVALID_PHASE' };
  }

  // 2. Turn Check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  // 3. Position Validity
  if (
    !isValidPosition(action.from, state.board.type, state.board.size) ||
    !isValidPosition(action.to, state.board.type, state.board.size)
  ) {
    return { valid: false, reason: 'Position off board', code: 'INVALID_POSITION' };
  }

  const fromKey = positionToString(action.from);
  const toKey = positionToString(action.to);

  // 4. Stack Ownership
  const stack = state.board.stacks.get(fromKey);
  if (!stack) {
    return { valid: false, reason: 'No stack at starting position', code: 'NO_STACK' };
  }
  if (stack.controllingPlayer !== action.playerId) {
    return { valid: false, reason: 'You do not control this stack', code: 'NOT_YOUR_STACK' };
  }

  // 5. Collapsed Space Check
  if (state.board.collapsedSpaces.has(toKey)) {
    return { valid: false, reason: 'Cannot move to collapsed space', code: 'COLLAPSED_SPACE' };
  }

  // 6. Direction Check
  const dx = action.to.x - action.from.x;
  const dy = action.to.y - action.from.y;
  const dz = (action.to.z || 0) - (action.from.z || 0);

  const directions = getMovementDirectionsForBoardType(state.board.type);
  let validDirection = false;

  // Normalize direction vector to check against canonical directions
  for (const dir of directions) {
    // Check if (dx, dy, dz) is a positive scalar multiple of dir
    // We need to find k > 0 such that dx = k*dir.x, dy = k*dir.y, dz = k*dir.z

    let k = 0;
    if (dir.x !== 0) k = dx / dir.x;
    else if (dir.y !== 0) k = dy / dir.y;
    else if (dir.z !== undefined && dir.z !== 0) k = dz / dir.z!;

    if (k > 0) {
      // Verify all components match
      const matchX = Math.abs(dx - k * dir.x) < 0.001;
      const matchY = Math.abs(dy - k * dir.y) < 0.001;
      const matchZ = dir.z !== undefined ? Math.abs(dz - k * dir.z) < 0.001 : true;

      if (matchX && matchY && matchZ) {
        validDirection = true;
        break;
      }
    }
  }

  if (!validDirection) {
    return { valid: false, reason: 'Invalid movement direction', code: 'INVALID_DIRECTION' };
  }

  // 7. Minimum Distance Check
  const distance = calculateDistance(state.board.type, action.from, action.to);
  if (distance < stack.stackHeight) {
    return {
      valid: false,
      reason: 'Move distance less than stack height',
      code: 'INSUFFICIENT_DISTANCE',
    };
  }

  // 8. Path Check (excluding start and end)
  const path = getPathPositions(action.from, action.to);
  // Remove start and end
  const innerPath = path.slice(1, -1);

  for (const pos of innerPath) {
    const key = positionToString(pos);

    // Cannot pass through collapsed spaces
    if (state.board.collapsedSpaces.has(key)) {
      return { valid: false, reason: 'Path blocked by collapsed space', code: 'PATH_BLOCKED' };
    }

    // Cannot pass through other stacks (unless capturing, which is a different action)
    const pathStack = state.board.stacks.get(key);
    if (pathStack && pathStack.stackHeight > 0) {
      return { valid: false, reason: 'Path blocked by stack', code: 'PATH_BLOCKED' };
    }
  }

  // 9. Landing Check
  const landingStack = state.board.stacks.get(toKey);
  const landingMarker = state.board.markers.get(toKey);

  // Can land on empty space
  if (!landingStack && !landingMarker) {
    return { valid: true };
  }

  // Can land on own marker (will be removed and top ring eliminated)
  if (landingMarker && landingMarker.player === action.playerId && !landingStack) {
    return { valid: true };
  }

  // Cannot land on opponent marker
  // Rule 8.2: "Landing on opponent markers or collapsed spaces remains illegal."
  if (landingMarker && landingMarker.player !== action.playerId && !landingStack) {
    return { valid: false, reason: 'Cannot land on opponent marker', code: 'INVALID_LANDING' };
  }

  // Cannot land on existing stack
  // Rule 8.1: "Cannot pass through other rings or stacks".
  // Rule 8.2: "Landing on ... empty or occupied by a single marker".
  if (landingStack) {
    return { valid: false, reason: 'Cannot land on existing stack', code: 'INVALID_LANDING' };
  }

  return { valid: true };
}

// ═══════════════════════════════════════════════════════════════════════════
// Enumeration Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Enumerate all legal simple (non-capturing) movement targets for the
 * stack controlled by `player` at `from` on a given board.
 *
 * This aggregate-level helper is a thin wrapper around the shared
 * {@link enumerateSimpleMoveTargetsFromStack} function from
 * movementLogic.ts so that all non-capture movement enumeration
 * semantics remain single-sourced.
 */
export function enumerateSimpleMoveTargetsFromStack(
  boardType: BoardType,
  from: Position,
  player: number,
  board: MovementBoardAdapters
): SimpleMoveTarget[] {
  return enumerateSimpleMoveTargetsFromStackCore(boardType, from, player, board);
}

/**
 * Enumerate all legal movement targets on a board as Position arrays.
 *
 * This is a convenience wrapper around enumerateSimpleMoveTargetsFromStack
 * that returns only the destination positions from a specific origin.
 */
export function enumerateMovementTargets(state: GameState, fromPosition: Position): Position[] {
  const boardView = createBoardView(state.board, state.board.type, state.board.size);
  const player = state.currentPlayer;

  const targets = enumerateSimpleMoveTargetsFromStack(
    state.board.type,
    fromPosition,
    player,
    boardView
  );

  return targets.map((t) => t.to);
}

/**
 * Enumerate all simple (non-capturing) moves for a player.
 *
 * Iterates over all stacks controlled by the player and returns Move
 * objects for each valid simple move.
 */
export function enumerateSimpleMovesForPlayer(state: GameState, player: number): Move[] {
  const moves: Move[] = [];
  const boardView = createBoardView(state.board, state.board.type, state.board.size);
  const moveNumber = state.moveHistory.length + 1;

  for (const stack of state.board.stacks.values()) {
    if (stack.controllingPlayer !== player || stack.stackHeight <= 0) {
      continue;
    }

    const targets = enumerateSimpleMoveTargetsFromStack(
      state.board.type,
      stack.position,
      player,
      boardView
    );

    for (const target of targets) {
      moves.push({
        id: `move-${positionToString(target.from)}-${positionToString(target.to)}-${moveNumber}`,
        type: 'move_stack',
        player,
        from: target.from,
        to: target.to,
        moveNumber,
        timestamp: new Date(),
        thinkTime: 0,
      });
    }
  }

  return moves;
}

/**
 * Enumerate all movement moves (alias matching design doc).
 */
export function enumerateAllMovementMoves(state: GameState, player: number): Move[] {
  return enumerateSimpleMovesForPlayer(state, player);
}

// ═══════════════════════════════════════════════════════════════════════════
// Mutation Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Apply a movement mutation to the state.
 *
 * This is the core mutation function that handles:
 * - Removing the stack from origin
 * - Placing a marker at origin
 * - Handling landing on own marker (eliminate top ring per rules 8.2, 8.3, 16.5.1)
 * - Moving the stack to destination
 *
 * Preconditions: The move has been validated by validateMovement.
 */
export function mutateMovement(state: GameState, action: MoveStackAction): GameState {
  // Deep copy state for immutability
  const newState = {
    ...state,
    board: {
      ...state.board,
      stacks: new Map(state.board.stacks),
      markers: new Map(state.board.markers),
      eliminatedRings: { ...state.board.eliminatedRings },
    },
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    totalRingsEliminated: number;
    lastMoveAt: Date;
    totalRingsInPlay: number;
  };

  const fromKey = positionToString(action.from);
  const toKey = positionToString(action.to);
  const stack = newState.board.stacks.get(fromKey);

  if (!stack) {
    throw new Error('MovementMutator: No stack at origin');
  }

  // 1. Remove stack from origin
  newState.board.stacks.delete(fromKey);

  // 2. Place marker at origin
  newState.board.markers.set(fromKey, {
    player: action.playerId,
    position: action.from,
    type: 'regular',
  });

  // 3. Handle landing
  const landingMarker = newState.board.markers.get(toKey);

  if (landingMarker) {
    // Landing on a marker
    if (landingMarker.player === action.playerId) {
      // Landing on OWN marker:
      // - Remove the marker
      newState.board.markers.delete(toKey);

      // - Eliminate the TOP ring of the moving stack (per rules 8.2, 8.3, 16.5.1)
      // TOP ring is rings[0] per actual codebase convention (consistent with calculateCapHeight)
      const topRingOwner = stack.rings[0];
      const newRings = stack.rings.slice(1); // Remove first element (the top)

      // Update elimination counts
      newState.totalRingsEliminated++;
      newState.board.eliminatedRings[topRingOwner] =
        (newState.board.eliminatedRings[topRingOwner] || 0) + 1;

      const player = newState.players.find((p) => p.playerNumber === topRingOwner);
      if (player) {
        player.eliminatedRings++;
      }

      // If stack becomes empty (was height 1), it's gone. Otherwise place it.
      if (newRings.length > 0) {
        newState.board.stacks.set(toKey, {
          position: action.to,
          rings: newRings,
          stackHeight: newRings.length,
          capHeight: calculateCapHeight(newRings),
          controllingPlayer: newRings[0], // New top ring is the controller
        });
      }
    } else {
      // Landing on OPPONENT marker:
      // This is generally not allowed for simple moves.
      // MovementValidator blocks this, but throw for safety.
      throw new Error(
        'MovementMutator: Landing on opponent marker is not supported in simple movement'
      );
    }
  } else {
    // Landing on empty space
    newState.board.stacks.set(toKey, {
      ...stack,
      position: action.to,
    });
  }

  // 4. Update timestamps
  newState.lastMoveAt = new Date();

  return newState;
}

/**
 * Apply a non-capturing stack movement with full marker effects.
 *
 * This is a higher-level mutation function that includes marker path
 * processing (flipping opponent markers, collapsing own markers along path).
 *
 * Responsibilities:
 *
 * - Leave a departure marker at `from` (unless `leaveDepartureMarker === false`)
 * - Process intermediate markers along the path:
 *   - Own markers collapse to territory
 *   - Opponent markers flip to mover's color
 * - Move or merge the stack at `from` into `to`
 * - Handle the "landing on own marker eliminates the bottom ring" rule
 * - Update per-player and global elimination counters
 *
 * Preconditions:
 *
 * - The move has already been validated using validateMovement or equivalent.
 * - The provided GameState must obey standard board invariants.
 *
 * Note: Hosts are responsible for updating currentPhase, currentPlayer,
 * decision-phase flags, and history; this helper focuses on board effects.
 */
export function applySimpleMovement(
  state: GameState,
  params: SimpleMovementParams
): MovementApplicationOutcome {
  // Deep copy state for immutability
  const newBoard: BoardState = {
    ...state.board,
    stacks: new Map(state.board.stacks),
    markers: new Map(state.board.markers),
    collapsedSpaces: new Map(state.board.collapsedSpaces),
    eliminatedRings: { ...state.board.eliminatedRings },
  };

  const newState = {
    ...state,
    board: newBoard,
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    totalRingsEliminated: number;
    lastMoveAt: Date;
    totalRingsInPlay: number;
  };

  const fromKey = positionToString(params.from);
  const toKey = positionToString(params.to);
  const stack = newState.board.stacks.get(fromKey);

  if (!stack) {
    throw new Error('applySimpleMovement: No stack at origin');
  }

  const eliminatedRingsByPlayer: { [player: number]: number } = {};
  let markerEffectsApplied = false;

  // 1. Remove stack from origin
  newState.board.stacks.delete(fromKey);

  // 2. BEFORE applying marker effects, capture landing marker state for own-marker elimination check
  // This is critical because applyMarkerEffectsAlongPathOnBoard DELETES the own marker at landing,
  // so we must capture the state before that happens. (Race condition fix)
  const landingMarkerBeforeEffects = newState.board.markers.get(toKey);
  const landedOnOwnMarker =
    landingMarkerBeforeEffects && landingMarkerBeforeEffects.player === params.player;

  // Apply marker effects along the path
  const leaveDepartureMarker = params.leaveDepartureMarker !== false;
  const helpers = createMarkerHelpers();

  // Track initial collapsed count to detect new collapses
  const initialCollapsedCount = newState.board.collapsedSpaces.size;

  applyMarkerEffectsAlongPathOnBoard(
    newState.board,
    params.from,
    params.to,
    params.player,
    helpers,
    { leaveDepartureMarker }
  );

  // Check if any marker effects were applied
  if (newState.board.collapsedSpaces.size > initialCollapsedCount) {
    markerEffectsApplied = true;
  }

  // 3. Handle landing on own marker (using pre-captured state)
  if (landedOnOwnMarker) {
    // Landing on OWN marker:
    // - Remove the marker (already done by applyMarkerEffectsAlongPathOnBoard)
    // - Eliminate the TOP ring of the moving stack (per rules section 8.2, 8.3, 16.5.1)

    // TOP ring is rings[0] per actual codebase convention (consistent with calculateCapHeight)
    const topRingOwner = stack.rings[0];
    const newRings = stack.rings.slice(1); // Remove first element (the top)

    // Update elimination counts
    newState.totalRingsEliminated = (newState.totalRingsEliminated || 0) + 1;
    newState.board.eliminatedRings[topRingOwner] =
      (newState.board.eliminatedRings[topRingOwner] || 0) + 1;

    eliminatedRingsByPlayer[topRingOwner] = (eliminatedRingsByPlayer[topRingOwner] || 0) + 1;

    const player = newState.players.find((p) => p.playerNumber === topRingOwner);
    if (player) {
      player.eliminatedRings++;
    }

    // If stack becomes empty (was height 1), it's gone. Otherwise place it.
    if (newRings.length > 0) {
      newState.board.stacks.set(toKey, {
        position: params.to,
        rings: newRings,
        stackHeight: newRings.length,
        capHeight: calculateCapHeight(newRings),
        controllingPlayer: newRings[0], // New top ring is the controller
      });
    }

    markerEffectsApplied = true;
  } else {
    // Check for landing on existing stack (merge). This is only reachable
    // for own-stack merges; movement validators/enumerators never allow
    // landing on opponent stacks.
    const existingStack = newState.board.stacks.get(toKey);
    if (existingStack) {
      // Merge: existing destination rings remain on top, moving stack goes below.
      const newRings = [...existingStack.rings, ...stack.rings];
      newState.board.stacks.set(toKey, {
        position: params.to,
        rings: newRings,
        stackHeight: newRings.length,
        capHeight: calculateCapHeight(newRings),
        controllingPlayer: newRings[0],
      });
    } else {
      // Landing on empty space
      newState.board.stacks.set(toKey, {
        ...stack,
        position: params.to,
      });
    }
  }

  // 4. Update timestamps
  newState.lastMoveAt = new Date();

  return {
    nextState: newState,
    markerEffectsApplied,
    eliminatedRingsByPlayer,
  };
}

/**
 * Apply movement and return a result type for easier error handling.
 *
 * This is a wrapper around applySimpleMovement that catches errors
 * and returns a discriminated union result.
 */
export function applyMovement(state: GameState, move: Move): MovementMutationResult {
  if (move.type !== 'move_stack' && move.type !== 'move_ring') {
    return {
      success: false,
      reason: `Expected 'move_stack' or 'move_ring' move, got '${move.type}'`,
    };
  }

  if (!move.from) {
    return {
      success: false,
      reason: 'Move.from is required for movement moves',
    };
  }

  try {
    const outcome = applySimpleMovement(state, {
      from: move.from,
      to: move.to,
      player: move.player,
    });

    return {
      success: true,
      newState: outcome.nextState,
    };
  } catch (error) {
    return {
      success: false,
      reason: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}
