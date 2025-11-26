/**
 * ═══════════════════════════════════════════════════════════════════════════
 * PlacementAggregate - Consolidated Ring Placement Domain
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This aggregate consolidates all ring placement validation, mutation, and
 * enumeration logic from:
 *
 * - validators/PlacementValidator.ts
 * - mutators/PlacementMutator.ts
 * - placementHelpers.ts
 *
 * Rule Reference: Section 4.1 - Ring Placement
 *
 * Design principles:
 * - Pure functions: No side effects, return new state
 * - Type safety: Full TypeScript typing
 * - Backward compatibility: Source files continue to export their functions
 */

import type { GameState, BoardState, Position, Move, BoardType } from '../../types/game';
import { BOARD_CONFIGS, positionToString } from '../../types/game';

import type { ValidationResult, PlaceRingAction, SkipPlacementAction } from '../types';

import {
  hasAnyLegalMoveOrCaptureFromOnBoard,
  MovementBoardView,
  countRingsOnBoardForPlayer,
  calculateCapHeight,
} from '../core';

import { isValidPosition } from '../validators/utils';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Context required to validate a prospective ring placement independently of
 * any particular engine/GameState host.
 */
export interface PlacementContext {
  /** Logical board type ('square8' | 'square19' | 'hexagonal'). */
  boardType: BoardType;
  /** Numeric player index performing the placement. */
  player: number;
  /** Rings the player currently has in hand (unplaced). */
  ringsInHand: number;
  /** Per-player cap for total rings that may ever be placed on the board. */
  ringsPerPlayerCap: number;
  /**
   * Optional optimisation: precomputed total number of this player's rings
   * currently on the board, counting all rings of their colour in every
   * stack (regardless of which player controls those stacks).
   * When omitted, the validator computes this from the board.
   */
  ringsOnBoard?: number;
  /**
   * Optional optimisation: precomputed global capacity remaining for the
   * player (min(ringsPerPlayerCap - ringsOnBoard, ringsInHand)). When
   * provided, the validator trusts this value and avoids recomputing it.
   */
  maxAvailableGlobal?: number;
}

/**
 * Result of validating a placement at a specific cell with an optional
 * requested placement count.
 */
export interface PlacementValidationResult {
  valid: boolean;
  /**
   * Maximum number of rings that may be placed at `to` in a single action
   * given the current board occupancy and global capacity constraints.
   * When validation fails due to count, this still reports the legal cap.
   */
  maxPlacementCount?: number;
  /** Optional human-readable explanation for invalid placements. */
  reason?: string;
  /** Optional machine-readable error code. */
  code?: string;
}

/**
 * Result of applying a canonical `place_ring` move.
 */
export interface PlacementApplicationOutcome {
  /**
   * Next GameState after applying the placement, including:
   *
   * - updated board stacks/markers in accordance with applyPlacementOnBoard,
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
  appliedCount: number;
}

/**
 * Result of evaluating whether the active player is allowed to perform a
 * `skip_placement` decision in the current state.
 */
export interface SkipPlacementEligibilityResult {
  /**
   * True when a `skip_placement` move is legal for the specified player in
   * the current state; false otherwise.
   */
  eligible: boolean;
  /**
   * Optional human-readable explanation for why skip is disallowed. Provided
   * only when `eligible === false`.
   */
  reason?: string;
  /**
   * Optional machine-readable code describing the reason.
   */
  code?: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Helpers
// ═══════════════════════════════════════════════════════════════════════════

function computeRingsOnBoardForPlayer(board: BoardState, player: number): number {
  return countRingsOnBoardForPlayer(board, player);
}

// ═══════════════════════════════════════════════════════════════════════════
// Validation Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Canonical, host-agnostic validator for ring placement on a concrete board.
 *
 * Responsibilities:
 * - Enforce board geometry / collapsed-space / marker-stack exclusivity.
 * - Enforce per-player ring cap and ringsInHand supply.
 * - Enforce single-ring vs multi-ring placement rules:
 *   - Existing stacks: at most 1 ring per action.
 *   - Empty cells: up to 3 rings per action (subject to capacity).
 * - Enforce no-dead-placement via hasAnyLegalMoveOrCaptureFromOnBoard.
 *
 * This helper does not know about phases or turn order; callers are
 * responsible for ensuring it is only used during ring_placement and for
 * the active player.
 */
export function validatePlacementOnBoard(
  board: BoardState,
  to: Position,
  requestedCount: number,
  ctx: PlacementContext
): PlacementValidationResult {
  // Basic supply check
  if (ctx.ringsInHand <= 0) {
    return {
      valid: false,
      reason: 'No rings in hand to place',
      code: 'INSUFFICIENT_RINGS',
      maxPlacementCount: 0,
    };
  }

  // Position validity
  if (!isValidPosition(to, ctx.boardType, board.size)) {
    return {
      valid: false,
      reason: 'Position off board',
      code: 'INVALID_POSITION',
      maxPlacementCount: 0,
    };
  }

  const posKey = positionToString(to);

  // Collapsed territory spaces are never legal for placement.
  if (board.collapsedSpaces.has(posKey)) {
    return {
      valid: false,
      reason: 'Cannot place on collapsed space',
      code: 'COLLAPSED_SPACE',
      maxPlacementCount: 0,
    };
  }

  // Stack/marker exclusivity: never allow placement onto an existing marker.
  if (board.markers.has(posKey)) {
    return {
      valid: false,
      reason: 'Cannot place on a marker',
      code: 'MARKER_BLOCKED',
      maxPlacementCount: 0,
    };
  }

  const existingStack = board.stacks.get(posKey);
  const isOccupied = !!(existingStack && existingStack.rings.length > 0);

  const ringsOnBoard =
    ctx.ringsOnBoard !== undefined
      ? ctx.ringsOnBoard
      : computeRingsOnBoardForPlayer(board, ctx.player);

  const remainingByCap = ctx.ringsPerPlayerCap - ringsOnBoard;
  const remainingBySupply = ctx.ringsInHand;
  const maxAvailableGlobal =
    ctx.maxAvailableGlobal !== undefined
      ? ctx.maxAvailableGlobal
      : Math.min(remainingByCap, remainingBySupply);

  if (maxAvailableGlobal <= 0) {
    return {
      valid: false,
      reason: 'No rings available to place (cap reached or no rings in hand)',
      code: 'NO_RINGS_AVAILABLE',
      maxPlacementCount: 0,
    };
  }

  const perCellCap = isOccupied ? 1 : 3;
  const maxPlacementCount = Math.min(perCellCap, maxAvailableGlobal);

  if (maxPlacementCount <= 0) {
    return {
      valid: false,
      reason: 'No rings available to place at this position',
      code: 'NO_RINGS_AVAILABLE',
      maxPlacementCount: 0,
    };
  }

  const count = requestedCount;
  if (count < 1 || count > maxPlacementCount) {
    return {
      valid: false,
      reason: isOccupied
        ? 'Can only place 1 ring on an existing stack'
        : 'Must place between 1 and 3 rings on an empty space',
      code: 'INVALID_COUNT',
      maxPlacementCount,
    };
  }

  // --- No-dead-placement: resulting stack must have at least one legal move/capture.

  const hypotheticalStack = existingStack
    ? {
        controllingPlayer: ctx.player,
        stackHeight: existingStack.stackHeight + count,
        capHeight:
          existingStack.controllingPlayer === ctx.player ? existingStack.capHeight + count : count,
      }
    : {
        controllingPlayer: ctx.player,
        stackHeight: count,
        capHeight: count,
      };

  const movementView: MovementBoardView = {
    isValidPosition: (pos: Position) => isValidPosition(pos, ctx.boardType, board.size),
    isCollapsedSpace: (pos: Position) => board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      if (key === posKey) {
        return hypotheticalStack;
      }
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

  const hasLegalMove = hasAnyLegalMoveOrCaptureFromOnBoard(
    ctx.boardType,
    to,
    ctx.player,
    movementView
  );

  if (!hasLegalMove) {
    return {
      valid: false,
      reason: 'Placement would result in a stack with no legal moves or captures',
      code: 'NO_LEGAL_MOVES',
      maxPlacementCount,
    };
  }

  return { valid: true, maxPlacementCount };
}

/**
 * GameEngine-facing placement validator. This is a thin wrapper around the
 * board-level validatePlacementOnBoard helper that adds phase/turn checks
 * and maps errors into the shared ValidationResult shape.
 */
export function validatePlacement(state: GameState, action: PlaceRingAction): ValidationResult {
  // 1. Phase check
  if (state.currentPhase !== 'ring_placement') {
    return { valid: false, reason: 'Not in ring placement phase', code: 'INVALID_PHASE' };
  }

  // 2. Turn check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  const player = state.players.find((p) => p.playerNumber === action.playerId);
  if (!player) {
    return { valid: false, reason: 'Player not found', code: 'PLAYER_NOT_FOUND' };
  }

  if (action.count <= 0) {
    return { valid: false, reason: 'Must place at least 1 ring', code: 'INVALID_COUNT' };
  }

  if (player.ringsInHand <= 0 || player.ringsInHand < action.count) {
    return {
      valid: false,
      reason: 'Not enough rings in hand',
      code: 'INSUFFICIENT_RINGS',
    };
  }

  const boardType = state.board.type;
  const boardConfig = BOARD_CONFIGS[boardType];

  const ctx: PlacementContext = {
    boardType,
    player: action.playerId,
    ringsInHand: player.ringsInHand,
    ringsPerPlayerCap: boardConfig.ringsPerPlayer,
  };

  const result = validatePlacementOnBoard(state.board, action.position, action.count, ctx);

  if (!result.valid) {
    return {
      valid: false,
      reason: result.reason ?? 'Invalid placement',
      code: result.code ?? 'INVALID_PLACEMENT',
    };
  }

  return { valid: true };
}

/**
 * Validate a SKIP_PLACEMENT action for the shared GameEngine.
 */
export function validateSkipPlacement(
  state: GameState,
  action: SkipPlacementAction
): ValidationResult {
  // 1. Phase check
  if (state.currentPhase !== 'ring_placement') {
    return { valid: false, reason: 'Not in ring placement phase', code: 'INVALID_PHASE' };
  }

  // 2. Turn check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  const player = state.players.find((p) => p.playerNumber === action.playerId);
  if (!player) {
    return { valid: false, reason: 'Player not found', code: 'PLAYER_NOT_FOUND' };
  }

  // 3. Player must have rings in hand for skip_placement to be meaningful
  if (player.ringsInHand <= 0) {
    return {
      valid: false,
      reason: 'Cannot skip placement when you have no rings in hand',
      code: 'NO_RINGS_IN_HAND',
    };
  }

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
    if (stack.controllingPlayer !== action.playerId || stack.stackHeight <= 0) {
      continue;
    }
    hasControlledStack = true;

    if (
      hasAnyLegalMoveOrCaptureFromOnBoard(
        state.board.type,
        stack.position,
        action.playerId,
        boardView
      )
    ) {
      hasLegalActionFromStack = true;
      break;
    }
  }

  if (!hasControlledStack) {
    return {
      valid: false,
      reason: 'Cannot skip placement when you control no stacks on the board',
      code: 'NO_CONTROLLED_STACKS',
    };
  }

  if (!hasLegalActionFromStack) {
    return {
      valid: false,
      reason: 'Cannot skip placement when no legal moves or captures are available',
      code: 'NO_LEGAL_ACTIONS',
    };
  }

  return { valid: true };
}

// ═══════════════════════════════════════════════════════════════════════════
// Query / Enumeration Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Enumerate all legal placement positions for a player.
 *
 * This function iterates over all valid board positions and returns those
 * where validatePlacementOnBoard reports a valid placement with count=1.
 */
export function enumeratePlacementPositions(state: GameState, player: number): Position[] {
  const playerObj = state.players.find((p) => p.playerNumber === player);
  if (!playerObj) return [];

  if (playerObj.ringsInHand <= 0) return [];

  const boardType = state.board.type;
  const boardConfig = BOARD_CONFIGS[boardType];
  const board = state.board;

  const ctx: PlacementContext = {
    boardType,
    player,
    ringsInHand: playerObj.ringsInHand,
    ringsPerPlayerCap: boardConfig.ringsPerPlayer,
    ringsOnBoard: computeRingsOnBoardForPlayer(board, player),
  };

  // Pre-compute maxAvailableGlobal for efficiency
  const remainingByCap = ctx.ringsPerPlayerCap - (ctx.ringsOnBoard ?? 0);
  const maxAvailableGlobal = Math.min(remainingByCap, ctx.ringsInHand);
  ctx.maxAvailableGlobal = maxAvailableGlobal;

  if (maxAvailableGlobal <= 0) return [];

  const legalPositions: Position[] = [];
  const size = board.size;

  // Different iteration for hexagonal vs square boards
  if (boardType === 'hexagonal') {
    // Hexagonal board uses axial coordinates with radius = size
    const radius = size;
    for (let q = -radius; q <= radius; q++) {
      const r1 = Math.max(-radius, -q - radius);
      const r2 = Math.min(radius, -q + radius);
      for (let r = r1; r <= r2; r++) {
        const pos: Position = { x: q, y: r, z: -q - r };
        const result = validatePlacementOnBoard(board, pos, 1, ctx);
        if (result.valid) {
          legalPositions.push(pos);
        }
      }
    }
  } else {
    // Square boards use simple x,y grid
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        const pos: Position = { x, y };
        const result = validatePlacementOnBoard(board, pos, 1, ctx);
        if (result.valid) {
          legalPositions.push(pos);
        }
      }
    }
  }

  return legalPositions;
}

/**
 * Evaluate whether skip placement is legal for the specified player.
 */
export function evaluateSkipPlacementEligibility(
  state: GameState,
  player: number
): SkipPlacementEligibilityResult {
  // Phase check
  if (state.currentPhase !== 'ring_placement') {
    return {
      eligible: false,
      reason: 'Not in ring placement phase',
      code: 'INVALID_PHASE',
    };
  }

  // Turn check
  if (player !== state.currentPlayer) {
    return {
      eligible: false,
      reason: 'Not your turn',
      code: 'NOT_YOUR_TURN',
    };
  }

  const playerObj = state.players.find((p) => p.playerNumber === player);
  if (!playerObj) {
    return {
      eligible: false,
      reason: 'Player not found',
      code: 'PLAYER_NOT_FOUND',
    };
  }

  // Player must have rings in hand
  if (playerObj.ringsInHand <= 0) {
    return {
      eligible: false,
      reason: 'Cannot skip placement when you have no rings in hand',
      code: 'NO_RINGS_IN_HAND',
    };
  }

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
      eligible: false,
      reason: 'Cannot skip placement when you control no stacks on the board',
      code: 'NO_CONTROLLED_STACKS',
    };
  }

  if (!hasLegalActionFromStack) {
    return {
      eligible: false,
      reason: 'Cannot skip placement when no legal moves or captures are available',
      code: 'NO_LEGAL_ACTIONS',
    };
  }

  return { eligible: true };
}

// ═══════════════════════════════════════════════════════════════════════════
// Mutation Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Apply placement mutation to board only.
 *
 * Canonical board-level placement mutator used by both the shared GameEngine
 * and client/server hosts. This applies a placement for `playerId` at
 * `position` and returns a new BoardState with:
 *
 * - stack/marker exclusivity enforced (any marker at the destination is
 *   cleared before writing the stack),
 * - new rings added on top of the stack (front of the `rings` array),
 * - capHeight / stackHeight / controllingPlayer recomputed from the
 *   resulting ring sequence.
 */
export function applyPlacementOnBoard(
  board: BoardState,
  position: Position,
  playerId: number,
  count: number
): BoardState {
  const effectiveCount = Math.max(1, count);

  const newBoard: BoardState = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
  };

  const posKey = positionToString(position);

  // Maintain stack/marker exclusivity
  newBoard.markers.delete(posKey);

  const existingStack = newBoard.stacks.get(posKey);
  const placementRings = new Array(effectiveCount).fill(playerId);

  if (existingStack && existingStack.rings.length > 0) {
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
      position,
      rings,
      stackHeight: rings.length,
      capHeight: calculateCapHeight(rings),
      controllingPlayer: rings[0],
    });
  }

  return newBoard;
}

/**
 * Apply placement mutation to full GameState.
 *
 * GameEngine-level placement mutator. This is a thin wrapper around
 * applyPlacementOnBoard that is responsible for updating player
 * ringsInHand and bookkeeping fields on GameState.
 */
export function mutatePlacement(state: GameState, action: PlaceRingAction): GameState {
  const players = state.players.map((p) => ({ ...p }));
  const player = players.find((p) => p.playerNumber === action.playerId);

  if (!player) {
    throw new Error('PlacementMutator: Player not found');
  }

  const toSpend = Math.min(action.count, player.ringsInHand);
  if (toSpend <= 0) {
    return state;
  }
  player.ringsInHand -= toSpend;

  const updatedBoard = applyPlacementOnBoard(
    state.board as BoardState,
    action.position,
    action.playerId,
    toSpend
  );

  const newState: GameState & { totalRingsInPlay: number; lastMoveAt: Date } = {
    ...(state as GameState & { totalRingsInPlay: number; lastMoveAt: Date }),
    board: updatedBoard,
    players,
    moveHistory: [...state.moveHistory],
    lastMoveAt: new Date(),
  };

  return newState;
}

/**
 * Apply placement via Move representation.
 *
 * This helper assumes the move has already been validated by the shared
 * placement validators or an equivalent rules layer.
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
    // No rings to place - return state unchanged with zero applied
    return {
      nextState: state,
      appliedCount: 0,
    };
  }

  // Create action for the mutatePlacement call
  const action: PlaceRingAction = {
    type: 'PLACE_RING',
    playerId: move.player,
    position: move.to,
    count: effectiveCount,
  };

  const nextState = mutatePlacement(state, action);

  return {
    nextState,
    appliedCount: effectiveCount,
  };
}
