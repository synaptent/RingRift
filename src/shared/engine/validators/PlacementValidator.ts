import { GameState, PlaceRingAction, SkipPlacementAction, ValidationResult } from '../types';
import { BoardState, BoardType, Position, BOARD_CONFIGS, positionToString } from '../../types/game';
import {
  hasAnyLegalMoveOrCaptureFromOnBoard,
  MovementBoardView,
  countRingsOnBoardForPlayer,
} from '../core';
import { isValidPosition } from './utils';

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

function computeRingsOnBoardForPlayer(board: BoardState, player: number): number {
  // Own-colour-based helper: count all rings of `player`'s colour that are
  // present in any stack on the board, regardless of which player currently
  // controls those stacks. This mirrors the canonical ringsPerPlayer
  // semantics (RR-CANON-R020 / compact §1.1).
  return countRingsOnBoardForPlayer(board, player);
}

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
  // Basic supply check – callers should usually gate on ringsInHand > 0,
  // but we defensively handle the degenerate case here as well.
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
 * Validate a SKIP_PLACEMENT action for the shared GameEngine. Semantics are
 * aligned with the backend RuleEngine.validateSkipPlacement helper:
 *
 * - Only legal during the ring_placement phase.
 * - Player must have rings in hand (otherwise there is nothing to skip).
 * - Player must control at least one stack on the board.
 * - At least one controlled stack must have a legal move or capture.
 */
export function validateSkipPlacement(
  state: GameState,
  action: SkipPlacementAction
): ValidationResult {
  // 1. Phase check – skip_placement is only meaningful during ring_placement.
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

  // 3. Supply gate: skip_placement is only valid when the player still has rings.
  if (player.ringsInHand <= 0) {
    return {
      valid: false,
      reason: 'Cannot skip placement with no rings in hand; use no_placement_action',
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
