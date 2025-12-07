/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Board View Adapter Factory
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Factory functions that create board view adapters for sandbox modules.
 * Consolidates duplicate adapter construction patterns from:
 *
 * - sandboxPlacement.ts
 * - sandboxMovement.ts
 * - sandboxCaptures.ts
 *
 * These adapters wrap a BoardState to provide the interfaces expected by
 * the shared engine functions (MovementBoardView, CaptureBoardAdapters).
 *
 * Design Rationale:
 * - Shared engine interfaces capture board state in closure (isCollapsedSpace(pos))
 * - Sandbox-specific interfaces pass board as parameter (isCollapsedSpace(pos, board))
 * - This factory bridges the two by creating closures over a specific board state
 */

import type {
  BoardState,
  BoardType,
  Position,
  MovementBoardView,
  CaptureBoardAdapters as SharedCaptureBoardAdapters,
} from '../../shared/engine';
import { positionToString, BOARD_CONFIGS } from '../../shared/engine';
import { isValidPosition } from '../../shared/engine/validators/utils';

// Re-export the sandbox-specific interfaces for backward compatibility
export type { PlacementBoardView } from './sandboxPlacement';
export type { CaptureBoardAdapters } from './sandboxCaptures';

// ═══════════════════════════════════════════════════════════════════════════
// Unified Board View Type
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Unified board view interface that combines all board query methods.
 * This is a superset of MovementBoardView and CaptureBoardAdapters.
 */
export interface UnifiedBoardView {
  /** True if the position is on the board and addressable. */
  isValidPosition(pos: Position): boolean;

  /** True if this space is a collapsed territory space. */
  isCollapsedSpace(pos: Position): boolean;

  /**
   * Lightweight stack view at a position.
   * Returns undefined if no stack exists at the position.
   */
  getStackAt(pos: Position):
    | {
        controllingPlayer: number;
        capHeight: number;
        stackHeight: number;
      }
    | undefined;

  /** Marker ownership at a position. */
  getMarkerOwner(pos: Position): number | undefined;
}

// ═══════════════════════════════════════════════════════════════════════════
// Factory Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a unified board view adapter from a board state.
 *
 * This is the primary factory function that creates an adapter implementing
 * all board query methods. The returned adapter captures the board state
 * in closure, making it compatible with shared engine interfaces.
 *
 * @example
 * ```typescript
 * const view = createBoardView('square8', board);
 * const moves = enumerateSimpleMoveTargetsFromStack(boardType, from, player, view);
 * ```
 */
export function createBoardView(boardType: BoardType, board: BoardState): UnifiedBoardView {
  const config = BOARD_CONFIGS[boardType];
  return {
    isValidPosition: (pos: Position) => isValidPosition(pos, boardType, config.size),

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
      const key = positionToString(pos);
      const marker = board.markers.get(key);
      return marker?.player;
    },
  };
}

/**
 * Create a MovementBoardView adapter for movement enumeration.
 *
 * This adapter is compatible with the shared engine's movement functions:
 * - enumerateSimpleMoveTargetsFromStack
 * - hasAnyLegalMoveOrCaptureFromOnBoard
 *
 * @example
 * ```typescript
 * const view = createMovementBoardView('square8', board);
 * const targets = enumerateSimpleMoveTargetsFromStack(boardType, from, player, view);
 * ```
 */
export function createMovementBoardView(boardType: BoardType, board: BoardState): MovementBoardView {
  return createBoardView(boardType, board);
}

/**
 * Create a CaptureBoardAdapters instance for capture enumeration.
 *
 * This adapter is compatible with the shared engine's capture functions:
 * - enumerateCaptureMoves
 * - validateCaptureSegmentOnBoard
 *
 * @example
 * ```typescript
 * const adapters = createCaptureBoardAdapters('square8', board);
 * const moves = enumerateCaptureMoves(boardType, from, player, adapters, moveNumber);
 * ```
 */
export function createCaptureBoardAdapters(
  boardType: BoardType,
  board: BoardState
): SharedCaptureBoardAdapters {
  return createBoardView(boardType, board);
}

// ═══════════════════════════════════════════════════════════════════════════
// Sandbox-Specific Adapter Factories
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Sandbox-style board view that passes board state to each method.
 *
 * This interface maintains backward compatibility with existing sandbox code
 * that uses the `isCollapsedSpace(pos, board)` signature.
 */
export interface SandboxBoardView {
  isValidPosition(pos: Position): boolean;
  isCollapsedSpace(pos: Position, board: BoardState): boolean;
  getMarkerOwner(pos: Position, board: BoardState): number | undefined;
}

/**
 * Create a sandbox-style board view adapter.
 *
 * Unlike the closure-based adapters, this returns methods that take board
 * as a parameter. This is useful when the same adapter needs to be used
 * with multiple board states (e.g., hypothetical boards for validation).
 *
 * @example
 * ```typescript
 * const view = createSandboxBoardView('square8');
 *
 * // Can use with different boards
 * const collapsed1 = view.isCollapsedSpace(pos, board1);
 * const collapsed2 = view.isCollapsedSpace(pos, board2);
 * ```
 */
export function createSandboxBoardView(boardType: BoardType): SandboxBoardView {
  const config = BOARD_CONFIGS[boardType];
  return {
    isValidPosition: (pos: Position) => isValidPosition(pos, boardType, config.size),

    isCollapsedSpace: (pos: Position, board: BoardState) =>
      board.collapsedSpaces.has(positionToString(pos)),

    getMarkerOwner: (pos: Position, board: BoardState) => {
      const key = positionToString(pos);
      const marker = board.markers.get(key);
      return marker?.player;
    },
  };
}

/**
 * Convert a sandbox-style board view to a shared engine adapter by
 * capturing a specific board state in closure.
 *
 * This is useful when you have existing sandbox-style code and need to
 * call shared engine functions that expect closure-based adapters.
 *
 * @example
 * ```typescript
 * const sandboxView = createSandboxBoardView('square8');
 * const closureView = bindSandboxViewToBoard(sandboxView, board, 'square8');
 * const targets = enumerateSimpleMoveTargetsFromStack(boardType, from, player, closureView);
 * ```
 */
export function bindSandboxViewToBoard(
  sandboxView: SandboxBoardView,
  board: BoardState,
  boardType: BoardType
): MovementBoardView {
  return {
    isValidPosition: sandboxView.isValidPosition,
    isCollapsedSpace: (pos: Position) => sandboxView.isCollapsedSpace(pos, board),
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
    getMarkerOwner: (pos: Position) => sandboxView.getMarkerOwner(pos, board),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if a position has a stack controlled by the specified player.
 */
export function hasPlayerStackAt(
  board: BoardState,
  pos: Position,
  playerNumber: number
): boolean {
  const key = positionToString(pos);
  const stack = board.stacks.get(key);
  return stack !== undefined && stack.controllingPlayer === playerNumber;
}

/**
 * Get all stack positions for a player.
 */
export function getPlayerStackPositions(board: BoardState, playerNumber: number): Position[] {
  const positions: Position[] = [];
  const stackValues = Array.from(board.stacks.values());
  for (const stack of stackValues) {
    if (stack.controllingPlayer === playerNumber) {
      positions.push(stack.position);
    }
  }
  return positions;
}

/**
 * Count total rings for a player across all their stacks.
 */
export function countPlayerRings(board: BoardState, playerNumber: number): number {
  let count = 0;
  const stackValues = Array.from(board.stacks.values());
  for (const stack of stackValues) {
    if (stack.controllingPlayer === playerNumber) {
      count += stack.stackHeight;
    }
  }
  return count;
}
