import {
  BoardState,
  BoardType,
  BOARD_CONFIGS,
  Position,
  RingStack,
  positionToString,
} from '../../shared/types/game';
import {
  hasAnyLegalMoveOrCaptureFromOnBoard,
  MovementBoardView,
  calculateCapHeight,
} from '../../shared/engine/core';
import {
  validatePlacementOnBoard,
  PlacementContext,
} from '../../shared/engine/validators/PlacementValidator';

export interface PlacementBoardView {
  isValidPosition(pos: Position): boolean;
  isCollapsedSpace(pos: Position, board: BoardState): boolean;
  getMarkerOwner(pos: Position, board: BoardState): number | undefined;
}

/**
 * Create a hypothetical board with one or more rings placed at the given
 * position for the specified player. Used for no-dead-placement validation.
 */
export function createHypotheticalBoardWithPlacement(
  board: BoardState,
  position: Position,
  playerNumber: number,
  count: number = 1
): BoardState {
  const hypothetical: BoardState = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings },
  };

  const key = positionToString(position);

  // Placement semantics: a stack cannot coexist with a marker. When we
  // model a placement on a cell that currently has a marker, clear the
  // marker so the hypothetical board matches real placement behaviour.
  hypothetical.markers.delete(key);

  const existing = hypothetical.stacks.get(key);
  const effectiveCount = Math.max(1, count);

  if (existing && existing.rings.length > 0) {
    const addedRings = Array(effectiveCount).fill(playerNumber);
    const rings = [...addedRings, ...existing.rings];
    const newStack: RingStack = {
      ...existing,
      rings,
      stackHeight: rings.length,
      capHeight: calculateCapHeight(rings),
      controllingPlayer: playerNumber,
    };
    hypothetical.stacks.set(key, newStack);
  } else {
    const rings = Array(effectiveCount).fill(playerNumber);
    const newStack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: calculateCapHeight(rings),
      controllingPlayer: playerNumber,
    };
    hypothetical.stacks.set(key, newStack);
  }

  return hypothetical;
}

/**
 * Check whether a stack at `from` would have at least one legal move or
 * capture on the provided board. Analogue of RuleEngine.hasAnyLegalMoveOrCaptureFrom.
 */
export function hasAnyLegalMoveOrCaptureFrom(
  boardType: BoardType,
  board: BoardState,
  from: Position,
  playerNumber: number,
  view: PlacementBoardView
): boolean {
  const movementView: MovementBoardView = {
    isValidPosition: view.isValidPosition,
    isCollapsedSpace: (pos) => view.isCollapsedSpace(pos, board),
    getStackAt: (pos) => {
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos) => view.getMarkerOwner(pos, board),
  };

  return hasAnyLegalMoveOrCaptureFromOnBoard(boardType, from, playerNumber, movementView);
}

/**
 * Enumerate legal ring placement positions for the given player.
 *
 * When a {@link PlacementContext} is provided, this function delegates
 * legality checks (including board invariants, per-player capacity, and
 * no-dead-placement) to the shared {@link validatePlacementOnBoard}
 * helper so sandbox, backend, and shared GameEngine all agree on which
 * destinations admit at least one legal placement count.
 *
 * When no context is provided, it falls back to the legacy behaviour:
 * pure board-geometry + no-dead-placement without capacity checks. This
 * legacy path is retained for backward-compatible tests and diagnostic
 * tools that operate on BoardState alone.
 */
export function enumerateLegalRingPlacements(
  boardType: BoardType,
  board: BoardState,
  playerNumber: number,
  view: PlacementBoardView,
  ctx?: PlacementContext
): Position[] {
  const config = BOARD_CONFIGS[boardType];
  const results: Position[] = [];

  // === Canonical path: use shared validator when context is supplied ===
  if (ctx) {
    const baseCtx: PlacementContext = {
      ...ctx,
      boardType,
      player: playerNumber,
    };

    const testPosition = (pos: Position) => {
      if (!view.isValidPosition(pos)) return;
      const validation = validatePlacementOnBoard(board, pos, 1, baseCtx);
      if (validation.valid) {
        results.push(pos);
      }
    };

    if (boardType === 'hexagonal') {
      const radius = config.size - 1;
      for (let x = -radius; x <= radius; x++) {
        for (let y = -radius; y <= radius; y++) {
          const z = -x - y;
          const pos: Position = { x, y, z };
          testPosition(pos);
        }
      }
    } else {
      // square boards: 0..size-1 grid
      for (let x = 0; x < config.size; x++) {
        for (let y = 0; y < config.size; y++) {
          const pos: Position = { x, y };
          testPosition(pos);
        }
      }
    }

    return results;
  }

  // === Legacy path: board-geometry + no-dead-placement only ===
  if (boardType === 'hexagonal') {
    const radius = config.size - 1;
    for (let x = -radius; x <= radius; x++) {
      for (let y = -radius; y <= radius; y++) {
        const z = -x - y;
        const pos: Position = { x, y, z };

        if (!view.isValidPosition(pos)) continue;

        const key = positionToString(pos);

        // Do not allow placement on collapsed territory or on markers
        // (stacks+markers must never coexist).
        if (board.collapsedSpaces.has(key)) continue;
        if (board.markers.has(key)) continue;

        const hypothetical = createHypotheticalBoardWithPlacement(board, pos, playerNumber);

        if (hasAnyLegalMoveOrCaptureFrom(boardType, hypothetical, pos, playerNumber, view)) {
          results.push(pos);
        }
      }
    }
  } else {
    // square boards: 0..size-1 grid
    for (let x = 0; x < config.size; x++) {
      for (let y = 0; y < config.size; y++) {
        const pos: Position = { x, y };

        if (!view.isValidPosition(pos)) continue;

        const key = positionToString(pos);

        // Do not allow placement on collapsed territory or on markers to
        // match backend RuleEngine.validateRingPlacement semantics.
        if (board.collapsedSpaces.has(key)) continue;
        if (board.markers.has(key)) continue;

        const hypothetical = createHypotheticalBoardWithPlacement(board, pos, playerNumber);

        if (hasAnyLegalMoveOrCaptureFrom(boardType, hypothetical, pos, playerNumber, view)) {
          results.push(pos);
        }
      }
    }
  }

  return results;
}
