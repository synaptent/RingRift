import { BoardState, Position, RingStack } from '../../../shared/types/game';
import { BoardManager } from '../BoardManager';
import { calculateCapHeight, getMovementDirectionsForBoardType } from '../../../shared/engine/core';

/**
 * Create a hypothetical board state with a single-ring placement at the
 * specified position for the given player. This is a direct extraction of
 * the RuleEngine.createHypotheticalBoardWithPlacement behaviour prior to
 * modularization so existing tests remain valid.
 */
export function createHypotheticalBoardWithPlacement(
  board: BoardState,
  position: Position,
  player: number,
  count: number = 1
): BoardState {
  const hypotheticalBoard: BoardState = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings }
  };

  const posKey = `${position.x},${position.y}${
    position.z !== undefined ? `,${position.z}` : ''
  }`;
  const existingStack = hypotheticalBoard.stacks.get(posKey);

  const placementRings = Array(Math.max(1, count)).fill(player);

  let newRings: number[];
  if (existingStack && existingStack.rings.length > 0) {
    // Placing on an existing stack: new rings sit on top (front of array)
    newRings = [...placementRings, ...existingStack.rings];
  } else {
    // Placing on an empty space
    newRings = placementRings;
  }

  const newStack: RingStack = {
    position,
    rings: newRings,
    stackHeight: newRings.length,
    capHeight: calculateCapHeight(newRings),
    controllingPlayer: newRings[0]
  };

  hypotheticalBoard.stacks.set(posKey, newStack);

  return hypotheticalBoard;
}

/**
 * Check whether a stack at `from` would have at least one legal move or
 * capture on the provided board. This mirrors the original
 * RuleEngine.hasAnyLegalMoveOrCaptureFrom implementation but is extracted
 * so that placement validation logic can be shared.
 */
export function hasAnyLegalMoveOrCaptureFrom(
  boardManager: BoardManager,
  boardType: 'square' | 'hexagonal',
  from: Position,
  player: number,
  board: BoardState
): boolean {
  const fromKey = positionToStringLocal(from);
  const stack = board.stacks.get(fromKey);
  if (!stack || stack.controllingPlayer !== player) {
    return false;
  }

  const directions = getMovementDirectionsForBoardType(boardType as any);

  // Check for any legal non-capture movement
  for (const dir of directions) {
    for (let distance = stack.stackHeight; distance <= 8; distance++) {
      const targetPos: Position = {
        x: from.x + dir.x * distance,
        y: from.y + dir.y * distance,
        ...(dir.z !== undefined && { z: (from.z || 0) + dir.z * distance })
      };

      if (!boardManager.isValidPosition(targetPos)) {
        break; // Off board in this direction
      }

      if (boardManager.isCollapsedSpace(targetPos, board)) {
        break; // Can't move through collapsed space
      }

      // Check if path is clear
      const pathClear = isPathClearForHypothetical(boardManager, from, targetPos, board);
      if (!pathClear) {
        break; // Blocked by stacks/collapsed spaces
      }

      // Check landing position
      const targetKey = positionToStringLocal(targetPos);
      const targetStack = board.stacks.get(targetKey);
      const targetMarker = boardManager.getMarker(targetPos, board);

      // Can land on empty space, same-color marker, or own/opponent stacks (for merging)
      if (!targetStack || targetStack.rings.length === 0) {
        // Empty space or marker
        if (targetMarker === undefined || targetMarker === player) {
          return true; // Found a legal move
        }
      } else {
        // Landing on a stack - allowed for merging
        return true;
      }
    }
  }

  // Check for any legal capture
  for (const dir of directions) {
    let step = 1;
    let targetPos: Position | undefined;

    // Find first stack along this ray
    while (true) {
      const pos: Position = {
        x: from.x + dir.x * step,
        y: from.y + dir.y * step,
        ...(dir.z !== undefined && { z: (from.z || 0) + dir.z * step })
      };

      if (!boardManager.isValidPosition(pos)) {
        break;
      }

      if (boardManager.isCollapsedSpace(pos, board)) {
        break;
      }

      const posKey = positionToStringLocal(pos);
      const stackAtPos = board.stacks.get(posKey);

      if (stackAtPos && stackAtPos.rings.length > 0) {
        // Check if this stack is capturable
        if (stack.capHeight >= stackAtPos.capHeight) {
          targetPos = pos;
        }
        break;
      }

      step++;
    }

    if (!targetPos) continue;

    // Try to find at least one valid landing position beyond the target
    let landingStep = 1;
    while (landingStep <= 5) {
      const landingPos: Position = {
        x: targetPos.x + dir.x * landingStep,
        y: targetPos.y + dir.y * landingStep,
        ...(dir.z !== undefined && { z: (targetPos.z || 0) + dir.z * landingStep })
      };

      if (!boardManager.isValidPosition(landingPos)) {
        break;
      }

      if (boardManager.isCollapsedSpace(landingPos, board)) {
        break;
      }

      const landingKey = positionToStringLocal(landingPos);
      const landingStack = board.stacks.get(landingKey);
      if (landingStack && landingStack.rings.length > 0) {
        break;
      }

      // NOTE: The original hasAnyLegalMoveOrCaptureFrom called
      // this.validateCaptureSegment; here we rely on the caller to
      // perform full capture validation, so we conservatively treat any
      // reachable landing beyond a capturable target as a legal capture.
      return true;
    }
  }

  return false; // No legal moves or captures found
}

function positionToStringLocal(pos: Position): string {
  return pos.z !== undefined ? `${pos.x},${pos.y},${pos.z}` : `${pos.x},${pos.y}`;
}

function getPathPositionsLocal(from: Position, to: Position): Position[] {
  const positions: Position[] = [];

  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dz = (to.z || 0) - (from.z || 0);

  const steps = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz));

  if (steps === 0) {
    return positions; // Same position
  }

  const stepX = dx / steps;
  const stepY = dy / steps;
  const stepZ = dz / steps;

  for (let i = 1; i < steps; i++) {
    const x = Math.round(from.x + stepX * i);
    const y = Math.round(from.y + stepY * i);

    if (from.z !== undefined || to.z !== undefined) {
      const z = Math.round((from.z || 0) + stepZ * i);
      positions.push({ x, y, z });
    } else {
      positions.push({ x, y });
    }
  }

  return positions;
}

function isPathClearForHypothetical(
  boardManager: BoardManager,
  from: Position,
  to: Position,
  board: BoardState
): boolean {
  const pathPositions = getPathPositionsLocal(from, to);

  for (const pos of pathPositions) {
    if (boardManager.isCollapsedSpace(pos, board)) {
      return false;
    }

    const posKey = positionToStringLocal(pos);
    const stack = board.stacks.get(posKey);
    if (stack && stack.rings.length > 0) {
      return false;
    }
  }

  return true;
}
