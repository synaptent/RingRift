import {
  BoardState,
  BoardType,
  Position,
  RingStack,
  positionToString
} from '../../shared/types/game';
import {
  getMovementDirectionsForBoardType,
  validateCaptureSegmentOnBoard,
  CaptureSegmentBoardView,
  calculateCapHeight,
  getPathPositions
} from '../../shared/engine/core';

export interface CaptureSegment {
  from: Position;
  target: Position;
  landing: Position;
}

export interface CaptureBoardAdapters {
  isValidPosition(pos: Position): boolean;
  isCollapsedSpace(pos: Position, board: BoardState): boolean;
  getMarkerOwner(pos: Position, board: BoardState): number | undefined;
}

/**
 * Enumerate all legal overtaking capture segments for the given player from
 * the specified stack position, using the same ray-walk semantics as the
 * backend RuleEngine / GameEngine.
 */
export function enumerateCaptureSegmentsFromBoard(
  boardType: BoardType,
  board: BoardState,
  from: Position,
  playerNumber: number,
  adapters: CaptureBoardAdapters
): CaptureSegment[] {
  const results: CaptureSegment[] = [];
  const directions = getMovementDirectionsForBoardType(boardType);
  const fromKey = positionToString(from);
  const attacker = board.stacks.get(fromKey);
  if (!attacker) return results;

  for (const dir of directions) {
    let step = 1;
    let targetPos: Position | undefined;

    while (true) {
      const pos: Position = {
        x: from.x + dir.x * step,
        y: from.y + dir.y * step,
        ...(dir.z !== undefined && { z: (from.z || 0) + dir.z * step })
      };

      if (!adapters.isValidPosition(pos)) {
        break;
      }

      if (adapters.isCollapsedSpace(pos, board)) {
        break;
      }

      const sKey = positionToString(pos);
      const stack = board.stacks.get(sKey);
      if (stack && stack.stackHeight > 0) {
        if (attacker.capHeight >= stack.capHeight) {
          targetPos = pos;
        }
        break;
      }

      step++;
    }

    if (!targetPos) continue;

    let landingStep = 1;
    while (true) {
      const landingPos: Position = {
        x: targetPos.x + dir.x * landingStep,
        y: targetPos.y + dir.y * landingStep,
        ...(dir.z !== undefined && { z: (targetPos.z || 0) + dir.z * landingStep })
      };

      if (!adapters.isValidPosition(landingPos)) {
        break;
      }

      if (adapters.isCollapsedSpace(landingPos, board)) {
        break;
      }

      const landingKey = positionToString(landingPos);
      const landingStack = board.stacks.get(landingKey);
      if (landingStack && landingStack.stackHeight > 0) {
        break;
      }

      const view: CaptureSegmentBoardView = {
        isValidPosition: (pos: Position) => adapters.isValidPosition(pos),
        isCollapsedSpace: (pos: Position) => adapters.isCollapsedSpace(pos, board),
        getStackAt: (pos: Position) => {
          const sKey = positionToString(pos);
          const stack = board.stacks.get(sKey);
          if (!stack) return undefined;
          return {
            controllingPlayer: stack.controllingPlayer,
            capHeight: stack.capHeight,
            stackHeight: stack.stackHeight
          };
        },
        getMarkerOwner: (pos: Position) => adapters.getMarkerOwner(pos, board)
      };

      const ok = validateCaptureSegmentOnBoard(
        boardType,
        from,
        targetPos,
        landingPos,
        playerNumber,
        view
      );

      if (ok) {
        results.push({ from, target: targetPos, landing: landingPos });
      }

      landingStep++;
    }
  }

  return results;
}

export interface CaptureApplyAdapters {
  applyMarkerEffectsAlongPath(
    from: Position,
    to: Position,
    playerNumber: number,
    options?: { leaveDepartureMarker?: boolean }
  ): void;
}

/**
 * Apply a single overtaking capture segment, including marker processing and
 * top-ring-only overtaking semantics, mutating the provided board.
 *
 * Marker behaviour mirrors backend GameEngine.performOvertakingCapture:
 *   - Leave a marker on the true departure space (`from`).
 *   - Process markers along the path from `from` to `target`.
 *   - Process markers along the path from `target` to `landing` WITHOUT
 *     placing a new departure marker on `target`.
 *   - Markers on the capture target and landing cells themselves are not
 *     treated as intermediate path markers here; landing-on-own-marker
 *     elimination is handled by the caller.
 */
export function applyCaptureSegmentOnBoard(
  board: BoardState,
  from: Position,
  target: Position,
  landing: Position,
  playerNumber: number,
  adapters: CaptureApplyAdapters
): void {
  const fromKey = positionToString(from);
  const targetKey = positionToString(target);

  const attacker = board.stacks.get(fromKey);
  const targetStack = board.stacks.get(targetKey);
  if (!attacker || !targetStack) {
    return;
  }

  // First process markers along the path from the true departure space to the
  // capture target, leaving a departure marker at `from`.
  adapters.applyMarkerEffectsAlongPath(from, target, playerNumber, {
    leaveDepartureMarker: true,
  });

  // Then process markers along the path from the capture target to the landing
  // cell without placing a new departure marker on `target`. This keeps
  // marker-path semantics aligned with the backend's two-leg processing while
  // still handling intermediate flips/collapses on the second leg.
  adapters.applyMarkerEffectsAlongPath(target, landing, playerNumber, {
    leaveDepartureMarker: false,
  });

  if (targetStack.rings.length === 0) {
    return;
  }

  const [capturedRing, ...remaining] = targetStack.rings;

  if (remaining.length > 0) {
    const newTarget: RingStack = {
      ...targetStack,
      rings: remaining,
      stackHeight: remaining.length,
      capHeight: calculateCapHeight(remaining),
      controllingPlayer: remaining[0],
    };
    board.stacks.set(targetKey, newTarget);
  } else {
    board.stacks.delete(targetKey);
  }

  const newRings = [...attacker.rings, capturedRing];
  const landingKey = positionToString(landing);
  const updatedStack: RingStack = {
    position: landing,
    rings: newRings,
    stackHeight: newRings.length,
    capHeight: calculateCapHeight(newRings),
    controllingPlayer: newRings[0],
  };

  board.stacks.delete(fromKey);
  board.stacks.set(landingKey, updatedStack);
}
