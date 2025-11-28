import type {
  BoardState,
  BoardType,
  Position,
  RingStack,
  CaptureSegmentBoardView,
  CaptureBoardAdapters as SharedCaptureBoardAdapters,
} from '../../shared/engine';
import {
  positionToString,
  getMovementDirectionsForBoardType,
  validateCaptureSegmentOnBoard,
  calculateCapHeight,
  getPathPositions,
  enumerateCaptureMoves as enumerateCaptureMovesShared,
} from '../../shared/engine';

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
  // First, delegate to the shared capture-move helper so that sandbox
  // enumeration stays aligned with the unified rules geometry.
  const sharedAdapters: SharedCaptureBoardAdapters = {
    isValidPosition: (pos: Position) => adapters.isValidPosition(pos),
    isCollapsedSpace: (pos: Position) => adapters.isCollapsedSpace(pos, board),
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
    getMarkerOwner: (pos: Position) => adapters.getMarkerOwner(pos, board),
  };

  // moveNumber is irrelevant for sandbox enumeration; pass 0.
  const moves = enumerateCaptureMovesShared(boardType, from, playerNumber, sharedAdapters, 0);

  const segments: CaptureSegment[] = moves.map((m) => ({
    from: m.from as Position,
    target: m.captureTarget as Position,
    landing: m.to as Position,
  }));

  // The shared helper now correctly enumerates all valid landings (distance >= stackHeight),
  // so we no longer need the manual extension logic here.
  return segments;
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
 * IMPORTANT:
 * - This helper is intentionally NON-CANONICAL and is used only by sandbox-side
 *   analysis/debug tooling (e.g. capture sequence search, parity harnesses).
 * - Live sandbox engine flows (ClientSandboxEngine, sandboxMovementEngine) now
 *   delegate capture mutation exclusively to the shared CaptureAggregate via
 *   {@link applyCaptureSegment} / {@link applyCapture} in src/shared/engine.
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

  // Invariant alignment: a cell may host either a stack, a marker, or
  // collapsed territory, but never combinations. The backend enforces
  // this by routing all stack writes through BoardManager.setStack,
  // which drops any marker on the destination cell. The sandbox
  // capture path historically wrote directly to board.stacks without
  // clearing markers, which allowed stack+marker overlaps when a
  // capture landed on a marked cell. To keep the S-invariant and
  // board exclusivity rules consistent between engines, we explicitly
  // clear any marker at the landing key before placing the updated
  // stack.
  board.markers.delete(landingKey);

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
