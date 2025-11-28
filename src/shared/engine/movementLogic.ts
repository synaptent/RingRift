import { BoardType, Position } from '../types/game';
import {
  getMovementDirectionsForBoardType,
  getPathPositions,
  calculateDistance,
  MovementBoardView,
} from './core';

/**
 * Shared helpers for non-capturing movement reachability.
 *
 * These functions are intentionally board-agnostic and operate on the
 * minimal {@link MovementBoardView} interface so they can be reused by
 * the backend RuleEngine as well as the client sandbox engine
 * and shared GameEngine parity tests.
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
 * Enumerate all legal simple (non-capturing) movement targets for the
 * stack controlled by {@code player} at {@code from} on a given board.
 *
 * Semantics are aligned with:
 *
 * - Backend RuleEngine.validateStackMovement,
 * - Sandbox ClientSandboxEngine.getValidLandingPositionsForCurrentPlayer, and
 * - hasAnyLegalMoveOrCaptureFromOnBoard in the shared core.
 *
 * Rules encoded:
 *
 * - Movement follows a straight ray along the canonical direction set
 *   for the board type (8-direction Moore for square, 6 cube axes for hex).
 * - The stack must move a distance >= its stackHeight.
 * - The path (excluding endpoints) must be clear of stacks and collapsed
 *   spaces; markers are allowed on the path.
 * - Landing is allowed on:
 *   - Empty spaces;
 *   - Spaces containing a single marker owned by the moving player.
 * - Landing on an existing stack is NOT allowed (stacks block the ray).
 * - Landing on an opponent marker is illegal, but such markers do not
 *   block further movement along the ray.
 */
export function enumerateSimpleMoveTargetsFromStack(
  boardType: BoardType,
  from: Position,
  player: number,
  board: MovementBoardAdapters
): SimpleMoveTarget[] {
  const stack = board.getStackAt(from);
  if (!stack || stack.controllingPlayer !== player || stack.stackHeight <= 0) {
    return [];
  }

  const directions = getMovementDirectionsForBoardType(boardType);
  const results: SimpleMoveTarget[] = [];

  for (const dir of directions) {
    let step = 1;
    // Walk outward along this ray until we leave the board or hit an
    // obstruction that blocks further movement.
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const to: Position = {
        x: from.x + dir.x * step,
        y: from.y + dir.y * step,
      };
      if (dir.z !== undefined) {
        to.z = (from.z || 0) + dir.z * step;
      }

      if (!board.isValidPosition(to)) {
        break; // Off board along this ray.
      }

      if (board.isCollapsedSpace(to)) {
        break; // Cannot move into collapsed territory.
      }

      // Path between from and to (excluding endpoints) must be clear of
      // stacks and collapsed spaces. Markers are allowed and processed
      // separately by marker mutators.
      const path = getPathPositions(from, to).slice(1, -1);
      let blocked = false;
      for (const pos of path) {
        if (!board.isValidPosition(pos)) {
          blocked = true;
          break;
        }
        if (board.isCollapsedSpace(pos)) {
          blocked = true;
          break;
        }
        const pathStack = board.getStackAt(pos);
        if (pathStack && pathStack.stackHeight > 0) {
          blocked = true;
          break;
        }
      }
      if (blocked) {
        // Further positions along this ray are also blocked.
        break;
      }

      const landingStack = board.getStackAt(to);
      const markerOwner = board.getMarkerOwner?.(to);

      const distance = calculateDistance(boardType, from, to);

      if (!landingStack || landingStack.stackHeight === 0) {
        // Empty space or marker-only cell.
        if (markerOwner !== undefined && markerOwner !== player) {
          // Opponent marker: cannot land here, but the ray continues
          // past this cell.
          step += 1;
          continue;
        }

        if (distance >= stack.stackHeight) {
          results.push({ from, to });
        }

        // Empty/own-marker cells do not block further exploration
        // along this ray.
        step += 1;
        continue;
      }

      // Landing on an existing stack is NOT allowed - stacks block the ray.
      // Rule 8.1: "Cannot pass through other rings or stacks"
      // Rule 8.2: "Landing on ... empty or occupied by a single marker"
      break;
    }
  }

  return results;
}
