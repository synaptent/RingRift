import {
  BoardState,
  BoardType,
  BOARD_CONFIGS,
  Position,
  RingStack,
  positionToString
} from '../../shared/types/game';
import { calculateDistance, getMovementDirectionsForBoardType, getPathPositions } from '../../shared/engine/core';

export interface SimpleLanding {
  fromKey: string;
  to: Position;
}

/**
 * Enumerate simple, non-capturing movement options for the given player on
 * the provided board. This is a pure helper that mirrors the path/occupancy
 * checks used by ClientSandboxEngine for simple moves.
 */
export function enumerateSimpleMovementLandings(
  boardType: BoardType,
  board: BoardState,
  playerNumber: number,
  isValidPosition: (pos: Position) => boolean
): SimpleLanding[] {
  const results: SimpleLanding[] = [];
  const directions = getMovementDirectionsForBoardType(boardType);
  const config = BOARD_CONFIGS[boardType];

  for (const stack of board.stacks.values()) {
    if (stack.controllingPlayer !== playerNumber) continue;
    const from = stack.position;
    const fromKey = positionToString(from);

    for (const dir of directions) {
      let step = 1;
      for (;;) {
        const to: Position = {
          x: from.x + dir.x * step,
          y: from.y + dir.y * step,
          ...(dir.z !== undefined && { z: (from.z || 0) + dir.z * step })
        };

        const toKey = positionToString(to);

        if (!isValidPosition(to) || Math.abs(to.x) > config.size * 2 || Math.abs(to.y) > config.size * 2) {
          break;
        }

        // Disallow landing on collapsed spaces or occupied stacks.
        if (board.collapsedSpaces.has(toKey) || board.stacks.has(toKey)) {
          break;
        }

        // Check that the path between from and to is unobstructed (no stacks
        // or collapsed spaces).
        const path = getPathPositions(from, to).slice(1, -1);
        let blocked = false;
        for (const pos of path) {
          const pathKey = positionToString(pos);
          if (board.collapsedSpaces.has(pathKey) || board.stacks.has(pathKey)) {
            blocked = true;
            break;
          }
        }
        if (blocked) {
          break;
        }

        const distance = calculateDistance(boardType, from, to);
        if (distance >= stack.stackHeight) {
          results.push({ fromKey, to });
        }

        step += 1;
      }
    }
  }

  return results;
}

export interface MarkerPathHelpers {
  setMarker(position: Position, playerNumber: number, board: BoardState): void;
  collapseMarker(position: Position, playerNumber: number, board: BoardState): void;
  flipMarker(position: Position, playerNumber: number, board: BoardState): void;
}

/**
 * Apply marker effects for a move or capture segment from `from` to `to` on
 * the given board, using the provided helper callbacks. This mirrors the
 * backend marker-path behaviour and is used by ClientSandboxEngine for both
 * movement and capture paths.
 */
export function applyMarkerEffectsAlongPathOnBoard(
  board: BoardState,
  from: Position,
  to: Position,
  playerNumber: number,
  helpers: MarkerPathHelpers
): void {
  const path = getPathPositions(from, to);
  if (path.length === 0) return;

  const fromKey = positionToString(from);
  // Leave a marker on the departure space if it isn't already collapsed.
  if (!board.collapsedSpaces.has(fromKey)) {
    const existing = board.markers.get(fromKey);
    if (!existing) {
      helpers.setMarker(from, playerNumber, board);
    }
  }

  // Process intermediate positions (excluding endpoints)
  const intermediate = path.slice(1, -1);
  for (const pos of intermediate) {
    const key = positionToString(pos);
    if (board.collapsedSpaces.has(key)) {
      continue;
    }
    const marker = board.markers.get(key);
    if (!marker) {
      continue;
    }
    if (marker.player === playerNumber) {
      // Own marker collapses to territory
      helpers.collapseMarker(pos, playerNumber, board);
    } else {
      // Opponent marker flips to mover's color
      helpers.flipMarker(pos, playerNumber, board);
    }
  }

  // Landing: remove own marker if present
  const landingKey = positionToString(to);
  const landingMarker = board.markers.get(landingKey);
  if (landingMarker && landingMarker.player === playerNumber) {
    board.markers.delete(landingKey);
  }
}
