import {
  BoardState,
  BoardType,
  BOARD_CONFIGS,
  LineInfo,
  Position,
  positionToString
} from '../../shared/types/game';

export type LineDirection = { x: number; y: number; z?: number };

/**
 * Directions along which lines may form.
 * Mirrors BoardManager.getLineDirections (4 directions for square, 3 for hex),
 * scanning both forward and backward.
 */
export function getLineDirectionsForBoard(boardType: BoardType): LineDirection[] {
  if (boardType === 'hexagonal') {
    return [
      { x: 1, y: 0, z: -1 }, // East
      { x: 1, y: -1, z: 0 }, // Northeast
      { x: 0, y: -1, z: 1 }  // Northwest
    ];
  }

  // Square boards: 4 principal directions (E, SE, S, NE)
  return [
    { x: 1, y: 0 },  // East
    { x: 1, y: 1 },  // Southeast
    { x: 0, y: 1 },  // South
    { x: 1, y: -1 }  // Northeast
  ];
}

/**
 * Find a contiguous line of same-player markers starting from `start`
 * and extending in both forward and backward directions along `direction`.
 *
 * This is intentionally kept close to BoardManager.findLineInDirection:
 *   - Lines are formed by MARKERS, not stacks.
 *   - Lines cannot be interrupted by empty spaces, collapsed spaces, or stacks.
 */
export function findLineInDirectionOnBoard(
  start: Position,
  direction: LineDirection,
  playerNumber: number,
  board: BoardState,
  isValidPosition: (pos: Position) => boolean
): Position[] {
  const line: Position[] = [start];
  const isHex = board.type === 'hexagonal';

  // Walk forward
  let current = start;
  while (true) {
    const next: Position = isHex
      ? {
          x: current.x + direction.x,
          y: current.y + direction.y,
          z: (current.z ?? 0) + (direction.z ?? 0)
        }
      : {
          x: current.x + direction.x,
          y: current.y + direction.y
        };

    if (!isValidPosition(next)) break;

    const key = positionToString(next);
    const marker = board.markers.get(key);
    if (!marker || marker.player !== playerNumber) break;

    if (board.collapsedSpaces.has(key)) break;
    if (board.stacks.has(key)) break;

    line.push(next);
    current = next;
  }

  // Walk backward
  current = start;
  while (true) {
    const prev: Position = isHex
      ? {
          x: current.x - direction.x,
          y: current.y - direction.y,
          z: (current.z ?? 0) - (direction.z ?? 0)
        }
      : {
          x: current.x - direction.x,
          y: current.y - direction.y
        };

    if (!isValidPosition(prev)) break;

    const key = positionToString(prev);
    const marker = board.markers.get(key);
    if (!marker || marker.player !== playerNumber) break;

    if (board.collapsedSpaces.has(key)) break;
    if (board.stacks.has(key)) break;

    line.unshift(prev);
    current = prev;
  }

  return line;
}

/**
 * Find all marker lines on the board for all players.
 * Mirrors BoardManager.findAllLines; only returns lines of at least the
 * configured minimum length.
 */
export function findAllLinesOnBoard(
  boardType: BoardType,
  board: BoardState,
  isValidPosition: (pos: Position) => boolean,
  stringToPosition: (posStr: string) => Position
): LineInfo[] {
  const lines: LineInfo[] = [];
  const processed = new Set<string>();
  const requiredLength = BOARD_CONFIGS[boardType].lineLength;
  const directions = getLineDirectionsForBoard(boardType);

  for (const [posStr, marker] of board.markers) {
    const position = stringToPosition(posStr);
    for (const direction of directions) {
      const linePositions = findLineInDirectionOnBoard(
        position,
        direction,
        marker.player,
        board,
        isValidPosition
      );

      if (linePositions.length >= requiredLength) {
        const key = linePositions
          .map(p => positionToString(p))
          .sort()
          .join('|');

        if (!processed.has(key)) {
          processed.add(key);
          lines.push({
            positions: linePositions,
            player: marker.player,
            length: linePositions.length,
            direction
          });
        }
      }
    }
  }

  return lines;
}
