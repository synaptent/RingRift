import {
  BoardState,
  LineInfo,
  Position,
  BOARD_CONFIGS,
  positionToString,
  stringToPosition,
} from '../types/game';

/**
 * Find all marker lines on the board (4+ for 8x8, 5+ for 19x19/hex)
 * Rule Reference: Section 11.1 - Line Formation Rules
 * CRITICAL: Lines are formed by MARKERS, not stacks!
 */
export function findAllLines(board: BoardState): LineInfo[] {
  const lines: LineInfo[] = [];
  const processedLines = new Set<string>();
  const config = BOARD_CONFIGS[board.type];

  // Iterate through all MARKERS (not stacks!).
  for (const [posStr, marker] of board.markers) {
    const position = stringToPosition(posStr);

    // Treat stacks and collapsed spaces as hard blockers
    if (isCollapsedSpace(position, board) || getStack(position, board)) {
      continue;
    }

    const directions = getLineDirections(board.type);

    for (const direction of directions) {
      const line = findLineInDirection(position, direction, marker.player, board);

      if (line.length >= config.lineLength) {
        // Create a unique key for this line (sorted positions to avoid duplicates)
        const lineKey = line
          .map((p) => positionToString(p))
          .sort()
          .join('|');

        if (!processedLines.has(lineKey)) {
          processedLines.add(lineKey);
          lines.push({
            positions: line,
            player: marker.player,
            length: line.length,
            direction: direction,
          });
        }
      }
    }
  }

  return lines;
}

function getLineDirections(boardType: string): Position[] {
  if (boardType === 'hexagonal') {
    // 6 directions for hexagonal
    return [
      { x: 1, y: 0, z: -1 }, // East
      { x: 1, y: -1, z: 0 }, // Northeast
      { x: 0, y: -1, z: 1 }, // Northwest
    ];
  } else {
    // 8 directions for square (Moore adjacency for lines)
    // Actually we only need 4 directions to cover all lines if we iterate all markers?
    // No, we need to check all directions from each marker because we don't know where the line starts.
    // But findAllLines iterates ALL markers.
    // If we check only "positive" directions, we avoid duplicates?
    // The original code used 4 directions for square: E, SE, S, NE.
    // This covers horizontal, vertical, and both diagonals.
    return [
      { x: 1, y: 0 }, // East
      { x: 1, y: 1 }, // Southeast
      { x: 0, y: 1 }, // South
      { x: 1, y: -1 }, // Northeast
    ];
  }
}

function findLineInDirection(
  startPosition: Position,
  direction: Position,
  playerId: number,
  board: BoardState
): Position[] {
  const line: Position[] = [startPosition];
  const isHex = board.type === 'hexagonal';

  // Helper to step one cell in the given direction
  const step = (current: Position, sign: 1 | -1): Position => {
    if (isHex) {
      return {
        x: current.x + sign * direction.x,
        y: current.y + sign * direction.y,
        z: (current.z || 0) + sign * (direction.z || 0),
      };
    }
    return {
      x: current.x + sign * direction.x,
      y: current.y + sign * direction.y,
    };
  };

  // Check forward direction
  let current = startPosition;
  while (true) {
    const next = step(current, 1);

    if (!isValidPosition(next, board)) break;

    const marker = getMarker(next, board);
    if (marker !== playerId) break;

    if (isCollapsedSpace(next, board) || getStack(next, board)) break;

    line.push(next);
    current = next;
  }

  // Check backward direction
  current = startPosition;
  while (true) {
    const prev = step(current, -1);

    if (!isValidPosition(prev, board)) break;

    const marker = getMarker(prev, board);
    if (marker !== playerId) break;

    if (isCollapsedSpace(prev, board) || getStack(prev, board)) break;

    line.unshift(prev);
    current = prev;
  }

  return line;
}

// Helpers adapted from BoardManager/utils

function isValidPosition(position: Position, board: BoardState): boolean {
  // We can check bounds based on board size
  const size = board.size;
  if (board.type === 'hexagonal') {
    const radius = size - 1;
    const q = position.x;
    const r = position.y;
    const s = position.z || -q - r;
    return (
      Math.abs(q) <= radius && Math.abs(r) <= radius && Math.abs(s) <= radius && q + r + s === 0
    );
  } else {
    return position.x >= 0 && position.x < size && position.y >= 0 && position.y < size;
  }
}

function getMarker(position: Position, board: BoardState): number | undefined {
  const posKey = positionToString(position);
  const marker = board.markers.get(posKey);
  return marker?.player;
}

function isCollapsedSpace(position: Position, board: BoardState): boolean {
  const posKey = positionToString(position);
  return board.collapsedSpaces.has(posKey);
}

function getStack(position: Position, board: BoardState): any | undefined {
  const posKey = positionToString(position);
  return board.stacks.get(posKey);
}
