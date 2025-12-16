import {
  BoardState,
  BoardType,
  LineInfo,
  Position,
  RingStack,
  positionToString,
  stringToPosition,
} from '../types/game';
import { getEffectiveLineLengthThreshold } from './rulesConfig';

/**
 * Detect all marker lines on the board according to the canonical
 * RingRift line rules (Section 11.1).
 *
 * Line length thresholds per RR-CANON-R120:
 * - square8 2-player: 4
 * - square8 3-4 player: 3
 * - square19 and hexagonal: 4 (all player counts)
 *
 * This helper is the single source of truth for line geometry used by:
 * - the shared GameEngine (advanced phases),
 * - the backend BoardManager / RuleEngine, and
 * - the client sandbox line engines.
 *
 * CRITICAL: Lines are formed by MARKERS, not stacks.
 *
 * @param board The current board state
 * @param numPlayers Number of players in the game. Required for determining
 *   the correct line length threshold. Defaults to 3 (uses base threshold)
 *   for backward compatibility, but callers should pass the actual value.
 */
export function findAllLines(board: BoardState, numPlayers: number = 3): LineInfo[] {
  const lines: LineInfo[] = [];
  const processedLines = new Set<string>();
  const requiredLength = getEffectiveLineLengthThreshold(board.type as BoardType, numPlayers);

  // Iterate through all MARKERS (not stacks!). If a space currently
  // hosts a stack or has already collapsed to territory, it cannot be
  // part of an active marker line.
  // Support both Map and plain object for markers (plain objects come from JSON deserialization).
  const markerEntries = iterateMapOrObject(board.markers);
  for (const [posStr, marker] of markerEntries) {
    const position = stringToPosition(posStr);

    // Treat stacks and collapsed spaces as hard blockers
    if (isCollapsedSpace(position, board) || getStack(position, board)) {
      continue;
    }

    const directions = getLineDirections(board.type);

    for (const direction of directions) {
      const line = findLineInDirection(position, direction, marker.player, board);

      if (line.length >= requiredLength) {
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

/**
 * Detect all marker lines on the board that belong to a specific player.
 * Thin convenience wrapper over {@link findAllLines} used by hosts that
 * want a player-filtered view without re-implementing geometry.
 *
 * @param board The current board state
 * @param playerNumber The player to find lines for
 * @param numPlayers Number of players in the game. Required for determining
 *   the correct line length threshold. Defaults to 3 (uses base threshold)
 *   for backward compatibility, but callers should pass the actual value.
 */
export function findLinesForPlayer(
  board: BoardState,
  playerNumber: number,
  numPlayers: number = 3
): LineInfo[] {
  return findAllLines(board, numPlayers).filter((line) => line.player === playerNumber);
}

function getLineDirections(boardType: string): Position[] {
  if (boardType === 'hexagonal' || boardType === 'hex8') {
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
  const isHex = board.type === 'hexagonal' || board.type === 'hex8';

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
  if (board.type === 'hexagonal' || board.type === 'hex8') {
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
  const marker = getFromMapOrObject(board.markers, posKey);
  return marker?.player;
}

function isCollapsedSpace(position: Position, board: BoardState): boolean {
  const posKey = positionToString(position);
  return hasInMapOrObject(board.collapsedSpaces, posKey);
}

function getStack(position: Position, board: BoardState): RingStack | undefined {
  const posKey = positionToString(position);
  return getFromMapOrObject(board.stacks, posKey);
}

// Helper to iterate over Map or plain object (for JSON-deserialized states)
function iterateMapOrObject<T>(mapOrObj: Map<string, T> | Record<string, T>): [string, T][] {
  if (mapOrObj instanceof Map) {
    return Array.from(mapOrObj.entries());
  }
  // Plain object from JSON deserialization
  return Object.entries(mapOrObj);
}

// Helper to get from Map or plain object
function getFromMapOrObject<T>(
  mapOrObj: Map<string, T> | Record<string, T>,
  key: string
): T | undefined {
  if (mapOrObj instanceof Map) {
    return mapOrObj.get(key);
  }
  // Plain object from JSON deserialization
  return mapOrObj[key];
}

// Helper to check existence in Map or plain object
function hasInMapOrObject<T>(mapOrObj: Map<string, T> | Record<string, T>, key: string): boolean {
  if (mapOrObj instanceof Map) {
    return mapOrObj.has(key);
  }
  // Plain object from JSON deserialization
  return key in mapOrObj;
}
