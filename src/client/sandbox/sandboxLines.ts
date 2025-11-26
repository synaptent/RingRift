import type { BoardState, BoardType, LineInfo, Position } from '../../shared/engine';
import { positionToString, findAllLines as findAllLinesShared } from '../../shared/engine';

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
      { x: 0, y: -1, z: 1 }, // Northwest
    ];
  }

  // Square boards: 4 principal directions (E, SE, S, NE)
  return [
    { x: 1, y: 0 }, // East
    { x: 1, y: 1 }, // Southeast
    { x: 0, y: 1 }, // South
    { x: 1, y: -1 }, // Northeast
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
          z: (current.z ?? 0) + (direction.z ?? 0),
        }
      : {
          x: current.x + direction.x,
          y: current.y + direction.y,
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
          z: (current.z ?? 0) - (direction.z ?? 0),
        }
      : {
          x: current.x - direction.x,
          y: current.y - direction.y,
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
 *
 * This is a thin adapter over the canonical shared helper
 * {@link findAllLinesShared} so that sandbox code and rules/parity
 * tests can continue to depend on the historical
 * findAllLinesOnBoard(...) export without re-implementing geometry.
 *
 * The extra parameters (boardType, isValidPosition, stringToPosition)
 * are retained for backwards compatibility and for tests that stub
 * this function, but they are no longer used to derive line geometry.
 * All hosts (backend, sandbox, shared GameEngine) now share a single
 * source of truth in src/shared/engine/lineDetection.ts.
 */
export function findAllLinesOnBoard(
  boardType: BoardType,
  board: BoardState,
  isValidPosition: (pos: Position) => boolean,
  stringToPosition: (posStr: string) => Position
): LineInfo[] {
  void boardType;
  void isValidPosition;
  void stringToPosition;
  return findAllLinesShared(board);
}
