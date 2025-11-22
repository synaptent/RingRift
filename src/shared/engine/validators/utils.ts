import { BoardType, Position } from '../../types/game';

/**
 * Checks if a position is within the bounds of the board.
 */
export function isValidPosition(pos: Position, boardType: BoardType, boardSize: number): boolean {
  if (boardType === 'hexagonal') {
    // For hex, size is the number of hexes from center to edge (inclusive)
    // In boardMovementGrid.ts: const radius = board.size - 1;
    // So valid range is [-radius, radius].
    const radius = boardSize - 1;
    const q = pos.x;
    const r = pos.y;
    const s = pos.z ?? -q - r;
    return (
      Math.abs(q) <= radius &&
      Math.abs(r) <= radius &&
      Math.abs(s) <= radius &&
      Math.round(q + r + s) === 0
    );
  } else {
    // Square
    return pos.x >= 0 && pos.x < boardSize && pos.y >= 0 && pos.y < boardSize;
  }
}
