import { BoardType, Position } from '../../types/game';

/**
 * Checks if a position is within the bounds of the board.
 *
 * Note: Uses STRICT hex validation (q + r + s === 0) to match VictoryAggregate.ts.
 * This ensures consistent validation across all engine code paths.
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

    // Strict hex validation: q + r + s must exactly equal 0
    // This matches VictoryAggregate.ts isValidBoardPosition for consistency
    const sum = q + r + s;
    const boundsValid = Math.abs(q) <= radius && Math.abs(r) <= radius && Math.abs(s) <= radius;
    const sumValid = sum === 0;

    // Debug logging to detect when lenient would differ from strict
    if (process.env.RINGRIFT_DEBUG_VALIDATION === 'true') {
      const lenientSumValid = Math.round(sum) === 0;
      if (lenientSumValid !== sumValid) {
        console.log('[isValidPosition] VALIDATION DISCREPANCY DETECTED:', {
          position: { q, r, s },
          sum,
          strictValid: sumValid,
          lenientValid: lenientSumValid,
          boundsValid,
        });
      }
    }

    return boundsValid && sumValid;
  } else {
    // Square
    return pos.x >= 0 && pos.x < boardSize && pos.y >= 0 && pos.y < boardSize;
  }
}
