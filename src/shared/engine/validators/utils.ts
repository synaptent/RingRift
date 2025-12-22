import { BoardType, Position } from '../../types/game';
import { debugLog, flagEnabled } from '../../utils/envFlags';

/**
 * Checks if a position is within the bounds of the board.
 *
 * Note: Uses STRICT hex validation (q + r + s === 0) to match VictoryAggregate.ts.
 * This ensures consistent validation across all engine code paths.
 */
export function isValidPosition(pos: Position, boardType: BoardType, boardSize: number): boolean {
  if (boardType === 'hexagonal' || boardType === 'hex8') {
    // For hex boards: boardSize is the bounding box (2*radius + 1).
    // hex8: boardSize=9 → radius=4, hexagonal: boardSize=25 → radius=12
    const radius = (boardSize - 1) / 2;
    const q = pos.x;
    const r = pos.y;
    const s = pos.z ?? -q - r;

    // Strict hex validation: q + r + s must exactly equal 0
    // This matches VictoryAggregate.ts isValidBoardPosition for consistency
    const sum = q + r + s;
    const boundsValid = Math.abs(q) <= radius && Math.abs(r) <= radius && Math.abs(s) <= radius;
    const sumValid = sum === 0;

    // Debug logging to detect when lenient would differ from strict
    const debugValidation = flagEnabled('RINGRIFT_DEBUG_VALIDATION');
    if (debugValidation) {
      const lenientSumValid = Math.round(sum) === 0;
      if (lenientSumValid !== sumValid) {
        debugLog(debugValidation, '[isValidPosition] VALIDATION DISCREPANCY DETECTED:', {
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
