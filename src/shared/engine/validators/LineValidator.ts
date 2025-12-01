import { GameState, ProcessLineAction, ChooseLineRewardAction, ValidationResult } from '../types';
import { positionToString, BoardType } from '../../types/game';
import { getEffectiveLineLengthThreshold } from '../rulesConfig';

export function validateProcessLine(state: GameState, action: ProcessLineAction): ValidationResult {
  // 1. Phase Check
  if (state.currentPhase !== 'line_processing') {
    return { valid: false, reason: 'Not in line processing phase', code: 'INVALID_PHASE' };
  }

  // 2. Turn Check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  // 3. Line Existence Check
  if (action.lineIndex < 0 || action.lineIndex >= state.board.formedLines.length) {
    return { valid: false, reason: 'Invalid line index', code: 'INVALID_LINE_INDEX' };
  }

  const line = state.board.formedLines[action.lineIndex];

  // 4. Line Ownership Check
  if (line.player !== action.playerId) {
    return { valid: false, reason: 'Cannot process opponent line', code: 'NOT_YOUR_LINE' };
  }

  return { valid: true };
}

export function validateChooseLineReward(
  state: GameState,
  action: ChooseLineRewardAction
): ValidationResult {
  // 1. Phase Check
  if (state.currentPhase !== 'line_processing') {
    return { valid: false, reason: 'Not in line processing phase', code: 'INVALID_PHASE' };
  }

  // 2. Turn Check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  // 3. Line Existence Check
  if (action.lineIndex < 0 || action.lineIndex >= state.board.formedLines.length) {
    return { valid: false, reason: 'Invalid line index', code: 'INVALID_LINE_INDEX' };
  }

  const line = state.board.formedLines[action.lineIndex];
 
  // 4. Line Ownership Check
  if (line.player !== action.playerId) {
    return { valid: false, reason: 'Cannot process opponent line', code: 'NOT_YOUR_LINE' };
  }
 
  // Effective threshold depends on board + player count. The current engine
  // GameState type does not carry per-game rulesOptions (those are host-level),
  // but the only active variant (2p 8x8 → 4-in-a-row) is determined purely by
  // boardType and number of players.
  const requiredLength = getEffectiveLineLengthThreshold(
    state.board.type as BoardType,
    state.players.length
  );

  // 5. Option Validity Check
  if (line.length === requiredLength) {
    // Exact length lines MUST be fully collapsed (Option 1 equivalent, but implicit)
    // The action type might be CHOOSE_LINE_REWARD but effectively it's forced
    // However, if the client sends MINIMUM_COLLAPSE for an exact length line, that's invalid.
    if (action.selection === 'MINIMUM_COLLAPSE') {
      return {
        valid: false,
        reason: 'Cannot choose minimum collapse for exact length line',
        code: 'INVALID_SELECTION',
      };
    }
  }

  if (action.selection === 'MINIMUM_COLLAPSE') {
    if (!action.collapsedPositions) {
      return {
        valid: false,
        reason: 'Must provide collapsed positions for minimum collapse',
        code: 'MISSING_POSITIONS',
      };
    }

    if (action.collapsedPositions.length !== requiredLength) {
      return {
        valid: false,
        reason: `Must select exactly ${requiredLength} positions`,
        code: 'INVALID_POSITION_COUNT',
      };
    }

    // Verify all selected positions are actually part of the line
    const linePosKeys = new Set(line.positions.map((p) => positionToString(p)));
    for (const pos of action.collapsedPositions) {
      if (!linePosKeys.has(positionToString(pos))) {
        return {
          valid: false,
          reason: 'Selected position is not part of the line',
          code: 'INVALID_POSITION',
        };
      }
    }

    // Verify selected positions are consecutive
    // This is complex to validate purely from positions without re-deriving the line geometry.
    // However, since we know they are part of the line (which is straight), we can check adjacency.
    // Or simpler: check if they form a subset of the line array which is contiguous.
    // But line.positions might not be sorted in order.
    // Assuming line.positions IS sorted/ordered (it should be from detection logic),
    // we can check if the selected indices are consecutive.

    // For now, we'll assume the engine's line detection provides ordered positions.
    // We can map selected positions to indices in the line.
    const indices = action.collapsedPositions
      .map((pos) => {
        const key = positionToString(pos);
        return line.positions.findIndex((p) => positionToString(p) === key);
      })
      .sort((a, b) => a - b);

    for (let i = 0; i < indices.length - 1; i++) {
      if (indices[i + 1] !== indices[i] + 1) {
        return {
          valid: false,
          reason: 'Selected positions must be consecutive',
          code: 'NON_CONSECUTIVE',
        };
      }
    }
  }

  return { valid: true };
}
