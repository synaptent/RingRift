import { GameState, OvertakingCaptureAction, ValidationResult } from '../types';
import { positionToString, Position } from '../../types/game';
import { validateCaptureSegmentOnBoard, CaptureSegmentBoardView } from '../core';
import { isValidPosition } from './utils';

export function validateCapture(
  state: GameState,
  action: OvertakingCaptureAction
): ValidationResult {
  // 1. Phase Check
  // Capture can happen in 'movement' (initial capture) or 'capture'/'chain_capture' phases
  if (
    state.currentPhase !== 'movement' &&
    state.currentPhase !== 'capture' &&
    state.currentPhase !== 'chain_capture'
  ) {
    return { valid: false, reason: 'Not in a phase allowing capture', code: 'INVALID_PHASE' };
  }

  // 2. Turn Check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  // 3. Position Validity
  if (
    !isValidPosition(action.from, state.board.type, state.board.size) ||
    !isValidPosition(action.to, state.board.type, state.board.size) ||
    !isValidPosition(action.captureTarget, state.board.type, state.board.size)
  ) {
    return { valid: false, reason: 'Position off board', code: 'INVALID_POSITION' };
  }

  // 4. Use Shared Core Validator
  // We construct a minimal view of the board to pass to the shared validator
  const boardView: CaptureSegmentBoardView = {
    isValidPosition: (pos: Position) => isValidPosition(pos, state.board.type, state.board.size),
    isCollapsedSpace: (pos: Position) => state.board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const stack = state.board.stacks.get(positionToString(pos));
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos: Position) => {
      const marker = state.board.markers.get(positionToString(pos));
      return marker?.player;
    },
  };

  const isValid = validateCaptureSegmentOnBoard(
    state.board.type,
    action.from,
    action.captureTarget,
    action.to,
    action.playerId,
    boardView
  );

  if (!isValid) {
    // The core validator returns boolean, so we provide a generic reason.
    // In a real implementation, we might want more specific error codes from the core validator,
    // but for now this ensures strict adherence to the rules implemented in core.ts.
    return {
      valid: false,
      reason: 'Invalid capture move according to core rules',
      code: 'INVALID_CAPTURE',
    };
  }

  return { valid: true };
}
