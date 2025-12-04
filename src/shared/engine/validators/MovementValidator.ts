import { GameState, MoveStackAction, ValidationResult } from '../types';
import { positionToString } from '../../types/game';
import { getPathPositions, calculateDistance, getMovementDirectionsForBoardType } from '../core';
import { isValidPosition } from './utils';

export function validateMovement(state: GameState, action: MoveStackAction): ValidationResult {
  // 1. Phase Check
  if (state.currentPhase !== 'movement') {
    return { valid: false, reason: 'Not in movement phase', code: 'INVALID_PHASE' };
  }

  // 2. Turn Check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  // 3. Position Validity
  if (
    !isValidPosition(action.from, state.board.type, state.board.size) ||
    !isValidPosition(action.to, state.board.type, state.board.size)
  ) {
    return { valid: false, reason: 'Position off board', code: 'INVALID_POSITION' };
  }

  const fromKey = positionToString(action.from);
  const toKey = positionToString(action.to);

  // 4. Stack Ownership
  const stack = state.board.stacks.get(fromKey);
  if (!stack) {
    return { valid: false, reason: 'No stack at starting position', code: 'NO_STACK' };
  }
  if (stack.controllingPlayer !== action.playerId) {
    return { valid: false, reason: 'You do not control this stack', code: 'NOT_YOUR_STACK' };
  }

  // 5. Collapsed Space Check
  if (state.board.collapsedSpaces.has(toKey)) {
    return { valid: false, reason: 'Cannot move to collapsed space', code: 'COLLAPSED_SPACE' };
  }

  // 6. Direction Check
  const dx = action.to.x - action.from.x;
  const dy = action.to.y - action.from.y;
  const dz = (action.to.z || 0) - (action.from.z || 0);

  const directions = getMovementDirectionsForBoardType(state.board.type);
  let validDirection = false;

  // Normalize direction vector to check against canonical directions
  // This is a bit tricky because distance varies.
  // Instead, we can check if the move aligns with any canonical direction.
  for (const dir of directions) {
    // Check if (dx, dy, dz) is a positive scalar multiple of dir
    // We need to find k > 0 such that dx = k*dir.x, dy = k*dir.y, dz = k*dir.z

    let k = 0;
    if (dir.x !== 0) k = dx / dir.x;
    else if (dir.y !== 0) k = dy / dir.y;
    else if (dir.z !== undefined && dir.z !== 0) k = dz / dir.z;

    if (k > 0) {
      // Verify all components match
      const matchX = Math.abs(dx - k * dir.x) < 0.001;
      const matchY = Math.abs(dy - k * dir.y) < 0.001;
      const matchZ = dir.z !== undefined ? Math.abs(dz - k * dir.z) < 0.001 : true;

      if (matchX && matchY && matchZ) {
        validDirection = true;
        break;
      }
    }
  }

  if (!validDirection) {
    return { valid: false, reason: 'Invalid movement direction', code: 'INVALID_DIRECTION' };
  }

  // 7. Minimum Distance Check
  const distance = calculateDistance(state.board.type, action.from, action.to);
  if (distance < stack.stackHeight) {
    return {
      valid: false,
      reason: 'Move distance less than stack height',
      code: 'INSUFFICIENT_DISTANCE',
    };
  }

  // 8. Path Check (excluding start and end)
  const path = getPathPositions(action.from, action.to);
  // Remove start and end
  const innerPath = path.slice(1, -1);

  for (const pos of innerPath) {
    const key = positionToString(pos);

    // Cannot pass through collapsed spaces
    if (state.board.collapsedSpaces.has(key)) {
      return { valid: false, reason: 'Path blocked by collapsed space', code: 'PATH_BLOCKED' };
    }

    // Cannot pass through other stacks (unless capturing, which is a different action)
    const pathStack = state.board.stacks.get(key);
    if (pathStack && pathStack.stackHeight > 0) {
      return { valid: false, reason: 'Path blocked by stack', code: 'PATH_BLOCKED' };
    }
  }

  // 9. Landing Check
  const landingStack = state.board.stacks.get(toKey);
  const landingMarker = state.board.markers.get(toKey);

  // Can land on empty space
  if (!landingStack && !landingMarker) {
    return { valid: true };
  }

  // Per RR-CANON-R091/R092: Can land on any marker (own or opponent).
  // The marker is removed and a ring from the cap is eliminated.
  if (landingMarker && !landingStack) {
    return { valid: true };
  }

  // Cannot land on existing stack
  // Rule 8.1: "Cannot pass through other rings or stacks".
  if (landingStack) {
    return { valid: false, reason: 'Cannot land on existing stack', code: 'INVALID_LANDING' };
  }

  return { valid: true };
}
