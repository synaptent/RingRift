import { GameState, PlaceRingAction, SkipPlacementAction, ValidationResult } from '../types';
import { positionToString, Position } from '../../types/game';
import { hasAnyLegalMoveOrCaptureFromOnBoard, MovementBoardView } from '../core';
import { isValidPosition } from './utils';

export function validatePlacement(state: GameState, action: PlaceRingAction): ValidationResult {
  // 1. Phase Check
  if (state.currentPhase !== 'ring_placement') {
    return { valid: false, reason: 'Not in ring placement phase', code: 'INVALID_PHASE' };
  }

  // 2. Turn Check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  const player = state.players.find((p) => p.playerNumber === action.playerId);
  if (!player) {
    return { valid: false, reason: 'Player not found', code: 'PLAYER_NOT_FOUND' };
  }

  // 3. Rings in Hand Check
  if (player.ringsInHand < action.count) {
    return { valid: false, reason: 'Not enough rings in hand', code: 'INSUFFICIENT_RINGS' };
  }

  // 4. Position Validity Check
  if (!isValidPosition(action.position, state.board.type, state.board.size)) {
    return { valid: false, reason: 'Position off board', code: 'INVALID_POSITION' };
  }

  const posKey = positionToString(action.position);

  // 5. Collapsed Space Check
  if (state.board.collapsedSpaces.has(posKey)) {
    return { valid: false, reason: 'Cannot place on collapsed space', code: 'COLLAPSED_SPACE' };
  }

  const existingStack = state.board.stacks.get(posKey);

  // 6. Placement Logic Checks
  if (existingStack) {
    // Placing on existing stack
    if (action.count !== 1) {
      return {
        valid: false,
        reason: 'Can only place 1 ring on an existing stack',
        code: 'INVALID_COUNT',
      };
    }
  } else {
    // Placing on empty space
    if (action.count < 1 || action.count > 3) {
      return { valid: false, reason: 'Must place 1-3 rings on empty space', code: 'INVALID_COUNT' };
    }
    // Note: ringsInHand check above covers the capacity limit implicitly
  }

  // 7. No-Dead-Placement Rule Check
  // Construct a hypothetical stack representing the state AFTER placement
  const hypotheticalStack = existingStack
    ? {
        controllingPlayer: action.playerId,
        stackHeight: existingStack.stackHeight + action.count,
        // If placing on own stack, cap increases. If placing on other, cap becomes 1 (the new ring).
        capHeight:
          existingStack.controllingPlayer === action.playerId
            ? existingStack.capHeight + action.count
            : action.count,
      }
    : {
        controllingPlayer: action.playerId,
        stackHeight: action.count,
        capHeight: action.count,
      };

  // Create a temporary board view that includes this hypothetical stack
  const tempBoardView: MovementBoardView = {
    isValidPosition: (pos: Position) => isValidPosition(pos, state.board.type, state.board.size),
    isCollapsedSpace: (pos: Position) => state.board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      if (key === posKey) {
        return hypotheticalStack;
      }
      return state.board.stacks.get(key);
    },
    getMarkerOwner: (pos: Position) => {
      const marker = state.board.markers.get(positionToString(pos));
      return marker?.player;
    },
  };

  const hasLegalMove = hasAnyLegalMoveOrCaptureFromOnBoard(
    state.board.type,
    action.position,
    action.playerId,
    tempBoardView
  );

  if (!hasLegalMove) {
    return {
      valid: false,
      reason: 'Placement would result in a stack with no legal moves',
      code: 'NO_LEGAL_MOVES',
    };
  }

  return { valid: true };
}

export function validateSkipPlacement(
  state: GameState,
  action: SkipPlacementAction
): ValidationResult {
  // SKIP_PLACEMENT models the explicit choice to forego optional placement
  // at the start of a turn. Per the written rules (Sections 4.1 and 4.2):
  //
  // - Placement is mandatory only when the player has no rings on the board
  //   (and rings in hand), or when they are "blocked with stacks" and must
  //   resolve the situation via forced elimination.
  // - When a player already controls at least one stack that has a legal
  //   move or capture available, placement becomes optional; they may
  //   choose to skip it and proceed directly to movement.

  // 1. Phase check – skip_placement is only meaningful during ring_placement.
  if (state.currentPhase !== 'ring_placement') {
    return { valid: false, reason: 'Not in ring placement phase', code: 'INVALID_PHASE' };
  }

  // 2. Turn check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  const player = state.players.find((p) => p.playerNumber === action.playerId);
  if (!player) {
    return { valid: false, reason: 'Player not found', code: 'PLAYER_NOT_FOUND' };
  }

  // 3. If the player has no rings in hand, there is nothing to place and
  //    skipping placement is always legal; the turn proceeds to movement.
  if (player.ringsInHand <= 0) {
    return { valid: true };
  }

  // 4. Otherwise, placement is only optional when the player already has at
  //    least one legal move or capture from a controlled stack. If they have
  //    rings in hand but no such stack, they are in one of the mandatory
  //    placement / forced-elimination cases and may not explicitly skip.

  const boardView: MovementBoardView = {
    isValidPosition: (pos: Position) => isValidPosition(pos, state.board.type, state.board.size),
    isCollapsedSpace: (pos: Position) => state.board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      const stack = state.board.stacks.get(key);
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

  let hasControlledStack = false;
  let hasLegalActionFromStack = false;

  for (const stack of state.board.stacks.values()) {
    if (stack.controllingPlayer !== action.playerId || stack.stackHeight <= 0) {
      continue;
    }
    hasControlledStack = true;

    if (
      hasAnyLegalMoveOrCaptureFromOnBoard(
        state.board.type,
        stack.position,
        action.playerId,
        boardView
      )
    ) {
      hasLegalActionFromStack = true;
      break;
    }
  }

  if (!hasControlledStack) {
    return {
      valid: false,
      reason: 'Cannot skip placement when you control no stacks on the board',
      code: 'NO_CONTROLLED_STACKS',
    };
  }

  if (!hasLegalActionFromStack) {
    return {
      valid: false,
      reason: 'Cannot skip placement when no legal moves or captures are available',
      code: 'NO_LEGAL_ACTIONS',
    };
  }

  return { valid: true };
}
