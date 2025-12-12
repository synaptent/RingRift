/**
 * CaptureValidator Unit Tests
 *
 * Tests the shared engine capture validation logic including:
 * - Phase validation (movement, capture, chain_capture)
 * - Turn validation
 * - Position validity
 * - Core capture rules delegation
 */

import { validateCapture } from '../../../src/shared/engine/validators/CaptureValidator';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  pos,
  posStr,
} from '../../utils/fixtures';
import { BoardState, RingStack, MarkerInfo } from '../../../src/shared/types/game';
import { OvertakingCaptureAction, GameState } from '../../../src/shared/engine/types';

describe('CaptureValidator', () => {
  let state: GameState;

  beforeEach(() => {
    state = createTestGameState({
      currentPhase: 'movement',
      currentPlayer: 1,
      players: [createTestPlayer(1, { ringsInHand: 10 }), createTestPlayer(2, { ringsInHand: 10 })],
    });

    // Set up a capture scenario:
    // Player 1 stack at (2, 3) with height 2
    // Player 2 stack at (3, 3) with height 1 (capturable)
    // Empty landing at (4, 3)
    const attackerStack: RingStack = {
      position: pos(2, 3),
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    };
    state.board.stacks.set(posStr(2, 3), attackerStack);

    const targetStack: RingStack = {
      position: pos(3, 3),
      rings: [2],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 2,
    };
    state.board.stacks.set(posStr(3, 3), targetStack);
  });

  describe('phase validation', () => {
    it('allows capture in movement phase', () => {
      state.currentPhase = 'movement';
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('allows capture in capture phase', () => {
      state.currentPhase = 'capture';
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('allows capture in chain_capture phase', () => {
      state.currentPhase = 'chain_capture';
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects capture in ring_placement phase', () => {
      state.currentPhase = 'ring_placement';
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('rejects capture in line_processing phase', () => {
      state.currentPhase = 'line_processing';
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });
  });

  describe('turn validation', () => {
    it('allows capture by current player', () => {
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects capture by non-current player', () => {
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 2,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });
  });

  describe('position validation', () => {
    it('rejects capture from off-board position', () => {
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(-1, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('rejects capture with off-board target', () => {
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(-1, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('rejects capture to off-board landing', () => {
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(10, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });
  });

  describe('core capture rules', () => {
    it('validates a proper overtaking capture', () => {
      // Attacker at (2,3) height 2, target at (3,3) height 1, landing at (4,3)
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects capture when attacker capHeight <= target capHeight', () => {
      // Make target taller than attacker
      const tallerTarget: RingStack = {
        position: pos(3, 3),
        rings: [2, 2, 2],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 2,
      };
      state.board.stacks.set(posStr(3, 3), tallerTarget);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_CAPTURE');
    });

    it('allows capture of own stack (per rules clarification)', () => {
      // Per RingRift rules clarification, players CAN overtake their own stacks
      // This allows tactical repositioning and stack consolidation
      const ownStack: RingStack = {
        position: pos(3, 3),
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      };
      state.board.stacks.set(posStr(3, 3), ownStack);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects capture with blocked landing', () => {
      // Put a stack at landing position
      const blockingStack: RingStack = {
        position: pos(4, 3),
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      };
      state.board.stacks.set(posStr(4, 3), blockingStack);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_CAPTURE');
    });

    it('rejects capture to collapsed space', () => {
      state.board.collapsedSpaces.set(posStr(4, 3), 1);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_CAPTURE');
    });

    it('rejects capture when target is not adjacent in the direction', () => {
      // Move target away from the straight line
      state.board.stacks.delete(posStr(3, 3));
      const movedTarget: RingStack = {
        position: pos(3, 4),
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      };
      state.board.stacks.set(posStr(3, 4), movedTarget);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 4), // Not on the same line as from -> to
        to: pos(4, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_CAPTURE');
    });
  });

  describe('chain capture scenarios', () => {
    beforeEach(() => {
      state.currentPhase = 'chain_capture';

      // Set up a chain capture scenario
      // After first capture, attacker is at (2, 3) with combined height
      // New target at (4, 3), landing at (6, 3)
      // Distance from (2,3) to (6,3) = 4 spaces, which is >= combined stackHeight of 3
      // Clear all existing stacks first
      state.board.stacks.clear();

      const chainAttacker: RingStack = {
        position: pos(2, 3),
        rings: [1, 1, 2], // Combined stack after capture
        stackHeight: 3,
        capHeight: 2, // Only attacker's original height counts
        controllingPlayer: 1,
      };
      state.board.stacks.set(posStr(2, 3), chainAttacker);

      const chainTarget: RingStack = {
        position: pos(4, 3),
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      };
      state.board.stacks.set(posStr(4, 3), chainTarget);
    });

    it('allows valid chain capture', () => {
      // Chain capture: from (2,3) capture at (4,3) land at (6,3)
      // Distance = 4 spaces >= stackHeight 3
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(4, 3),
        to: pos(6, 3),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects chain capture when distance less than stack height', () => {
      // Try to capture with landing too close (distance 2 < stackHeight 3)
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 3),
        captureTarget: pos(3, 3), // Adjacent to attacker
        to: pos(4, 3), // Distance = 2, but stackHeight = 3
      };
      // Need to set up target at (3,3)
      const closeTarget: RingStack = {
        position: pos(3, 3),
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      };
      state.board.stacks.set(posStr(3, 3), closeTarget);

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_CAPTURE');
    });
  });

  describe('hexagonal board captures', () => {
    beforeEach(() => {
      state = createTestGameState({
        boardType: 'hexagonal',
        currentPhase: 'movement',
        currentPlayer: 1,
        players: [
          createTestPlayer(1, { ringsInHand: 72 }),
          createTestPlayer(2, { ringsInHand: 72 }),
        ],
      });
      state.board = createTestBoard('hexagonal');

      // Attacker at center
      const attacker: RingStack = {
        position: pos(0, 0, 0),
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      };
      state.board.stacks.set(posStr(0, 0, 0), attacker);

      // Target along hex axis
      const target: RingStack = {
        position: pos(1, -1, 0),
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      };
      state.board.stacks.set(posStr(1, -1, 0), target);
    });

    it('allows capture along hex axis', () => {
      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(0, 0, 0),
        captureTarget: pos(1, -1, 0),
        to: pos(2, -2, 0),
      };
      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });
  });
});
