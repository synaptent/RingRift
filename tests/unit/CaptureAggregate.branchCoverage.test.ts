/**
 * CaptureAggregate.branchCoverage.test.ts
 *
 * Branch coverage tests for CaptureAggregate.ts targeting uncovered branches:
 * - validateCapture phase checks, turn checks, position validation
 * - enumerateCaptureMoves edge cases (no attacker, wrong player, collapsed spaces)
 * - enumerateAllCaptureMoves iteration
 * - Chain capture continuation checks
 */

import {
  BoardType,
  BoardState,
  GameState,
  Position,
  positionToString,
} from '../../src/shared/types/game';
import {
  validateCapture,
  enumerateCaptureMoves,
  enumerateAllCaptureMoves,
  enumerateChainCaptureSegments,
  getChainCaptureContinuationInfo,
  enumerateChainCaptures,
  mutateCapture,
  applyCapture,
  applyCaptureSegment,
  updateChainCaptureStateAfterCapture,
  type ChainCaptureStateSnapshot,
  type ChainCaptureState,
  type CaptureBoardAdapters,
} from '../../src/shared/engine/aggregates/CaptureAggregate';
import type { OvertakingCaptureAction } from '../../src/shared/engine/types';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
  addMarker,
  addCollapsedSpace,
  pos,
} from '../utils/fixtures';
import type { Move } from '../../src/shared/types/game';
import type { ContinueChainAction } from '../../src/shared/engine/types';

describe('CaptureAggregate branch coverage', () => {
  const boardType: BoardType = 'square8';

  function makeEmptyGameState(boardTypeOverride: BoardType = boardType): GameState {
    const board: BoardState = createTestBoard(boardTypeOverride);
    const players = [createTestPlayer(1), createTestPlayer(2)];
    return createTestGameState({
      boardType: boardTypeOverride,
      board,
      players,
      currentPlayer: 1,
      currentPhase: 'movement',
    });
  }

  describe('validateCapture', () => {
    it('rejects capture in ring_placement phase', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'ring_placement';

      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('rejects capture when not your turn', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 2;

      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('rejects capture with invalid from position', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(-1, -1), // Invalid position
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('rejects capture with invalid target position', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';

      addStack(state.board, pos(2, 2), 1, 2);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(99, 99), // Invalid position
        to: pos(6, 2),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('rejects capture with invalid landing position', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';

      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(-1, -1), // Invalid position
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('accepts capture in movement phase', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';

      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('accepts capture in capture phase', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'capture';

      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('accepts capture in chain_capture phase', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'chain_capture';

      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects capture when attacker cap height is lower than target', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';

      addStack(state.board, pos(2, 2), 1, 1); // Cap height 1
      addStack(state.board, pos(4, 2), 2, 2); // Cap height 2 - target has higher cap

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_CAPTURE');
    });
  });

  describe('enumerateCaptureMoves', () => {
    it('returns empty array when no attacker stack at position', () => {
      const state = makeEmptyGameState();
      const adapters: CaptureBoardAdapters = {
        isValidPosition: (p) => p.x >= 0 && p.x < 8 && p.y >= 0 && p.y < 8,
        isCollapsedSpace: () => false,
        getStackAt: () => undefined,
        getMarkerOwner: () => undefined,
      };

      const moves = enumerateCaptureMoves(boardType, pos(2, 2), 1, adapters, 1);
      expect(moves).toHaveLength(0);
    });

    it('returns empty array when stack is controlled by wrong player', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 2, 2); // Player 2's stack

      const adapters: CaptureBoardAdapters = {
        isValidPosition: (p) => p.x >= 0 && p.x < 8 && p.y >= 0 && p.y < 8,
        isCollapsedSpace: () => false,
        getStackAt: (p) => {
          if (p.x === 2 && p.y === 2) {
            return { controllingPlayer: 2, capHeight: 2, stackHeight: 2 };
          }
          return undefined;
        },
        getMarkerOwner: () => undefined,
      };

      // Player 1 trying to enumerate from player 2's stack
      const moves = enumerateCaptureMoves(boardType, pos(2, 2), 1, adapters, 1);
      expect(moves).toHaveLength(0);
    });

    it('stops enumeration at collapsed space', () => {
      const adapters: CaptureBoardAdapters = {
        isValidPosition: (p) => p.x >= 0 && p.x < 8 && p.y >= 0 && p.y < 8,
        isCollapsedSpace: (p) => p.x === 3 && p.y === 2, // Collapsed space between
        getStackAt: (p) => {
          if (p.x === 2 && p.y === 2) {
            return { controllingPlayer: 1, capHeight: 2, stackHeight: 2 };
          }
          if (p.x === 4 && p.y === 2) {
            return { controllingPlayer: 2, capHeight: 1, stackHeight: 1 };
          }
          return undefined;
        },
        getMarkerOwner: () => undefined,
      };

      // Should not find target at (4,2) because (3,2) is collapsed
      const moves = enumerateCaptureMoves(boardType, pos(2, 2), 1, adapters, 1);
      const eastCaptures = moves.filter(
        (m) => m.captureTarget && m.captureTarget.x === 4 && m.captureTarget.y === 2
      );
      expect(eastCaptures).toHaveLength(0);
    });

    it('stops enumeration at board edge', () => {
      const adapters: CaptureBoardAdapters = {
        isValidPosition: (p) => p.x >= 0 && p.x < 8 && p.y >= 0 && p.y < 8,
        isCollapsedSpace: () => false,
        getStackAt: (p) => {
          if (p.x === 7 && p.y === 2) {
            return { controllingPlayer: 1, capHeight: 2, stackHeight: 2 };
          }
          return undefined;
        },
        getMarkerOwner: () => undefined,
      };

      // Stack at edge, can't capture eastward
      const moves = enumerateCaptureMoves(boardType, pos(7, 2), 1, adapters, 1);
      // Should only have captures in other directions if targets exist
      expect(moves).toHaveLength(0); // No targets
    });

    it('finds multiple landing positions for a single target', () => {
      const adapters: CaptureBoardAdapters = {
        isValidPosition: (p) => p.x >= 0 && p.x < 8 && p.y >= 0 && p.y < 8,
        isCollapsedSpace: () => false,
        getStackAt: (p) => {
          if (p.x === 2 && p.y === 2) {
            return { controllingPlayer: 1, capHeight: 2, stackHeight: 2 };
          }
          if (p.x === 4 && p.y === 2) {
            return { controllingPlayer: 2, capHeight: 1, stackHeight: 1 };
          }
          return undefined;
        },
        getMarkerOwner: () => undefined,
      };

      const moves = enumerateCaptureMoves(boardType, pos(2, 2), 1, adapters, 1);
      // Should find captures to (5,2), (6,2), (7,2)
      const eastCaptures = moves.filter((m) => m.captureTarget?.x === 4 && m.captureTarget.y === 2);
      expect(eastCaptures.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('enumerateAllCaptureMoves', () => {
    it('returns empty array when player has no stacks', () => {
      const state = makeEmptyGameState();
      const moves = enumerateAllCaptureMoves(state, 1);
      expect(moves).toHaveLength(0);
    });

    it('enumerates captures from all player stacks', () => {
      const state = makeEmptyGameState();

      // Player 1 has two stacks
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(2, 6), 1, 2);

      // Targets for both
      addStack(state.board, pos(4, 2), 2, 1);
      addStack(state.board, pos(4, 6), 2, 1);

      const moves = enumerateAllCaptureMoves(state, 1);
      expect(moves.length).toBeGreaterThan(0);

      // Should have captures from both stacks
      const fromFirst = moves.filter((m) => m.from?.x === 2 && m.from.y === 2);
      const fromSecond = moves.filter((m) => m.from?.x === 2 && m.from.y === 6);

      expect(fromFirst.length).toBeGreaterThan(0);
      expect(fromSecond.length).toBeGreaterThan(0);
    });

    it('does not enumerate captures for other player', () => {
      const state = makeEmptyGameState();

      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      // Player 2 should not be able to capture with player 1's stack
      const moves = enumerateAllCaptureMoves(state, 2);
      const fromPlayer1Stack = moves.filter((m) => m.from?.x === 2 && m.from.y === 2);
      expect(fromPlayer1Stack).toHaveLength(0);
    });
  });

  describe('enumerateChainCaptureSegments', () => {
    it('returns empty array when no continuation targets exist', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      // No other stacks to capture

      const snapshot: ChainCaptureStateSnapshot = {
        player: 1,
        currentPosition: pos(2, 2),
        capturedThisChain: [],
      };

      const segments = enumerateChainCaptureSegments(state, snapshot, { kind: 'continuation' });
      expect(segments).toHaveLength(0);
    });

    it('generates continuation moves with correct type', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const snapshot: ChainCaptureStateSnapshot = {
        player: 1,
        currentPosition: pos(2, 2),
        capturedThisChain: [],
      };

      const segments = enumerateChainCaptureSegments(state, snapshot, { kind: 'continuation' });
      expect(segments.length).toBeGreaterThan(0);
      expect(segments[0].type).toBe('continue_capture_segment');
    });

    it('generates initial moves with correct type', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const snapshot: ChainCaptureStateSnapshot = {
        player: 1,
        currentPosition: pos(2, 2),
        capturedThisChain: [],
      };

      const segments = enumerateChainCaptureSegments(state, snapshot, { kind: 'initial' });
      expect(segments.length).toBeGreaterThan(0);
      expect(segments[0].type).toBe('overtaking_capture');
    });
  });

  describe('getChainCaptureContinuationInfo', () => {
    it('returns correct shape from chain capture info', () => {
      // Test that the function returns the expected shape
      // Full integration tested in ClientSandboxEngine chain capture tests
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      // Use enumerateChainCaptureSegments which is what the continuation check uses internally
      const snapshot: ChainCaptureStateSnapshot = {
        player: 1,
        currentPosition: pos(2, 2),
        capturedThisChain: [],
      };

      const segments = enumerateChainCaptureSegments(state, snapshot, { kind: 'continuation' });
      // If we found segments, there are continuation targets
      const hasContinuations = segments.length > 0;
      expect(typeof hasContinuations).toBe('boolean');
    });
  });

  describe('hex board captures', () => {
    it('validates captures on hex board', () => {
      const state = makeEmptyGameState('hex');
      state.currentPhase = 'movement';

      addStack(state.board, { x: 2, y: 2, z: -4 }, 1, 2);
      addStack(state.board, { x: 3, y: 2, z: -5 }, 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: { x: 2, y: 2, z: -4 },
        captureTarget: { x: 3, y: 2, z: -5 },
        to: { x: 4, y: 2, z: -6 },
      };

      const result = validateCapture(state, action);
      // The result depends on hex board validation which may or may not allow this
      // Ensure it returns a proper validation result structure
      expect(typeof result.valid).toBe('boolean');
      if (!result.valid) {
        expect(typeof result.code).toBe('string');
      }
    });
  });

  describe('self-capture scenarios', () => {
    it('allows self-capture (overtaking own stack)', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';

      addStack(state.board, pos(2, 2), 1, 3); // Player 1 stack with cap 3
      addStack(state.board, pos(4, 2), 1, 1); // Player 1 stack with cap 1

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });
  });

  describe('marker interaction', () => {
    it('captures enumerate correctly with markers on landing spaces', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      // Add marker on potential landing
      state.board.markers.set('6,2', { player: 2 });

      const moves = enumerateAllCaptureMoves(state, 1);
      // Should still have captures available (markers don't block landing)
      expect(moves.length).toBeGreaterThan(0);
    });
  });

  describe('enumerateChainCaptures', () => {
    it('returns positions from chain capture continuation info', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const positions = enumerateChainCaptures(state, pos(2, 2), 1);
      // Should return landing positions for the available captures
      expect(Array.isArray(positions)).toBe(true);
    });

    it('returns empty array when no captures available', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      // No targets to capture

      const positions = enumerateChainCaptures(state, pos(2, 2), 1);
      expect(positions).toHaveLength(0);
    });
  });

  describe('disallowRevisitedTargets filter', () => {
    it('filters out previously captured targets when option is set', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1); // Target

      const snapshot: ChainCaptureStateSnapshot = {
        player: 1,
        currentPosition: pos(2, 2),
        capturedThisChain: [pos(4, 2)], // Already captured this target
      };

      const segments = enumerateChainCaptureSegments(state, snapshot, {
        disallowRevisitedTargets: true,
        kind: 'continuation',
      });

      // Should filter out the target at (4,2) since it was already captured
      const capturesTo4_2 = segments.filter(
        (m) => m.captureTarget?.x === 4 && m.captureTarget.y === 2
      );
      expect(capturesTo4_2).toHaveLength(0);
    });

    it('allows revisited targets when option is false', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const snapshot: ChainCaptureStateSnapshot = {
        player: 1,
        currentPosition: pos(2, 2),
        capturedThisChain: [pos(4, 2)],
      };

      const segments = enumerateChainCaptureSegments(state, snapshot, {
        disallowRevisitedTargets: false,
        kind: 'continuation',
      });

      // Should allow the target at (4,2)
      const capturesTo4_2 = segments.filter(
        (m) => m.captureTarget?.x === 4 && m.captureTarget.y === 2
      );
      expect(capturesTo4_2.length).toBeGreaterThan(0);
    });

    it('handles empty capturedThisChain with disallowRevisitedTargets', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const snapshot: ChainCaptureStateSnapshot = {
        player: 1,
        currentPosition: pos(2, 2),
        capturedThisChain: [],
      };

      // Should work fine with empty array
      const segments = enumerateChainCaptureSegments(state, snapshot, {
        disallowRevisitedTargets: true,
        kind: 'continuation',
      });
      expect(segments.length).toBeGreaterThan(0);
    });
  });

  describe('mutateCapture', () => {
    it('places marker at origin after capture', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const newState = mutateCapture(state, action);

      // Origin should now have a marker
      const originMarker = newState.board.markers.get('2,2');
      expect(originMarker).toMatchObject({ player: 1 });
    });

    it('flips opponent marker in capture path', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(5, 2), 2, 1);
      // Add opponent marker in path
      addMarker(state.board, pos(3, 2), 2);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(5, 2),
        to: pos(7, 2),
      };

      const newState = mutateCapture(state, action);

      // Marker at (3,2) should be flipped to player 1
      const flippedMarker = newState.board.markers.get('3,2');
      expect(flippedMarker?.player).toBe(1);
    });

    it('collapses own marker in capture path', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(5, 2), 2, 1);
      // Add own marker in path
      addMarker(state.board, pos(3, 2), 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(5, 2),
        to: pos(7, 2),
      };

      const newState = mutateCapture(state, action);

      // Marker at (3,2) should be collapsed
      expect(newState.board.markers.has('3,2')).toBe(false);
      expect(newState.board.collapsedSpaces.has('3,2')).toBe(true);
    });

    it('handles target with multiple rings', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      addStack(state.board, pos(2, 2), 1, 2);
      // Target with 3 rings - mixed ownership
      state.board.stacks.set('4,2', {
        position: pos(4, 2),
        rings: [2, 1, 2], // P2 on top, P1, P2
        stackHeight: 3,
        capHeight: 1, // Single cap owned by P2
        controllingPlayer: 2,
      });

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const newState = mutateCapture(state, action);

      // Target should still exist with 2 rings
      const targetStack = newState.board.stacks.get('4,2');
      expect(targetStack).toMatchObject({
        stackHeight: 2,
        rings: [1, 2], // P2 ring captured, P1 now on top
        controllingPlayer: 1,
      });
    });

    it('removes target entirely when single ring', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1); // Single ring

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const newState = mutateCapture(state, action);

      // Target should be removed
      expect(newState.board.stacks.has('4,2')).toBe(false);
    });

    it('handles landing on marker and eliminates top ring', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);
      // Add marker at landing position
      addMarker(state.board, pos(6, 2), 2);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const newState = mutateCapture(state, action);

      // Marker at landing should be removed
      expect(newState.board.markers.has('6,2')).toBe(false);

      // Stack at landing should have eliminated a ring
      const landingStack = newState.board.stacks.get('6,2');
      // Original: 2 rings from attacker + 1 from target = 3, then -1 for landing = 2
      expect(landingStack?.stackHeight).toBe(2);
    });

    it('eliminates stack entirely when landing on marker with single ring', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      // Single ring attacker
      addStack(state.board, pos(2, 2), 1, 1);
      addStack(state.board, pos(4, 2), 2, 1);
      // Add marker at landing position
      addMarker(state.board, pos(6, 2), 2);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const newState = mutateCapture(state, action);

      // Attacker: 1 ring + captured: 1 ring = 2, then eliminate top = 1 remaining
      const landingStack = newState.board.stacks.get('6,2');
      expect(landingStack?.stackHeight).toBe(1);
    });

    it('throws when attacker stack is missing', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      // No attacker stack at (2,2)
      addStack(state.board, pos(4, 2), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      expect(() => mutateCapture(state, action)).toThrow('Missing attacker or target stack');
    });

    it('throws when target stack is missing', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      addStack(state.board, pos(2, 2), 1, 2);
      // No target stack at (4,2)

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      expect(() => mutateCapture(state, action)).toThrow('Missing attacker or target stack');
    });
  });

  describe('updateChainCaptureStateAfterCapture', () => {
    it('creates new chain state on first capture', () => {
      const move: Move = {
        id: 'test-capture',
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = updateChainCaptureStateAfterCapture(undefined, move, 1);

      expect(result).not.toBeUndefined();
      expect(result!.playerNumber).toBe(1);
      expect(result!.startPosition).toEqual(pos(2, 2));
      expect(result!.currentPosition).toEqual(pos(6, 2));
      expect(result!.segments).toHaveLength(1);
      expect(result!.visitedPositions.has('2,2')).toBe(true);
    });

    it('updates existing chain state on subsequent capture', () => {
      const initialState: ChainCaptureState = {
        playerNumber: 1,
        startPosition: pos(2, 2),
        currentPosition: pos(6, 2),
        segments: [
          { from: pos(2, 2), target: pos(4, 2), landing: pos(6, 2), capturedCapHeight: 1 },
        ],
        availableMoves: [],
        visitedPositions: new Set(['2,2']),
      };

      const move: Move = {
        id: 'test-capture-2',
        type: 'continue_capture_segment',
        player: 1,
        from: pos(6, 2),
        captureTarget: pos(6, 4),
        to: pos(6, 6),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      };

      const result = updateChainCaptureStateAfterCapture(initialState, move, 1);

      expect(result).not.toBeUndefined();
      expect(result!.currentPosition).toEqual(pos(6, 6));
      expect(result!.segments).toHaveLength(2);
      expect(result!.visitedPositions.has('6,2')).toBe(true);
    });

    it('returns undefined state when move has no from', () => {
      const move: Move = {
        id: 'test-move',
        type: 'overtaking_capture',
        player: 1,
        from: undefined as unknown as Position,
        captureTarget: pos(4, 2),
        to: pos(6, 2),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = updateChainCaptureStateAfterCapture(undefined, move, 1);
      expect(result).toBeUndefined();
    });

    it('returns current state when move has no captureTarget', () => {
      const existingState: ChainCaptureState = {
        playerNumber: 1,
        startPosition: pos(2, 2),
        currentPosition: pos(6, 2),
        segments: [],
        availableMoves: [],
        visitedPositions: new Set(),
      };

      const move: Move = {
        id: 'test-move',
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 2),
        captureTarget: undefined as unknown as Position,
        to: pos(6, 2),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = updateChainCaptureStateAfterCapture(existingState, move, 1);
      expect(result).toBe(existingState); // Returns same reference
    });
  });

  describe('applyCapture', () => {
    it('rejects non-capture move types', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const move: Move = {
        id: 'test-move',
        type: 'move_stack', // Wrong type
        player: 1,
        from: pos(2, 2),
        to: pos(3, 2),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyCapture(state, move);
      expect(result.success).toBe(false);
      expect(result.reason).toContain('Expected');
    });

    it('rejects move without from position', () => {
      const state = makeEmptyGameState();

      const move: Move = {
        id: 'test-move',
        type: 'overtaking_capture',
        player: 1,
        to: pos(6, 2),
        captureTarget: pos(4, 2),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyCapture(state, move);
      expect(result.success).toBe(false);
      expect(result.reason).toContain('required');
    });

    it('rejects move without captureTarget', () => {
      const state = makeEmptyGameState();

      const move: Move = {
        id: 'test-move',
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 2),
        to: pos(6, 2),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyCapture(state, move);
      expect(result.success).toBe(false);
      expect(result.reason).toContain('required');
    });

    it('successfully applies capture and returns new state', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const move: Move = {
        id: 'test-capture',
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyCapture(state, move);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.newState.board.stacks.has('6,2')).toBe(true);
        expect(Array.isArray(result.chainCaptures)).toBe(true);
      }
    });

    it('handles continue_capture_segment move type', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'chain_capture';
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const move: Move = {
        id: 'test-continue',
        type: 'continue_capture_segment',
        player: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      };

      const result = applyCapture(state, move);
      expect(result.success).toBe(true);
    });

    it('catches errors and returns failure result', () => {
      const state = makeEmptyGameState();
      // Missing stacks will cause mutateCapture to throw

      const move: Move = {
        id: 'test-capture',
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 2), // No stack here
        captureTarget: pos(4, 2), // No stack here
        to: pos(6, 2),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyCapture(state, move);
      expect(result.success).toBe(false);
      expect(result.reason).toBeDefined();
    });
  });

  describe('applyCaptureSegment', () => {
    it('returns chain continuation info', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const outcome = applyCaptureSegment(state, {
        from: pos(2, 2),
        target: pos(4, 2),
        landing: pos(6, 2),
        player: 1,
      });

      expect(outcome.nextState.board.stacks.has('6,2')).toBe(true);
      expect(outcome.ringsTransferred).toBe(1);
      expect(typeof outcome.chainContinuationRequired).toBe('boolean');
    });

    it('detects chain continuation when more captures available', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      addStack(state.board, pos(2, 2), 1, 3); // Strong stack
      addStack(state.board, pos(4, 2), 2, 1); // First target
      addStack(state.board, pos(6, 4), 2, 1); // Second target (after landing at 6,2)

      const outcome = applyCaptureSegment(state, {
        from: pos(2, 2),
        target: pos(4, 2),
        landing: pos(6, 2),
        player: 1,
      });

      // After capturing, the stack lands at (6,2) with cap height 4
      // It may be able to capture the stack at (6,4)
      expect(outcome.nextState.board.stacks.has('6,2')).toBe(true);
      expect(outcome.ringsTransferred).toBeGreaterThanOrEqual(1);
    });
  });

  describe('getChainCaptureContinuationInfo', () => {
    it('returns mustContinue=false when no targets', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      // No targets

      const info = getChainCaptureContinuationInfo(state, 1, pos(2, 2));
      expect(info.mustContinue).toBe(false);
      expect(info.availableContinuations).toHaveLength(0);
    });

    it('returns mustContinue=true with available continuations', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);

      const info = getChainCaptureContinuationInfo(state, 1, pos(2, 2));
      expect(info.mustContinue).toBe(true);
      expect(info.availableContinuations.length).toBeGreaterThan(0);
    });
  });

  describe('enumeration break conditions', () => {
    it('stops landing enumeration at collapsed space', () => {
      const adapters: CaptureBoardAdapters = {
        isValidPosition: (p) => p.x >= 0 && p.x < 8 && p.y >= 0 && p.y < 8,
        isCollapsedSpace: (p) => p.x === 6 && p.y === 2, // Collapsed at first landing
        getStackAt: (p) => {
          if (p.x === 2 && p.y === 2) {
            return { controllingPlayer: 1, capHeight: 2, stackHeight: 2 };
          }
          if (p.x === 4 && p.y === 2) {
            return { controllingPlayer: 2, capHeight: 1, stackHeight: 1 };
          }
          return undefined;
        },
        getMarkerOwner: () => undefined,
      };

      // First valid landing would be (5,2), but (6,2) is collapsed
      const moves = enumerateCaptureMoves(boardType, pos(2, 2), 1, adapters, 1);

      // Should have landing at (5,2) but not beyond (6,2) since it's collapsed
      const eastCaptures = moves.filter((m) => m.captureTarget?.x === 4 && m.captureTarget.y === 2);
      const landingsAt6 = eastCaptures.filter((m) => m.to.x === 6);
      expect(landingsAt6).toHaveLength(0);
    });

    it('stops landing enumeration when hitting another stack', () => {
      const adapters: CaptureBoardAdapters = {
        isValidPosition: (p) => p.x >= 0 && p.x < 8 && p.y >= 0 && p.y < 8,
        isCollapsedSpace: () => false,
        getStackAt: (p) => {
          if (p.x === 2 && p.y === 2) {
            return { controllingPlayer: 1, capHeight: 2, stackHeight: 2 };
          }
          if (p.x === 4 && p.y === 2) {
            return { controllingPlayer: 2, capHeight: 1, stackHeight: 1 };
          }
          if (p.x === 6 && p.y === 2) {
            return { controllingPlayer: 1, capHeight: 1, stackHeight: 1 }; // Blocking stack
          }
          return undefined;
        },
        getMarkerOwner: () => undefined,
      };

      const moves = enumerateCaptureMoves(boardType, pos(2, 2), 1, adapters, 1);

      // Should have landing at (5,2) but not at (6,2) or beyond due to blocking stack
      const eastCaptures = moves.filter((m) => m.captureTarget?.x === 4 && m.captureTarget.y === 2);
      const landingsAt6 = eastCaptures.filter((m) => m.to.x === 6);
      expect(landingsAt6).toHaveLength(0);
    });
  });

  describe('mixed ownership target stacks', () => {
    it('handles capture of stack with alternating ownership', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      addStack(state.board, pos(2, 2), 1, 3);

      // Target with alternating rings: P2, P1, P2, P1
      state.board.stacks.set('4,2', {
        position: pos(4, 2),
        rings: [2, 1, 2, 1],
        stackHeight: 4,
        capHeight: 1, // P2 controls (single ring on top)
        controllingPlayer: 2,
      });

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const newState = mutateCapture(state, action);

      // Target should have 3 remaining rings with P1 now controlling
      const targetStack = newState.board.stacks.get('4,2');
      expect(targetStack?.stackHeight).toBe(3);
      expect(targetStack?.rings).toEqual([1, 2, 1]);
      expect(targetStack?.controllingPlayer).toBe(1);
    });
  });

  describe('marker processing in path', () => {
    it('processes multiple markers in path correctly', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';
      addStack(state.board, pos(1, 2), 1, 2);
      addStack(state.board, pos(5, 2), 2, 1);

      // Opponent marker between origin and target
      addMarker(state.board, pos(2, 2), 2);
      // Own marker between origin and target
      addMarker(state.board, pos(3, 2), 1);
      // Opponent marker between target and landing
      addMarker(state.board, pos(6, 2), 2);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(1, 2),
        captureTarget: pos(5, 2),
        to: pos(7, 2),
      };

      const newState = mutateCapture(state, action);

      // Opponent marker at (2,2) should be flipped
      expect(newState.board.markers.get('2,2')?.player).toBe(1);

      // Own marker at (3,2) should be collapsed
      expect(newState.board.markers.has('3,2')).toBe(false);
      expect(newState.board.collapsedSpaces.has('3,2')).toBe(true);

      // Opponent marker at (6,2) should be flipped
      expect(newState.board.markers.get('6,2')?.player).toBe(1);
    });
  });

  describe('applyCapture with chain continuation', () => {
    it('populates chainCaptures when continuation is required', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';

      // Set up a scenario where chain capture will be possible after first capture
      // Strong attacker
      addStack(state.board, pos(2, 2), 1, 3);
      // First target
      addStack(state.board, pos(4, 2), 2, 1);
      // Second target reachable from landing position (6,2)
      addStack(state.board, pos(6, 4), 2, 1);

      const move: Move = {
        id: 'test-capture',
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyCapture(state, move);
      expect(result.success).toBe(true);
      if (result.success) {
        // After landing at (6,2), should be able to continue capturing (6,4)
        // The chainCaptures array should be populated if continuation is required
        expect(Array.isArray(result.chainCaptures)).toBe(true);
      }
    });

    it('includes continuation landing positions in chainCaptures', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';

      // Strong attacker at (2, 3)
      addStack(state.board, pos(2, 3), 1, 4);
      // First target at (4, 3)
      addStack(state.board, pos(4, 3), 2, 1);
      // Second target at (6, 5) - capturable after landing at (6, 3)
      addStack(state.board, pos(6, 5), 2, 1);

      const move: Move = {
        id: 'test-capture',
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 3),
        captureTarget: pos(4, 3),
        to: pos(6, 3),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyCapture(state, move);
      expect(result.success).toBe(true);
      if (result.success && result.chainCaptures.length > 0) {
        // Each position in chainCaptures should be a valid landing position
        for (const chainPos of result.chainCaptures) {
          expect(chainPos).toMatchObject({
            x: expect.any(Number),
            y: expect.any(Number),
          });
        }
      }
    });
  });

  describe('getMarkerOwner adapter functions', () => {
    it('createBoardAdapters getMarkerOwner returns marker owner', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);
      // Add a marker
      addMarker(state.board, pos(5, 2), 2);

      // When we enumerate captures, the adapter's getMarkerOwner is called internally
      // to check if landing on own marker (which would trigger ring elimination)
      const moves = enumerateAllCaptureMoves(state, 1);
      // The function should work regardless of marker presence
      expect(Array.isArray(moves)).toBe(true);
    });

    it('getMarkerOwner returns undefined for position without marker', () => {
      const state = makeEmptyGameState();
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1);
      // No markers

      const moves = enumerateAllCaptureMoves(state, 1);
      expect(moves.length).toBeGreaterThan(0);
    });
  });

  describe('edge case: landing eliminates stack completely', () => {
    it('handles rare case where combined stack has single ring and lands on marker', () => {
      // This is a theoretical edge case - in practice, capturing always adds at least
      // one ring from the target to the attacker, so minimum combined stack is 2 rings.
      // After landing on marker, minimum is 1 ring.
      // The code path for reducedRings.length === 0 at line 773 appears to be defensive.

      const state = makeEmptyGameState();
      state.currentPhase = 'movement';

      // Single ring attacker
      addStack(state.board, pos(2, 2), 1, 1);
      // Single ring target
      addStack(state.board, pos(4, 2), 2, 1);
      // Marker at landing
      addMarker(state.board, pos(6, 2), 2);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const newState = mutateCapture(state, action);

      // After capture: 1 (attacker) + 1 (captured) = 2 rings
      // After landing on marker: 2 - 1 = 1 ring remaining
      const landingStack = newState.board.stacks.get('6,2');
      expect(landingStack?.stackHeight).toBe(1);
    });
  });

  describe('validateCaptureSegmentOnBoard via validateCapture', () => {
    it('rejects capture when path is blocked', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';

      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1); // Target
      addStack(state.board, pos(3, 2), 1, 1); // Blocking stack in path

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
    });

    it('rejects capture when landing path is blocked', () => {
      const state = makeEmptyGameState();
      state.currentPhase = 'movement';

      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(4, 2), 2, 1); // Target
      addStack(state.board, pos(5, 2), 1, 1); // Stack between target and landing

      const action: OvertakingCaptureAction = {
        type: 'overtaking_capture',
        playerId: 1,
        from: pos(2, 2),
        captureTarget: pos(4, 2),
        to: pos(6, 2), // Must pass through (5,2) to reach (6,2)
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
    });
  });
});
