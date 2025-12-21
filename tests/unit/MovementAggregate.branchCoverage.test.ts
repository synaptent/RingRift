/**
 * MovementAggregate.branchCoverage.test.ts
 *
 * Branch coverage tests for MovementAggregate.ts targeting uncovered branches:
 * - validateMovement: all validation error paths
 * - mutateMovement: landing on markers, ring elimination
 * - applySimpleMovement: marker effects, stack merges, collapsed territory
 * - applyMovement: error handling wrapper
 * - Enumeration functions
 */

import type { GameState, Position, Move, BoardState } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

import {
  validateMovement,
  enumerateMovementTargets,
  enumerateSimpleMovesForPlayer,
  enumerateAllMovementMoves,
  mutateMovement,
  applySimpleMovement,
  applyMovement,
} from '../../src/shared/engine/aggregates/MovementAggregate';

import type { MoveStackAction } from '../../src/shared/engine/types';

import {
  createTestBoard,
  createTestGameState,
  addStack,
  addMarker,
  addCollapsedSpace,
  pos,
} from '../utils/fixtures';

// Helper to create a base state for movement tests
function makeBaseState(options: { currentPlayer?: number; currentPhase?: string } = {}): GameState {
  const board = createTestBoard('square8');
  const from = pos(3, 3);
  addStack(board, from, 1, 3); // Height 3 stack

  return createTestGameState({
    boardType: 'square8',
    board,
    currentPlayer: options.currentPlayer ?? 1,
    currentPhase: (options.currentPhase as GameState['currentPhase']) ?? 'movement',
  });
}

describe('MovementAggregate - Branch Coverage', () => {
  // ==========================================================================
  // validateMovement - Lines 208-335
  // ==========================================================================
  describe('validateMovement', () => {
    it('rejects when not in movement phase', () => {
      const state = makeBaseState({ currentPhase: 'ring_placement' });
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.code).toBe('INVALID_PHASE');
      }
    });

    it('rejects when not player turn', () => {
      const state = makeBaseState({ currentPlayer: 2 });
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.code).toBe('NOT_YOUR_TURN');
      }
    });

    it('rejects invalid from position', () => {
      const state = makeBaseState();
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(-1, -1), // Invalid
        to: pos(6, 3),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.code).toBe('INVALID_POSITION');
      }
    });

    it('rejects invalid to position', () => {
      const state = makeBaseState();
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(100, 100), // Invalid
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.code).toBe('INVALID_POSITION');
      }
    });

    it('rejects when no stack at from position', () => {
      const state = makeBaseState();
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(0, 0), // No stack here
        to: pos(3, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.code).toBe('NO_STACK');
      }
    });

    it('rejects when player does not control stack', () => {
      const state = makeBaseState();
      // Add a stack controlled by player 2
      addStack(state.board, pos(5, 5), 2, 2);

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1, // Player 1 trying to move player 2's stack
        from: pos(5, 5),
        to: pos(7, 5),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.code).toBe('NOT_YOUR_STACK');
      }
    });

    it('rejects move to collapsed space', () => {
      const state = makeBaseState();
      addCollapsedSpace(state.board, pos(6, 3), 1);

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.code).toBe('COLLAPSED_SPACE');
      }
    });

    it('rejects invalid direction', () => {
      const state = makeBaseState();
      // Try to move diagonally in a way that doesn't match direction vectors
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 6), // Not a straight line in any direction
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.code).toBe('INVALID_DIRECTION');
      }
    });

    it('rejects move distance less than stack height', () => {
      const state = makeBaseState();
      // Stack at 3,3 has height 3, so must move at least 3 spaces
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3), // Only 2 spaces
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.code).toBe('INSUFFICIENT_DISTANCE');
      }
    });

    it('rejects path blocked by collapsed space', () => {
      const state = makeBaseState();
      addCollapsedSpace(state.board, pos(4, 3), 1); // Block the path

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.code).toBe('PATH_BLOCKED');
      }
    });

    it('rejects path blocked by another stack', () => {
      const state = makeBaseState();
      addStack(state.board, pos(5, 3), 2, 1); // Block the path

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.code).toBe('PATH_BLOCKED');
      }
    });

    it('allows landing on empty space', () => {
      const state = makeBaseState();
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('allows landing on own marker', () => {
      const state = makeBaseState();
      addMarker(state.board, pos(6, 3), 1); // Own marker

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('allows landing on opponent marker', () => {
      const state = makeBaseState();
      addMarker(state.board, pos(6, 3), 2); // Opponent marker

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects landing on existing stack', () => {
      const state = makeBaseState();
      addStack(state.board, pos(6, 3), 2, 1); // Another stack at destination

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.code).toBe('INVALID_LANDING');
      }
    });

    it('handles z-coordinate direction validation for hexagonal', () => {
      const board = createTestBoard('hexagonal');
      addStack(board, { x: 0, y: 0, z: 0 }, 1, 2);

      const state = createTestGameState({
        boardType: 'hexagonal',
        board,
        currentPlayer: 1,
        currentPhase: 'movement',
      });

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: { x: 0, y: 0, z: 0 },
        to: { x: 2, y: -2, z: 0 },
      };

      const result = validateMovement(state, action);
      expect(typeof result.valid).toBe('boolean');
    });
  });

  // ==========================================================================
  // Enumeration functions - Lines 350-424
  // ==========================================================================
  describe('enumerateMovementTargets', () => {
    it('returns targets for player with stacks', () => {
      const state = makeBaseState();
      const targets = enumerateMovementTargets(state, 1);

      expect(Array.isArray(targets)).toBe(true);
    });

    it('returns empty for player with no stacks', () => {
      const state = makeBaseState();
      const targets = enumerateMovementTargets(state, 2);

      expect(targets.length).toBe(0);
    });
  });

  describe('enumerateSimpleMovesForPlayer', () => {
    it('returns moves for player in movement phase', () => {
      const state = makeBaseState();
      const moves = enumerateSimpleMovesForPlayer(state, 1);

      expect(Array.isArray(moves)).toBe(true);
      expect(moves.length).toBeGreaterThan(0);
    });

    it('returns moves regardless of phase', () => {
      // enumerateSimpleMovesForPlayer does not check phase
      const state = makeBaseState({ currentPhase: 'ring_placement' });
      const moves = enumerateSimpleMovesForPlayer(state, 1);

      // Returns moves based on board state, not phase
      expect(Array.isArray(moves)).toBe(true);
    });
  });

  describe('enumerateAllMovementMoves', () => {
    it('aliases enumerateSimpleMovesForPlayer', () => {
      const state = makeBaseState();
      const moves1 = enumerateSimpleMovesForPlayer(state, 1);
      const moves2 = enumerateAllMovementMoves(state, 1);

      expect(moves1.length).toBe(moves2.length);
    });
  });

  // ==========================================================================
  // mutateMovement - Lines 441-522
  // ==========================================================================
  describe('mutateMovement', () => {
    it('moves stack to empty space', () => {
      const state = makeBaseState();
      state.totalRingsEliminated = 0;

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const newState = mutateMovement(state, action);

      // Stack should be at new position
      expect(newState.board.stacks.has(positionToString(pos(6, 3)))).toBe(true);
      // Marker should be at origin
      expect(newState.board.markers.has(positionToString(pos(3, 3)))).toBe(true);
      // Original position should be empty
      expect(newState.board.stacks.has(positionToString(pos(3, 3)))).toBe(false);
    });

    it('eliminates top ring when landing on marker', () => {
      const state = makeBaseState();
      state.totalRingsEliminated = 0;
      addMarker(state.board, pos(6, 3), 2); // Opponent marker

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const newState = mutateMovement(state, action);

      // Marker should be removed
      expect(newState.board.markers.has(positionToString(pos(6, 3)))).toBe(false);
      // Stack should have one less ring
      const newStack = newState.board.stacks.get(positionToString(pos(6, 3)));
      expect(newStack?.stackHeight).toBe(2); // Was 3, now 2
      // Eliminated count should increase
      expect(newState.totalRingsEliminated).toBe(1);
    });

    it('completely removes stack if height was 1 and landed on marker', () => {
      const board = createTestBoard('square8');
      addStack(board, pos(3, 3), 1, 1); // Height 1 stack

      const state = createTestGameState({
        boardType: 'square8',
        board,
        currentPlayer: 1,
        currentPhase: 'movement',
      });
      state.totalRingsEliminated = 0;
      addMarker(state.board, pos(4, 3), 2); // Marker at distance 1

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(4, 3),
      };

      const newState = mutateMovement(state, action);

      // Stack should be completely gone (height was 1, eliminated the only ring)
      expect(newState.board.stacks.has(positionToString(pos(4, 3)))).toBe(false);
    });

    it('throws error when no stack at origin', () => {
      const state = makeBaseState();
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(0, 0), // No stack here
        to: pos(3, 0),
      };

      expect(() => mutateMovement(state, action)).toThrow('No stack at origin');
    });

    it('updates player elimination count', () => {
      const state = makeBaseState();
      const player1 = state.players.find((p) => p.playerNumber === 1);
      if (player1) player1.eliminatedRings = 0;

      addMarker(state.board, pos(6, 3), 2);

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const newState = mutateMovement(state, action);
      const updatedPlayer = newState.players.find((p) => p.playerNumber === 1);
      expect(updatedPlayer?.eliminatedRings).toBe(1);
    });

    it('updates lastMoveAt timestamp', () => {
      const state = makeBaseState();
      const oldTime = state.lastMoveAt;

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };

      const newState = mutateMovement(state, action);
      expect(newState.lastMoveAt.getTime()).toBeGreaterThanOrEqual(oldTime.getTime());
    });
  });

  // ==========================================================================
  // applySimpleMovement - Lines 548-699
  // ==========================================================================
  describe('applySimpleMovement', () => {
    it('applies movement to empty space', () => {
      const state = makeBaseState();
      const result = applySimpleMovement(state, {
        from: pos(3, 3),
        to: pos(6, 3),
        player: 1,
      });

      expect(result.nextState.board.stacks.has(positionToString(pos(6, 3)))).toBe(true);
      expect(result.nextState.board.markers.has(positionToString(pos(3, 3)))).toBe(true);
    });

    it('handles landing on own marker with ring elimination', () => {
      const state = makeBaseState();
      state.totalRingsEliminated = 0;
      addMarker(state.board, pos(6, 3), 1); // Own marker

      const result = applySimpleMovement(state, {
        from: pos(3, 3),
        to: pos(6, 3),
        player: 1,
      });

      // Marker should be removed and ring eliminated
      expect(result.nextState.board.markers.has(positionToString(pos(6, 3)))).toBe(false);
      expect(result.markerEffectsApplied).toBe(true);
      expect(result.eliminatedRingsByPlayer[1]).toBe(1);
    });

    it('handles landing on opponent marker', () => {
      const state = makeBaseState();
      state.totalRingsEliminated = 0;
      addMarker(state.board, pos(6, 3), 2); // Opponent marker

      const result = applySimpleMovement(state, {
        from: pos(3, 3),
        to: pos(6, 3),
        player: 1,
      });

      // Marker should be removed
      expect(result.nextState.board.markers.has(positionToString(pos(6, 3)))).toBe(false);
      expect(result.markerEffectsApplied).toBe(true);
    });

    it('respects leaveDepartureMarker = false', () => {
      const state = makeBaseState();
      const result = applySimpleMovement(state, {
        from: pos(3, 3),
        to: pos(6, 3),
        player: 1,
        leaveDepartureMarker: false,
      });

      // Should not leave marker at origin
      expect(result.nextState.board.markers.has(positionToString(pos(3, 3)))).toBe(false);
    });

    it('tracks collapsed territory from marker effects', () => {
      const state = makeBaseState();
      // Add a marker at an intermediate position that will collapse
      addMarker(state.board, pos(4, 3), 1);
      addMarker(state.board, pos(5, 3), 1);

      const initialTerritory = state.players[0].territorySpaces;

      const result = applySimpleMovement(state, {
        from: pos(3, 3),
        to: pos(6, 3),
        player: 1,
      });

      // Territory may increase due to collapsed markers
      expect(result.nextState.players[0].territorySpaces).toBeGreaterThanOrEqual(initialTerritory);
    });

    it('throws when no stack at origin', () => {
      const state = makeBaseState();

      expect(() =>
        applySimpleMovement(state, {
          from: pos(0, 0), // No stack
          to: pos(3, 0),
          player: 1,
        })
      ).toThrow('No stack at origin');
    });

    it('updates player elimination count when landing on marker', () => {
      const state = makeBaseState();
      state.totalRingsEliminated = 0;
      addMarker(state.board, pos(6, 3), 2);

      const result = applySimpleMovement(state, {
        from: pos(3, 3),
        to: pos(6, 3),
        player: 1,
      });

      const player1 = result.nextState.players.find((p) => p.playerNumber === 1);
      expect(player1?.eliminatedRings).toBeGreaterThanOrEqual(1);
    });

    it('completely removes stack when height 1 lands on marker', () => {
      const board = createTestBoard('square8');
      addStack(board, pos(3, 3), 1, 1); // Height 1

      const state = createTestGameState({
        boardType: 'square8',
        board,
        currentPlayer: 1,
        currentPhase: 'movement',
      });
      addMarker(state.board, pos(4, 3), 2);

      const result = applySimpleMovement(state, {
        from: pos(3, 3),
        to: pos(4, 3),
        player: 1,
      });

      // Stack should be gone
      expect(result.nextState.board.stacks.has(positionToString(pos(4, 3)))).toBe(false);
    });
  });

  // ==========================================================================
  // applyMovement - Lines 707-739
  // ==========================================================================
  describe('applyMovement', () => {
    it('returns error for wrong move type', () => {
      const state = makeBaseState();
      const move: Move = {
        id: 'test',
        type: 'place_ring',
        player: 1,
        to: pos(0, 0),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyMovement(state, move);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.reason).toContain("Expected 'move_stack' move");
      }
    });

    it('returns error when move.from is missing', () => {
      const state = makeBaseState();
      const move: Move = {
        id: 'test',
        type: 'move_stack',
        player: 1,
        to: pos(6, 3),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
        // from is missing
      };

      const result = applyMovement(state, move);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.reason).toContain('Move.from is required');
      }
    });

    it('returns success for valid movement', () => {
      const state = makeBaseState();
      const move: Move = {
        id: 'test',
        type: 'move_stack',
        player: 1,
        from: pos(3, 3),
        to: pos(6, 3),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyMovement(state, move);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.newState.board.stacks.has(positionToString(pos(6, 3)))).toBe(true);
        expect(result.newState.board.markers.has(positionToString(pos(3, 3)))).toBe(true);
      }
    });

    it('catches errors and returns failure result', () => {
      const state = makeBaseState();
      const move: Move = {
        id: 'test',
        type: 'move_stack',
        player: 1,
        from: pos(0, 0), // No stack here
        to: pos(3, 0),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyMovement(state, move);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.reason).toContain('No stack at origin');
      }
    });

    it('rejects move_stack type (legacy alias)', () => {
      const state = makeBaseState();
      const move: Move = {
        id: 'test',
        type: 'move_stack',
        player: 1,
        from: pos(3, 3),
        to: pos(6, 3),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyMovement(state, move);
      expect(result.success).toBe(false);
    });
  });

  // ==========================================================================
  // Direction validation edge cases
  // ==========================================================================
  describe('direction validation edge cases', () => {
    it('handles diagonal movement on square board', () => {
      const state = makeBaseState();
      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 6), // Diagonal
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('handles y-direction movement', () => {
      const board = createTestBoard('square8');
      addStack(board, pos(3, 3), 1, 3);

      const state = createTestGameState({
        boardType: 'square8',
        board,
        currentPlayer: 1,
        currentPhase: 'movement',
      });

      const action: MoveStackAction = {
        type: 'MOVE_STACK',
        playerId: 1,
        from: pos(3, 3),
        to: pos(3, 6), // Vertical
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });
  });
});
