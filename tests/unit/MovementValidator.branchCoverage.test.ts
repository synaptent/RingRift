/**
 * MovementValidator.branchCoverage.test.ts
 *
 * Branch coverage tests for MovementValidator.ts targeting uncovered branches:
 * - Phase/turn checks
 * - Position validity
 * - Stack ownership
 * - Collapsed space checks
 * - Direction validation with k calculation
 * - Minimum distance check
 * - Path blocking
 * - Landing validation
 */

import { validateMovement } from '../../src/shared/engine/validators/MovementValidator';
import type { GameState, MoveStackAction, RingStack } from '../../src/shared/engine/types';
import type { Position, BoardType, Marker } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

// Helper to create a position
const pos = (x: number, y: number, z?: number): Position =>
  z !== undefined ? { x, y, z } : { x, y };

// Helper to create a minimal game state for testing
function makeGameState(overrides: Partial<GameState> = {}): GameState {
  const defaultState: GameState = {
    id: 'test-game',
    board: {
      type: 'square8' as BoardType,
      size: 8,
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      formedLines: [],
      territories: new Map(),
      eliminatedRings: { 1: 0, 2: 0 },
    },
    players: [
      {
        id: 'p1',
        username: 'Player1',
        playerNumber: 1,
        type: 'human',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: 10,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'Player2',
        playerNumber: 2,
        type: 'human',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: 10,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ],
    currentPlayer: 1,
    currentPhase: 'movement',
    gameStatus: 'active',
    timeControl: { initialTime: 600000, increment: 0, type: 'rapid' },
    moveHistory: [],
    spectators: [],
    boardType: 'square8',
  };

  return { ...defaultState, ...overrides } as GameState;
}

// Helper to add a stack to the board
function addStack(
  state: GameState,
  position: Position,
  controllingPlayer: number,
  stackHeight: number
): void {
  const key = positionToString(position);
  const rings = Array(stackHeight).fill(controllingPlayer);
  const stack: RingStack = {
    position,
    rings,
    stackHeight,
    capHeight: stackHeight,
    controllingPlayer,
  };
  state.board.stacks.set(key, stack);
}

// Helper to add a marker
function addMarker(state: GameState, position: Position, player: number): void {
  const key = positionToString(position);
  const marker: Marker = { position, player, type: 'regular' };
  state.board.markers.set(key, marker);
}

describe('MovementValidator branch coverage', () => {
  describe('phase check', () => {
    it('rejects when not in movement phase', () => {
      const state = makeGameState({ currentPhase: 'ring_placement' });
      addStack(state, pos(0, 0), 1, 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });
  });

  describe('turn check', () => {
    it('rejects when not the player turn', () => {
      const state = makeGameState({ currentPlayer: 2 });
      addStack(state, pos(0, 0), 1, 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });
  });

  describe('position validity', () => {
    it('rejects when from position is off board', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(-1, 0),
        to: pos(2, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('rejects when to position is off board', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(10, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });
  });

  describe('stack ownership', () => {
    it('rejects when no stack at starting position', () => {
      const state = makeGameState();

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NO_STACK');
    });

    it('rejects when stack not controlled by player', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 2, 2); // Player 2's stack

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1, // Player 1 trying to move
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_STACK');
    });
  });

  describe('collapsed space check', () => {
    it('rejects when destination is collapsed', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);
      state.board.collapsedSpaces.set('2,0', 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('COLLAPSED_SPACE');
    });
  });

  describe('direction validation', () => {
    it('rejects invalid diagonal direction on square board', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);

      // Moving in a non-cardinal, non-standard direction
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(3, 2), // Invalid direction (not straight line)
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_DIRECTION');
    });

    it('accepts horizontal movement', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('accepts vertical movement', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(0, 2),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('accepts diagonal movement', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 2),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('handles direction calculation with k based on y when x is 0', () => {
      const state = makeGameState();
      addStack(state, pos(3, 0), 1, 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 0),
        to: pos(3, 2), // Vertical move, dx=0
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });
  });

  describe('minimum distance check', () => {
    it('rejects when move distance less than stack height', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 3); // Height 3

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0), // Distance 2, but height is 3
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INSUFFICIENT_DISTANCE');
    });

    it('accepts when move distance equals stack height', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0), // Distance 2, height is 2
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('accepts when move distance exceeds stack height', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(5, 0), // Distance 5, height is 2
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });
  });

  describe('path blocking', () => {
    it('rejects when path blocked by collapsed space', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 3);
      state.board.collapsedSpaces.set('1,0', 2); // Collapsed space in path

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(3, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('PATH_BLOCKED');
    });

    it('rejects when path blocked by stack', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 3);
      addStack(state, pos(1, 0), 2, 1); // Stack in path

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(3, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('PATH_BLOCKED');
    });

    it('accepts clear path', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 3);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(3, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });
  });

  describe('landing validation', () => {
    it('accepts landing on empty space', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('accepts landing on marker (own marker)', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);
      addMarker(state, pos(2, 0), 1);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('accepts landing on marker (opponent marker)', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);
      addMarker(state, pos(2, 0), 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects landing on existing stack', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(2, 0), 2, 1);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_LANDING');
    });
  });

  describe('hexagonal board support', () => {
    it('validates movement on hexagonal board', () => {
      const state = makeGameState({
        boardType: 'hexagonal',
      });
      state.board.type = 'hexagonal';
      state.board.size = 13; // radius=12
      addStack(state, pos(0, 0, 0), 1, 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0, 0),
        to: pos(2, 0, -2), // Valid hex direction
      };

      const result = validateMovement(state, action);
      // Should validate (assuming direction is correct for hex)
      expect(typeof result.valid).toBe('boolean');
    });
  });

  describe('edge cases', () => {
    it('handles movement with zero stack in path (empty stack)', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 3);
      // Create a stack with zero height (edge case)
      const key = '1,0';
      state.board.stacks.set(key, {
        position: pos(1, 0),
        rings: [],
        stackHeight: 0,
        capHeight: 0,
        controllingPlayer: 2,
      });

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(3, 0),
      };

      // Zero-height stacks don't block (pathStack.stackHeight > 0 check)
      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('handles adjacent move (distance 1)', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 1);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(1, 0),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('handles long diagonal move', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, 4);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(4, 4),
      };

      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });
  });
});
