/**
 * MovementMutator.branchCoverage.test.ts
 *
 * Branch coverage tests for MovementMutator.ts targeting uncovered branches:
 * - No stack at origin (throws error)
 * - Landing on marker (own/opponent marker)
 * - Landing on empty space
 * - EliminatedRings fallback when not initialized
 * - Player found vs not found for elimination tracking
 * - Stack becomes empty after marker landing (height 1)
 * - Stack remains after marker landing (height > 1)
 */

import { mutateMovement } from '../../src/shared/engine/mutators/MovementMutator';
import type { GameState, MoveStackAction, RingStack } from '../../src/shared/engine/types';
import type { Position, BoardType } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';
import { inferTotalRingsInPlay } from '../utils/fixtures';

// Helper to create a position
const pos = (x: number, y: number): Position => ({ x, y });

// Helper to create a minimal GameState
function makeGameState(overrides: Partial<GameState> = {}): GameState {
  const board = {
    type: 'square8' as BoardType,
    size: 8,
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    formedLines: [],
    territories: new Map(),
    eliminatedRings: { 1: 0, 2: 0 },
  };
  const players = [
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
  ];
  return {
    id: 'test-game',
    boardType: 'square8',
    board,
    players,
    currentPlayer: 1,
    currentPhase: 'movement',
    moveHistory: [],
    history: [],
    gameStatus: 'active',
    winner: undefined,
    timeControl: { initialTime: 600000, increment: 0, type: 'rapid' },
    spectators: [],
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: inferTotalRingsInPlay(players, board),
    totalRingsEliminated: 0,
    victoryThreshold: 15,
    territoryVictoryThreshold: 8,
    ...overrides,
  } as GameState;
}

// Helper to add a stack
function addStack(
  state: GameState,
  position: Position,
  controllingPlayer: number,
  rings: number[]
): void {
  const key = positionToString(position);
  const stack: RingStack = {
    position,
    rings,
    stackHeight: rings.length,
    capHeight: rings.length, // Simplified for tests
    controllingPlayer,
  };
  state.board.stacks.set(key, stack);
}

// Helper to add a marker
function addMarker(state: GameState, position: Position, player: number): void {
  const key = positionToString(position);
  state.board.markers.set(key, {
    position,
    player,
    type: 'regular',
  });
}

describe('MovementMutator branch coverage', () => {
  describe('no stack at origin', () => {
    it('throws error when no stack at origin position', () => {
      const state = makeGameState();
      // Don't add any stack

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0), // No stack here
        to: pos(2, 0),
      };

      expect(() => mutateMovement(state, action)).toThrow('MovementMutator: No stack at origin');
    });
  });

  describe('landing on empty space', () => {
    it('moves stack to empty destination', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1, 1]);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(3, 0),
      };

      const result = mutateMovement(state, action);

      // Stack should be at new position
      expect(result.board.stacks.has('0,0')).toBe(false);
      expect(result.board.stacks.has('3,0')).toBe(true);
      const stack = result.board.stacks.get('3,0');
      expect(stack?.rings).toEqual([1, 1, 1]);
      expect(stack?.position).toEqual(pos(3, 0));
    });

    it('places marker at origin after movement', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1]);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = mutateMovement(state, action);

      // Marker should be placed at origin
      expect(result.board.markers.has('0,0')).toBe(true);
      const marker = result.board.markers.get('0,0');
      expect(marker?.player).toBe(1);
      expect(marker?.position).toEqual(pos(0, 0));
    });
  });

  describe('landing on marker', () => {
    it('removes marker when landing on own marker', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1, 1]);
      addMarker(state, pos(3, 0), 1); // Own marker

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(3, 0),
      };

      const result = mutateMovement(state, action);

      // Marker should be removed at destination
      expect(result.board.markers.has('3,0')).toBe(false);
    });

    it('removes marker when landing on opponent marker', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1, 1]);
      addMarker(state, pos(3, 0), 2); // Opponent marker

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(3, 0),
      };

      const result = mutateMovement(state, action);

      // Marker should be removed at destination
      expect(result.board.markers.has('3,0')).toBe(false);
    });

    it('eliminates top ring when landing on marker', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1, 1]); // Top ring is player 1's
      addMarker(state, pos(3, 0), 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(3, 0),
      };

      const result = mutateMovement(state, action);

      // Stack should have one less ring
      const stack = result.board.stacks.get('3,0');
      expect(stack?.rings).toEqual([1, 1]); // Top ring eliminated
      expect(stack?.stackHeight).toBe(2);
    });

    it('updates totalRingsEliminated when landing on marker', () => {
      const state = makeGameState({ totalRingsEliminated: 5 });
      addStack(state, pos(0, 0), 1, [1, 1]);
      addMarker(state, pos(2, 0), 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = mutateMovement(state, action);

      expect((result as { totalRingsEliminated: number }).totalRingsEliminated).toBe(6);
    });

    it('updates board.eliminatedRings for top ring owner', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [2, 1, 1]); // Top ring is player 2's
      addMarker(state, pos(3, 0), 1);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(3, 0),
      };

      const result = mutateMovement(state, action);

      expect(result.board.eliminatedRings[2]).toBe(1);
    });

    it('handles eliminatedRings not initialized for player (fallback to 0)', () => {
      const state = makeGameState();
      delete (state.board.eliminatedRings as Record<number, number>)[1];
      addStack(state, pos(0, 0), 1, [1, 1]); // Top ring is player 1's
      addMarker(state, pos(2, 0), 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = mutateMovement(state, action);

      // Should initialize to 0 and then increment
      expect(result.board.eliminatedRings[1]).toBe(1);
    });

    it('updates player.eliminatedRings when player found', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1]);
      addMarker(state, pos(2, 0), 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = mutateMovement(state, action);

      const player1 = result.players.find((p) => p.playerNumber === 1);
      expect(player1?.eliminatedRings).toBe(1);
    });

    it('handles player not found for elimination tracking', () => {
      const state = makeGameState();
      // Stack with top ring owned by non-existent player
      addStack(state, pos(0, 0), 99, [99, 1]); // Top ring is player 99's (doesn't exist)
      addMarker(state, pos(2, 0), 1);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      // Should not throw
      const result = mutateMovement(state, action);

      // Board elimination should still update
      expect(result.board.eliminatedRings[99]).toBe(1);
      // Player elimination won't update (no player 99)
      expect(result.players.every((p) => p.playerNumber !== 99)).toBe(true);
    });
  });

  describe('stack becomes empty after landing on marker', () => {
    it('removes stack entirely when height was 1', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1]); // Single ring stack
      addMarker(state, pos(1, 0), 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(1, 0),
      };

      const result = mutateMovement(state, action);

      // Stack should be completely removed (no rings left)
      expect(result.board.stacks.has('1,0')).toBe(false);
    });

    it('keeps stack when height > 1 after elimination', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1]); // Height 2 stack
      addMarker(state, pos(2, 0), 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = mutateMovement(state, action);

      // Stack should remain with height 1
      expect(result.board.stacks.has('2,0')).toBe(true);
      const stack = result.board.stacks.get('2,0');
      expect(stack?.stackHeight).toBe(1);
    });
  });

  describe('controlling player updates', () => {
    it('updates controlling player after top ring elimination', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 2, 2]); // Player 1 controls, top is p1
      addMarker(state, pos(3, 0), 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(3, 0),
      };

      const result = mutateMovement(state, action);

      // After eliminating player 1's top ring, player 2's ring is now on top
      const stack = result.board.stacks.get('3,0');
      expect(stack?.controllingPlayer).toBe(2);
      expect(stack?.rings).toEqual([2, 2]);
    });
  });

  describe('state immutability', () => {
    it('does not mutate original state', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1, 1]);
      const originalStackCount = state.board.stacks.size;

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(3, 0),
      };

      mutateMovement(state, action);

      // Original state should be unchanged
      expect(state.board.stacks.size).toBe(originalStackCount);
      expect(state.board.stacks.has('0,0')).toBe(true);
      expect(state.board.stacks.has('3,0')).toBe(false);
    });

    it('returns new state object', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1]);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = mutateMovement(state, action);

      expect(result).not.toBe(state);
      expect(result.board).not.toBe(state.board);
      expect(result.board.stacks).not.toBe(state.board.stacks);
    });
  });

  describe('timestamp updates', () => {
    it('updates lastMoveAt timestamp', () => {
      const oldDate = new Date('2020-01-01');
      const state = makeGameState({ lastMoveAt: oldDate });
      addStack(state, pos(0, 0), 1, [1, 1]);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = mutateMovement(state, action);

      expect((result as { lastMoveAt: Date }).lastMoveAt.getTime()).toBeGreaterThan(
        oldDate.getTime()
      );
    });
  });

  describe('mixed ownership stacks', () => {
    it('handles stack with alternating ownership rings', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 2, 1, 2]); // Alternating
      addMarker(state, pos(4, 0), 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(4, 0),
      };

      const result = mutateMovement(state, action);

      const stack = result.board.stacks.get('4,0');
      expect(stack?.rings).toEqual([2, 1, 2]); // Top removed
      expect(stack?.controllingPlayer).toBe(2); // New top is player 2
    });
  });

  describe('edge cases', () => {
    it('handles movement to adjacent cell', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1]);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(1, 0),
      };

      const result = mutateMovement(state, action);

      expect(result.board.stacks.has('1,0')).toBe(true);
      expect(result.board.markers.has('0,0')).toBe(true);
    });

    it('handles diagonal movement', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1]);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 2),
      };

      const result = mutateMovement(state, action);

      expect(result.board.stacks.has('2,2')).toBe(true);
    });

    it('preserves other stacks on board', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1]);
      addStack(state, pos(5, 5), 2, [2, 2]);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = mutateMovement(state, action);

      // Other stack should still exist
      expect(result.board.stacks.has('5,5')).toBe(true);
    });

    it('preserves other markers on board', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1]);
      addMarker(state, pos(5, 5), 2);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0),
        to: pos(2, 0),
      };

      const result = mutateMovement(state, action);

      // Other marker should still exist
      expect(result.board.markers.has('5,5')).toBe(true);
    });
  });
});
