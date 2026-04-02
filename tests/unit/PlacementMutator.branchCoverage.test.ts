/**
 * PlacementMutator.branchCoverage.test.ts
 *
 * Branch coverage tests for PlacementMutator.ts targeting uncovered branches:
 * - applyPlacementOnBoard: existing stack vs empty cell, marker removal
 * - mutatePlacement: player not found, toSpend <= 0
 */

import {
  applyPlacementOnBoard,
  mutatePlacement,
} from '../../src/shared/engine/mutators/PlacementMutator';
import type { GameState, PlaceRingAction, RingStack } from '../../src/shared/engine/types';
import type { Position, BoardType, BoardState } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';
import { inferTotalRingsInPlay } from '../utils/fixtures';

// Helper to create a position
const pos = (x: number, y: number): Position => ({ x, y });

// Helper to create a minimal BoardState
function makeBoardState(overrides: Partial<BoardState> = {}): BoardState {
  return {
    type: 'square8' as BoardType,
    size: 8,
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    formedLines: [],
    territories: new Map(),
    eliminatedRings: { 1: 0, 2: 0 },
    ...overrides,
  };
}

// Helper to create a minimal GameState
function makeGameState(overrides: Partial<GameState> = {}): GameState {
  const board = makeBoardState();
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
    currentPhase: 'ring_placement',
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

// Helper to add a stack to the board
function addStack(
  board: BoardState,
  position: Position,
  controllingPlayer: number,
  rings: number[]
): void {
  const key = positionToString(position);
  const stack: RingStack = {
    position,
    rings,
    stackHeight: rings.length,
    capHeight: rings.length,
    controllingPlayer,
  };
  board.stacks.set(key, stack);
}

// Helper to add a marker to the board
function addMarker(board: BoardState, position: Position, player: number): void {
  const key = positionToString(position);
  board.markers.set(key, {
    position,
    player,
    type: 'regular',
  });
}

describe('PlacementMutator branch coverage', () => {
  describe('applyPlacementOnBoard', () => {
    describe('empty cell placement', () => {
      it('creates new stack on empty cell', () => {
        const board = makeBoardState();

        const result = applyPlacementOnBoard(board, pos(0, 0), 1, 1);

        expect(result.stacks.has('0,0')).toBe(true);
        const stack = result.stacks.get('0,0');
        expect(stack?.rings).toEqual([1]);
        expect(stack?.stackHeight).toBe(1);
        expect(stack?.controllingPlayer).toBe(1);
      });

      it('creates new stack with multiple rings', () => {
        const board = makeBoardState();

        const result = applyPlacementOnBoard(board, pos(0, 0), 2, 3);

        const stack = result.stacks.get('0,0');
        expect(stack?.rings).toEqual([2, 2, 2]);
        expect(stack?.stackHeight).toBe(3);
        expect(stack?.controllingPlayer).toBe(2);
      });

      it('sets position on new stack', () => {
        const board = makeBoardState();

        const result = applyPlacementOnBoard(board, pos(3, 5), 1, 1);

        const stack = result.stacks.get('3,5');
        expect(stack?.position).toEqual(pos(3, 5));
      });
    });

    describe('existing stack placement', () => {
      it('adds rings on top of existing stack', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 2, [2, 2]);

        const result = applyPlacementOnBoard(board, pos(0, 0), 1, 2);

        const stack = result.stacks.get('0,0');
        // New rings go on top (front of array)
        expect(stack?.rings).toEqual([1, 1, 2, 2]);
        expect(stack?.stackHeight).toBe(4);
      });

      it('updates controlling player after placement', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 2, [2, 2]);

        const result = applyPlacementOnBoard(board, pos(0, 0), 1, 1);

        const stack = result.stacks.get('0,0');
        expect(stack?.controllingPlayer).toBe(1); // New top ring
      });

      it('updates capHeight after placement', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 2, [2, 2, 1]);

        const result = applyPlacementOnBoard(board, pos(0, 0), 1, 2);

        const stack = result.stacks.get('0,0');
        // Rings: [1, 1, 2, 2, 1] - capHeight should be calculated correctly
        expect(stack?.capHeight).toBeGreaterThan(0);
      });
    });

    describe('marker removal', () => {
      it('removes marker at placement position', () => {
        const board = makeBoardState();
        addMarker(board, pos(0, 0), 2);

        const result = applyPlacementOnBoard(board, pos(0, 0), 1, 1);

        expect(result.markers.has('0,0')).toBe(false);
      });

      it('preserves other markers', () => {
        const board = makeBoardState();
        addMarker(board, pos(0, 0), 2);
        addMarker(board, pos(5, 5), 1);

        const result = applyPlacementOnBoard(board, pos(0, 0), 1, 1);

        expect(result.markers.has('5,5')).toBe(true);
      });
    });

    describe('count normalization', () => {
      it('treats count 0 as 1', () => {
        const board = makeBoardState();

        const result = applyPlacementOnBoard(board, pos(0, 0), 1, 0);

        const stack = result.stacks.get('0,0');
        expect(stack?.rings).toEqual([1]);
      });

      it('treats negative count as 1', () => {
        const board = makeBoardState();

        const result = applyPlacementOnBoard(board, pos(0, 0), 1, -5);

        const stack = result.stacks.get('0,0');
        expect(stack?.rings).toEqual([1]);
      });
    });

    describe('immutability', () => {
      it('does not mutate original board', () => {
        const board = makeBoardState();
        const originalStackCount = board.stacks.size;

        applyPlacementOnBoard(board, pos(0, 0), 1, 1);

        expect(board.stacks.size).toBe(originalStackCount);
      });

      it('returns new board object', () => {
        const board = makeBoardState();

        const result = applyPlacementOnBoard(board, pos(0, 0), 1, 1);

        expect(result).not.toBe(board);
        expect(result.stacks).not.toBe(board.stacks);
      });
    });
  });

  describe('mutatePlacement', () => {
    describe('player not found', () => {
      it('throws error when player not found', () => {
        const state = makeGameState();

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 99, // Non-existent player
          position: pos(0, 0),
          count: 1,
        };

        expect(() => mutatePlacement(state, action)).toThrow('PlacementMutator: Player not found');
      });
    });

    describe('toSpend <= 0', () => {
      it('returns unchanged state when player has 0 rings', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 0;

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 1,
        };

        const result = mutatePlacement(state, action);

        // State should be returned unchanged
        expect(result).toBe(state);
      });

      it('returns unchanged state when count exceeds rings in hand but min is 0', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 0;

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 5,
        };

        const result = mutatePlacement(state, action);

        expect(result).toBe(state);
      });
    });

    describe('successful placement', () => {
      it('decrements ringsInHand', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 10;

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 3,
        };

        const result = mutatePlacement(state, action);

        const player = result.players.find((p) => p.playerNumber === 1);
        expect(player?.ringsInHand).toBe(7);
      });

      it('clamps count to available rings', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 2;

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 5, // More than available
        };

        const result = mutatePlacement(state, action);

        const player = result.players.find((p) => p.playerNumber === 1);
        expect(player?.ringsInHand).toBe(0); // Used all 2 rings

        const stack = result.board.stacks.get('0,0');
        expect(stack?.rings).toHaveLength(2); // Only 2 rings placed
      });

      it('applies placement to board', () => {
        const state = makeGameState();

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(2, 3),
          count: 1,
        };

        const result = mutatePlacement(state, action);

        expect(result.board.stacks.has('2,3')).toBe(true);
      });

      it('updates lastMoveAt timestamp', () => {
        const oldDate = new Date('2020-01-01');
        const state = makeGameState({ lastMoveAt: oldDate });

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 1,
        };

        const result = mutatePlacement(state, action);

        expect((result as { lastMoveAt: Date }).lastMoveAt.getTime()).toBeGreaterThan(
          oldDate.getTime()
        );
      });

      it('preserves other players state', () => {
        const state = makeGameState();
        state.players[1].ringsInHand = 5;

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 2,
        };

        const result = mutatePlacement(state, action);

        const player2 = result.players.find((p) => p.playerNumber === 2);
        expect(player2?.ringsInHand).toBe(5);
      });
    });

    describe('immutability', () => {
      it('does not mutate original state', () => {
        const state = makeGameState();
        const originalRings = state.players[0].ringsInHand;

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 2,
        };

        mutatePlacement(state, action);

        expect(state.players[0].ringsInHand).toBe(originalRings);
      });

      it('returns new state object', () => {
        const state = makeGameState();

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 1,
        };

        const result = mutatePlacement(state, action);

        expect(result).not.toBe(state);
      });

      it('creates new players array', () => {
        const state = makeGameState();

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 1,
        };

        const result = mutatePlacement(state, action);

        expect(result.players).not.toBe(state.players);
      });

      it('creates new moveHistory array', () => {
        const state = makeGameState();

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 1,
        };

        const result = mutatePlacement(state, action);

        expect(result.moveHistory).not.toBe(state.moveHistory);
      });
    });

    describe('edge cases', () => {
      it('handles placement with count 1', () => {
        const state = makeGameState();

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 1,
        };

        const result = mutatePlacement(state, action);

        const stack = result.board.stacks.get('0,0');
        expect(stack?.rings).toEqual([1]);
      });

      it('handles placement on existing stack', () => {
        const state = makeGameState();
        addStack(state.board, pos(0, 0), 2, [2, 2]);

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 2,
        };

        const result = mutatePlacement(state, action);

        const stack = result.board.stacks.get('0,0');
        expect(stack?.rings).toEqual([1, 1, 2, 2]);
      });

      it('handles player 2 placement', () => {
        const state = makeGameState();

        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 2,
          position: pos(3, 3),
          count: 2,
        };

        const result = mutatePlacement(state, action);

        const stack = result.board.stacks.get('3,3');
        expect(stack?.rings).toEqual([2, 2]);
        expect(stack?.controllingPlayer).toBe(2);
      });
    });
  });
});
