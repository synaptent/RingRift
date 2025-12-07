/**
 * PlacementAggregate.branchCoverage.test.ts
 *
 * Branch coverage tests for PlacementAggregate.ts targeting uncovered branches:
 * - validatePlacementOnBoard: rings in hand, position validity, collapsed space,
 *   marker blocked, stack occupancy, ring cap, per-cell cap, count validation,
 *   no-dead-placement
 * - validatePlacement: phase check, turn check, player not found, count validation
 * - validateSkipPlacement: phase check, turn check, player not found, controlled stacks
 * - enumeratePlacementPositions: player not found, no rings, board iteration
 * - evaluateSkipPlacementEligibility: phase, turn, player, stacks, legal actions
 * - applyPlacementOnBoard: existing stack vs empty, marker removal
 * - mutatePlacement: player not found, toSpend <= 0, collapsed space invariant
 * - applyPlacementMove: wrong type, player not found, effectiveCount = 0
 */

import {
  validatePlacementOnBoard,
  validatePlacement,
  validateSkipPlacement,
  enumeratePlacementPositions,
  evaluateSkipPlacementEligibility,
  applyPlacementOnBoard,
  mutatePlacement,
  applyPlacementMove,
  type PlacementContext,
} from '../../src/shared/engine/aggregates/PlacementAggregate';
import type {
  GameState,
  PlaceRingAction,
  SkipPlacementAction,
  RingStack,
} from '../../src/shared/engine/types';
import type { Position, BoardType, BoardState, Move, Marker } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

// Helper to create a position
const pos = (x: number, y: number, z?: number): Position =>
  z !== undefined ? { x, y, z } : { x, y };

// Helper to create a minimal board state for testing
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

// Helper to create a minimal game state for testing
function makeGameState(overrides: Partial<GameState> = {}): GameState {
  const defaultState: GameState = {
    id: 'test-game',
    board: makeBoardState(),
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
    currentPhase: 'ring_placement',
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
  rings: number[]
): void {
  const key = positionToString(position);
  const stack: RingStack = {
    position,
    rings,
    stackHeight: rings.length,
    capHeight: rings.filter((r, i) => i === 0 || rings[i - 1] === r).length,
    controllingPlayer,
  };
  state.board.stacks.set(key, stack);
}

// Helper to add a stack to board state directly
function addStackToBoard(
  board: BoardState,
  position: Position,
  controllingPlayer: number,
  rings: number[]
): void {
  const key = positionToString(position);
  const capHeight = calculateSimpleCapHeight(rings);
  const stack: RingStack = {
    position,
    rings,
    stackHeight: rings.length,
    capHeight,
    controllingPlayer,
  };
  board.stacks.set(key, stack);
}

// Simple cap height calculation for tests
function calculateSimpleCapHeight(rings: number[]): number {
  if (rings.length === 0) return 0;
  let count = 1;
  for (let i = 1; i < rings.length; i++) {
    if (rings[i] === rings[0]) count++;
    else break;
  }
  return count;
}

// Helper to add a marker
function addMarker(state: GameState, position: Position, player: number): void {
  const key = positionToString(position);
  const marker: Marker = { position, player, type: 'regular' };
  state.board.markers.set(key, marker);
}

// Helper to add a marker to board state directly
function addMarkerToBoard(board: BoardState, position: Position, player: number): void {
  const key = positionToString(position);
  const marker: Marker = { position, player, type: 'regular' };
  board.markers.set(key, marker);
}

describe('PlacementAggregate branch coverage', () => {
  describe('validatePlacementOnBoard', () => {
    const baseCtx: PlacementContext = {
      boardType: 'square8',
      player: 1,
      ringsInHand: 10,
      ringsPerPlayerCap: 18,
    };

    describe('rings in hand check', () => {
      it('rejects when no rings in hand', () => {
        const board = makeBoardState();
        const ctx = { ...baseCtx, ringsInHand: 0 };

        const result = validatePlacementOnBoard(board, pos(0, 0), 1, ctx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INSUFFICIENT_RINGS');
        expect(result.maxPlacementCount).toBe(0);
      });
    });

    describe('position validity', () => {
      it('rejects position off board', () => {
        const board = makeBoardState();

        const result = validatePlacementOnBoard(board, pos(-1, 0), 1, baseCtx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_POSITION');
      });

      it('rejects position beyond board size', () => {
        const board = makeBoardState();

        const result = validatePlacementOnBoard(board, pos(10, 10), 1, baseCtx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_POSITION');
      });
    });

    describe('collapsed space check', () => {
      it('rejects placement on collapsed space', () => {
        const board = makeBoardState();
        board.collapsedSpaces.set('3,3', 2);

        const result = validatePlacementOnBoard(board, pos(3, 3), 1, baseCtx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('COLLAPSED_SPACE');
      });
    });

    describe('marker blocked check', () => {
      it('rejects placement on marker', () => {
        const board = makeBoardState();
        addMarkerToBoard(board, pos(2, 2), 2);

        const result = validatePlacementOnBoard(board, pos(2, 2), 1, baseCtx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('MARKER_BLOCKED');
      });
    });

    describe('ring capacity checks', () => {
      it('uses provided ringsOnBoard when available', () => {
        const board = makeBoardState();
        // Simulate cap reached via provided ringsOnBoard
        const ctx = { ...baseCtx, ringsOnBoard: 18 };

        const result = validatePlacementOnBoard(board, pos(0, 0), 1, ctx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NO_RINGS_AVAILABLE');
      });

      it('uses provided maxAvailableGlobal when available', () => {
        const board = makeBoardState();
        const ctx = { ...baseCtx, maxAvailableGlobal: 0 };

        const result = validatePlacementOnBoard(board, pos(0, 0), 1, ctx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NO_RINGS_AVAILABLE');
      });

      it('computes ringsOnBoard when not provided', () => {
        const board = makeBoardState();
        // Add stacks to nearly exhaust capacity
        for (let i = 0; i < 6; i++) {
          addStackToBoard(board, pos(i, 0), 1, [1, 1, 1]);
        }
        // Now 18 rings on board, should reject

        const result = validatePlacementOnBoard(board, pos(0, 1), 1, baseCtx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NO_RINGS_AVAILABLE');
      });
    });

    describe('per-cell cap and count validation', () => {
      it('allows up to 3 rings on empty cell', () => {
        const board = makeBoardState();

        const result = validatePlacementOnBoard(board, pos(3, 3), 3, baseCtx);
        expect(result.valid).toBe(true);
        expect(result.maxPlacementCount).toBe(3);
      });

      it('rejects more than 3 rings on empty cell', () => {
        const board = makeBoardState();

        const result = validatePlacementOnBoard(board, pos(3, 3), 4, baseCtx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_COUNT');
        expect(result.maxPlacementCount).toBe(3);
      });

      it('allows only 1 ring on occupied cell', () => {
        const board = makeBoardState();
        addStackToBoard(board, pos(3, 3), 2, [2, 2]);

        const result = validatePlacementOnBoard(board, pos(3, 3), 1, baseCtx);
        expect(result.valid).toBe(true);
        expect(result.maxPlacementCount).toBe(1);
      });

      it('rejects more than 1 ring on occupied cell', () => {
        const board = makeBoardState();
        addStackToBoard(board, pos(3, 3), 2, [2, 2]);

        const result = validatePlacementOnBoard(board, pos(3, 3), 2, baseCtx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_COUNT');
        expect(result.reason).toContain('only place 1 ring');
      });

      it('rejects count less than 1', () => {
        const board = makeBoardState();

        const result = validatePlacementOnBoard(board, pos(3, 3), 0, baseCtx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_COUNT');
      });
    });

    describe('no-dead-placement check', () => {
      it('accepts placement with legal move from resulting stack', () => {
        const board = makeBoardState();

        const result = validatePlacementOnBoard(board, pos(3, 3), 2, baseCtx);
        expect(result.valid).toBe(true);
      });

      it('rejects placement resulting in no legal moves (corner completely blocked)', () => {
        const board = makeBoardState();
        // Block all adjacent cells to corner
        board.collapsedSpaces.set('1,0', 1);
        board.collapsedSpaces.set('0,1', 1);
        board.collapsedSpaces.set('1,1', 1);

        const result = validatePlacementOnBoard(board, pos(0, 0), 1, baseCtx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NO_LEGAL_MOVES');
      });

      it('uses hypothetical stack for existing stack', () => {
        const board = makeBoardState();
        addStackToBoard(board, pos(3, 3), 2, [2]);

        const result = validatePlacementOnBoard(board, pos(3, 3), 1, baseCtx);
        // Stack would be [1, 2], controlled by player 1, height 2
        expect(result.valid).toBe(true);
      });
    });
  });

  describe('validatePlacement', () => {
    describe('phase check', () => {
      it('rejects when not in ring_placement phase', () => {
        const state = makeGameState({ currentPhase: 'movement' });
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 1,
        };

        const result = validatePlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_PHASE');
      });
    });

    describe('turn check', () => {
      it('rejects when not player turn', () => {
        const state = makeGameState({ currentPlayer: 2 });
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 1,
        };

        const result = validatePlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NOT_YOUR_TURN');
      });
    });

    describe('player not found', () => {
      it('rejects when player does not exist (after turn check)', () => {
        const state = makeGameState({ currentPlayer: 99 }); // Make it player 99's turn
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 99,
          position: pos(0, 0),
          count: 1,
        };

        const result = validatePlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('PLAYER_NOT_FOUND');
      });
    });

    describe('count validation', () => {
      it('rejects count <= 0', () => {
        const state = makeGameState();
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 0,
        };

        const result = validatePlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_COUNT');
      });

      it('rejects when count exceeds rings in hand', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 2;
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(0, 0),
          count: 3,
        };

        const result = validatePlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INSUFFICIENT_RINGS');
      });
    });

    describe('valid placement', () => {
      it('accepts valid placement action', () => {
        const state = makeGameState();
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(3, 3),
          count: 2,
        };

        const result = validatePlacement(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('board-level validation propagation', () => {
      it('propagates board-level validation errors', () => {
        const state = makeGameState();
        state.board.collapsedSpaces.set('3,3', 2);
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(3, 3),
          count: 1,
        };

        const result = validatePlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('COLLAPSED_SPACE');
      });
    });
  });

  describe('validateSkipPlacement', () => {
    describe('phase check', () => {
      it('rejects when not in ring_placement phase', () => {
        const state = makeGameState({ currentPhase: 'movement' });
        addStack(state, pos(3, 3), 1, [1, 1]);
        const action: SkipPlacementAction = {
          type: 'SKIP_PLACEMENT',
          playerId: 1,
        };

        const result = validateSkipPlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_PHASE');
      });
    });

    describe('turn check', () => {
      it('rejects when not player turn', () => {
        const state = makeGameState({ currentPlayer: 2 });
        addStack(state, pos(3, 3), 1, [1, 1]);
        const action: SkipPlacementAction = {
          type: 'SKIP_PLACEMENT',
          playerId: 1,
        };

        const result = validateSkipPlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NOT_YOUR_TURN');
      });
    });

    describe('player not found', () => {
      it('rejects when player does not exist (after turn check)', () => {
        const state = makeGameState({ currentPlayer: 99 }); // Make it player 99's turn
        const action: SkipPlacementAction = {
          type: 'SKIP_PLACEMENT',
          playerId: 99,
        };

        const result = validateSkipPlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('PLAYER_NOT_FOUND');
      });
    });

    describe('controlled stacks check', () => {
      it('rejects when player has no controlled stacks', () => {
        const state = makeGameState();
        // No stacks on board
        const action: SkipPlacementAction = {
          type: 'SKIP_PLACEMENT',
          playerId: 1,
        };

        const result = validateSkipPlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NO_CONTROLLED_STACKS');
      });

      it('skips stacks not controlled by player', () => {
        const state = makeGameState();
        addStack(state, pos(3, 3), 2, [2, 2]); // Player 2's stack
        const action: SkipPlacementAction = {
          type: 'SKIP_PLACEMENT',
          playerId: 1,
        };

        const result = validateSkipPlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NO_CONTROLLED_STACKS');
      });

      it('skips stacks with zero height', () => {
        const state = makeGameState();
        const key = positionToString(pos(3, 3));
        state.board.stacks.set(key, {
          position: pos(3, 3),
          rings: [],
          stackHeight: 0,
          capHeight: 0,
          controllingPlayer: 1,
        });
        const action: SkipPlacementAction = {
          type: 'SKIP_PLACEMENT',
          playerId: 1,
        };

        const result = validateSkipPlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NO_CONTROLLED_STACKS');
      });
    });

    describe('legal actions check', () => {
      it('rejects when no legal moves from any controlled stack', () => {
        const state = makeGameState();
        // Place a single-ring stack in corner with all adjacent blocked
        addStack(state, pos(0, 0), 1, [1]);
        state.board.collapsedSpaces.set('1,0', 1);
        state.board.collapsedSpaces.set('0,1', 1);
        state.board.collapsedSpaces.set('1,1', 1);

        const action: SkipPlacementAction = {
          type: 'SKIP_PLACEMENT',
          playerId: 1,
        };

        const result = validateSkipPlacement(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NO_LEGAL_ACTIONS');
      });
    });

    describe('valid skip', () => {
      it('accepts skip when player has controlled stack with legal moves', () => {
        const state = makeGameState();
        addStack(state, pos(3, 3), 1, [1, 1]);

        const action: SkipPlacementAction = {
          type: 'SKIP_PLACEMENT',
          playerId: 1,
        };

        const result = validateSkipPlacement(state, action);
        expect(result.valid).toBe(true);
      });
    });
  });

  describe('enumeratePlacementPositions', () => {
    describe('player not found', () => {
      it('returns empty array when player does not exist', () => {
        const state = makeGameState();
        const positions = enumeratePlacementPositions(state, 99);
        expect(positions).toEqual([]);
      });
    });

    describe('no rings in hand', () => {
      it('returns empty array when player has no rings', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 0;

        const positions = enumeratePlacementPositions(state, 1);
        expect(positions).toEqual([]);
      });
    });

    describe('ring capacity exhausted', () => {
      it('returns empty array when capacity exhausted', () => {
        const state = makeGameState();
        // Fill board with 18 rings for player 1
        for (let i = 0; i < 6; i++) {
          addStack(state, pos(i, 0), 1, [1, 1, 1]);
        }

        const positions = enumeratePlacementPositions(state, 1);
        expect(positions).toEqual([]);
      });
    });

    describe('square board iteration', () => {
      it('finds valid positions on square8 board', () => {
        const state = makeGameState();
        const positions = enumeratePlacementPositions(state, 1);
        expect(positions.length).toBeGreaterThan(0);
        // All positions should be valid
        positions.forEach((p) => {
          expect(p.x).toBeGreaterThanOrEqual(0);
          expect(p.x).toBeLessThan(8);
          expect(p.y).toBeGreaterThanOrEqual(0);
          expect(p.y).toBeLessThan(8);
        });
      });

      it('excludes blocked positions', () => {
        const state = makeGameState();
        state.board.collapsedSpaces.set('3,3', 1);
        addMarker(state, pos(4, 4), 2);

        const positions = enumeratePlacementPositions(state, 1);
        const keys = positions.map((p) => positionToString(p));
        expect(keys).not.toContain('3,3');
        expect(keys).not.toContain('4,4');
      });
    });

    describe('hexagonal board iteration', () => {
      it('finds valid positions on hexagonal board', () => {
        const state = makeGameState();
        state.board.type = 'hexagonal';
        state.board.size = 5; // radius

        const positions = enumeratePlacementPositions(state, 1);
        expect(positions.length).toBeGreaterThan(0);
        // Hex positions should have x, y, z
        positions.forEach((p) => {
          expect(typeof p.z).toBe('number');
          expect(p.x + p.y + p.z).toBe(0); // Hex coordinates should sum to 0
        });
      });
    });
  });

  describe('evaluateSkipPlacementEligibility', () => {
    describe('phase check', () => {
      it('returns ineligible when not in ring_placement phase', () => {
        const state = makeGameState({ currentPhase: 'movement' });
        addStack(state, pos(3, 3), 1, [1, 1]);

        const result = evaluateSkipPlacementEligibility(state, 1);
        expect(result.eligible).toBe(false);
        expect(result.code).toBe('INVALID_PHASE');
      });
    });

    describe('turn check', () => {
      it('returns ineligible when not player turn', () => {
        const state = makeGameState({ currentPlayer: 2 });
        addStack(state, pos(3, 3), 1, [1, 1]);

        const result = evaluateSkipPlacementEligibility(state, 1);
        expect(result.eligible).toBe(false);
        expect(result.code).toBe('NOT_YOUR_TURN');
      });
    });

    describe('player not found', () => {
      it('returns ineligible when player does not exist (after turn check)', () => {
        const state = makeGameState({ currentPlayer: 99 }); // Make it player 99's turn

        const result = evaluateSkipPlacementEligibility(state, 99);
        expect(result.eligible).toBe(false);
        expect(result.code).toBe('PLAYER_NOT_FOUND');
      });
    });

    describe('controlled stacks check', () => {
      it('returns ineligible when no controlled stacks', () => {
        const state = makeGameState();

        const result = evaluateSkipPlacementEligibility(state, 1);
        expect(result.eligible).toBe(false);
        expect(result.code).toBe('NO_CONTROLLED_STACKS');
      });
    });

    describe('legal actions check', () => {
      it('returns ineligible when no legal actions from stacks', () => {
        const state = makeGameState();
        addStack(state, pos(0, 0), 1, [1]);
        // Block all adjacent
        state.board.collapsedSpaces.set('1,0', 1);
        state.board.collapsedSpaces.set('0,1', 1);
        state.board.collapsedSpaces.set('1,1', 1);

        const result = evaluateSkipPlacementEligibility(state, 1);
        expect(result.eligible).toBe(false);
        expect(result.code).toBe('NO_LEGAL_ACTIONS');
      });
    });

    describe('eligible result', () => {
      it('returns eligible when player has controlled stack with moves', () => {
        const state = makeGameState();
        addStack(state, pos(3, 3), 1, [1, 1]);

        const result = evaluateSkipPlacementEligibility(state, 1);
        expect(result.eligible).toBe(true);
      });
    });
  });

  describe('applyPlacementOnBoard', () => {
    describe('empty cell placement', () => {
      it('creates new stack on empty cell', () => {
        const board = makeBoardState();
        const newBoard = applyPlacementOnBoard(board, pos(3, 3), 1, 2);

        const stack = newBoard.stacks.get('3,3');
        expect(stack).toBeDefined();
        expect(stack!.rings).toEqual([1, 1]);
        expect(stack!.stackHeight).toBe(2);
        expect(stack!.controllingPlayer).toBe(1);
      });

      it('handles single ring placement', () => {
        const board = makeBoardState();
        const newBoard = applyPlacementOnBoard(board, pos(3, 3), 2, 1);

        const stack = newBoard.stacks.get('3,3');
        expect(stack!.rings).toEqual([2]);
        expect(stack!.capHeight).toBe(1);
      });
    });

    describe('existing stack placement', () => {
      it('adds rings on top of existing stack', () => {
        const board = makeBoardState();
        addStackToBoard(board, pos(3, 3), 2, [2, 2]);

        const newBoard = applyPlacementOnBoard(board, pos(3, 3), 1, 1);

        const stack = newBoard.stacks.get('3,3');
        expect(stack!.rings).toEqual([1, 2, 2]);
        expect(stack!.stackHeight).toBe(3);
        expect(stack!.controllingPlayer).toBe(1);
      });
    });

    describe('marker removal', () => {
      it('removes marker at placement position', () => {
        const board = makeBoardState();
        addMarkerToBoard(board, pos(3, 3), 2);
        expect(board.markers.has('3,3')).toBe(true);

        const newBoard = applyPlacementOnBoard(board, pos(3, 3), 1, 1);

        expect(newBoard.markers.has('3,3')).toBe(false);
        expect(newBoard.stacks.has('3,3')).toBe(true);
      });
    });

    describe('count handling', () => {
      it('ensures minimum count of 1', () => {
        const board = makeBoardState();
        const newBoard = applyPlacementOnBoard(board, pos(3, 3), 1, 0);

        const stack = newBoard.stacks.get('3,3');
        expect(stack!.rings.length).toBe(1);
      });
    });
  });

  describe('mutatePlacement', () => {
    describe('player not found', () => {
      it('throws when player does not exist', () => {
        const state = makeGameState();
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 99,
          position: pos(3, 3),
          count: 1,
        };

        expect(() => mutatePlacement(state, action)).toThrow('Player not found');
      });
    });

    describe('insufficient rings', () => {
      it('returns state unchanged when toSpend <= 0', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 0;
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(3, 3),
          count: 1,
        };

        const newState = mutatePlacement(state, action);
        expect(newState).toBe(state);
      });
    });

    describe('successful mutation', () => {
      it('decrements ringsInHand', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 10;
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(3, 3),
          count: 2,
        };

        const newState = mutatePlacement(state, action);
        expect(newState.players[0].ringsInHand).toBe(8);
      });

      it('clamps count to available rings', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 1;
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(3, 3),
          count: 3,
        };

        const newState = mutatePlacement(state, action);
        expect(newState.players[0].ringsInHand).toBe(0);
        const stack = newState.board.stacks.get('3,3');
        expect(stack!.rings.length).toBe(1);
      });

      it('updates lastMoveAt', () => {
        const state = makeGameState() as GameState & { lastMoveAt?: Date };
        const before = new Date();
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(3, 3),
          count: 1,
        };

        const newState = mutatePlacement(state, action) as GameState & { lastMoveAt: Date };
        expect(newState.lastMoveAt).toBeInstanceOf(Date);
        expect(newState.lastMoveAt.getTime()).toBeGreaterThanOrEqual(before.getTime());
      });
    });

    describe('collapsed space invariant', () => {
      it('preserves collapsedSpaces if board mutation reduces them', () => {
        const state = makeGameState();
        state.board.collapsedSpaces.set('5,5', 2);
        // Normally placement shouldn't reduce collapsed spaces, but test the defensive check
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: pos(3, 3),
          count: 1,
        };

        const newState = mutatePlacement(state, action);
        // Collapsed spaces should be preserved or greater
        expect(newState.board.collapsedSpaces.size).toBeGreaterThanOrEqual(1);
      });
    });
  });

  describe('applyPlacementMove', () => {
    describe('wrong move type', () => {
      it('throws for non-place_ring move', () => {
        const state = makeGameState();
        const move: Move = {
          id: 'test-move',
          type: 'move_stack',
          player: 1,
          from: pos(0, 0),
          to: pos(2, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => applyPlacementMove(state, move)).toThrow("Expected 'place_ring' move");
      });
    });

    describe('player not found', () => {
      it('throws when player does not exist', () => {
        const state = makeGameState();
        const move: Move = {
          id: 'test-move',
          type: 'place_ring',
          player: 99,
          to: pos(3, 3),
          placementCount: 1,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => applyPlacementMove(state, move)).toThrow('Player not found');
      });
    });

    describe('no rings available', () => {
      it('returns state unchanged with appliedCount 0', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 0;
        const move: Move = {
          id: 'test-move',
          type: 'place_ring',
          player: 1,
          to: pos(3, 3),
          placementCount: 1,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyPlacementMove(state, move);
        expect(result.nextState).toBe(state);
        expect(result.appliedCount).toBe(0);
      });
    });

    describe('successful application', () => {
      it('applies placement and returns correct appliedCount', () => {
        const state = makeGameState();
        const move: Move = {
          id: 'test-move',
          type: 'place_ring',
          player: 1,
          to: pos(3, 3),
          placementCount: 2,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyPlacementMove(state, move);
        expect(result.appliedCount).toBe(2);
        expect(result.nextState.board.stacks.has('3,3')).toBe(true);
      });

      it('defaults placementCount to 1', () => {
        const state = makeGameState();
        const move: Move = {
          id: 'test-move',
          type: 'place_ring',
          player: 1,
          to: pos(3, 3),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyPlacementMove(state, move);
        expect(result.appliedCount).toBe(1);
      });

      it('clamps to available rings', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 2;
        const move: Move = {
          id: 'test-move',
          type: 'place_ring',
          player: 1,
          to: pos(3, 3),
          placementCount: 5,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyPlacementMove(state, move);
        expect(result.appliedCount).toBe(2);
      });
    });
  });
});
