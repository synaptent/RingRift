/**
 * historyHelpers.branchCoverage.test.ts
 *
 * Branch coverage tests for historyHelpers.ts targeting all branches:
 * - createHistoryEntry: normalizeMoveNumber, progressBefore/After options
 * - createProgressFromBoardSummary: basic computation
 * - appendHistoryEntryToState: immutable append
 */

import {
  createHistoryEntry,
  createProgressFromBoardSummary,
  appendHistoryEntryToState,
  CreateHistoryEntryOptions,
} from '../../src/shared/engine/historyHelpers';
import type {
  GameState,
  Move,
  GameHistoryEntry,
  ProgressSnapshot,
} from '../../src/shared/types/game';
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

// Helper to create a minimal Move
function makeMove(overrides: Partial<Move> = {}): Move {
  return {
    type: 'place_ring',
    player: 1,
    moveNumber: 1,
    position: pos(0, 0),
    count: 1,
    timestamp: Date.now(),
    ...overrides,
  } as Move;
}

describe('historyHelpers branch coverage', () => {
  describe('createHistoryEntry', () => {
    it('creates basic history entry with default options', () => {
      const before = makeGameState({ currentPhase: 'ring_placement', gameStatus: 'active' });
      const after = makeGameState({ currentPhase: 'movement', gameStatus: 'active' });
      const action = makeMove({ moveNumber: 5 });

      const entry = createHistoryEntry(before, after, action);

      expect(entry.moveNumber).toBe(5); // Uses action.moveNumber
      expect(entry.actor).toBe(1);
      expect(entry.phaseBefore).toBe('ring_placement');
      expect(entry.phaseAfter).toBe('movement');
      expect(entry.statusBefore).toBe('active');
      expect(entry.statusAfter).toBe('active');
    });

    describe('normalizeMoveNumber option', () => {
      it('uses action.moveNumber when normalizeMoveNumber is false', () => {
        const before = makeGameState({ history: [] }); // length 0
        const after = makeGameState();
        const action = makeMove({ moveNumber: 10 });

        const entry = createHistoryEntry(before, after, action, { normalizeMoveNumber: false });

        expect(entry.moveNumber).toBe(10);
        expect(entry.action.moveNumber).toBe(10);
      });

      it('normalizes moveNumber to history.length + 1 when true', () => {
        const existingHistory = [
          { moveNumber: 1 } as GameHistoryEntry,
          { moveNumber: 2 } as GameHistoryEntry,
          { moveNumber: 3 } as GameHistoryEntry,
        ];
        const before = makeGameState({ history: existingHistory }); // length 3
        const after = makeGameState();
        const action = makeMove({ moveNumber: 100 }); // Original move number

        const entry = createHistoryEntry(before, after, action, { normalizeMoveNumber: true });

        expect(entry.moveNumber).toBe(4); // history.length + 1 = 3 + 1
        expect(entry.action.moveNumber).toBe(4);
      });

      it('creates normalized action copy when normalizeMoveNumber is true', () => {
        const before = makeGameState({ history: [] });
        const after = makeGameState();
        const action = makeMove({ moveNumber: 99, player: 2, type: 'move_stack' });

        const entry = createHistoryEntry(before, after, action, { normalizeMoveNumber: true });

        // Action should be a copy with normalized moveNumber
        expect(entry.action).not.toBe(action);
        expect(entry.action.moveNumber).toBe(1);
        expect(entry.action.player).toBe(2);
        expect(entry.action.type).toBe('move_stack');
      });

      it('uses original action when normalizeMoveNumber is false', () => {
        const before = makeGameState();
        const after = makeGameState();
        const action = makeMove({ moveNumber: 5 });

        const entry = createHistoryEntry(before, after, action, { normalizeMoveNumber: false });

        expect(entry.action).toBe(action);
      });

      it('defaults to normalizeMoveNumber false when not specified', () => {
        const before = makeGameState({ history: [{ moveNumber: 1 } as GameHistoryEntry] });
        const after = makeGameState();
        const action = makeMove({ moveNumber: 99 });

        const entry = createHistoryEntry(before, after, action); // No options

        expect(entry.moveNumber).toBe(99); // Not normalized
      });
    });

    describe('progressBefore option', () => {
      it('computes progressBefore when not provided', () => {
        const before = makeGameState();
        before.board.markers.set('0,0', { position: pos(0, 0), player: 1, type: 'regular' });
        before.board.collapsedSpaces.set('1,1', 1);
        const after = makeGameState();
        const action = makeMove();

        const entry = createHistoryEntry(before, after, action);

        expect(entry.progressBefore.markers).toBe(1);
        expect(entry.progressBefore.collapsed).toBe(1);
      });

      it('uses provided progressBefore when specified', () => {
        const before = makeGameState();
        const after = makeGameState();
        const action = makeMove();
        const customProgress: ProgressSnapshot = {
          markers: 99,
          collapsed: 88,
          eliminated: 77,
          S: 264,
        };

        const entry = createHistoryEntry(before, after, action, { progressBefore: customProgress });

        expect(entry.progressBefore).toBe(customProgress);
        expect(entry.progressBefore.markers).toBe(99);
      });
    });

    describe('progressAfter option', () => {
      it('computes progressAfter when not provided', () => {
        const before = makeGameState();
        const after = makeGameState();
        after.board.markers.set('2,2', { position: pos(2, 2), player: 2, type: 'regular' });
        after.board.markers.set('3,3', { position: pos(3, 3), player: 1, type: 'regular' });
        const action = makeMove();

        const entry = createHistoryEntry(before, after, action);

        expect(entry.progressAfter.markers).toBe(2);
      });

      it('uses provided progressAfter when specified', () => {
        const before = makeGameState();
        const after = makeGameState();
        const action = makeMove();
        const customProgress: ProgressSnapshot = {
          markers: 111,
          collapsed: 222,
          eliminated: 333,
          S: 666,
        };

        const entry = createHistoryEntry(before, after, action, { progressAfter: customProgress });

        expect(entry.progressAfter).toBe(customProgress);
        expect(entry.progressAfter.eliminated).toBe(333);
      });
    });

    it('computes board summaries', () => {
      const before = makeGameState();
      before.board.stacks.set('0,0', {
        position: pos(0, 0),
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });
      const after = makeGameState();
      after.board.stacks.set('0,0', {
        position: pos(0, 0),
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });
      const action = makeMove();

      const entry = createHistoryEntry(before, after, action);

      expect(entry.boardBeforeSummary.stacks).toHaveLength(1);
      expect(entry.boardAfterSummary.stacks).toHaveLength(1);
    });

    it('computes state hashes', () => {
      const before = makeGameState();
      const after = makeGameState({ currentPhase: 'movement' });
      const action = makeMove();

      const entry = createHistoryEntry(before, after, action);

      expect(entry.stateHashBefore).toBeDefined();
      expect(entry.stateHashAfter).toBeDefined();
      expect(typeof entry.stateHashBefore).toBe('string');
      expect(entry.stateHashBefore).not.toBe(entry.stateHashAfter);
    });

    it('captures phase and status changes', () => {
      const before = makeGameState({
        currentPhase: 'capture',
        gameStatus: 'active',
      });
      const after = makeGameState({
        currentPhase: 'territory_processing',
        gameStatus: 'completed',
        winner: 1,
      });
      const action = makeMove();

      const entry = createHistoryEntry(before, after, action);

      expect(entry.phaseBefore).toBe('capture');
      expect(entry.phaseAfter).toBe('territory_processing');
      expect(entry.statusBefore).toBe('active');
      expect(entry.statusAfter).toBe('completed');
    });

    it('sets actor from action.player', () => {
      const before = makeGameState();
      const after = makeGameState();
      const action = makeMove({ player: 2 });

      const entry = createHistoryEntry(before, after, action);

      expect(entry.actor).toBe(2);
    });
  });

  describe('createProgressFromBoardSummary', () => {
    it('creates progress snapshot from board summary', () => {
      const boardSummary = {
        stacks: ['0,0:1:2:2', '3,3:2:1:1'],
        markers: ['1,1:1', '2,2:2'],
        collapsedSpaces: ['4,4:1'],
      };

      const progress = createProgressFromBoardSummary(boardSummary, 5);

      expect(progress.markers).toBe(2);
      expect(progress.collapsed).toBe(1);
      expect(progress.eliminated).toBe(5);
      expect(progress.S).toBe(8); // 2 + 1 + 5
    });

    it('handles empty board summary', () => {
      const boardSummary = {
        stacks: [],
        markers: [],
        collapsedSpaces: [],
      };

      const progress = createProgressFromBoardSummary(boardSummary, 0);

      expect(progress.markers).toBe(0);
      expect(progress.collapsed).toBe(0);
      expect(progress.eliminated).toBe(0);
      expect(progress.S).toBe(0);
    });

    it('computes S correctly with various values', () => {
      const boardSummary = {
        stacks: [],
        markers: ['a', 'b', 'c'], // 3 markers
        collapsedSpaces: ['x', 'y'], // 2 collapsed
      };

      const progress = createProgressFromBoardSummary(boardSummary, 10);

      expect(progress.S).toBe(15); // 3 + 2 + 10
    });
  });

  describe('appendHistoryEntryToState', () => {
    it('appends entry to state history', () => {
      const state = makeGameState({ history: [] });
      const entry: GameHistoryEntry = {
        moveNumber: 1,
        action: makeMove(),
        actor: 1,
        phaseBefore: 'ring_placement',
        phaseAfter: 'ring_placement',
        statusBefore: 'active',
        statusAfter: 'active',
        progressBefore: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
        progressAfter: { markers: 1, collapsed: 0, eliminated: 0, S: 1 },
        stateHashBefore: 'hash1',
        stateHashAfter: 'hash2',
        boardBeforeSummary: { stacks: [], markers: [], collapsedSpaces: [] },
        boardAfterSummary: { stacks: [], markers: ['0,0:1'], collapsedSpaces: [] },
      };

      const newState = appendHistoryEntryToState(state, entry);

      expect(newState.history).toHaveLength(1);
      expect(newState.history[0]).toBe(entry);
    });

    it('preserves existing history entries', () => {
      const existingEntry: GameHistoryEntry = {
        moveNumber: 1,
        action: makeMove(),
        actor: 1,
        phaseBefore: 'ring_placement',
        phaseAfter: 'ring_placement',
        statusBefore: 'active',
        statusAfter: 'active',
        progressBefore: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
        progressAfter: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
        stateHashBefore: 'hash1',
        stateHashAfter: 'hash2',
        boardBeforeSummary: { stacks: [], markers: [], collapsedSpaces: [] },
        boardAfterSummary: { stacks: [], markers: [], collapsedSpaces: [] },
      };
      const state = makeGameState({ history: [existingEntry] });
      const newEntry: GameHistoryEntry = {
        moveNumber: 2,
        action: makeMove({ moveNumber: 2 }),
        actor: 2,
        phaseBefore: 'ring_placement',
        phaseAfter: 'movement',
        statusBefore: 'active',
        statusAfter: 'active',
        progressBefore: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
        progressAfter: { markers: 1, collapsed: 0, eliminated: 0, S: 1 },
        stateHashBefore: 'hash2',
        stateHashAfter: 'hash3',
        boardBeforeSummary: { stacks: [], markers: [], collapsedSpaces: [] },
        boardAfterSummary: { stacks: [], markers: ['1,1:2'], collapsedSpaces: [] },
      };

      const newState = appendHistoryEntryToState(state, newEntry);

      expect(newState.history).toHaveLength(2);
      expect(newState.history[0]).toBe(existingEntry);
      expect(newState.history[1]).toBe(newEntry);
    });

    it('does not mutate original state', () => {
      const state = makeGameState({ history: [] });
      const entry: GameHistoryEntry = {
        moveNumber: 1,
        action: makeMove(),
        actor: 1,
        phaseBefore: 'ring_placement',
        phaseAfter: 'ring_placement',
        statusBefore: 'active',
        statusAfter: 'active',
        progressBefore: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
        progressAfter: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
        stateHashBefore: 'hash1',
        stateHashAfter: 'hash2',
        boardBeforeSummary: { stacks: [], markers: [], collapsedSpaces: [] },
        boardAfterSummary: { stacks: [], markers: [], collapsedSpaces: [] },
      };

      appendHistoryEntryToState(state, entry);

      expect(state.history).toHaveLength(0);
    });

    it('returns new state object', () => {
      const state = makeGameState();
      const entry: GameHistoryEntry = {
        moveNumber: 1,
        action: makeMove(),
        actor: 1,
        phaseBefore: 'ring_placement',
        phaseAfter: 'ring_placement',
        statusBefore: 'active',
        statusAfter: 'active',
        progressBefore: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
        progressAfter: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
        stateHashBefore: 'hash1',
        stateHashAfter: 'hash2',
        boardBeforeSummary: { stacks: [], markers: [], collapsedSpaces: [] },
        boardAfterSummary: { stacks: [], markers: [], collapsedSpaces: [] },
      };

      const newState = appendHistoryEntryToState(state, entry);

      expect(newState).not.toBe(state);
      expect(newState.history).not.toBe(state.history);
    });

    it('preserves other state properties', () => {
      const state = makeGameState({
        id: 'my-game',
        currentPlayer: 2,
        currentPhase: 'movement',
      });
      const entry: GameHistoryEntry = {
        moveNumber: 1,
        action: makeMove(),
        actor: 1,
        phaseBefore: 'ring_placement',
        phaseAfter: 'ring_placement',
        statusBefore: 'active',
        statusAfter: 'active',
        progressBefore: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
        progressAfter: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
        stateHashBefore: 'hash1',
        stateHashAfter: 'hash2',
        boardBeforeSummary: { stacks: [], markers: [], collapsedSpaces: [] },
        boardAfterSummary: { stacks: [], markers: [], collapsedSpaces: [] },
      };

      const newState = appendHistoryEntryToState(state, entry);

      expect(newState.id).toBe('my-game');
      expect(newState.currentPlayer).toBe(2);
      expect(newState.currentPhase).toBe('movement');
    });
  });
});
