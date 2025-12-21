/**
 * History Helpers Module Unit Tests
 *
 * Tests for shared history entry creation functions including:
 * - createHistoryEntry for generating complete GameHistoryEntry records
 * - createProgressFromBoardSummary for deriving progress snapshots
 * - appendHistoryEntryToState for immutable state updates
 */

import {
  createHistoryEntry,
  createProgressFromBoardSummary,
  appendHistoryEntryToState,
} from '../../src/shared/engine/historyHelpers';
import * as core from '../../src/shared/engine/core';
import type {
  GameState,
  Move,
  GameHistoryEntry,
  GamePhase,
  Coord,
} from '../../src/shared/types/game';

// Mock the core module functions
jest.mock('../../src/shared/engine/core', () => ({
  computeProgressSnapshot: jest.fn(),
  summarizeBoard: jest.fn(),
  hashGameState: jest.fn(),
}));

describe('historyHelpers module', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('createHistoryEntry', () => {
    const createMockState = (overrides: Partial<GameState> = {}): GameState =>
      ({
        currentPhase: 'movement' as GamePhase,
        gameStatus: 'active',
        history: [],
        board: {
          type: 'square8',
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: [],
        },
        ...overrides,
      }) as unknown as GameState;

    const createMockMove = (overrides: Partial<Move> = {}): Move =>
      ({
        type: 'move_stack',
        player: 1,
        moveNumber: 5,
        from: { x: 0, y: 0 } as Coord,
        to: { x: 1, y: 1 } as Coord,
        ...overrides,
      }) as Move;

    beforeEach(() => {
      (core.computeProgressSnapshot as jest.Mock).mockReturnValue({
        markers: 10,
        collapsed: 2,
        eliminated: 3,
        S: 15,
      });
      (core.summarizeBoard as jest.Mock).mockReturnValue({
        markers: [{ x: 0, y: 0 }],
        collapsedSpaces: [{ x: 1, y: 1 }],
      });
      (core.hashGameState as jest.Mock).mockReturnValue('mock-hash');
    });

    it('should create entry with correct moveNumber from action', () => {
      const before = createMockState({ currentPhase: 'ring_placement' as GamePhase });
      const after = createMockState({ currentPhase: 'movement' as GamePhase });
      const action = createMockMove({ moveNumber: 7 });

      const entry = createHistoryEntry(before, after, action);

      expect(entry.moveNumber).toBe(7);
    });

    it('should include the action in the entry', () => {
      const before = createMockState();
      const after = createMockState();
      const action = createMockMove({ type: 'place_ring' });

      const entry = createHistoryEntry(before, after, action);

      expect(entry.action.type).toBe('place_ring');
    });

    it('should set actor from action.player', () => {
      const before = createMockState();
      const after = createMockState();
      const action = createMockMove({ player: 2 });

      const entry = createHistoryEntry(before, after, action);

      expect(entry.actor).toBe(2);
    });

    it('should record phaseBefore and phaseAfter', () => {
      const before = createMockState({ currentPhase: 'ring_placement' as GamePhase });
      const after = createMockState({ currentPhase: 'movement' as GamePhase });
      const action = createMockMove();

      const entry = createHistoryEntry(before, after, action);

      expect(entry.phaseBefore).toBe('ring_placement');
      expect(entry.phaseAfter).toBe('movement');
    });

    it('should record statusBefore and statusAfter', () => {
      const before = createMockState({ gameStatus: 'active' });
      const after = createMockState({ gameStatus: 'completed' });
      const action = createMockMove();

      const entry = createHistoryEntry(before, after, action);

      expect(entry.statusBefore).toBe('active');
      expect(entry.statusAfter).toBe('completed');
    });

    it('should compute progressBefore from before state', () => {
      const before = createMockState();
      const after = createMockState();
      const action = createMockMove();

      const entry = createHistoryEntry(before, after, action);

      expect(core.computeProgressSnapshot).toHaveBeenCalledWith(before);
      expect(entry.progressBefore).toEqual({
        markers: 10,
        collapsed: 2,
        eliminated: 3,
        S: 15,
      });
    });

    it('should compute progressAfter from after state', () => {
      const before = createMockState();
      const after = createMockState();
      const action = createMockMove();

      (core.computeProgressSnapshot as jest.Mock)
        .mockReturnValueOnce({ markers: 10, collapsed: 2, eliminated: 3, S: 15 })
        .mockReturnValueOnce({ markers: 11, collapsed: 3, eliminated: 4, S: 18 });

      const entry = createHistoryEntry(before, after, action);

      expect(entry.progressAfter).toEqual({
        markers: 11,
        collapsed: 3,
        eliminated: 4,
        S: 18,
      });
    });

    it('should use custom progressBefore when provided', () => {
      const before = createMockState();
      const after = createMockState();
      const action = createMockMove();
      const customProgress = { markers: 100, collapsed: 50, eliminated: 25, S: 175 };

      const entry = createHistoryEntry(before, after, action, {
        progressBefore: customProgress,
      });

      expect(entry.progressBefore).toEqual(customProgress);
    });

    it('should use custom progressAfter when provided', () => {
      const before = createMockState();
      const after = createMockState();
      const action = createMockMove();
      const customProgress = { markers: 200, collapsed: 100, eliminated: 50, S: 350 };

      const entry = createHistoryEntry(before, after, action, {
        progressAfter: customProgress,
      });

      expect(entry.progressAfter).toEqual(customProgress);
    });

    it('should compute board summaries', () => {
      const before = createMockState();
      const after = createMockState();
      const action = createMockMove();

      const entry = createHistoryEntry(before, after, action);

      expect(core.summarizeBoard).toHaveBeenCalledWith(before.board);
      expect(core.summarizeBoard).toHaveBeenCalledWith(after.board);
      expect(entry.boardBeforeSummary).toBeDefined();
      expect(entry.boardAfterSummary).toBeDefined();
    });

    it('should compute state hashes', () => {
      const before = createMockState();
      const after = createMockState();
      const action = createMockMove();

      (core.hashGameState as jest.Mock)
        .mockReturnValueOnce('before-hash')
        .mockReturnValueOnce('after-hash');

      const entry = createHistoryEntry(before, after, action);

      expect(entry.stateHashBefore).toBe('before-hash');
      expect(entry.stateHashAfter).toBe('after-hash');
    });

    it('should normalize moveNumber when option is true', () => {
      const before = createMockState({
        history: [{} as GameHistoryEntry, {} as GameHistoryEntry],
      });
      const after = createMockState();
      const action = createMockMove({ moveNumber: 999 }); // Original moveNumber ignored

      const entry = createHistoryEntry(before, after, action, {
        normalizeMoveNumber: true,
      });

      // history.length = 2, so normalized moveNumber = 3
      expect(entry.moveNumber).toBe(3);
      expect(entry.action.moveNumber).toBe(3);
    });

    it('should not normalize moveNumber when option is false (default)', () => {
      const before = createMockState({
        history: [{} as GameHistoryEntry, {} as GameHistoryEntry],
      });
      const after = createMockState();
      const action = createMockMove({ moveNumber: 999 });

      const entry = createHistoryEntry(before, after, action, {
        normalizeMoveNumber: false,
      });

      expect(entry.moveNumber).toBe(999);
      expect(entry.action.moveNumber).toBe(999);
    });

    it('should not mutate the original action when normalizing', () => {
      const before = createMockState({ history: [] });
      const after = createMockState();
      const action = createMockMove({ moveNumber: 100 });

      createHistoryEntry(before, after, action, { normalizeMoveNumber: true });

      expect(action.moveNumber).toBe(100); // Original unchanged
    });
  });

  describe('createProgressFromBoardSummary', () => {
    it('should create progress with markers from summary', () => {
      const boardSummary = {
        markers: [
          { x: 0, y: 0 },
          { x: 1, y: 1 },
          { x: 2, y: 2 },
        ],
        collapsedSpaces: [],
      };

      const progress = createProgressFromBoardSummary(boardSummary, 5);

      expect(progress.markers).toBe(3);
    });

    it('should create progress with collapsed from summary', () => {
      const boardSummary = {
        markers: [],
        collapsedSpaces: [
          { x: 0, y: 0 },
          { x: 1, y: 1 },
        ],
      };

      const progress = createProgressFromBoardSummary(boardSummary, 0);

      expect(progress.collapsed).toBe(2);
    });

    it('should use provided eliminatedRings', () => {
      const boardSummary = {
        markers: [],
        collapsedSpaces: [],
      };

      const progress = createProgressFromBoardSummary(boardSummary, 7);

      expect(progress.eliminated).toBe(7);
    });

    it('should compute S-invariant correctly', () => {
      const boardSummary = {
        markers: [
          { x: 0, y: 0 },
          { x: 1, y: 1 },
        ], // 2
        collapsedSpaces: [{ x: 2, y: 2 }], // 1
      };

      const progress = createProgressFromBoardSummary(boardSummary, 3);

      expect(progress.S).toBe(6); // 2 + 1 + 3
    });

    it('should handle empty board summary', () => {
      const boardSummary = {
        markers: [],
        collapsedSpaces: [],
      };

      const progress = createProgressFromBoardSummary(boardSummary, 0);

      expect(progress.markers).toBe(0);
      expect(progress.collapsed).toBe(0);
      expect(progress.eliminated).toBe(0);
      expect(progress.S).toBe(0);
    });

    it('should handle large counts', () => {
      const markers = Array(100).fill({ x: 0, y: 0 });
      const collapsed = Array(50).fill({ x: 0, y: 0 });
      const boardSummary = {
        markers,
        collapsedSpaces: collapsed,
      };

      const progress = createProgressFromBoardSummary(boardSummary, 25);

      expect(progress.markers).toBe(100);
      expect(progress.collapsed).toBe(50);
      expect(progress.eliminated).toBe(25);
      expect(progress.S).toBe(175);
    });
  });

  describe('appendHistoryEntryToState', () => {
    const createMockState = (history: GameHistoryEntry[] = []): GameState =>
      ({
        gameStatus: 'active',
        currentPhase: 'movement',
        history,
        board: {},
      }) as unknown as GameState;

    const createMockEntry = (moveNumber: number): GameHistoryEntry =>
      ({
        moveNumber,
        action: { type: 'move_stack' } as Move,
        actor: 1,
        phaseBefore: 'movement',
        phaseAfter: 'movement',
      }) as unknown as GameHistoryEntry;

    it('should return new state with entry appended', () => {
      const state = createMockState([]);
      const entry = createMockEntry(1);

      const newState = appendHistoryEntryToState(state, entry);

      expect(newState.history).toHaveLength(1);
      expect(newState.history[0]).toBe(entry);
    });

    it('should not mutate original state', () => {
      const state = createMockState([createMockEntry(1)]);
      const entry = createMockEntry(2);

      const newState = appendHistoryEntryToState(state, entry);

      expect(state.history).toHaveLength(1);
      expect(newState.history).toHaveLength(2);
    });

    it('should create new history array reference', () => {
      const state = createMockState([]);
      const entry = createMockEntry(1);

      const newState = appendHistoryEntryToState(state, entry);

      expect(newState.history).not.toBe(state.history);
    });

    it('should create new state reference', () => {
      const state = createMockState([]);
      const entry = createMockEntry(1);

      const newState = appendHistoryEntryToState(state, entry);

      expect(newState).not.toBe(state);
    });

    it('should preserve other state properties', () => {
      const state = {
        gameStatus: 'active',
        currentPhase: 'capture',
        currentPlayer: 2,
        history: [],
        board: { type: 'hex' },
      } as unknown as GameState;
      const entry = createMockEntry(1);

      const newState = appendHistoryEntryToState(state, entry);

      expect(newState.gameStatus).toBe('active');
      expect(newState.currentPhase).toBe('capture');
      expect(newState.currentPlayer).toBe(2);
      expect(newState.board).toBe(state.board);
    });

    it('should handle appending to existing history', () => {
      const existing = [createMockEntry(1), createMockEntry(2)];
      const state = createMockState(existing);
      const entry = createMockEntry(3);

      const newState = appendHistoryEntryToState(state, entry);

      expect(newState.history).toHaveLength(3);
      expect(newState.history[0]).toBe(existing[0]);
      expect(newState.history[1]).toBe(existing[1]);
      expect(newState.history[2]).toBe(entry);
    });
  });
});
