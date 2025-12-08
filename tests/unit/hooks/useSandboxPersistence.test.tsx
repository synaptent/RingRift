import { renderHook, act, waitFor } from '@testing-library/react';
import { useSandboxPersistence } from '../../../src/client/hooks/useSandboxPersistence';
import type { ClientSandboxEngine } from '../../../src/client/services/ClientSandboxEngine';
import type { GameState } from '../../../src/shared/types/game';

// Mock toast
jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

// Mock ReplayService
const mockStoreGame = jest.fn();
jest.mock('../../../src/client/services/ReplayService', () => ({
  getReplayService: () => ({
    storeGame: mockStoreGame,
  }),
}));

// Mock LocalGameStorage
const mockStoreGameLocally = jest.fn();
const mockGetPendingCount = jest.fn();
jest.mock('../../../src/client/services/LocalGameStorage', () => ({
  storeGameLocally: (...args: unknown[]) => mockStoreGameLocally(...args),
  getPendingCount: () => mockGetPendingCount(),
}));

// Mock GameSyncService
const mockSubscribers: Array<(state: { pendingCount: number }) => void> = [];
jest.mock('../../../src/client/services/GameSyncService', () => ({
  GameSyncService: {
    start: jest.fn(),
    stop: jest.fn(),
    subscribe: jest.fn((callback: (state: { pendingCount: number }) => void) => {
      mockSubscribers.push(callback);
      return () => {
        const idx = mockSubscribers.indexOf(callback);
        if (idx >= 0) mockSubscribers.splice(idx, 1);
      };
    }),
  },
}));

describe('useSandboxPersistence', () => {
  // Create mock engine
  const createMockEngine = (
    overrides: {
      gameState?: Partial<GameState>;
      victoryResult?: { winner: number; reason: string } | null;
    } = {}
  ): ClientSandboxEngine => {
    const defaultGameState: GameState = {
      gameId: 'test-game-id',
      board: { type: 'square8', cells: {} },
      players: [
        { number: 1, rings: [], markers: [], victoryCondition: null },
        { number: 2, rings: [], markers: [], victoryCondition: null },
      ],
      currentPlayer: 1,
      currentPhase: 'placement',
      moveHistory: [],
      placedRingCount: 0,
      ...overrides.gameState,
    } as GameState;

    return {
      getGameState: jest.fn().mockReturnValue(defaultGameState),
      getVictoryResult: jest.fn().mockReturnValue(overrides.victoryResult ?? null),
    } as unknown as ClientSandboxEngine;
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockSubscribers.length = 0;
    mockStoreGame.mockReset();
    mockStoreGameLocally.mockReset();
    mockGetPendingCount.mockReset();
  });

  describe('initial state', () => {
    it('returns correct default state with no engine', () => {
      const { result } = renderHook(() =>
        useSandboxPersistence({
          engine: null,
          playerTypes: ['human', 'human'],
          numPlayers: 2,
        })
      );

      expect(result.current.autoSaveGames).toBe(true);
      expect(result.current.gameSaveStatus).toBe('idle');
      expect(result.current.pendingLocalGames).toBe(0);
      expect(result.current.syncState).toBeNull();
      expect(typeof result.current.setAutoSaveGames).toBe('function');
      expect(typeof result.current.cloneInitialGameState).toBe('function');
    });

    it('respects defaultAutoSave option', () => {
      const { result } = renderHook(() =>
        useSandboxPersistence({
          engine: null,
          playerTypes: ['human', 'human'],
          numPlayers: 2,
          defaultAutoSave: false,
        })
      );

      expect(result.current.autoSaveGames).toBe(false);
    });

    it('returns correct default state with engine', () => {
      const mockEngine = createMockEngine();

      const { result } = renderHook(() =>
        useSandboxPersistence({
          engine: mockEngine,
          playerTypes: ['human', 'human'],
          numPlayers: 2,
        })
      );

      expect(result.current.autoSaveGames).toBe(true);
      expect(result.current.gameSaveStatus).toBe('idle');
    });
  });

  describe('setAutoSaveGames', () => {
    it('toggles auto-save setting', () => {
      const { result } = renderHook(() =>
        useSandboxPersistence({
          engine: null,
          playerTypes: ['human', 'human'],
          numPlayers: 2,
        })
      );

      expect(result.current.autoSaveGames).toBe(true);

      act(() => {
        result.current.setAutoSaveGames(false);
      });

      expect(result.current.autoSaveGames).toBe(false);

      act(() => {
        result.current.setAutoSaveGames(true);
      });

      expect(result.current.autoSaveGames).toBe(true);
    });
  });

  describe('initial state capture', () => {
    it('captures initial game state when engine is provided with empty move history', () => {
      const mockEngine = createMockEngine({
        gameState: { moveHistory: [] },
      });

      const { result } = renderHook(() =>
        useSandboxPersistence({
          engine: mockEngine,
          playerTypes: ['human', 'human'],
          numPlayers: 2,
        })
      );

      expect(result.current.initialGameStateRef.current).not.toBeNull();
      expect(result.current.initialGameStateRef.current?.gameId).toBe('test-game-id');
    });

    it('does not overwrite initial state if already captured', () => {
      const mockEngine = createMockEngine({
        gameState: { moveHistory: [] },
      });

      const { result, rerender } = renderHook(() =>
        useSandboxPersistence({
          engine: mockEngine,
          playerTypes: ['human', 'human'],
          numPlayers: 2,
        })
      );

      const capturedState = result.current.initialGameStateRef.current;
      expect(capturedState).not.toBeNull();

      // Re-render shouldn't change the captured state
      rerender();

      expect(result.current.initialGameStateRef.current).toBe(capturedState);
    });
  });

  describe('engine destruction', () => {
    it('resets refs and status when engine becomes null', () => {
      const mockEngine = createMockEngine({
        gameState: { moveHistory: [] },
      });

      const { result, rerender } = renderHook(
        ({ engine }) =>
          useSandboxPersistence({
            engine,
            playerTypes: ['human', 'human'],
            numPlayers: 2,
          }),
        { initialProps: { engine: mockEngine as ClientSandboxEngine | null } }
      );

      // Initial state captured
      expect(result.current.initialGameStateRef.current).not.toBeNull();
      expect(result.current.gameSavedRef.current).toBe(false);

      // Engine destroyed
      rerender({ engine: null });

      expect(result.current.initialGameStateRef.current).toBeNull();
      expect(result.current.gameSavedRef.current).toBe(false);
      expect(result.current.gameSaveStatus).toBe('idle');
    });
  });

  describe('GameSyncService subscription', () => {
    it('subscribes to GameSyncService on mount', () => {
      const { GameSyncService } = jest.requireMock(
        '../../../src/client/services/GameSyncService'
      ) as { GameSyncService: { start: jest.Mock; subscribe: jest.Mock } };

      renderHook(() =>
        useSandboxPersistence({
          engine: null,
          playerTypes: ['human', 'human'],
          numPlayers: 2,
        })
      );

      expect(GameSyncService.start).toHaveBeenCalled();
      expect(GameSyncService.subscribe).toHaveBeenCalled();
    });

    it('updates syncState when GameSyncService emits', async () => {
      const { result } = renderHook(() =>
        useSandboxPersistence({
          engine: null,
          playerTypes: ['human', 'human'],
          numPlayers: 2,
        })
      );

      expect(result.current.syncState).toBeNull();
      expect(result.current.pendingLocalGames).toBe(0);

      // Simulate sync service update
      act(() => {
        mockSubscribers.forEach((cb) => cb({ pendingCount: 5 }));
      });

      expect(result.current.pendingLocalGames).toBe(5);
    });

    it('unsubscribes on unmount', () => {
      const { GameSyncService } = jest.requireMock(
        '../../../src/client/services/GameSyncService'
      ) as { GameSyncService: { stop: jest.Mock } };

      const { unmount } = renderHook(() =>
        useSandboxPersistence({
          engine: null,
          playerTypes: ['human', 'human'],
          numPlayers: 2,
        })
      );

      expect(mockSubscribers.length).toBe(1);

      unmount();

      expect(mockSubscribers.length).toBe(0);
      expect(GameSyncService.stop).toHaveBeenCalled();
    });
  });

  describe('auto-save on victory', () => {
    it('does not save when autoSaveGames is false', async () => {
      // Create engine that simulates game that started and ended (victory)
      // Note: The hook captures initial state when moveHistory is empty,
      // so we need an engine that transitions from empty to victory
      const mockEngine = createMockEngine({
        gameState: { moveHistory: [] },
        victoryResult: null,
      });

      const { rerender } = renderHook(
        ({ victoryResult }) => {
          // Update the mock return value
          (mockEngine.getVictoryResult as jest.Mock).mockReturnValue(victoryResult);
          return useSandboxPersistence({
            engine: mockEngine,
            playerTypes: ['human', 'human'],
            numPlayers: 2,
            defaultAutoSave: false,
          });
        },
        { initialProps: { victoryResult: null as { winner: number; reason: string } | null } }
      );

      // Simulate victory
      rerender({ victoryResult: { winner: 1, reason: 'five_in_row' } });

      await act(async () => {
        await new Promise((r) => setTimeout(r, 50));
      });

      expect(mockStoreGame).not.toHaveBeenCalled();
    });

    it('does not save when no victory result', async () => {
      const mockEngine = createMockEngine({
        gameState: { moveHistory: [] },
        victoryResult: null,
      });

      renderHook(() =>
        useSandboxPersistence({
          engine: mockEngine,
          playerTypes: ['human', 'human'],
          numPlayers: 2,
        })
      );

      await act(async () => {
        await new Promise((r) => setTimeout(r, 50));
      });

      expect(mockStoreGame).not.toHaveBeenCalled();
    });

    it('saves game to server when victory detected', async () => {
      // Create an engine that persists but whose game state changes
      const gameStateMock = {
        gameId: 'test-game-id',
        board: { type: 'square8', cells: {} },
        players: [
          { number: 1, rings: [], markers: [], victoryCondition: null },
          { number: 2, rings: [], markers: [], victoryCondition: null },
        ],
        currentPlayer: 1,
        currentPhase: 'placement',
        moveHistory: [] as unknown[],
        placedRingCount: 0,
      } as GameState;

      const mockEngine = {
        getGameState: jest.fn().mockReturnValue(gameStateMock),
        getVictoryResult: jest.fn().mockReturnValue(null),
      } as unknown as ClientSandboxEngine;

      mockStoreGame.mockResolvedValue({ success: true, totalMoves: 2 });

      const { result, rerender } = renderHook(
        ({ stateVersion }) =>
          useSandboxPersistence({
            engine: mockEngine,
            playerTypes: ['human', 'ai'],
            numPlayers: 2,
            stateVersion,
          }),
        { initialProps: { stateVersion: 0 } }
      );

      // Verify initial state was captured
      expect(result.current.initialGameStateRef.current).not.toBeNull();

      // Simulate game progressing to victory
      gameStateMock.moveHistory = [{ type: 'place_ring' }, { type: 'move' }];
      (mockEngine.getVictoryResult as jest.Mock).mockReturnValue({
        winner: 1,
        reason: 'five_in_row',
      });

      rerender({ stateVersion: 1 });

      await waitFor(() => {
        expect(mockStoreGame).toHaveBeenCalled();
      });
    });

    it('updates status to saved on successful server save', async () => {
      const gameStateMock = {
        gameId: 'test-game-id',
        board: { type: 'square8', cells: {} },
        players: [
          { number: 1, rings: [], markers: [], victoryCondition: null },
          { number: 2, rings: [], markers: [], victoryCondition: null },
        ],
        currentPlayer: 1,
        currentPhase: 'placement',
        moveHistory: [] as unknown[],
        placedRingCount: 0,
      } as GameState;

      const mockEngine = {
        getGameState: jest.fn().mockReturnValue(gameStateMock),
        getVictoryResult: jest.fn().mockReturnValue(null),
      } as unknown as ClientSandboxEngine;

      mockStoreGame.mockResolvedValue({ success: true, totalMoves: 1 });

      const { result, rerender } = renderHook(
        ({ stateVersion }) =>
          useSandboxPersistence({
            engine: mockEngine,
            playerTypes: ['human', 'human'],
            numPlayers: 2,
            stateVersion,
          }),
        { initialProps: { stateVersion: 0 } }
      );

      expect(result.current.gameSaveStatus).toBe('idle');

      // Simulate victory
      gameStateMock.moveHistory = [{ type: 'place_ring' }];
      (mockEngine.getVictoryResult as jest.Mock).mockReturnValue({
        winner: 1,
        reason: 'five_in_row',
      });

      rerender({ stateVersion: 1 });

      await waitFor(() => {
        expect(result.current.gameSaveStatus).toBe('saved');
      });
    });

    it('falls back to local storage when server save fails', async () => {
      const gameStateMock = {
        gameId: 'test-game-id',
        board: { type: 'square8', cells: {} },
        players: [
          { number: 1, rings: [], markers: [], victoryCondition: null },
          { number: 2, rings: [], markers: [], victoryCondition: null },
        ],
        currentPlayer: 1,
        currentPhase: 'placement',
        moveHistory: [] as unknown[],
        placedRingCount: 0,
      } as GameState;

      const mockEngine = {
        getGameState: jest.fn().mockReturnValue(gameStateMock),
        getVictoryResult: jest.fn().mockReturnValue(null),
      } as unknown as ClientSandboxEngine;

      mockStoreGame.mockRejectedValue(new Error('Server unavailable'));
      mockStoreGameLocally.mockResolvedValue({ success: true });
      mockGetPendingCount.mockResolvedValue(1);

      const { result, rerender } = renderHook(
        ({ stateVersion }) =>
          useSandboxPersistence({
            engine: mockEngine,
            playerTypes: ['human', 'human'],
            numPlayers: 2,
            stateVersion,
          }),
        { initialProps: { stateVersion: 0 } }
      );

      // Simulate victory
      gameStateMock.moveHistory = [{ type: 'place_ring' }];
      (mockEngine.getVictoryResult as jest.Mock).mockReturnValue({
        winner: 1,
        reason: 'five_in_row',
      });

      rerender({ stateVersion: 1 });

      await waitFor(() => {
        expect(result.current.gameSaveStatus).toBe('saved-local');
      });

      expect(mockStoreGameLocally).toHaveBeenCalled();
    });

    it('sets error status when both server and local save fail', async () => {
      const gameStateMock = {
        gameId: 'test-game-id',
        board: { type: 'square8', cells: {} },
        players: [
          { number: 1, rings: [], markers: [], victoryCondition: null },
          { number: 2, rings: [], markers: [], victoryCondition: null },
        ],
        currentPlayer: 1,
        currentPhase: 'placement',
        moveHistory: [] as unknown[],
        placedRingCount: 0,
      } as GameState;

      const mockEngine = {
        getGameState: jest.fn().mockReturnValue(gameStateMock),
        getVictoryResult: jest.fn().mockReturnValue(null),
      } as unknown as ClientSandboxEngine;

      mockStoreGame.mockRejectedValue(new Error('Server unavailable'));
      mockStoreGameLocally.mockRejectedValue(new Error('IndexedDB unavailable'));

      const { result, rerender } = renderHook(
        ({ stateVersion }) =>
          useSandboxPersistence({
            engine: mockEngine,
            playerTypes: ['human', 'human'],
            numPlayers: 2,
            stateVersion,
          }),
        { initialProps: { stateVersion: 0 } }
      );

      // Simulate victory
      gameStateMock.moveHistory = [{ type: 'place_ring' }];
      (mockEngine.getVictoryResult as jest.Mock).mockReturnValue({
        winner: 1,
        reason: 'five_in_row',
      });

      rerender({ stateVersion: 1 });

      await waitFor(() => {
        expect(result.current.gameSaveStatus).toBe('error');
      });
    });

    it('does not save twice for the same game (gameSavedRef prevents re-save)', async () => {
      const gameStateMock = {
        gameId: 'test-game-id',
        board: { type: 'square8', cells: {} },
        players: [
          { number: 1, rings: [], markers: [], victoryCondition: null },
          { number: 2, rings: [], markers: [], victoryCondition: null },
        ],
        currentPlayer: 1,
        currentPhase: 'placement',
        moveHistory: [] as unknown[],
        placedRingCount: 0,
      } as GameState;

      const mockEngine = {
        getGameState: jest.fn().mockReturnValue(gameStateMock),
        getVictoryResult: jest.fn().mockReturnValue(null),
      } as unknown as ClientSandboxEngine;

      mockStoreGame.mockResolvedValue({ success: true, totalMoves: 1 });

      const { result, rerender } = renderHook(
        ({ stateVersion }) =>
          useSandboxPersistence({
            engine: mockEngine,
            playerTypes: ['human', 'human'],
            numPlayers: 2,
            stateVersion,
          }),
        { initialProps: { stateVersion: 0 } }
      );

      // Simulate victory
      gameStateMock.moveHistory = [{ type: 'place_ring' }];
      (mockEngine.getVictoryResult as jest.Mock).mockReturnValue({
        winner: 1,
        reason: 'five_in_row',
      });

      rerender({ stateVersion: 1 });

      // Wait for the first save to COMPLETE (status becomes 'saved')
      await waitFor(() => {
        expect(result.current.gameSaveStatus).toBe('saved');
      });

      // Track how many times it was called during initial save
      const callCountAfterFirstSave = mockStoreGame.mock.calls.length;

      // Re-render again with same victory state after save completed
      rerender({ stateVersion: 2 });

      await act(async () => {
        await new Promise((r) => setTimeout(r, 100));
      });

      // Should not have been called again (gameSavedRef prevents re-save)
      expect(mockStoreGame.mock.calls.length).toBe(callCountAfterFirstSave);
    });
  });

  describe('cloneInitialGameState', () => {
    it('creates a deep clone of game state', () => {
      const { result } = renderHook(() =>
        useSandboxPersistence({
          engine: null,
          playerTypes: ['human', 'human'],
          numPlayers: 2,
        })
      );

      const original: GameState = {
        gameId: 'test',
        board: { type: 'square8', cells: { a1: { ring: 1 } } },
        players: [{ number: 1, rings: [], markers: [], victoryCondition: null }],
        currentPlayer: 1,
        currentPhase: 'placement',
        moveHistory: [],
        placedRingCount: 0,
      } as GameState;

      const cloned = result.current.cloneInitialGameState(original);

      // Should be equal in value
      expect(cloned.gameId).toBe(original.gameId);
      expect(cloned.board.cells).toEqual(original.board.cells);

      // But different references
      expect(cloned).not.toBe(original);
      expect(cloned.board).not.toBe(original.board);
      expect(cloned.board.cells).not.toBe(original.board.cells);
    });
  });

  describe('callback stability', () => {
    it('setAutoSaveGames and cloneInitialGameState have stable references', () => {
      const { result, rerender } = renderHook(() =>
        useSandboxPersistence({
          engine: null,
          playerTypes: ['human', 'human'],
          numPlayers: 2,
        })
      );

      const setAutoSaveGames1 = result.current.setAutoSaveGames;
      const cloneInitialGameState1 = result.current.cloneInitialGameState;

      rerender();

      expect(result.current.setAutoSaveGames).toBe(setAutoSaveGames1);
      expect(result.current.cloneInitialGameState).toBe(cloneInitialGameState1);
    });
  });
});
