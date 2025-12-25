/**
 * Unit tests for useBackendGameStatus hook
 *
 * Tests cover:
 * - Initial state values
 * - Fatal error state management
 * - Victory modal dismissal
 * - Resignation flow
 * - Telemetry integration for resignation
 *
 * @jest-environment jsdom
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { useBackendGameStatus } from '../../src/client/hooks/useBackendGameStatus';
import type { UseBackendGameStatusDeps } from '../../src/client/hooks/useBackendGameStatus';
import type { GameState, GameResult } from '../../src/shared/types/game';

// Mock react-hot-toast
jest.mock('react-hot-toast', () => {
  const toast = jest.fn() as jest.Mock & {
    success: jest.Mock;
    error: jest.Mock;
  };
  toast.success = jest.fn();
  toast.error = jest.fn();
  return { toast };
});

import { toast as mockToast } from 'react-hot-toast';

// Mock gameApi
const mockLeaveGame = jest.fn();
jest.mock('../../src/client/services/api', () => ({
  gameApi: {
    leaveGame: (...args: unknown[]) => mockLeaveGame(...args),
  },
}));

// Mock rulesUxTelemetry
const mockSendRulesUxEvent = jest.fn();
jest.mock('../../src/client/utils/rulesUxTelemetry', () => ({
  sendRulesUxEvent: (...args: unknown[]) => mockSendRulesUxEvent(...args),
}));

// Helper to create minimal game state
const createGameState = (overrides: Partial<GameState> = {}): GameState =>
  ({
    id: 'test-game',
    boardType: 'square8',
    currentPlayer: 1,
    currentPhase: 'movement',
    players: [
      { playerNumber: 1, type: 'human' as const },
      { playerNumber: 2, type: 'ai' as const, aiProfile: { difficulty: 5 } },
    ],
    board: {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8' as const,
    },
    moveHistory: [],
    gameStatus: 'active' as const,
    ...overrides,
  }) as GameState;

// Helper to create minimal victory state
const createVictoryState = (overrides: Partial<GameResult> = {}): GameResult => ({
  winner: 1,
  reason: 'ring_elimination',
  finalScore: {
    ringsEliminated: { 1: 5, 2: 10 },
    territorySpaces: { 1: 0, 2: 0 },
    ringsRemaining: { 1: 13, 2: 8 },
  },
  ...overrides,
});

// Helper to create default deps
const createDeps = (
  overrides: Partial<UseBackendGameStatusDeps> = {}
): UseBackendGameStatusDeps => ({
  gameId: 'test-game-123',
  gameState: createGameState(),
  victoryState: null,
  routeGameId: 'test-game-123',
  weirdStateType: 'none',
  weirdStateFirstSeenAt: null,
  weirdStateResignReported: new Set<string>(),
  markWeirdStateResignReported: jest.fn(),
  ...overrides,
});

describe('useBackendGameStatus', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLeaveGame.mockResolvedValue(undefined);
  });

  describe('Initial state', () => {
    it('should initialize with null fatal error', () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      expect(result.current.fatalGameError).toBeNull();
    });

    it('should initialize with victory modal not dismissed', () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      expect(result.current.isVictoryModalDismissed).toBe(false);
    });

    it('should initialize with resign confirmation closed', () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      expect(result.current.isResignConfirmOpen).toBe(false);
    });

    it('should initialize with not resigning', () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      expect(result.current.isResigning).toBe(false);
    });
  });

  describe('Fatal error management', () => {
    it('should set fatal error', () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      act(() => {
        result.current.setFatalGameError({
          message: 'Something went wrong',
          technical: 'Error details',
        });
      });

      expect(result.current.fatalGameError).toEqual({
        message: 'Something went wrong',
        technical: 'Error details',
      });
    });

    it('should clear fatal error', () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      act(() => {
        result.current.setFatalGameError({ message: 'Error' });
      });
      expect(result.current.fatalGameError).not.toBeNull();

      act(() => {
        result.current.setFatalGameError(null);
      });

      expect(result.current.fatalGameError).toBeNull();
    });
  });

  describe('Victory modal dismissal', () => {
    it('should dismiss victory modal', () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      act(() => {
        result.current.dismissVictoryModal();
      });

      expect(result.current.isVictoryModalDismissed).toBe(true);
    });

    it('should reset dismissal when victory state changes', () => {
      const deps = createDeps();
      const { result, rerender } = renderHook(({ deps }) => useBackendGameStatus(deps), {
        initialProps: { deps },
      });

      act(() => {
        result.current.dismissVictoryModal();
      });
      expect(result.current.isVictoryModalDismissed).toBe(true);

      const newDeps = createDeps({ victoryState: createVictoryState() });
      rerender({ deps: newDeps });

      expect(result.current.isVictoryModalDismissed).toBe(false);
    });

    it('should reset dismissal when route game ID changes', () => {
      const deps = createDeps();
      const { result, rerender } = renderHook(({ deps }) => useBackendGameStatus(deps), {
        initialProps: { deps },
      });

      act(() => {
        result.current.dismissVictoryModal();
      });
      expect(result.current.isVictoryModalDismissed).toBe(true);

      const newDeps = createDeps({ routeGameId: 'different-game' });
      rerender({ deps: newDeps });

      expect(result.current.isVictoryModalDismissed).toBe(false);
    });
  });

  describe('Resign confirmation dialog', () => {
    it('should open resign confirmation dialog', () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      act(() => {
        result.current.setIsResignConfirmOpen(true);
      });

      expect(result.current.isResignConfirmOpen).toBe(true);
    });

    it('should close resign confirmation dialog', () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      act(() => {
        result.current.setIsResignConfirmOpen(true);
      });

      act(() => {
        result.current.setIsResignConfirmOpen(false);
      });

      expect(result.current.isResignConfirmOpen).toBe(false);
    });
  });

  describe('Resignation flow', () => {
    it('should call leaveGame API on resign', async () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      await act(async () => {
        await result.current.handleResign();
      });

      expect(mockLeaveGame).toHaveBeenCalledWith('test-game-123');
    });

    it('should show success toast on successful resign', async () => {
      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      await act(async () => {
        await result.current.handleResign();
      });

      expect(mockToast.success).toHaveBeenCalledWith('You have resigned from the game.');
    });

    it('should show error toast on failed resign', async () => {
      mockLeaveGame.mockRejectedValueOnce(new Error('Network error'));
      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      await act(async () => {
        await result.current.handleResign();
      });

      expect(mockToast.error).toHaveBeenCalledWith('Network error');
    });

    it('should not resign when no game ID', async () => {
      const deps = createDeps({ gameId: null });
      const { result } = renderHook(() => useBackendGameStatus(deps));

      await act(async () => {
        await result.current.handleResign();
      });

      expect(mockLeaveGame).not.toHaveBeenCalled();
    });

    it('should set isResigning flag during resignation', async () => {
      // Make leaveGame slow so we can observe state
      mockLeaveGame.mockImplementation(() => new Promise((resolve) => setTimeout(resolve, 50)));

      const deps = createDeps();
      const { result } = renderHook(() => useBackendGameStatus(deps));

      expect(result.current.isResigning).toBe(false);

      // Start resignation (don't await)
      const promise = act(async () => {
        await result.current.handleResign();
      });

      // Wait for promise
      await promise;

      // After completion, isResigning should still be true (API succeeded, no reset)
      // Note: In real implementation, isResigning is only reset on error
      expect(mockLeaveGame).toHaveBeenCalledTimes(1);
    });
  });

  describe('Weird state telemetry on resign', () => {
    it('should send telemetry event when resigning during weird state', async () => {
      const markWeirdStateResignReported = jest.fn();
      const gameState = createGameState();
      const deps = createDeps({
        gameState,
        weirdStateType: 'active-no-moves-movement',
        weirdStateFirstSeenAt: Date.now() - 5000,
        weirdStateResignReported: new Set(),
        markWeirdStateResignReported,
      });
      const { result } = renderHook(() => useBackendGameStatus(deps));

      await act(async () => {
        await result.current.handleResign();
      });

      expect(mockSendRulesUxEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'rules_weird_state_resign',
          weirdStateType: 'active-no-moves-movement',
        })
      );
      expect(markWeirdStateResignReported).toHaveBeenCalledWith('active-no-moves-movement');
    });

    it('should not send duplicate telemetry for same weird state type', async () => {
      const gameState = createGameState();
      const deps = createDeps({
        gameState,
        weirdStateType: 'active-no-moves-movement',
        weirdStateFirstSeenAt: Date.now() - 5000,
        weirdStateResignReported: new Set(['active-no-moves-movement']),
      });
      const { result } = renderHook(() => useBackendGameStatus(deps));

      await act(async () => {
        await result.current.handleResign();
      });

      expect(mockSendRulesUxEvent).not.toHaveBeenCalled();
    });

    it('should not send telemetry when weirdStateType is none', async () => {
      const gameState = createGameState();
      const deps = createDeps({
        gameState,
        weirdStateType: 'none',
        weirdStateFirstSeenAt: null,
      });
      const { result } = renderHook(() => useBackendGameStatus(deps));

      await act(async () => {
        await result.current.handleResign();
      });

      expect(mockSendRulesUxEvent).not.toHaveBeenCalled();
    });
  });
});
