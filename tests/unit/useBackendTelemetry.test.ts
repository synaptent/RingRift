/**
 * Unit tests for useBackendTelemetry hook
 *
 * Tests cover:
 * - Initial state values
 * - Mark weird state resign reported
 * - Get weird state context
 * - Calibration event handling
 *
 * Note: The hook uses refs internally for weird state tracking, so rerender-based
 * tests for weirdStateType changes are not straightforward. We test the API contract
 * of the hook (markWeirdStateResignReported, getWeirdStateContext) and the
 * calibration event emission logic.
 *
 * @jest-environment jsdom
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { useBackendTelemetry } from '../../src/client/hooks/useBackendTelemetry';
import type { GameState, GameResult } from '../../src/shared/types/game';

// Mock getWeirdStateBanner
jest.mock('../../src/client/utils/gameStateWeirdness', () => ({
  getWeirdStateBanner: jest.fn(() => ({ type: 'none' })),
}));

import { getWeirdStateBanner } from '../../src/client/utils/gameStateWeirdness';
const mockGetWeirdStateBanner = getWeirdStateBanner as jest.Mock;

// Mock weirdStateReasons
jest.mock('../../src/shared/engine/weirdStateReasons', () => ({
  isSurfaceableWeirdStateType: jest.fn((type) => type !== 'none'),
}));

// Mock difficultyCalibrationTelemetry
const mockSendDifficultyCalibrationEvent = jest.fn();
const mockGetDifficultyCalibrationSession = jest.fn();
const mockClearDifficultyCalibrationSession = jest.fn();
jest.mock('../../src/client/utils/difficultyCalibrationTelemetry', () => ({
  sendDifficultyCalibrationEvent: (...args: unknown[]) =>
    mockSendDifficultyCalibrationEvent(...args),
  getDifficultyCalibrationSession: (...args: unknown[]) =>
    mockGetDifficultyCalibrationSession(...args),
  clearDifficultyCalibrationSession: (...args: unknown[]) =>
    mockClearDifficultyCalibrationSession(...args),
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
      { playerNumber: 2, type: 'ai' as const },
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

describe('useBackendTelemetry', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetWeirdStateBanner.mockReturnValue({ type: 'none' });
    mockGetDifficultyCalibrationSession.mockReturnValue(null);
  });

  describe('Initial state', () => {
    it('should initialize with weirdStateType as none', () => {
      const { result } = renderHook(() => useBackendTelemetry(null, null, 'game-123'));

      expect(result.current.weirdStateType).toBe('none');
    });

    it('should initialize with null weirdStateFirstSeenAt', () => {
      const { result } = renderHook(() => useBackendTelemetry(null, null, 'game-123'));

      expect(result.current.weirdStateFirstSeenAt).toBeNull();
    });

    it('should initialize with empty weirdStateResignReported set', () => {
      const { result } = renderHook(() => useBackendTelemetry(null, null, 'game-123'));

      expect(result.current.weirdStateResignReported).toBeInstanceOf(Set);
      expect(result.current.weirdStateResignReported.size).toBe(0);
    });

    it('should initialize with isCalibrationEventReported as false', () => {
      const { result } = renderHook(() => useBackendTelemetry(null, null, 'game-123'));

      expect(result.current.isCalibrationEventReported).toBe(false);
    });

    it('should return null for getWeirdStateContext when no weird state', () => {
      const { result } = renderHook(() => useBackendTelemetry(null, null, 'game-123'));

      expect(result.current.getWeirdStateContext()).toBeNull();
    });
  });

  describe('markWeirdStateResignReported', () => {
    it('should add weird state type to reported set', () => {
      const { result } = renderHook(() => useBackendTelemetry(null, null, 'game-123'));

      act(() => {
        result.current.markWeirdStateResignReported('active-no-moves-movement');
      });

      expect(result.current.weirdStateResignReported.has('active-no-moves-movement')).toBe(true);
    });

    it('should handle multiple weird state types', () => {
      const { result } = renderHook(() => useBackendTelemetry(null, null, 'game-123'));

      act(() => {
        result.current.markWeirdStateResignReported('active-no-moves-movement');
        result.current.markWeirdStateResignReported('forced-elimination');
      });

      expect(result.current.weirdStateResignReported.has('active-no-moves-movement')).toBe(true);
      expect(result.current.weirdStateResignReported.has('forced-elimination')).toBe(true);
      expect(result.current.weirdStateResignReported.size).toBe(2);
    });

    it('should not duplicate entries', () => {
      const { result } = renderHook(() => useBackendTelemetry(null, null, 'game-123'));

      act(() => {
        result.current.markWeirdStateResignReported('active-no-moves-movement');
        result.current.markWeirdStateResignReported('active-no-moves-movement');
      });

      expect(result.current.weirdStateResignReported.size).toBe(1);
    });
  });

  describe('getWeirdStateContext', () => {
    it('should return null when no weird state is tracked', () => {
      mockGetWeirdStateBanner.mockReturnValue({ type: 'none' });
      const { result } = renderHook(() => useBackendTelemetry(null, null, 'game-123'));

      expect(result.current.getWeirdStateContext()).toBeNull();
    });

    it('should be a stable function reference', () => {
      const { result, rerender } = renderHook(() => useBackendTelemetry(null, null, 'game-123'));

      const firstRef = result.current.getWeirdStateContext;
      rerender();
      const secondRef = result.current.getWeirdStateContext;

      expect(firstRef).toBe(secondRef);
    });
  });

  describe('Calibration event handling', () => {
    it('should not emit event when no victory state', () => {
      const gameState = createGameState();
      mockGetDifficultyCalibrationSession.mockReturnValue({
        isCalibrationOptIn: true,
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 5,
      });

      renderHook(() => useBackendTelemetry(gameState, null, 'game-123'));

      expect(mockSendDifficultyCalibrationEvent).not.toHaveBeenCalled();
    });

    it('should not emit event when no calibration session', () => {
      const gameState = createGameState();
      const victoryState = createVictoryState();
      mockGetDifficultyCalibrationSession.mockReturnValue(null);

      renderHook(() => useBackendTelemetry(gameState, victoryState, 'game-123'));

      expect(mockSendDifficultyCalibrationEvent).not.toHaveBeenCalled();
    });

    it('should not emit event when calibration session is not opt-in', () => {
      const gameState = createGameState();
      const victoryState = createVictoryState();
      mockGetDifficultyCalibrationSession.mockReturnValue({
        isCalibrationOptIn: false,
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 5,
      });

      renderHook(() => useBackendTelemetry(gameState, victoryState, 'game-123'));

      expect(mockSendDifficultyCalibrationEvent).not.toHaveBeenCalled();
    });

    it('should emit calibration event on victory with calibration session', async () => {
      const gameState = createGameState({
        moveHistory: [{}, {}, {}] as any[], // 3 moves
      });
      const victoryState = createVictoryState({ winner: 1 });
      mockGetDifficultyCalibrationSession.mockReturnValue({
        isCalibrationOptIn: true,
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 5,
      });

      renderHook(() => useBackendTelemetry(gameState, victoryState, 'game-123'));

      await waitFor(() => {
        expect(mockSendDifficultyCalibrationEvent).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'difficulty_calibration_game_completed',
            boardType: 'square8',
            numPlayers: 2,
            difficulty: 5,
            isCalibrationOptIn: true,
          })
        );
      });
    });

    it('should clear calibration session after emitting event', async () => {
      const gameState = createGameState();
      const victoryState = createVictoryState();
      mockGetDifficultyCalibrationSession.mockReturnValue({
        isCalibrationOptIn: true,
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 5,
      });

      renderHook(() => useBackendTelemetry(gameState, victoryState, 'game-123'));

      await waitFor(() => {
        expect(mockClearDifficultyCalibrationSession).toHaveBeenCalledWith('game-123');
      });
    });

    it('should set result to win when human player wins', async () => {
      const gameState = createGameState();
      const victoryState = createVictoryState({ winner: 1 });
      mockGetDifficultyCalibrationSession.mockReturnValue({
        isCalibrationOptIn: true,
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 5,
      });

      renderHook(() => useBackendTelemetry(gameState, victoryState, 'game-123'));

      await waitFor(() => {
        expect(mockSendDifficultyCalibrationEvent).toHaveBeenCalledWith(
          expect.objectContaining({
            result: 'win',
          })
        );
      });
    });

    it('should set result to loss when AI wins', async () => {
      const gameState = createGameState();
      const victoryState = createVictoryState({ winner: 2 });
      mockGetDifficultyCalibrationSession.mockReturnValue({
        isCalibrationOptIn: true,
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 5,
      });

      renderHook(() => useBackendTelemetry(gameState, victoryState, 'game-123'));

      await waitFor(() => {
        expect(mockSendDifficultyCalibrationEvent).toHaveBeenCalledWith(
          expect.objectContaining({
            result: 'loss',
          })
        );
      });
    });

    it('should set result to draw for draw games', async () => {
      const gameState = createGameState();
      const victoryState = createVictoryState({ reason: 'draw', winner: undefined as any });
      mockGetDifficultyCalibrationSession.mockReturnValue({
        isCalibrationOptIn: true,
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 5,
      });

      renderHook(() => useBackendTelemetry(gameState, victoryState, 'game-123'));

      await waitFor(() => {
        expect(mockSendDifficultyCalibrationEvent).toHaveBeenCalledWith(
          expect.objectContaining({
            result: 'draw',
          })
        );
      });
    });

    it('should set result to abandoned for abandonments', async () => {
      const gameState = createGameState();
      const victoryState = createVictoryState({ reason: 'abandonment' });
      mockGetDifficultyCalibrationSession.mockReturnValue({
        isCalibrationOptIn: true,
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 5,
      });

      renderHook(() => useBackendTelemetry(gameState, victoryState, 'game-123'));

      await waitFor(() => {
        expect(mockSendDifficultyCalibrationEvent).toHaveBeenCalledWith(
          expect.objectContaining({
            result: 'abandoned',
          })
        );
      });
    });

    it('should include movesPlayed when moveHistory exists', async () => {
      const gameState = createGameState({
        moveHistory: [{}, {}, {}, {}, {}] as any[], // 5 moves
      });
      const victoryState = createVictoryState();
      mockGetDifficultyCalibrationSession.mockReturnValue({
        isCalibrationOptIn: true,
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 5,
      });

      renderHook(() => useBackendTelemetry(gameState, victoryState, 'game-123'));

      await waitFor(() => {
        expect(mockSendDifficultyCalibrationEvent).toHaveBeenCalledWith(
          expect.objectContaining({
            movesPlayed: 5,
          })
        );
      });
    });
  });
});
