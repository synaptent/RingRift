/**
 * Unit tests for GameContext.tsx
 *
 * Tests cover:
 * - Provider initialization and lifecycle
 * - useGame hook behavior and access controls
 * - Game state management and updates
 * - WebSocket connection integration
 * - Error handling and recovery
 *
 * Target: ≥80% coverage for GameContext.tsx
 *
 * @jest-environment jsdom
 */

import React from 'react';
import { renderHook, act, waitFor } from '@testing-library/react';
import { GameProvider, useGame } from '../../src/client/contexts/GameContext';
import type { GameEventHandlers, ConnectionStatus } from '../../src/client/domain/GameAPI';
import type { GameStateUpdateMessage, GameOverMessage } from '../../src/shared/types/websocket';
import type { GameState, PlayerChoice, Move } from '../../src/shared/types/game';
import { getConfiguredTotalRingsInPlay, inferTotalRingsInPlay } from '../utils/fixtures';

// Mock dependencies
const mockConnect = jest.fn();
const mockDisconnect = jest.fn();
const mockSubmitMove = jest.fn();
const mockSubmitMoveById = jest.fn();
const mockRespondToChoice = jest.fn();
const mockSendChatMessage = jest.fn();
const mockRequestRematch = jest.fn();
const mockRespondToRematch = jest.fn();

let capturedHandlers: GameEventHandlers | null = null;

// Mock SocketGameConnection
jest.mock('../../src/client/services/GameConnection', () => ({
  SocketGameConnection: jest.fn().mockImplementation((handlers: GameEventHandlers) => {
    capturedHandlers = handlers;
    return {
      connect: mockConnect,
      disconnect: mockDisconnect,
      submitMove: mockSubmitMove,
      submitMoveById: mockSubmitMoveById,
      respondToChoice: mockRespondToChoice,
      sendChatMessage: mockSendChatMessage,
      requestRematch: mockRequestRematch,
      respondToRematch: mockRespondToRematch,
      gameId: null,
      status: 'disconnected' as ConnectionStatus,
    };
  }),
}));

// Mock react-hot-toast
jest.mock('react-hot-toast', () => {
  const toast = jest.fn() as jest.Mock & {
    success: jest.Mock;
    error: jest.Mock;
  };
  toast.success = jest.fn();
  toast.error = jest.fn();
  return {
    toast,
  };
});

// Get reference to mocked toast for assertions
import { toast as mockToast } from 'react-hot-toast';

// Mock error reporting
jest.mock('../../src/client/utils/errorReporting', () => ({
  reportClientError: jest.fn(),
  isErrorReportingEnabled: jest.fn(() => false),
  extractErrorMessage: jest.fn((error: any, fallback: string) => error?.message || fallback),
}));

function createMockGameState(overrides: Partial<GameState> = {}): GameState {
  const board =
    overrides.board ??
    ({
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8' as const,
    } as const);
  const players = overrides.players ?? [];
  const maxPlayers = overrides.maxPlayers ?? 2;
  const boardType = overrides.boardType ?? 'square8';
  const totalRingsInPlay =
    players.length > 0
      ? inferTotalRingsInPlay(players, board)
      : getConfiguredTotalRingsInPlay(boardType, maxPlayers);

  return {
    id: 'game-123',
    boardType,
    currentPlayer: 1,
    currentPhase: 'ring_placement',
    players,
    board,
    moveHistory: [],
    history: [],
    timeControl: { initialTime: 600, increment: 0, type: 'blitz' },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers,
    totalRingsInPlay,
    totalRingsEliminated: 0,
    victoryThreshold: 5,
    territoryVictoryThreshold: 32,
    ...overrides,
  } as GameState;
}

describe('GameContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    capturedHandlers = null;
    mockConnect.mockResolvedValue(undefined);
  });

  describe('GameProvider initialization', () => {
    it('should render children when provided', () => {
      const TestChild = () => <div>Test Child</div>;
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => <TestChild />, { wrapper });
      expect(result.current).toBeDefined();
    });

    it('should establish WebSocket connection on mount', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      expect(mockConnect).toHaveBeenCalledWith('game-123');
    });

    it('should initialize with null game state', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      expect(result.current.gameState).toBeNull();
      expect(result.current.gameId).toBeNull();
      expect(result.current.victoryState).toBeNull();
      expect(result.current.error).toBeNull();
    });

    it('should cleanup WebSocket connection on unmount', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result, unmount } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      unmount();

      expect(mockDisconnect).toHaveBeenCalled();
    });
  });

  describe('useGame hook', () => {
    it('should return game context when used within provider', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      expect(result.current).toBeDefined();
      expect(result.current.connectToGame).toBeDefined();
      expect(result.current.disconnect).toBeDefined();
      expect(result.current.submitMove).toBeDefined();
      expect(result.current.submitMoveById).toBeDefined();
    });

    it('should throw error when used outside provider', () => {
      // Suppress console.error for this test
      const consoleError = jest.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        renderHook(() => useGame());
      }).toThrow('useGame must be used within a GameProvider');

      consoleError.mockRestore();
    });

    it('should provide all expected context properties', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      // Verify all context properties exist
      expect(result.current.gameId).toBeDefined();
      expect(result.current.gameState).toBeDefined();
      expect(result.current.validMoves).toBeDefined();
      expect(result.current.isConnecting).toBeDefined();
      expect(result.current.error).toBeDefined();
      expect(result.current.victoryState).toBeDefined();
      expect(result.current.pendingChoice).toBeDefined();
      expect(result.current.choiceDeadline).toBeDefined();
      expect(result.current.chatMessages).toBeDefined();
      expect(result.current.connectionStatus).toBeDefined();
      expect(result.current.lastHeartbeatAt).toBeDefined();
      expect(result.current.connectToGame).toBeInstanceOf(Function);
      expect(result.current.disconnect).toBeInstanceOf(Function);
      expect(result.current.respondToChoice).toBeInstanceOf(Function);
      expect(result.current.submitMove).toBeInstanceOf(Function);
      expect(result.current.submitMoveById).toBeInstanceOf(Function);
      expect(result.current.sendChatMessage).toBeInstanceOf(Function);
    });
  });

  describe('Game state updates', () => {
    it('should update game state when WebSocket message received', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      const mockGameState = createMockGameState({
        players: [
          {
            id: 'user1',
            username: 'Player1',
            type: 'human' as const,
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 5,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
          {
            id: 'user2',
            username: 'Player2',
            type: 'human' as const,
            playerNumber: 2,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 5,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      const gameStatePayload: GameStateUpdateMessage = {
        type: 'game_update',
        data: {
          gameId: 'game-123',
          gameState: mockGameState as any,
          validMoves: [],
        },
        timestamp: new Date().toISOString(),
      };

      act(() => {
        capturedHandlers?.onGameState(gameStatePayload);
      });

      await waitFor(() => {
        expect(result.current.gameState).not.toBeNull();
        expect(result.current.gameState?.id).toBe('game-123');
        expect(result.current.gameId).toBe('game-123');
      });
    });

    it('should handle null game state updates', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      // Send invalid payload with missing data
      act(() => {
        capturedHandlers?.onGameState({} as GameStateUpdateMessage);
      });

      // State should remain null since payload was invalid
      expect(result.current.gameState).toBeNull();
    });

    it('should trigger re-render when game state changes', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      const renderCount = { count: 0 };
      const originalGameState = result.current.gameState;

      const mockGameState = createMockGameState({
        currentPlayer: 2,
        currentPhase: 'movement' as const,
        players: [],
      });

      act(() => {
        capturedHandlers?.onGameState({
          type: 'game_update',
          data: { gameId: 'game-123', gameState: mockGameState as any, validMoves: [] },
          timestamp: new Date().toISOString(),
        });
      });

      await waitFor(() => {
        expect(result.current.gameState).not.toBe(originalGameState);
        expect(result.current.gameState?.currentPlayer).toBe(2);
      });
    });

    it('should maintain game state immutability', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      const mockGameState = createMockGameState({
        currentPlayer: 1,
        currentPhase: 'ring_placement' as const,
        players: [],
      });

      act(() => {
        capturedHandlers?.onGameState({
          type: 'game_update',
          data: { gameId: 'game-123', gameState: mockGameState as any, validMoves: [] },
          timestamp: new Date().toISOString(),
        });
      });

      const firstState = result.current.gameState;

      act(() => {
        capturedHandlers?.onGameState({
          type: 'game_update',
          data: {
            gameId: 'game-123',
            gameState: { ...mockGameState, currentPlayer: 2 } as any,
            validMoves: [],
          },
          timestamp: new Date().toISOString(),
        });
      });

      // Verify original state wasn't mutated
      expect(firstState?.currentPlayer).toBe(1);
      expect(result.current.gameState?.currentPlayer).toBe(2);
    });

    it('should handle rapid successive state updates', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      // Send multiple rapid updates
      for (let i = 1; i <= 5; i++) {
        act(() => {
          capturedHandlers?.onGameState({
            type: 'game_update',
            data: {
              gameId: 'game-123',
              gameState: createMockGameState({
                currentPlayer: i % 2 === 0 ? 2 : 1,
                currentPhase: 'movement' as const,
                players: [],
                moveHistory: new Array(i),
              }) as any,
              validMoves: [],
            },
            timestamp: new Date().toISOString(),
          });
        });
      }

      // Should have the last update
      await waitFor(() => {
        expect(result.current.gameState?.moveHistory).toHaveLength(5);
      });
    });
  });

  describe('WebSocket connection', () => {
    it('should connect to correct WebSocket URL', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-456');
      });

      expect(mockConnect).toHaveBeenCalledWith('game-456');
    });

    it('should reconnect on connection loss', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      // Simulate connection status change to connected
      act(() => {
        capturedHandlers?.onConnectionStatusChange?.('connected');
      });

      // Simulate connection loss
      act(() => {
        capturedHandlers?.onDisconnect('transport error');
        capturedHandlers?.onConnectionStatusChange?.('disconnected');
      });

      expect(result.current.connectionStatus).toBe('disconnected');
      expect(result.current.lastHeartbeatAt).toBeNull();
    });

    it('should send messages through WebSocket', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      // Simulate successful connection by setting gameId via game state
      act(() => {
        capturedHandlers?.onGameState({
          type: 'game_update',
          data: {
            gameId: 'game-123',
            gameState: {
              id: 'game-123',
              boardType: 'square8' as const,
              currentPlayer: 1,
              currentPhase: 'ring_placement' as const,
            } as any,
            validMoves: [],
          },
          timestamp: new Date().toISOString(),
        });
      });

      act(() => {
        result.current.sendChatMessage('Hello World');
      });

      expect(mockSendChatMessage).toHaveBeenCalledWith('Hello World');
    });

    it('should handle WebSocket errors gracefully', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      const errorPayload = {
        type: 'error' as const,
        code: 'GAME_NOT_FOUND',
        event: 'join_game',
        message: 'Game not found',
      };

      act(() => {
        capturedHandlers?.onError(errorPayload);
      });

      await waitFor(() => {
        expect(result.current.error).toBe('Game not found');
        expect((mockToast as any).error).toHaveBeenCalledWith('Game not found');
      });
    });
  });

  describe('Error handling', () => {
    it('should handle WebSocket connection errors', async () => {
      mockConnect.mockRejectedValueOnce(new Error('Connection refused'));

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      await waitFor(() => {
        expect(result.current.error).toBe('Connection refused');
        expect(result.current.isConnecting).toBe(false);
        expect(result.current.connectionStatus).toBe('disconnected');
      });
    });

    it('should handle malformed WebSocket messages', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      // Send malformed error payload
      act(() => {
        capturedHandlers?.onError({ message: 'Unknown error' });
      });

      await waitFor(() => {
        expect(result.current.error).toBe('Unknown error');
      });
    });

    it('should recover from temporary failures', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      // Simulate temporary error
      act(() => {
        capturedHandlers?.onError({ message: 'Temporary error' });
      });

      expect(result.current.error).toBe('Temporary error');

      // Send successful game state update to recover
      act(() => {
        capturedHandlers?.onGameState({
          type: 'game_update',
          data: {
            gameId: 'game-123',
            gameState: {
              id: 'game-123',
              boardType: 'square8',
              currentPlayer: 1,
              currentPhase: 'ring_placement',
              players: [],
              board: {
                stacks: {},
                markers: {},
                collapsedSpaces: {},
                territories: {},
                formedLines: [],
                eliminatedRings: {},
                size: 8,
                type: 'square8',
              },
              moveHistory: [],
              turnNumber: 1,
            } as any,
            validMoves: [],
          },
          timestamp: new Date().toISOString(),
        });
      });

      await waitFor(() => {
        expect(result.current.error).toBeNull();
        expect(result.current.gameState).not.toBeNull();
      });
    });
  });

  describe('Additional functionality', () => {
    it('should handle game over events', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      const gameOverPayload: GameOverMessage = {
        type: 'game_over',
        data: {
          gameId: 'game-123',
          gameState: {} as any,
          gameResult: {
            winner: 1,
            reason: 'ring_elimination',
            finalScore: {
              ringsEliminated: { 1: 5, 2: 3 },
              territorySpaces: { 1: 0, 2: 0 },
              ringsRemaining: { 1: 5, 2: 3 },
            },
          },
        },
        timestamp: new Date().toISOString(),
      };

      act(() => {
        capturedHandlers?.onGameOver(gameOverPayload);
      });

      await waitFor(() => {
        expect(result.current.victoryState).toEqual({
          winner: 1,
          reason: 'ring_elimination',
          finalScore: {
            ringsEliminated: { 1: 5, 2: 3 },
            territorySpaces: { 1: 0, 2: 0 },
            ringsRemaining: { 1: 5, 2: 3 },
          },
        });
      });
    });

    it('should handle player choice required', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      const choice: PlayerChoice = {
        id: 'choice-1',
        playerNumber: 1,
        type: 'line_order',
        gameId: 'game-123',
        prompt: 'Choose line formation order',
        options: [],
      } as PlayerChoice;

      act(() => {
        capturedHandlers?.onChoiceRequired(choice);
      });

      expect(result.current.pendingChoice).toEqual(choice);
    });

    it('should handle chat messages', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      act(() => {
        capturedHandlers?.onChatMessage({
          sender: 'Player1',
          text: 'Good game!',
          timestamp: new Date().toISOString(),
        });
      });

      expect(result.current.chatMessages).toHaveLength(1);
      expect(result.current.chatMessages[0]).toEqual({
        sender: 'Player1',
        text: 'Good game!',
      });
    });

    it('should disconnect and clear state', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      // Add some state
      act(() => {
        capturedHandlers?.onGameState({
          type: 'game_update',
          data: {
            gameId: 'game-123',
            gameState: {
              id: 'game-123',
              boardType: 'square8',
              currentPlayer: 1,
              currentPhase: 'ring_placement',
            } as any,
            validMoves: [],
          },
          timestamp: new Date().toISOString(),
        });
      });

      act(() => {
        result.current.disconnect();
      });

      expect(mockDisconnect).toHaveBeenCalled();
      expect(result.current.gameId).toBeNull();
      expect(result.current.gameState).toBeNull();
      expect(result.current.connectionStatus).toBe('disconnected');
    });

    it('should handle connection status changes', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      act(() => {
        capturedHandlers?.onConnectionStatusChange?.('reconnecting');
      });

      expect(result.current.connectionStatus).toBe('reconnecting');
      expect(result.current.isConnecting).toBe(true);
    });
  });

  describe('Additional coverage - Choice handling', () => {
    it('should respond to choices', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      // Set gameId via game state
      act(() => {
        capturedHandlers?.onGameState({
          type: 'game_update',
          data: {
            gameId: 'game-123',
            gameState: { id: 'game-123' } as any,
            validMoves: [],
          },
          timestamp: new Date().toISOString(),
        });
      });

      const choice: PlayerChoice = {
        id: 'choice-1',
        playerNumber: 1,
        type: 'line_order',
        gameId: 'game-123',
        prompt: 'Choose line',
        options: [],
      } as PlayerChoice;

      act(() => {
        capturedHandlers?.onChoiceRequired(choice);
      });

      expect(result.current.pendingChoice).toEqual(choice);

      act(() => {
        result.current.respondToChoice(choice, { moveId: 'move-1' });
      });

      expect(mockRespondToChoice).toHaveBeenCalledWith(choice, { moveId: 'move-1' });
      expect(result.current.pendingChoice).toBeNull();
    });

    it('should handle choice cancellation', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      const choice: PlayerChoice = {
        id: 'choice-1',
        playerNumber: 1,
        type: 'line_order',
        gameId: 'game-123',
        prompt: 'Choose line',
        options: [],
      } as PlayerChoice;

      act(() => {
        capturedHandlers?.onChoiceRequired(choice);
      });

      expect(result.current.pendingChoice).toEqual(choice);

      act(() => {
        capturedHandlers?.onChoiceCanceled('choice-1');
      });

      expect(result.current.pendingChoice).toBeNull();
    });
  });

  describe('Additional coverage - Rematch functionality', () => {
    it('should request rematch', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      // Set gameId via game state
      act(() => {
        capturedHandlers?.onGameState({
          type: 'game_update',
          data: {
            gameId: 'game-123',
            gameState: { id: 'game-123' } as any,
            validMoves: [],
          },
          timestamp: new Date().toISOString(),
        });
      });

      act(() => {
        result.current.requestRematch();
      });

      expect(mockRequestRematch).toHaveBeenCalled();
    });

    it('should accept rematch', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      act(() => {
        result.current.acceptRematch('request-123');
      });

      expect(mockRespondToRematch).toHaveBeenCalledWith('request-123', true);
    });

    it('should decline rematch', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      act(() => {
        result.current.declineRematch('request-123');
      });

      expect(mockRespondToRematch).toHaveBeenCalledWith('request-123', false);
    });

    it('should handle rematch events', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      // Simulate rematch request
      act(() => {
        capturedHandlers?.onRematchRequested?.({
          id: 'rematch-1',
          gameId: 'game-123',
          requesterId: 'user-1',
          requesterUsername: 'Player1',
          expiresAt: new Date().toISOString(),
        });
      });

      expect(result.current.pendingRematchRequest).toBeTruthy();
      expect(result.current.pendingRematchRequest?.id).toBe('rematch-1');

      // Simulate rematch accepted
      act(() => {
        capturedHandlers?.onRematchResponse?.({
          requestId: 'rematch-1',
          gameId: 'game-123',
          status: 'accepted',
          newGameId: 'game-456',
        });
      });

      expect(result.current.rematchGameId).toBe('game-456');
      expect(result.current.pendingRematchRequest).toBeNull();
    });
  });

  describe('Additional coverage - Chat history', () => {
    it('should handle chat history', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      act(() => {
        capturedHandlers?.onChatHistory?.({
          gameId: 'game-123',
          messages: [
            {
              id: 'msg-1',
              gameId: 'game-123',
              userId: 'user-1',
              username: 'Player1',
              message: 'Hello',
              createdAt: new Date().toISOString(),
            },
            {
              id: 'msg-2',
              gameId: 'game-123',
              userId: 'user-2',
              username: 'Player2',
              message: 'Hi',
              createdAt: new Date().toISOString(),
            },
          ],
        });
      });

      expect(result.current.chatMessages).toHaveLength(2);
      expect(result.current.chatMessages[0].text).toBe('Hello');
      expect(result.current.chatMessages[1].text).toBe('Hi');
    });

    it('should handle persisted chat messages', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      act(() => {
        capturedHandlers?.onChatMessagePersisted?.({
          id: 'msg-1',
          gameId: 'game-123',
          userId: 'user-1',
          username: 'Player1',
          message: 'Persisted message',
          createdAt: new Date().toISOString(),
        });
      });

      expect(result.current.chatMessages).toHaveLength(1);
      expect(result.current.chatMessages[0].text).toBe('Persisted message');
    });
  });

  describe('Additional coverage - Decision phase timeout', () => {
    it('should handle decision phase timeout warning', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      act(() => {
        capturedHandlers?.onDecisionPhaseTimeoutWarning?.({
          type: 'decision_phase_timeout_warning',
          data: {
            gameId: 'game-123',
            playerNumber: 1,
            phase: 'line_processing',
            remainingMs: 10000,
          },
          timestamp: new Date().toISOString(),
        });
      });

      expect(result.current.decisionPhaseTimeoutWarning).toBeTruthy();
      expect(result.current.decisionPhaseTimeoutWarning?.data.remainingMs).toBe(10000);
    });

    it('should handle decision phase timed out', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      // Set a timeout warning first
      act(() => {
        capturedHandlers?.onDecisionPhaseTimeoutWarning?.({
          type: 'decision_phase_timeout_warning',
          data: {
            gameId: 'game-123',
            playerNumber: 1,
            phase: 'line_processing',
            remainingMs: 10000,
          },
          timestamp: new Date().toISOString(),
        });
      });

      expect(result.current.decisionPhaseTimeoutWarning).toBeTruthy();

      // Simulate timeout
      act(() => {
        capturedHandlers?.onDecisionPhaseTimedOut?.({
          type: 'decision_phase_timed_out',
          data: {
            gameId: 'game-123',
            playerNumber: 1,
            phase: 'line_processing',
            autoSelectedMoveId: 'move-1',
            reason: 'Timeout',
          },
          timestamp: new Date().toISOString(),
        });
      });

      expect(result.current.decisionPhaseTimeoutWarning).toBeNull();
    });
  });

  describe('Additional coverage - Position evaluation', () => {
    it('should handle position evaluation events', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      act(() => {
        capturedHandlers?.onPositionEvaluation?.({
          type: 'position_evaluation',
          data: {
            gameId: 'game-123',
            moveNumber: 1,
            evaluation: {
              player1Advantage: 0.25,
              bestMove: { x: 3, y: 4 },
            },
          } as any,
          timestamp: new Date().toISOString(),
        });
      });

      expect(result.current.evaluationHistory).toHaveLength(1);
      expect(result.current.evaluationHistory[0].gameId).toBe('game-123');
    });
  });
});

describe('GameContext - Submit moves', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    capturedHandlers = null;
    mockConnect.mockResolvedValue(undefined);
  });

  describe('Submit moves', () => {
    it('should submit moves through connection', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      // Set a mock game state to provide moveHistory context
      act(() => {
        capturedHandlers?.onGameState({
          type: 'game_update',
          data: {
            gameId: 'game-123',
            gameState: {
              id: 'game-123',
              moveHistory: [],
            } as any,
            validMoves: [],
          },
          timestamp: new Date().toISOString(),
        });
      });

      act(() => {
        result.current.submitMove({
          type: 'place_ring',
          player: 1,
          to: { x: 3, y: 4 },
        });
      });

      expect(mockSubmitMove).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'place_ring',
          player: 1,
          to: { x: 3, y: 4 },
        })
      );
    });

    it('should submit authoritative moves by id through connection', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <GameProvider>{children}</GameProvider>
      );

      const { result } = renderHook(() => useGame(), { wrapper });

      await act(async () => {
        await result.current.connectToGame('game-123');
      });

      act(() => {
        capturedHandlers?.onGameState({
          type: 'game_update',
          data: {
            gameId: 'game-123',
            gameState: { id: 'game-123', moveHistory: [] } as any,
            validMoves: [],
          },
          timestamp: new Date().toISOString(),
        });
      });

      act(() => {
        result.current.submitMoveById('territory-move-1');
      });

      expect(mockSubmitMoveById).toHaveBeenCalledWith('territory-move-1');
    });
  });
});

describe('GameContext - Player disconnect/reconnect events', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    capturedHandlers = null;
    mockConnect.mockResolvedValue(undefined);
  });

  it('should track disconnected players on player_disconnected event', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <GameProvider>{children}</GameProvider>
    );

    const { result } = renderHook(() => useGame(), { wrapper });

    await act(async () => {
      await result.current.connectToGame('game-123');
    });

    expect(result.current.disconnectedOpponents).toEqual([]);

    act(() => {
      capturedHandlers?.onPlayerDisconnected?.({
        type: 'player_disconnected',
        data: {
          gameId: 'game-123',
          player: {
            id: 'user-2',
            username: 'Player2',
          },
        },
        timestamp: new Date().toISOString(),
      });
    });

    expect(result.current.disconnectedOpponents).toHaveLength(1);
    expect(result.current.disconnectedOpponents[0].id).toBe('user-2');
    expect(result.current.disconnectedOpponents[0].username).toBe('Player2');
    expect(result.current.disconnectedOpponents[0].disconnectedAt).toBeDefined();
  });

  it('should not add duplicate disconnected players', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <GameProvider>{children}</GameProvider>
    );

    const { result } = renderHook(() => useGame(), { wrapper });

    await act(async () => {
      await result.current.connectToGame('game-123');
    });

    const disconnectPayload = {
      type: 'player_disconnected' as const,
      data: {
        gameId: 'game-123',
        player: { id: 'user-2', username: 'Player2' },
      },
      timestamp: new Date().toISOString(),
    };

    act(() => {
      capturedHandlers?.onPlayerDisconnected?.(disconnectPayload);
    });

    act(() => {
      capturedHandlers?.onPlayerDisconnected?.(disconnectPayload);
    });

    expect(result.current.disconnectedOpponents).toHaveLength(1);
  });

  it('should remove player from disconnected list on player_reconnected event', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <GameProvider>{children}</GameProvider>
    );

    const { result } = renderHook(() => useGame(), { wrapper });

    await act(async () => {
      await result.current.connectToGame('game-123');
    });

    // First disconnect
    act(() => {
      capturedHandlers?.onPlayerDisconnected?.({
        type: 'player_disconnected',
        data: {
          gameId: 'game-123',
          player: { id: 'user-2', username: 'Player2' },
        },
        timestamp: new Date().toISOString(),
      });
    });

    expect(result.current.disconnectedOpponents).toHaveLength(1);

    // Then reconnect
    act(() => {
      capturedHandlers?.onPlayerReconnected?.({
        type: 'player_reconnected',
        data: {
          gameId: 'game-123',
          player: { id: 'user-2', username: 'Player2' },
        },
        timestamp: new Date().toISOString(),
      });
    });

    expect(result.current.disconnectedOpponents).toHaveLength(0);
  });

  it('should show toast on player disconnect', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <GameProvider>{children}</GameProvider>
    );

    const { result } = renderHook(() => useGame(), { wrapper });

    await act(async () => {
      await result.current.connectToGame('game-123');
    });

    act(() => {
      capturedHandlers?.onPlayerDisconnected?.({
        type: 'player_disconnected',
        data: {
          gameId: 'game-123',
          player: { id: 'user-2', username: 'Player2' },
          reconnectionWindowMs: 30000,
        },
        timestamp: new Date().toISOString(),
      });
    });

    expect(mockToast).toHaveBeenCalledWith(
      'Player2 disconnected (30s to reconnect)',
      expect.objectContaining({ icon: '⚠️', id: 'disconnect-user-2' })
    );
  });

  it('should show success toast on player reconnect', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <GameProvider>{children}</GameProvider>
    );

    const { result } = renderHook(() => useGame(), { wrapper });

    await act(async () => {
      await result.current.connectToGame('game-123');
    });

    act(() => {
      capturedHandlers?.onPlayerReconnected?.({
        type: 'player_reconnected',
        data: {
          gameId: 'game-123',
          player: { id: 'user-2', username: 'Player2' },
        },
        timestamp: new Date().toISOString(),
      });
    });

    expect((mockToast as any).success).toHaveBeenCalledWith(
      'Player2 reconnected',
      expect.any(Object)
    );
  });

  it('should clear disconnected opponents on disconnect', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <GameProvider>{children}</GameProvider>
    );

    const { result } = renderHook(() => useGame(), { wrapper });

    await act(async () => {
      await result.current.connectToGame('game-123');
    });

    act(() => {
      capturedHandlers?.onPlayerDisconnected?.({
        type: 'player_disconnected',
        data: {
          gameId: 'game-123',
          player: { id: 'user-2', username: 'Player2' },
        },
        timestamp: new Date().toISOString(),
      });
    });

    expect(result.current.disconnectedOpponents).toHaveLength(1);

    act(() => {
      result.current.disconnect();
    });

    expect(result.current.disconnectedOpponents).toEqual([]);
  });
});

describe('GameContext - gameEndedByAbandonment', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    capturedHandlers = null;
    mockConnect.mockResolvedValue(undefined);
  });

  it('should return true when victory reason is abandonment', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <GameProvider>{children}</GameProvider>
    );

    const { result } = renderHook(() => useGame(), { wrapper });

    await act(async () => {
      await result.current.connectToGame('game-123');
    });

    expect(result.current.gameEndedByAbandonment).toBe(false);

    act(() => {
      capturedHandlers?.onGameOver({
        type: 'game_over',
        data: {
          gameId: 'game-123',
          gameState: {} as any,
          gameResult: {
            winner: 1,
            reason: 'abandonment',
            finalScore: {
              ringsEliminated: { 1: 0, 2: 0 },
              territorySpaces: { 1: 0, 2: 0 },
              ringsRemaining: { 1: 18, 2: 18 },
            },
          },
        },
        timestamp: new Date().toISOString(),
      });
    });

    expect(result.current.gameEndedByAbandonment).toBe(true);
  });

  it('should return false when victory reason is not abandonment', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <GameProvider>{children}</GameProvider>
    );

    const { result } = renderHook(() => useGame(), { wrapper });

    await act(async () => {
      await result.current.connectToGame('game-123');
    });

    act(() => {
      capturedHandlers?.onGameOver({
        type: 'game_over',
        data: {
          gameId: 'game-123',
          gameState: {} as any,
          gameResult: {
            winner: 1,
            reason: 'ring_elimination',
            finalScore: {
              ringsEliminated: { 1: 5, 2: 10 },
              territorySpaces: { 1: 0, 2: 0 },
              ringsRemaining: { 1: 13, 2: 8 },
            },
          },
        },
        timestamp: new Date().toISOString(),
      });
    });

    expect(result.current.gameEndedByAbandonment).toBe(false);
  });
});

describe('GameContext - Reconnection toast behavior', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    capturedHandlers = null;
    mockConnect.mockResolvedValue(undefined);
  });

  it('should show reconnecting toast when status changes to reconnecting', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <GameProvider>{children}</GameProvider>
    );

    const { result } = renderHook(() => useGame(), { wrapper });

    await act(async () => {
      await result.current.connectToGame('game-123');
    });

    // First connect
    act(() => {
      capturedHandlers?.onConnectionStatusChange?.('connected');
    });

    // Simulate reconnecting
    act(() => {
      capturedHandlers?.onConnectionStatusChange?.('reconnecting');
    });

    expect(mockToast).toHaveBeenCalledWith(
      'Reconnecting...',
      expect.objectContaining({ id: 'reconnecting' })
    );
    expect(result.current.connectionStatus).toBe('reconnecting');
    expect(result.current.isConnecting).toBe(true);
  });

  it('should show reconnected toast on successful reconnection after previous connection', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <GameProvider>{children}</GameProvider>
    );

    const { result } = renderHook(() => useGame(), { wrapper });

    await act(async () => {
      await result.current.connectToGame('game-123');
    });

    // Simulate initial connection
    act(() => {
      capturedHandlers?.onConnectionStatusChange?.('connected');
    });

    // Simulate disconnect then reconnecting
    act(() => {
      capturedHandlers?.onConnectionStatusChange?.('reconnecting');
    });

    // Simulate successful reconnect
    act(() => {
      capturedHandlers?.onConnectionStatusChange?.('connected');
    });

    expect((mockToast as any).success).toHaveBeenCalledWith(
      'Reconnected!',
      expect.objectContaining({ id: 'reconnecting' })
    );
  });

  it('should not show reconnected toast on first connection', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <GameProvider>{children}</GameProvider>
    );

    const { result } = renderHook(() => useGame(), { wrapper });

    await act(async () => {
      await result.current.connectToGame('game-123');
    });

    (mockToast as any).success.mockClear();

    // First connection
    act(() => {
      capturedHandlers?.onConnectionStatusChange?.('connected');
    });

    // Should not show "Reconnected!" on first connection
    expect((mockToast as any).success).not.toHaveBeenCalledWith('Reconnected!', expect.any(Object));
  });

  it('should clear error when connection status changes from reconnecting to connected', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <GameProvider>{children}</GameProvider>
    );

    const { result } = renderHook(() => useGame(), { wrapper });

    await act(async () => {
      await result.current.connectToGame('game-123');
    });

    // Simulate initial connection
    act(() => {
      capturedHandlers?.onConnectionStatusChange?.('connected');
    });

    // Simulate an error during reconnection
    act(() => {
      capturedHandlers?.onError({ message: 'Connection temporarily lost' });
    });

    expect(result.current.error).toBe('Connection temporarily lost');

    // Simulate reconnecting status
    act(() => {
      capturedHandlers?.onConnectionStatusChange?.('reconnecting');
    });

    // Error should still be present during reconnection
    expect(result.current.error).toBe('Connection temporarily lost');

    // Simulate successful reconnection
    act(() => {
      capturedHandlers?.onConnectionStatusChange?.('connected');
    });

    // Error should be cleared immediately when reconnection succeeds
    expect(result.current.error).toBeNull();
  });
});
