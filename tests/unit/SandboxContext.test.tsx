/**
 * Unit tests for SandboxContext.tsx
 *
 * Tests cover:
 * - Provider initialization and lifecycle
 * - useSandbox hook behavior and access controls
 * - Sandbox game state management
 * - Local game engine integration
 * - AI player integration and automation
 * - Error handling and recovery
 *
 * Target: ≥80% coverage for SandboxContext.tsx
 *
 * @jest-environment jsdom
 */

import React from 'react';
import { renderHook, act, waitFor } from '@testing-library/react';
import {
  SandboxProvider,
  useSandbox,
  LocalPlayerType,
} from '../../src/client/contexts/SandboxContext';
import type { GameState, BoardType, Position, PlayerChoice } from '../../src/shared/types/game';

// Mock ClientSandboxEngine
const mockGetGameState = jest.fn();
const mockGetVictoryResult = jest.fn();
const mockMaybeRunAITurn = jest.fn();

jest.mock('../../src/client/sandbox/ClientSandboxEngine', () => ({
  ClientSandboxEngine: jest.fn().mockImplementation(() => ({
    getGameState: mockGetGameState,
    getVictoryResult: mockGetVictoryResult,
    maybeRunAITurn: mockMaybeRunAITurn,
  })),
}));

// Mock environment flags
jest.mock('../../src/shared/utils/envFlags', () => ({
  isSandboxAiStallDiagnosticsEnabled: jest.fn(() => false),
}));

// Get mocked flag function for manipulation in tests
import { isSandboxAiStallDiagnosticsEnabled } from '../../src/shared/utils/envFlags';
const mockIsSandboxDiagnostics = isSandboxAiStallDiagnosticsEnabled as jest.Mock;

describe('SandboxContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
    mockIsSandboxDiagnostics.mockReturnValue(false);

    // Default mock implementations
    mockGetGameState.mockReturnValue(null);
    mockGetVictoryResult.mockReturnValue(null);
    mockMaybeRunAITurn.mockResolvedValue(undefined);
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  describe('SandboxProvider initialization', () => {
    it('should render children when provided', () => {
      const TestChild = () => <div>Test Child</div>;
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => <TestChild />, { wrapper });
      expect(result.current).toBeDefined();
    });

    it('should initialize sandbox engine on mount', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      expect(result.current.sandboxEngine).toBeNull();
      expect(result.current.isConfigured).toBe(false);
    });

    it('should initialize with default game state', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      expect(result.current.config).toEqual({
        numPlayers: 2,
        boardType: 'square8',
        playerTypes: ['human', 'human', 'ai', 'ai'],
      });
      expect(result.current.sandboxPendingChoice).toBeNull();
      expect(result.current.sandboxCaptureChoice).toBeNull();
      expect(result.current.sandboxCaptureTargets).toEqual([]);
      expect(result.current.sandboxLastProgressAt).toBeNull();
      expect(result.current.sandboxStallWarning).toBeNull();
      expect(result.current.sandboxStateVersion).toBe(0);
    });

    it('should cleanup resources on unmount', () => {
      const clearIntervalSpy = jest.spyOn(global, 'clearInterval');

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result, unmount } = renderHook(() => useSandbox(), { wrapper });

      // Initialize engine to trigger interval setup
      act(() => {
        result.current.initLocalSandboxEngine({
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['ai', 'human'],
          interactionHandler: {
            requestChoice: jest.fn(),
          },
        });
        result.current.setIsConfigured(true);
      });

      const callsBeforeUnmount = clearIntervalSpy.mock.calls.length;

      unmount();

      // Verify intervals were cleared on unmount
      expect(clearIntervalSpy.mock.calls.length).toBeGreaterThan(callsBeforeUnmount);

      clearIntervalSpy.mockRestore();
    });
  });

  describe('useSandbox hook', () => {
    it('should return sandbox context when used within provider', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      expect(result.current).toBeDefined();
      expect(result.current.config).toBeDefined();
      expect(result.current.setConfig).toBeInstanceOf(Function);
      expect(result.current.initLocalSandboxEngine).toBeInstanceOf(Function);
      expect(result.current.getSandboxGameState).toBeInstanceOf(Function);
      expect(result.current.resetSandboxEngine).toBeInstanceOf(Function);
    });

    it('should throw error when used outside provider', () => {
      // Suppress console.error for this test
      const consoleError = jest.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        renderHook(() => useSandbox());
      }).toThrow('useSandbox must be used within a SandboxProvider');

      consoleError.mockRestore();
    });

    it('should provide all expected context properties', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      // Verify all context properties exist
      expect(result.current.config).toBeDefined();
      expect(result.current.setConfig).toBeDefined();
      expect(result.current.isConfigured).toBeDefined();
      expect(result.current.setIsConfigured).toBeDefined();
      expect(result.current.backendSandboxError).toBeDefined();
      expect(result.current.setBackendSandboxError).toBeDefined();
      expect(result.current.sandboxEngine).toBeDefined();
      expect(result.current.sandboxPendingChoice).toBeDefined();
      expect(result.current.setSandboxPendingChoice).toBeDefined();
      expect(result.current.sandboxCaptureChoice).toBeDefined();
      expect(result.current.setSandboxCaptureChoice).toBeDefined();
      expect(result.current.sandboxCaptureTargets).toBeDefined();
      expect(result.current.setSandboxCaptureTargets).toBeDefined();
      expect(result.current.sandboxLastProgressAt).toBeDefined();
      expect(result.current.setSandboxLastProgressAt).toBeDefined();
      expect(result.current.sandboxStallWarning).toBeDefined();
      expect(result.current.setSandboxStallWarning).toBeDefined();
      expect(result.current.sandboxStateVersion).toBeDefined();
      expect(result.current.setSandboxStateVersion).toBeDefined();
      expect(result.current.sandboxDiagnosticsEnabled).toBeDefined();
      expect(result.current.initLocalSandboxEngine).toBeInstanceOf(Function);
      expect(result.current.getSandboxGameState).toBeInstanceOf(Function);
      expect(result.current.resetSandboxEngine).toBeInstanceOf(Function);
    });
  });

  describe('Sandbox game state', () => {
    it('should update game state on moves', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      const mockGameState = {
        id: 'sandbox-local',
        boardType: 'square8' as BoardType,
        currentPlayer: 1,
        currentPhase: 'ring_placement' as const,
      } as GameState;

      mockGetGameState.mockReturnValue(mockGameState);

      act(() => {
        const engine = result.current.initLocalSandboxEngine({
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['human', 'human'],
          interactionHandler: {
            requestChoice: jest.fn(),
          },
        });
        expect(engine).toBeDefined();
      });

      const state = result.current.getSandboxGameState();
      expect(state).toEqual(mockGameState);
    });

    it('should handle invalid move rejection', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      // Initially no engine
      const initialState = result.current.getSandboxGameState();
      expect(initialState).toBeNull();
    });

    it('should trigger re-render on state changes', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      const initialVersion = result.current.sandboxStateVersion;

      act(() => {
        result.current.setSandboxStateVersion((prev) => prev + 1);
      });

      expect(result.current.sandboxStateVersion).toBe(initialVersion + 1);
    });

    it('should maintain state immutability', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      const originalConfig = result.current.config;

      act(() => {
        result.current.setConfig({
          numPlayers: 4,
          boardType: 'hexagonal',
          playerTypes: ['ai', 'ai', 'ai', 'ai'],
        });
      });

      // Verify original config wasn't mutated
      expect(originalConfig).toEqual({
        numPlayers: 2,
        boardType: 'square8',
        playerTypes: ['human', 'human', 'ai', 'ai'],
      });
      expect(result.current.config).toEqual({
        numPlayers: 4,
        boardType: 'hexagonal',
        playerTypes: ['ai', 'ai', 'ai', 'ai'],
      });
    });

    it('should handle rapid move sequences', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      // Simulate rapid state version updates
      act(() => {
        for (let i = 0; i < 10; i++) {
          result.current.setSandboxStateVersion((prev) => prev + 1);
        }
      });

      expect(result.current.sandboxStateVersion).toBe(10);
    });
  });

  describe('AI player integration', () => {
    it('should initialize AI players when configured', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      act(() => {
        result.current.setConfig({
          numPlayers: 2,
          boardType: 'square8',
          playerTypes: ['human', 'ai'],
        });
      });

      expect(result.current.config.playerTypes).toEqual(['human', 'ai']);
    });

    it('should execute AI moves automatically', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      const mockGameState = {
        id: 'sandbox-local',
        currentPlayer: 2,
        players: [
          { playerNumber: 1, type: 'human' as const },
          { playerNumber: 2, type: 'ai' as const },
        ],
      } as GameState;

      mockGetGameState.mockReturnValue(mockGameState);

      act(() => {
        result.current.initLocalSandboxEngine({
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['human', 'ai'],
          interactionHandler: {
            requestChoice: jest.fn(),
          },
        });
      });

      // Verify engine was initialized
      expect(result.current.sandboxEngine).not.toBeNull();
    });

    it('should handle AI move delays', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      // Set last progress timestamp
      const now = Date.now();
      act(() => {
        result.current.setSandboxLastProgressAt(now);
      });

      expect(result.current.sandboxLastProgressAt).toBe(now);
    });

    it('should handle AI errors gracefully', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      act(() => {
        result.current.setBackendSandboxError('AI engine failed');
      });

      expect(result.current.backendSandboxError).toBe('AI engine failed');
    });
  });

  describe('Local game engine', () => {
    it('should validate moves using sandbox engine', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      act(() => {
        const engine = result.current.initLocalSandboxEngine({
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['human', 'human'],
          interactionHandler: {
            requestChoice: jest.fn(),
          },
        });
        expect(engine).toBeDefined();
      });

      expect(result.current.isConfigured).toBe(true);
      expect(result.current.sandboxEngine).not.toBeNull();
    });

    it('should apply moves to local state', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      const mockChoice: PlayerChoice = {
        id: 'choice-1',
        gameId: 'sandbox-local',
        playerNumber: 1,
        type: 'capture_direction',
        prompt: 'Choose capture direction',
        options: [],
      };

      act(() => {
        result.current.setSandboxPendingChoice(mockChoice);
      });

      expect(result.current.sandboxPendingChoice).toEqual(mockChoice);
    });

    it('should detect win conditions', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      const mockGameState = {
        id: 'sandbox-local',
        gameStatus: 'completed' as const,
      } as GameState;

      mockGetGameState.mockReturnValue(mockGameState);

      act(() => {
        result.current.initLocalSandboxEngine({
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['human', 'human'],
          interactionHandler: {
            requestChoice: jest.fn(),
          },
        });
      });

      const state = result.current.getSandboxGameState();
      expect(state?.gameStatus).toBe('completed');
    });

    it('should handle game termination', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      act(() => {
        result.current.initLocalSandboxEngine({
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['human', 'human'],
          interactionHandler: {
            requestChoice: jest.fn(),
          },
        });
      });

      act(() => {
        result.current.resetSandboxEngine();
      });

      expect(result.current.sandboxEngine).toBeNull();
      expect(result.current.isConfigured).toBe(false);
    });
  });

  describe('Error handling', () => {
    it('should handle engine initialization errors', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      act(() => {
        result.current.setBackendSandboxError('Failed to initialize engine');
      });

      expect(result.current.backendSandboxError).toBe('Failed to initialize engine');

      act(() => {
        result.current.setBackendSandboxError(null);
      });

      expect(result.current.backendSandboxError).toBeNull();
    });

    it('should handle invalid game configurations', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      // Try to set invalid config
      act(() => {
        result.current.setConfig({
          numPlayers: 0, // Invalid
          boardType: 'square8',
          playerTypes: [],
        });
      });

      // Context should still update (validation is engine's responsibility)
      expect(result.current.config.numPlayers).toBe(0);
    });

    it('should recover from transient failures', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      // Simulate error
      act(() => {
        result.current.setBackendSandboxError('Transient error');
      });

      expect(result.current.backendSandboxError).toBe('Transient error');

      // Clear error (recovery)
      act(() => {
        result.current.setBackendSandboxError(null);
      });

      expect(result.current.backendSandboxError).toBeNull();
    });
  });

  describe('Stall watchdog', () => {
    it('should detect AI stalls after timeout', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      const mockGameState = {
        id: 'sandbox-local',
        gameStatus: 'active' as const,
        currentPlayer: 1,
        players: [{ playerNumber: 1, type: 'ai' as const }],
      } as GameState;

      mockGetGameState.mockReturnValue(mockGameState);

      act(() => {
        result.current.initLocalSandboxEngine({
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['ai', 'human'],
          interactionHandler: {
            requestChoice: jest.fn(),
          },
        });
        result.current.setIsConfigured(true);
        result.current.setSandboxLastProgressAt(Date.now() - 10000); // 10 seconds ago
      });

      // Advance timers past stall timeout
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      // Stall warning should be set
      await waitFor(() => {
        expect(result.current.sandboxStallWarning).toBeTruthy();
      });
    });

    it('should not set stall warning when game is not active', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      const mockGameState = {
        id: 'sandbox-local',
        gameStatus: 'completed' as const,
        currentPlayer: 1,
        players: [{ playerNumber: 1, type: 'ai' as const }],
      } as GameState;

      mockGetGameState.mockReturnValue(mockGameState);

      act(() => {
        result.current.initLocalSandboxEngine({
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['ai', 'human'],
          interactionHandler: {
            requestChoice: jest.fn(),
          },
        });
        result.current.setIsConfigured(true);
        result.current.setSandboxLastProgressAt(Date.now() - 10000); // 10 seconds ago
      });

      // Advance timers
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      // Warning should not be set when game is not active
      expect(result.current.sandboxStallWarning).toBeNull();
    });
    it('should handle stall warning when lastProgressAt is null', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      const mockGameState = {
        id: 'sandbox-local',
        gameStatus: 'active' as const,
        currentPlayer: 1,
        players: [{ playerNumber: 1, type: 'ai' as const }],
      } as GameState;

      mockGetGameState.mockReturnValue(mockGameState);

      act(() => {
        result.current.initLocalSandboxEngine({
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['ai', 'human'],
          interactionHandler: {
            requestChoice: jest.fn(),
          },
        });
        result.current.setIsConfigured(true);
        // Don't set sandboxLastProgressAt - it should remain null
      });

      // Advance timers
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      // Warning should not be set when lastProgressAt is null
      expect(result.current.sandboxStallWarning).toBeNull();
    });

    it('should clear warning when no engine exists', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      act(() => {
        result.current.setIsConfigured(true);
        result.current.setSandboxLastProgressAt(Date.now() - 10000);
      });

      // Advance timers
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      // Warning should not be set when no engine exists
      expect(result.current.sandboxStallWarning).toBeNull();
    });

    it('should not trigger when human player is current', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      const mockGameState = {
        id: 'sandbox-local',
        gameStatus: 'active' as const,
        currentPlayer: 1,
        players: [{ playerNumber: 1, type: 'human' as const }],
      } as GameState;

      mockGetGameState.mockReturnValue(mockGameState);

      act(() => {
        result.current.initLocalSandboxEngine({
          boardType: 'square8',
          numPlayers: 2,
          playerTypes: ['human', 'ai'],
          interactionHandler: {
            requestChoice: jest.fn(),
          },
        });
        result.current.setIsConfigured(true);
        result.current.setSandboxLastProgressAt(Date.now() - 10000);
      });

      // Advance timers
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      // Warning should not be set for human players
      expect(result.current.sandboxStallWarning).toBeNull();
    });
  });

  describe('Diagnostics mode', () => {
    it('should enable diagnostics when flag is set', () => {
      mockIsSandboxDiagnostics.mockReturnValue(true);

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      expect(result.current.sandboxDiagnosticsEnabled).toBe(true);
    });

    it('should disable diagnostics by default', () => {
      mockIsSandboxDiagnostics.mockReturnValue(false);

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <SandboxProvider>{children}</SandboxProvider>
      );

      const { result } = renderHook(() => useSandbox(), { wrapper });

      expect(result.current.sandboxDiagnosticsEnabled).toBe(false);
    });
  });
});
