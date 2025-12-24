/**
 * ═══════════════════════════════════════════════════════════════════════════
 * useGameConnection Hook Tests
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Comprehensive test suite for the useGameConnection hook which provides
 * connection status and management functions for WebSocket-based game
 * connections via the GameContext.
 *
 * Test Categories:
 * - Initialization tests
 * - Connection health derivation tests
 * - Heartbeat staleness detection tests
 * - Connection lifecycle tests
 * - State management tests
 * - Edge cases
 */

import React from 'react';
import { renderHook, act } from '@testing-library/react';
import {
  useGameConnection,
  useConnectionStatus,
  useIsConnected,
  type ConnectionHealth,
  type GameConnectionState,
} from '../../../../src/client/hooks/useGameConnection';
import { GameProvider, useGame } from '../../../../src/client/contexts/GameContext';
import type { ConnectionStatus, DisconnectedPlayer } from '../../../../src/client/contexts/GameContext';

// ═══════════════════════════════════════════════════════════════════════════
// Mocks
// ═══════════════════════════════════════════════════════════════════════════

// Mock the GameContext to control the underlying state
const mockGameContextValue = {
  gameId: null as string | null,
  connectionStatus: 'disconnected' as ConnectionStatus,
  lastHeartbeatAt: null as number | null,
  error: null as string | null,
  isConnecting: false,
  connectToGame: jest.fn().mockResolvedValue(undefined),
  disconnect: jest.fn(),
  disconnectedOpponents: [] as DisconnectedPlayer[],
  gameEndedByAbandonment: false,
  // Minimal required fields to satisfy GameContextType
  gameState: null,
  validMoves: null,
  victoryState: null,
  pendingChoice: null,
  choiceDeadline: null,
  respondToChoice: jest.fn(),
  submitMove: jest.fn(),
  sendChatMessage: jest.fn(),
  chatMessages: [],
  decisionAutoResolved: null,
  decisionPhaseTimeoutWarning: null,
  pendingRematchRequest: null,
  requestRematch: jest.fn(),
  acceptRematch: jest.fn(),
  declineRematch: jest.fn(),
  rematchGameId: null,
  rematchLastStatus: null,
  evaluationHistory: [],
};

jest.mock('../../../../src/client/contexts/GameContext', () => {
  const actual = jest.requireActual('../../../../src/client/contexts/GameContext');
  return {
    ...actual,
    useGame: jest.fn(() => mockGameContextValue),
  };
});

// ═══════════════════════════════════════════════════════════════════════════
// Test Utilities
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Reset mock to default disconnected state
 */
function resetMockGameContext() {
  mockGameContextValue.gameId = null;
  mockGameContextValue.connectionStatus = 'disconnected';
  mockGameContextValue.lastHeartbeatAt = null;
  mockGameContextValue.error = null;
  mockGameContextValue.isConnecting = false;
  mockGameContextValue.disconnectedOpponents = [];
  mockGameContextValue.gameEndedByAbandonment = false;
  mockGameContextValue.connectToGame.mockClear();
  mockGameContextValue.disconnect.mockClear();
}

/**
 * Configure mock for a connected state
 */
function setConnectedState(options: {
  gameId?: string;
  lastHeartbeatAt?: number | null;
  error?: string | null;
  disconnectedOpponents?: DisconnectedPlayer[];
  gameEndedByAbandonment?: boolean;
} = {}) {
  mockGameContextValue.gameId = options.gameId ?? 'game-123';
  mockGameContextValue.connectionStatus = 'connected';
  mockGameContextValue.lastHeartbeatAt = options.lastHeartbeatAt ?? Date.now();
  mockGameContextValue.error = options.error ?? null;
  mockGameContextValue.isConnecting = false;
  mockGameContextValue.disconnectedOpponents = options.disconnectedOpponents ?? [];
  mockGameContextValue.gameEndedByAbandonment = options.gameEndedByAbandonment ?? false;
}

/**
 * Configure mock for a specific connection status
 */
function setConnectionStatus(status: ConnectionStatus) {
  mockGameContextValue.connectionStatus = status;
  mockGameContextValue.isConnecting = status === 'connecting' || status === 'reconnecting';
}

// ═══════════════════════════════════════════════════════════════════════════
// Initialization Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('useGameConnection', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    resetMockGameContext();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Initialization', () => {
    it('returns correct default state when disconnected', () => {
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.status).toBe('disconnected');
      expect(result.current.isDisconnected).toBe(true);
      expect(result.current.isConnecting).toBe(false);
      expect(result.current.isHealthy).toBe(false);
      expect(result.current.isStale).toBe(false);
      expect(result.current.gameId).toBeNull();
      expect(result.current.error).toBeNull();
      expect(result.current.lastHeartbeatAt).toBeNull();
      expect(result.current.timeSinceHeartbeat).toBeNull();
    });

    it('provides stable function references for connectToGame and disconnect', () => {
      const { result, rerender } = renderHook(() => useGameConnection());

      const initialConnectToGame = result.current.connectToGame;
      const initialDisconnect = result.current.disconnect;

      rerender();

      expect(result.current.connectToGame).toBe(initialConnectToGame);
      expect(result.current.disconnect).toBe(initialDisconnect);
    });

    it('exposes disconnectedOpponents from context', () => {
      const opponent: DisconnectedPlayer = {
        id: 'player-2',
        username: 'Opponent',
        disconnectedAt: Date.now() - 5000,
      };
      setConnectedState({ disconnectedOpponents: [opponent] });

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.disconnectedOpponents).toHaveLength(1);
      expect(result.current.disconnectedOpponents[0].id).toBe('player-2');
      expect(result.current.disconnectedOpponents[0].username).toBe('Opponent');
    });

    it('exposes gameEndedByAbandonment from context', () => {
      setConnectedState({ gameEndedByAbandonment: true });

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.gameEndedByAbandonment).toBe(true);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Connection Health Derivation Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Connection Health Derivation', () => {
    it('returns isHealthy=true when connected with recent heartbeat', () => {
      setConnectedState({ lastHeartbeatAt: Date.now() });

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.isHealthy).toBe(true);
      expect(result.current.isStale).toBe(false);
      expect(result.current.statusLabel).toBe('Connected');
      expect(result.current.statusColorClass).toBe('text-emerald-300');
    });

    it('returns isHealthy=false and isStale=true when heartbeat exceeds threshold', () => {
      // Set heartbeat to 10 seconds ago (threshold is 8 seconds)
      const staleHeartbeat = Date.now() - 10000;
      setConnectedState({ lastHeartbeatAt: staleHeartbeat });

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.isHealthy).toBe(false);
      expect(result.current.isStale).toBe(true);
      expect(result.current.statusLabel).toBe('Awaiting update…');
      expect(result.current.statusColorClass).toBe('text-amber-300');
    });

    it('returns isStale=false when disconnected even with old heartbeat', () => {
      // Even if we have an old heartbeat, if status is disconnected, isStale should be false
      mockGameContextValue.lastHeartbeatAt = Date.now() - 20000;
      mockGameContextValue.connectionStatus = 'disconnected';

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.isStale).toBe(false);
      expect(result.current.isDisconnected).toBe(true);
      expect(result.current.status).toBe('disconnected');
    });

    it('calculates timeSinceHeartbeat correctly', () => {
      const heartbeatTime = Date.now() - 5000;
      setConnectedState({ lastHeartbeatAt: heartbeatTime });

      const { result } = renderHook(() => useGameConnection());

      // Should be approximately 5000ms (with some tolerance for test execution time)
      expect(result.current.timeSinceHeartbeat).toBeGreaterThanOrEqual(5000);
      expect(result.current.timeSinceHeartbeat).toBeLessThan(6000);
    });

    it('returns timeSinceHeartbeat=null when no heartbeat received', () => {
      setConnectionStatus('connected');
      mockGameContextValue.lastHeartbeatAt = null;

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.timeSinceHeartbeat).toBeNull();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Connection Status Label and Styling Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Connection Status Labels and Styling', () => {
    it('returns correct label and color for disconnected status', () => {
      setConnectionStatus('disconnected');

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusLabel).toBe('Disconnected');
      expect(result.current.statusColorClass).toBe('text-rose-300');
    });

    it('returns correct label and color for connecting status', () => {
      setConnectionStatus('connecting');

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusLabel).toBe('Connecting…');
      expect(result.current.isConnecting).toBe(true);
      // When connecting, it's not healthy and not stale (since not connected)
      expect(result.current.statusColorClass).toBe('text-rose-300');
    });

    it('returns correct label and color for reconnecting status', () => {
      setConnectionStatus('reconnecting');

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusLabel).toBe('Reconnecting…');
      expect(result.current.isConnecting).toBe(true);
      expect(result.current.statusColorClass).toBe('text-amber-300');
    });

    it('returns correct label and color for healthy connected status', () => {
      setConnectedState({ lastHeartbeatAt: Date.now() });

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusLabel).toBe('Connected');
      expect(result.current.statusColorClass).toBe('text-emerald-300');
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Connection Lifecycle Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Connection Lifecycle', () => {
    it('connectToGame calls context connectToGame with gameId', async () => {
      const { result } = renderHook(() => useGameConnection());

      await act(async () => {
        await result.current.connectToGame('game-456');
      });

      expect(mockGameContextValue.connectToGame).toHaveBeenCalledTimes(1);
      expect(mockGameContextValue.connectToGame).toHaveBeenCalledWith('game-456');
    });

    it('disconnect calls context disconnect', () => {
      setConnectedState();
      const { result } = renderHook(() => useGameConnection());

      act(() => {
        result.current.disconnect();
      });

      expect(mockGameContextValue.disconnect).toHaveBeenCalledTimes(1);
    });

    it('tracks connection status transitions through connecting', () => {
      const { result, rerender } = renderHook(() => useGameConnection());

      // Initial: disconnected
      expect(result.current.status).toBe('disconnected');
      expect(result.current.isConnecting).toBe(false);

      // Transition to connecting
      setConnectionStatus('connecting');
      rerender();
      expect(result.current.status).toBe('connecting');
      expect(result.current.isConnecting).toBe(true);

      // Transition to connected
      setConnectedState({ lastHeartbeatAt: Date.now() });
      rerender();
      expect(result.current.status).toBe('connected');
      expect(result.current.isConnecting).toBe(false);
      expect(result.current.isHealthy).toBe(true);
    });

    it('tracks reconnecting status correctly', () => {
      // Start connected
      setConnectedState({ lastHeartbeatAt: Date.now() });
      const { result, rerender } = renderHook(() => useGameConnection());

      expect(result.current.status).toBe('connected');

      // Transition to reconnecting
      setConnectionStatus('reconnecting');
      rerender();
      expect(result.current.status).toBe('reconnecting');
      expect(result.current.isConnecting).toBe(true);
      expect(result.current.isHealthy).toBe(false);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Error State Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Error State', () => {
    it('exposes error from context', () => {
      mockGameContextValue.error = 'Connection failed';

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.error).toBe('Connection failed');
    });

    it('error is null when no error occurred', () => {
      setConnectedState({ error: null });

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.error).toBeNull();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Game ID Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Game ID', () => {
    it('exposes gameId when connected', () => {
      setConnectedState({ gameId: 'my-game-id-789' });

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.gameId).toBe('my-game-id-789');
    });

    it('gameId is null when not connected', () => {
      resetMockGameContext();

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.gameId).toBeNull();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Heartbeat Staleness Threshold Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Heartbeat Staleness Threshold', () => {
    it('is not stale at exactly 8 seconds (threshold boundary)', () => {
      // At exactly 8000ms, should NOT be stale (staleness is strictly greater than threshold)
      const exactly8Seconds = Date.now() - 8000;
      setConnectedState({ lastHeartbeatAt: exactly8Seconds });

      const { result } = renderHook(() => useGameConnection());

      // 8000ms is not greater than 8000ms threshold, so not stale
      expect(result.current.isStale).toBe(false);
      expect(result.current.isHealthy).toBe(true);
    });

    it('becomes stale after 8 seconds', () => {
      // At 8001ms, should be stale
      const over8Seconds = Date.now() - 8001;
      setConnectedState({ lastHeartbeatAt: over8Seconds });

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.isStale).toBe(true);
      expect(result.current.isHealthy).toBe(false);
    });

    it('is healthy at 7 seconds', () => {
      const sevenSeconds = Date.now() - 7000;
      setConnectedState({ lastHeartbeatAt: sevenSeconds });

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.isStale).toBe(false);
      expect(result.current.isHealthy).toBe(true);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Edge Cases
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Edge Cases', () => {
    it('handles rapid status changes gracefully', () => {
      const { result, rerender } = renderHook(() => useGameConnection());

      // Rapid transitions
      setConnectionStatus('connecting');
      rerender();
      expect(result.current.status).toBe('connecting');

      setConnectionStatus('connected');
      mockGameContextValue.lastHeartbeatAt = Date.now();
      rerender();
      expect(result.current.status).toBe('connected');

      setConnectionStatus('reconnecting');
      rerender();
      expect(result.current.status).toBe('reconnecting');

      setConnectionStatus('disconnected');
      rerender();
      expect(result.current.status).toBe('disconnected');
    });

    it('handles multiple disconnected opponents', () => {
      const opponents: DisconnectedPlayer[] = [
        { id: 'p2', username: 'Player 2', disconnectedAt: Date.now() - 10000 },
        { id: 'p3', username: 'Player 3', disconnectedAt: Date.now() - 5000 },
        { id: 'p4', username: 'Player 4', disconnectedAt: Date.now() - 2000 },
      ];
      setConnectedState({ disconnectedOpponents: opponents });

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.disconnectedOpponents).toHaveLength(3);
      expect(result.current.disconnectedOpponents.map(p => p.id)).toEqual(['p2', 'p3', 'p4']);
    });

    it('maintains health info when error is present but connected', () => {
      // It's possible to have an error message while still connected
      // (e.g., a non-fatal error like a failed move submission)
      setConnectedState({ error: 'Move submission failed', lastHeartbeatAt: Date.now() });

      const { result } = renderHook(() => useGameConnection());

      expect(result.current.error).toBe('Move submission failed');
      expect(result.current.isHealthy).toBe(true);
      expect(result.current.status).toBe('connected');
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// useConnectionStatus Tests (Lightweight variant)
// ═══════════════════════════════════════════════════════════════════════════

describe('useConnectionStatus', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    resetMockGameContext();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('returns only connection health info without actions', () => {
    setConnectedState({ lastHeartbeatAt: Date.now() });

    const { result } = renderHook(() => useConnectionStatus());

    expect(result.current.status).toBe('connected');
    expect(result.current.isHealthy).toBe(true);
    expect(result.current.isStale).toBe(false);
    expect(result.current.isConnecting).toBe(false);
    expect(result.current.isDisconnected).toBe(false);
    expect(result.current.statusLabel).toBe('Connected');
    expect(result.current.statusColorClass).toBe('text-emerald-300');
    expect(result.current.timeSinceHeartbeat).not.toBeNull();

    // Should not have action methods (they are not part of ConnectionHealth type)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const fullResult = result.current as unknown as Record<string, unknown>;
    expect(fullResult.connectToGame).toBeUndefined();
    expect(fullResult.disconnect).toBeUndefined();
  });

  it('derives correct status for disconnected state', () => {
    resetMockGameContext();

    const { result } = renderHook(() => useConnectionStatus());

    expect(result.current.status).toBe('disconnected');
    expect(result.current.isDisconnected).toBe(true);
    expect(result.current.isHealthy).toBe(false);
    expect(result.current.statusLabel).toBe('Disconnected');
    expect(result.current.statusColorClass).toBe('text-rose-300');
  });

  it('derives correct status for stale connection', () => {
    setConnectedState({ lastHeartbeatAt: Date.now() - 10000 });

    const { result } = renderHook(() => useConnectionStatus());

    expect(result.current.isStale).toBe(true);
    expect(result.current.isHealthy).toBe(false);
    expect(result.current.statusLabel).toBe('Awaiting update…');
    expect(result.current.statusColorClass).toBe('text-amber-300');
  });

  it('derives correct status for reconnecting', () => {
    setConnectionStatus('reconnecting');

    const { result } = renderHook(() => useConnectionStatus());

    expect(result.current.status).toBe('reconnecting');
    expect(result.current.isConnecting).toBe(true);
    expect(result.current.statusLabel).toBe('Reconnecting…');
    expect(result.current.statusColorClass).toBe('text-amber-300');
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// useIsConnected Tests (Boolean variant)
// ═══════════════════════════════════════════════════════════════════════════

describe('useIsConnected', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    resetMockGameContext();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('returns true when connected with fresh heartbeat', () => {
    setConnectedState({ lastHeartbeatAt: Date.now() });

    const { result } = renderHook(() => useIsConnected());

    expect(result.current).toBe(true);
  });

  it('returns false when disconnected', () => {
    resetMockGameContext();

    const { result } = renderHook(() => useIsConnected());

    expect(result.current).toBe(false);
  });

  it('returns false when connecting', () => {
    setConnectionStatus('connecting');

    const { result } = renderHook(() => useIsConnected());

    expect(result.current).toBe(false);
  });

  it('returns false when reconnecting', () => {
    setConnectionStatus('reconnecting');

    const { result } = renderHook(() => useIsConnected());

    expect(result.current).toBe(false);
  });

  it('returns false when heartbeat is stale', () => {
    setConnectedState({ lastHeartbeatAt: Date.now() - 10000 });

    const { result } = renderHook(() => useIsConnected());

    expect(result.current).toBe(false);
  });

  it('returns true when connected with no heartbeat (null)', () => {
    // Edge case: connected but no heartbeat received yet
    setConnectionStatus('connected');
    mockGameContextValue.lastHeartbeatAt = null;

    const { result } = renderHook(() => useIsConnected());

    // When lastHeartbeatAt is null, we can't determine staleness, so default to connected
    expect(result.current).toBe(true);
  });

  it('transitions correctly as connection status changes', () => {
    const { result, rerender } = renderHook(() => useIsConnected());

    // Initially disconnected
    expect(result.current).toBe(false);

    // Connect
    setConnectedState({ lastHeartbeatAt: Date.now() });
    rerender();
    expect(result.current).toBe(true);

    // Become stale
    mockGameContextValue.lastHeartbeatAt = Date.now() - 10000;
    rerender();
    expect(result.current).toBe(false);

    // Fresh heartbeat
    mockGameContextValue.lastHeartbeatAt = Date.now();
    rerender();
    expect(result.current).toBe(true);

    // Disconnect
    setConnectionStatus('disconnected');
    rerender();
    expect(result.current).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Integration with Time-Based Staleness Detection
// ═══════════════════════════════════════════════════════════════════════════

describe('Time-Based Staleness Detection', () => {
  beforeEach(() => {
    resetMockGameContext();
  });

  it('detects staleness based on heartbeat age', () => {
    // Set heartbeat that is 10 seconds old (past the 8-second threshold)
    const staleHeartbeat = Date.now() - 10000;
    setConnectedState({ lastHeartbeatAt: staleHeartbeat });

    const { result } = renderHook(() => useGameConnection());

    expect(result.current.isStale).toBe(true);
    expect(result.current.isHealthy).toBe(false);
    expect(result.current.statusLabel).toBe('Awaiting update…');
  });

  it('detects healthy state based on recent heartbeat', () => {
    // Set heartbeat that is 1 second old (within the 8-second threshold)
    const recentHeartbeat = Date.now() - 1000;
    setConnectedState({ lastHeartbeatAt: recentHeartbeat });

    const { result } = renderHook(() => useGameConnection());

    expect(result.current.isStale).toBe(false);
    expect(result.current.isHealthy).toBe(true);
    expect(result.current.statusLabel).toBe('Connected');
  });

  it('calculates timeSinceHeartbeat as positive value for past heartbeat', () => {
    const heartbeatAge = 3000; // 3 seconds ago
    const heartbeatTime = Date.now() - heartbeatAge;
    setConnectedState({ lastHeartbeatAt: heartbeatTime });

    const { result } = renderHook(() => useGameConnection());

    // Should be approximately 3000ms (allow some execution time variance)
    expect(result.current.timeSinceHeartbeat).toBeGreaterThanOrEqual(heartbeatAge);
    expect(result.current.timeSinceHeartbeat).toBeLessThan(heartbeatAge + 100);
  });

  it('reacts to heartbeat change from stale to fresh', () => {
    // Start with stale heartbeat
    const staleHeartbeat = Date.now() - 15000;
    setConnectedState({ lastHeartbeatAt: staleHeartbeat });

    const { result, rerender } = renderHook(() => useGameConnection());

    expect(result.current.isStale).toBe(true);
    expect(result.current.isHealthy).toBe(false);

    // Simulate receiving a fresh heartbeat
    mockGameContextValue.lastHeartbeatAt = Date.now();
    rerender();

    expect(result.current.isStale).toBe(false);
    expect(result.current.isHealthy).toBe(true);
  });
});
