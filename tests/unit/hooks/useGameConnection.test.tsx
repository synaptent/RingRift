/**
 * Unit tests for useGameConnection hook
 *
 * Tests connection status, health derivation, and connection management functions.
 */

import { renderHook, act } from '@testing-library/react';
import {
  useGameConnection,
  useConnectionStatus,
  useIsConnected,
} from '@/client/hooks/useGameConnection';
import type { ConnectionStatus } from '@/client/contexts/GameContext';

// ─────────────────────────────────────────────────────────────────────────────
// Mock Setup
// ─────────────────────────────────────────────────────────────────────────────

const mockConnectToGame = jest.fn();
const mockDisconnect = jest.fn();

const createMockGameContext = (overrides: Record<string, unknown> = {}) => ({
  gameId: 'game-123',
  connectionStatus: 'connected' as ConnectionStatus,
  lastHeartbeatAt: Date.now(),
  error: null,
  isConnecting: false,
  connectToGame: mockConnectToGame,
  disconnect: mockDisconnect,
  gameState: null,
  validMoves: null,
  victoryState: null,
  pendingChoice: null,
  choiceDeadline: null,
  respondToChoice: jest.fn(),
  submitMove: jest.fn(),
  sendChatMessage: jest.fn(),
  chatMessages: [],
  ...overrides,
});

let mockContextValue = createMockGameContext();

jest.mock('@/client/contexts/GameContext', () => ({
  useGame: () => mockContextValue,
}));

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useGameConnection
// ─────────────────────────────────────────────────────────────────────────────

describe('useGameConnection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns all expected properties and functions', () => {
    const { result } = renderHook(() => useGameConnection());

    expect(result.current.gameId).toBeDefined();
    expect(result.current.status).toBeDefined();
    expect(result.current.isConnecting).toBeDefined();
    expect(result.current.isHealthy).toBeDefined();
    expect(result.current.isStale).toBeDefined();
    expect(result.current.isDisconnected).toBeDefined();
    expect(result.current.statusLabel).toBeDefined();
    expect(result.current.statusColorClass).toBeDefined();
    expect(result.current.connectToGame).toBeDefined();
    expect(result.current.disconnect).toBeDefined();
    expect(result.current.error).toBeDefined();
  });

  it('returns correct gameId from context', () => {
    mockContextValue = createMockGameContext({ gameId: 'test-game-xyz' });
    const { result } = renderHook(() => useGameConnection());

    expect(result.current.gameId).toBe('test-game-xyz');
  });

  it('connectToGame calls context connectToGame', async () => {
    const { result } = renderHook(() => useGameConnection());

    await act(async () => {
      await result.current.connectToGame('new-game-id');
    });

    expect(mockConnectToGame).toHaveBeenCalledWith('new-game-id');
    expect(mockConnectToGame).toHaveBeenCalledTimes(1);
  });

  it('disconnect calls context disconnect', () => {
    const { result } = renderHook(() => useGameConnection());

    act(() => {
      result.current.disconnect();
    });

    expect(mockDisconnect).toHaveBeenCalledTimes(1);
  });

  it('returns error message from context', () => {
    mockContextValue = createMockGameContext({ error: 'Connection lost' });
    const { result } = renderHook(() => useGameConnection());

    expect(result.current.error).toBe('Connection lost');
  });

  describe('connection health derivation', () => {
    it('returns isHealthy true when connected with recent heartbeat', () => {
      mockContextValue = createMockGameContext({
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.isHealthy).toBe(true);
      expect(result.current.isStale).toBe(false);
      expect(result.current.isDisconnected).toBe(false);
    });

    it('returns isStale true when heartbeat exceeds threshold', () => {
      const staleTime = Date.now() - 10000; // 10 seconds ago (> 8s threshold)
      mockContextValue = createMockGameContext({
        connectionStatus: 'connected',
        lastHeartbeatAt: staleTime,
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.isStale).toBe(true);
      expect(result.current.isHealthy).toBe(false);
    });

    it('returns isDisconnected true when status is disconnected', () => {
      mockContextValue = createMockGameContext({
        connectionStatus: 'disconnected',
        lastHeartbeatAt: null,
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.isDisconnected).toBe(true);
      expect(result.current.isHealthy).toBe(false);
    });

    it('returns isConnecting true during connection attempts', () => {
      mockContextValue = createMockGameContext({
        connectionStatus: 'connecting',
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.isConnecting).toBe(true);
      expect(result.current.isHealthy).toBe(false);
    });

    it('returns isConnecting true during reconnection', () => {
      mockContextValue = createMockGameContext({
        connectionStatus: 'reconnecting',
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.isConnecting).toBe(true);
    });
  });

  describe('status labels', () => {
    it('returns "Connected" when healthy', () => {
      mockContextValue = createMockGameContext({
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusLabel).toBe('Connected');
    });

    it('returns "Awaiting update…" when stale', () => {
      const staleTime = Date.now() - 10000;
      mockContextValue = createMockGameContext({
        connectionStatus: 'connected',
        lastHeartbeatAt: staleTime,
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusLabel).toBe('Awaiting update…');
    });

    it('returns "Connecting…" when connecting', () => {
      mockContextValue = createMockGameContext({
        connectionStatus: 'connecting',
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusLabel).toBe('Connecting…');
    });

    it('returns "Reconnecting…" when reconnecting', () => {
      mockContextValue = createMockGameContext({
        connectionStatus: 'reconnecting',
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusLabel).toBe('Reconnecting…');
    });

    it('returns "Disconnected" when disconnected', () => {
      mockContextValue = createMockGameContext({
        connectionStatus: 'disconnected',
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusLabel).toBe('Disconnected');
    });
  });

  describe('status color classes', () => {
    it('returns emerald color when healthy', () => {
      mockContextValue = createMockGameContext({
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now(),
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusColorClass).toBe('text-emerald-300');
    });

    it('returns amber color when stale', () => {
      const staleTime = Date.now() - 10000;
      mockContextValue = createMockGameContext({
        connectionStatus: 'connected',
        lastHeartbeatAt: staleTime,
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusColorClass).toBe('text-amber-300');
    });

    it('returns amber color when reconnecting', () => {
      mockContextValue = createMockGameContext({
        connectionStatus: 'reconnecting',
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusColorClass).toBe('text-amber-300');
    });

    it('returns rose color when disconnected', () => {
      mockContextValue = createMockGameContext({
        connectionStatus: 'disconnected',
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.statusColorClass).toBe('text-rose-300');
    });
  });

  describe('timeSinceHeartbeat', () => {
    it('returns null when lastHeartbeatAt is null', () => {
      mockContextValue = createMockGameContext({
        connectionStatus: 'disconnected',
        lastHeartbeatAt: null,
      });
      const { result } = renderHook(() => useGameConnection());

      expect(result.current.timeSinceHeartbeat).toBeNull();
    });

    it('returns time since last heartbeat in milliseconds', () => {
      const now = Date.now();
      const heartbeatTime = now - 5000; // 5 seconds ago
      // Mock Date.now for the hook calculation
      jest.spyOn(Date, 'now').mockReturnValue(now);

      mockContextValue = createMockGameContext({
        connectionStatus: 'connected',
        lastHeartbeatAt: heartbeatTime,
      });
      const { result } = renderHook(() => useGameConnection());

      // Allow some tolerance for timing
      expect(result.current.timeSinceHeartbeat).toBeGreaterThanOrEqual(4900);
      expect(result.current.timeSinceHeartbeat).toBeLessThanOrEqual(5100);

      jest.restoreAllMocks();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useConnectionStatus
// ─────────────────────────────────────────────────────────────────────────────

describe('useConnectionStatus', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns all ConnectionHealth properties', () => {
    const { result } = renderHook(() => useConnectionStatus());

    expect(result.current.status).toBeDefined();
    expect(result.current.isConnecting).toBeDefined();
    expect(result.current.isHealthy).toBeDefined();
    expect(result.current.isStale).toBeDefined();
    expect(result.current.isDisconnected).toBeDefined();
    expect(result.current.statusLabel).toBeDefined();
    expect(result.current.statusColorClass).toBeDefined();
    expect(result.current.timeSinceHeartbeat).toBeDefined();
  });

  it('does not include connection action functions', () => {
    const { result } = renderHook(() => useConnectionStatus());

    // useConnectionStatus is a lightweight hook without actions
    expect((result.current as any).connectToGame).toBeUndefined();
    expect((result.current as any).disconnect).toBeUndefined();
  });

  it('derives correct health for connected state', () => {
    mockContextValue = createMockGameContext({
      connectionStatus: 'connected',
      lastHeartbeatAt: Date.now(),
    });
    const { result } = renderHook(() => useConnectionStatus());

    expect(result.current.status).toBe('connected');
    expect(result.current.isHealthy).toBe(true);
  });

  it('derives correct health for disconnected state', () => {
    mockContextValue = createMockGameContext({
      connectionStatus: 'disconnected',
      lastHeartbeatAt: null,
    });
    const { result } = renderHook(() => useConnectionStatus());

    expect(result.current.status).toBe('disconnected');
    expect(result.current.isHealthy).toBe(false);
    expect(result.current.isDisconnected).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useIsConnected
// ─────────────────────────────────────────────────────────────────────────────

describe('useIsConnected', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns true when connected with fresh heartbeat', () => {
    mockContextValue = createMockGameContext({
      connectionStatus: 'connected',
      lastHeartbeatAt: Date.now(),
    });
    const { result } = renderHook(() => useIsConnected());

    expect(result.current).toBe(true);
  });

  it('returns false when disconnected', () => {
    mockContextValue = createMockGameContext({
      connectionStatus: 'disconnected',
    });
    const { result } = renderHook(() => useIsConnected());

    expect(result.current).toBe(false);
  });

  it('returns false when connecting', () => {
    mockContextValue = createMockGameContext({
      connectionStatus: 'connecting',
    });
    const { result } = renderHook(() => useIsConnected());

    expect(result.current).toBe(false);
  });

  it('returns false when reconnecting', () => {
    mockContextValue = createMockGameContext({
      connectionStatus: 'reconnecting',
    });
    const { result } = renderHook(() => useIsConnected());

    expect(result.current).toBe(false);
  });

  it('returns false when heartbeat is stale', () => {
    const staleTime = Date.now() - 10000; // 10 seconds ago
    mockContextValue = createMockGameContext({
      connectionStatus: 'connected',
      lastHeartbeatAt: staleTime,
    });
    const { result } = renderHook(() => useIsConnected());

    expect(result.current).toBe(false);
  });

  it('returns true when connected without heartbeat (no timeout)', () => {
    mockContextValue = createMockGameContext({
      connectionStatus: 'connected',
      lastHeartbeatAt: null,
    });
    const { result } = renderHook(() => useIsConnected());

    // When lastHeartbeatAt is null but connected, we treat it as connected
    // (no heartbeat received yet, but socket is connected)
    expect(result.current).toBe(true);
  });
});
