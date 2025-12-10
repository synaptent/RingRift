/**
 * Unit tests for useMatchmaking hook
 *
 * Tests matchmaking queue state management including:
 * - Joining and leaving the queue
 * - State management
 * - Clear error functionality
 */

import { renderHook, act } from '@testing-library/react';
import { useMatchmaking } from '@/client/hooks/useMatchmaking';
import type { MatchmakingPreferences } from '@/shared/types/websocket';

// Mock react-router-dom
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
}));

// Track registered event handlers
const eventHandlers: Record<string, Function> = {};

// Mock socket.io-client
const mockSocket = {
  connected: true,
  on: jest.fn((event: string, handler: Function) => {
    eventHandlers[event] = handler;
    return mockSocket;
  }),
  off: jest.fn(),
  emit: jest.fn(),
  disconnect: jest.fn(),
};

jest.mock('socket.io-client', () => ({
  io: jest.fn(() => mockSocket),
}));

// Mock envFlags
jest.mock('@/shared/utils/envFlags', () => ({
  readEnv: jest.fn(() => null),
}));

describe('useMatchmaking', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockSocket.connected = true;
    Object.keys(eventHandlers).forEach((key) => delete eventHandlers[key]);
    localStorage.clear();
  });

  const createPreferences = (overrides?: Partial<MatchmakingPreferences>): MatchmakingPreferences => ({
    boardType: 'square8',
    timeControl: { min: 300, max: 600 },
    ratingRange: { min: 1000, max: 1200 },
    ...overrides,
  });

  describe('initial state', () => {
    it('should start with default state', () => {
      const { result } = renderHook(() => useMatchmaking());

      expect(result.current.inQueue).toBe(false);
      expect(result.current.estimatedWaitTime).toBeNull();
      expect(result.current.queuePosition).toBeNull();
      expect(result.current.searchCriteria).toBeNull();
      expect(result.current.matchFound).toBe(false);
      expect(result.current.matchedGameId).toBeNull();
      expect(result.current.error).toBeNull();
    });
  });

  describe('joinQueue', () => {
    it('should emit matchmaking:join event when joining queue', () => {
      const { result } = renderHook(() => useMatchmaking());
      const preferences = createPreferences();

      act(() => {
        result.current.joinQueue(preferences);
      });

      expect(mockSocket.emit).toHaveBeenCalledWith('matchmaking:join', { preferences });
    });

    it('should set isConnecting to true when joining', () => {
      const { result } = renderHook(() => useMatchmaking());

      act(() => {
        result.current.joinQueue(createPreferences());
      });

      expect(result.current.isConnecting).toBe(true);
    });

    it('should clear previous match state when joining', () => {
      const { result } = renderHook(() => useMatchmaking());

      act(() => {
        result.current.joinQueue(createPreferences());
      });

      expect(result.current.matchFound).toBe(false);
      expect(result.current.matchedGameId).toBeNull();
    });
  });

  describe('leaveQueue', () => {
    it('should emit matchmaking:leave event when leaving queue after joining', () => {
      const { result } = renderHook(() => useMatchmaking());

      // First join to create socket
      act(() => {
        result.current.joinQueue(createPreferences());
      });

      // Then leave
      act(() => {
        result.current.leaveQueue();
      });

      expect(mockSocket.emit).toHaveBeenCalledWith('matchmaking:leave');
    });

    it('should reset queue state when leaving', () => {
      const { result } = renderHook(() => useMatchmaking());

      act(() => {
        result.current.leaveQueue();
      });

      expect(result.current.inQueue).toBe(false);
      expect(result.current.estimatedWaitTime).toBeNull();
      expect(result.current.queuePosition).toBeNull();
      expect(result.current.searchCriteria).toBeNull();
    });
  });

  describe('clearError', () => {
    it('should clear error state', () => {
      const { result } = renderHook(() => useMatchmaking());

      // Join to trigger socket creation and register handlers
      act(() => {
        result.current.joinQueue(createPreferences());
      });

      // Simulate an error via the registered handler
      act(() => {
        eventHandlers['error']?.({ event: 'matchmaking:join', message: 'Test error' });
      });

      expect(result.current.error).toBe('Test error');

      // Clear the error
      act(() => {
        result.current.clearError();
      });

      expect(result.current.error).toBeNull();
    });
  });

  describe('socket event handling', () => {
    it('should update state when matchmaking-status event is received', () => {
      const { result } = renderHook(() => useMatchmaking());

      // Join to trigger socket creation
      act(() => {
        result.current.joinQueue(createPreferences());
      });

      // Simulate status update
      act(() => {
        eventHandlers['matchmaking-status']?.({
          inQueue: true,
          estimatedWaitTime: 30000,
          queuePosition: 5,
          searchCriteria: createPreferences(),
        });
      });

      expect(result.current.inQueue).toBe(true);
      expect(result.current.estimatedWaitTime).toBe(30000);
      expect(result.current.queuePosition).toBe(5);
    });

    it('should set matchFound and navigate on match-found event', () => {
      const { result } = renderHook(() => useMatchmaking(true));

      // Join to trigger socket creation
      act(() => {
        result.current.joinQueue(createPreferences());
      });

      // Simulate match found
      act(() => {
        eventHandlers['match-found']?.({ gameId: 'new-game-123' });
      });

      expect(result.current.matchFound).toBe(true);
      expect(result.current.matchedGameId).toBe('new-game-123');
      expect(result.current.inQueue).toBe(false);
      expect(mockNavigate).toHaveBeenCalledWith('/game/new-game-123');
    });

    it('should not navigate when autoNavigate is false', () => {
      const { result } = renderHook(() => useMatchmaking(false));

      // Join to trigger socket creation
      act(() => {
        result.current.joinQueue(createPreferences());
      });

      // Simulate match found
      act(() => {
        eventHandlers['match-found']?.({ gameId: 'new-game-123' });
      });

      expect(result.current.matchFound).toBe(true);
      expect(result.current.matchedGameId).toBe('new-game-123');
      expect(mockNavigate).not.toHaveBeenCalled();
    });

    it('should clear isConnecting on connect event', () => {
      const { result } = renderHook(() => useMatchmaking());

      // Join to set isConnecting
      act(() => {
        result.current.joinQueue(createPreferences());
      });

      expect(result.current.isConnecting).toBe(true);

      // Simulate connect
      act(() => {
        eventHandlers['connect']?.();
      });

      expect(result.current.isConnecting).toBe(false);
    });

    it('should set error on connect_error event', () => {
      const { result } = renderHook(() => useMatchmaking());

      // Join to trigger socket creation
      act(() => {
        result.current.joinQueue(createPreferences());
      });

      // Simulate connect error
      act(() => {
        eventHandlers['connect_error']?.({ message: 'Connection failed' });
      });

      expect(result.current.error).toBe('Connection error: Connection failed');
      expect(result.current.isConnecting).toBe(false);
    });
  });
});
