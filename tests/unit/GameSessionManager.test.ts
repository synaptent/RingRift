/**
 * Tests for GameSessionManager.
 *
 * This file covers:
 * - Session creation and retrieval (getOrCreateSession, getSession)
 * - Session removal with status gauge updates
 * - Distributed locking (withGameLock)
 * - Concurrent operation handling
 */

import { Server as SocketIOServer } from 'socket.io';
import { GameSessionManager } from '../../src/server/game/GameSessionManager';
import { GameSession } from '../../src/server/game/GameSession';

// Mock GameSession
jest.mock('../../src/server/game/GameSession', () => ({
  GameSession: jest.fn().mockImplementation((gameId: string) => ({
    gameId,
    initialize: jest.fn().mockResolvedValue(undefined),
    getSessionStatusSnapshot: jest.fn(() => ({ kind: 'active_turn' })),
    terminate: jest.fn(),
    getGameState: jest.fn(() => ({
      id: gameId,
      gameStatus: 'active',
      players: [],
    })),
  })),
}));

// Mock PythonRulesClient
jest.mock('../../src/server/services/PythonRulesClient', () => ({
  PythonRulesClient: jest.fn().mockImplementation(() => ({
    evaluateMove: jest.fn(),
    healthCheck: jest.fn(),
  })),
}));

// Mock MetricsService
jest.mock('../../src/server/services/MetricsService', () => ({
  getMetricsService: () => ({
    updateGameSessionStatusCurrent: jest.fn(),
    recordGameSessionStatusTransition: jest.fn(),
  }),
}));

// Mock Redis cache service
const mockAcquireLock = jest.fn();
const mockReleaseLock = jest.fn();

jest.mock('../../src/server/cache/redis', () => ({
  getCacheService: jest.fn(() => ({
    acquireLock: mockAcquireLock,
    releaseLock: mockReleaseLock,
  })),
}));

const createMockIo = (): SocketIOServer =>
  ({
    to: jest.fn().mockReturnThis(),
    emit: jest.fn(),
    sockets: {
      adapter: {
        rooms: new Map(),
      },
      sockets: new Map(),
    },
  }) as any;

describe('GameSessionManager', () => {
  let io: SocketIOServer;
  let userSockets: Map<string, string>;
  let manager: GameSessionManager;

  beforeEach(() => {
    jest.clearAllMocks();
    io = createMockIo();
    userSockets = new Map();
    manager = new GameSessionManager(io, userSockets);
    mockAcquireLock.mockResolvedValue(true);
    mockReleaseLock.mockResolvedValue(undefined);
  });

  describe('getOrCreateSession', () => {
    it('creates new session for unknown gameId', async () => {
      const session = await manager.getOrCreateSession('game-1');

      expect(session).toBeDefined();
      expect(session.gameId).toBe('game-1');
      expect(GameSession).toHaveBeenCalledWith('game-1', io, expect.anything(), userSockets);
      expect(session.initialize).toHaveBeenCalled();
    });

    it('returns existing session for known gameId', async () => {
      const session1 = await manager.getOrCreateSession('game-1');
      const session2 = await manager.getOrCreateSession('game-1');

      expect(session1).toBe(session2);
      expect(GameSession).toHaveBeenCalledTimes(1);
    });

    it('creates separate sessions for different gameIds', async () => {
      const session1 = await manager.getOrCreateSession('game-1');
      const session2 = await manager.getOrCreateSession('game-2');

      expect(session1).not.toBe(session2);
      expect(session1.gameId).toBe('game-1');
      expect(session2.gameId).toBe('game-2');
      expect(GameSession).toHaveBeenCalledTimes(2);
    });
  });

  describe('getSession', () => {
    it('returns undefined for unknown gameId', () => {
      const session = manager.getSession('unknown-game');

      expect(session).toBeUndefined();
    });

    it('returns session for known gameId', async () => {
      await manager.getOrCreateSession('game-1');
      const session = manager.getSession('game-1');

      expect(session).toBeDefined();
      expect(session?.gameId).toBe('game-1');
    });
  });

  describe('removeSession', () => {
    it('removes existing session', async () => {
      await manager.getOrCreateSession('game-1');
      expect(manager.getSession('game-1')).toBeDefined();

      manager.removeSession('game-1');

      expect(manager.getSession('game-1')).toBeUndefined();
    });

    it('handles removal of non-existent session gracefully', () => {
      // Should not throw
      expect(() => manager.removeSession('non-existent')).not.toThrow();
    });

    it('calls getSessionStatusSnapshot on removal', async () => {
      const session = await manager.getOrCreateSession('game-1');

      manager.removeSession('game-1');

      // Verify snapshot was called to get status for metrics
      expect(session.getSessionStatusSnapshot).toHaveBeenCalled();
    });
  });

  describe('withGameLock', () => {
    it('acquires and releases lock for operation', async () => {
      const operation = jest.fn().mockResolvedValue('result');

      const result = await manager.withGameLock('game-1', operation);

      expect(result).toBe('result');
      expect(mockAcquireLock).toHaveBeenCalledWith('lock:game:game-1', 5);
      expect(operation).toHaveBeenCalled();
      expect(mockReleaseLock).toHaveBeenCalledWith('lock:game:game-1');
    });

    it('releases lock even if operation throws', async () => {
      const operation = jest.fn().mockRejectedValue(new Error('Operation failed'));

      await expect(manager.withGameLock('game-1', operation)).rejects.toThrow('Operation failed');

      expect(mockAcquireLock).toHaveBeenCalled();
      expect(mockReleaseLock).toHaveBeenCalled();
    });

    it('retries lock acquisition on failure', async () => {
      mockAcquireLock
        .mockResolvedValueOnce(false)
        .mockResolvedValueOnce(false)
        .mockResolvedValueOnce(true);

      const operation = jest.fn().mockResolvedValue('result');

      const result = await manager.withGameLock('game-1', operation);

      expect(result).toBe('result');
      expect(mockAcquireLock).toHaveBeenCalledTimes(3);
    });

    it('throws error after max retries', async () => {
      mockAcquireLock.mockResolvedValue(false);

      const operation = jest.fn().mockResolvedValue('result');

      await expect(manager.withGameLock('game-1', operation)).rejects.toThrow(
        'Game is busy, please try again'
      );

      expect(mockAcquireLock).toHaveBeenCalledTimes(5); // maxRetries = 5
      expect(operation).not.toHaveBeenCalled();
    });

    it('proceeds without lock when Redis is unavailable', async () => {
      // Mock getCacheService to return null
      const redisModule = require('../../src/server/cache/redis');
      redisModule.getCacheService.mockReturnValueOnce(null);

      const operation = jest.fn().mockResolvedValue('result');

      const result = await manager.withGameLock('game-1', operation);

      expect(result).toBe('result');
      expect(operation).toHaveBeenCalled();
      // Lock functions should not be called
      expect(mockAcquireLock).not.toHaveBeenCalled();
    });
  });

  describe('concurrent operations', () => {
    it('returns same session for sequential calls with same gameId', async () => {
      // Sequential requests should return same session
      const session1 = await manager.getOrCreateSession('game-1');
      const session2 = await manager.getOrCreateSession('game-1');
      const session3 = await manager.getOrCreateSession('game-1');

      // All should return the same session instance
      expect(session1).toBe(session2);
      expect(session2).toBe(session3);
    });

    it('handles multiple concurrent session creations for different gameIds', async () => {
      const [session1, session2, session3] = await Promise.all([
        manager.getOrCreateSession('game-1'),
        manager.getOrCreateSession('game-2'),
        manager.getOrCreateSession('game-3'),
      ]);

      expect(session1.gameId).toBe('game-1');
      expect(session2.gameId).toBe('game-2');
      expect(session3.gameId).toBe('game-3');
    });

    it('returns existing session on second call after first completes', async () => {
      const session1 = await manager.getOrCreateSession('game-1');

      // Second call should return same session
      const session2 = await manager.getOrCreateSession('game-1');

      expect(session1).toBe(session2);
      // GameSession constructor should only be called once
      expect(GameSession).toHaveBeenCalledTimes(1);
    });
  });
});

describe('GameSessionManager session lifecycle', () => {
  let io: SocketIOServer;
  let userSockets: Map<string, string>;
  let manager: GameSessionManager;

  beforeEach(() => {
    jest.clearAllMocks();
    io = createMockIo();
    userSockets = new Map();
    manager = new GameSessionManager(io, userSockets);
    mockAcquireLock.mockResolvedValue(true);
  });

  it('tracks multiple active sessions', async () => {
    await manager.getOrCreateSession('game-1');
    await manager.getOrCreateSession('game-2');
    await manager.getOrCreateSession('game-3');

    expect(manager.getSession('game-1')).toBeDefined();
    expect(manager.getSession('game-2')).toBeDefined();
    expect(manager.getSession('game-3')).toBeDefined();
  });

  it('removes sessions independently', async () => {
    await manager.getOrCreateSession('game-1');
    await manager.getOrCreateSession('game-2');
    await manager.getOrCreateSession('game-3');

    manager.removeSession('game-2');

    expect(manager.getSession('game-1')).toBeDefined();
    expect(manager.getSession('game-2')).toBeUndefined();
    expect(manager.getSession('game-3')).toBeDefined();
  });

  it('allows re-creation of removed session', async () => {
    const session1 = await manager.getOrCreateSession('game-1');
    manager.removeSession('game-1');

    const session2 = await manager.getOrCreateSession('game-1');

    expect(session2).not.toBe(session1);
    expect(session2.gameId).toBe('game-1');
  });
});

describe('GameSessionManager error handling', () => {
  let io: SocketIOServer;
  let userSockets: Map<string, string>;
  let manager: GameSessionManager;

  beforeEach(() => {
    jest.clearAllMocks();
    io = createMockIo();
    userSockets = new Map();
    manager = new GameSessionManager(io, userSockets);
  });

  it('handles session initialization failure', async () => {
    const MockGameSession = GameSession as jest.Mock;
    MockGameSession.mockImplementationOnce(() => ({
      gameId: 'game-1',
      initialize: jest.fn().mockRejectedValue(new Error('Init failed')),
    }));

    await expect(manager.getOrCreateSession('game-1')).rejects.toThrow('Init failed');
  });

  it('handles session status snapshot error gracefully', async () => {
    const MockGameSession = GameSession as jest.Mock;
    MockGameSession.mockImplementationOnce(() => ({
      gameId: 'game-1',
      initialize: jest.fn().mockResolvedValue(undefined),
      getSessionStatusSnapshot: jest.fn(() => {
        throw new Error('Snapshot error');
      }),
    }));

    await manager.getOrCreateSession('game-1');

    // Should not throw when removing
    expect(() => manager.removeSession('game-1')).not.toThrow();
  });
});
