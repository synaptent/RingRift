import { Server as SocketIOServer } from 'socket.io';
import { GameSession } from './GameSession';
import { PythonRulesClient } from '../services/PythonRulesClient';
import { getCacheService } from '../cache/redis';
import { logger } from '../utils/logger';
import type { ClientToServerEvents, ServerToClientEvents } from '../../shared/types/websocket';

export class GameSessionManager {
  private sessions: Map<string, GameSession> = new Map();
  private pythonRulesClient: PythonRulesClient;

  constructor(
    private io: SocketIOServer<ClientToServerEvents, ServerToClientEvents>,
    private userSockets: Map<string, string>
  ) {
    this.pythonRulesClient = new PythonRulesClient();
  }

  public async getOrCreateSession(gameId: string): Promise<GameSession> {
    if (this.sessions.has(gameId)) {
      return this.sessions.get(gameId)!;
    }

    const session = new GameSession(gameId, this.io, this.pythonRulesClient, this.userSockets);

    await session.initialize();
    this.sessions.set(gameId, session);
    return session;
  }

  public getSession(gameId: string): GameSession | undefined {
    return this.sessions.get(gameId);
  }

  public removeSession(gameId: string): void {
    this.sessions.delete(gameId);
  }

  /**
   * Execute an operation with a distributed lock on the gameId.
   * This prevents race conditions where multiple requests (e.g. concurrent moves)
   * attempt to modify the game state simultaneously.
   */
  public async withGameLock<T>(gameId: string, operation: () => Promise<T>): Promise<T> {
    const cacheService = getCacheService();

    // If Redis is not available, fall back to executing without a lock.
    // This degrades gracefully but reintroduces race condition risks.
    if (!cacheService) {
      logger.warn('Redis not available for locking, proceeding without lock', { gameId });
      return operation();
    }

    const lockKey = `lock:game:${gameId}`;
    const ttlSeconds = 5; // Short TTL sufficient for move processing
    const maxRetries = 5;
    const retryDelayMs = 200;

    for (let i = 0; i < maxRetries; i++) {
      const acquired = await cacheService.acquireLock(lockKey, ttlSeconds);
      if (acquired) {
        try {
          return await operation();
        } finally {
          await cacheService.releaseLock(lockKey);
        }
      }

      // Wait before retrying
      await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
    }

    throw new Error('Game is busy, please try again');
  }
}
