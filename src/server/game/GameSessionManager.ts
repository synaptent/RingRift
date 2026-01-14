import { Server as SocketIOServer } from 'socket.io';
import { GameSession } from './GameSession';
import { PythonRulesClient } from '../services/PythonRulesClient';
import { getCacheService } from '../cache/redis';
import { logger } from '../utils/logger';
import { getMetricsService } from '../services/MetricsService';
import type { ClientToServerEvents, ServerToClientEvents } from '../../shared/types/websocket';

/**
 * Metadata tracked alongside each GameSession for lifecycle management.
 */
interface SessionEntry {
  session: GameSession;
  /** Timestamp when session was last accessed (created, move made, etc.) */
  lastActivityAt: number;
}

export class GameSessionManager {
  private sessions: Map<string, SessionEntry> = new Map();
  private pythonRulesClient: PythonRulesClient;
  private cleanupIntervalHandle: NodeJS.Timeout | null = null;

  /** Max age in ms before a completed/abandoned session is evicted (24 hours) */
  private static readonly MAX_COMPLETED_SESSION_AGE_MS = 24 * 60 * 60 * 1000;
  /** Max age in ms before any inactive session is evicted (48 hours) */
  private static readonly MAX_SESSION_AGE_MS = 48 * 60 * 60 * 1000;
  /** Cleanup interval in ms (5 minutes) */
  private static readonly CLEANUP_INTERVAL_MS = 5 * 60 * 1000;

  constructor(
    private io: SocketIOServer<ClientToServerEvents, ServerToClientEvents>,
    private userSockets: Map<string, string>
  ) {
    this.pythonRulesClient = new PythonRulesClient();
    this.startCleanupInterval();
  }

  /**
   * Start periodic cleanup of stale sessions to prevent memory leaks.
   */
  private startCleanupInterval(): void {
    if (this.cleanupIntervalHandle) return;

    this.cleanupIntervalHandle = setInterval(() => {
      this.cleanupStaleSessions();
    }, GameSessionManager.CLEANUP_INTERVAL_MS);

    // Don't keep the process alive just for cleanup
    this.cleanupIntervalHandle.unref();

    logger.info('GameSessionManager cleanup interval started', {
      intervalMs: GameSessionManager.CLEANUP_INTERVAL_MS,
    });
  }

  /**
   * Stop the cleanup interval (for graceful shutdown).
   */
  public stopCleanupInterval(): void {
    if (this.cleanupIntervalHandle) {
      clearInterval(this.cleanupIntervalHandle);
      this.cleanupIntervalHandle = null;
      logger.info('GameSessionManager cleanup interval stopped');
    }
  }

  /**
   * Evict sessions that are stale based on age and completion status.
   * - Completed/abandoned sessions: evicted after 24 hours of inactivity
   * - Any session: evicted after 48 hours of inactivity
   */
  private cleanupStaleSessions(): void {
    const now = Date.now();
    let evictedCount = 0;

    for (const [gameId, entry] of this.sessions.entries()) {
      const age = now - entry.lastActivityAt;
      const snapshot = entry.session.getSessionStatusSnapshot();
      const isTerminal = snapshot?.kind === 'completed' || snapshot?.kind === 'abandoned';

      // Evict completed/abandoned sessions after 24h, any session after 48h
      const maxAge = isTerminal
        ? GameSessionManager.MAX_COMPLETED_SESSION_AGE_MS
        : GameSessionManager.MAX_SESSION_AGE_MS;

      if (age > maxAge) {
        // Update metrics before removal
        if (snapshot?.kind) {
          getMetricsService().updateGameSessionStatusCurrent(snapshot.kind, null);
        }

        this.sessions.delete(gameId);
        evictedCount++;

        logger.info('Evicted stale game session', {
          gameId,
          ageHours: Math.round(age / (60 * 60 * 1000)),
          status: snapshot?.kind ?? 'unknown',
        });
      }
    }

    if (evictedCount > 0) {
      logger.info('Session cleanup completed', {
        evictedCount,
        remainingSessions: this.sessions.size,
      });
    }
  }

  public async getOrCreateSession(gameId: string): Promise<GameSession> {
    const existing = this.sessions.get(gameId);
    if (existing) {
      // Update last activity time on access
      existing.lastActivityAt = Date.now();
      return existing.session;
    }

    // Create a bound lock function for this specific game
    const withLockForGame = <T>(operation: () => Promise<T>): Promise<T> => {
      return this.withGameLock(gameId, operation);
    };

    const session = new GameSession(
      gameId,
      this.io,
      this.pythonRulesClient,
      this.userSockets,
      withLockForGame
    );

    await session.initialize();

    const entry: SessionEntry = {
      session,
      lastActivityAt: Date.now(),
    };
    this.sessions.set(gameId, entry);

    return session;
  }

  public getSession(gameId: string): GameSession | undefined {
    const entry = this.sessions.get(gameId);
    if (entry) {
      // Update last activity time on access
      entry.lastActivityAt = Date.now();
      return entry.session;
    }
    return undefined;
  }

  public removeSession(gameId: string): void {
    const entry = this.sessions.get(gameId);

    // Keep the ringrift_game_session_status_current gauge in sync when a
    // session is explicitly torn down. We decrement the gauge for the
    // session's last known derived status kind, if available.
    if (entry) {
      try {
        const snapshot = entry.session.getSessionStatusSnapshot();
        const currentKind = snapshot?.kind ?? null;
        if (currentKind) {
          getMetricsService().updateGameSessionStatusCurrent(currentKind, null);
        }
      } catch (err) {
        logger.warn('Failed to update session status gauge on removeSession', {
          gameId,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    this.sessions.delete(gameId);
  }

  /**
   * Get the current number of tracked sessions (for monitoring).
   */
  public getSessionCount(): number {
    return this.sessions.size;
  }

  /**
   * Touch a session to update its last activity timestamp.
   * Call this when significant activity occurs (moves, choices, etc.).
   */
  public touchSession(gameId: string): void {
    const entry = this.sessions.get(gameId);
    if (entry) {
      entry.lastActivityAt = Date.now();
    }
  }

  /**
   * Execute an operation with a distributed lock on the gameId.
   * This prevents race conditions where multiple requests (e.g. concurrent moves)
   * attempt to modify the game state simultaneously.
   *
   * P0 FIX (2026-01-11): Increased TTL from 5s to 15s to handle complex operations
   * that involve Python parity checks, database persistence, and broadcasting.
   * Operations typically complete in 100-500ms, but network issues or database
   * latency can occasionally extend this. The 15s TTL provides safety margin
   * while still preventing deadlocks from crashed processes.
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
    // P0 FIX: Increased from 5s to 15s to prevent lock expiration during
    // complex operations (Python rules calls, DB persistence, broadcasts).
    const ttlSeconds = 15;
    const maxRetries = 8; // More retries with longer wait between
    const retryDelayMs = 250;

    for (let i = 0; i < maxRetries; i++) {
      const acquired = await cacheService.acquireLock(lockKey, ttlSeconds);
      if (acquired) {
        const startTime = Date.now();
        try {
          const result = await operation();
          const elapsed = Date.now() - startTime;
          // Warn if operation took more than 50% of TTL
          if (elapsed > ttlSeconds * 500) {
            logger.warn('Game lock operation took significant time', {
              gameId,
              elapsedMs: elapsed,
              ttlSeconds,
            });
          }
          return result;
        } finally {
          await cacheService.releaseLock(lockKey);
        }
      }

      // Wait before retrying with exponential backoff
      const backoffDelay = retryDelayMs * Math.min(2 ** i, 4);
      await new Promise((resolve) => setTimeout(resolve, backoffDelay));
    }

    throw new Error('Game is busy, please try again');
  }
}
