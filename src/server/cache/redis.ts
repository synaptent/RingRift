import { createClient } from 'redis';
import { logger } from '../utils/logger';
import { initializeRateLimiters } from '../middleware/rateLimiter';
import { config } from '../config';
import { getMetricsService } from '../services/MetricsService';

type RedisClient = ReturnType<typeof createClient>;
let redisClient: RedisClient | null = null;

export const connectRedis = async (): Promise<RedisClient> => {
  try {
    const clientOptions: Parameters<typeof createClient>[0] = {
      url: config.redis.url,
      // rate-limiter-flexible requires enableOfflineQueue: false for redis v4+ compatibility
      // to ensure Lua scripts are properly registered and executed.
      // See: https://github.com/animir/node-rate-limiter-flexible/wiki/Redis#redis-v4-support
      disableOfflineQueue: true,
      socket: {
        connectTimeout: 60000,
        reconnectStrategy: (retries: number) => {
          // Never give up on Redis - use exponential backoff with cap at 30 seconds
          // This ensures the application can recover from Redis outages automatically
          const delay = Math.min(retries * 100, 30000);
          if (retries % 10 === 0) {
            logger.warn(`Redis reconnection attempt ${retries}, next retry in ${delay}ms`);
          }
          return delay;
        },
      },
    };

    if (config.redis.password && clientOptions) {
      clientOptions.password = config.redis.password;
    }

    const client = createClient(clientOptions);

    client.on('error', (error) => {
      logger.error('Redis Client Error:', error);
    });

    client.on('connect', () => {
      logger.info('Redis client connected');
    });

    client.on('ready', () => {
      logger.info('Redis client ready');
    });

    client.on('end', () => {
      logger.info('Redis client disconnected');
    });

    client.on('reconnecting', () => {
      logger.info('Redis client reconnecting...');
    });

    await client.connect();
    redisClient = client;

    // Initialize rate limiters with Redis client
    // Note: rate-limiter-flexible@9+ with redis@5+ requires the client to be connected
    // and ready before passing. The Lua scripts are auto-registered on first use.
    // If Redis rate limiting fails, the system falls back to memory-based limiting
    // per the graceful degradation in rateLimiter.ts.
    try {
      initializeRateLimiters(client);
    } catch (rateLimiterError) {
      logger.warn('Failed to initialize Redis-backed rate limiters, falling back to memory', {
        error:
          rateLimiterError instanceof Error ? rateLimiterError.message : String(rateLimiterError),
      });
      initializeRateLimiters(null);
    }

    return client;
  } catch (error) {
    logger.error('Failed to connect to Redis:', error);
    throw error;
  }
};

export const getRedisClient = (): RedisClient | null => {
  return redisClient;
};

export const disconnectRedis = async (): Promise<void> => {
  if (redisClient) {
    await redisClient.quit();
    redisClient = null;
    logger.info('Redis client disconnected');
  }
};

// Cache utility functions
export class CacheService {
  private client: RedisClient;

  constructor(client: RedisClient) {
    this.client = client;
  }

  async get<T>(key: string): Promise<T | null> {
    try {
      const value = await this.client.get(key);
      if (value) {
        getMetricsService().recordCacheHit();
        try {
          return JSON.parse(value);
        } catch (parseError) {
          // Corrupted cache entry - log and delete it
          logger.warn(`Corrupted cache value for key ${key}, deleting entry`, {
            error: parseError instanceof Error ? parseError.message : String(parseError),
            valueLength: value.length,
            valuePreview: value.substring(0, 100),
          });
          // Delete corrupted entry asynchronously (don't await to avoid blocking)
          this.del(key).catch((delError) => {
            logger.error(`Failed to delete corrupted cache key ${key}:`, delError);
          });
          return null;
        }
      }
      getMetricsService().recordCacheMiss();
      return null;
    } catch (error) {
      logger.error(`Cache get error for key ${key}:`, error);
      return null;
    }
  }

  async set<T>(key: string, value: T, ttlSeconds?: number): Promise<boolean> {
    try {
      const serialized = JSON.stringify(value);
      if (ttlSeconds) {
        await this.client.setEx(key, ttlSeconds, serialized);
      } else {
        await this.client.set(key, serialized);
      }
      return true;
    } catch (error) {
      logger.error(`Cache set error for key ${key}:`, error);
      return false;
    }
  }

  async del(key: string): Promise<boolean> {
    try {
      await this.client.del(key);
      return true;
    } catch (error) {
      logger.error(`Cache delete error for key ${key}:`, error);
      return false;
    }
  }

  async exists(key: string): Promise<boolean> {
    try {
      const result = await this.client.exists(key);
      if (result === 1) {
        getMetricsService().recordCacheHit();
      } else {
        getMetricsService().recordCacheMiss();
      }
      return result === 1;
    } catch (error) {
      logger.error(`Cache exists error for key ${key}:`, error);
      return false;
    }
  }

  async expire(key: string, ttlSeconds: number): Promise<boolean> {
    try {
      await this.client.expire(key, ttlSeconds);
      return true;
    } catch (error) {
      logger.error(`Cache expire error for key ${key}:`, error);
      return false;
    }
  }

  async keys(pattern: string): Promise<string[]> {
    try {
      return await this.client.keys(pattern);
    } catch (error) {
      logger.error(`Cache keys error for pattern ${pattern}:`, error);
      return [];
    }
  }

  async flushAll(): Promise<boolean> {
    try {
      await this.client.flushAll();
      return true;
    } catch (error) {
      logger.error('Cache flush all error:', error);
      return false;
    }
  }

  // Hash operations
  async hGet(key: string, field: string): Promise<string | null> {
    try {
      const result = await this.client.hGet(key, field);
      if (result) {
        getMetricsService().recordCacheHit();
        return result;
      }
      getMetricsService().recordCacheMiss();
      return null;
    } catch (error) {
      logger.error(`Cache hGet error for key ${key}, field ${field}:`, error);
      return null;
    }
  }

  async hSet(key: string, field: string, value: string): Promise<boolean> {
    try {
      await this.client.hSet(key, field, value);
      return true;
    } catch (error) {
      logger.error(`Cache hSet error for key ${key}, field ${field}:`, error);
      return false;
    }
  }

  async hGetAll(key: string): Promise<Record<string, string>> {
    try {
      const result = await this.client.hGetAll(key);
      if (Object.keys(result).length > 0) {
        getMetricsService().recordCacheHit();
      } else {
        getMetricsService().recordCacheMiss();
      }
      return result;
    } catch (error) {
      logger.error(`Cache hGetAll error for key ${key}:`, error);
      return {};
    }
  }

  async hDel(key: string, field: string): Promise<boolean> {
    try {
      await this.client.hDel(key, field);
      return true;
    } catch (error) {
      logger.error(`Cache hDel error for key ${key}, field ${field}:`, error);
      return false;
    }
  }

  // List operations
  async lPush(key: string, ...values: string[]): Promise<number> {
    try {
      return await this.client.lPush(key, values);
    } catch (error) {
      logger.error(`Cache lPush error for key ${key}:`, error);
      return 0;
    }
  }

  async rPush(key: string, ...values: string[]): Promise<number> {
    try {
      return await this.client.rPush(key, values);
    } catch (error) {
      logger.error(`Cache rPush error for key ${key}:`, error);
      return 0;
    }
  }

  async lPop(key: string): Promise<string | null> {
    try {
      return await this.client.lPop(key);
    } catch (error) {
      logger.error(`Cache lPop error for key ${key}:`, error);
      return null;
    }
  }

  async rPop(key: string): Promise<string | null> {
    try {
      return await this.client.rPop(key);
    } catch (error) {
      logger.error(`Cache rPop error for key ${key}:`, error);
      return null;
    }
  }

  async lRange(key: string, start: number, stop: number): Promise<string[]> {
    try {
      return await this.client.lRange(key, start, stop);
    } catch (error) {
      logger.error(`Cache lRange error for key ${key}:`, error);
      return [];
    }
  }

  // Set operations
  async sAdd(key: string, ...members: string[]): Promise<number> {
    try {
      return await this.client.sAdd(key, members);
    } catch (error) {
      logger.error(`Cache sAdd error for key ${key}:`, error);
      return 0;
    }
  }

  async sRem(key: string, ...members: string[]): Promise<number> {
    try {
      return await this.client.sRem(key, members);
    } catch (error) {
      logger.error(`Cache sRem error for key ${key}:`, error);
      return 0;
    }
  }

  async sMembers(key: string): Promise<string[]> {
    try {
      const result = await this.client.sMembers(key);
      if (result.length > 0) {
        getMetricsService().recordCacheHit();
      } else {
        getMetricsService().recordCacheMiss();
      }
      return result;
    } catch (error) {
      logger.error(`Cache sMembers error for key ${key}:`, error);
      return [];
    }
  }

  async sIsMember(key: string, member: string): Promise<boolean> {
    try {
      const result = await this.client.sIsMember(key, member);
      // redis.sIsMember returns 1 when the member exists, 0 otherwise.
      // Normalizing to a boolean keeps the public API consistent with the
      // rest of CacheService (e.g. `exists`).
      if (result === 1) {
        getMetricsService().recordCacheHit();
        return true;
      }
      getMetricsService().recordCacheMiss();
      return false;
    } catch (error) {
      logger.error(`Cache sIsMember error for key ${key}, member ${member}:`, error);
      return false;
    }
  }

  // Locking operations
  async acquireLock(key: string, ttlSeconds: number): Promise<boolean> {
    try {
      // SET key value NX EX ttlSeconds
      // Returns 'OK' if set, null if not set (already exists)
      const result = await this.client.set(key, 'locked', {
        NX: true,
        EX: ttlSeconds,
      });
      return result === 'OK';
    } catch (error) {
      logger.error(`Cache acquireLock error for key ${key}:`, error);
      return false;
    }
  }

  async releaseLock(key: string): Promise<boolean> {
    try {
      await this.client.del(key);
      return true;
    } catch (error) {
      logger.error(`Cache releaseLock error for key ${key}:`, error);
      return false;
    }
  }
}

// Global cache service instance
let cacheService: CacheService | null = null;

export const getCacheService = (): CacheService | null => {
  if (!redisClient) {
    return null;
  }

  if (!cacheService) {
    cacheService = new CacheService(redisClient);
  }

  return cacheService;
};

// Cache key generators
export const CacheKeys = {
  user: (userId: string) => `user:${userId}`,
  userSession: (sessionId: string) => `session:${sessionId}`,
  game: (gameId: string) => `game:${gameId}`,
  gameState: (gameId: string) => `game:${gameId}:state`,
  userGames: (userId: string) => `user:${userId}:games`,
  onlineUsers: () => 'users:online',
  gameQueue: (boardType: string) => `queue:${boardType}`,
  userStats: (userId: string) => `user:${userId}:stats`,
  leaderboard: (boardType: string) => `leaderboard:${boardType}`,
  tournament: (tournamentId: string) => `tournament:${tournamentId}`,
  chatHistory: (gameId: string) => `chat:${gameId}:history`,
  // Auth / login abuse-protection
  authLoginFailures: (email: string) => `auth:login:failures:${email}`,
  authLoginLockout: (email: string) => `auth:login:lockout:${email}`,
};
