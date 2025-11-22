import { createClient } from 'redis';
import { logger } from '../utils/logger';
import { initializeRateLimiters } from '../middleware/rateLimiter';

type RedisClient = ReturnType<typeof createClient>;
let redisClient: RedisClient | null = null;

export const connectRedis = async (): Promise<RedisClient> => {
  try {
    const clientOptions: any = {
      url: process.env.REDIS_URL || 'redis://localhost:6379',
      socket: {
        connectTimeout: 60000,
        reconnectStrategy: (retries: number) => {
          if (retries > 10) {
            logger.error('Redis reconnection failed after 10 attempts');
            return false;
          }
          return Math.min(retries * 50, 1000);
        },
      },
    };

    if (process.env.REDIS_PASSWORD) {
      clientOptions.password = process.env.REDIS_PASSWORD;
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
    initializeRateLimiters(client);

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
      return value ? JSON.parse(value) : null;
    } catch (error) {
      logger.error(`Cache get error for key ${key}:`, error);
      return null;
    }
  }

  async set(key: string, value: any, ttlSeconds?: number): Promise<boolean> {
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
      return result || null;
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
      return await this.client.hGetAll(key);
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
      return await this.client.sMembers(key);
    } catch (error) {
      logger.error(`Cache sMembers error for key ${key}:`, error);
      return [];
    }
  }

  async sIsMember(key: string, member: string): Promise<boolean> {
    try {
      return await this.client.sIsMember(key, member);
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
};
