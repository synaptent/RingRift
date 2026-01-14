/**
 * connectRedis / getRedisClient / disconnectRedis tests
 *
 * These tests exercise the connection lifecycle logic in
 * src/server/cache/redis.ts without talking to a real Redis instance.
 */

// Mock redis.createClient to return a lightweight client stub.
const mockCreateClient = jest.fn();
const mockOn = jest.fn();
const mockConnect = jest.fn<Promise<void>, []>(async () => {});
const mockQuit = jest.fn<Promise<void>, []>(async () => {});

jest.mock('redis', () => ({
  createClient: (...args: unknown[]) => mockCreateClient(...args),
}));

// Mock logger so we can assert error/info logging behaviour.
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

// Mock config.redis so connectRedis has a deterministic URL/password.
jest.mock('../../src/server/config', () => ({
  config: {
    redis: {
      url: 'redis://localhost:6379',
      password: 'test-password',
    },
  },
}));

// Mock rateLimiter initializer to verify it is invoked with null.
jest.mock('../../src/server/middleware/rateLimiter', () => ({
  initializeRateLimiters: jest.fn(),
}));

import { connectRedis, getRedisClient, disconnectRedis } from '../../src/server/cache/redis';
import { logger } from '../../src/server/utils/logger';

describe('redis connection lifecycle', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // For each test, have createClient return a fresh stub client.
    mockCreateClient.mockReturnValue({
      on: mockOn,
      connect: mockConnect,
      quit: mockQuit,
    });
  });

  it('connectRedis configures client with URL, password, and reconnect strategy, and initializes rate limiters', async () => {
    const client = await connectRedis();

    // createClient should be called once with expected options.
    expect(mockCreateClient).toHaveBeenCalledTimes(1);
    const options = mockCreateClient.mock.calls[0][0] as {
      url: string;
      disableOfflineQueue: boolean;
      socket: { connectTimeout: number; reconnectStrategy: (retries: number) => number | false };
      password?: string;
    };

    expect(options.url).toBe('redis://localhost:6379');
    expect(options.disableOfflineQueue).toBe(true);
    expect(options.password).toBe('test-password');
    expect(options.socket.connectTimeout).toBe(60000);

    // Reconnect strategy should back off and eventually stop retrying after >10 attempts.
    expect(options.socket.reconnectStrategy(1)).toBeGreaterThan(0);
    expect(options.socket.reconnectStrategy(5)).toBeLessThanOrEqual(1000);
    expect(options.socket.reconnectStrategy(11)).toBe(false);

    // Client should be connected once.
    expect(mockConnect).toHaveBeenCalledTimes(1);

    // Event handlers wired up for key lifecycle events.
    const events = mockOn.mock.calls.map((c) => c[0]);
    expect(events).toEqual(
      expect.arrayContaining(['error', 'connect', 'ready', 'end', 'reconnecting'])
    );

    // Rate limiters are initialized with the Redis client (falls back to null only on error).
    const { initializeRateLimiters } = require('../../src/server/middleware/rateLimiter');
    expect(initializeRateLimiters).toHaveBeenCalledWith(client);

    // getRedisClient should return the same instance.
    expect(getRedisClient()).toBe(client);
  });

  it('disconnectRedis quits the client and clears global reference', async () => {
    await connectRedis();
    expect(getRedisClient()).not.toBeNull();

    await disconnectRedis();

    expect(mockQuit).toHaveBeenCalledTimes(1);
    expect(getRedisClient()).toBeNull();
  });

  it('propagates connection errors and logs them', async () => {
    const error = new Error('connect failed');
    mockConnect.mockRejectedValueOnce(error);

    await expect(connectRedis()).rejects.toThrow('connect failed');

    expect(logger.error).toHaveBeenCalledWith(
      'Failed to connect to Redis:',
      expect.objectContaining({ message: 'connect failed' })
    );
  });
});
