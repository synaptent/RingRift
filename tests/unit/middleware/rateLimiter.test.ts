/**
 * Rate Limiter Middleware Unit Tests
 *
 * Tests for the rate limiting middleware including:
 * - Configuration loading
 * - Rate limit enforcement
 * - Header setting
 * - Fallback behavior
 * - Memory vs Redis modes
 */

import { Request, Response, NextFunction } from 'express';
import {
  getRateLimitConfigs,
  initializeMemoryRateLimiters,
  isUsingRedisRateLimiting,
  setRateLimitHeaders,
  consumeRateLimit,
  getRateLimitConfig,
  __testResetRateLimiters,
  rateLimiter,
  authLoginRateLimiter,
  fallbackRateLimiter,
} from '../../../src/server/middleware/rateLimiter';

// Mock dependencies
jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

jest.mock('../../../src/server/services/MetricsService', () => ({
  getMetricsService: () => ({
    recordRateLimitHit: jest.fn(),
  }),
}));

describe('Rate Limiter Configuration', () => {
  describe('getRateLimitConfigs', () => {
    it('should return all rate limit configurations', () => {
      const configs = getRateLimitConfigs();

      expect(configs.api).toBeDefined();
      expect(configs.apiAuthenticated).toBeDefined();
      expect(configs.auth).toBeDefined();
      expect(configs.authLogin).toBeDefined();
      expect(configs.authRegister).toBeDefined();
      expect(configs.game).toBeDefined();
      expect(configs.websocket).toBeDefined();
    });

    it('should have valid config structure for each limiter', () => {
      const configs = getRateLimitConfigs();

      Object.values(configs).forEach((config) => {
        expect(config.keyPrefix).toBeDefined();
        expect(typeof config.points).toBe('number');
        expect(typeof config.duration).toBe('number');
        expect(typeof config.blockDuration).toBe('number');
        expect(config.points).toBeGreaterThan(0);
        expect(config.duration).toBeGreaterThan(0);
      });
    });

    it('should have appropriate default values', () => {
      const configs = getRateLimitConfigs();

      // API should be relatively permissive
      expect(configs.api.points).toBe(50);
      expect(configs.api.duration).toBe(60);

      // Auth login should be restrictive
      expect(configs.authLogin.points).toBe(5);
      expect(configs.authLogin.duration).toBe(900);

      // Registration should be very restrictive
      expect(configs.authRegister.points).toBe(3);
    });
  });

  describe('getRateLimitConfig', () => {
    beforeEach(() => {
      initializeMemoryRateLimiters();
    });

    it('should return config for valid limiter key', () => {
      const config = getRateLimitConfig('api');

      expect(config).toBeDefined();
      expect(config?.keyPrefix).toBe('api_limit');
    });

    it('should return undefined for invalid limiter key', () => {
      const config = getRateLimitConfig('nonexistent');

      expect(config).toBeUndefined();
    });
  });
});

describe('Rate Limiter Initialization', () => {
  describe('initializeMemoryRateLimiters', () => {
    it('should initialize without Redis', () => {
      expect(() => initializeMemoryRateLimiters()).not.toThrow();
    });

    it('should report not using Redis after memory init', () => {
      initializeMemoryRateLimiters();
      expect(isUsingRedisRateLimiting()).toBe(false);
    });
  });
});

describe('Rate Limit Headers', () => {
  describe('setRateLimitHeaders', () => {
    let mockRes: Partial<Response>;
    let headers: Record<string, string>;

    beforeEach(() => {
      headers = {};
      mockRes = {
        set: jest.fn((key: string, value: string) => {
          headers[key] = value;
          return mockRes as Response;
        }),
      };
    });

    it('should set X-RateLimit-Limit header', () => {
      setRateLimitHeaders(mockRes as Response, 100, 50, 1234567890);

      expect(headers['X-RateLimit-Limit']).toBe('100');
    });

    it('should set X-RateLimit-Remaining header', () => {
      setRateLimitHeaders(mockRes as Response, 100, 50, 1234567890);

      expect(headers['X-RateLimit-Remaining']).toBe('50');
    });

    it('should set X-RateLimit-Reset header', () => {
      setRateLimitHeaders(mockRes as Response, 100, 50, 1234567890);

      expect(headers['X-RateLimit-Reset']).toBe('1234567890');
    });

    it('should clamp remaining to 0 when negative', () => {
      setRateLimitHeaders(mockRes as Response, 100, -5, 1234567890);

      expect(headers['X-RateLimit-Remaining']).toBe('0');
    });
  });
});

describe('consumeRateLimit', () => {
  beforeEach(() => {
    initializeMemoryRateLimiters();
    __testResetRateLimiters();
  });

  it('should allow requests within limit', async () => {
    const result = await consumeRateLimit('api', 'test-key-1');

    expect(result.allowed).toBe(true);
    expect(result.remaining).toBeDefined();
  });

  it('should return limit info on success', async () => {
    const result = await consumeRateLimit('api', 'test-key-2');

    expect(result.limit).toBe(50); // default api limit
    expect(result.remaining).toBeLessThan(50);
    expect(result.reset).toBeDefined();
  });

  it('should allow request for non-existent limiter with warning', async () => {
    const result = await consumeRateLimit('nonexistent', 'test-key');

    expect(result.allowed).toBe(true);
  });

  it('should block requests that exceed limit', async () => {
    // Use authLogin which has very low limit (5)
    const key = 'test-key-block';

    // Consume all allowed requests
    for (let i = 0; i < 5; i++) {
      await consumeRateLimit('authLogin', key);
    }

    // Next request should be blocked
    const result = await consumeRateLimit('authLogin', key);

    expect(result.allowed).toBe(false);
    expect(result.retryAfter).toBeDefined();
    expect(result.retryAfter).toBeGreaterThan(0);
  });
});

describe('Rate Limiter Middleware', () => {
  let mockReq: Partial<Request>;
  let mockRes: Partial<Response>;
  let mockNext: NextFunction;
  let headers: Record<string, string>;

  beforeEach(() => {
    initializeMemoryRateLimiters();
    __testResetRateLimiters();

    headers = {};
    mockReq = {
      ip: '127.0.0.1',
      path: '/api/test',
    };
    mockRes = {
      set: jest.fn((key: string, value: string) => {
        headers[key] = value;
        return mockRes as Response;
      }),
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };
    mockNext = jest.fn();
  });

  describe('rateLimiter (api)', () => {
    it('should allow requests within limit', async () => {
      mockReq.ip = 'unique-ip-1';

      await rateLimiter(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(headers['X-RateLimit-Limit']).toBe('50');
    });

    it('should set rate limit headers on response', async () => {
      mockReq.ip = 'unique-ip-2';

      await rateLimiter(mockReq as Request, mockRes as Response, mockNext);

      expect(headers['X-RateLimit-Limit']).toBeDefined();
      expect(headers['X-RateLimit-Remaining']).toBeDefined();
      expect(headers['X-RateLimit-Reset']).toBeDefined();
    });
  });

  describe('authLoginRateLimiter', () => {
    it('should have stricter limits', async () => {
      mockReq.ip = 'unique-ip-3';

      await authLoginRateLimiter(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(headers['X-RateLimit-Limit']).toBe('5');
    });

    it('should block after limit exceeded', async () => {
      mockReq.ip = 'rate-limit-test-ip';

      // Exhaust the limit
      for (let i = 0; i < 5; i++) {
        mockNext = jest.fn();
        await authLoginRateLimiter(mockReq as Request, mockRes as Response, mockNext);
      }

      // Next request should be blocked
      mockNext = jest.fn();
      await authLoginRateLimiter(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).not.toHaveBeenCalled();
      expect(mockRes.status).toHaveBeenCalledWith(429);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({
            code: 'RATE_LIMIT_EXCEEDED',
          }),
        })
      );
    });
  });
});

describe('Fallback Rate Limiter', () => {
  let mockReq: Partial<Request>;
  let mockRes: Partial<Response>;
  let mockNext: NextFunction;
  let headers: Record<string, string>;

  beforeEach(() => {
    headers = {};
    mockReq = {
      ip: `fallback-test-${Date.now()}-${Math.random()}`,
      path: '/api/test',
    };
    mockRes = {
      set: jest.fn((key: string, value: string) => {
        headers[key] = value;
        return mockRes as Response;
      }),
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };
    mockNext = jest.fn();
  });

  it('should allow first request', () => {
    fallbackRateLimiter(mockReq as Request, mockRes as Response, mockNext);

    expect(mockNext).toHaveBeenCalled();
  });

  it('should set rate limit headers', () => {
    fallbackRateLimiter(mockReq as Request, mockRes as Response, mockNext);

    expect(headers['X-RateLimit-Limit']).toBeDefined();
    expect(headers['X-RateLimit-Remaining']).toBeDefined();
  });

  it('should track request counts per IP', () => {
    const ip = `count-test-${Date.now()}`;
    mockReq.ip = ip;

    fallbackRateLimiter(mockReq as Request, mockRes as Response, mockNext);
    const firstRemaining = parseInt(headers['X-RateLimit-Remaining']);

    mockNext = jest.fn();
    fallbackRateLimiter(mockReq as Request, mockRes as Response, mockNext);
    const secondRemaining = parseInt(headers['X-RateLimit-Remaining']);

    expect(secondRemaining).toBe(firstRemaining - 1);
  });
});

describe('Test Utilities', () => {
  describe('__testResetRateLimiters', () => {
    beforeEach(() => {
      initializeMemoryRateLimiters();
    });

    it('should reset rate limiters in memory mode', async () => {
      const key = 'reset-test-key';

      // Consume some quota
      await consumeRateLimit('api', key);
      await consumeRateLimit('api', key);
      const beforeReset = await consumeRateLimit('api', key);

      // Reset
      __testResetRateLimiters();

      // After reset, should have full quota again
      const afterReset = await consumeRateLimit('api', key);

      expect(afterReset.remaining).toBeGreaterThan(beforeReset.remaining!);
    });
  });
});
