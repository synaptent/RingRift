import { Request, Response, NextFunction } from 'express';
import { RateLimiterRedis } from 'rate-limiter-flexible';
import { createError } from './errorHandler';
import { logger } from '../utils/logger';

// Redis client will be imported from cache/redis
let redisClient: any = null;

// Rate limiter configurations
const rateLimiterConfigs = {
  // General API rate limiting
  api: {
    storeClient: null, // Will be set when Redis is available
    keyPrefix: 'api_limit',
    points: 100, // Number of requests
    duration: 900, // Per 15 minutes (900 seconds)
    blockDuration: 900, // Block for 15 minutes if limit exceeded
  },
  
  // Authentication endpoints (more restrictive)
  auth: {
    storeClient: null,
    keyPrefix: 'auth_limit',
    points: 5, // Number of requests
    duration: 900, // Per 15 minutes
    blockDuration: 1800, // Block for 30 minutes
  },
  
  // Game actions (moderate limiting)
  game: {
    storeClient: null,
    keyPrefix: 'game_limit',
    points: 200, // Number of requests
    duration: 60, // Per minute
    blockDuration: 300, // Block for 5 minutes
  },
  
  // WebSocket connections
  websocket: {
    storeClient: null,
    keyPrefix: 'ws_limit',
    points: 10, // Number of connections
    duration: 60, // Per minute
    blockDuration: 300, // Block for 5 minutes
  }
};

let rateLimiters: { [key: string]: RateLimiterRedis } = {};

// Initialize rate limiters
export const initializeRateLimiters = (redis: any) => {
  redisClient = redis;
  
  Object.entries(rateLimiterConfigs).forEach(([key, config]) => {
    rateLimiters[key] = new RateLimiterRedis({
      ...config,
      storeClient: redisClient
    });
  });
  
  logger.info('Rate limiters initialized');
};

// Generic rate limiter middleware factory
const createRateLimiter = (limiterKey: string) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    const limiter = rateLimiters[limiterKey];
    
    if (!limiter) {
      // If rate limiter is not available, allow the request but log warning
      logger.warn(`Rate limiter '${limiterKey}' not available, allowing request`);
      return next();
    }

    try {
      const key = req.ip || 'unknown';
      await limiter.consume(key);
      next();
    } catch (rejRes: any) {
      const secs = Math.round(rejRes.msBeforeNext / 1000) || 1;
      
      logger.warn('Rate limit exceeded', {
        ip: req.ip,
        limiter: limiterKey,
        path: req.path,
        retryAfter: secs
      });

      res.set('Retry-After', String(secs));
      
      const error = createError(
        'Too many requests, please try again later',
        429,
        'RATE_LIMIT_EXCEEDED'
      );
      
      res.status(429).json({
        success: false,
        error: {
          message: error.message,
          code: error.code,
          retryAfter: secs,
          timestamp: new Date().toISOString()
        }
      });
    }
  };
};

// Specific rate limiter middlewares
export const rateLimiter = createRateLimiter('api');
export const authRateLimiter = createRateLimiter('auth');
export const gameRateLimiter = createRateLimiter('game');
export const websocketRateLimiter = createRateLimiter('websocket');

// Custom rate limiter for specific endpoints
export const customRateLimiter = (points: number, duration: number, blockDuration?: number) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    if (!redisClient) {
      logger.warn('Redis client not available for custom rate limiter, allowing request');
      return next();
    }

    const limiter = new RateLimiterRedis({
      storeClient: redisClient,
      keyPrefix: 'custom_limit',
      points,
      duration,
      blockDuration: blockDuration || duration
    });

    try {
      const key = req.ip || 'unknown';
      await limiter.consume(key);
      next();
    } catch (rejRes: any) {
      const secs = Math.round(rejRes.msBeforeNext / 1000) || 1;
      
      logger.warn('Custom rate limit exceeded', {
        ip: req.ip,
        path: req.path,
        points,
        duration,
        retryAfter: secs
      });

      res.set('Retry-After', String(secs));
      
      res.status(429).json({
        success: false,
        error: {
          message: 'Too many requests, please try again later',
          code: 'RATE_LIMIT_EXCEEDED',
          retryAfter: secs,
          timestamp: new Date().toISOString()
        }
      });
    }
  };
};

// Rate limiter for user-specific actions (using user ID instead of IP)
export const userRateLimiter = (limiterKey: string) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    const limiter = rateLimiters[limiterKey];
    
    if (!limiter) {
      logger.warn(`Rate limiter '${limiterKey}' not available, allowing request`);
      return next();
    }

    try {
      // Use user ID if authenticated, otherwise fall back to IP
      const key = (req as any).user?.id || req.ip || 'unknown';
      await limiter.consume(key);
      next();
    } catch (rejRes: any) {
      const secs = Math.round(rejRes.msBeforeNext / 1000) || 1;
      
      logger.warn('User rate limit exceeded', {
        userId: (req as any).user?.id,
        ip: req.ip,
        limiter: limiterKey,
        path: req.path,
        retryAfter: secs
      });

      res.set('Retry-After', String(secs));
      
      res.status(429).json({
        success: false,
        error: {
          message: 'Too many requests, please try again later',
          code: 'RATE_LIMIT_EXCEEDED',
          retryAfter: secs,
          timestamp: new Date().toISOString()
        }
      });
    }
  };
};

// Fallback rate limiter when Redis is not available
export const fallbackRateLimiter = (() => {
  const requests = new Map<string, { count: number; resetTime: number }>();
  const WINDOW_SIZE = 15 * 60 * 1000; // 15 minutes
  const MAX_REQUESTS = 100;

  return (req: Request, res: Response, next: NextFunction) => {
    const key = req.ip || 'unknown';
    const now = Date.now();
    const windowStart = now - WINDOW_SIZE;

    // Clean up old entries
    for (const [ip, data] of requests.entries()) {
      if (data.resetTime < windowStart) {
        requests.delete(ip);
      }
    }

    const current = requests.get(key);
    
    if (!current) {
      requests.set(key, { count: 1, resetTime: now });
      return next();
    }

    if (current.resetTime < windowStart) {
      requests.set(key, { count: 1, resetTime: now });
      return next();
    }

    if (current.count >= MAX_REQUESTS) {
      logger.warn('Fallback rate limit exceeded', {
        ip: req.ip,
        path: req.path,
        count: current.count
      });

      return res.status(429).json({
        success: false,
        error: {
          message: 'Too many requests, please try again later',
          code: 'RATE_LIMIT_EXCEEDED',
          timestamp: new Date().toISOString()
        }
      });
    }

    current.count++;
    next();
  };
})();