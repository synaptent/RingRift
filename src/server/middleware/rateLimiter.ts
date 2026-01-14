import { Request, Response, NextFunction } from 'express';
import { RateLimiterRedis, RateLimiterMemory, RateLimiterRes } from 'rate-limiter-flexible';
import { createClient } from 'redis';
import { createError } from './errorHandler';
import { logger } from '../utils/logger';
import { auditRateLimitExceeded } from '../utils/auditLogger';
import { getMetricsService } from '../services/MetricsService';
import { type AuthenticatedRequest, getAuthUserId } from './auth';

/**
 * Redis client type from the redis package.
 */
type RedisClientType = ReturnType<typeof createClient>;
type RateLimitBypassLoggedRequest = Request & { __rateLimitBypassLogged?: boolean };

/**
 * Rate limiter rejection response - returned when limit is exceeded.
 * The rate-limiter-flexible library throws this object on rejection.
 */
interface RateLimiterRejection {
  msBeforeNext: number;
  remainingPoints: number;
  consumedPoints: number;
  isFirstInDuration: boolean;
}

// Redis client reference
let redisClient: RedisClientType | null = null;

/**
 * Normalize IP strings so that equivalent loopback / IPv4-mapped variants do not
 * accidentally receive independent quotas (and so tests behave deterministically).
 */
const normalizeIpKey = (ip: string | undefined | null): string => {
  const raw = (ip ?? '').trim();
  if (!raw) return 'unknown';
  const lower = raw.toLowerCase();
  if (lower === '::1') return '127.0.0.1';
  if (lower.startsWith('::ffff:')) return raw.slice('::ffff:'.length);
  return raw;
};

// =============================================================================
// RATE LIMIT BYPASS FOR LOAD TESTING
// =============================================================================
//
// ⚠️  SECURITY WARNING - READ CAREFULLY ⚠️
//
// This bypass mechanism allows load test users to skip rate limiting.
// It is intended ONLY for staging environments during load testing.
//
// CRITICAL SECURITY REQUIREMENTS:
// 1. RATE_LIMIT_BYPASS_ENABLED must default to false
// 2. NEVER enable in production - it defeats DDoS protection
// 3. Only use with controlled load test user accounts
// 4. All bypass events are logged for audit purposes
//
// The bypass is opt-in and requires BOTH:
// - RATE_LIMIT_BYPASS_ENABLED=true in environment
// - AND one of:
//   - Request with valid X-RateLimit-Bypass-Token header matching RATE_LIMIT_BYPASS_TOKEN
//   - Request from a user matching RATE_LIMIT_BYPASS_USER_PATTERN
//   - Request from an IP in RATE_LIMIT_BYPASS_IPS
//
// =============================================================================

/**
 * Check if rate limit bypass is enabled in environment.
 *
 * ⚠️  CRITICAL: This must default to false to prevent accidental exposure.
 */
const isRateLimitBypassEnabled = (): boolean => {
  return process.env.RATE_LIMIT_BYPASS_ENABLED === 'true';
};

/**
 * Get the bypass token from environment.
 * Returns null if bypass is disabled or token is not configured.
 */
const getBypassToken = (): string | null => {
  if (!isRateLimitBypassEnabled()) return null;
  const token = process.env.RATE_LIMIT_BYPASS_TOKEN;
  // Token must be at least 16 characters for security
  if (token && token.length >= 16) {
    return token;
  }
  return null;
};

/**
 * Get the compiled regex pattern for load test user emails.
 * Returns null if bypass is disabled or pattern is invalid.
 */
const getBypassUserPattern = (): RegExp | null => {
  if (!isRateLimitBypassEnabled()) return null;
  // Default pattern: ^loadtest[._].+@loadtest\.local$
  // Matches both underscore and dot separators:
  //   - loadtest.user1@loadtest.local
  //   - loadtest_user_1@loadtest.local
  //   - loadtest_vu_42@loadtest.local
  const pattern = process.env.RATE_LIMIT_BYPASS_USER_PATTERN || '^loadtest[._].+@loadtest\\.local$';
  try {
    return new RegExp(pattern);
  } catch {
    logger.warn('Invalid RATE_LIMIT_BYPASS_USER_PATTERN, bypass disabled', {
      event: 'rate_limit_bypass_pattern_invalid',
      pattern,
    });
    return null;
  }
};

/**
 * Get the set of whitelisted IPs for rate limit bypass.
 * Returns empty set if bypass is disabled.
 */
const getBypassIPs = (): Set<string> => {
  if (!isRateLimitBypassEnabled()) return new Set();
  const ips = process.env.RATE_LIMIT_BYPASS_IPS || '';
  return new Set(
    ips
      .split(',')
      .map((ip) => ip.trim())
      .filter(Boolean)
  );
};

/**
 * Log when rate limit bypass is triggered.
 * This provides an audit trail for security review.
 */
const logBypassTriggered = (
  req: Request,
  reason: 'ip' | 'user_pattern' | 'bypass_token',
  identifier: string
): void => {
  const authReq = req as AuthenticatedRequest;
  logger.info('Rate limit bypass triggered', {
    event: 'rate_limit_bypass_triggered',
    reason,
    identifier,
    path: req.path,
    method: req.method,
    ip: req.ip,
    userId: authReq.user?.id,
    userEmail: authReq.user?.email,
  });
};

const shouldLogBypass = (req: Request, logBypass: boolean): boolean => {
  if (!logBypass) return false;
  const trackedReq = req as RateLimitBypassLoggedRequest;
  if (trackedReq.__rateLimitBypassLogged) {
    return false;
  }
  trackedReq.__rateLimitBypassLogged = true;
  return true;
};

/**
 * Check if a request should bypass rate limiting.
 *
 * Returns true when bypass is enabled AND one of:
 * - Request has valid X-RateLimit-Bypass-Token header matching RATE_LIMIT_BYPASS_TOKEN
 * - Request from a user matching RATE_LIMIT_BYPASS_USER_PATTERN
 * - Request from an IP in RATE_LIMIT_BYPASS_IPS
 *
 * All bypass events are logged for audit purposes.
 *
 * @param req - Express request object
 * @param logBypass - Whether to log bypass events (default: true)
 * @returns true if the request should bypass rate limiting
 */
export const shouldBypassRateLimit = (req: Request, logBypass: boolean = true): boolean => {
  if (!isRateLimitBypassEnabled()) return false;

  // Check bypass token header first (most reliable for load tests)
  const bypassToken = getBypassToken();
  if (bypassToken) {
    const headerToken = req.headers['x-ratelimit-bypass-token'] as string | undefined;
    if (headerToken && headerToken === bypassToken) {
      if (shouldLogBypass(req, logBypass)) {
        logBypassTriggered(req, 'bypass_token', '(token)');
      }
      return true;
    }
  }

  // Check IP whitelist
  const normalizedIp = normalizeIpKey(req.ip);
  const bypassIPs = getBypassIPs();
  if (bypassIPs.has(normalizedIp)) {
    if (shouldLogBypass(req, logBypass)) {
      logBypassTriggered(req, 'ip', normalizedIp);
    }
    return true;
  }

  // Check user email pattern
  const authReq = req as AuthenticatedRequest;
  const userEmail = authReq.user?.email;
  if (userEmail) {
    const pattern = getBypassUserPattern();
    if (pattern && pattern.test(userEmail)) {
      if (shouldLogBypass(req, logBypass)) {
        logBypassTriggered(req, 'user_pattern', userEmail);
      }
      return true;
    }
  }

  return false;
};

/**
 * Rate limit configuration type for type safety
 */
export interface RateLimitConfig {
  keyPrefix: string;
  points: number; // Number of requests allowed
  duration: number; // Window duration in seconds
  blockDuration: number; // Block duration when exceeded in seconds
}

/**
 * Environment-driven rate limit configurations.
 * All values can be overridden via environment variables.
 *
 * Environment variables follow the pattern:
 * RATE_LIMIT_{TYPE}_{SETTING}
 * e.g., RATE_LIMIT_AUTH_POINTS, RATE_LIMIT_API_DURATION
 */
const getEnvNumber = (key: string, defaultValue: number): number => {
  const value = process.env[key];
  if (value === undefined) return defaultValue;
  const parsed = parseInt(value, 10);
  return isNaN(parsed) ? defaultValue : parsed;
};

export const getRateLimitConfigs = (): Record<string, RateLimitConfig> => ({
  // General API rate limiting - standard operations (anonymous)
  api: {
    keyPrefix: 'api_limit',
    points: getEnvNumber('RATE_LIMIT_API_POINTS', 50), // requests for anonymous
    duration: getEnvNumber('RATE_LIMIT_API_DURATION', 60), // per minute
    blockDuration: getEnvNumber('RATE_LIMIT_API_BLOCK_DURATION', 300), // 5 min block
  },

  // Higher limits for authenticated API users
  apiAuthenticated: {
    keyPrefix: 'api_auth_limit',
    points: getEnvNumber('RATE_LIMIT_API_AUTH_POINTS', 200), // requests
    duration: getEnvNumber('RATE_LIMIT_API_AUTH_DURATION', 60), // per minute
    blockDuration: getEnvNumber('RATE_LIMIT_API_AUTH_BLOCK_DURATION', 300), // 5 min block
  },

  // Authentication endpoints (most restrictive for security)
  auth: {
    keyPrefix: 'auth_limit',
    points: getEnvNumber('RATE_LIMIT_AUTH_POINTS', 10), // requests
    duration: getEnvNumber('RATE_LIMIT_AUTH_DURATION', 900), // per 15 minutes
    blockDuration: getEnvNumber('RATE_LIMIT_AUTH_BLOCK_DURATION', 1800), // 30 min block
  },

  // Login specifically - stricter
  authLogin: {
    keyPrefix: 'auth_login_limit',
    points: getEnvNumber('RATE_LIMIT_AUTH_LOGIN_POINTS', 5), // attempts
    duration: getEnvNumber('RATE_LIMIT_AUTH_LOGIN_DURATION', 900), // per 15 minutes
    blockDuration: getEnvNumber('RATE_LIMIT_AUTH_LOGIN_BLOCK_DURATION', 1800), // 30 min block
  },

  // Registration - prevent spam account creation
  authRegister: {
    keyPrefix: 'auth_register_limit',
    points: getEnvNumber('RATE_LIMIT_AUTH_REGISTER_POINTS', 3), // registrations
    duration: getEnvNumber('RATE_LIMIT_AUTH_REGISTER_DURATION', 3600), // per hour
    blockDuration: getEnvNumber('RATE_LIMIT_AUTH_REGISTER_BLOCK_DURATION', 3600), // 1 hour block
  },

  // Password reset - prevent abuse
  authPasswordReset: {
    keyPrefix: 'auth_pwd_reset_limit',
    points: getEnvNumber('RATE_LIMIT_AUTH_PWD_RESET_POINTS', 3), // attempts
    duration: getEnvNumber('RATE_LIMIT_AUTH_PWD_RESET_DURATION', 3600), // per hour
    blockDuration: getEnvNumber('RATE_LIMIT_AUTH_PWD_RESET_BLOCK_DURATION', 3600), // 1 hour block
  },

  // Game actions (moderate limiting)
  game: {
    keyPrefix: 'game_limit',
    points: getEnvNumber('RATE_LIMIT_GAME_POINTS', 200), // requests
    duration: getEnvNumber('RATE_LIMIT_GAME_DURATION', 60), // per minute
    blockDuration: getEnvNumber('RATE_LIMIT_GAME_BLOCK_DURATION', 300), // 5 min block
  },

  // Game moves - need to be higher for active gameplay
  gameMoves: {
    keyPrefix: 'game_moves_limit',
    points: getEnvNumber('RATE_LIMIT_GAME_MOVES_POINTS', 100), // moves
    duration: getEnvNumber('RATE_LIMIT_GAME_MOVES_DURATION', 60), // per minute
    blockDuration: getEnvNumber('RATE_LIMIT_GAME_MOVES_BLOCK_DURATION', 60), // 1 min block
  },

  // WebSocket connections
  websocket: {
    keyPrefix: 'ws_limit',
    points: getEnvNumber('RATE_LIMIT_WS_POINTS', 10), // connections
    duration: getEnvNumber('RATE_LIMIT_WS_DURATION', 60), // per minute
    blockDuration: getEnvNumber('RATE_LIMIT_WS_BLOCK_DURATION', 300), // 5 min block
  },

  // Game creation quotas (per-user)
  gameCreateUser: {
    keyPrefix: 'game_create_user',
    points: getEnvNumber('RATE_LIMIT_GAME_CREATE_USER_POINTS', 20), // games
    duration: getEnvNumber('RATE_LIMIT_GAME_CREATE_USER_DURATION', 600), // per 10 minutes
    blockDuration: getEnvNumber('RATE_LIMIT_GAME_CREATE_USER_BLOCK_DURATION', 600),
  },

  // Game creation quotas (per-IP for unauthenticated requests)
  gameCreateIp: {
    keyPrefix: 'game_create_ip',
    points: getEnvNumber('RATE_LIMIT_GAME_CREATE_IP_POINTS', 50), // requests
    duration: getEnvNumber('RATE_LIMIT_GAME_CREATE_IP_DURATION', 600), // per 10 minutes
    blockDuration: getEnvNumber('RATE_LIMIT_GAME_CREATE_IP_BLOCK_DURATION', 600),
  },

  // Data export (GDPR) - very restrictive per-user limit
  dataExport: {
    keyPrefix: 'data_export',
    points: getEnvNumber('RATE_LIMIT_DATA_EXPORT_POINTS', 1), // 1 request
    duration: getEnvNumber('RATE_LIMIT_DATA_EXPORT_DURATION', 3600), // per hour
    blockDuration: getEnvNumber('RATE_LIMIT_DATA_EXPORT_BLOCK_DURATION', 3600), // 1 hour block
  },

  // Telemetry events - moderate limit to prevent abuse while allowing normal gameplay
  telemetry: {
    keyPrefix: 'telemetry_limit',
    points: getEnvNumber('RATE_LIMIT_TELEMETRY_POINTS', 100), // events
    duration: getEnvNumber('RATE_LIMIT_TELEMETRY_DURATION', 60), // per minute
    blockDuration: getEnvNumber('RATE_LIMIT_TELEMETRY_BLOCK_DURATION', 300), // 5 min block
  },

  // Client error reporting - limited to prevent log flooding
  clientErrors: {
    keyPrefix: 'client_errors_limit',
    points: getEnvNumber('RATE_LIMIT_CLIENT_ERRORS_POINTS', 20), // errors
    duration: getEnvNumber('RATE_LIMIT_CLIENT_ERRORS_DURATION', 60), // per minute
    blockDuration: getEnvNumber('RATE_LIMIT_CLIENT_ERRORS_BLOCK_DURATION', 300), // 5 min block
  },

  // Internal health check endpoints - lenient but prevents probe flooding
  internalHealth: {
    keyPrefix: 'internal_health_limit',
    points: getEnvNumber('RATE_LIMIT_INTERNAL_HEALTH_POINTS', 30), // requests
    duration: getEnvNumber('RATE_LIMIT_INTERNAL_HEALTH_DURATION', 60), // per minute
    blockDuration: getEnvNumber('RATE_LIMIT_INTERNAL_HEALTH_BLOCK_DURATION', 60), // 1 min block
  },

  // Alert webhook - conservative to prevent log flooding attacks
  alertWebhook: {
    keyPrefix: 'alert_webhook_limit',
    points: getEnvNumber('RATE_LIMIT_ALERT_WEBHOOK_POINTS', 10), // requests
    duration: getEnvNumber('RATE_LIMIT_ALERT_WEBHOOK_DURATION', 60), // per minute
    blockDuration: getEnvNumber('RATE_LIMIT_ALERT_WEBHOOK_BLOCK_DURATION', 300), // 5 min block
  },

  // User rating lookup - prevent enumeration attacks
  userRating: {
    keyPrefix: 'user_rating_limit',
    points: getEnvNumber('RATE_LIMIT_USER_RATING_POINTS', 30), // requests
    duration: getEnvNumber('RATE_LIMIT_USER_RATING_DURATION', 60), // per minute
    blockDuration: getEnvNumber('RATE_LIMIT_USER_RATING_BLOCK_DURATION', 120), // 2 min block
  },

  // User search - prevent enumeration and database load
  userSearch: {
    keyPrefix: 'user_search_limit',
    points: getEnvNumber('RATE_LIMIT_USER_SEARCH_POINTS', 20), // requests
    duration: getEnvNumber('RATE_LIMIT_USER_SEARCH_DURATION', 60), // per minute
    blockDuration: getEnvNumber('RATE_LIMIT_USER_SEARCH_BLOCK_DURATION', 120), // 2 min block
  },

  // Sandbox AI endpoints - high limits for local AI games
  // Each AI move in sandbox mode requires a request to the AI service
  sandboxAi: {
    keyPrefix: 'sandbox_ai_limit',
    points: getEnvNumber('RATE_LIMIT_SANDBOX_AI_POINTS', 1000), // requests
    duration: getEnvNumber('RATE_LIMIT_SANDBOX_AI_DURATION', 60), // per minute
    blockDuration: getEnvNumber('RATE_LIMIT_SANDBOX_AI_BLOCK_DURATION', 60), // 1 min block
  },
});

// Cache the configs to avoid re-parsing env vars on every request
let cachedConfigs: Record<string, RateLimitConfig> | null = null;

const getRateLimitConfigsCached = (): Record<string, RateLimitConfig> => {
  if (!cachedConfigs) {
    cachedConfigs = getRateLimitConfigs();
  }
  return cachedConfigs;
};

// Rate limiters - can be either Redis or Memory based
const rateLimiters: { [key: string]: RateLimiterRedis | RateLimiterMemory } = {};

// Flag to track whether we're using Redis or in-memory
let usingRedis = false;

/**
 * Initialize rate limiters with Redis client.
 * Falls back to in-memory limiters if Redis is not available.
 */
export const initializeRateLimiters = (redis: RedisClientType | null) => {
  redisClient = redis;
  usingRedis = !!redis;

  const configs = getRateLimitConfigsCached();

  Object.entries(configs).forEach(([key, config]) => {
    if (redis) {
      rateLimiters[key] = new RateLimiterRedis({
        storeClient: redis,
        useRedisPackage: true, // Required for node-redis v5 compatibility
        keyPrefix: config.keyPrefix,
        points: config.points,
        duration: config.duration,
        blockDuration: config.blockDuration,
      });
    } else {
      // Fallback to in-memory limiter for development/testing
      rateLimiters[key] = new RateLimiterMemory({
        keyPrefix: config.keyPrefix,
        points: config.points,
        duration: config.duration,
        blockDuration: config.blockDuration,
      });
    }
  });

  logger.info('Rate limiters initialized', {
    mode: usingRedis ? 'redis' : 'memory',
    limiterCount: Object.keys(rateLimiters).length,
    configs: Object.fromEntries(Object.entries(configs).map(([k, v]) => [k, v.points])),
  });

  // Production safety: warn if rate limit bypass is enabled
  if (isRateLimitBypassEnabled()) {
    const isProduction = process.env.NODE_ENV === 'production';
    const level = isProduction ? 'error' : 'warn';
    const bypassTokenConfigured = !!(
      process.env.RATE_LIMIT_BYPASS_TOKEN && process.env.RATE_LIMIT_BYPASS_TOKEN.length >= 16
    );
    logger[level](
      `SECURITY: Rate limit bypass is ENABLED. ${isProduction ? 'This is dangerous in production!' : 'Acceptable for testing only.'}`,
      {
        event: 'rate_limit_bypass_enabled',
        bypassTokenConfigured,
        bypassPattern:
          process.env.RATE_LIMIT_BYPASS_USER_PATTERN || '^loadtest[._].+@loadtest\\.local$',
        bypassIPs: process.env.RATE_LIMIT_BYPASS_IPS || '(none)',
        isProduction,
      }
    );
  }
};

/**
 * Initialize in-memory rate limiters for development without Redis.
 * This allows the server to start even when Redis is not available.
 */
export const initializeMemoryRateLimiters = () => {
  const configs = getRateLimitConfigsCached();

  Object.entries(configs).forEach(([key, config]) => {
    rateLimiters[key] = new RateLimiterMemory({
      keyPrefix: config.keyPrefix,
      points: config.points,
      duration: config.duration,
      blockDuration: config.blockDuration,
    });
  });

  usingRedis = false;
  logger.info('In-memory rate limiters initialized (Redis not available)', {
    limiterCount: Object.keys(rateLimiters).length,
  });
};

/**
 * Check if rate limiters are using Redis backing store.
 */
export const isUsingRedisRateLimiting = (): boolean => usingRedis;

export interface RateLimitResult {
  allowed: boolean;
  retryAfter?: number;
  limit?: number;
  remaining?: number;
  reset?: number; // Unix timestamp when limit resets
}

/**
 * Set standard rate limit headers on a response.
 * These headers inform clients of their quota status.
 *
 * Headers:
 * - X-RateLimit-Limit: Maximum requests in the window
 * - X-RateLimit-Remaining: Remaining requests in the window
 * - X-RateLimit-Reset: Unix timestamp when the window resets
 * - Retry-After: Seconds to wait before retrying (only on rate limit exceeded)
 */
export const setRateLimitHeaders = (
  res: Response,
  limit: number,
  remaining: number,
  resetTimestamp: number
): void => {
  res.set('X-RateLimit-Limit', String(limit));
  res.set('X-RateLimit-Remaining', String(Math.max(0, remaining)));
  res.set('X-RateLimit-Reset', String(resetTimestamp));
};

/**
 * Low-level helper used by HTTP routes and other callers that need to perform
 * ad-hoc rate limiting (for example, game creation quotas).
 *
 * This wrapper centralises graceful degradation semantics:
 * - When the limiter is not initialised, the request is allowed and a warning
 *   is logged.
 * - When Redis or the underlying store fails, the request is allowed and a
 *   warning is logged so that dependency outages do not cascade.
 * - When the configured quota is exceeded, the caller receives
 *   { allowed: false, retryAfter } without the helper writing any response.
 */
export const consumeRateLimit = async (
  limiterKey: string,
  key: string,
  req?: Request
): Promise<RateLimitResult> => {
  if (req && shouldBypassRateLimit(req)) {
    return { allowed: true };
  }

  const limiter = rateLimiters[limiterKey];
  const config = getRateLimitConfigsCached()[limiterKey];

  if (!limiter) {
    logger.warn(`Rate limiter '${limiterKey}' not available, allowing request`, {
      key,
    });
    return { allowed: true };
  }

  try {
    const rateLimiterRes = await limiter.consume(key);
    const resetTimestamp = Math.ceil(Date.now() / 1000 + rateLimiterRes.msBeforeNext / 1000);

    return {
      allowed: true,
      limit: config?.points,
      remaining: rateLimiterRes.remainingPoints,
      reset: resetTimestamp,
    };
  } catch (error: unknown) {
    // rate-limiter-flexible returns a special object with msBeforeNext when
    // the limit is exceeded. Treat all other errors as infrastructure
    // failures and allow the request to proceed to avoid cascading
    // outages when Redis is unavailable.
    const rejection = error as RateLimiterRejection | null;
    const msBeforeNext =
      rejection && typeof rejection === 'object' ? rejection.msBeforeNext : undefined;

    if (typeof msBeforeNext === 'number') {
      const secs = Math.round(msBeforeNext / 1000) || 1;
      return {
        allowed: false,
        retryAfter: secs,
        limit: config?.points,
        remaining: 0,
        reset: Math.ceil(Date.now() / 1000 + secs),
      };
    }

    logger.warn(`Rate limiter '${limiterKey}' error, allowing request`, {
      key,
      error: error instanceof Error ? error.message : String(error),
    });
    return { allowed: true };
  }
};

/**
 * Calculate reset timestamp from RateLimiterRes
 */
const getResetTimestamp = (rateLimiterRes: RateLimiterRes): number => {
  return Math.ceil(Date.now() / 1000 + rateLimiterRes.msBeforeNext / 1000);
};

/**
 * Generic rate limiter middleware factory.
 * Creates middleware that enforces rate limits and sets appropriate headers.
 *
 * @param limiterKey - Key to identify which rate limiter config to use
 * @param options - Additional options for customization
 */
const createRateLimiter = (
  limiterKey: string,
  options: {
    keyGenerator?: (req: Request) => string;
    skipSuccessfulRequests?: boolean;
  } = {}
) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    // Check for load test bypass
    if (shouldBypassRateLimit(req)) {
      return next();
    }

    const limiter = rateLimiters[limiterKey];
    const config = getRateLimitConfigsCached()[limiterKey];

    if (!limiter || !config) {
      // If rate limiter is not available, allow the request but log warning
      logger.warn(`Rate limiter '${limiterKey}' not available, allowing request`);
      return next();
    }

    try {
      // Generate key - default to IP, but can be customized
      const key = options.keyGenerator ? options.keyGenerator(req) : normalizeIpKey(req.ip);
      const rateLimiterRes = await limiter.consume(key);

      // Set rate limit headers on successful consumption
      const resetTimestamp = getResetTimestamp(rateLimiterRes);
      setRateLimitHeaders(res, config.points, rateLimiterRes.remainingPoints, resetTimestamp);

      next();
    } catch (rateLimitError: unknown) {
      // Rate limit exceeded - rateLimitError is a RateLimiterRejection object
      const rejRes = rateLimitError as RateLimiterRejection;
      const secs = Math.round(rejRes.msBeforeNext / 1000) || 1;
      const resetTimestamp = Math.ceil(Date.now() / 1000 + secs);

      // Set headers even on rate limit exceeded
      setRateLimitHeaders(res, config.points, 0, resetTimestamp);
      res.set('Retry-After', String(secs));

      const authReq = req as AuthenticatedRequest;
      logger.warn('Rate limit exceeded', {
        ip: req.ip,
        userId: authReq.user?.id,
        limiter: limiterKey,
        path: req.path,
        retryAfter: secs,
        error: rateLimitError instanceof Error ? rateLimitError.message : String(rateLimitError),
        msBeforeNext: rejRes.msBeforeNext,
      });

      // Audit log the rate limit exceeded event
      auditRateLimitExceeded(limiterKey, req.ip || 'unknown', req);

      // Record rate limit hit metric
      getMetricsService().recordRateLimitHit(req.path, limiterKey);

      const apiError = createError(
        'Too many requests, please try again later',
        429,
        'RATE_LIMIT_EXCEEDED'
      );

      res.status(429).json({
        success: false,
        error: {
          message: apiError.message,
          code: apiError.code,
          retryAfter: secs,
          timestamp: new Date().toISOString(),
        },
      });
    }
  };
};

// Specific rate limiter middlewares for different endpoint types
export const rateLimiter = createRateLimiter('api');
export const authRateLimiter = createRateLimiter('auth');
export const authLoginRateLimiter = createRateLimiter('authLogin');
export const authRegisterRateLimiter = createRateLimiter('authRegister');
export const authPasswordResetRateLimiter = createRateLimiter('authPasswordReset');
export const gameRateLimiter = createRateLimiter('game');
export const gameMovesRateLimiter = createRateLimiter('gameMoves');
export const websocketRateLimiter = createRateLimiter('websocket');

/**
 * Rate limiter for data export endpoints (GDPR/privacy).
 * Uses user ID as the key to limit exports per user, not per IP.
 * Default: 1 request per hour per user.
 */
export const dataExportRateLimiter = createRateLimiter('dataExport', {
  keyGenerator: (req: Request) => {
    const authReq = req as AuthenticatedRequest;
    return authReq.user?.id || req.ip || 'unknown';
  },
});
export const telemetryRateLimiter = createRateLimiter('telemetry');
export const clientErrorsRateLimiter = createRateLimiter('clientErrors', {
  keyGenerator: (req: Request) => {
    // Use IP for anonymous error reporting
    return req.ip || 'unknown';
  },
});

// Internal route rate limiters
export const internalHealthRateLimiter = createRateLimiter('internalHealth');
export const alertWebhookRateLimiter = createRateLimiter('alertWebhook');

// User data rate limiters
export const userRatingRateLimiter = createRateLimiter('userRating');
export const userSearchRateLimiter = createRateLimiter('userSearch');

// Sandbox AI rate limiter - high limits for local AI games
export const sandboxAiRateLimiter = createRateLimiter('sandboxAi', {
  keyGenerator: (req: Request) => {
    // Use user ID if authenticated, otherwise IP
    const authReq = req as AuthenticatedRequest;
    return authReq.user?.id || req.ip || 'unknown';
  },
});

/**
 * Rate limiter that differentiates between authenticated and anonymous users.
 * Authenticated users get higher limits.
 *
 * @param authenticatedKey - Limiter key for authenticated users
 * @param anonymousKey - Limiter key for anonymous users (defaults to standard 'api')
 */
export const adaptiveRateLimiter = (
  authenticatedKey: string = 'apiAuthenticated',
  anonymousKey: string = 'api'
) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    // Check for load test bypass
    if (shouldBypassRateLimit(req)) {
      return next();
    }

    const authReq = req as AuthenticatedRequest;
    // Choose limiter based on authentication status
    const isAuthenticated = !!authReq.user?.id;
    const limiterKey = isAuthenticated ? authenticatedKey : anonymousKey;
    const limiter = rateLimiters[limiterKey];
    const config = getRateLimitConfigsCached()[limiterKey];

    if (!limiter || !config) {
      logger.warn(`Rate limiter '${limiterKey}' not available, allowing request`);
      return next();
    }

    try {
      // Use user ID for authenticated, IP for anonymous
      const key = isAuthenticated ? getAuthUserId(authReq) : normalizeIpKey(req.ip);
      const rateLimiterRes = await limiter.consume(key);

      // Set rate limit headers
      const resetTimestamp = getResetTimestamp(rateLimiterRes);
      setRateLimitHeaders(res, config.points, rateLimiterRes.remainingPoints, resetTimestamp);

      next();
    } catch (error: unknown) {
      const rejRes = error as RateLimiterRejection;
      const secs = Math.round(rejRes.msBeforeNext / 1000) || 1;
      const resetTimestamp = Math.ceil(Date.now() / 1000 + secs);

      setRateLimitHeaders(res, config.points, 0, resetTimestamp);
      res.set('Retry-After', String(secs));

      logger.warn('Adaptive rate limit exceeded', {
        ip: req.ip,
        userId: authReq.user?.id,
        limiter: limiterKey,
        isAuthenticated,
        path: req.path,
        retryAfter: secs,
        error: error instanceof Error ? error.message : String(error),
        msBeforeNext: rejRes.msBeforeNext,
      });

      // Record rate limit hit metric
      getMetricsService().recordRateLimitHit(req.path, limiterKey);

      res.status(429).json({
        success: false,
        error: {
          message: 'Too many requests, please try again later',
          code: 'RATE_LIMIT_EXCEEDED',
          retryAfter: secs,
          timestamp: new Date().toISOString(),
        },
      });
    }
  };
};

// Custom rate limiter for specific endpoints with runtime configuration
export const customRateLimiter = (points: number, duration: number, blockDuration?: number) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    // Check for load test bypass
    if (shouldBypassRateLimit(req)) {
      return next();
    }

    // Create limiter with appropriate storage
    const limiter = redisClient
      ? new RateLimiterRedis({
          storeClient: redisClient,
          useRedisPackage: true, // Required for node-redis v5 compatibility
          keyPrefix: 'custom_limit',
          points,
          duration,
          blockDuration: blockDuration || duration,
        })
      : new RateLimiterMemory({
          keyPrefix: 'custom_limit',
          points,
          duration,
          blockDuration: blockDuration || duration,
        });

    try {
      const key = req.ip || 'unknown';
      const rateLimiterRes = await limiter.consume(key);

      // Set rate limit headers
      const resetTimestamp = getResetTimestamp(rateLimiterRes);
      setRateLimitHeaders(res, points, rateLimiterRes.remainingPoints, resetTimestamp);

      next();
    } catch (error: unknown) {
      const rejRes = error as RateLimiterRejection;
      const secs = Math.round(rejRes.msBeforeNext / 1000) || 1;
      const resetTimestamp = Math.ceil(Date.now() / 1000 + secs);

      setRateLimitHeaders(res, points, 0, resetTimestamp);
      res.set('Retry-After', String(secs));

      logger.warn('Custom rate limit exceeded', {
        ip: req.ip,
        path: req.path,
        points,
        duration,
        retryAfter: secs,
      });

      // Record rate limit hit metric
      getMetricsService().recordRateLimitHit(req.path, 'custom');

      res.status(429).json({
        success: false,
        error: {
          message: 'Too many requests, please try again later',
          code: 'RATE_LIMIT_EXCEEDED',
          retryAfter: secs,
          timestamp: new Date().toISOString(),
        },
      });
    }
  };
};

/**
 * Rate limiter for user-specific actions (using user ID instead of IP).
 * Useful for per-user quotas like game creation.
 */
export const userRateLimiter = (limiterKey: string) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    // Check for load test bypass
    if (shouldBypassRateLimit(req)) {
      return next();
    }

    const limiter = rateLimiters[limiterKey];
    const config = getRateLimitConfigsCached()[limiterKey];

    if (!limiter || !config) {
      logger.warn(`Rate limiter '${limiterKey}' not available, allowing request`);
      return next();
    }

    const authReq = req as AuthenticatedRequest;
    try {
      // Use user ID if authenticated, otherwise fall back to IP
      const key = authReq.user?.id || req.ip || 'unknown';
      const rateLimiterRes = await limiter.consume(key);

      // Set rate limit headers
      const resetTimestamp = getResetTimestamp(rateLimiterRes);
      setRateLimitHeaders(res, config.points, rateLimiterRes.remainingPoints, resetTimestamp);

      next();
    } catch (error: unknown) {
      const rejRes = error as RateLimiterRejection;
      const secs = Math.round(rejRes.msBeforeNext / 1000) || 1;
      const resetTimestamp = Math.ceil(Date.now() / 1000 + secs);

      setRateLimitHeaders(res, config.points, 0, resetTimestamp);
      res.set('Retry-After', String(secs));

      logger.warn('User rate limit exceeded', {
        userId: authReq.user?.id,
        ip: req.ip,
        limiter: limiterKey,
        path: req.path,
        retryAfter: secs,
      });

      // Record rate limit hit metric
      getMetricsService().recordRateLimitHit(req.path, limiterKey);

      res.status(429).json({
        success: false,
        error: {
          message: 'Too many requests, please try again later',
          code: 'RATE_LIMIT_EXCEEDED',
          retryAfter: secs,
          timestamp: new Date().toISOString(),
        },
      });
    }
  };
};

/**
 * Fallback rate limiter when Redis is not available.
 * Implements a simple sliding window algorithm in memory.
 *
 * This is kept for backwards compatibility but the preferred approach
 * is to use initializeMemoryRateLimiters() which provides full feature parity.
 */
export const fallbackRateLimiter = (() => {
  const requests = new Map<string, { count: number; resetTime: number }>();
  const WINDOW_SIZE = getEnvNumber('RATE_LIMIT_FALLBACK_WINDOW_MS', 15 * 60 * 1000); // 15 minutes
  const MAX_REQUESTS = getEnvNumber('RATE_LIMIT_FALLBACK_MAX_REQUESTS', 100);
  // Maximum number of IP entries to track to prevent memory exhaustion under attack.
  // When exceeded, oldest entries are evicted even if still within the window.
  const MAX_ENTRIES = getEnvNumber('RATE_LIMIT_FALLBACK_MAX_ENTRIES', 50000);

  return (req: Request, res: Response, next: NextFunction) => {
    const key = normalizeIpKey(req.ip);
    const now = Date.now();
    const windowStart = now - WINDOW_SIZE;

    // Clean up old entries
    for (const [ip, data] of requests.entries()) {
      if (data.resetTime < windowStart) {
        requests.delete(ip);
      }
    }

    // Evict oldest entries if we've exceeded the maximum size to prevent
    // memory exhaustion from IP enumeration or distributed attacks.
    // Map maintains insertion order, so first entries are oldest.
    if (requests.size > MAX_ENTRIES) {
      const entriesToRemove = requests.size - MAX_ENTRIES;
      let removed = 0;
      for (const ip of requests.keys()) {
        if (removed >= entriesToRemove) break;
        requests.delete(ip);
        removed++;
      }
      logger.warn('Fallback rate limiter evicted entries due to size limit', {
        evicted: removed,
        currentSize: requests.size,
        maxEntries: MAX_ENTRIES,
      });
    }

    const current = requests.get(key);
    const resetTimestamp = Math.ceil((now + WINDOW_SIZE) / 1000);

    if (!current) {
      requests.set(key, { count: 1, resetTime: now });
      setRateLimitHeaders(res, MAX_REQUESTS, MAX_REQUESTS - 1, resetTimestamp);
      return next();
    }

    if (current.resetTime < windowStart) {
      requests.set(key, { count: 1, resetTime: now });
      setRateLimitHeaders(res, MAX_REQUESTS, MAX_REQUESTS - 1, resetTimestamp);
      return next();
    }

    if (current.count >= MAX_REQUESTS) {
      const retryAfter = Math.ceil((current.resetTime + WINDOW_SIZE - now) / 1000);
      setRateLimitHeaders(
        res,
        MAX_REQUESTS,
        0,
        Math.ceil((current.resetTime + WINDOW_SIZE) / 1000)
      );
      res.set('Retry-After', String(retryAfter));

      logger.warn('Fallback rate limit exceeded', {
        ip: req.ip,
        path: req.path,
        count: current.count,
      });

      // Record rate limit hit metric
      getMetricsService().recordRateLimitHit(req.path, 'fallback');

      return res.status(429).json({
        success: false,
        error: {
          message: 'Too many requests, please try again later',
          code: 'RATE_LIMIT_EXCEEDED',
          retryAfter,
          timestamp: new Date().toISOString(),
        },
      });
    }

    current.count++;
    setRateLimitHeaders(res, MAX_REQUESTS, MAX_REQUESTS - current.count, resetTimestamp);
    next();
  };
})();

/**
 * Utility to reset rate limiters for testing purposes.
 * Only available when not using Redis.
 */
export const __testResetRateLimiters = () => {
  if (usingRedis) {
    logger.warn('Cannot reset Redis-backed rate limiters in test mode');
    return;
  }

  // Re-create all memory limiters
  const configs = getRateLimitConfigsCached();
  Object.entries(configs).forEach(([key, config]) => {
    rateLimiters[key] = new RateLimiterMemory({
      keyPrefix: config.keyPrefix,
      points: config.points,
      duration: config.duration,
      blockDuration: config.blockDuration,
    });
  });
};

/**
 * Utility to clear config cache for testing purposes.
 * Call this before initializeMemoryRateLimiters() to pick up new env vars.
 */
export const __testClearConfigCache = () => {
  cachedConfigs = null;
};

/**
 * Get the current rate limit config for a specific limiter.
 * Useful for tests and debugging.
 */
export const getRateLimitConfig = (limiterKey: string): RateLimitConfig | undefined => {
  return getRateLimitConfigsCached()[limiterKey];
};
