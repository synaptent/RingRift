/**
 * Environment Variable Schema and Validation
 *
 * This module defines the Zod schema for all environment variables,
 * validates them at startup, and exports a typed env object.
 *
 * All environment variables should be defined here with appropriate
 * validation rules and defaults.
 */

import { z } from 'zod';
import { isJestRuntime } from '../../shared/utils/envFlags';

/**
 * Node environment schema - supports development, staging, production, and test.
 */
export const NodeEnvSchema = z.enum(['development', 'staging', 'production', 'test']);
export type NodeEnv = z.infer<typeof NodeEnvSchema>;

/**
 * Application topology schema - defines deployment configuration.
 */
export const AppTopologySchema = z.enum(['single', 'multi-unsafe', 'multi-sticky']);
export type AppTopology = z.infer<typeof AppTopologySchema>;

/**
 * Rules engine mode schema.
 */
export const RulesModeSchema = z.enum(['ts', 'python', 'shadow']);
export type RulesMode = z.infer<typeof RulesModeSchema>;

/**
 * Log level schema.
 */
export const LogLevelSchema = z.enum(['error', 'warn', 'info', 'debug', 'trace']);
export type LogLevel = z.infer<typeof LogLevelSchema>;

/**
 * Log format schema.
 */
export const LogFormatSchema = z.enum(['json', 'pretty']);
export type LogFormat = z.infer<typeof LogFormatSchema>;

/**
 * Complete environment variable schema with validation rules and defaults.
 *
 * Variables are organized by category for clarity:
 * - Environment & Server
 * - Database
 * - Redis
 * - Authentication
 * - AI Service
 * - Rate Limiting
 * - Logging
 * - CORS
 * - Feature Flags
 * - Game Configuration
 * - Email (optional)
 */
export const EnvSchema = z.object({
  // ===================================================================
  // ENVIRONMENT & SERVER
  // ===================================================================

  /** Application environment mode */
  NODE_ENV: NodeEnvSchema.default('development'),

  /** HTTP server port */
  PORT: z.coerce.number().int().min(1).max(65535).default(3000),

  /** Server bind address */
  HOST: z.string().default('0.0.0.0'),

  /** Application topology for deployment */
  RINGRIFT_APP_TOPOLOGY: AppTopologySchema.default('single'),

  /** Application version (injected by npm) */
  npm_package_version: z.string().optional(),

  // ===================================================================
  // DATABASE
  // ===================================================================

  /** PostgreSQL connection URL (required in production) */
  DATABASE_URL: z.string().optional(),

  /** Minimum database pool connections */
  DATABASE_POOL_MIN: z.coerce.number().int().min(1).default(2),

  /** Maximum database pool connections */
  DATABASE_POOL_MAX: z.coerce.number().int().min(1).default(10),

  // ===================================================================
  // REDIS
  // ===================================================================

  /** Redis connection URL (required in production) */
  REDIS_URL: z.string().optional(),

  /** Redis authentication password */
  REDIS_PASSWORD: z.string().optional(),

  /** Enable TLS for Redis connections */
  REDIS_TLS: z
    .string()
    .optional()
    .transform((val) => val === 'true' || val === '1'),

  // ===================================================================
  // AUTHENTICATION
  // ===================================================================

  /** JWT secret for signing access tokens (required in production, min 32 chars) */
  JWT_SECRET: z.string().optional(),

  /** JWT secret for signing refresh tokens (required in production, min 32 chars) */
  JWT_REFRESH_SECRET: z.string().optional(),

  /** Access token expiration time */
  JWT_EXPIRES_IN: z.string().default('15m'),

  /** Refresh token expiration time */
  JWT_REFRESH_EXPIRES_IN: z.string().default('7d'),

  /** Maximum failed login attempts before lockout */
  AUTH_MAX_FAILED_LOGIN_ATTEMPTS: z.coerce.number().int().positive().default(10),

  /** Time window for counting failed login attempts (seconds) */
  AUTH_FAILED_LOGIN_WINDOW_SECONDS: z.coerce.number().int().positive().default(900),

  /** Duration of login lockout (seconds) */
  AUTH_LOCKOUT_DURATION_SECONDS: z.coerce.number().int().positive().default(900),

  /** Enable/disable login lockout feature */
  AUTH_LOGIN_LOCKOUT_ENABLED: z.string().optional(),

  /** bcrypt rounds for password hashing */
  BCRYPT_ROUNDS: z.coerce.number().int().min(4).max(31).default(12),

  // ===================================================================
  // AI SERVICE
  // ===================================================================

  /** AI service base URL (required in production) */
  AI_SERVICE_URL: z.string().optional(),

  /** AI service request timeout (milliseconds) */
  AI_SERVICE_REQUEST_TIMEOUT_MS: z.coerce.number().int().positive().default(5000),

  /** AI rules request timeout (milliseconds) */
  AI_RULES_REQUEST_TIMEOUT_MS: z.coerce.number().int().positive().default(5000),

  /** Maximum concurrent AI requests */
  AI_MAX_CONCURRENT_REQUESTS: z.coerce.number().int().positive().default(16),

  /** Enable AI fallback to local heuristics when service unavailable */
  AI_FALLBACK_ENABLED: z
    .string()
    .optional()
    .transform((val) => (val === undefined ? true : val !== 'false' && val !== '0')),

  // ===================================================================
  // RATE LIMITING
  // ===================================================================

  /** General API rate limit - requests per window (anonymous) */
  RATE_LIMIT_API_POINTS: z.coerce.number().int().positive().default(50),

  /** General API rate limit - window duration (seconds) */
  RATE_LIMIT_API_DURATION: z.coerce.number().int().positive().default(60),

  /** General API rate limit - block duration (seconds) */
  RATE_LIMIT_API_BLOCK_DURATION: z.coerce.number().int().positive().default(300),

  /** Authenticated API rate limit - requests per window */
  RATE_LIMIT_API_AUTH_POINTS: z.coerce.number().int().positive().default(200),

  /** Authenticated API rate limit - window duration (seconds) */
  RATE_LIMIT_API_AUTH_DURATION: z.coerce.number().int().positive().default(60),

  /** Authenticated API rate limit - block duration (seconds) */
  RATE_LIMIT_API_AUTH_BLOCK_DURATION: z.coerce.number().int().positive().default(300),

  /** Auth endpoint rate limit - requests per window */
  RATE_LIMIT_AUTH_POINTS: z.coerce.number().int().positive().default(10),

  /** Auth endpoint rate limit - window duration (seconds) */
  RATE_LIMIT_AUTH_DURATION: z.coerce.number().int().positive().default(900),

  /** Auth endpoint rate limit - block duration (seconds) */
  RATE_LIMIT_AUTH_BLOCK_DURATION: z.coerce.number().int().positive().default(1800),

  /** Login endpoint rate limit - requests per window */
  RATE_LIMIT_AUTH_LOGIN_POINTS: z.coerce.number().int().positive().default(5),

  /** Login endpoint rate limit - window duration (seconds) */
  RATE_LIMIT_AUTH_LOGIN_DURATION: z.coerce.number().int().positive().default(900),

  /** Login endpoint rate limit - block duration (seconds) */
  RATE_LIMIT_AUTH_LOGIN_BLOCK_DURATION: z.coerce.number().int().positive().default(1800),

  /** Registration rate limit - requests per window */
  RATE_LIMIT_AUTH_REGISTER_POINTS: z.coerce.number().int().positive().default(3),

  /** Registration rate limit - window duration (seconds) */
  RATE_LIMIT_AUTH_REGISTER_DURATION: z.coerce.number().int().positive().default(3600),

  /** Registration rate limit - block duration (seconds) */
  RATE_LIMIT_AUTH_REGISTER_BLOCK_DURATION: z.coerce.number().int().positive().default(3600),

  /** Password reset rate limit - requests per window */
  RATE_LIMIT_AUTH_PWD_RESET_POINTS: z.coerce.number().int().positive().default(3),

  /** Password reset rate limit - window duration (seconds) */
  RATE_LIMIT_AUTH_PWD_RESET_DURATION: z.coerce.number().int().positive().default(3600),

  /** Password reset rate limit - block duration (seconds) */
  RATE_LIMIT_AUTH_PWD_RESET_BLOCK_DURATION: z.coerce.number().int().positive().default(3600),

  /** Game endpoint rate limit - requests per window */
  RATE_LIMIT_GAME_POINTS: z.coerce.number().int().positive().default(200),

  /** Game endpoint rate limit - window duration (seconds) */
  RATE_LIMIT_GAME_DURATION: z.coerce.number().int().positive().default(60),

  /** Game endpoint rate limit - block duration (seconds) */
  RATE_LIMIT_GAME_BLOCK_DURATION: z.coerce.number().int().positive().default(300),

  /** Game moves rate limit - requests per window */
  RATE_LIMIT_GAME_MOVES_POINTS: z.coerce.number().int().positive().default(100),

  /** Game moves rate limit - window duration (seconds) */
  RATE_LIMIT_GAME_MOVES_DURATION: z.coerce.number().int().positive().default(60),

  /** Game moves rate limit - block duration (seconds) */
  RATE_LIMIT_GAME_MOVES_BLOCK_DURATION: z.coerce.number().int().positive().default(60),

  /** WebSocket connection rate limit - connections per window */
  RATE_LIMIT_WS_POINTS: z.coerce.number().int().positive().default(10),

  /** WebSocket connection rate limit - window duration (seconds) */
  RATE_LIMIT_WS_DURATION: z.coerce.number().int().positive().default(60),

  /** WebSocket connection rate limit - block duration (seconds) */
  RATE_LIMIT_WS_BLOCK_DURATION: z.coerce.number().int().positive().default(300),

  /** Per-user game creation rate limit - games per window */
  RATE_LIMIT_GAME_CREATE_USER_POINTS: z.coerce.number().int().positive().default(20),

  /** Per-user game creation rate limit - window duration (seconds) */
  RATE_LIMIT_GAME_CREATE_USER_DURATION: z.coerce.number().int().positive().default(600),

  /** Per-user game creation rate limit - block duration (seconds) */
  RATE_LIMIT_GAME_CREATE_USER_BLOCK_DURATION: z.coerce.number().int().positive().default(600),

  /** Per-IP game creation rate limit - games per window */
  RATE_LIMIT_GAME_CREATE_IP_POINTS: z.coerce.number().int().positive().default(50),

  /** Per-IP game creation rate limit - window duration (seconds) */
  RATE_LIMIT_GAME_CREATE_IP_DURATION: z.coerce.number().int().positive().default(600),

  /** Per-IP game creation rate limit - block duration (seconds) */
  RATE_LIMIT_GAME_CREATE_IP_BLOCK_DURATION: z.coerce.number().int().positive().default(600),

  /** Fallback (in-memory) rate limit window (milliseconds) */
  RATE_LIMIT_FALLBACK_WINDOW_MS: z.coerce.number().int().positive().default(900000),

  /** Fallback (in-memory) rate limit max requests */
  RATE_LIMIT_FALLBACK_MAX_REQUESTS: z.coerce.number().int().positive().default(100),

  // ===================================================================
  // LOGGING
  // ===================================================================

  /** Application log level */
  LOG_LEVEL: LogLevelSchema.default('info'),

  /** Log output format */
  LOG_FORMAT: LogFormatSchema.default('json'),

  /** Log file path (optional) */
  LOG_FILE: z.string().optional(),

  // ===================================================================
  // CORS
  // ===================================================================

  /** Primary CORS origin */
  CORS_ORIGIN: z.string().default('http://localhost:5173'),

  /** Public client URL for redirects etc. */
  CLIENT_URL: z.string().default('http://localhost:3000'),

  /** Comma-separated list of allowed origins */
  ALLOWED_ORIGINS: z.string().default('http://localhost:5173,http://localhost:3000'),

  // ===================================================================
  // FEATURE FLAGS
  // ===================================================================

  /** Enable Prometheus metrics endpoint */
  ENABLE_METRICS: z
    .string()
    .optional()
    .transform((val) => (val === undefined ? true : val !== 'false' && val !== '0')),

  /** Enable health check endpoints */
  ENABLE_HEALTH_CHECKS: z
    .string()
    .optional()
    .transform((val) => (val === undefined ? true : val !== 'false' && val !== '0')),

  /** Metrics server port */
  METRICS_PORT: z.coerce.number().int().min(1).max(65535).default(9090),

  /** Rules engine mode */
  RINGRIFT_RULES_MODE: RulesModeSchema.optional(),

  /**
   * Orchestrator adapter for turn processing.
   *
   * PERMANENTLY ENABLED as of 2025-12-01 (Phase 3 migration complete).
   * The orchestrator is now the canonical turn processor. Legacy path removed.
   *
   * This flag is hardcoded to `true` and no longer reads from environment variables.
   * The `useOrchestratorAdapter` property on GameEngine/ClientSandboxEngine remains
   * for internal state management but always evaluates to true.
   *
   * @see docs/ORCHESTRATOR_MIGRATION_COMPLETION_PLAN.md
   */
  ORCHESTRATOR_ADAPTER_ENABLED: z
    .any()
    .transform((): true => true)
    .default(true),

  /** Enable AI analysis mode (position evaluation streaming) */
  ENABLE_ANALYSIS_MODE: z
    .string()
    .optional()
    .transform((val) => (val === undefined ? false : val === 'true' || val === '1')),

  // ===================================================================
  // ORCHESTRATOR ROLLOUT CONFIGURATION
  // ===================================================================

  /** Enable shadow mode - run both engines, compare results */
  ORCHESTRATOR_SHADOW_MODE_ENABLED: z
    .string()
    .default('false')
    .transform((val) => val === 'true' || val === '1'),

  /** Comma-separated list of user IDs to force-enable orchestrator */
  ORCHESTRATOR_ALLOWLIST_USERS: z.string().default(''),

  /** Comma-separated list of user IDs to force-disable orchestrator */
  ORCHESTRATOR_DENYLIST_USERS: z.string().default(''),

  /** Enable circuit breaker for auto-disable on errors */
  ORCHESTRATOR_CIRCUIT_BREAKER_ENABLED: z
    .string()
    .default('true')
    .transform((val) => val === 'true' || val === '1'),

  /** Error rate percentage to trip circuit breaker */
  ORCHESTRATOR_ERROR_THRESHOLD_PERCENT: z.coerce.number().int().min(0).max(100).default(5),

  /** Time window in seconds for error rate calculation */
  ORCHESTRATOR_ERROR_WINDOW_SECONDS: z.coerce.number().int().positive().default(300),

  /** P99 latency threshold in milliseconds for warnings */
  ORCHESTRATOR_LATENCY_THRESHOLD_MS: z.coerce.number().int().positive().default(500),

  // ===================================================================
  // GAME CONFIGURATION
  // ===================================================================

  /** Maximum concurrent games */
  MAX_CONCURRENT_GAMES: z.coerce.number().int().positive().default(1000),

  /** Game inactivity timeout (minutes) */
  GAME_TIMEOUT_MINUTES: z.coerce.number().int().positive().default(60),

  /** AI thinking time (milliseconds) */
  AI_THINK_TIME_MS: z.coerce.number().int().positive().default(2000),

  /**
   * Optional override for decision-phase timeouts (milliseconds).
   *
   * These are primarily intended for non-production environments and
   * specialised test harnesses (for example Playwright E2E) that need
   * shorter timeouts to exercise decision timeout behaviour end-to-end.
   *
   * When unset, the server falls back to the hard-coded defaults in
   * unified.ts (30s total timeout, 5s warning, 15s extension).
   */
  DECISION_PHASE_TIMEOUT_MS: z.coerce.number().int().positive().optional(),
  DECISION_PHASE_TIMEOUT_WARNING_MS: z.coerce.number().int().positive().optional(),
  DECISION_PHASE_TIMEOUT_EXTENSION_MS: z.coerce.number().int().positive().optional(),

  /** Maximum spectators per game */
  MAX_SPECTATORS_PER_GAME: z.coerce.number().int().positive().default(50),

  // ===================================================================
  // FILE STORAGE
  // ===================================================================

  /** Directory for user uploads */
  UPLOAD_DIR: z.string().default('./uploads'),

  /** Maximum upload file size (bytes) */
  MAX_FILE_SIZE: z.coerce.number().int().positive().default(5242880),

  // ===================================================================
  // EMAIL (OPTIONAL)
  // ===================================================================

  /** SMTP server host */
  SMTP_HOST: z.string().optional(),

  /** SMTP server port */
  SMTP_PORT: z.coerce.number().int().min(1).max(65535).optional(),

  /** SMTP authentication username */
  SMTP_USER: z.string().optional(),

  /** SMTP authentication password */
  SMTP_PASSWORD: z.string().optional(),

  /** Enable TLS for SMTP */
  SMTP_TLS: z
    .string()
    .optional()
    .transform((val) => val === 'true' || val === '1'),

  /** Email sender address */
  SMTP_FROM: z.string().optional(),
});

/**
 * Inferred type for raw environment variables.
 */
export type RawEnv = z.infer<typeof EnvSchema>;

/**
 * Result of environment validation.
 */
export interface EnvValidationResult {
  success: boolean;
  data?: RawEnv;
  errors?: Array<{ path: string; message: string }>;
}

/**
 * Parse and validate environment variables.
 *
 * @param env - Environment object to parse (defaults to process.env)
 * @returns Validation result with data or errors
 */
export function parseEnv(
  env: Record<string, string | undefined> = process.env as Record<string, string | undefined>
): EnvValidationResult {
  const result = EnvSchema.safeParse(env);

  if (!result.success) {
    // Zod 4 exposes `issues`, older versions used `errors`. Support both defensively.
    const zodError = result.error as unknown as {
      issues?: Array<{ path: (string | number)[]; message: string }>;
      errors?: Array<{ path: (string | number)[]; message: string }>;
    };

    const issues = Array.isArray(zodError.issues)
      ? zodError.issues
      : Array.isArray((zodError as any).errors)
        ? (zodError as any).errors
        : [];

    const errors =
      issues.length > 0
        ? issues.map((e: { path: (string | number)[]; message: string }) => ({
            path: e.path.join('.'),
            message: e.message,
          }))
        : [
            {
              path: '',
              message: result.error.message,
            },
          ];

    return {
      success: false,
      errors,
    };
  }

  return {
    success: true,
    data: result.data,
  };
}

/**
 * Load and validate environment variables, exiting on failure.
 *
 * This function should be called once at startup. If validation fails,
 * it prints clear error messages and exits the process.
 *
 * @param env - Environment object to parse (defaults to process.env)
 * @returns Validated environment object
 */
export function loadEnvOrExit(
  env: Record<string, string | undefined> = process.env as Record<string, string | undefined>
): RawEnv {
  const result = parseEnv(env);

  if (!result.success) {
    console.error('❌ Invalid environment configuration:');
    for (const error of result.errors ?? []) {
      console.error(`  - ${error.path || 'root'}: ${error.message}`);
    }
    process.exit(1);
  }

  return (
    result.data ??
    (() => {
      throw new Error('Missing env data after successful parse');
    })()
  );
}

/**
 * Determine effective node environment.
 *
 * When running under Jest, always treats the environment as 'test'
 * regardless of NODE_ENV to ensure test-specific behavior.
 *
 * @param rawEnv - Raw environment variables
 * @returns Effective node environment
 */
export function getEffectiveNodeEnv(rawEnv: RawEnv): NodeEnv {
  return isJestRuntime() ? 'test' : rawEnv.NODE_ENV;
}

/**
 * Check if running in production mode.
 */
export function isProduction(nodeEnv: NodeEnv): boolean {
  return nodeEnv === 'production';
}

/**
 * Check if running in staging mode.
 */
export function isStaging(nodeEnv: NodeEnv): boolean {
  return nodeEnv === 'staging';
}

/**
 * Check if running in development mode.
 */
export function isDevelopment(nodeEnv: NodeEnv): boolean {
  return nodeEnv === 'development';
}

/**
 * Check if running in test mode.
 */
export function isTest(nodeEnv: NodeEnv): boolean {
  return nodeEnv === 'test';
}

/**
 * Check if running in a production-like environment (production or staging).
 */
export function isProductionLike(nodeEnv: NodeEnv): boolean {
  return nodeEnv === 'production' || nodeEnv === 'staging';
}
