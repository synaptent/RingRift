/**
 * Unified Application Configuration
 *
 * This module is the canonical source of truth for all application configuration.
 * It parses environment variables, validates them with Zod, and exports a frozen
 * config object that all server code should use.
 *
 * Architecture:
 * - `env.ts` defines the raw environment variable schema
 * - `unified.ts` (this file) assembles the typed application config
 * - `index.ts` re-exports everything for convenient imports
 * - `topology.ts` handles deployment topology enforcement
 *
 * Usage:
 *   import { config } from './config';
 *   // or
 *   import { config, validateSecretsOrThrow } from './config';
 */

import dotenv from 'dotenv';
import { z } from 'zod';
import { getRulesMode, isJestRuntime } from '../../shared/utils/envFlags';
import {
  isPlaceholderSecret,
  SECRET_MIN_LENGTHS,
  validateSecretsOrThrow,
} from '../utils/secretsValidation';
import {
  NodeEnvSchema,
  AppTopologySchema,
  RulesModeSchema,
  LogFormatSchema,
  parseEnv,
  getEffectiveNodeEnv,
} from './env';

// Load .env into process.env before we read anything from it.
// Skip in test mode to prevent .env from overriding test-specific env vars
// (e.g., rate limit settings are set to 100000 in .env for dev but tests expect defaults).
if (process.env.NODE_ENV !== 'test') {
  dotenv.config();
}

// Parse the raw environment with comprehensive Zod schema validation.
// Uses the centralized schema from config/env.ts for consistency.
const envResult = parseEnv(process.env as Record<string, string | undefined>);
if (!envResult.success) {
  console.error('❌ Invalid environment configuration:');
  for (const error of envResult.errors ?? []) {
    console.error(`  - ${error.path || 'root'}: ${error.message}`);
  }
  process.exit(1);
}
const env =
  envResult.data ??
  (() => {
    throw new Error('Missing env data after successful parse');
  })();

// Determine effective node environment.
// When running under Jest, JEST_WORKER_ID is always defined. In that case we
// treat the effective nodeEnv as "test" even if NODE_ENV was set to
// "development" via a .env file, so that test-only switches (e.g. timer
// suppression in GameEngine) behave correctly.
const nodeEnv = getEffectiveNodeEnv(env);
const inJestRuntime = isJestRuntime();
const isProduction = nodeEnv === 'production';
const isTest = nodeEnv === 'test';
const isDevelopment = nodeEnv === 'development';

// Database URL – required in production, optional elsewhere.
const databaseUrl = env.DATABASE_URL?.trim() || undefined;
if (isProduction && !databaseUrl) {
  throw new Error('DATABASE_URL is required when NODE_ENV=production');
}

// Redis configuration – URL required in production, optional in dev/test
// (defaults to local Redis).
let redisUrl = env.REDIS_URL?.trim() || undefined;
if (!redisUrl && !isProduction) {
  redisUrl = 'redis://localhost:6379';
}
if (isProduction && !redisUrl) {
  throw new Error('REDIS_URL is required when NODE_ENV=production');
}
const redisPassword = env.REDIS_PASSWORD?.trim() || undefined;

// Auth/JWT configuration.
//
// In non-production environments we intentionally fall back to stable
// in-memory secrets so that local development and tests work even when
// JWT env vars are omitted. This mirrors the previous behaviour in
// auth middleware.
const jwtSecretFromEnv = env.JWT_SECRET?.trim() || undefined;
const jwtRefreshFromEnv = env.JWT_REFRESH_SECRET?.trim() || undefined;

let jwtSecret = jwtSecretFromEnv;
if (!jwtSecret && !isProduction) {
  jwtSecret = 'dev-access-token-secret';
}

let jwtRefreshSecret = jwtRefreshFromEnv || jwtSecretFromEnv;
if (!jwtRefreshSecret && !isProduction) {
  jwtRefreshSecret = 'dev-refresh-token-secret';
}

if (isProduction) {
  const missingOrEmpty =
    !jwtSecret || !jwtSecret.trim() || !jwtRefreshSecret || !jwtRefreshSecret.trim();

  const usingPlaceholder = isPlaceholderSecret(jwtSecret) || isPlaceholderSecret(jwtRefreshSecret);

  // Check minimum length requirement for production (32+ characters)
  const minLength = SECRET_MIN_LENGTHS.JWT_SECRET || 32;
  const jwtTooShort = jwtSecret && jwtSecret.trim().length < minLength;
  const refreshTooShort = jwtRefreshSecret && jwtRefreshSecret.trim().length < minLength;

  if (missingOrEmpty || usingPlaceholder || jwtTooShort || refreshTooShort) {
    const problems: string[] = [];
    if (missingOrEmpty) {
      problems.push('JWT secrets must be non-empty');
    }
    if (usingPlaceholder) {
      problems.push(
        'JWT secrets must not use placeholder values from .env.example or docker-compose.yml'
      );
    }
    if (jwtTooShort) {
      problems.push(`JWT_SECRET must be at least ${minLength} characters`);
    }
    if (refreshTooShort) {
      problems.push(`JWT_REFRESH_SECRET must be at least ${minLength} characters`);
    }

    throw new Error(
      `Invalid JWT configuration for NODE_ENV=production: ${problems.join(
        '; '
      )}. Please set JWT_SECRET and JWT_REFRESH_SECRET to strong, unique values (minimum ${minLength} characters).`
    );
  }
}

// AI service URL.
// - In non-production (development/test), defaults to a local FastAPI service
//   on port 8001 when unset so that developers can run the AI service on the
//   host without extra configuration.
// - In production, we *require* AI_SERVICE_URL to be explicitly set so that
//   containerized deployments never accidentally point at localhost.
let resolvedAiServiceUrl: string | undefined = env.AI_SERVICE_URL?.trim() || undefined;

if (!resolvedAiServiceUrl && !isProduction) {
  resolvedAiServiceUrl = 'http://localhost:8001';
}

if (!resolvedAiServiceUrl && isProduction) {
  throw new Error(
    'AI_SERVICE_URL is required when NODE_ENV=production. ' +
      "For Docker deployments, set AI_SERVICE_URL to the internal service URL, for example 'http://ai-service:8001' when using docker-compose."
  );
}

const aiServiceUrl = resolvedAiServiceUrl as string;

// Application version – driven by npm's injected env var when available.
const appVersion = env.npm_package_version?.trim() || '1.0.0';

// CORS / client origins
const corsOrigin = env.CORS_ORIGIN;
const publicClientUrl = env.CLIENT_URL;
const allowedOrigins = env.ALLOWED_ORIGINS.split(',')
  .map((v) => v.trim())
  .filter(Boolean);

// WebSocket origin follows the same precedence that was previously
// in src/server/websocket/server.ts.
const websocketOrigin =
  env.CLIENT_URL?.trim() ||
  env.CORS_ORIGIN?.trim() ||
  (allowedOrigins[0] ?? 'http://localhost:5173');

const metricsPortExplicit = Object.prototype.hasOwnProperty.call(process.env, 'METRICS_PORT');
const metricsPort = env.METRICS_PORT;
const metricsExposeOnMain = !metricsPortExplicit || metricsPort === env.PORT;

/**
 * Application configuration schema with comprehensive validation.
 */
const ConfigSchema = z.object({
  nodeEnv: NodeEnvSchema,
  isProduction: z.boolean(),
  isDevelopment: z.boolean(),
  isTest: z.boolean(),
  app: z.object({
    version: z.string().min(1),
    topology: AppTopologySchema,
  }),
  server: z.object({
    port: z.number().int().positive(),
    host: z.string().min(1),
    corsOrigin: z.string().min(1),
    publicClientUrl: z.string().min(1),
    allowedOrigins: z.array(z.string().min(1)).nonempty(),
    websocketOrigin: z.string().min(1),
    wsReconnectionTimeoutMs: z.number().int().positive(),
  }),
  metrics: z.object({
    /** API key for securing the /metrics endpoint */
    apiKey: z.string().optional(),
    /** Whether the metrics endpoint is enabled */
    enabled: z.boolean(),
    /** Port for the optional dedicated metrics server */
    port: z.number().int().positive(),
    /** Whether /metrics should be exposed on the main HTTP server */
    exposeOnMain: z.boolean(),
  }),
  healthChecks: z.object({
    /** Whether health check endpoints are enabled */
    enabled: z.boolean(),
  }),
  database: z.object({
    // Optional outside production; required in production via manual guard above.
    url: z.string().min(1).optional(),
  }),
  redis: z.object({
    url: z.string().min(1),
    password: z.string().optional(),
  }),
  auth: z.object({
    jwtSecret: z.string().min(1),
    jwtRefreshSecret: z.string().min(1),
    accessTokenExpiresIn: z.string().min(1),
    refreshTokenExpiresIn: z.string().min(1),
    maxFailedLoginAttempts: z.number().int().positive(),
    failedLoginWindowSeconds: z.number().int().positive(),
    lockoutDurationSeconds: z.number().int().positive(),
    loginLockoutEnabled: z.boolean(),
  }),
  aiService: z.object({
    url: z.string().url(),
    requestTimeoutMs: z.number().int().positive(),
    rulesTimeoutMs: z.number().int().positive(),
    maxConcurrent: z.number().int().positive(),
  }),
  logging: z.object({
    level: z.string().min(1),
    format: LogFormatSchema,
    file: z.string().optional(),
  }),
  rules: z.object({
    mode: RulesModeSchema,
  }),
  decisionPhaseTimeouts: z.object({
    defaultTimeoutMs: z.number().int().positive(),
    warningBeforeTimeoutMs: z.number().int().positive(),
    extensionMs: z.number().int().positive(),
  }),
  featureFlags: z.object({
    orchestrator: z.object({
      adapterEnabled: z.boolean(),
      // NOTE: rolloutPercentage removed in Phase 3 migration - orchestrator is permanently enabled
      // NOTE: shadowModeEnabled removed - FSM is now canonical
      allowlistUsers: z.array(z.string()),
      denylistUsers: z.array(z.string()),
      circuitBreaker: z.object({
        enabled: z.boolean(),
        errorThresholdPercent: z.number().int().min(0).max(100),
        errorWindowSeconds: z.number().int().positive(),
      }),
      latencyThresholdMs: z.number().int().positive(),
    }),
    analysisMode: z
      .object({
        enabled: z.boolean(),
      })
      .optional(),
    sandboxAi: z.object({
      enabled: z.boolean(),
    }),
    httpMoveHarness: z.object({
      /**
       * Master switch for the internal HTTP move harness endpoint
       * (POST /api/games/:gameId/moves).
       *
       * When false, the harness route must behave as if it does not exist
       * (e.g. 404/NOT_FOUND) so that production environments can keep the
       * surface fully disabled by default.
       */
      enabled: z.boolean(),
      /**
       * Request timeout in milliseconds. If handlePlayerMoveFromHttp takes
       * longer than this threshold, the endpoint returns 504 Gateway Timeout.
       */
      timeoutMs: z.number().int().positive(),
    }),
  }),
  orchestrator: z.object({
    /**
     * High-level rules mode selector taken from RINGRIFT_RULES_MODE.
     * Note: current values are 'ts' | 'python'. Shadow mode has been removed.
     */
    rulesMode: RulesModeSchema,
    /** Master switch for using the orchestrator adapter on backend hosts. */
    adapterEnabled: z.boolean(),
    // NOTE: rolloutPercentage removed in Phase 3 migration - orchestrator is permanently enabled
    // NOTE: shadowModeEnabled removed - FSM is now canonical
    /** Whether the orchestrator circuit breaker is enabled. */
    circuitBreakerEnabled: z.boolean(),
  }),
});

const preliminaryConfig = {
  nodeEnv,
  isProduction,
  isDevelopment,
  isTest,
  app: {
    version: appVersion,
    topology: env.RINGRIFT_APP_TOPOLOGY,
  },
  server: {
    port: env.PORT,
    host: env.HOST,
    corsOrigin,
    publicClientUrl,
    allowedOrigins,
    websocketOrigin,
    wsReconnectionTimeoutMs: env.WS_RECONNECTION_TIMEOUT_MS,
  },
  metrics: {
    apiKey: env.METRICS_API_KEY?.trim() || undefined,
    enabled: env.ENABLE_METRICS,
    port: metricsPort,
    exposeOnMain: metricsExposeOnMain,
  },
  healthChecks: {
    enabled: env.ENABLE_HEALTH_CHECKS,
  },
  database: {
    url: databaseUrl,
  },
  redis: {
    url: redisUrl as string,
    password: redisPassword,
  },
  auth: {
    jwtSecret: jwtSecret as string,
    jwtRefreshSecret: jwtRefreshSecret as string,
    accessTokenExpiresIn: env.JWT_EXPIRES_IN,
    refreshTokenExpiresIn: env.JWT_REFRESH_EXPIRES_IN,
    maxFailedLoginAttempts: env.AUTH_MAX_FAILED_LOGIN_ATTEMPTS,
    failedLoginWindowSeconds: env.AUTH_FAILED_LOGIN_WINDOW_SECONDS,
    lockoutDurationSeconds: env.AUTH_LOCKOUT_DURATION_SECONDS,
    loginLockoutEnabled:
      env.AUTH_LOGIN_LOCKOUT_ENABLED === undefined
        ? true
        : !['false', '0'].includes(env.AUTH_LOGIN_LOCKOUT_ENABLED.toLowerCase()),
  },
  aiService: {
    url: aiServiceUrl,
    requestTimeoutMs: env.AI_SERVICE_REQUEST_TIMEOUT_MS,
    rulesTimeoutMs: env.AI_RULES_REQUEST_TIMEOUT_MS,
    maxConcurrent: env.AI_MAX_CONCURRENT_REQUESTS,
  },
  logging: {
    level: env.LOG_LEVEL,
    format: env.LOG_FORMAT,
    file: env.LOG_FILE,
  },
  rules: {
    mode: getRulesMode(),
  },
  orchestrator: {
    rulesMode: getRulesMode(),
    adapterEnabled: env.ORCHESTRATOR_ADAPTER_ENABLED,
    // NOTE: rolloutPercentage removed in Phase 3 migration - orchestrator is permanently enabled
    // NOTE: shadowModeEnabled removed - FSM is now canonical
    circuitBreakerEnabled: env.ORCHESTRATOR_CIRCUIT_BREAKER_ENABLED,
  },
  decisionPhaseTimeouts: {
    // Allow optional overrides via DECISION_PHASE_TIMEOUT_* env vars so
    // non-production harnesses (for example Playwright E2E) can shorten
    // decision-phase timers without affecting production defaults.
    defaultTimeoutMs: env.DECISION_PHASE_TIMEOUT_MS ?? 30_000, // 30 seconds
    warningBeforeTimeoutMs: env.DECISION_PHASE_TIMEOUT_WARNING_MS ?? 5_000, // warn 5s before
    extensionMs: env.DECISION_PHASE_TIMEOUT_EXTENSION_MS ?? 15_000, // optional extension
  },
  featureFlags: {
    orchestrator: {
      adapterEnabled: env.ORCHESTRATOR_ADAPTER_ENABLED,
      // NOTE: rolloutPercentage removed in Phase 3 migration - orchestrator is permanently enabled
      // NOTE: shadowModeEnabled removed - FSM is now canonical
      allowlistUsers: env.ORCHESTRATOR_ALLOWLIST_USERS.split(',').filter(Boolean),
      denylistUsers: env.ORCHESTRATOR_DENYLIST_USERS.split(',').filter(Boolean),
      circuitBreaker: {
        enabled: env.ORCHESTRATOR_CIRCUIT_BREAKER_ENABLED,
        errorThresholdPercent: env.ORCHESTRATOR_ERROR_THRESHOLD_PERCENT,
        errorWindowSeconds: env.ORCHESTRATOR_ERROR_WINDOW_SECONDS,
      },
      latencyThresholdMs: env.ORCHESTRATOR_LATENCY_THRESHOLD_MS,
    },
    analysisMode: {
      enabled: env.ENABLE_ANALYSIS_MODE,
    },
    sandboxAi: {
      enabled: env.ENABLE_SANDBOX_AI_ENDPOINTS,
    },
    httpMoveHarness: {
      enabled: env.ENABLE_HTTP_MOVE_HARNESS,
      timeoutMs: env.HTTP_MOVE_HARNESS_TIMEOUT_MS,
    },
  },
};

/**
 * Application configuration type inferred from the schema.
 */
export type AppConfig = z.infer<typeof ConfigSchema>;

// Run comprehensive secrets validation (will throw on critical errors in production).
// In test/development, this logs warnings but doesn't fail for missing optional secrets.
// This is done before the final config parse to provide better error messages.
if (!inJestRuntime) {
  // Skip secrets validation in Jest to avoid test interference; tests should
  // use validateSecretsOrThrow directly with controlled env vars.
  validateSecretsOrThrow(isProduction);
}

// Parse and freeze the final config so downstream code gets a fully
// validated, immutable view.
export const config: AppConfig = Object.freeze(ConfigSchema.parse(preliminaryConfig));

// Re-export validation utilities for use in tests and other modules.
export { validateSecretsOrThrow, isPlaceholderSecret } from '../utils/secretsValidation';
