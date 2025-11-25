// Centralized application configuration for the Node.js backend.
// Parses and validates process.env once at startup and exposes a
// typed, runtime-validated config object for server code.
//
// This module is intentionally Node-only and should not be imported
// from client/browser bundles.

import dotenv from 'dotenv';
import { z } from 'zod';
import { getRulesMode } from '../shared/utils/envFlags';
import {
  isPlaceholderSecret,
  SECRET_MIN_LENGTHS,
  validateSecretsOrThrow,
} from './utils/secretsValidation';
import { NodeEnvSchema, AppTopologySchema, parseEnv, getEffectiveNodeEnv } from './config/env';

// Load .env into process.env before we read anything from it.
dotenv.config();

// Parse the raw environment with comprehensive Zod schema validation.
// Uses the centralized schema from config/env.ts for consistency.
const envResult = parseEnv(process.env as Record<string, string | undefined>);
if (!envResult.success) {
  console.error('❌ Invalid environment configuration:');
  for (const error of envResult.errors!) {
    console.error(`  - ${error.path || 'root'}: ${error.message}`);
  }
  process.exit(1);
}
const env = envResult.data!;

// Determine effective node environment.
// When running under Jest, JEST_WORKER_ID is always defined. In that case we
// treat the effective nodeEnv as "test" even if NODE_ENV was set to
// "development" via a .env file, so that test-only switches (e.g. timer
// suppression in GameEngine) behave correctly.
const nodeEnv = getEffectiveNodeEnv(env);
const isJestRuntime =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  (process as any).env.JEST_WORKER_ID !== undefined;
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
    corsOrigin: z.string().min(1),
    publicClientUrl: z.string().min(1),
    allowedOrigins: z.array(z.string().min(1)).nonempty(),
    websocketOrigin: z.string().min(1),
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
  }),
  rules: z.object({
    mode: z.union([z.literal('ts'), z.literal('python'), z.literal('shadow')]),
  }),
  decisionPhaseTimeouts: z.object({
    defaultTimeoutMs: z.number().int().positive(),
    warningBeforeTimeoutMs: z.number().int().positive(),
    extensionMs: z.number().int().positive(),
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
    corsOrigin,
    publicClientUrl,
    allowedOrigins,
    websocketOrigin,
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
  },
  rules: {
    mode: getRulesMode(),
  },
  decisionPhaseTimeouts: {
    defaultTimeoutMs: 30_000, // 30 seconds for decision phases
    warningBeforeTimeoutMs: 5_000, // Warning 5 seconds before timeout
    extensionMs: 15_000, // Optional extension time
  },
};

export type AppConfig = z.infer<typeof ConfigSchema>;

// Run comprehensive secrets validation (will throw on critical errors in production).
// In test/development, this logs warnings but doesn't fail for missing optional secrets.
// This is done before the final config parse to provide better error messages.
if (!isJestRuntime) {
  // Skip secrets validation in Jest to avoid test interference; tests should
  // use validateSecretsOrThrow directly with controlled env vars.
  validateSecretsOrThrow(isProduction);
}

// Parse and freeze the final config so downstream code gets a fully
// validated, immutable view.
export const config: AppConfig = Object.freeze(ConfigSchema.parse(preliminaryConfig));

// Re-export validation utilities for use in tests and other modules.
export {
  validateSecretsOrThrow,
  validateAllSecrets,
  isPlaceholderSecret,
} from './utils/secretsValidation';
