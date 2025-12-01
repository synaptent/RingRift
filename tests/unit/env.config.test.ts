/**
 * Environment Configuration Tests
 *
 * Tests for the Zod-based environment variable validation system.
 * These tests verify that:
 * - Valid configurations pass validation
 * - Invalid configurations fail with clear error messages
 * - Defaults are applied correctly
 * - Type coercion works as expected
 * - Environment-specific validation rules are enforced
 */

import {
  EnvSchema,
  parseEnv,
  getEffectiveNodeEnv,
  isProduction,
  isStaging,
  isDevelopment,
  isTest,
  isProductionLike,
  type RawEnv,
} from '../../src/server/config/env';

// Minimal valid environment for testing
const baseValidEnv: Record<string, string> = {
  NODE_ENV: 'development',
  PORT: '3000',
  CORS_ORIGIN: 'http://localhost:5173',
  CLIENT_URL: 'http://localhost:3000',
  ALLOWED_ORIGINS: 'http://localhost:5173,http://localhost:3000',
  JWT_EXPIRES_IN: '15m',
  JWT_REFRESH_EXPIRES_IN: '7d',
  LOG_LEVEL: 'info',
};

describe('EnvSchema', () => {
  describe('NODE_ENV validation', () => {
    it('should accept valid NODE_ENV values', () => {
      const validEnvs = ['development', 'staging', 'production', 'test'];
      for (const nodeEnv of validEnvs) {
        const result = parseEnv({ ...baseValidEnv, NODE_ENV: nodeEnv });
        expect(result.success).toBe(true);
        expect(result.data?.NODE_ENV).toBe(nodeEnv);
      }
    });

    it('should default to development when NODE_ENV is not set', () => {
      const envWithoutNodeEnv = { ...baseValidEnv };
      delete envWithoutNodeEnv.NODE_ENV;
      const result = parseEnv(envWithoutNodeEnv);
      expect(result.success).toBe(true);
      expect(result.data?.NODE_ENV).toBe('development');
    });

    it('should reject invalid NODE_ENV values', () => {
      const result = parseEnv({ ...baseValidEnv, NODE_ENV: 'invalid' });
      expect(result.success).toBe(false);
      expect(result.errors).toBeDefined();
      expect(result.errors?.some((e) => e.path === 'NODE_ENV')).toBe(true);
    });
  });

  describe('PORT validation', () => {
    it('should coerce string PORT to number', () => {
      const result = parseEnv({ ...baseValidEnv, PORT: '8080' });
      expect(result.success).toBe(true);
      expect(result.data?.PORT).toBe(8080);
    });

    it('should default to 3000 when PORT is not set', () => {
      const envWithoutPort = { ...baseValidEnv };
      delete envWithoutPort.PORT;
      const result = parseEnv(envWithoutPort);
      expect(result.success).toBe(true);
      expect(result.data?.PORT).toBe(3000);
    });

    it('should reject PORT below 1', () => {
      const result = parseEnv({ ...baseValidEnv, PORT: '0' });
      expect(result.success).toBe(false);
    });

    it('should reject PORT above 65535', () => {
      const result = parseEnv({ ...baseValidEnv, PORT: '65536' });
      expect(result.success).toBe(false);
    });

    it('should accept valid port numbers within range', () => {
      const validPorts = ['1', '80', '443', '3000', '8080', '65535'];
      for (const port of validPorts) {
        const result = parseEnv({ ...baseValidEnv, PORT: port });
        expect(result.success).toBe(true);
        expect(result.data?.PORT).toBe(parseInt(port, 10));
      }
    });
  });

  describe('Database pool validation', () => {
    it('should default DATABASE_POOL_MIN to 2', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.DATABASE_POOL_MIN).toBe(2);
    });

    it('should default DATABASE_POOL_MAX to 10', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.DATABASE_POOL_MAX).toBe(10);
    });

    it('should coerce pool settings to numbers', () => {
      const result = parseEnv({
        ...baseValidEnv,
        DATABASE_POOL_MIN: '5',
        DATABASE_POOL_MAX: '20',
      });
      expect(result.success).toBe(true);
      expect(result.data?.DATABASE_POOL_MIN).toBe(5);
      expect(result.data?.DATABASE_POOL_MAX).toBe(20);
    });

    it('should reject pool settings below 1', () => {
      const result = parseEnv({ ...baseValidEnv, DATABASE_POOL_MIN: '0' });
      expect(result.success).toBe(false);
    });
  });

  describe('LOG_LEVEL validation', () => {
    it('should accept valid log levels', () => {
      const validLevels = ['error', 'warn', 'info', 'debug', 'trace'];
      for (const level of validLevels) {
        const result = parseEnv({ ...baseValidEnv, LOG_LEVEL: level });
        expect(result.success).toBe(true);
        expect(result.data?.LOG_LEVEL).toBe(level);
      }
    });

    it('should default to info when LOG_LEVEL is not set', () => {
      const envWithoutLogLevel = { ...baseValidEnv };
      delete envWithoutLogLevel.LOG_LEVEL;
      const result = parseEnv(envWithoutLogLevel);
      expect(result.success).toBe(true);
      expect(result.data?.LOG_LEVEL).toBe('info');
    });

    it('should reject invalid log levels', () => {
      const result = parseEnv({ ...baseValidEnv, LOG_LEVEL: 'verbose' });
      expect(result.success).toBe(false);
    });
  });

  describe('LOG_FORMAT validation', () => {
    it('should accept valid log formats', () => {
      const validFormats = ['json', 'pretty'];
      for (const format of validFormats) {
        const result = parseEnv({ ...baseValidEnv, LOG_FORMAT: format });
        expect(result.success).toBe(true);
        expect(result.data?.LOG_FORMAT).toBe(format);
      }
    });

    it('should default to json when LOG_FORMAT is not set', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.LOG_FORMAT).toBe('json');
    });
  });

  describe('Boolean transforms', () => {
    it('should transform REDIS_TLS to boolean', () => {
      let result = parseEnv({ ...baseValidEnv, REDIS_TLS: 'true' });
      expect(result.success).toBe(true);
      expect(result.data?.REDIS_TLS).toBe(true);

      result = parseEnv({ ...baseValidEnv, REDIS_TLS: '1' });
      expect(result.success).toBe(true);
      expect(result.data?.REDIS_TLS).toBe(true);

      result = parseEnv({ ...baseValidEnv, REDIS_TLS: 'false' });
      expect(result.success).toBe(true);
      expect(result.data?.REDIS_TLS).toBe(false);

      result = parseEnv({ ...baseValidEnv, REDIS_TLS: '0' });
      expect(result.success).toBe(true);
      expect(result.data?.REDIS_TLS).toBe(false);
    });

    it('should transform ENABLE_METRICS to boolean', () => {
      let result = parseEnv({ ...baseValidEnv, ENABLE_METRICS: 'false' });
      expect(result.success).toBe(true);
      expect(result.data?.ENABLE_METRICS).toBe(false);

      result = parseEnv({ ...baseValidEnv, ENABLE_METRICS: '0' });
      expect(result.success).toBe(true);
      expect(result.data?.ENABLE_METRICS).toBe(false);
    });

    it('should default ENABLE_METRICS to true', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.ENABLE_METRICS).toBe(true);
    });

    it('should transform AI_FALLBACK_ENABLED to boolean', () => {
      let result = parseEnv({ ...baseValidEnv, AI_FALLBACK_ENABLED: 'false' });
      expect(result.success).toBe(true);
      expect(result.data?.AI_FALLBACK_ENABLED).toBe(false);

      result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.AI_FALLBACK_ENABLED).toBe(true);
    });

    it('should transform ENABLE_HEALTH_CHECKS to boolean', () => {
      let result = parseEnv({ ...baseValidEnv, ENABLE_HEALTH_CHECKS: 'false' });
      expect(result.success).toBe(true);
      expect(result.data?.ENABLE_HEALTH_CHECKS).toBe(false);

      result = parseEnv({ ...baseValidEnv, ENABLE_HEALTH_CHECKS: '0' });
      expect(result.success).toBe(true);
      expect(result.data?.ENABLE_HEALTH_CHECKS).toBe(false);
    });

    it('should default ENABLE_HEALTH_CHECKS to true', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.ENABLE_HEALTH_CHECKS).toBe(true);
    });
  });

  describe('Rate limiting defaults', () => {
    it('should set default rate limit values', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.RATE_LIMIT_API_POINTS).toBe(50);
      expect(result.data?.RATE_LIMIT_API_DURATION).toBe(60);
      expect(result.data?.RATE_LIMIT_AUTH_LOGIN_POINTS).toBe(5);
      expect(result.data?.RATE_LIMIT_AUTH_REGISTER_POINTS).toBe(3);
    });

    it('should allow overriding rate limit values', () => {
      const result = parseEnv({
        ...baseValidEnv,
        RATE_LIMIT_API_POINTS: '100',
        RATE_LIMIT_AUTH_LOGIN_POINTS: '10',
      });
      expect(result.success).toBe(true);
      expect(result.data?.RATE_LIMIT_API_POINTS).toBe(100);
      expect(result.data?.RATE_LIMIT_AUTH_LOGIN_POINTS).toBe(10);
    });
  });

  describe('AI service configuration', () => {
    it('should default AI_SERVICE_REQUEST_TIMEOUT_MS to 5000', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.AI_SERVICE_REQUEST_TIMEOUT_MS).toBe(5000);
    });

    it('should default AI_MAX_CONCURRENT_REQUESTS to 16', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.AI_MAX_CONCURRENT_REQUESTS).toBe(16);
    });

    it('should accept valid AI service URL', () => {
      const result = parseEnv({
        ...baseValidEnv,
        AI_SERVICE_URL: 'http://ai-service:8001',
      });
      expect(result.success).toBe(true);
      expect(result.data?.AI_SERVICE_URL).toBe('http://ai-service:8001');
    });
  });

  describe('Orchestrator configuration', () => {
    it('should set orchestrator defaults', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.ORCHESTRATOR_ADAPTER_ENABLED).toBe(true);
      expect(result.data?.ORCHESTRATOR_ROLLOUT_PERCENTAGE).toBe(100);
      expect(result.data?.ORCHESTRATOR_SHADOW_MODE_ENABLED).toBe(false);
      expect(result.data?.ORCHESTRATOR_CIRCUIT_BREAKER_ENABLED).toBe(true);
      expect(result.data?.ORCHESTRATOR_ERROR_THRESHOLD_PERCENT).toBe(5);
      expect(result.data?.ORCHESTRATOR_ERROR_WINDOW_SECONDS).toBe(300);
      expect(result.data?.ORCHESTRATOR_LATENCY_THRESHOLD_MS).toBe(500);
    });

    it('should respect orchestrator adapter and shadow mode flags', () => {
      const result = parseEnv({
        ...baseValidEnv,
        ORCHESTRATOR_ADAPTER_ENABLED: '0',
        ORCHESTRATOR_SHADOW_MODE_ENABLED: '1',
      });
      expect(result.success).toBe(true);
      expect(result.data?.ORCHESTRATOR_ADAPTER_ENABLED).toBe(false);
      expect(result.data?.ORCHESTRATOR_SHADOW_MODE_ENABLED).toBe(true);
    });
  });

  describe('Game configuration defaults', () => {
    it('should set default game configuration values', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.MAX_CONCURRENT_GAMES).toBe(1000);
      expect(result.data?.GAME_TIMEOUT_MINUTES).toBe(60);
      expect(result.data?.AI_THINK_TIME_MS).toBe(2000);
      expect(result.data?.MAX_SPECTATORS_PER_GAME).toBe(50);
    });
  });

  describe('Auth lockout configuration', () => {
    it('should set default auth lockout values', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.AUTH_MAX_FAILED_LOGIN_ATTEMPTS).toBe(10);
      expect(result.data?.AUTH_FAILED_LOGIN_WINDOW_SECONDS).toBe(900);
      expect(result.data?.AUTH_LOCKOUT_DURATION_SECONDS).toBe(900);
    });
  });

  describe('RINGRIFT_APP_TOPOLOGY validation', () => {
    it('should accept valid topology values', () => {
      const validTopologies = ['single', 'multi-unsafe', 'multi-sticky'];
      for (const topology of validTopologies) {
        const result = parseEnv({ ...baseValidEnv, RINGRIFT_APP_TOPOLOGY: topology });
        expect(result.success).toBe(true);
        expect(result.data?.RINGRIFT_APP_TOPOLOGY).toBe(topology);
      }
    });

    it('should default to single when not set', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.RINGRIFT_APP_TOPOLOGY).toBe('single');
    });

    it('should reject invalid topology values', () => {
      const result = parseEnv({ ...baseValidEnv, RINGRIFT_APP_TOPOLOGY: 'clustered' });
      expect(result.success).toBe(false);
    });
  });

  describe('RINGRIFT_RULES_MODE validation', () => {
    it('should accept valid rules mode values', () => {
      const validModes = ['ts', 'python', 'shadow'];
      for (const mode of validModes) {
        const result = parseEnv({ ...baseValidEnv, RINGRIFT_RULES_MODE: mode });
        expect(result.success).toBe(true);
        expect(result.data?.RINGRIFT_RULES_MODE).toBe(mode);
      }
    });

    it('should accept undefined rules mode', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.RINGRIFT_RULES_MODE).toBeUndefined();
    });

    it('should reject invalid rules mode', () => {
      const result = parseEnv({ ...baseValidEnv, RINGRIFT_RULES_MODE: 'hybrid' });
      expect(result.success).toBe(false);
    });
  });

  describe('BCRYPT_ROUNDS validation', () => {
    it('should default to 12', () => {
      const result = parseEnv(baseValidEnv);
      expect(result.success).toBe(true);
      expect(result.data?.BCRYPT_ROUNDS).toBe(12);
    });

    it('should accept valid bcrypt rounds (4-31)', () => {
      const result = parseEnv({ ...baseValidEnv, BCRYPT_ROUNDS: '14' });
      expect(result.success).toBe(true);
      expect(result.data?.BCRYPT_ROUNDS).toBe(14);
    });

    it('should reject bcrypt rounds below 4', () => {
      const result = parseEnv({ ...baseValidEnv, BCRYPT_ROUNDS: '3' });
      expect(result.success).toBe(false);
    });

    it('should reject bcrypt rounds above 31', () => {
      const result = parseEnv({ ...baseValidEnv, BCRYPT_ROUNDS: '32' });
      expect(result.success).toBe(false);
    });
  });

  describe('parseEnv error reporting', () => {
    it('should return detailed errors for multiple invalid fields', () => {
      const result = parseEnv({
        NODE_ENV: 'invalid',
        PORT: '-1',
        LOG_LEVEL: 'verbose',
      });
      expect(result.success).toBe(false);
      expect(result.errors).toBeDefined();
      expect(result.errors!.length).toBeGreaterThanOrEqual(2);
    });

    it('should include path in error messages', () => {
      const result = parseEnv({ ...baseValidEnv, NODE_ENV: 'invalid' });
      expect(result.success).toBe(false);
      expect(result.errors?.some((e) => e.path.includes('NODE_ENV'))).toBe(true);
    });
  });

  describe('Environment preset profiles', () => {
    const withBaseEnv = (overrides: Record<string, string>): Record<string, string> => ({
      ...baseValidEnv,
      ...overrides,
    });

    it('CI orchestrator profile matches orchestrator-ON defaults', () => {
      const env = withBaseEnv({
        NODE_ENV: 'test',
        RINGRIFT_RULES_MODE: 'ts',
        ORCHESTRATOR_ADAPTER_ENABLED: 'true',
        ORCHESTRATOR_ROLLOUT_PERCENTAGE: '100',
        ORCHESTRATOR_SHADOW_MODE_ENABLED: 'false',
      });

      const result = parseEnv(env);
      expect(result.success).toBe(true);

      const data = result.data!;
      expect(data.NODE_ENV).toBe('test');
      expect(data.RINGRIFT_RULES_MODE).toBe('ts');
      expect(data.ORCHESTRATOR_ADAPTER_ENABLED).toBe(true);
      expect(data.ORCHESTRATOR_ROLLOUT_PERCENTAGE).toBe(100);
      expect(data.ORCHESTRATOR_SHADOW_MODE_ENABLED).toBe(false);
    });

    it('staging Phase 1 (orchestrator-only) profile is valid', () => {
      const env = withBaseEnv({
        NODE_ENV: 'staging',
        RINGRIFT_APP_TOPOLOGY: 'multi-sticky',
        RINGRIFT_RULES_MODE: 'ts',
        ORCHESTRATOR_ADAPTER_ENABLED: 'true',
        ORCHESTRATOR_ROLLOUT_PERCENTAGE: '100',
        ORCHESTRATOR_SHADOW_MODE_ENABLED: 'false',
      });

      const result = parseEnv(env);
      expect(result.success).toBe(true);

      const data = result.data!;
      expect(data.NODE_ENV).toBe('staging');
      expect(data.RINGRIFT_APP_TOPOLOGY).toBe('multi-sticky');
      expect(data.RINGRIFT_RULES_MODE).toBe('ts');
      expect(data.ORCHESTRATOR_ADAPTER_ENABLED).toBe(true);
      expect(data.ORCHESTRATOR_ROLLOUT_PERCENTAGE).toBe(100);
      expect(data.ORCHESTRATOR_SHADOW_MODE_ENABLED).toBe(false);
    });

    it('production Phase 2 (legacy authoritative + shadow) profile is valid', () => {
      const env = withBaseEnv({
        NODE_ENV: 'production',
        RINGRIFT_APP_TOPOLOGY: 'multi-sticky',
        RINGRIFT_RULES_MODE: 'shadow',
        ORCHESTRATOR_ADAPTER_ENABLED: 'true',
        ORCHESTRATOR_ROLLOUT_PERCENTAGE: '0',
        ORCHESTRATOR_SHADOW_MODE_ENABLED: 'true',
      });

      const result = parseEnv(env);
      expect(result.success).toBe(true);

      const data = result.data!;
      expect(data.NODE_ENV).toBe('production');
      expect(data.RINGRIFT_APP_TOPOLOGY).toBe('multi-sticky');
      expect(data.RINGRIFT_RULES_MODE).toBe('shadow');
      expect(data.ORCHESTRATOR_ADAPTER_ENABLED).toBe(true);
      expect(data.ORCHESTRATOR_ROLLOUT_PERCENTAGE).toBe(0);
      expect(data.ORCHESTRATOR_SHADOW_MODE_ENABLED).toBe(true);
    });

    it('production Phase 3 (incremental rollout) profile accepts 1–99% rollout', () => {
      const env = withBaseEnv({
        NODE_ENV: 'production',
        RINGRIFT_APP_TOPOLOGY: 'multi-sticky',
        RINGRIFT_RULES_MODE: 'ts',
        ORCHESTRATOR_ADAPTER_ENABLED: 'true',
        ORCHESTRATOR_ROLLOUT_PERCENTAGE: '25',
        ORCHESTRATOR_SHADOW_MODE_ENABLED: 'false',
      });

      const result = parseEnv(env);
      expect(result.success).toBe(true);

      const data = result.data!;
      expect(data.NODE_ENV).toBe('production');
      expect(data.RINGRIFT_APP_TOPOLOGY).toBe('multi-sticky');
      expect(data.RINGRIFT_RULES_MODE).toBe('ts');
      expect(data.ORCHESTRATOR_ADAPTER_ENABLED).toBe(true);
      expect(data.ORCHESTRATOR_ROLLOUT_PERCENTAGE).toBe(25);
      expect(data.ORCHESTRATOR_SHADOW_MODE_ENABLED).toBe(false);
    });
  });
});

describe('Environment helper functions', () => {
  describe('getEffectiveNodeEnv', () => {
    // Note: This test works correctly in Jest because JEST_WORKER_ID is set
    it('should return test when running in Jest', () => {
      const rawEnv = { NODE_ENV: 'development' } as RawEnv;
      // In Jest environment, this should return 'test' regardless of NODE_ENV
      expect(getEffectiveNodeEnv(rawEnv)).toBe('test');
    });
  });

  describe('isProduction', () => {
    it('should return true for production', () => {
      expect(isProduction('production')).toBe(true);
    });

    it('should return false for non-production', () => {
      expect(isProduction('development')).toBe(false);
      expect(isProduction('staging')).toBe(false);
      expect(isProduction('test')).toBe(false);
    });
  });

  describe('isStaging', () => {
    it('should return true for staging', () => {
      expect(isStaging('staging')).toBe(true);
    });

    it('should return false for non-staging', () => {
      expect(isStaging('production')).toBe(false);
      expect(isStaging('development')).toBe(false);
      expect(isStaging('test')).toBe(false);
    });
  });

  describe('isDevelopment', () => {
    it('should return true for development', () => {
      expect(isDevelopment('development')).toBe(true);
    });

    it('should return false for non-development', () => {
      expect(isDevelopment('production')).toBe(false);
      expect(isDevelopment('staging')).toBe(false);
      expect(isDevelopment('test')).toBe(false);
    });
  });

  describe('isTest', () => {
    it('should return true for test', () => {
      expect(isTest('test')).toBe(true);
    });

    it('should return false for non-test', () => {
      expect(isTest('production')).toBe(false);
      expect(isTest('staging')).toBe(false);
      expect(isTest('development')).toBe(false);
    });
  });

  describe('isProductionLike', () => {
    it('should return true for production', () => {
      expect(isProductionLike('production')).toBe(true);
    });

    it('should return true for staging', () => {
      expect(isProductionLike('staging')).toBe(true);
    });

    it('should return false for development and test', () => {
      expect(isProductionLike('development')).toBe(false);
      expect(isProductionLike('test')).toBe(false);
    });
  });
});

describe('Schema type coercion', () => {
  it('should coerce numeric strings to numbers', () => {
    const result = parseEnv({
      ...baseValidEnv,
      PORT: '8080',
      DATABASE_POOL_MIN: '5',
      DATABASE_POOL_MAX: '20',
      AI_SERVICE_REQUEST_TIMEOUT_MS: '10000',
      RATE_LIMIT_API_POINTS: '100',
    });
    expect(result.success).toBe(true);
    expect(typeof result.data?.PORT).toBe('number');
    expect(typeof result.data?.DATABASE_POOL_MIN).toBe('number');
    expect(typeof result.data?.DATABASE_POOL_MAX).toBe('number');
    expect(typeof result.data?.AI_SERVICE_REQUEST_TIMEOUT_MS).toBe('number');
    expect(typeof result.data?.RATE_LIMIT_API_POINTS).toBe('number');
  });
});

describe('Optional fields', () => {
  it('should allow optional fields to be undefined', () => {
    const result = parseEnv(baseValidEnv);
    expect(result.success).toBe(true);
    expect(result.data?.DATABASE_URL).toBeUndefined();
    expect(result.data?.REDIS_URL).toBeUndefined();
    expect(result.data?.JWT_SECRET).toBeUndefined();
    expect(result.data?.JWT_REFRESH_SECRET).toBeUndefined();
    expect(result.data?.AI_SERVICE_URL).toBeUndefined();
    expect(result.data?.SMTP_HOST).toBeUndefined();
  });

  it('should accept optional fields when provided', () => {
    const result = parseEnv({
      ...baseValidEnv,
      DATABASE_URL: 'postgresql://user:pass@localhost:5432/db',
      REDIS_URL: 'redis://localhost:6379',
      JWT_SECRET: 'a-secret-key-that-is-at-least-32-chars',
      AI_SERVICE_URL: 'http://localhost:8001',
      SMTP_HOST: 'smtp.example.com',
      SMTP_PORT: '587',
    });
    expect(result.success).toBe(true);
    expect(result.data?.DATABASE_URL).toBe('postgresql://user:pass@localhost:5432/db');
    expect(result.data?.REDIS_URL).toBe('redis://localhost:6379');
    expect(result.data?.JWT_SECRET).toBe('a-secret-key-that-is-at-least-32-chars');
    expect(result.data?.AI_SERVICE_URL).toBe('http://localhost:8001');
    expect(result.data?.SMTP_HOST).toBe('smtp.example.com');
    expect(result.data?.SMTP_PORT).toBe(587);
  });
});
