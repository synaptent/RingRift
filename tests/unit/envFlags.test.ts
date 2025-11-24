import {
  readEnv,
  flagEnabled,
  isSandboxAiStallDiagnosticsEnabled,
  isSandboxCaptureDebugEnabled,
  isSandboxAiCaptureDebugEnabled,
  isSandboxAiTraceModeEnabled,
  isSandboxAiParityModeEnabled,
  isLocalAIHeuristicModeEnabled,
} from '../../src/shared/utils/envFlags';

describe('envFlags helpers', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('readEnv reads from process.env when present', () => {
    delete (process.env as any).RINGRIFT_TEST_FLAG;
    expect(readEnv('RINGRIFT_TEST_FLAG')).toBeUndefined();

    (process.env as any).RINGRIFT_TEST_FLAG = 'abc';
    expect(readEnv('RINGRIFT_TEST_FLAG')).toBe('abc');
  });

  it('flagEnabled returns true only for "1", "true", or "TRUE"', () => {
    (process.env as any).RINGRIFT_FLAG = '1';
    expect(flagEnabled('RINGRIFT_FLAG')).toBe(true);

    (process.env as any).RINGRIFT_FLAG = 'true';
    expect(flagEnabled('RINGRIFT_FLAG')).toBe(true);

    (process.env as any).RINGRIFT_FLAG = 'TRUE';
    expect(flagEnabled('RINGRIFT_FLAG')).toBe(true);

    (process.env as any).RINGRIFT_FLAG = '0';
    expect(flagEnabled('RINGRIFT_FLAG')).toBe(false);

    (process.env as any).RINGRIFT_FLAG = 'false';
    expect(flagEnabled('RINGRIFT_FLAG')).toBe(false);

    delete (process.env as any).RINGRIFT_FLAG;
    expect(flagEnabled('RINGRIFT_FLAG')).toBe(false);
  });

  it('isSandboxAiStallDiagnosticsEnabled proxies the correct env name', () => {
    (process.env as any).RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS = '1';
    expect(isSandboxAiStallDiagnosticsEnabled()).toBe(true);

    (process.env as any).RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS = '0';
    expect(isSandboxAiStallDiagnosticsEnabled()).toBe(false);
  });

  it('isSandboxCaptureDebugEnabled proxies the correct env name', () => {
    (process.env as any).RINGRIFT_SANDBOX_CAPTURE_DEBUG = '1';
    expect(isSandboxCaptureDebugEnabled()).toBe(true);

    (process.env as any).RINGRIFT_SANDBOX_CAPTURE_DEBUG = '0';
    expect(isSandboxCaptureDebugEnabled()).toBe(false);
  });

  it('isSandboxAiCaptureDebugEnabled proxies the correct env name', () => {
    (process.env as any).RINGRIFT_SANDBOX_AI_CAPTURE_DEBUG = '1';
    expect(isSandboxAiCaptureDebugEnabled()).toBe(true);

    (process.env as any).RINGRIFT_SANDBOX_AI_CAPTURE_DEBUG = '0';
    expect(isSandboxAiCaptureDebugEnabled()).toBe(false);
  });

  it('isSandboxAiTraceModeEnabled proxies the correct env name', () => {
    (process.env as any).RINGRIFT_SANDBOX_AI_TRACE_MODE = '1';
    expect(isSandboxAiTraceModeEnabled()).toBe(true);

    (process.env as any).RINGRIFT_SANDBOX_AI_TRACE_MODE = '0';
    expect(isSandboxAiTraceModeEnabled()).toBe(false);
  });

  it('isSandboxAiParityModeEnabled proxies the correct env name', () => {
    (process.env as any).RINGRIFT_SANDBOX_AI_PARITY_MODE = '1';
    expect(isSandboxAiParityModeEnabled()).toBe(true);

    (process.env as any).RINGRIFT_SANDBOX_AI_PARITY_MODE = '0';
    expect(isSandboxAiParityModeEnabled()).toBe(false);

    delete (process.env as any).RINGRIFT_SANDBOX_AI_PARITY_MODE;
    expect(isSandboxAiParityModeEnabled()).toBe(false);
  });

  it('isLocalAIHeuristicModeEnabled proxies the correct env name', () => {
    (process.env as any).RINGRIFT_LOCAL_AI_HEURISTIC_MODE = '1';
    expect(isLocalAIHeuristicModeEnabled()).toBe(true);

    (process.env as any).RINGRIFT_LOCAL_AI_HEURISTIC_MODE = '0';
    expect(isLocalAIHeuristicModeEnabled()).toBe(false);

    delete (process.env as any).RINGRIFT_LOCAL_AI_HEURISTIC_MODE;
    expect(isLocalAIHeuristicModeEnabled()).toBe(false);
  });

  //
  // Server config / JWT secret validation
  //
  // These tests intentionally live alongside the envFlags helpers to keep
  // environment-related behaviour covered in a single place.
  //

  it('allows placeholder JWT secrets in development', async () => {
    process.env = {
      ...process.env,
      NODE_ENV: 'development',
      JWT_SECRET: 'your-super-secret-jwt-key-change-this-in-production',
      JWT_REFRESH_SECRET: 'your-super-secret-refresh-key-change-this-in-production',
    } as any;

    jest.resetModules();
    const { config } = await import('../../src/server/config');

    expect(config.nodeEnv).toBe('development');
    expect(config.auth.jwtSecret).toBe('your-super-secret-jwt-key-change-this-in-production');
    expect(config.auth.jwtRefreshSecret).toBe(
      'your-super-secret-refresh-key-change-this-in-production'
    );
  });

  it('accepts strong non-placeholder JWT secrets in production', async () => {
    process.env = {
      ...process.env,
      NODE_ENV: 'production',
      DATABASE_URL: 'postgresql://user:pass@localhost:5432/testdb',
      REDIS_URL: 'redis://localhost:6379',
      JWT_SECRET: 'strong-access-secret-123',
      JWT_REFRESH_SECRET: 'strong-refresh-secret-123',
    } as any;

    jest.resetModules();
    const { config } = await import('../../src/server/config');

    expect(config.nodeEnv).toBe('production');
    expect(config.auth.jwtSecret).toBe('strong-access-secret-123');
    expect(config.auth.jwtRefreshSecret).toBe('strong-refresh-secret-123');
  });

  it('rejects missing JWT secrets in production', async () => {
    process.env = {
      ...process.env,
      NODE_ENV: 'production',
      DATABASE_URL: 'postgresql://user:pass@localhost:5432/testdb',
      REDIS_URL: 'redis://localhost:6379',
    } as any;

    delete (process.env as any).JWT_SECRET;
    delete (process.env as any).JWT_REFRESH_SECRET;

    jest.resetModules();
    await expect(import('../../src/server/config')).rejects.toThrow(
      /Invalid JWT configuration for NODE_ENV=production/
    );
  });

  it('rejects placeholder JWT secrets in production', async () => {
    process.env = {
      ...process.env,
      NODE_ENV: 'production',
      DATABASE_URL: 'postgresql://user:pass@localhost:5432/testdb',
      REDIS_URL: 'redis://localhost:6379',
      JWT_SECRET: 'change-this-secret',
      JWT_REFRESH_SECRET: 'change-this-refresh-secret',
    } as any;

    jest.resetModules();
    await expect(import('../../src/server/config')).rejects.toThrow(
      /Invalid JWT configuration for NODE_ENV=production/
    );
  });

  it('defaults app topology to "single" when unset', async () => {
    process.env = {
      ...process.env,
      NODE_ENV: 'development',
    } as any;

    delete (process.env as any).RINGRIFT_APP_TOPOLOGY;

    jest.resetModules();
    const { config } = await import('../../src/server/config');

    expect(config.app.topology).toBe('single');
  });

  it('parses RINGRIFT_APP_TOPOLOGY=multi-unsafe', async () => {
    process.env = {
      ...process.env,
      NODE_ENV: 'development',
      RINGRIFT_APP_TOPOLOGY: 'multi-unsafe',
    } as any;

    jest.resetModules();
    const { config } = await import('../../src/server/config');

    expect(config.app.topology).toBe('multi-unsafe');
  });

  it('parses RINGRIFT_APP_TOPOLOGY=multi-sticky', async () => {
    process.env = {
      ...process.env,
      NODE_ENV: 'development',
      RINGRIFT_APP_TOPOLOGY: 'multi-sticky',
    } as any;

    jest.resetModules();
    const { config } = await import('../../src/server/config');

    expect(config.app.topology).toBe('multi-sticky');
  });
});
