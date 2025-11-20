import {
  readEnv,
  flagEnabled,
  isSandboxAiStallDiagnosticsEnabled,
  isSandboxCaptureDebugEnabled,
  isSandboxAiCaptureDebugEnabled,
  isSandboxAiTraceModeEnabled,
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
});
