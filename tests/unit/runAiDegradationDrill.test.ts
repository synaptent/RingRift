/**
 * AI Service Degradation Drill Harness Tests
 *
 * Focus on orchestration and CLI parsing only:
 * - Backend HTTP health check wiring
 * - AI service health integration (via injectable helper)
 * - Optional AI fallback behaviour placeholder
 * - JSON report generation under results/ops
 * - CLI arg parsing behaviour
 */

import fs from 'fs';

import * as DrillHarness from '../../scripts/run-ai-degradation-drill';
import { parseArgs } from '../../scripts/run-ai-degradation-drill';

// Jest module mocks ----------------------------------------------------------------

jest.mock('fs', () => {
  const actual = jest.requireActual('fs') as typeof import('fs');
  return {
    ...actual,
    mkdirSync: jest.fn(),
    writeFileSync: jest.fn(),
  };
});

describe('runAiDegradationDrill (orchestration)', () => {
  const mkdirMock = fs.mkdirSync as jest.MockedFunction<typeof fs.mkdirSync>;
  const writeFileMock = fs.writeFileSync as jest.MockedFunction<typeof fs.writeFileSync>;

  beforeEach(() => {
    delete process.env.APP_BASE;
    delete process.env.BASE_URL;
    jest.clearAllMocks();
  });

  it('runs all checks and writes a passing report for baseline when everything is healthy', async () => {
    const httpHealthCheck = jest.fn(async (_url: string) => ({ statusCode: 200 }));
    const aiHealthCheck = jest.fn(async (_opts: any) => ({
      ok: true,
      details: {
        source: 'test',
        status: 'healthy',
      },
    }));
    const aiFallbackCheck = jest.fn(async (_phase: 'baseline' | 'degraded' | 'recovery') => ({
      ok: true,
      details: {
        skipped: true,
        reason: 'placeholder in tests',
      },
    }));

    const { report } = await DrillHarness.runAiDegradationDrill({
      env: 'staging',
      operator: 'alice',
      phase: 'baseline',
      // Testing-only injectable hooks
      httpHealthCheck,
      aiHealthCheck,
      aiFallbackCheck,
    } as any);

    // Overall result
    expect(report.environment).toBe('staging');
    expect(report.operator).toBe('alice');
    expect(report.phase).toBe('baseline');
    expect(report.overallPass).toBe(true);
    expect(report.checks).toHaveLength(3);

    const names = report.checks.map((c) => c.name).sort();
    expect(names).toEqual(['ai_fallback_behaviour', 'ai_service_health', 'backend_http_health']);

    const backendCheck = report.checks.find((c) => c.name === 'backend_http_health');
    expect(backendCheck).toBeDefined();
    expect(backendCheck?.status).toBe('pass');
    expect(backendCheck?.details).toEqual({
      url: 'http://localhost:3000/health',
      statusCode: 200,
    });

    const aiHealth = report.checks.find((c) => c.name === 'ai_service_health');
    expect(aiHealth).toBeDefined();
    expect(aiHealth?.status).toBe('pass');
    const aiDetails = aiHealth?.details as {
      source?: string;
      status?: string;
      expectedStatus?: string;
      phase?: string;
    };
    expect(aiDetails.source).toBe('test');
    expect(aiDetails.status).toBe('healthy');
    expect(aiDetails.expectedStatus).toBe('healthy');
    expect(aiDetails.phase).toBe('baseline');

    const fallbackCheck = report.checks.find((c) => c.name === 'ai_fallback_behaviour');
    expect(fallbackCheck).toBeDefined();
    expect(fallbackCheck?.status).toBe('pass');
    const fallbackDetails = fallbackCheck?.details as { skipped?: boolean; reason?: string };
    expect(fallbackDetails.skipped).toBe(true);

    // Filesystem writes
    expect(mkdirMock).toHaveBeenCalledTimes(1);
    expect(mkdirMock).toHaveBeenCalledWith(expect.stringContaining('results/ops'), {
      recursive: true,
    });

    expect(writeFileMock).toHaveBeenCalledTimes(1);
    const [outputPathArg, jsonContent] = writeFileMock.mock.calls[0];
    expect(String(outputPathArg)).toContain('results/ops/ai_degradation.staging.baseline.');
    const parsed = JSON.parse(String(jsonContent));
    expect(parsed.drillType).toBe('ai_service_degradation');
    expect(parsed.environment).toBe('staging');
    expect(parsed.phase).toBe('baseline');
    expect(parsed.overallPass).toBe(true);
    expect(Array.isArray(parsed.checks)).toBe(true);

    // HTTP and AI health helpers invoked
    expect(httpHealthCheck).toHaveBeenCalledTimes(1);
    const [healthUrl] = httpHealthCheck.mock.calls[0];
    expect(healthUrl).toBe('http://localhost:3000/health');

    expect(aiHealthCheck).toHaveBeenCalledTimes(1);
  });

  it('records AI service as failing in degraded phase and marks overallPass=false', async () => {
    const httpHealthCheck = jest.fn(async (_url: string) => ({ statusCode: 200 }));
    const aiHealthCheck = jest.fn(async (_opts: any) => ({
      ok: false,
      details: {
        source: 'test',
        status: 'unhealthy',
        reason: 'simulated outage',
      },
    }));
    const aiFallbackCheck = jest.fn(async (_phase: 'baseline' | 'degraded' | 'recovery') => ({
      ok: true,
      details: {
        simulated: true,
      },
    }));

    const { report } = await DrillHarness.runAiDegradationDrill({
      env: 'staging',
      phase: 'degraded',
      httpHealthCheck,
      aiHealthCheck,
      aiFallbackCheck,
    } as any);

    expect(report.environment).toBe('staging');
    expect(report.phase).toBe('degraded');
    expect(report.overallPass).toBe(false);

    const backendCheck = report.checks.find((c) => c.name === 'backend_http_health');
    expect(backendCheck?.status).toBe('pass');

    const aiHealth = report.checks.find((c) => c.name === 'ai_service_health');
    expect(aiHealth).toBeDefined();
    expect(aiHealth?.status).toBe('fail');
    const aiDetails = aiHealth?.details as {
      status?: string;
      reason?: string;
      expectedStatus?: string;
      phase?: string;
    };
    expect(aiDetails.status).toBe('unhealthy');
    expect(aiDetails.reason).toBe('simulated outage');
    // In degraded phase, we still record expectedStatus=unhealthy for operator context.
    expect(aiDetails.expectedStatus).toBe('unhealthy');
    expect(aiDetails.phase).toBe('degraded');

    const fallbackCheck = report.checks.find((c) => c.name === 'ai_fallback_behaviour');
    expect(fallbackCheck).toBeDefined();
    expect(fallbackCheck?.status).toBe('pass');
  });

  it('allows AI health to recover in recovery phase after degraded phase', async () => {
    const httpHealthCheck = jest.fn(async (_url: string) => ({ statusCode: 200 }));

    const aiHealthCheckDegraded = jest.fn(async (_opts: any) => ({
      ok: false,
      details: { status: 'unhealthy' },
    }));
    const aiHealthCheckRecovery = jest.fn(async (_opts: any) => ({
      ok: true,
      details: { status: 'healthy' },
    }));
    const aiFallbackCheck = jest.fn(async (_phase: 'baseline' | 'degraded' | 'recovery') => ({
      ok: true,
      details: {
        simulated: true,
      },
    }));

    // First run: degraded
    const degradedResult = await DrillHarness.runAiDegradationDrill({
      env: 'staging',
      phase: 'degraded',
      httpHealthCheck,
      aiHealthCheck: aiHealthCheckDegraded,
      aiFallbackCheck,
    } as any);

    expect(degradedResult.report.overallPass).toBe(false);
    const degradedAi = degradedResult.report.checks.find((c) => c.name === 'ai_service_health');
    expect(degradedAi?.status).toBe('fail');

    // Second run: recovery
    const recoveryResult = await DrillHarness.runAiDegradationDrill({
      env: 'staging',
      phase: 'recovery',
      httpHealthCheck,
      aiHealthCheck: aiHealthCheckRecovery,
      aiFallbackCheck,
    } as any);

    expect(recoveryResult.report.phase).toBe('recovery');
    const recoveryAi = recoveryResult.report.checks.find((c) => c.name === 'ai_service_health');
    expect(recoveryAi).toBeDefined();
    expect(recoveryAi?.status).toBe('pass');
    const recoveryDetails = recoveryAi?.details as { expectedStatus?: string; phase?: string };
    expect(recoveryDetails.expectedStatus).toBe('healthy');
    expect(recoveryDetails.phase).toBe('recovery');
  });

  it('uses BASE_URL env when computing backend health URL by default', async () => {
    const httpHealthCheck = jest.fn(async (_url: string) => ({ statusCode: 200 }));
    const aiHealthCheck = jest.fn(async (_opts: any) => ({
      ok: true,
      details: { status: 'healthy' },
    }));
    const aiFallbackCheck = jest.fn(async (_phase: 'baseline' | 'degraded' | 'recovery') => ({
      ok: true,
      details: { simulated: true },
    }));

    const previousBaseUrl = process.env.BASE_URL;
    process.env.BASE_URL = 'http://staging.example.com';

    try {
      const { report } = await DrillHarness.runAiDegradationDrill({
        env: 'staging',
        phase: 'baseline',
        httpHealthCheck,
        aiHealthCheck,
        aiFallbackCheck,
      } as any);

      expect(report.overallPass).toBe(true);
      expect(httpHealthCheck).toHaveBeenCalledTimes(1);
      const [healthUrl] = httpHealthCheck.mock.calls[0];
      expect(healthUrl).toBe('http://staging.example.com/health');
    } finally {
      process.env.BASE_URL = previousBaseUrl;
    }
  });
});

describe('parseArgs (CLI parsing)', () => {
  it('parses required --env and --phase plus optional flags', () => {
    const parsed = parseArgs([
      'node',
      'script',
      '--env',
      'staging',
      '--phase',
      'baseline',
      '--operator',
      'alice',
      '--output',
      'results/ops/custom.json',
      '--baseUrl',
      'http://example.com',
      '--aiServiceUrl',
      'http://ai.example.com',
    ]);

    expect(parsed.env).toBe('staging');
    expect(parsed.phase).toBe('baseline');
    expect(parsed.operator).toBe('alice');
    expect(parsed.output).toBe('results/ops/custom.json');
    expect(parsed.baseUrl).toBe('http://example.com');
    expect(parsed.aiServiceUrl).toBe('http://ai.example.com');
  });

  it('supports kebab-case aliases for base and AI service URLs', () => {
    const parsed = parseArgs([
      'node',
      'script',
      '--env=production',
      '--phase=recovery',
      '--base-url=https://prod.example.com',
      '--ai-service-url=https://ai.prod.example.com',
    ]);

    expect(parsed.env).toBe('production');
    expect(parsed.phase).toBe('recovery');
    expect(parsed.baseUrl).toBe('https://prod.example.com');
    expect(parsed.aiServiceUrl).toBe('https://ai.prod.example.com');
  });

  it('throws when --env is missing', () => {
    expect(() => parseArgs(['node', 'script', '--phase', 'baseline'])).toThrowError(
      /Missing required --env <env> argument/
    );
  });

  it('throws when --phase is missing', () => {
    expect(() => parseArgs(['node', 'script', '--env', 'staging'])).toThrowError(
      /Missing required --phase <baseline|degraded|recovery> argument/
    );
  });

  it('throws when --phase is invalid', () => {
    expect(() =>
      parseArgs(['node', 'script', '--env', 'staging', '--phase', 'unknown'])
    ).toThrowError(/Invalid --phase; expected baseline|degraded|recovery/);
  });
});
