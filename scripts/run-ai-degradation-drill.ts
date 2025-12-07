#!/usr/bin/env ts-node
/* eslint-disable no-console */
/**
 * AI Service Degradation Drill Harness
 * -----------------------------------------------------------------------------
 *
 * Verification / reporting harness for the AI service degradation drill described
 * in docs/runbooks/AI_SERVICE_DEGRADATION_DRILL.md.
 *
 * Responsibilities:
 * - Run a small set of checks against the backend and AI service wiring:
 *   - Backend HTTP /health
 *   - AI service health (via HealthCheckService readiness)
 *   - Optional AI fallback behaviour placeholder
 * - Produce a single JSON report under results/ops/ with:
 *   - drillType: 'ai_service_degradation'
 *   - environment, operator, phase
 *   - per-check results
 *   - overallPass (all checks pass)
 *
 * This script does NOT induce degradation itself. Operators are expected to
 * follow the runbook to start/stop containers or otherwise degrade the AI
 * service, then run this harness at each phase (baseline, degraded, recovery).
 */

import fs from 'fs';
import path from 'path';
import http from 'http';
import https from 'https';
import { URL } from 'url';

import { getReadinessStatus } from '../src/server/services/HealthCheckService';

export type DrillStatus = 'pass' | 'fail';

export interface DrillCheck {
  name: string;
  status: DrillStatus;
  details?: unknown;
}

export interface AiDegradationDrillReport {
  drillType: 'ai_service_degradation';
  environment: string;
  operator?: string;
  runTimestamp: string; // ISO UTC
  phase: Phase;
  checks: DrillCheck[];
  overallPass: boolean;
}

export interface AiDegradationDrillOptions {
  env: string;
  operator?: string;
  phase: Phase;
  outputPath?: string;
  baseUrl?: string; // backend URL for HTTP health checks
  aiServiceUrl?: string; // optional explicit AI service URL, fallback to config/ENV
}

type Phase = 'baseline' | 'degraded' | 'recovery';

interface ParsedCliArgs {
  env: string;
  operator?: string;
  phase: Phase;
  output?: string;
  baseUrl?: string;
  aiServiceUrl?: string;
}

/**
 * Format a UTC timestamp suitable for filenames, e.g. 20251205T101200Z.
 */
export function formatTimestampForFilename(date: Date): string {
  const iso = date.toISOString(); // e.g. 2025-12-05T10:13:20.035Z
  // Strip "-", ":" and fractional seconds for filesystem-safe timestamp
  return iso.replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
}

/**
 * Truncate potentially large strings for inclusion in the JSON report.
 */
export function truncateOutput(output: string, maxLength = 2000): string {
  if (output.length <= maxLength) {
    return output;
  }
  const truncatedBytes = output.length - maxLength;
  return `${output.slice(0, maxLength)}\n...[truncated ${truncatedBytes} bytes]`;
}

/**
 * Minimal HTTP GET health check using Node's http/https modules.
 * Returns { statusCode } and never throws; on network errors or timeouts
 * statusCode will be null.
 */
export async function performHttpHealthCheck(
  url: string,
  timeoutMs = 5000
): Promise<{ statusCode: number | null }> {
  return await new Promise<{ statusCode: number | null }>((resolve) => {
    let resolved = false;
    try {
      const urlObj = new URL(url);
      const client = urlObj.protocol === 'https:' ? https : http;

      const req = client.request(urlObj, (res) => {
        if (resolved) {
          return;
        }
        resolved = true;
        const statusCode = res.statusCode ?? null;
        // Consume response data to free up memory / sockets
        res.resume();
        resolve({ statusCode });
      });

      req.on('error', () => {
        if (resolved) {
          return;
        }
        resolved = true;
        resolve({ statusCode: null });
      });

      req.setTimeout(timeoutMs, () => {
        if (resolved) {
          return;
        }
        resolved = true;
        req.destroy();
        resolve({ statusCode: null });
      });

      req.end();
    } catch {
      if (!resolved) {
        resolved = true;
        resolve({ statusCode: null });
      }
    }
  });
}

/**
 * AI service health check helper.
 *
 * Uses the server's existing readiness check (HealthCheckService.getReadinessStatus)
 * and inspects the aiService dependency entry. This ensures we are validating the
 * same wiring that powers /ready, metrics, and alerts.
 *
 * The options object is intentionally the full AiDegradationDrillOptions type so
 * callers (and tests) can pass additional fields without affecting behaviour.
 */
export async function checkAiServiceHealth(
  options: AiDegradationDrillOptions
): Promise<{ ok: boolean; details: unknown }> {
  try {
    const readiness = await getReadinessStatus({ includeAIService: true });
    const aiServiceCheck = readiness.checks?.aiService;
    const ok = aiServiceCheck?.status === 'healthy';

    const details: Record<string, unknown> = {
      overallStatus: readiness.status,
      timestamp: readiness.timestamp,
      aiService: aiServiceCheck
        ? {
            ...aiServiceCheck,
            error: aiServiceCheck.error ? truncateOutput(aiServiceCheck.error) : undefined,
          }
        : null,
    };

    if (options.aiServiceUrl) {
      details.aiServiceUrlOverride = options.aiServiceUrl;
    }

    return { ok, details };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      details: {
        error: truncateOutput(message),
        aiServiceUrlOverride: options.aiServiceUrl,
      },
    };
  }
}

/**
 * Optional / soft check for AI fallback behaviour.
 *
 * For this slice we implement a lightweight placeholder that documents its
 * limitations in the JSON report. Implementing a full end-to-end fallback
 * check (games + load harness + metrics) would require significantly more
 * wiring and is better handled by existing k6 scenarios and dashboards.
 */
export async function checkAiFallbackBehaviour(
  phase: Phase
): Promise<{ ok: boolean; details: unknown }> {
  if (phase === 'degraded') {
    return {
      ok: true,
      details: {
        skipped: true,
        reason:
          'ai_fallback_behaviour is a placeholder for this slice; rely on k6 load harness and Grafana/Prometheus dashboards to validate fallback metrics per the runbook.',
        phase,
      },
    };
  }

  return {
    ok: true,
    details: {
      skipped: true,
      reason: 'ai_fallback_behaviour is only meaningful during the degraded phase.',
      phase,
    },
  };
}

/**
 * Parse CLI arguments for the drill harness.
 *
 * Supported flags:
 *   --env <env>                     (required) environment name, e.g. staging
 *   --operator <id>                 (optional) operator identifier for audit/reporting
 *   --phase <baseline|degraded|recovery> (required) drill phase
 *   --output <path>                 (optional) explicit output path for the JSON report
 *   --baseUrl <url>                (optional) backend base URL (alias: --base-url)
 *   --aiServiceUrl <url>           (optional) AI service URL (alias: --ai-service-url)
 */
export function parseArgs(argv: string[]): ParsedCliArgs {
  const args: Record<string, string | boolean> = {};

  for (let i = 2; i < argv.length; i += 1) {
    const raw = argv[i];
    if (!raw.startsWith('--')) continue;

    const eqIndex = raw.indexOf('=');
    let key: string;
    let value: string | boolean;
    if (eqIndex !== -1) {
      key = raw.slice(2, eqIndex);
      value = raw.slice(eqIndex + 1);
    } else {
      key = raw.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith('--')) {
        value = next;
        i += 1;
      } else {
        value = true;
      }
    }

    args[key] = value;
  }

  const env = args.env as string | undefined;
  if (!env) {
    throw new Error('Missing required --env <env> argument');
  }

  const phaseRaw = args.phase as string | undefined;
  if (!phaseRaw) {
    throw new Error('Missing required --phase <baseline|degraded|recovery> argument');
  }

  const allowedPhases: Phase[] = ['baseline', 'degraded', 'recovery'];
  if (!allowedPhases.includes(phaseRaw as Phase)) {
    throw new Error('Invalid --phase; expected baseline|degraded|recovery');
  }
  const phase = phaseRaw as Phase;

  const operator = args.operator as string | undefined;
  const output = (args.output as string | undefined) ?? (args.o as string | undefined);
  const baseUrl = (args.baseUrl as string | undefined) ?? (args['base-url'] as string | undefined);
  const aiServiceUrl =
    (args.aiServiceUrl as string | undefined) ?? (args['ai-service-url'] as string | undefined);

  const result: ParsedCliArgs = { env, phase };

  if (operator !== undefined) {
    result.operator = operator;
  }
  if (output !== undefined) {
    result.output = output;
  }
  if (baseUrl !== undefined) {
    result.baseUrl = baseUrl;
  }
  if (aiServiceUrl !== undefined) {
    result.aiServiceUrl = aiServiceUrl;
  }

  return result;
}

/**
 * Core programmatic entrypoint for the AI degradation drill.
 * Orchestrates all checks and writes the JSON report.
 */
export async function runAiDegradationDrill(
  options: AiDegradationDrillOptions
): Promise<{ report: AiDegradationDrillReport; outputPath: string }> {
  const { env, operator, phase } = options;

  const checks: DrillCheck[] = [];

  // Check 1: Backend HTTP /health
  const baseUrl =
    options.baseUrl ??
    // Align with runbook examples which use APP_BASE for the app URL.
    process.env.APP_BASE ??
    process.env.BASE_URL ??
    'http://localhost:3000';
  const normalizedBaseUrl = baseUrl.replace(/\/$/, '');
  const healthUrl = `${normalizedBaseUrl}/health`;

  // Allow tests (and potential callers) to inject a custom health check implementation.
  const httpHealthCheckFn =
    (options as any).httpHealthCheck ?? performHttpHealthCheck.bind(null as never);

  const { statusCode } = await httpHealthCheckFn(healthUrl);
  checks.push({
    name: 'backend_http_health',
    status: statusCode === 200 ? 'pass' : 'fail',
    details: {
      url: healthUrl,
      statusCode,
    },
  });

  // Check 2: AI service health via HealthCheckService readiness.
  const aiHealthCheckFn =
    (options as any).aiHealthCheck ??
    (async (opts: AiDegradationDrillOptions) => checkAiServiceHealth(opts));

  const aiHealthResult = await aiHealthCheckFn(options);
  const aiOk = aiHealthResult.ok;
  const aiDetails = aiHealthResult.details as Record<string, unknown> | undefined;

  checks.push({
    name: 'ai_service_health',
    status: aiOk ? 'pass' : 'fail',
    details: {
      ...(aiDetails ?? {}),
      expectedStatus: phase === 'degraded' ? 'unhealthy' : 'healthy',
      phase,
    },
  });

  // Check 3: Optional AI fallback behaviour placeholder.
  const aiFallbackCheckFn =
    (options as any).aiFallbackCheck ?? ((p: Phase) => checkAiFallbackBehaviour(p));
  const fallbackResult = await aiFallbackCheckFn(phase);
  checks.push({
    name: 'ai_fallback_behaviour',
    status: fallbackResult.ok ? 'pass' : 'fail',
    details: fallbackResult.details,
  });

  const now = new Date();
  const runTimestamp = now.toISOString();
  const overallPass = checks.every((check) => check.status === 'pass');

  const report: AiDegradationDrillReport = {
    drillType: 'ai_service_degradation',
    environment: env,
    runTimestamp,
    phase,
    checks,
    overallPass,
    ...(operator !== undefined ? { operator } : {}),
  };

  const defaultOutputPath = path.join(
    'results',
    'ops',
    `ai_degradation.${env}.${phase}.${formatTimestampForFilename(now)}.json`
  );
  const outputPath = options.outputPath ?? defaultOutputPath;

  const outputDir = path.dirname(outputPath);
  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), 'utf-8');

  return { report, outputPath };
}

/**
 * CLI entrypoint.
 */
async function main(): Promise<void> {
  try {
    const { env, operator, phase, output, baseUrl, aiServiceUrl } = parseArgs(process.argv);

    const options: AiDegradationDrillOptions = {
      env,
      phase,
    };

    if (operator !== undefined) {
      options.operator = operator;
    }
    if (output !== undefined) {
      options.outputPath = output;
    }
    if (baseUrl !== undefined) {
      options.baseUrl = baseUrl;
    }
    if (aiServiceUrl !== undefined) {
      options.aiServiceUrl = aiServiceUrl;
    }

    const { report, outputPath } = await runAiDegradationDrill(options);

    console.log(
      `AI degradation drill (env=${report.environment}, phase=${report.phase}): ${
        report.overallPass ? 'PASS' : 'FAIL'
      }`
    );
    console.log(`Report written to ${outputPath}`);

    // Use exitCode instead of process.exit to make this friendlier to test runners.
    process.exitCode = report.overallPass ? 0 : 1;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`AI degradation drill failed: ${message}`);
    process.exitCode = 1;
  }
}

if (require.main === module) {
  main();
}
