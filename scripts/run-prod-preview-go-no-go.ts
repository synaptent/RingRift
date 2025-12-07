#!/usr/bin/env ts-node
/**
 * Production Preview Go/No-Go Harness
 * -----------------------------------------------------------------------------
 *
 * Validates a prod-like deployment topology/configuration and runs a focused
 * smoke of critical end-to-end flows:
 *   - Topology + deployment config validation
 *   - Auth HTTP flow via scripts/test-auth.sh
 *   - Lobby/game creation + WebSocket + reconnection + AI via game-session-load-smoke
 *   - AI service readiness (via HealthCheckService through the AI drill helper)
 *
 * Produces a single JSON report under results/ops/:
 *   - drillType: 'prod_preview_go_no_go'
 *   - environment, operator
 *   - topologySummary
 *   - per-check results
 *   - overallPass (all checks pass)
 *
 * This harness assumes a prod-like stack is already running for the target env.
 * It does NOT start or manage containers.
 */

import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';

import {
  validateDeploymentConfigProgrammatically,
  type DeploymentConfigValidationResult,
} from './validate-deployment-config';
import { checkAiServiceHealth } from './run-ai-degradation-drill';

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

export type CheckStatus = 'pass' | 'fail';

export interface GoNoGoCheck {
  name: string;
  status: CheckStatus;
  details?: unknown;
}

export interface ProdPreviewGoNoGoReport {
  drillType: 'prod_preview_go_no_go';
  environment: string;
  operator?: string;
  runTimestamp: string; // ISO UTC
  topologySummary: {
    appTopology: string;
    expectedTopology: string;
    configOk: boolean;
  };
  checks: GoNoGoCheck[];
  overallPass: boolean;
}

export interface ProdPreviewGoNoGoOptions {
  env: string;
  operator?: string;
  outputPath?: string;
  expectedTopology?: string; // default 'single' unless overridden
  baseUrl?: string;
}

export interface ParsedCliArgs {
  env: string;
  operator?: string;
  output?: string;
  expectedTopology?: string;
  baseUrl?: string;
}

export interface ShellCheckResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/**
 * Format a UTC timestamp suitable for filenames, e.g. 20251205T101200Z.
 */
export function formatTimestampForFilename(date: Date): string {
  const iso = date.toISOString(); // e.g. 2025-12-05T10:13:20.035Z
  // Strip "-", ":" and fractional seconds for filesystem-safe timestamp
  return iso.replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
}

/**
 * Truncate potentially large stdout/stderr logs for inclusion in the JSON report.
 */
export function truncateOutput(output: string, maxLength = 2000): string {
  if (output.length <= maxLength) {
    return output;
  }
  const truncatedBytes = output.length - maxLength;
  return `${output.slice(0, maxLength)}\n...[truncated ${truncatedBytes} bytes]`;
}

// -----------------------------------------------------------------------------
// CLI parsing
// -----------------------------------------------------------------------------

/**
 * Parse CLI arguments for the prod-preview go/no-go harness.
 *
 * Supported flags:
 *   --env <env>               (required) environment name, e.g. staging, prod-preview
 *   --operator <id>           (optional) operator identifier for audit/reporting
 *   --output <path>           (optional) explicit output path for the JSON report
 *   --expectedTopology <topology>
 *                             (optional) expected app topology (default: 'single')
 *   --baseUrl <url>          (optional) backend base URL (alias: --base-url)
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

  const operator = args.operator as string | undefined;
  const output = (args.output as string | undefined) ?? (args.o as string | undefined);
  const expectedTopology =
    (args.expectedTopology as string | undefined) ??
    (args['expected-topology'] as string | undefined);
  const baseUrl = (args.baseUrl as string | undefined) ?? (args['base-url'] as string | undefined);

  const result: ParsedCliArgs = { env };

  if (operator !== undefined) {
    result.operator = operator;
  }
  if (output !== undefined) {
    result.output = output;
  }
  if (expectedTopology !== undefined) {
    result.expectedTopology = expectedTopology;
  }
  if (baseUrl !== undefined) {
    result.baseUrl = baseUrl;
  }

  return result;
}

// -----------------------------------------------------------------------------
// Shell helpers
// -----------------------------------------------------------------------------

/**
 * Run the existing auth smoke test shell script with the given environment.
 * Wraps scripts/test-auth.sh and returns a structured result.
 */
async function runAuthSmoke(env: string): Promise<ShellCheckResult> {
  return await new Promise<ShellCheckResult>((resolve) => {
    const childEnv: NodeJS.ProcessEnv = {
      ...process.env,
      ENV: env,
    };

    execFile(
      'bash',
      [path.join(__dirname, 'test-auth.sh')],
      { env: childEnv },
      (error, stdout, stderr) => {
        let exitCode: number | null = null;
        if (error) {
          const anyErr = error as any;
          if (typeof anyErr.code === 'number') {
            exitCode = anyErr.code;
          } else {
            exitCode = 1;
          }
        } else {
          exitCode = 0;
        }

        resolve({
          exitCode,
          stdout: stdout ?? '',
          stderr: stderr ?? '',
        });
      }
    );
  });
}

/**
 * Run a minimal game session smoke against the backend using the existing
 * game-session-load-smoke harness.
 *
 * This exercises:
 *   - Lobby / game creation
 *   - Join flow
 *   - WebSocket connections (players + spectator)
 *   - Basic reconnect behaviour
 *   - At least one AI path (via the harness's aiFraction)
 */
async function runGameSessionSmoke(env: string, baseUrl: string): Promise<ShellCheckResult> {
  return await new Promise<ShellCheckResult>((resolve) => {
    const childEnv: NodeJS.ProcessEnv = {
      ...process.env,
      ENV: env,
      TS_NODE_PROJECT: process.env.TS_NODE_PROJECT ?? 'tsconfig.server.json',
    };

    const scriptPath = path.join(__dirname, 'game-session-load-smoke.ts');

    const args = [
      'ts-node',
      scriptPath,
      '--games',
      '2',
      '--concurrency',
      '1',
      '--baseUrl',
      baseUrl,
    ];

    execFile(
      'npx',
      args,
      {
        env: childEnv,
        maxBuffer: 10 * 1024 * 1024,
      },
      (error, stdout, stderr) => {
        let exitCode: number | null = null;
        if (error) {
          const anyErr = error as any;
          if (typeof anyErr.code === 'number') {
            exitCode = anyErr.code;
          } else {
            exitCode = 1;
          }
        } else {
          exitCode = 0;
        }

        resolve({
          exitCode,
          stdout: stdout ?? '',
          stderr: stderr ?? '',
        });
      }
    );
  });
}

// -----------------------------------------------------------------------------
// Core orchestration
// -----------------------------------------------------------------------------

export async function runProdPreviewGoNoGo(
  options: ProdPreviewGoNoGoOptions
): Promise<{ report: ProdPreviewGoNoGoReport; outputPath: string }> {
  const { env, operator } = options;

  const checks: GoNoGoCheck[] = [];

  // 1) Topology + deployment config sanity check --------------------------------
  const appTopologyRaw = (process.env.RINGRIFT_APP_TOPOLOGY ?? 'unknown').trim();
  const appTopology = appTopologyRaw || 'unknown';
  const expectedTopology = options.expectedTopology ?? 'single';
  const topologyMatches = appTopology === expectedTopology;

  const configValidation: DeploymentConfigValidationResult =
    validateDeploymentConfigProgrammatically();
  const configOk = configValidation.ok;

  checks.push({
    name: 'topology_and_config',
    status: topologyMatches && configOk ? 'pass' : 'fail',
    details: {
      appTopology,
      expectedTopology,
      topologyMatches,
      configErrors: configValidation.errors,
      configWarnings: configValidation.warnings,
    },
  });

  // 2) Auth smoke test via test-auth.sh -----------------------------------------
  const effectiveBaseUrl =
    options.baseUrl ??
    // Prefer APP_BASE to align with operational runbooks.
    process.env.APP_BASE ??
    process.env.BASE_URL ??
    'http://localhost:3000';
  const normalizedBaseUrl = effectiveBaseUrl.replace(/\/$/, '');

  // Allow tests (and callers) to inject a custom auth smoke helper.
  const authSmokeFn = (options as any).authSmoke ?? ((e: string) => runAuthSmoke(e));
  const authResult = await authSmokeFn(env);
  checks.push({
    name: 'auth_smoke_test',
    status: authResult.exitCode === 0 ? 'pass' : 'fail',
    details: {
      exitCode: authResult.exitCode,
      stdoutSnippet: truncateOutput(authResult.stdout),
      stderrSnippet: truncateOutput(authResult.stderr),
      // Note: test-auth.sh currently targets /api/auth on localhost:3000 internally.
      // Future improvements can teach it to respect ENV-specific BASE_URL.
    },
  });

  // 3) Game/lobby/WebSocket/AI smoke via game-session-load-smoke ----------------
  // Allow tests (and callers) to inject a custom game-session smoke helper.
  const gameSmokeFn =
    (options as any).gameSmoke ?? ((e: string, url: string) => runGameSessionSmoke(e, url));
  const gameSmokeResult = await gameSmokeFn(env, normalizedBaseUrl);
  checks.push({
    name: 'game_session_smoke',
    status: gameSmokeResult.exitCode === 0 ? 'pass' : 'fail',
    details: {
      exitCode: gameSmokeResult.exitCode,
      stdoutSnippet: truncateOutput(gameSmokeResult.stdout),
      stderrSnippet: truncateOutput(gameSmokeResult.stderr),
      baseUrl: normalizedBaseUrl,
    },
  });

  // 4) AI service readiness ------------------------------------------------------
  // Allow tests (and potential callers) to inject a custom AI health helper so we
  // do not need to mock the HealthCheckService module directly.
  const aiHealthCheckFn =
    (options as any).aiHealthCheck ??
    (async (_opts: ProdPreviewGoNoGoOptions) =>
      checkAiServiceHealth({ env, phase: 'baseline' } as any));

  const aiHealthResult = await aiHealthCheckFn(options);
  const aiOk = !!aiHealthResult.ok;
  const aiDetails = aiHealthResult.details;

  checks.push({
    name: 'ai_service_readiness',
    status: aiOk ? 'pass' : 'fail',
    details: aiDetails,
  });

  // ---------------------------------------------------------------------------
  // Report assembly
  // ---------------------------------------------------------------------------

  const now = new Date();
  const runTimestamp = now.toISOString();
  const overallPass = checks.every((check) => check.status === 'pass');

  const report: ProdPreviewGoNoGoReport = {
    drillType: 'prod_preview_go_no_go',
    environment: env,
    runTimestamp,
    topologySummary: {
      appTopology,
      expectedTopology,
      configOk,
    },
    checks,
    overallPass,
    ...(operator !== undefined ? { operator } : {}),
  };

  const defaultOutputPath = path.join(
    'results',
    'ops',
    `prod_preview_go_no_go.${env}.${formatTimestampForFilename(now)}.json`
  );
  const outputPath = options.outputPath ?? defaultOutputPath;

  const outputDir = path.dirname(outputPath);
  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), 'utf-8');

  return { report, outputPath };
}

// -----------------------------------------------------------------------------
// CLI entrypoint
// -----------------------------------------------------------------------------

async function main(): Promise<void> {
  try {
    const { env, operator, output, expectedTopology, baseUrl } = parseArgs(process.argv);

    const options: ProdPreviewGoNoGoOptions = {
      env,
    };

    if (operator !== undefined) {
      options.operator = operator;
    }
    if (output !== undefined) {
      options.outputPath = output;
    }
    if (expectedTopology !== undefined) {
      options.expectedTopology = expectedTopology;
    }
    if (baseUrl !== undefined) {
      options.baseUrl = baseUrl;
    }

    const { report, outputPath } = await runProdPreviewGoNoGo(options);

    // eslint-disable-next-line no-console
    console.log(
      `Production preview go/no-go (env=${report.environment}): ${
        report.overallPass ? 'PASS' : 'FAIL'
      }`
    );
    // eslint-disable-next-line no-console
    console.log(`Report written to ${outputPath}`);

    // Use exitCode instead of process.exit to make this friendlier to test runners.
    process.exitCode = report.overallPass ? 0 : 1;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);

    console.error(`Production preview go/no-go failed: ${message}`);
    process.exitCode = 1;
  }
}

if (require.main === module) {
  main();
}
