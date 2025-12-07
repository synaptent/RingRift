#!/usr/bin/env ts-node
/**
 * Secrets Rotation Drill Harness
 * -----------------------------------------------------------------------------
 *
 * Orchestrates post-rotation verification steps:
 * - Deployment config / secrets validation
 * - Auth smoke test via scripts/test-auth.sh
 * - Optional HTTP health check against /health
 *
 * Produces a JSON report under results/ops/ describing all checks.
 *
 * This script does NOT perform the actual secret rotation; it is intended to be
 * run after secrets have been rotated according to the runbook.
 */

import fs from 'fs';
import path from 'path';
import http from 'http';
import https from 'https';
import { execFile } from 'child_process';
import { URL } from 'url';

import { validateDeploymentConfigProgrammatically } from './validate-deployment-config';

export interface ShellCheckResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
}

export interface DrillCheck {
  name: string;
  status: 'pass' | 'fail';
  details?: unknown;
}

export interface SecretsRotationDrillReport {
  drillType: 'secrets_rotation';
  environment: string;
  operator?: string;
  runTimestamp: string;
  checks: DrillCheck[];
  overallPass: boolean;
}

export interface SecretsRotationDrillOptions {
  env: string;
  operator?: string;
  outputPath?: string;
  baseUrl?: string;
}

interface ParsedCliArgs {
  env: string;
  operator?: string;
  output?: string;
  baseUrl?: string;
}

/**
 * Format a UTC timestamp suitable for filenames, e.g. 20251205T101200Z.
 */
function formatTimestampForFilename(date: Date): string {
  const iso = date.toISOString(); // e.g. 2025-12-05T10:13:20.035Z
  // Strip "-", ":" and fractional seconds for filesystem-safe timestamp
  return iso.replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
}

/**
 * Truncate potentially large stdout/stderr logs for inclusion in the JSON report.
 */
function truncateOutput(output: string, maxLength = 2000): string {
  if (output.length <= maxLength) {
    return output;
  }
  const truncatedBytes = output.length - maxLength;
  return `${output.slice(0, maxLength)}\n...[truncated ${truncatedBytes} bytes]`;
}

/**
 * Run the existing auth smoke test shell script with the given environment.
 * Wraps scripts/test-auth.sh and returns a structured result.
 */
async function runAuthSmokeTest(env: string): Promise<ShellCheckResult> {
  return await new Promise<ShellCheckResult>((resolve) => {
    const childEnv = {
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
 * Minimal HTTP GET health check using Node's http/https modules.
 * Exported to allow Jest tests to mock or stub this behavior.
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
 * Parse CLI arguments for the drill harness.
 *
 * Supported flags:
 *   --env <env>        (required) environment name, e.g. staging, prod-preview, production
 *   --operator <id>    (optional) operator identifier for audit/reporting
 *   --output <path>    (optional) explicit output path for the JSON report
 *   --baseUrl <url>   (optional) base URL for HTTP health check (default: process.env.BASE_URL or http://localhost:3000)
 *   --base-url <url>  (alias for --baseUrl)
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
  const baseUrl = (args.baseUrl as string | undefined) ?? (args['base-url'] as string | undefined);

  const result: ParsedCliArgs = { env };

  if (operator !== undefined) {
    result.operator = operator;
  }
  if (output !== undefined) {
    result.output = output;
  }
  if (baseUrl !== undefined) {
    result.baseUrl = baseUrl;
  }

  return result;
}

/**
 * Core programmatic entrypoint for the secrets rotation drill.
 * Orchestrates all checks and writes the JSON report.
 */
export async function runSecretsRotationDrill(
  options: SecretsRotationDrillOptions
): Promise<{ report: SecretsRotationDrillReport; outputPath: string }> {
  const { env, operator } = options;

  const checks: DrillCheck[] = [];

  // Step A: deployment config / secrets validation
  const configResult = validateDeploymentConfigProgrammatically();
  checks.push({
    name: 'deployment_config_validation',
    status: configResult.ok ? 'pass' : 'fail',
    details: {
      errors: configResult.errors,
      warnings: configResult.warnings,
    },
  });

  // Step B: auth smoke test via scripts/test-auth.sh
  const shellResult = await runAuthSmokeTest(env);
  const stdoutSnippet = truncateOutput(shellResult.stdout);
  const stderrSnippet = truncateOutput(shellResult.stderr);

  checks.push({
    name: 'auth_smoke_test',
    status: shellResult.exitCode === 0 ? 'pass' : 'fail',
    details: {
      exitCode: shellResult.exitCode,
      stdoutSnippet,
      stderrSnippet,
    },
  });

  // Step C: HTTP health check against /health
  const baseUrl = options.baseUrl ?? process.env.BASE_URL ?? 'http://localhost:3000';
  const normalizedBaseUrl = baseUrl.replace(/\/$/, '');
  const healthUrl = `${normalizedBaseUrl}/health`;

  // Allow tests (and potential callers) to inject a custom health check
  const httpHealthCheckFn = (options as any).httpHealthCheck ?? performHttpHealthCheck;
  const { statusCode } = await httpHealthCheckFn(healthUrl);

  checks.push({
    name: 'http_health_check',
    status: statusCode === 200 ? 'pass' : 'fail',
    details: {
      url: healthUrl,
      statusCode,
    },
  });

  const now = new Date();
  const runTimestamp = now.toISOString();
  const overallPass = checks.every((check) => check.status === 'pass');

  const report: SecretsRotationDrillReport = {
    drillType: 'secrets_rotation',
    environment: env,
    runTimestamp,
    checks,
    overallPass,
  };

  if (operator !== undefined) {
    report.operator = operator;
  }

  const defaultOutputPath = path.join(
    'results',
    'ops',
    `secrets_rotation.${env}.${formatTimestampForFilename(now)}.json`
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
    const { env, operator, output, baseUrl } = parseArgs(process.argv);

    const options: SecretsRotationDrillOptions = {
      env,
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

    const { report, outputPath } = await runSecretsRotationDrill(options);

    // eslint-disable-next-line no-console
    console.log(
      `Secrets rotation drill (env=${report.environment}): ${report.overallPass ? 'PASS' : 'FAIL'}`
    );
    // eslint-disable-next-line no-console
    console.log(`Report written to ${outputPath}`);

    process.exit(report.overallPass ? 0 : 1);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);

    console.error(`Secrets rotation drill failed: ${message}`);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}
