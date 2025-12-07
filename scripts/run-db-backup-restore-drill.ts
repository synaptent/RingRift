#!/usr/bin/env ts-node
/**
 * Database Backup & Restore Drill Harness
 * -----------------------------------------------------------------------------
 *
 * Orchestrates the non-destructive backup/restore drill described in
 * docs/runbooks/DATABASE_BACKUP_AND_RESTORE_DRILL.md:
 *
 * - Take a logical backup of the primary DB (via docker compose + pg_dump).
 * - Restore that backup into a separate scratch database.
 * - Run a minimal Prisma-based smoke test against the restored DB.
 *
 * Produces a JSON report under results/ops/ describing all checks.
 *
 * This script does NOT change schema or production settings; it assumes you are
 * running it in a safe non-production environment (e.g. staging host).
 */

/* eslint-disable no-console */

import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { PrismaClient } from '@prisma/client';

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

export interface DbBackupRestoreDrillReport {
  drillType: 'db_backup_restore';
  environment: string;
  operator?: string;
  targetDatabase?: string;
  runTimestamp: string;
  checks: DrillCheck[];
  overallPass: boolean;
}

export interface DbBackupRestoreDrillOptions {
  env: string;
  operator?: string;
  outputPath?: string;
  // Name/URL/DSN of the *source* DB we are backing up.
  sourceDatabaseUrl?: string;
  // Optional name/URL/DSN for the scratch/restore target DB.
  restoreDatabaseUrl?: string;
}

interface ParsedCliArgs {
  env: string;
  operator?: string;
  output?: string;
  sourceDatabaseUrl?: string;
  restoreDatabaseUrl?: string;
}

/**
 * Format a UTC timestamp suitable for filenames, e.g. 20251205T101200Z.
 * Mirrors the helper used in run-secrets-rotation-drill.ts.
 */
function formatTimestampForFilename(date: Date): string {
  const iso = date.toISOString(); // e.g. 2025-12-05T10:13:20.035Z
  // Strip "-", ":" and fractional seconds for filesystem-safe timestamp
  return iso.replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
}

/**
 * Format a timestamp for backup file names, matching the runbook style
 * (YYYYMMDD_HHMMSS).
 */
function formatTimestampForBackup(date: Date): string {
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, '0');
  const day = String(date.getUTCDate()).padStart(2, '0');
  const hours = String(date.getUTCHours()).padStart(2, '0');
  const minutes = String(date.getUTCMinutes()).padStart(2, '0');
  const seconds = String(date.getUTCSeconds()).padStart(2, '0');
  return `${year}${month}${day}_${hours}${minutes}${seconds}`;
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
 * Derive the logical database name to use for the restore DB from a URL, falling
 * back to the runbook default `ringrift_restore_drill` when not available.
 */
function deriveRestoreDbName(restoreDatabaseUrl?: string): string {
  if (!restoreDatabaseUrl) {
    return 'ringrift_restore_drill';
  }

  try {
    const url = new URL(restoreDatabaseUrl);
    const dbName = url.pathname.replace(/^\//, '').trim();
    return dbName || 'ringrift_restore_drill';
  } catch {
    return 'ringrift_restore_drill';
  }
}

/**
 * Run a shell script via bash -lc, capturing stdout/stderr and an exit code.
 */
async function runShellScript(script: string, env: NodeJS.ProcessEnv): Promise<ShellCheckResult> {
  return await new Promise<ShellCheckResult>((resolve) => {
    execFile(
      'bash',
      ['-lc', script],
      {
        env,
        maxBuffer: 10 * 1024 * 1024,
      },
      (error, stdout, stderr) => {
        let exitCode: number | null = 0;
        if (error) {
          const anyErr = error as any;
          if (typeof anyErr.code === 'number') {
            exitCode = anyErr.code;
          } else {
            exitCode = 1;
          }
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
 * Run the backup command using docker-compose / docker compose and pg_dump,
 * following the runbook pattern.
 *
 * This assumes:
 * - A `postgres` service is defined in docker-compose.yml
 * - That service mounts `./backups` on the host to `/backups` in the container
 * - The primary DB is `ringrift` owned by `ringrift`
 */
async function runBackupCommand(
  envName: string
): Promise<ShellCheckResult & { backupPath?: string }> {
  const now = new Date();
  const backupTimestamp = formatTimestampForBackup(now);
  const backupFileName = `${envName}_drill_${backupTimestamp}.sql`;
  const backupHostPath = path.join('backups', backupFileName);

  const script = `
set -euo pipefail

if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
elif command -v docker >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
else
  echo "[db-backup-restore] ERROR: Neither docker-compose nor docker is available on PATH." >&2
  exit 1
fi

BACKUP_FILE_NAME="${backupFileName}"
BACKUP_CONTAINER_PATH="/backups/${backupFileName}"

echo "[db-backup-restore] Taking logical backup into \${BACKUP_CONTAINER_PATH}..."
"\${COMPOSE_CMD}" exec postgres \\
  pg_dump -U ringrift -d ringrift \\
  -f "\${BACKUP_CONTAINER_PATH}"

# Emit the host-path of the backup so the harness can record it
echo "__BACKUP_FILE__=${backupHostPath}"
`;

  const result = await runShellScript(script, {
    ...process.env,
  });

  let backupPath: string | undefined;
  const match = result.stdout.match(/__BACKUP_FILE__=(.+)/);
  if (match) {
    backupPath = match[1].trim();
  }

  if (backupPath !== undefined) {
    return { ...result, backupPath };
  }

  return result;
}

/**
 * Restore the given backup file into a scratch database using docker-compose
 * / docker compose and psql, following the runbook pattern.
 */
async function runRestoreFromBackup(
  restoreDbName: string,
  backupHostPath: string
): Promise<ShellCheckResult> {
  const script = `
set -euo pipefail

if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
elif command -v docker >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
else
  echo "[db-backup-restore] ERROR: Neither docker-compose nor docker is available on PATH." >&2
  exit 1
fi

RESTORE_DB="${restoreDbName}"
BACKUP_HOST_PATH="${backupHostPath}"

echo "[db-backup-restore] Creating restore database \${RESTORE_DB} (if needed)..."
# createdb will fail if the DB already exists; allow callers to manage cleanup/name
"\${COMPOSE_CMD}" exec postgres \\
  createdb -U ringrift "\${RESTORE_DB}" || true

echo "[db-backup-restore] Restoring from \${BACKUP_HOST_PATH} into \${RESTORE_DB}..."
"\${COMPOSE_CMD}" exec -T postgres \\
  psql -U ringrift -d "\${RESTORE_DB}" < "\${BACKUP_HOST_PATH}"
`;

  return await runShellScript(script, {
    ...process.env,
  });
}

/**
 * Minimal Prisma-based smoke test against the restored DB.
 *
 * This is exported so Jest tests can mock it instead of touching a real DB.
 */
export async function runRestoreSmokeTest(
  restoreDatabaseUrl?: string
): Promise<{ ok: boolean; error?: string }> {
  if (!restoreDatabaseUrl) {
    return {
      ok: false,
      error: 'restoreDatabaseUrl was not provided; cannot perform smoke test against restored DB.',
    };
  }

  let prisma: PrismaClient | null = null;
  const previousDatabaseUrl = process.env.DATABASE_URL;

  try {
    process.env.DATABASE_URL = restoreDatabaseUrl;
    prisma = new PrismaClient();

    await prisma.$connect();
    // Lightweight health query; aligns with checkDatabaseHealth in connection.ts.

    await prisma.$queryRaw`SELECT 1`;

    return { ok: true };
  } catch (err) {
    return {
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    };
  } finally {
    if (prisma) {
      await prisma.$disconnect().catch(() => undefined);
    }
    if (previousDatabaseUrl === undefined) {
      delete process.env.DATABASE_URL;
    } else {
      process.env.DATABASE_URL = previousDatabaseUrl;
    }
  }
}

/**
 * Parse CLI arguments for the drill harness.
 *
 * Supported flags:
 *   --env <env>                 (required) environment name, e.g. staging
 *   --operator <id>             (optional) operator identifier for audit/reporting
 *   --output <path>              (optional) explicit output path for the JSON report
 *   --sourceDatabaseUrl <url>    (optional) DSN of source DB (alias: --source-database-url)
 *   --restoreDatabaseUrl <url>   (optional) DSN of restore DB (alias: --restore-database-url)
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
  const sourceDatabaseUrl =
    (args.sourceDatabaseUrl as string | undefined) ??
    (args['source-database-url'] as string | undefined);
  const restoreDatabaseUrl =
    (args.restoreDatabaseUrl as string | undefined) ??
    (args['restore-database-url'] as string | undefined);

  const result: ParsedCliArgs = { env };

  if (operator !== undefined) {
    result.operator = operator;
  }
  if (output !== undefined) {
    result.output = output;
  }
  if (sourceDatabaseUrl !== undefined) {
    result.sourceDatabaseUrl = sourceDatabaseUrl;
  }
  if (restoreDatabaseUrl !== undefined) {
    result.restoreDatabaseUrl = restoreDatabaseUrl;
  }

  return result;
}

/**
 * Core programmatic entrypoint for the DB backup/restore drill.
 * Orchestrates all checks and writes the JSON report.
 */
export async function runDbBackupRestoreDrill(
  options: DbBackupRestoreDrillOptions
): Promise<{ report: DbBackupRestoreDrillReport; outputPath: string }> {
  const { env, operator } = options;

  const effectiveSourceUrl = options.sourceDatabaseUrl ?? process.env.DATABASE_URL;
  const effectiveRestoreUrl = options.restoreDatabaseUrl ?? process.env.DATABASE_URL_RESTORE;
  const restoreDbName = deriveRestoreDbName(effectiveRestoreUrl);

  const checks: DrillCheck[] = [];

  // Step 1: backup creation
  const backupResult = await runBackupCommand(env);
  const backupStdoutSnippet = truncateOutput(backupResult.stdout);
  const backupStderrSnippet = truncateOutput(backupResult.stderr);

  const backupStatus: 'pass' | 'fail' = backupResult.exitCode === 0 ? 'pass' : 'fail';
  checks.push({
    name: 'db_backup_create',
    status: backupStatus,
    details: {
      exitCode: backupResult.exitCode,
      stdoutSnippet: backupStdoutSnippet,
      stderrSnippet: backupStderrSnippet,
      backupPath: backupResult.backupPath,
    },
  });

  // Step 2: restore into scratch DB
  let restoreStatus: 'pass' | 'fail' = 'fail';

  if (backupStatus === 'pass' && backupResult.backupPath) {
    const restoreResult = await runRestoreFromBackup(restoreDbName, backupResult.backupPath);
    const restoreStdoutSnippet = truncateOutput(restoreResult.stdout);
    const restoreStderrSnippet = truncateOutput(restoreResult.stderr);

    restoreStatus = restoreResult.exitCode === 0 ? 'pass' : 'fail';

    checks.push({
      name: 'db_restore_to_scratch',
      status: restoreStatus,
      details: {
        exitCode: restoreResult.exitCode,
        stdoutSnippet: restoreStdoutSnippet,
        stderrSnippet: restoreStderrSnippet,
        restoreDatabaseName: restoreDbName,
        backupPath: backupResult.backupPath,
      },
    });
  } else {
    checks.push({
      name: 'db_restore_to_scratch',
      status: 'fail',
      details: {
        skipped: true,
        reason: backupStatus !== 'pass' ? 'backup_failed' : 'backup_path_missing',
        restoreDatabaseName: restoreDbName,
      },
    });
  }

  // Step 3: smoke test
  // Allow tests (and potential callers) to inject a custom smoke test
  // implementation so we do not need to mock Prisma directly.
  const smokeTestFn = (options as any).smokeTestFn ?? runRestoreSmokeTest;

  if (restoreStatus === 'pass') {
    const smokeResult = await smokeTestFn(effectiveRestoreUrl);
    checks.push({
      name: 'db_restore_smoke_test',
      status: smokeResult.ok ? 'pass' : 'fail',
      details: {
        restoreDatabaseUrl: effectiveRestoreUrl,
        error: smokeResult.error,
      },
    });
  } else {
    checks.push({
      name: 'db_restore_smoke_test',
      status: 'fail',
      details: {
        skipped: true,
        reason: 'restore_failed',
        restoreDatabaseUrl: effectiveRestoreUrl,
      },
    });
  }

  const now = new Date();
  const runTimestamp = now.toISOString();
  const overallPass = checks.every((check) => check.status === 'pass');

  const targetDatabase = effectiveRestoreUrl ?? effectiveSourceUrl;

  const report: DbBackupRestoreDrillReport = {
    drillType: 'db_backup_restore',
    environment: env,
    runTimestamp,
    checks,
    overallPass,
    ...(targetDatabase !== undefined ? { targetDatabase } : {}),
    ...(operator !== undefined ? { operator } : {}),
  };

  const defaultOutputPath = path.join(
    'results',
    'ops',
    `db_backup_restore.${env}.${formatTimestampForFilename(now)}.json`
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
    const { env, operator, output, sourceDatabaseUrl, restoreDatabaseUrl } = parseArgs(
      process.argv
    );

    const options: DbBackupRestoreDrillOptions = {
      env,
    };

    if (operator !== undefined) {
      options.operator = operator;
    }
    if (output !== undefined) {
      options.outputPath = output;
    }

    const effectiveSourceUrl = sourceDatabaseUrl ?? process.env.DATABASE_URL;
    const effectiveRestoreUrl = restoreDatabaseUrl ?? process.env.DATABASE_URL_RESTORE;

    if (effectiveSourceUrl !== undefined) {
      options.sourceDatabaseUrl = effectiveSourceUrl;
    }
    if (effectiveRestoreUrl !== undefined) {
      options.restoreDatabaseUrl = effectiveRestoreUrl;
    }

    const { report, outputPath } = await runDbBackupRestoreDrill(options);

    console.log(
      `DB backup/restore drill (env=${report.environment}): ${report.overallPass ? 'PASS' : 'FAIL'}`
    );
    console.log(`Report written to ${outputPath}`);

    process.exit(report.overallPass ? 0 : 1);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`DB backup/restore drill failed: ${message}`);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}
