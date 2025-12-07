#!/usr/bin/env ts-node
/**
 * dev_doctor.ts
 * =============
 *
 * Developer environment doctor for the Node/TS backend.
 *
 * Runs a small set of checks to help new or returning developers get to a
 * "green tests" baseline quickly:
 *
 * - Environment & config sanity (NODE_ENV, loaded .env, AI service URL)
 * - Database connectivity (Prisma + PostgreSQL)
 * - Redis connectivity (optional but recommended)
 * - AI service model presence (basic file existence check)
 *
 * Usage (from repo root):
 *
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/dev_doctor.ts
 *
 * Exit code:
 *   0  – all hard checks passed
 *   1  – one or more hard checks failed (see output)
 *
 * Notes:
 * - Redis and AI model checks are treated as soft warnings in development;
 *   they become effectively mandatory in production deployments.
 */

/* eslint-disable no-console */

import fs from 'fs';
import path from 'path';
import axios from 'axios';

import { config } from '../src/server/config';
import {
  connectDatabase,
  disconnectDatabase,
  checkDatabaseHealth,
} from '../src/server/database/connection';
import { connectRedis, disconnectRedis } from '../src/server/cache/redis';

interface CheckResult {
  name: string;
  ok: boolean;
  message: string;
  fatal: boolean;
}

async function checkNodeEnvironment(): Promise<CheckResult> {
  const nodeVersion = process.versions.node;
  const major = parseInt(nodeVersion.split('.')[0] ?? '0', 10);

  const recommendedMajor = 18;
  const ok = major >= recommendedMajor;

  const message = ok
    ? `Node.js ${nodeVersion} (>=${recommendedMajor}.x) – OK`
    : `Node.js ${nodeVersion} detected, but >=${recommendedMajor}.x is recommended.`;

  return {
    name: 'Node / env',
    ok,
    message:
      `${message}\n` +
      `  NODE_ENV: ${process.env.NODE_ENV ?? 'undefined'}\n` +
      `  Effective env: ${config.nodeEnv}\n` +
      `  App version: ${config.app.version}`,
    fatal: false,
  };
}

async function checkDatabase(): Promise<CheckResult> {
  if (!config.database.url) {
    // Optional in non-production; warn but do not fail hard.
    return {
      name: 'Database',
      ok: false,
      message:
        'DATABASE_URL is not set; Prisma/Postgres-backed features will be unavailable. ' +
        'For full backend tests, configure a Postgres instance and set DATABASE_URL.',
      fatal: false,
    };
  }

  try {
    await connectDatabase();
    const healthy = await checkDatabaseHealth();

    return {
      name: 'Database',
      ok: healthy,
      message: healthy
        ? `Connected to PostgreSQL (${config.database.url}) and health check passed.`
        : 'Connected to PostgreSQL but health check failed (SELECT 1). See logs for details.',
      // Treat DB connectivity as fatal for dev_doctor: most backend tests require it.
      fatal: true,
    };
  } catch (err) {
    return {
      name: 'Database',
      ok: false,
      message: `Failed to connect to PostgreSQL via Prisma. Error: ${
        err instanceof Error ? err.message : String(err)
      }`,
      fatal: true,
    };
  } finally {
    await disconnectDatabase().catch(() => undefined);
  }
}

async function checkRedis(): Promise<CheckResult> {
  // In non-production we can treat missing/failed Redis as a soft warning.
  if (!config.redis.url) {
    return {
      name: 'Redis',
      ok: false,
      message:
        'Redis URL is empty; caching, rate-limiting, and WebSocket session features may be disabled.',
      fatal: false,
    };
  }

  try {
    const client = await connectRedis();
    // Use a simple PING to verify basic connectivity.
    const pong = await client.ping();
    const ok = pong === 'PONG';

    return {
      name: 'Redis',
      ok,
      message: ok
        ? `Connected to Redis at ${config.redis.url} (PING => PONG).`
        : `Connected to Redis at ${config.redis.url}, but PING did not return PONG (got ${pong}).`,
      fatal: false,
    };
  } catch (err) {
    return {
      name: 'Redis',
      ok: false,
      message: `Failed to connect to Redis at ${config.redis.url}. Error: ${
        err instanceof Error ? err.message : String(err)
      }`,
      fatal: false,
    };
  } finally {
    await disconnectRedis().catch(() => undefined);
  }
}

async function checkAiModels(): Promise<CheckResult> {
  // Basic existence check for the primary AI model file used by the Python service.
  const repoRoot = path.resolve(__dirname, '..');
  const aiServiceModelsDir = path.join(repoRoot, 'ai-service', 'models');
  const primaryModel = path.join(aiServiceModelsDir, 'ringrift_v1.pth');

  const hasDir = fs.existsSync(aiServiceModelsDir);
  const hasModel = fs.existsSync(primaryModel);

  if (!hasDir) {
    return {
      name: 'AI models',
      ok: false,
      message:
        'ai-service/models directory not found. If you plan to run the Python AI service, ' +
        'ensure that repository is checked out and any required model files are present.',
      fatal: false,
    };
  }

  if (!hasModel) {
    return {
      name: 'AI models',
      ok: false,
      message:
        'Primary model file ai-service/models/ringrift_v1.pth is missing. ' +
        'Self-play soaks and neural-network-backed AIs may fall back to random/heuristic behaviour.',
      fatal: false,
    };
  }

  return {
    name: 'AI models',
    ok: true,
    message: 'Found ai-service/models/ringrift_v1.pth (primary v1 model present).',
    fatal: false,
  };
}

async function checkAiServiceHttp(): Promise<CheckResult> {
  // Opt-in health check so dev_doctor does not hang or fail when the
  // Python AI service is not running. Enable via:
  //
  //   DEV_DOCTOR_CHECK_AI_SERVICE=true npm run dev:doctor
  //
  const enabled =
    process.env.DEV_DOCTOR_CHECK_AI_SERVICE === 'true' ||
    process.env.DEV_DOCTOR_CHECK_AI_SERVICE === '1';

  const baseUrl = (config as any).aiService?.url as string | undefined;

  if (!enabled || !baseUrl) {
    return {
      name: 'AI service (HTTP)',
      ok: false,
      message: enabled
        ? 'AI service URL is not configured; skipping HTTP health check.'
        : 'AI service HTTP health check skipped (set DEV_DOCTOR_CHECK_AI_SERVICE=true to enable).',
      fatal: false,
    };
  }

  const healthUrl = `${baseUrl.replace(/\/+$/, '')}/health`;

  try {
    const response = await axios.get(healthUrl, {
      timeout: (config as any).aiService?.requestTimeoutMs ?? 2000,
    });

    const status = (response.data as { status?: string } | undefined)?.status ?? 'unknown';
    const ok = response.status === 200;

    return {
      name: 'AI service (HTTP)',
      ok,
      message: ok
        ? `AI service responded at ${healthUrl} with status=${status}.`
        : `AI service at ${healthUrl} returned HTTP ${response.status} (status=${status}).`,
      fatal: false,
    };
  } catch (err) {
    return {
      name: 'AI service (HTTP)',
      ok: false,
      message: `Failed to reach AI service at ${healthUrl}. Error: ${
        err instanceof Error ? err.message : String(err)
      }`,
      fatal: false,
    };
  }
}

async function main(): Promise<void> {
  console.log('=== RingRift dev_doctor ===\n');

  const checks: Array<() => Promise<CheckResult>> = [
    checkNodeEnvironment,
    checkDatabase,
    checkRedis,
    checkAiModels,
    checkAiServiceHttp,
  ];

  const results: CheckResult[] = [];
  for (const check of checks) {
    const result = await check();
    results.push(result);
  }

  let fatalFailures = 0;
  let warnings = 0;

  for (const result of results) {
    const status = result.ok ? 'OK' : result.fatal ? 'FAIL' : 'WARN';
    console.log(`[${status}] ${result.name}`);
    console.log(`  ${result.message}\n`);

    if (!result.ok) {
      if (result.fatal) {
        fatalFailures += 1;
      } else {
        warnings += 1;
      }
    }
  }

  console.log('Summary:');
  console.log(`  Fatal failures: ${fatalFailures}`);
  console.log(`  Warnings:       ${warnings}`);

  if (fatalFailures > 0) {
    console.log(
      '\nOne or more fatal checks failed. Fix the issues above before running full backend tests.'
    );
    process.exitCode = 1;
  } else {
    console.log('\nAll hard checks passed. You should be close to a green backend test run.');
  }
}

main().catch((err) => {
  console.error('dev_doctor encountered an unexpected error:', err);
  process.exitCode = 1;
});
