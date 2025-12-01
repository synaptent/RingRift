#!/usr/bin/env ts-node
/**
 * Orchestrator HTTP load/scale smoke harness.
 *
 * This script performs a small multi-game run against a running RingRift
 * server, exercising the /api/auth and /api/games HTTP surface under an
 * orchestrator‑ON posture. It is intended as a lightweight, developer‑driven
 * load/scale smoke rather than a full benchmark.
 *
 * Behaviour:
 * - Registers N throwaway users via POST /api/auth/register.
 * - For each user, creates one short game via POST /api/games.
 * - Optionally fetches the user's games list via GET /api/games.
 * - Optionally inspects /metrics to confirm orchestrator metrics are exposed.
 *
 * Usage (from repo root, with server running on localhost:3000):
 *
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/orchestrator-load-smoke.ts
 *
 * Options:
 *   --games=<number>       Total number of games to create (default: 10)
 *   --concurrency=<number> Number of concurrent workers (default: 5)
 *   --baseUrl=<url>        Base URL of the API (default: http://localhost:3000)
 *
 * Notes:
 * - The server should be started with the orchestrator‑ON profile, e.g. the
 *   defaults described in ORCHESTRATOR_ROLLOUT_PLAN Table 4:
 *     RINGRIFT_RULES_MODE=ts
 *     ORCHESTRATOR_ADAPTER_ENABLED=true
 *     ORCHESTRATOR_ROLLOUT_PERCENTAGE=100
 *     ORCHESTRATOR_SHADOW_MODE_ENABLED=false
 * - This harness focuses on HTTP surface and basic DB/metrics wiring. For
 *   deep orchestrator semantics and invariants, use:
 *     npm run soak:orchestrator:smoke
 *     npm run test:orchestrator:s-invariant
 */

import axios from 'axios';
import { performance } from 'perf_hooks';

type SmokeArgs = {
  games: number;
  concurrency: number;
  baseUrl: string;
};

type LatencySample = {
  label: string;
  ms: number;
};

function parseArgs(argv: string[]): SmokeArgs {
  const defaults: SmokeArgs = {
    games: 10,
    concurrency: 5,
    baseUrl: 'http://localhost:3000',
  };

  const args: Record<string, string | boolean> = {};

  for (let i = 2; i < argv.length; i += 1) {
    const raw = argv[i];
    if (!raw.startsWith('--')) continue;

    const eq = raw.indexOf('=');
    let key: string;
    let value: string | boolean;
    if (eq !== -1) {
      key = raw.slice(2, eq);
      value = raw.slice(eq + 1);
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

  const games = args.games ? Number(args.games) : defaults.games;
  const concurrency = args.concurrency ? Number(args.concurrency) : defaults.concurrency;
  const baseUrl = (args.baseUrl as string | undefined) ?? defaults.baseUrl;

  return {
    games: Number.isFinite(games) && games > 0 ? games : defaults.games,
    concurrency: Number.isFinite(concurrency) && concurrency > 0 ? concurrency : defaults.concurrency,
    baseUrl,
  };
}

async function preflightMetrics(baseUrl: string): Promise<void> {
  try {
    const res = await axios.get<string>(`${baseUrl}/metrics`, {
      timeout: 10_000,
      validateStatus: () => true,
    });

    if (res.status !== 200) {
      // eslint-disable-next-line no-console
      console.warn(
        `[orchestrator-load-smoke] /metrics returned status ${res.status} – metrics inspection skipped`
      );
      return;
    }

    const body = res.data;
    const hasRolloutMetric = body.includes('ringrift_orchestrator_rollout_percentage');
    const rolloutLine = body
      .split('\n')
      .find((line) => line.startsWith('ringrift_orchestrator_rollout_percentage'));

    if (!hasRolloutMetric) {
      // eslint-disable-next-line no-console
      console.warn(
        '[orchestrator-load-smoke] Orchestrator rollout metric not found in /metrics. ' +
          'Ensure MetricsService is initialised and orchestrator metrics are registered.'
      );
      return;
    }

    if (rolloutLine) {
      // Metric line is of the form: name{labels} value
      const parts = rolloutLine.trim().split(/\s+/);
      const value = parts[parts.length - 1];
      // eslint-disable-next-line no-console
      console.log(
        `[orchestrator-load-smoke] Detected ringrift_orchestrator_rollout_percentage sample: ${value}`
      );
    } else {
      // eslint-disable-next-line no-console
      console.log(
        '[orchestrator-load-smoke] ringrift_orchestrator_rollout_percentage metric present in /metrics'
      );
    }
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn(
      `[orchestrator-load-smoke] Failed to query /metrics at ${baseUrl}/metrics: ${
        (error as Error).message ?? String(error)
      }`
    );
  }
}

async function runUserScenario(
  index: number,
  baseUrl: string,
  latencySamples: LatencySample[]
): Promise<void> {
  const api = axios.create({
    baseURL: `${baseUrl}/api`,
    timeout: 15_000,
    validateStatus: () => true,
  });

  const uniqueSuffix = `${Date.now()}_${index}_${Math.floor(Math.random() * 10_000)}`;
  const email = `orch_smoke_${uniqueSuffix}@example.com`;
  const username = `orch_smoke_${uniqueSuffix}`;
  const password = 'OrchSmoke123!';

  // 1. Register user
  const t0 = performance.now();
  const registerRes = await api.post('/auth/register', {
    email,
    username,
    password,
  });
  const t1 = performance.now();
  latencySamples.push({ label: 'auth_register', ms: t1 - t0 });

  if (registerRes.status !== 201 || !registerRes.data?.success) {
    throw new Error(
      `Register failed (status=${registerRes.status}): ${JSON.stringify(registerRes.data)}`
    );
  }

  const accessToken: string | undefined = registerRes.data?.data?.accessToken;
  const userId: string | undefined = registerRes.data?.data?.user?.id;

  if (!accessToken || !userId) {
    throw new Error('Register response missing accessToken or user.id');
  }

  const authHeaders = {
    Authorization: `Bearer ${accessToken}`,
  };

  // 2. Create a simple game
  const t2 = performance.now();
  const createRes = await api.post(
    '/games',
    {
      boardType: 'square8',
      timeControl: { initialTime: 300, increment: 0 },
      // Rely on server defaults for isRated/maxPlayers; keep payload minimal.
    },
    { headers: authHeaders }
  );
  const t3 = performance.now();
  latencySamples.push({ label: 'games_create', ms: t3 - t2 });

  if (createRes.status !== 201 || !createRes.data?.success) {
    throw new Error(
      `Game create failed (status=${createRes.status}): ${JSON.stringify(createRes.data)}`
    );
  }

  const gameId: string | undefined = createRes.data?.data?.game?.id;
  if (!gameId) {
    throw new Error('Game create response missing data.game.id');
  }

  // 3. Fetch games list for the user (exercise read path)
  const t4 = performance.now();
  const listRes = await api.get('/games', {
    headers: authHeaders,
    params: { status: 'waiting' },
  });
  const t5 = performance.now();
  latencySamples.push({ label: 'games_list', ms: t5 - t4 });

  if (listRes.status !== 200 || !listRes.data?.success) {
    throw new Error(
      `Game list failed (status=${listRes.status}): ${JSON.stringify(listRes.data)}`
    );
  }

  // 4. Fetch game details once
  const t6 = performance.now();
  const detailsRes = await api.get(`/games/${gameId}`, { headers: authHeaders });
  const t7 = performance.now();
  latencySamples.push({ label: 'games_details', ms: t7 - t6 });

  if (detailsRes.status !== 200 || !detailsRes.data?.success) {
    throw new Error(
      `Game details failed (status=${detailsRes.status}): ${JSON.stringify(detailsRes.data)}`
    );
  }
}

function summarizeLatencies(samples: LatencySample[]): void {
  if (samples.length === 0) {
    // eslint-disable-next-line no-console
    console.log('[orchestrator-load-smoke] No latency samples recorded.');
    return;
  }

  const byLabel = new Map<string, number[]>();
  for (const s of samples) {
    const arr = byLabel.get(s.label) ?? [];
    arr.push(s.ms);
    byLabel.set(s.label, arr);
  }

  // eslint-disable-next-line no-console
  console.log('\n[orchestrator-load-smoke] Latency summary (ms):');
  for (const [label, values] of byLabel.entries()) {
    const sorted = [...values].sort((a, b) => a - b);
    const count = sorted.length;
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    const p50 = sorted[Math.floor(0.5 * (count - 1))];
    const p90 = sorted[Math.floor(0.9 * (count - 1))];
    const p99 = sorted[Math.floor(0.99 * (count - 1))];
    const avg = sorted.reduce((acc, v) => acc + v, 0) / count;

    // eslint-disable-next-line no-console
    console.log(
      `  ${label.padEnd(14)} count=${String(count).padStart(3)} ` +
        `min=${min.toFixed(1)} avg=${avg.toFixed(1)} ` +
        `p50=${p50.toFixed(1)} p90=${p90.toFixed(1)} p99=${p99.toFixed(1)} max=${max.toFixed(1)}`
    );
  }
}

async function main() {
  const args = parseArgs(process.argv);

  // eslint-disable-next-line no-console
  console.log(
    `[orchestrator-load-smoke] Starting HTTP load smoke: games=${args.games}, concurrency=${args.concurrency}, baseUrl=${args.baseUrl}`
  );

  await preflightMetrics(args.baseUrl);

  const totalGames = args.games;
  const concurrency = args.concurrency;

  let nextIndex = 0;
  let successCount = 0;
  let failureCount = 0;
  const latencySamples: LatencySample[] = [];

  async function worker(workerId: number): Promise<void> {
    while (true) {
      const index = nextIndex;
      nextIndex += 1;
      if (index >= totalGames) break;

      try {
        // eslint-disable-next-line no-console
        console.log(
          `[orchestrator-load-smoke] Worker ${workerId} running scenario ${index + 1}/${totalGames}`
        );
        await runUserScenario(index, args.baseUrl, latencySamples);
        successCount += 1;
      } catch (error) {
        failureCount += 1;
        // eslint-disable-next-line no-console
        console.error(
          `[orchestrator-load-smoke] Scenario ${index + 1} failed: ${
            (error as Error).message ?? String(error)
          }`
        );
      }
    }
  }

  const workers: Promise<void>[] = [];
  for (let i = 0; i < concurrency; i += 1) {
    workers.push(worker(i + 1));
  }

  await Promise.all(workers);

  // eslint-disable-next-line no-console
  console.log(
    `\n[orchestrator-load-smoke] Completed. success=${successCount}, failed=${failureCount}, total=${totalGames}`
  );

  summarizeLatencies(latencySamples);

  if (failureCount > 0) {
    process.exitCode = 1;
  }
}

// eslint-disable-next-line unicorn/prefer-top-level-await
main().catch((error) => {
  // eslint-disable-next-line no-console
  console.error('[orchestrator-load-smoke] Unexpected error:', error);
  process.exitCode = 1;
});

