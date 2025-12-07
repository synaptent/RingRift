#!/usr/bin/env ts-node
/**
 * Game session load/coordination smoke harness.
 * ============================================================================
 *
 * This script performs a small multi-game run against a running RingRift
 * backend, exercising:
 * - HTTP surfaces (/api/auth, /api/games, /api/games/:id/join)
 * - WebSocket game sessions for:
 *   - PvP games (2 human players)
 *   - Human vs AI games (1 human + AI opponents)
 *   - Optional spectators joining via WebSocket
 * - Basic reconnect behaviour (disconnect + reconnect for one player)
 *
 * It is intended as a lightweight “production preview” harness to run while
 * watching the Grafana Game Performance / System Health dashboards, not as a
 * full benchmark.
 *
 * Usage (from repo root, with server running on localhost:3000):
 *
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/game-session-load-smoke.ts
 *
 * Options:
 *   --games=<number>        Total number of games to create (default: 6)
 *   --concurrency=<number>  Number of concurrent workers (default: 3)
 *   --baseUrl=<url>         Base URL of the API (default: http://localhost:3000)
 *   --aiFraction=<number>   Fraction of games that should be human vs AI (0–1, default: 0.5)
 *
 * Notes:
 * - This harness reuses the same CreateGameRequest shape the lobby does:
 *   PvP games mirror the lobby defaults; AI games mirror the AI lobby path.
 * - For deeper rules coverage and invariants, use:
 *     npm run soak:orchestrator:smoke
 *     npm run test:orchestrator:s-invariant
 */

import axios, { type AxiosRequestConfig } from 'axios';
import { performance } from 'perf_hooks';
import { io, Socket } from 'socket.io-client';
import type { CreateGameRequest } from '../src/shared/types/game';

type LoadArgs = {
  games: number;
  concurrency: number;
  baseUrl: string;
  aiFraction: number;
};

type LatencySample = {
  label: string;
  ms: number;
};

type User = {
  index: number;
  email: string;
  username: string;
  password: string;
  accessToken: string;
};

type GameKind = 'pvp' | 'ai';

type GameScenarioResult = {
  kind: GameKind;
  gameId: string;
};

function parseArgs(argv: string[]): LoadArgs {
  const defaults: LoadArgs = {
    games: 6,
    concurrency: 3,
    baseUrl: 'http://localhost:3000',
    aiFraction: 0.5,
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

  const gamesRaw = args.games ? Number(args.games) : defaults.games;
  const concurrencyRaw = args.concurrency ? Number(args.concurrency) : defaults.concurrency;
  const aiFractionRaw = args.aiFraction ? Number(args.aiFraction) : defaults.aiFraction;

  const games = Number.isFinite(gamesRaw) && gamesRaw > 0 ? gamesRaw : defaults.games;
  const concurrency =
    Number.isFinite(concurrencyRaw) && concurrencyRaw > 0 ? concurrencyRaw : defaults.concurrency;
  const aiFraction =
    Number.isFinite(aiFractionRaw) && aiFractionRaw >= 0 && aiFractionRaw <= 1
      ? aiFractionRaw
      : defaults.aiFraction;

  const baseUrl = (args.baseUrl as string | undefined) ?? defaults.baseUrl;

  return {
    games,
    concurrency,
    baseUrl,
    aiFraction,
  };
}

function getApiClient(baseUrl: string, accessToken?: string) {
  const config: AxiosRequestConfig = {
    baseURL: `${baseUrl.replace(/\/$/, '')}/api`,
    timeout: 20_000,
    validateStatus: () => true,
  };

  if (accessToken) {
    config.headers = { Authorization: `Bearer ${accessToken}` };
  }

  return axios.create(config);
}

async function registerUser(baseUrl: string, index: number): Promise<User> {
  const api = getApiClient(baseUrl);
  const uniqueSuffix = `${Date.now()}_${index}_${Math.floor(Math.random() * 10_000)}`;
  const email = `load_user_${uniqueSuffix}@example.com`;
  const username = `load_user_${uniqueSuffix}`;
  const password = 'LoadHarness123!';

  const res = await api.post('/auth/register', {
    email,
    username,
    password,
  });

  if (res.status !== 201 || !res.data?.success) {
    throw new Error(
      `registerUser(${index}) failed (status=${res.status}): ${JSON.stringify(res.data)}`
    );
  }

  const accessToken: string | undefined = res.data?.data?.accessToken;
  if (!accessToken) {
    throw new Error(`registerUser(${index}) missing accessToken in response`);
  }

  return { index, email, username, password, accessToken };
}

async function createPvpGame(baseUrl: string, creator: User): Promise<GameScenarioResult> {
  const api = getApiClient(baseUrl, creator.accessToken);

  const payload: CreateGameRequest = {
    boardType: 'square8',
    maxPlayers: 2,
    isRated: false,
    isPrivate: true,
    timeControl: { type: 'rapid', initialTime: 300, increment: 0 },
    rulesOptions: { swapRuleEnabled: true },
  };

  const res = await api.post('/games', payload);
  if (res.status !== 201 || !res.data?.success) {
    throw new Error(`createPvpGame failed: status=${res.status} body=${JSON.stringify(res.data)}`);
  }

  const gameId: string | undefined = res.data?.data?.game?.id;
  if (!gameId) {
    throw new Error('createPvpGame missing data.game.id');
  }

  return { kind: 'pvp', gameId };
}

async function joinPvpGame(baseUrl: string, joiner: User, gameId: string): Promise<void> {
  const api = getApiClient(baseUrl, joiner.accessToken);
  const res = await api.post(`/games/${gameId}/join`);
  if (res.status !== 200 && res.status !== 201) {
    throw new Error(`joinPvpGame failed: status=${res.status} body=${JSON.stringify(res.data)}`);
  }
}

async function createAiGame(baseUrl: string, creator: User): Promise<GameScenarioResult> {
  const api = getApiClient(baseUrl, creator.accessToken);

  const payload: CreateGameRequest = {
    boardType: 'square8',
    maxPlayers: 2,
    isRated: false, // AI games must be unrated
    isPrivate: true,
    timeControl: { type: 'rapid', initialTime: 300, increment: 0 },
    aiOpponents: {
      count: 1,
      difficulty: [3],
      mode: 'service',
      aiType: 'heuristic',
    },
    rulesOptions: { swapRuleEnabled: true },
  };

  const res = await api.post('/games', payload);
  if (res.status !== 201 || !res.data?.success) {
    throw new Error(`createAiGame failed: status=${res.status} body=${JSON.stringify(res.data)}`);
  }

  const gameId: string | undefined = res.data?.data?.game?.id;
  if (!gameId) {
    throw new Error('createAiGame missing data.game.id');
  }

  return { kind: 'ai', gameId };
}

type SocketClient = {
  socket: Socket;
  label: string;
};

async function openGameSocket(
  baseUrl: string,
  accessToken: string,
  gameId: string,
  label: string
): Promise<SocketClient> {
  const url = baseUrl.replace(/\/$/, '');

  return await new Promise<SocketClient>((resolve, reject) => {
    const socket: Socket = io(url, {
      transports: ['websocket', 'polling'],
      auth: { token: accessToken },
      autoConnect: false,
      reconnection: true,
      reconnectionAttempts: 3,
    });

    const timeoutId = setTimeout(() => {
      socket.disconnect();
      reject(new Error(`[${label}] WebSocket connection timeout`));
    }, 10_000);

    socket.on('connect', () => {
      socket.emit('join_game', { gameId });
      clearTimeout(timeoutId);
      // eslint-disable-next-line no-console
      console.log(`[game-session-load-smoke] ${label} connected to game ${gameId}`);
      resolve({ socket, label });
    });

    socket.on('connect_error', (err: Error) => {
      clearTimeout(timeoutId);

      console.warn(
        `[game-session-load-smoke] ${label} WebSocket connect_error: ${err.message || String(err)}`
      );
      reject(err);
    });

    socket.on('game_over', () => {
      // eslint-disable-next-line no-console
      console.log(`[game-session-load-smoke] ${label} received game_over for ${gameId}`);
    });

    socket.connect();
  });
}

async function simulateReconnect(client: SocketClient, delayMs: number): Promise<void> {
  // eslint-disable-next-line no-console
  console.log(`[game-session-load-smoke] ${client.label} simulating reconnect...`);

  client.socket.disconnect();
  await new Promise((resolve) => setTimeout(resolve, delayMs));
  client.socket.connect();
}

async function runGameScenario(
  index: number,
  args: LoadArgs,
  users: User[],
  latency: LatencySample[]
): Promise<void> {
  const baseUrl = args.baseUrl;

  // Choose scenario kind based on aiFraction
  const rand = Math.random();
  const kind: GameKind = rand < args.aiFraction ? 'ai' : 'pvp';

  const t0 = performance.now();
  if (kind === 'pvp') {
    if (users.length < 3) {
      // Make sure we have at least 3 users (two players + spectator slot)
      const needed = 3 - users.length;
      for (let i = 0; i < needed; i += 1) {
        users.push(await registerUser(baseUrl, users.length));
      }
    }

    const creator = users[index % users.length];
    const joiner = users[(index + 1) % users.length];
    const spectator = users[(index + 2) % users.length];

    const scenario = await createPvpGame(baseUrl, creator);
    await joinPvpGame(baseUrl, joiner, scenario.gameId);

    const t1 = performance.now();
    latency.push({ label: 'pvp_create_and_join', ms: t1 - t0 });

    // Open WebSocket connections for both players and one spectator.
    const [p1Client, p2Client, specClient] = await Promise.all([
      openGameSocket(baseUrl, creator.accessToken, scenario.gameId, `p1-${index}`),
      openGameSocket(baseUrl, joiner.accessToken, scenario.gameId, `p2-${index}`),
      openGameSocket(baseUrl, spectator.accessToken, scenario.gameId, `spec-${index}`),
    ]);

    // Simulate a brief reconnect for player 2.
    await simulateReconnect(p2Client, 500);

    // Let the sockets sit briefly to generate heartbeats and potential decisions.
    await new Promise((resolve) => setTimeout(resolve, 1_000));

    p1Client.socket.disconnect();
    p2Client.socket.disconnect();
    specClient.socket.disconnect();
  } else {
    // AI game: one human + AI opponents
    if (users.length < 1) {
      users.push(await registerUser(baseUrl, users.length));
    }
    const creator = users[index % users.length];

    const scenario = await createAiGame(baseUrl, creator);
    const t1 = performance.now();
    latency.push({ label: 'ai_create', ms: t1 - t0 });

    const client = await openGameSocket(
      baseUrl,
      creator.accessToken,
      scenario.gameId,
      `ai-${index}`
    );

    // Let the AI play for a short window; we don't wait for full completion.
    await new Promise((resolve) => setTimeout(resolve, 1_000));
    client.socket.disconnect();
  }
}

function summarizeLatencies(samples: LatencySample[]): void {
  if (samples.length === 0) {
    // eslint-disable-next-line no-console
    console.log('[game-session-load-smoke] No latency samples recorded.');
    return;
  }

  const byLabel = new Map<string, number[]>();
  for (const s of samples) {
    const list = byLabel.get(s.label) ?? [];
    list.push(s.ms);
    byLabel.set(s.label, list);
  }

  // eslint-disable-next-line no-console
  console.log('\n[game-session-load-smoke] Latency summary (ms):');
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
      `  ${label.padEnd(22)} count=${String(count).padStart(3)} ` +
        `min=${min.toFixed(1)} avg=${avg.toFixed(1)} ` +
        `p50=${p50.toFixed(1)} p90=${p90.toFixed(1)} ` +
        `p99=${p99.toFixed(1)} max=${max.toFixed(1)}`
    );
  }
}

async function main() {
  const args = parseArgs(process.argv);

  // eslint-disable-next-line no-console
  console.log(
    `[game-session-load-smoke] Starting game-session load smoke: games=${args.games}, ` +
      `concurrency=${args.concurrency}, aiFraction=${args.aiFraction}, baseUrl=${args.baseUrl}`
  );

  const totalGames = args.games;
  const concurrency = args.concurrency;

  let nextIndex = 0;
  let successCount = 0;
  let failureCount = 0;
  const latencySamples: LatencySample[] = [];
  const users: User[] = [];

  async function worker(workerId: number): Promise<void> {
    while (true) {
      const index = nextIndex;
      nextIndex += 1;
      if (index >= totalGames) break;

      try {
        // eslint-disable-next-line no-console
        console.log(
          `[game-session-load-smoke] Worker ${workerId} running scenario ${index + 1}/${totalGames}`
        );
        await runGameScenario(index, args, users, latencySamples);
        successCount += 1;
      } catch (error) {
        failureCount += 1;

        console.error(
          `[game-session-load-smoke] Scenario ${index + 1} failed: ${
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
    `\n[game-session-load-smoke] Completed. success=${successCount}, failed=${failureCount}, total=${totalGames}`
  );

  summarizeLatencies(latencySamples);

  if (failureCount > 0) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error('[game-session-load-smoke] Unexpected error:', error);
  process.exitCode = 1;
});
