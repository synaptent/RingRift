import express from 'express';
import request from 'supertest';
import client from 'prom-client';
import setupRoutes from '../../src/server/routes';
import { requestContext } from '../../src/server/middleware/requestContext';
import { logger, httpLogger } from '../../src/server/utils/logger';
import { errorHandler } from '../../src/server/middleware/errorHandler';
import '../../src/server/utils/rulesParityMetrics';
import * as rateLimiterModule from '../../src/server/middleware/rateLimiter';

// Minimal Prisma-like stub used for protected-game route tests.
// We only implement the subset touched by src/server/routes/game.ts in
// this file (game.findUnique / move.findMany). The variable name is
// prefixed with "mock" so Jest's module factory scoping rules allow it
// to be referenced from jest.mock() factories.
const mockPrisma = {
  game: {
    findUnique: jest.fn(),
    findMany: jest.fn(),
    count: jest.fn(),
    create: jest.fn(),
  },
  move: {
    findMany: jest.fn(),
  },
} as any;

// Wire the stub into the database connection module for this test file.
jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => mockPrisma,
}));

// Stub the game-creation quota helper so tests can simulate in-quota and
// over-quota behaviour without requiring Redis or real rate-limiter state.
jest.mock('../../src/server/middleware/rateLimiter', () => {
  const actual = jest.requireActual('../../src/server/middleware/rateLimiter');
  return {
    ...actual,
    consumeRateLimit: jest.fn(),
  };
});

// Convenience alias for the mocked consumeRateLimit so tests can control
// and assert quota behaviour.
const mockConsumeRateLimit = rateLimiterModule.consumeRateLimit as jest.MockedFunction<any>;

// Stub authenticate() so tests can simulate authenticated and
// unauthenticated requests without relying on real JWTs.
jest.mock('../../src/server/middleware/auth', () => {
  const actual = jest.requireActual('../../src/server/middleware/auth');
  const { createError } = jest.requireActual('../../src/server/middleware/errorHandler');

  const authenticate = (req: any, _res: any, next: any) => {
    const authHeader = req.headers['authorization'] as string | undefined;
    if (!authHeader) {
      return next(createError('Authentication token required', 401, 'TOKEN_REQUIRED'));
    }

    const [, token] = authHeader.split(' ');
    const userId = token || 'user-1';

    req.user = {
      id: userId,
      email: `${userId}@example.com`,
      username: userId,
      role: 'USER',
    };

    return next();
  };

  return {
    ...actual,
    authenticate,
  };
});

/**
 * Lightweight server harness for testing the public HTTP surface:
 * - /health
 * - /api (root API info endpoint)
 *
 * We intentionally avoid importing src/server/index.ts here to keep
 * tests isolated from real DB/Redis connections and actual network
 * listeners. Instead we mirror the minimal wiring from index.ts that
 * is relevant for these endpoints.
 */

function createTestApp() {
  const app = express();
  app.use(express.json());
  // Mirror the main server pipeline by attaching per-request context so that
  // downstream logging helpers (httpLogger / withRequestContext) can include
  // a correlation id.
  app.use(requestContext as any);

  // Health check endpoint (lightweight mirror of src/server/index.ts)
  app.get('/health', (_req, res) => {
    res.status(200).json({
      status: 'healthy',
      // Use concrete runtime values; the test asserts only on types and
      // the presence of these fields, not their exact values.
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: process.env.npm_package_version || 'test',
    });
  });

  // Prometheus metrics endpoint (lightweight mirror of src/server/index.ts).
  // This uses the shared default registry from prom-client so that metrics
  // declared in src/server/utils/rulesParityMetrics.ts are exposed.
  app.get('/metrics', async (_req, res) => {
    res.set('Content-Type', client.register.contentType);
    const metrics = await client.register.metrics();
    res.send(metrics);
  });

  // API routes mounted at /api
  app.use('/api', setupRoutes());
  // Attach global error handler so that route errors (including auth /
  // authorization failures) are rendered using the standard JSON shape.
  app.use(errorHandler as any);

  return app;
}

describe('Server health and API info routes', () => {
  it('GET /health responds with healthy status and basic metadata', async () => {
    const app = createTestApp();

    const res = await request(app).get('/health').expect(200);

    expect(res.body).toMatchObject({
      status: 'healthy',
    });

    // Ensure the shape matches the contract without asserting exact values.
    expect(typeof res.body.timestamp).toBe('string');
    expect(typeof res.body.uptime).toBe('number');
    expect(typeof res.body.version).toBe('string');
  });

  it('GET /api returns API metadata and endpoint map', async () => {
    const app = createTestApp();

    const res = await request(app).get('/api').expect(200);

    expect(res.body).toMatchObject({
      success: true,
      message: 'RingRift API',
      version: expect.any(String),
      endpoints: {
        auth: '/api/auth',
        games: '/api/games',
        users: '/api/users',
      },
    });

    expect(typeof res.body.timestamp).toBe('string');
  });

  it('GET /metrics exposes core Prometheus metrics', async () => {
    const app = createTestApp();

    const res = await request(app).get('/metrics').expect(200);
    const body = res.text;

    expect(body).toContain('ai_move_latency_ms');
    expect(body).toContain('ai_fallback_total');
    expect(body).toContain('game_move_latency_ms');
    expect(body).toContain('websocket_connections_current');
  });

  it('requestContext + httpLogger attach requestId to log metadata', async () => {
    const app = express();
    app.use(express.json());
    app.use(requestContext as any);

    app.get('/test-log', (req, res) => {
      httpLogger.info(req, 'test_request_log');
      res.json({ ok: true });
    });

    const infoSpy = jest.spyOn(logger, 'info');

    const REQUEST_ID = 'test-request-id-123';
    await request(app).get('/test-log').set('X-Request-Id', REQUEST_ID).expect(200);

    expect(infoSpy).toHaveBeenCalled();

    const lastCall = infoSpy.mock.calls[infoSpy.mock.calls.length - 1] as any[];
    // Winston logger.info may be invoked as (infoObject) or (message, meta)
    const metaOrInfo = lastCall[lastCall.length - 1] as any;

    expect(metaOrInfo).toBeDefined();
    expect(metaOrInfo.requestId).toBe(REQUEST_ID);

    infoSpy.mockRestore();
  });
});

describe('Protected game route authorization', () => {
  beforeEach(() => {
    mockPrisma.game.findUnique.mockReset();
    mockPrisma.move.findMany.mockReset();
  });

  it('GET /api/games/:gameId returns 401 TOKEN_REQUIRED when unauthenticated', async () => {
    const app = createTestApp();

    const res = await request(app).get('/api/games/game-1').expect(401);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('TOKEN_REQUIRED');
  });

  it('GET /api/games/:gameId/moves returns 401 TOKEN_REQUIRED when unauthenticated', async () => {
    const app = createTestApp();

    const res = await request(app).get('/api/games/game-1/moves').expect(401);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('TOKEN_REQUIRED');
  });

  it('GET /api/games/:gameId denies access to non-participants when spectators are disabled', async () => {
    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: 'game-1',
      player1Id: 'other-user',
      player2Id: null,
      player3Id: null,
      player4Id: null,
      allowSpectators: false,
      moves: [],
      player1: { id: 'other-user', username: 'Owner', rating: 1500 },
      player2: null,
      player3: null,
      player4: null,
    } as any);

    const app = createTestApp();

    const res = await request(app)
      .get('/api/games/game-1')
      .set('Authorization', 'Bearer user-1')
      .expect(403);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('ACCESS_DENIED');
  });

  it('GET /api/games/:gameId/moves reuses the same participant-or-spectator invariant', async () => {
    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: 'game-1',
      player1Id: 'other-user',
      player2Id: null,
      player3Id: null,
      player4Id: null,
      allowSpectators: false,
    } as any);

    const app = createTestApp();

    const res = await request(app)
      .get('/api/games/game-1/moves')
      .set('Authorization', 'Bearer user-1')
      .expect(403);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('ACCESS_DENIED');
  });

  it('POST /api/games/:gameId/leave denies non-participants with ACCESS_DENIED', async () => {
    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: 'game-1',
      player1Id: 'other-user',
      player2Id: null,
      player3Id: null,
      player4Id: null,
      status: 'waiting',
    } as any);

    const app = createTestApp();

    const res = await request(app)
      .post('/api/games/game-1/leave')
      .set('Authorization', 'Bearer user-1')
      .expect(403);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('ACCESS_DENIED');
  });
});

describe('Game creation quotas', () => {
  beforeEach(() => {
    mockPrisma.game.create.mockReset();
    mockConsumeRateLimit.mockReset();
  });

  it('allows POST /api/games when per-user and per-IP quotas allow', async () => {
    mockConsumeRateLimit.mockResolvedValue({ allowed: true });

    const now = new Date();

    mockPrisma.game.create.mockResolvedValue({
      id: 'game-1',
      boardType: 'square8',
      maxPlayers: 2,
      timeControl: { initialTime: 300, increment: 0 },
      isRated: true,
      allowSpectators: true,
      player1Id: 'user-1',
      status: 'waiting',
      gameState: {},
      rngSeed: null,
      createdAt: now,
      updatedAt: now,
      player1: { id: 'user-1', username: 'user-1', rating: 1500 },
    } as any);

    const app = createTestApp();

    const res = await request(app)
      .post('/api/games')
      .set('Authorization', 'Bearer user-1')
      .set('X-Forwarded-For', '203.0.113.1')
      .send({
        boardType: 'square8',
        timeControl: { initialTime: 300, increment: 0 },
      })
      .expect(201);

    expect(res.body.success).toBe(true);
    expect(mockPrisma.game.create).toHaveBeenCalledTimes(1);

    // First quota is per-user, second is per-IP.
    expect(mockConsumeRateLimit).toHaveBeenCalledWith('gameCreateUser', 'user-1');
    expect(mockConsumeRateLimit).toHaveBeenCalledWith('gameCreateIp', '203.0.113.1');
  });

  it('returns 429 GAME_CREATE_RATE_LIMITED when per-user quota is exceeded', async () => {
    mockConsumeRateLimit.mockImplementation(async (limiterKey: string) => {
      if (limiterKey === 'gameCreateUser') {
        return { allowed: false, retryAfter: 60 };
      }
      return { allowed: true };
    });

    const app = createTestApp();

    const res = await request(app)
      .post('/api/games')
      .set('Authorization', 'Bearer user-1')
      .set('X-Forwarded-For', '203.0.113.1')
      .send({
        boardType: 'square8',
        timeControl: { initialTime: 300, increment: 0 },
      })
      .expect(429);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('GAME_CREATE_RATE_LIMITED');
    expect(mockPrisma.game.create).not.toHaveBeenCalled();
  });
});

describe('Protected user routes', () => {
  it('DELETE /api/users/me returns 401 TOKEN_REQUIRED when unauthenticated', async () => {
    const app = createTestApp();

    const res = await request(app).delete('/api/users/me').expect(401);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('TOKEN_REQUIRED');
  });
});
