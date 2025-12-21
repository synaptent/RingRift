import express from 'express';
import request from 'supertest';
import client from 'prom-client';
import setupRoutes from '../../src/server/routes';
import { requestContext } from '../../src/server/middleware/requestContext';
import { logger, httpLogger } from '../../src/server/utils/logger';
import { errorHandler } from '../../src/server/middleware/errorHandler';
import '../../src/server/utils/rulesParityMetrics';
import * as rateLimiterModule from '../../src/server/middleware/rateLimiter';
import {
  HealthCheckService,
  isServiceReady,
  HealthCheckResponse,
} from '../../src/server/services/HealthCheckService';
import { RatingService } from '../../src/server/services/RatingService';

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
    update: jest.fn(),
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

// Stub RatingService so we can assert when HTTP leave/resign flows do (or do not)
// trigger rating updates. The engine/session-level rating semantics are covered
// in dedicated GamePersistenceService and RatingService tests; here we only care
// that the HTTP layer does not bypass or duplicate those updates.
jest.mock('../../src/server/services/RatingService', () => ({
  RatingService: {
    processGameResult: jest.fn(),
  },
}));

const mockProcessGameResult = RatingService.processGameResult as jest.MockedFunction<any>;

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

function createTestApp(wsServer?: any) {
  const app = express();
  app.use(express.json());
  // Mirror the main server pipeline by attaching per-request context so that
  // downstream logging helpers (httpLogger / withRequestContext) can include
  // a correlation id.
  app.use(requestContext as any);

  // Health check endpoints - liveness probe (mirrors src/server/index.ts)
  // These are placed BEFORE any rate limiting in the real server.
  app.get(['/health', '/healthz'], (_req, res) => {
    const status = HealthCheckService.getLivenessStatus();
    res.status(200).json(status);
  });

  // Readiness probe endpoint (mirrors src/server/index.ts)
  app.get(['/ready', '/readyz'], async (_req, res) => {
    const status = await HealthCheckService.getReadinessStatus();
    const httpStatus = isServiceReady(status) ? 200 : 503;
    res.status(httpStatus).json(status);
  });

  // Prometheus metrics endpoint (lightweight mirror of src/server/index.ts).
  // This uses the shared default registry from prom-client so that metrics
  // declared in src/server/utils/rulesParityMetrics.ts are exposed.
  app.get('/metrics', async (_req, res) => {
    res.set('Content-Type', client.register.contentType);
    const metrics = await client.register.metrics();
    res.send(metrics);
  });

  // API routes mounted at /api. Allow an optional WebSocketServer-like
  // instance to be injected so routes that depend on it (e.g. game
  // diagnostics) can be exercised in isolation.
  app.use('/api', setupRoutes(wsServer));
  // Attach global error handler so that route errors (including auth /
  // authorization failures) are rendered using the standard JSON shape.
  app.use(errorHandler as any);

  return app;
}

describe('Server health and API info routes', () => {
  describe('Liveness probe endpoints', () => {
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
      // Liveness should NOT include detailed checks (to be fast)
      expect(res.body.checks).toBeUndefined();
    });

    it('GET /healthz responds identically to /health (Kubernetes convention)', async () => {
      const app = createTestApp();

      const res = await request(app).get('/healthz').expect(200);

      expect(res.body).toMatchObject({
        status: 'healthy',
      });
      expect(typeof res.body.timestamp).toBe('string');
      expect(typeof res.body.uptime).toBe('number');
      expect(typeof res.body.version).toBe('string');
    });
  });

  describe('Readiness probe endpoints', () => {
    it('GET /ready responds with health status and dependency checks', async () => {
      const app = createTestApp();

      const res = await request(app).get('/ready');

      // Status code depends on actual dependency state - just verify structure
      expect([200, 503]).toContain(res.status);

      expect(res.body).toMatchObject({
        status: expect.stringMatching(/^(healthy|degraded|unhealthy)$/),
      });

      expect(typeof res.body.timestamp).toBe('string');
      expect(typeof res.body.uptime).toBe('number');
      expect(typeof res.body.version).toBe('string');
      // Readiness SHOULD include detailed checks
      expect(res.body.checks).toBeDefined();
    });

    it('GET /readyz responds identically to /ready (Kubernetes convention)', async () => {
      const app = createTestApp();

      const res = await request(app).get('/readyz');

      expect([200, 503]).toContain(res.status);
      expect(res.body).toMatchObject({
        status: expect.stringMatching(/^(healthy|degraded|unhealthy)$/),
      });
      expect(res.body.checks).toBeDefined();
    });

    it('returns 503 when service is unhealthy', async () => {
      // Create an unhealthy response mock
      const unhealthyResponse: HealthCheckResponse = {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        uptime: 100,
        checks: {
          database: { status: 'unhealthy', error: 'Connection refused' },
        },
      };

      // Verify isServiceReady returns false for unhealthy
      expect(isServiceReady(unhealthyResponse)).toBe(false);
    });

    it('returns 200 when service is degraded (can still serve traffic)', async () => {
      // Create a degraded response mock
      const degradedResponse: HealthCheckResponse = {
        status: 'degraded',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        uptime: 100,
        checks: {
          database: { status: 'healthy', latency: 5 },
          redis: { status: 'degraded', error: 'Not connected' },
        },
      };

      // Verify isServiceReady returns true for degraded
      expect(isServiceReady(degradedResponse)).toBe(true);
    });

    it('GET /ready returns 503 when HealthCheckService reports unhealthy', async () => {
      const unhealthyResponse: HealthCheckResponse = {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0-test',
        uptime: 42,
        checks: {
          database: { status: 'unhealthy', error: 'Connection refused' },
        },
      };

      const readinessSpy = jest
        .spyOn(HealthCheckService, 'getReadinessStatus')
        .mockResolvedValue(unhealthyResponse);

      const app = createTestApp();

      const res = await request(app).get('/ready').expect(503);

      expect(res.body.status).toBe('unhealthy');
      expect(res.body.checks?.database?.status).toBe('unhealthy');
      expect(res.body.checks?.database?.error).toBe('Connection refused');

      readinessSpy.mockRestore();
    });

    it('GET /ready returns 200 when HealthCheckService reports degraded (database still healthy)', async () => {
      const degradedResponse: HealthCheckResponse = {
        status: 'degraded',
        timestamp: new Date().toISOString(),
        version: '1.0.0-test',
        uptime: 42,
        checks: {
          database: { status: 'healthy', latency: 5 },
          redis: { status: 'degraded', error: 'Redis client not connected' },
        },
      };

      const readinessSpy = jest
        .spyOn(HealthCheckService, 'getReadinessStatus')
        .mockResolvedValue(degradedResponse);

      const app = createTestApp();

      const res = await request(app).get('/ready').expect(200);

      expect(res.body.status).toBe('degraded');
      expect(res.body.checks?.database?.status).toBe('healthy');
      expect(res.body.checks?.redis?.status).toBe('degraded');

      readinessSpy.mockRestore();
    });
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
    mockPrisma.game.update.mockReset();
    mockPrisma.move.findMany.mockReset();
    mockProcessGameResult.mockReset();
  });

  it('GET /api/games/:gameId returns 401 TOKEN_REQUIRED when unauthenticated', async () => {
    const app = createTestApp();

    const res = await request(app).get('/api/games/game-1').expect(401);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('AUTH_TOKEN_REQUIRED');
  });

  it('GET /api/games/:gameId/moves returns 401 TOKEN_REQUIRED when unauthenticated', async () => {
    const app = createTestApp();

    const res = await request(app).get('/api/games/game-1/moves').expect(401);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('AUTH_TOKEN_REQUIRED');
  });

  it('POST /api/games/:gameId/moves returns 404 RESOURCE_ROUTE_NOT_FOUND when HTTP move harness is disabled', async () => {
    const app = createTestApp();

    const res = await request(app)
      .post('/api/games/550e8400-e29b-41d4-a716-446655440099/moves')
      .set('Authorization', 'Bearer user-1')
      .send({
        moveType: 'place_ring',
        position: {
          to: { x: 3, y: 3 },
        },
      })
      .expect(404);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('RESOURCE_ROUTE_NOT_FOUND');
    // Harness should be fully dark when disabled – no DB lookups for the game.
    expect(mockPrisma.game.findUnique).not.toHaveBeenCalled();
  });

  it('GET /api/games/:gameId returns 404 GAME_NOT_FOUND for non-existent string gameId', async () => {
    const app = createTestApp();

    const res = await request(app)
      .get('/api/games/not-a-valid-id')
      .set('Authorization', 'Bearer user-1')
      .expect(404);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('GAME_NOT_FOUND');
    // String IDs that pass the lightweight format check are resolved via the DB
    // so unknown values surface as GAME_NOT_FOUND rather than GAME_INVALID_ID.
    expect(mockPrisma.game.findUnique).toHaveBeenCalled();
  });

  it('GET /api/games/:gameId returns 404 GAME_NOT_FOUND for well-formed but missing gameId', async () => {
    const validButMissingGameId = '550e8400-e29b-41d4-a716-4466554400ff';
    mockPrisma.game.findUnique.mockResolvedValueOnce(null as any);

    const app = createTestApp();

    const res = await request(app)
      .get(`/api/games/${validButMissingGameId}`)
      .set('Authorization', 'Bearer user-1')
      .expect(404);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('GAME_NOT_FOUND');
    expect(mockPrisma.game.findUnique).toHaveBeenCalled();
  });

  it('GET /api/games/:gameId denies access to non-participants when spectators are disabled', async () => {
    const validGameId = '550e8400-e29b-41d4-a716-446655440001';
    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
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
      .get(`/api/games/${validGameId}`)
      .set('Authorization', 'Bearer user-1')
      .expect(403);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('RESOURCE_ACCESS_DENIED');
  });

  it('GET /api/games/:gameId/moves reuses the same participant-or-spectator invariant', async () => {
    const validGameId = '550e8400-e29b-41d4-a716-446655440002';
    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
      player1Id: 'other-user',
      player2Id: null,
      player3Id: null,
      player4Id: null,
      allowSpectators: false,
    } as any);

    const app = createTestApp();

    const res = await request(app)
      .get(`/api/games/${validGameId}/moves`)
      .set('Authorization', 'Bearer user-1')
      .expect(403);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('RESOURCE_ACCESS_DENIED');
  });

  it('POST /api/games/:gameId/leave denies non-participants with ACCESS_DENIED', async () => {
    const validGameId = '550e8400-e29b-41d4-a716-446655440003';
    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
      player1Id: 'other-user',
      player2Id: null,
      player3Id: null,
      player4Id: null,
      status: 'waiting',
    } as any);

    const app = createTestApp();

    const res = await request(app)
      .post(`/api/games/${validGameId}/leave`)
      .set('Authorization', 'Bearer user-1')
      .expect(403);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('RESOURCE_ACCESS_DENIED');
  });

  it('POST /api/games/:gameId/leave marks an active game as completed and broadcasts lobby cancellation', async () => {
    const validGameId = '550e8400-e29b-41d4-a716-446655440020';

    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
      player1Id: 'user-1',
      player2Id: null,
      player3Id: null,
      player4Id: null,
      status: 'active',
    } as any);

    mockPrisma.game.update.mockResolvedValueOnce({
      id: validGameId,
      status: 'completed',
      endedAt: new Date(),
      updatedAt: new Date(),
    } as any);

    const wsServerMock = {
      getGameDiagnosticsForGame: jest.fn(),
      broadcastLobbyEvent: jest.fn(),
    };

    const app = createTestApp(wsServerMock);

    const res = await request(app)
      .post(`/api/games/${validGameId}/leave`)
      .set('Authorization', 'Bearer user-1')
      .expect(200);

    expect(res.body.success).toBe(true);
    expect(res.body.message).toBe('Resigned from game');

    expect(mockPrisma.game.update).toHaveBeenCalledWith({
      where: { id: validGameId },
      data: expect.objectContaining({
        status: 'completed',
        endedAt: expect.any(Date),
        updatedAt: expect.any(Date),
      }),
    });

    expect(wsServerMock.broadcastLobbyEvent).toHaveBeenCalledWith('lobby:game_cancelled', {
      gameId: validGameId,
    });
  });

  it('POST /api/games/:gameId/leave cancels a waiting game with no remaining players and broadcasts lobby cancellation', async () => {
    const validGameId = '550e8400-e29b-41d4-a716-446655440021';

    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
      player1Id: 'user-1',
      player2Id: null,
      player3Id: null,
      player4Id: null,
      status: 'waiting',
    } as any);

    // First update removes the leaving player and returns a game with no remaining players.
    mockPrisma.game.update
      .mockResolvedValueOnce({
        id: validGameId,
        player1Id: null,
        player2Id: null,
        player3Id: null,
        player4Id: null,
        status: 'waiting',
      } as any)
      // Second update marks the game as abandoned.
      .mockResolvedValueOnce({
        id: validGameId,
        status: 'abandoned',
        endedAt: new Date(),
        updatedAt: new Date(),
      } as any);

    const wsServerMock = {
      getGameDiagnosticsForGame: jest.fn(),
      broadcastLobbyEvent: jest.fn(),
    };

    const app = createTestApp(wsServerMock);

    const res = await request(app)
      .post(`/api/games/${validGameId}/leave`)
      .set('Authorization', 'Bearer user-1')
      .expect(200);

    expect(res.body.success).toBe(true);
    expect(res.body.message).toBe('Left game successfully');

    expect(mockPrisma.game.update).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        where: { id: validGameId },
        data: expect.objectContaining({
          updatedAt: expect.any(Date),
        }),
      })
    );

    expect(mockPrisma.game.update).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        where: { id: validGameId },
        data: expect.objectContaining({
          status: 'abandoned',
          endedAt: expect.any(Date),
          updatedAt: expect.any(Date),
        }),
      })
    );

    expect(wsServerMock.broadcastLobbyEvent).toHaveBeenCalledWith('lobby:game_cancelled', {
      gameId: validGameId,
    });
  });

  it('POST /api/games/:gameId/leave delegates active rated resignations to the WebSocket host when available', async () => {
    const validGameId = '550e8400-e29b-41d4-a716-446655440030';

    // First call: route-level authorization + status/isRated
    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
      player1Id: 'user-1',
      player2Id: 'user-2',
      player3Id: null,
      player4Id: null,
      status: 'active',
      isRated: true,
    } as any);
    // Second call: winnerId after host-driven resign has completed
    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
      winnerId: 'user-2',
    } as any);

    const wsServerMock = {
      getGameDiagnosticsForGame: jest.fn(),
      broadcastLobbyEvent: jest.fn(),
      handlePlayerResignFromHttp: jest.fn().mockResolvedValue(undefined),
    };

    const app = createTestApp(wsServerMock);

    const res = await request(app)
      .post(`/api/games/${validGameId}/leave`)
      .set('Authorization', 'Bearer user-1')
      .expect(200);

    // Host path must be used for active games when a WebSocketServer is wired.
    expect(wsServerMock.handlePlayerResignFromHttp).toHaveBeenCalledWith(validGameId, 'user-1');

    // Fallback DB-only completion semantics (direct game.update) should not run
    // when the host path succeeds.
    expect(mockPrisma.game.update).not.toHaveBeenCalled();
    expect(mockProcessGameResult).not.toHaveBeenCalled();

    expect(res.body.success).toBe(true);
    expect(res.body.message).toBe('Resigned from game');
    expect(res.body.data).toEqual(
      expect.objectContaining({
        winnerId: 'user-2',
      })
    );
  });

  it('POST /api/games/:gameId/leave also uses host path for active unrated games (no direct DB completion)', async () => {
    const validGameId = '550e8400-e29b-41d4-a716-446655440031';

    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
      player1Id: 'user-1',
      player2Id: 'user-2',
      player3Id: null,
      player4Id: null,
      status: 'active',
      isRated: false,
    } as any);
    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
      winnerId: 'user-2',
    } as any);

    const wsServerMock = {
      getGameDiagnosticsForGame: jest.fn(),
      broadcastLobbyEvent: jest.fn(),
      handlePlayerResignFromHttp: jest.fn().mockResolvedValue(undefined),
    };

    const app = createTestApp(wsServerMock);

    const res = await request(app)
      .post(`/api/games/${validGameId}/leave`)
      .set('Authorization', 'Bearer user-1')
      .expect(200);

    expect(wsServerMock.handlePlayerResignFromHttp).toHaveBeenCalledWith(validGameId, 'user-1');
    expect(mockPrisma.game.update).not.toHaveBeenCalled();
    expect(mockProcessGameResult).not.toHaveBeenCalled();

    expect(res.body.success).toBe(true);
    expect(res.body.message).toBe('Resigned from game');
    expect(res.body.data).toEqual(
      expect.objectContaining({
        winnerId: 'user-2',
      })
    );
  });

  it('POST /api/games/:gameId/leave on a completed game does not trigger rating updates or host resign handling', async () => {
    const validGameId = '550e8400-e29b-41d4-a716-446655440032';

    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
      player1Id: 'user-1',
      player2Id: 'user-2',
      player3Id: null,
      player4Id: null,
      status: 'completed',
      isRated: true,
    } as any);

    // After the leaving player is removed there is still one remaining participant,
    // so the route should not attempt to re-complete or abandon the game.
    mockPrisma.game.update.mockResolvedValueOnce({
      id: validGameId,
      player1Id: null,
      player2Id: 'user-2',
      player3Id: null,
      player4Id: null,
      status: 'completed',
    } as any);

    const wsServerMock = {
      getGameDiagnosticsForGame: jest.fn(),
      broadcastLobbyEvent: jest.fn(),
      handlePlayerResignFromHttp: jest.fn(),
    };

    const app = createTestApp(wsServerMock);

    const res = await request(app)
      .post(`/api/games/${validGameId}/leave`)
      .set('Authorization', 'Bearer user-1')
      .expect(200);

    expect(res.body.success).toBe(true);
    expect(res.body.message).toBe('Left game successfully');

    // Completed games should use the "leave" path, not the active-game resign path.
    expect(wsServerMock.handlePlayerResignFromHttp).not.toHaveBeenCalled();

    // Only the disconnect update should be performed.
    expect(mockPrisma.game.update).toHaveBeenCalledTimes(1);

    // No rating updates should be triggered when leaving a completed game.
    expect(mockProcessGameResult).not.toHaveBeenCalled();
  });
});

describe('Game diagnostics session route', () => {
  beforeEach(() => {
    mockPrisma.game.findUnique.mockReset();
  });

  it('GET /api/games/:gameId/diagnostics/session returns 401 AUTH_TOKEN_REQUIRED when unauthenticated', async () => {
    const app = createTestApp();

    const res = await request(app).get('/api/games/game-1/diagnostics/session').expect(401);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('AUTH_TOKEN_REQUIRED');
  });

  it('GET /api/games/:gameId/diagnostics/session denies access to non-participants when spectators are disabled', async () => {
    const validGameId = '550e8400-e29b-41d4-a716-446655440010';
    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
      player1Id: 'other-user',
      player2Id: null,
      player3Id: null,
      player4Id: null,
      allowSpectators: false,
    } as any);

    const app = createTestApp();

    const res = await request(app)
      .get(`/api/games/${validGameId}/diagnostics/session`)
      .set('Authorization', 'Bearer user-1')
      .expect(403);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('RESOURCE_ACCESS_DENIED');
  });

  it('GET /api/games/:gameId/diagnostics/session returns 404 GAME_NOT_FOUND when game does not exist', async () => {
    const validGameId = '550e8400-e29b-41d4-a716-446655440011';
    mockPrisma.game.findUnique.mockResolvedValueOnce(null as any);

    const app = createTestApp();

    const res = await request(app)
      .get(`/api/games/${validGameId}/diagnostics/session`)
      .set('Authorization', 'Bearer user-1')
      .expect(404);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('GAME_NOT_FOUND');
  });

  it('returns minimal diagnostics snapshot when no WebSocketServer instance is wired', async () => {
    const validGameId = '550e8400-e29b-41d4-a716-446655440012';
    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
      player1Id: 'user-1',
      player2Id: null,
      player3Id: null,
      player4Id: null,
      allowSpectators: false,
    } as any);

    const app = createTestApp();

    const res = await request(app)
      .get(`/api/games/${validGameId}/diagnostics/session`)
      .set('Authorization', 'Bearer user-1')
      .expect(200);

    expect(res.body.success).toBe(true);
    expect(res.body.data).toEqual({
      sessionStatus: null,
      lastAIRequestState: null,
      aiDiagnostics: null,
      connections: {},
      meta: {
        hasInMemorySession: false,
      },
    });
  });

  it('returns diagnostics snapshot from injected WebSocketServer instance when available', async () => {
    const validGameId = '550e8400-e29b-41d4-a716-446655440013';
    mockPrisma.game.findUnique.mockResolvedValueOnce({
      id: validGameId,
      player1Id: 'user-1',
      player2Id: null,
      player3Id: null,
      player4Id: null,
      allowSpectators: false,
    } as any);

    const diagnosticsSnapshot = {
      sessionStatus: { kind: 'active', reason: 'test' },
      lastAIRequestState: { kind: 'completed', terminalCode: 'OK' },
      aiDiagnostics: { degraded: false, aiErrorCount: 0 },
      connections: {
        'user-1': { state: 'connected' },
      },
      hasInMemorySession: true,
    };

    // Minimal WebSocketServer-like test double that supports both diagnostics
    // and lobby broadcasting used by the game routes. The diagnostics test
    // injects this instance via createTestApp(wsServerMock), and subsequent
    // tests that construct an app without explicitly passing a wsServer will
    // still see this instance via setWebSocketServer(), so it must implement
    // all methods the routes might call (at minimum, getGameDiagnosticsForGame
    // and broadcastLobbyEvent).
    const wsServerMock = {
      getGameDiagnosticsForGame: jest.fn().mockReturnValue(diagnosticsSnapshot),
      broadcastLobbyEvent: jest.fn(),
    };

    const app = createTestApp(wsServerMock);

    const res = await request(app)
      .get(`/api/games/${validGameId}/diagnostics/session`)
      .set('Authorization', 'Bearer user-1')
      .expect(200);

    expect(wsServerMock.getGameDiagnosticsForGame).toHaveBeenCalledWith(validGameId);
    expect(res.body.success).toBe(true);
    // Non-admin users get sanitized diagnostics (AI state hidden, connections as count only)
    expect(res.body.data).toEqual({
      sessionStatus: diagnosticsSnapshot.sessionStatus,
      lastAIRequestState: null, // Hidden from non-admins
      aiDiagnostics: null, // Hidden from non-admins
      connections: { count: 1 }, // Count only for non-admins
      meta: {
        hasInMemorySession: true,
        sanitized: true,
      },
    });
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
      timeControl: { type: 'rapid', initialTime: 300, increment: 0 },
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
        timeControl: { type: 'rapid', initialTime: 300, increment: 0 },
      })
      .expect(201);

    expect(res.body.success).toBe(true);
    expect(mockPrisma.game.create).toHaveBeenCalledTimes(1);

    // First quota is per-user, second is per-IP.
    // Third argument is the request object (passed for context/logging).
    expect(mockConsumeRateLimit).toHaveBeenCalledWith(
      'gameCreateUser',
      'user-1',
      expect.anything()
    );
    expect(mockConsumeRateLimit).toHaveBeenCalledWith(
      'gameCreateIp',
      '203.0.113.1',
      expect.anything()
    );
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
        timeControl: { type: 'rapid', initialTime: 300, increment: 0 },
      })
      .expect(429);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('RATE_LIMIT_GAME_CREATE');
    expect(mockPrisma.game.create).not.toHaveBeenCalled();
  });
});

describe('Protected user routes', () => {
  it('DELETE /api/users/me returns 401 TOKEN_REQUIRED when unauthenticated', async () => {
    const app = createTestApp();

    const res = await request(app).delete('/api/users/me').expect(401);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('AUTH_TOKEN_REQUIRED');
  });
});
