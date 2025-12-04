/**
 * Unit tests for the internal HTTP move harness route
 * POST /api/games/:gameId/moves.
 *
 * These tests exercise the Express route in src/server/routes/game.ts
 * in isolation with a mocked WebSocketServer instance so that the
 * harness behaviour can be validated without touching the real
 * GameSessionManager or rules engine.
 */

import express, { Request, Response } from 'express';
import request from 'supertest';
import { errorHandler } from '../../src/server/middleware/errorHandler';

// Enable the HTTP move harness feature flag for this test file while
// preserving the rest of the unified configuration.
jest.mock('../../src/server/config', () => {
  const actual = jest.requireActual('../../src/server/config');
  return {
    ...actual,
    config: {
      ...actual.config,
      isTest: true,
      featureFlags: {
        ...actual.config.featureFlags,
        httpMoveHarness: {
          enabled: true,
        },
      },
    },
  };
});

// Stub rate limiter middleware so tests don't depend on Redis or
// specific quota state.
jest.mock('../../src/server/middleware/rateLimiter', () => ({
  consumeRateLimit: jest.fn().mockResolvedValue({ allowed: true }),
  adaptiveRateLimiter: jest.fn(() => (_req: Request, _res: Response, next: () => void) => next()),
}));

// Silence HTTP route logging for these tests.
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
  httpLogger: {
    info: jest.fn(),
    error: jest.fn(),
  },
}));

import gameRoutes, { setWebSocketServer } from '../../src/server/routes/game';

interface WebSocketServerMock {
  handlePlayerMoveFromHttp: jest.Mock;
}

function createHarnessApp(wsServer: WebSocketServerMock, currentUserId: string = 'user-1') {
  const app = express();
  app.use(express.json());

  // Inject the mocked WebSocket server into the game routes so that the
  // HTTP move harness delegates to it instead of the real host.
  setWebSocketServer(wsServer as any);

  // Minimal auth stub: attach a user object so getAuthUserId(req) sees
  // a consistent authenticated identity.
  app.use(
    '/api/games',
    (req, _res, next) => {
      (req as any).user = {
        id: currentUserId,
        email: `${currentUserId}@example.com`,
        username: 'TestUser',
        role: 'USER',
      };
      next();
    },
    gameRoutes
  );

  app.use(errorHandler as any);
  return app;
}

describe('HTTP move harness route POST /api/games/:gameId/moves', () => {
  let wsServerMock: WebSocketServerMock;

  beforeEach(() => {
    jest.clearAllMocks();
    wsServerMock = {
      handlePlayerMoveFromHttp: jest.fn(),
    };
  });

  it('applies a valid move via the WebSocket host when harness is enabled', async () => {
    const gameId = '550e8400-e29b-41d4-a716-446655440050';

    wsServerMock.handlePlayerMoveFromHttp.mockResolvedValue({
      success: true,
      gameState: { id: gameId, boardType: 'square8' },
      gameResult: null,
    });

    const app = createHarnessApp(wsServerMock);

    const res = await request(app)
      .post(`/api/games/${gameId}/moves`)
      .send({
        moveType: 'place_ring',
        position: {
          to: { x: 3, y: 3 },
        },
      })
      .expect(200);

    expect(wsServerMock.handlePlayerMoveFromHttp).toHaveBeenCalledTimes(1);
    expect(wsServerMock.handlePlayerMoveFromHttp).toHaveBeenCalledWith(
      gameId,
      'user-1',
      expect.objectContaining({
        moveType: 'place_ring',
        position: {
          to: { x: 3, y: 3 },
        },
      })
    );

    expect(res.body.success).toBe(true);
    expect(res.body.data).toEqual(
      expect.objectContaining({
        gameId,
        gameState: expect.objectContaining({ id: gameId }),
        gameResult: null,
      })
    );
  });

  it('supports a wrapped { move } payload for compatibility with gameApi.makeMove', async () => {
    const gameId = '550e8400-e29b-41d4-a716-446655440051';

    wsServerMock.handlePlayerMoveFromHttp.mockResolvedValue({
      success: true,
      gameState: { id: gameId, boardType: 'square8' },
      gameResult: null,
    });

    const app = createHarnessApp(wsServerMock);

    const res = await request(app)
      .post(`/api/games/${gameId}/moves`)
      .send({
        move: {
          moveType: 'place_ring',
          position: {
            to: { x: 4, y: 4 },
          },
        },
      })
      .expect(200);

    expect(wsServerMock.handlePlayerMoveFromHttp).toHaveBeenCalledTimes(1);
    expect(wsServerMock.handlePlayerMoveFromHttp).toHaveBeenCalledWith(
      gameId,
      'user-1',
      expect.objectContaining({
        moveType: 'place_ring',
        position: {
          to: { x: 4, y: 4 },
        },
      })
    );

    expect(res.body.success).toBe(true);
    expect(res.body.data.gameId).toBe(gameId);
  });

  it('rejects invalid move payloads with GAME_INVALID_MOVE and does not call the host', async () => {
    const gameId = '550e8400-e29b-41d4-a716-446655440052';
    const app = createHarnessApp(wsServerMock);

    // Missing required moveType / position fields
    const res = await request(app).post(`/api/games/${gameId}/moves`).send({}).expect(400);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('GAME_INVALID_MOVE');
    expect(wsServerMock.handlePlayerMoveFromHttp).not.toHaveBeenCalled();
  });

  it('maps "Game not found" errors from the host to 404 GAME_NOT_FOUND', async () => {
    const gameId = '550e8400-e29b-41d4-a716-446655440053';

    wsServerMock.handlePlayerMoveFromHttp.mockRejectedValue(new Error('Game not found'));

    const app = createHarnessApp(wsServerMock);

    const res = await request(app)
      .post(`/api/games/${gameId}/moves`)
      .send({
        moveType: 'place_ring',
        position: {
          to: { x: 2, y: 2 },
        },
      })
      .expect(404);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('GAME_NOT_FOUND');
  });

  it('maps "Not your turn" domain errors to 400 GAME_NOT_YOUR_TURN', async () => {
    const gameId = '550e8400-e29b-41d4-a716-446655440054';

    wsServerMock.handlePlayerMoveFromHttp.mockRejectedValue(new Error('Not your turn'));

    const app = createHarnessApp(wsServerMock);

    const res = await request(app)
      .post(`/api/games/${gameId}/moves`)
      .send({
        moveType: 'place_ring',
        position: {
          to: { x: 1, y: 1 },
        },
      })
      .expect(400);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('GAME_NOT_YOUR_TURN');
  });
});
