import express from 'express';
import request from 'supertest';

// Simple error handler
function testErrorHandler(err: any, _req: any, res: any, _next: any) {
  const statusCode = err.statusCode || 500;
  const code = err.code || 'SERVER_ERROR';
  res.status(statusCode).json({
    success: false,
    error: { code, message: err.message || 'Internal server error' },
  });
}

// --- Mock state ---
let mockAuthUser: any = null;
let mockAuthError: any = null;

const mockGameFindUnique = jest.fn();
const mockGameFindMany = jest.fn();
const mockGameCreate = jest.fn();
const mockGameUpdate = jest.fn();
const mockGameCount = jest.fn();
const mockUserFindUnique = jest.fn();
const mockMoveFindMany = jest.fn();
const mockMoveCount = jest.fn();
const mockCreateDecisionPhaseFixtureGame = jest.fn().mockResolvedValue('fixture-game-123');

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => ({
    game: {
      findUnique: mockGameFindUnique,
      findMany: mockGameFindMany,
      create: mockGameCreate,
      update: mockGameUpdate,
      count: mockGameCount,
    },
    user: { findUnique: mockUserFindUnique },
    move: { findMany: mockMoveFindMany, count: mockMoveCount },
    $transaction: jest.fn(async (fn: any) =>
      fn({
        game: {
          findUnique: mockGameFindUnique,
          create: mockGameCreate,
          update: mockGameUpdate,
        },
        user: { findUnique: mockUserFindUnique, update: jest.fn() },
      })
    ),
  })),
  withQueryTimeoutStrict: jest.fn(async (promise: Promise<any>) => ({
    success: true,
    data: await promise,
  })),
}));

jest.mock('../../src/server/middleware/auth', () => ({
  authenticate: (req: any, _res: any, next: any) => {
    if (mockAuthError) return next(mockAuthError);
    req.user = mockAuthUser;
    next();
  },
  getAuthUserId: (req: any) => req.user?.id,
  AuthenticatedRequest: {},
}));

jest.mock('../../src/server/middleware/rateLimiter', () => ({
  consumeRateLimit: jest.fn().mockResolvedValue(undefined),
  adaptiveRateLimiter: () => (_req: any, _res: any, next: any) => next(),
  sandboxAiRateLimiter: (_req: any, _res: any, next: any) => next(),
  dataExportRateLimiter: (_req: any, _res: any, next: any) => next(),
  userRatingRateLimiter: (_req: any, _res: any, next: any) => next(),
  userSearchRateLimiter: (_req: any, _res: any, next: any) => next(),
}));

jest.mock('../../src/server/utils/logger', () => ({
  logger: { info: jest.fn(), error: jest.fn(), warn: jest.fn(), debug: jest.fn() },
  httpLogger: { info: jest.fn(), error: jest.fn(), warn: jest.fn() },
  getRequestContext: jest.fn(() => ({})),
  withRequestContext: jest.fn((_req: any, meta: any) => meta),
}));

jest.mock('../../src/server/config', () => ({
  config: {
    isDevelopment: true,
    isProduction: false,
    isTest: true,
    featureFlags: { sandboxAi: true, orchestrator: { adapterEnabled: false } },
    auth: { jwtSecret: 'test-secret-at-least-32-characters-long' },
    ai: { serviceUrl: 'http://localhost:5000' },
  },
}));

jest.mock('../../src/server/services/AIServiceClient', () => ({
  getAIServiceClient: jest.fn(() => null),
}));

jest.mock('../../src/server/game/testFixtures/decisionPhaseFixtures', () => ({
  createDecisionPhaseFixtureGame: mockCreateDecisionPhaseFixtureGame,
}));

jest.mock('../../src/server/services/RatingService', () => ({
  RatingService: { getRatingHistory: jest.fn().mockResolvedValue({ history: [], total: 0 }) },
  RatingUpdateResult: {},
}));

// Import after mocks
import gameRoutes, { setWebSocketServer } from '../../src/server/routes/game';
import { authenticate } from '../../src/server/middleware/auth';

function createTestApp() {
  setWebSocketServer(null);
  const app = express();
  app.use(express.json());
  app.use('/api/games', authenticate as any, gameRoutes);
  app.use(testErrorHandler);
  return app;
}

const mockGame = {
  id: 'game-abc',
  boardType: 'square8',
  maxPlayers: 2,
  status: 'waiting',
  isRated: true,
  allowSpectators: true,
  inviteCode: 'ABC123',
  timeControl: { initialTime: 600, increment: 5 },
  gameState: '{}',
  player1Id: 'user-123',
  player2Id: null,
  player3Id: null,
  player4Id: null,
  winnerId: null,
  createdAt: new Date('2025-01-01'),
  updatedAt: new Date('2025-01-01'),
  startedAt: null,
  endedAt: null,
};

describe('Game HTTP routes', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockAuthUser = {
      id: 'user-123',
      email: 'test@example.com',
      username: 'testuser',
      role: 'USER',
    };
    mockAuthError = null;
  });

  describe('GET /api/games (list games)', () => {
    it('returns 401 when not authenticated', async () => {
      mockAuthError = Object.assign(new Error('Auth required'), {
        statusCode: 401,
        code: 'AUTH_REQUIRED',
      });
      const app = createTestApp();
      const res = await request(app).get('/api/games').expect(401);

      expect(res.body.success).toBe(false);
    });
  });

  describe('POST /api/games/fixtures/decision-phase', () => {
    it('persists a distinct registered second player for multiplayer fixtures', async () => {
      mockUserFindUnique.mockResolvedValue({ id: 'user-456' });
      const app = createTestApp();

      const res = await request(app)
        .post('/api/games/fixtures/decision-phase')
        .send({
          scenario: 'line_processing',
          isRated: false,
          secondPlayerUsername: 'opponent',
        })
        .expect(201);

      expect(res.body.data.gameId).toBe('fixture-game-123');
      expect(mockUserFindUnique).toHaveBeenCalledWith({
        where: { username: 'opponent' },
        select: { id: true },
      });
      expect(mockCreateDecisionPhaseFixtureGame).toHaveBeenCalledWith(
        expect.objectContaining({
          creatorUserId: 'user-123',
          secondPlayerUserId: 'user-456',
          scenario: 'line_processing',
          isRated: false,
        })
      );
    });

    it('rejects an unknown second player', async () => {
      mockUserFindUnique.mockResolvedValue(null);
      const app = createTestApp();

      const res = await request(app)
        .post('/api/games/fixtures/decision-phase')
        .send({ scenario: 'line_processing', secondPlayerUsername: 'missing-user' })
        .expect(400);

      expect(res.body.error.code).toBe('VALIDATION_INVALID_REQUEST');
      expect(mockCreateDecisionPhaseFixtureGame).not.toHaveBeenCalled();
    });

    it.each([42, null, true, {}, []])(
      'rejects a non-string second player username (%p)',
      async (secondPlayerUsername) => {
        const app = createTestApp();

        const res = await request(app)
          .post('/api/games/fixtures/decision-phase')
          .send({ scenario: 'line_processing', secondPlayerUsername })
          .expect(400);

        expect(res.body.error.code).toBe('VALIDATION_INVALID_REQUEST');
        expect(mockUserFindUnique).not.toHaveBeenCalled();
        expect(mockCreateDecisionPhaseFixtureGame).not.toHaveBeenCalled();
      }
    );
  });

  describe('GET /api/games/:gameId (game details)', () => {
    it('returns 404 for non-existent game', async () => {
      mockGameFindUnique.mockResolvedValue(null);
      const app = createTestApp();
      const res = await request(app).get('/api/games/game-abc').expect(404);

      expect(res.body.success).toBe(false);
    });

    it('returns 401 when not authenticated', async () => {
      mockAuthError = Object.assign(new Error('Auth required'), {
        statusCode: 401,
        code: 'AUTH_REQUIRED',
      });
      const app = createTestApp();
      await request(app).get('/api/games/game-abc').expect(401);
    });
  });

  describe('POST /api/games (create game)', () => {
    it('returns 401 when not authenticated', async () => {
      mockAuthError = Object.assign(new Error('Auth required'), {
        statusCode: 401,
        code: 'AUTH_REQUIRED',
      });
      const app = createTestApp();
      await request(app)
        .post('/api/games')
        .send({
          boardType: 'square8',
          maxPlayers: 2,
          timeControl: { initialTime: 600, increment: 5 },
        })
        .expect(401);
    });
  });
});
