import express from 'express';
import request from 'supertest';
import userRoutes from '../../src/server/routes/user';
import authRoutes, { __testResetLoginLockoutState } from '../../src/server/routes/auth';
import { errorHandler, createError } from '../../src/server/middleware/errorHandler';
import * as authModule from '../../src/server/middleware/auth';
import { mockDb, prismaStub, resetPrismaMockDb } from '../utils/prismaTestUtils';

// Stub out rate limiting so tests don't depend on Redis or global state.
jest.mock('../../src/server/middleware/rateLimiter', () => ({
  authRateLimiter: (_req: any, _res: any, next: any) => next(),
  rateLimiter: (_req: any, _res: any, next: any) => next(),
}));

// Mocked database client, defaulting to the shared prismaStub.
let mockDatabaseClient: any = prismaStub;

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => mockDatabaseClient,
}));

// Mock auth helpers so we don't depend on real JWT configuration.
jest.mock('../../src/server/middleware/auth', () => {
  const actual = jest.requireActual('../../src/server/middleware/auth');
  const generateToken = jest.fn(() => 'ACCESS_TOKEN');
  const generateRefreshToken = jest.fn(() => 'REFRESH_TOKEN');
  const verifyRefreshToken = jest.fn(() => ({
    userId: 'auth-user-1',
    email: 'user1@example.com',
    tokenVersion: 0,
  }));
  const authenticate = jest.fn((req: any, _res: any, next: any) => {
    // Default authenticated user for tests that hit /api/users.
    req.user = {
      id: 'auth-user-1',
      email: 'user1@example.com',
      username: 'user1',
      role: 'USER',
    };
    next();
  });

  return {
    ...actual,
    generateToken,
    generateRefreshToken,
    verifyRefreshToken,
    authenticate,
  };
});

// Mock bcrypt to keep hashing/compare deterministic.
jest.mock('bcryptjs', () => ({
  hash: jest.fn(async (password: string) => `hashed:${password}`),
  compare: jest.fn(async (password: string, hashed: string) => hashed === `hashed:${password}`),
}));

// --- Test app factory ---------------------------------------------------

function createTestApp() {
  const app = express();
  app.use(express.json());
  app.use('/api/auth', authRoutes);
  // Mirror main routing: /api/users is protected by authenticate middleware.
  app.use('/api/users', (authModule as any).authenticate, userRoutes);
  app.use(errorHandler);
  return app;
}

// Convenience alias for the mocked auth module so tests can override behaviour.
const mockedAuth = authModule as jest.Mocked<typeof authModule>;

describe('User account deletion HTTP route', () => {
  beforeEach(() => {
    resetPrismaMockDb();
    __testResetLoginLockoutState();
    mockDatabaseClient = prismaStub;

    // Reset authenticate to its default stubbed implementation for each test.
    mockedAuth.authenticate.mockClear();
  });

  it('DELETE /api/users/me requires authentication', async () => {
    const app = createTestApp();
    const initialUserCount = mockDb.users.length;

    // Force the mocked authenticate middleware to fail for this request.
    mockedAuth.authenticate.mockImplementationOnce(async (_req: any, _res: any, next: any) => {
      next(createError('Authentication token required', 401, 'TOKEN_REQUIRED'));
    });

    const res = await request(app).delete('/api/users/me').send({}).expect(401);

    expect(res.body.success).toBe(false);
    expect(res.body.error.code).toBe('TOKEN_REQUIRED');
    expect(mockDb.users.length).toBe(initialUserCount);
  });

  it('soft-deletes the current user, revokes tokens, and anonymises PII', async () => {
    const now = new Date();

    mockDb.users.push({
      id: 'auth-user-1',
      email: 'user1@example.com',
      username: 'user1',
      passwordHash: 'hashed:Secret123',
      role: 'USER',
      rating: 1200,
      gamesPlayed: 0,
      gamesWon: 0,
      emailVerified: true,
      isActive: true,
      createdAt: now,
      updatedAt: now,
      lastLoginAt: null,
      tokenVersion: 0,
      verificationToken: 'verify-token',
      verificationTokenExpires: new Date(now.getTime() + 60 * 60 * 1000),
      passwordResetToken: 'reset-token',
      passwordResetExpires: new Date(now.getTime() + 60 * 60 * 1000),
      deletedAt: null,
    });

    mockDb.refreshTokens.push(
      {
        id: 'rt-1',
        token: 'T1',
        userId: 'auth-user-1',
        expiresAt: new Date(now.getTime() + 60 * 60 * 1000),
      },
      {
        id: 'rt-2',
        token: 'T2',
        userId: 'other-user',
        expiresAt: new Date(now.getTime() + 60 * 60 * 1000),
      }
    );

    const app = createTestApp();

    const res = await request(app).delete('/api/users/me').send({}).expect(200);

    expect(res.body.success).toBe(true);
    expect(res.body.message).toBe('Account deleted successfully');

    const updated = mockDb.users.find((u) => u.id === 'auth-user-1') as any;
    expect(updated).toBeDefined();
    expect(updated.isActive).toBe(false);
    expect(updated.deletedAt).toBeInstanceOf(Date);
    expect(updated.tokenVersion).toBe(1);

    // PII anonymisation
    expect(updated.email).toBe(`deleted+${updated.id}@example.invalid`);
    expect(updated.username).toBe(`DeletedPlayer_${String(updated.id).slice(0, 8)}`);

    // Reset/verification tokens cleared
    expect(updated.verificationToken).toBeNull();
    expect(updated.verificationTokenExpires).toBeNull();
    expect(updated.passwordResetToken).toBeNull();
    expect(updated.passwordResetExpires).toBeNull();

    // All refresh tokens for this user are deleted; others are preserved.
    const remainingForUser = mockDb.refreshTokens.filter((rt) => rt.userId === 'auth-user-1');
    expect(remainingForUser).toHaveLength(0);
    const remainingForOthers = mockDb.refreshTokens.filter((rt) => rt.userId === 'other-user');
    expect(remainingForOthers).toHaveLength(1);
  });

  it('is idempotent when deleting an already-deleted account', async () => {
    const now = new Date();

    mockDb.users.push({
      id: 'auth-user-1',
      email: 'user1@example.com',
      username: 'user1',
      passwordHash: 'hashed:Secret123',
      role: 'USER',
      rating: 1200,
      gamesPlayed: 0,
      gamesWon: 0,
      emailVerified: true,
      isActive: true,
      createdAt: now,
      updatedAt: now,
      lastLoginAt: null,
      tokenVersion: 0,
      verificationToken: null,
      verificationTokenExpires: null,
      passwordResetToken: null,
      passwordResetExpires: null,
      deletedAt: null,
    });

    const app = createTestApp();

    const first = await request(app).delete('/api/users/me').send({}).expect(200);
    expect(first.body.success).toBe(true);

    const afterFirst = mockDb.users.find((u) => u.id === 'auth-user-1') as any;
    expect(afterFirst).toBeDefined();
    const { deletedAt: deletedAtFirst, tokenVersion: tokenVersionFirst } = afterFirst;

    const second = await request(app).delete('/api/users/me').send({}).expect(200);
    expect(second.body.success).toBe(true);

    const afterSecond = mockDb.users.find((u) => u.id === 'auth-user-1') as any;
    expect(afterSecond).toBeDefined();
    expect(afterSecond.isActive).toBe(false);
    expect(afterSecond.deletedAt).toEqual(deletedAtFirst);
    expect(afterSecond.tokenVersion).toBe(tokenVersionFirst);
  });

  it('blocks login attempts after account deletion', async () => {
    const now = new Date();

    mockDb.users.push({
      id: 'auth-user-1',
      email: 'user1@example.com',
      username: 'user1',
      passwordHash: 'hashed:Secret123',
      role: 'USER',
      rating: 1200,
      gamesPlayed: 0,
      gamesWon: 0,
      emailVerified: true,
      isActive: true,
      createdAt: now,
      updatedAt: now,
      lastLoginAt: null,
      tokenVersion: 0,
      verificationToken: null,
      verificationTokenExpires: null,
      passwordResetToken: null,
      passwordResetExpires: null,
      deletedAt: null,
    });

    const app = createTestApp();

    // First, delete the account via the new endpoint.
    await request(app).delete('/api/users/me').send({}).expect(200);

    const res = await request(app)
      .post('/api/auth/login')
      .send({ email: 'user1@example.com', password: 'Secret123' })
      .expect(401);

    expect(res.body.success).toBe(false);
    // Current implementation returns INVALID_CREDENTIALS for deleted users
    // because the email no longer exists; both this and ACCOUNT_DEACTIVATED
    // represent a blocked login, which satisfies S-05.E.1 for this slice.
    expect(
      res.body.error.code === 'ACCOUNT_DEACTIVATED' || res.body.error.code === 'INVALID_CREDENTIALS'
    ).toBe(true);

    // No new tokens should be issued.
    expect(mockDb.refreshTokens.length).toBe(0);
  });
});
