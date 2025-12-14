import express from 'express';
import request from 'supertest';
import crypto from 'crypto';
import authRoutes, { __testResetLoginLockoutState } from '../../src/server/routes/auth';
import { errorHandler, createError } from '../../src/server/middleware/errorHandler';
import * as authModule from '../../src/server/middleware/auth';
import { mockDb, prismaStub, resetPrismaMockDb } from '../utils/prismaTestUtils';
import { config } from '../../src/server/config';
import { logger } from '../../src/server/utils/logger';

// --- Mocks --------------------------------------------------------------

// Stub out rate limiting so tests don't depend on Redis or global state.
jest.mock('../../src/server/middleware/rateLimiter', () => ({
  authRateLimiter: (_req: any, _res: any, next: any) => next(),
  authLoginRateLimiter: (_req: any, _res: any, next: any) => next(),
  authRegisterRateLimiter: (_req: any, _res: any, next: any) => next(),
  authPasswordResetRateLimiter: (_req: any, _res: any, next: any) => next(),
  rateLimiter: (_req: any, _res: any, next: any) => next(),
}));

// Jest module factories cannot close over arbitrary imports, so we route the
// mocked database client through a top-level variable whose name begins with
// "mock" (whitelisted by Jest). Most tests point this at prismaStub; the
// DATABASE_UNAVAILABLE test sets it to null to simulate a connectivity loss.
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
    userId: 'user-1',
    email: 'user1@example.com',
    tokenVersion: 0,
  }));
  const authenticate = jest.fn((req: any, _res: any, next: any) => {
    // Default authenticated user for tests that don't care about auth details.
    req.user = {
      id: 'auth-user-1',
      email: 'auth@example.com',
      username: 'auth-user',
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
  app.use(errorHandler);
  return app;
}

// --- Helpers ------------------------------------------------------------

const mockedAuth = authModule as jest.Mocked<typeof authModule>;

// --- Tests --------------------------------------------------------------

describe('Auth HTTP routes', () => {
  beforeEach(() => {
    // Defensive: ensure any fake timers enabled by other suites do not leak
    // into these supertest-driven HTTP route tests.
    jest.useRealTimers();

    resetPrismaMockDb();
    __testResetLoginLockoutState();
    mockDatabaseClient = prismaStub;

    // Reset verifyRefreshToken to its default stubbed behaviour for each test.
    mockedAuth.verifyRefreshToken.mockReset();
    mockedAuth.verifyRefreshToken.mockReturnValue({
      userId: 'user-1',
      email: 'user1@example.com',
      tokenVersion: 0,
    } as any);

    // Reset authenticate to the default stub from the jest.mock factory.
    mockedAuth.authenticate.mockClear();
  });

  describe('POST /api/auth/register', () => {
    it('registers a new user and returns tokens', async () => {
      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'user1@example.com',
          username: 'user1',
          password: 'Secret123',
          confirmPassword: 'Secret123',
        })
        .expect(201);

      expect(prismaStub.user.create).toHaveBeenCalled();
      expect(prismaStub.refreshToken.create).toHaveBeenCalled();

      expect(res.body.success).toBe(true);
      expect(res.body.data.user).toMatchObject({
        email: 'user1@example.com',
        username: 'user1',
        role: 'USER',
      });
      expect(res.body.data.accessToken).toBe('ACCESS_TOKEN');
      expect(res.body.data.refreshToken).toBe('REFRESH_TOKEN');
    });

    // Duplicate email/username should fail with 409 and the appropriate code.
    it('returns 409 EMAIL_EXISTS when email already registered', async () => {
      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'other',
        passwordHash: 'hashed:Secret123',
        role: 'USER',
        isActive: true,
        emailVerified: false,
        createdAt: new Date(),
      });

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'user1@example.com',
          username: 'user1',
          password: 'Secret123',
          confirmPassword: 'Secret123',
        })
        .expect(409);

      expect(res.body.success).toBe(false);
      expect(res.body.error.code).toBe('RESOURCE_EMAIL_EXISTS');
    });

    it('returns 409 USERNAME_EXISTS when username already taken', async () => {
      mockDb.users.push({
        id: 'user-1',
        email: 'other@example.com',
        username: 'user1',
        passwordHash: 'hashed:Secret123',
        role: 'USER',
        isActive: true,
        emailVerified: false,
        createdAt: new Date(),
      });

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'user1@example.com',
          username: 'user1',
          password: 'Secret123',
          confirmPassword: 'Secret123',
        })
        .expect(409);

      expect(res.body.success).toBe(false);
      expect(res.body.error.code).toBe('RESOURCE_USERNAME_EXISTS');
    });

    it('rejects invalid registration payloads with 400 INVALID_REQUEST', async () => {
      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'user1@example.com',
          username: 'user1',
          password: 'short',
          confirmPassword: 'short',
        })
        .expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error.code).toBe('VALIDATION_FAILED');
      expect(res.body.error).not.toHaveProperty('stack');
      // Validation errors now include field-level details
      expect(res.body.error.details).toBeDefined();
    });
  });

  describe('POST /api/auth/login', () => {
    it('returns 401 INVALID_CREDENTIALS when user not found', async () => {
      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/login')
        .send({ email: 'missing@example.com', password: 'Secret123' })
        .expect(401);

      expect(res.body.error.code).toBe('AUTH_INVALID_CREDENTIALS');
    });

    // Inactive users should be rejected before password verification.
    it('returns 401 ACCOUNT_DEACTIVATED for inactive user', async () => {
      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        passwordHash: 'hashed:Secret123',
        role: 'USER',
        isActive: false,
        emailVerified: true,
        createdAt: new Date(),
      });

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/login')
        .send({ email: 'user1@example.com', password: 'Secret123' })
        .expect(401);

      expect(res.body.error.code).toBe('AUTH_ACCOUNT_DEACTIVATED');
    });

    it('returns 401 INVALID_CREDENTIALS for wrong password', async () => {
      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        passwordHash: 'hashed:Correct123',
        role: 'USER',
        isActive: true,
        emailVerified: true,
        createdAt: new Date(),
      });

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/login')
        .send({ email: 'user1@example.com', password: 'Wrong123' })
        .expect(401);

      expect(res.body.error.code).toBe('AUTH_INVALID_CREDENTIALS');
    });

    it('logs in successfully and returns tokens', async () => {
      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        passwordHash: 'hashed:Secret123',
        role: 'USER',
        isActive: true,
        emailVerified: true,
        createdAt: new Date(),
        lastLoginAt: null,
      });

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/login')
        .send({ email: 'user1@example.com', password: 'Secret123' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.data.user).toMatchObject({
        email: 'user1@example.com',
        username: 'user1',
      });
      expect(res.body.data.accessToken).toBe('ACCESS_TOKEN');
      expect(res.body.data.refreshToken).toBe('REFRESH_TOKEN');

      expect(prismaStub.refreshToken.create).toHaveBeenCalled();
      expect(prismaStub.user.update).toHaveBeenCalled();
    });

    it('locks out an email after too many failed login attempts', async () => {
      const app = createTestApp();
      const email = 'lockout@example.com';
      const maxAttempts = config.auth.maxFailedLoginAttempts;

      for (let i = 0; i < maxAttempts; i++) {
        const res = await request(app)
          .post('/api/auth/login')
          .send({ email, password: 'Wrong123' })
          .expect(401);

        expect(res.body.error.code).toBe('AUTH_INVALID_CREDENTIALS');
      }

      const lockedRes = await request(app)
        .post('/api/auth/login')
        .send({ email, password: 'Wrong123' })
        .expect(429);

      expect(lockedRes.body.error.code).toBe('AUTH_LOGIN_LOCKED_OUT');
    });

    it('resets failed-attempt counter after a successful login', async () => {
      const app = createTestApp();
      const email = 'user1@example.com';

      mockDb.users.push({
        id: 'user-1',
        email,
        username: 'user1',
        passwordHash: 'hashed:Secret123',
        role: 'USER',
        isActive: true,
        emailVerified: true,
        createdAt: new Date(),
        lastLoginAt: null,
      });

      const belowThreshold = config.auth.maxFailedLoginAttempts - 1;

      for (let i = 0; i < belowThreshold; i++) {
        const res = await request(app)
          .post('/api/auth/login')
          .send({ email, password: 'WrongPassword' })
          .expect(401);

        expect(res.body.error.code).toBe('AUTH_INVALID_CREDENTIALS');
      }

      const successRes = await request(app)
        .post('/api/auth/login')
        .send({ email, password: 'Secret123' })
        .expect(200);

      expect(successRes.body.success).toBe(true);

      for (let i = 0; i < belowThreshold; i++) {
        const res = await request(app)
          .post('/api/auth/login')
          .send({ email, password: 'WrongPassword' })
          .expect(401);

        expect(res.body.error.code).toBe('AUTH_INVALID_CREDENTIALS');
      }
    });

    it('allows login attempts again after lockout duration has passed (in-memory fallback)', async () => {
      const app = createTestApp();
      const email = 'lockout-expiry@example.com';
      const maxAttempts = config.auth.maxFailedLoginAttempts;
      const originalLockoutDuration = config.auth.lockoutDurationSeconds;
      let dateNowSpy: jest.SpyInstance<number, []> | null = null;

      try {
        config.auth.lockoutDurationSeconds = 1;
        const baseTimeMs = new Date('2020-01-01T00:00:00.000Z').getTime();
        dateNowSpy = jest.spyOn(Date, 'now').mockImplementation(() => baseTimeMs);

        for (let i = 0; i < maxAttempts; i++) {
          const res = await request(app)
            .post('/api/auth/login')
            .send({ email, password: 'Wrong123' })
            .expect(401);

          expect(res.body.error.code).toBe('AUTH_INVALID_CREDENTIALS');
        }

        const lockedRes = await request(app)
          .post('/api/auth/login')
          .send({ email, password: 'Wrong123' })
          .expect(429);

        expect(lockedRes.body.error.code).toBe('AUTH_LOGIN_LOCKED_OUT');

        const afterLockoutMs = baseTimeMs + (config.auth.lockoutDurationSeconds + 1) * 1000;
        dateNowSpy.mockImplementation(() => afterLockoutMs);

        const resAfter = await request(app)
          .post('/api/auth/login')
          .send({ email, password: 'Wrong123' })
          .expect(401);

        expect(resAfter.body.error.code).toBe('AUTH_INVALID_CREDENTIALS');
      } finally {
        config.auth.lockoutDurationSeconds = originalLockoutDuration;
        dateNowSpy?.mockRestore();
      }
    });

    it('returns 500 DATABASE_UNAVAILABLE when database client is not available', async () => {
      const app = createTestApp();

      // Simulate a database connectivity failure for this request only.
      mockDatabaseClient = null;

      const res = await request(app)
        .post('/api/auth/login')
        .send({ email: 'user1@example.com', password: 'Secret123' })
        .expect(500);

      expect(res.body.success).toBe(false);
      expect(res.body.error.code).toBe('SERVER_DATABASE_UNAVAILABLE');
    });

    it('rejects invalid email format with 400 INVALID_REQUEST and no stack leak', async () => {
      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/login')
        .send({ email: 'not-an-email', password: 'Secret123' })
        .expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error.code).toBe('VALIDATION_FAILED');
      expect(res.body.error).not.toHaveProperty('stack');
      // Validation errors now include field-level details
      expect(res.body.error.details).toBeDefined();
    });
  });

  describe('POST /api/auth/refresh', () => {
    it('returns 400 REFRESH_TOKEN_REQUIRED when missing token', async () => {
      const app = createTestApp();

      const res = await request(app).post('/api/auth/refresh').send({}).expect(400);

      expect(res.body.error.code).toBe('AUTH_REFRESH_TOKEN_REQUIRED');
    });

    it('returns 401 INVALID_REFRESH_TOKEN when DB lookup fails', async () => {
      // Force verifyRefreshToken to succeed, seed a valid user record so that
      // validateUser passes, but leave the DB without a matching stored
      // refresh token so that the route returns INVALID_REFRESH_TOKEN.
      mockedAuth.verifyRefreshToken.mockReturnValueOnce({
        userId: 'user-1',
        email: 'user1@example.com',
        tokenVersion: 0,
      } as any);

      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        passwordHash: 'hashed:Secret123',
        role: 'USER',
        isActive: true,
        emailVerified: true,
        createdAt: new Date(),
      });

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/refresh')
        .send({ refreshToken: 'SOME_TOKEN' })
        .expect(401);

      expect(res.body.error.code).toBe('AUTH_REFRESH_TOKEN_INVALID');
    });

    // Happy-path refresh should rotate the stored token and return new tokens.
    it('refreshes tokens successfully with token rotation', async () => {
      const hash = (token: string) => crypto.createHash('sha256').update(token).digest('hex');

      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        passwordHash: 'hashed:Secret123',
        role: 'USER',
        isActive: true,
        emailVerified: true,
        createdAt: new Date(),
        tokenVersion: 0,
      });

      const existingRt = {
        id: 'rt-1',
        token: hash('OLD_REFRESH'),
        userId: 'user-1',
        familyId: 'family-1',
        revokedAt: null,
        expiresAt: new Date(Date.now() + 1000 * 60 * 60),
        user: {
          id: 'user-1',
          email: 'user1@example.com',
          username: 'user1',
          role: 'USER',
          isActive: true,
        },
      };
      mockDb.refreshTokens.push(existingRt);

      mockedAuth.verifyRefreshToken.mockReturnValue({
        userId: 'user-1',
        email: 'user1@example.com',
        tokenVersion: 0,
      } as any);

      const app = createTestApp();

      // First refresh with the current refresh token succeeds and rotates the stored token.
      const res = await request(app)
        .post('/api/auth/refresh')
        .send({ refreshToken: 'OLD_REFRESH' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.data.accessToken).toBe('ACCESS_TOKEN');
      expect(res.body.data.refreshToken).toBe('REFRESH_TOKEN');

      // Old token should be marked as revoked (not deleted), new one created.
      expect(prismaStub.refreshToken.update).toHaveBeenCalledWith({
        where: { id: 'rt-1' },
        data: { revokedAt: expect.any(Date) },
      });
      expect(prismaStub.refreshToken.create).toHaveBeenCalled();
    });

    // Reuse of a revoked token should trigger family revocation (security feature).
    it('detects refresh token reuse and revokes entire token family', async () => {
      const hash = (token: string) => crypto.createHash('sha256').update(token).digest('hex');

      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        passwordHash: 'hashed:Secret123',
        role: 'USER',
        isActive: true,
        emailVerified: true,
        createdAt: new Date(),
        tokenVersion: 0,
      });

      // A token that was already revoked (rotated previously)
      const revokedRt = {
        id: 'rt-1',
        token: hash('OLD_REFRESH'),
        userId: 'user-1',
        familyId: 'family-1',
        revokedAt: new Date(Date.now() - 1000), // Already revoked
        expiresAt: new Date(Date.now() + 1000 * 60 * 60),
        user: {
          id: 'user-1',
          email: 'user1@example.com',
          username: 'user1',
          role: 'USER',
          isActive: true,
        },
      };
      mockDb.refreshTokens.push(revokedRt);

      // Also add a newer token in the same family that's still valid
      mockDb.refreshTokens.push({
        id: 'rt-2',
        token: hash('NEW_REFRESH'),
        userId: 'user-1',
        familyId: 'family-1',
        revokedAt: null,
        expiresAt: new Date(Date.now() + 1000 * 60 * 60),
      });

      mockedAuth.verifyRefreshToken.mockReturnValue({
        userId: 'user-1',
        email: 'user1@example.com',
        tokenVersion: 0,
      } as any);

      const app = createTestApp();

      // Attempting to reuse the old (already revoked) token should fail
      const res = await request(app)
        .post('/api/auth/refresh')
        .send({ refreshToken: 'OLD_REFRESH' })
        .expect(401);

      expect(res.body.error.code).toBe('AUTH_REFRESH_TOKEN_REUSED');

      // The entire family should be revoked
      expect(prismaStub.refreshToken.updateMany).toHaveBeenCalledWith({
        where: { familyId: 'family-1' },
        data: { revokedAt: expect.any(Date) },
      });

      // User's tokenVersion should be incremented to invalidate access tokens too
      expect(prismaStub.user.update).toHaveBeenCalledWith({
        where: { id: 'user-1' },
        data: { tokenVersion: { increment: 1 } },
      });
    });

    it('returns 401 REFRESH_TOKEN_EXPIRED for expired refresh tokens', async () => {
      const hash = (token: string) => crypto.createHash('sha256').update(token).digest('hex');

      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        passwordHash: 'hashed:Secret123',
        role: 'USER',
        isActive: true,
        emailVerified: true,
        createdAt: new Date(),
        tokenVersion: 0,
      });

      // An expired token
      const expiredRt = {
        id: 'rt-1',
        token: hash('EXPIRED_REFRESH'),
        userId: 'user-1',
        familyId: 'family-1',
        revokedAt: null,
        expiresAt: new Date(Date.now() - 1000), // Already expired
        user: {
          id: 'user-1',
          email: 'user1@example.com',
          username: 'user1',
          role: 'USER',
          isActive: true,
        },
      };
      mockDb.refreshTokens.push(expiredRt);

      mockedAuth.verifyRefreshToken.mockReturnValue({
        userId: 'user-1',
        email: 'user1@example.com',
        tokenVersion: 0,
      } as any);

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/refresh')
        .send({ refreshToken: 'EXPIRED_REFRESH' })
        .expect(401);

      expect(res.body.error.code).toBe('AUTH_REFRESH_TOKEN_EXPIRED');
    });
  });

  describe('POST /api/auth/logout', () => {
    it('returns success and revokes the refresh token if provided', async () => {
      const hash = (token: string) => crypto.createHash('sha256').update(token).digest('hex');

      mockDb.refreshTokens.push({
        id: 'rt-1',
        token: hash('MY_REFRESH_TOKEN'),
        userId: 'auth-user-1',
        familyId: 'family-1',
        revokedAt: null,
        expiresAt: new Date(Date.now() + 1000 * 60 * 60),
      });

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/logout')
        .send({ refreshToken: 'MY_REFRESH_TOKEN' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.message).toBe('Logged out successfully');

      // The refresh token should be marked as revoked
      expect(prismaStub.refreshToken.updateMany).toHaveBeenCalledWith({
        where: {
          token: hash('MY_REFRESH_TOKEN'),
          userId: 'auth-user-1',
        },
        data: { revokedAt: expect.any(Date) },
      });
    });

    it('returns success even without a refresh token present', async () => {
      const app = createTestApp();

      const res = await request(app).post('/api/auth/logout').send({}).expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.message).toBe('Logged out successfully');
    });

    it('returns 401 TOKEN_REQUIRED when authentication fails', async () => {
      const app = createTestApp();

      // Force the mocked authenticate middleware to fail for this request.
      mockedAuth.authenticate.mockImplementationOnce(async (_req: any, _res: any, next: any) => {
        next(createError('Authentication token required', 401, 'TOKEN_REQUIRED'));
      });

      const res = await request(app).post('/api/auth/logout').send({}).expect(401);

      expect(res.body.success).toBe(false);
      expect(res.body.error.code).toBe('AUTH_TOKEN_REQUIRED');
    });
  });

  describe('POST /api/auth/logout-all', () => {
    it('returns 401 TOKEN_REQUIRED when authentication fails', async () => {
      const app = createTestApp();

      mockedAuth.authenticate.mockImplementationOnce(async (_req: any, _res: any, next: any) => {
        next(createError('Authentication token required', 401, 'TOKEN_REQUIRED'));
      });

      const res = await request(app).post('/api/auth/logout-all').send({}).expect(401);

      expect(res.body.success).toBe(false);
      expect(res.body.error.code).toBe('AUTH_TOKEN_REQUIRED');
    });

    it('increments tokenVersion and deletes all refresh tokens for the authenticated user', async () => {
      // Seed a user with an initial tokenVersion of 0.
      mockDb.users.push({
        id: 'auth-user-1',
        email: 'user1@example.com',
        username: 'user1',
        passwordHash: 'hashed:Secret123',
        role: 'USER',
        isActive: true,
        emailVerified: true,
        createdAt: new Date(),
        tokenVersion: 0,
      });

      // Seed multiple refresh tokens across two users.
      mockDb.refreshTokens.push(
        { id: 'rt-1', token: 'T1', userId: 'auth-user-1', expiresAt: new Date() },
        { id: 'rt-2', token: 'T2', userId: 'auth-user-1', expiresAt: new Date() },
        { id: 'rt-3', token: 'T3', userId: 'user-2', expiresAt: new Date() }
      );

      const app = createTestApp();

      // Ensure the mocked authenticate middleware sets the expected user id.
      mockedAuth.authenticate.mockImplementationOnce(async (req: any, _res: any, next: any) => {
        req.user = {
          id: 'auth-user-1',
          email: 'user1@example.com',
          username: 'user1',
          role: 'USER',
        };
        next();
      });

      const res = await request(app).post('/api/auth/logout-all').send({}).expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.message).toBe('Logged out from all devices successfully');

      expect(prismaStub.user.update).toHaveBeenCalledWith({
        where: { id: 'auth-user-1' },
        data: {
          tokenVersion: {
            increment: 1,
          },
        },
      });

      // The Prisma stub applies the increment; verify in-memory state.
      const updatedUser = mockDb.users.find((u) => u.id === 'auth-user-1');
      expect(updatedUser?.tokenVersion).toBe(1);

      expect(prismaStub.refreshToken.deleteMany).toHaveBeenCalledWith({
        where: { userId: 'auth-user-1' },
      });
    });
  });

  describe('Email and password reset flows (current implementation)', () => {
    it('verify-email returns 400 INVALID_TOKEN for an unknown or expired token', async () => {
      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/verify-email')
        .send({ token: 'dummy' })
        .expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error.code).toBe('AUTH_TOKEN_INVALID');
    });

    it('forgot-password returns a generic success message for existing or unknown emails', async () => {
      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/forgot-password')
        .send({ email: 'user@example.com' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.message).toContain('a password reset link has been sent');
    });

    it('reset-password returns 400 INVALID_TOKEN for an unknown or expired token', async () => {
      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/reset-password')
        .send({ token: 'dummy', newPassword: 'new-Secret123' })
        .expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error.code).toBe('AUTH_TOKEN_INVALID');
    });
  });

  describe('POST /api/auth/reset-password security', () => {
    it('invalidates all tokens after password reset by incrementing tokenVersion', async () => {
      const resetToken = 'valid-reset-token';

      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        passwordHash: 'hashed:OldPassword123',
        role: 'USER',
        isActive: true,
        emailVerified: true,
        createdAt: new Date(),
        tokenVersion: 0,
        passwordResetToken: resetToken,
        passwordResetExpires: new Date(Date.now() + 1000 * 60 * 60), // Valid for 1 hour
      });

      // Add some refresh tokens that should be revoked
      mockDb.refreshTokens.push({
        id: 'rt-1',
        token: 'some-token-hash',
        userId: 'user-1',
        familyId: 'family-1',
        revokedAt: null,
        expiresAt: new Date(Date.now() + 1000 * 60 * 60),
      });

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/reset-password')
        .send({ token: resetToken, newPassword: 'NewSecure123' })
        .expect(200);

      expect(res.body.success).toBe(true);

      // tokenVersion should be incremented to invalidate existing access tokens
      expect(prismaStub.user.update).toHaveBeenCalledWith(
        expect.objectContaining({
          where: { id: 'user-1' },
          data: expect.objectContaining({
            tokenVersion: { increment: 1 },
          }),
        })
      );

      // All refresh tokens should be revoked
      expect(prismaStub.refreshToken.updateMany).toHaveBeenCalledWith({
        where: { userId: 'user-1' },
        data: { revokedAt: expect.any(Date) },
      });
    });
  });

  describe('Token family tracking', () => {
    it('creates new token family on login', async () => {
      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        passwordHash: 'hashed:Secret123',
        role: 'USER',
        isActive: true,
        emailVerified: true,
        createdAt: new Date(),
        tokenVersion: 0,
      });

      const app = createTestApp();

      await request(app)
        .post('/api/auth/login')
        .send({ email: 'user1@example.com', password: 'Secret123' })
        .expect(200);

      // Check that a familyId was provided when creating the refresh token
      expect(prismaStub.refreshToken.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          familyId: expect.any(String),
        }),
      });
    });

    it('creates new token family on registration', async () => {
      const app = createTestApp();

      await request(app)
        .post('/api/auth/register')
        .send({
          email: 'newuser@example.com',
          username: 'newuser',
          password: 'Secret123',
          confirmPassword: 'Secret123',
        })
        .expect(201);

      // Check that a familyId was provided when creating the refresh token
      expect(prismaStub.refreshToken.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          familyId: expect.any(String),
        }),
      });
    });
  });

  describe('Cookie handling', () => {
    it('sets httpOnly cookie on login', async () => {
      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        passwordHash: 'hashed:Secret123',
        role: 'USER',
        isActive: true,
        emailVerified: true,
        createdAt: new Date(),
        tokenVersion: 0,
      });

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/login')
        .send({ email: 'user1@example.com', password: 'Secret123' })
        .expect(200);

      // Check that Set-Cookie header is present with refreshToken
      const cookies = res.headers['set-cookie'];
      expect(cookies).toBeDefined();
      expect(Array.isArray(cookies) ? cookies.join(';') : cookies).toContain('refreshToken=');
      expect(Array.isArray(cookies) ? cookies.join(';') : cookies).toContain('HttpOnly');
    });

    it('sets httpOnly cookie on registration', async () => {
      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'newuser@example.com',
          username: 'newuser',
          password: 'Secret123',
          confirmPassword: 'Secret123',
        })
        .expect(201);

      // Check that Set-Cookie header is present with refreshToken
      const cookies = res.headers['set-cookie'];
      expect(cookies).toBeDefined();
      expect(Array.isArray(cookies) ? cookies.join(';') : cookies).toContain('refreshToken=');
      expect(Array.isArray(cookies) ? cookies.join(';') : cookies).toContain('HttpOnly');
    });

    it('clears cookie on logout', async () => {
      const app = createTestApp();

      const res = await request(app).post('/api/auth/logout').send({}).expect(200);

      // Check that Set-Cookie header clears the refreshToken
      const cookies = res.headers['set-cookie'];
      expect(cookies).toBeDefined();
      // A cleared cookie typically has an empty value or past expiry
      const cookieStr = Array.isArray(cookies) ? cookies.join(';') : cookies;
      expect(cookieStr).toMatch(/refreshToken=/);
    });
  });

  describe('Auth logging hygiene', () => {
    it('does not log raw passwords or refresh tokens on auth failures', async () => {
      const app = createTestApp();

      const warnSpy = jest.spyOn(logger, 'warn');
      const errorSpy = jest.spyOn(logger, 'error');

      const passwordSentinel = 'SuperSecretPassword!123-LOG-SENTINEL';
      const refreshTokenSentinel = 'REFRESH_TOKEN_LOG_SENTINEL';

      await request(app)
        .post('/api/auth/login')
        .send({ email: 'missing@example.com', password: passwordSentinel })
        .expect(401);

      await request(app)
        .post('/api/auth/refresh')
        .send({ refreshToken: refreshTokenSentinel })
        .expect(401);

      const allCalls = [...warnSpy.mock.calls, ...errorSpy.mock.calls];
      const serialized = allCalls
        .map((args) => {
          try {
            return JSON.stringify(args);
          } catch {
            return String(args);
          }
        })
        .join('\n');

      expect(serialized).not.toContain(passwordSentinel);
      expect(serialized).not.toContain(refreshTokenSentinel);

      warnSpy.mockRestore();
      errorSpy.mockRestore();
    });
  });
});
