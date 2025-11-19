import express from 'express';
import request from 'supertest';
import authRoutes from '../../src/server/routes/auth';
import { errorHandler } from '../../src/server/middleware/errorHandler';
import * as authModule from '../../src/server/middleware/auth';
import { mockDb, prismaStub, resetPrismaMockDb } from '../utils/prismaTestUtils';

// --- Mocks --------------------------------------------------------------

// Stub out rate limiting so tests don't depend on Redis or global state.
jest.mock('../../src/server/middleware/rateLimiter', () => ({
  authRateLimiter: (_req: any, _res: any, next: any) => next(),
  rateLimiter: (_req: any, _res: any, next: any) => next(),
}));

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => prismaStub,
}));

// Mock auth helpers so we don't depend on real JWT configuration.
jest.mock('../../src/server/middleware/auth', () => {
  const actual = jest.requireActual('../../src/server/middleware/auth');
  const generateToken = jest.fn(() => 'ACCESS_TOKEN');
  const generateRefreshToken = jest.fn(() => 'REFRESH_TOKEN');
  const verifyRefreshToken = jest.fn(() => ({
    userId: 'user-1',
    email: 'user1@example.com',
  }));

  return {
    ...actual,
    generateToken,
    generateRefreshToken,
    verifyRefreshToken,
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
    resetPrismaMockDb();
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
        password: 'hashed:Secret123',
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
      expect(res.body.error.code).toBe('EMAIL_EXISTS');
    });

    it('returns 409 USERNAME_EXISTS when username already taken', async () => {
      mockDb.users.push({
        id: 'user-1',
        email: 'other@example.com',
        username: 'user1',
        password: 'hashed:Secret123',
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
      expect(res.body.error.code).toBe('USERNAME_EXISTS');
    });
  });

  describe('POST /api/auth/login', () => {
    it('returns 401 INVALID_CREDENTIALS when user not found', async () => {
      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/login')
        .send({ email: 'missing@example.com', password: 'Secret123' })
        .expect(401);

      expect(res.body.error.code).toBe('INVALID_CREDENTIALS');
    });

    // Inactive users should be rejected before password verification.
    it('returns 401 ACCOUNT_DEACTIVATED for inactive user', async () => {
      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        password: 'hashed:Secret123',
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

      expect(res.body.error.code).toBe('ACCOUNT_DEACTIVATED');
    });

    it('returns 401 INVALID_CREDENTIALS for wrong password', async () => {
      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        password: 'hashed:Correct123',
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

      expect(res.body.error.code).toBe('INVALID_CREDENTIALS');
    });

    it('logs in successfully and returns tokens', async () => {
      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        password: 'hashed:Secret123',
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
  });

  describe('POST /api/auth/refresh', () => {
    it('returns 400 REFRESH_TOKEN_REQUIRED when missing token', async () => {
      const app = createTestApp();

      const res = await request(app).post('/api/auth/refresh').send({}).expect(400);

      expect(res.body.error.code).toBe('REFRESH_TOKEN_REQUIRED');
    });

    it('returns 401 INVALID_REFRESH_TOKEN when DB lookup fails', async () => {
      // Force verifyRefreshToken to succeed but DB to return no token.
      mockedAuth.verifyRefreshToken.mockReturnValueOnce({
        userId: 'user-1',
        email: 'user1@example.com',
      } as any);

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/refresh')
        .send({ refreshToken: 'SOME_TOKEN' })
        .expect(401);

      expect(res.body.error.code).toBe('INVALID_REFRESH_TOKEN');
    });

    // Happy-path refresh should rotate the stored token and return new tokens.
    it('refreshes tokens successfully when refresh token is valid', async () => {
      mockDb.users.push({
        id: 'user-1',
        email: 'user1@example.com',
        username: 'user1',
        password: 'hashed:Secret123',
        role: 'USER',
        isActive: true,
        emailVerified: true,
        createdAt: new Date(),
      });

      const existingRt = {
        id: 'rt-1',
        token: 'OLD_REFRESH',
        userId: 'user-1',
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

      mockedAuth.verifyRefreshToken.mockReturnValueOnce({
        userId: 'user-1',
        email: 'user1@example.com',
      } as any);

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/refresh')
        .send({ refreshToken: 'OLD_REFRESH' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.data.accessToken).toBe('ACCESS_TOKEN');
      expect(res.body.data.refreshToken).toBe('REFRESH_TOKEN');

      // Old token should be deleted; new one created via $transaction.
      expect(prismaStub.refreshToken.delete).toHaveBeenCalled();
      expect(prismaStub.refreshToken.create).toHaveBeenCalled();
    });
  });

  describe('POST /api/auth/logout', () => {
    it('is idempotent when no refreshToken provided', async () => {
      const app = createTestApp();

      const res = await request(app).post('/api/auth/logout').send({}).expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.message).toBe('Logged out successfully');
    });

    it('deletes matching refresh tokens when provided', async () => {
      mockDb.refreshTokens.push({
        id: 'rt-1',
        token: 'TOKEN_TO_DELETE',
        userId: 'user-1',
        expiresAt: new Date(Date.now() + 1000 * 60 * 60),
      });

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/logout')
        .send({ refreshToken: 'TOKEN_TO_DELETE' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(prismaStub.refreshToken.deleteMany).toHaveBeenCalledWith({
        where: { token: 'TOKEN_TO_DELETE' },
      });
    });
  });

  describe('POST /api/auth/logout-all', () => {
    it('returns 400 REFRESH_TOKEN_REQUIRED when missing token', async () => {
      const app = createTestApp();

      const res = await request(app).post('/api/auth/logout-all').send({}).expect(400);

      expect(res.body.error.code).toBe('REFRESH_TOKEN_REQUIRED');
    });

    it('deletes all refresh tokens for the user when token is valid', async () => {
      mockDb.refreshTokens.push(
        { id: 'rt-1', token: 'T1', userId: 'user-1', expiresAt: new Date() },
        { id: 'rt-2', token: 'T2', userId: 'user-1', expiresAt: new Date() },
        { id: 'rt-3', token: 'T3', userId: 'user-2', expiresAt: new Date() }
      );

      mockedAuth.verifyRefreshToken.mockReturnValueOnce({
        userId: 'user-1',
        email: 'user1@example.com',
      } as any);

      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/logout-all')
        .send({ refreshToken: 'ANY_VALID_TOKEN' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.message).toBe('Logged out from all devices successfully');
      expect(prismaStub.refreshToken.deleteMany).toHaveBeenCalledWith({
        where: { userId: 'user-1' },
      });
    });
  });

  describe('Email and password reset placeholders', () => {
    it('verify-email returns not implemented message', async () => {
      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/verify-email')
        .send({ token: 'dummy' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.message).toMatch(/not implemented/i);
    });

    it('forgot-password returns not implemented message', async () => {
      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/forgot-password')
        .send({ email: 'user@example.com' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.message).toMatch(/not implemented/i);
    });

    it('reset-password returns not implemented message', async () => {
      const app = createTestApp();

      const res = await request(app)
        .post('/api/auth/reset-password')
        .send({ token: 'dummy', newPassword: 'new-Secret123' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.message).toMatch(/not implemented/i);
    });
  });
});
