/**
 * Integration tests for Account Deletion (DELETE /api/users/me)
 *
 * Tests the soft-delete functionality including:
 * - Successful deletion with authentication
 * - Rejection without authentication
 * - Token invalidation after deletion
 * - PII anonymization (email/username)
 * - Login prevention after deletion
 * - Email reuse after deletion
 * - Idempotent handling of already-deleted accounts
 */

import request from 'supertest';
import express, { Express, Request, Response, NextFunction } from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

// Mock the database connection, but keep timeout wrappers real so route code
// (which uses withQueryTimeoutStrict) behaves the same as production.
jest.mock('../../src/server/database/connection', () => {
  const actual = jest.requireActual('../../src/server/database/connection');
  return {
    ...actual,
    getDatabaseClient: jest.fn(),
  };
});

// Mock the logger to reduce noise
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
  httpLogger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
  redactEmail: (email: string) => email.replace(/(.{2}).*(@.*)/, '$1***$2'),
  getRequestContext: jest.fn(() => ({})),
  withRequestContext: jest.fn((_req: any, context: any) => context),
}));

// Mock the cache service (redis)
jest.mock('../../src/server/cache/redis', () => ({
  getCacheService: jest.fn(() => null), // No cache for tests
  CacheKeys: {
    authLoginLockout: (email: string) => `auth:lockout:${email}`,
    authLoginFailures: (email: string) => `auth:failures:${email}`,
  },
}));

// Mock email service
jest.mock('../../src/server/utils/email', () => ({
  sendVerificationEmail: jest.fn().mockResolvedValue(undefined),
  sendPasswordResetEmail: jest.fn().mockResolvedValue(undefined),
}));

// Mock config
jest.mock('../../src/server/config', () => ({
  config: {
    isProduction: false,
    isDevelopment: true,
    auth: {
      jwtSecret: 'test-jwt-secret-12345678901234567890',
      jwtRefreshSecret: 'test-refresh-secret-12345678901234567890',
      accessTokenExpiresIn: '15m',
      refreshTokenExpiresIn: '7d',
      loginLockoutEnabled: false,
      maxFailedLoginAttempts: 5,
      failedLoginWindowSeconds: 900,
      lockoutDurationSeconds: 1800,
    },
  },
  // Mock parseDurationToSeconds - used by auth route to calculate token TTL
  parseDurationToSeconds: (duration: string): number => {
    const match = duration.match(/^(\d+)([smhd])$/);
    if (!match) return 0;
    const value = parseInt(match[1], 10);
    const unit = match[2];
    switch (unit) {
      case 's':
        return value;
      case 'm':
        return value * 60;
      case 'h':
        return value * 3600;
      case 'd':
        return value * 86400;
      default:
        return 0;
    }
  },
}));

import { getDatabaseClient } from '../../src/server/database/connection';
import userRouter from '../../src/server/routes/user';
import authRouter from '../../src/server/routes/auth';
import { authenticate } from '../../src/server/middleware/auth';
import { errorHandler } from '../../src/server/middleware/errorHandler';

// =============================================================================
// In-Memory Database Mock
// =============================================================================

// Type for the mock Prisma client to avoid implicit any
interface MockPrismaClient {
  user: {
    findUnique: jest.Mock;
    findFirst: jest.Mock;
    create: jest.Mock;
    update: jest.Mock;
  };
  refreshToken: {
    create: jest.Mock;
    findFirst: jest.Mock;
    deleteMany: jest.Mock;
    updateMany: jest.Mock;
  };
  $transaction: jest.Mock;
}

interface MockUser {
  id: string;
  email: string;
  username: string;
  passwordHash: string;
  role: string;
  rating: number;
  gamesPlayed: number;
  gamesWon: number;
  emailVerified: boolean;
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
  lastLoginAt: Date | null;
  tokenVersion: number;
  deletedAt: Date | null;
  verificationToken: string | null;
  verificationTokenExpires: Date | null;
  passwordResetToken: string | null;
  passwordResetExpires: Date | null;
}

interface MockRefreshToken {
  id: string;
  token: string;
  userId: string;
  familyId: string | null;
  expiresAt: Date;
  createdAt: Date;
  revokedAt: Date | null;
}

// In-memory stores
let users: Map<string, MockUser>;
let refreshTokens: Map<string, MockRefreshToken>;
let userIdCounter: number;

function resetDatabase() {
  users = new Map();
  refreshTokens = new Map();
  userIdCounter = 0;
}

function createMockPrisma(): MockPrismaClient {
  return {
    user: {
      findUnique: jest.fn(async ({ where }: { where: { id?: string; email?: string } }) => {
        if (where.id) {
          return users.get(where.id) || null;
        }
        if (where.email) {
          for (const user of users.values()) {
            if (user.email === where.email) return user;
          }
        }
        return null;
      }),
      findFirst: jest.fn(async ({ where }: { where: any }) => {
        for (const user of users.values()) {
          let matches = true;
          if (where.email && user.email !== where.email) matches = false;
          if (where.username && user.username !== where.username) matches = false;
          if (where.OR) {
            matches = where.OR.some((cond: any) => {
              if (cond.email && user.email === cond.email) return true;
              if (cond.username && user.username === cond.username) return true;
              return false;
            });
          }
          if (where.verificationToken && user.verificationToken !== where.verificationToken) {
            matches = false;
          }
          if (where.passwordResetToken && user.passwordResetToken !== where.passwordResetToken) {
            matches = false;
          }
          if (matches) return user;
        }
        return null;
      }),
      create: jest.fn(async ({ data }: { data: any }) => {
        userIdCounter++;
        const id = `user-${userIdCounter}`;
        const user: MockUser = {
          id,
          email: data.email,
          username: data.username,
          passwordHash: data.passwordHash,
          role: data.role || 'USER',
          rating: data.rating || 1200,
          gamesPlayed: data.gamesPlayed || 0,
          gamesWon: data.gamesWon || 0,
          emailVerified: data.emailVerified || false,
          isActive: data.isActive !== undefined ? data.isActive : true,
          createdAt: new Date(),
          updatedAt: new Date(),
          lastLoginAt: null,
          tokenVersion: data.tokenVersion || 0,
          deletedAt: null,
          verificationToken: data.verificationToken || null,
          verificationTokenExpires: data.verificationTokenExpires || null,
          passwordResetToken: data.passwordResetToken || null,
          passwordResetExpires: data.passwordResetExpires || null,
        };
        users.set(id, user);
        return user;
      }),
      update: jest.fn(async ({ where, data }: { where: { id: string }; data: any }) => {
        const user = users.get(where.id);
        if (!user) throw new Error('User not found');

        if (data.isActive !== undefined) user.isActive = data.isActive;
        if (data.deletedAt !== undefined) user.deletedAt = data.deletedAt;
        if (data.email !== undefined) user.email = data.email;
        if (data.username !== undefined) user.username = data.username;
        if (data.lastLoginAt !== undefined) user.lastLoginAt = data.lastLoginAt;
        if (data.passwordHash !== undefined) user.passwordHash = data.passwordHash;
        if (data.verificationToken !== undefined) user.verificationToken = data.verificationToken;
        if (data.verificationTokenExpires !== undefined)
          user.verificationTokenExpires = data.verificationTokenExpires;
        if (data.passwordResetToken !== undefined)
          user.passwordResetToken = data.passwordResetToken;
        if (data.passwordResetExpires !== undefined)
          user.passwordResetExpires = data.passwordResetExpires;
        if (data.tokenVersion !== undefined) {
          if (typeof data.tokenVersion === 'object' && data.tokenVersion.increment) {
            user.tokenVersion += data.tokenVersion.increment;
          } else {
            user.tokenVersion = data.tokenVersion;
          }
        }
        user.updatedAt = new Date();

        return user;
      }),
    },
    refreshToken: {
      create: jest.fn(async ({ data }: { data: any }) => {
        const id = `rt-${Date.now()}-${Math.random()}`;
        const token: MockRefreshToken = {
          id,
          token: data.token,
          userId: data.userId,
          familyId: data.familyId || null,
          expiresAt: data.expiresAt,
          createdAt: new Date(),
          revokedAt: null,
        };
        refreshTokens.set(id, token);
        return token;
      }),
      findFirst: jest.fn(async ({ where }: { where: any }) => {
        for (const token of refreshTokens.values()) {
          let matches = true;
          if (where.token && token.token !== where.token) matches = false;
          if (where.userId && token.userId !== where.userId) matches = false;
          if (matches) {
            const user = users.get(token.userId);
            return { ...token, user };
          }
        }
        return null;
      }),
      deleteMany: jest.fn(async ({ where }: { where: { userId: string } }) => {
        const toDelete: string[] = [];
        for (const [id, token] of refreshTokens.entries()) {
          if (token.userId === where.userId) {
            toDelete.push(id);
          }
        }
        toDelete.forEach((id) => refreshTokens.delete(id));
        return { count: toDelete.length };
      }),
      updateMany: jest.fn(async ({ where, data }: { where: any; data: any }) => {
        let count = 0;
        for (const token of refreshTokens.values()) {
          let matches = true;
          if (where.token && token.token !== where.token) matches = false;
          if (where.userId && token.userId !== where.userId) matches = false;
          if (where.familyId && token.familyId !== where.familyId) matches = false;
          if (matches) {
            if (data.revokedAt !== undefined) token.revokedAt = data.revokedAt;
            count++;
          }
        }
        return { count };
      }),
    },
    $transaction: jest.fn(async (callback: (tx: MockPrismaClient) => Promise<unknown>) => {
      // For transaction, we pass the current mock prisma instance directly
      // This avoids a recursive createMockPrisma call
      const currentPrisma = getDatabaseClient() as unknown as MockPrismaClient;
      return callback(currentPrisma);
    }),
  };
}

// =============================================================================
// Test Setup
// =============================================================================

let app: Express;
let mockPrisma: MockPrismaClient;

function setupApp(): Express {
  app = express();
  app.use(express.json());

  // Mount routers
  app.use('/api/auth', authRouter);
  app.use('/api/users', authenticate, userRouter);

  // Error handler
  app.use(errorHandler);

  return app;
}

// =============================================================================
// Test Helpers
// =============================================================================

async function createTestUser(
  email: string = 'test@example.com',
  username: string = 'testuser',
  password: string = 'password123'
): Promise<{ user: MockUser; plainPassword: string }> {
  const passwordHash = await bcrypt.hash(password, 10);
  const user = await mockPrisma.user.create({
    data: {
      email,
      username,
      passwordHash,
      role: 'USER',
      isActive: true,
      emailVerified: true,
    },
  });
  return { user: user as MockUser, plainPassword: password };
}

function generateAccessToken(user: MockUser): string {
  return jwt.sign(
    {
      userId: user.id,
      email: user.email,
      tv: user.tokenVersion,
    },
    'test-jwt-secret-12345678901234567890',
    {
      expiresIn: '15m',
      issuer: 'ringrift',
      audience: 'ringrift-users',
    }
  );
}

async function loginUser(email: string, password: string): Promise<{ accessToken: string }> {
  const res = await request(app).post('/api/auth/login').send({ email, password });

  if (res.status !== 200) {
    throw new Error(`Login failed: ${res.status} - ${JSON.stringify(res.body)}`);
  }

  return { accessToken: res.body.data.accessToken };
}

function getUser(userId: string): MockUser | undefined {
  return users.get(userId);
}

// =============================================================================
// Tests
// =============================================================================

describe('DELETE /api/users/me - Account Deletion', () => {
  beforeEach(() => {
    resetDatabase();
    mockPrisma = createMockPrisma();
    (getDatabaseClient as jest.Mock).mockReturnValue(mockPrisma);
    setupApp();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Authentication Requirements', () => {
    test('should reject unauthenticated deletion request with 401', async () => {
      // Attempt DELETE without any auth header
      const res = await request(app).delete('/api/users/me');

      expect(res.status).toBe(401);
      expect(res.body.success).toBe(false);
      expect(res.body.error.code).toMatch(/TOKEN_REQUIRED|AUTH_REQUIRED/);
    });

    test('should reject deletion with invalid token', async () => {
      const res = await request(app)
        .delete('/api/users/me')
        .set('Authorization', 'Bearer invalid-token-12345');

      expect(res.status).toBe(401);
      expect(res.body.success).toBe(false);
    });
  });

  describe('Successful Deletion', () => {
    test('should successfully delete authenticated user account', async () => {
      // 1. Create user
      const { user } = await createTestUser('delete-me@example.com', 'deleteme');

      // 2. Generate token
      const accessToken = generateAccessToken(user);

      // 3. DELETE /api/users/me with auth header
      const res = await request(app)
        .delete('/api/users/me')
        .set('Authorization', `Bearer ${accessToken}`);

      // 4. Verify 200 response
      expect(res.status).toBe(200);
      expect(res.body.success).toBe(true);
      expect(res.body.message).toBe('Account deleted successfully');

      // 5. Verify user.deletedAt is set
      const deletedUser = getUser(user.id);
      expect(deletedUser).toBeDefined();
      expect(deletedUser!.deletedAt).toBeInstanceOf(Date);

      // 6. Verify user.isActive is false
      expect(deletedUser!.isActive).toBe(false);
    });

    test('should increment tokenVersion for JWT invalidation', async () => {
      const { user } = await createTestUser('token-test@example.com', 'tokentest');
      const originalTokenVersion = user.tokenVersion;
      const accessToken = generateAccessToken(user);

      await request(app).delete('/api/users/me').set('Authorization', `Bearer ${accessToken}`);

      const deletedUser = getUser(user.id);
      expect(deletedUser!.tokenVersion).toBe(originalTokenVersion + 1);
    });
  });

  describe('Token Invalidation', () => {
    test('should invalidate existing tokens after deletion', async () => {
      // 1. Create user, get token
      const { user } = await createTestUser('token-invalidation@example.com', 'invalidateuser');
      const accessToken = generateAccessToken(user);

      // Verify token works before deletion
      const profileRes1 = await request(app)
        .get('/api/users/profile')
        .set('Authorization', `Bearer ${accessToken}`);
      expect(profileRes1.status).toBe(200);

      // 2. DELETE account
      const deleteRes = await request(app)
        .delete('/api/users/me')
        .set('Authorization', `Bearer ${accessToken}`);
      expect(deleteRes.status).toBe(200);

      // 3. Try to use old token for any authenticated endpoint
      const profileRes2 = await request(app)
        .get('/api/users/profile')
        .set('Authorization', `Bearer ${accessToken}`);

      // 4. Verify 401 response (token should be invalid due to tokenVersion mismatch)
      expect(profileRes2.status).toBe(401);
    });
  });

  describe('PII Anonymization', () => {
    test('should anonymize user email and username', async () => {
      // 1. Create user with known email/username
      const originalEmail = 'pii-test@personal-domain.com';
      const originalUsername = 'PersonalUsername123';
      const { user } = await createTestUser(originalEmail, originalUsername);
      const accessToken = generateAccessToken(user);

      // 2. DELETE account
      await request(app).delete('/api/users/me').set('Authorization', `Bearer ${accessToken}`);

      // 3. Query database directly for user
      const deletedUser = getUser(user.id);

      // 4. Verify email is anonymized (matches pattern deleted+{id}@example.invalid)
      expect(deletedUser!.email).toMatch(/^deleted\+.+@example\.invalid$/);
      expect(deletedUser!.email).not.toBe(originalEmail);

      // 5. Verify username is anonymized (matches pattern DeletedPlayer_{first8chars})
      // Note: user IDs may contain hyphens, so we allow them in the pattern
      expect(deletedUser!.username).toMatch(/^DeletedPlayer_[a-z0-9-]+$/i);
      expect(deletedUser!.username).not.toBe(originalUsername);
    });
  });

  describe('Login Prevention', () => {
    test('should prevent login after account deletion', async () => {
      // 1. Create user
      const email = 'no-login@example.com';
      const password = 'securepassword123';
      const { user } = await createTestUser(email, 'nologinuser', password);
      const accessToken = generateAccessToken(user);

      // Verify login works before deletion
      const loginRes1 = await request(app).post('/api/auth/login').send({ email, password });
      expect(loginRes1.status).toBe(200);

      // 2. DELETE account
      await request(app).delete('/api/users/me').set('Authorization', `Bearer ${accessToken}`);

      // 3. Try to login with original credentials
      const loginRes2 = await request(app).post('/api/auth/login').send({ email, password });

      // 4. Verify login fails (401 - account is deactivated or credentials invalid due to anonymization)
      expect(loginRes2.status).toBe(401);
    });
  });

  describe('Email Reuse', () => {
    test('should allow email reuse after deletion and anonymization', async () => {
      // 1. Create user with email
      const email = 'reusable@example.com';
      const { user } = await createTestUser(email, 'firstuser');
      const accessToken = generateAccessToken(user);

      // 2. DELETE account
      const deleteRes = await request(app)
        .delete('/api/users/me')
        .set('Authorization', `Bearer ${accessToken}`);
      expect(deleteRes.status).toBe(200);

      // Verify original email is no longer on the user
      const deletedUser = getUser(user.id);
      expect(deletedUser!.email).not.toBe(email);
      expect(deletedUser!.email).toMatch(/^deleted\+.+@example\.invalid$/);

      // 3. Register new user with same email (must use different username)
      // Password must have lowercase, uppercase, number, and confirmPassword is required
      const registerRes = await request(app).post('/api/auth/register').send({
        email,
        username: 'seconduser',
        password: 'NewPassword123',
        confirmPassword: 'NewPassword123',
      });

      // 4. Verify success - email can be reused since it was anonymized
      expect(registerRes.status).toBe(201);
      expect(registerRes.body.success).toBe(true);
      expect(registerRes.body.data.user.email).toBe(email);
    });
  });

  describe('Account State After Deletion', () => {
    test('should reject API calls with token from deleted account (deactivated)', async () => {
      // Create and properly delete user via the endpoint
      const { user } = await createTestUser('already-deleted@example.com', 'deleteduser');
      const accessToken = generateAccessToken(user);

      // Delete the account
      const deleteRes = await request(app)
        .delete('/api/users/me')
        .set('Authorization', `Bearer ${accessToken}`);
      expect(deleteRes.status).toBe(200);

      // Verify account is properly deactivated
      const deletedUser = getUser(user.id);
      expect(deletedUser!.isActive).toBe(false);
      expect(deletedUser!.deletedAt).toBeInstanceOf(Date);

      // Try to use the old token again - should fail because:
      // 1. isActive is false -> ACCOUNT_DEACTIVATED
      // 2. tokenVersion incremented -> INVALID_TOKEN
      const secondDeleteRes = await request(app)
        .delete('/api/users/me')
        .set('Authorization', `Bearer ${accessToken}`);

      // Should reject with 401 (either token version mismatch or account deactivated)
      expect(secondDeleteRes.status).toBe(401);
      expect(secondDeleteRes.body.success).toBe(false);
    });
  });

  describe('Refresh Token Cleanup', () => {
    test('should revoke all refresh tokens for deleted user', async () => {
      const { user } = await createTestUser('refresh-cleanup@example.com', 'refreshcleanup');

      // Create some refresh tokens for the user
      await mockPrisma.refreshToken.create({
        data: {
          token: 'hashed-token-1',
          userId: user.id,
          familyId: 'family-1',
          expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
        },
      });
      await mockPrisma.refreshToken.create({
        data: {
          token: 'hashed-token-2',
          userId: user.id,
          familyId: 'family-2',
          expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
        },
      });

      // Verify tokens exist
      expect(refreshTokens.size).toBe(2);

      const accessToken = generateAccessToken(user);

      // Delete account
      await request(app).delete('/api/users/me').set('Authorization', `Bearer ${accessToken}`);

      // Verify refresh tokens were deleted
      let userTokenCount = 0;
      for (const token of refreshTokens.values()) {
        if (token.userId === user.id) userTokenCount++;
      }
      expect(userTokenCount).toBe(0);
    });
  });
});
