/**
 * Unit tests for auth middleware
 * Tests authentication, authorization, token handling, and user context
 */

import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import {
  authenticate,
  optionalAuth,
  authorize,
  verifyToken,
  validateUser,
  generateToken,
  generateRefreshToken,
  verifyRefreshToken,
  getAccessTokenSecret,
  AuthenticatedRequest,
} from '../../../src/server/middleware/auth';

// =============================================================================
// Mocks
// =============================================================================

// Mock database connection
const mockUserFindUnique = jest.fn();
jest.mock('../../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => ({
    user: {
      findUnique: mockUserFindUnique,
    },
  })),
}));

// Mock config
jest.mock('../../../src/server/config', () => ({
  config: {
    auth: {
      jwtSecret: 'test-jwt-secret-at-least-32-characters',
      jwtRefreshSecret: 'test-refresh-secret-at-least-32-characters',
      accessTokenExpiresIn: '15m',
      refreshTokenExpiresIn: '7d',
    },
    isDevelopment: true,
    isProduction: false,
    isTest: true,
  },
}));

// Mock logger
jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
  getRequestContext: jest.fn(() => ({})),
  withRequestContext: jest.fn((req, meta) => meta),
}));

// =============================================================================
// Test Utilities
// =============================================================================

interface MockRequest extends Partial<Request> {
  headers: Record<string, string | undefined>;
  cookies?: Record<string, string>;
  query?: Record<string, string>;
  user?: AuthenticatedRequest['user'];
}

const createMockRequest = (overrides: Partial<MockRequest> = {}): MockRequest => ({
  headers: {},
  cookies: {},
  query: {},
  get: jest.fn((header: string): string | string[] | undefined => {
    const headers = overrides.headers || {};
    return headers[header.toLowerCase()];
  }) as any,
  ...overrides,
});

const createMockResponse = (): Partial<Response> => {
  const res: Partial<Response> = {
    status: jest.fn().mockReturnThis(),
    json: jest.fn().mockReturnThis(),
    send: jest.fn().mockReturnThis(),
  };
  return res;
};

const createMockNext = (): NextFunction & jest.Mock => jest.fn();

const mockUser = {
  id: 'user-123',
  email: 'test@example.com',
  username: 'testuser',
  role: 'user',
  isActive: true,
  tokenVersion: 0,
};

// =============================================================================
// Token Extraction Tests
// =============================================================================

describe('auth middleware', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUserFindUnique.mockReset();
  });

  describe('Token Extraction', () => {
    it('should extract token from Authorization header with Bearer prefix', async () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email, tokenVersion: 0 });
      const req = createMockRequest({
        headers: { authorization: `Bearer ${token}` },
      });
      const res = createMockResponse();
      const next = createMockNext();

      mockUserFindUnique.mockResolvedValue(mockUser);

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith();
      expect(req.user).toBeDefined();
      expect(req.user?.id).toBe(mockUser.id);
    });

    it('should extract token from cookies', async () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email, tokenVersion: 0 });
      const req = createMockRequest({
        cookies: { token },
      });
      const res = createMockResponse();
      const next = createMockNext();

      mockUserFindUnique.mockResolvedValue(mockUser);

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith();
      expect(req.user).toBeDefined();
    });

    it('should extract token from query parameter (WebSocket support)', async () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email, tokenVersion: 0 });
      const req = createMockRequest({
        query: { token },
      });
      const res = createMockResponse();
      const next = createMockNext();

      mockUserFindUnique.mockResolvedValue(mockUser);

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith();
      expect(req.user).toBeDefined();
    });

    it('should return 401 when Authorization header is missing', async () => {
      const req = createMockRequest();
      const res = createMockResponse();
      const next = createMockNext();

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          message: 'Authentication token required',
          statusCode: 401,
        })
      );
    });

    it('should return 401 for malformed Authorization header (missing Bearer)', async () => {
      const req = createMockRequest({
        headers: { authorization: 'invalid-token-without-bearer' },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          statusCode: 401,
        })
      );
    });

    it('should return 401 when Bearer token is empty', async () => {
      const req = createMockRequest({
        headers: { authorization: 'Bearer ' },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          statusCode: 401,
        })
      );
    });

    it('should prioritize Authorization header over cookies', async () => {
      const headerToken = generateToken({
        id: 'header-user',
        email: 'header@test.com',
        tokenVersion: 0,
      });
      const cookieToken = generateToken({
        id: 'cookie-user',
        email: 'cookie@test.com',
        tokenVersion: 0,
      });

      const req = createMockRequest({
        headers: { authorization: `Bearer ${headerToken}` },
        cookies: { token: cookieToken },
      });
      const res = createMockResponse();
      const next = createMockNext();

      mockUserFindUnique.mockResolvedValue({
        ...mockUser,
        id: 'header-user',
        email: 'header@test.com',
      });

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      expect(req.user?.id).toBe('header-user');
    });
  });

  // ===========================================================================
  // Token Validation Tests
  // ===========================================================================

  describe('Token Validation (verifyToken)', () => {
    it('should verify a valid token and extract payload', () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email, tokenVersion: 1 });

      const result = verifyToken(token);

      expect(result.userId).toBe(mockUser.id);
      expect(result.email).toBe(mockUser.email);
      expect(result.tokenVersion).toBe(1);
    });

    it('should throw TOKEN_EXPIRED for expired tokens', () => {
      const secret = getAccessTokenSecret();
      const expiredToken = jwt.sign({ userId: mockUser.id, email: mockUser.email }, secret, {
        expiresIn: '-1s',
      });

      expect(() => verifyToken(expiredToken)).toThrow(
        expect.objectContaining({
          message: 'Token has expired',
          statusCode: 401,
        })
      );
    });

    it('should throw INVALID_TOKEN for tokens with wrong signature', () => {
      const wrongSecretToken = jwt.sign(
        { userId: mockUser.id, email: mockUser.email },
        'wrong-secret-key-that-is-at-least-32-chars',
        { expiresIn: '15m' }
      );

      expect(() => verifyToken(wrongSecretToken)).toThrow(
        expect.objectContaining({
          message: 'Invalid token',
          statusCode: 401,
        })
      );
    });

    it('should throw for malformed JWT', () => {
      expect(() => verifyToken('not.a.valid.jwt.token')).toThrow(
        expect.objectContaining({
          statusCode: 401,
        })
      );
    });

    it('should throw for token with missing required claims (userId)', () => {
      const secret = getAccessTokenSecret();
      const tokenWithoutUserId = jwt.sign({ email: mockUser.email }, secret, { expiresIn: '15m' });

      expect(() => verifyToken(tokenWithoutUserId)).toThrow(
        expect.objectContaining({
          message: 'Token verification failed',
          statusCode: 401,
        })
      );
    });

    it('should throw for token with missing required claims (email)', () => {
      const secret = getAccessTokenSecret();
      const tokenWithoutEmail = jwt.sign({ userId: mockUser.id }, secret, { expiresIn: '15m' });

      expect(() => verifyToken(tokenWithoutEmail)).toThrow(
        expect.objectContaining({
          message: 'Token verification failed',
          statusCode: 401,
        })
      );
    });

    it('should handle tokenVersion=0 when tv claim is absent (backwards compatibility)', () => {
      const secret = getAccessTokenSecret();
      const tokenWithoutVersion = jwt.sign({ userId: mockUser.id, email: mockUser.email }, secret, {
        expiresIn: '15m',
      });

      const result = verifyToken(tokenWithoutVersion);

      expect(result.userId).toBe(mockUser.id);
      expect(result.tokenVersion).toBeUndefined();
    });
  });

  // ===========================================================================
  // User Context Tests
  // ===========================================================================

  describe('User Context (validateUser)', () => {
    it('should attach user object to request on successful authentication', async () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email, tokenVersion: 0 });
      const req = createMockRequest({
        headers: { authorization: `Bearer ${token}` },
      });
      const res = createMockResponse();
      const next = createMockNext();

      mockUserFindUnique.mockResolvedValue(mockUser);

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      expect(req.user).toEqual({
        id: mockUser.id,
        email: mockUser.email,
        username: mockUser.username,
        role: mockUser.role,
      });
    });

    it('should correctly extract user ID from validated user', async () => {
      mockUserFindUnique.mockResolvedValue(mockUser);

      const result = await validateUser(mockUser.id, 0);

      expect(result.id).toBe(mockUser.id);
    });

    it('should correctly attach user roles', async () => {
      const adminUser = { ...mockUser, role: 'admin' };
      mockUserFindUnique.mockResolvedValue(adminUser);

      const result = await validateUser(mockUser.id, 0);

      expect(result.role).toBe('admin');
    });

    it('should throw USER_NOT_FOUND when user does not exist', async () => {
      mockUserFindUnique.mockResolvedValue(null);

      await expect(validateUser('non-existent-user')).rejects.toThrow(
        expect.objectContaining({
          message: 'User not found',
          statusCode: 401,
        })
      );
    });

    it('should throw ACCOUNT_DEACTIVATED for inactive users', async () => {
      mockUserFindUnique.mockResolvedValue({ ...mockUser, isActive: false });

      await expect(validateUser(mockUser.id, 0)).rejects.toThrow(
        expect.objectContaining({
          message: 'Account is deactivated',
          statusCode: 401,
        })
      );
    });
  });

  // ===========================================================================
  // Error Handling Tests
  // ===========================================================================

  describe('Error Handling', () => {
    it('should return 401 for missing token', async () => {
      const req = createMockRequest();
      const res = createMockResponse();
      const next = createMockNext();

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          statusCode: 401,
        })
      );
    });

    it('should return 401 for invalid token', async () => {
      const req = createMockRequest({
        headers: { authorization: 'Bearer invalid.token.here' },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          statusCode: 401,
        })
      );
    });

    it('should return 403 for insufficient permissions (authorize)', () => {
      const req = createMockRequest() as AuthenticatedRequest;
      req.user = {
        id: mockUser.id,
        email: mockUser.email,
        username: mockUser.username,
        role: 'user',
      };
      const res = createMockResponse();
      const next = createMockNext();

      const adminOnly = authorize(['admin']);
      adminOnly(req, res as Response, next);

      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          message: 'Insufficient permissions',
          statusCode: 403,
        })
      );
    });

    it('should not leak sensitive information in error messages', async () => {
      const req = createMockRequest({
        headers: { authorization: 'Bearer invalid.token' },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      const errorArg = next.mock.calls[0][0];
      expect(errorArg.message).not.toContain('secret');
      expect(errorArg.message).not.toContain('password');
      expect(errorArg.message).not.toContain(mockUser.email);
    });

    it('should return proper error codes in responses', async () => {
      const req = createMockRequest();
      const res = createMockResponse();
      const next = createMockNext();

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      const errorArg = next.mock.calls[0][0];
      expect(errorArg.code || errorArg.message).toBeDefined();
    });

    it('should handle database unavailability gracefully', async () => {
      // Mock getDatabaseClient to return null
      jest.resetModules();
      jest.doMock('../../../src/server/database/connection', () => ({
        getDatabaseClient: () => null,
      }));

      // Re-import to get the mock
      const {
        validateUser: validateUserWithNullDb,
      } = require('../../../src/server/middleware/auth');

      await expect(validateUserWithNullDb('some-user-id')).rejects.toThrow(
        expect.objectContaining({
          message: 'Database not available',
          statusCode: 500,
        })
      );

      // Restore the original mock
      jest.resetModules();
    });

    it('should reject revoked tokens (tokenVersion mismatch)', async () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email, tokenVersion: 0 });
      const req = createMockRequest({
        headers: { authorization: `Bearer ${token}` },
      });
      const res = createMockResponse();
      const next = createMockNext();

      // User has tokenVersion 1 but token has tokenVersion 0
      mockUserFindUnique.mockResolvedValue({ ...mockUser, tokenVersion: 1 });

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          message: 'Token has been revoked',
          statusCode: 401,
        })
      );
    });
  });

  // ===========================================================================
  // Optional Auth Tests
  // ===========================================================================

  describe('Optional Auth (optionalAuth)', () => {
    it('should proceed without error when no token is provided', async () => {
      const req = createMockRequest();
      const res = createMockResponse();
      const next = createMockNext();

      await optionalAuth(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith();
      expect(req.user).toBeUndefined();
    });

    it('should validate and attach user when valid token is provided', async () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email, tokenVersion: 0 });
      const req = createMockRequest({
        headers: { authorization: `Bearer ${token}` },
      });
      const res = createMockResponse();
      const next = createMockNext();

      mockUserFindUnique.mockResolvedValue(mockUser);

      await optionalAuth(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith();
      expect(req.user).toBeDefined();
      expect(req.user?.id).toBe(mockUser.id);
    });

    it('should continue without user if token is invalid (no error thrown)', async () => {
      const req = createMockRequest({
        headers: { authorization: 'Bearer invalid.token.here' },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await optionalAuth(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith();
      expect(req.user).toBeUndefined();
    });

    it('should continue without user if user not found in database', async () => {
      const token = generateToken({ id: 'non-existent', email: 'test@test.com', tokenVersion: 0 });
      const req = createMockRequest({
        headers: { authorization: `Bearer ${token}` },
      });
      const res = createMockResponse();
      const next = createMockNext();

      mockUserFindUnique.mockResolvedValue(null);

      await optionalAuth(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith();
      expect(req.user).toBeUndefined();
    });

    it('should continue without user if account is deactivated', async () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email, tokenVersion: 0 });
      const req = createMockRequest({
        headers: { authorization: `Bearer ${token}` },
      });
      const res = createMockResponse();
      const next = createMockNext();

      mockUserFindUnique.mockResolvedValue({ ...mockUser, isActive: false });

      await optionalAuth(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith();
      expect(req.user).toBeUndefined();
    });
  });

  // ===========================================================================
  // Role-Based Access Tests (authorize)
  // ===========================================================================

  describe('Role-Based Access (authorize)', () => {
    it('should allow access for admin role when admin is required', () => {
      const req = createMockRequest() as AuthenticatedRequest;
      req.user = { ...mockUser, role: 'admin' };
      const res = createMockResponse();
      const next = createMockNext();

      const adminOnly = authorize(['admin']);
      adminOnly(req, res as Response, next);

      expect(next).toHaveBeenCalledWith();
    });

    it('should reject non-admin when admin role is required', () => {
      const req = createMockRequest() as AuthenticatedRequest;
      req.user = { ...mockUser, role: 'user' };
      const res = createMockResponse();
      const next = createMockNext();

      const adminOnly = authorize(['admin']);
      adminOnly(req, res as Response, next);

      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          message: 'Insufficient permissions',
          statusCode: 403,
        })
      );
    });

    it('should allow access when user has one of multiple allowed roles', () => {
      const req = createMockRequest() as AuthenticatedRequest;
      req.user = { ...mockUser, role: 'moderator' };
      const res = createMockResponse();
      const next = createMockNext();

      const multiRoleAccess = authorize(['admin', 'moderator']);
      multiRoleAccess(req, res as Response, next);

      expect(next).toHaveBeenCalledWith();
    });

    it('should return 401 when user is not authenticated', () => {
      const req = createMockRequest() as AuthenticatedRequest;
      // req.user is undefined
      const res = createMockResponse();
      const next = createMockNext();

      const adminOnly = authorize(['admin']);
      adminOnly(req, res as Response, next);

      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          message: 'Authentication required',
          statusCode: 401,
        })
      );
    });

    it('should handle case-sensitive role comparison', () => {
      const req = createMockRequest() as AuthenticatedRequest;
      req.user = { ...mockUser, role: 'Admin' }; // Capitalized
      const res = createMockResponse();
      const next = createMockNext();

      const adminOnly = authorize(['admin']); // lowercase
      adminOnly(req, res as Response, next);

      // Should fail because 'Admin' !== 'admin'
      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          statusCode: 403,
        })
      );
    });
  });

  // ===========================================================================
  // Token Generation Tests
  // ===========================================================================

  describe('Token Generation (generateToken)', () => {
    it('should generate a valid access token', () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email });

      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
      expect(token.split('.')).toHaveLength(3); // JWT format
    });

    it('should include userId and email in token payload', () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email });
      const decoded = jwt.decode(token) as any;

      expect(decoded.userId).toBe(mockUser.id);
      expect(decoded.email).toBe(mockUser.email);
    });

    it('should include tokenVersion (tv) when provided', () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email, tokenVersion: 5 });
      const decoded = jwt.decode(token) as any;

      expect(decoded.tv).toBe(5);
    });

    it('should not include tv claim when tokenVersion is not provided', () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email });
      const decoded = jwt.decode(token) as any;

      expect(decoded.tv).toBeUndefined();
    });

    it('should include proper issuer and audience', () => {
      const token = generateToken({ id: mockUser.id, email: mockUser.email });
      const decoded = jwt.decode(token) as any;

      expect(decoded.iss).toBe('ringrift');
      expect(decoded.aud).toBe('ringrift-users');
    });
  });

  // ===========================================================================
  // Refresh Token Tests
  // ===========================================================================

  describe('Refresh Token Handling', () => {
    it('should generate a valid refresh token', () => {
      const token = generateRefreshToken({ id: mockUser.id, email: mockUser.email });

      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
      expect(token.split('.')).toHaveLength(3);
    });

    it('should include type=refresh in refresh token payload', () => {
      const token = generateRefreshToken({ id: mockUser.id, email: mockUser.email });
      const decoded = jwt.decode(token) as any;

      expect(decoded.type).toBe('refresh');
    });

    it('should verify valid refresh token', () => {
      const token = generateRefreshToken({
        id: mockUser.id,
        email: mockUser.email,
        tokenVersion: 2,
      });

      const result = verifyRefreshToken(token);

      expect(result.userId).toBe(mockUser.id);
      expect(result.email).toBe(mockUser.email);
      expect(result.tokenVersion).toBe(2);
    });

    it('should throw REFRESH_TOKEN_EXPIRED for expired refresh token', () => {
      const { config } = require('../../../src/server/config');
      const secret = config.auth.jwtRefreshSecret;
      const expiredToken = jwt.sign(
        { userId: mockUser.id, email: mockUser.email, type: 'refresh' },
        secret,
        { expiresIn: '-1s' }
      );

      expect(() => verifyRefreshToken(expiredToken)).toThrow(
        expect.objectContaining({
          message: 'Refresh token has expired',
          statusCode: 401,
        })
      );
    });

    it('should throw INVALID_REFRESH_TOKEN for access token used as refresh', () => {
      // Access token doesn't have type: 'refresh'
      const accessToken = generateToken({ id: mockUser.id, email: mockUser.email });

      expect(() => verifyRefreshToken(accessToken)).toThrow(
        expect.objectContaining({
          statusCode: 401,
        })
      );
    });

    it('should throw for refresh token with wrong signature', () => {
      const wrongSecretToken = jwt.sign(
        { userId: mockUser.id, email: mockUser.email, type: 'refresh' },
        'completely-wrong-secret-that-is-long-enough',
        { expiresIn: '7d' }
      );

      expect(() => verifyRefreshToken(wrongSecretToken)).toThrow(
        expect.objectContaining({
          message: 'Invalid refresh token',
          statusCode: 401,
        })
      );
    });

    it('should throw for malformed refresh token', () => {
      expect(() => verifyRefreshToken('not.valid.jwt')).toThrow(
        expect.objectContaining({
          statusCode: 401,
        })
      );
    });
  });

  // ===========================================================================
  // Token Version / Revocation Tests
  // ===========================================================================

  describe('Token Version and Revocation', () => {
    it('should accept token when version matches database', async () => {
      mockUserFindUnique.mockResolvedValue({ ...mockUser, tokenVersion: 5 });

      const result = await validateUser(mockUser.id, 5);

      expect(result.id).toBe(mockUser.id);
    });

    it('should reject token when version is lower than database', async () => {
      mockUserFindUnique.mockResolvedValue({ ...mockUser, tokenVersion: 5 });

      await expect(validateUser(mockUser.id, 4)).rejects.toThrow(
        expect.objectContaining({
          message: 'Token has been revoked',
          statusCode: 401,
        })
      );
    });

    it('should reject token when version is higher than database (tampering)', async () => {
      mockUserFindUnique.mockResolvedValue({ ...mockUser, tokenVersion: 5 });

      await expect(validateUser(mockUser.id, 6)).rejects.toThrow(
        expect.objectContaining({
          message: 'Token has been revoked',
          statusCode: 401,
        })
      );
    });

    it('should treat undefined tokenVersion as 0 for backwards compatibility', async () => {
      mockUserFindUnique.mockResolvedValue({ ...mockUser, tokenVersion: 0 });

      // Token without tv claim (undefined tokenVersion)
      const result = await validateUser(mockUser.id, undefined);

      expect(result.id).toBe(mockUser.id);
    });

    it('should treat missing database tokenVersion as 0', async () => {
      // User without tokenVersion field (legacy data)
      mockUserFindUnique.mockResolvedValue({
        ...mockUser,
        tokenVersion: undefined,
      });

      // Token with version 0 should match
      const result = await validateUser(mockUser.id, 0);

      expect(result.id).toBe(mockUser.id);
    });
  });

  // ===========================================================================
  // Integration-like Tests
  // ===========================================================================

  describe('Full Authentication Flow', () => {
    it('should complete full auth flow: generate -> verify -> authenticate', async () => {
      // Step 1: Generate token
      const token = generateToken({
        id: mockUser.id,
        email: mockUser.email,
        tokenVersion: 0,
      });

      // Step 2: Verify token directly
      const verified = verifyToken(token);
      expect(verified.userId).toBe(mockUser.id);

      // Step 3: Use authenticate middleware
      const req = createMockRequest({
        headers: { authorization: `Bearer ${token}` },
      });
      const res = createMockResponse();
      const next = createMockNext();

      mockUserFindUnique.mockResolvedValue(mockUser);

      await authenticate(req as AuthenticatedRequest, res as Response, next);

      expect(next).toHaveBeenCalledWith();
      expect(req.user?.id).toBe(mockUser.id);
    });

    it('should handle refresh token rotation flow', () => {
      // Step 1: Generate refresh token
      const refreshToken = generateRefreshToken({
        id: mockUser.id,
        email: mockUser.email,
        tokenVersion: 0,
      });

      // Step 2: Verify refresh token
      const verified = verifyRefreshToken(refreshToken);
      expect(verified.userId).toBe(mockUser.id);
      expect(verified.tokenVersion).toBe(0);

      // Step 3: Generate new access token (simulating rotation)
      const newAccessToken = generateToken({
        id: verified.userId,
        email: verified.email,
        tokenVersion: 1, // Incremented version
      });

      const decoded = jwt.decode(newAccessToken) as any;
      expect(decoded.tv).toBe(1);
    });
  });
});
