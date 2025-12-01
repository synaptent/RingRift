/**
 * Error Handler Middleware Unit Tests
 *
 * Tests for the centralized error handling middleware including:
 * - API error conversion
 * - Zod validation error handling
 * - Legacy error handling
 * - Response formatting
 * - Async handler wrapping
 */

import { Request, Response, NextFunction } from 'express';
import { ZodError, z } from 'zod';
import {
  errorHandler,
  createError,
  asyncHandler,
  notFoundHandler,
} from '../../../src/server/middleware/errorHandler';
import { ApiError, ErrorCodes } from '../../../src/server/errors';
import {
  ValidationError,
  AuthenticationError,
  AuthorizationError,
} from '../../../src/shared/validation/schemas';

// Mock dependencies
jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
  withRequestContext: jest.fn((req, ctx) => ctx),
}));

jest.mock('../../../src/server/config', () => ({
  config: {
    isDevelopment: false,
    isProduction: true,
  },
}));

describe('Error Handler Middleware', () => {
  let mockReq: Partial<Request & { requestId?: string }>;
  let mockRes: Partial<Response>;
  let mockNext: NextFunction;
  let responseBody: unknown;
  let responseStatus: number;

  beforeEach(() => {
    responseBody = null;
    responseStatus = 0;

    mockReq = {
      url: '/api/test',
      method: 'GET',
      ip: '127.0.0.1',
      originalUrl: '/api/test',
      requestId: 'test-request-id',
      get: jest.fn().mockReturnValue('Test User Agent'),
    };

    mockRes = {
      status: jest.fn((code: number) => {
        responseStatus = code;
        return mockRes as Response;
      }),
      json: jest.fn((body: unknown) => {
        responseBody = body;
        return mockRes as Response;
      }),
    };

    mockNext = jest.fn();
  });

  describe('ApiError handling', () => {
    it('should handle ApiError correctly', () => {
      const error = new ApiError({
        code: ErrorCodes.AUTH_INVALID_CREDENTIALS,
        message: 'Invalid credentials',
      });

      errorHandler(error, mockReq as Request, mockRes as Response, mockNext);

      expect(responseStatus).toBe(401);
      expect(responseBody).toMatchObject({
        success: false,
        error: expect.objectContaining({
          code: 'AUTH_INVALID_CREDENTIALS',
          message: 'Invalid credentials',
        }),
      });
    });

    it('should include requestId in response', () => {
      const error = new ApiError({
        code: ErrorCodes.SERVER_INTERNAL_ERROR,
        message: 'Something went wrong',
      });

      errorHandler(error, mockReq as Request, mockRes as Response, mockNext);

      expect(responseBody).toMatchObject({
        error: expect.objectContaining({
          requestId: 'test-request-id',
        }),
      });
    });

    it('should use correct status code from ApiError', () => {
      const error = new ApiError({
        code: ErrorCodes.RESOURCE_NOT_FOUND,
        message: 'Not found',
      });

      errorHandler(error, mockReq as Request, mockRes as Response, mockNext);

      expect(responseStatus).toBe(404);
    });
  });

  describe('ZodError handling', () => {
    it('should handle ZodError with validation details', () => {
      const schema = z.object({
        email: z.string().email(),
        age: z.number().min(18),
      });

      let zodError: ZodError | undefined;
      try {
        schema.parse({ email: 'invalid', age: 10 });
      } catch (e) {
        zodError = e as ZodError;
      }

      errorHandler(zodError!, mockReq as Request, mockRes as Response, mockNext);

      expect(responseStatus).toBe(400);
      expect(responseBody).toMatchObject({
        success: false,
        error: expect.objectContaining({
          code: 'VALIDATION_FAILED',
        }),
      });
    });

    it('should map Zod issues to validation details', () => {
      const schema = z.object({
        username: z.string().min(3),
      });

      let zodError: ZodError | undefined;
      try {
        schema.parse({ username: 'ab' });
      } catch (e) {
        zodError = e as ZodError;
      }

      errorHandler(zodError!, mockReq as Request, mockRes as Response, mockNext);

      expect(responseBody).toMatchObject({
        error: expect.objectContaining({
          details: expect.arrayContaining([
            expect.objectContaining({
              field: 'username',
            }),
          ]),
        }),
      });
    });
  });

  describe('Legacy error handling', () => {
    it('should handle standard Error', () => {
      const error = new Error('Something went wrong');

      errorHandler(error, mockReq as Request, mockRes as Response, mockNext);

      expect(responseStatus).toBe(500);
      expect(responseBody).toMatchObject({
        success: false,
        error: expect.objectContaining({
          message: 'Something went wrong',
        }),
      });
    });

    it('should handle ValidationError', () => {
      const error = new ValidationError('Validation failed');

      errorHandler(error, mockReq as Request, mockRes as Response, mockNext);

      expect(responseBody).toMatchObject({
        error: expect.objectContaining({
          code: 'VALIDATION_FAILED',
        }),
      });
    });

    it('should handle AuthenticationError', () => {
      const error = new AuthenticationError('Invalid token');

      errorHandler(error, mockReq as Request, mockRes as Response, mockNext);

      expect(responseStatus).toBe(401);
      expect(responseBody).toMatchObject({
        error: expect.objectContaining({
          code: 'AUTH_TOKEN_INVALID',
        }),
      });
    });

    it('should handle AuthorizationError', () => {
      const error = new AuthorizationError('Access denied');

      errorHandler(error, mockReq as Request, mockRes as Response, mockNext);

      expect(responseStatus).toBe(403);
      expect(responseBody).toMatchObject({
        error: expect.objectContaining({
          code: 'AUTH_FORBIDDEN',
        }),
      });
    });

    it('should handle JWT errors', () => {
      const error = new Error('jwt malformed');
      error.name = 'JsonWebTokenError';

      errorHandler(error, mockReq as Request, mockRes as Response, mockNext);

      expect(responseBody).toMatchObject({
        error: expect.objectContaining({
          code: 'AUTH_TOKEN_INVALID',
        }),
      });
    });

    it('should handle expired token errors', () => {
      const error = new Error('jwt expired');
      error.name = 'TokenExpiredError';

      errorHandler(error, mockReq as Request, mockRes as Response, mockNext);

      expect(responseBody).toMatchObject({
        error: expect.objectContaining({
          code: 'AUTH_TOKEN_EXPIRED',
        }),
      });
    });
  });

  describe('Response formatting', () => {
    it('should include timestamp in response', () => {
      const error = new ApiError({
        code: ErrorCodes.SERVER_INTERNAL_ERROR,
        message: 'Error',
      });

      errorHandler(error, mockReq as Request, mockRes as Response, mockNext);

      const body = responseBody as { error: { timestamp: string } };
      expect(body.error.timestamp).toBeDefined();
      expect(new Date(body.error.timestamp)).toBeInstanceOf(Date);
    });

    it('should always set success to false', () => {
      const error = new Error('Any error');

      errorHandler(error, mockReq as Request, mockRes as Response, mockNext);

      expect(responseBody).toMatchObject({ success: false });
    });
  });
});

describe('createError', () => {
  it('should create AppError with message and status code', () => {
    const error = createError('Not found', 404);

    expect(error.message).toBe('Not found');
    expect((error as any).statusCode).toBe(404);
  });

  it('should create ApiError when code is provided', () => {
    const error = createError('Unauthorized', 401, 'AUTH_TOKEN_INVALID');

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe('AUTH_TOKEN_INVALID');
  });

  it('should default to 500 status code', () => {
    const error = createError('Server error');

    expect((error as any).statusCode).toBe(500);
  });

  it('should mark error as operational', () => {
    const error = createError('Operational error', 400);

    expect((error as any).isOperational).toBe(true);
  });
});

describe('asyncHandler', () => {
  let mockReq: Partial<Request>;
  let mockRes: Partial<Response>;
  let mockNext: NextFunction;

  beforeEach(() => {
    mockReq = {};
    mockRes = {
      json: jest.fn(),
    };
    mockNext = jest.fn();
  });

  it('should call the wrapped function', async () => {
    const handler = jest.fn().mockResolvedValue(undefined);
    const wrapped = asyncHandler(handler);

    await wrapped(mockReq as Request, mockRes as Response, mockNext);

    expect(handler).toHaveBeenCalledWith(mockReq, mockRes, mockNext);
  });

  it('should pass errors to next', async () => {
    const error = new Error('Async error');
    const handler = jest.fn().mockRejectedValue(error);
    const wrapped = asyncHandler(handler);

    await wrapped(mockReq as Request, mockRes as Response, mockNext);

    expect(mockNext).toHaveBeenCalledWith(error);
  });

  it('should handle sync functions that return rejected promise', async () => {
    const error = new Error('Sync-like error');
    const handler = jest.fn().mockReturnValue(Promise.reject(error));
    const wrapped = asyncHandler(handler);

    await wrapped(mockReq as Request, mockRes as Response, mockNext);

    expect(mockNext).toHaveBeenCalledWith(error);
  });

  it('should not call next on success', async () => {
    const handler = jest.fn().mockResolvedValue(undefined);
    const wrapped = asyncHandler(handler);

    await wrapped(mockReq as Request, mockRes as Response, mockNext);

    expect(mockNext).not.toHaveBeenCalled();
  });
});

describe('notFoundHandler', () => {
  let mockReq: Partial<Request>;
  let mockRes: Partial<Response>;
  let mockNext: NextFunction;

  beforeEach(() => {
    mockReq = {
      originalUrl: '/api/unknown',
    };
    mockRes = {};
    mockNext = jest.fn();
  });

  it('should create ApiError with RESOURCE_ROUTE_NOT_FOUND code', () => {
    notFoundHandler(mockReq as Request, mockRes as Response, mockNext);

    expect(mockNext).toHaveBeenCalledWith(
      expect.objectContaining({
        code: 'RESOURCE_ROUTE_NOT_FOUND',
      })
    );
  });

  it('should include original URL in error message', () => {
    notFoundHandler(mockReq as Request, mockRes as Response, mockNext);

    const error = (mockNext as jest.Mock).mock.calls[0][0] as ApiError;
    expect(error.message).toContain('/api/unknown');
  });

  it('should pass error to next middleware', () => {
    notFoundHandler(mockReq as Request, mockRes as Response, mockNext);

    expect(mockNext).toHaveBeenCalledWith(expect.any(ApiError));
  });
});
