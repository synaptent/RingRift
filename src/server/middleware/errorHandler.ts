import { Request, Response, NextFunction } from 'express';
import { ZodError } from 'zod';
import * as Sentry from '@sentry/node';
import { logger, withRequestContext } from '../utils/logger';
import {
  ValidationError,
  AuthenticationError,
  AuthorizationError,
} from '../../shared/validation/schemas';
import { config } from '../config';
import {
  ApiError,
  ErrorCodes,
  normalizeErrorCode,
  ErrorCodeToStatus,
  ErrorCodeMessages,
} from '../errors';
import type { ApiErrorResponse, ValidationErrorDetail, ErrorCode } from '../errors';

/**
 * Legacy AppError interface for backward compatibility with existing code.
 * New code should use ApiError class instead.
 */
export interface AppError extends Error {
  statusCode?: number;
  code?: string;
  isOperational?: boolean;
}

/**
 * Extended error type that may have additional properties from various
 * error sources (MongoDB, JWT, etc.).
 */
interface ExtendedError extends Error {
  code?: string | number;
  statusCode?: number;
  isOperational?: boolean;
}

/**
 * Express Request with requestId attached by middleware.
 */
interface RequestWithId extends Request {
  requestId?: string;
}

/**
 * Map ZodError to standardized validation error details.
 */
function mapZodErrorToDetails(error: ZodError): ValidationErrorDetail[] {
  return error.issues.map((issue) => ({
    field: issue.path.join('.'),
    message: issue.message,
    code: issue.code,
  }));
}

/**
 * Determine error code from legacy error types.
 */
function getErrorCodeFromLegacy(error: Error): ErrorCode {
  const extError = error as ExtendedError;

  // Check for explicit code property
  if (extError.code !== undefined) {
    return normalizeErrorCode(String(extError.code));
  }

  // Handle JWT errors
  if (error.name === 'JsonWebTokenError') {
    return ErrorCodes.AUTH_TOKEN_INVALID;
  }
  if (error.name === 'TokenExpiredError') {
    return ErrorCodes.AUTH_TOKEN_EXPIRED;
  }

  // Handle MongoDB/Prisma errors
  if (error.name === 'CastError') {
    return ErrorCodes.VALIDATION_INVALID_ID;
  }
  if (error.name === 'MongoError' && extError.code === 11000) {
    return ErrorCodes.RESOURCE_ALREADY_EXISTS;
  }

  // Handle custom error types from validation schemas
  if (error instanceof ValidationError) {
    return ErrorCodes.VALIDATION_FAILED;
  }
  if (error instanceof AuthenticationError) {
    return ErrorCodes.AUTH_TOKEN_INVALID;
  }
  if (error instanceof AuthorizationError) {
    return ErrorCodes.AUTH_FORBIDDEN;
  }

  return ErrorCodes.SERVER_INTERNAL_ERROR;
}

/**
 * Centralized error handler middleware.
 *
 * Converts all errors to standardized API error response format:
 * ```json
 * {
 *   "success": false,
 *   "error": {
 *     "code": "AUTH_INVALID_CREDENTIALS",
 *     "message": "Invalid credentials",
 *     "details": [...],  // For validation errors
 *     "requestId": "abc-123",
 *     "timestamp": "2024-01-01T00:00:00.000Z"
 *   }
 * }
 * ```
 *
 * In development mode, additional debug information (stack trace) may be included.
 * In production, stack traces are never exposed to clients.
 */
export const errorHandler = (
  error: Error | ApiError | AppError,
  req: RequestWithId,
  res: Response,
  _next: NextFunction
) => {
  // Get correlation ID from request (attached by middleware)
  const requestId = req.requestId;

  // Convert to ApiError if not already
  let apiError: ApiError;
  let validationDetails: ValidationErrorDetail[] | undefined;

  if (error instanceof ApiError) {
    // Already an ApiError, use as-is
    apiError = error;
  } else if (error instanceof ZodError) {
    // Zod validation error
    validationDetails = mapZodErrorToDetails(error);
    apiError = new ApiError({
      code: ErrorCodes.VALIDATION_FAILED,
      message: validationDetails[0]?.message || 'Validation failed',
      details: validationDetails,
    });
  } else {
    // Convert legacy error to ApiError
    const errorCode = getErrorCodeFromLegacy(error);
    const extError = error as ExtendedError;

    apiError = new ApiError({
      code: errorCode,
      message: error.message || ErrorCodeMessages[errorCode],
      statusCode: extError.statusCode || ErrorCodeToStatus[errorCode],
      cause: error,
      isOperational: extError.isOperational ?? true,
    });
  }

  // Log error with correlation ID and context
  const logContext = withRequestContext(req, {
    error: apiError.message,
    code: apiError.code,
    statusCode: apiError.statusCode,
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    ...(apiError.cause && { originalError: apiError.cause.message }),
    ...(apiError.cause?.stack && { stack: apiError.cause.stack }),
  });

  // Log at appropriate level based on status code
  if (apiError.statusCode >= 500) {
    logger.error('Server Error:', logContext);
    Sentry.captureException(apiError.cause ?? apiError, {
      tags: { code: apiError.code, url: req.url },
      extra: { requestId, method: req.method },
    });
  } else if (apiError.statusCode >= 400) {
    logger.warn('Client Error:', logContext);
  } else {
    logger.info('Error:', logContext);
  }

  // Build response using ApiError's toResponse method
  const response = apiError.toResponse(requestId);

  // In development, include debug details for 5xx errors
  if (config.isDevelopment && apiError.statusCode >= 500) {
    const debugResponse = response as ApiErrorResponse & {
      error: ApiErrorResponse['error'] & {
        stack?: string;
        debug?: Record<string, unknown>;
      };
    };
    if (apiError.cause?.stack) {
      debugResponse.error.stack = apiError.cause.stack;
    } else if (apiError.stack) {
      debugResponse.error.stack = apiError.stack;
    }
    res.status(apiError.statusCode).json(debugResponse);
    return;
  }

  // Send standardized error response
  res.status(apiError.statusCode).json(response);
};

// ============================================================================
// Backward Compatibility Layer
// ============================================================================

/**
 * Async error wrapper for route handlers.
 * Catches errors from async handlers and passes them to the error handler.
 *
 * @example
 * ```ts
 * router.get('/users', asyncHandler(async (req, res) => {
 *   const users = await getUsers();
 *   res.json({ success: true, data: users });
 * }));
 * ```
 */
type AsyncRequestHandler = (
  req: Request,
  res: Response,
  next: NextFunction
) => Promise<void> | void;

export const asyncHandler = (fn: AsyncRequestHandler) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

/**
 * Create a custom error.
 *
 * @deprecated Use `new ApiError({ code, message })` instead.
 * This function is kept for backward compatibility with existing code.
 */
export const createError = (
  message: string,
  statusCode: number = 500,
  code?: string
): AppError | ApiError => {
  // If a standardized code is provided, use ApiError
  if (code) {
    const normalizedCode = normalizeErrorCode(code);
    return new ApiError({
      code: normalizedCode,
      message,
      statusCode,
    });
  }

  // Fallback for legacy usage without code
  const error: AppError = new Error(message);
  error.statusCode = statusCode;
  error.isOperational = true;
  return error;
};

/**
 * Not found handler middleware.
 * Handles requests to undefined routes with standardized error response.
 */
export const notFoundHandler = (req: Request, _res: Response, next: NextFunction) => {
  const error = new ApiError({
    code: ErrorCodes.RESOURCE_ROUTE_NOT_FOUND,
    message: `Route ${req.originalUrl} not found`,
  });
  next(error);
};
