import { Request, Response, NextFunction } from 'express';
import { ZodError } from 'zod';
import { logger, withRequestContext } from '../utils/logger';
import {
  ValidationError,
  AuthenticationError,
  AuthorizationError,
} from '../../shared/validation/schemas';
import { config } from '../config';

export interface AppError extends Error {
  statusCode?: number;
  code?: string;
  isOperational?: boolean;
}

export const errorHandler = (error: AppError, req: Request, res: Response, _next: NextFunction) => {
  let statusCode = error.statusCode || 500;
  let message = error.message || 'Internal Server Error';
  let code = error.code || 'INTERNAL_ERROR';

  // Handle specific error types
  if (error instanceof ZodError) {
    statusCode = 400;
    code = 'VALIDATION_ERROR';
    // Use the first issue message when available, fall back to generic message
    if (error.issues?.length > 0) {
      message = error.issues[0]?.message || message;
    }
  } else if (error instanceof ValidationError) {
    statusCode = 400;
    code = 'VALIDATION_ERROR';
  } else if (error instanceof AuthenticationError) {
    statusCode = 401;
    code = 'AUTHENTICATION_ERROR';
  } else if (error instanceof AuthorizationError) {
    statusCode = 403;
    code = 'AUTHORIZATION_ERROR';
  } else if (error.name === 'JsonWebTokenError') {
    statusCode = 401;
    code = 'INVALID_TOKEN';
    message = 'Invalid authentication token';
  } else if (error.name === 'TokenExpiredError') {
    statusCode = 401;
    code = 'TOKEN_EXPIRED';
    message = 'Authentication token has expired';
  } else if (error.name === 'CastError') {
    statusCode = 400;
    code = 'INVALID_ID';
    message = 'Invalid ID format';
  } else if (error.name === 'MongoError' && (error as any).code === 11000) {
    statusCode = 409;
    code = 'DUPLICATE_ENTRY';
    message = 'Duplicate entry found';
  }

  // Log error with correlation id when available
  if (statusCode >= 500) {
    logger.error(
      'Server Error:',
      withRequestContext(req as any, {
        error: error.message,
        stack: error.stack,
        url: req.url,
        method: req.method,
        ip: req.ip,
        userAgent: req.get('User-Agent'),
      })
    );
  } else {
    logger.warn(
      'Client Error:',
      withRequestContext(req as any, {
        error: error.message,
        url: req.url,
        method: req.method,
        ip: req.ip,
        statusCode,
      })
    );
  }

  // Send error response
  const errorResponse = {
    success: false,
    error: {
      message,
      code,
      timestamp: new Date().toISOString(),
      ...(config.isDevelopment && {
        stack: error.stack,
        details: error,
      }),
    },
  };

  res.status(statusCode).json(errorResponse);
};

// Async error wrapper
type AsyncRequestHandler = (req: Request, res: Response, next: NextFunction) => Promise<any> | any;

export const asyncHandler = (fn: AsyncRequestHandler) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

// Create custom error
export const createError = (message: string, statusCode: number = 500, code?: string): AppError => {
  const error: AppError = new Error(message);
  error.statusCode = statusCode;
  if (code) {
    error.code = code;
  }
  error.isOperational = true;
  return error;
};

// Not found handler
export const notFoundHandler = (req: Request, _res: Response, next: NextFunction) => {
  const error = createError(`Route ${req.originalUrl} not found`, 404, 'NOT_FOUND');
  next(error);
};
