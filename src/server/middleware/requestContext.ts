import { Request, Response, NextFunction } from 'express';
import { randomUUID } from 'crypto';
import {
  RequestContext as LoggerRequestContext,
  runWithContext,
  getRequestContext,
} from '../utils/logger';

/**
 * Lightweight per-request context middleware.
 *
 * - Derives requestId from X-Request-Id header when present.
 * - Falls back to a server-generated UUID when absent.
 * - Attaches requestId to req.requestId and res.locals.requestId.
 * - Echoes X-Request-Id response header for downstream clients/log aggregation.
 * - Establishes AsyncLocalStorage context for automatic log correlation.
 */
export interface RequestWithId extends Request {
  requestId?: string;
  user?: {
    id: string;
    email?: string;
    username?: string;
  };
}

/**
 * Express.Request augmentation so that req.requestId is available
 * throughout the codebase without additional casting.
 */
declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    // eslint-disable-next-line @typescript-eslint/no-empty-interface
    interface Request {
      requestId?: string;
    }
  }
}

/**
 * Request context middleware that:
 * 1. Generates or extracts a request ID for correlation
 * 2. Attaches the ID to the request object for explicit access
 * 3. Establishes AsyncLocalStorage context for automatic log propagation
 *
 * This enables all logs within a request to automatically include the
 * request ID, even deeply nested async operations, without explicitly
 * passing the request object.
 */
export const requestContext = (req: RequestWithId, res: Response, next: NextFunction): void => {
  const headerId = (req.header('x-request-id') || req.header('X-Request-Id') || '').trim();

  const requestId = headerId.length > 0 ? headerId : randomUUID();

  req.requestId = requestId;
  (res.locals as Record<string, unknown>).requestId = requestId;

  // Expose the correlation id back to the client for easier debugging.
  res.setHeader('X-Request-Id', requestId);

  // Create the context for AsyncLocalStorage
  const context: LoggerRequestContext = {
    requestId,
    method: req.method,
    path: req.path,
    startTime: Date.now(),
    // userId will be populated by auth middleware after authentication
  };

  // Run the rest of the request handling within the AsyncLocalStorage context
  // This allows any code (including deeply nested async functions) to access
  // the request context via getRequestContext()
  runWithContext(context, () => {
    next();
  });
};

/**
 * Update the AsyncLocalStorage context with user information after authentication.
 * Call this from auth middleware after successfully authenticating a user.
 */
export const updateContextWithUser = (userId: string): void => {
  // Note: AsyncLocalStorage store is shared by reference, so we can modify it directly
  const context = getRequestContext();
  if (context) {
    context.userId = userId;
  }
};
