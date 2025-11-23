import { Request, Response, NextFunction } from 'express';
import { randomUUID } from 'crypto';

/**
 * Lightweight per-request context middleware.
 *
 * - Derives requestId from X-Request-Id header when present.
 * - Falls back to a server-generated UUID when absent.
 * - Attaches requestId to req.requestId and res.locals.requestId.
 * - Echoes X-Request-Id response header for downstream clients/log aggregation.
 */
export interface RequestWithId extends Request {
  requestId?: string;
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

export const requestContext = (req: RequestWithId, res: Response, next: NextFunction): void => {
  const headerId = (req.header('x-request-id') || req.header('X-Request-Id') || '').trim();

  const requestId = headerId.length > 0 ? headerId : randomUUID();

  req.requestId = requestId;
  (res.locals as any).requestId = requestId;

  // Expose the correlation id back to the client for easier debugging.
  res.setHeader('X-Request-Id', requestId);

  next();
};
