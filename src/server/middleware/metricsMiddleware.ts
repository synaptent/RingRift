/**
 * Metrics Middleware - Automatic HTTP request metrics collection.
 *
 * This middleware automatically tracks:
 * - Request duration
 * - Request count by method, path, and status
 * - Request and response body sizes
 *
 * It normalizes URL paths to prevent high cardinality in metrics labels.
 * Health check and metrics endpoints are skipped to avoid noise.
 */

import { Request, Response, NextFunction } from 'express';
import { getMetricsService } from '../services/MetricsService';

/**
 * Paths to skip from metrics collection.
 * These are typically health checks and the metrics endpoint itself.
 */
const SKIP_PATHS = new Set(['/health', '/healthz', '/ready', '/readyz', '/metrics']);

/**
 * Check if a path should be skipped from metrics collection.
 */
function shouldSkipPath(path: string): boolean {
  // Direct match
  if (SKIP_PATHS.has(path)) {
    return true;
  }

  // Also skip socket.io paths
  if (path.startsWith('/socket.io')) {
    return true;
  }

  return false;
}

/**
 * Get content length from headers or body.
 */
function getContentLength(
  headers: Record<string, string | string[] | undefined>,
  body?: unknown
): number | undefined {
  // Try to get from Content-Length header
  const headerLength = headers['content-length'];
  if (headerLength) {
    const value = Array.isArray(headerLength) ? headerLength[0] : headerLength;
    if (value) {
      const parsed = parseInt(value, 10);
      if (!isNaN(parsed)) {
        return parsed;
      }
    }
  }

  // Try to estimate from body
  if (body !== undefined) {
    if (typeof body === 'string') {
      return Buffer.byteLength(body, 'utf8');
    }
    if (Buffer.isBuffer(body)) {
      return body.length;
    }
    if (typeof body === 'object') {
      try {
        return Buffer.byteLength(JSON.stringify(body), 'utf8');
      } catch {
        return undefined;
      }
    }
  }

  return undefined;
}

/**
 * Express middleware for collecting HTTP request metrics.
 *
 * Usage:
 * ```typescript
 * import { metricsMiddleware } from './middleware/metricsMiddleware';
 * app.use(metricsMiddleware);
 * ```
 */
export function metricsMiddleware(req: Request, res: Response, next: NextFunction): void {
  // Skip paths that shouldn't be tracked
  if (shouldSkipPath(req.path)) {
    return next();
  }

  const startTime = process.hrtime.bigint();
  const metrics = getMetricsService();

  // Get request size
  const requestSize = getContentLength(req.headers, req.body);

  // Track response size by intercepting write/end
  let responseSize = 0;
  const originalWrite = res.write.bind(res);
  const originalEnd = res.end.bind(res);
  const writeWithUnknownArgs = originalWrite as (...writeArgs: unknown[]) => boolean;
  const endWithUnknownArgs = originalEnd as (...endArgs: unknown[]) => Response;

  // Override write to track response size

  res.write = function (chunk: unknown, ...args: unknown[]): boolean {
    if (chunk) {
      if (Buffer.isBuffer(chunk)) {
        responseSize += chunk.length;
      } else if (typeof chunk === 'string') {
        const encoding = typeof args[0] === 'string' ? args[0] : 'utf8';
        responseSize += Buffer.byteLength(chunk, encoding as BufferEncoding);
      }
    }
    return writeWithUnknownArgs(chunk, ...args);
  } as typeof res.write;

  // Override end to track final response size and record metrics

  res.end = function (chunk?: unknown, ...args: unknown[]): Response {
    // Track final chunk size if present
    if (chunk && typeof chunk !== 'function') {
      if (Buffer.isBuffer(chunk)) {
        responseSize += chunk.length;
      } else if (typeof chunk === 'string') {
        const encoding = typeof args[0] === 'string' ? args[0] : 'utf8';
        responseSize += Buffer.byteLength(chunk, encoding as BufferEncoding);
      }
    }

    // Calculate duration
    const endTime = process.hrtime.bigint();
    const durationNs = Number(endTime - startTime);
    const durationSeconds = durationNs / 1e9;

    // Record metrics
    metrics.recordHttpRequest(
      req.method,
      req.path,
      res.statusCode,
      durationSeconds,
      requestSize,
      responseSize > 0 ? responseSize : undefined
    );

    // Call original end
    return endWithUnknownArgs(chunk, ...args);
  } as typeof res.end;

  next();
}

/**
 * Express middleware that wraps metricsMiddleware with error handling.
 * Ensures metrics collection failures don't break request processing.
 */
export function safeMetricsMiddleware(req: Request, res: Response, next: NextFunction): void {
  try {
    metricsMiddleware(req, res, next);
  } catch (error) {
    // Log error but don't fail the request
    console.error('Metrics middleware error:', error);
    next();
  }
}

export default metricsMiddleware;
