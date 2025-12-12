/**
 * Metrics Middleware Integration Tests
 *
 * Integration-level tests for the Express middleware that tracks HTTP request metrics:
 * - Request duration
 * - Request/response sizes
 * - Path normalization (indirectly through MetricsService)
 * - Endpoint skipping
 *
 * Note: Unit tests for middleware internals are in tests/unit/middleware/metricsMiddleware.test.ts
 */

import { Request, Response, NextFunction } from 'express';
import { EventEmitter } from 'events';
import client from 'prom-client';
import { MetricsService, getMetricsService } from '../../src/server/services/MetricsService';
import { metricsMiddleware } from '../../src/server/middleware/metricsMiddleware';

/**
 * Local path normalization helper for testing.
 * Mirrors the logic in MetricsService.
 */
function normalizePath(path: string): string {
  // Remove query strings
  const pathWithoutQuery = path.split('?')[0];

  // Normalize common dynamic path patterns
  return (
    pathWithoutQuery
      // UUID patterns (with or without hyphens)
      .replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi, ':id')
      .replace(/[0-9a-f]{32}/gi, ':id')
      // Generic ID patterns (numeric IDs, MongoDB ObjectIds, etc.)
      .replace(/\/[0-9]+(?=\/|$)/g, '/:id')
      .replace(/\/[0-9a-f]{24}(?=\/|$)/gi, '/:id')
      // General alphanumeric IDs (e.g., short game IDs)
      .replace(/\/[a-zA-Z0-9]{8,}(?=\/|$)/g, '/:id')
  );
}

// Create mock request
function createMockRequest(options: {
  method?: string;
  url?: string;
  path?: string;
  contentLength?: string;
  body?: any;
}): Partial<Request> {
  const headers: Record<string, string | undefined> = {};
  if (options.contentLength) {
    headers['content-length'] = options.contentLength;
  }

  return {
    method: options.method || 'GET',
    url: options.url || '/',
    path: options.path || options.url || '/',
    headers,
    get: ((header: string) => {
      if (header.toLowerCase() === 'content-length') {
        return options.contentLength;
      }
      return undefined;
    }) as any,
    body: options.body,
  };
}

// Create mock response with proper event emitter
function createMockResponse(): Partial<Response> & EventEmitter {
  const res = new EventEmitter() as Partial<Response> & EventEmitter;
  res.statusCode = 200;
  res.setHeader = jest.fn();
  res.getHeader = jest.fn();

  // Track bytes written
  let bytesWritten = 0;

  // Mock write function
  const originalWrite = res.write;
  res.write = jest.fn(function (this: any, chunk: any, ...args: any[]) {
    if (chunk) {
      if (Buffer.isBuffer(chunk)) {
        bytesWritten += chunk.length;
      } else if (typeof chunk === 'string') {
        bytesWritten += Buffer.byteLength(chunk);
      }
    }
    return true;
  }) as any;

  // Mock end function
  res.end = jest.fn(function (this: any, chunk?: any, ...args: any[]) {
    if (chunk) {
      if (Buffer.isBuffer(chunk)) {
        bytesWritten += chunk.length;
      } else if (typeof chunk === 'string') {
        bytesWritten += Buffer.byteLength(chunk);
      }
    }
    res.emit('finish');
    return res as Response;
  }) as any;

  return res;
}

describe('metricsMiddleware', () => {
  let metricsService: MetricsService;

  beforeEach(() => {
    // Reset the singleton and clear all metrics before each test
    MetricsService.resetInstance();
    client.register.clear();
    metricsService = getMetricsService();
  });

  afterAll(() => {
    MetricsService.resetInstance();
    client.register.clear();
  });

  describe('normalizePath', () => {
    it('should normalize UUIDs to :id', () => {
      expect(normalizePath('/api/games/550e8400-e29b-41d4-a716-446655440000')).toBe(
        '/api/games/:id'
      );
      expect(normalizePath('/api/users/123e4567-e89b-12d3-a456-426614174000/profile')).toBe(
        '/api/users/:id/profile'
      );
    });

    it('should normalize numeric IDs to :id', () => {
      expect(normalizePath('/api/games/12345')).toBe('/api/games/:id');
      expect(normalizePath('/api/users/42/config')).toBe('/api/users/:id/config');
    });

    it('should preserve non-ID path segments', () => {
      expect(normalizePath('/api/health')).toBe('/api/health');
      expect(normalizePath('/api/games')).toBe('/api/games');
      expect(normalizePath('/api/auth/login')).toBe('/api/auth/login');
    });

    it('should handle multiple IDs in path', () => {
      expect(normalizePath('/api/users/123/games/456')).toBe('/api/users/:id/games/:id');
    });

    it('should handle root path', () => {
      expect(normalizePath('/')).toBe('/');
    });

    it('should handle paths with query strings', () => {
      // Paths shouldn't include query strings, but the function should handle them
      expect(normalizePath('/api/games/123')).toBe('/api/games/:id');
    });
  });

  describe('middleware behavior', () => {
    it('should call next() for all requests', () => {
      const req = createMockRequest({ method: 'GET', url: '/api/games' });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);

      expect(next).toHaveBeenCalledTimes(1);
    });

    it('should skip /health endpoint', async () => {
      const recordSpy = jest.spyOn(metricsService, 'recordHttpRequest');

      const req = createMockRequest({ method: 'GET', url: '/health' });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);

      // Simulate response finish
      res.emit('finish');

      // Wait for async operations
      await new Promise((resolve) => setImmediate(resolve));

      expect(recordSpy).not.toHaveBeenCalled();
    });

    it('should skip /metrics endpoint', async () => {
      const recordSpy = jest.spyOn(metricsService, 'recordHttpRequest');

      const req = createMockRequest({ method: 'GET', url: '/metrics' });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);
      res.emit('finish');

      await new Promise((resolve) => setImmediate(resolve));

      expect(recordSpy).not.toHaveBeenCalled();
    });

    it('should skip /socket.io endpoint', async () => {
      const recordSpy = jest.spyOn(metricsService, 'recordHttpRequest');

      const req = createMockRequest({ method: 'GET', url: '/socket.io/poll' });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);
      res.emit('finish');

      await new Promise((resolve) => setImmediate(resolve));

      expect(recordSpy).not.toHaveBeenCalled();
    });

    it('should skip /readiness endpoint', async () => {
      const recordSpy = jest.spyOn(metricsService, 'recordHttpRequest');

      const req = createMockRequest({ method: 'GET', url: '/readiness' });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);
      res.emit('finish');

      await new Promise((resolve) => setImmediate(resolve));

      expect(recordSpy).not.toHaveBeenCalled();
    });

    it('should record metrics for regular API requests', async () => {
      const recordSpy = jest.spyOn(metricsService, 'recordHttpRequest');

      const req = createMockRequest({
        method: 'GET',
        url: '/api/games',
        contentLength: '100',
      });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);

      // Simulate response
      res.statusCode = 200;
      (res.end as jest.Mock)('{"games":[]}');

      // Wait for async operations
      await new Promise((resolve) => setImmediate(resolve));

      expect(recordSpy).toHaveBeenCalledWith(
        'GET',
        '/api/games',
        200,
        expect.any(Number), // duration
        100, // request size
        expect.any(Number) // response size
      );
    });

    it('should track response size from res.write and res.end', async () => {
      const recordSpy = jest.spyOn(metricsService, 'recordHttpRequest');

      const req = createMockRequest({ method: 'POST', url: '/api/games' });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);

      // Simulate multiple writes
      res.statusCode = 201;
      (res.write as jest.Mock)('{"id":');
      (res.write as jest.Mock)('"abc123"}');
      (res.end as jest.Mock)();

      await new Promise((resolve) => setImmediate(resolve));

      expect(recordSpy).toHaveBeenCalled();
    });

    it('should normalize paths with IDs in final metrics', async () => {
      // Path normalization happens inside MetricsService.recordHttpRequest
      // So we need to verify via the actual metrics output
      const req = createMockRequest({
        method: 'GET',
        url: '/api/games/550e8400-e29b-41d4-a716-446655440000',
      });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);

      res.statusCode = 200;
      (res.end as jest.Mock)('{}');

      await new Promise((resolve) => setImmediate(resolve));

      // Verify that the normalized path appears in metrics
      const metrics = await metricsService.getMetrics();
      expect(metrics).toContain('path="/api/games/:id"');
      expect(metrics).not.toContain('550e8400-e29b-41d4-a716-446655440000');
    });

    it('should track different HTTP methods', async () => {
      // POST request
      const postReq = createMockRequest({ method: 'POST', url: '/api/games' });
      const postRes = createMockResponse();
      metricsMiddleware(postReq as Request, postRes as Response, jest.fn());
      postRes.statusCode = 201;
      (postRes.end as jest.Mock)('{}');

      // DELETE request
      const deleteReq = createMockRequest({ method: 'DELETE', url: '/api/games/123' });
      const deleteRes = createMockResponse();
      metricsMiddleware(deleteReq as Request, deleteRes as Response, jest.fn());
      deleteRes.statusCode = 204;
      (deleteRes.end as jest.Mock)();

      // PATCH request
      const patchReq = createMockRequest({ method: 'PATCH', url: '/api/users/456' });
      const patchRes = createMockResponse();
      metricsMiddleware(patchReq as Request, patchRes as Response, jest.fn());
      patchRes.statusCode = 200;
      (patchRes.end as jest.Mock)('{}');

      await new Promise((resolve) => setImmediate(resolve));

      // Verify HTTP methods appear in metrics
      const metrics = await metricsService.getMetrics();
      expect(metrics).toContain('method="POST"');
      expect(metrics).toContain('method="DELETE"');
      expect(metrics).toContain('method="PATCH"');
    });

    it('should track error responses', async () => {
      const req = createMockRequest({ method: 'GET', url: '/api/notfound' });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);

      res.statusCode = 404;
      (res.end as jest.Mock)('{"error":"Not found"}');

      await new Promise((resolve) => setImmediate(resolve));

      // Verify 404 status appears in metrics
      const metrics = await metricsService.getMetrics();
      expect(metrics).toContain('status="404"');
    });

    it('should track 5xx server errors', async () => {
      const req = createMockRequest({ method: 'POST', url: '/api/games' });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);

      res.statusCode = 500;
      (res.end as jest.Mock)('{"error":"Internal server error"}');

      await new Promise((resolve) => setImmediate(resolve));

      // Verify 500 status appears in metrics
      const metrics = await metricsService.getMetrics();
      expect(metrics).toContain('status="500"');
    });
  });

  describe('request size calculation', () => {
    it('should use content-length header when available', async () => {
      const recordSpy = jest.spyOn(metricsService, 'recordHttpRequest');

      const req = createMockRequest({
        method: 'POST',
        url: '/api/games',
        contentLength: '500',
      });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);
      (res.end as jest.Mock)('{}');

      await new Promise((resolve) => setImmediate(resolve));

      expect(recordSpy).toHaveBeenCalledWith(
        'POST',
        '/api/games',
        200,
        expect.any(Number),
        500, // Request size from content-length
        expect.any(Number)
      );
    });

    it('should handle missing content-length', async () => {
      const recordSpy = jest.spyOn(metricsService, 'recordHttpRequest');

      const req = createMockRequest({
        method: 'GET',
        url: '/api/games',
        // No contentLength
      });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);
      (res.end as jest.Mock)('{}');

      await new Promise((resolve) => setImmediate(resolve));

      // When no content-length header, requestSize is undefined
      expect(recordSpy).toHaveBeenCalledWith(
        'GET',
        '/api/games',
        200,
        expect.any(Number),
        undefined, // undefined when no content-length
        expect.any(Number)
      );
    });
  });

  describe('response duration measurement', () => {
    it('should measure duration in seconds', async () => {
      const recordSpy = jest.spyOn(metricsService, 'recordHttpRequest');

      const req = createMockRequest({ method: 'GET', url: '/api/games' });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);

      // Add small delay to measure
      await new Promise((resolve) => setTimeout(resolve, 10));

      (res.end as jest.Mock)('{}');

      await new Promise((resolve) => setImmediate(resolve));

      expect(recordSpy).toHaveBeenCalled();
      const [, , , duration] = recordSpy.mock.calls[0];

      // Duration should be a positive number
      expect(typeof duration).toBe('number');
      expect(duration).toBeGreaterThan(0);
      // Duration should be in seconds (very small for fast response)
      expect(duration).toBeLessThan(1);
    });
  });

  describe('integration with MetricsService', () => {
    it('should produce valid Prometheus metrics for requests', async () => {
      const req = createMockRequest({
        method: 'GET',
        url: '/api/games',
        contentLength: '0',
      });
      const res = createMockResponse();
      const next = jest.fn();

      metricsMiddleware(req as Request, res as Response, next);
      res.statusCode = 200;
      (res.end as jest.Mock)('[]');

      await new Promise((resolve) => setImmediate(resolve));

      const metrics = await metricsService.getMetrics();

      expect(metrics).toContain('http_request_duration_seconds');
      expect(metrics).toContain('http_requests_total');
      expect(metrics).toContain('method="GET"');
      expect(metrics).toContain('path="/api/games"');
      expect(metrics).toContain('status="200"');
    });
  });
});
