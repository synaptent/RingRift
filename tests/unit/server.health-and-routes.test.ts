import express from 'express';
import request from 'supertest';
import client from 'prom-client';
import setupRoutes from '../../src/server/routes';
import { requestContext } from '../../src/server/middleware/requestContext';
import { logger, httpLogger } from '../../src/server/utils/logger';
import '../../src/server/utils/rulesParityMetrics';

/**
 * Lightweight server harness for testing the public HTTP surface:
 * - /health
 * - /api (root API info endpoint)
 *
 * We intentionally avoid importing src/server/index.ts here to keep
 * tests isolated from real DB/Redis connections and actual network
 * listeners. Instead we mirror the minimal wiring from index.ts that
 * is relevant for these endpoints.
 */

function createTestApp() {
  const app = express();
  app.use(express.json());
  // Mirror the main server pipeline by attaching per-request context so that
  // downstream logging helpers (httpLogger / withRequestContext) can include
  // a correlation id.
  app.use(requestContext as any);

  // Health check endpoint (lightweight mirror of src/server/index.ts)
  app.get('/health', (_req, res) => {
    res.status(200).json({
      status: 'healthy',
      // Use concrete runtime values; the test asserts only on types and
      // the presence of these fields, not their exact values.
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: process.env.npm_package_version || 'test',
    });
  });

  // Prometheus metrics endpoint (lightweight mirror of src/server/index.ts).
  // This uses the shared default registry from prom-client so that metrics
  // declared in src/server/utils/rulesParityMetrics.ts are exposed.
  app.get('/metrics', async (_req, res) => {
    res.set('Content-Type', client.register.contentType);
    const metrics = await client.register.metrics();
    res.send(metrics);
  });

  // API routes mounted at /api
  app.use('/api', setupRoutes());

  return app;
}

describe('Server health and API info routes', () => {
  it('GET /health responds with healthy status and basic metadata', async () => {
    const app = createTestApp();

    const res = await request(app).get('/health').expect(200);

    expect(res.body).toMatchObject({
      status: 'healthy',
    });

    // Ensure the shape matches the contract without asserting exact values.
    expect(typeof res.body.timestamp).toBe('string');
    expect(typeof res.body.uptime).toBe('number');
    expect(typeof res.body.version).toBe('string');
  });

  it('GET /api returns API metadata and endpoint map', async () => {
    const app = createTestApp();

    const res = await request(app).get('/api').expect(200);

    expect(res.body).toMatchObject({
      success: true,
      message: 'RingRift API',
      version: expect.any(String),
      endpoints: {
        auth: '/api/auth',
        games: '/api/games',
        users: '/api/users',
      },
    });

    expect(typeof res.body.timestamp).toBe('string');
  });

  it('GET /metrics exposes core Prometheus metrics', async () => {
    const app = createTestApp();

    const res = await request(app).get('/metrics').expect(200);
    const body = res.text;

    expect(body).toContain('ai_move_latency_ms');
    expect(body).toContain('ai_fallback_total');
    expect(body).toContain('game_move_latency_ms');
    expect(body).toContain('websocket_connections_current');
  });

  it('requestContext + httpLogger attach requestId to log metadata', async () => {
    const app = express();
    app.use(express.json());
    app.use(requestContext as any);

    app.get('/test-log', (req, res) => {
      httpLogger.info(req, 'test_request_log');
      res.json({ ok: true });
    });

    const infoSpy = jest.spyOn(logger, 'info');

    const REQUEST_ID = 'test-request-id-123';
    await request(app).get('/test-log').set('X-Request-Id', REQUEST_ID).expect(200);

    expect(infoSpy).toHaveBeenCalled();

    const lastCall = infoSpy.mock.calls[infoSpy.mock.calls.length - 1] as any[];
    // Winston logger.info may be invoked as (infoObject) or (message, meta)
    const metaOrInfo = lastCall[lastCall.length - 1] as any;

    expect(metaOrInfo).toBeDefined();
    expect(metaOrInfo.requestId).toBe(REQUEST_ID);

    infoSpy.mockRestore();
  });
});
