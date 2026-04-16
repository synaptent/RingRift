import express from 'express';
import request from 'supertest';
import { requestContext } from '../../src/server/middleware/requestContext';
import { HealthCheckService } from '../../src/server/services/HealthCheckService';
import internalRoutes from '../../src/server/routes/internal';

jest.mock('../../src/server/middleware/rateLimiter', () => ({
  internalHealthRateLimiter: (_req: unknown, _res: unknown, next: () => void) => next(),
  alertWebhookRateLimiter: (_req: unknown, _res: unknown, next: () => void) => next(),
}));

function createTestApp() {
  const app = express();
  app.use(express.json());
  app.use(requestContext as any);
  app.use('/api/internal', internalRoutes);
  return app;
}

describe('Internal routes', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('GET /api/internal/health/live returns liveness payload', async () => {
    const app = createTestApp();
    const res = await request(app).get('/api/internal/health/live').expect(200);
    expect(res.body).toMatchObject({ status: expect.any(String) });
  });

  it('GET /api/internal/health/ready returns 200 when ready', async () => {
    jest.spyOn(HealthCheckService, 'getReadinessStatus').mockResolvedValue({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: '1.0.0-test',
      uptime: 1,
      checks: {},
    } as any);

    const app = createTestApp();
    await request(app).get('/api/internal/health/ready').expect(200);
  });

  it('GET /api/internal/health/ready returns 503 when not ready', async () => {
    jest.spyOn(HealthCheckService, 'getReadinessStatus').mockResolvedValue({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      version: '1.0.0-test',
      uptime: 1,
      checks: {
        database: { status: 'unhealthy', error: 'Connection refused' },
      },
    } as any);

    const app = createTestApp();
    await request(app).get('/api/internal/health/ready').expect(503);
  });

  it('POST /api/internal/alert-webhook accepts payload', async () => {
    const app = createTestApp();

    const payload = {
      receiver: 'webhook-log',
      status: 'firing',
      alerts: [
        {
          status: 'firing',
          labels: { alertname: 'TestAlert', severity: 'warning' },
          annotations: { summary: 'Test', description: 'Test alert' },
          startsAt: new Date().toISOString(),
        },
      ],
    };

    await request(app).post('/api/internal/alert-webhook').send(payload).expect(204);
  });
});
