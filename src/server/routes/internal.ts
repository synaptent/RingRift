import { Router, Request, Response } from 'express';
import { HealthCheckService, isServiceReady } from '../services/HealthCheckService';
import { httpLogger } from '../utils/logger';
import { internalHealthRateLimiter, alertWebhookRateLimiter } from '../middleware/rateLimiter';
import { config } from '../config';

const router = Router();

// Be defensive for unit tests that mock `config` with partial shapes.
// In normal runtime `config.healthChecks` is always present (validated in unified config).
if (config.healthChecks?.enabled) {
  /**
   * @openapi
   * /internal/health/live:
   *   get:
   *     summary: Internal liveness probe
   *     description: |
   *       Lightweight liveness probe intended for internal routing and container health checks.
   *     tags:
   *       - Internal
   *     responses:
   *       200:
   *         description: Service is live
   */
  router.get('/health/live', internalHealthRateLimiter, (_req: Request, res: Response) => {
    const status = HealthCheckService.getLivenessStatus();
    res.status(200).json(status);
  });

  /**
   * @openapi
   * /internal/health/ready:
   *   get:
   *     summary: Internal readiness probe
   *     description: |
   *       Readiness probe intended for internal routing and orchestration.
   *     tags:
   *       - Internal
   *     responses:
   *       200:
   *         description: Service is ready
   *       503:
   *         description: Service is not ready
   */
  router.get('/health/ready', internalHealthRateLimiter, async (_req: Request, res: Response) => {
    const status = await HealthCheckService.getReadinessStatus();
    const httpStatus = isServiceReady(status) ? 200 : 503;
    res.status(httpStatus).json(status);
  });
}

/**
 * Alertmanager webhook receiver (local/dev convenience).
 *
 * This endpoint intentionally performs minimal validation and only logs a
 * bounded subset of the payload so that local monitoring stacks can deliver
 * alerts without external integrations.
 */
router.post('/alert-webhook', alertWebhookRateLimiter, (req: Request, res: Response) => {
  const body = (req.body ?? {}) as Record<string, unknown>;
  const receiver = typeof body.receiver === 'string' ? body.receiver.slice(0, 200) : undefined;
  const status = typeof body.status === 'string' ? body.status.slice(0, 50) : undefined;

  const alertsRaw = Array.isArray(body.alerts) ? body.alerts : [];
  const alerts = alertsRaw.slice(0, 25).map((alert) => {
    const a = (alert ?? {}) as Record<string, unknown>;
    const labels =
      typeof a.labels === 'object' && a.labels !== null
        ? (a.labels as Record<string, unknown>)
        : {};
    const annotations =
      typeof a.annotations === 'object' && a.annotations !== null
        ? (a.annotations as Record<string, unknown>)
        : {};

    const safeStringMap = (source: Record<string, unknown>) => {
      const out: Record<string, string> = {};
      for (const [key, value] of Object.entries(source)) {
        if (typeof value === 'string') {
          out[key] = value.slice(0, 500);
        }
      }
      return out;
    };

    return {
      status: typeof a.status === 'string' ? a.status.slice(0, 50) : undefined,
      labels: safeStringMap(labels),
      annotations: safeStringMap(annotations),
      startsAt: typeof a.startsAt === 'string' ? a.startsAt : undefined,
      endsAt: typeof a.endsAt === 'string' ? a.endsAt : undefined,
    };
  });

  httpLogger.warn(req, 'Alertmanager webhook received', {
    event: 'alert_webhook',
    receiver,
    status,
    alerts,
    alertCount: alertsRaw.length,
  });

  res.status(204).send();
});

export default router;
