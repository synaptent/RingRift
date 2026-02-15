import { Router } from 'express';
import swaggerUi from 'swagger-ui-express';
import authRoutes from './auth';
import gameRoutes, {
  sandboxHelperRoutes,
  setWebSocketServer as setGameWebSocketServer,
} from './game';
import userRoutes, { setWebSocketServer as setUserWebSocketServer } from './user';
import adminRoutes from './admin';
import selfplayRoutes from './selfplay';
import internalRoutes from './internal';
import rulesUxTelemetryRoutes from './rulesUxTelemetry';
import difficultyCalibrationTelemetryRoutes from './difficultyCalibrationTelemetry';
import trainingExportRoutes from './training-export';
import { authenticate } from '../middleware/auth';
import { clientErrorsRateLimiter } from '../middleware/rateLimiter';
import { httpLogger } from '../utils/logger';
import { swaggerSpec } from '../openapi/config';
import type { WebSocketServer } from '../websocket/server';

export const setupRoutes = (wsServer: WebSocketServer): Router => {
  const router = Router();

  // Inject WebSocket server into routes that need it.
  // In production we always attach a WebSocketServer instance here; tests that
  // need to exercise routes without WebSocket integration should wire the game
  // and user routers directly rather than going through setupRoutes.
  setGameWebSocketServer(wsServer);
  setUserWebSocketServer(wsServer);

  // OpenAPI Documentation
  // Swagger UI at /api/docs
  router.use(
    '/docs',
    swaggerUi.serve,
    swaggerUi.setup(swaggerSpec, {
      customCss: '.swagger-ui .topbar { display: none }',
      customSiteTitle: 'RingRift API Documentation',
    })
  );

  // Raw OpenAPI spec at /api/docs.json
  router.get('/docs.json', (_req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.send(swaggerSpec);
  });

  /**
   * @openapi
   * /health:
   *   get:
   *     summary: Simple health check
   *     description: |
   *       Returns a simple health status. For detailed health checks, use /api/internal/health/live
   *       or /api/internal/health/ready.
   *     responses:
   *       200:
   *         description: Service is healthy
   *         content:
   *           application/json:
   *             schema:
   *               type: object
   *               properties:
   *                 status:
   *                   type: string
   *                   example: ok
   *                 timestamp:
   *                   type: string
   *                   format: date-time
   */
  router.get('/health', (_req, res) => {
    res.json({
      status: 'ok',
      timestamp: new Date().toISOString(),
    });
  });

  // Public routes
  router.use('/auth', authRoutes);
  router.use('/selfplay', selfplayRoutes); // Read-only access to recorded self-play games
  router.use('/internal', internalRoutes);
  router.use('/training-export', trainingExportRoutes); // Export human vs AI games for training
  router.use('/telemetry', rulesUxTelemetryRoutes);
  router.use('/telemetry', difficultyCalibrationTelemetryRoutes);
  router.use('/games', sandboxHelperRoutes);

  // Protected routes (require authentication)
  router.use('/games', authenticate, gameRoutes);
  router.use('/users', authenticate, userRoutes);
  router.use('/admin', authenticate, adminRoutes);

  /**
   * @openapi
   * /client-errors:
   *   post:
   *     summary: Report client-side errors from the SPA
   *     description: |
   *       Accepts error reports from the RingRift web client and logs them for diagnostics.
   *       This endpoint is unauthenticated and returns no content.
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             description: Arbitrary error payload from the client
   *             additionalProperties: true
   *     responses:
   *       204:
   *         description: Error report accepted
   */
  // Client-side error reporting endpoint (no auth; used by SPA error reporter)
  router.post('/client-errors', clientErrorsRateLimiter, (req, res) => {
    const body = (req.body ?? {}) as Record<string, unknown>;
    const { name, message, stack, type, url, userAgent, timestamp, context } = body as {
      name?: unknown;
      message?: unknown;
      stack?: unknown;
      type?: unknown;
      url?: unknown;
      userAgent?: unknown;
      timestamp?: unknown;
      context?: unknown;
    };

    const safe = {
      name: typeof name === 'string' ? name.slice(0, 200) : undefined,
      message: typeof message === 'string' ? message.slice(0, 500) : undefined,
      stack: typeof stack === 'string' ? stack.slice(0, 2000) : undefined,
      type: typeof type === 'string' ? type.slice(0, 100) : undefined,
      url: typeof url === 'string' ? url.slice(0, 500) : undefined,
      userAgent: typeof userAgent === 'string' ? userAgent.slice(0, 500) : undefined,
      timestamp: typeof timestamp === 'string' ? timestamp : undefined,
      context: typeof context === 'object' && context !== null ? context : undefined,
    };

    httpLogger.error(req, 'Client error reported from SPA', {
      event: 'client_error',
      ...safe,
    });

    res.status(204).send();
  });

  /**
   * @openapi
   * /:
   *   get:
   *     summary: API info
   *     description: |
   *       Returns basic metadata about the RingRift API and links to documentation.
   *     responses:
   *       200:
   *         description: API information
   */
  // API info endpoint
  router.get('/', (_req, res) => {
    res.json({
      success: true,
      message: 'RingRift API',
      version: '1.0.0',
      endpoints: {
        auth: '/api/auth',
        games: '/api/games',
        users: '/api/users',
        selfplay: '/api/selfplay',
        internal: '/api/internal',
        docs: '/api/docs',
      },
      documentation: '/api/docs',
      openApiSpec: '/api/docs.json',
      timestamp: new Date().toISOString(),
    });
  });

  return router;
};

export default setupRoutes;
