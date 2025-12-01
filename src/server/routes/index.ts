import { Router } from 'express';
// eslint-disable-next-line @typescript-eslint/no-var-requires, @typescript-eslint/no-require-imports
const swaggerUi = require('swagger-ui-express');
import authRoutes from './auth';
import gameRoutes, { setWebSocketServer as setGameWebSocketServer } from './game';
import userRoutes, { setWebSocketServer as setUserWebSocketServer } from './user';
import adminRoutes from './admin';
import { authenticate } from '../middleware/auth';
import { httpLogger } from '../utils/logger';
import { swaggerSpec } from '../openapi/config';

export const setupRoutes = (wsServer?: any): Router => {
  const router = Router();

  // Inject WebSocket server into routes that need it.
  // Explicitly set to null when not provided to ensure proper test isolation
  // and predictable behavior when routes are set up without a WebSocket server.
  setGameWebSocketServer(wsServer ?? null);
  setUserWebSocketServer(wsServer ?? null);

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

  // Public routes
  router.use('/auth', authRoutes);

  // Protected routes (require authentication)
  router.use('/games', authenticate, gameRoutes);
  router.use('/users', authenticate, userRoutes);
  router.use('/admin', adminRoutes);

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
  router.post('/client-errors', (req, res) => {
    const body = (req as any).body ?? {};
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
