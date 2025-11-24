import { Router } from 'express';
import authRoutes from './auth';
import gameRoutes, { setWebSocketServer } from './game';
import userRoutes from './user';
import { authenticate } from '../middleware/auth';
import { httpLogger } from '../utils/logger';

export const setupRoutes = (wsServer?: any): Router => {
  const router = Router();

  // Inject WebSocket server into game routes for lobby broadcasting
  if (wsServer) {
    setWebSocketServer(wsServer);
  }

  // Public routes
  router.use('/auth', authRoutes);

  // Protected routes (require authentication)
  router.use('/games', authenticate, gameRoutes);
  router.use('/users', authenticate, userRoutes);

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
      },
      timestamp: new Date().toISOString(),
    });
  });

  return router;
};

export default setupRoutes;
