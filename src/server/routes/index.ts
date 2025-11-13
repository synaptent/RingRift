import { Router } from 'express';
import authRoutes from './auth';
import gameRoutes from './game';
import userRoutes from './user';
import { authenticate } from '../middleware/auth';

export const setupRoutes = (): Router => {
  const router = Router();

  // Public routes
  router.use('/auth', authRoutes);

  // Protected routes (require authentication)
  router.use('/games', authenticate, gameRoutes);
  router.use('/users', authenticate, userRoutes);

  // API info endpoint
  router.get('/', (_req, res) => {
    res.json({
      success: true,
      message: 'RingRift API',
      version: '1.0.0',
      endpoints: {
        auth: '/api/auth',
        games: '/api/games',
        users: '/api/users'
      },
      timestamp: new Date().toISOString()
    });
  });

  return router;
};

export default setupRoutes;