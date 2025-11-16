import { Router, Response } from 'express';
import { getDatabaseClient } from '../database/connection';
import { AuthenticatedRequest } from '../middleware/auth';
import { createError, asyncHandler } from '../middleware/errorHandler';
import { logger } from '../utils/logger';

const router = Router();

// Get current user profile
router.get('/profile', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const userId = req.user!.id;

  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: {
      id: true,
      email: true,
      username: true,
      role: true,
      rating: true,
      gamesPlayed: true,
      gamesWon: true,
      createdAt: true,
      lastLoginAt: true,
      emailVerified: true,
      isActive: true
    }
  });

  if (!user) {
    throw createError('User not found', 404, 'USER_NOT_FOUND');
  }

  res.json({
    success: true,
    data: { user }
  });
}));

// Update user profile
router.put('/profile', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const userId = req.user!.id;
  const { username } = req.body;

  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  // Validate username if provided
  if (username) {
    if (username.length < 3 || username.length > 20) {
      throw createError('Username must be between 3 and 20 characters', 400, 'INVALID_USERNAME');
    }

    // Check if username is already taken
    const existingUser = await prisma.user.findFirst({
      where: {
        username,
        NOT: { id: userId }
      }
    });

    if (existingUser) {
      throw createError('Username already taken', 409, 'USERNAME_EXISTS');
    }
  }

  const updatedUser = await prisma.user.update({
    where: { id: userId },
    data: {
      ...(username && { username }),
      updatedAt: new Date()
    },
    select: {
      id: true,
      email: true,
      username: true,
      role: true,
      rating: true,
      gamesPlayed: true,
      gamesWon: true,
      createdAt: true,
      lastLoginAt: true,
      emailVerified: true,
      isActive: true
    }
  });

  logger.info('User profile updated', { userId });

  res.json({
    success: true,
    data: { user: updatedUser },
    message: 'Profile updated successfully'
  });
}));

// Get user statistics
router.get('/stats', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const userId = req.user!.id;

  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: {
      rating: true,
      gamesPlayed: true,
      gamesWon: true
    }
  });

  if (!user) {
    throw createError('User not found', 404, 'USER_NOT_FOUND');
  }

  // Get recent games
  const recentGames = await prisma.game.findMany({
    where: {
      OR: [
        { player1Id: userId },
        { player2Id: userId },
        { player3Id: userId },
        { player4Id: userId }
      ],
      status: 'completed'
    },
    orderBy: { endedAt: 'desc' },
    take: 10,
    select: {
      id: true,
      boardType: true,
      status: true,
      winnerId: true,
      endedAt: true,
      player1Id: true,
      player2Id: true,
      player3Id: true,
      player4Id: true
    }
  });

  // Calculate win rate
  const winRate = user.gamesPlayed > 0 ? (user.gamesWon / user.gamesPlayed) * 100 : 0;

  // Get rating history (placeholder - would need a separate table in production)
  const ratingHistory = [
    { date: new Date(), rating: user.rating }
  ];

  const stats = {
    rating: user.rating,
    gamesPlayed: user.gamesPlayed,
    gamesWon: user.gamesWon,
    gamesLost: user.gamesPlayed - user.gamesWon,
    winRate: Math.round(winRate * 100) / 100,
    recentGames,
    ratingHistory
  };

  res.json({
    success: true,
    data: { stats }
  });
}));

// Get user's game history
router.get('/games', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const userId = req.user!.id;
  const { status, limit = 20, offset = 0 } = req.query;

  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  const whereClause: any = {
    OR: [
      { player1Id: userId },
      { player2Id: userId },
      { player3Id: userId },
      { player4Id: userId }
    ]
  };

  if (status) {
    whereClause.status = status;
  }

  const games = await prisma.game.findMany({
    where: whereClause,
    include: {
      player1: { select: { id: true, username: true, rating: true } },
      player2: { select: { id: true, username: true, rating: true } },
      player3: { select: { id: true, username: true, rating: true } },
      player4: { select: { id: true, username: true, rating: true } }
    },
    orderBy: { createdAt: 'desc' },
    take: Number(limit),
    skip: Number(offset)
  });

  const total = await prisma.game.count({ where: whereClause });

  res.json({
    success: true,
    data: {
      games,
      pagination: {
        total,
        limit: Number(limit),
        offset: Number(offset),
        hasMore: Number(offset) + Number(limit) < total
      }
    }
  });
}));

// Search users
router.get('/search', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { q, limit = 10 } = req.query;

  if (!q || typeof q !== 'string') {
    throw createError('Search query required', 400, 'SEARCH_QUERY_REQUIRED');
  }

  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  const users = await prisma.user.findMany({
    where: {
      username: {
        contains: q,
        mode: 'insensitive'
      },
      isActive: true
    },
    select: {
      id: true,
      username: true,
      rating: true,
      gamesPlayed: true,
      gamesWon: true
    },
    take: Number(limit),
    orderBy: { rating: 'desc' }
  });

  res.json({
    success: true,
    data: { users }
  });
}));

// Get leaderboard
router.get('/leaderboard', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { limit = 50, offset = 0 } = req.query;

  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  const users = await prisma.user.findMany({
    where: {
      isActive: true,
      gamesPlayed: { gt: 0 }
    },
    select: {
      id: true,
      username: true,
      rating: true,
      gamesPlayed: true,
      gamesWon: true
    },
    orderBy: { rating: 'desc' },
    take: Number(limit),
    skip: Number(offset)
  });

  const total = await prisma.user.count({
    where: {
      isActive: true,
      gamesPlayed: { gt: 0 }
    }
  });

  // Add rank to each user
  const usersWithRank = users.map((user: any, index: number) => ({
    ...user,
    rank: Number(offset) + index + 1,
    winRate: user.gamesPlayed > 0 ? Math.round((user.gamesWon / user.gamesPlayed) * 10000) / 100 : 0
  }));

  res.json({
    success: true,
    data: {
      users: usersWithRank,
      pagination: {
        total,
        limit: Number(limit),
        offset: Number(offset),
        hasMore: Number(offset) + Number(limit) < total
      }
    }
  });
}));

export default router;
