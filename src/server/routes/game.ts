import { Router, Response } from 'express';
import { GameStatus } from '@prisma/client';
import { getDatabaseClient } from '../database/connection';
import { AuthenticatedRequest } from '../middleware/auth';
import { createError, asyncHandler } from '../middleware/errorHandler';
import { gameRateLimiter } from '../middleware/rateLimiter';
import { logger } from '../utils/logger';
import { CreateGameSchema, CreateGameInput } from '../../shared/validation/schemas';
import { AiOpponentsConfig } from '../../shared/types/game';
import { GameEngine } from '../game/GameEngine';

const router = Router();

// Apply rate limiting to game routes
router.use(gameRateLimiter);

// Active games storage (in production, this would be in Redis)
const activeGames = new Map<string, GameEngine>();

// Get user's games
router.get('/', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  const userId = req.user!.id;
  const { status, limit = 10, offset = 0 } = req.query;

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

// Get specific game
router.get('/:gameId', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { gameId } = req.params;
  const userId = req.user!.id;

  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  const game = await prisma.game.findUnique({
    where: { id: gameId },
    include: {
      player1: { select: { id: true, username: true, rating: true } },
      player2: { select: { id: true, username: true, rating: true } },
      player3: { select: { id: true, username: true, rating: true } },
      player4: { select: { id: true, username: true, rating: true } },
      moves: {
        orderBy: { moveNumber: 'asc' },
        include: {
          player: { select: { id: true, username: true } }
        }
      }
    }
  });

  if (!game) {
    throw createError('Game not found', 404, 'GAME_NOT_FOUND');
  }

  // Check if user is a participant or spectator
  const isParticipant = [game.player1Id, game.player2Id, game.player3Id, game.player4Id]
    .filter(Boolean)
    .includes(userId);

  if (!isParticipant && !game.allowSpectators) {
    throw createError('Access denied', 403, 'ACCESS_DENIED');
  }

  res.json({
    success: true,
    data: { game }
  });
}));

// Create new game
router.post('/', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const gameData: CreateGameInput = CreateGameSchema.parse(req.body);
  const userId = req.user!.id;

  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  // Derive any initial engine-side state we want to persist, such as
  // AI opponent configuration. This remains a loose JSON blob so we can
  // evolve it without schema migrations.
  // Persist a minimal initial engine-side snapshot. We keep this loosely
  // typed at the DB boundary but base it on the shared AiOpponentsConfig
  // so WebSocketServer.getOrCreateGameEngine can reconstruct per-player
  // AIProfile values in a type-safe way.
  const initialGameState: { aiOpponents?: AiOpponentsConfig } = {};
  if (gameData.aiOpponents) {
    initialGameState.aiOpponents = gameData.aiOpponents;
  }

  // Create game in database
  const game = await prisma.game.create({
    data: {
      boardType: gameData.boardType,
      maxPlayers: gameData.maxPlayers,
      timeControl: gameData.timeControl,
      isRated: gameData.isRated,
      allowSpectators: !gameData.isPrivate,
      player1Id: userId,
      status: GameStatus.waiting,
      gameState: initialGameState,
      createdAt: new Date(),
      updatedAt: new Date()
    },
    include: {
      player1: { select: { id: true, username: true, rating: true } }
    }
  });

  // Create game engine instance (simplified for now)
  // TODO: Properly integrate with GameEngine class
  activeGames.set(game.id, {} as any);

  logger.info('Game created', { gameId: game.id, creatorId: userId });

  res.status(201).json({
    success: true,
    data: { game },
    message: 'Game created successfully'
  });
}));

// Join game
router.post('/:gameId/join', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { gameId } = req.params;
  // Simple join - no additional data needed for now
  const userId = req.user!.id;

  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  const game = await prisma.game.findUnique({
    where: { id: gameId },
    include: {
      player1: true,
      player2: true,
      player3: true,
      player4: true
    }
  });

  if (!game) {
    throw createError('Game not found', 404, 'GAME_NOT_FOUND');
  }

  if (game.status !== GameStatus.waiting) {
    throw createError('Game is not accepting players', 400, 'GAME_NOT_JOINABLE');
  }

  // Check if user is already in the game
  const existingPlayerIds = [game.player1Id, game.player2Id, game.player3Id, game.player4Id]
    .filter(Boolean);
  
  if (existingPlayerIds.includes(userId)) {
    throw createError('Already joined this game', 400, 'ALREADY_JOINED');
  }

  // Find next available player slot
  let playerSlot: string | null = null;
  if (!game.player2Id) playerSlot = 'player2Id';
  else if (!game.player3Id && game.maxPlayers >= 3) playerSlot = 'player3Id';
  else if (!game.player4Id && game.maxPlayers >= 4) playerSlot = 'player4Id';

  if (!playerSlot) {
    throw createError('Game is full', 400, 'GAME_FULL');
  }

  // Update game in database
  const updatedGame = await prisma.game.update({
    where: { id: gameId },
    data: {
      [playerSlot]: userId,
      updatedAt: new Date()
    },
    include: {
      player1: { select: { id: true, username: true, rating: true } },
      player2: { select: { id: true, username: true, rating: true } },
      player3: { select: { id: true, username: true, rating: true } },
      player4: { select: { id: true, username: true, rating: true } }
    }
  });

  // Update game engine
  const gameEngine = activeGames.get(gameId);
  if (gameEngine) {
    // Add player to game engine (simplified for now)

    // Check if game should start
    const playerCount = [updatedGame.player1Id, updatedGame.player2Id, updatedGame.player3Id, updatedGame.player4Id]
      .filter(Boolean).length;
    if (playerCount >= 2) { // Minimum players to start
      // Update game status in database
      await prisma.game.update({
        where: { id: gameId },
        data: {
          status: GameStatus.active,
          startedAt: new Date(),
          updatedAt: new Date()
        }
      });
    }
  }

  logger.info('Player joined game', { gameId, userId, playerSlot });

  res.json({
    success: true,
    data: { game: updatedGame },
    message: 'Joined game successfully'
  });
}));

// Leave game
router.post('/:gameId/leave', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { gameId } = req.params;
  const userId = req.user!.id;

  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  const game = await prisma.game.findUnique({
    where: { id: gameId }
  });

  if (!game) {
    throw createError('Game not found', 404, 'GAME_NOT_FOUND');
  }

  // Check if user is in the game
  const playerIds = [game.player1Id, game.player2Id, game.player3Id, game.player4Id];
  const playerIndex = playerIds.indexOf(userId);
  
  if (playerIndex === -1) {
    throw createError('Not a player in this game', 400, 'NOT_A_PLAYER');
  }

  if (game.status === GameStatus.active) {
    // If game is active, this is a resignation
    const gameEngine = activeGames.get(gameId);
    if (gameEngine) {
      // Handle resignation (simplified for now)
      // TODO: Implement proper resignation logic
    }

    await prisma.game.update({
      where: { id: gameId },
      data: {
        status: GameStatus.completed,
        endedAt: new Date(),
        updatedAt: new Date()
      }
    });

    logger.info('Player resigned from game', { gameId, userId });

    res.json({
      success: true,
      message: 'Resigned from game'
    });
  } else {
    // If game is waiting, remove player
    const updateData: any = { updatedAt: new Date() };
    
    if (game.player1Id === userId) updateData.player1Id = null;
    else if (game.player2Id === userId) updateData.player2Id = null;
    else if (game.player3Id === userId) updateData.player3Id = null;
    else if (game.player4Id === userId) updateData.player4Id = null;

    await prisma.game.update({
      where: { id: gameId },
      data: updateData
    });

    // Update game engine
    const gameEngine = activeGames.get(gameId);
    if (gameEngine) {
      // Remove player from game engine (simplified for now)
    }

    logger.info('Player left game', { gameId, userId });

    res.json({
      success: true,
      message: 'Left game successfully'
    });
  }
}));

// Get game moves
router.get('/:gameId/moves', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { gameId } = req.params;
  const userId = req.user!.id;

  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  const game = await prisma.game.findUnique({
    where: { id: gameId },
    select: {
      id: true,
      player1Id: true,
      player2Id: true,
      player3Id: true,
      player4Id: true,
      allowSpectators: true
    }
  });

  if (!game) {
    throw createError('Game not found', 404, 'GAME_NOT_FOUND');
  }

  // Check access permissions
  const isParticipant = [game.player1Id, game.player2Id, game.player3Id, game.player4Id]
    .filter(Boolean)
    .includes(userId);

  if (!isParticipant && !game.allowSpectators) {
    throw createError('Access denied', 403, 'ACCESS_DENIED');
  }

  const moves = await prisma.move.findMany({
    where: { gameId },
    include: {
      player: { select: { id: true, username: true } }
    },
    orderBy: { moveNumber: 'asc' }
  });

  res.json({
    success: true,
    data: { moves }
  });
}));

// Get available games to join
router.get('/lobby/available', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  const { boardType, maxPlayers } = req.query;
  const userId = req.user!.id;

  const whereClause: any = {
    status: GameStatus.waiting,
    // Exclude games where user is already a player
    NOT: {
      OR: [
        { player1Id: userId },
        { player2Id: userId },
        { player3Id: userId },
        { player4Id: userId }
      ]
    }
  };

  if (boardType) {
    whereClause.boardType = boardType;
  }

  if (maxPlayers) {
    whereClause.maxPlayers = Number(maxPlayers);
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
    take: 20
  });

  res.json({
    success: true,
    data: { games }
  });
}));

export default router;
