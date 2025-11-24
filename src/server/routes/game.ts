import { Router, Response } from 'express';
import { getDatabaseClient } from '../database/connection';
import { AuthenticatedRequest } from '../middleware/auth';
import { createError, asyncHandler } from '../middleware/errorHandler';
import { gameRateLimiter, consumeRateLimit } from '../middleware/rateLimiter';
import { httpLogger, logger } from '../utils/logger';
import { CreateGameSchema, CreateGameInput } from '../../shared/validation/schemas';
import { AiOpponentsConfig } from '../../shared/types/game';
import { GameEngine } from '../game/GameEngine';

const router = Router();

// WebSocket server instance will be injected
let wsServerInstance: any = null;

export function setWebSocketServer(wsServer: any) {
  wsServerInstance = wsServer;
}

// Apply rate limiting to game routes
router.use(gameRateLimiter);

// Active games storage (in production, this would be in Redis)
const activeGames = new Map<string, GameEngine>();

/**
 * Lightweight view of game participants used for HTTP-level authorization
 * checks. This intentionally mirrors only the player slots and spectator
 * flag that are required to enforce access control invariants.
 */
type GameParticipantSnapshot = {
  player1Id: string | null;
  player2Id: string | null;
  player3Id: string | null;
  player4Id: string | null;
  allowSpectators?: boolean | null;
};

const isUserParticipantInGame = (userId: string, game: GameParticipantSnapshot): boolean => {
  return [game.player1Id, game.player2Id, game.player3Id, game.player4Id]
    .filter(Boolean)
    .includes(userId);
};

/**
 * Enforce the invariant that only participants (or, when enabled, permitted
 * spectators) may inspect game-scoped HTTP resources.
 *
 * This guard is shared by both the game-details and move-history endpoints so
 * that the authorization rule remains obvious and consistent.
 */
const assertUserCanViewGame = (
  userId: string,
  game: GameParticipantSnapshot & { allowSpectators: boolean }
): void => {
  const isParticipant = isUserParticipantInGame(userId, game);

  if (!isParticipant && !game.allowSpectators) {
    throw createError('Access denied', 403, 'ACCESS_DENIED');
  }
};

/**
 * Enforce the invariant that only seated human participants in a game may
 * perform HTTP mutations that change its state (e.g. resignation via
 * POST /api/games/:gameId/leave).
 *
 * This helper intentionally mirrors the lightweight GameParticipantSnapshot
 * view so that authorization rules stay consistent across endpoints without
 * depending on full Prisma types.
 */
const assertUserIsGameParticipant = (userId: string, game: GameParticipantSnapshot): void => {
  if (!isUserParticipantInGame(userId, game)) {
    throw createError('Access denied', 403, 'ACCESS_DENIED');
  }
};

// Get user's games
router.get(
  '/',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
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
        { player4Id: userId },
      ],
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
        player4: { select: { id: true, username: true, rating: true } },
      },
      orderBy: { createdAt: 'desc' },
      take: Number(limit),
      skip: Number(offset),
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
          hasMore: Number(offset) + Number(limit) < total,
        },
      },
    });
  })
);

// Get specific game
router.get(
  '/:gameId',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
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
            player: { select: { id: true, username: true } },
          },
        },
      },
    });

    if (!game) {
      throw createError('Game not found', 404, 'GAME_NOT_FOUND');
    }

    // Enforce game-level authorization: only participants (or permitted
    // spectators when allowSpectators=true) may inspect game details.
    assertUserCanViewGame(userId, game);

    res.json({
      success: true,
      data: { game },
    });
  })
);

// Create new game
router.post(
  '/',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const gameData: CreateGameInput = CreateGameSchema.parse(req.body);
    const userId = req.user!.id;

    // Derive an IP address for quota enforcement, preferring X-Forwarded-For
    // when present so deployments behind a proxy still get a stable key.
    const forwardedForHeader = req.headers['x-forwarded-for'] as string | undefined;
    const forwardedFor = forwardedForHeader?.split(',')[0]?.trim();
    const clientIp = forwardedFor || req.ip || 'unknown';

    // Per-user game creation quota.
    const userQuota = await consumeRateLimit('gameCreateUser', userId);
    if (!userQuota.allowed) {
      logger.warn('Game creation quota exceeded for user', {
        userId,
        ip: clientIp,
        limiter: 'gameCreateUser',
        retryAfter: userQuota.retryAfter,
      });

      throw createError(
        'Too many games created in a short period. Please try again later.',
        429,
        'GAME_CREATE_RATE_LIMITED'
      );
    }

    // Per-IP game creation quota as an additional guard (including for any
    // edge cases where authentication might be missing or misconfigured).
    const ipQuota = await consumeRateLimit('gameCreateIp', clientIp);
    if (!ipQuota.allowed) {
      logger.warn('Game creation quota exceeded for IP', {
        userId,
        ip: clientIp,
        limiter: 'gameCreateIp',
        retryAfter: ipQuota.retryAfter,
      });

      throw createError(
        'Too many games created from this IP address. Please try again later.',
        429,
        'GAME_CREATE_RATE_LIMITED'
      );
    }

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Validate AI game configuration
    if (gameData.aiOpponents && gameData.aiOpponents.count > 0) {
      // AI games cannot be rated (initially)
      if (gameData.isRated) {
        throw createError('AI games cannot be rated', 400, 'AI_GAMES_UNRATED');
      }

      // Ensure we have enough difficulty values
      if (gameData.aiOpponents.difficulty.length < gameData.aiOpponents.count) {
        throw createError('Must provide difficulty for each AI opponent', 400, 'INVALID_AI_CONFIG');
      }

      // Validate difficulty range
      for (const diff of gameData.aiOpponents.difficulty) {
        if (diff < 1 || diff > 10) {
          throw createError('AI difficulty must be between 1 and 10', 400, 'INVALID_DIFFICULTY');
        }
      }
    }

    // Derive initial engine-side state including AI opponent configuration
    const initialGameState: { aiOpponents?: AiOpponentsConfig } = {};
    if (gameData.aiOpponents && gameData.aiOpponents.count > 0) {
      initialGameState.aiOpponents = gameData.aiOpponents;
    }

    // Determine initial game status: AI games start immediately, human-only games wait
    const hasAIOpponents = gameData.aiOpponents && gameData.aiOpponents.count > 0;
    const initialStatus = hasAIOpponents ? ('active' as any) : ('waiting' as any);
    const startedAt = hasAIOpponents ? new Date() : undefined;

    // Create game in database
    const game = await prisma.game.create({
      data: {
        boardType: gameData.boardType as any,
        maxPlayers: gameData.maxPlayers,
        timeControl: gameData.timeControl,
        isRated: gameData.isRated,
        allowSpectators: !gameData.isPrivate,
        player1Id: userId,
        status: initialStatus,
        gameState: initialGameState as any,
        // Store seed in database when provided; coerce missing seeds to null
        // to satisfy Prisma's `number | null` type and avoid `undefined`.
        rngSeed: gameData.seed ?? null,
        createdAt: new Date(),
        updatedAt: new Date(),
        ...(startedAt && { startedAt }),
      },
      include: {
        player1: { select: { id: true, username: true, rating: true } },
      },
    });

    httpLogger.info(req, 'Game created', {
      gameId: game.id,
      creatorId: userId,
      hasAI: hasAIOpponents,
      aiCount: gameData.aiOpponents?.count ?? 0,
      status: initialStatus,
    });

    // Broadcast to lobby if game is waiting for players
    if (initialStatus === 'waiting' && wsServerInstance) {
      wsServerInstance.broadcastLobbyEvent('lobby:game_created', game);
    }

    res.status(201).json({
      success: true,
      data: { game },
      message: hasAIOpponents
        ? 'Game created and started with AI opponents'
        : 'Game created successfully',
    });
  })
);

// Join game
router.post(
  '/:gameId/join',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
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
        player4: true,
      },
    });

    if (!game) {
      throw createError('Game not found', 404, 'GAME_NOT_FOUND');
    }

    if (game.status !== ('waiting' as any)) {
      throw createError('Game is not accepting players', 400, 'GAME_NOT_JOINABLE');
    }

    // Check if user is already in the game
    const existingPlayerIds = [
      game.player1Id,
      game.player2Id,
      game.player3Id,
      game.player4Id,
    ].filter(Boolean);

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
        updatedAt: new Date(),
      },
      include: {
        player1: { select: { id: true, username: true, rating: true } },
        player2: { select: { id: true, username: true, rating: true } },
        player3: { select: { id: true, username: true, rating: true } },
        player4: { select: { id: true, username: true, rating: true } },
      },
    });

    // Broadcast player joined event to lobby
    const currentPlayerCount = [
      updatedGame.player1Id,
      updatedGame.player2Id,
      updatedGame.player3Id,
      updatedGame.player4Id,
    ].filter(Boolean).length;

    if (wsServerInstance) {
      wsServerInstance.broadcastLobbyEvent('lobby:game_joined', {
        gameId,
        playerCount: currentPlayerCount,
      });
    }

    // Update game engine
    const gameEngine = activeGames.get(gameId);
    if (gameEngine) {
      // Add player to game engine (simplified for now)

      // Check if game should start
      if (currentPlayerCount >= 2) {
        // Minimum players to start
        // Update game status in database
        const startedGame = await prisma.game.update({
          where: { id: gameId },
          data: {
            status: 'active' as any,
            startedAt: new Date(),
            updatedAt: new Date(),
          },
        });

        // Broadcast game started event to remove from lobby and provide
        // basic metadata for lobby consumers.
        if (wsServerInstance) {
          wsServerInstance.broadcastLobbyEvent('lobby:game_started', {
            gameId,
            status: startedGame.status,
            startedAt: startedGame.startedAt,
            playerCount: currentPlayerCount,
          });
        }
      }
    }

    httpLogger.info(req, 'Player joined game', { gameId, userId, playerSlot });

    res.json({
      success: true,
      data: { game: updatedGame },
      message: 'Joined game successfully',
    });
  })
);

// Leave game
router.post(
  '/:gameId/leave',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { gameId } = req.params;
    const userId = req.user!.id;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const game = await prisma.game.findUnique({
      where: { id: gameId },
    });

    if (!game) {
      throw createError('Game not found', 404, 'GAME_NOT_FOUND');
    }

    // HTTP-level authorization: only seated players may leave or resign this game.
    assertUserIsGameParticipant(userId, {
      player1Id: game.player1Id,
      player2Id: game.player2Id,
      player3Id: game.player3Id,
      player4Id: game.player4Id,
    });

    if (game.status === ('active' as any)) {
      // If game is active, this is a resignation
      const gameEngine = activeGames.get(gameId);
      if (gameEngine) {
        // Handle resignation (simplified for now)
        // TODO: Implement proper resignation logic
      }

      await prisma.game.update({
        where: { id: gameId },
        data: {
          status: 'completed' as any,
          endedAt: new Date(),
          updatedAt: new Date(),
        },
      });

      // Broadcast game ended/cancelled to lobby
      if (wsServerInstance) {
        wsServerInstance.broadcastLobbyEvent('lobby:game_cancelled', { gameId });
      }

      httpLogger.info(req, 'Player resigned from game', { gameId, userId });

      res.json({
        success: true,
        message: 'Resigned from game',
      });
    } else {
      // If game is waiting, remove player
      const updateData: any = { updatedAt: new Date() };

      if (game.player1Id === userId) updateData.player1Id = null;
      else if (game.player2Id === userId) updateData.player2Id = null;
      else if (game.player3Id === userId) updateData.player3Id = null;
      else if (game.player4Id === userId) updateData.player4Id = null;

      const updatedGame = await prisma.game.update({
        where: { id: gameId },
        data: updateData,
      });

      // Check if game should be cancelled (no players left)
      const remainingPlayers = [
        updatedGame.player1Id,
        updatedGame.player2Id,
        updatedGame.player3Id,
        updatedGame.player4Id,
      ].filter(Boolean);

      if (remainingPlayers.length === 0) {
        // Cancel the game
        await prisma.game.update({
          where: { id: gameId },
          data: { status: 'abandoned' as any, endedAt: new Date() },
        });

        // Broadcast game cancelled to lobby
        if (wsServerInstance) {
          wsServerInstance.broadcastLobbyEvent('lobby:game_cancelled', { gameId });
        }
      } else {
        // Broadcast updated player count
        if (wsServerInstance) {
          wsServerInstance.broadcastLobbyEvent('lobby:game_joined', {
            gameId,
            playerCount: remainingPlayers.length,
          });
        }
      }

      // Update game engine
      const gameEngine = activeGames.get(gameId);
      if (gameEngine) {
        // Remove player from game engine (simplified for now)
      }

      httpLogger.info(req, 'Player left game', { gameId, userId });

      res.json({
        success: true,
        message: 'Left game successfully',
      });
    }
  })
);

// Get game moves
router.get(
  '/:gameId/moves',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
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
        allowSpectators: true,
      },
    });

    if (!game) {
      throw createError('Game not found', 404, 'GAME_NOT_FOUND');
    }

    // Reuse the same authorization invariant as the game-details endpoint:
    // a caller must be either a participant or, when enabled, a permitted
    // spectator to inspect the move history.
    assertUserCanViewGame(userId, game);

    const moves = await prisma.move.findMany({
      where: { gameId },
      include: {
        player: { select: { id: true, username: true } },
      },
      orderBy: { moveNumber: 'asc' },
    });

    res.json({
      success: true,
      data: { moves },
    });
  })
);

// Get available games to join
router.get(
  '/lobby/available',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const { boardType, maxPlayers } = req.query;
    const userId = req.user!.id;

    const whereClause: any = {
      status: 'waiting' as any,
      // Exclude games where user is already a player
      NOT: {
        OR: [
          { player1Id: userId },
          { player2Id: userId },
          { player3Id: userId },
          { player4Id: userId },
        ],
      },
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
        player4: { select: { id: true, username: true, rating: true } },
      },
      orderBy: { createdAt: 'desc' },
      take: 20,
    });

    res.json({
      success: true,
      data: { games },
    });
  })
);

export default router;
