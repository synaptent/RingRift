import { Router, Response } from 'express';
import { getDatabaseClient } from '../database/connection';
import { AuthenticatedRequest } from '../middleware/auth';
import { createError, asyncHandler } from '../middleware/errorHandler';
import { consumeRateLimit, adaptiveRateLimiter } from '../middleware/rateLimiter';
import { httpLogger, logger } from '../utils/logger';
import {
  CreateGameSchema,
  CreateGameInput,
  GameIdParamSchema,
  GameListingQuerySchema,
} from '../../shared/validation/schemas';
import { AiOpponentsConfig } from '../../shared/types/game';
import { GameEngine } from '../game/GameEngine';

const router = Router();

// WebSocket server instance will be injected
let wsServerInstance: any = null;

export function setWebSocketServer(wsServer: any) {
  wsServerInstance = wsServer;
}

// Apply adaptive rate limiting to game routes
// Authenticated users get higher limits than anonymous
router.use(adaptiveRateLimiter('game', 'api'));

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

/**
 * @openapi
 * /games:
 *   get:
 *     summary: Get user's games
 *     description: |
 *       Returns a paginated list of games the authenticated user is participating in.
 *       Can be filtered by game status.
 *     tags: [Games]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: status
 *         schema:
 *           type: string
 *           enum: [waiting, active, completed, abandoned, paused]
 *         description: Filter by game status
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 100
 *           default: 20
 *         description: Number of results per page
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           minimum: 0
 *           default: 0
 *         description: Pagination offset
 *     responses:
 *       200:
 *         description: Games retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     games:
 *                       type: array
 *                       items:
 *                         $ref: '#/components/schemas/Game'
 *                     pagination:
 *                       $ref: '#/components/schemas/Pagination'
 *       400:
 *         description: Invalid query parameters
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: VALIDATION_INVALID_QUERY_PARAMS
 *                 message: Invalid query parameters
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    // Validate query parameters with schema
    const queryResult = GameListingQuerySchema.safeParse(req.query);
    if (!queryResult.success) {
      throw createError('Invalid query parameters', 400, 'INVALID_QUERY_PARAMS');
    }
    const { status, limit, offset } = queryResult.data;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const userId = req.user!.id;

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
      take: limit,
      skip: offset,
    });

    const total = await prisma.game.count({ where: whereClause });

    res.json({
      success: true,
      data: {
        games,
        pagination: {
          total,
          limit,
          offset,
          hasMore: offset + limit < total,
        },
      },
    });
  })
);

/**
 * @openapi
 * /games/{gameId}:
 *   get:
 *     summary: Get specific game
 *     description: |
 *       Returns detailed information about a specific game including players and move history.
 *       Only participants and spectators (when enabled) can access this endpoint.
 *     tags: [Games]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: gameId
 *         required: true
 *         schema:
 *           type: string
 *           format: uuid
 *         description: Game ID
 *     responses:
 *       200:
 *         description: Game retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     game:
 *                       $ref: '#/components/schemas/Game'
 *       400:
 *         description: Invalid game ID format
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: GAME_INVALID_ID
 *                 message: Invalid game ID format
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       403:
 *         description: Access denied (not a participant and spectators not allowed)
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: GAME_ACCESS_DENIED
 *                 message: Access denied
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/:gameId',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    // Validate gameId parameter
    const paramResult = GameIdParamSchema.safeParse(req.params);
    if (!paramResult.success) {
      throw createError('Invalid game ID format', 400, 'INVALID_GAME_ID');
    }
    const { gameId } = paramResult.data;
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

/**
 * @openapi
 * /games:
 *   post:
 *     summary: Create new game
 *     description: |
 *       Creates a new game with the specified settings.
 *       The authenticated user becomes player 1 (game creator).
 *
 *       Rate limited:
 *       - 5 games per hour per user
 *       - 10 games per hour per IP address
 *
 *       AI games:
 *       - Cannot be rated (isRated must be false)
 *       - Start immediately with AI opponents
 *       - Must provide difficulty for each AI opponent
 *     tags: [Games]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/CreateGameRequest'
 *     responses:
 *       201:
 *         description: Game created successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     game:
 *                       $ref: '#/components/schemas/Game'
 *                 message:
 *                   type: string
 *                   example: Game created successfully
 *       400:
 *         description: Invalid game configuration
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             examples:
 *               aiUnrated:
 *                 summary: AI games must be unrated
 *                 value:
 *                   success: false
 *                   error:
 *                     code: GAME_AI_UNRATED
 *                     message: AI games cannot be rated
 *               invalidAiConfig:
 *                 summary: Invalid AI configuration
 *                 value:
 *                   success: false
 *                   error:
 *                     code: VALIDATION_INVALID_AI_CONFIG
 *                     message: Must provide difficulty for each AI opponent
 *               invalidDifficulty:
 *                 summary: Invalid difficulty level
 *                 value:
 *                   success: false
 *                   error:
 *                     code: VALIDATION_INVALID_DIFFICULTY
 *                     message: AI difficulty must be between 1 and 10
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       429:
 *         description: Rate limit exceeded
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: RATE_LIMIT_GAME_CREATE
 *                 message: Too many games created in a short period. Please try again later.
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
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

/**
 * @openapi
 * /games/{gameId}/join:
 *   post:
 *     summary: Join a game
 *     description: |
 *       Joins an existing game that is waiting for players.
 *       The user is assigned to the next available player slot.
 *       When enough players have joined, the game starts automatically.
 *     tags: [Games]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: gameId
 *         required: true
 *         schema:
 *           type: string
 *           format: uuid
 *         description: Game ID to join
 *     responses:
 *       200:
 *         description: Joined game successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     game:
 *                       $ref: '#/components/schemas/Game'
 *                 message:
 *                   type: string
 *                   example: Joined game successfully
 *       400:
 *         description: Cannot join game
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             examples:
 *               invalidId:
 *                 summary: Invalid game ID
 *                 value:
 *                   success: false
 *                   error:
 *                     code: GAME_INVALID_ID
 *                     message: Invalid game ID format
 *               notJoinable:
 *                 summary: Game not accepting players
 *                 value:
 *                   success: false
 *                   error:
 *                     code: GAME_NOT_JOINABLE
 *                     message: Game is not accepting players
 *               alreadyJoined:
 *                 summary: Already in game
 *                 value:
 *                   success: false
 *                   error:
 *                     code: GAME_ALREADY_JOINED
 *                     message: Already joined this game
 *               gameFull:
 *                 summary: Game is full
 *                 value:
 *                   success: false
 *                   error:
 *                     code: GAME_FULL
 *                     message: Game is full
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.post(
  '/:gameId/join',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    // Validate gameId parameter
    const paramResult = GameIdParamSchema.safeParse(req.params);
    if (!paramResult.success) {
      throw createError('Invalid game ID format', 400, 'INVALID_GAME_ID');
    }
    const { gameId } = paramResult.data;
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

/**
 * @openapi
 * /games/{gameId}/leave:
 *   post:
 *     summary: Leave or resign from a game
 *     description: |
 *       Leaves a waiting game or resigns from an active game.
 *
 *       If the game is **waiting**:
 *       - User is removed from their player slot
 *       - If no players remain, the game is cancelled
 *
 *       If the game is **active**:
 *       - This counts as a resignation
 *       - The game ends immediately
 *       - Rating changes are applied (if rated)
 *     tags: [Games]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: gameId
 *         required: true
 *         schema:
 *           type: string
 *           format: uuid
 *         description: Game ID to leave
 *     responses:
 *       200:
 *         description: Left/resigned from game successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 message:
 *                   type: string
 *                   example: Left game successfully
 *       400:
 *         description: Invalid game ID format
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: GAME_INVALID_ID
 *                 message: Invalid game ID format
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       403:
 *         description: Not a participant in this game
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: GAME_ACCESS_DENIED
 *                 message: Access denied
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.post(
  '/:gameId/leave',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    // Validate gameId parameter
    const paramResult = GameIdParamSchema.safeParse(req.params);
    if (!paramResult.success) {
      throw createError('Invalid game ID format', 400, 'INVALID_GAME_ID');
    }
    const { gameId } = paramResult.data;
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

/**
 * @openapi
 * /games/{gameId}/moves:
 *   get:
 *     summary: Get game moves
 *     description: |
 *       Returns all moves made in a game, ordered by move number.
 *       Only participants and spectators (when enabled) can access this endpoint.
 *     tags: [Games]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: gameId
 *         required: true
 *         schema:
 *           type: string
 *           format: uuid
 *         description: Game ID
 *     responses:
 *       200:
 *         description: Moves retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     moves:
 *                       type: array
 *                       items:
 *                         $ref: '#/components/schemas/Move'
 *       400:
 *         description: Invalid game ID format
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: GAME_INVALID_ID
 *                 message: Invalid game ID format
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       403:
 *         $ref: '#/components/responses/Forbidden'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/:gameId/moves',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    // Validate gameId parameter
    const paramResult = GameIdParamSchema.safeParse(req.params);
    if (!paramResult.success) {
      throw createError('Invalid game ID format', 400, 'INVALID_GAME_ID');
    }
    const { gameId } = paramResult.data;
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

/**
 * @openapi
 * /games/{gameId}/diagnostics/session:
 *   get:
 *     summary: Get in-memory session and connection diagnostics for a game
 *     description: |
 *       Returns a compact diagnostics snapshot for a specific game, combining
 *       the GameSession state-machine projections with WebSocket connection
 *       state. Only participants and permitted spectators may access this
 *       endpoint.
 *
 *       The diagnostics are best-effort and reflect only in-memory sessions;
 *       games that are not currently loaded into memory will return
 *       `hasInMemorySession: false`.
 *     tags: [Games]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: gameId
 *         required: true
 *         schema:
 *           type: string
 *           format: uuid
 *         description: Game ID
 *     responses:
 *       200:
 *         description: Diagnostics snapshot retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     sessionStatus:
 *                       type: object
 *                       nullable: true
 *                       description: Derived GameSessionStatus projection (shape is internal and may evolve)
 *                     lastAIRequestState:
 *                       type: object
 *                       nullable: true
 *                       description: Last AIRequestState snapshot for this game
 *                     aiDiagnostics:
 *                       type: object
 *                       nullable: true
 *                       description: Per-game AI/rules degraded-mode diagnostics
 *                     connections:
 *                       type: object
 *                       additionalProperties:
 *                         $ref: '#/components/schemas/PlayerConnectionState'
 *                     meta:
 *                       type: object
 *                       properties:
 *                         hasInMemorySession:
 *                           type: boolean
 *                           description: Whether an in-memory GameSession was found on this node
 *       400:
 *         description: Invalid game ID format
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: GAME_INVALID_ID
 *                 message: Invalid game ID format
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       403:
 *         $ref: '#/components/responses/Forbidden'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/:gameId/diagnostics/session',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const paramResult = GameIdParamSchema.safeParse(req.params);
    if (!paramResult.success) {
      throw createError('Invalid game ID format', 400, 'INVALID_GAME_ID');
    }
    const { gameId } = paramResult.data;
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

    // Enforce the same participant-or-spectator invariant used by
    // game details and move history endpoints.
    assertUserCanViewGame(userId, game);

    // If the WebSocket server is not wired in this process (for example,
    // in certain tests or CLI tools), return a minimal diagnostics view
    // that reflects only the absence of an in-memory session.
    if (!wsServerInstance || typeof wsServerInstance.getGameDiagnosticsForGame !== 'function') {
      return res.json({
        success: true,
        data: {
          sessionStatus: null,
          lastAIRequestState: null,
          aiDiagnostics: null,
          connections: {},
          meta: {
            hasInMemorySession: false,
          },
        },
      });
    }

    const diagnostics = wsServerInstance.getGameDiagnosticsForGame(gameId) as {
      sessionStatus: any | null;
      lastAIRequestState: any | null;
      aiDiagnostics: any | null;
      connections: Record<string, unknown>;
      hasInMemorySession: boolean;
    };

    res.json({
      success: true,
      data: {
        sessionStatus: diagnostics.sessionStatus,
        lastAIRequestState: diagnostics.lastAIRequestState,
        aiDiagnostics: diagnostics.aiDiagnostics,
        connections: diagnostics.connections || {},
        meta: {
          hasInMemorySession: diagnostics.hasInMemorySession,
        },
      },
    });
  })
);

/**
 * @openapi
 * /games/lobby/available:
 *   get:
 *     summary: Get available games to join
 *     description: |
 *       Returns a list of games that are waiting for players.
 *       Excludes games where the authenticated user is already a participant.
 *     tags: [Games]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: boardType
 *         schema:
 *           type: string
 *           enum: [square8, square19, hexagonal]
 *         description: Filter by board type
 *       - in: query
 *         name: maxPlayers
 *         schema:
 *           type: integer
 *           minimum: 2
 *           maximum: 4
 *         description: Filter by max players
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 100
 *           default: 20
 *         description: Maximum results to return
 *     responses:
 *       200:
 *         description: Available games retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     games:
 *                       type: array
 *                       items:
 *                         $ref: '#/components/schemas/Game'
 *       400:
 *         description: Invalid query parameters
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: VALIDATION_INVALID_QUERY_PARAMS
 *                 message: Invalid query parameters
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/lobby/available',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    // Validate query parameters
    const queryResult = GameListingQuerySchema.safeParse(req.query);
    if (!queryResult.success) {
      throw createError('Invalid query parameters', 400, 'INVALID_QUERY_PARAMS');
    }
    const { boardType, maxPlayers, limit } = queryResult.data;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

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
      whereClause.maxPlayers = maxPlayers;
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
      take: limit,
    });

    res.json({
      success: true,
      data: { games },
    });
  })
);

export default router;
