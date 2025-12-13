import { Router, Response } from 'express';
import {
  Prisma,
  GameStatus as PrismaGameStatus,
  BoardType as PrismaBoardType,
} from '@prisma/client';
import { getDatabaseClient } from '../database/connection';
import { AuthenticatedRequest, getAuthUserId } from '../middleware/auth';
import { createError, asyncHandler } from '../middleware/errorHandler';
import { consumeRateLimit, adaptiveRateLimiter } from '../middleware/rateLimiter';
import { httpLogger, logger } from '../utils/logger';
import { ErrorCodes, ErrorCodeMessages } from '../errors';
import {
  CreateGameSchema,
  CreateGameInput,
  GameIdParamSchema,
  GameListingQuerySchema,
  MoveSchema,
  type MoveInput,
} from '../../shared/validation/schemas';
import { RatingService, RatingUpdateResult } from '../services/RatingService';
import { AiOpponentsConfig, BoardType, GameStatus, GameState } from '../../shared/types/game';
import { generateGameSeed } from '../../shared/utils/rng';
import { GameEngine } from '../game/GameEngine';
import { getDisplayUsername } from './user';
import { config } from '../config';
import { createDecisionPhaseFixtureGame } from '../game/testFixtures/decisionPhaseFixtures';
import { getAIServiceClient } from '../services/AIServiceClient';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';
import {
  deserializeGameState,
  type SerializedGameState,
} from '../../shared/engine/contracts/serialization';
import type { WebSocketServer } from '../websocket/server';
const router = Router();

// WebSocket server instance will be injected
let wsServerInstance: WebSocketServer | null = null;

export function setWebSocketServer(wsServer: WebSocketServer | null) {
  wsServerInstance = wsServer;
}

// Apply adaptive rate limiting to game routes.
// For load testing and to avoid double-limiting game creation, use the
// authenticated API limiter instead of the dedicated "game" limiter here.
// Game creation still has its own per-user and per-IP quotas via
// gameCreateUser/gameCreateIp in this module.
router.use(adaptiveRateLimiter('apiAuthenticated', 'api'));

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
 * Test/dev-only fixture endpoint for creating games that start in a
 * known decision phase (currently line_processing). This is primarily
 * used by Playwright E2E scenarios that need to exercise decision-phase
 * timeout and reconnect behaviour without driving a full game to that
 * state through the UI.
 *
 * The route is deliberately guarded so it is not exposed in production.
 */
router.post(
  '/fixtures/decision-phase',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    if (!config.isTest && !config.isDevelopment) {
      throw createError('Not found', 404, 'NOT_FOUND');
    }

    const body = (req.body || {}) as {
      scenario?:
        | 'line_processing'
        | 'territory_processing'
        | 'chain_capture_choice'
        | 'near_victory_elimination'
        | 'near_victory_territory';
      isRated?: boolean;
      /** Optional short timeout for E2E testing (milliseconds) */
      shortTimeoutMs?: number;
      /** Optional short warning time (milliseconds before timeout) */
      shortWarningBeforeMs?: number;
    };

    const scenario = body.scenario ?? 'line_processing';
    const validScenarios = [
      'line_processing',
      'territory_processing',
      'chain_capture_choice',
      'near_victory_elimination',
      'near_victory_territory',
    ];
    if (!validScenarios.includes(scenario)) {
      throw createError('Unsupported decision-phase fixture scenario', 400, 'INVALID_FIXTURE');
    }

    const isRated = body.isRated ?? true;

    // Validate timeout overrides if provided (only allow in test/dev)
    const shortTimeoutMs = body.shortTimeoutMs;
    const shortWarningBeforeMs = body.shortWarningBeforeMs;
    if (shortTimeoutMs !== undefined && (shortTimeoutMs < 1000 || shortTimeoutMs > 60000)) {
      throw createError('shortTimeoutMs must be between 1000 and 60000', 400, 'INVALID_TIMEOUT');
    }
    if (
      shortWarningBeforeMs !== undefined &&
      (shortWarningBeforeMs < 500 || shortWarningBeforeMs > 30000)
    ) {
      throw createError(
        'shortWarningBeforeMs must be between 500 and 30000',
        400,
        'INVALID_TIMEOUT'
      );
    }

    const gameId = await createDecisionPhaseFixtureGame({
      creatorUserId: getAuthUserId(req),
      scenario,
      isRated,
      ...(shortTimeoutMs !== undefined && { shortTimeoutMs }),
      ...(shortWarningBeforeMs !== undefined && { shortWarningBeforeMs }),
    });

    res.status(201).json({
      success: true,
      data: {
        gameId,
        scenario,
      },
    });
  })
);

/**
 * Test/dev-only helper for evaluating arbitrary sandbox positions via the
 * Python AI service. This endpoint accepts a serialized GameState (using the
 * same wire format as scenario persistence) and returns a single
 * PositionEvaluationPayload['data'] object suitable for feeding into the
 * client-side EvaluationPanel.
 *
 * Guarded to test/development environments to avoid exposing raw evaluation
 * of arbitrary positions in production.
 */
router.post(
  '/sandbox/evaluate',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    if (!config.isTest && !config.isDevelopment) {
      throw createError('Not found', 404, 'NOT_FOUND');
    }

    const body = (req.body || {}) as { state?: SerializedGameState };
    if (!body.state) {
      throw createError('Missing serialized sandbox state', 400, 'INVALID_REQUEST');
    }

    const gameState = deserializeGameState(body.state);
    const aiClient = getAIServiceClient();
    try {
      const response = await aiClient.evaluatePositionMulti(gameState);

      const data: PositionEvaluationPayload['data'] = {
        gameId: gameState.id,
        moveNumber: response.move_number,
        boardType: (response.board_type as GameState['boardType']) ?? gameState.boardType,
        perPlayer: response.per_player,
        engineProfile: response.engine_profile,
        evaluationScale: response.evaluation_scale,
      };

      res.status(200).json(data);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'AI Service failed to evaluate sandbox position';
      logger.warn('Sandbox position evaluation failed', {
        gameId: gameState.id,
        error: message,
      });
      res.status(503).json({
        error:
          'Sandbox AI evaluation is unavailable. Ensure the AI service is running and analysis mode is enabled.',
        details: message,
      });
    }
  })
);

/**
 * Sandbox helper for requesting an AI move via the Python AI service.
 *
 * This endpoint accepts a serialized sandbox GameState (see
 * src/shared/engine/contracts/serialization.ts) plus a numeric difficulty,
 * then returns the selected Move. It is primarily used by the client-side
 * /sandbox host so that local sandbox games can use the same canonical
 * difficulty ladder (minimax/mcts/descent + neural variants) as backend games.
 */
router.post(
  '/sandbox/ai/move',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const enabled = config.isTest || config.isDevelopment || config.featureFlags.sandboxAi.enabled;

    if (!enabled) {
      throw createError('Not found', 404, 'NOT_FOUND');
    }

    const body = (req.body || {}) as {
      state?: SerializedGameState;
      difficulty?: number;
      playerNumber?: number;
    };

    if (!body.state) {
      throw createError('Missing serialized sandbox state', 400, 'INVALID_REQUEST');
    }

    const gameState = deserializeGameState(body.state);
    const playerNumber = body.playerNumber ?? gameState.currentPlayer;

    const difficultyRaw = body.difficulty;
    const difficulty =
      typeof difficultyRaw === 'number' ? Math.max(1, Math.min(10, Math.round(difficultyRaw))) : 5;

    const aiClient = getAIServiceClient();
    try {
      const response = await aiClient.getAIMove(gameState, playerNumber, difficulty);

      res.status(200).json({
        move: response.move,
        evaluation: response.evaluation,
        thinkingTimeMs: response.thinking_time_ms,
        aiType: response.ai_type,
        difficulty: response.difficulty,
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'AI Service failed to generate sandbox move';
      logger.warn('Sandbox AI move request failed', {
        gameId: gameState.id,
        playerNumber,
        difficulty,
        error: message,
      });
      res.status(503).json({
        error:
          'Sandbox AI move is unavailable. Ensure the AI service is running and sandbox AI endpoints are enabled.',
        details: message,
      });
    }
  })
);

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

    const userId = getAuthUserId(req);

    const whereClause: Prisma.GameWhereInput = {
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

    // Project a lightweight terminal result reason for completed/abandoned games
    // so profile/recent-games views can distinguish resignation/abandonment/timeout
    // without requiring an additional history/details call.
    const serializedGames = games.map((game) => {
      let resultReason: string | undefined;

      if (
        game.status === PrismaGameStatus.completed ||
        game.status === PrismaGameStatus.abandoned ||
        (game.status as string) === 'finished'
      ) {
        const finalState = game.finalState as Prisma.JsonObject | null | undefined;
        const gameResult = (finalState?.gameResult ?? null) as { reason?: string } | null;
        if (gameResult && typeof gameResult.reason === 'string') {
          resultReason = gameResult.reason;
        }
      }

      return {
        ...game,
        ...(resultReason && { resultReason }),
      };
    });

    const total = await prisma.game.count({ where: whereClause });

    res.json({
      success: true,
      data: {
        games: serializedGames,
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
    // Validate gameId parameter using a lightweight, format-tolerant check
    // that accepts both legacy UUIDs and the CUID values generated for the
    // Game model. Any non-empty, reasonably sized string is treated as a
    // candidate ID and then resolved via the database so that:
    //   - 400 is reserved for truly malformed/empty IDs
    //   - 404 is used for well-formed but unknown/expired IDs
    const rawGameId = req.params.gameId;

    if (typeof rawGameId !== 'string' || rawGameId.trim().length < 3 || rawGameId.length > 64) {
      throw createError('Invalid game ID format', 400, 'INVALID_GAME_ID');
    }

    const gameId = rawGameId.trim();
    const userId = getAuthUserId(req);

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
    // Temporary debug logging for load-test investigation: inspect raw request body shape.
    logger.warn('create-game debug: incoming request body snapshot', {
      path: req.path,
      contentType: req.headers['content-type'],
      bodyType: typeof req.body,
      bodyKeys:
        req.body && typeof req.body === 'object'
          ? Object.keys(req.body as Record<string, unknown>)
          : null,
      timeControlType:
        req.body && typeof req.body === 'object'
          ? typeof (req.body as Record<string, unknown>).timeControl
          : null,
      hasTimeControl: !!(
        req.body &&
        typeof req.body === 'object' &&
        (req.body as Record<string, unknown>).timeControl !== undefined
      ),
      timeControlValue:
        req.body && typeof req.body === 'object'
          ? ((req.body as Record<string, unknown>).timeControl ?? null)
          : null,
    });

    const gameData: CreateGameInput = CreateGameSchema.parse(req.body);
    const userId = getAuthUserId(req);

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

    // Derive initial engine-side state including AI opponent configuration and
    // per-game rules options (e.g., swap rule configuration). This object is
    // persisted as JSON in the Game row and later used by the GameEngine host
    // to construct a full GameState; it may safely include extra metadata
    // fields that are not part of the canonical GameState type.
    const initialGameState: {
      aiOpponents?: AiOpponentsConfig;
      rulesOptions?: GameState['rulesOptions'];
      calibration?: {
        isCalibrationGame: boolean;
        difficulty?: number;
      };
    } = {};
    if (gameData.aiOpponents && gameData.aiOpponents.count > 0) {
      initialGameState.aiOpponents = gameData.aiOpponents;
    }

    // Compute effective rulesOptions for this game.
    //
    // Canonical defaults:
    // - swapRuleEnabled defaults to true for 2-player games only.
    //
    // Experimental overrides (non-canonical; used for research / ablations):
    // - rulesOptions.ringsPerPlayer
    // - rulesOptions.lpsRoundsRequired
    //
    // Guardrail: production environments must not accept non-canonical overrides.
    const requestedRulesOptions = gameData.rulesOptions;
    const requestedRingsPerPlayer = requestedRulesOptions?.ringsPerPlayer;
    const requestedLpsRoundsRequired = requestedRulesOptions?.lpsRoundsRequired;
    if (
      config.isProduction &&
      (requestedRingsPerPlayer !== undefined || requestedLpsRoundsRequired !== undefined)
    ) {
      throw createError(
        'Experimental rulesOptions overrides are not permitted in production.',
        400,
        'INVALID_RULES_OPTIONS'
      );
    }

    const swapRuleEnabled =
      gameData.maxPlayers === 2
        ? typeof requestedRulesOptions?.swapRuleEnabled === 'boolean'
          ? requestedRulesOptions.swapRuleEnabled
          : true
        : undefined;
    const ringsPerPlayer =
      typeof requestedRingsPerPlayer === 'number' ? requestedRingsPerPlayer : undefined;
    const lpsRoundsRequired =
      typeof requestedLpsRoundsRequired === 'number' ? requestedLpsRoundsRequired : undefined;

    let effectiveRulesOptions: GameState['rulesOptions'] | undefined;
    if (
      swapRuleEnabled !== undefined ||
      ringsPerPlayer !== undefined ||
      lpsRoundsRequired !== undefined
    ) {
      effectiveRulesOptions = {
        ...(swapRuleEnabled !== undefined ? { swapRuleEnabled } : {}),
        ...(ringsPerPlayer !== undefined ? { ringsPerPlayer } : {}),
        ...(lpsRoundsRequired !== undefined ? { lpsRoundsRequired } : {}),
      };
    }

    if (effectiveRulesOptions) {
      initialGameState.rulesOptions = effectiveRulesOptions;
    }

    // Determine initial game status: AI games start immediately, human-only games wait
    const hasAIOpponents = gameData.aiOpponents && gameData.aiOpponents.count > 0;
    const initialStatus = hasAIOpponents ? PrismaGameStatus.active : PrismaGameStatus.waiting;
    const startedAt = hasAIOpponents ? new Date() : undefined;

    // Compute the canonical per-game RNG seed. When the client supplies an
    // explicit seed (used by parity tooling and diagnostics), honour it;
    // otherwise generate a fresh seed and persist it so backend hosts and
    // the Python AI service share a stable per-game RNG root.
    const rngSeed = typeof gameData.seed === 'number' ? gameData.seed : generateGameSeed();

    // Create game in database
    const game = await prisma.game.create({
      data: {
        boardType: gameData.boardType as PrismaBoardType,
        maxPlayers: gameData.maxPlayers,
        timeControl: gameData.timeControl,
        isRated: gameData.isRated,
        allowSpectators: !gameData.isPrivate,
        player1Id: userId,
        status: initialStatus,
        gameState: initialGameState as Prisma.InputJsonValue,
        // Store the per-game RNG seed explicitly; avoid undefined to satisfy
        // Prisma's `number | null` type.
        rngSeed,
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
      // Type assertion: Prisma Game with JsonValue fields is runtime-compatible
      // with shared Game type that expects specific object shapes
      wsServerInstance.broadcastLobbyEvent(
        'lobby:game_created',
        game as unknown as Parameters<
          typeof wsServerInstance.broadcastLobbyEvent<'lobby:game_created'>
        >[1]
      );
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
    const userId = getAuthUserId(req);
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

    if (game.status !== PrismaGameStatus.waiting) {
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
            status: PrismaGameStatus.active,
            startedAt: new Date(),
            updatedAt: new Date(),
          },
        });

        // Broadcast game started event to remove from lobby and provide
        // basic metadata for lobby consumers.
        if (wsServerInstance) {
          wsServerInstance.broadcastLobbyEvent('lobby:game_started', {
            gameId,
            status: startedGame.status as GameStatus,
            startedAt: startedGame.startedAt ?? undefined,
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
    const userId = getAuthUserId(req);

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

    if (game.status === PrismaGameStatus.active) {
      // If game is active, this is a resignation.
      // Prefer routing through the WebSocket/GameSession host so that the
      // engine produces a canonical GameResult and ratings are updated via
      // GamePersistenceService.finishGame for rated games. Fallback behaviour
      // (when no WebSocketServer is attached) preserves the previous simple
      // DB-only completion semantics.
      let handledViaSession = false;
      if (wsServerInstance && typeof wsServerInstance.handlePlayerResignFromHttp === 'function') {
        try {
          await wsServerInstance.handlePlayerResignFromHttp(gameId, userId);
          handledViaSession = true;
        } catch (err) {
          logger.error('Failed to route resignation through GameSession', {
            gameId,
            userId,
            error: err instanceof Error ? err.message : String(err),
          });
        }
      }

      let winnerId: string | null = null;
      let ratingUpdates: RatingUpdateResult[] | undefined;

      if (!handledViaSession) {
        // Fallback: determine winner and rating updates directly from the
        // database record, preserving legacy behaviour for environments
        // without an attached WebSocketServer.
        const playerIds = [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
          (id): id is string => id !== null
        );

        const remainingPlayerIds = playerIds.filter((id) => id !== userId);
        winnerId =
          remainingPlayerIds.length === 1 ? remainingPlayerIds[0] : (remainingPlayerIds[0] ?? null);

        if (game.isRated && winnerId) {
          try {
            ratingUpdates = await RatingService.processGameResult(gameId, winnerId, playerIds);
            logger.info('Rating updates applied for resignation', {
              gameId,
              resigningPlayer: userId,
              winnerId,
              ratingUpdates: ratingUpdates.map((r) => ({
                playerId: r.playerId,
                change: r.change,
              })),
            });
          } catch (err) {
            logger.error('Failed to process ratings for resignation', {
              gameId,
              error: err instanceof Error ? err.message : String(err),
            });
          }
        }

        await prisma.game.update({
          where: { id: gameId },
          data: {
            status: PrismaGameStatus.completed,
            winnerId: winnerId,
            endedAt: new Date(),
            updatedAt: new Date(),
          },
        });
      } else {
        // When handled via GameSession/GameEngine, fetch the winnerId from
        // the updated Game row for logging/response only; rating updates are
        // applied by GamePersistenceService.finishGame.
        const updated = await prisma.game.findUnique({
          where: { id: gameId },
          select: { winnerId: true },
        });
        winnerId = updated?.winnerId ?? null;
      }

      // Broadcast game ended/cancelled to lobby
      if (wsServerInstance) {
        wsServerInstance.broadcastLobbyEvent('lobby:game_cancelled', { gameId });
      }

      httpLogger.info(req, 'Player resigned from game', {
        gameId,
        userId,
        winnerId,
        ratingChangesApplied: !!ratingUpdates && ratingUpdates.length > 0,
      });

      res.json({
        success: true,
        message: 'Resigned from game',
        data: {
          winnerId,
          ratingChanges: ratingUpdates && ratingUpdates.length > 0 ? ratingUpdates : undefined,
        },
      });
    } else {
      // If game is waiting, remove player
      const updateData: Prisma.GameUpdateInput = { updatedAt: new Date() };

      if (game.player1Id === userId) updateData.player1 = { disconnect: true };
      else if (game.player2Id === userId) updateData.player2 = { disconnect: true };
      else if (game.player3Id === userId) updateData.player3 = { disconnect: true };
      else if (game.player4Id === userId) updateData.player4 = { disconnect: true };

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
          data: { status: PrismaGameStatus.abandoned, endedAt: new Date(), updatedAt: new Date() },
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
    const userId = getAuthUserId(req);

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
        status: true,
        finalState: true,
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
 * INTERNAL: HTTP move harness endpoint for load tests and diagnostics.
 *
 * This route is a thin adapter over the canonical GameSession /
 * RulesBackendFacade pipeline used by WebSocket moves. It is guarded
 * by the ENABLE_HTTP_MOVE_HARNESS feature flag and is not intended
 * as a public client API.
 */
router.post(
  '/:gameId/moves',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    // Feature flag guard - behave as if the route does not exist when
    // the harness is disabled, so production can keep the surface dark.
    if (!config.featureFlags.httpMoveHarness.enabled) {
      throw createError('Route not found', 404, 'RESOURCE_ROUTE_NOT_FOUND');
    }

    if (!wsServerInstance || typeof wsServerInstance.handlePlayerMoveFromHttp !== 'function') {
      throw createError('Service temporarily unavailable', 503, 'SERVER_SERVICE_UNAVAILABLE');
    }

    // Validate gameId parameter
    const paramResult = GameIdParamSchema.safeParse(req.params);
    if (!paramResult.success) {
      throw createError('Invalid game ID format', 400, 'INVALID_GAME_ID');
    }
    const { gameId } = paramResult.data;
    const userId = getAuthUserId(req);

    // Validate move payload using the same wire-level MoveSchema used
    // by WebSocket player_move. Support both a bare Move payload and a
    // wrapped { move } object so that internal harnesses and the public
    // gameApi.makeMove client helper can share this endpoint.
    const rawBody = req.body as unknown;
    const candidateMove =
      rawBody && typeof rawBody === 'object'
        ? ((rawBody as Record<string, unknown>).move ?? rawBody)
        : rawBody;

    const moveResult = MoveSchema.safeParse(candidateMove);
    if (!moveResult.success) {
      const code = ErrorCodes.GAME_INVALID_MOVE;
      res.status(400).json({
        success: false,
        error: {
          code,
          message: ErrorCodeMessages[code],
        },
      });
      return;
    }
    const moveInput: MoveInput = moveResult.data;

    let rulesResult: import('../game/RulesBackendFacade').RulesResult | undefined;
    try {
      rulesResult = await wsServerInstance.handlePlayerMoveFromHttp(gameId, userId, moveInput);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);

      if (message === 'Database not available') {
        const code = ErrorCodes.SERVER_DATABASE_UNAVAILABLE;
        res.status(503).json({
          success: false,
          error: {
            code,
            message: ErrorCodeMessages[code],
          },
        });
        return;
      }

      if (message === 'Game not found') {
        const code = ErrorCodes.GAME_NOT_FOUND;
        res.status(404).json({
          success: false,
          error: {
            code,
            message: ErrorCodeMessages[code],
          },
        });
        return;
      }

      if (message === 'Game is not active') {
        const code = ErrorCodes.GAME_ALREADY_ENDED;
        res.status(400).json({
          success: false,
          error: {
            code,
            message: ErrorCodeMessages[code],
          },
        });
        return;
      }

      if (
        message === 'Spectators cannot make moves' ||
        message === 'Current user is not a player in this game' ||
        message === 'Current socket user is not a player in this game'
      ) {
        const code = ErrorCodes.RESOURCE_ACCESS_DENIED;
        res.status(403).json({
          success: false,
          error: {
            code,
            message: ErrorCodeMessages[code],
          },
        });
        return;
      }

      if (message.toLowerCase().includes('not your turn')) {
        const code = ErrorCodes.GAME_NOT_YOUR_TURN;
        res.status(400).json({
          success: false,
          error: {
            code,
            message: ErrorCodeMessages[code],
          },
        });
        return;
      }

      if (
        message === 'Invalid move position payload' ||
        message === 'Move destination is required'
      ) {
        const code = ErrorCodes.GAME_INVALID_MOVE;
        res.status(400).json({
          success: false,
          error: {
            code,
            message: ErrorCodeMessages[code],
          },
        });
        return;
      }

      // For all other domain-level rejections surfaced as exceptions, treat
      // them as illegal moves rather than generic server errors so that the
      // HTTP harness mirrors the WebSocket MOVE_REJECTED semantics.
      logger.warn('Engine rejected move via HTTP harness (exception path)', {
        gameId,
        userId,
        error: message,
      });
      const code = ErrorCodes.GAME_INVALID_MOVE;
      res.status(400).json({
        success: false,
        error: {
          code,
          message: 'Move was not valid in the current game state',
        },
      });
      return;
    }

    // Defensive: handle an explicit non-success RulesResult if the host path
    // ever returns one instead of throwing.
    if (!rulesResult || !rulesResult.success) {
      logger.warn('Engine rejected move via HTTP harness (result path)', {
        gameId,
        userId,
        reason: rulesResult?.error,
      });
      const code = ErrorCodes.GAME_INVALID_MOVE;
      res.status(400).json({
        success: false,
        error: {
          code,
          message: rulesResult?.error || 'Move was not valid in the current game state',
        },
      });
      return;
    }

    res.status(200).json({
      success: true,
      data: {
        gameId,
        gameState: rulesResult.gameState,
        gameResult: rulesResult.gameResult ?? null,
      },
    });
  })
);

/**
 * @openapi
 * /games/{gameId}/history:
 *   get:
 *     summary: Get move history for a game
 *     description: |
 *       Returns the complete move history for a specific game in a structured format.
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
 *         description: Move history retrieved successfully
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
 *                     gameId:
 *                       type: string
 *                     moves:
 *                       type: array
 *                       items:
 *                         type: object
 *                         properties:
 *                           moveNumber:
 *                             type: integer
 *                           playerId:
 *                             type: string
 *                           moveType:
 *                             type: string
 *                           moveData:
 *                             type: object
 *                           timestamp:
 *                             type: string
 *                             format: date-time
 *                     totalMoves:
 *                       type: integer
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
  '/:gameId/history',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    // Validate gameId parameter
    const paramResult = GameIdParamSchema.safeParse(req.params);
    if (!paramResult.success) {
      throw createError('Invalid game ID format', 400, 'INVALID_GAME_ID');
    }
    const { gameId } = paramResult.data;
    const userId = getAuthUserId(req);

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // First check game exists and user has access
    const game = await prisma.game.findUnique({
      where: { id: gameId },
      select: {
        id: true,
        player1Id: true,
        player2Id: true,
        player3Id: true,
        player4Id: true,
        allowSpectators: true,
        status: true,
        finalState: true,
      },
    });

    if (!game) {
      throw createError('Game not found', 404, 'GAME_NOT_FOUND');
    }

    // Enforce game-level authorization
    assertUserCanViewGame(userId, game);

    // Get move history using the persistence service
    const moves = await prisma.move.findMany({
      where: { gameId },
      orderBy: { moveNumber: 'asc' },
      include: {
        player: { select: { id: true, username: true } },
      },
    });

    // Format response according to the API specification
    // Player names are transformed to show "Deleted Player" for anonymized users
    const formattedMoves = moves.map((move) => {
      const moveData = move.moveData as Prisma.JsonObject | null;
      const rawAutoResolved = moveData?.decisionAutoResolved as Prisma.JsonObject | undefined;

      // Project any persisted decisionAutoResolved metadata into a compact
      // autoResolved badge payload for the history API. This intentionally
      // mirrors a subset of DecisionAutoResolvedMeta so that the client can
      // render lightweight badges without depending on WebSocket types.
      const autoResolved = rawAutoResolved
        ? {
            reason: rawAutoResolved.reason as 'timeout' | 'disconnected' | 'fallback',
            choiceKind: rawAutoResolved.choiceKind as string | undefined,
            choiceType: rawAutoResolved.choiceType as string | undefined,
          }
        : undefined;

      return {
        moveNumber: move.moveNumber,
        playerId: move.playerId,
        playerName: getDisplayUsername(move.player.username),
        moveType: move.moveType,
        moveData: moveData || {},
        timestamp: move.timestamp.toISOString(),
        ...(autoResolved && { autoResolved }),
      };
    });

    // When a final GameState snapshot is available for a finished game, surface
    // the terminal GameResult.reason (and optional winner) so that history
    // consumers can distinguish timeout, resignation, abandonment, and other
    // victory conditions without making a separate details request.
    let result: { reason: string; winner?: number | null } | undefined;
    if (game.status === 'completed' || game.status === 'abandoned' || game.status === 'finished') {
      const finalState = game.finalState as Prisma.JsonObject | null | undefined;
      const gameResult = (finalState?.gameResult ?? null) as {
        reason?: string;
        winner?: number | null;
      } | null;
      if (gameResult && typeof gameResult.reason === 'string') {
        // Use spread to conditionally add winner only when valid (exactOptionalPropertyTypes)
        result = {
          reason: gameResult.reason,
          ...(typeof gameResult.winner === 'number' || gameResult.winner === null
            ? { winner: gameResult.winner }
            : {}),
        };
      }
    }

    res.json({
      success: true,
      data: {
        gameId,
        moves: formattedMoves,
        totalMoves: moves.length,
        ...(result && { result }),
      },
    });
  })
);

/**
 * @openapi
 * /games/user/{userId}:
 *   get:
 *     summary: Get games for a specific user
 *     description: |
 *       Returns a list of games that a specific user has participated in.
 *       Results are paginated and sorted by creation date (newest first).
 *     tags: [Games]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: userId
 *         required: true
 *         schema:
 *           type: string
 *         description: User ID
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 100
 *           default: 10
 *         description: Number of results to return
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           minimum: 0
 *           default: 0
 *         description: Pagination offset
 *       - in: query
 *         name: status
 *         schema:
 *           type: string
 *           enum: [waiting, active, completed, abandoned, paused]
 *         description: Filter by game status
 *     responses:
 *       200:
 *         description: User games retrieved successfully
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
 *                         type: object
 *                         properties:
 *                           id:
 *                             type: string
 *                           boardType:
 *                             type: string
 *                           status:
 *                             type: string
 *                           playerCount:
 *                             type: integer
 *                           maxPlayers:
 *                             type: integer
 *                           winnerId:
 *                             type: string
 *                             nullable: true
 *                           createdAt:
 *                             type: string
 *                             format: date-time
 *                           endedAt:
 *                             type: string
 *                             format: date-time
 *                             nullable: true
 *                           moveCount:
 *                             type: integer
 *                     pagination:
 *                       type: object
 *                       properties:
 *                         total:
 *                           type: integer
 *                         limit:
 *                           type: integer
 *                         offset:
 *                           type: integer
 *                         hasMore:
 *                           type: boolean
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/user/:userId',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { userId } = req.params;

    // Parse query parameters
    const limit = Math.min(Math.max(parseInt(req.query.limit as string) || 10, 1), 100);
    const offset = Math.max(parseInt(req.query.offset as string) || 0, 0);
    const status = req.query.status as string | undefined;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Build where clause for user's games
    const whereClause: Prisma.GameWhereInput = {
      OR: [
        { player1Id: userId },
        { player2Id: userId },
        { player3Id: userId },
        { player4Id: userId },
      ],
    };

    if (status) {
      // Status from query param maps to Prisma GameStatus enum
      // Cast through Prisma's enum type (excludes undefined from the assigned value)
      (whereClause as { status?: string }).status = status;
    }

    // Get games with move count
    const games = await prisma.game.findMany({
      where: whereClause,
      orderBy: { createdAt: 'desc' },
      take: limit,
      skip: offset,
      include: {
        _count: {
          select: { moves: true },
        },
        winner: { select: { id: true, username: true } },
      },
    });

    const total = await prisma.game.count({ where: whereClause });

    // Format response:
    // - Winner name is transformed to show "Deleted Player" for anonymized users.
    // - recordMetadata.source is surfaced as `source` so clients can distinguish
    //   online games from imported self-play/training records.
    // - A lightweight terminal result reason is projected when available so
    //   profile/replay views can show "Timeout", "Resignation", etc. without
    //   requiring a separate history/details round-trip.
    const formattedGames = games.map((game) => {
      const participantIds = [
        game.player1Id,
        game.player2Id,
        game.player3Id,
        game.player4Id,
      ].filter(Boolean);
      const playerCount = participantIds.length;

      // Project a lightweight terminal result reason for completed/abandoned games,
      // mirroring the main /games listing behaviour.
      let resultReason: string | undefined;
      // Access Prisma record fields using typed approach
      const gameRecord = game as typeof game & {
        finalState?: Prisma.JsonObject | null;
        recordMetadata?: (Prisma.JsonObject & { source?: string }) | null;
        outcome?: string | null;
        isRated?: boolean;
      };
      if (
        game.status === PrismaGameStatus.completed ||
        game.status === PrismaGameStatus.abandoned ||
        (game.status as string) === 'finished'
      ) {
        const finalState = gameRecord.finalState;
        const gameResult = (finalState?.gameResult ?? null) as { reason?: string } | null;
        if (gameResult && typeof gameResult.reason === 'string') {
          resultReason = gameResult.reason;
        }
      }

      // Surface recordMetadata.source when available so callers can distinguish
      // online games from imported self-play games and other record sources.
      const recordMetadata = gameRecord.recordMetadata;
      const source =
        recordMetadata && typeof recordMetadata.source === 'string'
          ? recordMetadata.source
          : 'online_game';

      // Where available, prefer the canonical outcome column; otherwise fall back
      // to the projected resultReason derived from finalState.gameResult.
      const rawOutcome = gameRecord.outcome;
      const outcome = typeof rawOutcome === 'string' ? rawOutcome : resultReason;

      return {
        id: game.id,
        boardType: game.boardType as BoardType,
        status: game.status as GameStatus,
        // playerCount is preserved for backwards compatibility; numPlayers is a
        // more explicit alias used by the replay browser.
        playerCount,
        numPlayers: playerCount,
        maxPlayers: game.maxPlayers,
        winnerId: game.winnerId,
        winnerName: game.winner ? getDisplayUsername(game.winner.username) : null,
        createdAt: game.createdAt.toISOString(),
        endedAt: game.endedAt?.toISOString() || null,
        moveCount: game._count.moves,
        // New fields for replay/profile consumers
        isRated: gameRecord.isRated === true,
        source,
        ...(outcome && { outcome }),
        ...(resultReason && { resultReason }),
      };
    });

    res.json({
      success: true,
      data: {
        games: formattedGames,
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
    const userId = getAuthUserId(req);

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
      res.json({
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
      return;
    }

    const diagnostics = wsServerInstance.getGameDiagnosticsForGame(gameId);

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

    const userId = getAuthUserId(req);

    const whereClause: Prisma.GameWhereInput = {
      status: 'waiting',
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
