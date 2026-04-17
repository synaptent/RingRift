import { Router, Response } from 'express';
import { Prisma, GameStatus as PrismaGameStatus } from '@prisma/client';
import { getDatabaseClient } from '../database/connection';
import { AuthenticatedRequest, getAuthUserId } from '../middleware/auth';
import { createError, asyncHandler } from '../middleware/errorHandler';
import { adaptiveRateLimiter, sandboxAiRateLimiter } from '../middleware/rateLimiter';
import { httpLogger, logger } from '../utils/logger';
import { ErrorCodes, ErrorCodeMessages } from '../errors';
import {
  GameIdParamSchema,
  GameListingQuerySchema,
  GameListingQueryInput,
  InviteCodeParamSchema,
  MoveSchema,
  type MoveInput,
} from '../../shared/validation/schemas';
import { validateQuery, validateParams } from '../middleware/validateRequest';
import { GameStatus, GameState } from '../../shared/types/game';
import { GameEngine } from '../game/GameEngine';
import { getDisplayUsername } from './user';
import { config } from '../config';
import { createDecisionPhaseFixtureGame } from '../game/testFixtures/decisionPhaseFixtures';
import { getAIServiceClient, type LadderHealthQuery } from '../services/AIServiceClient';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';
import {
  deserializeGameState,
  type SerializedGameState,
} from '../../shared/engine/contracts/serialization';
import type { WebSocketServer } from '../websocket/server';
import { assertUserCanViewGame } from './game/accessControl';
import type { GameRouteContext } from './game/routeContext';
import { registerCreateGameRoute } from './game/createGameRoute';
import { registerLeaveGameRoute } from './game/leaveGameRoute';
import { registerUserGamesRoute } from './game/userGamesRoute';
const router = Router();
export const sandboxHelperRoutes = Router();

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

// Sandbox helper endpoints are explicitly gated by config.featureFlags.sandboxAi
// and are allowed to be unauthenticated so the /sandbox host can use them
// without requiring a logged-in session. Use dedicated high-limit rate limiter
// since AI games can generate many move requests per minute.
sandboxHelperRoutes.use(sandboxAiRateLimiter);

// Active games storage (in production, this would be in Redis)
const activeGames = new Map<string, GameEngine>();
const routeContext: GameRouteContext = {
  getWebSocketServer: () => wsServerInstance,
};

/**
 * @openapi
 * /games/fixtures/decision-phase:
 *   post:
 *     summary: Create decision-phase fixture game (dev/test only)
 *     description: |
 *       Creates a game in a known decision phase for E2E and diagnostics.
 *       Only available in development/test; returns 404 otherwise.
 *     tags: [Games]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: false
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               scenario:
 *                 type: string
 *                 enum: [line_processing, territory_processing, chain_capture_choice, near_victory_elimination, near_victory_territory]
 *               isRated:
 *                 type: boolean
 *               shortTimeoutMs:
 *                 type: integer
 *               shortWarningBeforeMs:
 *                 type: integer
 *     responses:
 *       201:
 *         description: Fixture game created
 *       400:
 *         description: Invalid fixture payload
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 */
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
// Maximum size for serialized game state in sandbox endpoints (100KB)
const MAX_SERIALIZED_STATE_SIZE = 100 * 1024;

function isSandboxAiEnabled(): boolean {
  return config.isTest || config.isDevelopment || config.featureFlags.sandboxAi.enabled;
}

/**
 * @openapi
 * /games/sandbox/evaluate:
 *   post:
 *     summary: Evaluate sandbox position (dev/test only)
 *     description: |
 *       Evaluates a serialized sandbox GameState via the AI service.
 *       Available only in development/test; returns 404 otherwise.
 *     tags: [Games]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               state:
 *                 type: object
 *             required: [state]
 *     responses:
 *       200:
 *         description: Evaluation result
 *       400:
 *         description: Invalid request payload
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
sandboxHelperRoutes.post(
  '/sandbox/evaluate',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    if (!isSandboxAiEnabled()) {
      throw createError('Not found', 404, 'NOT_FOUND');
    }

    const body = (req.body || {}) as { state?: SerializedGameState };
    if (!body.state) {
      throw createError('Missing serialized sandbox state', 400, 'INVALID_REQUEST');
    }

    // Validate payload size to prevent memory exhaustion
    const stateSize = JSON.stringify(body.state).length;
    if (stateSize > MAX_SERIALIZED_STATE_SIZE) {
      throw createError(
        `Serialized state too large (${stateSize} bytes, max ${MAX_SERIALIZED_STATE_SIZE})`,
        400,
        'PAYLOAD_TOO_LARGE'
      );
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

sandboxHelperRoutes.get(
  '/sandbox/ai/health',
  asyncHandler(async (_req: AuthenticatedRequest, res: Response) => {
    if (!isSandboxAiEnabled()) {
      throw createError('Not found', 404, 'NOT_FOUND');
    }

    const aiClient = getAIServiceClient();
    const healthy = await aiClient.healthCheck();

    res.status(healthy ? 200 : 503).json({
      status: healthy ? 'healthy' : 'degraded',
    });
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
/**
 * @openapi
 * /games/sandbox/ai/move:
 *   post:
 *     summary: Get sandbox AI move (dev/test or feature-flagged)
 *     description: |
 *       Returns an AI move for a serialized sandbox GameState.
 *       Enabled in development/test or when sandbox AI endpoints are explicitly enabled.
 *     tags: [Games]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               state:
 *                 type: object
 *               difficulty:
 *                 type: integer
 *                 minimum: 1
 *                 maximum: 10
 *               playerNumber:
 *                 type: integer
 *             required: [state]
 *     responses:
 *       200:
 *         description: AI move response
 *       400:
 *         description: Invalid request payload
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
sandboxHelperRoutes.post(
  '/sandbox/ai/move',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    if (!isSandboxAiEnabled()) {
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

    // Validate payload size to prevent memory exhaustion
    const stateSize = JSON.stringify(body.state).length;
    if (stateSize > MAX_SERIALIZED_STATE_SIZE) {
      throw createError(
        `Serialized state too large (${stateSize} bytes, max ${MAX_SERIALIZED_STATE_SIZE})`,
        400,
        'PAYLOAD_TOO_LARGE'
      );
    }

    const gameState = deserializeGameState(body.state);
    const playerNumber = body.playerNumber ?? gameState.currentPlayer;

    const difficultyRaw = body.difficulty;
    const difficulty =
      typeof difficultyRaw === 'number' ? Math.max(1, Math.min(10, Math.round(difficultyRaw))) : 5;

    const aiClient = getAIServiceClient();
    try {
      const requestStart = performance.now();
      const response = await aiClient.getAIMove(gameState, playerNumber, difficulty);
      const latencyMs = Math.round(performance.now() - requestStart);
      const modelVersion = response.nn_model_version ?? response.model_version ?? null;
      const modelPath =
        response.nn_model_path ??
        response.model_path ??
        response.nn_checkpoint ??
        response.nnue_checkpoint ??
        response.model_id ??
        null;

      res.status(200).json({
        move: response.move,
        evaluation: response.evaluation,
        thinkingTimeMs: response.thinking_time_ms,
        aiType: response.ai_type,
        difficulty: response.difficulty,
        heuristicProfileId: response.heuristic_profile_id,
        useNeuralNet: response.use_neural_net,
        nnModelId: response.nn_model_id,
        nnCheckpoint: response.nn_checkpoint,
        nnueCheckpoint: response.nnue_checkpoint,
        modelId: response.model_id,
        evalMode: response.eval_mode,
        simulationBudget: response.simulation_budget,
        device: response.device,
        searchStatsSummary: response.search_stats_summary,
        modelVersion,
        modelPath,
        aiTier: difficulty,
        latencyMs,
        fallbackUsed: false,
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
 * Sandbox helper for inspecting the effective AI ladder configuration and
 * artifact availability from the Python AI service.
 *
 * Proxies `/internal/ladder/health` so the sandbox UI can verify that the
 * expected NN/NNUE checkpoints + heuristic profiles are available for the
 * current board type / player count.
 */
/**
 * @openapi
 * /games/sandbox/ai/ladder/health:
 *   get:
 *     summary: Get sandbox AI ladder health (dev/test or feature-flagged)
 *     description: |
 *       Proxies the AI service ladder health for sandbox diagnostics.
 *       Enabled in development/test or when sandbox AI endpoints are explicitly enabled.
 *     tags: [Games]
 *     parameters:
 *       - in: query
 *         name: boardType
 *         schema:
 *           type: string
 *       - in: query
 *         name: numPlayers
 *         schema:
 *           type: integer
 *           minimum: 2
 *           maximum: 4
 *       - in: query
 *         name: difficulty
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 10
 *     responses:
 *       200:
 *         description: Ladder health response
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
sandboxHelperRoutes.get(
  '/sandbox/ai/ladder/health',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    if (!isSandboxAiEnabled()) {
      throw createError('Not found', 404, 'NOT_FOUND');
    }

    const rawBoardType = typeof req.query.boardType === 'string' ? req.query.boardType : undefined;
    const rawNumPlayers =
      typeof req.query.numPlayers === 'string' ? req.query.numPlayers : undefined;
    const rawDifficulty =
      typeof req.query.difficulty === 'string' ? req.query.difficulty : undefined;

    const boardType = rawBoardType;
    const numPlayers =
      rawNumPlayers !== undefined
        ? Math.max(2, Math.min(4, parseInt(rawNumPlayers, 10)))
        : undefined;
    const difficulty =
      rawDifficulty !== undefined
        ? Math.max(1, Math.min(10, parseInt(rawDifficulty, 10)))
        : undefined;

    const aiClient = getAIServiceClient();
    try {
      const query: LadderHealthQuery = {};
      if (boardType) {
        query.boardType = boardType;
      }
      if (typeof numPlayers === 'number' && Number.isFinite(numPlayers)) {
        query.numPlayers = numPlayers;
      }
      if (typeof difficulty === 'number' && Number.isFinite(difficulty)) {
        query.difficulty = difficulty;
      }

      const data = await aiClient.getLadderHealth(query);
      res.status(200).json(data);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'AI Service failed to fetch ladder health';
      logger.warn('Sandbox AI ladder health request failed', {
        boardType,
        numPlayers,
        difficulty,
        error: message,
      });
      res.status(503).json({
        error:
          'Sandbox AI ladder health is unavailable. Ensure the AI service is running and sandbox AI endpoints are enabled.',
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
  validateQuery(GameListingQuerySchema),
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { status, limit, offset } = req.query as unknown as GameListingQueryInput;

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

// ====================================================================
// INVITE CODE ENDPOINTS
// These must be registered before the /:gameId catch-all to avoid
// Express matching "invite" as a gameId parameter.
// ====================================================================

/**
 * @openapi
 * /games/invite/{inviteCode}:
 *   get:
 *     summary: Look up a game by invite code
 *     tags: [Games]
 *     parameters:
 *       - in: path
 *         name: inviteCode
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Game info for the invite code
 *       404:
 *         description: Invite code not found
 */
router.get(
  '/invite/:inviteCode',
  validateParams(InviteCodeParamSchema),
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { inviteCode } = req.params;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const game = await prisma.game.findUnique({
      where: { inviteCode },
      include: {
        player1: { select: { id: true, username: true, rating: true } },
        player2: { select: { id: true, username: true, rating: true } },
        player3: { select: { id: true, username: true, rating: true } },
        player4: { select: { id: true, username: true, rating: true } },
      },
    });

    if (!game) {
      throw createError('Game not found for this invite code', 404, 'GAME_NOT_FOUND');
    }

    const playerCount = [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
      Boolean
    ).length;

    res.json({
      success: true,
      data: {
        game: {
          id: game.id,
          inviteCode: game.inviteCode,
          boardType: game.boardType,
          maxPlayers: game.maxPlayers,
          status: game.status,
          isRated: game.isRated,
          playerCount,
          players: [game.player1, game.player2, game.player3, game.player4].filter(Boolean),
          createdAt: game.createdAt,
        },
      },
    });
  })
);

/**
 * @openapi
 * /games/invite/{inviteCode}/join:
 *   post:
 *     summary: Join a game via invite code
 *     tags: [Games]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: inviteCode
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Joined game successfully
 *       400:
 *         description: Cannot join game
 *       404:
 *         description: Invite code not found
 */
router.post(
  '/invite/:inviteCode/join',
  validateParams(InviteCodeParamSchema),
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { inviteCode } = req.params;
    const userId = getAuthUserId(req);

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const game = await prisma.game.findUnique({
      where: { inviteCode },
      include: {
        player1: true,
        player2: true,
        player3: true,
        player4: true,
      },
    });

    if (!game) {
      throw createError('Game not found for this invite code', 404, 'GAME_NOT_FOUND');
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
      // Already in the game - redirect to it rather than erroring
      res.json({
        success: true,
        data: { game: { id: game.id } },
        message: 'Already in this game',
      });
      return;
    }

    // Find next available player slot
    let playerSlot: string | null = null;
    if (!game.player2Id) playerSlot = 'player2Id';
    else if (!game.player3Id && game.maxPlayers >= 3) playerSlot = 'player3Id';
    else if (!game.player4Id && game.maxPlayers >= 4) playerSlot = 'player4Id';

    if (!playerSlot) {
      throw createError('Game is full', 400, 'GAME_FULL');
    }

    const updatedGame = await prisma.game.update({
      where: { id: game.id },
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

    const currentPlayerCount = [
      updatedGame.player1Id,
      updatedGame.player2Id,
      updatedGame.player3Id,
      updatedGame.player4Id,
    ].filter(Boolean).length;

    if (wsServerInstance) {
      wsServerInstance.broadcastLobbyEvent('lobby:game_joined', {
        gameId: game.id,
        playerCount: currentPlayerCount,
      });
    }

    // Check if game should start via engine
    const gameEngine = activeGames.get(game.id);
    if (gameEngine && currentPlayerCount >= 2) {
      const startedGame = await prisma.game.update({
        where: { id: game.id },
        data: {
          status: PrismaGameStatus.active,
          startedAt: new Date(),
          updatedAt: new Date(),
        },
      });

      if (wsServerInstance) {
        wsServerInstance.broadcastLobbyEvent('lobby:game_started', {
          gameId: game.id,
          status: startedGame.status as GameStatus,
          startedAt: startedGame.startedAt ?? undefined,
          playerCount: currentPlayerCount,
        });
      }
    }

    httpLogger.info(req, 'Player joined game via invite code', {
      gameId: game.id,
      inviteCode,
      userId,
      playerSlot,
    });

    res.json({
      success: true,
      data: { game: updatedGame },
      message: 'Joined game successfully',
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
 *       Rate limited (defaults, configurable via RATE_LIMIT_GAME_CREATE_*):
 *       - 20 games per 10 minutes per user
 *       - 50 games per 10 minutes per IP address
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
registerCreateGameRoute(router, routeContext);

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
  validateParams(GameIdParamSchema),
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { gameId } = req.params;
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
registerLeaveGameRoute(router, routeContext);

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
  validateParams(GameIdParamSchema),
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { gameId } = req.params;
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
/**
 * @openapi
 * /games/{gameId}/moves:
 *   post:
 *     summary: Internal HTTP move harness (feature-flagged)
 *     description: |
 *       Feature-flagged HTTP move submission used for load tests and diagnostics.
 *       Returns 404 when the move harness is disabled.
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
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             oneOf:
 *               - $ref: '#/components/schemas/Move'
 *               - type: object
 *                 properties:
 *                   move:
 *                     $ref: '#/components/schemas/Move'
 *     responses:
 *       200:
 *         description: Move applied successfully
 *       400:
 *         description: Invalid move
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       403:
 *         $ref: '#/components/responses/Forbidden'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 *       504:
 *         description: Move application timed out
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

    // Apply timeout protection to prevent unbounded hangs on lock contention or
    // rules engine deadlocks. Returns 504 Gateway Timeout if exceeded.
    const timeoutMs = config.featureFlags.httpMoveHarness.timeoutMs;
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error('HTTP_MOVE_HARNESS_TIMEOUT')), timeoutMs);
    });

    let rulesResult: import('../game/RulesBackendFacade').RulesResult | undefined;
    try {
      rulesResult = await Promise.race([
        wsServerInstance.handlePlayerMoveFromHttp(gameId, userId, moveInput),
        timeoutPromise,
      ]);
    } catch (error) {
      // Handle timeout specifically
      const message = error instanceof Error ? error.message : String(error);
      if (message === 'HTTP_MOVE_HARNESS_TIMEOUT') {
        const code = ErrorCodes.SERVER_GATEWAY_TIMEOUT;
        logger.warn('HTTP move harness request timed out', {
          gameId,
          userId,
          timeoutMs,
        });
        res.status(504).json({
          success: false,
          error: {
            code,
            message: ErrorCodeMessages[code],
          },
        });
        return;
      }

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
  validateParams(GameIdParamSchema),
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { gameId } = req.params;
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
registerUserGamesRoute(router);

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
  validateParams(GameIdParamSchema),
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { gameId } = req.params;
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

    // Non-admin users get sanitized diagnostics (hide internal AI state details)
    const isAdmin = req.user?.role === 'ADMIN';

    res.json({
      success: true,
      data: {
        sessionStatus: diagnostics.sessionStatus,
        // Only expose detailed AI state to admins to prevent information leakage
        lastAIRequestState: isAdmin ? diagnostics.lastAIRequestState : null,
        aiDiagnostics: isAdmin ? diagnostics.aiDiagnostics : null,
        connections: isAdmin
          ? diagnostics.connections || {}
          : { count: Object.keys(diagnostics.connections || {}).length },
        meta: {
          hasInMemorySession: diagnostics.hasInMemorySession,
          sanitized: !isAdmin,
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
 *           enum: [square8, square19, hex8, hexagonal]
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
  validateQuery(GameListingQuerySchema),
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { boardType, maxPlayers, limit } = req.query as unknown as GameListingQueryInput;

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

/**
 * @openapi
 * /games/matchmaking/stats:
 *   get:
 *     summary: Get matchmaking queue statistics
 *     description: |
 *       Returns current matchmaking queue statistics including queue size,
 *       breakdown by board type, and average wait times. Useful for lobby UI.
 *     tags:
 *       - Games
 *     responses:
 *       200:
 *         description: Matchmaking statistics
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 data:
 *                   type: object
 *                   properties:
 *                     queueSize:
 *                       type: number
 *                     byBoardType:
 *                       type: object
 *                     avgWaitTimeMs:
 *                       type: number
 */
router.get(
  '/matchmaking/stats',
  asyncHandler(async (_req, res: Response) => {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 503, ErrorCodes.SERVER_INTERNAL_ERROR);
    }

    // Get queue statistics from database
    const searching = await prisma.matchmakingQueue.findMany({
      where: { status: 'searching' },
      select: {
        boardType: true,
        joinedAt: true,
      },
    });

    const now = Date.now();
    const byBoardType: Record<string, number> = {};
    let totalWaitTime = 0;

    for (const entry of searching) {
      byBoardType[entry.boardType] = (byBoardType[entry.boardType] || 0) + 1;
      totalWaitTime += now - entry.joinedAt.getTime();
    }

    // Get recent match metrics for average wait time (last 24 hours)
    const recentMatches = await prisma.matchmakingMetrics.aggregate({
      where: {
        outcome: 'matched',
        createdAt: { gte: new Date(now - 24 * 60 * 60 * 1000) },
      },
      _avg: { waitTimeMs: true },
      _count: true,
    });

    res.json({
      success: true,
      data: {
        queueSize: searching.length,
        byBoardType,
        avgWaitTimeMs: searching.length > 0 ? Math.round(totalWaitTime / searching.length) : 0,
        recentStats: {
          avgWaitTimeMs: Math.round(recentMatches._avg.waitTimeMs || 0),
          matchCount24h: recentMatches._count,
        },
      },
    });
  })
);

export default router;
export { registerCreateGameRoute } from './game/createGameRoute';
export { registerLeaveGameRoute } from './game/leaveGameRoute';
export { registerUserGamesRoute } from './game/userGamesRoute';
