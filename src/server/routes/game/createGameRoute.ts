import crypto from 'crypto';
import { Router, Response } from 'express';
import {
  Prisma,
  GameStatus as PrismaGameStatus,
  BoardType as PrismaBoardType,
} from '@prisma/client';
import { getDatabaseClient } from '../../database/connection';
import { AuthenticatedRequest, getAuthUserId } from '../../middleware/auth';
import { createError, asyncHandler } from '../../middleware/errorHandler';
import { consumeRateLimit } from '../../middleware/rateLimiter';
import { httpLogger, logger } from '../../utils/logger';
import { CreateGameSchema, CreateGameInput } from '../../../shared/validation/schemas';
import { AiOpponentsConfig, GameState } from '../../../shared/types/game';
import { generateGameSeed } from '../../../shared/utils/rng';
import { config } from '../../config';
import type { GameRouteContext } from './routeContext';

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
export function registerCreateGameRoute(router: Router, context: GameRouteContext): void {
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

      const forwardedForHeader = req.headers['x-forwarded-for'] as string | undefined;
      const forwardedFor = forwardedForHeader?.split(',')[0]?.trim();
      const clientIp = forwardedFor || req.ip || 'unknown';

      const userQuota = await consumeRateLimit('gameCreateUser', userId, req);
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

      const ipQuota = await consumeRateLimit('gameCreateIp', clientIp, req);
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

      if (gameData.aiOpponents && gameData.aiOpponents.count > 0) {
        if (gameData.isRated) {
          throw createError('AI games cannot be rated', 400, 'AI_GAMES_UNRATED');
        }

        if (gameData.aiOpponents.difficulty.length < gameData.aiOpponents.count) {
          throw createError(
            'Must provide difficulty for each AI opponent',
            400,
            'INVALID_AI_CONFIG'
          );
        }

        for (const diff of gameData.aiOpponents.difficulty) {
          if (diff < 1 || diff > 10) {
            throw createError('AI difficulty must be between 1 and 10', 400, 'INVALID_DIFFICULTY');
          }
        }
      }

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
            : false
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

      const hasAIOpponents = gameData.aiOpponents && gameData.aiOpponents.count > 0;
      const initialStatus = hasAIOpponents ? PrismaGameStatus.active : PrismaGameStatus.waiting;
      const startedAt = hasAIOpponents ? new Date() : undefined;

      const rngSeed = typeof gameData.seed === 'number' ? gameData.seed : generateGameSeed();

      let inviteCode: string;
      for (let attempt = 0; ; attempt++) {
        inviteCode = crypto.randomBytes(6).toString('base64url').slice(0, 8);
        const existing = await prisma.game.findUnique({
          where: { inviteCode },
          select: { id: true },
        });
        if (!existing) break;
        if (attempt >= 5) {
          throw createError(
            'Failed to generate unique invite code',
            500,
            'INVITE_CODE_GENERATION_FAILED'
          );
        }
      }

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
          inviteCode,
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

      const wsServer = context.getWebSocketServer();
      if (initialStatus === 'waiting' && wsServer) {
        wsServer.broadcastLobbyEvent(
          'lobby:game_created',
          game as unknown as Parameters<
            typeof wsServer.broadcastLobbyEvent<'lobby:game_created'>
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
}
