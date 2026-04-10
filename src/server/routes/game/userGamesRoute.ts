import { Router, Response } from 'express';
import { Prisma, GameStatus as PrismaGameStatus } from '@prisma/client';
import { getDatabaseClient, withQueryTimeoutStrict } from '../../database/connection';
import { AuthenticatedRequest } from '../../middleware/auth';
import { createError, asyncHandler } from '../../middleware/errorHandler';
import { ErrorCodes } from '../../errors';
import { UUIDSchema } from '../../../shared/validation/schemas';
import { BoardType, GameStatus } from '../../../shared/types/game';
import { getDisplayUsername } from '../user';

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
export function registerUserGamesRoute(router: Router): void {
  router.get(
    '/user/:userId',
    asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
      const { userId } = req.params;

      const userIdResult = UUIDSchema.safeParse(userId);
      if (!userIdResult.success) {
        throw createError('Invalid user ID format', 400, 'INVALID_USER_ID');
      }

      const limit = Math.min(Math.max(parseInt(req.query.limit as string) || 10, 1), 100);
      const offset = Math.max(parseInt(req.query.offset as string) || 0, 0);
      const statusParam = req.query.status as string | undefined;

      const validStatuses = new Set([
        'waiting',
        'active',
        'completed',
        'cancelled',
        'paused',
        'abandoned',
        'finished',
      ]);
      let status: PrismaGameStatus | undefined;
      if (statusParam) {
        if (!validStatuses.has(statusParam)) {
          throw createError(
            `Invalid status parameter. Must be one of: ${[...validStatuses].join(', ')}`,
            400,
            'INVALID_STATUS_PARAMETER'
          );
        }
        status = statusParam as PrismaGameStatus;
      }

      const prisma = getDatabaseClient();
      if (!prisma) {
        throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
      }

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

      const [gamesResult, totalResult] = await Promise.all([
        withQueryTimeoutStrict(
          prisma.game.findMany({
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
          })
        ),
        withQueryTimeoutStrict(prisma.game.count({ where: whereClause })),
      ]);

      if (!gamesResult.success || !totalResult.success) {
        throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
      }

      const games = gamesResult.data;
      const total = totalResult.data;

      const formattedGames = games.map((game) => {
        const participantIds = [
          game.player1Id,
          game.player2Id,
          game.player3Id,
          game.player4Id,
        ].filter(Boolean);
        const playerCount = participantIds.length;

        let resultReason: string | undefined;
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

        const recordMetadata = gameRecord.recordMetadata;
        const source =
          recordMetadata && typeof recordMetadata.source === 'string'
            ? recordMetadata.source
            : 'online_game';

        const rawOutcome = gameRecord.outcome;
        const outcome = typeof rawOutcome === 'string' ? rawOutcome : resultReason;

        return {
          id: game.id,
          boardType: game.boardType as BoardType,
          status: game.status as GameStatus,
          playerCount,
          numPlayers: playerCount,
          maxPlayers: game.maxPlayers,
          winnerId: game.winnerId,
          winnerName: game.winner ? getDisplayUsername(game.winner.username) : null,
          createdAt: game.createdAt.toISOString(),
          endedAt: game.endedAt?.toISOString() || null,
          moveCount: game._count.moves,
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
}
