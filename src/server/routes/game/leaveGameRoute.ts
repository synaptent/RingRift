import { Router, Response } from 'express';
import { GameStatus as PrismaGameStatus } from '@prisma/client';
import { getDatabaseClient } from '../../database/connection';
import { AuthenticatedRequest, getAuthUserId } from '../../middleware/auth';
import { createError, asyncHandler } from '../../middleware/errorHandler';
import { GameIdParamSchema } from '../../../shared/validation/schemas';
import { validateParams } from '../../middleware/validateRequest';
import { RatingService, RatingUpdateResult } from '../../services/RatingService';
import { httpLogger, logger } from '../../utils/logger';
import type { GameRouteContext } from './routeContext';
import { assertUserIsGameParticipant } from './accessControl';

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
export function registerLeaveGameRoute(router: Router, context: GameRouteContext): void {
  router.post(
    '/:gameId/leave',
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
      });

      if (!game) {
        throw createError('Game not found', 404, 'GAME_NOT_FOUND');
      }

      assertUserIsGameParticipant(userId, {
        player1Id: game.player1Id,
        player2Id: game.player2Id,
        player3Id: game.player3Id,
        player4Id: game.player4Id,
      });

      const wsServer = context.getWebSocketServer();

      if (game.status === PrismaGameStatus.active) {
        let handledViaSession = false;
        if (wsServer && typeof wsServer.handlePlayerResignFromHttp === 'function') {
          try {
            await wsServer.handlePlayerResignFromHttp(gameId, userId);
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
          const playerIds = [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
            (id): id is string => id !== null
          );

          const remainingPlayerIds = playerIds.filter((id) => id !== userId);
          winnerId =
            remainingPlayerIds.length === 1
              ? remainingPlayerIds[0]
              : (remainingPlayerIds[0] ?? null);

          if (game.isRated && winnerId) {
            try {
              ratingUpdates = await RatingService.processGameResult(gameId, winnerId, playerIds);
              logger.info('Rating updates applied for resignation', {
                gameId,
                resigningPlayer: userId,
                winnerId,
                ratingUpdates: ratingUpdates.map((ratingUpdate) => ({
                  playerId: ratingUpdate.playerId,
                  change: ratingUpdate.change,
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
              winnerId,
              endedAt: new Date(),
              updatedAt: new Date(),
            },
          });
        } else {
          const updated = await prisma.game.findUnique({
            where: { id: gameId },
            select: { winnerId: true },
          });
          winnerId = updated?.winnerId ?? null;
        }

        if (wsServer) {
          wsServer.broadcastLobbyEvent('lobby:game_cancelled', { gameId });
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
        return;
      }

      const updateData = { updatedAt: new Date() };

      if (game.player1Id === userId) {
        Object.assign(updateData, { player1: { disconnect: true } });
      } else if (game.player2Id === userId) {
        Object.assign(updateData, { player2: { disconnect: true } });
      } else if (game.player3Id === userId) {
        Object.assign(updateData, { player3: { disconnect: true } });
      } else if (game.player4Id === userId) {
        Object.assign(updateData, { player4: { disconnect: true } });
      }

      const updatedGame = await prisma.game.update({
        where: { id: gameId },
        data: updateData,
      });

      const remainingPlayers = [
        updatedGame.player1Id,
        updatedGame.player2Id,
        updatedGame.player3Id,
        updatedGame.player4Id,
      ].filter(Boolean);

      if (remainingPlayers.length === 0) {
        await prisma.game.update({
          where: { id: gameId },
          data: { status: PrismaGameStatus.abandoned, endedAt: new Date(), updatedAt: new Date() },
        });

        if (wsServer) {
          wsServer.broadcastLobbyEvent('lobby:game_cancelled', { gameId });
        }
      } else if (wsServer) {
        wsServer.broadcastLobbyEvent('lobby:game_joined', {
          gameId,
          playerCount: remainingPlayers.length,
        });
      }

      httpLogger.info(req, 'Player left game', { gameId, userId });

      res.json({
        success: true,
        message: 'Left game successfully',
      });
    })
  );
}
