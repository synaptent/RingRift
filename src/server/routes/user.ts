import { Router, Response } from 'express';
import { Prisma, GameStatus as PrismaGameStatus } from '@prisma/client';
import {
  getDatabaseClient,
  TransactionClient,
  withQueryTimeoutStrict,
} from '../database/connection';
import { ErrorCodes } from '../errors';
import { AuthenticatedRequest, getAuthUserId } from '../middleware/auth';
import { createError, asyncHandler } from '../middleware/errorHandler';
import {
  dataExportRateLimiter,
  userRatingRateLimiter,
  userSearchRateLimiter,
} from '../middleware/rateLimiter';
import { httpLogger } from '../utils/logger';
import {
  UpdateProfileSchema,
  GameListingQuerySchema,
  GameListingQueryInput,
  UserSearchQuerySchema,
  UserSearchQueryInput,
  LeaderboardQuerySchema,
  LeaderboardQueryInput,
  UUIDSchema,
} from '../../shared/validation/schemas';
import { validateBody, validateQuery } from '../middleware/validateRequest';
import { RatingService } from '../services/RatingService';
import type { WebSocketServer } from '../websocket/server';

// Module-level reference to the WebSocket server for session termination
let wsServer: WebSocketServer | null = null;

/**
 * Set the WebSocket server reference for this route module.
 * Called during route setup to enable WebSocket session termination.
 */
export function setWebSocketServer(server: WebSocketServer | null): void {
  wsServer = server;
}

const router = Router();

/**
 * @openapi
 * /users/profile:
 *   get:
 *     summary: Get current user profile
 *     description: Returns the authenticated user's profile information including stats.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: User profile retrieved successfully
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
 *                     user:
 *                       $ref: '#/components/schemas/User'
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/profile',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = getAuthUserId(req);

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const userResult = await withQueryTimeoutStrict(
      prisma.user.findUnique({
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
          isActive: true,
        },
      })
    );

    if (!userResult.success) {
      throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
    }

    const user = userResult.data;
    if (!user) {
      throw createError('User not found', 404, 'USER_NOT_FOUND');
    }

    res.json({
      success: true,
      data: { user },
    });
  })
);

/**
 * @openapi
 * /users/profile:
 *   put:
 *     summary: Update user profile
 *     description: |
 *       Updates the authenticated user's profile information.
 *       Only provided fields will be updated. All fields are optional.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/UpdateProfileRequest'
 *     responses:
 *       200:
 *         description: Profile updated successfully
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
 *                     user:
 *                       $ref: '#/components/schemas/User'
 *                 message:
 *                   type: string
 *                   example: Profile updated successfully
 *       400:
 *         description: Invalid profile data
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: VALIDATION_INVALID_PROFILE_DATA
 *                 message: Invalid profile data
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       409:
 *         description: Username or email already taken
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             examples:
 *               usernameExists:
 *                 summary: Username taken
 *                 value:
 *                   success: false
 *                   error:
 *                     code: RESOURCE_USERNAME_EXISTS
 *                     message: Username already taken
 *               emailExists:
 *                 summary: Email taken
 *                 value:
 *                   success: false
 *                   error:
 *                     code: RESOURCE_EMAIL_EXISTS
 *                     message: Email already registered
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.put(
  '/profile',
  validateBody(UpdateProfileSchema),
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = getAuthUserId(req);

    const { username, email, preferences } = req.body;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Check if username is already taken (if provided, exclude soft-deleted users)
    if (username) {
      const usernameCheckResult = await withQueryTimeoutStrict(
        prisma.user.findFirst({
          where: {
            username,
            NOT: { id: userId },
            deletedAt: null,
          },
        })
      );

      if (!usernameCheckResult.success) {
        throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
      }

      if (usernameCheckResult.data) {
        throw createError('Username already taken', 409, 'USERNAME_EXISTS');
      }
    }

    // Check if email is already taken (if provided, exclude soft-deleted users)
    if (email) {
      const emailCheckResult = await withQueryTimeoutStrict(
        prisma.user.findFirst({
          where: {
            email,
            NOT: { id: userId },
            deletedAt: null,
          },
        })
      );

      if (!emailCheckResult.success) {
        throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
      }

      if (emailCheckResult.data) {
        throw createError('Email already registered', 409, 'EMAIL_EXISTS');
      }
    }

    const updateResult = await withQueryTimeoutStrict(
      prisma.user.update({
        where: { id: userId },
        data: {
          ...(username && { username }),
          ...(email && { email }),
          ...(preferences && { preferences: preferences as Prisma.InputJsonValue }),
          updatedAt: new Date(),
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
          isActive: true,
        },
      })
    );

    if (!updateResult.success) {
      throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
    }

    const updatedUser = updateResult.data;

    httpLogger.info(req, 'User profile updated', { userId });

    res.json({
      success: true,
      data: { user: updatedUser },
      message: 'Profile updated successfully',
    });
  })
);

/**
 * @openapi
 * /users/stats:
 *   get:
 *     summary: Get user statistics
 *     description: |
 *       Returns detailed statistics for the authenticated user including
 *       rating, win/loss record, recent games, and rating history.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Statistics retrieved successfully
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
 *                     stats:
 *                       $ref: '#/components/schemas/UserStats'
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/stats',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = getAuthUserId(req);

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const userResult = await withQueryTimeoutStrict(
      prisma.user.findUnique({
        where: { id: userId },
        select: {
          rating: true,
          gamesPlayed: true,
          gamesWon: true,
        },
      })
    );

    if (!userResult.success) {
      throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
    }

    const user = userResult.data;
    if (!user) {
      throw createError('User not found', 404, 'USER_NOT_FOUND');
    }

    // Get recent games
    const recentGamesResult = await withQueryTimeoutStrict(
      prisma.game.findMany({
        where: {
          OR: [
            { player1Id: userId },
            { player2Id: userId },
            { player3Id: userId },
            { player4Id: userId },
          ],
          status: PrismaGameStatus.completed,
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
          player4Id: true,
        },
      })
    );

    if (!recentGamesResult.success) {
      throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
    }

    const recentGames = recentGamesResult.data;

    // Calculate win rate
    const winRate = user.gamesPlayed > 0 ? (user.gamesWon / user.gamesPlayed) * 100 : 0;

    // Get rating history from database (last 30 entries)
    const { history: ratingHistoryData } = await RatingService.getRatingHistory(userId, 30, 0);

    // Map to expected format (date/rating for charting)
    const ratingHistory = ratingHistoryData.map((entry) => ({
      date: entry.timestamp,
      rating: entry.newRating,
      change: entry.change,
      gameId: entry.gameId,
    }));

    // If no history yet, include current rating as starting point
    if (ratingHistory.length === 0) {
      ratingHistory.push({ date: new Date(), rating: user.rating, change: 0, gameId: null });
    }

    const stats = {
      rating: user.rating,
      gamesPlayed: user.gamesPlayed,
      gamesWon: user.gamesWon,
      gamesLost: user.gamesPlayed - user.gamesWon,
      winRate: Math.round(winRate * 100) / 100,
      recentGames,
      ratingHistory,
    };

    res.json({
      success: true,
      data: { stats },
    });
  })
);

/**
 * @openapi
 * /users/games:
 *   get:
 *     summary: Get user's game history
 *     description: |
 *       Returns a paginated list of games the authenticated user has participated in.
 *       Can be filtered by game status.
 *     tags: [Users]
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
 *         description: Offset for pagination
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
  '/games',
  validateQuery(GameListingQuerySchema),
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = getAuthUserId(req);

    const { status, limit, offset } = req.query as unknown as GameListingQueryInput;

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

    const gamesResult = await withQueryTimeoutStrict(
      prisma.game.findMany({
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
      })
    );

    if (!gamesResult.success) {
      throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
    }

    const games = gamesResult.data;

    const totalResult = await withQueryTimeoutStrict(prisma.game.count({ where: whereClause }));

    if (!totalResult.success) {
      throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
    }

    const total = totalResult.data;

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
 * /users/search:
 *   get:
 *     summary: Search users
 *     description: |
 *       Searches for users by username. Only returns active users.
 *       Results are sorted by rating (descending).
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: q
 *         required: true
 *         schema:
 *           type: string
 *           minLength: 1
 *           maxLength: 100
 *         description: Search query (matches username)
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 50
 *           default: 10
 *         description: Maximum results to return
 *     responses:
 *       200:
 *         description: Search results
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
 *                     users:
 *                       type: array
 *                       items:
 *                         $ref: '#/components/schemas/UserPublic'
 *       400:
 *         description: Search query required or invalid
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             examples:
 *               queryRequired:
 *                 summary: Query required
 *                 value:
 *                   success: false
 *                   error:
 *                     code: VALIDATION_SEARCH_QUERY_REQUIRED
 *                     message: Search query required
 *               invalidParams:
 *                 summary: Invalid parameters
 *                 value:
 *                   success: false
 *                   error:
 *                     code: VALIDATION_INVALID_QUERY_PARAMS
 *                     message: Invalid query parameters
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/search',
  userSearchRateLimiter,
  validateQuery(UserSearchQuerySchema),
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { q, limit } = req.query as unknown as UserSearchQueryInput;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const searchResult = await withQueryTimeoutStrict(
      prisma.user.findMany({
        where: {
          username: {
            contains: q,
            mode: 'insensitive',
          },
          isActive: true,
        },
        select: {
          id: true,
          username: true,
          rating: true,
          gamesPlayed: true,
          gamesWon: true,
        },
        take: limit,
        orderBy: { rating: 'desc' },
      })
    );

    if (!searchResult.success) {
      throw createError('Search query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
    }

    res.json({
      success: true,
      data: { users: searchResult.data },
    });
  })
);

/**
 * Placeholder username prefix for deleted users.
 * Used to identify anonymized accounts in game history displays.
 */
export const DELETED_USER_PREFIX = 'DeletedPlayer_';

/**
 * Display name shown in UI for deleted/anonymized users.
 */
export const DELETED_USER_DISPLAY_NAME = 'Deleted Player';

function anonymizedEmail(user: { id: string; email: string }): string {
  return `deleted+${user.id}@example.invalid`;
}

function anonymizedUsername(user: { id: string; username: string }): string {
  return `${DELETED_USER_PREFIX}${user.id.slice(0, 8)}`;
}

/**
 * Check if a username represents a deleted/anonymized user.
 */
export function isDeletedUserUsername(username: string): boolean {
  return username.startsWith(DELETED_USER_PREFIX);
}

/**
 * Get a user-friendly display name, showing "Deleted Player" for anonymized users.
 * @param username The username to format
 * @returns The display-friendly name
 */
export function getDisplayUsername(username: string | null | undefined): string {
  if (!username) {
    return DELETED_USER_DISPLAY_NAME;
  }
  if (isDeletedUserUsername(username)) {
    return DELETED_USER_DISPLAY_NAME;
  }
  return username;
}

/**
 * Format a player object for display, handling deleted users.
 * @param player Player data from database query
 * @returns Player object with display-friendly username
 */
export function formatPlayerForDisplay(player: null | undefined): null;
export function formatPlayerForDisplay<T extends { username: string }>(
  player: T
): T & { displayName: string };
export function formatPlayerForDisplay<T extends { username: string } | null | undefined>(
  player: T
): (T & { displayName: string }) | null {
  if (!player) {
    return null;
  }
  return {
    ...player,
    displayName: getDisplayUsername(player.username),
  };
}

/**
 * @openapi
 * /users/leaderboard:
 *   get:
 *     summary: Get leaderboard
 *     description: |
 *       Returns a paginated leaderboard of active users sorted by rating.
 *       Only includes users who have played at least one game.
 *       Supports filtering by board type and time period.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 100
 *           default: 50
 *         description: Number of results per page
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           minimum: 0
 *           default: 0
 *         description: Offset for pagination
 *       - in: query
 *         name: boardType
 *         schema:
 *           type: string
 *           enum: [all, square8, square19, hex8, hexagonal]
 *           default: all
 *         description: Filter by board type (only show players who have played on this board)
 *       - in: query
 *         name: timePeriod
 *         schema:
 *           type: string
 *           enum: [all, week, month, year]
 *           default: all
 *         description: Filter by time period (only show stats from games within this period)
 *     responses:
 *       200:
 *         description: Leaderboard retrieved successfully
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
 *                     users:
 *                       type: array
 *                       items:
 *                         $ref: '#/components/schemas/LeaderboardEntry'
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
  '/leaderboard',
  validateQuery(LeaderboardQuerySchema),
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { limit, offset, boardType, timePeriod } = req.query as unknown as LeaderboardQueryInput;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Build time filter based on timePeriod
    let dateFilter: Date | undefined;
    if (timePeriod && timePeriod !== 'all') {
      const now = new Date();
      switch (timePeriod) {
        case 'week':
          dateFilter = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
          break;
        case 'month':
          dateFilter = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
          break;
        case 'year':
          dateFilter = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
          break;
      }
    }

    // If filtering by boardType or timePeriod, we need to query via games
    const hasFilters = (boardType && boardType !== 'all') || dateFilter;

    if (hasFilters) {
      // Build game filter conditions for Prisma query
      const gameWhere: Prisma.GameWhereInput = {
        status: 'finished',
        isRated: true,
      };

      if (boardType && boardType !== 'all') {
        gameWhere.boardType = boardType;
      }

      if (dateFilter) {
        gameWhere.endedAt = { gte: dateFilter };
      }

      // First, get all games matching the filter criteria
      const matchingGamesResult = await withQueryTimeoutStrict(
        prisma.game.findMany({
          where: gameWhere,
          select: {
            id: true,
            winnerId: true,
            player1Id: true,
            player2Id: true,
            player3Id: true,
            player4Id: true,
          },
        })
      );

      if (!matchingGamesResult.success) {
        throw createError('Leaderboard query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
      }

      const matchingGames = matchingGamesResult.data;

      // Aggregate stats per user from matching games
      const userStats = new Map<string, { gamesPlayed: number; gamesWon: number }>();

      for (const game of matchingGames) {
        const playerIds = [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
          (id): id is string => !!id
        );

        for (const playerId of playerIds) {
          const stats = userStats.get(playerId) || { gamesPlayed: 0, gamesWon: 0 };
          stats.gamesPlayed++;
          if (game.winnerId === playerId) {
            stats.gamesWon++;
          }
          userStats.set(playerId, stats);
        }
      }

      const userIds = Array.from(userStats.keys());

      if (userIds.length === 0) {
        // No users match the filter
        res.json({
          success: true,
          data: {
            users: [],
            filters: {
              boardType: boardType || 'all',
              timePeriod: timePeriod || 'all',
            },
            pagination: {
              total: 0,
              limit,
              offset,
              hasMore: false,
            },
          },
        });
        return;
      }

      // Get user details for those who played matching games
      const usersResult = await withQueryTimeoutStrict(
        prisma.user.findMany({
          where: {
            id: { in: userIds },
            isActive: true,
          },
          select: {
            id: true,
            username: true,
            rating: true,
          },
          orderBy: { rating: 'desc' },
        })
      );

      if (!usersResult.success) {
        throw createError('Leaderboard query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
      }

      const users = usersResult.data;

      // Combine user data with filtered stats
      const usersWithStats = users
        .map((user) => {
          const stats = userStats.get(user.id) || { gamesPlayed: 0, gamesWon: 0 };
          return {
            ...user,
            gamesPlayed: stats.gamesPlayed,
            gamesWon: stats.gamesWon,
            winRate:
              stats.gamesPlayed > 0
                ? Math.round((stats.gamesWon / stats.gamesPlayed) * 10000) / 100
                : 0,
          };
        })
        .filter((u) => u.gamesPlayed > 0);

      // Apply pagination
      const total = usersWithStats.length;
      const paginatedUsers = usersWithStats.slice(offset, offset + limit);

      // Add rank
      const usersWithRank = paginatedUsers.map((user, index) => ({
        ...user,
        rank: offset + index + 1,
      }));

      res.json({
        success: true,
        data: {
          users: usersWithRank,
          filters: {
            boardType: boardType || 'all',
            timePeriod: timePeriod || 'all',
          },
          pagination: {
            total,
            limit,
            offset,
            hasMore: offset + limit < total,
          },
        },
      });
    } else {
      // No filters - use the simpler query
      const usersResult = await withQueryTimeoutStrict(
        prisma.user.findMany({
          where: {
            isActive: true,
            gamesPlayed: { gt: 0 },
          },
          select: {
            id: true,
            username: true,
            rating: true,
            gamesPlayed: true,
            gamesWon: true,
          },
          orderBy: { rating: 'desc' },
          take: limit,
          skip: offset,
        })
      );

      if (!usersResult.success) {
        throw createError('Leaderboard query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
      }

      const users = usersResult.data;

      const totalResult = await withQueryTimeoutStrict(
        prisma.user.count({
          where: {
            isActive: true,
            gamesPlayed: { gt: 0 },
          },
        })
      );

      if (!totalResult.success) {
        throw createError('Leaderboard query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
      }

      const total = totalResult.data;

      // Add rank to each user
      const usersWithRank = users.map((user, index) => ({
        ...user,
        rank: offset + index + 1,
        winRate:
          user.gamesPlayed > 0 ? Math.round((user.gamesWon / user.gamesPlayed) * 10000) / 100 : 0,
      }));

      res.json({
        success: true,
        data: {
          users: usersWithRank,
          filters: {
            boardType: 'all',
            timePeriod: 'all',
          },
          pagination: {
            total,
            limit,
            offset,
            hasMore: offset + limit < total,
          },
        },
      });
    }
  })
);

/**
 * @openapi
 * /users/{userId}/rating:
 *   get:
 *     summary: Get user rating and rank
 *     description: |
 *       Returns the rating information for a specific user, including their
 *       current rating, rank on the leaderboard, and whether the rating is
 *       provisional (fewer than 20 games played).
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: userId
 *         required: true
 *         schema:
 *           type: string
 *         description: The user's ID
 *     responses:
 *       200:
 *         description: Rating information retrieved successfully
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
 *                     userId:
 *                       type: string
 *                       example: "clxyz123"
 *                     username:
 *                       type: string
 *                       example: "player1"
 *                     rating:
 *                       type: integer
 *                       example: 1350
 *                     rank:
 *                       type: integer
 *                       example: 42
 *                     gamesPlayed:
 *                       type: integer
 *                       example: 25
 *                     gamesWon:
 *                       type: integer
 *                       example: 15
 *                     winRate:
 *                       type: number
 *                       example: 60.0
 *                     isProvisional:
 *                       type: boolean
 *                       example: false
 *                       description: True if player has fewer than 20 games
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/:userId/rating',
  userRatingRateLimiter,
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { userId } = req.params;

    // Validate userId format to prevent enumeration with invalid IDs
    const userIdResult = UUIDSchema.safeParse(userId);
    if (!userIdResult.success) {
      throw createError('Invalid user ID format', 400, 'INVALID_USER_ID');
    }

    try {
      const ratingResult = await withQueryTimeoutStrict(RatingService.getPlayerRating(userId));

      if (!ratingResult.success) {
        throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
      }

      const ratingInfo = ratingResult.data;
      if (!ratingInfo) {
        throw createError('User not found', 404, 'USER_NOT_FOUND');
      }

      // Calculate win rate
      const winRate =
        ratingInfo.gamesPlayed > 0
          ? Math.round((ratingInfo.gamesWon / ratingInfo.gamesPlayed) * 10000) / 100
          : 0;

      res.json({
        success: true,
        data: {
          userId: ratingInfo.userId,
          username: ratingInfo.username,
          rating: ratingInfo.rating,
          rank: ratingInfo.rank,
          gamesPlayed: ratingInfo.gamesPlayed,
          gamesWon: ratingInfo.gamesWon,
          winRate,
          isProvisional: ratingInfo.isProvisional,
        },
      });
    } catch (error) {
      if (error instanceof Error && 'code' in error && error.code === 'USER_NOT_FOUND') {
        throw error;
      }
      throw createError('Database not available', 503, 'DATABASE_UNAVAILABLE');
    }
  })
);

/**
 * @openapi
 * /users/{userId}/public-profile:
 *   get:
 *     summary: Get public profile for a user
 *     description: |
 *       Returns public profile information for a user, including stats,
 *       recent games, and rating history. No private data is exposed.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: userId
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Public profile data
 *       404:
 *         description: User not found
 */
router.get(
  '/:userId/public-profile',
  userRatingRateLimiter,
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { userId } = req.params;

    const userIdResult = UUIDSchema.safeParse(userId);
    if (!userIdResult.success) {
      throw createError('Invalid user ID format', 400, 'INVALID_USER_ID');
    }

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const userResult = await withQueryTimeoutStrict(
      prisma.user.findUnique({
        where: { id: userId, isActive: true },
        select: {
          id: true,
          username: true,
          rating: true,
          gamesPlayed: true,
          gamesWon: true,
          createdAt: true,
        },
      })
    );

    if (!userResult.success) {
      throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
    }

    const user = userResult.data;
    if (!user) {
      throw createError('User not found', 404, 'USER_NOT_FOUND');
    }

    // Recent completed games (public info only)
    const recentGamesResult = await withQueryTimeoutStrict(
      prisma.game.findMany({
        where: {
          OR: [
            { player1Id: userId },
            { player2Id: userId },
            { player3Id: userId },
            { player4Id: userId },
          ],
          status: PrismaGameStatus.completed,
        },
        orderBy: { endedAt: 'desc' },
        take: 10,
        select: {
          id: true,
          boardType: true,
          winnerId: true,
          endedAt: true,
          maxPlayers: true,
          player1: { select: { id: true, username: true } },
          player2: { select: { id: true, username: true } },
          player3: { select: { id: true, username: true } },
          player4: { select: { id: true, username: true } },
        },
      })
    );

    // Rating history (last 30 entries)
    let ratingHistory: Array<{ date: Date; rating: number; change: number }> = [];
    try {
      const { history } = await RatingService.getRatingHistory(userId, 30, 0);
      ratingHistory = history.map((entry) => ({
        date: entry.timestamp,
        rating: entry.newRating,
        change: entry.change,
      }));
    } catch {
      // Rating history unavailable - not critical
    }

    if (ratingHistory.length === 0) {
      ratingHistory.push({ date: user.createdAt, rating: user.rating, change: 0 });
    }

    const winRate =
      user.gamesPlayed > 0 ? Math.round((user.gamesWon / user.gamesPlayed) * 10000) / 100 : 0;

    res.json({
      success: true,
      data: {
        user: {
          id: user.id,
          username: user.username,
          rating: user.rating,
          gamesPlayed: user.gamesPlayed,
          gamesWon: user.gamesWon,
          winRate,
          isProvisional: user.gamesPlayed < 20,
          memberSince: user.createdAt,
        },
        recentGames: recentGamesResult.success ? recentGamesResult.data : [],
        ratingHistory,
      },
    });
  })
);

/**
 * @openapi
 * /users/{userId}/public-profile:
 *   get:
 *     summary: Get public profile for a user
 *     description: Returns public profile info, recent games, and rating history for any user.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: userId
 *         required: true
 *         schema:
 *           type: string
 *         description: The user's ID
 *     responses:
 *       200:
 *         description: Public profile retrieved successfully
 *       404:
 *         $ref: '#/components/responses/NotFound'
 */
router.get(
  '/:userId/public-profile',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const { userId } = req.params;

    const userIdResult = UUIDSchema.safeParse(userId);
    if (!userIdResult.success) {
      throw createError('Invalid user ID format', 400, 'INVALID_USER_ID');
    }

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const userResult = await withQueryTimeoutStrict(
      prisma.user.findUnique({
        where: { id: userId, isActive: true },
        select: {
          id: true,
          username: true,
          rating: true,
          gamesPlayed: true,
          gamesWon: true,
          createdAt: true,
        },
      })
    );

    if (!userResult.success) {
      throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
    }

    const user = userResult.data;
    if (!user) {
      throw createError('User not found', 404, 'USER_NOT_FOUND');
    }

    // Recent completed games
    const gamesResult = await withQueryTimeoutStrict(
      prisma.game.findMany({
        where: {
          OR: [
            { player1Id: userId },
            { player2Id: userId },
            { player3Id: userId },
            { player4Id: userId },
          ],
          status: PrismaGameStatus.completed,
        },
        orderBy: { endedAt: 'desc' },
        take: 10,
        select: {
          id: true,
          boardType: true,
          winnerId: true,
          endedAt: true,
          maxPlayers: true,
          player1: { select: { id: true, username: true } },
          player2: { select: { id: true, username: true } },
          player3: { select: { id: true, username: true } },
          player4: { select: { id: true, username: true } },
        },
      })
    );

    if (!gamesResult.success) {
      throw createError('Database query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
    }

    // Rating history
    const { history: ratingHistoryData } = await RatingService.getRatingHistory(userId, 30, 0);
    const ratingHistory = ratingHistoryData.map((entry) => ({
      date: entry.timestamp,
      rating: entry.newRating,
      change: entry.change,
    }));

    const winRate =
      user.gamesPlayed > 0 ? Math.round((user.gamesWon / user.gamesPlayed) * 10000) / 100 : 0;

    const isProvisional = user.gamesPlayed < 20;

    res.json({
      success: true,
      data: {
        user: {
          id: user.id,
          username: user.username,
          rating: user.rating,
          gamesPlayed: user.gamesPlayed,
          gamesWon: user.gamesWon,
          winRate,
          isProvisional,
          memberSince: user.createdAt,
        },
        recentGames: gamesResult.data,
        ratingHistory,
      },
    });
  })
);

/**
 * @openapi
 * /users/me:
 *   delete:
 *     summary: Delete current user account
 *     description: |
 *       Soft-deletes the authenticated user's account. This action:
 *       - Deactivates the account (isActive = false)
 *       - Sets deletedAt timestamp
 *       - Invalidates all tokens (tokenVersion incremented)
 *       - Clears sensitive tokens (verification, reset)
 *       - Anonymizes email and username to prevent PII exposure
 *       - Revokes all refresh tokens
 *       - Terminates active WebSocket connections
 *
 *       **Game History Preservation:**
 *       Game records are preserved with the user's player slot intact (player1Id, etc.)
 *       but the User record is anonymized. When games are displayed, the anonymized
 *       username ("DeletedPlayer_xxx") is shown, which the UI should display as
 *       "Deleted Player" for a user-friendly experience.
 *
 *       This approach ensures:
 *       - Opponents can still see their complete game history
 *       - Game statistics remain accurate
 *       - Move history is preserved (moves contain no PII)
 *       - No personally identifiable information is exposed
 *
 *       The account cannot be recovered after deletion.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Account deleted successfully
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
 *                   example: Account deleted successfully
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.delete(
  '/me',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = getAuthUserId(req);

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    await prisma.$transaction(async (tx: TransactionClient) => {
      const user = await tx.user.findUnique({
        where: { id: userId },
      });

      if (!user) {
        throw createError('User not found', 404, 'USER_NOT_FOUND');
      }

      if (user.deletedAt) {
        // Already soft-deleted; idempotent behaviour
        return;
      }

      // Soft-delete and anonymize the user record.
      // Game history is preserved: Game.player1Id/player2Id/etc. still reference this user,
      // but queries that JOIN to User will receive the anonymized username.
      // This ensures opponents can still see their game history while protecting
      // the deleted user's PII. The Move table retains playerId references but
      // moves contain no PII (just game positions and actions).
      await tx.user.update({
        where: { id: userId },
        data: {
          isActive: false,
          deletedAt: new Date(),
          tokenVersion: {
            increment: 1,
          },
          verificationToken: null,
          verificationTokenExpires: null,
          passwordResetToken: null,
          passwordResetExpires: null,
          email: anonymizedEmail(user),
          username: anonymizedUsername(user),
        },
      });

      // Best-effort cleanup of any persisted refresh tokens for this user.
      await tx.refreshToken.deleteMany({ where: { userId } });
    });

    // Terminate any active WebSocket connections for the deleted user.
    // This is done outside the transaction to ensure the DB changes are committed
    // before we disconnect the user's sessions. The tokenVersion increment within
    // the transaction ensures that even if this call fails, the user cannot
    // establish new connections.
    if (wsServer) {
      const terminatedCount = wsServer.terminateUserSessions(userId, 'Account deleted');
      httpLogger.info(req, 'WebSocket sessions terminated for deleted user', {
        event: 'user_delete_ws_termination',
        userId,
        terminatedCount,
      });
    }

    httpLogger.info(req, 'User account deleted (soft-delete)', {
      event: 'user_delete',
      userId,
    });

    res.status(200).json({
      success: true,
      message: 'Account deleted successfully',
    });
  })
);

/**
 * @openapi
 * /users/me/export:
 *   get:
 *     summary: Export all user data (GDPR Data Portability)
 *     description: |
 *       Returns all user data in JSON format for GDPR data portability compliance.
 *       The response includes:
 *       - Profile information (excluding password and internal tokens)
 *       - Complete game history with move details
 *       - Statistics (rating, win/loss record)
 *       - Any stored preferences
 *
 *       The response is formatted as a downloadable JSON file.
 *
 *       **Rate Limiting:** This endpoint is rate-limited to prevent abuse.
 *       Users may request their data export once per hour.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: User data exported successfully
 *         headers:
 *           Content-Disposition:
 *             schema:
 *               type: string
 *             description: Attachment header for file download
 *             example: 'attachment; filename="ringrift-data-export-user123.json"'
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 exportedAt:
 *                   type: string
 *                   format: date-time
 *                   example: "2024-01-15T10:30:00Z"
 *                 exportFormat:
 *                   type: string
 *                   example: "1.0"
 *                 profile:
 *                   type: object
 *                   properties:
 *                     id:
 *                       type: string
 *                       example: "user-123"
 *                     username:
 *                       type: string
 *                       example: "PlayerOne"
 *                     email:
 *                       type: string
 *                       format: email
 *                       example: "player@example.com"
 *                     createdAt:
 *                       type: string
 *                       format: date-time
 *                     emailVerified:
 *                       type: boolean
 *                     role:
 *                       type: string
 *                       enum: [USER, ADMIN, MODERATOR]
 *                 statistics:
 *                   type: object
 *                   properties:
 *                     rating:
 *                       type: integer
 *                       example: 1250
 *                     gamesPlayed:
 *                       type: integer
 *                       example: 50
 *                     wins:
 *                       type: integer
 *                       example: 28
 *                     losses:
 *                       type: integer
 *                       example: 22
 *                     winRate:
 *                       type: number
 *                       example: 56.0
 *                 games:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       id:
 *                         type: string
 *                       createdAt:
 *                         type: string
 *                         format: date-time
 *                       completedAt:
 *                         type: string
 *                         format: date-time
 *                         nullable: true
 *                       status:
 *                         type: string
 *                       result:
 *                         type: string
 *                         enum: [win, loss, draw, in_progress, abandoned]
 *                       boardType:
 *                         type: string
 *                       opponent:
 *                         type: string
 *                         nullable: true
 *                       moves:
 *                         type: array
 *                         items:
 *                           type: object
 *                           properties:
 *                             moveNumber:
 *                               type: integer
 *                             moveType:
 *                               type: string
 *                             timestamp:
 *                               type: string
 *                               format: date-time
 *                             isUserMove:
 *                               type: boolean
 *                 preferences:
 *                   type: object
 *                   description: User preferences (if any stored)
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       429:
 *         description: Rate limit exceeded - try again later
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: RATE_LIMIT_EXCEEDED
 *                 message: Data export rate limit exceeded. Please try again in 1 hour.
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/me/export',
  dataExportRateLimiter,
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = getAuthUserId(req);

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Limit games exported to prevent excessive load
    const MAX_EXPORT_GAMES = 1000;
    const MAX_MOVES_PER_GAME = 500;

    // Fetch user profile data (excluding sensitive fields)
    const userResult = await withQueryTimeoutStrict(
      prisma.user.findUnique({
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
          isActive: true,
          updatedAt: true,
          // Note: We intentionally exclude:
          // - passwordHash (security)
          // - tokenVersion (internal)
          // - verificationToken, passwordResetToken (security)
          // - deletedAt (internal)
        },
      })
    );

    if (!userResult.success) {
      throw createError('Export query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
    }

    const user = userResult.data;
    if (!user) {
      throw createError('User not found', 404, 'USER_NOT_FOUND');
    }

    // Fetch games where user participated (limited to prevent excessive load)
    const gamesResult = await withQueryTimeoutStrict(
      prisma.game.findMany({
        where: {
          OR: [
            { player1Id: userId },
            { player2Id: userId },
            { player3Id: userId },
            { player4Id: userId },
          ],
        },
        select: {
          id: true,
          createdAt: true,
          startedAt: true,
          endedAt: true,
          status: true,
          winnerId: true,
          boardType: true,
          maxPlayers: true,
          isRated: true,
          player1Id: true,
          player2Id: true,
          player3Id: true,
          player4Id: true,
          player1: { select: { id: true, username: true } },
          player2: { select: { id: true, username: true } },
          player3: { select: { id: true, username: true } },
          player4: { select: { id: true, username: true } },
          moves: {
            select: {
              moveNumber: true,
              moveType: true,
              timestamp: true,
              playerId: true,
              position: true,
            },
            orderBy: { moveNumber: 'asc' },
            take: MAX_MOVES_PER_GAME,
          },
        },
        orderBy: { createdAt: 'desc' },
        take: MAX_EXPORT_GAMES,
      })
    );

    if (!gamesResult.success) {
      throw createError('Export query timed out', 504, ErrorCodes.SERVER_GATEWAY_TIMEOUT);
    }

    const games = gamesResult.data;
    const wasGamesTruncated = games.length >= MAX_EXPORT_GAMES;

    // Transform games to user-friendly format
    const formattedGames = games.map((game) => {
      // Determine user's result in this game
      let result: 'win' | 'loss' | 'draw' | 'in_progress' | 'abandoned' = 'in_progress';
      if (game.status === 'completed' || game.status === 'finished') {
        if (game.winnerId === null) {
          result = 'draw';
        } else if (game.winnerId === userId) {
          result = 'win';
        } else {
          result = 'loss';
        }
      } else if (game.status === 'abandoned' || game.status === 'cancelled') {
        result = 'abandoned';
      }

      // Find opponent(s) - get usernames of other players
      const playerSlots = [
        { id: game.player1Id, user: game.player1 },
        { id: game.player2Id, user: game.player2 },
        { id: game.player3Id, user: game.player3 },
        { id: game.player4Id, user: game.player4 },
      ];

      const opponents = playerSlots
        .filter((slot) => slot.id && slot.id !== userId)
        .map((slot) => getDisplayUsername(slot.user?.username));

      // Format moves with user context
      const formattedMoves = game.moves.map((move) => ({
        moveNumber: move.moveNumber,
        moveType: move.moveType,
        timestamp: move.timestamp,
        isUserMove: move.playerId === userId,
      }));

      return {
        id: game.id,
        createdAt: game.createdAt,
        startedAt: game.startedAt,
        completedAt: game.endedAt,
        status: game.status,
        result,
        boardType: game.boardType,
        isRated: game.isRated,
        opponent: opponents.length === 1 ? opponents[0] : opponents.length > 0 ? opponents : null,
        moves: formattedMoves,
      };
    });

    // Calculate statistics
    const losses = user.gamesPlayed - user.gamesWon;
    const winRate =
      user.gamesPlayed > 0 ? Math.round((user.gamesWon / user.gamesPlayed) * 10000) / 100 : 0;

    // Build export data structure
    const exportData = {
      exportedAt: new Date().toISOString(),
      exportFormat: '1.0',
      profile: {
        id: user.id,
        username: user.username,
        email: user.email,
        createdAt: user.createdAt,
        emailVerified: user.emailVerified,
        role: user.role,
        isActive: user.isActive,
        lastLoginAt: user.lastLoginAt,
        updatedAt: user.updatedAt,
      },
      statistics: {
        rating: user.rating,
        gamesPlayed: user.gamesPlayed,
        wins: user.gamesWon,
        losses,
        winRate,
      },
      games: formattedGames,
      preferences: {}, // Placeholder for future preferences storage
      exportLimits: {
        maxGames: MAX_EXPORT_GAMES,
        maxMovesPerGame: MAX_MOVES_PER_GAME,
        wasGamesTruncated,
        gamesExported: formattedGames.length,
        totalGamesPlayed: user.gamesPlayed,
      },
    };

    // Set headers for file download
    res.setHeader('Content-Type', 'application/json');
    res.setHeader(
      'Content-Disposition',
      `attachment; filename="ringrift-data-export-${userId}.json"`
    );

    httpLogger.info(req, 'User data exported', {
      userId,
      gamesCount: formattedGames.length,
    });

    res.json(exportData);
  })
);

export default router;
