import { RATING_CONSTANTS } from '../../shared/types/user';
import { getDatabaseClient } from '../database/connection';
import { logger } from '../utils/logger';

/**
 * Leaderboard entry returned by getLeaderboard()
 */
export interface LeaderboardEntry {
  userId: string;
  username: string;
  rating: number;
  wins: number;
  losses: number;
  gamesPlayed: number;
  winRate: number;
  rank: number;
}

/**
 * Player rating info returned by getPlayerRating()
 */
export interface PlayerRatingInfo {
  userId: string;
  username: string;
  rating: number;
  gamesPlayed: number;
  gamesWon: number;
  rank: number;
  isProvisional: boolean;
}

/**
 * Result of processing a game rating update
 */
export interface RatingUpdateResult {
  playerId: string;
  oldRating: number;
  newRating: number;
  change: number;
}

/**
 * Service for calculating and managing player Elo ratings.
 *
 * The Elo rating system is used to calculate relative skill levels.
 * Key concepts:
 * - Expected score: probability of winning based on rating difference
 * - K-factor: determines sensitivity of rating changes (higher = more volatile)
 * - Provisional period: new players have higher K-factor for faster calibration
 *
 * Formula: newRating = oldRating + K * (actualScore - expectedScore)
 * Expected score: 1 / (1 + 10^((opponentRating - playerRating) / 400))
 */
export class RatingService {
  /**
   * Get K-factor based on player's experience.
   * New players (< PROVISIONAL_GAMES) get higher K-factor for faster calibration.
   * Established players get standard K-factor for more stable ratings.
   *
   * @param gamesPlayed Number of rated games the player has completed
   * @returns K-factor to use for rating calculation
   */
  static getKFactor(gamesPlayed: number): number {
    if (gamesPlayed < RATING_CONSTANTS.PROVISIONAL_GAMES) {
      // Higher K-factor for new players - faster rating adjustment
      return RATING_CONSTANTS.K_FACTOR; // 32
    }
    // Lower K-factor for established players - more stable ratings
    return RATING_CONSTANTS.K_FACTOR / 2; // 16
  }

  /**
   * Calculate expected score based on rating difference.
   * Formula: 1 / (1 + 10^((Rb - Ra) / 400))
   *
   * @param playerRating Rating of the player
   * @param opponentRating Rating of the opponent
   * @returns Expected score (probability of winning) between 0 and 1
   */
  static calculateExpectedScore(playerRating: number, opponentRating: number): number {
    return 1 / (1 + Math.pow(10, (opponentRating - playerRating) / 400));
  }

  /**
   * Alias for calculateExpectedScore (for backward compatibility)
   */
  static getExpectedScore(ratingA: number, ratingB: number): number {
    return this.calculateExpectedScore(ratingA, ratingB);
  }

  /**
   * Calculate new rating after a game.
   *
   * @param currentRating Player's current rating
   * @param expectedScore Expected score (from calculateExpectedScore)
   * @param actualScore Actual result: 1.0 = win, 0.5 = draw, 0.0 = loss
   * @param kFactor K-factor for rating adjustment (default based on experience)
   * @returns The new rating, clamped to min/max bounds
   */
  static calculateNewRating(
    currentRating: number,
    expectedScore: number,
    actualScore: number,
    kFactor: number = RATING_CONSTANTS.K_FACTOR
  ): number {
    const ratingChange = Math.round(kFactor * (actualScore - expectedScore));
    let newRating = currentRating + ratingChange;

    // Clamp rating within bounds
    newRating = Math.max(
      RATING_CONSTANTS.MIN_RATING,
      Math.min(RATING_CONSTANTS.MAX_RATING, newRating)
    );

    return newRating;
  }

  /**
   * Calculate rating using player and opponent ratings directly.
   * Convenience method that combines expected score calculation and rating update.
   *
   * @param playerRating Current rating of the player
   * @param opponentRating Current rating of the opponent
   * @param actualScore Actual result: 1.0 = win, 0.5 = draw, 0.0 = loss
   * @param kFactor K-factor for rating adjustment
   * @returns The new rating for the player
   */
  static calculateRatingFromMatch(
    playerRating: number,
    opponentRating: number,
    actualScore: number,
    kFactor: number = RATING_CONSTANTS.K_FACTOR
  ): number {
    const expectedScore = this.calculateExpectedScore(playerRating, opponentRating);
    return this.calculateNewRating(playerRating, expectedScore, actualScore, kFactor);
  }

  /**
   * Calculate rating changes for a multiplayer game (e.g. 3 or 4 players).
   * Treats the game as a set of pairwise matches against all opponents.
   *
   * @param players Array of players with their id, rating, and final rank (1 = winner)
   * @returns Map of player IDs to new ratings
   */
  static calculateMultiplayerRatings(
    players: { id: string; rating: number; rank: number; gamesPlayed?: number }[]
  ): Map<string, number> {
    const newRatings = new Map<string, number>();

    for (const player of players) {
      let totalExpectedScore = 0;
      let totalActualScore = 0;

      for (const opponent of players) {
        if (player.id === opponent.id) continue;

        // Calculate expected score against this opponent
        totalExpectedScore += this.calculateExpectedScore(player.rating, opponent.rating);

        // Calculate actual score based on rank (lower rank is better)
        if (player.rank < opponent.rank) {
          totalActualScore += 1; // Win
        } else if (player.rank === opponent.rank) {
          totalActualScore += 0.5; // Draw
        }
        // Loss = 0, no addition needed
      }

      // Calculate K-factor based on games played
      const kFactor = this.getKFactor(player.gamesPlayed ?? 0) / (players.length - 1);

      const ratingChange = Math.round(kFactor * (totalActualScore - totalExpectedScore));
      let newRating = player.rating + ratingChange;

      newRating = Math.max(
        RATING_CONSTANTS.MIN_RATING,
        Math.min(RATING_CONSTANTS.MAX_RATING, newRating)
      );

      newRatings.set(player.id, newRating);
    }

    return newRatings;
  }

  /**
   * Process a completed game and update player ratings in the database.
   *
   * @param gameId ID of the completed game
   * @param winnerId ID of the winner, or null for a draw
   * @param playerIds Array of all player IDs in the game
   * @returns Array of rating update results
   */
  static async processGameResult(
    gameId: string,
    winnerId: string | null,
    playerIds: string[]
  ): Promise<RatingUpdateResult[]> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      logger.warn('Database not available for rating update', { gameId });
      return [];
    }

    // Filter out null/undefined player IDs
    const validPlayerIds = playerIds.filter((id): id is string => !!id);

    if (validPlayerIds.length < 2) {
      logger.warn('Not enough players for rating update', {
        gameId,
        playerCount: validPlayerIds.length,
      });
      return [];
    }

    try {
      // Check if game is rated
      const game = await prisma.game.findUnique({
        where: { id: gameId },
        select: { isRated: true },
      });

      if (!game?.isRated) {
        logger.info('Skipping rating update for unrated game', { gameId });
        return [];
      }

      // Fetch current ratings for all players
      const players = await prisma.user.findMany({
        where: { id: { in: validPlayerIds } },
        select: { id: true, rating: true, gamesPlayed: true },
      });

      if (players.length !== validPlayerIds.length) {
        logger.warn('Some players not found for rating update', {
          gameId,
          expected: validPlayerIds.length,
          found: players.length,
        });
      }

      const results: RatingUpdateResult[] = [];

      if (validPlayerIds.length === 2) {
        // 2-player game: direct head-to-head calculation
        const [player1Id, player2Id] = validPlayerIds;
        const player1 = players.find((p) => p.id === player1Id);
        const player2 = players.find((p) => p.id === player2Id);

        if (!player1 || !player2) {
          logger.error('Players not found for rating update', { gameId, player1Id, player2Id });
          return [];
        }

        // Determine scores
        let player1Score: number;
        let player2Score: number;

        if (winnerId === null) {
          // Draw
          player1Score = 0.5;
          player2Score = 0.5;
        } else if (winnerId === player1Id) {
          player1Score = 1;
          player2Score = 0;
        } else {
          player1Score = 0;
          player2Score = 1;
        }

        // Calculate new ratings
        const k1 = this.getKFactor(player1.gamesPlayed);
        const k2 = this.getKFactor(player2.gamesPlayed);

        const expected1 = this.calculateExpectedScore(player1.rating, player2.rating);
        const expected2 = this.calculateExpectedScore(player2.rating, player1.rating);

        const newRating1 = this.calculateNewRating(player1.rating, expected1, player1Score, k1);
        const newRating2 = this.calculateNewRating(player2.rating, expected2, player2Score, k2);

        // Update players in database
        const player1Data: {
          rating: number;
          gamesPlayed: { increment: number };
          gamesWon?: { increment: number };
        } = {
          rating: newRating1,
          gamesPlayed: { increment: 1 },
        };
        if (winnerId === player1Id) {
          player1Data.gamesWon = { increment: 1 };
        }

        const player2Data: {
          rating: number;
          gamesPlayed: { increment: number };
          gamesWon?: { increment: number };
        } = {
          rating: newRating2,
          gamesPlayed: { increment: 1 },
        };
        if (winnerId === player2Id) {
          player2Data.gamesWon = { increment: 1 };
        }

        await prisma.$transaction([
          prisma.user.update({
            where: { id: player1Id },
            data: player1Data,
          }),
          prisma.user.update({
            where: { id: player2Id },
            data: player2Data,
          }),
          // Create rating history entries
          prisma.ratingHistory.create({
            data: {
              userId: player1Id,
              gameId: gameId,
              oldRating: player1.rating,
              newRating: newRating1,
              change: newRating1 - player1.rating,
            },
          }),
          prisma.ratingHistory.create({
            data: {
              userId: player2Id,
              gameId: gameId,
              oldRating: player2.rating,
              newRating: newRating2,
              change: newRating2 - player2.rating,
            },
          }),
        ]);

        results.push(
          {
            playerId: player1Id,
            oldRating: player1.rating,
            newRating: newRating1,
            change: newRating1 - player1.rating,
          },
          {
            playerId: player2Id,
            oldRating: player2.rating,
            newRating: newRating2,
            change: newRating2 - player2.rating,
          }
        );

        logger.info('Ratings updated for 2-player game', {
          gameId,
          player1: {
            id: player1Id,
            old: player1.rating,
            new: newRating1,
            change: newRating1 - player1.rating,
          },
          player2: {
            id: player2Id,
            old: player2.rating,
            new: newRating2,
            change: newRating2 - player2.rating,
          },
        });
      } else {
        // Multiplayer game (3-4 players): use multiplayer rating calculation
        // Assign ranks based on winner
        const playersWithRanks = players.map((p) => ({
          id: p.id,
          rating: p.rating,
          gamesPlayed: p.gamesPlayed,
          rank: p.id === winnerId ? 1 : 2, // Winner gets rank 1, all others rank 2
        }));

        const newRatings = this.calculateMultiplayerRatings(playersWithRanks);

        // Build update transactions for users
        const userUpdates = players.map((player) => {
          const newRating = newRatings.get(player.id) ?? player.rating;
          const updateData: {
            rating: number;
            gamesPlayed: { increment: number };
            gamesWon?: { increment: number };
          } = {
            rating: newRating,
            gamesPlayed: { increment: 1 },
          };
          if (player.id === winnerId) {
            updateData.gamesWon = { increment: 1 };
          }
          return prisma.user.update({
            where: { id: player.id },
            data: updateData,
          });
        });

        // Build rating history entries
        const historyCreates = players.map((player) => {
          const newRating = newRatings.get(player.id) ?? player.rating;
          return prisma.ratingHistory.create({
            data: {
              userId: player.id,
              gameId: gameId,
              oldRating: player.rating,
              newRating: newRating,
              change: newRating - player.rating,
            },
          });
        });

        await prisma.$transaction([...userUpdates, ...historyCreates]);

        // Build results
        for (const player of players) {
          const newRating = newRatings.get(player.id) ?? player.rating;
          results.push({
            playerId: player.id,
            oldRating: player.rating,
            newRating,
            change: newRating - player.rating,
          });
        }

        logger.info('Ratings updated for multiplayer game', {
          gameId,
          playerCount: players.length,
          winnerId,
          updates: results,
        });
      }

      return results;
    } catch (error) {
      logger.error('Failed to process game result for ratings', {
        gameId,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Get a player's current rating and rank.
   *
   * @param userId ID of the user
   * @returns Player rating info including rank, or null if user not found
   */
  static async getPlayerRating(userId: string): Promise<PlayerRatingInfo | null> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      const user = await prisma.user.findUnique({
        where: { id: userId },
        select: {
          id: true,
          username: true,
          rating: true,
          gamesPlayed: true,
          gamesWon: true,
          isActive: true,
        },
      });

      if (!user) {
        return null;
      }

      // Calculate rank (number of active players with higher rating + 1)
      const higherRatedCount = await prisma.user.count({
        where: {
          isActive: true,
          rating: { gt: user.rating },
        },
      });

      return {
        userId: user.id,
        username: user.username,
        rating: user.rating,
        gamesPlayed: user.gamesPlayed,
        gamesWon: user.gamesWon,
        rank: higherRatedCount + 1,
        isProvisional: user.gamesPlayed < RATING_CONSTANTS.PROVISIONAL_GAMES,
      };
    } catch (error) {
      logger.error('Failed to get player rating', {
        userId,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Get the leaderboard of top players.
   *
   * @param limit Maximum number of players to return (default 50)
   * @param offset Offset for pagination (default 0)
   * @returns Array of leaderboard entries sorted by rating
   */
  static async getLeaderboard(limit: number = 50, offset: number = 0): Promise<LeaderboardEntry[]> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      const users = await prisma.user.findMany({
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
      });

      return users.map((user, index) => ({
        userId: user.id,
        username: user.username,
        rating: user.rating,
        wins: user.gamesWon,
        losses: user.gamesPlayed - user.gamesWon,
        gamesPlayed: user.gamesPlayed,
        winRate:
          user.gamesPlayed > 0 ? Math.round((user.gamesWon / user.gamesPlayed) * 10000) / 100 : 0,
        rank: offset + index + 1,
      }));
    } catch (error) {
      logger.error('Failed to get leaderboard', {
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Get total count of ranked players (for pagination).
   *
   * @returns Total number of active players with at least one game
   */
  static async getLeaderboardCount(): Promise<number> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      return await prisma.user.count({
        where: {
          isActive: true,
          gamesPlayed: { gt: 0 },
        },
      });
    } catch (error) {
      logger.error('Failed to get leaderboard count', {
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Get rating history for a player.
   *
   * @param userId ID of the user
   * @param limit Maximum number of entries to return (default 30)
   * @param offset Offset for pagination (default 0)
   * @returns Rating history entries and total count
   */
  static async getRatingHistory(
    userId: string,
    limit: number = 30,
    offset: number = 0
  ): Promise<{
    history: Array<{
      id: string;
      gameId: string | null;
      oldRating: number;
      newRating: number;
      change: number;
      timestamp: Date;
    }>;
    total: number;
  }> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      const [history, total] = await Promise.all([
        prisma.ratingHistory.findMany({
          where: { userId },
          orderBy: { timestamp: 'desc' },
          take: limit,
          skip: offset,
          select: {
            id: true,
            gameId: true,
            oldRating: true,
            newRating: true,
            change: true,
            timestamp: true,
          },
        }),
        prisma.ratingHistory.count({ where: { userId } }),
      ]);

      return { history, total };
    } catch (error) {
      logger.error('Failed to get rating history', {
        userId,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }
}

export default RatingService;
