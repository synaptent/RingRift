import { RatingService } from '../../src/server/services/RatingService';
import { RATING_CONSTANTS } from '../../src/shared/types/user';
import { getDatabaseClient } from '../../src/server/database/connection';
import { logger } from '../../src/server/utils/logger';

// Mock dependencies
jest.mock('../../src/server/database/connection');
jest.mock('../../src/server/utils/logger');

describe('RatingService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getKFactor', () => {
    it('should return full K-factor for provisional players (< 20 games)', () => {
      expect(RatingService.getKFactor(0)).toBe(RATING_CONSTANTS.K_FACTOR);
      expect(RatingService.getKFactor(10)).toBe(RATING_CONSTANTS.K_FACTOR);
      expect(RatingService.getKFactor(19)).toBe(RATING_CONSTANTS.K_FACTOR);
    });

    it('should return half K-factor for established players (>= 20 games)', () => {
      expect(RatingService.getKFactor(20)).toBe(RATING_CONSTANTS.K_FACTOR / 2);
      expect(RatingService.getKFactor(50)).toBe(RATING_CONSTANTS.K_FACTOR / 2);
      expect(RatingService.getKFactor(100)).toBe(RATING_CONSTANTS.K_FACTOR / 2);
    });

    it('should use K_FACTOR constant value correctly', () => {
      // Verify the constant is 32
      expect(RATING_CONSTANTS.K_FACTOR).toBe(32);
      expect(RatingService.getKFactor(0)).toBe(32);
      expect(RatingService.getKFactor(20)).toBe(16);
    });
  });

  describe('calculateExpectedScore', () => {
    it('should return 0.5 for equal ratings', () => {
      const expected = RatingService.calculateExpectedScore(1200, 1200);
      expect(expected).toBe(0.5);
    });

    it('should return higher score for higher-rated player', () => {
      const higherRatedExpected = RatingService.calculateExpectedScore(1400, 1200);
      const lowerRatedExpected = RatingService.calculateExpectedScore(1200, 1400);

      expect(higherRatedExpected).toBeGreaterThan(0.5);
      expect(lowerRatedExpected).toBeLessThan(0.5);
    });

    it('should return symmetric expected scores (sum to 1)', () => {
      const player1Expected = RatingService.calculateExpectedScore(1400, 1200);
      const player2Expected = RatingService.calculateExpectedScore(1200, 1400);

      expect(player1Expected + player2Expected).toBeCloseTo(1, 10);
    });

    it('should return approximately 0.76 for 200 rating point advantage', () => {
      // 200 point difference: 1 / (1 + 10^(-200/400)) ≈ 0.76
      const expected = RatingService.calculateExpectedScore(1400, 1200);
      expect(expected).toBeCloseTo(0.76, 1);
    });

    it('should return approximately 0.91 for 400 rating point advantage', () => {
      // 400 point difference: 1 / (1 + 10^(-400/400)) = 1 / (1 + 0.1) ≈ 0.909
      const expected = RatingService.calculateExpectedScore(1600, 1200);
      expect(expected).toBeCloseTo(0.91, 1);
    });

    it('should handle very large rating differences', () => {
      const highRatedExpected = RatingService.calculateExpectedScore(2800, 1200);
      const lowRatedExpected = RatingService.calculateExpectedScore(1200, 2800);

      expect(highRatedExpected).toBeGreaterThan(0.99);
      expect(lowRatedExpected).toBeLessThan(0.01);
    });

    it('should handle edge case with MIN_RATING', () => {
      const expected = RatingService.calculateExpectedScore(
        RATING_CONSTANTS.MIN_RATING,
        RATING_CONSTANTS.MIN_RATING
      );
      expect(expected).toBe(0.5);
    });

    it('should handle edge case with MAX_RATING', () => {
      const expected = RatingService.calculateExpectedScore(
        RATING_CONSTANTS.MAX_RATING,
        RATING_CONSTANTS.MAX_RATING
      );
      expect(expected).toBe(0.5);
    });
  });

  describe('getExpectedScore (alias)', () => {
    it('should return same value as calculateExpectedScore', () => {
      const calc = RatingService.calculateExpectedScore(1400, 1200);
      const alias = RatingService.getExpectedScore(1400, 1200);
      expect(calc).toBe(alias);
    });
  });

  describe('calculateNewRating', () => {
    it('should increase rating on win when expected was lower', () => {
      const expectedScore = 0.5; // Even match
      const actualScore = 1; // Win
      const newRating = RatingService.calculateNewRating(1200, expectedScore, actualScore, 32);

      expect(newRating).toBeGreaterThan(1200);
      // Rating change = 32 * (1 - 0.5) = 16
      expect(newRating).toBe(1216);
    });

    it('should decrease rating on loss', () => {
      const expectedScore = 0.5;
      const actualScore = 0; // Loss
      const newRating = RatingService.calculateNewRating(1200, expectedScore, actualScore, 32);

      expect(newRating).toBeLessThan(1200);
      // Rating change = 32 * (0 - 0.5) = -16
      expect(newRating).toBe(1184);
    });

    it('should not change rating significantly on draw with equal expected score', () => {
      const expectedScore = 0.5;
      const actualScore = 0.5; // Draw
      const newRating = RatingService.calculateNewRating(1200, expectedScore, actualScore, 32);

      // Rating change = 32 * (0.5 - 0.5) = 0
      expect(newRating).toBe(1200);
    });

    it('should give larger gains when underdog wins', () => {
      const underdogExpected = 0.25; // Low expected score
      const underdogWin = RatingService.calculateNewRating(1200, underdogExpected, 1, 32);

      const favoriteExpected = 0.75;
      const favoriteWin = RatingService.calculateNewRating(1200, favoriteExpected, 1, 32);

      // Underdog winning should gain more rating
      expect(underdogWin - 1200).toBeGreaterThan(favoriteWin - 1200);
    });

    it('should respect MIN_RATING floor', () => {
      // Try to go below minimum rating
      const newRating = RatingService.calculateNewRating(
        RATING_CONSTANTS.MIN_RATING + 10,
        0.9, // High expected score
        0, // Loss
        100 // High K-factor
      );

      expect(newRating).toBeGreaterThanOrEqual(RATING_CONSTANTS.MIN_RATING);
    });

    it('should respect MAX_RATING ceiling', () => {
      // Try to go above maximum rating
      const newRating = RatingService.calculateNewRating(
        RATING_CONSTANTS.MAX_RATING - 10,
        0.1, // Low expected score
        1, // Win
        100 // High K-factor
      );

      expect(newRating).toBeLessThanOrEqual(RATING_CONSTANTS.MAX_RATING);
    });

    it('should round rating changes to integers', () => {
      const expectedScore = 0.333;
      const newRating = RatingService.calculateNewRating(1200, expectedScore, 1, 32);

      expect(Number.isInteger(newRating)).toBe(true);
    });

    it('should use default K-factor if not provided', () => {
      const newRating = RatingService.calculateNewRating(1200, 0.5, 1);
      // Default K_FACTOR = 32, change = 32 * (1 - 0.5) = 16
      expect(newRating).toBe(1216);
    });
  });

  describe('calculateRatingFromMatch', () => {
    it('should correctly combine expected score and new rating calculation', () => {
      const playerRating = 1400;
      const opponentRating = 1200;
      const actualScore = 1; // Win

      const expectedScore = RatingService.calculateExpectedScore(playerRating, opponentRating);
      const newRatingManual = RatingService.calculateNewRating(
        playerRating,
        expectedScore,
        actualScore,
        32
      );
      const newRatingConvenience = RatingService.calculateRatingFromMatch(
        playerRating,
        opponentRating,
        actualScore,
        32
      );

      expect(newRatingConvenience).toBe(newRatingManual);
    });

    it('should handle win scenario', () => {
      const newRating = RatingService.calculateRatingFromMatch(1200, 1200, 1, 32);
      expect(newRating).toBe(1216); // 1200 + 32 * (1 - 0.5)
    });

    it('should handle loss scenario', () => {
      const newRating = RatingService.calculateRatingFromMatch(1200, 1200, 0, 32);
      expect(newRating).toBe(1184); // 1200 + 32 * (0 - 0.5)
    });

    it('should handle draw scenario', () => {
      const newRating = RatingService.calculateRatingFromMatch(1200, 1200, 0.5, 32);
      expect(newRating).toBe(1200); // No change for draw with equal ratings
    });

    it('should use default K-factor if not provided', () => {
      const newRating = RatingService.calculateRatingFromMatch(1200, 1200, 1);
      expect(newRating).toBe(1216);
    });
  });

  describe('calculateMultiplayerRatings', () => {
    it('should update ratings for 3-player game with clear winner', () => {
      const players = [
        { id: 'p1', rating: 1200, rank: 1, gamesPlayed: 10 }, // Winner
        { id: 'p2', rating: 1200, rank: 2, gamesPlayed: 10 },
        { id: 'p3', rating: 1200, rank: 2, gamesPlayed: 10 },
      ];

      const newRatings = RatingService.calculateMultiplayerRatings(players);

      // Winner should gain rating
      expect(newRatings.get('p1')).toBeGreaterThan(1200);
      // Losers should lose rating (or stay same if draw between them)
      expect(newRatings.get('p2')).toBeLessThanOrEqual(1200);
      expect(newRatings.get('p3')).toBeLessThanOrEqual(1200);
    });

    it('should update ratings for 4-player game', () => {
      const players = [
        { id: 'p1', rating: 1200, rank: 1, gamesPlayed: 10 },
        { id: 'p2', rating: 1200, rank: 2, gamesPlayed: 10 },
        { id: 'p3', rating: 1200, rank: 3, gamesPlayed: 10 },
        { id: 'p4', rating: 1200, rank: 4, gamesPlayed: 10 },
      ];

      const newRatings = RatingService.calculateMultiplayerRatings(players);

      // All players should have new ratings
      expect(newRatings.size).toBe(4);
      // Better rank should generally result in higher new rating
      const r1 = newRatings.get('p1')!;
      const r4 = newRatings.get('p4')!;
      expect(r1).toBeGreaterThan(r4);
    });

    it('should handle tied ranks', () => {
      const players = [
        { id: 'p1', rating: 1200, rank: 1, gamesPlayed: 10 },
        { id: 'p2', rating: 1200, rank: 1, gamesPlayed: 10 }, // Tied for first
        { id: 'p3', rating: 1200, rank: 3, gamesPlayed: 10 },
      ];

      const newRatings = RatingService.calculateMultiplayerRatings(players);

      // Players tied for first should have similar ratings
      const r1 = newRatings.get('p1')!;
      const r2 = newRatings.get('p2')!;
      expect(r1).toBe(r2);
    });

    it('should respect rating bounds in multiplayer', () => {
      const players = [
        { id: 'p1', rating: RATING_CONSTANTS.MAX_RATING - 5, rank: 1, gamesPlayed: 10 },
        { id: 'p2', rating: 1200, rank: 2, gamesPlayed: 10 },
        { id: 'p3', rating: RATING_CONSTANTS.MIN_RATING + 5, rank: 3, gamesPlayed: 10 },
      ];

      const newRatings = RatingService.calculateMultiplayerRatings(players);

      expect(newRatings.get('p1')).toBeLessThanOrEqual(RATING_CONSTANTS.MAX_RATING);
      expect(newRatings.get('p3')).toBeGreaterThanOrEqual(RATING_CONSTANTS.MIN_RATING);
    });

    it('should handle missing gamesPlayed (default to 0)', () => {
      const players = [
        { id: 'p1', rating: 1200, rank: 1 }, // No gamesPlayed
        { id: 'p2', rating: 1200, rank: 2 },
      ];

      const newRatings = RatingService.calculateMultiplayerRatings(players);

      expect(newRatings.get('p1')).toBeDefined();
      expect(newRatings.get('p2')).toBeDefined();
    });

    it('should scale K-factor by number of opponents', () => {
      // In a 3-player game, K-factor is divided by 2 (3-1)
      // In a 4-player game, K-factor is divided by 3 (4-1)
      const twoPlayers = [
        { id: 'p1', rating: 1200, rank: 1, gamesPlayed: 10 },
        { id: 'p2', rating: 1200, rank: 2, gamesPlayed: 10 },
      ];

      const fourPlayers = [
        { id: 'p1', rating: 1200, rank: 1, gamesPlayed: 10 },
        { id: 'p2', rating: 1200, rank: 2, gamesPlayed: 10 },
        { id: 'p3', rating: 1200, rank: 3, gamesPlayed: 10 },
        { id: 'p4', rating: 1200, rank: 4, gamesPlayed: 10 },
      ];

      const twoPlayerRatings = RatingService.calculateMultiplayerRatings(twoPlayers);
      const fourPlayerRatings = RatingService.calculateMultiplayerRatings(fourPlayers);

      // Winner's gain in 2-player game should be different from 4-player
      const twoPlayerWinnerGain = twoPlayerRatings.get('p1')! - 1200;
      const fourPlayerWinnerGain = fourPlayerRatings.get('p1')! - 1200;

      // Both should gain, but amounts differ due to K-factor scaling
      expect(twoPlayerWinnerGain).toBeGreaterThan(0);
      expect(fourPlayerWinnerGain).toBeGreaterThan(0);
    });
  });

  describe('processGameResult', () => {
    let mockPrisma: any;

    beforeEach(() => {
      mockPrisma = {
        game: {
          findUnique: jest.fn(),
        },
        user: {
          findMany: jest.fn(),
          update: jest.fn(),
        },
        ratingHistory: {
          create: jest.fn(),
          findMany: jest.fn(),
          count: jest.fn(),
        },
        $transaction: jest.fn(),
      };
      (getDatabaseClient as jest.Mock).mockReturnValue(mockPrisma);
    });

    it('should return empty array when database is not available', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      const results = await RatingService.processGameResult('game1', 'player1', [
        'player1',
        'player2',
      ]);

      expect(results).toEqual([]);
      expect(logger.warn).toHaveBeenCalledWith(
        'Database not available for rating update',
        expect.any(Object)
      );
    });

    it('should return empty array for fewer than 2 players', async () => {
      const results = await RatingService.processGameResult('game1', 'player1', ['player1']);

      expect(results).toEqual([]);
      expect(logger.warn).toHaveBeenCalledWith(
        'Not enough players for rating update',
        expect.any(Object)
      );
    });

    it('should skip unrated games', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: false });

      const results = await RatingService.processGameResult('game1', 'player1', [
        'player1',
        'player2',
      ]);

      expect(results).toEqual([]);
      expect(logger.info).toHaveBeenCalledWith(
        'Skipping rating update for unrated game',
        expect.any(Object)
      );
    });

    it('should update ratings for 2-player win', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
        { id: 'player2', rating: 1200, gamesPlayed: 10 },
      ]);
      mockPrisma.$transaction.mockResolvedValue([{}, {}]);

      const results = await RatingService.processGameResult('game1', 'player1', [
        'player1',
        'player2',
      ]);

      expect(results).toHaveLength(2);
      expect(results[0].playerId).toBe('player1');
      expect(results[0].newRating).toBeGreaterThan(1200);
      expect(results[1].playerId).toBe('player2');
      expect(results[1].newRating).toBeLessThan(1200);
    });

    it('should update ratings for 2-player draw', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
        { id: 'player2', rating: 1200, gamesPlayed: 10 },
      ]);
      mockPrisma.$transaction.mockResolvedValue([{}, {}]);

      const results = await RatingService.processGameResult('game1', null, ['player1', 'player2']);

      expect(results).toHaveLength(2);
      // Equal ratings + draw = no change
      expect(results[0].change).toBe(0);
      expect(results[1].change).toBe(0);
    });

    it('should filter out null/undefined player IDs', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
        { id: 'player2', rating: 1200, gamesPlayed: 10 },
      ]);
      mockPrisma.$transaction.mockResolvedValue([{}, {}]);

      const results = await RatingService.processGameResult('game1', 'player1', [
        'player1',
        null as any,
        'player2',
        undefined as any,
      ]);

      expect(results).toHaveLength(2);
    });

    it('should handle multiplayer games', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
        { id: 'player2', rating: 1200, gamesPlayed: 10 },
        { id: 'player3', rating: 1200, gamesPlayed: 10 },
      ]);
      mockPrisma.$transaction.mockResolvedValue([{}, {}, {}]);

      const results = await RatingService.processGameResult('game1', 'player1', [
        'player1',
        'player2',
        'player3',
      ]);

      expect(results).toHaveLength(3);
      // Winner should gain rating
      const winner = results.find((r) => r.playerId === 'player1');
      expect(winner?.change).toBeGreaterThan(0);
    });

    it('should log error and throw on database failure', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockRejectedValue(new Error('Database error'));

      await expect(
        RatingService.processGameResult('game1', 'player1', ['player1', 'player2'])
      ).rejects.toThrow('Database error');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to process game result for ratings',
        expect.any(Object)
      );
    });

    it('should skip game not found (null)', async () => {
      mockPrisma.game.findUnique.mockResolvedValue(null);

      const results = await RatingService.processGameResult('game1', 'player1', [
        'player1',
        'player2',
      ]);

      expect(results).toEqual([]);
      expect(logger.info).toHaveBeenCalledWith(
        'Skipping rating update for unrated game',
        expect.any(Object)
      );
    });

    it('should warn when some players not found but still process', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      // Only return one player when two are expected
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
      ]);

      const results = await RatingService.processGameResult('game1', 'player1', [
        'player1',
        'player2',
      ]);

      expect(logger.warn).toHaveBeenCalledWith(
        'Some players not found for rating update',
        expect.objectContaining({
          expected: 2,
          found: 1,
        })
      );
      // Should return empty because can't find both players for 2-player game
      expect(results).toEqual([]);
    });

    it('should return empty array when neither player found in 2-player game', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockResolvedValue([]);

      const results = await RatingService.processGameResult('game1', 'player1', [
        'player1',
        'player2',
      ]);

      expect(results).toEqual([]);
      expect(logger.error).toHaveBeenCalledWith(
        'Players not found for rating update',
        expect.any(Object)
      );
    });

    it('should update ratings when player2 wins 2-player game', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
        { id: 'player2', rating: 1200, gamesPlayed: 10 },
      ]);
      mockPrisma.$transaction.mockResolvedValue([{}, {}]);

      const results = await RatingService.processGameResult('game1', 'player2', [
        'player1',
        'player2',
      ]);

      expect(results).toHaveLength(2);
      // Player 1 should lose rating (they lost)
      expect(results[0].playerId).toBe('player1');
      expect(results[0].newRating).toBeLessThan(1200);
      // Player 2 should gain rating (they won)
      expect(results[1].playerId).toBe('player2');
      expect(results[1].newRating).toBeGreaterThan(1200);
    });

    it('should handle multiplayer with no clear winner (null winnerId)', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
        { id: 'player2', rating: 1200, gamesPlayed: 10 },
        { id: 'player3', rating: 1200, gamesPlayed: 10 },
      ]);
      mockPrisma.$transaction.mockResolvedValue([{}, {}, {}]);

      const results = await RatingService.processGameResult('game1', null, [
        'player1',
        'player2',
        'player3',
      ]);

      expect(results).toHaveLength(3);
      // All players should have rank 2 (no winner), so minimal changes
    });

    it('should handle non-Error thrown in processGameResult catch block', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockRejectedValue('String error thrown');

      await expect(
        RatingService.processGameResult('game1', 'player1', ['player1', 'player2'])
      ).rejects.toBe('String error thrown');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to process game result for ratings',
        expect.objectContaining({
          error: 'String error thrown',
        })
      );
    });
  });

  describe('getPlayerRating', () => {
    let mockPrisma: any;

    beforeEach(() => {
      mockPrisma = {
        user: {
          findUnique: jest.fn(),
          count: jest.fn(),
        },
      };
      (getDatabaseClient as jest.Mock).mockReturnValue(mockPrisma);
    });

    it('should throw error when database is not available', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(RatingService.getPlayerRating('user1')).rejects.toThrow(
        'Database not available'
      );
    });

    it('should return null for non-existent user', async () => {
      mockPrisma.user.findUnique.mockResolvedValue(null);

      const result = await RatingService.getPlayerRating('nonexistent');

      expect(result).toBeNull();
    });

    it('should return correct player rating info', async () => {
      mockPrisma.user.findUnique.mockResolvedValue({
        id: 'user1',
        username: 'TestPlayer',
        rating: 1500,
        gamesPlayed: 50,
        gamesWon: 30,
        isActive: true,
      });
      mockPrisma.user.count.mockResolvedValue(10); // 10 players with higher rating

      const result = await RatingService.getPlayerRating('user1');

      expect(result).toEqual({
        userId: 'user1',
        username: 'TestPlayer',
        rating: 1500,
        gamesPlayed: 50,
        gamesWon: 30,
        rank: 11, // 10 higher + 1
        isProvisional: false, // 50 > 20
      });
    });

    it('should mark player as provisional with fewer than 20 games', async () => {
      mockPrisma.user.findUnique.mockResolvedValue({
        id: 'user1',
        username: 'NewPlayer',
        rating: 1200,
        gamesPlayed: 5,
        gamesWon: 3,
        isActive: true,
      });
      mockPrisma.user.count.mockResolvedValue(100);

      const result = await RatingService.getPlayerRating('user1');

      expect(result?.isProvisional).toBe(true);
    });

    it('should calculate rank correctly (higher rated players + 1)', async () => {
      mockPrisma.user.findUnique.mockResolvedValue({
        id: 'user1',
        username: 'TopPlayer',
        rating: 2500,
        gamesPlayed: 100,
        gamesWon: 80,
        isActive: true,
      });
      mockPrisma.user.count.mockResolvedValue(0); // No one higher

      const result = await RatingService.getPlayerRating('user1');

      expect(result?.rank).toBe(1); // Rank 1 (top player)
    });

    it('should log error and rethrow on database failure', async () => {
      mockPrisma.user.findUnique.mockRejectedValue(new Error('Query failed'));

      await expect(RatingService.getPlayerRating('user1')).rejects.toThrow('Query failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get player rating',
        expect.objectContaining({
          userId: 'user1',
          error: 'Query failed',
        })
      );
    });

    it('should handle non-Error thrown in getPlayerRating catch block', async () => {
      mockPrisma.user.findUnique.mockRejectedValue('String error');

      await expect(RatingService.getPlayerRating('user1')).rejects.toBe('String error');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get player rating',
        expect.objectContaining({
          error: 'String error',
        })
      );
    });
  });

  describe('getLeaderboard', () => {
    let mockPrisma: any;

    beforeEach(() => {
      mockPrisma = {
        user: {
          findMany: jest.fn(),
        },
      };
      (getDatabaseClient as jest.Mock).mockReturnValue(mockPrisma);
    });

    it('should throw error when database is not available', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(RatingService.getLeaderboard()).rejects.toThrow('Database not available');
    });

    it('should return formatted leaderboard entries', async () => {
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'u1', username: 'Player1', rating: 1800, gamesPlayed: 100, gamesWon: 70 },
        { id: 'u2', username: 'Player2', rating: 1700, gamesPlayed: 50, gamesWon: 30 },
      ]);

      const result = await RatingService.getLeaderboard();

      expect(result).toHaveLength(2);
      expect(result[0]).toEqual({
        userId: 'u1',
        username: 'Player1',
        rating: 1800,
        wins: 70,
        losses: 30,
        gamesPlayed: 100,
        winRate: 70.0,
        rank: 1,
      });
      expect(result[1].rank).toBe(2);
    });

    it('should apply limit and offset for pagination', async () => {
      mockPrisma.user.findMany.mockResolvedValue([]);

      await RatingService.getLeaderboard(10, 5);

      expect(mockPrisma.user.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 10,
          skip: 5,
        })
      );
    });

    it('should calculate win rate correctly', async () => {
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'u1', username: 'P1', rating: 1500, gamesPlayed: 3, gamesWon: 1 },
      ]);

      const result = await RatingService.getLeaderboard();

      // 1/3 = 0.333... => 33.33%
      expect(result[0].winRate).toBeCloseTo(33.33, 1);
    });

    it('should handle zero games played', async () => {
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'u1', username: 'P1', rating: 1200, gamesPlayed: 0, gamesWon: 0 },
      ]);

      const result = await RatingService.getLeaderboard();

      expect(result[0].winRate).toBe(0);
    });

    it('should use default limit of 50', async () => {
      mockPrisma.user.findMany.mockResolvedValue([]);

      await RatingService.getLeaderboard();

      expect(mockPrisma.user.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 50,
          skip: 0,
        })
      );
    });

    it('should log error and rethrow on database failure', async () => {
      mockPrisma.user.findMany.mockRejectedValue(new Error('Query failed'));

      await expect(RatingService.getLeaderboard()).rejects.toThrow('Query failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get leaderboard',
        expect.objectContaining({
          error: 'Query failed',
        })
      );
    });

    it('should handle non-Error thrown in getLeaderboard catch block', async () => {
      mockPrisma.user.findMany.mockRejectedValue('String error');

      await expect(RatingService.getLeaderboard()).rejects.toBe('String error');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get leaderboard',
        expect.objectContaining({
          error: 'String error',
        })
      );
    });
  });

  describe('getLeaderboardCount', () => {
    let mockPrisma: any;

    beforeEach(() => {
      mockPrisma = {
        user: {
          count: jest.fn(),
        },
      };
      (getDatabaseClient as jest.Mock).mockReturnValue(mockPrisma);
    });

    it('should throw error when database is not available', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(RatingService.getLeaderboardCount()).rejects.toThrow('Database not available');
    });

    it('should return count of active players with games', async () => {
      mockPrisma.user.count.mockResolvedValue(150);

      const result = await RatingService.getLeaderboardCount();

      expect(result).toBe(150);
      expect(mockPrisma.user.count).toHaveBeenCalledWith({
        where: {
          isActive: true,
          gamesPlayed: { gt: 0 },
        },
      });
    });

    it('should log error and rethrow on database failure', async () => {
      mockPrisma.user.count.mockRejectedValue(new Error('Count failed'));

      await expect(RatingService.getLeaderboardCount()).rejects.toThrow('Count failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get leaderboard count',
        expect.objectContaining({
          error: 'Count failed',
        })
      );
    });

    it('should handle non-Error thrown in getLeaderboardCount catch block', async () => {
      mockPrisma.user.count.mockRejectedValue('String error');

      await expect(RatingService.getLeaderboardCount()).rejects.toBe('String error');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get leaderboard count',
        expect.objectContaining({
          error: 'String error',
        })
      );
    });
  });

  describe('Elo Formula Accuracy', () => {
    it('should produce expected rating change for standard scenario', () => {
      // Standard scenario: 1200 vs 1200, K=32
      // Win: 1200 + 32*(1-0.5) = 1216
      // Loss: 1200 + 32*(0-0.5) = 1184
      const winnerNew = RatingService.calculateRatingFromMatch(1200, 1200, 1, 32);
      const loserNew = RatingService.calculateRatingFromMatch(1200, 1200, 0, 32);

      expect(winnerNew).toBe(1216);
      expect(loserNew).toBe(1184);
    });

    it('should maintain rating symmetry (winner gain ≈ loser loss)', () => {
      const player1Rating = 1400;
      const player2Rating = 1200;

      const winner = RatingService.calculateRatingFromMatch(player1Rating, player2Rating, 1, 32);
      const loser = RatingService.calculateRatingFromMatch(player2Rating, player1Rating, 0, 32);

      const winnerGain = winner - player1Rating;
      const loserLoss = player2Rating - loser;

      // Due to rounding, they may not be exactly equal
      expect(Math.abs(winnerGain - loserLoss)).toBeLessThanOrEqual(1);
    });

    it('should handle FIDE-like rating calculation', () => {
      // FIDE uses K=20 for established players
      // Test: 1500 vs 1600, K=20
      const expected = RatingService.calculateExpectedScore(1500, 1600);
      // Expected ≈ 0.36 for 100 point deficit

      const winNew = RatingService.calculateNewRating(1500, expected, 1, 20);
      const lossNew = RatingService.calculateNewRating(1500, expected, 0, 20);

      expect(winNew).toBeGreaterThan(1500);
      expect(lossNew).toBeLessThan(1500);
      // Winning as underdog gives bigger gain
      expect(winNew - 1500).toBeGreaterThan(10);
    });

    it('should verify expected score formula precision', () => {
      // Formula: 1 / (1 + 10^((Rb - Ra) / 400))
      // For Ra=1200, Rb=1000: 1 / (1 + 10^(-200/400)) = 1 / (1 + 10^(-0.5)) ≈ 0.76
      const expected = RatingService.calculateExpectedScore(1200, 1000);
      const manual = 1 / (1 + Math.pow(10, (1000 - 1200) / 400));

      expect(expected).toBe(manual);
    });

    it('should produce ratings that sum to original total (conservation)', () => {
      // For a 2-player game, total rating should be approximately conserved
      const r1 = 1200;
      const r2 = 1300;
      const total = r1 + r2;

      const new1 = RatingService.calculateRatingFromMatch(r1, r2, 1, 32);
      const new2 = RatingService.calculateRatingFromMatch(r2, r1, 0, 32);

      const newTotal = new1 + new2;

      // Due to rounding, there may be small differences
      expect(Math.abs(newTotal - total)).toBeLessThanOrEqual(2);
    });
  });

  describe('Edge Cases and Boundary Conditions', () => {
    it('should handle rating at exact minimum', () => {
      const newRating = RatingService.calculateNewRating(RATING_CONSTANTS.MIN_RATING, 0.5, 0, 32);
      expect(newRating).toBe(RATING_CONSTANTS.MIN_RATING);
    });

    it('should handle rating at exact maximum', () => {
      const newRating = RatingService.calculateNewRating(RATING_CONSTANTS.MAX_RATING, 0.5, 1, 32);
      expect(newRating).toBe(RATING_CONSTANTS.MAX_RATING);
    });

    it('should handle expected score of 0 (impossible but edge case)', () => {
      const newRating = RatingService.calculateNewRating(1200, 0, 1, 32);
      // Win with 0 expected = max gain
      expect(newRating).toBe(1232);
    });

    it('should handle expected score of 1 (impossible but edge case)', () => {
      const newRating = RatingService.calculateNewRating(1200, 1, 0, 32);
      // Loss with 1 expected = max loss
      expect(newRating).toBe(1168);
    });

    it('should handle K-factor of 0', () => {
      const newRating = RatingService.calculateNewRating(1200, 0.5, 1, 0);
      expect(newRating).toBe(1200); // No change
    });

    it('should handle very small rating differences', () => {
      const expected = RatingService.calculateExpectedScore(1200, 1201);
      expect(expected).toBeCloseTo(0.5, 2);
    });
  });

  describe('processGameResult - Additional Branch Coverage', () => {
    let mockPrisma: any;

    beforeEach(() => {
      mockPrisma = {
        game: {
          findUnique: jest.fn(),
        },
        user: {
          findMany: jest.fn(),
          update: jest.fn(),
        },
        ratingHistory: {
          create: jest.fn(),
          findMany: jest.fn(),
          count: jest.fn(),
        },
        $transaction: jest.fn(),
      };
      (getDatabaseClient as jest.Mock).mockReturnValue(mockPrisma);
    });

    it('should update ratings when player2 wins (covers else branch)', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
        { id: 'player2', rating: 1200, gamesPlayed: 10 },
      ]);
      mockPrisma.$transaction.mockResolvedValue([{}, {}]);

      // player2 wins (winnerId is player2, not player1)
      const results = await RatingService.processGameResult('game1', 'player2', [
        'player1',
        'player2',
      ]);

      expect(results).toHaveLength(2);
      // Player1 should lose rating
      expect(results[0].playerId).toBe('player1');
      expect(results[0].change).toBeLessThan(0);
      // Player2 should gain rating
      expect(results[1].playerId).toBe('player2');
      expect(results[1].change).toBeGreaterThan(0);
    });

    it('should warn when some players not found', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      // Return only 1 player when 2 expected
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
      ]);

      const results = await RatingService.processGameResult('game1', 'player1', [
        'player1',
        'player2',
      ]);

      expect(logger.warn).toHaveBeenCalledWith(
        'Some players not found for rating update',
        expect.objectContaining({
          expected: 2,
          found: 1,
        })
      );
      // Should return empty because player2 not found
      expect(results).toEqual([]);
    });

    it('should return empty array when player1 not found in 2-player game', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      // Return player2 but not player1
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player2', rating: 1200, gamesPlayed: 10 },
      ]);

      const results = await RatingService.processGameResult('game1', 'player1', [
        'player1',
        'player2',
      ]);

      expect(logger.error).toHaveBeenCalledWith(
        'Players not found for rating update',
        expect.objectContaining({
          gameId: 'game1',
        })
      );
      expect(results).toEqual([]);
    });

    it('should return empty array when player2 not found in 2-player game', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      // Return player1 but not player2
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
      ]);

      const results = await RatingService.processGameResult('game1', 'player1', [
        'player1',
        'player2',
      ]);

      expect(logger.error).toHaveBeenCalledWith(
        'Players not found for rating update',
        expect.any(Object)
      );
      expect(results).toEqual([]);
    });
  });

  describe('Error Handling - Database Failures', () => {
    let mockPrisma: any;

    beforeEach(() => {
      mockPrisma = {
        user: {
          findUnique: jest.fn(),
          findMany: jest.fn(),
          count: jest.fn(),
        },
      };
      (getDatabaseClient as jest.Mock).mockReturnValue(mockPrisma);
    });

    it('should log error and throw when getPlayerRating fails', async () => {
      mockPrisma.user.findUnique.mockResolvedValue({
        id: 'user1',
        username: 'Test',
        rating: 1200,
        gamesPlayed: 10,
        gamesWon: 5,
      });
      mockPrisma.user.count.mockRejectedValue(new Error('Count query failed'));

      await expect(RatingService.getPlayerRating('user1')).rejects.toThrow('Count query failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get player rating',
        expect.objectContaining({
          userId: 'user1',
          error: 'Count query failed',
        })
      );
    });

    it('should log error and throw when getLeaderboard fails', async () => {
      mockPrisma.user.findMany.mockRejectedValue(new Error('Query failed'));

      await expect(RatingService.getLeaderboard()).rejects.toThrow('Query failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get leaderboard',
        expect.objectContaining({
          error: 'Query failed',
        })
      );
    });

    it('should log error and throw when getLeaderboardCount fails', async () => {
      mockPrisma.user.count.mockRejectedValue(new Error('Count failed'));

      await expect(RatingService.getLeaderboardCount()).rejects.toThrow('Count failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get leaderboard count',
        expect.objectContaining({
          error: 'Count failed',
        })
      );
    });
  });

  describe('getRatingHistory', () => {
    let mockPrisma: any;

    beforeEach(() => {
      mockPrisma = {
        ratingHistory: {
          findMany: jest.fn(),
          count: jest.fn(),
        },
      };
      (getDatabaseClient as jest.Mock).mockReturnValue(mockPrisma);
    });

    it('should throw error when database is not available', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(RatingService.getRatingHistory('user1')).rejects.toThrow(
        'Database not available'
      );
    });

    it('should return rating history entries for a user', async () => {
      const mockHistory = [
        {
          id: 'entry1',
          gameId: 'game1',
          oldRating: 1200,
          newRating: 1216,
          change: 16,
          timestamp: new Date('2025-01-01'),
        },
        {
          id: 'entry2',
          gameId: 'game2',
          oldRating: 1216,
          newRating: 1200,
          change: -16,
          timestamp: new Date('2025-01-02'),
        },
      ];
      mockPrisma.ratingHistory.findMany.mockResolvedValue(mockHistory);
      mockPrisma.ratingHistory.count.mockResolvedValue(2);

      const result = await RatingService.getRatingHistory('user1');

      expect(result.history).toEqual(mockHistory);
      expect(result.total).toBe(2);
    });

    it('should apply limit and offset for pagination', async () => {
      mockPrisma.ratingHistory.findMany.mockResolvedValue([]);
      mockPrisma.ratingHistory.count.mockResolvedValue(0);

      await RatingService.getRatingHistory('user1', 10, 5);

      expect(mockPrisma.ratingHistory.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 10,
          skip: 5,
        })
      );
    });

    it('should use default limit of 30 and offset of 0', async () => {
      mockPrisma.ratingHistory.findMany.mockResolvedValue([]);
      mockPrisma.ratingHistory.count.mockResolvedValue(0);

      await RatingService.getRatingHistory('user1');

      expect(mockPrisma.ratingHistory.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 30,
          skip: 0,
        })
      );
    });

    it('should handle empty history', async () => {
      mockPrisma.ratingHistory.findMany.mockResolvedValue([]);
      mockPrisma.ratingHistory.count.mockResolvedValue(0);

      const result = await RatingService.getRatingHistory('newuser');

      expect(result.history).toEqual([]);
      expect(result.total).toBe(0);
    });

    it('should log error and throw on database failure', async () => {
      mockPrisma.ratingHistory.findMany.mockRejectedValue(new Error('Query failed'));

      await expect(RatingService.getRatingHistory('user1')).rejects.toThrow('Query failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get rating history',
        expect.objectContaining({
          userId: 'user1',
          error: 'Query failed',
        })
      );
    });

    it('should handle non-Error thrown in getRatingHistory catch block', async () => {
      mockPrisma.ratingHistory.findMany.mockRejectedValue('String error');

      await expect(RatingService.getRatingHistory('user1')).rejects.toBe('String error');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get rating history',
        expect.objectContaining({
          error: 'String error',
        })
      );
    });
  });
});
