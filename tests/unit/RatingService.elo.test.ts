/**
 * Unit tests for RatingService Elo calculations
 *
 * These tests verify the mathematical correctness of the Elo rating algorithm
 * without any database dependencies.
 */

import { RatingService } from '../../src/server/services/RatingService';
import { RATING_CONSTANTS } from '../../src/shared/types/user';

describe('RatingService - Elo Calculations', () => {
  describe('calculateExpectedScore', () => {
    it('should return 0.5 for equal ratings', () => {
      const expected = RatingService.calculateExpectedScore(1200, 1200);
      expect(expected).toBeCloseTo(0.5, 4);
    });

    it('should return higher score for higher-rated player', () => {
      const expected = RatingService.calculateExpectedScore(1400, 1200);
      expect(expected).toBeGreaterThan(0.5);
      expect(expected).toBeLessThan(1);
    });

    it('should return lower score for lower-rated player', () => {
      const expected = RatingService.calculateExpectedScore(1000, 1200);
      expect(expected).toBeLessThan(0.5);
      expect(expected).toBeGreaterThan(0);
    });

    it('should return approximately 0.76 for 200 point advantage', () => {
      // Formula: 1 / (1 + 10^(-200/400)) = 1 / (1 + 10^-0.5) ≈ 0.759
      const expected = RatingService.calculateExpectedScore(1400, 1200);
      expect(expected).toBeCloseTo(0.759, 2);
    });

    it('should return approximately 0.24 for 200 point disadvantage', () => {
      // Formula: 1 / (1 + 10^(200/400)) = 1 / (1 + 10^0.5) ≈ 0.240
      const expected = RatingService.calculateExpectedScore(1000, 1200);
      expect(expected).toBeCloseTo(0.24, 2);
    });

    it('should sum to 1 for opposite ratings', () => {
      const expectedA = RatingService.calculateExpectedScore(1400, 1200);
      const expectedB = RatingService.calculateExpectedScore(1200, 1400);
      expect(expectedA + expectedB).toBeCloseTo(1, 4);
    });

    it('should return approximately 0.91 for 400 point advantage', () => {
      // Formula: 1 / (1 + 10^-1) = 1 / 1.1 ≈ 0.909
      const expected = RatingService.calculateExpectedScore(1600, 1200);
      expect(expected).toBeCloseTo(0.909, 2);
    });
  });

  describe('getExpectedScore (alias)', () => {
    it('should be identical to calculateExpectedScore', () => {
      const calc = RatingService.calculateExpectedScore(1400, 1200);
      const alias = RatingService.getExpectedScore(1400, 1200);
      expect(alias).toBe(calc);
    });
  });

  describe('getKFactor', () => {
    it('should return full K-factor for new players', () => {
      // Players with fewer than PROVISIONAL_GAMES (20) get K=32
      expect(RatingService.getKFactor(0)).toBe(32);
      expect(RatingService.getKFactor(10)).toBe(32);
      expect(RatingService.getKFactor(19)).toBe(32);
    });

    it('should return half K-factor for established players', () => {
      // Players with 20+ games get K=16
      expect(RatingService.getKFactor(20)).toBe(16);
      expect(RatingService.getKFactor(50)).toBe(16);
      expect(RatingService.getKFactor(100)).toBe(16);
    });
  });

  describe('calculateNewRating', () => {
    it('should increase rating on unexpected win', () => {
      // Lower-rated player (1000) beats higher-rated (1200)
      const expectedScore = RatingService.calculateExpectedScore(1000, 1200);
      const newRating = RatingService.calculateNewRating(1000, expectedScore, 1, 32);

      expect(newRating).toBeGreaterThan(1000);
      // With K=32 and expected ~0.24, gain should be ~24 points
      expect(newRating).toBeCloseTo(1024, 0);
    });

    it('should decrease rating on unexpected loss', () => {
      // Higher-rated player (1400) loses to lower-rated (1200)
      const expectedScore = RatingService.calculateExpectedScore(1400, 1200);
      const newRating = RatingService.calculateNewRating(1400, expectedScore, 0, 32);

      expect(newRating).toBeLessThan(1400);
      // With K=32 and expected ~0.76, loss should be ~24 points
      expect(newRating).toBeCloseTo(1376, 0);
    });

    it('should minimally change rating on expected outcome', () => {
      // Higher-rated player (1400) beats lower-rated (1200) - as expected
      const expectedScore = RatingService.calculateExpectedScore(1400, 1200);
      const newRating = RatingService.calculateNewRating(1400, expectedScore, 1, 32);

      expect(newRating).toBeGreaterThan(1400);
      // Expected outcome, so gain is small (~8 points)
      expect(newRating).toBeCloseTo(1408, 0);
    });

    it('should handle draws correctly', () => {
      // Equal ratings with draw should not change rating
      const expectedScore = RatingService.calculateExpectedScore(1200, 1200);
      const newRating = RatingService.calculateNewRating(1200, expectedScore, 0.5, 32);

      expect(newRating).toBe(1200);
    });

    it('should use smaller changes with lower K-factor', () => {
      const expectedScore = RatingService.calculateExpectedScore(1000, 1200);

      const newRatingHighK = RatingService.calculateNewRating(1000, expectedScore, 1, 32);
      const newRatingLowK = RatingService.calculateNewRating(1000, expectedScore, 1, 16);

      const changeHighK = newRatingHighK - 1000;
      const changeLowK = newRatingLowK - 1000;

      expect(changeHighK).toBeGreaterThan(changeLowK);
      expect(changeHighK).toBeCloseTo(changeLowK * 2, 0);
    });

    it('should clamp rating to minimum', () => {
      // Very heavy loss that would push below MIN_RATING
      const newRating = RatingService.calculateNewRating(
        RATING_CONSTANTS.MIN_RATING + 10,
        0.99, // Very expected win
        0, // But lost
        100 // High K-factor
      );

      expect(newRating).toBe(RATING_CONSTANTS.MIN_RATING);
    });

    it('should clamp rating to maximum', () => {
      // Very heavy win near MAX_RATING
      const newRating = RatingService.calculateNewRating(
        RATING_CONSTANTS.MAX_RATING - 10,
        0.01, // Very unexpected
        1, // Won
        100 // High K-factor
      );

      expect(newRating).toBe(RATING_CONSTANTS.MAX_RATING);
    });
  });

  describe('calculateRatingFromMatch', () => {
    it('should combine expected score and rating calculation', () => {
      // Convenience method should give same result as doing it manually
      const playerRating = 1200;
      const opponentRating = 1400;
      const actualScore = 1; // Win
      const kFactor = 32;

      const expected = RatingService.calculateExpectedScore(playerRating, opponentRating);
      const manualNew = RatingService.calculateNewRating(
        playerRating,
        expected,
        actualScore,
        kFactor
      );
      const convenience = RatingService.calculateRatingFromMatch(
        playerRating,
        opponentRating,
        actualScore,
        kFactor
      );

      expect(convenience).toBe(manualNew);
    });
  });

  describe('calculateMultiplayerRatings', () => {
    it('should increase winner rating and decrease loser ratings', () => {
      const players = [
        { id: 'player1', rating: 1200, rank: 1, gamesPlayed: 5 }, // Winner
        { id: 'player2', rating: 1200, rank: 2, gamesPlayed: 5 }, // Loser
        { id: 'player3', rating: 1200, rank: 2, gamesPlayed: 5 }, // Loser
      ];

      const newRatings = RatingService.calculateMultiplayerRatings(players);

      const player1New = newRatings.get('player1')!;
      const player2New = newRatings.get('player2')!;
      const player3New = newRatings.get('player3')!;

      expect(player1New).toBeGreaterThan(1200); // Winner gains
      expect(player2New).toBeLessThan(1200); // Loser loses
      expect(player3New).toBeLessThan(1200); // Loser loses
    });

    it('should give larger gains to underrated winners', () => {
      const players = [
        { id: 'player1', rating: 1100, rank: 1, gamesPlayed: 5 }, // Lower-rated winner
        { id: 'player2', rating: 1300, rank: 2, gamesPlayed: 5 }, // Higher-rated loser
      ];

      const highUnderdog = [
        { id: 'player1', rating: 900, rank: 1, gamesPlayed: 5 }, // Much lower-rated winner
        { id: 'player2', rating: 1500, rank: 2, gamesPlayed: 5 }, // Much higher-rated loser
      ];

      const smallGap = RatingService.calculateMultiplayerRatings(players);
      const largeGap = RatingService.calculateMultiplayerRatings(highUnderdog);

      const smallGapGain = smallGap.get('player1')! - 1100;
      const largeGapGain = largeGap.get('player1')! - 900;

      expect(largeGapGain).toBeGreaterThan(smallGapGain);
    });

    it('should handle 4-player games', () => {
      const players = [
        { id: 'player1', rating: 1200, rank: 1, gamesPlayed: 5 }, // Winner
        { id: 'player2', rating: 1200, rank: 2, gamesPlayed: 5 },
        { id: 'player3', rating: 1200, rank: 2, gamesPlayed: 5 },
        { id: 'player4', rating: 1200, rank: 2, gamesPlayed: 5 },
      ];

      const newRatings = RatingService.calculateMultiplayerRatings(players);

      expect(newRatings.size).toBe(4);
      expect(newRatings.get('player1')!).toBeGreaterThan(1200);
      expect(newRatings.get('player2')!).toBeLessThan(1200);
      expect(newRatings.get('player3')!).toBeLessThan(1200);
      expect(newRatings.get('player4')!).toBeLessThan(1200);
    });

    it('should be roughly zero-sum for equal-rated players', () => {
      const players = [
        { id: 'player1', rating: 1200, rank: 1, gamesPlayed: 25 },
        { id: 'player2', rating: 1200, rank: 2, gamesPlayed: 25 },
      ];

      const newRatings = RatingService.calculateMultiplayerRatings(players);

      const change1 = newRatings.get('player1')! - 1200;
      const change2 = newRatings.get('player2')! - 1200;

      // Changes should roughly cancel out (may not be exact due to rounding)
      expect(Math.abs(change1 + change2)).toBeLessThanOrEqual(2);
    });

    it('should use lower K-factor for established players', () => {
      const newPlayers = [
        { id: 'player1', rating: 1200, rank: 1, gamesPlayed: 5 },
        { id: 'player2', rating: 1200, rank: 2, gamesPlayed: 5 },
      ];

      const establishedPlayers = [
        { id: 'player1', rating: 1200, rank: 1, gamesPlayed: 50 },
        { id: 'player2', rating: 1200, rank: 2, gamesPlayed: 50 },
      ];

      const newRatings = RatingService.calculateMultiplayerRatings(newPlayers);
      const establishedRatings = RatingService.calculateMultiplayerRatings(establishedPlayers);

      const newChange = Math.abs(newRatings.get('player1')! - 1200);
      const establishedChange = Math.abs(establishedRatings.get('player1')! - 1200);

      // New players should have larger rating changes
      expect(newChange).toBeGreaterThan(establishedChange);
    });
  });

  describe('Edge cases', () => {
    it('should handle very large rating differences', () => {
      const expected = RatingService.calculateExpectedScore(2500, 500);
      expect(expected).toBeGreaterThan(0.99);
      expect(expected).toBeLessThanOrEqual(1);
    });

    it('should handle minimum ratings', () => {
      const newRating = RatingService.calculateRatingFromMatch(
        RATING_CONSTANTS.MIN_RATING,
        RATING_CONSTANTS.MIN_RATING,
        0.5, // Draw
        32
      );
      expect(newRating).toBe(RATING_CONSTANTS.MIN_RATING);
    });

    it('should handle maximum ratings', () => {
      const newRating = RatingService.calculateRatingFromMatch(
        RATING_CONSTANTS.MAX_RATING,
        RATING_CONSTANTS.MAX_RATING,
        0.5, // Draw
        32
      );
      expect(newRating).toBe(RATING_CONSTANTS.MAX_RATING);
    });
  });
});
