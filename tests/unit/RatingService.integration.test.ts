/**
 * Integration tests for RatingService database operations
 *
 * These tests mock the database client to test the integration logic
 * of processGameResult, getPlayerRating, and getLeaderboard.
 */

import { RatingService } from '../../src/server/services/RatingService';
import * as connection from '../../src/server/database/connection';

// Mock the database connection module
jest.mock('../../src/server/database/connection');

const mockGetDatabaseClient = connection.getDatabaseClient as jest.MockedFunction<
  typeof connection.getDatabaseClient
>;

describe('RatingService - Database Integration', () => {
  let mockPrisma: any;

  beforeEach(() => {
    jest.clearAllMocks();

    mockPrisma = {
      game: {
        findUnique: jest.fn(),
      },
      user: {
        findUnique: jest.fn(),
        findMany: jest.fn(),
        update: jest.fn(),
        count: jest.fn(),
      },
      $transaction: jest.fn(),
    };

    mockGetDatabaseClient.mockReturnValue(mockPrisma);
  });

  describe('processGameResult', () => {
    it('should skip processing for unrated games', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: false });

      const results = await RatingService.processGameResult('game-123', 'winner-id', [
        'player1',
        'player2',
      ]);

      expect(results).toEqual([]);
      expect(mockPrisma.user.findMany).not.toHaveBeenCalled();
    });

    it('should skip processing if fewer than 2 valid players', async () => {
      const results = await RatingService.processGameResult(
        'game-123',
        'winner-id',
        ['player1'] // Only one player
      );

      expect(results).toEqual([]);
    });

    it('should process 2-player game correctly', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
        { id: 'player2', rating: 1200, gamesPlayed: 10 },
      ]);
      mockPrisma.$transaction.mockResolvedValue([{}, {}]);

      const results = await RatingService.processGameResult(
        'game-123',
        'player1', // player1 wins
        ['player1', 'player2']
      );

      expect(results).toHaveLength(2);

      // Winner should gain rating
      const winner = results.find((r) => r.playerId === 'player1');
      expect(winner?.change).toBeGreaterThan(0);

      // Loser should lose rating
      const loser = results.find((r) => r.playerId === 'player2');
      expect(loser?.change).toBeLessThan(0);

      // Verify transaction was called
      expect(mockPrisma.$transaction).toHaveBeenCalled();
    });

    it('should process draw correctly', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
        { id: 'player2', rating: 1200, gamesPlayed: 10 },
      ]);
      mockPrisma.$transaction.mockResolvedValue([{}, {}]);

      const results = await RatingService.processGameResult(
        'game-123',
        null, // Draw - no winner
        ['player1', 'player2']
      );

      expect(results).toHaveLength(2);

      // Both players should have minimal rating change for equal ratings
      results.forEach((result) => {
        expect(result.change).toBe(0);
      });
    });

    it('should use higher K-factor for new players', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });

      // Player1 is new (5 games), Player2 is established (30 games)
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 5 },
        { id: 'player2', rating: 1200, gamesPlayed: 30 },
      ]);
      mockPrisma.$transaction.mockResolvedValue([{}, {}]);

      const results = await RatingService.processGameResult(
        'game-123',
        'player1', // New player wins
        ['player1', 'player2']
      );

      // New player should have larger change due to higher K-factor
      const newPlayer = results.find((r) => r.playerId === 'player1');
      const establishedPlayer = results.find((r) => r.playerId === 'player2');

      // New player K=32, established K=16
      // With equal starting ratings, winner gains more with higher K
      expect(Math.abs(newPlayer!.change)).toBeGreaterThan(Math.abs(establishedPlayer!.change));
    });

    it('should handle multiplayer games', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
        { id: 'player2', rating: 1200, gamesPlayed: 10 },
        { id: 'player3', rating: 1200, gamesPlayed: 10 },
      ]);
      mockPrisma.$transaction.mockResolvedValue([{}, {}, {}]);

      const results = await RatingService.processGameResult(
        'game-123',
        'player1', // player1 wins
        ['player1', 'player2', 'player3']
      );

      expect(results).toHaveLength(3);

      // Winner should gain
      const winner = results.find((r) => r.playerId === 'player1');
      expect(winner?.change).toBeGreaterThan(0);

      // Others should lose
      const loser2 = results.find((r) => r.playerId === 'player2');
      const loser3 = results.find((r) => r.playerId === 'player3');
      expect(loser2?.change).toBeLessThan(0);
      expect(loser3?.change).toBeLessThan(0);
    });

    it('should return empty array when database unavailable', async () => {
      mockGetDatabaseClient.mockReturnValue(null);

      const results = await RatingService.processGameResult('game-123', 'winner-id', [
        'player1',
        'player2',
      ]);

      expect(results).toEqual([]);
    });

    it('should filter out null player IDs', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ isRated: true });
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'player1', rating: 1200, gamesPlayed: 10 },
        { id: 'player2', rating: 1200, gamesPlayed: 10 },
      ]);
      mockPrisma.$transaction.mockResolvedValue([{}, {}]);

      const results = await RatingService.processGameResult('game-123', 'player1', [
        'player1',
        null as any,
        'player2',
        undefined as any,
      ]);

      // Should process only the 2 valid players
      expect(results).toHaveLength(2);
    });
  });

  describe('getPlayerRating', () => {
    it('should return player rating info with rank', async () => {
      mockPrisma.user.findUnique.mockResolvedValue({
        id: 'user-123',
        username: 'testplayer',
        rating: 1350,
        gamesPlayed: 25,
        gamesWon: 15,
        isActive: true,
      });
      mockPrisma.user.count.mockResolvedValue(10); // 10 players with higher rating

      const result = await RatingService.getPlayerRating('user-123');

      expect(result).toEqual({
        userId: 'user-123',
        username: 'testplayer',
        rating: 1350,
        gamesPlayed: 25,
        gamesWon: 15,
        rank: 11, // 10 higher + 1
        isProvisional: false, // 25 games > 20 threshold
      });
    });

    it('should mark rating as provisional for new players', async () => {
      mockPrisma.user.findUnique.mockResolvedValue({
        id: 'user-123',
        username: 'newplayer',
        rating: 1200,
        gamesPlayed: 5,
        gamesWon: 2,
        isActive: true,
      });
      mockPrisma.user.count.mockResolvedValue(0);

      const result = await RatingService.getPlayerRating('user-123');

      expect(result?.isProvisional).toBe(true);
    });

    it('should return null for non-existent user', async () => {
      mockPrisma.user.findUnique.mockResolvedValue(null);

      const result = await RatingService.getPlayerRating('non-existent');

      expect(result).toBeNull();
    });

    it('should throw when database unavailable', async () => {
      mockGetDatabaseClient.mockReturnValue(null);

      await expect(RatingService.getPlayerRating('user-123')).rejects.toThrow(
        'Database not available'
      );
    });
  });

  describe('getLeaderboard', () => {
    it('should return leaderboard entries with ranks', async () => {
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'user1', username: 'player1', rating: 1500, gamesPlayed: 50, gamesWon: 30 },
        { id: 'user2', username: 'player2', rating: 1400, gamesPlayed: 40, gamesWon: 20 },
        { id: 'user3', username: 'player3', rating: 1300, gamesPlayed: 30, gamesWon: 10 },
      ]);

      const result = await RatingService.getLeaderboard(10, 0);

      expect(result).toHaveLength(3);
      expect(result[0]).toEqual({
        userId: 'user1',
        username: 'player1',
        rating: 1500,
        wins: 30,
        losses: 20,
        gamesPlayed: 50,
        winRate: 60,
        rank: 1,
      });
      expect(result[1].rank).toBe(2);
      expect(result[2].rank).toBe(3);
    });

    it('should handle pagination offset for ranks', async () => {
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'user11', username: 'player11', rating: 1100, gamesPlayed: 20, gamesWon: 8 },
      ]);

      const result = await RatingService.getLeaderboard(10, 10); // Offset of 10

      expect(result[0].rank).toBe(11); // 10 + 0 + 1
    });

    it('should calculate win rate correctly', async () => {
      mockPrisma.user.findMany.mockResolvedValue([
        { id: 'user1', username: 'player1', rating: 1500, gamesPlayed: 100, gamesWon: 33 },
      ]);

      const result = await RatingService.getLeaderboard();

      expect(result[0].winRate).toBe(33); // 33/100 = 33%
    });

    it('should throw when database unavailable', async () => {
      mockGetDatabaseClient.mockReturnValue(null);

      await expect(RatingService.getLeaderboard()).rejects.toThrow('Database not available');
    });
  });

  describe('getLeaderboardCount', () => {
    it('should return total count of ranked players', async () => {
      mockPrisma.user.count.mockResolvedValue(42);

      const count = await RatingService.getLeaderboardCount();

      expect(count).toBe(42);
      expect(mockPrisma.user.count).toHaveBeenCalledWith({
        where: {
          isActive: true,
          gamesPlayed: { gt: 0 },
        },
      });
    });

    it('should throw when database unavailable', async () => {
      mockGetDatabaseClient.mockReturnValue(null);

      await expect(RatingService.getLeaderboardCount()).rejects.toThrow('Database not available');
    });
  });
});
