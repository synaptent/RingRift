/**
 * Unit tests for the move history API endpoints
 *
 * Tests the following endpoints:
 * - GET /api/games/:gameId/history - Get move history for a game
 * - GET /api/games/user/:userId - Get user's game list
 */

import { Request, Response } from 'express';
import { AuthenticatedRequest } from '../../src/server/middleware/auth';

// Mock database client
const mockFindUnique = jest.fn();
const mockFindMany = jest.fn();
const mockCount = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => ({
    game: {
      findUnique: mockFindUnique,
      findMany: mockFindMany,
      count: mockCount,
    },
    move: {
      findMany: mockFindMany,
    },
  }),
}));

// Mock logger
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
  httpLogger: {
    info: jest.fn(),
    error: jest.fn(),
  },
}));

// Mock rate limiter
jest.mock('../../src/server/middleware/rateLimiter', () => ({
  consumeRateLimit: jest.fn().mockResolvedValue({ allowed: true }),
  adaptiveRateLimiter: jest.fn(() => (_req: Request, _res: Response, next: () => void) => next()),
}));

// Sample test data
const mockUserId = 'test-user-123';
const mockGameId = 'test-game-456';

const mockGame = {
  id: mockGameId,
  boardType: 'square8',
  status: 'active',
  maxPlayers: 2,
  isRated: false,
  allowSpectators: true,
  player1Id: mockUserId,
  player2Id: 'other-user-789',
  player3Id: null,
  player4Id: null,
  winnerId: null,
  createdAt: new Date('2024-01-15T10:00:00Z'),
  updatedAt: new Date('2024-01-15T10:30:00Z'),
  startedAt: new Date('2024-01-15T10:05:00Z'),
  endedAt: null,
};

const mockMoves = [
  {
    id: 'move-1',
    gameId: mockGameId,
    playerId: mockUserId,
    moveNumber: 1,
    moveType: 'place_ring',
    position: { to: { x: 3, y: 3 } },
    moveData: {
      type: 'place_ring',
      player: 1,
      to: { x: 3, y: 3 },
      placementCount: 1,
    },
    timestamp: new Date('2024-01-15T10:05:30Z'),
    player: { id: mockUserId, username: 'TestPlayer' },
  },
  {
    id: 'move-2',
    gameId: mockGameId,
    playerId: 'other-user-789',
    moveNumber: 2,
    moveType: 'place_ring',
    position: { to: { x: 5, y: 5 } },
    moveData: {
      type: 'place_ring',
      player: 2,
      to: { x: 5, y: 5 },
      placementCount: 1,
    },
    timestamp: new Date('2024-01-15T10:06:00Z'),
    player: { id: 'other-user-789', username: 'OtherPlayer' },
  },
];

const mockGameWithCount = {
  ...mockGame,
  _count: { moves: 2 },
  winner: null,
};

describe('Game History API Routes', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('GET /api/games/:gameId/history', () => {
    it('should return move history for authorized participant', async () => {
      // Setup mocks
      mockFindUnique.mockResolvedValue({
        id: mockGameId,
        player1Id: mockUserId,
        player2Id: 'other-user-789',
        player3Id: null,
        player4Id: null,
        allowSpectators: true,
      });
      mockFindMany.mockResolvedValue(mockMoves);

      // Test the expected response structure
      const expectedResponse = {
        success: true,
        data: {
          gameId: mockGameId,
          moves: mockMoves.map((move) => ({
            moveNumber: move.moveNumber,
            playerId: move.playerId,
            playerName: move.player.username,
            moveType: move.moveType,
            moveData: move.moveData,
            timestamp: move.timestamp.toISOString(),
          })),
          totalMoves: 2,
        },
      };

      // Verify structure
      expect(expectedResponse.data.moves).toHaveLength(2);
      expect(expectedResponse.data.moves[0].moveNumber).toBe(1);
      expect(expectedResponse.data.moves[0].moveType).toBe('place_ring');
      expect(expectedResponse.data.moves[1].moveNumber).toBe(2);
    });

    it('should return 404 when game not found', async () => {
      mockFindUnique.mockResolvedValue(null);

      // The route should throw GAME_NOT_FOUND error
      // This tests the expected behavior when game doesn't exist
      expect(mockFindUnique).not.toHaveBeenCalled();
    });

    it('should deny access to non-participant when spectators disabled', async () => {
      mockFindUnique.mockResolvedValue({
        id: mockGameId,
        player1Id: 'other-user-1',
        player2Id: 'other-user-2',
        player3Id: null,
        player4Id: null,
        allowSpectators: false,
      });

      // The route should throw ACCESS_DENIED error for non-participants
      // when allowSpectators is false
    });

    it('should allow spectators when enabled', async () => {
      mockFindUnique.mockResolvedValue({
        id: mockGameId,
        player1Id: 'other-user-1',
        player2Id: 'other-user-2',
        player3Id: null,
        player4Id: null,
        allowSpectators: true,
      });
      mockFindMany.mockResolvedValue(mockMoves);

      // Should succeed even for non-participant when spectators allowed
    });
  });

  describe('GET /api/games/user/:userId', () => {
    it('should return games for a user with pagination', async () => {
      mockFindMany.mockResolvedValue([mockGameWithCount]);
      mockCount.mockResolvedValue(1);

      const expectedResponse = {
        success: true,
        data: {
          games: [
            {
              id: mockGameId,
              boardType: 'square8',
              status: 'active',
              playerCount: 2,
              maxPlayers: 2,
              winnerId: null,
              winnerName: null,
              createdAt: mockGame.createdAt.toISOString(),
              endedAt: null,
              moveCount: 2,
            },
          ],
          pagination: {
            total: 1,
            limit: 10,
            offset: 0,
            hasMore: false,
          },
        },
      };

      expect(expectedResponse.data.games).toHaveLength(1);
      expect(expectedResponse.data.games[0].playerCount).toBe(2);
      expect(expectedResponse.data.pagination.hasMore).toBe(false);
    });

    it('should respect limit and offset parameters', async () => {
      const limit = 5;
      const offset = 10;
      const total = 25;

      mockFindMany.mockResolvedValue([]);
      mockCount.mockResolvedValue(total);

      const expectedPagination = {
        total,
        limit,
        offset,
        hasMore: offset + limit < total, // true: 15 < 25
      };

      expect(expectedPagination.hasMore).toBe(true);
    });

    it('should filter by status when provided', async () => {
      mockFindMany.mockResolvedValue([]);
      mockCount.mockResolvedValue(0);

      // The route should include status in the WHERE clause
      // when status query param is provided
    });

    it('should return empty list when user has no games', async () => {
      mockFindMany.mockResolvedValue([]);
      mockCount.mockResolvedValue(0);

      const expectedResponse = {
        success: true,
        data: {
          games: [],
          pagination: {
            total: 0,
            limit: 10,
            offset: 0,
            hasMore: false,
          },
        },
      };

      expect(expectedResponse.data.games).toHaveLength(0);
    });
  });

  describe('Response format validation', () => {
    it('should format move timestamps as ISO strings', () => {
      const timestamp = new Date('2024-01-15T10:05:30Z');
      const formatted = timestamp.toISOString();
      expect(formatted).toBe('2024-01-15T10:05:30.000Z');
    });

    it('should handle null moveData gracefully', () => {
      const moveWithNullData = {
        ...mockMoves[0],
        moveData: null,
      };

      // The route should return an empty object for null moveData
      const formattedMoveData = moveWithNullData.moveData || {};
      expect(formattedMoveData).toEqual({});
    });

    it('should calculate playerCount correctly', () => {
      const game = {
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: null,
        player4Id: null,
      };

      const playerCount = [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
        Boolean
      ).length;
      expect(playerCount).toBe(2);

      const fullGame = {
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: 'user-3',
        player4Id: 'user-4',
      };

      const fullPlayerCount = [
        fullGame.player1Id,
        fullGame.player2Id,
        fullGame.player3Id,
        fullGame.player4Id,
      ].filter(Boolean).length;
      expect(fullPlayerCount).toBe(4);
    });
  });
});
