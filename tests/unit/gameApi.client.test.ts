/**
 * Unit tests for the client-side game API service
 *
 * Tests the following client methods:
 * - gameApi.getGameHistory(gameId)
 * - gameApi.getGameDetails(gameId)
 * - gameApi.getUserGames(userId, options)
 */

import axios from 'axios';

// Mock axios
jest.mock('axios', () => {
  const mockAxiosInstance = {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
    interceptors: {
      request: { use: jest.fn() },
      response: { use: jest.fn() },
    },
    defaults: {
      headers: {
        common: {},
      },
    },
  };

  return {
    create: jest.fn(() => mockAxiosInstance),
    default: mockAxiosInstance,
  };
});

// Mock localStorage
const mockLocalStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
Object.defineProperty(global, 'localStorage', { value: mockLocalStorage });

// Import after mocks are set up
import {
  gameApi,
  GameHistoryResponse,
  GameDetailsResponse,
  UserGamesResponse,
} from '../../src/client/services/api';

// Get the mocked axios instance
const mockedAxios = axios.create() as jest.Mocked<typeof axios>;

describe('Client Game API Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.getItem.mockReturnValue('mock-token');
  });

  describe('getGameHistory', () => {
    it('should fetch game history and return formatted response', async () => {
      const mockResponse: GameHistoryResponse = {
        gameId: 'game-123',
        moves: [
          {
            moveNumber: 1,
            playerId: 'player-1',
            playerName: 'TestPlayer1',
            moveType: 'place_ring',
            moveData: { to: { x: 3, y: 3 }, player: 1 },
            timestamp: '2024-01-15T10:05:30.000Z',
          },
          {
            moveNumber: 2,
            playerId: 'player-2',
            playerName: 'TestPlayer2',
            moveType: 'place_ring',
            moveData: { to: { x: 5, y: 5 }, player: 2 },
            timestamp: '2024-01-15T10:06:00.000Z',
          },
        ],
        totalMoves: 2,
      };

      (mockedAxios.get as jest.Mock).mockResolvedValue({
        data: { success: true, data: mockResponse },
      });

      const result = await gameApi.getGameHistory('game-123');

      expect(mockedAxios.get).toHaveBeenCalledWith('/games/game-123/history');
      expect(result).toEqual(mockResponse);
      expect(result.moves).toHaveLength(2);
      expect(result.totalMoves).toBe(2);
    });

    it('should handle empty move history', async () => {
      const mockResponse: GameHistoryResponse = {
        gameId: 'game-123',
        moves: [],
        totalMoves: 0,
      };

      (mockedAxios.get as jest.Mock).mockResolvedValue({
        data: { success: true, data: mockResponse },
      });

      const result = await gameApi.getGameHistory('game-123');

      expect(result.moves).toHaveLength(0);
      expect(result.totalMoves).toBe(0);
    });

    it('should propagate errors from the API', async () => {
      const mockError = new Error('Network error');
      (mockedAxios.get as jest.Mock).mockRejectedValue(mockError);

      await expect(gameApi.getGameHistory('game-123')).rejects.toThrow('Network error');
    });
  });

  describe('getGameDetails', () => {
    it('should fetch game details and transform response', async () => {
      const mockApiResponse = {
        game: {
          id: 'game-123',
          status: 'active',
          boardType: 'square8',
          maxPlayers: 2,
          isRated: false,
          allowSpectators: true,
          player1: { id: 'player-1', username: 'Player1', rating: 1200 },
          player2: { id: 'player-2', username: 'Player2', rating: 1150 },
          player3: null,
          player4: null,
          winner: null,
          createdAt: '2024-01-15T10:00:00.000Z',
          updatedAt: '2024-01-15T10:30:00.000Z',
          startedAt: '2024-01-15T10:05:00.000Z',
          endedAt: null,
          moves: [{ id: 'move-1' }, { id: 'move-2' }],
        },
      };

      (mockedAxios.get as jest.Mock).mockResolvedValue({
        data: { success: true, data: mockApiResponse },
      });

      const result = await gameApi.getGameDetails('game-123');

      expect(mockedAxios.get).toHaveBeenCalledWith('/games/game-123');
      expect(result.id).toBe('game-123');
      expect(result.status).toBe('active');
      expect(result.boardType).toBe('square8');
      expect(result.players).toHaveLength(2);
      expect(result.moveCount).toBe(2);
    });

    it('should handle games with all 4 players', async () => {
      const mockApiResponse = {
        game: {
          id: 'game-456',
          status: 'active',
          boardType: 'hexagonal',
          maxPlayers: 4,
          isRated: true,
          allowSpectators: false,
          player1: { id: 'p1', username: 'Player1', rating: 1200 },
          player2: { id: 'p2', username: 'Player2', rating: 1150 },
          player3: { id: 'p3', username: 'Player3', rating: 1300 },
          player4: { id: 'p4', username: 'Player4', rating: 1100 },
          winner: null,
          createdAt: '2024-01-15T10:00:00.000Z',
          updatedAt: '2024-01-15T10:30:00.000Z',
          startedAt: '2024-01-15T10:05:00.000Z',
          endedAt: null,
          moves: [],
        },
      };

      (mockedAxios.get as jest.Mock).mockResolvedValue({
        data: { success: true, data: mockApiResponse },
      });

      const result = await gameApi.getGameDetails('game-456');

      expect(result.players).toHaveLength(4);
      expect(result.maxPlayers).toBe(4);
    });

    it('should handle completed games with winner', async () => {
      const mockApiResponse = {
        game: {
          id: 'game-789',
          status: 'completed',
          boardType: 'square8',
          maxPlayers: 2,
          isRated: true,
          allowSpectators: true,
          player1: { id: 'player-1', username: 'Winner', rating: 1220 },
          player2: { id: 'player-2', username: 'Loser', rating: 1130 },
          player3: null,
          player4: null,
          winner: { id: 'player-1', username: 'Winner' },
          createdAt: '2024-01-15T10:00:00.000Z',
          updatedAt: '2024-01-15T11:00:00.000Z',
          startedAt: '2024-01-15T10:05:00.000Z',
          endedAt: '2024-01-15T11:00:00.000Z',
          moves: Array(50).fill({ id: 'move' }),
        },
      };

      (mockedAxios.get as jest.Mock).mockResolvedValue({
        data: { success: true, data: mockApiResponse },
      });

      const result = await gameApi.getGameDetails('game-789');

      expect(result.status).toBe('completed');
      expect(result.winner).toEqual({ id: 'player-1', username: 'Winner' });
      expect(result.endedAt).toBe('2024-01-15T11:00:00.000Z');
      expect(result.moveCount).toBe(50);
    });
  });

  describe('getUserGames', () => {
    it('should fetch user games with default options', async () => {
      const mockResponse: UserGamesResponse = {
        games: [
          {
            id: 'game-1',
            boardType: 'square8',
            status: 'completed',
            playerCount: 2,
            maxPlayers: 2,
            winnerId: 'user-123',
            winnerName: 'TestUser',
            createdAt: '2024-01-15T10:00:00.000Z',
            endedAt: '2024-01-15T11:00:00.000Z',
            moveCount: 45,
          },
        ],
        pagination: {
          total: 1,
          limit: 10,
          offset: 0,
          hasMore: false,
        },
      };

      (mockedAxios.get as jest.Mock).mockResolvedValue({
        data: { success: true, data: mockResponse },
      });

      const result = await gameApi.getUserGames('user-123');

      expect(mockedAxios.get).toHaveBeenCalledWith('/games/user/user-123');
      expect(result.games).toHaveLength(1);
      expect(result.pagination.total).toBe(1);
    });

    it('should include query parameters when provided', async () => {
      const mockResponse: UserGamesResponse = {
        games: [],
        pagination: {
          total: 50,
          limit: 5,
          offset: 10,
          hasMore: true,
        },
      };

      (mockedAxios.get as jest.Mock).mockResolvedValue({
        data: { success: true, data: mockResponse },
      });

      const result = await gameApi.getUserGames('user-123', {
        limit: 5,
        offset: 10,
        status: 'completed',
      });

      expect(mockedAxios.get).toHaveBeenCalledWith(
        '/games/user/user-123?limit=5&offset=10&status=completed'
      );
      expect(result.pagination.limit).toBe(5);
      expect(result.pagination.offset).toBe(10);
    });

    it('should handle empty game list', async () => {
      const mockResponse: UserGamesResponse = {
        games: [],
        pagination: {
          total: 0,
          limit: 10,
          offset: 0,
          hasMore: false,
        },
      };

      (mockedAxios.get as jest.Mock).mockResolvedValue({
        data: { success: true, data: mockResponse },
      });

      const result = await gameApi.getUserGames('new-user');

      expect(result.games).toHaveLength(0);
      expect(result.pagination.total).toBe(0);
      expect(result.pagination.hasMore).toBe(false);
    });

    it('should handle pagination correctly', async () => {
      const mockResponse: UserGamesResponse = {
        games: Array(10)
          .fill(null)
          .map((_, i) => ({
            id: `game-${i}`,
            boardType: 'square8',
            status: 'completed',
            playerCount: 2,
            maxPlayers: 2,
            winnerId: 'user-123',
            winnerName: 'TestUser',
            createdAt: '2024-01-15T10:00:00.000Z',
            endedAt: '2024-01-15T11:00:00.000Z',
            moveCount: 30 + i,
          })),
        pagination: {
          total: 100,
          limit: 10,
          offset: 20,
          hasMore: true, // 30 < 100
        },
      };

      (mockedAxios.get as jest.Mock).mockResolvedValue({
        data: { success: true, data: mockResponse },
      });

      const result = await gameApi.getUserGames('user-123', { limit: 10, offset: 20 });

      expect(result.games).toHaveLength(10);
      expect(result.pagination.total).toBe(100);
      expect(result.pagination.hasMore).toBe(true);
    });
  });

  describe('Type interfaces', () => {
    it('GameHistoryMove should have required fields', () => {
      const move: import('../../src/client/services/api').GameHistoryMove = {
        moveNumber: 1,
        playerId: 'player-1',
        playerName: 'Player',
        moveType: 'place_ring',
        moveData: {},
        timestamp: '2024-01-15T10:00:00.000Z',
      };

      expect(move.moveNumber).toBeDefined();
      expect(move.playerId).toBeDefined();
      expect(move.playerName).toBeDefined();
      expect(move.moveType).toBeDefined();
      expect(move.moveData).toBeDefined();
      expect(move.timestamp).toBeDefined();
    });

    it('GameSummary should have required fields', () => {
      const summary: import('../../src/client/services/api').GameSummary = {
        id: 'game-1',
        boardType: 'square8',
        status: 'active',
        playerCount: 2,
        maxPlayers: 2,
        createdAt: '2024-01-15T10:00:00.000Z',
        moveCount: 10,
      };

      expect(summary.id).toBeDefined();
      expect(summary.boardType).toBeDefined();
      expect(summary.status).toBeDefined();
      expect(summary.playerCount).toBeDefined();
      expect(summary.maxPlayers).toBeDefined();
      expect(summary.createdAt).toBeDefined();
      expect(summary.moveCount).toBeDefined();

      // Optional fields
      expect(summary.winnerId).toBeUndefined();
      expect(summary.winnerName).toBeUndefined();
      expect(summary.endedAt).toBeUndefined();
    });
  });
});
