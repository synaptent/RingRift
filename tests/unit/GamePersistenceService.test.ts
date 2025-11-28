import {
  GamePersistenceService,
  CreateGameConfig,
  SaveMoveData,
} from '../../src/server/services/GamePersistenceService';
import { Move, GameState, GameResult, BoardType } from '../../src/shared/types/game';

// Mock the database connection module
jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(),
  withTransaction: jest.fn(),
}));

// Mock the logger
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    debug: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

import { getDatabaseClient } from '../../src/server/database/connection';
import { logger } from '../../src/server/utils/logger';

describe('GamePersistenceService', () => {
  let mockPrisma: any;

  beforeEach(() => {
    jest.clearAllMocks();

    // Setup mock Prisma client
    mockPrisma = {
      game: {
        create: jest.fn(),
        findUnique: jest.fn(),
        findMany: jest.fn(),
        update: jest.fn(),
        delete: jest.fn(),
      },
      move: {
        create: jest.fn(),
        findMany: jest.fn(),
      },
    };

    (getDatabaseClient as jest.Mock).mockReturnValue(mockPrisma);
  });

  describe('createGame', () => {
    const baseConfig: CreateGameConfig = {
      boardType: 'square8',
      maxPlayers: 2,
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      isRated: true,
    };

    it('should create a game record in the database', async () => {
      const mockGameId = 'test-game-id-123';
      mockPrisma.game.create.mockResolvedValue({ id: mockGameId });

      const result = await GamePersistenceService.createGame(baseConfig);

      expect(result).toBe(mockGameId);
      expect(mockPrisma.game.create).toHaveBeenCalledTimes(1);
      expect(mockPrisma.game.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          boardType: 'square8',
          maxPlayers: 2,
          isRated: true,
          status: 'waiting',
        }),
      });
    });

    it('should include optional player IDs when provided', async () => {
      const configWithPlayers: CreateGameConfig = {
        ...baseConfig,
        player1Id: 'player1-id',
        player2Id: 'player2-id',
      };
      mockPrisma.game.create.mockResolvedValue({ id: 'test-id' });

      await GamePersistenceService.createGame(configWithPlayers);

      const createCall = mockPrisma.game.create.mock.calls[0][0];
      expect(createCall.data.player1Id).toBe('player1-id');
      expect(createCall.data.player2Id).toBe('player2-id');
    });

    it('should include RNG seed when provided', async () => {
      const configWithSeed: CreateGameConfig = {
        ...baseConfig,
        rngSeed: 12345,
      };
      mockPrisma.game.create.mockResolvedValue({ id: 'test-id' });

      await GamePersistenceService.createGame(configWithSeed);

      const createCall = mockPrisma.game.create.mock.calls[0][0];
      expect(createCall.data.rngSeed).toBe(12345);
    });

    it('should throw error when database is not available', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(GamePersistenceService.createGame(baseConfig)).rejects.toThrow(
        'Database not available'
      );
    });

    it('should log game creation', async () => {
      mockPrisma.game.create.mockResolvedValue({ id: 'test-id' });

      await GamePersistenceService.createGame(baseConfig);

      expect(logger.info).toHaveBeenCalledWith(
        'Game created in database',
        expect.objectContaining({
          gameId: 'test-id',
          boardType: 'square8',
        })
      );
    });
  });

  describe('saveMove', () => {
    const mockMove: Move = {
      id: 'move-123',
      type: 'place_ring',
      player: 1,
      to: { x: 3, y: 3 },
      timestamp: new Date(),
      thinkTime: 1000,
      moveNumber: 1,
    };

    const saveMoveData: SaveMoveData = {
      gameId: 'game-123',
      playerId: 'player-123',
      moveNumber: 1,
      move: mockMove,
    };

    it('should save move to database', async () => {
      mockPrisma.move.create.mockResolvedValue({ id: 'db-move-id' });

      await GamePersistenceService.saveMove(saveMoveData);

      expect(mockPrisma.move.create).toHaveBeenCalledTimes(1);
      expect(mockPrisma.move.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          gameId: 'game-123',
          playerId: 'player-123',
          moveNumber: 1,
          moveType: 'place_ring',
        }),
      });
    });

    it('should include rich move data', async () => {
      const moveWithCapture: Move = {
        ...mockMove,
        type: 'overtaking_capture',
        from: { x: 2, y: 2 },
        captureTarget: { x: 3, y: 3 },
        capturedStacks: [],
      };

      await GamePersistenceService.saveMove({
        ...saveMoveData,
        move: moveWithCapture,
      });

      const createCall = mockPrisma.move.create.mock.calls[0][0];
      expect(createCall.data.moveData).toBeDefined();
      expect(createCall.data.moveData.captureTarget).toEqual({ x: 3, y: 3 });
    });

    it('should not throw when database is unavailable (non-blocking)', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      // Should not throw
      await expect(GamePersistenceService.saveMove(saveMoveData)).resolves.toBeUndefined();
      expect(logger.warn).toHaveBeenCalledWith(
        'Database not available for saving move',
        expect.any(Object)
      );
    });

    it('should log errors but not throw on database failure', async () => {
      mockPrisma.move.create.mockRejectedValue(new Error('DB error'));

      await GamePersistenceService.saveMove(saveMoveData);

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to save move to database',
        expect.objectContaining({
          error: 'DB error',
        })
      );
    });
  });

  describe('saveMoveSync', () => {
    const mockMove: Move = {
      id: 'move-123',
      type: 'place_ring',
      player: 1,
      to: { x: 3, y: 3 },
      timestamp: new Date(),
      thinkTime: 1000,
      moveNumber: 1,
    };

    it('should return true on successful save', async () => {
      mockPrisma.move.create.mockResolvedValue({ id: 'db-move-id' });

      const result = await GamePersistenceService.saveMoveSync({
        gameId: 'game-123',
        playerId: 'player-123',
        moveNumber: 1,
        move: mockMove,
      });

      expect(result).toBe(true);
    });

    it('should return false on failure', async () => {
      mockPrisma.move.create.mockRejectedValue(new Error('DB error'));

      const result = await GamePersistenceService.saveMoveSync({
        gameId: 'game-123',
        playerId: 'player-123',
        moveNumber: 1,
        move: mockMove,
      });

      expect(result).toBe(false);
    });

    it('should return false when database is unavailable', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      const result = await GamePersistenceService.saveMoveSync({
        gameId: 'game-123',
        playerId: 'player-123',
        moveNumber: 1,
        move: mockMove,
      });

      expect(result).toBe(false);
    });
  });

  describe('loadGame', () => {
    const mockGame = {
      id: 'game-123',
      boardType: 'square8',
      status: 'active',
      player1: { id: 'p1', username: 'Player1' },
      player2: { id: 'p2', username: 'Player2' },
      player3: null,
      player4: null,
      moves: [
        { id: 'm1', moveNumber: 1, position: { to: { x: 3, y: 3 } }, moveType: 'place_ring' },
      ],
    };

    it('should load game with moves and players', async () => {
      mockPrisma.game.findUnique.mockResolvedValue(mockGame);

      const result = await GamePersistenceService.loadGame('game-123');

      expect(result).not.toBeNull();
      expect(result?.game.id).toBe('game-123');
      expect(result?.moves).toHaveLength(1);
      expect(result?.players.player1?.username).toBe('Player1');
    });

    it('should return null when game not found', async () => {
      mockPrisma.game.findUnique.mockResolvedValue(null);

      const result = await GamePersistenceService.loadGame('non-existent');

      expect(result).toBeNull();
    });

    it('should throw when database unavailable', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(GamePersistenceService.loadGame('game-123')).rejects.toThrow(
        'Database not available'
      );
    });
  });

  describe('getGameHistory', () => {
    const mockMoves = [
      {
        id: 'm1',
        gameId: 'game-123',
        playerId: 'p1',
        moveNumber: 1,
        position: { to: { x: 3, y: 3 } },
        moveType: 'place_ring',
        moveData: {
          id: 'm1',
          type: 'place_ring',
          player: 1,
          to: { x: 3, y: 3 },
          thinkTime: 1000,
          moveNumber: 1,
        },
        timestamp: new Date(),
      },
      {
        id: 'm2',
        gameId: 'game-123',
        playerId: 'p2',
        moveNumber: 2,
        position: { to: { x: 4, y: 4 } },
        moveType: 'place_ring',
        moveData: {
          id: 'm2',
          type: 'place_ring',
          player: 2,
          to: { x: 4, y: 4 },
          thinkTime: 500,
          moveNumber: 2,
        },
        timestamp: new Date(),
      },
    ];

    it('should return deserialized moves', async () => {
      mockPrisma.move.findMany.mockResolvedValue(mockMoves);

      const result = await GamePersistenceService.getGameHistory('game-123');

      expect(result).toHaveLength(2);
      expect(result[0].type).toBe('place_ring');
      expect(result[0].player).toBe(1);
      expect(result[1].moveNumber).toBe(2);
    });

    it('should order moves by moveNumber ascending', async () => {
      mockPrisma.move.findMany.mockResolvedValue(mockMoves);

      await GamePersistenceService.getGameHistory('game-123');

      expect(mockPrisma.move.findMany).toHaveBeenCalledWith({
        where: { gameId: 'game-123' },
        orderBy: { moveNumber: 'asc' },
      });
    });
  });

  describe('finishGame', () => {
    it('should update game status to completed', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ gameState: '{}' });
      mockPrisma.game.update.mockResolvedValue({ id: 'game-123' });

      await GamePersistenceService.finishGame('game-123', 'winner-id');

      expect(mockPrisma.game.update).toHaveBeenCalledWith({
        where: { id: 'game-123' },
        data: expect.objectContaining({
          status: 'completed',
          winnerId: 'winner-id',
          endedAt: expect.any(Date),
        }),
      });
    });

    it('should include final state when provided', async () => {
      const mockFinalState = {
        id: 'game-123',
        board: {
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
          size: 8,
          type: 'square8' as BoardType,
        },
      } as unknown as GameState;

      mockPrisma.game.findUnique.mockResolvedValue({ gameState: '{}' });
      mockPrisma.game.update.mockResolvedValue({ id: 'game-123' });

      await GamePersistenceService.finishGame('game-123', null, mockFinalState);

      const updateCall = mockPrisma.game.update.mock.calls[0][0];
      expect(updateCall.data.finalState).toBeDefined();
    });

    it('should include game result when provided', async () => {
      const mockResult: GameResult = {
        winner: 1,
        reason: 'ring_elimination',
        finalScore: {
          ringsEliminated: { 1: 0, 2: 10 },
          territorySpaces: { 1: 5, 2: 3 },
          ringsRemaining: { 1: 18, 2: 8 },
        },
      };

      mockPrisma.game.findUnique.mockResolvedValue({ gameState: '{}' });
      mockPrisma.game.update.mockResolvedValue({ id: 'game-123' });

      await GamePersistenceService.finishGame('game-123', 'winner-id', undefined, mockResult);

      const updateCall = mockPrisma.game.update.mock.calls[0][0];
      expect(updateCall.data.gameState).toContain('ring_elimination');
    });

    it('should log game completion', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ gameState: '{}' });
      mockPrisma.game.update.mockResolvedValue({ id: 'game-123' });

      await GamePersistenceService.finishGame('game-123', 'winner-id');

      expect(logger.info).toHaveBeenCalledWith(
        'Game finished',
        expect.objectContaining({
          gameId: 'game-123',
          winnerId: 'winner-id',
        })
      );
    });
  });

  describe('updateGameStatus', () => {
    it('should update game status', async () => {
      mockPrisma.game.update.mockResolvedValue({ id: 'game-123' });

      await GamePersistenceService.updateGameStatus('game-123', 'active');

      expect(mockPrisma.game.update).toHaveBeenCalledWith({
        where: { id: 'game-123' },
        data: expect.objectContaining({
          status: 'active',
        }),
      });
    });

    it('should set startedAt when transitioning to active', async () => {
      mockPrisma.game.update.mockResolvedValue({ id: 'game-123' });

      await GamePersistenceService.updateGameStatus('game-123', 'active');

      const updateCall = mockPrisma.game.update.mock.calls[0][0];
      expect(updateCall.data.startedAt).toBeInstanceOf(Date);
    });
  });

  describe('getUserGames', () => {
    it('should return games for user', async () => {
      const mockGames = [
        {
          id: 'game-1',
          boardType: 'square8',
          status: 'completed',
          player1Id: 'user-123',
          player2Id: 'other',
          player3Id: null,
          player4Id: null,
          maxPlayers: 2,
          winnerId: 'user-123',
          createdAt: new Date(),
          endedAt: new Date(),
          _count: { moves: 20 },
        },
      ];
      mockPrisma.game.findMany.mockResolvedValue(mockGames);

      const result = await GamePersistenceService.getUserGames('user-123');

      expect(result).toHaveLength(1);
      expect(result[0].id).toBe('game-1');
      expect(result[0].moveCount).toBe(20);
    });

    it('should respect limit parameter', async () => {
      mockPrisma.game.findMany.mockResolvedValue([]);

      await GamePersistenceService.getUserGames('user-123', 5);

      expect(mockPrisma.game.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 5,
        })
      );
    });
  });

  describe('getActiveGames', () => {
    it('should return list of active game IDs', async () => {
      mockPrisma.game.findMany.mockResolvedValue([{ id: 'game-1' }, { id: 'game-2' }]);

      const result = await GamePersistenceService.getActiveGames();

      expect(result).toEqual(['game-1', 'game-2']);
      expect(mockPrisma.game.findMany).toHaveBeenCalledWith({
        where: { status: 'active' },
        select: { id: true },
      });
    });
  });

  describe('gameExists', () => {
    it('should return true when game exists', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ id: 'game-123' });

      const result = await GamePersistenceService.gameExists('game-123');

      expect(result).toBe(true);
    });

    it('should return false when game does not exist', async () => {
      mockPrisma.game.findUnique.mockResolvedValue(null);

      const result = await GamePersistenceService.gameExists('non-existent');

      expect(result).toBe(false);
    });

    it('should return false when database unavailable', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      const result = await GamePersistenceService.gameExists('game-123');

      expect(result).toBe(false);
    });
  });

  describe('deleteGame', () => {
    it('should delete game from database', async () => {
      mockPrisma.game.delete.mockResolvedValue({ id: 'game-123' });

      await GamePersistenceService.deleteGame('game-123');

      expect(mockPrisma.game.delete).toHaveBeenCalledWith({
        where: { id: 'game-123' },
      });
    });

    it('should log game deletion', async () => {
      mockPrisma.game.delete.mockResolvedValue({ id: 'game-123' });

      await GamePersistenceService.deleteGame('game-123');

      expect(logger.info).toHaveBeenCalledWith('Game deleted', { gameId: 'game-123' });
    });
  });

  describe('deserializeGameState', () => {
    it('should reconstruct Maps from serialized arrays', () => {
      const serialized = JSON.stringify({
        id: 'game-123',
        board: {
          stacks: [['0,0', { position: { x: 0, y: 0 }, rings: [1], stackHeight: 1 }]],
          markers: [['1,1', { player: 1, position: { x: 1, y: 1 } }]],
          collapsedSpaces: [['2,2', 1]],
          territories: [],
        },
      });

      const result = GamePersistenceService.deserializeGameState(serialized);

      expect(result.board.stacks).toBeInstanceOf(Map);
      expect(result.board.markers).toBeInstanceOf(Map);
      expect(result.board.collapsedSpaces).toBeInstanceOf(Map);
    });
  });
});
