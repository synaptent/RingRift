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

// Mock the RatingService so we can assert rating behaviour in finishGame tests
jest.mock('../../src/server/services/RatingService', () => ({
  RatingService: {
    processGameResult: jest.fn(),
  },
}));

import { getDatabaseClient } from '../../src/server/database/connection';
import { logger } from '../../src/server/utils/logger';
import { RatingService } from '../../src/server/services/RatingService';

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
      expect(createCall.data.player1).toEqual({ connect: { id: 'player1-id' } });
      expect(createCall.data.player2).toEqual({ connect: { id: 'player2-id' } });
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
          game: { connect: { id: 'game-123' } },
          player: { connect: { id: 'player-123' } },
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
          winner: { connect: { id: 'winner-id' } },
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

    it('invokes RatingService.processGameResult for rated games with a winner', async () => {
      const mockResult: GameResult = {
        winner: 1,
        reason: 'resignation',
        finalScore: {
          ringsEliminated: {},
          territorySpaces: {},
          ringsRemaining: {},
        },
      };

      mockPrisma.game.findUnique.mockResolvedValue({
        player1Id: 'p1',
        player2Id: 'p2',
        player3Id: null,
        player4Id: null,
        isRated: true,
        gameState: '{}',
      });
      mockPrisma.game.update.mockResolvedValue({ id: 'game-123' });

      const processSpy = RatingService.processGameResult as jest.Mock;

      await GamePersistenceService.finishGame('game-123', 'p1', undefined, mockResult);

      expect(processSpy).toHaveBeenCalledTimes(1);
      expect(processSpy).toHaveBeenCalledWith(
        'game-123',
        'p1',
        expect.arrayContaining(['p1', 'p2'])
      );
    });

    it('skips RatingService.processGameResult for rated abandonment draws without a winner', async () => {
      const mockResult: GameResult = {
        winner: undefined,
        reason: 'abandonment',
        finalScore: {
          ringsEliminated: {},
          territorySpaces: {},
          ringsRemaining: {},
        },
      };

      mockPrisma.game.findUnique.mockResolvedValue({
        player1Id: 'p1',
        player2Id: 'p2',
        player3Id: null,
        player4Id: null,
        isRated: true,
        gameState: '{}',
      });
      mockPrisma.game.update.mockResolvedValue({ id: 'game-123' });

      const processSpy = RatingService.processGameResult as jest.Mock;

      await GamePersistenceService.finishGame('game-123', null, undefined, mockResult);

      expect(processSpy).not.toHaveBeenCalled();
    });

    it('does not invoke RatingService.processGameResult for unrated games even with a winner', async () => {
      const mockResult: GameResult = {
        winner: 1,
        reason: 'resignation',
        finalScore: {
          ringsEliminated: {},
          territorySpaces: {},
          ringsRemaining: {},
        },
      };

      mockPrisma.game.findUnique.mockResolvedValue({
        player1Id: 'p1',
        player2Id: 'p2',
        player3Id: null,
        player4Id: null,
        isRated: false,
        gameState: '{}',
      });
      mockPrisma.game.update.mockResolvedValue({ id: 'game-123' });

      const processSpy = RatingService.processGameResult as jest.Mock;

      await GamePersistenceService.finishGame('game-123', 'p1', undefined, mockResult);

      expect(processSpy).not.toHaveBeenCalled();
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
        boardType: 'square8',
        numPlayers: 2,
        currentPlayer: 0,
        currentPhase: 'placement',
        turnNumber: 1,
        gameStatus: 'active',
        players: [
          { id: 'p1', color: 0 },
          { id: 'p2', color: 1 },
        ],
        board: {
          type: 'square8',
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

    it('should reject state with non-array board properties', () => {
      const serialized = JSON.stringify({
        id: 'game-123',
        boardType: 'square8',
        numPlayers: 2,
        currentPlayer: 0,
        currentPhase: 'placement',
        turnNumber: 1,
        gameStatus: 'active',
        players: [
          { id: 'p1', color: 0 },
          { id: 'p2', color: 1 },
        ],
        board: {
          type: 'square8',
          stacks: {},
          markers: {},
          collapsedSpaces: {},
          territories: {},
        },
      });

      // Schema requires arrays for Map reconstruction
      expect(() => GamePersistenceService.deserializeGameState(serialized)).toThrow(
        'Game state validation failed'
      );
    });

    it('should reject state without required properties', () => {
      const serialized = JSON.stringify({
        id: 'game-123',
        turnNumber: 5,
      });

      expect(() => GamePersistenceService.deserializeGameState(serialized)).toThrow(
        'Game state validation failed'
      );
    });
  });

  describe('createGame error handling', () => {
    it('should log error and rethrow on database error', async () => {
      const dbError = new Error('Database constraint violation');
      mockPrisma.game.create.mockRejectedValue(dbError);

      await expect(
        GamePersistenceService.createGame({
          boardType: 'square8',
          maxPlayers: 2,
          timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
          isRated: true,
        })
      ).rejects.toThrow('Database constraint violation');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to create game in database',
        expect.objectContaining({
          error: 'Database constraint violation',
        })
      );
    });
  });

  describe('loadGame error handling', () => {
    it('should log error and rethrow on database error', async () => {
      const dbError = new Error('Connection timeout');
      mockPrisma.game.findUnique.mockRejectedValue(dbError);

      await expect(GamePersistenceService.loadGame('game-123')).rejects.toThrow(
        'Connection timeout'
      );

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to load game from database',
        expect.objectContaining({
          gameId: 'game-123',
          error: 'Connection timeout',
        })
      );
    });
  });

  describe('deserializeMove optional fields', () => {
    it('should deserialize moveData with buildAmount and placedOnStack', async () => {
      mockPrisma.move.findMany.mockResolvedValue([
        {
          id: 'm1',
          gameId: 'game-123',
          playerId: 'p1',
          moveNumber: 1,
          position: { to: { x: 3, y: 3 } },
          moveType: 'build_stack',
          moveData: {
            id: 'm1',
            type: 'build_stack',
            player: 1,
            to: { x: 3, y: 3 },
            buildAmount: 3,
            placedOnStack: true,
          },
          timestamp: new Date(),
        },
      ]);

      const result = await GamePersistenceService.getGameHistory('game-123');

      expect(result).toHaveLength(1);
      expect(result[0].buildAmount).toBe(3);
      expect(result[0].placedOnStack).toBe(true);
    });

    it('should deserialize moveData with placementCount and stackMoved', async () => {
      const mockStack = { position: { x: 1, y: 1 }, rings: [1, 1], controllingPlayer: 1 };
      mockPrisma.move.findMany.mockResolvedValue([
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
            placementCount: 2,
            stackMoved: mockStack,
          },
          timestamp: new Date(),
        },
      ]);

      const result = await GamePersistenceService.getGameHistory('game-123');

      expect(result[0].placementCount).toBe(2);
      expect(result[0].stackMoved).toEqual(mockStack);
    });

    it('should deserialize moveData with distance fields', async () => {
      mockPrisma.move.findMany.mockResolvedValue([
        {
          id: 'm1',
          gameId: 'game-123',
          playerId: 'p1',
          moveNumber: 1,
          position: { from: { x: 1, y: 1 }, to: { x: 3, y: 3 } },
          moveType: 'move_stack',
          moveData: {
            id: 'm1',
            type: 'move_stack',
            player: 1,
            from: { x: 1, y: 1 },
            to: { x: 3, y: 3 },
            minimumDistance: 2,
            actualDistance: 4,
            markerLeft: { x: 1, y: 1 },
          },
          timestamp: new Date(),
        },
      ]);

      const result = await GamePersistenceService.getGameHistory('game-123');

      expect(result[0].minimumDistance).toBe(2);
      expect(result[0].actualDistance).toBe(4);
      expect(result[0].markerLeft).toEqual({ x: 1, y: 1 });
    });

    it('should deserialize moveData with capture fields', async () => {
      const capturedStack = { position: { x: 2, y: 2 }, rings: [2], controllingPlayer: 2 };
      mockPrisma.move.findMany.mockResolvedValue([
        {
          id: 'm1',
          gameId: 'game-123',
          playerId: 'p1',
          moveNumber: 1,
          position: { from: { x: 1, y: 1 }, to: { x: 3, y: 3 } },
          moveType: 'overtaking_capture',
          moveData: {
            id: 'm1',
            type: 'overtaking_capture',
            player: 1,
            from: { x: 1, y: 1 },
            to: { x: 3, y: 3 },
            captureType: 'overtaking',
            captureTarget: { x: 2, y: 2 },
            capturedStacks: [capturedStack],
            captureChain: [{ x: 2, y: 2 }],
            overtakenRings: [2, 2],
          },
          timestamp: new Date(),
        },
      ]);

      const result = await GamePersistenceService.getGameHistory('game-123');

      expect(result[0].captureType).toBe('overtaking');
      expect(result[0].captureTarget).toEqual({ x: 2, y: 2 });
      expect(result[0].capturedStacks).toEqual([capturedStack]);
      expect(result[0].captureChain).toEqual([{ x: 2, y: 2 }]);
      expect(result[0].overtakenRings).toEqual([2, 2]);
    });

    it('should deserialize moveData with line and territory fields', async () => {
      const mockLine = {
        positions: [
          { x: 0, y: 0 },
          { x: 1, y: 1 },
        ],
        player: 1,
      };
      const mockTerritory = { positions: [{ x: 2, y: 2 }], owner: 1 };
      mockPrisma.move.findMany.mockResolvedValue([
        {
          id: 'm1',
          gameId: 'game-123',
          playerId: 'p1',
          moveNumber: 1,
          position: { to: { x: 3, y: 3 } },
          moveType: 'process_line',
          moveData: {
            id: 'm1',
            type: 'process_line',
            player: 1,
            to: { x: 3, y: 3 },
            formedLines: [mockLine],
            collapsedMarkers: [{ x: 1, y: 1 }],
            claimedTerritory: [mockTerritory],
            disconnectedRegions: [mockTerritory],
          },
          timestamp: new Date(),
        },
      ]);

      const result = await GamePersistenceService.getGameHistory('game-123');

      expect(result[0].formedLines).toEqual([mockLine]);
      expect(result[0].collapsedMarkers).toEqual([{ x: 1, y: 1 }]);
      expect(result[0].claimedTerritory).toEqual([mockTerritory]);
      expect(result[0].disconnectedRegions).toEqual([mockTerritory]);
    });

    it('should deserialize moveData with elimination fields', async () => {
      mockPrisma.move.findMany.mockResolvedValue([
        {
          id: 'm1',
          gameId: 'game-123',
          playerId: 'p1',
          moveNumber: 1,
          position: { to: { x: 3, y: 3 } },
          moveType: 'eliminate_rings_from_stack',
          moveData: {
            id: 'm1',
            type: 'eliminate_rings_from_stack',
            player: 1,
            to: { x: 3, y: 3 },
            eliminatedRings: [{ player: 2, count: 3 }],
            eliminationFromStack: {
              position: { x: 3, y: 3 },
              capHeight: 3,
              totalHeight: 5,
            },
          },
          timestamp: new Date(),
        },
      ]);

      const result = await GamePersistenceService.getGameHistory('game-123');

      expect(result[0].eliminatedRings).toEqual([{ player: 2, count: 3 }]);
      expect(result[0].eliminationFromStack).toEqual({
        position: { x: 3, y: 3 },
        capHeight: 3,
        totalHeight: 5,
      });
    });
  });

  describe('gameExists error handling', () => {
    it('should return false when database query throws', async () => {
      mockPrisma.game.findUnique.mockRejectedValue(new Error('DB error'));

      const result = await GamePersistenceService.gameExists('game-123');

      expect(result).toBe(false);
    });
  });

  describe('serializeMove decisionAutoResolved', () => {
    it('should include decisionAutoResolved in serialized move when present', async () => {
      const moveWithAutoResolved = {
        id: 'm1',
        type: 'choose_territory_option' as const,
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 100,
        moveNumber: 1,
        decisionAutoResolved: { reason: 'single_option', timestamp: Date.now() },
      };

      mockPrisma.move.create.mockResolvedValue({ id: 'm1' });

      const saveMoveData: SaveMoveData = {
        gameId: 'game-123',
        playerId: 'player-1',
        moveNumber: 1,
        move: moveWithAutoResolved as any,
      };

      await GamePersistenceService.saveMove(saveMoveData);

      const createCall = mockPrisma.move.create.mock.calls[0][0];
      expect(createCall.data.moveData).toHaveProperty('decisionAutoResolved');
    });
  });

  describe('getGameHistory error handling', () => {
    it('should throw when database unavailable', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(GamePersistenceService.getGameHistory('game-123')).rejects.toThrow(
        'Database not available'
      );
    });

    it('should log error and rethrow on database error', async () => {
      const dbError = new Error('Query failed');
      mockPrisma.move.findMany.mockRejectedValue(dbError);

      await expect(GamePersistenceService.getGameHistory('game-123')).rejects.toThrow(
        'Query failed'
      );

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get game history',
        expect.objectContaining({
          gameId: 'game-123',
          error: 'Query failed',
        })
      );
    });

    it('should use fallback deserialization when moveData is missing', async () => {
      // Move without moveData - triggers fallback path
      mockPrisma.move.findMany.mockResolvedValue([
        {
          id: 'm1',
          gameId: 'game-123',
          playerId: 'p1',
          moveNumber: 1,
          position: { to: { x: 3, y: 3 } },
          moveType: 'place_ring',
          moveData: null,
          timestamp: new Date(),
        },
      ]);

      const result = await GamePersistenceService.getGameHistory('game-123');

      expect(result).toHaveLength(1);
      expect(result[0].type).toBe('place_ring');
      expect(result[0].to).toEqual({ x: 3, y: 3 });
      expect(result[0].player).toBe(0); // Default when not in moveData
    });

    it('should include from position in fallback deserialization', async () => {
      mockPrisma.move.findMany.mockResolvedValue([
        {
          id: 'm1',
          gameId: 'game-123',
          playerId: 'p1',
          moveNumber: 1,
          position: { from: { x: 2, y: 2 }, to: { x: 3, y: 3 } },
          moveType: 'move_stack',
          moveData: null,
          timestamp: new Date(),
        },
      ]);

      const result = await GamePersistenceService.getGameHistory('game-123');

      expect(result[0].from).toEqual({ x: 2, y: 2 });
    });
  });

  describe('finishGame error handling', () => {
    it('should throw when database unavailable', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(GamePersistenceService.finishGame('game-123', 'winner-id')).rejects.toThrow(
        'Database not available'
      );
    });

    it('should throw when game not found', async () => {
      mockPrisma.game.findUnique.mockResolvedValue(null);

      await expect(GamePersistenceService.finishGame('game-123', 'winner-id')).rejects.toThrow(
        'Game not found: game-123'
      );
    });

    it('should log error and rethrow on database error during update', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({ gameState: '{}' });
      mockPrisma.game.update.mockRejectedValue(new Error('Update failed'));

      await expect(GamePersistenceService.finishGame('game-123', 'winner-id')).rejects.toThrow(
        'Update failed'
      );

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to finish game',
        expect.objectContaining({
          gameId: 'game-123',
          error: 'Update failed',
        })
      );
    });

    it('should log but not fail when RatingService.processGameResult throws', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({
        player1Id: 'p1',
        player2Id: 'p2',
        player3Id: null,
        player4Id: null,
        isRated: true,
        gameState: '{}',
      });
      mockPrisma.game.update.mockResolvedValue({ id: 'game-123' });

      const processSpy = RatingService.processGameResult as jest.Mock;
      processSpy.mockRejectedValue(new Error('Rating calculation failed'));

      const result = await GamePersistenceService.finishGame('game-123', 'p1');

      expect(result).toEqual({}); // Should succeed despite rating error
      expect(logger.error).toHaveBeenCalledWith(
        'Failed to update ratings after game completion',
        expect.objectContaining({
          gameId: 'game-123',
          error: 'Rating calculation failed',
        })
      );
    });

    it('should process ratings for 3-4 player games', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({
        player1Id: 'p1',
        player2Id: 'p2',
        player3Id: 'p3',
        player4Id: 'p4',
        isRated: true,
        gameState: '{}',
      });
      mockPrisma.game.update.mockResolvedValue({ id: 'game-123' });

      const processSpy = RatingService.processGameResult as jest.Mock;
      processSpy.mockResolvedValue([]);

      await GamePersistenceService.finishGame('game-123', 'p1');

      expect(processSpy).toHaveBeenCalledWith(
        'game-123',
        'p1',
        expect.arrayContaining(['p1', 'p2', 'p3', 'p4'])
      );
    });

    it('should skip rating update when fewer than 2 players', async () => {
      mockPrisma.game.findUnique.mockResolvedValue({
        player1Id: 'p1',
        player2Id: null,
        player3Id: null,
        player4Id: null,
        isRated: true,
        gameState: '{}',
      });
      mockPrisma.game.update.mockResolvedValue({ id: 'game-123' });

      const processSpy = RatingService.processGameResult as jest.Mock;

      await GamePersistenceService.finishGame('game-123', 'p1');

      expect(processSpy).not.toHaveBeenCalled();
    });
  });

  describe('updateGameStatus error handling', () => {
    it('should throw when database unavailable', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(GamePersistenceService.updateGameStatus('game-123', 'active')).rejects.toThrow(
        'Database not available'
      );
    });

    it('should log error and rethrow on database error', async () => {
      mockPrisma.game.update.mockRejectedValue(new Error('Update failed'));

      await expect(GamePersistenceService.updateGameStatus('game-123', 'active')).rejects.toThrow(
        'Update failed'
      );

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to update game status',
        expect.objectContaining({
          gameId: 'game-123',
          status: 'active',
          error: 'Update failed',
        })
      );
    });
  });

  describe('updateGameState', () => {
    it('should silently return when database unavailable', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      // Should not throw
      await expect(
        GamePersistenceService.updateGameState('game-123', {} as GameState)
      ).resolves.toBeUndefined();
    });

    it('should log error but not throw on database error', async () => {
      const mockState = {
        board: {
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
        },
      } as unknown as GameState;

      mockPrisma.game.update.mockRejectedValue(new Error('Update failed'));

      // Should not throw
      await GamePersistenceService.updateGameState('game-123', mockState);

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to update game state',
        expect.objectContaining({
          gameId: 'game-123',
          error: 'Update failed',
        })
      );
    });
  });

  describe('getUserGames error handling', () => {
    it('should throw when database unavailable', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(GamePersistenceService.getUserGames('user-123')).rejects.toThrow(
        'Database not available'
      );
    });

    it('should log error and rethrow on database error', async () => {
      mockPrisma.game.findMany.mockRejectedValue(new Error('Query failed'));

      await expect(GamePersistenceService.getUserGames('user-123')).rejects.toThrow('Query failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get user games',
        expect.objectContaining({
          userId: 'user-123',
          error: 'Query failed',
        })
      );
    });
  });

  describe('getActiveGames error handling', () => {
    it('should throw when database unavailable', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(GamePersistenceService.getActiveGames()).rejects.toThrow(
        'Database not available'
      );
    });

    it('should log error and rethrow on database error', async () => {
      mockPrisma.game.findMany.mockRejectedValue(new Error('Query failed'));

      await expect(GamePersistenceService.getActiveGames()).rejects.toThrow('Query failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to get active games',
        expect.objectContaining({
          error: 'Query failed',
        })
      );
    });
  });

  describe('deleteGame error handling', () => {
    it('should throw when database unavailable', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(GamePersistenceService.deleteGame('game-123')).rejects.toThrow(
        'Database not available'
      );
    });

    it('should log error and rethrow on database error', async () => {
      mockPrisma.game.delete.mockRejectedValue(new Error('Delete failed'));

      await expect(GamePersistenceService.deleteGame('game-123')).rejects.toThrow('Delete failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to delete game',
        expect.objectContaining({
          gameId: 'game-123',
          error: 'Delete failed',
        })
      );
    });
  });
});
