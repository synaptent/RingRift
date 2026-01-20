/**
 * Tests for GameSession move handling operations.
 *
 * This file covers:
 * - handlePlayerMove validation and execution
 * - handlePlayerMoveById
 * - Move persistence
 * - Broadcast updates after moves
 * - AI turn triggering after human moves
 * - Engine selection shadow mode
 */

import { Server as SocketIOServer } from 'socket.io';
import { GameSession } from '../../src/server/game/GameSession';
import type {
  GameState,
  Move,
  Position,
  Player,
  GameResult,
  MoveType,
} from '../../src/shared/types/game';
import { GamePersistenceService } from '../../src/server/services/GamePersistenceService';
import type { AuthenticatedSocket } from '../../src/server/websocket/server';

// Mock database
const mockGameFindUnique = jest.fn();
const mockGameUpdate = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => ({
    game: {
      findUnique: mockGameFindUnique,
      update: mockGameUpdate,
    },
    move: {
      create: jest.fn(),
      findMany: jest.fn().mockResolvedValue([]),
    },
  })),
}));

jest.mock('../../src/server/services/GamePersistenceService', () => ({
  GamePersistenceService: {
    saveMove: jest.fn(),
    finishGame: jest.fn().mockResolvedValue({}),
    updateGameStateWithInternal: jest.fn().mockResolvedValue(undefined),
  },
}));

jest.mock('../../src/server/services/PythonRulesClient', () => ({
  PythonRulesClient: jest.fn().mockImplementation(() => ({
    evaluateMove: jest.fn(),
    healthCheck: jest.fn(),
  })),
}));

jest.mock('../../src/server/game/ai/AIEngine', () => ({
  globalAIEngine: {
    createAI: jest.fn(),
    createAIFromProfile: jest.fn(),
    getAIConfig: jest.fn(),
    getAIMove: jest.fn(),
    chooseLocalMoveFromCandidates: jest.fn(),
    getLocalFallbackMove: jest.fn(),
    getDiagnostics: jest.fn(() => ({
      serviceFailureCount: 0,
      localFallbackCount: 0,
    })),
  },
}));

jest.mock('../../src/server/services/AIUserService', () => ({
  getOrCreateAIUser: jest.fn(() => Promise.resolve({ id: 'ai-user-id' })),
}));

jest.mock('../../src/server/services/MetricsService', () => ({
  getMetricsService: () => ({
    recordAITurnRequestTerminal: jest.fn(),
    recordGameSessionStatusTransition: jest.fn(),
    recordAbnormalTermination: jest.fn(),
    updateGameSessionStatusCurrent: jest.fn(),
    recordMoveApplied: jest.fn(),
    recordMoveRejected: jest.fn(),
  }),
}));

const createMockIo = (): jest.Mocked<SocketIOServer> =>
  ({
    to: jest.fn().mockReturnThis(),
    emit: jest.fn(),
    sockets: {
      adapter: {
        rooms: new Map(),
      },
      sockets: new Map(),
    },
  }) as any;

const createMockSocket = (userId: string): AuthenticatedSocket =>
  ({
    id: 'socket-id',
    userId,
    username: 'TestUser',
    emit: jest.fn(),
    join: jest.fn(),
    leave: jest.fn(),
    rooms: new Set(),
  }) as any;

const mockSaveMove = GamePersistenceService.saveMove as jest.MockedFunction<any>;
const mockFinishGame = GamePersistenceService.finishGame as jest.MockedFunction<any>;

describe('GameSession Move Handling', () => {
  const now = new Date();

  function createBaseGameState(overrides: Partial<GameState> = {}): GameState {
    return {
      id: 'test-game-id',
      boardType: 'square8',
      board: {
        type: 'square8',
        size: 8,
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: { 1: 0, 2: 0 },
      },
      players: [
        {
          id: 'player-1',
          username: 'Player1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'player-2',
          username: 'Player2',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      currentPlayer: 1,
      currentPhase: 'ring_placement',
      moveHistory: [],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      spectators: [],
      gameStatus: 'active',
      createdAt: now,
      lastMoveAt: now,
      isRated: true,
      maxPlayers: 2,
      totalRingsInPlay: 36,
      totalRingsEliminated: 0,
      victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
      territoryVictoryThreshold: 33,
      ...(overrides as any),
    };
  }

  beforeEach(() => {
    jest.clearAllMocks();
    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      status: 'active',
    });
  });

  describe('handlePlayerMove', () => {
    describe('Authentication and Authorization', () => {
      it('rejects move when socket is not authenticated', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const socket = { ...createMockSocket('player-1'), userId: undefined } as any;

        await expect(
          session.handlePlayerMove(socket, {
            position: { x: 0, y: 0 },
            moveType: 'ring_placement',
          })
        ).rejects.toThrow('Socket not authenticated');
      });

      it('rejects move when game is not found', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        mockGameFindUnique.mockResolvedValue(null);

        const socket = createMockSocket('player-1');

        await expect(
          session.handlePlayerMove(socket, {
            position: { x: 0, y: 0 },
            moveType: 'ring_placement',
          })
        ).rejects.toThrow('Game not found');
      });

      it('rejects move when game is not active', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        mockGameFindUnique.mockResolvedValue({
          id: 'test-game-id',
          status: 'completed',
        });

        const state = createBaseGameState();
        (session as any).gameEngine = {
          getGameState: jest.fn(() => state),
        };

        const socket = createMockSocket('player-1');

        await expect(
          session.handlePlayerMove(socket, {
            position: { x: 0, y: 0 },
            moveType: 'ring_placement',
          })
        ).rejects.toThrow('Game is not active');
      });

      it('rejects move when user is not a player', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState();
        (session as any).gameEngine = {
          getGameState: jest.fn(() => state),
        };

        const socket = createMockSocket('not-a-player');

        await expect(
          session.handlePlayerMove(socket, {
            position: { x: 0, y: 0 },
            moveType: 'ring_placement',
          })
        ).rejects.toThrow('Current user is not a player in this game');
      });

      it('rejects move from spectator', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState({
          spectators: ['spectator-1'],
        });
        (session as any).gameEngine = {
          getGameState: jest.fn(() => state),
        };

        const socket = createMockSocket('spectator-1');

        await expect(
          session.handlePlayerMove(socket, {
            position: { x: 0, y: 0 },
            moveType: 'ring_placement',
          })
        ).rejects.toThrow('Spectators cannot make moves');
      });
    });

    describe('Move Position Parsing', () => {
      it('parses position from JSON string with to field', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState();
        const applyMoveMock = jest.fn().mockResolvedValue({
          success: true,
          state: { ...state, currentPlayer: 2 },
        });

        (session as any).gameEngine = {
          getGameState: jest.fn(() => state),
          getValidMoves: jest.fn(() => []),
        };
        (session as any).rulesFacade = {
          applyMove: applyMoveMock,
        };

        jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
        jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
        jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

        const socket = createMockSocket('player-1');

        await session.handlePlayerMove(socket, {
          position: JSON.stringify({ to: { x: 3, y: 3 } }),
          moveType: 'ring_placement',
        });

        expect(applyMoveMock).toHaveBeenCalledWith(
          expect.objectContaining({
            player: 1,
            type: 'ring_placement',
            to: { x: 3, y: 3 },
          })
        );
      });

      it('parses position from JSON string as direct coordinates', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState();
        const applyMoveMock = jest.fn().mockResolvedValue({
          success: true,
          state: { ...state, currentPlayer: 2 },
        });

        (session as any).gameEngine = {
          getGameState: jest.fn(() => state),
          getValidMoves: jest.fn(() => []),
        };
        (session as any).rulesFacade = {
          applyMove: applyMoveMock,
        };

        jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
        jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
        jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

        const socket = createMockSocket('player-1');

        // When position is just { x, y }, it's treated as 'to'
        await session.handlePlayerMove(socket, {
          position: JSON.stringify({ x: 4, y: 4 }),
          moveType: 'ring_placement',
        });

        expect(applyMoveMock).toHaveBeenCalledWith(
          expect.objectContaining({
            to: { x: 4, y: 4 },
          })
        );
      });

      it('handles from/to positions for movement', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState({ currentPhase: 'movement' });
        state.players[0].ringsInHand = 0;
        state.players[1].ringsInHand = 0;

        const applyMoveMock = jest.fn().mockResolvedValue({
          success: true,
          state: { ...state, currentPlayer: 2 },
        });

        (session as any).gameEngine = {
          getGameState: jest.fn(() => state),
          getValidMoves: jest.fn(() => []),
        };
        (session as any).rulesFacade = {
          applyMove: applyMoveMock,
        };

        jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
        jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
        jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

        const socket = createMockSocket('player-1');

        await session.handlePlayerMove(socket, {
          position: JSON.stringify({ from: { x: 0, y: 0 }, to: { x: 1, y: 1 } }),
          moveType: 'movement',
        });

        expect(applyMoveMock).toHaveBeenCalledWith(
          expect.objectContaining({
            from: { x: 0, y: 0 },
            to: { x: 1, y: 1 },
            type: 'movement',
          })
        );
      });

      it('throws on invalid JSON position string', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState();
        (session as any).gameEngine = {
          getGameState: jest.fn(() => state),
        };

        const socket = createMockSocket('player-1');

        await expect(
          session.handlePlayerMove(socket, {
            position: 'not-valid-json',
            moveType: 'ring_placement',
          })
        ).rejects.toThrow('Invalid move position payload');
      });

      it('falls back to whole object as position when no to field', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState();
        const applyMoveMock = jest.fn().mockResolvedValue({
          success: true,
          state: { ...state, currentPlayer: 2 },
        });

        (session as any).gameEngine = {
          getGameState: jest.fn(() => state),
          getValidMoves: jest.fn(() => []),
        };
        (session as any).rulesFacade = {
          applyMove: applyMoveMock,
        };

        jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
        jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
        jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

        const socket = createMockSocket('player-1');

        // When position has only 'from' field, the whole object becomes 'to'
        // This tests the fallback behavior: (parsed.to ?? parsed)
        await session.handlePlayerMove(socket, {
          position: JSON.stringify({ from: { x: 0, y: 0 } }),
          moveType: 'movement',
        });

        // The whole object {from: ...} becomes 'to'
        expect(applyMoveMock).toHaveBeenCalledWith(
          expect.objectContaining({
            from: { x: 0, y: 0 },
            to: { from: { x: 0, y: 0 } },
          })
        );
      });
    });

    describe('Move Validation and Execution', () => {
      it('rejects invalid move from engine', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState();
        const applyMoveMock = jest.fn().mockResolvedValue({
          success: false,
          error: 'Position already occupied',
        });

        (session as any).gameEngine = {
          getGameState: jest.fn(() => state),
          getValidMoves: jest.fn(() => []),
        };
        (session as any).rulesFacade = {
          applyMove: applyMoveMock,
        };

        const socket = createMockSocket('player-1');

        await expect(
          session.handlePlayerMove(socket, {
            position: JSON.stringify({ x: 0, y: 0 }),
            moveType: 'ring_placement',
          })
        ).rejects.toThrow('Position already occupied');
      });

      it('persists successful move', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState();
        const updatedState = {
          ...state,
          currentPlayer: 2,
          moveHistory: [
            {
              id: 'move-1',
              type: 'ring_placement',
              player: 1,
              to: { x: 0, y: 0 },
              moveNumber: 1,
              timestamp: now,
              thinkTime: 100,
            },
          ],
        };

        (session as any).gameEngine = {
          getGameState: jest.fn().mockReturnValueOnce(state).mockReturnValue(updatedState),
          getValidMoves: jest.fn(() => []),
          getInternalStateForPersistence: jest.fn().mockReturnValue({}),
        };
        (session as any).rulesFacade = {
          applyMove: jest.fn().mockResolvedValue({
            success: true,
            state: updatedState,
          }),
        };

        jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
        jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

        const socket = createMockSocket('player-1');

        await session.handlePlayerMove(socket, {
          position: JSON.stringify({ x: 0, y: 0 }),
          moveType: 'ring_placement',
        });

        expect(mockSaveMove).toHaveBeenCalled();
      });

      it('broadcasts update after successful move', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState();
        const updatedState = { ...state, currentPlayer: 2, moveHistory: [] };

        (session as any).gameEngine = {
          getGameState: jest.fn().mockReturnValue(updatedState),
          getValidMoves: jest.fn(() => []),
        };
        (session as any).rulesFacade = {
          applyMove: jest.fn().mockResolvedValue({
            success: true,
            state: updatedState,
          }),
        };

        const broadcastSpy = jest
          .spyOn(session as any, 'broadcastUpdate')
          .mockResolvedValue(undefined);
        jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

        const socket = createMockSocket('player-1');

        await session.handlePlayerMove(socket, {
          position: JSON.stringify({ x: 0, y: 0 }),
          moveType: 'ring_placement',
        });

        expect(broadcastSpy).toHaveBeenCalled();
      });

      it('triggers AI turn after human move when applicable', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState();
        state.players[1] = { ...state.players[1], type: 'ai' };

        const updatedState = { ...state, currentPlayer: 2, moveHistory: [] };

        (session as any).gameEngine = {
          getGameState: jest.fn().mockReturnValue(updatedState),
          getValidMoves: jest.fn(() => []),
        };
        (session as any).rulesFacade = {
          applyMove: jest.fn().mockResolvedValue({
            success: true,
            state: updatedState,
          }),
        };

        jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
        const aiTurnSpy = jest
          .spyOn(session as any, 'maybePerformAITurn')
          .mockResolvedValue(undefined);

        const socket = createMockSocket('player-1');

        await session.handlePlayerMove(socket, {
          position: JSON.stringify({ x: 0, y: 0 }),
          moveType: 'ring_placement',
        });

        expect(aiTurnSpy).toHaveBeenCalled();
      });
    });

    describe('Game Completion', () => {
      it('finishes game when result is returned', async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState();
        const gameResult: GameResult = {
          winner: 1,
          reason: 'ring_elimination',
          finalScore: {
            ringsEliminated: { 1: 5, 2: 19 },
            territorySpaces: { 1: 0, 2: 0 },
            ringsRemaining: { 1: 13, 2: 0 },
          },
        };

        const updatedState = {
          ...state,
          gameStatus: 'completed' as const,
          moveHistory: [
            {
              id: 'move-1',
              type: 'ring_placement' as const,
              player: 1,
              to: { x: 0, y: 0 },
              moveNumber: 1,
              timestamp: now,
              thinkTime: 100,
            },
          ],
        };

        (session as any).gameEngine = {
          getGameState: jest.fn().mockReturnValue(updatedState),
          getValidMoves: jest.fn(() => []),
          getInternalStateForPersistence: jest.fn().mockReturnValue({}),
        };
        (session as any).rulesFacade = {
          applyMove: jest.fn().mockResolvedValue({
            success: true,
            state: updatedState,
            gameResult,
          }),
        };

        jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
        jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

        const socket = createMockSocket('player-1');

        await session.handlePlayerMove(socket, {
          position: JSON.stringify({ x: 0, y: 0 }),
          moveType: 'ring_placement',
        });

        expect(mockFinishGame).toHaveBeenCalled();
      });
    });
  });

  describe('handlePlayerMoveById', () => {
    it('rejects when socket is not authenticated', async () => {
      const io = createMockIo();
      const session = new GameSession('test-game-id', io, {} as any, new Map());

      const socket = { ...createMockSocket('player-1'), userId: undefined } as any;

      await expect(session.handlePlayerMoveById(socket, 'move-id-1')).rejects.toThrow(
        'Socket not authenticated'
      );
    });

    it('rejects when game is not found', async () => {
      const io = createMockIo();
      const session = new GameSession('test-game-id', io, {} as any, new Map());

      mockGameFindUnique.mockResolvedValue(null);

      const socket = createMockSocket('player-1');

      await expect(session.handlePlayerMoveById(socket, 'move-id-1')).rejects.toThrow(
        'Game not found'
      );
    });

    it('rejects when game is not active', async () => {
      const io = createMockIo();
      const session = new GameSession('test-game-id', io, {} as any, new Map());

      mockGameFindUnique.mockResolvedValue({
        id: 'test-game-id',
        status: 'completed',
      });

      const socket = createMockSocket('player-1');

      await expect(session.handlePlayerMoveById(socket, 'move-id-1')).rejects.toThrow(
        'Game is not active'
      );
    });

    it('rejects when user is not a player', async () => {
      const io = createMockIo();
      const session = new GameSession('test-game-id', io, {} as any, new Map());

      const state = createBaseGameState();
      (session as any).gameEngine = {
        getGameState: jest.fn(() => state),
      };

      const socket = createMockSocket('not-a-player');

      await expect(session.handlePlayerMoveById(socket, 'move-id-1')).rejects.toThrow(
        'Current user is not a player in this game'
      );
    });

    it('rejects move from spectator', async () => {
      const io = createMockIo();
      const session = new GameSession('test-game-id', io, {} as any, new Map());

      const state = createBaseGameState({
        spectators: ['spectator-1'],
      });
      (session as any).gameEngine = {
        getGameState: jest.fn(() => state),
      };

      const socket = createMockSocket('spectator-1');

      await expect(session.handlePlayerMoveById(socket, 'move-id-1')).rejects.toThrow(
        'Spectators cannot make moves'
      );
    });

    it('executes valid move by ID', async () => {
      const io = createMockIo();
      const session = new GameSession('test-game-id', io, {} as any, new Map());

      const state = createBaseGameState();
      const updatedState = { ...state, currentPlayer: 2, moveHistory: [] };

      (session as any).gameEngine = {
        getGameState: jest.fn().mockReturnValue(updatedState),
        getValidMoves: jest.fn(() => []),
      };
      (session as any).rulesFacade = {
        applyMoveById: jest.fn().mockResolvedValue({
          success: true,
          state: updatedState,
        }),
      };

      jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
      jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

      const socket = createMockSocket('player-1');

      await session.handlePlayerMoveById(socket, 'move-id-1');

      expect((session as any).rulesFacade.applyMoveById).toHaveBeenCalledWith(1, 'move-id-1');
    });

    it('rejects invalid move by ID', async () => {
      const io = createMockIo();
      const session = new GameSession('test-game-id', io, {} as any, new Map());

      const state = createBaseGameState();
      (session as any).gameEngine = {
        getGameState: jest.fn(() => state),
      };
      (session as any).rulesFacade = {
        applyMoveById: jest.fn().mockResolvedValue({
          success: false,
          error: 'Invalid move selection',
        }),
      };

      const socket = createMockSocket('player-1');

      await expect(session.handlePlayerMoveById(socket, 'invalid-move-id')).rejects.toThrow(
        'Invalid move selection'
      );
    });
  });

  describe('Move Types', () => {
    const moveTypes: MoveType[] = [
      'ring_placement',
      'movement',
      'capture',
      'chain_capture',
      'process_line',
      'line_reward',
      'process_territory',
      'territory_decision',
      'eliminate_rings_from_stack',
      'pass',
    ];

    moveTypes.forEach((moveType) => {
      it(`handles ${moveType} move type`, async () => {
        const io = createMockIo();
        const session = new GameSession('test-game-id', io, {} as any, new Map());

        const state = createBaseGameState({ currentPhase: 'movement' });
        state.players[0].ringsInHand = 0;

        (session as any).gameEngine = {
          getGameState: jest.fn(() => state),
          getValidMoves: jest.fn(() => []),
        };
        (session as any).rulesFacade = {
          applyMove: jest.fn().mockResolvedValue({
            success: true,
            state: { ...state, currentPlayer: 2 },
          }),
        };

        jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
        jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
        jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

        const socket = createMockSocket('player-1');

        await session.handlePlayerMove(socket, {
          position: JSON.stringify({ x: 0, y: 0 }),
          moveType,
        });

        expect((session as any).rulesFacade.applyMove).toHaveBeenCalledWith(
          expect.objectContaining({
            type: moveType,
          })
        );
      });
    });
  });
});

describe('Move Handling Edge Cases', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGameFindUnique.mockResolvedValue({
      id: 'test-game-id',
      status: 'active',
    });
  });

  it('handles hexagonal coordinates with z component', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as any, new Map());

    const state = {
      id: 'test-game-id',
      boardType: 'hexagonal',
      board: {
        type: 'hexagonal',
        size: 5,
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: { 1: 0, 2: 0 },
      },
      players: [
        {
          id: 'player-1',
          username: 'Player1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'player-2',
          username: 'Player2',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      currentPlayer: 1,
      currentPhase: 'ring_placement',
      moveHistory: [],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      spectators: [],
      gameStatus: 'active',
    } as any;

    const applyMoveMock = jest.fn().mockResolvedValue({
      success: true,
      state: { ...state, currentPlayer: 2 },
    });

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      getValidMoves: jest.fn(() => []),
    };
    (session as any).rulesFacade = {
      applyMove: applyMoveMock,
    };

    jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

    const socket = createMockSocket('player-1');

    await session.handlePlayerMove(socket, {
      position: JSON.stringify({ x: 1, y: -1, z: 0 }),
      moveType: 'ring_placement',
    });

    expect(applyMoveMock).toHaveBeenCalledWith(
      expect.objectContaining({
        to: { x: 1, y: -1, z: 0 },
      })
    );
  });

  it('handles position as JSON string with x, y coordinates', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as any, new Map());

    const state = {
      id: 'test-game-id',
      boardType: 'square8',
      board: { type: 'square8', size: 8, stacks: new Map(), markers: new Map() },
      players: [
        { id: 'player-1', type: 'human', playerNumber: 1, ringsInHand: 18 },
        { id: 'player-2', type: 'human', playerNumber: 2, ringsInHand: 18 },
      ],
      currentPlayer: 1,
      currentPhase: 'ring_placement',
      spectators: [],
      gameStatus: 'active',
    } as any;

    const applyMoveMock = jest.fn().mockResolvedValue({
      success: true,
      state: { ...state, currentPlayer: 2 },
    });

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      getValidMoves: jest.fn(() => []),
    };
    (session as any).rulesFacade = {
      applyMove: applyMoveMock,
    };

    jest.spyOn(session as any, 'persistMove').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);
    jest.spyOn(session as any, 'maybePerformAITurn').mockResolvedValue(undefined);

    const socket = createMockSocket('player-1');

    // Pass position as JSON string with x, y
    await session.handlePlayerMove(socket, {
      position: JSON.stringify({ x: 5, y: 5 }),
      moveType: 'ring_placement',
    });

    // Should work - position treated as 'to'
    expect(applyMoveMock).toHaveBeenCalledWith(
      expect.objectContaining({
        to: { x: 5, y: 5 },
      })
    );
  });
});
