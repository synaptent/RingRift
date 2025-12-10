import { Server as SocketIOServer } from 'socket.io';
import { GameSession } from '../../src/server/game/GameSession';
import type { GameResult, GameState } from '../../src/shared/types/game';
import { GamePersistenceService } from '../../src/server/services/GamePersistenceService';

jest.mock('../../src/server/services/GamePersistenceService', () => ({
  GamePersistenceService: {
    saveMove: jest.fn(),
    finishGame: jest.fn().mockResolvedValue({}),
  },
}));

// Minimal Socket.IO server stub for GameSession tests.
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

const mockFinishGame = GamePersistenceService.finishGame as jest.MockedFunction<any>;

describe('GameSession resign and abandonment helpers', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  function createBaseState(overrides: Partial<GameState> = {}): GameState {
    const now = new Date();
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
        eliminatedRings: {},
      },
      players: [
        {
          id: 'p1',
          username: 'P1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'P2',
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

  it('handlePlayerResignationByUserId routes through GameEngine.resignPlayer and finishGame with resignation', async () => {
    const io = createMockIo();
    const session = new GameSession('test-game-id', io, {} as any, new Map());

    const state = createBaseState();
    const gameResult: GameResult = {
      winner: 2,
      reason: 'resignation',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: {},
      },
    };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      resignPlayer: jest.fn(() => ({ success: true, gameResult })),
    };

    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);

    const result = await session.handlePlayerResignationByUserId('p1');

    expect(result).toEqual(gameResult);

    expect(mockFinishGame).toHaveBeenCalledTimes(1);
    expect(mockFinishGame).toHaveBeenCalledWith(
      'test-game-id',
      'p2', // winnerId derived from winner playerNumber 2
      expect.any(Object),
      gameResult
    );

    expect((session as any).broadcastUpdate).toHaveBeenCalledWith({
      success: true,
      gameResult,
    });
  });

  it('handleAbandonmentForDisconnectedPlayer awards win when awardWinToOpponent=true', async () => {
    const io = createMockIo();
    const session = new GameSession('abandon-rated', io, {} as any, new Map());

    const state = createBaseState({ id: 'abandon-rated' });
    const gameResult: GameResult = {
      winner: 2,
      reason: 'abandonment',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: {},
      },
    };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      abandonPlayer: jest.fn(() => ({ success: true, gameResult })),
      abandonGameAsDraw: jest.fn(),
    };

    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);

    const result = await session.handleAbandonmentForDisconnectedPlayer(1, true);

    expect(result).toEqual(gameResult);
    expect((session as any).gameEngine.abandonPlayer).toHaveBeenCalledWith(1);
    expect(mockFinishGame).toHaveBeenCalledWith(
      'abandon-rated',
      'p2',
      expect.any(Object),
      gameResult
    );
  });

  it('handleAbandonmentForDisconnectedPlayer records abandonment without winner when awardWinToOpponent=false', async () => {
    const io = createMockIo();
    const session = new GameSession('abandon-unrated', io, {} as any, new Map());

    const state = createBaseState({ id: 'abandon-unrated', isRated: false });
    const gameResult: GameResult = {
      winner: undefined,
      reason: 'abandonment',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: {},
      },
    };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => state),
      abandonPlayer: jest.fn(),
      abandonGameAsDraw: jest.fn(() => ({ success: true, gameResult })),
    };

    jest.spyOn(session as any, 'broadcastUpdate').mockResolvedValue(undefined);

    const result = await session.handleAbandonmentForDisconnectedPlayer(1, false);

    expect(result).toEqual(gameResult);
    expect((session as any).gameEngine.abandonGameAsDraw).toHaveBeenCalledTimes(1);
    expect(mockFinishGame).toHaveBeenCalledWith(
      'abandon-unrated',
      null,
      expect.any(Object),
      gameResult
    );
  });
});
