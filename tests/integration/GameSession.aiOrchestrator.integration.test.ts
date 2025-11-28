import type { Server as SocketIOServer } from 'socket.io';
import { createServer } from 'http';
import { GameSession } from '../../src/server/game/GameSession';
import { getDatabaseClient } from '../../src/server/database/connection';
import { PythonRulesClient } from '../../src/server/services/PythonRulesClient';
import { getAIServiceClient } from '../../src/server/services/AIServiceClient';
import { hashGameState } from '../../src/shared/engine/core';
import type { GameState, Move, TimeControl } from '../../src/shared/types/game';

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(),
}));

jest.mock('../../src/server/services/PythonRulesClient', () => ({
  PythonRulesClient: jest.fn().mockImplementation(() => ({})),
}));

jest.mock('../../src/server/services/AIServiceClient');

jest.mock('../../src/server/services/OrchestratorRolloutService', () => {
  const actual = jest.requireActual('../../src/server/services/OrchestratorRolloutService');
  const { EngineSelection } = actual;
  return {
    ...actual,
    orchestratorRollout: {
      selectEngine: jest.fn((_sessionId: string, _userId?: string) => ({
        engine: EngineSelection.ORCHESTRATOR,
        reason: 'test_orchestrator_integration',
      })),
      recordSuccess: jest.fn(),
      recordError: jest.fn(),
      isCircuitBreakerOpen: jest.fn().mockReturnValue(false),
      resetCircuitBreaker: jest.fn(),
      getCircuitBreakerState: jest.fn().mockReturnValue({
        isOpen: false,
        errorCount: 0,
        requestCount: 0,
        windowStart: Date.now(),
      }),
      getErrorRate: jest.fn().mockReturnValue(0),
    },
  };
});

const mockedGetDatabaseClient = getDatabaseClient as jest.MockedFunction<typeof getDatabaseClient>;
const mockedGetAIServiceClient = getAIServiceClient as jest.MockedFunction<
  typeof getAIServiceClient
>;

/**
 * Integration test that verifies GameSession + AI turns work correctly when
 * the backend GameEngine is delegated to the shared orchestrator via
 * TurnEngineAdapter. This is a complement to the existing AI determinism test,
 * but forces EngineSelection.ORCHESTRATOR so we exercise the adapter path.
 */
describe('GameSession AI turn integration with orchestrator adapter', () => {
  const gameId = 'ai-orchestrator-game-1';

  // Prevent real turn timers from running in this test.
  let startTimerSpy: jest.SpyInstance | undefined;
  let stopTimerSpy: jest.SpyInstance | undefined;

  beforeAll(() => {
    const proto = GameSession.prototype as any;

    if (typeof proto.startTurnTimer === 'function') {
      startTimerSpy = jest.spyOn(proto, 'startTurnTimer').mockImplementation(() => {});
    }

    if (typeof proto.stopTurnTimer === 'function') {
      stopTimerSpy = jest.spyOn(proto, 'stopTurnTimer').mockImplementation(() => {});
    }
  });

  afterAll(() => {
    if (startTimerSpy) {
      startTimerSpy.mockRestore();
    }
    if (stopTimerSpy) {
      stopTimerSpy.mockRestore();
    }
  });

  it('completes multiple AI turns under orchestrator without stalls and keeps state consistent', async () => {
    const timeControl: TimeControl = {
      type: 'rapid',
      initialTime: 600,
      increment: 0,
    };

    const dbGame = {
      id: gameId,
      boardType: 'square8' as const,
      maxPlayers: 1,
      timeControl: {
        type: timeControl.type,
        initialTime: timeControl.initialTime,
        increment: timeControl.increment,
      },
      isRated: true,
      allowSpectators: true,
      status: 'waiting' as any,
      gameState: {
        aiOpponents: {
          count: 1,
          difficulty: [5],
          mode: 'service' as const,
        },
      },
      rngSeed: 9876,
      player1Id: null as string | null,
      player2Id: null as string | null,
      player3Id: null as string | null,
      player4Id: null as string | null,
      player1: null as any,
      player2: null as any,
      player3: null as any,
      player4: null as any,
      moves: [] as Array<{
        id: string;
        gameId: string;
        playerId: string;
        moveNumber: number;
        position: string | object;
        moveType: string;
        timestamp: Date;
      }>,
    };

    const mockPrisma = {
      game: {
        findUnique: jest.fn(async (args: any) => {
          if (args?.where?.id === gameId) {
            return dbGame;
          }
          return null;
        }),
        update: jest.fn(async (args: any) => ({ ...dbGame, ...args.data })),
      },
      move: {
        create: jest.fn(),
      },
      user: {
        findFirst: jest.fn(),
        create: jest.fn(),
      },
    } as any;

    mockedGetDatabaseClient.mockReturnValue(mockPrisma);

    // Force AIEngine to prefer local fallback by making the AI service fail,
    // but we still go through the AI turn pipeline and orchestrator adapter.
    const fakeAIClient = {
      getAIMove: jest
        .fn()
        .mockRejectedValue(
          Object.assign(new Error('Service down'), { aiErrorType: 'connection_refused' })
        ),
      healthCheck: jest.fn(),
      clearCache: jest.fn(),
      getLineRewardChoice: jest.fn(),
      getRingEliminationChoice: jest.fn(),
      getRegionOrderChoice: jest.fn(),
      getCircuitBreakerStatus: jest.fn(() => ({ isOpen: false, failureCount: 0 })),
    } as any;

    mockedGetAIServiceClient.mockReturnValue(fakeAIClient);

    const httpServer = createServer();

    const io = {
      to: () => ({
        emit: jest.fn(),
      }),
      sockets: {
        adapter: {
          rooms: new Map(),
        },
        sockets: new Map(),
      },
      on: jest.fn(),
      close: jest.fn(),
    } as unknown as SocketIOServer;

    const userSockets = new Map<string, string>();

    const session = new GameSession(gameId, io, new PythonRulesClient() as any, userSockets);

    await session.initialize();

    // Start the underlying engine so AI turns are allowed.
    (session as any).gameEngine.startGame();

    const initialState: GameState = session.getGameState();
    if (initialState.rngSeed !== undefined) {
      expect(initialState.rngSeed).toBe(9876);
    }

    let lastHash = hashGameState(initialState);
    let lastMoveCount = initialState.moveHistory.length;

    const maxTurns = 8;
    for (let i = 0; i < maxTurns; i++) {
      const before: GameState = session.getGameState();
      if (before.gameStatus !== 'active') {
        break;
      }

      await (session as any).maybePerformAITurn();

      const after: GameState = session.getGameState();
      const newHash = hashGameState(after);

      // State must change on a successful AI move; if the AI turn was
      // skipped (e.g. no moves), we allow the hash to remain equal.
      if (after.moveHistory.length > lastMoveCount) {
        expect(newHash).not.toBe(lastHash);
      }

      lastHash = newHash;
      lastMoveCount = after.moveHistory.length;
    }

    const finalState = session.getGameState();

    // Sanity: no illegal stuck state where the game is active but there are
    // zero valid moves for the current player.
    if (finalState.gameStatus === 'active') {
      const moves = (session as any).gameEngine.getValidMoves(finalState.currentPlayer) as Move[];
      expect(Array.isArray(moves)).toBe(true);
      expect(moves.length).toBeGreaterThan(0);
    }

    io.close();
    httpServer.close();
  });
});
