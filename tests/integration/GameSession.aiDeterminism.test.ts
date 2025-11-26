import { createServer } from 'http';
import type { Server as SocketIOServer } from 'socket.io';
import { GameSession } from '../../src/server/game/GameSession';
import { hashGameState } from '../../src/shared/engine/core';
import { TimeControl, GameState, Move } from '../../src/shared/types/game';
import { getDatabaseClient } from '../../src/server/database/connection';
import { PythonRulesClient } from '../../src/server/services/PythonRulesClient';
import { getAIServiceClient } from '../../src/server/services/AIServiceClient';

/**
 * Backend-focused integration test that verifies AI move selection is
 * deterministic for a given game when seeded via Game.rngSeed and
 * reconstructed through GameSession.initialize.
 *
 * Given the same Prisma-style Game snapshot (including rngSeed and
 * aiOpponents config), two independent GameSession instances should:
 *   - construct identical initial GameState (same hash), and
 *   - after running a single AI turn, arrive at identical post-move
 *     GameState (same hash, same last move) driven by the per-game
 *     SeededRNG stream.
 */

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(),
}));

jest.mock('../../src/server/services/PythonRulesClient', () => ({
  PythonRulesClient: jest.fn().mockImplementation(() => ({})),
}));

jest.mock('../../src/server/services/AIServiceClient');

const mockedGetDatabaseClient = getDatabaseClient as jest.MockedFunction<typeof getDatabaseClient>;
const mockedGetAIServiceClient = getAIServiceClient as jest.MockedFunction<
  typeof getAIServiceClient
>;

describe('GameSession + AI determinism (per-game rngSeed)', () => {
  const gameId = 'ai-determinism-game-1';

  // Prevent real turn timers from being started in this integration test,
  // which would otherwise leave setInterval handles active and trigger
  // Jest's open-handle warning.
  let startTimerSpy: jest.SpyInstance;
  let stopTimerSpy: jest.SpyInstance;

  beforeAll(() => {
    startTimerSpy = jest
      .spyOn(GameSession.prototype as any, 'startTurnTimer')
      .mockImplementation(() => {});
    stopTimerSpy = jest
      .spyOn(GameSession.prototype as any, 'stopTurnTimer')
      .mockImplementation(() => {});
  });

  afterAll(() => {
    startTimerSpy.mockRestore();
    stopTimerSpy.mockRestore();
  });

  it('produces identical AI moves and post-move GameState across sessions', async () => {
    const timeControl: TimeControl = {
      type: 'rapid',
      initialTime: 600,
      increment: 0,
    };

    // Prisma-like Game row: no human players, a single AI opponent
    // configured via gameState.aiOpponents, and a fixed rngSeed.
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
      rngSeed: 4321,
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
        update: jest.fn(async (args: any) => {
          return { ...dbGame, ...args.data };
        }),
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

    // Force AIEngine to hit the local-heuristic fallback path by making
    // the AI service client consistently fail. This ensures that the
    // deterministic move we observe is coming from the shared
    // localAIMoveSelection policy rather than the Python service.
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

    const session1 = new GameSession(gameId, io, new PythonRulesClient(), userSockets);
    const session2 = new GameSession(gameId, io, new PythonRulesClient(), userSockets);

    await session1.initialize();
    await session2.initialize();

    // In normal flows the game is started via lobby/game lifecycle before
    // the first AI turn. For this focused determinism test we explicitly
    // start the underlying engines so maybePerformAITurn is allowed to
    // act and we observe a real AI move.
    (session1 as any).gameEngine.startGame();
    (session2 as any).gameEngine.startGame();

    const state1Before: GameState = session1.getGameState();
    const state2Before: GameState = session2.getGameState();

    expect(state1Before.rngSeed).toBe(4321);
    expect(state2Before.rngSeed).toBe(4321);

    const beforeHash1 = hashGameState(state1Before);
    const beforeHash2 = hashGameState(state2Before);
    expect(beforeHash1).toBe(beforeHash2);

    // Drive a single AI turn in each session via maybePerformAITurn and
    // compare the resulting states. We reach into the private method via
    // `any` to avoid changing the production GameSession API.
    await (session1 as any).maybePerformAITurn();
    await (session2 as any).maybePerformAITurn();

    const state1After = session1.getGameState();
    const state2After = session2.getGameState();

    const afterHash1 = hashGameState(state1After);
    const afterHash2 = hashGameState(state2After);

    // Determinism contract: with the same rngSeed and history, both
    // sessions pick the same AI move and arrive at identical states.
    expect(afterHash1).toBe(afterHash2);

    const lastMove1: Move | undefined = state1After.moveHistory[state1After.moveHistory.length - 1];
    const lastMove2: Move | undefined = state2After.moveHistory[state2After.moveHistory.length - 1];

    expect(lastMove1).toBeDefined();
    expect(lastMove2).toBeDefined();

    if (lastMove1 && lastMove2) {
      expect(lastMove1.type).toBe(lastMove2.type);
      expect(lastMove1.player).toBe(lastMove2.player);
      expect(lastMove1.from ?? null).toEqual(lastMove2.from ?? null);
      expect(lastMove1.to ?? null).toEqual(lastMove2.to ?? null);
    }

    // Confirm via AI diagnostics that the move came from the local
    // heuristic fallback rather than the remote service.
    const diag1 = session1.getAIDiagnosticsSnapshotForTesting();
    const diag2 = session2.getAIDiagnosticsSnapshotForTesting();

    expect(diag1.aiFallbackMoveCount).toBeGreaterThan(0);
    expect(diag2.aiFallbackMoveCount).toBeGreaterThan(0);
    expect(diag1.aiQualityMode).toBe('fallbackLocalAI');
    expect(diag2.aiQualityMode).toBe('fallbackLocalAI');

    // Also assert that the derived GameSessionStatus projection stays in
    // sync with the underlying GameState for both sessions after the
    // first AI move.
    const status1 = session1.getSessionStatusSnapshot();
    const status2 = session2.getSessionStatusSnapshot();

    expect(status1).not.toBeNull();
    expect(status2).not.toBeNull();

    if (status1 && status2) {
      expect(status1.kind).toBe('active_turn');
      expect(status2.kind).toBe('active_turn');
      expect(status1.gameId).toBe(gameId);
      expect(status2.gameId).toBe(gameId);
    }

    io.close();
    httpServer.close();
  });
});
