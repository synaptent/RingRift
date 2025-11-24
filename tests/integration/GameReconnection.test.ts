import { createServer } from 'http';
import type { Server as SocketIOServer } from 'socket.io';
import { GameEngine } from '../../src/server/game/GameEngine';
import { GameSession } from '../../src/server/game/GameSession';
import { hashGameState } from '../../src/shared/engine/core';
import { Player, TimeControl, GameState, Move } from '../../src/shared/types/game';
import { getDatabaseClient } from '../../src/server/database/connection';
import { PythonRulesClient } from '../../src/server/services/PythonRulesClient';

/**
 * Backend-focused integration test (without hitting a real database) that
 * verifies we can reconstruct an authoritative GameState from persisted
 * history and that doing so is deterministic across multiple GameSession
 * instances. This underpins robust reconnection flows: when a client
 * reconnects and the in-memory session is missing (e.g. after a restart),
 * GameSession.initialize must be able to rebuild the engine state purely
 * from DB history.
 */

// Mock the database connection layer so GameSession.initialize can run
// without a real Postgres instance.
jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(),
}));

// Mock the PythonRulesClient so RulesBackendFacade wiring does not attempt
// any real HTTP calls during session initialization.
jest.mock('../../src/server/services/PythonRulesClient', () => ({
  PythonRulesClient: jest.fn().mockImplementation(() => ({})),
}));

const mockedGetDatabaseClient = getDatabaseClient as jest.MockedFunction<typeof getDatabaseClient>;

describe('Game reconnection and state resync (GameSession.initialize)', () => {
  const gameId = 'reconnect-game-1';
  let baselineFinalState: GameState;

  beforeAll(async () => {
    // Step 1: Use the canonical GameEngine APIs to produce a simple but legal
    // opening move for a fresh 2-player game. This gives us a baseline
    // authoritative state and a Move payload that we can encode into a
    // Prisma-style history record.
    const players: Player[] = [
      {
        id: 'p1',
        username: 'Player1',
        playerNumber: 1,
        type: 'human',
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'Player2',
        playerNumber: 2,
        type: 'human',
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    const timeControl: TimeControl = {
      type: 'rapid',
      initialTime: 600,
      increment: 0,
    };

    const engine = new GameEngine(gameId, 'square8', players, timeControl);
    engine.startGame();

    const stateBefore = engine.getGameState();
    const validMoves = engine.getValidMoves(stateBefore.currentPlayer);

    if (validMoves.length === 0) {
      throw new Error('Expected at least one valid move in the initial position');
    }

    const firstMove = validMoves[0];
    const { id, timestamp, moveNumber, ...payload } = firstMove as any;

    const result = await engine.makeMove(payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>);
    if (!result.success) {
      throw new Error(`Baseline move failed: ${result.error ?? 'unknown error'}`);
    }

    baselineFinalState = engine.getGameState();

    // Step 2: Build a Prisma-like Game + Move record that encodes the same
    // opening move history that produced baselineFinalState. This is exactly
    // the shape that GameSession.initialize expects from
    // prisma.game.findUnique({ include: { moves: { orderBy: { moveNumber: 'asc' } } } }).
    const dbGame = {
      id: gameId,
      boardType: 'square8' as const,
      maxPlayers: 2,
      timeControl: {
        type: timeControl.type,
        initialTime: timeControl.initialTime,
        increment: timeControl.increment,
      },
      isRated: true,
      allowSpectators: true,
      status: 'active' as any,
      gameState: {},
      rngSeed: 1234,
      player1Id: 'p1',
      player2Id: 'p2',
      player3Id: null as string | null,
      player4Id: null as string | null,
      player1: { id: 'p1', username: 'Player1' },
      player2: { id: 'p2', username: 'Player2' },
      player3: null as any,
      player4: null as any,
      moves: [
        {
          id: 'm1',
          gameId,
          playerId: String(firstMove.player),
          moveNumber: 1,
          position: JSON.stringify({
            from: firstMove.from ?? undefined,
            to: firstMove.to,
          }),
          moveType: firstMove.type as any,
          timestamp: new Date(),
        },
      ],
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
    } as any;

    mockedGetDatabaseClient.mockReturnValue(mockPrisma);
  });

  it('reconstructs identical GameState from persisted history on new sessions', async () => {
    // Use a minimal fake Socket.IO server instance so that GameSession can
    // wire its WebSocketInteractionHandler without requiring a real
    // engine.io implementation (which is unnecessary for reconnection
    // determinism tests and can be brittle under Jest).
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
      // The WebSocketInteractionHandler attaches listeners via `io.on`,
      // but this reconnection test never drives real socket traffic, so
      // a no-op implementation is sufficient.
      on: jest.fn(),
      close: jest.fn(),
    } as unknown as SocketIOServer;

    const userSockets = new Map<string, string>();

    // First session simulates the original in-memory engine that would exist
    // while players are connected.
    const session1 = new GameSession(gameId, io, new PythonRulesClient(), userSockets);
    await session1.initialize();
    const stateFromFirstSession = session1.getGameState();

    // Drop the in-memory session and construct a brand new GameSession that
    // must reconstruct purely from the mocked DB history. This models the
    // reconnection path after a process restart or session eviction.
    const session2 = new GameSession(gameId, io, new PythonRulesClient(), userSockets);
    await session2.initialize();
    const stateFromSecondSession = session2.getGameState();

    const baselineHash = hashGameState(baselineFinalState);
    const firstHash = hashGameState(stateFromFirstSession);
    const secondHash = hashGameState(stateFromSecondSession);

    // Both reconstructed sessions must match the authoritative baseline state
    // produced by the canonical GameEngine flow.
    expect(firstHash).toBe(baselineHash);
    expect(secondHash).toBe(baselineHash);

    // And basic structural invariants (history length, current player) should
    // also agree between sessions.
    expect(stateFromFirstSession.moveHistory.length).toBe(1);
    expect(stateFromSecondSession.moveHistory.length).toBe(1);
    expect(stateFromFirstSession.currentPlayer).toBe(stateFromSecondSession.currentPlayer);

    io.close();
    httpServer.close();
  });
});
