import type { Server as SocketIOServer } from 'socket.io';
import { GameSession } from '../../src/server/game/GameSession';
import { getDatabaseClient } from '../../src/server/database/connection';
import { PythonRulesClient } from '../../src/server/services/PythonRulesClient';

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(),
}));

jest.mock('../../src/server/services/PythonRulesClient', () => ({
  PythonRulesClient: jest.fn().mockImplementation(() => ({})),
}));

const mockedGetDatabaseClient = getDatabaseClient as jest.MockedFunction<typeof getDatabaseClient>;

/**
 * Integration-style test that exercises GameSession.handleAIFatalFailure
 * directly and verifies that:
 *   - the derived GameSessionStatus is updated to `abandoned`, and
 *   - expected game_error and game_over events are emitted on the room.
 *
 * This complements the AI resilience tests (which focus on AIEngine and
 * AIServiceClient) by asserting the session-level projection for the
 * fatal AI failure path.
 */

describe('GameSession AI fatal failure integration', () => {
  const gameId = 'ai-fatal-failure-game-1';

  it('sets sessionStatus to abandoned and emits game_error and game_over', async () => {
    const roomEmitter = {
      emit: jest.fn(),
    };

    const io = {
      to: jest.fn(() => roomEmitter),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as unknown as SocketIOServer;

    // Minimal Prisma mock: only the game.update used in handleAIFatalFailure.
    const mockPrisma = {
      game: {
        update: jest.fn(async (args: any) => ({ id: gameId, ...args.data })),
      },
    } as any;

    mockedGetDatabaseClient.mockReturnValue(mockPrisma);

    const userSockets = new Map<string, string>();
    const session = new GameSession(gameId, io, new PythonRulesClient() as any, userSockets) as any;

    // Stub out a minimal GameState so that handleAIFatalFailure can
    // construct a GameResult snapshot.
    const stubState = {
      id: gameId,
      gameStatus: 'active',
      boardType: 'square8',
      currentPlayer: 1,
      currentPhase: 'ring_placement',
      players: [
        {
          playerNumber: 1,
          eliminatedRings: 2,
          territorySpaces: 10,
          ringsInHand: 3,
        },
      ],
      spectators: [],
      moveHistory: [],
      history: [],
      rngSeed: 42,
      board: {},
    } as any;

    session.gameEngine = {
      getGameState: jest.fn(() => stubState),
    };

    const failureContext = { reason: 'AI service and local fallback exhausted' };

    await session.handleAIFatalFailure(1, failureContext);

    // The derived session status should now reflect an abandoned game
    // with the correct gameId.
    const status = session.getSessionStatusSnapshot();
    expect(status).not.toBeNull();
    if (status) {
      expect(status.kind).toBe('abandoned');
      expect(status.gameId).toBe(gameId);
      expect(status.status).toBe('abandoned');
    }

    // game_error and game_over should have been emitted to the room
    // corresponding to gameId.
    expect((io as any).to).toHaveBeenCalledWith(gameId);

    const calls = roomEmitter.emit.mock.calls;
    const gameErrorCall = calls.find(([event]) => event === 'game_error');
    const gameOverCall = calls.find(([event]) => event === 'game_over');

    expect(gameErrorCall).toBeDefined();
    expect(gameOverCall).toBeDefined();

    if (gameErrorCall) {
      const [, payload] = gameErrorCall;
      expect(payload.type).toBe('game_error');
      expect(payload.data.gameId).toBe(gameId);
    }

    if (gameOverCall) {
      const [, payload] = gameOverCall;
      expect(payload.type).toBe('game_over');
      expect(payload.data.gameId).toBe(gameId);
      expect(payload.data.gameResult.reason).toBe('abandonment');
    }
  });
});
