import { GameSession } from '../../src/server/game/GameSession';
import { getAIServiceClient } from '../../src/server/services/AIServiceClient';
import type { GameState } from '../../src/shared/types/game';

jest.mock('../../src/server/services/AIServiceClient');

describe('GameSession.terminate → AI cancellation integration', () => {
  it('cancels session token so service-backed AI calls never issue HTTP after termination', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const fakeClient = {
      // Representative AI HTTP entrypoint – any service-backed method would do.
      getAIMove: jest.fn(
        async (
          _state: GameState,
          _playerNumber: number,
          _difficulty: number,
          _aiType: any,
          _requestOptions?: { token?: { isCanceled: boolean; throwIfCanceled?: () => void } }
        ) => {
          // Minimal shape; in this test, we assert that this method is
          // never invoked once the session has been terminated.
          return null;
        }
      ),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const io = {
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any;

    const pythonClient: any = {
      evaluateMove: jest.fn(),
      healthCheck: jest.fn(),
    };

    const userSockets = new Map<string, string>();
    const session = new GameSession('game-terminate-ai', io, pythonClient, userSockets);

    const sessionToken = (session as any).sessionCancellationSource.token as {
      isCanceled: boolean;
    };
    expect(sessionToken.isCanceled).toBe(false);

    // Seed a minimal gameEngine so that maybePerformAITurn observes an
    // active AI player and attempts to call into the AI service.
    (session as any).gameEngine = {
      getGameState: jest.fn(() => ({
        gameId: 'game-terminate-ai',
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
            id: 'p1',
            username: 'P1',
            playerNumber: 1,
            type: 'ai',
            isReady: true,
            timeRemaining: 0,
            ringsInHand: 0,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
          {
            id: 'p2',
            username: 'P2',
            playerNumber: 2,
            type: 'human',
            isReady: true,
            timeRemaining: 0,
            ringsInHand: 0,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
        spectators: [],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
        moveHistory: [],
        history: [],
        rngSeed: 0,
        timeControl: { initialTime: 600, increment: 5, type: 'blitz' },
        victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold: 33,
        totalRingsEliminated: 0,
      })),
    };

    // Ensure session termination has run before any AI work is attempted.
    session.terminate('session_cleanup');
    expect(sessionToken.isCanceled).toBe(true);

    // Invoke maybePerformAITurn; any service-backed call it triggers must
    // observe the canceled session token and therefore short-circuit before
    // issuing HTTP. In this test, that means the mocked AIServiceClient
    // should never be asked to perform getAIMove.
    await (session as any).maybePerformAITurn();

    expect(fakeClient.getAIMove).not.toHaveBeenCalled();
  });
});
