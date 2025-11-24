import { AIServiceClient } from '../../src/server/services/AIServiceClient';
import { GameState } from '../../src/shared/types/game';

// Mock axios so that AIServiceClient does not perform real HTTP calls.
// We use `var` for the mock handles because Jest hoists `jest.mock` calls,
// and `var` avoids temporal-dead-zone issues when the factory initializes them.
// eslint-disable-next-line no-var
var mockPost: jest.Mock;
// eslint-disable-next-line no-var
var mockCreate: jest.Mock;

jest.mock('axios', () => {
  mockPost = jest.fn(
    () =>
      new Promise(() => {
        /* never resolve */
      })
  );
  mockCreate = jest.fn(() => ({
    post: mockPost,
    get: jest.fn(),
    delete: jest.fn(),
    interceptors: {
      response: {
        // AIServiceClient registers an interceptor; we only need a stub.
        use: jest.fn(),
      },
    },
  }));

  return {
    __esModule: true,
    default: {
      create: mockCreate,
    },
    create: mockCreate,
  };
});

describe('AIServiceClient concurrency backpressure', () => {
  beforeEach(() => {
    mockPost.mockClear();
    mockCreate.mockClear();
    // Reset internal counters between tests.
    (AIServiceClient as any).inFlightRequests = 0;
    (AIServiceClient as any).maxConcurrent = 16;
  });

  it('fails fast with AI_SERVICE_OVERLOADED when max concurrent requests is exceeded', async () => {
    // Shrink the concurrency cap so the test can exercise the overloaded path
    // with just two in-flight requests.
    (AIServiceClient as any).maxConcurrent = 1;

    const client = new AIServiceClient('http://ai.test');

    const gameState: GameState = {
      id: 'test-game',
      boardType: 'square8',
      board: {
        type: 'square8',
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        pendingCaptureEvaluations: [],
        eliminatedRings: {} as any,
        size: 8,
      } as any,
      players: [] as any,
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      moveHistory: [],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 } as any,
      spectators: [] as any,
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 0,
      totalRingsEliminated: 0,
      victoryThreshold: 0,
      territoryVictoryThreshold: 0,
      rngSeed: 123,
    };

    // First request acquires the single concurrency slot and never resolves.
    // We intentionally do not await this promise so the in-flight counter
    // remains > 0 while we issue the second request.
    // eslint-disable-next-line @typescript-eslint/no-floating-promises
    client.getAIMove(gameState, 1, 5);

    await expect(client.getAIMove(gameState, 1, 5)).rejects.toMatchObject({
      code: 'AI_SERVICE_OVERLOADED',
      statusCode: 503,
    });

    // Only the first request should have attempted an HTTP call.
    expect(mockPost).toHaveBeenCalledTimes(1);
  });
});
