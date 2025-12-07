import { AIServiceClient } from '../../../src/server/services/AIServiceClient';
import { GameState } from '../../../src/shared/types/game';

// Shared mocks for axios – declared with `var` so they can be assigned
// inside the jest.mock factory without TDZ issues.

var mockAxiosPost: jest.Mock;

var mockAxiosCreate: jest.Mock;

// Shared mocks for MetricsService.getMetricsService()

var mockRecordAIRequest: jest.Mock;

var mockRecordAIRequestDuration: jest.Mock;

var mockRecordAIRequestLatencyMs: jest.Mock;

var mockRecordAIRequestTimeout: jest.Mock;

jest.mock('axios', () => {
  mockAxiosPost = jest.fn();
  mockAxiosCreate = jest.fn(() => ({
    post: mockAxiosPost,
    get: jest.fn(),
    delete: jest.fn(),
    interceptors: {
      response: {
        // AIServiceClient registers an interceptor; we only need a stub here.
        use: jest.fn(),
      },
    },
  }));

  return {
    __esModule: true,
    default: {
      create: mockAxiosCreate,
    },
    create: mockAxiosCreate,
  };
});

jest.mock('../../../src/server/services/MetricsService', () => {
  mockRecordAIRequest = jest.fn();
  mockRecordAIRequestDuration = jest.fn();
  mockRecordAIRequestLatencyMs = jest.fn();
  mockRecordAIRequestTimeout = jest.fn();

  return {
    __esModule: true,
    getMetricsService: () => ({
      recordAIRequest: mockRecordAIRequest,
      recordAIRequestDuration: mockRecordAIRequestDuration,
      recordAIRequestLatencyMs: mockRecordAIRequestLatencyMs,
      recordAIRequestTimeout: mockRecordAIRequestTimeout,
    }),
  };
});

describe('AIServiceClient.getAIMove metrics integration', () => {
  const baseGameState: GameState = {
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

  beforeEach(() => {
    // Use mockClear() instead of mockReset() to preserve the mock implementation
    // mockReset() would clear the return value, causing axios.create() to return undefined
    mockAxiosPost.mockClear();
    mockAxiosCreate.mockClear();

    mockRecordAIRequest.mockClear();
    mockRecordAIRequestDuration.mockClear();
    mockRecordAIRequestLatencyMs.mockClear();
    mockRecordAIRequestTimeout.mockClear();

    // Reset concurrency counters between tests
    (AIServiceClient as any).inFlightRequests = 0;
    (AIServiceClient as any).maxConcurrent = 16;
  });

  it('records success metrics when AI move request succeeds', async () => {
    const client = new AIServiceClient('http://ai.test');

    mockAxiosPost.mockResolvedValue({
      data: {
        move: {
          id: 'service-move',
          type: 'place_ring',
          player: 1,
          to: { x: 0, y: 0 },
          placementCount: 1,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
        evaluation: 0.5,
        thinking_time_ms: 100,
        ai_type: 'heuristic',
        difficulty: 5,
      },
    });

    await client.getAIMove(baseGameState, 1, 5);

    expect(mockRecordAIRequest).toHaveBeenCalledWith('success');
    expect(mockRecordAIRequestDuration).toHaveBeenCalledWith('python', '5', expect.any(Number));
    expect(mockRecordAIRequestLatencyMs).toHaveBeenCalledWith(expect.any(Number), 'success');
    expect(mockRecordAIRequestTimeout).not.toHaveBeenCalled();
  });

  it('records timeout metrics and surfaces AI_SERVICE_TIMEOUT code on timeout errors', async () => {
    const client = new AIServiceClient('http://ai.test');

    const timeoutError = Object.assign(new Error('ECONNABORTED'), { aiErrorType: 'timeout' });
    mockAxiosPost.mockRejectedValue(timeoutError);

    await expect(client.getAIMove(baseGameState, 1, 5)).rejects.toMatchObject({
      code: 'AI_SERVICE_TIMEOUT',
      statusCode: 503,
    });

    expect(mockRecordAIRequest).toHaveBeenCalledWith('error');
    expect(mockRecordAIRequestDuration).toHaveBeenCalledWith('python', '5', expect.any(Number));
    expect(mockRecordAIRequestLatencyMs).toHaveBeenCalledWith(expect.any(Number), 'timeout');
    expect(mockRecordAIRequestTimeout).toHaveBeenCalled();
  });

  it('records error and timeout metrics when AI move request times out', async () => {
    const client = new AIServiceClient('http://ai.test');

    const timeoutError: any = new Error('AI service timeout');
    timeoutError.aiErrorType = 'timeout';

    mockAxiosPost.mockRejectedValue(timeoutError);

    await expect(client.getAIMove(baseGameState, 1, 5)).rejects.toMatchObject({
      code: 'AI_SERVICE_TIMEOUT',
    });

    expect(mockRecordAIRequest).toHaveBeenCalledWith('error');
    expect(mockRecordAIRequestDuration).toHaveBeenCalledWith('python', '5', expect.any(Number));
    expect(mockRecordAIRequestLatencyMs).toHaveBeenCalledWith(expect.any(Number), 'timeout');
    expect(mockRecordAIRequestTimeout).toHaveBeenCalledTimes(1);
  });

  it('records error latency metrics and skips axios when concurrency limit is exceeded', async () => {
    (AIServiceClient as any).maxConcurrent = 0;
    const client = new AIServiceClient('http://ai.test');

    await expect(client.getAIMove(baseGameState, 1, 5)).rejects.toMatchObject({
      code: 'AI_SERVICE_OVERLOADED',
      statusCode: 503,
    });

    expect(mockAxiosPost).not.toHaveBeenCalled();

    expect(mockRecordAIRequest).toHaveBeenCalledWith('error');
    expect(mockRecordAIRequestLatencyMs).toHaveBeenCalledWith(expect.any(Number), 'error');
    expect(mockRecordAIRequestDuration).not.toHaveBeenCalled();
    expect(mockRecordAIRequestTimeout).not.toHaveBeenCalled();
  });
});
