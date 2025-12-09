/**
 * AIServiceClient.outage.test.ts
 *
 * Comprehensive tests for AI service outage scenarios including:
 * - Service unavailability detection
 * - Timeout handling
 * - Circuit breaker behavior
 * - Error categorization
 * - ServiceStatusManager integration
 *
 * Created as part of P18.7-3: AI Service Outage and Fallback Flow Coverage
 */

import { AIServiceClient } from '../../../src/server/services/AIServiceClient';
import { GameState } from '../../../src/shared/types/game';
import { logger } from '../../../src/server/utils/logger';
import { getServiceStatusManager } from '../../../src/server/services/ServiceStatusManager';

// Mock axios so that AIServiceClient does not perform real HTTP calls.

var mockPost: jest.Mock;

var mockGet: jest.Mock;

var mockCreate: jest.Mock;

jest.mock('axios', () => {
  mockPost = jest.fn();
  mockGet = jest.fn();
  mockCreate = jest.fn(() => ({
    post: mockPost,
    get: mockGet,
    delete: jest.fn(),
    interceptors: {
      response: {
        use: jest.fn((onFulfilled, onRejected) => {
          // Store the error handler so we can test error categorization
          (mockPost as any).errorHandler = onRejected;
          (mockGet as any).errorHandler = onRejected;
        }),
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

jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

jest.mock('../../../src/server/services/ServiceStatusManager', () => ({
  getServiceStatusManager: jest.fn(() => ({
    updateServiceStatus: jest.fn(),
  })),
}));

jest.mock('../../../src/server/services/MetricsService', () => ({
  getMetricsService: jest.fn(() => ({
    recordAIRequest: jest.fn(),
    recordAIRequestDuration: jest.fn(),
    recordAIRequestLatencyMs: jest.fn(),
    recordAIRequestTimeout: jest.fn(),
    recordAIFallback: jest.fn(),
    recordAIChoiceRequest: jest.fn(),
    recordAIChoiceLatencyMs: jest.fn(),
  })),
}));

describe('AIServiceClient - Outage Scenarios', () => {
  let client: AIServiceClient;
  let mockGameState: GameState;
  let mockStatusManager: any;

  beforeEach(() => {
    jest.clearAllMocks();
    // Reset internal counters between tests.
    (AIServiceClient as any).inFlightRequests = 0;
    (AIServiceClient as any).maxConcurrent = 16;

    mockStatusManager = {
      updateServiceStatus: jest.fn(),
    };
    (getServiceStatusManager as jest.Mock).mockReturnValue(mockStatusManager);

    client = new AIServiceClient('http://ai.test');

    mockGameState = {
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
      players: [
        {
          id: 'player1',
          username: 'Player 1',
          playerNumber: 1,
          type: 'ai',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 10,
          eliminatedRings: 0,
          territorySpaces: 0,
          aiDifficulty: 5,
        },
      ] as any,
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
  });

  describe('Service Unavailable Scenarios', () => {
    it('should detect AI service unavailable via connection refused', async () => {
      const connRefusedError = new Error('connect ECONNREFUSED 127.0.0.1:8001');
      (connRefusedError as any).code = 'ECONNREFUSED';
      // Simulate what the axios interceptor would set
      (connRefusedError as any).aiErrorType = 'connection_refused';

      mockPost.mockRejectedValue(connRefusedError);

      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toMatchObject({
        code: 'AI_SERVICE_UNAVAILABLE',
        statusCode: 503,
      });
    });

    it('should detect AI service unavailable via 503 status', async () => {
      const serviceUnavailableError = new Error('Service Unavailable');
      (serviceUnavailableError as any).response = { status: 503 };
      // Simulate what the axios interceptor would set
      (serviceUnavailableError as any).aiErrorType = 'service_unavailable';

      mockPost.mockRejectedValue(serviceUnavailableError);

      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toMatchObject({
        code: 'AI_SERVICE_UNAVAILABLE',
        statusCode: 503,
      });
    });

    it('should update ServiceStatusManager to unhealthy on service error', async () => {
      const error = new Error('Service error');
      // Without aiErrorType set, it will be treated as unknown -> unhealthy

      mockPost.mockRejectedValue(error);

      try {
        await client.getAIMove(mockGameState, 1, 5);
      } catch {
        // Expected to throw
      }

      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'unhealthy',
        expect.any(String),
        expect.any(Number)
      );
    });

    it('should emit degraded status on timeout', async () => {
      const timeoutError = new Error('timeout of 30000ms exceeded');
      (timeoutError as any).code = 'ECONNABORTED';
      // Simulate what the axios interceptor would set
      (timeoutError as any).aiErrorType = 'timeout';

      mockPost.mockRejectedValue(timeoutError);

      try {
        await client.getAIMove(mockGameState, 1, 5);
      } catch {
        // Expected to throw
      }

      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'degraded',
        expect.any(String),
        expect.any(Number)
      );
    });
  });

  describe('Timeout Scenarios', () => {
    it('should timeout AI move request and return structured error', async () => {
      const timeoutError = new Error('timeout of 30000ms exceeded');
      (timeoutError as any).code = 'ECONNABORTED';
      // Simulate what the axios interceptor would set
      (timeoutError as any).aiErrorType = 'timeout';

      mockPost.mockRejectedValue(timeoutError);

      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toMatchObject({
        code: 'AI_SERVICE_TIMEOUT',
        statusCode: 503,
      });
    });

    it('should timeout AI move request with ETIMEDOUT code', async () => {
      const timeoutError = new Error('ETIMEDOUT');
      (timeoutError as any).code = 'ETIMEDOUT';
      // Simulate what the axios interceptor would set
      (timeoutError as any).aiErrorType = 'timeout';

      mockPost.mockRejectedValue(timeoutError);

      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toMatchObject({
        code: 'AI_SERVICE_TIMEOUT',
        statusCode: 503,
      });
    });

    it('should log timeout errors with aiErrorType', async () => {
      const timeoutError = new Error('timeout');
      (timeoutError as any).code = 'ECONNABORTED';
      // Simulate what the axios interceptor would set
      (timeoutError as any).aiErrorType = 'timeout';

      mockPost.mockRejectedValue(timeoutError);

      try {
        await client.getAIMove(mockGameState, 1, 5);
      } catch {
        // Expected
      }

      expect(logger.error).toHaveBeenCalledWith(
        expect.stringContaining('Failed to get AI move'),
        expect.objectContaining({
          aiErrorType: 'timeout',
        })
      );
    });
  });

  describe('Error Categorization', () => {
    it('should categorize connection refused errors correctly when interceptor sets aiErrorType', async () => {
      const error = new Error('ECONNREFUSED');
      (error as any).code = 'ECONNREFUSED';
      // Simulate what the axios interceptor would set
      (error as any).aiErrorType = 'connection_refused';

      mockPost.mockRejectedValue(error);

      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toMatchObject({
        code: 'AI_SERVICE_UNAVAILABLE',
      });
    });

    it('should treat unknown errors as AI_SERVICE_ERROR', async () => {
      const error = new Error('Unknown error');
      // No aiErrorType - treated as unknown

      mockPost.mockRejectedValue(error);

      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toMatchObject({
        code: 'AI_SERVICE_ERROR',
        statusCode: 502,
      });
    });

    it('should treat 500 with FastAPI-style detail as AI_SERVICE_ERROR with server_error aiErrorType', async () => {
      const error = new Error('Request failed with status code 500');
      (error as any).response = {
        status: 500,
        data: { detail: 'boom' },
      };
      // Simulate interceptor classification attaching aiErrorType based on status.
      (error as any).aiErrorType = 'server_error';

      mockPost.mockRejectedValue(error);

      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toMatchObject({
        code: 'AI_SERVICE_ERROR',
        statusCode: 502,
        aiErrorType: 'server_error',
      });
    });

    it('should categorize server errors (500) correctly', async () => {
      const error = new Error('Internal Server Error');
      (error as any).response = { status: 500 };
      // Without interceptor setting aiErrorType, depends on how categorizeError maps it
      // Actually the interceptor runs on axios instance, so in mocked scenario
      // we need to set aiErrorType manually

      mockPost.mockRejectedValue(error);

      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toMatchObject({
        code: 'AI_SERVICE_ERROR',
        statusCode: 502,
      });
    });

    it('should mark errors as operational for proper handling', async () => {
      const error = new Error('Any error');

      mockPost.mockRejectedValue(error);

      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toMatchObject({
        isOperational: true,
      });
    });
  });

  describe('Circuit Breaker Behavior', () => {
    it('should track failures correctly in circuit breaker', async () => {
      const error = new Error('Service down');
      mockPost.mockRejectedValue(error);

      // Make multiple failing requests
      for (let i = 0; i < 3; i++) {
        try {
          await client.getAIMove(mockGameState, 1, 5);
        } catch {
          // Expected
        }
      }

      const status = client.getCircuitBreakerStatus();
      expect(status.failureCount).toBe(3);
      expect(status.isOpen).toBe(false); // Not yet at threshold (5)
    });

    it('should open circuit breaker after 5 failures', async () => {
      const error = new Error('Service down');
      mockPost.mockRejectedValue(error);

      // Make 5 failing requests to trigger circuit breaker
      for (let i = 0; i < 5; i++) {
        try {
          await client.getAIMove(mockGameState, 1, 5);
        } catch {
          // Expected
        }
      }

      const status = client.getCircuitBreakerStatus();
      expect(status.failureCount).toBe(5);
      expect(status.isOpen).toBe(true);

      // Verify warning was logged
      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('Circuit breaker opened'),
        expect.objectContaining({
          failureCount: 5,
          threshold: 5,
        })
      );
    });

    it('should reject requests immediately when circuit is open', async () => {
      const error = new Error('Service down');
      mockPost.mockRejectedValue(error);

      // Open the circuit breaker
      for (let i = 0; i < 5; i++) {
        try {
          await client.getAIMove(mockGameState, 1, 5);
        } catch {
          // Expected
        }
      }

      // Clear mocks to verify no more HTTP calls
      mockPost.mockClear();

      // Next request should fail immediately with circuit breaker error
      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toThrow(
        'Circuit breaker is open'
      );

      // Should not have made another HTTP call
      expect(mockPost).not.toHaveBeenCalled();
    });

    it('should reset circuit breaker on successful request', async () => {
      const error = new Error('Service down');

      // First, make some failures (but not enough to open)
      mockPost.mockRejectedValue(error);
      for (let i = 0; i < 3; i++) {
        try {
          await client.getAIMove(mockGameState, 1, 5);
        } catch {
          // Expected
        }
      }

      expect(client.getCircuitBreakerStatus().failureCount).toBe(3);

      // Now make a successful request
      mockPost.mockResolvedValue({
        data: {
          move: { type: 'place_ring', player: 1, to: { x: 0, y: 0 }, placementCount: 1 },
          evaluation: 0.5,
          thinking_time_ms: 100,
          ai_type: 'heuristic',
          difficulty: 5,
        },
      });

      await client.getAIMove(mockGameState, 1, 5);

      // Circuit breaker should be reset
      const status = client.getCircuitBreakerStatus();
      expect(status.failureCount).toBe(0);
      expect(status.isOpen).toBe(false);
    });

    it('should report service availability based on circuit breaker state', async () => {
      expect(client.isServiceAvailable()).toBe(true);

      const error = new Error('Service down');
      mockPost.mockRejectedValue(error);

      // Open the circuit breaker
      for (let i = 0; i < 5; i++) {
        try {
          await client.getAIMove(mockGameState, 1, 5);
        } catch {
          // Expected
        }
      }

      expect(client.isServiceAvailable()).toBe(false);
    });
  });

  describe('Health Check Integration', () => {
    it('should return true for healthy AI service', async () => {
      mockGet.mockResolvedValue({
        data: { status: 'healthy' },
      });

      const isHealthy = await client.healthCheck();

      expect(isHealthy).toBe(true);
      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'healthy',
        undefined,
        expect.any(Number)
      );
    });

    it('should return false for unhealthy AI service', async () => {
      mockGet.mockResolvedValue({
        data: { status: 'unhealthy' },
      });

      const isHealthy = await client.healthCheck();

      expect(isHealthy).toBe(false);
      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'degraded',
        expect.stringContaining('non-healthy'),
        expect.any(Number)
      );
    });

    it('should return false and update status on health check failure', async () => {
      mockGet.mockRejectedValue(new Error('Connection failed'));

      const isHealthy = await client.healthCheck();

      expect(isHealthy).toBe(false);
      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'unhealthy',
        expect.any(String),
        expect.any(Number)
      );
    });
  });

  describe('Concurrency Backpressure', () => {
    it('should track in-flight requests', async () => {
      expect(AIServiceClient.getInFlightRequestsForTest()).toBe(0);

      // Create a never-resolving promise
      mockPost.mockImplementation(
        () =>
          new Promise(() => {
            /* never resolves */
          })
      );

      // Start a request (don't await)
      const req1 = client.getAIMove(mockGameState, 1, 5);

      // Give it a moment to register
      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(AIServiceClient.getInFlightRequestsForTest()).toBe(1);

      // Clean up - just let the promise hang, test will end
    });

    it('should reject with overload error when concurrent limit exceeded', async () => {
      // Set max concurrent to 1
      (AIServiceClient as any).maxConcurrent = 1;

      // Create a never-resolving promise
      mockPost.mockImplementation(
        () =>
          new Promise(() => {
            /* never resolves */
          })
      );

      // Start first request (don't await)

      client.getAIMove(mockGameState, 1, 5);

      // Second request should fail immediately
      await expect(client.getAIMove(mockGameState, 1, 5)).rejects.toMatchObject({
        code: 'AI_SERVICE_OVERLOADED',
        statusCode: 503,
      });
    });
  });

  describe('Service Status Updates', () => {
    it('should update status to healthy on successful request', async () => {
      mockPost.mockResolvedValue({
        data: {
          move: { type: 'place_ring', player: 1, to: { x: 0, y: 0 }, placementCount: 1 },
          evaluation: 0.5,
          thinking_time_ms: 100,
          ai_type: 'heuristic',
          difficulty: 5,
        },
      });

      await client.getAIMove(mockGameState, 1, 5);

      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'healthy',
        undefined,
        expect.any(Number)
      );
    });

    it('should update status to unhealthy on failure', async () => {
      const error = new Error('Service down');
      mockPost.mockRejectedValue(error);

      try {
        await client.getAIMove(mockGameState, 1, 5);
      } catch {
        // Expected
      }

      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'unhealthy',
        expect.any(String),
        expect.any(Number)
      );
    });

    it('should update status to degraded on timeout', async () => {
      const timeoutError = new Error('timeout');
      (timeoutError as any).code = 'ECONNABORTED';
      (timeoutError as any).aiErrorType = 'timeout';

      mockPost.mockRejectedValue(timeoutError);

      try {
        await client.getAIMove(mockGameState, 1, 5);
      } catch {
        // Expected
      }

      expect(mockStatusManager.updateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'degraded',
        expect.any(String),
        expect.any(Number)
      );
    });
  });
});
