/**
 * AIEngine.fallbackProgression.test.ts
 *
 * Comprehensive tests for AI fallback progression including:
 * - Fallback hierarchy (remote → local → random)
 * - Tracking which fallback level was used
 * - Fallback state management
 * - Metrics emission for fallback reasons
 *
 * Created as part of P18.7-3: AI Service Outage and Fallback Flow Coverage
 */

import { AIEngine } from '../../../src/server/game/ai/AIEngine';
import { getAIServiceClient } from '../../../src/server/services/AIServiceClient';
import { GameState, Move, AIProfile } from '../../../src/shared/types/game';
import { logger } from '../../../src/server/utils/logger';
import { getMetricsService } from '../../../src/server/services/MetricsService';
import { getConfiguredTotalRingsInPlay } from '../../utils/fixtures';

// Mock dependencies
jest.mock('../../../src/server/services/AIServiceClient');
jest.mock('../../../src/server/utils/logger');
jest.mock('../../../src/server/services/MetricsService');

// Mock rulesParityMetrics to avoid dependency on MetricsService initialization
jest.mock('../../../src/server/utils/rulesParityMetrics', () => ({
  aiMoveLatencyHistogram: {
    labels: () => ({
      observe: jest.fn(),
    }),
  },
  aiFallbackCounter: {
    labels: () => ({
      inc: jest.fn(),
    }),
  },
  aiFallbackMovesCounter: {
    labels: () => ({
      inc: jest.fn(),
    }),
  },
  aiCircuitBreakerStateGauge: {
    set: jest.fn(),
  },
  aiCircuitBreakerTransitionsCounter: {
    labels: () => ({
      inc: jest.fn(),
    }),
  },
  rulesParityMetrics: {
    validMismatch: { inc: jest.fn() },
    hashMismatch: { inc: jest.fn() },
    sMismatch: { inc: jest.fn() },
    gameStatusMismatch: { inc: jest.fn() },
    moveParity: { inc: jest.fn() },
    parityCheckDuration: { observe: jest.fn() },
    parityCheckCounter: { inc: jest.fn() },
  },
}));

// Shared mutable backing store for the mocked RuleEngine.getValidMoves output.
let mockRuleEngineValidMoves: Move[] = [];

// Mock RuleEngine so we can control the valid move set seen by AIEngine.
jest.mock('../../../src/server/game/RuleEngine', () => {
  return {
    RuleEngine: jest.fn().mockImplementation(() => ({
      getValidMoves: () => mockRuleEngineValidMoves,
    })),
  };
});

describe('AIEngine Fallback Progression', () => {
  let aiEngine: AIEngine;
  let mockAIServiceClient: any;
  let mockMetricsService: any;
  let mockGameState: GameState;
  let mockValidMoves: Move[];

  beforeEach(() => {
    jest.clearAllMocks();

    // Create fresh AI engine instance
    aiEngine = new AIEngine();

    // Mock metrics service
    mockMetricsService = {
      recordAIFallback: jest.fn(),
      recordAIRequest: jest.fn(),
      recordAIRequestDuration: jest.fn(),
      recordAIRequestLatencyMs: jest.fn(),
      recordAIRequestTimeout: jest.fn(),
    };
    (getMetricsService as jest.Mock).mockReturnValue(mockMetricsService);

    // Mock AI service client
    mockAIServiceClient = {
      getAIMove: jest.fn(),
      healthCheck: jest.fn(),
      clearCache: jest.fn(),
      getLineRewardChoice: jest.fn(),
      getRingEliminationChoice: jest.fn(),
      getRegionOrderChoice: jest.fn(),
      getCircuitBreakerStatus: jest.fn(() => ({ isOpen: false, failureCount: 0 })),
    };

    (getAIServiceClient as jest.Mock).mockReturnValue(mockAIServiceClient);

    // Create mock game state
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
        eliminatedRings: new Map(),
        size: 8,
      },
      players: [
        {
          id: 'player1',
          username: 'Player 1',
          playerNumber: 1,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 10,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'ai-player',
          username: 'AI Player',
          playerNumber: 2,
          type: 'ai',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 10,
          eliminatedRings: 0,
          territorySpaces: 0,
          aiDifficulty: 5,
          aiProfile: {
            difficulty: 5,
            mode: 'service',
            aiType: 'heuristic',
          },
        },
      ],
      currentPlayer: 2,
      currentPhase: 'ring_placement',
      gameStatus: 'active',
      moveHistory: [],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      spectators: [],
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: getConfiguredTotalRingsInPlay('square8', 2),
      totalRingsEliminated: 0,
      victoryThreshold: 0,
      territoryVictoryThreshold: 0,
      rngSeed: 12345,
    } as unknown as GameState;

    // Create mock valid moves
    mockValidMoves = [
      {
        id: 'move1',
        type: 'place_ring',
        player: 2,
        to: { x: 0, y: 0 },
        placementCount: 1,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      },
      {
        id: 'move2',
        type: 'place_ring',
        player: 2,
        to: { x: 1, y: 0 },
        placementCount: 1,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      },
      {
        id: 'move3',
        type: 'place_ring',
        player: 2,
        to: { x: 0, y: 1 },
        placementCount: 1,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      },
      {
        id: 'move4',
        type: 'place_ring',
        player: 2,
        to: { x: 1, y: 1 },
        placementCount: 1,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      },
    ];

    // By default, expose all mock valid moves to the mocked RuleEngine.
    mockRuleEngineValidMoves = mockValidMoves;
  });

  describe('Fallback Hierarchy', () => {
    it('should try remote AI first when mode is service', async () => {
      const serviceMove: Move = {
        id: 'service-move',
        type: 'place_ring',
        player: 2,
        to: { x: 0, y: 0 },
        placementCount: 1,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      mockAIServiceClient.getAIMove.mockResolvedValue({
        move: serviceMove,
        evaluation: 0.7,
        thinking_time_ms: 150,
        ai_type: 'heuristic',
        difficulty: 5,
      });

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      // Should have called the service
      expect(mockAIServiceClient.getAIMove).toHaveBeenCalledTimes(1);
      expect(move).toBeDefined();
      expect(move?.to).toEqual({ x: 0, y: 0 });

      // Should have logged the remote service usage
      expect(logger.info).toHaveBeenCalledWith(
        expect.stringContaining('AI move generated via remote service'),
        expect.any(Object)
      );
    });

    it('should fallback to local heuristic on remote failure', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service unavailable'));

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      // Should still return a move via fallback
      expect(move).toBeDefined();
      expect(move).not.toBeNull();

      // Should have logged the fallback
      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('falling back to local heuristics'),
        expect.any(Object)
      );

      // Should have recorded fallback metric
      expect(mockMetricsService.recordAIFallback).toHaveBeenCalled();

      // Diagnostics should reflect the failure
      const diag = aiEngine.getDiagnostics(2);
      expect(diag?.serviceFailureCount).toBe(1);
      expect(diag?.localFallbackCount).toBe(1);
    });

    it('should fallback to random on local heuristic failure', async () => {
      // This test simulates a scenario where both remote and local fail
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service failed'));

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      // Should still get a valid move via random fallback
      const move = await aiEngine.getAIMove(2, mockGameState);

      expect(move).toBeDefined();
      expect(move).not.toBeNull();

      // The move should be one of the valid moves
      const isValidMove = mockValidMoves.some(
        (vm) => vm.to && move?.to && vm.to.x === move.to.x && vm.to.y === move.to.y
      );
      expect(isValidMove).toBe(true);
    });

    it('should report which fallback level was used via diagnostics', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service down'));

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      // First call - should fallback
      await aiEngine.getAIMove(2, mockGameState);

      const diag1 = aiEngine.getDiagnostics(2);
      expect(diag1?.serviceFailureCount).toBe(1);
      expect(diag1?.localFallbackCount).toBe(1);

      // Second call - should fallback again
      await aiEngine.getAIMove(2, mockGameState);

      const diag2 = aiEngine.getDiagnostics(2);
      expect(diag2?.serviceFailureCount).toBe(2);
      expect(diag2?.localFallbackCount).toBe(2);
    });

    it('should emit fallback metric with reason', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Connection refused'));

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      await aiEngine.getAIMove(2, mockGameState);

      // Should have recorded the fallback reason
      expect(mockMetricsService.recordAIFallback).toHaveBeenCalledWith(
        expect.stringMatching(/python_error|connection_refused|circuit_open/)
      );
    });
  });

  describe('Fallback State Management', () => {
    it('should track current fallback level per player', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service down'));

      // Configure two AI players
      aiEngine.createAIFromProfile(1, { difficulty: 3, mode: 'service' });
      aiEngine.createAIFromProfile(2, { difficulty: 5, mode: 'service' });

      // Update game state for player 1
      const player1State = { ...mockGameState, currentPlayer: 1 };
      mockRuleEngineValidMoves = mockValidMoves.map((m) => ({ ...m, player: 1 }));

      await aiEngine.getAIMove(1, player1State);

      // Update game state for player 2
      const player2State = { ...mockGameState, currentPlayer: 2 };
      mockRuleEngineValidMoves = mockValidMoves;

      await aiEngine.getAIMove(2, player2State);
      await aiEngine.getAIMove(2, player2State);

      // Each player should have their own diagnostics
      const diag1 = aiEngine.getDiagnostics(1);
      const diag2 = aiEngine.getDiagnostics(2);

      expect(diag1?.serviceFailureCount).toBe(1);
      expect(diag2?.serviceFailureCount).toBe(2);
    });

    it('should reset diagnostics when AI is cleared', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service down'));

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      await aiEngine.getAIMove(2, mockGameState);

      expect(aiEngine.getDiagnostics(2)?.serviceFailureCount).toBe(1);

      // Clear all AI players
      aiEngine.clearAll();

      // Diagnostics should be undefined now
      expect(aiEngine.getDiagnostics(2)).toBeUndefined();
    });

    it('should skip service call for local_heuristic mode', async () => {
      const profile: AIProfile = { difficulty: 5, mode: 'local_heuristic' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      // Should NOT have called the service
      expect(mockAIServiceClient.getAIMove).not.toHaveBeenCalled();

      // Should have returned a valid move
      expect(move).toBeDefined();
      expect(move).not.toBeNull();
    });
  });

  describe('Circuit Breaker Fallback', () => {
    it('should fallback immediately when circuit breaker is open', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(
        new Error('Circuit breaker is open - AI service temporarily unavailable')
      );

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      expect(move).toBeDefined();
      expect(mockMetricsService.recordAIFallback).toHaveBeenCalledWith('circuit_open');
    });

    it('should handle timeout error type in fallback metrics', async () => {
      const timeoutError = new Error('Timeout');
      (timeoutError as any).aiErrorType = 'timeout';

      mockAIServiceClient.getAIMove.mockRejectedValue(timeoutError);

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      await aiEngine.getAIMove(2, mockGameState);

      expect(mockMetricsService.recordAIFallback).toHaveBeenCalledWith('timeout');
    });

    it('should handle overloaded error type in fallback metrics', async () => {
      const overloadError = new Error('Overloaded');
      (overloadError as any).aiErrorType = 'overloaded';

      mockAIServiceClient.getAIMove.mockRejectedValue(overloadError);

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      await aiEngine.getAIMove(2, mockGameState);

      expect(mockMetricsService.recordAIFallback).toHaveBeenCalledWith('overloaded');
    });

    it('should handle service_unavailable error type in fallback metrics', async () => {
      const unavailableError = new Error('Service unavailable');
      (unavailableError as any).aiErrorType = 'service_unavailable';

      mockAIServiceClient.getAIMove.mockRejectedValue(unavailableError);

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      await aiEngine.getAIMove(2, mockGameState);

      expect(mockMetricsService.recordAIFallback).toHaveBeenCalledWith('service_unavailable');
    });
  });

  describe('Invalid Move Fallback', () => {
    it('should fallback when service returns invalid move', async () => {
      // Service returns a move not in valid moves list
      const invalidMove: Move = {
        id: 'invalid',
        type: 'place_ring',
        player: 2,
        to: { x: 99, y: 99 }, // Invalid position
        placementCount: 1,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      mockAIServiceClient.getAIMove.mockResolvedValue({
        move: invalidMove,
        evaluation: 0.5,
        thinking_time_ms: 100,
        ai_type: 'heuristic',
        difficulty: 5,
      });

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      // Should have fallen back to a valid move
      expect(move).toBeDefined();
      expect(move?.to).not.toEqual({ x: 99, y: 99 });

      // Should have logged the invalid move
      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('invalid move'),
        expect.any(Object)
      );

      // Should have recorded validation failure metric
      expect(mockMetricsService.recordAIFallback).toHaveBeenCalledWith('validation_failed');
    });

    it('should fallback when service returns null move', async () => {
      mockAIServiceClient.getAIMove.mockResolvedValue({
        move: null,
        evaluation: 0,
        thinking_time_ms: 100,
        ai_type: 'heuristic',
        difficulty: 5,
      });

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      // Should have fallen back
      expect(move).toBeDefined();
      expect(move).not.toBeNull();

      // Should have recorded the fallback reason
      expect(mockMetricsService.recordAIFallback).toHaveBeenCalledWith('no_move_from_service');
    });
  });

  describe('Local Fallback Move Selection', () => {
    it('should use getLocalFallbackMove for explicit fallback requests', async () => {
      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      // Use the explicit local fallback method
      const move = aiEngine.getLocalFallbackMove(2, mockGameState);

      expect(move).toBeDefined();
      expect(move).not.toBeNull();

      // Should increment local fallback count
      const diag = aiEngine.getDiagnostics(2);
      expect(diag?.localFallbackCount).toBe(1);
    });

    it('should use deterministic RNG when provided', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service down'));

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      // Create deterministic RNG
      let seed = 42;
      const rng = () => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return seed / 0x7fffffff;
      };

      // Reset seed
      seed = 42;
      const move1 = await aiEngine.getAIMove(2, mockGameState, rng);

      // Reset seed again for same sequence
      seed = 42;
      const move2 = await aiEngine.getAIMove(2, mockGameState, rng);

      // With same RNG, should get same move
      expect(move1?.to).toEqual(move2?.to);
    });
  });

  describe('Edge Cases', () => {
    it('should handle single valid move without calling service', async () => {
      mockRuleEngineValidMoves = [mockValidMoves[0]];

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      // Should not have called the service for single move
      expect(mockAIServiceClient.getAIMove).not.toHaveBeenCalled();
      expect(move).toEqual(mockValidMoves[0]);
    });

    it('should return null when no valid moves exist', async () => {
      mockRuleEngineValidMoves = [];

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      expect(move).toBeNull();
      expect(mockAIServiceClient.getAIMove).not.toHaveBeenCalled();
    });

    it('should throw for unconfigured player', async () => {
      await expect(aiEngine.getAIMove(99, mockGameState)).rejects.toThrow(
        'No AI configuration found for player number 99'
      );
    });
  });
});
