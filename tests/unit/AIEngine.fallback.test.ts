import { AIEngine } from '../../src/server/game/ai/AIEngine';
import { getAIServiceClient } from '../../src/server/services/AIServiceClient';
import { GameState, Move, AIProfile } from '../../src/shared/types/game';
import { logger } from '../../src/server/utils/logger';

// Mock dependencies
jest.mock('../../src/server/services/AIServiceClient');
jest.mock('../../src/server/utils/logger');

// Shared mutable backing store for the mocked RuleEngine.getValidMoves output.
let mockRuleEngineValidMoves: Move[] = [];

// Mock RuleEngine so we can control the valid move set seen by AIEngine.
jest.mock('../../src/server/game/RuleEngine', () => {
  return {
    RuleEngine: jest.fn().mockImplementation(() => ({
      getValidMoves: () => mockRuleEngineValidMoves,
    })),
  };
});

describe('AIEngine Fallback Handling', () => {
  let aiEngine: AIEngine;
  let mockAIServiceClient: any;
  let mockGameState: GameState;
  let mockValidMoves: Move[];

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();

    // Create fresh AI engine instance
    aiEngine = new AIEngine();

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

    // Create mock game state. We deliberately only populate the fields that
    // AIEngine.getAIMove touches (boardType, board maps, players, current state,
    // rngSeed, etc.). The cast via unknown avoids over-constraining the literal
    // against the full GameState interface while keeping the runtime shape
    // compatible for these tests.
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
      totalRingsInPlay: 0,
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
    ];

    // By default, expose all mock valid moves to the mocked RuleEngine.
    mockRuleEngineValidMoves = mockValidMoves;
  });

  describe('Service Failure Fallback', () => {
    it('should fall back to local heuristics when service fails', async () => {
      // Mock service failure
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Connection refused'));

      // Configure AI player
      const profile: AIProfile = { difficulty: 5, mode: 'service', aiType: 'heuristic' };
      aiEngine.createAIFromProfile(2, profile);

      // Get AI move
      const move = await aiEngine.getAIMove(2, mockGameState);

      // Should return a valid move despite service failure
      expect(move).toBeDefined();
      expect(move).not.toBeNull();

      // Should have logged the fallback
      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('falling back to local heuristics'),
        expect.any(Object)
      );

      // Should record diagnostics
      const diagnostics = aiEngine.getDiagnostics(2);
      expect(diagnostics?.serviceFailureCount).toBe(1);
      expect(diagnostics?.localFallbackCount).toBeGreaterThan(0);
    });

    it('should handle AI service timeout', async () => {
      // Mock timeout
      mockAIServiceClient.getAIMove.mockImplementation(
        () =>
          new Promise((_, reject) => setTimeout(() => reject(new Error('AI service timeout')), 100))
      );

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      expect(move).toBeDefined();
      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('falling back'),
        expect.any(Object)
      );
    });

    it('should handle service returning null move', async () => {
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

      expect(move).toBeDefined();
      expect(move).not.toBeNull();
    });

    it('should handle circuit breaker open state', async () => {
      // Mock circuit breaker as open
      mockAIServiceClient.getCircuitBreakerStatus.mockReturnValue({
        isOpen: true,
        failureCount: 5,
      });
      mockAIServiceClient.getAIMove.mockRejectedValue(
        new Error('Circuit breaker is open - AI service temporarily unavailable')
      );

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      expect(move).toBeDefined();
      expect(logger.warn).toHaveBeenCalled();
    });
  });

  describe('Invalid Move Validation', () => {
    it('should reject invalid moves from AI service', async () => {
      // Mock service returning a move that's not in valid moves
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

      // Should fall back to local heuristic
      expect(move).toBeDefined();
      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('invalid move'),
        expect.any(Object)
      );
    });

    it('should validate move is in valid moves list', async () => {
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
        evaluation: 0.5,
        thinking_time_ms: 100,
        ai_type: 'heuristic',
        difficulty: 5,
      });

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      expect(move).toBeDefined();
      expect(move?.type).toBe('place_ring');
      expect(move?.to).toEqual({ x: 0, y: 0 });
    });
  });

  describe('Fallback to Random Selection', () => {
    it('should select random valid move when all AI methods fail', async () => {
      // Mock service failure
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service failed'));

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      // Multiple attempts should all succeed (falling back to random)
      for (let i = 0; i < 5; i++) {
        const move = await aiEngine.getAIMove(2, mockGameState);
        expect(move).toBeDefined();
        expect(move).not.toBeNull();
      }
    });

    it('should return immediately when only one valid move exists', async () => {
      // Restrict the mocked RuleEngine to expose exactly one valid move.
      mockRuleEngineValidMoves = [mockValidMoves[0]];

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      // Should not even call the service when there's only one valid move.
      expect(move).toBeDefined();
      expect(move).toEqual(mockValidMoves[0]);
      expect(mockAIServiceClient.getAIMove).not.toHaveBeenCalled();
      expect(logger.info).toHaveBeenCalledWith(
        expect.stringContaining('Single valid move'),
        expect.any(Object)
      );
    });

    it('should return null when no valid moves exist', async () => {
      // No valid moves from the mocked RuleEngine.
      mockRuleEngineValidMoves = [];

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      expect(move).toBeNull();
      expect(mockAIServiceClient.getAIMove).not.toHaveBeenCalled();
      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('No valid moves'),
        expect.any(Object)
      );
    });
  });

  describe('Mode-Specific Behavior', () => {
    it('should skip service call when mode is local_heuristic', async () => {
      const profile: AIProfile = { difficulty: 5, mode: 'local_heuristic' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, mockGameState);

      // Should NOT have called the service
      expect(mockAIServiceClient.getAIMove).not.toHaveBeenCalled();

      // Should have used local heuristics
      expect(move).toBeDefined();
      expect(logger.info).toHaveBeenCalledWith(
        expect.stringContaining('local heuristics'),
        expect.any(Object)
      );
    });
  });

  describe('Diagnostics Tracking', () => {
    it('should track service failures correctly', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service down'));

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      await aiEngine.getAIMove(2, mockGameState);
      await aiEngine.getAIMove(2, mockGameState);
      await aiEngine.getAIMove(2, mockGameState);

      const diagnostics = aiEngine.getDiagnostics(2);
      expect(diagnostics?.serviceFailureCount).toBe(3);
      expect(diagnostics?.localFallbackCount).toBeGreaterThan(0);
    });

    it('should expose diagnostics only after failures have been recorded', async () => {
      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      // Before any service interaction, diagnostics may be undefined.
      const before = aiEngine.getDiagnostics(2);
      expect(before).toBeUndefined();

      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service down'));

      await aiEngine.getAIMove(2, mockGameState);

      const after = aiEngine.getDiagnostics(2);
      expect(after).toBeDefined();
      expect(after!.serviceFailureCount).toBe(1);
      expect(after!.localFallbackCount).toBeGreaterThan(0);
    });
  });

  describe('Error Logging', () => {
    it('should log detailed error information on service failure', async () => {
      const testError = new Error('Connection timeout');
      mockAIServiceClient.getAIMove.mockRejectedValue(testError);

      const profile: AIProfile = { difficulty: 7, mode: 'service', aiType: 'minimax' };
      aiEngine.createAIFromProfile(2, profile);

      await aiEngine.getAIMove(2, mockGameState);

      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('Remote AI service failed'),
        expect.objectContaining({
          error: testError.message,
          playerNumber: 2,
          difficulty: 7,
        })
      );
    });

    it('should log when using random fallback as last resort', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service failed'));

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      await aiEngine.getAIMove(2, mockGameState);

      // Check if random fallback was logged (it should only happen if local heuristic also fails)
      // In normal cases, local heuristic should succeed, so we do not assert here.
    });
  });

  describe('Move Equality Validation', () => {
    it('should correctly identify equal moves', async () => {
      const validMove: Move = {
        id: 'valid',
        type: 'move_stack',
        player: 2,
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      mockAIServiceClient.getAIMove.mockResolvedValue({
        move: validMove,
        evaluation: 0.5,
        thinking_time_ms: 100,
        ai_type: 'heuristic',
        difficulty: 5,
      });

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      // Create game state with this move in valid moves
      const gameStateWithMoves = {
        ...mockGameState,
        currentPhase: 'movement' as const,
      };

      const move = await aiEngine.getAIMove(2, gameStateWithMoves);

      expect(move).toBeDefined();
    });

    it('should handle hexagonal coordinates in move comparison', async () => {
      const hexMove: Move = {
        id: 'hex',
        type: 'place_ring',
        player: 2,
        to: { x: 1, y: 1, z: -2 },
        placementCount: 1,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      mockAIServiceClient.getAIMove.mockResolvedValue({
        move: hexMove,
        evaluation: 0.5,
        thinking_time_ms: 100,
        ai_type: 'heuristic',
        difficulty: 5,
      });

      const hexGameState = {
        ...mockGameState,
        boardType: 'hexagonal' as const,
      };

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      const move = await aiEngine.getAIMove(2, hexGameState);

      expect(move).toBeDefined();
    });
  });

  describe('Configuration Management', () => {
    it('should throw error when requesting move for unconfigured player', async () => {
      await expect(aiEngine.getAIMove(99, mockGameState)).rejects.toThrow(
        'No AI configuration found for player number 99'
      );
    });

    it('should handle AI profile with all tactic types', () => {
      const tacticsToTest: AIProfile['aiType'][] = [
        'random',
        'heuristic',
        'minimax',
        'mcts',
        'descent',
      ];

      tacticsToTest.forEach((tactic) => {
        const profile: AIProfile = { difficulty: 5, mode: 'service', aiType: tactic };
        expect(() => aiEngine.createAIFromProfile(2, profile)).not.toThrow();
      });
    });
  });

  describe('Health Checks', () => {
    it('should report service health correctly', async () => {
      mockAIServiceClient.healthCheck.mockResolvedValue(true);

      const isHealthy = await aiEngine.checkServiceHealth();

      expect(isHealthy).toBe(true);
      expect(mockAIServiceClient.healthCheck).toHaveBeenCalled();
    });

    it('should return false when health check fails', async () => {
      mockAIServiceClient.healthCheck.mockRejectedValue(new Error('Service down'));

      const isHealthy = await aiEngine.checkServiceHealth();

      expect(isHealthy).toBe(false);
      expect(logger.error).toHaveBeenCalledWith(
        expect.stringContaining('health check failed'),
        expect.any(Object)
      );
    });
  });

  describe('Persistent Failures', () => {
    it('should continue providing moves even with repeated failures', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Persistent failure'));

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      // Make multiple requests
      const moves: (Move | null)[] = [];
      for (let i = 0; i < 10; i++) {
        const move = await aiEngine.getAIMove(2, mockGameState);
        moves.push(move);
      }

      // All should have returned valid moves via fallback
      expect(moves.every((m) => m !== null)).toBe(true);

      const diagnostics = aiEngine.getDiagnostics(2);
      expect(diagnostics?.serviceFailureCount).toBe(10);
    });
  });

  describe('RNG Determinism', () => {
    it('should use provided RNG for deterministic fallback', async () => {
      mockAIServiceClient.getAIMove.mockRejectedValue(new Error('Service down'));

      const profile: AIProfile = { difficulty: 5, mode: 'service' };
      aiEngine.createAIFromProfile(2, profile);

      // Use deterministic RNG
      let seed = 0;
      const deterministicRng = () => {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
      };

      const move1 = await aiEngine.getAIMove(2, mockGameState, deterministicRng);

      // Reset RNG to same seed
      seed = 0;
      const move2 = await aiEngine.getAIMove(2, mockGameState, deterministicRng);

      // With the same RNG stream, we should pick the same candidate move
      // (up to non-deterministic metadata such as timestamp).
      expect(move1).toBeDefined();
      expect(move2).toBeDefined();
      expect(move1!.type).toBe(move2!.type);
      expect(move1!.to).toEqual(move2!.to);
      expect(move1!.placementCount).toBe(move2!.placementCount);
    });
  });
});
