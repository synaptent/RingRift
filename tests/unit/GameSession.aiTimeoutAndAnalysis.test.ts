/**
 * Focused branch coverage tests for GameSession AI timeout handling
 * and analysis-mode position evaluation.
 *
 * NOTE: Shadow mode tests removed - FSM is now canonical (shadow mode no longer exists)
 */

import type { Server as SocketIOServer } from 'socket.io';
import { GameSession } from '../../src/server/game/GameSession';
import type { GameState, Move } from '../../src/shared/types/game';
// RulesResult import removed - shadow mode tests no longer exist

// Mocks
jest.mock('../../src/shared/utils/timeout', () => ({
  runWithTimeout: jest.fn(),
}));

jest.mock('../../src/server/game/ai/AIEngine', () => ({
  globalAIEngine: {
    getAIConfig: jest.fn(),
    createAI: jest.fn(),
    createAIFromProfile: jest.fn(),
    getAIMove: jest.fn(),
    getLocalFallbackMove: jest.fn(),
    chooseLocalMoveFromCandidates: jest.fn(),
    getDiagnostics: jest.fn(() => ({
      serviceFailureCount: 0,
      localFallbackCount: 0,
    })),
  },
}));

jest.mock('../../src/server/services/AIServiceClient', () => ({
  getAIServiceClient: jest.fn(),
}));

// NOTE: ShadowModeComparator mock removed - FSM is now canonical

jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

jest.mock('../../src/server/config', () => ({
  config: {
    aiService: { requestTimeoutMs: 1234 },
    featureFlags: {
      analysisMode: {
        enabled: false,
      },
    },
    decisionPhaseTimeouts: {
      defaultTimeoutMs: 30000,
      warningBeforeTimeoutMs: 5000,
      extensionMs: 15000,
    },
    isTest: true,
    isDevelopment: true,
  },
}));

const { runWithTimeout } = require('../../src/shared/utils/timeout') as {
  runWithTimeout: jest.Mock;
};

const { globalAIEngine } = require('../../src/server/game/ai/AIEngine') as {
  globalAIEngine: {
    getAIMove: jest.Mock;
    getAIConfig: jest.Mock;
    createAI: jest.Mock;
  };
};

const { getAIServiceClient } = require('../../src/server/services/AIServiceClient') as {
  getAIServiceClient: jest.Mock;
};

// NOTE: shadowComparator and turnAdapterModule imports removed - FSM is now canonical

const { logger } = require('../../src/server/utils/logger');
const { config } = require('../../src/server/config');

const createMockIo = (): jest.Mocked<SocketIOServer> =>
  ({
    to: jest.fn().mockReturnThis(),
    emit: jest.fn(),
    sockets: {
      adapter: { rooms: new Map() },
      sockets: new Map(),
    },
  }) as any;

const createMinimalState = (): GameState =>
  ({
    id: 'game-1',
    boardType: 'square8',
    gameStatus: 'active',
    currentPlayer: 1,
    currentPhase: 'ring_placement',
    players: [],
    spectators: [],
    moveHistory: [],
    board: {
      type: 'square8',
      size: 8,
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      eliminatedRings: { 1: 0, 2: 0 },
      formedLines: [],
    },
    timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
  }) as any;

describe('GameSession.getAIMoveResultWithTimeout branches', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('passes timeoutMs through to runWithTimeout without adding extra delay layers', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    const fakeMove: Move = {
      id: 'm-timeout-prop',
      type: 'place_ring',
      player: 1,
      to: { x: 1, y: 1 },
      moveNumber: 1,
      timestamp: new Date(),
      thinkTime: 0,
    };

    // Ensure globalAIEngine.getAIMove is actually called by the operation.
    globalAIEngine.getAIMove.mockResolvedValueOnce(fakeMove);

    runWithTimeout.mockImplementationOnce(async (operation, options) => {
      // The timeout budget should be exactly what the caller provided.
      expect(options.timeoutMs).toBe(1234);

      const value = await operation();
      return {
        kind: 'ok',
        value,
        durationMs: 1,
      };
    });

    const state = createMinimalState();
    const result = await (session as any).getAIMoveResultWithTimeout(1, state, 1234);

    // getAIMove signature: (playerNumber, gameState, rng?, options?)
    expect(globalAIEngine.getAIMove).toHaveBeenCalledWith(1, state, undefined, undefined);
    expect(result.move).toBe(fakeMove);
  });

  it('returns move when runWithTimeout resolves with ok value', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    const fakeMove: Move = {
      id: 'm1',
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      moveNumber: 1,
      timestamp: new Date(),
      thinkTime: 0,
    };

    runWithTimeout.mockResolvedValueOnce({
      kind: 'ok',
      value: {
        move: fakeMove,
        telemetry: {
          modelVersion: null,
          modelPath: null,
          aiTier: 5,
          latencyMs: 100,
          fallbackUsed: false,
          fallbackReason: null,
          aiType: 'unknown',
          source: 'python_service',
        },
      },
    });

    const state = createMinimalState();
    const result = await (session as any).getAIMoveResultWithTimeout(1, state, 1000);

    expect(result.move).toBe(fakeMove);
  });

  it('returns move telemetry result when telemetry-capable AIEngine method is available', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    const fakeMove: Move = {
      id: 'm1',
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      moveNumber: 1,
      timestamp: new Date(),
      thinkTime: 0,
    };

    (globalAIEngine as any).getAIMoveWithTelemetry = jest.fn().mockResolvedValueOnce({
      move: fakeMove,
      telemetry: {
        modelVersion: 'v2.0.0',
        modelPath: 'models/canonical_square8_2p.pth',
        aiTier: 8,
        latencyMs: 912,
        fallbackUsed: false,
        fallbackReason: null,
        aiType: 'mcts',
        source: 'python_service',
      },
    });

    runWithTimeout.mockImplementationOnce(async (operation) => ({
      kind: 'ok',
      value: await operation(),
      durationMs: 1,
    }));

    const state = createMinimalState();
    const result = await (session as any).getAIMoveResultWithTimeout(1, state, 1000);

    expect((globalAIEngine as any).getAIMoveWithTelemetry).toHaveBeenCalledWith(
      1,
      state,
      undefined,
      undefined
    );
    expect(result).toEqual({
      move: fakeMove,
      telemetry: expect.objectContaining({
        modelVersion: 'v2.0.0',
        modelPath: 'models/canonical_square8_2p.pth',
        aiTier: 8,
        fallbackUsed: false,
        source: 'python_service',
      }),
    });

    delete (globalAIEngine as any).getAIMoveWithTelemetry;
  });

  it('returns null when runWithTimeout resolves with ok and null value', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    runWithTimeout.mockResolvedValueOnce({ kind: 'ok', value: null });

    const state = createMinimalState();
    const result = await (session as any).getAIMoveResultWithTimeout(1, state, 1000);

    expect(result.move).toBeNull();
  });

  it('throws timeout error when runWithTimeout reports timeout', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    runWithTimeout.mockResolvedValueOnce({ kind: 'timeout' });

    const state = createMinimalState();

    await expect((session as any).getAIMoveResultWithTimeout(1, state, 1000)).rejects.toMatchObject(
      {
        message: 'AI request timeout',
        isTimeout: true,
      }
    );
  });

  it('throws cancellation error when runWithTimeout reports canceled', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    runWithTimeout.mockResolvedValueOnce({
      kind: 'canceled',
      cancellationReason: 'manual',
    });

    const state = createMinimalState();

    await expect((session as any).getAIMoveResultWithTimeout(1, state, 1000)).rejects.toMatchObject(
      {
        message: 'AI request canceled',
        cancellationReason: 'manual',
      }
    );
  });

  it('throws guard error for unexpected TimedOperationResult kind', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    runWithTimeout.mockResolvedValueOnce({ kind: 'unexpected_kind' });

    const state = createMinimalState();

    await expect((session as any).getAIMoveResultWithTimeout(1, state, 1000)).rejects.toThrow(
      'Unhandled TimedOperationResult outcome in getAIMoveResultWithTimeout'
    );
  });

  it('derives aiRequestTimeoutMs from config.aiService.requestTimeoutMs', () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    // Private field, but stable for tests: host-level AI timeout budget.
    expect((session as any).aiRequestTimeoutMs).toBe(config.aiService.requestTimeoutMs);
  });
});

describe('GameSession analysis mode and position evaluation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('isAnalysisModeEnabled reflects feature flag', () => {
    const io = createMockIo();

    config.featureFlags.analysisMode.enabled = false;
    const sessionDisabled = new GameSession('game-1', io, {} as any, new Map());
    expect((sessionDisabled as any).isAnalysisModeEnabled()).toBe(false);

    config.featureFlags.analysisMode.enabled = true;
    const sessionEnabled = new GameSession('game-2', io, {} as any, new Map());
    expect((sessionEnabled as any).isAnalysisModeEnabled()).toBe(true);
  });

  it('evaluateAndBroadcastPosition emits position_evaluation on success', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    const evaluatePositionMulti = jest.fn().mockResolvedValue({
      move_number: 5,
      board_type: 'square8',
      per_player: { 1: { score: 0.1 }, 2: { score: -0.1 } },
      engine_profile: { name: 'test-profile' },
      evaluation_scale: 'centipawns',
      generated_at: '2025-01-01T00:00:00.000Z',
    });

    getAIServiceClient.mockReturnValue({ evaluatePositionMulti });

    const state = createMinimalState();

    await (session as any).evaluateAndBroadcastPosition(state);

    expect(getAIServiceClient).toHaveBeenCalled();
    expect(evaluatePositionMulti).toHaveBeenCalledWith(state);
    expect(io.to).toHaveBeenCalledWith('game-1');
    expect(io.emit).toHaveBeenCalledWith(
      'position_evaluation',
      expect.objectContaining({
        type: 'position_evaluation',
        data: expect.objectContaining({
          gameId: 'game-1',
          moveNumber: 5,
          boardType: 'square8',
        }),
      })
    );
  });

  it('evaluateAndBroadcastPosition logs warning when AI service fails', async () => {
    const io = createMockIo();
    const session = new GameSession('game-err', io, {} as any, new Map());

    const evaluatePositionMulti = jest.fn().mockRejectedValue(new Error('service down'));
    getAIServiceClient.mockReturnValue({ evaluatePositionMulti });

    const state = createMinimalState();

    await (session as any).evaluateAndBroadcastPosition(state);

    expect(logger.warn).toHaveBeenCalledWith(
      'Failed to emit position evaluation',
      expect.objectContaining({
        gameId: 'game-err',
        error: 'service down',
      })
    );
    expect(io.emit).not.toHaveBeenCalledWith('position_evaluation', expect.anything());
  });
});

// NOTE: GameSession orchestrator shadow path tests removed - FSM is now canonical
// Shadow mode methods (applyMoveWithOrchestratorShadow, runOrchestratorShadow) no longer exist
