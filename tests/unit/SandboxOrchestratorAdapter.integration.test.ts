/**
 * Integration tests for SandboxOrchestratorAdapter
 *
 * These tests verify that the sandbox adapter correctly delegates
 * to the shared orchestrator and handles state management properly.
 */

import {
  SandboxOrchestratorAdapter,
  SandboxStateAccessor,
  SandboxDecisionHandler,
  createSandboxAdapter,
  createAISandboxAdapter,
} from '../../src/client/sandbox/SandboxOrchestratorAdapter';
import { BOARD_CONFIGS } from '../../src/shared/engine';
import type {
  GameState,
  Move,
  Position,
  BoardType,
  Player,
  TimeControl,
} from '../../src/shared/engine';
import type { PendingDecision } from '../../src/shared/engine/orchestration/types';

// ═══════════════════════════════════════════════════════════════════════════
// Test Helpers
// ═══════════════════════════════════════════════════════════════════════════

function createTestGameState(boardType: BoardType = 'square8', numPlayers: number = 2): GameState {
  const config = BOARD_CONFIGS[boardType];

  const players: Player[] = Array.from({ length: numPlayers }, (_, idx) => ({
    id: `player-${idx + 1}`,
    username: `Player ${idx + 1}`,
    type: 'human' as const,
    playerNumber: idx + 1,
    isReady: true,
    timeRemaining: 600000,
    ringsInHand: config.ringsPerPlayer,
    eliminatedRings: 0,
    territorySpaces: 0,
  }));

  const board = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {} as { [player: number]: number },
    size: config.size,
    type: boardType,
  };

  players.forEach((p) => {
    board.eliminatedRings[p.playerNumber] = 0;
  });

  return {
    id: 'test-game',
    boardType,
    rngSeed: 12345,
    board,
    players,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    gameStatus: 'active',
    spectators: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: numPlayers,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: Math.floor((config.ringsPerPlayer * numPlayers) / 2) + 1,
    territoryVictoryThreshold: Math.floor(config.totalSpaces / 2) + 1,
  };
}

function createTestStateAccessor(initialState: GameState): {
  accessor: SandboxStateAccessor;
  getState: () => GameState;
} {
  let state = initialState;

  const accessor: SandboxStateAccessor = {
    getGameState: () => ({ ...state }),
    updateGameState: (newState: GameState) => {
      state = newState;
    },
    getPlayerInfo: (_playerId: string) => ({ type: 'human' as const }),
  };

  return { accessor, getState: () => state };
}

function createTestDecisionHandler(): {
  handler: SandboxDecisionHandler;
  getDecisions: () => PendingDecision[];
} {
  const decisions: PendingDecision[] = [];

  const handler: SandboxDecisionHandler = {
    requestDecision: async (decision: PendingDecision): Promise<Move> => {
      decisions.push(decision);
      // Return first option
      if (decision.options.length > 0) {
        return decision.options[0];
      }
      throw new Error('No options available');
    },
  };

  return { handler, getDecisions: () => decisions };
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('SandboxOrchestratorAdapter', () => {
  describe('Construction', () => {
    it('should construct with required dependencies', () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      expect(adapter).toBeDefined();
    });

    it('should construct with optional callbacks', () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const onMoveStarted = jest.fn();
      const onMoveCompleted = jest.fn();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
        callbacks: {
          onMoveStarted,
          onMoveCompleted,
        },
      });

      expect(adapter).toBeDefined();
    });
  });

  describe('State Access', () => {
    it('should return game state via accessor', () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      const returnedState = adapter.getGameState();
      expect(returnedState.boardType).toBe(state.boardType);
      expect(returnedState.currentPlayer).toBe(1);
    });

    it('should report current player correctly', () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      expect(adapter.getCurrentPlayer()).toBe(1);
    });

    it('should report current phase correctly', () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      expect(adapter.getCurrentPhase()).toBe('ring_placement');
    });

    it('should detect game over state', () => {
      const state = createTestGameState();
      const { accessor, getState } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      expect(adapter.isGameOver()).toBe(false);

      // Mark game as completed
      accessor.updateGameState({ ...getState(), gameStatus: 'completed' });
      expect(adapter.isGameOver()).toBe(true);
    });
  });

  describe('Move Validation', () => {
    it('should validate placement moves', () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      // Valid placement
      const validMove: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = adapter.validateMove(validMove);
      expect(result.valid).toBe(true);
    });

    it('should reject invalid placement moves', () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      // Invalid placement (out of bounds)
      const invalidMove: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 100, y: 100 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = adapter.validateMove(invalidMove);
      expect(result.valid).toBe(false);
    });
  });

  describe('Valid Move Enumeration', () => {
    it('should enumerate valid placement positions', () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      const moves = adapter.getValidMoves();
      expect(moves.length).toBeGreaterThan(0);

      // All moves should be placements in initial state
      const placementMoves = moves.filter((m) => m.type === 'place_ring');
      expect(placementMoves.length).toBeGreaterThan(0);
    });
  });

  describe('Move Processing', () => {
    it('should process placement move successfully', async () => {
      const state = createTestGameState();
      const { accessor, getState } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      const move: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = await adapter.processMove(move);

      expect(result.success).toBe(true);
      expect(result.nextState.currentPhase).toBe('movement');

      // State should be updated
      const currentState = getState();
      expect(currentState.currentPhase).toBe('movement');
      expect(currentState.board.stacks.has('3,3')).toBe(true);
    });

    it('should fail for invalid moves', async () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      const invalidMove: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 100, y: 100 }, // Out of bounds
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = await adapter.processMove(invalidMove);

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });

    it('should include metadata in result', async () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      const move: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = await adapter.processMove(move);

      expect(result.metadata).toBeDefined();
      expect(result.metadata?.hashBefore).toBeDefined();
      expect(result.metadata?.hashAfter).toBeDefined();
      expect(result.metadata?.stateChanged).toBe(true);
      expect(typeof result.metadata?.durationMs).toBe('number');
    });
  });

  describe('Synchronous Processing', () => {
    it('should process move synchronously', () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      const move: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = adapter.processMoveSync(move);

      expect(result.success).toBe(true);
      expect(result.nextState.currentPhase).toBe('movement');
    });
  });

  describe('Preview Mode', () => {
    it('should preview move without affecting state', () => {
      const state = createTestGameState();
      const { accessor, getState } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      const move: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const originalState = getState();
      const preview = adapter.previewMove(move);

      // Preview should show the resulting state
      expect(preview.valid).toBe(true);
      expect(preview.nextState.board.stacks.has('3,3')).toBe(true);

      // Original state should be unchanged
      const currentState = getState();
      expect(currentState.currentPhase).toBe(originalState.currentPhase);
      expect(currentState.board.stacks.has('3,3')).toBe(false);
    });

    it('should return invalid for bad moves in preview', () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      const invalidMove: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 100, y: 100 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const preview = adapter.previewMove(invalidMove);
      expect(preview.valid).toBe(false);
      expect(preview.reason).toBeDefined();
    });
  });

  describe('Callbacks', () => {
    it('should call onMoveStarted callback', async () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();
      const onMoveStarted = jest.fn();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
        callbacks: { onMoveStarted },
      });

      const move: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await adapter.processMove(move);

      expect(onMoveStarted).toHaveBeenCalledWith(move);
    });

    it('should call onMoveCompleted callback', async () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();
      const onMoveCompleted = jest.fn();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
        callbacks: { onMoveCompleted },
      });

      const move: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await adapter.processMove(move);

      expect(onMoveCompleted).toHaveBeenCalled();
      expect(onMoveCompleted.mock.calls[0][0]).toEqual(move);
      expect(onMoveCompleted.mock.calls[0][1].success).toBe(true);
    });

    it('should call debugHook callback', async () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();
      const debugHook = jest.fn();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
        callbacks: { debugHook },
      });

      const move: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await adapter.processMove(move);

      expect(debugHook).toHaveBeenCalled();
    });
  });

  describe('Factory Functions', () => {
    it('should create adapter with createSandboxAdapter', () => {
      const state = createTestGameState();
      let currentState = state;

      const adapter = createSandboxAdapter(
        () => currentState,
        (newState) => {
          currentState = newState;
        },
        {
          requestDecision: async (decision) => decision.options[0],
        }
      );

      expect(adapter).toBeDefined();
      expect(adapter.getCurrentPlayer()).toBe(1);
    });

    it('should create AI adapter with createAISandboxAdapter', () => {
      const state = createTestGameState();
      let currentState = state;

      const adapter = createAISandboxAdapter(
        () => currentState,
        (newState) => {
          currentState = newState;
        }
      );

      expect(adapter).toBeDefined();
      expect(adapter.getCurrentPlayer()).toBe(1);
    });
  });

  describe('isMoveValid', () => {
    it('should check if move is valid', () => {
      const state = createTestGameState();
      const { accessor } = createTestStateAccessor(state);
      const { handler } = createTestDecisionHandler();

      const adapter = new SandboxOrchestratorAdapter({
        stateAccessor: accessor,
        decisionHandler: handler,
      });

      const validMove: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      expect(adapter.isMoveValid(validMove)).toBe(true);

      const invalidMove: Move = {
        id: 'test-2',
        type: 'place_ring',
        player: 1,
        to: { x: 100, y: 100 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      expect(adapter.isMoveValid(invalidMove)).toBe(false);
    });
  });
});
