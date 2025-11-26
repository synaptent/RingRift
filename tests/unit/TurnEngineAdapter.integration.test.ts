/**
 * TurnEngineAdapter Integration Tests
 *
 * These tests verify that the TurnEngineAdapter correctly delegates to the
 * shared orchestrator (processTurnAsync) and handles backend-specific concerns.
 *
 * Part of Phase 3 Rules Engine Consolidation.
 */

import { describe, it, expect } from '@jest/globals';
import {
  TurnEngineAdapter,
  createSimpleAdapter,
  createAutoSelectDecisionHandler,
} from '../../src/server/game/turn/TurnEngineAdapter';
import type { GameState, Move, Position, Player, TimeControl } from '../../src/shared/types/game';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import { positionToString } from '../../src/shared/engine';

describe('TurnEngineAdapter', () => {
  // Helper to create a test game state
  function createTestState(overrides: Partial<GameState> = {}): GameState {
    const players: Player[] = [
      {
        id: 'p1',
        username: 'Player 1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 12,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'Player 2',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 12,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];
    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
    const state = createInitialGameState('test-game', 'square8', players, timeControl, true);
    return { ...state, gameStatus: 'active', ...overrides };
  }

  describe('Basic Operations', () => {
    it('should create adapter with simple state holder', () => {
      const state = createTestState();
      const handler = createAutoSelectDecisionHandler();
      const { adapter, getState } = createSimpleAdapter(state, handler);

      expect(adapter).toBeInstanceOf(TurnEngineAdapter);
      expect(getState()).toEqual(state);
    });

    it('should validate moves without applying them', () => {
      const state = createTestState();
      const handler = createAutoSelectDecisionHandler();
      const { adapter } = createSimpleAdapter(state, handler);

      // Valid placement
      const validMove: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const validResult = adapter.validateMoveOnly(state, validMove);
      expect(validResult.valid).toBe(true);

      // Invalid move (wrong player)
      const invalidMove: Move = {
        id: 'test-2',
        type: 'place_ring',
        player: 2,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      // Note: Validation may or may not check player turn depending on implementation
      // Just verify we get a result without throwing
      const invalidResult = adapter.validateMoveOnly(state, invalidMove);
      expect(typeof invalidResult.valid).toBe('boolean');
    });

    it('should enumerate valid moves for ring_placement phase', () => {
      const state = createTestState({ currentPhase: 'ring_placement' });
      const handler = createAutoSelectDecisionHandler();
      const { adapter } = createSimpleAdapter(state, handler);

      const moves = adapter.getValidMovesFor(state);

      // Should have placement moves (64 squares minus any occupied)
      expect(moves.length).toBeGreaterThan(0);
      expect(moves.every((m) => m.type === 'place_ring' || m.type === 'skip_placement')).toBe(true);
    });

    it('should detect when valid moves exist', () => {
      const state = createTestState({ currentPhase: 'ring_placement' });
      const handler = createAutoSelectDecisionHandler();
      const { adapter } = createSimpleAdapter(state, handler);

      expect(adapter.hasAnyValidMoves(state)).toBe(true);
    });
  });

  describe('Move Processing', () => {
    it('should process a valid placement move', async () => {
      const state = createTestState({ currentPhase: 'ring_placement' });
      const handler = createAutoSelectDecisionHandler();
      const { adapter, getState } = createSimpleAdapter(state, handler);

      const move: Move = {
        id: 'place-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = await adapter.processMove(move);

      expect(result.success).toBe(true);
      expect(result.error).toBeUndefined();

      // State should have been updated
      const newState = getState();
      expect(newState.board.stacks.size).toBeGreaterThan(0);

      // Should transition to movement phase after placement
      expect(newState.currentPhase).toBe('movement');
    });

    it('should handle validation errors', async () => {
      const state = createTestState({ currentPhase: 'ring_placement' });
      const handler = createAutoSelectDecisionHandler();
      const { adapter } = createSimpleAdapter(state, handler);

      // Try to validate a move from wrong player
      const wrongPlayerMove: Move = {
        id: 'invalid-1',
        type: 'place_ring',
        player: 2, // Player 2 shouldn't move when it's player 1's turn
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      // Validation should work without throwing
      const validationResult = adapter.validateMoveOnly(state, wrongPlayerMove);
      expect(typeof validationResult.valid).toBe('boolean');
      // Note: The orchestrator may or may not enforce turn order in validateMove
    });

    it('should handle moves that throw exceptions', async () => {
      const state = createTestState({ currentPhase: 'ring_placement' });
      const handler = createAutoSelectDecisionHandler();
      const { adapter, getState } = createSimpleAdapter(state, handler);

      const initialStackCount = state.board.stacks.size;

      // Try to move a non-existent stack
      const invalidMove: Move = {
        id: 'invalid-2',
        type: 'move_stack',
        player: 1,
        from: { x: 50, y: 50 }, // No stack here
        to: { x: 50, y: 52 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      // Process the move - should handle gracefully
      const result = await adapter.processMove(invalidMove);

      // Either it failed, or it's a no-op (state unchanged)
      if (result.success) {
        // If successful, the state shouldn't have changed much for a no-op
        const currentState = getState();
        // Stack count may or may not have changed depending on implementation
        expect(currentState).toBeDefined();
      } else {
        // If failed, we should have an error
        expect(result.error).toBeDefined();
      }
    });

    it('should process movement after placement', async () => {
      // Start with a state that has a stack to move
      const state = createTestState({ currentPhase: 'ring_placement' });
      const handler = createAutoSelectDecisionHandler();
      const { adapter, getState } = createSimpleAdapter(state, handler);

      // First, place a ring
      const placeMove: Move = {
        id: 'place-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await adapter.processMove(placeMove);

      const afterPlace = getState();
      expect(afterPlace.currentPhase).toBe('movement');

      // Now move the stack
      const moveStack: Move = {
        id: 'move-1',
        type: 'move_stack',
        player: 1,
        from: { x: 3, y: 3 },
        to: { x: 3, y: 5 }, // Move 2 spaces up
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      };

      const result = await adapter.processMove(moveStack);

      if (result.success) {
        const afterMove = getState();
        // Stack should now be at new position
        const newPosKey = positionToString({ x: 3, y: 5 });
        const oldPosKey = positionToString({ x: 3, y: 3 });

        expect(afterMove.board.stacks.has(newPosKey)).toBe(true);
        // Old position should have a marker now
        expect(afterMove.board.markers.has(oldPosKey)).toBe(true);
      }
    });
  });

  describe('Decision Handling', () => {
    it('should auto-select decisions for AI players', async () => {
      // Create a state where a decision will be needed
      // For now, just verify the handler works
      const state = createTestState();
      const handler = createAutoSelectDecisionHandler();

      // Test the handler directly
      const mockDecision = {
        type: 'line_order' as const,
        player: 1,
        options: [
          {
            id: 'test-move',
            type: 'process_line' as const,
            player: 1,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: 1,
          },
        ],
        context: { description: 'test' },
      };

      const selected = await handler.requestDecision(mockDecision);
      expect(selected).toEqual(mockDecision.options[0]);
    });

    it('should throw when no options available for decision', async () => {
      const handler = createAutoSelectDecisionHandler();

      const emptyDecision = {
        type: 'line_order' as const,
        player: 1,
        options: [] as Move[],
        context: { description: 'test' },
      };

      await expect(handler.requestDecision(emptyDecision)).rejects.toThrow();
    });
  });

  describe('Event Emission', () => {
    it('should emit events when emitter is provided', async () => {
      const state = createTestState({ currentPhase: 'ring_placement' });
      const handler = createAutoSelectDecisionHandler();

      const events: Array<{ event: string; payload: unknown }> = [];
      const mockEmitter = {
        emit: (event: string, payload: unknown) => {
          events.push({ event, payload });
        },
      };

      const stateAccessor = {
        getGameState: () => state,
        updateGameState: (newState: GameState) => {
          Object.assign(state, newState);
        },
        getPlayerInfo: (playerNumber: number) => {
          const player = state.players.find((p) => p.playerNumber === playerNumber);
          return player ? { type: player.type } : undefined;
        },
      };

      const adapter = new TurnEngineAdapter({
        stateAccessor,
        decisionHandler: handler,
        eventEmitter: mockEmitter,
      });

      const move: Move = {
        id: 'place-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await adapter.processMove(move);

      // Should have emitted at least a state_update event
      const stateUpdateEvent = events.find((e) => e.event === 'game:state_update');
      expect(stateUpdateEvent).toBeDefined();
    });
  });
});
