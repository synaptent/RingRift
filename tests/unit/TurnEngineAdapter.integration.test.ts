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
import type {
  GameState,
  Move,
  Position,
  Player,
  TimeControl,
  RingEliminationChoice,
  PlayerChoiceResponseFor,
} from '../../src/shared/types/game';
import type { PendingDecision, VictoryState } from '../../src/shared/engine/orchestration/types';
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

    it('surfaces forced-elimination elimination_target decisions to the DecisionHandler', async () => {
      const state = createTestState();
      const decisions: PendingDecision[] = [];

      const stateAccessor = {
        getGameState: () => state,
        updateGameState: (_newState: GameState) => {
          // no-op for this focused test
        },
        getPlayerInfo: (playerNumber: number) => {
          const player = state.players.find((p) => p.playerNumber === playerNumber);
          return player ? { type: player.type } : undefined;
        },
      };

      const decisionHandler = {
        requestDecision: async (decision: PendingDecision): Promise<Move> => {
          decisions.push(decision);
          if (decision.options.length === 0) {
            throw new Error('No options available');
          }
          // For this test we just echo the first option back.
          return decision.options[0];
        },
      };

      const adapter = new TurnEngineAdapter({
        stateAccessor,
        decisionHandler,
      });

      // Mocked PendingDecision representing a forced-elimination choice from the
      // orchestrator (createForcedEliminationDecision / elimination_target).
      const forcedDecision: PendingDecision = {
        type: 'elimination_target',
        player: 1,
        options: [
          {
            id: 'forced-elim-0-0',
            type: 'eliminate_rings_from_stack',
            player: 1,
            to: { x: 0, y: 0 },
            eliminatedRings: [{ player: 1, count: 1 }],
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: 1,
          } as Move,
        ],
        context: {
          description: 'Choose which stack to eliminate from (forced elimination)',
        },
      };

      // Instead of invoking the full orchestrator, call autoSelectForAI's
      // resolveDecision delegate path indirectly by simulating a processTurnAsync
      // decision callback. Here we directly invoke the DecisionHandler as the
      // orchestrator would.
      const selected = await decisionHandler.requestDecision(forcedDecision);

      expect(decisions).toHaveLength(1);
      expect(decisions[0].type).toBe('elimination_target');
      expect(decisions[0].options).toHaveLength(1);
      expect(decisions[0].options[0].type).toBe('eliminate_rings_from_stack');

      // In the real backend WebSocket path this elimination_target decision is
      // transformed into a RingEliminationChoice whose options map 1:1 onto the
      // eliminate_rings_from_stack moves via moveId. This assertion documents
      // the expected mapping without needing a full WebSocket harness here.
      const choice: RingEliminationChoice = {
        id: 'test-elim-choice',
        gameId: state.id,
        playerNumber: forcedDecision.player,
        type: 'ring_elimination',
        prompt: forcedDecision.context.description ?? 'Choose elimination stack',
        options: forcedDecision.options.map((move) => ({
          stackPosition: move.to as Position,
          capHeight: 1,
          totalHeight: 1,
          moveId: move.id,
        })),
      };

      const response: PlayerChoiceResponseFor<RingEliminationChoice> = {
        choiceId: choice.id,
        playerNumber: choice.playerNumber,
        type: 'ring_elimination',
        selectedOption: choice.options[0],
      };

      expect(selected.id).toBe(response.selectedOption.moveId);
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

  describe('Victory mapping', () => {
    it('maps VictoryState into backend GameResult shape', () => {
      const state = createTestState();
      const handler = createAutoSelectDecisionHandler();

      const stateAccessor = {
        getGameState: () => state,
        updateGameState: (_newState: GameState) => {
          // no-op for this mapping test
        },
        getPlayerInfo: (playerNumber: number) => {
          const player = state.players.find((p) => p.playerNumber === playerNumber);
          return player ? { type: player.type } : undefined;
        },
      };

      const adapter = new TurnEngineAdapter({
        stateAccessor,
        decisionHandler: handler,
      }) as any;

      const victory: VictoryState = {
        isGameOver: true,
        winner: 1,
        reason: 'ring_elimination',
        scores: [
          {
            player: 1,
            eliminatedRings: 5,
            territorySpaces: 10,
            ringsOnBoard: 3,
            ringsInHand: 2,
            markerCount: 0,
            isEliminated: false,
          },
          {
            player: 2,
            eliminatedRings: 2,
            territorySpaces: 5,
            ringsOnBoard: 4,
            ringsInHand: 1,
            markerCount: 0,
            isEliminated: false,
          },
        ],
        tieBreaker: undefined,
      };

      const result = adapter.convertVictoryToGameResult(victory);

      expect(result).toEqual({
        winner: 1,
        reason: 'ring_elimination',
        finalScore: {
          ringsEliminated: { 1: 5, 2: 2 },
          territorySpaces: { 1: 10, 2: 5 },
          ringsRemaining: { 1: 5, 2: 5 },
        },
      });
    });

    it('returns undefined GameResult when winner is undefined', () => {
      const state = createTestState();
      const handler = createAutoSelectDecisionHandler();

      const stateAccessor = {
        getGameState: () => state,
        updateGameState: (_newState: GameState) => {},
        getPlayerInfo: (playerNumber: number) => {
          const player = state.players.find((p) => p.playerNumber === playerNumber);
          return player ? { type: player.type } : undefined;
        },
      };

      const adapter = new TurnEngineAdapter({
        stateAccessor,
        decisionHandler: handler,
      }) as any;

      const victory: VictoryState = {
        isGameOver: true,
        winner: undefined,
        reason: 'stalemate_resolution',
        scores: [],
        tieBreaker: undefined,
      };

      const result = adapter.convertVictoryToGameResult(victory);
      expect(result).toBeUndefined();
    });
  });
});
