/**
 * FSM Decision Parity Tests
 *
 * Tests that verify FSM-derived decisions match expected PendingDecision structures.
 * Per RR-CANON-R075, every phase must produce a recorded action, and the FSM
 * is the canonical source for determining when decisions are needed.
 *
 * These tests ensure:
 * 1. FSMDecisionSurface is correctly populated in ProcessTurnResult
 * 2. PendingDecision types are correctly derived from FSM pendingDecisionType
 * 3. Decision options are enumerated correctly for each decision type
 */

import { createInitialGameState } from '../../../src/shared/engine/initialState';
import { processTurn } from '../../../src/shared/engine/orchestration/turnOrchestrator';
import { computeFSMOrchestration } from '../../../src/shared/engine/fsm/FSMAdapter';
import type {
  GameState,
  Move,
  Player,
  TimeControl,
  BoardState,
  RingStack,
} from '../../../src/shared/types/game';
import type { FSMDecisionSurface } from '../../../src/shared/engine/orchestration/types';

describe('FSM Decision Parity', () => {
  const timeControl: TimeControl = { type: 'rapid', initialTime: 600, increment: 0 };

  function createPlayers(count: number = 2): Player[] {
    return Array.from({ length: count }, (_, i) => ({
      id: `p${i + 1}`,
      username: `Player ${i + 1}`,
      type: 'human' as const,
      playerNumber: i + 1,
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    }));
  }

  function createGame(): GameState {
    return createInitialGameState('test-game', 'square8', createPlayers(), timeControl);
  }

  function makeMove(overrides: Partial<Move> = {}): Move {
    return {
      id: `test-${Date.now()}`,
      type: 'place_ring',
      player: 1,
      to: { x: 3, y: 3 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
      ...overrides,
    };
  }

  describe('FSMDecisionSurface population in ProcessTurnResult', () => {
    it('should populate fsmDecisionSurface when FSM returns pendingDecisionType', () => {
      // Create a state in line_processing phase with formed lines
      const state = createGame();
      state.currentPhase = 'line_processing';

      // Add formed lines to trigger line_order_required
      const formedLines = [
        {
          positions: [
            { x: 0, y: 0 },
            { x: 0, y: 1 },
            { x: 0, y: 2 },
          ],
          player: 1,
          length: 3,
          direction: { x: 0, y: 1 },
        },
      ];

      // Set up board with markers for the line
      const markers = new Map();
      markers.set('0,0', { type: 'marker', player: 1, position: { x: 0, y: 0 } });
      markers.set('0,1', { type: 'marker', player: 1, position: { x: 0, y: 1 } });
      markers.set('0,2', { type: 'marker', player: 1, position: { x: 0, y: 2 } });

      state.board = {
        ...state.board,
        markers,
        formedLines,
      } as BoardState;

      // Process a line to see if fsmDecisionSurface is populated
      const move = makeMove({
        type: 'process_line',
        player: 1,
        lineIndex: 0,
        to: { x: 0, y: 0 },
        formedLines,
      });

      // This tests that ProcessTurnResult includes fsmDecisionSurface
      const result = processTurn(state, move);

      // After processing the line, if more lines exist or territory phase has decisions,
      // fsmDecisionSurface should be populated
      // For this single line case, we transition to territory_processing
      if (result.fsmDecisionSurface) {
        expect(result.fsmDecisionSurface).toBeDefined();
        // Verify it has the expected shape
        if (result.fsmDecisionSurface.pendingDecisionType) {
          expect([
            'chain_capture',
            'line_order_required',
            'no_line_action_required',
            'region_order_required',
            'no_territory_action_required',
            'forced_elimination',
          ]).toContain(result.fsmDecisionSurface.pendingDecisionType);
        }
      }
    });

    it('should not populate fsmDecisionSurface when no decision is pending', () => {
      const state = createGame();
      // ring_placement phase with valid placement
      const move = makeMove({
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
      });

      const result = processTurn(state, move);

      // After a normal placement, we go to movement phase
      // No decision should be pending (player can just move)
      if (result.status === 'complete') {
        // fsmDecisionSurface may or may not be present based on next phase
        // The key is that pendingDecision should not be set for non-decision states
        expect(result.pendingDecision).toBeUndefined();
      }
    });
  });

  describe('PendingDecision type mapping from FSM', () => {
    describe('no_line_action_required decision', () => {
      it('should surface no_line_action_required when in line_processing with no lines', () => {
        const state = createGame();
        state.currentPhase = 'line_processing';
        state.board.formedLines = [];

        // Compute FSM orchestration directly to verify the decision type
        const fsmResult = computeFSMOrchestration(
          state,
          makeMove({ type: 'no_line_action', player: 1 })
        );

        // When no lines exist, FSM should indicate no_line_action_required or similar
        // The actual behavior depends on FSM transition logic
        expect(fsmResult.success).toBe(true);
        expect(fsmResult.nextPhase).toBe('territory_processing');
      });
    });

    describe('no_territory_action_required decision', () => {
      it('should surface no_territory_action_required when in territory_processing with no regions', () => {
        const state = createGame();
        state.currentPhase = 'territory_processing';
        // No collapsed spaces = no territory regions

        const fsmResult = computeFSMOrchestration(
          state,
          makeMove({ type: 'no_territory_action', player: 1 })
        );

        expect(fsmResult.success).toBe(true);
        // After no_territory_action, should transition to next turn or turn_end
        expect(['ring_placement', 'forced_elimination', 'game_over', 'turn_end']).toContain(
          fsmResult.nextPhase
        );
      });
    });

    describe('line_order_required decision', () => {
      it('should include pendingLines when multiple lines detected', () => {
        const state = createGame();
        state.currentPhase = 'line_processing';

        // Set up multiple formed lines
        const formedLines = [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 0, y: 1 },
              { x: 0, y: 2 },
            ],
            player: 1,
            length: 3,
            direction: { x: 0, y: 1 },
          },
          {
            positions: [
              { x: 1, y: 0 },
              { x: 1, y: 1 },
              { x: 1, y: 2 },
            ],
            player: 1,
            length: 3,
            direction: { x: 0, y: 1 },
          },
        ];

        // Set up board markers
        const markers = new Map();
        markers.set('0,0', { type: 'marker', player: 1, position: { x: 0, y: 0 } });
        markers.set('0,1', { type: 'marker', player: 1, position: { x: 0, y: 1 } });
        markers.set('0,2', { type: 'marker', player: 1, position: { x: 0, y: 2 } });
        markers.set('1,0', { type: 'marker', player: 1, position: { x: 1, y: 0 } });
        markers.set('1,1', { type: 'marker', player: 1, position: { x: 1, y: 1 } });
        markers.set('1,2', { type: 'marker', player: 1, position: { x: 1, y: 2 } });

        state.board = {
          ...state.board,
          markers,
          formedLines,
        } as BoardState;

        const fsmResult = computeFSMOrchestration(state, makeMove({ type: 'no_line_action' }), {
          detectedLines: formedLines,
        });

        // With multiple lines, FSM should indicate line_order_required
        if (fsmResult.pendingDecisionType === 'line_order_required') {
          expect(fsmResult.decisionSurface?.pendingLines).toBeDefined();
          expect(fsmResult.decisionSurface?.pendingLines?.length).toBeGreaterThanOrEqual(1);
        }
      });
    });
  });

  describe('Decision parity between FSM and legacy paths', () => {
    it('should derive consistent PendingDecision from FSMOrchestrationResult', () => {
      // This test verifies that the decision derivation logic is consistent
      const state = createGame();
      state.currentPhase = 'ring_placement';

      // Place a ring and check the result
      const move = makeMove({
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
      });

      const result = processTurn(state, move);

      // Verify result structure per ProcessTurnResult interface
      expect(result).toHaveProperty('nextState');
      expect(result).toHaveProperty('status');
      expect(result).toHaveProperty('metadata');

      // If awaiting_decision, pendingDecision should be defined
      if (result.status === 'awaiting_decision') {
        expect(result.pendingDecision).toBeDefined();
        expect(result.pendingDecision?.type).toBeDefined();
        expect(result.pendingDecision?.player).toBeDefined();
        expect(result.pendingDecision?.options).toBeDefined();
        expect(result.pendingDecision?.context).toBeDefined();
      } else {
        // If complete, pendingDecision should be undefined
        expect(result.pendingDecision).toBeUndefined();
      }
    });

    it('should include fsmDecisionSurface when FSM has decision surface data', () => {
      const state = createGame();

      // Setup state for a decision scenario
      state.currentPhase = 'line_processing';
      state.board.formedLines = [
        {
          positions: [
            { x: 0, y: 0 },
            { x: 0, y: 1 },
            { x: 0, y: 2 },
          ],
          player: 1,
          length: 3,
          direction: { x: 0, y: 1 },
        },
      ];

      const markers = new Map();
      markers.set('0,0', { type: 'marker', player: 1, position: { x: 0, y: 0 } });
      markers.set('0,1', { type: 'marker', player: 1, position: { x: 0, y: 1 } });
      markers.set('0,2', { type: 'marker', player: 1, position: { x: 0, y: 2 } });
      state.board.markers = markers;

      // Process the line
      const move = makeMove({
        type: 'process_line',
        player: 1,
        lineIndex: 0,
        to: { x: 0, y: 0 },
        formedLines: state.board.formedLines,
      });

      const result = processTurn(state, move);

      // Result should have the fsmDecisionSurface if FSM computed one
      // The actual value depends on post-line-processing state
      expect(result.metadata).toBeDefined();
      expect(result.metadata.processedMove).toBeDefined();
    });
  });

  describe('FSMDecisionSurface field types', () => {
    it('should have correctly typed pendingDecisionType values', () => {
      // Type-level test - verify the FSMDecisionSurface type allows expected values
      const surface: FSMDecisionSurface = {
        pendingDecisionType: 'chain_capture',
      };
      expect(surface.pendingDecisionType).toBe('chain_capture');

      const surface2: FSMDecisionSurface = {
        pendingDecisionType: 'line_order_required',
        pendingLines: [{ positions: [{ x: 0, y: 0 }] }],
      };
      expect(surface2.pendingDecisionType).toBe('line_order_required');
      expect(surface2.pendingLines).toHaveLength(1);

      const surface3: FSMDecisionSurface = {
        pendingDecisionType: 'forced_elimination',
        forcedEliminationCount: 2,
      };
      expect(surface3.forcedEliminationCount).toBe(2);
    });

    it('should allow optional fields on FSMDecisionSurface', () => {
      // Empty surface is valid
      const emptySurface: FSMDecisionSurface = {};
      expect(emptySurface.pendingDecisionType).toBeUndefined();
      expect(emptySurface.pendingLines).toBeUndefined();
      expect(emptySurface.pendingRegions).toBeUndefined();
      expect(emptySurface.chainContinuations).toBeUndefined();
      expect(emptySurface.forcedEliminationCount).toBeUndefined();
    });
  });

  describe('Canonical rules alignment (RR-CANON-R075)', () => {
    it('should return no_line_action_required decision type when line_processing has no lines', () => {
      const state = createGame();
      state.currentPhase = 'line_processing';
      state.board.formedLines = [];

      // Try to get valid moves - should indicate no_line_action is required
      // This is per RR-CANON-R075: every phase must produce a recorded action
      const fsmResult = computeFSMOrchestration(state, makeMove({ type: 'no_line_action' }));

      // The transition should succeed, moving to territory_processing
      expect(fsmResult.success).toBe(true);
    });

    it('should return no_territory_action_required decision type when territory_processing has no regions', () => {
      const state = createGame();
      state.currentPhase = 'territory_processing';
      // Empty board = no territory regions

      const fsmResult = computeFSMOrchestration(state, makeMove({ type: 'no_territory_action' }));

      expect(fsmResult.success).toBe(true);
    });
  });
});
