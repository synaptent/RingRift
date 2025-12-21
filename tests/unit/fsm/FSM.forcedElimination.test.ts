/**
 * Forced Elimination (FE) Phase - Targeted Edge Case Tests
 *
 * Tests comprehensive FE scenarios including:
 * - Multi-elimination sequences
 * - Entry conditions from territory/line processing
 * - Boundary cases and counter tracking
 * - Invalid event handling
 * - Interaction with global events (RESIGN/TIMEOUT)
 */

import {
  transition,
  onTerritoryProcessingComplete,
  onLineProcessingComplete,
  type TurnEvent,
  type GameContext,
  type ForcedEliminationState,
  type TerritoryProcessingState,
  type LineProcessingState,
  type TurnEndState,
  type GameOverState,
} from '../../../src/shared/engine/fsm';

describe('Forced Elimination Phase - Edge Cases', () => {
  const context: GameContext = {
    boardType: 'square8',
    numPlayers: 2,
    ringsPerPlayer: 18,
    lineLength: 3,
  };

  const context4p: GameContext = {
    boardType: 'square8',
    numPlayers: 4,
    ringsPerPlayer: 12,
    lineLength: 3,
  };

  describe('FE entry conditions', () => {
    it('onTerritoryProcessingComplete returns forced_elimination when no action and has stacks', () => {
      // Player made no meaningful action (no territory to process) but has stacks
      // This triggers FE check to see if player exceeds ring limit
      const result = onTerritoryProcessingComplete(false, true);
      expect(result).toBe('forced_elimination');
    });

    it('onTerritoryProcessingComplete returns turn_end when player had action', () => {
      // Player performed territory processing action - skip FE
      const result = onTerritoryProcessingComplete(true, true);
      expect(result).toBe('turn_end');
    });

    it('onTerritoryProcessingComplete returns turn_end when no stacks', () => {
      // Player has no stacks on board - can't be over limit
      const result = onTerritoryProcessingComplete(false, false);
      expect(result).toBe('turn_end');
    });

    it('onLineProcessingComplete returns forced_elimination when no action and has stacks', () => {
      // No lines to process, no regions, but player has stacks
      const result = onLineProcessingComplete(false, false, true);
      expect(result).toBe('forced_elimination');
    });

    it('onLineProcessingComplete returns territory_processing when regions exist', () => {
      // Must process territory first before checking FE
      const result = onLineProcessingComplete(true, false, true);
      expect(result).toBe('territory_processing');
    });

    it('onLineProcessingComplete returns turn_end when player had action', () => {
      // Player performed line processing - skip FE
      const result = onLineProcessingComplete(false, true, true);
      expect(result).toBe('turn_end');
    });
  });

  describe('FE multi-elimination sequences', () => {
    it('should require exact number of eliminations (ringsOverLimit=3)', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 3,
        eliminationsDone: 0,
      };

      // First elimination
      let result = transition(state, { type: 'FORCED_ELIMINATE', target: { x: 0, y: 0 } }, context);
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('forced_elimination');
        expect((result.state as ForcedEliminationState).eliminationsDone).toBe(1);
      }

      // Second elimination
      const state2: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 3,
        eliminationsDone: 1,
      };
      result = transition(state2, { type: 'FORCED_ELIMINATE', target: { x: 1, y: 1 } }, context);
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('forced_elimination');
        expect((result.state as ForcedEliminationState).eliminationsDone).toBe(2);
      }

      // Third (final) elimination
      const state3: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 3,
        eliminationsDone: 2,
      };
      result = transition(state3, { type: 'FORCED_ELIMINATE', target: { x: 2, y: 2 } }, context);
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
        // Actions should include CHECK_VICTORY
        expect(result.actions).toContainEqual({ type: 'CHECK_VICTORY' });
      }
    });

    it('should handle single elimination (ringsOverLimit=1)', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 2,
        ringsOverLimit: 1,
        eliminationsDone: 0,
      };

      const result = transition(
        state,
        { type: 'FORCED_ELIMINATE', target: { x: 3, y: 3 } },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
        const turnEnd = result.state as TurnEndState;
        expect(turnEnd.completedPlayer).toBe(2);
        expect(turnEnd.nextPlayer).toBe(1); // Wraps around in 2p
      }
    });

    it('should handle high elimination count (ringsOverLimit=10)', () => {
      // Simulate reaching 9 eliminations done, need 1 more
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 10,
        eliminationsDone: 9,
      };

      const result = transition(
        state,
        { type: 'FORCED_ELIMINATE', target: { x: 5, y: 5 } },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
      }
    });
  });

  describe('FE action emission', () => {
    it('should emit FORCED_ELIMINATE action with correct player', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 2,
        ringsOverLimit: 2,
        eliminationsDone: 0,
      };

      const result = transition(
        state,
        { type: 'FORCED_ELIMINATE', target: { x: 4, y: 4 } },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.actions).toContainEqual({
          type: 'FORCED_ELIMINATE',
          target: { x: 4, y: 4 },
          player: 2,
        });
      }
    });

    it('should emit CHECK_VICTORY only on final elimination', () => {
      // Mid-elimination (not final)
      const stateMid: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 3,
        eliminationsDone: 1,
      };

      let result = transition(
        stateMid,
        { type: 'FORCED_ELIMINATE', target: { x: 0, y: 0 } },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.actions).not.toContainEqual({ type: 'CHECK_VICTORY' });
      }

      // Final elimination
      const stateFinal: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 3,
        eliminationsDone: 2,
      };

      result = transition(
        stateFinal,
        { type: 'FORCED_ELIMINATE', target: { x: 1, y: 1 } },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.actions).toContainEqual({ type: 'CHECK_VICTORY' });
      }
    });
  });

  describe('FE invalid events', () => {
    const feState: ForcedEliminationState = {
      phase: 'forced_elimination',
      player: 1,
      ringsOverLimit: 2,
      eliminationsDone: 0,
    };

    // Canonical spec: only forced_elimination move is valid in forced_elimination phase.
    // All other moves should be rejected.
    const invalidEvents: Array<{ event: TurnEvent; name: string }> = [
      { event: { type: 'PLACE_RING', to: { x: 0, y: 0 } }, name: 'PLACE_RING' },
      {
        event: { type: 'MOVE_STACK', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } },
        name: 'MOVE_STACK',
      },
      { event: { type: 'CAPTURE', target: { x: 2, y: 2 } }, name: 'CAPTURE' },
      { event: { type: 'SKIP_PLACEMENT' }, name: 'SKIP_PLACEMENT' },
      { event: { type: 'NO_MOVEMENT_ACTION' }, name: 'NO_MOVEMENT_ACTION' },
      { event: { type: 'NO_LINE_ACTION' }, name: 'NO_LINE_ACTION' },
      { event: { type: 'NO_TERRITORY_ACTION' }, name: 'NO_TERRITORY_ACTION' },
      { event: { type: 'PROCESS_LINE', lineIndex: 0 }, name: 'PROCESS_LINE' },
      { event: { type: 'PROCESS_REGION', regionIndex: 0 }, name: 'PROCESS_REGION' },
      { event: { type: '_ADVANCE_TURN' }, name: '_ADVANCE_TURN' },
    ];

    invalidEvents.forEach(({ event, name }) => {
      it(`should reject ${name} in forced_elimination phase`, () => {
        const result = transition(feState, event, context);
        expect(result.ok).toBe(false);
        if (!result.ok) {
          expect(result.error.code).toBe('INVALID_EVENT');
          expect(result.error.currentPhase).toBe('forced_elimination');
        }
      });
    });
  });

  describe('FE global events', () => {
    it('should allow RESIGN and transition to game_over', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 5,
        eliminationsDone: 2,
      };

      const result = transition(state, { type: 'RESIGN', player: 1 }, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        const gameOver = result.state as GameOverState;
        expect(gameOver.reason).toBe('resignation');
        // Winner is null in FSM - computed by orchestrator from remaining players
        expect(gameOver.winner).toBeNull();
      }
    });

    it('should allow TIMEOUT and transition to game_over', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 2,
        ringsOverLimit: 3,
        eliminationsDone: 1,
      };

      const result = transition(state, { type: 'TIMEOUT', player: 2 }, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        const gameOver = result.state as GameOverState;
        expect(gameOver.reason).toBe('timeout');
        // Winner is null in FSM - computed by orchestrator from remaining players
        expect(gameOver.winner).toBeNull();
      }
    });

    it('should handle RESIGN in 4-player FE', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 3,
        ringsOverLimit: 2,
        eliminationsDone: 0,
      };

      const result = transition(state, { type: 'RESIGN', player: 3 }, context4p);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        const gameOver = result.state as GameOverState;
        expect(gameOver.reason).toBe('resignation');
      }
    });
  });

  describe('FE player rotation', () => {
    it('should advance to correct next player in 2p game', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 1,
        eliminationsDone: 0,
      };

      const result = transition(
        state,
        { type: 'FORCED_ELIMINATE', target: { x: 0, y: 0 } },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
        const turnEnd = result.state as TurnEndState;
        expect(turnEnd.completedPlayer).toBe(1);
        expect(turnEnd.nextPlayer).toBe(2);
      }
    });

    it('should advance to correct next player in 4p game (wraparound)', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 4,
        ringsOverLimit: 1,
        eliminationsDone: 0,
      };

      const result = transition(
        state,
        { type: 'FORCED_ELIMINATE', target: { x: 0, y: 0 } },
        context4p
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
        const turnEnd = result.state as TurnEndState;
        expect(turnEnd.completedPlayer).toBe(4);
        expect(turnEnd.nextPlayer).toBe(1); // Wraps around
      }
    });

    it('should advance to correct next player in 4p game (mid-sequence)', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 2,
        ringsOverLimit: 1,
        eliminationsDone: 0,
      };

      const result = transition(
        state,
        { type: 'FORCED_ELIMINATE', target: { x: 0, y: 0 } },
        context4p
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        const turnEnd = result.state as TurnEndState;
        expect(turnEnd.nextPlayer).toBe(3);
      }
    });
  });

  describe('FE counter invariants', () => {
    it('should maintain eliminationsDone counter correctly', () => {
      const scenarios = [
        { overLimit: 5, done: 0, expectedDone: 1, expectedPhase: 'forced_elimination' },
        { overLimit: 5, done: 1, expectedDone: 2, expectedPhase: 'forced_elimination' },
        { overLimit: 5, done: 2, expectedDone: 3, expectedPhase: 'forced_elimination' },
        { overLimit: 5, done: 3, expectedDone: 4, expectedPhase: 'forced_elimination' },
        { overLimit: 5, done: 4, expectedDone: 5, expectedPhase: 'turn_end' },
      ];

      scenarios.forEach(({ overLimit, done, expectedDone, expectedPhase }) => {
        const state: ForcedEliminationState = {
          phase: 'forced_elimination',
          player: 1,
          ringsOverLimit: overLimit,
          eliminationsDone: done,
        };

        const result = transition(
          state,
          { type: 'FORCED_ELIMINATE', target: { x: done, y: done } },
          context
        );

        expect(result.ok).toBe(true);
        if (result.ok) {
          expect(result.state.phase).toBe(expectedPhase);
          if (result.state.phase === 'forced_elimination') {
            expect((result.state as ForcedEliminationState).eliminationsDone).toBe(expectedDone);
          }
        }
      });
    });

    it('should preserve player and ringsOverLimit during elimination sequence', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 2,
        ringsOverLimit: 4,
        eliminationsDone: 1,
      };

      const result = transition(
        state,
        { type: 'FORCED_ELIMINATE', target: { x: 0, y: 0 } },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok && result.state.phase === 'forced_elimination') {
        const nextState = result.state as ForcedEliminationState;
        expect(nextState.player).toBe(2);
        expect(nextState.ringsOverLimit).toBe(4);
        expect(nextState.eliminationsDone).toBe(2);
      }
    });
  });

  describe('FE boundary conditions', () => {
    it('should handle ringsOverLimit=0 edge case (should not normally occur)', () => {
      // This shouldn't happen in practice, but test defensive behavior
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 0,
        eliminationsDone: 0,
      };

      // Any elimination with ringsOverLimit=0 should complete immediately
      // (newCount=1 >= ringsOverLimit=0)
      const result = transition(
        state,
        { type: 'FORCED_ELIMINATE', target: { x: 0, y: 0 } },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
      }
    });

    it('should handle eliminationsDone starting at non-zero (resumption)', () => {
      // Test case where we're resuming mid-FE sequence
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 5,
        eliminationsDone: 3,
      };

      let result = transition(state, { type: 'FORCED_ELIMINATE', target: { x: 0, y: 0 } }, context);
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('forced_elimination');
        expect((result.state as ForcedEliminationState).eliminationsDone).toBe(4);
      }

      // Final elimination
      const state2: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 5,
        eliminationsDone: 4,
      };
      result = transition(state2, { type: 'FORCED_ELIMINATE', target: { x: 1, y: 1 } }, context);
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
      }
    });
  });
});
