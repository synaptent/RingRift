/**
 * Territory Processing Phase - Loop and Edge Case Tests
 *
 * Tests comprehensive territory processing scenarios including:
 * - Multi-region processing loops
 * - Elimination sequences within regions
 * - Territory → Forced Elimination transitions
 * - Invalid event handling
 * - Counter and state invariants
 */

import {
  transition,
  onTerritoryProcessingComplete,
  type TurnEvent,
  type GameContext,
  type TerritoryProcessingState,
  type TurnEndState,
  type DisconnectedRegion,
  type EliminationTarget,
} from '../../../src/shared/engine/fsm';

describe('Territory Processing Phase - Loop Tests', () => {
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

  // Helper to create regions
  const makeRegion = (
    positions: Array<{ x: number; y: number }>,
    controllingPlayer: number,
    eliminationsRequired: number
  ): DisconnectedRegion => ({
    positions,
    controllingPlayer,
    eliminationsRequired,
  });

  // Helper to create elimination targets
  const makeElimTarget = (x: number, y: number, player: number, count: number): EliminationTarget => ({
    position: { x, y },
    player,
    count,
  });

  describe('empty region handling', () => {
    it('should allow NO_TERRITORY_ACTION when no regions', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const result = transition(state, { type: 'NO_TERRITORY_ACTION' }, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
        expect(result.actions).toContainEqual({ type: 'CHECK_VICTORY' });
      }
    });

    it('should reject NO_TERRITORY_ACTION when regions exist', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [makeRegion([{ x: 0, y: 0 }], 2, 1)],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const result = transition(state, { type: 'NO_TERRITORY_ACTION' }, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
      }
    });
  });

  describe('single region processing', () => {
    it('should process region with no eliminations and transition to turn_end', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [makeRegion([{ x: 2, y: 2 }, { x: 2, y: 3 }], 1, 0)],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const result = transition(state, { type: 'PROCESS_REGION', regionIndex: 0 }, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
        expect(result.actions).toContainEqual({
          type: 'PROCESS_DISCONNECTION',
          region: state.disconnectedRegions[0],
        });
        expect(result.actions).toContainEqual({ type: 'CHECK_VICTORY' });
      }
    });

    it('should queue eliminations when region requires them', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [makeRegion([{ x: 3, y: 3 }], 2, 2)],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const result = transition(state, { type: 'PROCESS_REGION', regionIndex: 0 }, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('territory_processing');
        const nextState = result.state as TerritoryProcessingState;
        expect(nextState.eliminationsPending.length).toBeGreaterThan(0);
        expect(nextState.eliminationsPending[0].count).toBe(2);
      }
    });
  });

  describe('multi-region loops', () => {
    it('should process multiple regions in sequence', () => {
      const regions = [
        makeRegion([{ x: 0, y: 0 }], 1, 0),
        makeRegion([{ x: 5, y: 5 }], 2, 0),
        makeRegion([{ x: 7, y: 7 }], 1, 0),
      ];

      // Process first region
      const state1: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: regions,
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      let result = transition(state1, { type: 'PROCESS_REGION', regionIndex: 0 }, context);
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('territory_processing');
        expect((result.state as TerritoryProcessingState).currentRegionIndex).toBe(1);
      }

      // Process second region
      const state2: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: regions,
        currentRegionIndex: 1,
        eliminationsPending: [],
      };

      result = transition(state2, { type: 'PROCESS_REGION', regionIndex: 1 }, context);
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('territory_processing');
        expect((result.state as TerritoryProcessingState).currentRegionIndex).toBe(2);
      }

      // Process third (final) region
      const state3: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: regions,
        currentRegionIndex: 2,
        eliminationsPending: [],
      };

      result = transition(state3, { type: 'PROCESS_REGION', regionIndex: 2 }, context);
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
      }
    });

    it('should handle regions belonging to different players', () => {
      const regions = [
        makeRegion([{ x: 0, y: 0 }], 1, 0),
        makeRegion([{ x: 2, y: 2 }], 2, 0),
      ];

      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: regions,
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const result = transition(state, { type: 'PROCESS_REGION', regionIndex: 0 }, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('territory_processing');
        // Region 0 belonged to player 1, region 1 to player 2
        expect((result.state as TerritoryProcessingState).currentRegionIndex).toBe(1);
      }
    });
  });

  describe('elimination sequences within territory', () => {
    it('should execute elimination when pending', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [makeRegion([{ x: 3, y: 3 }], 2, 1)],
        currentRegionIndex: 0,
        eliminationsPending: [makeElimTarget(3, 3, 2, 1)],
      };

      const result = transition(
        state,
        { type: 'ELIMINATE_FROM_STACK', target: { x: 3, y: 3 }, count: 1 },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.actions).toContainEqual({
          type: 'ELIMINATE_RINGS',
          target: { x: 3, y: 3 },
          count: 1,
        });
      }
    });

    it('should continue to next region after elimination completes', () => {
      const regions = [
        makeRegion([{ x: 1, y: 1 }], 1, 1),
        makeRegion([{ x: 5, y: 5 }], 2, 0),
      ];

      // State with one elimination pending and more regions to process
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: regions,
        currentRegionIndex: 0,
        eliminationsPending: [makeElimTarget(1, 1, 1, 1)],
      };

      const result = transition(
        state,
        { type: 'ELIMINATE_FROM_STACK', target: { x: 1, y: 1 }, count: 1 },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('territory_processing');
        const nextState = result.state as TerritoryProcessingState;
        expect(nextState.currentRegionIndex).toBe(1);
        expect(nextState.eliminationsPending).toHaveLength(0);
      }
    });

    it('should handle multiple eliminations in single region', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [makeRegion([{ x: 2, y: 2 }], 2, 3)],
        currentRegionIndex: 0,
        eliminationsPending: [
          makeElimTarget(2, 2, 2, 1),
          makeElimTarget(2, 3, 2, 1),
          makeElimTarget(2, 4, 2, 1),
        ],
      };

      // First elimination
      let result = transition(
        state,
        { type: 'ELIMINATE_FROM_STACK', target: { x: 2, y: 2 }, count: 1 },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('territory_processing');
        expect((result.state as TerritoryProcessingState).eliminationsPending).toHaveLength(2);
      }

      // Second elimination
      const state2: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [makeRegion([{ x: 2, y: 2 }], 2, 3)],
        currentRegionIndex: 0,
        eliminationsPending: [makeElimTarget(2, 3, 2, 1), makeElimTarget(2, 4, 2, 1)],
      };

      result = transition(
        state2,
        { type: 'ELIMINATE_FROM_STACK', target: { x: 2, y: 3 }, count: 1 },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('territory_processing');
        expect((result.state as TerritoryProcessingState).eliminationsPending).toHaveLength(1);
      }

      // Third (final) elimination
      const state3: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [makeRegion([{ x: 2, y: 2 }], 2, 3)],
        currentRegionIndex: 0,
        eliminationsPending: [makeElimTarget(2, 4, 2, 1)],
      };

      result = transition(
        state3,
        { type: 'ELIMINATE_FROM_STACK', target: { x: 2, y: 4 }, count: 1 },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
      }
    });
  });

  describe('territory → FE transition', () => {
    it('onTerritoryProcessingComplete returns forced_elimination when no action and has stacks', () => {
      const result = onTerritoryProcessingComplete(false, true);
      expect(result).toBe('forced_elimination');
    });

    it('onTerritoryProcessingComplete returns turn_end when player had action', () => {
      const result = onTerritoryProcessingComplete(true, true);
      expect(result).toBe('turn_end');
    });

    it('onTerritoryProcessingComplete returns turn_end when player has no stacks', () => {
      const result = onTerritoryProcessingComplete(false, false);
      expect(result).toBe('turn_end');
    });
  });

  describe('invalid events in territory_processing', () => {
    const territoryState: TerritoryProcessingState = {
      phase: 'territory_processing',
      player: 1,
      disconnectedRegions: [makeRegion([{ x: 0, y: 0 }], 1, 0)],
      currentRegionIndex: 0,
      eliminationsPending: [],
    };

    const invalidEvents: Array<{ event: TurnEvent; name: string }> = [
      { event: { type: 'PLACE_RING', to: { x: 0, y: 0 } }, name: 'PLACE_RING' },
      { event: { type: 'MOVE_STACK', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } }, name: 'MOVE_STACK' },
      { event: { type: 'CAPTURE', target: { x: 2, y: 2 } }, name: 'CAPTURE' },
      { event: { type: 'SKIP_PLACEMENT' }, name: 'SKIP_PLACEMENT' },
      { event: { type: 'NO_MOVEMENT_ACTION' }, name: 'NO_MOVEMENT_ACTION' },
      { event: { type: 'NO_LINE_ACTION' }, name: 'NO_LINE_ACTION' },
      { event: { type: 'FORCED_ELIMINATE', target: { x: 0, y: 0 } }, name: 'FORCED_ELIMINATE' },
      { event: { type: 'PROCESS_LINE', lineIndex: 0 }, name: 'PROCESS_LINE' },
      { event: { type: '_ADVANCE_TURN' }, name: '_ADVANCE_TURN' },
    ];

    invalidEvents.forEach(({ event, name }) => {
      it(`should reject ${name} in territory_processing phase`, () => {
        const result = transition(territoryState, event, context);
        expect(result.ok).toBe(false);
        if (!result.ok) {
          expect(result.error.code).toBe('INVALID_EVENT');
          expect(result.error.currentPhase).toBe('territory_processing');
        }
      });
    });
  });

  describe('region index bounds checking', () => {
    it('should reject PROCESS_REGION with negative index (conceptually)', () => {
      // Note: TypeScript prevents negative indices at compile time,
      // but runtime JS could pass invalid values
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [makeRegion([{ x: 0, y: 0 }], 1, 0)],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      // Index out of bounds (too high)
      const result = transition(state, { type: 'PROCESS_REGION', regionIndex: 10 }, context);
      expect(result.ok).toBe(false);
    });

    it('should reject PROCESS_REGION with index beyond array length', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [
          makeRegion([{ x: 0, y: 0 }], 1, 0),
          makeRegion([{ x: 1, y: 1 }], 2, 0),
        ],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const result = transition(state, { type: 'PROCESS_REGION', regionIndex: 5 }, context);
      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
      }
    });
  });

  describe('global events in territory_processing', () => {
    it('should allow RESIGN and transition to game_over', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [makeRegion([{ x: 0, y: 0 }], 1, 1)],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const result = transition(state, { type: 'RESIGN', player: 1 }, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
      }
    });

    it('should allow TIMEOUT and transition to game_over', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 2,
        disconnectedRegions: [],
        currentRegionIndex: 0,
        eliminationsPending: [makeElimTarget(0, 0, 2, 1)],
      };

      const result = transition(state, { type: 'TIMEOUT', player: 2 }, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
      }
    });
  });

  describe('player rotation after territory processing', () => {
    it('should advance to correct next player in 2p game', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const result = transition(state, { type: 'NO_TERRITORY_ACTION' }, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        const turnEnd = result.state as TurnEndState;
        expect(turnEnd.completedPlayer).toBe(1);
        expect(turnEnd.nextPlayer).toBe(2);
      }
    });

    it('should advance to correct next player in 4p game (wraparound)', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 4,
        disconnectedRegions: [],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const result = transition(state, { type: 'NO_TERRITORY_ACTION' }, context4p);

      expect(result.ok).toBe(true);
      if (result.ok) {
        const turnEnd = result.state as TurnEndState;
        expect(turnEnd.completedPlayer).toBe(4);
        expect(turnEnd.nextPlayer).toBe(1);
      }
    });
  });

  describe('action emission', () => {
    it('should emit PROCESS_DISCONNECTION action when processing region', () => {
      const region = makeRegion([{ x: 1, y: 1 }, { x: 1, y: 2 }], 1, 0);
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [region],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const result = transition(state, { type: 'PROCESS_REGION', regionIndex: 0 }, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.actions).toContainEqual({
          type: 'PROCESS_DISCONNECTION',
          region,
        });
      }
    });

    it('should emit ELIMINATE_RINGS action on ELIMINATE_FROM_STACK', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [],
        currentRegionIndex: 0,
        eliminationsPending: [makeElimTarget(4, 4, 2, 3)],
      };

      const result = transition(
        state,
        { type: 'ELIMINATE_FROM_STACK', target: { x: 4, y: 4 }, count: 3 },
        context
      );

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.actions).toContainEqual({
          type: 'ELIMINATE_RINGS',
          target: { x: 4, y: 4 },
          count: 3,
        });
      }
    });

    it('should emit CHECK_VICTORY on final transition to turn_end', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const result = transition(state, { type: 'NO_TERRITORY_ACTION' }, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.actions).toContainEqual({ type: 'CHECK_VICTORY' });
      }
    });
  });

  describe('elimination guard checks', () => {
    it('should reject ELIMINATE_FROM_STACK when no eliminations pending', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [makeRegion([{ x: 0, y: 0 }], 1, 0)],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const result = transition(
        state,
        { type: 'ELIMINATE_FROM_STACK', target: { x: 0, y: 0 }, count: 1 },
        context
      );

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
        expect(result.error.message).toContain('No eliminations pending');
      }
    });
  });
});
