/**
 * FSMAdapter integration tests
 *
 * Tests the bidirectional conversion between Move types and FSM events,
 * as well as deriving FSM state from game state.
 */

import {
  moveToEvent,
  eventToMove,
  deriveStateFromGame,
  deriveGameContext,
  validateEvent,
  getValidEvents,
  validateMoveWithFSM,
  isMoveTypeValidForPhase,
  getAllowedMoveTypesForPhase,
  computeFSMOrchestration,
} from '../../../src/shared/engine/fsm/FSMAdapter';
import type { Move, GameState, GamePhase, MoveType } from '../../../src/shared/types/game';
import { createInitialGameState } from '../../../src/shared/engine/initialState';

describe('FSMAdapter', () => {
  describe('moveToEvent', () => {
    it('should convert place_ring to PLACE_RING event', () => {
      const move: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'PLACE_RING', to: { x: 3, y: 3 } });
    });

    it('should convert skip_placement to SKIP_PLACEMENT event', () => {
      const move: Move = {
        id: 'test-2',
        type: 'skip_placement',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'SKIP_PLACEMENT' });
    });

    it('should convert no_placement_action to NO_PLACEMENT_ACTION event', () => {
      const move: Move = {
        id: 'test-no-placement',
        type: 'no_placement_action',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'NO_PLACEMENT_ACTION' });
    });

    it('should convert no_movement_action to NO_MOVEMENT_ACTION event', () => {
      const move: Move = {
        id: 'test-no-movement',
        type: 'no_movement_action',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'NO_MOVEMENT_ACTION' });
    });

    it('should convert no_line_action to NO_LINE_ACTION event', () => {
      const move: Move = {
        id: 'test-no-line',
        type: 'no_line_action',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 3,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'NO_LINE_ACTION' });
    });

    it('should convert no_territory_action to NO_TERRITORY_ACTION event', () => {
      const move: Move = {
        id: 'test-no-territory',
        type: 'no_territory_action',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 4,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'NO_TERRITORY_ACTION' });
    });

    it('should convert move_stack to MOVE_STACK event', () => {
      const move: Move = {
        id: 'test-3',
        type: 'move_stack',
        player: 1,
        from: { x: 2, y: 2 },
        to: { x: 4, y: 4 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({
        type: 'MOVE_STACK',
        from: { x: 2, y: 2 },
        to: { x: 4, y: 4 },
      });
    });

    it('should return null for move_stack without from position', () => {
      const move: Move = {
        id: 'test-4',
        type: 'move_stack',
        player: 1,
        to: { x: 4, y: 4 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      };

      const event = moveToEvent(move);

      expect(event).toBeNull();
    });

    it('should convert overtaking_capture to CAPTURE event', () => {
      const move: Move = {
        id: 'test-5',
        type: 'overtaking_capture',
        player: 1,
        from: { x: 2, y: 2 },
        captureTarget: { x: 3, y: 3 },
        to: { x: 4, y: 4 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 3,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'CAPTURE', target: { x: 3, y: 3 } });
    });

    it('should convert skip_capture to END_CHAIN event', () => {
      const move: Move = {
        id: 'test-6',
        type: 'skip_capture',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 4,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'END_CHAIN' });
    });

    it('should convert forced_elimination to FORCED_ELIMINATE event', () => {
      const move: Move = {
        id: 'test-7',
        type: 'forced_elimination',
        player: 1,
        to: { x: 5, y: 5 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 5,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'FORCED_ELIMINATE', target: { x: 5, y: 5 } });
    });

    it('should return null for swap_sides (meta-move)', () => {
      const move: Move = {
        id: 'test-8',
        type: 'swap_sides',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const event = moveToEvent(move);

      expect(event).toBeNull();
    });
  });

  describe('eventToMove', () => {
    it('should convert PLACE_RING event to place_ring move', () => {
      const event = { type: 'PLACE_RING' as const, to: { x: 3, y: 3 } };

      const move = eventToMove(event, 1, 1);

      expect(move).not.toBeNull();
      expect(move!.type).toBe('place_ring');
      expect(move!.to).toEqual({ x: 3, y: 3 });
      expect(move!.player).toBe(1);
    });

    it('should convert MOVE_STACK event to move_stack move', () => {
      const event = { type: 'MOVE_STACK' as const, from: { x: 2, y: 2 }, to: { x: 4, y: 4 } };

      const move = eventToMove(event, 1, 2);

      expect(move).not.toBeNull();
      expect(move!.type).toBe('move_stack');
      expect(move!.from).toEqual({ x: 2, y: 2 });
      expect(move!.to).toEqual({ x: 4, y: 4 });
    });

    it('should convert NO_PLACEMENT_ACTION event to no_placement_action move', () => {
      const event = { type: 'NO_PLACEMENT_ACTION' as const };

      const move = eventToMove(event, 1, 10);

      expect(move).not.toBeNull();
      expect(move!.type).toBe('no_placement_action');
      expect(move!.player).toBe(1);
      expect(move!.to).toEqual({ x: 0, y: 0 });
    });

    it('should convert NO_MOVEMENT_ACTION event to no_movement_action move', () => {
      const event = { type: 'NO_MOVEMENT_ACTION' as const };

      const move = eventToMove(event, 2, 11);

      expect(move).not.toBeNull();
      expect(move!.type).toBe('no_movement_action');
      expect(move!.player).toBe(2);
      expect(move!.to).toEqual({ x: 0, y: 0 });
    });

    it('should convert NO_LINE_ACTION event to no_line_action move', () => {
      const event = { type: 'NO_LINE_ACTION' as const };

      const move = eventToMove(event, 1, 12);

      expect(move).not.toBeNull();
      expect(move!.type).toBe('no_line_action');
      expect(move!.player).toBe(1);
      expect(move!.to).toEqual({ x: 0, y: 0 });
    });

    it('should convert NO_TERRITORY_ACTION event to no_territory_action move', () => {
      const event = { type: 'NO_TERRITORY_ACTION' as const };

      const move = eventToMove(event, 1, 13);

      expect(move).not.toBeNull();
      expect(move!.type).toBe('no_territory_action');
      expect(move!.player).toBe(1);
      expect(move!.to).toEqual({ x: 0, y: 0 });
    });

    it('should return null for RESIGN event', () => {
      const event = { type: 'RESIGN' as const, player: 1 };

      const move = eventToMove(event, 1, 5);

      expect(move).toBeNull();
    });

    it('should return null for _ADVANCE_TURN event', () => {
      const event = { type: '_ADVANCE_TURN' as const };

      const move = eventToMove(event, 1, 5);

      expect(move).toBeNull();
    });
  });

  describe('deriveGameContext', () => {
    it('should derive correct context for square8 2-player game', () => {
      const players = [
        {
          id: 'p1',
          username: 'Player 1',
          type: 'human' as const,
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Player 2',
          type: 'ai' as const,
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      const state = createInitialGameState('test-game', 'square8', players, {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      });

      const context = deriveGameContext(state);

      expect(context.boardType).toBe('square8');
      expect(context.numPlayers).toBe(2);
      expect(context.ringsPerPlayer).toBe(18);
      // Per RR-CANON-R120: square8 2-player uses line length 4 (3+ player uses 3)
      expect(context.lineLength).toBe(4);
    });
  });

  describe('deriveStateFromGame', () => {
    it('should derive ring_placement state correctly', () => {
      const players = [
        {
          id: 'p1',
          username: 'Player 1',
          type: 'human' as const,
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Player 2',
          type: 'ai' as const,
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      const state = createInitialGameState('test-game', 'square8', players, {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      });

      const fsmState = deriveStateFromGame(state);

      expect(fsmState.phase).toBe('ring_placement');
      if (fsmState.phase === 'ring_placement') {
        expect(fsmState.player).toBe(1);
        expect(fsmState.canPlace).toBe(true);
        expect(fsmState.validPositions.length).toBeGreaterThan(0);
      }
    });
  });

  describe('roundtrip conversion', () => {
    it('should preserve move semantics through Move → Event → Move conversion', () => {
      const originalMove: Move = {
        id: 'original-1',
        type: 'place_ring',
        player: 1,
        to: { x: 5, y: 5 },
        timestamp: new Date(),
        thinkTime: 100,
        moveNumber: 3,
      };

      const event = moveToEvent(originalMove);
      expect(event).not.toBeNull();

      const convertedMove = eventToMove(event!, 1, 3);
      expect(convertedMove).not.toBeNull();

      // Key fields should be preserved
      expect(convertedMove!.type).toBe(originalMove.type);
      expect(convertedMove!.to).toEqual(originalMove.to);
      expect(convertedMove!.player).toBe(originalMove.player);
    });

    it('should preserve movement semantics', () => {
      const originalMove: Move = {
        id: 'original-2',
        type: 'move_stack',
        player: 2,
        from: { x: 3, y: 3 },
        to: { x: 6, y: 6 },
        timestamp: new Date(),
        thinkTime: 200,
        moveNumber: 7,
      };

      const event = moveToEvent(originalMove);
      expect(event).not.toBeNull();

      const convertedMove = eventToMove(event!, 2, 7);
      expect(convertedMove).not.toBeNull();

      expect(convertedMove!.type).toBe('move_stack');
      expect(convertedMove!.from).toEqual({ x: 3, y: 3 });
      expect(convertedMove!.to).toEqual({ x: 6, y: 6 });
    });
  });

  describe('validateMoveWithFSM', () => {
    const createTestGameState = (): GameState => {
      const players = [
        {
          id: 'p1',
          username: 'Player 1',
          type: 'human' as const,
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Player 2',
          type: 'ai' as const,
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      return createInitialGameState('test-game', 'square8', players, {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      });
    };

    it('should validate place_ring in ring_placement phase', () => {
      const state = createTestGameState();
      const move: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = validateMoveWithFSM(state, move);

      expect(result.valid).toBe(true);
      expect(result.currentPhase).toBe('ring_placement');
    });

    it('should reject move_stack in ring_placement phase', () => {
      const state = createTestGameState();
      const move: Move = {
        id: 'test-2',
        type: 'move_stack',
        player: 1,
        from: { x: 2, y: 2 },
        to: { x: 4, y: 4 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = validateMoveWithFSM(state, move);

      expect(result.valid).toBe(false);
      expect(result.currentPhase).toBe('ring_placement');
      expect(result.errorCode).toBe('INVALID_EVENT');
      expect(result.validEventTypes).toBeDefined();
    });

    it('should accept swap_sides as a meta-move in ring_placement', () => {
      const state = createTestGameState();
      const move: Move = {
        id: 'test-3',
        type: 'swap_sides',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = validateMoveWithFSM(state, move);

      // swap_sides is a meta-move that bypasses FSM event conversion
      // and is allowed in ring_placement via isMoveTypeValidForPhase
      expect(result.valid).toBe(true);
    });

    it('should accept skip_placement when player has stacks (player choice to not place)', () => {
      // skip_placement is a player CHOICE - the PlacementAggregate's evaluateSkipPlacementEligibility
      // allows it when the player has stacks with legal moves, regardless of whether placements exist.
      // This gives players strategic flexibility to focus on movement instead of placement.
      const state = createTestGameState();
      const move: Move = {
        id: 'test-4',
        type: 'skip_placement',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = validateMoveWithFSM(state, move);

      // Should PASS because skip_placement is trusted as a player choice
      expect(result.valid).toBe(true);
    });

    it('should reject no_placement_action during live validation when player has rings and valid placements exist', () => {
      // RR-FIX-2026-01-19: This test ensures the AI cannot pass (no_placement_action)
      // when it has rings in hand and valid positions to place them.
      // no_placement_action is for when the player CANNOT place, not when they CHOOSE not to.
      const state = createTestGameState();
      const move: Move = {
        id: 'test-5',
        type: 'no_placement_action',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = validateMoveWithFSM(state, move);

      // Should FAIL because player 1 has rings and valid placements exist
      expect(result.valid).toBe(false);
      expect(result.errorCode).toBe('GUARD_FAILED');
    });

    it('should accept no_placement_action during replay when currentPlayer differs (parity compatibility)', () => {
      // During replay, no_placement_action may be recorded for a player other than currentPlayer.
      // In this case, trust the recorded move for parity compatibility.
      const state = createTestGameState();
      // Change currentPlayer to 2, but the move is for player 1 (replay scenario)
      state.currentPlayer = 2;
      const move: Move = {
        id: 'test-6',
        type: 'no_placement_action',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      // Use replayCompatibility option
      const result = validateMoveWithFSM(state, move, false, { replayCompatibility: true });

      // Should pass because in replay mode with player mismatch, we trust the hint
      expect(result.valid).toBe(true);
    });
  });

  describe('phase ↔ MoveType mapping', () => {
    // These match VALID_MOVES_BY_PHASE in phaseValidation.ts exactly
    // (canonical only; legacy aliases live in legacyPhaseValidation.ts).
    const phaseExpectations: Record<GamePhase, MoveType[]> = {
      ring_placement: ['place_ring', 'skip_placement', 'no_placement_action', 'swap_sides'],
      movement: [
        'move_stack',
        'overtaking_capture',
        'recovery_slide',
        'skip_recovery',
        'no_movement_action',
      ],
      capture: ['overtaking_capture', 'skip_capture'],
      chain_capture: ['continue_capture_segment'],
      line_processing: [
        'process_line',
        'choose_line_option',
        // RR-CANON-R123: line reward elimination uses eliminate_rings_from_stack
        'eliminate_rings_from_stack',
        'no_line_action',
      ],
      territory_processing: [
        'choose_territory_option',
        'eliminate_rings_from_stack',
        'no_territory_action',
        'skip_territory_processing',
      ],
      // Canonical: forced_elimination only (legacy compatibility lives elsewhere)
      forced_elimination: ['forced_elimination'],
      game_over: [],
    };

    it('getAllowedMoveTypesForPhase should expose the canonical mapping for each phase', () => {
      (Object.keys(phaseExpectations) as GamePhase[]).forEach((phase) => {
        const expected = new Set(phaseExpectations[phase]);
        const actual = new Set(getAllowedMoveTypesForPhase(phase));
        expect(actual).toEqual(expected);
      });
    });

    it('isMoveTypeValidForPhase should agree with getAllowedMoveTypesForPhase for canonical phases', () => {
      (Object.keys(phaseExpectations) as GamePhase[]).forEach((phase) => {
        const allowed = new Set(phaseExpectations[phase]);
        const allTestMoveTypes: MoveType[] = [
          'place_ring',
          'skip_placement',
          'no_placement_action',
          'swap_sides',
          'move_stack',
          'move_stack',
          'build_stack',
          'overtaking_capture',
          'continue_capture_segment',
          'no_movement_action',
          'recovery_slide',
          'skip_recovery',
          'process_line',
          'choose_line_option',
          'choose_line_option',
          'no_line_action',
          'line_formation',
          'choose_territory_option',
          'choose_territory_option',
          'eliminate_rings_from_stack',
          'skip_territory_processing',
          'no_territory_action',
          'territory_claim',
          'skip_capture',
          'forced_elimination',
        ];

        for (const moveType of allTestMoveTypes) {
          const expected = allowed.has(moveType);
          const actual = isMoveTypeValidForPhase(phase, moveType);
          expect(actual).toBe(expected);
        }
      });
    });

    it('should treat resign/timeout as valid in any phase for backwards compatibility', () => {
      const phases: GamePhase[] = [
        'ring_placement',
        'movement',
        'capture',
        'chain_capture',
        'line_processing',
        'territory_processing',
        'forced_elimination',
        'game_over',
      ];
      // Per ALWAYS_VALID_MOVES, only resign and timeout are valid everywhere
      const alwaysValidMoveTypes: MoveType[] = ['resign', 'timeout'];

      for (const phase of phases) {
        for (const moveType of alwaysValidMoveTypes) {
          expect(isMoveTypeValidForPhase(phase, moveType)).toBe(true);
        }
      }
    });

    it('should treat swap_sides as valid in ring_placement only', () => {
      // Per canonical spec, swap_sides (pie rule) is only valid during ring_placement phase
      expect(isMoveTypeValidForPhase('ring_placement', 'swap_sides')).toBe(true);
      expect(isMoveTypeValidForPhase('movement', 'swap_sides')).toBe(false);
      expect(isMoveTypeValidForPhase('capture', 'swap_sides')).toBe(false);
      expect(isMoveTypeValidForPhase('chain_capture', 'swap_sides')).toBe(false);
      expect(isMoveTypeValidForPhase('line_processing', 'swap_sides')).toBe(false);
      expect(isMoveTypeValidForPhase('territory_processing', 'swap_sides')).toBe(false);
    });

    it('should reject obviously invalid combinations (smoke test)', () => {
      expect(isMoveTypeValidForPhase('ring_placement', 'move_stack')).toBe(false);
      expect(isMoveTypeValidForPhase('ring_placement', 'overtaking_capture')).toBe(false);
      expect(isMoveTypeValidForPhase('ring_placement', 'forced_elimination')).toBe(false);
      // Canonical: forced_elimination only accepts forced_elimination moves
      expect(isMoveTypeValidForPhase('forced_elimination', 'move_stack')).toBe(false);
      expect(isMoveTypeValidForPhase('forced_elimination', 'overtaking_capture')).toBe(false);
      expect(isMoveTypeValidForPhase('forced_elimination', 'place_ring')).toBe(false);
    });
  });

  describe('computeFSMOrchestration', () => {
    const createTestGameState = (): GameState => {
      const players = [
        {
          id: 'p1',
          username: 'Player 1',
          type: 'human' as const,
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Player 2',
          type: 'ai' as const,
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      return createInitialGameState('test-game', 'square8', players, {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      });
    };

    it('should return success for valid place_ring move', () => {
      const state = createTestGameState();
      const move: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = computeFSMOrchestration(state, move);

      expect(result.success).toBe(true);
      expect(result.nextPhase).toBe('movement');
      expect(result.nextPlayer).toBe(1);
    });

    it('should return decision surface with pending decision type for line phase', () => {
      // Create state in line_processing phase
      const state = createTestGameState();
      (state as { currentPhase: string }).currentPhase = 'line_processing';

      const move: Move = {
        id: 'test-2',
        type: 'no_line_action',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      };

      const result = computeFSMOrchestration(state, move);

      expect(result.success).toBe(true);
      expect(result.nextPhase).toBe('territory_processing');
      // Decision surface should be populated for the next phase
      // (territory_processing with no regions)
      expect(result.pendingDecisionType).toBe('no_territory_action_required');
      expect(result.decisionSurface).toBeDefined();
      expect(result.decisionSurface!.pendingLines).toEqual([]);
      expect(result.decisionSurface!.pendingRegions).toEqual([]);
    });

    it('should include debug information', () => {
      const state = createTestGameState();
      const move: Move = {
        id: 'test-3',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = computeFSMOrchestration(state, move);

      expect(result.debug).toBeDefined();
      expect(result.debug!.inputPhase).toBe('ring_placement');
      expect(result.debug!.inputPlayer).toBe(1);
      expect(result.debug!.fsmState).toBeDefined();
      expect(result.debug!.event).toEqual({ type: 'PLACE_RING', to: { x: 3, y: 3 } });
    });

    it('should handle meta-moves gracefully', () => {
      const state = createTestGameState();
      const move: Move = {
        id: 'test-4',
        type: 'swap_sides',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = computeFSMOrchestration(state, move);

      // Meta-moves should succeed without transition
      expect(result.success).toBe(true);
      expect(result.actions).toEqual([]);
    });
  });
});
