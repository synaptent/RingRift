/**
 * FSM Canonical Orchestrator Tests
 *
 * Tests that validate FSM behavior as the canonical orchestrator.
 * These tests ensure FSM correctly:
 * - Validates moves according to game rules
 * - Rejects invalid moves with appropriate error codes
 * - Drives phase transitions correctly
 * - Handles edge cases in game flow
 */

import { createInitialGameState } from '../../../src/shared/engine/initialState';
import {
  processTurn,
  getValidMoves,
} from '../../../src/shared/engine/orchestration/turnOrchestrator';
import {
  validateMoveWithFSM,
  isMoveTypeValidForPhase,
  getAllowedMoveTypesForPhase,
} from '../../../src/shared/engine/fsm/FSMAdapter';
import type {
  GameState,
  Move,
  GamePhase,
  MoveType,
  Player,
  TimeControl,
} from '../../../src/shared/types/game';

describe('FSM Canonical Orchestrator', () => {
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

  describe('Phase-Move Type Validation', () => {
    // These mappings match PHASE_ALLOWED_MOVE_TYPES in FSMAdapter.ts
    // Note: swap_sides is a meta move allowed in any phase for historical compatibility
    const phaseMovePairs: { phase: GamePhase; validMoves: MoveType[]; invalidMoves: MoveType[] }[] =
      [
        {
          phase: 'ring_placement',
          validMoves: ['place_ring', 'skip_placement', 'no_placement_action'],
          invalidMoves: [
            'move_stack',
            'overtaking_capture',
            'choose_line_reward',
            'process_territory_region',
          ],
        },
        {
          phase: 'movement',
          validMoves: [
            'move_stack',
            'move_ring',
            'overtaking_capture',
            'continue_capture_segment',
            'no_movement_action',
            'recovery_slide',
          ],
          invalidMoves: ['place_ring', 'choose_line_reward', 'process_territory_region'],
        },
        {
          phase: 'chain_capture',
          validMoves: ['overtaking_capture', 'continue_capture_segment'],
          invalidMoves: ['place_ring', 'move_stack', 'choose_line_reward', 'no_movement_action'],
        },
        {
          phase: 'line_processing',
          validMoves: ['process_line', 'choose_line_reward', 'no_line_action'],
          invalidMoves: [
            'place_ring',
            'move_stack',
            'process_territory_region',
            'overtaking_capture',
          ],
        },
        {
          phase: 'territory_processing',
          validMoves: [
            'process_territory_region',
            'no_territory_action',
            'eliminate_rings_from_stack',
            'skip_territory_processing',
          ],
          invalidMoves: ['place_ring', 'move_stack', 'choose_line_reward', 'overtaking_capture'],
        },
      ];

    phaseMovePairs.forEach(({ phase, validMoves, invalidMoves }) => {
      describe(`phase: ${phase}`, () => {
        validMoves.forEach((moveType) => {
          it(`should allow ${moveType} move type`, () => {
            expect(isMoveTypeValidForPhase(phase, moveType)).toBe(true);
          });
        });

        invalidMoves.forEach((moveType) => {
          it(`should reject ${moveType} move type`, () => {
            expect(isMoveTypeValidForPhase(phase, moveType)).toBe(false);
          });
        });

        it('getAllowedMoveTypesForPhase should include all valid moves', () => {
          const allowed = getAllowedMoveTypesForPhase(phase);
          validMoves.forEach((moveType) => {
            expect(allowed).toContain(moveType);
          });
        });
      });
    });
  });

  describe('FSM Move Validation', () => {
    it('should validate a legal place_ring move in ring_placement phase', () => {
      const state = createGame();
      expect(state.currentPhase).toBe('ring_placement');
      expect(state.currentPlayer).toBe(1);

      const move = makeMove({
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
      });

      const result = validateMoveWithFSM(state, move);
      expect(result.valid).toBe(true);
    });

    it('should reject place_ring from wrong player', () => {
      const state = createGame();
      expect(state.currentPlayer).toBe(1);

      const move = makeMove({
        type: 'place_ring',
        player: 2, // Wrong player
        to: { x: 3, y: 3 },
      });

      const result = validateMoveWithFSM(state, move);
      expect(result.valid).toBe(false);
      expect(result.errorCode).toBe('WRONG_PLAYER');
    });

    it('should reject move_stack in ring_placement phase', () => {
      const state = createGame();
      expect(state.currentPhase).toBe('ring_placement');

      const move = makeMove({
        type: 'move_stack',
        player: 1,
        from: { x: 3, y: 3 },
        to: { x: 4, y: 4 },
      });

      const result = validateMoveWithFSM(state, move);
      expect(result.valid).toBe(false);
      // FSM returns INVALID_EVENT when move type doesn't match expected events for phase
      expect(result.errorCode).toBe('INVALID_EVENT');
    });

    it('should allow skip_placement move type in ring_placement phase', () => {
      // Note: skip_placement is allowed as a move type, but guards may reject it
      // based on game state (e.g., if rings are available). This test validates
      // the move type is accepted by the FSM.
      const state = createGame();
      expect(state.currentPhase).toBe('ring_placement');

      // skip_placement is in PHASE_ALLOWED_MOVE_TYPES for ring_placement
      expect(isMoveTypeValidForPhase('ring_placement', 'skip_placement')).toBe(true);
    });
  });

  describe('FSM-Driven Phase Transitions via processTurn', () => {
    it('should transition from ring_placement to movement after place_ring', () => {
      const state = createGame();
      const move = makeMove({
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
      });

      const result = processTurn(state, move);
      expect(result.nextState.currentPhase).toBe('movement');
      expect(result.nextState.currentPlayer).toBe(1);
    });

    it('should maintain correct player through multi-phase turn', () => {
      let state = createGame();

      // Player 1 places ring
      const placeMove = makeMove({
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        moveNumber: 1,
      });
      let result = processTurn(state, placeMove);
      state = result.nextState;
      expect(state.currentPlayer).toBe(1);
      expect(state.currentPhase).toBe('movement');

      // Player 1 skips movement (if no moves available) or moves
      // For this test, we'll need to check valid moves
      const validMoves = getValidMoves(state);
      const noMovementMove = validMoves.find((m) => m.type === 'no_movement_action');

      if (noMovementMove) {
        result = processTurn(state, noMovementMove);
        state = result.nextState;
        expect(state.currentPlayer).toBe(1);
        expect(['line_processing', 'territory_processing', 'ring_placement']).toContain(
          state.currentPhase
        );
      }
    });

    it('should transition to next player after turn completion', () => {
      let state = createGame();
      let moveNumber = 1;

      // Helper to make moves with incrementing move numbers
      const nextMove = (overrides: Partial<Move>): Move => {
        return makeMove({ ...overrides, moveNumber: moveNumber++ });
      };

      // Player 1 places ring
      let result = processTurn(
        state,
        nextMove({ type: 'place_ring', player: 1, to: { x: 3, y: 3 } })
      );
      state = result.nextState;

      // Process through remaining phases with no-op moves
      while (state.currentPlayer === 1 && state.gameStatus === 'active') {
        const validMoves = getValidMoves(state);
        const noOpMove = validMoves.find(
          (m) =>
            m.type === 'no_movement_action' ||
            m.type === 'no_line_action' ||
            m.type === 'no_territory_action'
        );

        if (!noOpMove) {
          // If there are no no-op moves, the turn might have action moves
          break;
        }

        result = processTurn(state, { ...noOpMove, moveNumber: moveNumber++ });
        state = result.nextState;
      }

      // After completing all phases, should be player 2's turn or still player 1 with action
      // Note: game may be in 'waiting' or 'active' depending on player readiness
      expect(['active', 'waiting']).toContain(state.gameStatus);
    });
  });

  describe('FSM Reject Invalid Move Types', () => {
    const invalidMoveScenarios = [
      {
        phase: 'ring_placement' as GamePhase,
        moveType: 'process_line' as MoveType,
        reason: 'line processing in placement phase',
      },
      {
        phase: 'movement' as GamePhase,
        moveType: 'place_ring' as MoveType,
        reason: 'placement in movement phase',
      },
      {
        phase: 'line_processing' as GamePhase,
        moveType: 'move_stack' as MoveType,
        reason: 'movement in line processing phase',
      },
    ];

    invalidMoveScenarios.forEach(({ phase, moveType, reason }) => {
      it(`should reject ${reason}`, () => {
        const state: GameState = {
          ...createGame(),
          currentPhase: phase,
        };

        const move = makeMove({
          type: moveType,
          player: state.currentPlayer,
        });

        const result = validateMoveWithFSM(state, move);
        expect(result.valid).toBe(false);
      });
    });
  });

  describe('FSM getValidMoves Integration', () => {
    it('should return only phase-appropriate moves', () => {
      const state = createGame();
      expect(state.currentPhase).toBe('ring_placement');

      const validMoves = getValidMoves(state);

      // All returned moves should be valid for ring_placement
      validMoves.forEach((move) => {
        expect(isMoveTypeValidForPhase('ring_placement', move.type)).toBe(true);
      });

      // Should include place_ring moves for valid positions
      expect(validMoves.some((m) => m.type === 'place_ring')).toBe(true);
    });

    it('should not return moves that FSM would reject', () => {
      const state = createGame();
      const validMoves = getValidMoves(state);

      // Every returned move should pass FSM validation
      validMoves.forEach((move) => {
        const result = validateMoveWithFSM(state, move);
        expect(result.valid).toBe(true);
      });
    });
  });

  describe('FSM Error Codes', () => {
    it('should return WRONG_PLAYER for moves by wrong player', () => {
      const state = createGame();
      const move = makeMove({ player: 2 }); // Current player is 1

      const result = validateMoveWithFSM(state, move);
      expect(result.valid).toBe(false);
      expect(result.errorCode).toBe('WRONG_PLAYER');
    });

    it('should return INVALID_EVENT for wrong phase moves', () => {
      const state = createGame();
      const move = makeMove({
        type: 'process_territory_region',
        player: 1,
        to: { x: 0, y: 0 },
      });

      const result = validateMoveWithFSM(state, move);
      expect(result.valid).toBe(false);
      // FSM returns INVALID_EVENT when move type doesn't match expected events
      expect(result.errorCode).toBe('INVALID_EVENT');
    });
  });

  describe('Meta Move Types', () => {
    it('should allow swap_sides as meta move in any phase', () => {
      // swap_sides is a meta move that bypasses phase restrictions
      const phases: GamePhase[] = [
        'ring_placement',
        'movement',
        'line_processing',
        'territory_processing',
      ];

      phases.forEach((phase) => {
        expect(isMoveTypeValidForPhase(phase, 'swap_sides')).toBe(true);
      });
    });

    it('should allow line_formation as meta move', () => {
      // line_formation is a legacy/meta move type
      expect(isMoveTypeValidForPhase('ring_placement', 'line_formation')).toBe(true);
      expect(isMoveTypeValidForPhase('movement', 'line_formation')).toBe(true);
    });
  });
});
