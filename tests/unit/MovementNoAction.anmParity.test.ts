import type { GameState, Move } from '../../src/shared/types/game';
import { createTestBoard, createTestGameState, pos, addStack } from '../utils/fixtures';
import { getValidMoves } from '../../src/shared/engine/orchestration/turnOrchestrator';
import { validateMoveWithFSM, getCurrentFSMState } from '../../src/shared/engine/fsm/FSMAdapter';
import {
  computeGlobalLegalActionsSummary,
  isANMState,
} from '../../src/shared/engine/globalActions';
import {
  makeAnmScen01_MovementNoMovesButFEAvailable,
  makeAnmScen02_MovementPlacementsOnly,
  makeAnmScen03_MovementCurrentPlayerFullyEliminated,
  makeMovementMustMoveFromStackKeyConstrainedState,
} from '../fixtures/anmFixtures';

/**
 * Movement ANM / no_movement_action parity tests.
 *
 * These tests verify that:
 * - When a player truly has no legal movement/capture/recovery moves in
 *   MOVEMENT, TS enumerates zero interactive moves and the FSM flags
 *   canMove = false and accepts no_movement_action.
 * - When movement moves exist, canMove = true so hosts never treat the
 *   state as ANM.
 *
 * They are a TS-side analogue of the Python GameEngine.get_valid_moves /
 * get_phase_requirement contract for MOVEMENT and RR‑CANON‑R075/R076.
 *
 * Additional ANM coverage (ANM-SCEN-01/02/03 + mustMoveFromStackKey) checks
 * the R200 global legal action summary and ANM predicate from
 * src/shared/engine/globalActions.ts.
 */
describe('Movement ANM / no_movement_action parity', () => {
  /**
   * ANM-SCEN-01 – Movement phase, no moves but forced elimination available.
   *
   * This forwards to the shared ANM fixture so movement tests stay aligned
   * with the canonical global-actions catalogue.
   */
  function makeForcedAnmState(): GameState {
    return makeAnmScen01_MovementNoMovesButFEAvailable();
  }

  function makeNonAnmStateWithMovement(): GameState {
    const board = createTestBoard('square8');
    const origin = pos(3, 3);
    const player = 1;

    // Height‑2 stack in the middle of an otherwise empty board → many moves.
    addStack(board, origin, player, 2);

    return createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: player,
      currentPhase: 'movement',
    });
  }

  test('forced ANM: no interactive movement moves and FSM accepts no_movement_action', () => {
    const state = makeForcedAnmState();

    const validMoves = getValidMoves(state);
    const movementLike = validMoves.filter(
      (m) =>
        m.player === state.currentPlayer &&
        (m.type === 'move_stack' ||
          m.type === 'move_ring' ||
          m.type === 'overtaking_capture' ||
          m.type === 'continue_capture_segment' ||
          m.type === 'recovery_slide')
    );

    // Phase‑local surface for MOVEMENT has no interactive actions.
    expect(movementLike.length).toBe(0);

    const fsmState = getCurrentFSMState(state) as any;
    expect(fsmState.phase).toBe('movement');
    expect(fsmState.player).toBe(1);
    expect(fsmState.canMove).toBe(false);

    const move: Move = {
      id: 'anm-no-move-1',
      type: 'no_movement_action',
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: state.moveHistory.length + 1,
    };

    const fsmResult = validateMoveWithFSM(state, move);
    expect(fsmResult.valid).toBe(true);
    expect(fsmResult.errorCode).toBeUndefined();
  });

  test('non‑ANM: movement moves exist and FSM reports canMove = true', () => {
    const state = makeNonAnmStateWithMovement();

    const validMoves = getValidMoves(state);
    const movementLike = validMoves.filter(
      (m) =>
        m.player === state.currentPlayer &&
        (m.type === 'move_stack' ||
          m.type === 'move_ring' ||
          m.type === 'overtaking_capture' ||
          m.type === 'continue_capture_segment' ||
          m.type === 'recovery_slide')
    );

    expect(movementLike.length).toBeGreaterThan(0);

    const fsmState = getCurrentFSMState(state) as any;
    expect(fsmState.phase).toBe('movement');
    expect(fsmState.player).toBe(1);
    expect(fsmState.canMove).toBe(true);
  });

  /**
   * ANM-SCEN-01 – Movement phase, no movement/capture, FE available.
   *
   * Global legal action expectations:
   * - hasTurnMaterial            === true
   * - hasPhaseLocalInteractiveMove === false  (no movement/capture)
   * - hasGlobalPlacementAction   === false
   * - hasForcedEliminationAction === true
   * - isANMState(state)          === false (FE is a global action)
   */
  test('ANM-SCEN-01: movement no moves but forced elimination available is not ANM', () => {
    const state = makeAnmScen01_MovementNoMovesButFEAvailable();
    const currentPlayer = state.currentPlayer;

    const summary = computeGlobalLegalActionsSummary(state, currentPlayer);

    expect(summary.hasTurnMaterial).toBe(true);
    expect(summary.hasPhaseLocalInteractiveMove).toBe(false);
    expect(summary.hasGlobalPlacementAction).toBe(false);
    expect(summary.hasForcedEliminationAction).toBe(true);
    expect(isANMState(state)).toBe(false);
  });

  /**
   * ANM-SCEN-02 – Movement phase, placements-only global actions.
   *
   * Global legal action expectations:
   * - hasTurnMaterial            === true (rings in hand)
   * - hasPhaseLocalInteractiveMove === false  (no movement/capture)
   * - hasGlobalPlacementAction   === true
   * - hasForcedEliminationAction === false
   * - isANMState(state)          === false (placements exist globally)
   */
  test('ANM-SCEN-02: movement with placements-only global actions is not ANM', () => {
    const state = makeAnmScen02_MovementPlacementsOnly();
    const currentPlayer = state.currentPlayer;

    const summary = computeGlobalLegalActionsSummary(state, currentPlayer);

    expect(summary.hasTurnMaterial).toBe(true);
    expect(summary.hasPhaseLocalInteractiveMove).toBe(false);
    expect(summary.hasGlobalPlacementAction).toBe(true);
    expect(summary.hasForcedEliminationAction).toBe(false);
    expect(isANMState(state)).toBe(false);
  });

  /**
   * ANM-SCEN-03 – Movement phase with a fully eliminated current player.
   *
   * Global legal action expectations:
   * - hasTurnMaterial            === false
   * - hasGlobalPlacementAction   === false
   * - hasPhaseLocalInteractiveMove === false
   * - hasForcedEliminationAction === false
   * - isANMState(state)          === false (ANM is defined only for players with material)
   */
  test('ANM-SCEN-03: fully eliminated current player is not classified as ANM', () => {
    const state = makeAnmScen03_MovementCurrentPlayerFullyEliminated();
    const currentPlayer = state.currentPlayer;

    const summary = computeGlobalLegalActionsSummary(state, currentPlayer);

    expect(summary.hasTurnMaterial).toBe(false);
    expect(summary.hasGlobalPlacementAction).toBe(false);
    expect(summary.hasPhaseLocalInteractiveMove).toBe(false);
    expect(summary.hasForcedEliminationAction).toBe(false);
    expect(isANMState(state)).toBe(false);
  });

  /**
   * Edge case: mustMoveFromStackKey constrains movement surface.
   *
   * Scenario:
   * - Player 1 controls two stacks:
   *   - A "stuck" stack surrounded by collapsed territory (no legal moves).
   *   - A "free" stack that would have legal moves if unconstrained.
   * - state.mustMoveFromStackKey is set to the stuck stack.
   *
   * Expectations:
   * - With mustMoveFromStackKey set:
   *     hasPhaseLocalInteractiveMove === false for P1 in MOVEMENT.
   * - With the constraint removed:
   *     hasPhaseLocalInteractiveMove === true for P1.
   * - In both cases placements remain available, so isANMState(state) === false.
   *
   * This guards against regressions where movement constraints cause
   * movement-only ANM mis-classification.
   */
  test('mustMoveFromStackKey-constrained movement surface does not mis-classify ANM', () => {
    const constrained = makeMovementMustMoveFromStackKeyConstrainedState();
    const currentPlayer = constrained.currentPlayer;

    const constrainedSummary = computeGlobalLegalActionsSummary(constrained, currentPlayer);
    expect(constrainedSummary.hasTurnMaterial).toBe(true);
    expect(constrainedSummary.hasGlobalPlacementAction).toBe(true);
    // Movement/capture surface is empty when constrained to the stuck stack.
    expect(constrainedSummary.hasPhaseLocalInteractiveMove).toBe(false);
    expect(isANMState(constrained)).toBe(false);

    // Remove mustMoveFromStackKey while keeping the same board so that the free
    // stack can move; this simulates the unconstrained global reachability check.
    const unconstrained: GameState = {
      ...constrained,
      mustMoveFromStackKey: undefined,
    };

    const unconstrainedSummary = computeGlobalLegalActionsSummary(unconstrained, currentPlayer);
    expect(unconstrainedSummary.hasTurnMaterial).toBe(true);
    expect(unconstrainedSummary.hasGlobalPlacementAction).toBe(true);
    // With the constraint removed, movement/capture should now be available.
    expect(unconstrainedSummary.hasPhaseLocalInteractiveMove).toBe(true);
    expect(isANMState(unconstrained)).toBe(false);
  });
});
