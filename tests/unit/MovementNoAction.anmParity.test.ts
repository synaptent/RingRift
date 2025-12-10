import type { GameState, Move } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';
import { createTestBoard, createTestGameState, pos, addStack } from '../utils/fixtures';
import { getValidMoves } from '../../src/shared/engine/orchestration/turnOrchestrator';
import { validateMoveWithFSM, getCurrentFSMState } from '../../src/shared/engine/fsm/FSMAdapter';

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
 */
describe('Movement ANM / no_movement_action parity', () => {
  function makeForcedAnmState(): GameState {
    const board = createTestBoard('square8');
    const origin = pos(3, 3);
    const player = 1;

    const originKey = positionToString(origin);

    // Single stack for player 1 at origin.
    addStack(board, origin, player, 1);

    // Collapse every other space so the stack is completely stuck:
    // no simple movement and no captures are possible.
    for (let x = 0; x < board.size; x += 1) {
      for (let y = 0; y < board.size; y += 1) {
        const key = positionToString({ x, y });
        if (key === originKey) continue;
        board.collapsedSpaces.set(key, player);
      }
    }

    return createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: player,
      currentPhase: 'movement',
    });
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
});