import type { GameState, Position, Move } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

import {
  validateMovement,
  enumerateMovementTargets,
  enumerateSimpleMovesForPlayer,
  enumerateAllMovementMoves,
  mutateMovement,
  applySimpleMovement as applySimpleMovementAggregate,
  applyMovement,
} from '../../src/shared/engine/aggregates/MovementAggregate';

import type { MoveStackAction } from '../../src/shared/engine/types';

import {
  createTestBoard,
  createTestGameState,
  addStack,
  addMarker,
  addCollapsedSpace,
  pos,
} from '../utils/fixtures';

describe('MovementAggregate.validateMovement', () => {
  function makeBaseState(): GameState {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    addStack(board, from, 1, 2);

    return createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });
  }

  test('rejects when not in movement phase', () => {
    const state = makeBaseState();
    // Override phase to a non-movement value.
    const badPhaseState: GameState = {
      ...state,
      currentPhase: 'ring_placement',
    };

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from: pos(2, 2),
      to: pos(4, 2),
    };

    const result = validateMovement(badPhaseState, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure for wrong phase');
    }
    expect(result.code).toBe('INVALID_PHASE');
  });

  test('rejects when it is not the players turn', () => {
    const state = makeBaseState();
    const badTurnState: GameState = {
      ...state,
      currentPlayer: 2,
    };

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from: pos(2, 2),
      to: pos(4, 2),
    };

    const result = validateMovement(badTurnState, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure for wrong player');
    }
    expect(result.code).toBe('NOT_YOUR_TURN');
  });

  test('rejects when from or to position is off board', () => {
    const state = makeBaseState();

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from: pos(-1, 0),
      to: pos(0, 0),
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure for off-board position');
    }
    expect(result.code).toBe('INVALID_POSITION');
  });

  test('rejects when there is no stack at the origin', () => {
    const base = makeBaseState();
    // Clear all stacks so origin is empty.
    const emptyBoard = createTestBoard('square8');
    const state: GameState = {
      ...base,
      board: emptyBoard,
    };

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from: pos(2, 2),
      to: pos(4, 2),
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure when origin has no stack');
    }
    expect(result.code).toBe('NO_STACK');
  });

  test('rejects when the stack at origin is not controlled by the player', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    // Stack is controlled by player 2, but player 1 tries to move it.
    addStack(board, from, 2, 2);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to: pos(4, 2),
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure for foreign-controlled stack');
    }
    expect(result.code).toBe('NOT_YOUR_STACK');
  });

  test('rejects moves that land on collapsed spaces', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    const to = pos(4, 2);
    addStack(board, from, 1, 2);
    addCollapsedSpace(board, to, 1);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to,
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure for collapsed landing');
    }
    expect(result.code).toBe('COLLAPSED_SPACE');
  });

  test('rejects invalid (off-axis) movement direction for square boards', () => {
    const board = createTestBoard('square8');
    const from = pos(1, 1);
    // Height 1 stack so distance requirement is trivially satisfied; only direction fails.
    addStack(board, from, 1, 1);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      // (2,3) is neither orthogonal nor diagonal from (1,1).
      to: pos(2, 3),
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure for invalid square direction');
    }
    expect(result.code).toBe('INVALID_DIRECTION');
  });

  test('rejects invalid (off-axis) movement direction for hex boards', () => {
    const board = createTestBoard('hexagonal');
    const from: Position = { x: 0, y: 0, z: 0 };
    addStack(board, from, 1, 1);

    const state = createTestGameState({
      boardType: 'hexagonal',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      // (1,0,0) is not on any of the six cube axes from (0,0,0).
      to: { x: 1, y: 0, z: 0 },
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure for invalid hex direction');
    }
    // Accept either INVALID_POSITION (cube coords don't sum to 0) or
    // INVALID_DIRECTION (off-axis) since both indicate an invalid hex move.
    expect(['INVALID_DIRECTION', 'INVALID_POSITION']).toContain(result.code);
  });

  test('rejects moves whose distance is less than stack height', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    const to = pos(3, 2); // distance 1
    // Height 2 requires distance >= 2.
    addStack(board, from, 1, 2);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to,
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure for insufficient distance');
    }
    expect(result.code).toBe('INSUFFICIENT_DISTANCE');
  });

  test('rejects moves whose inner path is blocked by a collapsed space', () => {
    const board = createTestBoard('square8');
    const from = pos(1, 1);
    const to = pos(4, 1);
    addStack(board, from, 1, 1);
    // Collapse an inner path cell (2,1).
    addCollapsedSpace(board, pos(2, 1), 2);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to,
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure for path blocked by collapsed space');
    }
    expect(result.code).toBe('PATH_BLOCKED');
  });

  test('rejects moves whose inner path is blocked by another stack', () => {
    const board = createTestBoard('square8');
    const from = pos(1, 1);
    const to = pos(4, 1);
    addStack(board, from, 1, 1);
    // Stack at (2,1) should block the ray.
    addStack(board, pos(2, 1), 2, 1);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to,
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure for path blocked by stack');
    }
    expect(result.code).toBe('PATH_BLOCKED');
  });

  test('rejects landing on an opponent marker', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    const to = pos(4, 2);
    addStack(board, from, 1, 2);
    addMarker(board, to, 2);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to,
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure for landing on opponent marker');
    }
    expect(result.code).toBe('INVALID_LANDING');
  });

  test('allows landing on own marker with clear path', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    const to = pos(4, 2);
    addStack(board, from, 1, 2);
    addMarker(board, to, 1);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to,
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(true);
  });

  test('rejects landing on an existing stack', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    const to = pos(4, 2);
    addStack(board, from, 1, 2);
    // Destination already has a stack.
    addStack(board, to, 1, 1);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to,
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected validation failure for landing on stack');
    }
    expect(result.code).toBe('INVALID_LANDING');
  });

  test('accepts a simple move along a clear path to an empty cell', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    const to = pos(5, 2);
    addStack(board, from, 1, 2);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to,
    };

    const result = validateMovement(state, action);
    expect(result.valid).toBe(true);
  });
});

describe('MovementAggregate.mutateMovement', () => {
  test('moves stack, leaves departure marker, and updates destination on empty cell', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    const to = pos(4, 2);
    addStack(board, from, 1, 2);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to,
    };

    const newState = mutateMovement(state as any, action);
    const fromKey = positionToString(from);
    const toKey = positionToString(to);

    expect(newState.board.stacks.has(fromKey)).toBe(false);
    const destStack = newState.board.stacks.get(toKey);
    expect(destStack).toBeDefined();
    expect(destStack?.stackHeight).toBe(2);
    expect(destStack?.controllingPlayer).toBe(1);

    const departureMarker = newState.board.markers.get(fromKey);
    expect(departureMarker).toBeDefined();
    expect(departureMarker?.player).toBe(1);
  });

  test('eliminates top ring when landing on own marker', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    const to = pos(4, 2);
    addStack(board, from, 1, 2);
    addMarker(board, to, 1);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to,
    };

    const newState = mutateMovement(state as any, action);
    const toKey = positionToString(to);
    const fromKey = positionToString(from);

    const destStack = newState.board.stacks.get(toKey);
    expect(destStack).toBeDefined();
    // Height should be reduced by one after eliminating top ring.
    expect(destStack?.stackHeight).toBe(1);
    // Landing marker should have been removed.
    expect(newState.board.markers.has(toKey)).toBe(false);
    // Departure marker still present.
    expect(newState.board.markers.has(fromKey)).toBe(true);
    // Elimination accounting updated.
    expect(newState.board.eliminatedRings[1]).toBe(1);
    const player = newState.players.find((p) => p.playerNumber === 1);
    expect(player?.eliminatedRings).toBe(1);
  });

  test('throws when landing on opponent marker (defensive guard)', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    const to = pos(4, 2);
    addStack(board, from, 1, 1);
    addMarker(board, to, 2);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to,
    };

    expect(() => mutateMovement(state as any, action)).toThrow(
      'MovementMutator: Landing on opponent marker is not supported in simple movement'
    );
  });

  test('throws when there is no stack at origin', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    const to = pos(4, 2);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const action: MoveStackAction = {
      type: 'MOVE_STACK',
      playerId: 1,
      from,
      to,
    };

    expect(() => mutateMovement(state as any, action)).toThrow(
      'MovementMutator: No stack at origin'
    );
  });
});

describe('MovementAggregate enumeration helpers', () => {
  test('enumerateMovementTargets returns reachable positions respecting minimum distance', () => {
    const board = createTestBoard('square8');
    const from = pos(3, 3);
    // Height 2 stack gives minimum distance of 2.
    addStack(board, from, 1, 2);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const targets = enumerateMovementTargets(state, from);
    expect(targets.length).toBeGreaterThan(0);

    for (const to of targets) {
      const dx = Math.abs(to.x - from.x);
      const dy = Math.abs(to.y - from.y);
      const distance = Math.max(dx, dy);
      expect(distance).toBeGreaterThanOrEqual(2);
    }
  });

  test('enumerateSimpleMovesForPlayer produces Move objects and matches enumerateAllMovementMoves', () => {
    const board = createTestBoard('square8');
    const from1 = pos(2, 2);
    const from2 = pos(5, 5);
    addStack(board, from1, 1, 1);
    addStack(board, from2, 1, 2);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const movesForPlayer = enumerateSimpleMovesForPlayer(state, 1);
    const allMoves = enumerateAllMovementMoves(state, 1);

    // Normalize timestamps before comparison (they may differ by a few ms)
    const normalizeTimestamp = (move: any) => ({ ...move, timestamp: new Date(0) });
    const normalizedMovesForPlayer = movesForPlayer.map(normalizeTimestamp);
    const normalizedAllMoves = allMoves.map(normalizeTimestamp);

    expect(normalizedAllMoves).toEqual(normalizedMovesForPlayer);
    expect(movesForPlayer.length).toBeGreaterThan(0);

    for (const move of movesForPlayer) {
      expect(move.type === 'move_stack' || move.type === 'move_ring').toBe(true);
      expect(move.player).toBe(1);
      expect(move.from).toBeDefined();
      expect(move.to).toBeDefined();
    }
  });
});

describe('MovementAggregate.applyMovement', () => {
  test('rejects non-movement move types', () => {
    const board = createTestBoard('square8');
    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const move: Move = {
      id: 'm1',
      type: 'place_ring',
      player: 1,
      to: pos(0, 0),
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const result = applyMovement(state as any, move);
    if (result.success) {
      throw new Error('Expected applyMovement to fail for non-movement move type');
    }
    expect(result.reason).toMatch(/Expected 'move_stack' or 'move_ring'/);
  });

  test('rejects movement moves without a from position', () => {
    const board = createTestBoard('square8');
    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const move: Move = {
      id: 'm2',
      type: 'move_stack',
      player: 1,
      // Intentionally omit from.
      from: undefined as any,
      to: pos(1, 1),
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const result = applyMovement(state as any, move);
    if (result.success) {
      throw new Error('Expected applyMovement to fail when from is missing');
    }
    expect(result.reason).toBe('Move.from is required for movement moves');
  });

  test('successfully applies a simple movement move via applySimpleMovement', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    const to = pos(4, 2);
    addStack(board, from, 1, 2);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const move: Move = {
      id: 'm3',
      type: 'move_stack',
      player: 1,
      from,
      to,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const outcome = applySimpleMovementAggregate(state as any, {
      from,
      to,
      player: 1,
    });

    const wrapped = applyMovement(state as any, move);
    expect(wrapped.success).toBe(true);
    if (!wrapped.success) {
      throw new Error('Expected applyMovement to succeed for simple move');
    }

    const directStacks = outcome.nextState.board.stacks;
    const wrappedStacks = wrapped.newState.board.stacks;
    expect(Array.from(wrappedStacks.entries())).toEqual(Array.from(directStacks.entries()));
  });

  test('returns a structured error when underlying movement throws', () => {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    const to = pos(4, 2);

    const state = createTestGameState({
      boardType: 'square8',
      board,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const move: Move = {
      id: 'm4',
      type: 'move_stack',
      player: 1,
      from,
      to,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const result = applyMovement(state as any, move);
    if (result.success) {
      throw new Error('Expected applyMovement to surface underlying movement error');
    }
    expect(result.reason).toBe('applySimpleMovement: No stack at origin');
  });
});
