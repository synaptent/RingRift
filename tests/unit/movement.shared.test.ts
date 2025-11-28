import {
  BoardType,
  BoardState,
  Position,
  Move,
  positionToString,
} from '../../src/shared/types/game';
import {
  MovementBoardView,
  hasAnyLegalMoveOrCaptureFromOnBoard,
} from '../../src/shared/engine/core';
import { enumerateSimpleMoveTargetsFromStack } from '../../src/shared/engine/movementLogic';
import { createTestBoard, createTestGameState, addStack, addMarker, pos } from '../utils/fixtures';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import { BoardManager } from '../../src/server/game/BoardManager';
import { enumerateSimpleMovementLandings } from '../../src/client/sandbox/sandboxMovement';

function isValidPositionForBoard(boardType: BoardType, board: BoardState, p: Position): boolean {
  if (boardType === 'hexagonal') {
    const radius = board.size - 1;
    const x = p.x;
    const y = p.y;
    const z = p.z !== undefined ? p.z : -x - y;
    const dist = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
    return dist <= radius;
  }

  return p.x >= 0 && p.x < board.size && p.y >= 0 && p.y < board.size;
}

function makeMovementView(boardType: BoardType, board: BoardState): MovementBoardView {
  return {
    isValidPosition: (p: Position) => isValidPositionForBoard(boardType, board, p),
    isCollapsedSpace: (p: Position) => {
      const key = positionToString(p);
      return board.collapsedSpaces.has(key);
    },
    getStackAt: (p: Position) => {
      const key = positionToString(p);
      const stack = board.stacks.get(key);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (p: Position) => {
      const key = positionToString(p);
      const marker = board.markers.get(key);
      return marker?.player;
    },
  };
}

// Classification: canonical shared movement helper tests for enumerateSimpleMoveTargetsFromStack
// plus cross-host parity against sandbox and backend RuleEngine.

// Skip parity tests with orchestrator adapter - these test internal cross-host consistency
// which may differ with the orchestrator's unified processing flow.
const skipParityWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

describe('enumerateSimpleMoveTargetsFromStack shared helper', () => {
  test('square8: shared vs sandbox vs RuleEngine on open board', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(3, 3);
    const player = 1;

    // Height 2 stack in the middle of an otherwise empty board.
    addStack(board, from, player, 2);

    const view = makeMovementView(boardType, board);
    const fromKey = positionToString(from);

    const sharedTargets = enumerateSimpleMoveTargetsFromStack(boardType, from, player, view)
      .map((t) => positionToString(t.to))
      .sort();

    const sandboxTargets = enumerateSimpleMovementLandings(
      boardType,
      board,
      player,
      (p: Position) => isValidPositionForBoard(boardType, board, p)
    )
      .filter((m) => m.fromKey === fromKey)
      .map((m) => positionToString(m.to))
      .sort();

    const bm = new BoardManager(boardType);
    const engine = new RuleEngine(bm as any, boardType as any);
    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'movement',
    });

    const backendTargets = engine
      .getValidMoves(state)
      .filter(
        (m: Move) =>
          (m.type === 'move_stack' || m.type === 'move_ring') &&
          m.from &&
          positionToString(m.from) === fromKey
      )
      .map((m: Move) => positionToString(m.to as Position))
      .sort();

    expect(sharedTargets).toEqual(sandboxTargets);
    expect(sharedTargets).toEqual(backendTargets);
  });

  (skipParityWithOrchestrator ? test.skip : test)(
    'square8: opponent markers and blocking stacks handled consistently',
    () => {
      const boardType: BoardType = 'square8';
      const board = createTestBoard(boardType);
      const from = pos(1, 3);
      const markerPos = pos(2, 3);
      const blockPos = pos(3, 3);
      const player = 1;

      // Attacker height 2, moving east. Path squares: (2,3) then (3,3).
      addStack(board, from, player, 2);
      // Opponent marker on the intermediate square: cannot land there, but the ray continues
      // past it in search of further legal landings (RR-CANON-R091–R092).
      addMarker(board, markerPos, 2);
      // Blocking stack at (3,3): stacks block movement rays and cannot be used as landing
      // cells for non-capture movement (RR-CANON-R091). The ray must stop here.
      addStack(board, blockPos, 1, 1);

      const view = makeMovementView(boardType, board);
      const fromKey = positionToString(from);
      const markerKey = positionToString(markerPos);
      const blockKey = positionToString(blockPos);

      const sharedTargets = enumerateSimpleMoveTargetsFromStack(boardType, from, player, view).map(
        (t) => positionToString(t.to)
      );

      const sandboxTargets = enumerateSimpleMovementLandings(
        boardType,
        board,
        player,
        (p: Position) => isValidPositionForBoard(boardType, board, p)
      )
        .filter((m) => m.fromKey === fromKey)
        .map((m) => positionToString(m.to));

      const bm = new BoardManager(boardType);
      const engine = new RuleEngine(bm as any, boardType as any);
      const state = createTestGameState({
        boardType,
        board,
        currentPlayer: player,
        currentPhase: 'movement',
      });

      const backendTargets = engine
        .getValidMoves(state)
        .filter(
          (m: Move) =>
            (m.type === 'move_stack' || m.type === 'move_ring') &&
            m.from &&
            positionToString(m.from) === fromKey
        )
        .map((m: Move) => positionToString(m.to as Position));

      // Marker cell must not be a landing target in any engine.
      expect(sharedTargets).not.toContain(markerKey);
      expect(sandboxTargets).not.toContain(markerKey);
      expect(backendTargets).not.toContain(markerKey);

      // Blocking stack cell must also not be a landing target in any engine; stacks
      // are hard obstacles for non-capture movement (RR-CANON-R091).
      expect(sharedTargets).not.toContain(blockKey);
      expect(sandboxTargets).not.toContain(blockKey);
      expect(backendTargets).not.toContain(blockKey);

      // Full parity check for this scenario.
      expect(sharedTargets.sort()).toEqual(sandboxTargets.sort());
      expect(sharedTargets.sort()).toEqual(backendTargets.sort());
    }
  );

  test('hexagonal: shared vs sandbox vs RuleEngine on open board', () => {
    const boardType: BoardType = 'hexagonal';
    const board = createTestBoard(boardType);
    const from: Position = { x: 0, y: 0, z: 0 };
    const player = 1;

    addStack(board, from, player, 1);

    const view = makeMovementView(boardType, board);
    const fromKey = positionToString(from);

    const sharedTargets = enumerateSimpleMoveTargetsFromStack(boardType, from, player, view)
      .map((t) => positionToString(t.to))
      .sort();

    const sandboxTargets = enumerateSimpleMovementLandings(
      boardType,
      board,
      player,
      (p: Position) => isValidPositionForBoard(boardType, board, p)
    )
      .filter((m) => m.fromKey === fromKey)
      .map((m) => positionToString(m.to))
      .sort();

    const bm = new BoardManager(boardType);
    const engine = new RuleEngine(bm as any, boardType as any);
    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'movement',
    });

    const backendTargets = engine
      .getValidMoves(state)
      .filter(
        (m: Move) =>
          (m.type === 'move_stack' || m.type === 'move_ring') &&
          m.from &&
          positionToString(m.from) === fromKey
      )
      .map((m: Move) => positionToString(m.to as Position))
      .sort();

    expect(sharedTargets).toEqual(sandboxTargets);
    expect(sharedTargets).toEqual(backendTargets);
  });
});

describe('hasAnyLegalMoveOrCaptureFromOnBoard shared helper', () => {
  test('returns true when a stack has at least one simple movement target', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(3, 3);
    const player = 1;

    addStack(board, from, player, 2);

    const view = makeMovementView(boardType, board);
    const hasAny = hasAnyLegalMoveOrCaptureFromOnBoard(boardType, from, player, view);

    expect(hasAny).toBe(true);
  });

  test('returns true when a stack has no simple moves but at least one capture', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(3, 3);
    const target = pos(4, 3);
    const landing = pos(5, 3);
    const player = 1;

    const fromKey = positionToString(from);
    const targetKey = positionToString(target);
    const landingKey = positionToString(landing);

    // Collapsed everywhere except along the intended capture ray so that
    // non-capturing movement is blocked but the single capture segment
    // remains legal.
    for (let x = 0; x < board.size; x++) {
      for (let y = 0; y < board.size; y++) {
        const key = positionToString({ x, y });
        if (key === fromKey || key === targetKey || key === landingKey) {
          continue;
        }
        board.collapsedSpaces.set(key, 1);
      }
    }

    // Attacker height 2, target height 1 – capture over (4,3) to (5,3)
    // is legal by the shared capture rules.
    addStack(board, from, player, 2);
    addStack(board, target, 2, 1);

    const view = makeMovementView(boardType, board);

    const simpleTargets = enumerateSimpleMoveTargetsFromStack(boardType, from, player, view);
    expect(simpleTargets).toHaveLength(0);

    const hasAny = hasAnyLegalMoveOrCaptureFromOnBoard(boardType, from, player, view);
    expect(hasAny).toBe(true);
  });

  test('returns false when a stack has no legal moves or captures', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(3, 3);
    const player = 1;

    const fromKey = positionToString(from);

    // Collapse all spaces except the origin so the stack is completely stuck.
    for (let x = 0; x < board.size; x++) {
      for (let y = 0; y < board.size; y++) {
        const key = positionToString({ x, y });
        if (key === fromKey) {
          continue;
        }
        board.collapsedSpaces.set(key, 1);
      }
    }

    addStack(board, from, player, 1);

    const view = makeMovementView(boardType, board);
    const hasAny = hasAnyLegalMoveOrCaptureFromOnBoard(boardType, from, player, view);

    expect(hasAny).toBe(false);
  });
});

// ============================================================================
// Tests for applySimpleMovement and applyCaptureSegment shared helpers
// ============================================================================

import {
  applySimpleMovement,
  applyCaptureSegment,
} from '../../src/shared/engine/movementApplication';

describe('applySimpleMovement shared helper', () => {
  test('moves a stack from source to empty destination', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const to = pos(4, 2);
    const player = 1;

    // Add a height-2 stack at the source
    addStack(board, from, player, 2);

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'movement',
    });

    const result = applySimpleMovement(state, { from, to, player });

    // Stack should be at destination
    const toKey = positionToString(to);
    const fromKey = positionToString(from);
    expect(result.nextState.board.stacks.get(toKey)).toBeDefined();
    expect(result.nextState.board.stacks.get(toKey)?.stackHeight).toBe(2);
    expect(result.nextState.board.stacks.get(toKey)?.controllingPlayer).toBe(player);

    // Source should be empty (no stack)
    expect(result.nextState.board.stacks.has(fromKey)).toBe(false);

    // Departure marker should be at source
    expect(result.nextState.board.markers.has(fromKey)).toBe(true);
    expect(result.nextState.board.markers.get(fromKey)?.player).toBe(player);
  });

  test('leaves no departure marker when leaveDepartureMarker is false', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const to = pos(4, 2);
    const player = 1;

    addStack(board, from, player, 2);

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'movement',
    });

    const result = applySimpleMovement(state, {
      from,
      to,
      player,
      leaveDepartureMarker: false,
    });

    const fromKey = positionToString(from);
    expect(result.nextState.board.markers.has(fromKey)).toBe(false);
  });

  test('merges stacks when landing on existing stack', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const to = pos(4, 2);
    const player = 1;

    // Add moving stack (height 2) and existing stack at destination (height 1)
    addStack(board, from, player, 2);
    addStack(board, to, player, 1);

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'movement',
    });

    const result = applySimpleMovement(state, { from, to, player });

    const toKey = positionToString(to);
    const mergedStack = result.nextState.board.stacks.get(toKey);

    // Merged stack should have height 3 (1 existing + 2 moved, but existing on top)
    expect(mergedStack?.stackHeight).toBe(3);
    expect(mergedStack?.rings.length).toBe(3);
  });

  test('collapses own marker on path to territory', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const to = pos(5, 2);
    const markerPos = pos(3, 2);
    const player = 1;

    addStack(board, from, player, 3);
    addMarker(board, markerPos, player); // Own marker on path

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'movement',
    });

    const result = applySimpleMovement(state, { from, to, player });

    const markerKey = positionToString(markerPos);
    // Marker should be removed
    expect(result.nextState.board.markers.has(markerKey)).toBe(false);
    // Should be collapsed to territory
    expect(result.nextState.board.collapsedSpaces.has(markerKey)).toBe(true);
    expect(result.nextState.board.collapsedSpaces.get(markerKey)).toBe(player);
  });

  test('flips opponent marker on path to own color', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const to = pos(5, 2);
    const markerPos = pos(3, 2);
    const player = 1;
    const opponent = 2;

    addStack(board, from, player, 3);
    addMarker(board, markerPos, opponent); // Opponent marker on path

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'movement',
    });

    const result = applySimpleMovement(state, { from, to, player });

    const markerKey = positionToString(markerPos);
    // Marker should still exist but be flipped to player's color
    expect(result.nextState.board.markers.has(markerKey)).toBe(true);
    expect(result.nextState.board.markers.get(markerKey)?.player).toBe(player);
  });

  test('eliminates top ring when landing on own marker', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const to = pos(4, 2);
    const player = 1;

    addStack(board, from, player, 2);
    addMarker(board, to, player); // Own marker at destination

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'movement',
    });

    const result = applySimpleMovement(state, { from, to, player });

    const toKey = positionToString(to);
    // Stack should have height 1 (one ring eliminated)
    expect(result.nextState.board.stacks.get(toKey)?.stackHeight).toBe(1);
    // Marker should be removed
    expect(result.nextState.board.markers.has(toKey)).toBe(false);
    // Elimination should be tracked
    expect(result.eliminatedRingsByPlayer).toEqual({ [player]: 1 });
    expect(result.nextState.board.eliminatedRings[player]).toBe(1);
  });

  test('throws error when no stack at source', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const to = pos(4, 2);
    const player = 1;

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'movement',
    });

    expect(() => applySimpleMovement(state, { from, to, player })).toThrow();
  });
});

describe('applyCaptureSegment shared helper', () => {
  test('captures top ring from target and adds to attacker', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const target = pos(3, 2);
    const landing = pos(4, 2);
    const player = 1;
    const opponent = 2;

    // Attacker height 2, target height 1
    addStack(board, from, player, 2);
    addStack(board, target, opponent, 1);

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'capture',
    });

    const result = applyCaptureSegment(state, { from, target, landing, player });

    const landingKey = positionToString(landing);
    const targetKey = positionToString(target);
    const fromKey = positionToString(from);

    // Attacker should be at landing with captured ring added to bottom
    const landedStack = result.nextState.board.stacks.get(landingKey);
    expect(landedStack).toBeDefined();
    expect(landedStack?.stackHeight).toBe(3); // 2 + 1 captured
    expect(landedStack?.rings[0]).toBe(player); // Attacker on top
    expect(landedStack?.rings[2]).toBe(opponent); // Captured at bottom

    // Target stack should be removed (was height 1)
    expect(result.nextState.board.stacks.has(targetKey)).toBe(false);

    // Source should be empty
    expect(result.nextState.board.stacks.has(fromKey)).toBe(false);

    // Departure marker at source
    expect(result.nextState.board.markers.has(fromKey)).toBe(true);
  });

  test('leaves remaining rings at target when target has multiple rings', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const target = pos(3, 2);
    const landing = pos(4, 2);
    const player = 1;
    const opponent = 2;

    // Attacker height 2, target height 3
    addStack(board, from, player, 2);
    addStack(board, target, opponent, 3);

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'capture',
    });

    const result = applyCaptureSegment(state, { from, target, landing, player });

    const targetKey = positionToString(target);

    // Target should have 2 remaining rings
    const remainingTarget = result.nextState.board.stacks.get(targetKey);
    expect(remainingTarget).toBeDefined();
    expect(remainingTarget?.stackHeight).toBe(2);
  });

  test('eliminates top ring when capture lands on own marker', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const target = pos(3, 2);
    const landing = pos(4, 2);
    const player = 1;
    const opponent = 2;

    addStack(board, from, player, 2);
    addStack(board, target, opponent, 1);
    addMarker(board, landing, player); // Own marker at landing

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'capture',
    });

    const result = applyCaptureSegment(state, { from, target, landing, player });

    const landingKey = positionToString(landing);

    // Stack should have height 2 (3 - 1 eliminated)
    expect(result.nextState.board.stacks.get(landingKey)?.stackHeight).toBe(2);
    // Marker should be removed
    expect(result.nextState.board.markers.has(landingKey)).toBe(false);
    // Elimination tracked
    expect(result.eliminatedRingsByPlayer).toEqual({ [player]: 1 });
  });

  test('processes markers along both path legs', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(1, 2);
    const target = pos(3, 2);
    const landing = pos(5, 2);
    const player = 1;
    const opponent = 2;

    addStack(board, from, player, 4);
    addStack(board, target, opponent, 1);

    // Marker on first leg (from->target path)
    addMarker(board, pos(2, 2), player);
    // Marker on second leg (target->landing path)
    addMarker(board, pos(4, 2), opponent);

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'capture',
    });

    const result = applyCaptureSegment(state, { from, target, landing, player });

    // First leg marker should be collapsed
    expect(result.nextState.board.collapsedSpaces.has(positionToString(pos(2, 2)))).toBe(true);
    // Second leg opponent marker should be flipped
    expect(result.nextState.board.markers.get(positionToString(pos(4, 2)))?.player).toBe(player);
  });

  test('throws error when no attacker at source', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const target = pos(3, 2);
    const landing = pos(4, 2);
    const player = 1;

    // Only target stack, no attacker
    addStack(board, target, 2, 1);

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'capture',
    });

    expect(() => applyCaptureSegment(state, { from, target, landing, player })).toThrow();
  });

  test('throws error when no target stack', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const target = pos(3, 2);
    const landing = pos(4, 2);
    const player = 1;

    // Only attacker, no target
    addStack(board, from, player, 2);

    const state = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'capture',
    });

    expect(() => applyCaptureSegment(state, { from, target, landing, player })).toThrow();
  });
});
