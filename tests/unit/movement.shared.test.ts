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

  test('square8: opponent markers and blocking stacks handled consistently', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(1, 3);
    const markerPos = pos(2, 3);
    const blockPos = pos(3, 3);
    const player = 1;

    // Attacker height 2, moving east. Path squares: (2,3) then (3,3).
    addStack(board, from, player, 2);
    // Opponent marker on the intermediate square: cannot land, but ray continues.
    addMarker(board, markerPos, 2);
    // Blocking stack at (3,3): legal landing (merge), but ray must stop there.
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

    // Blocking stack cell must be a landing target (merge) in all engines.
    expect(sharedTargets).toContain(blockKey);
    expect(sandboxTargets).toContain(blockKey);
    expect(backendTargets).toContain(blockKey);

    // Full parity check for this scenario.
    expect(sharedTargets.sort()).toEqual(sandboxTargets.sort());
    expect(sharedTargets.sort()).toEqual(backendTargets.sort());
  });

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
