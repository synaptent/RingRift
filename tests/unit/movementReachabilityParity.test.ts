import { BoardType, BoardState, Position } from '../../src/shared/types/game';
import { hasAnyLegalMoveOrCaptureFromOnBoard, MovementBoardView } from '../../src/shared/engine/core';
import { createTestBoard, addStack, pos } from '../utils/fixtures';

function makeMovementView(boardType: BoardType, board: BoardState): MovementBoardView {
  return {
    isValidPosition: (p: Position) => {
      if (boardType === 'hexagonal') {
        const radius = board.size - 1;
        const x = p.x;
        const y = p.y;
        const z = p.z !== undefined ? p.z : -x - y;
        const dist = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
        return dist <= radius;
      }
      return p.x >= 0 && p.x < board.size && p.y >= 0 && p.y < board.size;
    },
    isCollapsedSpace: (p: Position) => {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      return board.collapsedSpaces.has(key);
    },
    getStackAt: (p: Position) => {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      const stack = board.stacks.get(key);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight
      };
    },
    getMarkerOwner: () => undefined
  };
}

describe('hasAnyLegalMoveOrCaptureFromOnBoard', () => {
  test('square8: detects movement when a simple non-capture move exists', () => {
    const board = createTestBoard('square8');
    const from = pos(3, 3);
    addStack(board, from, 1, 2); // height 2, can move at least distance 2

    const view = makeMovementView('square8', board);
    const result = hasAnyLegalMoveOrCaptureFromOnBoard('square8', from, 1, view);
    expect(result).toBe(true);
  });

  test('square8: returns false when stack is completely surrounded by collapsed spaces', () => {
    const board = createTestBoard('square8');
    const from = pos(3, 3);
    addStack(board, from, 1, 1);

    // Surround with collapsed spaces
    const deltas = [
      [-1, -1], [-1, 0], [-1, 1],
      [0, -1],           [0, 1],
      [1, -1],  [1, 0],  [1, 1]
    ];
    for (const [dx, dy] of deltas) {
      const p = pos(from.x + dx, from.y + dy);
      const key = `${p.x},${p.y}`;
      board.collapsedSpaces.set(key, 2);
    }

    const view = makeMovementView('square8', board);
    const result = hasAnyLegalMoveOrCaptureFromOnBoard('square8', from, 1, view);
    expect(result).toBe(false);
  });

  test('hexagonal: detects movement on an otherwise empty board', () => {
    const board = createTestBoard('hexagonal');
    const from: Position = { x: 0, y: 0, z: 0 };
    addStack(board, from, 1, 1);

    const view = makeMovementView('hexagonal', board);
    const result = hasAnyLegalMoveOrCaptureFromOnBoard('hexagonal', from, 1, view);
    expect(result).toBe(true);
  });
});
