import {
  BoardType,
  BoardState,
  BOARD_CONFIGS,
  positionToString,
  stringToPosition,
  LineInfo,
} from '../../src/shared/types/game';
import { BoardManager } from '../../src/server/game/BoardManager';
import { findAllLinesOnBoard } from '../../src/client/sandbox/sandboxLines';

function createEmptyBoard(boardType: BoardType): BoardState {
  const manager = new BoardManager(boardType);
  return manager.createBoard();
}

function addMarker(board: BoardState, x: number, y: number, player: number): void {
  const key = positionToString({ x, y });
  board.markers.set(key, { player, position: { x, y }, type: 'regular' });
}

function addStack(board: BoardState, x: number, y: number, player: number, height = 1): void {
  const key = positionToString({ x, y });
  board.stacks.set(key, {
    position: { x, y },
    rings: new Array(height).fill(player),
    stackHeight: height,
    capHeight: height,
    controllingPlayer: player,
  });
}

function canonicalLineKeys(lines: LineInfo[]): string[] {
  return lines
    .map((line) =>
      line.positions
        .map((p) => positionToString(p))
        .sort()
        .join('|')
    )
    .sort();
}

function backendLines(boardType: BoardType, board: BoardState): string[] {
  const manager = new BoardManager(boardType);

  // Single source of truth for backend debug + comparison: delegate to
  // BoardManager.debugFindAllLines, which internally uses
  // BoardManager.findAllLines with the same semantics the engine uses
  // during real games.
  const debug = manager.debugFindAllLines(board);
  // eslint-disable-next-line no-console
  console.log('[LineDetectionParity.backendLines]', {
    boardType,
    keys: debug.keys,
  });

  // The parity comparison works over canonical, order-independent line
  // keys, so we can return the debug keys directly.
  return debug.keys;
}

function sandboxLines(boardType: BoardType, board: BoardState): string[] {
  const cfg = BOARD_CONFIGS[boardType];

  const isValidPosition = (pos: { x: number; y: number; z?: number }): boolean => {
    if (boardType === 'hexagonal') {
      const radius = cfg.size - 1;
      const x = pos.x;
      const y = pos.y;
      const z = pos.z !== undefined ? pos.z : -x - y;
      const distance = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
      return distance <= radius;
    }
    return pos.x >= 0 && pos.x < cfg.size && pos.y >= 0 && pos.y < cfg.size;
  };

  const lines = findAllLinesOnBoard(boardType, board, isValidPosition, stringToPosition);
  return canonicalLineKeys(lines);
}

describe('Backend vs sandbox line detection parity (rules-level)', () => {
  const boardType: BoardType = 'square8';

  it('produces no lines when there are no markers', () => {
    const board = createEmptyBoard(boardType);

    const backend = backendLines(boardType, board);
    const sandbox = sandboxLines(boardType, board);

    expect(backend).toEqual([]);
    expect(sandbox).toEqual([]);
  });

  it('detects the same simple horizontal line', () => {
    const board = createEmptyBoard(boardType);
    // P1 markers on a1–d1 (x increasing along the bottom rank).
    addMarker(board, 0, 7, 1);
    addMarker(board, 1, 7, 1);
    addMarker(board, 2, 7, 1);
    addMarker(board, 3, 7, 1);

    const backend = backendLines(boardType, board);
    const sandbox = sandboxLines(boardType, board);

    expect(backend).toEqual(sandbox);
    expect(backend.length).toBe(1);
  });

  it('ignores would-be lines that are broken by a stack or collapsed space', () => {
    const board = createEmptyBoard(boardType);
    // Potential horizontal line a1–d1 but with a stack on c1, which should break the line.
    addMarker(board, 0, 7, 1);
    addMarker(board, 1, 7, 1);
    addMarker(board, 2, 7, 1);
    addMarker(board, 3, 7, 1);

    // Place a stack on c1.
    addStack(board, 2, 7, 1, 1);

    const backend = backendLines(boardType, board);
    const sandbox = sandboxLines(boardType, board);

    expect(backend).toEqual([]);
    expect(sandbox).toEqual([]);
  });

  it('treats mixed-player markers correctly (only same-color contiguous markers form a line)', () => {
    const board = createEmptyBoard(boardType);
    // P1 markers at a1, b1, c1; P2 marker at d1 – breaks the P1 line.
    addMarker(board, 0, 7, 1);
    addMarker(board, 1, 7, 1);
    addMarker(board, 2, 7, 1);
    addMarker(board, 3, 7, 2);

    const backend = backendLines(boardType, board);
    const sandbox = sandboxLines(boardType, board);

    expect(backend).toEqual([]);
    expect(sandbox).toEqual([]);
  });
});
