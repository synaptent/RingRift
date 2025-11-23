import {
  BoardType,
  LineInfo,
  Position,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';
import { findAllLines, findLinesForPlayer } from '../../src/shared/engine/lineDetection';
import {
  createTestBoard,
  createMarkerLine,
  addStack,
  addCollapsedSpace,
  pos,
} from '../utils/fixtures';

describe('shared lineDetection.findAllLines / findLinesForPlayer', () => {
  test('square8: detects simple horizontal line and respects minimum length', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const requiredLength = BOARD_CONFIGS[boardType].lineLength;
    const start = pos(0, 0);

    createMarkerLine(board, start, { dx: 1, dy: 0 }, requiredLength, 1);

    const lines = findAllLines(board);
    expect(lines).toHaveLength(1);
    const line = lines[0];
    expect(line.player).toBe(1);
    expect(line.length).toBe(requiredLength);

    const keys = line.positions.map((p) => positionToString(p)).sort();
    const expectedKeys = Array.from({ length: requiredLength }, (_, i) =>
      positionToString(pos(i, 0))
    ).sort();

    expect(keys).toEqual(expectedKeys);
  });

  test('square8: stacks and collapsed spaces break otherwise-valid lines', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const requiredLength = BOARD_CONFIGS[boardType].lineLength;
    const start = pos(0, 1);

    createMarkerLine(board, start, { dx: 1, dy: 0 }, requiredLength, 1);

    // Put a stack on the middle marker
    const middle = pos(1, 1);
    addStack(board, middle, 1, 1);

    // And mark one end as collapsed territory
    const collapsedEnd = pos(requiredLength - 1, 1);
    addCollapsedSpace(board, collapsedEnd, 1);

    const lines = findAllLines(board);
    expect(lines).toHaveLength(0);
  });

  test('square8: overlapping and perpendicular lines are all reported', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const requiredLength = BOARD_CONFIGS[boardType].lineLength;

    // Horizontal and vertical lines crossing at the center file/rank.
    createMarkerLine(board, pos(1, 3), { dx: 1, dy: 0 }, requiredLength, 1);
    createMarkerLine(board, pos(3, 1), { dx: 0, dy: 1 }, requiredLength, 1);

    const lines = findAllLines(board);
    expect(lines.length).toBeGreaterThanOrEqual(2);

    const canonical = (l: LineInfo) =>
      l.positions
        .map((p) => positionToString(p))
        .sort()
        .join('|');
    const keys = lines.map(canonical);

    const horizontalKey = Array.from({ length: requiredLength }, (_, i) =>
      positionToString(pos(1 + i, 3))
    )
      .sort()
      .join('|');
    const verticalKey = Array.from({ length: requiredLength }, (_, i) =>
      positionToString(pos(3, 1 + i))
    )
      .sort()
      .join('|');

    expect(keys).toContain(horizontalKey);
    expect(keys).toContain(verticalKey);
  });

  test('hexagonal: detects straight line along a cube axis', () => {
    const boardType: BoardType = 'hexagonal';
    const board = createTestBoard(boardType);
    const requiredLength = BOARD_CONFIGS[boardType].lineLength;
    const start: Position = pos(0, 0, 0);

    createMarkerLine(board, start, { dx: 1, dy: 0, dz: -1 }, requiredLength, 2);

    const lines = findAllLines(board);
    expect(lines).toHaveLength(1);
    const line = lines[0];
    expect(line.player).toBe(2);
    expect(line.length).toBe(requiredLength);
  });

  test('findLinesForPlayer filters lines by owner', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const requiredLength = BOARD_CONFIGS[boardType].lineLength;

    createMarkerLine(board, pos(0, 0), { dx: 1, dy: 0 }, requiredLength, 1);
    createMarkerLine(board, pos(0, 2), { dx: 1, dy: 0 }, requiredLength, 2);

    const allLines = findAllLines(board);
    expect(allLines.length).toBe(2);

    const p1Lines = findLinesForPlayer(board, 1);
    const p2Lines = findLinesForPlayer(board, 2);

    expect(p1Lines.length).toBe(1);
    expect(p2Lines.length).toBe(1);
    expect(p1Lines[0].player).toBe(1);
    expect(p2Lines[0].player).toBe(2);
  });
});
