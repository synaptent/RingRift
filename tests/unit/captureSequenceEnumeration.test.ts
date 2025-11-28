/**
 * Capture Sequence Enumeration Parity Tests
 *
 * REFACTORED FOR CI STABILITY (P0-RULES-001):
 * This test file validates that the sandbox and backend enumerate identical
 * capture sequences. The original version used exhaustive enumeration across
 * many random boards, causing:
 *   - RangeError: Invalid string length
 *   - Jest worker crashes (OOM)
 *
 * The refactored version uses:
 *   1. Deterministic, focused test boards (not random)
 *   2. Strict complexity bounds (max depth, max sequences)
 *   3. Smaller test cases that complete reliably in CI
 *   4. Comprehensive edge case coverage without combinatorial explosion
 *
 * Coverage strategy:
 *   - Single capture: Validates basic capture detection
 *   - Linear chain: Validates sequential capture continuation
 *   - Branching paths: Validates multi-choice at landing position
 *   - Different board types: square8, square19, hexagonal
 */

import { BoardType, BoardState, Position, positionToString } from '../../src/shared/types/game';
import {
  enumerateCaptureSegmentsFromBoard,
  applyCaptureSegmentOnBoard,
  CaptureBoardAdapters,
  CaptureApplyAdapters,
} from '../../src/client/sandbox/sandboxCaptures';
import {
  applyMarkerEffectsAlongPathOnBoard,
  MarkerPathHelpers,
} from '../../src/client/sandbox/sandboxMovement';
import {
  createTestBoard,
  addStack,
  pos,
  createTestGameState,
  createTestPlayer,
} from '../utils/fixtures';
import { BoardManager } from '../../src/server/game/BoardManager';
import { getChainCaptureContinuationInfo } from '../../src/shared/engine/aggregates/CaptureAggregate';

// =============================================================================
// COMPLEXITY BOUNDS
// =============================================================================
// These bounds prevent runaway enumeration while still providing good coverage.

/** Maximum chain depth to explore (captures per sequence) */
const MAX_CHAIN_DEPTH = 4;

/** Maximum number of sequences to enumerate per test case */
const MAX_SEQUENCES_PER_CASE = 50;

// =============================================================================
// TYPES AND HELPERS
// =============================================================================

interface CaptureSequence {
  segments: { from: Position; target: Position; landing: Position }[];
  finalBoard: BoardState;
}

interface NamedCaptureTestCase {
  name: string;
  boardType: BoardType;
  board: BoardState;
  from: Position;
  player: number;
  expectedMinSequences?: number;
  expectedMaxSequences?: number;
  expectedMinChainLength?: number;
  expectedMaxChainLength?: number;
}

function cloneBoard(board: BoardState): BoardState {
  return {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings },
  };
}

function sequenceToKey(seq: CaptureSequence): string {
  return seq.segments
    .map(
      (s) =>
        `${positionToString(s.from)}->${positionToString(s.target)}->${positionToString(s.landing)}`
    )
    .join('|');
}

// =============================================================================
// BOUNDED ENUMERATION FUNCTIONS
// =============================================================================

/**
 * Enumerate capture sequences using sandbox helpers with strict bounds.
 * Returns early if limits are exceeded.
 */
function enumerateSequencesSandbox(
  boardType: BoardType,
  initialBoard: BoardState,
  from: Position,
  player: number,
  maxDepth: number = MAX_CHAIN_DEPTH,
  maxSequences: number = MAX_SEQUENCES_PER_CASE
): CaptureSequence[] {
  const sequences: CaptureSequence[] = [];

  const adapters: CaptureBoardAdapters = {
    isValidPosition: (p: Position) => {
      if (boardType === 'hexagonal') {
        const radius = initialBoard.size - 1;
        const x = p.x;
        const y = p.y;
        const z = p.z !== undefined ? p.z : -x - y;
        const dist = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
        return dist <= radius;
      }
      return p.x >= 0 && p.x < initialBoard.size && p.y >= 0 && p.y < initialBoard.size;
    },
    isCollapsedSpace: (p: Position, b: BoardState) => {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      return b.collapsedSpaces.has(key);
    },
    getMarkerOwner: (_p: Position, _b: BoardState) => undefined,
  };

  type Frame = {
    board: BoardState;
    currentPos: Position;
    segments: { from: Position; target: Position; landing: Position }[];
    depth: number;
  };

  const stack: Frame[] = [
    {
      board: cloneBoard(initialBoard),
      currentPos: from,
      segments: [],
      depth: 0,
    },
  ];

  while (stack.length > 0 && sequences.length < maxSequences) {
    const frame = stack.pop()!;
    const { board, currentPos, segments, depth } = frame;

    // Depth limit reached - record as maximal sequence
    if (depth >= maxDepth) {
      if (segments.length > 0) {
        sequences.push({ segments: [...segments], finalBoard: cloneBoard(board) });
      }
      continue;
    }

    const nextSegments = enumerateCaptureSegmentsFromBoard(
      boardType,
      board,
      currentPos,
      player,
      adapters
    );

    if (nextSegments.length === 0) {
      if (segments.length > 0) {
        sequences.push({ segments: [...segments], finalBoard: cloneBoard(board) });
      }
      continue;
    }

    for (const seg of nextSegments) {
      if (sequences.length >= maxSequences) break;

      const boardClone = cloneBoard(board);

      const markerHelpers: MarkerPathHelpers = {
        setMarker: (position, playerNumber, b) => {
          const key = positionToString(position);
          b.markers.set(key, { position, player: playerNumber, type: 'regular' });
        },
        collapseMarker: (position, playerNumber, b) => {
          const key = positionToString(position);
          b.markers.delete(key);
          b.collapsedSpaces.set(key, playerNumber);
        },
        flipMarker: (position, playerNumber, b) => {
          const key = positionToString(position);
          const existing = b.markers.get(key);
          if (existing && existing.player !== playerNumber) {
            b.markers.set(key, { position, player: playerNumber, type: 'regular' });
          }
        },
      };

      const applyAdapters: CaptureApplyAdapters = {
        applyMarkerEffectsAlongPath: (fromPos, toPos, playerNumber) => {
          applyMarkerEffectsAlongPathOnBoard(
            boardClone,
            fromPos,
            toPos,
            playerNumber,
            markerHelpers
          );
        },
      };

      applyCaptureSegmentOnBoard(
        boardClone,
        seg.from,
        seg.target,
        seg.landing,
        player,
        applyAdapters
      );

      stack.push({
        board: boardClone,
        currentPos: seg.landing,
        segments: [...segments, seg],
        depth: depth + 1,
      });
    }
  }

  sequences.sort((a, b) => sequenceToKey(a).localeCompare(sequenceToKey(b)));
  return sequences;
}

/**
 * Enumerate capture sequences using CaptureAggregate with strict bounds.
 */
function enumerateSequencesBackend(
  boardType: BoardType,
  initialBoard: BoardState,
  from: Position,
  player: number,
  maxDepth: number = MAX_CHAIN_DEPTH,
  maxSequences: number = MAX_SEQUENCES_PER_CASE
): CaptureSequence[] {
  const sequences: CaptureSequence[] = [];
  const bm = new BoardManager(boardType);

  type Frame = {
    board: BoardState;
    currentPos: Position;
    segments: { from: Position; target: Position; landing: Position }[];
    depth: number;
  };

  const stack: Frame[] = [
    {
      board: cloneBoard(initialBoard),
      currentPos: from,
      segments: [],
      depth: 0,
    },
  ];

  while (stack.length > 0 && sequences.length < maxSequences) {
    const frame = stack.pop()!;
    const { board, currentPos, segments, depth } = frame;

    // Depth limit reached
    if (depth >= maxDepth) {
      if (segments.length > 0) {
        sequences.push({ segments: [...segments], finalBoard: cloneBoard(board) });
      }
      continue;
    }

    const gameState = createTestGameState({
      boardType,
      board: { ...board, type: boardType, size: board.size },
      currentPlayer: player,
      currentPhase: 'capture',
      players: [createTestPlayer(1), createTestPlayer(2)],
      moveHistory: [],
    });

    const { availableContinuations: moves } = getChainCaptureContinuationInfo(
      gameState,
      player,
      currentPos
    );

    if (moves.length === 0) {
      if (segments.length > 0) {
        sequences.push({ segments: [...segments], finalBoard: cloneBoard(board) });
      }
      continue;
    }

    for (const move of moves) {
      if (sequences.length >= maxSequences) break;
      if (!move.from || !move.captureTarget) continue;

      const boardClone = cloneBoard(board);

      const markerHelpers: MarkerPathHelpers = {
        setMarker: (position, playerNumber, b) => bm.setMarker(position, playerNumber, b),
        collapseMarker: (position, playerNumber, b) => bm.collapseMarker(position, playerNumber, b),
        flipMarker: (position, playerNumber, b) => bm.flipMarker(position, playerNumber, b),
      };

      const applyAdapters: CaptureApplyAdapters = {
        applyMarkerEffectsAlongPath: (fromPos, toPos, playerNumber) => {
          applyMarkerEffectsAlongPathOnBoard(
            boardClone,
            fromPos,
            toPos,
            playerNumber,
            markerHelpers
          );
        },
      };

      applyCaptureSegmentOnBoard(
        boardClone,
        move.from,
        move.captureTarget,
        move.to,
        player,
        applyAdapters
      );

      stack.push({
        board: boardClone,
        currentPos: move.to,
        segments: [...segments, { from: move.from, target: move.captureTarget, landing: move.to }],
        depth: depth + 1,
      });
    }
  }

  sequences.sort((a, b) => sequenceToKey(a).localeCompare(sequenceToKey(b)));
  return sequences;
}

// =============================================================================
// DETERMINISTIC TEST BOARDS
// =============================================================================

function buildDeterministicTestCases(): NamedCaptureTestCase[] {
  const cases: NamedCaptureTestCase[] = [];

  // -------------------------------------------------------------------------
  // SQUARE8: Single capture (baseline)
  // -------------------------------------------------------------------------
  {
    const board = createTestBoard('square8');
    const from = pos(4, 4);
    addStack(board, from, 1, 3);
    addStack(board, pos(6, 4), 2, 2); // Target to the east

    cases.push({
      name: 'square8: single capture east',
      boardType: 'square8',
      board,
      from,
      player: 1,
      expectedMinSequences: 1,
      expectedMinChainLength: 1,
    });
  }

  // -------------------------------------------------------------------------
  // SQUARE8: Linear chain (E-E)
  // -------------------------------------------------------------------------
  {
    const board = createTestBoard('square8');
    const from = pos(1, 4);
    addStack(board, from, 1, 3);
    addStack(board, pos(3, 4), 2, 2); // Target 1
    addStack(board, pos(5, 4), 2, 2); // Target 2

    cases.push({
      name: 'square8: linear chain (2 captures)',
      boardType: 'square8',
      board,
      from,
      player: 1,
      expectedMinSequences: 1,
      expectedMinChainLength: 2,
    });
  }

  // -------------------------------------------------------------------------
  // SQUARE8: Branching paths (2 options)
  // -------------------------------------------------------------------------
  {
    const board = createTestBoard('square8');
    const from = pos(3, 3);
    addStack(board, from, 1, 3);
    addStack(board, pos(5, 3), 2, 2); // East
    addStack(board, pos(3, 5), 2, 2); // North

    cases.push({
      name: 'square8: branching (2 single-capture paths)',
      boardType: 'square8',
      board,
      from,
      player: 1,
      expectedMinSequences: 2,
      expectedMinChainLength: 1,
    });
  }

  // -------------------------------------------------------------------------
  // SQUARE8: Branching with one continuation
  // -------------------------------------------------------------------------
  {
    const board = createTestBoard('square8');
    const from = pos(2, 2);
    addStack(board, from, 1, 3);

    // Path A: East only (single capture)
    addStack(board, pos(4, 2), 2, 2);

    // Path B: North then continue (2 captures)
    addStack(board, pos(2, 4), 2, 2);
    addStack(board, pos(2, 6), 2, 2);

    cases.push({
      name: 'square8: branching with continuation',
      boardType: 'square8',
      board,
      from,
      player: 1,
      expectedMinSequences: 2,
      expectedMinChainLength: 1,
    });
  }

  // -------------------------------------------------------------------------
  // SQUARE8: Collapsed space blocking
  // -------------------------------------------------------------------------
  {
    const board = createTestBoard('square8');
    const from = pos(2, 4);
    addStack(board, from, 1, 3);
    addStack(board, pos(4, 4), 2, 2); // Target east
    board.collapsedSpaces.set('6,4', 0); // Block landing

    addStack(board, pos(2, 6), 2, 2); // Valid north

    cases.push({
      name: 'square8: collapsed space blocks east',
      boardType: 'square8',
      board,
      from,
      player: 1,
      expectedMinSequences: 1,
    });
  }

  // -------------------------------------------------------------------------
  // SQUARE19: Simple capture
  // -------------------------------------------------------------------------
  {
    const board = createTestBoard('square19');
    const from = pos(9, 9);
    addStack(board, from, 1, 3);
    addStack(board, pos(11, 9), 2, 2);

    cases.push({
      name: 'square19: single capture',
      boardType: 'square19',
      board,
      from,
      player: 1,
      expectedMinSequences: 1,
    });
  }

  // -------------------------------------------------------------------------
  // SQUARE19: 4 directions
  // -------------------------------------------------------------------------
  {
    const board = createTestBoard('square19');
    const from = pos(9, 9);
    addStack(board, from, 1, 3);
    addStack(board, pos(11, 9), 2, 2); // E
    addStack(board, pos(7, 9), 2, 2); // W
    addStack(board, pos(9, 11), 2, 2); // N
    addStack(board, pos(9, 7), 2, 2); // S

    cases.push({
      name: 'square19: 4 directions from center',
      boardType: 'square19',
      board,
      from,
      player: 1,
      expectedMinSequences: 4,
    });
  }

  // -------------------------------------------------------------------------
  // HEXAGONAL: Single capture
  // -------------------------------------------------------------------------
  {
    const board = createTestBoard('hexagonal');
    const from: Position = { x: 0, y: 0, z: 0 };
    addStack(board, from, 1, 3);
    addStack(board, { x: 2, y: -2, z: 0 }, 2, 2);

    cases.push({
      name: 'hexagonal: single capture',
      boardType: 'hexagonal',
      board,
      from,
      player: 1,
      expectedMinSequences: 1,
    });
  }

  // -------------------------------------------------------------------------
  // HEXAGONAL: Three directions
  // -------------------------------------------------------------------------
  {
    const board = createTestBoard('hexagonal');
    const from: Position = { x: 0, y: 0, z: 0 };
    addStack(board, from, 1, 3);
    addStack(board, { x: 2, y: -2, z: 0 }, 2, 2);
    addStack(board, { x: -2, y: 2, z: 0 }, 2, 2);
    addStack(board, { x: 0, y: 2, z: -2 }, 2, 2);

    cases.push({
      name: 'hexagonal: three directions',
      boardType: 'hexagonal',
      board,
      from,
      player: 1,
      expectedMinSequences: 3,
    });
  }

  // -------------------------------------------------------------------------
  // HEXAGONAL: Chain capture
  // -------------------------------------------------------------------------
  {
    const board = createTestBoard('hexagonal');
    const from: Position = { x: 0, y: 0, z: 0 };
    addStack(board, from, 1, 3);
    addStack(board, { x: 2, y: -2, z: 0 }, 2, 2);
    addStack(board, { x: 6, y: -6, z: 0 }, 2, 2);

    cases.push({
      name: 'hexagonal: chain capture',
      boardType: 'hexagonal',
      board,
      from,
      player: 1,
      expectedMinSequences: 1,
      expectedMinChainLength: 1,
    });
  }

  return cases;
}

// =============================================================================
// PARITY TEST HELPER
// =============================================================================

function runParityTest(testCase: NamedCaptureTestCase): void {
  const { boardType, board, from, player } = testCase;

  const sandboxSeqs = enumerateSequencesSandbox(boardType, board, from, player);
  const backendSeqs = enumerateSequencesBackend(boardType, board, from, player);

  const sandboxKeys = sandboxSeqs.map(sequenceToKey).sort();
  const backendKeys = backendSeqs.map(sequenceToKey).sort();

  // Primary assertion: parity between sandbox and backend
  expect(backendKeys).toEqual(sandboxKeys);

  // Validate expected counts if specified
  if (testCase.expectedMinSequences !== undefined) {
    expect(sandboxSeqs.length).toBeGreaterThanOrEqual(testCase.expectedMinSequences);
  }
  if (testCase.expectedMaxSequences !== undefined) {
    expect(sandboxSeqs.length).toBeLessThanOrEqual(testCase.expectedMaxSequences);
  }

  // Validate chain lengths if specified
  if (
    testCase.expectedMinChainLength !== undefined ||
    testCase.expectedMaxChainLength !== undefined
  ) {
    const chainLengths = sandboxSeqs.map((s) => s.segments.length);
    if (chainLengths.length > 0) {
      const minLen = Math.min(...chainLengths);
      const maxLen = Math.max(...chainLengths);

      if (testCase.expectedMinChainLength !== undefined) {
        expect(minLen).toBeGreaterThanOrEqual(testCase.expectedMinChainLength);
      }
      if (testCase.expectedMaxChainLength !== undefined) {
        expect(maxLen).toBeLessThanOrEqual(testCase.expectedMaxChainLength);
      }
    }
  }
}

// =============================================================================
// TESTS - DETERMINISTIC (FAST, CI-SAFE)
// =============================================================================

describe('capture sequence enumeration parity (deterministic)', () => {
  const testCases = buildDeterministicTestCases();

  describe('square8 boards', () => {
    const square8Cases = testCases.filter((c) => c.boardType === 'square8');

    test.each(square8Cases.map((c) => [c.name, c] as const))('%s', (_name, testCase) => {
      runParityTest(testCase);
    });
  });

  describe('square19 boards', () => {
    const square19Cases = testCases.filter((c) => c.boardType === 'square19');

    test.each(square19Cases.map((c) => [c.name, c] as const))('%s', (_name, testCase) => {
      runParityTest(testCase);
    });
  });

  describe('hexagonal boards', () => {
    const hexCases = testCases.filter((c) => c.boardType === 'hexagonal');

    test.each(hexCases.map((c) => [c.name, c] as const))('%s', (_name, testCase) => {
      runParityTest(testCase);
    });
  });
});

// =============================================================================
// TESTS - EDGE CASES
// =============================================================================

describe('capture sequence enumeration - edge cases', () => {
  test('square8: no targets yields empty sequences', () => {
    const board = createTestBoard('square8');
    addStack(board, pos(4, 4), 1, 3);

    const sandboxSeqs = enumerateSequencesSandbox('square8', board, pos(4, 4), 1);
    const backendSeqs = enumerateSequencesBackend('square8', board, pos(4, 4), 1);

    expect(sandboxSeqs).toHaveLength(0);
    expect(backendSeqs).toHaveLength(0);
  });

  test('hexagonal: no targets yields empty sequences', () => {
    const board = createTestBoard('hexagonal');
    const from: Position = { x: 0, y: 0, z: 0 };
    addStack(board, from, 1, 3);

    const sandboxSeqs = enumerateSequencesSandbox('hexagonal', board, from, 1);
    const backendSeqs = enumerateSequencesBackend('hexagonal', board, from, 1);

    expect(sandboxSeqs).toHaveLength(0);
    expect(backendSeqs).toHaveLength(0);
  });

  test('square8: attacker surrounded by own pieces - can overtake (capture friendly stacks)', () => {
    // In RingRift, overtaking allows capturing your own stacks
    const board = createTestBoard('square8');
    const from = pos(4, 4);
    addStack(board, from, 1, 3);
    // Surround with friendly stacks (all capturable via overtaking)
    addStack(board, pos(5, 4), 1, 2);
    addStack(board, pos(3, 4), 1, 2);
    addStack(board, pos(4, 5), 1, 2);
    addStack(board, pos(4, 3), 1, 2);

    const sandboxSeqs = enumerateSequencesSandbox('square8', board, from, 1);
    const backendSeqs = enumerateSequencesBackend('square8', board, from, 1);

    // Should have capture paths (4 directions, with possible chains)
    expect(sandboxSeqs.length).toBeGreaterThan(0);
    expect(backendSeqs.length).toBeGreaterThan(0);
    // Parity check
    expect(sandboxSeqs.length).toEqual(backendSeqs.length);
  });
});

// =============================================================================
// TESTS - BOUNDED RANDOM (SMALL SAMPLE, CI-SAFE)
// =============================================================================

describe('capture sequence enumeration parity (bounded random)', () => {
  /**
   * Deterministic pseudo-random number generator (LCG).
   */
  function makeRng(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 0xffffffff;
    };
  }

  function buildBoundedRandomSquare(
    boardType: 'square8' | 'square19',
    seed: number,
    targetCount: number
  ): NamedCaptureTestCase {
    const rng = makeRng(seed);
    const board = createTestBoard(boardType);
    const size = board.size;
    const from = pos(Math.floor(size / 2), Math.floor(size / 2));
    const player = 1;

    addStack(board, from, player, 3);

    const usedPositions = new Set<string>();
    usedPositions.add(positionToString(from));

    for (let t = 0; t < targetCount; t++) {
      for (let attempts = 0; attempts < 50; attempts++) {
        const direction = Math.floor(rng() * 4);
        const distance = 2;
        let dx = 0,
          dy = 0;
        if (direction === 0) dx = distance;
        else if (direction === 1) dx = -distance;
        else if (direction === 2) dy = distance;
        else dy = -distance;

        const tx = from.x + dx;
        const ty = from.y + dy;

        if (tx < 0 || tx >= size || ty < 0 || ty >= size) continue;

        const targetPos = pos(tx, ty);
        const key = positionToString(targetPos);
        if (usedPositions.has(key)) continue;

        usedPositions.add(key);
        addStack(board, targetPos, 2, 2);
        break;
      }
    }

    return {
      name: `${boardType} random seed=${seed}`,
      boardType,
      board,
      from,
      player,
    };
  }

  test('square8: bounded random boards maintain parity (seed 99999)', () => {
    const testCase = buildBoundedRandomSquare('square8', 99999, 2);
    runParityTest(testCase);
  });

  test('square8: bounded random boards maintain parity (seed 88888)', () => {
    const testCase = buildBoundedRandomSquare('square8', 88888, 2);
    runParityTest(testCase);
  });

  test('square19: bounded random boards maintain parity (seed 11111)', () => {
    const testCase = buildBoundedRandomSquare('square19', 11111, 2);
    runParityTest(testCase);
  });

  test('square19: bounded random boards maintain parity (seed 22222)', () => {
    const testCase = buildBoundedRandomSquare('square19', 22222, 2);
    runParityTest(testCase);
  });
});
