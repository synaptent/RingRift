import { BoardType, BoardState, Position, positionToString } from '../../src/shared/types/game';
import {
  enumerateCaptureSegmentsFromBoard,
  applyCaptureSegmentOnBoard,
  CaptureBoardAdapters,
  CaptureApplyAdapters
} from '../../src/client/sandbox/sandboxCaptures';
import { applyMarkerEffectsAlongPathOnBoard, MarkerPathHelpers } from '../../src/client/sandbox/sandboxMovement';
import { createTestBoard, addStack, pos, createTestGameState, createTestPlayer } from '../utils/fixtures';
import { getMovementDirectionsForBoardType } from '../../src/shared/engine/core';
import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import { getCaptureOptionsFromPosition as getBackendCaptureOptions } from '../../src/server/game/rules/captureChainEngine';

interface CaptureSequence {
  segments: { from: Position; target: Position; landing: Position }[];
  finalBoard: BoardState;
}

type CaptureTestCase = { boardType: BoardType; board: BoardState; from: Position; player: number };

/* MAX_SEQUENCES limit removed; enumeration now explores the full search space. */

function cloneBoard(board: BoardState): BoardState {
  return {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings }
  };
}

/**
 * Deterministic pseudo-random number generator (LCG) so that the
 * randomly generated test boards are stable across runs.
 */
function makeRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

function summarizeStacksForBoard(board: BoardState, attackerPlayer: number) {
  const attackerStacks: {
    owner: number;
    pos: string;
    stackHeight: number;
    capHeight: number;
  }[] = [];
  const targetStacks: {
    owner: number;
    pos: string;
    stackHeight: number;
    capHeight: number;
  }[] = [];

  board.stacks.forEach(stack => {
    const entry = {
      owner: stack.controllingPlayer,
      pos: positionToString(stack.position),
      stackHeight: stack.stackHeight,
      capHeight: stack.capHeight
    };

    if (stack.controllingPlayer === attackerPlayer) {
      attackerStacks.push(entry);
    } else {
      targetStacks.push(entry);
    }
  });

  return { attackerStacks, targetStacks };
}

function formatCaptureChain(seq: CaptureSequence): string {
  return seq.segments
    .map(s => `${positionToString(s.from)}->${positionToString(s.target)}->${positionToString(s.landing)}`)
    .join('|');
}

type SummaryKind = 'max_sequences' | 'max_chain_length';

function countMarkersAndCollapsed(board: BoardState): { markers: number; collapsed: number } {
  return {
    markers: board.markers.size,
    collapsed: board.collapsedSpaces.size
  };
}

function logCaseSummary(
  boardLabel: string,
  summaryKind: SummaryKind,
  info: {
    index: number;
    caseData: CaptureTestCase;
    sequences: CaptureSequence[];
  }
): void {
  const { caseData: c, index, sequences } = info;
  const { attackerStacks, targetStacks } = summarizeStacksForBoard(c.board, c.player);

  console.log(`\n[${boardLabel} ${summaryKind} case index=${index}] from=${positionToString(c.from)}`);
  console.log('  attacker stacks:');
  attackerStacks.forEach(s => {
    console.log(`    owner=${s.owner} at ${s.pos} height=${s.stackHeight} cap=${s.capHeight}`);
  });
  console.log('  target stacks:');
  targetStacks.forEach(s => {
    console.log(`    owner=${s.owner} at ${s.pos} height=${s.stackHeight} cap=${s.capHeight}`);
  });

  const numSequences = sequences.length;
  let maxChainLenForCase = 0;
  for (const seq of sequences) {
    if (seq.segments.length > maxChainLenForCase) {
      maxChainLenForCase = seq.segments.length;
    }
  }
  console.log(`  number of distinct capture sequences: ${numSequences}`);
  console.log(`  longest capture chain length in this case: ${maxChainLenForCase}`);

  if (maxChainLenForCase > 0) {
    const longestSeq =
      sequences.find(seq => seq.segments.length === maxChainLenForCase) || sequences[0];
    console.log(`  example longest chain: ${formatCaptureChain(longestSeq)}`);
  }
}

function logOutcomeSummary(
  boardLabel: string,
  summaryKind: 'max_markers' | 'max_collapsed_spaces',
  info: {
    index: number;
    caseData: CaptureTestCase;
    sequence: CaptureSequence;
    markerCount: number;
    collapsedCount: number;
  }
): void {
  const { caseData: c, index, sequence, markerCount, collapsedCount } = info;
  const { attackerStacks, targetStacks } = summarizeStacksForBoard(c.board, c.player);

  console.log(`\n[${boardLabel} ${summaryKind} case index=${index}] from=${positionToString(c.from)}`);
  console.log('  attacker stacks:');
  attackerStacks.forEach(s => {
    console.log(`    owner=${s.owner} at ${s.pos} height=${s.stackHeight} cap=${s.capHeight}`);
  });
  console.log('  target stacks:');
  targetStacks.forEach(s => {
    console.log(`    owner=${s.owner} at ${s.pos} height=${s.stackHeight} cap=${s.capHeight}`);
  });

  console.log(`  markers on final board: ${markerCount}`);
  console.log(`  collapsed spaces on final board: ${collapsedCount}`);
  console.log(`  example sequence: ${formatCaptureChain(sequence)}`);
}

/**
 * Exhaustively enumerate all maximal capture sequences using the
 * sandbox helper enumerateCaptureSegmentsFromBoard + applyCaptureSegmentOnBoard.
 */
function enumerateAllCaptureSequencesSandbox(
  boardType: BoardType,
  initialBoard: BoardState,
  from: Position,
  player: number
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
    getMarkerOwner: (_p: Position, _b: BoardState) => undefined
  };

  type Frame = {
    board: BoardState;
    currentPos: Position;
    segments: { from: Position; target: Position; landing: Position }[];
  };

  const stack: Frame[] = [
    {
      board: cloneBoard(initialBoard),
      currentPos: from,
      segments: []
    }
  ];

  while (stack.length > 0) {
    const frame = stack.pop()!;
    const { board, currentPos, segments } = frame;

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
        }
      };

      const applyAdapters: CaptureApplyAdapters = {
        applyMarkerEffectsAlongPath: (fromPos, toPos, playerNumber) => {
          applyMarkerEffectsAlongPathOnBoard(boardClone, fromPos, toPos, playerNumber, markerHelpers);
        }
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
        segments: [...segments, seg]
      });
    }
  }

  // Normalize sequences by stringifying positions, to simplify equality
  // checks and to ensure deterministic ordering.
  sequences.sort((a, b) => {
    const aKey = a.segments
      .map(s => `${positionToString(s.from)}->${positionToString(s.target)}->${positionToString(s.landing)}`)
      .join('|');
    const bKey = b.segments
      .map(s => `${positionToString(s.from)}->${positionToString(s.target)}->${positionToString(s.landing)}`)
      .join('|');
    return aKey.localeCompare(bKey);
  });

  return sequences;
}

/**
 * Exhaustively enumerate all maximal capture sequences using the backend
 * RuleEngine.getValidMoves (capture moves only) combined with the same
 * capture-application helper used by the sandbox.
 */
function enumerateAllCaptureSequencesBackend(
  boardType: BoardType,
  initialBoard: BoardState,
  from: Position,
  player: number
): CaptureSequence[] {
  const sequences: CaptureSequence[] = [];
  const bm = new BoardManager(boardType);
  const engine = new RuleEngine(bm as any, boardType as any);

  type Frame = {
    board: BoardState;
    currentPos: Position;
    segments: { from: Position; target: Position; landing: Position }[];
  };

  const stack: Frame[] = [
    {
      board: cloneBoard(initialBoard),
      currentPos: from,
      segments: []
    }
  ];

  while (stack.length > 0) {
    const frame = stack.pop()!;
    const { board, currentPos, segments } = frame;

    const gameState = createTestGameState({
      boardType,
      board,
      currentPlayer: player,
      currentPhase: 'capture',
      players: [createTestPlayer(1), createTestPlayer(2)],
      moveHistory: []
    });

    const moves = getBackendCaptureOptions(currentPos, player, gameState, {
      boardManager: bm,
      ruleEngine: engine
    });

    if (moves.length === 0) {
      if (segments.length > 0) {
        sequences.push({ segments: [...segments], finalBoard: cloneBoard(board) });
      }
      continue;
    }

    for (const move of moves) {
      if (!move.from || !move.captureTarget) continue;
      const boardClone = cloneBoard(board);

      const markerHelpers: MarkerPathHelpers = {
        setMarker: (position, playerNumber, b) => bm.setMarker(position, playerNumber, b),
        collapseMarker: (position, playerNumber, b) => bm.collapseMarker(position, playerNumber, b),
        flipMarker: (position, playerNumber, b) => bm.flipMarker(position, playerNumber, b)
      };

      const applyAdapters: CaptureApplyAdapters = {
        applyMarkerEffectsAlongPath: (fromPos, toPos, playerNumber) => {
          applyMarkerEffectsAlongPathOnBoard(boardClone, fromPos, toPos, playerNumber, markerHelpers);
        }
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
        segments: [...segments, { from: move.from, target: move.captureTarget, landing: move.to }]
      });
    }
  }

  sequences.sort((a, b) => {
    const aKey = a.segments
      .map(s => `${positionToString(s.from)}->${positionToString(s.target)}->${positionToString(s.landing)}`)
      .join('|');
    const bKey = b.segments
      .map(s => `${positionToString(s.from)}->${positionToString(s.target)}->${positionToString(s.landing)}`)
      .join('|');
    return aKey.localeCompare(bKey);
  });

  return sequences;
}

/**
 * Board generators for targeted capture-sequence parity tests.
 *
 * These now generate ~50 randomised positions per board type, within
 * constrained target-count and collapsed-space ranges, using a seeded
 * RNG for stability.
 */

function buildRandomSquareCaptureBoards(
  boardType: 'square8' | 'square19',
  numCases: number,
  seed: number,
  minTargets: number,
  maxTargets: number
): CaptureTestCase[] {
  const rng = makeRng(seed);
  const cases: CaptureTestCase[] = [];

  const directions = getMovementDirectionsForBoardType(boardType);

  for (let i = 0; i < numCases; i++) {
    const board = createTestBoard(boardType);
    const size = board.size;
    const from = pos(Math.floor(size / 2), Math.floor(size / 2));
    const player = 1;

    addStack(board, from, player, 3);

    const usedPositions = new Set<string>();
    usedPositions.add(positionToString(from));

    const numTargets = minTargets + Math.floor(rng() * (maxTargets - minTargets + 1));

    // Primary target along the east ray at distance 2 to guarantee at least
    // one straightforward capture path that cannot be blocked by random
    // collapsed spaces.
    const primaryDir = { x: 1, y: 0 };
    const primaryDistance = 2;
    const primaryTarget = pos(from.x + primaryDir.x * primaryDistance, from.y + primaryDir.y * primaryDistance);
    addStack(board, primaryTarget, 2, 2);
    usedPositions.add(positionToString(primaryTarget));

    // Additional targets in random directions/distances.
    for (let t = 1; t < numTargets; t++) {
      let placed = false;
      for (let attempts = 0; attempts < 50 && !placed; attempts++) {
        const dir = directions[Math.floor(rng() * directions.length)];
        const maxStep = size - 2; // leave a margin
        const distance = 2 + Math.floor(rng() * Math.max(1, Math.min(maxStep, 6)));

        const tx = from.x + dir.x * distance;
        const ty = from.y + dir.y * distance;
        if (tx < 0 || tx >= size || ty < 0 || ty >= size) continue;

        const targetPos = pos(tx, ty);
        const key = positionToString(targetPos);
        if (usedPositions.has(key)) continue;

        usedPositions.add(key);
        const height = rng() < 0.5 ? 2 : 3;
        addStack(board, targetPos, 2, height);
        placed = true;
      }
    }

    // Collapsed spaces: choose 0–2, avoiding the attacker, targets, and the
    // primary capture ray (so at least one capture remains available).
    const forbiddenCollapsed = new Set<string>();
    forbiddenCollapsed.add(positionToString(from));
    forbiddenCollapsed.add(positionToString(primaryTarget));
    for (let step = 1; step < primaryDistance; step++) {
      const p = pos(from.x + primaryDir.x * step, from.y + primaryDir.y * step);
      forbiddenCollapsed.add(positionToString(p));
    }

    const numCollapsed = Math.floor(rng() * 3); // 0–2
    for (let c = 0; c < numCollapsed; c++) {
      let placed = false;
      for (let attempts = 0; attempts < 50 && !placed; attempts++) {
        const x = Math.floor(rng() * size);
        const y = Math.floor(rng() * size);
        const p = pos(x, y);
        const key = positionToString(p);
        if (forbiddenCollapsed.has(key) || usedPositions.has(key)) continue;
        board.collapsedSpaces.set(key, 0);
        forbiddenCollapsed.add(key);
        placed = true;
      }
    }

    cases.push({ boardType, board, from, player });
  }

  return cases;
}

function buildRandomHexCaptureBoards(
  numCases: number,
  seed: number,
  minTargets: number,
  maxTargets: number
): CaptureTestCase[] {
  const rng = makeRng(seed);
  const cases: CaptureTestCase[] = [];

  const boardType: BoardType = 'hexagonal';
  const directions = getMovementDirectionsForBoardType('hexagonal');

  for (let i = 0; i < numCases; i++) {
    const board = createTestBoard(boardType);
    const radius = board.size - 1;

    const from: Position = { x: 0, y: 0, z: 0 };
    const player = 1;
    addStack(board, from, player, 3);

    const usedPositions = new Set<string>();
    usedPositions.add(positionToString(from));

    const numTargets = minTargets + Math.floor(rng() * (maxTargets - minTargets + 1));

    // Primary target along the first hex direction at distance 2.
    const primaryDir = directions[0];
    const primaryDistance = 2;
    const primaryTarget: Position = {
      x: from.x + primaryDir.x * primaryDistance,
      y: from.y + primaryDir.y * primaryDistance,
      z: (from.z || 0) + (primaryDir.z || 0) * primaryDistance
    };
    addStack(board, primaryTarget, 2, 2);
    usedPositions.add(positionToString(primaryTarget));

    // Additional targets.
    for (let t = 1; t < numTargets; t++) {
      let placed = false;
      for (let attempts = 0; attempts < 80 && !placed; attempts++) {
        const dir = directions[Math.floor(rng() * directions.length)];
        const maxStep = radius - 1;
        const distance = 2 + Math.floor(rng() * Math.max(1, Math.min(maxStep, 6)));

        const tx = from.x + dir.x * distance;
        const ty = from.y + dir.y * distance;
        const tz = (from.z || 0) + (dir.z || 0) * distance;

        const dist = Math.max(Math.abs(tx), Math.abs(ty), Math.abs(tz));
        if (dist > radius) continue;

        const targetPos: Position = { x: tx, y: ty, z: tz };
        const key = positionToString(targetPos);
        if (usedPositions.has(key)) continue;

        usedPositions.add(key);
        const height = rng() < 0.5 ? 2 : 3;
        addStack(board, targetPos, 2, height);
        placed = true;
      }
    }

    // Collapsed spaces: choose 0–2, avoiding the attacker, targets, and the
    // primary capture ray.
    const forbiddenCollapsed = new Set<string>();
    forbiddenCollapsed.add(positionToString(from));
    forbiddenCollapsed.add(positionToString(primaryTarget));
    for (let step = 1; step < primaryDistance; step++) {
      const p: Position = {
        x: from.x + primaryDir.x * step,
        y: from.y + primaryDir.y * step,
        z: (from.z || 0) + (primaryDir.z || 0) * step
      };
      forbiddenCollapsed.add(positionToString(p));
    }

    const numCollapsed = Math.floor(rng() * 3); // 0–2
    for (let c = 0; c < numCollapsed; c++) {
      let placed = false;
      for (let attempts = 0; attempts < 80 && !placed; attempts++) {
        const x = Math.floor(rng() * (2 * radius + 1)) - radius;
        const y = Math.floor(rng() * (2 * radius + 1)) - radius;
        const z = -x - y;
        const dist = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
        if (dist > radius) continue;

        const p: Position = { x, y, z };
        const key = positionToString(p);
        if (forbiddenCollapsed.has(key) || usedPositions.has(key)) continue;

        board.collapsedSpaces.set(key, 0);
        forbiddenCollapsed.add(key);
        placed = true;
      }
    }

    cases.push({ boardType, board, from, player });
  }

  return cases;
}

// --- Tests ---

describe('capture sequence enumeration parity (sandbox vs backend)', () => {
  test('square8: for random positions with 2–6 targets, sandbox and backend enumerate identical capture sequences', () => {
    const cases = buildRandomSquareCaptureBoards('square8', 50, 12345, 2, 6);

    let maxSeqCount = 0;
    let maxSeqCase:
      | {
          index: number;
          caseData: CaptureTestCase;
          sequences: CaptureSequence[];
        }
      | null = null;

    let maxChainLen = 0;
    let maxChainCase:
      | {
          index: number;
          caseData: CaptureTestCase;
          sequences: CaptureSequence[];
        }
      | null = null;

    let maxMarkersCount = 0;
    let maxMarkersCase:
      | {
          index: number;
          caseData: CaptureTestCase;
          sequence: CaptureSequence;
          markerCount: number;
          collapsedCount: number;
        }
      | null = null;

    let maxCollapsedCount = 0;
    let maxCollapsedCase:
      | {
          index: number;
          caseData: CaptureTestCase;
          sequence: CaptureSequence;
          markerCount: number;
          collapsedCount: number;
        }
      | null = null;

    cases.forEach((c, index) => {
      const sandboxSeqs = enumerateAllCaptureSequencesSandbox(
        c.boardType,
        c.board,
        c.from,
        c.player
      );
      const backendSeqs = enumerateAllCaptureSequencesBackend(
        c.boardType,
        c.board,
        c.from,
        c.player
      );

      const sandboxKeys = sandboxSeqs.map(seq =>
        seq.segments
          .map(s => `${positionToString(s.from)}->${positionToString(s.target)}->${positionToString(s.landing)}`)
          .join('|')
      );
      const backendKeys = backendSeqs.map(seq =>
        seq.segments
          .map(s => `${positionToString(s.from)}->${positionToString(s.target)}->${positionToString(s.landing)}`)
          .join('|')
      );

      sandboxKeys.sort();
      backendKeys.sort();

      // Parity check only; do not emit exhaustive sequence lists.
      expect(backendKeys).toEqual(sandboxKeys);

      const numSequences = sandboxSeqs.length;
      let maxChainLenForCase = 0;
      for (const seq of sandboxSeqs) {
        if (seq.segments.length > maxChainLenForCase) {
          maxChainLenForCase = seq.segments.length;
        }
      }

      if (numSequences > maxSeqCount) {
        maxSeqCount = numSequences;
        maxSeqCase = { index, caseData: c, sequences: sandboxSeqs };
      }

      if (maxChainLenForCase > maxChainLen) {
        maxChainLen = maxChainLenForCase;
        maxChainCase = { index, caseData: c, sequences: sandboxSeqs };
      }

      // Track positions yielding the most markers and collapsed spaces on the
      // final board after a valid capture sequence.
      sandboxSeqs.forEach(seq => {
        const { markers, collapsed } = countMarkersAndCollapsed(seq.finalBoard);

        if (markers > maxMarkersCount) {
          maxMarkersCount = markers;
          maxMarkersCase = {
            index,
            caseData: c,
            sequence: seq,
            markerCount: markers,
            collapsedCount: collapsed
          };
        }

        if (collapsed > maxCollapsedCount) {
          maxCollapsedCount = collapsed;
          maxCollapsedCase = {
            index,
            caseData: c,
            sequence: seq,
            markerCount: markers,
            collapsedCount: collapsed
          };
        }
      });
    });

    if (maxSeqCase) {
      logCaseSummary('square8', 'max_sequences', maxSeqCase);
    }
    if (maxChainCase) {
      logCaseSummary('square8', 'max_chain_length', maxChainCase);
    }
    if (maxMarkersCase) {
      logOutcomeSummary('square8', 'max_markers', maxMarkersCase);
    }
    if (maxCollapsedCase) {
      logOutcomeSummary('square8', 'max_collapsed_spaces', maxCollapsedCase);
    }
  });

  test('square19: for random positions with 2–4 targets, sandbox and backend enumerate identical capture sequences', () => {
    const cases = buildRandomSquareCaptureBoards('square19', 50, 23456, 2, 4);

    let maxSeqCount = 0;
    let maxSeqCase:
      | {
          index: number;
          caseData: CaptureTestCase;
          sequences: CaptureSequence[];
        }
      | null = null;

    let maxChainLen = 0;
    let maxChainCase:
      | {
          index: number;
          caseData: CaptureTestCase;
          sequences: CaptureSequence[];
        }
      | null = null;

    let maxMarkersCount = 0;
    let maxMarkersCase:
      | {
          index: number;
          caseData: CaptureTestCase;
          sequence: CaptureSequence;
          markerCount: number;
          collapsedCount: number;
        }
      | null = null;

    let maxCollapsedCount = 0;
    let maxCollapsedCase:
      | {
          index: number;
          caseData: CaptureTestCase;
          sequence: CaptureSequence;
          markerCount: number;
          collapsedCount: number;
        }
      | null = null;

    cases.forEach((c, index) => {
      const sandboxSeqs = enumerateAllCaptureSequencesSandbox(
        c.boardType,
        c.board,
        c.from,
        c.player
      );
      const backendSeqs = enumerateAllCaptureSequencesBackend(
        c.boardType,
        c.board,
        c.from,
        c.player
      );

      const sandboxKeys = sandboxSeqs.map(seq =>
        seq.segments
          .map(s => `${positionToString(s.from)}->${positionToString(s.target)}->${positionToString(s.landing)}`)
          .join('|')
      );
      const backendKeys = backendSeqs.map(seq =>
        seq.segments
          .map(s => `${positionToString(s.from)}->${positionToString(s.target)}->${positionToString(s.landing)}`)
          .join('|')
      );

      sandboxKeys.sort();
      backendKeys.sort();

      // Parity check only; do not emit exhaustive sequence lists.
      expect(backendKeys).toEqual(sandboxKeys);

      const numSequences = sandboxSeqs.length;
      let maxChainLenForCase = 0;
      for (const seq of sandboxSeqs) {
        if (seq.segments.length > maxChainLenForCase) {
          maxChainLenForCase = seq.segments.length;
        }
      }

      if (numSequences > maxSeqCount) {
        maxSeqCount = numSequences;
        maxSeqCase = { index, caseData: c, sequences: sandboxSeqs };
      }

      if (maxChainLenForCase > maxChainLen) {
        maxChainLen = maxChainLenForCase;
        maxChainCase = { index, caseData: c, sequences: sandboxSeqs };
      }

      sandboxSeqs.forEach(seq => {
        const { markers, collapsed } = countMarkersAndCollapsed(seq.finalBoard);

        if (markers > maxMarkersCount) {
          maxMarkersCount = markers;
          maxMarkersCase = {
            index,
            caseData: c,
            sequence: seq,
            markerCount: markers,
            collapsedCount: collapsed
          };
        }

        if (collapsed > maxCollapsedCount) {
          maxCollapsedCount = collapsed;
          maxCollapsedCase = {
            index,
            caseData: c,
            sequence: seq,
            markerCount: markers,
            collapsedCount: collapsed
          };
        }
      });
    });

    if (maxSeqCase) {
      logCaseSummary('square19', 'max_sequences', maxSeqCase);
    }
    if (maxChainCase) {
      logCaseSummary('square19', 'max_chain_length', maxChainCase);
    }
    if (maxMarkersCase) {
      logOutcomeSummary('square19', 'max_markers', maxMarkersCase);
    }
    if (maxCollapsedCase) {
      logOutcomeSummary('square19', 'max_collapsed_spaces', maxCollapsedCase);
    }
  });

  test('hexagonal: for random positions with 2–4 targets, sandbox and backend enumerate identical capture sequences', () => {
    const cases = buildRandomHexCaptureBoards(50, 34567, 2, 4);

    let maxSeqCount = 0;
    let maxSeqCase:
      | {
          index: number;
          caseData: CaptureTestCase;
          sequences: CaptureSequence[];
        }
      | null = null;

    let maxChainLen = 0;
    let maxChainCase:
      | {
          index: number;
          caseData: CaptureTestCase;
          sequences: CaptureSequence[];
        }
      | null = null;

    let maxMarkersCount = 0;
    let maxMarkersCase:
      | {
          index: number;
          caseData: CaptureTestCase;
          sequence: CaptureSequence;
          markerCount: number;
          collapsedCount: number;
        }
      | null = null;

    let maxCollapsedCount = 0;
    let maxCollapsedCase:
      | {
          index: number;
          caseData: CaptureTestCase;
          sequence: CaptureSequence;
          markerCount: number;
          collapsedCount: number;
        }
      | null = null;

    cases.forEach((c, index) => {
      const sandboxSeqs = enumerateAllCaptureSequencesSandbox(
        c.boardType,
        c.board,
        c.from,
        c.player
      );
      const backendSeqs = enumerateAllCaptureSequencesBackend(
        c.boardType,
        c.board,
        c.from,
        c.player
      );

      const sandboxKeys = sandboxSeqs.map(seq =>
        seq.segments
          .map(s => `${positionToString(s.from)}->${positionToString(s.target)}->${positionToString(s.landing)}`)
          .join('|')
      );
      const backendKeys = backendSeqs.map(seq =>
        seq.segments
          .map(s => `${positionToString(s.from)}->${positionToString(s.target)}->${positionToString(s.landing)}`)
          .join('|')
      );

      sandboxKeys.sort();
      backendKeys.sort();

      // Parity check only; do not emit exhaustive sequence lists.
      expect(backendKeys).toEqual(sandboxKeys);

      const numSequences = sandboxSeqs.length;
      let maxChainLenForCase = 0;
      for (const seq of sandboxSeqs) {
        if (seq.segments.length > maxChainLenForCase) {
          maxChainLenForCase = seq.segments.length;
        }
      }

      if (numSequences > maxSeqCount) {
        maxSeqCount = numSequences;
        maxSeqCase = { index, caseData: c, sequences: sandboxSeqs };
      }

      if (maxChainLenForCase > maxChainLen) {
        maxChainLen = maxChainLenForCase;
        maxChainCase = { index, caseData: c, sequences: sandboxSeqs };
      }

      sandboxSeqs.forEach(seq => {
        const { markers, collapsed } = countMarkersAndCollapsed(seq.finalBoard);

        if (markers > maxMarkersCount) {
          maxMarkersCount = markers;
          maxMarkersCase = {
            index,
            caseData: c,
            sequence: seq,
            markerCount: markers,
            collapsedCount: collapsed
          };
        }

        if (collapsed > maxCollapsedCount) {
          maxCollapsedCount = collapsed;
          maxCollapsedCase = {
            index,
            caseData: c,
            sequence: seq,
            markerCount: markers,
            collapsedCount: collapsed
          };
        }
      });
    });

    if (maxSeqCase) {
      logCaseSummary('hexagonal', 'max_sequences', maxSeqCase);
    }
    if (maxChainCase) {
      logCaseSummary('hexagonal', 'max_chain_length', maxChainCase);
    }
    if (maxMarkersCase) {
      logOutcomeSummary('hexagonal', 'max_markers', maxMarkersCase);
    }
    if (maxCollapsedCase) {
      logOutcomeSummary('hexagonal', 'max_collapsed_spaces', maxCollapsedCase);
    }
  });
});
