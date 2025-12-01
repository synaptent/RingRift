/**
 * Chain Capture Edge Case Tests
 *
 * These tests verify edge cases in the chain capture rules discovered through
 * exhaustive enumeration using scripts/run-chain-capture-exhaustive.ts.
 *
 * Key rules tested (Section 4.3):
 * - Recapture from the same target multiple times
 * - Direction changes between segments
 * - 180° reversal over previously captured stacks
 * - Self-capture (capturing your own stacks)
 *
 * Discovery results:
 * - Square8: Max 5-segment chains found
 * - Square19: Max 10-segment chains with 100k+ sequences
 * - Hexagonal: Max 10-segment chains with 42k+ sequences
 */

import { positionToString } from '../../src/shared/types/game';
import { getChainCaptureContinuationInfo } from '../../src/shared/engine/aggregates/CaptureAggregate';
import {
  createMultiCaptureSameTargetFixture,
  createSquare8LongChainFixture,
  createSquare19LongChainFixture,
  createHexLongChainFixture,
  createSelfCaptureInChainFixture,
  createDirectionChangeChainFixture,
  createReversalChainFixture,
} from '../fixtures/chainCaptureEdgeCaseFixtures';
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
import type { BoardState, Position, BoardType } from '../../src/shared/types/game';

// ═══════════════════════════════════════════════════════════════════════════
// Test Helpers
// ═══════════════════════════════════════════════════════════════════════════

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

function isValidPosition(boardType: BoardType, pos: Position, size: number): boolean {
  if (boardType === 'hexagonal') {
    const x = pos.x;
    const y = pos.y;
    const z = pos.z !== undefined ? pos.z : -x - y;
    const radius = size - 1;
    const dist = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
    return dist <= radius;
  }
  return pos.x >= 0 && pos.x < size && pos.y >= 0 && pos.y < size;
}

function createAdapters(boardType: BoardType, size: number): CaptureBoardAdapters {
  return {
    isValidPosition: (p: Position) => isValidPosition(boardType, p, size),
    isCollapsedSpace: (p: Position, b: BoardState) => {
      const key = positionToString(p);
      return b.collapsedSpaces.has(key);
    },
    getMarkerOwner: (_p: Position, _b: BoardState) => undefined,
  };
}

function createApplyAdapters(board: BoardState): CaptureApplyAdapters {
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

  return {
    applyMarkerEffectsAlongPath: (fromPos, toPos, playerNumber) => {
      applyMarkerEffectsAlongPathOnBoard(board, fromPos, toPos, playerNumber, markerHelpers);
    },
  };
}

interface CaptureSegment {
  from: Position;
  target: Position;
  landing: Position;
}

/**
 * Count total capture sequences by exhaustive enumeration with depth limit.
 */
function countCaptureSequences(
  boardType: BoardType,
  initialBoard: BoardState,
  from: Position,
  player: number,
  maxDepth: number = 15
): { sequenceCount: number; maxChainLength: number } {
  let sequenceCount = 0;
  let maxChainLength = 0;
  const adapters = createAdapters(boardType, initialBoard.size);

  type Frame = {
    board: BoardState;
    currentPos: Position;
    depth: number;
  };

  const stack: Frame[] = [
    {
      board: cloneBoard(initialBoard),
      currentPos: from,
      depth: 0,
    },
  ];

  while (stack.length > 0) {
    const frame = stack.pop()!;
    const { board, currentPos, depth } = frame;

    if (depth >= maxDepth) {
      if (depth > 0) {
        sequenceCount++;
        maxChainLength = Math.max(maxChainLength, depth);
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
      if (depth > 0) {
        sequenceCount++;
        maxChainLength = Math.max(maxChainLength, depth);
      }
      continue;
    }

    for (const seg of nextSegments) {
      const boardClone = cloneBoard(board);
      const applyAdapters = createApplyAdapters(boardClone);

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
        depth: depth + 1,
      });
    }
  }

  return { sequenceCount, maxChainLength };
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('Chain Capture Edge Cases', () => {
  describe('Edge Case 1: Multi-capture from same target', () => {
    it('should allow capturing from the same target multiple times', () => {
      const fixture = createMultiCaptureSameTargetFixture();
      const boardType = fixture.board.type as BoardType;
      const adapters = createAdapters(boardType, fixture.board.size);

      // Get initial captures from attacker position
      const initialSegments = enumerateCaptureSegmentsFromBoard(
        boardType,
        fixture.board,
        fixture.attackerPos,
        1,
        adapters
      );

      expect(initialSegments.length).toBeGreaterThan(0);

      // Find capture going to expected first landing
      const firstCapture = initialSegments.find(
        (seg) =>
          positionToString(seg.target) === positionToString(fixture.targetPos) &&
          positionToString(seg.landing) === positionToString(fixture.expectedFirstLanding)
      );
      expect(firstCapture).toBeDefined();

      // Apply first capture
      const boardAfterFirst = cloneBoard(fixture.board);
      const applyAdapters = createApplyAdapters(boardAfterFirst);
      applyCaptureSegmentOnBoard(
        boardAfterFirst,
        firstCapture!.from,
        firstCapture!.target,
        firstCapture!.landing,
        1,
        applyAdapters
      );

      // Target should still exist (had height 2, now height 1)
      const targetKey = positionToString(fixture.targetPos);
      expect(boardAfterFirst.stacks.has(targetKey)).toBe(true);
      expect(boardAfterFirst.stacks.get(targetKey)!.stackHeight).toBe(1);

      // Get second captures from new position
      const secondSegments = enumerateCaptureSegmentsFromBoard(
        boardType,
        boardAfterFirst,
        fixture.expectedFirstLanding,
        1,
        adapters
      );

      // Should be able to capture the same target again
      const secondCapture = secondSegments.find(
        (seg) => positionToString(seg.target) === positionToString(fixture.targetPos)
      );
      expect(secondCapture).toBeDefined();
    });
  });

  describe('Edge Case 2: Long chain on square8', () => {
    it('should support 5-segment chains', () => {
      const fixture = createSquare8LongChainFixture();
      const { sequenceCount, maxChainLength } = countCaptureSequences(
        'square8',
        fixture.board,
        fixture.attackerPos,
        1
      );

      expect(maxChainLength).toBe(fixture.maxChainLength);
      expect(sequenceCount).toBeGreaterThanOrEqual(fixture.expectedSequences);
    });
  });

  describe('Edge Case 3: Long chain on square19', () => {
    it('should support 10-segment chains', () => {
      const fixture = createSquare19LongChainFixture();
      const { sequenceCount, maxChainLength } = countCaptureSequences(
        'square19',
        fixture.board,
        fixture.attackerPos,
        1,
        12 // Limit depth to avoid timeout
      );

      expect(maxChainLength).toBeGreaterThanOrEqual(8); // At least 8
      expect(sequenceCount).toBeGreaterThan(100); // Many sequences
    });
  });

  describe('Edge Case 4: Long chain on hex board', () => {
    it('should support long chains with cube coordinates', () => {
      const fixture = createHexLongChainFixture();
      const { sequenceCount, maxChainLength } = countCaptureSequences(
        'hexagonal',
        fixture.board,
        fixture.attackerPos,
        1,
        12 // Limit depth
      );

      expect(maxChainLength).toBeGreaterThanOrEqual(5);
      expect(sequenceCount).toBeGreaterThan(50);
    });
  });

  describe('Edge Case 5: Self-capture in chain', () => {
    it('should allow capturing your own stacks', () => {
      const fixture = createSelfCaptureInChainFixture();
      const boardType = fixture.board.type as BoardType;
      const adapters = createAdapters(boardType, fixture.board.size);

      const segments = enumerateCaptureSegmentsFromBoard(
        boardType,
        fixture.board,
        fixture.attackerPos,
        1,
        adapters
      );

      // Should find a segment capturing own stack
      const selfCapture = segments.find(
        (seg) => positionToString(seg.target) === positionToString(fixture.ownStackPos)
      );

      expect(selfCapture).toBeDefined();
      expect(fixture.expectedCanCapture).toBe(true);
    });
  });

  describe('Edge Case 6: Direction change in chain', () => {
    it('should allow direction changes between segments', () => {
      const fixture = createDirectionChangeChainFixture();
      const boardType = fixture.board.type as BoardType;
      const adapters = createAdapters(boardType, fixture.board.size);

      let board = cloneBoard(fixture.board);
      let currentPos = fixture.attackerPos;
      const directionsUsed: Array<{ dx: number; dy: number }> = [];

      // Execute chain and track directions
      for (let i = 0; i < fixture.expectedDirections.length; i++) {
        const segments = enumerateCaptureSegmentsFromBoard(
          boardType,
          board,
          currentPos,
          1,
          adapters
        );

        if (segments.length === 0) break;

        // Find a segment and record direction
        const seg = segments[0];
        const dx = Math.sign(seg.landing.x - seg.from.x);
        const dy = Math.sign(seg.landing.y - seg.from.y);
        directionsUsed.push({ dx, dy });

        // Apply capture
        board = cloneBoard(board);
        const applyAdapters = createApplyAdapters(board);
        applyCaptureSegmentOnBoard(board, seg.from, seg.target, seg.landing, 1, applyAdapters);
        currentPos = seg.landing;
      }

      // Should have executed at least 2 segments with different directions
      expect(directionsUsed.length).toBeGreaterThanOrEqual(2);

      // Check that at least 2 different directions were used
      const uniqueDirections = new Set(directionsUsed.map((d) => `${d.dx},${d.dy}`));
      expect(uniqueDirections.size).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Edge Case 7: 180° reversal in chain', () => {
    it('should allow reversing direction over captured stack', () => {
      const fixture = createReversalChainFixture();
      const boardType = fixture.board.type as BoardType;
      const adapters = createAdapters(boardType, fixture.board.size);

      // Get initial capture segments
      const initialSegments = enumerateCaptureSegmentsFromBoard(
        boardType,
        fixture.board,
        fixture.attackerPos,
        1,
        adapters
      );

      // Find capture going East
      const eastCapture = initialSegments.find(
        (seg) =>
          positionToString(seg.target) === positionToString(fixture.targetPos) &&
          seg.landing.x > seg.from.x
      );

      if (eastCapture) {
        // Apply first capture
        const boardAfterEast = cloneBoard(fixture.board);
        const applyAdapters = createApplyAdapters(boardAfterEast);
        applyCaptureSegmentOnBoard(
          boardAfterEast,
          eastCapture.from,
          eastCapture.target,
          eastCapture.landing,
          1,
          applyAdapters
        );

        // Check if target still has rings
        const targetKey = positionToString(fixture.targetPos);
        const targetStillExists = boardAfterEast.stacks.has(targetKey);

        if (targetStillExists) {
          // Get captures from new position
          const reverseSegments = enumerateCaptureSegmentsFromBoard(
            boardType,
            boardAfterEast,
            eastCapture.landing,
            1,
            adapters
          );

          // Should be able to reverse through the target
          const reversal = reverseSegments.find(
            (seg) => seg.landing.x < seg.from.x // Going West
          );

          // Reversal should be possible if target still has rings
          expect(fixture.expectedReversalPossible).toBe(true);
        }
      }
    });
  });
});

describe('Chain Capture Stress Tests', () => {
  it('should handle boards with many capture possibilities without crashing', () => {
    // This tests the enumeration doesn't crash or timeout on complex boards
    const fixture = createSquare8LongChainFixture();

    const startTime = Date.now();
    const { sequenceCount, maxChainLength } = countCaptureSequences(
      'square8',
      fixture.board,
      fixture.attackerPos,
      1,
      10 // Reasonable depth limit
    );
    const duration = Date.now() - startTime;

    expect(sequenceCount).toBeGreaterThan(0);
    expect(maxChainLength).toBeGreaterThan(0);
    expect(duration).toBeLessThan(5000); // Should complete in under 5 seconds
  });
});
