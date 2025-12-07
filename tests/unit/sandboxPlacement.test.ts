/**
 * Unit tests for sandboxPlacement.ts
 *
 * Tests for createHypotheticalBoardWithPlacement, hasAnyLegalMoveOrCaptureFrom,
 * and enumerateLegalRingPlacements functions.
 */

import {
  createHypotheticalBoardWithPlacement,
  hasAnyLegalMoveOrCaptureFrom,
  enumerateLegalRingPlacements,
  PlacementBoardView,
} from '../../src/client/sandbox/sandboxPlacement';
import { createTestBoard, pos, posStr } from '../utils/fixtures';
import type {
  BoardState,
  BoardType,
  PlacementContext,
  Position,
} from '../../src/shared/types/game';
import { BOARD_CONFIGS, isValidPosition } from '../../src/shared/engine';

describe('sandboxPlacement', () => {
  // Helper to create a view
  function createView(boardType: BoardType): PlacementBoardView {
    const config = BOARD_CONFIGS[boardType];
    return {
      isValidPosition: (p: Position) => isValidPosition(p, boardType, config.size),
      isCollapsedSpace: (p: Position, board: BoardState) =>
        board.collapsedSpaces.has(posStr(p.x, p.y, p.z)),
      getMarkerOwner: (p: Position, board: BoardState) =>
        board.markers.get(posStr(p.x, p.y, p.z))?.player,
    };
  }

  describe('createHypotheticalBoardWithPlacement', () => {
    it('creates new stack when no existing stack at position', () => {
      const board = createTestBoard('square8');
      const position = pos(3, 3);

      const result = createHypotheticalBoardWithPlacement(board, position, 1, 2);

      expect(result.stacks.has(posStr(3, 3))).toBe(true);
      const stack = result.stacks.get(posStr(3, 3))!;
      expect(stack.rings).toEqual([1, 1]);
      expect(stack.stackHeight).toBe(2);
      expect(stack.controllingPlayer).toBe(1);
    });

    it('adds rings on top of existing stack', () => {
      const board = createTestBoard('square8');
      board.stacks.set(posStr(2, 2), {
        position: pos(2, 2),
        rings: [2, 2, 2],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 2,
      });

      const result = createHypotheticalBoardWithPlacement(board, pos(2, 2), 1, 2);

      const stack = result.stacks.get(posStr(2, 2))!;
      expect(stack.rings).toEqual([1, 1, 2, 2, 2]);
      expect(stack.stackHeight).toBe(5);
      expect(stack.controllingPlayer).toBe(1);
      expect(stack.capHeight).toBe(2);
    });

    it('removes marker at position when placing', () => {
      const board = createTestBoard('square8');
      board.markers.set(posStr(4, 4), {
        position: pos(4, 4),
        player: 2,
        type: 'regular',
      });

      const result = createHypotheticalBoardWithPlacement(board, pos(4, 4), 1, 1);

      expect(result.markers.has(posStr(4, 4))).toBe(false);
      expect(result.stacks.has(posStr(4, 4))).toBe(true);
    });

    it('defaults to count of 1 when count is 0', () => {
      const board = createTestBoard('square8');
      const position = pos(5, 5);

      const result = createHypotheticalBoardWithPlacement(board, position, 1, 0);

      const stack = result.stacks.get(posStr(5, 5))!;
      expect(stack.rings).toEqual([1]); // Math.max(1, 0) = 1
      expect(stack.stackHeight).toBe(1);
    });

    it('preserves original board unchanged', () => {
      const board = createTestBoard('square8');
      board.stacks.set(posStr(1, 1), {
        position: pos(1, 1),
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      const result = createHypotheticalBoardWithPlacement(board, pos(1, 1), 1, 2);

      // Original should be unchanged
      expect(board.stacks.get(posStr(1, 1))!.rings).toEqual([2]);
      // Result should be updated
      expect(result.stacks.get(posStr(1, 1))!.rings).toEqual([1, 1, 2]);
    });
  });

  describe('hasAnyLegalMoveOrCaptureFrom', () => {
    it('returns true when stack has legal moves', () => {
      const board = createTestBoard('square8');
      board.stacks.set(posStr(4, 4), {
        position: pos(4, 4),
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      });

      const view = createView('square8');
      const result = hasAnyLegalMoveOrCaptureFrom('square8', board, pos(4, 4), 1, view);

      expect(result).toBe(true);
    });

    it('returns false when stack has no legal moves due to surrounding obstacles', () => {
      const board = createTestBoard('square8');
      // Place stack in corner
      board.stacks.set(posStr(0, 0), {
        position: pos(0, 0),
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      // Block all directions with collapsed spaces
      board.collapsedSpaces.set(posStr(1, 0), { position: pos(1, 0), collapsed: true });
      board.collapsedSpaces.set(posStr(0, 1), { position: pos(0, 1), collapsed: true });
      board.collapsedSpaces.set(posStr(1, 1), { position: pos(1, 1), collapsed: true });

      const view = createView('square8');
      const result = hasAnyLegalMoveOrCaptureFrom('square8', board, pos(0, 0), 1, view);

      expect(result).toBe(false);
    });

    it('handles position with no stack', () => {
      const board = createTestBoard('square8');
      const view = createView('square8');

      // No stack at position - should return false
      const result = hasAnyLegalMoveOrCaptureFrom('square8', board, pos(3, 3), 1, view);

      expect(result).toBe(false);
    });
  });

  describe('enumerateLegalRingPlacements', () => {
    describe('legacy path (no context)', () => {
      it('returns legal placements for square board', () => {
        const board = createTestBoard('square8');
        const view = createView('square8');

        const placements = enumerateLegalRingPlacements('square8', board, 1, view);

        // Should have some legal placements on empty board
        expect(placements.length).toBeGreaterThan(0);
      });

      it('excludes collapsed spaces', () => {
        const board = createTestBoard('square8');
        board.collapsedSpaces.set(posStr(4, 4), { position: pos(4, 4), collapsed: true });
        const view = createView('square8');

        const placements = enumerateLegalRingPlacements('square8', board, 1, view);
        const has44 = placements.some((p) => p.x === 4 && p.y === 4);

        expect(has44).toBe(false);
      });

      it('excludes positions with markers', () => {
        const board = createTestBoard('square8');
        board.markers.set(posStr(3, 3), {
          position: pos(3, 3),
          player: 2,
          type: 'regular',
        });
        const view = createView('square8');

        const placements = enumerateLegalRingPlacements('square8', board, 1, view);
        const has33 = placements.some((p) => p.x === 3 && p.y === 3);

        expect(has33).toBe(false);
      });

      it('returns legal placements for hexagonal board', () => {
        const board: BoardState = {
          type: 'hexagonal',
          size: 13, // radius=12
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
        };
        const view = createView('hexagonal');

        const placements = enumerateLegalRingPlacements('hexagonal', board, 1, view);

        // Should have some legal placements on empty hex board
        expect(placements.length).toBeGreaterThan(0);

        // Hex positions should have z coordinate
        const allHaveZ = placements.every((p) => p.z !== undefined);
        expect(allHaveZ).toBe(true);
      });

      it('excludes invalid positions on hexagonal board', () => {
        const board: BoardState = {
          type: 'hexagonal',
          size: 13, // radius=12
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
        };
        const view = createView('hexagonal');

        const placements = enumerateLegalRingPlacements('hexagonal', board, 1, view);

        // All positions should be valid hex positions (|x|, |y|, |z| <= radius and x+y+z = 0)
        const radius = 12; // size - 1
        const allValid = placements.every(
          (p) =>
            p.z !== undefined &&
            Math.abs(p.x) <= radius &&
            Math.abs(p.y) <= radius &&
            Math.abs(p.z) <= radius &&
            p.x + p.y + p.z === 0
        );
        expect(allValid).toBe(true);
      });

      it('excludes collapsed spaces on hexagonal board', () => {
        const board: BoardState = {
          type: 'hexagonal',
          size: 13, // radius=12
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
        };
        board.collapsedSpaces.set('0,0,0', { position: { x: 0, y: 0, z: 0 }, collapsed: true });
        const view = createView('hexagonal');

        const placements = enumerateLegalRingPlacements('hexagonal', board, 1, view);
        const hasOrigin = placements.some((p) => p.x === 0 && p.y === 0 && p.z === 0);

        expect(hasOrigin).toBe(false);
      });

      it('excludes markers on hexagonal board', () => {
        const board: BoardState = {
          type: 'hexagonal',
          size: 13, // radius=12
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
        };
        board.markers.set('1,0,-1', {
          position: { x: 1, y: 0, z: -1 },
          player: 2,
          type: 'regular',
        });
        const view = createView('hexagonal');

        const placements = enumerateLegalRingPlacements('hexagonal', board, 1, view);
        const hasMarkerPos = placements.some((p) => p.x === 1 && p.y === 0 && p.z === -1);

        expect(hasMarkerPos).toBe(false);
      });
    });

    describe('canonical path (with context)', () => {
      it('uses validatePlacementOnBoard for square board with context', () => {
        const board = createTestBoard('square8');
        const view = createView('square8');
        const ctx: PlacementContext = {
          boardType: 'square8',
          player: 1,
          ringsInHand: 18,
          maxPlacementCount: 18,
        };

        const placements = enumerateLegalRingPlacements('square8', board, 1, view, ctx);

        expect(placements.length).toBeGreaterThan(0);
      });

      it('uses validatePlacementOnBoard for hexagonal board with context', () => {
        const board: BoardState = {
          type: 'hexagonal',
          size: 13, // radius=12
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          formedLines: [],
          eliminatedRings: {},
        };
        const view = createView('hexagonal');
        const ctx: PlacementContext = {
          boardType: 'hexagonal',
          player: 1,
          ringsInHand: 27,
          maxPlacementCount: 27,
        };

        const placements = enumerateLegalRingPlacements('hexagonal', board, 1, view, ctx);

        expect(placements.length).toBeGreaterThan(0);
        const allHaveZ = placements.every((p) => p.z !== undefined);
        expect(allHaveZ).toBe(true);
      });

      it('respects ringsInHand constraint in context', () => {
        const board = createTestBoard('square8');
        const view = createView('square8');
        const ctx: PlacementContext = {
          boardType: 'square8',
          player: 1,
          ringsInHand: 0, // No rings to place
          maxPlacementCount: 0,
        };

        const placements = enumerateLegalRingPlacements('square8', board, 1, view, ctx);

        expect(placements.length).toBe(0);
      });
    });
  });
});
