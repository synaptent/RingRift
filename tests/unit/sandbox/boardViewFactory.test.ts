import {
  createBoardView,
  createMovementBoardView,
  createCaptureBoardAdapters,
  createSandboxBoardView,
  bindSandboxViewToBoard,
  hasPlayerStackAt,
  getPlayerStackPositions,
  countPlayerRings,
} from '../../../src/client/sandbox/boardViewFactory';
import type { BoardState, Position, RingStack, MarkerInfo } from '../../../src/shared/engine';

describe('boardViewFactory', () => {
  // Helper to create a minimal board state for testing
  const createTestBoard = (): BoardState => {
    const stacks = new Map<string, RingStack>();
    const markers = new Map<string, MarkerInfo>();
    const collapsedSpaces = new Map<string, number>();
    const territories = new Map<string, any>();
    const formedLines: any[] = [];
    const eliminatedRings = { 1: 0, 2: 0 };

    // Add a stack at (2, 3) controlled by player 1
    stacks.set('2,3', {
      position: { x: 2, y: 3 },
      rings: [1, 1, 2],
      stackHeight: 3,
      capHeight: 2,
      controllingPlayer: 1,
    });

    // Add a stack at (4, 4) controlled by player 2
    stacks.set('4,4', {
      position: { x: 4, y: 4 },
      rings: [2, 2],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 2,
    });

    // Add a marker at (5, 5) owned by player 1
    markers.set('5,5', { position: { x: 5, y: 5 }, player: 1 });

    // Add a collapsed space at (1, 1)
    collapsedSpaces.set('1,1', 1);

    return {
      stacks,
      markers,
      collapsedSpaces,
      territories,
      formedLines,
      eliminatedRings,
    };
  };

  describe('createBoardView', () => {
    it('creates a unified board view with all methods', () => {
      const board = createTestBoard();
      const view = createBoardView('square8', board);

      expect(view.isValidPosition).toBeDefined();
      expect(view.isCollapsedSpace).toBeDefined();
      expect(view.getStackAt).toBeDefined();
      expect(view.getMarkerOwner).toBeDefined();
    });

    it('isValidPosition returns true for valid positions', () => {
      const board = createTestBoard();
      const view = createBoardView('square8', board);

      expect(view.isValidPosition({ x: 0, y: 0 })).toBe(true);
      expect(view.isValidPosition({ x: 7, y: 7 })).toBe(true);
      expect(view.isValidPosition({ x: 3, y: 5 })).toBe(true);
    });

    it('isValidPosition returns false for invalid positions', () => {
      const board = createTestBoard();
      const view = createBoardView('square8', board);

      expect(view.isValidPosition({ x: -1, y: 0 })).toBe(false);
      expect(view.isValidPosition({ x: 8, y: 0 })).toBe(false);
      expect(view.isValidPosition({ x: 0, y: 8 })).toBe(false);
    });

    it('isCollapsedSpace returns true for collapsed positions', () => {
      const board = createTestBoard();
      const view = createBoardView('square8', board);

      expect(view.isCollapsedSpace({ x: 1, y: 1 })).toBe(true);
    });

    it('isCollapsedSpace returns false for non-collapsed positions', () => {
      const board = createTestBoard();
      const view = createBoardView('square8', board);

      expect(view.isCollapsedSpace({ x: 0, y: 0 })).toBe(false);
      expect(view.isCollapsedSpace({ x: 2, y: 3 })).toBe(false);
    });

    it('getStackAt returns stack info at valid position', () => {
      const board = createTestBoard();
      const view = createBoardView('square8', board);

      const stack = view.getStackAt({ x: 2, y: 3 });
      expect(stack).toBeDefined();
      expect(stack!.controllingPlayer).toBe(1);
      expect(stack!.capHeight).toBe(2);
      expect(stack!.stackHeight).toBe(3);
    });

    it('getStackAt returns undefined for empty position', () => {
      const board = createTestBoard();
      const view = createBoardView('square8', board);

      expect(view.getStackAt({ x: 0, y: 0 })).toBeUndefined();
    });

    it('getMarkerOwner returns owner for marker position', () => {
      const board = createTestBoard();
      const view = createBoardView('square8', board);

      expect(view.getMarkerOwner({ x: 5, y: 5 })).toBe(1);
    });

    it('getMarkerOwner returns undefined for non-marker position', () => {
      const board = createTestBoard();
      const view = createBoardView('square8', board);

      expect(view.getMarkerOwner({ x: 0, y: 0 })).toBeUndefined();
    });
  });

  describe('createMovementBoardView', () => {
    it('returns same interface as createBoardView', () => {
      const board = createTestBoard();
      const view = createMovementBoardView('square8', board);

      expect(view.isValidPosition({ x: 3, y: 3 })).toBe(true);
      expect(view.isCollapsedSpace({ x: 1, y: 1 })).toBe(true);
      expect(view.getStackAt({ x: 2, y: 3 })?.controllingPlayer).toBe(1);
      expect(view.getMarkerOwner({ x: 5, y: 5 })).toBe(1);
    });
  });

  describe('createCaptureBoardAdapters', () => {
    it('returns same interface as createBoardView', () => {
      const board = createTestBoard();
      const adapters = createCaptureBoardAdapters('square8', board);

      expect(adapters.isValidPosition({ x: 3, y: 3 })).toBe(true);
      expect(adapters.isCollapsedSpace({ x: 1, y: 1 })).toBe(true);
      expect(adapters.getStackAt({ x: 4, y: 4 })?.controllingPlayer).toBe(2);
      expect(adapters.getMarkerOwner({ x: 5, y: 5 })).toBe(1);
    });
  });

  describe('createSandboxBoardView', () => {
    it('creates a sandbox-style view with board parameter', () => {
      const view = createSandboxBoardView('square8');

      expect(view.isValidPosition).toBeDefined();
      expect(view.isCollapsedSpace).toBeDefined();
      expect(view.getMarkerOwner).toBeDefined();
    });

    it('isValidPosition works without board parameter', () => {
      const view = createSandboxBoardView('square8');

      expect(view.isValidPosition({ x: 3, y: 3 })).toBe(true);
      expect(view.isValidPosition({ x: -1, y: 0 })).toBe(false);
    });

    it('isCollapsedSpace accepts board as parameter', () => {
      const view = createSandboxBoardView('square8');
      const board = createTestBoard();

      expect(view.isCollapsedSpace({ x: 1, y: 1 }, board)).toBe(true);
      expect(view.isCollapsedSpace({ x: 0, y: 0 }, board)).toBe(false);
    });

    it('getMarkerOwner accepts board as parameter', () => {
      const view = createSandboxBoardView('square8');
      const board = createTestBoard();

      expect(view.getMarkerOwner({ x: 5, y: 5 }, board)).toBe(1);
      expect(view.getMarkerOwner({ x: 0, y: 0 }, board)).toBeUndefined();
    });

    it('can be used with different boards', () => {
      const view = createSandboxBoardView('square8');
      const board1 = createTestBoard();
      const board2 = createTestBoard();

      // Modify board2 to have different collapsed space
      board2.collapsedSpaces.delete('1,1');
      board2.collapsedSpaces.set('2,2', 2);

      expect(view.isCollapsedSpace({ x: 1, y: 1 }, board1)).toBe(true);
      expect(view.isCollapsedSpace({ x: 1, y: 1 }, board2)).toBe(false);
      expect(view.isCollapsedSpace({ x: 2, y: 2 }, board1)).toBe(false);
      expect(view.isCollapsedSpace({ x: 2, y: 2 }, board2)).toBe(true);
    });
  });

  describe('bindSandboxViewToBoard', () => {
    it('converts sandbox view to closure-based view', () => {
      const sandboxView = createSandboxBoardView('square8');
      const board = createTestBoard();
      const closureView = bindSandboxViewToBoard(sandboxView, board, 'square8');

      expect(closureView.isValidPosition({ x: 3, y: 3 })).toBe(true);
      expect(closureView.isCollapsedSpace({ x: 1, y: 1 })).toBe(true);
      expect(closureView.getStackAt({ x: 2, y: 3 })?.controllingPlayer).toBe(1);
      expect(closureView.getMarkerOwner({ x: 5, y: 5 })).toBe(1);
    });
  });

  describe('utility functions', () => {
    describe('hasPlayerStackAt', () => {
      it('returns true when player has stack at position', () => {
        const board = createTestBoard();

        expect(hasPlayerStackAt(board, { x: 2, y: 3 }, 1)).toBe(true);
        expect(hasPlayerStackAt(board, { x: 4, y: 4 }, 2)).toBe(true);
      });

      it('returns false when different player owns stack', () => {
        const board = createTestBoard();

        expect(hasPlayerStackAt(board, { x: 2, y: 3 }, 2)).toBe(false);
        expect(hasPlayerStackAt(board, { x: 4, y: 4 }, 1)).toBe(false);
      });

      it('returns false when no stack at position', () => {
        const board = createTestBoard();

        expect(hasPlayerStackAt(board, { x: 0, y: 0 }, 1)).toBe(false);
      });
    });

    describe('getPlayerStackPositions', () => {
      it('returns all stack positions for a player', () => {
        const board = createTestBoard();

        const p1Positions = getPlayerStackPositions(board, 1);
        expect(p1Positions).toHaveLength(1);
        expect(p1Positions[0]).toEqual({ x: 2, y: 3 });

        const p2Positions = getPlayerStackPositions(board, 2);
        expect(p2Positions).toHaveLength(1);
        expect(p2Positions[0]).toEqual({ x: 4, y: 4 });
      });

      it('returns empty array when player has no stacks', () => {
        const board = createTestBoard();

        const p3Positions = getPlayerStackPositions(board, 3);
        expect(p3Positions).toHaveLength(0);
      });
    });

    describe('countPlayerRings', () => {
      it('counts total rings for a player', () => {
        const board = createTestBoard();

        // Player 1 has stack at (2,3) with 3 rings
        expect(countPlayerRings(board, 1)).toBe(3);

        // Player 2 has stack at (4,4) with 2 rings
        expect(countPlayerRings(board, 2)).toBe(2);
      });

      it('returns 0 when player has no stacks', () => {
        const board = createTestBoard();

        expect(countPlayerRings(board, 3)).toBe(0);
      });

      it('sums rings across multiple stacks', () => {
        const board = createTestBoard();

        // Add another stack for player 1
        board.stacks.set('6,6', {
          position: { x: 6, y: 6 },
          rings: [1, 1],
          stackHeight: 2,
          capHeight: 2,
          controllingPlayer: 1,
        });

        // Player 1 now has 3 + 2 = 5 rings
        expect(countPlayerRings(board, 1)).toBe(5);
      });
    });
  });

  describe('hexagonal board support', () => {
    it('validates hexagonal positions correctly', () => {
      const board = createTestBoard();
      const view = createBoardView('hexagonal', board);

      // Center position
      expect(view.isValidPosition({ x: 0, y: 0, z: 0 })).toBe(true);

      // Edge positions (hex board has size=13, radius=12)
      expect(view.isValidPosition({ x: 12, y: -12, z: 0 })).toBe(true);
      expect(view.isValidPosition({ x: 0, y: 12, z: -12 })).toBe(true);

      // Out of bounds (radius > 12)
      expect(view.isValidPosition({ x: 13, y: -13, z: 0 })).toBe(false);
    });
  });
});
