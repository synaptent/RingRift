/**
 * BoardManager Unit Tests
 * Tests core board management functionality including position generation,
 * adjacency calculations, and board state operations.
 */

import { BoardManager, __testSetStrictModeOverride } from '../../src/server/game/BoardManager';
import { RingStack } from '../../src/shared/types/game';
import { createTestBoard, addStack, addMarker, addCollapsedSpace, pos } from '../utils/fixtures';

describe('BoardManager', () => {
  describe('Square 8x8 Board', () => {
    let boardManager: BoardManager;

    beforeEach(() => {
      boardManager = new BoardManager('square8');
    });

    describe('Position Generation', () => {
      it('should generate 64 valid positions for 8x8 board', () => {
        const positions = boardManager['validPositions'];
        expect(positions.size).toBe(64);
      });

      it('should include corner positions', () => {
        const positions = boardManager['validPositions'];
        expect(positions.has('0,0')).toBe(true);
        expect(positions.has('0,7')).toBe(true);
        expect(positions.has('7,0')).toBe(true);
        expect(positions.has('7,7')).toBe(true);
      });

      it('should include center position', () => {
        const positions = boardManager['validPositions'];
        expect(positions.has('3,3')).toBe(true);
        expect(positions.has('4,4')).toBe(true);
      });
    });

    describe('isValidPosition', () => {
      it('should return true for valid positions', () => {
        expect(boardManager.isValidPosition(pos(0, 0))).toBe(true);
        expect(boardManager.isValidPosition(pos(7, 7))).toBe(true);
        expect(boardManager.isValidPosition(pos(3, 3))).toBe(true);
      });

      it('should return false for out of bounds positions', () => {
        expect(boardManager.isValidPosition(pos(-1, 0))).toBe(false);
        expect(boardManager.isValidPosition(pos(0, -1))).toBe(false);
        expect(boardManager.isValidPosition(pos(8, 0))).toBe(false);
        expect(boardManager.isValidPosition(pos(0, 8))).toBe(false);
      });

      it('should return false for negative coordinates', () => {
        expect(boardManager.isValidPosition(pos(-1, -1))).toBe(false);
      });
    });

    describe('Moore Adjacency (Movement)', () => {
      it('should return 8 neighbors for center position', () => {
        const neighbors = boardManager.getAdjacentPositions(pos(3, 3), 'moore');
        expect(neighbors).toHaveLength(8);
      });

      it('should return 3 neighbors for corner position', () => {
        const neighbors = boardManager.getAdjacentPositions(pos(0, 0), 'moore');
        expect(neighbors).toHaveLength(3);
      });

      it('should include diagonal neighbors', () => {
        const neighbors = boardManager.getAdjacentPositions(pos(3, 3), 'moore');
        const neighborStrings = neighbors.map((p) => `${p.x},${p.y}`);

        // Check all 8 directions
        expect(neighborStrings).toContain('2,2'); // NW
        expect(neighborStrings).toContain('3,2'); // N
        expect(neighborStrings).toContain('4,2'); // NE
        expect(neighborStrings).toContain('4,3'); // E
        expect(neighborStrings).toContain('4,4'); // SE
        expect(neighborStrings).toContain('3,4'); // S
        expect(neighborStrings).toContain('2,4'); // SW
        expect(neighborStrings).toContain('2,3'); // W
      });
    });

    describe('Edge Detection', () => {
      it('should identify edge positions', () => {
        expect(boardManager.isOnEdge(pos(0, 0))).toBe(true);
        expect(boardManager.isOnEdge(pos(0, 3))).toBe(true);
        expect(boardManager.isOnEdge(pos(7, 7))).toBe(true);
      });

      it('should identify non-edge positions', () => {
        expect(boardManager.isOnEdge(pos(1, 1))).toBe(false);
        expect(boardManager.isOnEdge(pos(3, 3))).toBe(false);
      });

      it('should return all edge positions', () => {
        const edgePositions = boardManager.getEdgePositions();
        expect(edgePositions.length).toBe(28); // 8*4 - 4 corners counted once
      });
    });

    describe('Stack Operations', () => {
      let board: ReturnType<typeof createTestBoard>;

      beforeEach(() => {
        board = createTestBoard('square8');
      });

      it('should retrieve stack at position', () => {
        addStack(board, pos(3, 3), 1, 2);
        const stack = boardManager.getStack(pos(3, 3), board);

        expect(stack).toBeDefined();
        expect(stack?.stackHeight).toBe(2);
        expect(stack?.controllingPlayer).toBe(1);
      });

      it('should return undefined for empty position', () => {
        const stack = boardManager.getStack(pos(3, 3), board);
        expect(stack).toBeUndefined();
      });

      it('should remove stack from position', () => {
        addStack(board, pos(3, 3), 1);
        boardManager.removeStack(pos(3, 3), board);

        const stack = boardManager.getStack(pos(3, 3), board);
        expect(stack).toBeUndefined();
      });
    });

    describe('Marker Operations', () => {
      let board: ReturnType<typeof createTestBoard>;

      beforeEach(() => {
        board = createTestBoard('square8');
      });

      it('should set and retrieve marker', () => {
        boardManager.setMarker(pos(2, 2), 1, board);
        const markerPlayer = boardManager.getMarker(pos(2, 2), board);

        expect(markerPlayer).toBeDefined();
        expect(markerPlayer).toBe(1);
      });

      it('should flip marker to different player', () => {
        addMarker(board, pos(2, 2), 1);
        boardManager.flipMarker(pos(2, 2), 2, board);

        const markerPlayer = boardManager.getMarker(pos(2, 2), board);
        expect(markerPlayer).toBe(2);
      });

      it('should collapse marker and create collapsed space', () => {
        addMarker(board, pos(2, 2), 1);
        boardManager.collapseMarker(pos(2, 2), 1, board);

        const markerPlayer = boardManager.getMarker(pos(2, 2), board);
        const collapsed = boardManager.getCollapsedSpace(pos(2, 2), board);

        expect(markerPlayer).toBeUndefined();
        expect(collapsed).toBe(1);
      });

      it('should remove marker', () => {
        addMarker(board, pos(2, 2), 1);
        boardManager.removeMarker(pos(2, 2), board);

        const markerPlayer = boardManager.getMarker(pos(2, 2), board);
        expect(markerPlayer).toBeUndefined();
      });
    });

    describe('Collapsed Space Operations', () => {
      let board: ReturnType<typeof createTestBoard>;

      beforeEach(() => {
        board = createTestBoard('square8');
      });

      it('should set and retrieve collapsed space', () => {
        boardManager.setCollapsedSpace(pos(3, 3), 1, board);
        const player = boardManager.getCollapsedSpace(pos(3, 3), board);

        expect(player).toBe(1);
      });

      it('should identify collapsed space', () => {
        addCollapsedSpace(board, pos(3, 3), 1);
        expect(boardManager.isCollapsedSpace(pos(3, 3), board)).toBe(true);
      });

      it('should return false for non-collapsed space', () => {
        expect(boardManager.isCollapsedSpace(pos(3, 3), board)).toBe(false);
      });
    });

    describe('Board invariants', () => {
      let board: ReturnType<typeof createTestBoard>;

      beforeEach(() => {
        board = createTestBoard('square8');
      });

      it('should not throw for a clean board state when using core mutators', () => {
        const stack: RingStack = {
          position: pos(0, 0),
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        };

        expect(() => {
          boardManager.setStack(pos(0, 0), stack, board);
          boardManager.setCollapsedSpace(pos(1, 1), 1, board);
          boardManager.collapseMarker(pos(2, 2), 1, board);
        }).not.toThrow();

        // Legal trajectories built from core mutators must never rely on the
        // defensive repair pass.
        expect(boardManager.getRepairCountForTesting()).toBe(0);
      });

      it('logs but does not throw for stack+collapsed overlap in non-strict mode', () => {
        // Temporarily disable strict mode for this test to verify logging
        // behaviour in non-strict (production-like) environments.
        __testSetStrictModeOverride(false);

        try {
          // Create an illegal state directly on the BoardState
          addStack(board, pos(1, 1), 1, 1);
          addCollapsedSpace(board, pos(1, 1), 1);

          const safeStack: RingStack = {
            position: pos(0, 0),
            rings: [1],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 1,
          };

          // In non-strict mode, invariant violations are logged but not thrown.
          // The stack+collapsed overlap is detected (error logged) but not
          // auto-repaired, unlike marker+collapsed overlaps.
          const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

          expect(() => {
            boardManager.setStack(pos(0, 0), safeStack, board);
          }).not.toThrow();

          // Verify that an invariant violation was logged
          expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('invariant violation'));

          consoleSpy.mockRestore();
        } finally {
          // Restore strict mode for subsequent tests
          __testSetStrictModeOverride(undefined);
        }
      });

      it('repairs a marker+collapsed overlap elsewhere on the board when touched by a core mutator', () => {
        // Construct an illegal marker + collapsed overlap
        addMarker(board, pos(2, 2), 1);
        addCollapsedSpace(board, pos(2, 2), 1);

        const safeStack: RingStack = {
          position: pos(0, 0),
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        };

        const key = '2,2';
        const beforeRepairs = boardManager.getRepairCountForTesting();

        expect(() => {
          // Any core mutator that triggers invariant checks must first repair
          // the illegal marker+collapsed overlap, but this repair should not
          // surface as a hard invariant failure.
          boardManager.setStack(pos(0, 0), safeStack, board);
        }).not.toThrow();

        const afterRepairs = boardManager.getRepairCountForTesting();
        expect(afterRepairs).toBe(beforeRepairs + 1);

        // The marker at the illegal cell has been removed, while the collapsed
        // territory remains.
        expect(board.markers.has(key)).toBe(false);
        expect(board.collapsedSpaces.has(key)).toBe(true);
      });

      it('should repair and log when placing a stack on a marker while still enforcing invariants', () => {
        // Marker at destination but no collapsed spaces
        addMarker(board, pos(3, 3), 1);

        const stack: RingStack = {
          position: pos(3, 3),
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        };

        const beforeRepairs = boardManager.getRepairCountForTesting();

        // The implementation removes the marker before asserting invariants,
        // so the resulting state is legal and no error is thrown, but the
        // invariant helper still protects against wider board corruption.
        expect(() => {
          boardManager.setStack(pos(3, 3), stack, board);
        }).not.toThrow();

        const afterRepairs = boardManager.getRepairCountForTesting();

        // This path is expected to trigger exactly one additional repair:
        // the pre-existing marker on the landing cell is removed before the
        // stack is placed.
        expect(afterRepairs).toBe(beforeRepairs + 1);
      });

      it('repairs a directly injected stack+marker overlap via assertBoardInvariants and counts the repair', () => {
        const illegalPos = pos(4, 4);
        const key = `${illegalPos.x},${illegalPos.y}`;

        // Construct an illegal state directly on the BoardState: both a stack
        // and a marker on the same cell. This bypasses the normal mutators so
        // we exercise the defensive repair pass inside assertBoardInvariants.
        addStack(board, illegalPos, 1, 1);
        addMarker(board, illegalPos, 1);

        expect(board.stacks.has(key)).toBe(true);
        expect(board.markers.has(key)).toBe(true);

        const beforeRepairs = boardManager.getRepairCountForTesting();

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (boardManager as any).assertBoardInvariants(board as any, 'test_stack_marker_overlap');

        // After the repair pass, the overlap is resolved and the repair counter
        // has been incremented exactly once.
        expect(board.stacks.has(key)).toBe(true);
        expect(board.markers.has(key)).toBe(false);

        const afterRepairs = boardManager.getRepairCountForTesting();
        expect(afterRepairs).toBe(beforeRepairs + 1);
      });

      it('repairs a directly injected marker+collapsed overlap via assertBoardInvariants and counts the repair', () => {
        const illegalPos = pos(5, 5);
        const key = `${illegalPos.x},${illegalPos.y}`;

        addMarker(board, illegalPos, 1);
        addCollapsedSpace(board, illegalPos, 1);

        expect(board.markers.has(key)).toBe(true);
        expect(board.collapsedSpaces.has(key)).toBe(true);

        const beforeRepairs = boardManager.getRepairCountForTesting();

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (boardManager as any).assertBoardInvariants(board as any, 'test_marker_collapsed_overlap');

        expect(board.markers.has(key)).toBe(false);
        expect(board.collapsedSpaces.has(key)).toBe(true);

        const afterRepairs = boardManager.getRepairCountForTesting();
        expect(afterRepairs).toBe(beforeRepairs + 1);
      });
    });
  });

  describe('Square 19x19 Board', () => {
    let boardManager: BoardManager;

    beforeEach(() => {
      boardManager = new BoardManager('square19');
    });

    describe('Position Generation', () => {
      it('should generate 361 valid positions for 19x19 board', () => {
        const positions = boardManager['validPositions'];
        expect(positions.size).toBe(361);
      });
    });

    describe('Von Neumann Adjacency (Territory)', () => {
      it('should return 4 neighbors for center position', () => {
        const neighbors = boardManager.getAdjacentPositions(pos(9, 9), 'von_neumann');
        expect(neighbors).toHaveLength(4);
      });

      it('should only include orthogonal neighbors', () => {
        const neighbors = boardManager.getAdjacentPositions(pos(9, 9), 'von_neumann');
        const neighborStrings = neighbors.map((p) => `${p.x},${p.y}`);

        // Check 4 orthogonal directions only
        expect(neighborStrings).toContain('9,8'); // N
        expect(neighborStrings).toContain('10,9'); // E
        expect(neighborStrings).toContain('9,10'); // S
        expect(neighborStrings).toContain('8,9'); // W

        // Should NOT include diagonals
        expect(neighborStrings).not.toContain('8,8'); // NW
        expect(neighborStrings).not.toContain('10,8'); // NE
        expect(neighborStrings).not.toContain('10,10'); // SE
        expect(neighborStrings).not.toContain('8,10'); // SW
      });

      it('should return 2 neighbors for corner position', () => {
        const neighbors = boardManager.getAdjacentPositions(pos(0, 0), 'von_neumann');
        expect(neighbors).toHaveLength(2);
      });
    });
  });

  describe('Hexagonal Board', () => {
    let boardManager: BoardManager;

    beforeEach(() => {
      boardManager = new BoardManager('hexagonal');
    });

    describe('Position Generation', () => {
      it('should generate 331 valid positions for hexagonal board', () => {
        const positions = boardManager['validPositions'];
        expect(positions.size).toBe(331);
      });

      it('should include center position', () => {
        const positions = boardManager['validPositions'];
        expect(positions.has('0,0,0')).toBe(true);
      });

      it('should validate cube coordinate constraint', () => {
        const positions = Array.from(boardManager['validPositions'].values());

        positions.forEach((posStr) => {
          const parts = posStr.split(',').map(Number);
          if (parts.length === 3) {
            const [x, y, z] = parts;
            expect(x + y + z).toBe(0);
          }
        });
      });
    });

    describe('Hexagonal Adjacency', () => {
      it('should return 6 neighbors for center position', () => {
        const neighbors = boardManager.getAdjacentPositions(pos(0, 0, 0), 'hexagonal');
        expect(neighbors).toHaveLength(6);
      });

      it('should include all 6 hexagonal directions', () => {
        const neighbors = boardManager.getAdjacentPositions(pos(0, 0, 0), 'hexagonal');
        const neighborStrings = neighbors.map((p) => `${p.x},${p.y},${p.z}`);

        // Check all 6 hexagonal directions
        expect(neighborStrings).toContain('1,0,-1'); // E
        expect(neighborStrings).toContain('-1,0,1'); // W
        expect(neighborStrings).toContain('0,1,-1'); // NE
        expect(neighborStrings).toContain('0,-1,1'); // SW
        expect(neighborStrings).toContain('1,-1,0'); // SE
        expect(neighborStrings).toContain('-1,1,0'); // NW
      });

      it('should have fewer neighbors for edge positions', () => {
        const edgePos = pos(10, 0, -10); // Edge position
        const neighbors = boardManager.getAdjacentPositions(edgePos, 'hexagonal');
        expect(neighbors.length).toBeLessThan(6);
      });
    });

    describe('Edge Detection', () => {
      it('should identify center as not on edge', () => {
        expect(boardManager.isOnEdge(pos(0, 0, 0))).toBe(false);
      });

      it('should identify edge positions correctly', () => {
        expect(boardManager.isOnEdge(pos(10, 0, -10))).toBe(true);
        expect(boardManager.isOnEdge(pos(-10, 10, 0))).toBe(true);
      });

      it('should use radius = size - 1', () => {
        // Size is 11, so radius should be 10
        expect(boardManager.isOnEdge(pos(10, 0, -10))).toBe(true);
        expect(boardManager.isOnEdge(pos(11, 0, -11))).toBe(false); // Beyond board
      });
    });
  });

  describe('Additional Board Operations', () => {
    let boardManager: BoardManager;
    let board: ReturnType<typeof createTestBoard>;

    beforeEach(() => {
      boardManager = new BoardManager('square8');
      board = createTestBoard('square8');
    });

    describe('createBoard', () => {
      it('should create empty board with correct structure', () => {
        const newBoard = boardManager.createBoard();

        expect(newBoard.stacks.size).toBe(0);
        expect(newBoard.markers.size).toBe(0);
        expect(newBoard.collapsedSpaces.size).toBe(0);
        expect(newBoard.territories.size).toBe(0);
        expect(newBoard.formedLines).toEqual([]);
        expect(newBoard.size).toBe(8);
        expect(newBoard.type).toBe('square8');
      });

      it('should reset repair count when creating new board', () => {
        // Create some state that increments repair count
        addMarker(board, pos(0, 0), 1);
        addStack(board, pos(0, 0), 1, 1);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (boardManager as any).assertBoardInvariants(board, 'test');

        expect(boardManager.getRepairCountForTesting()).toBeGreaterThan(0);

        // Create new board should reset
        boardManager.createBoard();
        expect(boardManager.getRepairCountForTesting()).toBe(0);
      });
    });

    describe('calculateDistance', () => {
      it('should calculate distance for adjacent positions', () => {
        const dist = boardManager.calculateDistance(pos(3, 3), pos(4, 4));
        expect(dist).toBe(1); // Chebyshev distance
      });

      it('should calculate distance for orthogonal positions', () => {
        const dist = boardManager.calculateDistance(pos(0, 0), pos(0, 5));
        expect(dist).toBe(5);
      });

      it('should calculate distance for diagonal positions', () => {
        const dist = boardManager.calculateDistance(pos(0, 0), pos(3, 3));
        expect(dist).toBe(3); // Chebyshev distance = max(|dx|, |dy|)
      });

      it('should return 0 for same position', () => {
        const dist = boardManager.calculateDistance(pos(3, 3), pos(3, 3));
        expect(dist).toBe(0);
      });
    });

    describe('hasPlayerStack', () => {
      it('should return true when player has stack at position', () => {
        addStack(board, pos(3, 3), 1, 2);
        expect(boardManager.hasPlayerStack(pos(3, 3), 1, board)).toBe(true);
      });

      it('should return false when other player has stack', () => {
        addStack(board, pos(3, 3), 2, 2);
        expect(boardManager.hasPlayerStack(pos(3, 3), 1, board)).toBe(false);
      });

      it('should return false for empty position', () => {
        expect(boardManager.hasPlayerStack(pos(3, 3), 1, board)).toBe(false);
      });
    });

    describe('getPlayerStackPositions', () => {
      it('should return positions of all player stacks', () => {
        addStack(board, pos(0, 0), 1, 1);
        addStack(board, pos(1, 1), 1, 1);
        addStack(board, pos(2, 2), 2, 1);

        const positions = boardManager.getPlayerStackPositions(1, board);

        expect(positions).toHaveLength(2);
        const posStrings = positions.map((p) => `${p.x},${p.y}`);
        expect(posStrings).toContain('0,0');
        expect(posStrings).toContain('1,1');
      });

      it('should return empty array when player has no stacks', () => {
        addStack(board, pos(0, 0), 2, 1);

        const positions = boardManager.getPlayerStackPositions(1, board);
        expect(positions).toEqual([]);
      });
    });

    describe('getPlayerStacks', () => {
      it('should return all stacks for a player', () => {
        addStack(board, pos(0, 0), 1, 2);
        addStack(board, pos(1, 1), 1, 3);
        addStack(board, pos(2, 2), 2, 1);

        const stacks = boardManager.getPlayerStacks(board, 1);

        expect(stacks).toHaveLength(2);
        expect(stacks.every((s) => s.controllingPlayer === 1)).toBe(true);
      });
    });

    describe('getAllStackPositions', () => {
      it('should return positions of all stacks regardless of player', () => {
        addStack(board, pos(0, 0), 1, 1);
        addStack(board, pos(3, 3), 2, 1);

        const positions = boardManager.getAllStackPositions(board);

        expect(positions).toHaveLength(2);
      });

      it('should return empty array for empty board', () => {
        const positions = boardManager.getAllStackPositions(board);
        expect(positions).toEqual([]);
      });
    });

    describe('getAllPositions', () => {
      it('should return all valid positions', () => {
        const positions = boardManager.getAllPositions();
        expect(positions).toHaveLength(64); // 8x8 board
      });
    });

    describe('getCenterPositions', () => {
      it('should return positions near center', () => {
        const centerPositions = boardManager.getCenterPositions();

        // All returned positions should be within center range
        const center = 4; // floor(8/2)
        for (const p of centerPositions) {
          const xDist = Math.abs(p.x - center);
          const yDist = Math.abs(p.y - center);
          expect(xDist <= 2).toBe(true);
          expect(yDist <= 2).toBe(true);
        }
      });
    });

    describe('isInCenter', () => {
      it('should return true for center position', () => {
        expect(boardManager.isInCenter(pos(4, 4))).toBe(true);
      });

      it('should return true for positions near center', () => {
        expect(boardManager.isInCenter(pos(3, 3))).toBe(true);
        expect(boardManager.isInCenter(pos(5, 5))).toBe(true);
      });

      it('should return false for corner positions', () => {
        expect(boardManager.isInCenter(pos(0, 0))).toBe(false);
        expect(boardManager.isInCenter(pos(7, 7))).toBe(false);
      });
    });

    describe('getConfig', () => {
      it('should return board configuration', () => {
        const config = boardManager.getConfig();

        expect(config).toBeDefined();
        expect(config.size).toBe(8);
      });
    });

    describe('getNeighbors', () => {
      it('should use default movement adjacency when not specified', () => {
        const neighbors = boardManager.getNeighbors(pos(3, 3));
        expect(neighbors.length).toBeGreaterThan(0);
      });

      it('should respect specified adjacency type', () => {
        const mooreNeighbors = boardManager.getNeighbors(pos(3, 3), 'moore');
        const vonNeumannNeighbors = boardManager.getNeighbors(pos(3, 3), 'von_neumann');

        expect(mooreNeighbors).toHaveLength(8);
        expect(vonNeumannNeighbors).toHaveLength(4);
      });
    });

    describe('setMarker special cases', () => {
      it('should not place marker on collapsed space', () => {
        addCollapsedSpace(board, pos(2, 2), 1);
        boardManager.setMarker(pos(2, 2), 2, board);

        expect(boardManager.getMarker(pos(2, 2), board)).toBeUndefined();
      });

      it('should remove existing stack when placing marker', () => {
        addStack(board, pos(2, 2), 1, 1);
        boardManager.setMarker(pos(2, 2), 2, board);

        expect(boardManager.getStack(pos(2, 2), board)).toBeUndefined();
        expect(boardManager.getMarker(pos(2, 2), board)).toBe(2);
      });
    });

    describe('flipMarker edge cases', () => {
      it('should not flip marker if same player', () => {
        addMarker(board, pos(2, 2), 1);
        boardManager.flipMarker(pos(2, 2), 1, board);

        expect(boardManager.getMarker(pos(2, 2), board)).toBe(1);
      });

      it('should not flip if no marker exists', () => {
        boardManager.flipMarker(pos(2, 2), 1, board);
        expect(boardManager.getMarker(pos(2, 2), board)).toBeUndefined();
      });
    });
  });

  describe('Territory Operations', () => {
    let boardManager: BoardManager;
    let board: ReturnType<typeof createTestBoard>;

    beforeEach(() => {
      boardManager = new BoardManager('square8');
      board = createTestBoard('square8');
    });

    describe('findConnectedTerritory', () => {
      it('should find connected stack positions for a player', () => {
        // Create a connected group of stacks
        addStack(board, pos(3, 3), 1, 1);
        addStack(board, pos(3, 4), 1, 1);
        addStack(board, pos(4, 3), 1, 1);

        const territory = boardManager.findConnectedTerritory(pos(3, 3), 1, board);

        expect(territory.size).toBeGreaterThanOrEqual(1);
        expect(territory.has('3,3')).toBe(true);
      });

      it('should not include disconnected stacks', () => {
        addStack(board, pos(0, 0), 1, 1);
        addStack(board, pos(7, 7), 1, 1);

        const territory = boardManager.findConnectedTerritory(pos(0, 0), 1, board);

        expect(territory.has('0,0')).toBe(true);
        expect(territory.has('7,7')).toBe(false);
      });

      it('should return empty set for position without player stack', () => {
        addStack(board, pos(3, 3), 2, 1);

        const territory = boardManager.findConnectedTerritory(pos(3, 3), 1, board);
        expect(territory.size).toBe(0);
      });
    });

    describe('findAllTerritories', () => {
      it('should find all separate territories for a player', () => {
        // Create two disconnected territories
        addStack(board, pos(0, 0), 1, 1);
        addStack(board, pos(0, 1), 1, 1);
        addStack(board, pos(7, 7), 1, 1);

        const territories = boardManager.findAllTerritories(1, board);

        expect(territories.length).toBeGreaterThanOrEqual(1);
        territories.forEach((t) => expect(t.controllingPlayer).toBe(1));
      });

      it('should return empty array for player with no stacks', () => {
        addStack(board, pos(0, 0), 2, 1);

        const territories = boardManager.findAllTerritories(1, board);
        expect(territories).toEqual([]);
      });
    });

    describe('findAllTerritoriesForAllPlayers', () => {
      it('should find territories for all players', () => {
        addStack(board, pos(0, 0), 1, 1);
        addStack(board, pos(7, 7), 2, 1);

        const territories = boardManager.findAllTerritoriesForAllPlayers(board);

        expect(territories.length).toBeGreaterThanOrEqual(2);
        const players = new Set(territories.map((t) => t.controllingPlayer));
        expect(players.has(1)).toBe(true);
        expect(players.has(2)).toBe(true);
      });
    });

    describe('findPlayerTerritories', () => {
      it('should find territories for specified player', () => {
        addStack(board, pos(0, 0), 1, 1);
        addStack(board, pos(7, 7), 2, 1);

        const territories = boardManager.findPlayerTerritories(board, 1);

        expect(territories.every((t) => t.controllingPlayer === 1)).toBe(true);
      });
    });

    describe('findDisconnectedRegions', () => {
      it('should delegate to shared helper and return array', () => {
        const regions = boardManager.findDisconnectedRegions(board, 1);
        expect(Array.isArray(regions)).toBe(true);
      });
    });

    describe('getBorderMarkerPositions', () => {
      it('should return border markers for a region', () => {
        // Create a region with adjacent markers
        addStack(board, pos(3, 3), 1, 1);
        addMarker(board, pos(2, 3), 2);
        addMarker(board, pos(4, 3), 2);

        const regionSpaces = [pos(3, 3)];
        const borders = boardManager.getBorderMarkerPositions(regionSpaces, board);

        expect(Array.isArray(borders)).toBe(true);
      });

      it('should return empty array for region with no border markers', () => {
        addStack(board, pos(3, 3), 1, 1);

        const regionSpaces = [pos(3, 3)];
        const borders = boardManager.getBorderMarkerPositions(regionSpaces, board);

        expect(borders).toEqual([]);
      });
    });
  });

  describe('Line Detection', () => {
    let boardManager: BoardManager;
    let board: ReturnType<typeof createTestBoard>;

    beforeEach(() => {
      boardManager = new BoardManager('square8');
      board = createTestBoard('square8');
    });

    describe('findLinesFromPosition', () => {
      it('should return empty array for position without marker', () => {
        const lines = boardManager.findLinesFromPosition(pos(3, 3), board);
        expect(lines).toEqual([]);
      });

      it('should find lines containing the position', () => {
        // Create a horizontal line of markers (4+ needed for line)
        for (let x = 0; x < 5; x++) {
          addMarker(board, pos(x, 3), 1);
        }

        const lines = boardManager.findLinesFromPosition(pos(2, 3), board);

        expect(lines.length).toBeGreaterThanOrEqual(1);
      });
    });

    describe('findAllLines', () => {
      it('should return empty array for board with no markers', () => {
        const lines = boardManager.findAllLines(board);
        expect(lines).toEqual([]);
      });

      it('should find lines on board', () => {
        // Create a vertical line of 4+ markers
        for (let y = 0; y < 5; y++) {
          addMarker(board, pos(3, y), 1);
        }

        const lines = boardManager.findAllLines(board);

        expect(lines.length).toBeGreaterThanOrEqual(1);
      });
    });

    describe('debugFindAllLines', () => {
      it('should return object with keys array', () => {
        const result = boardManager.debugFindAllLines(board);

        expect(result).toBeDefined();
        expect(Array.isArray(result.keys)).toBe(true);
      });
    });
  });

  describe('Pathfinding', () => {
    let boardManager: BoardManager;

    beforeEach(() => {
      boardManager = new BoardManager('square8');
    });

    describe('findPath', () => {
      it('should find path between adjacent positions', () => {
        const path = boardManager.findPath(pos(0, 0), pos(1, 0), new Set());

        expect(path).not.toBeNull();
        expect(path!.length).toBe(2);
      });

      it('should return null when path is blocked', () => {
        // Create wall of obstacles
        const obstacles = new Set(['1,0', '1,1', '1,2', '0,1', '0,2']);
        const path = boardManager.findPath(pos(0, 0), pos(2, 0), obstacles);

        expect(path).toBeNull();
      });

      it('should find path around obstacles', () => {
        const obstacles = new Set(['1,0']);
        const path = boardManager.findPath(pos(0, 0), pos(2, 0), obstacles);

        expect(path).not.toBeNull();
        // Path should go around the obstacle
        expect(path!.some((p) => p.x === 1 && p.y === 0)).toBe(false);
      });

      it('should find path to same position', () => {
        const path = boardManager.findPath(pos(3, 3), pos(3, 3), new Set());

        expect(path).not.toBeNull();
        expect(path!.length).toBe(1);
      });

      it('should find diagonal path when allowed by Moore adjacency', () => {
        const path = boardManager.findPath(pos(0, 0), pos(3, 3), new Set());

        expect(path).not.toBeNull();
        // With Moore adjacency, diagonal movement means max 4 steps
        expect(path!.length).toBeLessThanOrEqual(4);
      });
    });
  });

  describe('Hexagonal Board Additional Tests', () => {
    let boardManager: BoardManager;

    beforeEach(() => {
      boardManager = new BoardManager('hexagonal');
    });

    describe('calculateDistance hexagonal', () => {
      it('should calculate hexagonal distance for adjacent positions', () => {
        const dist = boardManager.calculateDistance(pos(0, 0, 0), pos(1, -1, 0));
        expect(dist).toBe(1);
      });

      it('should calculate hexagonal distance for far positions', () => {
        const dist = boardManager.calculateDistance(pos(0, 0, 0), pos(5, -3, -2));
        expect(dist).toBe(5); // max(|5|, |-3|, |-2|)
      });
    });

    describe('isInCenter hexagonal', () => {
      it('should return true for center position', () => {
        expect(boardManager.isInCenter(pos(0, 0, 0))).toBe(true);
      });

      it('should return true for positions within central area', () => {
        expect(boardManager.isInCenter(pos(1, -1, 0))).toBe(true);
        expect(boardManager.isInCenter(pos(2, -1, -1))).toBe(true);
      });

      it('should return false for positions far from center', () => {
        expect(boardManager.isInCenter(pos(5, -5, 0))).toBe(false);
      });
    });

    describe('getCenterPositions hexagonal', () => {
      it('should return hexagonal center positions', () => {
        const centers = boardManager.getCenterPositions();

        expect(centers.length).toBeGreaterThan(0);
        // All should be within distance 2 from origin
        for (const p of centers) {
          const dist = Math.max(Math.abs(p.x), Math.abs(p.y), Math.abs(p.z || 0));
          expect(dist).toBeLessThanOrEqual(2);
        }
      });
    });

    describe('getEdgePositions hexagonal', () => {
      it('should return hexagonal edge positions', () => {
        const edges = boardManager.getEdgePositions();

        expect(edges.length).toBeGreaterThan(0);
        // All edge positions should be at radius 10
        for (const p of edges) {
          const dist = Math.max(Math.abs(p.x), Math.abs(p.y), Math.abs(p.z || 0));
          expect(dist).toBe(10);
        }
      });
    });
  });
});
