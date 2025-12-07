/**
 * BoardManager Branch Coverage Tests
 *
 * This file targets uncovered branches in BoardManager.ts to increase
 * overall branch coverage towards the 70% target.
 *
 * Key areas tested:
 * - Board invariant repair paths
 * - Different adjacency types
 * - flipMarker edge cases
 * - setStack with marker conflict
 * - A* pathfinding edge cases
 * - Hexagonal board specifics
 * - Line detection branches
 */

import { BoardManager, __testSetStrictModeOverride } from '../../src/server/game/BoardManager';
import { createTestBoard, addStack, addMarker, addCollapsedSpace, pos } from '../utils/fixtures';
import { positionToString } from '../../src/shared/engine';

describe('BoardManager Branch Coverage', () => {
  beforeEach(() => {
    __testSetStrictModeOverride(false);
  });

  afterEach(() => {
    __testSetStrictModeOverride(undefined);
  });

  describe('Board invariants strict mode', () => {
    it('throws when stack exists on collapsed space in strict mode', () => {
      const boardManager = new BoardManager('square8');
      const board = boardManager.createBoard();

      // Enable strict invariant mode for this test only.
      __testSetStrictModeOverride(true);

      const position = pos(2, 2);
      const key = positionToString(position);

      // Manually create an illegal state: stack and collapsed space on the same cell.
      (board.stacks as any).set(key, {
        position,
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });
      (board.collapsedSpaces as any).set(key, 1);

      // Call the private invariant checker via an any cast to ensure the
      // strict branch and error path are exercised.
      expect(() =>
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (boardManager as any).assertBoardInvariants(board, 'test-strict')
      ).toThrow(/\[BoardManager] invariant violation/);
    });

    it('does not throw when strict mode is explicitly disabled', () => {
      // This test verifies the non-strict code path where board invariant
      // violations are logged but do not throw. This exercises the branch
      // where isBoardInvariantsStrict() returns false.
      __testSetStrictModeOverride(false);

      const boardManager = new BoardManager('square8');
      const board = boardManager.createBoard();

      const position = pos(1, 1);
      const key = positionToString(position);

      (board.stacks as any).set(key, {
        position,
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });
      (board.collapsedSpaces as any).set(key, 1);

      // Should NOT throw when strict mode is disabled
      expect(() =>
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (boardManager as any).assertBoardInvariants(board, 'non-strict')
      ).not.toThrow();
    });
  });

  describe('Board Invariant Repair Paths', () => {
    it('should repair stack+marker overlap when setting stack on marker', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // First add a marker
      addMarker(board, pos(3, 3), 1);
      expect(boardManager.getMarker(pos(3, 3), board)).toBe(1);

      // Now set a stack at the same position - should trigger repair
      const stack = {
        stackHeight: 3,
        controllingPlayer: 2,
        position: pos(3, 3),
      };
      boardManager.setStack(pos(3, 3), stack, board);

      // Marker should be removed (repaired)
      expect(boardManager.getMarker(pos(3, 3), board)).toBeUndefined();
      // Stack should exist
      expect(boardManager.getStack(pos(3, 3), board)).toBeDefined();
      expect(boardManager.getStack(pos(3, 3), board)?.controllingPlayer).toBe(2);
    });

    it('should repair marker when setting marker on stack', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // First add a stack
      addStack(board, pos(4, 4), 1, 3);
      expect(boardManager.getStack(pos(4, 4), board)).toBeDefined();

      // Now set a marker at the same position - should trigger repair
      boardManager.setMarker(pos(4, 4), 2, board);

      // Stack should be removed (repaired)
      expect(boardManager.getStack(pos(4, 4), board)).toBeUndefined();
      // Marker should exist
      expect(boardManager.getMarker(pos(4, 4), board)).toBe(2);
    });

    it('should handle assertBoardInvariants during collapseMarker', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Add a marker to collapse
      addMarker(board, pos(2, 2), 1);

      // Collapse the marker
      boardManager.collapseMarker(pos(2, 2), 1, board);

      // Should be collapsed space now
      expect(boardManager.isCollapsedSpace(pos(2, 2), board)).toBe(true);
      expect(boardManager.getMarker(pos(2, 2), board)).toBeUndefined();
    });

    it('should clear marker and stack when setting collapsed space', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Add both marker and stack at positions to be collapsed
      addMarker(board, pos(5, 5), 1);
      addStack(board, pos(6, 6), 2, 2);

      // Collapse with marker
      boardManager.setCollapsedSpace(pos(5, 5), 1, board);
      expect(boardManager.getMarker(pos(5, 5), board)).toBeUndefined();
      expect(boardManager.isCollapsedSpace(pos(5, 5), board)).toBe(true);

      // Collapse with stack
      boardManager.setCollapsedSpace(pos(6, 6), 2, board);
      expect(boardManager.getStack(pos(6, 6), board)).toBeUndefined();
      expect(boardManager.isCollapsedSpace(pos(6, 6), board)).toBe(true);
    });

    it('repairs illegal overlaps and tracks repair count', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      expect(boardManager.getRepairCountForTesting()).toBe(0);

      // stack+marker overlap, which should trigger stack_marker_overlap repair
      const overlapPos = pos(2, 2);
      addStack(board, overlapPos, 1, 1);
      addMarker(board, overlapPos, 1);

      // marker on collapsed space, which should trigger marker_on_collapsed_space repair
      const collapsedPos = pos(3, 3);
      addMarker(board, collapsedPos, 1);
      addCollapsedSpace(board, collapsedPos, 1);

      expect(() =>
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (boardManager as any).assertBoardInvariants(board, 'repair-paths')
      ).not.toThrow();

      // Both repairs should have been recorded.
      expect(boardManager.getRepairCountForTesting()).toBe(2);
    });
  });

  describe('flipMarker Edge Cases', () => {
    it('should do nothing when flipping nonexistent marker', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // No marker at position
      boardManager.flipMarker(pos(3, 3), 1, board);

      // Should remain empty
      expect(boardManager.getMarker(pos(3, 3), board)).toBeUndefined();
    });

    it('should do nothing when flipping marker to same player', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Add marker for player 1
      addMarker(board, pos(3, 3), 1);

      // Try to flip to same player
      boardManager.flipMarker(pos(3, 3), 1, board);

      // Should still be player 1
      expect(boardManager.getMarker(pos(3, 3), board)).toBe(1);
    });

    it('should flip marker when different player', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Add marker for player 1
      addMarker(board, pos(3, 3), 1);

      // Flip to player 2
      boardManager.flipMarker(pos(3, 3), 2, board);

      // Should now be player 2
      expect(boardManager.getMarker(pos(3, 3), board)).toBe(2);
    });
  });

  describe('getNeighbors with Different Adjacency Types', () => {
    it('should return moore neighbors when explicitly requested', () => {
      const boardManager = new BoardManager('square8');
      const neighbors = boardManager.getNeighbors(pos(3, 3), 'moore');

      // Moore has 8 neighbors for center position
      expect(neighbors).toHaveLength(8);

      // Check diagonal neighbors are included
      const neighborKeys = neighbors.map((p) => positionToString(p));
      expect(neighborKeys).toContain('2,2'); // diagonal
      expect(neighborKeys).toContain('4,4'); // diagonal
    });

    it('should return von_neumann neighbors when explicitly requested', () => {
      const boardManager = new BoardManager('square8');
      const neighbors = boardManager.getNeighbors(pos(3, 3), 'von_neumann');

      // Von Neumann has 4 neighbors (orthogonal only)
      expect(neighbors).toHaveLength(4);

      // Check only orthogonal neighbors
      const neighborKeys = neighbors.map((p) => positionToString(p));
      expect(neighborKeys).toContain('3,2'); // N
      expect(neighborKeys).toContain('3,4'); // S
      expect(neighborKeys).toContain('2,3'); // W
      expect(neighborKeys).toContain('4,3'); // E

      // No diagonals
      expect(neighborKeys).not.toContain('2,2');
      expect(neighborKeys).not.toContain('4,4');
    });

    it('should return hexagonal neighbors for hex board', () => {
      const boardManager = new BoardManager('hexagonal');
      const neighbors = boardManager.getNeighbors({ x: 0, y: 0, z: 0 }, 'hexagonal');

      // Hex has 6 neighbors for center
      expect(neighbors).toHaveLength(6);
    });

    it('should return empty array for unknown adjacency type', () => {
      const boardManager = new BoardManager('square8');
      const neighbors = boardManager.getNeighbors(pos(3, 3), 'unknown' as any);

      expect(neighbors).toEqual([]);
    });
  });

  describe('Hexagonal Board Specifics', () => {
    let hexBoardManager: BoardManager;

    beforeEach(() => {
      hexBoardManager = new BoardManager('hexagonal');
    });

    it('should validate hexagonal positions correctly', () => {
      // Center is valid
      expect(hexBoardManager.isValidPosition({ x: 0, y: 0, z: 0 })).toBe(true);

      // Edge positions are valid (radius = 12 for size 13)
      expect(hexBoardManager.isValidPosition({ x: 12, y: -12, z: 0 })).toBe(true);
      expect(hexBoardManager.isValidPosition({ x: -12, y: 12, z: 0 })).toBe(true);

      // Beyond edge is invalid
      expect(hexBoardManager.isValidPosition({ x: 13, y: -13, z: 0 })).toBe(false);
    });

    it('should detect edge positions on hexagonal board', () => {
      // Center is not on edge
      expect(hexBoardManager.isOnEdge({ x: 0, y: 0, z: 0 })).toBe(false);

      // Edge positions (distance = radius)
      expect(hexBoardManager.isOnEdge({ x: 12, y: -12, z: 0 })).toBe(true);
      expect(hexBoardManager.isOnEdge({ x: 6, y: 6, z: -12 })).toBe(true);
    });

    it('should detect center positions on hexagonal board', () => {
      // Center is in center
      expect(hexBoardManager.isInCenter({ x: 0, y: 0, z: 0 })).toBe(true);

      // Near center positions
      expect(hexBoardManager.isInCenter({ x: 1, y: -1, z: 0 })).toBe(true);
      expect(hexBoardManager.isInCenter({ x: 2, y: -2, z: 0 })).toBe(true);

      // Far from center
      expect(hexBoardManager.isInCenter({ x: 5, y: -5, z: 0 })).toBe(false);
    });

    it('should get edge positions for hexagonal board', () => {
      const edgePositions = hexBoardManager.getEdgePositions();

      // All edge positions should have distance = radius (12)
      for (const pos of edgePositions) {
        const distance = Math.max(Math.abs(pos.x), Math.abs(pos.y), Math.abs(pos.z || 0));
        expect(distance).toBe(12);
      }
    });

    it('should get center positions for hexagonal board', () => {
      const centerPositions = hexBoardManager.getCenterPositions();

      // All center positions should have distance <= 2
      for (const pos of centerPositions) {
        const distance = Math.max(Math.abs(pos.x), Math.abs(pos.y), Math.abs(pos.z || 0));
        expect(distance).toBeLessThanOrEqual(2);
      }
    });

    it('should calculate distance correctly for hexagonal board', () => {
      const from = { x: 0, y: 0, z: 0 };
      const to = { x: 5, y: -3, z: -2 };

      const distance = hexBoardManager.calculateDistance(from, to);

      // Hex distance is max of absolute cube coordinates difference
      expect(distance).toBe(5);
    });

    it('should get hexagonal neighbors correctly', () => {
      const neighbors = hexBoardManager.getNeighbors({ x: 0, y: 0, z: 0 });

      // 6 neighbors for hex center
      expect(neighbors.length).toBe(6);

      // Each neighbor should be at distance 1
      for (const neighbor of neighbors) {
        const distance = hexBoardManager.calculateDistance({ x: 0, y: 0, z: 0 }, neighbor);
        expect(distance).toBe(1);
      }
    });
  });

  describe('A* Pathfinding', () => {
    it('should find path when one exists', () => {
      const boardManager = new BoardManager('square8');
      const obstacles = new Set<string>();

      const path = boardManager.findPath(pos(0, 0), pos(3, 3), obstacles);

      expect(path).not.toBeNull();
      expect(path!.length).toBeGreaterThan(0);
      // First position should be start
      expect(path![0]).toEqual(pos(0, 0));
      // Last position should be target
      expect(path![path!.length - 1]).toEqual(pos(3, 3));
    });

    it('should return null when path is blocked', () => {
      const boardManager = new BoardManager('square8');

      // Create a wall of obstacles blocking the path
      const obstacles = new Set<string>();
      for (let y = 0; y < 8; y++) {
        obstacles.add(positionToString(pos(4, y)));
      }

      const path = boardManager.findPath(pos(0, 0), pos(7, 0), obstacles);

      expect(path).toBeNull();
    });

    it('should navigate around obstacles', () => {
      const boardManager = new BoardManager('square8');

      // Partial wall
      const obstacles = new Set<string>();
      for (let y = 0; y < 6; y++) {
        obstacles.add(positionToString(pos(3, y)));
      }

      const path = boardManager.findPath(pos(0, 0), pos(5, 0), obstacles);

      expect(path).not.toBeNull();
      // Path should not include any obstacles
      for (const p of path!) {
        expect(obstacles.has(positionToString(p))).toBe(false);
      }
    });

    it('should find path for adjacent positions', () => {
      const boardManager = new BoardManager('square8');
      const obstacles = new Set<string>();

      const path = boardManager.findPath(pos(3, 3), pos(3, 4), obstacles);

      expect(path).not.toBeNull();
      expect(path!.length).toBe(2);
    });
  });

  describe('getAdjacentPositions Switch Cases', () => {
    it('should handle moore adjacency in switch', () => {
      const boardManager = new BoardManager('square8');
      const adjacent = boardManager.getAdjacentPositions(pos(4, 4), 'moore');

      expect(adjacent).toHaveLength(8);
    });

    it('should handle von_neumann adjacency in switch', () => {
      const boardManager = new BoardManager('square8');
      const adjacent = boardManager.getAdjacentPositions(pos(4, 4), 'von_neumann');

      expect(adjacent).toHaveLength(4);
    });

    it('should handle hexagonal adjacency in switch with z coordinate', () => {
      const boardManager = new BoardManager('hexagonal');
      const adjacent = boardManager.getAdjacentPositions({ x: 0, y: 0, z: 0 }, 'hexagonal');

      expect(adjacent).toHaveLength(6);
    });

    it('should return empty for hexagonal adjacency without z coordinate', () => {
      const boardManager = new BoardManager('square8');
      // Position without z coordinate
      const adjacent = boardManager.getAdjacentPositions(pos(4, 4), 'hexagonal');

      // No z coordinate means hexagonal branch returns empty
      expect(adjacent).toHaveLength(0);
    });
  });

  describe('Line Detection Branches', () => {
    it('should return empty array when no marker at position', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      const lines = boardManager.findLinesFromPosition(pos(3, 3), board);

      expect(lines).toHaveLength(0);
    });

    it('should find lines from marker position', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Create a horizontal line of 5 markers
      for (let x = 0; x < 5; x++) {
        addMarker(board, pos(x, 3), 1);
      }

      const lines = boardManager.findLinesFromPosition(pos(2, 3), board);

      expect(lines.length).toBeGreaterThan(0);
      // Should find a line of length 5
      const fiveLengthLine = lines.find((l) => l.length === 5);
      expect(fiveLengthLine).toBeDefined();
    });

    it('should not count stacks as part of marker lines', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Create markers with a stack in the middle
      addMarker(board, pos(0, 3), 1);
      addMarker(board, pos(1, 3), 1);
      addStack(board, pos(2, 3), 1, 2); // Stack breaks the line
      addMarker(board, pos(3, 3), 1);
      addMarker(board, pos(4, 3), 1);

      // Line from first marker should be length 2
      const lines = boardManager.findLinesFromPosition(pos(0, 3), board);

      if (lines.length > 0) {
        const horizontalLine = lines.find((l) => l.positions.some((p) => p.x === 0 && p.y === 3));
        if (horizontalLine) {
          // Line should be broken by the stack
          expect(horizontalLine.length).toBeLessThan(5);
        }
      }
    });

    it('should not count collapsed spaces as part of marker lines', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Create markers with a collapsed space in the middle
      addMarker(board, pos(0, 4), 1);
      addMarker(board, pos(1, 4), 1);
      addCollapsedSpace(board, pos(2, 4), 1); // Collapsed breaks the line
      addMarker(board, pos(3, 4), 1);
      addMarker(board, pos(4, 4), 1);

      const lines = boardManager.findLinesFromPosition(pos(0, 4), board);

      // Lines should be short (broken by collapsed space)
      for (const line of lines) {
        expect(line.length).toBeLessThanOrEqual(2);
      }
    });

    it('should call debugFindAllLines without error', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Add a full horizontal marker line so debugFindAllLines sees at least one line.
      for (let x = 0; x < 4; x++) {
        addMarker(board, pos(x, 0), 1);
      }

      // Should not throw
      const result = boardManager.debugFindAllLines(board);

      expect(result).toBeDefined();
      expect(result.keys).toBeDefined();
      expect(result.keys.length).toBeGreaterThan(0);
    });
  });

  describe('Territory Detection', () => {
    it('should find connected territory for player', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Add connected stacks
      addStack(board, pos(3, 3), 1, 2);
      addStack(board, pos(3, 4), 1, 2);
      addStack(board, pos(4, 3), 1, 2);

      const territory = boardManager.findConnectedTerritory(pos(3, 3), 1, board);

      expect(territory.size).toBe(3);
    });

    it('should not include opponent stacks in territory', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Add player 1 stacks
      addStack(board, pos(3, 3), 1, 2);
      addStack(board, pos(3, 4), 1, 2);
      // Add player 2 stack adjacent
      addStack(board, pos(4, 3), 2, 2);

      const territory = boardManager.findConnectedTerritory(pos(3, 3), 1, board);

      expect(territory.size).toBe(2);
      expect(territory.has(positionToString(pos(4, 3)))).toBe(false);
    });

    it('should find all territories for all players', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Add stacks for multiple players
      addStack(board, pos(1, 1), 1, 2);
      addStack(board, pos(1, 2), 1, 2);
      addStack(board, pos(5, 5), 2, 2);
      addStack(board, pos(5, 6), 2, 2);

      const territories = boardManager.findAllTerritoriesForAllPlayers(board);

      expect(territories.length).toBe(2);
    });

    it('should return empty for findAllTerritoriesForAllPlayers with no stacks', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      const territories = boardManager.findAllTerritoriesForAllPlayers(board);

      expect(territories).toHaveLength(0);
    });

    it('should find player territories via findPlayerTerritories', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      addStack(board, pos(2, 2), 1, 2);
      addStack(board, pos(2, 3), 1, 2);

      const territories = boardManager.findPlayerTerritories(board, 1);

      expect(territories.length).toBe(1);
      expect(territories[0].controllingPlayer).toBe(1);
    });
  });

  describe('Stack and Player Operations', () => {
    it('should check hasPlayerStack correctly', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      addStack(board, pos(3, 3), 1, 2);

      expect(boardManager.hasPlayerStack(pos(3, 3), 1, board)).toBe(true);
      expect(boardManager.hasPlayerStack(pos(3, 3), 2, board)).toBe(false);
      expect(boardManager.hasPlayerStack(pos(4, 4), 1, board)).toBe(false);
    });

    it('should get all stack positions', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      addStack(board, pos(1, 1), 1, 2);
      addStack(board, pos(2, 2), 2, 2);
      addStack(board, pos(3, 3), 1, 2);

      const positions = boardManager.getAllStackPositions(board);

      expect(positions).toHaveLength(3);
    });

    it('should get player stack positions', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      addStack(board, pos(1, 1), 1, 2);
      addStack(board, pos(2, 2), 2, 2);
      addStack(board, pos(3, 3), 1, 2);

      const player1Positions = boardManager.getPlayerStackPositions(1, board);
      const player2Positions = boardManager.getPlayerStackPositions(2, board);

      expect(player1Positions).toHaveLength(2);
      expect(player2Positions).toHaveLength(1);
    });

    it('should get player stacks', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      addStack(board, pos(1, 1), 1, 2);
      addStack(board, pos(2, 2), 2, 3);
      addStack(board, pos(3, 3), 1, 4);

      const player1Stacks = boardManager.getPlayerStacks(board, 1);
      const player2Stacks = boardManager.getPlayerStacks(board, 2);

      expect(player1Stacks).toHaveLength(2);
      expect(player2Stacks).toHaveLength(1);
      expect(player2Stacks[0].stackHeight).toBe(3);
    });

    it('should remove stack from board', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      addStack(board, pos(2, 2), 1, 2);
      expect(boardManager.getStack(pos(2, 2), board)).toBeDefined();

      boardManager.removeStack(pos(2, 2), board);

      expect(boardManager.getStack(pos(2, 2), board)).toBeUndefined();
    });
  });

  describe('Collapsed Space Operations', () => {
    it('should get collapsed space owner', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      addCollapsedSpace(board, pos(3, 3), 1);
      addCollapsedSpace(board, pos(4, 4), 2);

      expect(boardManager.getCollapsedSpace(pos(3, 3), board)).toBe(1);
      expect(boardManager.getCollapsedSpace(pos(4, 4), board)).toBe(2);
      expect(boardManager.getCollapsedSpace(pos(5, 5), board)).toBeUndefined();
    });

    it('should identify collapsed spaces', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      addCollapsedSpace(board, pos(3, 3), 1);

      expect(boardManager.isCollapsedSpace(pos(3, 3), board)).toBe(true);
      expect(boardManager.isCollapsedSpace(pos(4, 4), board)).toBe(false);
    });
  });

  describe('Distance Calculation', () => {
    it('should calculate distance for square board', () => {
      const boardManager = new BoardManager('square8');

      const distance = boardManager.calculateDistance(pos(0, 0), pos(3, 4));

      // Chebyshev distance (max of dx, dy)
      expect(distance).toBe(4);
    });

    it('should calculate zero distance for same position', () => {
      const boardManager = new BoardManager('square8');

      const distance = boardManager.calculateDistance(pos(3, 3), pos(3, 3));

      expect(distance).toBe(0);
    });
  });

  describe('Board Configuration', () => {
    it('should return board config', () => {
      const boardManager = new BoardManager('square8');
      const config = boardManager.getConfig();

      expect(config).toBeDefined();
      expect(config.movementAdjacency).toBeDefined();
      expect(config.territoryAdjacency).toBeDefined();
    });

    it('should return all positions', () => {
      const boardManager = new BoardManager('square8');
      const positions = boardManager.getAllPositions();

      expect(positions).toHaveLength(64);
    });

    it('should get edge positions for square board', () => {
      const boardManager = new BoardManager('square8');
      const edgePositions = boardManager.getEdgePositions();

      expect(edgePositions.length).toBeGreaterThan(0);
      for (const p of edgePositions) {
        expect(boardManager.isOnEdge(p)).toBe(true);
      }
    });

    it('should get center positions for square board', () => {
      const boardManager = new BoardManager('square8');
      const centerPositions = boardManager.getCenterPositions();

      expect(centerPositions.length).toBeGreaterThan(0);
      for (const p of centerPositions) {
        expect(boardManager.isInCenter(p)).toBe(true);
      }
    });
  });

  describe('Square 19x19 Board', () => {
    it('should handle larger board size', () => {
      const boardManager = new BoardManager('square19');
      const positions = boardManager.getAllPositions();

      expect(positions).toHaveLength(361); // 19 * 19
    });

    it('should detect edge on larger board', () => {
      const boardManager = new BoardManager('square19');

      expect(boardManager.isOnEdge(pos(0, 0))).toBe(true);
      expect(boardManager.isOnEdge(pos(18, 18))).toBe(true);
      expect(boardManager.isOnEdge(pos(9, 9))).toBe(false);
    });

    it('should detect center on larger board', () => {
      const boardManager = new BoardManager('square19');

      expect(boardManager.isInCenter(pos(9, 9))).toBe(true);
      expect(boardManager.isInCenter(pos(0, 0))).toBe(false);
    });
  });

  describe('Marker getMarker with Collapsed Space', () => {
    it('should return marker player correctly', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      addMarker(board, pos(3, 3), 2);
      const result = boardManager.getMarker(pos(3, 3), board);

      expect(result).toBe(2);
    });

    it('should return undefined for position without marker', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      const result = boardManager.getMarker(pos(3, 3), board);

      expect(result).toBeUndefined();
    });

    it('should not place marker on collapsed space', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      const position = pos(4, 4);
      addCollapsedSpace(board, position, 1);

      boardManager.setMarker(position, 2, board);

      expect(boardManager.getMarker(position, board)).toBeUndefined();
      expect(boardManager.isCollapsedSpace(position, board)).toBe(true);
    });
  });

  describe('removeMarker', () => {
    it('should remove marker from board', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      addMarker(board, pos(3, 3), 1);
      expect(boardManager.getMarker(pos(3, 3), board)).toBe(1);

      boardManager.removeMarker(pos(3, 3), board);
      expect(boardManager.getMarker(pos(3, 3), board)).toBeUndefined();
    });

    it('should not error when removing nonexistent marker', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Should not throw
      expect(() => boardManager.removeMarker(pos(3, 3), board)).not.toThrow();
    });
  });

  describe('findDisconnectedRegions', () => {
    it('should delegate to shared helper', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      // Just verify it doesn't throw
      const regions = boardManager.findDisconnectedRegions(board, 1);

      expect(Array.isArray(regions)).toBe(true);
    });
  });

  describe('getBorderMarkerPositions', () => {
    it('should delegate to shared helper', () => {
      const boardManager = new BoardManager('square8');
      const board = createTestBoard('square8');

      const regionSpaces = [pos(3, 3), pos(3, 4), pos(4, 3)];

      // Just verify it returns an array
      const borders = boardManager.getBorderMarkerPositions(regionSpaces, board);

      expect(Array.isArray(borders)).toBe(true);
    });
  });
});
