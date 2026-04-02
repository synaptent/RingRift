/**
 * core.branchCoverage.test.ts
 *
 * Branch coverage tests for core.ts targeting uncovered branches:
 * - getMovementDirectionsForBoardType: hex vs square
 * - countRingsInPlayForPlayer: player not found
 * - calculateCapHeight: empty rings, single ring, multiple same, mixed colors
 * - getPathPositions: z coordinate handling
 * - calculateDistance: hex vs square
 * - validateCaptureSegmentOnBoard: various validation branches
 * - hasAnyLegalMoveOrCaptureFromOnBoard: movement and capture branches
 * - computeProgressSnapshot: eliminatedRings fallback
 * - summarizeBoard: stacks, markers, collapsedSpaces
 * - applyMarkerEffectsAlongPathOnBoard: marker processing branches
 */

import {
  getMovementDirectionsForBoardType,
  countRingsOnBoardForPlayer,
  countRingsInPlayForPlayer,
  calculateCapHeight,
  getPathPositions,
  calculateDistance,
  validateCaptureSegmentOnBoard,
  hasAnyLegalMoveOrCaptureFromOnBoard,
  computeProgressSnapshot,
  summarizeBoard,
  fingerprintGameState,
  hashGameStateSHA256,
  hashGameState,
  applyMarkerEffectsAlongPathOnBoard,
  SQUARE_MOORE_DIRECTIONS,
  HEX_DIRECTIONS,
  CaptureSegmentBoardView,
  MovementBoardView,
  MarkerPathHelpers,
} from '../../src/shared/engine/core';
import type { GameState, RingStack } from '../../src/shared/engine/types';
import type { Position, BoardType, BoardState } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';
import { inferTotalRingsInPlay } from '../utils/fixtures';

// Helper to create a position
const pos = (x: number, y: number, z?: number): Position => {
  const p: Position = { x, y };
  if (z !== undefined) p.z = z;
  return p;
};

// Helper to create a minimal BoardState
function makeBoardState(overrides: Partial<BoardState> = {}): BoardState {
  return {
    type: 'square8' as BoardType,
    size: 8,
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    formedLines: [],
    territories: new Map(),
    eliminatedRings: { 1: 0, 2: 0 },
    ...overrides,
  };
}

// Helper to create a minimal GameState
function makeGameState(overrides: Partial<GameState> = {}): GameState {
  const board = makeBoardState();
  const players = [
    {
      id: 'p1',
      username: 'Player1',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 10,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 10,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  return {
    id: 'test-game',
    boardType: 'square8',
    board,
    players,
    currentPlayer: 1,
    currentPhase: 'ring_placement',
    moveHistory: [],
    history: [],
    gameStatus: 'active',
    winner: undefined,
    timeControl: { initialTime: 600000, increment: 0, type: 'rapid' },
    spectators: [],
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: inferTotalRingsInPlay(players, board),
    totalRingsEliminated: 0,
    victoryThreshold: 15,
    territoryVictoryThreshold: 8,
    ...overrides,
  } as GameState;
}

// Helper to add a stack to the board
function addStack(
  board: BoardState,
  position: Position,
  controllingPlayer: number,
  rings: number[]
): void {
  const key = positionToString(position);
  const stack: RingStack = {
    position,
    rings,
    stackHeight: rings.length,
    capHeight: calculateCapHeight(rings),
    controllingPlayer,
  };
  board.stacks.set(key, stack);
}

// Helper to add a marker to the board
function addMarker(board: BoardState, position: Position, player: number): void {
  const key = positionToString(position);
  board.markers.set(key, {
    position,
    player,
    type: 'regular',
  });
}

// Helper to add collapsed space
function addCollapsed(board: BoardState, position: Position, owner: number): void {
  const key = positionToString(position);
  board.collapsedSpaces.set(key, owner);
}

describe('core.ts branch coverage', () => {
  describe('getMovementDirectionsForBoardType', () => {
    it('returns HEX_DIRECTIONS for hexagonal board', () => {
      const result = getMovementDirectionsForBoardType('hexagonal');
      expect(result).toBe(HEX_DIRECTIONS);
      expect(result).toHaveLength(6);
    });

    it('returns SQUARE_MOORE_DIRECTIONS for square8 board', () => {
      const result = getMovementDirectionsForBoardType('square8');
      expect(result).toBe(SQUARE_MOORE_DIRECTIONS);
      expect(result).toHaveLength(8);
    });

    it('returns SQUARE_MOORE_DIRECTIONS for square10 board', () => {
      const result = getMovementDirectionsForBoardType('square10');
      expect(result).toBe(SQUARE_MOORE_DIRECTIONS);
    });
  });

  describe('countRingsOnBoardForPlayer', () => {
    it('counts rings of player color in single stack', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1, 1, 2]);

      expect(countRingsOnBoardForPlayer(board, 1)).toBe(2);
      expect(countRingsOnBoardForPlayer(board, 2)).toBe(1);
    });

    it('counts rings across multiple stacks', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1, 1]);
      addStack(board, pos(3, 3), 2, [2, 1, 2]);

      expect(countRingsOnBoardForPlayer(board, 1)).toBe(3); // 2 in first + 1 in second
      expect(countRingsOnBoardForPlayer(board, 2)).toBe(2);
    });

    it('returns 0 when no rings of that player', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 2, [2, 2]);

      expect(countRingsOnBoardForPlayer(board, 1)).toBe(0);
    });

    it('returns 0 for empty board', () => {
      const board = makeBoardState();
      expect(countRingsOnBoardForPlayer(board, 1)).toBe(0);
    });
  });

  describe('countRingsInPlayForPlayer', () => {
    it('adds board rings and rings in hand', () => {
      const state = makeGameState();
      addStack(state.board, pos(0, 0), 1, [1, 1]);
      state.players[0].ringsInHand = 5;

      expect(countRingsInPlayForPlayer(state, 1)).toBe(7); // 2 on board + 5 in hand
    });

    it('returns just board rings when player not found', () => {
      const state = makeGameState();
      addStack(state.board, pos(0, 0), 1, [1, 1, 1]);

      // Player 99 doesn't exist, should return just board count
      expect(countRingsInPlayForPlayer(state, 99)).toBe(0);
    });

    it('handles player with no rings in hand', () => {
      const state = makeGameState();
      state.players[0].ringsInHand = 0;
      addStack(state.board, pos(0, 0), 1, [1, 1]);

      expect(countRingsInPlayForPlayer(state, 1)).toBe(2);
    });
  });

  describe('calculateCapHeight', () => {
    it('returns 0 for empty rings array', () => {
      expect(calculateCapHeight([])).toBe(0);
    });

    it('returns 1 for single ring', () => {
      expect(calculateCapHeight([1])).toBe(1);
    });

    it('returns correct height for same color rings', () => {
      expect(calculateCapHeight([1, 1, 1])).toBe(3);
    });

    it('returns 1 when second ring differs', () => {
      expect(calculateCapHeight([1, 2, 1])).toBe(1);
    });

    it('returns correct height for mixed stack', () => {
      expect(calculateCapHeight([2, 2, 1, 2])).toBe(2);
    });
  });

  describe('getPathPositions', () => {
    it('returns single position when from equals to', () => {
      const result = getPathPositions(pos(3, 3), pos(3, 3));
      expect(result).toHaveLength(1);
      expect(result[0]).toEqual(pos(3, 3));
    });

    it('returns horizontal path', () => {
      const result = getPathPositions(pos(0, 0), pos(3, 0));
      expect(result).toHaveLength(4);
      expect(result[0]).toEqual(pos(0, 0));
      expect(result[1]).toEqual(pos(1, 0));
      expect(result[2]).toEqual(pos(2, 0));
      expect(result[3]).toEqual(pos(3, 0));
    });

    it('returns diagonal path', () => {
      const result = getPathPositions(pos(0, 0), pos(2, 2));
      expect(result).toHaveLength(3);
      expect(result[0]).toEqual(pos(0, 0));
      expect(result[1]).toEqual(pos(1, 1));
      expect(result[2]).toEqual(pos(2, 2));
    });

    it('handles z coordinate when present', () => {
      const result = getPathPositions(pos(0, 0, 0), pos(2, -2, 0));
      expect(result).toHaveLength(3);
      expect(result[2].z).toBe(0);
    });

    it('handles path with z changes', () => {
      const result = getPathPositions(pos(0, 0, 0), pos(2, 0, -2));
      expect(result).toHaveLength(3);
      expect(result[0].z).toBe(0);
      expect(result[2].z).toBe(-2);
    });

    it('preserves z in intermediate positions', () => {
      const result = getPathPositions(pos(0, 0, 0), pos(3, 0, -3));
      expect(result).toHaveLength(4);
      expect(result[1].z).toBe(-1);
      expect(result[2].z).toBe(-2);
    });
  });

  describe('calculateDistance', () => {
    describe('square boards', () => {
      it('calculates horizontal distance', () => {
        expect(calculateDistance('square8', pos(0, 0), pos(5, 0))).toBe(5);
      });

      it('calculates vertical distance', () => {
        expect(calculateDistance('square8', pos(0, 0), pos(0, 3))).toBe(3);
      });

      it('calculates diagonal distance (Chebyshev)', () => {
        expect(calculateDistance('square8', pos(0, 0), pos(3, 4))).toBe(4);
      });

      it('returns 0 for same position', () => {
        expect(calculateDistance('square8', pos(2, 2), pos(2, 2))).toBe(0);
      });
    });

    describe('hexagonal boards', () => {
      it('calculates hex distance using cube coordinates', () => {
        expect(calculateDistance('hexagonal', pos(0, 0, 0), pos(2, -1, -1))).toBe(2);
      });

      it('calculates hex distance with z coordinates', () => {
        expect(calculateDistance('hexagonal', pos(0, 0, 0), pos(1, 1, -2))).toBe(2);
      });

      it('handles missing z (defaults to 0)', () => {
        expect(calculateDistance('hexagonal', pos(0, 0), pos(2, 0))).toBe(1);
      });
    });
  });

  describe('validateCaptureSegmentOnBoard', () => {
    // Helper to create a board view
    function makeBoardView(board: BoardState): CaptureSegmentBoardView {
      return {
        isValidPosition: (p: Position) => p.x >= 0 && p.x < 8 && p.y >= 0 && p.y < 8,
        isCollapsedSpace: (p: Position) => board.collapsedSpaces.has(positionToString(p)),
        getStackAt: (p: Position) => {
          const stack = board.stacks.get(positionToString(p));
          if (!stack) return undefined;
          return {
            controllingPlayer: stack.controllingPlayer,
            capHeight: stack.capHeight,
            stackHeight: stack.stackHeight,
          };
        },
        getMarkerOwner: (p: Position) => {
          const marker = board.markers.get(positionToString(p));
          return marker?.player;
        },
      };
    }

    it('returns false for invalid from position', () => {
      const board = makeBoardState();
      const view = makeBoardView(board);

      expect(
        validateCaptureSegmentOnBoard('square8', pos(-1, 0), pos(1, 0), pos(2, 0), 1, view)
      ).toBe(false);
    });

    it('returns false for invalid target position', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1, 1]);
      const view = makeBoardView(board);

      expect(
        validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(10, 0), pos(12, 0), 1, view)
      ).toBe(false);
    });

    it('returns false for invalid landing position', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1, 1]);
      addStack(board, pos(1, 0), 2, [2]);
      const view = makeBoardView(board);

      expect(
        validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(1, 0), pos(-5, 0), 1, view)
      ).toBe(false);
    });

    it('returns false when no attacker at from', () => {
      const board = makeBoardState();
      const view = makeBoardView(board);

      expect(
        validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(1, 0), pos(2, 0), 1, view)
      ).toBe(false);
    });

    it('returns false when attacker not controlled by player', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 2, [2, 2]); // Player 2 controls
      addStack(board, pos(1, 0), 1, [1]);
      const view = makeBoardView(board);

      expect(
        validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(1, 0), pos(2, 0), 1, view)
      ).toBe(false);
    });

    it('returns false when no target stack', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1, 1]);
      const view = makeBoardView(board);

      expect(
        validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(1, 0), pos(2, 0), 1, view)
      ).toBe(false);
    });

    it('returns false when cap height insufficient', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1]); // capHeight 1
      addStack(board, pos(1, 0), 2, [2, 2]); // capHeight 2
      const view = makeBoardView(board);

      expect(
        validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(1, 0), pos(2, 0), 1, view)
      ).toBe(false);
    });

    describe('direction validation', () => {
      it('returns false for zero movement on square board', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1, 1]);
        addStack(board, pos(0, 0), 2, [2]); // Same position (invalid)
        const view = makeBoardView(board);

        expect(
          validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(0, 0), pos(0, 0), 1, view)
        ).toBe(false);
      });

      it('returns false for non-straight diagonal on square board', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1, 1]);
        addStack(board, pos(2, 1), 2, [2]); // Not aligned diagonally
        const view = makeBoardView(board);

        expect(
          validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(2, 1), pos(4, 2), 1, view)
        ).toBe(false);
      });

      it('validates hex direction with two coordinate changes', () => {
        // Test that the hex direction validation branch is exercised
        // For hex boards, exactly 2 of 3 cube coordinates must change
        const view: CaptureSegmentBoardView = {
          isValidPosition: () => true,
          isCollapsedSpace: () => false,
          getStackAt: (p: Position) => {
            const key = `${p.x},${p.y}`;
            if (key === '0,0') return { controllingPlayer: 1, capHeight: 3, stackHeight: 3 };
            if (key === '1,0') return { controllingPlayer: 2, capHeight: 1, stackHeight: 1 };
            return undefined;
          },
        };

        // Valid hex direction (dx=1, dy=0, dz=-1 -> exactly 2 changes)
        // Landing at 3,0,-3 is 3 steps away, satisfies distance >= height (3)
        expect(
          validateCaptureSegmentOnBoard(
            'hexagonal',
            pos(0, 0, 0),
            pos(1, 0, -1),
            pos(3, 0, -3),
            1,
            view
          )
        ).toBe(true);
      });

      it('returns false for invalid hex direction', () => {
        const board = makeBoardState({ type: 'hexagonal' as BoardType });
        addStack(board, pos(0, 0, 0), 1, [1, 1, 1]);
        addStack(board, pos(1, 1, 1), 2, [2]); // Invalid direction (all coords change)
        const view: CaptureSegmentBoardView = {
          isValidPosition: () => true,
          isCollapsedSpace: () => false,
          getStackAt: (p: Position) => {
            const key = positionToString(p);
            if (key === '0,0') return { controllingPlayer: 1, capHeight: 3, stackHeight: 3 };
            if (key === '1,1') return { controllingPlayer: 2, capHeight: 1, stackHeight: 1 };
            return undefined;
          },
        };

        expect(
          validateCaptureSegmentOnBoard(
            'hexagonal',
            pos(0, 0, 0),
            pos(1, 1, 1),
            pos(2, 2, 2),
            1,
            view
          )
        ).toBe(false);
      });
    });

    describe('path blocking', () => {
      it('returns false when path to target blocked by stack', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1, 1, 1]);
        addStack(board, pos(1, 0), 2, [2]); // Blocker
        addStack(board, pos(2, 0), 2, [2]); // Target
        const view = makeBoardView(board);

        expect(
          validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(2, 0), pos(3, 0), 1, view)
        ).toBe(false);
      });

      it('returns false when path to target blocked by collapsed space', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1, 1, 1]);
        addCollapsed(board, pos(1, 0), 1);
        addStack(board, pos(2, 0), 2, [2]);
        const view = makeBoardView(board);

        expect(
          validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(2, 0), pos(3, 0), 1, view)
        ).toBe(false);
      });

      it('returns false when path has invalid position', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1, 1, 1]);
        addStack(board, pos(3, 0), 2, [2]);
        const view: CaptureSegmentBoardView = {
          isValidPosition: (p: Position) => p.x !== 1 || p.y !== 0, // Make (1,0) invalid
          isCollapsedSpace: () => false,
          getStackAt: (p: Position) => {
            const stack = board.stacks.get(positionToString(p));
            if (!stack) return undefined;
            return {
              controllingPlayer: stack.controllingPlayer,
              capHeight: stack.capHeight,
              stackHeight: stack.stackHeight,
            };
          },
        };

        expect(
          validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(3, 0), pos(4, 0), 1, view)
        ).toBe(false);
      });
    });

    describe('landing direction validation', () => {
      it('returns false when landing in wrong X direction', () => {
        const board = makeBoardState();
        addStack(board, pos(4, 0), 1, [1, 1]);
        addStack(board, pos(5, 0), 2, [2]);
        const view = makeBoardView(board);

        // Target is east but landing is west
        expect(
          validateCaptureSegmentOnBoard('square8', pos(4, 0), pos(5, 0), pos(3, 0), 1, view)
        ).toBe(false);
      });

      it('returns false when landing in wrong Y direction', () => {
        const board = makeBoardState();
        addStack(board, pos(4, 4), 1, [1, 1]);
        addStack(board, pos(4, 5), 2, [2]);
        const view = makeBoardView(board);

        // Target is south but landing is north
        expect(
          validateCaptureSegmentOnBoard('square8', pos(4, 4), pos(4, 5), pos(4, 3), 1, view)
        ).toBe(false);
      });

      it('returns false when landing not beyond target', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1]);
        addStack(board, pos(3, 0), 2, [2]);
        const view = makeBoardView(board);

        // Landing at target, not beyond
        expect(
          validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(3, 0), pos(3, 0), 1, view)
        ).toBe(false);
      });
    });

    describe('distance validation', () => {
      it('returns false when distance less than stack height', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1, 1, 1, 1]); // Height 4
        addStack(board, pos(1, 0), 2, [2]);
        const view = makeBoardView(board);

        // Distance to landing is only 2, need at least 4
        expect(
          validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(1, 0), pos(2, 0), 1, view)
        ).toBe(false);
      });
    });

    describe('landing path validation', () => {
      it('returns false when landing path blocked by stack', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1, 1, 1, 1, 1]);
        addStack(board, pos(1, 0), 2, [2]);
        addStack(board, pos(3, 0), 2, [2]); // Blocker between target and landing
        const view = makeBoardView(board);

        expect(
          validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(1, 0), pos(5, 0), 1, view)
        ).toBe(false);
      });

      it('returns false when landing path blocked by collapsed', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1, 1, 1, 1, 1]);
        addStack(board, pos(1, 0), 2, [2]);
        addCollapsed(board, pos(3, 0), 1);
        const view = makeBoardView(board);

        expect(
          validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(1, 0), pos(5, 0), 1, view)
        ).toBe(false);
      });
    });

    describe('landing space validation', () => {
      it('returns false when landing is collapsed', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1, 1, 1]);
        addStack(board, pos(1, 0), 2, [2]);
        addCollapsed(board, pos(3, 0), 1);
        const view = makeBoardView(board);

        expect(
          validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(1, 0), pos(3, 0), 1, view)
        ).toBe(false);
      });

      it('returns false when landing occupied by stack', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1, 1, 1]);
        addStack(board, pos(1, 0), 2, [2]);
        addStack(board, pos(3, 0), 1, [1]); // Stack at landing
        const view = makeBoardView(board);

        expect(
          validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(1, 0), pos(3, 0), 1, view)
        ).toBe(false);
      });

      it('allows landing on marker', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1, 1, 1]);
        addStack(board, pos(1, 0), 2, [2]);
        addMarker(board, pos(3, 0), 2);
        const view = makeBoardView(board);

        expect(
          validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(1, 0), pos(3, 0), 1, view)
        ).toBe(true);
      });
    });

    it('returns true for valid capture', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1, 1, 1]);
      addStack(board, pos(1, 0), 2, [2]);
      const view = makeBoardView(board);

      expect(
        validateCaptureSegmentOnBoard('square8', pos(0, 0), pos(1, 0), pos(3, 0), 1, view)
      ).toBe(true);
    });
  });

  describe('hasAnyLegalMoveOrCaptureFromOnBoard', () => {
    function makeMoveView(board: BoardState): MovementBoardView {
      return {
        isValidPosition: (p: Position) => p.x >= 0 && p.x < 8 && p.y >= 0 && p.y < 8,
        isCollapsedSpace: (p: Position) => board.collapsedSpaces.has(positionToString(p)),
        getStackAt: (p: Position) => {
          const stack = board.stacks.get(positionToString(p));
          if (!stack) return undefined;
          return {
            controllingPlayer: stack.controllingPlayer,
            capHeight: stack.capHeight,
            stackHeight: stack.stackHeight,
          };
        },
        getMarkerOwner: (p: Position) => {
          const marker = board.markers.get(positionToString(p));
          return marker?.player;
        },
      };
    }

    it('returns false when no stack at position', () => {
      const board = makeBoardState();
      const view = makeMoveView(board);

      expect(hasAnyLegalMoveOrCaptureFromOnBoard('square8', pos(0, 0), 1, view)).toBe(false);
    });

    it('returns false when stack not controlled by player', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 2, [2, 2]);
      const view = makeMoveView(board);

      expect(hasAnyLegalMoveOrCaptureFromOnBoard('square8', pos(0, 0), 1, view)).toBe(false);
    });

    it('returns true when simple move available', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1]);
      const view = makeMoveView(board);

      expect(hasAnyLegalMoveOrCaptureFromOnBoard('square8', pos(0, 0), 1, view)).toBe(true);
    });

    it('returns true when landing on marker available', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1]);
      addMarker(board, pos(1, 0), 2);
      const view = makeMoveView(board);

      expect(hasAnyLegalMoveOrCaptureFromOnBoard('square8', pos(0, 0), 1, view)).toBe(true);
    });

    it('returns false when completely blocked', () => {
      const board = makeBoardState();
      addStack(board, pos(3, 3), 1, [1, 1, 1]); // Height 3, needs to move 3 spaces

      // Surround with collapsed spaces
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx !== 0 || dy !== 0) {
            addCollapsed(board, pos(3 + dx, 3 + dy), 2);
          }
        }
      }
      const view = makeMoveView(board);

      expect(hasAnyLegalMoveOrCaptureFromOnBoard('square8', pos(3, 3), 1, view)).toBe(false);
    });

    it('handles stack blocking the path', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1, 1]); // Height 2
      // Block all directions at distance 1
      addStack(board, pos(1, 0), 2, [2]);
      addStack(board, pos(0, 1), 2, [2]);
      addStack(board, pos(1, 1), 2, [2]);
      const view = makeMoveView(board);

      // Can still capture since stack at (1,0) is capturable
      expect(hasAnyLegalMoveOrCaptureFromOnBoard('square8', pos(0, 0), 1, view)).toBe(true);
    });

    it('handles z coordinates for hex boards', () => {
      const board = makeBoardState({ type: 'hexagonal' as BoardType });
      addStack(board, pos(0, 0, 0), 1, [1]);
      const view: MovementBoardView = {
        isValidPosition: () => true,
        isCollapsedSpace: () => false,
        getStackAt: (p: Position) => {
          if (p.x === 0 && p.y === 0) {
            return { controllingPlayer: 1, capHeight: 1, stackHeight: 1 };
          }
          return undefined;
        },
      };

      expect(hasAnyLegalMoveOrCaptureFromOnBoard('hexagonal', pos(0, 0, 0), 1, view)).toBe(true);
    });

    it('respects maxNonCaptureDistance option', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1]); // Height 1
      // Block distance 1
      addStack(board, pos(1, 0), 2, [2, 2]); // Can't capture (cap height 2 > 1)
      const view = makeMoveView(board);

      // With very small max distance, shouldn't find moves
      expect(
        hasAnyLegalMoveOrCaptureFromOnBoard('square8', pos(0, 0), 1, view, {
          maxNonCaptureDistance: 0,
          maxCaptureLandingDistance: 0,
        })
      ).toBe(false);
    });

    describe('capture reachability', () => {
      it('finds capture target along ray', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1, 1, 1]); // Height 3, cap 3
        addStack(board, pos(3, 0), 2, [2, 2]); // Capturable (cap 2 <= 3)
        const view = makeMoveView(board);

        expect(hasAnyLegalMoveOrCaptureFromOnBoard('square8', pos(0, 0), 1, view)).toBe(true);
      });

      it('ignores uncapturable stacks (higher cap)', () => {
        const board = makeBoardState();
        addStack(board, pos(0, 0), 1, [1]); // Height 1, cap 1
        // Block with uncapturable stack
        addStack(board, pos(1, 0), 2, [2, 2, 2]); // Cap 3 > 1
        addStack(board, pos(0, 1), 2, [2, 2, 2]);
        addStack(board, pos(1, 1), 2, [2, 2, 2]);
        const view: MovementBoardView = {
          isValidPosition: (p: Position) => p.x >= 0 && p.x < 4 && p.y >= 0 && p.y < 4,
          isCollapsedSpace: () => false,
          getStackAt: (p: Position) => {
            const stack = board.stacks.get(positionToString(p));
            if (!stack) return undefined;
            return {
              controllingPlayer: stack.controllingPlayer,
              capHeight: stack.capHeight,
              stackHeight: stack.stackHeight,
            };
          },
        };

        expect(hasAnyLegalMoveOrCaptureFromOnBoard('square8', pos(0, 0), 1, view)).toBe(false);
      });
    });
  });

  describe('computeProgressSnapshot', () => {
    it('computes S-invariant correctly', () => {
      const state = makeGameState();
      addMarker(state.board, pos(0, 0), 1);
      addMarker(state.board, pos(1, 1), 2);
      addCollapsed(state.board, pos(2, 2), 1);

      (state as GameState & { totalRingsEliminated?: number }).totalRingsEliminated = 5;

      const snapshot = computeProgressSnapshot(state);

      expect(snapshot.markers).toBe(2);
      expect(snapshot.collapsed).toBe(1);
      expect(snapshot.eliminated).toBe(5);
      expect(snapshot.S).toBe(8); // 2 + 1 + 5
    });

    it('uses board eliminatedRings as fallback', () => {
      const state = makeGameState();
      state.board.eliminatedRings = { 1: 3, 2: 2 };
      // Remove totalRingsEliminated
      delete (state as GameState & { totalRingsEliminated?: number }).totalRingsEliminated;

      const snapshot = computeProgressSnapshot(state);

      expect(snapshot.eliminated).toBe(5); // 3 + 2
    });

    it('handles missing eliminatedRings', () => {
      const state = makeGameState();
      (state.board as BoardState & { eliminatedRings?: Record<number, number> }).eliminatedRings =
        undefined;
      delete (state as GameState & { totalRingsEliminated?: number }).totalRingsEliminated;

      const snapshot = computeProgressSnapshot(state);

      expect(snapshot.eliminated).toBe(0);
    });
  });

  describe('summarizeBoard', () => {
    it('summarizes stacks', () => {
      const board = makeBoardState();
      addStack(board, pos(0, 0), 1, [1, 1]);
      addStack(board, pos(3, 3), 2, [2]);

      const summary = summarizeBoard(board);

      expect(summary.stacks).toContain('0,0:1:2:2');
      expect(summary.stacks).toContain('3,3:2:1:1');
      expect(summary.stacks).toHaveLength(2);
    });

    it('summarizes markers', () => {
      const board = makeBoardState();
      addMarker(board, pos(1, 1), 1);
      addMarker(board, pos(5, 5), 2);

      const summary = summarizeBoard(board);

      expect(summary.markers).toContain('1,1:1');
      expect(summary.markers).toContain('5,5:2');
      expect(summary.markers).toHaveLength(2);
    });

    it('summarizes collapsed spaces', () => {
      const board = makeBoardState();
      addCollapsed(board, pos(2, 2), 1);
      addCollapsed(board, pos(4, 4), 2);

      const summary = summarizeBoard(board);

      expect(summary.collapsedSpaces).toContain('2,2:1');
      expect(summary.collapsedSpaces).toContain('4,4:2');
      expect(summary.collapsedSpaces).toHaveLength(2);
    });

    it('sorts all arrays', () => {
      const board = makeBoardState();
      addStack(board, pos(5, 5), 1, [1]);
      addStack(board, pos(0, 0), 1, [1]);

      const summary = summarizeBoard(board);

      expect(summary.stacks[0]).toBe('0,0:1:1:1');
      expect(summary.stacks[1]).toBe('5,5:1:1:1');
    });

    it('returns empty arrays for empty board', () => {
      const board = makeBoardState();

      const summary = summarizeBoard(board);

      expect(summary.stacks).toEqual([]);
      expect(summary.markers).toEqual([]);
      expect(summary.collapsedSpaces).toEqual([]);
    });
  });

  describe('fingerprintGameState', () => {
    it('creates deterministic fingerprint', () => {
      const state = makeGameState();
      state.currentPlayer = 2;
      state.currentPhase = 'movement';
      addStack(state.board, pos(0, 0), 1, [1]);

      const fp1 = fingerprintGameState(state);
      const fp2 = fingerprintGameState(state);

      expect(fp1).toBe(fp2);
    });

    it('includes player meta', () => {
      const state = makeGameState();
      state.players[0].ringsInHand = 5;
      state.players[0].eliminatedRings = 2;
      state.players[0].territorySpaces = 3;

      const fp = fingerprintGameState(state);

      expect(fp).toContain('1:5:2:3');
    });

    it('includes game meta', () => {
      const state = makeGameState();
      state.currentPlayer = 2;
      state.currentPhase = 'capture';
      state.gameStatus = 'active';

      const fp = fingerprintGameState(state);

      expect(fp).toContain('2:capture:active');
    });
  });

  describe('hashGameStateSHA256', () => {
    it('returns 16 character hex string', () => {
      const state = makeGameState();

      const hash = hashGameStateSHA256(state);

      expect(hash).toHaveLength(16);
      expect(/^[0-9a-f]+$/.test(hash)).toBe(true);
    });

    it('is deterministic', () => {
      const state = makeGameState();

      const hash1 = hashGameStateSHA256(state);
      const hash2 = hashGameStateSHA256(state);

      expect(hash1).toBe(hash2);
    });

    it('changes with state changes', () => {
      const state1 = makeGameState();
      const state2 = makeGameState();
      state2.currentPlayer = 2;

      expect(hashGameStateSHA256(state1)).not.toBe(hashGameStateSHA256(state2));
    });
  });

  describe('hashGameState (legacy)', () => {
    it('returns fingerprint string', () => {
      const state = makeGameState();

      const hash = hashGameState(state);
      const fp = fingerprintGameState(state);

      expect(hash).toBe(fp);
    });
  });

  describe('applyMarkerEffectsAlongPathOnBoard', () => {
    let setMarkerCalls: Array<{ pos: Position; player: number }>;
    let collapseMarkerCalls: Array<{ pos: Position; player: number }>;
    let flipMarkerCalls: Array<{ pos: Position; player: number }>;
    let helpers: MarkerPathHelpers;

    beforeEach(() => {
      setMarkerCalls = [];
      collapseMarkerCalls = [];
      flipMarkerCalls = [];
      helpers = {
        setMarker: (position: Position, player: number) => {
          setMarkerCalls.push({ pos: position, player });
        },
        collapseMarker: (position: Position, player: number) => {
          collapseMarkerCalls.push({ pos: position, player });
        },
        flipMarker: (position: Position, player: number) => {
          flipMarkerCalls.push({ pos: position, player });
        },
      };
    });

    it('returns early for empty path', () => {
      const board = makeBoardState();

      // Same from and to - path length 1
      applyMarkerEffectsAlongPathOnBoard(board, pos(0, 0), pos(0, 0), 1, helpers);

      // Should still set departure marker (even though from=to)
      expect(setMarkerCalls).toHaveLength(1);
    });

    it('sets departure marker on empty space', () => {
      const board = makeBoardState();

      applyMarkerEffectsAlongPathOnBoard(board, pos(0, 0), pos(3, 0), 1, helpers);

      expect(setMarkerCalls).toHaveLength(1);
      expect(setMarkerCalls[0].pos).toEqual(pos(0, 0));
      expect(setMarkerCalls[0].player).toBe(1);
    });

    it('does not set departure marker when collapsed', () => {
      const board = makeBoardState();
      addCollapsed(board, pos(0, 0), 1);

      applyMarkerEffectsAlongPathOnBoard(board, pos(0, 0), pos(3, 0), 1, helpers);

      expect(setMarkerCalls).toHaveLength(0);
    });

    it('does not set departure marker when marker exists', () => {
      const board = makeBoardState();
      addMarker(board, pos(0, 0), 1);

      applyMarkerEffectsAlongPathOnBoard(board, pos(0, 0), pos(3, 0), 1, helpers);

      expect(setMarkerCalls).toHaveLength(0);
    });

    it('respects leaveDepartureMarker: false option', () => {
      const board = makeBoardState();

      applyMarkerEffectsAlongPathOnBoard(board, pos(0, 0), pos(3, 0), 1, helpers, {
        leaveDepartureMarker: false,
      });

      expect(setMarkerCalls).toHaveLength(0);
    });

    it('collapses own marker in path', () => {
      const board = makeBoardState();
      addMarker(board, pos(1, 0), 1); // Own marker in path

      applyMarkerEffectsAlongPathOnBoard(board, pos(0, 0), pos(3, 0), 1, helpers);

      expect(collapseMarkerCalls).toHaveLength(1);
      expect(collapseMarkerCalls[0].pos).toEqual(pos(1, 0));
    });

    it('flips opponent marker in path', () => {
      const board = makeBoardState();
      addMarker(board, pos(1, 0), 2); // Opponent marker in path

      applyMarkerEffectsAlongPathOnBoard(board, pos(0, 0), pos(3, 0), 1, helpers);

      expect(flipMarkerCalls).toHaveLength(1);
      expect(flipMarkerCalls[0].pos).toEqual(pos(1, 0));
    });

    it('skips collapsed spaces in path', () => {
      const board = makeBoardState();
      addCollapsed(board, pos(1, 0), 1);
      addMarker(board, pos(1, 0), 1); // This shouldn't be processed

      applyMarkerEffectsAlongPathOnBoard(board, pos(0, 0), pos(3, 0), 1, helpers);

      expect(collapseMarkerCalls).toHaveLength(0);
    });

    it('removes own marker at landing', () => {
      const board = makeBoardState();
      addMarker(board, pos(3, 0), 1); // Own marker at landing

      applyMarkerEffectsAlongPathOnBoard(board, pos(0, 0), pos(3, 0), 1, helpers);

      expect(board.markers.has('3,0')).toBe(false);
    });

    it('preserves opponent marker at landing', () => {
      const board = makeBoardState();
      addMarker(board, pos(3, 0), 2); // Opponent marker at landing

      applyMarkerEffectsAlongPathOnBoard(board, pos(0, 0), pos(3, 0), 1, helpers);

      expect(board.markers.has('3,0')).toBe(true);
    });

    it('processes multiple markers in path', () => {
      const board = makeBoardState();
      addMarker(board, pos(1, 0), 1); // Own - collapse
      addMarker(board, pos(2, 0), 2); // Opponent - flip

      applyMarkerEffectsAlongPathOnBoard(board, pos(0, 0), pos(4, 0), 1, helpers);

      expect(collapseMarkerCalls).toHaveLength(1);
      expect(flipMarkerCalls).toHaveLength(1);
    });

    it('handles diagonal path', () => {
      const board = makeBoardState();
      addMarker(board, pos(1, 1), 2); // Opponent marker

      applyMarkerEffectsAlongPathOnBoard(board, pos(0, 0), pos(3, 3), 1, helpers);

      expect(flipMarkerCalls).toHaveLength(1);
      expect(flipMarkerCalls[0].pos).toEqual(pos(1, 1));
    });
  });
});
