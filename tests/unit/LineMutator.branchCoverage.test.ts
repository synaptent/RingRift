/**
 * LineMutator.branchCoverage.test.ts
 *
 * Branch coverage tests for LineMutator.ts targeting uncovered branches:
 * - mutateProcessLine: line length > required check, error path
 * - mutateChooseLineReward: COLLAPSE_ALL vs MINIMUM_COLLAPSE, missing positions error
 * - executeCollapse: stack handling, marker handling, player lookup, broken lines filter
 */

import {
  mutateProcessLine,
  mutateChooseLineReward,
} from '../../src/shared/engine/mutators/LineMutator';
import type {
  GameState,
  ProcessLineAction,
  ChooseLineRewardAction,
  RingStack,
  FormedLine,
} from '../../src/shared/engine/types';
import type { Position, BoardType } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

// Helper to create a position
const pos = (x: number, y: number): Position => ({ x, y });

// Helper to create a minimal game state for testing
function makeGameState(overrides: Partial<GameState> = {}): GameState {
  // Use square19 board type because lineLength=4 (matching our 4-position test lines).
  // square8 has lineLength=3, so 4-position lines would be > minimum and require
  // ChooseLineRewardAction instead of ProcessLineAction.
  const defaultState: GameState = {
    id: 'test-game',
    board: {
      type: 'square19' as BoardType,
      size: 19,
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      formedLines: [],
      territories: new Map(),
      eliminatedRings: { 1: 0, 2: 0 },
    },
    players: [
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
    ],
    currentPlayer: 1,
    currentPhase: 'line_processing',
    gameStatus: 'active',
    timeControl: { initialTime: 600000, increment: 0, type: 'rapid' },
    moveHistory: [],
    spectators: [],
    boardType: 'square19',
  };

  return { ...defaultState, ...overrides } as GameState & {
    totalRingsEliminated: number;
    lastMoveAt: Date;
    totalRingsInPlay: number;
  };
}

// Helper to create a formed line
function makeFormedLine(player: number, positions: Position[]): FormedLine {
  return {
    player,
    positions,
    length: positions.length,
  };
}

// Helper to add a stack to the board
function addStack(
  state: GameState,
  position: Position,
  controllingPlayer: number,
  rings: number[]
): void {
  const key = positionToString(position);
  const stack: RingStack = {
    position,
    rings,
    stackHeight: rings.length,
    capHeight: rings.length,
    controllingPlayer,
  };
  state.board.stacks.set(key, stack);
}

// Helper to add a marker to the board
function addMarker(state: GameState, position: Position, player: number): void {
  const key = positionToString(position);
  state.board.markers.set(key, player);
}

describe('LineMutator branch coverage', () => {
  describe('mutateProcessLine', () => {
    it('processes exact length line (no choice needed)', () => {
      // For 2-player square8, minimum line length is 4
      const state = makeGameState();
      const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
      state.board.formedLines = [line];

      const action: ProcessLineAction = {
        type: 'process_line',
        playerId: 1,
        lineIndex: 0,
      };

      const result = mutateProcessLine(state, action);

      // Verify line was processed - positions are now collapsed
      expect(result.board.collapsedSpaces.has('0,0')).toBe(true);
      expect(result.board.collapsedSpaces.has('1,0')).toBe(true);
      expect(result.board.collapsedSpaces.has('2,0')).toBe(true);
      expect(result.board.collapsedSpaces.has('3,0')).toBe(true);
      // Line should be removed
      expect(result.board.formedLines.length).toBe(0);
    });

    it('throws error for line longer than minimum (choice required)', () => {
      const state = makeGameState();
      // 5-space line when minimum is 4
      const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
      state.board.formedLines = [line];

      const action: ProcessLineAction = {
        type: 'process_line',
        playerId: 1,
        lineIndex: 0,
      };

      expect(() => mutateProcessLine(state, action)).toThrow(
        'LineMutator: Line length > minimum requires ChooseLineRewardAction'
      );
    });
  });

  describe('mutateChooseLineReward', () => {
    describe('COLLAPSE_ALL selection', () => {
      it('collapses entire line with COLLAPSE_ALL', () => {
        const state = makeGameState();
        const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
        state.board.formedLines = [line];

        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'COLLAPSE_ALL',
        };

        const result = mutateChooseLineReward(state, action);

        // All positions collapsed
        expect(result.board.collapsedSpaces.has('0,0')).toBe(true);
        expect(result.board.collapsedSpaces.has('4,0')).toBe(true);
        expect(result.board.formedLines.length).toBe(0);
      });
    });

    describe('MINIMUM_COLLAPSE selection', () => {
      it('collapses only specified positions with MINIMUM_COLLAPSE', () => {
        const state = makeGameState();
        const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
        state.board.formedLines = [line];

        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'MINIMUM_COLLAPSE',
          collapsedPositions: [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)], // Only 4 of 5
        };

        const result = mutateChooseLineReward(state, action);

        // Only specified positions collapsed
        expect(result.board.collapsedSpaces.has('0,0')).toBe(true);
        expect(result.board.collapsedSpaces.has('3,0')).toBe(true);
        // pos(4,0) was not collapsed
        expect(result.board.collapsedSpaces.has('4,0')).toBe(false);
      });

      it('throws error when MINIMUM_COLLAPSE missing collapsedPositions', () => {
        const state = makeGameState();
        const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
        state.board.formedLines = [line];

        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'MINIMUM_COLLAPSE',
          // collapsedPositions is missing!
        };

        expect(() => mutateChooseLineReward(state, action)).toThrow(
          'LineMutator: Missing collapsedPositions for MINIMUM_COLLAPSE'
        );
      });
    });
  });

  describe('executeCollapse (via mutateProcessLine)', () => {
    describe('stack handling', () => {
      it('removes stacks at collapsed positions and credits eliminations to the acting player', () => {
        const state = makeGameState();
        // Add a stack at position that will be collapsed
        addStack(state, pos(0, 0), 1, [1, 1]);
        const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];
        state.players[0].ringsInHand = 8; // Started with some rings

        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 0,
        };

        const result = mutateProcessLine(state, action);

        // Stack should be removed
        expect(result.board.stacks.has('0,0')).toBe(false);
        // Rings on collapsed spaces are eliminated and credited to player 1.
        expect(result.players[0].ringsInHand).toBe(8);
        expect(result.players[0].eliminatedRings).toBe(2);
        expect(result.totalRingsEliminated).toBe(2);
        expect(result.board.eliminatedRings[1]).toBe(2);
      });

      it('credits mixed-ownership stack eliminations to the acting player', () => {
        const state = makeGameState();
        // Stack with rings from both players
        addStack(state, pos(1, 0), 1, [1, 2, 1]); // P1 controls, rings owned by P1, P2, P1
        const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];
        state.players[0].ringsInHand = 8;
        state.players[1].ringsInHand = 8;

        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 0,
        };

        const result = mutateProcessLine(state, action);

        expect(result.players[0].ringsInHand).toBe(8);
        expect(result.players[1].ringsInHand).toBe(8);
        expect(result.players[0].eliminatedRings).toBe(3);
        expect(result.players[1].eliminatedRings).toBe(0);
        expect(result.totalRingsEliminated).toBe(3);
        expect(result.board.eliminatedRings[1]).toBe(3);
      });

      it('handles position without stack (no-op for stack removal)', () => {
        const state = makeGameState();
        // No stack at any collapse position
        const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 0,
        };

        // Should not throw
        const result = mutateProcessLine(state, action);
        expect(result.board.collapsedSpaces.size).toBe(4);
      });
    });

    describe('marker handling', () => {
      it('removes markers at collapsed positions', () => {
        const state = makeGameState();
        // Add markers at positions that will be collapsed
        addMarker(state, pos(0, 0), 1);
        addMarker(state, pos(2, 0), 2);
        const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 0,
        };

        const result = mutateProcessLine(state, action);

        // Markers should be removed
        expect(result.board.markers.has('0,0')).toBe(false);
        expect(result.board.markers.has('2,0')).toBe(false);
      });

      it('handles position without marker (no-op for marker removal)', () => {
        const state = makeGameState();
        // No markers
        const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 0,
        };

        // Should not throw
        const result = mutateProcessLine(state, action);
        expect(result.board.markers.size).toBe(0);
      });
    });

    describe('broken lines handling', () => {
      it('removes other lines that share collapsed positions', () => {
        const state = makeGameState();
        // Two intersecting lines
        const line1 = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        const line2 = makeFormedLine(1, [pos(2, 0), pos(2, 1), pos(2, 2), pos(2, 3)]); // Intersects at (2,0)
        state.board.formedLines = [line1, line2];

        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 0, // Process line1
        };

        const result = mutateProcessLine(state, action);

        // Both lines should be removed - line1 processed, line2 broken
        expect(result.board.formedLines.length).toBe(0);
      });

      it('keeps lines that do not share collapsed positions', () => {
        const state = makeGameState();
        // Two non-intersecting lines
        const line1 = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        const line2 = makeFormedLine(2, [pos(0, 5), pos(1, 5), pos(2, 5), pos(3, 5)]); // Different row
        state.board.formedLines = [line1, line2];

        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 0, // Process line1
        };

        const result = mutateProcessLine(state, action);

        // line2 should remain (but index shifted)
        expect(result.board.formedLines.length).toBe(1);
        expect(result.board.formedLines[0].player).toBe(2);
      });
    });

    describe('player lookup', () => {
      it('throws if current player not found', () => {
        const state = makeGameState();
        // Remove all players to trigger the error - but keep players array for threshold calc
        const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
        state.board.formedLines = [line];
        // Set current player to one that doesn't exist in players array
        state.currentPlayer = 99;

        // Use ChooseLineReward to bypass the line length check
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'COLLAPSE_ALL',
        };

        expect(() => mutateChooseLineReward(state, action)).toThrow(
          'LineMutator: Player not found'
        );
      });
    });

    describe('collapsed spaces ownership', () => {
      it('marks collapsed spaces with current player number', () => {
        const state = makeGameState();
        state.currentPlayer = 2;
        const line = makeFormedLine(2, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 2,
          lineIndex: 0,
        };

        const result = mutateProcessLine(state, action);

        // All collapsed spaces owned by player 2
        expect(result.board.collapsedSpaces.get('0,0')).toBe(2);
        expect(result.board.collapsedSpaces.get('1,0')).toBe(2);
        expect(result.board.collapsedSpaces.get('2,0')).toBe(2);
        expect(result.board.collapsedSpaces.get('3,0')).toBe(2);
      });
    });

    describe('totalRingsInPlay tracking', () => {
      it('decrements totalRingsInPlay when stacks are collapsed', () => {
        const state = makeGameState() as GameState & { totalRingsInPlay: number };
        state.totalRingsInPlay = 36;
        addStack(state, pos(0, 0), 1, [1, 1]); // 2 rings
        addStack(state, pos(1, 0), 2, [2]); // 1 ring
        const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 0,
        };

        const result = mutateProcessLine(state, action) as GameState & { totalRingsInPlay: number };

        // 3 rings were eliminated from play, so totalRingsInPlay decreased by 3
        expect(result.totalRingsInPlay).toBe(33);
      });
    });

    describe('immutability', () => {
      it('does not mutate original state', () => {
        const state = makeGameState();
        addStack(state, pos(0, 0), 1, [1]);
        addMarker(state, pos(1, 0), 1);
        const line = makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const originalStacksSize = state.board.stacks.size;
        const originalMarkersSize = state.board.markers.size;
        const originalLinesLength = state.board.formedLines.length;
        const originalCollapsedSize = state.board.collapsedSpaces.size;

        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 0,
        };

        mutateProcessLine(state, action);

        // Original state should be unchanged
        expect(state.board.stacks.size).toBe(originalStacksSize);
        expect(state.board.markers.size).toBe(originalMarkersSize);
        expect(state.board.formedLines.length).toBe(originalLinesLength);
        expect(state.board.collapsedSpaces.size).toBe(originalCollapsedSize);
      });
    });
  });
});
