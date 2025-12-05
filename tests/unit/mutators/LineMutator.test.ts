/**
 * LineMutator branch coverage tests
 * Tests for src/shared/engine/mutators/LineMutator.ts
 */

import {
  mutateProcessLine,
  mutateChooseLineReward,
} from '../../../src/shared/engine/mutators/LineMutator';
import type {
  GameState,
  ProcessLineAction,
  ChooseLineRewardAction,
} from '../../../src/shared/engine/types';
import type { BoardState, BoardType, Position, RingStack } from '../../../src/shared/types/game';

function posStr(x: number, y: number): string {
  return `${x},${y}`;
}

function createMinimalBoard(
  overrides: Partial<{
    type: BoardType;
    size: number;
    stacks: Map<string, RingStack>;
    markers: Map<string, { player: number }>;
    collapsedSpaces: Map<string, number>;
    formedLines: Array<{
      player: number;
      length: number;
      positions: Position[];
      direction: { x: number; y: number };
    }>;
  }>
): BoardState {
  return {
    type: overrides.type ?? 'square8',
    size: overrides.size ?? 8,
    stacks: overrides.stacks ?? new Map(),
    markers: overrides.markers ?? new Map(),
    collapsedSpaces: overrides.collapsedSpaces ?? new Map(),
    rings: new Map(),
    territories: new Map(),
    formedLines: overrides.formedLines ?? [],
    eliminatedRings: {},
    geometry: { type: overrides.type ?? 'square8', size: overrides.size ?? 8 },
  } as BoardState;
}

function createMinimalState(
  overrides: Partial<{
    currentPhase: string;
    currentPlayer: number;
    boardType: BoardType;
    boardSize: number;
    stacks: Map<string, RingStack>;
    markers: Map<string, { player: number }>;
    collapsedSpaces: Map<string, number>;
    formedLines: Array<{
      player: number;
      length: number;
      positions: Position[];
      direction: { x: number; y: number };
    }>;
    players: Array<{ playerNumber: number; ringsInHand: number; eliminated: boolean }>;
  }>
): GameState {
  const boardType = overrides.boardType ?? 'square8';
  const boardSize = overrides.boardSize ?? 8;
  const playerCount = overrides.players?.length ?? 2;

  const players =
    overrides.players ??
    Array.from({ length: playerCount }, (_, i) => ({
      playerNumber: i + 1,
      ringsInHand: 10,
      eliminated: false,
      score: 0,
      reserveStacks: 0,
      reserveRings: 0,
    }));

  const base = {
    board: createMinimalBoard({
      type: boardType,
      size: boardSize,
      stacks: overrides.stacks,
      markers: overrides.markers,
      collapsedSpaces: overrides.collapsedSpaces,
      formedLines: overrides.formedLines,
    }),
    currentPhase: overrides.currentPhase ?? 'line_processing',
    currentPlayer: overrides.currentPlayer ?? 1,
    players,
    turnNumber: 1,
    gameStatus: 'active' as const,
    moveHistory: [],
    pendingDecision: null,
    victoryCondition: null,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
  };
  return base as unknown as GameState;
}

describe('LineMutator', () => {
  describe('mutateProcessLine', () => {
    it('should collapse line when exact length matches threshold', () => {
      // square8 has lineLength: 3
      const positions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
      ];

      const state = createMinimalState({
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 3, positions, direction: { x: 1, y: 0 } }],
      });

      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      const result = mutateProcessLine(state, action);

      // All positions should be collapsed
      expect(result.board.collapsedSpaces.size).toBe(3);
      expect(result.board.collapsedSpaces.has('0,0')).toBe(true);
      expect(result.board.collapsedSpaces.has('2,0')).toBe(true);
      // Line should be removed
      expect(result.board.formedLines.length).toBe(0);
    });

    it('should throw when line length exceeds threshold', () => {
      // square8 has lineLength: 3, so 4 positions exceeds threshold
      const positions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
        { x: 3, y: 0 },
      ];

      const state = createMinimalState({
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 5, positions, direction: { x: 1, y: 0 } }],
      });

      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      expect(() => mutateProcessLine(state, action)).toThrow(
        'LineMutator: Line length > minimum requires ChooseLineRewardAction'
      );
    });

    it('should remove stacks at collapsed positions and return rings to hand', () => {
      // square8 has lineLength: 3
      const positions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
      ];

      const stacks = new Map<string, RingStack>([
        [
          '1,0',
          {
            position: { x: 1, y: 0 },
            rings: [1, 2], // Player 1's and Player 2's rings
            stackHeight: 2,
            capHeight: 1,
            controllingPlayer: 1,
          },
        ],
      ]);

      const state = createMinimalState({
        currentPlayer: 1,
        stacks,
        formedLines: [{ player: 1, length: 3, positions, direction: { x: 1, y: 0 } }],
        players: [
          { playerNumber: 1, ringsInHand: 5, eliminated: false },
          { playerNumber: 2, ringsInHand: 5, eliminated: false },
        ],
      });

      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      const result = mutateProcessLine(state, action);

      // Stack should be removed
      expect(result.board.stacks.has('1,0')).toBe(false);
      // Rings returned to owners' hands
      expect(result.players.find((p) => p.playerNumber === 1)?.ringsInHand).toBe(6);
      expect(result.players.find((p) => p.playerNumber === 2)?.ringsInHand).toBe(6);
    });

    it('should remove markers at collapsed positions', () => {
      // square8 has lineLength: 3
      const positions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
      ];

      const markers = new Map<string, { player: number }>([['2,0', { player: 1 }]]);

      const state = createMinimalState({
        currentPlayer: 1,
        markers,
        formedLines: [{ player: 1, length: 3, positions, direction: { x: 1, y: 0 } }],
      });

      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      const result = mutateProcessLine(state, action);

      // Marker should be removed
      expect(result.board.markers.has('2,0')).toBe(false);
    });

    it('should break other lines that share collapsed positions', () => {
      // square8 has lineLength: 3
      const line1Positions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
      ];

      const line2Positions: Position[] = [
        { x: 1, y: 0 }, // Shares position with line 1
        { x: 1, y: 1 },
        { x: 1, y: 2 },
      ];

      const state = createMinimalState({
        currentPlayer: 1,
        formedLines: [
          { player: 1, length: 3, positions: line1Positions, direction: { x: 1, y: 0 } },
          { player: 1, length: 3, positions: line2Positions, direction: { x: 0, y: 1 } },
        ],
      });

      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      const result = mutateProcessLine(state, action);

      // Line 2 should be broken because it shares (1,0) which is now collapsed
      expect(result.board.formedLines.length).toBe(0);
    });
  });

  describe('mutateChooseLineReward', () => {
    it('should handle COLLAPSE_ALL selection', () => {
      const positions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
        { x: 3, y: 0 },
        { x: 4, y: 0 },
      ];

      const state = createMinimalState({
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 5, positions, direction: { x: 1, y: 0 } }],
      });

      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'COLLAPSE_ALL',
      };

      const result = mutateChooseLineReward(state, action);

      // All 5 positions should be collapsed
      expect(result.board.collapsedSpaces.size).toBe(5);
      expect(result.board.formedLines.length).toBe(0);
    });

    it('should handle MINIMUM_COLLAPSE selection with collapsedPositions', () => {
      const allPositions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
        { x: 3, y: 0 },
        { x: 4, y: 0 },
      ];

      const minPositions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
        { x: 3, y: 0 },
      ];

      const state = createMinimalState({
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 5, positions: allPositions, direction: { x: 1, y: 0 } }],
      });

      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: minPositions,
      };

      const result = mutateChooseLineReward(state, action);

      // Only 4 positions should be collapsed
      expect(result.board.collapsedSpaces.size).toBe(4);
      expect(result.board.collapsedSpaces.has('4,0')).toBe(false); // Not collapsed
    });

    it('should throw when MINIMUM_COLLAPSE is missing collapsedPositions', () => {
      const positions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
        { x: 3, y: 0 },
        { x: 4, y: 0 },
      ];

      const state = createMinimalState({
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 5, positions, direction: { x: 1, y: 0 } }],
      });

      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        // Missing collapsedPositions
      };

      expect(() => mutateChooseLineReward(state, action)).toThrow(
        'LineMutator: Missing collapsedPositions for MINIMUM_COLLAPSE'
      );
    });
  });

  describe('edge cases', () => {
    it('should handle empty stacks map', () => {
      // square8 has lineLength: 3
      const positions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
      ];

      const state = createMinimalState({
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 3, positions, direction: { x: 1, y: 0 } }],
      });

      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      const result = mutateProcessLine(state, action);

      expect(result.board.collapsedSpaces.size).toBe(3);
    });

    it('should handle 3-player game line length threshold', () => {
      const positions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
      ];

      const state = createMinimalState({
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 3, positions, direction: { x: 1, y: 0 } }],
        players: [
          { playerNumber: 1, ringsInHand: 10, eliminated: false },
          { playerNumber: 2, ringsInHand: 10, eliminated: false },
          { playerNumber: 3, ringsInHand: 10, eliminated: false },
        ],
      });

      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      // 3-player games have line threshold of 3
      const result = mutateProcessLine(state, action);
      expect(result.board.collapsedSpaces.size).toBe(3);
    });

    it('should preserve unaffected lines', () => {
      // square8 has lineLength: 3
      const line1Positions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
      ];

      const line2Positions: Position[] = [
        { x: 5, y: 5 },
        { x: 6, y: 5 },
        { x: 7, y: 5 }, // Different area, should be preserved
      ];

      const state = createMinimalState({
        currentPlayer: 1,
        boardSize: 10,
        formedLines: [
          { player: 1, length: 3, positions: line1Positions, direction: { x: 1, y: 0 } },
          { player: 1, length: 3, positions: line2Positions, direction: { x: 1, y: 0 } },
        ],
      });

      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      const result = mutateProcessLine(state, action);

      // Line 2 should be preserved (doesn't share positions)
      expect(result.board.formedLines.length).toBe(1);
      expect(result.board.formedLines[0].positions[0].x).toBe(5);
    });

    it('should throw when current player not found in players array', () => {
      // square8 has lineLength: 3
      const positions: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
      ];

      // Create state with currentPlayer=99 which doesn't exist in players array
      const state = createMinimalState({
        currentPlayer: 99, // Non-existent player
        formedLines: [{ player: 1, length: 3, positions, direction: { x: 1, y: 0 } }],
        players: [
          { playerNumber: 1, ringsInHand: 10, eliminated: false },
          { playerNumber: 2, ringsInHand: 10, eliminated: false },
        ],
      });

      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      expect(() => mutateProcessLine(state, action)).toThrow('LineMutator: Player not found');
    });
  });
});
