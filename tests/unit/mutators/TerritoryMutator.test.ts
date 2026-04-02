/**
 * TerritoryMutator branch coverage tests
 * Tests for src/shared/engine/mutators/TerritoryMutator.ts
 */

import {
  mutateProcessTerritory,
  mutateEliminateStack,
} from '../../../src/shared/engine/mutators/TerritoryMutator';
import type {
  GameState,
  ProcessTerritoryAction,
  EliminateStackAction,
} from '../../../src/shared/engine/types';
import type { BoardType, Position, RingStack, Territory } from '../../../src/shared/types/game';
import { inferTotalRingsInPlay } from '../../utils/fixtures';

function posStr(x: number, y: number): string {
  return `${x},${y}`;
}

function createMinimalState(
  overrides: Partial<{
    currentPhase: string;
    currentPlayer: number;
    boardType: BoardType;
    boardSize: number;
    stacks: Map<string, RingStack>;
    markers: Map<string, { player: number }>;
    territories: Map<string, Territory>;
    collapsedSpaces: Map<string, number>;
    eliminatedRings: Record<number, number>;
    players: Array<{
      playerNumber: number;
      ringsInHand: number;
      eliminated: boolean;
      eliminatedRings: number;
      territorySpaces: number;
    }>;
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
      eliminatedRings: 0,
      score: 0,
      reserveStacks: 0,
      reserveRings: 0,
      territorySpaces: 0,
    }));
  const board = {
    type: boardType,
    size: boardSize,
    stacks: overrides.stacks ?? new Map(),
    markers: overrides.markers ?? new Map(),
    collapsedSpaces: overrides.collapsedSpaces ?? new Map(),
    rings: new Map(),
    territories: overrides.territories ?? new Map(),
    formedLines: [],
    eliminatedRings: overrides.eliminatedRings ?? {},
    geometry: { type: boardType, size: boardSize },
  };

  return {
    board,
    currentPhase: overrides.currentPhase ?? 'territory_processing',
    currentPlayer: overrides.currentPlayer ?? 1,
    players,
    turnNumber: 1,
    gameStatus: 'active',
    moveHistory: [],
    pendingDecision: null,
    victoryCondition: null,
    totalRingsInPlay: inferTotalRingsInPlay(players, board),
    totalRingsEliminated: 0,
  } as unknown as GameState;
}

describe('TerritoryMutator', () => {
  describe('mutateProcessTerritory', () => {
    it('should throw when region is not found', () => {
      const state = createMinimalState({
        territories: new Map(),
      });

      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'non-existent-region',
        playerId: 1,
      };

      expect(() => mutateProcessTerritory(state, action)).toThrow(
        'TerritoryMutator: Region not found'
      );
    });

    it('should collapse all spaces in the region', () => {
      const territories = new Map<string, Territory>([
        [
          'region-1',
          {
            id: 'region-1',
            player: 1,
            spaces: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
            ],
            enclosingLine: [],
          },
        ],
      ]);

      const state = createMinimalState({
        currentPlayer: 1,
        territories,
      });

      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'region-1',
        playerId: 1,
      };

      const result = mutateProcessTerritory(state, action);

      expect(result.board.collapsedSpaces.has('0,0')).toBe(true);
      expect(result.board.collapsedSpaces.has('1,0')).toBe(true);
      expect(result.board.collapsedSpaces.has('2,0')).toBe(true);
      expect(result.board.territories.has('region-1')).toBe(false);
    });

    it('should eliminate stacks inside the region and credit to processing player', () => {
      const territories = new Map<string, Territory>([
        [
          'region-1',
          {
            id: 'region-1',
            player: 1,
            spaces: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
            ],
            enclosingLine: [],
          },
        ],
      ]);

      const stacks = new Map<string, RingStack>([
        [
          '1,0',
          {
            position: { x: 1, y: 0 },
            rings: [2, 2, 1], // Player 2 controls, 3 rings total
            stackHeight: 3,
            capHeight: 2,
            controllingPlayer: 2,
          },
        ],
      ]);

      const state = createMinimalState({
        currentPlayer: 1,
        territories,
        stacks,
        players: [
          {
            playerNumber: 1,
            ringsInHand: 5,
            eliminated: false,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
          {
            playerNumber: 2,
            ringsInHand: 5,
            eliminated: false,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'region-1',
        playerId: 1,
      };

      const result = mutateProcessTerritory(state, action);

      // Stack should be eliminated
      expect(result.board.stacks.has('1,0')).toBe(false);
      // Player 1 should get credit for eliminated rings
      expect(result.board.eliminatedRings[1]).toBe(3);
      expect(result.players.find((p) => p.playerNumber === 1)?.eliminatedRings).toBe(3);
    });

    it('should remove markers in the region', () => {
      const territories = new Map<string, Territory>([
        [
          'region-1',
          {
            id: 'region-1',
            player: 1,
            spaces: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
            ],
            enclosingLine: [],
          },
        ],
      ]);

      const markers = new Map<string, { player: number }>([['0,0', { player: 2 }]]);

      const state = createMinimalState({
        currentPlayer: 1,
        territories,
        markers,
      });

      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'region-1',
        playerId: 1,
      };

      const result = mutateProcessTerritory(state, action);

      expect(result.board.markers.has('0,0')).toBe(false);
    });

    it('should update player territorySpaces count', () => {
      const territories = new Map<string, Territory>([
        [
          'region-1',
          {
            id: 'region-1',
            player: 1,
            spaces: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
            ],
            enclosingLine: [],
          },
        ],
      ]);

      const state = createMinimalState({
        currentPlayer: 1,
        territories,
        players: [
          {
            playerNumber: 1,
            ringsInHand: 5,
            eliminated: false,
            eliminatedRings: 0,
            territorySpaces: 5,
          },
          {
            playerNumber: 2,
            ringsInHand: 5,
            eliminated: false,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'region-1',
        playerId: 1,
      };

      const result = mutateProcessTerritory(state, action);

      expect(result.players.find((p) => p.playerNumber === 1)?.territorySpaces).toBe(8); // 5 + 3
    });

    it('should handle region with no stacks (no internal eliminations)', () => {
      const territories = new Map<string, Territory>([
        [
          'region-1',
          {
            id: 'region-1',
            player: 1,
            spaces: [{ x: 0, y: 0 }],
            enclosingLine: [],
          },
        ],
      ]);

      const state = createMinimalState({
        currentPlayer: 1,
        territories,
        stacks: new Map(), // No stacks
      });

      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'region-1',
        playerId: 1,
      };

      const result = mutateProcessTerritory(state, action);

      expect(result.board.collapsedSpaces.has('0,0')).toBe(true);
      expect(result.board.eliminatedRings[1]).toBeUndefined();
    });

    it('should handle player not found for territory gain (unlikely edge case)', () => {
      const territories = new Map<string, Territory>([
        [
          'region-1',
          {
            id: 'region-1',
            player: 99, // Non-existent player
            spaces: [{ x: 0, y: 0 }],
            enclosingLine: [],
          },
        ],
      ]);

      const state = createMinimalState({
        currentPlayer: 1,
        territories,
      });

      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'region-1',
        playerId: 99,
      };

      // Should not throw, just skip player update
      const result = mutateProcessTerritory(state, action);
      expect(result.board.collapsedSpaces.has('0,0')).toBe(true);
    });
  });

  describe('mutateEliminateStack', () => {
    it('should throw when stack is not found', () => {
      const state = createMinimalState({
        stacks: new Map(),
      });

      const action: EliminateStackAction = {
        type: 'eliminateStack',
        stackPosition: { x: 5, y: 5 },
        playerId: 1,
      };

      expect(() => mutateEliminateStack(state, action)).toThrow(
        'TerritoryMutator: Stack to eliminate not found'
      );
    });

    it('should eliminate the cap from a stack and leave remaining rings', () => {
      const stacks = new Map<string, RingStack>([
        [
          '2,2',
          {
            position: { x: 2, y: 2 },
            rings: [1, 1, 2], // Player 1 cap (height 2), player 2 at bottom
            stackHeight: 3,
            capHeight: 2,
            controllingPlayer: 1,
          },
        ],
      ]);

      const state = createMinimalState({
        currentPlayer: 1,
        stacks,
        players: [
          {
            playerNumber: 1,
            ringsInHand: 5,
            eliminated: false,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
          {
            playerNumber: 2,
            ringsInHand: 5,
            eliminated: false,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      const action: EliminateStackAction = {
        type: 'eliminateStack',
        stackPosition: { x: 2, y: 2 },
        playerId: 1,
      };

      const result = mutateEliminateStack(state, action);

      // Stack should now have only player 2's ring
      const remainingStack = result.board.stacks.get('2,2');
      expect(remainingStack).toBeDefined();
      expect(remainingStack?.rings).toEqual([2]);
      expect(remainingStack?.stackHeight).toBe(1);
      expect(remainingStack?.capHeight).toBe(1);
      expect(remainingStack?.controllingPlayer).toBe(2);

      // Player 1 should have 2 eliminated rings
      expect(result.board.eliminatedRings[1]).toBe(2);
      expect(result.players.find((p) => p.playerNumber === 1)?.eliminatedRings).toBe(2);
    });

    it('should delete stack entirely when all rings eliminated', () => {
      const stacks = new Map<string, RingStack>([
        [
          '2,2',
          {
            position: { x: 2, y: 2 },
            rings: [1, 1], // All same player, cap = 2
            stackHeight: 2,
            capHeight: 2,
            controllingPlayer: 1,
          },
        ],
      ]);

      const state = createMinimalState({
        currentPlayer: 1,
        stacks,
        players: [
          {
            playerNumber: 1,
            ringsInHand: 5,
            eliminated: false,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      const action: EliminateStackAction = {
        type: 'eliminateStack',
        stackPosition: { x: 2, y: 2 },
        playerId: 1,
      };

      const result = mutateEliminateStack(state, action);

      // Stack should be completely removed
      expect(result.board.stacks.has('2,2')).toBe(false);
      expect(result.board.eliminatedRings[1]).toBe(2);
    });

    it('should handle single-ring stack elimination', () => {
      const stacks = new Map<string, RingStack>([
        [
          '3,3',
          {
            position: { x: 3, y: 3 },
            rings: [2], // Single ring
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 2,
          },
        ],
      ]);

      const state = createMinimalState({
        currentPlayer: 1,
        stacks,
        players: [
          {
            playerNumber: 1,
            ringsInHand: 5,
            eliminated: false,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
          {
            playerNumber: 2,
            ringsInHand: 5,
            eliminated: false,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      const action: EliminateStackAction = {
        type: 'eliminateStack',
        stackPosition: { x: 3, y: 3 },
        playerId: 1,
      };

      const result = mutateEliminateStack(state, action);

      expect(result.board.stacks.has('3,3')).toBe(false);
      expect(result.board.eliminatedRings[2]).toBe(1);
      expect(result.players.find((p) => p.playerNumber === 2)?.eliminatedRings).toBe(1);
    });

    it('should correctly calculate cap height for mixed stacks', () => {
      const stacks = new Map<string, RingStack>([
        [
          '4,4',
          {
            position: { x: 4, y: 4 },
            rings: [1, 1, 1, 2, 2], // Player 1 cap of 3
            stackHeight: 5,
            capHeight: 3,
            controllingPlayer: 1,
          },
        ],
      ]);

      const state = createMinimalState({
        currentPlayer: 1,
        stacks,
        players: [
          {
            playerNumber: 1,
            ringsInHand: 5,
            eliminated: false,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
          {
            playerNumber: 2,
            ringsInHand: 5,
            eliminated: false,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      const action: EliminateStackAction = {
        type: 'eliminateStack',
        stackPosition: { x: 4, y: 4 },
        playerId: 1,
      };

      const result = mutateEliminateStack(state, action);

      const remainingStack = result.board.stacks.get('4,4');
      expect(remainingStack).toBeDefined();
      expect(remainingStack?.rings).toEqual([2, 2]);
      expect(remainingStack?.stackHeight).toBe(2);
      expect(remainingStack?.capHeight).toBe(2);
      expect(remainingStack?.controllingPlayer).toBe(2);

      expect(result.board.eliminatedRings[1]).toBe(3);
    });

    it('should update totalRingsEliminated', () => {
      const stacks = new Map<string, RingStack>([
        [
          '1,1',
          {
            position: { x: 1, y: 1 },
            rings: [1, 1],
            stackHeight: 2,
            capHeight: 2,
            controllingPlayer: 1,
          },
        ],
      ]);

      const state = createMinimalState({
        currentPlayer: 1,
        stacks,
      }) as GameState & { totalRingsEliminated: number };
      state.totalRingsEliminated = 5;

      const action: EliminateStackAction = {
        type: 'eliminateStack',
        stackPosition: { x: 1, y: 1 },
        playerId: 1,
      };

      const result = mutateEliminateStack(state, action) as GameState & {
        totalRingsEliminated: number;
      };

      expect(result.totalRingsEliminated).toBe(7); // 5 + 2
    });
  });
});
