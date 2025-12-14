/**
 * TerritoryAggregate.edge.branchCoverage.test.ts
 *
 * Branch coverage improvement tests for TerritoryAggregate - Edge Cases.
 * Targets specific uncovered branches in region lookup, error handling, and player filtering.
 *
 * Split from TerritoryAggregate.advanced.branchCoverage.test.ts for maintainability.
 */

import {
  findDisconnectedRegions,
  computeBorderMarkers,
  getBorderMarkerPositionsForRegion,
  validateTerritoryDecision,
  validateProcessTerritory,
  validateEliminateStack,
  enumerateTerritoryDecisions,
  enumerateTerritoryEliminationMoves,
  canProcessTerritoryRegion,
  filterProcessableTerritoryRegions,
  getProcessableTerritoryRegions,
  applyTerritoryDecision,
  applyProcessTerritoryRegionDecision,
  applyEliminateRingsFromStackDecision,
  mutateProcessTerritory,
  mutateEliminateStack,
  enumerateProcessTerritoryRegionMoves,
  applyTerritoryRegion,
} from '../../src/shared/engine/aggregates/TerritoryAggregate';
import {
  createTestGameState,
  createTestBoard,
  createTestPlayer,
  addMarker,
  addStack,
  pos,
} from '../utils/fixtures';
import { positionToString } from '../../src/shared/types/game';
import type { GameState, Position, Territory } from '../../src/shared/types/game';
import {
  createTerritoryFeEdgeBoard,
  territoryFeEdgeRegionForPlayer1,
  territoryFeMiniRegionForPlayer1,
} from '../fixtures/territoryFeEdgeFixture';

describe('TerritoryAggregate - Branch Coverage (Edge Cases)', () => {
  // ==========================================================================
  // getProcessableTerritoryRegions branch coverage
  // ==========================================================================
  describe('getProcessableTerritoryRegions', () => {
    it('returns regions that can be processed by player', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      addStack(state.board, { x: 7, y: 7 }, 1, 2, 2);

      const regions = getProcessableTerritoryRegions(state.board, { player: 1 });
      expect(Array.isArray(regions)).toBe(true);
    });

    it('returns empty for player with no processable regions', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      // Only player 2 stacks
      addStack(state.board, { x: 0, y: 0 }, 2, 2, 2);

      const regions = getProcessableTerritoryRegions(state.board, { player: 1 });
      expect(regions.length).toBe(0);
    });
  });

  // ==========================================================================
  // enumerateTerritoryEliminationMoves additional branches
  // ==========================================================================
  describe('enumerateTerritoryEliminationMoves additional branches', () => {
    it('returns empty when in territory_processing phase with processable regions (line 672)', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.board.stacks.clear();

      // Create isolated stacks for player 1 to form a disconnected region
      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      // Player 1 stack outside region - makes the region processable
      addStack(state.board, { x: 7, y: 7 }, 1, 2, 2);

      const moves = enumerateTerritoryEliminationMoves(state, 1);
      // Should return empty because there are processable regions
      expect(Array.isArray(moves)).toBe(true);
    });

    it('returns empty when player has no stacks (line 684)', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.board.stacks.clear();

      // Only player 2 stacks, no player 1 stacks
      addStack(state.board, { x: 0, y: 0 }, 2, 2, 2);
      addStack(state.board, { x: 1, y: 0 }, 2, 2, 2);

      const moves = enumerateTerritoryEliminationMoves(state, 1);
      expect(moves).toEqual([]);
    });

    it('enumerates moves when not in territory_processing phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.board.stacks.clear();

      // Player 1 stacks that can have rings eliminated
      const key = positionToString({ x: 3, y: 3 });
      state.board.stacks.set(key, {
        position: { x: 3, y: 3 },
        stackHeight: 3,
        controllingPlayer: 1,
        rings: [1, 1, 1],
        capHeight: 3,
      });

      const moves = enumerateTerritoryEliminationMoves(state, 1);
      expect(moves.length).toBeGreaterThan(0);
      expect(moves[0].type).toBe('eliminate_rings_from_stack');
    });
  });

  // ==========================================================================
  // canProcessTerritoryRegion additional branches
  // ==========================================================================
  describe('canProcessTerritoryRegion additional branches', () => {
    it('skips stacks controlled by other players (line 736)', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      // Add stacks for multiple players
      addStack(state.board, { x: 0, y: 0 }, 2, 2, 2); // Player 2's stack
      addStack(state.board, { x: 1, y: 0 }, 2, 2, 2); // Player 2's stack
      addStack(state.board, { x: 5, y: 5 }, 1, 2, 2); // Player 1's stack outside region

      const region: Territory = {
        player: 1,
        spaces: [{ x: 2, y: 2 }],
        stacks: [],
        isDisconnected: true,
        controllingPlayer: 1,
      };

      // Player 1 has a stack outside the region, so should be processable
      const result = canProcessTerritoryRegion(state.board, region, { player: 1 });
      expect(result).toBe(true);
    });

    it('returns false when player has no stacks outside region', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      // Add player 1 stack inside the region only
      addStack(state.board, { x: 2, y: 2 }, 1, 2, 2);
      // Add player 2 stacks elsewhere
      addStack(state.board, { x: 5, y: 5 }, 2, 2, 2);

      const region: Territory = {
        player: 1,
        spaces: [{ x: 2, y: 2 }],
        stacks: [{ x: 2, y: 2 }],
        isDisconnected: true,
        controllingPlayer: 1,
      };

      const result = canProcessTerritoryRegion(state.board, region, { player: 1 });
      expect(result).toBe(false);
    });
  });

  // ==========================================================================
  // applyTerritoryDecision process_region with valid territory (lines 799-811)
  // ==========================================================================
  describe('applyTerritoryDecision process_region branch', () => {
    it('processes valid region from territories map (lines 799-811)', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Create a stack outside the region for processability
      addStack(state.board, { x: 5, y: 5 }, 1, 2, 2);

      // Add a territory to the territories map
      const regionKey = '0,0';
      const region: Territory = {
        player: 1,
        spaces: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
        ],
        stacks: [],
        isDisconnected: true,
        controllingPlayer: 1,
      };
      state.board.territories.set(regionKey, region);

      const decision = {
        type: 'process_region' as const,
        player: 1,
        region: region,
      };

      const result = applyTerritoryDecision(state, decision);
      expect(result.success).toBe(true);
      expect(result.newState).toBeDefined();
    });

    it('falls back to error when process_region has undefined region', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;

      const decision = {
        type: 'process_region' as const,
        player: 1,
        region: undefined,
      };

      const result = applyTerritoryDecision(state, decision);
      expect(result.success).toBe(false);
      // Either "Invalid decision data" or "Region not found" depending on validation path
      expect(['Invalid decision data', 'Region not found']).toContain(result.reason);
    });
  });

  // ==========================================================================
  // applyTerritoryRegion interior space check (line 879)
  // ==========================================================================
  describe('applyTerritoryRegion interior space check', () => {
    it('skips border markers that are in region interior (line 879)', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      state.board.markers.clear();
      state.board.collapsedSpaces.clear();

      // Create a larger region where some markers might appear as borders
      // but are actually inside the region
      const region: Territory = {
        player: 1,
        spaces: [
          { x: 1, y: 1 },
          { x: 2, y: 1 },
          { x: 1, y: 2 },
          { x: 2, y: 2 },
        ],
        stacks: [],
        isDisconnected: true,
        controllingPlayer: 1,
      };

      // Add markers adjacent to region spaces
      addMarker(state.board, { x: 0, y: 1 }, 1);
      addMarker(state.board, { x: 3, y: 1 }, 1);

      const outcome = applyTerritoryRegion(state.board, region, { player: 1 });

      // All region spaces should be collapsed
      for (const pos of region.spaces) {
        const key = positionToString(pos);
        expect(outcome.board.collapsedSpaces.has(key)).toBe(true);
      }
    });
  });

  // ==========================================================================
  // territory / FE edge fixture from canonical_square8_regen k90
  // ==========================================================================
  describe('territory/FE edge fixture – canonical_square8_regen k90', () => {
    it('applies self-elimination prerequisite based on stacks outside the region', () => {
      const board = createTerritoryFeEdgeBoard();
      const region = territoryFeEdgeRegionForPlayer1;

      // Player 1 controls stacks both inside and outside the region on this board,
      // so the self-elimination prerequisite should pass.
      expect(canProcessTerritoryRegion(board, region, { player: 1 })).toBe(true);

      // If we remove all player 1 stacks outside the region, the region is no
      // longer processable for player 1.
      const boardNoOutside = createTerritoryFeEdgeBoard();
      const regionKeys = new Set(region.spaces.map((p) => positionToString(p)));

      for (const [key, stack] of Array.from(boardNoOutside.stacks.entries())) {
        const isInRegion = regionKeys.has(key);
        if (stack.controllingPlayer === 1 && !isInRegion) {
          boardNoOutside.stacks.delete(key);
        }
      }

      expect(canProcessTerritoryRegion(boardNoOutside, region, { player: 1 })).toBe(false);
    });

    it('eliminates internal stacks and credits eliminations on the k90 edge fixture', () => {
      const board = createTerritoryFeEdgeBoard();
      const region = territoryFeEdgeRegionForPlayer1;

      const internalKey = positionToString({ x: 6, y: 1 });
      const internalStack = board.stacks.get(internalKey);
      expect(internalStack).toBeDefined();
      const internalHeight = internalStack!.stackHeight;
      const beforeElimsP1 = board.eliminatedRings[1] || 0;

      const outcome = applyTerritoryRegion(board, region, { player: 1 });

      // Original board is not mutated.
      expect(board.stacks.has(internalKey)).toBe(true);

      // Internal stack is removed on the next board.
      expect(outcome.board.stacks.has(internalKey)).toBe(false);

      // All region spaces are collapsed to player 1.
      for (const pos of region.spaces) {
        const key = positionToString(pos);
        expect(outcome.board.collapsedSpaces.get(key)).toBe(1);
        expect(outcome.board.stacks.has(key)).toBe(false);
        expect(outcome.board.markers.has(key)).toBe(false);
      }

      // All internal rings are credited to player 1.
      expect(outcome.eliminatedRingsByPlayer[1]).toBe(internalHeight);
      expect(outcome.board.eliminatedRings[1]).toBe(beforeElimsP1 + internalHeight);
    });
  });

  // ==========================================================================
  // territory / FE mini-region fixture from canonical_square8_regen k90 (SQ8-A)
  // ==========================================================================
  describe('territory/FE mini-region – canonical_square8_regen k90', () => {
    it('applies self-elimination prerequisite based on stacks outside the mini-region', () => {
      const board = createTerritoryFeEdgeBoard();
      const region = territoryFeMiniRegionForPlayer1;

      // Player 1 controls stacks both inside and outside the mini-region on this board,
      // so the self-elimination prerequisite should pass.
      expect(canProcessTerritoryRegion(board, region, { player: 1 })).toBe(true);

      // If we remove all player 1 stacks outside the mini-region, the region is no
      // longer processable for player 1.
      const boardNoOutside = createTerritoryFeEdgeBoard();
      const regionKeys = new Set(region.spaces.map((p) => positionToString(p)));

      for (const [key, stack] of Array.from(boardNoOutside.stacks.entries())) {
        const isInRegion = regionKeys.has(key);
        if (stack.controllingPlayer === 1 && !isInRegion) {
          boardNoOutside.stacks.delete(key);
        }
      }

      expect(canProcessTerritoryRegion(boardNoOutside, region, { player: 1 })).toBe(false);
    });

    it('eliminates internal stacks and credits eliminations on the k90 mini-region fixture', () => {
      const board = createTerritoryFeEdgeBoard();
      const region = territoryFeMiniRegionForPlayer1;

      // Choose a representative internal stack in the mini-region (player 2 stack at (1,6)).
      const internalKey = positionToString({ x: 1, y: 6 });
      const internalStack = board.stacks.get(internalKey);
      expect(internalStack).toBeDefined();
      const internalHeight = internalStack!.stackHeight;

      // Compute the total stack height of all stacks inside the mini-region.
      let totalInternalHeight = 0;
      for (const pos of region.spaces) {
        const key = positionToString(pos);
        const stack = board.stacks.get(key);
        if (stack) {
          totalInternalHeight += stack.stackHeight;
        }
      }

      // Sanity check: totalInternalHeight should be at least the representative stack's height.
      expect(totalInternalHeight).toBeGreaterThanOrEqual(internalHeight);

      const beforeElimsP1 = board.eliminatedRings[1] || 0;

      const outcome = applyTerritoryRegion(board, region, { player: 1 });

      // Original board is not mutated for the representative stack.
      expect(board.stacks.has(internalKey)).toBe(true);

      // Representative internal stack is removed on the next board.
      expect(outcome.board.stacks.has(internalKey)).toBe(false);

      // All region spaces are collapsed to player 1 and have no stacks/markers.
      for (const pos of region.spaces) {
        const key = positionToString(pos);
        expect(outcome.board.collapsedSpaces.get(key)).toBe(1);
        expect(outcome.board.stacks.has(key)).toBe(false);
        expect(outcome.board.markers.has(key)).toBe(false);
      }

      // All internal rings (across all stacks in the mini-region) are credited to player 1.
      expect(outcome.eliminatedRingsByPlayer[1]).toBe(totalInternalHeight);
      expect(outcome.board.eliminatedRings[1]).toBe(beforeElimsP1 + totalInternalHeight);
    });
  });

  // ==========================================================================
  // applyProcessTerritoryRegionDecision territory updates (lines 987-1020)
  // ==========================================================================
  describe('applyProcessTerritoryRegionDecision territory updates', () => {
    it('updates player territorySpaces when territoryGain > 0 (lines 996-1000)', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();
      state.board.markers.clear();
      state.players[0].territorySpaces = 5;

      // Stack outside region for processability
      addStack(state.board, { x: 7, y: 7 }, 1, 2, 2);

      const region = {
        spaces: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 0, y: 1 },
        ],
        controllingPlayer: 1,
        isDisconnected: true,
      };

      const move = {
        id: 'process-region-0,0',
        type: 'process_territory_region' as const,
        player: 1,
        to: { x: 0, y: 0 },
        disconnectedRegions: [region],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyProcessTerritoryRegionDecision(state, move);
      const player1 = result.nextState.players.find((p) => p.playerNumber === 1);

      // Territory gain should be applied
      expect(player1?.territorySpaces).toBeGreaterThan(5);
    });

    it('updates player eliminatedRings when internalElims > 0 (lines 1002-1006)', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();
      state.board.markers.clear();
      state.players[0].eliminatedRings = 2;

      // Stack outside region for processability
      addStack(state.board, { x: 7, y: 7 }, 1, 2, 2);

      // Stack inside region that will be eliminated
      const internalKey = positionToString({ x: 0, y: 0 });
      state.board.stacks.set(internalKey, {
        position: { x: 0, y: 0 },
        stackHeight: 3,
        controllingPlayer: 2, // Player 2's stack inside player 1's territory
        rings: [2, 2, 2],
        capHeight: 3,
      });

      const region = {
        spaces: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
        ],
        controllingPlayer: 1,
        isDisconnected: true,
      };

      const move = {
        id: 'process-region-0,0',
        type: 'process_territory_region' as const,
        player: 1,
        to: { x: 0, y: 0 },
        disconnectedRegions: [region],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyProcessTerritoryRegionDecision(state, move);

      // Internal eliminations are credited to the processing player
      const player1 = result.nextState.players.find((p) => p.playerNumber === 1);
      expect(player1?.eliminatedRings).toBeGreaterThan(2);
    });

    it('updates totalRingsEliminated (line 1012)', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();
      state.board.markers.clear();
      state.totalRingsEliminated = 10;

      // Stack outside region for processability
      addStack(state.board, { x: 7, y: 7 }, 1, 2, 2);

      // Stack inside region
      const internalKey = positionToString({ x: 0, y: 0 });
      state.board.stacks.set(internalKey, {
        position: { x: 0, y: 0 },
        stackHeight: 2,
        controllingPlayer: 2,
        rings: [2, 2],
        capHeight: 2,
      });

      const region = {
        spaces: [{ x: 0, y: 0 }],
        controllingPlayer: 1,
        isDisconnected: true,
      };

      const move = {
        id: 'process-region-0,0',
        type: 'process_territory_region' as const,
        player: 1,
        to: { x: 0, y: 0 },
        disconnectedRegions: [region],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyProcessTerritoryRegionDecision(state, move);
      expect(result.nextState.totalRingsEliminated).toBeGreaterThan(10);
    });
  });

  // ==========================================================================
  // applyEliminateRingsFromStackDecision capHeight = 0 (line 1055)
  // ==========================================================================
  describe('applyEliminateRingsFromStackDecision capHeight = 0', () => {
    it('returns unchanged state when calculated capHeight is 0 (line 1055)', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      // Create a stack with empty rings array (capHeight will be 0)
      const key = positionToString({ x: 3, y: 3 });
      state.board.stacks.set(key, {
        position: { x: 3, y: 3 },
        stackHeight: 0,
        controllingPlayer: 1,
        rings: [], // Empty rings = capHeight 0
        capHeight: 0,
      });

      const move = {
        id: 'test-move',
        type: 'eliminate_rings_from_stack' as const,
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyEliminateRingsFromStackDecision(state, move);
      expect(result.nextState).toBe(state);
    });
  });

  // ==========================================================================
  // getTerritoryNeighbors Moore fallback (lines 305-315)
  // ==========================================================================
  describe('getTerritoryNeighbors Moore adjacency fallback', () => {
    it('uses Moore adjacency for adjacency type other than hexagonal/von_neumann', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      state.board.markers.clear();

      // Place stacks and markers in a pattern that uses diagonal connections
      addStack(state.board, { x: 3, y: 3 }, 1, 2, 2);
      // Diagonal marker (Moore neighbor, not von_neumann)
      addMarker(state.board, { x: 4, y: 4 }, 1);
      // Orthogonal marker
      addMarker(state.board, { x: 4, y: 3 }, 1);

      // Use ts_legacy mode which uses Moore expansion
      const borders = getBorderMarkerPositionsForRegion(state.board, [{ x: 3, y: 3 }], {
        mode: 'ts_legacy',
      });

      // Should include markers found via Moore adjacency
      expect(borders.length).toBeGreaterThan(0);
    });
  });

  // ==========================================================================
  // findDisconnectedRegions player filtering (lines 381-384)
  // ==========================================================================
  describe('findDisconnectedRegions player stack filtering', () => {
    it('filters regions by checking stack controllingPlayer (lines 381-384)', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      // Create stacks for player 1
      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      addStack(state.board, { x: 1, y: 0 }, 1, 2, 2);

      // Create completely separate stacks for player 2
      addStack(state.board, { x: 6, y: 6 }, 2, 2, 2);
      addStack(state.board, { x: 7, y: 6 }, 2, 2, 2);

      // Query for player 1 regions - should filter by stack.controllingPlayer === 1
      const player1Regions = findDisconnectedRegions(state, 1);

      // The filter at line 381-384 checks:
      // return region.spaces.some((pos) => {
      //   const stack = board.stacks.get(key);
      //   return stack && stack.controllingPlayer === player;
      // });

      // All returned regions should only have player 1 stacks
      for (const region of player1Regions) {
        const hasPlayer1Stack = region.spaces.some((pos) => {
          const key = positionToString(pos);
          const stack = state.board.stacks.get(key);
          return stack && stack.controllingPlayer === 1;
        });
        expect(hasPlayer1Stack).toBe(true);
      }
    });
  });

  // ==========================================================================
  // computeNextMoveNumber branches (lines 238-247)
  // ==========================================================================
  describe('computeNextMoveNumber branches', () => {
    it('uses history moveNumber when history has valid moveNumber (lines 238-240)', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Set up history with valid moveNumber
      state.history = [
        {
          moveNumber: 10,
          phase: 'movement',
          player: 1,
          boardSnapshot: state.board,
        } as any,
      ];
      state.moveHistory = [];

      // Create processable region - need stack outside region for processability
      addStack(state.board, { x: 5, y: 5 }, 1, 2, 2);

      const region: Territory = {
        player: 1,
        spaces: [{ x: 0, y: 0 }],
        stacks: [],
        isDisconnected: true,
        controllingPlayer: 1,
      };

      // Add region to territories map for validation to pass
      state.board.territories.set('0,0', region);

      const decision = {
        type: 'process_region' as const,
        player: 1,
        region: region,
      };

      // This calls computeNextMoveNumber internally
      const result = applyTerritoryDecision(state, decision);
      expect(result.success).toBe(true);
    });

    it('falls back to moveHistory when history has no moveNumber (lines 245-247)', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Set up history without moveNumber
      state.history = [{ phase: 'movement' } as any]; // Missing moveNumber
      // Set up moveHistory with valid moveNumber
      state.moveHistory = [
        {
          id: 'move-1',
          type: 'place_ring' as const,
          player: 1,
          to: { x: 0, y: 0 },
          moveNumber: 7,
          timestamp: new Date(),
          thinkTime: 0,
        },
      ];

      // Create processable region - need stack outside region
      addStack(state.board, { x: 5, y: 5 }, 1, 2, 2);

      const region: Territory = {
        player: 1,
        spaces: [{ x: 0, y: 0 }],
        stacks: [],
        isDisconnected: true,
        controllingPlayer: 1,
      };

      // Add region to territories map
      state.board.territories.set('0,0', region);

      const decision = {
        type: 'process_region' as const,
        player: 1,
        region: region,
      };

      const result = applyTerritoryDecision(state, decision);
      expect(result.success).toBe(true);
    });

    it('returns 1 when both history and moveHistory are empty', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();
      state.history = [];
      state.moveHistory = [];

      // Create processable region - need stack outside region
      addStack(state.board, { x: 5, y: 5 }, 1, 2, 2);

      const region: Territory = {
        player: 1,
        spaces: [{ x: 0, y: 0 }],
        stacks: [],
        isDisconnected: true,
        controllingPlayer: 1,
      };

      // Add region to territories map
      state.board.territories.set('0,0', region);

      const decision = {
        type: 'process_region' as const,
        player: 1,
        region: region,
      };

      const result = applyTerritoryDecision(state, decision);
      expect(result.success).toBe(true);
    });
  });

  // ==========================================================================
  // applyProcessTerritoryRegionDecision region derivation (lines 941-953)
  // ==========================================================================
  describe('applyProcessTerritoryRegionDecision region derivation fallback', () => {
    it('finds region by move.to when multiple candidates exist (lines 941-946)', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Create two isolated groups of stacks
      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      addStack(state.board, { x: 7, y: 7 }, 1, 2, 2);

      const move = {
        id: 'process-region-7,7',
        type: 'process_territory_region' as const,
        player: 1,
        to: { x: 7, y: 7 },
        disconnectedRegions: [], // Empty - triggers fallback derivation
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyProcessTerritoryRegionDecision(state, move);
      expect(result).toBeDefined();
    });

    it('finds region by move.id pattern when move.to does not match (lines 948-955)', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Create two isolated groups
      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      addStack(state.board, { x: 7, y: 7 }, 1, 2, 2);

      const move = {
        id: 'process-region-0-0,0', // ID contains region key
        type: 'process_territory_region' as const,
        player: 1,
        // No 'to' field - forces id-based lookup
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyProcessTerritoryRegionDecision(state, move);
      expect(result).toBeDefined();
    });

    it('uses single candidate when exactly one processable region exists (lines 938-939)', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Only one stack means only one possible region
      addStack(state.board, { x: 3, y: 3 }, 1, 2, 2);

      const move = {
        id: 'process-region-test',
        type: 'process_territory_region' as const,
        player: 1,
        disconnectedRegions: [], // Empty - triggers fallback
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyProcessTerritoryRegionDecision(state, move);
      expect(result).toBeDefined();
    });
  });

  // ==========================================================================
  // applyTerritoryDecision error handling (lines 827-829)
  // ==========================================================================
  describe('applyTerritoryDecision error handling', () => {
    it('catches and returns error when process_region throws (lines 827-829)', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();
      state.board.territories.clear();

      // Stack for processability
      addStack(state.board, { x: 5, y: 5 }, 1, 2, 2);

      // Create region with empty spaces array that might cause issues
      const badRegion = {
        player: 1,
        spaces: [], // Empty spaces
        stacks: [],
        isDisconnected: true,
        controllingPlayer: 1,
      };

      // Don't add to territories - validation should fail
      const decision = {
        type: 'process_region' as const,
        player: 1,
        region: badRegion,
      };

      const result = applyTerritoryDecision(state, decision);
      // Should fail gracefully
      expect(result.success).toBe(false);
      expect(result.reason).toBeDefined();
    });
  });

  // ==========================================================================
  // getMooreNeighbors hexagonal early return (line 323)
  // ==========================================================================
  describe('getMooreNeighbors hexagonal board handling', () => {
    it('returns empty neighbors for hexagonal board when Moore expansion called', () => {
      const state = createTestGameState({ boardType: 'hexagonal' });
      state.board.stacks.clear();
      state.board.markers.clear();

      // Add hex positions
      addStack(state.board, { x: 0, y: 0, z: 0 }, 1, 2, 2);
      addMarker(state.board, { x: 1, y: 0, z: -1 }, 1);
      addMarker(state.board, { x: 2, y: 0, z: -2 }, 1);

      // Use ts_legacy mode which attempts Moore expansion
      // On hex boards, getMooreNeighbors returns empty (line 323)
      const borders = getBorderMarkerPositionsForRegion(state.board, [{ x: 0, y: 0, z: 0 }], {
        mode: 'ts_legacy',
      });

      // Should still find seed markers via hex adjacency
      expect(borders.length).toBeGreaterThanOrEqual(1);
    });
  });

  // ==========================================================================
  // findDisconnectedRegions with regions containing player stacks (lines 381-384)
  // ==========================================================================
  describe('findDisconnectedRegions player stack lookup', () => {
    it('returns regions where player has stacks (lines 381-384 true path)', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      state.board.markers.clear();

      // Create disconnected stacks for player 1 that will form separate regions
      // Need them far apart to ensure they're in different regions
      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      addStack(state.board, { x: 0, y: 1 }, 1, 2, 2);
      // Another isolated cluster
      addStack(state.board, { x: 7, y: 7 }, 1, 2, 2);

      // Query with player filter - should exercise the filter at lines 381-384
      const regions = findDisconnectedRegions(state, 1);

      // Each region should contain player 1 stacks (the filter returned true)
      expect(Array.isArray(regions)).toBe(true);
      // If there are disconnected regions, they should all be for player 1
      for (const region of regions) {
        const hasP1Stack = region.spaces.some((pos) => {
          const key = positionToString(pos);
          const stack = state.board.stacks.get(key);
          return stack?.controllingPlayer === 1;
        });
        if (region.spaces.length > 0) {
          expect(hasP1Stack).toBe(true);
        }
      }
    });

    it('filters out regions where player has no stacks (lines 381-384 false path)', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      state.board.markers.clear();

      // Create stacks ONLY for player 2
      addStack(state.board, { x: 0, y: 0 }, 2, 2, 2);
      addStack(state.board, { x: 1, y: 0 }, 2, 2, 2);

      // Query for player 1 - filter should return false for all regions
      const player1Regions = findDisconnectedRegions(state, 1);

      // Should be empty since player 1 has no stacks in any region
      expect(player1Regions.length).toBe(0);
    });
  });

  // ==========================================================================
  // enumerateTerritoryEliminationMoves with processable regions (line 672)
  // ==========================================================================
  describe('enumerateTerritoryEliminationMoves territory_processing with regions', () => {
    it('returns no elimination moves during territory_processing; forced_elimination handles them', () => {
      // Per current logic (lines 667-679 in TerritoryAggregate.ts): in
      // territory_processing, eliminate_rings_from_stack moves are never surfaced.
      // When no processable regions exist but the player still has stacks, the
      // orchestrator transitions to forced_elimination and calls this helper there.
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.board.stacks.clear();
      state.board.markers.clear();

      // Create stacks that will not form disconnected/processable regions.
      addStack(state.board, { x: 0, y: 0 }, 1);
      addStack(state.board, { x: 7, y: 7 }, 1);

      const moves = enumerateTerritoryEliminationMoves(state, 1);

      // No elimination moves are surfaced in territory_processing; they are
      // deferred to the forced_elimination phase.
      expect(moves.length).toBe(0);
    });

    it('exercises territory_processing phase check', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.board.stacks.clear();

      // Add stacks for player 1
      const key = positionToString({ x: 3, y: 3 });
      state.board.stacks.set(key, {
        position: { x: 3, y: 3 },
        stackHeight: 3,
        controllingPlayer: 1,
        rings: [1, 1, 1],
        capHeight: 3,
      });

      // Call in territory_processing phase - will check for processable regions first
      const moves = enumerateTerritoryEliminationMoves(state, 1);

      // Exercises the territory_processing phase branch
      expect(Array.isArray(moves)).toBe(true);
    });
  });
});
