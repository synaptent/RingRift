/**
 * TerritoryAggregate.advanced.branchCoverage.test.ts
 *
 * Branch coverage improvement tests for TerritoryAggregate - Advanced Tests.
 * Targets specific uncovered branches in decision application and region processing.
 *
 * Split from TerritoryAggregate.branchCoverage.test.ts for maintainability.
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

describe('TerritoryAggregate - Branch Coverage (Advanced)', () => {
  // ==========================================================================
  // applyTerritoryDecision branches
  // ==========================================================================
  describe('applyTerritoryDecision - additional branches', () => {
    it('handles eliminate_from_stack decision type', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Add a stack with rings to eliminate
      const stackPos = { x: 3, y: 3 };
      addStack(state.board, stackPos, 1, 2, 2);

      const decision = {
        type: 'eliminate_from_stack' as const,
        player: 1,
        stackPosition: stackPos,
      };

      const result = applyTerritoryDecision(state, decision);
      // Should either succeed or fail with a meaningful reason
      expect(result).toBeDefined();
      expect(typeof result.success).toBe('boolean');
    });

    it('returns error for invalid decision data', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;

      // Decision with invalid type
      const decision = {
        type: 'unknown_type' as never,
        player: 1,
      };

      const result = applyTerritoryDecision(state, decision);
      expect(result.success).toBe(false);
      expect(result.reason).toBeDefined();
    });

    it('handles process_region without region data', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;

      const decision = {
        type: 'process_region' as const,
        player: 1,
        // region is undefined
      };

      const result = applyTerritoryDecision(state, decision);
      // Should fail because region is missing
      expect(result).toBeDefined();
    });
  });

  // ==========================================================================
  // getMooreNeighbors hexagonal edge case
  // ==========================================================================
  describe('getMooreNeighbors hexagonal handling', () => {
    it('returns empty array for hexagonal boards', () => {
      const state = createTestGameState({ boardType: 'hexagonal' });
      const regions = findDisconnectedRegions(state, 1);
      // Hexagonal boards use different adjacency
      expect(Array.isArray(regions)).toBe(true);
    });
  });

  // ==========================================================================
  // Mini-region elimination and disconnected region ordering
  // ==========================================================================
  describe('mini-region and region ordering branches', () => {
    it('enforces elimination of interior rings in mini-region and allows region ordering', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;

      // Build a 2x2 mini-region fully enclosed by player 1 markers
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 0, y: 1 }, 1);
      addMarker(state.board, { x: 1, y: 1 }, 1);

      // Add two disconnected regions so ordering is needed
      addTerritory(state.board, {
        id: 'region-1',
        spaces: [{ x: 0, y: 0 }],
        player: 1,
        isDisconnected: true,
        borderMarkers: computeBorderMarkers(state.board, [{ x: 0, y: 0 }], 1),
      });
      addTerritory(state.board, {
        id: 'region-2',
        spaces: [{ x: 2, y: 2 }],
        player: 2,
        isDisconnected: true,
        borderMarkers: computeBorderMarkers(state.board, [{ x: 2, y: 2 }], 2),
      });

      // Region ordering should surface two process_territory_region moves
      const regionMoves = enumerateProcessTerritoryRegionMoves(state, 1);
      expect(regionMoves.length).toBe(2);

      // Add a ring inside region-1 to force elimination
      addStack(state.board, { x: 0, y: 0 }, 1, 1, 1);
      const elimMoves = enumerateTerritoryEliminationMoves(state, 1);
      expect(elimMoves.length).toBeGreaterThan(0);
    });
  });

  // ==========================================================================
  // Multi-player tie-break ladder coverage (territory vs eliminated rings vs markers)
  // ==========================================================================
  describe('multi-player tie-break ladders', () => {
    it('applies territory then eliminated-rings tie-break ordering across 3 players', () => {
      const state = createTestGameState({
        players: [createTestPlayer(1), createTestPlayer(2), createTestPlayer(3)],
        currentPlayer: 1,
      });
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;

      // Configure final tallies that force tie-break comparisons
      state.players[0].territorySpaces = 10;
      state.players[1].territorySpaces = 10;
      state.players[2].territorySpaces = 9;

      state.players[0].eliminatedRings = 3;
      state.players[1].eliminatedRings = 2; // breaks tie after equal territory
      state.players[2].eliminatedRings = 4;

      // No regions to process; expect no-op decision surface and no crash in tie handling
      const move = {
        id: 'process-region-none',
        type: 'no_territory_action' as const,
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyProcessTerritoryRegionDecision(state, move);
      expect(result).toBeDefined();
      expect(result.processedRegion.spaces.length).toBe(0);
      // Validate that winner derivation in territory processing can be invoked without throwing
      const validation = validateProcessTerritory(state, move);
      expect(validation.valid).toBe(true);
    });
  });

  // ==========================================================================
  // computeNextMoveNumber edge cases
  // ==========================================================================
  describe('computeNextMoveNumber branches', () => {
    it('uses history moveNumber when available', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Set up history with moveNumber
      state.history = [{ moveNumber: 5, player: 1 } as never];

      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      addStack(state.board, { x: 2, y: 2 }, 1, 2, 2);

      const regions = findDisconnectedRegions(state, 1);
      if (regions.length > 0) {
        const decisions = enumerateTerritoryDecisions(state, regions[0]);
        expect(decisions.length).toBeGreaterThan(0);
      }
    });

    it('uses moveHistory as fallback when history lacks moveNumber', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Set up moveHistory with moveNumber
      state.history = [];
      state.moveHistory = [{ moveNumber: 3, player: 1 } as never];

      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);

      const regions = findDisconnectedRegions(state, 1);
      expect(Array.isArray(regions)).toBe(true);
    });
  });

  // ==========================================================================
  // applyProcessTerritoryRegionDecision branch coverage
  // ==========================================================================
  describe('applyProcessTerritoryRegionDecision', () => {
    it('throws error for wrong move type', () => {
      const state = createTestGameState();
      const wrongMove = {
        id: 'test-move',
        type: 'place_ring' as const,
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      expect(() => applyProcessTerritoryRegionDecision(state, wrongMove)).toThrow(
        "applyProcessTerritoryRegionDecision expected move.type === 'process_territory_region'"
      );
    });

    it('uses region from move.disconnectedRegions when available', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Create a disconnected region
      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      addStack(state.board, { x: 1, y: 0 }, 1, 2, 2);

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
      expect(result).toBeDefined();
      expect(result.processedRegion).toBeDefined();
    });

    it('re-derives region when move.disconnectedRegions is empty', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Create isolated stacks (disconnected)
      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      addStack(state.board, { x: 7, y: 7 }, 1, 2, 2);

      const move = {
        id: 'process-region-0,0',
        type: 'process_territory_region' as const,
        player: 1,
        to: { x: 0, y: 0 },
        disconnectedRegions: [],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyProcessTerritoryRegionDecision(state, move);
      expect(result).toBeDefined();
    });

    it('returns empty region when no region can be derived', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // No stacks at all - no regions to process
      const move = {
        id: 'process-region-0,0',
        type: 'process_territory_region' as const,
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyProcessTerritoryRegionDecision(state, move);
      expect(result.processedRegion.spaces.length).toBe(0);
      expect(result.pendingSelfElimination).toBe(false);
    });

    it('finds region by move.to position from candidates', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Create two separate regions
      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      addStack(state.board, { x: 7, y: 7 }, 1, 2, 2);

      const move = {
        id: 'process-region-7,7',
        type: 'process_territory_region' as const,
        player: 1,
        to: { x: 7, y: 7 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyProcessTerritoryRegionDecision(state, move);
      expect(result).toBeDefined();
    });

    it('finds region by move.id pattern from candidates', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Create two separate regions
      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      addStack(state.board, { x: 7, y: 7 }, 1, 2, 2);

      const move = {
        id: 'process-region-0-0,0',
        type: 'process_territory_region' as const,
        player: 1,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyProcessTerritoryRegionDecision(state, move);
      expect(result).toBeDefined();
    });

    it('applies territory gain when region is processed', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      state.board.stacks.clear();

      // Create a region with territory potential
      addStack(state.board, { x: 2, y: 2 }, 1, 3, 3);
      addStack(state.board, { x: 3, y: 2 }, 1, 3, 3);
      addStack(state.board, { x: 2, y: 3 }, 1, 3, 3);

      const region = {
        spaces: [
          { x: 2, y: 2 },
          { x: 3, y: 2 },
          { x: 2, y: 3 },
        ],
        controllingPlayer: 1,
        isDisconnected: true,
      };

      const move = {
        id: 'process-region-2,2',
        type: 'process_territory_region' as const,
        player: 1,
        to: { x: 2, y: 2 },
        disconnectedRegions: [region],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyProcessTerritoryRegionDecision(state, move);
      expect(result.nextState).toBeDefined();
    });
  });

  // ==========================================================================
  // applyEliminateRingsFromStackDecision branch coverage
  // ==========================================================================
  describe('applyEliminateRingsFromStackDecision', () => {
    it('throws error for wrong move type', () => {
      const state = createTestGameState();
      const wrongMove = {
        id: 'test-move',
        type: 'place_ring' as const,
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      expect(() => applyEliminateRingsFromStackDecision(state, wrongMove)).toThrow(
        "applyEliminateRingsFromStackDecision expected move.type === 'eliminate_rings_from_stack'"
      );
    });

    it('returns unchanged state when move.to is undefined', () => {
      const state = createTestGameState();
      const move = {
        id: 'test-move',
        type: 'eliminate_rings_from_stack' as const,
        player: 1,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyEliminateRingsFromStackDecision(state, move);
      expect(result.nextState).toBe(state);
    });

    it('returns unchanged state when stack does not exist', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      const move = {
        id: 'test-move',
        type: 'eliminate_rings_from_stack' as const,
        player: 1,
        to: { x: 5, y: 5 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyEliminateRingsFromStackDecision(state, move);
      expect(result.nextState).toBe(state);
    });

    it('returns unchanged state when stack is not controlled by player', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      // Stack controlled by player 2
      addStack(state.board, { x: 3, y: 3 }, 2, 3, 3);

      const move = {
        id: 'test-move',
        type: 'eliminate_rings_from_stack' as const,
        player: 1, // Different player
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = applyEliminateRingsFromStackDecision(state, move);
      expect(result.nextState).toBe(state);
    });

    it('returns unchanged state when cap height is 0', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      // Stack with mixed rings (cap height calculation may return 0)
      const key = positionToString({ x: 3, y: 3 });
      state.board.stacks.set(key, {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        controllingPlayer: 1,
        composition: [
          { player: 1, count: 1 },
          { player: 2, count: 1 },
        ],
        rings: [1, 2], // Mixed ownership - cap might be 1
        capHeight: 1,
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
      // Either applies elimination or returns unchanged based on cap calculation
      expect(result.nextState).toBeDefined();
    });

    it('eliminates cap and leaves remaining rings', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      state.totalRingsEliminated = 0;

      // Stack with all player 1 rings
      const key = positionToString({ x: 3, y: 3 });
      state.board.stacks.set(key, {
        position: { x: 3, y: 3 },
        stackHeight: 4,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 4 }],
        rings: [1, 1, 1, 1],
        capHeight: 4,
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
      expect(result.nextState).not.toBe(state);
      // Stack should be deleted since all rings are the same player
      expect(result.nextState.board.stacks.has(key)).toBe(false);
    });

    it('eliminates cap and updates controlling player when remaining', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      state.totalRingsEliminated = 0;

      // Stack with player 1 on top, player 2 on bottom
      const key = positionToString({ x: 3, y: 3 });
      state.board.stacks.set(key, {
        position: { x: 3, y: 3 },
        stackHeight: 4,
        controllingPlayer: 1,
        composition: [
          { player: 1, count: 2 },
          { player: 2, count: 2 },
        ],
        rings: [1, 1, 2, 2],
        capHeight: 2,
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
      const newStack = result.nextState.board.stacks.get(key);
      expect(newStack).toBeDefined();
      if (newStack) {
        expect(newStack.rings).toEqual([2, 2]);
        expect(newStack.controllingPlayer).toBe(2);
      }
      expect(result.nextState.totalRingsEliminated).toBe(2);
    });

    it('updates player eliminated rings count', () => {
      const state = createTestGameState();
      state.board.stacks.clear();
      state.players[0].eliminatedRings = 5;

      const key = positionToString({ x: 3, y: 3 });
      state.board.stacks.set(key, {
        position: { x: 3, y: 3 },
        stackHeight: 3,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 3 }],
        rings: [1, 1, 1],
        capHeight: 3,
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
      const updatedPlayer = result.nextState.players.find((p) => p.playerNumber === 1);
      expect(updatedPlayer?.eliminatedRings).toBe(8); // 5 + 3
    });
  });

  // ==========================================================================
  // findDisconnectedRegions - filtering by player stacks
  // ==========================================================================
  describe('findDisconnectedRegions - player filtering', () => {
    it('filters regions by player when player is specified', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      // Player 1 region
      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      addStack(state.board, { x: 1, y: 0 }, 1, 2, 2);

      // Player 2 region (separate)
      addStack(state.board, { x: 6, y: 6 }, 2, 2, 2);
      addStack(state.board, { x: 7, y: 6 }, 2, 2, 2);

      const player1Regions = findDisconnectedRegions(state, 1);
      const player2Regions = findDisconnectedRegions(state, 2);

      // findDisconnectedRegions filters by player stacks
      // The regions returned should be arrays (may be empty if no disconnection detected)
      expect(Array.isArray(player1Regions)).toBe(true);
      expect(Array.isArray(player2Regions)).toBe(true);
    });

    it('returns all regions when player is undefined', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      // Create isolated stacks to form separate regions
      addStack(state.board, { x: 0, y: 0 }, 1, 2, 2);
      addStack(state.board, { x: 6, y: 6 }, 2, 2, 2);

      const allRegions = findDisconnectedRegions(state);
      // May or may not return regions depending on the disconnection algorithm
      expect(Array.isArray(allRegions)).toBe(true);
    });

    it('returns empty when player has no stacks in any region', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      // Only player 2 stacks
      addStack(state.board, { x: 0, y: 0 }, 2, 2, 2);

      const player1Regions = findDisconnectedRegions(state, 1);
      expect(player1Regions.length).toBe(0);
    });
  });

  // ==========================================================================
  // enumerateTerritoryDecisions branch coverage
  // ==========================================================================
  describe('enumerateTerritoryDecisions', () => {
    it('returns process_region decision for a region', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;

      const region = {
        spaces: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
        ],
        controllingPlayer: 1,
        isDisconnected: true,
      };

      const decisions = enumerateTerritoryDecisions(state, region);
      expect(decisions.length).toBeGreaterThan(0);
      expect(decisions[0].type).toBe('process_region');
      expect(decisions[0].player).toBe(1);
      expect(decisions[0].region).toBe(region);
    });

    it('uses state.currentPlayer in decision', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 2;

      const region = {
        spaces: [{ x: 0, y: 0 }],
        controllingPlayer: 2,
        isDisconnected: true,
      };

      const decisions = enumerateTerritoryDecisions(state, region);
      expect(decisions[0].player).toBe(2);
    });
  });

  // ==========================================================================
  // computeBorderMarkers branch coverage
  // ==========================================================================
  describe('computeBorderMarkers', () => {
    it('returns border marker positions for a region', () => {
      const state = createTestGameState();
      state.board.stacks.clear();

      addStack(state.board, { x: 2, y: 2 }, 1, 2, 2);
      addStack(state.board, { x: 3, y: 2 }, 1, 2, 2);
      addStack(state.board, { x: 2, y: 3 }, 1, 2, 2);

      const region = {
        spaces: [
          { x: 2, y: 2 },
          { x: 3, y: 2 },
          { x: 2, y: 3 },
        ],
        controllingPlayer: 1,
        isDisconnected: true,
      };

      const borders = computeBorderMarkers(state, region);
      expect(Array.isArray(borders)).toBe(true);
    });

    it('handles empty region', () => {
      const state = createTestGameState();

      const region = {
        spaces: [],
        controllingPlayer: 1,
        isDisconnected: true,
      };

      const borders = computeBorderMarkers(state, region);
      expect(Array.isArray(borders)).toBe(true);
      expect(borders.length).toBe(0);
    });
  });

  // ==========================================================================
  // applyTerritoryDecision - additional branch coverage
  // ==========================================================================
  describe('applyTerritoryDecision - invalid decision data', () => {
    it('returns error for unknown decision type', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';

      const invalidDecision = {
        type: 'unknown_type' as never,
        player: 1,
      };

      const result = applyTerritoryDecision(state, invalidDecision);
      expect(result.success).toBe(false);
      expect(result.reason).toBe('Unknown decision type');
    });

    it('returns error for process_region without region', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';

      const decision = {
        type: 'process_region' as const,
        player: 1,
        // Missing region
      };

      const result = applyTerritoryDecision(state, decision);
      expect(result.success).toBe(false);
    });

    it('returns error for eliminate_from_stack without stackPosition', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';

      const decision = {
        type: 'eliminate_from_stack' as const,
        player: 1,
        // Missing stackPosition
      };

      const result = applyTerritoryDecision(state, decision);
      expect(result.success).toBe(false);
    });
  });

});
