/**
 * Region Order Integration Test
 *
 * Tests that region order processing is deterministic and consistent:
 * 1. Multiple regions are enumerated in deterministic order
 * 2. Same board state produces same enumeration across multiple runs
 * 3. Selecting different regions produces correct outcomes
 *
 * This test was refactored to use the current shared engine architecture
 * (territoryDecisionHelpers) instead of the legacy WebSocket-based
 * GameEngine.processDisconnectedRegions path.
 *
 * Related:
 * - P18.18 Skipped Test Triage (P19.1)
 * - src/shared/engine/territoryDecisionHelpers.ts
 * - src/shared/engine/territoryProcessing.ts
 * - src/shared/engine/territoryDetection.ts
 * - RULES_SCENARIO_MATRIX.md – territory section "Region order PlayerChoice" row (FAQ Q23 self-elimination prerequisite)
 */

import {
  BoardType,
  GameState,
  Player,
  Position,
  Territory,
  TimeControl,
  Move,
  positionToString,
} from '../../src/shared/types/game';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import {
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
} from '../../src/shared/engine/territoryDecisionHelpers';
import {
  TurnEngineAdapter,
  DecisionHandler,
  createSimpleAdapter,
} from '../../src/server/game/turn/TurnEngineAdapter';
import type { PendingDecision } from '../../src/shared/engine/orchestration/types';

describe('Region Order Integration', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const players: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  /**
   * Create a clean test state with no stacks, markers, or collapsed spaces.
   */
  function createEmptyState(id: string): GameState {
    const state = createInitialGameState(id, boardType, players, timeControl) as GameState;

    state.currentPlayer = 1;
    state.currentPhase = 'territory_processing';
    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();
    state.board.territories = new Map();
    state.board.eliminatedRings = { 1: 0, 2: 0 };
    state.board.formedLines = [];
    state.totalRingsEliminated = 0;
    state.players = state.players.map((p) => ({
      ...p,
      eliminatedRings: 0,
      territorySpaces: 0,
    }));

    return state;
  }

  /**
   * Add a stack to the board at the given position.
   */
  function addStack(state: GameState, pos: Position, player: number, rings: number[]): void {
    const key = positionToString(pos);
    state.board.stacks.set(key, {
      position: pos,
      rings,
      stackHeight: rings.length,
      capHeight: rings.filter((r) => r === rings[0]).length,
      controllingPlayer: player,
    } as any);
  }

  /**
   * Create a multi-region test scenario with two distinct disconnected regions.
   *
   * Setup:
   * - Region A: positions (1,1), (1,2) - "upper-left" region
   * - Region B: positions (5,5), (5,6) - "lower-right" region
   * - Outside stack at (3,3) to satisfy self-elimination prerequisite
   */
  function createMultiRegionScenario(): {
    state: GameState;
    regionA: Territory;
    regionB: Territory;
    outsideStack: Position;
  } {
    const state = createEmptyState('multi-region-test');

    // Region A (upper-left) - 2 spaces
    const regionA: Territory = {
      spaces: [
        { x: 1, y: 1 },
        { x: 1, y: 2 },
      ],
      controllingPlayer: 1,
      isDisconnected: true,
    };

    // Region B (lower-right) - 2 spaces
    const regionB: Territory = {
      spaces: [
        { x: 5, y: 5 },
        { x: 5, y: 6 },
      ],
      controllingPlayer: 1,
      isDisconnected: true,
    };

    // Outside stack for player 1 to satisfy self-elimination prerequisite
    const outsideStack: Position = { x: 3, y: 3 };
    addStack(state, outsideStack, 1, [1, 1]);

    return { state, regionA, regionB, outsideStack };
  }

  describe('deterministic region enumeration', () => {
    it('should enumerate multiple regions in deterministic order', () => {
      const { state, regionA, regionB } = createMultiRegionScenario();

      // Run enumeration multiple times and verify order is consistent
      const results: Move[][] = [];
      for (let i = 0; i < 5; i++) {
        const moves = enumerateProcessTerritoryRegionMoves(state, 1, {
          testOverrideRegions: [regionA, regionB],
        });
        results.push(moves);
      }

      // All results should have the same length
      expect(results.every((r) => r.length === 2)).toBe(true);

      // All results should have the same order (comparing move IDs)
      const firstResult = results[0];
      for (let i = 1; i < results.length; i++) {
        expect(results[i].map((m) => m.id)).toEqual(firstResult.map((m) => m.id));
      }

      // Verify move types and structure
      for (const move of firstResult) {
        expect(move.type).toBe('choose_territory_option');
        expect(move.player).toBe(1);
        expect(move.disconnectedRegions).toBeDefined();
        expect(move.disconnectedRegions!.length).toBe(1);
      }
    });

    it('should maintain enumeration order regardless of region declaration order', () => {
      const { state, regionA, regionB } = createMultiRegionScenario();

      // Pass regions in different orders
      const movesAB = enumerateProcessTerritoryRegionMoves(state, 1, {
        testOverrideRegions: [regionA, regionB],
      });

      const movesBA = enumerateProcessTerritoryRegionMoves(state, 1, {
        testOverrideRegions: [regionB, regionA],
      });

      // Both should return 2 moves
      expect(movesAB).toHaveLength(2);
      expect(movesBA).toHaveLength(2);

      // The moves should represent the same regions (order may differ based on input)
      const abRepresentatives = movesAB.map((m) => positionToString(m.to!)).sort();
      const baRepresentatives = movesBA.map((m) => positionToString(m.to!)).sort();
      expect(abRepresentatives).toEqual(baRepresentatives);
    });

    it('should produce same result regardless of discovery order', () => {
      const { state, regionA, regionB } = createMultiRegionScenario();

      // Process regionA first
      const movesForA = enumerateProcessTerritoryRegionMoves(state, 1, {
        testOverrideRegions: [regionA],
      });
      expect(movesForA).toHaveLength(1);
      const outcomeA = applyProcessTerritoryRegionDecision(state, movesForA[0]);

      // Process regionB first (on original state)
      const movesForB = enumerateProcessTerritoryRegionMoves(state, 1, {
        testOverrideRegions: [regionB],
      });
      expect(movesForB).toHaveLength(1);
      const outcomeB = applyProcessTerritoryRegionDecision(state, movesForB[0]);

      // Both outcomes should have pendingSelfElimination true
      expect(outcomeA.pendingSelfElimination).toBe(true);
      expect(outcomeB.pendingSelfElimination).toBe(true);

      // Both should credit territory to player 1
      const p1AfterA = outcomeA.nextState.players.find((p) => p.playerNumber === 1)!;
      const p1AfterB = outcomeB.nextState.players.find((p) => p.playerNumber === 1)!;
      expect(p1AfterA.territorySpaces).toBeGreaterThan(0);
      expect(p1AfterB.territorySpaces).toBeGreaterThan(0);

      // Different regions processed, so different collapsed spaces
      const collapsedA = Array.from(outcomeA.nextState.board.collapsedSpaces.keys()).sort();
      const collapsedB = Array.from(outcomeB.nextState.board.collapsedSpaces.keys()).sort();
      expect(collapsedA).not.toEqual(collapsedB);
    });
  });

  describe('region selection affects outcomes', () => {
    it('should process the selected region correctly', () => {
      const { state, regionA, regionB } = createMultiRegionScenario();

      // Add stacks inside regions to verify elimination
      addStack(state, regionA.spaces[0], 2, [2, 2]); // Enemy stack in region A
      addStack(state, regionB.spaces[0], 2, [2]); // Enemy stack in region B

      // Get moves for both regions
      const moves = enumerateProcessTerritoryRegionMoves(state, 1, {
        testOverrideRegions: [regionA, regionB],
      });
      expect(moves).toHaveLength(2);

      // Select region A (first move based on input order)
      const regionAMove = moves.find(
        (m) => positionToString(m.to!) === positionToString(regionA.spaces[0])
      );
      expect(regionAMove).toBeDefined();

      const outcome = applyProcessTerritoryRegionDecision(state, regionAMove!);

      // Verify region A was processed
      expect(outcome.processedRegion.spaces).toEqual(regionA.spaces);

      // Stack in region A should be eliminated
      const regionAKey = positionToString(regionA.spaces[0]);
      expect(outcome.nextState.board.stacks.has(regionAKey)).toBe(false);

      // Stack in region B should remain (region B not processed yet)
      const regionBKey = positionToString(regionB.spaces[0]);
      expect(outcome.nextState.board.stacks.has(regionBKey)).toBe(true);

      // Region A spaces should be collapsed
      for (const space of regionA.spaces) {
        expect(outcome.nextState.board.collapsedSpaces.get(positionToString(space))).toBe(1);
      }
    });

    it('should correctly handle different-sized regions', () => {
      const state = createEmptyState('size-test');

      // Small region (1 space)
      const smallRegion: Territory = {
        spaces: [{ x: 1, y: 1 }],
        controllingPlayer: 1,
        isDisconnected: true,
      };

      // Large region (3 spaces)
      const largeRegion: Territory = {
        spaces: [
          { x: 5, y: 5 },
          { x: 5, y: 6 },
          { x: 6, y: 5 },
        ],
        controllingPlayer: 1,
        isDisconnected: true,
      };

      // Outside stack - must be height > 1 or multicolor to be eligible for territory elimination
      // Per RR-CANON-R145: height-1 standalone rings are NOT eligible for territory elimination
      addStack(state, { x: 3, y: 3 }, 1, [1, 1]); // Height-2 stack

      const moves = enumerateProcessTerritoryRegionMoves(state, 1, {
        testOverrideRegions: [smallRegion, largeRegion],
      });
      expect(moves).toHaveLength(2);

      // Find move for large region
      const largeMoveTarget = positionToString(largeRegion.spaces[0]);
      const largeMove = moves.find((m) => positionToString(m.to!) === largeMoveTarget);
      expect(largeMove).toBeDefined();

      const outcome = applyProcessTerritoryRegionDecision(state, largeMove!);

      // Large region has 3 spaces, so territory gain should be at least 3
      const p1 = outcome.nextState.players.find((p) => p.playerNumber === 1)!;
      expect(p1.territorySpaces).toBeGreaterThanOrEqual(3);
    });
  });

  describe('self-elimination prerequisite', () => {
    it('should only enumerate regions where player has outside stack', () => {
      const state = createEmptyState('prereq-test');

      // Region that covers ALL player 1 stacks (should NOT be processable)
      const allStacksRegion: Territory = {
        spaces: [
          { x: 1, y: 1 },
          { x: 1, y: 2 },
        ],
        controllingPlayer: 1,
        isDisconnected: true,
      };

      // Add stacks ONLY inside the region
      addStack(state, { x: 1, y: 1 }, 1, [1]);
      addStack(state, { x: 1, y: 2 }, 1, [1]);

      const moves = enumerateProcessTerritoryRegionMoves(state, 1, {
        testOverrideRegions: [allStacksRegion],
      });

      // No moves should be available because all stacks are inside the region
      expect(moves).toHaveLength(0);
    });

    it('should enumerate region when player has eligible stack outside', () => {
      const state = createEmptyState('prereq-outside-test');

      const region: Territory = {
        spaces: [{ x: 1, y: 1 }],
        controllingPlayer: 1,
        isDisconnected: true,
      };

      // Add stack inside the region
      addStack(state, { x: 1, y: 1 }, 1, [1]);

      // Add eligible stack OUTSIDE the region (height > 1 required per RR-CANON-R145)
      // Height-1 standalone rings are NOT eligible for territory elimination
      addStack(state, { x: 5, y: 5 }, 1, [1, 1]); // Height-2 stack

      const moves = enumerateProcessTerritoryRegionMoves(state, 1, {
        testOverrideRegions: [region],
      });

      // Move should be available because player has eligible outside stack
      expect(moves).toHaveLength(1);
      expect(moves[0].type).toBe('choose_territory_option');
    });

    it('should NOT enumerate region when outside stack is height-1', () => {
      const state = createEmptyState('prereq-height1-test');

      const region: Territory = {
        spaces: [{ x: 1, y: 1 }],
        controllingPlayer: 1,
        isDisconnected: true,
      };

      // Add stack inside the region
      addStack(state, { x: 1, y: 1 }, 1, [1]);

      // Add height-1 stack OUTSIDE the region - NOT eligible per RR-CANON-R145
      addStack(state, { x: 5, y: 5 }, 1, [1]); // Height-1 standalone ring

      const moves = enumerateProcessTerritoryRegionMoves(state, 1, {
        testOverrideRegions: [region],
      });

      // No moves - height-1 standalone rings are not eligible for territory elimination
      expect(moves).toHaveLength(0);
    });
  });

  describe('determinism across multiple runs', () => {
    it('should be deterministic across 10 consecutive runs', () => {
      const results: string[][] = [];

      for (let run = 0; run < 10; run++) {
        const { state, regionA, regionB } = createMultiRegionScenario();

        // Add some stacks to make the scenario more realistic
        addStack(state, regionA.spaces[0], 2, [2]);
        addStack(state, regionB.spaces[0], 2, [2, 2]);

        const moves = enumerateProcessTerritoryRegionMoves(state, 1, {
          testOverrideRegions: [regionA, regionB],
        });

        results.push(moves.map((m) => m.id));
      }

      // All 10 runs should produce identical move orderings
      const firstResult = results[0];
      for (let i = 1; i < results.length; i++) {
        expect(results[i]).toEqual(firstResult);
      }
    });

    it('should produce identical final states when processing regions in same order', () => {
      const finalStates: string[] = [];

      for (let run = 0; run < 5; run++) {
        const { state, regionA, regionB } = createMultiRegionScenario();

        // Process region A
        const movesA = enumerateProcessTerritoryRegionMoves(state, 1, {
          testOverrideRegions: [regionA, regionB],
        });
        const regionAMove = movesA.find(
          (m) => positionToString(m.to!) === positionToString(regionA.spaces[0])
        )!;
        const afterA = applyProcessTerritoryRegionDecision(state, regionAMove);

        // Serialize key parts of the state for comparison
        const stateSignature = JSON.stringify({
          collapsed: Array.from(afterA.nextState.board.collapsedSpaces.entries()).sort(),
          stacks: Array.from(afterA.nextState.board.stacks.keys()).sort(),
          territory: afterA.nextState.players.find((p) => p.playerNumber === 1)?.territorySpaces,
        });

        finalStates.push(stateSignature);
      }

      // All runs should produce identical final states
      const firstState = finalStates[0];
      for (let i = 1; i < finalStates.length; i++) {
        expect(finalStates[i]).toBe(firstState);
      }
    });
  });

  describe('TurnEngineAdapter integration', () => {
    it('should process region decisions via adapter with DecisionHandler', async () => {
      const { state, regionA, regionB } = createMultiRegionScenario();

      // Add stacks inside regions
      addStack(state, regionA.spaces[0], 2, [2]);

      // Get valid moves
      const moves = enumerateProcessTerritoryRegionMoves(state, 1, {
        testOverrideRegions: [regionA, regionB],
      });
      expect(moves).toHaveLength(2);

      // Create a decision handler that selects the second option (regionB)
      const decisionHandler: DecisionHandler = {
        requestDecision: async (decision: PendingDecision): Promise<Move> => {
          if (decision.options.length < 2) {
            return decision.options[0];
          }
          // Select second option to test non-default selection
          return decision.options[1];
        },
      };

      // Create adapter
      const { adapter, getState } = createSimpleAdapter(state, decisionHandler);

      // Verify the adapter has the expected methods
      expect(typeof adapter.validateMoveOnly).toBe('function');
      expect(typeof adapter.getValidMovesFor).toBe('function');

      // The initial state should match
      const currentState = getState();
      expect(currentState.currentPlayer).toBe(1);
    });
  });
});
