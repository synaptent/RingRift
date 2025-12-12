/**
 * Near-Victory Territory Scenarios Test Suite
 *
 * Tests for territory-based victory conditions using the near-victory
 * territory fixture. Verifies that:
 * - Territory processing can trigger victory when threshold is crossed
 * - Territory counting is accurate
 * - Victory reason is correctly identified as 'territory_control'
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import {
  createNearVictoryTerritoryFixture,
  createNearVictoryTerritoryFixtureMultiRegion,
  type NearVictoryTerritoryFixture,
} from '../fixtures/nearVictoryTerritoryFixture';
import { evaluateVictory } from '../../src/shared/engine';
import type { BoardType, GameState, Territory, Position } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';
import { computeProgressSnapshot } from '../../src/shared/engine/core';

const BOARD_CASES: Array<{ boardType: BoardType; totalSpaces: number; expectedThreshold: number }> =
  [
    { boardType: 'square8', totalSpaces: 64, expectedThreshold: 33 },
    { boardType: 'square19', totalSpaces: 361, expectedThreshold: 181 },
  ];

describe('Near-victory territory scenarios', () => {
  describe.each(BOARD_CASES)(
    'createNearVictoryTerritoryFixture ($boardType)',
    ({ boardType, expectedThreshold }) => {
      let fixture: NearVictoryTerritoryFixture;

      beforeEach(() => {
        fixture = createNearVictoryTerritoryFixture({ boardType });
      });

      it('should create a valid game state', () => {
        expect(fixture.gameState).toBeDefined();
        expect(fixture.gameState.board).toBeDefined();
        expect(fixture.gameState.players).toHaveLength(2);
        expect(fixture.gameState.boardType).toBe(boardType);
      });

      it('should set Player 1 territory spaces just below victory threshold', () => {
        const threshold = fixture.territoryVictoryThreshold;
        const initialSpaces = fixture.initialTerritorySpaces;

        expect(initialSpaces).toBeLessThan(threshold);
        expect(initialSpaces).toBe(threshold - 1);
        expect(threshold).toBe(expectedThreshold);
      });

      it('should have Player 1 collapsed territory matching initial territory spaces', () => {
        const collapsedCount = fixture.gameState.board.collapsedSpaces.size;
        expect(collapsedCount).toBe(fixture.initialTerritorySpaces);
      });

      it('should have a pending territory region', () => {
        expect(fixture.gameState.board.territories.size).toBeGreaterThan(0);

        const pendingDecision = (fixture.gameState as unknown as Record<string, unknown>)
          .pendingTerritoryDecision as { territories: string[]; currentIndex: number } | undefined;

        expect(pendingDecision).toBeDefined();
        expect(pendingDecision?.territories).toHaveLength(1);
      });

      it('should be in territory_processing phase', () => {
        expect(fixture.gameState.currentPhase).toBe('territory_processing');
        expect(fixture.gameState.currentPlayer).toBe(1);
      });

      it('should have a valid winning move defined', () => {
        expect(fixture.winningMove).toBeDefined();
        expect(fixture.winningMove.type).toBe('process_territory_region');
        expect(fixture.winningMove.player).toBe(1);
        expect(fixture.winningMove.disconnectedRegions).toBeDefined();
        expect(fixture.winningMove.disconnectedRegions).toHaveLength(1);
      });

      it('should set expected winner to 1 with territory victory type', () => {
        expect(fixture.expectedWinner).toBe(1);
        expect(fixture.victoryType).toBe('territory');
      });
    }
  );

  describe.each(BOARD_CASES)('Victory condition detection ($boardType)', (caseData) => {
    const { boardType, totalSpaces } = caseData;

    it('should not detect victory before region is processed', () => {
      const fixture = createNearVictoryTerritoryFixture({ boardType });
      const result = evaluateVictory(fixture.gameState);

      // Game should not be over yet – Player 1 is just below the territory threshold.
      expect(result.isGameOver).toBe(false);
    });

    it('should detect territory victory after region is processed', () => {
      const fixture = createNearVictoryTerritoryFixture({ boardType });

      // Simulate processing the territory region by:
      // 1. Adding the pending region spaces to collapsed territory
      // 2. Updating player territory count
      const state = fixture.gameState;
      const territories = state.board.territories;

      // Get the pending region
      const regionEntry = Array.from(territories.entries())[0];
      if (!regionEntry) {
        throw new Error('No pending territory found');
      }

      const [, region] = regionEntry;

      // Capture S-invariant before processing for numeric cross-check.
      const beforeSnapshot = computeProgressSnapshot(state);
      const SBefore = beforeSnapshot.S;

      // Apply territory processing effects (region collapse only; this
      // fixture does not model self-elimination).
      for (const space of region.spaces) {
        state.board.collapsedSpaces.set(positionToString(space), 1);
      }

      // Update Player 1's territory count
      state.players[0].territorySpaces += region.spaces.length;

      // Now verify victory is detected
      const result = evaluateVictory(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('territory_control');

      // S-invariant delta should equal the number of newly collapsed
      // spaces in this curated near-victory scenario.
      const afterSnapshot = computeProgressSnapshot(state);
      const SAfter = afterSnapshot.S;
      const deltaS = SAfter - SBefore;
      expect(deltaS).toBe(region.spaces.length);
    });

    it('should correctly calculate territory percentages', () => {
      const fixture = createNearVictoryTerritoryFixture({ boardType });
      const state = fixture.gameState;
      const region =
        fixture.winningMove.disconnectedRegions && fixture.winningMove.disconnectedRegions[0];
      const pendingRegionSize = region ? region.spaces.length : 1;

      // Initial state should be at or just below 50% (territoryVictoryThreshold - 1)
      const player1Initial = state.players[0].territorySpaces;
      const initialPercentage = (player1Initial / totalSpaces) * 100;

      // Victory requires > 50%, so at or below 50% is not a victory yet
      expect(initialPercentage).toBeLessThanOrEqual(50);

      // After processing, percentage should exceed 50%
      const afterProcessing = player1Initial + pendingRegionSize;
      const finalPercentage = (afterProcessing / totalSpaces) * 100;

      expect(finalPercentage).toBeGreaterThan(50);
    });
  });

  describe('Multi-region near-victory fixture', () => {
    it('should create a fixture with larger pending region', () => {
      const fixture = createNearVictoryTerritoryFixtureMultiRegion();

      // Should have pending region with multiple spaces
      const territories = fixture.gameState.board.territories;
      expect(territories.size).toBeGreaterThan(0);

      const region = Array.from(territories.values())[0];
      expect(region).toBeDefined();
      expect(region!.spaces.length).toBeGreaterThan(1);
    });

    it('should still trigger victory when multi-cell region is processed', () => {
      const fixture = createNearVictoryTerritoryFixtureMultiRegion();
      const state = fixture.gameState;

      // Get the pending region
      const region = Array.from(state.board.territories.values())[0];
      if (!region) {
        throw new Error('No pending territory found');
      }

      // Apply territory processing effects
      for (const space of region.spaces) {
        state.board.collapsedSpaces.set(positionToString(space), 1);
      }

      // Update Player 1's territory count
      state.players[0].territorySpaces += region.spaces.length;

      // Verify victory is detected
      const result = evaluateVictory(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('territory_control');
    });
  });

  describe('Fixture serialization', () => {
    it('should serialize fixture to JSON-compatible format', async () => {
      // Dynamically import to avoid issues if module not built yet
      const { serializeNearVictoryTerritoryFixture } =
        await import('../fixtures/nearVictoryTerritoryFixture');

      const fixture = createNearVictoryTerritoryFixture();
      const serialized = serializeNearVictoryTerritoryFixture(fixture);

      expect(serialized.gameState).toBeDefined();
      expect(serialized.winningMove).toBeDefined();

      // Verify it can be stringified (valid JSON)
      const jsonString = JSON.stringify(serialized);
      expect(typeof jsonString).toBe('string');

      // Verify it can be parsed back
      const parsed = JSON.parse(jsonString);
      expect(parsed.gameState.currentPhase).toBe('territory_processing');
      expect(parsed.winningMove.type).toBe('process_territory_region');
    });
  });

  describe('Edge cases', () => {
    it('should handle fixture with exact threshold match', () => {
      // Create fixture where Player 1 is exactly at threshold after processing
      const fixture = createNearVictoryTerritoryFixture({
        spacesbelowThreshold: 1,
        pendingRegionSize: 1,
      });

      const state = fixture.gameState;
      const threshold = state.territoryVictoryThreshold;

      // Initial: threshold - 1
      expect(state.players[0].territorySpaces).toBe(threshold - 1);

      // After adding 1 space: exactly at threshold
      const finalCount = state.players[0].territorySpaces + 1;
      expect(finalCount).toBe(threshold);

      // Victory should trigger at exactly threshold (>= threshold)
      state.players[0].territorySpaces = finalCount;

      const result = evaluateVictory(state);
      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
    });

    it('should have Player 2 with no territory', () => {
      const fixture = createNearVictoryTerritoryFixture();

      expect(fixture.gameState.players[1].territorySpaces).toBe(0);
    });

    it('should maintain board validity with stacks for both players', () => {
      const fixture = createNearVictoryTerritoryFixture();

      const stacks = fixture.gameState.board.stacks;
      expect(stacks.size).toBeGreaterThanOrEqual(2);

      // Find stacks by player
      const player1Stacks = Array.from(stacks.values()).filter((s) => s.controllingPlayer === 1);
      const player2Stacks = Array.from(stacks.values()).filter((s) => s.controllingPlayer === 2);

      expect(player1Stacks.length).toBeGreaterThan(0);
      expect(player2Stacks.length).toBeGreaterThan(0);
    });
  });
});
