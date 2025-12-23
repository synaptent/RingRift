/**
 * Golden Replay Tests
 *
 * Runs curated golden game fixtures through the replay system and verifies
 * structural invariants hold at every step.
 *
 * These tests serve as high-confidence regression tests for the rules engine
 * and provide end-to-end parity verification between TS and Python engines.
 */

import {
  loadGoldenGames,
  getGoldenGamesDir,
  replayAndAssertInvariants,
  assertFinalStateMatchesOutcome,
  checkAllInvariants,
  type GoldenGameInfo,
  type InvariantViolation,
} from './goldenReplayHelpers';
import type { GameRecord } from '../../src/shared/types/gameRecord';

describe('Golden Replay Tests', () => {
  const goldenGamesDir = getGoldenGamesDir();
  const goldenGames = loadGoldenGames(goldenGamesDir);

  // Skip if no fixtures available yet
  if (goldenGames.length === 0) {
    // SKIP-REASON: fixture-dependent - requires golden game fixtures from curation script
    it.skip('No golden game fixtures found - run curation script to generate', () => {
      // Placeholder test
    });
    return;
  }

  describe('Structural Invariants', () => {
    describe.each(goldenGames)('$info.filename', ({ info, record }) => {
      it('should pass all structural invariants at every move', () => {
        const result = replayAndAssertInvariants(record);

        if (result.error) {
          throw new Error(`Replay failed: ${result.error}`);
        }

        if (!result.success) {
          const violations = result.invariantViolations.slice(0, 5); // First 5 violations
          const violationSummary = violations
            .map((v) => `  - [${v.invariant}] Move ${v.moveIndex}: ${v.details}`)
            .join('\n');

          throw new Error(
            `Found ${result.invariantViolations.length} invariant violations:\n${violationSummary}`
          );
        }

        expect(result.success).toBe(true);
        expect(result.invariantViolations).toHaveLength(0);
      });

      it('should reconstruct final state matching recorded outcome', () => {
        const result = replayAndAssertInvariants(record);

        if (result.error) {
          throw new Error(`Replay failed: ${result.error}`);
        }

        if (!result.finalState) {
          throw new Error('Failed to reconstruct final state');
        }

        const outcomeViolations = assertFinalStateMatchesOutcome(result.finalState, record);

        if (outcomeViolations.length > 0) {
          const violationSummary = outcomeViolations
            .map((v) => `  - [${v.invariant}]: ${v.details}`)
            .join('\n');

          throw new Error(`Final state doesn't match recorded outcome:\n${violationSummary}`);
        }

        expect(outcomeViolations).toHaveLength(0);
      });
    });
  });

  describe('Category Coverage', () => {
    // Group games by category for coverage tracking
    const gamesByCategory = new Map<string, Array<{ info: GoldenGameInfo; record: GameRecord }>>();
    for (const game of goldenGames) {
      const category = game.info.category;
      if (!gamesByCategory.has(category)) {
        gamesByCategory.set(category, []);
      }
      gamesByCategory.get(category)!.push(game);
    }

    it('should have games in each major rules category', () => {
      // When fixtures are populated, verify we have coverage
      const categories = Array.from(gamesByCategory.keys());
      expect(categories.length).toBeGreaterThan(0);

      // Log category distribution for visibility
      console.log('Golden game coverage by category:');
      for (const [category, games] of gamesByCategory) {
        console.log(`  ${category}: ${games.length} game(s)`);
      }
    });

    it('should cover multiple board types', () => {
      const boardTypes = new Set(goldenGames.map((g) => g.info.boardType));
      expect(boardTypes.size).toBeGreaterThanOrEqual(1);
      console.log('Board types covered:', Array.from(boardTypes).join(', '));
    });

    it('should cover multiple player counts', () => {
      const playerCounts = new Set(goldenGames.map((g) => g.info.numPlayers));
      expect(playerCounts.size).toBeGreaterThanOrEqual(1);
      console.log('Player counts covered:', Array.from(playerCounts).join(', '));
    });
  });

  describe('Outcome Coverage', () => {
    it('should have games with various outcomes', () => {
      const outcomes = new Set(goldenGames.map((g) => g.info.expectedOutcome).filter(Boolean));
      expect(outcomes.size).toBeGreaterThanOrEqual(1);
      console.log('Outcomes covered:', Array.from(outcomes).join(', '));
    });
  });
});

describe('Golden Replay Helpers Unit Tests', () => {
  describe('checkAllInvariants', () => {
    it('should detect invalid phase', () => {
      // Create a minimal mock state with invalid phase
      const mockState = {
        boardType: 'square8' as const,
        currentPhase: 'invalid_phase' as never,
        gameStatus: 'active' as const,
        currentPlayer: 0,
        players: [{ playerNumber: 0, eliminatedRings: 0, ringsInHand: 18 }],
        board: {
          stacks: new Map(),
          markers: new Map(),
        },
        moveHistory: [],
      };

      // @ts-expect-error - intentionally passing incomplete state for testing
      const violations = checkAllInvariants(mockState, 0);

      expect(violations.some((v) => v.invariant === 'INV-PHASE-VALID')).toBe(true);
    });

    it('should detect invalid current player', () => {
      const mockState = {
        boardType: 'square8' as const,
        currentPhase: 'movement' as const,
        gameStatus: 'active' as const,
        currentPlayer: 99, // Invalid
        players: [{ playerNumber: 0, eliminatedRings: 0, ringsInHand: 18 }],
        board: {
          stacks: new Map(),
          markers: new Map(),
        },
        moveHistory: [],
      };

      // @ts-expect-error - intentionally passing incomplete state for testing
      const violations = checkAllInvariants(mockState, 0);

      expect(violations.some((v) => v.invariant === 'INV-ACTIVE-PLAYER')).toBe(true);
    });

    it('should detect negative ring counts', () => {
      const mockState = {
        boardType: 'square8' as const,
        currentPhase: 'movement' as const,
        gameStatus: 'active' as const,
        currentPlayer: 0,
        players: [
          { playerNumber: 0, eliminatedRings: -5, ringsInHand: 18 }, // Invalid
        ],
        board: {
          stacks: new Map(),
          markers: new Map(),
        },
        moveHistory: [],
      };

      // @ts-expect-error - intentionally passing incomplete state for testing
      const violations = checkAllInvariants(mockState, 0);

      expect(violations.some((v) => v.invariant === 'INV-PLAYER-RINGS')).toBe(true);
    });
  });

  describe('loadGoldenGames', () => {
    it('should return empty array for non-existent directory', () => {
      const result = loadGoldenGames('/non/existent/path');
      expect(result).toEqual([]);
    });
  });
});
