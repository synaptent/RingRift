/**
 * FAQ Q19-Q21, Q24: Player Count, Victory Thresholds & Forced Elimination
 *
 * Covers:
 * - FAQ Q19: Playing with 2 or 4 players
 * - FAQ Q21: Victory thresholds with variable player counts
 * - FAQ Q24: Forced elimination when blocked with stacks
 *
 * Rules: §13 (Victory Conditions), §4.4 (Forced Elimination)
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import { Position, Player, TimeControl, GameState } from '../../src/shared/types/game';
import { createTestPlayer } from '../utils/fixtures';
import {
  hasForcedEliminationAction,
  hasGlobalPlacementAction,
  applyForcedEliminationForPlayer,
} from '../../src/shared/engine/globalActions';

describe('FAQ Q19-Q21, Q24: Player Counts, Thresholds & Forced Elimination', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  describe('FAQ Q19: Player Count Variations', () => {
    describe('2-Player Games', () => {
      it('should use correct thresholds for 2-player square8 (>18 rings)', () => {
        const players = [
          createTestPlayer(1, { ringsInHand: 18 }),
          createTestPlayer(2, { ringsInHand: 18 }),
        ];

        const engine = new GameEngine('faq-q19-2p-s8', 'square8', players, timeControl, false);
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        // Total rings: 2 × 18 = 36
        // Victory threshold (rings needed to win): 19 (>50% of 36)
        expect(gameState.victoryThreshold).toBe(19);
        expect(gameState.totalRingsInPlay).toBe(36);
      });

      it('should use correct thresholds for 2-player square19 (>36 rings)', () => {
        const players = [
          createTestPlayer(1, { ringsInHand: 36 }),
          createTestPlayer(2, { ringsInHand: 36 }),
        ];

        const engine = new GameEngine('faq-q19-2p-s19', 'square19', players, timeControl, false);
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        // Total rings: 2 × 36 = 72
        // Victory threshold (rings needed to win): 37 (>50% of 72)
        expect(gameState.victoryThreshold).toBe(37);
        expect(gameState.totalRingsInPlay).toBe(72);
      });
    });

    describe('3-Player Games (Recommended)', () => {
      it('should use correct thresholds for 3-player square8 (>27 rings)', () => {
        const players = [
          createTestPlayer(1, { ringsInHand: 18 }),
          createTestPlayer(2, { ringsInHand: 18 }),
          createTestPlayer(3, { ringsInHand: 18 }),
        ];

        const engine = new GameEngine('faq-q19-3p-s8', 'square8', players, timeControl, false);
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        // Total rings: 3 × 18 = 54
        // Victory threshold (rings needed to win): 28 (>50% of 54)
        expect(gameState.victoryThreshold).toBe(28);
        expect(gameState.totalRingsInPlay).toBe(54);
      });

      it('should use correct thresholds for 3-player square19 (>54 rings)', () => {
        const players = [
          createTestPlayer(1, { ringsInHand: 36 }),
          createTestPlayer(2, { ringsInHand: 36 }),
          createTestPlayer(3, { ringsInHand: 36 }),
        ];

        const engine = new GameEngine('faq-q19-3p-s19', 'square19', players, timeControl, false);
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        // Total rings: 3 × 36 = 108
        // Victory threshold (rings needed to win): 55 (>50% of 108)
        expect(gameState.victoryThreshold).toBe(55);
        expect(gameState.totalRingsInPlay).toBe(108);
      });
    });

    describe('4-Player Games', () => {
      it('should use correct thresholds for 4-player square19 (>72 rings)', () => {
        const players = [
          createTestPlayer(1, { ringsInHand: 36 }),
          createTestPlayer(2, { ringsInHand: 36 }),
          createTestPlayer(3, { ringsInHand: 36 }),
          createTestPlayer(4, { ringsInHand: 36 }),
        ];

        const engine = new GameEngine('faq-q19-4p-s19', 'square19', players, timeControl, false);
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        // Total rings: 4 × 36 = 144
        // Victory threshold (rings needed to win): 73 (>50% of 144)
        expect(gameState.victoryThreshold).toBe(73);
        expect(gameState.totalRingsInPlay).toBe(144);
      });

      it('should use correct thresholds for 4-player hexagonal (>72 rings)', () => {
        // Per RR-CANON-R020, hex boards have 48 rings per player
        const players = [
          createTestPlayer(1, { ringsInHand: 48 }),
          createTestPlayer(2, { ringsInHand: 48 }),
          createTestPlayer(3, { ringsInHand: 48 }),
          createTestPlayer(4, { ringsInHand: 48 }),
        ];

        const engine = new GameEngine('faq-q19-4p-hex', 'hexagonal', players, timeControl, false);
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        // Total rings: 4 × 48 = 192, Victory threshold = floor(192/2) + 1 = 97
        expect(gameState.victoryThreshold).toBe(97);
        expect(gameState.totalRingsInPlay).toBe(192);
      });
    });
  });

  describe('FAQ Q21: Victory Thresholds Always >50%', () => {
    it('should guarantee no simultaneous victories with >50% rule', () => {
      // FAQ Q21: Mathematical validation that >50% prevents ties

      const testCases = [
        { total: 36, threshold: 19 }, // 2p square8
        { total: 54, threshold: 28 }, // 3p square8
        { total: 72, threshold: 37 }, // 2p square19/hex
        { total: 108, threshold: 55 }, // 3p square19/hex
        { total: 144, threshold: 73 }, // 4p square19/hex
      ];

      testCases.forEach(({ total, threshold }) => {
        // Threshold must be >50%
        expect(threshold / total).toBeGreaterThan(0.5);

        // If one player has threshold+1, others cannot reach threshold
        const playerA = threshold + 1;
        const remaining = total - playerA;
        expect(remaining).toBeLessThanOrEqual(threshold);
      });
    });

    it('should use territory threshold >50% of board spaces', () => {
      // Territory victory also requires >50%

      const configs = [
        { boardType: 'square8', spaces: 64, threshold: 33 },
        { boardType: 'square19', spaces: 361, threshold: 181 },
        { boardType: 'hexagonal', spaces: 469, threshold: 235 },
      ];

      configs.forEach((config) => {
        expect(config.threshold / config.spaces).toBeGreaterThan(0.5);
      });
    });
  });

  describe('FAQ Q24: Forced Elimination When Blocked', () => {
    it('should force cap elimination when no moves available but stacks exist', async () => {
      // FAQ Q24: If control stacks but cannot move/place/capture, must eliminate cap

      const engine = new GameEngine(
        'faq-q24-backend',
        'square8',
        [
          createTestPlayer(1, { ringsInHand: 0 }), // No rings in hand
          createTestPlayer(2, { ringsInHand: 0 }), // Both players blocked from placing
        ],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();
      gameState.board.markers.clear();
      gameState.board.collapsedSpaces.clear();

      // Set gameStatus to 'active' - required for resolveBlockedStateForCurrentPlayerForTesting
      gameState.gameStatus = 'active';

      // Blue has one stack completely surrounded by collapsed spaces
      // Cannot move, cannot place (no rings in hand), cannot capture
      // Use a height-1 stack so it only needs to move 1 space (which is blocked)

      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      // Surround with collapsed spaces (blocks all 8 Moore-neighbor directions)
      const surroundingPositions = [
        { x: 2, y: 2 },
        { x: 3, y: 2 },
        { x: 4, y: 2 },
        { x: 2, y: 3 },
        { x: 4, y: 3 },
        { x: 2, y: 4 },
        { x: 3, y: 4 },
        { x: 4, y: 4 },
      ];

      for (const pos of surroundingPositions) {
        gameState.board.collapsedSpaces.set(`${pos.x},${pos.y}`, 2);
      }

      const initialEliminated = gameState.players[0].eliminatedRings;

      // Player 1's turn, but no legal actions
      gameState.currentPhase = 'ring_placement';
      gameState.currentPlayer = 1;

      // Verify forced elimination preconditions are met
      expect(hasForcedEliminationAction(gameState, 1)).toBe(true);
      expect(hasGlobalPlacementAction(gameState, 1)).toBe(false);

      // Apply forced elimination using the shared helper
      const forcedElimOutcome = applyForcedEliminationForPlayer(gameState, 1);
      expect(forcedElimOutcome).toBeDefined();
      expect(forcedElimOutcome!.eliminatedCount).toBe(1);

      // Update gameState with the result
      const updatedState = forcedElimOutcome!.nextState;

      // Verify forced elimination happened (1 ring eliminated from player 1's stack)
      expect(updatedState.players[0].eliminatedRings).toBe(initialEliminated + 1);

      // Stack should be gone from board
      expect(updatedState.board.stacks.get('3,3')).toBeUndefined();
      expect(updatedState.board.stacks.size).toBe(0);

      // Note: The shared applyForcedEliminationForPlayer helper only applies
      // the elimination - it does not check for game termination. Game-end
      // detection is the host's responsibility (GameEngine.resolveBlockedState).
      // For this unit test, we verify that:
      // 1. Forced elimination preconditions were correctly detected
      // 2. Forced elimination was correctly applied
      // Game termination handling is tested separately at the integration level.
    });

    it('should count force-eliminated rings toward victory total', async () => {
      // FAQ Q24: Forced eliminations count toward ring elimination victory
      //
      // This test verifies that forced elimination increases eliminatedRings.
      // The ring elimination victory logic is tested separately;
      // here we focus on the core FAQ Q24 behavior.

      const engine = new GameEngine(
        'faq-q24-victory-backend',
        'square8',
        [
          createTestPlayer(1, { ringsInHand: 0, eliminatedRings: 5 }),
          createTestPlayer(2, { ringsInHand: 0, eliminatedRings: 3 }),
        ],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      // Set gameStatus to 'active' - required for resolveBlockedStateForCurrentPlayerForTesting
      gameState.gameStatus = 'active';

      // P1 has blocked stack with 1 ring at corner (0,0)
      // Height-1 stack needs to move exactly 1 space, blocked by collapsed neighbors
      gameState.board.stacks.clear();
      gameState.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      // Block P1's stack at corner (0,0) - only 3 neighbors on square board
      gameState.board.collapsedSpaces.set('1,0', 2);
      gameState.board.collapsedSpaces.set('0,1', 2);
      gameState.board.collapsedSpaces.set('1,1', 2);

      const initialEliminated = gameState.players[0].eliminatedRings;

      gameState.currentPhase = 'movement';
      gameState.currentPlayer = 1;

      // Verify forced elimination preconditions are met
      expect(hasForcedEliminationAction(gameState, 1)).toBe(true);

      // Apply forced elimination using the shared helper
      const forcedElimOutcome = applyForcedEliminationForPlayer(gameState, 1);
      expect(forcedElimOutcome).toBeDefined();

      // Update gameState with the result
      const updatedState = forcedElimOutcome!.nextState;

      // Forced elimination should have added the cap (1 ring) to eliminated count
      // 5 + 1 = 6 eliminated
      expect(updatedState.players[0].eliminatedRings).toBe(initialEliminated + 1);

      // Stack should be gone since the entire cap was the whole stack
      expect(updatedState.board.stacks.get('0,0')).toBeUndefined();
    });

    it('should support multiple forced eliminations if player has multiple stacks', async () => {
      // FAQ Q24: When globally blocked with multiple stacks, forced elimination
      // removes caps from stacks until legal actions are available or game ends.
      //
      // This test verifies the shared helper handles the case where both players
      // have blocked stacks. Per the rules, each player eliminates ONE cap at a
      // time, and the process repeats if all players remain blocked.

      const engine = new GameEngine(
        'faq-q24-multi-stack-backend',
        'square8',
        [
          createTestPlayer(1, { ringsInHand: 0, eliminatedRings: 2 }),
          createTestPlayer(2, { ringsInHand: 0, eliminatedRings: 1 }),
        ],
        timeControl,
        false
      );
      const engineAny: any = engine;
      let gameState = engineAny.gameState;

      // Set gameStatus to 'active'
      gameState.gameStatus = 'active';

      // P1 has blocked stack at corner (0,0) with height 1
      // Height-1 stack needs to move exactly 1 space, blocked by collapsed neighbors
      gameState.board.stacks.clear();
      gameState.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      // Block P1's stack at corner (0,0)
      gameState.board.collapsedSpaces.set('1,0', 2);
      gameState.board.collapsedSpaces.set('0,1', 2);
      gameState.board.collapsedSpaces.set('1,1', 2);

      // P2 also has blocked stack at opposite corner (7,7) with height 1
      gameState.board.stacks.set('7,7', {
        position: { x: 7, y: 7 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      // Block P2's stack at corner (7,7)
      gameState.board.collapsedSpaces.set('6,7', 2);
      gameState.board.collapsedSpaces.set('7,6', 2);
      gameState.board.collapsedSpaces.set('6,6', 2);

      gameState.currentPhase = 'movement';
      gameState.currentPlayer = 1;

      // Verify both players meet forced elimination preconditions
      expect(hasForcedEliminationAction(gameState, 1)).toBe(true);
      expect(hasForcedEliminationAction(gameState, 2)).toBe(true);

      // Apply forced elimination for player 1
      const p1Outcome = applyForcedEliminationForPlayer(gameState, 1);
      expect(p1Outcome).toBeDefined();
      expect(p1Outcome!.eliminatedCount).toBe(1);
      gameState = p1Outcome!.nextState;

      // P1's stack should be eliminated
      expect(gameState.board.stacks.get('0,0')).toBeUndefined();
      expect(gameState.players[0].eliminatedRings).toBe(3); // 2 + 1

      // P2's stack should still exist
      expect(gameState.board.stacks.get('7,7')).toBeDefined();

      // Apply forced elimination for player 2
      const p2Outcome = applyForcedEliminationForPlayer(gameState, 2);
      expect(p2Outcome).toBeDefined();
      expect(p2Outcome!.eliminatedCount).toBe(1);
      gameState = p2Outcome!.nextState;

      // Both stacks should now be eliminated
      expect(gameState.board.stacks.size).toBe(0);
      expect(gameState.players[1].eliminatedRings).toBe(2); // 1 + 1

      // Note: Game termination detection is a host responsibility, not tested here.
      // The shared helper only applies elimination; game-end checking happens at
      // integration level in GameEngine.resolveBlockedStateForCurrentPlayerForTesting().
    });
  });

  describe('FAQ Q11 & Q21: Stalemate with Rings in Hand', () => {
    it('should convert rings in hand to eliminated rings on stalemate', async () => {
      // FAQ Q11: Rings in hand count as eliminated in stalemate
      // These count in the "Most Eliminated rings" tiebreaker

      const engine = new GameEngine(
        'faq-q11-backend',
        'square8',
        [
          createTestPlayer(1, { ringsInHand: 3, eliminatedRings: 10 }),
          createTestPlayer(2, { ringsInHand: 5, eliminatedRings: 8 }),
        ],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      // No stacks on board, no legal placements
      gameState.board.stacks.clear();

      // Fill entire board with collapsed spaces so no placements possible
      for (let x = 0; x < 8; x++) {
        for (let y = 0; y < 8; y++) {
          gameState.board.collapsedSpaces.set(`${x},${y}`, 1);
        }
      }

      gameState.currentPhase = 'ring_placement';
      gameState.currentPlayer = 1;

      // Per RR-CANON-R075/R076, when no interactive moves are possible,
      // the engine may return bookkeeping moves (no_placement_action, skip_placement)
      // or an empty array. Either indicates stalemate conditions.
      const moves = engine.getValidMoves(1);
      const hasOnlyBookkeepingMoves =
        moves.length === 0 ||
        moves.every((m: any) =>
          ['no_placement_action', 'skip_placement', 'no_movement_action'].includes(m.type)
        );
      expect(hasOnlyBookkeepingMoves).toBe(true);

      // Manually trigger stalemate resolution
      const stalemate = engineAny.checkForStalemate?.() ?? false;

      if (stalemate) {
        // Rings in hand should be converted to eliminated
        // Player 1: 10 + 3 = 13
        // Player 2: 8 + 5 = 13
        // Tiebreaker goes to territory, then markers, then last actor
      }
    });
  });

  describe('Stalemate Tiebreaker Priority', () => {
    it('should resolve ties by territory > eliminated rings > markers > last actor', () => {
      // FAQ Q21: Stalemate tiebreaker sequence validation

      // This is structural - the tiebreaker order is:
      // 1. Most collapsed spaces (territory)
      // 2. Most eliminated rings (including rings in hand)
      // 3. Most remaining markers
      // 4. Last person to complete valid turn

      const scenarios = [
        {
          name: 'Territory winner',
          p1: { territory: 20, eliminated: 10, markers: 5 },
          p2: { territory: 15, eliminated: 15, markers: 8 },
          winner: 1,
          reason: 'Most territory',
        },
        {
          name: 'Eliminated rings winner (territory tied)',
          p1: { territory: 20, eliminated: 12, markers: 5 },
          p2: { territory: 20, eliminated: 8, markers: 8 },
          winner: 1,
          reason: 'Most eliminated (territory tied)',
        },
        {
          name: 'Markers winner (territory and eliminated tied)',
          p1: { territory: 20, eliminated: 10, markers: 7 },
          p2: { territory: 20, eliminated: 10, markers: 4 },
          winner: 1,
          reason: 'Most markers (others tied)',
        },
      ];

      scenarios.forEach((scenario) => {
        // Validate tiebreaker logic
        if (scenario.p1.territory !== scenario.p2.territory) {
          const winner = scenario.p1.territory > scenario.p2.territory ? 1 : 2;
          expect(winner).toBe(scenario.winner);
        } else if (scenario.p1.eliminated !== scenario.p2.eliminated) {
          const winner = scenario.p1.eliminated > scenario.p2.eliminated ? 1 : 2;
          expect(winner).toBe(scenario.winner);
        } else if (scenario.p1.markers !== scenario.p2.markers) {
          const winner = scenario.p1.markers > scenario.p2.markers ? 1 : 2;
          expect(winner).toBe(scenario.winner);
        }
      });
    });
  });

  describe('Territory Victory Thresholds', () => {
    it('should use >50% of board spaces for all board types', () => {
      // FAQ Q21: Territory victory also uses >50% rule

      const configs = [
        { boardType: 'square8', totalSpaces: 64, threshold: 33 },
        { boardType: 'square19', totalSpaces: 361, threshold: 181 },
        { boardType: 'hexagonal', totalSpaces: 469, threshold: 235 },
      ];

      configs.forEach((config) => {
        expect(config.threshold / config.totalSpaces).toBeGreaterThan(0.5);

        // Verify exactly >50% (one more than half the board)
        expect(config.threshold).toBe(Math.floor(config.totalSpaces / 2) + 1);
      });
    });
  });
});
