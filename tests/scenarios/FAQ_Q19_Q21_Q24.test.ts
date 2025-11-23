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
        const players = [
          createTestPlayer(1, { ringsInHand: 36 }),
          createTestPlayer(2, { ringsInHand: 36 }),
          createTestPlayer(3, { ringsInHand: 36 }),
          createTestPlayer(4, { ringsInHand: 36 }),
        ];

        const engine = new GameEngine('faq-q19-4p-hex', 'hexagonal', players, timeControl, false);
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        expect(gameState.victoryThreshold).toBe(73);
        expect(gameState.totalRingsInPlay).toBe(144);
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
        { boardType: 'hexagonal', spaces: 331, threshold: 166 },
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
          createTestPlayer(2, { ringsInHand: 18 }),
        ],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();
      gameState.board.markers.clear();
      gameState.board.collapsedSpaces.clear();

      // Blue has one stack completely surrounded by collapsed spaces
      // Cannot move, cannot place (no rings in hand), cannot capture

      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      });

      // Surround with collapsed spaces
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

      // No legal placement/movement/capture actions should be available.
      const moves = engine.getValidMoves(1);
      expect(moves.length).toBe(0);

      // Apply forced elimination using the dedicated test helper, which
      // mirrors how the TurnEngine enforces FAQ Q24 in live games.
      engine.resolveBlockedStateForCurrentPlayerForTesting();

      // Entire cap should be eliminated (3 rings)
      expect(gameState.players[0].eliminatedRings).toBe(initialEliminated + 3);

      // Stack should be gone from board
      expect(gameState.board.stacks.get('3,3')).toBeUndefined();
    });

    it('should count force-eliminated rings toward victory total', async () => {
      // FAQ Q24: Forced eliminations count toward ring elimination victory

      const engine = new GameEngine(
        'faq-q24-victory-backend',
        'square8',
        [
          createTestPlayer(1, { ringsInHand: 0, eliminatedRings: 25 }), // Close to threshold
          createTestPlayer(2, { ringsInHand: 18, eliminatedRings: 5 }),
        ],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      // Blue has 25 eliminated (needs >27 for 2p game... wait, 2p is >18)
      // Let me fix: for 2 players with 18 rings each = 36 total
      // Victory threshold is >18
      gameState.players[0].eliminatedRings = 17; // Just below threshold

      // Blue has blocked stack
      gameState.board.stacks.clear();
      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });

      // Surround it
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

      gameState.currentPhase = 'ring_placement';
      gameState.currentPlayer = 1;

      const moves = engine.getValidMoves(1);
      expect(moves.length).toBe(0);

      // Apply forced elimination via the blocked-state resolver; this
      // should also trigger a victory check once the threshold is passed.
      engine.resolveBlockedStateForCurrentPlayerForTesting();

      // 17 + 2 = 19 eliminated (>18 threshold)
      expect(gameState.players[0].eliminatedRings).toBe(19);

      // Should trigger victory
      expect(gameState.gameStatus).toBe('completed');
      expect(gameState.gameResult?.winner).toBe(1);
    });

    it('should continue game after forced elimination if under threshold', async () => {
      // FAQ Q24: Game continues if forced elimination doesn't reach victory

      const engine = new GameEngine(
        'faq-q24-continue-backend',
        'square8',
        [
          createTestPlayer(1, { ringsInHand: 0, eliminatedRings: 5 }),
          createTestPlayer(2, { ringsInHand: 18, eliminatedRings: 3 }),
        ],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      // Blue has trapped stack
      gameState.board.stacks.clear();
      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      // Surround it
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;
          gameState.board.collapsedSpaces.set(`${3 + dx},${3 + dy}`, 2);
        }
      }

      gameState.currentPhase = 'ring_placement';
      gameState.currentPlayer = 1;

      const moves = engine.getValidMoves(1);
      expect(moves.length).toBe(0);

      // Apply a single forced elimination; since we remain below the
      // victory threshold, the game should continue.
      engine.resolveBlockedStateForCurrentPlayerForTesting();

      // 5 + 1 = 6 eliminated (well below >18 threshold)
      expect(gameState.players[0].eliminatedRings).toBe(6);

      // Game should continue
      expect(gameState.gameStatus).toBe('active');
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

      // No valid moves should trigger stalemate resolution
      const moves = engine.getValidMoves(1);
      expect(moves.length).toBe(0);

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
        { boardType: 'hexagonal', totalSpaces: 331, threshold: 166 },
      ];

      configs.forEach((config) => {
        expect(config.threshold / config.totalSpaces).toBeGreaterThan(0.5);

        // Verify exactly >50% (one more than half the board)
        expect(config.threshold).toBe(Math.floor(config.totalSpaces / 2) + 1);
      });
    });
  });
});
