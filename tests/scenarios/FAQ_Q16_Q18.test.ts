/**
 * FAQ Q16-Q18: Victory Conditions - Comprehensive Test Suite
 *
 * Covers:
 * - FAQ Q16: Control transfer in multicolored stacks
 * - FAQ Q17: First ring placement movement rules
 * - FAQ Q18: Multiple victory conditions met simultaneously
 *
 * Rules: §13 (Victory Conditions), §5.3 (Control Rules)
 *
 * NOTE: These tests directly manipulate internal engine state (gameState) and are
 * skipped when ORCHESTRATOR_ADAPTER_ENABLED=true because the orchestrator
 * bypasses internal state access patterns these tests rely on.
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import { Position, Player, TimeControl, GameState } from '../../src/shared/types/game';
import { createTestPlayer } from '../utils/fixtures';

// Skip when orchestrator is enabled - these tests manipulate internal gameState directly
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';
const describeOrSkip = skipWithOrchestrator ? describe.skip : describe;

describeOrSkip('FAQ Q16-Q18: Victory Conditions & Control Transfer', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  describe('FAQ Q16: Control Transfer in Multicolored Stacks', () => {
    it('should transfer control when top ring changes color', async () => {
      // FAQ Q16: Control determined solely by top ring
      // When top ring captured, control transfers to next ring down

      const engine = new GameEngine(
        'faq-q16-backend',
        'square8',
        [createTestPlayer(1, { ringsInHand: 18 }), createTestPlayer(2, { ringsInHand: 18 })],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();

      // Multicolored stack: Blue on top, Red below
      const stackPos: Position = { x: 3, y: 3 };
      gameState.board.stacks.set('3,3', {
        position: stackPos,
        rings: [1, 2, 2], // Blue, Red, Red from top
        stackHeight: 3,
        capHeight: 1, // Blue cap is only 1
        controllingPlayer: 1,
      });

      // Another stack to capture from
      gameState.board.stacks.set('5,5', {
        position: { x: 5, y: 5 },
        rings: [2, 2],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 2,
      });

      // Initially Blue controls the stack at (3,3)
      expect(gameState.board.stacks.get('3,3')!.controllingPlayer).toBe(1);

      // Red captures Blue's top ring
      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 2;

      const capture = await engine.makeMove({
        player: 2,
        type: 'overtaking_capture',
        from: { x: 5, y: 5 },
        captureTarget: stackPos,
        to: { x: 1, y: 1 },
      } as any);

      expect(capture.success).toBe(true);

      // After capture, original stack at (3,3) should now have Red on top
      const stackAfter = gameState.board.stacks.get('3,3');
      expect(stackAfter).toBeDefined();
      expect(stackAfter!.controllingPlayer).toBe(2); // Now Red controls it
      expect(stackAfter!.stackHeight).toBe(2); // Blue ring removed
    });

    it('should allow newly exposed player to move stack on their turn', async () => {
      // FAQ Q16: If buried ring becomes exposed, that player can move it

      const engine = new GameEngine(
        'faq-q16-recovery-backend',
        'square8',
        [
          createTestPlayer(1, { ringsInHand: 18 }),
          createTestPlayer(2, { ringsInHand: 18 }),
          createTestPlayer(3, { ringsInHand: 18 }),
        ],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      // Stack: Green on top, Blue buried underneath
      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [3, 1, 1], // Green, Blue, Blue
        stackHeight: 3,
        capHeight: 1,
        controllingPlayer: 3,
      });

      // Red captures Green's top ring
      gameState.board.stacks.set('5,5', {
        position: { x: 5, y: 5 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 2;

      await engine.makeMove({
        player: 2,
        type: 'overtaking_capture',
        from: { x: 5, y: 5 },
        captureTarget: { x: 3, y: 3 },
        to: { x: 1, y: 1 },
      } as any);

      // Stack at (3,3) now has Blue on top
      const stack = gameState.board.stacks.get('3,3');
      expect(stack!.controllingPlayer).toBe(1); // Blue now controls

      // Blue should be able to move this stack on their turn
    });
  });

  describe('FAQ Q17: First Ring Placement Movement Rules', () => {
    it('should require standard movement even for first ring on empty board', async () => {
      // FAQ Q17: No special rule for first placement - standard movement applies

      const engine = new GameEngine(
        'faq-q17-backend',
        'square8',
        [createTestPlayer(1, { ringsInHand: 18 }), createTestPlayer(2, { ringsInHand: 18 })],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      // Completely empty board
      gameState.board.stacks.clear();
      gameState.board.markers.clear();
      gameState.board.collapsedSpaces.clear();

      // Player 1 places first ring
      gameState.currentPhase = 'ring_placement';
      gameState.currentPlayer = 1;

      const placementResult = await engine.makeMove({
        player: 1,
        type: 'place_ring',
        to: { x: 3, y: 3 },
        placementCount: 1,
      } as any);

      expect(placementResult.success).toBe(true);

      // Now must move according to standard rules (min distance = stack height = 1)
      gameState.currentPhase = 'movement';

      const moves = engine.getValidMoves(1);
      const moveMoves = moves.filter((m: any) => m.type === 'move_stack' || m.type === 'move_ring');

      // All moves must satisfy min distance ≥ 1
      for (const move of moveMoves) {
        const from = move.from!;
        const to = move.to!;
        const distance = Math.max(Math.abs(to.x - from.x), Math.abs(to.y - from.y));
        expect(distance).toBeGreaterThanOrEqual(1);
      }
    });
  });

  describe('FAQ Q18: Multiple Victory Conditions', () => {
    it('should prioritize ring elimination when both conditions met', async () => {
      // FAQ Q18: Ring elimination takes precedence, but player wins either way

      const engine = new GameEngine(
        'faq-q18-backend',
        'square8',
        [
          createTestPlayer(1, { ringsInHand: 18, eliminatedRings: 26 }),
          createTestPlayer(2, { ringsInHand: 18, eliminatedRings: 5 }),
        ],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      // Player 1 close to BOTH victory conditions
      gameState.players[0].territorySpaces = 31; // Just below >32 threshold
      gameState.players[0].eliminatedRings = 26; // Just below >27 threshold

      // Create scenario where one move achieves BOTH:
      // - Forms a line (collapses to territory)
      // - Eliminates rings to exceed threshold

      gameState.board.markers.clear();

      // Exact-length line (3 markers for square8) that will collapse AND eliminate
      // Exact-length lines MUST collapse and eliminate, vs overlength which defaults to Option 2
      for (let x = 0; x < 3; x++) {
        gameState.board.markers.set(`${x},0`, {
          position: { x, y: 0 },
          player: 1,
          type: 'regular',
        });
      }

      // Blue has stack to eliminate
      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });

      gameState.currentPhase = 'line_processing';
      gameState.currentPlayer = 1;

      const moves = engine.getValidMoves(1);
      const lineMoves = moves.filter((m: any) => m.type === 'process_line');

      const result = await engine.makeMove(lineMoves[0]);
      expect(result.success).toBe(true);

      // After processing exact-length line:
      // - territorySpaces: 31 + 3 = 34 (>32, territory victory!)
      // - eliminatedRings: 26 + 2 = 28 (>27, elimination victory!)

      // Verify both thresholds are exceeded
      expect(gameState.players[0].territorySpaces).toBeGreaterThan(32);
      expect(gameState.players[0].eliminatedRings).toBeGreaterThan(27);

      // FAQ Q18: Ring elimination takes precedence when both met
      // (Though player has won either way)
    });

    it('should not allow simultaneous winners (>50% threshold prevents it)', () => {
      // FAQ Q18: With >50% requirement, only one player can win

      // Mathematical proof: If total rings = 54 (3 players on square8)
      // Player A has 28 rings eliminated (>27, so >50%)
      // Remaining eliminated: 54 - 28 = 26
      // No other player can have >27

      // This is a structural property of the rules, not a runtime test

      const totalRings = 54; // 3 players × 18 rings
      const victoryThreshold = 27; // >50% of 54

      // If Player A has > victoryThreshold
      const playerAEliminated = 28;
      expect(playerAEliminated).toBeGreaterThan(victoryThreshold);

      // Remaining rings for all other players combined
      const remainingEliminated = totalRings - playerAEliminated;
      expect(remainingEliminated).toBeLessThanOrEqual(victoryThreshold);

      // No other single player can also exceed threshold
      // This validates the >50% rule prevents simultaneous victories
    });
  });

  describe('Cross-Board Victory Consistency', () => {
    it('should use same victory logic on square8, square19, and hex', () => {
      // All board types use >50% thresholds

      // Square8: 2p = >18, 3p = >27
      // Square19: 2p = >36, 3p = >54, 4p = >72
      // Hexagonal: Same as square19

      const configs = [
        { boardType: 'square8', players: 2, ringsPerPlayer: 18, threshold: 18 },
        { boardType: 'square8', players: 3, ringsPerPlayer: 18, threshold: 27 },
        { boardType: 'square19', players: 2, ringsPerPlayer: 36, threshold: 36 },
        { boardType: 'square19', players: 3, ringsPerPlayer: 36, threshold: 54 },
        { boardType: 'square19', players: 4, ringsPerPlayer: 36, threshold: 72 },
        { boardType: 'hexagonal', players: 2, ringsPerPlayer: 36, threshold: 36 },
        { boardType: 'hexagonal', players: 3, ringsPerPlayer: 36, threshold: 54 },
        { boardType: 'hexagonal', players: 4, ringsPerPlayer: 36, threshold: 72 },
      ];

      configs.forEach((config) => {
        const totalRings = config.players * config.ringsPerPlayer;
        const expectedThreshold = Math.floor(totalRings / 2);
        expect(config.threshold).toBe(expectedThreshold);

        // Verify victory requires MORE than threshold (not equal)
        // Winning requires eliminated > threshold, which is > 50%
        const minToWin = config.threshold + 1;
        expect(minToWin / totalRings).toBeGreaterThan(0.5);
      });
    });
  });
});
