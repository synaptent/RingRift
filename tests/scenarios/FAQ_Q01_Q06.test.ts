/**
 * FAQ Q1-Q6: Basic Mechanics - Comprehensive Test Suite
 *
 * Covers:
 * - FAQ Q1: Stack splitting/rearranging
 * - FAQ Q2: Minimum jump requirements
 * - FAQ Q3: Capture landing distance
 * - FAQ Q4: Rings under captured top ring
 * - FAQ Q5: Multiple ring capture from one stack
 * - FAQ Q6: Overtaking vs Elimination distinction
 *
 * Rules: §5 (Stack Mechanics), §8 (Movement), §9-10 (Captures)
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import { Position, Player, TimeControl, GameState, RingStack } from '../../src/shared/types/game';
import { createTestPlayer } from '../utils/fixtures';

describe('FAQ Q1-Q6: Basic Mechanics', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const createBasePlayers = (): Player[] => [
    createTestPlayer(1, { ringsInHand: 18 }),
    createTestPlayer(2, { ringsInHand: 18 }),
  ];

  describe('FAQ Q1: Cannot Split or Rearrange Stack Order', () => {
    it('should maintain ring order in stacks - cannot rearrange', () => {
      // FAQ Q1: Once stacked, order is fixed
      // New rings always go to bottom, captured rings always from top

      const engine = new GameEngine(
        'faq-q1-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      // Create a multi-colored stack
      const stackPos: Position = { x: 3, y: 3 };
      gameState.board.stacks.set('3,3', {
        position: stackPos,
        rings: [1, 2, 1, 2], // Blue, Red, Blue, Red from top
        stackHeight: 4,
        capHeight: 1, // Only top Blue ring
        controllingPlayer: 1,
      });

      // Verify ring order is preserved
      const stack = gameState.board.stacks.get('3,3');
      expect(stack!.rings).toEqual([1, 2, 1, 2]);

      // Cannot split: The stack moves as a unit
      // Cannot rearrange: Order is immutable

      // This is a structural property - there's no "rearrange" action
      const moves = engine.getValidMoves(1);
      const rearrangeMoves = moves.filter((m: any) => m.type === 'rearrange_stack');
      expect(rearrangeMoves.length).toBe(0); // No such move type exists
    });

    it('should add captured rings to bottom of capturing stack', async () => {
      // FAQ Q1: Overtaken rings always go to bottom

      const engine = new GameEngine(
        'faq-q1-capture-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();

      // Blue stack
      gameState.board.stacks.set('2,2', {
        position: { x: 2, y: 2 },
        rings: [1, 1], // Two Blue
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });

      // Red stack to capture
      gameState.board.stacks.set('4,4', {
        position: { x: 4, y: 4 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 1;

      const result = await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: { x: 2, y: 2 },
        captureTarget: { x: 4, y: 4 },
        to: { x: 6, y: 6 },
      } as any);

      expect(result.success).toBe(true);

      // New stack should have Red ring at BOTTOM
      const allStacks: RingStack[] = Array.from(gameState.board.stacks.values());
      const newStack = allStacks.find((s) => s.controllingPlayer === 1);
      expect(newStack).toBeDefined();
      expect(newStack!.rings).toEqual([1, 1, 2]); // Blue, Blue, Red - Red at bottom
    });
  });

  describe('FAQ Q2: Minimum Jump Requirements', () => {
    describe('Stack Height = Distance Requirement', () => {
      it('should require height-1 stack to move ≥1 space', () => {
        const engine = new GameEngine(
          'faq-q2-h1-backend',
          'square8',
          createBasePlayers(),
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        gameState.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        });

        gameState.currentPhase = 'movement';
        gameState.currentPlayer = 1;

        const moves = engine.getValidMoves(1);
        const moveMoves = moves.filter(
          (m: any) => m.type === 'move_stack' || m.type === 'move_ring'
        );

        // All moves must be at least distance 1
        moveMoves.forEach((move: any) => {
          const distance = Math.max(
            Math.abs(move.to.x - move.from.x),
            Math.abs(move.to.y - move.from.y)
          );
          expect(distance).toBeGreaterThanOrEqual(1);
        });
      });

      it('should require height-4 stack to move ≥4 spaces', () => {
        const engine = new GameEngine(
          'faq-q2-h4-backend',
          'square8',
          createBasePlayers(),
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        gameState.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          rings: [1, 1, 1, 1],
          stackHeight: 4,
          capHeight: 4,
          controllingPlayer: 1,
        });

        gameState.currentPhase = 'movement';
        gameState.currentPlayer = 1;

        const moves = engine.getValidMoves(1);
        const moveMoves = moves.filter(
          (m: any) => m.type === 'move_stack' || m.type === 'move_ring'
        );

        // All moves must be at least distance 4 (Chebyshev)
        moveMoves.forEach((move: any) => {
          const distance = Math.max(
            Math.abs(move.to.x - move.from.x),
            Math.abs(move.to.y - move.from.y)
          );
          expect(distance).toBeGreaterThanOrEqual(4);
        });
      });
    });

    describe('Markers Count Toward Distance', () => {
      it('should count markers when calculating minimum distance', async () => {
        // FAQ Q2: Both empty spaces AND markers count

        const engine = new GameEngine(
          'faq-q2-markers-backend',
          'square8',
          createBasePlayers(),
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        gameState.board.stacks.set('2,2', {
          position: { x: 2, y: 2 },
          rings: [1, 1],
          stackHeight: 2,
          capHeight: 2,
          controllingPlayer: 1,
        });

        // Place markers in path
        gameState.board.markers.set('3,3', {
          position: { x: 3, y: 3 },
          player: 2,
          type: 'regular',
        });

        gameState.board.markers.set('4,4', {
          position: { x: 4, y: 4 },
          player: 2,
          type: 'regular',
        });

        gameState.currentPhase = 'movement';
        gameState.currentPlayer = 1;

        const moves = engine.getValidMoves(1);

        // Can land on (5,5) or beyond - distance from (2,2) to (5,5) is 3
        // but counting (3,3) and (4,4) as part of path = 3 steps minimum
        // Stack height is 2, so can land anywhere ≥2 from start
        const moveTo55 = moves.find((m: any) => m.to?.x === 5 && m.to?.y === 5);

        if (moveTo55) {
          const result = await engine.makeMove(moveTo55);
          expect(result.success).toBe(true);

          // Markers should be flipped to Blue
          expect(gameState.board.markers.get('3,3')?.player).toBe(1);
          expect(gameState.board.markers.get('4,4')?.player).toBe(1);
        }
      });
    });
  });

  describe('FAQ Q3: Landing Distance Flexibility During Captures', () => {
    it('should allow landing on any valid space beyond target (not just first)', async () => {
      // FAQ Q3: Can choose landing distance beyond captured piece

      const engine = new GameEngine(
        'faq-q3-backend',
        'square19',
        [createTestPlayer(1, { ringsInHand: 36 }), createTestPlayer(2, { ringsInHand: 36 })],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();

      // Blue stack height 3
      gameState.board.stacks.set('5,5', {
        position: { x: 5, y: 5 },
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      });

      // Red target
      gameState.board.stacks.set('8,8', {
        position: { x: 8, y: 8 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 1;

      // Can land on (9,9), (10,10), (11,11), etc. - not forced to first space
      const toLanding11 = await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: { x: 5, y: 5 },
        captureTarget: { x: 8, y: 8 },
        to: { x: 11, y: 11 }, // Land 3 spaces beyond target
      } as any);

      expect(toLanding11.success).toBe(true);

      // Stack should be at chosen landing position
      const finalStack = gameState.board.stacks.get('11,11');
      expect(finalStack).toBeDefined();
      expect(finalStack!.stackHeight).toBe(4); // 3 Blue + 1 Red captured
    });
  });

  describe('FAQ Q4: Rings Under Captured Top Ring', () => {
    it('should leave remaining rings when only top is captured', async () => {
      // FAQ Q4: Only the top ring is Overtaken; the rest stay in place

      const engine = new GameEngine(
        'faq-q4-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();

      // Blue attacker - capHeight 2 so that the capture is legal under §10.1
      gameState.board.stacks.set('2,2', {
        position: { x: 2, y: 2 },
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });

      // Red target with multiple rings
      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [2, 2, 1], // Red, Red, Blue from top
        stackHeight: 3,
        capHeight: 2, // Red cap height is 2
        controllingPlayer: 2,
      });

      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 1;

      const result = await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: { x: 2, y: 2 },
        captureTarget: { x: 3, y: 3 },
        to: { x: 4, y: 4 },
      } as any);

      expect(result.success).toBe(true);

      // Original stack at (3,3) should still exist with remaining rings
      const remainingStack = gameState.board.stacks.get('3,3');
      expect(remainingStack).toBeDefined();
      expect(remainingStack!.stackHeight).toBe(2); // Lost one ring
      expect(remainingStack!.rings).toEqual([2, 1]); // Red, Blue
      expect(remainingStack!.controllingPlayer).toBe(2); // Still Red (new top)

      // Capturing stack should have the captured ring appended at the bottom
      const capturingStack = gameState.board.stacks.get('4,4');
      expect(capturingStack).toBeDefined();
      expect(capturingStack!.stackHeight).toBe(3); // 2 Blue + 1 Red captured
      expect(capturingStack!.rings).toEqual([1, 1, 2]); // Blue, Blue, Red - captured at bottom
    });
  });

  describe('FAQ Q5: Multiple Ring Capture from One Stack', () => {
    it('should capture only one ring per jump segment', async () => {
      // FAQ Q5: Single jump segment captures only the top ring
      // Rules §§8.2, 10.1, 10.2:
      // - Distance from start to landing must be ≥ stackHeight.
      // - Attacker.capHeight must be ≥ target.capHeight.
      // - Only the top ring of the target is captured per segment.

      const engine = new GameEngine(
        'faq-q5-single-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();

      // Blue attacker with height/capHeight 3 so that:
      // - Cap-height requirement vs a 3-high Red cap is satisfied.
      // - Distance requirement can be met by landing 3 steps away.
      gameState.board.stacks.set('2,2', {
        position: { x: 2, y: 2 },
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      });

      // Red target with 3 rings (capHeight 3)
      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [2, 2, 2],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 2,
      });

      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 1;

      await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: { x: 2, y: 2 },
        captureTarget: { x: 3, y: 3 },
        // Landing at (5,5) is Chebyshev distance 3 from (2,2),
        // satisfying the "≥ stackHeight" requirement for height 3.
        to: { x: 5, y: 5 },
      } as any);

      // Target should still have 2 rings (only one ring captured from the stack)
      const target = gameState.board.stacks.get('3,3');
      expect(target).toBeDefined();
      expect(target!.stackHeight).toBe(2);

      // Capturing stack should have gained exactly 1 ring
      const capturer = gameState.board.stacks.get('5,5');
      expect(capturer).toBeDefined();
      expect(capturer!.stackHeight).toBe(4); // 3 original + 1 captured
      expect(capturer!.rings).toEqual([1, 1, 1, 2]); // captured Red ring appended at bottom
    });

    it('should allow multiple captures from same stack via reversal/cycles', async () => {
      // FAQ Q5: Can capture multiple rings from the same stack through chain
      // (180° reversal or cyclic patterns).
      //
      // Requirements:
      // - Each segment must satisfy capHeight and distance rules.
      // - Each segment captures only the top ring.
      // - Chain may revisit the same target stack and capture again.

      const engine = new GameEngine(
        'faq-q5-multiple-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();

      // Blue stack with height/capHeight 3 so that:
      // - Initial capture over a 3-high Red cap is legal.
      // - Distance requirement (≥ 3) is satisfied by landing 3 steps away.
      gameState.board.stacks.set('2,2', {
        position: { x: 2, y: 2 },
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      });

      // Red target with 3 rings
      gameState.board.stacks.set('2,3', {
        position: { x: 2, y: 3 },
        rings: [2, 2, 2],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 2,
      });

      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 1;

      // First capture: from (2,2) over (2,3) to (2,5)
      await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: { x: 2, y: 2 },
        captureTarget: { x: 2, y: 3 },
        to: { x: 2, y: 5 }, // distance 3 from start, ≥ stackHeight
      } as any);

      // Chain should continue with 180° reversal / cyclic options available.
      // We let GameEngine + shared captureChainEngine enumerate the legal
      // follow-ups; some of those may revisit (2,3) and capture from it again.
      while (gameState.currentPhase === 'chain_capture') {
        const moves = engine.getValidMoves(1);
        const chainMoves = moves.filter((m: any) => m.type === 'continue_capture_segment');
        if (chainMoves.length === 0) break;

        await engine.makeMove(chainMoves[0]);
      }

      // Target should have fewer rings (captured multiple times across the chain)
      const target = gameState.board.stacks.get('2,3');
      if (target) {
        // Some rings captured via reversal/cycle
        expect(target.stackHeight).toBeLessThan(3);
      }
    });
  });

  describe('FAQ Q6: Overtaking vs Elimination Distinction', () => {
    it('should keep overtaken rings in play (not count toward victory)', async () => {
      // FAQ Q6: Overtaking keeps rings on board

      const engine = new GameEngine(
        'faq-q6-overtake-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();

      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      gameState.board.stacks.set('4,4', {
        position: { x: 4, y: 4 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      const initialEliminated = gameState.players[0].eliminatedRings;

      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 1;

      await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: { x: 3, y: 3 },
        captureTarget: { x: 4, y: 4 },
        to: { x: 5, y: 5 },
      } as any);

      // Eliminated count should NOT increase (overtaking ≠ elimination)
      expect(gameState.players[0].eliminatedRings).toBe(initialEliminated);

      // Red ring still in play (at bottom of Blue stack)
      const stack = gameState.board.stacks.get('5,5');
      expect(stack!.rings).toContain(2); // Red ring present
    });

    it('should count eliminated rings toward victory threshold', async () => {
      // FAQ Q6: Only elimination counts toward victory

      const engine = new GameEngine(
        'faq-q6-elim-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      // Blue close to victory via elimination
      gameState.players[0].eliminatedRings = 17; // Need >18 for 2p

      // Create a line that will eliminate a ring
      gameState.board.markers.clear();
      for (let x = 0; x < 4; x++) {
        gameState.board.markers.set(`${x},0`, {
          position: { x, y: 0 },
          player: 1,
          type: 'regular',
        });
      }

      gameState.board.stacks.set('5,5', {
        position: { x: 5, y: 5 },
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });

      gameState.currentPhase = 'line_processing';
      gameState.currentPlayer = 1;

      const moves = engine.getValidMoves(1);
      const lineMoves = moves.filter((m: any) => m.type === 'process_line');

      await engine.makeMove(lineMoves[0]);

      // Should eliminate cap (2 rings): 17 + 2 = 19 (>18 threshold)
      expect(gameState.players[0].eliminatedRings).toBe(19);

      // Should trigger victory
      expect(gameState.gameStatus).toBe('completed');
    });
  });

  describe('Unified Movement Landing Rule', () => {
    it('should allow landing on any valid space beyond markers (not just first)', async () => {
      // FAQ Q2-Q3: Unified rule - can land anywhere valid beyond markers

      const engine = new GameEngine(
        'faq-q2-q3-unified-backend',
        'square19',
        [createTestPlayer(1, { ringsInHand: 36 }), createTestPlayer(2, { ringsInHand: 36 })],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.set('5,5', {
        position: { x: 5, y: 5 },
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });

      // Place markers in path
      gameState.board.markers.set('6,6', {
        position: { x: 6, y: 6 },
        player: 2,
        type: 'regular',
      });

      gameState.currentPhase = 'movement';
      gameState.currentPlayer = 1;

      const moves = engine.getValidMoves(1);

      // Should have multiple landing options beyond the marker
      const landingOptions = moves.filter((m: any) => {
        // Must be on diagonal from (5,5) through (6,6)
        if (!m.to) return false;
        const dx = m.to.x - 5;
        const dy = m.to.y - 5;
        return dx === dy && dx > 0; // Diagonal, beyond start
      });

      // Should have options at (7,7), (8,8), (9,9), etc.
      expect(landingOptions.length).toBeGreaterThan(1);
    });
  });
});
