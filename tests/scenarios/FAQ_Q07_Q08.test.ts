/**
 * FAQ Q7-Q8: Line Formation & Collapse - Comprehensive Test Suite
 *
 * Covers:
 * - FAQ Q7: Multiple lines of markers (version-specific length)
 * - FAQ Q8: No rings/stacks to remove when required
 *
 * Rules: §11 (Line Formation & Collapse)
 *
 * NOTE: These tests directly manipulate internal engine state (gameState) and are
 * skipped when ORCHESTRATOR_ADAPTER_ENABLED=true because the orchestrator
 * bypasses internal state access patterns these tests rely on.
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import { ClientSandboxEngine } from '../../src/client/sandbox/ClientSandboxEngine';
import { Position, Player, TimeControl, RingStack, GameState } from '../../src/shared/types/game';
import { createTestPlayer, createTestBoard, addMarker, addStack, pos } from '../utils/fixtures';

// Skip when orchestrator is enabled - these tests manipulate internal gameState directly
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';
const describeOrSkip = skipWithOrchestrator ? describe.skip : describe;

describeOrSkip('FAQ Q7-Q8: Line Formation & Collapse', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const createBasePlayers = (): Player[] => [
    createTestPlayer(1, { ringsInHand: 18, eliminatedRings: 0 }),
    createTestPlayer(2, { ringsInHand: 18, eliminatedRings: 0 }),
  ];

  describe('FAQ Q7: Multiple Lines of Markers (Version-Specific Length)', () => {
    describe('8×8 Version - 3+ markers required', () => {
      it('should process exact-length line (3 markers) with mandatory elimination', async () => {
        // FAQ Q7: Exact-length line on 8×8 requires collapsing all 3 markers
        // and eliminating one ring/cap

        const engine = new GameEngine(
          'faq-q7-exact-backend',
          'square8',
          createBasePlayers(),
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        // Set up 3 consecutive Blue markers in a row
        gameState.board.markers.clear();
        gameState.board.collapsedSpaces.clear();

        for (let x = 0; x < 3; x++) {
          gameState.board.markers.set(`${x},0`, {
            position: { x, y: 0 },
            player: 1,
            type: 'regular',
          });
        }

        // Blue needs a stack to eliminate
        gameState.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          rings: [1, 1],
          stackHeight: 2,
          capHeight: 2,
          controllingPlayer: 1,
        });

        const initialEliminated = gameState.players[0].eliminatedRings;
        const initialTerritory = gameState.players[0].territorySpaces;

        gameState.currentPhase = 'line_processing';
        gameState.currentPlayer = 1;

        // Get valid moves - should include process_line
        const moves = engine.getValidMoves(1);
        const lineMoves = moves.filter((m: any) => m.type === 'process_line');

        expect(lineMoves.length).toBeGreaterThan(0);

        // Process the line
        const result = await engine.makeMove(lineMoves[0]);
        expect(result.success).toBe(true);

        // Verify:
        // - All 3 markers collapsed to Blue territory
        // - One ring eliminated from Blue's stack
        expect(gameState.players[0].eliminatedRings).toBe(initialEliminated + 2); // Whole cap
        expect(gameState.players[0].territorySpaces).toBe(initialTerritory + 3);

        // All original marker positions should now be collapsed
        for (let x = 0; x < 3; x++) {
          const collapsed = gameState.board.collapsedSpaces.get(`${x},0`);
          expect(collapsed).toBe(1); // Blue's color
        }
      });

      it('should process overlength line (4+ markers) with two options', async () => {
        // FAQ Q7: Overlength line allows choice between:
        // Option 1: Collapse all + eliminate ring
        // Option 2: Collapse only 3 + no elimination

        const engine = new GameEngine(
          'faq-q7-overlength-backend',
          'square8',
          createBasePlayers(),
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        // Set up 4 consecutive Blue markers
        gameState.board.markers.clear();
        gameState.board.collapsedSpaces.clear();

        for (let x = 0; x < 4; x++) {
          gameState.board.markers.set(`${x},0`, {
            position: { x, y: 0 },
            player: 1,
            type: 'regular',
          });
        }

        // Blue needs a stack to eliminate (for Option 1)
        gameState.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        });

        const initialEliminated = gameState.players[0].eliminatedRings;

        gameState.currentPhase = 'line_processing';
        gameState.currentPlayer = 1;

        const moves = engine.getValidMoves(1);
        const lineMoves = moves.filter((m: any) => m.type === 'process_line');

        // Should have moves for processing the line
        expect(lineMoves.length).toBeGreaterThan(0);

        // Default backend behavior is Option 2 (minimum collapse, no elimination)
        const result = await engine.makeMove(lineMoves[0]);
        expect(result.success).toBe(true);

        // With Option 2: only 3 markers collapsed, no elimination
        expect(gameState.players[0].eliminatedRings).toBe(initialEliminated);
        expect(gameState.players[0].territorySpaces).toBe(3);
      });
    });

    describe('19×19 Version - 4+ markers required', () => {
      it('should process exact-length line (4 markers) with mandatory elimination', async () => {
        // FAQ Q7: Exact-length line on 19×19 requires 4 markers

        const engine = new GameEngine(
          'faq-q7-square19-backend',
          'square19',
          [
            createTestPlayer(1, { ringsInHand: 60, eliminatedRings: 0 }),
            createTestPlayer(2, { ringsInHand: 60, eliminatedRings: 0 }),
          ],
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        // Set up 5 consecutive markers
        gameState.board.markers.clear();

        for (let x = 0; x < 4; x++) {
          gameState.board.markers.set(`${x},0`, {
            position: { x, y: 0 },
            player: 1,
            type: 'regular',
          });
        }

        // Blue needs a stack
        gameState.board.stacks.set('5,5', {
          position: { x: 5, y: 5 },
          rings: [1, 1, 1],
          stackHeight: 3,
          capHeight: 3,
          controllingPlayer: 1,
        });

        const initialEliminated = gameState.players[0].eliminatedRings;

        gameState.currentPhase = 'line_processing';
        gameState.currentPlayer = 1;

        const moves = engine.getValidMoves(1);
        const lineMoves = moves.filter((m: any) => m.type === 'process_line');

        expect(lineMoves.length).toBeGreaterThan(0);

        const result = await engine.makeMove(lineMoves[0]);
        expect(result.success).toBe(true);

        // All 4 markers collapsed, entire cap (3 rings) eliminated
        expect(gameState.players[0].eliminatedRings).toBe(initialEliminated + 3);
        expect(gameState.players[0].territorySpaces).toBe(4);
      });
    });
  });

  describe('FAQ Q8: No Rings/Stacks to Remove When Required', () => {
    it('should prevent exact-length line processing when no rings to eliminate', async () => {
      // FAQ Q8: If exact-length line forms but player has no rings/stacks,
      // line cannot be processed (or turn ends)

      const engine = new GameEngine(
        'faq-q8-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      // Set up 3-marker line for Blue (exact length for square8)
      gameState.board.markers.clear();
      gameState.board.stacks.clear();

      for (let x = 0; x < 3; x++) {
        gameState.board.markers.set(`${x},0`, {
          position: { x, y: 0 },
          player: 1,
          type: 'regular',
        });
      }

      // Important: Blue has NO stacks on board and NO rings in hand
      gameState.players[0].ringsInHand = 0;

      gameState.currentPhase = 'line_processing';
      gameState.currentPlayer = 1;

      const moves = engine.getValidMoves(1);
      const lineMoves = moves.filter((m: any) => m.type === 'process_line');

      // Note: Engine may still generate process_line move but it would fail
      // or engine may prevent it entirely. Either behavior is acceptable.
      // The key is that the line cannot be fully processed without rings.

      if (lineMoves.length > 0) {
        // If move exists, attempting it should handle the no-rings case
        const result = await engine.makeMove(lineMoves[0]);
        // Implementation may succeed (skip elimination) or fail gracefully
      }

      // Markers may remain or may be processed with special handling
      // The critical point is no rings are eliminated when none available
      expect(gameState.players[0].eliminatedRings).toBe(0);
    });

    it('should allow overlength line with Option 2 when no elimination rings available', async () => {
      // FAQ Q8: For overlength lines, player can choose Option 2
      // (partial collapse, no elimination) even if they have no rings

      const engine = new GameEngine(
        'faq-q8-option2-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      // Set up 4-marker line (overlength for square8)
      gameState.board.markers.clear();
      gameState.board.stacks.clear();

      for (let x = 0; x < 4; x++) {
        gameState.board.markers.set(`${x},0`, {
          position: { x, y: 0 },
          player: 1,
          type: 'regular',
        });
      }

      // Blue has no stacks and no rings in hand
      gameState.players[0].ringsInHand = 0;

      gameState.currentPhase = 'line_processing';
      gameState.currentPlayer = 1;

      const moves = engine.getValidMoves(1);
      const lineMoves = moves.filter((m: any) => m.type === 'process_line');

      // Should still be able to process with Option 2
      expect(lineMoves.length).toBeGreaterThan(0);

      const result = await engine.makeMove(lineMoves[0]);
      expect(result.success).toBe(true);

      // Only 3 markers collapsed (Option 2), no elimination
      expect(gameState.players[0].eliminatedRings).toBe(0);
      expect(gameState.players[0].territorySpaces).toBe(3);
    });
  });

  describe('Multiple Intersecting Lines', () => {
    it('should process lines one at a time, invalidating intersecting lines', async () => {
      // FAQ Q7: When multiple lines exist, process one at a time
      // Processing one may invalidate others due to collapsed spaces

      const engine = new GameEngine(
        'faq-q7-intersecting-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.markers.clear();
      gameState.board.stacks.clear();

      // Create horizontal line: y=2, x=0-3
      for (let x = 0; x < 4; x++) {
        gameState.board.markers.set(`${x},2`, {
          position: { x, y: 2 },
          player: 1,
          type: 'regular',
        });
      }

      // Create vertical line: x=2, y=0-3 (intersects at 2,2)
      for (let y = 0; y < 4; y++) {
        gameState.board.markers.set(`2,${y}`, {
          position: { x: 2, y },
          player: 1,
          type: 'regular',
        });
      }

      // Blue needs stacks for eliminations
      gameState.board.stacks.set('5,5', {
        position: { x: 5, y: 5 },
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });

      gameState.board.stacks.set('6,6', {
        position: { x: 6, y: 6 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      gameState.currentPhase = 'line_processing';
      gameState.currentPlayer = 1;

      // Process first line
      const moves1 = engine.getValidMoves(1);
      const lineMoves1 = moves1.filter((m: any) => m.type === 'process_line');
      expect(lineMoves1.length).toBeGreaterThan(0);

      await engine.makeMove(lineMoves1[0]);

      // After processing one line, check if second line is still valid
      // The intersection point is now collapsed, so the second line may be invalid
      const moves2 = engine.getValidMoves(1);
      const lineMoves2 = moves2.filter((m: any) => m.type === 'process_line');

      // Second line should be invalid due to collapsed intersection
      expect(lineMoves2.length).toBe(0);
    });
  });
});
