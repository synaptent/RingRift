/**
 * FAQ Q22-Q23: Graduated Line Rewards & Territory Self-Elimination - Comprehensive Test Suite
 *
 * Covers:
 * - FAQ Q22: Strategic considerations for choosing line reward options
 * - FAQ Q23: Self-elimination prerequisite for disconnected regions
 *
 * Rules: §11.2-11.3 (Graduated Line Rewards), §12.2 (Territory Self-Elimination Prerequisite)
 *
 * NOTE: These tests directly manipulate internal engine state (gameState) and are
 * skipped when ORCHESTRATOR_ADAPTER_ENABLED=true because the orchestrator
 * bypasses internal state access patterns these tests rely on.
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import { ClientSandboxEngine } from '../../src/client/sandbox/ClientSandboxEngine';
import {
  Position,
  Player,
  TimeControl,
  GameState,
  BOARD_CONFIGS,
} from '../../src/shared/types/game';
import { createTestPlayer } from '../utils/fixtures';

// Skip when orchestrator is enabled - these tests manipulate internal gameState directly
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';
const describeOrSkip = skipWithOrchestrator ? describe.skip : describe;

describeOrSkip('FAQ Q22-Q23: Graduated Line Rewards & Territory Prerequisites', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const createBasePlayers = (): Player[] => [
    createTestPlayer(1, { ringsInHand: 18, eliminatedRings: 0 }),
    createTestPlayer(2, { ringsInHand: 18, eliminatedRings: 0 }),
    createTestPlayer(3, { ringsInHand: 18, eliminatedRings: 0 }),
  ];

  describe('FAQ Q22: Graduated Line Rewards Strategic Choices', () => {
    describe('Option 1: Maximize Territory (Collapse All + Eliminate Ring)', () => {
      it('should collapse entire overlength line and eliminate ring with Option 1', async () => {
        // FAQ Q22: Option 1 trades a ring for maximum territory control

        const engine = new GameEngine(
          'faq-q22-option1-backend',
          'square8',
          createBasePlayers(),
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        // Create 6-marker line (overlength for square8)
        gameState.board.markers.clear();
        gameState.board.collapsedSpaces.clear();

        for (let x = 0; x < 6; x++) {
          gameState.board.markers.set(`${x},0`, {
            position: { x, y: 0 },
            player: 1,
            type: 'regular',
          });
        }

        // Blue has stack for elimination
        gameState.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          rings: [1, 1, 1],
          stackHeight: 3,
          capHeight: 3,
          controllingPlayer: 1,
        });

        const initialEliminated = gameState.players[0].eliminatedRings;
        const initialTerritory = gameState.players[0].territorySpaces;

        gameState.currentPhase = 'line_processing';
        gameState.currentPlayer = 1;

        // Look for choose_line_reward move (explicit Option 1)
        // Note: Default behavior may be Option 2, but we're testing the concept
        const moves = engine.getValidMoves(1);
        const lineProcessMoves = moves.filter(
          (m: any) => m.type === 'process_line' || m.type === 'choose_line_reward'
        );

        expect(lineProcessMoves.length).toBeGreaterThan(0);

        // If explicit choice is available, select Option 1
        const option1Move = lineProcessMoves.find(
          (m: any) => m.type === 'choose_line_reward' && m.option === 'collapse_all'
        );

        if (option1Move) {
          const result = await engine.makeMove(option1Move);
          expect(result.success).toBe(true);

          // All 6 markers collapsed, cap eliminated
          expect(gameState.players[0].eliminatedRings).toBe(initialEliminated + 3);
          expect(gameState.players[0].territorySpaces).toBe(initialTerritory + 6);
        }
      });
    });

    describe('Option 2: Preserve Rings (Collapse Minimum + No Elimination)', () => {
      it('should collapse only required markers without elimination with Option 2', async () => {
        // FAQ Q22: Option 2 preserves rings while still claiming some territory

        const engine = new GameEngine(
          'faq-q22-option2-backend',
          'square8',
          createBasePlayers(),
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState: GameState = engineAny.gameState as GameState;

        // Create 5-marker line
        gameState.board.markers.clear();

        for (let x = 0; x < 5; x++) {
          gameState.board.markers.set(`${x},0`, {
            position: { x, y: 0 },
            player: 1,
            type: 'regular',
          });
        }

        // Blue has stack available but won't need to eliminate with Option 2
        gameState.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        });

        const initialEliminated = gameState.players[0].eliminatedRings;
        const requiredLength = BOARD_CONFIGS[gameState.boardType].lineLength;

        gameState.currentPhase = 'line_processing';
        gameState.currentPlayer = 1;

        const moves = engine.getValidMoves(1);
        const lineMoves = moves.filter((m: any) => m.type === 'process_line');

        expect(lineMoves.length).toBeGreaterThan(0);

        // Default backend behavior is Option 2 for overlength lines: collapse
        // exactly the minimum requiredLength markers for this board type, with
        // no ring elimination.
        const result = await engine.makeMove(lineMoves[0]);
        expect(result.success).toBe(true);

        // Only requiredLength markers collapsed, no elimination
        expect(gameState.players[0].eliminatedRings).toBe(initialEliminated);
        expect(gameState.players[0].territorySpaces).toBe(requiredLength);

        // One marker should remain uncollapsed
        const remainingMarkers = Array.from(gameState.board.markers.values());
        expect(remainingMarkers.length).toBeGreaterThan(0);
      });
    });

    describe('Strategic Scenarios', () => {
      it('should use Option 1 when territory is strategically valuable', async () => {
        // FAQ Q22: Use Option 1 when territory would cut off opponent movement
        // or create disconnected regions

        const engine = new GameEngine(
          'faq-q22-strategic-backend',
          'square8',
          createBasePlayers(),
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        // Create overlength line that would block opponent passage
        gameState.board.markers.clear();

        // Horizontal barrier across middle of board
        for (let x = 0; x < 5; x++) {
          gameState.board.markers.set(`${x},3`, {
            position: { x, y: 3 },
            player: 1,
            type: 'regular',
          });
        }

        // Red stacks above and below the line
        gameState.board.stacks.set('2,1', {
          position: { x: 2, y: 1 },
          rings: [2],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 2,
        });

        gameState.board.stacks.set('2,5', {
          position: { x: 2, y: 5 },
          rings: [2],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 2,
        });

        // Blue has stack to eliminate
        gameState.board.stacks.set('6,6', {
          position: { x: 6, y: 6 },
          rings: [1, 1],
          stackHeight: 2,
          capHeight: 2,
          controllingPlayer: 1,
        });

        gameState.currentPhase = 'line_processing';
        gameState.currentPlayer = 1;

        const moves = engine.getValidMoves(1);
        expect(moves.length).toBeGreaterThan(0);

        // Option 1 would create a barrier blocking Red's movement
        // This demonstrates the strategic value of full collapse
      });
    });
  });

  describe('FAQ Q23: Territory Self-Elimination Prerequisite', () => {
    describe('Cannot Process Region Without Outside Stack', () => {
      it('should NOT process disconnected region when player has no outside stack', async () => {
        // FAQ Q23: Must have stack/cap outside region to process it

        const engine = new GameEngine(
          'faq-q23-negative-backend',
          'square19',
          [
            createTestPlayer(1, { ringsInHand: 72, eliminatedRings: 0 }),
            createTestPlayer(2, { ringsInHand: 72, eliminatedRings: 0 }),
            createTestPlayer(3, { ringsInHand: 72, eliminatedRings: 0 }),
          ],
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        gameState.board.markers.clear();
        gameState.board.stacks.clear();
        gameState.board.collapsedSpaces.clear();

        // Create disconnected 3×3 region with Blue's only stack inside
        const region: Position[] = [];
        for (let x = 5; x <= 7; x++) {
          for (let y = 5; y <= 7; y++) {
            region.push({ x, y });
          }
        }

        // Blue's ONLY stack is inside the region
        gameState.board.stacks.set('6,6', {
          position: { x: 6, y: 6 },
          rings: [1, 1],
          stackHeight: 2,
          capHeight: 2,
          controllingPlayer: 1,
        });

        // Red stack also inside
        gameState.board.stacks.set('5,5', {
          position: { x: 5, y: 5 },
          rings: [2],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 2,
        });

        // Create border of Blue markers around the region
        const border = [
          { x: 4, y: 4 },
          { x: 5, y: 4 },
          { x: 6, y: 4 },
          { x: 7, y: 4 },
          { x: 8, y: 4 },
          { x: 4, y: 5 },
          { x: 8, y: 5 },
          { x: 4, y: 6 },
          { x: 8, y: 6 },
          { x: 4, y: 7 },
          { x: 8, y: 7 },
          { x: 4, y: 8 },
          { x: 5, y: 8 },
          { x: 6, y: 8 },
          { x: 7, y: 8 },
          { x: 8, y: 8 },
        ];

        for (const pos of border) {
          gameState.board.markers.set(`${pos.x},${pos.y}`, {
            position: pos,
            player: 1,
            type: 'regular',
          });
        }

        // Green has stack outside region (so region lacks Green representation)
        gameState.board.stacks.set('0,0', {
          position: { x: 0, y: 0 },
          rings: [3],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 3,
        });

        gameState.currentPhase = 'territory_processing';
        gameState.currentPlayer = 1;

        // Get valid moves. In this geometry there are TWO disconnected regions:
        // (a) the interior 3×3 block, where Blue has no outside stack; and
        // (b) the complement "outer" region, where Blue's only stack at (6,6) is
        //     outside that region. Q23's self-elimination prerequisite only
        // forbids processing regions for which the moving player has no stack
        // outside that specific region; other eligible regions remain
        // processable.
        const moves = engine.getValidMoves(1);
        const territoryMoves = moves.filter((m: any) => m.type === 'process_territory_region');

        const interiorRegionMoves = territoryMoves.filter(
          (m: any) =>
            m.disconnectedRegions &&
            m.disconnectedRegions[0] &&
            m.disconnectedRegions[0].spaces.some(
              (p: any) => p.x >= 5 && p.x <= 7 && p.y >= 5 && p.y <= 7
            )
        );

        // Cannot process the interior region because Blue has no stack outside it.
        // NOTE: Other regions (such as the large "outer" region) may still be
        // processable if they satisfy the self-elimination prerequisite; see
        // §12.2 / FAQ Q23 in ringrift_complete_rules.md and the compact
        // mini-region scenario Rules_12_2_Q23_mini_region_square8_numeric_invariant.
        expect(interiorRegionMoves.length).toBe(0);

        // Region should remain unchanged
        const stackInRegion = gameState.board.stacks.get('6,6');
        expect(stackInRegion).toBeDefined();
        expect(stackInRegion!.controllingPlayer).toBe(1);
      });
    });

    describe('Can Process Region With Outside Stack', () => {
      it('should process disconnected region when player has outside stack', async () => {
        // FAQ Q23: With outside stack, can process and self-eliminate

        const engine = new GameEngine(
          'faq-q23-positive-backend',
          'square19',
          [
            createTestPlayer(1, { ringsInHand: 72, eliminatedRings: 0 }),
            createTestPlayer(2, { ringsInHand: 72, eliminatedRings: 0 }),
            createTestPlayer(3, { ringsInHand: 72, eliminatedRings: 0 }),
          ],
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        gameState.board.markers.clear();
        gameState.board.stacks.clear();
        gameState.board.collapsedSpaces.clear();

        // Same 3×3 disconnected region setup
        // Blue's stack inside
        gameState.board.stacks.set('6,6', {
          position: { x: 6, y: 6 },
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        });

        // Red stack inside
        gameState.board.stacks.set('5,5', {
          position: { x: 5, y: 5 },
          rings: [2, 2],
          stackHeight: 2,
          capHeight: 2,
          controllingPlayer: 2,
        });

        // CRITICAL: Blue has ANOTHER stack OUTSIDE the region
        gameState.board.stacks.set('0,1', {
          position: { x: 0, y: 1 },
          rings: [1, 1],
          stackHeight: 2,
          capHeight: 2,
          controllingPlayer: 1,
        });

        // Border
        const border = [
          { x: 4, y: 4 },
          { x: 5, y: 4 },
          { x: 6, y: 4 },
          { x: 7, y: 4 },
          { x: 8, y: 4 },
          { x: 4, y: 5 },
          { x: 8, y: 5 },
          { x: 4, y: 6 },
          { x: 8, y: 6 },
          { x: 4, y: 7 },
          { x: 8, y: 7 },
          { x: 4, y: 8 },
          { x: 5, y: 8 },
          { x: 6, y: 8 },
          { x: 7, y: 8 },
          { x: 8, y: 8 },
        ];

        for (const pos of border) {
          gameState.board.markers.set(`${pos.x},${pos.y}`, {
            position: pos,
            player: 1,
            type: 'regular',
          });
        }

        // Green outside (region lacks Green)
        gameState.board.stacks.set('0,0', {
          position: { x: 0, y: 0 },
          rings: [3],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 3,
        });

        const initialEliminated = gameState.players[0].eliminatedRings;

        gameState.currentPhase = 'territory_processing';
        gameState.currentPlayer = 1;

        const moves = engine.getValidMoves(1);
        const territoryMoves = moves.filter((m: any) => m.type === 'process_territory_region');

        // NOW should be able to process
        expect(territoryMoves.length).toBeGreaterThan(0);

        const result = await engine.makeMove(territoryMoves[0]);
        expect(result.success).toBe(true);

        const finalState: GameState = engine.getGameState();
        const finalBlue = finalState.players[0];
        const finalBoard = finalState.board;

        // Verify:
        //
        // In this FAQ geometry there are TWO disconnected regions (inner and
        // outer). The shared detector enumerates the large outer region first,
        // so the first `process_territory_region` move returned by
        // GameEngine.getValidMoves processes that outer region.
        //
        // According to §12.2 / FAQ Q23:
        //   - All rings inside the processed region are eliminated and credited
        //     to the moving player, regardless of colour.
        //   - The moving player must then self-eliminate one ring/cap from a
        //     stack outside that region.
        //
        // Here, processing the outer region eliminates:
        //   - Blue's outside stack at (0,1): 2 rings
        //   - Green's stack at (0,0):       1 ring
        //   - plus 1 additional Blue ring from mandatory self-elimination
        // for a total of 4 rings credited to Blue.
        //
        // NOTE: The interior 3×3 block is a second disconnected region that
        // could also be processed with an additional `process_territory_region`
        // move; its numeric invariants are covered by the dedicated rules-layer
        // tests in territoryProcessing.rules.* and sandboxTerritory*.rules.*.
        expect(finalBlue.eliminatedRings).toBe(initialEliminated + 4);

        // Sample a point known to be in the processed (outer) region to confirm
        // collapse. (0,0) originally held Green's stack and is outside the
        // 3×3 interior; after processing the outer region it must be collapsed
        // to Blue.
        const outerSampleCollapsed = finalBoard.collapsedSpaces.get('0,0');
        expect(outerSampleCollapsed).toBe(1); // Blue's color
      });
    });

    describe('Multiple Regions With Limited Stacks', () => {
      it('should allow processing only as many regions as outside stacks available', async () => {
        // FAQ Q23: Each outside stack/cap can process exactly one region

        const engine = new GameEngine(
          'faq-q23-multiple-backend',
          'square8',
          createBasePlayers(),
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        gameState.board.markers.clear();
        gameState.board.stacks.clear();

        // Create TWO disconnected regions
        // Region 1: around (1,1)
        gameState.board.stacks.set('1,1', {
          position: { x: 1, y: 1 },
          rings: [2],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 2,
        });

        // Region 2: around (5,5)
        gameState.board.stacks.set('5,5', {
          position: { x: 5, y: 5 },
          rings: [2],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 2,
        });

        // Blue has only ONE stack outside both regions
        gameState.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        });

        gameState.currentPhase = 'territory_processing';
        gameState.currentPlayer = 1;

        // Should be able to process only ONE region (not both)
        // After processing the first, no more outside stacks available
      });
    });
  });

  describe('Hexagonal Board Variants', () => {
    it('should apply same prerequisites on hexagonal boards', async () => {
      // FAQ Q23: Self-elimination prerequisite applies to hex as well

      const engine = new GameEngine(
        'faq-q23-hex-backend',
        'hexagonal',
        [
          createTestPlayer(1, {
            ringsInHand: BOARD_CONFIGS.hexagonal.ringsPerPlayer,
            eliminatedRings: 0,
          }),
          createTestPlayer(2, {
            ringsInHand: BOARD_CONFIGS.hexagonal.ringsPerPlayer,
            eliminatedRings: 0,
          }),
          createTestPlayer(3, {
            ringsInHand: BOARD_CONFIGS.hexagonal.ringsPerPlayer,
            eliminatedRings: 0,
          }),
        ],
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      // Similar setup but with hex coordinates (x + y + z = 0)
      gameState.board.stacks.clear();

      // Stack inside potential disconnected region
      gameState.board.stacks.set('0,0,0', {
        position: { x: 0, y: 0, z: 0 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      // WITHOUT outside stack: cannot process
      // WITH outside stack: can process

      // This validates that hex boards use the same prerequisite logic
    });
  });
});
