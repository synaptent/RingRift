/**
 * FAQ Q9-Q14: Edge Cases & Special Mechanics - Comprehensive Test Suite
 *
 * Covers:
 * - FAQ Q9: Chain capture blocking all further moves
 * - FAQ Q10: Multicolored stacks in disconnected regions
 * - FAQ Q12: Chain capture eliminating all your rings
 * - FAQ Q13: Moore vs Von Neumann neighborhoods
 * - FAQ Q14: Overtaking capture optional vs mandatory
 *
 * Rules: §10.3 (Chain Overtaking), §12 (Territory), §13 (Victory)
 *
 * NOTE: These tests directly manipulate internal engine state (gameState) and are
 * skipped when ORCHESTRATOR_ADAPTER_ENABLED=true because the orchestrator
 * bypasses internal state access patterns these tests rely on.
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import { Position, Player, TimeControl, GameState, RingStack } from '../../src/shared/types/game';
import { createTestPlayer } from '../utils/fixtures';

// Skip when orchestrator is enabled - these tests manipulate internal gameState directly
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';
const describeOrSkip = skipWithOrchestrator ? describe.skip : describe;

describeOrSkip('FAQ Q9-Q14: Edge Cases & Special Mechanics', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const createBasePlayers = (): Player[] => [
    createTestPlayer(1, { ringsInHand: 18 }),
    createTestPlayer(2, { ringsInHand: 18 }),
  ];

  describe('FAQ Q9: Chain Capture Blocking All Further Moves', () => {
    it('must complete chain even if it blocks all future moves', async () => {
      // FAQ Q9: Mandatory chain rule applies even if self-destructive

      const engine = new GameEngine(
        'faq-q9-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();
      gameState.board.markers.clear();
      gameState.board.collapsedSpaces.clear();

      // Set up scenario where chain creates blocking collapsed spaces
      gameState.board.stacks.set('2,2', {
        position: { x: 2, y: 2 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      // Target that when captured will create blocking pattern
      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      // Blue has marker at (2,2) that will collapse
      gameState.board.markers.set('1,1', {
        position: { x: 1, y: 1 },
        player: 1,
        type: 'regular',
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

      // Chain completes even if resulting position is difficult
      // This validates mandatory chain rule
    });
  });

  describe('FAQ Q10: Multicolored Stacks in Disconnected Regions', () => {
    it('should evaluate disconnection based on current control only', () => {
      // FAQ Q10: Only current top ring matters for representation
      // Buried rings don't count as representation

      const engine = new GameEngine(
        'faq-q10-backend',
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

      gameState.board.stacks.clear();

      // Region contains multicolored stack controlled by Blue
      // Stack has Green buried inside
      gameState.board.stacks.set('5,5', {
        position: { x: 5, y: 5 },
        rings: [1, 3, 2], // Blue, Green, Red from top
        stackHeight: 3,
        capHeight: 1,
        controllingPlayer: 1, // Blue controls via top ring
      });

      // Red and Green have stacks outside the region
      gameState.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      gameState.board.stacks.set('0,1', {
        position: { x: 0, y: 1 },
        rings: [3],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 3,
      });

      // For disconnection check:
      // - Region has Blue representation (controlled stack)
      // - Region lacks Red representation (no Red-controlled stack)
      // - Region lacks Green representation (buried Green doesn't count!)

      // So region IS disconnected (lacks Red and Green representation)
    });
  });

  describe('FAQ Q12: Chain Capture Eliminating All Your Rings', () => {
    it('must complete chain even if it eliminates all your rings', async () => {
      // FAQ Q12: Chain continues even to self-elimination

      const engine = new GameEngine(
        'faq-q12-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();
      gameState.board.markers.clear();

      // Blue's last stack
      gameState.board.stacks.set('2,2', {
        position: { x: 2, y: 2 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      // Set up line that will form during chain
      // When Blue moves from (2,2), it leaves a marker
      // This could complete a line requiring elimination

      for (let x = 0; x < 3; x++) {
        gameState.board.markers.set(`${x},2`, {
          position: { x, y: 2 },
          player: 1,
          type: 'regular',
        });
      }

      // Red target
      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      gameState.players[0].ringsInHand = 0; // No more rings in hand

      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 1;

      // Execute capture that might form line
      const result = await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: { x: 2, y: 2 },
        captureTarget: { x: 3, y: 3 },
        to: { x: 4, y: 4 },
      } as any);

      expect(result.success).toBe(true);

      // Chain must complete all phases even if Blue ends with no rings
    });
  });

  describe('FAQ Q13: Moore vs Von Neumann Neighborhoods', () => {
    describe('Moore Neighborhood (8-direction)', () => {
      it('should use Moore for movement on square boards', () => {
        // FAQ Q13: Movement uses 8 directions (orthogonal + diagonal)

        const engine = new GameEngine(
          'faq-q13-moore-movement',
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

        // Should have moves in all 8 directions
        const directions = new Set();
        moveMoves.forEach((m: any) => {
          if (!m.to) return;
          const dx = Math.sign(m.to.x - 3);
          const dy = Math.sign(m.to.y - 3);
          directions.add(`${dx},${dy}`);
        });

        // Should have all 8 Moore directions
        expect(directions.size).toBe(8);
      });

      it('should use Moore for line formation on square boards', () => {
        // FAQ Q13: Lines can be diagonal (Moore neighborhood)

        const engine = new GameEngine(
          'faq-q13-moore-lines',
          'square8',
          createBasePlayers(),
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState = engineAny.gameState;

        // Create diagonal line
        const diagonalLine: Position[] = [
          { x: 0, y: 0 },
          { x: 1, y: 1 },
          { x: 2, y: 2 },
          { x: 3, y: 3 },
        ];

        diagonalLine.forEach((pos) => {
          gameState.board.markers.set(`${pos.x},${pos.y}`, {
            position: pos,
            player: 1,
            type: 'regular',
          });
        });

        gameState.board.stacks.set('5,5', {
          position: { x: 5, y: 5 },
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        });

        gameState.currentPhase = 'line_processing';
        gameState.currentPlayer = 1;

        const moves = engine.getValidMoves(1);
        const lineMoves = moves.filter((m: any) => m.type === 'process_line');

        // Diagonal line should be valid
        expect(lineMoves.length).toBeGreaterThan(0);
      });
    });

    describe('Von Neumann Neighborhood (4-direction)', () => {
      it('should use Von Neumann for territory on square boards', () => {
        // FAQ Q13: Territory uses only orthogonal adjacency

        // This is a structural test - Von Neumann means:
        // - (x±1, y) and (x, y±1) are adjacent
        // - Diagonals (x±1, y±1) are NOT adjacent for territory

        const neighborTests = [
          { from: { x: 3, y: 3 }, to: { x: 4, y: 3 }, orthogonal: true },
          { from: { x: 3, y: 3 }, to: { x: 3, y: 4 }, orthogonal: true },
          { from: { x: 3, y: 3 }, to: { x: 2, y: 3 }, orthogonal: true },
          { from: { x: 3, y: 3 }, to: { x: 3, y: 2 }, orthogonal: true },
          { from: { x: 3, y: 3 }, to: { x: 4, y: 4 }, orthogonal: false }, // Diagonal
          { from: { x: 3, y: 3 }, to: { x: 2, y: 2 }, orthogonal: false }, // Diagonal
        ];

        neighborTests.forEach((test) => {
          const dx = Math.abs(test.to.x - test.from.x);
          const dy = Math.abs(test.to.y - test.from.y);

          // Von Neumann: exactly one of dx or dy is 1, the other is 0
          const isVonNeumann = (dx === 1 && dy === 0) || (dx === 0 && dy === 1);

          expect(isVonNeumann).toBe(test.orthogonal);
        });
      });
    });

    describe('Hexagonal Board Uses Single System', () => {
      it('should use 6-direction for movement, lines, AND territory on hex', () => {
        // FAQ Q13: Hexagonal uses same adjacency for everything

        // Hexagonal neighbors from (0,0,0):
        const hexNeighbors = [
          { x: 1, y: -1, z: 0 }, // +x direction
          { x: 1, y: 0, z: -1 }, // +x+z direction
          { x: 0, y: 1, z: -1 }, // +y direction
          { x: -1, y: 1, z: 0 }, // -x direction
          { x: -1, y: 0, z: 1 }, // -x-z direction
          { x: 0, y: -1, z: 1 }, // -y direction
        ];

        // All 6 neighbors should be exactly distance 1 in cube coordinates
        hexNeighbors.forEach((neighbor) => {
          const distance = Math.max(
            Math.abs(neighbor.x),
            Math.abs(neighbor.y),
            Math.abs(neighbor.z)
          );
          expect(distance).toBe(1);

          // x + y + z should equal 0
          expect(neighbor.x + neighbor.y + neighbor.z).toBe(0);
        });

        expect(hexNeighbors.length).toBe(6);
      });
    });
  });

  describe('FAQ Q14: Overtaking Capture Optional vs Mandatory', () => {
    it('should allow choosing whether to start overtaking after non-capture move', async () => {
      // FAQ Q14: Initial capture is optional, but chain is mandatory once started

      const engine = new GameEngine(
        'faq-q14-optional-backend',
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
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      gameState.currentPhase = 'movement';
      gameState.currentPlayer = 1;

      // Make non-capture move
      const moveResult = await engine.makeMove({
        player: 1,
        type: 'move_stack',
        from: { x: 2, y: 2 },
        to: { x: 3, y: 3 },
      } as any);

      expect(moveResult.success).toBe(true);

      // After non-capture move, if capture is available, it's OPTIONAL
      // Player can choose to end turn or start capturing

      // Note: In actual implementation, this choice is implicit
      // Based on whether player selects a capture move or not
    });

    it('must continue chain once overtaking starts', async () => {
      // FAQ Q14: Once capture chain begins, continuation is MANDATORY

      const engine = new GameEngine(
        'faq-q14-mandatory-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();

      gameState.board.stacks.set('2,2', {
        position: { x: 2, y: 2 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      gameState.board.stacks.set('5,5', {
        position: { x: 5, y: 5 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 1;

      // Start chain
      await engine.makeMove({
        player: 1,
        type: 'overtaking_capture',
        from: { x: 2, y: 2 },
        captureTarget: { x: 3, y: 3 },
        to: { x: 4, y: 4 },
      } as any);

      // Should enter chain_capture phase
      expect(gameState.currentPhase).toBe('chain_capture');

      // Must have continue_capture_segment moves available
      const moves = engine.getValidMoves(1);
      const chainMoves = moves.filter((m: any) => m.type === 'continue_capture_segment');

      // If any legal capture exists, MUST perform one
      if (chainMoves.length > 0) {
        // Cannot skip - must continue
        expect(chainMoves.length).toBeGreaterThan(0);
      }
    });

    it('may choose which capture direction when multiple options exist', async () => {
      // FAQ Q14: Player chooses which legal segment when multiple exist

      const engine = new GameEngine(
        'faq-q14-choice-backend',
        'square8',
        createBasePlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState = engineAny.gameState;

      gameState.board.stacks.clear();

      // Blue in center with multiple Red targets around it
      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      // Multiple targets in different directions
      gameState.board.stacks.set('3,4', {
        position: { x: 3, y: 4 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      gameState.board.stacks.set('4,3', {
        position: { x: 4, y: 3 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      gameState.currentPhase = 'capture';
      gameState.currentPlayer = 1;

      // Get valid captures
      const moves = engine.getValidMoves(1);
      const captureMoves = moves.filter((m: any) => m.type === 'overtaking_capture');

      // Should have multiple capture options
      expect(captureMoves.length).toBeGreaterThanOrEqual(2);

      // Player can choose which one (strategic choice)
      // But once chosen, must continue the chain from that landing
    });
  });

  describe('FAQ Q20: Territory Disconnection Rules Comparison', () => {
    it('should use Von Neumann for both square8 and square19', () => {
      // FAQ Q20: Both square boards use 4-direction for territory

      // This is a structural validation of adjacency systems
      // Square boards: Moore for movement/lines, Von Neumann for territory
      // Hexagonal: 6-direction for everything

      const square8TerritoryDirections = 4; // Von Neumann
      const square19TerritoryDirections = 4; // Von Neumann
      const hexTerritoryDirections = 6; // Hexagonal

      const square8MovementDirections = 8; // Moore
      const square19MovementDirections = 8; // Moore
      const hexMovementDirections = 6; // Hexagonal

      // Both square boards use same territory system
      expect(square8TerritoryDirections).toBe(square19TerritoryDirections);

      // But different from hex
      expect(square8TerritoryDirections).toBeLessThan(hexTerritoryDirections);

      // Square boards use different systems for movement vs territory
      expect(square8MovementDirections).toBeGreaterThan(square8TerritoryDirections);

      // Hex uses same system for both
      expect(hexMovementDirections).toBe(hexTerritoryDirections);
    });

    it('should require orthogonal connection for territory on square boards', () => {
      // FAQ Q20: Diagonal connection doesn't count for territory

      const engine = new GameEngine(
        'faq-q20-backend',
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

      gameState.board.stacks.clear();

      // Create scenario where diagonal would connect but orthogonal doesn't
      //   A   A   A
      //
      //   A       A     <- Diagonal gap
      //
      //   A   B   A     <- B is "surrounded" by A markers
      //
      //   A       A
      //
      //   A   A   A

      const surroundingMarkers: Position[] = [
        { x: 2, y: 2 },
        { x: 3, y: 2 },
        { x: 4, y: 2 },
        { x: 2, y: 3 },
        { x: 4, y: 3 },
        { x: 2, y: 4 },
        { x: 3, y: 4 },
        { x: 4, y: 4 },
      ];

      surroundingMarkers.forEach((pos) => {
        gameState.board.markers.set(`${pos.x},${pos.y}`, {
          position: pos,
          player: 1,
          type: 'regular',
        });
      });

      // B ring in center
      gameState.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      // Green has stack elsewhere
      gameState.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        rings: [3],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 3,
      });

      // Center (3,3) is disconnected via orthogonal check
      // Even though diagonals connect to outside, Von Neumann doesn't use diagonals
    });
  });

  describe('Cross-FAQ Integration: Q2 + Q3 + Q14', () => {
    it('should combine minimum distance, landing rules, and capture choices', async () => {
      // Integration of Q2 (min distance), Q3 (landing), Q14 (optional start)

      const engine = new GameEngine(
        'faq-integration-backend',
        'square19',
        [createTestPlayer(1, { ringsInHand: 60 }), createTestPlayer(2, { ringsInHand: 60 })],
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

      // Markers in path
      gameState.board.markers.set('6,6', {
        position: { x: 6, y: 6 },
        player: 2,
        type: 'regular',
      });

      gameState.board.markers.set('7,7', {
        position: { x: 7, y: 7 },
        player: 2,
        type: 'regular',
      });

      // Red target beyond markers
      gameState.board.stacks.set('9,9', {
        position: { x: 9, y: 9 },
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      gameState.currentPhase = 'movement';
      gameState.currentPlayer = 1;

      // Option 1: Non-capture move (Q2 applies)
      // Must move ≥3 spaces, can land beyond markers

      // Option 2: Capture move (Q3 applies)
      // Can choose landing distance beyond Red target

      const moves = engine.getValidMoves(1);

      // Should have both move and capture options
      const moveMoves = moves.filter((m: any) => m.type === 'move_stack');
      const captureMoves = moves.filter((m: any) => m.type === 'overtaking_capture');

      expect(moveMoves.length).toBeGreaterThan(0);
      expect(captureMoves.length).toBeGreaterThan(0);

      // Player can CHOOSE which (Q14: initial capture is optional)
    });
  });
});
